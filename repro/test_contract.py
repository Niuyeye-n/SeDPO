"""Scientific correctness checks for the shared comparison boundary (CPU only)."""
import copy
import tempfile
import unittest
from pathlib import Path
from common import digest, file_sha, messages, metrics, paired, validate_rows, validate_selection, write


class ComparisonContract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'selection.json'
        self.pool = [{'id': f'train{i}', 'question': f'Example {i}', 'options': ['a', 'b', 'c', 'd'],
                      'gold': i % 4, 'fact': f'Fact {i}'} for i in range(5)]
        self.queries = [{'id': 'official-test-id', 'question': 'Target', 'options': ['w', 'x', 'y', 'z'], 'fact': 'Target fact'}]
        self.selection = {'method': 'coopes', 'protocol_sha256': 'fixed', 'task_name': 'openbookqa',
                          'pool_sha256': digest(self.pool), 'queries_sha256': digest(self.queries),
                          'selector_manifest': {'test_labels_used': False, 'tuning_splits': ['dev'],
                                                'implementation_files': [{'path': __file__, 'sha256': file_sha(__file__)}]},
                          'rows': [{'query_id': 'official-test-id', 'demo_ids': ['train0', 'train1', 'train2', 'train3']}]}

    def validate(self, selection):
        write(self.path, selection)
        return validate_selection(self.path, protocol_sha='fixed', task='openbookqa', pool=self.pool, queries=self.queries)

    def test_valid_common_selection_and_shared_fact_rendering(self):
        _, chosen = self.validate(self.selection)
        rendered = messages(self.queries[0], chosen['official-test-id'], {r['id']: r for r in self.pool}, 'fixed')
        self.assertEqual([r['role'] for r in rendered], ['system'] + ['user', 'assistant'] * 4 + ['user'])
        self.assertIn('Fact: Target fact', rendered[-1]['content'])
        self.assertTrue(all('Fact:' in rendered[i]['content'] for i in [1, 3, 5, 7]))

    def test_rejects_changed_protocol_candidates_or_questions(self):
        for key in ['protocol_sha256', 'pool_sha256', 'queries_sha256']:
            changed = copy.deepcopy(self.selection)
            changed[key] = 'changed'
            with self.assertRaises(ValueError):
                self.validate(changed)

    def test_rejects_test_label_guided_selection(self):
        for key, value in [('test_labels_used', True), ('tuning_splits', ['test'])]:
            changed = copy.deepcopy(self.selection)
            changed['selector_manifest'][key] = value
            with self.assertRaises(ValueError):
                self.validate(changed)

    def test_rejects_foreign_or_repeated_demonstrations(self):
        for demos in [['train0'] * 4, ['train0', 'train1', 'train2', 'foreign'], ['train0', 'train1']]:
            changed = copy.deepcopy(self.selection)
            changed['rows'][0]['demo_ids'] = demos
            with self.assertRaises(ValueError):
                self.validate(changed)

    def test_rejects_missing_repeated_or_wrong_test_ids(self):
        for rows in [[], self.selection['rows'] * 2, [{'query_id': 'different-test', 'demo_ids': ['train0', 'train1', 'train2', 'train3']}]]:
            changed = copy.deepcopy(self.selection)
            changed['rows'] = rows
            with self.assertRaises(ValueError):
                self.validate(changed)

    def test_gold_must_be_absent_from_selector_test_requests(self):
        query = dict(self.queries[0], gold=0)
        with self.assertRaises(ValueError):
            validate_rows([query], labeled=False, task='openbookqa')

    def test_obqa_fact_is_required(self):
        query = {k: v for k, v in self.queries[0].items() if k != 'fact'}
        with self.assertRaises(ValueError):
            validate_rows([query], labeled=False, task='openbookqa')

    def test_invalid_predictions_are_not_dropped_or_replaced(self):
        with self.assertRaises(ValueError):
            metrics([{'gold': 0, 'pred': -1}])

    def test_pairing_rejects_different_gold_even_on_same_ids(self):
        with self.assertRaises(ValueError):
            paired([{'id': 'x', 'gold': 0, 'pred': 0}], [{'id': 'x', 'gold': 1, 'pred': 1}])

    def test_exact_mcnemar_uses_both_discordant_directions(self):
        left = [{'id': str(i), 'gold': 0, 'pred': 1} for i in range(6)]
        right = [{'id': str(i), 'gold': 0, 'pred': 0} for i in range(6)]
        result = paired(left, right)
        self.assertEqual(result['coopes_minus_sedpo_accuracy_pp'], 100)
        self.assertAlmostEqual(result['mcnemar_exact_two_sided_p'], 0.03125)
        self.assertEqual(paired(right, right)['mcnemar_exact_two_sided_p'], 1)


if __name__ == '__main__':
    unittest.main()
