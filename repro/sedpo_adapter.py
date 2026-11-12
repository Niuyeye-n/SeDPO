"""Data adapters only: leave the original DPR encoder and beam search unchanged."""
from common import LETTERS, load_protocol
from dpr.data.biencoder_data import BiEncoderPassage, remove_double_space
from dpr.data.retriever_data import QASample, QASrc
from dpr.utils.tasks import task_map


def task_metadata(row, task_name):
    result = {'id': row['id'], 'task_name': task_name,
              'choices': {'text': row['options'], 'label': list(LETTERS)}}
    result['question' if task_name == 'arc_c' else 'question_stem'] = row['question']
    if task_name == 'openbookqa':
        result['fact1'] = row['fact']
    if 'gold' in row:
        result['answerKey'] = LETTERS[row['gold']]
    return result


class ControlledContexts:
    def __init__(self, protocol_file, task_name, **kwargs):
        _, self.data, _ = load_protocol(protocol_file, task_name, include_gold=False)
        self.task_name, self.id_prefix = task_name, None

    def load_data_to(self, passages):
        task = task_map.cls_dic[self.task_name]()
        for index, row in enumerate(self.data['pool']):
            meta = task_metadata(row, self.task_name)
            text = task.get_question(meta) + task.get_answer(meta)
            passages[index] = BiEncoderPassage(remove_double_space(text), '', meta)


class ControlledQuestions(QASrc):
    def __init__(self, protocol_file, task_name, selector=None, special_query_token=None,
                 query_special_suffix=None, **kwargs):
        super().__init__('', selector, special_query_token, query_special_suffix)
        _, self.inputs, _ = load_protocol(protocol_file, task_name, include_gold=False)
        self.task_name = task_name

    def load_data(self):
        task = task_map.cls_dic[self.task_name]()
        self.data = []
        for row in self.inputs['queries']:
            meta = task_metadata(row, self.task_name)
            if 'answerKey' in meta:
                raise ValueError('Test answers must not enter retrieval')
            text = remove_double_space(task.get_question(meta))
            # Query IDs are strings, while DPR's pool index IDs are integers.
            # Thus split-local ID reuse cannot accidentally exclude a candidate.
            self.data.append(QASample(text, row['id'], ['None'], meta))
