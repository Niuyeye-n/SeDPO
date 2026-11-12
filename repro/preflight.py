"""Validate actual frozen data and prove selector adapters do not read test gold."""
import argparse
from pathlib import Path
from unittest.mock import patch
import common


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--with-dpr', action='store_true')
    args = parser.parse_args()
    original_read = common.read

    def without_test_gold(path):
        if Path(path).name.endswith('_gold.json'):
            raise AssertionError('Selector attempted to open test gold')
        return original_read(path)

    for task in ['arc_c', 'openbookqa']:
        with patch('common.read', side_effect=without_test_gold):
            _, data, checksum = common.load_protocol(args.protocol, task, include_gold=False)
            if args.with_dpr:
                from sedpo_adapter import ControlledContexts, ControlledQuestions
                contexts = ControlledContexts(args.protocol, task)
                passages = {}
                contexts.load_data_to(passages)
                queries = ControlledQuestions(args.protocol, task)
                queries.load_data()
                if len(passages) != 50 or len(queries.data) != 500:
                    raise AssertionError('DPR did not receive the frozen counts')
                if len({row.query for row in queries.data}) != 500:
                    raise AssertionError('Original DPR keys queries by text; duplicate retrieval text needs explicit treatment')
                if any('answerKey' in row.meta_data for row in queries.data):
                    raise AssertionError('DPR query metadata contains a test answer')
        print(f'{task}: pool={len(data["pool"])} dev={len(data["dev"])} test={len(data["queries"])} '
              f'protocol={checksum}; test-gold access blocked successfully; DPR adapters={args.with_dpr}')


if __name__ == '__main__':
    main()
