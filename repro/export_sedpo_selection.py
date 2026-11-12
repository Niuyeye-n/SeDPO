"""Convert the original SeDPO beam output into the shared selection contract."""
import argparse
from pathlib import Path
from common import digest, file_sha, load_protocol, read, validate_selection, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--task', required=True, choices=['arc_c', 'openbookqa'])
    parser.add_argument('--beam-output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Preserve the previous selection')
    _, data, protocol_sha = load_protocol(args.protocol, args.task, include_gold=False)
    rows = [{'query_id': str(row['meta_data']['id']),
             'demo_ids': [str(ctx['meta_data']['id']) for ctx in row['ctxs']]}
            for row in read(args.beam_output)]
    files = [Path(__file__), Path(__file__).with_name('sedpo_adapter.py'),
             args.repo / 'DPR/dense_retriever_beam_score.py', args.repo / 'DPR/generate_dense_embeddings.py',
             args.repo / 'DPR/dpr/utils/tasks.py']
    selection = {'method': 'sedpo', 'protocol_sha256': protocol_sha, 'task_name': args.task,
                 'pool_sha256': digest(data['pool']), 'queries_sha256': digest(data['queries']),
                 'selector_manifest': {'test_labels_used': False, 'tuning_splits': ['train', 'dev'],
                                       'implementation_files': [{'path': str(path.resolve()), 'sha256': file_sha(path)} for path in files],
                                       'checkpoint': {'path': str(args.checkpoint.resolve()), 'sha256': file_sha(args.checkpoint)},
                                       'beam_output_sha256': file_sha(args.beam_output), 'beam_size': 3, 'k': 4,
                                       'description': 'Original DPR SeDPO beam search, restricted to the frozen 50 candidates; adapters only replace data loading; original-ID-based task templates',
                                       'training_budget_note': 'Retriever was trained with the previously recorded full-pool preference data; no equal offline-cost claim'},
                 'rows': rows}
    write(args.out, selection)
    validate_selection(args.out, protocol_sha=protocol_sha, task=args.task, pool=data['pool'], queries=data['queries'])
    print(f'Exported {len(rows)} SeDPO selections: {args.out}')


if __name__ == '__main__':
    main()
