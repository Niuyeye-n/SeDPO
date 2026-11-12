"""Compute the actual SeDPO IF_PREFER_LOSS=2 training expansion before setting warmup."""
import argparse
import json
from pathlib import Path
from common import file_sha, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--valid', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    train, valid = [json.loads(path.read_text()) for path in [args.train, args.valid]]
    train_ids, valid_ids = [{row['id'] for row in rows} for rows in [train, valid]]
    if train_ids & valid_ids:
        raise ValueError('Training and validation query IDs overlap')
    # The unchanged get_multi_prefer_entry() creates min(correct, T, L//2)
    # pairs per entry when IF_PREFER_LOSS=2 and SCORE_RATIO=1.
    pairs = sum(min(sum(float(ctx['one_shot_acc']) > 0 for ctx in row['ctxs']), 20, len(row['ctxs']) // 2) for row in train)
    per_epoch, epochs, batch = pairs // 64, 6, 64
    if per_epoch < 1:
        raise ValueError('No effective training batches')
    total = per_epoch * epochs
    warmup = max(1, round(total * 0.1))
    if warmup >= total:
        raise ValueError('Warmup must be smaller than total updates')
    plan = {'training_entries': len(train), 'validation_entries': len(valid),
            'training_unique_queries': len(train_ids), 'validation_unique_queries': len(valid_ids),
            'expanded_training_samples': pairs, 'effective_batch_size': batch, 'epochs': epochs,
            'updates_per_epoch': per_epoch, 'total_updates': total, 'warmup_steps': warmup,
            'train_sha256': file_sha(args.train), 'valid_sha256': file_sha(args.valid)}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write(args.out_dir / 'training_plan.json', plan)
    (args.out_dir / 'warmup_steps.txt').write_text(str(warmup) + '\n')
    print(json.dumps(plan))


if __name__ == '__main__':
    main()
