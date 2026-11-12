#!/usr/bin/env bash
# ARC-C controlled rerun: only warmup changes from 1000 to 86.
# Prepared for review; run explicitly after approval and when the GPU is idle.
# Existing scored data and the original experiment are read-only inputs.
set -euo pipefail

REPO=/root/autodl-tmp/SeDPO_NIPS25
EXP="$REPO/my_data_qwen/experiment/close_qa_warmup86_seed12345"
PY=/root/autodl-tmp/envs/se_dpo/bin/python
SCRIPT_PATH=$(realpath -- "${BASH_SOURCE[0]}")

export PATH="/root/autodl-tmp/envs/se_dpo/bin:$PATH"
export PYTHONPATH="$REPO"
export HF_ENDPOINT=https://hf-mirror.com
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export IF_PREFER_LOSS=2
export DIV_REJECT=1
export PEFT_BETA=0.1
export SCORE_RATIO=1.0
export GRAD_CKPT=1
export PYTHONUNBUFFERED=1
# PT_MODEL overrides model_file in the trainer, so it must be absent.
unset PT_MODEL

cd "$REPO"
test -x "$PY"
test -s my_data_qwen/scored/close_qa/arc_c_scored_train.json
test -s my_data_qwen/scored/close_qa/arc_c_scored_valid.json

# get_model_file() also searches output_dir automatically. Refuse an existing
# experiment directory so a second invocation cannot silently resume weights.
if ! mkdir "$EXP"; then
    printf 'Refusing to reuse experiment directory: %s\n' "$EXP" >&2
    exit 1
fi
mkdir "$EXP/model"
cp -- "$SCRIPT_PATH" "$EXP/run_script.sh"
sha256sum \
    DPR/train_dense_encoder.py \
    DPR/dpr/data/biencoder_data.py \
    DPR/dpr/models/biencoder.py \
    DPR/dpr/models/hf_models.py \
    DPR/conf/biencoder_train_cfg.yaml \
    DPR/conf/train/biencoder_se2.yaml \
    DPR/conf/encoder/hf_bert.yaml \
    my_data_qwen/scored/close_qa/arc_c_scored_train.json \
    my_data_qwen/scored/close_qa/arc_c_scored_valid.json \
    my_data_qwen/prompt_pool/close_qa/arc_c_prompts.json \
    > "$EXP/input_sha256.txt"

# seed=12345 and model_file=null spell out the original effective settings.
# CLI batch 8 is multiplied by 8 inside SeDPO, yielding the original batch 64.
"$PY" DPR/train_dense_encoder.py \
    'train_datasets=[se2_dataset]' \
    'dev_datasets=[se2_valid_dataset]' \
    train=biencoder_se2 \
    "output_dir=$EXP/model" \
    datasets.train_clusters=close_qa \
    "datasets.train_file=$REPO/my_data_qwen/scored" \
    "datasets.valid_file=$REPO/my_data_qwen/scored" \
    datasets.hard_neg=true datasets.top_k=20 \
    train.hard_negatives=20 train.batch_size=8 \
    train.num_train_epochs=6 train.num_save_epochs=2 \
    train.learning_rate=1e-5 train.warmup_steps=86 \
    seed=12345 model_file=null \
    "datasets.prompt_pool_path=$REPO/my_data_qwen/prompt_pool" \
    datasets.prompt_setup_type=qa datasets.task_setup_type=q \
    "encoder.cache_dir=$REPO/cache" \
    "hydra.run.dir=$EXP" \
    2>&1 | tee "$EXP/train.log"

test -s "$EXP/model/dpr_biencoder.best_valid"
printf 'Training completed. Checkpoint: %s\n' "$EXP/model/dpr_biencoder.best_valid"
