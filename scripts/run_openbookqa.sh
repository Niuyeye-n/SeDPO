#!/bin/bash
# =============================================================================
# run_openbookqa.sh — 2026-11-07 19:2x
# 只补两组 openbookqa 推理。prompts 已在上次（19:13/19:14）检索完成，
# 这次直接进 inference，不再做检索。
# 依赖：inference.py 已打「分块 loss」补丁（patch_inference2.py）
# =============================================================================
REPO=/root/autodl-tmp/SeDPO_NIPS25
LOG=/root/autodl-tmp/openbookqa_run.log
PYBIN=/root/autodl-tmp/envs/se_dpo/bin

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

export PATH=$PYBIN:$PATH
export PYTHONPATH=$REPO:${PYTHONPATH:-}
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export IF_PREFER_LOSS=2
export PEFT_BETA=0.1
export DIV_REJECT=1
export SCORE_RATIO=1.0

cd "$REPO" || exit 1
log "===== openbookqa 补跑启动 ====="
log "python=$(which python)"

run_one() {
    local tag=$1 model=$2 port=$3
    local E=$REPO/my_data_qwen/experiment_sedpo_${tag}/reading_new
    local PF=$E/se2_prompts_for_reading_new/openbookqa_prompts_beam_score_0.json

    if [ ! -f "$PF" ]; then
        log "[跳过] $tag 缺 openbookqa prompt"
        return 1
    fi

    log "--- $tag/openbookqa 推理开始"
    accelerate launch --num_processes 1 --main_process_port "$port" \
        scripts/inference.py \
        prompt_file=$PF \
        task_name=openbookqa \
        output_file=$E/preds_for_reading_new/openbookqa_prompts3_retrieverse2_preds_0.json \
        res_file=$E/eval_res_for_reading_new.txt \
        batch_size=1 train_clusters=reading_new \
        model_name=$model \
        prompt_pool_path=$REPO/my_data_qwen/prompt_pool_sedpo \
        random_sample=False random_seed=42 \
        cache_dir=$REPO/cache \
        hydra.run.dir=$E >> "$LOG" 2>&1
    local rc=$?
    log "--- $tag/openbookqa 推理结束 rc=$rc"
    return $rc
}

log "===== [1/2] llama ====="
run_one llama /root/autodl-tmp/models/llama3-8b 24201
log "===== [2/2] qwen ====="
run_one qwen /root/autodl-tmp/models/qwen2.5-7b 24202

log "===== 汇总 ====="
for f in $REPO/my_data_qwen/experiment_sedpo_*/{close_qa,reading_new}/eval_res_for_*.txt; do
    [ -f "$f" ] && { log "== $(echo $f | sed 's#.*experiment_##')"; sed 's/^/    /' "$f" | tee -a "$LOG"; }
done
log "===== 完成 ====="
