#!/bin/bash
# =============================================================================
# finish_v3.sh — 2026-11-07 18:0x
# 补齐 SeDPO 四组缺失的推理与指标。不重训任何已完成的组。
#
# 与 v2 的差别（v2 未跑，问题已查清）：
#   v1 直接 bash infer.sh 没激活 conda → python/accelerate 全部 command not found
#   v2 修了环境，但 MultiRC 的 preds 是在 10:26 / 12:56 两次 CUDA OOM 时中断的
#      残留中间文件（llama 542 条、qwen 516 条，条数都不对），指标不可用
#   v3 = 先打 inference.py 死代码补丁降显存 → 重跑 MultiRC → 再补 OpenBookRC
#        → 最后跑 qwen/close_qa 全流程
# =============================================================================
REPO=/root/autodl-tmp/SeDPO_NIPS25
LOG=/root/autodl-tmp/finish_v3.log
PYBIN=/root/autodl-tmp/envs/se_dpo/bin
QWEN=/root/autodl-tmp/models/qwen2.5-7b
LLAMA=/root/autodl-tmp/models/llama3-8b

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
log "===== finish_v3 启动 ====="
log "python=$(which python) $(python -V 2>&1) | accelerate=$(which accelerate)"

# ---- 把不完整的 preds 移入回收站（不物理删除）----
trash_partial() {
    local tag=$1 task=$2
    local E=$REPO/my_data_qwen/experiment_sedpo_${tag}/reading_new
    local f=$E/preds_for_reading_new/${task}_prompts3_retrieverse2_preds_0.json
    if [ -f "$f" ]; then
        mkdir -p "$E/_trash_partial"
        mv "$f" "$E/_trash_partial/${task}_partial_$(date +%H%M).json"
        log "  已移走不完整的 ${tag}/${task} preds -> _trash_partial/"
    fi
}

# ---- reading_new 的单个任务推理（prompts 已存在，不再检索）----
run_infer_reading() {
    local tag=$1 model=$2 task=$3 port=$4
    local E=$REPO/my_data_qwen/experiment_sedpo_${tag}/reading_new
    local PF=$E/se2_prompts_for_reading_new/${task}_prompts_beam_score_0.json

    if [ ! -f "$PF" ]; then
        log "[跳过] $tag/$task 缺 prompt 文件"
        return 1
    fi
    log "--- $tag/$task 推理开始"
    accelerate launch --num_processes 1 --main_process_port "$port" \
        scripts/inference.py \
        prompt_file=$PF \
        task_name=$task \
        output_file=$E/preds_for_reading_new/${task}_prompts3_retrieverse2_preds_0.json \
        res_file=$E/eval_res_for_reading_new.txt \
        batch_size=1 train_clusters=reading_new \
        model_name=$model \
        prompt_pool_path=$REPO/my_data_qwen/prompt_pool_sedpo \
        random_sample=False random_seed=42 \
        cache_dir=$REPO/cache \
        hydra.run.dir=$E >> "$LOG" 2>&1
    local rc=$?
    log "--- $tag/$task 推理结束 rc=$rc"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed "s/^/    峰值后显存: /" | tee -a "$LOG"
    return $rc
}

# ---- openbookqa 检索 + 推理 ----
run_openbookqa() {
    local tag=$1 model=$2 port=$3
    local E=$REPO/my_data_qwen/experiment_sedpo_${tag}/reading_new

    if [ ! -f "$E/model/dpr_biencoder.best_valid" ]; then
        log "[跳过] $tag/reading_new 缺 best_valid"
        return 1
    fi

    log "--- $tag/openbookqa 检索开始"
    python DPR/dense_retriever_beam_score.py \
        model_file=$E/model/dpr_biencoder.best_valid \
        qa_dataset=qa_se2 ctx_datatsets=[dpr_se2] \
        "encoded_ctx_files=[\"$E/model/dpr_enc_index_*\"]" \
        out_file=$E/se2_prompts_for_reading_new/openbookqa_prompts.json \
        datasets.qa_se2.task_name=openbookqa \
        datasets.qa_se2.task_setup_type=q \
        datasets.qa_se2.cache_dir=$REPO/cache beam_size=3 \
        shot_num=3 \
        ctx_sources.dpr_se2.prompt_pool_path=$REPO/my_data_qwen/prompt_pool_sedpo \
        ctx_sources.dpr_se2.train_clusters=reading_new \
        ctx_sources.dpr_se2.prompt_setup_type=qa \
        encoder.cache_dir=$REPO/cache \
        hydra.run.dir=my_data_qwen/experiment_sedpo_${tag}/reading_new >> "$LOG" 2>&1
    local rc=$?
    log "--- $tag/openbookqa 检索结束 rc=$rc"
    [ $rc -ne 0 ] && return $rc

    run_infer_reading "$tag" "$model" openbookqa "$port"
}

# ---- qwen/close_qa 全流程（缺 index + prompts + preds）----
run_close_qa_qwen() {
    local E=$REPO/my_data_qwen/experiment_sedpo_qwen/close_qa
    if [ ! -f "$E/model/dpr_biencoder.best_valid" ]; then
        log "[跳过] qwen/close_qa 缺 best_valid"
        return 1
    fi
    log "--- qwen/close_qa 全流程开始（index → 检索 → 推理）"
    bash "$E/infer.sh" >> "$LOG" 2>&1
    local rc=$?
    log "--- qwen/close_qa 全流程结束 rc=$rc"
    return $rc
}

# ============================ 执行 ============================
log "===== [1] 清理不完整的 MultiRC preds ====="
trash_partial llama multirc
trash_partial qwen  multirc

log "===== [2] 重跑 MultiRC：llama ====="
run_infer_reading llama "$LLAMA" multirc 24101

log "===== [3] 重跑 MultiRC：qwen ====="
run_infer_reading qwen "$QWEN" multirc 24102

log "===== [4] OpenBookRC：llama ====="
run_openbookqa llama "$LLAMA" 24103

log "===== [5] OpenBookRC：qwen ====="
run_openbookqa qwen "$QWEN" 24104

log "===== [6] qwen/close_qa 全流程 ====="
run_close_qa_qwen

log "===== 全部步骤结束，指标汇总 ====="
for f in $REPO/my_data_qwen/experiment_sedpo_*/{close_qa,reading_new}/eval_res_for_*.txt; do
    [ -f "$f" ] && { log "== $f"; sed 's/^/    /' "$f" | tee -a "$LOG"; }
done
log "===== finish_v3 完成 ====="
