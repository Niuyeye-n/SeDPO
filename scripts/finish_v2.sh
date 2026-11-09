#!/bin/bash
# =============================================================================
# finish_v2.sh — 2026-11-07 17:5x
# 目的：补齐 SeDPO 四组缺失的推理与指标，不重训、不重跑已有 preds。
#
# 为什么重写 v1：v1 直接 bash infer.sh，但没激活 conda 环境，
#   infer.sh 里裸调 python / accelerate 全部 command not found（rc=127），
#   六个步骤在 1 秒内连环失败，等于什么都没跑。
#
# 本版做法：
#   1. 显式导出 se_dpo 环境的 PATH / PYTHONPATH / SeDPO 版本变量
#   2. 只补缺口：qwen/close_qa 全流程；两组 reading_new 的 openbookqa 段
#   3. MultiRC 的 preds 已存在 → 用 compute_res.py 反算指标，不重跑推理
# =============================================================================
REPO=/root/autodl-tmp/SeDPO_NIPS25
LOG=/root/autodl-tmp/finish_v2.log
PYBIN=/root/autodl-tmp/envs/se_dpo/bin
QWEN=/root/autodl-tmp/models/qwen2.5-7b
LLAMA=/root/autodl-tmp/models/llama3-8b

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# ---- 环境 ----
export PATH=$PYBIN:$PATH
export PYTHONPATH=$REPO:${PYTHONPATH:-}
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# SeDPO 版本信息（决定 res_file 里写 dp2-r1.0-b0.1，必须与之前一致）
export IF_PREFER_LOSS=2
export PEFT_BETA=0.1
export DIV_REJECT=1
export SCORE_RATIO=1.0

cd "$REPO" || exit 1
log "===== finish_v2 启动 ====="
log "python: $(which python) $(python -V 2>&1)"
log "accelerate: $(which accelerate)"

# ---- 补 MultiRC 指标行（preds 已存在，只需反算）----
fill_multirc() {
    local tag=$1 model=$2
    local EXP=$REPO/my_data_qwen/experiment_sedpo_${tag}/reading_new
    python /root/autodl-tmp/compute_res.py \
        "$EXP/preds_for_reading_new/multirc_prompts3_retrieverse2_preds_0.json" \
        "$EXP/eval_res_for_reading_new.txt" \
        multirc "$model" 2>&1 | tee -a "$LOG"
}

# ---- openbookqa 检索 + 推理 ----
run_openbookqa() {
    local tag=$1 model=$2 port=$3
    local EXP=$REPO/my_data_qwen/experiment_sedpo_${tag}/reading_new

    if [ ! -f "$EXP/model/dpr_biencoder.best_valid" ]; then
        log "[跳过] $tag/reading_new 缺 best_valid 权重"
        return 1
    fi

    log "--- $tag openbookqa 检索开始"
    python DPR/dense_retriever_beam_score.py \
        model_file=$EXP/model/dpr_biencoder.best_valid \
        qa_dataset=qa_se2 ctx_datatsets=[dpr_se2] \
        "encoded_ctx_files=[\"$EXP/model/dpr_enc_index_*\"]" \
        out_file=$EXP/se2_prompts_for_reading_new/openbookqa_prompts.json \
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
    log "--- $tag openbookqa 检索结束 rc=$rc"
    [ $rc -ne 0 ] && return $rc

    log "--- $tag openbookqa 推理开始"
    accelerate launch --num_processes 1 --main_process_port "$port" \
        scripts/inference.py \
        prompt_file=$EXP/se2_prompts_for_reading_new/openbookqa_prompts_beam_score_0.json \
        task_name=openbookqa \
        output_file=$EXP/preds_for_reading_new/openbookqa_prompts3_retrieverse2_preds_0.json \
        res_file=$EXP/eval_res_for_reading_new.txt \
        batch_size=1 train_clusters=reading_new \
        model_name=$model \
        prompt_pool_path=$REPO/my_data_qwen/prompt_pool_sedpo \
        random_sample=False random_seed=42 \
        cache_dir=$REPO/cache \
        hydra.run.dir=$EXP >> "$LOG" 2>&1
    rc=$?
    log "--- $tag openbookqa 推理结束 rc=$rc"
    return $rc
}

# ---- qwen/close_qa 全流程（缺 index + prompts + preds）----
run_close_qa_qwen() {
    local EXP=$REPO/my_data_qwen/experiment_sedpo_qwen/close_qa
    if [ ! -f "$EXP/model/dpr_biencoder.best_valid" ]; then
        log "[跳过] qwen/close_qa 缺 best_valid 权重"
        return 1
    fi
    log "--- qwen/close_qa 全流程开始（index → 检索 → 推理）"
    bash "$EXP/infer.sh" >> "$LOG" 2>&1
    local rc=$?
    log "--- qwen/close_qa 全流程结束 rc=$rc"
    return $rc
}

# ============================ 执行 ============================
# 顺序：先补两组 openbookqa（各约 20-40 分钟），最后跑 qwen close_qa 全流程
# 这样能最快把 reading_new 两组收口，permit janitor 释放权重腾空间

log "===== [1/4] llama/reading_new openbookqa ====="
run_openbookqa llama "$LLAMA" 24001
log "===== [2/4] qwen/reading_new openbookqa ====="
run_openbookqa qwen "$QWEN" 24002
log "===== [3/4] 反算 MultiRC 指标 ====="
fill_multirc llama "$LLAMA"
fill_multirc qwen "$QWEN"
log "===== [4/4] qwen/close_qa 全流程 ====="
run_close_qa_qwen

log "===== 全部步骤结束，指标汇总 ====="
for f in $REPO/my_data_qwen/experiment_sedpo_*/{close_qa,reading_new}/eval_res_for_*.txt; do
    [ -f "$f" ] && { log "== $f"; sed 's/^/    /' "$f" | tee -a "$LOG"; }
done
log "===== finish_v2 完成 ====="
