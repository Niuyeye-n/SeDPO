#!/bin/bash
# =============================================================================
# 收尾脚本（2026-11-07 13:40）
# =============================================================================
# 现状（四组都已完成训练并产出 best_valid）：
#   llama/close_qa   : prompts+preds+eval 全齐（acc 50.64%）—— 完成
#   qwen/close_qa    : 训练正在进行，缺 arc_c 推理
#   qwen/reading_new : multirc 已完成，缺 openbookqa 推理 + multirc 的 eval 行
#   llama/reading_new: 同上
#
# 本脚本只补缺口，不重训：
#   1. 等 qwen/close_qa 训练结束
#   2. qwen/close_qa 全量 infer（arc_c）
#   3. 每组 reading_new：先补 multirc 的 eval 行（复用已存 prompts，只重跑推理），
#      再跑 openbookqa 段（检索 + 推理）
#
# 顺序注意：必须先 multirc 后 openbookqa。原因是 janitor_safe 的 B 规则在
#   「两个 task 的 preds 都齐」时会删掉 model 目录，若先跑 openbookqa，
#   multirc 重跑就会因缺 best_valid 失败。
# =============================================================================
set -u
R=/root/autodl-tmp/SeDPO_NIPS25
LOG=/root/autodl-tmp/finish_all.log
STALE=/root/autodl-tmp/_stale_tmp
cd "$R" || exit 1
mkdir -p "$STALE"

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

log "===== 收尾开始 ====="

# ---- 1) 等 qwen/close_qa 训练结束 ----
log "等待 qwen/close_qa 训练结束..."
n=0
while pgrep -f "train_dense_encoder.py.*experiment_sedpo_qwen/close_qa/model" >/dev/null 2>&1; do
    sleep 30
    n=$((n+1))
    [ $((n % 20)) -eq 0 ] && log "  ...仍在训练（已等 $((n/2)) 分钟）"
done
log "训练结束，model 目录内容: $(ls my_data_qwen/experiment_sedpo_qwen/close_qa/model/ 2>/dev/null | tr '\n' ' ')"

# ---- 2) qwen/close_qa 全量推理 ----
if [ -f my_data_qwen/experiment_sedpo_qwen/close_qa/model/dpr_biencoder.best_valid ]; then
    log "--- qwen/close_qa infer 开始"
    bash my_data_qwen/experiment_sedpo_qwen/close_qa/infer.sh >> "$LOG" 2>&1
    log "--- qwen/close_qa infer 结束 rc=$?"
else
    log "[跳过] qwen/close_qa 无 best_valid，无法推理"
fi

# ---- 3) 两组 reading_new 补缺口 ----
for t in qwen llama; do
    EXP="my_data_qwen/experiment_sedpo_${t}/reading_new"

    # 3a. 先补 multirc 的指标行（复用已存在的 prompts，输出覆盖同一 preds）
    if [ -f "$EXP/se2_prompts_for_reading_new/multirc_prompts_beam_score_0.json" ]; then
        # 清掉上次中断留下的临时分片，避免与本次结果叠加
        for f in "$EXP"/preds_for_reading_new/multirc_*jsontmp_*.bin; do
            [ -e "$f" ] && mv -f "$f" "$STALE"/
        done
        log "--- ${t}/reading_new multirc 推理（补指标行）"
        sed -n '32,45p' "$EXP/infer.sh" > "/tmp/mul_${t}.sh"
        bash "/tmp/mul_${t}.sh" >> "$LOG" 2>&1
        log "--- ${t}/reading_new multirc 结束 rc=$?"
    else
        log "[跳过] ${t}/reading_new 缺 multirc prompts"
    fi

    # 3b. 再跑 openbookqa（检索 + 推理），放最后
    if [ -f "$EXP/model/dpr_biencoder.best_valid" ]; then
        log "--- ${t}/reading_new openbookqa 检索+推理"
        sed -n '47,$p' "$EXP/infer.sh" > "/tmp/obq_${t}.sh"
        bash "/tmp/obq_${t}.sh" >> "$LOG" 2>&1
        log "--- ${t}/reading_new openbookqa 结束 rc=$?"
    else
        log "[跳过] ${t}/reading_new 无 best_valid"
    fi
done

# ---- 4) 汇总所有指标 ----
log "===== 收尾完成，指标汇总 ====="
for f in my_data_qwen/experiment_sedpo_*/*/eval_res_*.txt; do
    [ -f "$f" ] || continue
    echo "== $f" >> "$LOG"
    cat "$f" >> "$LOG"
done
log "===== 全部结束 ====="
