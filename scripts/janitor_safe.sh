#!/bin/bash
# =============================================================================
# 磁盘清理（安全版 v3，2026-11-07）
# =============================================================================
# 为什么重写：旧 janitor_v2.sh 的「紧急模式」在可用空间 < 3 GB 时无条件
#   rm -rf 所有 experiment_*/model 目录，**不判断该组是否正在训练**。
#   2026-11-06 20:06、11-07 05:47、11-07 06:54 三次触发，
#   分别删掉了正在训练的 reading_new_qwen、reading_new_llama、close_qa_qwen
#   的 model 目录 → 训练在保存 checkpoint 时崩溃，白跑 50 分钟~2.2 小时。
#
# 本版规则（只做两件事，绝无紧急删除）：
#   B. 某组 infer 已产出全部 preds（说明该组彻底结束）→ 删整个 model 目录
#   A. 某组 best_valid 已生成 且 该组无训练进程 → 只删 dpr_biencoder.<epoch> 中间权重
#
# 硬约束：只要该组还有 train_dense_encoder 进程，绝不碰它的 model 目录。
# =============================================================================
REPO=/root/autodl-tmp/SeDPO_NIPS25
LOG=/root/autodl-tmp/janitor_safe.log
TAGS="sedpo_qwen sedpo_llama"

log() { echo "[$(date +%m-%d\ %H:%M:%S)] $*" >> "$LOG"; }

# 该组是否有训练进程在跑（用输出目录做精确匹配）
train_running() {
    local tag=$1 cluster=$2
    ps -eo args 2>/dev/null | grep -F "experiment_${tag}/${cluster}/model" \
        | grep -v grep | grep -q "train_dense_encoder" && return 0 || return 1
}

# 该组是否有推理进程在跑
infer_running() {
    local tag=$1 cluster=$2
    ps -eo args 2>/dev/null | grep -F "experiment_${tag}/${cluster}" \
        | grep -v grep | grep -qE "inference.py|generate_dense_embeddings|dense_retriever" && return 0 || return 1
}

check_group() {
    local tag=$1 cluster=$2 exp md tasks t p done sz
    exp="$REPO/my_data_qwen/experiment_${tag}/${cluster}"
    md="$exp/model"
    [ -d "$md" ] || return 0

    # 有训练或推理在跑 → 一律不碰
    train_running "$tag" "$cluster" && return 0
    infer_running "$tag" "$cluster" && return 0

    if [ "$cluster" = "close_qa" ]; then
        tasks="arc_c"
    else
        tasks="openbookqa multirc"
    fi

    # ---- B: infer 全部完成 → 整个 model 目录可删（infer 不会再读它）----
    done=1
    for t in $tasks; do
        p="$exp/preds_for_${cluster}/${t}_prompts3_retrieve${cluster%%_*}_preds_0.json"
        [ -f "$p" ] || p="$exp/preds_for_${cluster}/${t}_prompts3_retrieverse2_preds_0.json"
        [ -f "$p" ] || done=0
    done
    if [ "$done" -eq 1 ]; then
        sz=$(du -sm "$md" 2>/dev/null | cut -f1)
        rm -rf "$md"
        log "B: 删除 ${tag}/${cluster}/model (释放 ${sz} MB) — 该组 infer 已完成"
        return 0
    fi

    # ---- A: 训练结束、best_valid 已保存 → 删中间 epoch 权重 ----
    if [ -f "$md/dpr_biencoder.best_valid" ]; then
        local n
        n=$(ls "$md"/dpr_biencoder.[0-9]* 2>/dev/null | wc -l)
        if [ "$n" -gt 0 ]; then
            sz=$(du -cm "$md"/dpr_biencoder.[0-9]* 2>/dev/null | tail -1 | cut -f1)
            rm -f "$md"/dpr_biencoder.[0-9]*
            log "A: 删 ${tag}/${cluster} 的 ${n} 个 epoch 权重 (${sz} MB)，保留 best_valid"
        fi
    fi
}

log "=== 安全清理 v3 启动（tags: $TAGS，无紧急删除） ==="
while true; do
    for tag in $TAGS; do
        for cluster in close_qa reading_new; do
            check_group "$tag" "$cluster"
        done
    done
    sleep 60
done
