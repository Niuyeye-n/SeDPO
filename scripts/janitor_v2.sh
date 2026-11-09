#!/bin/bash
# =============================================================================
# 磁盘守护 v2（2026-11-05）
# =============================================================================
# 在原 janitor.sh 基础上，额外覆盖 v3 实验的目录 experiment_sedpo_{qwen,llama}/
# （原版只覆盖 experiment_{qwen,llama}/，会漏掉 v3 的产物）
#
# 为什么需要：
#   SeDPO 每组训练会在 <exp>/model/ 下产生约 4~7.4 GB：
#     - dpr_biencoder.<epoch>   每 epoch checkpoint（含 optimizer state，
#                               BERT-base 110M 参数 fp32 → 单个约 1.3 GB）
#     - dpr_biencoder.best_valid  验证最优 checkpoint（infer.sh 必须用）
#     - dpr_enc_index*            整库 prompt_pool 的稠密向量索引
#   4 组（2 模型 × 2 集群）累计约 30 GB，而数据盘仅约 12 GB 可用 → 必须边跑边清。
#
# 清理时机（严格遵守，避免破坏正在运行的任务）：
#   A. 训练已结束（无 train_dense_encoder 进程）且 best_valid 已生成
#      → 删掉各 epoch 的中间 checkpoint，只保留 best_valid
#   B. infer 的 preds json 已产出（infer 全部结束后）
#      → 整个 model 目录可删，infer 不会再用到
#
# 兜底：可用空间 < 3 GB 时紧急清理已完成的组，并打印告警。
# =============================================================================
REPO=/root/autodl-tmp/SeDPO_NIPS25
LOG=/root/autodl-tmp/janitor.log

log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

free_gb() { df -BG --output=avail /root/autodl-tmp 2>/dev/null | tail -1 | tr -dc '0-9'; }

# 该组是否有训练进程在跑
train_running() {
    local tag=$1 cluster=$2
    ps -eo args 2>/dev/null | grep -F "experiment_${tag}/${cluster}/model" \
        | grep -v grep | grep -q "train_dense_encoder" && return 0 || return 1
}

check_group() {
    local tag=$1 cluster=$2 exp md done tasks t p n sz
    exp="$REPO/my_data_qwen/experiment_${tag}/${cluster}"
    md="$exp/model"
    [ -d "$md" ] || return 0

    if [ "$cluster" = "close_qa" ]; then
        tasks="arc_c"
    else
        tasks="openbookqa multirc"
    fi

    # ---- B: infer 完成（preds 齐全）→ 整个 model 目录删除 ----
    done=1
    for t in $tasks; do
        p="$exp/preds_for_${cluster}/${t}_prompts3_retrieverse2_preds_0.json"
        [ -f "$p" ] || done=0
    done
    if [ "$done" -eq 1 ]; then
        sz=$(du -sm "$md" 2>/dev/null | cut -f1)
        rm -rf "$md"
        log "B: 删除 ${tag}/${cluster}/model (释放 ${sz} MB) — infer 已完成"
        return 0
    fi

    # ---- A: 训练已结束但 infer 未完成 → 只删 epoch 中间 checkpoint ----
    if [ -f "$md/dpr_biencoder.best_valid" ] && ! train_running "$tag" "$cluster"; then
        n=$(ls "$md"/dpr_biencoder.[0-9]* 2>/dev/null | wc -l)
        if [ "$n" -gt 0 ]; then
            sz=$(du -cm "$md"/dpr_biencoder.[0-9]* 2>/dev/null | tail -1 | cut -f1)
            rm -f "$md"/dpr_biencoder.[0-9]*
            log "A: 删除 ${tag}/${cluster} 的 ${n} 个 epoch checkpoint (释放 ${sz} MB)，保留 best_valid"
        fi
    fi
}

# 需要巡检的 tag 列表：原版目录 + v3 隔离目录
TAGS="qwen llama sedpo_qwen sedpo_llama"

log "=== 磁盘守护 v2 启动（覆盖 tags: $TAGS） ==="
while true; do
    for tag in $TAGS; do
        for cluster in close_qa reading_new; do
            check_group "$tag" "$cluster"
        done
    done

    fg=$(free_gb)
    if [ -n "$fg" ] && [ "$fg" -lt 3 ]; then
        log "!! 磁盘告急：仅剩 ${fg} GB，尝试紧急清理已完成的组"
        for tag in $TAGS; do
            for cluster in close_qa reading_new; do
                exp="$REPO/my_data_qwen/experiment_${tag}/${cluster}"
                if [ -d "$exp/model" ]; then
                    rm -rf "$exp/model"
                    log "!! 紧急删除 ${tag}/${cluster}/model"
                fi
            done
        done
    fi
    sleep 10
done
