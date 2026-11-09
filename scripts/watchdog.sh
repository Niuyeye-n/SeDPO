#!/bin/bash
# ==============================================================================
# SeDPO 实验看门狗 v2
#
# v1 的两个 bug（已修）：
#   1. main_alive 用 `ps -eo args | grep -F "bash $MAIN"` → 会匹配到 watchdog 自己
#      所在的 bash -c 命令行 → 误判"主编排已死" → 每 60 秒误重启一次。
#      改为 grep -xE 精确整行匹配，且排除含 watchdog.sh 的行。
#   2. 误重启会与仍在运行的旧实例并存 → 两个实例同时写同一批产物 → 产物互相破坏
#      （2026-11-05 18:22 就发生过，导致 step1 产物与 prompt_pool 被重写）。
#      v2 在拉起前先确认旧实例确实已退出。
#
# 另一个并发会话会抢 GPU 显存，导致我们的 scorer 拿不到显存被 SIGKILL（rc=137），
# 所以重启前必须先等 GPU 空出来。
#
# 完成标记：/root/autodl-tmp/EXPERIMENT_DONE
# ==============================================================================
MAIN=/root/autodl-tmp/run_sedpo.sh
LOG=/root/autodl-tmp/watchdog.log
DONE_FLAG=/root/autodl-tmp/EXPERIMENT_DONE

log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

# 主编排是否在跑（精确整行匹配，避免自匹配）
main_alive() {
    ps -eo args 2>/dev/null \
        | grep -v watchdog.sh \
        | grep -v " grep" \
        | grep -xE "bash $MAIN" > /dev/null
}

# GPU 上是否有**不属于本实验**的进程占显存
foreign_gpu_user() {
    local used p mine
    used=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
    [ -z "$used" ] && return 1
    for p in $used; do
        mine=$(ps -eo pid,args 2>/dev/null \
               | grep -E "^ *$p " \
               | grep -E "scorer\.py|inference\.py|train_dense_encoder|generate_dense_embeddings|dense_retriever|bench_scorer" \
               | grep -v grep)
        if [ -z "$mine" ]; then
            echo "$p"
            return 0
        fi
    done
    return 1
}

log "=== 看门狗 v2 启动 $(date +%Y-%m-%d_%H:%M:%S) ==="
RESTARTS=0
while true; do
    sleep 60
    [ -f "$DONE_FLAG" ] && { log "检测到完成标记，退出"; break; }

    if main_alive; then
        continue
    fi

    # 主编排已退出 —— 先确认不是"还在收尾"（train/infer 子进程可能仍在）
    if ps -eo args 2>/dev/null | grep -qE "scripts/(scorer|inference)\.py|train_dense_encoder"; then
        continue
    fi

    if grep -q "SeDPO 正式实验全部结束" /root/autodl-tmp/sedpo_all.log 2>/dev/null; then
        log "主编排正常结束，补写完成标记"
        touch "$DONE_FLAG"
        break
    fi

    RESTARTS=$((RESTARTS + 1))
    if [ "$RESTARTS" -gt 40 ]; then
        log "!! 已重启 40 次仍失败，停止自动拉起，需人工介入"
        break
    fi

    # 等 GPU 上的外来占用者结束（不争抢，等它自己退出）
    WAITED=0
    while f=$(foreign_gpu_user) && [ -n "$f" ]; do
        [ "$WAITED" -eq 0 ] && log "检测到外来进程占用 GPU (PID $f)，等待其退出"
        sleep 20
        WAITED=$((WAITED + 1))
        if [ "$WAITED" -gt 150 ]; then   # 等 50 分钟
            log "外来进程 PID $f 长时间不退出，放弃等待（不强抢 GPU）"
            break
        fi
    done

    # 双重确认：旧实例确实不在了才拉起新实例
    if main_alive; then
        log "旧实例仍在运行，取消重启"
        continue
    fi

    log "第 $RESTARTS 次重启主编排（scorer 会 SKIP 已完成 step，从断点继续）"
    TS=$(date +%Y%m%d_%H%M%S)
    mv -f /root/autodl-tmp/sedpo_all.log /root/autodl-tmp/sedpo_all_restart_${TS}.log 2>/dev/null
    setsid bash "$MAIN" > /root/autodl-tmp/sedpo_all.log 2>&1 &
    sleep 40
    if main_alive; then
        log "重启成功"
    else
        log "重启失败"
    fi
done
