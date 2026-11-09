#!/bin/bash
# =============================================================================
# SeDPO 实验自动巡检（2026-11-05）
# 作用：定期检查实验健康度；发现「进程死了但编排没退出」时自动拉起；
#       磁盘不足时清理；把每次巡检快照写进日志便于回溯。
#
# 设计要点（针对已踩过的坑）：
#   1. 不使用 while+sleep 无限循环（避免成为 PPID=1 的孤儿守护进程），
#      改为有限轮次 + 外部 cron 触发。
#   2. 只重启「编排脚本已死但 GPU 空闲」的情况，绝不重启正在正常跑的打分。
#   3. 每次动作都记日志，出问题能查。
#
# 用法: bash /root/autodl-tmp/watchdog_v3.sh [轮数] [间隔秒]
# =============================================================================
set -u

REPO=/root/autodl-tmp/SeDPO_NIPS25
RESULTS=$REPO/results_sedpo_v3
LOG=/root/autodl-tmp/watchdog_v3.log
ROUNDS=${1:-2880}          # 2880 轮 × 60s ≈ 48 小时
INTERVAL=${2:-60}

mkdir -p /root/autodl-tmp

log() {
    echo "[$(date +%Y-%m-%d_%H:%M:%S)] $*" >> $LOG
}

log "===== watchdog_v3 启动 (rounds=$ROUNDS interval=${INTERVAL}s) ====="

for r in $(seq 1 $ROUNDS); do
    # ---- 1. 磁盘守护：低于 3G 时清理已完成的 model 目录 ----
    AVAIL=$(df -BG /root/autodl-tmp | tail -1 | awk '{gsub("G","",$4); print $4}')
    if [ "${AVAIL:-0}" -lt 3 ]; then
        log "[磁盘] 仅剩 ${AVAIL}G → 清理已完成的 model 目录"
        find $REPO/my_data_qwen/experiment_sedpo_*/ -maxdepth 2 -type d -name model \
             -exec rm -rf {} + 2>/dev/null
        # 再不够就清 epoch checkpoint（只留 best_valid）
        find $REPO/my_data_qwen/experiment_sedpo_*/ -maxdepth 3 -name 'dpr_biencoder.[0-9]*' \
             -delete 2>/dev/null
        AVAIL=$(df -BG /root/autodl-tmp | tail -1 | awk '{gsub("G","",$4); print $4}')
        log "[磁盘] 清理后剩余 ${AVAIL}G"
    fi

    # ---- 2. 健康检查：编排进程是否还活着 ----
    RUNNER=$(ps -eo pid,cmd | grep 'run_sedpo_v3.sh' | grep -v grep | head -1 | awk '{print $1}')
    WORKER=$(ps -eo pid,cmd | grep -E 'scorer.py|train_dense_encoder.py|inference.py' | grep -v grep | head -1 | awk '{print $1}')

    if [ -z "$RUNNER" ]; then
        # 编排已退出：区分「全部完成」与「中途死掉」
        DONE=$(grep -c "# 全部结束" $RESULTS/../sedpo_v3.log 2>/dev/null || echo 0)
        if [ "${DONE:-0}" -gt 0 ]; then
            log "[完成] 编排已正常结束（检测到'全部结束'标记），退出巡检"
            break
        fi
        # 中途死掉 → 检查 GPU 是否空闲，空闲则重启
        USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        if [ "${USED:-9999}" -lt 500 ]; then
            log "[异常] 编排已死且 GPU 空闲(${USED}MiB) → 自动重启"
            cd /root/autodl-tmp && setsid nohup bash run_sedpo_v3.sh \
                >> /root/autodl-tmp/sedpo_v3.log 2>&1 < /dev/null &
            sleep 30
            log "[异常] 重启完成"
        else
            log "[异常] 编排已死但 GPU 仍占用(${USED}MiB) → 不重启，等待子进程收尾"
        fi
    elif [ -z "$WORKER" ]; then
        # 编排活着但没有工作进程 → 可能卡在阶段切换，加日志但不重启
        log "[观察] 编排存活但暂无工作进程（可能在阶段切换/预处理）"
    fi

    # ---- 3. 结果自动入表：每轮检查是否有新完成的组，幂等追加进 runs_log.csv ----
    if [ -f $REPO/20261105_append_results.py ]; then
        APP=$(/root/autodl-tmp/envs/se_dpo/bin/python $REPO/20261105_append_results.py \
              $RESULTS $REPO/experiment_log/runs_log.csv 2>&1)
        case "$APP" in
            *"[OK] 已追加"*) log "[入表] $APP" ;;
        esac
    fi

    sleep $INTERVAL
done

log "===== watchdog_v3 结束 ====="
