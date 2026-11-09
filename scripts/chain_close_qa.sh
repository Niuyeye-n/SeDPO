#!/bin/bash
# =============================================================================
# close_qa 补跑排队器（2026-11-06）
# =============================================================================
# 背景：
#   qwen/close_qa 与 llama/close_qa 的三轮打分均已完成（各约 7 小时产物），
#   但训练阶段因 batch 被放大 8 倍（8→64）导致 24GB 显存 OOM 而失败。
#   现在已修：梯度检查点 + 限制 checkpoint 数量（见 patch_grad_ckpt.py /
#   patch_train_sh.py）。打分产物存在时 scorer 会跳过，所以补跑只需
#   训练 + 推理，每组约 1.5~2 小时。
#
# 本脚本作用：
#   等主实验（run_sedpo_v3.sh 的两个 reading_new 组）全部结束后，
#   自动用 run_sedpo_v3_resume.sh 补跑 close_qa 两组，避免 GPU 空转。
#
# 用法: setsid nohup bash /root/autodl-tmp/chain_close_qa.sh >/dev/null 2>&1 &
# =============================================================================
set -u

LOG=/root/autodl-tmp/closeqa_chain.log
MAX_WAIT=2160          # 最多等 36 小时

log() { echo "[$(date +%F' '%T)] $*" >> "$LOG"; }

log "===== 排队器启动：等待主实验结束 ====="

for i in $(seq 1 $MAX_WAIT); do
    # 主编排：注意 grep 的是 run_sedpo_v3.sh，本脚本补跑用的是 resume 名字，不会自我命中
    MAIN=$(ps -eo pid,cmd | grep 'run_sedpo_v3\.sh' | grep -v grep | head -1)
    RESUME=$(ps -eo pid,cmd | grep 'run_sedpo_v3_resume\.sh' | grep -v grep | head -1)

    if [ -n "$RESUME" ]; then
        log "补跑进程已在运行，退出排队器"
        exit 0
    fi

    if [ -z "$MAIN" ]; then
        USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        if [ "${USED:-9999}" -lt 500 ]; then
            log "主实验已结束且 GPU 空闲(${USED}MiB) → 开始补跑 close_qa（qwen+llama）"
            cd /root/autodl-tmp || exit 1
            SELECT=qwen:close_qa,llama:close_qa \
                setsid bash /root/autodl-tmp/run_sedpo_v3_resume.sh \
                >> /root/autodl-tmp/sedpo_v3_closeqa.log 2>&1
            RC=$?
            log "close_qa 补跑结束 rc=$RC"
            log "===== 现有指标文件 ====="
            find /root/autodl-tmp/SeDPO_NIPS25/results_sedpo_v3 -name 'metrics_sedpo.txt' \
                 -printf '%p\n' >> "$LOG" 2>/dev/null
            exit 0
        else
            log "主实验进程已退出但 GPU 仍占用 ${USED}MiB → 等待收尾"
        fi
    fi

    sleep 60
done

log "等待超时（${MAX_WAIT} 分钟），未启动补跑"
exit 1
