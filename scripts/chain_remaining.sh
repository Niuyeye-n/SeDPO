#!/bin/bash
# =============================================================================
# 剩余队列（2026-11-07 07:30）
# =============================================================================
# 背景：4 组 SeDPO 的打分(Q1/Q2/Q3)已全部完成，但训练三次被旧 janitor 的
#       「紧急模式」删掉 model 目录而中断。现在：
#   - llama:close_qa 正在由主编排跑（预计 07:55 训练完，随后 infer）
#   - 剩余待补：qwen:close_qa、qwen:reading_new、llama:reading_new
#
# 本脚本：等主编排彻底退出后，串行补跑这三组（score 会全部跳过，直接训练）。
# =============================================================================
set -u
MAIN_PID=${MAIN_PID:-265286}
LOG=/root/autodl-tmp/chain_remaining.log
REMOTE=/root/autodl-tmp/run_sedpo_v3_resume.sh
SELECT="qwen:close_qa,qwen:reading_new,llama:reading_new"

echo "[$(date '+%F %T')] 队列启动：等待主编排 PID=$MAIN_PID 结束" >> $LOG

while kill -0 "$MAIN_PID" 2>/dev/null; do
    sleep 60
done
# 双保险：等到确实没有 run_sedpo_v3 编排进程（排除自身）
while pgrep -f "bash run_sedpo_v3.sh" >/dev/null 2>&1; do
    sleep 30
done

echo "[$(date '+%F %T')] 主编排已结束，开始补跑：$SELECT" >> $LOG
df -h /root/autodl-tmp | tail -1 >> $LOG

SELECT="$SELECT" bash "$REMOTE" >> $LOG 2>&1
rc=$?
echo "[$(date '+%F %T')] 补跑队列结束 rc=$rc" >> $LOG
