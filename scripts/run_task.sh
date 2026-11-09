#!/bin/bash
# SeDPO 单阶段执行器：自动记录时间戳与耗时，供实验记录表填写
# 用法: run_task.sh <output_dir> <cluster> <stage>
#   output_dir 例: my_data_qwen
#   cluster     例: close_qa / reading_new
#   stage       例: score / train / infer
set -eo pipefail

OD="$1"
CL="$2"
STAGE="$3"

cd /root/autodl-tmp/SeDPO_NIPS25
source /etc/network_turbo
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/se_dpo
export PYTHONPATH=$PWD:$PYTHONPATH

LOG="${OD}/experiment/${CL}/${STAGE}.log"
mkdir -p "$(dirname "$LOG")"
T0=$(date +%s)
echo "" | tee -a "$LOG"
echo "=== START ${STAGE} $(date -Iseconds) ===" | tee -a "$LOG"

case "$STAGE" in
  score) bash "${OD}/experiment/${CL}/score.sh" 2>&1 | tee -a "$LOG" ;;
  train) bash "${OD}/experiment/${CL}/train.sh" 2>&1 | tee -a "$LOG" ;;
  infer) bash "${OD}/experiment/${CL}/infer.sh" 2>&1 | tee -a "$LOG" ;;
  *) echo "未知阶段: $STAGE"; exit 1 ;;
esac

T1=$(date +%s)
echo "=== END ${STAGE} duration_s=$((T1 - T0)) $(date -Iseconds) ===" | tee -a "$LOG"
echo "阶段 ${STAGE} 用时 $((T1 - T0)) 秒"
