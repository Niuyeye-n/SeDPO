#!/bin/bash
# ==============================================================================
# 基线运行脚本 —— 已按用户要求停用（2026-11-05 18:20）
#
# 用户指示："只跑 SeDPO 方法就行，论文中的其他方法不用跑"
# 因此本脚本直接跳过，不再执行 zeroshot / random3 推理。
#
# 原实现完整保留在同目录 run_baseline_full.sh，需要时可手动调用：
#   bash /root/autodl-tmp/run_baseline_full.sh <task> <model_path> zeroshot|random3 [model_tag]
#
# 已完成的基线结果（保留作参考，不删除）：
#   ARC-C / Qwen2.5-7B  zeroshot 44.21%   random3 47.47%
# ==============================================================================
TASK=${1:-unknown}
MODE=${2:-unknown}
TAG=${3:-unknown}
echo "[$(date +%H:%M:%S)] 基线已按用户要求停用（跳过 task=$TASK mode=$MODE tag=$TAG）"
echo "  仅跑 SeDPO；如需基线请手动执行 run_baseline_full.sh"
exit 0
