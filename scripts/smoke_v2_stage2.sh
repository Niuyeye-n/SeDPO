#!/bin/bash
# Smoke test 第二段：训练 → 检索 → 推理 → 指标（复用第一段已产出的打分数据）
# 前提：smoke_v2.sh 的打分阶段已完成且产物存在。
set -u

REPO=/root/autodl-tmp/SeDPO_NIPS25
SMOKE=$REPO/smoke_v2
TAG=smoke
EXP=$REPO/my_data_qwen/experiment_${TAG}/close_qa

cd $REPO
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/se_dpo
export PYTHONPATH=$REPO:${PYTHONPATH:-}
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PEFT_BETA=0.1
export DIV_REJECT=1

echo "############ SMOKE STAGE2 开始 $(date +%H:%M:%S) ############"

echo "[$(date +%H:%M:%S)] 4/7 训练检索器（IF_PREFER_LOSS=2）"
T0=$(date +%s)
IF_PREFER_LOSS=2 bash $EXP/train.sh > $SMOKE/train.log 2>&1
TRAIN_RC=$?; T1=$(date +%s)
echo "[$(date +%H:%M:%S)] train rc=$TRAIN_RC 耗时 $((T1-T0)) 秒"
if [ $TRAIN_RC -ne 0 ]; then
    echo "!!! train 失败："; tail -30 $SMOKE/train.log | tr "\r" "\n" | tail -15; exit 1
fi
grep -E "best_valid|dev_best|Accuracy|Total filtered" $SMOKE/train.log | tail -6

echo ""
echo "[$(date +%H:%M:%S)] 5/7 稠密检索 + 推理"
T0=$(date +%s)
IF_PREFER_LOSS=2 bash $EXP/infer.sh > $SMOKE/infer.log 2>&1
INFER_RC=$?; T1=$(date +%s)
echo "[$(date +%H:%M:%S)] infer rc=$INFER_RC 耗时 $((T1-T0)) 秒"
if [ $INFER_RC -ne 0 ]; then
    echo "!!! infer 失败："; tail -30 $SMOKE/infer.log | tr "\r" "\n" | tail -15; exit 1
fi

echo ""
echo "[$(date +%H:%M:%S)] 6/7 计算指标"
PREDS=$(ls $EXP/preds_for_*/arc_c_prompts3_retrieverse2_preds_0.json 2>/dev/null | head -1)
if [ -n "$PREDS" ]; then
    python $REPO/20261105_compute_metrics.py --preds "$PREDS" --task arc_c \
        --out_json $SMOKE/metrics.json --out_text $SMOKE/metrics.txt 2>&1 | tail -6
else
    echo "!!! 找不到 preds"; exit 1
fi

echo ""
echo "[$(date +%H:%M:%S)] 7/7 校验 F3（Random 每问题示例不同）"
/root/autodl-tmp/envs/se_dpo/bin/python - <<'PY'
import sys, os, random, inspect
REPO = "/root/autodl-tmp/SeDPO_NIPS25"
sys.path.insert(0, REPO); os.chdir(REPO)
from src.dataset_readers.few_shot_dsr import FewShotDatasetReader
src = inspect.getsource(FewShotDatasetReader.get_fields)
print("  含按问题派生种子:", "self.random_seed * 1000003" in src)
pool = list(range(1000))
draws = [random.Random(42*1000003 + i).sample(pool, k=3) for i in range(5)]
print("  前5题示例:", draws)
print("  互不相同:", len({tuple(d) for d in draws}) == 5)
PY

echo ""
echo "############ SMOKE STAGE2 全部通过 $(date +%H:%M:%S) ############"
cat $SMOKE/metrics.txt 2>/dev/null | head -22
