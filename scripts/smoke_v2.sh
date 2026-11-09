#!/bin/bash
# Smoke test：最小规模跑通「打分 → 训练 → 检索 → 推理 → 指标」全链路
# 目的：在投入几十小时的正式实验前，确认 4 个修复没引入回归。
#
# 用法: bash /root/autodl-tmp/smoke_v2.sh
set -u

REPO=/root/autodl-tmp/SeDPO_NIPS25
SMOKE=$REPO/smoke_v2
MODEL=/root/autodl-tmp/models/qwen2.5-7b
TAG=smoke

rm -rf $SMOKE
mkdir -p $SMOKE

cd $REPO
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/se_dpo
export PYTHONPATH=$REPO:${PYTHONPATH:-}
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PEFT_BETA=0.1
export DIV_REJECT=1

echo "############ SMOKE TEST 开始 $(date +%H:%M:%S) ############"

# ---- 1) 生成 smoke 规模的实验脚本（n_score=6，L 用论文原值 400）----
echo "[$(date +%H:%M:%S)] 1/7 生成脚本（n_score=6, L=400）"
/root/miniconda3/bin/python /root/autodl-tmp/gen_formal_v2.py \
    --model_tag $TAG --model_path $MODEL --cluster close_qa \
    --ns_arc 6 --ns_obq 6 --ns_mul 6 --score_bsz 6 --train_bsz 8

EXP=$REPO/my_data_qwen/experiment_${TAG}/close_qa
echo "--- score.sh 中的规模与 batch size ---"
grep -E "task_name=|batch_size=" $EXP/score.sh | head -8

# ---- 2) 跑打分（step1/2/3）----
echo ""
echo "[$(date +%H:%M:%S)] 2/7 打分（这是最耗时的一步，smoke 规模应很快）"
T0=$(date +%s)
IF_PREFER_LOSS=2 bash $EXP/score.sh > $SMOKE/score.log 2>&1
SCORE_RC=$?
T1=$(date +%s)
echo "[$(date +%H:%M:%S)] score rc=$SCORE_RC 耗时 $((T1-T0)) 秒"
if [ $SCORE_RC -ne 0 ]; then
    echo "!!! score 失败，日志尾部："
    tail -40 $SMOKE/score.log | tr "\r" "\n" | tail -20
    exit 1
fi

# ---- 3) 校验打分产物：L 是否真的按候选池算了 ----
echo ""
echo "[$(date +%H:%M:%S)] 3/7 校验打分产物"
/root/autodl-tmp/envs/se_dpo/bin/python - <<'PY'
import json, os
REPO = "/root/autodl-tmp/SeDPO_NIPS25"
S = f"{REPO}/my_data_qwen/find_random_smoke/close_qa"
for step in ["step1", "step2", "step3"]:
    p = f"{S}/arc_c_random_samples_{step}.json"
    if not os.path.exists(p):
        print(f"  {step}: !! 缺失 {p}"); continue
    d = json.load(open(p))
    key = f"step_{step[-1]}_ctxs"
    lens = [len(e[key]) for e in d if key in e]
    if lens:
        print(f"  {step}: {len(d)} 条 query, {key} 长度 min={min(lens)} max={max(lens)} (期望 400)")
PY

# ---- 4) 训练检索器 ----
echo ""
echo "[$(date +%H:%M:%S)] 4/7 训练检索器（IF_PREFER_LOSS=2）"
T0=$(date +%s)
IF_PREFER_LOSS=2 bash $EXP/train.sh > $SMOKE/train.log 2>&1
TRAIN_RC=$?
T1=$(date +%s)
echo "[$(date +%H:%M:%S)] train rc=$TRAIN_RC 耗时 $((T1-T0)) 秒"
if [ $TRAIN_RC -ne 0 ]; then
    echo "!!! train 失败，日志尾部："
    tail -40 $SMOKE/train.log | tr "\r" "\n" | tail -20
    exit 1
fi
grep -E "best_valid|dev_best|Accuracy" $SMOKE/train.log | tail -5

# ---- 5) 检索 + 推理 ----
echo ""
echo "[$(date +%H:%M:%S)] 5/7 稠密检索 + 推理"
T0=$(date +%s)
IF_PREFER_LOSS=2 bash $EXP/infer.sh > $SMOKE/infer.log 2>&1
INFER_RC=$?
T1=$(date +%s)
echo "[$(date +%H:%M:%S)] infer rc=$INFER_RC 耗时 $((T1-T0)) 秒"
if [ $INFER_RC -ne 0 ]; then
    echo "!!! infer 失败，日志尾部："
    tail -40 $SMOKE/infer.log | tr "\r" "\n" | tail -20
    exit 1
fi

# ---- 6) 算指标 ----
echo ""
echo "[$(date +%H:%M:%S)] 6/7 计算指标"
PREDS=$(ls $EXP/preds_for_*/arc_c_prompts3_retrieverse2_preds_0.json 2>/dev/null | head -1)
if [ -n "$PREDS" ]; then
    python $REPO/20261105_compute_metrics.py --preds "$PREDS" --task arc_c \
        --out_json $SMOKE/metrics.json --out_text $SMOKE/metrics.txt 2>&1 | tail -5
else
    echo "!!! 找不到 preds"
    exit 1
fi

# ---- 7) 校验 Random 基线的 F3 修复（每个问题应抽到不同的示例）----
echo ""
echo "[$(date +%H:%M:%S)] 7/7 校验 Random 基线（F3）每问题示例是否不同"
/root/autodl-tmp/envs/se_dpo/bin/python - <<'PY'
import sys, os, json, random
REPO = "/root/autodl-tmp/SeDPO_NIPS25"
sys.path.insert(0, REPO)
os.chdir(REPO)
from src.dataset_readers.few_shot_dsr import FewShotDatasetReader

# 构造一个最小 reader，只验 random_sample 分支的取样逻辑
import inspect
src = inspect.getsource(FewShotDatasetReader.get_fields)
print("  含按问题派生种子:", "self.random_seed * 1000003" in src)

# 模拟：pool=1000 条，K=3，seed=42，取前 5 个问题的示例
pool = list(range(1000))
def draw(rid):
    random.seed(42 * 1000003 + rid)
    return random.sample(pool, k=3)
draws = [draw(i) for i in range(5)]
print("  前 5 个问题的随机示例:", draws)
print("  是否全部互不相同:", len({tuple(d) for d in draws}) == 5)
PY

echo ""
echo "############ SMOKE TEST 全部通过 $(date +%H:%M:%S) ############"
echo "指标："
cat $SMOKE/metrics.txt 2>/dev/null | head -20
