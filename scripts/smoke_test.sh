#!/bin/bash
# 冒烟测速：独立目录 my_data_smoke，ARC-C，FINDER_L=20，ds_size=50
# 只跑 find_random_step1 + scorer.py step1，测「每秒打多少分」与显存占用。
#
# 用法: bash /root/autodl-tmp/smoke_test.sh <model_path> <tag>
set -e

MODEL_PATH=$1
TAG=$2
REPO=/root/autodl-tmp/SeDPO_NIPS25
SMOKE=$REPO/my_data_smoke
TASK=arc_c
DS_SIZE=50
FINDER_L=20

cd $REPO
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/se_dpo
export PYTHONPATH=$REPO:$PYTHONPATH
export FINDER_L=$FINDER_L
export IF_PREFER_LOSS=2
export PEFT_BETA=0.1
export DIV_REJECT=1
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false

mkdir -p $SMOKE/{find_random,prompt_pool,scored,exp}

POOL=$SMOKE/prompt_pool/${TASK}_prompts.json
SAMPLES=$SMOKE/find_random/${TASK}_random_samples_step1.json
SCORED_TRAIN=$SMOKE/scored/${TASK}_scored_train_step1.json
SCORED_VALID=$SMOKE/scored/${TASK}_scored_valid_step1.json

# ---- 1) 构造小规模 prompt pool + step1 采样 ----
echo "[$(date +%H:%M:%S)] [SMOKE $TAG] 开始 find_random_step1 (ds_size=$DS_SIZE L=$FINDER_L)"
T0=$(date +%s)

python scripts/find_random_step1.py \
    output_path=$SAMPLES \
    task_name=$TASK +ds_size=$DS_SIZE L=$FINDER_L \
    prompt_pool_path=$POOL \
    cache_dir=$PWD/cache \
    hydra.run.dir=$SMOKE/exp

T1=$(date +%s)
echo "[$(date +%H:%M:%S)] [SMOKE $TAG] find_random_step1 用时 $((T1-T0)) 秒"
echo "--- pool 条数 / samples 条数 ---"
/root/miniconda3/bin/python -c "
import json
p=json.load(open('$POOL')); s=json.load(open('$SAMPLES'))
print('pool:',len(p),'samples:',len(s))
print('每样本候选数 L =', len(s[0]['step_1_ctxs']))
print('打分实例总数 =', len(s)*len(s[0]['step_1_ctxs']))
"

# ---- 2) 跑 scorer step1 并监控显存 ----
echo "[$(date +%H:%M:%S)] [SMOKE $TAG] 开始 scorer step1"
T2=$(date +%s)

# 后台采样 nvidia-smi 峰值
( for i in $(seq 1 400); do
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits >> $SMOKE/gpu_$TAG.csv
    sleep 2
  done ) &
MONPID=$!

accelerate launch --num_processes 1 --main_process_port 24579 \
    scripts/scorer.py \
    example_file=$SAMPLES \
    output_train_file=$SCORED_TRAIN \
    output_valid_file=$SCORED_VALID \
    batch_size=1 task_name=$TASK \
    model_name=$MODEL_PATH \
    prompt_pool_path=$POOL \
    cache_dir=$PWD/cache \
    hydra.run.dir=$SMOKE/exp 2>&1 | tee $SMOKE/scorer_$TAG.log

T3=$(date +%s)
kill $MONPID 2>/dev/null || true

echo "[$(date +%H:%M:%S)] [SMOKE $TAG] scorer step1 用时 $((T3-T2)) 秒"

# ---- 3) 汇总测速 ----
/root/miniconda3/bin/python - <<'PYEOF'
import json, os, re
SMOKE = os.environ.get("SMOKE_DIR", "/root/autodl-tmp/SeDPO_NIPS25/my_data_smoke")
TAG = os.environ.get("TAG", "qwen")
tag = TAG
samples = json.load(open(f"{SMOKE}/find_random/arc_c_random_samples_step1.json"))
n_inst = len(samples) * len(samples[0]["step_1_ctxs"])
log = f"{SMOKE}/scorer_{tag}.log"

# 从日志里找 tqdm 的 it/s
speeds = []
for line in open(log, errors="ignore"):
    m = re.findall(r"(\d+\.\d+)it/s", line)
    if m:
        speeds.append(float(m[-1]))
    m2 = re.findall(r"(\d+:\d+)/(\d+)", line)
    if m2 and "it/s" not in line:
        pass

elapsed = None
for line in open(log, errors="ignore"):
    m = re.search(r"Scoring:.*?(\d+)/(\d+)", line)
    if m:
        pass

gpu = f"{SMOKE}/gpu_{tag}.csv"
mem = []
if os.path.exists(gpu):
    for line in open(gpu):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit():
            mem.append(int(parts[0]))

print("=" * 60)
print(f"[SMOKE-{tag}] 打分实例总数 = {n_inst}")
print(f"[SMOKE-{tag}] 平均速度 = {sum(speeds)/len(speeds):.2f} it/s  (采样 {len(speeds)} 个点)" if speeds else "[SMOKE] 未抓到 it/s")
if mem:
    print(f"[SMOKE-{tag}] 显存峰值 = {max(mem)} MiB / 24564 MiB")
print("=" * 60)
with open(f"{SMOKE}/smoke_result_{tag}.json", "w") as f:
    json.dump({
        "tag": tag,
        "n_instances": n_inst,
        "avg_it_per_s": (sum(speeds)/len(speeds)) if speeds else None,
        "peak_gpu_mem_mib": max(mem) if mem else None,
    }, f, indent=2)
PYEOF

echo "[$(date +%H:%M:%S)] [SMOKE $TAG] 测速完成"
