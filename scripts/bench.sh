#!/bin/bash
# 精确测速：对指定任务/模型/批量，打分固定数量实例，输出墙钟速度。
#
# 用法: bash /root/autodl-tmp/bench.sh <task> <model_path> <bsz> <tag> [n_queries] [L]
set -eu

TASK=$1
MODEL_PATH=$2
BSZ=$3
TAG=$4
NQ=${5:-60}
L=${6:-20}

REPO=/root/autodl-tmp/SeDPO_NIPS25
BENCH=$REPO/my_data_bench
mkdir -p $BENCH/{find_random,prompt_pool,scored,exp}

source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/se_dpo
export PYTHONPATH=$REPO:${PYTHONPATH:-}
export FINDER_L=$L
export IF_PREFER_LOSS=2
export PEFT_BETA=0.1
export DIV_REJECT=1
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false

cd $REPO

POOL=$BENCH/prompt_pool/${TASK}_prompts.json
SAMPLES=$BENCH/find_random/${TASK}_samples.json
TRAIN=$BENCH/scored/${TASK}_${TAG}_train.json
VALID=$BENCH/scored/${TASK}_${TAG}_valid.json
LOG=$BENCH/scorer_${TASK}_${TAG}_b${BSZ}.log

rm -f $TRAIN $VALID ${TRAIN}tmp_*.bin

python scripts/find_random_step1.py \
    output_path=$SAMPLES task_name=$TASK +ds_size=$NQ L=$L \
    prompt_pool_path=$POOL cache_dir=$PWD/cache \
    hydra.run.dir=$BENCH/exp > /dev/null 2>&1

N_INST=$(/root/miniconda3/bin/python -c "
import json; d=json.load(open('$SAMPLES')); print(len(d)*len(d[0]['step_1_ctxs']))")
echo "[BENCH $TASK/$TAG bsz=$BSZ] 打分实例数 = $N_INST"

( for i in $(seq 1 2000); do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> $BENCH/gpu_${TASK}_${TAG}_b${BSZ}.csv
    sleep 3
  done ) &
MONPID=$!

T0=$(date +%s)
accelerate launch --num_processes 1 --main_process_port 24579 \
    scripts/scorer.py \
    example_file=$SAMPLES output_train_file=$TRAIN output_valid_file=$VALID \
    batch_size=$BSZ task_name=$TASK model_name=$MODEL_PATH \
    prompt_pool_path=$POOL cache_dir=$PWD/cache \
    hydra.run.dir=$BENCH/exp > $LOG 2>&1
T1=$(date +%s)
kill $MONPID 2>/dev/null || true

ELAPSED=$((T1-T0))
PEAK=$(sort -n $BENCH/gpu_${TASK}_${TAG}_b${BSZ}.csv 2>/dev/null | tail -1)

# 从 tqdm 的最终状态解析纯打分耗时（排除模型加载与过滤阶段）
/root/miniconda3/bin/python - "$N_INST" "$ELAPSED" "$PEAK" "$TASK" "$TAG" "$BSZ" <<'PYEOF'
import re, sys
n_inst, elapsed, peak, task, tag, bsz = sys.argv[1:7]
n_inst = int(n_inst)
elapsed = int(elapsed)
log = f"/root/autodl-tmp/SeDPO_NIPS25/my_data_bench/scorer_{task}_{tag}_b{bsz}.log"
score_el = None
try:
    txt = open(log, errors="ignore").read()
    # tqdm 的 Scoring 进度条最后一帧：[MM:SS<MM:SS,  X.XXit/s]
    frames = re.findall(r"Scoring:\s+\d+%\|[^|]*\|\s*(\d+)/(\d+)\s*\[(\d+):(\d+)<(\d+):(\d+),\s*([\d.]+)it/s", txt)
    if frames:
        f = frames[-1]
        done, total = int(f[0]), int(f[1])
        m, s = int(f[2]), int(f[3])
        score_el = m * 60 + s
        rate = float(f[6])
    else:
        done = total = 0
        rate = None
except Exception as e:
    print("parse err", e)

print("=" * 66)
print(f"[BENCH {task}/{tag} bsz={bsz}]")
print(f"  实例数        = {n_inst}")
print(f"  总墙钟(含加载) = {elapsed}s")
if score_el:
    print(f"  纯打分耗时    = {score_el}s")
    print(f"  实测速度      = {n_inst/score_el:.2f} 实例/秒 (tqdm 报告 {rate} it/s)")
else:
    print("  [警告] 未能解析打分耗时")
print(f"  显存峰值      = {peak} MiB / 24564 MiB")
print("=" * 66)
with open(f"/root/autodl-tmp/SeDPO_NIPS25/my_data_bench/result_{task}_{tag}_b{bsz}.txt", "w") as f:
    f.write(f"task={task}\ntag={tag}\nbsz={bsz}\nn_inst={n_inst}\n"
            f"total_elapsed={elapsed}\nscore_elapsed={score_el}\n"
            f"inst_per_s={n_inst/score_el if score_el else None}\npeak_mem_mib={peak}\n")
PYEOF
