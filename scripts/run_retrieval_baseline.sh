#!/bin/bash
# SeDPO 检索基线 v2：BM25 / SBERT
#
# 用法: bash run_retrieval_baseline.sh <task_name> <model_path> <mode> <model_tag> <cluster>
#   mode: bm25 | sbert
#
# 流程：先用检索器从 cluster 候选池检索 top-3 示例 → 再用 LLM 做 3-shot 推理 → 算指标
# 与 SeDPO 的差别只在「示例从哪来」，其余（LLM、shot 数、评测集）完全一致 → 对照公平。
set -e

TASK=$1
MODEL=$2
MODE=$3
TAG=$4
CLUSTER=$5
REPO=/root/autodl-tmp/SeDPO_NIPS25
OUT_DIR=$REPO/my_data_qwen/baseline_${TAG}/${TASK}_${MODE}
PORT=24701

cd $REPO
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/se_dpo
export PYTHONPATH=$REPO:$PYTHONPATH
export IF_PREFER_LOSS=0
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p $OUT_DIR
RETR_FILE=$OUT_DIR/${TASK}_${MODE}_prompts.json

echo "[$(date +%H:%M:%S)] 检索基线开始: task=$TASK mode=$MODE model=$TAG"
START=$(date +%s)

# ---- 第 1 步：检索 top-3 示例 ----
if [ "$MODE" = "bm25" ]; then
    echo "[$(date +%H:%M:%S)]   BM25 检索中..."
    python scripts/retrieve_bm25.py \
        train_clusters=$CLUSTER \
        task_name=$TASK \
        cache_dir=$REPO/cache \
        prompt_pool_path=$REPO/my_data_qwen/prompt_pool \
        out_file=$RETR_FILE \
        prompt_setup_type=qa n_docs=3 \
        hydra.run.dir=$OUT_DIR 2>&1 | tee $OUT_DIR/retrieval.log
elif [ "$MODE" = "sbert" ]; then
    echo "[$(date +%H:%M:%S)]   SBERT 检索中..."
    python scripts/retrieve_sbert.py \
        train_clusters=$CLUSTER \
        task_name=$TASK \
        cache_dir=$REPO/cache \
        prompt_pool_path=$REPO/my_data_qwen/prompt_pool \
        out_file=$RETR_FILE \
        prompt_setup_type=qa n_docs=3 \
        hydra.run.dir=$OUT_DIR 2>&1 | tee $OUT_DIR/retrieval.log
else
    echo "未知 mode: $MODE"; exit 1
fi

# ---- 第 2 步：用检索到的示例做 3-shot 推理 ----
echo "[$(date +%H:%M:%S)]   3-shot 推理中..."
accelerate launch --num_processes 1 --main_process_port $PORT \
    scripts/inference.py \
    prompt_file=$RETR_FILE \
    task_name=$TASK \
    output_file=$OUT_DIR/${TASK}_${MODE}_preds.json \
    res_file=$OUT_DIR/eval_res.txt \
    batch_size=1 \
    shot_num=3 \
    random_sample=False \
    random_seed=42 \
    model_name=$MODEL \
    prompt_pool_path=$REPO/my_data_qwen/prompt_pool \
    train_clusters=$CLUSTER \
    cache_dir=$REPO/cache \
    hydra.run.dir=$OUT_DIR 2>&1 | tee $OUT_DIR/inference.log

END=$(date +%s)
echo "[$(date +%H:%M:%S)] 检索基线结束: task=$TASK mode=$MODE 耗时 $((END - START)) 秒"
echo "--- 评测结果 ---"
cat $OUT_DIR/eval_res.txt
