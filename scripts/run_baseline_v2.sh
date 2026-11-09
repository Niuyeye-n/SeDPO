#!/bin/bash
# SeDPO 基线运行脚本 v2（0-shot / 随机 3-shot），按模型隔离输出目录
#
# 用法: bash run_baseline_v2.sh <task_name> <model_path> <mode> <model_tag>
#   task_name: arc_c | openbookqa | multirc
#   model_path: 7B/8B 模型目录
#   mode: zeroshot | random3
#   model_tag: qwen | llama
#
# 说明：random3 走 random_sample=True 分支，few_shot_dsr 从 prompt_pool 随机抽 3 条，
#       与 SeDPO 检索出的 3 条做对照。抽样池 = 整个 cluster 的候选池（与 SeDPO 检索池一致）。
#       F3 修复后每个问题抽到不同的随机示例（此前所有问题复用同一组）。
set -e

TASK=$1
MODEL=$2
MODE=$3
TAG=$4
REPO=/root/autodl-tmp/SeDPO_NIPS25
OUT_DIR=$REPO/my_data_qwen/baseline_${TAG}/${TASK}_${MODE}
PROMPT_DIR=$REPO/my_data_qwen/prompts_for_eval

cd $REPO
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/se_dpo
export PYTHONPATH=$REPO:$PYTHONPATH
export IF_PREFER_LOSS=0
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p $OUT_DIR

if [ "$MODE" = "zeroshot" ]; then
    PROMPT_FILE=$PROMPT_DIR/${TASK}_zeroshot.json
    SHOT=0
    RANDOM_SAMPLE=False
elif [ "$MODE" = "random3" ]; then
    PROMPT_FILE=$PROMPT_DIR/${TASK}_random3.json
    SHOT=3
    RANDOM_SAMPLE=True
else
    echo "未知 mode: $MODE"; exit 1
fi

# cluster 要与任务对应，否则 random_sample 会从错误的 prompt pool 抽样
case $TASK in
    arc_c)      CLUSTER=close_qa ;;
    openbookqa) CLUSTER=reading_new ;;
    multirc)    CLUSTER=reading_new ;;
    *) echo "未知 task: $TASK"; exit 1 ;;
esac

echo "[$(date +%H:%M:%S)] 基线开始: task=$TASK mode=$MODE shot=$SHOT model=$TAG"
START=$(date +%s)

accelerate launch --num_processes 1 --main_process_port 24579 \
    scripts/inference.py \
    prompt_file=$PROMPT_FILE \
    task_name=$TASK \
    output_file=$OUT_DIR/${TASK}_${MODE}_preds.json \
    res_file=$OUT_DIR/eval_res.txt \
    batch_size=1 \
    shot_num=$SHOT \
    random_sample=$RANDOM_SAMPLE \
    random_seed=42 \
    model_name=$MODEL \
    prompt_pool_path=$REPO/my_data_qwen/prompt_pool \
    train_clusters=$CLUSTER \
    cache_dir=$REPO/cache \
    hydra.run.dir=$OUT_DIR 2>&1 | tee $OUT_DIR/inference.log

END=$(date +%s)
echo "[$(date +%H:%M:%S)] 基线结束: task=$TASK mode=$MODE 耗时 $((END - START)) 秒"
echo "--- 评测结果 ---"
cat $OUT_DIR/eval_res.txt
