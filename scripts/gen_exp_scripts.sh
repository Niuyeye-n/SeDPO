#!/bin/bash
# 生成带可调规模的实验脚本（score/train/infer），供不同模型与不同规模复用。
#
# 用法: bash /root/autodl-tmp/gen_exp_scripts.sh <model_tag> <model_path> <cluster> <L> <ds_size>
#   model_tag: qwen | llama
#   cluster:   close_qa | reading_new
#   L:         finder_L（每个 query 打多少个候选）
#   ds_size:   训练集（候选池 + 打分样本数）上限
#
# 生成目录: my_data_qwen/experiment_<tag>/<cluster>/
# 关键点：prompt_pool 保持全量（检索语料与基线可比），只调 L 与 ds_size 控制打分开销。
set -eu

MODEL_TAG=$1
MODEL_PATH=$2
CLUSTER=$3
L=$4
DS_SIZE=$5

REPO=/root/autodl-tmp/SeDPO_NIPS25
SRCEXP=$REPO/my_data_qwen/experiment/$CLUSTER
EXP=$REPO/my_data_qwen/experiment_${MODEL_TAG}/$CLUSTER
mkdir -p $EXP

for f in score.sh train.sh infer.sh; do
    sed -e "s#/root/autodl-tmp/models/qwen2.5-7b#$MODEL_PATH#g" \
        -e "s#my_data_qwen/experiment/$CLUSTER#my_data_qwen/experiment_${MODEL_TAG}/$CLUSTER#g" \
        -e "s#my_data_qwen/scored#my_data_qwen/scored_${MODEL_TAG}#g" \
        -e "s#my_data_qwen/find_random#my_data_qwen/find_random_${MODEL_TAG}#g" \
        -e "s#+ds_size=[0-9]* L=[0-9]*#+ds_size=$DS_SIZE L=$L#g" \
        $SRCEXP/$f > $EXP/$f
    chmod +x $EXP/$f
done

echo "[OK] 已生成 $EXP/{score,train,infer}.sh  (L=$L ds_size=$DS_SIZE model=$MODEL_PATH)"
grep -oE "\+ds_size=[0-9]+ L=[0-9]+" $EXP/score.sh | sort -u
grep -c "model_name=$MODEL_PATH" $EXP/score.sh | xargs echo "score.sh 中模型路径出现次数:"
