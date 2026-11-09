#!/bin/bash
# 生成正式实验脚本（score/train/infer），带规模控制参数。
#
# 关键设计：**候选池保持全量，只限制打分 query 数（n_score）**
#   - 候选池 = prompt_pool = 检索语料，与基线 random3 抽样来源一致 → 对照公平
#   - n_score = 被打分的 query 数，决定打分开销
#   - L = finder_L，每个 query 的候选数，决定排序信号强度（沿用论文原值）
#
# 用法: bash /root/autodl-tmp/gen_formal.sh <model_tag> <model_path> <cluster> <ns_arc> <ns_obq> <ns_mul>
#   cluster=close_qa    → 只用 ns_arc（arc_c 的 L 固定 400）
#   cluster=reading_new → 用 ns_obq（openbookqa, L=200）与 ns_mul（multirc, L=50）
# L 取值固定为论文/官方原值，不随规模改变。
set -eu

MODEL_TAG=$1
MODEL_PATH=$2
CLUSTER=$3
NS_ARC=$4
NS_OBQ=$5
NS_MUL=$6

REPO=/root/autodl-tmp/SeDPO_NIPS25
SRCEXP=$REPO/my_data_qwen/experiment/$CLUSTER
EXP=$REPO/my_data_qwen/experiment_${MODEL_TAG}/$CLUSTER
mkdir -p $EXP

# 各任务的 L 与 n_score（按 task_name 分别替换，避免串改）
case $CLUSTER in
    close_qa)
        SED_ARGS=(
            "s#task_name=arc_c +ds_size=[0-9]* L=[0-9]*#task_name=arc_c +ds_size=10000 L=400 +n_score=${NS_ARC}#g"
        )
        ;;
    reading_new)
        SED_ARGS=(
            "s#task_name=multirc +ds_size=[0-9]* L=[0-9]*#task_name=multirc +ds_size=10000 L=50 +n_score=${NS_MUL}#g"
            "s#task_name=openbookqa +ds_size=[0-9]* L=[0-9]*#task_name=openbookqa +ds_size=10000 L=200 +n_score=${NS_OBQ}#g"
        )
        ;;
    *) echo "未知 cluster: $CLUSTER"; exit 1 ;;
esac

for f in score.sh train.sh infer.sh; do
    sed -e "s#/root/autodl-tmp/models/qwen2.5-7b#$MODEL_PATH#g" \
        -e "s#my_data_qwen/experiment/$CLUSTER#my_data_qwen/experiment_${MODEL_TAG}/$CLUSTER#g" \
        -e "s#my_data_qwen/scored#my_data_qwen/scored_${MODEL_TAG}#g" \
        -e "s#my_data_qwen/find_random#my_data_qwen/find_random_${MODEL_TAG}#g" \
        $(printf -- "-e %s " "${SED_ARGS[@]}") \
        $SRCEXP/$f > $EXP/$f
    chmod +x $EXP/$f
done

echo "[OK] 已生成 $EXP/{score,train,infer}.sh"
echo "     模型=$MODEL_PATH"
echo "--- 各任务规模参数 ---"
grep -oE "task_name=[a-z_]+ \+ds_size=[0-9]+ L=[0-9]+ \+n_score=[0-9]+" $EXP/score.sh | sort -u
