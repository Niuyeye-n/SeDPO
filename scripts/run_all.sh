#!/bin/bash
# SeDPO 三个新数据集 × 两个 8B 模型 的完整实验编排（按「集群」组织）。
#
# 集群结构（与官方仓库一致）：
#   close_qa    → arc_c                        （单数据集）
#   reading_new → openbookqa + multirc        （共享集群，一次训练两个数据集共用检索器）
# 因此每个 (集群 × 模型) 只需跑一次 score → train → infer，然后分别对集群内每个
# 数据集算指标。
#
# 每个 (集群 × 模型) 的流程：
#   a) 基线 zeroshot（逐数据集）
#   b) 基线 random3（逐数据集）
#   c) SeDPO: score.sh → train.sh → infer.sh
#   d) 立即用 20261105_compute_metrics.py 算准确率+F1 落盘
#
# 用法: bash /root/autodl-tmp/run_all.sh <model_tag> <model_path> <cluster>
#   model_tag: qwen | llama
#   cluster:   close_qa | reading_new
set -u

MODEL_TAG=$1
MODEL_PATH=$2
CLUSTER=$3

REPO=/root/autodl-tmp/SeDPO_NIPS25
RESULTS=$REPO/results_new_tasks_20261105
mkdir -p $RESULTS

source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/se_dpo
export PYTHONPATH=$REPO:$PYTHONPATH
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
# 打分阶段用 DPO 偏好损失；LLM 不训练
export IF_PREFER_LOSS=2
export PEFT_BETA=0.1
export DIV_REJECT=1

case $CLUSTER in
    close_qa)    TASKS="arc_c" ;;
    reading_new) TASKS="openbookqa multirc" ;;
    *) echo "未知 cluster: $CLUSTER"; exit 1 ;;
esac

TS=$(date +%Y%m%d_%H%M%S)
LOGDIR=$RESULTS/logs/${CLUSTER}_${MODEL_TAG}_${TS}
mkdir -p $LOGDIR
echo "$LOGDIR" > $RESULTS/last_logdir_${CLUSTER}_${MODEL_TAG}.txt

echo "############################################################"
echo "# [$(date +%H:%M:%S)] 集群=$CLUSTER 模型=$MODEL_TAG 数据集=[$TASKS]"
echo "# 日志目录: $LOGDIR"
echo "############################################################"

# ---------- a) + b) 基线 ----------
BASE_SEC=""
for TASK in $TASKS; do
    for MODE in zeroshot random3; do
        echo "[$(date +%H:%M:%S)] ===== 基线 $TASK / $MODE ====="
        T0=$(date +%s)
        bash /root/autodl-tmp/run_baseline.sh $TASK $MODEL_PATH $MODE $MODEL_TAG > $LOGDIR/baseline_${TASK}_${MODE}.log 2>&1
        RC=$?
        T1=$(date +%s)
        echo "  $TASK/$MODE 完成 rc=$RC 耗时 $((T1-T0)) 秒"
        if [ $RC -ne 0 ]; then
            echo "[错误] 基线失败，详见 $LOGDIR/baseline_${TASK}_${MODE}.log"
            exit 1
        fi
        BASE_SEC="$BASE_SEC \"baseline_${TASK}_${MODE}_sec\": $((T1-T0)),"
    done
done

# ---------- c) SeDPO ----------
# 为每个模型单独一套配置/中间产物目录，避免两个模型互相覆盖
EXP=$REPO/my_data_qwen/experiment_${MODEL_TAG}/$CLUSTER
mkdir -p $EXP
SRCEXP=$REPO/my_data_qwen/experiment/$CLUSTER

for f in score.sh train.sh infer.sh; do
    sed -e "s#/root/autodl-tmp/models/qwen2.5-7b#$MODEL_PATH#g" \
        -e "s#my_data_qwen/experiment/$CLUSTER#my_data_qwen/experiment_${MODEL_TAG}/$CLUSTER#g" \
        -e "s#my_data_qwen/scored#my_data_qwen/scored_${MODEL_TAG}#g" \
        -e "s#my_data_qwen/find_random#my_data_qwen/find_random_${MODEL_TAG}#g" \
        $SRCEXP/$f > $EXP/$f
    chmod +x $EXP/$f
done

echo "[$(date +%H:%M:%S)] ===== SeDPO score ====="
T0=$(date +%s)
( cd $REPO && bash $EXP/score.sh ) > $LOGDIR/sedpo_score.log 2>&1
RC=$?
T1=$(date +%s)
SCORE_SEC=$((T1-T0))
echo "score 完成 rc=$RC 耗时 $SCORE_SEC 秒"
if [ $RC -ne 0 ]; then
    echo "[错误] score 失败 → $LOGDIR/sedpo_score.log"; exit 1
fi

echo "[$(date +%H:%M:%S)] ===== SeDPO train ====="
T0=$(date +%s)
( cd $REPO && bash $EXP/train.sh ) > $LOGDIR/sedpo_train.log 2>&1
RC=$?
T1=$(date +%s)
TRAIN_SEC=$((T1-T0))
echo "train 完成 rc=$RC 耗时 $TRAIN_SEC 秒"
if [ $RC -ne 0 ]; then
    echo "[错误] train 失败 → $LOGDIR/sedpo_train.log"; exit 1
fi

echo "[$(date +%H:%M:%S)] ===== SeDPO infer ====="
T0=$(date +%s)
( cd $REPO && bash $EXP/infer.sh ) > $LOGDIR/sedpo_infer.log 2>&1
RC=$?
T1=$(date +%s)
INFER_SEC=$((T1-T0))
echo "infer 完成 rc=$RC 耗时 $INFER_SEC 秒"
if [ $RC -ne 0 ]; then
    echo "[错误] infer 失败 → $LOGDIR/sedpo_infer.log"; exit 1
fi

# ---------- d) 指标后处理 ----------
echo "[$(date +%H:%M:%S)] ===== 计算指标 ====="
for TASK in $TASKS; do
    OUTD=$RESULTS/${TASK}_${MODEL_TAG}
    mkdir -p $OUTD

    python $REPO/20261105_compute_metrics.py \
        --preds $REPO/my_data_qwen/baseline_${MODEL_TAG}/${TASK}_zeroshot/${TASK}_zeroshot_preds.json \
        --task $TASK --out_json $OUTD/metrics_zeroshot.json --out_text $OUTD/metrics_zeroshot.txt 2>&1 | tail -4

    python $REPO/20261105_compute_metrics.py \
        --preds $REPO/my_data_qwen/baseline_${MODEL_TAG}/${TASK}_random3/${TASK}_random3_preds.json \
        --task $TASK --out_json $OUTD/metrics_random3.json --out_text $OUTD/metrics_random3.txt 2>&1 | tail -4

    SP=$(ls $EXP/preds_for_*/${TASK}_prompts3_retrieverse2_preds_0.json 2>/dev/null | head -1)
    if [ -n "$SP" ]; then
        python $REPO/20261105_compute_metrics.py \
            --preds $SP --task $TASK --out_json $OUTD/metrics_sedpo.json --out_text $OUTD/metrics_sedpo.txt 2>&1 | tail -4
    else
        echo "[警告] 未找到 $TASK 的 SeDPO preds"
    fi

    # 收集产物
    [ -f $EXP/train_dense_encoder.log ] && cp $EXP/train_dense_encoder.log $OUTD/
    find $EXP -name "eval_res*.txt" -exec cp {} $OUTD/ \; 2>/dev/null || true
    for p in $SP $REPO/my_data_qwen/baseline_${MODEL_TAG}/${TASK}_zeroshot/${TASK}_zeroshot_preds.json \
             $REPO/my_data_qwen/baseline_${MODEL_TAG}/${TASK}_random3/${TASK}_random3_preds.json; do
        [ -f "$p" ] && cp "$p" $OUTD/ 2>/dev/null || true
    done
    cp $LOGDIR/*.log $OUTD/ 2>/dev/null || true

    cat > $OUTD/runtime_${CLUSTER}.json <<EOF
{
  "task": "$TASK",
  "model_tag": "$MODEL_TAG",
  "cluster": "$CLUSTER",
  "sedpo_score_sec": $SCORE_SEC,
  "sedpo_train_sec": $TRAIN_SEC,
  "sedpo_infer_sec": $INFER_SEC,
  "cluster_total_sec": $((SCORE_SEC + TRAIN_SEC + INFER_SEC)),
  "log_dir": "$LOGDIR"
}
EOF
    echo "--- $TASK / $MODEL_TAG 指标 ---"
    for m in zeroshot random3 sedpo; do
        [ -f $OUTD/metrics_$m.txt ] && echo "[$m] $(head -c 400 $OUTD/metrics_$m.txt | tr '\n' ' ')"
    done
done

echo "[$(date +%H:%M:%S)] ===== 集群 $CLUSTER × $MODEL_TAG 全部完成 ====="
