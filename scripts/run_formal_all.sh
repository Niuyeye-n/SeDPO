#!/bin/bash
# 正式实验总编排：三个数据集 × 两个模型 × (基线 zeroshot + 基线 random3 + SeDPO)
#
# 规模说明（2026-11-05，实测后确定）：
#   实测打分速度 19.4 实例/s（7B fp16, RTX4090, batch_size=1）
#   实测推理速度 4.85 样本/s（修复 device_map="auto" 导致的 CPU offload 后）
#   实测训练速度 bsz=8 时 81 样本 × 6 epoch = 7 秒
#   全量规模（候选池全量 × 论文 L）需 166 小时 ≫ 12 小时预算 → 必须降规模
#
# 降规模方式：**候选池保持全量不变，只限制「被打分的 query 数」n_score**
#   - 候选池 = prompt_pool = 检索语料，与基线 random3 的抽样来源完全一致 → 对照公平
#   - n_score = 被打分的 query 数，直接决定打分开销
#   - L = finder_L 保持论文原值，保证排序信号强度不降
#
# 用法: bash /root/autodl-tmp/run_formal_all.sh
set -u

REPO=/root/autodl-tmp/SeDPO_NIPS25
RESULTS=$REPO/results_new_tasks_20261105
mkdir -p $RESULTS

# ---- 规模参数（实测后选定）----
# n_score：每个数据集被打分的 query 数；候选池始终全量
NS_ARC=150
NS_OBQ=150
NS_MUL=150
# L：finder_L，沿用论文/官方原值（arc_c=400, openbookqa=200, multirc=50）
L_ARC=400
L_OBQ=200
L_MUL=50
# 训练 batch size（bsz=8 安全，bsz=16 会 OOM）
TRAIN_BSZ=8

QWEN=/root/autodl-tmp/models/qwen2.5-7b
LLAMA=/root/autodl-tmp/models/llama3-8b

echo "############################################################"
echo "# 正式实验开始 $(date +%Y-%m-%d_%H:%M:%S)"
echo "# 规模: n_score(arc_c=$NS_ARC, obqa=$NS_OBQ, mul=$NS_MUL)  L(arc_c=$L_ARC, obqa=$L_OBQ, mul=$L_MUL)"
echo "# 候选池全量不变（arc_c=1117, obqa=4957, mul=10000），保证与基线对照公平"
echo "############################################################"

run_cluster() {
    MODEL_TAG=$1
    MODEL_PATH=$2
    CLUSTER=$3

    echo ""
    echo "================================================================"
    echo "[$(date +%H:%M:%S)] 集群=$CLUSTER 模型=$MODEL_TAG"
    echo "================================================================"

    # 生成该模型专属脚本（隔离不同模型的中间产物）
    /root/miniconda3/bin/python /root/autodl-tmp/gen_formal.py \
        --model_tag "$MODEL_TAG" --model_path "$MODEL_PATH" --cluster "$CLUSTER" \
        --ns_arc $NS_ARC --ns_obq $NS_OBQ --ns_mul $NS_MUL

    EXP=$REPO/my_data_qwen/experiment_${MODEL_TAG}/$CLUSTER
    TS=$(date +%Y%m%d_%H%M%S)
    LOGDIR=$RESULTS/logs/${CLUSTER}_${MODEL_TAG}_${TS}
    mkdir -p $LOGDIR

    case $CLUSTER in
        close_qa)    TASKS="arc_c" ;;
        reading_new) TASKS="openbookqa multirc" ;;
    esac

    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate /root/autodl-tmp/envs/se_dpo
    export PYTHONPATH=$REPO:${PYTHONPATH:-}
    export HYDRA_FULL_ERROR=1
    export TOKENIZERS_PARALLELISM=false
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export IF_PREFER_LOSS=2
    export PEFT_BETA=0.1
    export DIV_REJECT=1

    # ---- 基线 ----
    for TASK in $TASKS; do
        for MODE in zeroshot random3; do
            echo "[$(date +%H:%M:%S)]   基线 $TASK / $MODE"
            T0=$(date +%s)
            bash /root/autodl-tmp/run_baseline.sh $TASK $MODEL_PATH $MODE $MODEL_TAG \
                > $LOGDIR/baseline_${TASK}_${MODE}.log 2>&1
            RC=$?
            T1=$(date +%s)
            echo "[$(date +%H:%M:%S)]   → rc=$RC 耗时 $((T1-T0)) 秒"
            if [ $RC -ne 0 ]; then
                echo "[错误] 基线 $TASK/$MODE 失败，见 $LOGDIR/baseline_${TASK}_${MODE}.log"
                tail -20 $LOGDIR/baseline_${TASK}_${MODE}.log | tr "\r" "\n" | tail -10
            fi
        done
    done

    # ---- SeDPO: score → train → infer ----
    echo "[$(date +%H:%M:%S)]   SeDPO score"
    # 重试 3 次：scorer 自带 buffer 断点续传（*.jsontmp_cuda.bin），
    # 被中断后重跑会从已完成处继续，不会浪费已打的分。
    SCORE_RC=1
    for try in 1 2 3; do
        T0=$(date +%s)
        ( cd $REPO && bash $EXP/score.sh ) >> $LOGDIR/sedpo_score.log 2>&1
        SCORE_RC=$?
        T1=$(date +%s)
        SCORE_SEC=$((T1-T0))
        echo "[$(date +%H:%M:%S)]   → score rc=$SCORE_RC 耗时 $SCORE_SEC 秒（第 $try 次尝试）"
        [ $SCORE_RC -eq 0 ] && break
        echo "[$(date +%H:%M:%S)]   score 未成功，10 秒后续跑（可从 buffer 断点续传）"
        sleep 10
    done
    if [ $SCORE_RC -ne 0 ]; then
        echo "[错误] score 三次尝试均失败，见 $LOGDIR/sedpo_score.log"
        tail -25 $LOGDIR/sedpo_score.log | tr "\r" "\n" | tail -12
        return 1
    fi

    # 训练 batch size 调优（bsz=8 安全；bsz=1 慢 23 倍）
    sed -i "s/train.batch_size=1 /train.batch_size=$TRAIN_BSZ /" $EXP/train.sh

    echo "[$(date +%H:%M:%S)]   SeDPO train"
    T0=$(date +%s)
    ( cd $REPO && bash $EXP/train.sh ) > $LOGDIR/sedpo_train.log 2>&1
    RC=$?
    T1=$(date +%s)
    TRAIN_SEC=$((T1-T0))
    echo "[$(date +%H:%M:%S)]   → rc=$RC 耗时 $TRAIN_SEC 秒"
    if [ $RC -ne 0 ]; then
        echo "[错误] train 失败，见 $LOGDIR/sedpo_train.log"
        tail -25 $LOGDIR/sedpo_train.log | tr "\r" "\n" | tail -12
        return 1
    fi

    echo "[$(date +%H:%M:%S)]   SeDPO infer"
    T0=$(date +%s)
    ( cd $REPO && bash $EXP/infer.sh ) > $LOGDIR/sedpo_infer.log 2>&1
    RC=$?
    T1=$(date +%s)
    INFER_SEC=$((T1-T0))
    echo "[$(date +%H:%M:%S)]   → rc=$RC 耗时 $INFER_SEC 秒"
    if [ $RC -ne 0 ]; then
        echo "[错误] infer 失败，见 $LOGDIR/sedpo_infer.log"
        tail -25 $LOGDIR/sedpo_infer.log | tr "\r" "\n" | tail -12
        return 1
    fi

    # ---- 指标 ----
    for TASK in $TASKS; do
        OUTD=$RESULTS/${TASK}_${MODEL_TAG}
        mkdir -p $OUTD
        python $REPO/20261105_compute_metrics.py \
            --preds $REPO/my_data_qwen/baseline_${MODEL_TAG}/${TASK}_zeroshot/${TASK}_zeroshot_preds.json \
            --task $TASK --out_json $OUTD/metrics_zeroshot.json --out_text $OUTD/metrics_zeroshot.txt 2>&1 | tail -3
        python $REPO/20261105_compute_metrics.py \
            --preds $REPO/my_data_qwen/baseline_${MODEL_TAG}/${TASK}_random3/${TASK}_random3_preds.json \
            --task $TASK --out_json $OUTD/metrics_random3.json --out_text $OUTD/metrics_random3.txt 2>&1 | tail -3
        SP=$(ls $EXP/preds_for_*/${TASK}_prompts3_retrieverse2_preds_0.json 2>/dev/null | head -1)
        if [ -n "$SP" ]; then
            python $REPO/20261105_compute_metrics.py \
                --preds $SP --task $TASK --out_json $OUTD/metrics_sedpo.json --out_text $OUTD/metrics_sedpo.txt 2>&1 | tail -3
        fi

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
  "n_score": "$( [ "$TASK" = arc_c ] && echo $NS_ARC || { [ "$TASK" = openbookqa ] && echo $NS_OBQ || echo $NS_MUL; } )",
  "finder_L": "$( [ "$TASK" = arc_c ] && echo 400 || { [ "$TASK" = openbookqa ] && echo 200 || echo 50; } )",
  "pool_size_full": true,
  "sedpo_score_sec": $SCORE_SEC,
  "sedpo_train_sec": $TRAIN_SEC,
  "sedpo_infer_sec": $INFER_SEC,
  "cluster_total_sec": $((SCORE_SEC + TRAIN_SEC + INFER_SEC)),
  "log_dir": "$LOGDIR"
}
EOF
        echo "--- $TASK / $MODEL_TAG 指标 ---"
        for m in zeroshot random3 sedpo; do
            [ -f $OUTD/metrics_$m.txt ] && echo "[$m] $(head -c 300 $OUTD/metrics_$m.txt | tr '\n' ' ')"
        done
    done
    echo "[$(date +%H:%M:%S)] 集群 $CLUSTER × $MODEL_TAG 完成"
}

# ============ 执行顺序：ARC-C → OpenBookRC/MultiRC，两模型交替 ============
# 先 ARC-C 打通全链路（最小数据集），再 reading_new 集群（含两数据集）

run_cluster qwen  $QWEN  close_qa
run_cluster llama $LLAMA close_qa
run_cluster qwen  $QWEN  reading_new
run_cluster llama $LLAMA reading_new

echo ""
echo "############################################################"
echo "# 正式实验全部结束 $(date +%Y-%m-%d_%H:%M:%S)"
echo "############################################################"
