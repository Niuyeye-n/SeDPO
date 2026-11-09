#!/bin/bash
# ==============================================================================
# SeDPO 正式实验总编排（只跑 SeDPO 方法，隔离目录版）
#
# 2026-11-05 用户指示：只跑 SeDPO，论文其他方法（zeroshot/random3/BM25/SBERT/Se2）
#                      一律不跑，已完成的基线结果保留作参考。
#
# 与服务器上另一并发会话的隔离：
#   本脚本用 sedpo_ 前缀的所有目录（experiment_sedpo_{tag} / scored_sedpo_{tag} /
#   find_random_sedpo_{tag} / prompt_pool_sedpo），产物互不覆盖。
#   prompt_pool 是双方原本的争抢点（都会重写），现已完全隔离。
#
# 规模（实测后确定，理由见实验记录）：
#   打分吞吐 19.4 实例/s（7B fp16, RTX4090）；推理 4.85 样本/s；训练 bsz=8
#   全量规模需 166h ≫ 预算 → 候选池保持全量，只限制打分 query 数 n_score
#   L 保持论文原值 400/200/50（且已修掉官方 len(data_list) 截断 L 的缺陷）
#
# 用法: bash /root/autodl-tmp/run_sedpo.sh
# ==============================================================================
set -u

REPO=/root/autodl-tmp/SeDPO_NIPS25
RESULTS=$REPO/results_new_tasks_20261105
mkdir -p $RESULTS

# ---- 规模参数 ----
NS_ARC=${NS_ARC:-300}
NS_OBQ=${NS_OBQ:-300}
NS_MUL=${NS_MUL:-300}
SCORE_BSZ=${SCORE_BSZ:-12}
TRAIN_BSZ=${TRAIN_BSZ:-8}

QWEN=/root/autodl-tmp/models/qwen2.5-7b
LLAMA=/root/autodl-tmp/models/llama3-8b

echo "############################################################"
echo "# SeDPO 正式实验开始 $(date +%Y-%m-%d_%H:%M:%S)"
echo "# n_score: arc_c=$NS_ARC obqa=$NS_OBQ mul=$NS_MUL | score_bsz=$SCORE_BSZ"
echo "# 候选池全量不变；L=400/200/50（论文原值）"
echo "# 仅跑 SeDPO 方法（其他基线按用户要求跳过）"
echo "############################################################"

# 清理某组训练产生的 checkpoint（保留 best_valid，infer 需要）
cleanup_ckpt() {
    local md=$1
    [ -d "$md" ] || return 0
    local n sz
    n=$(ls "$md"/dpr_biencoder.[0-9]* 2>/dev/null | wc -l)
    if [ "$n" -gt 0 ]; then
        sz=$(du -cm "$md"/dpr_biencoder.[0-9]* 2>/dev/null | tail -1 | cut -f1)
        rm -f "$md"/dpr_biencoder.[0-9]*
        echo "[$(date +%H:%M:%S)]   清理 $n 个 epoch checkpoint（释放 ${sz} MB）"
    fi
}

run_cluster() {
    MODEL_TAG=$1
    MODEL_PATH=$2
    CLUSTER=$3

    echo ""
    echo "================================================================"
    echo "[$(date +%H:%M:%S)] 集群=$CLUSTER 模型=$MODEL_TAG"
    echo "================================================================"

    /root/miniconda3/bin/python /root/autodl-tmp/gen_sedpo.py \
        --model_tag "$MODEL_TAG" --model_path "$MODEL_PATH" --cluster "$CLUSTER" \
        --ns_arc $NS_ARC --ns_obq $NS_OBQ --ns_mul $NS_MUL \
        --score_bsz $SCORE_BSZ || { echo "[错误] 生成脚本失败"; return 1; }

    EXP=$REPO/my_data_qwen/experiment_sedpo_${MODEL_TAG}/$CLUSTER
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

    # ---- SeDPO step1: score → train → infer ----
    echo "[$(date +%H:%M:%S)]   SeDPO score"
    # 重试 5 次：scorer 自带 buffer 断点续传（*.jsontmp_cuda.bin），
    # 且输出 json 存在时会 SKIP 整个 step → 重启几乎无损
    SCORE_RC=1
    for try in 1 2 3 4 5; do
        T0=$(date +%s)
        ( cd $REPO && bash $EXP/score.sh ) >> $LOGDIR/sedpo_score.log 2>&1
        SCORE_RC=$?
        T1=$(date +%s)
        SCORE_SEC=$((T1-T0))
        echo "[$(date +%H:%M:%S)]   → score rc=$SCORE_RC 耗时 $SCORE_SEC 秒（第 $try 次）"
        [ $SCORE_RC -eq 0 ] && break
        echo "[$(date +%H:%M:%S)]   score 未成功，30 秒后续跑（可从 buffer 断点续传）"
        sleep 30
    done
    if [ $SCORE_RC -ne 0 ]; then
        echo "[错误] score 五次尝试均失败，见 $LOGDIR/sedpo_score.log"
        tail -25 $LOGDIR/sedpo_score.log | tr "\r" "\n" | tail -12
        return 1
    fi

    # 训练 batch size（bsz=8 安全；bsz=16 会 OOM）
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
    # 训练产物很大（每 epoch checkpoint 含 optimizer state）→ 立刻清理，只留 best_valid
    cleanup_ckpt "$EXP/model"

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
        SP=$(ls $EXP/preds_for_*/${TASK}_prompts3_retrieverse2_preds_0.json 2>/dev/null | head -1)
        if [ -n "$SP" ]; then
            python $REPO/20261105_compute_metrics.py \
                --preds "$SP" --task $TASK \
                --out_json $OUTD/metrics_sedpo.json --out_text $OUTD/metrics_sedpo.txt 2>&1 | tail -4
        else
            echo "[警告] 未找到 $TASK 的 SeDPO preds"
        fi

        [ -f $EXP/train_dense_encoder.log ] && cp $EXP/train_dense_encoder.log $OUTD/
        find $EXP -name "eval_res*.txt" -exec cp {} $OUTD/ \; 2>/dev/null || true
        [ -n "$SP" ] && cp "$SP" $OUTD/ 2>/dev/null || true
        cp $LOGDIR/*.log $OUTD/ 2>/dev/null || true

        NS=$( [ "$TASK" = arc_c ] && echo $NS_ARC || { [ "$TASK" = openbookqa ] && echo $NS_OBQ || echo $NS_MUL; } )
        LV=$( [ "$TASK" = arc_c ] && echo 400 || { [ "$TASK" = openbookqa ] && echo 200 || echo 50; } )
        cat > $OUTD/runtime_${CLUSTER}.json <<EOF
{
  "task": "$TASK",
  "model_tag": "$MODEL_TAG",
  "cluster": "$CLUSTER",
  "method": "SeDPO",
  "n_score": $NS,
  "finder_L": $LV,
  "pool_size_full": true,
  "score_bsz": $SCORE_BSZ,
  "train_bsz": $TRAIN_BSZ,
  "sedpo_score_sec": $SCORE_SEC,
  "sedpo_train_sec": $TRAIN_SEC,
  "sedpo_infer_sec": $INFER_SEC,
  "cluster_total_sec": $((SCORE_SEC + TRAIN_SEC + INFER_SEC)),
  "log_dir": "$LOGDIR"
}
EOF
        echo "--- $TASK / $MODEL_TAG 指标 ---"
        [ -f $OUTD/metrics_sedpo.txt ] && echo "[sedpo] $(head -c 300 $OUTD/metrics_sedpo.txt | tr '\n' ' ')"
    done

    # infer 已完成 → model 目录（含稠密索引）可删
    rm -rf "$EXP/model"
    echo "[$(date +%H:%M:%S)] 集群 $CLUSTER × $MODEL_TAG 完成"
}

# ============ 执行顺序：ARC-C → OpenBookRC/MultiRC，两模型交替 ============
run_cluster qwen  $QWEN  close_qa
run_cluster llama $LLAMA close_qa
run_cluster qwen  $QWEN  reading_new
run_cluster llama $LLAMA reading_new

echo ""
echo "############################################################"
echo "# SeDPO 正式实验全部结束 $(date +%Y-%m-%d_%H:%M:%S)"
echo "############################################################"
touch /root/autodl-tmp/EXPERIMENT_DONE
