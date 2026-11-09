#!/bin/bash
# =============================================================================
# SeDPO 复现 —— 正式实验总编排 v2（2026-11-05 修复版）
# =============================================================================
# 与 v1 (run_formal_all.sh) 的关键差异：
#   [1] 修掉 4 个已确认缺陷（F1 L截断 / F2 id错配 / F3 Random同种子 / F4 打分显存）
#       → 见 tools/patch_fixes.py + tools/patch_fixes_b.py
#   [2] 打分 batch_size 从 5 提到 12（F4b 显存优化后实测可到 16）
#   [3] 候选池保持全量（arc_c=1117 / openbookqa=4957 / multirc=10000），
#       L 保持论文原值（400/200/50），n_score 放大到预算允许的最大值
#   [4] 补齐全部 6 个方法：Zeroshot / Random3 / BM25 / SBERT / Se2 / SeDPO
#   [5] 增加 Se2 消融（IF_PREFER_LOSS=0，复用同一份打分数据）
#   [6] 不再使用 watchdog 自动重启（避免旧配置复活）
#
# 规模参数（由 tools/plan_scale.py 按实测吞吐推导）：
#   n_score：被打分的 query 数。候选池始终全量不变。
#   L：finder_L，沿用论文/官方原值，保证排序信号强度不降。
#
# 用法: bash /root/autodl-tmp/run_formal_v2.sh
# =============================================================================
set -u

REPO=/root/autodl-tmp/SeDPO_NIPS25
RESULTS=$REPO/results_v2_20261105
mkdir -p $RESULTS

# ---- 规模参数 ----
NS_ARC=${NS_ARC:-300}      # ARC-C 被打分 query 数
NS_OBQ=${NS_OBQ:-300}      # OpenBookRC
NS_MUL=${NS_MUL:-300}      # MultiRC
L_ARC=400                  # 论文/官方原值
L_OBQ=200
L_MUL=50
SCORE_BSZ=${SCORE_BSZ:-12} # 打分 batch size（F4b 后实测安全）
TRAIN_BSZ=${TRAIN_BSZ:-8}

QWEN=/root/autodl-tmp/models/qwen2.5-7b
LLAMA=/root/autodl-tmp/models/llama3-8b

echo "############################################################"
echo "# SeDPO 正式实验 v2 开始 $(date +%Y-%m-%d_%H:%M:%S)"
echo "# n_score: arc_c=$NS_ARC obqa=$NS_OBQ mul=$NS_MUL | L: $L_ARC/$L_OBQ/$L_MUL | score_bsz=$SCORE_BSZ"
echo "# 候选池全量: arc_c=1117 obqa=4957 mul=10000"
echo "############################################################"

# =============================================================================
# 通用：跑一个 (cluster, model) 组合的全部流程
# =============================================================================
run_cluster() {
    MODEL_TAG=$1
    MODEL_PATH=$2
    CLUSTER=$3

    echo ""
    echo "================================================================"
    echo "[$(date +%H:%M:%S)] 集群=$CLUSTER 模型=$MODEL_TAG"
    echo "================================================================"

    /root/miniconda3/bin/python /root/autodl-tmp/gen_formal_v2.py \
        --model_tag "$MODEL_TAG" --model_path "$MODEL_PATH" --cluster "$CLUSTER" \
        --ns_arc $NS_ARC --ns_obq $NS_OBQ --ns_mul $NS_MUL \
        --score_bsz $SCORE_BSZ --train_bsz $TRAIN_BSZ || return 1

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
    export PEFT_BETA=0.1
    export DIV_REJECT=1

    # ---------- 1) 无检索基线：Zeroshot / Random3 ----------
    for TASK in $TASKS; do
        for MODE in zeroshot random3; do
            echo "[$(date +%H:%M:%S)]   基线 $TASK / $MODE"
            T0=$(date +%s)
            bash /root/autodl-tmp/run_baseline.sh $TASK $MODEL_PATH $MODE $MODEL_TAG \
                > $LOGDIR/baseline_${TASK}_${MODE}.log 2>&1
            RC=$?; T1=$(date +%s)
            echo "[$(date +%H:%M:%S)]   → rc=$RC 耗时 $((T1-T0)) 秒"
            [ $RC -ne 0 ] && { echo "[错误] 基线失败"; tail -15 $LOGDIR/baseline_${TASK}_${MODE}.log | tr "\r" "\n" | tail -8; }
        done
    done

    # ---------- 2) 检索基线：BM25 / SBERT（不需要训练检索器）----------
    for TASK in $TASKS; do
        for MODE in bm25 sbert; do
            echo "[$(date +%H:%M:%S)]   检索基线 $TASK / $MODE"
            T0=$(date +%s)
            bash /root/autodl-tmp/run_retrieval_baseline.sh $TASK $MODEL_PATH $MODE $MODEL_TAG $CLUSTER \
                > $LOGDIR/retrieval_${TASK}_${MODE}.log 2>&1
            RC=$?; T1=$(date +%s)
            echo "[$(date +%H:%M:%S)]   → rc=$RC 耗时 $((T1-T0)) 秒"
            [ $RC -ne 0 ] && { echo "[错误] 检索基线失败"; tail -20 $LOGDIR/retrieval_${TASK}_${MODE}.log | tr "\r" "\n" | tail -10; }
        done
    done

    # ---------- 3) SeDPO 主流程：score -> train -> infer ----------
    echo "[$(date +%H:%M:%S)]   SeDPO score (L 修复后为 $L_ARC/$L_OBQ/$L_MUL)"
    SCORE_RC=1; SCORE_SEC=0
    for try in 1 2 3; do
        T0=$(date +%s)
        ( cd $REPO && IF_PREFER_LOSS=2 bash $EXP/score.sh ) >> $LOGDIR/sedpo_score.log 2>&1
        SCORE_RC=$?; T1=$(date +%s); SCORE_SEC=$((T1-T0))
        echo "[$(date +%H:%M:%S)]   → score rc=$SCORE_RC 耗时 $SCORE_SEC 秒（第 $try 次）"
        [ $SCORE_RC -eq 0 ] && break
        sleep 10
    done
    if [ $SCORE_RC -ne 0 ]; then
        echo "[错误] score 失败"; tail -25 $LOGDIR/sedpo_score.log | tr "\r" "\n" | tail -12; return 1
    fi

    echo "[$(date +%H:%M:%S)]   SeDPO train"
    T0=$(date +%s)
    ( cd $REPO && IF_PREFER_LOSS=2 bash $EXP/train.sh ) > $LOGDIR/sedpo_train.log 2>&1
    RC=$?; T1=$(date +%s); TRAIN_SEC=$((T1-T0))
    echo "[$(date +%H:%M:%S)]   → rc=$RC 耗时 $TRAIN_SEC 秒"
    if [ $RC -ne 0 ]; then
        echo "[错误] train 失败"; tail -25 $LOGDIR/sedpo_train.log | tr "\r" "\n" | tail -12; return 1
    fi

    echo "[$(date +%H:%M:%S)]   SeDPO infer"
    T0=$(date +%s)
    ( cd $REPO && IF_PREFER_LOSS=2 bash $EXP/infer.sh ) > $LOGDIR/sedpo_infer.log 2>&1
    RC=$?; T1=$(date +%s); INFER_SEC=$((T1-T0))
    echo "[$(date +%H:%M:%S)]   → rc=$RC 耗时 $INFER_SEC 秒"
    if [ $RC -ne 0 ]; then
        echo "[错误] infer 失败"; tail -25 $LOGDIR/sedpo_infer.log | tr "\r" "\n" | tail -12; return 1
    fi

    # ---------- 4) Se2 消融：复用同一份打分数据，只换训练目标 ----------
    echo "[$(date +%H:%M:%S)]   Se2 消融（IF_PREFER_LOSS=0，复用打分数据）"
    cp -r $EXP/model $EXP/model_se2 2>/dev/null || true
    SE2EXP=$REPO/my_data_qwen/experiment_${MODEL_TAG}/${CLUSTER}_se2
    mkdir -p $SE2EXP
    cp $EXP/train.sh $SE2EXP/train.sh
    cp $EXP/infer.sh $SE2EXP/infer.sh
    sed -i "s|experiment_${MODEL_TAG}/${CLUSTER}|experiment_${MODEL_TAG}/${CLUSTER}_se2|g" $SE2EXP/train.sh $SE2EXP/infer.sh
    T0=$(date +%s)
    ( cd $REPO && IF_PREFER_LOSS=0 bash $SE2EXP/train.sh ) > $LOGDIR/se2_train.log 2>&1
    RC=$?; T1=$(date +%s); SE2_TRAIN_SEC=$((T1-T0))
    echo "[$(date +%H:%M:%S)]   → se2 train rc=$RC 耗时 $SE2_TRAIN_SEC 秒"
    if [ $RC -eq 0 ]; then
        T0=$(date +%s)
        ( cd $REPO && IF_PREFER_LOSS=0 bash $SE2EXP/infer.sh ) > $LOGDIR/se2_infer.log 2>&1
        RC=$?; T1=$(date +%s); SE2_INFER_SEC=$((T1-T0))
        echo "[$(date +%H:%M:%S)]   → se2 infer rc=$RC 耗时 $SE2_INFER_SEC 秒"
    else
        SE2_INFER_SEC=0
        echo "[错误] se2 train 失败"; tail -20 $LOGDIR/se2_train.log | tr "\r" "\n" | tail -10
    fi

    # ---------- 5) 汇总指标 ----------
    for TASK in $TASKS; do
        OUTD=$RESULTS/${TASK}_${MODEL_TAG}
        mkdir -p $OUTD
        NS=$( [ "$TASK" = arc_c ] && echo $NS_ARC || { [ "$TASK" = openbookqa ] && echo $NS_OBQ || echo $NS_MUL; } )
        LL=$( [ "$TASK" = arc_c ] && echo $L_ARC || { [ "$TASK" = openbookqa ] && echo $L_OBQ || echo $L_MUL; } )

        # 逐方法算指标
        for M in zeroshot random3 bm25 sbert se2 sedpo; do
            P=""
            case $M in
                zeroshot) P=$REPO/my_data_qwen/baseline_${MODEL_TAG}/${TASK}_zeroshot/${TASK}_zeroshot_preds.json ;;
                random3)  P=$REPO/my_data_qwen/baseline_${MODEL_TAG}/${TASK}_random3/${TASK}_random3_preds.json ;;
                bm25)     P=$REPO/my_data_qwen/baseline_${MODEL_TAG}/${TASK}_bm25/${TASK}_bm25_preds.json ;;
                sbert)    P=$REPO/my_data_qwen/baseline_${MODEL_TAG}/${TASK}_sbert/${TASK}_sbert_preds.json ;;
                se2)      P=$(ls $SE2EXP/preds_for_*/${TASK}_prompts3_retrieverse2_preds_0.json 2>/dev/null | head -1) ;;
                sedpo)    P=$(ls $EXP/preds_for_*/${TASK}_prompts3_retrieverse2_preds_0.json 2>/dev/null | head -1) ;;
            esac
            if [ -f "$P" ]; then
                python $REPO/20261105_compute_metrics.py --preds "$P" --task $TASK \
                    --out_json $OUTD/metrics_${M}.json --out_text $OUTD/metrics_${M}.txt 2>&1 | tail -2
                cp "$P" $OUTD/preds_${M}.json 2>/dev/null || true
            else
                echo "[WARN] 缺 $M 的 preds: $P"
            fi
        done

        cat > $OUTD/runtime.json <<EOF
{
  "task": "$TASK",
  "model_tag": "$MODEL_TAG",
  "cluster": "$CLUSTER",
  "n_score": $NS,
  "finder_L": $LL,
  "pool_size": "full",
  "score_bsz": $SCORE_BSZ,
  "sedpo_score_sec": $SCORE_SEC,
  "sedpo_train_sec": $TRAIN_SEC,
  "sedpo_infer_sec": $INFER_SEC,
  "se2_train_sec": ${SE2_TRAIN_SEC:-0},
  "se2_infer_sec": ${SE2_INFER_SEC:-0},
  "log_dir": "$LOGDIR"
}
EOF
        cp $LOGDIR/*.log $OUTD/ 2>/dev/null || true
        echo "--- $TASK / $MODEL_TAG 指标 ---"
        for m in zeroshot random3 bm25 sbert se2 sedpo; do
            [ -f $OUTD/metrics_${m}.txt ] && echo "  [$m] $(head -c 220 $OUTD/metrics_${m}.txt | tr '\n' ' ')"
        done
    done
    echo "[$(date +%H:%M:%S)] 集群 $CLUSTER × $MODEL_TAG 完成"
}

# =============================================================================
# 执行顺序：先 ARC-C 打通全链路（最小数据集），再 reading_new
# 每个模型跑两个 cluster，保证两个模型都能拿到完整结果
# =============================================================================
run_cluster qwen  $QWEN  close_qa    || echo "[错误] qwen/close_qa 失败"
run_cluster llama $LLAMA close_qa    || echo "[错误] llama/close_qa 失败"
run_cluster qwen  $QWEN  reading_new || echo "[错误] qwen/reading_new 失败"
run_cluster llama $LLAMA reading_new || echo "[错误] llama/reading_new 失败"

echo ""
echo "############################################################"
echo "# 全部结束 $(date +%Y-%m-%d_%H:%M:%S)  结果目录: $RESULTS"
echo "############################################################"
