#!/bin/bash
# =============================================================================
# SeDPO 正式实验总编排 v3（2026-11-05）
# =============================================================================
# 范围：只做论文方法 SeDPO（用户 2026-11-05 明确指示），不跑论文其他方法。
#   保留 zeroshot / random3 两个已完成的基线结果作参考（runs_log.csv 已记录实测值）。
#
# 与早期版本的差异（本版是修复 + 放大后的最终版）：
#   [1] 4 个缺陷已修复（F1 L截断 / F2 id错配 / F3 Random同种子 / F4 打分显存）
#   [2] 打分 batch_size 5 → 12（F4b 显存优化后实测可到 16）
#   [3] n_score 从 150 → 600（省下跑其他方法的算力全部投入 SeDPO 自身规模）
#   [4] 候选池始终全量：arc_c=1117 / openbookqa=4957 / multirc=10000
#   [5] L 保持论文原值：arc_c=400 / openbookqa=200 / multirc=50
#   [6] 复用 my_data_qwen/find_random_sedpo_{tag}/ 已有产物（scorer 会跳过已完成步骤）
#   [7] 不使用 watchdog 自动重启（避免旧配置复活）；磁盘清理交给 janitor.sh
#
# 关键：score.sh 里 find_random_step1/2/3 与 scorer 都有「已完成则跳过」逻辑，
#       中断后重跑会自动续跑，不浪费已打的分。
#
# 用法: bash /root/autodl-tmp/run_sedpo_v3.sh
# =============================================================================
set -u

REPO=/root/autodl-tmp/SeDPO_NIPS25
RESULTS=$REPO/results_sedpo_v3
mkdir -p $RESULTS

# ---- 规模参数 ----
# n_score：被打分的 query 数（候选池始终全量）。
#   选 600 的理由：ARC-C 候选池 1117 条 → 覆盖 53.7%；OpenBookRC 4957 → 12.1%；
#   MultiRC 10000 → 6.0%。实测吞吐 24 实例/s（bsz=12），两模型打分约 27 小时，
#   保守估计 54 小时。用户指示「慢慢做，把实验做完」，故取规模优先。
#   脚本支持断点续跑：scorer 自带 buffer（*.jsontmp_cuda.bin）+ 产物存在即跳过，
#   因此即便中途实例重启，已打的分不会浪费。
NS_ARC=${NS_ARC:-300}      # ARC-C 被打分 query 数（2026-11-05 用户确认降档：600→300）
NS_OBQ=${NS_OBQ:-300}      # OpenBookRC
NS_MUL=${NS_MUL:-300}      # MultiRC
L_ARC=400                  # 论文/官方原值
L_OBQ=200
L_MUL=50
SCORE_BSZ=${SCORE_BSZ:-12} # 打分 batch size（F4b 后实测 16 也能跑，12 更稳）
TRAIN_BSZ=${TRAIN_BSZ:-8}  # 训练 batch size（16 会 OOM）

QWEN=/root/autodl-tmp/models/qwen2.5-7b
LLAMA=/root/autodl-tmp/models/llama3-8b

echo "############################################################"
echo "# SeDPO 正式实验 v3 开始 $(date +%Y-%m-%d_%H:%M:%S)"
echo "# 范围：仅 SeDPO（论文方法）"
echo "# n_score: arc_c=$NS_ARC obqa=$NS_OBQ mul=$NS_MUL | L: $L_ARC/$L_OBQ/$L_MUL"
echo "# 候选池全量: arc_c=1117 obqa=4957 mul=10000 | score_bsz=$SCORE_BSZ"
echo "############################################################"

# =============================================================================
# 规模一致性自检：已有产物必须与当前 n_score 一致，否则删除重建。
# 原因：find_random_step1/2/3 在输出文件已存在时会「跳过」，若不检查，
#       换 n_score 后会静默沿用旧规模的采样数据 → 结果与记录不符。
# =============================================================================
check_scale_consistency() {
    TAG=$1; CLUSTER=$2
    FR=$REPO/my_data_qwen/find_random_sedpo_${TAG}/${CLUSTER}
    [ -d "$FR" ] || return 0
    for f in "$FR"/*.json; do
        [ -e "$f" ] || continue
        N=$(/root/autodl-tmp/envs/se_dpo/bin/python -c "
import json,sys
d=json.load(open('$f'))
print(len(d))" 2>/dev/null)
        [ -z "$N" ] && continue
        # 该文件对应的任务期望的 n_score
        case "$f" in
            *arc_c*)      EXP=$NS_ARC ;;
            *openbookqa*) EXP=$NS_OBQ ;;
            *multirc*)    EXP=$NS_MUL ;;
            *)            EXP="" ;;
        esac
        if [ -n "$EXP" ] && [ "$N" != "$EXP" ]; then
            echo "[自检] $f 有 $N 条，与当前 n_score=$EXP 不一致 → 删除重建"
            rm -f "$f"
            # 对应的 scored 产物也要删，否则会复用不匹配的分数
            for sf in "$REPO/my_data_qwen/scored_sedpo_${TAG}/${CLUSTER}"/*$(basename $f .json | sed 's/_random_samples_//')*.json*; do
                [ -e "$sf" ] && { echo "[自检]   连带删除 $sf"; rm -f "$sf"; }
            done
        fi
    done
}

# =============================================================================
run_cluster() {
    MODEL_TAG=$1
    MODEL_PATH=$2
    CLUSTER=$3

    echo ""
    echo "================================================================"
    echo "[$(date +%H:%M:%S)] 集群=$CLUSTER 模型=$MODEL_TAG"
    echo "================================================================"

    # ---- 候选池隔离目录自愈 ----
    # 背景：prompt_pool_sedpo/ 是隔离目录，换实例或清理后可能缺失，
    #       缺失会导致 get_prompt_files() 拿不到候选池 → 后续组直接失败。
    #       这里从主池自动补齐（只复制，不修改内容）。
    POOL_SRC=$REPO/my_data_qwen/prompt_pool/$CLUSTER
    POOL_DST=$REPO/my_data_qwen/prompt_pool_sedpo/$CLUSTER
    if [ -d "$POOL_SRC" ]; then
        mkdir -p "$POOL_DST"
        for pf in "$POOL_SRC"/*.json; do
            [ -e "$pf" ] || continue
            if [ ! -f "$POOL_DST/$(basename $pf)" ]; then
                cp "$pf" "$POOL_DST/" && echo "[自愈] 补齐候选池 $POOL_DST/$(basename $pf)"
            fi
        done
    fi

    check_scale_consistency "$MODEL_TAG" "$CLUSTER"

    # 生成该模型专属脚本（隔离不同模型的中间产物）
    # 注意：gen_formal_v3.py 内部会加 "sedpo_" 前缀，这里传原始 tag 即可
    /root/miniconda3/bin/python /root/autodl-tmp/gen_formal_v3.py \
        --model_tag "$MODEL_TAG" --model_path "$MODEL_PATH" --cluster "$CLUSTER" \
        --ns_arc $NS_ARC --ns_obq $NS_OBQ --ns_mul $NS_MUL \
        --score_bsz $SCORE_BSZ --train_bsz $TRAIN_BSZ || return 1

    EXP=$REPO/my_data_qwen/experiment_sedpo_${MODEL_TAG}/${CLUSTER}
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

    # ---------- SeDPO: score -> train -> infer ----------
    echo "[$(date +%H:%M:%S)]   SeDPO score"
    # 重试 5 次：scorer 自带 buffer 断点续传（*.jsontmp_cuda.bin），
    # 被中断后重跑会从已完成处继续，不会浪费已打的分。
    SCORE_RC=1; SCORE_SEC=0
    for try in 1 2 3 4 5; do
        T0=$(date +%s)
        ( cd $REPO && bash $EXP/score.sh ) >> $LOGDIR/sedpo_score.log 2>&1
        SCORE_RC=$?; T1=$(date +%s); SCORE_SEC=$((T1-T0))
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

    echo "[$(date +%H:%M:%S)]   SeDPO train"
    T0=$(date +%s)
    ( cd $REPO && bash $EXP/train.sh ) > $LOGDIR/sedpo_train.log 2>&1
    RC=$?; T1=$(date +%s); TRAIN_SEC=$((T1-T0))
    echo "[$(date +%H:%M:%S)]   → rc=$RC 耗时 $TRAIN_SEC 秒"
    if [ $RC -ne 0 ]; then
        echo "[错误] train 失败，见 $LOGDIR/sedpo_train.log"
        tail -25 $LOGDIR/sedpo_train.log | tr "\r" "\n" | tail -12
        return 1
    fi
    # 训练产物很大（每 epoch checkpoint 含 optimizer state）→ 立刻清理，只留 best_valid
    if [ -d "$EXP/model" ]; then
        rm -f "$EXP/model"/dpr_biencoder.[0-9]* 2>/dev/null || true
    fi

    echo "[$(date +%H:%M:%S)]   SeDPO infer"
    T0=$(date +%s)
    ( cd $REPO && bash $EXP/infer.sh ) > $LOGDIR/sedpo_infer.log 2>&1
    RC=$?; T1=$(date +%s); INFER_SEC=$((T1-T0))
    echo "[$(date +%H:%M:%S)]   → rc=$RC 耗时 $INFER_SEC 秒"
    if [ $RC -ne 0 ]; then
        echo "[错误] infer 失败，见 $LOGDIR/sedpo_infer.log"
        tail -25 $LOGDIR/sedpo_infer.log | tr "\r" "\n" | tail -12
        return 1
    fi

    # ---------- 汇总指标 ----------
    for TASK in $TASKS; do
        OUTD=$RESULTS/${TASK}_${MODEL_TAG}
        mkdir -p $OUTD
        NS=$( [ "$TASK" = arc_c ] && echo $NS_ARC || { [ "$TASK" = openbookqa ] && echo $NS_OBQ || echo $NS_MUL; } )
        LV=$( [ "$TASK" = arc_c ] && echo $L_ARC || { [ "$TASK" = openbookqa ] && echo $L_OBQ || echo $L_MUL; } )

        SP=$(ls $EXP/preds_for_*/${TASK}_prompts3_retrieverse2_preds_0.json 2>/dev/null | head -1)
        if [ -n "$SP" ]; then
            python $REPO/20261105_compute_metrics.py --preds "$SP" --task $TASK \
                --out_json $OUTD/metrics_sedpo.json --out_text $OUTD/metrics_sedpo.txt 2>&1 | tail -4
            cp "$SP" $OUTD/preds_sedpo.json 2>/dev/null || true
        else
            echo "[警告] 未找到 $TASK 的 SeDPO preds"
        fi

        [ -f $EXP/train_dense_encoder.log ] && cp $EXP/train_dense_encoder.log $OUTD/ 2>/dev/null || true
        find $EXP -name "eval_res*.txt" -exec cp {} $OUTD/ \; 2>/dev/null || true
        cp $LOGDIR/*.log $OUTD/ 2>/dev/null || true

        cat > $OUTD/runtime.json <<EOF
{
  "task": "$TASK",
  "model_tag": "$MODEL_TAG",
  "cluster": "$CLUSTER",
  "method": "SeDPO",
  "n_score": $NS,
  "finder_L": $LV,
  "pool_size": "full",
  "score_bsz": $SCORE_BSZ,
  "train_bsz": $TRAIN_BSZ,
  "sedpo_score_sec": $SCORE_SEC,
  "sedpo_train_sec": $TRAIN_SEC,
  "sedpo_infer_sec": $INFER_SEC,
  "log_dir": "$LOGDIR"
}
EOF
        echo "--- $TASK / $MODEL_TAG 指标 ---"
        [ -f $OUTD/metrics_sedpo.txt ] && echo "  [sedpo] $(head -c 320 $OUTD/metrics_sedpo.txt | tr '\n' ' ')"
    done

    # infer 已完成 → model 目录（含稠密索引，约 4-7GB）可删
    rm -rf "$EXP/model"
    echo "[$(date +%H:%M:%S)] 集群 $CLUSTER × $MODEL_TAG 完成"
}

# =============================================================================
# 执行顺序：ARC-C 先行（最小数据集，打通全链路），再 reading_new（两数据集）
# 每完成一组立即出结果，可随时查看进度
# =============================================================================
run_cluster qwen  $QWEN  close_qa    || echo "[错误] qwen/close_qa 失败"
run_cluster llama $LLAMA close_qa    || echo "[错误] llama/close_qa 失败"
run_cluster qwen  $QWEN  reading_new || echo "[错误] qwen/reading_new 失败"
run_cluster llama $LLAMA reading_new || echo "[错误] llama/reading_new 失败"

echo ""
echo "############################################################"
echo "# 全部结束 $(date +%Y-%m-%d_%H:%M:%S)  结果目录: $RESULTS"
echo "############################################################"
