#!/bin/bash
# 全链路连通性检查（极小规模 ds_size=30, L=20）：
#   find_random step1 → scorer step1 → find_random step2 → scorer step2
#   → find_random step3 → scorer step3 → merge_data
#   → train_dense_encoder → generate_dense_embeddings
#   → dense_retriever_beam_score → inference → compute_metrics
#
# 目的：在投入正式实验前，确认训练/检索/推理三个阶段不会因配置或路径问题失败。
# 用法: bash /root/autodl-tmp/pipeline_check.sh <task> <model_path> <tag>
set -eu

TASK=$1
MODEL_PATH=$2
TAG=$3

REPO=/root/autodl-tmp/SeDPO_NIPS25
CK=$REPO/my_data_check
DS=30
L=20

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

case $TASK in
    arc_c)      CLUSTER=close_qa ;;
    openbookqa) CLUSTER=reading_new ;;
    multirc)    CLUSTER=reading_new ;;
esac

mkdir -p $CK/{find_random,scored,exp,model}
mkdir -p $CK/scored/$CLUSTER
mkdir -p $CK/prompt_pool/$CLUSTER

POOL=$CK/prompt_pool/$CLUSTER/${TASK}_prompts.json

step() { echo "=== [$1] $2 ==="; }

# ---------------- step1 ----------------
step 1/11 "find_random_step1"
S1=$CK/find_random/${TASK}_step1.json
rm -f $S1
python scripts/find_random_step1.py output_path=$S1 task_name=$TASK \
    +ds_size=$DS L=$L prompt_pool_path=$POOL cache_dir=$PWD/cache \
    hydra.run.dir=$CK/exp > $CK/log_1_findrandom.txt 2>&1

step 2/11 "scorer step1"
T1=$CK/scored/$CLUSTER/${TASK}_scored_train_step1.json
V1=$CK/scored/$CLUSTER/${TASK}_scored_valid_step1.json
rm -f $T1 $V1 ${T1}tmp_*.bin
accelerate launch --num_processes 1 --main_process_port 24579 scripts/scorer.py \
    example_file=$S1 output_train_file=$T1 output_valid_file=$V1 \
    batch_size=1 task_name=$TASK model_name=$MODEL_PATH \
    prompt_pool_path=$POOL cache_dir=$PWD/cache \
    hydra.run.dir=$CK/exp > $CK/log_2_scorer1.txt 2>&1

# ---------------- step2 ----------------
step 3/11 "find_random_step2"
S2=$CK/find_random/${TASK}_step2.json
rm -f $S2
python scripts/find_random_step2.py output_path=$S2 \
    train_path=$T1 valid_path=$V1 \
    task_name=$TASK +ds_size=$DS L=$L prompt_pool_path=$POOL \
    cache_dir=$PWD/cache hydra.run.dir=$CK/exp > $CK/log_3_findrandom2.txt 2>&1

step 4/11 "scorer step2"
T2=$CK/scored/$CLUSTER/${TASK}_scored_train_step2.json
V2=$CK/scored/$CLUSTER/${TASK}_scored_valid_step2.json
rm -f $T2 $V2 ${T2}tmp_*.bin
accelerate launch --num_processes 1 --main_process_port 24579 scripts/scorer.py \
    example_file=$S2 output_train_file=$T2 output_valid_file=$V2 \
    batch_size=1 task_name=$TASK model_name=$MODEL_PATH \
    prompt_pool_path=$POOL cache_dir=$PWD/cache \
    hydra.run.dir=$CK/exp > $CK/log_4_scorer2.txt 2>&1

# ---------------- step3 ----------------
step 5/11 "find_random_step3"
S3=$CK/find_random/${TASK}_step3.json
rm -f $S3
python scripts/find_random_step3.py output_path=$S3 \
    train_path=$T2 valid_path=$V2 \
    task_name=$TASK +ds_size=$DS L=$L prompt_pool_path=$POOL \
    cache_dir=$PWD/cache hydra.run.dir=$CK/exp > $CK/log_5_findrandom3.txt 2>&1

step 6/11 "scorer step3"
T3=$CK/scored/$CLUSTER/${TASK}_scored_train_step3.json
V3=$CK/scored/$CLUSTER/${TASK}_scored_valid_step3.json
rm -f $T3 $V3 ${T3}tmp_*.bin
accelerate launch --num_processes 1 --main_process_port 24579 scripts/scorer.py \
    example_file=$S3 output_train_file=$T3 output_valid_file=$V3 \
    batch_size=1 task_name=$TASK model_name=$MODEL_PATH \
    prompt_pool_path=$POOL cache_dir=$PWD/cache \
    hydra.run.dir=$CK/exp > $CK/log_6_scorer3.txt 2>&1

# ---------------- merge ----------------
step 7/11 "merge_data"
MT=$CK/scored/$CLUSTER/${TASK}_scored_train.json
MV=$CK/scored/$CLUSTER/${TASK}_scored_valid.json
python scripts/merge_data.py \
    step1_train=$T1 step1_valid=$V1 \
    step2_train=$T2 step2_valid=$V2 \
    step3_train=$T3 step3_valid=$V3 \
    train_output_path=$MT valid_output_path=$MV \
    hydra.run.dir=$CK/exp > $CK/log_7_merge.txt 2>&1

# ---------------- train ----------------
step 8/11 "train_dense_encoder"
rm -rf $CK/model
python DPR/train_dense_encoder.py train_datasets=[se2_dataset] \
    dev_datasets=[se2_valid_dataset] train=biencoder_se2 \
    output_dir=$CK/model \
    datasets.train_clusters=$CLUSTER \
    datasets.train_file=$CK/scored \
    datasets.valid_file=$CK/scored \
    datasets.hard_neg=true datasets.top_k=20 \
    train.hard_negatives=20 train.batch_size=1 \
    train.num_train_epochs=1 train.learning_rate=1e-05 \
    datasets.prompt_pool_path=$CK/prompt_pool \
    datasets.prompt_setup_type=qa datasets.task_setup_type=q \
    encoder.cache_dir=$PWD/cache \
    hydra.run.dir=$CK/exp > $CK/log_8_train.txt 2>&1

# ---------------- embed ----------------
step 9/11 "generate_dense_embeddings"
python DPR/generate_dense_embeddings.py \
    model_file=$CK/model/dpr_biencoder.best_valid \
    ctx_src=dpr_se2 shard_id=0 num_shards=1 \
    out_file=$CK/model/dpr_enc_index \
    ctx_sources.dpr_se2.train_clusters=$CLUSTER \
    ctx_sources.dpr_se2.prompt_pool_path=$CK/prompt_pool \
    ctx_sources.dpr_se2.prompt_setup_type=qa \
    encoder.cache_dir=$PWD/cache \
    hydra.run.dir=$CK/exp > $CK/log_9_embed.txt 2>&1

# ---------------- retrieve ----------------
step 10/11 "dense_retriever_beam_score"
python DPR/dense_retriever_beam_score.py \
    model_file=$CK/model/dpr_biencoder.best_valid \
    qa_dataset=qa_se2 ctx_datatsets=[dpr_se2] \
    encoded_ctx_files=["$CK/model/dpr_enc_index_*"] \
    out_file=$CK/se2_prompts/${TASK}_prompts.json \
    datasets.qa_se2.task_name=$TASK \
    datasets.qa_se2.task_setup_type=q \
    datasets.qa_se2.cache_dir=$PWD/cache beam_size=3 \
    shot_num=3 \
    ctx_sources.dpr_se2.prompt_pool_path=$CK/prompt_pool \
    ctx_sources.dpr_se2.train_clusters=$CLUSTER \
    ctx_sources.dpr_se2.prompt_setup_type=qa \
    encoder.cache_dir=$PWD/cache \
    hydra.run.dir=$CK/exp > $CK/log_10_retrieve.txt 2>&1

# ---------------- inference ----------------
step 11/11 "inference + metrics"
accelerate launch --num_processes 1 --main_process_port 24579 scripts/inference.py \
    prompt_file=$CK/se2_prompts/${TASK}_prompts_beam_score_0.json \
    task_name=$TASK \
    output_file=$CK/preds/${TASK}_preds.json \
    res_file=$CK/eval_res.txt \
    batch_size=1 train_clusters=$CLUSTER \
    model_name=$MODEL_PATH \
    prompt_pool_path=$CK/prompt_pool \
    random_sample=False random_seed=42 \
    cache_dir=$PWD/cache \
    hydra.run.dir=$CK/exp > $CK/log_11_inference.txt 2>&1

echo "=== 评测结果 ==="
cat $CK/eval_res.txt 2>/dev/null || echo "(无 eval_res)"
echo
echo "=== compute_metrics ==="
python $REPO/20261105_compute_metrics.py \
    --preds $CK/preds/${TASK}_preds.json --task $TASK \
    --out_json $CK/metrics.json --out_text $CK/metrics.txt 2>&1 | tail -14
echo
echo "=== 训练 loss（最后 5 条）==="
grep -oE "loss['\"]?\s*[:=]\s*[0-9.]+" $CK/log_8_train.txt 2>/dev/null | tail -5
echo
echo "###### PIPELINE_CHECK_PASSED ######"
