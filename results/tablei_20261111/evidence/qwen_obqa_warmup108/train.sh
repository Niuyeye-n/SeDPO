set -euo pipefail
unset PT_MODEL
warmup=$(cat "/root/autodl-tmp/SeDPO_NIPS25/my_data_qwen/experiment/obqa_only_warmup108_seed12345/warmup_steps.txt")
export GRAD_CKPT=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -e

echo "start training the retriever..."

python DPR/train_dense_encoder.py         train_datasets=[se2_dataset] \
	 dev_datasets=[se2_valid_dataset] train=biencoder_se2 \
	 output_dir=$PWD/my_data_qwen/experiment/obqa_only_warmup108_seed12345/model \
	 datasets.train_clusters=obqa_only \
	 datasets.train_file=$PWD/my_data_qwen/scored \
	 datasets.valid_file=$PWD/my_data_qwen/scored \
	 datasets.hard_neg=true datasets.top_k=20 \
	 train.hard_negatives=20         train.batch_size=8 \
	 train.num_train_epochs=6 train.num_save_epochs=2         train.learning_rate=3e-05 train.warmup_steps=${warmup} \
	 datasets.prompt_pool_path=$PWD/my_data_qwen/prompt_pool \
	 datasets.prompt_setup_type=qa datasets.task_setup_type=q \
	 encoder.cache_dir=$PWD/cache \
	 seed=12345 model_file=null hydra.run.dir=$PWD/my_data_qwen/experiment/obqa_only_warmup108_seed12345