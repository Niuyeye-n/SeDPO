set -e

echo "scoring arc_c task..."

python scripts/find_random_step1.py \
	 output_path=$PWD/my_data_llama_table/find_random/close_qa/arc_c_random_samples_step1.json \
	 task_name=arc_c +ds_size=10000 L=50 +n_score=300 \
	 prompt_pool_path=$PWD/my_data_llama_table/prompt_pool/close_qa/arc_c_prompts.json \
	 cache_dir=$PWD/cache \
	 hydra.run.dir=$PWD/my_data_llama_table/experiment/close_qa

accelerate launch --num_processes 1 --main_process_port 22832 \
	 scripts/scorer.py \
	 example_file=$PWD/my_data_llama_table/find_random/close_qa/arc_c_random_samples_step1.json \
	 output_train_file=$PWD/my_data_llama_table/scored/close_qa/arc_c_scored_train_step1.json \
	 output_valid_file=$PWD/my_data_llama_table/scored/close_qa/arc_c_scored_valid_step1.json \
	 batch_size=5 task_name=arc_c \
	 model_name=/root/autodl-tmp/models/llama3-8b \
	 prompt_pool_path=$PWD/my_data_llama_table/prompt_pool/close_qa/arc_c_prompts.json \
	 cache_dir=$PWD/cache \
	 hydra.run.dir=$PWD/my_data_llama_table/experiment/close_qa

python scripts/find_random_step2.py \
	 output_path=$PWD/my_data_llama_table/find_random/close_qa/arc_c_random_samples_step2.json \
	 train_path=$PWD/my_data_llama_table/scored/close_qa/arc_c_scored_train_step1.json \
	 valid_path=$PWD/my_data_llama_table/scored/close_qa/arc_c_scored_valid_step1.json \
	 task_name=arc_c +ds_size=10000 L=50 +n_score=300 \
	 prompt_pool_path=$PWD/my_data_llama_table/prompt_pool/close_qa/arc_c_prompts.json \
	 cache_dir=$PWD/cache \
	 hydra.run.dir=$PWD/my_data_llama_table/experiment/close_qa

accelerate launch --num_processes 1 --main_process_port 22832 \
	 scripts/scorer.py \
	 example_file=$PWD/my_data_llama_table/find_random/close_qa/arc_c_random_samples_step2.json \
	 output_train_file=$PWD/my_data_llama_table/scored/close_qa/arc_c_scored_train_step2.json \
	 output_valid_file=$PWD/my_data_llama_table/scored/close_qa/arc_c_scored_valid_step2.json \
	 batch_size=5 task_name=arc_c \
	 model_name=/root/autodl-tmp/models/llama3-8b \
	 prompt_pool_path=$PWD/my_data_llama_table/prompt_pool/close_qa/arc_c_prompts.json \
	 cache_dir=$PWD/cache \
	 hydra.run.dir=$PWD/my_data_llama_table/experiment/close_qa

python scripts/find_random_step3.py \
	 output_path=$PWD/my_data_llama_table/find_random/close_qa/arc_c_random_samples_step3.json \
	 train_path=$PWD/my_data_llama_table/scored/close_qa/arc_c_scored_train_step2.json \
	 valid_path=$PWD/my_data_llama_table/scored/close_qa/arc_c_scored_valid_step2.json \
	 task_name=arc_c +ds_size=10000 L=50 +n_score=300 \
	 prompt_pool_path=$PWD/my_data_llama_table/prompt_pool/close_qa/arc_c_prompts.json \
	 cache_dir=$PWD/cache \
	 hydra.run.dir=$PWD/my_data_llama_table/experiment/close_qa

accelerate launch --num_processes 1 --main_process_port 22832 \
	 scripts/scorer.py \
	 example_file=$PWD/my_data_llama_table/find_random/close_qa/arc_c_random_samples_step3.json \
	 output_train_file=$PWD/my_data_llama_table/scored/close_qa/arc_c_scored_train_step3.json \
	 output_valid_file=$PWD/my_data_llama_table/scored/close_qa/arc_c_scored_valid_step3.json \
	 batch_size=5 task_name=arc_c \
	 model_name=/root/autodl-tmp/models/llama3-8b \
	 prompt_pool_path=$PWD/my_data_llama_table/prompt_pool/close_qa/arc_c_prompts.json \
	 cache_dir=$PWD/cache \
	 hydra.run.dir=$PWD/my_data_llama_table/experiment/close_qa

echo "merging multi-step data..."

python scripts/merge_data.py \
	 step1_train=$PWD/my_data_llama_table/scored/close_qa/arc_c_scored_train_step1.json \
	 step1_valid=$PWD/my_data_llama_table/scored/close_qa/arc_c_scored_valid_step1.json \
	 step2_train=$PWD/my_data_llama_table/scored/close_qa/arc_c_scored_train_step2.json \
	 step2_valid=$PWD/my_data_llama_table/scored/close_qa/arc_c_scored_valid_step2.json \
	 step3_train=$PWD/my_data_llama_table/scored/close_qa/arc_c_scored_train_step3.json \
	 step3_valid=$PWD/my_data_llama_table/scored/close_qa/arc_c_scored_valid_step3.json \
	 train_output_path=$PWD/my_data_llama_table/scored/close_qa/arc_c_scored_train.json \
	 valid_output_path=$PWD/my_data_llama_table/scored/close_qa/arc_c_scored_valid.json