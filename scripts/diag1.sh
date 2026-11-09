#!/bin/bash
cd /root/autodl-tmp/SeDPO_NIPS25
echo "===== A. prompt_pool 目录结构 ====="
find my_data_qwen/prompt_pool -type f | head -20
echo "----- 各文件条数 -----"
for f in $(find my_data_qwen/prompt_pool -name "*.json"); do
  n=$(/root/autodl-tmp/envs/se_dpo/bin/python -c "import json;print(len(json.load(open('$f'))))" 2>/dev/null)
  echo "$f -> $n"
done
echo "===== B. get_prompt_files 实现 ====="
grep -n "def get_prompt_files" -A 25 DPR/dpr/utils/tasks.py
echo "===== C. read_data_from_json_files ====="
grep -n "def read_data_from_json_files" -A 20 DPR/dpr/utils/data_utils.py
echo "===== D. scorer.log 时间分布 (filter vs score) ====="
grep -a -n "filter by max length" my_data_qwen/experiment_qwen/close_qa/scorer.log | tr '\r' '\n' | tail -3
echo "===== E. 官方 tasks.py 中三个任务的 finder_L 与 dataset 大小 ====="
grep -n "class Arc_c\|class Openbookqa\|class Multirc" -A 12 DPR/dpr/utils/tasks.py | grep -n "finder_L\|class \|def get_dataset" | head -20
