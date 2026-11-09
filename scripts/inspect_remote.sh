#!/bin/bash
cd /root/autodl-tmp/SeDPO_NIPS25
echo "===== A. my_data_qwen/experiment 目录 ====="
ls my_data_qwen/experiment/ 2>/dev/null
echo "----- close_qa/score.sh -----"
cat my_data_qwen/experiment/close_qa/score.sh 2>/dev/null
echo "----- close_qa/train.sh -----"
cat my_data_qwen/experiment/close_qa/train.sh 2>/dev/null
echo "----- close_qa/infer.sh -----"
cat my_data_qwen/experiment/close_qa/infer.sh 2>/dev/null
echo "===== B. my_data/experiment (官方) ====="
ls -R my_data/experiment/ 2>/dev/null | head -40
echo "===== C. tasks.py 中 L/choosen 相关 ====="
grep -n "finder_L\|choosen_num\|class .*Task\|class_num\|num_train\|num_dev" DPR/dpr/utils/tasks.py | head -40
echo "===== D. scorer tmp bin 条目数 ====="
/root/autodl-tmp/envs/se_dpo/bin/python - <<'PY'
import glob,sys
sys.path.insert(0,'/root/autodl-tmp/SeDPO_NIPS25')
from src.utils.cache_util import BufferedJsonReader
tot=0
for p in glob.glob('my_data_qwen/scored_qwen/close_qa/*tmp_*.bin'):
    n=0
    with BufferedJsonReader(p) as f:
        for datas in f.read():
            n+=len(datas)
    print(p, n)
    tot+=n
print('TOTAL scored instances so far (step2):', tot)
PY
