#!/bin/bash
cd /root/autodl-tmp/SeDPO_NIPS25
echo "===== A. glob 顺序 (决定 pool 拼接顺序) ====="
/root/autodl-tmp/envs/se_dpo/bin/python - <<'PY'
import glob
for cluster in ["close_qa","reading_new"]:
    print(cluster, "->", glob.glob(f"/root/autodl-tmp/SeDPO_NIPS25/my_data_qwen/prompt_pool/{cluster}/*"))
PY
echo "===== B. 依赖可用性 ====="
/root/autodl-tmp/envs/se_dpo/bin/python -c "
import importlib
for m in ['faiss','rank_bm25','sentence_transformers','datasets','transformers','sklearn']:
    try:
        importlib.import_module(m); print('OK  ',m)
    except Exception as e: print('MISS',m,type(e).__name__)
"
echo "===== C. 模型目录 ====="
ls /root/autodl-tmp/models/
echo "===== D. 内存/CPU ====="
free -g | head -2
nproc
echo "===== E. 是否有 sbart/mpnet 缓存 ====="
ls /root/autodl-tmp/SeDPO_NIPS25/cache/ 2>/dev/null | head -20
find /root -maxdepth 6 -iname "*mpnet*" 2>/dev/null | head -5
