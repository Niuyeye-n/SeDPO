#!/bin/bash
# 诊断脚本：模型完整性 / 产物体积 / 磁盘 / 依赖
echo "=========== 1. llama3-8b 目录完整性 ==========="
ls -la /root/autodl-tmp/models/llama3-8b/
echo
echo "--- index 里声明的分片 vs 实际存在的 ---"
python3 - <<'PY'
import json, os
d = "/root/autodl-tmp/models/llama3-8b"
idx = os.path.join(d, "model.safetensors.index.json")
if os.path.exists(idx):
    wm = json.load(open(idx))["weight_map"]
    shards = sorted(set(wm.values()))
    print("index 声明分片:", shards)
    for s in shards:
        p = os.path.join(d, s)
        print(f"  {s}: exists={os.path.exists(p)} size={os.path.getsize(p) if os.path.exists(p) else 0}")
else:
    print("!! 没有 index.json")
for f in ["config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json", "special_tokens_map.json"]:
    p = os.path.join(d, f)
    print(f"  {f}: exists={os.path.exists(p)}")
if os.path.exists(os.path.join(d, "config.json")):
    c = json.load(open(os.path.join(d, "config.json")))
    print("  model_type:", c.get("model_type"), "| torch_dtype:", c.get("torch_dtype"),
          "| vocab:", c.get("vocab_size"), "| max_pos:", c.get("max_position_embeddings"))
PY

echo
echo "=========== 2. 已产出文件体积 ==========="
for f in /root/autodl-tmp/SeDPO_NIPS25/my_data_qwen/scored_qwen/close_qa/*.json* \
         /root/autodl-tmp/SeDPO_NIPS25/my_data_qwen/find_random_qwen/close_qa/*.json; do
  [ -e "$f" ] && ls -la "$f"
done

echo
echo "=========== 3. 磁盘 ==========="
df -h /root/autodl-tmp / | tail -3
echo
du -sh /root/autodl-tmp/SeDPO_NIPS25/my_data_qwen/* 2>/dev/null

echo
echo "=========== 4. 依赖可用性 ==========="
/root/autodl-tmp/envs/se_dpo/bin/python - <<'PY'
mods = ["torch","transformers","datasets","sklearn","faiss","rank_bm25","sentence_transformers",
        "scipy","numpy","hydra","accelerate","jsonlines"]
import importlib
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"  {m}: OK {getattr(mod,'__version__','')}")
    except Exception as e:
        print(f"  {m}: MISSING ({type(e).__name__})")
import torch
print("  torch.cuda:", torch.cuda.is_available(), "| gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
PY

echo
echo "=========== 5. 评测集规模 ==========="
ls /root/autodl-tmp/SeDPO_NIPS25/cache/saved_datasets/ 2>/dev/null
