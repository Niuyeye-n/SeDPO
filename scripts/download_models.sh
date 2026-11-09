#!/bin/bash
# 两个模型的完整下载编排：复用有效前缀续传 → 分块下载 → 官方 sha256 校验。
# 磁盘策略：块文件下载完立刻 append 并删除，峰值额外占用 ≈ 线程数×块大小。
set -u
PY=/root/miniconda3/bin/python
DL=/root/autodl-tmp/dl_fast.py

SHARDS=$(seq -f "model-%05g-of-00004.safetensors" 1 4 | tr "\n" "," | sed "s/,$//")

echo "=============== [1/2] Qwen2.5-7B ==============="
$PY $DL --repo Qwen/Qwen2.5-7B \
    --dest /root/autodl-tmp/models/qwen2.5-7b \
    --shards "$SHARDS" \
    --extra "config.json,generation_config.json,merges.txt,tokenizer.json,tokenizer_config.json,vocab.json,model.safetensors.index.json" \
    --threads 8 --chunk-mb 128

echo "=============== [2/2] Meta-Llama-3-8B ==============="
$PY $DL --repo NousResearch/Meta-Llama-3-8B \
    --dest /root/autodl-tmp/models/llama3-8b \
    --shards "$SHARDS" \
    --extra "config.json,generation_config.json,special_tokens_map.json,tokenizer.json,tokenizer_config.json,model.safetensors.index.json" \
    --threads 8 --chunk-mb 128

echo "=============== 汇总 ==============="
du -sh /root/autodl-tmp/models/qwen2.5-7b /root/autodl-tmp/models/llama3-8b
ls -l /root/autodl-tmp/models/qwen2.5-7b/*.safetensors /root/autodl-tmp/models/llama3-8b/*.safetensors
df -h /root/autodl-tmp | tail -1
echo "ALL_MODELS_DONE"
