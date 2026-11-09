#!/bin/bash
# 找出每个分片的有效前缀（相对 ModelScope 源），并把文件截断到有效长度，
# 之后 dl_fast.py 会在此基础上续传剩余部分。
set -u
LOG=/root/autodl-tmp/prefix_scan.log
: > $LOG

scan() {
  REPO="$1"; DIR="$2"; F="$3"
  echo "=== $DIR/$F ===" >> $LOG
  /root/miniconda3/bin/python /root/autodl-tmp/find_prefix.py --repo "$REPO" --file "$DIR/$F" >> $LOG 2>&1
}

# llama3-8b
for i in 1 2 3 4; do
  f=$(printf "model-%05d-of-00004.safetensors" $i)
  scan "NousResearch/Meta-Llama-3-8B" /root/autodl-tmp/models/llama3-8b "$f"
done
# qwen2.5-7b
for i in 1 2 3 4; do
  f=$(printf "model-%05d-of-00004.safetensors" $i)
  scan "Qwen/Qwen2.5-7B" /root/autodl-tmp/models/qwen2.5-7b "$f"
done
echo "PREFIX_SCAN_DONE" >> $LOG
