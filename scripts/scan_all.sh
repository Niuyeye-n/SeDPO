#!/bin/bash
# 对所有分片（含 qwen 的 part0 副本）做窗口扫描，标定有效前缀。
set -u
LOG=/root/autodl-tmp/scan_all.log
: > $LOG
PY=/root/miniconda3/bin/python
SCAN=/root/autodl-tmp/scan_prefix.py

run_one() {
  REPO="$1"; F="$2"; TAG="$3"
  echo "### $TAG $F" >> $LOG
  $PY $SCAN --repo "$REPO" --file "$F" --window-mb 16 --workers 4 >> $LOG 2>&1
}

LL=NousResearch/Meta-Llama-3-8B
QW=Qwen/Qwen2.5-7B
LDIR=/root/autodl-tmp/models/llama3-8b
QDIR=/root/autodl-tmp/models/qwen2.5-7b

# 阶段1：llama 4 个分片（顺序，稳妥）
for i in 1 2 3 4; do
  f=$(printf "model-%05d-of-00004.safetensors" $i)
  run_one "$LL" "$LDIR/$f" "llama"
done

# 阶段2：qwen 主分片
for i in 1 2 3 4; do
  f=$(printf "model-%05d-of-00004.safetensors" $i)
  run_one "$QW" "$QDIR/$f" "qwen"
done

# 阶段3：qwen part0 副本
for i in 1 2 3 4; do
  f=$(printf "model-%05d-of-00004.safetensors.part0" $i)
  [ -f "$QDIR/$f" ] && run_one "$QW" "$QDIR/$f" "qwen-part0"
done

echo "SCAN_ALL_DONE" >> $LOG
