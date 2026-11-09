#!/bin/bash
# 用 curl 从 hf-mirror 直取固定文件（绕开 huggingface_hub 的 API 列表请求，避开 429 限流）
# 用法: download_hf_files.sh <repo_id> <dest_dir> <file_list.txt> [并发数]
#   file_list.txt 每行一个仓库内相对路径，空行与 # 开头行忽略
set -u
REPO="$1"
DEST="$2"
LIST="$3"
PAR="${4:-4}"
BASE="${HF_BASE:-https://hf-mirror.com/${REPO}/resolve/main}"

mkdir -p "$DEST"
cd "$DEST" || exit 1

grep -v '^[[:space:]]*$' "$LIST" | grep -v '^[[:space:]]*#' | \
xargs -P "$PAR" -I{} bash -c '
  f="{}"
  if [ -f "$f.done" ]; then echo "[SKIP] $f"; exit 0; fi
  for i in $(seq 1 200); do
    if curl -sSL --retry 8 --retry-delay 5 --retry-all-errors -C - \
         -o "$f" "'"$BASE"'/$f" && [ -s "$f" ]; then
      touch "$f.done"; echo "[OK] $f ($(du -h "$f" | cut -f1))"; exit 0
    fi
    echo "[retry $i] $f"; sleep 8
  done
  echo "[FAIL] $f"
'

echo "ALL_FILES_FINISHED"
ls -lh "$DEST" | tail -20
