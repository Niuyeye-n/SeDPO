#!/bin/bash
# 大文件分块并行下载（hf-mirror CDN 支持 Range，分块后每块独立重试，不怕断线丢进度）
# 用法: download_bigfile.sh <repo_id> <相对路径> <输出目录> [每文件块数] [全局并发文件数]
set -u
REPO="$1"
REL="$2"
DEST="$3"
PARTS="${4:-4}"
MAXFILES="${5:-4}"
BASE="${HF_BASE:-https://hf-mirror.com/${REPO}/resolve/main}"
URL="${HF_BASE:-${BASE}}/${REL}"
OUT="${DEST}/$(basename "$REL")"

mkdir -p "$DEST"
if [ -f "${OUT}.done" ]; then echo "[SKIP] $REL"; exit 0; fi

SIZE=$(curl -sIL "$URL" | grep -i "^content-length" | tail -1 | tr -d '\r' | awk '{print $2}')
if [ -z "$SIZE" ] || [ "$SIZE" -lt 1000 ]; then
  echo "[FAIL-SIZE] $REL 无法获取大小，改为整文件下载"
  curl -sSL --retry 20 --retry-delay 5 --retry-all-errors -o "$OUT" "$URL" && touch "${OUT}.done" && exit 0
fi
echo "[INFO] $REL 大小 $((SIZE / 1024 / 1024)) MB，分 $PARTS 块，源 $(echo "$URL" | cut -d/ -f3)"

CHUNK=$(( (SIZE + PARTS - 1) / PARTS ))
for i in $(seq 0 $((PARTS - 1))); do
  START=$((i * CHUNK))
  END=$((START + CHUNK - 1))
  [ "$END" -ge "$SIZE" ] && END=$((SIZE - 1))
  (
    for a in $(seq 1 30); do
      HAVE=0
      [ -f "${OUT}.part${i}" ] && HAVE=$(stat -c%s "${OUT}.part${i}")
      WANT=$((END - START + 1))
      if [ "$HAVE" -ge "$WANT" ]; then break; fi
      curl -sSL --retry 5 --retry-delay 5 --retry-all-errors \
           -r $((START + HAVE))-${END} "$URL" >> "${OUT}.part${i}"
      sleep 3
    done
    echo "[PART-DONE] $REL part$i ($(stat -c%s "${OUT}.part${i}" 2>/dev/null) bytes)"
  ) &
  # 控制全局并发
  while [ "$(jobs -r | wc -l)" -ge "$MAXFILES" ]; do sleep 2; done
done
wait

# 校验并合并
TOTAL=0
for i in $(seq 0 $((PARTS - 1))); do
  P="${OUT}.part${i}"
  [ -f "$P" ] && TOTAL=$((TOTAL + $(stat -c%s "$P")))
done
if [ "$TOTAL" -ne "$SIZE" ]; then
  echo "[FAIL-SIZE-MISMATCH] $REL 合并前 $TOTAL != $SIZE"
  exit 1
fi
cat "${OUT}".part* > "$OUT"
rm -f "${OUT}".part*
touch "${OUT}.done"
echo "[OK] $REL $(du -h "$OUT" | cut -f1)"
