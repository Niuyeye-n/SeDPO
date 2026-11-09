#!/bin/bash
# =============================================================================
# 实验结果快照（2026-11-07）
# =============================================================================
# 目的：实验结果（检索到的 3-shot prompts、最终 preds、eval_res 指标）绝不能再丢。
#
# 为什么需要：11-06~11-07 训练三次被旧 janitor 删目录而中断；此外 AutoDL 实例
#   一旦释放，/root/autodl-tmp 全盘消失。所以把「结果类文件」单独打包到系统盘
#   /root/results_backup/，并由本地每小时回传一份。
#
# 只收结果，不收权重（权重可从 best_valid 重现，结果不能）：
#   - */preds_for_*/**          最终预测（含答案与指标统计所需字段）
#   - */eval_res_*.txt          官方评测指标输出
#   - */se2_prompts_for_*/**    实际喂给模型的 3-shot prompt（可复现性凭证）
#   - results_sedpo_v3/**       编排日志与汇总
#   - 各组 *.log                训练/推理日志
# =============================================================================
set -u
REPO=/root/autodl-tmp/SeDPO_NIPS25
BK=/root/results_backup
LOG=$BK/backup.log
mkdir -p "$BK"

TS=$(date +%Y%m%d_%H%M%S)
LIST=/tmp/_reslist_$$.txt
: > "$LIST"

for t in qwen llama; do
    for c in close_qa reading_new; do
        exp="$REPO/my_data_qwen/experiment_sedpo_$t/$c"
        [ -d "$exp" ] || continue
        # 结果类文件（preds / eval 指标 / prompts）
        find "$exp" -type f \( -name "*_preds_*.json" -o -name "eval_res_*.txt" \
             -o -name "*_prompts*.json" \) >> "$LIST" 2>/dev/null
        # 日志单独收，便于回溯
        find "$exp" -maxdepth 1 -type f -name "*.log" >> "$LIST" 2>/dev/null
    done
done
find "$REPO/results_sedpo_v3" -type f >> "$LIST" 2>/dev/null

n_files=$(wc -l < "$LIST")
# 注意：grep -c 无匹配时会打印 0 并返回 1，直接接 `|| echo 0` 会多打一个 0
n_preds=$(grep -c "_preds_" "$LIST" 2>/dev/null) || true
n_preds=${n_preds:-0}

if [ "$n_files" -eq 0 ]; then
    echo "[$(date '+%F %T')] 暂无可备份文件" >> "$LOG"
    rm -f "$LIST"
    exit 0
fi

# 内容指纹：文件路径+大小+修改时间，避免结果没变化时反复写盘
FP=$(while read -r f; do printf "%s %s %s\n" "$(stat -c '%s %Y' "$f" 2>/dev/null)" "$f"; done < "$LIST" | sort | md5sum | cut -d" " -f1)
LAST_FP=$(cat "$BK/.last_fp" 2>/dev/null || echo none)
if [ "$FP" = "$LAST_FP" ]; then
    echo "[$(date '+%F %T')] 无变化，跳过（preds=$n_preds files=$n_files）" >> "$LOG"
    rm -f "$LIST"
    exit 0
fi

tar -czf "$BK/snap_$TS.tgz" -T "$LIST" 2>/dev/null
sz=$(du -sm "$BK/snap_$TS.tgz" 2>/dev/null | cut -f1)
cp -f "$BK/snap_$TS.tgz" "$BK/latest.tgz"
echo "$FP" > "$BK/.last_fp"

# 只保留最近 12 个快照（约 2 小时），latest.tgz 始终是最新
ls -1t "$BK"/snap_*.tgz 2>/dev/null | tail -n +13 | xargs -r rm -f

echo "[$(date '+%F %T')] 快照 snap_$TS.tgz (${sz} MB, preds=$n_preds, files=$n_files)" >> "$LOG"
rm -f "$LIST"
