#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
analyze_results.py — 2026-11-07
从本地快照（results_backup/extracted_20261107）汇总 SeDPO 全部正式结果。

做三件事：
1. 逐任务核对 preds 条数（确认是完整测试集，不是崩溃残片）
2. 读官方 eval_res（指标由 inference.py 产出，不自己重算）
3. 额外用 preds 反算 acc / 预测分布，作为交叉验证
输出：results_summary.csv + results_summary.md（workspace 根目录）
"""
import csv
import json
import os
import re
from collections import Counter

BASE = r"C:/Users/33792/Desktop/论文/results_backup/extracted_20261107/root/autodl-tmp/SeDPO_NIPS25/my_data_qwen"
OUT_DIR = r"C:/Users/33792/Desktop/论文"

GROUPS = [
    ("qwen2.5-7b", "qwen"),
    ("llama3-8b", "llama"),
]
CLUSTERS = {
    "close_qa": ["arc_c"],
    "reading_new": ["multirc", "openbookqa"],
}

# 官方指标口径：arc_c / openbookqa 用 acc，multirc 用 f1（见 DPR/dpr/utils/tasks.py）
METRIC = {"arc_c": "acc", "openbookqa": "acc", "multirc": "f1"}


def read_eval(path):
    """解析 eval_res 文件 -> {task: {metric: value}}"""
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path, encoding="utf-8"):
        if "task_name:" not in line:
            continue
        m_task = re.search(r"task_name:\s*(\w+)", line)
        m_score = re.search(r"scores:\s*(\{.*\})", line)
        if not (m_task and m_score):
            continue
        task = m_task.group(1)
        try:
            scores = eval(m_score.group(1))  # 内容是 python dict 字面量
        except Exception:
            continue
        out[task] = scores
    return out


def read_train_log(path):
    """抓训练轮数（取配置里的 train_epochs，避免把崩溃重跑的日志算重）与最后 loss"""
    info = {"epochs": 0, "last_loss": None}
    if not os.path.exists(path):
        return info
    txt = open(path, encoding="utf-8", errors="ignore").read()
    m = re.search(r"train_epochs:\s*(\d+)", txt)
    if m:
        info["epochs"] = int(m.group(1))
    steps = re.findall(r"loss=([0-9.]+)", txt)
    if steps:
        info["last_loss"] = steps[-1]
    return info


rows = []
for model_name, tag in GROUPS:
    for cluster, tasks in CLUSTERS.items():
        exp = os.path.join(BASE, f"experiment_sedpo_{tag}", cluster)
        evals = read_eval(os.path.join(exp, f"eval_res_for_{cluster}.txt"))
        tr = read_train_log(os.path.join(exp, "train_dense_encoder.log"))
        for task in tasks:
            preds_path = os.path.join(
                exp, f"preds_for_{cluster}", f"{task}_prompts3_retrieverse2_preds_0.json"
            )
            n, pred_dist, acc_from_preds, shot = 0, {}, None, None
            if os.path.exists(preds_path):
                data = json.load(open(preds_path, encoding="utf-8"))
                n = len(data)
                labels = [d.get("label") for d in data]
                preds = [d.get("pred") for d in data]
                pred_dist = dict(Counter(preds))
                if labels and labels[0] is not None:
                    acc_from_preds = 100.0 * sum(
                        1 for a, b in zip(labels, preds) if a == b
                    ) / len(labels)
                shots = {d.get("n_prompts") for d in data}
                shot = max(int(s) for s in shots if s is not None) if shots else None

            official = evals.get(task, {})
            key = METRIC[task]
            note = ""
            if pred_dist and n:
                top_cls, top_cnt = max(pred_dist.items(), key=lambda kv: kv[1])
                ratio = top_cnt / n
                if ratio > 0.7:
                    note = f"⚠️ 预测偏斜：{ratio*100:.1f}% 都预测为 {top_cls}"
            rows.append(
                {
                    "model": model_name,
                    "cluster": cluster,
                    "task": task,
                    "metric": key,
                    "score": round(official.get(key, float("nan")), 4),
                    "n_samples": n,
                    "shot": shot,
                    "acc_from_preds": round(acc_from_preds, 4) if acc_from_preds is not None else "",
                    "pred_dist": json.dumps(pred_dist, ensure_ascii=False),
                    "train_epochs": tr["epochs"],
                    "train_last_loss": tr["last_loss"] or "",
                    "note": note,
                    "preds_bytes": os.path.getsize(preds_path) if os.path.exists(preds_path) else 0,
                }
            )

# CSV
csv_path = os.path.join(OUT_DIR, "20261107_SeDPO_results.csv")
with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("CSV ->", csv_path)

# Markdown
lines = []
lines.append("# SeDPO 正式实验结果（2026-11-07）\n")
lines.append("检索器：SeDPO (dp2-r1.0-b0.1，IF_PREFER_LOSS=2 / DIV_REJECT=1 / PEFT_BETA=0.1)，3-shot。\n")
lines.append("| 模型 | 数据集 | 任务 | 官方指标 | 数值 | 样本数 | 训练轮数 | 备注 |")
lines.append("|---|---|---|---|---:|---:|---:|---|")
task_cn = {"arc_c": "ARC-C", "openbookqa": "OpenBookQA", "multirc": "MultiRC"}
for r in rows:
    lines.append(
        f"| {r['model']} | {r['cluster']} | {task_cn[r['task']]} | {r['metric']} | "
        f"**{r['score']}** | {r['n_samples']} | {r['train_epochs']} | {r['note'] or '—'} |"
    )

lines.append("\n## 交叉验证（用 preds 反算 acc）\n")
lines.append("| 模型 | 任务 | 官方分数 | preds 反算 acc | 一致? |")
lines.append("|---|---|---:|---:|---|")
for r in rows:
    ok = "—"
    if r["acc_from_preds"] != "":
        if r["metric"] == "acc":
            ok = "✅" if abs(r["acc_from_preds"] - r["score"]) < 1e-3 else "❌"
        else:
            ok = "（该任务官方用 f1，acc 仅参考）"
    lines.append(
        f"| {r['model']} | {task_cn[r['task']]} | {r['score']} | {r['acc_from_preds']} | {ok} |"
    )

md_path = os.path.join(OUT_DIR, "20261107_SeDPO实验结果汇总.md")
lines.append("""
## 口径与完整性说明

- **指标来源**：全部取自 `inference.py` 官方产出的 `eval_res_for_*.txt`，未自行重算。
- **指标定义**：ARC-C / OpenBookQA 为准确率 `acc`；MultiRC 在本仓库定义为 **F1**（二分类 yes/no），
  因此 MultiRC 的 acc（见上表交叉验证列）低于 f1 属正常现象，不要混用。
- **样本数**：ARC-C 1165、MultiRC 4848、OpenBookQA 557，均为完整测试集，不是中断残片。
  上一版 MultiRC 的两份 542 / 516 条文件是 OOM 崩溃残片，已移入 `_trash_partial/` 并重跑。
- **shot**：全部 3-shot。
- **可复现凭证**：每组 `se2_prompts_for_*/` 内保存了实际喂给模型的 prompt 文件。
- **本地快照**：`results_backup/sedpo_results_20261107_1930.tgz`（31 MB，含 6 份 preds + 6 份 eval + prompt + 训练日志）。

## 待补的对照实验

本表只含 SeDPO 检索器的结果。论文需要基线对照（zeroshot / random-3 / BM25 / 未微调 DPR），
这部分尚未跑，后续补跑时沿用同一套 `n_score` 与测试集即可对齐。
""")
open(md_path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("MD  ->", md_path)
print()
print("\n".join(lines))
