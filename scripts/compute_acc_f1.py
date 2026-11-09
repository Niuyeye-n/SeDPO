#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从本地 preds 补算每个「模型 × 数据集」的 Acc 与 F1。

- 二分类（MultiRC）：f1 用 sklearn 默认 binary（pos_label=1），与仓库 compute_metrics('f1') 完全一致
- 四分类（ARC-C / OpenBookQA）：f1 用 macro 平均，另给 weighted
- acc 一律 = 预测 == 标签 的比例，与仓库 simple_accuracy 一致
"""
import json
import os
import sys

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

BASE = r"C:/Users/33792/Desktop/论文/results_backup/extracted_20261107/root/autodl-tmp/SeDPO_NIPS25/my_data_qwen"

RUNS = [
    ("Qwen2.5-7B", "ARC-C",      "experiment_sedpo_qwen/close_qa/preds_for_close_qa/arc_c_prompts3_retrieverse2_preds_0.json"),
    ("Qwen2.5-7B", "OpenBookQA", "experiment_sedpo_qwen/reading_new/preds_for_reading_new/openbookqa_prompts3_retrieverse2_preds_0.json"),
    ("Qwen2.5-7B", "MultiRC",    "experiment_sedpo_qwen/reading_new/preds_for_reading_new/multirc_prompts3_retrieverse2_preds_0.json"),
    ("Llama3-8B",  "ARC-C",      "experiment_sedpo_llama/close_qa/preds_for_close_qa/arc_c_prompts3_retrieverse2_preds_0.json"),
    ("Llama3-8B",  "OpenBookQA", "experiment_sedpo_llama/reading_new/preds_for_reading_new/openbookqa_prompts3_retrieverse2_preds_0.json"),
    ("Llama3-8B",  "MultiRC",    "experiment_sedpo_llama/reading_new/preds_for_reading_new/multirc_prompts3_retrieverse2_preds_0.json"),
]


def load_preds(path):
    """兼容单 JSON 与多块拼接 JSON"""
    s = open(path, encoding="utf-8").read()
    dec = json.JSONDecoder()
    i, rows = 0, []
    while i < len(s):
        while i < len(s) and s[i] in " \n\r\t":
            i += 1
        if i >= len(s):
            break
        obj, i = dec.raw_decode(s, i)
        rows.extend(obj)
    return rows


def main():
    hdr = f"{'模型':<12}{'数据集':<12}{'n':>6}{'Acc':>9}{'F1':>9}{'F1(macro)':>11}{'F1(weighted)':>13}"
    print(hdr)
    print("-" * len(hdr))
    out = []
    for model, task, rel in RUNS:
        path = os.path.join(BASE, rel)
        rows = load_preds(path)
        preds = [int(r["pred"]) for r in rows]
        labels = [int(r["label"]) if not isinstance(r["label"], list) else int(r["label"][0]) for r in rows]

        acc = accuracy_score(labels, preds)
        binary = len(set(labels)) <= 2
        if binary:
            f1_bin = f1_score(labels, preds)           # 与仓库一致
            f1_mac = f1_score(labels, preds, average="macro")
            f1_w = f1_score(labels, preds, average="weighted")
            show = f1_bin
        else:
            f1_bin = float("nan")
            f1_mac = f1_score(labels, preds, average="macro")
            f1_w = f1_score(labels, preds, average="weighted")
            show = f1_mac

        print(f"{model:<12}{task:<12}{len(labels):>6}{acc*100:>8.2f}%{show*100:>8.2f}%"
              f"{f1_mac*100:>10.2f}%{f1_w*100:>12.2f}%")

        cm = confusion_matrix(labels, preds)
        out.append({
            "model": model, "task": task, "n": len(labels),
            "acc": round(acc * 100, 4),
            "f1": round(show * 100, 4),
            "f1_macro": round(f1_mac * 100, 4),
            "f1_weighted": round(f1_w * 100, 4),
            "pred_dist": [int(x) for x in cm.sum(axis=0)],
            "confusion": cm.tolist(),
        })

    print()
    print("=== 预测分布 / 混淆矩阵 ===")
    for r in out:
        print(f"{r['model']} × {r['task']}: 预测分布={r['pred_dist']} 混淆矩阵={r['confusion']}")

    dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "20261109_acc_f1_补算结果.json")
    with open(os.path.abspath(dst), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n已保存: {os.path.abspath(dst)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
