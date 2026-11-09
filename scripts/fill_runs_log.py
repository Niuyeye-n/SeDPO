#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把远端算好的指标汇总进 runs_log.csv（按既有表头追加行），并生成本地实验报告。

输入: results_new_tasks_20261105/<dataset>_<model>/metrics_*.json + runtime_*.json
输出:
  - experiment_log/runs_log.csv   （追加）
  - results_new_tasks_20261105/<日期>_实验报告.md

用法（在本地跑）:
  python fill_runs_log.py --results_dir <本地 results 目录> --csv <runs_log.csv 路径>
"""
import argparse
import csv
import json
import os
from datetime import datetime

# 既有表头（保持不变，按行追加）
HEADER = [
    "run_id", "日期", "任务", "打分与推理模型", "方法", "shots",
    "训练集条数", "测试集条数", "标签类别数", "temperature", "top_k", "top_p", "max_tokens",
    "模型精度dtype", "检索器", "lr", "dpo_beta", "epochs", "batch_size",
    "最终train_loss", "最佳checkpoint", "验证avg_rank",
    "准确率accuracy", "F1", "打分耗时", "训练耗时", "推理耗时", "总耗时", "GPU", "备注",
]

MODEL_META = {
    "qwen":  {"path": "/root/autodl-tmp/models/qwen2.5-7b", "label": "Qwen2.5-7B"},
    "llama": {"path": "/root/autodl-tmp/models/llama3-8b", "label": "Meta-Llama-3-8B"},
}

TASK_META = {
    "arc_c":      {"pool": 1117,  "test": 1165, "class_num": 4, "f1_type": "macro"},
    "openbookqa": {"pool": 4957,  "test": 500,  "class_num": 4, "f1_type": "macro"},
    "multirc":    {"pool": 10000, "test": 4848, "class_num": 2, "f1_type": "binary"},
}

METHOD_META = {
    "zeroshot": {"retriever": "无(未训练)", "shots": 0, "prompt_source": "zeroshot.json(无示例)"},
    "random3":  {"retriever": "无(未训练,随机3-shot)", "shots": 3, "prompt_source": "random3.json(prompt_pool随机抽)"},
    "sedpo":    {"retriever": "SeDPO(dp2-r1.0-b0.1)", "shots": 3, "prompt_source": "SeDPO检索top-3"},
}


def fmt(v, nd=2):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def load_json(p):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--scale_note", default="")
    args = ap.parse_args()

    rd = args.results_dir
    today = datetime.now().strftime("%Y-%m-%d")

    rows = []
    # 先扫已有 CSV，跳过占位行（示例/准备）
    if os.path.exists(args.csv):
        with open(args.csv, "r", encoding="utf-8-sig", newline="") as f:
            rd0 = csv.reader(f)
            head = next(rd0, None)
            for r in rd0:
                if not r:
                    continue
                if r[0] and (r[0].startswith("示例") or r[0].startswith("准备")):
                    continue
                rows.append(r)
    else:
        head = HEADER

    new_rows = []
    for task in ["arc_c", "openbookqa", "multirc"]:
        for mtag in ["qwen", "llama"]:
            d = os.path.join(rd, f"{task}_{mtag}")
            if not os.path.isdir(d):
                continue
            mlabel = MODEL_META[mtag]["label"]
            tmeta = TASK_META[task]
            rt = load_json(os.path.join(d, "runtime_close_qa.json")) or \
                load_json(os.path.join(d, "runtime_reading_new.json"))
            if not rt:
                # 逐文件找 runtime
                for fn in os.listdir(d):
                    if fn.startswith("runtime_"):
                        rt = load_json(os.path.join(d, fn))
                        break
            score_sec = rt.get("sedpo_score_sec") if rt else None
            train_sec = rt.get("sedpo_train_sec") if rt else None
            infer_sec = rt.get("sedpo_infer_sec") if rt else None

            # 训练 loss：从 train_dense_encoder.log 提取最后一条 loss
            loss_final = "N/A"
            loss_ref = "N/A"
            tlog = os.path.join(d, "train_dense_encoder.log")
            if os.path.exists(tlog):
                loss_ref = "train_dense_encoder.log"
                last = None
                with open(tlog, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        low = line.lower()
                        if "loss" in low:
                            import re
                            nums = re.findall(r"loss['\"]?\s*[:=]\s*([0-9]*\.?[0-9]+)", low)
                            if nums:
                                last = float(nums[-1])
                if last is not None:
                    loss_final = f"{last:.4f}"

            for meth in ["zeroshot", "random3", "sedpo"]:
                mj = load_json(os.path.join(d, f"metrics_{meth}.json"))
                if not mj:
                    continue
                acc = mj.get("accuracy")
                f1 = mj.get("f1")
                f1_type = mj.get("f1_type", tmeta["f1_type"])
                mm = METHOD_META[meth]
                is_sedpo = (meth == "sedpo")

                if is_sedpo:
                    rmin = ((score_sec or 0) + (train_sec or 0) + (infer_sec or 0)) / 60
                    score_t = f"{score_sec/60:.1f}min" if score_sec else "N/A"
                    train_t = f"{train_sec/60:.1f}min" if train_sec else "N/A"
                    infer_t = f"{infer_sec/60:.1f}min" if infer_sec else "N/A"
                    note = (f"prompt_source={mm['prompt_source']}; f1_type={f1_type}; "
                            f"loss_curve_ref={loss_ref}; 选项似然打分，非采样生成; "
                            f"{args.scale_note}")
                else:
                    rmin = None
                    score_t = train_t = infer_t = "N/A"
                    note = (f"prompt_source={mm['prompt_source']}; f1_type={f1_type}; "
                            f"选项似然打分，非采样生成; LLM 不训练; {args.scale_note}")

                new_rows.append([
                    f"{task}-{mtag}-{meth}", today, task, mlabel, meth, mm["shots"],
                    tmeta["pool"], tmeta["test"], tmeta["class_num"],
                    "N/A", "N/A", "N/A", "N/A",
                    "fp16",
                    mm["retriever"],
                    "3e-05" if is_sedpo else "N/A",
                    "0.1" if is_sedpo else "N/A",
                    "6" if is_sedpo else "N/A",
                    "1",
                    loss_final if is_sedpo else "N/A",
                    "dpr_biencoder.best_valid" if is_sedpo else "N/A",
                    "N/A",
                    fmt(acc), fmt(f1),
                    score_t, train_t, infer_t,
                    f"{rmin:.1f}min" if rmin else "N/A",
                    "RTX4090-24G",
                    note,
                ])

    with open(args.csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for r in rows:
            w.writerow(r)
        for r in new_rows:
            w.writerow(r)
    print(f"[OK] 已写入 {args.csv}：新增 {len(new_rows)} 行，保留原有 {len(rows)} 行")

    # 输出 markdown 表格供报告用
    md = ["| dataset | model | method | shots | acc | F1 | f1_type | loss | runtime(min) |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in new_rows:
        md.append(f"| {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[21]} | {r[22]} | "
                  f"{'binary' if r[2]=='multirc' else 'macro'} | {r[19]} | {r[27]} |")
    out_md = os.path.join(rd, "_runs_log_table.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"[OK] 表格已写入 {out_md}")


if __name__ == "__main__":
    main()
