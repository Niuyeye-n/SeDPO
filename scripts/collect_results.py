#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SeDPO 实验产物收集 + 指标汇总 + CSV 填表 + 报告生成（服务器端）

用法（在服务器上跑）:
    python collect_results.py --repo /root/autodl-tmp/SeDPO_NIPS25 \
        --out_dir /root/autodl-tmp/SeDPO_NIPS25/results_new_tasks_20261105 \
        --csv /root/autodl-tmp/runs_log_rows.csv

产物布局（本地与服务器一致）:
    results_new_tasks_20261105/<dataset>_<model>_<method>/
        preds json / eval_res*.txt / train_dense_encoder.log / metrics_{json,txt} / runtime*.json
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys

TASK_MAP = {
    "arc_c": {"dataset": "ARC-C", "cluster": "close_qa"},
    "openbookqa": {"dataset": "OpenBookRC", "cluster": "reading_new"},
    "multirc": {"dataset": "MultiRC", "cluster": "reading_new"},
}
MODEL_TAG = {"qwen": "Qwen2.5-7B", "llama": "Llama-3-8B"}


def sh(cmd):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return p.stdout.strip()


def parse_metrics(txt_path):
    """从 compute_metrics 的文本输出里抽 accuracy / f1"""
    if not os.path.exists(txt_path):
        return None
    txt = open(txt_path, encoding="utf-8", errors="ignore").read()
    acc = f1 = None
    f1_type = ""
    m = re.search(r"accuracy\s*[:=]\s*([0-9.]+)", txt, re.I)
    if m:
        acc = float(m.group(1))
    # F1 可能有多种（micro/macro/weighted），记录是哪一种
    fm = re.search(r"(micro[\-_ ]?f1|macro[\-_ ]?f1|weighted[\-_ ]?f1|f1)\s*[:=]\s*([0-9.]+)", txt, re.I)
    if fm:
        f1 = float(fm.group(2))
        f1_type = fm.group(1).replace("_", "-").replace(" ", "-")
    if acc is None and f1 is None:
        # 兜底：把 "key: value" 形式的行全抓出来
        for k, v in re.findall(r"([A-Za-z_ ]+)\s*[:=]\s*([0-9.]+)", txt):
            kl = k.strip().lower()
            if "acc" in kl and acc is None:
                acc = float(v)
            if "f1" in kl and f1 is None:
                f1 = float(v)
                f1_type = k.strip()
    return {"acc": acc, "f1": f1, "f1_type": f1_type}


def collect(repo, out_dir, tag, task, cluster, method, exp_dir):
    """把一组实验的产物收集到 <out_dir>/<dataset>_<model>_<method>/"""
    meta = TASK_MAP[task]
    dst = os.path.join(out_dir, f"{meta['dataset']}_{MODEL_TAG[tag]}_{method}")
    os.makedirs(dst, exist_ok=True)

    # preds
    for pat in [f"{exp_dir}/preds_for_{cluster}/{task}_prompts3_retrieverse2_preds_0.json",
                f"{exp_dir}/preds_for_{cluster}/{task}_*_preds_0.json"]:
        for p in glob.glob(pat):
            shutil.copy2(p, dst)
            break
    # eval_res
    for p in glob.glob(f"{exp_dir}/eval_res*.txt"):
        shutil.copy2(p, dst)
    # 训练日志
    for p in [f"{exp_dir}/train_dense_encoder.log"]:
        if os.path.exists(p):
            shutil.copy2(p, dst)
    # runtime
    for p in glob.glob(f"{exp_dir}/runtime*.json"):
        shutil.copy2(p, dst)
    # metrics
    for m in ["zeroshot", "random3", "sedpo"]:
        for ext in ["json", "txt"]:
            p = os.path.join(out_dir, f"{task}_{tag}", f"metrics_{m}.{ext}")
            if os.path.exists(p):
                shutil.copy2(p, dst)
    # 打分日志（体积大，只留关键行）
    for p in glob.glob(f"{exp_dir}/scorer.log"):
        try:
            with open(p, encoding="utf-8", errors="ignore") as f:
                keep = [ln for ln in f
                        if re.search(r"one_shot_acc|first_rank_acc|len\(example_list\)|len\(data\)", ln)]
            with open(os.path.join(dst, "scorer_keylines.log"), "w", encoding="utf-8") as f:
                f.write("\n".join(keep[-40:]))
        except Exception:
            pass
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/root/autodl-tmp/SeDPO_NIPS25")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--csv", default="/root/autodl-tmp/runs_log_rows.csv")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    repo = args.repo
    out_dir = args.out_dir or os.path.join(repo, "results_new_tasks_20261105")
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    print("=" * 78)
    print("收集 SeDPO 实验产物")
    print("=" * 78)

    for tag in ["qwen", "llama"]:
        for task, meta in TASK_MAP.items():
            cluster = meta["cluster"]
            exp_dir = f"{repo}/my_data_qwen/experiment_{tag}/{cluster}"
            if not os.path.isdir(exp_dir):
                continue

            # ---- SeDPO ----
            sedpo_dir = os.path.join(out_dir, f"{task}_{tag}")
            mfile = os.path.join(sedpo_dir, "metrics_sedpo.txt")
            if os.path.exists(mfile):
                if not args.dry:
                    dst = collect(repo, out_dir, tag, task, cluster, "SeDPO", exp_dir)
                    print(f"[OK] SeDPO {task}/{tag} → {dst}")
                mt = parse_metrics(mfile)
                rt = {}
                for p in glob.glob(os.path.join(sedpo_dir, "runtime*.json")):
                    try:
                        rt = json.load(open(p, encoding="utf-8"))
                    except Exception:
                        pass
                loss = None
                lossref = ""
                logp = os.path.join(exp_dir, "train_dense_encoder.log")
                if os.path.exists(logp):
                    lossref = "train_dense_encoder.log"
                    txt = open(logp, encoding="utf-8", errors="ignore").read()
                    ls = re.findall(r"loss['\"]?\s*[:=]\s*([0-9.]+)", txt)
                    if ls:
                        loss = ls[-1]
                runtime_min = None
                if rt.get("cluster_total_sec"):
                    runtime_min = round(rt["cluster_total_sec"] / 60, 1)
                rows.append({
                    "dataset": meta["dataset"],
                    "task_code": task,
                    "model": MODEL_TAG[tag],
                    "method": "SeDPO",
                    "shots": 3,
                    "metric_acc": mt["acc"] if mt else None,
                    "metric_f1": mt["f1"] if mt else None,
                    "f1_type": (mt["f1_type"] if mt else "") or "N/A",
                    "loss_final": loss or "N/A",
                    "loss_curve_ref": lossref or "N/A",
                    "runtime_min": runtime_min if runtime_min else "N/A",
                    "n_score": rt.get("n_score", "N/A"),
                    "finder_L": rt.get("finder_L", "N/A"),
                    "notes": "选项似然打分，非采样生成；候选池全量，n_score 限制打分 query 数",
                })

            # ---- 基线（已跑完的保留作参考）----
            for m, mname in [("zeroshot", "zeroshot"), ("random3", "random3")]:
                mfile = os.path.join(sedpo_dir, f"metrics_{m}.txt")
                if not os.path.exists(mfile):
                    continue
                mt = parse_metrics(mfile)
                rows.append({
                    "dataset": meta["dataset"],
                    "task_code": task,
                    "model": MODEL_TAG[tag],
                    "method": f"baseline-{mname}",
                    "shots": 0 if m == "zeroshot" else 3,
                    "metric_acc": mt["acc"] if mt else None,
                    "metric_f1": mt["f1"] if mt else None,
                    "f1_type": (mt["f1_type"] if mt else "") or "N/A",
                    "loss_final": "N/A", "loss_curve_ref": "N/A",
                    "runtime_min": "N/A", "n_score": "N/A", "finder_L": "N/A",
                    "notes": "参考值（用户已确认无需再跑其他基线）",
                })

    # 写 CSV
    cols = ["dataset", "task_code", "model", "method", "shots", "metric_acc", "metric_f1",
            "f1_type", "loss_final", "loss_curve_ref", "runtime_min", "n_score",
            "finder_L", "notes"]
    import csv
    with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n已写 {args.csv}（{len(rows)} 行）")
    for r in rows:
        print(f"  {r['dataset']:11s} {r['model']:14s} {r['method']:18s} "
              f"acc={r['metric_acc']} f1={r['metric_f1']} loss={r['loss_final']} "
              f"t={r['runtime_min']}min")


if __name__ == "__main__":
    main()
