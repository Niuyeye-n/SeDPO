#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验进度监控：一条命令看清 SeDPO 实验跑到哪了。

用法: python monitor.py
"""
import json
import os
import re
import subprocess
import sys

REPO = "/root/autodl-tmp/SeDPO_NIPS25"
RESULTS = f"{REPO}/results_sedpo_v3"
LOG = "/root/autodl-tmp/sedpo_v3.log"


def sh(cmd):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return r.stdout.strip()
    except Exception:
        return ""


print("=" * 72)
print("SeDPO 实验进度监控")
print("=" * 72)

# ---- 1. 主编排日志 ----
if os.path.exists(LOG):
    lines = [l for l in open(LOG, encoding="utf-8", errors="ignore").read().split("\n")
             if l.strip() and not re.search(r"it/s|^\s*$", l)]
    print("\n--- 编排日志（最后 8 行）---")
    for l in lines[-8:]:
        print("  " + l[:110])

# ---- 2. 各组打分进度 ----
print("\n--- 各组打分进度 ---")
found = False
import glob
for tag in ["sedpo_qwen", "sedpo_llama"]:
    short = tag.replace("sedpo_", "")
    for cluster in ["close_qa", "reading_new"]:
        logs = sorted(glob.glob(f"{RESULTS}/logs/{cluster}_{short}_*"))
        for lg in logs:
            score_log = os.path.join(lg, "sedpo_score.log")
            if not os.path.isfile(score_log):
                continue
            found = True
            txt = open(score_log, encoding="utf-8", errors="ignore").read()
            # 进度条形如: Scoring:  12%|##  | 2400/20000 [03:20<00:00, 12.00it/s]
            sc = re.findall(r"Scoring:\s*(\d+)%.*?\|\s*(\d+)/(\d+)\s*\[([\d:]+)<([\d:]+),\s*([\d.]+)\s*(it/s|s/it)", txt)
            if sc:
                g = sc[-1]
                p, cur, tot, eta, rate = g[0], g[1], g[2], g[4], g[5]
                print(f"  {cluster}/{short}: {p}% ({cur}/{tot}) 速率 {rate} 预计剩余 {eta}")
            else:
                print(f"  {cluster}/{short}: (无进度条，日志 {os.path.basename(lg)})")
if not found:
    print("  (还没有日志目录)")

# ---- 3. 已完成结果 ----
print("\n--- 已出结果 ---")
rows = []
if os.path.isdir(RESULTS):
    for d in sorted(os.listdir(RESULTS)):
        od = os.path.join(RESULTS, d)
        if not os.path.isdir(od):
            continue
        mf = os.path.join(od, "metrics_sedpo.json")
        if not os.path.exists(mf):
            continue
        try:
            m = json.load(open(mf, encoding="utf-8"))
        except Exception:
            continue
        acc = m.get("accuracy", m.get("acc", "?"))
        f1 = m.get("f1", "?")
        rows.append((d, acc, f1))
if rows:
    print(f"  {'组':<26}{'accuracy':>10}{'F1':>10}")
    for d, a, f in rows:
        print(f"  {d:<26}{str(a):>10}{str(f):>10}")
else:
    print("  (还没有完成的结果)")

# ---- 4. 资源状态 ----
print("\n--- 资源 ---")
gpu = sh("nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader")
print(f"  GPU: {gpu}")
disk = sh("df -h /root/autodl-tmp | tail -1 | awk '{print $4\" 可用 / \"$5\" 已用\"}'")
print(f"  磁盘: {disk}")
procs = sh("ps -eo etime,cmd | grep -E 'scorer.py|train_dense_encoder|inference.py' | grep -v grep | head -2")
for p in procs.split("\n")[:2]:
    if p.strip():
        print(f"  运行中: {p[:100]}")
print()
