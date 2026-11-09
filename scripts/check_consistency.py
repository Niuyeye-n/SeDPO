# -*- coding: utf-8 -*-
"""检查已有 find_random/scored 产物与当前 n_score 是否一致"""
import json
import os
import sys

REPO = "/root/autodl-tmp/SeDPO_NIPS25"
NS = int(sys.argv[1]) if len(sys.argv) > 1 else 600

for tag in ["sedpo_qwen", "sedpo_llama"]:
    for cluster in ["close_qa", "reading_new"]:
        fr = f"{REPO}/my_data_qwen/find_random_{tag}/{cluster}"
        sc = f"{REPO}/my_data_qwen/scored_{tag}/{cluster}"
        if not os.path.isdir(fr):
            continue
        print(f"\n### {tag} / {cluster}")
        for f in sorted(os.listdir(fr)):
            if not f.endswith(".json"):
                continue
            p = os.path.join(fr, f)
            try:
                d = json.load(open(p))
            except Exception as e:
                print(f"  {f}: 读取失败 {e}")
                continue
            key = None
            for k in ("step_1_ctxs", "step_2_ctxs", "step_3_ctxs"):
                if k in d[0]:
                    key = k
                    break
            n = len(d)
            mark = "OK " if n == NS else "!! "
            print(f"  {mark}{f}: {n} 条 query (期望 {NS})  字段={key}")
        if os.path.isdir(sc):
            for f in sorted(os.listdir(sc)):
                print(f"     scored: {f}  {os.path.getsize(os.path.join(sc,f))/1e6:.1f}MB")
