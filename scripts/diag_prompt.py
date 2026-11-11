#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断：检查 OBQA / ARC 的 prompt 与产物是否齐备"""
import glob
import json
import os

R = "/root/autodl-tmp/SeDPO_NIPS25"
EXP = os.path.join(R, "my_data_qwen", "experiment")


def load_blocks(path):
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


for cluster, task in [("close_qa", "arc_c"), ("obqa_only", "openbookqa")]:
    e = os.path.join(EXP, cluster)
    print("=" * 60)
    print(f"[{cluster}]")
    if not os.path.isdir(e):
        print("  目录不存在")
        continue
    print("  目录内容:", sorted(os.listdir(e)))
    model_dir = os.path.join(e, "model")
    print("  model/:", sorted(os.listdir(model_dir)) if os.path.isdir(model_dir) else "(无)")

    for pat in [f"se2_prompts_for_{cluster}/*.json", f"preds_for_{cluster}/*.json"]:
        fs = glob.glob(os.path.join(e, pat))
        print(f"  {pat}: {[os.path.basename(x) for x in fs]}")

    # prompt 结构
    pf = glob.glob(os.path.join(e, f"se2_prompts_for_{cluster}", f"{task}_prompts_beam_score_0.json"))
    if pf:
        rows = load_blocks(pf[0])
        print(f"  prompt 条数: {len(rows)}")
        r = rows[0]
        print("  字段:", list(r.keys()))
        instr = r["instruction"]
        print(f"  instruction 长度: {len(instr)}")
        print("  尾部 260:", repr(instr[-260:]))
        ch = r.get("meta_data", {}).get("choices", {})
        print("  choices.text:", [c[:40] for c in ch.get("text", [])])
        print("  choices.label:", ch.get("label"))
        # 待测题是否与示例格式一致
        print("  是否含 'Question:' 分隔:", instr.count("Question:"))
        print("  是否以 'options.' 结尾:", instr.rstrip().endswith("options."))
    else:
        print("  !! 无 prompt 文件")

    inf = os.path.join(e, "infer.sh")
    if os.path.exists(inf):
        txt = open(inf, encoding="utf-8", errors="ignore").read()
        for line in txt.splitlines():
            if "checkpoint" in line or "--model_name" in line or "task_name=" in line:
                print("  infer.sh :", line.strip()[:160])
