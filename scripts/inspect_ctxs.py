#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查 prompt 文件里 ctxs 字段的结构——用它重建干净 4-shot 比抠字符串可靠"""
import json
import os

R = "/root/autodl-tmp/SeDPO_NIPS25"
E = os.path.join(R, "my_data_qwen", "experiment", "close_qa")
PROMPT = os.path.join(E, "se2_prompts_for_close_qa", "arc_c_prompts_beam_score_0.json")


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


P = load_blocks(PROMPT)
print("总条数:", len(P))
print("顶层字段:", list(P[0].keys()))
print()

for idx in (0, 1):
    r = P[idx]
    print("=" * 70)
    print(f"### idx {idx}")
    print("ctxs 类型:", type(r["ctxs"]).__name__, "长度:", len(r["ctxs"]) if hasattr(r["ctxs"], "__len__") else "?")
    ctxs = r["ctxs"]
    if isinstance(ctxs, list) and ctxs:
        c0 = ctxs[0]
        print("单条 ctx 类型:", type(c0).__name__)
        if isinstance(c0, dict):
            print("ctx 字段:", list(c0.keys()))
            for k, v in c0.items():
                print(f"   {k} = {repr(v)[:200]}")
        else:
            print("ctx[0]:", repr(c0)[:400])
    print()
    print("meta_data 字段:", list(r["meta_data"].keys()))
    for k, v in r["meta_data"].items():
        print(f"   {k} = {repr(v)[:220]}")
    print()
