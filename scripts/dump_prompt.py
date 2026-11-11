#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""打印最终送给模型的 prompt 原文，定位为什么模型在续写 Question:"""
import json
import os
import re

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

for idx in (1, 3):
    instr = P[idx]["instruction"]
    print("=" * 72)
    print(f"### idx {idx}  长度 {len(instr)}")
    print("--- 完整 instruction 原文 ---")
    print(instr)
    print()
    print("--- 是否含 'Sam builds a ramp' ---", "Sam builds" in instr)
    print("--- 'Answer:' 出现次数 ---", instr.count("Answer:"))
    print("--- 'Question:' 出现次数 ---", instr.count("Question:"))
    print()

# 统计全量 instruction 结尾模式
print("=" * 72)
ends = {"answer_colon": 0, "options_dot": 0, "question": 0, "other": 0}
for p in P:
    t = p["instruction"].rstrip()
    if t.lower().endswith("answer:"):
        ends["answer_colon"] += 1
    elif t.lower().endswith("options."):
        ends["options_dot"] += 1
    elif t.endswith("?"):
        ends["question"] += 1
    else:
        ends["other"] += 1
print("全量 ending 分布:", ends)
print("n_prompts 字段分布:", {})
