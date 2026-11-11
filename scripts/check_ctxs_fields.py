#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查 ctxs 里每条示例的字段完整性（是否有 choices / answerKey）"""
import json
import os
from collections import Counter

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

keys_cnt = Counter()
has_ak = 0
has_ch = 0
total_ctx = 0
sample = None

for r in P:
    for c in r["ctxs"]:
        total_ctx += 1
        md = c.get("meta_data", {})
        keys_cnt[tuple(sorted(md.keys()))] += 1
        if md.get("answerKey"):
            has_ak += 1
        if md.get("choices"):
            has_ch += 1
        if sample is None and not md.get("answerKey"):
            sample = c

print(f"示例总数: {total_ctx}")
print(f"带 answerKey: {has_ak} ({has_ak/total_ctx*100:.1f}%)")
print(f"带 choices  : {has_ch} ({has_ch/total_ctx*100:.1f}%)")
print("\n字段组合分布:")
for k, v in keys_cnt.most_common(6):
    print(f"  {v:6d}  {k}")

print("\n=== 一条没有 answerKey 的示例 ===")
if sample:
    print(json.dumps(sample, ensure_ascii=False)[:800])
else:
    print("(全部都有 answerKey)")

print("\n=== 第一条完整 ctx[0] ===")
print(json.dumps(P[0]["ctxs"][0], ensure_ascii=False)[:900])

print("\n=== passage 是否含 'Answer:' ===")
c0 = P[0]["ctxs"][0]["passage"]
print("长度:", len(c0))
print("含 Answer: ", "Answer:" in c0)
print(repr(c0[-300:]))
