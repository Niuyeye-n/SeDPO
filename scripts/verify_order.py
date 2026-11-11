#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证：prompt 文件与 preds 文件是否同序（同一条测试题）"""
import json
import os
import re

R = "/root/autodl-tmp/SeDPO_NIPS25"
E = os.path.join(R, "my_data_qwen", "experiment", "close_qa")
PROMPT = os.path.join(E, "se2_prompts_for_close_qa", "arc_c_prompts_beam_score_0.json")
PREDS = os.path.join(E, "preds_for_close_qa", "arc_c_prompts4_retrieverse2_preds_0.json")


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
D = load_blocks(PREDS)
print("prompt 条数:", len(P), " preds 条数:", len(D))
print("preds[0] 字段:", list(D[0].keys()))
print()


def last_q(instr):
    """取 instruction 里最后一段题干（最后一个 'options.' 之前的那句）"""
    t = instr.strip()
    idx = t.rfind("Pick the answer from these options.")
    seg = t[:idx] if idx > 0 else t
    # 取最后一句较长文本作为题干
    parts = re.split(r"\n|Question:", seg)
    parts = [p.strip() for p in parts if len(p.strip()) > 40]
    return parts[-1] if parts else seg[-200:]


print("=" * 70)
for i in range(4):
    p = P[i]
    d = D[i]
    opts = p["meta_data"]["choices"]["text"]
    q = last_q(p["instruction"])
    print(f"\n--- idx {i} ---")
    print(f"prompt 题干尾: ...{q[-110:]!r}")
    print(f"prompt 选项  : {[o[:45] for o in opts]}")
    for k in ("enc_answer", "ctxs", "label", "pred", "n_prompts"):
        if k in d:
            v = d[k]
            if isinstance(v, list) and v and isinstance(v[0], str):
                print(f"preds.{k} : {[x[:45] for x in v]}")
            else:
                print(f"preds.{k} : {repr(v)[:160]}")
    # 判定：prompt 的选项 是否出现在 preds.enc_answer 里
    if "enc_answer" in d:
        same = [a.strip() == o.strip() for a, o in zip(d["enc_answer"], opts)]
        print(f"  >> 选项是否逐条相同: {same}")
