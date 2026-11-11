#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全量 zero-shot 基线：Qwen2.5-7B 在 ARC-C 上到底能答多少

目的：区分「我们的 prompt/检索拖后腿」与「数据/标签有问题」。
同时输出错误样例，便于人工核对。
"""
import argparse
import json
import os
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score

sys.path.insert(0, "/root/autodl-tmp")
from infer_gen import load_blocks, parse  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--prompt_file", required=True)
ap.add_argument("--preds_file", required=True)
ap.add_argument("--model", default="/root/autodl-tmp/models/qwen2.5-7b")
ap.add_argument("--out", default="/root/autodl-tmp/zero_shot_result.json")
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()

P = load_blocks(args.prompt_file)
D = load_blocks(args.preds_file)
if args.limit:
    P, D = P[: args.limit], D[: args.limit]
labels = [int(r["label"]) for r in D]
print(f"[info] 样本 {len(P)}", flush=True)

tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(
    args.model, torch_dtype=torch.float16, device_map={"": 0},
    max_memory={0: "22GiB"}, trust_remote_code=True).eval()

SYS = "You are an expert answering multiple choice science questions. Reply with only the letter of the correct option."


def last_q(instr):
    t = instr.rstrip()
    if t.lower().endswith("answer:"):
        t = t[: t.lower().rfind("answer:")].rstrip()
    i = max(t.rfind("Question:"), t.rfind("Q:"))
    if i > 0:
        t = t[i:]
        t = t[8:] if t.startswith("Question:") else t[2:]
    return t.replace("Pick the answer from these options.", "").strip()


texts = []
for r in P:
    q = last_q(r["instruction"])
    opts = r["meta_data"]["choices"]["text"]
    body = [f"Question: {q}"]
    for i, o in enumerate(opts):
        body.append(f"{'ABCD'[i]}. {o.strip()}")
    body.append("Answer:")
    s = "\n".join(body)
    s = tok.apply_chat_template(
        [{"role": "system", "content": SYS}, {"role": "user", "content": s}],
        tokenize=False, add_generation_prompt=True)
    texts.append(s)

preds, gens = [], []
B = args.batch
for s0 in range(0, len(texts), B):
    chunk = texts[s0: s0 + B]
    enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=2048).to("cuda")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=10, do_sample=False,
                             num_beams=1, pad_token_id=tok.pad_token_id)
    gid = out[:, enc["input_ids"].shape[1]:]
    for j in range(len(chunk)):
        g = tok.decode(gid[j], skip_special_tokens=True)
        gens.append(g)
        preds.append(parse(g, P[s0 + j]["meta_data"]["choices"]["text"]))
    if (s0 // B) % 20 == 0:
        print(f"  {min(s0+B, len(texts))}/{len(texts)}", flush=True)

valid = sum(1 for p in preds if p >= 0)
acc_all = sum(1 for l, p in zip(labels, preds) if l == p) / len(labels) * 100
pairs = [(l, p) for l, p in zip(labels, preds) if p >= 0]
acc_valid = accuracy_score([a for a, _ in pairs], [b for _, b in pairs]) * 100 if pairs else 0.0
print(f"\n[RESULT] zero-shot  acc(全部,invalid计错)={acc_all:.2f}%  "
      f"acc(仅有效)={acc_valid:.2f}%  valid={valid}/{len(labels)}")
print(f"[gold 分布] {{i: labels.count(i) for i in range(4)}}".replace(
    "{i: labels.count(i) for i in range(4)}", str({i: labels.count(i) for i in range(4)})))
print(f"[pred 分布] {dict((k, preds.count(k)) for k in sorted(set(preds)))}")

errs = [(i, labels[i], preds[i], gens[i][:90]) for i in range(len(labels)) if labels[i] != preds[i]]
print(f"\n=== 错误样例（前 12 条，共 {len(errs)}）===")
for e in errs[:12]:
    print(f"  idx={e[0]} gold={e[1]} pred={e[2]} gen={e[3]!r}")

json.dump({"acc_all": acc_all, "acc_valid": acc_valid, "n": len(labels), "valid": valid,
           "records": [{"i": i, "gold": l, "pred": p, "gen": g}
                       for i, (l, p, g) in enumerate(zip(labels, preds, gens))]},
          open(args.out, "w", encoding="utf-8"), ensure_ascii=False)
print("\nsaved:", args.out)
