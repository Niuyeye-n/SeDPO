#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""通用 dialog 评测：SeDPO / Random / Zero 三配置

用法：
  python eval_dialog.py --prompt_file <path> --n 0 --out <json>
"""
import argparse
import json
import os
import random
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, "/root/autodl-tmp")
from infer_gen import load_blocks, parse  # noqa: E402

LETTERS = "ABCD"
SYS = "You are a helpful assistant."

ap = argparse.ArgumentParser()
ap.add_argument("--prompt_file", required=True)
ap.add_argument("--model", default="/root/autodl-tmp/models/qwen2.5-7b")
ap.add_argument("--n", type=int, default=0, help="0 = 全量")
ap.add_argument("--out", default="/root/autodl-tmp/eval_dialog.json")
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--max_length", type=int, default=3072)
args = ap.parse_args()

P = load_blocks(args.prompt_file)
if args.n:
    P = P[: args.n]
allmd = [r["meta_data"] for r in load_blocks(args.prompt_file)]
random.seed(42)
print(f"[info] 样本 {len(P)}", flush=True)


def gold(md):
    ak = md.get("answerKey")
    if isinstance(ak, str):
        a = ak.strip().upper()
        if a in LETTERS:
            return LETTERS.index(a)
        if a.isdigit():
            return int(a) - 1
    if isinstance(ak, (int, float)):
        return int(ak) - 1
    raise ValueError(ak)


labels = [gold(r["meta_data"]) for r in P]

tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(
    args.model, torch_dtype=torch.float16, device_map={"": 0},
    max_memory={0: "22GiB"}, trust_remote_code=True).eval()

_ALLOWED = None


def allowed_ids():
    global _ALLOWED
    if _ALLOWED is not None:
        return _ALLOWED
    ids = set()
    for L in LETTERS:
        for v in (L, " " + L, "(" + L):
            e = tok.encode(v, add_special_tokens=False)
            if len(e) == 1:
                ids.add(e[0])
        ids.add(tok.encode(L, add_special_tokens=False)[-1])
    _ALLOWED = sorted(ids)
    return _ALLOWED


def make_proc():
    from transformers import LogitsProcessor
    allow = allowed_ids()
    st = {"fired": False}

    class FirstOnly(LogitsProcessor):
        def __call__(self, input_ids, scores):
            if st["fired"]:
                return scores
            st["fired"] = True
            mask = torch.full_like(scores, float("-inf"))
            for i in allow:
                if i < scores.shape[-1]:
                    mask[:, i] = 0.0
            return scores + mask

    return FirstOnly()


def q_block(md, with_tail=False):
    lines = [md["question"].strip(), ""]
    for i, o in enumerate(md["choices"]["text"]):
        lines.append(f"{LETTERS[i]}. {o.strip()}")
    if with_tail:
        lines.append("\nAnswer:")
    return "\n".join(lines)


def dialog_msgs(row, demos):
    msgs = []
    for d in demos:
        md = d["meta_data"]
        msgs.append({"role": "user", "content": q_block(md)})
        msgs.append({"role": "assistant", "content": str(md["answerKey"]).strip()})
    msgs.append({"role": "user", "content": q_block(row["meta_data"], with_tail=True)})
    return msgs


def run(name, demos_all):
    texts = [
        tok.apply_chat_template([{"role": "system", "content": SYS}] + dialog_msgs(row, demos),
                                tokenize=False, add_generation_prompt=True)
        for row, demos in zip(P, demos_all)
    ]
    preds = []
    for s0 in range(0, len(texts), args.batch):
        ch = texts[s0: s0 + args.batch]
        enc = tok(ch, return_tensors="pt", padding=True, truncation=True,
                  max_length=args.max_length).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=1, do_sample=False, num_beams=1,
                                 pad_token_id=tok.pad_token_id,
                                 logits_processor=[make_proc()])
        gid = out[:, enc["input_ids"].shape[1]:]
        for j in range(len(ch)):
            preds.append(parse(tok.decode(gid[j], skip_special_tokens=True),
                               P[s0 + j]["meta_data"]["choices"]["text"]))
        if (s0 // args.batch) % 20 == 0:
            print(f"  [{name}] {min(s0+args.batch, len(texts))}/{len(texts)}", flush=True)
    acc = accuracy_score(labels, preds) * 100
    f1w = f1_score(labels, preds, average="weighted", labels=list(range(4)),
                   zero_division=0) * 100
    print(f"### {name:22s} acc={acc:6.2f}%  f1w={f1w:6.2f}%  "
          f"dist={ {i: preds.count(i) for i in range(4)} }", flush=True)
    return {"acc": acc, "f1_weighted": f1w,
            "dist": {str(i): preds.count(i) for i in range(4)}, "n": len(labels)}


res = {}
res["sedpo"] = run("SeDPO 检索 4 条", [r["ctxs"] for r in P])
res["random"] = run("随机 4 条", [[{"meta_data": random.choice(allmd)} for _ in range(4)] for _ in P])
res["zero"] = run("无示例", [[] for _ in P])
json.dump(res, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("\n=== 汇总 ===")
for k, v in res.items():
    print(f"  {k:8s} acc={v['acc']:6.2f}%  f1w={v['f1_weighted']:6.2f}%  n={v['n']}")
print("saved:", args.out)
