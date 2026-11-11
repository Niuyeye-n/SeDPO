#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""风格对比：4 种 few-shot 呈现方式，找出能让 4-shot 不塌缩的那一个

背景：plain（4 道题堆一段文本）会让 Qwen2.5-7B-Instruct 塌缩到全选 A（22%），
而 zero-shot 有 72%。怀疑是"多道题挤在一条 user 消息里"导致模型分不清哪道要答。

四种风格：
  plain     4 题顺序拼成一段（当前基线，已知会塌）
  dialog    多轮对话：user 问题 / assistant 字母，重复 4 次，最后 user 新题
  numbered  Example 1/2/3/4 显式编号 + 空行分隔 + "Now answer this question:"
  shortans  只给 "题干 → 字母"，不重复选项
"""
import argparse
import json
import os
import random
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score

sys.path.insert(0, "/root/autodl-tmp")
from infer_gen import load_blocks, parse  # noqa: E402

R = "/root/autodl-tmp/SeDPO_NIPS25"
E = os.path.join(R, "my_data_qwen", "experiment", "close_qa")
PROMPT = os.path.join(E, "se2_prompts_for_close_qa", "arc_c_prompts_beam_score_0.json")
MODEL = "/root/autodl-tmp/models/qwen2.5-7b"
LETTERS = "ABCD"

ap = argparse.ArgumentParser()
ap.add_argument("--prompt_file", default=PROMPT)
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--styles", default="zero,dialog,numbered,shortans,plain")
ap.add_argument("--out", default="/root/autodl-tmp/style_cmp.json")
args = ap.parse_args()

P = load_blocks(args.prompt_file)[: args.n]
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, device_map={"": 0},
    max_memory={0: "22GiB"}, trust_remote_code=True).eval()

SYS = "You are a helpful assistant."

# ---- 约束解码 ----
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


def q_block(md, with_tail=False, with_answer=None):
    """题干 + 选项（+ Answer 尾部）"""
    lines = [md["question"].strip(), ""]
    for i, o in enumerate(md["choices"]["text"]):
        lines.append(f"{LETTERS[i]}. {o.strip()}")
    if with_answer:
        lines.append(f"\nAnswer: {with_answer}")
    elif with_tail:
        lines.append("\nAnswer:")
    return "\n".join(lines)


def gold(md):
    ak = md.get("answerKey")
    if isinstance(ak, str):
        a = ak.strip().upper()
        if a in LETTERS:
            return LETTERS.index(a)
        if a.isdigit():
            v = int(a)
            return v - 1 if v >= 1 else v
    if isinstance(ak, (int, float)):
        v = int(ak)
        return v - 1 if v >= 1 else v
    raise ValueError(ak)


def build_msgs(row, demos, style):
    if style == "zero":
        return [{"role": "user", "content": q_block(row["meta_data"], with_tail=True)}]

    if style == "dialog":
        msgs = []
        for d in demos:
            md = d["meta_data"]
            msgs.append({"role": "user", "content": q_block(md)})
            msgs.append({"role": "assistant", "content": str(md["answerKey"]).strip()})
        msgs.append({"role": "user", "content": q_block(row["meta_data"], with_tail=True)})
        return msgs

    if style == "numbered":
        parts = []
        for i, d in enumerate(demos, 1):
            md = d["meta_data"]
            parts.append(f"Example {i}:")
            parts.append(q_block(md, with_answer=str(md["answerKey"]).strip()))
            parts.append("")
        parts.append("Now answer this question:")
        parts.append(q_block(row["meta_data"], with_tail=True))
        return [{"role": "user", "content": "\n".join(parts)}]

    if style == "shortans":
        lines = ["Here are some examples:"]
        for d in demos:
            md = d["meta_data"]
            lines.append(f"{md['question'].strip()} -> {str(md['answerKey']).strip()}")
        lines.append("")
        lines.append("Now answer this question:")
        lines.append(q_block(row["meta_data"], with_tail=True))
        return [{"role": "user", "content": "\n".join(lines)}]

    # plain: 4 题顺序拼一段（旧基线）
    parts = []
    for d in demos:
        md = d["meta_data"]
        parts.append(q_block(md, with_answer=str(md["answerKey"]).strip()))
        parts.append("")
    parts.append(q_block(row["meta_data"], with_tail=True))
    return [{"role": "user", "content": "\n".join(parts)}]


labels = [gold(r["meta_data"]) for r in P]
all_md = [r["meta_data"] for r in load_blocks(args.prompt_file)]
random.seed(42)
rnd_demos = [[{"meta_data": random.choice(all_md)} for _ in range(4)] for _ in P]

styles = [s.strip() for s in args.styles.split(",") if s.strip()]
results = {}

for style in styles:
    demos_all = [[] for _ in P] if style == "zero" else [r["ctxs"] for r in P]
    texts = []
    for row, demos in zip(P, demos_all):
        msgs = build_msgs(row, demos, style)
        s = tok.apply_chat_template(
            [{"role": "system", "content": SYS}] + msgs,
            tokenize=False, add_generation_prompt=True)
        texts.append(s)

    preds, gens = [], []
    B = args.batch
    for s0 in range(0, len(texts), B):
        chunk = texts[s0: s0 + B]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=3072).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=1, do_sample=False,
                                 num_beams=1, pad_token_id=tok.pad_token_id,
                                 logits_processor=[make_proc()])
        gid = out[:, enc["input_ids"].shape[1]:]
        for j in range(len(chunk)):
            g = tok.decode(gid[j], skip_special_tokens=True)
            gens.append(g)
            preds.append(parse(g, P[s0 + j]["meta_data"]["choices"]["text"]))

    acc = accuracy_score(labels, preds) * 100
    dist = {i: preds.count(i) for i in range(-1, 4)}
    results[style] = {"acc": acc, "dist": dist}
    print(f"\n### {style:9s} acc={acc:6.2f}%   dist={dist}")
    for i in range(2):
        print(f"    [{i}] gold={labels[i]} pred={preds[i]} gen={repr(gens[i])[:40]}")

print("\n=== 汇总 ===")
for k, v in sorted(results.items(), key=lambda x: -x[1]["acc"]):
    print(f"  {k:9s} {v['acc']:6.2f}%  {v['dist']}")
json.dump(results, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
