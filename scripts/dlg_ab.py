#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dialog 风格下：SeDPO 检索 vs 随机示例 vs 无示例

目的：判断 88% 是"检索有效"还是"格式本身的功劳"。
"""
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
SYS = "You are a helpful assistant."
N = 200

P = load_blocks(PROMPT)[:N]
allmd = [r["meta_data"] for r in load_blocks(PROMPT)]
random.seed(42)


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

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, device_map={"": 0},
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
    for s0 in range(0, len(texts), 8):
        ch = texts[s0: s0 + 8]
        enc = tok(ch, return_tensors="pt", padding=True, truncation=True,
                  max_length=3072).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=1, do_sample=False, num_beams=1,
                                 pad_token_id=tok.pad_token_id,
                                 logits_processor=[make_proc()])
        gid = out[:, enc["input_ids"].shape[1]:]
        for j in range(len(ch)):
            preds.append(parse(tok.decode(gid[j], skip_special_tokens=True),
                               P[s0 + j]["meta_data"]["choices"]["text"]))
    acc = accuracy_score(labels, preds) * 100
    print(f"### {name:26s} acc={acc:6.2f}%  dist={ {i: preds.count(i) for i in range(4)} }", flush=True)
    return acc


print(f"N = {N}", flush=True)
a1 = run("SeDPO 检索 4 条", [r["ctxs"] for r in P])
a2 = run("随机 4 条", [[{"meta_data": random.choice(allmd)} for _ in range(4)] for _ in P])
a3 = run("无示例（zero-shot）", [[] for _ in P])
json.dump({"sedpo": a1, "random": a2, "zero": a3, "n": N},
          open("/root/autodl-tmp/dlg_ab.json", "w"))
