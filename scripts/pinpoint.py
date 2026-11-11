#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""精确定位：为什么模型输出 'Question:' 而不是答案

变量：system 有无 / "Question:" 前缀有无 / max_new_tokens
每条都打印完整生成 + prompt token 数
"""
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/root/autodl-tmp")
from infer_gen import load_blocks  # noqa: E402

R = "/root/autodl-tmp/SeDPO_NIPS25"
E = os.path.join(R, "my_data_qwen", "experiment", "close_qa")
PROMPT = os.path.join(E, "se2_prompts_for_close_qa", "arc_c_prompts_beam_score_0.json")
MODEL = "/root/autodl-tmp/models/qwen2.5-7b"

P = load_blocks(PROMPT)[:10]
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, device_map={"": 0},
    max_memory={0: "22GiB"}, trust_remote_code=True).eval()

L = "ABCD"


def body(r, prefix_q=True, tail="Answer:"):
    q = r["meta_data"]["question"].strip()
    opts = r["meta_data"]["choices"]["text"]
    lines = [f"Question: {q}"] if prefix_q else [q, ""]
    for i, o in enumerate(opts):
        lines.append(f"{L[i]}. {o.strip()}")
    lines.append("")
    lines.append(tail)
    return "\n".join(lines)


def run(tag, texts, sys_msg=None, mx=40):
    print(f"\n{'='*66}\n### {tag}")
    for i in range(min(3, len(texts))):
        msgs = ([{"role": "system", "content": sys_msg}] if sys_msg else []) + \
               [{"role": "user", "content": texts[i]}]
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(s, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=mx, do_sample=False,
                                 num_beams=1, pad_token_id=tok.pad_token_id)
        g = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"  [{i}] tokens={enc['input_ids'].shape[1]}  gen={g!r}")
        if i == 0:
            print(f"      prompt 尾部: {texts[i][-160:]!r}")


SYS1 = ("You are an expert answering multiple choice science questions. "
        "Read the examples, then answer the final question. "
        "Reply with only the letter of the correct option.")
SYS2 = "You are a helpful assistant."

t_pref = [body(r) for r in P]
t_nopref = [body(r, prefix_q=False) for r in P]

run("A. system(长) + Question: 前缀", t_pref, SYS1)
run("B. 无 system + Question: 前缀", t_pref, None)
run("C. system(短) + 无 Question: 前缀", t_nopref, SYS2)
run("D. 无 system + 无前缀 + 明确指令", [x + "\nRespond with only the letter." for x in t_nopref], None)
run("E. system(长) + 前缀 + tail='The answer is'", [body(r, tail="The answer is") for r in P], SYS1)
