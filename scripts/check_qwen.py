#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""判定 Qwen 是 base 还是 instruct，并校验权重完整性"""
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

D = os.path.realpath("/root/autodl-tmp/models/qwen2.5-7b")
print("实际路径:", D)

# ---------- 完整性 ----------
idx = os.path.join(D, "model.safetensors.index.json")
if os.path.exists(idx):
    meta = json.load(open(idx))
    total = meta["metadata"]["total_size"]
    print(f"index 声明总大小: {total/1024**3:.2f} GiB")
    got = 0
    for f in sorted(os.listdir(D)):
        if f.endswith(".safetensors"):
            got += os.path.getsize(os.path.join(D, f))
    print(f"实际分片合计:   {got/1024**3:.2f} GiB  ({got} bytes)")
    print(f"是否完整: {abs(got-total) < 1024}")
print("分片:", [f for f in sorted(os.listdir(D)) if f.endswith(".safetensors")])
print("prefix 残留:", [f for f in os.listdir(D) if f.endswith(".prefix")])

# ---------- 行为判定 ----------
tok = AutoTokenizer.from_pretrained(D, trust_remote_code=True)
print("\nchat_template 存在:", bool(getattr(tok, "chat_template", None)))
print("eos_token:", tok.eos_token, tok.eos_token_id)
print("pad_token:", tok.pad_token, tok.pad_token_id)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    D, torch_dtype=torch.float16, device_map={"": 0},
    max_memory={0: "22GiB"}, trust_remote_code=True).eval()

tests = [
    ("raw", "The capital of France is"),
    ("chat", "What is 2+2? Reply with only the number."),
    ("chat", "Answer this multiple choice question. Reply with only the letter.\n\n"
             "Question: What is the capital of France?\nA. London\nB. Paris\nC. Rome\nD. Berlin\nAnswer:"),
]
for mode, q in tests:
    if mode == "chat":
        s = tok.apply_chat_template([{"role": "user", "content": q}],
                                    tokenize=False, add_generation_prompt=True)
    else:
        s = q
    enc = tok(s, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=24, do_sample=False,
                             num_beams=1, pad_token_id=tok.pad_token_id)
    g = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\n[{mode}] Q={q[:70]!r}\n         A={g!r}")
