#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""最终评测：用结构化 meta_data/ctxs 重建统一格式的 prompt

三种配置（同模型、同测试集、同选项渲染，唯一变量是示例来源）：
  zero  无示例
  fs    SeDPO 检索出的 4 条示例（ctxs）
  rnd   随机 4 条示例

格式统一为：
  Question: {题干}
  A. {选项1}
  B. {选项2}
  C. {选项3}
  D. {选项4}
  Answer: {正确字母}      ← 仅示例有
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

R = "/root/autodl-tmp/SeDPO_NIPS25"
E = os.path.join(R, "my_data_qwen", "experiment", "close_qa")
DEFAULT_PROMPT = os.path.join(E, "se2_prompts_for_close_qa", "arc_c_prompts_beam_score_0.json")

ap = argparse.ArgumentParser()
ap.add_argument("--prompt_file", default=DEFAULT_PROMPT)
ap.add_argument("--preds", default=os.path.join(E, "preds_for_close_qa",
                                                "arc_c_prompts4_retrieverse2_preds_0.json"))
ap.add_argument("--model", default="/root/autodl-tmp/models/qwen2.5-7b")
ap.add_argument("--n", type=int, default=100, help="0=全量")
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--constrained", action="store_true",
                help="约束解码：首个生成 token 只允许 A/B/C/D，一步出答案")
ap.add_argument("--out", default="/root/autodl-tmp/final_eval.json")
args = ap.parse_args()

P = load_blocks(args.prompt_file)
if args.n:
    P = P[: args.n]
print(f"[info] 样本 {len(P)}", flush=True)

tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(
    args.model, torch_dtype=torch.float16, device_map={"": 0},
    max_memory={0: "22GiB"}, trust_remote_code=True).eval()

SYS = "You are a helpful assistant."

LETTERS = "ABCD"

# === 约束解码：只允许首个 token 取 A/B/C/D ===
_ALLOWED = None


def allowed_letter_ids():
    global _ALLOWED
    if _ALLOWED is not None:
        return _ALLOWED
    ids = set()
    for L in LETTERS:
        for variant in (L, " " + L, "(" + L, " " + L + "."):
            enc = tok.encode(variant, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(enc[0])
        # 单独字母本身一定可取
        ids.add(tok.encode(L, add_special_tokens=False)[-1])
    _ALLOWED = sorted(ids)
    print(f"[constrained] 允许的 token ids: {_ALLOWED}")
    return _ALLOWED


def make_processor():
    from transformers import LogitsProcessor

    allow = allowed_letter_ids()
    state = {"fired": False}

    class FirstTokenOnly(LogitsProcessor):
        def __call__(self, input_ids, scores):
            if state["fired"]:
                return scores
            state["fired"] = True
            mask = torch.full_like(scores, float("-inf"))
            for i in allow:
                if i < scores.shape[-1]:
                    mask[:, i] = 0.0
            return scores + mask

    return FirstTokenOnly()


def render_q(md, with_answer=True):
    """渲染一道题。

    【重要】不要加 "Question: " 前缀。实测 Qwen2.5-7B-Instruct 看到
    "Question: xxx" 会进入「续写题目列表」模式，输出下一道题而不是答案；
    去掉前缀后模型正常作答（见 pinpoint.py 配置 C，3/3 正确）。
    """
    lines = [md["question"].strip(), ""]
    for i, o in enumerate(md["choices"]["text"]):
        lines.append(f"{LETTERS[i]}. {o.strip()}")
    ak = md.get("answerKey")
    if with_answer and ak:
        lines.append(f"\nThe answer is {ak}.")
    else:
        lines.append("\nAnswer:")
    return "\n".join(lines)


def gold_index(md):
    """answerKey 可能是 'C' / 'c' / 3 / 2 等多种形式，统一转成 0-based 下标"""
    ak = md.get("answerKey")
    if ak is None:
        raise ValueError("answerKey 缺失")
    if isinstance(ak, str):
        a = ak.strip().upper()
        if a in LETTERS:
            return LETTERS.index(a)
        if a.isdigit():
            v = int(a)
            return v - 1 if v >= 1 else v
        # 与选项文本匹配
        for i, o in enumerate(md["choices"]["text"]):
            if o.strip() == ak.strip():
                return i
    if isinstance(ak, (int, float)):
        v = int(ak)
        return v - 1 if v >= 1 else v
    raise ValueError(f"无法解析 answerKey={ak!r}")


def build(row, demos):
    parts = []
    for d in demos:
        parts.append(render_q(d["meta_data"], with_answer=True))
        parts.append("")
    parts.append(render_q(row["meta_data"], with_answer=False))
    return "\n".join(parts)


# 收集所有可用的示例池（全量题，作为 random 的来源）
all_md = [r["meta_data"] for r in load_blocks(args.prompt_file)]
random.seed(42)

configs = {
    "zero": [[ ] for _ in P],
    "fs": [r["ctxs"] for r in P],
    "rnd": [[{"meta_data": random.choice(all_md)} for _ in range(4)] for _ in P],
}

results = {}
for name, demos_all in configs.items():
    texts = []
    for row, demos in zip(P, demos_all):
        body = build(row, demos)
        s = tok.apply_chat_template(
            [{"role": "system", "content": SYS}, {"role": "user", "content": body}],
            tokenize=False, add_generation_prompt=True)
        texts.append(s)

    preds, gens = [], []
    B = args.batch
    for s0 in range(0, len(texts), B):
        chunk = texts[s0: s0 + B]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=2048).to("cuda")
        gen_kw = dict(max_new_tokens=1 if args.constrained else 16,
                      do_sample=False, num_beams=1, pad_token_id=tok.pad_token_id)
        if args.constrained:
            gen_kw["logits_processor"] = [make_processor()]
        with torch.no_grad():
            out = model.generate(**enc, **gen_kw)
        gid = out[:, enc["input_ids"].shape[1]:]
        for j in range(len(chunk)):
            g = tok.decode(gid[j], skip_special_tokens=True)
            gens.append(g)
            preds.append(parse(g, P[s0 + j]["meta_data"]["choices"]["text"]))
        if (s0 // B) % 20 == 0:
            print(f"  [{name}] {min(s0+B, len(texts))}/{len(texts)}", flush=True)

    labels = [gold_index(r["meta_data"]) for r in P]
    # 交叉校验：answerKey 推出的下标 vs preds 文件里的 label
    if os.path.exists(args.preds):
        try:
            pl = [int(x["label"]) for x in load_blocks(args.preds)[: len(P)]]
            mism = sum(1 for a, b in zip(labels, pl) if a != b)
            print(f"[校验] answerKey vs preds.label 不一致 {mism}/{len(labels)}")
        except Exception as e:
            print(f"[校验] 跳过：{e}")
    valid = sum(1 for p in preds if p >= 0)
    acc_all = sum(1 for l, p in zip(labels, preds) if l == p) / len(labels) * 100
    pr = [(l, p) for l, p in zip(labels, preds) if p >= 0]
    acc_v = accuracy_score([a for a, _ in pr], [b for _, b in pr]) * 100 if pr else 0.0
    f1w = f1_score(labels, preds, average="weighted", labels=list(range(4))) * 100

    results[name] = {"acc_all": acc_all, "acc_valid": acc_v, "valid": valid,
                     "n": len(labels), "f1_weighted": f1w}
    print(f"\n### {name}: acc_all={acc_all:.2f}%  acc_valid={acc_v:.2f}%  "
          f"valid={valid}/{len(labels)}  f1w={f1w:.2f}")
    print(f"    pred 分布 {dict((k, preds.count(k)) for k in sorted(set(preds)))}")
    bad = [(i, labels[i], p, g) for i, (p, g) in enumerate(zip(preds, gens)) if p < 0][:3]
    for i, l, p, g in bad:
        print(f"    invalid[{i}] gold={l} gen={repr(g)[:80]}")

json.dump(results, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("\n=== 汇总 ===")
for k, v in results.items():
    print(f"  {k:6s} acc_all={v['acc_all']:6.2f}%  acc_valid={v['acc_valid']:6.2f}%  "
          f"valid={v['valid']}/{v['n']}")
print("saved:", args.out)
