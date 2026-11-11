#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成式评测 v2：与 few-shot 示例格式保持一致（模型输出答案原文，再匹配选项）。

关键修正（相对 v1）：
  1. v1 让模型输出 A/B/C/D 字母，但 few-shot 示例的答案全是「选项原文」，
     格式不一致导致模型跑偏（实测生成 "The correct answer is" / "Question: ..."）。
     v2 改为让模型直接输出答案文本，与示例完全同构。
  2. v1 解析失败时兜底返回 0（=A），把 875/1165 条全判成 A，acc 假跌到 23.3。
     v2 解析失败返回 -1（invalid），并在结果里报告无效率；无效率过高直接告警。
  3. max_new_tokens 4 → 24（答案是完整句子）。
  4. 新增三级解析：字母直答 → 选项文本匹配 → 前缀/关键片段匹配。
"""
import argparse
import json
import os
import re
import sys
import unicodedata

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score

LETTERS = "ABCDEFGH"


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


def norm(t):
    """归一化：去标点/空白/大小写，便于比对"""
    t = unicodedata.normalize("NFKD", str(t))
    t = t.lower()
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def build_prompt(instruction, options):
    """原 prompt 结构（实测）：

        Question: {示例1题干} Answer: {示例1答案原文}
        Question: {示例2题干} Answer: {示例2答案原文}
        ...
        Question: {待测题题干} Answer:      ← 待测题已在末尾，答案留空

    全量 1165 条的结尾分布：`Answer:` 178 条 / `?` 767 条 /
    `Pick the answer from these options.` 145 条 / 其他 75 条。

    做法：把带字母的候选选项插到末尾 `Answer:` 之前（防止模型不知道候选），
    然后保留单个 `Answer:` 让模型续写。绝不重复追加题干或第二个 Answer:。
    """
    t = instruction.rstrip()
    opt_block = "\n".join(f"{LETTERS[i]}. {o.strip()}" for i, o in enumerate(options))

    m = re.search(r"answer\s*:\s*$", t, re.I)
    if m:
        head = t[: m.start()].rstrip()
        return f"{head}\n{opt_block}\n\nAnswer:"
    # 没有末尾 Answer: 的样本，补一个
    return f"{t}\n{opt_block}\n\nAnswer:"


def parse(text, options):
    """返回选项下标；无法判定返回 -1"""
    n = len(options)
    t = text.strip()
    if not t:
        return -1

    # ① 显式字母： "Answer: B" / "B." / "(C)" / 开头的裸字母
    m = re.search(r"answer\s*(?:is|:)?\s*\(?\s*([A-%s])\b" % LETTERS[n - 1], t, re.I)
    if m:
        return LETTERS.index(m.group(1).upper())
    m = re.match(r"^\s*\(?\s*([A-%s])\s*[\.\)\:]" % LETTERS[n - 1], t)
    if m:
        return LETTERS.index(m.group(1).upper())
    m = re.match(r"^\s*\(?\s*([A-%s])\s*$" % LETTERS[n - 1], t)
    if m:
        return LETTERS.index(m.group(1).upper())

    # ② 与选项原文匹配（few-shot 的标准答案形式）
    gt = norm(t)
    if not gt:
        return -1
    cands = []
    for i, opt in enumerate(options):
        go = norm(opt)
        if not go:
            continue
        if go == gt:
            cands.append((3, i))
        elif go in gt:
            cands.append((2, i))
        elif gt in go and len(gt) >= max(12, int(0.5 * len(go))):
            cands.append((1, i))
    if cands:
        cands.sort(key=lambda x: (-x[0], -len(norm(options[x[1]]))))
        return cands[0][1]

    # ③ 前缀若干词的宽松匹配
    words = gt.split()
    for take in (10, 7, 5):
        if len(words) < take:
            continue
        head = " ".join(words[:take])
        for i, opt in enumerate(options):
            if norm(opt).startswith(head):
                return i
    return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt_file", required=True)
    ap.add_argument("--label_from", required=True)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--task_name", required=True)
    ap.add_argument("--shot", type=int, default=4)
    ap.add_argument("--out_preds", required=True)
    ap.add_argument("--out_res", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=24)
    ap.add_argument("--no_chat", action="store_true")
    ap.add_argument("--dump", type=int, default=0, help="打印前 N 条原始生成，用于诊断")
    ap.add_argument("--max_length", type=int, default=2048)
    args = ap.parse_args()

    prompts = load_blocks(args.prompt_file)
    labels = [int(r["label"]) for r in load_blocks(args.label_from)]
    assert len(prompts) == len(labels), f"条数不一致 prompt={len(prompts)} label={len(labels)}"
    if args.limit:
        prompts, labels = prompts[:args.limit], labels[:args.limit]
    print(f"[info] 样本 {len(prompts)}  task={args.task_name}  shot={args.shot}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map={"": 0},
        max_memory={0: "22GiB"},
        trust_remote_code=True,
    ).eval()

    use_chat = (not args.no_chat) and bool(getattr(tok, "chat_template", None))
    print(f"[info] chat_template={'ON' if use_chat else 'OFF'}  max_new_tokens={args.max_new_tokens}", flush=True)

    texts, opts_all = [], []
    for r in prompts:
        opts = r["meta_data"]["choices"]["text"]
        raw = build_prompt(r["instruction"], opts)
        if use_chat:
            raw = tok.apply_chat_template(
                [{"role": "user", "content": raw}],
                tokenize=False, add_generation_prompt=True)
        texts.append(raw)
        opts_all.append(opts)

    preds, gens = [], []
    for start in range(0, len(texts), args.batch):
        chunk = texts[start:start + args.batch]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=args.max_length).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tok.pad_token_id,
            )
        gen_ids = out[:, enc["input_ids"].shape[1]:]
        for j in range(len(chunk)):
            g = tok.decode(gen_ids[j], skip_special_tokens=True)
            preds.append(parse(g, opts_all[start + j]))
            gens.append(g)
        if (start // args.batch) % 20 == 0:
            print(f"  {min(start + len(chunk), len(texts))}/{len(texts)}", flush=True)

    valid = [(l, p) for l, p in zip(labels, preds) if p >= 0]
    n_invalid = len(labels) - len(valid)
    inv_rate = n_invalid / max(1, len(labels)) * 100
    acc = accuracy_score([v[0] for v in valid], [v[1] for v in valid]) * 100 if valid else 0.0
    f1w = f1_score([v[0] for v in valid], [v[1] for v in valid], average="weighted") * 100 if valid else 0.0
    f1m = f1_score([v[0] for v in valid], [v[1] for v in valid], average="macro") * 100 if valid else 0.0
    # invalid 一律按错计入
    acc_all = sum(1 for l, p in zip(labels, preds) if l == p) / max(1, len(labels)) * 100
    f1w_all = f1_score(labels, [p if p >= 0 else -1 for p in preds], average="weighted") * 100

    if args.dump:
        print("\n=== 原始生成样本 ===")
        for k in range(min(args.dump, len(gens))):
            print(f"[{k}] label={labels[k]} pred={preds[k]}")
            print(f"    gen={gens[k]!r}")
            print(f"    opts={opts_all[k]}")

    os.makedirs(os.path.dirname(args.out_preds), exist_ok=True)
    json.dump([{"label": l, "pred": p, "gen": g} for l, p, g in zip(labels, preds, gens)],
              open(args.out_preds, "w", encoding="utf-8"), ensure_ascii=False)

    line = (f"retriever: SeDPO(dp2-r1.0-b0.1); model: {args.model_name}; "
            f"task_name: {args.task_name}; shot_num: {args.shot}; eval: generative_v2; "
            f"scores: {{'acc': {acc:.6f}, 'f1_weighted': {f1w:.6f}, 'f1_macro': {f1m:.6f}, "
            f"'acc_all_incl_invalid': {acc_all:.6f}, 'f1_weighted_all': {f1w_all:.6f}, "
            f"'invalid': {n_invalid}, 'invalid_rate': {inv_rate:.2f}, "
            f"'n': {len(labels)}, 'n_valid': {len(valid)}}}\n")
    open(args.out_res, "w", encoding="utf-8").write(line)
    print("\n[RESULT] " + line.strip())
    n_opt = max(len(o) for o in opts_all)
    print("[pred 分布] " + str({i: preds.count(i) for i in range(-1, n_opt)}))
    print("[gold 分布] " + str({i: labels.count(i) for i in range(n_opt)}))
    if inv_rate > 10:
        print(f"!! 警告：无效率 {inv_rate:.1f}% 偏高，指标不可信，先看 --dump 输出")
    return 0


if __name__ == "__main__":
    sys.exit(main())
