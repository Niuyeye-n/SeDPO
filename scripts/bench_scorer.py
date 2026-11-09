# -*- coding: utf-8 -*-
"""
F4 打分吞吐基准：修复前 vs 修复后
测 3 件事：
  1. 单 batch 峰值显存
  2. 可用的最大 batch_size
  3. 单位时间吞吐（打分实例/秒）—— 直接决定实验规模能放大多少

用法：python bench_scorer.py <repo_path> <model_path>
"""
import os
import sys
import time

REPO = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/SeDPO_NIPS25"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "/root/autodl-tmp/models/qwen2.5-7b"
sys.path.insert(0, REPO)
os.chdir(REPO)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SEQ = 1024      # 典型打分序列长度（ARC-C 带 3-shot 大约几百到 1k）
OPT = 4         # ARC-C / OpenBookQA 都是 4 选项
VOCAB = 152064  # Qwen2.5 的词表

print(f"model={MODEL}  seq={SEQ}  option={OPT}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL, cache_dir=os.path.join(REPO, "cache"), torch_dtype=torch.float16
).half().cuda().eval()


def old_loss(logits, input_ids, loss_mask):
    bsz, option_num, seq_len, _ = logits.shape
    targets = input_ids[:, :, 1:].unsqueeze(-1)
    lg = logits[:, :, :-1, :]
    logit_probs = torch.nn.functional.log_softmax(lg.float(), dim=-1)
    lm = loss_mask[:, :, 1:]
    loss = -torch.gather(logit_probs, -1, targets).squeeze(-1) * lm
    return loss.sum(-1) / lm.sum(-1)


def new_loss(logits, input_ids, loss_mask):
    bsz, option_num, seq_len, _ = logits.shape
    targets = input_ids[:, :, 1:].unsqueeze(-1)
    lg = logits[:, :, :-1, :]
    lm = loss_mask[:, :, 1:]
    flat_logits = lg.reshape(-1, lg.shape[-1])
    flat_targets = targets.reshape(-1)
    flat_mask = lm.reshape(-1).bool()
    sel_logits = flat_logits[flat_mask]
    sel_targets = flat_targets[flat_mask].squeeze(-1)
    lse = torch.logsumexp(sel_logits.float(), dim=-1)
    tgt = torch.gather(sel_logits, -1, sel_targets.unsqueeze(-1)).squeeze(-1).float()
    nll = lse - tgt
    loss = torch.zeros(flat_logits.shape[0], device=lg.device, dtype=torch.float32)
    loss[flat_mask] = nll
    loss = loss.view(bsz, option_num, -1)
    return loss.sum(-1) / lm.sum(-1).clamp(min=1)


def bench(fn, bsz, n_iter=3):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    input_ids = torch.randint(0, VOCAB, (bsz, OPT, SEQ), device="cuda")
    loss_mask = torch.zeros(bsz, OPT, SEQ, device="cuda")
    loss_mask[:, :, -8:] = 1
    try:
        with torch.no_grad():
            for _ in range(n_iter):
                out = model(input_ids=input_ids.reshape(bsz * OPT, SEQ),
                            attention_mask=torch.ones(bsz * OPT, SEQ, dtype=torch.long, device="cuda"))
                logits = out.logits.reshape(bsz, OPT, SEQ, -1)
                loss = fn(logits, input_ids, loss_mask)
                del out, logits, loss
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1e9
        # 计时
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            for _ in range(n_iter):
                out = model(input_ids=input_ids.reshape(bsz * OPT, SEQ),
                            attention_mask=torch.ones(bsz * OPT, SEQ, dtype=torch.long, device="cuda"))
                logits = out.logits.reshape(bsz, OPT, SEQ, -1)
                loss = fn(logits, input_ids, loss_mask)
                del out, logits, loss
        torch.cuda.synchronize()
        dt = (time.time() - t0) / n_iter
        return peak, dt, bsz / dt
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None, None, None


print()
print(f"{'实现':<10}{'bsz':>6}{'峰值显存GB':>14}{'秒/batch':>12}{'实例/秒':>12}")
print("-" * 56)

results = {}
for name, fn in [("原实现", old_loss), ("修复后", new_loss)]:
    best = None
    for bsz in [2, 5, 8, 12, 16, 24, 32]:
        peak, dt, tps = bench(fn, bsz)
        if peak is None:
            print(f"{name:<10}{bsz:>6}{'OOM':>14}")
            continue
        print(f"{name:<10}{bsz:>6}{peak:>14.2f}{dt:>12.3f}{tps:>12.1f}")
        if best is None or tps > best[2]:
            best = (bsz, peak, tps, dt)
    results[name] = best
    print()

print("=" * 56)
for name, best in results.items():
    if best:
        print(f"{name}: 最优 bsz={best[0]}, 峰值显存 {best[1]:.2f}GB, {best[2]:.1f} 实例/秒")
if results.get("原实现") and results.get("修复后"):
    sp = results["修复后"][2] / results["原实现"][2]
    print(f"\n=> 吞吐提升 {sp:.2f}x")
