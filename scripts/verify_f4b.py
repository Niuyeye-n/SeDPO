# -*- coding: utf-8 -*-
"""
F4b 真实模型等价性验证：
  用真实权重（Qwen2.5-7B / Llama3-8B）+ 真实 tokenizer 构造 batch，
  对比「原始实现」与「F4b 实现」算出的 loss 与预测是否一致。

  同时输出显存对比。

用法：python verify_f4b.py <repo_path> <model_path> [tokenizer_path]
"""
import os
import sys
import time

REPO = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/SeDPO_NIPS25"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "/root/autodl-tmp/models/qwen2.5-7b"
TOK = sys.argv[3] if len(sys.argv) > 3 else MODEL
sys.path.insert(0, REPO)
os.chdir(REPO)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(TOK, cache_dir=os.path.join(REPO, "cache"))
model = AutoModelForCausalLM.from_pretrained(
    MODEL, cache_dir=os.path.join(REPO, "cache"), torch_dtype=torch.float16
).half().cuda().eval()

OPT = 4
SEQ = 640


def build_batch(bsz, seed=0):
    """构造一批「问题 + 4 个选项」的假 batch，答案 token 在末尾"""
    torch.manual_seed(seed)
    input_ids = torch.randint(1000, 50000, (bsz, OPT, SEQ))
    attention = torch.ones(bsz, OPT, SEQ, dtype=torch.long)
    loss_mask = torch.zeros(bsz, OPT, SEQ)
    for b in range(bsz):
        for o in range(OPT):
            n = 3
            loss_mask[b, o, SEQ - n:] = 1
    labels = torch.randint(0, OPT, (bsz, 1))
    return input_ids.cuda(), attention.cuda(), loss_mask.cuda(), labels.cuda()


def original_impl(input_ids, attn, loss_mask, labels):
    """原始上游实现"""
    bsz, option_num, seq_len = input_ids.shape
    with torch.no_grad():
        output = model(input_ids=input_ids.reshape(bsz * option_num, seq_len),
                       attention_mask=attn.reshape(bsz * option_num, seq_len))
    targets = input_ids[:, :, 1:].unsqueeze(-1)
    logits = output.logits.reshape(bsz, option_num, seq_len, -1)
    logits = logits[:, :, :-1, :]
    logit_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
    lm = loss_mask[:, :, 1:]
    loss = -torch.gather(logit_probs, -1, targets).squeeze(-1) * lm
    loss = loss.sum(-1) / lm.sum(-1)
    preds = torch.argmin(loss, dim=-1)
    normed = torch.nn.functional.normalize(loss, p=1, dim=-1)
    labels_losses = torch.gather(normed, -1, labels).squeeze(-1)
    return loss, preds, labels_losses


def f4b_impl(input_ids, attn, loss_mask, labels):
    """F4b：hidden states + 定点 lm_head"""
    bsz, option_num, seq_len = input_ids.shape
    targets = input_ids[:, :, 1:].unsqueeze(-1)
    lm = loss_mask[:, :, 1:]
    with torch.no_grad():
        out = model.model(input_ids=input_ids.reshape(bsz * option_num, seq_len),
                          attention_mask=attn.reshape(bsz * option_num, seq_len),
                          use_cache=False)
        hidden = out.last_hidden_state.reshape(bsz, option_num, seq_len, -1)
    flat_hidden = hidden[:, :, :-1, :].reshape(-1, hidden.shape[-1])
    flat_targets = targets.reshape(-1)
    flat_mask = lm.reshape(-1).bool()
    sel_hidden = flat_hidden[flat_mask]
    sel_targets = flat_targets[flat_mask].squeeze(-1)
    with torch.no_grad():
        sel_logits = model.lm_head(sel_hidden).float()
    lse = torch.logsumexp(sel_logits, dim=-1)
    tgt = torch.gather(sel_logits, -1, sel_targets.unsqueeze(-1)).squeeze(-1)
    nll = lse - tgt
    loss = torch.zeros(flat_mask.shape[0], device=hidden.device, dtype=torch.float32)
    loss[flat_mask] = nll
    loss = loss.view(bsz, option_num, -1)
    loss = loss.sum(-1) / lm.sum(-1).clamp(min=1)
    preds = torch.argmin(loss, dim=-1)
    normed = torch.nn.functional.normalize(loss, p=1, dim=-1)
    labels_losses = torch.gather(normed, -1, labels).squeeze(-1)
    return loss, preds, labels_losses


print(f"model={MODEL}")
print(f"{'bsz':>5} {'实现':<10}{'峰值显存GB':>13}{'秒/batch':>11}{'实例/秒':>10}  一致性")
print("-" * 78)

for bsz in [2, 4, 8, 16, 32, 48, 64]:
    input_ids, attn, lm, labels = build_batch(bsz)
    row = []
    for name, fn in [("原始", original_impl), ("F4b", f4b_impl)]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            with torch.no_grad():
                out = fn(input_ids, attn, lm, labels)
                torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(2):
                    out = fn(input_ids, attn, lm, labels)
                torch.cuda.synchronize()
                dt = (time.time() - t0) / 2
            peak = torch.cuda.max_memory_allocated() / 1e9
            row.append((name, peak, dt, bsz / dt, out))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            row.append((name, None, None, None, None))
    # 一致性对比
    cons = ""
    if row[0][4] is not None and row[1][4] is not None:
        l0, p0, y0 = row[0][4]
        l1, p1, y1 = row[1][4]
        dl = (l0 - l1).abs().max().item()
        dp = (p0 != p1).sum().item()
        dy = (y0 - y1).abs().max().item()
        cons = f"loss Δ={dl:.2e} pred 不一致={dp}/{p0.numel()} yΔ={dy:.2e}"
    for name, peak, dt, tps, _ in row:
        if peak is None:
            print(f"{bsz:>5} {name:<10}{'OOM':>13}")
        else:
            print(f"{bsz:>5} {name:<10}{peak:>13.2f}{dt:>11.3f}{tps:>10.1f}  {cons if name=='F4b' else ''}")
    print("-" * 78)
