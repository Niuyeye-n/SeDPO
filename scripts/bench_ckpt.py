#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BERT-base 训练显存实测：对比「开/关梯度检查点」。

用途：在不打扰正在跑的打分任务的前提下，量化每条序列(512 token)的显存占用，
      再线性外推到真实 batch（约 640 条序列），判断 24GB 卡是否放得下。

用法: python bench_ckpt.py "96,192" "nockpt,ckpt"
"""
import sys

import torch
from transformers import BertModel

MODEL = "/root/autodl-tmp/models/bert-base-uncased"
L = 512


def run(n_seq, use_ckpt):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = BertModel.from_pretrained(MODEL).cuda()
    model.train()
    if use_ckpt:
        model.gradient_checkpointing_enable()
    ids = torch.randint(0, 20000, (n_seq, L), device="cuda")
    mask = torch.ones_like(ids)
    out = model(input_ids=ids, attention_mask=mask).last_hidden_state
    loss = out.mean()
    loss.backward()
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    del model, out, loss, ids, mask
    torch.cuda.empty_cache()
    return peak


def main():
    seqs = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "96,192").split(",")]
    modes = (sys.argv[2] if len(sys.argv) > 2 else "nockpt,ckpt").split(",")
    free, total = torch.cuda.mem_get_info()
    print("GPU 空闲 %.2f GiB / 共 %.2f GiB" % (free / 2 ** 30, total / 2 ** 30))

    for mode in modes:
        for n in seqs:
            try:
                peak = run(n, mode == "ckpt")
                per = peak / n * 1024
                print("[%s] n=%4d  peak=%6.2f GiB  单序列=%6.1f MiB  外推640条=%6.1f GiB"
                      % (mode, n, peak, per, per * 640 / 1024))
            except torch.cuda.OutOfMemoryError:
                print("[%s] n=%4d  OOM（跳过）" % (mode, n))
                torch.cuda.empty_cache()
            sys.stdout.flush()


if __name__ == "__main__":
    main()
