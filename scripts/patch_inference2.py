#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
patch_inference2.py — 2026-11-07

背景：OpenBookRC 两组推理在 35 秒内 rc=1 崩掉（两次 OOM）。
根因：openbookqa 是 4 选项任务，seq_len 最长 2048，logits 形状
[bsz, option_num=4, 2048, vocab=152064]，
转 float32 后一份就是 4×2048×152064×4B ≈ 4.98 GB，
加上模型本身约 16 GB，24 GB 卡必炸。

改法：把一次性的整块 log_softmax 拆成**按 option 逐个算**。
每个 option 独立做 log_softmax，最后 stack 起来取 argmin。
数学上完全等价（log_softmax 在 vocab 维上逐位置独立，option 之间本来就不耦合），
峰值从 ~5 GB 降到 ~1.25 GB。

脚本自带数值等价性验证：用随机张量同时跑「原整块算法」和「分块算法」，
比对 argmin 出来的 preds 是否完全一致。
"""
import os
import shutil
import sys

import torch

TARGET = "/root/autodl-tmp/SeDPO_NIPS25/scripts/inference.py"
BAK = TARGET + ".bak_before_chunkloss_20261107"

OLD = """        logit_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
        loss = -torch.gather(logit_probs, -1, targets).squeeze(-1) * loss_mask
        loss = loss.sum(-1) / loss_mask.sum(-1)
        preds = torch.argmin(loss, dim=-1).tolist()
"""

NEW = """        # --- 分块计算 loss（按 option 逐个做 log_softmax）---
        # 原实现一次性把 [bsz, option, seq, vocab] 全部转 float32，
        # openbookqa（4 选项 + 2048 长度）时单份就是约 5 GB，必然 OOM。
        # log_softmax 在 vocab 维逐位置独立，option 之间无耦合，
        # 因此逐 option 计算再 stack，结果与整块计算完全等价。
        loss_list = []
        for _oi in range(option_num):
            _lg = logits[:, _oi]                       # [bsz, seq-1, vocab]
            _lp = torch.nn.functional.log_softmax(_lg.float(), dim=-1)
            _l = -torch.gather(_lp, -1, targets[:, _oi]).squeeze(-1) * loss_mask[:, _oi]
            loss_list.append(_l.sum(-1) / loss_mask[:, _oi].sum(-1))
            del _lg, _lp, _l
        loss = torch.stack(loss_list, dim=-1)          # [bsz, option]
        preds = torch.argmin(loss, dim=-1).tolist()
"""


def ref_impl(logits, targets, loss_mask):
    """原实现：整块 log_softmax"""
    logit_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
    loss = -torch.gather(logit_probs, -1, targets).squeeze(-1) * loss_mask
    loss = loss.sum(-1) / loss_mask.sum(-1)
    return torch.argmin(loss, dim=-1).tolist()


def new_impl(logits, targets, loss_mask, option_num):
    """新实现：按 option 分块"""
    loss_list = []
    for _oi in range(option_num):
        _lg = logits[:, _oi]
        _lp = torch.nn.functional.log_softmax(_lg.float(), dim=-1)
        _l = -torch.gather(_lp, -1, targets[:, _oi]).squeeze(-1) * loss_mask[:, _oi]
        loss_list.append(_l.sum(-1) / loss_mask[:, _oi].sum(-1))
    loss = torch.stack(loss_list, dim=-1)
    return torch.argmin(loss, dim=-1).tolist()


def verify():
    torch.manual_seed(0)
    ok = True
    for trial in range(5):
        bsz, opt, seq, vocab = 2, 4, 9, 31
        logits = torch.randn(bsz, opt, seq, vocab).half()
        targets = torch.randint(0, vocab, (bsz, opt, seq, 1))
        loss_mask = torch.ones(bsz, opt, seq)
        loss_mask[:, :, -1] = 0  # 模拟 padding
        a = ref_impl(logits, targets, loss_mask)
        b = new_impl(logits, targets, loss_mask, opt)
        same = a == b
        ok = ok and same
        print(f"  trial{trial}: 原={a} 新={b} {'一致' if same else '不一致'}")
    print("[%s] 数值等价性验证" % ("PASS" if ok else "FAIL"))
    return ok


def main():
    if not verify():
        print("[FAIL] 数值不等价，未修改文件")
        return 1

    src = open(TARGET, encoding="utf-8").read()
    if "分块计算 loss" in src:
        print("[SKIP] 已打过分块补丁")
        return 0
    if OLD not in src:
        print("[FAIL] 未匹配到预期代码块，未修改")
        i = src.find("logit_probs")
        print(repr(src[i - 200 : i + 300]))
        return 1

    if not os.path.exists(BAK):
        shutil.copy2(TARGET, BAK)
        print(f"[OK] 备份 -> {BAK}")

    out = src.replace(OLD, NEW)
    compile(out, TARGET, "exec")
    open(TARGET, "w", encoding="utf-8").write(out)
    print("[OK] 分块 loss 补丁已写入")

    idx = out.find("分块计算 loss")
    print("--- 修改后 ---")
    for line in out[idx - 260 : idx + 800].splitlines():
        print("  " + line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
