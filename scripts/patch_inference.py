#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
patch_inference.py — 2026-11-07

问题：scripts/inference.py 的 choice_losses 里有一段死代码——

    logit_probs = log_softmax(logits.float(), dim=-1)
    # test
    probs = softmax(logits.float(), dim=-1)          # ← 只为算 targets_probs
    targets_probs = gather(probs, -1, targets) ...
    targets_probs = where(...)
    targets_probs = cumprod(...)

targets_probs 算完之后**从未被使用**（后面的 loss/preds 全部基于 logit_probs），
但它会额外分配一份 [bsz, option_num, seq_len, vocab] 的 float32 张量。
max_length=2048、vocab=152k 时这一份就是 2.5 GB，正好是 MultiRC 推理
两次 OOM 报错里申请失败的字节数。

改动：只删除这段死代码，保留 logit_probs。predictions 与 loss 的数学完全不变。
"""
import os
import re
import shutil
import sys

TARGET = "/root/autodl-tmp/SeDPO_NIPS25/scripts/inference.py"
BAK = TARGET + ".bak_before_deadcode_20261107"

DEAD = """        # test
        probs = torch.nn.functional.softmax(logits.float(), dim=-1)
        targets_probs = torch.gather(probs, -1, targets).squeeze(-1) * loss_mask
        targets_probs = torch.where(abs(targets_probs)>1e-6, targets_probs, 1.)
        targets_probs = torch.cumprod(targets_probs, -1)[:,:,-1]

"""


def main():
    src = open(TARGET, encoding="utf-8").read()

    if "targets_probs" not in src:
        print("[SKIP] 已无死代码，可能已打过补丁")
        return 0

    if DEAD not in src:
        print("[FAIL] 未匹配到预期文本块，未做修改。请人工核对 93-99 行")
        for i, line in enumerate(src.splitlines()[91:100], start=92):
            print(f"  {i}: {line}")
        return 1

    if not os.path.exists(BAK):
        shutil.copy2(TARGET, BAK)
        print(f"[OK] 已备份 -> {BAK}")

    out = src.replace(DEAD, "")
    open(TARGET, "w", encoding="utf-8").write(out)

    # 校验：语法可编译 + 死代码消失 + logit_probs 仍在
    compile(out, TARGET, "exec")
    assert "targets_probs" not in out, "死代码未清除干净"
    assert "logit_probs = torch.nn.functional.log_softmax" in out, "logit_probs 被误删"

    n = len(re.findall(r"^\s*probs = torch\.nn\.functional\.softmax", out, re.M))
    print(f"[OK] 补丁完成；剩余 softmax(probs) 调用数 = {n}（应为 0）")

    idx = out.splitlines().index(
        "        logit_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)"
    )
    print("--- 修改后上下文 ---")
    for i, line in enumerate(out.splitlines()[idx - 2 : idx + 5], start=idx - 1):
        print(f"  {i + 1}: {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
