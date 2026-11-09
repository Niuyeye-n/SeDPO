#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""验证梯度检查点补丁：属性链是否命中 + CPU 上前向/反向是否正常。

在 CPU 上跑小尺寸（batch 4, seq 64），不占用正在做打分的 GPU。
"""
import sys

import torch

sys.path.insert(0, "/root/autodl-tmp/SeDPO_NIPS25")
sys.path.insert(0, "/root/autodl-tmp/SeDPO_NIPS25/DPR")

from dpr.models.biencoder import BiEncoder  # noqa: E402
from dpr.models.hf_models import HFBertEncoder  # noqa: E402
from DPR.train_dense_encoder import _enable_grad_ckpt  # noqa: E402

MODEL = "/root/autodl-tmp/models/bert-base-uncased"


def main():
    q = HFBertEncoder.init_encoder(MODEL)
    c = HFBertEncoder.init_encoder(MODEL)
    be = BiEncoder(q, c)
    be.train()

    print("question_model 类型:", type(be.question_model).__name__)
    print("ctx_model 类型:", type(be.ctx_model).__name__)

    n = _enable_grad_ckpt(be)
    print("启用编码器数:", n)
    # 标志实际挂在内层 BertEncoder 上
    for nm in ("question_model", "ctx_model"):
        m = getattr(be, nm)
        print("%s.encoder.gradient_checkpointing = %s"
              % (nm, getattr(getattr(m, "encoder", None), "gradient_checkpointing", "N/A")))

    if n != 2:
        print("[FAIL] 属性链没命中，补丁无效")
        return 1

    b, L = 4, 64
    ids = torch.randint(0, 2000, (b, L))
    seg = torch.zeros_like(ids)
    mask = torch.ones_like(ids)

    q_vec, ctx_vec = be(ids, seg, mask, ids, seg, mask, "query")
    loss = q_vec.sum() + ctx_vec.sum()
    loss.backward()

    grads = sum(1 for p in be.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    print("有梯度的参数张量数:", grads)
    print("loss =", float(loss))
    if grads == 0:
        print("[FAIL] 反向没有梯度，梯度检查点破坏了计算图")
        return 1

    print("[OK] 梯度检查点可用，前向/反向正常")
    return 0


if __name__ == "__main__":
    sys.exit(main())
