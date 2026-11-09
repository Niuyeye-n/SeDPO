#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
给 DPR/train_dense_encoder.py 打「可选梯度检查点」补丁（v2，幂等，可回滚）。

为什么需要（2026-11-06 定位）：
  train_dense_encoder.py 第 ~909 行在 IF_PREFER_LOSS>=1 时把 batch 再乘 8：
      cfg.train.batch_size = ... else cfg.train.batch_size*8
  run_sedpo_v3.sh 传 train.batch_size=8 → 实际 64。
  实测 BERT-base 训练 64 条 512 长序列峰值 19.78 GiB，
  加上参考模型与优化器状态后超 24GB → 两次训练都在首个 iteration OOM。

  为什么不直接砍 batch：
  DPOLoss 在 DIV_REJECT=1 时从「整批候选」里随机抽拒绝样本，
  缩小 batch 会改变负例池规模与难度分布，属于实验语义变更。

  梯度检查点只保存每层输入、反向重算中间激活，
  实测 64 条序列峰值 19.78 → 3.95 GiB（约 1/5），数学上完全等价。

属性链（已核对源码）：
  BiEncoder.question_model / ctx_model  →  HFBertEncoder(BertModel)
  注意不是 question_encoder/ctx_encoder（那是别处的命名）。

用法：
  GRAD_CKPT=1 启用；未设置或为 0 时行为与原始代码完全一致。
  --restore 从 .bak_20261106 回滚。
"""
import io
import os
import shutil
import sys

TARGET = "/root/autodl-tmp/SeDPO_NIPS25/DPR/train_dense_encoder.py"
BAK = TARGET + ".bak_20261106"
MARK = "[PATCH-GRAD-CKPT-20261106]"

HELPER = '''
# ---- {mark} helper ----
def _enable_grad_ckpt(model):
    """在 BiEncoder 的两个 HFBertEncoder 上启用梯度检查点，返回启用个数。"""
    target = model
    if hasattr(target, "module"):          # DataParallel 包装
        target = target.module
    n = 0
    for attr in ("question_model", "ctx_model", "question_encoder", "ctx_encoder"):
        enc = getattr(target, attr, None)
        if enc is None or not hasattr(enc, "gradient_checkpointing_enable"):
            continue
        try:
            # use_reentrant=False 对无梯度输入更稳健（transformers>=4.34 支持）
            enc.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={{"use_reentrant": False}}
            )
        except TypeError:
            enc.gradient_checkpointing_enable()
        n += 1
    return n
# ---- /{mark} helper ----
'''.format(mark=MARK)

CALL = '''
        # ---- {mark} ----
        if int(os.environ.get("GRAD_CKPT", "0")) == 1:
            _n_ck = _enable_grad_ckpt(self.biencoder)
            if _n_ck:
                logger.info("[GRAD_CKPT] 已启用梯度检查点，编码器数=%d（显存降至约1/5，耗时+20~25%%）", _n_ck)
            else:
                logger.warning("[GRAD_CKPT] 未找到任何编码器，梯度检查点未生效")
        # ---- /{mark} ----
'''.format(mark=MARK)

ANCHOR_CALL = "        self.biencoder = model\n"
ANCHOR_HELPER = "@hydra.main(config_path=\"conf\", config_name=\"biencoder_train_cfg\")"


def restore():
    if not os.path.exists(BAK):
        print("[FAIL] 备份不存在，无法回滚: %s" % BAK)
        return 2
    shutil.copyfile(BAK, TARGET)
    print("[OK] 已回滚到备份: %s" % TARGET)
    return 0


def main():
    if "--restore" in sys.argv:
        return restore()

    with io.open(TARGET, "r", encoding="utf-8") as f:
        src = f.read()

    # 若存在旧版（属性名错误的）补丁，先回滚
    if MARK in src:
        print("[INFO] 检测到已有补丁，先回滚后重打")
        restore()
        with io.open(TARGET, "r", encoding="utf-8") as f:
            src = f.read()

    if ANCHOR_CALL not in src:
        print("[FAIL] 未找到锚点: self.biencoder = model")
        return 2
    if ANCHOR_HELPER not in src:
        print("[FAIL] 未找到 helper 插入点: @hydra.main")
        return 2

    # 备份原始文件（只在首次）
    if not os.path.exists(BAK):
        with io.open(BAK, "w", encoding="utf-8") as f:
            f.write(src)

    new = src.replace(ANCHOR_HELPER, HELPER.strip("\n") + "\n\n\n" + ANCHOR_HELPER, 1)
    new = new.replace(ANCHOR_CALL, ANCHOR_CALL + CALL, 1)

    if new == src:
        print("[FAIL] 替换未生效")
        return 2

    try:
        compile(new, TARGET, "exec")
    except SyntaxError as e:
        print("[FAIL] 语法错误，未写入: %s" % e)
        return 3

    with io.open(TARGET, "w", encoding="utf-8") as f:
        f.write(new)

    print("[OK] 补丁已写入: %s" % TARGET)
    print("     备份: %s" % BAK)
    print("     用法: GRAD_CKPT=1 时启用；不设则行为不变")
    return 0


if __name__ == "__main__":
    sys.exit(main())
