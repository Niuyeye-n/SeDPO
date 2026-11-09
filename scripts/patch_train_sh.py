#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
给已生成的 train.sh 加两件事（幂等）：
  1) export GRAD_CKPT=1        —— 启用梯度检查点，解决 24GB 显存 OOM
  2) train.num_save_epochs=100 —— 减少 epoch checkpoint，避免撑爆 9.7GB 磁盘

为什么改文件而不是改环境变量：
  run_sedpo_v3.sh 的编排进程已经启动，环境变量在启动时就固定了，
  改脚本不会影响正在运行的进程；已生成的 train.sh 会在训练阶段被直接执行。

用法:
  python patch_train_sh.py <train.sh 路径> [...]
  python patch_train_sh.py --glob /root/autodl-tmp/SeDPO_NIPS25/my_data_qwen
"""
import glob
import io
import os
import re
import sys

MARK = "# [PATCH-TRAIN-20261106] 梯度检查点 + 限制 checkpoint 数量"
EXPORT_LINE = "export GRAD_CKPT=${GRAD_CKPT:-1}   " + MARK


def patch_one(path):
    with io.open(path, "r", encoding="utf-8") as f:
        src = f.read()

    orig = src
    changed = []

    # 1) 插入 export（放在 set -e 之后，若没有 set -e 则放最前面）
    if "GRAD_CKPT" not in src:
        if re.search(r"^set -e\s*$", src, flags=re.M):
            src = re.sub(r"^(set -e\s*)$", r"\1\n\n" + EXPORT_LINE, src, count=1, flags=re.M)
        else:
            src = EXPORT_LINE + "\n" + src
        changed.append("GRAD_CKPT")

    # 2) 限制 checkpoint 数量
    if "num_save_epochs" not in src:
        src, n = re.subn(r"(train\.num_train_epochs=\d+)", r"\1 train.num_save_epochs=100", src, count=1)
        if n:
            changed.append("num_save_epochs")

    if src == orig:
        print("[SKIP] 无需修改: %s" % path)
        return 0

    # 备份
    bak = path + ".bak_20261106"
    if not os.path.exists(bak):
        with io.open(bak, "w", encoding="utf-8") as f:
            f.write(orig)

    with io.open(path, "w", encoding="utf-8") as f:
        f.write(src)

    print("[OK] %s  ← %s" % (path, ", ".join(changed)))
    return 0


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2

    if args[0] == "--glob":
        root = args[1]
        files = sorted(glob.glob(os.path.join(root, "experiment_sedpo_*", "*", "train.sh")))
        if not files:
            print("[WARN] 未找到 train.sh: %s" % root)
            return 1
    else:
        files = args

    rc = 0
    for p in files:
        if not os.path.isfile(p):
            print("[WARN] 不存在: %s" % p)
            rc = 1
            continue
        patch_one(p)
    return rc


if __name__ == "__main__":
    sys.exit(main())
