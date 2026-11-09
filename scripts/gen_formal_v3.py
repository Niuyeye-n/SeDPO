#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 SeDPO 正式实验脚本 v3（score/train/infer）—— 隔离版

在 gen_sedpo.py 基础上增加：
  - train batch size 参数（上游 train.sh 默认为 1，太慢；16 会 OOM，8 安全）
  - multirc 单独用较小打分 batch size（段落级长文本，显存占用大）
  - 隔离校验加强（train.sh / infer.sh 也一并校验）

隔离目录（避免与服务器上另一并发会话冲突）：
  experiment  : my_data_qwen/experiment_sedpo_{tag}/{cluster}/
  scored      : my_data_qwen/scored_sedpo_{tag}/{cluster}/
  find_random : my_data_qwen/find_random_sedpo_{tag}/{cluster}/
  prompt_pool : my_data_qwen/prompt_pool_sedpo/{cluster}/

规模控制：
  - 候选池保持全量（arc_c=1117 / openbookqa=4957 / multirc=10000）
  - 只限制被打分的 query 数 n_score
  - L 保持论文原值 400/200/50

用法:
  python gen_formal_v3.py --model_tag qwen --model_path /root/autodl-tmp/models/qwen2.5-7b \
      --cluster close_qa --ns_arc 600 --ns_obq 600 --ns_mul 600 --score_bsz 12 --train_bsz 8
"""
import argparse
import os
import re
import sys

REPO = "/root/autodl-tmp/SeDPO_NIPS25"
L_BY_TASK = {"arc_c": 400, "openbookqa": 200, "multirc": 50}
SUF = "sedpo"          # 隔离前缀

# 各任务打分 batch size：multirc 是段落级长文本，显存占用大，单独给小值
SCORE_BSZ_BY_TASK = {"arc_c": None, "openbookqa": None, "multirc": 4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_tag", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--cluster", required=True)
    ap.add_argument("--ns_arc", type=int, default=600)
    ap.add_argument("--ns_obq", type=int, default=600)
    ap.add_argument("--ns_mul", type=int, default=600)
    ap.add_argument("--score_bsz", type=int, default=12)
    ap.add_argument("--train_bsz", type=int, default=8)
    args = ap.parse_args()

    tag, cluster = args.model_tag, args.cluster
    srcexp = f"{REPO}/my_data_qwen/experiment/{cluster}"
    exp = f"{REPO}/my_data_qwen/experiment_{SUF}_{tag}/{cluster}"
    os.makedirs(exp, exist_ok=True)

    ns_by_task = {"arc_c": args.ns_arc, "openbookqa": args.ns_obq, "multirc": args.ns_mul}

    for fname in ("score.sh", "train.sh", "infer.sh"):
        src = open(os.path.join(srcexp, fname), encoding="utf-8").read()

        # ---- 隔离目录替换（顺序要紧：先长后短）----
        src = src.replace(f"my_data_qwen/experiment/{cluster}",
                          f"my_data_qwen/experiment_{SUF}_{tag}/{cluster}")
        src = src.replace(f"my_data_qwen/prompt_pool/{cluster}",
                          f"my_data_qwen/prompt_pool_{SUF}/{cluster}")
        src = src.replace("my_data_qwen/scored", f"my_data_qwen/scored_{SUF}_{tag}")
        src = src.replace("my_data_qwen/find_random", f"my_data_qwen/find_random_{SUF}_{tag}")
        # 未带 cluster 子目录的 prompt_pool 路径（train.sh 用的是目录）
        src = src.replace("my_data_qwen/prompt_pool ",
                          f"my_data_qwen/prompt_pool_{SUF} ")

        # ---- 模型路径 ----
        src = src.replace("/root/autodl-tmp/models/qwen2.5-7b", args.model_path)

        # ---- 规模参数：逐任务替换 L 与 n_score ----
        for task, ns in ns_by_task.items():
            src = re.sub(
                rf"task_name={task} \+ds_size=\d+ L=\d+",
                f"task_name={task} +ds_size=10000 L={L_BY_TASK[task]} +n_score={ns}",
                src)

        # ---- 打分 batch_size：逐任务设置 ----
        if fname == "score.sh":
            # 顺序要紧：先用默认值统一替换（兜底），再逐任务覆盖为特殊值。
            # 若顺序颠倒，兜底正则会把已设定的特殊值又改回默认值。
            src = re.sub(r"batch_size=\d+ task_name=",
                         f"batch_size={args.score_bsz} task_name=", src)
            for task, bsz in SCORE_BSZ_BY_TASK.items():
                if not bsz:
                    continue
                src = re.sub(
                    rf"batch_size=\d+ task_name={task} ",
                    f"batch_size={bsz} task_name={task} ", src)

        # ---- 训练 batch size ----
        if fname == "train.sh":
            src = re.sub(r"train\.batch_size=\d+", f"train.batch_size={args.train_bsz}", src)

        out = os.path.join(exp, fname)
        with open(out, "w", encoding="utf-8") as f:
            f.write(src)
        os.chmod(out, 0o755)

    # ---- 2026-11-06 修 OOM：训练脚本统一加「梯度检查点 + 限制 checkpoint 数量」----
    # 上游在 IF_PREFER_LOSS>=1 时把 batch 再乘 8（8 → 64），64 条 512 长序列
    # 实测峰值 19.78 GiB，加上参考模型后超 24GB，必然 OOM。
    # 梯度检查点数学等价、实测降到 3.95 GiB；num_save_epochs 防止 6 个
    # epoch checkpoint 撑爆 9.7GB 磁盘。逻辑见 patch_train_sh.py（幂等）。
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from patch_train_sh import patch_one
        patch_one(os.path.join(exp, "train.sh"))
    except Exception as e:
        print(f"[WARN] 训练脚本补丁未应用: {e}")

    print(f"[OK] 已生成 {exp}/{{score,train,infer}}.sh")
    print(f"     模型={args.model_path}  打分bsz={args.score_bsz}  训练bsz={args.train_bsz}")
    print("--- 各任务规模参数 ---")
    for line in open(os.path.join(exp, "score.sh"), encoding="utf-8"):
        m = re.search(r"task_name=\w+ \+ds_size=\d+ L=\d+ \+n_score=\d+", line)
        if m:
            print("   ", m.group(0))
    print("--- 打分 batch size ---")
    for line in open(os.path.join(exp, "score.sh"), encoding="utf-8"):
        t = re.search(r"task_name=(\w+)", line)
        b = re.search(r"batch_size=(\d+)", line)
        if t and b and "scorer" not in line:
            print(f"    {t.group(1):<12} batch_size={b.group(1)}")
    print("--- 训练 batch size ---")
    for line in open(os.path.join(exp, "train.sh"), encoding="utf-8"):
        m = re.search(r"train\.batch_size=(\d+)", line)
        if m:
            print(f"    train.batch_size={m.group(1)}")

    # 校验隔离是否彻底
    bad = []
    for fname in ("score.sh", "train.sh", "infer.sh"):
        for line in open(os.path.join(exp, fname), encoding="utf-8"):
            if "my_data_qwen/prompt_pool/" in line and "prompt_pool_sedpo" not in line:
                bad.append(f"{fname}: {line.strip()[:110]}")
            if re.search(r"my_data_qwen/(scored|find_random)(?!_)", line):
                bad.append(f"{fname}: {line.strip()[:110]}")
            if f"experiment/{cluster}" in line:
                bad.append(f"{fname}: {line.strip()[:110]}")
    if bad:
        print("!! 警告：以下行可能未完全隔离：")
        for b in bad[:6]:
            print("   ", b)
    else:
        print("     隔离校验通过（无共享路径残留）")


if __name__ == "__main__":
    main()
