#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成正式实验脚本（score/train/infer），带规模控制参数。

关键设计：**候选池保持全量，只限制打分 query 数（n_score）**
  - 候选池 = prompt_pool = 检索语料，与基线 random3 抽样来源一致 → 对照公平
  - n_score = 被打分的 query 数，决定打分开销
  - L = finder_L，每个 query 的候选数，决定排序信号强度（沿用论文/官方原值）

用 Python 做替换而非 sed，避免 shell 转义地狱（# 与 & 在 sed 替换串里有特殊含义）。

用法:
  python gen_formal.py --model_tag qwen --model_path /root/autodl-tmp/models/qwen2.5-7b \
      --cluster close_qa --ns_arc 150 --ns_obq 150 --ns_mul 150
"""
import argparse
import os
import re

REPO = "/root/autodl-tmp/SeDPO_NIPS25"

# 各任务的 L 与 n_score：按 task_name 分别替换，避免串改
L_BY_TASK = {"arc_c": 400, "openbookqa": 200, "multirc": 50}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_tag", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--cluster", required=True)
    ap.add_argument("--ns_arc", type=int, default=150)
    ap.add_argument("--ns_obq", type=int, default=150)
    ap.add_argument("--ns_mul", type=int, default=150)
    args = ap.parse_args()

    tag, cluster = args.model_tag, args.cluster
    srcexp = f"{REPO}/my_data_qwen/experiment/{cluster}"
    exp = f"{REPO}/my_data_qwen/experiment_{tag}/{cluster}"
    os.makedirs(exp, exist_ok=True)

    ns_by_task = {"arc_c": args.ns_arc, "openbookqa": args.ns_obq, "multirc": args.ns_mul}

    for fname in ("score.sh", "train.sh", "infer.sh"):
        src = open(os.path.join(srcexp, fname), encoding="utf-8").read()

        # 模型路径
        src = src.replace("/root/autodl-tmp/models/qwen2.5-7b", args.model_path)
        # 目录隔离：不同模型的中间产物互不覆盖
        src = src.replace(f"my_data_qwen/experiment/{cluster}",
                          f"my_data_qwen/experiment_{tag}/{cluster}")
        src = src.replace("my_data_qwen/scored", f"my_data_qwen/scored_{tag}")
        src = src.replace("my_data_qwen/find_random", f"my_data_qwen/find_random_{tag}")

        # 规模参数：逐任务替换 L 与 n_score
        for task, ns in ns_by_task.items():
            src = re.sub(
                rf"task_name={task} \+ds_size=\d+ L=\d+",
                f"task_name={task} +ds_size=10000 L={L_BY_TASK[task]} +n_score={ns}",
                src)

        out = os.path.join(exp, fname)
        with open(out, "w", encoding="utf-8") as f:
            f.write(src)
        os.chmod(out, 0o755)

    print(f"[OK] 已生成 {exp}/{{score,train,infer}}.sh")
    print(f"     模型={args.model_path}")
    print("--- 各任务规模参数 ---")
    for line in open(os.path.join(exp, "score.sh"), encoding="utf-8"):
        m = re.search(r"task_name=\w+ \+ds_size=\d+ L=\d+ \+n_score=\d+", line)
        if m:
            print("   ", m.group(0))


if __name__ == "__main__":
    main()
