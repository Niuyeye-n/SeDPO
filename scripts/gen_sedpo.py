#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 SeDPO 正式实验脚本（score/train/infer）—— 隔离版

隔离设计（2026-11-05）：
  服务器上可能有另一个并发会话在跑同样的仓库，它会用
  my_data_qwen/experiment_{tag} / scored_{tag} / find_random_{tag}
  以及**共享的** prompt_pool/。
  本脚本改用 `sedpo_` 前缀 + 模型 tag 的目录名，并额外隔离 prompt_pool，
  两个会话的产物互不干扰。

  experiment 目录: my_data_qwen/experiment_sedpo_{tag}/{cluster}/
  scored  目录   : my_data_qwen/scored_sedpo_{tag}/{cluster}/
  find_random 目录: my_data_qwen/find_random_sedpo_{tag}/{cluster}/
  prompt_pool 目录: my_data_qwen/prompt_pool_sedpo/{cluster}/

规模控制（沿用已验证的设计）：
  - 候选池保持全量不变（arc_c=1117 / openbookqa=4957 / multirc=10000）
  - 只限制被打分的 query 数 n_score
  - L 保持论文原值 400/200/50

用法:
  python gen_sedpo.py --model_tag qwen --model_path /root/autodl-tmp/models/qwen2.5-7b \
      --cluster close_qa --ns_arc 300 --ns_obq 300 --ns_mul 300
"""
import argparse
import os
import re
import shutil

REPO = "/root/autodl-tmp/SeDPO_NIPS25"
L_BY_TASK = {"arc_c": 400, "openbookqa": 200, "multirc": 50}
SUF = "sedpo"          # 隔离前缀，避免与另一会话冲突


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_tag", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--cluster", required=True)
    ap.add_argument("--ns_arc", type=int, default=300)
    ap.add_argument("--ns_obq", type=int, default=300)
    ap.add_argument("--ns_mul", type=int, default=300)
    ap.add_argument("--score_bsz", type=int, default=12)
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

        # ---- 打分 batch_size ----
        src = re.sub(r"batch_size=\d+ task_name=", f"batch_size={args.score_bsz} task_name=", src)

        out = os.path.join(exp, fname)
        with open(out, "w", encoding="utf-8") as f:
            f.write(src)
        os.chmod(out, 0o755)

    print(f"[OK] 已生成 {exp}/{{score,train,infer}}.sh")
    print(f"     模型={args.model_path}  打分bsz={args.score_bsz}")
    print("--- 各任务规模参数 ---")
    for line in open(os.path.join(exp, "score.sh"), encoding="utf-8"):
        m = re.search(r"task_name=\w+ \+ds_size=\d+ L=\d+ \+n_score=\d+", line)
        if m:
            print("   ", m.group(0))
    # 校验隔离是否彻底
    bad = []
    for line in open(os.path.join(exp, "score.sh"), encoding="utf-8"):
        if "my_data_qwen/prompt_pool/" in line and "prompt_pool_sedpo" not in line:
            bad.append(line.strip()[:120])
        if re.search(r"my_data_qwen/(scored|find_random)(?!_)", line):
            bad.append(line.strip()[:120])
    if bad:
        print("!! 警告：以下行可能未完全隔离：")
        for b in bad[:5]:
            print("   ", b)
    else:
        print("     隔离校验通过（无共享路径残留）")


if __name__ == "__main__":
    main()
