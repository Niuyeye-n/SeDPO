#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成正式实验脚本 v2（score/train/infer），带规模控制与 batch size 参数。

与 v1 (gen_formal.py) 的差异：
  - 新增 --score_bsz / --train_bsz
  - 打分脚本里 batch_size 按任务覆盖（arc_c/openbookqa 用大 bsz，multirc 因文本长用小 bsz）
  - L 与 n_score 按任务分别替换

用 Python 做替换而非 sed，避免 shell 转义地狱（# 与 & 在 sed 替换串里有特殊含义）。

用法:
  python gen_formal_v2.py --model_tag qwen --model_path /root/autodl-tmp/models/qwen2.5-7b \
      --cluster close_qa --ns_arc 300 --ns_obq 300 --ns_mul 300 \
      --score_bsz 12 --train_bsz 8
"""
import argparse
import os
import re

REPO = "/root/autodl-tmp/SeDPO_NIPS25"

# 各任务的 L（论文/官方原值）：arc_c=400, openbookqa=200, multirc=50
L_BY_TASK = {"arc_c": 400, "openbookqa": 200, "multirc": 50}

# 各任务打分 batch size：multirc 文本长（段落级），显存占用大，单独给小值
SCORE_BSZ_BY_TASK = {"arc_c": None, "openbookqa": None, "multirc": 4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_tag", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--cluster", required=True)
    ap.add_argument("--ns_arc", type=int, default=300)
    ap.add_argument("--ns_obq", type=int, default=300)
    ap.add_argument("--ns_mul", type=int, default=300)
    ap.add_argument("--score_bsz", type=int, default=12)
    ap.add_argument("--train_bsz", type=int, default=8)
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

        # 打分 batch size：score.sh 里有 3 处 scorer 调用（step1/2/3），
        # 每处的 batch_size=5 都要按任务改。
        if fname == "score.sh":
            for task, bsz in SCORE_BSZ_BY_TASK.items():
                b = bsz or args.score_bsz
                # 匹配该任务那一段 scorer 命令里的 batch_size=5
                src = re.sub(
                    rf"(task_name={task} model_name=[^\s]+\s*\\?\s*\n?\s*prompt_pool_path=[^\s]*\s*\\?\s*\n?\s*cache_dir)",
                    rf"\1", src)  # 占位，真正的替换在下面按块做
            # 更稳的做法：按 scorer 命令块切分，逐块识别 task_name
            out_lines = []
            block = []
            for line in src.split("\n"):
                block.append(line)
                if line.strip().endswith("\\") or line.rstrip().endswith("\\"):
                    continue
                text = "\n".join(block)
                m = re.search(r"task_name=(\w+)", text)
                if m and "scorer.py" in text:
                    task = m.group(1)
                    b = SCORE_BSZ_BY_TASK.get(task) or args.score_bsz
                    text = re.sub(r"batch_size=\d+", f"batch_size={b}", text)
                    block = [text]
                out_lines.extend(block)
                block = []
            if block:
                out_lines.extend(block)
            src = "\n".join(out_lines)

        # 训练 batch size
        if fname == "train.sh":
            src = re.sub(r"train\.batch_size=\d+", f"train.batch_size={args.train_bsz}", src)

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
    print("--- 打分 batch size ---")
    for line in open(os.path.join(exp, "score.sh"), encoding="utf-8"):
        if "scorer.py" in line or "batch_size=" in line:
            t = re.search(r"task_name=(\w+)", line)
            b = re.search(r"batch_size=(\d+)", line)
            if t and b:
                print(f"    {t.group(1):<12} batch_size={b.group(1)}")


if __name__ == "__main__":
    main()
