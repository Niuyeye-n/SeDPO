#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从已存在的 preds 文件反算指标并追加到 eval_res 文件。
用途：inference.py 只在跑推理时写 res_file；若 preds 已生成但 res 行缺失，
      无需重跑（重跑要花几十分钟），直接复用 preds 计算即可。
用法：python compute_res.py <preds_json> <res_file> <task_name> <model_name>
"""
import json
import os
import sys

sys.path.insert(0, os.environ.get("REPO", "/root/autodl-tmp/SeDPO_NIPS25"))

from src.utils.metric import compute_scores  # noqa: E402


def main():
    if len(sys.argv) < 5:
        print(__doc__)
        return 2
    preds_path, res_path, task, model = sys.argv[1:5]
    if not os.path.exists(preds_path):
        print(f"[SKIP] preds 不存在: {preds_path}")
        return 1

    data = json.load(open(preds_path))
    if not data:
        print(f"[SKIP] preds 为空: {preds_path}")
        return 1

    scores = compute_scores("simple_accuracy", data)
    n_prompts = max(int(a["n_prompts"]) for a in data)
    line = (
        f"retriever: SeDPO(dp2-r1.0-b0.1); model: {model}; "
        f"task_name: {task}; shot_num: {n_prompts}; scores: {str(scores)}"
    )
    print(f"{task}: {scores}  (n={len(data)}, shot={n_prompts})")

    if os.path.exists(res_path):
        txt = open(res_path).read()
        if f"task_name: {task}" in txt:
            print(f"[OK] {res_path} 已有 {task} 的指标行，不重复写入")
            return 0
    with open(res_path, "a") as f:
        f.write(line + "\n")
    print(f"[OK] 已追加到 {res_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
