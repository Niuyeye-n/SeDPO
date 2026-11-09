# -*- coding: utf-8 -*-
"""
给 find_random_step1.py 打补丁：新增可选参数 n_score。

目的：**解耦「检索语料池大小」与「被打分的 query 数」**。

背景：
  - find_random_step1.py 里 prompt_pool 就是 train_dataset[:ds_size]，
    同时被打分的 query 也是同一批 → 缩小 ds_size 会同时缩小检索语料，
    导致 SeDPO 的检索难度变化，且与基线 random3（从 prompt_pool 随机抽）语料不一致，
    对照不公平。
  - 本补丁让 prompt_pool 保持全量（检索语料不变、与基线可比），
    只把「需要打分的 query 条数」限制为前 n_score 条。
  - n_score 未设置时行为与原来完全一致（向后兼容）。

用法：python patch_nscore.py <repo_path>
"""
import io
import sys

REPO = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/SeDPO_NIPS25"
PATH = REPO + "/scripts/find_random_step1.py"

src = io.open(PATH, encoding="utf-8").read()

if "n_score" in src:
    print("[SKIP] 补丁已存在")
    sys.exit(0)

old = """def find(cfg):
    random_finder = RandomFinder(cfg)
    data_list = random_finder.train_dataset
    idx_list = list(range(len(random_finder.prompt_pool)))
"""

new = """def find(cfg):
    random_finder = RandomFinder(cfg)
    data_list = random_finder.train_dataset
    idx_list = list(range(len(random_finder.prompt_pool)))

    # 2026-11-05 规模控制：候选池(prompt_pool)保持全量，只限制「被打分的 query 条数」。
    # 目的：算力受限时压缩打分开销，但保持检索语料与基线 random3 完全一致，
    # 保证 SeDPO 与基线的对照公平。未设置 n_score 时行为与原版完全一致。
    n_score = cfg.get("n_score", None)
    if n_score is not None and int(n_score) > 0:
        n_score = int(n_score)
        if n_score < len(data_list):
            print(f"[INFO] 规模控制：打分 query 数 {len(data_list)} -> {n_score}"
                  f"（候选池仍为 {len(random_finder.prompt_pool)} 条，不变）")
            data_list = data_list[:n_score]
"""

assert old in src, "未找到目标代码块，源文件可能已变动"
src = src.replace(old, new)

# 备份
io.open(PATH + ".orig", "w", encoding="utf-8").write(
    io.open(PATH, encoding="utf-8").read())
io.open(PATH, "w", encoding="utf-8").write(src)
print("[OK] 已为 find_random_step1.py 打上 n_score 补丁")
print("     备份:", PATH + ".orig")
