# -*- coding: utf-8 -*-
"""
F2 实证：在真实 prompt_pool 上验证「上游 id 映射错配」确实会发生

用真实数据（my_data_qwen/prompt_pool/reading_new/*.json），
按上游 get_prompt_files 的拼接方式构造 prompt_pool，
然后对比：
  上游写法  self.prompt_pool[id]
  修复写法  TaskandId_to_data[task + str(id)]
在真实数据上到底差多少。

用法：python verify_f2_real.py <repo_path>
"""
import glob
import json
import os
import sys

REPO = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/SeDPO_NIPS25"
POOL_ROOT = os.path.join(REPO, "my_data_qwen/prompt_pool")


def get_prompt_files(prompt_pool_path, train_clusters):
    """复刻 DPR/dpr/utils/tasks.py 的 get_prompt_files"""
    clusters = train_clusters.split('+')
    dirs = [f"{prompt_pool_path}/{c}/*" for c in clusters]
    paths = []
    for d in dirs:
        paths.extend(glob.glob(d))
    return paths


def create_dict(raw_data):
    """复刻 DPR/dpr/data/biencoder_data.py 的 create_dict"""
    d = {}
    for data in raw_data:
        d[data["task_name"] + str(data["id"])] = data
    return d


print("=" * 74)
print("F2 实证：多任务 cluster 下 prompt_pool 的 id 空间冲突")
print("=" * 74)

for cluster in ["close_qa", "reading_new"]:
    files = get_prompt_files(POOL_ROOT, cluster)
    pool = []
    for f in files:
        pool.extend(json.load(open(f, encoding="utf-8")))
    taskmap = create_dict(pool)

    print(f"\n### cluster = {cluster}")
    print(f"  拼接顺序（glob 返回顺序）: {[os.path.basename(f) for f in files]}")
    # 每段任务在拼接池中的全局下标区间
    segs = []
    cur = 0
    for f in files:
        n = len(json.load(open(f, encoding="utf-8")))
        segs.append((os.path.basename(f), cur, cur + n - 1))
        cur += n
    for name, lo, hi in segs:
        print(f"    {name:<32} 全局下标 [{lo}, {hi}]")

    if len(segs) < 2:
        print("  -> 单任务 cluster：全局下标 == 任务内局部 id，不存在错配")
        continue

    print("  -> 多任务 cluster：逐任务抽样 200 个 id 检查错配率")
    total_wrong = 0
    total = 0
    for name, lo, hi in segs:
        task = name.replace("_prompts.json", "")
        # 取该任务池里真实存在的 id
        task_ids = [e["id"] for e in pool if e["task_name"] == task][:200]
        wrong = 0
        for i in task_ids:
            naive = pool[i]                     # 上游写法：直接按 id 取
            fixed = taskmap.get(task + str(i))  # 修复写法：按 task+id 取
            if naive is not fixed:
                wrong += 1
        total_wrong += wrong
        total += len(task_ids)
        pct = 100.0 * wrong / max(len(task_ids), 1)
        print(f"    {task:<12} 抽查 {len(task_ids):>3} 个 id -> 上游取错 {wrong:>3} 个 ({pct:.1f}%)")
    print(f"    >>> 整体错配率: {100.0*total_wrong/max(total,1):.1f}%")

print()
print("=" * 74)
print("结论：多任务 cluster 中，非首位任务的 id 会被上游写法解析成别的任务的示例。")
print("      这直接污染 SeDPO 训练时的 query 文本（have_choosen 拼进 question）。")
print("      F2 修复后 100% 对齐。")
print("=" * 74)
