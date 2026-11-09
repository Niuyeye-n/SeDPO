# -*- coding: utf-8 -*-
"""
实验规模测算：基于实测吞吐推导 n_score 的合理上限。

打分开销模型（每个 (任务, 模型) 组合）：
  step1: n_score × L          （每个 query 打 L 个候选）
  step2: n_score × L          （choosen_num=1，再打 L 个候选）
  step3: n_score × L          （再打 L 个候选）
  合计 ≈ 3 × n_score × L

推理开销（每个方法）：
  测试集大小 × 1 次 LLM forward（3-shot 拼进 prompt）

用法：python plan_scale.py <吞吐实例/秒> [推理样本/秒]
"""
import sys

SCORE_TPS = float(sys.argv[1]) if len(sys.argv) > 1 else 3.7   # 打分实例/秒（F4b 后实测）
INFER_TPS = float(sys.argv[2]) if len(sys.argv) > 2 else 4.85  # 推理样本/秒

# 各任务：L（论文原值）、候选池大小、测试集大小
TASKS = {
    "arc_c":      {"L": 400, "pool": 1117,  "test": 1172},
    "openbookqa": {"L": 200, "pool": 4957,  "test": 500},
    "multirc":    {"L": 50,  "pool": 10000, "test": 908},
}
MODELS = ["qwen", "llama"]

print("=" * 78)
print(f"规模测算（打分 {SCORE_TPS} 实例/秒，推理 {INFER_TPS} 样本/秒）")
print("=" * 78)
print()
print(f"{'n_score':>8} {'打分总实例':>14} {'打分耗时(h)':>13} {'推理耗时(h)':>13} {'合计(h)':>10}")
print("-" * 78)

for ns in [150, 300, 500, 800, 1000, 1500, 2000]:
    total_instances = 0
    for t, cfg in TASKS.items():
        total_instances += 3 * ns * cfg["L"]
    score_h = total_instances / SCORE_TPS / 3600 / len(MODELS)  # 除以模型数(并行不了，单卡)
    score_h *= len(MODELS)
    # 推理：6 个方法 × 3 数据集 × 2 模型
    infer_samples = 6 * sum(c["test"] for c in TASKS.values()) * len(MODELS)
    infer_h = infer_samples / INFER_TPS / 3600
    print(f"{ns:>8} {total_instances:>14,} {score_h:>13.1f} {infer_h:>13.1f} {score_h+infer_h:>10.1f}")

print("-" * 78)
print()
print("注：单张 4090 串行跑完 2 模型 × 3 数据集 × 6 方法；")
print("    训练/检索/索引时间另计（BERT-base 训练约 10-20 分钟/次）。")
print()
print("=" * 78)
print("规模与论文的差距（论文每个 category 用满 10k 训练数据）")
print("=" * 78)
for t, cfg in TASKS.items():
    for ns in [150, 300, 500, 1000]:
        cov = 100.0 * ns / cfg["pool"]
        print(f"  {t:<12} n_score={ns:<5} 占候选池 {cov:>5.1f}%  "
              f"打分实例 {3*ns*cfg['L']:>9,}")
    print()
