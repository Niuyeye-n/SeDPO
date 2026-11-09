#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测算「检索器训练条数」与「训练轮次」的可调范围、耗时影响与质量影响。

背景：
  - 打分阶段（score）产出偏好数据：每个 query 一条记录，内含 top-20 正例 + bottom-20 负例
  - 训练阶段（DPR train_dense_encoder）读这份数据，每条记录展开成多个 (正例, 负例) 训练样本
  - 因此「训练条数」与 n_score 成正比，而「训练轮次」是独立的 num_train_epochs

本脚本只做分析，不改任何服务器状态。
"""
import json
import os
import glob
import sys

REPO = "/root/autodl-tmp/SeDPO_NIPS25"
SCORED = os.path.join(REPO, "my_data_qwen", "scored_sedpo_qwen")

TRAIN_BSZ = 8
EPOCHS_PAPER = 6
WARMUP = 1000
TOP_K = 20          # datasets.top_k=20 → 论文 T=20
HARD_NEG = 20       # train.hard_negatives=20

# 实测吞吐（来自 ARC-C/Qwen step1 打分日志：1.15 batch/s × bsz 12）
TPS_SCORE = 13.8
# 检索器训练实测：BERT-base, 512 seq, 每样本 1 正 + 20 负 = 21 个 pair, bsz=8
# 单步前向 168 pair/step；在 4090 上实测约 2.6 it/s（保守）
TRAIN_ITERS_PER_SEC = 2.6


def hr(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def analyze_scored_structure():
    """检查真实 scored 数据结构：每个 query 的正负例数量"""
    hr("[1] 实测 scored 数据结构（决定训练样本条数）")
    bins = sorted(glob.glob(os.path.join(SCORED, "*", "*.bin")))
    if not bins:
        print("  (暂无 scored bin，跳过)")
        return None
    b = bins[0]
    print(f"  文件: {b}")
    n_rec = 0
    pos_counts, neg_counts = [], []
    sample_keys = None
    try:
        from src.utils.cache_util import BufferedJsonReader
        with BufferedJsonReader(b) as f:
            for datas in f.read():
                for e in datas:
                    n_rec += 1
                    if sample_keys is None:
                        sample_keys = sorted(e.keys())
                    # 统计正/负例数量
                    for key in ("pos", "neg", "true_ctxs", "false_ctxs"):
                        if key in e and isinstance(e[key], list):
                            if key in ("pos", "true_ctxs"):
                                pos_counts.append(len(e[key]))
                            else:
                                neg_counts.append(len(e[key]))
                if n_rec >= 400:
                    break
    except Exception as ex:
        print(f"  读取失败: {ex}")
        return None

    print(f"  记录字段: {sample_keys}")
    print(f"  已读记录数: {n_rec}")
    if pos_counts:
        import statistics as st
        print(f"  正例数/query: min={min(pos_counts)} max={max(pos_counts)} "
              f"均值={st.mean(pos_counts):.1f}")
    if neg_counts:
        import statistics as st
        print(f"  负例数/query: min={min(neg_counts)} max={max(neg_counts)} "
              f"均值={st.mean(neg_counts):.1f}")
    return n_rec


def calc_train_samples(n_score, steps_per_query=3, num_pair=TOP_K):
    """每个 query 产生多少训练样本"""
    # 训练样本 = (query, 正例, 负例) 三元组；每条 scored 记录取 top-K 为正、bottom-K 为负
    # 交叉配对后通常取 num_pair 组；保守按 num_pair 计
    per_query = steps_per_query * num_pair
    total = n_score * per_query
    return total, per_query


def calc_steps(n_score, epochs, train_ratio=0.9):
    samples, per_query = calc_train_samples(n_score)
    train_samples = int(samples * train_ratio)
    steps = train_samples * epochs // TRAIN_BSZ
    return samples, train_samples, steps


def calc_train_time(steps):
    return steps / TRAIN_ITERS_PER_SEC / 3600


def calc_score_time(n_score):
    """两模型 × 三任务 × 三步打分耗时"""
    TASKS = {"arc_c": 400, "openbookqa": 200, "multirc": 50}
    total_inst = 0
    for L in TASKS.values():
        total_inst += 3 * n_score * L
    return total_inst / TPS_SCORE / 3600 * 2  # 两模型


def main():
    analyze_scored_structure()

    hr("[2] 「训练条数」随 n_score 的变化")
    print(f"  {'n_score':>8}{'训练样本数':>14}{'轮次':>6}{'总步数':>10}"
          f"{'warmup占比':>12}{'训练耗时(h)':>13}")
    print("  " + "-" * 66)
    for ns in [150, 200, 300, 450, 600]:
        for ep in [6]:
            s, ts, steps = calc_steps(ns, ep)
            wp = min(WARMUP, steps) / max(steps, 1)
            print(f"  {ns:>8}{s:>14,}{ep:>6}{steps:>10,}{wp:>11.0%}"
                  f"{calc_train_time(steps):>13.2f}")

    hr("[3] 「训练轮次」的影响（n_score=300）")
    print(f"  {'轮次':>6}{'总步数':>10}{'warmup占比':>12}{'训练耗时(h)':>13}"
          f"{'训练耗时(4组)':>15}")
    print("  " + "-" * 58)
    for ep in [1, 2, 3, 4, 6]:
        s, ts, steps = calc_steps(300, ep)
        wp = min(WARMUP, steps) / max(steps, 1)
        print(f"  {ep:>6}{steps:>10,}{wp:>11.0%}{calc_train_time(steps):>13.2f}"
              f"{calc_train_time(steps)*4:>15.2f}")

    hr("[4] 总耗时对比（打分占绝对大头）")
    print(f"  {'配置':<28}{'打分(h)':>10}{'训练(h)':>9}{'推理(h)':>9}{'合计(h)':>10}")
    print("  " + "-" * 68)
    INF_4 = 0.7 * 4
    PRE_4 = 0.3 * 4
    for ns, ep, label in [
        (600, 6, "现状 600 / 6轮"),
        (300, 6, "当前 300 / 6轮(论文值)"),
        (300, 3, "300 / 3轮"),
        (300, 2, "300 / 2轮"),
        (200, 6, "200 / 6轮"),
        (150, 6, "150 / 6轮(原计划)"),
        (150, 3, "150 / 3轮"),
    ]:
        s, ts, steps = calc_steps(ns, ep)
        sc = calc_score_time(ns)
        tr = calc_train_time(steps) * 4
        tot = sc + tr + INF_4 + PRE_4
        print(f"  {label:<28}{sc:>10.1f}{tr:>9.1f}{INF_4:>9.1f}{tot:>10.1f}")

    hr("[5] 关键结论")
    sc300 = calc_score_time(300)
    s6, _, st6 = calc_steps(300, 6)
    _, _, st3 = calc_steps(300, 3)
    tr6 = calc_train_time(st6) * 4
    tr3 = calc_train_time(st3) * 4
    print(f"  · 打分(300) ≈ {sc300:.1f}h，是训练的 {sc300/tr6:.0f} 倍")
    print(f"  · 训练 6轮→3轮，省 {tr6-tr3:.1f}h（占总 {sc300+tr6:.1f}h 的 "
          f"{100*(tr6-tr3)/(sc300+tr6):.1f}%）")
    print(f"  · 训练轮次降到 1 轮省最多也只有 {tr6-calc_train_time(calc_steps(300,1)[2])*4:.1f}h")
    print("  · 结论：调轮次省时间极有限；真正的大头是打分（= 训练条数的上游）")


if __name__ == "__main__":
    main()
