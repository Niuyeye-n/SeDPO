#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验规模预算计算器：根据实测速度（秒/打分实例）推算全量实验耗时，
并在超过 12 小时预算时给出降规模建议。

打分实例数定义（与 scorer_dsr 一致）：
  step1: 每条 query 打 L 个候选 → n_query × L
  step2: 每条 query 的 L 个候选里选 choosen_num 个，每个再打 L 个 → n_query × L
  step3: 同上 → n_query × L
  注：step2/step3 的输入条数等于 step1 输出的 query 数（打分后按 query 聚合）。

用法:
  python 20261105_plan_budget.py --speed 12.5 --out plan.json
"""
import argparse
import json

# 三个数据集的候选池规模（已生成的 find_random step1 条数 = query 数）
DATASETS = {
    "arc_c":      {"pool": 1117,  "finder_L": 400, "test": 1165, "class_num": 4},
    "openbookqa": {"pool": 4957,  "finder_L": 200, "test": 500,  "class_num": 4},
    "multirc":    {"pool": 10000, "finder_L": 50,  "test": 4848, "class_num": 2},
}
STEPS = 3          # SeDPO 打分步数
MODELS = 2         # qwen + llama
BUDGET_HOURS = 12.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", type=float, required=True,
                    help="实测打分速度（实例/秒），单条实例含 option_num 次前向")
    ap.add_argument("--infer-speed", type=float, default=4.0,
                    help="实测推理速度（测试样本/秒）")
    ap.add_argument("--train-min", type=float, default=20.0,
                    help="检索器 DPO 训练预估（分钟/组）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    s = args.speed
    print(f"实测打分速度: {s:.2f} 实例/秒")
    print(f"推理速度:     {args.infer_speed:.2f} 样本/秒")
    print("=" * 78)

    # ---- 基线开销（两个模型 × 每个数据集 × 2 种模式）----
    base_h = 0.0
    for name, d in DATASETS.items():
        t = d["test"] / args.infer_speed / 3600
        base_h += t * 2 * MODELS   # zeroshot + random3
    print(f"\n【基线总开销】zeroshot+random3 × 3 数据集 × 2 模型 ≈ {base_h:.2f} 小时")

    # ---- SeDPO 打分开销（全量）----
    print("\n【SeDPO 打分开销（全量规模）】")
    total_score_h = 0.0
    per_group = {}
    for name, d in DATASETS.items():
        n_inst = d["pool"] * d["finder_L"] * STEPS
        h = n_inst / s / 3600
        total_score_h += h * MODELS
        per_group[name] = h
        print(f"  {name:11s}: {d['pool']:6d} query × L={d['finder_L']:3d} × {STEPS} 步 "
              f"= {n_inst:9,d} 实例 → {h:8.2f} 小时/模型 → {h*MODELS:8.2f} 小时（两模型）")

    total_h = base_h + total_score_h + (args.train_min / 60) * 3 * MODELS
    print(f"\n【全量总耗时】基线 {base_h:.2f}h + 打分 {total_score_h:.2f}h + "
          f"训练 {args.train_min*3*MODELS/60:.2f}h ≈ {total_h:.2f} 小时")
    print(f"【预算】{BUDGET_HOURS} 小时 → "
          f"{'超出，需要降规模' if total_h > BUDGET_HOURS else '在预算内'}")

    if total_h > BUDGET_HOURS:
        need = total_h / BUDGET_HOURS
        print(f"\n【降规模建议】需要压缩到 {1/need*100:.0f}% 左右")
        print("  方案 A（推荐）：降 ds_size（打分 query 数），保持 L 不变")
        print("  方案 B：降 finder_L，保持 ds_size 不变")
        print("  注：ds_size 决定打分样本数；L 决定每个 query 的候选数，")
        print("      影响检索器排序信号强度，优先保 L、降 ds_size。")
        print("\n  各数据集降规模后预估（打分小时/两模型）：")
        for name, d in DATASETS.items():
            for factor in (0.5, 0.25, 0.1):
                n_inst = int(d["pool"] * factor) * d["finder_L"] * STEPS
                h = n_inst / s / 3600 * MODELS
                flag = ""
                tot = base_h + h + (args.train_min/60)*3*MODELS
                if tot <= BUDGET_HOURS:
                    flag = "  ← 满足预算"
                print(f"    {name:11s} ds_size×{factor:<5} → {n_inst:9,d} 实例 → {h:7.2f}h{flag}")

    result = {
        "speed_inst_per_s": s,
        "infer_speed_samples_per_s": args.infer_speed,
        "baseline_hours": base_h,
        "sedpo_score_hours_fullscale": total_score_h,
        "total_hours_fullscale": total_h,
        "budget_hours": BUDGET_HOURS,
        "per_dataset_score_hours_per_model": per_group,
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n[OK] 预算结果已写入 {args.out}")


if __name__ == "__main__":
    main()
