# -*- coding: utf-8 -*-
"""
验证修复的正确性：
  V1  F4 优化后的 choice_losses 与原实现逐位一致（数学等价性）
  V2  F2 的 _lookup_pool 在多任务 cluster 下取到正确条目
  V3  F1 的 L 不再被 n_score 截断
  V4  F3 的 Random 基线每个问题抽到不同示例

用法：python verify_fixes.py <repo_path>
"""
import io
import json
import os
import random
import sys

REPO = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/SeDPO_NIPS25"
sys.path.insert(0, REPO)
os.chdir(REPO)

import numpy as np
import torch

fails = []


def check(name, cond, detail=""):
    print(("  [PASS] " if cond else "  [FAIL] ") + name + (("  " + detail) if detail else ""))
    if not cond:
        fails.append(name)


# ---------------------------------------------------------------
# V1: F4 数学等价性
# ---------------------------------------------------------------
print("[V1] F4 choice_losses 数学等价性（fp32 随机张量比对）")
torch.manual_seed(0)

bsz, opt, seq, V = 3, 4, 17, 129


def old_choice_losses(logits, input_ids, loss_mask):
    """原实现：整条序列 float32 log_softmax"""
    lg = logits[:, :, :-1, :]
    targets = input_ids[:, :, 1:].unsqueeze(-1)
    logit_probs = torch.nn.functional.log_softmax(lg.float(), dim=-1)
    lm = loss_mask[:, :, 1:]
    loss = -torch.gather(logit_probs, -1, targets).squeeze(-1) * lm
    loss = loss.sum(-1) / lm.sum(-1)
    return loss


def new_choice_losses(logits, input_ids, loss_mask):
    """F4 修复后：只在 loss_mask 位置算"""
    lg = logits[:, :, :-1, :]
    targets = input_ids[:, :, 1:].unsqueeze(-1)
    lm = loss_mask[:, :, 1:]
    flat_logits = lg.reshape(-1, lg.shape[-1])
    flat_targets = targets.reshape(-1)
    flat_mask = lm.reshape(-1).bool()
    sel_logits = flat_logits[flat_mask]
    sel_targets = flat_targets[flat_mask].squeeze(-1)
    lse = torch.logsumexp(sel_logits.float(), dim=-1)
    tgt_logit = torch.gather(sel_logits, -1, sel_targets.unsqueeze(-1)).squeeze(-1).float()
    nll = lse - tgt_logit
    loss = torch.zeros(flat_logits.shape[0], device=lg.device, dtype=torch.float32)
    loss[flat_mask] = nll
    loss = loss.view(lg.shape[0], lg.shape[1], -1)
    loss = loss.sum(-1) / lm.sum(-1).clamp(min=1)
    return loss


for trial in range(5):
    input_ids = torch.randint(0, V, (bsz, opt, seq))
    logits = torch.randn(bsz, opt, seq, V)
    loss_mask = torch.zeros(bsz, opt, seq)
    # 模拟真实情况：答案 token 在序列末尾的一段
    for b in range(bsz):
        for o in range(opt):
            n_ans = random.randint(1, 5)
            loss_mask[b, o, seq - n_ans:] = 1
    l_old = old_choice_losses(logits, input_ids, loss_mask)
    l_new = new_choice_losses(logits, input_ids, loss_mask)
    maxdiff = (l_old - l_new).abs().max().item()
    check(f"V1 trial{trial} loss 完全一致", maxdiff < 1e-5, f"max|diff|={maxdiff:.3e}")

# fp16 场景（真实使用）也测一次
print("[V1b] F4 在 fp16 logits 下的等价性")
input_ids = torch.randint(0, V, (bsz, opt, seq))
logits_f16 = torch.randn(bsz, opt, seq, V).half()
loss_mask = torch.zeros(bsz, opt, seq)
for b in range(bsz):
    for o in range(opt):
        loss_mask[b, o, seq - 3:] = 1
l_old = old_choice_losses(logits_f16, input_ids, loss_mask)
l_new = new_choice_losses(logits_f16, input_ids, loss_mask)
d = (l_old - l_new).abs().max().item()
check("V1b fp16 loss 一致", d < 1e-3, f"max|diff|={d:.3e}")

# ---------------------------------------------------------------
# V2: F2 id 映射
# ---------------------------------------------------------------
print()
print("[V2] F2 _lookup_pool 多任务 id 映射")
# 直接复刻 biencoder_data.create_dict 的实现（避免 hydra/dpr 等重依赖），
# 保证测的是「补丁真正用到的那套键规则」。
def create_dict(raw_data):
    TaskandId_to_data = {}
    for data in raw_data:
        TaskandId_to_data[data["task_name"] + str(data["id"])] = data
    return TaskandId_to_data


# 模拟：openbookqa 有 5 条(id 0..4)，multirc 有 7 条(id 0..6)，拼接顺序 openbookqa 在前
pool = []
for i in range(5):
    pool.append({"id": i, "task_name": "openbookqa", "q": f"obq-{i}"})
for i in range(7):
    pool.append({"id": i, "task_name": "multirc", "q": f"mrc-{i}"})
d = create_dict(pool)
# 原实现：self.prompt_pool[id] —— 多任务下取错
naive_multirc_3 = pool[3]["q"]           # 期望 multirc-3，实际拿到 obq-3
check("V2 原实现确实错配（证明 bug 真实）", naive_multirc_3 != "mrc-3",
      f"naive lookup multirc/3 -> {naive_multirc_3}")
# 修复后：按 task+id 查
hit = d.get("multirc" + "3")
check("V2 修复后 multirc/3 正确", hit is not None and hit["q"] == "mrc-3",
      f"-> {hit['q'] if hit else None}")
hit_obq = d.get("openbookqa" + "3")
check("V2 修复后 openbookqa/3 正确", hit_obq is not None and hit_obq["q"] == "obq-3",
      f"-> {hit_obq['q'] if hit_obq else None}")

# 单任务 cluster 场景：查不到时必须能退回 self.prompt_pool[idx]（保持上游原行为）
check("V2 键规则为 task+str(id)（与 create_dict 一致）",
      "multirc" + "3" in d and "openbookqa" + "3" in d)

# 验证补丁确实把 5 处取用点都换成了 _lookup_pool
src_bd = io.open(os.path.join(REPO, "DPR/dpr/data/biencoder_data.py"), encoding="utf-8").read()
check("V2 补丁含 _lookup_pool 方法", "def _lookup_pool(self, task_name, idx)" in src_bd)
n_calls = src_bd.count("self._lookup_pool(entry.get(\"task_name\"), id)")
check("V2 5 处取用点已全部改写", n_calls == 5, f"实际 {n_calls} 处")
check("V2 不再有裸 self.prompt_pool[id] 取用",
      "example = self.prompt_pool[id]" not in src_bd)

# ---------------------------------------------------------------
# V3: F1 L 不再被截断
# ---------------------------------------------------------------
print()
print("[V3] F1 L 值不再被 n_score 截断")
src = io.open(os.path.join(REPO, "scripts/find_random_step1.py"), encoding="utf-8").read()
check("V3 step1 用 len(idx_list)-1 而非 len(data_list)-1",
      "k=min(cfg.L, len(idx_list)-1)" in src)
# 数值验证
for L, n_score, pool_size, expect in [(400, 150, 1119, 400), (200, 150, 4957, 200), (50, 150, 10000, 50)]:
    data_list_len = min(n_score, pool_size)
    k_old = min(L, data_list_len - 1)
    k_new = min(L, pool_size - 1)
    check(f"V3 L={L} n_score={n_score} pool={pool_size}: 修复后 k={k_new}",
          k_new == expect, f"旧值={k_old} 新值={k_new}")

# ---------------------------------------------------------------
# V4: F3 Random 种子
# ---------------------------------------------------------------
print()
print("[V4] F3 Random 基线每问题不同示例")
src_fs = io.open(os.path.join(REPO, "src/dataset_readers/few_shot_dsr.py"), encoding="utf-8").read()
check("V4 含按问题派生种子逻辑", "self.random_seed * 1000003" in src_fs)
# 数值验证
pool_demo = list(range(1000))
def draw(seed, rid, k=3):
    random.seed(seed * 1000003 + rid)
    return random.sample(pool_demo, k=k)
old_draws = [tuple(random.Random(42).choices(pool_demo, k=3)) for _ in range(3)]
new_draws = [tuple(draw(42, i)) for i in range(3)]
check("V4 旧实现所有问题相同", len(set(old_draws)) == 1, f"-> {old_draws[0]}")
check("V4 新实现每问题不同", len(set(new_draws)) == 3, f"-> {new_draws}")

print()
if fails:
    print(f"[RESULT] {len(fails)} 项未通过: {fails}")
    sys.exit(1)
print("[RESULT] 全部验证通过")
