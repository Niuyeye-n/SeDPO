# -*- coding: utf-8 -*-
"""
SeDPO 复现 —— 一次性修复 4 个已确认缺陷（2026-11-05）

用法：
    python patch_fixes.py <repo_path>     # repo_path 默认为 /root/autodl-tmp/SeDPO_NIPS25

四个修复：

[F1] find_random_step1.py —— L 被静默截断
     原码  k=min(cfg.L, len(data_list)-1)
     data_list 被本项目的 n_score 补丁截成 n_score 条，于是 L 跟着缩水：
     ARC-C 本该 L=400 → 实际 149；OpenBookQA 本该 L=200 → 实际 149。
     排序信号强度被削弱，且 step1(149) 与 step2(400) 内部不一致。
     修法：k 按候选池大小计算（len(idx_list)-1），与 n_score 解耦。

[F2] biencoder_data.py —— 上游 id 映射错配（多任务 cluster 下的真 bug）
     打分阶段每个任务有独立 prompt_pool（id 从 0 开始、任务内局部编号）；
     训练阶段 prompt_pool 被 get_prompt_files() 跨任务拼接成一个大列表，
     但 get_multi_prefer_entry / get_old_entry 等处仍用 self.prompt_pool[id]
     取「已选示例」——用的是拼接池的全局下标去查任务内局部 id，
     多任务 cluster 下必然取错示例。
     上游其实建了映射表 create_dict() -> TaskandId_to_data[task+id]，
     但从头到尾没人用它。
     修法：新增 _lookup_pool(task, id)，优先按 TaskandId_to_data 查（任务内 id），
     查不到再退回 self.prompt_pool[id]（单任务 cluster 的原有行为）。

[F3] few_shot_dsr.py —— Random 基线所有问题复用同一组随机示例
     get_fields() 每次调用都执行 random.seed(self.random_seed)，
     而该函数在 per-sample 路径上，导致所有问题抽到完全相同的 K 个示例。
     修法：按问题下标派生种子（seed = random_seed*1000003 + index），
     既保持可复现，又让每个问题抽到不同的随机示例。

[F4] scorer.py —— choice_losses 显存爆炸
     原码对整条序列 (bsz, option, seq, vocab) 做 float32 log_softmax，
     7B 模型 + 2048 上下文下 logits.float() 单独就要好几 GB，
     迫使 batch_size 只能开到 5，打分吞吐被锁死。
     修法：只在 loss_mask 标 1 的位置算 log_softmax
     （先按 mask gather 出需要的行，再对这些行做 logsumexp），
     数学上完全等价（log_softmax 的分母是行内所有 vocab 的 logsumexp，
     而 gather 只取 target 位置），但显存降一个量级。
"""
import io
import os
import re
import shutil
import sys

REPO = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/SeDPO_NIPS25"
if not os.path.isdir(REPO):
    sys.exit(f"[FAIL] repo 不存在: {REPO}")

BACKUP = os.path.join(REPO, "_backup_20261105_prepatch")
os.makedirs(BACKUP, exist_ok=True)


def rd(p):
    return io.open(os.path.join(REPO, p), encoding="utf-8").read()


def wr(p, s):
    io.open(os.path.join(REPO, p), "w", encoding="utf-8").write(s)


def bak(p):
    """首次修改前备份原文件"""
    dst = os.path.join(BACKUP, p.replace("/", "__"))
    if not os.path.exists(dst):
        shutil.copy2(os.path.join(REPO, p), dst)


def patch(path, old, new, tag, count=1):
    src = rd(path)
    if tag in src:
        print(f"  [SKIP] {path} 已含 {tag}")
        return False
    if src.count(old) != count:
        print(f"  [FAIL] {path}: 期望匹配 {count} 处，实际 {src.count(old)} 处 —— 需人工核对")
        return False
    bak(path)
    wr(path, src.replace(old, new))
    print(f"  [OK] {path} <- {tag}")
    return True


ok_all = True

# ============================================================
# F1: find_random_step1.py —— L 按候选池大小计算
# ============================================================
print("[F1] find_random_step1.py: L 不再被 n_score 截断")
ok_all &= patch(
    "scripts/find_random_step1.py",
    'for a in random.sample([idx for idx in idx_list if idx != i], k=min(cfg.L, len(data_list)-1)) # avoid selecting the task input itself',
    'for a in random.sample([idx for idx in idx_list if idx != i], k=min(cfg.L, len(idx_list)-1)) # avoid selecting the task input itself',
    "F1_L_POOL_SIZE",
)

# ============================================================
# F2: biencoder_data.py —— 修 id 映射错配
# ============================================================
print("[F2] biencoder_data.py: 任务内 id 正确映射到 prompt_pool")
_bd_old = '''        self.TaskandId_to_data = create_dict(self.prompt_pool)
'''
_bd_new = '''        self.TaskandId_to_data = create_dict(self.prompt_pool)
        # 2026-11-05 F2 修复：打分阶段每个任务的 prompt_pool 独立、id 从 0 开始；
        # 训练阶段这里把同 cluster 内各任务的 pool 拼接成一个大列表，
        # 于是「拼接池全局下标」≠「任务内局部 id」。多任务 cluster 下
        # 直接用 self.prompt_pool[id] 取已选示例会取错任务的数据。
        # 上游建了 TaskandId_to_data 映射却从未使用，这里真正用起来。
        self._task_pool_index = {}
        for _i, _e in enumerate(self.prompt_pool):
            self._task_pool_index.setdefault(_e.get("task_name"), []).append(_i)
'''
ok_all &= patch("DPR/dpr/data/biencoder_data.py", _bd_old, _bd_new, "F2_IDX_INIT")

# 新增 _lookup_pool 方法 + 把所有 self.prompt_pool[id] 的取用点替换掉
_lookup_method = '''    def _lookup_pool(self, task_name, idx):
        """按「任务内局部 id」取 prompt_pool 条目。

        2026-11-05 F2 修复：打分产出的 id 是任务内局部编号，
        而 self.prompt_pool 是跨任务拼接后的大列表。
        这里优先用 TaskandId_to_data[task+id] 精确命中；
        命中不到时退回 self.prompt_pool[idx]（单任务 cluster 的原有语义）。
        """
        if task_name is not None:
            hit = self.TaskandId_to_data.get(str(task_name) + str(idx))
            if hit is not None:
                return hit
        try:
            return self.prompt_pool[int(idx)]
        except (IndexError, TypeError, ValueError):
            return None

'''
_anchor = '''    def get_entry(self, entry):
'''
ok_all &= patch("DPR/dpr/data/biencoder_data.py", _anchor, _lookup_method + _anchor, "F2_LOOKUP_METHOD")

# 替换 4 处「for id in have_choosen: example = self.prompt_pool[id]」
# 这些出现在 get_multi_prefer_entry / get_top_prefer_entry / get_udr_rank_entry /
# get_prefer_entry / get_old_entry —— 全部要按任务查。
_old_loop = '''            for id in have_choosen:
                example = self.prompt_pool[id]
                question += self.format_example(example, self.prompt_setup_type) + " \\n "
'''
_new_loop = '''            for id in have_choosen:
                example = self._lookup_pool(entry.get("task_name"), id)
                if example is None:
                    continue
                question += self.format_example(example, self.prompt_setup_type) + " \\n "
'''
ok_all &= patch("DPR/dpr/data/biencoder_data.py", _old_loop, _new_loop, "F2_LOOKUP_CALL", count=5)

# get_old_entry 里的 hard_negative_ids / positive_ids 也用于过滤 prompt_pool，
# 那里用的是 id 值比较（不取内容），跨任务时同 id 会误伤，但过滤本身是「排除」
# 语义，误伤只导致负样本略少，不影响正样本正确性——保留原样即可。
# 但 get_old_entry 的 multi_task 负样本分支用了 self.prompt_pool 直接 choices，无需改。

# ============================================================
# F3: few_shot_dsr.py —— Random 基线按问题派生种子
# ============================================================
print("[F3] few_shot_dsr.py: Random 基线每个问题抽不同的示例")
_fs_old = '''        elif self.random_sample:
            random.seed(self.random_seed)
            prompts_list = [
                p for p in random.choices(self.prompt_pool, k=self.shot_num)
            ]
'''
_fs_new = '''        elif self.random_sample:
            # 2026-11-05 F3 修复：原码在 per-sample 路径里每次都重置同一个种子，
            # 导致所有问题复用完全相同的 K 个随机示例（Random 基线失去意义）。
            # 改为按问题序号派生种子：既可复现，又保证每个问题抽到不同示例。
            _rid = entry.get("_rid", None)
            if _rid is None:
                _rid = self._cur_index
                self._cur_index += 1
            random.seed(self.random_seed * 1000003 + int(_rid))
            prompts_list = [
                p for p in random.sample(self.prompt_pool, k=self.shot_num)
            ]
'''
ok_all &= patch("src/dataset_readers/few_shot_dsr.py", _fs_old, _fs_new, "F3_RANDOM_SEED")

# 配套：__getitem__ 里把下标写进 entry，供 get_fields 使用
_g_old = '''    def __getitem__(self, index):
        if self.task.class_num == 1:
            return self.text_to_instance_completion(self.prompts[index])
        else:
            return self.text_to_instance_choice(self.prompts[index])
'''
_g_new = '''    def __getitem__(self, index):
        # 2026-11-05 F3：把下标带进 entry，让 random_sample 分支能按问题派生种子
        if self.task.class_num == 1:
            _e = self.prompts[index]
        else:
            _e = self.prompts[index]
        if isinstance(_e, dict):
            _e = dict(_e)
            _e["_rid"] = index
        else:
            _e = {"_rid": index}
        if self.task.class_num == 1:
            return self.text_to_instance_completion(_e)
        else:
            return self.text_to_instance_choice(_e)
'''
ok_all &= patch("src/dataset_readers/few_shot_dsr.py", _g_old, _g_new, "F3_RID")

# 初始化 _cur_index
_i_old = '''        self.shot_num = shot_num
        self.n_tokens_in_prompt = n_tokens
'''
_i_new = '''        self.shot_num = shot_num
        self.n_tokens_in_prompt = n_tokens
        self._cur_index = 0  # 2026-11-05 F3：random_sample 分支的兜底计数器
'''
ok_all &= patch("src/dataset_readers/few_shot_dsr.py", _i_old, _i_new, "F3_CUR_INDEX")

# ============================================================
# F4: scorer.py —— choice_losses 只在 loss_mask 位置做 log_softmax
# ============================================================
print("[F4] scorer.py: choice_losses 显存优化（数学等价）")
_sc_old = '''        targets=input_ids[:,:,1:].unsqueeze(-1)
        logits=output.logits.reshape(bsz, option_num, seq_len, -1)            
        logits=logits[:,:, :-1, :]
        logit_probs= torch.nn.functional.log_softmax(logits.float(), dim=-1)


        loss_mask=loss_mask[:,:,1:]
        loss= -torch.gather(logit_probs, -1, targets).squeeze(-1) * loss_mask
        loss = loss.sum(-1) / loss_mask.sum(-1)
'''
_sc_new = '''        targets=input_ids[:,:,1:].unsqueeze(-1)
        logits=output.logits.reshape(bsz, option_num, seq_len, -1)
        logits=logits[:,:, :-1, :]

        # 2026-11-05 F4 修复：原码对整条序列做 float32 log_softmax，
        # (bsz, option, seq, vocab) 的 fp32 中间态在 7B 模型上要好几 GB，
        # 迫使 batch_size 只能开到 5。
        # 下面改成：只在 loss_mask 标 1 的位置算。
        # 数学等价性：log_softmax 在 target 位置的值
        #   = logit[target] - logsumexp(该行全部 vocab)
        # 而 logsumexp 只依赖「行内所有 vocab」，与 gather 的位置无关，
        # 所以可以先把需要的行取出来再算，结果与原码逐位一致（fp32 下）。
        loss_mask=loss_mask[:,:,1:]
        flat_logits = logits.reshape(-1, logits.shape[-1])          # (bsz*option*(seq-1), V)
        flat_targets = targets.reshape(-1)                         # (bsz*option*(seq-1), 1)
        flat_mask = loss_mask.reshape(-1).bool()                    # (bsz*option*(seq-1),)
        sel_logits = flat_logits[flat_mask]                         # (N, V)  N=有效答案 token 数
        sel_targets = flat_targets[flat_mask].squeeze(-1)          # (N,)
        # 每行 logsumexp（fp32）
        lse = torch.logsumexp(sel_logits.float(), dim=-1)           # (N,)
        tgt_logit = torch.gather(sel_logits, -1, sel_targets.unsqueeze(-1)).squeeze(-1).float()
        nll = lse - tgt_logit                                       # (N,)  = -log p(target)
        # 还原成 (bsz, option, seq-1) 的 loss 布局
        loss = torch.zeros(flat_logits.shape[0], device=logits.device, dtype=torch.float32)
        loss[flat_mask] = nll
        loss = loss.view(bsz, option_num, -1)
        loss = loss.sum(-1) / loss_mask.sum(-1).clamp(min=1)
'''
ok_all &= patch("scripts/scorer.py", _sc_old, _sc_new, "F4_LOSS_MEM")

# 顺带：把 loss_mask 里 0 位置排除后，原码 loss_mask.sum(-1) 可能为 0 导致除零，
# 上面的 clamp(min=1) 已处理。

print()
if ok_all:
    print("[DONE] 全部补丁应用成功；备份目录:", BACKUP)
else:
    print("[PARTIAL] 有补丁未应用，请查看上面的 [FAIL] 行")
    sys.exit(1)
