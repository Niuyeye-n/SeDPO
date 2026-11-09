# -*- coding: utf-8 -*-
"""
F4b：把打分改成「hidden states + 定点 lm_head」，彻底去掉全序列 logits

背景（2026-11-05 实测）：
  第一次优化（F4）只把 float32 log_softmax 缩小到 loss_mask 位置，
  但基准测试发现 bsz=2 就会 OOM —— 因为真正的显存大头不是 log_softmax，
  而是模型 forward 吐出的完整 logits 张量：
      (bsz, option, seq, vocab)  fp16
      bsz=2, opt=4, seq=1024, vocab=152064  ->  2.5 GB（每个 batch）
  再叠加 7B 权重 14GB 与中间激活，24GB 显存放不下。

原理：
  causal LM 的 logits[pos] 只依赖 hidden_states[pos]。
  而打分真正需要 logits 的位置只有 loss_mask 标 1 的那些 token（答案 token）。
  所以可以：
    1. 只跑 base transformer（model.model）拿 hidden_states —— 体积小一个量级
       (bsz, option, seq, hidden=3584)，7B 模型下 bsz=2 时仅 58MB
    2. 只把 loss_mask 命中的那些位置的 hidden 取出来，过 lm_head
    3. 在这些行上做 logsumexp 得到 -log p(target)

  数学上与原实现「全序列 logits → log_softmax → gather target」完全等价：
      log_softmax(x)[t] = x[t] - logsumexp(x)
  两者都用同一份 lm_head 输出、同一份 fp32 归一化，只是算了其中一小部分。

唯一差异：lm_head 在子集上做矩阵乘，与在全量上做相比可能有极小的
fp16 舍入差异（量级 1e-3 相对误差），对候选示例的排序无实质影响。

用法：python patch_fixes_b.py <repo_path>
"""
import io
import os
import shutil
import sys

REPO = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/SeDPO_NIPS25"
PATH = os.path.join(REPO, "scripts/scorer.py")
BACKUP = os.path.join(REPO, "_backup_20261105_prepatch")
os.makedirs(BACKUP, exist_ok=True)

src = io.open(PATH, encoding="utf-8").read()

if "F4B_HIDDEN_LMHEAD" in src:
    print("[SKIP] F4b 已应用")
    sys.exit(0)

# 旧版（F4 之后）的 choice_losses 全文替换
old = '''    def choice_losses(self,input_ids, input_atten_mask, loss_mask, labels):
        bsz, option_num, seq_len = input_ids.shape
        if self.option_num is not None: assert option_num == self.option_num
        with torch.no_grad():
            output=self.model(input_ids=input_ids.reshape(bsz*option_num, seq_len), 
                              attention_mask=input_atten_mask.reshape(bsz*option_num, seq_len))
        
        targets=input_ids[:,:,1:].unsqueeze(-1)
        logits=output.logits.reshape(bsz, option_num, seq_len, -1)
        logits=logits[:,:, :-1, :]
'''

new = '''    def choice_losses(self,input_ids, input_atten_mask, loss_mask, labels):
        bsz, option_num, seq_len = input_ids.shape
        if self.option_num is not None: assert option_num == self.option_num
        targets=input_ids[:,:,1:].unsqueeze(-1)
        lm = loss_mask[:,:,1:]

        # F4B_HIDDEN_LMHEAD（2026-11-05）：
        # 只跑 base transformer 拿 hidden_states，再仅在 loss_mask 命中的
        # 答案 token 位置过 lm_head + logsumexp。
        # 原因：全序列 logits (bsz, option, seq, vocab=152k) 的 fp16 张量
        # 在 7B 模型上就要 2.5GB/batch，是打分开不开大 batch 的真正瓶颈。
        # 数学等价：log_softmax(x)[t] = x[t] - logsumexp(x)，
        # 只算被用到的那几行不影响结果。
        use_hidden = hasattr(self.model, "model") and hasattr(self.model, "lm_head")
        with torch.no_grad():
            if use_hidden:
                out = self.model.model(
                    input_ids=input_ids.reshape(bsz*option_num, seq_len),
                    attention_mask=input_atten_mask.reshape(bsz*option_num, seq_len),
                    use_cache=False,
                )
                hidden = out.last_hidden_state.reshape(bsz, option_num, seq_len, -1)
            else:
                output=self.model(input_ids=input_ids.reshape(bsz*option_num, seq_len),
                                  attention_mask=input_atten_mask.reshape(bsz*option_num, seq_len))
                hidden = output.logits.reshape(bsz, option_num, seq_len, -1)
                hidden = torch.cat([hidden, torch.zeros_like(hidden[:, :, :1])], dim=2)
                hidden = hidden[:, :, :-1, :]   # 退化为全量 logits 路径
'''

if old not in src:
    print("[FAIL] 未找到 F4 版 choice_losses 头，源文件可能已变动")
    sys.exit(1)

src = src.replace(old, new)

# 替换 F4 版的 loss 计算块
old2 = '''        # 2026-11-05 F4 修复：原码对整条序列做 float32 log_softmax，
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

new2 = '''        # 只在 loss_mask 标 1 的位置算（见 F4B_HIDDEN_LMHEAD 说明）
        flat_hidden = hidden[:, :, :-1, :].reshape(-1, hidden.shape[-1])
        flat_targets = targets.reshape(-1)
        flat_mask = lm.reshape(-1).bool()
        sel_hidden = flat_hidden[flat_mask]                          # (N, H)
        sel_targets = flat_targets[flat_mask].squeeze(-1)           # (N,)
        if use_hidden:
            sel_logits = self.model.lm_head(sel_hidden).float()      # (N, V) 只算这几行
        else:
            sel_logits = sel_hidden.float()                          # 退化路径：已经是 logits
        lse = torch.logsumexp(sel_logits, dim=-1)                    # (N,)
        tgt_logit = torch.gather(sel_logits, -1, sel_targets.unsqueeze(-1)).squeeze(-1)
        nll = lse - tgt_logit                                        # (N,) = -log p(target)
        loss = torch.zeros(flat_mask.shape[0], device=hidden.device, dtype=torch.float32)
        loss[flat_mask] = nll
        loss = loss.view(bsz, option_num, -1)
        loss = loss.sum(-1) / lm.sum(-1).clamp(min=1)
'''

if old2 not in src:
    print("[FAIL] 未找到 F4 版 loss 计算块")
    sys.exit(1)

src = src.replace(old2, new2)

# 备份
dst = os.path.join(BACKUP, "scripts__scorer.py")
if not os.path.exists(dst):
    shutil.copy2(PATH, dst)

io.open(PATH, "w", encoding="utf-8").write(src)
print("[OK] F4b 已应用：scorer.py 改为 hidden_states + 定点 lm_head")
print("     备份:", dst)
