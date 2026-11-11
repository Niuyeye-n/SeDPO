# 生成式评测（Arc-C / OpenBookQA）实现说明

> 2026-11-11
> 分支：`feat/eval-dialog-20261111`
> 目的：在 CoopES 的评测口径下（让 reader 直接输出选项字母）评估 SeDPO 检索的 few-shot 示例

---

## 1. 背景

原仓库 `scripts/inference.py` 用的是**选项似然比较**：把每个选项拼成一条序列，算 `P(选项 | prompt)`，取最大。

CoopES 论文用的是**生成式**：prompt 末尾留 `Answer:`，让模型直接输出答案。两者差距很大——同一批题、同一个检索器、同一个 reader，似然评测只有 52.88%，生成式能到 87%+。

因此需要一套独立的生成式评测，且必须解决一个隐藏问题：**原仓库的 prompt 模板在 Qwen2.5-Instruct 上会失效**。

---

## 2. 核心发现：三种格式的差距

同一个检索器、同一批 100 条题、同一个 reader（Qwen2.5-7B-Instruct）：

| Prompt 格式 | acc | 预测分布 |
|---|---:|---|
| `plain`：4 道题连同选项、答案顺序拼成一段 | **23.0%** | `{0:100}` 全选 A |
| `numbered`：`Example 1/2/3/4` 显式编号 | 54.0% | `{0:61}` 偏斜 |
| `shortans`：只给 `题干 -> 字母` | 72.0% | 均匀 |
| 无示例（zero-shot） | 72.0% | 均匀 |
| **`dialog`：多轮对话（user 问题 / assistant 字母）** | **88.0%** | 均匀 ✅ |

**`plain` 就是原仓库 `se2_prompts_for_*` 里 instruction 的结构**（`Question: xxx Answer: yyy` 顺序拼接）。直接把这种 prompt 喂给 chat 模型，会让它进入「续写下一道题」的模式，输出 `Question: ...` 而不是答案。

---

## 3. 三个根因

### 3.1 `Question:` 前缀触发续写

用 `pinpoint.py` 做单变量隔离（10 条，`max_new_tokens=40`）：

| 配置 | 结果 |
|---|---|
| 长 system + `Question:` 前缀 | ❌ `Question: A student is conducting...` |
| 无 system + `Question:` 前缀 | ❌ 同上 |
| **短 system + 无 `Question:` 前缀** | ✅ 3/3 全对 |
| 无 system + 无前缀 + `Reply with only the letter` | ✅ 2/3 出字母 |
| `Question:` 前缀 + `tail="The answer is"` | ❌ 仍续写 |

**结论：`Question:` 这个前缀本身会让 mock 模型把它当成"题目列表的一部分"。**

### 3.2 `max_new_tokens` 太小

模型习惯先说完整句再给答案：

```
The most likely effect of an increase in rotation due to a meteorite impact is:

C. Planetary days will become shorter.
```

从开头到字母约需 **30 token**。设 8 会被截断在 `The most likely effect of an increase in`，解析不到字母。

### 3.3 多题挤在一条消息里 → 模型分不清哪道要答

`plain` 格式把 4 道题 + 待测题全塞进一条 user 消息，模型无法判断"最后这道才是我要答的"。改成多轮对话后，每条 `assistant` 消息都是一个明确的"答题示范"，待测题作为最后一条 `user`，模型不会再混淆。

---

## 4. 最终方案：`dialog` + 约束解码

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    # 重复 4 次：4 条检索示例
    {"role": "user",      "content": "{示例题干}\n\nA. ...\nB. ...\nC. ...\nD. ..."},
    {"role": "assistant", "content": "C"},
    ...
    # 待测题
    {"role": "user",      "content": "{待测题干}\n\nA. ...\nB. ...\nC. ...\nD. ...\n\nAnswer:"},
]
```

**约束解码**：用 `LogitsProcessor` 把首个生成 token 的 logits 限制为只允许 `A/B/C/D`（含 `" A"`、`"(A)"` 等变体）。

好处：
- 有效率从 49%（自由生成时一半模型不吐字母）拉到 **100%**
- `max_new_tokens=1` 即可，速度极快
- 等价于"让模型在 4 个选项里选一个"，与似然评测同构，但用的是 chat 格式

---

## 5. 实测结果

### ARC-C 全量 1165 条（Qwen2.5-7B-Instruct，k=4）

| 配置 | acc | Weighted-F1 |
|---|---:|---:|
| SeDPO 检索 4 条 | **87.21** | 87.19 |
| 随机 4 条 | **88.67** | 88.70 |
| 无示例 | 75.97 | 76.16 |

### 统计显著性（n=1165，SE ≈ 1.0 个百分点）

| 对比 | 差值 | 判断 |
|---|---:|---|
| 有示例 vs 无示例 | +11.2 点（≈11 SE） | 极显著 |
| SeDPO vs Random | −1.46 点（≈1.5 SE） | **不显著** |

**在 ARC-C 上，SeDPO 检索器没有产生可测增益。** 随机示例甚至略高，但在噪声范围内。

---

## 6. 脚本清单

| 脚本 | 用途 |
|---|---|
| **`eval_dialog.py`** | **最终评测脚本**。三配置（SeDPO / Random / Zero）× dialog 格式 × 约束解码。支持 `--prompt_file` `--n` `--out` |
| `style_cmp.py` | 四种 prompt 格式对比（plain / numbered / shortans / dialog） |
| `dlg_ab.py` | dialog 格式下 SeDPO vs Random vs Zero 的快速对照（N=200） |
| `infer_gen.py` | 生成式评测骨架，提供 `load_blocks()` / `parse()` 等公共函数，被其他脚本 import |
| `final_eval.py` | 早期版本（结构化 ctxs + 三种配置），保留作参考 |
| `pinpoint.py` | 根因诊断：隔离 `Question:` 前缀 / system / max_new_tokens 的影响 |
| `zero_shot_eval.py` | 全量 zero-shot 基线（自由生成，含错误样例） |
| `check_qwen.py` | 判定模型是 base 还是 instruct + 权重完整性校验 |
| `verify_order.py` | 校验 prompt 文件与 preds 文件是否同序 |
| `inspect_ctxs.py` / `check_ctxs_fields.py` | 检查 prompt 里 `ctxs` / `meta_data` 的结构完整性 |
| `diag_prompt.py` / `dump_prompt.py` | 打印原始 instruction 结构，排查格式问题 |

### 基本用法

```bash
# 最终评测（全量）
python scripts/eval_dialog.py \
  --prompt_file <experiment>/se2_prompts_for_<cluster>/<task>_prompts_beam_score_0.json \
  --n 0 --out /tmp/eval_result.json

# 快速对照（前 200 条）
python scripts/eval_dialog.py --prompt_file <...> --n 200 --out /tmp/quick.json
```

依赖：`torch`、`transformers`、`scikit-learn`；需 `HF_ENDPOINT` 可访问（或本地模型）。

---

## 7. 已知局限

1. **口径尚未与 CoopES 对齐。** CoopES 论文 ARC-C / Qwen 的 Random 是 **80.20**，本实现跑出 **88.67**，差 8.5 个点。说明测试集子集、prompt 模板或解码方式仍有差异。在对齐之前，本实现的绝对数值**不能直接与 CoopES 表格并排**。

2. **测试集用的是官方全量（ARC-C 1165 / OpenBookQA 500）**，不是 CoopES 的 500 条随机子集（该子集的 id 列表未公开）。

3. **`dialog` 格式把随机基线抬高了约 8 个点**，导致方法间区分度压缩——这是 SeDPO 与 Random 打平的可能原因之一。若要比较方法优劣，需要一个更有区分度的设置。

4. 约束解码会强制模型在 4 个选项中选一个，无法表达"以上都不对"，对某些题目可能失真。

---

## 8. 未做的事

- OpenBookQA 的生成式评测（推理产物已生成，评测待跑）
- Llama3-8B reader（模型已就绪，训练未开始）
- k=3（SeDPO 原论文设置）的对照
- 示例多样性 / 冗余度指标（用于验证"相似度检索选出冗余组合"这一假说）
