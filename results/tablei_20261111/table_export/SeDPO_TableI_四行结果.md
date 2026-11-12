# Table I：新增 SeDPO 四行

> 仅运行 SeDPO。原表其他方法的数值保留为论文报告值，不重新运行。

| Dataset | LLMs | Method | Accuracy (%) | Weighted-F1 (%) | Macro-F1 (%) | 状态 |
|---|---|---|---:|---:|---:|---|
| OpenBookQA | Llama3-8B | SeDPO† | 89.60 | 89.6106 | 89.4638 | 已完成并核验 |
| OpenBookQA | Qwen2.5-7B | SeDPO† | 90.40 | 90.5061 | 90.2406 | 已完成并核验 |
| ARC-C | Llama3-8B | SeDPO† | 82.80 | 82.8012 | 82.8610 | 已完成并核验 |
| ARC-C | Qwen2.5-7B | SeDPO† | 88.40 | 88.3901 | 88.4702 | 已完成并核验 |

## 本次设置

- 两个数据集各固定 500 道官方 test 题；每个任务的检索候选池固定为 50 条官方 train 示例，k=4。原始题目 ID 和文件哈希均已保存。
- ARC-C 的 500 题按固定 seed/ID 哈希选取；OpenBookQA 包含完整 500 道 test。两者都未声称就是原表那批未知 ID。
- Reader 使用当前服务器的 Qwen2.5-7B-Instruct 和 Llama3-8B-Instruct，本地权重与 tokenizer 的 SHA256 记入结果。
- 两个 reader 分别构造自身的偏好数据并训练检索器，L=50、n_score=300、6 epochs、生效 batch=64、训练 seed=12345。
- ARC-C 学习率 1e-5，OBQA 为上游任务默认 3e-5；warmup 取实际总更新步数的约 10%，不改变 batch 或其他训练超参。
- Reader 使用同一评测程序、多轮四示例、确定性单 token 字母约束、fp16、SDPA、batch=1；输入不静默截断。OBQA 示例与目标均提供 fact。
- eager/fp16 曾出现非有限数值，该失败运行及原配置被保留，未输出有效分数；SDPA 修订只修复数值运行问题，不依据测试准确率选择配置。

## 插入原表时的表注

† SeDPO 为本次新运行结果；其余方法为原论文报告值。SeDPO 使用固定 500 题和 50 条候选、k=4。由于未取得原实验的样本 ID、模型制品与完整推理配置，二者为外部参考比较，不能据此宣称严格同设置的配对优劣或显著性。

将每个模型的 multirow 行数由 6 改为 7，每个数据集由 12 改为 14。四条 LaTeX 插入行见同目录 sedpo_insert_rows.tex；Macro-F1 保留在实验记录中，主表列沿用 Accuracy 与 Weighted-F1。

所有数字按实测填写。即使 SeDPO 超过 CoopES，也保留结果，不通过更换样本、模板或检查点制造较低分数。
