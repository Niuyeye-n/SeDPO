# 本次发布验证记录

日期：2026-11-12。此记录区分历史模型运行与本次代码交付检查，不用 CPU 检查冒充完整 GPU 重跑。

## 已通过

| 检查 | 环境 | 结果 |
|---|---|---|
| 82 个实际运行源文件指纹 | 本地及 AutoDL 无卡实例 | 全部匹配 |
| 4 组结果完成哈希、题目、答案、示例及指标重算 | 本地及 AutoDL 无卡实例 | 每组 500 唯一 ID，2000 条预测；重算与归档完全一致 |
| 训练输入完整性与 query 内容隔离 | 已下载的四组真实输入 | 12 个输入文件全部匹配 SHA256；每组 810/90 条记录、270/30 个不同 query，无交集 |
| 17 项 CPU 测试 | 本地 Python、远端实测 venv | 全部通过；含篡改结果、gold、示例顺序及输出 token 的拒绝测试 |
| 原 DPR 冻结数据适配 | 远端实际运行依赖 | 两任务各 50 候选、500 测试 query；阻止 selector 打开 gold 文件；元数据不含测试答案 |
| 8 个 Hydra 入口配置 | 远端实际运行依赖 | find1/2/3、scorer、merge、train、embedding、beam retrieval 全部通过 |
| 实际 DPR 数据展开 | 远端原训练输入 | Qwen ARC=9152、Qwen OBQA=11567、Llama ARC=9355、Llama OBQA=11366，与日志和预算一致 |
| 新 setup shell 语法 | 本地 Git Bash | bash -n 通过 |

远端详细输出见 [verification/cpu_validation.txt](verification/cpu_validation.txt)。原训练脚本的 legacy launcher 会去掉所有 `--` 前缀，无法直接拿它的 `--cfg` 当检查入口；本发布只通过独立 Hydra compose 检查训练覆盖配置，不改该历史训练实现。其余七个配置入口使用原脚本的 --cfg job --resolve。

## 没有声称通过的项目

- 没有从零安装新 GPU 环境并重跑四套模型；本次实例处于无卡模式。
- 没有保证不同 reader 权重、BERT 初始化缓存或 GPU 内核可逐位复现原数字。模型文件不匹配默认拒绝启动，显式允许差异即属于新配置。
- 没有补造历史缺失的 ARC 原始完整打分计时、峰值资源及账单。
- 没有运行或补造 CoopES，也没有确认它与本次具有相同数据、模型和推理设置。

历史四组模型运行在 2026-11-11 完成，原始训练及评测日志、配置、完整预测和完成标记位于 results/tablei_20261111。本发布中的 run/evaluate 新包装器复用了那些实际训练、编码、beam 搜索和 reader 实现；本次验证覆盖了数据加载、配置、证据和结果汇总边界，包装器的完整 GPU 执行仍应在重新运行后记录。

## 交付复核

Git 提交后还将从提交快照导出并运行 README 的 CPU 命令，再打 ZIP 并通过包内 verify_package.py 校验全部文件、训练输入和四组结果。该交付核验的实际输出随 ZIP 的 code/verification 与发布说明记录，不将未执行的 GPU 流程称为已验证。
