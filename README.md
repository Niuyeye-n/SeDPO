# SeDPO 四组基线实验复现

本仓库保存 SeDPO 在 ARC-C、OpenBookQA × Qwen2.5-7B、Llama3-8B 上的四组真实实验代码、输入协议和可核验结果。方法来自 NeurIPS 2025 [Learning to Rank for In-Context Example Retrieval](https://github.com/2022neo/SeDPO_NIPS25)，实验运行于 **2026-11-11**。

**这里只运行了 SeDPO。CoopES 来自未发表稿件，其原代码和精确运行设置未提供；本结果不是“已经证明 CoopES 更好”的同设置对照。** 旧 3-shot、重复 OBQA 行和旧似然评测保留为历史资料，最新入口只有下述 `reproduce.py`。

## 1 最终四组结果

| 数据集 | Reader | 正确/题数 | Accuracy % | Weighted-F1 % | Macro-F1 % |
|---|---|---:|---:|---:|---:|
| ARC-C | Qwen2.5-7B | 442/500 | 88.40 | 88.3901 | 88.4702 |
| OpenBookQA | Qwen2.5-7B | 452/500 | 90.40 | 90.5061 | 90.2406 |
| ARC-C | Llama3-8B | 414/500 | 82.80 | 82.8012 | 82.8610 |
| OpenBookQA | Llama3-8B | 448/500 | 89.60 | 89.6106 | 89.4638 |

`results/tablei_20261111/` 含逐题输出、冻结样本 ID、示例选择、原始训练日志、模型文件哈希、配置与阶段耗时。历史绝对路径保留作溯源，新执行入口不依赖原服务器目录。

## 2 无需 GPU 的结果复核

```bash
git clone https://github.com/Niuyeye-n/SeDPO.git
cd SeDPO
python reproduce.py check-source
python reproduce.py verify
python -m unittest discover -s repro -p 'test_*.py' -v
```

Python 3.12 为实测版本。上面三条只用标准库，无需安装依赖或下载模型。它们检查实际运行源码的 SHA256，以及四组完成标记、500 个唯一题目 ID、冻结 gold、示例 ID 及顺序、输出字母/token、有限 logits，并从逐题预测重算 Accuracy/两种 F1。篡改结果、标签或示例会被拒绝。

这是“复核历史结果”，不是“重新执行模型”，也不是方法优越性检验。

## 3 材料与目录

```text
reproduce.py                    # verify / check-source / plan / run / evaluate
repro/                          # 数据协议、检索适配、评测、训练预算、测试
repro/runtime_source_sha256.json # 实际运行源码指纹
repro/training_input_sha256.json # 四组完整偏好输入指纹
src/                            # 上游代码及服务器实际运行的补丁
env/setup_tablei.sh              # Linux/Python 3.12 新建 venv
env/requirements-tablei.txt
results/tablei_20261111/         # 最新结果与原始证据
docs/                           # 完整报告、标黄疑问回应、历史说明
```

配套 ZIP `SeDPO_完整结果与复现包_20261112.zip` 包含：`code/`（该次 Git 提交快照）、`results/`（完整原始结果包及检索中间记录）、`training_inputs/`（Qwen/Llama 两任务合并 train/valid 偏好文件和全训练候选池）、`reports/`、包核验程序与逐文件 SHA256。

ZIP **不包含** reader LLM 权重、每组约 2.6 GB 的完整 retriever+optimizer 检查点、虚拟环境或服务器/GitHub 凭据。最佳检查点的原位置、大小与哈希见 `results/tablei_20261111/evidence/*/retention_manifest.json`。结果及偏好输入完整提供；模型重跑需自行准备权重，不能把本 ZIP 当作无需模型依赖的镜像。大体积训练输入随 ZIP 提供，不进入 Git 历史。

## 4 GPU 环境及模型

完整重跑要求 Linux、Python 3.12、可运行 PyTorch CUDA 12.8 轮子的 NVIDIA 驱动、单张 RTX 4090 24 GB 级 GPU。四组各轮检查点均保留，**模型已准备好后，工作盘建议至少空闲 80 GiB**；单组约需空闲 20 GiB。脚本不会自动删除旧 checkpoint。无卡/2 GiB 内存实例只适合读日志及核算结果。

```bash
# 新环境目录必须不存在；可放到数据盘
bash env/setup_tablei.sh /data/envs/sedpo-tablei
source /data/envs/sedpo-tablei/bin/activate
```

该脚本不会 clone 最新上游覆盖本仓库，不会修改其他环境，不会启动训练。固定版本包括 torch 2.8.0+cu128、Transformers 4.56.2、Datasets 4.1.1、Accelerate 1.10.1、FAISS CPU 1.12.0；完整原环境清单见 `evidence/software.txt`。旧的 `env/setup_sedpo.sh`、`scripts/run_*.sh` 不是最新四组的入口。

准备两个本地 HF 格式 reader 目录，含 config、tokenizer、全部 safetensors 分片及 index。**精确文件 SHA256 在每组 `result.json -> manifest.model_files`**；不要仅根据目录名或 “Instruct” 标签认定模型相同。默认逐文件核对，不匹配即停止。优先使用原实验保留的制品；本仓库无法仅靠模型简称确定其上游 revision，不自动猜测下载版本。

训练还需可访问的 `google-bert/bert-base-uncased`，由原训练器载入工作缓存。完全离线运行须预先准备其 HF 缓存。没有单独归档原 BERT 初始化制品的全量权重快照，不承诺不同上游缓存逐位等价。模型获取分别受其许可及访问条件约束，仓库不提供凭据。

若要测试其他 reader，可显式加 `--allow-different-model`；差异写入 manifest，属于新配置结果，不能宣称原模型逐位复现。

## 5 使用归档偏好数据重训并评测 推荐

先解压 ZIP。以下假设包目录为 `/data/SeDPO_results_20261112/`，模型路径由你自己准备。新工作目录必须不存在；Linux 路径不要带空格。

```bash
# 只显示 argv，不写工作目录、不加载模型、不使用 GPU
python reproduce.py plan --job all --mode archived-inputs \
  --training-inputs /data/SeDPO_results_20261112/training_inputs \
  --work-dir /data/runs/sedpo-tablei-run01 \
  --qwen-model /data/models/qwen2.5-7b \
  --llama-model /data/models/llama3-8b

# 实际运行四组：Qwen ARC、Qwen OBQA、Llama ARC、Llama OBQA，串行
python reproduce.py run --job all --mode archived-inputs \
  --training-inputs /data/SeDPO_results_20261112/training_inputs \
  --work-dir /data/runs/sedpo-tablei-run01 \
  --qwen-model /data/models/qwen2.5-7b \
  --llama-model /data/models/llama3-8b
```

每组流程：校验输入 → 计算真实展开与 warmup → FP32 BERT 双编码器训练 6 轮 → 原保存策略选 best_valid → 编码固定 50 候选 → beam=3 选有序 4 示例 → FP16/SDPA reader 评测 500 题。`plan` 中 `ACTUAL_WARMUP` 是执行到训练时由实际数据填入的占位符。

程序核对完整偏好文件与 prompt_pool 哈希及训练/验证 query 隔离；不匹配会停止。仅在明确开展新数据配置时才使用 `--allow-different-data`，不要用它掩盖复现失败。

| Job | 展开样本 | 更新/轮 | 总更新 | warmup | 学习率 |
|---|---:|---:|---:|---:|---:|
| qwen_arc | 9152 | 143 | 858 | 86 | 1e-5 |
| qwen_obqa | 11567 | 180 | 1080 | 108 | 3e-5 |
| llama_arc | 9355 | 146 | 876 | 88 | 1e-5 |
| llama_obqa | 11366 | 177 | 1062 | 106 | 3e-5 |

公共设置：CLI batch=8，SeDPO 内部乘 8 后为 64；seed=12345；梯度累积=1；GRAD_CKPT=1；IF_PREFER_LOSS=2；DIV_REJECT=1；β=0.1。batch 影响整批拒绝采样，不应随意减小。

单组运行把 `--job all` 改为 `qwen_arc` 等，仅提供对应模型。每组输出如下：

```text
run01/qwen_arc/data/                    # 本次偏好与候选池副本
run01/qwen_arc/commands.jsonl           # 实际命令及 UTC 时间
run01/qwen_arc/plan.log train.log ...   # 各阶段 stdout/stderr
run01/qwen_arc/training/training_plan.json
run01/qwen_arc/training/model/dpr_biencoder.best_valid
run01/qwen_arc/evaluation/selection.json
run01/qwen_arc/evaluation/result.json
run01/qwen_arc/evaluation/result.complete.json
run01/qwen_arc/job.complete.json
```

失败立即退出，不写完成标记。保留失败日志，使用新目录重试；没有自动 resume 或覆盖。已有成功训练的检查点，可用下一节单独评测，无需再次训练。

工作目录根部的 `summary.json` 和 `summary.csv` 会逐组汇总已完成结果；不生成未完成组的虚假分数。

## 6 已有检索器 只重新检索与答题

```bash
python reproduce.py evaluate --job qwen_arc \
  --checkpoint /data/checkpoints/qwen_arc/dpr_biencoder.best_valid \
  --work-dir /data/runs/sedpo-qwen-arc-eval01 \
  --qwen-model /data/models/qwen2.5-7b
```

仍需 GPU 和 reader，不需要偏好输入，不进行训练。checkpoint 哈希写入 selection，最终采用相同冻结协议。

## 7 从官方数据重新构造偏好 可选高成本路径

```bash
python reproduce.py run --job all --mode rescore \
  --work-dir /data/runs/sedpo-tablei-rescore01 \
  --qwen-model /data/models/qwen2.5-7b \
  --llama-model /data/models/llama3-8b
```

下载固定 revision 的 ARC/OBQA 官方数据，保存为任务类可读的 `cache/saved_datasets/arc_c` 和 `openbookqa`。执行原三阶段候选构造、似然打分，再按 step1 的 query 分组合并。L=50、n_score=300、打分 batch=5、max_length=2048。300 源 query 按 270/30 分组，合并 810/90 条阶段记录。完整训练候选池与最终推理的 50 条冻结候选池不是同一个规模。

该路径需网络和更多 GPU 时间。新合并输入哈希或展开数与原记录不符，严格模式会停止，应调查并报告；硬件/内核差异也可能改变分数排序，不能为了得到目标结果去调整样本或伪造输出。Qwen/Llama 独立生成各自偏好，不运行其他方法。

## 8 评测口径及限制

- ARC-C 从官方 test 的 1165 道四选项题中按 ID 哈希选 500；另有 7 道非四选项题排除。OBQA 使用全部 500 道官方 test。
- 每任务两个 reader 的候选和测试题相同；另保留 50 道 official validation，未用于本次内部选模。检索器内部验证是训练来源的 30 个源 query。
- 最终四示例对话使用各模型原生 chat template；OBQA 所有示例和目标均含 fact1。system 为 `Answer with exactly one letter: A, B, C or D.`
- 单 token ABCD 约束，do_sample=false、num_beams=1、max_new_tokens=1、repetition_penalty=1、FP16/SDPA、输入上限 3072、不静默截断。temperature/top_k/top_p 不参与采样，记 N/A。
- 偏好构造用完整选项文本似然；最终评测用对话中的一个字母，不能混写。
- 最佳检查点只在原策略保存的第 2/4/5/6 轮中选择；原四组最佳为第 5/4/4/2 轮，不按测试准确率挑选。
- 原训练总步数与 warmup 已按展开核实；旧约 76 步说法、557 条 OBQA 结果及旧似然结果都不是本次有效基线。
- 一套训练 seed、测试来源曾用于诊断，属于探索性实验。原 CoopES 精确设置缺失，不能据此作严格方法优劣或配对显著性结论。
- `docs/` 完整报告保留用户标黄 `==...==`。旧报告关于“原稿是否含 fact”的表述，以同目录逐条核查更正为准：CoopES 稿件模板确实含 `{fact}`，实际拼接细节未恢复。

## 9 发布验证范围

详见 [VALIDATION.md](VALIDATION.md)：结果重算、源码指纹、输入展开、DPR 适配、Hydra 配置与篡改拒绝检查分别记录。**本次代码整理没有在无卡实例上重新执行完整 GPU 训练或推理。** 历史四组运行原始日志已归档。相同代码与输入也不承诺跨硬件逐位一致，重跑结果应如实记录。

## 10 历史与许可

旧入口说明见 `docs/README_legacy_20261107.md`。源代码沿用 `src/LICENSE`，DPR 文件保留原版权。模型与数据分别遵循其许可；本仓库不授予 reader 权重的新许可。SSH/GitHub 凭据不属于交付材料。
