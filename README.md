# SeDPO 复现代码（NeurIPS 2025）

> 论文：*Learning to Rank for In-Context Example Retrieval*（SeDPO）
> 本仓库 = 论文官方代码 + 我在单卡 4090 上跑通全流程时补的脚本、补丁和实测结果。

---

## 1. 这个仓库里有什么

```
sedpo-repro/
├── src/          # 论文官方代码（SeDPO_NIPS25-main），仅删掉 __pycache__
│   ├── DPR/          # 稠密检索 (biencoder) 相关
│   ├── configs/      # hydra 配置
│   ├── scripts/      # 数据处理 / 训练 / 推理入口
│   ├── figures/      # 论文插图
│   ├── requirements.txt
│   ├── getcmd.sh     # 生成每个任务的 train.sh / infer.sh
│   ├── get_score.sh  # 给样本打 retrieval 分数（要先跑）
│   └── run_sedpo.sh  # 主实验入口：sh run_sedpo.sh <task> <beta>
├── scripts/      # ★ 我写的实验脚本（原 tools/ 目录）
│                   #   单卡适配补丁 patch_*.py、批量跑 run_*.sh、
│                   #   结果汇总 collect_results.py、指标补算 compute_acc_f1.py
├── env/          # 环境准备与运行变量
│   ├── setup_sedpo.sh   # 一键装环境 + 打单卡补丁（AutoDL 4090）
│   └── run_env.sh       # 8GB 小显存版环境变量（本地 4060 用）
├── results/      # 我跑出来的结果（csv / json）
└── README.md
```

**不含**（体积大或敏感，需自行准备）：数据集、模型权重、checkpoint、运行日志、服务器凭据。

---

## 2. 环境

已在 **AutoDL 单卡 RTX 4090 / Python 3.12 / CUDA 12.x** 上验证通过。

```bash
# 方式一：一键脚本（会 clone 官方仓库，此处可跳过 clone 直接用本仓库代码）
bash env/setup_sedpo.sh

# 方式二：手动
conda create -p /root/autodl-tmp/envs/se_dpo python=3.12 -y
conda activate /root/autodl-tmp/envs/se_dpo
pip install -r src/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

关键依赖：`torch==2.8.0`、`transformers==4.56.2`、`datasets==4.1.1`、`faiss-cpu==1.12.0`、`accelerate==1.10.1`

国内拉 HuggingFace 资源先设镜像：`export HF_ENDPOINT=https://hf-mirror.com`

---

## 3. 数据

论文附录给的 scored / prompt_pool 两个文件夹（Google Drive），下载后放到：

```
src/my_data/scored/
src/my_data/prompt_pool/
```

然后：

```bash
cd src
export PYTHONPATH=$PWD:$PYTHONPATH HYDRA_FULL_ERROR=1
export NUM_OF_GPUS=1 RETRIEVER_BSZ=8 SCORER_BSZ=6

sh getcmd.sh                              # 生成各任务命令
sh get_score.sh coreference               # 打 retrieval 分数（30~60 min）
sh run_sedpo.sh coreference 0.1           # 主实验：任务名 + beta
```

我的任务清单（详见 `scripts/`）：`coreference`、`multirc`、`openbookqa`、`arc_c`、`arc_easy`、`boolq`…

---

## 4. 实测结果

单卡 4090，3-shot，每个任务训练 6 epoch：

| 模型 | 任务 | 指标 | 分数 | 样本数 |
|---|---|---|---|---|
| Qwen2.5-7B | ARC-C | Acc | **54.68** | 1165 |
| Qwen2.5-7B | MultiRC | F1 | **80.79**（Acc 83.07） | 4848 |
| Qwen2.5-7B | OpenBookQA | Acc | **59.78** | 557 |
| Llama3-8B | ARC-C | Acc | **50.64** | 1165 |
| Llama3-8B | MultiRC | F1 | **60.73**（Acc 50.89） | 4848 |
| Llama3-8B | OpenBookQA | Acc | **58.71** | 557 |

完整数据见 `results/`。

**已知问题**：Llama3-8B 在 MultiRC 上预测严重偏斜（82.3% 预测为标签 1），F1 与 Acc 差距 10 个点，说明该模型的判别头没收敛好，这组数字不宜直接对标论文。

---

## 5. 相对官方代码的改动

1. **单卡适配**：官方 `scripts/get_cmds.py` 硬编码 `--multi_gpu`，单卡会报错。`env/setup_sedpo.sh` 里用 sed 删掉该参数（同时备份 `.bak`）；等价补丁见 `scripts/patch_train_sh.py`。
2. **显存适配**：8GB 卡上用 `env/run_env.sh` 的 `SCORER_BSZ=1` + `expandable_segments`。
3. **指标补算**：官方只输出 Acc，我加了 `scripts/compute_acc_f1.py` 从 `preds.jsonl` 重算 Acc / F1 / 混淆矩阵。

---

## 6. 致谢

官方代码：<https://github.com/2022neo/SeDPO_NIPS25>（原仓库 LICENSE 见 `src/LICENSE`）。
本仓库仅用于论文复现与学习，未用于商业用途。
