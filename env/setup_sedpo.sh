#!/usr/bin/env bash
# SeDPO 单卡 4090 环境一键准备（可选，等价于手册第 2、4 节的手工步骤）
# 用法：在 AutoDL 实例里执行  bash setup_sedpo.sh
# 注意：脚本只做安装与打补丁，不下载数据、不启动训练。

set -eo pipefail

REPO_DIR=/root/autodl-tmp/SeDPO_NIPS25
ENV_DIR=/root/autodl-tmp/envs/se_dpo

echo "== 1/6 检查硬件 =="
nvidia-smi --query-gpu=name,memory.total --format=csv || echo "!! 没检测到 GPU（无卡模式也能装，但装完记得换成带卡模式）"

echo "== 2/6 检查磁盘 =="
df -h /root/autodl-tmp | tail -1
echo "   建议数据盘剩余 > 60GB"

echo "== 3/6 拉代码 =="
source /etc/network_turbo 2>/dev/null || echo "!! 学术加速不可用，git clone 可能失败"
mkdir -p /root/autodl-tmp
if [ -d "$REPO_DIR/.git" ]; then
  echo "   已存在，跳过 clone"
else
  git clone https://github.com/2022neo/SeDPO_NIPS25.git "$REPO_DIR"
fi
cd "$REPO_DIR"

echo "== 4/6 建 conda 环境 =="
if [ -d "$ENV_DIR" ]; then
  echo "   环境已存在，跳过"
else
  conda create -p "$ENV_DIR" python=3.12 -y
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"

echo "== 5/6 装依赖 =="
export HF_ENDPOINT=https://hf-mirror.com
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "== 6/6 单卡补丁 =="
if [ ! -f scripts/get_cmds.py.bak ]; then
  cp scripts/get_cmds.py scripts/get_cmds.py.bak
  sed -i 's/--multi_gpu //g' scripts/get_cmds.py
  echo "   已删除 --multi_gpu（原文件备份为 get_cmds.py.bak）"
else
  echo "   补丁已打过，跳过"
fi
accelerate config default >/dev/null 2>&1 || true

echo
echo "== 自检 =="
python -c "import torch, transformers, datasets, faiss
print('torch      ', torch.__version__, '| cuda', torch.version.cuda, '| available', torch.cuda.is_available())
print('transformers', transformers.__version__)
print('datasets   ', datasets.__version__)"

cat <<'TIP'

环境好了。接下来（详见手册第 3、5 节）：
  export PYTHONPATH=$PWD:$PYTHONPATH HYDRA_FULL_ERROR=1
  export NUM_OF_GPUS=1 RETRIEVER_BSZ=8 SCORER_BSZ=6
  sh getcmd.sh
  sh get_score.sh coreference      # 约 30~60 分钟
  sh run_sedpo.sh coreference 0.1
TIP
