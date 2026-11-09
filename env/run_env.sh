#!/bin/sh
# SeDPO 本机运行环境（RTX 4060 Laptop 8GB）
# 用法：cd /root/SeDPO_NIPS25 && . ./run_env.sh
cd /root/SeDPO_NIPS25 || exit 1
. /root/venv/bin/activate

# 国内下载走镜像
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=$PWD:$PYTHONPATH
export HYDRA_FULL_ERROR=1

# 单卡 + 8GB 显存专用
export NUM_OF_GPUS=1
export RETRIEVER_BSZ=8
export SCORER_BSZ=1
export SCORE_LLM='EleutherAI/gpt-neo-2.7B'
export INF_LLM='EleutherAI/gpt-neo-2.7B'
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
