#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 ModelScope 下载 google-bert/bert-base-uncased 到一个**普通本地目录**。

为什么不用 HF 缓存布局：transformers 4.56 的缓存解析依赖 refs/<rev> → snapshots/<commit>
的精确对应，很容易踩坑；直接给 DPR 配置一个本地路径最稳（离线也能加载）。

用法:
  python fetch_bert_local.py --out /root/autodl-tmp/models/bert-base-uncased
"""
import argparse
import json
import os
import subprocess
import sys
import urllib.request

MS = "https://www.modelscope.cn/api/v1/models/{repo}/repo?Revision=master&FilePath={f}"

# 必需 + 可选文件（safetensors 与 pytorch_model.bin 只需其一，但都下更保险）
FILES = [
    "config.json",
    "vocab.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "model.safetensors",
    "pytorch_model.bin",
]


def log(m):
    print(f"[bert] {m}", flush=True)


def fetch(repo, fname, dest):
    url = MS.format(repo=repo, f=fname)
    for attempt in range(1, 8):
        r = subprocess.run(
            ["curl", "-sSL", "--retry", "4", "--retry-delay", "3", "--retry-all-errors",
             "--max-time", "900", "-o", dest, url], capture_output=True)
        if r.returncode == 0 and os.path.exists(dest) and os.path.getsize(dest) > 0:
            return True
        log(f"  {fname} 第{attempt}次失败，重试")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default="google-bert/bert-base-uncased")
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    ok = 0
    for fname in FILES:
        dest = os.path.join(out, fname)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            log(f"[SKIP] {fname}")
            ok += 1
            continue
        log(f"[DL] {fname}")
        if fetch(args.repo, fname, dest):
            log(f"[OK] {fname} ({os.path.getsize(dest)} 字节)")
            ok += 1
        else:
            log(f"[FAIL] {fname}")
            if os.path.exists(dest):
                os.remove(dest)

    # 校验：config 可解析 + 至少有一个权重文件
    cfgp = os.path.join(out, "config.json")
    if not os.path.exists(cfgp):
        log("config.json 缺失，失败")
        return 1
    cfg = json.load(open(cfgp))
    weights = [f for f in ("model.safetensors", "pytorch_model.bin")
               if os.path.exists(os.path.join(out, f))]
    if not weights:
        log("没有任何权重文件，失败")
        return 1

    log("--- 目录内容 ---")
    for fn in sorted(os.listdir(out)):
        log(f"  {fn}  ({os.path.getsize(os.path.join(out, fn))} 字节)")
    log(f"arch={cfg.get('architectures')} hidden={cfg.get('hidden_size')} "
        f"layers={cfg.get('num_hidden_layers')} 权重={weights}")
    log(f"完成（{ok}/{len(FILES)} 个文件）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
