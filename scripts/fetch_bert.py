#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 ModelScope 拉取 google-bert/bert-base-uncased，填充到 HuggingFace 缓存布局
（cache/models--google-bert--bert-base-uncased/{snapshots,refs,blobs}），
让 DPR 的 HF_BertEncoder 在离线环境下也能加载到预训练权重。

为什么需要：huggingface.co 在本机被墙，transformers 无法自动下载。

用法:
  python fetch_bert.py --cache_dir /root/autodl-tmp/SeDPO_NIPS25/cache
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import urllib.request

MS = "https://www.modelscope.cn/api/v1/models/{repo}/repo?Revision=master&FilePath={f}"

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
             "--max-time", "600", "-o", dest, url],
            capture_output=True)
        if r.returncode == 0 and os.path.exists(dest) and os.path.getsize(dest) > 0:
            return True
        log(f"  {fname} 第{attempt}次失败，重试")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--repo", default="google-bert/bert-base-uncased")
    args = ap.parse_args()

    cache = os.path.abspath(args.cache_dir)
    repo_dir = "models--" + args.repo.replace("/", "--")
    root = os.path.join(cache, repo_dir)
    blobs = os.path.join(root, "blobs")
    refs = os.path.join(root, "refs")
    os.makedirs(blobs, exist_ok=True)
    os.makedirs(refs, exist_ok=True)

    # 取 ModelScope 侧的 commit revision
    rev = "main"
    try:
        api = f"https://www.modelscope.cn/api/v1/models/{args.repo}/repo/files?Revision=master"
        with urllib.request.urlopen(api, timeout=60) as r:
            data = json.loads(r.read().decode())
        for f in data["Data"]["Files"]:
            if f["Path"] == "config.json":
                rev = f.get("Revision", "main")
                break
    except Exception as exc:  # noqa: BLE001
        log(f"取 revision 失败，用 main：{exc}")

    # HF 缓存约定：refs/main 存 commit hash，实际文件放在 snapshots/<commit_hash>/
    snaps = os.path.join(root, "snapshots", rev)
    os.makedirs(snaps, exist_ok=True)
    with open(os.path.join(refs, "main"), "w") as f:
        f.write(rev)
    log(f"缓存目录 {root}  revision={rev}")
    log(f"snapshot 目录 {snaps}")

    got_any = False
    for fname in FILES:
        target = os.path.join(snaps, fname)
        if os.path.exists(target) and os.path.getsize(target) > 0:
            log(f"[SKIP] {fname}")
            got_any = True
            continue
        tmp = os.path.join(blobs, fname + ".dl")
        log(f"[DL] {fname}")
        if not fetch(args.repo, fname, tmp):
            log(f"[FAIL] {fname} 下载失败")
            continue
        # HF 缓存用 sha256 作为 blob 名
        h = hashlib.sha256()
        with open(tmp, "rb") as f:
            while True:
                b = f.read(1 << 20)
                if not b:
                    break
                h.update(b)
        blob_name = h.hexdigest()
        blob_path = os.path.join(blobs, blob_name)
        if os.path.exists(blob_path):
            os.remove(tmp)
        else:
            os.replace(tmp, blob_path)
        # snapshot 里做符号链接
        if os.path.exists(target) or os.path.islink(target):
            os.remove(target)
        os.symlink(blob_path, target)
        log(f"[OK] {fname} → blob {blob_name[:12]}… ({os.path.getsize(blob_path)} 字节)")
        got_any = True

    # 列出结果
    log("--- snapshot 内容 ---")
    for fn in sorted(os.listdir(snaps)):
        p = os.path.join(snaps, fn)
        log(f"  {fn}  ({os.path.getsize(p)} 字节)")

    if not got_any:
        log("全部失败")
        return 1
    log("完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
