#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
二分查找本地已下载文件相对 ModelScope 源文件的"最后有效字节偏移"。

背景：原先多个 curl -C - 进程并发写同一 .safetensors，导致中段字节错位。
本脚本以 ModelScope 为基准，二分定位首个不匹配偏移，从而保留有效前缀、只补下缺失部分。

用法：
  python find_prefix.py --repo NousResearch/Meta-Llama-3-8B --file /root/.../model-00001-of-00004.safetensors
输出：<file>.prefix  写入有效前缀长度（字节）
"""
import argparse
import os
import subprocess
import sys

MS_URL = "https://www.modelscope.cn/api/v1/models/{repo}/repo?Revision=master&FilePath={file}"
PROBE = 1 << 20  # 每次探测 1MB


def fetch(repo, fname, start, end):
    """取区间字节到内存，返回 bytes（失败返回 None）。"""
    for _ in range(5):
        p = subprocess.run(
            ["curl", "-sSL", "--max-time", "90", "-r", f"{start}-{end}",
             url_for(repo, fname)],
            capture_output=True)
        if p.returncode == 0 and len(p.stdout) == end - start + 1:
            return p.stdout
    return None


def url_for(repo, fname):
    return MS_URL.format(repo=repo, file=fname)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--file", required=True)
    ap.add_argument("--max-check", type=int, default=64,
                    help="二分阶段最多探测次数")
    args = ap.parse_args()

    path = args.file
    local = os.path.getsize(path)
    fname = os.path.basename(path)
    log = lambda m: print(f"[{args.file}] {m}", flush=True)
    log(f"本地大小 {local}")

    def matches(off):
        """检查本地 offset off 处起 PROBE 字节是否与源一致。"""
        if off + PROBE > local:
            end = local - 1
            n = local - off
        else:
            end = off + PROBE - 1
            n = PROBE
        if n <= 0:
            return True
        ref = fetch(args.repo, fname, off, end)
        if ref is None:
            return None  # 网络问题，未知
        with open(path, "rb") as f:
            f.seek(off)
            got = f.read(n)
        return got == ref

    # 先确认头部有效
    head = matches(0)
    if head is False:
        log("头部即不匹配 → 本地文件完全不可用")
        with open(path + ".prefix", "w") as f:
            f.write("0")
        print("PREFIX=0")
        return 0
    if head is None:
        log("网络探测失败")
        return 1

    # 二分：找最大的 o 使 [o, o+PROBE) 仍匹配
    lo, hi = 0, local  # [lo, hi) 为可疑区间，lo 已知匹配前缀
    checks = 0
    while hi - lo > PROBE and checks < args.max_check:
        mid = (lo + hi) // 2
        r = matches(mid)
        checks += 1
        if r is None:
            log("网络探测失败，保守取 lo")
            break
        if r:
            lo = mid
        else:
            hi = mid
    log(f"二分 {checks} 次 → 有效前缀约 {lo} 字节 ({lo/1024/1024:.1f} MB)")
    with open(path + ".prefix", "w") as f:
        f.write(str(lo))
    print(f"PREFIX={lo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
