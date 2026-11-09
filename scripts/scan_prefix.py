#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并行窗口校验：找出本地分片相对源文件的"第一个损坏字节"。

为什么不能用二分：
  之前多个 curl -C - 并发写同一文件，损坏是"区间错位"而非"整体截断"，
  内容在坏区间之后又可能恢复对齐 → "是否匹配"对偏移非单调，二分会跳过早期损坏。
本脚本改为：把本地已下载区间切成窗口，逐窗口与源做 md5 比对，取最早不匹配窗口，
再在窗口内逐级细分（16MB → 1MB → 4KB），最终给出精确的首个坏字节偏移。

输出：<file>.prefix  = 首个坏字节偏移（即 [0, prefix) 为有效前缀）
"""
import argparse
import hashlib
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

MS_URL = "https://www.modelscope.cn/api/v1/models/{repo}/repo?Revision=master&FilePath={file}"


def log(m):
    print(f"[{time_str()}] {m}", flush=True)


def time_str():
    import time
    return time.strftime("%H:%M:%S")


def url_for(repo, fname):
    return MS_URL.format(repo=repo, file=fname)


def fetch(repo, fname, start, end, tries=6):
    """取 [start,end] 字节；失败返回 None。"""
    want = end - start + 1
    for _ in range(tries):
        p = subprocess.run(
            ["curl", "-sSL", "--max-time", "180", "-r", f"{start}-{end}", url_for(repo, fname)],
            capture_output=True)
        if p.returncode == 0 and len(p.stdout) == want:
            return p.stdout
    return None


def md5(b):
    return hashlib.md5(b).hexdigest()


def scan(repo, path, window=16 << 20, workers=8):
    # .part0/.part1 等是同一分片的分块副本，源文件名要去掉后缀
    fname = os.path.basename(path)
    for suf in (".part0", ".part1", ".part2", ".part3"):
        if fname.endswith(suf):
            fname = fname[: -len(suf)]
            break
    local = os.path.getsize(path)
    log(f"{fname} 本地 {local} 字节 ({local/1024/1024:.1f} MB)，窗口 {window>>20}MB")

    # 构造窗口列表
    wins = []
    s = 0
    while s < local:
        e = min(s + window, local) - 1
        wins.append((s, e))
        s = e + 1

    bad_windows = []

    def check(w):
        ws, we = w
        ref = fetch(repo, fname, ws, we)
        if ref is None:
            return (ws, we, None)
        with open(path, "rb") as f:
            f.seek(ws)
            got = f.read(we - ws + 1)
        return (ws, we, md5(got) != md5(ref))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(check, wins))

    for ws, we, bad in results:
        if bad is None:
            log(f"  窗口 [{ws}-{we}] 网络探测失败，保守视为损坏")
            bad = True
        if bad:
            bad_windows.append((ws, we))

    if not bad_windows:
        log(f"  全部 {len(wins)} 个窗口均匹配 → 本地内容完全有效")
        return local

    first_ws, first_we = min(bad_windows)[0], min(bad_windows)[1]
    log(f"  首个坏窗口 [{first_ws}-{first_we}]，开始细分")

    # 在坏窗口内细分：16MB -> 1MB -> 4KB
    def refine(lo, hi, step):
        """在 [lo,hi) 内找首个坏字节，步长 step。返回精确偏移。"""
        bad_off = None
        s = lo
        while s < hi:
            e = min(s + step - 1, hi - 1)
            ref = fetch(repo, fname, s, e)
            if ref is None:
                bad_off = s
                break
            with open(path, "rb") as f:
                f.seek(s)
                got = f.read(e - s + 1)
            if md5(got) != md5(ref):
                bad_off = s
                break
            s = e + 1
        return bad_off

    off = refine(first_ws, first_we + 1, 1 << 20)
    if off is None:
        off = first_ws
    else:
        off = refine(off, first_we + 1, 4096)
        if off is None:
            off = first_ws
    log(f"  → 首个坏字节偏移 = {off} ({off/1024/1024:.1f} MB)")
    return off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--file", required=True)
    ap.add_argument("--window-mb", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    path = args.file
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        log(f"{path} 不存在或为空，前缀=0")
        open(path + ".prefix", "w").write("0")
        print("PREFIX=0")
        return 0

    pfx = scan(args.repo, path, window=args.window_mb << 20, workers=args.workers)
    with open(path + ".prefix", "w") as f:
        f.write(str(pfx))
    print(f"PREFIX={pfx}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
