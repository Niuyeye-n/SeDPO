#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ModelScope 分块续传下载器（就地追加，节省磁盘）。

设计要点：
  1) 复用本地已下载的有效前缀（由 find_prefix.py 标定 .prefix），不从头重下；
  2) 剩余部分切成 128MB 块，并发下载到独立临时文件，块内严格校验字节数；
  3) 按序 append 进目标文件并立刻删除临时块 —— 峰值额外磁盘仅 ≈ 并发数×块大小；
  4) 全部分片完成后用 ModelScope 官方 Sha256 做整文件校验。

用法：
  python dl_fast.py --repo Qwen/Qwen2.5-7B --dest /root/autodl-tmp/models/qwen2.5-7b \
      --shards model-00001-of-00004.safetensors,... --threads 8 --chunk-mb 128
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

MS_URL = "https://www.modelscope.cn/api/v1/models/{repo}/repo?Revision=master&FilePath={file}"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def url_for(repo, fname):
    return MS_URL.format(repo=repo, file=fname)


def head_size(repo, fname):
    for _ in range(5):
        r = subprocess.run(["curl", "-sIL", "--max-time", "40", url_for(repo, fname)],
                           capture_output=True, text=True)
        sizes = []
        for line in r.stdout.splitlines():
            if line.lower().startswith("content-length"):
                try:
                    sizes.append(int(line.split(":", 1)[1].strip()))
                except ValueError:
                    pass
        if sizes:
            return max(sizes)
        time.sleep(3)
    return None


def fetch_range(repo, fname, start, end, tmp, tries=20):
    """下载 [start,end] 到 tmp，校验字节数；失败重试（块级，不重下整文件）。"""
    expect = end - start + 1
    for attempt in range(1, tries + 1):
        if os.path.exists(tmp) and os.path.getsize(tmp) == expect:
            return True
        if os.path.exists(tmp):
            os.remove(tmp)
        r = subprocess.run(
            ["curl", "-sSL", "--retry", "4", "--retry-delay", "3", "--retry-all-errors",
             "-r", f"{start}-{end}", "-o", tmp, "-w", "%{http_code}", url_for(repo, fname)],
            capture_output=True, text=True)
        code = r.stdout.strip()[-3:] if r.stdout.strip() else "000"
        got = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        if got == expect and code in ("206", "200"):
            return True
        log(f"    块[{start}-{end}] 第{attempt}次失败 code={code} {got}/{expect}，重试")
        time.sleep(4)
    return False


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 24)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def official_sha(repo):
    """取 ModelScope 官方 Sha256 映射。"""
    api = f"https://www.modelscope.cn/api/v1/models/{repo}/repo/files?Revision=master"
    try:
        with urllib.request.urlopen(api, timeout=60) as r:
            data = json.loads(r.read().decode())
        out = {}
        for f in data["Data"]["Files"]:
            if f.get("Sha256"):
                out[f["Path"]] = f["Sha256"]
        return out
    except Exception as exc:  # noqa: BLE001
        log(f"[WARN] 取官方 sha256 失败: {exc}")
        return {}


def pick_base(dest, fname, candidates):
    """在候选文件中挑有效前缀最长的那个作为续传基座。"""
    best = None
    for c in candidates:
        p = os.path.join(dest, c)
        if not os.path.exists(p):
            continue
        pfx_file = p + ".prefix"
        pfx = None
        if os.path.exists(pfx_file):
            try:
                pfx = int(open(pfx_file).read().strip())
            except ValueError:
                pfx = None
        if pfx is None:
            pfx = os.path.getsize(p)  # 无扫描结果时保守认为整体有效
        if best is None or pfx > best[1]:
            best = (p, pfx)
    return best


def verify_prefix_boundary(repo, path, pfx, fname, probe=1 << 20):
    """
    自校验前缀边界：比对本地 [pfx-probe, pfx) 与源同区间。
    不一致说明 .prefix 标定不可信（旧的二分法在"区间错位"场景下会高估），
    此时用二分向下收缩到真实有效长度，绝不把损坏字节带进最终文件。
    返回可信的 pfx。
    """
    if pfx <= 0:
        return 0
    local = os.path.getsize(path)
    hi = min(pfx, local)
    while hi > 0:
        s = max(0, hi - probe)
        ref = fetch_range_bytes(repo, fname, s, hi - 1)
        if ref is None:
            # 网络探测失败：保守认为 hi 之前的不可信，退到 0
            log(f"    [边界校验] 探测失败，退回 0")
            return 0
        with open(path, "rb") as f:
            f.seek(s)
            got = f.read(hi - s)
        if got == ref:
            return hi
        # 坏区间在这一段内，二分收缩
        lo = s
        while lo < hi:
            mid = (lo + hi) // 2
            s2 = max(0, mid - probe)
            ref2 = fetch_range_bytes(repo, fname, s2, mid - 1)
            if ref2 is None:
                return 0
            with open(path, "rb") as f:
                f.seek(s2)
                got2 = f.read(mid - s2)
            if got2 == ref2:
                lo = mid
            else:
                hi = mid
        hi = lo
    return 0


def fetch_range_bytes(repo, fname, start, end, tries=5):
    for _ in range(tries):
        p = subprocess.run(
            ["curl", "-sSL", "--max-time", "120", "-r", f"{start}-{end}",
             url_for(repo, fname)], capture_output=True)
        if p.returncode == 0 and len(p.stdout) == end - start + 1:
            return p.stdout
    return None


def do_shard(repo, fname, dest, chunk, threads, hashes, force_fresh):
    out = os.path.join(dest, fname)
    size = head_size(repo, fname)
    if size is None:
        log(f"[FAIL-HEAD] {fname}")
        return False

    if os.path.exists(out) and os.path.getsize(out) == size:
        log(f"[SKIP] {fname} 已是完整尺寸，校验 sha256")
        return verify(out, fname, size, hashes)

    log(f"[INFO] {fname} 总大小 {size} 字节 ({size/1024/1024:.1f} MB)")

    base = None
    if not force_fresh:
        cands = [fname, fname + ".part0", fname + ".part1", fname + ".part2", fname + ".part3"]
        base = pick_base(dest, fname, cands)
    if base is None:
        base = (out, 0)
    base_path, pfx = base
    if pfx > size:
        pfx = 0

    # 统一到 out 作为基座：若基座是 part 文件，先搬过来
    if base_path != out:
        if os.path.exists(out):
            os.remove(out)
        os.rename(base_path, out)
        base_path = out
    # 截断到有效前缀
    if os.path.exists(out) and os.path.getsize(out) != pfx:
        with open(out, "r+b") as f:
            f.truncate(pfx)

    # 自校验边界，防止 .prefix 标定不可信导致带入损坏字节
    if pfx > 0:
        log(f"    校验前缀边界 ...")
        good = verify_prefix_boundary(repo, out, pfx, fname)
        if good != pfx:
            log(f"    [警告] 边界校验未通过，pfx {pfx} → {good}，按可信值重新截断")
            pfx = good
            with open(out, "r+b") as f:
                f.truncate(pfx)
    log(f"    有效前缀 {pfx} 字节 ({pfx/1024/1024:.1f} MB)，需补 {size - pfx} 字节")

    # 清理冗余的 part 文件（它们是同一文件的前缀副本，留着浪费磁盘）
    for junk in [f"{out}.part0", f"{out}.part1", f"{out}.part2", f"{out}.part3"]:
        if os.path.exists(junk):
            os.remove(junk)

    remaining = size - pfx
    if remaining > 0:
        chunks = []
        s = pfx
        idx = 0
        while s < size:
            e = min(s + chunk - 1, size - 1)
            chunks.append((s, e, f"{out}.blk{idx}"))
            idx += 1
            s = e + 1
        log(f"    分 {len(chunks)} 块下载，并发 {threads}")

        with ThreadPoolExecutor(max_workers=threads) as ex:
            futs = {}
            for s, e, p in chunks:
                futs[ex.submit(fetch_range, repo, fname, s, e, p)] = (s, e, p)
            failed = []
            done = 0
            for fu in futs:
                pass
            for fu, meta in futs.items():
                r = fu.result()
                done += 1
                if not r:
                    failed.append(meta)
                elif done % 5 == 0 or done == len(chunks):
                    log(f"    下载进度 {done}/{len(chunks)} 块")
        if failed:
            log(f"[WARN] {fname} 有 {len(failed)} 块失败，保留现场可重跑续传")
            return False

        # 按序 append 进目标文件，并立刻删除临时块
        log(f"    合并 {len(chunks)} 块到目标文件")
        allparts = [p for (_s, _e, p) in chunks]
        with open(out, "ab") as w:
            for p in allparts:
                with open(p, "rb") as r:
                    while True:
                        b = r.read(1 << 24)
                        if not b:
                            break
                        w.write(b)
                os.remove(p)

    final = os.path.getsize(out)
    if final != size:
        log(f"[FAIL-SIZE] {fname} 合并后 {final} != {size}")
        return False
    return verify(out, fname, size, hashes)


def verify(path, fname, size, hashes):
    want = hashes.get(fname)
    if not want:
        log(f"[OK-SIZE] {fname} 尺寸正确（无官方 sha256 可比）")
        return True
    log(f"    校验 sha256 ...")
    got = sha256_of(path)
    if got == want:
        log(f"[OK] {fname} sha256 匹配 ✓")
        return True
    log(f"[FAIL-SHA] {fname} sha256 不匹配 got={got} want={want}")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--shards", required=True, help="逗号分隔文件名")
    ap.add_argument("--extra", default="", help="逗号分隔的小文件（config 等）")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--chunk-mb", type=int, default=128)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    hashes = official_sha(args.repo)
    log(f"官方 sha256 列表：{len(hashes)} 个文件")
    chunk = args.chunk_mb * 1024 * 1024

    ok = True
    # 小文件先下（体积小、快）
    for f in [x.strip() for x in args.extra.split(",") if x.strip()]:
        out = os.path.join(args.dest, f)
        if os.path.exists(out) and os.path.getsize(out) > 0:
            continue
        size = head_size(args.repo, f)
        if not size:
            continue
        tmp = out + ".tmp"
        if fetch_range(args.repo, f, 0, size - 1, tmp):
            os.replace(tmp, out)
            log(f"[OK] {f} ({size} 字节)")
        else:
            ok = False

    for fname in [x.strip() for x in args.shards.split(",") if x.strip()]:
        try:
            if not do_shard(args.repo, fname, args.dest, chunk, args.threads, hashes, args.fresh):
                ok = False
                # sha256 不匹配 → 该分片有残留损坏，整个分片重下（删干净重来）
                if verify(args.dest and os.path.join(args.dest, fname), fname,
                          os.path.getsize(os.path.join(args.dest, fname)), hashes) is False:
                    log(f"[RETRY] {fname} sha256 校验失败，删除后整片重下")
                    p = os.path.join(args.dest, fname)
                    for extra in (p + ".start", p + ".prefix"):
                        if os.path.exists(extra):
                            os.remove(extra)
                    if os.path.exists(p):
                        os.remove(p)
                    if do_shard(args.repo, fname, args.dest, chunk, args.threads, hashes, True):
                        log(f"[RETRY-OK] {fname} 重下后校验通过")
                    else:
                        ok = False
        except Exception as exc:  # noqa: BLE001
            log(f"[ERROR] {fname}: {exc}")
            ok = False

    log("ALL_DONE" if ok else "DONE_WITH_ERRORS")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
