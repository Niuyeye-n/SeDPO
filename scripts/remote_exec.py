# -*- coding: utf-8 -*-
"""
AutoDL 远程操作小工具（带密码的 SSH，免交互）

配置：同目录 remote_config.json
用法：
    python remote_exec.py exec "nvidia-smi"
    python remote_exec.py exec "ls /root" --timeout 600
    python remote_exec.py put 本地文件 /root/远端路径
    python remote_exec.py get /root/远端文件 本地路径
    python remote_exec.py putdir 本地目录 /root/远端目录
"""
import argparse
import json
import os
import posixpath
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), ".pylibs"))

import paramiko  # noqa: E402


def load_cfg():
    path = os.path.join(HERE, "remote_config.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def connect(cfg, timeout=30):
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(
        hostname=cfg["host"],
        port=int(cfg.get("port", 22)),
        username=cfg.get("user", "root"),
        password=cfg["password"],
        timeout=timeout,
        banner_timeout=timeout,
        auth_timeout=timeout,
        look_for_keys=False,
        allow_agent=False,
    )
    return cli


def cmd_exec(cli, command, timeout=None, stdin_bytes=None):
    chan = cli.get_transport().open_session()
    chan.settimeout(timeout)
    chan.get_pty(width=200, height=50)
    chan.exec_command(command)
    if stdin_bytes is not None:
        chan.sendall(stdin_bytes)
        chan.shutdown_write()
    out, err = b"", b""
    start = time.time()
    while True:
        if chan.recv_ready():
            data = chan.recv(65536)
            out += data
            sys.stdout.write(data.decode("utf-8", "ignore"))
            sys.stdout.flush()
        if chan.recv_stderr_ready():
            data = chan.recv_stderr(65536)
            err += data
            sys.stdout.write(data.decode("utf-8", "ignore"))
            sys.stdout.flush()
        if chan.exit_status_ready() and not chan.recv_ready() and not chan.recv_stderr_ready():
            break
        if timeout and time.time() - start > timeout:
            chan.close()
            print(f"\n[超时 {timeout}s，已断开]")
            return 124, out.decode("utf-8", "ignore"), err.decode("utf-8", "ignore")
        time.sleep(0.15)
    code = chan.recv_exit_status()
    print(f"\n[exit={code}, 耗时 {time.time() - start:.1f}s]")
    return code, out.decode("utf-8", "ignore"), err.decode("utf-8", "ignore")


def do_put(cli, local, remote):
    # 该实例的 SFTP 写通道不稳定，改走 SSH 标准输入 + base64 传输
    import base64

    with open(local, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("ascii")
    cmd = f"echo '{b64}' | base64 -d > '{remote}' && ls -l '{remote}'"
    code, _o, _e = cmd_exec(cli, cmd, timeout=300)
    if code != 0:
        raise RuntimeError(f"上传失败: {local} -> {remote}")
    print(f"已上传 {local} -> {remote} ({len(data)} 字节)")


def do_get(cli, remote, local):
    # paramiko 的 get() 在 Windows 下会把本地 C:/... 当成远端相对路径，故手动读写
    os.makedirs(os.path.dirname(os.path.abspath(local)) or ".", exist_ok=True)
    sftp = cli.open_sftp()
    with sftp.open(remote, "rb") as f:
        data = f.read()
    with open(local, "wb") as f:
        f.write(data)
    sftp.close()
    print(f"已下载 {remote} -> {local} ({len(data)} 字节)")


def do_putdir(cli, local, remote):
    sftp = cli.open_sftp()
    count = 0
    for root, _dirs, files in os.walk(local):
        rel = os.path.relpath(root, local).replace("\\", "/")
        rdir = remote if rel == "." else posixpath.join(remote, rel)
        try:
            sftp.stat(rdir)
        except IOError:
            sftp.mkdir(rdir)
        for fn in files:
            sftp.put(os.path.join(root, fn), posixpath.join(rdir, fn))
            count += 1
    sftp.close()
    print(f"已上传目录 {local} -> {remote}（{count} 个文件）")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="action", required=True)

    p_exec = sub.add_parser("exec")
    p_exec.add_argument("command")
    p_exec.add_argument("--timeout", type=int, default=None)

    p_put = sub.add_parser("put")
    p_put.add_argument("local")
    p_put.add_argument("remote")

    p_get = sub.add_parser("get")
    p_get.add_argument("remote")
    p_get.add_argument("local")

    p_putdir = sub.add_parser("putdir")
    p_putdir.add_argument("local")
    p_putdir.add_argument("remote")

    args = ap.parse_args()
    cfg = load_cfg()
    cli = connect(cfg)
    try:
        if args.action == "exec":
            code, _o, _e = cmd_exec(cli, args.command, args.timeout)
            sys.exit(code)
        elif args.action == "put":
            do_put(cli, args.local, args.remote)
        elif args.action == "get":
            do_get(cli, args.remote, args.local)
        elif args.action == "putdir":
            do_putdir(cli, args.local, args.remote)
    finally:
        cli.close()


if __name__ == "__main__":
    main()
