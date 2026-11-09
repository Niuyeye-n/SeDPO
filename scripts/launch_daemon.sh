#!/bin/bash
# 守护式启动：用 setsid 让实验进程彻底脱离当前 SSH 会话。
# 原因：之前用 nohup 启动的实验在 step2 中途被杀（SSH 通道关闭 → SIGHUP 杀进程组），
#      setsid 会话彻底隔离，不会再被父终端的信号影响。
#
# 用法: bash /root/autodl-tmp/launch_daemon.sh <脚本路径> <日志路径>
set -u
SCRIPT=$1
LOG=$2

# setsid + nohup + 完全脱离标准输入
setsid nohup bash "$SCRIPT" > "$LOG" 2>&1 < /dev/null &
PID=$!
disown 2>/dev/null || true
sleep 3
echo "已启动: $SCRIPT"
echo "  PID=$PID (setsid 独立会话)"
echo "  日志: $LOG"
ps -eo pid,ppid,sid,args | grep -E "$(basename $SCRIPT)" | grep -v grep | head -3
