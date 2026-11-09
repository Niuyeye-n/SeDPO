# -*- coding: utf-8 -*-
"""
给 scorer.py 打补丁：显式 fp16 加载模型，避免 fp32 中间态撑爆内存触发 OOM killer。

问题根因（实测：scorer 被 `Killed`）：
  scorer.py 原本
      self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=..., cache_dir=...)
      self.model = self.model.half().to(device)
  from_pretrained 不指定 dtype 时按 **fp32** 加载 7B 模型 = 28GB 内存，
  紧接着 .half() 又生成一份 fp16 副本（+14GB）→ 峰值约 42GB；
  叠加 cgroup 里已有的 33GB 页缓存（读模型权重 + 写 scored json 产生）
  → 超过 AutoDL 实例 90GB 的 cgroup 上限 → 被 OOM killer 杀掉。
  step1 侥幸跑过是因为当时页缓存还没涨上来；step2 必然爆。

修复：与已修好的 inference.py 保持一致，直接以 fp16 加载再上 GPU。
  数值上完全等价（随后 .half() 仍是 fp16），只是省掉 fp32 中间态。

用法：python patch_scorer_dtype.py <repo_path>
"""
import io
import sys

REPO = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/SeDPO_NIPS25"
PATH = REPO + "/scripts/scorer.py"

src = io.open(PATH, encoding="utf-8").read()

if "SCORER_DTYPE_FIX" in src:
    print("[SKIP] 补丁已存在")
    sys.exit(0)

old = """        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=cfg.model_name, cache_dir=cfg.cache_dir)
        self.model = self.model.half().to(self.accelerator.device)
"""

new = """        # SCORER_DTYPE_FIX（2026-11-05）：
        # 原实现不带 dtype，from_pretrained 会以 fp32 加载 7B 模型（28GB），
        # 紧接着 .half() 又生成 fp16 副本 → 峰值约 42GB，
        # 叠加已有页缓存后超过实例 90GB 的 cgroup 上限 → 被 OOM killer 杀掉。
        # 改为直接 fp16 加载再上 GPU，与 inference.py 一致，数值等价但省掉 fp32 中间态。
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=cfg.model_name,
            cache_dir=cfg.cache_dir,
            torch_dtype=torch.float16,
        )
        self.model = self.model.half().to(self.accelerator.device)
"""

assert old in src, "未找到目标代码块，源文件可能已变动"

io.open(PATH + ".orig", "w", encoding="utf-8").write(
    io.open(PATH, encoding="utf-8").read())
src = src.replace(old, new)
io.open(PATH, "w", encoding="utf-8").write(src)
print("[OK] 已为 scorer.py 打上 fp16 加载补丁")
print("     备份:", PATH + ".orig")
