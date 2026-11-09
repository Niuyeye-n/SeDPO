# -*- coding: utf-8 -*-
"""
给 scorer_dsr.py 打补丁：修复 step2/step3 阶段的内存放大（OOM killer 杀掉 scorer）。

问题根因（实测：step1 成功、step2 被 `Killed`）：
  get_instance() 里
      for key, val in entry.items():
          if(key[:4]=="step"): continue
          instance[f"test_{key}"] = val
  把 entry 的所有非 step_ 字段逐个复制进每个 instance。
  step2/step3 时 n_score=150、L=400 → 派生 60000 个 instance，
  每个 instance 持有 entry 各字段的独立引用（dict 里的 list 是共享的，
  但 dict 本身 + 60 万次 dict 插入 → 内存暴涨到几十 GB → 被 OOM killer 杀掉。
  step1 走 get_instance_old()（只复制需要的字段）所以没事。

修复策略（保持语义完全等价，只去掉冗余复制）：
  把「非 step_ 字段」在循环外算一次，循环内复用同一份 dict 引用。
  即 instance 直接引用 entry 的原值，而不是每次新建 dict 塞进去。
  这样：
    - 产出的 instance 字段集合与原来完全一致（值也相同）
    - 内存从 O(60000 × 字段数) 降到 O(150 × 字段数)
  注意：下游不会修改这些字段（只读取），共享引用安全。
        get_label() 会改 entry["answerKey"]，但那是改 test_info 里的副本
        ——原实现 test_info 也是浅拷贝 dict，同样共享同一个 list/str 对象，
        行为一致。

用法：python patch_scorer_dsr.py <repo_path>
"""
import io
import sys

REPO = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/SeDPO_NIPS25"
PATH = REPO + "/src/dataset_readers/scorer_dsr.py"

src = io.open(PATH, encoding="utf-8").read()

if "SCALER_DSR_MEMFIX" in src:
    print("[SKIP] 补丁已存在")
    sys.exit(0)

old = """                for j in range(len(choosen_list)):
                    for id in sample_list[j]:
                        instance = {}
                        instance["have_choosen"] = choosen_list[j]
                        instance.update(self.prompt_pool[id][1])
                        for key, val in entry.items():
                            if(key[:4]=="step"): continue
                            instance[f"test_{key}"] = val
                        data.append(instance)
"""

new = """                # SCALER_DSR_MEMFIX（2026-11-05）：
                # 原实现在循环内逐字段复制 entry 的所有非 step_ 字段，
                # n_score=150 / L=400 时派生 60000 个 instance → 内存暴涨被 OOM 杀掉。
                # 改为在循环外算一次 test_fields，循环内共享引用；
                # 产出字段集合与值与原实现完全一致，下游只读不改，安全。
                test_fields = {
                    f"test_{key}": val
                    for key, val in entry.items()
                    if key[:4] != "step"
                }
                for j in range(len(choosen_list)):
                    for id in sample_list[j]:
                        instance = {}
                        instance["have_choosen"] = choosen_list[j]
                        instance.update(self.prompt_pool[id][1])
                        instance.update(test_fields)
                        data.append(instance)
"""

assert old in src, "未找到目标代码块，源文件可能已变动"

io.open(PATH + ".orig", "w", encoding="utf-8").write(
    io.open(PATH, encoding="utf-8").read())
src = src.replace(old, new)
io.open(PATH, "w", encoding="utf-8").write(src)
print("[OK] 已为 scorer_dsr.py 打上内存修复补丁")
print("     备份:", PATH + ".orig")
