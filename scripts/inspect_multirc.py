import json
import os
import sys

REPO = "/root/autodl-tmp/SeDPO_NIPS25"
sys.path.insert(0, REPO)
from src.utils.metric import compute_scores  # noqa: E402


def load_concat(path):
    """文件里可能拼接了多个 JSON 值（每个是一批数据），全部解析并展平。"""
    raw = open(path, encoding="utf-8").read()
    dec = json.JSONDecoder()
    i, chunks = 0, 0
    out = []
    n = len(raw)
    while i < n:
        while i < n and raw[i] in " \r\n\t":
            i += 1
        if i >= n:
            break
        obj, end = dec.raw_decode(raw, i)
        chunks += 1
        stack = [obj]
        while stack:
            x = stack.pop()
            if isinstance(x, list):
                stack.extend(x)
            elif isinstance(x, dict):
                out.append(x)
            else:
                pass
        i = end
    return out, chunks


for tag in ("llama", "qwen"):
    p = f"{REPO}/my_data_qwen/experiment_sedpo_{tag}/reading_new/preds_for_reading_new/multirc_prompts3_retrieverse2_preds_0.json"
    print("=" * 70)
    print(tag, "multirc", os.path.getsize(p), "bytes")
    items, chunks = load_concat(p)
    print("  JSON 段数:", chunks, " 展平后条目:", len(items))
    if items:
        print("  字段:", list(items[0].keys()))
        np_set = sorted({str(x.get("n_prompts")) for x in items})
        print("  n_prompts 取值:", np_set[:10])
        print("  label 分布:", {k: sum(1 for x in items if x.get("label") == k) for k in {x.get("label") for x in items}})
        print("  pred 分布:", {k: sum(1 for x in items if x.get("pred") == k) for k in {x.get("pred") for x in items}})
        print("  scores:", compute_scores("simple_accuracy", items))

# 参考：正常格式的 arc_c preds
p = f"{REPO}/my_data_qwen/experiment_sedpo_llama/close_qa/preds_for_close_qa/arc_c_prompts3_retrieverse2_preds_0.json"
d = json.load(open(p))
print("=" * 70)
print("对照 arc_c: 条目", len(d), "n_prompts", d[0].get("n_prompts"))
