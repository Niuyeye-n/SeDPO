import json
import sys

REPO = "/root/autodl-tmp/SeDPO_NIPS25"
sys.path.insert(0, REPO)
from src.utils.metric import compute_scores  # noqa: E402

cases = [
    ("arc_c(已完成的 llama)", REPO + "/my_data_qwen/experiment_sedpo_llama/close_qa/preds_for_close_qa/arc_c_prompts3_retrieverse2_preds_0.json"),
    ("multirc(llama)", REPO + "/my_data_qwen/experiment_sedpo_llama/reading_new/preds_for_reading_new/multirc_prompts3_retrieverse2_preds_0.json"),
    ("multirc(qwen)", REPO + "/my_data_qwen/experiment_sedpo_qwen/reading_new/preds_for_reading_new/multirc_prompts3_retrieverse2_preds_0.json"),
]

for name, path in cases:
    try:
        d = json.load(open(path))
        print(name, "条数=", len(d), "字段=", list(d[0].keys()))
        print("   scores =", compute_scores("simple_accuracy", d))
        print("   n_prompts =", d[0].get("n_prompts"))
    except Exception as e:
        print(name, "[失败]", repr(e))
