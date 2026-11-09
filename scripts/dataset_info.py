# -*- coding: utf-8 -*-
"""打印三个数据集的规模与类别信息，用于实验报告"""
import os
import sys

REPO = "/root/autodl-tmp/SeDPO_NIPS25"
sys.path.insert(0, REPO)
os.chdir(REPO)

from DPR.dpr.utils.tasks import task_map

POOL = {
    "arc_c": "my_data_qwen/prompt_pool_sedpo/close_qa/arc_c_prompts.json",
    "openbookqa": "my_data_qwen/prompt_pool_sedpo/reading_new/openbookqa_prompts.json",
    "multirc": "my_data_qwen/prompt_pool_sedpo/reading_new/multirc_prompts.json",
}
# 候选池若不存在则退回主池
POOL_FALLBACK = {
    "arc_c": "my_data_qwen/prompt_pool/close_qa/arc_c_prompts.json",
    "openbookqa": "my_data_qwen/prompt_pool/reading_new/openbookqa_prompts.json",
    "multirc": "my_data_qwen/prompt_pool/reading_new/multirc_prompts.json",
}
L_BY_TASK = {"arc_c": 400, "openbookqa": 200, "multirc": 50}

for t in ["arc_c", "openbookqa", "multirc"]:
    p = os.path.join(REPO, POOL[t])
    if not os.path.exists(p):
        p = os.path.join(REPO, POOL_FALLBACK[t])
    import json
    pool = json.load(open(p, encoding="utf-8"))
    cls = task_map.cls_dic[t]()
    # 评测集
    ds = cls.get_dataset(cache_dir=os.path.join(REPO, "cache"))
    ans = cls.get_answers({"answerKey": "A", "answers": {"text": ["x"]},
                            "choices": {"text": ["a", "b", "c", "d"]},
                            "label": 0}) if t == "arc_c" else None
    print(f"### {t}")
    print(f"  候选池(train)条数 : {len(pool)}")
    print(f"  评测集条数        : {len(ds)}")
    print(f"  class_num(选项数) : {cls.class_num}")
    print(f"  metric            : {cls.metric}")
    print(f"  finder_L          : {L_BY_TASK[t]}")
    print(f"  候选池首条示例    : {str(pool[0].get('question', pool[0].get('choices','')))[:90]}")
    print()
