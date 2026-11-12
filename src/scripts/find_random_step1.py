import os
import sys
sys.path.insert(0, os.getcwd())
import hydra
import tqdm
import json
import random
import os
from DPR.dpr.utils.tasks import task_map
from pathlib import Path

class RandomFinder:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        os.makedirs(os.path.dirname(cfg.prompt_pool_path), exist_ok=True)
        os.makedirs(os.path.dirname(cfg.output_path), exist_ok=True)
        task = task_map.cls_dic[self.cfg.task_name]()
        self.train_dataset = task.get_dataset(
            split="train",
            ds_size=None if "ds_size" not in cfg else cfg.ds_size,
            cache_dir=cfg.cache_dir,
        )

        print("started creating the prompt pool")
        self.get_prompt_pool()
        print("finished creating the prompt pool")

    # sample and save prompt pool
    def get_prompt_pool(self):
        self.prompt_pool = self.train_dataset
        for i, entry in enumerate(self.prompt_pool):
            entry["id"] = i
            entry["task_name"] = self.cfg.task_name
        print("prompt_pool size", len(self.prompt_pool))
        with open(self.cfg.prompt_pool_path, "w") as f:
            json.dump(self.prompt_pool, f)

# for each task input, sample L prompts for scoring from the prompt pool (i.e., task training data)
def find(cfg):
    random_finder = RandomFinder(cfg)
    data_list = random_finder.train_dataset
    idx_list = list(range(len(random_finder.prompt_pool)))

    # 2026-11-05 规模控制：候选池(prompt_pool)保持全量，只限制「被打分的 query 条数」。
    # 目的：算力受限时压缩打分开销，但保持检索语料与基线 random3 完全一致，
    # 保证 SeDPO 与基线的对照公平。未设置 n_score 时行为与原版完全一致。
    n_score = cfg.get("n_score", None)
    if n_score is not None and int(n_score) > 0:
        n_score = int(n_score)
        if n_score < len(data_list):
            print(f"[INFO] 规模控制：打分 query 数 {len(data_list)} -> {n_score}"
                  f"（候选池仍为 {len(random_finder.prompt_pool)} 条，不变）")
            data_list = data_list[:n_score]

    for i, element in tqdm.tqdm(enumerate(data_list)):
        random.seed(i)
        element["id"] = i
        element["step_1_have_choosen"] = []
        # `ctxs` stores the sampled prompt ids 
        element["step_1_ctxs"] = [
            {"id": int(a)}
            for a in random.sample([idx for idx in idx_list if idx != i], k=min(cfg.L, len(idx_list)-1)) # avoid selecting the task input itself
        ]
    return data_list


@hydra.main(config_path="../configs", config_name="random_finder")
def main(cfg):
    print(cfg)
    if Path(cfg.output_path).exists() and Path(cfg.prompt_pool_path).exists():
        print(f"skip existed files: {cfg.output_path} {cfg.prompt_pool_path}")
        return
    data_list = find(cfg)
    with open(cfg.output_path, "w") as writer:
        writer.write(json.dumps(data_list, indent=4) + "\n")

if __name__ == "__main__":
    main()