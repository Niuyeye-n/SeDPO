"""Validate Hydra training overrides without its legacy launcher --flag rewrite."""
from pathlib import Path
import sys
from hydra import compose,initialize_config_dir
from omegaconf import OmegaConf
root=Path(__file__).resolve().parents[1]
with initialize_config_dir(config_dir=str(root/'src/DPR/conf'),version_base='1.1'):
    cfg=compose(config_name='biencoder_train_cfg',overrides=sys.argv[1:])
    OmegaConf.resolve(cfg)
    assert cfg.train.batch_size==8 and cfg.train.num_train_epochs==6
    assert cfg.seed==12345 and cfg.model_file is None and cfg.fp16 is False
    print(OmegaConf.to_yaml(cfg))
