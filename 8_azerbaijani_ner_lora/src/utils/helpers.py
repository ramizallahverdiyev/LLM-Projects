import os
import random
import yaml
import numpy as np
import torch

def load_config(config_path: str = "configs/model_config.yaml"):
    """
    Load a YAML configuration file.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_dirs(path: str):
    os.makedirs(path, exist_ok=True)
