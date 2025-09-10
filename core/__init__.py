"""
@Fire
https://github.com/fire717
"""
import os
import random

import numpy as np
import torch


def setRandomSeed(seed=42):
    """Reproducer for pytorch experiment.

    Parameters
    ----------
    seed: int, optional (default = 2019)
        Radnom seed.

    Example
    -------
    setRandomSeed(seed=2019).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True


def init(cfg):

    if cfg.get("cfg_verbose", False):
        print("=" * 80)
        print(cfg)
        print("=" * 80)

    # Use GPU 0
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg['GPU_ID']
    
    # Set a random int
    setRandomSeed(cfg.get('random_seed', random.randint(1, 1000)))

    # Create a folder to store the trained model weights
    if not os.path.exists(cfg.get('save_dir', 'output')):
        os.makedirs(cfg.get('save_dir', 'output'))