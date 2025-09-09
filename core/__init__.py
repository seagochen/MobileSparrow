"""
@Fire
https://github.com/fire717
"""
import os
import random

import numpy as np
import torch

from core.dataloader.dataloader import CoCo2017DataLoader
from core.task.task import Task
from core.models.movenet import MoveNet


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

    if cfg["cfg_verbose"]:
        print("=" * 80)
        print(cfg)
        print("=" * 80)

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg['GPU_ID']
    setRandomSeed(cfg['random_seed'])

    if not os.path.exists(cfg['save_dir']):
        os.makedirs(cfg['save_dir'])