# seagochen/movenet.pytorch/movenet.pytorch-dev_debug/core/dataloader/simple_loader.py

import os
from glob import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple


def _letterbox(img: np.ndarray, dst_size: int, color=(114, 114, 114)) -> np.ndarray:
    """
    将任意 HxW 图像 letterbox 到 dst_size×dst_size，返回处理后的图像。
    - 保持原始长宽比
    - 用指定颜色填充空白区域
    """
    h, w = img.shape[:2]
    scale = float(dst_size) / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img_resz = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    new_img = np.full((dst_size, dst_size, 3), color, dtype=img.dtype)
    pad_w = (dst_size - nw) // 2
    pad_h = (dst_size - nh) // 2
    new_img[pad_h:pad_h + nh, pad_w:pad_w + nw] = img_resz
    return new_img

class SimpleImageFolder(Dataset):
    def __init__(self, root, img_size=192):
        exts = ["*.jpg","*.jpeg","*.png","*.bmp","*.webp"]
        paths = []
        for e in exts:
            paths.extend(glob(os.path.join(root, e)))
        if not paths:
            raise FileNotFoundError(f"No images found in: {root}")
        self.paths = sorted(paths)
        self.img_size = int(img_size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {p}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 关键改动：使用和训练时一致的 letterbox 方法
        img = _letterbox(img, self.img_size)
        
        img = (img.astype(np.float32) / 255.0)
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = torch.from_numpy(img)       # float32 tensor [3,H,W]
        return img, [os.path.basename(p)]