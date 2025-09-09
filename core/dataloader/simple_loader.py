import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from core.utils.paths import iter_image_paths


class SimpleImageFolder(Dataset):
    def __init__(self, root, img_size=192):
        # 统一用 iter_image_paths 收集图像路径
        self.paths = iter_image_paths(root, recursive=False)
        if not self.paths:
            raise FileNotFoundError(f"No images found in: {root}")
        self.img_size = int(img_size)

    def __len__(self):
        return len(self.paths)

    @staticmethod
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

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        
        # --- 恢复为使用 letterbox 的正确版本 ---
        
        # 1. 转换颜色空间：从 BGR (OpenCV 默认) 到 RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 2. 使用 letterbox 保持长宽比进行缩放和填充
        img = self._letterbox(img, self.img_size)
        
        # 3. 归一化和格式转换
        img = (img.astype(np.float32) / 255.0)
        img = np.transpose(img, (2, 0, 1))
        
        # 4. 转换为 PyTorch 张量
        img_tensor = torch.from_numpy(img)
        
        # --- 修改结束 ---

        return img_tensor, path