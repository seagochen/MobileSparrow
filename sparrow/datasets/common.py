from typing import Tuple

import cv2
import numpy as np


def draw_gaussian(heatmap: np.ndarray, center: Tuple[int, int], radius: int, k: float = 1.0):
    """在 heatmap 上画带裁剪的高斯核。heatmap: H×W；center: (x, y)"""

    def gaussian2d(shape: Tuple[int, int], sigma: float) -> np.ndarray:
        h, w = shape
        y = np.arange(0, h, 1, dtype=np.float32)
        x = np.arange(0, w, 1, dtype=np.float32)
        yy, xx = np.meshgrid(y, x, indexing="ij")
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2))
        return g

    diameter = 2 * radius + 1
    gaussian = gaussian2d((diameter, diameter), sigma=diameter / 6.0)

    x, y = int(center[0]), int(center[1])
    h, w = heatmap.shape

    left, right = min(x, radius), min(w - x, radius + 1)
    top, bottom = min(y, radius), min(h - y, radius + 1)

    if right <= 0 or bottom <= 0 or left <= 0 or top <= 0:
        return

    masked_hm = heatmap[y - top:y + bottom, x - left:x + right]
    masked_g = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    np.maximum(masked_hm, masked_g * k, out=masked_hm)

def get_center_from_kps(kps_xyv: np.ndarray) -> Tuple[float, float]:
    """kps_xyv: [17,3]，只用 v>0 的点做均值"""
    vis = kps_xyv[:, 2] > 0
    if vis.sum() == 0:
        cx = kps_xyv[:, 0].mean()
        cy = kps_xyv[:, 1].mean()
    else:
        cx = kps_xyv[vis, 0].mean()
        cy = kps_xyv[vis, 1].mean()
    return float(cx), float(cy)

def letterbox(img: np.ndarray, dst_size: int, color=(114, 114, 114)) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    把任意 HxW 图像 letterbox 到 dst_size×dst_size，返回 (新图, 缩放比例, (pad_w, pad_h))
    - scale = dst_size / max(H, W)
    - 新图大小固定 dst_size×dst_size
    """
    h, w = img.shape[:2]
    scale = float(dst_size) / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img_resz = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    new_img = np.full((dst_size, dst_size, 3), color, dtype=img.dtype)
    pad_w = (dst_size - nw) // 2
    pad_h = (dst_size - nh) // 2
    new_img[pad_h:pad_h + nh, pad_w:pad_w + nw] = img_resz
    return new_img, scale, (pad_w, pad_h)

def apply_hsv(img: np.ndarray, hgain=0.015, sgain=0.7, vgain=0.4):
    if hgain == 0 and sgain == 0 and vgain == 0:
        return img
    r = np.random.uniform(-1, 1, 3) * np.array([hgain, sgain, vgain]) + 1.0
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    img_hsv[..., 0] = (img_hsv[..., 0] * r[0]) % 180.0
    img_hsv[..., 1] = np.clip(img_hsv[..., 1] * r[1], 0, 255.0)
    img_hsv[..., 2] = np.clip(img_hsv[..., 2] * r[2], 0, 255.0)
    img = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return img

def random_affine_points(pts: np.ndarray, M: np.ndarray) -> np.ndarray:
    """pts: [N,2]，仿射矩阵 2x3，输出映射后的 [N,2]"""
    if pts.size == 0:
        return pts
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    pts_aug = np.concatenate([pts, ones], axis=1)  # [N,3]
    pts_new = (M @ pts_aug.T).T  # [N,2]
    return pts_new
