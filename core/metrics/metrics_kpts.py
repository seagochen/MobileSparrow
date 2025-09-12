# -*- coding: utf-8 -*-
from typing import Tuple

import numpy as np
import torch


def _to_numpy(t: torch.Tensor):
    return t.detach().cpu().numpy()

def _getDist(pre: np.ndarray, labels: np.ndarray) -> np.ndarray:
    pre = pre.reshape([-1, 17, 2])
    labels = labels.reshape([-1, 17, 2])
    return (pre[:, :, 0] - labels[:, :, 0]) ** 2 + (pre[:, :, 1] - labels[:, :, 1]) ** 2

def _getAccRight(dist: np.ndarray, th: float = 5, img_size: int = 192) -> np.ndarray:
    thr = th / float(img_size)
    return np.sum(np.sqrt(dist) < thr, axis=0).astype(np.int64)

def pck_accuracy(output: np.ndarray, target: np.ndarray, img_size: int = 192) -> Tuple[np.ndarray, int]:
    """返回 (每个关节点的正确计数, batch中的总样本数)"""
    dist = _getDist(output, target)
    correct_counts = _getAccRight(dist, img_size=img_size)
    return correct_counts, output.shape[0]

def _maxPoint_numpy(hm: np.ndarray) -> tuple:
    B, C, H, W = hm.shape
    flat = hm.reshape(B, -1)
    ids = np.argmax(flat, axis=1)
    cy = (ids // W).astype(np.int32).reshape(B, 1)
    cx = (ids % W).astype(np.int32).reshape(B, 1)
    return cx, cy

def movenet_decode(data, kps_mask=None, mode='output', num_joints=17, hm_th=0.1) -> np.ndarray:
    # ... (这里粘贴您 task_kpts.py 中完整的 movenetDecode 函数代码) ...
    # ... (代码过长，此处省略，请直接从您的文件中复制) ...
    if mode == 'output':
        # unpack to numpy
        if isinstance(data, dict):
            hm_t, ct_t, rg_t, of_t = data["heatmaps"], data["centers"], data["regs"], data["offsets"]
        else:
            hm_t, ct_t, rg_t, of_t = data  # list/tuple
        heatmaps = _to_numpy(hm_t)  # [B,J,H,W]
        centers = _to_numpy(ct_t)  # [B,1,H,W]
        regs = _to_numpy(rg_t)  # [B,2J,H,W]
        offsets = _to_numpy(of_t)  # [B,2J,H,W]

        B, J, H, W = heatmaps.shape
        kps_mask_np = _to_numpy(kps_mask) if (kps_mask is not None) else np.ones((B, J), dtype=np.float32)

        heatmaps = heatmaps.copy()
        heatmaps[heatmaps < hm_th] = 0.0

        cx, cy = _maxPoint_numpy(centers)  # [B,1]
        dim0 = np.arange(B, dtype=np.int32).reshape(B, 1)
        dim1 = np.zeros((B, 1), dtype=np.int32)

        range_x = np.arange(W, dtype=np.float32).reshape(1, 1, 1, W)
        range_y = np.arange(H, dtype=np.float32).reshape(1, 1, H, 1)

        res = []
        for n in range(num_joints):
            # <修改 2025/09/11>
            # reg_x_o = (regs[dim0, dim1 + n * 2,     cy, cx] + 0.5).astype(np.int32)
            # reg_y_o = (regs[dim0, dim1 + n * 2 + 1, cy, cx] + 0.5).astype(np.int32)
            # reg_x = (reg_x_o + cx).clip(0, W - 1)
            # reg_y = (reg_y_o + cy).clip(0, H - 1)
            reg_x = (regs[dim0, dim1 + n * 2, cy, cx] + cx).astype(np.float32)
            reg_y = (regs[dim0, dim1 + n * 2 + 1, cy, cx] + cy).astype(np.float32)
            reg_x = np.clip(reg_x, 0, W - 1)
            reg_y = np.clip(reg_y, 0, H - 1)
            # ----

            reg_x_hw = np.broadcast_to(reg_x.reshape(B, 1, 1, 1), (B, 1, H, W)).astype(np.float32)
            reg_y_hw = np.broadcast_to(reg_y.reshape(B, 1, 1, 1), (B, 1, H, W)).astype(np.float32)
            d2 = (range_x - reg_x_hw) ** 2 + (range_y - reg_y_hw) ** 2
            tmp_reg = heatmaps[:, n:n + 1, :, :] / (np.sqrt(d2) + 1.8)

            jx, jy = _maxPoint_numpy(tmp_reg)
            jx = jx.clip(0, W - 1)
            jy = jy.clip(0, H - 1)

            score = heatmaps[dim0, dim1 + n, jy, jx]
            off_x = offsets[dim0, dim1 + n * 2, jy, jx]
            off_y = offsets[dim0, dim1 + n * 2 + 1, jy, jx]

            x_n = (jx + off_x) / float(W)
            y_n = (jy + off_y) / float(H)

            bad = np.logical_or((score < hm_th), (kps_mask_np[:, n:n + 1] < 0.5)).astype(np.float32)
            x_n = x_n * (1.0 - bad) + (-1.0) * bad
            y_n = y_n * (1.0 - bad) + (-1.0) * bad

            res.extend([x_n, y_n])
        res = np.concatenate(res, axis=1)
        return res

    elif mode == 'label':
        data_np = _to_numpy(data)
        B, C, H, W = data_np.shape
        J = num_joints

        heatmaps = data_np[:, :J, :, :]
        centers = data_np[:, J:J + 1, :, :]
        regs = data_np[:, J + 1:J + 1 + 2 * J, :, :]
        offsets = data_np[:, J + 1 + 2 * J:, :, :]

        kps_mask_np = _to_numpy(kps_mask) if (kps_mask is not None) else np.ones((B, J), dtype=np.float32)

        cx, cy = _maxPoint_numpy(centers)
        dim0 = np.arange(B, dtype=np.int32).reshape(B, 1)
        dim1 = np.zeros((B, 1), dtype=np.int32)

        res = []
        for n in range(J):
            # <修改 2025/09/11>
            # reg_x_o = (regs[dim0, dim1 + n * 2,     cy, cx] + 0.5).astype(np.int32)
            # reg_y_o = (regs[dim0, dim1 + n * 2 + 1, cy, cx] + 0.5).astype(np.int32)
            # jx = (reg_x_o + cx).clip(0, W - 1)
            # jy = (reg_y_o + cy).clip(0, H - 1)
            # 新（浮点，与 output 分支一致）
            reg_x = (regs[dim0, dim1 + n * 2, cy, cx] + cx).astype(np.float32)
            reg_y = (regs[dim0, dim1 + n * 2 + 1, cy, cx] + cy).astype(np.float32)
            # 关键修改：在clip之后，将浮点坐标转换为整数索引
            jx = np.clip(reg_x, 0, W - 1).astype(np.int32)
            jy = np.clip(reg_y, 0, H - 1).astype(np.int32)
            # ----

            off_x = offsets[dim0, dim1 + n * 2, jy, jx]
            off_y = offsets[dim0, dim1 + n * 2 + 1, jy, jx]

            x_n = (jx + off_x) / float(W)
            y_n = (jy + off_y) / float(H)

            mask = kps_mask_np[:, n:n + 1]
            x_n = x_n * mask + (-1.0) * (1.0 - mask)
            y_n = y_n * mask + (-1.0) * (1.0 - mask)

            res.extend([x_n, y_n])
        res = np.concatenate(res, axis=1)
        return res
    else:
        raise ValueError(f"unknown mode: {mode}")
