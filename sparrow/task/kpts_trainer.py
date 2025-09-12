# -*- coding: utf-8 -*-
"""
合并版：kpts_trainer + metrics_kpts
建议文件名（符合 snake_case）：kpts_trainer_metrics.py

说明：
- 原先 metrics_kpts.py 中的 movenet_decode、pck_accuracy 及辅助函数已内联至本文件上半部分。
- KptsTrainer 直接使用本文件内的 movenet_decode 与 pck_accuracy。
- 其他依赖（BaseTrainer、MoveNetLoss）保持原导入路径不变。
"""

from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sparrow.task.base_trainer import BaseTrainer
from sparrow.loss.movenet_loss import MoveNetLoss


# ===============================
# metrics_kpts.py 的内容（合并）
# ===============================

def _to_numpy(t: torch.Tensor):
    return t.detach().cpu().numpy()


def _getDist(pre: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    计算预测与标签坐标的欧氏距离平方（逐关键点）。
    输入坐标为 [B, 2J] 或 [B, J, 2] 展开后统一到 [-1,1] 归一化空间。
    """
    pre = pre.reshape([-1, 17, 2])
    labels = labels.reshape([-1, 17, 2])
    return (pre[:, :, 0] - labels[:, :, 0]) ** 2 + (pre[:, :, 1] - labels[:, :, 1]) ** 2


def _getAccRight(dist: np.ndarray, th: float = 5, img_size: int = 192) -> np.ndarray:
    """
    依据 PCK 思想将阈值从像素空间（th 像素）换算到归一化空间进行统计。
    返回每个关键点在 batch 内预测正确的数量数组（shape=[J]）。
    """
    thr = th / float(img_size)
    return np.sum(np.sqrt(dist) < thr, axis=0).astype(np.int64)


def pck_accuracy(output: np.ndarray, target: np.ndarray, img_size: int = 192) -> Tuple[np.ndarray, int]:
    """
    基于坐标计算 PCK 正确计数（逐关键点）。
    返回 (每个关节点的正确计数[J], batch_size)。
    """
    dist = _getDist(output, target)
    correct_counts = _getAccRight(dist, img_size=img_size)
    return correct_counts, output.shape[0]


def _maxPoint_numpy(hm: np.ndarray) -> tuple:
    """
    取每个样本热力图中响应最大的像素坐标（argmax）。
    输入：hm: [B, C, H, W]
    返回： (cx, cy)，均为 shape=[B,1] 的 int32
    """
    B, C, H, W = hm.shape
    flat = hm.reshape(B, -1)
    ids = np.argmax(flat, axis=1)
    cy = (ids // W).astype(np.int32).reshape(B, 1)
    cx = (ids % W).astype(np.int32).reshape(B, 1)
    return cx, cy


def movenet_decode(
    data,
    kps_mask=None,
    *,
    mode: str = 'output',
    num_joints: int = 17,
    hm_th: float = 0.1
) -> np.ndarray:
    """
    将 MoveNet 风格的网络输出解码为归一化关键点坐标 [B, 2J]，或从标签特征图中提取“伪坐标”用于度量。
    - mode='output': data 为模型输出（dict 或 tuple/list）
    - mode='label' : data 为标签特征图（tensor）
    坐标范围为 [-1,1]，无效点用 -1 填充。
    """
    if mode == 'output':
        # unpack tensors -> numpy
        if isinstance(data, dict):
            hm_t, ct_t, rg_t, of_t = data["heatmaps"], data["centers"], data["regs"], data["offsets"]
        else:
            hm_t, ct_t, rg_t, of_t = data  # list/tuple

        heatmaps = _to_numpy(hm_t)   # [B, J, H, W]
        centers  = _to_numpy(ct_t)   # [B, 1, H, W]
        regs     = _to_numpy(rg_t)   # [B, 2J, H, W]
        offsets  = _to_numpy(of_t)   # [B, 2J, H, W]

        B, J, H, W = heatmaps.shape
        kps_mask_np = _to_numpy(kps_mask) if (kps_mask is not None) else np.ones((B, J), dtype=np.float32)

        # 低于阈值的热力图记为 0，弱化干扰
        heatmaps = heatmaps.copy()
        heatmaps[heatmaps < hm_th] = 0.0

        # 选全局 center 的峰值作为参考（cx, cy）
        cx, cy = _maxPoint_numpy(centers)  # [B,1], [B,1]

        dim0 = np.arange(B, dtype=np.int32).reshape(B, 1)
        dim1 = np.zeros((B, 1), dtype=np.int32)

        range_x = np.arange(W, dtype=np.float32).reshape(1, 1, 1, W)
        range_y = np.arange(H, dtype=np.float32).reshape(1, 1, H, 1)

        res = []
        for n in range(num_joints):
            # 以 center 为基准的回归偏移（保持浮点精度）
            reg_x = (regs[dim0, dim1 + n * 2,     cy, cx] + cx).astype(np.float32)
            reg_y = (regs[dim0, dim1 + n * 2 + 1, cy, cx] + cy).astype(np.float32)
            reg_x = np.clip(reg_x, 0, W - 1)
            reg_y = np.clip(reg_y, 0, H - 1)

            # 基于 heatmap + 距离抑制得到该关键点的候选峰值
            reg_x_hw = np.broadcast_to(reg_x.reshape(B, 1, 1, 1), (B, 1, H, W)).astype(np.float32)
            reg_y_hw = np.broadcast_to(reg_y.reshape(B, 1, 1, 1), (B, 1, H, W)).astype(np.float32)
            d2 = (range_x - reg_x_hw) ** 2 + (range_y - reg_y_hw) ** 2
            tmp_reg = heatmaps[:, n:n + 1, :, :] / (np.sqrt(d2) + 1.8)

            jx, jy = _maxPoint_numpy(tmp_reg)
            jx = jx.clip(0, W - 1)
            jy = jy.clip(0, H - 1)

            score = heatmaps[dim0, dim1 + n, jy, jx]
            off_x = offsets[dim0, dim1 + n * 2,     jy, jx]
            off_y = offsets[dim0, dim1 + n * 2 + 1, jy, jx]

            x_n = (jx + off_x) / float(W)
            y_n = (jy + off_y) / float(H)

            bad = np.logical_or((score < hm_th), (kps_mask_np[:, n:n + 1] < 0.5)).astype(np.float32)
            x_n = x_n * (1.0 - bad) + (-1.0) * bad
            y_n = y_n * (1.0 - bad) + (-1.0) * bad

            res.extend([x_n, y_n])

        res = np.concatenate(res, axis=1)  # [B, 2J]
        return res

    elif mode == 'label':
        data_np = _to_numpy(data)  # [B, C, H, W]
        B, C, H, W = data_np.shape
        J = num_joints

        heatmaps = data_np[:, :J, :, :]
        centers  = data_np[:, J:J + 1, :, :]
        regs     = data_np[:, J + 1:J + 1 + 2 * J, :, :]
        offsets  = data_np[:, J + 1 + 2 * J:, :, :]

        kps_mask_np = _to_numpy(kps_mask) if (kps_mask is not None) else np.ones((B, J), dtype=np.float32)

        cx, cy = _maxPoint_numpy(centers)
        dim0 = np.arange(B, dtype=np.int32).reshape(B, 1)
        dim1 = np.zeros((B, 1), dtype=np.int32)

        res = []
        for n in range(J):
            # 与 output 分支一致，保持浮点索引再离散化为像素索引
            reg_x = (regs[dim0, dim1 + n * 2,     cy, cx] + cx).astype(np.float32)
            reg_y = (regs[dim0, dim1 + n * 2 + 1, cy, cx] + cy).astype(np.float32)
            jx = np.clip(reg_x, 0, W - 1).astype(np.int32)
            jy = np.clip(reg_y, 0, H - 1).astype(np.int32)

            off_x = offsets[dim0, dim1 + n * 2,     jy, jx]
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


# ===============================
# kpts_trainer.py 的内容（合并）
# ===============================

class KptsTrainer(BaseTrainer):
    """
    关键点训练器：
    - 前向产生四头输出（heatmaps/centers/regs/offsets）
    - 对齐输出尺寸、计算 MoveNetLoss
    - 评估指标为 PCK（acc）
    """

    def __init__(self, model: nn.Module,
                 *,  # 使用 * 强制其后参数为关键字参数，提高可读性
                 epochs: int,
                 save_dir: str,
                 img_size: int,
                 target_stride: int = 4,
                 num_joints: int = 17,
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 # 优化器与调度器参数
                 optimizer_cfg: Dict = None,
                 scheduler_cfg: Dict = None,
                 # 训练技巧参数
                 use_amp: bool = True,
                 use_ema: bool = True,
                 ema_decay: float = 0.9998,
                 clip_grad_norm: float = 0.0,
                 # 日志参数
                 log_interval: int = 10):

        super().__init__(model,
                         epochs=epochs,
                         save_dir=save_dir,
                         device=device,
                         optimizer_cfg=optimizer_cfg or {},
                         scheduler_cfg=scheduler_cfg or {},
                         use_amp=use_amp,
                         use_ema=use_ema,
                         ema_decay=ema_decay,
                         clip_grad_norm=clip_grad_norm,
                         log_interval=log_interval)

        # 损失函数
        self.loss_func = MoveNetLoss(num_joints=num_joints)

        # 任务相关参数
        self.img_size = img_size
        self.target_stride = target_stride
        self.target_hw = (img_size // self.target_stride, img_size // self.target_stride)

    def _align_output(self, output: Dict) -> Dict:
        """
        将模型输出的四头 resize 到与标签一致的尺寸（H_out=W_out=img_size/stride）。
        注意：为与既有实现对齐，按原逻辑分别对四个头做双线性插值。
        """
        # 逐个 resize，保持与原工程逻辑一致（即便原始代码中以 list 传 F.interpolate）
        hm_r = F.interpolate(output["heatmaps"], size=self.target_hw, mode='bilinear', align_corners=False)
        ct_r = F.interpolate(output["centers"],  size=self.target_hw, mode='bilinear', align_corners=False)
        rg_r = F.interpolate(output["regs"],     size=self.target_hw, mode='bilinear', align_corners=False)
        of_r = F.interpolate(output["offsets"],  size=self.target_hw, mode='bilinear', align_corners=False)
        return {"heatmaps": hm_r, "centers": ct_r, "regs": rg_r, "offsets": of_r}

    def _calculate_loss(self, batch) -> Tuple[torch.Tensor, Dict[str, float]]:
        imgs, labels, kps_mask, _ = batch

        raw_out = self.model(imgs)
        out_dict = self._align_output(raw_out)
        out_list = [out_dict["heatmaps"], out_dict["centers"], out_dict["regs"], out_dict["offsets"]]

        hm_loss, b_loss, c_loss, r_loss, o_loss = self.loss_func(out_list, labels, kps_mask)
        total_loss = hm_loss + c_loss + r_loss + o_loss + b_loss

        loss_dict = {
            "loss": total_loss.item(),
            "hm": hm_loss.item(), "b": b_loss.item(), "c": c_loss.item(),
            "r": r_loss.item(), "o": o_loss.item()
        }
        return total_loss, loss_dict

    def _calculate_metrics(self, model, batch) -> Dict[str, float]:
        imgs, labels, kps_mask, _ = batch
        raw_out = model(imgs)
        out_dict = self._align_output(raw_out)

        # 解码得到预测与 GT 坐标（均为 [-1,1]，无效点为 -1）
        pred_coords = movenet_decode(out_dict, kps_mask, mode='output')
        gt_coords   = movenet_decode(labels,   kps_mask, mode='label')

        # 计算 PCK 正确率
        correct_counts, total_in_batch = pck_accuracy(pred_coords, gt_coords, img_size=self.img_size)
        mean_acc = np.mean(correct_counts / total_in_batch)

        return {"acc": float(mean_acc)}

    def _get_main_metric(self, metrics: Dict[str, float]) -> float:
        return metrics.get("acc", 0.0)

    def _move_batch_to_device(self, batch):
        imgs, labels, kps_mask, img_names = batch
        imgs = imgs.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)
        kps_mask = kps_mask.to(self.device, non_blocking=True)
        return imgs, labels, kps_mask, img_names
