# -*- coding: utf-8 -*-
"""
MoveNet 损失函数（重构版）

包含五个核心损失组件：
1. Heatmap Loss: 监督关节点位置的主要热图。
2. Center Loss: 监督人体中心点热图。
3. Regression Loss: 监督从中心点到各关节点的粗略位移向量。
4. Offset Loss: 监督各关节点的亚像素偏移，用于精调定位。
5. Bone Loss: 作为正则化项，隐式地监督人体骨骼结构的合理性。
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Dict


class BoneLoss(nn.Module):
    """
    计算预测热图与目标热图之间，成对骨骼长度（范数）差异的损失。
    这是一种结构化的正则化，鼓励模型生成符合人体工学的姿态。
    """
    def __init__(self, num_joints: int = 17):
        super().__init__()
        # 创建所有关节点对的索引
        indices_i, indices_j = [], []
        for i in range(num_joints):
            for j in range(i + 1, num_joints):
                indices_i.append(i)
                indices_j.append(j)
        self.indices_i = indices_i
        self.indices_j = indices_j

    def forward(self, pred_heatmaps: torch.Tensor, gt_heatmaps: torch.Tensor) -> torch.Tensor:
        # [B, J, H, W] -> [B, num_pairs]
        pred_norm = torch.norm(pred_heatmaps[:, self.indices_i] - pred_heatmaps[:, self.indices_j], p=2, dim=(-2, -1))
        gt_norm = torch.norm(gt_heatmaps[:, self.indices_i] - gt_heatmaps[:, self.indices_j], p=2, dim=(-2, -1))
        
        loss = torch.abs(pred_norm - gt_norm)
        
        # 按 batch size 和骨骼数量进行平均
        batch_size = pred_heatmaps.shape[0]
        return torch.sum(loss) / batch_size / len(self.indices_i)


def _get_max_point(heatmap: torch.Tensor, center_weight: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """ 在特征图上找到峰值点坐标 """
    if center_weight is not None:
        heatmap = heatmap * center_weight

    B, _, H, W = heatmap.shape
    flat = heatmap.view(B, -1)
    _, max_id = torch.max(flat, dim=1)

    y = (max_id // W).long()
    x = (max_id % W).long()
    return x, y


class MoveNetLoss(nn.Module):
    def __init__(self, num_joints: int = 17):
        super().__init__()
        self.num_joints = num_joints
        self.bone_loss = BoneLoss(num_joints=self.num_joints)
        self._center_weight_cache: Dict[Tuple, torch.Tensor] = {}

    # --- 核心损失计算 ---
    def forward(self,
                predictions: List[torch.Tensor],
                targets: torch.Tensor,
                kps_mask: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        pred_heatmaps, pred_centers, pred_regs, pred_offsets = predictions
        B, _, H, W = pred_heatmaps.shape
        
        # 1. 从集成的 target tensor 中解析出各地图
        gt_heatmaps = targets[:, :self.num_joints]
        gt_centers = targets[:, self.num_joints:self.num_joints + 1]
        gt_regs = targets[:, self.num_joints + 1:self.num_joints + 1 + 2 * self.num_joints]
        gt_offsets = targets[:, self.num_joints + 1 + 2 * self.num_joints:]

        # 2. 计算各项损失
        loss_heatmap = self._weighted_mse_loss(pred_heatmaps, gt_heatmaps)
        loss_bone = self.bone_loss(pred_heatmaps, gt_heatmaps)
        loss_center = self._weighted_mse_loss(pred_centers, gt_centers)

        # 3. 找到GT中心点，作为回归和偏移损失的锚点
        center_weight = self._get_center_weight((B, H, W), device=targets.device, dtype=targets.dtype)
        center_x, center_y = _get_max_point(gt_centers, center_weight)

        loss_regs = self._compute_regs_loss(pred_regs, gt_regs, center_x, center_y, kps_mask)
        loss_offset = self._compute_offset_loss(pred_offsets, gt_offsets, center_x, center_y, gt_regs, kps_mask)

        return loss_heatmap, loss_bone, loss_center, loss_regs, loss_offset

    # --- 私有辅助方法 ---
    @staticmethod
    def _weighted_mse_loss(pred: torch.Tensor, gt: torch.Tensor, weight: float = 8.0) -> torch.Tensor:
        """ 带前景加权的均方误差损失 """
        loss = torch.pow((pred - gt), 2)
        weight_mask = gt * weight + 1.0
        loss = loss * weight_mask
        # 按 batch 和 channel 平均
        return torch.sum(loss) / (gt.shape[0] * gt.shape[1])

    @staticmethod
    def _l1_loss(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """ 带掩码的L1损失 """
        loss = torch.abs(pred - gt) * mask
        # 仅对有效数据点（mask.sum()）进行平均
        loss = torch.sum(loss) / (mask.sum() + 1e-4)
        return loss

    def _compute_regs_loss(self, pred_regs, gt_regs, center_x, center_y, kps_mask) -> torch.Tensor:
        """ 计算回归损失 (向量化版本) """
        B, _, H, W = pred_regs.shape
        batch_indices = torch.arange(B, device=pred_regs.device)

        # 在 GT 中心点 (center_y, center_x) 提取所有回归预测值和目标值
        # pred_at_center 的形状: [B, 2*J]
        pred_at_center = pred_regs[batch_indices, :, center_y, center_x]
        gt_at_center = gt_regs[batch_indices, :, center_y, center_x]

        # Reshape for easier calculation: [B, 2*J] -> [B, J, 2]
        pred_vectors = pred_at_center.view(B, self.num_joints, 2)
        gt_vectors = gt_at_center.view(B, self.num_joints, 2)
        
        # 扩展 kps_mask 以匹配向量维度: [B, J] -> [B, J, 1]
        mask = kps_mask.unsqueeze(-1)

        # 计算带掩码的 L1 损失
        return self._l1_loss(pred_vectors, gt_vectors, mask)

    def _compute_offset_loss(self, pred_offsets, gt_offsets, center_x, center_y, gt_regs, kps_mask) -> torch.Tensor:
        """
        计算偏移量（offset）损失。
        pred_offsets / gt_offsets: [B, 2*J, H, W]
        center_x / center_y:       [B]，每张图人体中心栅格坐标（int）
        gt_regs:                   [B, 2*J, H, W]，中心到各关节点的粗略位移向量
        kps_mask:                  [B, J]，有效关节点掩码
        """
        B, _, H, W = pred_offsets.shape
        device = pred_offsets.device
        dtype = pred_offsets.dtype

        batch_idx = torch.arange(B, device=device)

        # 1) 取出每张图在中心处的 GT 回归向量并 reshape 到 [B, J, 2]
        #    高级索引：对 [B, C, H, W] 用 (batch, :, y, x) 逐样本取值，得到 [B, 2*J]
        gt_regs_at_center = gt_regs[batch_idx, :, center_y, center_x]  # [B, 2*J]
        gt_vectors = gt_regs_at_center.view(B, self.num_joints, 2)  # [B, J, 2]

        # 2) 把中心坐标扩展到 [B, 1, 2] 再与向量相加 -> 得到各关节在特征图上的浮点坐标 [B, J, 2]
        #    这是修复点：原来把 [B, J, 2] + [B, 2] 直接相加会在 J 维发生错误广播。
        center_xy = torch.stack([center_x, center_y], dim=-1).to(gt_vectors.dtype)  # [B, 2]
        center_xy = center_xy.unsqueeze(1)  # [B, 1, 2]
        gt_kps_coords = gt_vectors + center_xy  # [B, J, 2]

        # 3) 转为整数栅格坐标并裁剪到边界
        gt_kps_x_int = torch.clamp(gt_kps_coords[..., 0], 0, W - 1).long()  # [B, J]
        gt_kps_y_int = torch.clamp(gt_kps_coords[..., 1], 0, H - 1).long()  # [B, J]

        # 4) 在这些坐标处采样预测/GT 的 offset，得到 [B, J, 2]
        batch_idx_flat = batch_idx.view(B, 1).expand(-1, self.num_joints)  # [B, J]
        joint_idx = torch.arange(self.num_joints, device=device).view(1, -1).expand(B, -1)

        p4 = pred_offsets.view(B, self.num_joints, 2, H, W)
        g4 = gt_offsets.view(B, self.num_joints, 2, H, W)

        pred_off_x = p4[batch_idx_flat, joint_idx, 0, gt_kps_y_int, gt_kps_x_int]  # [B, J]
        pred_off_y = p4[batch_idx_flat, joint_idx, 1, gt_kps_y_int, gt_kps_x_int]  # [B, J]
        gt_off_x = g4[batch_idx_flat, joint_idx, 0, gt_kps_y_int, gt_kps_x_int]  # [B, J]
        gt_off_y = g4[batch_idx_flat, joint_idx, 1, gt_kps_y_int, gt_kps_x_int]  # [B, J]

        pred_offsets_gathered = torch.stack([pred_off_x, pred_off_y], dim=-1)  # [B, J, 2]
        gt_offsets_gathered = torch.stack([gt_off_x, gt_off_y], dim=-1)  # [B, J, 2]

        # 5) 按关节掩码做 L1
        mask = kps_mask.unsqueeze(-1)  # [B, J, 1]
        return self._l1_loss(pred_offsets_gathered, gt_offsets_gathered, mask)

    @staticmethod
    def _create_center_weight(hw: Tuple[int, int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """ 动态生成高斯中心权重图 """
        H, W = hw
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device, dtype=dtype),
            torch.linspace(-1, 1, W, device=device, dtype=dtype),
            indexing="ij"
        )
        # sigma^2 约等于 0.15
        weight = torch.exp(-(xx ** 2 + yy ** 2) / 0.15)
        return weight

    def _get_center_weight(self, shape: Tuple, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """ 返回与输入尺寸匹配的中心权重图，使用缓存 """
        B, H, W = shape
        cache_key = (H, W, device, dtype)
        
        if cache_key not in self._center_weight_cache:
            weight_hw = self._create_center_weight((H, W), device, dtype)
            self._center_weight_cache[cache_key] = weight_hw
        
        weight = self._center_weight_cache[cache_key]
        return weight.view(1, 1, H, W).expand(B, -1, -1, -1).requires_grad_(False)