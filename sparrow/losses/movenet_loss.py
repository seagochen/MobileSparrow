# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class GaussianFocalLoss(nn.Module):
    """
    CenterNet 风格的高斯 Focal Loss（适用于关键点热图）

    特点：
      - 兼容软标签（高斯分布的热图，值在 [0,1] 之间）
      - 对简单样本（预测接近真值）降低权重
      - 对困难样本（预测远离真值）增加权重

    输入：
      - logits: [B, K, H, W] - 未经 sigmoid 的原始预测
      - target: [B, K, H, W] - 高斯热图标签，值在 [0,1]，峰值为关键点位置
    """

    def __init__(self, alpha: float = 2.0, beta: float = 4.0, eps: float = 1e-6):
        """
        参数:
          alpha: 正负样本的 focal 权重指数（控制对困难样本的关注度）
          beta: 负样本的额外衰减指数（控制远离关键点区域的权重下降速度）
          eps: 数值稳定性常量，避免 log(0) 或除零
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        前向传播计算损失

        返回:
          标量损失值（按正样本数量归一化）
        """
        # 1. 计算预测概率和基础 BCE 损失
        p = torch.sigmoid(logits)  # [B,K,H,W] 转为概率
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")  # 逐元素 BCE

        # 2. 计算 Focal 权重
        # 正样本权重：预测越接近 1，权重越小（简单正样本）
        # (1-p)^alpha 当 p→1 时 →0，降低简单样本权重
        # 乘以 target 使得只在有标签的位置生效
        pos_w = (1.0 - p).clamp_min(self.eps).pow(self.alpha) * target

        # 负样本权重：预测越接近 0，权重越小（简单负样本）
        # p^alpha 当 p→0 时 →0，降低简单样本权重
        # (1-target)^beta 使得远离关键点的负样本权重快速衰减
        neg_w = (p).clamp_min(self.eps).pow(self.alpha) * (1.0 - target).pow(self.beta)

        # 3. 加权损失
        loss = (pos_w + neg_w) * bce  # [B,K,H,W]

        # 4. 归一化：按正样本数量（热图中 > 0 的像素）
        num_pos = (target > 0).sum().clamp_min(1)  # 至少为 1，避免除零
        return loss.sum() / num_pos


class MoveNet2HeadLoss(nn.Module):
    """
    单人姿态估计的双头损失函数

    包含两个分支：
      1. Heatmaps 损失：使用 Gaussian Focal Loss，关注关键点位置的准确性
      2. Offsets 损失：使用 L1 Loss，优化亚像素级别的精度

    设计思路：
      - Heatmaps 负责粗定位（像素级别）
      - Offsets 负责精细调整（亚像素级别）
    """

    def __init__(self,
                 num_joints: int = 17,
                 hm_weight: float = 1.0,
                 off_weight: float = 1.0,
                 focal_alpha: float = 2.0,
                 focal_beta: float = 4.0):
        """
        参数:
          num_joints: 关键点数量（如 COCO 的 17 个关键点）
          hm_weight: 热图损失的权重
          off_weight: 偏移损失的权重
          focal_alpha: Focal Loss 的 alpha 参数
          focal_beta: Focal Loss 的 beta 参数
        """
        super().__init__()
        self.K = num_joints
        self.hm_w = hm_weight
        self.off_w = off_weight
        self.focal = GaussianFocalLoss(alpha=focal_alpha, beta=focal_beta)
        self.l1sum = nn.L1Loss(reduction="sum")  # 使用 sum 归约，后续手动归一化

    def forward(self,
                pred_hm: torch.Tensor, # [B,K,H,W] logits
                pred_off: torch.Tensor, # [B,2K,H,W]
                labels: torch.Tensor,  # [B, K+2K, H, W]
                kps_masks: torch.Tensor):  # [B, K]
        """
        前向传播计算总损失

        参数:
          pred_hm: [B,K,H,W] - 热图 logits
          pred_off: [B,2K,H,W] - 偏移量（每个关键点的 x,y 偏移）
          labels: 标签张量 [B, K+2K, H, W]
            - 前 K 个通道：热图标签（高斯分布）
            - 后 2K 个通道：偏移量标签（x,y 交替排列）
          kps_masks: 关键点可见性掩码 [B, K]
            - 1: 可见，参与损失计算
            - 0: 不可见或被遮挡，不参与损失计算

        返回:
          total: 总损失（标量）
          dict: 各项损失的详细信息（用于日志记录）
        """
        B = labels.shape[0]
        K = self.K

        gt_hm = labels[:, :K]  # [B,K,H,W] - 热图标签
        gt_off = labels[:, K:]  # [B,2K,H,W] - 偏移标签

        # 2. 计算热图损失（所有位置参与）
        loss_hm = self.focal(pred_hm, gt_hm)

        # 3. 计算偏移损失（仅在正样本邻域 & 可见关节）
        # 正样本掩码：热图值 > 0 的位置（关键点周围的高斯邻域）
        pos_mask = (gt_hm > 0).float()  # [B,K,H,W]
        pos_mask_2 = pos_mask.repeat_interleave(2, dim=1)  # [B,2K,H,W] - 复制到 x,y 两个通道

        # 可见性掩码：只对可见的关键点计算偏移损失
        vis = kps_masks.float().unsqueeze(-1).unsqueeze(-1)  # [B,K,1,1] - 扩展到空间维度
        vis_2 = vis.repeat_interleave(2, dim=1)  # [B,2K,1,1] - x,y 通道

        # 综合掩码：同时满足"在正样本邻域"和"关键点可见"
        valid = pos_mask_2 * vis_2  # [B,2K,H,W]
        denom = valid.sum().clamp_min(1.0)  # 有效像素总数（用于归一化）

        # L1 损失：只计算有效区域，并按有效像素数归一化
        loss_off = self.l1sum(pred_off * valid, gt_off * valid) / denom

        # 4. 加权合并损失
        total = self.hm_w * loss_hm + self.off_w * loss_off

        # 5. 返回总损失和详细信息（detach 避免影响梯度）
        return total, {
            "loss_heatmap": loss_hm.detach(),
            "loss_offsets": loss_off.detach(),
        }