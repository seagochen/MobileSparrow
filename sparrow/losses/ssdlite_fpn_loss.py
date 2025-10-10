# -*- coding: utf-8 -*-
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou, sigmoid_focal_loss


class AnchorGenerator:
    """
    Anchor 坐标转换工具类

    功能：提供边界框坐标格式之间的转换
      - xyxy 格式：(x1, y1, x2, y2) - 左上角和右下角坐标
      - cxcywh 格式：(cx, cy, w, h) - 中心点坐标和宽高

    说明：训练时 anchors 由模型直接生成，此类仅用于坐标转换
    """

    @staticmethod
    def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        """
        将中心点格式转换为角点格式

        参数:
          boxes: [N, 4] - (cx, cy, w, h) 格式的边界框

        返回:
          [N, 4] - (x1, y1, x2, y2) 格式的边界框

        计算：
          x1 = cx - w/2,  y1 = cy - h/2
          x2 = cx + w/2,  y2 = cy + h/2
        """
        return torch.cat([boxes[:, :2] - boxes[:, 2:] / 2,  # 左上角 (x1, y1)
                          boxes[:, :2] + boxes[:, 2:] / 2], dim=1)  # 右下角 (x2, y2)

    @staticmethod
    def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
        """
        将角点格式转换为中心点格式

        参数:
          boxes: [N, 4] - (x1, y1, x2, y2) 格式的边界框

        返回:
          [N, 4] - (cx, cy, w, h) 格式的边界框

        计算：
          w = x2 - x1,  h = y2 - y1
          cx = x1 + w/2,  cy = y1 + h/2
        """
        w = boxes[:, 2] - boxes[:, 0]  # 宽度
        h = boxes[:, 3] - boxes[:, 1]  # 高度
        cx = boxes[:, 0] + w / 2  # 中心点 x
        cy = boxes[:, 1] + h / 2  # 中心点 y
        return torch.stack([cx, cy, w, h], dim=1)


class SSDLoss(nn.Module):
    """
    SSD/RetinaNet 风格的目标检测损失函数

    组成：
      1. 分类损失：Sigmoid Focal Loss（解决类别不平衡）
      2. 回归损失：Smooth L1 Loss（边界框回归）

    关键改进：
      - 分类损失按"正样本数"归一化，避免被大量负样本淹没
      - 引入 ignore 区域（IoU 在 [neg, pos] 之间的灰色地带）
      - 强制匹配策略：确保每个 GT 至少有一个正样本
    """

    def __init__(self,
                 num_classes: int,
                 iou_threshold_pos: float = 0.5,
                 iou_threshold_neg: float = 0.4,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0,
                 reg_weight: float = 1.0,
                 cls_weight: float = 1.0,
                 use_awl: bool = False):
        """
        参数:
          num_classes: 前景类别数（不包括背景）
          iou_threshold_pos: 正样本 IoU 阈值（≥ 该值视为正样本）
          iou_threshold_neg: 负样本 IoU 阈值（< 该值视为背景）
          focal_alpha: Focal Loss 的 alpha 参数（正负样本平衡）
          focal_gamma: Focal Loss 的 gamma 参数（困难样本关注度）
          cls_weight: Weight of classification loss
          reg_weight: Weight of regression loss
          use_awl: Whether to use AWL (automatic weight loss)
        """
        super().__init__()
        self.num_classes = int(num_classes)
        self.iou_threshold_pos = float(iou_threshold_pos)
        self.iou_threshold_neg = float(iou_threshold_neg)
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)
        self.reg_weight = reg_weight
        self.cls_weight = cls_weight
        self.use_awl = use_awl

        self.anchor_utils = AnchorGenerator()
        # 边界框编码的标准差（用于归一化回归目标）
        # 作为 buffer 注册，会随模型移动到对应设备
        self.register_buffer("bbox_std", torch.tensor([0.1, 0.1, 0.2, 0.2], dtype=torch.float32))

    # -------- 目标分配 --------
    def assign_targets_to_anchors(self, anchors: torch.Tensor, targets: List[torch.Tensor]):
        """
        为每个 anchor 分配训练目标（GT 匹配）

        匹配策略：
          1. IoU >= pos_threshold: 正样本
          2. IoU < neg_threshold: 背景
          3. neg_threshold <= IoU < pos_threshold: 忽略（不参与训练）
          4. 强制匹配：每个 GT 的最佳 anchor 必须为正样本（即使 IoU < pos_threshold）

        参数:
          anchors: [A, 4] - xyxy 格式的 anchor boxes
          targets: List[Tensor] - 长度为 B 的列表，每个元素为 [Ni, 5]
                   格式：(cls, x1, y1, x2, y2)，cls 为类别标签

        返回:
          labels: [B, A] - 每个 anchor 的标签
                  0~C-1: 前景类别
                  C: 背景
                  -1: 忽略（不计算损失）
          matched_gt_boxes: [B, A, 4] - 每个 anchor 匹配的 GT 边界框（xyxy）
        """
        device = anchors.device
        B = len(targets)
        A = anchors.shape[0]

        # 初始化：默认所有 anchor 为背景（类别 C）
        labels = torch.full((B, A), self.num_classes, dtype=torch.int64, device=device)
        matched = torch.zeros((B, A, 4), dtype=torch.float32, device=device)

        for b in range(B):
            tgt = targets[b]
            if tgt.numel() == 0:  # 如果该图像没有目标，跳过
                continue
            gt_cls = tgt[:, 0].to(torch.int64)  # GT 类别标签
            gt_box = tgt[:, 1:5]  # GT 边界框 (x1, y1, x2, y2)
            iou = box_iou(gt_box, anchors)  # [Ng, A] - 计算 IoU 矩阵

            # 1. 每个 anchor 的最好 GT（基于 IoU 匹配）
            iou_a, idx_a = iou.max(dim=0)  # [A], [A] - 每个 anchor 的最大 IoU 和对应的 GT 索引

            # 2. 每个 GT 的最好 anchor（确保至少一个正样本）
            _, idx_g = iou.max(dim=1)  # [Ng] - 每个 GT 的最佳 anchor 索引

            # 3. 标记 ignore 区域（IoU 在灰色地带）
            ignore = (iou_a >= self.iou_threshold_neg) & (iou_a < self.iou_threshold_pos)
            labels[b, ignore] = -1

            # 4. 标记正样本（IoU >= pos_threshold）
            pos = iou_a >= self.iou_threshold_pos
            if pos.any():
                labels[b, pos] = gt_cls[idx_a[pos]]  # 分配类别标签
                matched[b, pos] = gt_box[idx_a[pos]]  # 分配匹配的 GT 边界框

            # 5. 强制匹配：每个 GT 的最佳 anchor 必须为正样本
            # 这保证了小目标或难以匹配的 GT 也能有至少一个正样本
            labels[b, idx_g] = gt_cls
            matched[b, idx_g] = gt_box

        return labels, matched

    # -------- 边界框编码 --------
    def encode_bbox(self, anchors_xyxy: torch.Tensor, gt_xyxy: torch.Tensor) -> torch.Tensor:
        """
        将 GT 边界框相对于 anchor 进行编码（Faster R-CNN 风格）

        编码公式：
          tx = (gt_cx - anc_cx) / anc_w
          ty = (gt_cy - anc_cy) / anc_h
          tw = log(gt_w / anc_w)
          th = log(gt_h / anc_h)
          deltas = [tx, ty, tw, th] / std  # 标准化

        参数:
          anchors_xyxy: [N, 4] - anchor boxes (xyxy)
          gt_xyxy: [N, 4] - ground truth boxes (xyxy)

        返回:
          deltas: [N, 4] - 编码后的偏移量（标准化并裁剪到 [-4, 4]）
        """
        # 1. 转换为中心点格式
        a = self.anchor_utils.xyxy_to_cxcywh(anchors_xyxy)  # [N, 4] (cx, cy, w, h)
        g = self.anchor_utils.xyxy_to_cxcywh(gt_xyxy)  # [N, 4] (cx, cy, w, h)

        # 2. 计算相对偏移
        tx = (g[:, 0] - a[:, 0]) / a[:, 2]  # x 方向的相对偏移（归一化到 anchor 宽度）
        ty = (g[:, 1] - a[:, 1]) / a[:, 3]  # y 方向的相对偏移（归一化到 anchor 高度）
        tw = torch.log((g[:, 2] / a[:, 2]).clamp(min=1e-6))  # 宽度的对数尺度变化
        th = torch.log((g[:, 3] / a[:, 3]).clamp(min=1e-6))  # 高度的对数尺度变化

        # 3. 堆叠并标准化
        deltas = torch.stack([tx, ty, tw, th], dim=1)
        std = self.bbox_std.to(device=deltas.device, dtype=deltas.dtype)
        # 标准化并裁剪到 [-4, 4]（防止梯度爆炸）
        return (deltas / std).clamp(min=-4.0, max=4.0)

    # -------- 前向传播 --------
    def forward(self,
                anchors: torch.Tensor,  # [A,4] 或 [B,A,4]
                cls_preds: torch.Tensor,  # [B,A,C] (logits)
                reg_preds: torch.Tensor,  # [B,A,4]
                targets: List[torch.Tensor]):  # len=B, each [Ni,5]
        """
        计算总损失

        参数:
          anchors: [A,4] 或 [B,A,4] - anchor boxes（允许批次维度，会自动去重）
          cls_preds: [B,A,C] - 分类 logits（未经 sigmoid）
          reg_preds: [B,A,4] - 边界框回归预测（编码后的偏移量）
          targets: List[Tensor] - 长度为 B，每个为 [Ni,5] (cls, x1, y1, x2, y2)

        返回:
          loss_cls: 分类损失（标量）
          loss_reg: 回归损失（标量）
        """
        # 1. 处理 anchors（如果带 batch 维度，取第一个 batch）
        if anchors.dim() == 3:
            anchors = anchors[0]  # [B,A,4] -> [A,4]
        device = cls_preds.device

        # 2. 目标分配：为每个 anchor 匹配 GT
        labels, matched = self.assign_targets_to_anchors(anchors.to(device), targets)
        B, A, C = cls_preds.shape

        # 3. 创建掩码
        ignore_mask = labels.eq(-1)  # [B,A] - 忽略区域（不计算损失）
        pos_mask = (labels >= 0) & (labels < self.num_classes)  # [B,A] - 正样本
        valid_mask = ~ignore_mask  # [B,A] - 有效样本（背景 + 正样本）

        # 4. 准备分类目标（one-hot 编码）
        # 背景类别用 num_classes 表示，one-hot 后去掉该维度
        labels_clamped = labels.clamp(min=0)  # 将 -1 转为 0（临时）
        tgt_oh = F.one_hot(labels_clamped, num_classes=self.num_classes + 1)  # [B,A,C+1]
        tgt_oh = tgt_oh[..., :self.num_classes].float()  # 去掉背景通道 -> [B,A,C]

        # 5. 仅在有效位置计算分类损失
        cls_pred_valid = cls_preds[valid_mask]  # [N_valid, C]
        tgt_valid = tgt_oh[valid_mask]  # [N_valid, C]

        # 统计正样本数量（用于归一化）
        num_pos = int(pos_mask.sum().item())

        # 6. 计算分类损失（Focal Loss）
        # ✅ 关键：按正样本数归一化，避免负样本主导训练
        if cls_pred_valid.numel() == 0:
            loss_cls = torch.tensor(0.0, device=device)
        else:
            loss_cls_sum = sigmoid_focal_loss(
                cls_pred_valid, tgt_valid,
                alpha=self.focal_alpha, gamma=self.focal_gamma,
                reduction="sum"
            )
            norm = max(1, num_pos)  # 至少为 1，避免除零
            loss_cls = loss_cls_sum / norm

        # 7. 计算回归损失（仅正样本）
        if num_pos == 0:  # 没有正样本，回归损失为 0
            return loss_cls, torch.tensor(0.0, device=device)

        reg_pred_pos = reg_preds[pos_mask]  # [N_pos, 4] - 正样本的回归预测
        gt_box_pos = matched[pos_mask]  # [N_pos, 4] - 正样本匹配的 GT

        # 获取正样本对应的 anchors
        pos_indices = pos_mask.nonzero(as_tuple=False)  # [N_pos, 2] -> (batch_idx, anchor_idx)
        a_idx = pos_indices[:, 1]  # 提取 anchor 索引
        anc_pos = anchors[a_idx]  # [N_pos, 4] - 正样本对应的 anchors

        # 8. 编码 GT 相对于 anchors 的偏移量
        target_deltas = self.encode_bbox(anc_pos, gt_box_pos)  # [N_pos, 4]

        # 9. 计算 Smooth L1 损失
        # beta=1/9 是 Faster R-CNN/RetinaNet 的常用设置
        loss_reg = F.smooth_l1_loss(reg_pred_pos, target_deltas, beta=1 / 9, reduction="sum") / num_pos

        # 10. Calculate the total loss
        total_loss = loss_cls * self.cls_weight + loss_reg * self.reg_weight

        # 10. 返回总损失和详细信息（detach 避免影响梯度）
        if self.use_awl:  # Return different losses separately for AWL
            return total_loss.detach(), {
                "cls_loss": loss_cls,
                "reg_loss": loss_reg
            }
        else:  # Normally
            return total_loss, {
                "cls_loss": loss_cls.detach(),
                "reg_loss": loss_reg.detach()
            }
