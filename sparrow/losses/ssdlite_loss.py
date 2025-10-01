# -*- coding: utf-8 -*-
import math
from typing import List, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou, sigmoid_focal_loss


class AnchorGenerator:
    """只保留坐标转换工具（训练时 anchors 由模型给出）"""

    @staticmethod
    def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        return torch.cat([boxes[:, :2] - boxes[:, 2:] / 2,
                          boxes[:, :2] + boxes[:, 2:] / 2], dim=1)

    @staticmethod
    def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        cx = boxes[:, 0] + w / 2
        cy = boxes[:, 1] + h / 2
        return torch.stack([cx, cy, w, h], dim=1)


class SSDLoss(nn.Module):
    """
    SSD/RetinaNet 风格损失：Focal（分类） + Smooth L1（回归）
    关键修改：分类项使用“正样本数”归一化，避免被负样本淹没。
    """
    def __init__(self,
                 num_classes: int,
                 iou_threshold_pos: float = 0.5,
                 iou_threshold_neg: float = 0.4,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0,
                 debug_assign: bool = False):
        super().__init__()
        self.num_classes = int(num_classes)
        self.iou_threshold_pos = float(iou_threshold_pos)
        self.iou_threshold_neg = float(iou_threshold_neg)
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)
        self.debug_assign = bool(debug_assign)

        self.anchor_utils = AnchorGenerator()
        # 作为 buffer，便于 .to(device)
        self.register_buffer("bbox_std", torch.tensor([0.1, 0.1, 0.2, 0.2], dtype=torch.float32))

    # -------- target assign --------
    def assign_targets_to_anchors(self, anchors: torch.Tensor, targets: List[torch.Tensor]):
        """
        anchors: [A,4]  xyxy
        targets: list of [Ni,5]  (cls, x1,y1,x2,y2)
        return:
          labels: [B,A] (0..C-1 为前景，C 为背景，-1 为 ignore)
          matched_gt_boxes: [B,A,4] (xyxy)
        """
        device = anchors.device
        B = len(targets)
        A = anchors.shape[0]

        labels = torch.full((B, A), self.num_classes, dtype=torch.int64, device=device)   # 默认背景
        matched = torch.zeros((B, A, 4), dtype=torch.float32, device=device)

        for b in range(B):
            tgt = targets[b]
            if tgt.numel() == 0:
                continue
            gt_cls = tgt[:, 0].to(torch.int64)
            gt_box = tgt[:, 1:5]
            iou = box_iou(gt_box, anchors)              # [Ng, A]

            # 每个 anchor 的最好 GT
            iou_a, idx_a = iou.max(dim=0)               # [A], [A]
            # 每个 GT 的最好 anchor（确保至少一个正样本）
            _, idx_g = iou.max(dim=1)                   # [Ng]

            # ignore：中间灰区
            ignore = (iou_a >= self.iou_threshold_neg) & (iou_a < self.iou_threshold_pos)
            labels[b, ignore] = -1

            # 正样本：IoU>=pos
            pos = iou_a >= self.iou_threshold_pos
            if pos.any():
                labels[b, pos] = gt_cls[idx_a[pos]]
                matched[b, pos] = gt_box[idx_a[pos]]

            # 强制匹配：每个 GT 的最佳 anchor
            labels[b, idx_g] = gt_cls
            matched[b, idx_g] = gt_box

            if self.debug_assign:
                # 打印本 batch 的前5类分布（可关）
                uniq, cnt = torch.unique(labels[b][labels[b] >= 0], return_counts=True)
                pairs = sorted([(int(c.item()), int(n.item())) for c, n in zip(uniq, cnt)],
                               key=lambda x: x[1], reverse=True)[:5]
                print("[assign] pos per-class top5:", pairs)

        return labels, matched

    # -------- encode --------
    def encode_bbox(self, anchors_xyxy: torch.Tensor, gt_xyxy: torch.Tensor) -> torch.Tensor:
        a = self.anchor_utils.xyxy_to_cxcywh(anchors_xyxy)
        g = self.anchor_utils.xyxy_to_cxcywh(gt_xyxy)
        tx = (g[:, 0] - a[:, 0]) / a[:, 2]
        ty = (g[:, 1] - a[:, 1]) / a[:, 3]
        tw = torch.log((g[:, 2] / a[:, 2]).clamp(min=1e-6))
        th = torch.log((g[:, 3] / a[:, 3]).clamp(min=1e-6))
        deltas = torch.stack([tx, ty, tw, th], dim=1)
        std = self.bbox_std.to(device=deltas.device, dtype=deltas.dtype)
        return (deltas / std).clamp(min=-4.0, max=4.0)

    # -------- forward --------
    def forward(self,
                anchors: torch.Tensor,          # [A,4] 或 [B,A,4]
                cls_preds: torch.Tensor,        # [B,A,C] (logits)
                reg_preds: torch.Tensor,        # [B,A,4]
                targets: List[torch.Tensor]):   # len=B, each [Ni,5]

        # 允许 anchors 带 batch 维（模型里是 [B,A,4]），这里去重用第 0 张
        if anchors.dim() == 3:
            anchors = anchors[0]
        device = cls_preds.device

        labels, matched = self.assign_targets_to_anchors(anchors.to(device), targets)
        B, A, C = cls_preds.shape

        # --- masks ---
        ignore_mask = labels.eq(-1)                          # [B,A]
        pos_mask = (labels >= 0) & (labels < self.num_classes)
        valid_mask = ~ignore_mask                            # 背景+正样本

        # --- classification targets (one-hot over C) ---
        # 背景= num_classes，会在 one_hot 后被丢弃成 0 向量
        labels_clamped = labels.clamp(min=0)                 # [-1,C] -> [0..C]
        tgt_oh = F.one_hot(labels_clamped, num_classes=self.num_classes + 1)  # [B,A,C+1]
        tgt_oh = tgt_oh[..., :self.num_classes].float()      # 去掉背景通道 -> [B,A,C]

        # 仅在 valid 位置参与 focal
        cls_pred_valid = cls_preds[valid_mask]               # [N_valid, C]
        tgt_valid = tgt_oh[valid_mask]                       # [N_valid, C]

        # 前景数量（用于归一化）
        num_pos = int(pos_mask.sum().item())

        # ✅ 关键：分类损失用“前景数”归一化
        if cls_pred_valid.numel() == 0:
            loss_cls = torch.tensor(0.0, device=device)
        else:
            loss_cls_sum = sigmoid_focal_loss(
                cls_pred_valid, tgt_valid,
                alpha=self.focal_alpha, gamma=self.focal_gamma,
                reduction="sum"
            )
            norm = max(1, num_pos)
            loss_cls = loss_cls_sum / norm

        # --- regression (only positives) ---
        if num_pos == 0:
            return loss_cls, torch.tensor(0.0, device=device)

        reg_pred_pos = reg_preds[pos_mask]                   # [N_pos,4]
        gt_box_pos   = matched[pos_mask]                     # [N_pos,4]
        # 展开 anchors 对应到正样本索引
        pos_indices = pos_mask.nonzero(as_tuple=False)       # [N_pos,2] -> (b_idx, a_idx)
        a_idx = pos_indices[:, 1]
        anc_pos = anchors[a_idx]

        target_deltas = self.encode_bbox(anc_pos, gt_box_pos)  # [N_pos,4]
        # Smooth L1（Retina/Faster 通常 beta=1/9）
        loss_reg = F.smooth_l1_loss(reg_pred_pos, target_deltas, beta=1/9, reduction="sum") / num_pos

        return loss_cls, loss_reg
