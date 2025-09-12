# -*- coding: utf-8 -*-
"""
SSDLite 损失（含强制匹配 + Hard Negative Mining）
- 分类：CrossEntropy + OHEM(neg:pos=3:1，默认)
- 回归：SmoothL1（对正样本）
- 编码：相对 anchor(cxcywh) 的平移/缩放偏移，使用 variances=(0.1,0.2)
期望输入：
  cls_logits: [B, A_total, C]  (C 含背景类，背景=0)
  bbox_regs : [B, A_total, 4]  (预测的编码 offsets)
  anchors   : [A_total, 4]     (cxcywh，归一化到[0,1])
  targets   : {
      "boxes":  List[Gi, 4] (xyxy，归一化到[0,1])
      "labels": List[Gi]    (1..C-1；0 预留背景，不应出现在此处)
  }
"""
from typing import Dict, List, Tuple
import math
import torch
import torch.nn as nn


# ----------------------- 基础几何工具 -----------------------
def cxcywh_to_xyxy(cxcywh: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = cxcywh.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)

def xyxy_to_cxcywh(xyxy: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = xyxy.unbind(-1)
    w = (x2 - x1).clamp(min=1e-6)
    h = (y2 - y1).clamp(min=1e-6)
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    return torch.stack([cx, cy, w, h], dim=-1)

def iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    a:[N,4], b:[M,4] (xyxy, 归一化到[0,1])
    返回: [N,M]
    """
    tl = torch.maximum(a[:, None, :2], b[None, :, :2])
    br = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (br - tl).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    union = area_a[:, None] + area_b[None, :] - inter + 1e-9
    return inter / union


# ----------------------- 编码/解码 -----------------------
def encode_gt_to_deltas(gt_xyxy: torch.Tensor,
                        anc_cxcywh: torch.Tensor,
                        variances: Tuple[float, float] = (0.1, 0.2)) -> torch.Tensor:
    """
    将 GT(xyxy) 编码为相对 anchors(cxcywh) 的偏移 (tx,ty,tw,th)
    """
    gt_cxcywh = xyxy_to_cxcywh(gt_xyxy)
    gcx, gcy, gw, gh = gt_cxcywh.unbind(-1)
    acx, acy, aw, ah = anc_cxcywh.unbind(-1)

    vx, vs = variances
    tx = (gcx - acx) / aw / vx
    ty = (gcy - acy) / ah / vx
    tw = torch.log((gw / aw).clamp(min=1e-6)) / vs
    th = torch.log((gh / ah).clamp(min=1e-6)) / vs
    return torch.stack([tx, ty, tw, th], dim=-1)

def decode_deltas_to_xyxy(deltas: torch.Tensor,
                          anc_cxcywh: torch.Tensor,
                          variances: Tuple[float, float] = (0.1, 0.2)) -> torch.Tensor:
    """
    将预测偏移解码回 xyxy（推理用；训练时不必须）
    """
    vx, vs = variances
    tx, ty, tw, th = deltas.unbind(-1)
    acx, acy, aw, ah = anc_cxcywh.unbind(-1)

    cx = tx * vx * aw + acx
    cy = ty * vx * ah + acy
    w  = torch.exp(tw * vs) * aw
    h  = torch.exp(th * vs) * ah
    return cxcywh_to_xyxy(torch.stack([cx, cy, w, h], dim=-1))


# ----------------------- 匹配（强制匹配 + 三分法） -----------------------
@torch.no_grad()
def assign_anchors(anchors_cxcywh: torch.Tensor,
                   gts_xyxy: torch.Tensor,
                   gts_labels: torch.Tensor,
                   pos_iou: float = 0.5,
                   neg_iou: float = 0.4) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    返回:
      cls_t: [A]  (0=bg, -1=ignore, 1..C-1=前景)
      reg_t: [A,4] (编码后的回归目标)
      pos:   [A]  bool 正样本掩码
    """
    A = anchors_cxcywh.shape[0]
    device = anchors_cxcywh.device
    cls_t = torch.zeros((A,), dtype=torch.long, device=device)
    reg_t = torch.zeros((A, 4), dtype=torch.float32, device=device)
    pos   = torch.zeros((A,), dtype=torch.bool, device=device)

    if gts_xyxy.numel() == 0:
        # 全部负样本
        return cls_t, reg_t, pos

    anc_xyxy = cxcywh_to_xyxy(anchors_cxcywh)  # [A,4]
    iou = iou_xyxy(anc_xyxy, gts_xyxy)         # [A,G]
    iou_max, iou_idx = iou.max(dim=1)          # 每个 anchor 的最佳 GT

    # 三分：正 / 忽略 / 负
    pos = iou_max >= pos_iou
    ign = (iou_max > neg_iou) & (~pos)
    cls_t[:]   = 0
    cls_t[ign] = -1

    # 强制匹配：每个 GT 至少有一个正样本
    _, gt_best_anchor = iou.max(dim=0)         # [G]
    pos[gt_best_anchor] = True
    iou_idx[gt_best_anchor] = torch.arange(gts_xyxy.shape[0], device=device)

    # 写入正样本的回归与类别
    if pos.any():
        matched_gt = gts_xyxy[iou_idx[pos]]     # [P,4]
        matched_lb = gts_labels[iou_idx[pos]]   # [P]  (1..C-1)
        reg_t[pos] = encode_gt_to_deltas(matched_gt, anchors_cxcywh[pos])
        cls_t[pos] = matched_lb

    return cls_t, reg_t, pos


# ----------------------- 损失主体 -----------------------
class SSDLoss(nn.Module):
    def __init__(self,
                 num_classes: int,
                 alpha: float = 1.0,
                 neg_pos_ratio: int = 3,
                 reg_type: str = "smoothl1"):
        """
        num_classes: 包含背景类的总类别数（背景=0，前景=1..C-1）
        alpha: 回归损失权重
        neg_pos_ratio: OHEM 负正比
        reg_type: 'smoothl1' 或 'l1'（默认 smoothl1）
        """
        super().__init__()
        self.num_classes   = int(num_classes)
        self.alpha         = float(alpha)
        self.neg_pos_ratio = int(neg_pos_ratio)
        self.ce  = nn.CrossEntropyLoss(reduction='none')
        if reg_type == "l1":
            self.reg_loss = nn.L1Loss(reduction='none')
        else:
            self.reg_loss = nn.SmoothL1Loss(reduction='none')

    def forward(self,
                cls_logits: torch.Tensor,    # [B, A, C]
                bbox_regs:  torch.Tensor,    # [B, A, 4]
                anchors_cxcywh: torch.Tensor,# [A, 4]  归一化
                targets: Dict[str, List[torch.Tensor]]  # {"boxes": [Gi,4], "labels":[Gi]}
                ) -> Tuple[torch.Tensor, Dict[str, float]]:
        B, A, C = cls_logits.shape
        assert C == self.num_classes, f"num_classes mismatch: {C} vs {self.num_classes}"
        assert bbox_regs.shape[:2] == (B, A)
        assert anchors_cxcywh.shape[0] == A

        total_cls = 0.0
        total_reg = 0.0
        total_img = 0

        for b in range(B):
            gt_boxes  = targets["boxes"][b]   # [Gi,4] (归一化 xyxy)
            gt_labels = targets["labels"][b]  # [Gi]   (1..C-1)
            cls_t, reg_t, pos_mask = assign_anchors(anchors_cxcywh, gt_boxes, gt_labels)

            logits_b = cls_logits[b]          # [A,C]
            # 有效样本（非 ignore）
            valid = cls_t >= 0
            if valid.any():
                # 分类 OHEM
                # 正样本索引（在 A 维度）
                pos_idx = torch.nonzero((cls_t > 0) & valid, as_tuple=False).squeeze(1)
                neg_idx = torch.nonzero((cls_t == 0) & valid, as_tuple=False).squeeze(1)

                num_pos = int(pos_idx.numel())
                if num_pos > 0:
                    # 负样本难例挖掘（与正样本按比率筛 K 个负样本）
                    K = min(self.neg_pos_ratio * num_pos, int(neg_idx.numel()))
                    if K > 0:
                        # 对负样本与“背景类 0”计算 CE，选最难的 K 个
                        neg_ce = self.ce(logits_b[neg_idx], torch.zeros_like(neg_idx))
                        topk = neg_ce.topk(K).indices
                        sel_neg_idx = neg_idx[topk]
                        # 正样本的 CE
                        pos_ce = self.ce(logits_b[pos_idx], cls_t[pos_idx])
                        cls_loss = torch.cat([pos_ce, neg_ce[topk]], dim=0).mean()
                    else:
                        cls_loss = self.ce(logits_b[pos_idx], cls_t[pos_idx]).mean()
                else:
                    # 没有正样本：取固定数量最难负样本稳定训练
                    if neg_idx.numel() > 0:
                        K = min(64, int(neg_idx.numel()))
                        neg_ce = self.ce(logits_b[neg_idx], torch.zeros_like(neg_idx))
                        topk = neg_ce.topk(K).indices
                        cls_loss = neg_ce[topk].mean()
                    else:
                        cls_loss = logits_b.new_zeros(())
                total_cls += cls_loss
            else:
                total_cls += logits_b.new_zeros(())

            # 回归（仅正样本）
            if pos_mask.any():
                reg_pred = bbox_regs[b][pos_mask]  # [P,4]
                reg_gt   = reg_t[pos_mask]         # [P,4]
                reg_loss = self.reg_loss(reg_pred, reg_gt).mean()
                total_reg += reg_loss
            else:
                total_reg += bbox_regs.new_zeros(())

            total_img += 1

        # 对 batch 求平均（更稳定）
        if total_img > 0:
            total_cls = total_cls / total_img
            total_reg = total_reg / total_img

        loss = total_cls + self.alpha * total_reg
        meter = {
            "loss_cls": float(total_cls.detach().cpu()),
            "loss_reg": float(total_reg.detach().cpu()),
        }
        return loss, meter


# ----------------------- 训练端便捷工具（可选） -----------------------
def pack_targets_for_ssd(targets_list: List[torch.Tensor], img_size: int) -> Dict[str, List[torch.Tensor]]:
    """
    将 CocoDetDataset 的 batch 输出打包成 SSDLoss 需要的字典格式。
    输入 targets_list: List[Tensor[Ni,5]] -> [cls, x1, y1, x2, y2] (像素坐标)
    输出:
      {"boxes": [Gi,4] (xyxy, 归一化), "labels": [Gi] (1..C-1)}
    说明：
      - 把类别 0..C-1 映射到 1..C-1（保留 0 给背景）
      - 坐标 / img_size 做归一化
    """
    boxes_norm, labels = [], []
    for t in targets_list:
        if t.numel() == 0:
            boxes_norm.append(t.new_zeros((0, 4)))
            labels.append(torch.zeros((0,), dtype=torch.long, device=t.device))
            continue
        cls = t[:, 0].long() + 1               # 0..C-1 -> 1..C-1
        b   = t[:, 1:5] / float(img_size)      # 归一化
        b   = b.clamp(0, 1)
        boxes_norm.append(b)
        labels.append(cls)
    return {"boxes": boxes_norm, "labels": labels}


@torch.no_grad()
def generate_ssd_anchors(img_size: int,
                         feat_shapes: List[Tuple[int, int]],
                         strides: List[int],
                         ratios: Tuple[float, ...] = (1.0, 2.0, 0.5),
                         scales: Tuple[float, ...] = (1.0, 1.26)) -> torch.Tensor:
    """
    生成标准 SSD anchors（cxcywh，归一化）。
    - feat_shapes: 如 [(H3,W3),(H4,W4),(H5,W5)]
    - strides:     与上对应（例如 [8,16,32]）
    - ratios/scales: 每层 A = len(ratios)*len(scales)，需与 Head 对齐
    """
    device = torch.device('cpu')
    all_anchors = []
    for (H, W), s in zip(feat_shapes, strides):
        ys = (torch.arange(H, device=device) + 0.5) * s
        xs = (torch.arange(W, device=device) + 0.5) * s
        cy, cx = torch.meshgrid(ys, xs, indexing='ij')  # [H,W]
        cx = (cx / img_size).reshape(-1, 1)
        cy = (cy / img_size).reshape(-1, 1)
        anchors_lvl = []
        for r in ratios:
            for sc in scales:
                w = (sc * s * math.sqrt(r)) / img_size
                h = (sc * s / math.sqrt(r)) / img_size
                wh = torch.full((H * W, 2), 0.0, device=device)
                wh[:, 0] = w; wh[:, 1] = h
                anchors_lvl.append(torch.cat([cx, cy, wh], dim=1))
        all_anchors.append(torch.cat(anchors_lvl, dim=0))  # [H*W*A,4]
    return torch.cat(all_anchors, dim=0)  # [A_total,4]
