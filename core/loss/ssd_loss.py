# -*- coding: utf-8 -*-
from typing import Dict
import torch, torch.nn as nn

def iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a:[N,4], b:[M,4] in normalized [0,1]
    tl = torch.maximum(a[:, None, :2], b[None, :, :2])
    br = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (br - tl).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    union = area_a[:, None] + area_b[None, :] - inter + 1e-9
    return inter / union

def xyxy_to_cxcywh(xyxy: torch.Tensor) -> torch.Tensor:
    x1,y1,x2,y2 = xyxy.unbind(dim=-1)
    cx = (x1+x2)/2; cy = (y1+y2)/2; w = (x2-x1).clamp(min=1e-6); h = (y2-y1).clamp(min=1e-6)
    return torch.stack([cx,cy,w,h], dim=-1)

def encode_ltrb_to_offsets(gt_xyxy: torch.Tensor, anc_cxcywh: torch.Tensor, var=(0.1,0.1,0.2,0.2)) -> torch.Tensor:
    # returns [N,4]: (dx,dy,dw,dh)
    gx,gy,gw,gh = xyxy_to_cxcywh(gt_xyxy).unbind(dim=-1)
    ax,ay,aw,ah = anc_cxcywh.unbind(dim=-1)
    dx = (gx-ax) / (aw*var[0]); dy = (gy-ay) / (ah*var[1])
    dw = torch.log(gw/aw) / var[2]; dh = torch.log(gh/ah) / var[3]
    return torch.stack([dx,dy,dw,dh], dim=-1)

@torch.no_grad()
def assign_anchors(anchors_cxcywh: torch.Tensor,
                   gts_xyxy: torch.Tensor, gts_labels: torch.Tensor,
                   pos_iou=0.5, neg_iou=0.4):
    """
    anchors_cxcywh: [A,4] normalized
    gts_xyxy:       [G,4] normalized
    gts_labels:     [G]
    return:
      cls_targets: LongTensor[A] in [0..C-1], 0=background, -1=ignore
      reg_targets: FloatTensor[A,4]
      pos_mask:    BoolTensor[A]
    """
    A = anchors_cxcywh.shape[0]
    cls_t = torch.zeros((A,), dtype=torch.long, device=anchors_cxcywh.device)       # background by default
    reg_t = torch.zeros((A,4), dtype=torch.float32, device=anchors_cxcywh.device)
    pos = torch.zeros((A,), dtype=torch.bool, device=anchors_cxcywh.device)

    if gts_xyxy.numel() == 0:
        # all negatives
        cls_t[:] = 0
        return cls_t, reg_t, pos

    # convert anchors to xyxy
    ax,ay,aw,ah = anchors_cxcywh.unbind(dim=-1)
    anc_xyxy = torch.stack([ax-aw/2, ay-ah/2, ax+aw/2, ay+ah/2], dim=-1)

    iou = iou_xyxy(anc_xyxy, gts_xyxy)  # [A,G]
    iou_max, iou_idx = iou.max(dim=1)   # best gt per anchor

    # 1) positive / ignore / negative by thresholds
    pos = iou_max >= pos_iou
    ign = (iou_max > neg_iou) & (~pos)

    cls_t[:] = 0
    cls_t[ign] = -1
    if pos.any():
        matched_gt = gts_xyxy[iou_idx[pos]]
        matched_lb = gts_labels[iou_idx[pos]]
        reg_t[pos] = encode_ltrb_to_offsets(matched_gt, anchors_cxcywh[pos])
        cls_t[pos] = matched_lb  # 注意：gt label 已经是 1..C-1（0保留给背景）
    return cls_t, reg_t, pos

class SSDLoss(nn.Module):
    def __init__(self, num_classes: int, alpha: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.cls_loss = nn.CrossEntropyLoss(reduction='none')  # targets: [A] 0..C-1, -1=ignore
        self.reg_loss = nn.SmoothL1Loss(reduction='none')

    def forward(self, cls_logits: torch.Tensor, bbox_regs: torch.Tensor,
                anchors_cxcywh: torch.Tensor,
                targets: Dict[str, torch.Tensor]):
        """
        cls_logits: [B,A,C]
        bbox_regs : [B,A,4]
        anchors   : [A,4]
        targets: dict with keys:
          "boxes": List[Tensor[n_i,4]] (normalized xyxy per-image)
          "labels": List[Tensor[n_i]]  (in 1..C-1)
        """
        B, A, C = cls_logits.shape
        assert C == self.num_classes

        total_cls = 0.0
        total_reg = 0.0
        total_pos = 0

        for b in range(B):
            gt_boxes = targets["boxes"][b]
            gt_labels= targets["labels"][b]
            cls_t, reg_t, pos_mask = assign_anchors(anchors_cxcywh, gt_boxes, gt_labels)

            # cls
            cls_logit = cls_logits[b]  # [A,C]
            valid = cls_t >= 0
            if valid.any():
                total_cls += self.cls_loss(cls_logit[valid], cls_t[valid]).mean()
            # reg
            if pos_mask.any():
                reg_pred = bbox_regs[b][pos_mask]
                reg_gt   = reg_t[pos_mask]
                total_reg += self.reg_loss(reg_pred, reg_gt).mean()
                total_pos += pos_mask.sum().item()

        # 最终损失仍是 Tensor，梯度稳定
        loss = total_cls + self.alpha * total_reg

        # 仅用于日志展示：先脱离计算图，再转成 Python number
        meter = {
            "loss_cls": total_cls.detach().item(),
            "loss_reg": total_reg.detach().item(),
            "pos": int(total_pos),
        }
        return loss, meter
