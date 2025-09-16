# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class DummySSDLite(nn.Module):
    """
    Wrap SSDLiteDet (train-mode dict outputs) -> (boxes, class_scores)
    - boxes:        [B, N, 4]，(x1,y1,x2,y2)。默认是像素坐标；可选归一化到0~1
    - class_scores: [B, N, C-1]，背景列已移除（类别1..C-1），为 softmax 概率
    说明：
      * 内部会临时切换到训练分支以拿到原始 (cls_logits, bbox_regs, anchors)。
      * 不做 NMS；下游可用 TRT 插件/后处理完成筛选。
    """
    def __init__(self, ssdlite_det: nn.Module,
                 num_classes: int,
                 variance=(0.1, 0.1, 0.2, 0.2),
                 return_normalized: bool = False,  # True: 返回0~1坐标；False: 返回像素坐标
                 drop_background: bool = True):
        super().__init__()
        self.m = ssdlite_det
        self.num_classes = num_classes
        self.var = torch.tensor(variance, dtype=torch.float32)
        self.return_normalized = return_normalized
        self.drop_background = drop_background

    @staticmethod
    def _cxcywh_to_xyxy(cxcywh: torch.Tensor) -> torch.Tensor:
        cx, cy, w, h = cxcywh.unbind(dim=-1)
        x1 = cx - w * 0.5
        y1 = cy - h * 0.5
        x2 = cx + w * 0.5
        y2 = cy + h * 0.5
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def _decode(self, loc: torch.Tensor, anc: torch.Tensor) -> torch.Tensor:
        """
        loc: [B,N,4] -> (dx,dy,dw,dh)
        anc: [N,4]   -> (cx,cy,w,h)  (normalized)
        out: [B,N,4] -> (x1,y1,x2,y2) (normalized)
        """
        # 扩展 anchors 到 batch
        B = loc.shape[0]
        anc = anc.unsqueeze(0).expand(B, -1, -1)

        dx, dy, dw, dh = loc.unbind(dim=-1)
        acx, acy, aw, ah = anc.unbind(dim=-1)

        vx, vy, vw, vh = self.var.to(loc.device)
        px = dx * vx * aw + acx
        py = dy * vy * ah + acy
        pw = torch.exp(dw * vw) * aw
        ph = torch.exp(dh * vh) * ah

        return self._cxcywh_to_xyxy(torch.stack([px, py, pw, ph], dim=-1))

    def forward(self, x: torch.Tensor):
        B, _, H, W = x.shape

        # 暂时切到“训练”分支，拿到 dict 输出（cls_logits,bbox_regs,anchors）
        was_training = self.m.training
        self.m.train(True)
        out = self.m(x)  # {"cls_logits":[B,N,C], "bbox_regs":[B,N,4], "anchors":[N,4]}
        if not was_training:
            self.m.train(False)

        cls_logits: torch.Tensor = out["cls_logits"]
        bbox_regs:  torch.Tensor = out["bbox_regs"]
        anchors:    torch.Tensor = out["anchors"]  # [N,4] (cx,cy,w,h), normalized

        # 解码成 xyxy（normalized）
        boxes_norm = self._decode(bbox_regs, anchors).clamp(0.0, 1.0)

        # 取概率并去掉背景列
        probs = F.softmax(cls_logits, dim=-1)  # [B,N,C]
        if self.drop_background:
            class_scores = probs[..., 1:]      # [B,N,C-1]
        else:
            class_scores = probs               # [B,N,C]

        if self.return_normalized:
            boxes = boxes_norm
        else:
            # 像素坐标（与输入分辨率一致）
            scale = torch.tensor([W, H, W, H], dtype=boxes_norm.dtype, device=boxes_norm.device)
            boxes = boxes_norm * scale

        return boxes, class_scores
