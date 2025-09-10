import math
from typing import List, Dict, Union

import torch
import torch.nn as nn
from torchvision.ops import nms

from core.models.backbones.mobilenet_v2 import MobileNetV2Backbone
from core.models.backbones.shufflenet_v2 import ShuffleNetV2Backbone
from core.models.necks.fpn_lite import FPNLite
from core.models.heads.ssd_head import SSDLiteHead


BACKBONES = {
    "mobilenet_v2": MobileNetV2Backbone,
    "shufflenet_v2": ShuffleNetV2Backbone,
}


class AnchorGenerator:
    """
    以 (cx, cy, w, h) 的归一化格式生成锚框（相对图像宽高，范围 0~1）。
    per_level_scales: List[List[float]]，每层的尺度（相对短边的比例，如 0.06）
    per_level_ratios: List[List[float]]，每层的宽高比（如 [0.5,1.0,2.0]）
    """
    def __init__(self, per_level_scales: List[List[float]], per_level_ratios: List[List[float]]):
        self.per_level_scales = per_level_scales
        self.per_level_ratios = per_level_ratios

    @staticmethod
    def _gen_one(feat_h: int, feat_w: int, scales: List[float], ratios: List[float], device):
        anchors = []
        for iy in range(feat_h):
            cy = (iy + 0.5) / feat_h
            for ix in range(feat_w):
                cx = (ix + 0.5) / feat_w
                for s in scales:
                    for r in ratios:
                        w = s * math.sqrt(r)
                        h = s / math.sqrt(r)
                        anchors.append([cx, cy, w, h])
        return torch.tensor(anchors, dtype=torch.float32, device=device)

    def grid_anchors(self, feats: List[torch.Tensor], device):
        assert len(feats) == len(self.per_level_scales) == len(self.per_level_ratios)
        all_anchors = []
        for i, f in enumerate(feats):
            _, _, H, W = f.shape
            a = self._gen_one(H, W, self.per_level_scales[i], self.per_level_ratios[i], device)
            all_anchors.append(a)
        return torch.cat(all_anchors, dim=0)  # [sum(HW*A), 4]


class SSDLiteDet(nn.Module):
    """
    像 YOLO 一样“一次前向多目标”的轻量检测器：
      backbone -> FPNLite -> 若干 SSDLiteHead -> 合并 -> (train/infer)
    - 训练时返回: {"cls_logits","bbox_regs","anchors"}
    - 推理时返回: List[{"boxes","scores","labels"}]
    """
    def __init__(self,
                 backbone: str = "mobilenet_v2",
                 num_classes: int = 81,          # 背景+80（COCO），自定义数据集请改成 1+N
                 width_mult: float = 1.0,
                 neck_outc: int = 64,
                 head_midc: int = 64,
                 per_level_scales: List[List[float]] = None,
                 per_level_ratios: List[List[float]] = None,
                 score_thresh: float = 0.35,
                 nms_thresh: float = 0.5,
                 topk: int = 200,
                 class_agnostic_nms: bool = False):
        super().__init__()
        assert backbone in BACKBONES, f"unknown backbone: {backbone}"
        self.num_classes = num_classes
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.topk = topk
        self.class_agnostic_nms = class_agnostic_nms

        # 1) Backbone
        self.backbone = BACKBONES[backbone](width_mult=width_mult)
        c3c, c4c, c5c = self.backbone.get_out_channels()

        # 2) Neck（与 MoveNet 同风格）
        self.neck = FPNLite(c3=c3c, c4=c4c, c5=c5c, outc=neck_outc)

        # 3) Head（多层时为多头），延迟构建（取决于 neck 输出层数）
        self._heads_built = False
        self._head_midc = head_midc
        self._per_level_scales = per_level_scales
        self._per_level_ratios = per_level_ratios

        # variance for SSD-style decode
        self.register_buffer("_var", torch.tensor([0.1, 0.1, 0.2, 0.2], dtype=torch.float32))

    # -------- utils --------
    @staticmethod
    def _ensure_list_feats(feats: Union[torch.Tensor, List[torch.Tensor]]):
        return feats if isinstance(feats, (list, tuple)) else [feats]

    def _build_heads_once(self, feats: List[torch.Tensor]):
        # 默认锚配置：单层 or 三层
        if self._per_level_scales is None or self._per_level_ratios is None:
            if len(feats) >= 3:
                self._per_level_scales = [[0.05, 0.08, 0.12], [0.16, 0.24], [0.32, 0.48]]
                self._per_level_ratios = [[0.5, 1.0, 2.0]] * 3
            else:
                self._per_level_scales = [[0.04, 0.06, 0.08, 0.12, 0.16]]
                self._per_level_ratios = [[0.5, 1.0, 2.0]]

        self.heads = nn.ModuleList()
        for i, f in enumerate(feats):
            in_ch = f.shape[1]
            A = len(self._per_level_scales[i]) * len(self._per_level_ratios[i])
            self.heads.append(SSDLiteHead(in_ch, A, self.num_classes, midc=self._head_midc))

        # 把创建好的 head 迁移到与特征相同的 device 避免 CPU/CUDA 混用
        dev = feats[0].device
        for m in self.heads:
            m.to(dev)

        self.anchor_gen = AnchorGenerator(self._per_level_scales, self._per_level_ratios)
        self._heads_built = True

    def _decode(self, locs: torch.Tensor, anchors: torch.Tensor):
        """
        locs:   [N, 4] -> (dx, dy, dw, dh)
        anchor: [N, 4] -> (cx, cy, w, h)  (归一化)
        return: [N, 4] -> (x1,y1,x2,y2)  (归一化)
        """
        dx, dy, dw, dh = locs.unbind(dim=1)
        acx, acy, aw, ah = anchors.unbind(dim=1)
        var = self._var

        px = dx * var[0] * aw + acx
        py = dy * var[1] * ah + acy
        pw = torch.exp(dw * var[2]) * aw
        ph = torch.exp(dh * var[3]) * ah

        x1 = px - pw * 0.5
        y1 = py - ph * 0.5
        x2 = px + pw * 0.5
        y2 = py + ph * 0.5
        return torch.stack([x1, y1, x2, y2], dim=1)

    # -------- forward --------
    def forward(self, x) -> Union[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        B, _, H, W = x.shape

        # Backbone & Neck（你的 FPNLite 可能返回单层 P3 或多层 [P3,P4,P5]）
        c3, c4, c5 = self.backbone(x)
        feats = self.neck(c3, c4, c5)
        feats_list = self._ensure_list_feats(feats)

        if not self._heads_built:
            self._build_heads_once(feats_list)

        # 逐层预测并拼接
        cls_list, reg_list = [], []
        for f, head in zip(feats_list, self.heads):
            cls, reg = head(f)             # [B, N_l, C], [B, N_l, 4]
            cls_list.append(cls)
            reg_list.append(reg)

        cls_all = torch.cat(cls_list, dim=1)   # [B, N, C]
        reg_all = torch.cat(reg_list, dim=1)   # [B, N, 4]
        anchors = self.anchor_gen.grid_anchors(feats_list, device=cls_all.device)  # [N, 4]

        if self.training:
            return {"cls_logits": cls_all, "bbox_regs": reg_all, "anchors": anchors}

        # ---------- Inference ----------
        results = []
        probs = torch.softmax(cls_all, dim=-1)  # [B, N, C]
        for b in range(B):
            boxes = self._decode(reg_all[b], anchors)  # 归一化
            # 反归一化到像素坐标
            boxes[:, [0, 2]] *= W
            boxes[:, [1, 3]] *= H

            if self.class_agnostic_nms:
                # 取每个候选的最大类别（忽略背景0）
                scores, labels = probs[b, :, 1:].max(dim=1)
                keep = scores > self.score_thresh
                if keep.any():
                    kb = boxes[keep]
                    ks = scores[keep]
                    kl = labels[keep] + 1
                    keep_idx = nms(kb, ks, self.nms_thresh)[: self.topk]
                    results.append({"boxes": kb[keep_idx], "scores": ks[keep_idx], "labels": kl[keep_idx]})
                else:
                    results.append({"boxes": torch.zeros((0, 4), device=x.device),
                                    "scores": torch.zeros((0,), device=x.device),
                                    "labels": torch.zeros((0,), dtype=torch.long, device=x.device)})
            else:
                sel_boxes, sel_scores, sel_labels = [], [], []
                for cls_id in range(1, self.num_classes):  # 跳过背景类0
                    s = probs[b, :, cls_id]
                    keep = s > self.score_thresh
                    if keep.sum() == 0:
                        continue
                    kb, ks = boxes[keep], s[keep]
                    keep_idx = nms(kb, ks, self.nms_thresh)[: self.topk]
                    sel_boxes.append(kb[keep_idx])
                    sel_scores.append(ks[keep_idx])
                    sel_labels.append(torch.full_like(ks[keep_idx], cls_id, dtype=torch.long))
                if sel_boxes:
                    results.append({
                        "boxes": torch.cat(sel_boxes, dim=0),
                        "scores": torch.cat(sel_scores, dim=0),
                        "labels": torch.cat(sel_labels, dim=0)
                    })
                else:
                    results.append({"boxes": torch.zeros((0, 4), device=x.device),
                                    "scores": torch.zeros((0,), device=x.device),
                                    "labels": torch.zeros((0,), dtype=torch.long, device=x.device)})
        return results


if __name__ == "__main__":
    # 最小可跑示例（单层 P3）
    m = SSDLiteDet(backbone="mobilenet_v2", num_classes=81, width_mult=1.0).eval()
    x = torch.randn(1, 3, 192, 192)
    with torch.no_grad():
        y = m(x)
    print(type(y), y[0]["boxes"].shape, y[0]["scores"].shape, y[0]["labels"].shape)
