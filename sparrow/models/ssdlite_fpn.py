# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

# -------------------------
# FPN
# -------------------------
class FPN(nn.Module):
    """
    Feature Pyramid Network (FPN) Neck
    对 Backbone 输出的 (C3, C4, C5) 特征进行融合，生成 (P3, P4, P5)。
    """
    def __init__(self, in_channels: Tuple[int, int, int], out_channels: int = 256):
        super().__init__()
        c3_in, c4_in, c5_in = in_channels

        self.lateral_conv3 = nn.Conv2d(c3_in, out_channels, kernel_size=1)
        self.lateral_conv4 = nn.Conv2d(c4_in, out_channels, kernel_size=1)
        self.lateral_conv5 = nn.Conv2d(c5_in, out_channels, kernel_size=1)

        self.output_conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.output_conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.output_conv5 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c3, c4, c5 = features

        p5_in = self.lateral_conv5(c5)

        p4_in = self.lateral_conv4(c4)
        p5_up = F.interpolate(p5_in, size=p4_in.shape[-2:], mode='nearest')
        p4_in = p4_in + p5_up

        p3_in = self.lateral_conv3(c3)
        p4_up = F.interpolate(p4_in, size=p3_in.shape[-2:], mode='nearest')
        p3_in = p3_in + p4_up

        p3 = self.output_conv3(p3_in)
        p4 = self.output_conv4(p4_in)
        p5 = self.output_conv5(p5_in)
        return p3, p4, p5


# -------------------------
# SSDLite 轻量预测头
# -------------------------
def SSDLitePredictionHead(in_channels, num_classes, num_anchors):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=1),
    )


# -------------------------
# Anchor Generator (SSD 风格，xyxy，像素)
# -------------------------
class SSDAnchorGenerator(nn.Module):
    """
    依据输入尺寸与多层特征图尺寸，生成每层的密集 anchors（xyxy，像素坐标）。
    默认每层 scales=[0.04,0.08,0.16,0.32,0.64]，aspect_ratios=[1,2,0.5]，再加一个中间尺度的 ar=1。
    => anchors_per_loc=4
    """
    def __init__(self,
                 img_size: int,
                 feature_strides: List[int],
                 scales: List[float] = None,
                 aspect_ratios: List[float] = None,
                 add_intermediate_scale: bool = True):
        super().__init__()
        self.img_size = int(img_size)
        self.feature_strides = feature_strides
        self.scales = scales if scales is not None else [0.04, 0.08, 0.16, 0.32, 0.64]
        self.aspect_ratios = aspect_ratios if aspect_ratios is not None else [1.0, 2.0, 0.5]
        self.add_intermediate = bool(add_intermediate_scale)

        assert len(self.feature_strides) == len(self.scales), "strides 与 scales 数量需一致"
        self.anchors_per_loc = len(self.aspect_ratios) + (1 if self.add_intermediate else 0)

    def _per_level_anchors(self, feat_h: int, feat_w: int, stride: int, s_k: float, s_k1: float):
        size_k = s_k * self.img_size
        size_k_prime = (s_k * s_k1) ** 0.5 * self.img_size

        shifts_x = (torch.arange(feat_w, dtype=torch.float32) + 0.5) * stride
        shifts_y = (torch.arange(feat_h, dtype=torch.float32) + 0.5) * stride
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
        cx = shift_x.reshape(-1)
        cy = shift_y.reshape(-1)

        ws, hs = [], []
        # ar=1
        ws.append(torch.full_like(cx, size_k));            hs.append(torch.full_like(cy, size_k))
        # ar=2
        ws.append(torch.full_like(cx, size_k * (2.0 ** 0.5))); hs.append(torch.full_like(cy, size_k / (2.0 ** 0.5)))
        # ar=0.5
        ws.append(torch.full_like(cx, size_k / (2.0 ** 0.5))); hs.append(torch.full_like(cy, size_k * (2.0 ** 0.5)))
        # 中间尺度 ar=1
        if self.add_intermediate:
            ws.append(torch.full_like(cx, size_k_prime));  hs.append(torch.full_like(cy, size_k_prime))

        ws = torch.stack(ws, dim=1)
        hs = torch.stack(hs, dim=1)

        x1 = cx[:, None] - 0.5 * ws
        y1 = cy[:, None] - 0.5 * hs
        x2 = cx[:, None] + 0.5 * ws
        y2 = cy[:, None] + 0.5 * hs

        anchors = torch.stack([x1, y1, x2, y2], dim=2).reshape(-1, 4)
        eps = 1e-3
        anchors[:, 0::2] = anchors[:, 0::2].clamp(0.0, self.img_size - eps)
        anchors[:, 1::2] = anchors[:, 1::2].clamp(0.0, self.img_size - eps)
        return anchors

    @torch.no_grad()
    def forward(self, feature_shapes: List[Tuple[int, int]]) -> torch.Tensor:
        all_anchors = []
        L = len(self.feature_strides)
        for i in range(L):
            Hf, Wf = feature_shapes[i]
            stride = self.feature_strides[i]
            s_k = self.scales[i]
            s_k1 = self.scales[min(i + 1, L - 1)]
            anchors_i = self._per_level_anchors(Hf, Wf, stride, s_k, s_k1)
            all_anchors.append(anchors_i)
        return torch.cat(all_anchors, dim=0)  # [A,4]


# -------------------------
# SSDLite FPN 主体
# -------------------------
class SSDLite_FPN(nn.Module):
    """
    返回:
      dict {
        'cls_logits': [B, A, C],
        'bbox_deltas': [B, A, 4],
        'anchors': [B, A, 4]  # batched 展开，使用更直观
      }
    """
    def __init__(self,
                 backbone,
                 num_classes: int = 80,
                 fpn_out_channels: int = 128,
                 img_size: int = 320,
                 feature_strides: List[int] = None,
                 anchor_scales: List[float] = None,
                 aspect_ratios: List[float] = None,
                 add_intermediate_scale: bool = True,
                 prior_pi: float = 0.01):
        super().__init__()
        self.backbone = backbone
        self.num_classes = int(num_classes)
        self.img_size = int(img_size)
        self.feature_strides = feature_strides or [8,16,32,64,128]

        # 1) FPN
        backbone_channels = self.backbone.feature_info.channels()  # expect (C3,C4,C5)
        self.fpn = FPN(in_channels=backbone_channels, out_channels=fpn_out_channels)

        # 2) extra layers -> P6,P7
        self.extra_layers = nn.ModuleList([
            nn.Conv2d(fpn_out_channels, fpn_out_channels, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(fpn_out_channels, fpn_out_channels, kernel_size=3, stride=2, padding=1)
        ])
        self.feature_map_channels = [fpn_out_channels] * 5  # P3..P7

        # 3) Anchors
        self.anchor_gen = SSDAnchorGenerator(
            img_size=img_size,
            feature_strides=self.feature_strides,
            scales=anchor_scales,
            aspect_ratios=aspect_ratios,
            add_intermediate_scale=add_intermediate_scale
        )
        num_anchors_per_loc = self.anchor_gen.anchors_per_loc

        # 4) 预测头
        self.cls_heads = nn.ModuleList()
        self.reg_heads = nn.ModuleList()
        for in_ch in self.feature_map_channels:
            self.cls_heads.append(SSDLitePredictionHead(in_ch, self.num_classes, num_anchors_per_loc))
            self.reg_heads.append(SSDLitePredictionHead(in_ch, 4, num_anchors_per_loc))

        # 缓存 anchors
        self.cached_shapes = None
        self.cached_anchors = None

        # ★ 分类头先验负偏置：让初始前景概率很小（~pi）
        self._init_cls_head_prior(prior_pi)

    def _init_cls_head_prior(self, pi=0.01):
        prior_bias = -float(np.log((1 - pi) / pi))
        for head in self.cls_heads:
            conv1x1 = head[-1]  # 最后一层 1x1 conv
            nn.init.constant_(conv1x1.bias, prior_bias)

    def _collect_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = self.backbone(x)  # list/tuple
        c3, c4, c5 = feats[-3], feats[-2], feats[-1]
        p3, p4, p5 = self.fpn((c3, c4, c5))
        p6 = self.extra_layers[0](p5)
        p7 = self.extra_layers[1](p6)
        return [p3, p4, p5, p6, p7]

    def _maybe_build_anchors(self, features: List[torch.Tensor], device: torch.device):
        shapes = [(f.shape[-2], f.shape[-1]) for f in features]
        if self.cached_shapes != shapes:
            anchors = self.anchor_gen(shapes)  # [A,4] cpu
            self.cached_anchors = anchors.to(device)
            self.cached_shapes = shapes
        return self.cached_anchors  # [A,4]

    def forward(self, x: torch.Tensor):
        B, _, H, W = x.shape
        assert H == W == self.img_size, f"Input must be {self.img_size}x{self.img_size}"

        feats = self._collect_features(x)
        anchors = self._maybe_build_anchors(feats, x.device)  # [A,4] on device

        cls_preds, reg_preds = [], []
        for i, f in enumerate(feats):
            cls = self.cls_heads[i](f)               # [B, Apl*C, Hf, Wf]
            cls = cls.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_classes)
            cls_preds.append(cls)

            reg = self.reg_heads[i](f)               # [B, Apl*4, Hf, Wf]
            reg = reg.permute(0, 2, 3, 1).contiguous().view(B, -1, 4)
            reg_preds.append(reg)

        cls_preds = torch.cat(cls_preds, dim=1)      # [B, A, C]
        reg_preds = torch.cat(reg_preds, dim=1)      # [B, A, 4]
        anchors_b = anchors.unsqueeze(0).expand(B, -1, -1).contiguous()  # [B, A, 4]

        return {
            "cls_logits": cls_preds,
            "bbox_deltas": reg_preds,
            "anchors": anchors_b
        }
