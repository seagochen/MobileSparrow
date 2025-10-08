# -*- coding: utf-8 -*-
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn

from sparrow.models.fpn import FPN


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
    SSD 风格的 Anchor 生成器

    功能：为多层特征图生成密集的 anchor boxes（预定义候选框）

    默认配置：
      - scales: [0.04, 0.08, 0.16, 0.32, 0.64] - 每层 anchor 相对图像的尺度
      - aspect_ratios: [1.0, 2.0, 0.5] - 宽高比（正方形、横向、纵向）
      - add_intermediate_scale: True - 额外添加中间尺度的正方形 anchor
      => 每个位置生成 4 个 anchors（3种宽高比 + 1个中间尺度）

    输出格式：xyxy（像素坐标，左上角和右下角）
    """

    def __init__(self,
                 img_size: int,
                 feature_strides: List[int],
                 scales: List[float] = None,
                 aspect_ratios: List[float] = None,
                 add_intermediate_scale: bool = True):
        """
        参数:
          img_size: 输入图像尺寸（正方形，如 320）
          feature_strides: 每层特征图的下采样率，如 [8,16,32,64,128] 对应 P3~P7
          scales: 每层 anchor 的尺度（相对图像大小的比例）
          aspect_ratios: anchor 的宽高比列表
          add_intermediate_scale: 是否添加中间尺度（s_k 和 s_{k+1} 的几何平均）
        """
        super().__init__()
        self.img_size = int(img_size)
        self.feature_strides = feature_strides
        # 默认 scales：从小到大，对应 P3(小目标) 到 P7(大目标)
        self.scales = scales if scales is not None else [0.04, 0.08, 0.16, 0.32, 0.64]
        # 默认 aspect_ratios：1(正方形), 2(横向), 0.5(纵向)
        self.aspect_ratios = aspect_ratios if aspect_ratios is not None else [1.0, 2.0, 0.5]
        self.add_intermediate = bool(add_intermediate_scale)

        # 确保每层都有对应的 scale
        assert len(self.feature_strides) == len(self.scales), "strides 与 scales 数量需一致"
        # 计算每个位置的 anchor 数量
        self.anchors_per_loc = len(self.aspect_ratios) + (1 if self.add_intermediate else 0)

    def _per_level_anchors(self, feat_h: int, feat_w: int, stride: int, s_k: float, s_k1: float):
        """
        为单层特征图生成 anchors

        参数:
          feat_h, feat_w: 特征图的高度和宽度
          stride: 该层的下采样率（如 8 表示特征图上1个像素对应原图8个像素）
          s_k: 当前层的 scale（如 0.08）
          s_k1: 下一层的 scale（用于计算中间尺度）

        返回:
          anchors: [N, 4] - N = feat_h * feat_w * anchors_per_loc
        """
        # 1. 计算 anchor 的基础尺寸（像素）
        size_k = s_k * self.img_size  # 当前层的基础尺寸
        size_k_prime = (s_k * s_k1) ** 0.5 * self.img_size  # 中间尺度（几何平均）

        # 2. 生成特征图上每个位置的中心点坐标（映射到原图像素坐标）
        # +0.5 使中心点落在特征图单元格的中心
        shifts_x = (torch.arange(feat_w, dtype=torch.float32) + 0.5) * stride  # [feat_w]
        shifts_y = (torch.arange(feat_h, dtype=torch.float32) + 0.5) * stride  # [feat_h]
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')  # [feat_h, feat_w]
        cx = shift_x.reshape(-1)  # [feat_h*feat_w] - 所有位置的 x 坐标
        cy = shift_y.reshape(-1)  # [feat_h*feat_w] - 所有位置的 y 坐标

        # 3. 为每个中心点生成不同宽高比的 anchors
        ws, hs = [], []  # 存储宽度和高度

        # aspect_ratio = 1.0 (正方形)
        ws.append(torch.full_like(cx, size_k))
        hs.append(torch.full_like(cy, size_k))

        # aspect_ratio = 2.0 (宽是高的2倍，横向矩形)
        # 面积保持不变：w*h = size_k^2, w/h=2 => w=size_k*√2, h=size_k/√2
        ws.append(torch.full_like(cx, size_k * (2.0 ** 0.5)))
        hs.append(torch.full_like(cy, size_k / (2.0 ** 0.5)))

        # aspect_ratio = 0.5 (高是宽的2倍，纵向矩形)
        ws.append(torch.full_like(cx, size_k / (2.0 ** 0.5)))
        hs.append(torch.full_like(cy, size_k * (2.0 ** 0.5)))

        # 中间尺度的正方形 anchor（ar=1, 但尺寸介于当前层和下一层之间）
        if self.add_intermediate:
            ws.append(torch.full_like(cx, size_k_prime))
            hs.append(torch.full_like(cy, size_k_prime))

        # 4. 将宽高堆叠：[N, anchors_per_loc]
        ws = torch.stack(ws, dim=1)  # [N, 3或4]
        hs = torch.stack(hs, dim=1)  # [N, 3或4]

        # 5. 从中心点和宽高转换为 xyxy 格式
        x1 = cx[:, None] - 0.5 * ws  # 左上角 x
        y1 = cy[:, None] - 0.5 * hs  # 左上角 y
        x2 = cx[:, None] + 0.5 * ws  # 右下角 x
        y2 = cy[:, None] + 0.5 * hs  # 右下角 y

        # 6. 组合为 [x1, y1, x2, y2] 并展平
        anchors = torch.stack([x1, y1, x2, y2], dim=2).reshape(-1, 4)  # [N*anchors_per_loc, 4]

        # 7. 裁剪到图像边界内（避免越界）
        eps = 1e-3  # 小偏移量，避免数值问题
        anchors[:, 0::2] = anchors[:, 0::2].clamp(0.0, self.img_size - eps)  # x 坐标
        anchors[:, 1::2] = anchors[:, 1::2].clamp(0.0, self.img_size - eps)  # y 坐标

        return anchors

    @torch.no_grad()
    def forward(self, feature_shapes: List[Tuple[int, int]]) -> torch.Tensor:
        """
        为所有特征层生成 anchors

        参数:
          feature_shapes: 每层特征图的 (H, W) 列表，如 [(40,40), (20,20), (10,10), (5,5), (3,3)]

        返回:
          anchors: [A, 4] - 所有层级的 anchors 拼接，A = Σ(Hi*Wi*anchors_per_loc)

        说明:
          - 使用 @torch.no_grad() 因为 anchor 生成不需要梯度
          - 按层级顺序拼接（P3 -> P4 -> ... -> P7）
        """
        all_anchors = []
        L = len(self.feature_strides)

        for i in range(L):
            Hf, Wf = feature_shapes[i]  # 当前层的特征图尺寸
            stride = self.feature_strides[i]  # 当前层的下采样率
            s_k = self.scales[i]  # 当前层的 scale
            s_k1 = self.scales[min(i + 1, L - 1)]  # 下一层的 scale（最后一层用自己）

            # 生成当前层的 anchors
            anchors_i = self._per_level_anchors(Hf, Wf, stride, s_k, s_k1)
            all_anchors.append(anchors_i)

        # 拼接所有层的 anchors
        return torch.cat(all_anchors, dim=0)  # [A, 4]


# -------------------------
# SSDLite FPN 主体
# -------------------------
class SSDLite_FPN(nn.Module):
    """
    SSDLite with FPN 目标检测模型

    架构：
      Backbone -> FPN(P3,P4,P5) -> ExtraLayers(P6,P7) -> 多尺度预测头

    返回:
      dict {
        'cls_logits': [B, A, C],      # 分类 logits
        'bbox_deltas': [B, A, 4],     # 边界框回归增量
        'anchors': [B, A, 4]          # 对应的 anchor boxes (xyxy 格式)
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
        self.feature_strides = feature_strides or [8, 16, 32, 64, 128]  # P3~P7 的下采样率

        # 1) 创建 FPN：将 Backbone 的 C3,C4,C5 融合成 P3,P4,P5
        backbone_channels = self.backbone.feature_info.channels()  # 获取 (C3,C4,C5) 的通道数
        self.fpn = FPN(in_channels=backbone_channels, out_channels=fpn_out_channels)

        # 2) 额外层：通过两个 stride=2 的卷积生成 P6 和 P7（更低分辨率，用于检测更大目标）
        self.extra_layers = nn.ModuleList([
            nn.Conv2d(fpn_out_channels, fpn_out_channels, kernel_size=3, stride=2, padding=1),  # P5 -> P6
            nn.Conv2d(fpn_out_channels, fpn_out_channels, kernel_size=3, stride=2, padding=1)  # P6 -> P7
        ])
        self.feature_map_channels = [fpn_out_channels] * 5  # P3~P7 都使用相同的通道数

        # 3) Anchor 生成器：为每个特征层生成密集的 anchor boxes
        self.anchor_gen = SSDAnchorGenerator(
            img_size=img_size,
            feature_strides=self.feature_strides,
            scales=anchor_scales,
            aspect_ratios=aspect_ratios,
            add_intermediate_scale=add_intermediate_scale
        )
        num_anchors_per_loc = self.anchor_gen.anchors_per_loc  # 每个位置的 anchor 数量

        # 4) 预测头：为每个特征层创建分类头和回归头
        self.cls_heads = nn.ModuleList()  # 分类头（预测类别）
        self.reg_heads = nn.ModuleList()  # 回归头（预测边界框偏移）
        for in_ch in self.feature_map_channels:
            self.cls_heads.append(SSDLitePredictionHead(in_ch, self.num_classes, num_anchors_per_loc))
            self.reg_heads.append(SSDLitePredictionHead(in_ch, 4, num_anchors_per_loc))

        # 5) 缓存机制：避免重复生成 anchors
        self.cached_shapes = None  # 缓存特征图尺寸
        self.cached_anchors = None  # 缓存生成的 anchors

        # 6) 初始化分类头偏置：让模型初始时倾向于预测背景（解决类别不平衡）
        self._init_cls_head_prior(prior_pi)

    def _init_cls_head_prior(self, pi=0.01):
        """
        初始化分类头的偏置，使得初始前景概率约为 pi（默认 0.01）
        这有助于训练初期的稳定性，避免过多的假阳性预测
        """
        prior_bias = -float(np.log((1 - pi) / pi))  # 计算对应的 logit 偏置
        for head in self.cls_heads:
            conv1x1 = head[-1]  # 获取预测头的最后一层 1x1 卷积
            nn.init.constant_(conv1x1.bias, prior_bias)

    def _collect_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        收集多尺度特征金字塔

        流程：
          1. Backbone 提取 C3, C4, C5
          2. FPN 融合生成 P3, P4, P5
          3. ExtraLayers 生成 P6, P7

        返回: [P3, P4, P5, P6, P7] - 5 个不同尺度的特征图
        """
        feats = self.backbone(x)  # Backbone 前向传播，返回多层特征
        c3, c4, c5 = feats[-3], feats[-2], feats[-1]  # 提取 C3, C4, C5
        p3, p4, p5 = self.fpn((c3, c4, c5))  # FPN 融合
        p6 = self.extra_layers[0](p5)  # P5 下采样生成 P6
        p7 = self.extra_layers[1](p6)  # P6 下采样生成 P7
        return [p3, p4, p5, p6, p7]

    def _maybe_build_anchors(self, features: List[torch.Tensor], device: torch.device):
        """
        按需生成并缓存 anchors

        只有当特征图尺寸发生变化时才重新生成，避免重复计算

        参数:
          features: 特征金字塔列表
          device: 目标设备（CPU/GPU）

        返回: anchors [A, 4] - 所有层级的 anchors 合并（xyxy 格式）
        """
        shapes = [(f.shape[-2], f.shape[-1]) for f in features]  # 提取每层的 (H, W)
        if self.cached_shapes != shapes:  # 检查是否需要重新生成
            anchors = self.anchor_gen(shapes)  # 生成 anchors（初始在 CPU）
            self.cached_anchors = anchors.to(device)  # 移到目标设备
            self.cached_shapes = shapes  # 更新缓存
        return self.cached_anchors  # [A, 4]

    def forward(self, x: torch.Tensor):
        """
        前向传播

        参数:
          x: 输入图像 [B, 3, H, W]，要求 H=W=img_size

        返回:
          dict {
            'cls_logits': [B, A, C] - 分类预测（未经 sigmoid）
            'bbox_deltas': [B, A, 4] - 边界框回归增量
            'anchors': [B, A, 4] - 对应的 anchor boxes
          }
        """
        B, _, H, W = x.shape
        assert H == W == self.img_size, f"Input must be {self.img_size}x{self.img_size}"

        # 1. 提取多尺度特征金字塔 P3~P7
        feats = self._collect_features(x)

        # 2. 生成 anchors（所有层级共 A 个 anchors）
        anchors = self._maybe_build_anchors(feats, x.device)  # [A, 4]

        # 3. 对每个特征层进行预测
        cls_preds, reg_preds = [], []
        for i, f in enumerate(feats):
            # 分类预测
            cls = self.cls_heads[i](f)  # [B, Apl*C, Hf, Wf] (Apl=每位置anchor数)
            cls = cls.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_classes)  # -> [B, Ai, C]
            cls_preds.append(cls)

            # 回归预测
            reg = self.reg_heads[i](f)  # [B, Apl*4, Hf, Wf]
            reg = reg.permute(0, 2, 3, 1).contiguous().view(B, -1, 4)  # -> [B, Ai, 4]
            reg_preds.append(reg)

        # 4. 拼接所有层级的预测结果
        cls_preds = torch.cat(cls_preds, dim=1)  # [B, A, C] - A 为所有层级 anchors 总数
        reg_preds = torch.cat(reg_preds, dim=1)  # [B, A, 4]

        # 5. 为每个 batch 复制 anchors（方便后续处理）
        anchors_b = anchors.unsqueeze(0).expand(B, -1, -1).contiguous()  # [B, A, 4]

        return {
            "cls_logits": cls_preds,  # 分类 logits（训练时配合 Focal Loss）
            "bbox_deltas": reg_preds,  # 边界框增量（需要解码到原图坐标）
            "anchors": anchors_b  # 对应的 anchor boxes
        }