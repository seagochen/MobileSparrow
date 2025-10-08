# -*- coding: utf-8 -*-
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    """
    Feature Pyramid Network (FPN) Neck
    对 Backbone 输出的 (C3, C4, C5) 特征进行融合，生成 (P3, P4, P5)。
    """

    def __init__(self, in_channels: Tuple[int, int, int], out_channels: int = 256):
        super().__init__()
        c3_in, c4_in, c5_in = in_channels

        # 1. 侧向连接的 1x1 卷积，用于统一通道数
        self.lateral_conv3 = nn.Conv2d(c3_in, out_channels, kernel_size=1)
        self.lateral_conv4 = nn.Conv2d(c4_in, out_channels, kernel_size=1)
        self.lateral_conv5 = nn.Conv2d(c5_in, out_channels, kernel_size=1)

        # 2. 输出处理的 3x3 卷积，用于平滑特征
        self.output_conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.output_conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.output_conv5 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self,
                features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> \
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # 解包 Backbone 输出的三层特征图：C3(低层)、C4(中层)、C5(高层)
        c3, c4, c5 = features

        # 自顶向下路径
        # 1. 处理最高层 C5：通过 1x1 卷积统一通道数
        p5_in = self.lateral_conv5(c5)

        # 2. 处理 C4 层：融合 C4 和上采样的 P5
        p4_in = self.lateral_conv4(c4)  # 1x1 卷积统一通道数
        p5_up = F.interpolate(p5_in, size=p4_in.shape[-2:], mode='nearest')  # 将 P5 上采样到 P4 的空间尺寸
        p4_in = p4_in + p5_up  # 特征融合：侧向连接 + 自顶向下

        # 3. 处理 C3 层：融合 C3 和上采样的 P4
        p3_in = self.lateral_conv3(c3)  # 1x1 卷积统一通道数
        p4_up = F.interpolate(p4_in, size=p3_in.shape[-2:], mode='nearest')  # 将 P4 上采样到 P3 的空间尺寸
        p3_in = p3_in + p4_up  # 特征融合：侧向连接 + 自顶向下

        # 输出卷积：使用 3x3 卷积平滑融合后的特征，消除上采样带来的混叠效应
        p3 = self.output_conv3(p3_in)  # 输出 P3：高分辨率特征（用于检测小目标）
        p4 = self.output_conv4(p4_in)  # 输出 P4：中分辨率特征（用于检测中等目标）
        p5 = self.output_conv5(p5_in)  # 输出 P5：低分辨率特征（用于检测大目标）

        return p3, p4, p5
