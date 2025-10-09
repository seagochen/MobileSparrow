# -*- coding: utf-8 -*-
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from sparrow.models.fpn import FPN


class SinglePoseHead(nn.Module):
    """
    单人姿态估计头部网络

    应用场景：
      - 输入已经是裁剪好的单人区域（由 SSDLite 检测器提供人体框）
      - 只负责框内关键点的精确定位

    输出：
      - heatmaps: [B, K, H, W] - 关键点热图（logits，未经 sigmoid）
      - offsets: [B, 2K, H, W] - 亚像素级偏移量（每个关键点的 x,y 偏移）

    设计特点：
      - 双塔结构：热图和偏移分支独立处理
      - 轻量化：使用深度可分离卷积（Depthwise + Pointwise）
      - 无中心点分支：单人场景不需要人体中心检测

    训练损失：
      - Heatmaps: Gaussian Focal Loss（适配软标签）
      - Offsets: L1 或 Smooth L1 Loss（仅在关键点邻域计算）
    """

    def __init__(self, in_ch: int, num_joints: int = 17, midc: int = 32):
        """
        初始化单人姿态估计头

        参数:
          in_ch: 输入特征通道数（来自 FPN 的 P3 特征）
          num_joints: 关键点数量（COCO 人体关键点为 17 个）
          midc: 中间层通道数（控制模型容量和计算量）
        """
        super().__init__()

        # 深度可分离卷积块（Depthwise + Pointwise）
        # 优势：大幅减少参数量和计算量，适合移动端部署
        def dw_pw(in_c, out_c):
            """
            深度可分离卷积：将标准卷积分解为两步
            1. Depthwise: 每个通道独立进行 3x3 卷积（groups=in_c）
            2. Pointwise: 1x1 卷积进行通道间信息融合
            """
            return nn.Sequential(
                # Depthwise 卷积：3x3 空间卷积，不改变通道数
                nn.Conv2d(in_c, in_c, 3, padding=1, groups=in_c, bias=False),
                nn.ReLU(inplace=True),
                # Pointwise 卷积：1x1 跨通道卷积，改变通道数
                nn.Conv2d(in_c, out_c, 1, bias=True)
            )

        # 热图分支：降维到中间通道
        self.hm_tower = dw_pw(in_ch, midc)
        # 偏移分支：降维到中间通道（与热图分支独立）
        self.off_tower = dw_pw(in_ch, midc)

        # 最终预测层（1x1 卷积）
        # 热图预测：每个关键点一个通道
        self.hm = nn.Conv2d(midc, num_joints, 1)
        # 偏移预测：每个关键点两个通道（x 和 y 方向的偏移）
        self.off = nn.Conv2d(midc, num_joints * 2, 1)

        # 初始化策略：让热图初始值偏向稀疏（更多背景，更少关键点）
        # bias=-2.0 意味着 sigmoid(-2.0) ≈ 0.12，使得初始预测偏向背景
        # 好处：避免训练初期过多假阳性，加快收敛
        nn.init.constant_(self.hm.bias, -2.0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播：从 FPN 特征预测关键点热图和偏移

        参数:
          x: 输入特征 [B, C, H, W]，通常来自 FPN 的 P3 层
             - 对于 stride=8 的 P3：H=W=input_size/8
             - 如果启用 upsample_to_quarter：H=W=input_size/4

        返回:
          字典包含：
            - "heatmaps": [B, K, H, W] - 关键点热图 logits
                         训练时配合 BCEWithLogitsLoss 或 Focal Loss
            - "offsets": [B, 2K, H, W] - 亚像素偏移量
                         格式：[x0, y0, x1, y1, ..., x_{K-1}, y_{K-1}]
        """
        # 热图分支：特征提取 -> 预测
        heatmaps = self.hm(self.hm_tower(x))  # [B, K, H, W]

        # 偏移分支：特征提取 -> 预测
        offsets = self.off(self.off_tower(x))  # [B, 2K, H, W]

        return {"heatmaps": heatmaps, "offsets": offsets}


class MoveNet_FPN(nn.Module):
    """
    MoveNet 单人姿态估计模型（基于 FPN）

    整体架构：
      Backbone -> FPN -> P3 提取 -> (可选上采样) -> SinglePoseHead

    特点：
      1. 使用 FPN 的 P3 层（stride=8）：平衡分辨率和语义信息
      2. 可选上采样到 1/4 分辨率：提高小关键点的定位精度
      3. 轻量化设计：适合实时应用

    适用场景：
      - 单人姿态估计（输入是裁剪好的人体区域）
      - 与 SSDLite 检测器配合使用（检测 + 姿态级联）

    参考：
      MoveNet 论文（Google）强调速度和精度的平衡
    """

    def __init__(self,
                 backbone,
                 num_joints: int = 17,
                 fpn_out_channels: int = 128,
                 head_midc: int = 32,
                 upsample_to_quarter: bool = False):
        """
        初始化 MoveNet_FPN 模型

        参数:
          backbone: 特征提取器（timm 模型，features_only=True）
                   要求输出 (C3, C4, C5) 三层特征
                   - C3: stride=8, 浅层特征（高分辨率）
                   - C4: stride=16, 中层特征
                   - C5: stride=32, 深层特征（高语义）
          num_joints: 关键点数量（COCO: 17，MPII: 16）
          fpn_out_channels: FPN 输出的统一通道数
          head_midc: 预测头的中间层通道数（控制模型大小）
          upsample_to_quarter: 是否将 P3 上采样到 1/4 分辨率
                              - False: 输出 H/8 × W/8（更快）
                              - True: 输出 H/4 × W/4（更精确）
        """
        super().__init__()
        self.backbone = backbone

        # 创建 FPN：融合多尺度特征
        # 获取 Backbone 的输出通道数 (C3, C4, C5)
        backbone_channels = self.backbone.feature_info.channels()
        self.fpn = FPN(in_channels=backbone_channels,
                       out_channels=fpn_out_channels)

        # 创建姿态估计头：双分支输出（热图 + 偏移）
        self.head = SinglePoseHead(in_ch=fpn_out_channels,
                                   num_joints=num_joints,
                                   midc=head_midc)

        # 保存配置：是否上采样
        self.upsample_to_quarter = upsample_to_quarter

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播：从输入图像预测关键点

        参数:
          x: 输入图像 [B, 3, H, W]
             - 通常是裁剪好的人体区域（由检测器提供）
             - 标准尺寸：192×192, 256×256 等

        返回:
          字典包含：
            - "heatmaps": [B, K, H', W'] - 关键点热图
            - "offsets": [B, 2K, H', W'] - 亚像素偏移
          其中 H', W' 取决于 upsample_to_quarter:
            - False: H'=H/8, W'=W/8（默认）
            - True: H'=H/4, W'=W/4（高精度模式）

        流程：
          1. Backbone 提取多尺度特征 (C3, C4, C5)
          2. FPN 融合生成 (P3, P4, P5)
          3. 使用 P3 作为姿态估计的输入（最高分辨率）
          4. 可选：上采样 P3 到 1/4 分辨率
          5. 预测头输出热图和偏移
        """
        # 1. Backbone 特征提取
        c3, c4, c5 = self.backbone(x)  # 三层特征金字塔

        # 2. FPN 融合：自顶向下 + 侧向连接
        p3, _, _ = self.fpn((c3, c4, c5))  # 只使用 P3（高分辨率特征）
        # 注意：忽略 P4 和 P5，因为单人姿态估计只需要高分辨率特征

        # 3. 可选上采样：提高输出分辨率
        if self.upsample_to_quarter:
            # 双线性插值上采样 2 倍：stride 8 -> stride 4
            # align_corners=False 是 PyTorch 的推荐设置（更精确）
            p3 = F.interpolate(p3, scale_factor=2.0, mode='bilinear', align_corners=False)

        # 4. 预测头：生成热图和偏移
        return self.head(p3)


