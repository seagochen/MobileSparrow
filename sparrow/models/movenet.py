from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from sparrow.models.backbones.mobilenet_v2 import MobileNetV2Backbone
from sparrow.models.backbones.mobilenet_v3 import MobileNetV3Backbone
from sparrow.models.backbones.shufflenet_v2 import ShuffleNetV2Backbone


BACKBONES = {
    "mobilenet_v2": MobileNetV2Backbone,
    "shufflenet_v2": ShuffleNetV2Backbone,
    "mobilenet_v3": MobileNetV3Backbone,
}


class MoveNet(nn.Module):
    def __init__(self,
                 backbone: str = "mobilenet_v2",
                 num_joints: int = 17,
                 neck_outc: int = 64,
                 head_midc: int = 32,
                 width_mult: float = 1.0):
        super().__init__()
        assert backbone in BACKBONES, f"unknown backbone: {backbone}"

        # 1) 构建 backbone，并把 width_mult 传进去
        self.backbone = BACKBONES[backbone](width_mult=width_mult)

        # 2) 动态获取 C3/C4/C5 通道，配置 FPN
        c3c, c4c, c5c = self.backbone.get_out_channels()
        self.neck = FPNLite(c3=c3c, c4=c4c, c5=c5c, outc=neck_outc)

        # 3) 头部
        self.head = MoveNetHead(neck_outc, num_joints=num_joints, midc=head_midc)

    def forward(self, x) -> Dict[str, torch.Tensor]:

        # 使用 mobilenet 或者 shufflenet 进行特征提取
        c3, c4, c5 = self.backbone(x)

        # 使用fpn进行特征融合
        p3 = self.neck(c3, c4, c5)

        # 结果推理
        return self.head(p3)


class MoveNetHead(nn.Module):
    """
    四头输出：heatmaps / centers / regs / offsets
    """
    def __init__(self, in_ch, num_joints=17, midc=32):
        super().__init__()

        self.hm = nn.Sequential(
            nn.Conv2d(in_ch, midc, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(midc, num_joints, 1), nn.Sigmoid()
        )
        self.ct = nn.Sequential(
            nn.Conv2d(in_ch, midc, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(midc, 1, 1), nn.Sigmoid()
        )
        self.reg = nn.Sequential(
            nn.Conv2d(in_ch, midc, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(midc, num_joints * 2, 1)   # (dx, dy)
        )
        self.off = nn.Sequential(
            nn.Conv2d(in_ch, midc, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(midc, num_joints * 2, 1)   # (ox, oy)
        )

    def forward(self, x):
        return {
            "heatmaps": self.hm(x),     # B×17×48×48,
            "centers":  self.ct(x),     # B×1 ×48×48,
            "regs":     self.reg(x),    # B×34×48×48,
            "offsets":  self.off(x),    # B×34×48×48
        }


class FPNLite(nn.Module):
    """
    接收 (C3, C4, C5) -> 输出单尺度 P3（或多尺度列表，按需改）
    这里直接用 nn.Sequential 定义 1x1 与 3x3 conv 块（Conv + BN + ReLU）。
    """

    def __init__(self, c3, c4, c5, outc=64):
        super().__init__()
        # 1x1 conv blocks
        self.l3 = nn.Sequential(
            nn.Conv2d(c3, outc, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True)
        )
        self.l4 = nn.Sequential(
            nn.Conv2d(c4, outc, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True)
        )
        self.l5 = nn.Sequential(
            nn.Conv2d(c5, outc, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True)
        )

        # 3x3 conv blocks
        self.smooth4 = nn.Sequential(
            nn.Conv2d(outc, outc, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True)
        )
        self.smooth3 = nn.Sequential(
            nn.Conv2d(outc, outc, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True)
        )

        # 上采样后的最终平滑层
        self.final_smooth = nn.Sequential(
            nn.Conv2d(outc, outc, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True)
        )

    def forward(self, c3, c4, c5):
        p5 = self.l5(c5)
        p4 = self.l4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p4 = self.smooth4(p4)

        p3 = self.l3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p3 = self.smooth3(p3)

        p3_upsampled = F.interpolate(p3, scale_factor=2.0, mode='bilinear', align_corners=False)
        p3_final = self.final_smooth(p3_upsampled)
        return p3_final


