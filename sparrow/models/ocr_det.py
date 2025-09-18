import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from sparrow.models.backbones.mobilenet_v2 import MobileNetV2Backbone
from sparrow.models.backbones.mobilenet_v3 import MobileNetV3Backbone
from sparrow.models.backbones.shufflenet_v2 import ShuffleNetV2Backbone


BACKBONES = {
    "mobilenet_v2": MobileNetV2Backbone,
    "mobilenet_v3": MobileNetV3Backbone,
    "shufflenet_v2": ShuffleNetV2Backbone,
}


class OCRDetDB(nn.Module):
    """Lightweight text detector using FPN-Lite + DB head.
    Args:
        backbone: one of {'mobilenet_v2','mobilenet_v3','shufflenet_v2'}
        width_mult: channel scaling for backbone
        neck_outc: FPN output channels per level
        head_midc: conv mid channels in DB head
        use_fixed_thresh: if True, only outputs prob_map (use scalar threshold in postprocess)
    Returns:
        dict with:
          'features': [P3,P4,P5]
          'prob_map': (B,1,H/4,W/4)  (assuming P3 stride=4 wrt input after stem; exact factor depends on backbone)
          'thresh_map': (B,1,H/4,W/4) or None
    """
    def __init__(self, backbone: str = "mobilenet_v3", width_mult: float = 1.0,
                 neck_outc: int = 96, head_midc: int = 64, use_fixed_thresh: bool = True):
        super().__init__()
        assert backbone in BACKBONES, f"unknown backbone: {backbone}"
        self.backbone = BACKBONES[backbone](width_mult=width_mult)
        c3, c4, c5 = self.backbone.get_out_channels()
        self.neck = FPNLite(c3, c4, c5, outc=neck_outc)
        self.head = DBHeadLite(neck_outc, head_midc, use_fixed_thresh=use_fixed_thresh)

    def forward(self, x):
        c3, c4, c5 = self.backbone(x)
        feats = self.neck(c3, c4, c5)  # [P3,P4,P5]
        p3 = feats[0]
        prob_map, thresh_map = self.head(p3)
        return {"features": feats, "prob_map": prob_map, "thresh_map": thresh_map}

    @staticmethod
    def db_binarize(prob_map, thresh=0.3):
        """prob_map: (B,1,H,W) -> (B,H,W) uint8 mask in [0,255]."""
        pm = prob_map.detach().cpu().numpy()
        masks = (pm > float(thresh)).astype(np.uint8) * 255
        masks = masks.squeeze(1)  # -> (B,H,W)
        return masks


class DBHeadLite(nn.Module):
    """A minimal Differentiable Binarization head (lite).
    Outputs:
      - prob_map: text probability (B,1,H,W)
      - thresh_map: threshold map (B,1,H,W)  (optional; can be None if use_fixed_thresh=True)
    """
    def __init__(self, in_ch: int, mid_ch: int = 64, use_fixed_thresh: bool = True):
        super().__init__()
        self.use_fixed_thresh = bool(use_fixed_thresh)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, 1, 1, bias=False), nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, 1, 1, bias=False), nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True),
        )
        self.prob = nn.Conv2d(mid_ch, 1, 1, 1, 0)
        if not self.use_fixed_thresh:
            self.thresh = nn.Conv2d(mid_ch, 1, 1, 1, 0)
        else:
            self.register_parameter('thresh', None)

    def forward(self, x):
        y = self.conv(x)
        prob_map = torch.sigmoid(self.prob(y))
        if self.use_fixed_thresh:
            return prob_map, None
        else:
            thresh_map = torch.sigmoid(self.thresh(y))
            return prob_map, thresh_map



class DepthwiseSeparableConv2d(nn.Module):
    """DW(3x3) + PW(1x1) + BN + ReLU —— 轻量卷积块"""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

        self._init_weights()

    def _init_weights(self):
        for m in (self.dw, self.pw):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        nn.init.ones_(self.bn.weight)
        nn.init.zeros_(self.bn.bias)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class FPNLite(nn.Module):
    """
    轻量版 FPN：
      输入: C3, C4, C5 (来自骨干网络的三个尺度特征)
      过程: 1x1 横向降维 -> 自顶向下逐级上采样相加 -> 3x3(深度可分离)平滑
      输出: [P3, P4, P5]，每级通道数均为 outc
    """
    def __init__(self, c3_in: int, c4_in: int, c5_in: int, outc: int = 96):
        super().__init__()

        # 横向 1x1 降维
        self.lateral3 = nn.Conv2d(c3_in, outc, kernel_size=1, bias=False)
        self.lateral4 = nn.Conv2d(c4_in, outc, kernel_size=1, bias=False)
        self.lateral5 = nn.Conv2d(c5_in, outc, kernel_size=1, bias=False)

        # 平滑（使用深度可分离卷积，轻量化）
        self.smooth3 = DepthwiseSeparableConv2d(outc, outc, kernel_size=3, stride=1, padding=1)
        self.smooth4 = DepthwiseSeparableConv2d(outc, outc, kernel_size=3, stride=1, padding=1)
        self.smooth5 = DepthwiseSeparableConv2d(outc, outc, kernel_size=3, stride=1, padding=1)

        # BN 用于横向层
        self.bn3 = nn.BatchNorm2d(outc)
        self.bn4 = nn.BatchNorm2d(outc)
        self.bn5 = nn.BatchNorm2d(outc)

        self.act = nn.ReLU(inplace=True)

        self._init_weights()

    def _init_weights(self):
        for m in [self.lateral3, self.lateral4, self.lateral5]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        for bn in [self.bn3, self.bn4, self.bn5]:
            nn.init.ones_(bn.weight)
            nn.init.zeros_(bn.bias)

    @staticmethod
    def _upsample_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        # 用 size 对齐，避免奇偶尺寸导致的 1 像素误差
        return F.interpolate(x, size=ref.shape[-2:], mode="nearest")

    def forward(self, c3: torch.Tensor, c4: torch.Tensor, c5: torch.Tensor):
        # 1) 横向降维
        l5 = self.act(self.bn5(self.lateral5(c5)))
        l4 = self.act(self.bn4(self.lateral4(c4)))
        l3 = self.act(self.bn3(self.lateral3(c3)))

        # 2) 自顶向下融合
        p5 = l5
        p4 = l4 + self._upsample_to(p5, l4)
        p3 = l3 + self._upsample_to(p4, l3)

        # 3) 平滑
        p5 = self.smooth5(p5)
        p4 = self.smooth4(p4)
        p3 = self.smooth3(p3)

        return [p3, p4, p5]