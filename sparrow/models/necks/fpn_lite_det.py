# 新文件（或放回 fpn_lite.py 中另起一个类）：fpn_lite_det.py
import torch.nn as nn
import torch.nn.functional as F
from sparrow.models import conv_utils

class FPNLiteDet(nn.Module):
    """
    输入: C3, C4, C5
    输出: [P3, P4, P5]   (默认通道一致 outc, 尺度分别约为 1/8, 1/16, 1/32)
    """
    def __init__(self, c3, c4, c5, outc=64):
        super().__init__()
        self.l3 = conv_utils.conv1x1(c3, outc)
        self.l4 = conv_utils.conv1x1(c4, outc)
        self.l5 = conv_utils.conv1x1(c5, outc)
        self.smooth3 = conv_utils.conv3x3(outc, outc)
        self.smooth4 = conv_utils.conv3x3(outc, outc)
        self.smooth5 = conv_utils.conv3x3(outc, outc)

    def forward(self, c3, c4, c5):
        p5 = self.l5(c5)
        p4 = self.l4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.l3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p5 = self.smooth5(p5)
        p4 = self.smooth4(p4)
        p3 = self.smooth3(p3)
        return [p3, p4, p5]
