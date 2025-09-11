import torch.nn as nn
import torch.nn.functional as F
from core.models import conv_utils


class PANetLite(nn.Module):
    """
    接收 (C3, C4, C5) -> 输出多尺度特征图 (P3, P4, P5)
    """

    def __init__(self, c3, c4, c5, outc=96):
        super().__init__()
        # --- FPN Top-Down Path ---
        self.l3 = conv_utils.conv1x1(c3, outc)
        self.l4 = conv_utils.conv1x1(c4, outc)
        self.l5 = conv_utils.conv1x1(c5, outc)
        self.smooth_topdown_p3 = conv_utils.conv3x3(outc, outc)
        self.smooth_topdown_p4 = conv_utils.conv3x3(outc, outc)

        # --- PANet Bottom-Up Path ---
        self.downsample_p3 = nn.Sequential(
            nn.Conv2d(outc, outc, 3, 2, 1, bias=False),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True)
        )
        self.downsample_p4 = nn.Sequential(
            nn.Conv2d(outc, outc, 3, 2, 1, bias=False),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True)
        )
        self.smooth_bottomup_p4 = conv_utils.conv3x3(outc, outc)
        self.smooth_bottomup_p5 = conv_utils.conv3x3(outc, outc)

    def forward(self, c3, c4, c5):
        # --- Top-Down Pathway (FPN) ---
        p5_td = self.l5(c5)
        p4_td = self.l4(c4) + F.interpolate(p5_td, size=c4.shape[-2:], mode="nearest")
        p3_td = self.l3(c3) + F.interpolate(p4_td, size=c3.shape[-2:], mode="nearest")

        p4_td = self.smooth_topdown_p4(p4_td)
        p3_td = self.smooth_topdown_p3(p3_td)

        # --- Bottom-Up Pathway (PANet) ---
        p3_out = p3_td  # P3的最终输出就是自顶向下路径的结果
        p4_out = self.smooth_bottomup_p4(p4_td + self.downsample_p3(p3_out))
        p5_out = self.smooth_bottomup_p5(p5_td + self.downsample_p4(p4_out))

        return p3_out, p4_out, p5_out