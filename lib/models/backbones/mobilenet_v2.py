import torch
import torch.nn as nn
from typing import Tuple, List

def conv_bn(inp, oup, k, s, p, g=1, act=True):
    layers = [nn.Conv2d(inp, oup, k, s, p, groups=g, bias=False),
              nn.BatchNorm2d(oup)]
    if act: layers += [nn.ReLU6(inplace=True)]
    return nn.Sequential(*layers)

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = stride == 1 and inp == oup
        layers = []
        if expand_ratio != 1:
            layers.append(conv_bn(inp, hidden_dim, 1, 1, 0))
        layers.extend([
            conv_bn(hidden_dim, hidden_dim, 3, stride, 1, g=hidden_dim),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.use_res_connect else self.conv(x)

class MobileNetV2Backbone(nn.Module):
    """
    输出三个层级特征 (C3, C4, C5)，供 FPN 使用。
    支持 width_mult：按比例缩放各层通道。
    """
    def __init__(self, in_ch: int = 3, width_mult: float = 1.0):
        super().__init__()
        self.width_mult = float(width_mult)

        # t=expand_ratio, c=输出通道(基准), n=重复次数, s=stride
        cfg = [
            [1,  16, 1, 1],
            [6,  24, 2, 2],  # -> C2
            [6,  32, 3, 2],  # -> C3
            [6,  64, 4, 2],  # -> C4
            [6,  96, 3, 1],
            [6, 160, 3, 2],  # -> C5
            [6, 320, 1, 1],
        ]

        out_ch = self._round_ch(32)
        self.stem = conv_bn(in_ch, out_ch, 3, 2, 1)

        layers = []
        input_channel = out_ch
        self.c3_idx = self.c4_idx = self.c5_idx = None
        c3c = c4c = c5c = None
        stage = 2
        for t, c, n, s in cfg:
            output_channel = self._round_ch(c)
            blocks = []
            for i in range(n):
                stride = s if i == 0 else 1
                blocks.append(InvertedResidual(input_channel, output_channel, stride, t))
                input_channel = output_channel
            layers.append(nn.Sequential(*blocks))
            if s == 2:
                stage += 1
                if stage == 3: self.c3_idx, c3c = len(layers)-1, output_channel
                if stage == 4: self.c4_idx, c4c = len(layers)-1, output_channel
                if stage == 5: self.c5_idx, c5c = len(layers)-1, output_channel

        self.features = nn.ModuleList(layers)
        # 记录各层通道数，供上层(FPN)查询
        self._out_channels = (c3c, c4c, c5c)

    def _round_ch(self, c: int) -> int:
        return max(8, int(c * self.width_mult + 0.5))

    def get_out_channels(self) -> Tuple[int, int, int]:
        """返回 (C3, C4, C5) 的通道数"""
        return self._out_channels

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        c3 = c4 = c5 = None
        for i, m in enumerate(self.features):
            x = m(x)
            if i == self.c3_idx: c3 = x
            if i == self.c4_idx: c4 = x
            if i == self.c5_idx: c5 = x
        return c3, c4, c5
