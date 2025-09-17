import torch
import torch.nn as nn
from typing import Tuple

class HSigmoid(nn.Module):
    def forward(self, x):
        return torch.nn.functional.relu6(x + 3., inplace=True) / 6.

class HSwish(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = HSigmoid()
    def forward(self, x):
        return x * self.act(x)

class SqueezeExcite(nn.Module):
    def __init__(self, in_ch: int, se_ratio: float = 0.25):
        super().__init__()
        mid = max(8, int(in_ch * se_ratio))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_ch, mid, 1, 1, 0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(mid, in_ch, 1, 1, 0)
        self.hsig = HSigmoid()
    def forward(self, x):
        s = self.pool(x)
        s = self.relu(self.fc1(s))
        s = self.hsig(self.fc2(s))
        return x * s

def conv_bn_act(c_in, c_out, k=3, s=1, p=1, act='hswish'):
    layers = [nn.Conv2d(c_in, c_out, k, s, p, bias=False), nn.BatchNorm2d(c_out)]
    if act == 'relu':
        layers.append(nn.ReLU(inplace=True))
    elif act == 'hswish':
        layers.append(HSwish())
    elif act is None:
        pass
    else:
        raise ValueError(f'unknown act: {act}')
    return nn.Sequential(*layers)

def make_divisible(v: float, divisor: int = 8, min_value: int = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return int(new_v)

class MBConvV3(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, exp: int, stride: int, use_se: bool, nl: str):
        super().__init__()
        self.use_res = (stride == 1 and in_ch == out_ch)
        act = 'relu' if nl == 'relu' else 'hswish'

        mid = exp
        layers = []
        if mid != in_ch:
            layers.append(conv_bn_act(in_ch, mid, k=1, s=1, p=0, act=act))
        layers.append(nn.Conv2d(mid, mid, k, stride, k//2, groups=mid, bias=False))
        layers.append(nn.BatchNorm2d(mid))
        layers.append(nn.ReLU(inplace=True) if nl == 'relu' else HSwish())
        if use_se:
            layers.append(SqueezeExcite(mid))
        layers.append(nn.Conv2d(mid, out_ch, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(out_ch))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        y = self.block(x)
        return x + y if self.use_res else y


class MobileNetV3Backbone(nn.Module):
    """
    MobileNetV3-Large 风格骨干，输出 (C3, C4, C5) 供 FPN 使用。
    - 默认 width_mult=1.0；可在 0.35 ~ 1.25 间调整。
    - 返回的 C3/C4/C5 对应约 1/8, 1/16, 1/32 下采样的特征图。
    """
    def __init__(self, in_ch: int = 3, width_mult: float = 1.0, divisor: int = 8):
        super().__init__()
        self.width_mult = float(width_mult)
        self.divisor = int(divisor)

        def C(x):
            return make_divisible(x * self.width_mult, self.divisor)

        self.stem = conv_bn_act(in_ch, C(16), k=3, s=2, p=1, act='hswish')

        cfg = [
            (3,  16,  16, False, 'relu',   1, 1),
            (3,  64,  24, False, 'relu',   2, 1),
            (3,  72,  24, False, 'relu',   1, 1),
            (5,  72,  40, True,  'relu',   2, 1),
            (5, 120,  40, True,  'relu',   1, 2),
            (3, 240,  80, False, 'hswish', 2, 1),
            (3, 200,  80, False, 'hswish', 1, 3),
            (3, 480, 112, True,  'hswish', 1, 1),
            (3, 672, 112, True,  'hswish', 1, 1),
            (5, 672, 160, True,  'hswish', 2, 1),
            (5, 960, 160, True,  'hswish', 1, 2),
        ]

        in_c = C(16)
        self.c3_idx = self.c4_idx = self.c5_idx = None
        c3c = c4c = c5c = None
        down_count = 1  # after stem: /2

        stage_blocks = []
        for (k, exp, c, se, nl, s, n) in cfg:
            exp_c = C(exp)
            out_c = C(c)
            blocks = []
            for i in range(n):
                stride = s if i == 0 else 1
                blocks.append(MBConvV3(in_c, out_c, k=k, exp=exp_c, stride=stride, use_se=se, nl=nl))
                in_c = out_c
            stage_blocks.append((blocks, s, out_c))

        self.features = nn.ModuleList()
        for blocks, s, out_c in stage_blocks:
            if s == 2:
                down_count += 1
                if down_count == 3 and self.c3_idx is None:
                    self.c3_idx, c3c = len(self.features), out_c
                elif down_count == 4 and self.c4_idx is None:
                    self.c4_idx, c4c = len(self.features), out_c
                elif down_count == 5 and self.c5_idx is None:
                    self.c5_idx, c5c = len(self.features), out_c

            self.features.append(nn.Sequential(*blocks))

        if any(x is None for x in (self.c3_idx, self.c4_idx, self.c5_idx)):
            n = len(self.features)
            self.c3_idx, self.c4_idx, self.c5_idx = n-3, n-2, n-1
            c3c = c3c or stage_blocks[-3][2]
            c4c = c4c or stage_blocks[-2][2]
            c5c = c5c or stage_blocks[-1][2]

        self._out_channels = (c3c, c4c, c5c)

    def get_out_channels(self) -> Tuple[int, int, int]:
        return self._out_channels

    def forward(self, x):
        x = self.stem(x)
        c3 = c4 = c5 = None
        for i, m in enumerate(self.features):
            x = m(x)
            if i == self.c3_idx: c3 = x
            if i == self.c4_idx: c4 = x
            if i == self.c5_idx: c5 = x
        return c3, c4, c5
