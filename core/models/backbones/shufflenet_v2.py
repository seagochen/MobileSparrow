import torch
import torch.nn as nn
from typing import Tuple


def channel_shuffle(x, groups: int):
    b, c, h, w = x.size()
    x = x.view(b, groups, c // groups, h, w).transpose(1, 2).contiguous()
    return x.view(b, c, h, w)

class ShuffleUnit(nn.Module):
    def __init__(self, inp, oup, stride):
        super().__init__()
        self.stride = stride
        mid = oup // 2
        if stride == 1:
            self.branch2 = nn.Sequential(
                nn.Conv2d(mid, mid, 1, 1, 0, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
                nn.Conv2d(mid, mid, 3, 1, 1, groups=mid, bias=False), nn.BatchNorm2d(mid),
                nn.Conv2d(mid, mid, 1, 1, 0, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False), nn.BatchNorm2d(inp),
                nn.Conv2d(inp, mid, 1, 1, 0, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            )
            self.branch2 = nn.Sequential(
                nn.Conv2d(inp, mid, 1, 1, 0, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
                nn.Conv2d(mid, mid, 3, stride, 1, groups=mid, bias=False), nn.BatchNorm2d(mid),
                nn.Conv2d(mid, mid, 1, 1, 0, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), 1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), 1)
        return channel_shuffle(out, 2)

class ShuffleNetV2Backbone(nn.Module):
    """
    产出 (C3, C4, C5)，兼容 FPN。
    支持 width_mult 的常见 preset；也可直接传自定义 stages_out。
    """
    _PRESETS = {
        0.5: (48, 96, 192),     # 论文/常用实现的通道配置
        1.0: (116, 232, 464),
        1.5: (176, 352, 704),
        2.0: (244, 488, 976),
    }

    def __init__(self, in_ch=3, width_mult: float = 1.0,
                 stages_out: Tuple[int, int, int] = None, repeats=(4, 8, 4)):
        super().__init__()
        self.width_mult = float(width_mult)

        if stages_out is None:
            # 取最接近的 preset（避免 0.75 之类非标准值报错）
            key = min(self._PRESETS.keys(), key=lambda k: abs(k - self.width_mult))
            stages_out = self._PRESETS[key]
        self._out_channels = (stages_out[0], stages_out[1], stages_out[2])

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 24, 3, 2, 1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )
        inp = 24
        self.stages = nn.ModuleList()
        for outp, rep in zip(stages_out, repeats):
            blocks = [ShuffleUnit(inp, outp, stride=2)]
            inp = outp
            for _ in range(rep-1):
                blocks.append(ShuffleUnit(inp, outp, stride=1))
            self.stages.append(nn.Sequential(*blocks))

    def get_out_channels(self) -> Tuple[int, int, int]:
        """返回 (C3, C4, C5) 的通道数"""
        # 这里的 C5 我们沿用第三个 stage 的通道数（下游 FPN 用同一通道处理）
        c3, c4, c5 = self._out_channels[0], self._out_channels[1], self._out_channels[2]
        return (c3, c4, c5)

    def forward(self, x):
        x = self.stem(x)           # 1/2
        c2 = self.stages[0](x)     # 1/4
        c3 = self.stages[1](c2)    # 1/8
        c4 = self.stages[2](c3)    # 1/16
        c5 = c4                     # 这里直接把 C4 当作 C5（如需更深层可再加一段）
        return c3, c4, c5
