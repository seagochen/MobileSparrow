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
        # 修改：增加第四个值，代表 C5 的输出通道
        0.5: (48, 96, 192, 384),
        1.0: (116, 232, 464, 928),
        1.5: (176, 352, 704, 1408),
        2.0: (244, 488, 976, 1952),
    }

    def __init__(self, in_ch=3, width_mult: float = 1.0,
                 stages_out: Tuple[int, int, int, int] = None, # 修改：现在接收4个值
                 repeats=(4, 8, 4, 4)): # 修改：为新的stage增加重复次数，例如4次
        super().__init__()
        self.width_mult = float(width_mult)

        if stages_out is None:
            # 取最接近的 preset（避免 0.75 之类非标准值报错）
            key = min(self._PRESETS.keys(), key=lambda k: abs(k - self.width_mult))
            stages_out = self._PRESETS[key]

        # 修改：现在我们有4个输出阶段
        self._out_channels = (stages_out[0], stages_out[1], stages_out[2], stages_out[3])

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
        # 修改：返回 stage-1, stage-2, stage-3 的输出通道
        # C2(stage-0) -> C3(stage-1) -> C4(stage-2) -> C5(stage-3)
        c3, c4, c5 = self._out_channels[1], self._out_channels[2], self._out_channels[3]
        return (c3, c4, c5)

    def forward(self, x):
        x = self.stem(x)           # stride: 2,  out: 24
        c2 = self.stages[0](x)     # stride: 4
        c3 = self.stages[1](c2)    # stride: 8  -> FPN C3
        c4 = self.stages[2](c3)    # stride: 16 -> FPN C4
        c5 = self.stages[3](c4)    # stride: 32 -> FPN C5  <-- 新增
        return c3, c4, c5
