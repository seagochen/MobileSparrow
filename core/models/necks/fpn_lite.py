import torch.nn as nn


def conv1x1(c_in, c_out): return nn.Sequential(nn.Conv2d(c_in, c_out, 1, 1, 0, bias=False),
                                               nn.BatchNorm2d(c_out),
                                               nn.ReLU(inplace=True))
def conv3x3(c_in, c_out): return nn.Sequential(nn.Conv2d(c_in, c_out, 3, 1, 1, bias=False),
                                               nn.BatchNorm2d(c_out),
                                               nn.ReLU(inplace=True))

class FPNLite(nn.Module):
    """
    接收 (C3,C4,C5) -> 输出单尺度 P3（或多尺度列表，按需改）
    """
    def __init__(self, c3, c4, c5, outc=64):
        super().__init__()
        self.l3 = conv1x1(c3, outc)
        self.l4 = conv1x1(c4, outc)
        self.l5 = conv1x1(c5, outc)
        self.smooth3 = conv3x3(outc, outc)
        self.smooth4 = conv3x3(outc, outc)

    def forward(self, c3, c4, c5):
        p5 = self.l5(c5)
        p4 = self.l4(c4) + nn.functional.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.l3(c3) + nn.functional.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p4 = self.smooth4(p4); p3 = self.smooth3(p3)
        return p3  # 你也可以返回 [p3, p4, p5]
