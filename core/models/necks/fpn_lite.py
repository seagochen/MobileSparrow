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
        
        # --- 新增代码 ---
        # 增加一个最终的平滑层，用于上采样后的特征图
        self.final_smooth = conv3x3(outc, outc)
        # --- 新增代码结束 ---

    def forward(self, c3, c4, c5):
        p5 = self.l5(c5)
        p4 = self.l4(c4) + nn.functional.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.l3(c3) + nn.functional.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p4 = self.smooth4(p4)
        p3 = self.smooth3(p3)
        
        # --- 修改返回逻辑 ---
        # 对 p3 进行 2 倍上采样，使其分辨率从 24x24 提升到 48x48
        p3_upsampled = nn.functional.interpolate(p3, scale_factor=2.0, mode='bilinear', align_corners=False)
        # 通过平滑层
        p3_final = self.final_smooth(p3_upsampled)
        
        return p3_final # 返回 48x48 的特征图
        # --- 修改结束 ---