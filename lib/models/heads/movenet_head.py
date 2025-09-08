import torch
import torch.nn as nn

class MoveNetHead(nn.Module):
    """
    四头输出：heatmaps / centers / regs / offsets
    """
    def __init__(self, in_ch, num_joints=17, midc=32, act="sigmoid"):
        super().__init__()
        Act = nn.Sigmoid if act == "sigmoid" else nn.Sigmoid  # 预留
        self.hm = nn.Sequential(nn.Conv2d(in_ch, midc, 3, 1, 1), nn.ReLU(inplace=True),
                                nn.Conv2d(midc, num_joints, 1), Act())
        self.ct = nn.Sequential(nn.Conv2d(in_ch, midc, 3, 1, 1), nn.ReLU(inplace=True),
                                nn.Conv2d(midc, 1, 1), Act())
        self.reg = nn.Sequential(nn.Conv2d(in_ch, midc, 3, 1, 1), nn.ReLU(inplace=True),
                                 nn.Conv2d(midc, num_joints*2, 1))     # (dx, dy)
        self.off = nn.Sequential(nn.Conv2d(in_ch, midc, 3, 1, 1), nn.ReLU(inplace=True),
                                 nn.Conv2d(midc, num_joints*2, 1))     # (ox, oy)

    def forward(self, x):
        return {
            "heatmaps": self.hm(x),
            "centers":  self.ct(x),
            "regs":     self.reg(x),
            "offsets":  self.off(x),
        }
