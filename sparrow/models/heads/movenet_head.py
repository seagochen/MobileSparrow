import torch.nn as nn


class MoveNetHead(nn.Module):
    """
    四头输出：heatmaps / centers / regs / offsets
    """
    def __init__(self, in_ch, num_joints=17, midc=32):
        super().__init__()

        self.hm = nn.Sequential(nn.Conv2d(in_ch, midc, 3, 1, 1), nn.ReLU(inplace=True),
                                nn.Conv2d(midc, num_joints, 1), nn.Sigmoid())
        self.ct = nn.Sequential(nn.Conv2d(in_ch, midc, 3, 1, 1), nn.ReLU(inplace=True),
                                nn.Conv2d(midc, 1, 1), nn.Sigmoid())
        self.reg = nn.Sequential(nn.Conv2d(in_ch, midc, 3, 1, 1), nn.ReLU(inplace=True),
                                 nn.Conv2d(midc, num_joints*2, 1))     # (dx, dy)
        self.off = nn.Sequential(nn.Conv2d(in_ch, midc, 3, 1, 1), nn.ReLU(inplace=True),
                                 nn.Conv2d(midc, num_joints*2, 1))     # (ox, oy)

    def forward(self, x):
        return {
            "heatmaps": self.hm(x),     # B×17×48×48,
            "centers":  self.ct(x),     # B×1 ×48×48,
            "regs":     self.reg(x),    # B×34×48×48,
            "offsets":  self.off(x),    # B×34×48×48
        }
