
import torch
import torch.nn as nn


class DBHeadLite(nn.Module):
    """A minimal Differentiable Binarization head (lite).
    Outputs:
      - prob_map: text probability (B,1,H,W)
      - thresh_map: threshold map (B,1,H,W)  (optional; can be None if use_fixed_thresh=True)
    """
    def __init__(self, in_ch: int, mid_ch: int = 64, use_fixed_thresh: bool = True):
        super().__init__()
        self.use_fixed_thresh = bool(use_fixed_thresh)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, 1, 1, bias=False), nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, 1, 1, bias=False), nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True),
        )
        self.prob = nn.Conv2d(mid_ch, 1, 1, 1, 0)
        if not self.use_fixed_thresh:
            self.thresh = nn.Conv2d(mid_ch, 1, 1, 1, 0)
        else:
            self.register_parameter('thresh', None)

    def forward(self, x):
        y = self.conv(x)
        prob_map = torch.sigmoid(self.prob(y))
        if self.use_fixed_thresh:
            return prob_map, None
        else:
            thresh_map = torch.sigmoid(self.thresh(y))
            return prob_map, thresh_map
