import torch.nn as nn
import torch.nn.functional as F

class SequenceNeck1D(nn.Module):
    """Convert 2D feature map into a 1D temporal sequence for CTC/attention heads.
    Strategy: small conv tower -> avg-pool along height to 1 -> permute to (B,T,C).
    """
    def __init__(self, in_ch: int, mid_ch: int = 128, out_ch: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, 1, 1, bias=False), nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, 1, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):  # x: (B,C,H,W)
        x = self.conv(x)
        x = F.adaptive_avg_pool2d(x, (1, None))   # (B,C,1,W')
        x = x.squeeze(2).permute(0, 2, 1).contiguous()  # (B,T,C)
        return x
