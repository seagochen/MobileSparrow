import numpy as np
import torch.nn as nn

from sparrow.models.backbones.mobilenet_v2 import MobileNetV2Backbone
from sparrow.models.backbones.mobilenet_v3 import MobileNetV3Backbone
from sparrow.models.backbones.shufflenet_v2 import ShuffleNetV2Backbone
from sparrow.models.necks.fpn_lite_dets import FPNLiteDet
from sparrow.models.heads.db_head import DBHeadLite

BACKBONES = {
    "mobilenet_v2": MobileNetV2Backbone,
    "mobilenet_v3": MobileNetV3Backbone,
    "shufflenet_v2": ShuffleNetV2Backbone,
}

class OCRDetDB(nn.Module):
    """Lightweight text detector using FPN-Lite + DB head.
    Args:
        backbone: one of {'mobilenet_v2','mobilenet_v3','shufflenet_v2'}
        width_mult: channel scaling for backbone
        neck_outc: FPN output channels per level
        head_midc: conv mid channels in DB head
        use_fixed_thresh: if True, only outputs prob_map (use scalar threshold in postprocess)
    Returns:
        dict with:
          'features': [P3,P4,P5]
          'prob_map': (B,1,H/4,W/4)  (assuming P3 stride=4 wrt input after stem; exact factor depends on backbone)
          'thresh_map': (B,1,H/4,W/4) or None
    """
    def __init__(self, backbone: str = "mobilenet_v3", width_mult: float = 1.0,
                 neck_outc: int = 96, head_midc: int = 64, use_fixed_thresh: bool = True):
        super().__init__()
        assert backbone in BACKBONES, f"unknown backbone: {backbone}"
        self.backbone = BACKBONES[backbone](width_mult=width_mult)
        c3, c4, c5 = self.backbone.get_out_channels()
        self.neck = FPNLiteDet(c3, c4, c5, outc=neck_outc)
        self.head = DBHeadLite(neck_outc, head_midc, use_fixed_thresh=use_fixed_thresh)

    def forward(self, x):
        c3, c4, c5 = self.backbone(x)
        feats = self.neck(c3, c4, c5)  # [P3,P4,P5]
        p3 = feats[0]
        prob_map, thresh_map = self.head(p3)
        return {"features": feats, "prob_map": prob_map, "thresh_map": thresh_map}


def db_binarize(prob_map, thresh=0.3):
    """prob_map: (B,1,H,W) -> (B,H,W) uint8 mask in [0,255]."""
    pm = prob_map.detach().cpu().numpy()
    masks = (pm > float(thresh)).astype(np.uint8) * 255
    return masks
