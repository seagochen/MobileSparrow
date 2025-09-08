import torch
import torch.nn as nn
from typing import Dict

from lib.models.backbones.mobilenet_v2 import MobileNetV2Backbone
from lib.models.backbones.shufflenet_v2 import ShuffleNetV2Backbone
from lib.models.necks.fpn_lite import FPNLite
from lib.models.heads.movenet_head import MoveNetHead

BACKBONES = {
    "mobilenet_v2": MobileNetV2Backbone,
    "shufflenet_v2": ShuffleNetV2Backbone,
}

class MoveNet(nn.Module):
    def __init__(self,
                 backbone: str = "mobilenet_v2",
                 num_joints: int = 17,
                 neck_outc: int = 64,
                 head_midc: int = 32,
                 width_mult: float = 1.0):
        super().__init__()
        assert backbone in BACKBONES, f"unknown backbone: {backbone}"

        # 1) 构建 backbone，并把 width_mult 传进去
        self.backbone = BACKBONES[backbone](width_mult=width_mult)

        # 2) 动态获取 C3/C4/C5 通道，配置 FPN
        c3c, c4c, c5c = self.backbone.get_out_channels()
        self.neck = FPNLite(c3=c3c, c4=c4c, c5=c5c, outc=neck_outc)

        # 3) 头部
        self.head = MoveNetHead(neck_outc, num_joints=num_joints, midc=head_midc)

    def forward(self, x) -> Dict[str, torch.Tensor]:
        c3, c4, c5 = self.backbone(x)
        p3 = self.neck(c3, c4, c5)
        return self.head(p3)

if __name__ == "__main__":
    m = MoveNet(backbone="mobilenet_v2", num_joints=17, width_mult=1.0).eval()
    x = torch.randn(1,3,192,192)
    with torch.no_grad():
        y = m(x)
    for k,v in y.items():
        print(k, v.shape)
