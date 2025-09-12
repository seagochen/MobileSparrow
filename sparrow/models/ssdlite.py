from typing import Dict, List, Tuple
import torch
import torch.nn as nn

from sparrow.models.backbones.mobilenet_v2 import MobileNetV2Backbone
from sparrow.models.backbones.shufflenet_v2 import ShuffleNetV2Backbone
from sparrow.models.necks.fpn_lite_det import FPNLiteDet
from sparrow.models.heads.ssd_head import SSDLiteHead

BACKBONES = {
    "mobilenet_v2": MobileNetV2Backbone,
    "shufflenet_v2": ShuffleNetV2Backbone,
}

class SSDLite(nn.Module):
    """
    轻量 SSDLite 检测模型骨架（不含训练/损失/解码）：
      - backbone: MobileNetV2 或 ShuffleNetV2
      - neck: FPNLiteDet -> [P3,P4,P5]
      - head: 每层一个 SSDLiteHead
    """
    def __init__(self,
                 num_classes: int,
                 backbone: str = "mobilenet_v2",
                 width_mult: float = 1.0,
                 neck_outc: int = 96,              # 建议略大于关键点任务, 64/96/128 皆可
                 anchor_ratios: Tuple[float, ...] = (1.0, 2.0, 0.5),
                 anchor_scales: Tuple[float, ...] = (1.0, 1.26),  # 每层2个scale作为示例
                 ):
        super().__init__()
        assert backbone in BACKBONES, f"unknown backbone: {backbone}"
        self.num_classes = int(num_classes)

        # 1) Backbone
        self.backbone = BACKBONES[backbone](width_mult=width_mult)
        c3c, c4c, c5c = self.backbone.get_out_channels()

        # 2) Neck (多尺度)
        self.neck = FPNLiteDet(c3=c3c, c4=c4c, c5=c5c, outc=neck_outc)

        # 3) Heads for each pyramid level
        #    anchors/位置数 A = len(ratios) * len(scales)
        self.ratios = anchor_ratios
        self.scales = anchor_scales
        A = len(self.ratios) * len(self.scales)

        self.heads = nn.ModuleList([
            SSDLiteHead(neck_outc, num_anchors=A, num_classes=self.num_classes),  # P3
            SSDLiteHead(neck_outc, num_anchors=A, num_classes=self.num_classes),  # P4
            SSDLiteHead(neck_outc, num_anchors=A, num_classes=self.num_classes),  # P5
        ])

    def forward(self, x) -> Dict[str, List[torch.Tensor]]:
        # Backbone -> C3,C4,C5
        c3, c4, c5 = self.backbone(x)

        # FPN -> [P3,P4,P5]
        feats = self.neck(c3, c4, c5)  # list of 3 tensors

        # Multi-level heads
        cls_list, reg_list = [], []
        for f, head in zip(feats, self.heads):
            cls, reg = head(f)  # [B, H*W*A, C], [B, H*W*A, 4]
            cls_list.append(cls)
            reg_list.append(reg)

        return {
            "cls_logits": cls_list,
            "bbox_regs":  reg_list,
            # 这里暂不返回 anchors（下一步我们再把 anchor 发生器接上）
        }
