# reidlite.py
import torch.nn as nn
import torch.nn.functional as F

from sparrow.models.backbones.mobilenet_v2 import MobileNetV2Backbone
from sparrow.models.backbones.mobilenet_v3 import MobileNetV3Backbone
from sparrow.models.backbones.shufflenet_v2 import ShuffleNetV2Backbone


BACKBONES = {
    "mobilenet_v2": MobileNetV2Backbone,
    "shufflenet_v2": ShuffleNetV2Backbone,
    "mobilenet_v3": MobileNetV3Backbone,
}


class ReIDLite(nn.Module):
    """
    输入: Bx3x256x128 (HxW=256x128的人框裁剪)
    输出: Bx128   (L2-normalized embedding)
    训练: 你可以额外加一个 classifier head 做 ArcFace / CE；推理时只用 embedding。
    """
    def __init__(self, backbone='mobilenet_v3', width_mult=0.5, emb_dim=128, freeze_stem=True):
        super().__init__()

        assert backbone in BACKBONES, f"unknown backbone: {backbone}"

        # 1) 构建 backbone，并把 width_mult 传进去
        self.backbone = BACKBONES[backbone](width_mult=width_mult)

        # 2) 取 C5 通道数作为 head 的输入维度
        c3, c4, c5 = self.backbone.get_out_channels()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.neck = BNNeck(c5, emb_dim)

        # 为防止backbone不包含stem
        if freeze_stem and hasattr(self.backbone, "stem"):
            for p in self.backbone.stem.parameters(): p.requires_grad = False

    def forward(self, x):
        # backbones 返回 (C3, C4, C5)
        _, _, c5 = self.backbone(x)
        feat = self.gap(c5).flatten(1)  # BxC
        emb = self.neck(feat)           # Bx128, 已 L2-norm
        return emb


class BNNeck(nn.Module):
    def __init__(self, in_dim, out_dim=128):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.bn = nn.BatchNorm1d(out_dim)
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_out')
        nn.init.constant_(self.bn.weight, 1.0)
        nn.init.constant_(self.bn.bias, 0.0)
    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = F.normalize(x, dim=1)   # L2-norm
        return x


