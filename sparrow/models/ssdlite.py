from typing import Dict, List, Tuple

import torch
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
                 anchor_strides: Tuple[float, ...] = (8, 16, 32)
                 ):
        super().__init__()
        assert backbone in BACKBONES, f"unknown backbone: {backbone}"
        self.num_classes = int(num_classes)
        self.strides = anchor_strides

        # 1) Backbone
        self.backbone = BACKBONES[backbone](width_mult=width_mult)
        c3c, c4c, c5c = self.backbone.get_out_channels()

        # 2) Neck (多尺度)
        self.neck = FPNLite(c3=c3c, c4=c4c, c5=c5c, outc=neck_outc)

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
        feats = self.neck(c3, c4, c5)  # 输出 p3, p4, p5 三个尺度的特征数据

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


class DWSeparable(nn.Module):
    """Depthwise-Separable Conv (SSDLite 风格)"""
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, k, s, p, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.bn1(self.dw(x)))
        x = self.act(self.bn2(self.pw(x)))
        return x


class SSDLiteHead(nn.Module):
    """
    单层特征图的 SSDLite 检测头：
      - 输入：一个特征图 (B, C, H, W)
      - 输出：cls_logits (B, H*W*A, num_classes), bbox_regs (B, H*W*A, 4)
    其中 A = 每个位置的锚框数量（由外部配置 scales × ratios 决定）
    """
    def __init__(self, in_ch: int, num_anchors: int, num_classes: int, midc: int = None):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        midc = midc or max(32, in_ch // 2)

        # SSDLite 两条支路：分类 & 回归（均用DW可分离卷积）
        self.cls_conv = DWSeparable(in_ch, midc, 3, 1, 1)
        self.reg_conv = DWSeparable(in_ch, midc, 3, 1, 1)

        self.cls_pred = nn.Conv2d(midc, num_anchors * num_classes, 1)
        self.reg_pred = nn.Conv2d(midc, num_anchors * 4, 1)

        # 初始化
        for m in [self.cls_pred, self.reg_pred]:
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        B, _, H, W = x.shape
        cls_feat = self.cls_conv(x)
        reg_feat = self.reg_conv(x)

        cls = self.cls_pred(cls_feat)  # [B, A*C, H, W]
        reg = self.reg_pred(reg_feat)  # [B, A*4, H, W]

        # 重排到 [B, H*W*A, C] & [B, H*W*A, 4]
        cls = cls.permute(0, 2, 3, 1).contiguous().view(B, H * W * self.num_anchors, self.num_classes)
        reg = reg.permute(0, 2, 3, 1).contiguous().view(B, H * W * self.num_anchors, 4)
        return cls, reg



class FPNLite(nn.Module):
    """
    输入: C3, C4, C5
    输出: [P3, P4, P5]   (默认通道一致 outc, 尺度分别约为 1/8, 1/16, 1/32)
    这里直接用 nn.Sequential 定义 1x1 与 3x3 block（Conv + BN + ReLU）。
    """
    def __init__(self, c3, c4, c5, outc=64):
        super().__init__()
        # 1x1 降维/对齐
        self.l3 = nn.Sequential(
            nn.Conv2d(c3, outc, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True),
        )
        self.l4 = nn.Sequential(
            nn.Conv2d(c4, outc, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True),
        )
        self.l5 = nn.Sequential(
            nn.Conv2d(c5, outc, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True),
        )

        # 3x3 平滑
        self.smooth3 = nn.Sequential(
            nn.Conv2d(outc, outc, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True),
        )
        self.smooth4 = nn.Sequential(
            nn.Conv2d(outc, outc, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True),
        )
        self.smooth5 = nn.Sequential(
            nn.Conv2d(outc, outc, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True),
        )

    def forward(self, c3, c4, c5):
        p5 = self.l5(c5)
        p4 = self.l4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.l3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")

        p5 = self.smooth5(p5)
        p4 = self.smooth4(p4)
        p3 = self.smooth3(p3)
        return [p3, p4, p5]

