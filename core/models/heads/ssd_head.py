import torch
import torch.nn as nn
import torch.nn.functional as F


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
