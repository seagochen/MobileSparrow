from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F # <-- 添加这行导入

from core.models.backbones.mobilenet_v2 import MobileNetV2Backbone
from core.models.backbones.shufflenet_v2 import ShuffleNetV2Backbone
# from core.models.necks.fpn_lite import FPNLite
from core.models.necks.panet_lite import PANetLite
from core.models.heads.movenet_head import MoveNetHead


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
        # self.neck = FPNLite(c3=c3c, c4=c4c, c5=c5c, outc=neck_outc)
        self.neck = PANetLite(c3=c3c, c4=c4c, c5=c5c, outc=neck_outc)

        # <新增 2025/09/11>
        # 我们需要一个最终的平滑层，就像在 FPNLite 中一样，
        # 用于处理上采样后的特征图。
        # self.final_smooth = nn.Conv2d(neck_outc, neck_outc, 3, 1, 1, bias=False)
        # self.final_bn = nn.BatchNorm2d(neck_outc)
        # self.final_relu = nn.ReLU(inplace=True)

        # 融合：concat(P3, up(P4), up(P5)) -> 1x1降维 -> 3x3平滑
        self.fuse_reduce = nn.Conv2d(neck_outc * 3, neck_outc, kernel_size=1, bias=False)
        self.fuse_reduce_bn = nn.BatchNorm2d(neck_outc)
        self.fuse_reduce_relu = nn.ReLU(inplace=True)

        self.fuse_conv = nn.Conv2d(neck_outc, neck_outc, kernel_size=3, padding=1, bias=False)
        self.fuse_conv_bn = nn.BatchNorm2d(neck_outc)
        self.fuse_conv_relu = nn.ReLU(inplace=True)

        # 最终上采样后的平滑（你原来就有）
        self.final_smooth = nn.Conv2d(neck_outc, neck_outc, 3, 1, 1, bias=False)
        self.final_bn = nn.BatchNorm2d(neck_outc)
        self.final_relu = nn.ReLU(inplace=True)
        # ----

        # 3) 头部
        self.head = MoveNetHead(neck_outc, num_joints=num_joints, midc=head_midc)

    def forward(self, x) -> Dict[str, torch.Tensor]:
        # <修改 2025/09/11> 下面内容
        # # 使用 mobilenet 或者 shufflenet 进行特征提取
        # c3, c4, c5 = self.backbone(x)
        # # p3 = self.neck(c3, c4, c5)
        # # return self.head(p3)
        #
        # # --- 修改这里的逻辑 ---
        # # PANetLite 返回一个包含3个特征图 (p3, p4, p5) 的元组
        # p3_out, _, _ = self.neck(c3, c4, c5)  # 我们只需要高分辨率的 p3_out
        #
        # # 和在 FPNLite 中一样，对 p3 进行上采样以获得期望的分辨率 (48x48)
        # p3_upsampled = F.interpolate(p3_out, scale_factor=2.0, mode='bilinear', align_corners=False)
        #
        # # 应用最终的平滑层
        # p3_final = self.final_relu(self.final_bn(self.final_smooth(p3_upsampled)))
        #
        # return self.head(p3_final)
        # # --- 修改结束 ---
        # 为：
        c3, c4, c5 = self.backbone(x)
        p3_out, p4_out, p5_out = self.neck(c3, c4, c5)  # (P3,P4,P5)

        H3, W3 = p3_out.shape[-2:]
        p4_up = F.interpolate(p4_out, size=(H3, W3), mode='bilinear', align_corners=False)
        p5_up = F.interpolate(p5_out, size=(H3, W3), mode='bilinear', align_corners=False)

        f = torch.cat([p3_out, p4_up, p5_up], dim=1)  # B x (3C) x H3 x W3
        f = self.fuse_reduce_relu(self.fuse_reduce_bn(self.fuse_reduce(f)))  # B x C x H3 x W3
        f = self.fuse_conv_relu(self.fuse_conv_bn(self.fuse_conv(f)))

        f = F.interpolate(f, scale_factor=2.0, mode='bilinear', align_corners=False)  # 到 48×48
        f = self.final_relu(self.final_bn(self.final_smooth(f)))
        return self.head(f)


if __name__ == "__main__":
    m = MoveNet(backbone="mobilenet_v2", num_joints=17, width_mult=1.0).eval()
    x = torch.randn(1,3,192,192)

    # 执行推论
    with torch.no_grad():
        y = m(x)

    # 输出并查看4头的形状
    for k,v in y.items():
        print(k, v.shape)
