# %% [markdown]
# # 构建一个MoveNet姿态检测模型
# 
# FPN + MobileNet 作为骨架，可以实现很多CV任务，本次我们将实现一个高精度的轻量级的姿态检测模型—— `MoveNet`。
# 
# 与 `003_SSDLite` 相似，本次我们只开发一个新的检测头，然后接入已有的架构。

# %% [markdown]
# ## 复用FPN Neck模块

# %%
import torch
import timm
import torch.nn as nn
from typing import Tuple
import torch.nn.functional as F


class FPN(nn.Module):
    """
    Feature Pyramid Network (FPN) Neck
    对 Backbone 输出的 (C3, C4, C5) 特征进行融合，生成 (P3, P4, P5)。
    """
    def __init__(self, in_channels: Tuple[int, int, int], out_channels: int = 256):
        super().__init__()
        c3_in, c4_in, c5_in = in_channels
        
        # 1. 侧向连接的 1x1 卷积，用于统一通道数
        self.lateral_conv3 = nn.Conv2d(c3_in, out_channels, kernel_size=1)
        self.lateral_conv4 = nn.Conv2d(c4_in, out_channels, kernel_size=1)
        self.lateral_conv5 = nn.Conv2d(c5_in, out_channels, kernel_size=1)
        
        # 2. 输出处理的 3x3 卷积，用于平滑特征
        self.output_conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.output_conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.output_conv5 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c3, c4, c5 = features
        
        # 自顶向下路径
        p5_in = self.lateral_conv5(c5)
        
        p4_in = self.lateral_conv4(c4)
        p5_up = F.interpolate(p5_in, size=p4_in.shape[-2:], mode='nearest')
        p4_in = p4_in + p5_up
        
        p3_in = self.lateral_conv3(c3)
        p4_up = F.interpolate(p4_in, size=p3_in.shape[-2:], mode='nearest')
        p3_in = p3_in + p4_up
        
        # 输出卷积
        p3 = self.output_conv3(p3_in)
        p4 = self.output_conv4(p4_in)
        p5 = self.output_conv5(p5_in)
        
        return p3, p4, p5

# %% [markdown]
# ## 构建MoveNet检测头
# 
# 

# %%
# 单人版 MoveNet 头部：只输出 heatmaps(K) + offsets(2K)
from typing import Dict
import torch
import torch.nn as nn

class SinglePoseHead(nn.Module):
    """
    单人姿态估计头（SSDLite 已提供人框；本头只负责框内关键点定位）
    输出：
      - heatmaps: [B, K, H, W]   关键点热图（logits）
      - offsets : [B, 2K, H, W]  关键点亚像素偏移（每点 x,y）
    说明：
      - 不包含 centers / regs 分支（多人场景才需要）
      - 训练时 heatmaps 用 BCEWithLogitsLoss / focal-variant；offsets 用 L1/SmoothL1
    """
    def __init__(self, in_ch: int, num_joints: int = 17, midc: int = 32):
        super().__init__()

        def dw_pw(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, in_c, 3, padding=1, groups=in_c, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_c, out_c, 1, bias=True)
            )

        self.hm_tower  = dw_pw(in_ch, midc)
        self.off_tower = dw_pw(in_ch, midc)

        self.hm  = nn.Conv2d(midc, num_joints, 1)
        self.off = nn.Conv2d(midc, num_joints * 2, 1)

        # 让热图初始更稀疏，便于收敛
        nn.init.constant_(self.hm.bias, -2.0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        heatmaps = self.hm(self.hm_tower(x))
        offsets  = self.off(self.off_tower(x))
        return {"heatmaps": heatmaps, "offsets": offsets}

# %% [markdown]
# ## 构建MoveNet模型
# 
# ### 设计说明
# 
# * **输入**：来自 SSDLite 的人框裁剪/对齐图（例如 192×192）。
# * **输出**（单人）：
# 
#   * `heatmaps`: `[B, K, H, W]` —— 关键点热图（logits）
#   * `offsets` : `[B, 2K, H, W]` —— 亚像素偏移（x,y 各一通道）
# * **去掉**：
# 
#   * `centers`（不需要找多人中心）
#   * `regs(2K)`（不需要中心→全骨架的大位移）
# 
# > 训练：`heatmaps` 用 BCEWithLogitsLoss（或 focal-variant），`offsets` 用 L1/SmoothL1，仅在关键点邻域的正样/ignore mask 上计算。
# > 推理：对 `heatmaps.sigmoid()` 取每通道峰值 → 再叠加 `offsets` 精修坐标。
# 
# * **本模型为单人姿态估计头**（与 SSDLite 人体检测联动：先检出人框 → 裁剪/对齐 → 本模型输出关键点）。
# * **输出**：
# 
#   * `heatmaps [B,K,H,W]`：每个关键点的热图（logits）。
#   * `offsets  [B,2K,H,W]`：关键点的亚像素偏移（x,y）。
# * **不输出**：`centers`、`regs`（这些多用于多人中心分组范式）。
# * **解码**：对 `heatmaps.sigmoid()` 每通道取峰值（可 NMS/Top-1），在对应网格处叠加 `offsets` 得到精细坐标；再按骨架拓扑连线。
# * **训练**：`heatmaps` 用 BCEWithLogitsLoss / focal（正样为高斯区域），`offsets` 用 L1/SmoothL1（只在关键点邻域计算；缺失关键点忽略）。
# * **分辨率**：默认输出 stride=8（例如 24×24），可选 `upsample_to_quarter=True` 输出 stride=4（48×48）提高高 IoU 精度。
# 
# ### 小贴士（和 SSDLite 配合）
# 
# * 在 SSDLite 的人框里做 **仿射对齐**（保持纵横比，pad 到正方形再 resize 到 192×192），并保留 **反变换矩阵**，把网格坐标还原到原图。
# * 如果一张图里有人数 >1，直接对每个框跑一遍**单人头**即可；**不需要** `centers`/`regs` 也能“多 人”——只是从**多框**的角度解决。
# * 推理阈值：`heatmaps.sigmoid() > t`（如 0.2～0.4），或 Top-1 直接取峰值（单人通常足够）。

# %%
import torch.nn.functional as F


class MoveNet_FPN(nn.Module):
    """
    单人版 MoveNet：FPN 取 P3（stride=8）作为头部输入，适配单人裁剪。
    参数：
      - backbone: timm features_only backbone（输出 C3,C4,C5）
      - num_joints: 关键点数（默认 17）
      - fpn_out_channels: FPN 输出通道
      - head_midc: 头部中间通道
      - upsample_to_quarter: 若为 True，将 P3 上采到 1/4（更高精度，H/4×W/4）
    """
    def __init__(self,
                 backbone,
                 num_joints: int = 17,
                 fpn_out_channels: int = 128,
                 head_midc: int = 32,
                 upsample_to_quarter: bool = False):
        super().__init__()
        self.backbone = backbone
        self.fpn = FPN(in_channels=self.backbone.feature_info.channels(),
                       out_channels=fpn_out_channels)
        self.head = SinglePoseHead(in_ch=fpn_out_channels,
                                   num_joints=num_joints,
                                   midc=head_midc)
        self.upsample_to_quarter = upsample_to_quarter

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        c3, c4, c5 = self.backbone(x)
        p3, _, _ = self.fpn((c3, c4, c5))  # 取 P3

        if self.upsample_to_quarter:
            p3 = F.interpolate(p3, scale_factor=2.0, mode='bilinear', align_corners=False)

        return self.head(p3)


# %% [markdown]
# ## 模型向前传播实验
# 
# 我们测试这个模型看看有无问题。

# %%
def model_runnable_test_single():
    backbone_mbv3 = timm.create_model(
        'mobilenetv3_large_100',
        pretrained=True,          # 无网可改 False
        features_only=True,
        out_indices=(2, 3, 4),
    )
    print("Backbone for FPN created. Output channels:", backbone_mbv3.feature_info.channels())

    model = MoveNet_FPN(backbone_mbv3,
                        fpn_out_channels=128,
                               num_joints=17,
                               upsample_to_quarter=False)
    model.eval()

    x = torch.randn(1, 3, 192, 192)
    with torch.no_grad():
        y = model(x)

    print({k: v.shape for k, v in y.items()})
    # 192 / 8 = 24
    assert y["heatmaps"].shape == (1, 17, 24, 24)
    assert y["offsets"].shape  == (1, 34, 24, 24)

model_runnable_test_single()

# %% [markdown]
# ## 关于输出结果
# 
# ### Backbone 的通道
# 
# ```
# Backbone for FPN created. Output channels: [40, 112, 960]
# ```
# 
# * 表示 `C3, C4, C5` 的通道数分别是 **40 / 112 / 960**。这三路会被 FPN 的 1×1 侧连卷积统一到 `fpn_out_channels=128`，然后自顶向下融合出 **P3、P4、P5**。我们只取 **P3**。
# 
# ### 为什么输出是 24×24？
# 
# 你的输入是 **192×192**，P3 的步幅（stride）是 **8**，所以空间尺寸为：
# 
# * $H_{P3} = W_{P3} = 192 / 8 = 24$
# 
# 如果你把 `upsample_to_quarter=True`，我们会把 P3 上采样到 1/4 分辨率（stride=4），这时就会是 **48×48**。
# 


