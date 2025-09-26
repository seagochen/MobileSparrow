# %% [markdown]
# # 构建一个现代SSDLite网络
# 
# 将 FPN 融合到我们现有的 SSDLite 架构中是一个非常棒的想法，这代表了从“经典”设计向“现代”高效检测器架构的演进。
# 
# 这个融合的思路是：**用 FPN Neck 替换掉原来 SSD-style 的 `ExtraLayers`**，让模型具备更强大的多尺度特征融合能力，同时保留 SSDLite 轻量化的预测头。
# 
# 我们将这个新模型称为 `SSDLite_FPN`。
# 
# -----

# %%
# !pip -q install timm

# %% [markdown]
# ## 架构演进思路
# 
# 我们将进行以下 **三步核心改造**：
# 
# 1.  **改造 Backbone**：FPN 通常需要从更高分辨率的特征图（如 C3, stride=8）开始融合。因此，我们需要修改 `timm` 的 `out_indices`，让它输出 `(C3, C4, C5)` 而不是只有 `(C4, C5)`。
# 2.  **引入 FPN Neck**：我们将之前讨论过的 `FPN` 类作为一个独立的 Neck 模块加入进来。它会接收 Backbone 输出的 `(C3, C4, C5)`，并生成融合后的新特征金字塔 `(P3, P4, P5)`。
# 3.  **替换 `ExtraLayers`**：FPN 已经为我们生成了高质量的多尺度特征，因此不再需要原来那种顺序叠加的 `ExtraLayers`。我们会用一个更简单的方式在 FPN 的输出 `P5` 基础上继续下采样，以生成 `P6` 等用于检测大物体的特征图。
# 
# -----

# %% [markdown]
# ## 融合 FPN 的代码实现
# 
# 下面是完整的代码，您可以将其整合到您现有的 Notebook 中。

# %% [markdown]
# ### 步骤一：定义 FPN Neck 模块
# 
# 我们首先需要一个 FPN Neck 的实现。这里我们使用之前讨论过的标准 FPN 类。

# %%
import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

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
# ### 步骤二：构建 SSDLite\_FPN 完整模型
# 
# 现在我们来构建新的 `SSDLite_FPN` 类。注意看 `__init__` 和 `forward` 方法中的变化。

# %%
# SSDLitePredictionHead 函数保持不变，因为我们仍然需要轻量化的预测头
def SSDLitePredictionHead(in_channels, num_classes, num_anchors):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=1),
    )

class SSDLite_FPN(nn.Module):
    def __init__(self, backbone, num_classes=21, fpn_out_channels=128, num_anchors=6):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        
        # 1. 初始化 FPN Neck
        backbone_channels = self.backbone.feature_info.channels()
        self.fpn = FPN(in_channels=backbone_channels, out_channels=fpn_out_channels)
        
        # 2. 在 FPN 输出的基础上构建额外的 P6, P7 层
        self.extra_layers = nn.ModuleList([
            # 从 P5 (fpn_out_channels) 生成 P6
            nn.Conv2d(fpn_out_channels, fpn_out_channels, kernel_size=3, stride=2, padding=1),
            # 从 P6 (fpn_out_channels) 生成 P7
            nn.Conv2d(fpn_out_channels, fpn_out_channels, kernel_size=3, stride=2, padding=1)
        ])
        
        # 3. 定义用于预测的特征图通道数 (P3, P4, P5, P6, P7)
        # 注意：所有特征图的通道数都被 FPN 统一为了 fpn_out_channels
        self.feature_map_channels = [fpn_out_channels] * 5
        
        # 4. 创建分类和回归头 (仍然是 SSDLite 的轻量化头)
        self.cls_heads = nn.ModuleList()
        self.reg_heads = nn.ModuleList()
        
        for in_channels in self.feature_map_channels:
            self.cls_heads.append(
                SSDLitePredictionHead(in_channels, self.num_classes, num_anchors)  # 动态创建 anchors
            )
            self.reg_heads.append(
                SSDLitePredictionHead(in_channels, 4, num_anchors) # 动态创建 anchors
            )

    def forward(self, x):
        # 1. 通过 Backbone 获取 (C3, C4, C5)
        features = self.backbone(x)
        
        # 2. 通过 FPN Neck 获得融合后的 (P3, P4, P5)
        p3, p4, p5 = self.fpn(features)
        
        # 3. 通过 ExtraLayers 生成 P6, P7
        p6 = self.extra_layers[0](p5)
        p7 = self.extra_layers[1](p6)
        
        # 4. 整理所有用于预测的特征图
        all_features = [p3, p4, p5, p6, p7]
        
        cls_preds = []
        reg_preds = []
        
        # 5. 对每个特征图应用预测头（与之前 SSDLite 逻辑相同）
        for i, feature in enumerate(all_features):
            cls_pred = self.cls_heads[i](feature)
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous()
            cls_preds.append(cls_pred.view(x.size(0), -1, self.num_classes))
            
            reg_pred = self.reg_heads[i](feature)
            reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous()
            reg_preds.append(reg_pred.view(x.size(0), -1, 4))
            
        cls_preds = torch.cat(cls_preds, dim=1)
        reg_preds = torch.cat(reg_preds, dim=1)
        
        return cls_preds, reg_preds

# %% [markdown]
# ### 步骤三：实验与测试
# 
# 现在，我们实例化并测试这个新的 `SSDLite_FPN` 模型。

# %%
if __name__ == "__main__":
    # --- 测试我们构建的 SSDLite_FPN 模型 ---
    
    # 1. 实例化 Backbone，注意修改 out_indices 来获取 C3, C4, C5
    backbone_fpn = timm.create_model(
        'mobilenetv3_large_100',
        pretrained=True,
        features_only=True,
        out_indices=(2, 3, 4), # <-- 核心改动：获取 stride=8, 16, 32 的特征
    )
    
    print("Backbone for FPN created. Output channels:", backbone_fpn.feature_info.channels())
    
    # 2. 实例化 SSDLite_FPN
    # 假设是 PASCAL VOC 数据集, 20个类别 + 1个背景 = 21
    # FPN 输出通道设为 128
    model_fpn = SSDLite_FPN(backbone_fpn, num_classes=21, fpn_out_channels=128)
    model_fpn.eval()

    # 3. 测试前向传播
    input_tensor = torch.randn(1, 3, 320, 320)
    cls_out, reg_out = model_fpn(input_tensor)
    
    print("\n--- SSDLite_FPN Model Test ---")
    print(f"Input shape: {input_tensor.shape}")
    print(f"Classification output shape: {cls_out.shape}")
    print(f"Regression output shape: {reg_out.shape}")
    
    # 总预测框数
    total_boxes = cls_out.shape[1]
    print(f"Total predicted boxes per image: {total_boxes}")

# %% [markdown]
# ## 总结：我们做了什么？
# 
# 1.  **强强联合**：我们保留了 `timm` 高质量的 **MobileNetV3 Backbone** 和 SSDLite 轻量化的 **Prediction Head**。
# 2.  **核心替换**：我们用现代检测器中更强大、效果更好的 **FPN Neck** 替换掉了经典 SSD 中的 `ExtraLayers` 模块。
# 3.  **性能提升**：这种新架构通常会带来更高的检测精度，尤其是对于小目标的识别能力，因为 FPN 让高分辨率的特征图（P3）也获得了来自深层的丰富语义信息。


