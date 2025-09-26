# %% [markdown]
# # 如何构建一个SSDLite网络
# 
# 训练像 MobileNet 这样的模型，如果要从零开始（train from scratch），确实需要大量的计算资源（多GPU、长时间训练）、庞大的数据集（如 ImageNet），以及复杂的训练技巧（特定的学习率调度、正则化等）。这对于个人开发者或小型团队来说是非常消耗精力的。
# 
# 因此，更常见的做法是 **直接使用 `timm` 库中经过 ImageNet 预训练的模型作为 Backbone，是目前学术界和工业界最高效、最主流的做法**。这能让您把精力集中在下游任务（如 MoveNet, SSDLite）的架构设计和调优上，而不是耗费在 Backbone 的预训练上。
# 
# 接下来，我们就遵循这个思路，尝试用 `timm` 的 `mobilenetv3_large_100` 作为 Backbone，来一步步构建 SSDLite 的网络架构。
# 
# ----

# %% [markdown]
# ## SSDLite 架构思想
# 
# 在构建之前，我们先要理解 SSDLite 相比于经典 SSD (Single Shot MultiBox Detector) 的核心区别：
# 
# 1.  **轻量化主干网络 (Backbone)**：SSDLite 通常搭配像 MobileNetV2/V3 这样的轻量级网络，而不是 VGG 或 ResNet。这一点我们通过 `timm` 来实现。
# 2.  **轻量化预测头 (Prediction Head)**：这是 **SSDLite 的精髓**。经典 SSD 在每个特征图上使用标准的 `3x3` 卷积来预测类别和边界框位置。而 SSDLite 将这些标准卷积替换为 **深度可分离卷积 (Depthwise Separable Convolutions)**。这极大地减少了检测头的参数量和计算量，使其与轻量级 Backbone 完美匹配。
# 
# ----

# %% [markdown]
# ## 构建步骤
# 
# 我们将分四步来构建 SSDLite 网络：
# 
# 1.  **步骤一：加载 `timm` MobileNetV3 Backbone 并提取多尺度特征**
# 2.  **步骤二：构建额外的特征层 (Extra Layers)**
# 3.  **步骤三：构建 SSDLite 预测头 (Prediction Heads)**
# 4.  **步骤四：组装成完整的 SSDLite 网络**
# 
# -----

# %%
# !pip -q install timm

# %% [markdown]
# ### 步骤一：加载 `timm` Backbone 并提取特征
# 
# `timm` 提供了一个非常强大的功能 `features_only=True`，可以直接让模型输出中间层的特征图，这正是我们所需要的。

# %%
import torch
import torch.nn as nn
import timm

# 1. 加载 timm 中的 MobileNetV3 作为特征提取器
# features_only=True: 让模型返回一个特征图列表，而不是最终的分类输出
# out_indices: 指定我们想要输出哪些特征图的索引。
#   对于 MobileNetV3 (large)，timm 默认返回5个特征层，stride 分别为 (2, 4, 8, 16, 32)
#   我们选择索引 3 和 4，也就是 stride=16 和 stride=32 的特征图
backbone = timm.create_model(
    'mobilenetv3_large_100',
    pretrained=True,
    features_only=True,
    out_indices=(3, 4), 
)

# 我们可以通过 model.feature_info 获取每个输出特征的详细信息
feature_info = backbone.feature_info
c4_channels = feature_info[0]['num_chs'] # stride 16
c5_channels = feature_info[1]['num_chs'] # stride 32

print(f"timm MobileNetV3 特征提取器加载成功！")
print(f"选择的特征图通道数: C4({c4_channels} channels), C5({c5_channels} channels)")

# 测试一下
dummy_input = torch.randn(1, 3, 320, 320)
features = backbone(dummy_input)

print("\nBackbone 输出特征图的尺寸:")
for i, f in enumerate(features):
    print(f"特征层 {i}: {f.shape}")

# %% [markdown]
# ### 步骤二：构建额外的特征层 (Extra Layers)
# 
# 经典 SSD 架构会在 Backbone 的基础上额外添加几个卷积层，以获得更小的特征图来检测更大的物体。SSDLite 沿用了这个设计，但同样可以使用轻量化的卷积块。

# %%
# 简单的卷积块，用于额外特征层
def extra_conv_block(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1),
        nn.ReLU(inplace=True),
    )

class ExtraLayers(nn.Module):
    """
    在 Backbone 最后一层特征图的基础上，构建更多下采样层
    """
    def __init__(self, in_channels, configs=[512, 256, 256, 128]):
        super().__init__()

        # 将配置集中存放，易于修改
        self.configs = configs

        self.layers = nn.ModuleList()
        self.output_channels = [] # <--- 用于记录输出通道的列表

        # 开始动态的创建卷积层
        current_in_channels = in_channels
        for out_channels in self.configs:

            # 创建卷积块
            block = extra_conv_block(current_in_channels, out_channels, stride=2)
            self.layers.append(block)
            
            # 记录这个块的输出通道数
            self.output_channels.append(out_channels)
            
            # 更新下一个块的输入通道数
            current_in_channels = out_channels

    def forward(self, x):
        features = []
        # 循环执行每个块
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return features
    

# %% [markdown]
# ### 步骤三：构建 SSDLite 预测头
# 
# 这是定义 **SSDLite** 的关键。我们用深度可分离卷积来构建分类头 (Classification Head) 和回归头 (Regression Head)。

# %%
def SSDLitePredictionHead(in_channels, num_classes, num_anchors):
    """
    构建 SSDLite 的分类或回归头
    
    Args:
        in_channels (int): 输入特征图的通道数
        num_classes (int): 如果是分类头，则是类别数；如果是回归头，则是4 (dx,dy,dw,dh)
        num_anchors (int): 每个位置的锚点框数量
    """
    # 使用深度可分离卷积
    # 3x3 depthwise conv -> 1x1 pointwise conv
    return nn.Sequential(
        # 深度卷积
        nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
        nn.ReLU(inplace=True),
        # 逐点卷积
        nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=1),
    )

# %% [markdown]
# ### 步骤四：组装成完整的 SSDLite 网络
# 
# 现在，我们将 Backbone、ExtraLayers 和 PredictionHeads 组装起来。

# %%
class SSDLite(nn.Module):
    def __init__(self, backbone, num_classes=21, num_anchors=6):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes  # 保存到实例属性
        
        # 用 timm 对齐 API，顺序与 out_indices 一致
        chs = self.backbone.feature_info.channels()  # e.g. [C4, C5]
        c4_channels, c5_channels = chs[0], chs[1]
        
        # 1. 额外特征层 (现在实例化的是新版 ExtraLayers)
        self.extra_layers = ExtraLayers(c5_channels)
        
        # 2. 核心改进：动态构建特征图通道列表
        backbone_channels = self.backbone.feature_info.channels()
        extra_layer_channels = self.extra_layers.output_channels # 直接从模块获取！

        # 将来自 backbone 和 extra_layers 的通道号拼接起来，不再需要硬编码
        self.feature_map_channels = backbone_channels + extra_layer_channels
        
        # 3. 后续的锚框列表创建和预测头创建逻辑完全不变
        #    因为它们已经依赖于 self.feature_map_channels，所以会自动适应
        self.num_anchors_list = [num_anchors] * len(self.feature_map_channels)
        
        self.cls_heads = nn.ModuleList()
        self.reg_heads = nn.ModuleList()
        
        # 4. 自适应创建预测头
        for i, in_channels in enumerate(self.feature_map_channels):
            current_num_anchors = self.num_anchors_list[i]
            self.cls_heads.append(
                SSDLitePredictionHead(in_channels, self.num_classes, current_num_anchors)
            )
            self.reg_heads.append(
                SSDLitePredictionHead(in_channels, 4, current_num_anchors)
            )
            
    def forward(self, x):
        # 1. 通过 Backbone 和 ExtraLayers 获取所有尺度的特征图
        backbone_features = self.backbone(x)          # [C4, C5]
        
        # self.extra_layers() 现在直接返回一个列表
        extra_features = self.extra_layers(backbone_features[-1])
        all_features = backbone_features + extra_features
        
        cls_preds = []
        reg_preds = []
        
        # 2. 对每个特征图应用对应的预测头
        for i, feature in enumerate(all_features):
            # 分类预测
            cls_pred = self.cls_heads[i](feature)
            # [B, num_anchors * num_classes, H, W] -> [B, H, W, num_anchors * num_classes]
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous()
            # 展平: [B, H * W * num_anchors, num_classes]
            cls_preds.append(cls_pred.view(x.size(0), -1, self.num_classes))
            
            # 回归预测
            reg_pred = self.reg_heads[i](feature)
            # [B, num_anchors * 4, H, W] -> [B, H, W, num_anchors * 4]
            reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous()
            # 展平: [B, H * W * num_anchors, 4]
            reg_preds.append(reg_pred.view(x.size(0), -1, 4))
            
        # 3. 将所有尺度的预测结果拼接起来
        cls_preds = torch.cat(cls_preds, dim=1)
        reg_preds = torch.cat(reg_preds, dim=1)
        
        return cls_preds, reg_preds

# %% [markdown]
# ### 实验与测试

# %%
# --- 测试我们构建的完整 SSDLite 模型 ---
if __name__ == "__main__":
    # 实例化 Backbone
    backbone = timm.create_model(
        'mobilenetv3_large_100',
        pretrained=True,
        features_only=True,
        out_indices=(3, 4),
    )
    
    # 实例化 SSDLite
    # 假设是 PASCAL VOC 数据集, 20个类别 + 1个背景 = 21
    model = SSDLite(backbone, num_classes=21)
    model.eval()

    # 测试前向传播
    input_tensor = torch.randn(1, 3, 320, 320)
    cls_out, reg_out = model(input_tensor)
    
    print("\n--- SSDLite Model Test ---")
    print(f"Input shape: {input_tensor.shape}")
    print(f"Classification output shape: {cls_out.shape}")
    print(f"Regression output shape: {reg_out.shape}")
    
    # 总预测框数
    total_boxes = cls_out.shape[1]
    print(f"Total predicted boxes per image: {total_boxes}")

# %% [markdown]
# 
# ## 总结与后续步骤
# 
# 我们已经成功地使用 `timm` 的预训练 MobileNetV3 作为 Backbone，并结合 SSDLite 的设计思想，构建了一个完整的检测网络架构。
# 
# **需要注意，这只是网络的前向传播架构。一个完整的检测项目还需要包括：**
# 
# 1.  **锚点框生成 (Anchor/Default Box Generation)**：您需要根据每个特征图的尺寸，生成一系列不同大小和长宽比的默认框。
# 2.  **损失函数 (Loss Function)**：实现 MultiBoxLoss，它包含分类损失（如交叉熵损失）和定位损失（如 Smooth L1 Loss）。
# 3.  **目标匹配策略 (Target Matching)**：在训练时，需要将生成的默认框与真实的标注框（ground-truth boxes）进行匹配。
# 4.  **数据增强 (Data Augmentation)**：对于检测任务至关重要，例如随机裁剪、翻转、颜色抖动等。
# 5.  **后处理 (Post-processing)**：在推理时，需要对网络输出的大量预测框进行解码和非极大值抑制（NMS），以得到最终的检测结果。
# 
# 这个架构为您提供了一个坚实的起点，您可以基于它来完成后续的模块，构建一个完整的 SSDLite 目标检测项目。


