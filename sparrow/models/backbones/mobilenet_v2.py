# %% [markdown]
# ## 📥 导入模块

# %%
from typing import Tuple

import torch
import torch.nn as nn

# %% [markdown]
# ## 📚 辅助函数

# %% [markdown]
# ### 📌 常用卷积块
# 
# `conv_bn` 是由一个卷积 + BatchNorm + ReLu(Optional) 构成的卷积块 

# %%
def conv_bn(inp, oup, k, s, p, g=1, act=True):
    # 一个常用的卷积块：Conv2d + BN (+ ReLU6)
    # inp: 输入通道数
    # oup: 输出通道数
    # k: 卷积核大小
    # s: 步幅
    # p: 填充
    # g: 分组卷积（默认 1，depthwise conv 时=输入通道数）
    # act: 是否添加激活函数 ReLU6
    layers = [nn.Conv2d(inp, oup, k, s, p, groups=g, bias=False),
              nn.BatchNorm2d(oup)]
    if act: layers += [nn.ReLU6(inplace=True)]
    return nn.Sequential(*layers)

# %% [markdown]
# ## 🔩 反残差模块 `InvertedResidual`
# 
# 这个 `InvertedResidual` 模块就是 **MobileNetV2 的核心构件**，也叫 **反残差块 (Inverted Residual Block)**。它的作用可以总结为：
# 
# ----

# %% [markdown]
# ### 🔑 核心思想
# 
# 1. **先升维 (expand)**
#    用 `1x1 卷积` 把输入通道数从 `inp` 扩展到更高维（`hidden_dim = inp * expand_ratio`），让后续的卷积有更强的表达能力。
# 
# 2. **深度卷积 (depthwise convolution)**
#    对每个通道单独做 `3x3 卷积`（groups = hidden\_dim）。
# 
#    * 计算量比普通卷积少很多（复杂度从 `O(C_in * C_out * K^2)` 降为 `O(C_in * K^2)`）。
# 
# 3. **再降维 (project)**
#    用 `1x1 卷积` 把通道数压缩回 `oup`（输出通道数），避免模型过大。
# 
# 4. **残差连接 (skip connection)**
# 
#    * 如果 stride=1 且 输入通道=输出通道，就加上 `x + F(x)`，实现残差学习；
#    * 否则直接输出卷积结果。
# 
# -----

# %% [markdown]
# ### 🚀 为什么叫 “反残差块”？
# 
# * 在 ResNet 的 bottleneck 结构里：
#   **先降维 → 做卷积 → 再升维**，叫做“残差块”。
# * 在 MobileNetV2 里：
#   **先升维 → 做卷积 → 再降维**，正好反过来 → 所以叫 **Inverted Residual**。
# 
# ----

# %% [markdown]
# ### 📦 模块的优点
# 
# 1. **轻量化**：大量用 `depthwise 卷积`，极大减少计算量。
# 2. **保持信息流动**：通过残差连接，缓解梯度消失问题。
# 3. **高效表达**：先扩展维度再计算，可以在低 FLOPs 下保持较强的特征提取能力。
# 
# ----

# %% [markdown]
# 
# ### 🔎 举个例子
# 
# 假设 `inp=32, oup=16, expand_ratio=6, stride=1`：
# 
# 1. 输入 shape `[N, 32, H, W]`
# 2. 扩展到 `hidden_dim=192` → `[N, 192, H, W]`
# 3. 做 `3x3 depthwise conv` → `[N, 192, H, W]`
# 4. 压缩到输出通道 `16` → `[N, 16, H, W]`
# 5. 因为 stride=1 且通道数不同（32≠16），所以 **没有残差**。
# 
# ---
# 
# 👉 总结一句话：
# **InvertedResidual 是 MobileNetV2 用来构建高效卷积网络的基本单元，靠“扩展–深度卷积–压缩”的反残差结构，在保持精度的同时大幅减少参数量和计算量。**

# %%
class InvertedResidual(nn.Module):
    """
    MobileNetV2 的核心模块：反残差块
    - 输入通道先扩展 (expand)
    - 再做 depthwise 卷积
    - 最后用 1x1 卷积降维到输出通道
    - 如果 stride=1 且输入输出通道相等，则使用残差连接
    """
    
    def __init__(self, in_ch: int, out_ch: int, stride: int, expand_ratio):
        super().__init__()
        hidden_dim = int(round(in_ch * expand_ratio))
        self.use_res_connect = stride == 1 and in_ch == out_ch  # 残差条件
        layers = []

        # 1x1 卷积扩展通道
        if expand_ratio != 1:
            layers.append(conv_bn(in_ch, hidden_dim, 1, 1, 0))
        
        # 3x3 depthwise 卷积
        layers.extend([
            conv_bn(hidden_dim, hidden_dim, 3, stride, 1, g=hidden_dim),

            # 1x1 卷积降维，不加激活
            nn.Conv2d(hidden_dim, out_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_ch),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        # 如果满足条件，走残差连接
        return x + self.conv(x) if self.use_res_connect else self.conv(x)

# %% [markdown]
# ## 🔑 MobileNetV2 的总体思路
# 
# MobileNetV2 是一个 **轻量化卷积神经网络**，主要用于移动端 / 边缘设备。
# 它的核心特征是：
# 
# * 使用 **InvertedResidual (反残差块)** 替代传统卷积层。
# * 使用 **depthwise separable convolution** 来减少计算量。
# * 提供 **width\_mult** 超参数来缩放网络宽度（通道数）。
# 
# 它的 backbone 最终会输出 **C3, C4, C5 三个特征层**，常用于检测网络 (比如 FPN, YOLO, SSD 等)。
# 
# ----

# %% [markdown]
# ### 📦 架构分解
# 
# #### 1. **Stem 层 (输入卷积层)**
# 
# ```python
# out_ch = self._round_ch(32)
# self.stem = conv_bn(in_ch, out_ch, 3, 2, 1)
# ```
# 
# * 输入是图片（通常 3 通道）。
# * 用一个 `3x3 卷积, stride=2`，把输入缩小一半。
# * 输出通道数 ≈ 32（会经过 `_round_ch` 调整成 8 的倍数）。
# 
# ---

# %% [markdown]
# #### 2. **配置表 cfg**
# 
# ```python
#     #t,   c, n, s
# cfg = [
#     [1,  16, 1, 1],
#     [6,  24, 2, 2],  # -> C2
#     [6,  32, 3, 2],  # -> C3
#     [6,  64, 4, 2],  # -> C4
#     [6,  96, 3, 1],
#     [6, 160, 3, 2],  # -> C5
#     [6, 320, 1, 1],
# ]
# ```
# 
# 每一行代表一个 **stage**：
# 
# * `t` = expand\_ratio（通道扩展因子）
# * `c` = 基础输出通道数（会乘 `width_mult` 再对齐 8）
# * `n` = block 数量（堆叠多少个 InvertedResidual）
# * `s` = 第一个 block 的 stride（是否下采样）
# 
# 👉 比如 `[6, 32, 3, 2]`
# 表示：
# 
# * 每个 block 先把通道扩展 6 倍；
# * 输出通道数是 32；
# * 重复 3 个 block；
# * 第一个 block stride=2（空间下采样）。
# 
# ---

# %% [markdown]
# #### 3. **堆叠 InvertedResidual blocks**
# 
# 代码循环里：
# 
# ```python
# for t, c, n, s in cfg:
#     output_channel = self._round_ch(c)
#     blocks = []
#     for i in range(n):
#         stride = s if i == 0 else 1
#         blocks.append(InvertedResidual(input_channel, output_channel, stride, t))
#         input_channel = output_channel
#     layers.append(nn.Sequential(*blocks))
# ```
# 
# 意思是：
# 
# * 每个 stage 会堆叠多个 **InvertedResidual**。
# * stage 的第一个 block 决定是否下采样 (stride=s)。
# * 后续的 block 都 stride=1。
# 
# ---

# %% [markdown]
# #### 4. **C3 / C4 / C5 的标记**
# 
# ```python
# if s == 2:
#     stage += 1
#     if stage == 3: self.c3_idx, c3c = len(layers)-1, output_channel
#     if stage == 4: self.c4_idx, c4c = len(layers)-1, output_channel
#     if stage == 5: self.c5_idx, c5c = len(layers)-1, output_channel
# ```
# 
# * 每次遇到 stride=2，相当于下采样，stage+1。
# * 分别把 **C3, C4, C5** 的索引和通道数记录下来。
# 
#   * C3: stride=8 的特征层
#   * C4: stride=16 的特征层
#   * C5: stride=32 的特征层
# 
# 这些就是 **FPN 等上层结构**要用的主干特征。
# 
# ---

# %% [markdown]
# #### 5. **前向传播**
# 
# ```python
# def forward(self, x):
#     x = self.stem(x)
#     for i, m in enumerate(self.features):
#         x = m(x)
#         if i == self.c3_idx: c3 = x
#         if i == self.c4_idx: c4 = x
#         if i == self.c5_idx: c5 = x
#     return c3, c4, c5
# ```
# 
# * 输入先经过 stem。
# * 依次通过每个 stage（`self.features`）。
# * 在 C3, C4, C5 对应位置保存特征。
# * 最终返回 `(C3, C4, C5)`。
# 
# ---

# %% [markdown]
# ### 🖼️ 总体结构图
# 
# 根据配置表cfg
# 
# ```python
#     #t,   c, n, s
# cfg = [
#     [1,  16, 1, 1],
#     [6,  24, 2, 2],  # -> C2
#     [6,  32, 3, 2],  # -> C3
#     [6,  64, 4, 2],  # -> C4
#     [6,  96, 3, 1],
#     [6, 160, 3, 2],  # -> C5
#     [6, 320, 1, 1],
# ]
# ```
# 
# 可以得到网络的实际拓扑结构
# 
# ```
# Input
#  │
#  ├── Stem (3x3 conv, stride=2) → 32通道
#  │
#  ├── Stage1: [1,16,1,1] → 16通道
#  ├── Stage2: [6,24,2,2] → 24通道 (C2)
#  ├── Stage3: [6,32,3,2] → 32通道 (C3, stride=8)
#  ├── Stage4: [6,64,4,2] → 64通道 (C4, stride=16)
#  ├── Stage5: [6,96,3,1] → 96通道
#  ├── Stage6: [6,160,3,2] → 160通道 (C5, stride=32)
#  └── Stage7: [6,320,1,1] → 320通道
# ```
# 
# ---

# %% [markdown]
# ### 🔨 具体实现
# 
# 需要注意在 **标准 MobileNetV2** 论文里，默认的 **`width_mult=1.0`**，也就是 **不缩放通道数**，保持论文里定义的通道配置。
# 
# 不过 `width_mult` 的设计是为了 **轻量化** 或 **扩展模型容量**：
# 
# * **`width_mult < 1.0`**
# 
#   * 比如 0.75、0.5、0.35
#   * 会把所有层的通道数缩小，相应减少计算量和参数量
#   * 适合在移动端、算力更受限的设备上使用
# 
# * **`width_mult = 1.0`**
# 
#   * 默认配置，论文中的 MobileNetV2-1.0
#   * 通道数和精度都是标准版本
# 
# * **`width_mult > 1.0`**
# 
#   * 比如 1.25
#   * 会扩大通道数，让模型更大
#   * 适合需要更高精度但算力允许的场景
# 
# ---
# 
# 所以实际用的时候：
# 
# * **学术复现 / benchmark**：一般都用 `1.0`
# * **推理部署**：会尝试 `0.75` 或 `0.5` 来换取速度
# * **精度优先**：有时会用 `1.25`

# %%
class MobileNetV2(nn.Module):
    """
    MobileNetV2 骨干网络
    - 输出三个层级特征 (C3, C4, C5)，供 FPN 使用。
    - 支持 width_mult：按比例缩放各层通道数（轻量化常用技巧）。
    """
    def __init__(self, in_ch: int = 3, width_mult: float = 1.0):
        super().__init__()
        self.width_mult = float(width_mult)

        # 配置表 cfg: 每一行表示一个 stage
        # t=expand_ratio 扩展因子
        # c=输出通道数（基准值，后续会乘 width_mult 再取整）
        # n=重复的 block 数
        # s=该 stage 第一次 block 的 stride
        #   t,    c, n, s
        cfg = [
            [1,  16, 1, 1],
            [6,  24, 2, 2],  # -> C2
            [6,  32, 3, 2],  # -> C3
            [6,  64, 4, 2],  # -> C4
            [6,  96, 3, 1],
            [6, 160, 3, 2],  # -> C5
            [6, 320, 1, 1],
        ]

        # stem: 输入卷积层，输出通道约 32
        out_ch = self._round_ch(32)
        self.stem = conv_bn(in_ch, out_ch, 3, 2, 1)

        layers = []
        input_channel = out_ch
        # 记录 C3, C4, C5 的索引位置和通道数
        self.c3_idx = self.c4_idx = self.c5_idx = None
        c3c = c4c = c5c = None
        stage = 2  # 起始是 stem 输出，相当于 C2

        # 堆叠 InvertedResidual blocks
        for t, c, n, s in cfg:
            output_channel = self._round_ch(c)
            blocks = []
            for i in range(n):
                stride = s if i == 0 else 1  # 只有第一个 block 才下采样
                blocks.append(InvertedResidual(in_ch=input_channel, out_ch=output_channel, stride=stride, expand_ratio=t))
                input_channel = output_channel
            layers.append(nn.Sequential(*blocks))

            # 根据 stride=2 的位置，确定 C3, C4, C5
            if s == 2:
                stage += 1
                if stage == 3: self.c3_idx, c3c = len(layers)-1, output_channel
                if stage == 4: self.c4_idx, c4c = len(layers)-1, output_channel
                if stage == 5: self.c5_idx, c5c = len(layers)-1, output_channel

        self.features = nn.ModuleList(layers)

        # 记录各层通道数，供上层(FPN)查询
        # 保存 (C3, C4, C5) 的输出通道数
        self._out_channels = (c3c, c4c, c5c)

    def _round_ch(self, c: int, divisor: int = 8) -> int:
        """
        将通道数调整为 8 的倍数（MobileNetV2 的做法）
        - 保证不会下降超过 10%
        - 提升硬件友好性（适配 GPU/加速器）
        """
        c = int(c * self.width_mult)
        new_c = max(divisor, (c + divisor // 2) // divisor * divisor)
        # Prevents channel count from dropping by more than 10%
        if new_c < 0.9 * c:
            new_c += divisor
        return new_c

    def get_out_channels(self) -> Tuple[int, int, int]:
        """返回 (C3, C4, C5) 的通道数"""
        return self._out_channels

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播，返回 FPN 所需的三层特征
        """
        x = self.stem(x)
        c3 = c4 = c5 = None
        for i, m in enumerate(self.features):
            x = m(x)
            if i == self.c3_idx: c3 = x
            if i == self.c4_idx: c4 = x
            if i == self.c5_idx: c5 = x
        return c3, c4, c5


# %% [markdown]
# ## 🧪 测试 MobileNetV2

# %%
if __name__ == "__main__":
    # 创建模型
    model = MobileNetV2(in_ch=3, width_mult=1.0)
    model.eval()  # 切换到 eval 模式（不会影响 forward，但更符合推理场景）

    # 打印输出通道数
    print("Output channels (C3, C4, C5):", model.get_out_channels())

    # 构造随机输入: batch=1, 3通道, 224x224
    x = torch.randn(1, 3, 224, 224)  # 注意数据维度必须能被16整除

    # 前向传播
    c3, c4, c5 = model(x)

    # 打印输出的尺寸
    print("C3 shape:", c3.shape)
    print("C4 shape:", c4.shape)
    print("C5 shape:", c5.shape)