# %% [markdown]
# ## 📥 导入模块

# %%
import torch
import torch.nn as nn
from typing import Tuple

# %% [markdown]
# ## 📚 辅助函数

# %% [markdown]
# ### 📌 `HSigmoid` 激活函数
# 
# 对应的数学公式就是：
# 
# $$
# \text{h-sigmoid}(x) = \frac{\text{ReLU6}(x+3)}{6}
# $$
# 
# 其中
# 
# $$
# \text{ReLU6}(z) = \min(\max(z,0),6)
# $$
# 
# 所以展开就是分段函数：
# 
# $$
# \text{h-sigmoid}(x) =
# \begin{cases}
# 0, & x \leq -3 \\[6pt]
# \dfrac{x+3}{6}, & -3 < x < 3 \\[6pt]
# 1, & x \geq 3
# \end{cases}
# $$
# 
# ---
# 
# 📌 总结：
# 
# * 这是一个 **分段线性函数**，在 $[-3,3]$ 区间内是斜率为 $1/6$ 的直线
# * 超过范围会饱和到 0 或 1
# * 近似标准 Sigmoid，但计算量更小，适合移动端

# %%
class HSigmoid(nn.Module):
    # Hard-Sigmoid 激活函数
    # 定义：ReLU6(x+3)/6
    # 范围在 [0, 1]，近似 Sigmoid，但计算更高效
    def forward(self, x):
        return torch.nn.functional.relu6(x + 3., inplace=True) / 6.

# %% [markdown]
# ### 📌 `HSwish` 激活函数
# 
# `HSwish`（Hard-Swish）是 MobileNetV3 的核心激活函数之一。我们来写出它的数学公式：
# 
# 代码逻辑是：
# 
# $$
# \text{h-swish}(x) = x \cdot \text{h-sigmoid}(x)
# $$
# 
# 而前面我们已经知道：
# 
# $$
# \text{h-sigmoid}(x) = \frac{\text{ReLU6}(x+3)}{6}
# $$
# 
# 因此，展开成分段函数，
# 
# $$
# \text{h-swish}(x) =
# \begin{cases}
# 0, & x \leq -3 \\[6pt]
# \dfrac{x(x+3)}{6}, & -3 < x < 3 \\[6pt]
# x, & x \geq 3
# \end{cases}
# $$
# 
# ---
# 
# 📌 对比 Swish
# 
# * **标准 Swish**:
# 
#   $$
#   \text{swish}(x) = x \cdot \sigma(x), \quad \sigma(x) = \frac{1}{1+e^{-x}}
#   $$
# 
# * **Hard-Swish**:
#   用 **分段线性函数近似 Sigmoid**，计算更高效
# 
#   * 训练和推理速度快
#   * 精度几乎不掉（尤其在 MobileNetV3 这种轻量模型里）
# 
# ---
# 
# 👉 总结一句话：
# 
# $$
# \text{h-swish}(x) = x \cdot \frac{\text{ReLU6}(x+3)}{6}
# $$
# 
# 是 **Swish 的轻量化近似版**，在 $[-3,3]$ 之间平滑过渡，在两侧饱和为 0 和 $x$。
# 

# %%
class HSwish(nn.Module):
    # Hard-Swish 激活函数
    # 定义：x * HSigmoid(x)
    # 近似 Swish(x) = x * sigmoid(x)，但计算代价更低
    def __init__(self):
        super().__init__()
        self.act = HSigmoid()
        
    def forward(self, x):
        return x * self.act(x)

# %% [markdown]
# ### 📌 Squeeze-and-Excitation (SE) 注意力机制
# 
# 这个模块就是经典的 **Squeeze-and-Excitation (SE) 注意力机制**，它在 MobileNetV3 里被广泛使用，用来增强模型的特征表达能力。
# 
# ---
# 
# #### 🔹 公式 / 流程
# 
# 输入特征图 $x \in \mathbb{R}^{B \times C \times H \times W}$，经过以下步骤：
# 
# 1. **Squeeze（全局信息压缩）**
# 
#    * 用全局平均池化（GAP）把空间维度 $H \times W$ 压缩掉，得到通道向量
# 
#    $$
#    s_c = \frac{1}{H \cdot W} \sum_{i=1}^H \sum_{j=1}^W x_{c,i,j}
#    $$
# 
#    得到形状 $B \times C \times 1 \times 1$
# 
# 2. **Excitation（通道注意力）**
# 
#    * 用两个 $1\times 1$ 卷积（相当于全连接层）形成一个小型“瓶颈”网络：
# 
#      * 先降维到 $\text{mid} = \max(8, \lfloor C \cdot \text{se\_ratio} \rfloor)$
#      * 再升回原通道数 $C$
#    * 中间用 **ReLU**，最后用 **HSigmoid**，得到范围在 $[0,1]$ 的通道权重向量
# 
#    $$
#    w = \sigma(W_2(\text{ReLU}(W_1(s))))
#    $$
# 
# 3. **Reweight（通道重标定）**
# 
#    * 把学到的通道权重 $w$ 乘到原始输入特征上：
# 
#    $$
#    y_c = x_c \cdot w_c
#    $$
# 
# ---

# %% [markdown]
# #### 🔹 模块作用
# 
# * 给每个 **通道** 分配一个重要性权重
# * 能让网络学会：
# 
#   * 增强“有用的通道”
#   * 抑制“无用的通道”
# * 相当于在通道维度做了一次“注意力机制”
# 
# ---

# %% [markdown]
# #### 🔹 应用场景
# 
# * **SE-Net** 首先提出，用在 ResNet 上显著提升精度
# * 在 **MobileNetV3** 里，SE 模块被轻量化后加入到 **MBConv** 块中
# * 广泛应用在：
# 
#   * 图像分类
#   * 检测（比如 EfficientDet、YOLO 系列也用过类似注意力）
#   * 分割
# 
# ---
# 
# 👉 一句话总结：
# **Squeeze-and-Excitation 模块就是给通道加“注意力”，让网络自己学会哪些通道更重要。**

# %%
class SqueezeExcite(nn.Module):
    """
    Squeeze-and-Excitation 模块 (SE 注意力机制)
    - 先全局池化 (squeeze)
    - 然后两个 1x1 卷积 (相当于全连接) 做通道注意力
    - 使用 HSigmoid 激活，将权重压缩到 [0,1]
    - 最终输出: x * 权重
    """
    
    def __init__(self, in_ch: int, se_ratio: float = 0.25):
        super().__init__()
        mid = max(8, int(in_ch * se_ratio))  # 中间通道数，不低于 8
        self.pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.fc1 = nn.Conv2d(in_ch, mid, 1, 1, 0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(mid, in_ch, 1, 1, 0)
        self.hsig = HSigmoid()
        
    def forward(self, x):
        s = self.pool(x)
        s = self.relu(self.fc1(s))
        s = self.hsig(self.fc2(s))
        return x * s

# %% [markdown]
# ### 📌 卷积块
# 
# `conv_bn_act` 是由一个卷积 + BatchNorm + hswish/relu(Optional) 构成的卷积块 

# %%
def conv_bn_act(c_in, c_out, k=3, s=1, p=1, act='hswish'):
    """
    卷积 + BN + 激活函数
    - act='relu'   → ReLU
    - act='hswish' → HSwish
    - act=None     → 不加激活
    """
    layers = [nn.Conv2d(c_in, c_out, k, s, p, bias=False), nn.BatchNorm2d(c_out)]
    if act == 'relu':
        layers.append(nn.ReLU(inplace=True))
    elif act == 'hswish':
        layers.append(HSwish())
    elif act is None:
        pass
    else:
        raise ValueError(f'unknown act: {act}')
    return nn.Sequential(*layers)

# %% [markdown]
# ### 📌 确保通道数适合硬件计算的 `make_divisible`

# %%
def make_divisible(v: float, divisor: int = 8, min_value: int = None) -> int:
    """
    将通道数调整为 divisor 的整数倍
    - MobileNetV3 官方实现方法
    - 保证不会比原始值低超过 10%
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return int(new_v)

# %% [markdown]
# ## 🔩 MBConvV3 (增强版 Inverted Residual Block)
# 
# 这个 **MBConvV3** 就是 **MobileNetV3 的核心模块**，它继承自 MobileNetV2 的 **Inverted Residual Block**，但是加入了 **SE 注意力** 和 **轻量激活函数 (HSwish)**，从而更高效。
# 
# ---

# %% [markdown]
# ### 🔹 结构解析
# 
# 输入：特征图 $x \in \mathbb{R}^{B \times C_{in} \times H \times W}$
# 输出：特征图 $y \in \mathbb{R}^{B \times C_{out} \times H' \times W'}$
# 
# #### 1. 可选的 **通道扩展 (1×1 卷积)**
# 
# * 如果 `exp (mid) != in_ch`，就先用 **1×1 卷积**把通道数升高到 `exp`
# * 相当于在低维空间学习特征后，再投影到高维空间，增加表达能力
# * 带 BN + 激活函数 (ReLU 或 HSwish)
# 
# ---

# %% [markdown]
# #### 2. **深度卷积 (Depthwise Conv)**
# 
# * 用 `groups=mid` 的 **k×k depthwise 卷积**在空间维度提取特征
# * 计算量远小于普通卷积（参数量 \~1/k²）
# * 再接 BN + 激活函数 (ReLU 或 HSwish)
# 
# ---

# %% [markdown]
# #### 3. **Squeeze-and-Excite (SE) 模块**（可选）
# 
# * 如果 `use_se=True`，就在中间插入 **SE 注意力模块**
# * 作用：自适应地给每个通道分配权重，强调重要通道，抑制无效通道
# * MobileNetV3 发现 SE 能显著提升效果，但几乎不增加计算量
# 
# ---

# %% [markdown]
# #### 4. **投影层 (1×1 卷积压缩)**
# 
# * 再用一个 **1×1 卷积**把通道数降到 `out_ch`
# * **没有激活函数**（和 MobileNetV2 一样），这样更适合残差连接
# 
# ---

# %% [markdown]
# #### 5. **残差连接**
# 
# * 如果满足条件：`stride=1 且 in_ch == out_ch`
# 
#   * 说明输入输出形状一样 → 可以加残差
#   * 输出：
# 
#     $$
#     y = x + F(x)
#     $$
# * 否则：
# 
#   $$
#   y = F(x)
#   $$
# 
# ---

# %% [markdown]
# ### 🔹 总结结构
# 
# 整体可以写成：
# 
# $$
# y =
# \begin{cases}
# x + \text{BN}( \text{Conv}_{1\times1}(\text{SE}(\phi(\text{DWConv}(\phi(\text{Conv}_{1\times1}(x))))))), & \text{if stride=1 and in=out} \\
# \text{BN}( \text{Conv}_{1\times1}(\text{SE}(\phi(\text{DWConv}(\phi(\text{Conv}_{1\times1}(x))))))), & \text{otherwise}
# \end{cases}
# $$
# 
# 其中 $\phi$ 是激活函数 ReLU 或 HSwish。
# 
# ---

# %% [markdown]
# ### 🔹 MBConvV2 vs MBConvV3
# 
# | 特征    | MBConvV2                          | MBConvV3                               |
# | ----- | --------------------------------- | -------------------------------------- |
# | 激活函数  | ReLU6                             | ReLU / HSwish                          |
# | SE 模块 | ❌ 无                               | ✅ 有 (部分层)                              |
# | 结构    | 1×1 expand → DWConv → 1×1 project | 1×1 expand → DWConv → SE → 1×1 project |
# | 效果    | 好                                 | 更好（精度更高）                               |
# 
# ---
# 
# 👉 一句话总结：
# **MBConvV3 = Inverted Residual Block (V2) + 更优的激活函数 (HSwish) + SE 通道注意力**。
# 它就是 MobileNetV3 的核心，保证了 **轻量化 + 高性能**。

# %%
class MBConvV3(nn.Module):
    """
    MobileNetV3 的核心模块：MBConv (Mobile Inverted Residual Bottleneck with SE and NL)
    - 结构：
        (1) 可选 1x1 卷积 (扩展通道)
        (2) 深度卷积 (dw conv)
        (3) 激活函数：ReLU 或 HSwish
        (4) 可选 Squeeze-and-Excite
        (5) 1x1 卷积 (压缩通道)
    - 如果 stride=1 且输入通道=输出通道，则使用残差连接
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int, k: int, exp: int, use_se: bool, nl: str):
        super().__init__()
        self.use_res = (stride == 1 and in_ch == out_ch)
        act = 'relu' if nl == 'relu' else 'hswish'

        mid = exp
        layers = []

        # 1x1 卷积扩展通道（如果需要）
        if mid != in_ch:
            layers.append(conv_bn_act(in_ch, mid, k=1, s=1, p=0, act=act))

        # depthwise 卷积
        layers.append(nn.Conv2d(mid, mid, k, stride, k//2, groups=mid, bias=False))
        layers.append(nn.BatchNorm2d(mid))
        layers.append(nn.ReLU(inplace=True) if nl == 'relu' else HSwish())

        # 可选 SE 注意力模块
        if use_se:
            layers.append(SqueezeExcite(mid))

        # 1x1 卷积压缩到 out_ch
        layers.append(nn.Conv2d(mid, out_ch, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(out_ch))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        y = self.block(x)
        return x + y if self.use_res else y

# %% [markdown]
# ## 🔑 MobileNetV3 的总体思路
# 
# MobileNetV3 是 **MobileNetV2 的改进版**，主要针对 **移动端推理速度** 做了优化。
# 它的核心特征是：
# 
# * 仍然基于 **InvertedResidual 结构**（V2 的精髓），但升级为 **MBConvV3**。
# * 引入了 **更高效的激活函数 HSwish**（近似 Swish，但硬件友好）。
# * 部分 block 使用 **SE (Squeeze-and-Excitation)** 注意力机制，增强特征表达。
# * 使用 **NAS (神经架构搜索)** 自动搜索得到更优的配置（kernel 大小、通道数、是否带 SE、激活函数）。
# * 保留了 **宽度倍率 width\_mult** 来调节网络规模。
# 
# 和 V2 一样，V3 也可以作为 **检测任务的 backbone**，输出 **C3、C4、C5** 三个尺度的特征层。
# 
# ---

# %% [markdown]
# ### 📦 架构分解
# 
# #### 1. **Stem 层 (输入卷积层)**
# 
# ```python
# self.stem = conv_bn_act(in_ch, C(16), k=3, s=2, p=1, act='hswish')
# ```
# 
# * 输入图片 (3 通道) → 经过一个 `3x3 卷积 + BN + HSwish`
# * 输出通道数调整为 16（经过 `make_divisible` 对齐成 8 的倍数）
# * 空间下采样一半 (stride=2)。
# 
# ---

# %% [markdown]
# #### 2. **配置表 cfg**
# 
# ```python
# # (kernel, exp, out, se, nl, stride, repeat)
# cfg = [
#     (3,  16,  16, False, 'relu',   1, 1),
#     (3,  64,  24, False, 'relu',   2, 1),
#     (3,  72,  24, False, 'relu',   1, 1),
#     (5,  72,  40, True,  'relu',   2, 1),
#     (5, 120,  40, True,  'relu',   1, 2),
#     (3, 240,  80, False, 'hswish', 2, 1),
#     (3, 200,  80, False, 'hswish', 1, 3),
#     (3, 480, 112, True,  'hswish', 1, 1),
#     (3, 672, 112, True,  'hswish', 1, 1),
#     (5, 672, 160, True,  'hswish', 2, 1),
#     (5, 960, 160, True,  'hswish', 1, 2),
# ]
# ```
# 
# 每一行代表一个 **stage**：
# 
# * `k` → depthwise 卷积的 kernel size (3 or 5)
# * `exp` → 中间扩展通道数
# * `out` → 输出通道数
# * `se` → 是否加 SE 模块
# * `nl` → 激活函数 ('relu' 或 'hswish')
# * `stride` → 下采样步长
# * `repeat` → 堆叠的 block 数量
# 
# 👉 例子：`(5, 72, 40, True, 'relu', 2, 1)`
# 表示：
# 
# * kernel=5，扩展到 72 通道 → depthwise conv → SE → ReLU → 输出 40 通道
# * stride=2（下采样），重复 1 次
# 
# ---

# %% [markdown]
# #### 3. **堆叠 MBConvV3 blocks**
# 
# ```python
# for (k, exp, c, se, nl, s, n) in cfg:
#     for i in range(n):
#         stride = s if i == 0 else 1
#         blocks.append(MBConvV3(in_c, out_c, stride=stride, k=k, exp=exp_c, use_se=se, nl=nl))
#         in_c = out_c
# ```
# 
# * 每个 stage 由多个 **MBConvV3** 组成。
# * 第一个 block 可以做下采样 (`stride=2`)，后续的 stride=1。
# * MBConvV3 内部结构：
# 
#   ```
#   (1x1 conv 扩展 → depthwise conv → 激活函数 → 可选 SE → 1x1 conv 压缩)
#   + 残差连接 (如果 stride=1 且 in=out)
#   ```
# 
# ---

# %% [markdown]
# #### 4. **C3 / C4 / C5 的标记**
# 
# ```python
# if s == 2:
#     down_count += 1
#     if down_count == 3: self.c3_idx, c3c = ...
#     if down_count == 4: self.c4_idx, c4c = ...
#     if down_count == 5: self.c5_idx, c5c = ...
# ```
# 
# * 每次遇到 stride=2，说明分辨率缩小一半，记录下采样层数。
# * 标记对应的特征输出：
# 
#   * C3 → stride=8
#   * C4 → stride=16
#   * C5 → stride=32
# 
# 👉 用于检测器的 FPN/SSD/YOLO。
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
# * 输入先经过 stem → 然后依次通过每个 stage。
# * 在 C3, C4, C5 的位置保存特征。
# * 返回 (C3, C4, C5)。
# 
# ---

# %% [markdown]
# ### 🖼️ MobileNetV3-Large 结构图
# 
# 根据 cfg 展开：
# 
# ```
# Input
#  │
#  ├── Stem (3x3 conv, stride=2, hswish) → 16通道
#  │
#  ├── Stage1: (3,16→16, relu) → 16通道
#  ├── Stage2: (3,64→24, relu, stride=2) → 24通道
#  ├── Stage3: (3,72→24, relu) → 24通道
#  ├── Stage4: (5,72→40, SE+relu, stride=2) → 40通道
#  ├── Stage5: (5,120→40, SE+relu) ×2 → 40通道 (C3, stride=8)
#  ├── Stage6: (3,240→80, hswish, stride=2) → 80通道
#  ├── Stage7: (3,200→80, hswish) ×3 → 80通道
#  ├── Stage8: (3,480→112, SE+hswish) → 112通道
#  ├── Stage9: (3,672→112, SE+hswish) → 112通道 (C4, stride=16)
#  ├── Stage10: (5,672→160, SE+hswish, stride=2) → 160通道
#  ├── Stage11: (5,960→160, SE+hswish) ×2 → 160通道 (C5, stride=32)
# ```
# 
# ---

# %% [markdown]
# ### 📌 总结对比
# 
# | 特点          | MobileNetV2      | MobileNetV3   |
# | ----------- | ---------------- | ------------- |
# | 基本单元        | InvertedResidual | MBConvV3      |
# | 激活函数        | ReLU6            | ReLU / HSwish |
# | 注意力机制       | 无                | 可选 SE         |
# | kernel size | 固定 3x3           | 3x3 / 5x5     |
# | 配置方式        | 人工设计             | NAS 搜索        |
# | 输出特征        | C3/C4/C5         | C3/C4/C5      |
# 
# 👉 可以理解为：**MobileNetV3 = MobileNetV2 + (NAS优化 + SE模块 + HSwish)**。

# %%
class MobileNetV3(nn.Module):
    """
    MobileNetV3-Large 风格骨干，输出 (C3, C4, C5) 供 FPN 使用
    - width_mult: 宽度倍率 (0.35 ~ 1.25)，控制通道数缩放
    - divisor: 通道对齐基数 (默认为 8)
    - 输出：
        C3: 下采样 1/8
        C4: 下采样 1/16
        C5: 下采样 1/32
    """
    def __init__(self, in_ch: int = 3, width_mult: float = 1.0, divisor: int = 8):
        super().__init__()
        self.width_mult = float(width_mult)
        self.divisor = int(divisor)

        def C(x):
            # 通道数调整函数
            return make_divisible(x * self.width_mult, self.divisor)

        # stem：第一个卷积层
        self.stem = conv_bn_act(in_ch, C(16), k=3, s=2, p=1, act='hswish')

        # 每一行配置一个 stage
        # (kernel, exp, out, se, nl, stride, repeat)
        cfg = [
            (3,  16,  16, False, 'relu',   1, 1),
            (3,  64,  24, False, 'relu',   2, 1),
            (3,  72,  24, False, 'relu',   1, 1),
            (5,  72,  40, True,  'relu',   2, 1),
            (5, 120,  40, True,  'relu',   1, 2),
            (3, 240,  80, False, 'hswish', 2, 1),
            (3, 200,  80, False, 'hswish', 1, 3),
            (3, 480, 112, True,  'hswish', 1, 1),
            (3, 672, 112, True,  'hswish', 1, 1),
            (5, 672, 160, True,  'hswish', 2, 1),
            (5, 960, 160, True,  'hswish', 1, 2),
        ]

        in_c = C(16)

        # 用于记录 C3, C4, C5 的层索引和通道数
        self.c3_idx = self.c4_idx = self.c5_idx = None
        c3c = c4c = c5c = None
        down_count = 1  # # stem 已经做了一次 /2

        stage_blocks = []
        for (k, exp, c, se, nl, s, n) in cfg:
            exp_c = C(exp)
            out_c = C(c)
            blocks = []
            for i in range(n): 
                stride = s if i == 0 else 1  # 只有第一个 block 才做下采样
                blocks.append(MBConvV3(in_c, out_c, stride=stride, k=k, exp=exp_c, use_se=se, nl=nl))
                in_c = out_c
            stage_blocks.append((blocks, s, out_c))

        self.features = nn.ModuleList()
        for blocks, s, out_c in stage_blocks:
            if s == 2: # 遇到 stride=2 表示分辨率降低一半
                down_count += 1
                if down_count == 3 and self.c3_idx is None:
                    self.c3_idx, c3c = len(self.features), out_c
                elif down_count == 4 and self.c4_idx is None:
                    self.c4_idx, c4c = len(self.features), out_c
                elif down_count == 5 and self.c5_idx is None:
                    self.c5_idx, c5c = len(self.features), out_c

            self.features.append(nn.Sequential(*blocks))
            
        # 如果 C3/C4/C5 没有标记到，就取最后三层
        if any(x is None for x in (self.c3_idx, self.c4_idx, self.c5_idx)):
            n = len(self.features)
            self.c3_idx, self.c4_idx, self.c5_idx = n-3, n-2, n-1
            c3c = c3c or stage_blocks[-3][2]
            c4c = c4c or stage_blocks[-2][2]
            c5c = c5c or stage_blocks[-1][2]

        self._out_channels = (c3c, c4c, c5c)

    def get_out_channels(self) -> Tuple[int, int, int]:
        """返回 (C3, C4, C5) 的通道数"""
        return self._out_channels

    def forward(self, x):
        """
        前向传播，返回 (C3, C4, C5)
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
# ## 🧪 测试 MobileNetV3

# %%
if __name__ == "__main__":
    # 创建模型
    model = MobileNetV3(in_ch=3, width_mult=1.0)
    model.eval()  # 切换到 eval 模式

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


