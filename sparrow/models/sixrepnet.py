# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class SixDRepNet(nn.Module):
    """
    SixDRepNet：基于 6D 旋转表征的头部姿态估计模型

    核心思想：
      - 使用 6 维连续表征（两个 3D 向量）来表示旋转矩阵
      - 避免了欧拉角的万向节锁问题和四元数的不连续性
      - 通过 Gram-Schmidt 正交化过程将 6D 向量转换为旋转矩阵

    架构：
      Backbone (特征提取) -> Linear (回归 6D 向量) -> 正交化 (生成旋转矩阵)

    参考文献：
      "On the Continuity of Rotation Representations in Neural Networks"
      (Zhou et al., CVPR 2019)
    """

    def __init__(self, backbone, feat_dim: int = None):
        """
        初始化 SixDRepNet 模型

        参数:
          backbone: 特征提取器（通常是 timm 的预训练模型，如 MobileNetV3）
                   要求输出为 [N, C] 的特征向量
          feat_dim: Backbone 输出的特征维度，如果为 None 则自动检测
        """
        super().__init__()
        self.backbone = backbone

        # 自动检测 Backbone 输出特征维度
        if feat_dim is None:
            with torch.no_grad():
                # 使用虚拟输入探测输出维度
                _probe = torch.zeros(1, 3, 224, 224)  # 标准 ImageNet 尺寸
                feat_dim = self.backbone(_probe).shape[1]  # 如 MobileNetV3: 1280

        # 创建回归头：将特征映射到 6D 旋转表征
        # 6D = 两个 3D 向量（分别近似旋转矩阵的前两列：X 轴和 Y 轴）
        self.linear_reg = nn.Linear(feat_dim, 6)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """
        初始化回归头的权重

        使用 Xavier 均匀初始化保证训练初期的稳定性
        """
        # Xavier 初始化：适用于线性层，保持方差稳定
        nn.init.xavier_uniform_(self.linear_reg.weight)
        # 偏置初始化为 0
        if self.linear_reg.bias is not None:
            nn.init.zeros_(self.linear_reg.bias)

    def forward(self, x):
        """
        前向传播：从输入图像预测 6D 旋转表征

        参数:
          x: 输入图像张量 [N, 3, H, W]，通常 H=W=224

        返回:
          output_6d: 6D 旋转表征 [N, 6]
                     - 前 3 维：旋转矩阵第一列的近似（X 轴方向）
                     - 后 3 维：旋转矩阵第二列的近似（Y 轴方向）

        说明:
          返回的 6D 向量需要通过 compute_rotation_matrix_from_orthod 方法
          转换为标准的 3x3 旋转矩阵
        """
        # 1. 使用 Backbone 提取特征
        # timm 模型（设置 num_classes=0）会自动移除分类头，输出 [N, C_feat]
        features = self.backbone(x)  # [N, feat_dim]

        # 2. 线性回归得到 6D 旋转表征
        # 不需要额外的池化或展平操作，因为 Backbone 已经输出全局特征向量
        output_6d = self.linear_reg(features)  # [N, 6]

        return output_6d

    @staticmethod
    def compute_rotation_matrix_from_orthod(orthod):
        """
        从 6D 表征计算 3x3 旋转矩阵（Gram-Schmidt 正交化）

        原理：
          1. 将 6D 向量拆分为两个 3D 向量：x_raw 和 y_raw
          2. 归一化 x_raw 得到旋转矩阵的第一列（X 轴）
          3. 将 y_raw 正交化到 x，得到第二列（Y 轴）
          4. 通过叉乘 x × y 得到第三列（Z 轴）
          5. 组合得到正交旋转矩阵 R = [x | y | z]

        数学细节：
          - x = normalize(x_raw)
          - y = normalize(y_raw - (x·y_raw)x)  # Gram-Schmidt 正交化
          - z = x × y  # 叉乘保证右手坐标系

        参数:
          orthod: 6D 表征向量 [N, 6]
                  - orthod[:, 0:3]: x_raw（X 轴近似）
                  - orthod[:, 3:6]: y_raw（Y 轴近似）

        返回:
          matrix: 旋转矩阵 [N, 3, 3]
                  - matrix[:, :, 0]: X 轴单位向量
                  - matrix[:, :, 1]: Y 轴单位向量
                  - matrix[:, :, 2]: Z 轴单位向量

        优点：
          - 连续性：6D 空间到 SO(3) 的映射是连续的
          - 无奇异性：避免万向节锁（欧拉角）和符号翻转（四元数）
          - 可微分：适合梯度优化
        """
        # 1. 拆分 6D 向量为两个 3D 向量
        x_raw = orthod[:, 0:3]  # [N, 3] - X 轴的原始向量
        y_raw = orthod[:, 3:6]  # [N, 3] - Y 轴的原始向量

        # 2. 归一化 X 轴（L2 范数归一化）
        x = nn.functional.normalize(x_raw, p=2, dim=1)  # [N, 3] - 单位向量

        # 3. Gram-Schmidt 正交化：将 Y 轴正交化到 X 轴
        # 投影：y_raw 在 x 方向上的投影分量
        z_proj = torch.sum(x * y_raw, dim=1, keepdim=True) * x  # [N, 3]
        # 正交化：去除投影分量，保留垂直部分
        y = nn.functional.normalize(y_raw - z_proj, p=2, dim=1)  # [N, 3] - 单位向量

        # 4. 计算 Z 轴：通过叉乘确保右手坐标系
        # x × y 得到垂直于 XY 平面的向量
        z = torch.cross(x, y, dim=1)  # [N, 3] - 自动归一化（x 和 y 都是单位向量）

        # 5. 组合为旋转矩阵：将三个列向量堆叠
        # matrix[:, :, 0] = x,  matrix[:, :, 1] = y,  matrix[:, :, 2] = z
        matrix = torch.stack((x, y, z), dim=2)  # [N, 3, 3]

        return matrix