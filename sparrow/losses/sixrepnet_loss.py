import torch
import torch.nn as nn
import torch.nn.functional as F


# 1. 主损失：测地距离 (Geodesic Loss)，使用 atan2 稳定版
class GeodesicLoss(nn.Module):
    """
    测地距离损失（Geodesic Loss）- 旋转矩阵的主要损失函数

    核心思想：
      测量两个旋转矩阵 R_pred 和 R_gt 在 SO(3)（3D 旋转群）流形上的距离
      即计算它们之间的最小旋转角度

    数学原理：
      给定两个旋转矩阵 R1 和 R2，它们之间的测地距离为：
        d(R1, R2) = ||log(R1^T * R2)||_F / sqrt(2)
      等价于计算相对旋转 R_rel = R1^T * R2 的旋转角度 θ

      通过 Rodrigues 公式：
        θ = arccos((trace(R_rel) - 1) / 2)

      但该公式在 trace 接近 -1 或 3 时数值不稳定，因此使用 atan2 替代

    优势：
      - 物理意义明确：直接测量旋转角度差异
      - 旋转不变性：对坐标系选择不敏感
      - 数值稳定：atan2 避免 arccos 的梯度消失

    应用：
      头部姿态估计、6DoF 姿态估计、机器人学等需要精确旋转预测的任务
    """

    def __init__(self, eps=1e-7):
        """
        初始化测地距离损失

        参数:
          eps: 数值稳定性常量，防止除零和 log(0)
        """
        super().__init__()
        self.eps = eps

    def forward(self, R_pred, R_gt):
        """
        计算预测旋转矩阵和真值旋转矩阵之间的测地距离

        参数:
          R_pred: 预测的旋转矩阵 [B, 3, 3]
          R_gt: 真值旋转矩阵 [B, 3, 3]

        返回:
          平均旋转角度误差（弧度）

        计算流程：
          1. 计算相对旋转矩阵 R_rel = R_pred * R_gt^T
          2. 从 R_rel 提取旋转轴和角度
          3. 使用 atan2(sin, cos) 稳定地计算角度
        """
        # 1. 计算相对旋转矩阵：R_rel = R_pred * R_gt^T
        # 物理含义：将 GT 旋转到预测姿态所需的旋转
        R_rel = torch.bmm(R_pred, R_gt.transpose(1, 2))  # [B, 3, 3]

        # 2. 计算旋转角度（使用 atan2 的稳定版本）
        #
        # 理论背景：
        # 对于旋转矩阵 R，其对应的旋转角度 θ 可由以下公式计算：
        #   trace(R) = 1 + 2*cos(θ)  =>  cos(θ) = (trace(R) - 1) / 2
        #   ||vee(R - R^T)|| = 2*sin(θ)  =>  sin(θ) = ||vee(skew)|| / 2
        #
        # 其中 vee(·) 是反对称矩阵到向量的映射（提取旋转轴）

        # 计算 trace(R_rel)：对角线元素之和
        trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]  # [B]

        # 计算反对称部分：skew = R_rel - R_rel^T
        # 对于旋转矩阵，skew 编码了旋转轴信息
        skew = R_rel - R_rel.transpose(1, 2)  # [B, 3, 3]

        # 提取反对称矩阵的三个独立分量（旋转轴的表征）
        # 反对称矩阵形式：
        #     [  0   -v3   v2 ]
        #     [ v3    0   -v1 ]
        #     [-v2   v1    0  ]
        v1 = skew[:, 2, 1] - skew[:, 1, 2]  # 2 * v1
        v2 = skew[:, 0, 2] - skew[:, 2, 0]  # 2 * v2
        v3 = skew[:, 1, 0] - skew[:, 0, 1]  # 2 * v3

        # 计算 sin(θ)：||vee(R - R^T)|| = 2*sin(θ)
        # 这里 s 实际上是 2*sin(θ)，但在 atan2 中会自动归一化
        s = torch.sqrt((v1 * v1 + v2 * v2 + v3 * v3).clamp_min(self.eps))  # [B]

        # 计算 cos(θ)：trace(R) = 1 + 2*cos(θ)
        # 裁剪到 [-2+eps, 2-eps] 以保证数值稳定性
        # （理论上 trace 范围是 [-1, 3]，对应 cos 范围 [-1, 1]）
        c = (trace - 1).clamp(min=-2 + self.eps, max=2 - self.eps)  # [B]

        # 使用 atan2 计算角度：θ = atan2(sin, cos)
        # 优势：
        #   1. 输出范围 [-π, π]，覆盖所有旋转角度
        #   2. 在 sin 和 cos 同时接近 0 时仍然稳定
        #   3. 梯度行为良好，无奇异点
        theta_rad = torch.atan2(s, c)  # [B] - 旋转角度（弧度）

        # 返回平均角度误差
        return theta_rad.mean()


# 2. 辅助损失：6D向量正交化后的列向量对齐损失
class SixDColumnCosineLoss(nn.Module):
    """
    6D 表征的列向量余弦相似度损失（辅助损失）

    作用：
      在训练 6D 旋转表征模型时，作为辅助监督信号
      直接约束预测的 6D 向量正交化后的列向量与 GT 旋转矩阵对齐

    设计动机：
      1. 加速收敛：直接监督 6D 向量，而非只监督最终的旋转矩阵
      2. 稳定训练：余弦相似度对尺度不敏感，更鲁棒
      3. 明确约束：确保预测的前两列向量尽可能接近 GT

    与 GeodesicLoss 的关系：
      - GeodesicLoss（主损失）：监督最终旋转矩阵的整体误差
      - SixDColumnCosineLoss（辅助损失）：监督中间 6D 表征的质量
      - 通常组合使用：loss_total = λ1 * geodesic + λ2 * column_cosine

    适用场景：
      配合 SixDRepNet 等基于 6D 表征的旋转预测模型
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def _orthonormalize(ortho6d):
        """
        Gram-Schmidt 正交化：将 6D 向量转换为正交的前两列

        参数:
          ortho6d: 6D 表征 [N, 6]
                   - ortho6d[:, 0:3]: x_raw（X 轴近似）
                   - ortho6d[:, 3:6]: y_raw（Y 轴近似）

        返回:
          x: 正交化后的 X 轴单位向量 [N, 3]
          y: 正交化后的 Y 轴单位向量 [N, 3]（与 x 正交）

        算法：
          1. 归一化 x_raw 得到 x
          2. 将 y_raw 在 x 方向的投影去除，得到垂直分量
          3. 归一化垂直分量得到 y
        """
        # 1. 拆分 6D 向量
        x_raw = ortho6d[:, 0:3]  # [N, 3] - X 轴原始向量
        y_raw = ortho6d[:, 3:6]  # [N, 3] - Y 轴原始向量

        # 2. 归一化 X 轴
        x = F.normalize(x_raw, p=2, dim=1)  # [N, 3] - 单位向量

        # 3. Gram-Schmidt 正交化 Y 轴
        # 计算 y_raw 在 x 方向的投影：proj_x(y_raw) = (x·y_raw) * x
        z_proj = (x * y_raw).sum(dim=1, keepdim=True) * x  # [N, 3]
        # 去除投影，保留垂直分量并归一化
        y = F.normalize(y_raw - z_proj, p=2, dim=1)  # [N, 3] - 与 x 正交的单位向量

        return x, y

    def forward(self, pred_6d, R_gt):
        """
        计算预测 6D 向量与 GT 旋转矩阵之间的列向量对齐损失

        参数:
          pred_6d: 模型预测的 6D 向量 [B, 6]
          R_gt: 真值旋转矩阵 [B, 3, 3]

        返回:
          平均余弦距离损失（标量）

        计算流程：
          1. 从 GT 旋转矩阵中提取前两列（X 轴和 Y 轴）
          2. 将预测 6D 向量正交化为两个单位向量
          3. 分别计算 X 轴和 Y 轴的余弦相似度
          4. 使用 1 - cosine_similarity 作为损失（范围 [0, 2]）

        损失含义：
          - 余弦相似度 = 1：完全对齐（损失为 0）
          - 余弦相似度 = 0：正交（损失为 1）
          - 余弦相似度 = -1：反向（损失为 2）
        """
        # 1. 从 Ground Truth 旋转矩阵中提取前两列
        # R_gt[:, :, 0] 是 X 轴方向的单位向量
        # R_gt[:, :, 1] 是 Y 轴方向的单位向量
        x_gt, y_gt = R_gt[:, :, 0], R_gt[:, :, 1]  # [B, 3], [B, 3]

        # 2. 从预测的 6D 向量中计算正交化的前两列
        x_pred, y_pred = self._orthonormalize(pred_6d)  # [B, 3], [B, 3]

        # 3. 计算余弦相似度损失
        # F.cosine_similarity 返回 [-1, 1]，值越大表示越相似
        # 1 - cosine_similarity 将其转换为损失（越小越好）

        # X 轴对齐损失
        loss_x = 1.0 - F.cosine_similarity(x_pred, x_gt, dim=1)  # [B]

        # Y 轴对齐损失
        loss_y = 1.0 - F.cosine_similarity(y_pred, y_gt, dim=1)  # [B]

        # 4. 返回平均损失
        # 同时约束 X 和 Y 轴，确保预测的 6D 表征质量
        return (loss_x + loss_y).mean()


# 正则化项：对原始6D向量进行约束
class SixDRawRegularizer(nn.Module):
    def __init__(self, target_norm=1.0):
        super().__init__()
        self.target_norm = target_norm

    def forward(self, pred_6d):
        x_raw = pred_6d[:, 0:3]
        y_raw = pred_6d[:, 3:6]

        # 范数约束：鼓励向量长度接近1
        norm_loss = ((x_raw.norm(p=2, dim=1) - self.target_norm)**2 +
                     (y_raw.norm(p=2, dim=1) - self.target_norm)**2).mean()

        # 正交约束：惩罚两个向量之间的非正交性（即cos相似度不为0）
        ortho_loss = (F.cosine_similarity(x_raw, y_raw, dim=1) ** 2).mean()

        return norm_loss + ortho_loss


# 最终的组合损失函数
class SixDCombinedLoss(nn.Module):
    def __init__(self, w_geo=1.0, w_col=0.5, w_reg=0.1):
        """
        组合损失函数。

        Args:
            w_geo (float): 测地距离主损失的权重。
            w_col (float): 6D列向量对齐辅助损失的权重。
            w_reg (float): 6D原始向量正则化项的权重。
        """
        super().__init__()
        self.geodesic_loss = GeodesicLoss()
        self.column_loss = SixDColumnCosineLoss()
        self.regularizer = SixDRawRegularizer()

        self.w_geo = w_geo
        self.w_col = w_col
        self.w_reg = w_reg
        print(f"Combined Loss Weights: Geodesic={self.w_geo}, Column Align={self.w_col}, Regularizer={self.w_reg}")

    def forward(self, pred_matrix, pred_6d, gt_matrix):
        """
        计算总损失。

        Args:
            pred_matrix (torch.Tensor): 模型预测的旋转矩阵 (N, 3, 3)。
            pred_6d (torch.Tensor): 模型预测的原始6D向量 (N, 6)。
            gt_matrix (torch.Tensor): 真实的旋转矩阵 (N, 3, 3)。

        Returns:
            torch.Tensor: 加权后的总损失（标量）。
            dict: 包含各项损失明细的字典，用于监控。
        """
        loss_geo = self.geodesic_loss(pred_matrix, gt_matrix)
        loss_col = self.column_loss(pred_6d, gt_matrix)
        loss_reg = self.regularizer(pred_6d)

        total_loss = (self.w_geo * loss_geo +
                      self.w_col * loss_col +
                      self.w_reg * loss_reg)

        # 返回总损失和详细信息（detach 避免影响梯度）
        return total_loss, {
            "geodesic_loss": loss_geo.detach(),
            "column_loss": loss_col.detach(),
            "regularizer_loss": loss_reg.detach()
        }