import torch
import torch.nn as nn
import torch.nn.functional as F


class SixDRepNetExportWrapper(nn.Module):
    """
    封装 SixDRepNet 模型，使其输出 [B, 9] 的旋转矩阵展平结果
    """

    def __init__(self, sixdrepnet: nn.Module):
        super().__init__()
        self.model = sixdrepnet

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
          x: [B, 3, H, W]
        输出:
          rot_flat: [B, 9] - 旋转矩阵展平结果
        """
        # 1 获取原始6D输出
        out_6d = self.model(x)  # [B, 6]

        # 2 转换为旋转矩阵 [B, 3, 3]
        x_raw = out_6d[:, 0:3]
        y_raw = out_6d[:, 3:6]

        # 归一化 x 向量
        x_vec = F.normalize(x_raw, p=2, dim=1)
        # Gram-Schmidt 正交化 y 向量
        z_proj = torch.sum(x_vec * y_raw, dim=1, keepdim=True) * x_vec
        y_vec = F.normalize(y_raw - z_proj, p=2, dim=1)
        # 叉乘得到 z
        z_vec = torch.cross(x_vec, y_vec, dim=1)

        # 拼接为旋转矩阵
        R = torch.stack((x_vec, y_vec, z_vec), dim=2)  # [B, 3, 3]

        # 3 展平为 [B, 9]
        R_flat = R.reshape(x.shape[0], 9)

        return R_flat
