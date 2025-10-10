import torch
import torch.nn as nn
from typing import List


class AutomaticWeightedLoss(nn.Module):
    """
    自动加权多任务损失（基于 Uncertainty Weighting）
    - 为每个损失项创建一个可学习的 log(sigma^2) 参数
    - 在训练中自动平衡多个损失
    - ref: https://arxiv.org/abs/1705.07115
    """

    def __init__(self, num_tasks: int, initial_log_sigma_sq: float = 0.0):
        """
        参数:
          num_tasks: 任务（损失）的数量
          initial_log_sigma_sq: log(sigma^2) 的初始值
        """
        super().__init__()
        # 验证任务数量
        if num_tasks <= 0:
            raise ValueError("Number of tasks must be positive.")
        self.num_tasks = int(num_tasks)

        # 创建可学习的参数 log(sigma^2)
        # 使用 nn.Parameter 将其注册为模型参数，这样优化器就能找到并更新它
        self.log_sigma_sq = nn.Parameter(
            torch.full((self.num_tasks,), float(initial_log_sigma_sq))
        )

    def forward(self, losses: List[torch.Tensor]) -> torch.Tensor:
        """
        计算加权后的总损失

        参数:
          losses: 包含各个任务原始损失的列表, e.g., [loss_cls, loss_reg]

        返回:
          加权后的总损失（标量）
        """
        # 确保传入的损失数量与初始化时一致
        if len(losses) != self.num_tasks:
            raise ValueError(
                f"Expected {self.num_tasks} losses, but got {len(losses)}"
            )

        total_loss = torch.tensor(0.0, device=losses[0].device)
        for i, loss_i in enumerate(losses):
            # 公式: L_total += 0.5 * exp(-log_sigma_sq) * L_i + 0.5 * log_sigma_sq
            # 等价于 (1 / (2*sigma^2)) * L_i + log(sigma)
            # 使用 exp(-log(s^2)) 可以提高数值稳定性

            # 1. 计算 sigma^2
            sigma_sq = torch.exp(self.log_sigma_sq[i])

            # 2. 计算加权损失项
            weighted_loss = 0.5 * (1.0 / sigma_sq) * loss_i

            # 3. 计算正则化项
            regularizer = 0.5 * self.log_sigma_sq[i]

            # 4. 累加到总损失
            total_loss += weighted_loss + regularizer

        return total_loss