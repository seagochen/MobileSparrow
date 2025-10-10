import torch
from torch import nn

from sparrow.models.movenet_fpn import MoveNet_FPN


class MoveNetExportWrapper(nn.Module):
    def __init__(self, movenet: MoveNet_FPN):
        super().__init__()
        self.movenet = movenet

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        将 MoveNet 的输出打包为单个 tensor 结构:
        [B, 56]
        """
        preds = self.movenet(x)
        heatmaps = preds["heatmaps"]  # [B, 17, H, W]
        offsets = preds["offsets"]  # [B, 34, H, W]

        # 1 解析出每个关键点的坐标和置信度
        # 假设我们直接取每个 heatmap 的 argmax 作为关键点位置
        B, K, H, W = heatmaps.shape
        heatmaps_sigmoid = torch.sigmoid(heatmaps)
        heatmaps_flat = heatmaps_sigmoid.view(B, K, -1)
        max_vals, idxs = torch.max(heatmaps_flat, dim=-1)  # [B, K]

        # 坐标索引
        ys = (idxs // W).float()
        xs = (idxs % W).float()

        # 利用 offsets 提升精度（亚像素修正）
        offsets_reshaped = offsets.view(B, K, 2, H, W)
        offsets_x = offsets_reshaped[:, :, 0, :, :].view(B, K, -1)
        offsets_y = offsets_reshaped[:, :, 1, :, :].view(B, K, -1)

        offx = torch.gather(offsets_x, 2, idxs.unsqueeze(-1)).squeeze(-1)
        offy = torch.gather(offsets_y, 2, idxs.unsqueeze(-1)).squeeze(-1)

        xs = xs + offx
        ys = ys + offy

        # 2 将关键点打包成 [B, 17, 3] （x, y, score）
        keypoints = torch.stack([xs, ys, max_vals], dim=-1)  # [B, 17, 3]

        # 3 简化假设框为关键点边界框（取 min/max）
        lx = torch.min(xs, dim=1, keepdim=True)[0]
        ly = torch.min(ys, dim=1, keepdim=True)[0]
        rx = torch.max(xs, dim=1, keepdim=True)[0]
        ry = torch.max(ys, dim=1, keepdim=True)[0]

        # 目标置信度：平均关键点置信度
        score = max_vals.mean(dim=1, keepdim=True)
    
        # 4 拼接为统一输出
        out = torch.cat([
            lx, ly, rx, ry, score, keypoints.view(B, -1)
        ], dim=1)  # [B, 56]
    
        return out
