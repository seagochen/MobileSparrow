# %% [markdown]
# # 损失函数（单人、两头：Heatmaps + Offsets）
# 
# 我们的单人 MoveNet 现阶段只保留 **两个输出头**：
# 1. **`heatmaps`**：大小为 `[B, K, Hf, Wf]` 的关键点热力图（logits），监督是**高斯软标签**；
# 2. **`offsets`**：大小为 `[B, 2K, Hf, Wf]` 的亚像素偏移。对每个关键点 j，我们在其**正样邻域**（由热图高斯>0定义）写入 `(dx, dy)`，其余位置为 0。
# 
# 因此，总损失写成两部分的加权和：
# 
# $$
# L_{\text{total}} = \lambda_{\text{hm}} \cdot L_{\text{hm}} + \lambda_{\text{off}} \cdot L_{\text{off}}
# $$

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

# %% [markdown]
# ## 选择什么损失？
# 
# **热图损失 $(L_{\text{hm}})$**
# 
# 热图是密集二分类且**正负极度不均衡**，并且我们的监督是**高斯软标签**（不是0/1）。
# 
# 因此采用 **Gaussian Focal Loss（logits 版）** ：
# 
# - 正样项：对预测正确的正样降权 $(1-p)^\alpha$；
# - 负样项：对靠近正样的“灰色负样”用 $(1-\text{target})^\beta$ 再降权，避免过惩罚；
# - 以**正样像素数**归一化，训练更稳。
# - **偏移损失 $L_{\text{off}}$**
# 
# 用 **L1 Loss**，但**只在正样邻域**（即 `gt_heatmap>0` 的格子）且**关键点可见**时计算；并以有效像素数归一化，防止被大片 0 稀释。

# %%
class GaussianFocalLoss(nn.Module):
    """
    CenterNet 风格的高斯 focal（兼容软标签）。
    logits: [B,K,H,W], target in [0,1] with Gaussian peaks.
    """
    def __init__(self, alpha: float = 2.0, beta: float = 4.0, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        pos_w = (1.0 - p).clamp_min(self.eps).pow(self.alpha) * target
        neg_w = (p).clamp_min(self.eps).pow(self.alpha) * (1.0 - target).pow(self.beta)
        loss = (pos_w + neg_w) * bce
        num_pos = (target > 0).sum().clamp_min(1)
        return loss.sum() / num_pos

# %% [markdown]
# ## 张量约定
# 
# - 模型输出：  
#   - `preds = {"heatmaps":[B,K,Hf,Wf], "offsets":[B,2K,Hf,Wf]}`
# 
# - 数据加载器标签：  
#   - `labels` 为拼接张量 `[B, K+2K, Hf, Wf]`（前 K=热图，后 2K=offset），  
#   - `kps_masks` 为 `[B,K]`
# 
# > 若未来扩展到多人/四头（centers/regs），只需在此框架上再加两项权重与损失即可；现在我们保持 2 头，便于聚焦原理。

# %%
class MoveNet2HeadLoss(nn.Module):
    """
    单人两头 MoveNet 损失：
      - heatmaps: Gaussian focal（logits）
      - offsets : L1（仅在正样邻域 & 可见关节）
    """
    def __init__(self,
                 num_joints: int = 17,
                 hm_weight: float = 1.0,
                 off_weight: float = 1.0,
                 focal_alpha: float = 2.0,
                 focal_beta: float = 4.0):
        super().__init__()
        self.K = num_joints
        self.hm_w = hm_weight
        self.off_w = off_weight
        self.focal = GaussianFocalLoss(alpha=focal_alpha, beta=focal_beta)
        self.l1sum = nn.L1Loss(reduction="sum")

    def forward(self,
                preds: Dict[str, torch.Tensor],
                labels: torch.Tensor,      # [B, K+2K, H, W]
                kps_masks: torch.Tensor):  # [B, K]
        B = labels.shape[0]
        K = self.K

        pred_hm  = preds["heatmaps"]          # [B,K,H,W] logits
        pred_off = preds["offsets"]           # [B,2K,H,W]

        gt_hm  = labels[:, :K]                # [B,K,H,W]
        gt_off = labels[:, K:]                # [B,2K,H,W]

        # 1) Heatmaps
        loss_hm = self.focal(pred_hm, gt_hm)

        # 2) Offsets（只算正样邻域 & 可见关节）
        pos_mask = (gt_hm > 0).float()                        # [B,K,H,W]
        pos_mask_2 = pos_mask.repeat_interleave(2, dim=1)     # [B,2K,H,W]
        vis = kps_masks.float().unsqueeze(-1).unsqueeze(-1)   # [B,K,1,1]
        vis_2 = vis.repeat_interleave(2, dim=1)               # [B,2K,1,1]
        valid = pos_mask_2 * vis_2                            # [B,2K,H,W]
        denom = valid.sum().clamp_min(1.0)

        loss_off = self.l1sum(pred_off * valid, gt_off * valid) / denom

        total = self.hm_w * loss_hm + self.off_w * loss_off
        return total, {
            "total_loss": total.detach(),
            "loss_heatmap": loss_hm.detach(),
            "loss_offsets": loss_off.detach(),
        }

# %% [markdown]
# ## 推荐超参
# - `hm_weight=1.0, off_weight=1.0`（坐标学不动可把 `off_weight` 提到 2.0）
# - `focal_alpha=2.0, focal_beta=4.0`
# - 模型的 `heatmaps` **不要带 Sigmoid**（损失里已经用 logits）


