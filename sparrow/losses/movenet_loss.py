import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class FocalLoss(nn.Module):
    """
    带 Logits 输入的 Focal Loss 实现。
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
        
        # 计算 pt
        p = torch.sigmoid(pred_logits)
        pt = p * target + (1 - p) * (1 - target)
        
        # 计算 focal loss
        focal_loss = (1 - pt).pow(self.gamma) * bce_loss
        
        # alpha 权重
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_loss = alpha_t * focal_loss
        
        return focal_loss.mean()
    
class MoveNetLoss(nn.Module):
    def __init__(self,
                 focal_loss_alpha: float = 0.25,
                 focal_loss_gamma: float = 2.0,
                 hm_weight: float = 1.0,
                 ct_weight: float = 1.0,
                 reg_weight: float = 2.0,
                 off_weight: float = 1.0):
        super().__init__()
        self.hm_weight = hm_weight
        self.ct_weight = ct_weight
        self.reg_weight = reg_weight
        self.off_weight = off_weight
        
        self.focal_loss = FocalLoss(alpha=focal_loss_alpha, gamma=focal_loss_gamma)
        self.l1_loss = nn.L1Loss(reduction='sum') # 先求和，再手动除以mask的数量

    def forward(self,
                preds: Dict[str, torch.Tensor],
                labels: torch.Tensor,
                kps_masks: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算 MoveNet 的总损失。
        Args:
            preds: 模型的输出字典。
            labels: 来自 DataLoader 的 [B, 86, Hf, Wf] 的标签张量。
            kps_masks: 来自 DataLoader 的 [B, 17] 的关键点可见性掩码。
        Returns:
            total_loss: 总损失。
            loss_dict: 包含各部分损失的字典，用于日志记录。
        """
        # 1. 从标签张量中切分出各个真值 (Ground Truth)
        gt_heatmaps = labels[:, :17, :, :]
        gt_centers  = labels[:, 17:18, :, :]
        gt_regs     = labels[:, 18:52, :, :]
        gt_offsets  = labels[:, 52:, :, :]
        
        B, K, H, W = gt_heatmaps.shape
        device = labels.device
        
        # 2. 计算热力图损失 (L_hm) 和中心点损失 (L_ct)
        loss_hm = self.focal_loss(preds['heatmaps'], gt_heatmaps)
        loss_ct = self.focal_loss(preds['centers'], gt_centers)
        
        # 3. 计算回归损失 (L_reg)
        # 只在中心点位置计算回归损失
        # 创建一个掩码，只保留中心点所在位置为1
        center_mask = (gt_centers.max(dim=1, keepdim=True)[0] > 0.99).float() # [B, 1, H, W]
        center_mask = center_mask.expand(-1, 2 * K, -1, -1) # 扩展到与 regs 相同的通道数
        
        pred_regs = preds['regs'] * center_mask
        gt_regs = gt_regs * center_mask
        
        num_centers = center_mask.sum()
        loss_reg = self.l1_loss(pred_regs, gt_regs) / (num_centers + 1e-8)
        
        # 4. 计算偏移损失 (L_off)
        # 只在关键点位置计算偏移损失
        # 创建一个掩码，只保留关键点所在位置为1
        kpt_mask_spatial = (gt_heatmaps.max(dim=1, keepdim=True)[0] > 0.99).float() # [B, 1, H, W]
        kpt_mask_spatial = kpt_mask_spatial.expand(-1, 2 * K, -1, -1) # 扩展到与 offsets 相同的通道数
        
        # 结合 kps_masks (可见性)
        kps_masks_expanded = kps_masks.view(B, K, 1, 1).expand(-1, -1, 2, -1).reshape(B, 2*K, 1, 1) # [B, 34, 1, 1]
        final_off_mask = kpt_mask_spatial * kps_masks_expanded
        
        pred_offsets = preds['offsets'] * final_off_mask
        gt_offsets = gt_offsets * final_off_mask

        num_kpts_offsets = final_off_mask.sum()
        loss_off = self.l1_loss(pred_offsets, gt_offsets) / (num_kpts_offsets + 1e-8)
        
        # 5. 计算总损失
        total_loss = (self.hm_weight * loss_hm + 
                      self.ct_weight * loss_ct + 
                      self.reg_weight * loss_reg + 
                      self.off_weight * loss_off)
        
        loss_dict = {
            "total_loss": total_loss,
            "loss_heatmap": loss_hm,
            "loss_center": loss_ct,
            "loss_regs": loss_reg,
            "loss_offsets": loss_off,
        }
        
        return total_loss, loss_dict