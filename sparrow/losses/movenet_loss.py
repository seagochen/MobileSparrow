# sparrow/losses/movenet_loss.py (替换后的完整文件)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from tqdm import tqdm

# =========================
# Focal Loss (logits version)
# =========================
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
        p = torch.sigmoid(pred_logits)
        pt = p * target + (1.0 - p) * (1.0 - target)
        pt = pt.clamp(min=self.eps, max=1.0 - self.eps)
        focal = (1.0 - pt).pow(self.gamma) * bce
        alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        focal = alpha_t * focal
        return focal.mean()

# ===============
# COCO 骨段拓扑
# ===============
COCO_LIMBS = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 12), (5, 11), (6, 12), (11, 13), (13, 15),
    (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)
]

# ==========================
# MoveNet 损失 (正确版本)
# ==========================
class MoveNetLoss(nn.Module):
    def __init__(self,
                 focal_loss_alpha: float = 0.25,
                 focal_loss_gamma: float = 2.0,
                 hm_weight: float = 1.0,
                 ct_weight: float = 1.0,
                 reg_weight: float = 1.5,
                 off_weight: float = 1.0,
                 bone_weight: float = 0.15,
                 bg_weight: float = 0.0):
        super().__init__()
        self.hm_weight = hm_weight
        self.ct_weight = ct_weight
        self.reg_weight = reg_weight
        self.off_weight = off_weight
        self.bone_weight = bone_weight
        self.bg_weight = bg_weight # 当前版本未使用

        self.focal_loss = FocalLoss(alpha=focal_loss_alpha, gamma=focal_loss_gamma)
        self.l1_loss = nn.L1Loss(reduction='sum')

    def _bone_loss_heatmap(self, pred_hm: torch.Tensor, gt_hm: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        count = 0
        for a, b in COCO_LIMBS:
            pa = pred_hm[:, a]
            pb = pred_hm[:, b]
            ga = gt_hm[:, a]
            gb = gt_hm[:, b]
            loss = loss + F.mse_loss(pa - pb, ga - gb, reduction='mean')
            count += 1
        return loss / max(1, count)
    
    def find_max_loc(self, heatmap: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n, c, h, w = heatmap.shape
        heatmap_flat = heatmap.view(n, c, -1)
        _, max_id = torch.max(heatmap_flat, dim=2)
        y = max_id // w
        x = max_id % w
        return x, y

    def forward(self, preds: Dict[str, torch.Tensor], labels: torch.Tensor, kps_masks: torch.Tensor):
        gt_heatmaps = labels[:, :17, :, :]
        gt_centers  = labels[:, 17:18, :, :]
        gt_regs     = labels[:, 18:52, :, :]
        gt_offsets  = labels[:, 52:86, :, :]

        B, K, Hf, Wf = gt_heatmaps.shape
        device = gt_heatmaps.device

        loss_hm = self.focal_loss(preds['heatmaps'], gt_heatmaps)
        loss_ct = self.focal_loss(preds['centers'],  gt_centers)

        gt_cx, gt_cy = self.find_max_loc(gt_centers)
        gt_cx = gt_cx.squeeze(1)
        gt_cy = gt_cy.squeeze(1)

        batch_idx = torch.arange(B, device=device).long()
        kps_masks = kps_masks.float()

        sampled_gt_regs = gt_regs[batch_idx, :, gt_cy, gt_cx]
        sampled_pred_regs = preds['regs'][batch_idx, :, gt_cy, gt_cx]
        
        mask_reg = kps_masks.repeat_interleave(2, dim=1)
        loss_reg = self.l1_loss(sampled_pred_regs * mask_reg, sampled_gt_regs * mask_reg) / (mask_reg.sum() + 1e-4)

        gt_kps_x = (sampled_gt_regs[:, 0::2] + gt_cx.unsqueeze(-1)).round().long()
        gt_kps_y = (sampled_gt_regs[:, 1::2] + gt_cy.unsqueeze(-1)).round().long()
        gt_kps_x = torch.clamp(gt_kps_x, 0, Wf - 1)
        gt_kps_y = torch.clamp(gt_kps_y, 0, Hf - 1)

        loss_off = torch.tensor(0.0, device=device)
        for k in range(K):
            k_mask = kps_masks[:, k]
            if k_mask.sum() == 0:
                continue
            k_x, k_y = gt_kps_x[:, k], gt_kps_y[:, k]
            gt_off_x = gt_offsets[batch_idx, 2*k, k_y, k_x]
            gt_off_y = gt_offsets[batch_idx, 2*k+1, k_y, k_x]
            pred_off_x = preds['offsets'][batch_idx, 2*k, k_y, k_x]
            pred_off_y = preds['offsets'][batch_idx, 2*k+1, k_y, k_x]
            loss_off += self.l1_loss(pred_off_x * k_mask, gt_off_x * k_mask)
            loss_off += self.l1_loss(pred_off_y * k_mask, gt_off_y * k_mask)
        
        loss_off = loss_off / (kps_masks.sum() * 2 + 1e-4)

        loss_bone = self._bone_loss_heatmap(torch.sigmoid(preds['heatmaps']), gt_heatmaps) if self.bone_weight > 0.0 else torch.zeros((), device=device)
        
        total = (self.hm_weight * loss_hm + self.ct_weight * loss_ct +
                 self.reg_weight * loss_reg + self.off_weight * loss_off +
                 self.bone_weight * loss_bone)

        loss_dict = {
            "total_loss": total.detach(), "loss_heatmap": loss_hm.detach(),
            "loss_center": loss_ct.detach(), "loss_regs": loss_reg.detach(),
            "loss_offsets": loss_off.detach(), "loss_bone": loss_bone.detach(), "loss_bg": torch.zeros_like(total),
        }
        return total, loss_dict

# =================
# 评估 (正确版本)
# =================
@torch.no_grad()
def evaluate_local(
    model, dataloader, criterion, device,
    stride: int, decoder=None, pck_alpha: float = 0.05
):
    model_was_train = model.training
    model.eval()

    sum_total, sum_hm, sum_ct, sum_reg, sum_off, sum_bone, sum_bg, n_batches = [0.0] * 8
    use_pck, pck_hit, pck_cnt = decoder is not None, 0, 0

    pbar = tqdm(dataloader, desc="  🟡 [Validating] ")
    for imgs, labels, kps_masks, _ in pbar:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        kps_masks = kps_masks.to(device, non_blocking=True) # <-- kps_masks 保持 [B, 17]

        # --- 【关键修正】删除错误的掩码广播 ---
        # 新的 criterion 期望 [B, 17] 的稀疏掩码，不要在这里广播它
        
        preds = model(imgs)
        # 将原始的 [B, 17] 掩码传入
        total_loss, loss_dict = criterion(preds, labels, kps_masks)

        sum_total += float(loss_dict["total_loss"])
        # ... (累加其他损失项)
        sum_hm += float(loss_dict["loss_heatmap"]); sum_ct += float(loss_dict["loss_center"]);
        sum_reg += float(loss_dict["loss_regs"]); sum_off += float(loss_dict["loss_offsets"]);
        sum_bone += float(loss_dict.get("loss_bone", 0.0)); sum_bg += float(loss_dict.get("loss_bg", 0.0));
        n_batches += 1

        if use_pck:
            B, _, H_img, W_img = imgs.shape
            thr = pck_alpha * float(max(H_img, W_img))
            for b in range(B):
                single_out = {k: v[b:b+1] for k, v in preds.items()}
                dets = decoder(single_out, img_size=(H_img, W_img), stride=stride)
                if not dets: continue
                pred_kps = dets[0]["keypoints"]
                vis_mask = kps_masks[b] > 0.5 # [17]
                
                gt_hm_b = labels[b, :17]; gt_off_b = labels[b, 52:]; _, Hf2, Wf2 = gt_hm_b.shape
                
                for j in range(17):
                    if not vis_mask[j]: continue # 【修正】直接检查bool值
                    pck_cnt += 1
                    flat = torch.argmax(gt_hm_b[j].view(-1)); gy = int(flat // Wf2); gx = int(flat % Wf2)
                    dx = float(gt_off_b[2*j+0, gy, gx]); dy = float(gt_off_b[2*j+1, gy, gx])
                    gt_x, gt_y = (gx + dx) * stride, (gy + dy) * stride
                    pred_x, pred_y, pred_conf = pred_kps[j]
                    if pred_conf > 0.0:
                        dist = ((pred_x - gt_x)**2 + (pred_y - gt_y)**2) ** 0.5
                        if dist <= thr: pck_hit += 1

        # ... (pbar.set_postfix 更新显示)
    
    avg_total = sum_total / max(1, n_batches)
    avg_dict = {"total_loss": avg_total} # ... (填充其他平均损失)
    if use_pck: avg_dict[f"pck@{pck_alpha:.2f}"] = (pck_hit / max(1, pck_cnt))

    if model_was_train: model.train()
    return avg_total, avg_dict