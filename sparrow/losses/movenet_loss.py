import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

# =========================
# Focal Loss (logits version)
# =========================
class FocalLoss(nn.Module):
    """
    带 Logits 输入的 Focal Loss 实现（稳定版）。
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # BCE with logits（逐元素）
        bce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')

        # pt = p*yt + (1-p)*(1-yt)
        p = torch.sigmoid(pred_logits)
        pt = p * target + (1.0 - p) * (1.0 - target)
        pt = pt.clamp(min=self.eps, max=1.0 - self.eps)

        # focal weighting
        focal = (1.0 - pt).pow(self.gamma) * bce

        # alpha weighting
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
# MoveNet 损失（含骨架/背景）
# ==========================
class MoveNetLoss(nn.Module):
    def __init__(self,
                 focal_loss_alpha: float = 0.25,
                 focal_loss_gamma: float = 2.0,
                 hm_weight: float = 1.0,
                 ct_weight: float = 1.0,
                 reg_weight: float = 2.0,
                 off_weight: float = 1.0,
                 bone_weight: float = 0.15,   # 建议 0.10~0.20
                 bg_weight: float = 0.0):     # 默认关闭
        super().__init__()
        self.hm_weight = hm_weight
        self.ct_weight = ct_weight
        self.reg_weight = reg_weight
        self.off_weight = off_weight
        self.bone_weight = bone_weight
        self.bg_weight = bg_weight

        self.focal_loss = FocalLoss(alpha=focal_loss_alpha, gamma=focal_loss_gamma)
        self.l1_loss = nn.L1Loss(reduction='sum')  # 用 sum 再按有效像素数归一

    # ------------- 辅助项：骨架一致性 ----------------
    def _bone_loss_heatmap(self, pred_hm: torch.Tensor, gt_hm: torch.Tensor) -> torch.Tensor:
        """
        在热力图通道上做骨段一致性：
        对每条骨段 (a,b)，计算 (pred[a]-pred[b]) 与 (gt[a]-gt[b]) 的 MSE。
        pred_hm / gt_hm : [B, 17, H, W]
        """
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

    # ------------- 辅助项：背景一致性（可选） ----------------
    def _bg_loss_heatmap(self, pred_hm: torch.Tensor, gt_hm: torch.Tensor) -> torch.Tensor:
        """
        背景一致性（可选）：
          - 各关键点通道通过 sigmoid 后求和得到前景置信 pred_fg
          - 背景 = 1 - clip(pred_fg, 0, 1)，与 gt 背景做 L2
        """
        pred_fg = pred_hm.sigmoid().sum(dim=1)              # [B, H, W]
        pred_bg = 1.0 - pred_fg.clamp(0.0, 1.0)

        gt_fg = gt_hm.clamp(0.0, 1.0).sum(dim=1)
        gt_bg = 1.0 - gt_fg.clamp(0.0, 1.0)

        return F.mse_loss(pred_bg, gt_bg, reduction='mean')

    # ----------------------- 前向总损失 -----------------------
    def forward(self, preds: Dict[str, torch.Tensor], labels: torch.Tensor, kps_masks: torch.Tensor):
        """
        只支持新版本：
          - labels:  [B, (17+1+34+34), Hf, Wf]，顺序 = heatmaps|centers|regs|offsets
          - kps_masks: [B, 17, Hf, Wf]  （逐像素关键点 mask）
        """
        # 1) 解析标签
        gt_heatmaps = labels[:, :17, :, :]
        gt_centers  = labels[:, 17:18, :, :]
        gt_regs     = labels[:, 18:52, :, :]
        gt_offsets  = labels[:, 52:, :, :]

        B, K, H, W = gt_heatmaps.shape
        device = gt_heatmaps.device

        # ★ 严格检查 kps_masks 形状
        assert kps_masks is not None, "kps_masks is None; new dataloader must provide [B,17,Hf,Wf] masks."
        assert kps_masks.dim() == 4 and kps_masks.shape == (B, 17, H, W), \
            f"kps_masks shape must be [B,17,Hf,Wf], got {tuple(kps_masks.shape)}"

        # 2) 热图/中心（Focal）
        loss_hm = self.focal_loss(preds['heatmaps'], gt_heatmaps)
        loss_ct = self.focal_loss(preds['centers'],  gt_centers)

        # 3) 稀疏 L1（只在 mask==1 的网格上）
        #    将 17 关节 mask 在 x/y 两分量上各复制一次 -> [B,34,H,W]
        kps_mask_pix = kps_masks.float().to(device)
        mask_xy = kps_mask_pix.unsqueeze(1).repeat(1, 2, 1, 1, 1)  # [B,2,17,H,W]
        mask_xy = mask_xy.reshape(B, -1, H, W)                     # [B,34,H,W]
        n_reg = mask_xy.sum().clamp_min(1.0)

        loss_reg = (torch.abs(preds['regs'] - gt_regs) * mask_xy).sum() / n_reg
        loss_off = (torch.abs(preds['offsets'] - gt_offsets) * mask_xy).sum() / n_reg

        # 4) 辅助项
        loss_bone = self._bone_loss_heatmap(preds['heatmaps'], gt_heatmaps) if self.bone_weight > 0.0 else torch.zeros((), device=device)
        loss_bg   = self._bg_loss_heatmap(preds['heatmaps'], gt_heatmaps)   if self.bg_weight   > 0.0 else torch.zeros((), device=device)

        # 5) 总损失
        total = (self.hm_weight * loss_hm
                 + self.ct_weight * loss_ct
                 + self.reg_weight * loss_reg
                 + self.off_weight * loss_off
                 + self.bone_weight * loss_bone
                 + self.bg_weight   * loss_bg)

        loss_dict = {
            "total_loss":   total.detach(),
            "loss_heatmap": loss_hm.detach(),
            "loss_center":  loss_ct.detach(),
            "loss_regs":    loss_reg.detach(),
            "loss_offsets": loss_off.detach(),
            "loss_bone":    loss_bone.detach(),
            "loss_bg":      loss_bg.detach(),
        }
        return total, loss_dict


# =================
# 评估（带可选PCK）
# =================
from tqdm import tqdm


@torch.no_grad()
def evaluate_local(
    model,
    dataloader,
    criterion,                 # MoveNetLoss 实例
    device,
    stride: int = 8,
    decoder=None,
    pck_alpha: float = 0.05
):
    model_was_train = model.training
    model.eval()

    sum_total = 0.0
    sum_hm = 0.0
    sum_ct = 0.0
    sum_reg = 0.0
    sum_off = 0.0
    sum_bone = 0.0
    sum_bg = 0.0
    n_batches = 0

    use_pck = decoder is not None
    pck_hit = 0
    pck_cnt = 0

    pbar = tqdm(dataloader, desc="  🟡 [Validating] ")
    for imgs, labels, kps_masks, _ in pbar:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        kps_masks = kps_masks.to(device, non_blocking=True)

        # --- 广播 kps_masks 到 [B,17,Hf,Wf] ---
        Hf, Wf = labels.shape[-2], labels.shape[-1]
        if kps_masks.dim() == 2 and kps_masks.shape[1] == 17:
            kps_masks = kps_masks[:, :, None, None].float().expand(-1, -1, Hf, Wf).contiguous()
        elif kps_masks.dim() == 4 and kps_masks.shape[2] == 1 and kps_masks.shape[3] == 1:
            kps_masks = kps_masks.float().expand(-1, -1, Hf, Wf).contiguous()

        preds = model(imgs)
        total_loss, loss_dict = criterion(preds, labels, kps_masks)

        sum_total += float(loss_dict["total_loss"])
        sum_hm    += float(loss_dict["loss_heatmap"])
        sum_ct    += float(loss_dict["loss_center"])
        sum_reg   += float(loss_dict["loss_regs"])
        sum_off   += float(loss_dict["loss_offsets"])
        sum_bone  += float(loss_dict.get("loss_bone", 0.0))
        sum_bg    += float(loss_dict.get("loss_bg",   0.0))
        n_batches += 1

        # —— 可选 PCK（与原 evaluate 一致）——
        if use_pck:
            B, _, H_img, W_img = imgs.shape
            thr = pck_alpha * float(max(H_img, W_img))
            for b in range(B):
                single_out = {
                    "heatmaps": preds["heatmaps"][b:b+1],
                    "centers":  preds["centers"][b:b+1],
                    "regs":     preds["regs"][b:b+1],
                    "offsets":  preds["offsets"][b:b+1],
                }
                dets = decoder(single_out, img_size=(H_img, W_img), stride=stride, topk_centers=1)
                if len(dets) == 0:
                    continue
                pred_kps = dets[0]["keypoints"]

                gt_hm = labels[b, :17]
                gt_off= labels[b, 52:]
                K, Hf2, Wf2 = gt_hm.shape
                vis_mask = (kps_masks[b] > 0.5)

                for j in range(K):
                    if not bool(vis_mask[j].any()):
                        continue
                    flat = torch.argmax(gt_hm[j].view(-1))
                    gy = int(flat // Wf2)
                    gx = int(flat %  Wf2)
                    dx = float(gt_off[2*j+0, gy, gx])
                    dy = float(gt_off[2*j+1, gy, gx])
                    gt_x = (gx + dx) * stride
                    gt_y = (gy + dy) * stride

                    pred_x, pred_y, pred_conf = pred_kps[j]
                    if pred_conf <= 0.0:
                        pck_cnt += 1
                        continue
                    dist = ((pred_x - gt_x)**2 + (pred_y - gt_y)**2) ** 0.5
                    pck_hit += 1 if dist <= thr else 0
                    pck_cnt += 1

        pbar.set_postfix(
            tot=f"{sum_total/max(1,n_batches):.6f}",
            hm=f"{sum_hm/max(1,n_batches):.6f}",
            ct=f"{sum_ct/max(1,n_batches):.6f}",
            reg=f"{sum_reg/max(1,n_batches):.6f}",
            off=f"{sum_off/max(1,n_batches):.6f}",
            bone=f"{sum_bone/max(1,n_batches):.6f}",
            bg=f"{sum_bg/max(1,n_batches):.6f}",
            pck=(f"{(100.0*pck_hit/max(1,pck_cnt)):.2f}%" if use_pck else "N/A")
        )

    avg_total = sum_total / max(1, n_batches)
    avg_dict = {
        "total_loss":   avg_total,
        "loss_heatmap": sum_hm  / max(1, n_batches),
        "loss_center":  sum_ct  / max(1, n_batches),
        "loss_regs":    sum_reg / max(1, n_batches),
        "loss_offsets": sum_off / max(1, n_batches),
        "loss_bone":    sum_bone / max(1, n_batches),
        "loss_bg":      sum_bg   / max(1, n_batches),
    }
    if use_pck:
        avg_dict[f"pck@{pck_alpha:.2f}"] = (pck_hit / max(1, pck_cnt)) if pck_cnt > 0 else 0.0

    if model_was_train:
        model.train()
    return avg_total, avg_dict
