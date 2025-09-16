import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Union

Tensor = torch.Tensor

class DBLossLite(nn.Module):
    """
    Lightweight DB loss (probability-only by default).
    Inputs:
        prob_map: (B,1,H,W) predicted text probability in [0,1]
        gt_prob:  (B,1,H,W) target prob (0/1 or soft), typically on shrunken polygons
        mask:     (B,1,H,W) training mask (1=valid, 0=ignore), optional
        thresh_map: (B,1,H,W) predicted threshold map in [0,1], optional
        gt_thresh:  (B,1,H,W) target threshold map, optional
    Returns:
        dict(loss=..., loss_bce=..., loss_dice=..., loss_l1=...)
    Notes:
        - This is a simplified version focusing on BCE + Dice; add threshold L1 with l1_ratio>0.
        - Normalize each term by valid pixels for stable scale across images.
    """
    def __init__(self, bce_ratio: float = 0.5, dice_ratio: float = 0.5, l1_ratio: float = 0.0, eps: float = 1e-6):
        super().__init__()
        self.bce_ratio = float(bce_ratio)
        self.dice_ratio = float(dice_ratio)
        self.l1_ratio  = float(l1_ratio)
        self.eps = float(eps)

    def forward(
        self,
        prob_map,
        gt_prob,
        mask=None,
        thresh_map=None,
        gt_thresh=None,
    ) -> Dict[str, torch.Tensor]:
        # BCE term
        bce = F.binary_cross_entropy(prob_map, gt_prob, reduction="none")
        if mask is not None:
            bce = bce * mask
            denom = mask.sum().clamp_min(self.eps)
        else:
            denom = torch.tensor(bce.numel(), device=prob_map.device, dtype=prob_map.dtype).clamp_min(self.eps)
        bce = bce.sum() / denom

        # Dice term (with optional masking)
        p = prob_map
        g = gt_prob
        if mask is not None:
            p = p * mask
            g = g * mask
        inter = (p * g).sum()
        union = p.sum() + g.sum()
        dice = 1.0 - (2.0 * inter + self.eps) / (union + self.eps)

        # L1 term for thresholds (optional)
        if (thresh_map is not None) and (gt_thresh is not None) and (self.l1_ratio > 0):
            l1 = (thresh_map - gt_thresh).abs()
            if mask is not None:
                l1 = l1 * mask
                denom_l1 = mask.sum().clamp_min(self.eps)
            else:
                denom_l1 = torch.tensor(l1.numel(), device=prob_map.device, dtype=prob_map.dtype).clamp_min(self.eps)
            l1 = l1.sum() / denom_l1
        else:
            l1 = prob_map.new_tensor(0.0)

        total = self.bce_ratio * bce + self.dice_ratio * dice + self.l1_ratio * l1
        return {"loss": total, "loss_bce": bce, "loss_dice": dice, "loss_l1": l1}


def _pack_targets(
    targets: List[Union[torch.Tensor, list, tuple]],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Flatten variable-length target sequences for CTC.
    Args:
        targets: list of int lists/tensors. Each element is a sequence WITHOUT blank ids.
        device: destination device
    Returns:
        (targets_flat, target_lengths)
    """
    flat = []
    lengths = []
    for t in targets:
        if isinstance(t, (list, tuple)):
            ids = torch.tensor(t, dtype=torch.long, device=device)
        elif isinstance(t, torch.Tensor):
            ids = t.to(device=device, dtype=torch.long)
        else:
            raise TypeError(f"Invalid target type: {type(t)}")
        lengths.append(int(ids.numel()))
        flat.append(ids)
    if len(flat) == 0:
        raise ValueError("Empty targets list")
    return torch.cat(flat, dim=0), torch.tensor(lengths, device=device, dtype=torch.long)


class CTCLossWrapper(nn.Module):
    """Thin wrapper around nn.CTCLoss for (B,T,C) logits.
    - Expects raw logits (unnormalized), shape (B,T,K).
    - Computes log_softmax and transposes to (T,B,K) as required by PyTorch.
    - targets: list of int sequences (no blanks).
    """
    def __init__(self, blank: int = 0, zero_infinity: bool = True, reduction: str = "mean"):
        super().__init__()
        self.ctc = nn.CTCLoss(blank=blank, zero_infinity=zero_infinity, reduction=reduction)

    def forward(
        self,
        logits,                    # (B,T,K)
        targets,                   # list[list[int]] or list[Tensor]
    ) -> Dict[str, torch.Tensor]:
        B, T, K = logits.shape
        device = logits.device
        logp = logits.log_softmax(dim=-1).permute(1, 0, 2).contiguous()  # (T,B,K)
        input_lengths = torch.full((B,), T, dtype=torch.long, device=device)
        targets_flat, target_lengths = _pack_targets(targets, device)
        loss = self.ctc(logp, targets_flat, input_lengths, target_lengths)
        return {"loss": loss}
