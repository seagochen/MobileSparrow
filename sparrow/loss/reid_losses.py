# reid_losses.py
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- utils ----------
def pairwise_cosine_dist(x: torch.Tensor) -> torch.Tensor:
    # x: BxD (assume already L2-normalized)
    # return: BxB cosine distance in [0,2]
    sim = x @ x.t().clamp(-1, 1)
    return 1.0 - sim  # distance = 1 - cosine_sim

def label_smoothing_ce(logits: torch.Tensor, targets: torch.Tensor, eps: float = 0.1):
    # logits: BxC (unnormalized), targets: B (long)
    num_classes = logits.size(1)
    log_probs = F.log_softmax(logits, dim=1)
    with torch.no_grad():
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(eps / (num_classes - 1))
        true_dist.scatter_(1, targets.unsqueeze(1), 1 - eps)
    return torch.mean(torch.sum(-true_dist * log_probs, dim=1))

# ---------- CosFace / ArcFace heads ----------
class CosMarginHead(nn.Module):
    """
    CosFace / AM-Softmax:  logits = s * (cos(theta) - m) for target class
    """
    def __init__(self, emb_dim: int, num_classes: int, m: float = 0.35, s: float = 30.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, emb_dim))
        nn.init.xavier_normal_(self.weight)
        self.m = m
        self.s = s

    def forward(self, emb: torch.Tensor, targets: torch.Tensor):
        # emb: BxD (assume L2-normalized), weight will be normalized on the fly
        W = F.normalize(self.weight, dim=1)               # CxD
        # 设备对齐（避免 CPU/GPU 不一致导致 F.linear 报错）
        if W.device != emb.device:
            W = W.to(emb.device)
        cos = F.linear(emb, W)                            # BxC
        # 目标索引检查（避免 CUDA scatter_ 内核越界断言）
        targets = targets.long()
        C = cos.size(1)
        if (targets.min() < 0) or (targets.max() >= C):
            raise IndexError(f"targets index out of range: min={int(targets.min())}, "
                             f"max={int(targets.max())}, num_classes={C}")
        # subtract margin on target logits
        one_hot = torch.zeros_like(cos).scatter_(1, targets.view(-1,1), 1.0)
        logits = self.s * (cos - one_hot * self.m)
        return logits

class ArcMarginHead(nn.Module):
    """
    ArcFace: logits = s * cos(theta + m)
    """
    def __init__(self, emb_dim: int, num_classes: int, m: float = 0.50, s: float = 30.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, emb_dim))
        nn.init.xavier_normal_(self.weight)
        self.m = m
        self.s = s

    def forward(self, emb: torch.Tensor, targets: torch.Tensor):
        W = F.normalize(self.weight, dim=1)               # CxD
        # 设备对齐
        if W.device != emb.device:
            W = W.to(emb.device)
        cos = F.linear(emb, W).clamp(-1, 1)               # BxC
        # 目标索引检查
        targets = targets.long()
        C = cos.size(1)
        if (targets.min() < 0) or (targets.max() >= C):
            raise IndexError(f"targets index out of range: min={int(targets.min())}, "
                             f"max={int(targets.max())}, num_classes={C}")
        sin = torch.sqrt((1.0 - cos**2).clamp_min(1e-6))
        cos_m = torch.cos(torch.tensor(self.m, device=emb.device))
        sin_m = torch.sin(torch.tensor(self.m, device=emb.device))
        cos_target = cos.gather(1, targets.view(-1,1))
        # cos(theta + m) = cos*cos_m - sin*sin_m
        phi = cos_target * cos_m - sin.gather(1, targets.view(-1,1)) * sin_m

        logits = cos.clone()
        logits.scatter_(1, targets.view(-1,1), phi)
        logits = self.s * logits
        return logits

# ---------- Triplet (batch-hard) ----------
class BatchHardTriplet(nn.Module):
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    @torch.no_grad()
    def _build_masks(self, labels: torch.Tensor):
        B = labels.size(0)
        eq = labels.view(B,1).eq(labels.view(1,B))
        pos_mask = eq ^ torch.eye(B, dtype=torch.bool, device=labels.device)
        neg_mask = ~eq
        return pos_mask, neg_mask

    def forward(self, emb: torch.Tensor, labels: torch.Tensor):
        # emb: BxD (L2-normalized)
        dist = pairwise_cosine_dist(emb)  # BxB
        pos_mask, neg_mask = self._build_masks(labels)

        # hardest positive: max distance among same ID (exclude self)
        pos_dist = dist.clone()
        pos_dist[~pos_mask] = -1.0
        hardest_pos, _ = pos_dist.max(dim=1)

        # hardest negative: min distance among different ID
        neg_dist = dist.clone()
        neg_dist[~neg_mask] = 10.0
        hardest_neg, _ = neg_dist.min(dim=1)

        # some classes may have no positives (K=1) → mask them out
        valid = (hardest_pos >= 0.0) & (hardest_neg < 10.0)
        if valid.any():
            y = torch.ones_like(hardest_pos[valid])
            loss = self.ranking_loss(hardest_neg[valid], hardest_pos[valid], y)
        else:
            loss = emb.sum() * 0.0  # zero, keep graph
        return loss

# ---------- Top-level criterion ----------
class ReIDCriterion(nn.Module):
    """
    Combine ID loss (ArcFace/CosFace) and Triplet.
    Usage:
        crit = ReIDCriterion(num_classes, emb_dim=128, id_head='cosface', m=0.35, s=30.0,
                             w_id=1.0, w_tri=1.0, smooth_eps=0.1)
        out = model(images)          # Bx128 (L2-norm)
        loss, log = crit(out, labels)
    """
    def __init__(self,
                 num_classes: int,
                 emb_dim: int = 128,
                 id_head: Literal["cosface", "arcface"] = "cosface",
                 m: float = 0.35,
                 s: float = 30.0,
                 w_id: float = 1.0,
                 w_tri: float = 1.0,
                 smooth_eps: float = 0.1,
                 triplet_margin: float = 0.3):
        super().__init__()

        if id_head == "cosface":
            self.head = CosMarginHead(emb_dim, num_classes, m=m, s=s)
        elif id_head == 'arcface':
            # 对应默认 m=0.5
            if m == 0.35: m = 0.50
            self.head = ArcMarginHead(emb_dim, num_classes, m=m, s=s)
        else:
            raise ValueError("id_head must be 'cosface' or 'arcface'")

        self.tri = BatchHardTriplet(margin=triplet_margin)
        self.w_id = w_id
        self.w_tri = w_tri
        self.smooth_eps = smooth_eps

    def forward(self, emb: torch.Tensor, labels: torch.Tensor):
        # Classification logits with margin
        logits = self.head(emb, labels)
        id_loss = label_smoothing_ce(logits, labels, eps=self.smooth_eps) if self.smooth_eps > 0 else F.cross_entropy(logits, labels)
        tri_loss = self.tri(emb, labels)

        loss = self.w_id * id_loss + self.w_tri * tri_loss
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            acc = (pred == labels).float().mean()
        log = {
            "loss": loss.detach().item(),
            "id_loss": id_loss.detach().item(),
            "tri_loss": tri_loss.detach().item(),
            "acc1": acc.item()
        }
        return loss, log
