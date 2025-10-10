import math

import torch


# ---- 6D -> R（稳健版，避免依赖外部命名差异）----
def _normalize(v, eps=1e-6):
    return v / (v.norm(dim=-1, keepdim=True) + eps)

def rotation_6d_to_matrix(x6: torch.Tensor) -> torch.Tensor:
    """
    x6: [B,6]  ->  R: [B,3,3]
    Zhou et al. 'On the Continuity of Rotation Representations in Neural Networks'
    """
    assert x6.dim() == 2 and x6.size(-1) == 6, f"Expect [B,6], got {tuple(x6.shape)}"
    a1 = x6[:, 0:3]
    a2 = x6[:, 3:6]
    b1 = _normalize(a1)
    b2 = _normalize(a2 - (b1 * a2).sum(dim=1, keepdim=True) * b1)
    b3 = torch.cross(b1, b2, dim=1)
    R = torch.stack((b1, b2, b3), dim=2)  # [B,3,3]
    return R

# ---- geodesic 角误差（度）----
@torch.no_grad()
def geodesic_angle_deg(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    """
    R_pred, R_gt: [B,3,3], 返回 [B] 的角度误差（度）
    """
    R_rel = R_pred @ R_gt.transpose(1, 2)
    trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
    skew = R_rel - R_rel.transpose(1, 2)
    v1 = skew[:, 2, 1] - skew[:, 1, 2]
    v2 = skew[:, 0, 2] - skew[:, 2, 0]
    v3 = skew[:, 1, 0] - skew[:, 0, 1]
    s = torch.sqrt((v1*v1 + v2*v2 + v3*v3).clamp_min(1e-12))
    c = (trace - 1.0).clamp(-2.0 + 1e-12, 2.0 - 1e-12)
    rad = torch.atan2(s, c)  # [B]
    return rad * (180.0 / math.pi)

@torch.no_grad()
def accuracy_at_threshold(errors_deg: torch.Tensor, thr_deg: float) -> float:
    return (errors_deg < thr_deg).float().mean().item()

@torch.no_grad()
def accuracy_curve_and_auc(errors_deg: torch.Tensor, max_deg: float = 30.0, step: float = 0.5):
    """
    画一条“Acc@δ vs δ”的曲线（δ ∈ [0, max_deg]），并计算 AUC（数值积分）。
    这相当于检测中的 PR→AP 的“角度版”总结：阈值从严到宽，综合你的整体准确性。
    """
    thrs = torch.arange(0.0, max_deg + 1e-9, step, device=errors_deg.device)
    accs = torch.stack([(errors_deg < t).float().mean() for t in thrs])  # [T]
    # 梯形积分（归一化到 [0,1]×[0,1] 区间）
    auc = torch.trapz(accs, thrs) / max_deg
    return thrs.cpu().numpy(), accs.cpu().numpy(), float(auc.item())
