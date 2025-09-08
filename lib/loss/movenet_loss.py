"""
@Fire
改：适配新 MoveNet( dict 输出 ) & 动态特征图尺寸
"""

import os
import torch
import numpy as np
import torch.nn.functional as F

# 原来写死：_img_size=192, _feature_map_size=_img_size//4, 现在全部改为“运行时从张量尺寸推断”
_CENTER_WEIGHT_NPY = 'lib/data/center_weight_origin.npy'


class JointBoneLoss(torch.nn.Module):
    def __init__(self, joint_num):
        super().__init__()
        id_i, id_j = [], []
        for i in range(joint_num):
            for j in range(i + 1, joint_num):
                id_i.append(i)
                id_j.append(j)
        self.id_i = id_i
        self.id_j = id_j

    def forward(self, joint_out, joint_gt):
        # joint_out/joint_gt: [B, J, H, W] (heatmaps) — 用热力图的成对骨段差的范数作为“骨段一致性”(代理)
        J = torch.norm(joint_out[:, self.id_i, :, :] - joint_out[:, self.id_j, :, :], p=2, dim=(-2, -1))
        Y = torch.norm(joint_gt[:, self.id_i, :, :] - joint_gt[:, self.id_j, :, :], p=2, dim=(-2, -1))
        loss = torch.abs(J - Y)
        return torch.sum(loss) / joint_out.shape[0] / len(self.id_i)


def _make_center_weight(hw, device, dtype):
    """按当前特征图尺寸生成中心加权图（值域 0~1，中心高权重）。"""
    H, W = hw
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device, dtype=dtype),
        torch.linspace(-1, 1, W, device=device, dtype=dtype),
        indexing="ij"
    )
    rr2 = xx ** 2 + yy ** 2
    # 高斯：中心 1，边缘趋近 0
    weight = torch.exp(-rr2 / 0.15)  # sigma^2 ~= 0.15，可按需调
    return weight  # [H, W]


class MovenetLoss(torch.nn.Module):
    def __init__(self, use_target_weight=False, target_weight=[1]):
        super().__init__()
        self.use_target_weight = use_target_weight
        self.target_weight = target_weight
        self.boneloss = JointBoneLoss(17)

        # 运行时缓存中心权重（按尺寸）
        self._center_cache = None  # (H, W, tensor)

        # 兼容旧逻辑：尝试读取 npy（若存在且尺寸匹配则用，否则动态生成）
        self._center_weight_npy = _CENTER_WEIGHT_NPY

    # ---------- 小工具 ----------
    def l1(self, pre, target, kps_mask):
        # pre/target: [B], kps_mask: [B]
        return torch.sum(torch.abs(pre - target) * kps_mask) / (kps_mask.sum() + 1e-4)

    def myMSEwithWeight(self, pre, target):
        # pre/target: [B, C, H, W], target ∈ [0,1]
        loss = torch.pow((pre - target), 2)
        weight_mask = target * 8 + 1  # 前景更重
        loss = loss * weight_mask
        return torch.sum(loss) / (target.shape[0] * target.shape[1])

    def heatmapL1(self, pre, target):
        loss = torch.abs(pre - target)
        weight_mask = target * 4 + 1
        loss = loss * weight_mask
        return torch.sum(loss) / (target.shape[0] * target.shape[1])

    def centernet_focal(self, pred, gt):
        pos = gt.eq(1).float()
        neg = gt.lt(1).float()
        neg_weights = torch.pow(1 - gt, 4)
        pos_loss = torch.log(pred + 1e-6) * torch.pow(1 - pred, 2) * pos
        neg_loss = torch.log(1 - pred + 1e-6) * torch.pow(pred, 2) * neg_weights * neg
        num_pos = pos.sum()
        if num_pos == 0:
            return -neg_loss.sum()
        return -(pos_loss.sum() + neg_loss.sum()) / num_pos

    def _ensure_center_weight(self, B, H, W, device, dtype):
        """
        返回形状为 [B,1,H,W] 的中心权重图。
        优先使用 npy（尺寸匹配）；否则按尺寸生成并缓存。
        """
        need_regen = True
        if self._center_cache is not None:
            h0, w0, ten = self._center_cache
            if (h0, w0) == (H, W) and ten.device == device and ten.dtype == dtype:
                need_regen = False

        if need_regen:
            weight = None
            if os.path.isfile(self._center_weight_npy):
                try:
                    arr = np.load(self._center_weight_npy)
                    if arr.ndim == 2 and arr.shape == (H, W):
                        weight = torch.from_numpy(arr).to(device=device, dtype=dtype)
                except Exception:
                    weight = None
            if weight is None:
                weight = _make_center_weight((H, W), device, dtype)

            self._center_cache = (H, W, weight)

        # [H,W] -> [B,1,H,W]
        return self._center_cache[2].view(1, 1, H, W).repeat(B, 1, 1, 1).requires_grad_(False)

    # ---------- 主损失 ----------
    def heatmapLoss(self, pred, target, batch_size):
        return self.myMSEwithWeight(pred, target)

    def centerLoss(self, pred, target, batch_size):
        return self.myMSEwithWeight(pred, target)

    def regsLoss(self, pred, target, cx0, cy0, kps_mask, batch_size, num_joints):
        # 逐关节点在 (cy0,cx0) 位置取回归值（dx,dy），做 L1
        _dim0 = torch.arange(0, batch_size, device=pred.device).long()
        _dim1 = torch.zeros(batch_size, device=pred.device).long()
        loss = 0
        for idx in range(num_joints):
            gt_x = target[_dim0, _dim1 + idx * 2, cy0, cx0]
            gt_y = target[_dim0, _dim1 + idx * 2 + 1, cy0, cx0]
            pre_x = pred[_dim0, _dim1 + idx * 2, cy0, cx0]
            pre_y = pred[_dim0, _dim1 + idx * 2 + 1, cy0, cx0]
            loss += self.l1(pre_x, gt_x, kps_mask[:, idx])
            loss += self.l1(pre_y, gt_y, kps_mask[:, idx])
        return loss / num_joints

    def offsetLoss(self, pred, target, cx0, cy0, regs, kps_mask, batch_size, num_joints, H, W):
        _dim0 = torch.arange(0, batch_size, device=pred.device).long()
        _dim1 = torch.zeros(batch_size, device=pred.device).long()
        loss = 0
        for idx in range(num_joints):
            gt_x = (regs[_dim0, _dim1 + idx * 2, cy0, cx0].long() + cx0).clamp_(0, W - 1)
            gt_y = (regs[_dim0, _dim1 + idx * 2 + 1, cy0, cx0].long() + cy0).clamp_(0, H - 1)

            gt_off_x = target[_dim0, _dim1 + idx * 2, gt_y, gt_x]
            gt_off_y = target[_dim0, _dim1 + idx * 2 + 1, gt_y, gt_x]
            pre_off_x = pred[_dim0, _dim1 + idx * 2, gt_y, gt_x]
            pre_off_y = pred[_dim0, _dim1 + idx * 2 + 1, gt_y, gt_x]

            loss += self.l1(pre_off_x, gt_off_x, kps_mask[:, idx])
            loss += self.l1(pre_off_y, gt_off_y, kps_mask[:, idx])
        return loss / num_joints

    def _maxPointPth(self, heatmap, center_weight=None):
        """
        heatmap: [B,1,H,W]
        center_weight: [B,1,H,W] or None
        返回坐标 (x,y): LongTensor [B]
        """
        if center_weight is not None:
            heatmap = heatmap * center_weight
        n, c, h, w = heatmap.shape
        flat = heatmap.view(n, -1)
        _, max_id = torch.max(flat, dim=1)
        y = (max_id // w).long()
        x = (max_id % w).long()
        return x, y

    def forward(self, output, target, kps_mask=None):
        """
        output:
          - 新模型(dict): {"heatmaps": [B,J,H,W], "centers":[B,1,H,W], "regs":[B,2J,H,W], "offsets":[B,2J,H,W]}
          - 兼容旧(list/tuple): [heatmaps, centers, regs, offsets]
        target: [B, 17 + 1 + 34 + 34, H, W]  (与 pred 同 H,W)
        kps_mask: [B, 17]，若为 None 则默认全 1
        """
        # --- 1) 解包预测 ---
        if isinstance(output, dict):
            heatmaps_p = output["heatmaps"]
            centers_p = output["centers"]
            regs_p = output["regs"]
            offsets_p = output["offsets"]
        else:
            heatmaps_p, centers_p, regs_p, offsets_p = output

        B, J, H, W = heatmaps_p.shape
        if kps_mask is None:
            kps_mask = torch.ones((B, J), device=heatmaps_p.device, dtype=heatmaps_p.dtype)

        # --- 2) 解析目标 ---
        # 约定 target channel 排布：17(hm) | 1(center) | 34(regs) | 34(offsets)
        heatmaps_t = target[:, :J, :, :]
        centers_t  = target[:, J:J+1, :, :]
        regs_t     = target[:, J+1:J+1+2*J, :, :]
        offsets_t  = target[:, J+1+2*J:, :, :]

        # --- 3) 中心权重（按尺寸） ---
        center_weight = self._ensure_center_weight(B, H, W, device=target.device, dtype=target.dtype)

        # --- 4) 各项损失 ---
        heatmap_loss = self.heatmapLoss(heatmaps_p, heatmaps_t, B)
        bone_loss    = self.boneloss(heatmaps_p, heatmaps_t)  # 代理骨段一致性
        center_loss  = self.centerLoss(centers_p, centers_t, B)

        # 最大中心位置（在 GT center 上取，以稳定回归）
        cx0, cy0 = self._maxPointPth(centers_t, center_weight=center_weight)
        cx0 = cx0.clamp_(0, W - 1)
        cy0 = cy0.clamp_(0, H - 1)

        regs_loss   = self.regsLoss(regs_p, regs_t, cx0, cy0, kps_mask, B, J)
        offset_loss = self.offsetLoss(offsets_p, offsets_t, cx0, cy0, regs_t, kps_mask, B, J, H, W)

        return [heatmap_loss, bone_loss, center_loss, regs_loss, offset_loss]


# 兼容你原来的全局实例 & 入口
movenetLoss = MovenetLoss(use_target_weight=False)

def calculate_loss(predict, label, kps_mask=None):
    return movenetLoss(predict, label, kps_mask)
