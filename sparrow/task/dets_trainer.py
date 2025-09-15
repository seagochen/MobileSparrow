# det_trainer_bt.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from sparrow.loss.ssdlite_loss import SSDLoss, pack_targets_for_ssd, generate_ssd_anchors
from sparrow.task.base_trainer import BaseTrainer

#
# @torch.no_grad()
# def generate_ssd_anchors(img_size: int,
#                          feat_shapes: List[Tuple[int, int]],
#                          strides: List[int],
#                          ratios: Tuple[float, ...] = (1.0, 2.0, 0.5),
#                          scales: Tuple[float, ...] = (1.0, 1.26)) -> torch.Tensor:
#     device = torch.device('cpu')
#     all_anchors = []
#     for (H, W), s in zip(feat_shapes, strides):
#         ys = (torch.arange(H, device=device) + 0.5) * s
#         xs = (torch.arange(W, device=device) + 0.5) * s
#         cy, cx = torch.meshgrid(ys, xs, indexing='ij')  # [H,W]
#         cx = (cx / img_size).reshape(-1, 1)
#         cy = (cy / img_size).reshape(-1, 1)
#         anchors_lvl = []
#         for r in ratios:
#             for sc in scales:
#                 w = (sc * s * math.sqrt(r)) / img_size
#                 h = (sc * s / math.sqrt(r)) / img_size
#                 wh = torch.full((H * W, 2), 0.0, device=device)
#                 wh[:, 0] = w; wh[:, 1] = h
#                 anchors_lvl.append(torch.cat([cx, cy, wh], dim=1))  # [H*W,4]
#         all_anchors.append(torch.cat(anchors_lvl, dim=0))  # [H*W*A,4]
#     return torch.cat(all_anchors, dim=0)  # [A_total,4]
#
#
# def pack_targets_for_ssd(targets_list: List[torch.Tensor], img_size: int) -> Dict[str, List[torch.Tensor]]:
#     boxes_norm, labels = [], []
#     for t in targets_list:
#         if t.numel() == 0:
#             boxes_norm.append(t.new_zeros((0, 4)))
#             labels.append(torch.zeros((0,), dtype=torch.long, device=t.device))
#             continue
#         cls = t[:, 0].long() + 1              # 0..C-1 -> 1..C-1（0留给背景）
#         b   = (t[:, 1:5] / float(img_size)).clamp(0, 1)
#         boxes_norm.append(b)
#         labels.append(cls)
#     return {"boxes": boxes_norm, "labels": labels}


def _infer_feat_shapes(img_size: int, strides: Tuple[int, int, int]) -> List[Tuple[int, int]]:
    assert img_size % max(strides) == 0, f"img_size({img_size}) 应被最大 stride 整除"
    return [(img_size // s, img_size // s) for s in strides]


class DetsTrainer(BaseTrainer):
    """
    - 训练：_calculate_loss -> 总损失 + {'loss','loss_cls','loss_reg',...}
    - 验证：_calculate_metrics -> 同上（在 EMA/模型上计算）
    - main metric：-loss（越大越好，沿用 BaseTrainer 逻辑）
    - dataloader batch：(imgs, targets_list, img_paths)
    """

    def __init__(self, model: nn.Module,
                 *,
                 epochs: int,
                 save_dir: str,
                 img_size: int,
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 optimizer_cfg: Dict = None,
                 scheduler_name: str = "MultiStepLR",
                 milestones=None,
                 gamma: float = 0.1,
                 step_size: int = 30,
                 mode: str = "max",
                 factor: float = 0.5,
                 patience: int = 5,
                 min_lr: float = 1e-6,
                 T_0: int = 10,
                 T_mult: int = 1,
                 last_epoch: int = -1,
                 use_amp: bool = True,
                 use_ema: bool = True,
                 ema_decay: float = 0.9998,
                 clip_grad_norm: float = 0.0,
                 log_interval: int = 10,
                 alpha_reg: float = 1.0):

        # ⭐️⭐️⭐️ 关键改动：在 super() 之前保存原始模型 ⭐️⭐️⭐️
        original_model = model

        super().__init__(
            model,
            epochs=epochs,
            save_dir=save_dir,
            device=device,
            optimizer_cfg=optimizer_cfg or {},
            scheduler_name=scheduler_name,
            milestones=milestones, gamma=gamma, step_size=step_size,
            mode=mode, factor=factor, patience=patience, min_lr=min_lr,
            T_0=T_0, T_mult=T_mult, last_epoch=last_epoch,
            use_amp=use_amp, use_ema=use_ema, ema_decay=ema_decay,
            clip_grad_norm=clip_grad_norm, log_interval=log_interval
        )

        # 确保原始模型有所需属性
        assert hasattr(original_model, "ratios"), "Model must have 'ratios' attribute."
        assert hasattr(original_model, "scales"), "Model must have 'scales' attribute."
        assert hasattr(original_model, "strides"), "Model must have 'strides' attribute."

        self.img_size = int(img_size)

        # ⭐️⭐️⭐️ 关键改动：从 original_model 获取配置 ⭐️⭐️⭐️
        self.ratios   = original_model.ratios
        self.scales   = original_model.scales
        self.strides  = original_model.strides

        # 损失函数（确保 model.num_classes 含背景类）
        self.loss_func = SSDLoss(num_classes=getattr(self.model, "num_classes", None) or 81,
                                 alpha=alpha_reg)

        # 预生成 anchors（cxcywh，归一化）
        self.feat_shapes = _infer_feat_shapes(self.img_size, self.strides)
        anchors = generate_ssd_anchors(self.img_size, self.feat_shapes, list(self.strides),
                                       ratios=self.ratios, scales=self.scales)

        # self.register_buffer("anchors", anchors.to(self.device), persistent=False)  # [A,4]
        self.anchors = anchors.to(self.device)  # [A,4]

    def _calculate_loss(self, batch) -> Tuple[torch.Tensor, Dict[str, float]]:
        imgs, targets_list, _ = batch

        out = self.model(imgs)
        self._maybe_build_anchors_from_output(out["cls_logits"])
        
        cls_logits = torch.cat(out["cls_logits"], dim=1)  # [B,A,C]
        bbox_regs  = torch.cat(out["bbox_regs"],  dim=1)  # [B,A,4]

        targets = pack_targets_for_ssd(targets_list, img_size=self.img_size)
        targets["boxes"]  = [b.to(self.device) for b in targets["boxes"]]
        targets["labels"] = [l.to(self.device) for l in targets["labels"]]

        loss, loss_dict = self.loss_func(cls_logits, bbox_regs, self.anchors, targets)
        loss_dict = {"loss": float(loss.detach().cpu()), **loss_dict}
        return loss, loss_dict

    @torch.no_grad()
    def _calculate_metrics(self, model, batch) -> Dict[str, float]:
        imgs, targets_list, _ = batch

        out = model(imgs)
        self._maybe_build_anchors_from_output(out["cls_logits"])

        cls_logits = torch.cat(out["cls_logits"], dim=1)
        bbox_regs  = torch.cat(out["bbox_regs"],  dim=1)

        targets = pack_targets_for_ssd(targets_list, img_size=self.img_size)
        targets["boxes"]  = [b.to(self.device) for b in targets["boxes"]]
        targets["labels"] = [l.to(self.device) for l in targets["labels"]]

        loss, loss_dict = self.loss_func(cls_logits, bbox_regs, self.anchors, targets)
        return {"loss": float(loss.detach().cpu()), **{k: float(v) for k, v in loss_dict.items()}}

    def _get_main_metric(self, metrics: Dict[str, float]) -> float:
        return -metrics.get("loss", float('inf'))

    def _move_batch_to_device(self, batch):
        imgs, targets_list, img_paths = batch
        imgs = imgs.to(self.device, non_blocking=True)
        return imgs, targets_list, img_paths

    def _maybe_build_anchors_from_output(self, cls_logits_list: List[torch.Tensor]) -> None:
        """
        根据实际输出长度，稳健地推断每层 H、W、stride，并生成 anchors。
        A_per_loc = len(ratios) * len(scales)
        L_i = H_i * W_i * A_per_loc
        """
        A_per_loc = len(self.ratios) * len(self.scales)
        # 已有且匹配则直接返回
        total_L = sum(int(t.shape[1]) for t in cls_logits_list)
        if self.anchors is not None and self.anchors.shape[0] == total_L:
            return

        feat_shapes: List[Tuple[int, int]] = []
        dyn_strides: List[int] = []  # 由 H 推断的 stride: s_i = img_size // H_i
        debug_rows = []  # 收集信息便于报错

        for i, cls_i in enumerate(cls_logits_list):
            L_i = int(cls_i.shape[1])  # H*W*A
            if L_i % A_per_loc != 0:
                raise RuntimeError(
                    f"[anchors] Level {i}: length {L_i} 不能被 A_per_loc={A_per_loc} 整除。"
                    f" 请检查 head 的 ratios/scales 是否与 trainer/model 一致。"
                )
            N_i = L_i // A_per_loc  # H*W
            # 经验上 H=W，多数轻量 FPN 都是方图；若不是完全平方数，做个兜底。
            H_i = int(round(N_i ** 0.5))
            W_i = H_i
            if H_i * W_i != N_i:
                # 兜底：用默认 stride 推一个近似网格
                default_stride = self.strides[i] if i < len(self.strides) else self.strides[-1]
                H_i = max(1, self.img_size // int(default_stride))
                W_i = H_i
            feat_shapes.append((H_i, W_i))
            s_i = max(1, self.img_size // H_i)
            dyn_strides.append(s_i)
            debug_rows.append((i, L_i, A_per_loc, N_i, H_i, W_i, s_i))

        # 生成 anchors（严格按推断出来的 H、W 和 stride）
        anchors = generate_ssd_anchors(
            img_size=self.img_size,
            feat_shapes=feat_shapes,
            strides=dyn_strides,
            ratios=self.ratios,
            scales=self.scales,
        )
        self.anchors = anchors.to(self.device)

        # 再校验一次长度是否一致，若不一致，打印详细信息
        if self.anchors.shape[0] != total_L:
            lines = ["[anchors] 生成后长度不匹配：",
                     f"  anchors={self.anchors.shape[0]}, total_L={total_L}, "
                     f"A_per_loc={A_per_loc}, ratios={self.ratios}, scales={self.scales}"]
            for (i, L_i, Apl, N_i, H_i, W_i, s_i) in debug_rows:
                lines.append(f"  L{i}={L_i} -> N{i}={N_i} = H{i}*W{i}={H_i}*{W_i}, s{i}={s_i}")
            raise RuntimeError("\n".join(lines))
