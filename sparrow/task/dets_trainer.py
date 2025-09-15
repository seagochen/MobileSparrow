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
                 strides: Tuple[int, int, int] = (8, 16, 32),
                 ratios: Tuple[float, ...] = (1.0, 2.0, 0.5),
                 scales: Tuple[float, ...] = (1.0, 1.26),
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

        self.img_size = int(img_size)
        # self.strides  = tuple(strides)

        # 直接从模型获取配置，保证一致性
        self.ratios   = self.model.ratios                         # <--- 修改后的代码
        self.scales   = self.model.scales                         # <--- 修改后的代码
        self.strides = self.model.strides

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

