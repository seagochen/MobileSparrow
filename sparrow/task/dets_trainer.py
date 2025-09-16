# sparrow/task/dets_trainer.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn

from sparrow.loss.ssdlite_loss import SSDLoss, pack_targets_for_ssd, generate_ssd_anchors
from sparrow.task.base_trainer import BaseTrainer


# -----------------------------
# Anchor builder with caching
# -----------------------------
class AnchorBuilder:
    """
    基于真实输出长度懒生成 anchors，并做结果缓存：
      - 支持 stride 提示（来自 model.strides），用于非完全平方数时兜底
      - cache key: (lengths_per_level, stride_hints, img_size, ratios, scales)
    """
    def __init__(self, img_size: int, ratios: Tuple[float, ...], scales: Tuple[float, ...]):
        self.img_size = int(img_size)
        self.ratios   = tuple(ratios)
        self.scales   = tuple(scales)
        self._cache_key: Optional[Tuple] = None
        self._cache: Optional[torch.Tensor] = None

    @staticmethod
    def _guess_hw(N: int, stride_hint: Optional[int], img_size: int) -> Tuple[int, int]:
        # 首选方形网格（多数轻量 FPN 形状一致）
        H = int(round(N ** 0.5))
        W = H
        if H * W == N:
            return H, W
        # 兜底：根据 stride 提示估计（img_size // stride）
        if stride_hint:
            H = max(1, img_size // int(stride_hint))
            W = H
        return H, W

    def ensure(self,
               cls_logits_list: List[torch.Tensor],
               stride_hints: Optional[Tuple[int, ...]] = None,
               device: Optional[torch.device] = None) -> torch.Tensor:
        A = len(self.ratios) * len(self.scales)
        lengths = tuple(int(t.shape[1]) for t in cls_logits_list)
        key = (lengths, stride_hints, self.img_size, self.ratios, self.scales)

        if self._cache_key == key and self._cache is not None:
            return self._cache if device is None else self._cache.to(device, non_blocking=True)

        feat_shapes: List[Tuple[int, int]] = []
        dyn_strides: List[int] = []
        for i, L_i in enumerate(lengths):
            if L_i % A != 0:
                raise RuntimeError(
                    f"[anchors] level={i}: output length {L_i} 不可被 A={A} 整除；"
                    f"请检查 head 的 ratios/scales 与 trainer/model 是否一致。"
                )
            N = L_i // A  # H*W
            stride_hint = stride_hints[i] if stride_hints and i < len(stride_hints) else None
            H, W = self._guess_hw(N, stride_hint, self.img_size)
            feat_shapes.append((H, W))
            dyn_strides.append(max(1, self.img_size // H))

        anchors = generate_ssd_anchors(self.img_size, feat_shapes, dyn_strides,
                                       ratios=self.ratios, scales=self.scales)
        if device is not None:
            anchors = anchors.to(device, non_blocking=True)
        self._cache_key = key
        self._cache = anchors
        return anchors


# -----------------------------
# DetsTrainer (clean)
# -----------------------------
class DetsTrainer(BaseTrainer):
    """
    基于 BaseTrainer 的 SSDLite 检测训练器：
      - anchors 懒生成并缓存（由 AnchorBuilder 维护）
      - 目标打包与损失计算保持最小职责
      - 仅从 model 读取 ratios/scales/strides（避免重复配置）
    """

    def __init__(self,
                 model: nn.Module,
                 *,
                 epochs: int,
                 save_dir: str,
                 img_size: int,
                 # ---- 透传给 BaseTrainer 的通用参数 ----
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
                 # ---- 损失参数 ----
                 alpha_reg: float = 1.0,
                 neg_pos_ratio: int = 3):
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

        # 任务维度
        self.img_size = int(img_size)

        # 只从模型读取 anchor 配置，避免重复定义（SSDLite 已公开这些属性）
        self.ratios  = tuple(getattr(self.model, "ratios", (1.0, 2.0, 0.5)))
        self.scales  = tuple(getattr(self.model, "scales", (1.0, 1.26)))
        self.strides = tuple(getattr(self.model, "strides", (8, 16, 32)))

        # 损失（确保 num_classes 包含背景类；COCO=81）
        num_classes = getattr(self.model, "num_classes", None)
        if not isinstance(num_classes, int):
            raise ValueError("SSDLite 模型缺少 num_classes；请在构造时传入（含背景）。")
        self.loss_func = SSDLoss(num_classes=num_classes, alpha=alpha_reg, neg_pos_ratio=neg_pos_ratio)

        # 懒生成 anchors 的构造器（按需生成 + 缓存）
        self._anchor_builder = AnchorBuilder(self.img_size, self.ratios, self.scales)

    # ---------- BaseTrainer hooks ----------
    def _move_batch_to_device(self, batch):
        imgs, targets_list, img_paths = batch
        imgs = imgs.to(self.device, non_blocking=True)
        return imgs, targets_list, img_paths

    def _calculate_loss(self, batch) -> Tuple[torch.Tensor, Dict[str, float]]:
        imgs, targets_list, _ = batch
        out = self.model(imgs)  # {'cls_logits':[B,L_i,C], 'bbox_regs':[B,L_i,4]}  :contentReference[oaicite:3]{index=3}

        anchors = self._anchor_builder.ensure(out["cls_logits"], stride_hints=self.strides, device=self.device)

        cls_logits = torch.cat(out["cls_logits"], dim=1)  # [B, A_total, C]
        bbox_regs  = torch.cat(out["bbox_regs"],  dim=1)  # [B, A_total, 4]

        targets = pack_targets_for_ssd(targets_list, img_size=self.img_size)
        targets["boxes"]  = [b.to(self.device) for b in targets["boxes"]]
        targets["labels"] = [l.to(self.device) for l in targets["labels"]]

        loss, ld = self.loss_func(cls_logits, bbox_regs, anchors, targets)
        # BaseTrainer 需要 (总损失Tensor, 可记录的 dict)
        return loss, {"loss": float(loss.detach().cpu()), **{k: float(v) for k, v in ld.items()}}

    @torch.no_grad()
    def _calculate_metrics(self, model, batch) -> Dict[str, float]:
        imgs, targets_list, _ = batch
        out = model(imgs)

        anchors = self._anchor_builder.ensure(out["cls_logits"], stride_hints=self.strides, device=self.device)

        cls_logits = torch.cat(out["cls_logits"], dim=1)
        bbox_regs  = torch.cat(out["bbox_regs"],  dim=1)

        targets = pack_targets_for_ssd(targets_list, img_size=self.img_size)
        targets["boxes"]  = [b.to(self.device) for b in targets["boxes"]]
        targets["labels"] = [l.to(self.device) for l in targets["labels"]]

        loss, ld = self.loss_func(cls_logits, bbox_regs, anchors, targets)
        return {"loss": float(loss.detach().cpu()), **{k: float(v) for k, v in ld.items()}}

    def _get_main_metric(self, metrics: Dict[str, float]) -> float:
        # BaseTrainer 默认“越大越好”，我们采用 -loss 作为主指标  :contentReference[oaicite:4]{index=4}
        return -metrics.get("loss", float("inf"))
