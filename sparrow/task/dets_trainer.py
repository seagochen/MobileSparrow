# sparrow/task/dets_trainer.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Literal

import torch
import torch.nn as nn

from sparrow.loss.ssdlite_loss import SSDLoss, pack_targets_for_ssd, generate_ssd_anchors
from sparrow.task.base_trainer import BaseTrainer


# -----------------------------
# Anchor builder with caching
# -----------------------------
class AnchorBuilder:
    """
    AnchorBuilder —— SSDLite 的“锚框（anchor）生成与对齐器”，带缓存。

    【作用 / 为什么需要】
    - 根据“本次前向传播真实的输出长度”与 anchor 配置（ratios / scales / strides / img_size），
      懒生成与各层 head 输出 **一一对齐** 的 anchors（cx, cy, w, h；归一化到 [0,1]）。
    - 解决多输入分辨率、不同骨干/宽度倍率下，特征图 H×W 难以预先写死的问题。
    - 结果带缓存：相同形状与配置下直接复用，避免重复计算/形状错配。

    【它不做什么】
    - 不做去重、合并框，也不做 NMS。AnchorBuilder 只生成 anchors。
      “多层重复命中 → 统一结果”由解码 + NMS（或 Soft-NMS/WBF）在 **推理** 阶段完成。
      训练阶段，它提供 anchors 给 SSDLoss 做匹配/编码/损失。

    【构造参数】
    - img_size: int
        训练/推理使用的方形输入尺寸（用于将像素坐标归一化到 [0,1] 并估算网格）。
    - ratios: Tuple[float, ...]
        长宽比集合（每个网格点会生成 len(ratios) × len(scales) 个 anchors）。
    - scales: Tuple[float, ...]
        尺度集合（与 ratios 笛卡尔积决定每点 anchor 数 A）。

    【核心方法】
    - ensure(cls_logits_list: List[Tensor],
             stride_hints: Optional[Tuple[int, ...]] = None,
             device: Optional[torch.device] = None) -> Tensor
        输入：
          * cls_logits_list：按层的分类输出列表，形状为 [B, H_i*W_i*A, C]（已 flatten）。
          * stride_hints：每层步长提示（如 [8,16,32]），用于反推 H_i、W_i 的兜底估计。
          * device：可选，返回 anchors 所在设备（默认跟随缓存/CPU，传入则会 .to(device)）。
        返回：
          * anchors：拼接后的全量 anchors，形状 [A_total, 4]，格式为 (cx, cy, w, h)，范围 [0,1]。
            拼接顺序为按层级依次连接；层内每个网格点按 ratios × scales 的嵌套顺序展开。

    【工作流程（简要算法）】
    1) 计算 A = len(ratios) * len(scales)。
    2) 对每一层 i，读取 logit 的长度 L_i = cls_logits_list[i].shape[1]，校验 L_i % A == 0。
       令 N_i = L_i / A，即该层网格点数 H_i * W_i。
    3) 反推 (H_i, W_i)：
       - 优先取近似方形：H ≈ W ≈ round(sqrt(N))；若 H*W ≠ N，再用 stride_hints[i] 兜底：
         H = W = img_size // stride_hints[i]（最常见的轻量 FPN 形状）。
    4) 调用 `generate_ssd_anchors(img_size, feat_shapes=[(H_i,W_i)...], strides, ratios, scales)`：
       - 以像素网格中心 ((x+0.5)*s, (y+0.5)*s) / img_size 计算 (cx, cy)；
       - 以 (w, h) = (sc*s*sqrt(r), sc*s/sqrt(r)) / img_size 生成每层 H*W*A 个 anchors。
    5) 将所有层的 anchors 级联为 [A_total, 4] 并缓存（见下）。

    【缓存机制（加速/防错）】
    - 以 key = (lengths_per_level, stride_hints, img_size, ratios, scales) 为缓存键；
    - 若命中，直接返回缓存；否则重新生成并更新缓存；
    - `device` 参数仅控制返回张量的设备拷贝，不改变缓存本体。

    【常见报错与排查】
    - “输出长度不可被 A 整除”：说明 head 的 A 与 (ratios, scales) 不一致，或 logits 形状异常。
    - 解码框错位/指标异常：大多是 strides/img_size 与预处理不一致，或复用了不对应分辨率的缓存。
      解决：核对 model.strides / 当前使用的 img_size，与训练/推理的实际 resize/letterbox 逻辑一致；
      或在分辨率切换时重建 AnchorBuilder。

    【典型用法（训练/验证）】
    - out = model(images)  # 得到各层 [B, H_i*W_i*A, C] / [B, H_i*W_i*A, 4]
    - anchors = anchor_builder.ensure(out["cls_logits"], stride_hints=model.strides, device=images.device)
    - loss = ssdloss(out["cls_logits"], out["bbox_regs"], anchors, targets)

    【与其它模块的关系】
    - SSDLoss：依赖 anchors 进行 IoU 匹配、正负样本划分、回归编码/解码与 OHEM；
    - 推理后处理：解码 (bbox_regs + anchors) → concat 各层 → NMS/Soft-NMS → 最终检测结果。
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
                 # —— 优化器改为明参数 ——
                 optimizer_name: str,
                 learning_rate: float,
                 weight_decay: float,
                 # —— 调度器改为明参数 ——
                 scheduler_name: str = "MultiStepLR",
                 milestones=None,
                 gamma: float = 0.1,
                 step_size: int = 30,
                 mode: Literal["min", "max"] = "min",
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
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            scheduler_name=scheduler_name,
            milestones=milestones,
            gamma=gamma,
            step_size=step_size,
            mode=mode,
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            T_0=T_0,
            T_mult=T_mult,
            last_epoch=last_epoch,
            use_amp=use_amp,
            use_ema=use_ema,
            ema_decay=ema_decay,
            clip_grad_norm=clip_grad_norm,
            log_interval=log_interval
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
        # 最小化验证损失作为主指标
        return metrics.get("loss", float("inf"))
