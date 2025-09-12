# -*- coding: utf-8 -*-
from typing import Tuple, Dict

import torch
import torch.nn as nn

from sparrow.loss.ssdlite_loss import SSDLoss
from sparrow.task.base_trainer import BaseTrainer


class DetTrainer(BaseTrainer):

    def __init__(self, model: nn.Module,
                 *,  # 使用*强制后面的参数为关键字参数，增加代码可读性
                 epochs: int,
                 save_dir: str,
                 img_size: int,
                 target_stride: int = 4,
                 num_classes: int,
                 device: torch.device,
                 # 优化器与调度器参数
                 optimizer_cfg: Dict,
                 scheduler_cfg: Dict,
                 # 训练技巧参数
                 use_amp: bool = True,
                 use_ema: bool = True,
                 ema_decay: float = 0.9998,
                 clip_grad_norm: float = 0.0,
                 # 日志参数
                 log_interval: int = 10):

        # 初始化父类
        super().__init__(model,
                         epochs=epochs,
                         save_dir=save_dir,
                         device=device,
                         optimizer_cfg=optimizer_cfg,
                         scheduler_cfg=scheduler_cfg,
                         use_amp=use_amp,
                         use_ema=use_ema,
                         ema_decay=ema_decay,
                         clip_grad_norm=clip_grad_norm,
                         log_interval=log_interval)

        num_classes = num_classes
        self.criterion = SSDLoss(num_classes=num_classes)

    def _calculate_loss(self, batch) -> Tuple[torch.Tensor, Dict[str, float]]:
        imgs, targets = batch
        out = self.model(imgs)
        loss, meter = self.criterion(out["cls_logits"], out["bbox_regs"], out["anchors"], targets)
        loss_dict = {"loss": loss.item(), "cls": meter["loss_cls"], "reg": meter["loss_reg"]}
        return loss, loss_dict

    def _calculate_metrics(self, model, batch) -> Dict[str, float]:
        # 对于检测任务，我们用验证集损失作为评价指标
        loss, loss_dict = self._calculate_loss(batch)
        return loss_dict

    def _get_main_metric(self, metrics: Dict[str, float]) -> float:
        # 约定：BaseTrainer中越大越好，所以返回负损失
        return -metrics.get("loss", float('inf'))

    def _move_batch_to_device(self, batch):
        imgs, targets = batch
        imgs = imgs.to(self.device, non_blocking=True)
        # detection的target是list of dicts, 需要特殊处理
        targets = [
            {"boxes": t["boxes"].to(self.device), "labels": t["labels"].to(self.device)} for t in targets
        ]
        return imgs, targets