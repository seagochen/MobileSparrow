# -*- coding: utf-8 -*-
from typing import Tuple, Dict

import torch
import torch.nn as nn

from core.loss.ssdlite_loss import SSDLoss
from core.task.base_trainer import BaseTrainer


class DetTrainer(BaseTrainer):
    def __init__(self, cfg: Dict, model: nn.Module):
        super().__init__(cfg, model)
        num_classes = getattr(self.model, "num_classes", int(cfg.get("num_classes", 81)))
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