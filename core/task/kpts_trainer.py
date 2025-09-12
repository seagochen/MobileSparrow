# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict

from core.task.base_trainer import BaseTrainer
from core.loss.movenet_loss import MovenetLoss
from core.metrics.metrics_kpts import movenet_decode, pck_accuracy


class KptsTrainer(BaseTrainer):
    def __init__(self, cfg: Dict, model: nn.Module):
        super().__init__(cfg, model)
        self.loss_func = MovenetLoss()

        # 从配置中获取任务特定参数
        self.img_size = int(self.cfg.get("img_size", 192))
        self.target_stride = int(self.cfg.get("target_stride", 4))
        self.target_hw = (self.img_size // self.target_stride, self.img_size // self.target_stride)

    def _align_output(self, output: Dict) -> Dict:
        """将模型输出的四头resize到与标签一致的尺寸"""
        hm_r, ct_r, rg_r, of_r = nn.functional.interpolate(
            [output["heatmaps"], output["centers"], output["regs"], output["offsets"]],
            size=self.target_hw,
            mode='bilinear',
            align_corners=False
        )
        return {"heatmaps": hm_r, "centers": ct_r, "regs": rg_r, "offsets": of_r}

    def _calculate_loss(self, batch) -> Tuple[torch.Tensor, Dict[str, float]]:
        imgs, labels, kps_mask, _ = batch

        raw_out = self.model(imgs)
        out_dict = self._align_output(raw_out)
        out_list = [out_dict["heatmaps"], out_dict["centers"], out_dict["regs"], out_dict["offsets"]]

        hm_loss, b_loss, c_loss, r_loss, o_loss = self.loss_func(out_list, labels, kps_mask)
        total_loss = hm_loss + c_loss + r_loss + o_loss + b_loss

        loss_dict = {
            "loss": total_loss.item(),
            "hm": hm_loss.item(), "b": b_loss.item(), "c": c_loss.item(),
            "r": r_loss.item(), "o": o_loss.item()
        }
        return total_loss, loss_dict

    def _calculate_metrics(self, model, batch) -> Dict[str, float]:
        imgs, labels, kps_mask, _ = batch
        raw_out = model(imgs)
        out_dict = self._align_output(raw_out)

        # 解码得到坐标
        pred_coords = movenet_decode(out_dict, kps_mask, mode='output', img_size=self.img_size)
        gt_coords = movenet_decode(labels, kps_mask, mode='label', img_size=self.img_size)

        # 计算PCK正确率
        correct_counts, total_in_batch = pck_accuracy(pred_coords, gt_coords, img_size=self.img_size)
        mean_acc = np.mean(correct_counts / total_in_batch)

        return {"acc": float(mean_acc)}

    def _get_main_metric(self, metrics: Dict[str, float]) -> float:
        return metrics.get("acc", 0.0)

    def _move_batch_to_device(self, batch):
        imgs, labels, kps_mask, img_names = batch
        imgs = imgs.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)
        kps_mask = kps_mask.to(self.device, non_blocking=True)
        return imgs, labels, kps_mask, img_names