# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict

from core.task.base_trainer import BaseTrainer
from core.loss.movenet_loss import MoveNetLoss
from core.metrics.metrics_kpts import movenet_decode, pck_accuracy


class KptsTrainer(BaseTrainer):

    def __init__(self, model: nn.Module,
                 *,  # 使用*强制后面的参数为关键字参数，增加代码可读性
                 epochs: int,
                 save_dir: str,
                 img_size: int,
                 target_stride: int = 4,
                 num_joints: int = 17,
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

        # 初始化损失函数
        self.loss_func = MoveNetLoss(num_joints=num_joints)

        # 从配置中获取任务特定参数
        self.img_size = img_size
        self.target_stride = target_stride
        self.target_hw = (img_size // self.target_stride, img_size // self.target_stride)

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
        pred_coords = movenet_decode(out_dict, kps_mask, mode='output')
        gt_coords = movenet_decode(labels, kps_mask, mode='label')

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