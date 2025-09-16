# -*- coding: utf-8 -*-
import gc
import os
from typing import Dict, Tuple

import torch
from torch import optim
import torch.nn as nn
from torch.amp import autocast, GradScaler

from sparrow.task import common
from sparrow.task.common import ModelEMA
from sparrow.utils.logger import logger


class BaseTrainer:

    """
    一个通用的训练器基类（重构版）。
    所有配置通过显式参数传入，而不是通过一个统一的cfg字典。
    """

    def __init__(self,
                 model: nn.Module,
                 *,  # 使用*强制后面的参数为关键字参数，增加代码可读性
                 epochs: int,
                 save_dir: str,
                 device: torch.device,
                 # 优化器与调度器参数
                 optimizer_cfg: Dict,
                # —— 学习率调度器改为明参 —— 
                 scheduler_name: str,
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
                 # 训练技巧参数
                 use_amp: bool = True,
                 use_ema: bool = True,
                 ema_decay: float = 0.9998,
                 clip_grad_norm: float = 0.0,
                 # 日志参数
                 log_interval: int = 10
                 ):

        # --- 核心组件 ---
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs
        self.save_dir = save_dir

        # --- 训练技巧 ---
        self.scaler = GradScaler(enabled=use_amp)
        self.ema = ModelEMA(self.model, decay=ema_decay) if use_ema else None
        self.clip_grad_norm = clip_grad_norm

        # --- 优化器和调度器 ---
        # 现在从专门的字典中创建，职责更清晰
        self.optimizer = common.select_optimizer(
            optimizer_cfg['name'], self.model, optimizer_cfg['lr'], optimizer_cfg['weight_decay']
        )
        # 明参构建调度器（不再接受 dict / 字符串）
        self.scheduler = common.build_scheduler(
            scheduler_name, self.optimizer,
            milestones=milestones, gamma=gamma, step_size=step_size,
            mode=mode, factor=factor, patience=patience, min_lr=min_lr,
            T_0=T_0, T_mult=T_mult, last_epoch=last_epoch
        )

        # --- 状态记录 ---
        os.makedirs(self.save_dir, exist_ok=True)
        self.best_score = float("inf")
        self.log_interval = log_interval

    def train(self, train_loader, val_loader):
        """ 训练主循环 (模板方法) """
        for epoch in range(self.epochs):
            # 训练
            self.model.train()
            train_meters = self._train_epoch(train_loader, epoch)
            self._log_stats("Train", epoch, self.epochs, train_meters)

            # 验证
            val_meters = self.evaluate(val_loader)
            self._log_stats("Val", epoch, self.epochs, val_meters)

            # 更新学习率和保存模型
            main_metric = self._get_main_metric(val_meters)

            # 只有 ReduceLROnPlateau 需要 metric，其它直接 step()
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(main_metric)
            else:
                self.scheduler.step()

            self._save_checkpoints(main_metric, epoch)

        self._on_train_end()

    def _train_epoch(self, data_loader, epoch: int) -> Dict[str, float]:
        """ 训练一个 epoch """
        meters = {}
        iters_per_epoch = len(data_loader)
        for i, batch in enumerate(data_loader):
            batch = self._move_batch_to_device(batch)

            with autocast('cuda', enabled=self.scaler.is_enabled()):
                loss, loss_dict = self._calculate_loss(batch)

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()

            if self.clip_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)                    # 先反缩放 AMP 梯度
                common.clip_gradient(self.optimizer, self.clip_grad_norm)    # 范数裁剪, 避免梯度爆炸

            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.ema:
                self.ema.update(self.model)

            self._update_meters(meters, loss_dict, batch_size=next(iter(batch)).shape[0])

            if i % self.log_interval == 0:
                self._log_iter_stats(epoch, self.epochs, i, iters_per_epoch, meters)
        print()
        return self._get_mean_meters(meters)

    @torch.no_grad()
    def evaluate(self, data_loader) -> Dict[str, float]:
        """ 在验证集上评估 """
        net = self.ema.ema if self.ema else self.model
        net.eval()
        meters = {}
        for batch in data_loader:
            batch = self._move_batch_to_device(batch)
            metrics_dict = self._calculate_metrics(net, batch)
            self._update_meters(meters, metrics_dict, batch_size=next(iter(batch)).shape[0])
        return self._get_mean_meters(meters)

    # --- 需要子类实现的抽象方法 ---
    def _calculate_loss(self, batch) -> Tuple[torch.Tensor, Dict[str, float]]:
        """ 计算损失。返回 (总损失Tensor, 包含各项损失的字典) """
        raise NotImplementedError

    def _calculate_metrics(self, model, batch) -> Dict[str, float]:
        """ 计算评价指标。返回包含各项指标的字典 """
        raise NotImplementedError

    def _get_main_metric(self, metrics: Dict[str, float]) -> float:
        """ 从指标字典中获取用于保存模型和调度LR的主指标 """
        raise NotImplementedError

    def _move_batch_to_device(self, batch) -> any:
        """ 将数据批次移动到指定设备 """
        raise NotImplementedError

    # --- Helper 方法 ---
    def _update_meters(self, meters: Dict, new_values: Dict, batch_size: int):
        for k, v in new_values.items():
            if k not in meters:
                meters[k] = [0.0, 0]  # value_sum, count
            meters[k][0] += v * batch_size
            meters[k][1] += batch_size

    def _get_mean_meters(self, meters: Dict) -> Dict[str, float]:
        return {k: v[0] / max(1, v[1]) for k, v in meters.items()}

    def _log_iter_stats(self, epoch, epochs, i, total_i, meters: Dict):
        mean_meters = self._get_mean_meters(meters)
        log_str = " ".join([f"{k}={v:.4f}" for k, v in mean_meters.items()])
        print(f"\r{epoch + 1}/{epochs} [{i}/{total_i}] {log_str}", end='', flush=True)

    def _log_stats(self, stage, epoch, epochs, meters: Dict):
        log_str = " ".join([f"{k}={v:.4f}" for k, v in meters.items()])
        lr = self.optimizer.param_groups[0]['lr']
        logger.info("STATUS", f"LR: {lr:f} - [{stage}] {epoch + 1}/{epochs} {log_str}")

    def _save_checkpoints(self, score: float, epoch: int):
        net_to_save = self.ema.ema if self.ema else self.model

        # Create a complete checkpoint dictionary
        checkpoint = {
            'model_state': net_to_save.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epoch': epoch,
            'best_score': self.best_score,  # Store the best score at this point
        }

        # Save the complete checkpoint for 'last.pt'
        torch.save(checkpoint, os.path.join(self.save_dir, "last.pt"))

        # Save the best checkpoint if necessary
        if score < self.best_score:
            self.best_score = score

            # Update the best score in the checkpoint for saving to 'best.pt'
            checkpoint['best_score'] = self.best_score

            best_path = os.path.join(self.save_dir, "best.pt")
            torch.save(checkpoint, best_path)  # Save the same complete checkpoint structure
            logger.warning("SAVING", f"[INFO] New best: main_score={score:.5f} -> saved to {best_path}")
        else:
            logger.warning("SAVING", f"[INFO] Kept best: main_score={self.best_score:.5f} (current {score:.5f})")

    def _on_train_end(self):
        print("Training finished.")
        del self.model
        gc.collect()
        torch.cuda.empty_cache()