# -*- coding: utf-8 -*-
import gc
import os
from copy import deepcopy
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

from core.task import common


class ModelEMA:
    """ 模型指数移动平均 (Model Exponential Moving Average) """

    def __init__(self, model, decay=0.9998):
        self.ema = deepcopy(model).eval()
        self.decay = float(decay)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if k in msd and v.dtype == msd[k].dtype:
                v.copy_(v * d + msd[k] * (1.0 - d))

    def state_dict(self):
        return self.ema.state_dict()


class BaseTrainer:
    """
    一个通用的训练器基类，封装了标准的训练和验证循环。
    子类需要实现 _calculate_loss 和 _calculate_metrics 方法。
    """

    def __init__(self, cfg: Dict, model: nn.Module):
        self.cfg = cfg
        use_cuda = (self.cfg.get('GPU_ID', '') != '' and torch.cuda.is_available())
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.model = model.to(self.device)

        # --- 通用组件 ---
        self.optimizer = common.getOptimizer(
            self.cfg['optimizer'], self.model, self.cfg['learning_rate'], self.cfg['weight_decay']
        )
        self.scheduler = common.getSchedu(self.cfg['scheduler'], self.optimizer)
        self.scaler = GradScaler(enabled=self.cfg.get("amp", True))
        self.ema = ModelEMA(self.model, decay=self.cfg.get("ema_decay", 0.9998)) if self.cfg.get("ema", True) else None

        # --- 状态记录 ---
        self.save_dir = self.cfg["save_dir"]
        os.makedirs(self.save_dir, exist_ok=True)
        # 约定：best_score 越大越好
        self.best_score = float("-inf")
        self.epochs = int(self.cfg.get("epochs", 140))
        self.log_interval = int(self.cfg.get("log_interval", 10))
        self.clip_grad = float(self.cfg.get("clip_gradient", 0.0))

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
            self.scheduler.step(main_metric)
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

            if self.clip_grad > 0:
                self.scaler.unscale_(self.optimizer)
                common.clipGradient(self.optimizer, self.clip_grad)

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
        print(f"LR: {lr:f} - [{stage}] {epoch + 1}/{epochs} {log_str}")

    def _save_checkpoints(self, score: float, epoch: int):
        net_to_save = self.ema.ema if self.ema else self.model
        torch.save(net_to_save.state_dict(), os.path.join(self.save_dir, "last.pt"))
        if score > self.best_score:
            self.best_score = score
            best_path = os.path.join(self.save_dir, "best.pt")
            torch.save(net_to_save.state_dict(), best_path)
            print(f"[INFO] New best: main_score={score:.5f} -> saved to {best_path}")
        else:
            print(f"[INFO] Kept best: main_score={self.best_score:.5f} (current {score:.5f})")

    def _on_train_end(self):
        print("Training finished.")
        del self.model
        gc.collect()
        torch.cuda.empty_cache()