import json
import os
import random
from typing import Optional, Dict

import timm
import torch
from torch import nn, autocast
from tqdm import tqdm

from sparrow.datasets.coco_dets import create_coco_ssd_dataloader
from sparrow.losses.ssdlite_loss import SSDLoss
from sparrow.models.ssdlite_fpn import SSDLite_FPN
from sparrow.trainer.base_trainer import BaseTrainer
from sparrow.trainer.components import clip_gradient, set_seed, load_ckpt_if_any, save_ckpt
from sparrow.utils.logger import logger
from sparrow.utils.plot_curves import plot_training_curve
from sparrow.utils.yaml_config import update_from_yaml


class SSDLiteTrainer(BaseTrainer):

    def __init__(self,  yaml_path: Optional[str] = None):

        # --- 加载训练配置信息 ---
        cfg = update_from_yaml(yaml_path)

        # --- 创建模型 ---
        backbone = timm.create_model(cfg.get("backbone", "mobilenetv3_large_100"),
                                     pretrained=True,
                                     features_only=True,
                                     out_indices=(2, 3, 4))
        model = SSDLite_FPN(
            backbone=backbone,
            num_classes=cfg.get("num_classes", 80),
            fpn_out_channels=cfg.get("fpn_out_channels", 128),
            img_size=cfg.get("img_size", 320)
        )

        # --- 确定训练设备 ---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Loss ---
        loss_fn = SSDLoss(
            num_classes=cfg.get("num_classes", 80),
            iou_threshold_pos=cfg.get("iou_pos", 0.45),
            iou_threshold_neg=cfg.get("iou_neg", 0.30),
            focal_alpha=cfg.get("focal_alpha", 0.25),
            debug_assign=cfg.get("debug_assign", False)
        )

        # --- 初始化 super ---
        super().__init__(
            model,
            loss_fn,
            data_dir=cfg.get("data_dir", "/home/user/datasets/coco2017_ssdlite"),
            save_dir=cfg.get("save_dir", "runs/ssdlite_fpn_mbv3"),
            device=device,

            # Optimizer
            optimizer_name=cfg.get("optimizer_name", "adamw"),
            lr=cfg.get("lr", 3e-4),
            weight_decay=cfg.get("weight_decay",  1e-4),

            # Scheduler
            scheduler_name=cfg.get("scheduler_name", "cosine"),
            epochs=cfg.get("epochs", 100),
            use_warmup_scheduler=cfg.get("use_warmup_scheduler", True),
            warmup_epochs=cfg.get("warmup_epochs", 10),
            start_factor=cfg.get("start_factor", 1.0 / max(1, cfg.get("warmup_epochs", 10))),
            end_factor=cfg.get("end_factor", 1.0),

            # 训练技巧
            use_amp = cfg.get("use_amp", True),
            use_ema = cfg.get("use_ema", True),
            ema_decay = cfg.get("ema_decay", 0.9998),
            use_clip_grad=cfg.get("use_clip_grad", True),
            clip_grad_norm = cfg.get("clip_grad_norm", 1.0),

            # 其他参数
            **cfg
        )

        # --- 加载数据集 ---
        self.train_dl, self.val_dl = self.create_dataloaders(cfg.get("seed",  random.randrange(1, 100)))


    def create_dataloaders(self, seed = 45):

        # 数据增强相关参数
        aug_cfg = {
            "p_flip": 0.5,
            "scale": (0.8, 1.2),
            "rotate": 10.0,
            "translate": 0.08,
            "color": True,
            "p_color": 0.8,
        }
        non_aug_cfg = {
            "p_flip":0.0,
            "scale":(1.0,1.0),
            "rotate":0.0,
            "translate":0.0,
            "color":False,
            "p_color":0.0
        }

        # 加载 COCO train
        train_loader = create_coco_ssd_dataloader(
            dataset_root=self.cfg.get("data_dir", "/home/user/datasets/coco2017_ssdlite"),
            img_size=self.cfg.get("img_size", 320),
            batch_size=self.cfg.get("batch_size", 64),
            num_workers=self.cfg.get("num_workers", 4),
            pin_memory=self.cfg.get("pin_memory", True),
            is_train=True,
            aug_cfg=self.cfg.get("aug_cfg", aug_cfg)
        )

        #  加载 COCO validation
        val_loader = create_coco_ssd_dataloader(
            dataset_root=self.cfg.get("data_dir", "/home/user/datasets/coco2017_ssdlite"),
            img_size=self.cfg.get("img_size", 320),
            batch_size=self.cfg.get("batch_size", 64),
            num_workers=self.cfg.get("num_workers", 4),
            pin_memory=self.cfg.get("pin_memory", True),
            is_train=False,
            aug_cfg=non_aug_cfg,
        )
        return train_loader, val_loader


    def train_one_epoch(self,
                        model: nn.Module,
                        loss_fn: nn.Module,
                        epoch: int,
                        loader: torch.utils.data.DataLoader,
                        optimizer: torch.optim.Optimizer,
                        scaler: torch.amp.GradScaler,
                        device: torch.device) -> Dict[str, float]:

        # 1. 设置模型为训练模式
        # 效果：启用 Dropout, BatchNorm 使用 batch 统计
        model.train()

        # 2. 初始化损失累加器
        running = {
            "total": 0.0,       # 总损失（加权组合）
            "cls": 0.0,         # 分类损失
            "reg": 0.0,         # bbox偏移损失
        }
        count = 0  # 已处理的 batch 数量

        # 3. 创建进度条
        pbar = tqdm(enumerate(loader, 1), total=len(loader), ncols=120,
                    desc=f"Epoch {epoch:03d}/{self.epochs}")

        # 4. 遍历所有训练批次
        for step, (images, labels, _) in pbar:
            # 4.1 加载数据到设备
            # non_blocking=True: 异步传输，提高 GPU 利用率
            images = images.to(self.device, non_blocking=True)
            labels = [t.to(self.device, non_blocking=True) for t in labels]

            # 4.2 清空梯度
            # set_to_none=True: 比 zero_grad() 更高效（直接释放内存）
            optimizer.zero_grad(set_to_none=True)

            # 4.3 前向传播（使用混合精度）
            # autocast: 自动将部分操作转为 float16，加速训练
            with autocast(device_type=device.type, enabled=self.use_amp, dtype=torch.float16):
                # 模型预测
                preds = model(images)

                # 计算损失（返回总损失和各项分损失）
                loss, details = self.loss_fn(
                    preds["anchors"],
                    preds["cls_logits"],
                    preds["bbox_deltas"],
                    labels)

            # 4.4 反向传播（使用梯度缩放）
            # 原因：float16 计算时梯度可能下溢，需要缩放
            scaler.scale(loss).backward()

            # 4.5 梯度裁剪（防止梯度爆炸）
            if self.use_clip_grad:
                # 先 unscale 梯度（还原到原始尺度）
                scaler.unscale_(optimizer)
                # 裁剪所有参数的梯度范数到指定值
                clip_gradient(optimizer, self.clip_grad_norm)

            # 4.6 更新参数
            # step: 根据缩放后的梯度更新参数
            # update: 动态调整缩放因子（如果出现 inf/nan 则减小缩放）
            scaler.step(optimizer)
            scaler.update()

            # 4.7 累加损失
            running["total"] += loss.detach().item()
            running["cls"]   += details["cls_loss"].item()
            running["reg"]   += details["reg_loss"].item()
            count += 1

            # 4.8 更新进度条显示, 显示当前平均损失和学习率
            pbar.set_postfix({
                "loss": f"{running['total'] / count:.4f}",
                "cls":   f"{running['cls'] / count:.4f}",
                "reg":  f"{running['reg'] / count:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}"  # 当前学习率
            })

        # 5. 返回本 epoch 的平均损失
        # max(1, count): 防止除零（虽然 count 不会为 0）
        return {k: v / max(1, count) for k, v in running.items()}


    @torch.no_grad()
    def evaluate(self,
                 model: nn.Module,
                 loss_fn: nn.Module,
                 epoch: int,
                 loader: torch.utils.data.DataLoader,
                 device: torch.device) -> Dict[str, float]:

        # 1. 设置模型为评估模式
        # 效果：禁用 Dropout，BatchNorm 使用全局统计
        model.eval()

        # 2. 初始化损失累加器
        running = {
            "total": 0.0,       # 总损失（加权组合）
            "cls": 0.0,         # 分类损失
            "reg": 0.0,         # bbox偏移损失
        }

        # 3. 创建进度条
        pbar = tqdm(enumerate(loader, 1), total=len(loader), ncols=120,
                    desc="Valid")

        # 4. 遍历所有训练批次
        for step, (images, labels, _) in pbar:
            # 4.1 加载数据到设备
            # non_blocking=True: 异步传输，提高 GPU 利用率
            images = images.to(self.device, non_blocking=True)
            labels = [t.to(self.device, non_blocking=True) for t in labels]

            # 4.2 前向传播（使用混合精度）
            with autocast(device_type=device.type, enabled=self.use_amp, dtype=torch.float16):
                # 预测
                preds = model(images)

                # 计算损失（返回总损失和各项分损失）
                loss, details = self.loss_fn(
                    preds["anchors"],
                    preds["cls_logits"],
                    preds["bbox_deltas"],
                    labels)

            # 4.3. 累加损失
            running["total"] += loss.detach().item()
            running["cls"]   += details["cls_loss"].item()
            running["reg"]   += details["reg_loss"].item()

        # 5. 计算平均损失，并返回
        n_batches = len(loader)
        return {k: v / max(1, n_batches) for k, v in running.items()}


    def train_model(self):

        # Set training seed
        set_seed(self.cfg.get("seed", random.randrange(1, 100)))

        # Resume the training process
        if self.cfg.get("resume", False):
            start_epoch, best_val = load_ckpt_if_any(
                model=self.model,
                ckpt_path=os.path.join(self.save_dir, "last.pt"),
                optimizer=self.optimizer,
                scaler=self.scaler,
                device=self.device)
            logger.info("Sparrow", "Continue SSDLite from previous checkpoint.")
        else:
            start_epoch = 0
            best_val = float("inf")
            logger.info("Sparrow", "Training SSDLite from beginning.")

        # Training logs to plot
        hist = dict(train=[], val=[])

        # Training the model
        for epoch in range(start_epoch, self.epochs):

            # Train the model
            train_loss = self.train_one_epoch(
                model=self.model,
                loss_fn=self.loss_fn,
                epoch=epoch,
                loader=self.train_dl,
                optimizer=self.optimizer,
                scaler=self.scaler,
                device=self.device)

            # Adjust the learning rates
            self.scheduler.step()

            # Evaluate the training effects
            val_loss = self.evaluate(
                model=self.model,
                loss_fn=self.loss_fn,
                epoch=epoch,
                loader=self.val_dl,
                device=self.device)

            hist["train"].append(train_loss)
            hist["val"].append(val_loss)

            # Output the metrics
            row = {
                "epoch": epoch + 1, "train": train_loss, "val": val_loss,
                "lr": self.optimizer.param_groups[0]["lr"]
            }
            print(json.dumps(row, ensure_ascii=False, indent=2))

            # save last
            save_ckpt({
                "epoch": epoch + 1,
                "model": self.model.state_dict(),
                "optim": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict(),
                "best_val_deg": best_val
            }, self.save_dir, "last.pt")

            # save best
            if val_loss["total"] < best_val:
                best_val = val_loss["total"]
                best_path = save_ckpt({
                    "epoch": epoch + 1,
                    "model": self.model.state_dict(),
                    "optim": self.optimizer.state_dict(),
                    "scaler": self.scaler.state_dict(),
                    "best_val_deg": best_val
                }, self.save_dir, "best.pt")
                print(f"[best] new best loss = {best_val:.3f}°  ->  {best_path}")
            # end-for: epoch in range(start_epoch, self.epochs)

        # Get the hist curves
        train_total_loss_hist = [t["total"] for t in hist["train"]]
        val_total_loss_hist = [t["total"] for t in hist["val"]]

        # Plot the training curves
        plot_training_curve(
            title="Training Loss",
            ylabel="Loss",
            save_path=os.path.join(self.save_dir, "training_curves.png"),
            train_vals=train_total_loss_hist,
            val_vals=val_total_loss_hist
        )

    def export_onnx(self, model: nn.Module):
        raise NotImplemented

    def export_wrapper(self, model: nn.Module):
        raise NotImplemented
