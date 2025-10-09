import json
import os
import random
from typing import Optional, Dict

import timm
import torch
from torch import nn, autocast
from tqdm import tqdm

from sparrow.datasets.coco_kpts import create_kpts_dataloader
from sparrow.evaluators.coco_kpts_evaluator import CocoKeypointsEvaluator
from sparrow.losses.movenet_fpn_sp_loss import MoveNet2HeadLoss
from sparrow.models.movenet_fpn_sp import MoveNet_FPN_SP
from sparrow.models.onnx.movenet_fpn_sp_wrapper import MoveNetExportWrapper
from sparrow.trainer.base_trainer import BaseTrainer
from sparrow.trainer.components import clip_gradient, set_seed, load_ckpt_if_any, save_ckpt
from sparrow.utils.logger import logger
from sparrow.utils.plot_curves import plot_training_curve
from sparrow.utils.yaml_config import update_from_yaml


class MoveNetSingleTrainer(BaseTrainer):

    def __init__(self,  yaml_path: Optional[str] = None):

        # --- 加载训练配置信息 ---
        cfg, extra_cfg = update_from_yaml(yaml_path, return_extra=True)

        # --- 创建模型 ---
        backbone = timm.create_model(cfg.get("backbone", "mobilenetv3_large_100"),
                                     pretrained=True,
                                     features_only=True,
                                     out_indices=(2, 3, 4))
        model = MoveNet_FPN_SP(
            backbone=backbone,
            num_joints=cfg.get("num_joints", 17),
            fpn_out_channels=cfg.get("fpn_out_channels", 128),
            head_midc=cfg.get("head_midc", 32),
            upsample_to_quarter=cfg.get("upsample_to_quarter", True)
        )

        # --- 确定训练设备 ---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Loss ---
        loss_fn = MoveNet2HeadLoss(
            num_joints=cfg.get("num_joints", 17),
            hm_weight=cfg.get("hm_weight", 1.0),
            off_weight=cfg.get("off_weight", 0.1),
            focal_alpha=cfg.get("focal_alpha", 2.0),
            focal_beta=cfg.get("focal_beta", 4.0)
        )

        # --- 初始化 super ---
        super().__init__(
            model,
            loss_fn,
            data_dir=cfg.get("data_dir", "/home/user/datasets/coco2017_movenet_sp"),
            save_dir=cfg.get("save_dir", "runs/movenet_fpn_mbv3"),
            device=device,
            resume=cfg.get("resume", False),

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
            **extra_cfg
        )

        # --- 加载数据集 ---
        self.train_dl, self.val_dl = self.create_dataloaders(cfg.get("seed",  random.randrange(1, 100)))


    def create_dataloaders(self, seed = 45):

        # 数据增强相关参数
        aug_cfg = {
            "p_flip": 0.5,
            "scale_min": -0.25,
            "scale_max": 0.25,
            "translate": 0.08,
            "rotate": 30.0,
            "color": True,
            "p_color": 0.8,
        }
        non_aug_cfg = {
            "p_flip": 0.0,
            "scale_min": 0.0,
            "scale_max": 0.0,
            "translate": 0.0,
            "rotate": 0.0,
            "color": False
        }

        # 加载 COCO train
        train_loader = create_kpts_dataloader(
            dataset_root=self.cfg.get("data_dir", "/home/user/datasets/coco2017_movenet_sp"),
            img_size=self.cfg.get("img_size", 192),
            batch_size=self.cfg.get("batch_size", 64),
            target_stride=self.cfg.get("stride", 4),
            num_workers=self.cfg.get("num_workers", 4),
            pin_memory=self.cfg.get("pin_memory", True),
            is_train=True,
            aug_cfg=self.cfg.get("aug_cfg", aug_cfg)
        )

        #  加载 COCO validation
        val_loader = create_kpts_dataloader(
            dataset_root=self.cfg.get("data_dir", "/home/user/datasets/coco2017_movenet_sp"),
            img_size=self.cfg.get("img_size", 192),
            batch_size=self.cfg.get("batch_size", 64),
            target_stride=self.cfg.get("stride", 4),
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
            "hm": 0.0,          # 热图损失
            "off": 0.0,         # 偏移损失
        }
        count = 0  # 已处理的 batch 数量

        # 3. 创建进度条
        pbar = tqdm(
            enumerate(loader, 1),
            total=len(loader),
            ncols=120,
            desc="    Train",  # ← 只保留缩进+名称
            bar_format=self.BAR_FMT,  # ← 使用统一格式
            leave=True  # ← 保留完成后的行
        )

        # 4. 遍历所有训练批次
        for step, (images, labels, kps_masks, _) in pbar:
            # 4.1 加载数据到设备
            # non_blocking=True: 异步传输，提高 GPU 利用率
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            kps_masks = kps_masks.to(self.device, non_blocking=True)

            # 4.2 清空梯度
            # set_to_none=True: 比 zero_grad() 更高效（直接释放内存）
            optimizer.zero_grad(set_to_none=True)

            # 4.3 前向传播（使用混合精度）
            # autocast: 自动将部分操作转为 float16，加速训练
            ph, pw = None, None
            with autocast(device_type=device.type, enabled=self.use_amp, dtype=torch.float16):
                # 模型预测
                preds = model(images)

                # 对生成的热力图维度确认
                ph, pw = preds["heatmaps"].shape[-2:]
                gh, gw = labels.shape[-2:]
                if (ph, pw) != (gh, gw):
                    raise ValueError(f"Pred/GT spatial mismatch: pred={ph}x{pw}, gt={gh}x{gw}. "
                                     f"Hint: set Cfg.upsample_to_quarter=True for stride=4 labels, "
                                     f"or set Cfg.stride=8 to match stride-8 outputs.")

                # 计算损失（返回总损失和各项分损失）
                loss, details = self.loss_fn(
                    preds["heatmaps"],
                    preds["offsets"],
                    labels,
                    kps_masks)

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
            running["hm"]    += details["loss_heatmap"].item()
            running["off"]   += details["loss_offsets"].item()
            count += 1

            # 4.8 更新进度条显示, 显示当前平均损失和学习率
            # pbar.set_postfix({
            #     "loss": f"{running['total'] / count:.4f}",
            #     "hm":   f"{running['hm'] / count:.4f}",
            #     "off":  f"{running['off'] / count:.4f}",
            #     "sz": f"{ph}x{pw}",
            #     "lr": f"{optimizer.param_groups[0]['lr']:.2e}"  # 当前学习率
            # })

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
            "hm": 0.0,          # 热图损失
            "off": 0.0,         # 偏移损失
        }

        # 3. 创建进度条
        pbar = tqdm(
            enumerate(loader, 1),
            total=len(loader),
            ncols=120,
            desc="    Valid",  # ← 缩进+名称
            bar_format=self.BAR_FMT,  # ← 使用统一格式
            leave=True  # ← 保留完成后的行
        )

        # 4. 遍历所有训练批次
        for step, (images, labels, kps_masks, _) in pbar:
            # 4.1 加载数据到设备
            # non_blocking=True: 异步传输，提高 GPU 利用率
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            kps_masks = kps_masks.to(self.device, non_blocking=True)

            # 4.2 前向传播（使用混合精度）
            with autocast(device_type=device.type, enabled=self.use_amp, dtype=torch.float16):
                # 预测
                preds = model(images)

                # 对生成的热力图维度确认
                ph, pw = preds["heatmaps"].shape[-2:]
                gh, gw = labels.shape[-2:]
                if (ph, pw) != (gh, gw):
                    raise ValueError(f"Pred/GT spatial mismatch: pred={ph}x{pw}, gt={gh}x{gw}. "
                                     f"Hint: set Cfg.upsample_to_quarter=True for stride=4 labels, "
                                     f"or set Cfg.stride=8 to match stride-8 outputs.")

                # 计算损失（返回总损失和各项分损失）
                loss, details = self.loss_fn(
                    preds["heatmaps"],
                    preds["offsets"],
                    labels,
                    kps_masks)

            # 4.3. 累加损失
            running["total"] += loss.detach().item()
            running["hm"] += details["loss_heatmap"].item()
            running["off"] += details["loss_offsets"].item()

        # 5. 计算平均损失，并返回
        n_batches = len(loader)
        return {k: v / max(1, n_batches) for k, v in running.items()}


    def train_model(self):

        # Set training seed
        set_seed(self.cfg.get("seed", random.randrange(1, 100)))

        # Resume the training process
        if self.resume:
            start_epoch, best_val = load_ckpt_if_any(
                model=self.model,
                ckpt_path=os.path.join(self.save_dir, "last.pt"),
                optimizer=self.optimizer,
                scaler=self.scaler,
                device=self.device)
            logger.info("Sparrow", "Continue MoveNet from previous checkpoint.")
        else:
            start_epoch = 0
            best_val = float("inf")
            logger.info("Sparrow", "Training MoveNet from beginning.")

        # Training logs to plot
        hist = dict(train=[], val=[])

        # Training the model
        for epoch in range(start_epoch, self.epochs):

            # 打印 epoch 头
            print(f"Epoch {epoch + 1}/{self.epochs}:")  # ← 单独一行

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
                "best_val": best_val
            }, self.save_dir, "last.pt")

            # save best
            if val_loss["total"] < best_val:
                best_val = val_loss["total"]
                best_path = save_ckpt({
                    "epoch": epoch + 1,
                    "model": self.model.state_dict(),
                    "optim": self.optimizer.state_dict(),
                    "scaler": self.scaler.state_dict(),
                    "best_val": best_val
                }, self.save_dir, "best.pt")
                logger.info("Sparrow", f"Best checkpoint saved to {best_path}")
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

    def export_onnx(self):
        # --- 包装模型 ---
        wrapper = MoveNetExportWrapper(self.model)
        wrapper.eval().to(self.device)

        # --- 创建 dummy 输入，并移动到相同设备 ---
        dummy = torch.randn(1, 3, 192, 192, device=self.device)

        # --- 输出路径 ---
        save_path = os.path.join(self.save_dir, "export.onnx")
        os.makedirs(self.save_dir, exist_ok=True)

        # Create an ONNX file
        torch.onnx.export(
            wrapper,
            dummy,
            save_path,
            input_names=["images"],
            output_names=["output"],
            dynamic_axes={"images": {0: "batch"}, "output": {0: "batch"}},
            opset_version=13
        )

        print(f"[export] ONNX model saved to {save_path}")


    @staticmethod
    @torch.no_grad()
    def _decode_predictions(heatmaps: torch.Tensor, offsets: torch.Tensor, stride: int) -> torch.Tensor:
        """
        Helper to decode heatmaps and offsets into keypoint coordinates (x, y, score).
        This can be used for both model predictions and ground truth labels.
        """
        batch_size, num_joints, h, w = heatmaps.shape
        # Heatmaps are already probabilities (if GT) or logits (if pred), sigmoid handles both
        heatmaps = torch.sigmoid(heatmaps)
        scores, inds = torch.max(heatmaps.view(batch_size, num_joints, -1), dim=2)
        y_coords = (inds / w).int().float()
        x_coords = (inds % w).int().float()

        offsets = offsets.view(batch_size, num_joints, 2, h, w)
        offset_x = offsets[:, :, 0, :, :].view(batch_size, num_joints, -1).gather(2, inds.unsqueeze(-1)).squeeze(-1)
        offset_y = offsets[:, :, 1, :, :].view(batch_size, num_joints, -1).gather(2, inds.unsqueeze(-1)).squeeze(-1)

        pred_x = (x_coords + offset_x) * stride
        pred_y = (y_coords + offset_y) * stride

        return torch.stack([pred_x, pred_y, scores], dim=2)

    def run_evaluation(self) -> Dict[str, float]:
        """
        重写基类的评估方法，以运行 PCK (Percentage of Correct Keypoints) 评估。
        此版本无需修改Dataset，直接从GT label张量中解码真实坐标。
        """
        print("\n" + "=" * 30)
        print("Running PCK Evaluation on Validation Set...")
        self.model.eval()

        pck_thresholds = [0.05, 0.10, 0.20]
        img_size = self.cfg.get("img_size", 192)
        stride = self.cfg.get("stride", 4)
        num_joints = self.cfg.get("num_joints", 17)

        thresholds_px = {f"PCK@{thr:.2f}": thr * img_size for thr in pck_thresholds}
        correct_kpts = {key: 0 for key in thresholds_px.keys()}
        total_visible_kpts = 0

        pbar = tqdm(self.val_dl, desc="[Evaluator] Calculating PCK", ncols=110)
        # The loop remains unchanged, expecting 4 items from the dataloader
        for images, labels, kps_masks, _ in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            kps_masks = kps_masks.to(self.device)

            # 1. 模型推理和解码预测坐标
            preds = self.model(images)
            pred_kpts = self._decode_predictions(
                preds["heatmaps"],
                preds["offsets"],
                stride
            ).to(self.device)

            # 2. 从标签张量中解码真实坐标
            gt_hm = labels[:, :num_joints]
            gt_off = labels[:, num_joints:]
            gt_kpts = self._decode_predictions(gt_hm, gt_off, stride).to(self.device)

            # 3. 计算PCK (逻辑与之前完全相同)
            visible_mask = kps_masks > 0

            if visible_mask.sum() == 0:
                continue

            total_visible_kpts += visible_mask.sum().item()
            errors = torch.linalg.norm(pred_kpts[:, :, :2] - gt_kpts[:, :, :2], dim=2)
            errors[~visible_mask] = float('inf')

            for key, thr_px in thresholds_px.items():
                correct_kpts[key] += (errors <= thr_px).sum().item()

        # 4. 计算最终结果
        metrics = {}
        if total_visible_kpts > 0:
            for key, count in correct_kpts.items():
                metrics[key] = count / total_visible_kpts
        else:
            for key in thresholds_px.keys():
                metrics[key] = 0.0

        print("=" * 30 + "\n")
        return metrics