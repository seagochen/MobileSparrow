import json
import math
import os.path
import random
from typing import Dict, Optional

import timm
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm

from sparrow.datasets.biwi_rotation import BIWIDataset
from sparrow.losses.sixrepnet_loss import SixDCombinedLoss
from sparrow.models.sixrepnet import SixDRepNet
from sparrow.trainer.base_trainer import BaseTrainer
from sparrow.trainer.components import clip_gradient, set_seed, load_ckpt_if_any, save_ckpt
from sparrow.utils.logger import logger
from sparrow.utils.plot_curves import plot_training_curves
from sparrow.utils.yaml_config import update_from_yaml


class SixDRepNetTrainer(BaseTrainer):

    def __init__(self, yaml_path: Optional[str] = None):

        # --- 加载训练配置信息 ---
        cfg = update_from_yaml(yaml_path)

        # --- 创建模型 ---
        backbone = timm.create_model(cfg.get("backbone", "mobilenetv3_large_100"),
                                     pretrained=True,
                                     num_classes=0)
        model = SixDRepNet(backbone)

        # --- 确定训练设备 ---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Loss ---
        loss_fn = SixDCombinedLoss(
            w_geo=cfg.get("w_geo", 1.0),
            w_col=cfg.get("w_col", 0.5),
            w_reg=cfg.get("w_reg", 0.1)
        )

        # --- 初始化 super ---
        super().__init__(
            model,
            loss_fn,
            data_dir=cfg.get("data_dir", "/home/user/datasets/biwi"),
            save_dir=cfg.get("save_dir", "runs/biwi_sixd_mbv3"),
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
            clip_grad_norm = cfg.get("clip_grad_norm", 0.0),

            # 其他参数
            **cfg
        )

        # --- 加载数据集 ---
        self.train_dl, self.val_dl = self.create_dataloaders(cfg.get("seed",  random.randrange(1, 100)))


    def create_dataloaders(self, seed = 42):

        # Load the dataset from the path
        full = BIWIDataset(root_dir=self.data_dir, img_size=224, crop_size=256, use_crop=True, return_path=False)

        # Split the dataset into train and val parts
        full_size = len(full)
        val_size = int(full_size * 0.1)
        train_size = full_size - val_size
        g = torch.Generator().manual_seed(seed)
        train_set, val_set = random_split(full, [train_size, val_size], generator=g)

        # Wrap the dataset with DataLoader
        def dataloader(ds, shuffle):
            return DataLoader(ds,
                              batch_size=self.cfg.get("batch_size", 64),
                              shuffle=shuffle,
                              num_workers=self.cfg.get("workers", 8),
                              pin_memory=True,
                              drop_last=shuffle)

        # Generate training and validation dataloaders
        return dataloader(train_set, True), dataloader(val_set, False)


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
            "total": 0.0,  # 总损失（加权组合）
            "geo": 0.0,  # 测地距离损失（主损失）
            "col": 0.0,  # 列向量对齐损失（辅助损失）
            "reg": 0.0  # 正则化损失（可选）
        }
        count = 0  # 已处理的 batch 数量

        # 3. 创建进度条（不保留到屏幕，避免多 epoch 时刷屏）
        pbar = tqdm(loader, desc="Train", leave=False)

        # 4. 遍历所有训练批次
        for it, batch in enumerate(pbar, 1):  # it 从 1 开始计数
            # 4.1 加载数据到设备
            # non_blocking=True: 异步传输，提高 GPU 利用率
            imgs = batch["image"].to(device, non_blocking=True)  # [B, 3, H, W]
            R_gt = batch["R_gt"].to(device, non_blocking=True)  # [B, 3, 3] - 真值旋转矩阵

            # 4.2 清空梯度
            # set_to_none=True: 比 zero_grad() 更高效（直接释放内存）
            optimizer.zero_grad(set_to_none=True)

            # 4.3 前向传播（使用混合精度）
            # autocast: 自动将部分操作转为 float16，加速训练
            with autocast(device_type=device.type, enabled=self.use_ema, dtype=torch.float16):
                # 模型预测 6D 向量
                pred_6d = model(imgs)  # [B, 6]

                # 将 6D 向量转换为旋转矩阵（Gram-Schmidt 正交化）
                R_pred = model.compute_rotation_matrix_from_orthod(pred_6d)  # [B, 3, 3]

                # 计算损失（返回总损失和各项分损失）
                total, parts = loss_fn(R_pred, pred_6d, R_gt)
                # total: 标量总损失
                # parts: 字典 {'total_loss', 'geodesic_loss', 'column_loss', 'regularizer_loss'}

            # 4.4 反向传播（使用梯度缩放）
            # 原因：float16 计算时梯度可能下溢，需要缩放
            scaler.scale(total).backward()

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

            # 4.7 累加损失（用于计算平均）
            running["total"] += float(parts["total_loss"].item())
            running["geo"] += float(parts["geodesic_loss"].item())
            running["col"] += float(parts["column_loss"].item())
            running["reg"] += float(parts["regularizer_loss"].item())
            count += 1

            # 4.8 更新进度条显示, 显示当前平均损失和学习率
            pbar.set_postfix({
                "loss": f"{running['total'] / count:.4f}",  # 平均总损失
                "geo": f"{running['geo'] / count:.4f}",  # 平均测地损失
                "col": f"{running['col'] / count:.4f}",  # 平均列向量损失
                "reg": f"{running['reg'] / count:.4f}",  # 平均正则化损失
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
                 device: torch.device) -> tuple[Dict[str, float], Dict[str, float]]:
        """
        验证模型性能（头部姿态估计任务）

        功能：
          在验证集上评估模型，计算损失和角度误差指标

        评估指标：
          1. 损失指标：测地距离、列向量对齐、正则化
          2. 角度误差：平均误差（mean）和中位数误差（median）

        参数:
          model: 验证模型
          loss_fn: 损失函数
          loader: 验证数据加载器
          device: 设备
          cfg: 配置字典（包含 use_amp）

        返回:
          (loss_avgs, deg_metrics)
          - loss_avgs: 平均损失字典 {'total', 'geo', 'col', 'reg'}
          - deg_metrics: 角度误差字典 {'deg_mean', 'deg_median'}（单位：度）

        说明:
          - 使用 @torch.no_grad() 装饰器禁用梯度计算（节省内存和时间）
          - model.eval() 禁用 Dropout 和 BatchNorm 的训练行为
          - 中位数误差通常比平均值更鲁棒（不受异常值影响）
        """

        # 1. 设置模型为评估模式
        # 效果：禁用 Dropout，BatchNorm 使用全局统计
        model.eval()

        # 2. 初始化累加器
        agg_loss = {
            "total": 0.0,
            "geo": 0.0,
            "col": 0.0,
            "reg": 0.0
        }
        agg_deg = []  # 存储每个样本的角度误差（用于计算统计量）

        # 3. 创建进度条
        pbar = tqdm(loader, desc="Valid", leave=False)

        # 4. 遍历验证集（无需梯度）
        for batch in pbar:
            # 4.1 加载数据
            imgs = batch["image"].to(device, non_blocking=True)
            R_gt = batch["R_gt"].to(device, non_blocking=True)

            # 4.2 前向传播（使用混合精度）
            with autocast(device_type=device.type, enabled=self.use_ema, dtype=torch.float16):
                # 预测 6D 向量并转换为旋转矩阵
                pred_6d = model(imgs)
                R_pred = model.compute_rotation_matrix_from_orthod(pred_6d)

                # 计算损失
                total, parts = loss_fn(R_pred, pred_6d, R_gt)

            # 4.3 累加损失
            agg_loss["total"] += float(parts["total_loss"].item())
            agg_loss["geo"] += float(parts["geodesic_loss"].item())
            agg_loss["col"] += float(parts["column_loss"].item())
            agg_loss["reg"] += float(parts["regularizer_loss"].item())

            # 4.4 计算角度误差（每个样本的测地角度，单位：度）
            deg = self.geodesic_deg(R_pred, R_gt)  # [B] - 每个样本的误差
            agg_deg.append(deg.cpu())  # 移到 CPU 节省 GPU 内存

        # 5. 计算平均损失
        n_batches = len(loader)
        loss_avgs = {k: v / max(1, n_batches) for k, v in agg_loss.items()}

        # 6. 计算角度误差统计量
        # 拼接所有批次的角度误差
        deg_all = torch.cat(agg_deg) if agg_deg else torch.tensor([])

        # 计算平均值和中位数
        deg_mean = float(deg_all.mean().item()) if deg_all.numel() else 0.0
        deg_med = float(deg_all.median().item()) if deg_all.numel() else 0.0

        # 7. 返回损失和角度指标
        return loss_avgs, {"deg_mean": deg_mean, "deg_median": deg_med}


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
            logger.info("Sparrow", "Continue 6DRepNet from previous checkpoint.")
        else:
            start_epoch = 0
            best_val = float("inf")
            logger.info("Sparrow", "Training the 6DRepNet from beginning.")

        # Training logs to plot
        hist = dict(train=[], val=[], deg_mean=[], deg_median=[])

        # Training the model
        for epoch in range(start_epoch, self.epochs):

            # Train the model
            tr = self.train_one_epoch(
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
            val_loss, val_metrics = self.evaluate(
                model=self.model,
                loss_fn=self.loss_fn,
                epoch=epoch,
                loader=self.val_dl,
                device=self.device)

            hist["train"].append(tr)
            hist["val"].append(val_loss)
            hist["deg_mean"].append(val_metrics["deg_mean"])
            hist["deg_median"].append(val_metrics["deg_median"])

            # Output the metrics
            row = {
                "epoch": epoch + 1, "train": tr, "val": val_loss,
                "val_deg_mean": val_metrics["deg_mean"],
                "val_deg_median": val_metrics["deg_median"],
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
            if val_metrics["deg_mean"] < best_val:
                best_val = val_metrics["deg_mean"]
                best_path = save_ckpt({
                    "epoch": epoch + 1,
                    "model": self.model.state_dict(),
                    "optim": self.optimizer.state_dict(),
                    "scaler": self.scaler.state_dict(),
                    "best_val_deg": best_val
                }, self.save_dir, "best.pt")
                print(f"[best] new best mean geodesic = {best_val:.3f}°  ->  {best_path}")
            # end-for: epoch in range(start_epoch, self.epochs)

        # Get the hist curves
        train_total_loss_hist = [t["total"] for t in hist["train"]]
        val_total_loss_hist = [t["total"] for t in hist["val"]]
        train_geo_loss_hist =  [t["geo"] for t in hist["train"]]
        val_geo_loss_hist = [v["geo"] for v in hist["val"]]
        deg_mean_hist = hist["deg_mean"]
        deg_median_hist = hist["deg_median"]

        # Plot the training curves
        plot_training_curves(
            suptitle="Head Pose Training Progress",
            save_path=os.path.join(self.save_dir, "training_curves.png"),
            curves_config=[
                {
                    'train_vals': train_total_loss_hist,
                    'val_vals': val_total_loss_hist,
                    'title': 'Total Loss',
                    'ylabel': 'Loss'
                },
                {
                    'train_vals': train_geo_loss_hist,
                    'val_vals': val_geo_loss_hist,
                    'title': 'Geodesic Loss',
                    'ylabel': 'Geodesic (rad)'
                },
                {
                    'deg_mean': deg_mean_hist,
                    'deg_median': deg_median_hist,
                    'title': 'Geodesic Loss',
                    'ylabel': 'Geodesic (rad)'
                },
            ]
        )

    def export_onnx(self, model: nn.Module):
        raise NotImplemented

    def export_wrapper(self, model: nn.Module):
        raise NotImplemented

    @staticmethod
    @torch.no_grad()
    def geodesic_deg(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
        """
        计算逐样本的测地角度误差（单位：度）

        功能：
          测量预测旋转矩阵和真值旋转矩阵之间的角度差异

        数学原理：
          1. 计算相对旋转：R_rel = R_pred * R_gt^T
          2. 从 R_rel 提取旋转角度 θ
          3. 使用 atan2(sin, cos) 稳定计算

          公式：
            sin(θ) ∝ ||vee(R_rel - R_rel^T)||  # 反对称部分
            cos(θ) ∝ (trace(R_rel) - 1) / 2    # 迹
            θ = atan2(sin, cos)

        参数:
          R_pred: 预测旋转矩阵 [B, 3, 3]
          R_gt: 真值旋转矩阵 [B, 3, 3]

        返回:
          theta: 角度误差 [B]，单位：度，范围 [0, 180]

        优势：
          - 数值稳定：使用 atan2 避免 arccos 的梯度消失
          - 物理意义：直接测量旋转角度（易于理解和评估）

        示例:
          >>> R_pred = torch.eye(3).unsqueeze(0)  # 单位矩阵
          >>> R_gt = torch.eye(3).unsqueeze(0)
          >>> deg = geodesic_deg(R_pred, R_gt)
          >>> print(deg)  # tensor([0.])
        """
        # 1. 计算相对旋转矩阵：R_rel = R_pred * R_gt^T
        # 物理含义：将 GT 旋转到预测姿态所需的旋转
        R_rel = torch.bmm(R_pred, R_gt.transpose(1, 2))  # [B, 3, 3]

        # 2. 计算 trace（对角线元素之和）
        # trace(R) = 1 + 2*cos(θ)
        trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]  # [B]

        # 3. 计算反对称矩阵：skew = R_rel - R_rel^T
        # 反对称矩阵编码了旋转轴信息
        skew = R_rel - R_rel.transpose(1, 2)  # [B, 3, 3]

        # 4. 提取反对称矩阵的三个独立分量
        # 反对称矩阵的形式：
        #     [  0   -v3   v2 ]
        #     [ v3    0   -v1 ]
        #     [-v2   v1    0  ]
        v1 = skew[:, 2, 1] - skew[:, 1, 2]  # 2*v1
        v2 = skew[:, 0, 2] - skew[:, 2, 0]  # 2*v2
        v3 = skew[:, 1, 0] - skew[:, 0, 1]  # 2*v3

        # 5. 计算 sin(θ)
        # ||vee(R - R^T)|| = 2*sin(θ)
        # clamp 防止数值误差导致负数
        s = torch.sqrt(torch.clamp(v1 * v1 + v2 * v2 + v3 * v3, min=1e-7))  # [B]

        # 6. 计算 cos(θ)
        # trace(R) = 1 + 2*cos(θ) => cos(θ) = (trace - 1) / 2
        # clamp 确保 cos 在 [-1, 1] 范围内（理论上 trace ∈ [-1, 3]）
        c = torch.clamp(trace - 1.0, min=-2 + 1e-7, max=2 - 1e-7)  # [B]

        # 7. 使用 atan2 计算角度（弧度）
        # atan2 的优势：
        #   - 输出范围 [-π, π]
        #   - 在 sin 和 cos 同时接近 0 时仍然稳定
        #   - 无梯度消失问题
        theta = torch.atan2(s, c)  # [B] - 弧度

        # 8. 转换为角度（度）
        return theta * (180.0 / math.pi)  # [B] - 度数

