import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn

from sparrow.utils.logger import logger


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_ckpt_if_any(
        model: nn.Module,
        ckpt_path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        device: Optional[torch.device] = None

) -> tuple[int, float]:
    """
    加载训练检查点（如果存在）- 用于恢复训练或模型推理

    功能：
      1. 加载模型权重（必须）
      2. 加载优化器状态（可选，用于恢复训练）
      3. 加载混合精度训练的 scaler 状态（可选）
      4. 恢复训练进度信息（epoch 和最佳验证指标）

    典型使用场景：
      - 训练中断后恢复训练（resume training）
      - 加载预训练模型进行微调（fine-tuning）
      - 加载最佳模型进行推理（inference）

    参数:
      model: PyTorch 模型实例
      ckpt_path: 检查点文件路径（.pth 或 .pt 文件）
      optimizer: 优化器实例（如 Adam, SGD），训练时传入，推理时可为 None
      scaler: 混合精度训练的 GradScaler 实例，不使用 AMP 时可为 None
      device: 目标设备（'cuda' 或 'cpu'），用于加载张量到正确的设备
              如果为 None，自动从模型参数推断设备

    返回:
      (start_epoch, best): 元组
        - start_epoch: 起始训练轮数
            - 如果成功加载检查点：返回保存时的 epoch
            - 如果未加载：返回 0（从头开始训练）
        - best: 最佳验证指标（这里是角度误差，单位：度）
            - 如果成功加载：返回保存的最佳验证结果
            - 如果未加载：返回 inf（表示还没有验证结果）

    检查点文件结构（字典）：
      {
          "model": model.state_dict(),          # 模型权重（必需）
          "optim": optimizer.state_dict(),      # 优化器状态（可选）
          "scaler": scaler.state_dict(),        # AMP scaler 状态（可选）
          "epoch": int,                         # 当前训练轮数（可选）
          "best_val_deg": float                 # 最佳验证角度误差（可选）
      }

    示例:
      # 场景 1：恢复训练（自动检测设备）
      start_epoch, best_val = load_ckpt_if_any(
          model, "checkpoints/last.pth", optimizer, scaler
      )

      # 场景 2：加载到指定设备
      start_epoch, best_val = load_ckpt_if_any(
          model, "checkpoints/last.pth", optimizer, scaler,
          device=torch.device("cuda:0")
      )

      # 场景 3：推理模式（无 optimizer 和 scaler）
      start_epoch, _ = load_ckpt_if_any(model, "checkpoints/best.pth")
    """
    # 自动检测设备：如果未指定，从模型参数推断
    if device is None:
        # 尝试获取模型第一个参数的设备
        try:
            device = next(model.parameters()).device
        except StopIteration:
            # 如果模型没有参数（罕见情况），默认使用 CPU
            device = torch.device("cpu")
            print("[warning] Model has no parameters, defaulting to CPU")

    # 检查检查点文件是否存在
    if ckpt_path and os.path.isfile(ckpt_path):
        # 1. 加载检查点文件
        # map_location: 指定加载到的设备，避免 GPU/CPU 设备不匹配的问题
        # 例如：在 CPU 上加载 GPU 训练的模型，或在不同 GPU 上加载
        ckpt = torch.load(ckpt_path, map_location=device)

        # 2. 加载模型权重（必须操作）
        # strict=True: 要求检查点中的 keys 与模型完全匹配
        #   - 缺少的 keys 或多余的 keys 都会报错
        #   - 确保模型结构一致性
        model.load_state_dict(ckpt["model"], strict=True)

        # 3. 加载优化器状态（可选，仅在继续训练时需要）
        # 优化器状态包括：
        #   - 动量信息（Momentum, Adam 的 m/v）
        #   - 学习率调度器状态
        #   - 累积梯度等
        # 作用：确保训练从中断处无缝继续
        if optimizer is not None and "optim" in ckpt:
            optimizer.load_state_dict(ckpt["optim"])

        # 4. 加载混合精度训练的 scaler 状态（可选）
        # GradScaler 用于自动混合精度（AMP）训练
        # 状态包括：
        #   - 梯度缩放因子（防止梯度下溢）
        #   - 缩放历史记录
        # 作用：确保混合精度训练的稳定性
        if scaler is not None and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])

        # 5. 恢复训练进度信息
        # 获取上次保存时的 epoch（默认为 0）
        start_epoch = ckpt.get("epoch", 0)

        # 获取最佳验证指标（这里是角度误差，单位：度）
        # 默认为 inf，表示还没有最佳结果
        best = ckpt.get("best_val_deg", float("inf"))

        # 6. 打印恢复信息（方便调试和日志记录）
        logger.info("load_ckpt_if_any", f"[weights] loaded {ckpt_path} "
                                        f"(epoch={start_epoch}, best={best:.3f}°, "
                                        f"device={device})")

        # 返回训练起始信息
        return start_epoch, best

    # 如果检查点不存在，从头开始训练
    # start_epoch=0: 从第 0 轮开始
    # best=inf: 还没有最佳验证结果
    return 0, float("inf")


def save_ckpt(state, out_dir, name):
    out = Path(out_dir);
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    torch.save(state, path)
    logger.info("save_ckpt", f"[weights] saved {path}")

    return str(path)