import warnings
from typing import Iterator
from typing import Union, Optional, Any

import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sparrow.utils.logger import logger


def select_optimizer(
        name: str,
        model: Union[nn.Module, Iterator[nn.Parameter]],
        lr: float,
        weight_decay: float = 0.0,
        **kw
) -> torch.optim.Optimizer:

    # ========== 1. 参数提取（兼容两种输入方式）==========
    # 检查是否传入的是模型（有 parameters 方法）还是参数迭代器
    params = model.parameters() if hasattr(model, "parameters") else model

    # ========== 2. 类型转换（处理 YAML 配置）==========
    # YAML 解析器可能将数值读取为字符串，需要强制转换
    try:
        lr = float(lr)
        weight_decay = float(weight_decay)
    except (ValueError, TypeError) as e:
        # 直接抛出异常，并附带清晰的错误信息
        raise ValueError(
            f"Failed to convert 'lr' or 'weight_decay' to float. "
            f"Received lr='{lr}' (type: {type(lr).__name__}), "
            f"weight_decay='{weight_decay}' (type: {type(weight_decay).__name__}). "
            "Please ensure they are valid numbers."
        ) from e

    # ========== 3. 优化器选择（不区分大小写）==========
    name = (name or "").lower().strip()

    # -------------------- Adam 系列 --------------------
    # Adam: 自适应学习率，快速收敛（经典）
    if name in ("adam",):
        return torch.optim.Adam(
            params,
            lr=lr,
            betas=kw.get("betas", (0.9, 0.999)),  # 一阶和二阶动量衰减系数
            eps=kw.get("eps", 1e-8),  # 数值稳定性
            weight_decay=weight_decay,
            amsgrad=kw.get("amsgrad", False)  # AMSGrad 变体（更保守）
        )

    # AdamW: Adam 的改进版（解耦权重衰减）✅ 推荐首选
    elif name in ("adamw", "adam_w"):
        return torch.optim.AdamW(
            params,
            lr=lr,
            betas=kw.get("betas", (0.9, 0.999)),
            eps=kw.get("eps", 1e-8),
            weight_decay=weight_decay,
            amsgrad=kw.get("amsgrad", False)
        )

    # Adamax: Adam 的变体（使用无穷范数）
    elif name in ("adamax",):
        return torch.optim.Adamax(
            params,
            lr=lr,
            betas=kw.get("betas", (0.9, 0.999)),
            eps=kw.get("eps", 1e-8),
            weight_decay=weight_decay
        )

    # NAdam: Adam + Nesterov 动量
    elif name in ("nadam",):
        return torch.optim.NAdam(
            params,
            lr=lr,
            betas=kw.get("betas", (0.9, 0.999)),
            eps=kw.get("eps", 1e-8),
            weight_decay=weight_decay,
            momentum_decay=kw.get("momentum_decay", 4e-3)
        )

    # RAdam: 修正版 Adam（鲁棒的预热）
    elif name in ("radam",):
        return torch.optim.RAdam(
            params,
            lr=lr,
            betas=kw.get("betas", (0.9, 0.999)),
            eps=kw.get("eps", 1e-8),
            weight_decay=weight_decay
        )

    # -------------------- SGD 系列 --------------------
    # SGD: 经典梯度下降（配合动量和学习率调度器效果好）
    elif name in ("sgd",):
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=kw.get("momentum", 0.9),  # 动量系数（0.9 是常用值）
            weight_decay=weight_decay,
            dampening=kw.get("dampening", 0),  # 动量阻尼
            nesterov=kw.get("nesterov", True)  # Nesterov 加速梯度
        )

    # -------------------- RMSprop --------------------
    # RMSprop: 自适应学习率（适合 RNN）
    elif name in ("rmsprop", "rms"):
        return torch.optim.RMSprop(
            params,
            lr=lr,
            alpha=kw.get("alpha", 0.99),  # 平滑常数
            eps=kw.get("eps", 1e-8),
            weight_decay=weight_decay,
            momentum=kw.get("momentum", 0),
            centered=kw.get("centered", False)  # 中心化梯度
        )

    # -------------------- AdaGrad 系列 --------------------
    # AdaGrad: 自适应学习率（适合稀疏梯度）
    elif name in ("adagrad", "ada"):
        return torch.optim.Adagrad(
            params,
            lr=lr,
            lr_decay=kw.get("lr_decay", 0),  # 学习率衰减
            weight_decay=weight_decay,
            eps=kw.get("eps", 1e-10)
        )

    # Adadelta: AdaGrad 的改进（无需手动设置学习率）
    elif name in ("adadelta",):
        return torch.optim.Adadelta(
            params,
            lr=lr,  # 通常设为 1.0
            rho=kw.get("rho", 0.9),  # 衰减率
            eps=kw.get("eps", 1e-6),
            weight_decay=weight_decay
        )

    # -------------------- 高级优化器 --------------------
    # LAMB: 大批次训练优化器（需要额外安装）
    # pip install pytorch-lamb
    elif name in ("lamb",):
        try:
            from pytorch_lamb import Lamb
            return Lamb(
                params,
                lr=lr,
                betas=kw.get("betas", (0.9, 0.999)),
                eps=kw.get("eps", 1e-8),
                weight_decay=weight_decay
            )
        except ImportError:
            raise ImportError(
                "LAMB optimizer requires 'pytorch-lamb' package.\n"
                "Install it with: pip install pytorch-lamb"
            )

    # AdamP: 投影约束的 Adam（需要额外安装）
    # pip install adamp
    elif name in ("adamp",):
        try:
            from adamp import AdamP
            return AdamP(
                params,
                lr=lr,
                betas=kw.get("betas", (0.9, 0.999)),
                eps=kw.get("eps", 1e-8),
                weight_decay=weight_decay,
                delta=kw.get("delta", 0.1),  # 投影半径
                wd_ratio=kw.get("wd_ratio", 0.1)  # 权重衰减比例
            )
        except ImportError:
            raise ImportError(
                "AdamP optimizer requires 'adamp' package.\n"
                "Install it with: pip install adamp"
            )

    # ========== 4. 未找到优化器 ==========
    # 提供有用的错误信息
    else:
        supported = [
            "adam", "adamw", "adamax", "nadam", "radam",
            "sgd", "rmsprop", "adagrad", "adadelta",
            "lamb (needs pytorch-lamb)", "adamp (needs adamp)"
        ]
        raise ValueError(
            f"Unknown optimizer: '{name}'\n"
            f"Supported optimizers: {', '.join(supported)}"
        )


def build_scheduler(
        name: str,
        optimizer: torch.optim.Optimizer,
        epochs: Optional[int] = None,
        **kwargs
) -> Union[
    optim.lr_scheduler.LRScheduler,
    optim.lr_scheduler.ReduceLROnPlateau
]:
    # ========== 参数提取辅助函数 ==========
    def _get_param(key: str, default: Any, cast_type: type = None) -> Any:
        """
        从 kwargs 中安全提取参数，支持类型转换和默认值

        参数:
          key: 参数名
          default: 默认值
          cast_type: 目标类型（如 int, float）

        返回:
          转换后的参数值
        """
        value = kwargs.get(key, default)
        if value is None:
            return default

        # 如果指定了类型转换
        if cast_type is not None:
            try:
                return cast_type(value)
            except (ValueError, TypeError):
                warnings.warn(
                    f"Failed to convert '{key}={value}' to {cast_type.__name__}, "
                    f"using default: {default}"
                )
                return default
        return value

    # ========== 标准化调度器名称 ==========
    name = (name or "").lower().strip()

    # ========== 1. StepLR: 固定间隔衰减 ==========
    # 每隔 step_size 个 epoch，学习率乘以 gamma
    # 公式: lr = initial_lr * gamma^(epoch // step_size)
    if name in ("steplr", "step"):
        step_size = _get_param("step_size", 30, int)
        gamma = _get_param("gamma", 0.1, float)
        last_epoch = _get_param("last_epoch", -1, int)

        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
            last_epoch=last_epoch
        )

    # ========== 2. MultiStepLR: 里程碑衰减 ==========
    # 在指定的 epochs 处衰减学习率
    # 例: milestones=[30,60,90] 表示在第 30/60/90 个 epoch 衰减
    if name in ("multisteplr", "multi_step", "multi-step"):
        milestones = kwargs.get("milestones", None)

        # 验证 milestones 参数
        if milestones is None:
            raise ValueError(
                "MultiStepLR requires 'milestones' parameter (list of epochs).\n"
                "Example: milestones=[30, 60, 90]"
            )

        # 确保 milestones 是列表
        if not isinstance(milestones, (list, tuple)):
            raise TypeError(f"milestones must be a list, got {type(milestones)}")

        gamma = _get_param("gamma", 0.1, float)
        last_epoch = _get_param("last_epoch", -1, int)

        return optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(milestones),
            gamma=gamma,
            last_epoch=last_epoch
        )

    # ========== 3. ExponentialLR: 指数衰减 ==========
    # 每个 epoch 学习率乘以 gamma
    # 公式: lr = initial_lr * gamma^epoch
    if name in ("exponentiallr", "exp", "exponential"):
        gamma = _get_param("gamma", 0.95, float)
        last_epoch = _get_param("last_epoch", -1, int)

        return optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=gamma,
            last_epoch=last_epoch
        )

    # ========== 4. CosineAnnealingLR: 余弦退火 ==========
    # 学习率按余弦函数从初始值平滑下降到最小值
    # 优点：训练后期学习率变化平缓，有利于收敛
    if name in ("cosineannealinglr", "cosine", "cos"):
        T_max = epochs if epochs is not None else _get_param("T_max", 100, int)
        eta_min = _get_param("eta_min", 0, float)
        last_epoch = _get_param("last_epoch", -1, int)

        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=eta_min,
            last_epoch=last_epoch
        )

    # ========== 5. CosineAnnealingWarmRestarts: 余弦退火 + 周期重启 ==========
    # SGDR (Stochastic Gradient Descent with Warm Restarts)
    # 周期性重启学习率，帮助跳出局部最优
    if name in ("cosineannealingwarmrestarts", "sgdr", "cosine_restart"):
        T_0 = _get_param("T_0", 10, int)  # 第一次重启的周期
        T_mult = _get_param("T_mult", 2, int)  # 每次重启后周期倍数
        eta_min = _get_param("eta_min", 0, float)
        last_epoch = _get_param("last_epoch", -1, int)

        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=eta_min,
            last_epoch=last_epoch
        )

    # ========== 6. ReduceLROnPlateau: 自适应降低（基于验证指标）==========
    # 当验证指标停止改善时降低学习率
    # 注意：需要在 scheduler.step(val_metric) 中传入验证指标
    if name in ("reducelronplateau", "plateau", "reduce"):
        mode = _get_param("mode", "min", str)  # 'min' 或 'max'
        factor = _get_param("factor", 0.1, float)  # 衰减因子
        patience = _get_param("patience", 10, int)  # 容忍多少 epoch 不改善
        threshold = _get_param("threshold", 1e-4, float)
        min_lr = _get_param("min_lr", 0, float)

        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
            min_lr=min_lr,
        )

    # ========== 7. OneCycleLR: 超收敛训练法 ==========
    # Leslie Smith 提出的超收敛方法
    # 学习率先上升后下降，配合动量反向变化
    if name in ("onecyclelr", "one_cycle", "1cycle"):
        max_lr = _get_param("max_lr", 0.1, float)
        total_steps = kwargs.get("total_steps", None)

        # 计算总步数
        if total_steps is None:
            if epochs is None:
                raise ValueError(
                    "OneCycleLR requires either 'total_steps' or 'epochs' + 'steps_per_epoch'"
                )
            steps_per_epoch = kwargs.get("steps_per_epoch", None)
            if steps_per_epoch is None:
                raise ValueError(
                    "OneCycleLR requires 'steps_per_epoch' when using 'epochs'"
                )
            total_steps = epochs * steps_per_epoch

        pct_start = _get_param("pct_start", 0.3, float)  # 上升阶段占比
        anneal_strategy = _get_param("anneal_strategy", "cos", str)
        div_factor = _get_param("div_factor", 25.0, float)
        final_div_factor = _get_param("final_div_factor", 1e4, float)

        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy=anneal_strategy,
            div_factor=div_factor,
            final_div_factor=final_div_factor
        )

    # ========== 8. LinearLR: 线性衰减 ==========
    # 学习率线性从 start_factor * initial_lr 变化到 end_factor * initial_lr
    if name in ("linearlr", "linear"):
        start_factor = _get_param("start_factor", 1.0, float)
        end_factor = _get_param("end_factor", 0.1, float)
        total_iters = epochs if epochs is not None else _get_param("total_iters", 100, int)
        last_epoch = _get_param("last_epoch", -1, int)

        return optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=start_factor,
            end_factor=end_factor,
            total_iters=total_iters,
            last_epoch=last_epoch
        )

    # ========== 9. PolynomialLR: 多项式衰减 ==========
    # 学习率按多项式函数衰减（常用于语义分割）
    if name in ("polynomiallr", "poly", "polynomial"):
        total_iters = epochs if epochs is not None else _get_param("total_iters", 100, int)
        power = _get_param("power", 1.0, float)
        last_epoch = _get_param("last_epoch", -1, int)

        return optim.lr_scheduler.PolynomialLR(
            optimizer,
            total_iters=total_iters,
            power=power,
            last_epoch=last_epoch
        )

    # ========== 10. ConstantLR / None: 固定学习率 ==========
    # 不调整学习率（用于调试或短期训练）
    if name in ("constant", "none", ""):
        # 返回一个虚拟调度器（LambdaLR 恒等变换）
        return optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: 1.0
        )

    # ========== 未找到调度器 ==========
    supported = [
        "step", "multi_step", "exponential", "cosine", "sgdr",
        "plateau", "one_cycle", "linear", "poly", "constant"
    ]
    raise ValueError(
        f"Unknown scheduler: '{name}'\n"
        f"Supported schedulers: {', '.join(supported)}\n"
        f"See function docstring for details."
    )


def create_warmup_scheduler(
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        main_scheduler: optim.lr_scheduler.LRScheduler,
        start_factor = 0.01,
        end_factor = 1.0
) -> optim.lr_scheduler.SequentialLR:

    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=start_factor,  # 从 1% 的学习率开始
        end_factor=end_factor,
        total_iters=warmup_epochs
    )

    return optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs]
    )


def clip_gradient(
        optimizer,
        max_norm: float = 1.0,
        norm_type: Union[float, int] = 2.0,
        error_if_nonfinite: bool = False
) -> float:
    """
    按“全局范数”裁剪梯度（更常用/更稳定）。
    - 需在 AMP 下先调用 scaler.unscale_(optimizer) 再调用本函数。
    - 返回裁剪前的总梯度范数，用于日志监控。
    """
    # 收集当前参与更新、且有梯度的参数
    params = []
    for group in optimizer.param_groups:
        for p in group.get("params", []):
            if p is not None and p.grad is not None:
                params.append(p)

    if not params:
        return 0.0

    total_norm = torch.nn.utils.clip_grad_norm_(
        params, max_norm, norm_type=norm_type, error_if_nonfinite=error_if_nonfinite
    )
    # 返回 float 方便打印/记录
    return float(total_norm) if isinstance(total_norm, torch.Tensor) else total_norm


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
      1. 自动检测设备（从模型参数推断）
      2. 加载模型权重（必需）
      3. 加载优化器状态（可选，用于恢复训练）
      4. 加载混合精度训练的 scaler 状态（可选）
      5. 恢复训练进度信息（epoch 和最佳验证指标）
    
    参数:
      model: PyTorch 模型实例
      ckpt_path: 检查点文件路径（.pth 或 .pt 文件）
      optimizer: 优化器实例（训练时需要，推理时可为 None）
      scaler: AMP GradScaler 实例（使用混合精度训练时需要）
      device: 目标设备，如果为 None，则自动从模型参数推断
    
    返回值:
      tuple[int, float]: (起始轮数, 最佳验证指标)
        - 起始轮数:
            - 加载成功: 返回检查点保存时的 epoch
            - 加载失败: 返回 0（从头开始训练）
        - 最佳验证指标:
            - 加载成功: 返回检查点中保存的最佳值
            - 加载失败: 返回 float("inf")（表示无最佳值）
    
    检查点文件结构:
    {
        "model": model.state_dict(),      # 必需：模型权重
        "optim": optimizer.state_dict(),  # 可选：优化器状态
        "scaler": scaler.state_dict(),    # 可选：AMP scaler 状态
        "epoch": int,                     # 可选：当前训练轮数
        "best_val": float                 # 可选：最佳验证指标
    }
    
    使用场景:
    1. 恢复训练
       ```python
       start_epoch, best_val = load_ckpt_if_any(
           model, "checkpoints/last.pth", optimizer, scaler
       )
       ```
    
    2. 指定设备加载
       ```python
       start_epoch, best_val = load_ckpt_if_any(
           model, "checkpoints/last.pth", optimizer, scaler,
           device=torch.device("cuda:0")
       )
       ```
    
    3. 仅加载模型（推理）
       ```python
       start_epoch, _ = load_ckpt_if_any(model, "checkpoints/best.pth")
       ```
    
    注意事项:
    1. 设备检测逻辑：
       - 如果指定了 device 参数，使用指定的设备
       - 否则从模型的第一个参数推断设备
       - 如果模型无参数，默认使用 CPU 并发出警告
    
    2. 检查点加载：
       - 使用 strict=True 确保模型结构完全匹配
       - 通过 map_location 将张量加载到正确的设备
       - 使用 logger.info 记录加载过程
    
    3. 可选组件：
       - optimizer 状态加载仅在继续训练时需要
       - scaler 状态仅在使用 AMP 时需要
       - 训练进度信息（epoch, best_val）用于恢复训练点
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

        # 获取最佳验证指标
        # 默认为 inf，表示还没有最佳结果
        best = ckpt.get("best_val", float("inf"))

        # 6. 打印恢复信息（方便调试和日志记录）
        logger.info("load", f"[weights] loaded {ckpt_path} "
                            f"(epoch={start_epoch}, best={best:.4f}, "
                            f"device={device})")

        # 返回训练起始信息
        return start_epoch, best

    # 如果检查点不存在，从头开始训练
    # start_epoch=0: 从第 0 轮开始
    # best=inf: 还没有最佳验证结果
    return 0, float("inf")


def save_ckpt(state, out_dir, name):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    torch.save(state, path)
    logger.info("save", f"[weights] saved {path}")

    return str(path)
