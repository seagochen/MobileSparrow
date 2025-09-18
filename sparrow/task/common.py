from copy import deepcopy
from typing import Union, Literal

import torch
from torch import optim

from sparrow.utils.logger import logger


# =========================
# Schedulers / Optimizers
# =========================
def build_scheduler(
    name: str,
    optimizer,
    *,
    # 仅在对应调度器下会用到的明参：
    milestones=None,
    gamma: float = 0.1,
    step_size: int = 30,
    mode: Literal["min", "max"] = "max",
    factor: float = 0.5,
    patience: int = 5,
    min_lr: float = 1e-6,
    T_0: int = 10,
    T_mult: int = 1,
    last_epoch: int = -1,
):
    n = (name or "").lower()

    if n in ("multisteplr", "multi_step", "multi-step"):
        if milestones is None:
            raise ValueError("MultiStepLR 需要提供 milestones=list[int]")
        return optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=list(milestones), gamma=float(gamma), last_epoch=int(last_epoch)
        )

    if n in ("steplr", "step"):
        return optim.lr_scheduler.StepLR(
            optimizer, step_size=int(step_size), gamma=float(gamma), last_epoch=int(last_epoch)
        )

    if n in ("cosineannealingwarmrestarts", "sgdr"):
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=int(T_0), T_mult=int(T_mult)
        )

    if n in ("reducelronplateau", "plateau", "default"):
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=float(factor),
            patience=int(patience), min_lr=float(min_lr)
        )

    raise ValueError(f"Unknown scheduler name: {name!r}")


def select_optimizer(name: str, model: torch.nn.Module, lr: float, weight_decay: float = 0.0, **kw):
    """
    兼容两种调用方式：
      - select_optimizer("adamw", model, lr=..., ...)            # 传 model，本函数内部取 .parameters()
      - select_optimizer("adamw", model.parameters(), lr=..., ...)  # 传 params，可迭代
    """
    # 获取模型参数（兼容直接传可迭代参数的情况）
    params = model.parameters() if hasattr(model, "parameters") else model

    # 强制把可被 YAML 写成字符串的超参转为 float
    try:
        lr = float(lr)
        weight_decay = float(weight_decay)
    except ArithmeticError:
        logger.warning("select_optimizer", "Convert the number to float failed, use default values")
        # 转换失败，使用默认参数
        lr = 3e-4
        weight_decay = 1e-4

    name = (name or "").lower()
    if name in ("adam",):
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay,
                                betas=kw.get("betas", (0.9, 0.999)))
    if name in ("adamw", "adam_w"):
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay,
                                 betas=kw.get("betas", (0.9, 0.999)))
    if name in ("sgd",):
        return torch.optim.SGD(params, lr=lr,
                               momentum=kw.get("momentum", 0.9),
                               weight_decay=weight_decay,
                               nesterov=kw.get("nesterov", True))
    raise Exception("Unknown optimizer.")



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

# =========================
# EMA
# =========================
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
