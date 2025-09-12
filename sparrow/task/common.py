from copy import deepcopy
from typing import Union

import torch
from torch import optim


# =========================
# Schedulers / Optimizers
# =========================
def select_scheduler(scheduler: str, optimizer):
    if 'default' in scheduler:
        factor = float(scheduler.strip().split('-')[1])
        patience = int(scheduler.strip().split('-')[2])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=factor, patience=patience, min_lr=1e-6
        )
    elif 'step' in scheduler:
        step_size = int(scheduler.strip().split('-')[1])
        gamma = float(scheduler.strip().split('-')[2])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma, last_epoch=-1)
    elif 'SGDR' in scheduler:
        T_0 = int(scheduler.strip().split('-')[1])
        T_mult = int(scheduler.strip().split('-')[2])
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult)
    elif 'MultiStepLR' in scheduler:
        milestones = [int(x) for x in scheduler.strip().split('-')[1].split(',')]
        gamma = float(scheduler.strip().split('-')[2])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    else:
        raise Exception("Unknown scheduler.")
    return scheduler


def select_optimizer(optims: str, model, learning_rate: float, weight_decay: float):
    if optims == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optims == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    else:
        raise Exception("Unknown optimizer.")
    return optimizer


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
