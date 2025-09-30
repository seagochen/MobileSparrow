import torch
from copy import deepcopy
from tqdm import tqdm


# =========================
# 1) EMA: 指数移动平均
# =========================
class EMA:
    """
    模型参数的指数移动平均（Exponential Moving Average）。

    作用
    ----
    - 维护一个“影子模型”（ema_model），其权重是在线模型参数的平滑平均。
    - 在验证/导出时使用 EMA 权重，通常更稳定，精度略优。

    用法
    ----
    >>> ema = EMA(model, decay=0.9999)
    >>> for each training step:
    ...     optimizer.step()
    ...     ema.update(model)  # 训练后更新 EMA
    >>> evaluate(ema.ema_model, ...)  # 使用 ema_model 验证/保存

    参数
    ----
    model : nn.Module
        需要被跟踪的在线模型（会 deepcopy 一份）。
    decay : float
        EMA 衰减系数，越接近 1 越“平滑”。此处还叠加了一个随更新步数变化的 warmup 因子。

    备注
    ----
    - 仅对 requires_grad 的参数做 EMA；
    - ema_model 置为 eval() 并冻结梯度（不参与反传）。
    """

    def __init__(self, model, decay=0.9999):
        self.ema_model = deepcopy(model).eval()  # 模型深拷贝（不共享参数）
        self.decay = decay
        self.updates = 0  # 已更新的步数统计

        # EMA 模型不需要梯度
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        """
        用在线模型的当前参数更新 EMA 模型。

        策略
        ----
        - 使用动态衰减：d = decay * (1 - 0.9 ** (updates / 2000))
          前期更“跟随”，后期更“平滑”。

        注意
        ----
        - 仅对 requires_grad 的参数执行 EMA；
        - 假设两边的 named_parameters 能一一对应（同名同结构）。
        """
        self.updates += 1
        d = self.decay * (1 - pow(0.9, self.updates / 2000))

        # 1) 参数做 EMA
        msd = dict(model.named_parameters())
        esd = dict(self.ema_model.named_parameters())
        for name, p in msd.items():
            if p.requires_grad:
                esd[name].mul_(d).add_(p.data, alpha=1 - d)

        # 2) **BN 等 buffers 必须同步**（最稳妥的做法是硬拷贝）
        mbuf = dict(model.named_buffers())
        ebuf = dict(self.ema_model.named_buffers())
        for name, b in mbuf.items():
            ebuf[name].copy_(b)
