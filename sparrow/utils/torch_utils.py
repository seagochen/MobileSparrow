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
        d = self.decay * (1 - pow(0.9, self.updates / 2000))  # 动态调整衰减率

        with torch.no_grad():
            model_params = dict(model.named_parameters())
            ema_params = dict(self.ema_model.named_parameters())

            for name, p in model_params.items():
                if p.requires_grad:
                    ema_p = ema_params[name]
                    # ema_p = d * ema_p + (1 - d) * p
                    ema_p.data.mul_(d).add_(p.data, alpha=1 - d)


# =========================
# 2) 验证循环
# =========================
@torch.no_grad()
def evaluate(model, dataloader, criterion, anchor_generator, precomputed_anchors, device):
    """
    在验证集上评估模型，返回 (平均总损失, 平均分类损失, 平均回归损失)。

    输入
    ----
    model : nn.Module
        前向返回 (cls_preds, reg_preds) 的检测模型：
        - cls_preds: [B, A, C]  (未 sigmoid)
        - reg_preds: [B, A, 4]
    dataloader : torch.utils.data.DataLoader
        迭代返回 (imgs, targets, paths)：
        - imgs    : [B, 3, H, W]，已归一化到与训练一致
        - targets : List[Tensor]，长度 B；每项 [N_i, 5]=[cls, x1, y1, x2, y2]
    criterion : nn.Module
        损失计算器（如你上面的 SSDLoss），调用方式：
        loss_cls, loss_reg = criterion(anchors, cls_preds, reg_preds, targets)
    anchor_generator : AnchorGenerator
        仅为接口统一，实际这里不直接用（由 criterion 使用）。
    precomputed_anchors : Tensor
        预先生成好的所有 anchors，形状 [A, 4]（与模型输出对齐）。
    device : str | torch.device
        推理设备。

    备注
    ----
    - 内部会将 model 置为 eval 模式，并在函数结束后恢复 train 模式；
    - 计算的是简单的 batch 平均再对 dataloader 取平均（没有按样本数加权）。
    """
    model.eval()

    total_loss_cls = 0.0
    total_loss_reg = 0.0

    pbar = tqdm(dataloader, desc="  [Validating]🟡 ")
    for imgs, targets, _ in pbar:
        imgs = imgs.to(device)
        targets_on_device = [t.to(device) for t in targets]

        # 前向：要求模型返回 (cls_preds, reg_preds)
        cls_preds, reg_preds = model(imgs)

        # 计算损失（anchors 直接传入 criterion）
        loss_cls, loss_reg = criterion(precomputed_anchors, cls_preds, reg_preds, targets_on_device)

        total_loss_cls += loss_cls.item()
        total_loss_reg += loss_reg.item()

        pbar.set_postfix(cls=f"{loss_cls.item():.6f}", reg=f"{loss_reg.item():.6f}")

    avg_cls_loss = total_loss_cls / len(dataloader)
    avg_reg_loss = total_loss_reg / len(dataloader)
    avg_total_loss = avg_cls_loss + avg_reg_loss

    model.train()  # 恢复训练模式
    return avg_total_loss, avg_cls_loss, avg_reg_loss
