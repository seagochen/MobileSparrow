from copy import deepcopy

import torch
import torch.nn as nn


class EMA:
    """
    指数移动平均（Exponential Moving Average）- 模型权重平滑工具

    核心思想：
      维护一个"影子模型"，其参数是训练模型参数的指数加权移动平均
      EMA 模型通常比训练模型更稳定，泛化性能更好

    数学原理：
      θ_ema = decay * θ_ema + (1 - decay) * θ_train
      其中：
        - θ_ema: EMA 模型参数
        - θ_train: 训练模型参数
        - decay: 衰减系数（越接近 1，平滑程度越高）

    动态衰减策略：
      实际使用的衰减系数会随训练步数动态调整：
        d_effective = decay * (1 - 0.9^(steps/2000))

      效果：
        - 训练初期（steps 小）：d_effective 较小，更快跟随训练模型
        - 训练后期（steps 大）：d_effective 接近 decay，更平滑

      例如（decay=0.9999）：
        - step 100:  d ≈ 0.9950 (快速跟随)
        - step 2000: d ≈ 0.9989 (开始平滑)
        - step 10000: d ≈ 0.9999 (高度平滑)

    使用场景：
      1. 验证/测试：使用 EMA 模型评估，通常比训练模型效果更好
      2. 模型导出：保存 EMA 权重用于部署
      3. 训练稳定性：在不稳定的训练过程中提供更鲁棒的权重

    典型用法：
      >>> # 初始化
      >>> model = MyModel()
      >>> ema = EMA(model, decay=0.9999)
      >>> 
      >>> # 训练循环
      >>> for epoch in range(epochs):
      ...     for batch in dataloader:
      ...         loss = criterion(model(batch), target)
      ...         optimizer.zero_grad()
      ...         loss.backward()
      ...         optimizer.step()
      ...         
      ...         # 每次更新模型后，同步更新 EMA
      ...         ema.update(model)
      >>> 
      >>> # 验证时使用 EMA 模型
      >>> val_loss = validate(ema.ema_model, val_loader)
      >>> 
      >>> # 保存时保存 EMA 权重
      >>> torch.save({
      ...     'model': model.state_dict(),
      ...     'ema_model': ema.ema_model.state_dict()
      ... }, 'checkpoint.pth')

    注意事项：
      1. EMA 模型不参与训练（无梯度，处于 eval 模式）
      2. 只对 requires_grad=True 的参数应用 EMA
      3. BatchNorm 等的 running stats（buffers）会直接硬拷贝
      4. EMA 更新频率应与优化器步数一致（每次 optimizer.step() 后调用）

    参数：
      model: 需要跟踪的训练模型（会深拷贝一份作为 EMA 模型）
      decay: EMA 衰减系数，范围 [0, 1]
             - 0.999: 较快响应，适合短期训练
             - 0.9999: 标准选择，适合大多数场景
             - 0.99999: 极度平滑，适合长期训练
      updates: 可选的初始更新步数（用于恢复训练）

    属性：
      ema_model: EMA 模型实例（nn.Module）
      decay: 基础衰减系数
      updates: 当前更新步数
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999, updates: int = 0):
        """
        初始化 EMA 跟踪器

        参数:
          model: 训练模型（会被深拷贝）
          decay: 衰减系数，推荐范围 [0.999, 0.99999]
          updates: 初始更新步数（恢复训练时使用）
        """
        # 1. 深拷贝模型（确保参数不共享）
        self.ema_model = deepcopy(model).eval()

        # 2. 保存配置
        self.decay = float(decay)
        self.updates = int(updates)

        # 3. 冻结 EMA 模型的梯度（不参与训练）
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    def get_decay(self) -> float:
        """
        计算当前步数的有效衰减系数

        返回:
          当前有效的衰减系数

        说明:
          使用 warmup 策略，训练初期衰减较弱（快速跟随），
          训练后期衰减较强（高度平滑）
        """
        # 动态衰减：前期快速跟随，后期高度平滑
        # warmup_factor = 1 - 0.9^(updates/2000)
        # 当 updates = 0 时，warmup_factor ≈ 0，实际衰减很小
        # 当 updates = 10000 时，warmup_factor ≈ 0.9933，实际衰减接近 decay
        warmup_factor = 1.0 - pow(0.9, self.updates / 2000.0)
        return self.decay * warmup_factor

    @torch.no_grad()
    def update(self, model: nn.Module):
        """
        使用训练模型的当前参数更新 EMA 模型

        参数:
          model: 训练模型（需要与初始化时的模型结构一致）

        更新公式：
          θ_ema = d * θ_ema + (1 - d) * θ_train
          其中 d 为动态衰减系数

        注意：
          1. 只更新 requires_grad=True 的参数
          2. BatchNorm 的 running_mean/var 等 buffers 会直接拷贝
          3. 假设两个模型的参数名完全对应

        调用时机：
          每次 optimizer.step() 后立即调用
        """
        # 1. 增加更新计数
        self.updates += 1

        # 2. 计算当前有效衰减系数
        d = self.get_decay()

        # 3. 更新可训练参数（EMA）
        model_params = dict(model.named_parameters())
        ema_params = dict(self.ema_model.named_parameters())

        for name, param in model_params.items():
            # 只对需要梯度的参数做 EMA（如冻结层不更新）
            if param.requires_grad:
                # θ_ema = d * θ_ema + (1-d) * θ_train
                ema_params[name].mul_(d).add_(param.data, alpha=1.0 - d)

        # 4. 同步 buffers（BatchNorm 的 running stats 等）
        # 重要：BN 的统计量不做 EMA，而是直接拷贝训练模型的值
        # 原因：BN 的 running stats 本身已经是移动平均了
        model_buffers = dict(model.named_buffers())
        ema_buffers = dict(self.ema_model.named_buffers())

        for name, buffer in model_buffers.items():
            ema_buffers[name].copy_(buffer)

    def update_attr(self, model: nn.Module, include: tuple = (), exclude: tuple = ('process_group', 'reducer')):
        """
        同步模型的属性（可选功能）

        参数:
          model: 训练模型
          include: 需要拷贝的属性名元组（白名单）
          exclude: 不拷贝的属性名元组（黑名单）

        说明:
          用于同步一些非参数属性，如自定义的配置、统计信息等
          默认排除 DDP 相关属性（process_group, reducer）
        """
        for name in dir(model):
            # 跳过私有属性和方法
            if name.startswith('_'):
                continue

            # 跳过黑名单
            if name in exclude:
                continue

            # 白名单模式：如果指定了 include，只拷贝白名单内的
            if include and name not in include:
                continue

            # 跳过方法和特殊属性
            attr = getattr(model, name)
            if callable(attr) or isinstance(attr, (nn.Module, nn.Parameter)):
                continue

            # 拷贝属性
            try:
                setattr(self.ema_model, name, attr)
            except Exception:
                pass  # 某些只读属性无法设置，忽略


# ==================== 使用示例 ====================
if __name__ == "__main__":
    import torch.nn as nn
    import torch.optim as optim
    # 创建示例模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 10)
            self.bn = nn.BatchNorm1d(10)

        def forward(self, x):
            return self.bn(self.fc(x))


    model = SimpleModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 初始化 EMA
    ema = EMA(model, decay=0.9999)

    # 跟踪最佳模型
    best_loss = float('inf')
    best_step = 0

    # 模拟训练
    print("=== 训练示例 ===")
    for step in range(100):
        # 前向传播
        x = torch.randn(32, 10)
        output = model(x)
        loss = output.sum()

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新 EMA
        ema.update(model)

        # 模拟验证并更新最佳检查点
        if (step + 1) % 20 == 0:
            # 使用 EMA 模型进行验证
            ema.ema_model.eval()
            with torch.no_grad():
                val_x = torch.randn(10, 10)
                val_output = ema.ema_model(val_x)
                val_loss = val_output.sum().item()

            print(f"Step {step + 1}: effective_decay = {ema.get_decay():.6f}, val_loss = {val_loss:.4f}")

            # 更新最佳检查点
            if val_loss < best_loss:
                best_loss = val_loss
                best_step = step + 1
                # 保存最佳检查点
                best_checkpoint = {
                    'step': step + 1,
                    'model': model.state_dict(),
                    'ema_model': ema.ema_model.state_dict(),
                    'ema_updates': ema.updates,
                    'optimizer': optimizer.state_dict(),
                    'best_loss': best_loss
                }
                torch.save(best_checkpoint, 'best_checkpoint.pth')
                print(f"✓ New best model saved at step {step + 1} with loss {best_loss:.4f}")

    # 验证时使用 EMA 模型
    print("\n=== 验证示例 ===")
    model.eval()
    ema.ema_model.eval()

    with torch.no_grad():
        x_test = torch.randn(10, 10)
        out_train = model(x_test)
        out_ema = ema.ema_model(x_test)
        print(f"训练模型输出: {out_train.mean().item():.4f}")
        print(f"EMA 模型输出: {out_ema.mean().item():.4f}")

    # 保存最终检查点
    print("\n=== 保存最终检查点示例 ===")
    final_checkpoint = {
        'step': step + 1,
        'model': model.state_dict(),
        'ema_model': ema.ema_model.state_dict(),
        'ema_updates': ema.updates,
        'optimizer': optimizer.state_dict(),
        'best_loss': best_loss,
        'best_step': best_step
    }
    torch.save(final_checkpoint, 'last_checkpoint.pth')
    print("✓ Final checkpoint saved with EMA weights")
    print(f"Best model was at step {best_step} with loss {best_loss:.4f}")

    # 恢复训练
    print("\n=== 恢复训练示例 ===")
    ckpt = torch.load('best_checkpoint.pth')  # 加载最佳检查点
    model.load_state_dict(ckpt['model'])
    ema_restored = EMA(model, decay=0.9999, updates=ckpt['ema_updates'])
    ema_restored.ema_model.load_state_dict(ckpt['ema_model'])
    print(f"✓ EMA restored from step {ema_restored.updates} (best checkpoint)")
