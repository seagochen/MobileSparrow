import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Dict, List, Union


def plot_training_curve(
        train_vals: Union[List[float], np.ndarray],
        val_vals: Optional[Union[List[float], np.ndarray]] = None,
        title: str = "Training Curve",
        train_label: str = "Train",
        val_label: str = "Val",
        xlabel: str = "Epoch",
        ylabel: str = "Loss",
        figsize: tuple = (8, 5),
        grid: bool = True,
        save_path: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
        show: bool = True
) -> plt.Axes:
    """
    通用训练曲线绘制函数（单图）

    功能：
      绘制训练/验证曲线，支持独立显示或作为子图嵌入

    参数:
      train_vals: 训练集指标值列表/数组（每个 epoch 的值）
      val_vals: 验证集指标值列表/数组（可选）
      title: 图表标题
      train_label: 训练曲线的图例标签
      val_label: 验证曲线的图例标签
      xlabel: X 轴标签（默认 "Epoch"）
      ylabel: Y 轴标签（默认 "Loss"）
      figsize: 图表尺寸（仅在独立绘制时有效）
      grid: 是否显示网格
      save_path: 保存路径（如 "curves/loss.png"），None 则不保存
      ax: Matplotlib Axes 对象，如果提供则在该轴上绘制（用于子图）
      show: 是否显示图表（plt.show()），独立绘制时为 True

    返回:
      ax: 绘制的 Axes 对象（方便后续调整）

    示例:
      # 独立绘制
      plot_training_curve(train_loss, val_loss, title="Loss Curve")

      # 作为子图
      fig, axes = plt.subplots(1, 2, figsize=(12, 5))
      plot_training_curve(train_loss, val_loss, ax=axes[0], show=False)
      plot_training_curve(train_acc, val_acc, ax=axes[1], show=False)
      plt.show()
    """
    # 1. 创建或使用现有的 Axes
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        is_standalone = True  # 标记是否为独立图表
    else:
        is_standalone = False

    # 2. 准备 X 轴数据（epoch 编号）
    epochs = np.arange(1, len(train_vals) + 1)

    # 3. 绘制训练曲线
    ax.plot(epochs, train_vals, label=train_label, marker='o', markersize=3)

    # 4. 绘制验证曲线（如果提供）
    if val_vals is not None:
        # 验证数据可能比训练数据短（如每 N 个 epoch 验证一次）
        val_epochs = np.arange(1, len(val_vals) + 1)
        ax.plot(val_epochs, val_vals, label=val_label, marker='s', markersize=3)

    # 5. 设置图表属性
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.legend(loc='best')
    if grid:
        ax.grid(True, alpha=0.3)

    # 6. 保存图表（如果指定路径）
    if save_path and is_standalone:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[save] Figure saved to {save_path}")

    # 7. 显示图表（仅在独立模式且 show=True 时）
    if is_standalone and show:
        plt.show()

    return ax


def plot_training_curves(
        curves_config: List[Dict],
        layout: tuple = None,
        figsize: tuple = (15, 5),
        suptitle: Optional[str] = None,
        save_path: Optional[str] = None,
        tight_layout: bool = True
):
    """
    批量绘制多个训练曲线（多子图）

    功能：
      在一个 figure 中绘制多个训练曲线，自动布局

    参数:
      curves_config: 曲线配置列表，每个元素是一个字典：
        {
            'train_vals': [...],           # 必需：训练数据
            'val_vals': [...],             # 可选：验证数据
            'title': 'Loss',               # 可选：子图标题
            'train_label': 'Train Loss',   # 可选：训练曲线标签
            'val_label': 'Val Loss',       # 可选：验证曲线标签
            'ylabel': 'Loss'               # 可选：Y 轴标签
        }
      layout: 子图布局 (rows, cols)，如 (1, 3) 表示 1 行 3 列
              如果为 None，自动推断（优先横向排列）
      figsize: 整个 figure 的尺寸
      suptitle: 总标题（显示在所有子图上方）
      save_path: 保存路径（如 "training_curves.png"）
      tight_layout: 是否使用紧凑布局

    示例:
      # 配置多个曲线
      config = [
          {
              'train_vals': train_loss,
              'val_vals': val_loss,
              'title': 'Total Loss',
              'ylabel': 'Loss'
          },
          {
              'train_vals': train_geo,
              'val_vals': val_geo,
              'title': 'Geodesic Loss',
              'ylabel': 'Geodesic (rad)'
          },
          {
              'train_vals': val_deg_mean,
              'val_vals': val_deg_median,
              'title': 'Validation Metrics',
              'train_label': 'Mean Error',
              'val_label': 'Median Error',
              'ylabel': 'Degrees'
          }
      ]

      plot_training_curves(config, layout=(1, 3), suptitle="Training Progress")
    """
    # 1. 确定布局
    n_curves = len(curves_config)
    if layout is None:
        # 自动布局：优先横向排列
        if n_curves <= 3:
            layout = (1, n_curves)
        elif n_curves <= 6:
            layout = (2, (n_curves + 1) // 2)
        else:
            layout = (3, (n_curves + 2) // 3)

    rows, cols = layout

    # 2. 创建子图
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    # 统一 axes 为数组（即使只有一个子图）
    if n_curves == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # 3. 遍历配置，绘制每个子图
    for idx, config in enumerate(curves_config):
        if idx >= len(axes):
            print(f"[warning] Too many curves ({n_curves}) for layout {layout}")
            break

        # 提取配置（带默认值）
        train_vals = config['train_vals']
        val_vals = config.get('val_vals', None)
        title = config.get('title', f'Curve {idx + 1}')
        train_label = config.get('train_label', 'Train')
        val_label = config.get('val_label', 'Val')
        ylabel = config.get('ylabel', 'Value')

        # 调用单图绘制函数
        plot_training_curve(
            train_vals=train_vals,
            val_vals=val_vals,
            title=title,
            train_label=train_label,
            val_label=val_label,
            ylabel=ylabel,
            ax=axes[idx],
            show=False
        )

    # 4. 隐藏多余的子图
    for idx in range(n_curves, len(axes)):
        axes[idx].set_visible(False)

    # 5. 设置总标题
    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight='bold')

    # 6. 调整布局
    if tight_layout:
        plt.tight_layout()

    # 7. 保存图表
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[save] Figure saved to {save_path}")

    # 8. 显示图表
    plt.show()


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 模拟训练数据
    np.random.seed(42)
    epochs = 50

    train_loss = 2.0 * np.exp(-np.arange(epochs) * 0.05) + np.random.normal(0, 0.05, epochs)
    val_loss = 2.0 * np.exp(-np.arange(epochs) * 0.04) + np.random.normal(0, 0.08, epochs)

    train_geo = 1.5 * np.exp(-np.arange(epochs) * 0.06) + np.random.normal(0, 0.03, epochs)
    val_geo = 1.5 * np.exp(-np.arange(epochs) * 0.05) + np.random.normal(0, 0.06, epochs)

    val_deg_mean = 15 * np.exp(-np.arange(epochs) * 0.04) + np.random.normal(0, 1, epochs)
    val_deg_median = 12 * np.exp(-np.arange(epochs) * 0.045) + np.random.normal(0, 0.8, epochs)

    # ========== 场景 1：单个曲线（独立绘制）==========
    print("=== 场景 1：独立绘制单个曲线 ===")
    plot_training_curve(
        train_vals=train_loss,
        val_vals=val_loss,
        title="Training Loss",
        ylabel="Loss",
        save_path="loss_curve.png"
    )

    # ========== 场景 2：多个曲线（自动布局）==========
    print("\n=== 场景 2：多子图自动布局 ===")
    config = [
        {
            'train_vals': train_loss,
            'val_vals': val_loss,
            'title': 'Total Loss',
            'ylabel': 'Loss'
        },
        {
            'train_vals': train_geo,
            'val_vals': val_geo,
            'title': 'Geodesic Loss',
            'ylabel': 'Geodesic (rad)'
        },
        {
            'train_vals': val_deg_mean,
            'val_vals': val_deg_median,
            'title': 'Validation Error',
            'train_label': 'Mean Error',
            'val_label': 'Median Error',
            'ylabel': 'Degrees'
        }
    ]

    plot_training_curves(
        curves_config=config,
        suptitle="Head Pose Training Progress",
        save_path="training_curves.png"
    )

    # ========== 场景 3：自定义布局（2x2）==========
    print("\n=== 场景 3：自定义布局 ===")
    config_2x2 = [
        {'train_vals': train_loss, 'val_vals': val_loss, 'title': 'Loss'},
        {'train_vals': train_geo, 'val_vals': val_geo, 'title': 'Geodesic'},
        {'train_vals': val_deg_mean, 'title': 'Mean Error (deg)', 'val_vals': None},
        {'train_vals': val_deg_median, 'title': 'Median Error (deg)', 'val_vals': None}
    ]

    plot_training_curves(
        curves_config=config_2x2,
        layout=(2, 2),
        figsize=(12, 8),
        suptitle="Detailed Training Metrics"
    )

    # ========== 场景 4：手动组合子图（高级用法）==========
    print("\n=== 场景 4：手动控制子图 ===")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 第一个子图
    plot_training_curve(
        train_vals=train_loss,
        val_vals=val_loss,
        title="Loss",
        ylabel="Loss",
        ax=axes[0],
        show=False
    )

    # 第二个子图
    plot_training_curve(
        train_vals=train_geo,
        val_vals=val_geo,
        title="Geodesic Loss",
        ylabel="Geodesic (rad)",
        ax=axes[1],
        show=False
    )

    # 第三个子图（自定义样式）
    axes[2].plot(val_deg_mean, label="Mean", color='red', linewidth=2)
    axes[2].plot(val_deg_median, label="Median", color='blue', linestyle='--')
    axes[2].set_title("Val Error (deg)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()