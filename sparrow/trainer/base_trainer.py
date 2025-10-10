
from typing import Optional, Dict

import torch
from torch import GradScaler
from torch import nn

from sparrow.losses.automatic_weighted_loss import AutomaticWeightedLoss
from sparrow.trainer.components import select_optimizer, build_scheduler, create_warmup_scheduler
from sparrow.trainer.ema import EMA


class BaseTrainer:

    def __init__(self,
                 model: nn.Module,
                 loss_fn: nn.Module,
                 *, data_dir: str,
                 resume: bool = False,
                 save_dir: Optional[str] = None,
                 device: Optional[torch.device] = None,

                 # 训练参数 - optimizer
                 optimizer_name: str,
                 lr: float,
                 weight_decay: float,

                 # 动态权重调节器
                 use_awl: bool=False,
                 num_tasks: Optional[int] = None,

                 # 训练参数 - scheduler
                 scheduler_name: str,
                 epochs: int = 100,
                 use_warmup_scheduler: bool = False,
                 warmup_epochs: int = 10,
                 start_factor: float = 0.01,
                 end_factor: float = 1.0,

                 # 训练技巧参数
                 use_amp: bool = True,
                 use_ema: bool = True,
                 ema_decay: float = 0.9998,
                 use_clip_grad: bool = False,
                 clip_grad_norm: float = 0.0,

                 # 其他补充参数
                 **kwargs):

        # --- 核心组件 ---
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.device = device
        self.epochs = epochs
        self.save_dir = save_dir
        self.data_dir = data_dir
        self.resume = resume
        self.cfg = kwargs

        # --- 训练技巧 ---
        self.scaler = GradScaler(enabled=use_amp)
        self.ema = EMA(self.model, decay=ema_decay) if use_ema else None
        self.clip_grad_norm = clip_grad_norm
        self.use_clip_grad = use_clip_grad
        self.use_ema = use_ema
        self.use_amp = use_amp
        self.use_awl = use_awl

        # --- 优化器和调度器 ---
        if self.use_awl:
            self.num_tasks = num_tasks
            self.awl = AutomaticWeightedLoss(num_tasks=self.num_tasks).to(device)
            self.optimizer = select_optimizer(
                name=optimizer_name,
                # 将模型参数和 awl 参数合并
                model=[
                    {'params': self.model.parameters()},
                    {'params': self.awl.parameters(), 'weight_decay': 0.0}  # 通常不对权重参数应用 weight_decay
                ],
                lr=lr,
                weight_decay=weight_decay,
                **kwargs
            )
        else:
            self.optimizer = select_optimizer(name=optimizer_name,
                                              model=self.model,
                                              lr=lr,
                                              weight_decay=weight_decay,
                                              **kwargs)

        self.scheduler = build_scheduler(name=scheduler_name,
                                         optimizer=self.optimizer,
                                         epochs=epochs,
                                         **kwargs)
        if use_warmup_scheduler:
            self.scheduler = create_warmup_scheduler(optimizer=self.optimizer,
                                                     warmup_epochs=warmup_epochs,
                                                     main_scheduler=self.scheduler,
                                                     start_factor=start_factor,
                                                     end_factor=end_factor)
        # --- Others ---
        self.BAR_FMT = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"


    # --- 需要子类实现的抽象方法 ---
    def create_dataloaders(self, seed=45):
        raise NotImplemented

    def train_one_epoch(self,
                        model: nn.Module,
                        loss_fn: nn.Module,
                        epoch: int,
                        loader: torch.utils.data.DataLoader,
                        optimizer: torch.optim.Optimizer,
                        scaler: torch.amp.GradScaler,
                        device: torch.device):
        raise NotImplemented

    def evaluate(self,
                 model: nn.Module,
                 loss_fn: nn.Module,
                 epoch: int,
                 loader: torch.utils.data.DataLoader,
                 device: torch.device):
        raise NotImplemented

    def train_model(self):
        raise NotImplemented

    def export_onnx(self):
        raise NotImplemented

    def run_evaluation(self) -> Dict[str, float]:
        """
        运行面向最终任务指标的评估 (例如 AP, mIoU 等)。
        子类应该重写此方法以实现特定任务的评估逻辑。
        """
        # 默认行为是打印一条消息并返回空字典
        print(f"'{self.__class__.__name__}' has not implemented the task-specific 'run_evaluation' method.")
        return {}

    # --- 辅助工具函数 ---