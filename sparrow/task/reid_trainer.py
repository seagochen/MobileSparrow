# -*- coding: utf-8 -*-
from __future__ import annotations
"""
ReIDTrainer
-----------
一个与 dets_trainer / kpts_trainer 同风格的训练 Task：
- 继承 BaseTrainer（统一训练循环 / AMP / EMA / 断点保存等） 参见：:contentReference[oaicite:0]{index=0}
- 仅实现 4 个钩子：_move_batch_to_device / _calculate_loss / _calculate_metrics / _get_main_metric
- 适配 ReID：嵌入+分类 (Arc/Cos) + batch-hard Triplet；验证时给出 in-batch Recall@1
- 主指标默认为 Recall@1（可通过 main_metric=loss 切换为最小化 val_loss）

使用方式（YAML）：
  trainer:
    class: reid_trainer
    args:
      epochs: 60
      save_dir: outputs/reid_run
      img_h: 256
      img_w: 128
      id_head: cosface               # or arcface
      triplet_margin: 0.3
      w_id: 1.0
      w_tri: 1.0
      main_metric: r1                # 或 'loss'
      mode: max                      # r1 -> 'max'；如果 main_metric=loss，则用 'min'
      optimizer_name: AdamW
      learning_rate: 3e-4
      weight_decay: 1e-4
      scheduler_name: MultiStepLR
      milestones: [20, 40, 55]
      gamma: 0.2
说明：
- Trainer 的核心模式与 dets_trainer / kpts_trainer 一致（优化器/调度/EMA/AMP 等均由基类完成）
  参考：DetsTrainer / KptsTrainer 的写法与参数组织。:contentReference[oaicite:1]{index=1} :contentReference[oaicite:2]{index=2}
- 该文件仅聚焦“如何算损失/指标、如何组织 batch 到设备”
"""

from typing import Dict, Tuple, Literal, Any

import torch
import torch.nn as nn

from sparrow.task.base_trainer import BaseTrainer
from sparrow.loss.reid_losses import ReIDCriterion

from sparrow.utils.logger import logger


class ReIDTrainer(BaseTrainer):
    """
    轻量 ReID 训练器：
    - 前向得到 L2-norm 的 embedding（模型外部约定）
    - 损失：分类头(Arc/CosFace) + batch-hard Triplet（组合权重可调）
    - 验证指标：in-batch Recall@1 + 平均类间/类内距离等
    """

    def __init__(self,
                 model: nn.Module,
                 *,
                 epochs: int,
                 save_dir: str,
                 # 训练图片分辨率（用于日志与潜在 resize 参考；dataloader 已保证送进来的尺寸正确）
                 img_h: int = 256,
                 img_w: int = 128,
                 # —— 优化器 ——（由 BaseTrainer 构建）
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 optimizer_name: str = "AdamW",
                 learning_rate: float = 3e-4,
                 weight_decay: float = 1e-4,
                 # —— 调度器 ——（由 BaseTrainer 构建）
                 scheduler_name: str = "MultiStepLR",
                 milestones=None,
                 gamma: float = 0.1,
                 step_size: int = 30,
                 # 保存/调度依据的主指标方向（'max' 或 'min'），建议配合 main_metric 使用
                 mode: Literal["min", "max"] = "max",
                 factor: float = 0.5,
                 patience: int = 5,
                 min_lr: float = 1e-6,
                 T_0: int = 10,
                 T_mult: int = 1,
                 last_epoch: int = -1,
                 # 训练技巧
                 use_amp: bool = True,
                 use_ema: bool = True,
                 ema_decay: float = 0.9998,
                 clip_grad_norm: float = 0.0,
                 log_interval: int = 10,
                 # —— ReID 损失参数 ——（必须提供 num_classes）
                 num_classes: int = None,
                 emb_dim: int = 128,
                 id_head: Literal["cosface", "arcface"] = "cosface",
                 w_id: float = 1.0,
                 w_tri: float = 1.0,
                 label_smooth_eps: float = 0.1,
                 triplet_margin: float = 0.3,
                 # 主指标选择：'r1'（Recall@1）或 'loss'
                 main_metric: Literal["r1", "loss"] = "r1"):
        super().__init__(
            model,
            epochs=epochs,
            save_dir=save_dir,
            device=device,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            scheduler_name=scheduler_name,
            milestones=milestones,
            gamma=gamma,
            step_size=step_size,
            mode=mode,
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            T_0=T_0,
            T_mult=T_mult,
            last_epoch=last_epoch,
            use_amp=use_amp,
            use_ema=use_ema,
            ema_decay=ema_decay,
            clip_grad_norm=clip_grad_norm,
            log_interval=log_interval
        )

        if not isinstance(num_classes, int):
            logger.warning("ReIDTrainer", "需要 num_classes（身份类别数），请在 YAML 的 trainer.args 中提供。")
            num_classes = 1000

        # 组合损失（分类 + 三元组）
        self.crit = ReIDCriterion(
            num_classes=num_classes,
            emb_dim=emb_dim,
            id_head=id_head,
            w_id=w_id,
            w_tri=w_tri,
            smooth_eps=label_smooth_eps,
            triplet_margin=triplet_margin
        )

        # 将损失函数与模型搬到一致的设备上
        self.crit.to(self.device)
        if hasattr(self, "optimizer"):
            self.optimizer.add_param_group({"params": self.crit.parameters()})

        self.img_h = int(img_h)
        self.img_w = int(img_w)
        self._val_main = main_metric  # 决定 _get_main_metric 返回什么
        # 记录 head 配置（用于后续按数据真实类数重建 head）
        self._id_head_type = id_head
        self._emb_dim = emb_dim
        self._label_smooth_eps = label_smooth_eps
        self._triplet_margin = triplet_margin

    # ============ BaseTrainer 抽象方法实现（与 dets/kpts 同套路） ============
    def _move_batch_to_device(self, batch) -> Any:
        """
        dataloader 输出约定：
          - 训练/验证：返回 (images, labels[, paths]) 或 (images, labels)
        """
        if len(batch) == 3:
            imgs, labels, _ = batch
        else:
            imgs, labels = batch
        return imgs.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)

    def ensure_num_classes(self, nc: int) -> None:
        """
        若当前 head 的类数与数据不一致，则重建 criterion 并把参数加入优化器。
        可避免 targets 超出 num_classes 导致 scatter_ 越界。
         """

        cur_w = getattr(getattr(self.crit, "head", None), "weight", None)
        cur_c = int(cur_w.size(0)) if cur_w is not None else -1

        if cur_c == nc:
            return  # 已一致

        # 记录设备，并重建 criterion
        dev = next(self.crit.parameters()).device
        self.crit = ReIDCriterion(
            num_classes = nc,
            emb_dim = self._emb_dim,
            id_head = self._id_head_type,
            w_id = self.crit.w_id,
            w_tri = self.crit.w_tri,
            smooth_eps = self._label_smooth_eps,
            triplet_margin = self._triplet_margin
        ).to(dev)

        # 把新 head 参数纳入优化器（否则不会更新）
        if hasattr(self, "optimizer"):
            self.optimizer.add_param_group({"params": self.crit.parameters()})


    def _calculate_loss(self, batch) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算训练阶段损失（分类 + Triplet）。
        - 模型应返回 L2-normalized 的 embedding（BxD）
        - ReIDCriterion 接受 (embeddings, labels) 返回 (total_loss, dict)
        """
        imgs, labels = batch
        emb = self.model(imgs)  # BxD，建议模型内部已 L2-norm
        loss, loss_dict = self.crit(emb, labels)

        # BaseTrainer 需要：(总损失Tensor, 可记录的 dict[float])
        return loss, {"loss": float(loss.detach().cpu()), **{k: float(v) for k, v in loss_dict.items()}}


    @torch.no_grad()
    def _calculate_metrics(self, model, batch) -> Dict[str, float]:
        """
        验证指标（轻量 in-batch）：
        - val_loss：与训练同配方计算一次
        - r1：in-batch Recall@1（排除自身；若同 ID 多样本，视为命中）
        - d_pos/d_neg：batch 内最近正样本/负样本的均值（可辅助观察训练动态）
        """
        imgs, labels = batch
        emb = model(imgs)  # BxD（已 L2-norm）

        # 验证损失
        loss, loss_dict = self.crit(emb, labels)

        # in-batch Recall@1 + 正负距离
        r1, d_pos, d_neg = self._recall_at_1_in_batch(emb, labels)

        out = {"val_loss": float(loss.detach().cpu()),
               "r1": float(r1),
               "d_pos": float(d_pos),
               "d_neg": float(d_neg)}
        # （可选）合并分类/三元组子项到日志
        for k, v in loss_dict.items():
            out[f"val_{k}"] = float(v)
        return out

    def _get_main_metric(self, metrics: Dict[str, float]) -> float:
        """
        决定保存 best.pt / 调度 LR 的主指标：
          - main_metric='r1' -> 返回 metrics['r1']（建议 BaseTrainer.mode='max'）
          - main_metric='loss' -> 返回 metrics['val_loss']（建议 BaseTrainer.mode='min'）
        """
        if self._val_main == "loss":
            return float(metrics.get("val_loss", 1e9))
        return float(metrics.get("r1", 0.0))

    # ============ 评价辅助 ============
    @staticmethod
    def _pairwise_dist(emb: torch.Tensor) -> torch.Tensor:
        """
        计算两两欧氏距离（emb 已 L2-norm，则等价于 sqrt(2-2cos)）
        返回 [B,B]
        """
        # 利用 (x - y)^2 = x^2 + y^2 - 2x·y；L2-norm 向量 x^2=y^2=1
        sim = emb @ emb.t()               # 余弦相似度
        sim = sim.clamp(-1, 1)
        dist = torch.sqrt(torch.clamp(2 - 2 * sim, min=0))
        return dist

    @torch.no_grad()
    def _recall_at_1_in_batch(self, emb: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float, float]:
        """
        in-batch Recall@1：
          - 对每个样本 i，在 batch\{i} 中找到最近邻 j*
          - 若 labels[i] == labels[j*] 计为命中
        另外统计：
          - d_pos：对每个 i 的“最近同类距离”均值
          - d_neg：对每个 i 的“最近异类距离”均值
        """
        B = emb.size(0)
        dist = self._pairwise_dist(emb)              # [B,B]
        mask_self = torch.eye(B, device=emb.device).bool()
        dist = dist.masked_fill(mask_self, float("inf"))

        label_eq = labels.unsqueeze(1) == labels.unsqueeze(0)  # [B,B]
        pos_mask = label_eq & (~mask_self)
        neg_mask = ~label_eq

        # 最近正/负样本距离（不存在时用 +inf 跳过）
        pos_dist = dist.masked_fill(~pos_mask, float("inf")).min(dim=1).values
        neg_dist = dist.masked_fill(~neg_mask, float("inf")).min(dim=1).values

        # Recall@1：最近邻是否为同类
        nn_idx = dist.argmin(dim=1)
        hit = (labels[nn_idx] == labels).float().mean().item()

        # 去掉 +inf（例如某 ID 仅 1 张时无正样本）
        d_pos = pos_dist[torch.isfinite(pos_dist)].mean().item() if torch.isfinite(pos_dist).any() else float("nan")
        d_neg = neg_dist[torch.isfinite(neg_dist)].mean().item() if torch.isfinite(neg_dist).any() else float("nan")
        return hit, d_pos, d_neg
