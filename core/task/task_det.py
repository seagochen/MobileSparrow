# task_det.py
# -*- coding: utf-8 -*-
"""
SSDlite detection task loop (COCO-style)
- Works with: core/datasets/coco_det.py + core/models/ssdlite.py + core/loss/ssdlite_loss.py
- Mirrors KptsTask structure: optimizer/scheduler/logging/checkpoints
"""
import os
import time
from typing import Dict, List

import torch
import torch.nn as nn

from core.task import common
from core.loss.ssdlite_loss import SSDLoss


class DetTask:
    """
    Minimal training/validation runner for SSDLite detection.
    Expect dataloader to yield:
        train/val: imgs: FloatTensor [B,3,H,W] in [0,1]
                   targets: List[{'boxes': FloatTensor[n_i,4] in [0,1], 'labels': LongTensor[n_i]}]
    """

    def __init__(self, cfg: Dict, model: nn.Module, num_classes: int = None):
        self.cfg = cfg
        use_cuda = (self.cfg.get("GPU_ID", "") != "" and torch.cuda.is_available())
        self.device = torch.device("cuda" if use_cuda else "cpu")

        # model / device
        self.model = model.to(self.device)

        # num_classes: prefer model.num_classes; fallback to arg/cfg
        self.num_classes = getattr(self.model, "num_classes", None) or num_classes or int(
            self.cfg.get("num_classes", 81)
        )
        assert self.num_classes is not None and self.num_classes >= 2, \
            "num_classes must be 1+N (background + classes)"

        # loss / optim / sched
        self.criterion = SSDLoss(num_classes=self.num_classes, alpha=float(self.cfg.get("reg_alpha", 1.0)))
        self.optimizer = common.getOptimizer(
            self.cfg["optimizer"], self.model, self.cfg["learning_rate"], self.cfg["weight_decay"]
        )
        self.scheduler = common.getSchedu(self.cfg["scheduler"], self.optimizer)

        # bookkeeping
        self.save_dir = self.cfg["save_dir"]
        os.makedirs(self.save_dir, exist_ok=True)
        self.best_val_loss = float("inf")

        # misc
        self.log_interval = int(self.cfg.get("log_interval", 10))
        self.clip_grad = float(self.cfg.get("clip_gradient", 0.0) or 0.0)

    @staticmethod
    def _targets_to_device(targets: List[Dict[str, torch.Tensor]], device: torch.device) -> Dict[str, List[torch.Tensor]]:
        # Keep per-image lists, just move tensors to device
        tb = [t["boxes"].to(device, non_blocking=True) for t in targets]
        tl = [t["labels"].to(device, non_blocking=True) for t in targets]
        return {"boxes": tb, "labels": tl}

    def train(self, train_loader, val_loader):
        epochs = int(self.cfg["epochs"])
        for epoch in range(epochs):
            self.onTrainStep(train_loader, epoch, epochs)
            val_loss = self.onValidation(val_loader, epoch, epochs)

            # NOTE: scheduler in KptsTask is 'mode=max'. We feed negative loss for consistency.
            try:
                self.scheduler.step(-val_loss)
            except Exception:
                # Fallback for schedulers that don't take a metric
                self.scheduler.step()

            # save best
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_ckpt("best")

        self.onTrainEnd()

    def onTrainStep(self, train_loader, epoch: int, epochs: int):
        self.model.train()
        running_loss = 0.0
        running_cls = 0.0
        running_reg = 0.0
        running_pos = 0
        seen = 0

        t0 = time.time()
        for it, (imgs, targets) in enumerate(train_loader):
            imgs = imgs.to(self.device, non_blocking=True)
            tdev = self._targets_to_device(targets, self.device)

            # forward
            out = self.model(imgs)  # {"cls_logits","bbox_regs","anchors"} in train mode
            loss, meter = self.criterion(out["cls_logits"], out["bbox_regs"], out["anchors"], tdev)

            # backward
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.clip_grad > 0:
                common.clipGradient(self.optimizer, self.clip_grad)
            self.optimizer.step()

            # meters
            bsz = imgs.shape[0]
            running_loss += loss.item() * bsz
            running_cls  += meter["loss_cls"] * bsz
            running_reg  += meter["loss_reg"] * bsz
            running_pos  += meter["pos"]
            seen += bsz

            if it % self.log_interval == 0:
                avg_loss = running_loss / max(1, seen)
                avg_cls  = running_cls  / max(1, seen)
                avg_reg  = running_reg  / max(1, seen)
                speed = (time.time() - t0) / max(1, seen)
                print(
                    f"\r[Train] {epoch+1}/{epochs} it={it}/{len(train_loader)} "
                    f"loss={avg_loss:.4f} (cls={avg_cls:.3f} reg={avg_reg:.3f}) "
                    f"pos={running_pos}  {speed:.3f}s/img",
                    end="", flush=True
                )
        print()

    @torch.no_grad()
    def onValidation(self, val_loader, epoch: int, epochs: int) -> float:
        self.model.eval()
        running_loss = 0.0
        running_cls = 0.0
        running_reg = 0.0
        running_pos = 0
        seen = 0

        for it, (imgs, targets) in enumerate(val_loader):
            imgs = imgs.to(self.device, non_blocking=True)
            tdev = self._targets_to_device(targets, self.device)

            out = self.model(imgs)  # train/eval both return dict in training mode branch
            loss, meter = self.criterion(out["cls_logits"], out["bbox_regs"], out["anchors"], tdev)

            bsz = imgs.shape[0]
            running_loss += loss.item() * bsz
            running_cls  += meter["loss_cls"] * bsz
            running_reg  += meter["loss_reg"] * bsz
            running_pos  += meter["pos"]
            seen += bsz

        avg_loss = running_loss / max(1, seen)
        avg_cls  = running_cls  / max(1, seen)
        avg_reg  = running_reg  / max(1, seen)
        print(f"[Val]   {epoch+1}/{epochs} loss={avg_loss:.4f} (cls={avg_cls:.3f} reg={avg_reg:.3f}) pos={running_pos}")
        return avg_loss

    def onTrainEnd(self):
        self._save_ckpt("last")

    def _save_ckpt(self, tag: str):
        path = os.path.join(self.save_dir, f"ssdlite_{tag}.pth")
        ckpt = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "cfg": self.cfg,
            "best_val_loss": self.best_val_loss,
        }
        torch.save(ckpt, path)
        print(f"[INFO] checkpoint saved: {path}")
