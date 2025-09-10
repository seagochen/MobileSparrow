# -*- coding: utf-8 -*-
import os, gc, torch
from typing import Dict, Any
import torch.optim as optim

from core.models.ssdlite import SSDLiteDet
from core.loss.ssd_loss import SSDLoss
from core.datasets.coco_det import create_coco_det_loaders

def getOptimizer(name: str, model, lr: float, wd: float):
    if name == 'Adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif name == 'SGD':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    else:
        raise ValueError(name)

def getSchedu(schedu: str, optimizer):
    if 'MultiStepLR' in schedu:
        milestones = [int(x) for x in schedu.strip().split('-')[1].split(',')]
        gamma = float(schedu.strip().split('-')[2])
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    elif 'step' in schedu:
        step_size = int(schedu.strip().split('-')[1])
        gamma = float(schedu.strip().split('-')[2])
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6)

class DetTask:
    """
    与你现有 Task 同职责：封装训练/验证/保存，只是把 loss/数据 改为检测版本。
    """
    def __init__(self, cfg: Dict[str, Any], only_person: bool = False):
        self.cfg = cfg
        use_cuda = (self.cfg.get('GPU_ID', '') != '' and torch.cuda.is_available())
        self.device = torch.device("cuda" if use_cuda else "cpu")

        # dataloader
        self.train_loader, self.val_loader, num_classes = create_coco_det_loaders(cfg, only_person=only_person)

        # model（按 dataloader 的类别数配置）
        self.model = SSDLiteDet(backbone="mobilenet_v2",
                                num_classes=num_classes,
                                width_mult=float(cfg.get("width_mult", 1.0)),
                                neck_outc=64, head_midc=64).to(self.device)

        # loss / optim / sched
        self.criterion = SSDLoss(num_classes=num_classes, alpha=1.0)
        self.optimizer = getOptimizer(cfg['optimizer'], self.model, cfg['learning_rate'], cfg['weight_decay'])
        self.scheduler = getSchedu(cfg['scheduler'], self.optimizer)

        self.best_mAP_proxy = -1e9  # 这里先用 val 正样本数/均值 IoU 等 proxy，简化演示
        self.save_dir = cfg["save_dir"]; os.makedirs(self.save_dir, exist_ok=True)

    # ---------------- core loops ----------------
    def train(self):
        epochs = int(self.cfg['epochs'])
        log_interval = int(self.cfg.get('log_interval', 20))
        for ep in range(epochs):
            self.model.train()
            run_loss = 0.0
            for it, (imgs, targets) in enumerate(self.train_loader):
                imgs = imgs.to(self.device)
                # 把 list of dict 拆到 GPU
                t_boxes = [t["boxes"].to(self.device) for t in targets]
                t_labels= [t["labels"].to(self.device) for t in targets]
                batch_targets = {"boxes": t_boxes, "labels": t_labels}

                out = self.model(imgs)  # 训练阶段: {"cls_logits","bbox_regs","anchors"}
                loss, meter = self.criterion(out["cls_logits"], out["bbox_regs"], out["anchors"], batch_targets)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.cfg.get('clip_gradient', 0):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg['clip_gradient']))
                self.optimizer.step()

                run_loss += float(loss.item())
                if it % log_interval == 0:
                    print(f"\r[Train] ep {ep+1}/{epochs} it {it}/{len(self.train_loader)} "
                          f"loss {loss.item():.4f} (cls {meter['loss_cls']:.3f} reg {meter['loss_reg']:.3f}) "
                          f"pos {meter['pos']}", end='', flush=True)
            print()

            # ---- validation (proxy) ----
            val_loss = self.validate()
            # 调度
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(-val_loss)  # 这里用负的 val_loss 作为“越大越好”的指标
            else:
                self.scheduler.step()

            # save
            last = os.path.join(self.save_dir, "last.pt")
            torch.save(self.model.state_dict(), last)
            if -val_loss > self.best_mAP_proxy:
                self.best_mAP_proxy = -val_loss
                best = os.path.join(self.save_dir, "best.pt")
                torch.save(self.model.state_dict(), best)
                print(f"[INFO] new best (proxy) -> {best}")

        self._cleanup()

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        tot, n = 0.0, 0
        for imgs, targets in self.val_loader:
            imgs = imgs.to(self.device)
            t_boxes = [t["boxes"].to(self.device) for t in targets]
            t_labels= [t["labels"].to(self.device) for t in targets]
            batch_targets = {"boxes": t_boxes, "labels": t_labels}

            out = self.model(imgs)
            loss, meter = self.criterion(out["cls_logits"], out["bbox_regs"], out["anchors"], batch_targets)
            tot += float(loss.item()); n += 1
        mean_loss = tot / max(1, n)
        print(f"[Val ] loss {mean_loss:.4f}")
        return mean_loss

    def _cleanup(self):
        del self.model
        gc.collect(); torch.cuda.empty_cache()
