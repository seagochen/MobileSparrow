# -*- coding: utf-8 -*-
"""
Unified Task file (refined)
- merges old task.py + task_tools.py + metrics.py (myAcc)
- Compatible with dict outputs: {"heatmaps","centers","regs","offsets"}
- Dynamic HxW decoding (no hard-coded 48)
- Auto-align model heads to label feature size (default img_size//4)
- Stable path handling for img_name (nested -> str)
"""

import os
import gc
import cv2
import numpy as np
from typing import Tuple, Dict, List, Iterable, Union

import torch
import torch.nn as nn
import torch.optim as optim

# ====== external (keep your existing loss) ======
from core.loss.movenet_loss import MovenetLoss


# =========================
# >>> metrics (inlined) <<<
# =========================
def _getDist(pre: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    pre:    [B, 2*J]   normalized coords in [0,1] or -1 for invalid
    labels: [B, 2*J]
    return: [B, J] squared distance per joint
    """
    pre = pre.reshape([-1, 17, 2])
    labels = labels.reshape([-1, 17, 2])
    res = (pre[:, :, 0] - labels[:, :, 0]) ** 2 + (pre[:, :, 1] - labels[:, :, 1]) ** 2
    return res


def _getAccRight(dist: np.ndarray, th: float = 5, img_size: int = 192) -> np.ndarray:
    """
    dist: [B, J] squared distance in normalized coords
    th:   pixel threshold (default 5px) on original img_size
    return: [J,] int counts of correct per joint
    """
    thr = th / float(img_size)  # compare sqrt(dist) with normalized threshold
    res = np.sum(np.sqrt(dist) < thr, axis=0).astype(np.int64)
    return res


def myAcc(output: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return [J,] ndarray: per-joint correct counts in this batch."""
    dist = _getDist(output, target)
    cate_acc = _getAccRight(dist)
    return cate_acc


# =========================
# Schedulers / Optimizers
# =========================
def getSchedu(schedu: str, optimizer):
    if 'default' in schedu:
        factor = float(schedu.strip().split('-')[1])
        patience = int(schedu.strip().split('-')[2])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=factor, patience=patience, min_lr=1e-6
        )
    elif 'step' in schedu:
        step_size = int(schedu.strip().split('-')[1])
        gamma = float(schedu.strip().split('-')[2])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma, last_epoch=-1)
    elif 'SGDR' in schedu:
        T_0 = int(schedu.strip().split('-')[1])
        T_mult = int(schedu.strip().split('-')[2])
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult)
    elif 'MultiStepLR' in schedu:
        milestones = [int(x) for x in schedu.strip().split('-')[1].split(',')]
        gamma = float(schedu.strip().split('-')[2])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    else:
        raise Exception("Unknown scheduler.")
    return scheduler


def getOptimizer(optims: str, model, learning_rate: float, weight_decay: float):
    if optims == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optims == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    else:
        raise Exception("Unknown optimizer.")
    return optimizer


def clipGradient(optimizer, grad_clip=1.0):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


# =========================
# Output helpers (compat)
# =========================
def _to_numpy(t: torch.Tensor):
    return t.detach().cpu().numpy()


def _split_pred_tensors(output) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return numpy (hm, ct, rg, of) for viz paths, supports dict / list outputs."""
    if isinstance(output, dict):
        hm = _to_numpy(output["heatmaps"])
        ct = _to_numpy(output["centers"])
        rg = _to_numpy(output["regs"])
        of = _to_numpy(output["offsets"])
    else:
        hm = _to_numpy(output[0])
        ct = _to_numpy(output[1])
        rg = _to_numpy(output[2])
        of = _to_numpy(output[3])
    return hm, ct, rg, of


def _resize_heads_to(hm: torch.Tensor,
                     ct: torch.Tensor,
                     rg: torch.Tensor,
                     of: torch.Tensor,
                     target_hw: Tuple[int, int]) -> List[torch.Tensor]:
    """
    Bilinear resize all heads to (H_t, W_t). This keeps loss/decoding consistent with labels.
    """
    Ht, Wt = target_hw
    align_corners = False
    mode = 'bilinear'
    hm_r = nn.functional.interpolate(hm, size=(Ht, Wt), mode=mode, align_corners=align_corners)
    ct_r = nn.functional.interpolate(ct, size=(Ht, Wt), mode=mode, align_corners=align_corners)
    rg_r = nn.functional.interpolate(rg, size=(Ht, Wt), mode=mode, align_corners=align_corners)
    of_r = nn.functional.interpolate(of, size=(Ht, Wt), mode=mode, align_corners=align_corners)
    return [hm_r, ct_r, rg_r, of_r]


# =========================
# Decoding (dynamic size)
# =========================
def _maxPoint_numpy(hm: np.ndarray) -> tuple:
    """
    hm: [B,1,H,W] numpy
    return: (cx, cy) both int numpy arrays with shape [B,1]
    """
    B, C, H, W = hm.shape
    flat = hm.reshape(B, -1)
    ids = np.argmax(flat, axis=1)
    cy = (ids // W).astype(np.int32).reshape(B, 1)
    cx = (ids % W).astype(np.int32).reshape(B, 1)
    return cx, cy


def movenetDecode(data,
                  kps_mask=None,
                  mode='output',
                  num_joints=17,
                  hm_th=0.1) -> np.ndarray:
    """
    Unified decoder:
      - mode='output': model prediction (dict or list/tuple of tensors)
      - mode='label' : label tensor [B, 17+1+34+34, H, W]
    Returns: np.ndarray [B, 2*num_joints], normalized 0~1 (low-conf => -1)
    """
    if mode == 'output':
        # unpack to numpy
        if isinstance(data, dict):
            hm_t, ct_t, rg_t, of_t = data["heatmaps"], data["centers"], data["regs"], data["offsets"]
        else:
            hm_t, ct_t, rg_t, of_t = data  # list/tuple
        heatmaps = _to_numpy(hm_t)   # [B,J,H,W]
        centers  = _to_numpy(ct_t)   # [B,1,H,W]
        regs     = _to_numpy(rg_t)   # [B,2J,H,W]
        offsets  = _to_numpy(of_t)   # [B,2J,H,W]

        B, J, H, W = heatmaps.shape
        kps_mask_np = _to_numpy(kps_mask) if (kps_mask is not None) else np.ones((B, J), dtype=np.float32)

        heatmaps = heatmaps.copy()
        heatmaps[heatmaps < hm_th] = 0.0

        cx, cy = _maxPoint_numpy(centers)  # [B,1]
        dim0 = np.arange(B, dtype=np.int32).reshape(B, 1)
        dim1 = np.zeros((B, 1), dtype=np.int32)

        range_x = np.arange(W, dtype=np.float32).reshape(1, 1, 1, W)
        range_y = np.arange(H, dtype=np.float32).reshape(1, 1, H, 1)

        res = []
        for n in range(num_joints):
            reg_x_o = (regs[dim0, dim1 + n * 2,     cy, cx] + 0.5).astype(np.int32)
            reg_y_o = (regs[dim0, dim1 + n * 2 + 1, cy, cx] + 0.5).astype(np.int32)
            reg_x = (reg_x_o + cx).clip(0, W - 1)
            reg_y = (reg_y_o + cy).clip(0, H - 1)

            reg_x_hw = np.broadcast_to(reg_x.reshape(B, 1, 1, 1), (B, 1, H, W)).astype(np.float32)
            reg_y_hw = np.broadcast_to(reg_y.reshape(B, 1, 1, 1), (B, 1, H, W)).astype(np.float32)
            d2 = (range_x - reg_x_hw) ** 2 + (range_y - reg_y_hw) ** 2
            tmp_reg = heatmaps[:, n:n + 1, :, :] / (np.sqrt(d2) + 1.8)

            jx, jy = _maxPoint_numpy(tmp_reg)
            jx = jx.clip(0, W - 1)
            jy = jy.clip(0, H - 1)

            score = heatmaps[dim0, dim1 + n, jy, jx]
            off_x = offsets[dim0, dim1 + n * 2,     jy, jx]
            off_y = offsets[dim0, dim1 + n * 2 + 1, jy, jx]

            x_n = (jx + off_x) / float(W)
            y_n = (jy + off_y) / float(H)

            bad = np.logical_or((score < hm_th), (kps_mask_np[:, n:n+1] < 0.5)).astype(np.float32)
            x_n = x_n * (1.0 - bad) + (-1.0) * bad
            y_n = y_n * (1.0 - bad) + (-1.0) * bad

            res.extend([x_n, y_n])
        res = np.concatenate(res, axis=1)
        return res

    elif mode == 'label':
        data_np = _to_numpy(data)
        B, C, H, W = data_np.shape
        J = num_joints

        heatmaps = data_np[:, :J, :, :]
        centers  = data_np[:, J:J + 1, :, :]
        regs     = data_np[:, J + 1:J + 1 + 2 * J, :, :]
        offsets  = data_np[:, J + 1 + 2 * J:, :, :]

        kps_mask_np = _to_numpy(kps_mask) if (kps_mask is not None) else np.ones((B, J), dtype=np.float32)

        cx, cy = _maxPoint_numpy(centers)
        dim0 = np.arange(B, dtype=np.int32).reshape(B, 1)
        dim1 = np.zeros((B, 1), dtype=np.int32)

        res = []
        for n in range(J):
            reg_x_o = (regs[dim0, dim1 + n * 2,     cy, cx] + 0.5).astype(np.int32)
            reg_y_o = (regs[dim0, dim1 + n * 2 + 1, cy, cx] + 0.5).astype(np.int32)
            jx = (reg_x_o + cx).clip(0, W - 1)
            jy = (reg_y_o + cy).clip(0, H - 1)

            off_x = offsets[dim0, dim1 + n * 2,     jy, jx]
            off_y = offsets[dim0, dim1 + n * 2 + 1, jy, jx]

            x_n = (jx + off_x) / float(W)
            y_n = (jy + off_y) / float(H)

            mask = kps_mask_np[:, n:n + 1]
            x_n = x_n * mask + (-1.0) * (1.0 - mask)
            y_n = y_n * mask + (-1.0) * (1.0 - mask)

            res.extend([x_n, y_n])
        res = np.concatenate(res, axis=1)
        return res
    else:
        raise ValueError(f"unknown mode: {mode}")


# =========================
# The Task class
# =========================
class KptsTask():
    def __init__(self, cfg, model):
        self.cfg = cfg
        use_cuda = (self.cfg.get('GPU_ID', '') != '' and torch.cuda.is_available())
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.model = model.to(self.device)

        # ===== loss / optim / sched =====
        self.loss_func = MovenetLoss()
        self.optimizer = getOptimizer(self.cfg['optimizer'],
                                      self.model,
                                      self.cfg['learning_rate'],
                                      self.cfg['weight_decay'])
        self.scheduler = getSchedu(self.cfg['scheduler'], self.optimizer)

        # ===== alignment target (label map size) =====
        self.img_size = int(self.cfg.get("img_size", 192))
        self.target_stride = int(self.cfg.get("target_stride", 4))  # label通常在 stride=4
        self.target_hw = (self.img_size // self.target_stride, self.img_size // self.target_stride)

        self.best_score = float("-inf")         # 记录验证集最佳指标（越大越好）
        self.save_dir = self.cfg["save_dir"]    # 统一保存目录

        self.num_joints = int(
            self.cfg.get(
                "num_classes",
                self.cfg.get("task_params", {}).get("num_joints", 17)
            )
        )
        os.makedirs(self.save_dir, exist_ok=True)

    def _align_output_for_loss_and_decode(self, output) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        1) 取出四头
        2) resize 到 (Ht,Wt) = label feature map 大小
        3) 返回：list 版（给 loss）和 dict 版（给 decode/可视化）
        """
        if isinstance(output, dict):
            hm_t, ct_t, rg_t, of_t = output["heatmaps"], output["centers"], output["regs"], output["offsets"]
        else:
            hm_t, ct_t, rg_t, of_t = output

        Ht, Wt = self.target_hw
        hm_r, ct_r, rg_r, of_r = _resize_heads_to(hm_t, ct_t, rg_t, of_t, (Ht, Wt))
        as_list = [hm_r, ct_r, rg_r, of_r]
        as_dict = {"heatmaps": hm_r, "centers": ct_r, "regs": rg_r, "offsets": of_r}
        return as_list, as_dict

    # -------------------------
    # >>> core loops
    # -------------------------
    def train(self, train_loader, val_loader):
        for epoch in range(self.cfg['epochs']):
            self.onTrainStep(train_loader, epoch)
            self.onValidation(val_loader, epoch)
        self.onTrainEnd()

    def onTrainStep(self, train_loader, epoch):
        self.model.train()
        right_count = np.zeros(self.num_joints, dtype=np.int64)
        total_count = 0

        for batch_idx, (imgs, labels, kps_mask, img_names) in enumerate(train_loader):
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)
            kps_mask = kps_mask.to(self.device)

            raw_out = self.model(imgs)
            out_list, out_dict = self._align_output_for_loss_and_decode(raw_out)

            # loss
            heatmap_loss, bone_loss, center_loss, regs_loss, offset_loss = self.loss_func(out_list, labels, kps_mask)
            total_loss = heatmap_loss + center_loss + regs_loss + offset_loss + bone_loss

            self.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            if self.cfg.get('clip_gradient', 0):
                clipGradient(self.optimizer, float(self.cfg['clip_gradient']))
            self.optimizer.step()

            # evaluate (decode both pred & label)
            pre = movenetDecode(out_dict, kps_mask, mode='output')
            gt  = movenetDecode(labels,  kps_mask, mode='label')
            acc = myAcc(pre, gt)
            right_count += acc
            total_count += labels.shape[0]

            if batch_idx % self.cfg['log_interval'] == 0:
                iters_per_epoch = max(1, len(train_loader.dataset) // self.cfg['batch_size'])
                print('\r',
                      '%d/%d [%d/%d] '
                      'loss: %.4f (hm: %.3f b: %.3f c: %.3f r: %.3f o: %.3f) - acc: %.4f      ' %
                      (epoch + 1, self.cfg['epochs'],
                       batch_idx, iters_per_epoch,
                       total_loss.item(),
                       heatmap_loss.item(), bone_loss.item(), center_loss.item(),
                       regs_loss.item(), offset_loss.item(),
                       float(np.mean(right_count / max(1, total_count)))),
                      end='', flush=True)
        print()

    def onValidation(self, val_loader, epoch):
        self.model.eval()
        right_count = np.zeros(self.num_joints, dtype=np.int64)
        total_count = 0

        sum_hm = sum_b = sum_c = sum_r = sum_o = sum_total = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch_idx, (imgs, labels, kps_mask, img_names) in enumerate(val_loader):
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                kps_mask = kps_mask.to(self.device)

                raw_out = self.model(imgs)
                out_list, out_dict = self._align_output_for_loss_and_decode(raw_out)

                heatmap_loss, bone_loss, center_loss, regs_loss, offset_loss = self.loss_func(out_list, labels, kps_mask)
                total_loss = heatmap_loss + center_loss + regs_loss + offset_loss + bone_loss

                pre = movenetDecode(out_dict, kps_mask, mode='output')
                gt  = movenetDecode(labels,  kps_mask, mode='label')
                acc = myAcc(pre, gt)

                right_count += acc
                total_count += labels.shape[0]

                sum_hm += float(heatmap_loss.item())
                sum_b  += float(bone_loss.item())
                sum_c  += float(center_loss.item())
                sum_r  += float(regs_loss.item())
                sum_o  += float(offset_loss.item())
                sum_total += float(total_loss.item())
                n_batches += 1

        mean_hm = sum_hm / max(1, n_batches)
        mean_b  = sum_b  / max(1, n_batches)
        mean_c  = sum_c  / max(1, n_batches)
        mean_r  = sum_r  / max(1, n_batches)
        mean_o  = sum_o  / max(1, n_batches)
        mean_total = sum_total / max(1, n_batches)
        val_acc = float(np.mean(right_count / max(1, total_count)))

        print('LR: %f - [Val] loss: %.5f [hm: %.4f b: %.4f c: %.4f r: %.4f o: %.4f] - acc: %.4f\n'
              % (self.optimizer.param_groups[0]["lr"],
                 mean_total, mean_hm, mean_b, mean_c, mean_r, mean_o,
                 val_acc))

        # 学习率调度
        if 'default' in self.cfg['scheduler']:
            self.scheduler.step(val_acc)
        else:
            self.scheduler.step()

        # 保存 last.pt / best.pt
        last_path = os.path.join(self.save_dir, "last.pt")
        torch.save(self.model.state_dict(), last_path)

        if val_acc > self.best_score:
            self.best_score = val_acc
            best_path = os.path.join(self.save_dir, "best.pt")
            torch.save(self.model.state_dict(), best_path)
            print(f"[INFO] New best: acc={val_acc:.5f}  -> saved to {best_path}")
        else:
            print(f"[INFO] Kept best: acc={self.best_score:.5f}  (current {val_acc:.5f})")

    def onTrainEnd(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        if self.cfg.get("cfg_verbose", False):
            print("-" * 80)
            print(self.cfg)
            print("-" * 80)

    # -------------------------
    # >>> utils
    # -------------------------
    @staticmethod
    def _to_path(name: Union[str, os.PathLike, Iterable]) -> Union[str, None]:
        """
        Recursively unwrap DataLoader-collated structures until a str/PathLike is reached.
        Returns an OS fspath string or None if unsupported.
        """
        while isinstance(name, (list, tuple)):
            if not name:
                break
            name = name[0]
        if isinstance(name, (str, os.PathLike)):
            return os.fspath(name)
        return None

    # -------------------------
    # >>> inference / viz
    # -------------------------
    def predict(self, data_loader, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        self.model.eval()
        with torch.no_grad():
            for (img, img_name) in data_loader:
                img = img.to(self.device)
                output = self.model(img)
                _, out_dict = self._align_output_for_loss_and_decode(output)

                pre = movenetDecode(out_dict, None, mode='output')

                # —— 路径处理（稳定）—— #
                name = self._to_path(img_name)
                if name is None:
                    raise TypeError(f"Unsupported img_name structure: {type(img_name)} -> {img_name!r}")
                basename = os.path.basename(name)

                # —— 可视化 —— #
                img_np = np.transpose(img[0].detach().cpu().numpy(), (1, 2, 0))
                img_np = (img_np * 255).astype(np.uint8)
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                img_to_draw = img_np.copy()
                h, w = img_to_draw.shape[:2]
                for i in range(pre.shape[1] // 2):
                    x = int(max(0, min(1, pre[0, i * 2])) * w) if pre[0, i * 2] >= 0 else -1
                    y = int(max(0, min(1, pre[0, i * 2 + 1])) * h) if pre[0, i * 2 + 1] >= 0 else -1
                    if x >= 0 and y >= 0:
                        cv2.circle(img_to_draw, (x, y), 3, (255, 0, 0), 2)
                cv2.imwrite(os.path.join(save_dir, basename), img_to_draw)

                # —— 调试图（与旧版一致：4 张）—— #
                hm, ct, rg, of = _split_pred_tensors(out_dict)
                hm = hm[0]; ct = ct[0]; rg = rg[0]
                H_vis = self.img_size; W_vis = self.img_size
                hm_sum = cv2.resize(np.sum(hm, axis=0), (W_vis, H_vis)) * 255
                cv2.imwrite(os.path.join(save_dir, basename[:-4] + "_heatmaps.jpg"), hm_sum)
                img_dbg = img_np.copy()
                img_dbg[:, :, 0] = np.clip(img_dbg[:, :, 0].astype(np.float32) + hm_sum, 0, 255).astype(np.uint8)
                cv2.imwrite(os.path.join(save_dir, basename[:-4] + "_img.jpg"), img_dbg)
                cv2.imwrite(os.path.join(save_dir, basename[:-4] + "_center.jpg"),
                            cv2.resize(ct[0] * 255, (W_vis, H_vis)))
                cv2.imwrite(os.path.join(save_dir, basename[:-4] + "_regs0.jpg"),
                            cv2.resize(rg[0] * 255, (W_vis, H_vis)))

    # -------------------------
    # >>> evaluation (mean acc)
    # -------------------------
    def evaluate(self, data_loader):
        self.model.eval()
        right_count = np.zeros(self.num_joints, dtype=np.int64)
        total_count = 0
        with torch.no_grad():
            for batch_idx, (imgs, labels, kps_mask, img_names) in enumerate(data_loader):
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                kps_mask = kps_mask.to(self.device)

                output = self.model(imgs)
                _, out_dict = self._align_output_for_loss_and_decode(output)

                pre = movenetDecode(out_dict, kps_mask, mode='output')
                gt  = movenetDecode(labels,  kps_mask, mode='label')
                acc = myAcc(pre, gt)

                right_count += acc
                total_count += labels.shape[0]

        mean_acc = float(np.mean(right_count / max(1, total_count)))
        print('[Info] acc: %.3f%%\n' % (100. * mean_acc))

    # -------------------------
    # >>> model IO
    # -------------------------
    def modelLoad(self, model_path, data_parallel: bool = False):
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state, strict=True)
        if data_parallel and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model).to(self.device)
