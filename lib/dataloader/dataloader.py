# lib/dataloader.py
# -*- coding: utf-8 -*-
"""
Unified COCO Keypoints dataset for MoveNet-style training
- 合并原 dataloader.py / data_augment.py / data_tools.py
- 输出 (img, label_tensor, kps_mask, img_path)
- label 通道顺序: [17 heatmaps] + [1 center] + [34 regs] + [34 offsets]
- 特征图尺寸: img_size // target_stride（默认为 4）
"""

import os
import json
import math
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# -----------------------
# 工具函数
# -----------------------
def _gaussian2d(shape: Tuple[int, int], sigma: float) -> np.ndarray:
    h, w = shape
    y = np.arange(0, h, 1, dtype=np.float32)
    x = np.arange(0, w, 1, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2))
    return g

def _draw_gaussian(heatmap: np.ndarray, center: Tuple[int, int], radius: int, k: float = 1.0):
    """在 heatmap 上画带裁剪的高斯核。heatmap: H×W；center: (x, y)"""
    diameter = 2 * radius + 1
    gaussian = _gaussian2d((diameter, diameter), sigma=diameter / 6.0)

    x, y = int(center[0]), int(center[1])
    h, w = heatmap.shape

    left, right = min(x, radius), min(w - x, radius + 1)
    top, bottom = min(y, radius), min(h - y, radius + 1)

    if right <= 0 or bottom <= 0 or left <= 0 or top <= 0:
        return

    masked_hm = heatmap[y - top:y + bottom, x - left:x + right]
    masked_g = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    np.maximum(masked_hm, masked_g * k, out=masked_hm)

def _letterbox(img: np.ndarray, dst_size: int, color=(114, 114, 114)) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    把任意 HxW 图像 letterbox 到 dst_size×dst_size，返回 (新图, 缩放比例, (pad_w, pad_h))
    - scale = dst_size / max(H, W)
    - 新图大小固定 dst_size×dst_size
    """
    h, w = img.shape[:2]
    scale = float(dst_size) / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img_resz = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    new_img = np.full((dst_size, dst_size, 3), color, dtype=img.dtype)
    pad_w = (dst_size - nw) // 2
    pad_h = (dst_size - nh) // 2
    new_img[pad_h:pad_h + nh, pad_w:pad_w + nw] = img_resz
    return new_img, scale, (pad_w, pad_h)

def _apply_hsv(img: np.ndarray, hgain=0.015, sgain=0.7, vgain=0.4):
    if hgain == 0 and sgain == 0 and vgain == 0:
        return img
    r = np.random.uniform(-1, 1, 3) * np.array([hgain, sgain, vgain]) + 1.0
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    img_hsv[..., 0] = (img_hsv[..., 0] * r[0]) % 180.0
    img_hsv[..., 1] = np.clip(img_hsv[..., 1] * r[1], 0, 255.0)
    img_hsv[..., 2] = np.clip(img_hsv[..., 2] * r[2], 0, 255.0)
    img = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return img

def _random_affine_points(pts: np.ndarray, M: np.ndarray) -> np.ndarray:
    """pts: [N,2]，仿射矩阵 2x3，输出映射后的 [N,2]"""
    if pts.size == 0:
        return pts
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    pts_aug = np.concatenate([pts, ones], axis=1)  # [N,3]
    pts_new = (M @ pts_aug.T).T  # [N,2]
    return pts_new

def _bbox_area(bbox: List[float]) -> float:
    x1, y1, w, h = bbox
    return max(0.0, w) * max(0.0, h)

def _get_center_from_kps(kps_xyv: np.ndarray) -> Tuple[float, float]:
    """kps_xyv: [17,3]，只用 v>0 的点做均值"""
    vis = kps_xyv[:, 2] > 0
    if vis.sum() == 0:
        # 如果都不可见，退化到直接平均全部（避免 nan）
        cx = kps_xyv[:, 0].mean()
        cy = kps_xyv[:, 1].mean()
    else:
        cx = kps_xyv[vis, 0].mean()
        cy = kps_xyv[vis, 1].mean()
    return float(cx), float(cy)

# -----------------------
# 数据集
# -----------------------
class CocoKeypointsDataset(Dataset):
    def __init__(self,
                 img_root: str,
                 ann_path: str,
                 img_size: int = 192,
                 target_stride: int = 4,
                 gaussian_radius: int = 2,
                 sigma_scale: float = 1.0,
                 use_color_aug: bool = True,
                 use_flip: bool = True,
                 use_rotate: bool = True,
                 rotate_deg: float = 30.0,
                 use_scale: bool = True,
                 scale_range: Tuple[float, float] = (0.75, 1.25),
                 select_person: str = "largest",  # 或 "random"
                 is_train: bool = True):
        """
        img_root: 图片目录
        ann_path: COCO keypoints json（person_keypoints_train2017.json / val2017.json）
        img_size: 最终输入分辨率（正方形）
        target_stride: label 特征图步长（默认4 => 192/4=48）
        gaussian_radius: 高斯半径（像素，作用在特征图上）
        sigma_scale: 高斯强度缩放
        其余为增广开关与强度
        """
        super().__init__()
        self.img_root = img_root
        self.ann_path = ann_path
        self.img_size = int(img_size)
        self.stride = int(target_stride)
        self.Hf = self.img_size // self.stride
        self.Wf = self.img_size // self.stride

        self.gr = int(gaussian_radius)
        self.sigma_scale = float(sigma_scale)
        self.is_train = is_train

        self.use_color_aug = use_color_aug
        self.use_flip = use_flip
        self.use_rotate = use_rotate
        self.rotate_deg = float(rotate_deg)
        self.use_scale = use_scale
        self.scale_range = scale_range
        self.select_person = select_person

        # 读取 COCO 风格标注（subset 仅 person）
        with open(self.ann_path, "r") as f:
            ann = json.load(f)

        self.imgid_to_info: Dict[int, Dict[str, Any]] = {im["id"]: im for im in ann["images"]}
        anns_all = [a for a in ann["annotations"] if a.get("category_id", 1) == 1]
        # 按 image_id 聚合
        imgid_to_anns: Dict[int, List[Dict[str, Any]]] = {}
        for a in anns_all:
            imgid_to_anns.setdefault(a["image_id"], []).append(a)

        # 仅保留至少有一个 annotation 的图像
        self.items: List[Tuple[str, List[Dict[str, Any]]]] = []
        for img_id, anns in imgid_to_anns.items():
            info = self.imgid_to_info.get(img_id)
            if info is None:
                continue
            file_name = info["file_name"]
            img_path = os.path.join(self.img_root, file_name)
            if os.path.isfile(img_path):
                self.items.append((img_path, anns))

    def __len__(self):
        return len(self.items)

    # ---- 增广：几何 + 颜色 ----
    def _augment_geom(self, img: np.ndarray, kps_xyv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        在 letterbox 前做随机仿射：缩放、旋转、水平翻转（对 kps 同步仿射）
        注意：这里的仿射在**原始坐标系**做，后续再 letterbox。
        """
        h, w = img.shape[:2]
        M = np.eye(3, dtype=np.float32)

        # 随机尺度
        s = 1.0
        if self.is_train and self.use_scale:
            s = np.random.uniform(self.scale_range[0], self.scale_range[1])

        # 随机旋转
        r = 0.0
        if self.is_train and self.use_rotate:
            r = np.random.uniform(-self.rotate_deg, self.rotate_deg)

        # 以图像中心为原点构建仿射
        cx, cy = w * 0.5, h * 0.5
        # 先平移 -> 旋转 -> 缩放 -> 平移回去
        T1 = np.array([[1, 0, -cx],
                       [0, 1, -cy],
                       [0, 0, 1]], dtype=np.float32)
        R = cv2.getRotationMatrix2D((0, 0), r, s)  # 2x3
        R = np.vstack([R, [0, 0, 1]]).astype(np.float32)
        T2 = np.array([[1, 0, cx],
                       [0, 1, cy],
                       [0, 0, 1]], dtype=np.float32)
        M = T2 @ R @ T1  # 3x3

        # 水平翻转
        if self.is_train and self.use_flip and (np.random.rand() < 0.5):
            F = np.array([[-1, 0, w],
                          [0, 1, 0],
                          [0, 0, 1]], dtype=np.float32)
            M = F @ M

            # 同时需要对关键点做对称索引交换（COCO 左右翻转），这里给出一对常见映射：
            flip_pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
            for a, b in flip_pairs:
                kps_xyv[[a, b]] = kps_xyv[[b, a]]

        # 应用仿射
        img = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
        kps_xyv[:, :2] = _random_affine_points(kps_xyv[:, :2], M[:2, :])

        return img, kps_xyv

    def _encode_targets(self,
                        kps_xyv: np.ndarray,
                        center_xy: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        输入：kps_xyv（letterbox 后、单位像素），center_xy（letterbox 后）
        输出特征图上的：
        - heatmaps: [17,Hf,Wf]
        - center:   [1,Hf,Wf]
        - regs:     [2*17,Hf,Wf]   (仅中心网格位置有效)
        - offsets:  [2*17,Hf,Wf]   (仅各关键点所在网格位置有效)
        - kps_mask: [17]   (可见=1，不可见=0)
        """
        J = 17
        heatmaps = np.zeros((J, self.Hf, self.Wf), dtype=np.float32)
        centers  = np.zeros((1, self.Hf, self.Wf), dtype=np.float32)
        regs     = np.zeros((2 * J, self.Hf, self.Wf), dtype=np.float32)
        offsets  = np.zeros((2 * J, self.Hf, self.Wf), dtype=np.float32)
        kps_mask = np.zeros((J,), dtype=np.float32)

        # 关键点映射到特征图坐标
        kps_f = kps_xyv.copy()
        kps_f[:, 0] = kps_f[:, 0] / self.stride
        kps_f[:, 1] = kps_f[:, 1] / self.stride

        # 画 heatmaps
        for j in range(J):
            v = kps_f[j, 2]
            if v > 0:
                xj, yj = kps_f[j, 0], kps_f[j, 1]
                if xj >= 0 and yj >= 0 and xj < self.Wf and yj < self.Hf:
                    _draw_gaussian(heatmaps[j], (int(round(xj)), int(round(yj))), self.gr, k=self.sigma_scale)
                    kps_mask[j] = 1.0

        # 画 center（用 center_xy）
        cx = center_xy[0] / self.stride
        cy = center_xy[1] / self.stride
        if 0 <= cx < self.Wf and 0 <= cy < self.Hf:
            _draw_gaussian(centers[0], (int(round(cx)), int(round(cy))), self.gr + 1, k=self.sigma_scale)

        # 计算 regs（以中心网格为锚，写 dx,dy 的整数位）
        cx_i = int(np.clip(np.floor(cx + 0.5), 0, self.Wf - 1))
        cy_i = int(np.clip(np.floor(cy + 0.5), 0, self.Hf - 1))
        for j in range(J):
            if kps_mask[j] > 0:
                dx = kps_f[j, 0] - cx
                dy = kps_f[j, 1] - cy
                regs[2 * j,     cy_i, cx_i] = float(np.floor(dx))  # 与你现有 loss 对齐：取整 + 0.5 再加回
                regs[2 * j + 1, cy_i, cx_i] = float(np.floor(dy))

        # 计算 offsets（在每个关键点所在网格记录小数偏移）
        for j in range(J):
            if kps_mask[j] > 0:
                xj, yj = kps_f[j, 0], kps_f[j, 1]
                gx = int(np.clip(np.floor(xj + 0.5), 0, self.Wf - 1))
                gy = int(np.clip(np.floor(yj + 0.5), 0, self.Hf - 1))
                ox = xj - gx
                oy = yj - gy
                offsets[2 * j,     gy, gx] = float(ox)
                offsets[2 * j + 1, gy, gx] = float(oy)

        return heatmaps, centers, regs, offsets, kps_mask

    def __getitem__(self, idx: int):
        img_path, anns = self.items[idx]
        img = cv2.imread(img_path)
        assert img is not None, f"fail to read image: {img_path}"
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # —— 选择一个 person 做“中心”的回归锚（其 keypoints 参与 regs/offset 的中心采样）——
        if len(anns) == 1:
            person = anns[0]
        else:
            if self.select_person == "largest":
                person = max(anns, key=lambda a: _bbox_area(a.get("bbox", [0, 0, 0, 0])))
            else:
                person = anns[np.random.randint(0, len(anns))]

        # 取该人的 keypoints（17*3: x,y,v）
        kp = np.array(person["keypoints"], dtype=np.float32).reshape(-1, 3)  # [17,3], 原图坐标

        # 可选：把所有人 heatmap 叠加（常见做法增强监督），但中心 & regs/offset 只用被选的人
        all_kps = [kp]
        for a in anns:
            if a is not person:
                all_kps.append(np.array(a["keypoints"], dtype=np.float32).reshape(-1, 3))
        # —— 先做几何增广（原图坐标系）——
        if self.is_train:
            img, kp = self._augment_geom(img, kp)

        # 其它人的关键点也要跟着同样的仿射（上面 _augment_geom 返回的仿射矩阵没直接暴露，
        # 这里简化做法：只对“被选中者”做几何增广，heatmap 仅来自被选中者；若你希望把所有人的 heatmap 累加，
        # 可把 _augment_geom 调整为返回 M，并同步应用到 all_kps。）
        all_kps = [kp]  # 此处只保留被选中者，避免错配

        # —— 颜色增广 ——
        if self.is_train and self.use_color_aug:
            # 注意：此处 img 还是 RGB，如果你更习惯在 BGR 里做 HSV，可先转 BGR 再转回。
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            bgr = _apply_hsv(bgr, hgain=0.015, sgain=0.7, vgain=0.4)
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # —— letterbox 到正方形输入 ——
        img_lb, scale, (pad_w, pad_h) = _letterbox(img, self.img_size, color=(114, 114, 114))

        # 把关键点映射到 letterbox 后坐标
        def map_kps_to_letterbox(kps_xyv: np.ndarray) -> np.ndarray:
            out = kps_xyv.copy()
            out[:, 0] = out[:, 0] * scale + pad_w
            out[:, 1] = out[:, 1] * scale + pad_h
            return out

        mapped_all = [map_kps_to_letterbox(k) for k in all_kps]
        # 选择“中心”坐标（被选中者）——用可见关键点的均值
        cx, cy = _get_center_from_kps(mapped_all[0])

        # —— 生成 supervision（特征图尺度）——
        # heatmaps 可按需对所有人累加，这里用被选中者（和你的 loss/decoder 更一致）
        heatmaps, centers, regs, offsets, kps_mask = self._encode_targets(mapped_all[0], (cx, cy))

        # label 拼接
        label = np.concatenate([heatmaps, centers, regs, offsets], axis=0)  # [17+1+34+34, Hf, Wf] = [86,Hf,Wf]

        # to tensor
        img_t = torch.from_numpy(img_lb.transpose(2, 0, 1)).float() / 255.0  # [3,H,W], 0~1
        label_t = torch.from_numpy(label).float()
        kps_mask_t = torch.from_numpy(kps_mask).float()

        return img_t, label_t, kps_mask_t, img_path


# -----------------------
# 对外封装（兼容旧接口）
# -----------------------
class Data:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.img_size = int(cfg.get("img_size", 192))
        self.target_stride = int(cfg.get("target_stride", 4))
        self.batch_size = int(cfg.get("batch_size", 64))
        self.num_workers = int(cfg.get("num_workers", 8))
        self.pin_memory = bool(cfg.get("pin_memory", True))

        self.train_img_path = cfg["img_path"]
        self.val_img_path = cfg["img_path"]  # 若验证集图像和训练集同目录，可共用
        self.train_label_path = cfg["train_label_path"]
        self.val_label_path = cfg["val_label_path"]

        # aug 配置（也可从 cfg 读取）
        self.aug = dict(
            gaussian_radius=cfg.get("gaussian_radius", 2),
            sigma_scale=cfg.get("sigma_scale", 1.0),
            use_color_aug=cfg.get("use_color_aug", True),
            use_flip=cfg.get("use_flip", True),
            use_rotate=cfg.get("use_rotate", True),
            rotate_deg=cfg.get("rotate_deg", 30.0),
            use_scale=cfg.get("use_scale", True),
            scale_range=tuple(cfg.get("scale_range", (0.75, 1.25))),
            select_person=cfg.get("select_person", "largest"),
        )

    def getTrainValDataloader(self) -> Tuple[DataLoader, DataLoader]:
        train_set = CocoKeypointsDataset(
            img_root=self.train_img_path,
            ann_path=self.train_label_path,
            img_size=self.img_size,
            target_stride=self.target_stride,
            is_train=True,
            **self.aug,
        )
        val_set = CocoKeypointsDataset(
            img_root=self.val_img_path,
            ann_path=self.val_label_path,
            img_size=self.img_size,
            target_stride=self.target_stride,
            is_train=False,
            # 验证默认关闭强增广，仅保留必要参数
            gaussian_radius=self.aug["gaussian_radius"],
            sigma_scale=self.aug["sigma_scale"],
            use_color_aug=False,
            use_flip=False,
            use_rotate=False,
            rotate_deg=0.0,
            use_scale=False,
            scale_range=(1.0, 1.0),
            select_person=self.aug["select_person"],
        )

        train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=min(self.batch_size, 64),
            shuffle=False,
            num_workers=max(1, self.num_workers // 2),
            pin_memory=self.pin_memory,
            drop_last=False,
        )
        print(f"[INFO] Total train images: {len(train_set)}, val images: {len(val_set)}")
        return train_loader, val_loader

    # 可选的可视化/调试函数
    @staticmethod
    def preview_label(label_t: torch.Tensor, save_path: str):
        """快速把 label 的热图合成可视化存盘（用于调试）"""
        with torch.no_grad():
            l = label_t.detach().cpu().numpy()
        J = 17
        hm = l[:J].sum(axis=0)
        hm = (hm / (hm.max() + 1e-6) * 255).astype(np.uint8)
        hm = cv2.resize(hm, (192, 192))
        cv2.imwrite(save_path, hm)
