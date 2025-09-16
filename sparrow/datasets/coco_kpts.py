# sparrow/datasets/coco_kpts.py
# -*- coding: utf-8 -*-
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from sparrow.datasets.common import (random_affine_points, draw_gaussian,
                                     apply_hsv, letterbox, get_center_from_kps)


class CocoKeypointsDataset(Dataset):
    """
    COCO 姿态估计数据集（内聚版）。
    - 将每个 image 中的每个 person 展开成一条独立样本
    - 统一完成几何/颜色增广、letterbox、以及 supervision 编码
    - 输出: (img_tensor [3,H,W], label_tensor [86,Hf,Wf], kps_mask [17], img_path)
    """

    # COCO 17 点左右翻转索引对
    FLIP_PAIRS = (
        (1, 2), (3, 4), (5, 6), (7, 8),
        (9, 10), (11, 12), (13, 14), (15, 16)
    )

    def __init__(self,
                 img_root: str,
                 ann_path: str,
                 img_size: int,
                 target_stride: int,
                 is_train: bool,
                 *,  # 之后强制关键字参数
                 use_dynamic_radius: bool = True,
                 kpt_radius_factor: float = 0.025,
                 ctr_radius_factor: float = 0.035,
                 gaussian_radius: int = 2,
                 sigma_scale: float = 1.0,
                 min_radius: int = 1,
                 use_color_aug: bool = True,
                 use_flip: bool = True,
                 use_rotate: bool = True,
                 rotate_deg: float = 30.0,
                 use_scale: bool = True,
                 scale_range: Tuple[float, float] = (0.75, 1.25)):
        super().__init__()

        # --- 核心属性 ---
        self.img_root = os.path.abspath(img_root)
        self.ann_path = ann_path
        self.img_size = int(img_size)
        self.stride = int(target_stride)
        self.is_train = bool(is_train)

        assert self.img_size % self.stride == 0, \
            f"img_size({self.img_size}) must be divisible by stride({self.stride})"
        self.Hf = self.img_size // self.stride
        self.Wf = self.img_size // self.stride

        # --- 增广与监督参数 ---
        self.use_dynamic_radius = bool(use_dynamic_radius)
        self.kpt_radius_factor = float(kpt_radius_factor)
        self.ctr_radius_factor = float(ctr_radius_factor)
        self.gr = int(gaussian_radius)          # 非动态半径下使用
        self.sigma_scale = float(sigma_scale)   # 高斯核缩放
        self.min_radius = int(min_radius)

        self.use_color_aug = bool(use_color_aug)
        self.use_flip = bool(use_flip)
        self.use_rotate = bool(use_rotate)
        self.rotate_deg = float(rotate_deg)
        self.use_scale = bool(use_scale)
        self.scale_range = scale_range

        # --- 加载与预处理标注 ---
        self._ann_images: Dict[int, Dict[str, Any]] = {}  # 缓存 images 字段，避免重复 I/O
        self.items = self._load_annotations()
        self.img_id_to_info = self._get_img_info_dict()  # 如需外部调试可用

    def _load_annotations(self) -> List[Tuple[str, Dict[str, Any]]]:
        with open(self.ann_path, "r") as f:
            ann_json = json.load(f)

        images = ann_json.get("images", [])
        annotations = ann_json.get("annotations", [])
        categories = ann_json.get("categories", [])

        self._ann_images = {im["id"]: im for im in images}

        # person 类别 id（默认 1）
        person_id = next((c.get("id") for c in categories if c.get("name") == "person"), 1)

        # 仅保留 person
        anns_all = [a for a in annotations if a.get("category_id", person_id) == person_id]
        if self.is_train:
            # 常见清洗：剔除 crowd / 无关键点
            anns_all = [a for a in anns_all if a.get("iscrowd", 0) == 0 and a.get("num_keypoints", 0) > 0]

        # 按 image_id 聚合
        img_id_to_anns: Dict[int, List[Dict[str, Any]]] = {}
        for a in anns_all:
            img_id_to_anns.setdefault(a["image_id"], []).append(a)

        # 将每个 image 中的每个 person 展开为一条样本
        items: List[Tuple[str, Dict[str, Any]]] = []
        for img_id, anns in img_id_to_anns.items():
            info = self._ann_images.get(img_id)
            if not info:
                continue
            file_name = info.get("file_name")
            if not file_name:
                continue
            path = os.path.abspath(os.path.join(self.img_root, file_name))
            if not os.path.isfile(path):
                continue
            for person_ann in anns:
                items.append((path, person_ann))

        if not items:
            raise FileNotFoundError(f"No valid items found under: {self.img_root} with {self.ann_path}")
        return items

    def _get_img_info_dict(self) -> Dict[int, Dict[str, Any]]:
        # 返回缓存，无需再次读盘
        return dict(self._ann_images)

    def __len__(self):
        return len(self.items)

    def _augment_geom(self, img: np.ndarray, kps_xyv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        在 letterbox 前做随机仿射：缩放、旋转、水平翻转（对 kps 同步仿射）
        注意：仿射在原始坐标系完成，随后再 letterbox。
        """
        h, w = img.shape[:2]

        # 随机尺度
        s = 1.0
        if self.is_train and self.use_scale:
            s = np.random.uniform(self.scale_range[0], self.scale_range[1])

        # 随机旋转
        r = 0.0
        if self.is_train and self.use_rotate:
            r = np.random.uniform(-self.rotate_deg, self.rotate_deg)

        # 以图像中心为原点构建仿射: 平移->旋转缩放->平移回去
        cx, cy = w * 0.5, h * 0.5
        T1 = np.array([[1, 0, -cx],
                       [0, 1, -cy],
                       [0, 0, 1]], dtype=np.float32)
        R = cv2.getRotationMatrix2D((0, 0), r, s)  # 2x3
        R = np.vstack([R, [0, 0, 1]]).astype(np.float32)
        T2 = np.array([[1, 0, cx],
                       [0, 1, cy],
                       [0, 0, 1]], dtype=np.float32)
        M = T2 @ R @ T1  # 3x3

        # 水平翻转（含关键点左右交换）
        if self.is_train and self.use_flip and (np.random.rand() < 0.5):
            F = np.array([[-1, 0, w],
                          [0, 1, 0],
                          [0, 0, 1]], dtype=np.float32)
            M = F @ M
            for a, b in self.FLIP_PAIRS:
                kps_xyv[[a, b]] = kps_xyv[[b, a]]

        # 应用仿射（2x3）并变换关键点
        A2 = M[:2, :]  # 2x3
        img = cv2.warpAffine(img, A2, (w, h), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
        kps_xyv[:, :2] = random_affine_points(kps_xyv[:, :2], A2)

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
        - kps_mask: [17]   (可见=1，且点落在特征图内)
        """
        J = 17
        heatmaps = np.zeros((J, self.Hf, self.Wf), dtype=np.float32)
        centers  = np.zeros((1, self.Hf, self.Wf), dtype=np.float32)
        regs     = np.zeros((2 * J, self.Hf, self.Wf), dtype=np.float32)
        offsets  = np.zeros((2 * J, self.Hf, self.Wf), dtype=np.float32)
        kps_mask = np.zeros((J,), dtype=np.float32)

        # 映射到特征图坐标
        kps_f = kps_xyv.copy()
        kps_f[:, 0] = kps_f[:, 0] / self.stride
        kps_f[:, 1] = kps_f[:, 1] / self.stride

        # 根据可见关键点外接框估算尺度
        vis = (kps_f[:, 2] > 0)
        if np.any(vis):
            xs, ys = kps_f[vis, 0], kps_f[vis, 1]
            w_box = float(xs.max() - xs.min())
            h_box = float(ys.max() - ys.min())
            side = max(1.0, max(w_box, h_box))
        else:
            side = float(max(self.Wf, self.Hf)) / 4.0  # 兜底

        # 半径设定
        if self.use_dynamic_radius:
            r_kpt = max(self.min_radius, int(round(self.kpt_radius_factor * side)))
            r_ctr = max(self.min_radius + 1, int(round(self.ctr_radius_factor * side)))
        else:
            r_kpt = int(self.gr)
            r_ctr = int(self.gr + 1)

        # 关键点热图与 mask
        for j in range(J):
            v = kps_f[j, 2]
            if v > 0:
                xj, yj = kps_f[j, 0].item(), kps_f[j, 1].item()
                if 0 <= xj < self.Wf and 0 <= yj < self.Hf:
                    draw_gaussian(heatmaps[j], (int(round(xj)), int(round(yj))), r_kpt, k=self.sigma_scale)
                    kps_mask[j] = 1.0

        # 中心热图（用 center_xy）
        cx = center_xy[0] / self.stride
        cy = center_xy[1] / self.stride
        if 0 <= cx < self.Wf and 0 <= cy < self.Hf:
            draw_gaussian(centers[0], (int(round(cx)), int(round(cy))), r_ctr, k=self.sigma_scale)

        # regs：以中心网格为锚，写 dx, dy（浮点）
        cx_i = int(np.clip(np.floor(cx + 0.5), 0, self.Wf - 1))
        cy_i = int(np.clip(np.floor(cy + 0.5), 0, self.Hf - 1))
        for j in range(J):
            if kps_mask[j] > 0:
                dx = kps_f[j, 0] - cx
                dy = kps_f[j, 1] - cy
                regs[2 * j,     cy_i, cx_i] = float(dx)
                regs[2 * j + 1, cy_i, cx_i] = float(dy)

        # offsets：在每个关键点所在网格记录小数偏移
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
        img_path, person = self.items[idx]
        img = cv2.imread(img_path)
        assert img is not None, f"fail to read image: {img_path}"
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 该 person 的 keypoints（[17,3] 原图坐标）
        kp = np.array(person["keypoints"], dtype=np.float32).reshape(-1, 3)

        # 几何增广
        if self.is_train:
            img, kp = self._augment_geom(img, kp)

        # 颜色增广
        if self.is_train and self.use_color_aug:
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            bgr = apply_hsv(bgr, hgain=0.015, sgain=0.7, vgain=0.4)
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # letterbox 到方形
        img_lb, scale, (pad_w, pad_h) = letterbox(img, self.img_size, color=(114, 114, 114))

        # 将关键点映射到 letterbox 后坐标
        def map_kps_to_letterbox(kps_xyv: np.ndarray) -> np.ndarray:
            out = kps_xyv.copy()
            out[:, 0] = out[:, 0] * scale + pad_w
            out[:, 1] = out[:, 1] * scale + pad_h
            return out

        kps_mapped = map_kps_to_letterbox(kp)

        # 中心点：可见关键点质心；若不可用则 bbox/图像中心兜底
        cx, cy = get_center_from_kps(kps_mapped)
        if not (np.isfinite(cx) and np.isfinite(cy)):
            if "bbox" in person:
                bx, by, bw, bh = person["bbox"]
                cx, cy = bx + bw * 0.5, by + bh * 0.5
            else:
                cx = cy = float(self.img_size * 0.5)

        # 生成 supervision
        heatmaps, centers, regs, offsets, kps_mask = self._encode_targets(kps_mapped, (cx, cy))

        # label 拼接: 17 + 1 + 34 + 34 = 86
        label = np.concatenate([heatmaps, centers, regs, offsets], axis=0)

        # 转 tensor
        img_t = torch.from_numpy(img_lb.transpose(2, 0, 1)).float() / 255.0
        label_t = torch.from_numpy(label).float()
        kps_mask_t = torch.from_numpy(kps_mask).float()

        return img_t, label_t, kps_mask_t, img_path


def create_kpts_dataloader(
        dataset_root: str,
        img_size: int,
        target_stride: int,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        aug_cfg: Dict[str, Any],
        is_train: bool
) -> DataLoader:
    """
    工厂函数：创建姿态估计 DataLoader。
    - 兼容 <root>/train2017 和 <root>/images/train2017 两种结构
    - 训练集支持完整增广；验证集默认关闭大部分增广，并采用固定半径（更稳定的评估）
    """
    img_dir_name = "train2017" if is_train else "val2017"
    ann_file_name = "person_keypoints_train2017.json" if is_train else "person_keypoints_val2017.json"

    root_path = Path(dataset_root)
    img_root = root_path / "images" / img_dir_name
    if not img_root.is_dir():
        img_root = root_path / img_dir_name
    ann_path = root_path / "annotations" / ann_file_name

    if not img_root.is_dir() or not ann_path.is_file():
        raise FileNotFoundError(f"Data not found. Checked: {img_root} and {ann_path}")

    # 过滤 aug_cfg 非法键，避免 TypeError
    valid_keys = {
        "use_dynamic_radius", "kpt_radius_factor", "ctr_radius_factor",
        "gaussian_radius", "sigma_scale", "min_radius",
        "use_color_aug", "use_flip", "use_rotate", "rotate_deg",
        "use_scale", "scale_range",
    }
    safe_aug = {k: v for k, v in (aug_cfg or {}).items() if k in valid_keys}

    if is_train:
        dataset = CocoKeypointsDataset(
            img_root=str(img_root),
            ann_path=str(ann_path),
            img_size=img_size,
            target_stride=target_stride,
            is_train=True,
            **safe_aug
        )
    else:
        # 验证集：默认关闭颜色/几何增广，采用固定半径（可按需改回动态）
        dataset = CocoKeypointsDataset(
            img_root=str(img_root),
            ann_path=str(ann_path),
            img_size=img_size,
            target_stride=target_stride,
            is_train=False,
            use_color_aug=False,
            use_flip=False,
            use_rotate=False,
            use_scale=False,
            use_dynamic_radius=False,   # 固定半径评估更稳定
            gaussian_radius=safe_aug.get("gaussian_radius", 2),
            sigma_scale=safe_aug.get("sigma_scale", 1.0)
        )

    # 组装 DataLoader kwargs（避免给 None 的 prefetch_factor）
    dl_kwargs: Dict[str, Any] = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=is_train,
    )
    if num_workers > 0:
        dl_kwargs.update(
            persistent_workers=bool(is_train),
        )
        # PyTorch < 2.0 也支持 prefetch_factor；若报错可去掉
        dl_kwargs.update(prefetch_factor=2)
        dl_kwargs.update(worker_init_fn=lambda wid: np.random.seed(torch.initial_seed() % 2**32))

    return DataLoader(**dl_kwargs)
