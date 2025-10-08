# -*- coding: utf-8 -*-
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2


def draw_gaussian(heatmap: np.ndarray, center_xy: Tuple[float, float], radius: int):
    H, W = heatmap.shape
    cx, cy = float(center_xy[0]), float(center_xy[1])

    x0 = max(0, int(np.floor(cx - radius)))
    x1 = min(W - 1, int(np.ceil(cx + radius)))
    y0 = max(0, int(np.floor(cy - radius)))
    y1 = min(H - 1, int(np.ceil(cy + radius)))
    if x1 < x0 or y1 < y0:
        return

    xs = np.arange(x0, x1 + 1, dtype=np.float32)
    ys = np.arange(y0, y1 + 1, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    sigma = max(1.0, radius / 3.0)
    G = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma ** 2)).astype(np.float32)

    patch = heatmap[y0:y1 + 1, x0:x1 + 1]
    np.maximum(patch, G, out=patch)


def encode_single_targets(kps_xyv: np.ndarray, Hf: int, Wf: int, stride: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    (J, Hf, Wf) heatmaps, (2J, Hf, Wf) offsets, (J,) mask
    - offset 在高斯中心邻域写入（r_off=1），避免 argmax 落邻格时读不到 offset。
    """
    J = kps_xyv.shape[0]
    heatmaps = np.zeros((J, Hf, Wf), dtype=np.float32)
    offsets  = np.zeros((2 * J, Hf, Wf), dtype=np.float32)
    kps_mask = np.zeros((J,), dtype=np.float32)

    kps_f = kps_xyv.copy().astype(np.float32)
    kps_f[:, :2] /= float(stride)

    vis = kps_f[:, 2] > 0
    if vis.any():
        xs, ys = kps_f[vis, 0], kps_f[vis, 1]
        side = max(1.0, float(max(xs.max() - xs.min(), ys.max() - ys.min())))
    else:
        side = max(Hf, Wf) / 4.0

    r_kpt = max(1, int(round(0.025 * side)))
    r_off = 1

    for j in range(J):
        if kps_f[j, 2] <= 0:
            continue
        xj, yj = float(kps_f[j, 0]), float(kps_f[j, 1])
        draw_gaussian(heatmaps[j], (xj, yj), r_kpt)
        kps_mask[j] = 1.0

        cx, cy = int(np.round(xj)), int(np.round(yj))
        x0, x1 = max(0, cx - r_off), min(Wf - 1, cx + r_off)
        y0, y1 = max(0, cy - r_off), min(Hf - 1, cy + r_off)

        us = np.arange(x0, x1 + 1, dtype=np.float32)
        vs = np.arange(y0, y1 + 1, dtype=np.float32)
        U, V = np.meshgrid(us, vs)

        offsets[2 * j,     y0:y1 + 1, x0:x1 + 1] = (xj - U).astype(np.float32)
        offsets[2 * j + 1, y0:y1 + 1, x0:x1 + 1] = (yj - V).astype(np.float32)

    return heatmaps, offsets, kps_mask


def recompute_visibility(kps_xyv: np.ndarray, w: int, h: int) -> np.ndarray:
    """增强后基于边界重算 visibility（原本 v>0 的点若出界则置 0）"""
    kps = kps_xyv.copy()
    v0 = kps[:, 2] > 0
    inb = (kps[:, 0] >= 0) & (kps[:, 0] < w) & (kps[:, 1] >= 0) & (kps[:, 1] < h)
    kps[:, 2] = (v0 & inb).astype(np.float32)
    return kps


class CocoKeypointsDatasetAug(Dataset):
    def __init__(self,
                 img_root: str,
                 ann_path: str,
                 img_size: int,
                 target_stride: int,
                 is_train: bool,
                 aug_cfg: Dict[str, Any] = None,
                 min_kps_count: int = 1,
                 index_remap: List[int] = None):
        """
        读取“预处理后的 COCO 单人方图”并做几何/颜色增强
        """
        super().__init__()
        self.img_root = os.path.abspath(img_root)
        self.img_size = int(img_size)
        self.stride   = int(target_stride)
        self.is_train = bool(is_train)
        self.aug_cfg  = aug_cfg or {}
        self.index_remap = np.array(index_remap if index_remap is not None else np.arange(17))

        assert self.img_size % self.stride == 0, "img_size must be divisible by stride"
        self.Hf = self.img_size // self.stride
        self.Wf = self.img_size // self.stride

        self.items = self._load_annotations(ann_path, min_kps_count)

        # --- Albumentations pipeline ---
        # 几何：先做翻转/仿射（保持方形），最后统一 Resize -> img_size
        geo_transforms = [
            A.HorizontalFlip(p=float(self.aug_cfg.get("p_flip", 0.5))),
            A.Affine(
                scale=(1.0 + self.aug_cfg.get("scale_min", -0.2),
                       1.0 + self.aug_cfg.get("scale_max",  0.2)),
                translate_percent=(0.0, self.aug_cfg.get("translate", 0.08)),
                rotate=(-abs(self.aug_cfg.get("rotate", 30.0)),
                         abs(self.aug_cfg.get("rotate", 30.0))),
                fit_output=False,                 # 保持尺寸不变
                cval=114, mode=cv2.BORDER_CONSTANT, p=1.0
            ),
            A.Resize(self.img_size, self.img_size, interpolation=cv2.INTER_LINEAR)
        ]

        color_transforms = []
        if self.aug_cfg.get("color", True):
            color_transforms += [
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05,
                    p=float(self.aug_cfg.get("p_color", 0.8))
                ),
                A.ToFloat(max_value=255.0, p=0.0)  # 占位，无实际变更
            ]

        self.albu = A.Compose(
            geo_transforms + color_transforms,
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False)
        )

        self.normalize_to_tensor = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def _load_annotations(self, ann_path: str, min_kps: int):
        with open(ann_path, "r") as f:
            ann_json = json.load(f)

        img_id_to_info = {im["id"]: im for im in ann_json["images"]}
        person_id = next(c["id"] for c in ann_json["categories"] if c["name"] == "person")

        img_id_to_anns = defaultdict(list)
        for ann in ann_json["annotations"]:
            if ann["category_id"] == person_id and not ann.get("iscrowd", 0):
                img_id_to_anns[ann["image_id"]].append(ann)

        items = []
        for img_id, anns in img_id_to_anns.items():
            img_info = img_id_to_info.get(img_id)
            if not img_info:
                continue
            img_path = os.path.join(self.img_root, img_info["file_name"])
            if not os.path.exists(img_path):
                continue
            for person_ann in anns:
                if person_ann.get("num_keypoints", 0) >= min_kps:
                    items.append((img_path, person_ann))
        if not items:
            raise RuntimeError(f"No valid items under: {self.img_root} with {ann_path}")
        print(f"[Aug] Loaded {len(items)} samples from {os.path.basename(ann_path)}")
        return items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        img_path, person_ann = self.items[idx]

        # 读图（预处理阶段已方形裁剪）
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 关键点 (x,y,v)
        kps = np.array(person_ann["keypoints"], dtype=np.float32).reshape(-1, 3)
        kps = kps[self.index_remap]

        # --- Albumentations：几何/颜色增强（同步变换 keypoints） ---
        # 只把 (x,y) 传进去；v 先保留，增强后再基于边界重算
        kps_xy = [(float(x), float(y)) for x, y, v in kps]
        out = self.albu(image=img, keypoints=kps_xy)
        aug_img = out["image"]
        aug_kps_xy = np.array(out["keypoints"], dtype=np.float32).reshape(-1, 2)

        # 组合回 (x,y,v) 并重算可见性
        kps_aug = np.concatenate([aug_kps_xy, kps[:, 2:3]], axis=1)
        h, w = aug_img.shape[:2]
        # 边界裁剪，防止 round 落到边外
        eps = 1e-3
        kps_aug[:, 0] = np.clip(kps_aug[:, 0], 0.0, w - eps)
        kps_aug[:, 1] = np.clip(kps_aug[:, 1], 0.0, h - eps)
        kps_aug = recompute_visibility(kps_aug, w, h)

        # 归一化 + ToTensor
        img_t = self.normalize_to_tensor(image=aug_img)["image"]

        # 监督编码
        heatmaps, offsets, kps_mask = encode_single_targets(kps_aug, self.Hf, self.Wf, self.stride)
        label = np.concatenate([heatmaps, offsets], axis=0)

        return img_t, torch.from_numpy(label).float(), torch.from_numpy(kps_mask).float(), img_path


def create_kpts_dataloader(
        dataset_root: str,
        img_size: int,
        batch_size: int,
        target_stride: int,
        num_workers: int,
        pin_memory: bool,
        is_train: bool,
        aug_cfg: Dict[str, Any] = None,
        index_remap: List[int] = None
) -> DataLoader:
    """
    dataset_root 指向预处理输出根目录（包含 images/*2017 与 annotations/）
    """
    img_dir_name = "train2017" if is_train else "val2017"
    ann_file_name = f"person_keypoints_{img_dir_name}.json"

    root_path = Path(dataset_root)
    img_root = root_path / "images" / img_dir_name
    if not img_root.is_dir():
        img_root = root_path / img_dir_name  # 兼容
    ann_path = root_path / "annotations" / ann_file_name

    if not img_root.is_dir() or not ann_path.is_file():
        raise FileNotFoundError(f"Data not found. Checked: {img_root} and {ann_path}")

    dataset = CocoKeypointsDatasetAug(
        img_root=str(img_root),
        ann_path=str(ann_path),
        img_size=img_size,
        target_stride=target_stride,
        is_train=is_train,
        aug_cfg=aug_cfg,
        index_remap=index_remap
    )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=is_train
    )
