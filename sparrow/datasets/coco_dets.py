# -*- coding: utf-8 -*-
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# 辅助：xyxy 裁剪
# -----------------------------
def clip_xyxy(boxes: np.ndarray, w: int, h: int):
    if boxes.size == 0:
        return boxes
    boxes[:, 0] = np.clip(boxes[:, 0], 0, w - 1e-3)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, h - 1e-3)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w - 1e-3)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h - 1e-3)
    return boxes


# -----------------------------
# 无形变 letterbox 到 (S,S)
#  - 先按最长边等比缩放到 S，再在两侧/上下 pad 到正方形
#  - 同步变换 bboxes (xyxy)
# -----------------------------
def letterbox_pad(
    img: np.ndarray,
    bboxes_xyxy: np.ndarray,
    size: int,
    pad_value: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, np.ndarray]:
    h0, w0 = img.shape[:2]
    if h0 == 0 or w0 == 0:
        raise ValueError("Invalid image with zero size.")
    scale = float(size) / max(h0, w0)
    nh, nw = int(round(h0 * scale)), int(round(w0 * scale))
    img_rs = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    # pad 到正方形
    top = (size - nh) // 2
    left = (size - nw) // 2
    bottom = size - nh - top
    right = size - nw - left
    img_pad = cv2.copyMakeBorder(img_rs, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=pad_value)

    # 同步 bbox
    if bboxes_xyxy.size > 0:
        b = bboxes_xyxy.astype(np.float32).copy()
        b *= scale
        b[:, [0, 2]] += left
        b[:, [1, 3]] += top
        b = clip_xyxy(b, size, size)
    else:
        b = bboxes_xyxy.reshape(0, 4).astype(np.float32)
    return img_pad, b


# -----------------------------
# Dataset
# -----------------------------
class CocoDetectionDataset(Dataset):
    def __init__(
        self,
        img_root: str,
        ann_path: str,
        img_size: int,
        is_train: bool,
        aug_cfg: Dict[str, Any] = None,
        filter_crowd: bool = True,
        min_box_size: float = 1.0,
    ):
        """
        img_root : COCO images 目录（含 train2017 / val2017 子目录）
        ann_path : annotations/instances_*.json
        img_size : 输出正方形尺寸（用于 letterbox+pad）
        aug_cfg  : {
            "p_flip":0.5, "scale":(0.9,1.1), "rotate":10.0,
            "translate":0.08, "shear":0.0, "color":True, "p_color":0.8
        }
        """
        super().__init__()
        self.img_root = os.path.abspath(img_root)
        self.img_size = int(img_size)
        self.is_train = bool(is_train)
        self.aug_cfg = aug_cfg or {}
        self.filter_crowd = bool(filter_crowd)
        self.min_box_size = float(min_box_size)

        self.items, self.catid2contig = self._load_annotations(ann_path)

        # ---------------- Albumentations pipeline ----------------
        # 读取配置并做“合法化”
        p_flip = float(self.aug_cfg.get("p_flip", 0.5))
        scale = self.aug_cfg.get("scale", (1.0, 1.0))
        if isinstance(scale, (int, float)):
            scale = (float(scale), float(scale))
        scale = (float(scale[0]), float(scale[1]))
        rot_deg = float(self.aug_cfg.get("rotate", 10.0))
        tmax = float(self.aug_cfg.get("translate", 0.08))
        shear_deg = float(self.aug_cfg.get("shear", 0.0))
        use_color = bool(self.aug_cfg.get("color", True))
        p_color = float(self.aug_cfg.get("p_color", 0.8))

        geo = [
            A.HorizontalFlip(p=p_flip),
            A.Affine(
                scale=scale,
                translate_percent={"x": (-tmax, tmax), "y": (-tmax, tmax)},
                rotate=(-abs(rot_deg), abs(rot_deg)),
                shear=(-abs(shear_deg), abs(shear_deg)),  # 不使用 None，避免参数校验错误
                fit_output=True,          # 先保证几何变换后不截断，稍后再 letterbox
                cval=114,
                mode=cv2.BORDER_CONSTANT,
                p=1.0,
            ),
        ]
        color = []
        if use_color:
            color = [
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=p_color
                )
            ]

        # bboxes 与 labels 同步变换
        self.albu = A.Compose(
            geo + color,
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["class_labels"],
                min_area=0,
                min_visibility=0.0,
            ),
        )

        self.normalize_to_tensor = A.Compose(
            [
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )

    def _load_annotations(self, ann_path: str):
        with open(ann_path, "r") as f:
            anno = json.load(f)

        # COCO 类别 id → 连续 [0..C-1]
        cats = sorted(anno["categories"], key=lambda c: c["id"])
        catid2contig = {c["id"]: i for i, c in enumerate(cats)}  # e.g. person->0, ...

        img_id_to_info = {im["id"]: im for im in anno["images"]}
        img_id_to_anns = defaultdict(list)
        for a in anno["annotations"]:
            if self.filter_crowd and a.get("iscrowd", 0) == 1:
                continue
            if a.get("area", 1.0) <= 0:
                continue
            img_id_to_anns[a["image_id"]].append(a)

        items = []
        for img_id, anns in img_id_to_anns.items():
            info = img_id_to_info.get(img_id)
            if not info:
                continue
            img_path = os.path.join(self.img_root, info["file_name"])
            if not os.path.exists(img_path):
                continue

            bboxes = []
            labels = []
            for a in anns:
                x, y, w, h = a["bbox"]  # COCO 为 xywh
                if w < self.min_box_size or h < self.min_box_size:
                    continue
                x2, y2 = x + w, y + h
                if x2 <= x or y2 <= y:
                    continue
                bboxes.append([x, y, x2, y2])
                labels.append(catid2contig[a["category_id"]])
            if len(bboxes) == 0:
                continue
            items.append(
                (
                    img_path,
                    np.array(bboxes, dtype=np.float32),
                    np.array(labels, dtype=np.int64),
                )
            )

        if not items:
            raise RuntimeError(f"No valid items under: {self.img_root} with {ann_path}")
        print(
            f"[COCO-Det] Loaded {len(items)} images, {len(catid2contig)} classes."
        )
        return items, catid2contig

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        img_path, boxes_xyxy, labels = self.items[idx]

        # 读图（RGB）
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Albumentations 增强（在原始分辨率）
        if self.is_train and boxes_xyxy.size > 0:
            out = self.albu(
                image=img,
                bboxes=boxes_xyxy.tolist(),
                class_labels=labels.tolist(),
            )
            img_aug = out["image"]
            b_aug = (
                np.array(out["bboxes"], dtype=np.float32)
                if len(out["bboxes"]) > 0
                else np.zeros((0, 4), np.float32)
            )
            l_aug = (
                np.array(out["class_labels"], dtype=np.int64)
                if len(out["bboxes"]) > 0
                else np.zeros((0,), np.int64)
            )
        else:
            img_aug, b_aug, l_aug = img, boxes_xyxy, labels

        # letterbox 到 (img_size, img_size)
        img_sq, b_sq = letterbox_pad(
            img_aug, b_aug, self.img_size, pad_value=(114, 114, 114)
        )

        # 去除越界/退化框（增强后可能出现很小或空框）
        if b_sq.size > 0:
            wS = hS = self.img_size
            b_sq = clip_xyxy(b_sq, wS, hS)
            wh = (b_sq[:, 2:4] - b_sq[:, 0:2]).astype(np.float32)
            valid = (wh[:, 0] >= self.min_box_size) & (wh[:, 1] >= self.min_box_size)
            b_sq = b_sq[valid]
            l_sq = l_aug[valid] if l_aug.size > 0 else l_aug
        else:
            b_sq = b_sq.reshape(0, 4).astype(np.float32)
            l_sq = np.zeros((0,), dtype=np.int64)

        # 归一化 + ToTensor
        img_t = self.normalize_to_tensor(image=img_sq)["image"]

        # targets: [Ni,5] = [cls, x1,y1,x2,y2] (float32)
        if b_sq.size > 0:
            cls_col = l_sq.reshape(-1, 1).astype(np.float32)
            t = np.concatenate([cls_col, b_sq.astype(np.float32)], axis=1)
        else:
            t = np.zeros((0, 5), dtype=np.float32)

        targets = torch.from_numpy(t)  # [Ni,5]
        return img_t, targets, img_path


# -----------------------------
# collate（变长目标）
# -----------------------------
def collate_detection(batch):
    imgs, targets, paths = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    targets = list(targets)  # list of [Ni,5] tensors
    paths = list(paths)
    return imgs, targets, paths


# -----------------------------
# 工厂函数
# -----------------------------
def create_coco_ssd_dataloader(
    dataset_root: str,
    img_size: int,
    batch_size: int,
    is_train: bool,
    num_workers: int = 4,
    pin_memory: bool = True,
    aug_cfg: Dict[str, Any] = None
) -> DataLoader:
    """
    dataset_root: 含 images/ 与 annotations/ 的根目录（标准 COCO 结构）
    split       : "train2017" 或 "val2017"
    自动匹配 annotations/instances_{split}.json
    """
    img_dir_name = "train2017" if is_train else "val2017"
    ann_file_name = f"instances_{img_dir_name}.json"

    root_path = Path(dataset_root)
    img_root = root_path / "images" / img_dir_name
    if not img_root.is_dir():
        img_root = root_path / img_dir_name  # 兼容
    ann_path = root_path / "annotations" / ann_file_name

    if not img_root.is_dir() or not ann_path.is_file():
        raise FileNotFoundError(f"Data not found. Checked: {img_root} and {ann_path}")

    dataset = CocoDetectionDataset(
        img_root=str(img_root),
        ann_path=str(ann_path),
        img_size=img_size,
        is_train=is_train,
        aug_cfg=aug_cfg,
    )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=is_train,
        collate_fn=collate_detection,
    )

