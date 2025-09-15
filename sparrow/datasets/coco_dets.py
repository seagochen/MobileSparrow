# sparrow/datasets/coco_dets.py
# -*- coding: utf-8 -*-
"""
COCO 目标检测数据集（简洁版）。
- 逐图片（image-level）产出：img_tensor, targets(N x 5: [cls,x1,y1,x2,y2]), img_path
- 几何/颜色/letterbox 处理与 coco_kpts 风格一致，便于同工程复用。
"""
from __future__ import annotations
import json, os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# 依赖与 coco_kpts 相同的工具函数（你工程里已有）
from sparrow.datasets.common import apply_hsv, letterbox, random_affine_points


# ---------- 小工具 ----------
def _xywh_to_xyxy(box_xywh: np.ndarray) -> np.ndarray:
    x, y, w, h = box_xywh
    return np.array([x, y, x + w, y + h], dtype=np.float32)

def _xyxy_to_corners(box_xyxy: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box_xyxy
    return np.array([[x1, y1],
                     [x2, y1],
                     [x2, y2],
                     [x1, y2]], dtype=np.float32)

def _corners_to_xyxy(pts: np.ndarray) -> np.ndarray:
    xs = pts[:, 0]; ys = pts[:, 1]
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)

def _clip_box_xyxy(box: np.ndarray, w: int, h: int) -> np.ndarray:
    x1, y1, x2, y2 = box
    x1 = float(np.clip(x1, 0, w - 1))
    y1 = float(np.clip(y1, 0, h - 1))
    x2 = float(np.clip(x2, 0, w - 1))
    y2 = float(np.clip(y2, 0, h - 1))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


# ---------- 数据集 ----------
class CocoDetDataset(Dataset):
    """
    COCO 检测数据集：逐图片产出。
    返回：
      - img_t:  [3,H,W] (float, 0~1)
      - targets: Tensor [N, 5] -> [cls, x1, y1, x2, y2] (像素坐标，已 letterbox 到 img_size)
      - img_path: str
    说明：
      - 训练时过滤 iscrowd/无效框；验证集尽量保留但仍会剔除无效框。
      - 类别 id 重映射为 0..C-1 的连续索引；background 隐式存在于损失中（后续环节处理）。
    """
    def __init__(self,
                 img_root: str,
                 ann_path: str,
                 img_size: int,
                 is_train: bool,
                 *,
                 use_color_aug: bool = True,
                 use_flip: bool = True,
                 use_rotate: bool = True,
                 rotate_deg: float = 15.0,
                 use_scale: bool = True,
                 scale_range: Tuple[float, float] = (0.75, 1.25),
                 min_box_size: float = 2.0):
        super().__init__()
        self.img_root = os.path.abspath(img_root)
        self.ann_path = os.path.abspath(ann_path)
        self.img_size = int(img_size)
        self.is_train = bool(is_train)

        self.use_color_aug = bool(use_color_aug)
        self.use_flip = bool(use_flip)
        self.use_rotate = bool(use_rotate)
        self.rotate_deg = float(rotate_deg)
        self.use_scale = bool(use_scale)
        self.scale_range = tuple(scale_range)
        self.min_box = float(min_box_size)

        # 读取 COCO 标注
        with open(self.ann_path, "r") as f:
            ann_json = json.load(f)
        images = ann_json.get("images", [])
        annotations = ann_json.get("annotations", [])
        categories = ann_json.get("categories", [])

        # 类别 id 连续化映射（0..C-1）
        cat_ids = sorted([c["id"] for c in categories])
        self.cat_id_to_idx = {cid: i for i, cid in enumerate(cat_ids)}
        self.idx_to_cat_id = {i: cid for cid, i in self.cat_id_to_idx.items()}
        self.num_classes = len(self.cat_id_to_idx)

        self._images = {im["id"]: im for im in images}
        img_id_to_anns: Dict[int, List[Dict[str, Any]]] = {}
        for a in annotations:
            img_id_to_anns.setdefault(a["image_id"], []).append(a)

        # 逐图片组装 items
        self.items: List[Tuple[str, List[Dict[str, Any]]]] = []
        for img_id, im in self._images.items():
            file_name = im.get("file_name")
            if not file_name:
                continue
            path = os.path.join(self.img_root, file_name)
            if not os.path.isfile(path):
                continue
            anns = img_id_to_anns.get(img_id, [])
            if self.is_train:
                # 训练：过滤 crowd
                anns = [a for a in anns if a.get("iscrowd", 0) == 0]
            self.items.append((path, anns))

        if not self.items:
            raise FileNotFoundError(f"No valid images under: {self.img_root} with {self.ann_path}")

    def __len__(self) -> int:
        return len(self.items)

    # --- 增广 & 变换 ---
    def _augment_geom(self, img: np.ndarray, boxes_xyxy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """仿射（缩放/旋转/水平翻转）与 coco_kpts 保持一致的矩阵流程。"""
        h, w = img.shape[:2]
        # 随机尺度
        s = 1.0
        if self.is_train and self.use_scale:
            s = float(np.random.uniform(self.scale_range[0], self.scale_range[1]))
        # 随机旋转
        r = 0.0
        if self.is_train and self.use_rotate:
            r = float(np.random.uniform(-self.rotate_deg, self.rotate_deg))

        # 以图像中心为原点构建仿射: 平移->旋转缩放->平移回去
        cx, cy = w * 0.5, h * 0.5
        T1 = np.array([[1, 0, -cx],
                       [0, 1, -cy],
                       [0, 0, 1]], dtype=np.float32)
        R = cv2.getRotationMatrix2D((0, 0), r, s)
        R = np.vstack([R, [0, 0, 1]]).astype(np.float32)
        T2 = np.array([[1, 0, cx],
                       [0, 1, cy],
                       [0, 0, 1]], dtype=np.float32)
        M = T2 @ R @ T1  # 3x3

        # 随机水平翻转
        if self.is_train and self.use_flip and (np.random.rand() < 0.5):
            F = np.array([[-1, 0, w],
                          [0, 1, 0],
                          [0, 0, 1]], dtype=np.float32)
            M = F @ M

        # 应用到图像
        A2 = M[:2, :]
        img = cv2.warpAffine(img, A2, (w, h), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))

        # 应用到框：角点->仿射->重组
        if boxes_xyxy.size > 0:
            new_boxes = []
            for b in boxes_xyxy:
                pts = _xyxy_to_corners(b)           # 4x2
                pts = random_affine_points(pts, A2) # 4x2
                b_new = _corners_to_xyxy(pts)
                b_new = _clip_box_xyxy(b_new, w, h)
                new_boxes.append(b_new)
            boxes_xyxy = np.stack(new_boxes, axis=0) if new_boxes else np.zeros((0, 4), np.float32)

        return img, boxes_xyxy

    def __getitem__(self, idx: int):
        img_path, anns = self.items[idx]
        img = cv2.imread(img_path)
        assert img is not None, f"fail to read image: {img_path}"
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 原始框（xyxy）与标签
        boxes = []
        labels = []
        for a in anns:
            if "bbox" not in a:
                continue
            xyxy = _xywh_to_xyxy(np.array(a["bbox"], dtype=np.float32))
            x1, y1, x2, y2 = xyxy
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append(xyxy)
            labels.append(self.cat_id_to_idx.get(a["category_id"], 0))
        boxes = np.array(boxes, dtype=np.float32) if len(boxes) else np.zeros((0, 4), dtype=np.float32)
        labels = np.array(labels, dtype=np.int64) if len(labels) else np.zeros((0,), dtype=np.int64)

        # 几何增广
        if self.is_train:
            img, boxes = self._augment_geom(img, boxes)

        # 颜色增强（与 kpts 相同的 HSV 抖动）
        if self.is_train and self.use_color_aug:
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            bgr = apply_hsv(bgr, hgain=0.015, sgain=0.7, vgain=0.4)
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # letterbox 到正方形
        img_lb, scale, (pad_w, pad_h) = letterbox(img, self.img_size, color=(114, 114, 114))

        # 映射框到 letterbox 后坐标 + 过滤小框（同步过滤 labels）
        if boxes.size > 0:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + pad_w
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + pad_h

            H, W = img_lb.shape[:2]
            new_boxes, new_labels = [], []
            for b, lab in zip(boxes, labels):
                b = _clip_box_xyxy(b, W, H)
                bw = b[2] - b[0]
                bh = b[3] - b[1]
                if bw >= self.min_box and bh >= self.min_box:
                    new_boxes.append(b)
                    new_labels.append(lab)
            if new_boxes:
                boxes = np.stack(new_boxes, 0)
                labels = np.array(new_labels, dtype=np.int64)
            else:
                boxes = np.zeros((0, 4), dtype=np.float32)
                labels = np.zeros((0,), dtype=np.int64)

        # 打包 targets: [cls, x1, y1, x2, y2]
        if boxes.size > 0:
            cls = labels.reshape(-1, 1).astype(np.float32)
            targets = np.concatenate([cls, boxes.astype(np.float32)], axis=1)
        else:
            targets = np.zeros((0, 5), dtype=np.float32)

        # 转 tensor
        img_t = torch.from_numpy(img_lb.transpose(2, 0, 1)).float() / 255.0
        targets_t = torch.from_numpy(targets).float()

        return img_t, targets_t, img_path


# ---------- DataLoader 辅助 ----------
def det_collate_fn(batch):
    """
    自定义 collate：
    - imgs: [B,3,H,W]
    - targets: List[Tensor[N_i, 5]]
    - paths: List[str]
    """
    imgs, targets, paths = zip(*batch)
    imgs = torch.stack(imgs, 0)
    return imgs, list(targets), list(paths)


def create_dets_dataloader(
        dataset_root: str,
        img_size: int,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        aug_cfg: Dict[str, Any],
        is_train: bool
) -> DataLoader:
    """
    工厂：创建 COCO 检测 DataLoader。
    - 兼容 <root>/train2017 和 <root>/images/train2017 结构
    - 训练集默认开启增广；验证集默认关闭大部分增广
    """
    img_dir_name = "train2017" if is_train else "val2017"
    ann_file_name = "instances_train2017.json" if is_train else "instances_val2017.json"

    root_path = Path(dataset_root)
    img_root = root_path / "images" / img_dir_name
    if not img_root.is_dir():
        img_root = root_path / img_dir_name
    ann_path = root_path / "annotations" / ann_file_name

    if not img_root.is_dir() or not ann_path.is_file():
        raise FileNotFoundError(f"Data not found. Checked: {img_root} and {ann_path}")

    valid_keys = {"use_color_aug", "use_flip", "use_rotate", "rotate_deg", "use_scale", "scale_range", "min_box_size"}
    safe_aug = {k: v for k, v in (aug_cfg or {}).items() if k in valid_keys}

    if is_train:
        dataset = CocoDetDataset(
            img_root=str(img_root),
            ann_path=str(ann_path),
            img_size=img_size,
            is_train=True,
            **safe_aug
        )
    else:
        dataset = CocoDetDataset(
            img_root=str(img_root),
            ann_path=str(ann_path),
            img_size=img_size,
            is_train=False,
            use_color_aug=False,
            use_flip=False,
            use_rotate=False,
            use_scale=False,
            **{k: safe_aug[k] for k in safe_aug if k in {"min_box_size"}}
        )

    dl_kwargs: Dict[str, Any] = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=is_train,
        collate_fn=det_collate_fn,
    )
    if num_workers > 0:
        dl_kwargs.update(persistent_workers=bool(is_train))
        dl_kwargs.update(prefetch_factor=2)
        dl_kwargs.update(worker_init_fn=lambda wid: np.random.seed(torch.initial_seed() % 2**32))

    return DataLoader(**dl_kwargs)
