# -*- coding: utf-8 -*-
import json, os
from typing import List, Tuple, Dict, Any, Optional
import cv2, numpy as np, torch
from torch.utils.data import Dataset
from core.datasets.common import letterbox, apply_hsv


class CocoDetDataset(Dataset):
    """
    通用 COCO detection 数据集（支持多类/子集/单类）：
    - ann_path: 建议 instances_train2017.json / instances_val2017.json
    - class_filter: 传 COCO 的 category_id 列表来筛选类（例如 [1] 仅 person）；
                    默认为 None 表示使用所有类别。
    - 返回:
        img_t:  float tensor [3,H,W], 0~1
        target: dict {"boxes": FloatTensor[n,4], "labels": LongTensor[n]}  # 归一化到 0~1（letterbox 后的 H,W）
        img_path: str
    """
    def __init__(self,
                 img_root: str,
                 ann_path: str,
                 img_size: int = 192,
                 class_filter: Optional[List[int]] = None,
                 use_color_aug: bool = True,
                 use_hflip: bool = True,
                 use_rotate: bool = True,
                 rotate_deg: float = 30.0,
                 use_scale: bool = True,
                 scale_range=(0.75, 1.25),
                 is_train: bool = True):
        super().__init__()
        self.img_root = os.path.abspath(img_root)
        self.ann_path = ann_path
        self.img_size = int(img_size)
        self.is_train = bool(is_train)

        # aug 开关
        self.use_color_aug = bool(use_color_aug)
        self.use_hflip = bool(use_hflip)
        self.use_rotate = bool(use_rotate)
        self.rotate_deg = float(rotate_deg)
        self.use_scale = bool(use_scale)
        self.scale_range = tuple(scale_range)

        # 类过滤（比如 [1] 表示 person-only）。None 表示不过滤。
        self.class_filter = sorted(set(class_filter)) if class_filter else None

        with open(self.ann_path, "r") as f:
            ann = json.load(f)

        self.imgs = {im["id"]: im for im in ann["images"]}

        if "annotations" not in ann:
            raise ValueError("COCO instances-style json required (with 'annotations').")

        # 过滤出合法 bbox & 非 crowd；如设置了 class_filter 再过滤类别
        anns = [
            a for a in ann["annotations"]
            if a.get("iscrowd", 0) == 0 and "bbox" in a and (self.class_filter is None or a["category_id"] in self.class_filter)
        ]

        # 聚合到 image_id
        imgid_to_anns: Dict[int, List[Dict[str, Any]]] = {}
        for a in anns:
            imgid_to_anns.setdefault(a["image_id"], []).append(a)

        # 生成样本列表
        self.items: List[Tuple[str, List[Dict[str, Any]]]] = []
        for img_id, alist in imgid_to_anns.items():
            info = self.imgs.get(img_id)
            if not info:
                continue
            path = os.path.join(self.img_root, info["file_name"])
            if os.path.isfile(path) and len(alist) > 0:
                self.items.append((path, alist))

        if not self.items:
            raise FileNotFoundError(f"No valid images under: {self.img_root} with {self.ann_path}")

        # 类别 id → 连续 id 的映射（背景=0 预留）
        self.cat_ids = sorted({a["category_id"] for _, L in self.items for a in L})
        self.cat2contig = {cid: i + 1 for i, cid in enumerate(self.cat_ids)}  # 1..C
        self.num_classes = 1 + len(self.cat_ids)  # 背景 + 有效类别数

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        img_path, anns = self.items[idx]
        img = cv2.imread(img_path); assert img is not None, img_path
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # 原始像素坐标的 boxes/labels
        boxes, labels = [], []
        for a in anns:
            x, y, bw, bh = a["bbox"]
            if bw <= 0 or bh <= 0:
                continue
            boxes.append([x, y, x + bw, y + bh])
            labels.append(self.cat2contig[a["category_id"]])
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        # --- 几何增强（旋转/缩放/水平翻转，与 kpts 对齐） ---
        if self.is_train:
            img, boxes, keep = self._augment_geom(img, boxes)
            if boxes.size and keep is not None:
                labels = labels[keep]

        # 颜色增强
        if self.is_train and self.use_color_aug:
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            bgr = apply_hsv(bgr, hgain=0.015, sgain=0.7, vgain=0.4)
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # letterbox 到正方形，并把框映射/归一化到 0~1
        img_lb, scale, (pad_w, pad_h) = letterbox(img, self.img_size, color=(114, 114, 114))
        if boxes.size > 0:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + pad_w
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + pad_h
            boxes[:, [0, 2]] /= float(self.img_size)
            boxes[:, [1, 3]] /= float(self.img_size)
            boxes = np.clip(boxes, 0.0, 1.0)

        img_t = torch.from_numpy(img_lb.transpose(2, 0, 1)).float() / 255.0
        target = {
            "boxes": torch.from_numpy(boxes).float(),   # [n,4], xyxy in [0,1]
            "labels": torch.from_numpy(labels).long(),  # [n], 1..C
            "path": img_path
        }
        return img_t, target

    # 几何增强：返回 img, boxes_xyxy（像素坐标）, keep（用来筛 labels）
    def _augment_geom(self, img: np.ndarray, boxes: np.ndarray):
        h, w = img.shape[:2]
        s = 1.0 if not (self.is_train and self.use_scale) else np.random.uniform(self.scale_range[0], self.scale_range[1])
        r = 0.0 if not (self.is_train and self.use_rotate) else np.random.uniform(-self.rotate_deg, self.rotate_deg)
        cx, cy = w * 0.5, h * 0.5

        T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float32)
        R = cv2.getRotationMatrix2D((0, 0), r, s)
        R = np.vstack([R, [0, 0, 1]]).astype(np.float32)
        T2 = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], dtype=np.float32)
        M = T2 @ R @ T1

        # 随机水平翻转（与 kpts 同序：放在旋转缩放之后）
        if self.is_train and self.use_hflip and (np.random.rand() < 0.5):
            Fm = np.array([[-1, 0, w], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
            M = Fm @ M

        img_out = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))

        if boxes.size == 0:
            return img_out, boxes, None

        # 透视变换四角点
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        pts = np.stack([np.stack([x1, y1], 1),
                        np.stack([x2, y1], 1),
                        np.stack([x2, y2], 1),
                        np.stack([x1, y2], 1)], axis=1).astype(np.float32)  # [N,4,2]
        pts = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), M).reshape(-1, 4, 2)
        x_min = pts[:, :, 0].min(axis=1); y_min = pts[:, :, 1].min(axis=1)
        x_max = pts[:, :, 0].max(axis=1); y_max = pts[:, :, 1].max(axis=1)
        boxes_t = np.stack([x_min, y_min, x_max, y_max], axis=1)

        # 裁剪并剔除太小的框
        boxes_t[:, [0, 2]] = np.clip(boxes_t[:, [0, 2]], 0, w - 1)
        boxes_t[:, [1, 3]] = np.clip(boxes_t[:, [1, 3]], 0, h - 1)
        wh = boxes_t[:, 2:] - boxes_t[:, :2]
        keep = (wh[:, 0] > 1) & (wh[:, 1] > 1)

        return img_out, boxes_t[keep], keep
