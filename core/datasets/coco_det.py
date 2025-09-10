# -*- coding: utf-8 -*-
import json, os
from typing import List, Tuple, Dict, Any
import cv2, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from core.datasets.common import letterbox, apply_hsv


class CocoDetDataset(Dataset):
    """
    COCO detection 版数据集。
    - ann_path: 建议用 instances_train2017.json / instances_val2017.json
      若只做 person 单类，也可用 keypoints json，但会把所有实例当成 person(1)。
    - 返回:
        img_t:  float tensor [3,H,W], 0~1
        target: dict {"boxes": FloatTensor[n,4], "labels": LongTensor[n]}  # 归一化到0~1(相对 letterbox 后的 H,W)
        img_path: str
    """
    def __init__(self,
                 img_root: str,
                 ann_path: str,
                 img_size: int = 192,
                 use_color_aug: bool = True,
                 use_hflip: bool = True,
                 is_train: bool = True,
                 only_person: bool = False):
        super().__init__()
        self.img_root = os.path.abspath(img_root)
        self.ann_path = ann_path
        self.img_size = int(img_size)
        self.is_train = bool(is_train)
        self.use_color_aug = bool(use_color_aug)
        self.use_hflip = bool(use_hflip)
        self.only_person = bool(only_person)

        with open(self.ann_path, "r") as f:
            ann = json.load(f)

        self.imgs = {im["id"]: im for im in ann["images"]}

        if "annotations" not in ann:
            raise ValueError("COCO instances-style json required (with 'annotations').")

        anns = ann["annotations"]
        if self.only_person:
            anns = [a for a in anns if a.get("category_id", 1) == 1]  # 1==person in COCO

        # 聚合到 image_id
        self.items: List[Tuple[str, List[Dict[str, Any]]]] = []
        imgid_to_anns: Dict[int, List[Dict[str, Any]]] = {}
        for a in anns:
            if a.get("iscrowd", 0) == 1:
                continue
            # bbox: [x,y,w,h] in original image pixels
            if "bbox" not in a: 
                continue
            imgid_to_anns.setdefault(a["image_id"], []).append(a)

        for img_id, alist in imgid_to_anns.items():
            info = self.imgs.get(img_id)
            if not info: 
                continue
            path = os.path.join(self.img_root, info["file_name"])
            if os.path.isfile(path) and len(alist) > 0:
                self.items.append((path, alist))

        if not self.items:
            raise FileNotFoundError(f"No valid images under: {self.img_root} with {self.ann_path}")

        # 类别 id → 连续 id 的映射（如用 instances json）
        self.cat_ids = sorted(list({a["category_id"] for _, L in self.items for a in L}))
        if self.only_person:
            self.cat2contig = {1: 1}  # 背景=0, person=1
        else:
            self.cat2contig = {cid: i + 1 for i, cid in enumerate(self.cat_ids)}  # 背景=0

    def __len__(self):
        return len(self.items)

    def _flip_boxes(self, boxes_xyxy: np.ndarray, w: int) -> np.ndarray:
        # 水平翻转
        if boxes_xyxy.size == 0: 
            return boxes_xyxy
        x1, y1, x2, y2 = boxes_xyxy.T
        nx1 = w - 1 - x2
        nx2 = w - 1 - x1
        return np.stack([nx1, y1, nx2, y2], axis=1)

    def __getitem__(self, idx: int):
        img_path, anns = self.items[idx]
        img = cv2.imread(img_path); assert img is not None, img_path
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        boxes = []
        labels = []
        for a in anns:
            x, y, bw, bh = a["bbox"]
            if bw <= 0 or bh <= 0: 
                continue
            boxes.append([x, y, x + bw, y + bh])
            lab = self.cat2contig[1 if self.only_person else a["category_id"]]
            labels.append(lab)
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        # aug（轻量：hflip + 颜色）
        if self.is_train and self.use_hflip and np.random.rand() < 0.5:
            img = cv2.flip(img, 1)
            boxes = self._flip_boxes(boxes, w)

        if self.is_train and self.use_color_aug:
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            bgr = apply_hsv(bgr, hgain=0.015, sgain=0.7, vgain=0.4)
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # letterbox 到正方形
        img_lb, scale, (pad_w, pad_h) = letterbox(img, self.img_size, color=(114, 114, 114))
        # 映射框坐标到 letterbox 后像素坐标
        if boxes.size > 0:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + pad_w
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + pad_h
            # 归一化到 0~1（相对 H,W）
            boxes[:, [0, 2]] /= float(self.img_size)
            boxes[:, [1, 3]] /= float(self.img_size)
            boxes = np.clip(boxes, 0.0, 1.0)

        img_t = torch.from_numpy(img_lb.transpose(2, 0, 1)).float() / 255.0
        target = {
            "boxes": torch.from_numpy(boxes).float(),   # [n,4] (x1,y1,x2,y2) normalized
            "labels": torch.from_numpy(labels).long(),  # [n]
            "path": img_path
        }
        return img_t, target


def det_collate_fn(batch):
    imgs, targets = zip(*batch)
    imgs = torch.stack(imgs, 0)
    return imgs, list(targets)


def create_coco_det_loaders(cfg: Dict[str, Any],
                            only_person: bool = False) -> Tuple[DataLoader, DataLoader, int]:
    from pathlib import Path
    root = Path(cfg["dataset_root_path"])
    train_root = str(root / "train2017") if (root / "train2017").is_dir() else str(root)
    val_root   = str(root / "val2017")   if (root / "val2017").is_dir()   else str(root)

    train_set = CocoDetDataset(train_root, cfg["train_label_path"], img_size=cfg["img_size"], 
                               is_train=True, only_person=only_person)
    val_set   = CocoDetDataset(val_root, cfg["val_label_path"],   img_size=cfg["img_size"], 
                               is_train=False, only_person=only_person)

    bs = int(cfg.get("batch_size", 64))
    nw = int(cfg.get("num_workers", 8))
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=nw,
                              pin_memory=bool(cfg.get("pin_memory", True)), drop_last=True,
                              collate_fn=det_collate_fn)
    val_loader = DataLoader(val_set, batch_size=min(bs, 64), shuffle=False, num_workers=max(1, nw//2),
                            pin_memory=bool(cfg.get("pin_memory", True)), drop_last=False,
                            collate_fn=det_collate_fn)
    # 返回类别数（背景+N）
    num_classes = (1 if only_person else len(train_set.cat_ids)) + 1
    return train_loader, val_loader, num_classes
