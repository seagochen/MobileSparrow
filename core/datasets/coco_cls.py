# -*- coding: utf-8 -*-
import json, os
from typing import List, Dict, Any, Tuple
import numpy as np, cv2, torch
from torch.utils.data import Dataset
from core.datasets.common import letterbox, apply_hsv

class CocoClsDataset(Dataset):
    """
    从 COCO instances 标注衍生分类数据：
      - single_label: 为每张图选一个“主类”（面积之和最大的类）
      - multi_label : 为每张图做 multi-hot（出现过的类为1）
    类别空间为 ann["categories"] 的非背景连续映射（背景不占槽位）。
    返回：
      img_t:  [3,H,W] 0~1
      label:  LongTensor() (single_label) 或 FloatTensor[C] (multi_label)
      path:   str
    """
    def __init__(self, img_root: str, 
                 ann_path: str, 
                 img_size: int = 192,
                 mode: str = "single_label", 
                 use_color_aug: bool = True, 
                 use_hflip: bool = True,
                 is_train: bool = True):
        super().__init__()
        assert mode in ("single_label", "multi_label")
        self.img_root = os.path.abspath(img_root)
        self.ann_path = ann_path
        self.img_size = int(img_size)
        self.mode = mode
        self.is_train = bool(is_train)
        self.use_color_aug = bool(use_color_aug)
        self.use_hflip = bool(use_hflip)

        with open(self.ann_path, "r") as f:
            ann = json.load(f)
        self.imgs = {im["id"]: im for im in ann["images"]}
        self.cats = {c["id"]: c["name"] for c in ann["categories"]}
        self.cat_ids_sorted = sorted(list(self.cats.keys()))
        self.cat2contig = {cid: i for i, cid in enumerate(self.cat_ids_sorted)}  # 0..C-1

        imgid_to_anns: Dict[int, List[Dict[str, Any]]] = {}
        for a in ann["annotations"]:
            if a.get("iscrowd", 0) == 1 or "bbox" not in a: 
                continue
            imgid_to_anns.setdefault(a["image_id"], []).append(a)

        self.items: List[Tuple[str, List[Dict[str, Any]]]] = []
        for img_id, alist in imgid_to_anns.items():
            info = self.imgs.get(img_id); 
            if not info: continue
            path = os.path.join(self.img_root, info["file_name"])
            if os.path.isfile(path) and len(alist) > 0:
                self.items.append((path, alist))
        if not self.items:
            raise FileNotFoundError(f"No valid images under: {self.img_root}")

        self.num_classes = len(self.cat_ids_sorted)

    def __len__(self): return len(self.items)

    def _build_label(self, anns: List[Dict[str, Any]]):
        if self.mode == "multi_label":
            vec = np.zeros((self.num_classes,), dtype=np.float32)
            for a in anns:
                vec[self.cat2contig[a["category_id"]]] = 1.0
            return torch.from_numpy(vec)
        else:
            # single_label: 选择面积和最大的类
            area_sum: Dict[int, float] = {}
            for a in anns:
                x,y,w,h = a["bbox"]; area_sum[a["category_id"]] = area_sum.get(a["category_id"], 0.0) + float(w*h)
            main_cid = max(area_sum.items(), key=lambda kv: kv[1])[0]
            return torch.tensor(self.cat2contig[main_cid], dtype=torch.long)

    def __getitem__(self, idx: int):
        img_path, anns = self.items[idx]
        img = cv2.imread(img_path); assert img is not None, img_path
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.is_train and self.use_hflip and np.random.rand() < 0.5:
            img = cv2.flip(img, 1)
        if self.is_train and self.use_color_aug:
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            bgr = apply_hsv(bgr, hgain=0.015, sgain=0.7, vgain=0.4)
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        img_lb, _, _ = letterbox(img, self.img_size, color=(114,114,114))
        img_t = torch.from_numpy(img_lb.transpose(2,0,1)).float()/255.0
        label = self._build_label(anns)
        return img_t, label, img_path
