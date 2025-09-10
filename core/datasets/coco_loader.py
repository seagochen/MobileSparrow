from pathlib import Path
from typing import Any, Dict, Tuple
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from core.datasets.coco_kpts import CocoKeypointsDataset

# -----------------------
# 对外封装（兼容旧接口）
# -----------------------
class CoCo2017KptsDataLoader:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.img_size = int(cfg.get("img_size", 192))
        self.target_stride = int(cfg.get("target_stride", 4))
        self.batch_size = int(cfg.get("batch_size", 64))
        self.num_workers = int(cfg.get("num_workers", 8))
        self.pin_memory = bool(cfg.get("pin_memory", True))

        # 原始配置路径（通常是 ./data/coco2017）
        self.root = Path(cfg["dataset_root_path"])
        self.train_label_path = self.root / "annotations" / "person_keypoints_train2017.json"
        self.test_label_path = self.root / "annotations" / "person_keypoints_test2017.json"

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

    def _resolve_roots(self) -> Tuple[str, str]:
        """
        自动识别 train/val 子目录；兼容:
        - <root>/train2017, <root>/val2017
        - <root>/images/train2017, <root>/images/(val2017|test2017)
        - 如果都找不到，就回退到 <root>
        """
        bases = [self.root, self.root / "images"]
        for base in bases:
            tr = base / "train2017"
            va = base / "val2017"
            te = base / "test2017"
            if tr.is_dir() and va.is_dir():
                return str(tr), str(va)
            if tr.is_dir() and te.is_dir():
                return str(tr), str(te)
        # 回退：都没有明确子目录时，把根当成图像根
        return str(self.root), str(self.root)


    def getTrainValDataloader(self) -> Tuple[DataLoader, DataLoader]:
        train_img_root, val_img_root = self._resolve_roots()

        train_set = CocoKeypointsDataset(
            img_root=train_img_root,
            ann_path=self.train_label_path,
            img_size=self.img_size,
            target_stride=self.target_stride,
            is_train=True,
            **self.aug,
        )
        val_set = CocoKeypointsDataset(
            img_root=val_img_root,
            ann_path=self.test_label_path,
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

        # 健壮性检查
        if len(train_set) == 0:
            raise ValueError(
                f"Train dataset is empty. Check paths:\n"
                f"  img_root={train_img_root}\n"
                f"  ann_path={self.train_label_path}\n"
                f"以及 COCO JSON 里的 file_name 与实际文件是否匹配。"
            )
        if len(val_set) == 0:
            raise ValueError(
                f"Val dataset is empty. Check paths:\n"
                f"  img_root={val_img_root}\n"
                f"  ann_path={self.test_label_path}\n"
                f"以及 COCO JSON 里的 file_name 与实际文件是否匹配。"
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

        print(f"[INFO] Train root: {train_img_root}")
        print(f"[INFO] Val   root: {val_img_root}")
        print(f"[INFO] Total train images: {len(train_set)}, val images: {len(val_set)}")
        return train_loader, val_loader

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