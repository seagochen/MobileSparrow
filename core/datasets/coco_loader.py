from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List
import torch
from torch.utils.data import DataLoader

from core.datasets.coco_kpts import CocoKeypointsDataset
from core.datasets.coco_det import CocoDetDataset
from core.datasets.coco_cls import CocoClsDataset

# ---------- det 专用 collate（可变数目标） ----------
def det_collate_fn(batch):
    imgs, targets = zip(*batch)
    return torch.stack(imgs, 0), list(targets)

# -----------------------
# 统一对外封装
# -----------------------
class CoCo2017DataLoader:
    def __init__(self, cfg: Dict[str, Any],
                 task: str = "kpts",
                 cls_mode: str = "single_label"):
        """
        task: 'kpts' | 'det' | 'cls'
        cls_mode: 'single_label' | 'multi_label'（仅 task='cls' 有效）
        """
        assert task in ("kpts", "det", "cls")
        assert cls_mode in ("single_label", "multi_label")
        self.cfg = cfg
        self.task = task
        self.cls_mode = cls_mode

        self.img_size = int(cfg.get("img_size", 192))
        self.target_stride = int(cfg.get("target_stride", 4))
        self.batch_size = int(cfg.get("batch_size", 64))
        self.num_workers = int(cfg.get("num_workers", 8))
        self.pin_memory = bool(cfg.get("pin_memory", True))

        # 根路径（通常是 ./data/coco2017 或数据准备脚本的 out-dir）
        self.root = Path(cfg["dataset_root_path"])

        # 任务参数（可选）
        tp = cfg.get("task_params", {}) if isinstance(cfg.get("task_params", {}), dict) else {}
        self.class_filter: Optional[List[int]] = tp.get("class_filter", None) if self.task == "det" else None

        # 增广配置（与 kpts 对齐）
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

        # 运行后填充，供外部读取（det/cls 会写入具体类别数；kpts 为 None）
        self.num_classes: Optional[int] = None

    def _resolve_roots(self) -> Tuple[str, str]:
        """
        仅接受官方/准备脚本导出的 train2017 + val2017 目录；兼容：
        - <root>/train2017, <root>/val2017
        - <root>/images/train2017, <root>/images/val2017
        """
        bases = [self.root, self.root / "images"]
        for base in bases:
            tr = base / "train2017"
            va = base / "val2017"
            if tr.is_dir() and va.is_dir():
                return str(tr), str(va)
        raise FileNotFoundError(
            f"Expected 'train2017' and 'val2017' under {self.root} or {self.root/'images'}; "
            f"please run the data prep script to create them."
        )

    def _ann_paths(self) -> Tuple[Path, Path]:
        ann_dir = self.root / "annotations"
        if self.task == "kpts":
            train = ann_dir / "person_keypoints_train2017.json"
            val   = ann_dir / "person_keypoints_val2017.json"
        else:
            train = ann_dir / "instances_train2017.json"
            val   = ann_dir / "instances_val2017.json"
        return train, val

    def getTrainValDataloader(self) -> Tuple[DataLoader, DataLoader]:
        train_img_root, val_img_root = self._resolve_roots()
        train_ann, val_ann = self._ann_paths()

        if self.task == "kpts":
            train_set = CocoKeypointsDataset(
                img_root=train_img_root,
                ann_path=train_ann,
                img_size=self.img_size,
                target_stride=self.target_stride,
                is_train=True,
                **self.aug,
            )
            val_set = CocoKeypointsDataset(
                img_root=val_img_root,
                ann_path=val_ann,
                img_size=self.img_size,
                target_stride=self.target_stride,
                is_train=False,
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
            self.num_classes = None
            collate_fn = None

        elif self.task == "det":
            train_set = CocoDetDataset(
                img_root=train_img_root,
                ann_path=str(train_ann),
                img_size=self.img_size,
                class_filter=self.class_filter,                 # 新增：透传类过滤（如 [1] 仅 person）
                is_train=True,
                use_color_aug=self.aug["use_color_aug"],
                use_hflip=self.aug["use_flip"],
                use_rotate=self.aug["use_rotate"],
                rotate_deg=self.aug["rotate_deg"],
                use_scale=self.aug["use_scale"],
                scale_range=self.aug["scale_range"],
            )
            val_set = CocoDetDataset(
                img_root=val_img_root,
                ann_path=str(val_ann),
                img_size=self.img_size,
                class_filter=self.class_filter,                 # 新增：验证与训练类空间一致
                is_train=False,
                use_color_aug=False,
                use_hflip=False,
                use_rotate=False,
                rotate_deg=0.0,
                use_scale=False,
                scale_range=(1.0, 1.0),
            )
            # 背景 + N 类（优先读数据集自带属性，回退到 cat_ids 推断）
            self.num_classes = getattr(train_set, "num_classes", None)
            if self.num_classes is None:
                self.num_classes = 1 + len(getattr(train_set, "cat_ids", []))
            collate_fn = det_collate_fn

        else:  # self.task == "cls"
            train_set = CocoClsDataset(
                img_root=train_img_root,
                ann_path=str(train_ann),
                img_size=self.img_size,
                mode=self.cls_mode,
                is_train=True,
                use_color_aug=self.aug["use_color_aug"],
                use_hflip=self.aug["use_flip"],
            )
            val_set = CocoClsDataset(
                img_root=val_img_root,
                ann_path=str(val_ann),
                img_size=self.img_size,
                mode=self.cls_mode,
                is_train=False,
                use_color_aug=False,
                use_hflip=False,
            )
            self.num_classes = int(train_set.num_classes)  # 0..C-1
            collate_fn = None

        # 健壮性检查
        if len(train_set) == 0:
            raise ValueError(f"Train dataset is empty. Check paths:\n"
                             f"  img_root={train_img_root}\n  ann_path={train_ann}")
        if len(val_set) == 0:
            raise ValueError(f"Val dataset is empty. Check paths:\n"
                             f"  img_root={val_img_root}\n  ann_path={val_ann}")

        train_loader = DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
            drop_last=True, collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_set, batch_size=min(self.batch_size, 64), shuffle=False,
            num_workers=max(1, self.num_workers // 2), pin_memory=self.pin_memory,
            drop_last=False, collate_fn=collate_fn
        )

        print(f"[INFO] Task: {self.task} ({self.cls_mode if self.task=='cls' else ''})")
        print(f"[INFO] Train root: {train_img_root}")
        print(f"[INFO] Val   root: {val_img_root}")
        print(f"[INFO] Total train images: {len(train_set)}, val images: {len(val_set)}")
        if self.num_classes is not None:
            print(f"[INFO] num_classes: {self.num_classes}")
        return train_loader, val_loader
