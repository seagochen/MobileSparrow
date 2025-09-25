# %% [markdown]
# # 加载COCO数据集
# 
# 为了便于测试我们的 `SSDLite` 和 `SSDLite_FPN`，我们使用标准的 `COCO` 数据集。但为了加载这个数据集，并应用于我们的模型里，需要一些特别的设计。

# %%
# !pip -q install albumentations

# %% [markdown]
# ## 导入依赖包

# %%
import json, os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

# %% [markdown]
# ## 辅助工具函数

# %% [markdown]
# ### `xywh_to_xyxy`

# %%
def xywh_to_xyxy(box_xywh: np.ndarray) -> np.ndarray:
    x, y, w, h = box_xywh
    return np.array([x, y, x + w, y + h], dtype=np.float32)

# %% [markdown]
# ## 数据集加载类

# %%
class CocoDetDataset(Dataset):
    """
    COCO 检测数据集 (使用 Albumentations 重构)
    - 自动处理 Letterbox, 归一化, 和复杂的数据增强
    - 返回:
        - img_t:   [3,H,W] (float, 已归一化)
        - targets: Tensor [N, 5] -> [cls, x1, y1, x2, y2]
        - img_path: str
    """
    def __init__(self,
                 img_root: str,
                 ann_path: str,
                 img_size: int,
                 is_train: bool,
                 aug_cfg: Dict[str, Any] = None):
        super().__init__()
        self.img_root = os.path.abspath(img_root)
        self.ann_path = os.path.abspath(ann_path)
        self.img_size = int(img_size)
        self.is_train = bool(is_train)
        self.aug_cfg = aug_cfg if aug_cfg is not None else {}

        # 1. 读取和解析 COCO 标注 (与之前相同)
        with open(self.ann_path, "r") as f:
            ann_json = json.load(f)
        images = ann_json.get("images", [])
        annotations = ann_json.get("annotations", [])
        categories = ann_json.get("categories", [])

        cat_ids = sorted([c["id"] for c in categories])
        self.cat_id_to_idx = {cid: i for i, cid in enumerate(cat_ids)}
        self.num_classes = len(self.cat_id_to_idx)

        self._images = {im["id"]: im for im in images}
        img_id_to_anns: Dict[int, List[Dict[str, Any]]] = {}
        for a in annotations:
            img_id_to_anns.setdefault(a["image_id"], []).append(a)

        self.items: List[Tuple[str, List[Dict[str, Any]]]] = []
        for img_id, im in self._images.items():
            file_name = im.get("file_name")
            if not file_name: continue
            path = os.path.join(self.img_root, file_name)
            if not os.path.isfile(path): continue
            
            anns = img_id_to_anns.get(img_id, [])
            if self.is_train:
                anns = [a for a in anns if a.get("iscrowd", 0) == 0]
            self.items.append((path, anns))

        if not self.items:
            raise FileNotFoundError(f"No valid images under: {self.img_root} with {self.ann_path}")

        # 2. <--- 核心改动：构建 Albumentations 增强管道 --->
        self.transform = self._build_transforms()

    def _build_transforms(self) -> A.Compose:
        """根据训练/验证模式和配置构建增强管道"""
        transforms = []
        # --- Letterbox 填充 ---
        # 1. 先将图片最长边缩放到 img_size
        transforms.append(A.LongestMaxSize(max_size=self.img_size))
        # 2. 再用 (114,114,114) 填充到 img_size x img_size
        transforms.append(A.PadIfNeeded(
            min_height=self.img_size,
            min_width=self.img_size,
            border_mode=cv2.BORDER_CONSTANT,
            value=(114, 114, 114)
        ))

        if self.is_train:
            # --- 训练时使用的增强 ---
            if self.aug_cfg.get("use_flip", True):
                transforms.append(A.HorizontalFlip(p=0.5))
            
            # 几何变换 (旋转, 缩放, 错切)
            transforms.append(A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=self.aug_cfg.get("scale_range", (-0.25, 0.25)), # (e.g., 0.75-1.25)
                rotate_limit=self.aug_cfg.get("rotate_deg", 15),
                p=0.7,
                border_mode=cv2.BORDER_CONSTANT,
                value=(114, 114, 114)
            ))
            
            # 颜色增强
            if self.aug_cfg.get("use_color_aug", True):
                transforms.append(A.HueSaturationValue(
                    hue_shift_limit=round(180 * self.aug_cfg.get("h_gain", 0.015)),
                    sat_shift_limit=round(255 * self.aug_cfg.get("s_gain", 0.7)),
                    val_shift_limit=round(255 * self.aug_cfg.get("v_gain", 0.4)),
                    p=0.7
                ))
                transforms.append(A.RandomBrightnessContrast(p=0.5))

        # --- 归一化和转换为 Tensor ---
        # 1. 像素值归一化 (使用ImageNet统计数据)
        transforms.append(A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        # 2. HWC -> CHW 并转为 torch.Tensor
        transforms.append(ToTensorV2())

        # Bbox参数告诉albumentations如何处理边界框
        bbox_params = A.BboxParams(
            format='pascal_voc', # [x_min, y_min, x_max, y_max]
            label_fields=['class_labels'],
            min_visibility=0.1, # 框被裁切后可见度低于该值则丢弃
        )
        return A.Compose(transforms, bbox_params=bbox_params)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        img_path, anns = self.items[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 1. 准备原始框和标签
        boxes, labels = [], []
        for a in anns:
            if "bbox" not in a: continue
            xyxy = xywh_to_xyxy(np.array(a["bbox"], dtype=np.float32))
            if xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]: continue
            
            boxes.append(xyxy)
            labels.append(self.cat_id_to_idx.get(a["category_id"], 0))
        
        # 如果没有标注框，创建一个空的
        if not boxes:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.array([], dtype=np.int64)
        
        # 2. <--- 核心改动：应用增强管道 --->
        transformed = self.transform(
            image=img,
            bboxes=boxes,
            class_labels=labels
        )
        
        img_t = transformed['image']
        transformed_boxes = np.array(transformed['bboxes'], dtype=np.float32)
        transformed_labels = np.array(transformed['class_labels'], dtype=np.int64)
        
        # 3. 打包 targets: [cls, x1, y1, x2, y2]
        if transformed_boxes.shape[0] > 0:
            # 过滤掉增强后可能产生的过小框
            min_box_size = self.aug_cfg.get("min_box_size", 2.0)
            valid_indices = (transformed_boxes[:, 2] - transformed_boxes[:, 0] >= min_box_size) & \
                            (transformed_boxes[:, 3] - transformed_boxes[:, 1] >= min_box_size)
            
            final_boxes = transformed_boxes[valid_indices]
            final_labels = transformed_labels[valid_indices]
            
            cls = final_labels.reshape(-1, 1).astype(np.float32)
            targets = np.concatenate([cls, final_boxes], axis=1)
        else:
            targets = np.zeros((0, 5), dtype=np.float32)

        targets_t = torch.from_numpy(targets).float()
        return img_t, targets_t, img_path

# %% [markdown]
# ## DataLoader 工厂方法

# %% [markdown]
# ### 工厂方法辅助函数 `det_collate_fn`

# %%
def det_collate_fn(batch):
    imgs, targets, paths = zip(*batch)
    imgs = torch.stack(imgs, 0)
    return imgs, list(targets), list(paths)

# %% [markdown]
# ### 工厂方法 `create_dets_dataloader`

# %%
def create_dets_dataloader(
        dataset_root: str,
        img_size: int,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        aug_cfg: Dict[str, Any], # <--- 现在这个配置更重要
        is_train: bool
) -> DataLoader:
    img_dir_name = "train2017" if is_train else "val2017"
    ann_file_name = "instances_train2017.json" if is_train else "instances_val2017.json"

    root_path = Path(dataset_root)
    img_root = root_path / "images" / img_dir_name
    if not img_root.is_dir():
        img_root = root_path / img_dir_name
    ann_path = root_path / "annotations" / ann_file_name

    if not img_root.is_dir() or not ann_path.is_file():
        raise FileNotFoundError(f"Data not found. Checked: {img_root} and {ann_path}")

    dataset = CocoDetDataset(
        img_root=str(img_root),
        ann_path=str(ann_path),
        img_size=img_size,
        is_train=is_train,
        aug_cfg=aug_cfg,
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

# %% [markdown]
# ## 测试：如何使用新的 Dataloader

# %%
if __name__ == "__main__":
    # 假设你的 COCO 数据集在 './coco' 目录下
    # 你需要下载 COCO 2017 train/val images and annotations
    DATASET_ROOT = "/home/user/projects/MobileSparrow/data/coco2017" 
    
    if not os.path.exists(DATASET_ROOT):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"!! 请将 COCO 2017 数据集解压到 '{DATASET_ROOT}' 目录下 !!")
        print("!! 目录结构应为:                                        !!")
        print("!!   ./coco/annotations/instances_train2017.json      !!")
        print("!!   ./coco/annotations/instances_val2017.json        !!")
        print("!!   ./coco/train2017/<很多图片>                       !!")
        print("!!   ./coco/val2017/<很多图片>                         !!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        # 1. 定义增强配置 (现在可以从外部文件加载，如 YAML)
        train_aug_config = {
            "use_flip": True,
            "use_color_aug": True,
            "rotate_deg": 20.0,
            "scale_range": (-0.3, 0.3), # 对应 0.7-1.3 的缩放
            "h_gain": 0.015,
            "s_gain": 0.7,
            "v_gain": 0.4,
            "min_box_size": 4.0
        }

        # 2. 创建训练 Dataloader
        train_loader = create_dets_dataloader(
            dataset_root=DATASET_ROOT,
            img_size=320,
            batch_size=4,
            num_workers=2,
            pin_memory=True,
            aug_cfg=train_aug_config,
            is_train=True
        )
        
        # 3. 取一个批次的数据进行测试
        print("--- Testing Dataloader ---")
        imgs, targets, paths = next(iter(train_loader))

        print(f"Images batch shape: {imgs.shape}")
        print(f"Images tensor dtype: {imgs.dtype}")
        print(f"Images tensor value range: [{imgs.min():.2f}, {imgs.max():.2f}] (已归一化)")
        print(f"Targets is a list of {len(targets)} tensors.")
        for i, t in enumerate(targets):
            print(f"  - Target {i} shape: {t.shape}") # [N_i, 5]
        print(f"Paths is a list of {len(paths)} strings. First path: {paths[0]}")
        
        # 检查类别数
        num_classes_from_dataset = train_loader.dataset.num_classes
        print(f"\nDetected {num_classes_from_dataset} classes from COCO annotations (e.g., 80 for COCO).")


