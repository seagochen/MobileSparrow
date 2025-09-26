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
    COCO 检测数据集（基于 Albumentations 的增强流水线）。

    功能与返回
    ----------
    - 自动完成：按最长边缩放 -> Letterbox 填充到方形 -> (可选)几何/颜色增强 -> 归一化 -> 转 Tensor。
    - 返回三元组：
        img_t   : Tensor，[3, H, W]，float，已经用 ImageNet 统计量归一化，H=W=img_size
        targets : Tensor，[N, 5]，每行为 [cls, x1, y1, x2, y2]（像素坐标，Pascal VOC 格式）
        img_path: str，图像的绝对路径

    约定与依赖
    ----------
    - 标注文件为 COCO JSON（images / annotations / categories 三段）。
    - annotation 的 bbox 源自 COCO 标准的 `xywh`，本类中在 __getitem__ 内部转换为 `xyxy`。
    - 需要以下依赖（示例）：
        import os, json, cv2, numpy as np
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        from torch.utils.data import Dataset
        # 以及一个工具函数：xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray
    - `aug_cfg` 用于控制增强强度（见 _build_transforms 的注释）。

    重要说明
    ----------
    - Albumentations 的 bbox 参数 `format='pascal_voc'`，即 `[x_min, y_min, x_max, y_max]`。
      若几何变换导致框可见度过低（min_visibility=0.1），该框会被丢弃。
    - 训练/验证模式：
        - `is_train=True` 时开启随机翻转、位移缩放旋转、色彩抖动等；`False` 时仅做 letterbox+归一化。
    - 类别 id 映射：
        - COCO 中的 category_id 可能不连续，这里将 `categories[].id` 排序后映射到 `[0..num_classes-1]`。

    """

    def __init__(self,
                 img_root: str,
                 ann_path: str,
                 img_size: int,
                 is_train: bool,
                 skip_crowd: bool = True,
                 aug_cfg: Dict[str, Any] = None):
        """
        参数
        ----------
        img_root : str
            图像根目录（COCO images 文件夹）。
        ann_path : str
            COCO 风格的标注 JSON 路径。
        img_size : int
            输出的方形目标尺寸（H=W=img_size）。LongestMaxSize+PadIfNeeded 会保证这一点。
        is_train : bool
            是否为训练模式。训练模式下会启用更强的随机增强。
        skip_crowd : bool
            是否需要过滤 crowd，默认开启
        aug_cfg : Dict[str, Any], optional
            增强配置（可选键）：
              - use_flip: bool，是否进行水平翻转（默认 True）
              - scale_range: Tuple[float,float]，A.ShiftScaleRotate 的 scale_limit（默认 (-0.25, 0.25)）
              - rotate_deg: int，A.ShiftScaleRotate 的 rotate_limit（默认 15）
              - use_color_aug: bool，是否使用颜色增强（默认 True）
              - h_gain, s_gain, v_gain: float，HSV 增强强度（默认 0.015, 0.7, 0.4）
              - min_box_size: float，增强后保留的最小框宽/高阈值（像素，默认 2.0）
        """
        super().__init__()
        self.img_root = os.path.abspath(img_root)
        self.ann_path = os.path.abspath(ann_path)
        self.img_size = int(img_size)
        self.is_train = bool(is_train)
        self.aug_cfg = aug_cfg if aug_cfg is not None else {}

        # 1) 读取 COCO 标注并建立索引（不改变原有逻辑）
        with open(self.ann_path, "r") as f:
            ann_json = json.load(f)
        images = ann_json.get("images", [])
        annotations = ann_json.get("annotations", [])
        categories = ann_json.get("categories", [])

        # 类别 id -> 连续索引（0..C-1）
        cat_ids = sorted([c["id"] for c in categories])
        self.cat_id_to_idx = {cid: i for i, cid in enumerate(cat_ids)}
        self.num_classes = len(self.cat_id_to_idx)

        # 根据 image_id 聚合标注
        self._images = {im["id"]: im for im in images}
        img_id_to_anns: Dict[int, List[Dict[str, Any]]] = {}
        for a in annotations:
            img_id_to_anns.setdefault(a["image_id"], []).append(a)

        # 生成可用数据项列表：[(绝对路径, anns), ...]
        self.items: List[Tuple[str, List[Dict[str, Any]]]] = []
        for img_id, im in self._images.items():
            file_name = im.get("file_name")
            if not file_name:
                continue
            path = os.path.join(self.img_root, file_name)
            if not os.path.isfile(path):
                continue

            # 根据图片id，获取meta信息
            anns = img_id_to_anns.get(img_id, [])

            # 过滤crowd，默认执行过滤
            if skip_crowd:
                anns = [a for a in anns if a.get("iscrowd", 0) == 0]

            # 将数据加入到list
            self.items.append((path, anns))

        if not self.items:
            raise FileNotFoundError(f"No valid images under: {self.img_root} with {self.ann_path}")

        # 2) 构建 Albumentations 增强流水线
        self.transform = self._build_transforms()

    def _build_transforms(self) -> A.Compose:
        """
        根据 `is_train` 与 `aug_cfg` 构建图像与边界框的增强流水线（Albumentations）。

        流水线（顺序执行）
        ----------
        1) A.LongestMaxSize(max_size=img_size)
           - 将最长边缩放到 img_size，保持纵横比不变
        2) A.PadIfNeeded(min_height=img_size, min_width=img_size, value=(114,114,114))
           - Letterbox：使用常见的(114,114,114) 灰色填充到方形
        3) 训练增强（可选）
           - A.HorizontalFlip(p=0.5)
           - A.ShiftScaleRotate：位移/缩放/旋转；边界外新像素用同样的灰色常量填充
           - 颜色增强：HSV 抖动 + 亮度对比度
        4) A.Normalize(mean=ImageNet, std=ImageNet)
        5) ToTensorV2()：HWC->CHW，转为 torch.Tensor

        BBox 参数
        ----------
        - format='pascal_voc'（xyxy）
        - label_fields=['class_labels']：保持 bbox 与标签的一一对应
        - min_visibility=0.1：经几何变换后，可见比例 < 0.1 的 bbox 会被丢弃
        """
        transforms = []

        # --- Letterbox 预处理（缩放 + 填充） ---
        transforms.append(A.LongestMaxSize(max_size=self.img_size))
        transforms.append(A.PadIfNeeded(
            min_height=self.img_size,
            min_width=self.img_size,
            border_mode=cv2.BORDER_CONSTANT,
            value=(114, 114, 114)
        ))

        if self.is_train:
            # --- 训练阶段的随机增强 ---
            if self.aug_cfg.get("use_flip", True):
                transforms.append(A.HorizontalFlip(p=0.5))

            transforms.append(A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=self.aug_cfg.get("scale_range", (-0.25, 0.25)),  # 约等于尺度 0.75~1.25
                rotate_limit=self.aug_cfg.get("rotate_deg", 15),
                p=0.7,
                border_mode=cv2.BORDER_CONSTANT,
                value=(114, 114, 114)
            ))

            if self.aug_cfg.get("use_color_aug", True):
                transforms.append(A.HueSaturationValue(
                    hue_shift_limit=round(180 * self.aug_cfg.get("h_gain", 0.015)),
                    sat_shift_limit=round(255 * self.aug_cfg.get("s_gain", 0.7)),
                    val_shift_limit=round(255 * self.aug_cfg.get("v_gain", 0.4)),
                    p=0.7
                ))
                transforms.append(A.RandomBrightnessContrast(p=0.5))

        # --- 归一化与张量化 ---
        transforms.append(A.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]))
        transforms.append(ToTensorV2())

        # 告诉 Albumentations 如何处理 bbox 与标签
        bbox_params = A.BboxParams(
            format='pascal_voc',
            label_fields=['class_labels'],
            min_visibility=0.1,
        )
        return A.Compose(transforms, bbox_params=bbox_params)

    def __len__(self) -> int:
        """数据集中图像的数量。"""
        return len(self.items)

    def __getitem__(self, idx: int):
        """
        读取第 idx 条数据并应用增强，返回 (img_t, targets_t, img_path)。

        返回
        ----------
        img_t : torch.Tensor
            [3, img_size, img_size]，float，已归一化（ImageNet 均值方差）。
        targets_t : torch.Tensor
            [N, 5]，每行为 [cls, x1, y1, x2, y2]（像素坐标，增强后坐标系）。
        img_path : str
            当前样本对应图像的绝对路径。
        """
        img_path, anns = self.items[idx]

        # 1) 读图（BGR->RGB）
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 2) 将 COCO `xywh` 标注转为 `xyxy`，收集标签
        boxes, labels = [], []
        for a in anns:
            if "bbox" not in a:
                continue
            xyxy = xywh_to_xyxy(np.array(a["bbox"], dtype=np.float32))
            # 跳过非法/退化框
            if xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
                continue
            boxes.append(xyxy)
            # category_id -> 连续类别索引
            labels.append(self.cat_id_to_idx.get(a["category_id"], 0))

        # 无框样本：保证 Albumentations 输入合法
        if not boxes:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.array([], dtype=np.int64)

        # 3) 通过 Albumentations 执行图像与 bbox 的同步变换
        transformed = self.transform(
            image=img,
            bboxes=boxes,
            class_labels=labels
        )
        img_t = transformed['image']  # Tensor, [3,H,W]
        transformed_boxes = np.array(transformed['bboxes'], dtype=np.float32)
        transformed_labels = np.array(transformed['class_labels'], dtype=np.int64)

        # 4) 过滤过小框并打包为 targets: [cls, x1, y1, x2, y2]
        if transformed_boxes.shape[0] > 0:
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
        aug_cfg: Dict[str, Any], 
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

    # 组装 DataLoader kwargs
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


