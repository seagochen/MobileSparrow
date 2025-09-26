# %% [markdown]
# # 加载COCO数据集
# 
# 为了便于测试我们的 `MoveNet`，我们使用标准的 `COCO` 数据集。但为了加载这个数据集，并应用于我们的模型里，需要一些特别的设计。

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
# ### `draw_gaussian`

# %%
def draw_gaussian(heatmap: np.ndarray, center: Tuple[int, int], radius: int, k: float = 1.0):
    """在 heatmap 上画带裁剪的高斯核。heatmap: H×W；center: (x, y)"""

    def gaussian2d(shape: Tuple[int, int], sigma: float) -> np.ndarray:
        h, w = shape
        y = np.arange(0, h, 1, dtype=np.float32)
        x = np.arange(0, w, 1, dtype=np.float32)
        yy, xx = np.meshgrid(y, x, indexing="ij")
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2))
        return g

    diameter = 2 * radius + 1
    gaussian = gaussian2d((diameter, diameter), sigma=diameter / 6.0)

    x, y = int(center[0]), int(center[1])
    h, w = heatmap.shape

    left, right = min(x, radius), min(w - x, radius + 1)
    top, bottom = min(y, radius), min(h - y, radius + 1)

    if right <= 0 or bottom <= 0 or left <= 0 or top <= 0:
        return

    masked_hm = heatmap[y - top:y + bottom, x - left:x + right]
    masked_g = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    np.maximum(masked_hm, masked_g * k, out=masked_hm)

# %% [markdown]
# ### `get_center_from_kps`

# %%
def get_center_from_kps(kps_xyv: np.ndarray) -> Tuple[float, float]:
    """kps_xyv: [17,3]，只用 v>0 的点做均值"""
    vis = kps_xyv[:, 2] > 0
    if vis.sum() == 0:
        cx = kps_xyv[:, 0].mean()
        cy = kps_xyv[:, 1].mean()
    else:
        cx = kps_xyv[vis, 0].mean()
        cy = kps_xyv[vis, 1].mean()
    return float(cx), float(cy)

# %% [markdown]
# ## 数据集加载类

# %%
class CocoKeypointsDataset(Dataset):
    """
    COCO 姿态估计数据集（内聚版）。
    - 使用 Albumentations 进行数据增强
    - 将每个 image 中的每个 person 展开成一条独立样本
    - 统一完成 letterbox、以及 supervision 编码
    - 输出: (img_tensor [3,H,W], label_tensor [86,Hf,Wf], kps_mask [17], img_path)
    """
    
    # COCO 17 点左右翻转索引对
    FLIP_PAIRS = [
        (1, 2), (3, 4), (5, 6), (7, 8),
        (9, 10), (11, 12), (13, 14), (15, 16)
    ]
    
    def __init__(self,
                 img_root: str,
                 ann_path: str,
                 img_size: int,
                 target_stride: int,
                 is_train: bool,
                 skip_crowd: bool = True,
                 aug_cfg: Dict[str, Any] = None):
        super().__init__()
        self.img_root = os.path.abspath(img_root)
        self.ann_path = ann_path
        self.img_size = int(img_size)
        self.stride = int(target_stride)
        self.is_train = bool(is_train)
        self.aug_cfg = aug_cfg if aug_cfg is not None else {}
        self.skip_crowd = bool(skip_crowd)
        
        assert self.img_size % self.stride == 0, \
            f"img_size({self.img_size}) must be divisible by stride({self.stride})"
        self.Hf = self.img_size // self.stride
        self.Wf = self.img_size // self.stride

        # --- 加载与预处理标注 (与原版相同) ---
        self._ann_images: Dict[int, Dict[str, Any]] = {}
        self.items = self._load_annotations()

        # --- 核心改动：构建 Albumentations 增强管道 ---
        self.transform = self._build_transforms()

    def _build_transforms(self) -> A.Compose:
        """根据训练/验证模式和配置构建增强管道"""
        transforms = []
        # Letterbox 填充
        transforms.append(A.LongestMaxSize(max_size=self.img_size))
        transforms.append(A.PadIfNeeded(
            min_height=self.img_size,
            min_width=self.img_size,
            border_mode=cv2.BORDER_CONSTANT,
            value=(114, 114, 114)
        ))

        if self.is_train:
            # 训练时使用的增强
            if self.aug_cfg.get("use_flip", True):
                transforms.append(A.HorizontalFlip(p=0.5))
            
            # 几何变换 (旋转, 缩放)
            transforms.append(A.ShiftScaleRotate(
                shift_limit=0, # 平移由后续的仿射处理
                scale_limit=self.aug_cfg.get("scale_range", (-0.25, 0.25)), # (e.g., 0.75-1.25)
                rotate_limit=self.aug_cfg.get("rotate_deg", 30),
                p=0.7,
                border_mode=cv2.BORDER_CONSTANT,
                value=(114, 114, 114)
            ))
            
            # 颜色增强
            if self.aug_cfg.get("use_color_aug", True):
                transforms.append(A.HueSaturationValue(p=0.5))
                transforms.append(A.RandomBrightnessContrast(p=0.5))
        
        # 归一化和转换为 Tensor
        transforms.append(A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        transforms.append(ToTensorV2())
        
        # !!! 关键：定义如何处理关键点 !!!
        keypoint_params = A.KeypointParams(
            format='xy', # 输入格式是 (x, y)
            label_fields=['keypoint_labels'], # 翻转时需要交换的标签
            remove_invisible=False # 保留变换后超出图像范围的点
        )
        return A.Compose(transforms, keypoint_params=keypoint_params)

    # _load_annotations 和 _encode_targets 方法可以保持不变 (从原文件复制)
    def _load_annotations(self) -> List[Tuple[str, Dict[str, Any]]]:
        with open(self.ann_path, "r") as f:
            ann_json = json.load(f)

        images = ann_json.get("images", [])
        annotations = ann_json.get("annotations", [])
        categories = ann_json.get("categories", [])

        self._ann_images = {im["id"]: im for im in images}
        person_id = next((c.get("id") for c in categories if c.get("name") == "person"), 1)
        anns_all = [a for a in annotations if a.get("category_id", person_id) == person_id]

        # 过滤数据
        if self.skip_crowd:
            anns_all = [a for a in anns_all if a.get("iscrowd", 0) == 0 and a.get("num_keypoints", 0) > 0]

        img_id_to_anns: Dict[int, List[Dict[str, Any]]] = {}
        for a in anns_all:
            img_id_to_anns.setdefault(a["image_id"], []).append(a)

        items: List[Tuple[str, Dict[str, Any]]] = []
        for img_id, anns in img_id_to_anns.items():
            info = self._ann_images.get(img_id)
            if not info: continue
            file_name = info.get("file_name")
            if not file_name: continue
            path = os.path.abspath(os.path.join(self.img_root, file_name))
            if not os.path.isfile(path): continue
            for person_ann in anns:
                items.append((path, person_ann))

        if not items:
            raise FileNotFoundError(f"No valid items found under: {self.img_root} with {self.ann_path}")
        return items

    def _encode_targets(self, kps_xyv: np.ndarray, center_xy: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # 这个函数逻辑不变，直接从原文件复制即可
        J = 17
        heatmaps = np.zeros((J, self.Hf, self.Wf), dtype=np.float32)
        centers = np.zeros((1, self.Hf, self.Wf), dtype=np.float32)
        regs = np.zeros((2 * J, self.Hf, self.Wf), dtype=np.float32)
        offsets = np.zeros((2 * J, self.Hf, self.Wf), dtype=np.float32)
        kps_mask = np.zeros((J,), dtype=np.float32)
        kps_f = kps_xyv.copy()
        kps_f[:, :2] /= self.stride
        vis = (kps_f[:, 2] > 0)
        if np.any(vis):
            xs, ys = kps_f[vis, 0], kps_f[vis, 1]
            w_box, h_box = float(xs.max() - xs.min()), float(ys.max() - ys.min())
            side = max(1.0, max(w_box, h_box))
        else:
            side = float(max(self.Wf, self.Hf)) / 4.0
        r_kpt = max(1, int(round(0.025 * side)))
        r_ctr = max(2, int(round(0.035 * side)))
        for j in range(J):
            if kps_f[j, 2] > 0:
                xj, yj = kps_f[j, 0].item(), kps_f[j, 1].item()
                if 0 <= xj < self.Wf and 0 <= yj < self.Hf:
                    draw_gaussian(heatmaps[j], (int(round(xj)), int(round(yj))), r_kpt)
                    kps_mask[j] = 1.0
        cx, cy = center_xy[0] / self.stride, center_xy[1] / self.stride
        if 0 <= cx < self.Wf and 0 <= cy < self.Hf:
            draw_gaussian(centers[0], (int(round(cx)), int(round(cy))), r_ctr)
        cx_i, cy_i = int(np.clip(np.floor(cx + 0.5), 0, self.Wf - 1)), int(np.clip(np.floor(cy + 0.5), 0, self.Hf - 1))
        for j in range(J):
            if kps_mask[j] > 0:
                regs[2*j:2*j+2, cy_i, cx_i] = kps_f[j, :2] - np.array([cx, cy])
        for j in range(J):
            if kps_mask[j] > 0:
                xj, yj = kps_f[j, :2]
                gx, gy = int(np.clip(np.floor(xj+0.5),0,self.Wf-1)), int(np.clip(np.floor(yj+0.5),0,self.Hf-1))
                offsets[2*j:2*j+2, gy, gx] = np.array([xj-gx, yj-gy])
        return heatmaps, centers, regs, offsets, kps_mask

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        img_path, person = self.items[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 准备原始关键点和标签
        kps_all = np.array(person["keypoints"], dtype=np.float32).reshape(-1, 3)
        kps_xy = kps_all[:, :2]
        kps_v = kps_all[:, 2]
        
        # 准备用于翻转时交换的标签
        keypoint_labels = np.arange(17)
        if self.is_train and self.aug_cfg.get("use_flip", True):
            # albumentations 的 HorizontalFlip 需要知道哪些标签对需要交换
            # 我们通过修改标签，然后反向映射来实现
            keypoint_labels_flipped = keypoint_labels.copy()
            for a, b in self.FLIP_PAIRS:
                keypoint_labels_flipped[a] = b
                keypoint_labels_flipped[b] = a
        
        # 应用增强
        transformed = self.transform(
            image=img,
            keypoints=kps_xy,
            keypoint_labels=keypoint_labels if not (self.is_train and self.aug_cfg.get("use_flip", True)) else keypoint_labels_flipped
        )
        
        img_t = transformed['image']
        kps_mapped_xy = np.array(transformed['keypoints'], dtype=np.float32)

        # 重新组合 kps
        if kps_mapped_xy.shape[0] != 17:
             # 如果增强后所有点都出界了，补一个空的
             kps_mapped_xy = np.zeros((17, 2), dtype=np.float32)

        # 如果发生了翻转，需要把坐标和可见性换回来
        if len(transformed['keypoint_labels']) > 0 and not np.array_equal(transformed['keypoint_labels'], keypoint_labels):
            remap_indices = np.argsort(transformed['keypoint_labels'])
            kps_mapped_xy = kps_mapped_xy[remap_indices]
            kps_v = kps_v[remap_indices]

        kps_mapped = np.hstack([kps_mapped_xy, kps_v[:, np.newaxis]])

        # 中心点计算 (逻辑不变)
        cx, cy = get_center_from_kps(kps_mapped)
        if not (np.isfinite(cx) and np.isfinite(cy)):
            cx = cy = float(self.img_size * 0.5)

        # 生成 supervision (逻辑不变)
        heatmaps, centers, regs, offsets, kps_mask = self._encode_targets(kps_mapped, (cx, cy))
        label = np.concatenate([heatmaps, centers, regs, offsets], axis=0)
        
        label_t = torch.from_numpy(label).float()
        kps_mask_t = torch.from_numpy(kps_mask).float()

        return img_t, label_t, kps_mask_t, img_path

# %% [markdown]
# ## DataLoader 工厂方法

# %%
def create_kpts_dataloader(
        dataset_root: str,
        img_size: int,
        batch_size: int,
        target_stride: int,
        num_workers: int,
        pin_memory: bool,
        aug_cfg: Dict[str, Any],
        is_train: bool
) -> DataLoader:
    """
    工厂函数：创建姿态估计 DataLoader。
    - 兼容 <root>/train2017 和 <root>/images/train2017 两种结构
    - 训练集支持完整增广；验证集默认关闭大部分增广，并采用固定半径（更稳定的评估）
    """
    img_dir_name = "train2017" if is_train else "val2017"
    ann_file_name = "person_keypoints_train2017.json" if is_train else "person_keypoints_val2017.json"

    root_path = Path(dataset_root)
    img_root = root_path / "images" / img_dir_name
    if not img_root.is_dir():
        img_root = root_path / img_dir_name
    ann_path = root_path / "annotations" / ann_file_name

    if not img_root.is_dir() or not ann_path.is_file():
        raise FileNotFoundError(f"Data not found. Checked: {img_root} and {ann_path}")

    dataset = CocoKeypointsDataset(
        img_root=str(img_root),
        ann_path=str(ann_path),
        img_size=img_size,
        target_stride=target_stride,
        is_train=is_train,
        aug_cfg=aug_cfg
    )

    # 组装 DataLoader kwargs
    dl_kwargs: Dict[str, Any] = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=is_train,
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
    DATASET_ROOT = "/home/user/projects/MobileSparrow/data/coco2017_movenet" 
    
    if not os.path.exists(DATASET_ROOT):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"!! 请将 COCO 2017 数据集解压到 '{DATASET_ROOT}' 目录下 !!")
        print("!! 目录结构应为:                                        !!")
        print("!!   ./coco/annotations/person_keypoints_train2017.json      !!")
        print("!!   ./coco/annotations/person_keypoints_val2017.json        !!")
        print("!!   ./coco/train2017/<很多图片>                       !!")
        print("!!   ./coco/val2017/<很多图片>                         !!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        # 1. 定义增强配置
        train_aug_config = {
            "use_flip": True,
            "use_color_aug": True,
            "rotate_deg": 20.0,
            "scale_range": (-0.3, 0.3), # 对应 0.7-1.3 的缩放
        }

        # 2. 创建训练 Dataloader
        train_loader = create_kpts_dataloader(
            dataset_root=DATASET_ROOT,
            img_size=192,
            batch_size=4,
            target_stride=4, # 注意 stride 应该和你的模型输出 stride 匹配
            num_workers=2,
            pin_memory=True,
            aug_cfg=train_aug_config,
            is_train=True
        )
        
        # 3. 取一个批次的数据进行测试
        print("--- Testing Dataloader ---")
        
        # --- 核心修正点在这里 ---
        # 接收全部4个返回值
        imgs, labels, kps_masks, paths = next(iter(train_loader))

        # --- 更新 print 语句以匹配新的输出格式 ---
        print(f"Images batch shape: {imgs.shape}")
        print(f"Images tensor dtype: {imgs.dtype}")
        print(f"Images tensor value range: [{imgs.min():.2f}, {imgs.max():.2f}] (已归一化)")
        
        # labels 是一个 [B, 86, Hf, Wf] 的张量, 而不是列表
        print(f"Labels batch shape: {labels.shape}")
        
        # kps_masks 是一个 [B, 17] 的张量
        print(f"Keypoint masks batch shape: {kps_masks.shape}")

        print(f"Paths is a list of {len(paths)} strings. First path: {paths[0]}")
        
        print("\n Dataloader test successful!")


