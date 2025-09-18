# -*- coding: utf-8 -*-
from __future__ import annotations
"""
ReID dataloader builder：
- Dataset：读取 final_pairs.csv（img_path, person_id, image_id），将 person_id 连续化到 [0..C-1]
- PK 采样（RandomIdentitySampler）：每 batch 采样 P 个 ID，每 ID 采 K 张
- ReID 友好增强（训练/验证两套），尺寸默认 256x128
- create_reid_dataloader(builder)：返回 DataLoader，供 sparrow_cli 通过 YAML 构建

YAML 用法：
  data:
    train:
      builder: reid_dataloader
      args:
        csv_path: data/reid/final_pairs.csv
        root_dir: data/reid
        batch_size: 64
        num_instances: 4
        num_workers: 8
        img_h: 256
        img_w: 128
    val:
      builder: reid_dataloader
      args:
        csv_path: data/reid/final_pairs.csv
        root_dir: data/reid
        batch_size: 64
        num_instances: 4
        num_workers: 8
        img_h: 256
        img_w: 128
说明：
- 与 sparrow_cli 的数据段约定对齐（builder + args），可一键被 CLI 解析/实例化使用。:contentReference[oaicite:7]{index=7}
"""

import os
import math
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms as T


# =========================
# Dataset：CSV + 连续标签
# =========================
class ReIDPairsDataset(Dataset):
    """
    期望 CSV 列: img_path, person_id, image_id
    - 将 person_id -> label 映射到 [0..C-1]
    - root_dir 与相对路径拼接
    """
    def __init__(self, csv_path: str, root_dir: Optional[str] = None, transform=None):
        self.df = pd.read_csv(csv_path)
        assert {'img_path', 'person_id', 'image_id'}.issubset(self.df.columns)

        uniq_pids = sorted(self.df['person_id'].unique().tolist())
        self.pid2label = {pid: i for i, pid in enumerate(uniq_pids)}
        self.label2pid = {i: pid for pid, i in self.pid2label.items()}
        self.num_classes = len(self.pid2label)

        self.root_dir = root_dir
        self.transform = transform

        self.label_to_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, row in self.df.iterrows():
            y = self.pid2label[row['person_id']]
            self.label_to_indices[y].append(idx)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        rel = row['img_path']
        path = os.path.join(self.root_dir, rel) if (self.root_dir and not os.path.isabs(rel)) else rel
        img = Image.open(path).convert('RGB')
        y = self.pid2label[row['person_id']]
        if self.transform:
            img = self.transform(img)
        # 返回 (img, label, path) 便于调试
        return img, torch.tensor(y, dtype=torch.long), path


# =========================
# PK 采样（P×K）
# =========================
class RandomIdentitySampler(Sampler[List[int]]):
    """
    每个 batch：P 个 ID × 每 ID K 张
    - 保证 batch-hard Triplet 需要的“同类对”
    """
    def __init__(self, dataset: ReIDPairsDataset, batch_size: int, num_instances: int, drop_last: bool = True):
        assert batch_size % num_instances == 0, "batch_size 必须能被 num_instances 整除"
        self.dataset = dataset
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // num_instances
        self.labels = list(self.dataset.label_to_indices.keys())
        self.drop_last = drop_last
        self._pools: Dict[int, List[int]] = {}

    def __iter__(self):
        labels = self.labels.copy()
        random.shuffle(labels)
        batch = []
        for pid in labels:
            pool = self._pools.get(pid)
            if not pool or len(pool) < self.num_instances:
                pool = self.dataset.label_to_indices[pid].copy()
                random.shuffle(pool)
                if len(pool) < self.num_instances:
                    pool = (pool * math.ceil(self.num_instances / max(1, len(pool))))[:self.num_instances]
                self._pools[pid] = pool
            inds = [pool.pop() for _ in range(self.num_instances)]
            batch.extend(inds)
            if len(batch) == self.num_pids_per_batch * self.num_instances:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        return len(self.dataset) // (self.num_pids_per_batch * self.num_instances)


# =========================
# Transforms：ReID 友好增强
# =========================
def _build_reid_transforms(img_h=256, img_w=128, is_train=True):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    if is_train:
        return T.Compose([
            T.Resize((img_h, img_w), interpolation=T.InterpolationMode.BILINEAR),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
            T.RandomGrayscale(p=0.1),
            T.RandomPerspective(distortion_scale=0.2, p=0.2),
            T.Pad(10, padding_mode='edge'),
            T.RandomCrop((img_h, img_w)),
            T.ToTensor(),
            normalize,
            T.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random'),
        ])
    else:
        return T.Compose([
            T.Resize((img_h, img_w), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            normalize,
        ])


# =========================
# Builder：给 sparrow_cli 用
# =========================
def reid_dataloader(*,
                    csv_path: str,
                    root_dir: str,
                    batch_size: int = 64,
                    num_instances: int = 4,
                    num_workers: int = 4,
                    img_h: int = 256,
                    img_w: int = 128,
                    is_train: bool = True):
    """
    统一入口（builder），与 sparrow_cli 的数据构建协议一致：接收 args -> 返回 DataLoader
    - 训练返回基于 PK-sampler 的 DataLoader
    - 验证/评估：若 is_train=False，则常规顺序采样
    """
    tf = _build_reid_transforms(img_h, img_w, is_train=is_train)
    ds = ReIDPairsDataset(csv_path, root_dir=root_dir, transform=tf)

    if is_train:
        sampler = RandomIdentitySampler(ds, batch_size=batch_size, num_instances=num_instances, drop_last=True)
        loader = DataLoader(ds, batch_sampler=sampler, num_workers=num_workers, pin_memory=True)
    else:
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return loader
