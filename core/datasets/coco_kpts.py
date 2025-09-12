# core/datasets/coco_kpts.py
# -*- coding: utf-8 -*-
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from core.datasets.common import (random_affine_points, draw_gaussian,
                                  apply_hsv, letterbox, get_center_from_kps)


class CocoKeypointsDataset(Dataset):
    """
    COCO姿态估计数据集 (内聚版本)。
    负责加载图像和标注，并应用所有必要的变换。
    """

    def __init__(self,
                 img_root: str,
                 ann_path: str,
                 img_size: int,
                 target_stride: int,
                 is_train: bool,
                 *,  # 强制后面的参数为关键字参数
                 use_dynamic_radius: bool = True,
                 kpt_radius_factor: float = 0.025,
                 ctr_radius_factor: float = 0.035,
                 min_radius: int = 1,
                 use_color_aug: bool = True,
                 use_flip: bool = True,
                 use_rotate: bool = True,
                 rotate_deg: float = 30.0,
                 use_scale: bool = True,
                 scale_range: Tuple[float, float] = (0.75, 1.25)):
        super().__init__()
        # --- 核心属性 ---
        self.img_root = os.path.abspath(img_root)
        self.ann_path = ann_path
        self.img_size = img_size
        self.stride = target_stride
        self.Hf = self.img_size // self.stride
        self.Wf = self.img_size // self.stride
        self.is_train = is_train

        # --- 增广和目标生成参数 ---
        self.use_dynamic_radius = use_dynamic_radius
        self.kpt_radius_factor = kpt_radius_factor
        self.ctr_radius_factor = ctr_radius_factor
        self.min_radius = min_radius
        self.use_color_aug = use_color_aug
        self.use_flip = use_flip
        self.use_rotate = use_rotate
        self.rotate_deg = rotate_deg
        self.use_scale = use_scale
        self.scale_range = scale_range

        # --- 加载和预处理标注 ---
        self.items = self._load_annotations()
        self.imgid_to_info = self._get_img_info_dict()

    def _load_annotations(self) -> List[Tuple[str, Dict[str, Any]]]:
        # ... (这里粘贴您 coco_kpts.py 中完整的标注加载和处理逻辑) ...
        # ... (即从 with open(ann_path) 到 for person_ann in anns: self.items.append(...) 的所有代码) ...
        with open(self.ann_path, "r") as f:
            ann_json = json.load(f)

        person_id = next((c["id"] for c in ann_json.get("categories", []) if c.get("name") == "person"), 1)
        anns_all = [a for a in ann_json["annotations"] if a.get("category_id", person_id) == person_id]
        if self.is_train:
            anns_all = [a for a in anns_all if a.get("iscrowd", 0) == 0 and a.get("num_keypoints", 0) > 0]

        imgid_to_info = {im["id"]: im for im in ann_json["images"]}
        imgid_to_anns = {}
        for a in anns_all:
            imgid_to_anns.setdefault(a["image_id"], []).append(a)

        items = []
        for img_id, anns in imgid_to_anns.items():
            info = imgid_to_info.get(img_id)
            if info and info.get("file_name"):
                path = os.path.abspath(os.path.join(self.img_root, info["file_name"]))
                if os.path.isfile(path):
                    for person_ann in anns:
                        items.append((path, person_ann))

        if not items:
            raise FileNotFoundError(f"No valid items found under: {self.img_root} with {self.ann_path}")
        return items

    def _get_img_info_dict(self) -> Dict[int, Dict[str, Any]]:
        with open(self.ann_path, "r") as f:
            ann_json = json.load(f)
        return {im["id"]: im for im in ann_json["images"]}

    def __len__(self):
        return len(self.items)

    def _augment_geom(self, img: np.ndarray, kps_xyv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        在 letterbox 前做随机仿射：缩放、旋转、水平翻转（对 kps 同步仿射）
        注意：这里的仿射在**原始坐标系**做，后续再 letterbox。
        """
        h, w = img.shape[:2]
        M = np.eye(3, dtype=np.float32)

        # 随机尺度
        s = 1.0
        if self.is_train and self.use_scale:
            s = np.random.uniform(self.scale_range[0], self.scale_range[1])

        # 随机旋转
        r = 0.0
        if self.is_train and self.use_rotate:
            r = np.random.uniform(-self.rotate_deg, self.rotate_deg)

        # 以图像中心为原点构建仿射
        cx, cy = w * 0.5, h * 0.5
        # 先平移 -> 旋转 -> 缩放 -> 平移回去
        T1 = np.array([[1, 0, -cx],
                       [0, 1, -cy],
                       [0, 0, 1]], dtype=np.float32)
        R = cv2.getRotationMatrix2D((0, 0), r, s)  # 2x3
        R = np.vstack([R, [0, 0, 1]]).astype(np.float32)
        T2 = np.array([[1, 0, cx],
                       [0, 1, cy],
                       [0, 0, 1]], dtype=np.float32)
        M = T2 @ R @ T1  # 3x3

        # 水平翻转
        if self.is_train and self.use_flip and (np.random.rand() < 0.5):
            F = np.array([[-1, 0, w],
                          [0, 1, 0],
                          [0, 0, 1]], dtype=np.float32)
            M = F @ M

            # 同时需要对关键点做对称索引交换（COCO 左右翻转），这里给出一对常见映射：
            flip_pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
            for a, b in flip_pairs:
                kps_xyv[[a, b]] = kps_xyv[[b, a]]

        # 应用仿射
        img = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
        kps_xyv[:, :2] = random_affine_points(kps_xyv[:, :2], M[:2, :])

        return img, kps_xyv

    def _encode_targets(self,
                        kps_xyv: np.ndarray,
                        center_xy: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        输入：kps_xyv（letterbox 后、单位像素），center_xy（letterbox 后）
        输出特征图上的：
        - heatmaps: [17,Hf,Wf]
        - center:   [1,Hf,Wf]
        - regs:     [2*17,Hf,Wf]   (仅中心网格位置有效)
        - offsets:  [2*17,Hf,Wf]   (仅各关键点所在网格位置有效)
        - kps_mask: [17]   (可见=1，不可见=0)
        """
        J = 17
        heatmaps = np.zeros((J, self.Hf, self.Wf), dtype=np.float32)
        centers  = np.zeros((1, self.Hf, self.Wf), dtype=np.float32)
        regs     = np.zeros((2 * J, self.Hf, self.Wf), dtype=np.float32)
        offsets  = np.zeros((2 * J, self.Hf, self.Wf), dtype=np.float32)
        kps_mask = np.zeros((J,), dtype=np.float32)

        # 关键点映射到特征图坐标
        kps_f = kps_xyv.copy()
        kps_f[:, 0] = kps_f[:, 0] / self.stride
        kps_f[:, 1] = kps_f[:, 1] / self.stride

        # <新增 2025/09/11> 根据可见关键点的外接框来估算人尺度
        vis = (kps_f[:, 2] > 0)
        if np.any(vis):
            xs, ys = kps_f[vis, 0], kps_f[vis, 1]
            w_box = float(xs.max() - xs.min())
            h_box = float(ys.max() - ys.min())
            side = max(1.0, max(w_box, h_box))  # 至少为1，避免0除
        else:
            side = float(max(self.Wf, self.Hf)) / 4.0  # 兜底

        # 计算关键点与中心的半径
        if self.use_dynamic_radius:
            r_kpt = max(self.min_radius, int(round(self.kpt_radius_factor * side)))
            r_ctr = max(self.min_radius + 1, int(round(self.ctr_radius_factor * side)))
        else:
            r_kpt = int(self.gr)
            r_ctr = int(self.gr + 1)

        # # 画 heatmaps
        # for j in range(J):
        #     v = kps_f[j, 2]
        #     if v > 0:
        #         xj, yj = kps_f[j, 0], kps_f[j, 1]
        #         if xj >= 0 and yj >= 0 and xj < self.Wf and yj < self.Hf:
        #             draw_gaussian(heatmaps[j], (int(round(xj)), int(round(yj))), self.gr, k=self.sigma_scale)
        #             kps_mask[j] = 1.0

        # <修改 2025/09/11> 使用 r_kpt 画 heatmaps
        for j in range(J):
            v = kps_f[j, 2]
            if v > 0:
                xj, yj = kps_f[j, 0], kps_f[j, 1]
                if 0 <= xj < self.Wf and 0 <= yj < self.Hf:
                    draw_gaussian(heatmaps[j], (int(round(xj)), int(round(yj))), r_kpt, k=self.sigma_scale)
                    kps_mask[j] = 1.0

        # # 画 center（用 center_xy）
        # cx = center_xy[0] / self.stride
        # cy = center_xy[1] / self.stride
        # if 0 <= cx < self.Wf and 0 <= cy < self.Hf:
        #     draw_gaussian(centers[0], (int(round(cx)), int(round(cy))), self.gr + 1, k=self.sigma_scale)

        # <修改 2025/09/11> 使用 r_crt 画 center
        cx = center_xy[0] / self.stride
        cy = center_xy[1] / self.stride
        if 0 <= cx < self.Wf and 0 <= cy < self.Hf:
            draw_gaussian(centers[0], (int(round(cx)), int(round(cy))), r_ctr, k=self.sigma_scale)

        # 计算 regs（以中心网格为锚，写 dx,dy 的整数位）
        cx_i = int(np.clip(np.floor(cx + 0.5), 0, self.Wf - 1))
        cy_i = int(np.clip(np.floor(cy + 0.5), 0, self.Hf - 1))
        for j in range(J):
            if kps_mask[j] > 0:
                dx = kps_f[j, 0] - cx
                dy = kps_f[j, 1] - cy
                # regs[2 * j,     cy_i, cx_i] = float(np.floor(dx))
                # regs[2 * j + 1, cy_i, cx_i] = float(np.floor(dy))
                regs[2 * j,     cy_i, cx_i] = float(dx)
                regs[2 * j + 1, cy_i, cx_i] = float(dy)

        # 计算 offsets（在每个关键点所在网格记录小数偏移）
        for j in range(J):
            if kps_mask[j] > 0:
                xj, yj = kps_f[j, 0], kps_f[j, 1]
                gx = int(np.clip(np.floor(xj + 0.5), 0, self.Wf - 1))
                gy = int(np.clip(np.floor(yj + 0.5), 0, self.Hf - 1))
                ox = xj - gx
                oy = yj - gy
                offsets[2 * j,     gy, gx] = float(ox)
                offsets[2 * j + 1, gy, gx] = float(oy)

        return heatmaps, centers, regs, offsets, kps_mask

    def __getitem__(self, idx: int):
        # img_path, anns = self.items[idx]
        # img = cv2.imread(img_path)
        # assert img is not None, f"fail to read image: {img_path}"
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #
        # # —— 选择一个 person 做“中心”的回归锚（其 keypoints 参与 regs/offset 的中心采样）——
        # if len(anns) == 1:
        #     person = anns[0]
        # else:
        #     if self.select_person == "largest":
        #         person = max(anns, key=lambda a: bbox_area(a.get("bbox", [0, 0, 0, 0])))
        #     else:
        #         person = anns[np.random.randint(0, len(anns))]
        #
        # # 取该人的 keypoints（17*3: x,y,v）
        # kp = np.array(person["keypoints"], dtype=np.float32).reshape(-1, 3)  # [17,3], 原图坐标

        # --- 核心修改：直接获取图片路径和单个person的标注 ---
        img_path, person = self.items[idx]  # <-- person 现在是单个标注字典
        img = cv2.imread(img_path)
        assert img is not None, f"fail to read image: {img_path}"
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # --- 移除整个 "选择一个 person" 的逻辑块 ---

        # 直接使用传入的 person 的 keypoints
        kp = np.array(person["keypoints"], dtype=np.float32).reshape(-1, 3)  # [17,3], 原图坐标

        # 可选：把所有人 heatmap 叠加（常见做法增强监督），但中心 & regs/offset 只用被选的人
        all_kps = [kp]
        # for a in anns:
        #     if a is not person:
        #         all_kps.append(np.array(a["keypoints"], dtype=np.float32).reshape(-1, 3))

        # —— 先做几何增广（原图坐标系）——
        if self.is_train:
            img, kp = self._augment_geom(img, kp)

        # 简化：只保留被选中者，避免错配
        all_kps = [kp]

        # —— 颜色增广 ——
        if self.is_train and self.use_color_aug:
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            bgr = apply_hsv(bgr, hgain=0.015, sgain=0.7, vgain=0.4)
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # —— letterbox 到正方形输入 ——
        img_lb, scale, (pad_w, pad_h) = letterbox(img, self.img_size, color=(114, 114, 114))

        # 把关键点映射到 letterbox 后坐标
        def map_kps_to_letterbox(kps_xyv: np.ndarray) -> np.ndarray:
            out = kps_xyv.copy()
            out[:, 0] = out[:, 0] * scale + pad_w
            out[:, 1] = out[:, 1] * scale + pad_h
            return out

        mapped_all = [map_kps_to_letterbox(k) for k in all_kps]
        # 选择“中心”坐标（被选中者）——用可见关键点的均值
        cx, cy = get_center_from_kps(mapped_all[0])

        # —— 生成 supervision（特征图尺度）——
        heatmaps, centers, regs, offsets, kps_mask = self._encode_targets(mapped_all[0], (cx, cy))

        # label 拼接
        label = np.concatenate([heatmaps, centers, regs, offsets], axis=0)  # [17+1+34+34, Hf, Wf] = [86,Hf,Wf]

        # to tensor
        img_t = torch.from_numpy(img_lb.transpose(2, 0, 1)).float() / 255.0  # [3,H,W], 0~1
        label_t = torch.from_numpy(label).float()
        kps_mask_t = torch.from_numpy(kps_mask).float()

        return img_t, label_t, kps_mask_t, img_path


def create_kpts_dataloader(
        dataset_root: str,
        img_size: int,
        target_stride: int,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        aug_cfg: Dict[str, Any],
        is_train: bool
) -> DataLoader:
    """
    一个工厂函数，用于创建姿态估计任务的 DataLoader。
    这是新的外部调用接口。
    """
    img_dir_name = "train2017" if is_train else "val2017"
    ann_file_name = "person_keypoints_train2017.json" if is_train else "person_keypoints_val2017.json"

    # 兼容两种目录结构: <root>/train2017 和 <root>/images/train2017
    root_path = Path(dataset_root)
    img_root = root_path / "images" / img_dir_name
    if not img_root.is_dir():
        img_root = root_path / img_dir_name

    ann_path = root_path / "annotations" / ann_file_name

    if not img_root.is_dir() or not ann_path.is_file():
        raise FileNotFoundError(f"Data not found. Checked: {img_root} and {ann_path}")

    # 根据是否为训练集，选择性地应用数据增强
    if is_train:
        dataset = CocoKeypointsDataset(
            img_root=str(img_root),
            ann_path=str(ann_path),
            img_size=img_size,
            target_stride=target_stride,
            is_train=True,
            **aug_cfg  # 将所有增强相关的配置显式传入
        )
    else:  # 验证集不使用大部分增强
        dataset = CocoKeypointsDataset(
            img_root=str(img_root),
            ann_path=str(ann_path),
            img_size=img_size,
            target_stride=target_stride,
            is_train=False,
            use_color_aug=False,
            use_flip=False,
            use_rotate=False,
            use_scale=False
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=is_train
    )