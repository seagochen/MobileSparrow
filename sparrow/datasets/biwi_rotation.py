# -*- coding: utf-8 -*-
import glob
import os

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


def load_intrinsics(calib_path: str) -> tuple[float, float, float, float]:
    """
    从 BIWI 的 rgb.cal 里解析 3x3 内参 K：
    取前 3 行，每行 3 个数，K = [[fx, 0, cx],[0, fy, cy],[0,0,1]]
    如失败再回退成“4 个数 (fx,fy,cx,cy)”。
    """
    K_rows = []
    with open(calib_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                if len(K_rows) >= 3:
                    break
                else:
                    continue
            parts = line.replace(",", " ").split()
            vals = []
            for p in parts:
                try:
                    vals.append(float(p))
                except:
                    pass
            if len(vals) == 3 and len(K_rows) < 3:
                K_rows.append(vals)
                if len(K_rows) == 3:
                    break

    if len(K_rows) == 3:
        K = np.asarray(K_rows, dtype=np.float32)
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        return fx, fy, cx, cy

    # 回退（一般用不到）
    import re
    with open(calib_path, "r") as f:
        txt = f.read()
    nums = [float(x) for x in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', txt)]
    if len(nums) >= 4:
        return float(nums[0]), float(nums[1]), float(nums[2]), float(nums[3])
    raise ValueError(f"Cannot parse intrinsics from {calib_path}")


def read_pose_txt(pose_txt: str):
    """
    读取 BIWI 姿态：前三行 R (3x3)，最后一行 t (3,)
    中间可有空行。返回 (R,t) 均为 np.float32。
    """
    with open(pose_txt, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) != 4:
        raise ValueError(f"Expect 4 lines (3 for R, 1 for t), got {len(lines)} in {pose_txt}")

    R_rows = [[float(x) for x in line.split()] for line in lines[:3]]
    R = np.array(R_rows, dtype=np.float32)
    t = np.array([float(x) for x in lines[3].split()], dtype=np.float32)

    # 基本健壮性检查（允许一点数值误差）
    RtR = R.T @ R
    if not (np.allclose(RtR, np.eye(3), atol=1e-2) and 0.9 < np.linalg.det(R) < 1.1):
        # 打印一下，但不中断
        print(
            f"[WARN] R may not be perfectly orthonormal: ||RtR-I||={np.linalg.norm(RtR - np.eye(3)):.3e}, det={np.linalg.det(R):.5f}")

    return R, t


def project_point(K, X):
    fx, fy, cx, cy = K
    Xx, Xy, Xz = X
    Xz = max(1e-6, float(Xz))
    u = fx * (Xx / Xz) + cx
    v = fy * (Xy / Xz) + cy
    return u, v


class BIWIDataset(Dataset):
    def __init__(self, root_dir,
                 split='train',
                 img_size=224,
                 crop_size=256,
                 center_jitter=0.05,
                 scale_jitter=0.1,
                 use_crop=True,
                 return_path=False):
        """
        root_dir: 指向包含 faces_0/ 的目录, e.g., '/home/user/datasets/biwi'
        """
        self.return_path = return_path
        self.use_crop = use_crop
        self.crop_size = int(crop_size)
        self.center_jitter = float(center_jitter)
        self.scale_jitter = float(scale_jitter)

        data_dir = os.path.join(root_dir, 'faces_0')
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"'faces_0' not found in {root_dir}")

        seq_dirs = sorted([d for d in glob.glob(os.path.join(data_dir, '*')) if os.path.isdir(d)])
        img_paths, pose_paths = [], []
        for sd in seq_dirs:
            imgs = sorted(glob.glob(os.path.join(sd, '*_rgb.png')))
            for ip in imgs:
                pp = ip.replace('_rgb.png', '_pose.txt')
                if os.path.exists(pp):
                    img_paths.append(ip)
                    pose_paths.append(pp)

        self.img_paths = img_paths
        self.pose_paths = pose_paths

        # 预处理
        self.resize = T.Resize((img_size, img_size), antialias=True)
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        print(f"[BIWI] Found {len(self.img_paths)} frames.")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        pose_path = self.pose_paths[idx]

        # 图像
        img = Image.open(img_path).convert('RGB')
        W, H = img.size

        # 姿态
        R, t = read_pose_txt(pose_path)  # np.float32
        R_gt = torch.from_numpy(R)  # torch.float32

        # 标定文件（优先 rgb.cal）
        calib_path_rgb = os.path.join(os.path.dirname(img_path), 'rgb.cal')
        calib_path_depth = os.path.join(os.path.dirname(img_path), 'depth.cal')
        calib_path = calib_path_rgb if os.path.exists(calib_path_rgb) else calib_path_depth
        fx, fy, cx, cy = load_intrinsics(calib_path)

        # 断言合理的中心（BIWI 640x480 常见中心约 (320, 240.5)）
        if not (300 <= cx <= 340 and 230 <= cy <= 250):
            print(f"[WARN] suspicious cx,cy: {(fx, fy, cx, cy)} in {calib_path}")

        # 投影头中心
        u, v = project_point((fx, fy, cx, cy), t)

        # 生成裁剪（训练一致）
        if self.use_crop:
            # 抖动与缩放
            jitter_u = (np.random.rand() * 2 - 1) * self.center_jitter * self.crop_size
            jitter_v = (np.random.rand() * 2 - 1) * self.center_jitter * self.crop_size
            scale = 1.0 + (np.random.rand() * 2 - 1) * self.scale_jitter
            cs = int(round(self.crop_size * scale))

            cx_i = int(round(u + jitter_u))
            cy_i = int(round(v + jitter_v))
            half = cs // 2

            x1 = int(np.clip(cx_i - half, 0, W - 1))
            y1 = int(np.clip(cy_i - half, 0, H - 1))
            x2 = int(np.clip(x1 + cs, 0, W))
            y2 = int(np.clip(y1 + cs, 0, H))
        else:
            # 不裁剪：用全图，并把较大边缩放到 img_size；记录等效“裁剪框=整图”
            cs = max(W, H)
            x1, y1, x2, y2 = 0, 0, W, H

        crop = img.crop((x1, y1, x2, y2))
        img_proc = self.resize(crop)

        # 从裁剪边长到网络输入的缩放
        resize_scale = self.resize.size[0] / float(x2 - x1 if (x2 - x1) > 0 else 1)

        # 张量化
        x = self.normalize(self.to_tensor(img_proc))

        # meta 统一成 numpy/基础类型（易于 DataLoader 堆叠）
        sample = {
            'image': x,
            'R_gt': R_gt,
            'meta': {
                'path': img_path if self.return_path else "",
                'intrinsics': np.array([fx, fy, cx, cy], dtype=np.float32),
                't': t.astype(np.float32),
                'u': np.float32(u),
                'v': np.float32(v),
                'crop_box': np.array([x1, y1, x2, y2], dtype=np.int32),
                'resize_scale': np.float32(resize_scale),
                'crop_side': np.int32(x2 - x1),
            }
        }
        return sample
