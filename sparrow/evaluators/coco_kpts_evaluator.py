import json
import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
from tqdm import tqdm

from sparrow.models.movenet_fpn import MoveNet_FPN


class CocoKeypointsEvaluator:
    """
    专用于COCO人体关键点任务的评估器 (OKS-AP)
    """

    def __init__(self,
                 val_loader: DataLoader,
                 stride: int,
                 results_dir: str = "eval_results"):
        """
        Args:
            val_loader (DataLoader): 验证集的数据加载器
            stride (int): 模型的总步长 (例如 4 或 8)
            results_dir (str): 保存中间结果JSON文件的目录
        """

        self.loader = val_loader
        self.stride = stride
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

        # --- 核心改动：动态构建 annotation file 路径 ---
        dataset = self.loader.dataset

        # 1. 从 dataset.img_root (e.g., ".../coco/images/val2017") 推断出数据集根目录
        #    Path(...).parent.parent 会向上两级，得到 ".../coco"
        dataset_root = Path(dataset.img_root).parent.parent

        # 2. 根据 is_train 标志确定文件名
        split = "train" if dataset.is_train else "val"
        ann_file_name = f"person_keypoints_{split}2017.json"

        # 3. 组合成完整路径
        self.ann_file = os.path.join(dataset_root, "annotations", ann_file_name)

        if not os.path.exists(self.ann_file):
            raise FileNotFoundError(f"Annotation file not found at constructed path: {self.ann_file}")

    @torch.no_grad()
    def _decode_predictions(self, heatmaps: torch.Tensor, offsets: torch.Tensor) -> np.ndarray:
        """从热图和偏移量解码出关键点坐标和分数"""
        batch_size, num_joints, h, w = heatmaps.shape
        heatmaps = torch.sigmoid(heatmaps)
        scores, inds = torch.max(heatmaps.view(batch_size, num_joints, -1), dim=2)
        y_coords = (inds / w).int().float()
        x_coords = (inds % w).int().float()

        offsets = offsets.view(batch_size, num_joints, 2, h, w)
        offset_x = offsets[:, :, 0, :, :].view(batch_size, num_joints, -1).gather(2, inds.unsqueeze(-1)).squeeze(-1)
        offset_y = offsets[:, :, 1, :, :].view(batch_size, num_joints, -1).gather(2, inds.unsqueeze(-1)).squeeze(-1)

        pred_x = (x_coords + offset_x) * self.stride
        pred_y = (y_coords + offset_y) * self.stride

        return torch.stack([pred_x, pred_y, scores], dim=2).cpu().numpy()

    def evaluate(self, model: MoveNet_FPN, device: torch.device) -> Dict[str, float]:
        """
        执行完整的评估流程并返回指标字典

        Args:
            model (MoveNet_FPN): 待评估的模型
            device (torch.device): 运行设备

        Returns:
            Dict[str, float]: 包含AP, AP50等指标的字典
        """
        model.eval()
        coco_results = []

        pbar = tqdm(self.loader, desc="[Evaluator] Running inference", ncols=110)
        for images, _, _, img_paths in pbar:
            images = images.to(device)
            preds = model(images)
            pred_kpts = self._decode_predictions(preds["heatmaps"], preds["offsets"])

            for i, path in enumerate(img_paths):
                # 从文件名(e.g., '000000123456.jpg')中提取 image_id
                # img_id = int(os.path.splitext(os.path.basename(path))[0])
                # kpts = pred_kpts[i]

                # --- START OF CHANGE ---

                # Original line that caused the error:
                # img_id = int(os.path.splitext(os.path.basename(path))[0])

                # NEW, ROBUST WAY:
                # Get filename without extension, e.g., "000000397133_aid200887"
                filename_base = os.path.splitext(os.path.basename(path))[0]
                # The original image_id is the part before the first underscore
                original_img_id_str = filename_base.split('_')[0]
                img_id = int(original_img_id_str)

                # --- END OF CHANGE ---

                kpts = pred_kpts[i]

                coco_kpts = np.zeros(17 * 3, dtype=np.float32)
                for j in range(kpts.shape[0]):
                    coco_kpts[j * 3 + 0] = kpts[j, 0]
                    coco_kpts[j * 3 + 1] = kpts[j, 1]
                    coco_kpts[j * 3 + 2] = kpts[j, 2]

                coco_results.append({
                    "image_id": img_id,
                    "category_id": 1,  # person
                    "keypoints": coco_kpts.tolist(),
                    "score": 1.0
                })

        # --- 保存结果并调用COCO官方评估 ---
        res_path = os.path.join(self.results_dir, "keypoints_val_results.json")
        with open(res_path, 'w') as f:
            json.dump(coco_results, f)
        print(f"[Evaluator] Prediction JSON saved to {res_path}")

        coco_gt = COCO(self.ann_file)
        coco_dt = coco_gt.loadRes(res_path)
        coco_eval = COCOeval(coco_gt, coco_dt, 'keypoints')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # 将结果封装成字典返回
        stats_names = ['AP', 'AP .50', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .50', 'AR .75', 'AR (M)', 'AR (L)']
        metrics = {name: val for name, val in zip(stats_names, coco_eval.stats)}

        return metrics