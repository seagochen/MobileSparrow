import json
import os
from pathlib import Path
from typing import Dict, List

import torch
import torchvision
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
from tqdm import tqdm

# 根据你的项目结构调整导入
from sparrow.models.ssdlite_fpn import SSDLite_FPN


def _box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    """(center_x, center_y, width, height) -> (x1, y1, x2, y2)"""
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def _box_xyxy_to_cxcywh(x: torch.Tensor) -> torch.Tensor:
    """(x1, y1, x2, y2) -> (center_x, center_y, width, height)"""
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


class CocoDetectionEvaluator:
    """
    专用于COCO物体检测任务的评估器 (IoU-mAP)
    """

    def __init__(self,
                 val_loader: DataLoader,
                 score_threshold: float = 0.05,
                 nms_iou_threshold: float = 0.5,
                 results_dir: str = "eval_results"):
        """
        Args:
            val_loader (DataLoader): 验证集数据加载器.
            score_threshold (float): 后处理中用于过滤低置信度框的分数阈值.
            nms_iou_threshold (float): NMS的IoU阈值.
            results_dir (str): 保存中间结果JSON文件的目录.
        """
        self.loader = val_loader
        self.score_thr = score_threshold
        self.nms_iou_thr = nms_iou_threshold
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

        self.dataset = self.loader.dataset

        # --- 核心改动 1：修正 annotation file 的文件名 ---
        dataset_root = Path(self.dataset.img_root).parent.parent
        split = "train" if self.dataset.is_train else "val"

        # 错误的文件名: ann_file_name = f"person_keypoints_{split}2017.json"
        # 正确的文件名 (用于物体检测):
        ann_file_name = f"instances_{split}2017.json"

        self.ann_file = os.path.join(dataset_root, "annotations", ann_file_name)
        if not os.path.exists(self.ann_file):
            raise FileNotFoundError(f"Annotation file not found at constructed path: {self.ann_file}")

        print("[Evaluator] Building image path to ID map...")
        self.path_to_img_id = {}

        # 直接从 annotation JSON 读取 mapping
        with open(self.ann_file, "r") as f:
            anno = json.load(f)
        id_map = {im["file_name"]: im["id"] for im in anno["images"]}

        for item in self.dataset.items:
            img_path = item[0]
            file_name = os.path.basename(img_path)
            if file_name in id_map:
                self.path_to_img_id[img_path] = id_map[file_name]

        print(f"[Evaluator] Map created with {len(self.path_to_img_id)} entries.")

        # COCO评估需要原始的category_id, 而非连续的ID. 我们需要一个反向映射.
        self.contig2catid = {v: k for k, v in self.dataset.catid2contig.items()}

    @torch.no_grad()
    def _postprocess(self,
                     cls_logits: torch.Tensor,
                     bbox_deltas: torch.Tensor,
                     anchors: torch.Tensor) -> List[torch.Tensor]:
        """
        对单批次的模型输出进行后处理：解码、过滤和NMS

        Returns:
            List[torch.Tensor]: batch中每张图片最终的检测结果列表.
                                每个tensor的形状为 [num_dets, 6], 格式为 [x1, y1, x2, y2, score, class_idx].
        """
        device = cls_logits.device
        num_classes = cls_logits.shape[-1]

        # 1. 解码边界框
        anchors_cxcywh = _box_xyxy_to_cxcywh(anchors)
        pred_boxes_cxcywh = torch.zeros_like(bbox_deltas, device=device)

        # 应用回归增量 (dx, dy, dw, dh)
        pred_boxes_cxcywh[..., :2] = anchors_cxcywh[..., :2] + bbox_deltas[..., :2] * anchors_cxcywh[..., 2:]
        pred_boxes_cxcywh[..., 2:] = anchors_cxcywh[..., 2:] * torch.exp(bbox_deltas[..., 2:])

        # 转换回 xyxy 格式
        pred_boxes = _box_cxcywh_to_xyxy(pred_boxes_cxcywh)

        # 2. 获取类别分数
        scores = torch.sigmoid(cls_logits)

        # 3. 逐图片进行NMS
        batch_detections = []
        for i in range(cls_logits.shape[0]):
            # 过滤掉分数低于阈值的预测
            scores_per_img = scores[i]
            boxes_per_img = pred_boxes[i]

            # 创建一个索引，表示每个框属于哪个类别
            class_ids = torch.arange(num_classes, device=device).unsqueeze(0).expand_as(scores_per_img)

            # 展平，并将所有类别的预测放在一起
            scores_flat = scores_per_img.flatten()  # [A * C]
            boxes_flat = boxes_per_img.repeat_interleave(num_classes, dim=0)  # [A * C, 4]
            class_ids_flat = class_ids.flatten()  # [A * C]

            # 过滤
            keep = scores_flat > self.score_thr
            scores_filtered = scores_flat[keep]
            boxes_filtered = boxes_flat[keep]
            class_ids_filtered = class_ids_flat[keep]

            # 执行NMS (batched_nms可以同时处理不同类别的框)
            # 它会确保同一个类别的框之间进行NMS，不同类别的框互不影响
            keep_nms = torchvision.ops.batched_nms(boxes_filtered, scores_filtered, class_ids_filtered,
                                                   self.nms_iou_thr)

            # 组合最终结果
            final_boxes = boxes_filtered[keep_nms]
            final_scores = scores_filtered[keep_nms]
            final_class_ids = class_ids_filtered[keep_nms]

            batch_detections.append(
                torch.cat([
                    final_boxes,
                    final_scores.unsqueeze(1),
                    final_class_ids.unsqueeze(1)
                ], dim=1)
            )

        return batch_detections

    def evaluate(self, model: SSDLite_FPN, device: torch.device) -> Dict[str, float]:
        """
        执行完整的评估流程并返回指标字典
        """
        model.eval()
        coco_results = []

        pbar = tqdm(self.loader, desc="[Evaluator] Running inference", ncols=110)
        for images, targets, img_paths in pbar:
            images = images.to(device)
            preds = model(images)

            # 后处理得到最终检测框
            detections = self._postprocess(
                preds["cls_logits"],
                preds["bbox_deltas"],
                preds["anchors"]
            )

            # 格式化为COCO结果
            for i, dets_per_img in enumerate(detections):
                img_path = img_paths[i]
                img_id = self.path_to_img_id[img_path]

                if dets_per_img.numel() == 0:
                    continue

                # 将坐标缩放回原始图像尺寸
                # (注意：这里的dataloader已经将图片letterbox到方形，评估也应在此空间进行)
                # 如果需要映射回原图，需要dataloader提供缩放和平移信息。为简化，我们直接在方形空间评估。

                boxes = dets_per_img[:, :4].cpu().numpy()
                scores = dets_per_img[:, 4].cpu().numpy()
                labels = dets_per_img[:, 5].cpu().numpy().astype(int)

                # xyxy -> xywh
                boxes[:, 2] -= boxes[:, 0]
                boxes[:, 3] -= boxes[:, 1]

                for box, score, label_contig in zip(boxes, scores, labels):
                    coco_results.append({
                        "image_id": img_id,
                        "category_id": self.contig2catid[label_contig],  # 转换为原始ID
                        "bbox": box.tolist(),
                        "score": float(score)
                    })

        # --- 保存结果并调用COCO官方评估 ---
        res_path = os.path.join(self.results_dir, "bbox_results.json")
        with open(res_path, 'w') as f:
            json.dump(coco_results, f)
        print(f"[Evaluator] Prediction JSON saved to {res_path}")

        coco_gt = COCO(self.ann_file)
        coco_dt = coco_gt.loadRes(res_path)

        # 关键：将iouType改为'bbox'
        coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats_names = ['AP', 'AP .50', 'AP .75', 'AP (S)', 'AP (M)', 'AP (L)', 'AR max 1', 'AR max 10', 'AR max 100',
                       'AR (S)', 'AR (M)', 'AR (L)']
        metrics = {name: val for name, val in zip(stats_names, coco_eval.stats)}

        return metrics