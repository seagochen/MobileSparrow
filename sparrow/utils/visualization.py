import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.ops import nms
from typing import Tuple, List


# COCO 80 类名称（索引即类别 id 映射）
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]


# -----------------------------
# 预处理：读图 + letterbox 到 320x320
# -----------------------------
def load_and_letterbox(image_path: str, dst: int = 320) -> Tuple[np.ndarray, np.ndarray, int, int, float, int, int]:
    """
    返回:
      img_rgb      : 原图 (H, W, 3), RGB
      input_img    : letterbox 后的 320x320 RGB uint8
      h, w         : 原图高宽
      scale        : dst / max(h, w)
      pad_h, pad_w : 高/宽方向的 padding (各一半)
    """
    img_bgr = cv2.imread(image_path)
    assert img_bgr is not None, f"Image not found: {image_path}"
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    scale = dst / max(h, w)
    resized_h, resized_w = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img_rgb, (resized_w, resized_h))

    input_img = np.full((dst, dst, 3), 114, dtype=np.uint8)
    pad_h = (dst - resized_h) // 2
    pad_w = (dst - resized_w) // 2
    input_img[pad_h:pad_h + resized_h, pad_w:pad_w + resized_w] = img_resized
    return img_rgb, input_img, h, w, scale, pad_h, pad_w


# -----------------------------
# 前向推理
# -----------------------------
@torch.no_grad()
def model_infer(model, input_img: np.ndarray, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    input_img: letterbox 后的 320x320 uint8 RGB
    返回:
      scores    : [A, C] (sigmoid 后)
      reg_delta : [A, 4]
    """
    img_tensor = (torch.from_numpy(input_img.transpose(2, 0, 1)).float() / 255.0).to(device).unsqueeze(0)
    cls_preds, reg_preds = model(img_tensor)   # [1, A, C], [1, A, 4]
    scores = torch.sigmoid(cls_preds[0])       # [A, C]
    reg_delta = reg_preds[0]                   # [A, 4]
    return scores, reg_delta


# -----------------------------
# 置信度筛选：每 anchor 取最大类别分数
# -----------------------------
def filter_by_conf(scores: torch.Tensor, conf_thresh: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    scores: [A, C]
    返回:
      keep_mask     : [A] bool
      max_scores    : [A] float
      best_cls_idxs : [A] long
    """
    max_scores, best_cls_idxs = scores.max(dim=1)
    keep_mask = max_scores > conf_thresh
    return keep_mask, max_scores, best_cls_idxs


# -----------------------------
# 解码回归量 -> 预测框 (cxcywh -> xyxy)【与训练 encode 完全对称】
# -----------------------------
def decode_boxes(selected_anchors_xyxy: torch.Tensor,
                 selected_deltas: torch.Tensor,
                 bbox_std_vals: List[float] = [0.1, 0.1, 0.2, 0.2],
                 clamp_delta: float = 10.0) -> torch.Tensor:
    """
    输入:
      selected_anchors_xyxy : [K, 4]
      selected_deltas       : [K, 4]
    返回:
      decoded_boxes_xyxy    : [K, 4]
    """
    # xyxy -> cxcywh
    anchors_cxcywh = xyxy_to_cxcywh(selected_anchors_xyxy)
    bbox_std = selected_deltas.new_tensor(bbox_std_vals)         # 同 dtype/device
    d = (selected_deltas * bbox_std).clamp(-clamp_delta, clamp_delta)

    pred_cx = d[:, 0] * anchors_cxcywh[:, 2] + anchors_cxcywh[:, 0]
    pred_cy = d[:, 1] * anchors_cxcywh[:, 3] + anchors_cxcywh[:, 1]
    pred_w  = torch.exp(d[:, 2]) * anchors_cxcywh[:, 2]
    pred_h  = torch.exp(d[:, 3]) * anchors_cxcywh[:, 3]
    decoded_xyxy = cxcywh_to_xyxy(torch.stack([pred_cx, pred_cy, pred_w, pred_h], dim=1))
    return decoded_xyxy


# -----------------------------
# 数值清理 + 裁剪到固定范围
# -----------------------------
def sanitize_and_clip(boxes_xyxy: torch.Tensor, max_xy: int = 320) -> torch.Tensor:
    boxes = torch.nan_to_num(boxes_xyxy, nan=0.0, posinf=float(max_xy), neginf=0.0)
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, max_xy)  # x1/x2
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, max_xy)  # y1/y2
    return boxes


# -----------------------------
# NMS
# -----------------------------
def run_nms(boxes_xyxy: torch.Tensor, scores_1d: torch.Tensor, nms_thresh: float) -> torch.Tensor:
    """
    返回: keep_indices (LongTensor)
    """
    return nms(boxes_xyxy, scores_1d, nms_thresh)


# -----------------------------
# 坐标转换：xyxy <-> cxcywh（和你的 AnchorGenerator 用法一致）
# -----------------------------
def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    cx = boxes[:, 0] + w / 2
    cy = boxes[:, 1] + h / 2
    return torch.stack([cx, cy, w, h], dim=1)

def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    return torch.stack([x1, y1, x2, y2], dim=1)


# -----------------------------
# 反 letterbox：回到原图坐标（保留你的实现）
# -----------------------------
def unletterbox_boxes_xyxy(boxes_320: torch.Tensor, orig_w: int, orig_h: int,
                           scale: float, pad_w: int, pad_h: int) -> torch.Tensor:
    boxes = boxes_320.clone()
    boxes[:, [0, 2]] = boxes[:, [0, 2]] - pad_w
    boxes[:, [1, 3]] = boxes[:, [1, 3]] - pad_h
    boxes /= scale
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp_(0, orig_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp_(0, orig_h)
    return boxes


# -----------------------------
# 绘制：在 320x320 letterbox 图上
# -----------------------------
def draw_on_letterboxed(input_img: np.ndarray,
                        boxes_xyxy: torch.Tensor,
                        scores_1d: torch.Tensor,
                        labels_1d: torch.Tensor,
                        keep_indices: torch.Tensor,
                        save_path: str | None = None,
                        show: bool = False) -> None:
    fig, ax = plt.subplots(1, figsize=(12, 9), dpi=100)
    ax.set_xlim(0, 320)
    ax.set_ylim(320, 0)
    ax.imshow(input_img, extent=[0, 320, 320, 0])
    for i in keep_indices:
        x1, y1, x2, y2 = boxes_xyxy[i].detach().cpu().numpy().tolist()
        cls_id = int(labels_1d[i])
        score  = float(scores_1d[i])
        name   = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f'Cls {cls_id}'
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, max(0, y1-5), f'{name}: {score:.2f}',
                color='black', bbox=dict(facecolor='lime', alpha=0.8), fontsize=10)
    ax.axis('off')
    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches='tight', pad_inches=0.1); plt.close(fig)
    elif show:
        plt.show(); plt.close(fig)
    else:
        plt.close(fig)


# -----------------------------
# 绘制：在原图上（坐标为原图像素）
# -----------------------------
def draw_on_original(img_rgb: np.ndarray,
                     boxes_xyxy_orig: torch.Tensor,
                     scores_1d: torch.Tensor,
                     labels_1d: torch.Tensor,
                     keep_indices: torch.Tensor,
                     save_path: str | None = None,
                     show: bool = False) -> None:
    h, w = img_rgb.shape[:2]
    fig, ax = plt.subplots(1, figsize=(12, 9), dpi=100)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)      # 让坐标原点在左上
    ax.imshow(img_rgb)
    for i in keep_indices:
        x1, y1, x2, y2 = boxes_xyxy_orig[i].detach().cpu().numpy().tolist()
        cls_id = int(labels_1d[i])
        score  = float(scores_1d[i])
        name   = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f'Cls {cls_id}'
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, max(0, y1-5), f'{name}: {score:.2f}',
                color='black', bbox=dict(facecolor='lime', alpha=0.8), fontsize=10)
    ax.axis('off')
    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches='tight', pad_inches=0.1); plt.close(fig)
    elif show:
        plt.show(); plt.close(fig)
    else:
        plt.close(fig)


# =============================
# 总控：可视化单张图片
# =============================
@torch.no_grad()
def visualize_predictions(
    model,
    image_path: str,
    device,
    precomputed_anchors: torch.Tensor,
    conf_thresh: float = 0.3,
    nms_thresh: float = 0.45,
    save_path: str | None = None,
    draw_on_orig: bool = True,
    show: bool = False,
):
    # 1) 读图 + letterbox
    img_rgb, input_img, H, W, scale, pad_h, pad_w = load_and_letterbox(image_path, dst=320)

    # 2) 前向
    scores, reg_delta = model_infer(model, input_img, device)              # [A,C], [A,4]

    # 3) 筛选
    keep_mask, max_scores, best_cls = filter_by_conf(scores, conf_thresh)  # [A], [A], [A]
    if keep_mask.sum().item() == 0:
        # 没有有效候选，仍然输出空图
        draw_on_letterboxed(input_img,
                            torch.empty((0, 4), device=device),
                            torch.empty(0, device=device),
                            torch.empty(0, dtype=torch.long, device=device),
                            keep_indices=torch.tensor([], dtype=torch.long),
                            save_path=save_path, show=show)
        return

    # 4) 解码
    anchors_dev = precomputed_anchors.to(device)
    sel_anchors = anchors_dev[keep_mask]     # [K,4]
    sel_deltas  = reg_delta[keep_mask]       # [K,4]

    boxes_320 = decode_boxes(sel_anchors, sel_deltas)  # [K,4] xyxy
    boxes_320 = sanitize_and_clip(boxes_320, max_xy=320)

    final_scores = max_scores[keep_mask].float()      # [K]
    final_labels = best_cls[keep_mask].long()         # [K]

    # 5) NMS
    keep_idx = run_nms(boxes_320, final_scores, nms_thresh)  # [M]

    # 6) 绘制
    if draw_on_orig:
        boxes_orig = unletterbox_boxes_xyxy(boxes_320, orig_w=W, orig_h=H, scale=scale, pad_w=pad_w, pad_h=pad_h)
        draw_on_original(img_rgb, boxes_orig, final_scores, final_labels, keep_idx, save_path=save_path, show=show)
    else:
        draw_on_letterboxed(input_img, boxes_320, final_scores, final_labels, keep_idx, save_path=save_path, show=show)