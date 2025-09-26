import cv2
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple, List, Union

# ---------- COCO 关键点与骨架 ----------
# 关键点名称（可用于调试/标注）
COCO_KEYPOINTS = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist",
    "left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"
]

# 骨架连线（COCO常用拓扑）
COCO_SKELETON = [
    (5, 6),   # left_shoulder - right_shoulder
    (5, 7), (7, 9),   # left_shoulder - left_elbow - left_wrist
    (6, 8), (8,10),   # right_shoulder - right_elbow - right_wrist
    (11,12),          # left_hip - right_hip
    (5,11), (6,12),   # shoulder-hip
    (11,13), (13,15), # left_hip - left_knee - left_ankle
    (12,14), (14,16), # right_hip - right_knee - right_ankle
    (0,1), (0,2), (1,3), (2,4)  # face links
]

# ---------- 读图 + letterbox 到指定方形尺寸 ----------
def load_and_letterbox(image: Union[str, np.ndarray], dst: int = 192) -> Tuple[np.ndarray, np.ndarray, int, int, float, int, int]:
    """
    返回:
      img_rgb      : 原图 (H, W, 3), RGB
      input_img    : letterbox 后的 dst x dst RGB uint8
      h, w         : 原图高宽
      scale        : dst / max(h, w)
      pad_h, pad_w : 高/宽方向的 padding (各一半)
    """
    if isinstance(image, str):
        img_bgr = cv2.imread(image)
        assert img_bgr is not None, f"Image not found: {image}"
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)
        elif arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
        else:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB) if arr.dtype != np.uint8 else arr[..., ::-1] if False else arr
        img_rgb = arr

    h, w = img_rgb.shape[:2]
    scale = dst / max(h, w)
    resized_h, resized_w = int(round(h * scale)), int(round(w * scale))
    img_resized = cv2.resize(img_rgb, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    input_img = np.full((dst, dst, 3), 114, dtype=np.uint8)
    pad_h = (dst - resized_h) // 2
    pad_w = (dst - resized_w) // 2
    input_img[pad_h:pad_h + resized_h, pad_w:pad_w + resized_w] = img_resized
    return img_rgb, input_img, h, w, scale, pad_h, pad_w

# ---------- 将 letterbox 坐标还原到原图 ----------
def unletterbox_points(points: np.ndarray, orig_w: int, orig_h: int, scale: float, pad_w: int, pad_h: int) -> np.ndarray:
    """
    points: [N,2] 在 letterbox 图（dst x dst）中像素坐标
    返回：对应到原图坐标的点
    """
    pts = points.copy()
    pts[:, 0] = (pts[:, 0] - pad_w) / scale
    pts[:, 1] = (pts[:, 1] - pad_h) / scale
    pts[:, 0] = np.clip(pts[:, 0], 0, orig_w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, orig_h - 1)
    return pts

def unletterbox_box_xyxy(box: np.ndarray, orig_w: int, orig_h: int, scale: float, pad_w: int, pad_h: int) -> np.ndarray:
    x1,y1,x2,y2 = box
    pts = np.array([[x1,y1],[x2,y2]], dtype=np.float32)
    pts = unletterbox_points(pts, orig_w, orig_h, scale, pad_w, pad_h)
    x1,y1 = pts[0]; x2,y2 = pts[1]
    return np.array([x1,y1,x2,y2], dtype=np.float32)

# ---------- 颜色/绘制 ----------
def _circle(ax, x, y, r=3, color="cyan", lw=1.5):
    circ = plt.Circle((x, y), r, color=color, fill=True, alpha=0.9)
    ax.add_patch(circ)

def _line(ax, x1, y1, x2, y2, color="yellow", lw=2):
    ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw)

# ---------- 主流程：可视化（单张） ----------
@torch.no_grad()
def visualize_movenet(
    model,
    image: Union[np.ndarray, str],
    device,
    decoder,                     # 传入 decode_movenet_outputs
    input_size: int = 192,       # 模型输入
    stride: int = 8,             # 输出步幅（P3=8；如果你上采样到1/4，请改为4）
    topk_centers: int = 5,
    center_thresh: float = 0.2,
    keypoint_thresh: float = 0.05,
    draw_bbox: bool = True,
    draw_skeleton: bool = True,
    draw_on_orig: bool = True,   # 在原图上画；否则在 letterboxed 图上画
    save_path: str | None = None,
    show: bool = False,
):
    """
    根据 MoveNet 输出，绘制 bbox + person_score + 关键点 + 骨架。
    """
    # 1) 读图 + letterbox
    img_rgb, input_img, H, W, scale, pad_h, pad_w = load_and_letterbox(image, dst=input_size)

    # 2) 前向
    img_tensor = (torch.from_numpy(input_img.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).to(device)
    model.eval()
    preds = model(img_tensor)  # dict: heatmaps/centers/regs/offsets

    # 3) 解码（注意：img_size 传 letterbox 后的输入大小）
    dets = decoder(
        preds,
        img_size=(input_size, input_size),
        stride=stride,
        topk_centers=topk_centers,
        center_thresh=center_thresh,
        keypoint_thresh=keypoint_thresh,
        regs_in_pixels=True
    )

    # 4) 选择绘制坐标系
    if draw_on_orig:
        fig, ax = plt.subplots(1, figsize=(12, 9), dpi=100)
        ax.set_xlim(0, W); ax.set_ylim(H, 0)
        ax.imshow(img_rgb)
    else:
        fig, ax = plt.subplots(1, figsize=(8, 8), dpi=100)
        ax.set_xlim(0, input_size); ax.set_ylim(input_size, 0)
        ax.imshow(input_img)

    # 5) 逐实例绘制
    for det in dets:
        bbox = np.array(det["bbox"], dtype=np.float32)                  # [x1,y1,x2,y2] (letterbox坐标)
        score = float(det["person_score"])
        kps = np.array([(x, y, c) for (x, y, c) in det["keypoints"]])   # [K,3] (x,y,conf) letterbox坐标

        if draw_on_orig:
            # 将坐标还原到原图
            bbox = unletterbox_box_xyxy(bbox, W, H, scale, pad_w, pad_h)
            if len(kps) > 0:
                pts = unletterbox_points(kps[:, :2], W, H, scale, pad_w, pad_h)
                kps[:, :2] = pts

        # 5.1 画 bbox
        if draw_bbox:
            x1,y1,x2,y2 = bbox.tolist()
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, max(0, y1-6), f'person: {score:.2f}', color='black',
                    bbox=dict(facecolor='lime', alpha=0.8), fontsize=10)

        # 5.2 画关键点 & 骨架
        if len(kps) > 0:
            # 关键点
            for j, (x,y,c) in enumerate(kps):
                if c <= 0.0 or math.isnan(x) or math.isnan(y):
                    continue
                _circle(ax, x, y, r=3, color="cyan", lw=1.5)

            # 骨架
            if draw_skeleton:
                for (a,b) in COCO_SKELETON:
                    if a >= len(kps) or b >= len(kps):
                        continue
                    x1,y1,c1 = kps[a]; x2,y2,c2 = kps[b]
                    if c1 > 0.0 and c2 > 0.0 and not (math.isnan(x1) or math.isnan(y1) or math.isnan(x2) or math.isnan(y2)):
                        _line(ax, x1, y1, x2, y2, color="yellow", lw=2)

    ax.axis('off')
    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches='tight', pad_inches=0.1); plt.close(fig)
    elif show:
        plt.show(); plt.close(fig)
    else:
        plt.close(fig)
