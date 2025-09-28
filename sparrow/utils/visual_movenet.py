import cv2
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Union

# ---------- COCO 关键点与骨架 ----------
COCO_KEYPOINTS = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist",
    "left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"
]

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
        # 默认 numpy 数组来自 cv2（BGR）；若你传入的是 RGB，可按需改成直接使用
        img_arr = np.asarray(image)
        if img_arr.ndim == 3 and img_arr.shape[2] == 3:
            img_rgb = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        elif img_arr.ndim == 2: # 灰度图
            img_rgb = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = img_arr  # 已是 RGB 的情况

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

# ---------- 颜色/绘制 ----------
def _circle(ax, x, y, r=3, color="cyan"):
    circ = plt.Circle((x, y), r, color=color, fill=True, alpha=0.9)
    ax.add_patch(circ)

def _line(ax, x1, y1, x2, y2, color="yellow", lw=2):
    ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw)

# --- 热力图绘制辅助（可选） ---
def _draw_heatmaps(
    base_img_rgb: np.ndarray,
    pred_heatmaps: torch.Tensor,
    pred_centers: torch.Tensor,
    alpha: float = 0.6
) -> Tuple[np.ndarray, np.ndarray]:
    """将预测的热力图叠加到基础图像上"""
    H, W = base_img_rgb.shape[:2]

    # 关键点热力图(取所有关节通道最大值)
    kpt_hm_prob = torch.sigmoid(pred_heatmaps[0])
    kpt_hm_composite = kpt_hm_prob.max(dim=0)[0].cpu().numpy()
    kpt_hm_resized = cv2.resize(kpt_hm_composite, (W, H), interpolation=cv2.INTER_LINEAR)
    kpt_hm_norm = (255 * (kpt_hm_resized - kpt_hm_resized.min()) / (kpt_hm_resized.max() - kpt_hm_resized.min() + 1e-6)).astype(np.uint8)
    kpt_heatmap_color = cv2.applyColorMap(kpt_hm_norm, cv2.COLORMAP_JET)

    # 中心点热力图
    center_hm_prob = torch.sigmoid(pred_centers[0, 0]).cpu().numpy()
    center_hm_resized = cv2.resize(center_hm_prob, (W, H), interpolation=cv2.INTER_LINEAR)
    center_hm_norm = (255 * (center_hm_resized - center_hm_resized.min()) / (center_hm_resized.max() - center_hm_resized.min() + 1e-6)).astype(np.uint8)
    center_heatmap_color = cv2.applyColorMap(center_hm_norm, cv2.COLORMAP_JET)

    # 融合
    base_img_bgr = cv2.cvtColor(base_img_rgb, cv2.COLOR_RGB2BGR)
    overlay_kpt = cv2.addWeighted(base_img_bgr, 1 - alpha, kpt_heatmap_color, alpha, 0)
    overlay_center = cv2.addWeighted(base_img_bgr, 1 - alpha, center_heatmap_color, alpha, 0)

    return cv2.cvtColor(overlay_kpt, cv2.COLOR_BGR2RGB), cv2.cvtColor(overlay_center, cv2.COLOR_BGR2RGB)

# ---------- 单人筛选（NMS + OKS + 中心先验） ----------
def _oks(a_kps, b_kps, img_wh, sigmas=None):
    if sigmas is None:
        sigmas = np.array([.26,.25,.25,.35,.35,.79,.79,.72,.72,.62,.62,1.07,1.07,.87,.87,.89,.89], dtype=np.float32)/10.0
    vars2 = (2 * (sigmas**2))
    a = np.array(a_kps); b = np.array(b_kps)
    vis = (a[:,2] > 1e-5) & (b[:,2] > 1e-5)
    if vis.sum() == 0:
        return 0.0
    dx = a[vis,0] - b[vis,0]; dy = a[vis,1] - b[vis,1]
    wh = max(img_wh)
    e = (dx*dx + dy*dy) / (vars2[vis] * (wh**2) + 1e-9)
    return float(np.mean(np.exp(-e)))

def _nms_and_pick_single(dets, img_w, img_h, iou_thr=0.5, oks_thr=0.75,
                         center_bias=0.35, conf_bias=0.65):
    """
    dets: [{'bbox':[x1,y1,x2,y2], 'person_score':s, 'keypoints':[(x,y,c)*17]}...]
    返回只保留一个实例（NMS + OKS 去重 + 中心先验 + 关键点质量）。
    坐标系使用 letterbox 坐标 (input_size x input_size)。
    """
    if len(dets) <= 1:
        return dets

    # IoU（用于初步去重）
    def iou(a,b):
        ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
        inter_x1=max(ax1,bx1); inter_y1=max(ay1,by1)
        inter_x2=min(ax2,bx2); inter_y2=min(ay2,by2)
        iw=max(0.0, inter_x2-inter_x1); ih=max(0.0, inter_y2-inter_y1)
        inter=iw*ih
        area_a=(ax2-ax1)*(ay2-ay1); area_b=(bx2-bx1)*(by2-by1)
        union=area_a+area_b-inter + 1e-9
        return inter/union

    keep=[]
    dets_sorted = sorted(dets, key=lambda d: d['person_score'], reverse=True)
    suppressed=[False]*len(dets_sorted)
    for i,di in enumerate(dets_sorted):
        if suppressed[i]: continue
        keep.append(di)
        for j,dj in enumerate(dets_sorted[i+1:], start=i+1):
            if suppressed[j]: continue
            if iou(di['bbox'], dj['bbox']) > iou_thr:
                suppressed[j]=True

    # OKS 去重（关键点几乎相同的合并）
    final=[]
    for d in keep:
        dup=False
        for f in final:
            if _oks(d['keypoints'], f['keypoints'], (img_w, img_h)) > oks_thr:
                dup=True; break
        if not dup: final.append(d)
    if len(final)==0:
        final = keep

    # 中心先验 + 关键点均值 + person_score 融合评分
    cx, cy = img_w/2.0, img_h/2.0
    def score_det(d):
        x1,y1,x2,y2 = d['bbox']
        bx = 0.5*(x1+x2); by = 0.5*(y1+y2)
        dist = np.hypot((bx-cx)/img_w, (by-cy)/img_h)
        center_term = 1.0 - np.clip(dist*2.0, 0.0, 1.0)  # [0,1]，越近越大
        kps = np.array(d['keypoints'])
        kp_conf = float(np.mean(kps[:,2])) if len(kps)>0 else 0.0
        return conf_bias * float(d['person_score']) + (1.0-conf_bias) * (0.5*kp_conf + 0.5*center_term)

    best = max(final, key=score_det)
    return [best]

# ---------- 主流程：仅绘制关键点与骨架 ----------
@torch.no_grad()
def visualize_movenet(
    model,
    image: Union[np.ndarray, str],
    device,
    decoder,                      # 解码函数：decode_movenet_outputs
    input_size: int = 192,        # 模型输入（letterbox 尺寸）
    stride: int = 8,              # 输出步幅
    topk_centers: int = 5,
    center_thresh: float = 0.2,
    keypoint_thresh: float = 0.05,
    draw_on_orig: bool = True,
    draw_heatmaps: bool = False,  # 是否叠加热力图
    heatmap_alpha: float = 0.6,   # 热力图透明度
    force_single: bool = True,    # 单人场景：只保留一个实例
    single_iou_thr: float = 0.5,
    single_oks_thr: float = 0.75,
    single_center_bias: float = 0.35,
    single_conf_bias: float = 0.65,
    save_path: Union[str, None] = None,
    show: bool = False,
) -> np.ndarray:
    """
    仅绘制关键点与骨架，不画 bbox。
    返回：np.ndarray shape (17, 3) 的关键点数组（[x, y, conf]，坐标为原图像素）。
    若未解出关键点，返回全 NaN。
    """
    # 1) 读图 + letterbox
    img_rgb, input_img, H, W, scale, pad_h, pad_w = load_and_letterbox(image, dst=input_size)

    # 2) 前向（与训练一致的标准化）
    img_np = input_img.astype(np.float32) / 255.0
    img_np = (img_np - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).to(device)

    model.eval()
    preds = model(img_tensor)  # dict: heatmaps/centers/regs/offsets

    # 3) 解码（此时 dets 坐标系为 letterbox 坐标：input_size x input_size）
    dets = decoder(
        preds,
        img_size=(input_size, input_size),
        stride=stride,
        topk_centers=topk_centers,
        center_thresh=center_thresh,
        keypoint_thresh=keypoint_thresh,
        regs_in_pixels=False
    )

    # 3.5) 单人筛选
    if force_single and len(dets) > 1:
        dets = _nms_and_pick_single(
            dets,
            img_w=input_size,
            img_h=input_size,
            iou_thr=single_iou_thr,
            oks_thr=single_oks_thr,
            center_bias=single_center_bias,
            conf_bias=single_conf_bias
        )

    # 4) 选择显示底图
    display_img = img_rgb if draw_on_orig else input_img
    display_W, display_H = (W, H) if draw_on_orig else (input_size, input_size)

    # 5) 画布与（可选）热力图
    if draw_heatmaps:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=100)
        ax_main, ax_hm_kpt, ax_hm_center = axes
        overlay_kpt, overlay_center = _draw_heatmaps(
            display_img, preds['heatmaps'], preds['centers'], alpha=heatmap_alpha
        )
        ax_hm_kpt.imshow(overlay_kpt); ax_hm_kpt.set_title("Keypoint Heatmap")
        ax_hm_center.imshow(overlay_center); ax_hm_center.set_title("Center Heatmap")
    else:
        fig, ax = plt.subplots(1, figsize=(12, 9), dpi=100)
        ax_main = ax

    ax_main.imshow(display_img)
    ax_main.set_title("Pose Estimation (Keypoints Only)")
    for ax in fig.axes:
        ax.set_xlim(0, display_W); ax.set_ylim(display_H, 0)
        ax.axis('off')

    # 6) 只取一个实例的关键点，映射回原图并绘制骨架
    kp_out = np.full((17, 3), np.nan, dtype=np.float32)  # 预置 NaN 返回
    if len(dets) >= 1:
        det = dets[0]
        kps = np.array([(x, y, c) for (x, y, c) in det["keypoints"]], dtype=np.float32)
        # 将 (x,y) 从 letterbox 坐标还原到原图
        if len(kps) > 0:
            pts = unletterbox_points(kps[:, :2], W, H, scale, pad_w, pad_h)
            kps[:, :2] = pts

            # 绘制关键点
            for j, (x, y, c) in enumerate(kps):
                if c > keypoint_thresh and not (math.isnan(x) or math.isnan(y)):
                    _circle(ax_main, float(x), float(y), r=3, color="cyan")

            # 绘制骨架
            for (a, b) in COCO_SKELETON:
                if a < len(kps) and b < len(kps):
                    x1, y1, c1 = kps[a]; x2, y2, c2 = kps[b]
                    if c1 > keypoint_thresh and c2 > keypoint_thresh and not (math.isnan(x1) or math.isnan(y1) or math.isnan(x2) or math.isnan(y2)):
                        _line(ax_main, float(x1), float(y1), float(x2), float(y2), color="yellow", lw=2)

            # 准备返回：确保长度为 17（不足则填 NaN）
            K = kps.shape[0]
            if K >= 17:
                kp_out = kps[:17]
            else:
                kp_out[:K] = kps

    # 7) 保存或显示
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches='tight', pad_inches=0.1)
    if show:
        plt.show()
    plt.close(fig)

    return kp_out
