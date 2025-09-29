# sparrow/utils/visualizer.py

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Union, Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ---------- COCO 关键点与骨架定义 ----------
# (保持与您项目其他部分一致)
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

COCO_SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4)
]

# 为骨架连接定义不同的颜色 (BGR格式)
SKELETON_COLORS = [
    (0, 255, 255), (0, 255, 255), (0, 255, 255), (0, 255, 255),  # 四肢 (黄色)
    (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0),          # 肩膀/臀部连接 (绿色)
    (255, 128, 0), (255, 128, 0), (255, 128, 0), (255, 128, 0),  # 躯干 (蓝色)
    (255, 0, 255), (255, 0, 255), (255, 0, 255), (255, 0, 255), (255, 0, 255) # 头部 (品红色)
]

# ---------- 核心绘图函数 ----------

def _draw_keypoints_and_skeleton(
    image: np.ndarray,
    detections: List[Dict],
    keypoint_thresh: float
) -> np.ndarray:
    """在图像上绘制所有检测到的实例的关键点和骨架"""
    canvas = image.copy()
    for det in detections:
        keypoints = np.array(det["keypoints"]) # [17, 3] with (x, y, conf)
        
        # 绘制骨架
        for i, (p1_idx, p2_idx) in enumerate(COCO_SKELETON):
            x1, y1, c1 = keypoints[p1_idx]
            x2, y2, c2 = keypoints[p2_idx]
            if c1 > keypoint_thresh and c2 > keypoint_thresh:
                color = SKELETON_COLORS[i]
                cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
        # 绘制关键点
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > keypoint_thresh:
                cv2.circle(canvas, (int(x), int(y)), 3, (0, 0, 255), -1) # BGR: 红色

    return canvas


def _create_heatmap_visualization(
    image: np.ndarray,
    heatmaps: torch.Tensor
) -> np.ndarray:
    """生成关键点热图的可视化图像"""
    if heatmaps.is_cuda:
        heatmaps = heatmaps.cpu()
        
    # 1. 将所有关键点热图融合成一张合成图 (取最大值)
    composite_heatmap = torch.sigmoid(heatmaps).max(dim=0)[0].numpy()
    
    # 2. 归一化并应用伪彩色
    composite_heatmap = (composite_heatmap - composite_heatmap.min()) / (composite_heatmap.max() - composite_heatmap.min() + 1e-6)
    heatmap_colored = cv2.applyColorMap((composite_heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # 3. 将热图resize到原图大小
    H, W, _ = image.shape
    heatmap_resized = cv2.resize(heatmap_colored, (W, H))
    
    # 4. 将热图与原图叠加
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # OpenCV使用BGR
    overlay = cv2.addWeighted(image_bgr, 0.5, heatmap_resized, 0.5, 0)
    
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def visualize_detections(
    image: Union[str, np.ndarray],
    detections: List[Dict],
    heatmaps: torch.Tensor,
    keypoint_thresh: float = 0.1
) -> np.ndarray:
    """
    主可视化函数，生成左右并排的对比图。
    左图：原图 + 关键点和骨架
    右图：原图 + 关键点热图叠加

    Args:
        image (Union[str, np.ndarray]): 图像路径或已加载的RGB图像 (H, W, 3)。
        detections (List[Dict]): 来自解码器的检测结果列表。
        heatmaps (torch.Tensor): 模型输出的原始关键点热图 [K, Hf, Wf]。
        keypoint_thresh (float): 绘制关键点和骨架的置信度阈值。

    Returns:
        np.ndarray: 拼接后的可视化图像 (RGB格式)。
    """
    if isinstance(image, str):
        image_rgb = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image.copy()

    # 绘制左图：关键点和骨架
    skeleton_viz = _draw_keypoints_and_skeleton(image_rgb, detections, keypoint_thresh)
    
    # 绘制右图：热图叠加
    heatmap_viz = _create_heatmap_visualization(image_rgb, heatmaps)
    
    # 将两张图拼接
    h1, w1, _ = skeleton_viz.shape
    h2, w2, _ = heatmap_viz.shape
    
    # 确保两张图高度一致
    if h1 != h2:
        target_h = max(h1, h2)
        skeleton_viz = cv2.resize(skeleton_viz, (int(w1 * target_h / h1), target_h))
        heatmap_viz = cv2.resize(heatmap_viz, (int(w2 * target_h / h2), target_h))

    combined_image = np.hstack((skeleton_viz, heatmap_viz))
    
    return combined_image


# ---------- [示例] 如何使用此工具 ----------
if __name__ == '__main__':
    # 这是一个演示如何调用 visualize_detections 的示例
    # 您需要根据您的项目结构，提供模型、解码器和测试图片
    
    # 1. 导入您的模型和解码器
    # 假设您的文件都在上一级目录
    import sys
    sys.path.append('..')
    from models.movenet_fpn import MoveNet_FPN, decode_movenet_outputs
    
    # 2. 参数设置 (与您的训练脚本保持一致)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    INPUT_SIZE = 192
    STRIDE = 4
    TEST_IMAGE_PATH = "../../res/girl_with_bags.png" # 请确保路径正确
    
    # 3. 加载模型 (这里仅为演示，不加载预训练权重)
    import timm
    backbone = timm.create_model(
        'mobilenetv3_large_100', pretrained=False, features_only=True, out_indices=(2, 3, 4)
    )
    model = MoveNet_FPN(backbone, upsample_to_quarter=True, out_stride=STRIDE).to(DEVICE)
    model.eval()
    
    # 4. 图像预处理 (与训练时严格一致！)
    image_bgr = cv2.imread(TEST_IMAGE_PATH)
    if image_bgr is None:
        print(f"Error: 无法加载图片，请检查路径: {TEST_IMAGE_PATH}")
    else:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Letterbox + 归一化
        h, w, _ = image_rgb.shape
        scale = INPUT_SIZE / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized_img = cv2.resize(image_rgb, (new_w, new_h))
        
        padded_img = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
        top, left = (INPUT_SIZE - new_h) // 2, (INPUT_SIZE - new_w) // 2
        padded_img[top:top+new_h, left:left+new_w] = resized_img
        
        transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        input_tensor = transform(image=padded_img)['image'].unsqueeze(0).to(DEVICE)
        
        # 5. 模型推理
        with torch.no_grad():
            preds = model(input_tensor)
            
        # 6. 解码
        detections = decode_movenet_outputs(
            preds, 
            img_size=(h, w), # 注意：传入原图尺寸
            stride=STRIDE
        )
        print(f"检测到 {len(detections)} 个人体实例。")
        
        # 7. 可视化
        # 注意：可视化函数需要原图、解码结果和模型原始热图
        viz_image = visualize_detections(
            image=image_rgb,
            detections=detections,
            heatmaps=preds['heatmaps'][0], # 传入batch中的第一张图的热图
            keypoint_thresh=0.1
        )
        
        # 8. 显示和保存
        plt.figure(figsize=(12, 6))
        plt.imshow(viz_image)
        plt.axis('off')
        plt.tight_layout()
        
        save_path = "visualization_result.png"
        plt.savefig(save_path)
        print(f"可视化结果已保存到: {save_path}")
        plt.show()