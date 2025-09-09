import cv2
import torch
import numpy as np
from typing import Tuple, Dict

# 从您现有的 task.py 中导入解码函数
# 确保这个导入路径相对于您的 movenet_cli.py 是正确的
from .task.task import movenetDecode

def letterbox(img: np.ndarray, 
              dst_size: int, 
              color=(114, 114, 114)) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    将图像进行 letterbox 缩放填充，并返回转换所需参数。

    Args:
        img (np.ndarray): 原始图像 (H, W, C)，RGB格式。
        dst_size (int): 目标正方形尺寸。
        color (tuple, optional): 填充颜色. Defaults to (114, 114, 114).

    Returns:
        Tuple[np.ndarray, float, Tuple[int, int]]:
            - new_img: letterbox 处理后的图像 (dst_size, dst_size, C)。
            - scale: 缩放比例。
            - (pad_w, pad_h): 填充的偏移量。
    """
    h, w = img.shape[:2]
    scale = float(dst_size) / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    
    img_resz = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    new_img = np.full((dst_size, dst_size, 3), color, dtype=np.uint8)
    pad_w = (dst_size - nw) // 2
    pad_h = (dst_size - nh) // 2
    new_img[pad_h:pad_h + nh, pad_w:pad_w + nw] = img_resz
    
    return new_img, scale, (pad_w, pad_h)


def preprocess(image_path: str, img_size: int) -> Tuple[torch.Tensor, np.ndarray]:
    """
    加载并预处理单张图像。

    Args:
        image_path (str): 图像文件路径。
        img_size (int): 模型输入尺寸。

    Returns:
        Tuple[torch.Tensor, np.ndarray]:
            - image_tensor: 准备好输入模型的张量。
            - original_image: 加载的原始图像 (BGR格式)。
    """
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # BGR -> RGB
    image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Letterbox
    image_letterboxed, _, _ = letterbox(image_rgb, img_size)

    # 归一化并转换为 Tensor
    image_tensor = (image_letterboxed.astype(np.float32) / 255.0)
    image_tensor = np.transpose(image_tensor, (2, 0, 1))  # HWC -> CHW
    image_tensor = torch.from_numpy(image_tensor).unsqueeze(0) # 添加 batch 维度

    return image_tensor, original_image


def postprocess(predictions: np.ndarray, 
                original_image: np.ndarray, 
                img_size: int) -> np.ndarray:
    """
    对模型输出进行后处理，并将坐标转换回原始图像尺寸。

    Args:
        predictions (np.ndarray): movenetDecode 解码后的坐标 (1, 34)，归一化到 [0,1]。
        original_image (np.ndarray): 原始图像。
        img_size (int): 模型输入尺寸。

    Returns:
        np.ndarray: 在原始图像上绘制了关键点的图像。
    """
    h_orig, w_orig = original_image.shape[:2]

    # Letterbox 参数需要重新计算以进行逆转换
    scale = float(img_size) / max(h_orig, w_orig)
    nh_letterbox, nw_letterbox = int(round(h_orig * scale)), int(round(w_orig * scale))
    pad_w = (img_size - nw_letterbox) // 2
    pad_h = (img_size - nh_letterbox) // 2

    # 复制原始图像用于绘制
    output_image = original_image.copy()

    # 遍历所有17个关节点
    for i in range(predictions.shape[1] // 2):
        # 预测的坐标是相对于 192x192 的
        x_pred_norm = predictions[0, i * 2]
        y_pred_norm = predictions[0, i * 2 + 1]

        if x_pred_norm < 0 or y_pred_norm < 0:
            continue

        # 1. 将归一化坐标转换为 letterbox 图像 (192x192) 上的像素坐标
        x_letterbox = x_pred_norm * img_size
        y_letterbox = y_pred_norm * img_size

        # 2. 逆向转换：将 letterbox 坐标转换为原始图像坐标
        #   x_orig = (x_letterbox - pad_w) / scale
        #   y_orig = (y_letterbox - pad_h) / scale
        x_original = int((x_letterbox - pad_w) / scale)
        y_original = int((y_letterbox - pad_h) / scale)
        
        # 绘制到 output_image 上
        cv2.circle(output_image, (x_original, y_original), 4, (0, 0, 255), -1) # BGR
        cv2.circle(output_image, (x_original, y_original), 5, (255, 255, 255), 1)

    return output_image


def run_prediction(model, 
                   image_path: str, 
                   device: torch.device,
                   cfg: dict) -> np.ndarray:
    """
    对单张图像执行完整的预测流程。

    Args:
        model: 加载好的 PyTorch 模型。
        image_path (str): 图像路径。
        device: 'cuda' or 'cpu'。
        cfg (dict): 配置字典。

    Returns:
        np.ndarray: 绘制好关键点的图像。
    """
    img_size = cfg['img_size']

    # 1. 预处理
    image_tensor, original_image = preprocess(image_path, img_size)
    image_tensor = image_tensor.to(device)

    # 2. 模型推理
    with torch.no_grad():
        # 假设您的模型输出是一个字典
        output_dict = model(image_tensor)
        # 如果模型输出是 list/tuple，则需要手动包装成 dict
        # output_list = model(image_tensor)
        # output_dict = {"heatmaps": output_list[0], "centers": output_list[1], "regs": output_list[2], "offsets": output_list[3]}

    # 3. 解码
    # 注意：movenetDecode 需要一个对齐步骤，您的 task.py 中应该有类似 _align_output_... 的函数
    # 这里为了简化，我们假设模型输出的特征图尺寸已经是我们需要的 (img_size/4)
    # 如果不是，您需要在这里插入对齐逻辑
    predictions_norm = movenetDecode(output_dict, mode='output', num_joints=cfg['num_classes'])

    # 4. 后处理和坐标逆转换
    result_image = postprocess(predictions_norm, original_image, img_size)

    return result_image