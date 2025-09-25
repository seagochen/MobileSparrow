import torch
from copy import deepcopy
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.ops import nms
from tqdm import tqdm


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


# --- EMA (Exponential Moving Average) ---
class EMA:
    """
    模型参数的指数移动平均。
    它会创建一个模型的“影子”，其权重是历史权重的平滑平均。
    在验证和最终保存模型时，使用 EMA 权重通常能获得更好的性能。
    """
    def __init__(self, model, decay=0.9999):
        self.ema_model = deepcopy(model).eval() # 创建一个模型的深拷贝
        self.decay = decay
        self.updates = 0

        for p in self.ema_model.parameters():
            p.requires_grad_(False) # EMA 模型不需要计算梯度

    def update(self, model):
        self.updates += 1
        d = self.decay * (1 - pow(0.9, self.updates / 2000)) # 动态调整衰减率

        with torch.no_grad():
            # 获取在线模型的参数
            model_params = dict(model.named_parameters())
            # 获取 EMA 模型的参数
            ema_params = dict(self.ema_model.named_parameters())
            
            for name, p in model_params.items():
                if p.requires_grad:
                    # 更新 EMA 参数
                    ema_p = ema_params[name]
                    ema_p.data.mul_(d).add_(p.data, alpha=1 - d)


# --- 验证循环 ---
@torch.no_grad() # 明确表示不需要计算梯度
def evaluate(model, dataloader, criterion, anchor_generator, precomputed_anchors, device):
    """
    在验证集上评估模型。
    返回: (平均总损失, 平均分类损失, 平均回归损失)
    """
    model.eval() # 设置为评估模式
    
    total_loss_cls = 0.0
    total_loss_reg = 0.0
    
    pbar = tqdm(dataloader, desc="[Validating]")
    for imgs, targets, _ in pbar:
        imgs = imgs.to(device)
        targets_on_device = [t.to(device) for t in targets]

        cls_preds, reg_preds = model(imgs)
        
        loss_cls, loss_reg = criterion(precomputed_anchors, cls_preds, reg_preds, targets_on_device)
        
        total_loss_cls += loss_cls.item()
        total_loss_reg += loss_reg.item()
        
        pbar.set_postfix(cls=f"{loss_cls.item():.4f}", reg=f"{loss_reg.item():.4f}")

    avg_cls_loss = total_loss_cls / len(dataloader)
    avg_reg_loss = total_loss_reg / len(dataloader)
    avg_total_loss = avg_cls_loss + avg_reg_loss
    
    model.train() # 恢复为训练模式
    return avg_total_loss, avg_cls_loss, avg_reg_loss


@torch.no_grad()
def visualize_predictions(model, image_path, anchor_generator, device, precomputed_anchors, conf_thresh=0.3, nms_thresh=0.45):
    """
    对单张图片进行预测并可视化结果 (完整版，包含解码逻辑)。
    """
    model.eval()
    
    # 1. 加载和预处理图片 (不变)
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # ... (letterbox 和归一化代码不变) ...
    h, w = img_rgb.shape[:2]
    scale = 320 / max(h, w)
    resized_h, resized_w = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img_rgb, (resized_w, resized_h))
    input_img = np.full((320, 320, 3), 114, dtype=np.uint8)
    pad_h = (320 - resized_h) // 2
    pad_w = (320 - resized_w) // 2
    input_img[pad_h:pad_h+resized_h, pad_w:pad_w+resized_w] = img_resized
    img_tensor = (torch.from_numpy(input_img.transpose(2, 0, 1)).float() / 255.0).to(device).unsqueeze(0)
    
    # 2. 模型推理 (不变)
    cls_preds, reg_preds = model(img_tensor)
    
    # 3. 后处理 (Post-processing)
    scores = torch.sigmoid(cls_preds[0])
    reg_deltas = reg_preds[0] # [Num_Anchors, 4]

    max_scores, best_class_indices = scores.max(dim=1)
    keep = max_scores > conf_thresh
    
    # --- 核心改动：解码边界框 ---
    # 筛选出置信度符合要求的锚框和回归偏移量
    selected_anchors = precomputed_anchors[keep]
    selected_deltas = reg_deltas[keep]
    
    # 将锚框从 (x1, y1, x2, y2) 转换为 (cx, cy, w, h)
    anchors_cxcywh = anchor_generator.xyxy_to_cxcywh(selected_anchors)
    
    # 解码：应用模型预测的偏移量
    pred_cx = selected_deltas[:, 0] * anchors_cxcywh[:, 2] + anchors_cxcywh[:, 0]
    pred_cy = selected_deltas[:, 1] * anchors_cxcywh[:, 3] + anchors_cxcywh[:, 1]
    pred_w = torch.exp(selected_deltas[:, 2]) * anchors_cxcywh[:, 2]
    pred_h = torch.exp(selected_deltas[:, 3]) * anchors_cxcywh[:, 3]
    
    # 将解码后的 (cx, cy, w, h) 坐标转换回 (x1, y1, x2, y2)
    decoded_boxes = anchor_generator.cxcywh_to_xyxy(torch.stack([pred_cx, pred_cy, pred_w, pred_h], dim=1))

    # -----------------------------
    
    final_scores = max_scores[keep]
    final_labels = best_class_indices[keep]
    
    # NMS (现在使用解码后的精确框)
    keep_indices = nms(decoded_boxes, final_scores, nms_thresh)
    
    # 4. 绘图
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(input_img)
    
    for i in keep_indices:
        box = decoded_boxes[i].cpu().numpy()
        label_idx = int(final_labels[i].cpu())
        score = float(final_scores[i].cpu())
        
        # 使用 COCO 类别名称
        label_name = COCO_CLASSES[label_idx] if label_idx < len(COCO_CLASSES) else f'Cls {label_idx}'
        
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        plt.text(x1, y1 - 5, f'{label_name}: {score:.2f}', color='black',
                 bbox=dict(facecolor='lime', alpha=0.8))
        
    plt.axis('off')
    plt.show()
    model.train() # 恢复训练模式