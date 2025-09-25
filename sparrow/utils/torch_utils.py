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


# 4. 绘图（固定dpi与坐标系，优先保存，避免show）
def _draw_and_save(input_img, decoded_boxes, final_scores, final_labels, keep_indices,
                   save_path=None, show=False):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # 明确设定一个正常的 dpi，避免环境把 dpi 弄到超大
    fig, ax = plt.subplots(1, figsize=(12, 9), dpi=100)

    # 明确坐标范围 = [0,320]，避免某些异常数值把坐标轴拉爆
    ax.set_xlim(0, 320)
    ax.set_ylim(320, 0)  # 注意图像坐标原点在左上，因此 y 轴反向
    ax.imshow(input_img, extent=[0, 320, 320, 0])

    for i in keep_indices:
        x1, y1, x2, y2 = decoded_boxes[i].detach().cpu().numpy().tolist()
        label_idx = int(final_labels[i].cpu())
        score = float(final_scores[i].cpu())
        label_name = COCO_CLASSES[label_idx] if label_idx < len(COCO_CLASSES) else f'Cls {label_idx}'

        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, max(0, y1 - 5), f'{label_name}: {score:.2f}',
                color='black', bbox=dict(facecolor='lime', alpha=0.8), fontsize=10)

    ax.axis('off')

    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
    elif show:
        # 只在你明确要求时才 show，训练循环里建议不用
        plt.show()
        plt.close(fig)
    else:
        # 默认为了安全直接关闭，避免notebook渲染器接手导致大图
        plt.close(fig)

@torch.no_grad()
def visualize_predictions(model, image_path, anchor_generator, device, precomputed_anchors,
                          conf_thresh=0.3, nms_thresh=0.45, save_path=None, show=False):
    model.eval()

    # 1) 读取与letterbox到 320x320
    img_bgr = cv2.imread(image_path)
    assert img_bgr is not None, f"Image not found: {image_path}"
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    scale = 320 / max(h, w)
    resized_h, resized_w = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img_rgb, (resized_w, resized_h))
    input_img = np.full((320, 320, 3), 114, dtype=np.uint8)
    pad_h = (320 - resized_h) // 2
    pad_w = (320 - resized_w) // 2
    input_img[pad_h:pad_h+resized_h, pad_w:pad_w+resized_w] = img_resized
    img_tensor = (torch.from_numpy(input_img.transpose(2, 0, 1)).float() / 255.0).to(device).unsqueeze(0)

    # 2) 前向
    cls_preds, reg_preds = model(img_tensor)                # [1, A, C], [1, A, 4]
    scores = torch.sigmoid(cls_preds[0])                    # [A, C]
    reg_deltas = reg_preds[0]                               # [A, 4]

    # 3) 置信度与类别
    max_scores, best_class_indices = scores.max(dim=1)      # [A], [A]
    keep = max_scores > conf_thresh
    if keep.sum().item() == 0:
        # 没有有效检测：仍然可保存原图，避免异常
        _draw_and_save(input_img, torch.empty((0,4), device=device), torch.empty(0, device=device),
                       torch.empty(0, dtype=torch.long, device=device), keep_indices=torch.tensor([], dtype=torch.long),
                       save_path=save_path, show=show)
        model.train()
        return

    # 4) 解码框（确保 anchors 在同设备）
    anchors_dev = precomputed_anchors.to(device)
    selected_anchors = anchors_dev[keep]                    # [K, 4]
    selected_deltas  = reg_deltas[keep]                     # [K, 4]

    anchors_cxcywh = anchor_generator.xyxy_to_cxcywh(selected_anchors)
    pred_cx = selected_deltas[:, 0] * anchors_cxcywh[:, 2] + anchors_cxcywh[:, 0]
    pred_cy = selected_deltas[:, 1] * anchors_cxcywh[:, 3] + anchors_cxcywh[:, 1]
    pred_w  = torch.exp(selected_deltas[:, 2]) * anchors_cxcywh[:, 2]
    pred_h  = torch.exp(selected_deltas[:, 3]) * anchors_cxcywh[:, 3]
    decoded_boxes = anchor_generator.cxcywh_to_xyxy(torch.stack([pred_cx, pred_cy, pred_w, pred_h], dim=1))

    # 4.1) 数值稳健处理 + 裁剪到[0,320]
    decoded_boxes = torch.nan_to_num(decoded_boxes, nan=0.0, posinf=320.0, neginf=0.0)
    decoded_boxes[:, 0::2] = decoded_boxes[:, 0::2].clamp(0, 320)
    decoded_boxes[:, 1::2] = decoded_boxes[:, 1::2].clamp(0, 320)

    final_scores = max_scores[keep].float()                 # [K]
    final_labels = best_class_indices[keep].long()          # [K]

    # 5) NMS
    keep_indices = nms(decoded_boxes, final_scores, nms_thresh)

    # 6) 绘图（固定dpi与坐标范围；默认保存，不show）
    _draw_and_save(input_img, decoded_boxes, final_scores, final_labels, keep_indices,
                   save_path=save_path, show=show)

    model.train()
