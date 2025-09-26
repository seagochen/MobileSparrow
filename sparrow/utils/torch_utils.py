import torch
from copy import deepcopy
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.ops import nms
from tqdm import tqdm

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


# =========================
# 1) EMA: 指数移动平均
# =========================
class EMA:
    """
    模型参数的指数移动平均（Exponential Moving Average）。

    作用
    ----
    - 维护一个“影子模型”（ema_model），其权重是在线模型参数的平滑平均。
    - 在验证/导出时使用 EMA 权重，通常更稳定，精度略优。

    用法
    ----
    >>> ema = EMA(model, decay=0.9999)
    >>> for each training step:
    ...     optimizer.step()
    ...     ema.update(model)  # 训练后更新 EMA
    >>> evaluate(ema.ema_model, ...)  # 使用 ema_model 验证/保存

    参数
    ----
    model : nn.Module
        需要被跟踪的在线模型（会 deepcopy 一份）。
    decay : float
        EMA 衰减系数，越接近 1 越“平滑”。此处还叠加了一个随更新步数变化的 warmup 因子。

    备注
    ----
    - 仅对 requires_grad 的参数做 EMA；
    - ema_model 置为 eval() 并冻结梯度（不参与反传）。
    """

    def __init__(self, model, decay=0.9999):
        self.ema_model = deepcopy(model).eval()  # 模型深拷贝（不共享参数）
        self.decay = decay
        self.updates = 0  # 已更新的步数统计

        # EMA 模型不需要梯度
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    def update(self, model):
        """
        用在线模型的当前参数更新 EMA 模型。

        策略
        ----
        - 使用动态衰减：d = decay * (1 - 0.9 ** (updates / 2000))
          前期更“跟随”，后期更“平滑”。

        注意
        ----
        - 仅对 requires_grad 的参数执行 EMA；
        - 假设两边的 named_parameters 能一一对应（同名同结构）。
        """
        self.updates += 1
        d = self.decay * (1 - pow(0.9, self.updates / 2000))  # 动态调整衰减率

        with torch.no_grad():
            model_params = dict(model.named_parameters())
            ema_params = dict(self.ema_model.named_parameters())

            for name, p in model_params.items():
                if p.requires_grad:
                    ema_p = ema_params[name]
                    # ema_p = d * ema_p + (1 - d) * p
                    ema_p.data.mul_(d).add_(p.data, alpha=1 - d)


# =========================
# 2) 验证循环
# =========================
@torch.no_grad()
def evaluate(model, dataloader, criterion, anchor_generator, precomputed_anchors, device):
    """
    在验证集上评估模型，返回 (平均总损失, 平均分类损失, 平均回归损失)。

    输入
    ----
    model : nn.Module
        前向返回 (cls_preds, reg_preds) 的检测模型：
        - cls_preds: [B, A, C]  (未 sigmoid)
        - reg_preds: [B, A, 4]
    dataloader : torch.utils.data.DataLoader
        迭代返回 (imgs, targets, paths)：
        - imgs    : [B, 3, H, W]，已归一化到与训练一致
        - targets : List[Tensor]，长度 B；每项 [N_i, 5]=[cls, x1, y1, x2, y2]
    criterion : nn.Module
        损失计算器（如你上面的 SSDLoss），调用方式：
        loss_cls, loss_reg = criterion(anchors, cls_preds, reg_preds, targets)
    anchor_generator : AnchorGenerator
        仅为接口统一，实际这里不直接用（由 criterion 使用）。
    precomputed_anchors : Tensor
        预先生成好的所有 anchors，形状 [A, 4]（与模型输出对齐）。
    device : str | torch.device
        推理设备。

    备注
    ----
    - 内部会将 model 置为 eval 模式，并在函数结束后恢复 train 模式；
    - 计算的是简单的 batch 平均再对 dataloader 取平均（没有按样本数加权）。
    """
    model.eval()

    total_loss_cls = 0.0
    total_loss_reg = 0.0

    pbar = tqdm(dataloader, desc="[Validating]")
    for imgs, targets, _ in pbar:
        imgs = imgs.to(device)
        targets_on_device = [t.to(device) for t in targets]

        # 前向：要求模型返回 (cls_preds, reg_preds)
        cls_preds, reg_preds = model(imgs)

        # 计算损失（anchors 直接传入 criterion）
        loss_cls, loss_reg = criterion(precomputed_anchors, cls_preds, reg_preds, targets_on_device)

        total_loss_cls += loss_cls.item()
        total_loss_reg += loss_reg.item()

        pbar.set_postfix(cls=f"{loss_cls.item():.4f}", reg=f"{loss_reg.item():.4f}")

    avg_cls_loss = total_loss_cls / len(dataloader)
    avg_reg_loss = total_loss_reg / len(dataloader)
    avg_total_loss = avg_cls_loss + avg_reg_loss

    model.train()  # 恢复训练模式
    return avg_total_loss, avg_cls_loss, avg_reg_loss


# =========================
# 3) 绘图辅助：画框并保存
# =========================
def _draw_and_save(input_img, decoded_boxes, final_scores, final_labels, keep_indices,
                   save_path=None, show=False):
    """
    将筛选后的检测结果绘制在输入图上，并保存/显示。

    输入
    ----
    input_img : np.ndarray
        形状 [H, W, 3]，RGB，当前实现固定为 320x320 的 letterbox 图像。
    decoded_boxes : Tensor
        [K, 4]，xyxy，模型回归解码后的预测框（已经裁剪到 [0,320]）。
    final_scores : Tensor
        [K]，每个候选框的分数（已取 max 类别分数）。
    final_labels : Tensor
        [K]，每个候选框的类别 id（与 COCO_CLASSES 对齐）。
    keep_indices : Tensor
        NMS 后保留的索引（用于从上述三个张量中选择最终框）。
    save_path : str | None
        若提供则保存到该路径（推荐在训练/验证时使用保存方式）。
    show : bool
        是否直接 plt.show()（不推荐在训练循环内使用）。

    细节
    ----
    - 使用固定坐标范围 [0,320]×[0,320] 并反转 y 轴，使图像坐标原点在左上；
    - 防止 notebook/环境的 DPI 设置造成巨大图像，显式指定 dpi。
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # 设定合理 dpi，避免某些环境下渲染过大
    fig, ax = plt.subplots(1, figsize=(12, 9), dpi=100)

    # 明确坐标范围，与 320x320 letterbox 对齐（注意 y 轴反向以贴合图像坐标）
    ax.set_xlim(0, 320)
    ax.set_ylim(320, 0)
    ax.imshow(input_img, extent=[0, 320, 320, 0])

    # 逐框绘制
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

    # 优先保存，其次可选展示；否则关闭图像释放内存
    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
    elif show:
        plt.show()
        plt.close(fig)
    else:
        plt.close(fig)


# =========================
# 4) 单张图片可视化推理
# =========================
@torch.no_grad()
def visualize_predictions(model, image_path, anchor_generator, device, precomputed_anchors,
                          conf_thresh=0.3, nms_thresh=0.45, save_path=None, show=False):
    """
    对单张图片进行推理、阈值筛选、NMS 去重，并将结果绘制保存/显示。

    输入
    ----
    model : nn.Module
        前向返回 (cls_preds, reg_preds) 的检测模型：
        - cls_preds: [1, A, C]  (未 sigmoid)
        - reg_preds: [1, A, 4]
    image_path : str
        待推理的图像路径。
    anchor_generator : AnchorGenerator
        只用于编码/解码坐标（此处用到 xyxy<->cxcywh 转换）。
    device : str | torch.device
        推理设备。
    precomputed_anchors : Tensor
        [A, 4]，与模型输出对齐的 anchors（xyxy，单位像素）。
    conf_thresh : float
        置信度阈值（先对每个 anchor 取 max 类别分数，再与该阈值比较）。
    nms_thresh : float
        NMS 的 IoU 阈值。
    save_path : str | None
        若给出则保存可视化图；否则可选 show=True 直接显示。
    show : bool
        是否使用 plt.show() 显示（训练/批量推理建议使用保存而非展示）。

    处理流程
    ----
    1) 读图并 letterbox 到 320x320（灰色填充到方形，保持比例）；
    2) 前向得到分类与回归输出；
    3) 取每个 anchor 的最大类别分数做第一次阈值筛选；
    4) 将回归量解码成预测框（cxcywh -> xyxy），并进行数值稳健处理 + 裁剪到 [0,320]；
    5) NMS 去重；
    6) 绘制并保存/展示。

    备注
    ----
    - 这里固定推理输入为 320×320，与训练/anchor 设计保持一致；
    - 若你的模型训练时输入可变，请同步修改此处 letterbox 的目标尺寸；
    - 该函数对类别只显示“最大分数对应的类别”，若希望多标签保留可自行扩展。
    """
    model.eval()

    # 1) 读取与 letterbox 到 320x320
    img_bgr = cv2.imread(image_path)
    assert img_bgr is not None, f"Image not found: {image_path}"
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    scale = 320 / max(h, w)
    resized_h, resized_w = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img_rgb, (resized_w, resized_h))

    # 以 114 灰色填充到 320×320
    input_img = np.full((320, 320, 3), 114, dtype=np.uint8)
    pad_h = (320 - resized_h) // 2
    pad_w = (320 - resized_w) // 2
    input_img[pad_h:pad_h+resized_h, pad_w:pad_w+resized_w] = img_resized

    # 归一化到 [0,1] 并转为 [1,3,320,320] Tensor
    img_tensor = (torch.from_numpy(input_img.transpose(2, 0, 1)).float() / 255.0).to(device).unsqueeze(0)

    # 2) 前向（模型需返回 [1, A, C] 与 [1, A, 4]）
    cls_preds, reg_preds = model(img_tensor)  # [1, A, C], [1, A, 4]
    scores = torch.sigmoid(cls_preds[0])      # [A, C]，对每类独立 sigmoid
    reg_deltas = reg_preds[0]                 # [A, 4]

    # 3) 置信度筛选（取每个 anchor 的最大类别分数与其类别索引）
    max_scores, best_class_indices = scores.max(dim=1)  # [A], [A]
    keep = max_scores > conf_thresh
    if keep.sum().item() == 0:
        # 没有有效检测：仍然画/存原图，避免报错
        _draw_and_save(input_img, torch.empty((0, 4), device=device), torch.empty(0, device=device),
                       torch.empty(0, dtype=torch.long, device=device),
                       keep_indices=torch.tensor([], dtype=torch.long),
                       save_path=save_path, show=show)
        model.train()
        return

    # 4) 解码框（确保 anchors 与 deltas 在同设备）
    anchors_dev = precomputed_anchors.to(device)
    selected_anchors = anchors_dev[keep]   # [K, 4]
    selected_deltas  = reg_deltas[keep]    # [K, 4]

    # xyxy -> cxcywh
    anchors_cxcywh = anchor_generator.xyxy_to_cxcywh(selected_anchors)

    # 解码：与训练时 encode 相反
    pred_cx = selected_deltas[:, 0] * anchors_cxcywh[:, 2] + anchors_cxcywh[:, 0]
    pred_cy = selected_deltas[:, 1] * anchors_cxcywh[:, 3] + anchors_cxcywh[:, 1]
    pred_w  = torch.exp(selected_deltas[:, 2]) * anchors_cxcywh[:, 2]
    pred_h  = torch.exp(selected_deltas[:, 3]) * anchors_cxcywh[:, 3]
    decoded_boxes = anchor_generator.cxcywh_to_xyxy(
        torch.stack([pred_cx, pred_cy, pred_w, pred_h], dim=1)
    )  # [K, 4]

    # 4.1) 数值稳健处理：替换 NaN/Inf，并裁剪到图像边界 [0, 320]
    decoded_boxes = torch.nan_to_num(decoded_boxes, nan=0.0, posinf=320.0, neginf=0.0)
    decoded_boxes[:, 0::2] = decoded_boxes[:, 0::2].clamp(0, 320)  # x1/x2
    decoded_boxes[:, 1::2] = decoded_boxes[:, 1::2].clamp(0, 320)  # y1/y2

    final_scores = max_scores[keep].float()            # [K]
    final_labels = best_class_indices[keep].long()     # [K]

    # 5) NMS 去重
    keep_indices = nms(decoded_boxes, final_scores, nms_thresh)

    # 6) 绘制与保存/展示
    _draw_and_save(input_img, decoded_boxes, final_scores, final_labels, keep_indices,
                   save_path=save_path, show=show)

    model.train()  # 恢复训练模式
