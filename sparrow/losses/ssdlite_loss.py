# %% [markdown]
# # 损失函数
# 
# 好的，我们已经准备好了现代化的 `SSDLite_FPN` 模型和强大的 `Albumentations` 数据加载器。现在，最后一块拼图就是为这个目标检测模型设计和实现一个合适的损失函数 (Loss Function)。
# 
# 对于像 SSD、RetinaNet 或我们这个 `SSDLite_FPN` 这样的单阶段（One-Stage）检测器，损失函数通常由两部分组成：
# 
# 1.  **分类损失 (Classification Loss)**：惩罚预测框的类别错误。
# 2.  **定位损失 (Localization/Regression Loss)**：惩罚预测框与真实框（Ground Truth）之间的位置偏差。
# 
# 总损失是这两部分损失的加权和：$L\_{total} = L\_{cls} + \\alpha L\_{loc}$ （权重 $\\alpha$ 通常设为1）。
# 
# 在计算这两个损失之前，最关键的一步是**目标分配（Target Assignment）**，也就是为模型生成的成千上万个锚框（Anchor Boxes）中的每一个，分配一个真实的标签（是背景，还是某个物体？如果是物体，对应的真实框是哪一个？）。
# 
# 下面，我将为你分步实现一个完整的、适用于 `SSDLite_FPN` 的损失函数模块。
# 
# -----

# %% [markdown]
# ## 核心步骤
# 
# 我们将创建一个 `SSDLoss` 类，其核心逻辑分为三步：
# 
# 1.  **生成锚框 (Anchor Generation)**：为模型输出的每个尺寸的特征图（P3, P4, P5, P6, P7）预先生成一组固定的锚框。这一步在初始化时完成。
# 2.  **目标分配 (Target Assignment)**：在每次训练迭代中，根据真实框（Ground Truth Boxes）和所有锚框的 IoU（交并比），为每个锚框分配匹配的真实物体或将其标记为背景。
# 3.  **计算损失 (Loss Calculation)**：
#       * **分类损失**：使用 **Focal Loss**。这是现代检测器中非常流行且有效的损失函数，它能自动关注于难分类的样本（hard examples），解决了正负样本（物体 vs. 背景）极度不平衡的问题。
#       * **定位损失**：使用 **Smooth L1 Loss**。这是一种对离群值不那么敏感的回归损失，比 L2 Loss 更鲁棒。我们只对被分配为正样本（匹配到物体）的锚框计算定位损失。
# 
# -----

# %% [markdown]
# ### 第一步：安装依赖
# 
# 我们需要 `torchvision` 来方便地计算 IoU 和损失。同时使用 `tqdm` 来显示训练进度和错误率。

# %%
# !pip -q install torchvision
# !pip -q install tqdm

# %% [markdown]
# ### 第二步：锚框生成器 (Anchor Generator)
# 
# 我们需要一个辅助类来为不同尺寸的特征图生成锚框。

# %%
# anchor_utils.py
import torch
import math
from typing import List

class AnchorGenerator:
    """
    为 FPN 输出的多个特征图生成锚框。
    这个类现在主要在训练开始前，用于“预计算”步骤。
    """
    def __init__(self,
                 sizes=(32, 64, 128, 256, 512),
                 aspect_ratios=(0.5, 1.0, 2.0, 1/3.0, 3.0)):
        super().__init__()
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.num_anchors_per_location = len(self.aspect_ratios)
        self.cell_anchors = self._generate_cell_anchors()

    def _generate_cell_anchors(self) -> List[torch.Tensor]:
        cell_anchors = []
        for s in self.sizes:
            w = torch.tensor([s * math.sqrt(ar) for ar in self.aspect_ratios])
            h = torch.tensor([s / math.sqrt(ar) for ar in self.aspect_ratios])
            # base_anchors = torch.stack([-w / 2, -h / 2, w / 2, h / 2], dim=1)
            # base_anchors = self.cxcywh_to_xyxy(base_anchors)
            # cell_anchors.append(base_anchors)

            # 按 cx,cy,w,h 构，再转 xyxy
            base_cxcywh = torch.stack([torch.zeros_like(w), torch.zeros_like(h), w, h], dim=1)
            base_anchors = self.cxcywh_to_xyxy(base_cxcywh)
            cell_anchors.append(base_anchors)

        return cell_anchors

    def cxcywh_to_xyxy(self, boxes: torch.Tensor) -> torch.Tensor:
        return torch.cat([boxes[:, :2] - boxes[:, 2:] / 2,
                          boxes[:, :2] + boxes[:, 2:] / 2], dim=1)
    
    def xyxy_to_cxcywh(self, boxes: torch.Tensor) -> torch.Tensor:
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        cx = boxes[:, 0] + w / 2
        cy = boxes[:, 1] + h / 2
        return torch.stack([cx, cy, w, h], dim=1)

    def generate_anchors_on_grid(self, feature_maps: List[torch.Tensor], device: str) -> torch.Tensor:
        all_anchors = []
        # 假设输入是 320x320，如果你的输入尺寸变化，需要相应调整
        input_size_h, input_size_w = 320, 320
        for i, fm in enumerate(feature_maps):
            fm_h, fm_w = fm.shape[-2:]
            stride_h = input_size_h / fm_h
            stride_w = input_size_w / fm_w
            
            shifts_x = torch.arange(0, fm_w, device=device) * stride_w
            shifts_y = torch.arange(0, fm_h, device=device) * stride_h
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
            shifts = torch.stack((shift_x.flatten(), shift_y.flatten(), shift_x.flatten(), shift_y.flatten()), dim=1)

            anchors = (self.cell_anchors[i].to(device).view(1, -1, 4) + shifts.view(-1, 1, 4)).reshape(-1, 4)
            all_anchors.append(anchors)
        
        return torch.cat(all_anchors, dim=0)

# %% [markdown]
# ### 第三步：完整的损失函数类 `SSDLoss`
# 
# 这个类将包含目标分配和损失计算的所有逻辑。

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou, sigmoid_focal_loss
from typing import List

# 假设 AnchorGenerator 在同一个文件中或已导入
# from anchor_utils import AnchorGenerator

class SSDLoss(nn.Module):
    def __init__(self, 
                 num_classes: int,
                 iou_threshold_pos: float = 0.5,
                 iou_threshold_neg: float = 0.4):
        super().__init__()
        self.num_classes = num_classes
        self.iou_threshold_pos = iou_threshold_pos
        self.iou_threshold_neg = iou_threshold_neg
        
        # AnchorGenerator 实例仍然需要，用于坐标变换等辅助功能
        self.anchor_generator = AnchorGenerator(
            sizes=(32, 64, 128, 256, 512),
            aspect_ratios=(0.5, 1.0, 2.0, 1/3.0, 3.0)
        )
        self.num_anchors_per_loc = self.anchor_generator.num_anchors_per_location

    def assign_targets_to_anchors(self, anchors: torch.Tensor, targets: List[torch.Tensor]):
        # 这个函数完全不变
        batch_size = len(targets)
        num_anchors = anchors.shape[0]
        device = anchors.device
        labels = torch.full((batch_size, num_anchors), self.num_classes, dtype=torch.int64, device=device)
        matched_gt_boxes = torch.zeros((batch_size, num_anchors, 4), dtype=torch.float32, device=device)
        for i in range(batch_size):
            gt_boxes = targets[i][:, 1:]
            gt_labels = targets[i][:, 0]
            if gt_boxes.shape[0] == 0: continue
            iou = box_iou(gt_boxes, anchors)
            max_iou_per_gt, max_iou_idx_per_gt = iou.max(dim=1)
            labels[i, max_iou_idx_per_gt] = gt_labels.to(torch.int64)
            matched_gt_boxes[i, max_iou_idx_per_gt] = gt_boxes
            max_iou_per_anchor, max_iou_idx_per_anchor = iou.max(dim=0)
            pos_mask = max_iou_per_anchor >= self.iou_threshold_pos
            labels[i, pos_mask] = gt_labels[max_iou_idx_per_anchor[pos_mask]].to(torch.int64)
            matched_gt_boxes[i, pos_mask] = gt_boxes[max_iou_idx_per_anchor[pos_mask]]
        return labels, matched_gt_boxes

    def encode_bbox(self, anchors: torch.Tensor, gt_boxes: torch.Tensor):
        # 这个函数完全不变
        anchors_cxcywh = self.anchor_generator.xyxy_to_cxcywh(anchors)
        gt_boxes_cxcywh = self.anchor_generator.xyxy_to_cxcywh(gt_boxes)
        tx = (gt_boxes_cxcywh[:, 0] - anchors_cxcywh[:, 0]) / anchors_cxcywh[:, 2]
        ty = (gt_boxes_cxcywh[:, 1] - anchors_cxcywh[:, 1]) / anchors_cxcywh[:, 3]
        tw = torch.log(gt_boxes_cxcywh[:, 2] / anchors_cxcywh[:, 2])
        th = torch.log(gt_boxes_cxcywh[:, 3] / anchors_cxcywh[:, 3])
        return torch.stack([tx, ty, tw, th], dim=1)

    # def forward(self, anchors: torch.Tensor, cls_preds: torch.Tensor, reg_preds: torch.Tensor, targets: List[torch.Tensor]):
    #     """
    #     计算总损失。
    #     - anchors:   [总锚框数, 4] 预先计算好的、在指定设备上的锚框
    #     - cls_preds: [B, 总锚框数, num_classes] 分类预测
    #     - reg_preds: [B, 总锚框数, 4]       回归预测
    #     - targets:   List of Tensors, 每个 Tensor [N_i, 5] (cls, x1, y1, x2, y2)
    #     """
    #     device = cls_preds.device
        
    #     # 1. 动态生成锚框的步骤被移除，因为 anchors 是直接传入的
    #     #    anchors = self.anchor_generator.generate_anchors_on_grid(feature_maps, device) <-- 移除

    #     # 2. 目标分配 (使用传入的 anchors)
    #     assigned_labels, assigned_gt_boxes = self.assign_targets_to_anchors(anchors.to(device), targets)
        
    #     # 3. 准备计算损失 (后续逻辑完全不变)
    #     pos_mask = (assigned_labels < self.num_classes) & (assigned_labels >= 0)
    #     num_pos = pos_mask.sum().item()
        
    #     # --- 分类损失 (Focal Loss) ---
    #     target_classes_one_hot = F.one_hot(assigned_labels, num_classes=self.num_classes + 1)
    #     target_classes_one_hot = target_classes_one_hot[..., :self.num_classes].float()
    #     loss_cls = sigmoid_focal_loss(
    #         cls_preds, target_classes_one_hot, alpha=0.25, gamma=2.0, reduction='sum'
    #     )

    #     # --- 定位损失 (Smooth L1 Loss) ---
    #     if num_pos > 0:
    #         pos_reg_preds = reg_preds[pos_mask]
    #         pos_assigned_gt_boxes = assigned_gt_boxes[pos_mask]
    #         pos_anchors = anchors.to(device).unsqueeze(0).expand_as(assigned_gt_boxes)[pos_mask]
    #         target_deltas = self.encode_bbox(pos_anchors, pos_assigned_gt_boxes)
    #         loss_reg = F.smooth_l1_loss(pos_reg_preds, target_deltas, beta=1.0, reduction='sum')
    #     else:
    #         loss_reg = torch.tensor(0.0, device=device)

    #     # 归一化
    #     loss_cls = loss_cls / max(1, num_pos)
    #     loss_reg = loss_reg / max(1, num_pos)
        
    #     return loss_cls, loss_reg

    def forward(self, 
            anchors: torch.Tensor, 
            cls_preds: torch.Tensor, 
            reg_preds: torch.Tensor, 
            targets: List[torch.Tensor]):
        """
        计算总损失。
        - anchors:   [总锚框数, 4] 预先计算好的、在指定设备上的锚框
        - cls_preds: [B, 总锚框数, num_classes] 分类预测
        - reg_preds: [B, 总锚框数, 4]       回归预测
        - targets:   List of Tensors, 每个 Tensor [N_i, 5] (cls, x1, y1, x2, y2)
        """
        device = cls_preds.device

        # 2. 目标分配 (使用传入的 anchors)
        assigned_labels, assigned_gt_boxes = self.assign_targets_to_anchors(anchors.to(device), targets)

        # 3. 准备计算损失 (后续逻辑完全不变)
        pos_mask = (assigned_labels < self.num_classes) & (assigned_labels >= 0)
        num_pos = pos_mask.sum().item()

        # === Classification Loss: Stabilized using 'mean' reduction ===
        target_one_hot = F.one_hot(assigned_labels, num_classes=self.num_classes + 1)
        target_one_hot = target_one_hot[..., :self.num_classes].float()

        loss_cls_all = sigmoid_focal_loss(
            cls_preds, target_one_hot, alpha=0.25, gamma=2.0, reduction='none'  # [B,A,C]
        )
        loss_cls = loss_cls_all.sum(dim=2).mean()  # 先按类别求和，再在B×A上取平均

        # === Regression Loss: Calculation remains the same ===
        if num_pos > 0:
            pos_reg_preds = reg_preds[pos_mask]
            pos_gt_boxes = assigned_gt_boxes[pos_mask]
            pos_anchors = anchors.to(device).unsqueeze(0).expand_as(assigned_gt_boxes)[pos_mask]

            target_deltas = self.encode_bbox(pos_anchors, pos_gt_boxes)
            loss_reg = F.smooth_l1_loss(pos_reg_preds, target_deltas, beta=1.0, reduction='sum')
            # Still normalize regression loss by the number of positive samples
            loss_reg = loss_reg / num_pos
        else:
            loss_reg = torch.tensor(0.0, device=device)

        return loss_cls, loss_reg

# %% [markdown]
# ### 第四步：如何将所有部分整合到训练循环中
# 
# 现在你可以将这个 `SSDLoss` 类与你的模型和数据加载器一起使用。
# 
# 
# ```python
# 
# import timm
# from tqdm import tqdm
# from ssdlite_fpn import SSDLite_FPN
# from dataloader import create_dets_dataloader
# 
# # --- 1. 实例化所有组件 ---
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# INPUT_SIZE = 320
# BATCH_SIZE = 8
# NUM_CLASSES = 80 # COCO
# ANCHOR_SIZES = [32, 64, 128, 256, 512]
# ANCHOR_RATIOS = [0.5, 1.0, 2.0, 1/3.0, 3.0]
# 
# # 模型 (接口保持干净，无需改动)
# backbone_fpn = timm.create_model('mobilenetv3_large_100', pretrained=True, features_only=True, out_indices=(2, 3, 4))
# model_fpn = SSDLite_FPN(backbone_fpn, num_classes=NUM_CLASSES, fpn_out_channels=128, num_anchors=len(ANCHOR_RATIOS))
# model_fpn.to(device)
# 
# # 数据加载器 (来自 dataloader.py)
# train_aug_config = { "rotate_deg": 15.0, "min_box_size": 2.0 }
# train_loader = create_dets_dataloader(
#     dataset_root="/home/user/projects/MobileSparrow/data/coco2017" ,
#     img_size=320,
#     batch_size=8,
#     num_workers=4,
#     pin_memory=True,
#     aug_cfg=train_aug_config,
#     is_train=True
# )
# 
# # 损失函数 (使用我们重构后的版本)
# criterion = SSDLoss(num_classes=NUM_CLASSES)
# 
# # 4. 优化器
# optimizer = torch.optim.AdamW(model_fpn.parameters(), lr=1e-4, weight_decay=1e-3)
# 
# 
# # --- 2. 预计算锚框 (核心步骤) ---
# print("Pre-computing anchors for fixed input size...")
# anchor_generator = AnchorGenerator(
#     sizes=ANCHOR_SIZES,
#     aspect_ratios=ANCHOR_RATIOS
# )
# 
# # 创建一个虚拟输入
# dummy_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE).to(device)
# 
# # 设置为 eval 模式，并确保没有梯度计算
# model_fpn.eval()
# with torch.no_grad():
#     # 手动执行一次特征提取流程，以获取特征图尺寸
#     features = model_fpn.backbone(dummy_input)
#     p3, p4, p5 = model_fpn.fpn(features)
#     p6 = model_fpn.extra_layers[0](p5)
#     p7 = model_fpn.extra_layers[1](p6)
#     feature_maps_for_size_calc = [p3, p4, p5, p6, p7]
# 
# # 使用获取的特征图列表生成一次性的、完整的锚框网格
# # 这个 precomputed_anchors 将在整个训练过程中被重复使用
# precomputed_anchors = anchor_generator.generate_anchors_on_grid(feature_maps_for_size_calc, device)
# print(f"Anchors pre-computed. Shape: {precomputed_anchors.shape}")
# 
# 
# # --- 3. 训练循环 ---
# print("\n--- Starting Training ---")
# model_fpn.train() # 切换回训练模式
# 
# # 定义日志打印的频率
# LOG_INTERVAL_SAMPLES = 1000
# log_interval_batches = max(1, LOG_INTERVAL_SAMPLES // BATCH_SIZE)
# print(f"Logging average loss every {log_interval_batches} batches.")
# 
# for epoch in range(5):
#     # --- 为每个 epoch 初始化累加器 ---
#     epoch_loss_cls = 0.0
#     epoch_loss_reg = 0.0
#     
#     # 使用 tqdm 包装数据加载器，以创建进度条
#     pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{5-1} [Training]")
#     
#     for i, (imgs, targets, _) in enumerate(pbar):
#         imgs = imgs.to(device)
#         targets_on_device = [t.to(device) for t in targets]
# 
#         # 前向传播
#         cls_preds, reg_preds = model_fpn(imgs)
# 
#         # 计算损失
#         loss_cls, loss_reg = criterion(precomputed_anchors, cls_preds, reg_preds, targets_on_device)
#         total_loss = loss_cls + loss_reg
# 
#         # 反向传播和优化
#         optimizer.zero_grad()
#         total_loss.backward()
#         optimizer.step()
# 
#         # --- 更新累加器 ---
#         # .item() 可以将只有一个元素的 tensor 转换为 python 数字，并释放计算图
#         epoch_loss_cls += loss_cls.item()
#         epoch_loss_reg += loss_reg.item()
#         
#         # --- 更新 tqdm 进度条的后缀信息，实时显示当前批次的损失 ---
#         pbar.set_postfix(
#             cls=f"{loss_cls.item():.4f}", 
#             reg=f"{loss_reg.item():.4f}", 
#             total=f"{total_loss.item():.4f}"
#         )
#         
#     # --- 每个 epoch 结束后，打印该 epoch 的总结 ---
#     avg_epoch_cls_loss = epoch_loss_cls / len(train_loader)
#     avg_epoch_reg_loss = epoch_loss_reg / len(train_loader)
#     avg_epoch_total_loss = avg_epoch_cls_loss + avg_epoch_reg_loss
#     
#     print(f"\n---> Epoch {epoch} Summary <---")
#     print(f"  Average Classification Loss: {avg_epoch_cls_loss:.4f}")
#     print(f"  Average Regression Loss:   {avg_epoch_reg_loss:.4f}")
#     print(f"  Average Total Loss:        {avg_epoch_total_loss:.4f}\n")
# 
# print("--- Training Finished ---")
# ```


