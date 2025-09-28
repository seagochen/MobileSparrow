# %% [markdown]
# # 损失函数
# 
# 我们已经有了模型和数据加载器，现在最关键的一步就是将它们通过损失函数（Loss Function）连接起来，形成训练的闭环。
# 
# 这是一个非常好的问题。为 MoveNet 定义一个合适的损失函数，关键在于要**分别处理模型的四个不同输出头**，然后将它们的损失加权相加。
# 
# ## 核心思想：分而治之，加权求和
# 
# 我们的 `MoveNetHead` 有四个输出：
# 
# 1.  `heatmaps` (关键点热力图)
# 2.  `centers` (中心点热力图)
# 3.  `regs` (从中心到关键点的回归向量)
# 4.  `offsets` (关键点的亚像素偏移)
# 
# 因此，总的损失函数 `L_total` 将是这四部分损失的加权和：
# 
# $$L_{total} = \lambda_{hm} \cdot L_{hm} + \lambda_{ct} \cdot L_{ct} + \lambda_{reg} \cdot L_{reg} + \lambda_{off} \cdot L_{off}$$
# 
# 其中 `λ` 是每部分损失的权重，是需要调整的超参数。
# 
# 下面我们来逐一分析每个损失函数应该如何设计。
# 
# -----

# %% [markdown]
# ### 1\. 热力图损失 `L_hm` 和中心点损失 `L_ct`
# 
#   - **任务类型**：这两者本质上都是对特征图进行密集的、像素级的二分类任务。对于热力图上的每一个像素点，我们要判断它“是”还是“不是”一个关键点（或中心点）。
#   - **挑战**：存在极其严重的**正负样本不均衡**问题。在一张热力图上，只有寥寥几个像素是正样本（值为1的高斯峰），其他成千上万的像素都是负样本（值为0）。
#   - **最佳选择**：**Focal Loss**。普通的二元交叉熵损失（BCE Loss）在这种不均衡情况下会被海量的负样本所主导，导致模型倾向于把所有点都预测成背景。Focal Loss 通过降低简单负样本的权重，让模型更专注于学习难分类的样本（也就是那些正样本和边界附近的负样本）。

# %% [markdown]
# ### 2\. 回归损失 `L_reg`
# 
#   - **任务类型**：这是一个回归任务，预测从**人体中心点**到**17个关键点**的位移向量 (dx, dy)。
#   - **特点**：这个损失是**稀疏**的。根据 `dataloader.py` 的实现，我们只在人体中心点所在的那个网格 (`cy_i`, `cx_i`) 上写入了真值。因此，我们**只应该在这个位置计算损失**。
#   - **最佳选择**：**L1 Loss** (Mean Absolute Error) 或 **Smooth L1 Loss**。L1 Loss 对异常值不那么敏感，是回归任务中非常稳健和常用的选择。

# %% [markdown]
# ### 3\. 偏移损失 `L_off`
# 
# - **任务类型**：这也是一个回归任务，预测每个关键点在其所在网格内的亚像素偏移。
# - **特点**：这个损失也是**稀疏**的。我们只在每个可见关键点所在的网格上计算损失。
# - **最佳选择**：同样，**L1 Loss** 或 **Smooth L1 Loss** 是最佳选择。
# 
#   
# -----

# %% [markdown]
# ## 代码实现：构建 `MoveNetLoss` 类
# 
# 下面是一个完整的 `MoveNetLoss` 类

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class FocalLoss(nn.Module):
    """
    带 Logits 输入的 Focal Loss 实现。
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
        
        # 计算 pt
        p = torch.sigmoid(pred_logits)
        pt = p * target + (1 - p) * (1 - target)
        
        # 计算 focal loss
        focal_loss = (1 - pt).pow(self.gamma) * bce_loss
        
        # alpha 权重
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_loss = alpha_t * focal_loss
        
        return focal_loss.mean()
    

class MoveNetLoss(nn.Module):
    def __init__(self,
                 focal_loss_alpha: float = 0.25,
                 focal_loss_gamma: float = 2.0,
                 hm_weight: float = 1.0,
                 ct_weight: float = 1.0,
                 reg_weight: float = 2.0,
                 off_weight: float = 1.0):
        super().__init__()
        self.hm_weight = hm_weight
        self.ct_weight = ct_weight
        self.reg_weight = reg_weight
        self.off_weight = off_weight
        
        self.focal_loss = FocalLoss(alpha=focal_loss_alpha, gamma=focal_loss_gamma)
        self.l1_loss = nn.L1Loss(reduction='sum')  # 先求和，再手动除以 mask 数量

class MoveNetLoss(nn.Module):
    def __init__(self,
                 focal_loss_alpha: float = 0.25,
                 focal_loss_gamma: float = 2.0,
                 hm_weight: float = 1.0,
                 ct_weight: float = 1.0,
                 reg_weight: float = 2.0,
                 off_weight: float = 1.0):
        super().__init__()
        self.hm_weight = hm_weight
        self.ct_weight = ct_weight
        self.reg_weight = reg_weight
        self.off_weight = off_weight
        
        self.focal_loss = FocalLoss(alpha=focal_loss_alpha, gamma=focal_loss_gamma)
        self.l1_loss = nn.L1Loss(reduction='sum') # 先求和，再手动除以mask的数量

    def forward(self,
                preds: Dict[str, torch.Tensor],
                labels: torch.Tensor,
                kps_masks: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算 MoveNet 的总损失。
        """
        # 1. 从标签张量中切分出各个真值 (Ground Truth)
        gt_heatmaps = labels[:, :17, :, :]
        gt_centers  = labels[:, 17:18, :, :]
        gt_regs     = labels[:, 18:52, :, :]
        gt_offsets  = labels[:, 52:, :, :]
        
        B, K, H, W = gt_heatmaps.shape
        device = labels.device

        # 2. 计算热力图损失 (L_hm) 和中心点损失 (L_ct)
        loss_hm = self.focal_loss(preds['heatmaps'], gt_heatmaps)
        loss_ct = self.focal_loss(preds['centers'],  gt_centers)

        # 3. 计算回归损失 (L_reg) —— 只在“中心单峰”监督
        with torch.no_grad():
            flat_centers = gt_centers.view(B, -1)
            idx = flat_centers.argmax(dim=1)
            cy = idx // W
            cx = idx % W
            center_1hot = torch.zeros_like(gt_centers)
            center_1hot[torch.arange(B), 0, cy, cx] = 1.0
        
        center_mask_2k = center_1hot.repeat(1, 2*K, 1, 1)
        pred_regs = preds['regs'] * center_mask_2k
        gt_regs_m = gt_regs * center_mask_2k
        denom_reg = center_mask_2k.sum().clamp(min=1.0)
        loss_reg = self.l1_loss(pred_regs, gt_regs_m) / denom_reg

        # 4. 计算偏移损失 (L_off) —— 每个关键点仅在“自身单峰”监督
        with torch.no_grad():
            flat_kpt_hms = gt_heatmaps.view(B, K, -1)
            idx = flat_kpt_hms.argmax(dim=-1)
            ky = idx // W
            kx = idx % W
            kpt_peak_2k = torch.zeros(B, 2*K, H, W, device=device)
            for j in range(K):
                kpt_peak_2k[torch.arange(B), 2*j,   ky[:, j], kx[:, j]] = 1.0
                kpt_peak_2k[torch.arange(B), 2*j+1, ky[:, j], kx[:, j]] = 1.0

        vis_2k = (kps_masks.view(B, K, 1, 1)
                        .repeat(1, 1, 2, H, W)
                        .view(B, 2*K, H, W))
        
        off_mask = kpt_peak_2k * vis_2k
        pred_offsets = preds['offsets'] * off_mask
        gt_offsets_m = gt_offsets * off_mask
        denom_off = off_mask.sum().clamp(min=1.0)
        loss_off = self.l1_loss(pred_offsets, gt_offsets_m) / denom_off
        
        # 5. 总损失
        total_loss = (self.hm_weight * loss_hm +
                    self.ct_weight * loss_ct +
                    self.reg_weight * loss_reg +
                    self.off_weight * loss_off)

        loss_dict = {
            "total_loss":   float(total_loss.detach().item()),
            "loss_heatmap": float(loss_hm.detach().item()),
            "loss_center":  float(loss_ct.detach().item()),
            "loss_regs":    float(loss_reg.detach().item()),
            "loss_offsets": float(loss_off.detach().item()),
        }
        return total_loss, loss_dict


# %% [markdown]
# ## 评估函数

# %%
import torch
from tqdm import tqdm

@torch.no_grad()
def evaluate (
    model,
    dataloader,
    criterion,                 # MoveNetLoss 实例
    device,
    stride: int = 8,           # 你的输出步幅：P3=1/8；若用1/4上采样则改为4
    decoder=None,              # 可选：decode_movenet_outputs 函数，用于算PCK
    pck_alpha: float = 0.05    # PCK@α，阈值=α*max(H_img,W_img)
):
    """
    返回：(avg_total_loss, avg_dict)，若提供 decoder 还会返回 avg_dict['pck@α']。
    avg_dict 含: loss_heatmap / loss_center / loss_regs / loss_offsets / total_loss / (可选) pck@α
    """
    model_was_train = model.training
    model.eval()

    sum_total = 0.0
    sum_hm = 0.0
    sum_ct = 0.0
    sum_reg = 0.0
    sum_off = 0.0
    n_batches = 0

    # 循环外（每个 epoch 初始化）
    epoch_reg_mae = 0.0; epoch_off_mae = 0.0
    epoch_n_reg = 0.0;   epoch_n_off  = 0.0

    # 可选 PCK 统计
    use_pck = decoder is not None
    pck_hit = 0
    pck_cnt = 0
    H_thr = None  # 每张图动态计算阈值 t = α * max(H_img, W_img)

    pbar = tqdm(dataloader, desc="  🟡 [Validating] ")
    for imgs, labels, kps_masks, _ in pbar:
        # 将数据迁移至设备上
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        kps_masks = kps_masks.to(device, non_blocking=True)

        # 前向
        preds = model(imgs)  # 期望返回 dict: heatmaps/centers/regs/offsets，支持batch
        total_loss, loss_dict = criterion(preds, labels, kps_masks)

        # 累计损失
        sum_total += float(loss_dict["total_loss"])
        sum_hm    += float(loss_dict["loss_heatmap"])
        sum_ct    += float(loss_dict["loss_center"])
        sum_reg   += float(loss_dict["loss_regs"])
        sum_off   += float(loss_dict["loss_offsets"])
        n_batches += 1

        # 可选：快速 PCK@α（单人/多人的通用粗评估）
        if use_pck:
            B, _, H_img, W_img = imgs.shape
            thr = pck_alpha * float(max(H_img, W_img))

            # 逐样本解码预测与重建真值关键点
            for b in range(B):
                # 1) 预测：用 decoder 将该样本的输出 -> keypoints [(x,y,conf)*K]
                # 注意 decoder 期望的输入是单样本字典
                single_out = {
                    "heatmaps": preds["heatmaps"][b:b+1],
                    "centers":  preds["centers"][b:b+1],
                    "regs":     preds["regs"][b:b+1],
                    "offsets":  preds["offsets"][b:b+1],
                }
                dets = decoder(single_out, img_size=(H_img, W_img), stride=stride, topk_centers=1)
                if len(dets) == 0:
                    # 该样本一个实例都没解出，跳过统计（或记为全错，按需更改）
                    continue
                pred_kps = dets[0]["keypoints"]  # [(x,y,conf)*K]
                # 2) 真值：从标签热图与 offset 重建 gt 坐标（与你的编码方式一致）
                #    gt heatmaps: [K, Hf, Wf]，gt offsets: [2K, Hf, Wf]
                gt_hm = labels[b, :17]            # [K,Hf,Wf]
                gt_off= labels[b, 52:]            # [2K,Hf,Wf]
                K, Hf, Wf = gt_hm.shape
                vis_mask = kps_masks[b] > 0.5     # [K]

                # 对每个关键点：取热图 argmax 的网格 (gx,gy) + 对应 offset -> 像素坐标
                for j in range(K):
                    if not vis_mask[j]:
                        continue
                    # argmax
                    flat = torch.argmax(gt_hm[j].flatten())
                    gy = int(flat // Wf)
                    gx = int(flat %  Wf)
                    dx = float(gt_off[2*j+0, gy, gx])
                    dy = float(gt_off[2*j+1, gy, gx])
                    gt_x = (gx + dx) * stride
                    gt_y = (gy + dy) * stride

                    # 预测坐标
                    pred_x, pred_y, pred_conf = pred_kps[j]
                    if pred_conf <= 0.0:  # 被阈值过滤掉的点
                        pck_cnt += 1
                        continue
                    # 距离
                    dist = ((pred_x - gt_x)**2 + (pred_y - gt_y)**2) ** 0.5
                    pck_hit += 1 if dist <= thr else 0
                    pck_cnt += 1
                # end-for: 对每个关键点：取热图 argmax 的网格 (gx,gy) + 对应 offset -> 像素坐标
            # end-for: 逐样本解码预测与重建真值关键点
        # end-if: 可选：快速 PCK@α（单人/多人的通用粗评估）

        # 进度条
        pbar.set_postfix(
            tot=f"{sum_total:.6f}",
            hm=f"{sum_hm:.6f}",
            ct=f"{sum_ct:.6f}",
            reg=f"{sum_reg:.6f}",
            off=f"{sum_off:.6f}",
            pck=(f"{(100.0*pck_hit/max(1,pck_cnt)):.2f}%" if use_pck else "N/A")
        )

    avg_total = sum_total / max(1, n_batches)
    avg_dict = {
        "total_loss": avg_total,
        "loss_heatmap": sum_hm / max(1, n_batches),
        "loss_center":  sum_ct / max(1, n_batches),
        "loss_regs":    sum_reg / max(1, n_batches),
        "loss_offsets": sum_off / max(1, n_batches),
    }
    if use_pck:
        avg_dict[f"pck@{pck_alpha:.2f}"] = (pck_hit / max(1, pck_cnt)) if pck_cnt>0 else 0.0

    if model_was_train:
        model.train()  # 恢复训练模式
    return avg_total, avg_dict


# %% [markdown]
# ## 如何在你的训练脚本中使用
# 
# 现在，你可以在你的主训练文件中实例化并使用这个 `MoveNetLoss`。
# 
# ```python
# # from movenet import MoveNet_FPN
# # from dataloader import create_kpts_dataloader
# from loss import MoveNetLoss # 导入我们刚创建的损失类
# 
# # ... (模型、数据加载器、优化器的初始化)
# # model = MoveNet_FPN(...)
# # train_loader = create_kpts_dataloader(...)
# # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# 
# # 实例化损失函数，可以使用默认权重，也可以自定义
# criterion = MoveNetLoss(reg_weight=2.0, off_weight=1.0) 
# 
# model.train()
# for imgs, labels, kps_masks, _ in train_loader:
#     imgs = imgs.to(device)
#     labels = labels.to(device)
#     kps_masks = kps_masks.to(device)
#     
#     # 前向传播
#     preds = model(imgs)
#     
#     # 计算损失
#     total_loss, loss_dict = criterion(preds, labels, kps_masks)
#     
#     # 反向传播和优化
#     optimizer.zero_grad()
#     total_loss.backward()
#     optimizer.step()
#     
#     # 打印日志
#     print(f"Total Loss: {loss_dict['total_loss'].item():.4f}, "
#           f"HM Loss: {loss_dict['loss_heatmap'].item():.4f}, "
#           f"Center Loss: {loss_dict['loss_center'].item():.4f}")
# ```
# 
# 这个 `MoveNetLoss` 类为你处理了所有复杂的逻辑，包括损失函数的选择、真值的切分和稀疏损失的掩码操作，让你的主训练循环变得非常清晰。


