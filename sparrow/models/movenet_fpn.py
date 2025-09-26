# %% [markdown]
# # 构建一个MoveNet姿态检测模型
# 
# FPN + MobileNet 作为骨架，可以实现很多CV任务，本次我们将实现一个高精度的轻量级的姿态检测模型—— `MoveNet`。
# 
# 与 `003_SSDLite` 相似，本次我们只开发一个新的检测头，然后接入已有的架构。

# %% [markdown]
# ## 复用FPN Neck模块

# %%
import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class FPN(nn.Module):
    """
    Feature Pyramid Network (FPN) Neck
    对 Backbone 输出的 (C3, C4, C5) 特征进行融合，生成 (P3, P4, P5)。
    """
    def __init__(self, in_channels: Tuple[int, int, int], out_channels: int = 256):
        super().__init__()
        c3_in, c4_in, c5_in = in_channels
        
        # 1. 侧向连接的 1x1 卷积，用于统一通道数
        self.lateral_conv3 = nn.Conv2d(c3_in, out_channels, kernel_size=1)
        self.lateral_conv4 = nn.Conv2d(c4_in, out_channels, kernel_size=1)
        self.lateral_conv5 = nn.Conv2d(c5_in, out_channels, kernel_size=1)
        
        # 2. 输出处理的 3x3 卷积，用于平滑特征
        self.output_conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.output_conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.output_conv5 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c3, c4, c5 = features
        
        # 自顶向下路径
        p5_in = self.lateral_conv5(c5)
        
        p4_in = self.lateral_conv4(c4)
        p5_up = F.interpolate(p5_in, size=p4_in.shape[-2:], mode='nearest')
        p4_in = p4_in + p5_up
        
        p3_in = self.lateral_conv3(c3)
        p4_up = F.interpolate(p4_in, size=p3_in.shape[-2:], mode='nearest')
        p3_in = p3_in + p4_up
        
        # 输出卷积
        p3 = self.output_conv3(p3_in)
        p4 = self.output_conv4(p4_in)
        p5 = self.output_conv5(p5_in)
        
        return p3, p4, p5

# %% [markdown]
# ## 构建MoveNet检测头

# %%
from typing import Dict


class MoveNetHead(nn.Module):
    """
    四个分支：
      - heatmaps: [B, K, H, W]   (关键点K的高斯/逻辑热力图)
      - centers:  [B, 1, H, W]   (人体中心/实例中心热力图，若用Center-based grouping)
      - regs:     [B, 2, H, W]   (从中心到人体框的回归，或从peak到root的偏移，按你训练目标定义)
      - offsets:  [B, 2K, H, W]  (关键点子像素偏移，用于提升定位精度)
    """
    def __init__(self, in_ch: int, num_joints: int = 17, midc: int = 32):
        super().__init__()
        # 一个轻量瓶颈：DWConv + PWConv
        def dw_pw(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, in_c, 3, padding=1, groups=in_c, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_c, out_c, 1, bias=True)
            )
        # 共享干路（可选），或为每个头独立一条小塔；这里给每个头单独塔以解耦
        self.hm_tower = dw_pw(in_ch, midc)
        self.ct_tower = dw_pw(in_ch, midc)
        self.reg_tower = dw_pw(in_ch, midc)
        self.off_tower = dw_pw(in_ch, midc)

        self.hm = nn.Conv2d(midc, num_joints, 1)
        self.ct = nn.Conv2d(midc, 1, 1)
        self.reg = nn.Conv2d(midc, 2, 1)
        self.off = nn.Conv2d(midc, num_joints * 2, 1)

        # 让 heatmap 初始较低，利于稳定训练（可选）
        nn.init.constant_(self.hm.bias, -2.0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "heatmaps": self.hm(self.hm_tower(x)),
            "centers":  self.ct(self.ct_tower(x)),
            "regs":     self.reg(self.reg_tower(x)),
            "offsets":  self.off(self.off_tower(x)),
        }


# %% [markdown]
# ## 构建MoveNet模型

# %%
class MoveNet_FPN(nn.Module):
    def __init__(self,
                 backbone,
                 num_joints: int = 17,
                 fpn_out_channels: int = 128,
                 head_midc: int = 32,
                 out_stride: int = 8,               # 以 P3 为 1/8 输出
                 upsample_to_quarter: bool = False  # 如需 1/4 分辨率，可启用上采样
                 ):
        super().__init__()
        self.backbone = backbone

        # 1. 初始化 FPN Neck
        backbone_channels = self.backbone.feature_info.channels()
        self.fpn = FPN(in_channels=backbone_channels, out_channels=fpn_out_channels)  # 直接复用你文件里的 FPN

        # 2. 使用 FPN 输出的P3作为输入
        self.head = MoveNetHead(in_ch=fpn_out_channels, num_joints=num_joints, midc=head_midc)

        self.out_stride = out_stride
        self.upsample_to_quarter = upsample_to_quarter

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # C3,C4,C5
        features = self.backbone(x)                       # [C3, C4, C5]
        # P3,P4,P5
        p3, _, _ = self.fpn(features)                     # 只取 P3（1/8分辨率）

        # 可选：对 P3 再上采样到 1/4 分辨率，很多姿态模型这么做以提升精度
        if self.upsample_to_quarter:
            p3 = F.interpolate(p3, scale_factor=2.0, mode='bilinear', align_corners=False)

        return self.head(p3)


# %% [markdown]
# ## 模型向前传播实验
# 
# 我们测试这个模型看看有无问题。

# %%
def model_runnable_test():

    # 1. 实例化 Backbone，注意修改 out_indices 来获取 C3, C4, C5
    backbone_mbv3 = timm.create_model(
        'mobilenetv3_large_100',
        pretrained=True,
        features_only=True,
        out_indices=(2, 3, 4), # <-- 核心改动：获取 stride=8, 16, 32 的特征
    )

    print("Backbone for FPN created. Output channels:", backbone_mbv3.feature_info.channels())
    
    # 2. 实例化 MoveNet_FPN
    model = MoveNet_FPN(backbone_mbv3, fpn_out_channels=128, num_joints=17, upsample_to_quarter=False)
    model.eval()

    # 测试前向传播
    input_tensor = torch.randn(1, 3, 192, 192)  # MoveNet 常用小尺寸
    y = model(input_tensor)

    # 打印预测结果
    print({k: v.shape for k, v in y.items()})


# 模型功能测试
# model_runnable_test()

# %% [markdown]
# ## 关于输出结果
# 
# ### 先解释那条预训练权重的提示
# 
# ```
# Unexpected keys (classifier.bias, classifier.weight, conv_head.bias, conv_head.weight) ...
# ```
# 
# * 这是 **正常** 的。因为你用 `features_only=True` 拿的是**中间特征**，不会构建原模型的分类头(`conv_head`/`classifier`)；timm 在加载预训练权重时发现这些头部权重没地方对上，就给出提示。对特征提取没有负面影响。
# 
# ### Backbone 的通道
# 
# ```
# Backbone for FPN created. Output channels: [40, 112, 960]
# ```
# 
# * 表示 `C3, C4, C5` 的通道数分别是 **40 / 112 / 960**。这三路会被 FPN 的 1×1 侧连卷积统一到 `fpn_out_channels=128`，然后自顶向下融合出 **P3、P4、P5**。我们只取 **P3**。
# 
# ### 为什么输出是 24×24？
# 
# 你的输入是 **192×192**，P3 的步幅（stride）是 **8**，所以空间尺寸为：
# 
# * $H_{P3} = W_{P3} = 192 / 8 = 24$
# 
# 如果你把 `upsample_to_quarter=True`，我们会把 P3 上采样到 1/4 分辨率（stride=4），这时就会是 **48×48**。
# 
# ### 四个输出张量分别是什么（语义 & 维度）
# 
# 你打印的结果：
# 
# ```python
# {
#  'heatmaps': torch.Size([1, 17, 24, 24]),
#  'centers':  torch.Size([1, 1,  24, 24]),
#  'regs':     torch.Size([1, 2,  24, 24]),
#  'offsets':  torch.Size([1, 34, 24, 24]),
# }
# ```
# 
# 逐个解释：
# 
# 1. `heatmaps`: **\[B, K, H, W] = \[1, 17, 24, 24]**
# 
#    * **含义**：每个关键点一个通道的**热力图**（logits）。这里 K=17（COCO 关键点数）。
#    * **使用**：训练/推理时一般对其做 `sigmoid()` 得到 \[0,1] 概率图；在每个通道内找峰值（或top-k）作为关键点候选位置。
#    * **坐标映射**：网格坐标 (u,v) 对应到原图坐标约为 $(u+0.5)\times 8,\ (v+0.5)\times 8$（取 stride=8 的像素中心近似；是否 +0.5 取决于你实现习惯）。
# 
# 2. `centers`: **\[B, 1, H, W] = \[1, 1, 24, 24]**
# 
#    * **含义**：**人体中心/实例中心热图**（logits）。多人体时可用它来产生实例种子点（center-based grouping）；单人任务可选用或忽略。
#    * **使用**：同样一般 `sigmoid()`，选峰值位置作为每个人体实例的中心。
# 
# 3. `regs`: **\[B, 2, H, W] = \[1, 2, 24, 24]**
# 
#    * **含义**：**2 通道的回归量**（实数，未激活），语义可按你的训练目标定义：
# 
#      * 常见做法 A：从 `centers` 的峰值处回归到人体框的宽高（或到框的两角）；
#      * 做法 B：从中心指向某个“root keypoint”（如躯干中心）的位移向量；
#    * **使用**：配合 `centers` 把每个中心扩展成一个实例范围或 anchor-free 的框，引导关键点归属。
# 
# 4. `offsets`: **\[B, 2K, H, W] = \[1, 34, 24, 24]**
# 
#    * **含义**：**每个关键点的亚像素偏移**（x,y 各一通道），所以是 2×K=34 个通道，实值（未激活）。
#    * **使用**：在 `heatmaps` 上找到整数网格峰值后，用对应位置的 `(dx, dy)` 做细化：
# 
#      $$
#      x = (u + \sigma(\text{dx})) \times 8,\quad
#      y = (v + \sigma(\text{dy})) \times 8
#      $$
# 
#      是否对偏移过 `tanh/sigmoid` 或直接用原值取决于你的训练规范（常见是直接回归到 \[-0.5, 0.5] 或 \[-1,1] 的范围，并在损失里约束）。
# 
# > 注意：`heatmaps/centers` 是 **logits**（未过激活），训练时用 `BCEWithLogitsLoss`/focal-variant；推理时用 `sigmoid()`。
# > `regs/offsets` 是**连续回归量**（未过激活），常用 `L1`/`SmoothL1`。
# 
# ### 快速自检清单（你现在都满足）
# 
# * ✅ P3 的尺寸是 192/8=24；
# * ✅ 通道数符合设定：`heatmaps = K=17`，`centers = 1`，`regs = 2`，`offsets = 2K=34`；
# * ✅ 预训练提示是预期的；
# * ✅ FPN 输入通道 `[40,112,960]` 平滑到 `fpn_out_channels=128`，与头部 `in_ch=128` 对齐。
# 
# ### 小建议（按需打开）
# 
# * 追求精度：把 `upsample_to_quarter=True`，把 P3 上采到 1/4 分辨率（48×48）；损失里相应用 1/4 的下采样标签。
# * 追求速度：把 `fpn_out_channels` 降到 96/64，或把 `head_midc` 从 32 降到 24/16。
# * 多人解码：用 `centers` 取实例中心，`regs` 估实例范围/根关键点，再将各关键点热图的峰值按几何距离/骨架先验分配到最近实例，并用 `offsets` 做亚像素修正。

# %% [markdown]
# ## 解码
# 
# 相对于 `SSDLite` 来说，MoveNet模型的输出不能直观的生成我们所需要的 `bbox` + `confidence` + `keypoints` 的结构，但是我们可以通过输出把模型的输出结果拼接成我们想要的内容。

# %%
import torch
import torch.nn.functional as F

@torch.no_grad()
def decode_movenet_outputs(
    outputs: dict,
    img_size: tuple,         # (H_img, W_img)
    stride: int = 8,
    topk_centers: int = 10,
    center_thresh: float = 0.2,
    keypoint_thresh: float = 0.05,
    use_mean_kp_in_presence: bool = True,
    regs_in_pixels: bool = True,   # 如果你的 regs 是特征图单位，设为 False
):
    """
    outputs: 来自模型的字典：
      - heatmaps: [1, K, H, W]
      - centers:  [1, 1, H, W]
      - regs:     [1, 2, H, W]   -> (w, h)
      - offsets:  [1, 2K, H, W]  -> (dx, dy) per keypoint
    返回：list[dict]，每个实例：
      {
        "bbox": [x1,y1,x2,y2],
        "person_score": float,
        "keypoints": [(x,y,conf), ...]  # len=K
      }
    """
    H_img, W_img = img_size
    hm = outputs["heatmaps"][0]           # [K,H,W]
    ct = outputs["centers"][0, 0]         # [H,W]
    rg = outputs["regs"][0]               # [2,H,W]
    off = outputs["offsets"][0]           # [2K,H,W]

    K, H, W = hm.shape
    device = hm.device

    # 1) centers: topk + 阈值
    ct_prob = torch.sigmoid(ct)           # [H,W]
    ct_flat = ct_prob.flatten()           # [H*W]
    scores, inds = torch.topk(ct_flat, k=min(topk_centers, ct_flat.numel()))
    keep = scores > center_thresh
    scores = scores[keep]
    inds = inds[keep]
    v_c = (inds // W).long()
    u_c = (inds %  W).long()

    detections = []
    if scores.numel() == 0:
        return detections

    # 2) 对每个中心生成 bbox + 关键点
    for sc, uc, vc in zip(scores.tolist(), u_c.tolist(), v_c.tolist()):
        # 2.1 bbox 宽高 (w,h)
        w = rg[0, vc, uc].item()
        h = rg[1, vc, uc].item()
        # 保证正值
        w = max(1.0, float(w))
        h = max(1.0, float(h))
        if not regs_in_pixels:
            w *= stride
            h *= stride

        # 中心坐标（像素）
        cx = (uc + 0.5) * stride
        cy = (vc + 0.5) * stride

        x1 = max(0.0, cx - w / 2)
        y1 = max(0.0, cy - h / 2)
        x2 = min(float(W_img - 1), cx + w / 2)
        y2 = min(float(H_img - 1), cy + h / 2)

        # 2.2 在 bbox 范围的特征图窗口里找关键点峰值
        # 把 bbox 映射到特征图坐标系
        fx1 = max(0, int(x1 // stride))
        fy1 = max(0, int(y1 // stride))
        fx2 = min(W - 1, int(x2 // stride))
        fy2 = min(H - 1, int(y2 // stride))

        kp_list = []
        kp_scores = []
        hm_prob = torch.sigmoid(hm)  # [K,H,W] 只算一次

        for k in range(K):
            # 裁窗口
            window = hm_prob[k, fy1:fy2+1, fx1:fx2+1]
            if window.numel() == 0:
                kp_list.append((float('nan'), float('nan'), 0.0))
                kp_scores.append(0.0)
                continue

            # 找峰值（窗口内）
            flat_idx = torch.argmax(window)
            wy = int(flat_idx // (fx2 - fx1 + 1))
            wx = int(flat_idx %  (fx2 - fx1 + 1))
            u_k = fx1 + wx
            v_k = fy1 + wy

            s_k = hm_prob[k, v_k, u_k].item()
            if s_k < keypoint_thresh:
                kp_list.append((float('nan'), float('nan'), 0.0))
                kp_scores.append(0.0)
                continue

            # 读 offsets
            dx = off[2*k + 0, v_k, u_k].item()
            dy = off[2*k + 1, v_k, u_k].item()
            # 你若训练在 [-0.5,0.5]，也可 clamp 一下
            # dx = max(-0.5, min(0.5, dx))
            # dy = max(-0.5, min(0.5, dy))

            x_k = (u_k + dx) * stride
            y_k = (v_k + dy) * stride

            # 限制到图像内
            x_k = float(max(0.0, min(W_img - 1, x_k)))
            y_k = float(max(0.0, min(H_img - 1, y_k)))

            kp_list.append((x_k, y_k, float(s_k)))
            kp_scores.append(float(s_k))

        # 2.3 人物置信度
        if use_mean_kp_in_presence and len(kp_scores) > 0:
            person_score = float(sc) * (sum(kp_scores)/max(1, len(kp_scores)))
        else:
            person_score = float(sc)

        det = {
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "person_score": person_score,
            "keypoints": kp_list  # [(x,y,conf)*K]
        }
        detections.append(det)

    # 3) 可选：按 person_score 做一次 bbox NMS（多中心接近时有用）
    detections = nms_by_iou(detections, iou_thr=0.5)
    return detections


def nms_by_iou(dets, iou_thr=0.5):
    if len(dets) <= 1:
        return dets
    # 排序
    order = sorted(range(len(dets)), key=lambda i: dets[i]["person_score"], reverse=True)
    keep, taken = [], [False]*len(dets)
    def iou(a,b):
        ax1,ay1,ax2,ay2 = a["bbox"]; bx1,by1,bx2,by2 = b["bbox"]
        iw = max(0.0, min(ax2,bx2) - max(ax1,bx1))
        ih = max(0.0, min(ay2,by2) - max(ay1,by1))
        inter = iw*ih
        ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
        return inter/ua if ua>0 else 0.0
    for i in order:
        if taken[i]: 
            continue
        keep.append(dets[i])
        for j in order:
            if not taken[j] and j!=i and iou(dets[i], dets[j])>iou_thr:
                taken[j]=True
    return keep

# %% [markdown]
# ## 完整测试

# %%
if __name__ == "__main__":

    # 1. 实例化 Backbone，注意修改 out_indices 来获取 C3, C4, C5
    backbone_mbv3 = timm.create_model(
        'mobilenetv3_large_100',
        pretrained=True,
        features_only=True,
        out_indices=(2, 3, 4), # <-- 核心改动：获取 stride=8, 16, 32 的特征
    )

    print("Backbone for FPN created. Output channels:", backbone_mbv3.feature_info.channels())
    
    # 2. 实例化 MoveNet_FPN
    model = MoveNet_FPN(backbone_mbv3, fpn_out_channels=128, num_joints=17, upsample_to_quarter=False)
    model.eval()

    # 测试前向传播
    input_tensor = torch.randn(1, 3, 192, 192)  # MoveNet 常用小尺寸
    preds = model(input_tensor)

    # 打印预测结果
    print({k: v.shape for k, v in y.items()})

    # 对模型的输出进行解码
    H_img, W_img = 192, 192
    dets = decode_movenet_outputs(preds, img_size=(H_img, W_img), stride=8)

    print(f"Results length: {len(dets)}, the keys of results: {dets[0].keys()}")


