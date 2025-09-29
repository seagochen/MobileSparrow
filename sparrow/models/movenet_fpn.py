# %% [markdown]
# # 构建一个MoveNet姿态检测模型
# 
# FPN + MobileNet 作为骨架，可以实现很多CV任务，本次我们将实现一个高精度的轻量级的姿态检测模型—— `MoveNet`。
# 
# 与 `003_SSDLite` 相似，本次我们只开发一个新的检测头，然后接入已有的架构。

# %% [markdown]
# ## 模块导入与模型定义（与您原来版本一致）

# %%
import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List
import numpy as np


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


class MoveNetHead(nn.Module):
    """
    四个分支：
      - heatmaps: [B, K, H, W]   (关键点K的高斯/逻辑热力图)
      - centers:  [B, 1, H, W]   (人体中心/实例中心热力图)
      - regs:     [B, 2K, H, W]  (从中心到关键点的位移向量)
      - offsets:  [B, 2K, H, W]  (关键点子像素偏移)
    """
    def __init__(self, in_ch: int, num_joints: int = 17, midc: int = 32):
        super().__init__()
        def dw_pw(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, in_c, 3, padding=1, groups=in_c, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_c, out_c, 1, bias=True)
            )
        self.hm_tower = dw_pw(in_ch, midc)
        self.ct_tower = dw_pw(in_ch, midc)
        self.reg_tower = dw_pw(in_ch, midc)
        self.off_tower = dw_pw(in_ch, midc)

        self.hm = nn.Conv2d(midc, num_joints, 1)
        self.ct = nn.Conv2d(midc, 1, 1)
        self.reg = nn.Conv2d(midc, num_joints * 2, 1)
        self.off = nn.Conv2d(midc, num_joints * 2, 1)

        nn.init.constant_(self.hm.bias, -2.0)
        nn.init.constant_(self.ct.bias, -2.0) 

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 注意：社区版的模型在输出层后加了 Sigmoid，而您没有。
        # 您的设计（输出logits）配合 BCEWithLogitsLoss 是更常见且数值稳定的做法，所以这里保持不变。
        return {
            "heatmaps": self.hm(self.hm_tower(x)),
            "centers":  self.ct(self.ct_tower(x)),
            "regs":     self.reg(self.reg_tower(x)),
            "offsets":  self.off(self.off_tower(x)),
        }


class MoveNet_FPN(nn.Module):
    def __init__(self,
                 backbone,
                 num_joints: int = 17,
                 fpn_out_channels: int = 128,
                 head_midc: int = 32,
                 out_stride: int = 8,
                 upsample_to_quarter: bool = False
                 ):
        super().__init__()
        self.backbone = backbone
        backbone_channels = self.backbone.feature_info.channels()
        self.fpn = FPN(in_channels=backbone_channels, out_channels=fpn_out_channels)
        self.head = MoveNetHead(in_ch=fpn_out_channels, num_joints=num_joints, midc=head_midc)
        self.out_stride = out_stride
        self.upsample_to_quarter = upsample_to_quarter

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        p3, _, _ = self.fpn(features)
        if self.upsample_to_quarter:
            p3 = F.interpolate(p3, scale_factor=2.0, mode='bilinear', align_corners=False)
        return self.head(p3)


# %% [markdown]
# ## 解码函数（关键修改）
# 
# 以下是整合了社区版解码逻辑的全新 `decode_movenet_outputs` 函数。

# %%

def decode_movenet_outputs(
    outputs: Dict[str, torch.Tensor],
    img_size: Tuple[int, int],
    stride: int,
    topk_centers: int = 1,          # 单人中心，直接取 Top-1
    center_thresh: float = 0.1,
    keypoint_thresh: float = 0.1,
    hm_th: float = 0.1,
) -> List[Dict[str, any]]:
    """
    无 bbox / 无 NMS 解码器：
    - 仅解出关键点与 person_score
    - 适用于单人且基本居中的场景
    """
    import numpy as np

    H_img, W_img = img_size

    # 1) Tensor -> numpy，取 batch 中第 1 张
    heatmaps = torch.sigmoid(outputs["heatmaps"][0]).cpu().numpy() # [K, Hf, Wf]
    centers  = torch.sigmoid(outputs["centers"][0]).cpu().numpy()  # [1, Hf, Wf]
    regs     = outputs["regs"][0].cpu().numpy()                    # [2K, Hf, Wf]
    offsets  = outputs["offsets"][0].cpu().numpy()                 # [2K, Hf, Wf]
    K, Hf, Wf = heatmaps.shape

    # 2) 取中心候选（默认只取 Top-1）
    center_scores = centers.flatten()
    top_inds = np.argsort(center_scores)[::-1][:max(1, topk_centers)]

    dets = []
    for center_ind in top_inds:
        center_score = center_scores[center_ind]
        if center_score < center_thresh:
            continue

        vc, uc = np.unravel_index(center_ind, (Hf, Wf))
        vc, uc = int(vc), int(uc)

        kp_list, kp_scores = [], []

        # 3) 对每个关键点做「回归引导 + 热图加权」的精定位
        for k in range(K):
            kp_heatmap = heatmaps[k].copy()
            kp_heatmap[kp_heatmap < hm_th] = 0.0

            # 在中心位置读取回归向量
            dx = regs[2 * k,     vc, uc]
            dy = regs[2 * k + 1, vc, uc]

            # 初始估计坐标（特征图尺度）
            kp_x_init = uc + dx
            kp_y_init = vc + dy

            # 构造距离加权，抑制远离回归点的响应（社区版思路）
            grid_x, grid_y = np.meshgrid(np.arange(Wf), np.arange(Hf))
            dist_map = np.sqrt((grid_x - kp_x_init) ** 2 + (grid_y - kp_y_init) ** 2) + 1.8
            weighted = kp_heatmap / dist_map

            # 在加权热图上取峰值（整数网格）
            v0, u0 = np.unravel_index(np.argmax(weighted), (Hf, Wf))

            # 亚像素细化：在 (u0, v0) 读取 offsets
            ox = offsets[2 * k,     v0, u0]
            oy = offsets[2 * k + 1, v0, u0]

            # 关键点置信度（可用加权后峰值，也可用原热图峰值）
            s_k = float(max(0.0, weighted[v0, u0]))

            # 转回原图尺度
            x_k = float(np.clip((u0 + ox) * stride, 0, W_img - 1))
            y_k = float(np.clip((v0 + oy) * stride, 0, H_img - 1))

            # 设置关键点是否可见（> keypoint_thresh）
            conf = s_k if s_k >= keypoint_thresh else 0.0

            kp_list.append((x_k, y_k, conf))
            kp_scores.append(conf)

        # 4) 计算 person_score（不生成 bbox）
        if len(kp_scores) == 0:
            continue
        person_score = float(center_score) * (sum(kp_scores) / max(1, len(kp_scores)))

        dets.append({
            "keypoints": kp_list,          # 长度 17，每个元素 (x, y, conf)
            "person_score": person_score,  # 仅供排序/参考
            # 不再包含 "bbox"
        })

    # 单人场景：只保留分数最高的一个
    if len(dets) > 1:
        dets = sorted(dets, key=lambda d: d["person_score"], reverse=True)[:1]

    return dets


# %% [markdown]
# ## 完整测试

# %%
if __name__ == "__main__":

    # 1. 实例化 Backbone
    backbone_mbv3 = timm.create_model(
        'mobilenetv3_large_100',
        pretrained=True,
        features_only=True,
        out_indices=(2, 3, 4), 
    )
    print("Backbone for FPN created. Output channels:", backbone_mbv3.feature_info.channels())
    
    # 2. 实例化 MoveNet_FPN
    # 社区版通常在 1/4 分辨率上操作，我们也设置 upsample_to_quarter=True
    # stride 也要相应地改为 4
    IMG_SIZE = 192
    STRIDE = 4
    model = MoveNet_FPN(backbone_mbv3, fpn_out_channels=128, num_joints=17, upsample_to_quarter=True)
    model.eval()

    # 3. 测试前向传播
    input_tensor = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    preds = model(input_tensor)

    # 打印预测结果的形状
    # 因为上采样，输出尺寸应该是 192/4 = 48x48
    print("Output shapes:", {k: v.shape for k, v in preds.items()})

    # 4. 对模型的输出进行解码
    # 模拟一次解码过程
    dets = decode_movenet_outputs(preds, img_size=(IMG_SIZE, IMG_SIZE), stride=STRIDE)

    print(f"\nDecoded {len(dets)} person(s).")
    if dets:
        print("Keys of the first detected person:", dets[0].keys())
        # 打印第一个人的第一个关键点
        print("First keypoint of first person:", dets[0]['keypoints'][0])