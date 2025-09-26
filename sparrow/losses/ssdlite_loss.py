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
import torch
import math
from typing import List, Sequence, Union

class AnchorGenerator:
    """
    为金字塔特征(FPN)的多个特征图生成锚框(anchor)的工具类。

    设计目标
    ----------
    - 在训练开始前“预计算”每个层级(尺度)的一组**基础单元锚框**(cell anchors)，
      后续在网格上平移复制以得到整张图像上的锚框集合。
    - 支持多尺度(sizes)和多长宽比(aspect_ratios)，可与常见的两阶段/单阶段检测器对齐。

    坐标/形状约定
    ----------
    - 框坐标格式：
        - `cx, cy, w, h`：中心点与宽高，单位=像素，`cx, cy`位于左上角为(0,0)的图像坐标系。
        - `x1, y1, x2, y2`：左上与右下角，单位=像素，满足 `x2 > x1` 且 `y2 > y1`。
    - `cell_anchors[i]` 的形状为 `[A, 4]`，A = `len(aspect_ratios)`。
      这是一组以(0,0)为中心的“原点锚框”，后续通过平移放置到特征图各网格中心。
    - `generate_anchors_on_grid(...)` 返回形状 `[N_total, 4]` 的张量（`xyxy` 格式），
      其中 `N_total = sum_i (H_i * W_i * A)`，i是金字塔层级，H_i/W_i为该层特征图高宽。

    关键假设
    ----------
    - 输入图像尺寸在当前实现中固定为 320x320（见 `generate_anchors_on_grid` 内部常量）。
      若你的训练/推理输入尺寸可变，请将其改为动态传入，或者根据实际 pipeline 替换这里的常量。
    - 每个层级 i 的基础尺寸 `sizes[i]` 与该层的 stride 对应关系需由调用方保证“合理”，
      否则锚框的感受野与特征层不匹配会影响性能（常见做法是 `sizes ~ base * 2**i`）。

    复杂度
    ----------
    - 预计算 `cell_anchors`：O(#levels * #ratios) —— 一次性。
    - 全图锚框生成：O( sum_i (H_i * W_i * #ratios) ) —— 与特征图网格数成正比。

    用法示例
    ----------
    >>> ag = AnchorGenerator(
    ...     sizes=(32, 64, 128, 256, 512),
    ...     aspect_ratios=(0.5, 1.0, 2.0, 1/3.0, 3.0)
    ... )
    >>> # 假设来自 FPN 的三个层级特征图 (N, C, H, W) —— 这里只用它们的空间尺寸
    >>> fms = [torch.empty(1, 256, 80, 80), torch.empty(1, 256, 40, 40), torch.empty(1, 256, 20, 20)]
    >>> anchors = ag.generate_anchors_on_grid([fm for fm in fms], device="cpu")
    >>> anchors.shape  # [ (80*80+40*40+20*20) * len(aspect_ratios), 4 ]
    torch.Size([ ( ... ), 4 ])

    参数
    ----------
    sizes : Sequence[int|float]
        每个层级(与FPN输出一一对应)的“基础尺寸”，理解为该层锚框的参考边长(像素)。
        这不是 w 或 h，而是在生成不同长宽比时派生 w/h 的标尺。
    aspect_ratios : Sequence[float]
        每个位置生成多少种长宽比的锚框，值为 w/h（例如 0.5=竖长，1.0=正方，2.0=横长）。

    属性
    ----------
    num_anchors_per_location : int
        每个位置(网格点)生成的锚框数量，等于 len(aspect_ratios)。
    cell_anchors : List[Tensor]
        预生成的“原点”基础锚框列表；长度等于 len(sizes)，每个元素形如 [A, 4] (xyxy)。
    """

    def __init__(self,
                 sizes: Sequence[Union[int, float]] = (32, 64, 128, 256, 512),
                 aspect_ratios: Sequence[float] = (0.5, 1.0, 2.0, 1/3.0, 3.0)):
        super().__init__()
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.num_anchors_per_location = len(self.aspect_ratios)
        # 预计算：每个层级的一组“以原点为中心”的基础锚框（xyxy，单位像素）
        self.cell_anchors = self._generate_cell_anchors()

    def _generate_cell_anchors(self) -> List[torch.Tensor]:
        """
        为每个尺度生成以(0,0)为中心的一组基础锚框（不含平移），并存为 xyxy 格式。

        生成规则
        ----------
        对于给定尺度 s 和每个长宽比 ar：
            w = s * sqrt(ar)
            h = s / sqrt(ar)
        先按 `cx,cy,w,h = (0,0,w,h)` 构造，再转换为 `xyxy`。

        返回
        ----------
        List[Tensor]：
            长度 = len(sizes)，第 i 项形状为 [A, 4]，A=len(aspect_ratios)，坐标为 xyxy。
        """
        cell_anchors: List[torch.Tensor] = []
        for s in self.sizes:
            # ar = w/h，因此 w = s*sqrt(ar), h = s/sqrt(ar)
            w = torch.tensor([s * math.sqrt(ar) for ar in self.aspect_ratios], dtype=torch.float32)
            h = torch.tensor([s / math.sqrt(ar) for ar in self.aspect_ratios], dtype=torch.float32)

            # 以 (cx, cy) = (0, 0) 的“原点锚框”，后续仅做平移即可铺到网格
            base_cx = torch.zeros_like(w)
            base_cy = torch.zeros_like(h)
            base_cxcywh = torch.stack([base_cx, base_cy, w, h], dim=1)  # [A, 4] (cx,cy,w,h)
            base_anchors = self.cxcywh_to_xyxy(base_cxcywh)             # [A, 4] (xyxy)
            cell_anchors.append(base_anchors)

        return cell_anchors

    def cxcywh_to_xyxy(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        将 (cx, cy, w, h) 转为 (x1, y1, x2, y2)。

        参数
        ----------
        boxes : Tensor
            形状 [N, 4]，单位像素。

        返回
        ----------
        Tensor
            形状 [N, 4]，(x1, y1, x2, y2)。
        """
        return torch.cat([boxes[:, :2] - boxes[:, 2:] / 2,
                          boxes[:, :2] + boxes[:, 2:] / 2], dim=1)

    def xyxy_to_cxcywh(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        将 (x1, y1, x2, y2) 转为 (cx, cy, w, h)。

        参数
        ----------
        boxes : Tensor
            形状 [N, 4]，单位像素。

        返回
        ----------
        Tensor
            形状 [N, 4]，(cx, cy, w, h)。
        """
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        cx = boxes[:, 0] + w / 2
        cy = boxes[:, 1] + h / 2
        return torch.stack([cx, cy, w, h], dim=1)

    def generate_anchors_on_grid(self, feature_maps: List[torch.Tensor], device: Union[str, torch.device]) -> torch.Tensor:
        """
        将每层的基础锚框平移到对应特征图的每个网格位置，得到整张图像的锚框集合（xyxy）。

        参数
        ----------
        feature_maps : List[Tensor]
            FPN 各层的特征图张量列表。仅使用其空间尺寸 `H, W`；
            形状一般为 [N, C, H, W]（N 和 C 不影响锚框生成）。
        device : str | torch.device
            生成锚框所在设备（例如 "cpu" 或 "cuda"），将与 `cell_anchors` 做一致化。

        返回
        ----------
        Tensor
            拼接后的所有层锚框，形状 `[sum_i(H_i*W_i)*A, 4]`，坐标为 xyxy。

        细节说明
        ----------
        - 当前实现假设输入图像大小为 **320x320**，并据此从特征图尺寸反推出 stride：
              stride_w = 320 / W_i, stride_h = 320 / H_i
          如果你的图像尺寸并非固定 320x320，请将下方的 `input_size_h, input_size_w` 改为动态值。
        - `torch.meshgrid(..., indexing='ij')` 保证 `shift_y` 对应第 0 维 (行/高度)，`shift_x` 对应第 1 维 (列/宽度)。
        - `shifts` 的每一行是 `(x, y, x, y)`，用于把“以原点为中心”的 `cell_anchors` 整体平移到网格点处。
        """
        all_anchors = []
        # ⚠️ 假设输入 320x320；若可变尺寸，请改为动态参数或从调用方传入
        input_size_h, input_size_w = 320, 320

        for i, fm in enumerate(feature_maps):
            fm_h, fm_w = fm.shape[-2], fm.shape[-1]
            stride_h = input_size_h / fm_h
            stride_w = input_size_w / fm_w

            # 网格位移：每个网格中心的 (x, y)（这里用左上对齐的整数步长位置；是否加0.5由检测头对齐策略决定）
            shifts_x = torch.arange(0, fm_w, device=device, dtype=torch.float32) * stride_w
            shifts_y = torch.arange(0, fm_h, device=device, dtype=torch.float32) * stride_h
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')  # [H, W]

            # 每个网格位置的平移量 (x, y, x, y)；与 cell_anchors 相加完成平移
            shifts = torch.stack(
                (shift_x.flatten(), shift_y.flatten(), shift_x.flatten(), shift_y.flatten()),
                dim=1
            )  # [H*W, 4]

            # 将“原点锚框”平移到每个网格位置；再展平
            anchors = (self.cell_anchors[i].to(device).view(1, -1, 4) + shifts.view(-1, 1, 4)).reshape(-1, 4)
            all_anchors.append(anchors)

        return torch.cat(all_anchors, dim=0)


# =========================
# 额外建议（可选，非必须改动）
# =========================
# 1) 若输入尺寸可变，可把 generate_anchors_on_grid 的 input_size 作为参数传入：
#    def generate_anchors_on_grid(self, feature_maps, device, input_size: Tuple[int, int]):
#        input_size_h, input_size_w = input_size
# 2) 若你的检测头假设网格中心在 (x+0.5, y+0.5)，可以在 shifts_x / shifts_y 上加 0.5：
#        shifts_x = (torch.arange(...)+0.5) * stride_w
#        shifts_y = (torch.arange(...)+0.5) * stride_h
# 3) 当希望导出到 ONNX/TS 时，注意避免使用 Python float 混入（确保 dtype 一致）。


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
class SSDLoss(nn.Module):
    """
    SSD/RetinaNet 风格的检测损失模块（分类 + 边框回归）。

    本模块以**预先给定的锚框 anchors**为基准，将每个 batch 的 GT 标注分配到锚框上，
    然后分别计算：
      1) 分类损失：使用 `sigmoid_focal_loss`（多标签式，按类别独立二分类）
      2) 回归损失：对正样本使用 Smooth L1 (Huber) 损失，回归 (tx, ty, tw, th) 变换量

    主要流程
    ----------
    1. `assign_targets_to_anchors`：
       - 基于 IoU 将 GT 与 anchors 匹配，得到每个 anchor 的类别标签与匹配的 GT 框。
       - 采用“为每个 GT 至少分配一个 anchor”的策略：对每个 GT，选取 IoU 最高的 anchor 强制正样本。
       - 其他 anchors 中，IoU >= `iou_threshold_pos` 视为正；其余保持为“背景/忽略”。
         （此实现未显式处理 [neg, pos) 的“忽略区间”，详见下方注意事项）

    2. `encode_bbox`：
       - 将匹配的 GT 框与对应 anchor 框都从 xyxy 转为 cxcywh，然后编码为回归目标：
            tx = (cx_gt - cx_a) / w_a
            ty = (cy_gt - cy_a) / h_a
            tw = log(w_gt / w_a)
            th = log(h_gt / h_a)

    3. `forward`：
       - 将 `assigned_labels` one-hot 到 C 维（丢弃背景/忽略的第 C+1 类），
         用 `sigmoid_focal_loss` 做分类损失（在 B×A×C 上求和再平均）。
       - 对正样本位置计算 Smooth L1 回归损失，并用正样本数进行归一化。

    坐标与形状约定
    ----------
    - 框坐标采用像素尺度：
        - 输入/输出 anchors 与 GT：`[x1, y1, x2, y2]` (xyxy)
        - 回归编码：`[tx, ty, tw, th]`
    - 输入张量形状：
        - `anchors`:   [A, 4]，A=总锚框数（所有层全部拼接）
        - `cls_preds`: [B, A, C]，C=类别数（不包含背景）
        - `reg_preds`: [B, A, 4]
        - `targets`:   长度为 B 的 List；每个元素形如 [N_i, 5]，列为 (cls, x1, y1, x2, y2)
    - 输出：
        - `loss_cls`: 标量分类损失（平均）
        - `loss_reg`: 标量回归损失（按正样本数归一化）

    参数
    ----------
    num_classes : int
        前景类别数（不含背景）。
    iou_threshold_pos : float
        正样本阈值；当某 anchor 与某 GT 的 IoU >= 该阈值时，可被标为正样本。
    iou_threshold_neg : float
        负样本阈值（当前实现没有显式使用“忽略区间”的逻辑，仅保留作参数占位）。

    注意事项
    ----------
    - 本实现的分类损失对所有 anchors 都参与（包含未匹配到 GT 的为“背景”），
      具体表现为：`one_hot` 后仅 C 个前景通道参与 focal loss，未匹配处相当于全零目标。
    - 经典 SSD 会使用“困难样本挖掘/负样本采样 (OHEM 或 fixed ratio)”，
      当前实现未加入采样或 `iou_threshold_neg` 的忽略区间逻辑；
      如需严格复现 SSD，可在 assign 或 loss 计算处增加采样/忽略机制。
    - `AnchorGenerator` 仅用于坐标变换（编码/解码）等辅助；anchors 由外部预计算并传入。
    """

    def __init__(self,
                 num_classes: int,
                 iou_threshold_pos: float = 0.5,
                 iou_threshold_neg: float = 0.4):
        super().__init__()
        self.num_classes = num_classes
        self.iou_threshold_pos = iou_threshold_pos
        self.iou_threshold_neg = iou_threshold_neg

        # AnchorGenerator 实例仍然需要，用于坐标变换等辅助功能（anchors 本身由外部传入）
        self.anchor_generator = AnchorGenerator()

        # 回归目标的标准差，用于归一化
        # 注册为 buffer：随 .to(device) 一起移动，随 state_dict 一起保存
        self.register_buffer(
            "bbox_std",
            torch.tensor([0.1, 0.1, 0.2, 0.2], dtype=torch.float32)   # 固定成 float32
        )

    def assign_targets_to_anchors(self, anchors: torch.Tensor, targets: List[torch.Tensor]):
        """
        将 GT 分配到 anchors，得到每个 anchor 的类别标签与对应的 GT 框（xyxy）。

        分配策略
        ----------
        - 为确保每个 GT 至少有一个正样本：对每个 GT，找到与其 IoU 最大的 anchor，强制标为该 GT 的类别。
        - 同时，对所有 anchors，若其与任一 GT 的最大 IoU >= `iou_threshold_pos`，标记为该 GT 的类别。
        - 其他 anchors 的标签保持为 `num_classes`（可视为“背景/忽略”）。
        - 该函数**未**使用 `iou_threshold_neg` 创建忽略区间；如需忽略，可在此基础上扩展。

        参数
        ----------
        anchors : Tensor
            形状 [A, 4] 的锚框（xyxy，已在正确设备上）。
        targets : List[Tensor]
            长度为 B 的列表；第 i 个元素形状 [N_i, 5]，列为 (cls, x1, y1, x2, y2)。

        返回
        ----------
        labels : Tensor
            形状 [B, A]，每个 anchor 的类别 id（0..C-1 为前景；C 表示背景/未分配）。
        matched_gt_boxes : Tensor
            形状 [B, A, 4]，与每个 anchor 匹配的 GT 框（若未匹配则为 0）。
        """
        batch_size = len(targets)
        num_anchors = anchors.shape[0]
        device = anchors.device

        # 初始化：标签为 C（表示背景/未分配），GT 框为 0
        labels = torch.full((batch_size, num_anchors), self.num_classes, dtype=torch.int64, device=device)
        matched_gt_boxes = torch.zeros((batch_size, num_anchors, 4), dtype=torch.float32, device=device)

        for i in range(batch_size):
            gt_boxes = targets[i][:, 1:]  # [N_i, 4]
            gt_labels = targets[i][:, 0]  # [N_i]
            if gt_boxes.shape[0] == 0:
                # 无 GT：保持背景
                continue

            # IoU 计算：行=GT，列=Anchor => [N_i, A]
            iou = box_iou(gt_boxes, anchors)

            # --- 核心修改：引入 Ignore Zone ---
            max_iou_per_anchor, max_iou_idx_per_anchor = iou.max(dim=0)

            # 1. 负样本：IoU < 0.4 的 anchor 保持为背景 (已经是默认值 self.num_classes)
            # neg_mask = max_iou_per_anchor < self.iou_threshold_neg (无需操作)

            # 2. 灰区/忽略样本：0.4 <= IoU < 0.5 的 anchor 标记为 -1
            ignore_mask = (max_iou_per_anchor >= self.iou_threshold_neg) & (max_iou_per_anchor < self.iou_threshold_pos)
            labels[i, ignore_mask] = -1

            # 3. 正样本：IoU >= 0.5 的 anchor
            # 2) 对所有 anchor 找其最匹配的 GT，并按阈值标为正样本
            pos_mask = max_iou_per_anchor >= self.iou_threshold_pos
            if pos_mask.any():
                labels[i, pos_mask] = gt_labels[max_iou_idx_per_anchor[pos_mask]]
                matched_gt_boxes[i, pos_mask] = gt_boxes[max_iou_idx_per_anchor[pos_mask]]

            # 4. 确保每个 GT 至少有一个 anchor 匹配 (最高 IoU 匹配)
            max_iou_per_gt, max_iou_idx_per_gt = iou.max(dim=1)
            labels[i, max_iou_idx_per_gt] = gt_labels
            matched_gt_boxes[i, max_iou_idx_per_gt] = gt_boxes

        return labels, matched_gt_boxes

    def encode_bbox(self, anchors: torch.Tensor, gt_boxes: torch.Tensor):
        """
        将 GT 框相对 anchor 框编码为 (tx, ty, tw, th)。

        参数
        ----------
        anchors : Tensor
            形状 [N, 4]，anchor 框（xyxy）。
        gt_boxes : Tensor
            形状 [N, 4]，与 anchors 一一对应的 GT 框（xyxy）。

        返回
        ----------
        Tensor
            形状 [N, 4]，列为 (tx, ty, tw, th)。
        """

        anchors_c = self.anchor_generator.xyxy_to_cxcywh(anchors)
        gt_c      = self.anchor_generator.xyxy_to_cxcywh(gt_boxes)

        # 位置
        tx = (gt_c[:, 0] - anchors_c[:, 0]) / anchors_c[:, 2]
        ty = (gt_c[:, 1] - anchors_c[:, 1]) / anchors_c[:, 3]

        # 尺寸（关键：clamp 防止 log(0) / log(负)）
        eps = 1e-6
        tw = torch.log((gt_c[:, 2] / anchors_c[:, 2]).clamp(min=eps))
        th = torch.log((gt_c[:, 3] / anchors_c[:, 3]).clamp(min=eps))

        # 缩放到更友好的量级（与推理 decode 对应）
        # 直接用 buffer；为防止混合精度/自定义 dtype，匹配到当前张量 dtype
        std = self.bbox_std.to(anchors.dtype) # bbox -> move to device

        # 保险：限制极端残差，避免单批“异常大梯度”
        deltas = torch.stack([tx, ty, tw, th], dim=1) / std
        deltas = deltas.clamp(min=-4.0, max=4.0)
        return deltas

    def forward(self, anchors: torch.Tensor, cls_preds: torch.Tensor, reg_preds: torch.Tensor, targets: List[torch.Tensor]):
        """
        计算总损失（分类 + 回归）。

        参数
        ----------
        anchors : Tensor
            [A, 4]，所有层级拼接后的锚框（xyxy），需与 `cls_preds`/`reg_preds` 的 A 对齐。
        cls_preds : Tensor
            [B, A, C]，分类预测（每类一个独立 sigmoid）。
        reg_preds : Tensor
            [B, A, 4]，回归预测（与 anchors 对齐，回归 (tx, ty, tw, th)）。
        targets : List[Tensor]
            长度为 B 的列表；每个元素 [N_i, 5]，列为 (cls, x1, y1, x2, y2)。

        返回
        ----------
        loss_cls : Tensor
            标量分类损失。
        loss_reg : Tensor
            标量回归损失（对正样本平均；若无正样本则为 0）。
        """
        device = cls_preds.device
        # 2. 目标分配 (使用传入的 anchors)
        assigned_labels, assigned_gt_boxes = self.assign_targets_to_anchors(anchors.to(device), targets)

        # --- 分类损失计算 (处理 ignore) ---
        valid_mask = assigned_labels != -1  # 过滤掉 ignore=-1 的 anchor
        if not valid_mask.any():
             return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

        # 准备 one-hot 目标 (对 valid anchor)
        labels_for_onehot = assigned_labels[valid_mask].clamp(min=0) # 将背景(num_classes)也转为 one-hot
        target_one_hot = F.one_hot(labels_for_onehot, num_classes=self.num_classes + 1)
        target_one_hot = target_one_hot[..., :self.num_classes].float()

        # 计算 focal loss (只在 valid anchor上)
        loss_cls = sigmoid_focal_loss(
            cls_preds[valid_mask], target_one_hot,
            alpha=0.25, gamma=2.0, reduction='mean'
        )

        # --- 回归损失计算 (只处理正样本) ---
        pos_mask = (assigned_labels >= 0) & (assigned_labels < self.num_classes)
        num_pos = pos_mask.sum().item()
        if num_pos > 0:
            # 取出正样本位置的回归预测与匹配 GT
            pos_reg_preds = reg_preds[pos_mask]                 # [N_pos, 4]
            pos_gt_boxes = assigned_gt_boxes[pos_mask]          # [N_pos, 4]
            # 对齐得到与正样本相同索引的 anchors
            pos_anchors = anchors.to(device).unsqueeze(0).expand_as(assigned_gt_boxes)[pos_mask]

            # 计算回归目标并求 Smooth L1
            target_deltas = self.encode_bbox(pos_anchors, pos_gt_boxes)  # [N_pos, 4]

            # 检测里常用 beta=1/9（Faster/Mask R-CNN 同款），能更快进入二次区，进一步抑制大残差的线性爆发：
            # loss_reg = F.smooth_l1_loss(pos_reg_preds, target_deltas, beta=1.0, reduction='sum')  # 原本的方式
            loss_reg = F.smooth_l1_loss(pos_reg_preds, target_deltas, beta=1/9, reduction='sum') / num_pos

            # 按正样本数归一化
            loss_reg = loss_reg / num_pos
        else:
            loss_reg = torch.tensor(0.0, device=device)

        return loss_cls, loss_reg

# %% [markdown]
# ## 如何在你的训练脚本中使用
# 
# 现在，你可以在你的主训练文件中实例化并使用这个 `SSDLoss`。
# 
# ```python
# 
# from ssdlite_fpn import SSDLite_FPN
# from dataloader import create_dets_dataloader
# from loss import SSDLoss # 导入我们刚创建的损失类
# 
# # 参数设置
# NUM_CLASSES = 80 # COCO
# ANCHOR_SIZES = [32, 64, 128, 256, 512]
# ANCHOR_RATIOS = [0.5, 1.0, 2.0, 1/3.0, 3.0]
# 
# # ... (模型、数据加载器、优化器的初始化)
# # model = SSDLite_FPN(...)
# # train_loader = create_dets_dataloader(...)
# # optimizer = torch.optim.AdamW(model_fpn.parameters(), lr=1e-4, weight_decay=1e-3)
# 
# # 实例化损失函数，可以使用默认权重，也可以自定义
# criterion = SSDLoss(num_classes=NUM_CLASSES)
# 
# # 预计算锚框 
# print("Pre-computing anchors for fixed input size...")
# anchor_generator = AnchorGenerator(
#     sizes=ANCHOR_SIZES,
#     aspect_ratios=ANCHOR_RATIOS
# )
# 
# # 这个 precomputed_anchors 将在整个训练过程中被重复使用
# precomputed_anchors = anchor_generator.generate_anchors_on_grid(feature_maps_for_size_calc, device)
# 
# model.train()
# for imgs, labels in train_loader:
#         # targets_on_device = [t.to(device) for t in targets]
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
# ```


