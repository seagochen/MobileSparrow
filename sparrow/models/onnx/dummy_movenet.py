# sparrow/models/onnx/dummy_movenet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DummyMoveNet(nn.Module):
    """
    Wrap MoveNet(dict outputs) -> [B, 51]  (17 joints × (x, y, score))
    只使用 ONNX 友好的张量算子（无 numpy / python float 参与计算图）。
    """
    def __init__(self, movenet: nn.Module,
                 num_joints: int = 17,
                 img_size: int = 192,
                 target_stride: int = 4):
        super().__init__()
        self.m = movenet
        self.num_joints = int(num_joints)
        self.Ht = img_size // target_stride
        self.Wt = img_size // target_stride

    @staticmethod
    def _flatten_argmax_2d(x: torch.Tensor):
        """x: [B,1,H,W] -> (cx, cy)，每个都是 [B,1]（long）。"""
        B, _, H, W = x.shape
        flat = x.view(B, -1)
        ids = torch.argmax(flat, dim=1)          # [B]
        cy = (ids // W).view(B, 1).long()
        cx = (ids %  W).view(B, 1).long()
        return cx, cy

    @staticmethod
    def _gather_hw(x: torch.Tensor,
                   b_idx: torch.Tensor,
                   c_slice,
                   yy: torch.Tensor,
                   xx: torch.Tensor,
                   B: int):
        """
        从 x[b, c, y, x] 按批次/通道/坐标取值，稳定返回 [B,1]。
        - x: [B, C, H, W]
        - b_idx: [B,1]，形如 [[0],[1],...]
        - c_slice: 形如 j0:j0+1 的切片或 [start:end] 的 range/tensor
        - yy/xx: [B,1]（long）
        """
        v = x[b_idx, c_slice, yy, xx]  # 结果可能是 [B,1,1,1] / [B,1] 等
        return v.reshape(B, 1)

    def _resize_heads(self, hm, ct, rg, of):
        # 双线性上采样到 (Ht, Wt)，与当前实现一致
        size = (self.Ht, self.Wt)
        return (F.interpolate(hm, size=size, mode='bilinear', align_corners=False),
                F.interpolate(ct, size=size, mode='bilinear', align_corners=False),
                F.interpolate(rg, size=size, mode='bilinear', align_corners=False),
                F.interpolate(of, size=size, mode='bilinear', align_corners=False))

    def forward(self, x):
        # 1) 主模型前向：期望返回 4 个头（heatmaps/centers/regs/offsets）
        y = self.m(x)
        hm, ct, rg, of = y["heatmaps"], y["centers"], y["regs"], y["offsets"]  # 参见 head 定义的形状约定
        hm, ct, rg, of = self._resize_heads(hm, ct, rg, of)

        B, J, H, W = hm.shape
        device = hm.device
        dtype = hm.dtype

        # 2) center 峰值（单通道）
        cx, cy = self._flatten_argmax_2d(ct)     # [B,1]
        cx_cl = torch.clamp(cx, 0, W - 1)
        cy_cl = torch.clamp(cy, 0, H - 1)

        # 3) 预生成网格（用于距离衰减）
        xs = torch.arange(W, device=device, dtype=dtype).view(1, 1, 1, W)   # [1,1,1,W]
        ys = torch.arange(H, device=device, dtype=dtype).view(1, 1, H, 1)   # [1,1,H,1]

        # 4) 常量用张量表达，避免 TracerWarning
        w_t = hm.new_tensor(W, dtype=dtype)
        h_t = hm.new_tensor(H, dtype=dtype)

        out_list = []
        b_idx = torch.arange(B, device=device).long().view(B, 1)

        # 5) 逐关节解码
        for n in range(J):
            j0 = 2 * n
            j1 = j0 + 1

            # 5.1 在 (cy,cx) 处取 regs 位移，叠加到 center 得到粗定位
            reg_x_o = self._gather_hw(rg, b_idx, slice(j0, j0 + 1), cy_cl, cx_cl, B) + 0.5  # [B,1]
            reg_y_o = self._gather_hw(rg, b_idx, slice(j1, j1 + 1), cy_cl, cx_cl, B) + 0.5  # [B,1]
            reg_x = torch.clamp(cx_cl.to(dtype) + reg_x_o, 0, W - 1)  # [B,1]
            reg_y = torch.clamp(cy_cl.to(dtype) + reg_y_o, 0, H - 1)  # [B,1]

            # 5.2 基于距离衰减的局部峰值搜索
            reg_x_hw = reg_x.view(B, 1, 1, 1)                     # [B,1,1,1]
            reg_y_hw = reg_y.view(B, 1, 1, 1)                     # [B,1,1,1]
            d2 = (xs - reg_x_hw) ** 2 + (ys - reg_y_hw) ** 2      # [B,1,H,W]
            tmp = hm[:, n:n+1, :, :] / torch.sqrt(d2 + 1e-9) / 1.8

            jx, jy = self._flatten_argmax_2d(tmp)                 # [B,1]
            jx = torch.clamp(jx, 0, W - 1)
            jy = torch.clamp(jy, 0, H - 1)

            # 5.3 取 offsets 精细化 + 原 heatmap 上的 score
            off_x = self._gather_hw(of, b_idx, slice(j0, j0 + 1), jy, jx, B)  # [B,1]
            off_y = self._gather_hw(of, b_idx, slice(j1, j1 + 1), jy, jx, B)  # [B,1]
            score = self._gather_hw(hm, b_idx, slice(n, n + 1), jy, jx, B)    # [B,1]

            x_norm = (jx.to(dtype) + off_x) / w_t    # [B,1], 0~1
            y_norm = (jy.to(dtype) + off_y) / h_t    # [B,1], 0~1

            out_list.extend([x_norm, y_norm, score])

        # 6) 拼成 [B, 3*J]
        out = torch.cat(out_list, dim=1)
        return out
