# core/models/dummy_movenet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DummyMoveNet(nn.Module):
    """
    Wrap MoveNet(dict outputs) -> [B, 51]   (17 joints × (x,y,score))
    All ops are ONNX-friendly (no numpy).
    """
    def __init__(self, movenet: nn.Module,
                num_joints: int = 17,
                img_size: int = 192, 
                target_stride: int = 4):
        super().__init__()
        self.m = movenet
        self.num_joints = num_joints
        self.Ht = img_size // target_stride
        self.Wt = img_size // target_stride

    @staticmethod
    def _flatten_argmax_2d(x: torch.Tensor):
        # x: [B,1,H,W] -> (cx, cy) each [B,1] long
        B, _, H, W = x.shape
        flat = x.view(B, -1)
        ids = torch.argmax(flat, dim=1)  # [B]
        cy = (ids // W).view(B, 1).long()
        cx = (ids %  W).view(B, 1).long()
        return cx, cy

    def _resize_heads(self, hm, ct, rg, of):
        # Bilinear to (Ht,Wt)
        size = (self.Ht, self.Wt)
        return (F.interpolate(hm, size=size, mode='bilinear', align_corners=False),
                F.interpolate(ct, size=size, mode='bilinear', align_corners=False),
                F.interpolate(rg, size=size, mode='bilinear', align_corners=False),
                F.interpolate(of, size=size, mode='bilinear', align_corners=False))

    def forward(self, x):
        # MoveNet.forward -> dict
        y = self.m(x)
        hm, ct, rg, of = y["heatmaps"], y["centers"], y["regs"], y["offsets"]
        hm, ct, rg, of = self._resize_heads(hm, ct, rg, of)

        B, J, H, W = hm.shape
        device = hm.device

        # 1) center peak
        cx, cy = self._flatten_argmax_2d(ct)            # [B,1]
        cx_cl = torch.clamp(cx, 0, W - 1)
        cy_cl = torch.clamp(cy, 0, H - 1)

        # 2) grid（用于距离衰减）
        xs = torch.arange(W, device=device).view(1, 1, 1, W).float()   # [1,1,1,W]
        ys = torch.arange(H, device=device).view(1, 1, H, 1).float()   # [1,1,H,1]

        out_list = []
        # per-joint decode
        for n in range(J):
            # 2.1 regs 在 (cy,cx) 位置的回归位移，并加到 center 得到粗定位
            # gather 索引准备
            b_idx = torch.arange(B, device=device).long().view(B, 1)
            j0 = 2 * n
            j1 = 2 * n + 1

            reg_x_o = rg[b_idx, j0:j0+1, cy_cl, cx_cl].squeeze(-1).squeeze(-1) + 0.5  # [B,1]
            reg_y_o = rg[b_idx, j1:j1+1, cy_cl, cx_cl].squeeze(-1).squeeze(-1) + 0.5  # [B,1]
            reg_x = torch.clamp(cx_cl.float() + reg_x_o, 0, W - 1)    # [B,1]
            reg_y = torch.clamp(cy_cl.float() + reg_y_o, 0, H - 1)    # [B,1]

            # 2.2 用距离衰减加权的 heatmap 寻找该关节的峰值
            reg_x_hw = reg_x.view(B, 1, 1, 1)    # [B,1,1,1]
            reg_y_hw = reg_y.view(B, 1, 1, 1)    # [B,1,1,1]
            d2 = (xs - reg_x_hw) ** 2 + (ys - reg_y_hw) ** 2          # [B,1,H,W]
            # 防止除零：+1.8 与社区版一致
            tmp = hm[:, n:n+1, :, :] / torch.sqrt(d2 + 1e-9) / 1.8

            jx, jy = self._flatten_argmax_2d(tmp)    # [B,1]
            jx = torch.clamp(jx, 0, W - 1)
            jy = torch.clamp(jy, 0, H - 1)

            # 2.3 取该点的 offset 细化坐标；score 取原 heatmap 在该点的值
            off_x = of[b_idx, j0:j0+1, jy, jx].squeeze(-1).squeeze(-1)   # [B,1]
            off_y = of[b_idx, j1:j1+1, jy, jx].squeeze(-1).squeeze(-1)   # [B,1]
            score = hm[b_idx, n:n+1, jy, jx].squeeze(-1).squeeze(-1)     # [B,1]

            x_norm = (jx.float() + off_x) / float(W)                     # [B,1]
            y_norm = (jy.float() + off_y) / float(H)                     # [B,1]

            # 将数据打包成 x, y, score 的格式
            out_list.extend([x_norm, y_norm, score])

        # 拼成 [B, 51]
        out = torch.cat(out_list, dim=1)
        return out
