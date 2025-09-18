# sparrow/models/onnx/dummy_ssdlite.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DummySSDLite(nn.Module):
    """
    Wrap SSDLite -> detections tensor (无需改动 ssdlite.py)
    输出: [B, N, 6]，每行为 (x1, y1, x2, y2, score, cls)
    说明:
      - 通过传参重建 anchors（cx,cy,w,h，归一化到0~1），据此对 bbox_regs 进行解码
      - 不做 NMS；部署端再处理
    """
    def __init__(self,
                 model: nn.Module,
                 num_classes: int,
                 img_size: int = 320,
                 ratios=(1.0, 2.0, 0.5),
                 scales=(1.0, 1.26),
                 strides=(8, 16, 32),
                 return_normalized: bool = False,
                 drop_background: bool = True,
                 variances=(0.1, 0.1, 0.2, 0.2)):
        super().__init__()
        self.m = model
        self.num_classes = int(num_classes)
        self.img_size = int(img_size)
        self.ratios = tuple(float(r) for r in ratios)
        self.scales = tuple(float(s) for s in scales)
        self.strides = tuple(int(s) for s in strides)
        self.return_normalized = bool(return_normalized)
        self.drop_background = bool(drop_background)
        self.vx, self.vy, self.vw, self.vh = [float(v) for v in variances]
        self.A = len(self.ratios) * len(self.scales)  # anchors per cell

    @torch.no_grad()
    def _mk_anchors_one_level(self, Hf, Wf, stride, device, dtype):
        # 网格中心 (cx,cy) 归一化到0~1；anchor宽高(w,h)同样归一化
        xs = (torch.arange(Wf, device=device, dtype=dtype) + 0.5) * stride / self.img_size
        ys = (torch.arange(Hf, device=device, dtype=dtype) + 0.5) * stride / self.img_size
        cy, cx = torch.meshgrid(ys, xs, indexing="ij")  # [Hf,Wf]

        r = torch.tensor(self.ratios, dtype=dtype, device=device).view(1, 1, -1)  # [1,1,R]
        sc = torch.tensor(self.scales, dtype=dtype, device=device).view(1, 1, -1) # [1,1,S]
        r = r.unsqueeze(-1).expand(-1, -1, r.shape[-1], sc.shape[-1])   # [1,1,R,S]
        sc = sc.unsqueeze(-2).expand(-1, -1, r.shape[-2], sc.shape[-1]) # [1,1,R,S]

        aw = (stride * (sc * torch.sqrt(r))) / self.img_size  # [1,1,R,S]
        ah = (stride * (sc / torch.sqrt(r))) / self.img_size  # [1,1,R,S]

        cxg = cx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, aw.shape[-2], aw.shape[-1])
        cyg = cy.unsqueeze(-1).unsqueeze(-1).expand_as(cxg)

        anc = torch.stack([cxg, cyg, aw.expand_as(cxg), ah.expand_as(cyg)], dim=-1)  # [Hf,Wf,R,S,4]
        return anc.reshape(-1, 4)  # [Hf*Wf*A, 4] (cx,cy,w,h) normalized

    def forward(self, x):
        # ssdlite.forward -> {"cls_logits":[B,Ni,C], "bbox_regs":[B,Ni,4]}（每层一个元素）
        out = self.m(x)
        cls_list = out["cls_logits"]
        reg_list = out["bbox_regs"]
        assert isinstance(cls_list, (list, tuple)) and isinstance(reg_list, (list, tuple)), \
            "ssdlite.py 保持不动时应返回每层列表"

        B = cls_list[0].shape[0]
        device = cls_list[0].device
        dtype = cls_list[0].dtype

        # 依据 strides 和输入尺寸估算各层特征图尺寸 (要求导出输入被各 stride 整除)
        exp_sizes = []
        for s in self.strides:
            Hf = self.img_size // s
            Wf = self.img_size // s
            exp_sizes.append((Hf, Wf))

        # 生成各层 anchors，并拼接
        anchors_per_level = []
        for (Hf, Wf), s in zip(exp_sizes, self.strides):
            anchors_per_level.append(self._mk_anchors_one_level(Hf, Wf, s, device, dtype))
        anchors = torch.cat(anchors_per_level, dim=0)  # [N,4] normalized

        # 拼接分类与回归
        cls_logits = torch.cat(cls_list, dim=1)  # [B,N,C]
        bbox_regs  = torch.cat(reg_list,  dim=1) # [B,N,4]

        # 分类概率（可选去掉背景）
        probs = F.softmax(cls_logits, dim=-1)     # [B,N,C]
        if self.drop_background:
            probs_nb = probs[..., 1:]            # [B,N,C-1]
            scores, cls_idx = probs_nb.max(dim=-1)
            cls_idx = cls_idx + 1                # 补回背景偏移
        else:
            scores, cls_idx = probs.max(dim=-1)  # 含背景

        # 解码 bbox_regs (tx,ty,tw,th) + anchors(cx,cy,w,h)
        # SSD 常用变换：cx' = cx + tx*vx*w；  w' = w*exp(tw*vw)
        anc = anchors.unsqueeze(0).expand(B, -1, -1)        # [B,N,4]
        cx, cy, w, h = anc.unbind(dim=-1)
        tx, ty, tw, th = bbox_regs.unbind(dim=-1)

        cx_p = cx + tx * self.vx * w
        cy_p = cy + ty * self.vy * h
        w_p  = w  * torch.exp(tw * self.vw)
        h_p  = h  * torch.exp(th * self.vh)

        # (x1,y1,x2,y2) 归一化坐标
        x1 = cx_p - 0.5 * w_p
        y1 = cy_p - 0.5 * h_p
        x2 = cx_p + 0.5 * w_p
        y2 = cy_p + 0.5 * h_p
        boxes_norm = torch.stack([x1, y1, x2, y2], dim=-1)  # [B,N,4]

        # 转像素坐标（如需）
        if self.return_normalized:
            boxes = boxes_norm
        else:
            scale = torch.tensor([self.img_size, self.img_size, self.img_size, self.img_size], dtype=boxes_norm.dtype, device=boxes_norm.device)
            boxes = boxes_norm * scale

        dets = torch.cat([boxes, scores.unsqueeze(-1), cls_idx.to(boxes.dtype).unsqueeze(-1)], dim=-1)  # [B,N,6]
        return dets
