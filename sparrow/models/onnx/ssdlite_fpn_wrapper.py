import torch
import torch.nn as nn


class SSDLiteExportWrapper(nn.Module):
    """
    将 SSDLite_FPN 输出打包为 [B, 84, S] 格式，
    方便导出到 ONNX / TensorRT。
    """

    def __init__(self, ssd_model, conf_activation="sigmoid"):
        super().__init__()
        self.ssd = ssd_model
        self.conf_activation = conf_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: [B, 3, H, W]
        输出:
            out: [B, 84, S]
        """
        preds = self.ssd(x)
        cls_logits = preds["cls_logits"]  # [B, A, C]
        bbox_deltas = preds["bbox_deltas"]  # [B, A, 4]
        anchors = preds["anchors"]  # [B, A, 4]
        B, A, C = cls_logits.shape

        # 1 激活分类得分
        if self.conf_activation == "sigmoid":
            cls_scores = torch.sigmoid(cls_logits)
        else:
            cls_scores = torch.softmax(cls_logits, dim=-1)

        # 2 解码框（简单中心偏移解码，可按训练时定义调整）
        # 这里假设 bbox_deltas = [tx, ty, tw, th]
        # anchors = [x1, y1, x2, y2] (xyxy 格式)
        anchor_w = anchors[..., 2] - anchors[..., 0]
        anchor_h = anchors[..., 3] - anchors[..., 1]
        anchor_cx = anchors[..., 0] + 0.5 * anchor_w
        anchor_cy = anchors[..., 1] + 0.5 * anchor_h

        tx, ty, tw, th = bbox_deltas.unbind(-1)
        pred_cx = tx * anchor_w + anchor_cx
        pred_cy = ty * anchor_h + anchor_cy
        pred_w = torch.exp(tw) * anchor_w
        pred_h = torch.exp(th) * anchor_h

        lx = pred_cx - 0.5 * pred_w
        ly = pred_cy - 0.5 * pred_h
        rx = pred_cx + 0.5 * pred_w
        ry = pred_cy + 0.5 * pred_h
        boxes = torch.stack([lx, ly, rx, ry], dim=-1)  # [B, A, 4]

        # 3 拼接 bbox + scores -> [B, A, 84]
        out = torch.cat([boxes, cls_scores], dim=-1)  # [B, A, 84]

        return out
