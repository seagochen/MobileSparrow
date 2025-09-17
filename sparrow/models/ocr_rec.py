import torch
import torch.nn as nn
import torch.nn.functional as F

from sparrow.models.backbones.mobilenet_v2 import MobileNetV2Backbone
from sparrow.models.backbones.mobilenet_v3 import MobileNetV3Backbone
from sparrow.models.backbones.shufflenet_v2 import ShuffleNetV2Backbone

BACKBONES = {
    "mobilenet_v2": MobileNetV2Backbone,
    "mobilenet_v3": MobileNetV3Backbone,
    "shufflenet_v2": ShuffleNetV2Backbone,
}


class OCRRecCTC(nn.Module):
    """Backbone -> FPNLite(P3) -> SequenceNeck1D -> CRNN-CTC"""
    def __init__(self,
                 backbone: str = "mobilenet_v3",
                 width_mult: float = 1.0,
                 neck_outc: int = 128,
                 seq_midc: int = 128,
                 seq_outc: int = 256,
                 rnn_hidden: int = 256,
                 num_classes: int = 96):
        super().__init__()
        assert backbone in BACKBONES, f"unknown backbone: {backbone}"
        self.backbone = BACKBONES[backbone](width_mult=width_mult)

        # 获取 C3/C4/C5 通道
        c3c, c4c, c5c = self.backbone.get_out_channels()

        # FPN (单尺度输出 P3)
        self.neck = FPNLite(c3c, c4c, c5c, outc=neck_outc)

        # 2D -> 1D 时序
        self.seq = SequenceNeck1D(neck_outc, seq_midc, seq_outc)

        # CTC 头
        self.head = CRNNCTCHead(seq_outc, hidden=rnn_hidden, num_classes=num_classes)

    def forward(self, x):
        # Backbone
        c3, c4, c5 = self.backbone(x)
        # FPN -> P3
        p3 = self.neck(c3, c4, c5)   # (B,C,H',W')
        # 2D -> 1D
        seq = self.seq(p3)           # (B,T,C)
        # CTC logits
        logits = self.head(seq)      # (B,T,K)
        return {"logits": logits}

    @staticmethod
    @torch.no_grad()
    def ctc_greedy_decode(logits: torch.Tensor, blank_id: int = 0, merge_repeats: bool = True):
        """
        logits: (B,T,K) -> List[List[int]]
        贪心解码（可选合并重复），仅返回类别 id 序列；字符映射由调用方完成。
        """
        # 直接对 logits 取 argmax 即可
        ids = logits.argmax(dim=-1)  # (B,T)
        B, T = ids.shape
        results = []
        for b in range(B):
            out = []
            prev = None
            for t in range(T):
                idx = int(ids[b, t].item())
                if idx == blank_id:
                    prev = idx
                    continue
                if merge_repeats and prev == idx:
                    # 跳过重复
                    prev = idx
                    continue
                out.append(idx)
                prev = idx
            results.append(out)
        return results


# ---------------------------
# FPNLite: C3/C4/C5 -> 单尺度 P3
# ---------------------------
class FPNLite(nn.Module):
    """
    输入: C3, C4, C5
    输出: P3  (通道 = outc, 尺度约为 1/8 输入)
    结构: 1x1对齐 + 自顶向下加和 + 3x3平滑
    """
    def __init__(self, c3: int, c4: int, c5: int, outc: int = 128):
        super().__init__()
        # 1x1 对齐
        self.l3 = nn.Sequential(
            nn.Conv2d(c3, outc, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True),
        )
        self.l4 = nn.Sequential(
            nn.Conv2d(c4, outc, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True),
        )
        self.l5 = nn.Sequential(
            nn.Conv2d(c5, outc, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True),
        )
        # 3x3 平滑
        self.smooth4 = nn.Sequential(
            nn.Conv2d(outc, outc, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True),
        )
        self.smooth3 = nn.Sequential(
            nn.Conv2d(outc, outc, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True),
        )

    def forward(self, c3: torch.Tensor, c4: torch.Tensor, c5: torch.Tensor) -> torch.Tensor:
        # P5
        p5 = self.l5(c5)
        # P4 = L4(C4) + up(P5)
        p4 = self.l4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p4 = self.smooth4(p4)
        # P3 = L3(C3) + up(P4)
        p3 = self.l3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p3 = self.smooth3(p3)
        return p3


class CRNNCTCHead(nn.Module):
    """BiLSTM + Linear -> logits for CTC.
    Input: sequence features (B, T, C)
    Output: logits (B, T, num_classes)
    """
    def __init__(self, in_ch: int, hidden: int, num_classes: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=in_ch,
            hidden_size=hidden,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True
        )
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, seq):  # (B,T,C)
        y, _ = self.rnn(seq)
        logits = self.fc(y)   # (B,T,num_classes)
        return logits


class SequenceNeck1D(nn.Module):
    """2D -> 1D 时序特征: 小卷积塔后沿高度自适应均值到 1，再转成 (B,T,C)。"""
    def __init__(self, in_ch: int, mid_ch: int = 128, out_ch: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            # 等价于 conv_utils.conv3x3(in_ch, mid_ch)
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            # 等价于 conv_utils.conv1x1(mid_ch, out_ch)
            nn.Conv2d(mid_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):  # x: (B,C,H,W)
        x = self.conv(x)
        x = F.adaptive_avg_pool2d(x, (1, None))   # (B,C,1,W')
        x = x.squeeze(2).permute(0, 2, 1).contiguous()  # (B,T,C)
        return x


if __name__ == "__main__":
    # 简单自测（以常见 OCR 输入 32x128 为例）
    model = OCRRecCTC(backbone="mobilenet_v3", width_mult=1.0,
                      neck_outc=128, seq_midc=128, seq_outc=256,
                      rnn_hidden=256, num_classes=96).eval()
    x = torch.randn(2, 3, 32, 128)  # B=2
    with torch.no_grad():
        out = model(x)
    logits = out["logits"]
    print("logits:", logits.shape)  # (B, T, K)

    # 贪心解码（示例）
    ids = model.ctc_greedy_decode(logits, blank_id=0)
    print("decoded ids:", ids)
