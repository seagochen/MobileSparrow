import torch.nn as nn

from sparrow.models.backbones.mobilenet_v2 import MobileNetV2Backbone
from sparrow.models.backbones.mobilenet_v3 import MobileNetV3Backbone
from sparrow.models.backbones.shufflenet_v2 import ShuffleNetV2Backbone
from sparrow.models.heads.crnn_ctc_head import CRNNCTCHead
from sparrow.models.necks.fpn_lite_kpts import FPNLiteKpts
from sparrow.models.necks.sequence_neck import SequenceNeck1D

BACKBONES = {
    "mobilenet_v2": MobileNetV2Backbone,
    "mobilenet_v3": MobileNetV3Backbone,
    "shufflenet_v2": ShuffleNetV2Backbone,
}

class OCRRecCTC(nn.Module):
    """Lightweight text recognizer: Backbone -> (FPN-Lite single-scale) -> SequenceNeck -> CRNN-CTC.
    Args:
        backbone: {'mobilenet_v2','mobilenet_v3','shufflenet_v2'}
        width_mult: channel scaling
        neck_outc: single-scale feature channels
        seq_midc/outc: sequence neck channels
        rnn_hidden: BiLSTM hidden size
        num_classes: charset size including CTC blank
    Forward returns:
        dict with {'logits': (B,T,num_classes)}
    """
    def __init__(self, backbone: str = "mobilenet_v3", width_mult: float = 1.0,
                 neck_outc: int = 128, seq_midc: int = 128, seq_outc: int = 256,
                 rnn_hidden: int = 256, num_classes: int = 96):
        super().__init__()
        assert backbone in BACKBONES, f"unknown backbone: {backbone}"
        self.backbone = BACKBONES[backbone](width_mult=width_mult)
        c3, c4, c5 = self.backbone.get_out_channels()
        self.neck = FPNLiteKpts(c3, c4, c5, outc=neck_outc)  # produces a single P3-like map
        self.seq = SequenceNeck1D(neck_outc, seq_midc, seq_outc)
        self.head = CRNNCTCHead(seq_outc, hidden=rnn_hidden, num_classes=num_classes)

    def forward(self, x):
        c3, c4, c5 = self.backbone(x)
        p3 = self.neck(c3, c4, c5)  # (B,C,H',W')
        seq = self.seq(p3)          # (B,T,C)
        logits = self.head(seq)     # (B,T,K)
        return {"logits": logits}


def ctc_greedy_decode(logits, blank_id=0):
    """logits: (B,T,K) -> list[str] using argmax collapse; caller maps ids to chars."""
    probs = logits.softmax(dim=-1)
    ids = probs.argmax(dim=-1).cpu().numpy()  # (B,T)
    texts = []
    for seq in ids:
        out = []
        prev = -1
        for t in seq:
            if t != blank_id and t != prev:
                out.append(int(t))
            prev = t
        texts.append(out)
    return texts  # list[list[int]]; map with charset later
