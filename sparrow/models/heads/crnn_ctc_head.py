import torch.nn as nn

class CRNNCTCHead(nn.Module):
    """BiLSTM + Linear -> logits for CTC.
    Input: sequence features (B, T, C)
    Output: logits (B, T, num_classes)
    """
    def __init__(self, in_ch: int, hidden: int, num_classes: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.rnn = nn.LSTM(input_size=in_ch, hidden_size=hidden, num_layers=num_layers,
                           dropout=dropout if num_layers > 1 else 0.0, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, seq):  # (B,T,C)
        y, _ = self.rnn(seq)
        logits = self.fc(y)   # (B,T,num_classes)
        return logits
