"""LSTM ``nn.Module`` definition used by the PyTorch trainer.

Single-layer LSTM + linear head; the only architecture shipped today.
"""

from __future__ import annotations

import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :])
