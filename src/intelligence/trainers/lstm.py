"""LSTM model definitions used by the PyTorch trainer.

Mirrors ``oasis/analytics/lstm_model.py`` while phase 1 is in flight.
The legacy module remains so existing pickled Bentos that reference
``analytics.lstm_model.LSTMModel`` keep loading; new training runs save
under this module path.
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


class LighterStudentLSTMModel(nn.Module):
    """Smaller student network used by the optional knowledge-distillation path."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, _ = self.gru(x)
        return self.fc(output[:, -1, :])
