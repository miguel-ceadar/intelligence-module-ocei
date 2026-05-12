"""Plain-Python training surface — see ``trainers.base.ModelTrainer``.

Three trainers (ARIMA, XGBoost, PyTorch-LSTM) plus shared helpers.
Each ``Model`` adapter in ``intelligence.ml.models`` is a thin wrapper
that prepares components and calls one of these.
"""

from intelligence.ml.trainers._mlflow import mlflow_gc
from intelligence.ml.trainers.base import ModelTrainer, TimeSeriesDataset
from intelligence.ml.trainers.lstm import LighterStudentLSTMModel, LSTMModel
from intelligence.ml.trainers.metrics import (
    metrics,
    metrics_pytorch,
    print_size_of_model,
    symmetric_mean_absolute_percentage_error,
)

__all__ = [
    "LSTMModel",
    "LighterStudentLSTMModel",
    "ModelTrainer",
    "TimeSeriesDataset",
    "metrics",
    "metrics_pytorch",
    "mlflow_gc",
    "print_size_of_model",
    "symmetric_mean_absolute_percentage_error",
]
