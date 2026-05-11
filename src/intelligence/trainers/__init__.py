"""Plain-Python training surface — see ``trainers.base.ModelTrainer``.

Phase-1 extraction of ``oasis.analytics.model_metrics.ModelMetricsDataClay``
(see intelligence-utility-plan.v2.md §2.2).
"""

from intelligence.trainers._mlflow import mlflow_gc
from intelligence.trainers.base import ModelTrainer, TimeSeriesDataset
from intelligence.trainers.lstm import LSTMModel, LighterStudentLSTMModel
from intelligence.trainers.metrics import (
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
