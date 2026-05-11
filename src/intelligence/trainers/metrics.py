"""Regression metric helpers used by ``ModelTrainer``."""

from __future__ import annotations

import logging
import os
import tempfile

import numpy as np
import torch
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

logger = logging.getLogger(__name__)


def symmetric_mean_absolute_percentage_error(y_true, y_pred) -> float:
    return 200 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))


def print_size_of_model(model, label: str = "") -> float:
    """Serialize the model state_dict and return its size in MB.

    The legacy implementation wrote ``temp.p`` into the current working
    directory; we use a real temp file so concurrent calls don't collide
    and a crash mid-call doesn't leave litter.
    """
    fd, path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    try:
        torch.save(model.state_dict(), path)
        size = os.path.getsize(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    logger.info("model: %s Size : %s (KB)", label, size / 1e3)
    return size / 1e6


def metrics(y_test, y_pred) -> dict:
    out: dict = {}
    out["mse"] = mean_squared_error(y_test, y_pred)
    out["rmse"] = float(np.sqrt(out["mse"]))
    out["mape"] = mean_absolute_percentage_error(y_test, y_pred)
    out["mae"] = round(mean_absolute_error(y_test, y_pred), 2)
    out["smape"] = round(symmetric_mean_absolute_percentage_error(y_test, y_pred), 2)
    out["r2"] = r2_score(y_test, y_pred)
    return out


def metrics_pytorch(model=None, y_test=0, y_pred=0) -> dict:
    out = metrics(y_test, y_pred)
    if model is not None:
        out["Model Size (MB)"] = print_size_of_model(model, "int8")
    return out
