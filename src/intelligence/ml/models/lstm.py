"""LSTM model.

Wraps ``ModelTrainer.train_pytorch`` for the ``Model`` contract.

``train_pytorch`` expects components produced by ``make_lstm_prepare``:
  - ``X_train`` / ``X_test``: torch tensors shape ``(samples, look_back, num_variables)``.
  - ``y_train`` / ``y_test``: torch tensors shape ``(samples, num_variables)``.
  - ``train_dataset`` / ``test_dataset``: ``TimeSeriesDataset`` instances.
  - ``batch_size``.
  - ``scaler_obj``: a single MinMaxScaler fit on the raw multivariate data,
    used to inverse-transform predictions and to normalize predict input.
  - ``model_parameters``: ``input_size``, ``output_size``, ``hidden_size``,
    ``num_epochs``. Optional: ``distill``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


class LstmModel:
    """PyTorch LSTM forecaster.

    Defaults are per-instance — different tasks reusing this model
    pick their own ``hidden_size``, ``num_epochs``, etc.
    """

    name = "lstm"
    has_drift = False  # NannyML drift only wired for ARIMA/XGB in the legacy code

    def __init__(self, **default_params: Any) -> None:
        self.default_params = default_params or {
            "input_size": 1,
            "output_size": 1,
            "hidden_size": 4,
            "num_epochs": 3,
        }

    def train(
        self,
        components: dict,
        bento_name: str,
        extras: dict | None = None,
    ) -> tuple[Any, dict]:
        from intelligence.ml.trainers import ModelTrainer

        params = {**self.default_params, **components.get("model_parameters", {})}
        components_with_params = {**components, "model_parameters": params}

        trainer = ModelTrainer(components_with_params)
        metrics, model, _step_epoch, _step_loss = trainer.train_pytorch()

        custom_objects = {
            "scaler_obj": components_with_params["scaler_obj"],
            "look_back": components_with_params["look_back"],
            "num_variables": components_with_params["num_variables"],
            "input_size": params["input_size"],
            "output_size": params["output_size"],
            "hidden_size": params["hidden_size"],
            "model_metrics": metrics,
            **(extras or {}),
        }

        import bentoml
        bento = bentoml.picklable_model.save_model(
            bento_name,
            model,
            custom_objects=custom_objects,
            signatures={"predict": {"batchable": True}},
        )
        return bento, _coerce_jsonable_nested(metrics)

    def predict(self, bento_model: Any, input_series: dict[str, list[float]]) -> Any:
        import torch

        if not input_series:
            raise ValueError("input_series is empty")

        look_back = int(bento_model.custom_objects["look_back"])
        num_variables = int(bento_model.custom_objects["num_variables"])
        scaler = bento_model.custom_objects["scaler_obj"]

        # Stack the input series in declared order, take the last ``look_back``
        # observations across all variables → shape (look_back, num_variables).
        series_keys = list(input_series.keys())[:num_variables]
        if len(series_keys) < num_variables:
            raise ValueError(
                f"need {num_variables} input series, got {len(series_keys)}"
            )
        window = np.column_stack([
            np.asarray(input_series[k], dtype=float)[-look_back:]
            for k in series_keys
        ])
        if window.shape[0] < look_back:
            raise ValueError(
                f"need at least {look_back} observations per series, got {window.shape[0]}"
            )

        scaled = scaler.transform(window)                # (look_back, num_variables)
        x = torch.from_numpy(scaled).float().unsqueeze(0)  # (1, look_back, num_variables)

        net = bento_model.load_model()  # actual nn.Module
        net.eval()
        with torch.no_grad():
            y_scaled = net(x).cpu().numpy()              # (1, num_variables)
        y = scaler.inverse_transform(y_scaled)
        if num_variables == 1:
            return round(float(y[0, 0]), 4)
        return [round(float(v), 4) for v in y[0]]


def make_lstm_prepare(
    look_back: int = 6,
    num_variables: int = 1,
    batch_size: int = 64,
) -> Callable[[pd.DataFrame], dict]:
    """Build a ``prepare`` callable that produces LSTM-shaped components
    (3-D X tensors + ``TimeSeriesDataset`` instances).
    """

    def prepare(df: pd.DataFrame) -> dict:
        import torch

        from intelligence.ml.trainers.base import TimeSeriesDataset

        cols = [
            c for c in df.columns
            if c.lower() not in {"time", "timestamp", "date"}
        ][:num_variables]
        if len(cols) < num_variables:
            raise ValueError(
                f"expected {num_variables} numeric column(s), found {len(cols)}: {cols}"
            )
        data = df[cols].astype(float).reset_index(drop=True)

        split = int(len(data) * 0.8)
        train_df, test_df = data.iloc[:split], data.iloc[split:]

        scaler = MinMaxScaler().fit(train_df)
        scaled_train = scaler.transform(train_df)
        scaled_test = scaler.transform(test_df)

        sup_train = _ts_supervised_structure(scaled_train, n_in=look_back, n_out=1)
        sup_test = _ts_supervised_structure(scaled_test, n_in=look_back, n_out=1)

        x_train = sup_train[:, :-num_variables].reshape(-1, look_back, num_variables)
        y_train = sup_train[:, -num_variables:]
        x_test = sup_test[:, :-num_variables].reshape(-1, look_back, num_variables)
        y_test = sup_test[:, -num_variables:]

        x_train_t = torch.from_numpy(x_train).float()
        y_train_t = torch.from_numpy(y_train).float()
        x_test_t = torch.from_numpy(x_test).float()
        y_test_t = torch.from_numpy(y_test).float()

        return {
            "X_train": x_train_t,
            "X_test": x_test_t,
            "y_train": y_train_t,
            "y_test": y_test_t.numpy(),   # trainer inverse-transforms this; needs ndarray
            "train_dataset": TimeSeriesDataset(x_train_t, y_train_t),
            "test_dataset": TimeSeriesDataset(x_test_t, y_test_t),
            "batch_size": batch_size,
            "scaler_obj": scaler,
            "look_back": look_back,
            "num_variables": num_variables,
        }

    return prepare


def _ts_supervised_structure(data: np.ndarray, n_in: int, n_out: int = 1) -> np.ndarray:
    """Numpy version of supervised-structure reshape — operates on the
    already-scaled 2-D array and returns a 2-D supervised array of
    width ``(n_in + n_out) * n_vars`` after dropping NaN rows.
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    n_rows, n_vars = data.shape
    window = n_in + n_out
    if n_rows < window:
        raise ValueError(
            f"need at least {window} rows for n_in={n_in}, n_out={n_out}; got {n_rows}"
        )
    out = np.empty((n_rows - window + 1, window * n_vars), dtype=float)
    for i in range(out.shape[0]):
        out[i] = data[i : i + window].reshape(-1)
    return out


def _coerce_jsonable_nested(metrics: dict) -> dict:
    """LSTM ``train_pytorch`` returns ``{metric_0: {...}, metric_1: {...}}``.
    Flatten ``.item()`` calls one level deep.
    """
    out: dict = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            out[k] = {kk: vv.item() if hasattr(vv, "item") else vv for kk, vv in v.items()}
        else:
            out[k] = v.item() if hasattr(v, "item") else v
    return out
