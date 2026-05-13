"""PyTorch LSTM implementing the ``Model`` protocol.

Expects components produced by ``make_lstm_prepare``:
  - ``X_train`` / ``X_test``: ``(samples, look_back, num_variables)`` tensors.
  - ``y_train`` / ``y_test``: ``(samples, horizon * num_variables)`` tensors.
  - ``train_dataset`` / ``test_dataset``: ``TimeSeriesDataset`` instances.
  - ``batch_size``.
  - ``scaler_obj``: a single MinMaxScaler fit on the raw multivariate data.
  - ``model_parameters``: ``input_size``, ``output_size``, ``hidden_size``,
    ``num_epochs``. Optional: ``distill``. ``output_size`` is the trained
    horizon; predict refuses requests above it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from intelligence.api.schemas import ForecastPoint

logger = logging.getLogger(__name__)


class LstmModel:
    """PyTorch LSTM forecaster. Defaults are per-instance, so tasks
    reusing this model pick their own ``hidden_size``, ``num_epochs``,
    etc.
    """

    name = "lstm"
    has_drift = False

    def __init__(self, **default_params: Any) -> None:
        self.default_params = default_params or {
            "input_size": 1,
            "output_size": 1,
            "hidden_size": 4,
            "num_epochs": 3,
        }

    def fit(self, components: dict) -> tuple[dict, dict]:
        """Train and return ``(artifacts, metrics)``. ``artifacts``
        carries the trained network, the scaler, and the architecture
        sizes plus window metadata.
        """
        from intelligence.ml.trainers import ModelTrainer

        params = {**self.default_params, **components.get("model_parameters", {})}
        components_with_params = {**components, "model_parameters": params}
        trainer = ModelTrainer(components_with_params)
        metrics, network, _step_epoch, _step_loss = trainer.train_pytorch()
        metrics_jsonable = _coerce_jsonable_nested(metrics)

        artifacts = {
            "network": network,
            "scaler_obj": components_with_params["scaler_obj"],
            "look_back": components_with_params["look_back"],
            "num_variables": components_with_params["num_variables"],
            "input_size": params["input_size"],
            "output_size": params["output_size"],
            "hidden_size": params["hidden_size"],
            "horizon": params["output_size"],  # alias kept for predict clarity
            "model_metrics": metrics_jsonable,
        }
        return artifacts, metrics_jsonable

    def save_artifacts(self, artifacts: dict, dest: Path) -> dict[str, str]:
        """Persist the artifacts and return the ``role -> filename`` map.

        The state_dict goes to ``lstm.safetensors``. ``arch.json``
        carries the constructor sizes so ``load_artifacts`` can rebuild
        the network before loading weights.
        """
        from safetensors.torch import save_file

        from intelligence.ml.artifact.sidecars import (
            save_input_spec,
            save_json,
            save_sklearn_scaler,
        )

        network = artifacts["network"]
        network.cpu()  # keep weights portable across devices
        save_file(network.state_dict(), str(dest / "lstm.safetensors"))

        save_json(
            dest,
            "arch.json",
            {
                "input_size": int(artifacts["input_size"]),
                "hidden_size": int(artifacts["hidden_size"]),
                "output_size": int(artifacts["output_size"]),
                "look_back": int(artifacts["look_back"]),
                "num_variables": int(artifacts["num_variables"]),
            },
        )
        save_sklearn_scaler(dest, "scaler", artifacts["scaler_obj"])
        save_json(dest, "metrics.json", artifacts.get("model_metrics", {}))

        files: dict[str, str] = {
            "model": "lstm.safetensors",
            "arch": "arch.json",
            "scaler_meta": "scaler.json",
            "scaler_arrays": "scaler.npz",
            "metrics": "metrics.json",
        }
        spec = artifacts.get("input_spec")
        if spec is not None:
            save_input_spec(dest, spec)
            files["input_spec"] = "input_spec.json"
        return files

    def load_artifacts(self, src: Path) -> dict:
        """Restore the dict shape ``fit`` emits, plus ``input_spec``
        if it was persisted.
        """
        from safetensors.torch import load_file

        from intelligence.ml.artifact.sidecars import (
            load_input_spec,
            load_json,
            load_sklearn_scaler,
        )
        from intelligence.ml.trainers.lstm import LSTMModel

        arch = load_json(src, "arch.json")
        network = LSTMModel(
            input_size=int(arch["input_size"]),
            hidden_size=int(arch["hidden_size"]),
            output_size=int(arch["output_size"]),
        )
        network.load_state_dict(load_file(str(src / "lstm.safetensors")))
        network.eval()

        loaded: dict[str, Any] = {
            "network": network,
            "scaler_obj": load_sklearn_scaler(src, "scaler"),
            "look_back": int(arch["look_back"]),
            "num_variables": int(arch["num_variables"]),
            "input_size": int(arch["input_size"]),
            "hidden_size": int(arch["hidden_size"]),
            "output_size": int(arch["output_size"]),
            "horizon": int(arch["output_size"]),
            "model_metrics": load_json(src, "metrics.json"),
        }
        if (src / "input_spec.json").exists():
            loaded["input_spec"] = load_input_spec(src)
        return loaded

    def predict(
        self,
        artifacts: dict,
        input_series: dict[str, list[float]],
        horizon: int = 1,
    ) -> list[ForecastPoint]:
        """Direct multi-output: the network emits ``trained_horizon``
        values in a single forward pass. Requests above
        ``trained_horizon`` are refused; below it, the output is
        truncated. No confidence intervals — ``lower``/``upper`` stay
        ``None``.
        """
        import torch

        if not input_series:
            raise ValueError("input_series is empty")

        look_back = int(artifacts["look_back"])
        num_variables = int(artifacts["num_variables"])
        scaler = artifacts["scaler_obj"]
        trained_horizon = int(artifacts.get("horizon", artifacts.get("output_size", 1)))

        if horizon > trained_horizon:
            raise ValueError(
                f"horizon {horizon} exceeds trained output_size={trained_horizon}; "
                f"retrain with a larger output window or request a shorter horizon"
            )

        # Stack the input series in declared order, take the last ``look_back``
        # observations across all variables → shape (look_back, num_variables).
        series_keys = list(input_series.keys())[:num_variables]
        if len(series_keys) < num_variables:
            raise ValueError(f"need {num_variables} input series, got {len(series_keys)}")
        window = np.column_stack(
            [np.asarray(input_series[k], dtype=float)[-look_back:] for k in series_keys]
        )
        if window.shape[0] < look_back:
            raise ValueError(
                f"need at least {look_back} observations per series, got {window.shape[0]}"
            )

        scaled = scaler.transform(window)  # (look_back, num_variables)
        x = torch.from_numpy(scaled).float().unsqueeze(0)  # (1, look_back, num_variables)

        net = artifacts["network"]
        net.eval()
        with torch.no_grad():
            y_scaled = net(x).cpu().numpy()  # (1, trained_horizon * num_variables)

        # The scaler was fit on (n, num_variables); inverse_transform expects
        # that shape regardless of how many horizon steps we're decoding.
        # Reshape (1, H * V) -> (H, V), inverse, then take the first `horizon` rows.
        y_flat = y_scaled.reshape(trained_horizon, num_variables)
        y_raw = scaler.inverse_transform(y_flat)  # (trained_horizon, num_variables)
        y_raw = y_raw[:horizon]  # truncate to requested

        if num_variables == 1:
            return [ForecastPoint(value=round(float(v), 4)) for v in y_raw[:, 0]]
        # Multivariate output is not yet supported; return the first
        # variable so the shape contract is preserved.
        return [ForecastPoint(value=round(float(row[0]), 4)) for row in y_raw]


def make_lstm_prepare(
    look_back: int = 6,
    num_variables: int = 1,
    batch_size: int = 64,
    horizon: int = 1,
) -> Callable[[pd.DataFrame], dict]:
    """Build a ``prepare`` callable that produces LSTM-shaped components
    (3-D X tensors + ``TimeSeriesDataset`` instances).

    ``horizon`` controls the target window: the supervised structure
    emits a ``(samples, horizon * num_variables)`` y-tensor so the
    network can be trained for multi-step direct output. Pair with
    ``model_parameters["output_size"] = horizon * num_variables``.
    """

    def prepare(df: pd.DataFrame) -> dict:
        import torch

        from intelligence.ml.trainers.base import TimeSeriesDataset

        cols = [c for c in df.columns if c.lower() not in {"time", "timestamp", "date"}][
            :num_variables
        ]
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

        sup_train = _ts_supervised_structure(scaled_train, n_in=look_back, n_out=horizon)
        sup_test = _ts_supervised_structure(scaled_test, n_in=look_back, n_out=horizon)

        y_width = horizon * num_variables
        x_train = sup_train[:, :-y_width].reshape(-1, look_back, num_variables)
        y_train = sup_train[:, -y_width:]
        x_test = sup_test[:, :-y_width].reshape(-1, look_back, num_variables)
        y_test = sup_test[:, -y_width:]

        x_train_t = torch.from_numpy(x_train).float()
        y_train_t = torch.from_numpy(y_train).float()
        x_test_t = torch.from_numpy(x_test).float()
        y_test_t = torch.from_numpy(y_test).float()

        return {
            "X_train": x_train_t,
            "X_test": x_test_t,
            "y_train": y_train_t,
            "y_test": y_test_t.numpy(),  # trainer inverse-transforms this; needs ndarray
            "train_dataset": TimeSeriesDataset(x_train_t, y_train_t),
            "test_dataset": TimeSeriesDataset(x_test_t, y_test_t),
            "batch_size": batch_size,
            "scaler_obj": scaler,
            "look_back": look_back,
            "num_variables": num_variables,
            "horizon": horizon,
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
