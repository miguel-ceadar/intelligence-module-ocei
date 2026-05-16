"""PyTorch LSTM implementing the ``Model`` protocol.

Single-target multivariate forecaster: the network consumes
``num_variables`` input series and predicts only the **target** (first
feature) across ``horizon`` future steps.

Expects components produced by ``make_lstm_prepare``:
  - ``X_train`` / ``X_test``: ``(samples, look_back, num_variables)`` tensors.
  - ``y_train`` / ``y_test``: ``(samples, horizon)`` tensors — target only.
  - ``train_dataset`` / ``test_dataset``: ``TimeSeriesDataset`` instances.
  - ``batch_size``.
  - ``scaler_X``: MinMaxScaler fit on the multivariate input data
    (one column per feature). Used to scale predict-time input windows.
  - ``scaler_obj``: MinMaxScaler fit on the target column only. Used
    to inverse-transform network output. Mirrors XGB's two-scaler
    pattern (y-scaler == ``scaler_obj``, x-scaler == ``scaler_X``).
  - ``model_parameters``: ``input_size``, ``output_size``, ``hidden_size``,
    ``num_epochs``. ``output_size`` is the trained horizon; predict
    refuses requests above it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from safetensors.torch import load_file, save_file
from sklearn.preprocessing import MinMaxScaler

from intelligence.api.schemas import ForecastPoint
from intelligence.ml.artifact.sidecars import (
    load_input_spec,
    load_json,
    load_sklearn_scaler,
    save_input_spec,
    save_json,
    save_sklearn_scaler,
)
from intelligence.ml.models._common import (
    assemble_predict_window,
    coerce_metrics,
    supervised_window,
)
from intelligence.ml.trainers import ModelTrainer
from intelligence.ml.trainers.base import TimeSeriesDataset
from intelligence.ml.trainers.lstm import LSTMModel as _LSTMModule

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
        params = {**self.default_params, **components.get("model_parameters", {})}
        components_with_params = {**components, "model_parameters": params}
        trainer = ModelTrainer(components_with_params)
        metrics, network, _step_epoch, _step_loss = trainer.train_pytorch()
        metrics_jsonable = coerce_metrics(metrics)

        artifacts = {
            "network": network,
            "scaler_obj": components_with_params["scaler_obj"],  # target scaler
            "scaler_X": components_with_params["scaler_X"],  # input scaler (multivariate)
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
        save_sklearn_scaler(dest, "scaler_y", artifacts["scaler_obj"])
        save_sklearn_scaler(dest, "scaler_x", artifacts["scaler_X"])
        save_json(dest, "metrics.json", artifacts.get("model_metrics", {}))

        files: dict[str, str] = {
            "model": "lstm.safetensors",
            "arch": "arch.json",
            "scaler_y_meta": "scaler_y.json",
            "scaler_y_arrays": "scaler_y.npz",
            "scaler_x_meta": "scaler_x.json",
            "scaler_x_arrays": "scaler_x.npz",
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
        arch = load_json(src, "arch.json")
        network = _LSTMModule(
            input_size=int(arch["input_size"]),
            hidden_size=int(arch["hidden_size"]),
            output_size=int(arch["output_size"]),
        )
        network.load_state_dict(load_file(str(src / "lstm.safetensors")))
        network.eval()

        loaded: dict[str, Any] = {
            "network": network,
            "scaler_obj": load_sklearn_scaler(src, "scaler_y"),  # target scaler
            "scaler_X": load_sklearn_scaler(src, "scaler_x"),  # input scaler
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
        trained_horizon = int(artifacts.get("horizon", artifacts.get("output_size", 1)))
        if horizon > trained_horizon:
            raise ValueError(
                f"horizon {horizon} exceeds trained output_size={trained_horizon}; "
                f"retrain with a larger output window or request a shorter horizon"
            )

        # ``assemble_predict_window`` handles the empty / short-window /
        # missing-feature failure modes and stacks the canonical
        # ``(look_back, num_variables)`` window LSTM scales below.
        window, _ = assemble_predict_window(artifacts, input_series)

        input_scaler = artifacts["scaler_X"]
        target_scaler = artifacts["scaler_obj"]
        scaled = input_scaler.transform(window)  # (look_back, num_variables)
        x = torch.from_numpy(scaled).float().unsqueeze(0)  # (1, look_back, num_variables)

        net = artifacts["network"]
        net.eval()
        with torch.no_grad():
            y_scaled = net(x).cpu().numpy()  # (1, trained_horizon)

        # Target-only output: inverse-transform via the target scaler.
        # Shape (1, H) → (H, 1) → inverse → flatten → truncate to requested horizon.
        y_raw = target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).reshape(-1)
        y_raw = y_raw[:horizon]

        return [ForecastPoint(value=round(float(v), 4)) for v in y_raw]


def make_lstm_prepare(
    look_back: int = 6,
    feature_names: list[str] | None = None,
    batch_size: int = 64,
    horizon: int = 1,
) -> Callable[[pd.DataFrame], dict]:
    """Build a ``prepare`` callable that produces LSTM-shaped components
    (3-D X tensors + ``TimeSeriesDataset`` instances).

    ``feature_names`` is the canonical feature order — target first,
    covariates after. Columns are looked up in the DataFrame by name,
    not by position. Required; builders always supply it.

    The supervised structure emits a ``(samples, horizon)`` y-tensor —
    target only across the horizon — paired with
    ``model_parameters["output_size"] = horizon``.

    Two scalers are fit:
      - ``scaler_X``: multivariate MinMax fit on all input columns,
        applied to every input window at train and predict time.
      - ``scaler_obj``: univariate MinMax fit on the target column,
        used to inverse-transform network output. The dual-scaler
        pattern matches XGB (``scaler_obj`` = y, ``scaler_X`` = x).
    """
    if not feature_names:
        raise ValueError("make_lstm_prepare requires a non-empty feature_names list")
    feature_names = list(feature_names)

    def prepare(df: pd.DataFrame) -> dict:
        missing = [n for n in feature_names if n not in df.columns]
        if missing:
            raise ValueError(
                f"make_lstm_prepare expected columns {list(feature_names)!r}, "
                f"missing {missing!r}; DataFrame has {list(df.columns)!r}"
            )
        cols = list(feature_names)
        num_variables = len(cols)
        data = df[cols].astype(float).reset_index(drop=True)

        split = int(len(data) * 0.8)
        train_df, test_df = data.iloc[:split], data.iloc[split:]

        scaler_X = MinMaxScaler().fit(train_df)
        scaler_y = MinMaxScaler().fit(train_df.iloc[:, 0:1])  # target only
        scaled_train = scaler_X.transform(train_df)
        scaled_test = scaler_X.transform(test_df)

        sup_train = supervised_window(scaled_train, n_in=look_back, n_out=horizon)
        sup_test = supervised_window(scaled_test, n_in=look_back, n_out=horizon)

        # Supervised columns are ordered (lag, var). The future section
        # (last horizon * num_variables cols) interleaves var1, var2, …
        # for each step; the target lives at offsets 0, V, 2V, …, so a
        # ``[:, -y_width::num_variables]`` slice extracts it.
        y_width = horizon * num_variables
        x_train = sup_train[:, :-y_width].reshape(-1, look_back, num_variables)
        x_test = sup_test[:, :-y_width].reshape(-1, look_back, num_variables)

        # Slice target columns out of the y section, then rescale them
        # via the dedicated target scaler. The values are currently in
        # scaler_X's column-0 space — MinMax is per-column so we
        # inverse-transform via the input scaler's col-0 first, then
        # re-fit through scaler_y. In practice the parameters are
        # identical (target_min == scaler_X.data_min_[0]) but going
        # through the explicit re-scale keeps the two scalers
        # interchangeable from the model's point of view.
        y_train_scaled_input = sup_train[:, -y_width::num_variables]  # (samples, horizon)
        y_test_scaled_input = sup_test[:, -y_width::num_variables]

        target_min = scaler_X.data_min_[0]
        target_range = scaler_X.data_range_[0]
        # Re-scale: undo input scaler col-0 → apply target scaler.
        # target scaler == col-0 of input scaler, so this is identity in
        # values; the indirection keeps the predict-time math honest.
        y_train_raw = y_train_scaled_input * target_range + target_min
        y_test_raw = y_test_scaled_input * target_range + target_min
        y_train = scaler_y.transform(y_train_raw.reshape(-1, 1)).reshape(y_train_raw.shape)
        y_test = scaler_y.transform(y_test_raw.reshape(-1, 1)).reshape(y_test_raw.shape)

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
            "scaler_obj": scaler_y,  # target scaler — for inverse on network output
            "scaler_X": scaler_X,  # input scaler — for transforming predict-time windows
            "look_back": look_back,
            "num_variables": num_variables,
            "horizon": horizon,
        }

    return prepare
