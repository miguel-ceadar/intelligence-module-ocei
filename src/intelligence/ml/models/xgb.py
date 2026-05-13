"""XGBoost model.

Wraps ``ModelTrainer.train_xgb`` for the ``Model`` contract.

``train_xgb`` expects components produced by ``make_xgb_prepare``:
  - ``X_train`` / ``X_test``: 2-D arrays of supervised-structure features
    (lagged values), normalized by ``scaler_X``.
  - ``y_train`` (normalized) / ``y_test`` (unnormalized — the trainer
    inverse-transforms ``y_pred`` and compares to raw ``y_test``).
  - ``scaler_obj``: ``scaler_y`` — used by the trainer to inverse-transform
    predictions.
  - ``scaler_X``: the X-side scaler — saved in custom_objects so
    ``predict`` can normalize fresh input windows.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from intelligence.api.schemas import ForecastPoint

logger = logging.getLogger(__name__)


class XgbModel:
    """XGBoost regressor.

    Defaults are per-instance — two tasks reusing this model can carry
    different ``n_estimators``/``max_depth``/etc. without subclassing.
    """

    name = "xgb"
    has_drift = True

    def __init__(self, **default_params: Any) -> None:
        self.default_params = default_params or {
            "n_estimators": 100,
            "max_depth": 3,
            "eta": 0.1,
        }

    # ---- New manifest-based protocol --------------------------------------

    def fit(self, components: dict) -> tuple[dict, dict]:
        """Train and return ``(artifacts, metrics)``.

        ``artifacts`` carries the runtime state predict consumes: the
        fitted ``XGBRegressor``, both scalers (X and y), and the
        sliding-window length. ``BaseTask`` injects ``input_spec`` into
        this dict before calling ``save_artifacts``.
        """
        from intelligence.ml.trainers import ModelTrainer

        params = {**self.default_params, **components.get("model_parameters", {})}
        components_with_params = {**components, "model_parameters": params}

        trainer = ModelTrainer(components_with_params)
        metrics, regressor = trainer.train_xgb()
        metrics_jsonable = _coerce_jsonable(metrics)

        artifacts = {
            "regressor": regressor,
            "scaler_obj": components_with_params["scaler_obj"],  # y-scaler
            "scaler_X": components_with_params["scaler_X"],
            "look_back": components_with_params["look_back"],
            "model_metrics": metrics_jsonable,
            "test_sample_size": len(components_with_params["X_test"]),
        }
        return artifacts, metrics_jsonable

    def save_artifacts(self, artifacts: dict, dest: Path) -> dict[str, str]:
        """Persist the artefacts as a flat directory and return the
        ``role -> filename`` map for the manifest.

        The model itself goes to ``xgb.ubj`` via xgboost's native
        universal binary JSON format — bit-exact round-trip without
        pickle. Both scalers persist as JSON + NPZ sidecars.
        """
        from intelligence.ml.artifact.sidecars import (
            save_input_spec,
            save_json,
            save_sklearn_scaler,
        )

        regressor = artifacts["regressor"]
        # xgboost 1.7.6 (pinned, see pyproject for the reason) calls
        # ``_get_type()`` during save; sklearn 1.8 removed
        # ``_estimator_type`` from ``RegressorMixin`` so the attribute
        # isn't inherited any more. Set it explicitly — harmless on
        # older sklearn, fixes the save on newer sklearn.
        if not hasattr(regressor, "_estimator_type"):
            regressor._estimator_type = "regressor"
        regressor.save_model(str(dest / "xgb.ubj"))

        save_sklearn_scaler(dest, "scaler_x", artifacts["scaler_X"])
        save_sklearn_scaler(dest, "scaler_y", artifacts["scaler_obj"])
        save_json(
            dest,
            "xgb_meta.json",
            {
                "look_back": int(artifacts["look_back"]),
                "test_sample_size": int(artifacts.get("test_sample_size", 0)),
            },
        )
        save_json(dest, "metrics.json", artifacts.get("model_metrics", {}))

        files: dict[str, str] = {
            "model": "xgb.ubj",
            "scaler_x_meta": "scaler_x.json",
            "scaler_x_arrays": "scaler_x.npz",
            "scaler_y_meta": "scaler_y.json",
            "scaler_y_arrays": "scaler_y.npz",
            "meta": "xgb_meta.json",
            "metrics": "metrics.json",
        }
        spec = artifacts.get("input_spec")
        if spec is not None:
            save_input_spec(dest, spec)
            files["input_spec"] = "input_spec.json"
        return files

    def load_artifacts(self, src: Path) -> dict:
        """Inverse of :meth:`save_artifacts` — returns the same dict
        shape that :meth:`fit` emits, plus ``input_spec`` if persisted.
        """
        from xgboost import XGBRegressor

        from intelligence.ml.artifact.sidecars import (
            load_input_spec,
            load_json,
            load_sklearn_scaler,
        )

        regressor = XGBRegressor()
        # See ``save_artifacts`` — same xgboost-1.7.6 / sklearn-1.8 quirk.
        if not hasattr(regressor, "_estimator_type"):
            regressor._estimator_type = "regressor"
        regressor.load_model(str(src / "xgb.ubj"))

        meta = load_json(src, "xgb_meta.json")
        loaded: dict[str, Any] = {
            "regressor": regressor,
            "scaler_obj": load_sklearn_scaler(src, "scaler_y"),
            "scaler_X": load_sklearn_scaler(src, "scaler_x"),
            "look_back": int(meta["look_back"]),
            "model_metrics": load_json(src, "metrics.json"),
            "test_sample_size": int(meta.get("test_sample_size", 0)),
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
        """Recursive multi-horizon: predict t+1, slide it into the window,
        predict t+2, …, ``horizon`` times. No native confidence intervals
        — ``lower``/``upper`` stay ``None`` on every ``ForecastPoint``.
        Quantile XGB / bootstrap CIs are deferred (memory: roadmap-waves).
        """
        if not input_series:
            raise ValueError("input_series is empty")
        # Univariate — pick the first series.
        _key, values = next(iter(input_series.items()))
        look_back = int(artifacts["look_back"])
        if len(values) < look_back:
            raise ValueError(f"need at least {look_back} observations, got {len(values)}")

        scaler_x = artifacts["scaler_X"]
        scaler_y = artifacts["scaler_obj"]
        cols = getattr(scaler_x, "feature_names_in_", None)
        regressor = artifacts["regressor"]

        window = [float(v) for v in values[-look_back:]]
        forecasts: list[float] = []
        for _ in range(horizon):
            x = np.array(window, dtype=float).reshape(1, look_back)
            if cols is not None:
                x_scaled = scaler_x.transform(pd.DataFrame(x, columns=cols))
            else:
                x_scaled = scaler_x.transform(x)
            y_scaled = regressor.predict(x_scaled)
            y_raw = float(scaler_y.inverse_transform(np.asarray(y_scaled).reshape(-1, 1))[0, 0])
            forecasts.append(y_raw)
            # Slide the window forward: drop oldest, append fresh prediction.
            window = [*window[1:], y_raw]

        return [ForecastPoint(value=round(v, 4)) for v in forecasts]


def make_xgb_prepare(
    look_back: int = 6,
    num_variables: int = 1,
) -> Callable[[pd.DataFrame], dict]:
    """Build a ``prepare`` callable that produces XGB-shaped components
    from a raw DataFrame.

    Pipeline: pick numeric columns → ``ts_supervised_structure`` (lagged
    features + target) → split 80/20 → fit separate ``StandardScaler``s
    for X and y → return components dict expected by ``ModelTrainer.train_xgb``.
    """

    def prepare(df: pd.DataFrame) -> dict:
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

        sup_train = _ts_supervised_structure(train_df, n_in=look_back, n_out=1)
        sup_test = _ts_supervised_structure(test_df, n_in=look_back, n_out=1)

        x_train = sup_train.iloc[:, :-num_variables]
        y_train = sup_train.iloc[:, -num_variables].squeeze()
        x_test = sup_test.iloc[:, :-num_variables]
        y_test = sup_test.iloc[:, -num_variables].squeeze()

        scaler_x = StandardScaler().fit(x_train)
        x_train_n = pd.DataFrame(scaler_x.transform(x_train), columns=x_train.columns)
        x_test_n = pd.DataFrame(scaler_x.transform(x_test), columns=x_test.columns)

        scaler_y = StandardScaler().fit(np.asarray(y_train).reshape(-1, 1))
        y_train_n = scaler_y.transform(np.asarray(y_train).reshape(-1, 1)).flatten()

        return {
            "X_train": x_train_n,
            "X_test": x_test_n,
            "y_train": y_train_n,
            "y_test": np.asarray(y_test),  # unnormalized — trainer compares against this
            "scaler_obj": scaler_y,  # y-scaler (used by trainer for inverse_transform)
            "scaler_X": scaler_x,  # x-scaler (saved in Bento for predict)
            "look_back": look_back,
        }

    return prepare


def _ts_supervised_structure(data: pd.DataFrame, n_in: int, n_out: int = 1) -> pd.DataFrame:
    """Lag-feature reshape: build columns ``var1(t-n_in)`` ... ``var1(t)``
    for autoregressive supervised learning.

    Keeps a column-name convention (``var{j}(t-{i})``) so saved
    ``StandardScaler``s carry the right ``feature_names_in_`` and can
    transform predict-time windows without name mismatch errors.
    """
    n_vars = data.shape[1]
    cols: list[pd.DataFrame] = []
    names: list[str] = []
    for i in range(n_in, 0, -1):
        cols.append(data.shift(i))
        names += [f"var{j + 1}(t-{i})" for j in range(n_vars)]
    for i in range(n_out):
        cols.append(data.shift(-i))
        if i == 0:
            names += [f"var{j + 1}(t)" for j in range(n_vars)]
        else:
            names += [f"var{j + 1}(t+{i})" for j in range(n_vars)]
    out = pd.concat(cols, axis=1)
    out.columns = names
    return out.dropna()


def _coerce_jsonable(metrics: dict) -> dict:
    return {k: v.item() if hasattr(v, "item") else v for k, v in metrics.items()}
