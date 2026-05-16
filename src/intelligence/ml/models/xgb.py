"""XGBoost model implementing the ``Model`` protocol.

Expects components produced by ``make_xgb_prepare``:
  - ``X_train`` / ``X_test``: 2-D lag-feature arrays scaled by ``scaler_X``.
  - ``y_train`` (scaled) / ``y_test`` (raw — trainer inverse-transforms
    ``y_pred`` and compares).
  - ``scaler_obj``: the y-scaler.
  - ``scaler_X``: the x-scaler, saved so ``predict`` can scale fresh
    input windows.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

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
    supervised_window_columns,
)
from intelligence.ml.trainers import ModelTrainer

logger = logging.getLogger(__name__)


def _ensure_estimator_type(reg) -> None:
    """xgboost 1.7.6 expects ``_estimator_type`` on the regressor during
    save and load, but sklearn 1.8 dropped it from ``RegressorMixin``.
    Set it explicitly — harmless on older sklearn.
    """
    if not hasattr(reg, "_estimator_type"):
        reg._estimator_type = "regressor"


class XgbModel:
    """XGBoost regressor. Defaults are per-instance so different tasks
    can carry different hyperparameters without subclassing.
    """

    name = "xgb"
    has_drift = True

    def __init__(self, **default_params: Any) -> None:
        self.default_params = default_params or {
            "n_estimators": 100,
            "max_depth": 3,
            "eta": 0.1,
        }

    def fit(self, components: dict) -> tuple[dict, dict]:
        """Train and return ``(artifacts, metrics)``. ``artifacts``
        carries the fitted regressor, both scalers, and the sliding-
        window length.
        """
        params = {**self.default_params, **components.get("model_parameters", {})}
        components_with_params = {**components, "model_parameters": params}

        trainer = ModelTrainer(components_with_params)
        metrics, regressor = trainer.train_xgb()
        metrics_jsonable = coerce_metrics(metrics)

        artifacts = {
            "regressor": regressor,
            "scaler_obj": components_with_params["scaler_obj"],  # y-scaler
            "scaler_X": components_with_params["scaler_X"],
            "look_back": components_with_params["look_back"],
            "num_variables": components_with_params.get("num_variables", 1),
            "model_metrics": metrics_jsonable,
            "test_sample_size": len(components_with_params["X_test"]),
        }
        return artifacts, metrics_jsonable

    def save_artifacts(self, artifacts: dict, dest: Path) -> dict[str, str]:
        """Persist the artifacts and return the ``role -> filename`` map.

        The model is saved as ``xgb.ubj`` via xgboost's native UBJ
        format. Both scalers persist as JSON + NPZ sidecars.
        """
        regressor = artifacts["regressor"]
        _ensure_estimator_type(regressor)
        regressor.save_model(str(dest / "xgb.ubj"))

        save_sklearn_scaler(dest, "scaler_x", artifacts["scaler_X"])
        save_sklearn_scaler(dest, "scaler_y", artifacts["scaler_obj"])
        save_json(
            dest,
            "xgb_meta.json",
            {
                "look_back": int(artifacts["look_back"]),
                "num_variables": int(artifacts.get("num_variables", 1)),
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
        """Restore the dict shape ``fit`` emits, plus ``input_spec``
        if it was persisted.
        """
        regressor = XGBRegressor()
        _ensure_estimator_type(regressor)
        regressor.load_model(str(src / "xgb.ubj"))

        meta = load_json(src, "xgb_meta.json")
        loaded: dict[str, Any] = {
            "regressor": regressor,
            "scaler_obj": load_sklearn_scaler(src, "scaler_y"),
            "scaler_X": load_sklearn_scaler(src, "scaler_x"),
            "look_back": int(meta["look_back"]),
            "num_variables": int(meta.get("num_variables", 1)),
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
        """Recursive multi-horizon: predict target t+1, slide it into
        the window, predict t+2, …, ``horizon`` times. Covariates are
        held at their last observed value across the horizon (we don't
        have future ground truth). No native confidence intervals —
        ``lower``/``upper`` stay ``None``.

        The supervised training structure orders columns by ``(lag,
        var)`` — the window returned by ``assemble_predict_window`` has
        the same shape, so a row-major flatten produces the model's
        expected input vector.
        """
        window_2d, num_variables = assemble_predict_window(artifacts, input_series)

        scaler_x = artifacts["scaler_X"]
        scaler_y = artifacts["scaler_obj"]
        cols = getattr(scaler_x, "feature_names_in_", None)
        regressor = artifacts["regressor"]

        # Hold covariates frozen across the recursive horizon — the
        # model has no notion of future covariate values, and a generic
        # forecasting lib can't assume the caller does either.
        last_covariates = window_2d[-1, 1:].copy() if num_variables > 1 else None

        forecasts: list[float] = []
        for _ in range(horizon):
            x_flat = window_2d.reshape(1, -1)
            if cols is not None:
                x_scaled = scaler_x.transform(pd.DataFrame(x_flat, columns=cols))
            else:
                x_scaled = scaler_x.transform(x_flat)
            y_scaled = regressor.predict(x_scaled)
            y_raw = float(scaler_y.inverse_transform(np.asarray(y_scaled).reshape(-1, 1))[0, 0])
            forecasts.append(y_raw)
            # Slide: drop oldest row, append [ŷ, *frozen_covariates].
            if last_covariates is None:
                new_row = np.array([y_raw])
            else:
                new_row = np.concatenate(([y_raw], last_covariates))
            window_2d = np.vstack([window_2d[1:], new_row])

        return [ForecastPoint(value=round(v, 4)) for v in forecasts]


def make_xgb_prepare(
    look_back: int = 6,
    feature_names: list[str] | None = None,
) -> Callable[[pd.DataFrame], dict]:
    """Build a ``prepare`` callable that produces XGB-shaped components
    from a raw DataFrame.

    ``feature_names`` is the canonical feature order — target first,
    covariates after. Columns are looked up in the DataFrame by name,
    not by position, so a CSV with reordered columns (or an upstream
    loader that joins in a different order) still trains the model on
    the right target. Missing names raise; extras are ignored. Required;
    builders always supply it.

    Pipeline: select named columns → ``ts_supervised_structure``
    (lagged features + target) → split 80/20 → fit separate
    ``StandardScaler``s for X and y → return components dict expected
    by ``ModelTrainer.train_xgb``.
    """
    if not feature_names:
        raise ValueError("make_xgb_prepare requires a non-empty feature_names list")
    feature_names = list(feature_names)

    def prepare(df: pd.DataFrame) -> dict:
        missing = [n for n in feature_names if n not in df.columns]
        if missing:
            raise ValueError(
                f"make_xgb_prepare expected columns {list(feature_names)!r}, "
                f"missing {missing!r}; DataFrame has {list(df.columns)!r}"
            )
        cols = list(feature_names)
        num_variables = len(cols)
        data = df[cols].astype(float).reset_index(drop=True)

        split = int(len(data) * 0.8)
        train_df, test_df = data.iloc[:split], data.iloc[split:]

        # ``supervised_window`` is the numpy core shared with LSTM.
        # XGB needs the ``var{j}(t-{i})`` column names so the fitted
        # scaler's ``feature_names_in_`` round-trips at predict time —
        # wrap the ndarray back into a DataFrame with the canonical names.
        col_names = supervised_window_columns(n_in=look_back, n_out=1, n_vars=num_variables)
        sup_train = pd.DataFrame(
            supervised_window(train_df.to_numpy(), n_in=look_back, n_out=1),
            columns=col_names,
        )
        sup_test = pd.DataFrame(
            supervised_window(test_df.to_numpy(), n_in=look_back, n_out=1),
            columns=col_names,
        )

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
            "num_variables": num_variables,
        }

    return prepare
