"""Helpers shared between the per-kind ``Model`` implementations.

Predict-time window assembly is duplicated between XGB (recursive
multi-horizon) and LSTM (direct multi-output) — both have to validate
the request, resolve the canonical feature order, and stack the last
``look_back`` observations across every feature into a 2-D window
before doing anything model-specific. ``assemble_predict_window``
captures that preamble.

The sliding-window reshape that turns a ``(T, V)`` history into the
``(samples, (look_back + horizon) * V)`` supervised structure is the
same math in both prepare callables; ``supervised_window`` is the
numpy core. ``supervised_window_columns`` produces the
``var{j}(t-{i})`` names XGB needs on its DataFrame so the saved
``StandardScaler.feature_names_in_`` round-trips at predict time.

``coerce_metrics`` flattens numpy/torch scalars to native Python types
one or two levels deep — covers ARIMA / XGB flat dicts and the nested
per-horizon dict ``train_pytorch`` emits.
"""

from __future__ import annotations

import numpy as np


def assemble_predict_window(
    artifacts: dict,
    input_series: dict[str, list[float]],
) -> tuple[np.ndarray, int]:
    """Validate ``input_series`` and stack it into the
    ``(look_back, num_variables)`` window XGB and LSTM both consume.

    Returns ``(window_2d, num_variables)`` — ``look_back`` is implicit
    in ``window_2d.shape[0]``, and ``num_variables`` is the only piece
    the recursive XGB path still needs after the helper (to freeze
    covariates). Raises ``ValueError`` on empty input,
    fewer-than-expected series, or shorter-than-``look_back`` windows.

    Canonical feature order comes from the saved ``InputSpec`` when
    present (predict-time API validation has already aligned the
    request to it); falls back to ``input_series`` insertion order for
    legacy artifacts that pre-date ``input_spec``.
    """
    if not input_series:
        raise ValueError("input_series is empty")

    look_back = int(artifacts["look_back"])
    num_variables = int(artifacts.get("num_variables", 1))

    spec = artifacts.get("input_spec")
    if spec is not None:
        series_keys = list(spec.feature_names)
    else:
        series_keys = list(input_series.keys())[:num_variables]
    if len(series_keys) < num_variables:
        raise ValueError(f"need {num_variables} input series, got {len(series_keys)}")

    # Length-check each series *before* stacking. ``column_stack`` on
    # mismatched 1-D arrays would raise a shape-mismatch error that
    # doesn't say which feature was short — surface the offender by name.
    short = [(k, len(input_series[k])) for k in series_keys if len(input_series[k]) < look_back]
    if short:
        details = ", ".join(f"{name!r} has {n}" for name, n in short)
        raise ValueError(f"need at least {look_back} observations per series, got: {details}")

    window = np.column_stack(
        [np.asarray(input_series[k], dtype=float)[-look_back:] for k in series_keys]
    )
    return window, num_variables


def supervised_window(data: np.ndarray, n_in: int, n_out: int = 1) -> np.ndarray:
    """Sliding-window reshape. ``data`` is ``(T, V)`` (1-D promoted to
    a single variable). Returns ``(T - n_in - n_out + 1,
    (n_in + n_out) * V)``. Row ``i`` is ``data[i : i + n_in + n_out]``
    flattened row-major, so columns are ordered ``(lag, var)``.
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


def supervised_window_columns(n_in: int, n_out: int, n_vars: int) -> list[str]:
    """Canonical column names for ``supervised_window`` output.
    Matches the order of the flattened windows so a DataFrame built
    from the ndarray + these names is what XGB's scaler fits against.
    """
    names: list[str] = []
    for i in range(n_in, 0, -1):
        names += [f"var{j + 1}(t-{i})" for j in range(n_vars)]
    for i in range(n_out):
        if i == 0:
            names += [f"var{j + 1}(t)" for j in range(n_vars)]
        else:
            names += [f"var{j + 1}(t+{i})" for j in range(n_vars)]
    return names


def coerce_metrics(metrics: dict) -> dict:
    """Coerce numpy / torch scalars to native Python types. Handles
    flat dicts (ARIMA, XGB) and the ``{metric_i: {…}}`` nested shape
    ``train_pytorch`` emits per horizon step.
    """
    out: dict = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            out[k] = {kk: vv.item() if hasattr(vv, "item") else vv for kk, vv in v.items()}
        else:
            out[k] = v.item() if hasattr(v, "item") else v
    return out
