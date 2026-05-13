"""Helpers shared by every per-kind builder."""

from __future__ import annotations

from intelligence.tasks.contracts import InputSpec


def build_input_spec(
    *,
    feature: str,
    steps_back: int,
    value_range: tuple[float, float] | None = None,
    max_horizon: int | None = None,
) -> InputSpec:
    """Build a univariate ``InputSpec`` for a task.

    Every built-in kind is single-feature: one PromQL series in, one
    forecast value out. ``value_range`` is enforced — a predict
    request outside the range returns ``422`` before the model runs;
    pass ``None`` to disable. ``max_horizon`` bounds the request
    horizon; ``None`` means unbounded (used by ARIMA and XGB). LSTM
    clamps to its trained ``output_size``.
    """
    return InputSpec(
        n_features=1,
        feature_names=[feature],
        steps_back=steps_back,
        max_horizon=max_horizon,
        value_range={feature: tuple(value_range)} if value_range is not None else {},
    )
