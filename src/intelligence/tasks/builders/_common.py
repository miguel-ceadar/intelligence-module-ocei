"""Shared bits used by every per-kind builder.

Builders are mechanically similar — they all take a typed config block,
construct a ``BaseTask`` (or subclass), and wire it to a loader. The
``InputSpec`` construction is identical in shape across kinds; centralising
it here keeps the per-kind files focused on the kind-specific knobs
(which model class, which prepare, which subclass).
"""

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

    Every shipped kind is single-feature: one PromQL series in, one
    forecast value out. ``value_range`` is descriptive but enforced — a
    predict request outside the range gets ``422`` before the model
    runs. Pass ``None`` to disable the range check.

    ``max_horizon`` bounds the request horizon. ``None`` (default) means
    unbounded — used by ARIMA (refits each call) and XGB (recursive).
    LSTM is direct multi-output and clamps to its trained ``output_size``.
    """
    return InputSpec(
        n_features=1,
        feature_names=[feature],
        steps_back=steps_back,
        max_horizon=max_horizon,
        value_range={feature: tuple(value_range)} if value_range is not None else {},
    )
