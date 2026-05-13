"""Helpers shared by every per-kind builder."""

from __future__ import annotations

from typing import TYPE_CHECKING

from intelligence.tasks.contracts import InputSpec

if TYPE_CHECKING:
    from intelligence.config.settings import FeatureSpec


def build_input_spec(
    *,
    features: list[FeatureSpec],
    steps_back: int,
    max_horizon: int | None = None,
) -> InputSpec:
    """Build an ``InputSpec`` from a task's ``FeatureSpec`` list.

    Length 1 is univariate; longer lists are multivariate with the
    first feature as the target. ``value_range`` from each
    ``FeatureSpec`` (when set) maps into ``InputSpec.value_range`` so
    the per-feature clamp is enforced at the contract boundary.
    ``max_horizon`` bounds the request horizon; ``None`` means
    unbounded (ARIMA, XGB). LSTM clamps to its trained ``output_size``.
    """
    return InputSpec(
        n_features=len(features),
        feature_names=[f.name for f in features],
        steps_back=steps_back,
        max_horizon=max_horizon,
        value_range={
            f.name: tuple(f.value_range)
            for f in features
            if f.value_range is not None
        },
    )
