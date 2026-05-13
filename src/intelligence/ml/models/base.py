"""``Model`` protocol — fit + persist + load + predict for one ML algorithm.

A ``Model`` is *stateless* with respect to the data domain. Given
prepared training components, ``fit`` returns an in-memory ``artifacts``
dict that ``save_artifacts`` writes to a flat directory using
framework-native formats (and never pickle). ``load_artifacts`` is the
inverse — restore the same dict shape from disk. ``predict`` then
consumes that dict directly.

The same dict shape threads through all four methods, so adding a new
kind (``prophet``, ``transformer``, …) is purely a matter of
implementing the protocol against a fresh artefact directory layout —
no manifest edit, no per-kind storage glue.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Model(Protocol):
    """Train one model family, persist it as a manifest-described
    artefact directory, predict from a loaded artefact dict."""

    name: str  # short tag — 'arima', 'xgb', 'lstm'
    has_drift: bool  # whether this model ships drift detection

    def fit(self, components: dict) -> tuple[dict, dict]:
        """Train and return ``(artifacts, metrics)``.

        ``artifacts`` carries the runtime state predict consumes —
        fitted model + scalers + window metadata. ``BaseTask`` injects
        ``input_spec`` into this dict before calling
        :meth:`save_artifacts` so the contract travels with the model.
        """
        ...

    def save_artifacts(self, artifacts: dict, dest: Path) -> dict[str, str]:
        """Persist ``artifacts`` into ``dest`` and return the
        ``role -> filename`` map for the manifest. Implementations write
        framework-native model files + typed sidecars; no pickle.
        """
        ...

    def load_artifacts(self, src: Path) -> dict:
        """Inverse of :meth:`save_artifacts` — return the same dict
        shape :meth:`fit` emits, plus ``input_spec`` if it was persisted.
        """
        ...

    def predict(
        self,
        artifacts: dict,
        input_series: dict[str, list[float]],
        horizon: int = 1,
    ) -> Any:
        """Predict from a loaded artefacts dict.

        ``horizon`` is the number of steps ahead to forecast. Forecasting
        models return ``list[ForecastPoint]`` of length ``horizon``;
        ``ForecastPoint.lower`` / ``upper`` carry the 95 % CI when the
        model exposes one (ARIMA does; recursive XGB and direct LSTM
        leave them ``None``).
        """
        ...
