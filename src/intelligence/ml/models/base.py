"""``Model`` protocol: fit, save, load, predict for one ML algorithm.

``fit`` returns an in-memory ``artifacts`` dict that ``save_artifacts``
writes to a flat directory using framework-native formats (no pickle).
``load_artifacts`` restores the same dict shape; ``predict`` consumes
it directly. Adding a new kind is purely a matter of implementing
the protocol against a fresh artifact layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Model(Protocol):
    """Train one model family, persist it as a manifest-described
    artifact directory, predict from a loaded artifact dict."""

    name: str  # 'arima', 'xgb', 'lstm'
    has_drift: bool

    def fit(self, components: dict) -> tuple[dict, dict]:
        """Train and return ``(artifacts, metrics)``. ``BaseTask``
        injects ``input_spec`` into ``artifacts`` before save.
        """
        ...

    def save_artifacts(self, artifacts: dict, dest: Path) -> dict[str, str]:
        """Persist ``artifacts`` into ``dest`` and return the
        ``role -> filename`` map for the manifest.
        """
        ...

    def load_artifacts(self, src: Path) -> dict:
        """Restore the dict shape ``fit`` emits, plus ``input_spec``
        if it was persisted.
        """
        ...

    def predict(
        self,
        artifacts: dict,
        input_series: dict[str, list[float]],
        horizon: int = 1,
    ) -> Any:
        """Forecast ``horizon`` steps. Returns ``list[ForecastPoint]``;
        ``lower``/``upper`` carry the 95 % CI when the model exposes
        one (ARIMA does; XGB and LSTM leave them ``None``).
        """
        ...
