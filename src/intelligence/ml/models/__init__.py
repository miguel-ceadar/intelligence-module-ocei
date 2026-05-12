"""Models — per-algorithm ``train`` + ``predict`` implementations.

Concrete models live alongside (``arima``, ``xgb``, ``lstm``).
Importing this package does NOT pull any concrete model; importers ask
for what they need directly (``from intelligence.ml.models.arima import
ArimaModel``).
"""

from intelligence.ml.models.base import Model

__all__ = ["Model"]
