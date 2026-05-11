"""Model adapters — see ``base.ModelAdapter``.

Concrete adapters live alongside (``arima``, ``xgb``, ``lstm``).
Importing this package does NOT pull any model implementation; importers
ask for what they need directly (``from intelligence.models.arima
import ArimaAdapter``).
"""

from intelligence.models.base import ModelAdapter

__all__ = ["ModelAdapter"]
