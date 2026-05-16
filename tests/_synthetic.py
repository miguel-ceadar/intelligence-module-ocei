"""Shared synthetic-series factories for unit tests.

Three shapes cover every model test:

- ``cpu_walk`` — cumulative random walk clipped to [0.05, 0.95]. Used
  by ARIMA/XGB/LSTM happy paths where order/structure matters.
- ``stationary_cpu`` — i.i.d. Gaussian clipped to [0.0, 1.0]. Used by
  drift tests where the reference distribution must be steady.
- ``correlated_multivariate`` — cpu (target) drives a one-step-lagged
  mem column; load is independent noise. Used by multivariate tests.

Kept as plain functions (not pytest fixtures) so call sites can pass
their own ``n`` / ``seed`` / clip params without going through
parametrize indirection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cpu_walk(n: int = 200, seed: int = 11) -> pd.DataFrame:
    """Random-walk CPU series with a ``timestamp`` index column."""
    rng = np.random.default_rng(seed)
    walk = np.cumsum(rng.standard_normal(n) * 0.02) + 0.5
    return pd.DataFrame({"timestamp": np.arange(n), "cpu": walk.clip(0.05, 0.95)})


def stationary_cpu(n: int, mean: float = 0.5, std: float = 0.05, seed: int = 1) -> pd.DataFrame:
    """Stationary Gaussian CPU series — no ``timestamp`` column (drift
    tests don't use it)."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"cpu": rng.normal(mean, std, n).clip(0.0, 1.0)})


def correlated_multivariate(n: int = 300, seed: int = 17) -> pd.DataFrame:
    """Three-column multivariate: ``cpu`` drives ``mem`` with a one-step
    lag; ``load`` is independent noise. Target is the first column."""
    rng = np.random.default_rng(seed)
    cpu = (np.cumsum(rng.standard_normal(n) * 0.02) + 0.5).clip(0.05, 0.95)
    mem = np.roll(cpu, 1) + rng.standard_normal(n) * 0.01
    load = rng.standard_normal(n) * 0.1 + 0.3
    return pd.DataFrame({"timestamp": np.arange(n), "cpu": cpu, "mem": mem, "load": load})
