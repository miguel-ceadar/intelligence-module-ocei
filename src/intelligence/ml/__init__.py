"""ML — models and trainers.

Two sub-packages:
  - ``models/`` — per-algorithm ``Model`` implementations
    (the lifecycle glue between a ``Task`` and a saved Bento).
  - ``trainers/`` — framework-specific training loops + helpers
    (``ModelTrainer``, LSTM model defs, metrics, MLflow housekeeping).

Concrete models import their trainer from ``intelligence.ml.trainers``.
Importing ``intelligence.ml`` itself is free — neither sub-package is
pulled until you import it explicitly.
"""
