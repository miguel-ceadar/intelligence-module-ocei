"""Column-name conventions shared across loaders and per-kind prepares.

Centralised here so loaders, ``DriftModel``, and the static source agree
on which DataFrame columns count as timestamps. Changing the set is a
one-file edit instead of a five-file refactor.
"""

from __future__ import annotations

# Case-insensitive — callers compare via ``c.lower() in TIMESTAMP_COLS``.
TIMESTAMP_COLS: frozenset[str] = frozenset({"time", "timestamp", "date"})
