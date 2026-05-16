"""CeADAR fork of the ICOS intelligence-module, adapted for the O-CEI continuum.

Forecasting + drift detection against generic Prometheus telemetry,
served as a FastAPI app via BentoML. See ``README.md`` and ``docs/``.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("icos-intelligence-ocei")
except PackageNotFoundError:  # editable install in a fresh checkout
    __version__ = "0.0.0+unknown"
