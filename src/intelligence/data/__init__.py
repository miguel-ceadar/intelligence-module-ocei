"""Bundled sample data shipped with the package.

``samples/`` holds small CSVs used by the StaticSource demo path and by
the test suite. They are intentionally tiny so the wheel stays small.
"""

from pathlib import Path

SAMPLES_DIR: Path = Path(__file__).resolve().parent / "samples"
