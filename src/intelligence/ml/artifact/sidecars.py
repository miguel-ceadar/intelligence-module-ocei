"""Typed save/load helpers used by each model's ``save_artifacts``.

sklearn scalers are restored via a strict class/module allowlist (no
arbitrary import). Fitted state is split into a JSON metadata file
(constructor params + non-array attributes) and an NPZ file (numpy
arrays). The NPZ is always loaded with ``allow_pickle=False``, so a
tampered archive can't smuggle pickled state. ``InputSpec`` round-trips
through pydantic JSON.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from intelligence.tasks.contracts import InputSpec

ALLOWED_SCALER_CLASSES = frozenset({"StandardScaler", "MinMaxScaler"})
ALLOWED_SCALER_MODULE_PREFIX = "sklearn.preprocessing"


def save_input_spec(dir_path: Path, spec: InputSpec, filename: str = "input_spec.json") -> None:
    (dir_path / filename).write_text(spec.model_dump_json())


def load_input_spec(dir_path: Path, filename: str = "input_spec.json") -> InputSpec:
    return InputSpec.model_validate_json((dir_path / filename).read_text())


def save_json(dir_path: Path, filename: str, data: dict) -> None:
    (dir_path / filename).write_text(json.dumps(_to_jsonable(data)))


def load_json(dir_path: Path, filename: str) -> dict:
    return json.loads((dir_path / filename).read_text())


def save_sklearn_scaler(dir_path: Path, role: str, scaler: Any) -> None:
    """Persist a fitted sklearn scaler as ``{role}.json`` (params +
    non-numeric attrs) plus ``{role}.npz`` (numeric numpy arrays).

    Object-dtype arrays (e.g. ``feature_names_in_``) go into the JSON
    rather than the NPZ so the load path can stay ``allow_pickle=False``.
    """
    cls = type(scaler)
    state = (
        dict(scaler.__getstate__()) if hasattr(scaler, "__getstate__") else dict(scaler.__dict__)
    )

    arrays: dict[str, np.ndarray] = {}
    attrs: dict[str, Any] = {}
    for key, value in state.items():
        if key.startswith("_"):  # skip private (e.g. _sklearn_version)
            continue
        if isinstance(value, np.ndarray):
            if _is_safe_numeric_dtype(value.dtype):
                arrays[key] = value
            else:
                attrs[key] = {
                    "__ndarray__": True,
                    "values": value.tolist(),
                    "dtype": str(value.dtype),
                }
        else:
            attrs[key] = _to_jsonable(value)

    meta = {
        "class": cls.__name__,
        "module": cls.__module__,
        "params": _to_jsonable(scaler.get_params()),
        "attrs": attrs,
    }
    np.savez(dir_path / f"{role}.npz", **arrays)
    (dir_path / f"{role}.json").write_text(json.dumps(meta))


def load_sklearn_scaler(dir_path: Path, role: str) -> Any:
    """Reconstruct a fitted sklearn scaler from its sidecar pair.

    Refuses any class not in ``ALLOWED_SCALER_CLASSES`` or any
    ``module`` outside ``sklearn.preprocessing`` — the JSON could
    otherwise point at an arbitrary import path.
    """
    meta = json.loads((dir_path / f"{role}.json").read_text())

    cls_name = meta.get("class")
    module_name = meta.get("module", "")
    if cls_name not in ALLOWED_SCALER_CLASSES:
        raise ValueError(f"unsupported scaler class: {cls_name!r}")
    if not module_name.startswith(ALLOWED_SCALER_MODULE_PREFIX):
        raise ValueError(
            f"unsafe scaler module: {module_name!r} "
            f"(must start with {ALLOWED_SCALER_MODULE_PREFIX!r})"
        )

    module = importlib.import_module(module_name)
    cls = getattr(module, cls_name)

    scaler = cls(**meta.get("params", {}))
    for key, value in meta.get("attrs", {}).items():
        if isinstance(value, dict) and value.get("__ndarray__"):
            arr = np.array(value["values"], dtype=value.get("dtype"))
            setattr(scaler, key, arr)
        else:
            setattr(scaler, key, value)

    # allow_pickle=False — refuse any object/pickle payload smuggled in.
    with np.load(dir_path / f"{role}.npz", allow_pickle=False) as npz:
        for key in npz.files:
            setattr(scaler, key, npz[key])

    return scaler


def _is_safe_numeric_dtype(dtype: np.dtype) -> bool:
    """True if a dtype round-trips through NPZ without object arrays."""
    return np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.bool_)


def _to_jsonable(value: Any) -> Any:
    """Coerce numpy scalars and tuples to JSON-native types. Anything
    unrecognised is returned untouched so ``json.dump`` raises rather
    than silently dropping data.
    """
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value
