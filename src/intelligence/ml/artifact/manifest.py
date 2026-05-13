"""Artifact manifest schema and the file-extension allowlist.

Each artifact's ``manifest.json`` declares ``kind`` plus a
``files: {role: filename}`` map. Validation requires that every file
in the directory is either ``manifest.json``, the bentoml-generated
``model.yaml``, or declared in the manifest, and that every filename
uses an extension on the allowlist. The structural guarantee is: no
executable types, no pickle, no path traversal, no stray files.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ValidationError

SCHEMA_VERSION = 1

# File extensions permitted in an artifact directory. No executable
# or opaque-binary types. A new model kind that needs another format
# extends this list.
ALLOWED_EXTENSIONS = frozenset(
    {
        ".json",  # manifest, hyperparams, scaler metadata, drift config
        ".npz",  # numerical numpy arrays (always loaded with allow_pickle=False)
        ".safetensors",  # neural-net weights
        ".ubj",  # native xgboost universal binary JSON
        ".parquet",  # tabular reference data (drift, future tabular models)
        ".yaml",  # bentoml's own model.yaml
    }
)

# Filenames always tolerated alongside whatever the manifest declares:
# the manifest itself and the small YAML that bentoml writes.
ALWAYS_ALLOWED = frozenset({"manifest.json", "model.yaml"})


class ManifestError(ValueError):
    """Raised when an artifact directory fails manifest or filename
    checks. Subclassing ``ValueError`` makes the API surface it as 422.
    """


class Manifest(BaseModel):
    """The on-disk manifest. ``files`` maps a stable role name
    (``"model"``, ``"scaler_x"``, ``"input_spec"``, …) to a bare
    filename; loaders read by role so each kind owns its file naming.
    ``kind`` is an open string — adding a kind doesn't require a
    manifest edit; the model-loader dispatch is where unknown kinds
    fail.
    """

    schema_version: int
    kind: str
    created_at: str  # ISO 8601 UTC
    files: dict[str, str]


def _check_filename(name: str) -> None:
    """Refuse path traversal, hidden, and unsafe-extension filenames.

    Called both at manifest-parse time (catches a hostile manifest) and
    at manifest-write time (catches a buggy ``save_artifacts``).
    """
    if not name:
        raise ManifestError("empty filename in manifest")
    if "/" in name or "\\" in name:
        raise ManifestError(f"filename {name!r} contains a path separator")
    if name.startswith("."):
        raise ManifestError(f"filename {name!r} is hidden")
    if ".." in name:
        raise ManifestError(f"filename {name!r} contains traversal")
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ManifestError(
            f"filename {name!r} has disallowed extension {suffix!r}; "
            f"allowed: {sorted(ALLOWED_EXTENSIONS)}"
        )


def validate_manifest_dict(data: dict) -> Manifest:
    """Parse a manifest dict and enforce schema_version + filename
    rules. ``schema_version`` is checked for exact equality; there is
    no compatibility shim for older versions.
    """
    try:
        manifest = Manifest.model_validate(data)
    except ValidationError as e:
        raise ManifestError(f"manifest schema invalid: {e}") from e
    if manifest.schema_version != SCHEMA_VERSION:
        raise ManifestError(
            f"manifest schema_version {manifest.schema_version} != supported {SCHEMA_VERSION}"
        )
    for fname in manifest.files.values():
        _check_filename(fname)
    return manifest


def validate_artifact_directory(path: Path, manifest: Manifest) -> None:
    """Cross-check directory contents against the manifest.

    Every file on disk must be in ``ALWAYS_ALLOWED`` or declared by
    the manifest; every declared file must exist; no subdirectories.
    """
    declared = set(manifest.files.values()) | ALWAYS_ALLOWED

    for entry in path.iterdir():
        if entry.is_dir():
            raise ManifestError(
                f"artifact directory must be flat; subdirectory not allowed: {entry.name!r}"
            )
        if entry.name not in declared:
            raise ManifestError(
                f"file {entry.name!r} present in artifact but not declared in manifest"
            )

    for role, fname in manifest.files.items():
        if not (path / fname).exists():
            raise ManifestError(f"manifest declares role {role!r} → {fname!r}, but file is missing")


def read_manifest(path: Path) -> Manifest:
    """Read and validate ``path/manifest.json``. Always call this before
    opening any other file in the directory.
    """
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        raise ManifestError(f"manifest.json missing in {path}")
    try:
        data = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as e:
        raise ManifestError(f"manifest.json malformed: {e}") from e
    return validate_manifest_dict(data)


def write_manifest(path: Path, kind: str, files: dict[str, str]) -> Manifest:
    """Write a fresh ``manifest.json`` into the artifact directory.

    Each model's ``save_artifacts`` writes its files and then calls
    this to commit the declaration. Bad filenames are refused here
    too, so a buggy ``save_artifacts`` can't poison its own artifact.
    """
    for fname in files.values():
        _check_filename(fname)

    manifest = Manifest(
        schema_version=SCHEMA_VERSION,
        kind=kind,
        created_at=datetime.now(UTC).isoformat(),
        files=files,
    )
    (path / "manifest.json").write_text(manifest.model_dump_json())
    return manifest
