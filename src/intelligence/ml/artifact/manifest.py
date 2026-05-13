"""Self-describing manifest + global extension allowlist.

The manifest is the security boundary for artefacts pulled from external
sources (HF Hub today). Each artefact's ``manifest.json`` declares
``kind`` plus a ``files: {role: filename}`` map; the validator enforces
that *every* file in the directory is either bentoml-generated
(``model.yaml``), our own ``manifest.json``, or explicitly declared,
*and* that every filename uses an extension on the global allowlist.

Adding a new model kind requires **no edits to this module** — the kind
writes whatever files it needs and declares them in its manifest. The
security guarantee is purely structural: no executable types, no
pickle, no path traversal, no stowaway files.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ValidationError

SCHEMA_VERSION = 1

# Global allowlist of file extensions ever permitted in an artefact
# directory. No executable / opaque-binary types — no ``.pkl``,
# ``.pickle``, ``.so``, ``.dylib``, ``.py``, ``.sh``, ``.exe``. Adding a
# new model kind that needs a new file format extends this list (and
# only this list — kind-specific declarations live in the manifest the
# kind writes, not here).
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

# Filenames always tolerated alongside whatever the kind declares: our
# own manifest and the small YAML bentoml writes when we ride its
# store/tag layer (see ``store.py``).
ALWAYS_ALLOWED = frozenset({"manifest.json", "model.yaml"})


class ManifestError(ValueError):
    """Raised when an artefact directory fails manifest or filename checks.

    Subclasses ``ValueError`` so the existing API error handler in
    ``service.py`` translates it to HTTP 422 without extra wiring.
    """


class Manifest(BaseModel):
    """The on-disk manifest. Self-describing: ``files`` maps a stable
    role name (``"model"``, ``"scaler_x"``, ``"input_spec"``, …) to a
    bare filename. The kind's loader reads roles, not literal filenames,
    so file naming stays a private detail of each kind.

    ``kind`` is an open string — the manifest layer doesn't enforce a
    closed set, so adding a kind doesn't require a manifest edit. The
    model-loader dispatch (where the kind → builder mapping lives)
    fails with a clear error if a manifest references an unknown kind.
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
    """Parse a manifest dict and enforce schema_version + filename rules.

    Schema_version is exact equality — we don't read older formats.
    Pre-pilot there are no historical artefacts to support, so a
    mismatch means a malformed or hostile manifest.
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
    """Cross-check the directory contents against the manifest.

    Every file on disk must be in :data:`ALWAYS_ALLOWED` or declared by
    the manifest; every declared file must actually exist; no
    subdirectories. Stowaways and missing artefacts both raise.
    """
    declared = set(manifest.files.values()) | ALWAYS_ALLOWED

    for entry in path.iterdir():
        if entry.is_dir():
            raise ManifestError(
                f"artefact directory must be flat; subdirectory not allowed: {entry.name!r}"
            )
        if entry.name not in declared:
            raise ManifestError(
                f"file {entry.name!r} present in artefact but not declared in manifest"
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
    """Write a fresh ``manifest.json`` into the artefact directory.

    Each model's ``save_artifacts`` writes its files and then calls this
    to commit the declaration. Path-traversing or unknown-extension
    filenames are refused here too, so a buggy ``save_artifacts`` can't
    poison its own artefact.
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
