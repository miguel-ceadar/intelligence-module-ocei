"""Manifest-based artifact persistence.

Each saved artifact is a flat directory containing a ``manifest.json``
(kind + ``files: {role: filename}`` map), one or more framework-native
model files, typed sidecars, and the bentoml-generated ``model.yaml``.
A new model kind writes whatever files it needs and declares them in
the manifest it commits — no edits to this package required.

Per-kind model classes import helpers from
``intelligence.ml.artifact.sidecars`` for typed save/load.
"""

from __future__ import annotations

from intelligence.ml.artifact.manifest import (
    Manifest,
    ManifestError,
)
from intelligence.ml.artifact.store import (
    SavedArtifact,
    get_artifact_by_tag,
    import_artifact,
    list_artifacts_by_name,
    save_artifact,
)

__all__ = [
    "Manifest",
    "ManifestError",
    "SavedArtifact",
    "get_artifact_by_tag",
    "import_artifact",
    "list_artifacts_by_name",
    "save_artifact",
]
