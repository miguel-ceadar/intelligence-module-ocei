"""Manifest-based artefact persistence — the pickle-free replacement for
``bentoml.picklable_model``.

Each saved artefact is a flat directory containing a self-describing
``manifest.json`` (kind + ``files: {role: filename}`` map), one or more
framework-native model files, typed sidecars, and the bentoml-generated
``model.yaml``. Adding a new model kind requires no edits to this
package — the kind writes whatever files it needs and declares them in
the manifest it commits.

Public surface (use these; avoid reaching into submodules):

    from intelligence.ml.artifact import (
        SavedArtifact,
        save_artifact,
        get_artifact_by_tag,
        list_artifacts_by_name,
        import_artifact,
        Manifest,
        ManifestError,
    )

Per-kind model classes additionally import from
``intelligence.ml.artifact.sidecars`` for the typed save/load helpers.
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
