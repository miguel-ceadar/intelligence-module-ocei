"""Artifact store — a thin wrapper over bentoml's tag/version layer.

We use bentoml's local store for tag/version semantics (``:latest``,
immutable versions, ``creation_time``) but skip its ``picklable_model``
machinery: ``save_artifact`` opens a fresh model directory, hands it
to a caller-supplied ``write_fn`` that populates files, writes a
manifest, and validates the directory before committing.

bentoml-specific types stay inside this module; call-sites work in
terms of ``SavedArtifact`` + string tags.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import bentoml
from bentoml.exceptions import NotFound
from bentoml.models import ModelContext

from intelligence.ml.artifact.manifest import (
    Manifest,
    read_manifest,
    validate_artifact_directory,
    write_manifest,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SavedArtifact:
    """Read-only handle to a committed artifact in the local store."""

    tag: str  # "name:version" string — pass back to ``get_artifact_by_tag``
    name: str
    version: str
    path: Path  # flat directory containing manifest.json + native files
    manifest: Manifest
    created_at: str  # ISO 8601 timestamp from bentoml.info.creation_time


def _model_context() -> ModelContext:
    # bentoml requires a context; the framework name is a marker so
    # the bentoml CLI shows these models distinctly.
    return ModelContext(framework_name="intelligence", framework_versions={})


def save_artifact(
    name: str,
    kind: str,
    write_fn: Callable[[Path], dict[str, str]],
) -> SavedArtifact:
    """Open a fresh artifact directory and commit it to the local store.

    ``write_fn(path)`` receives a writable directory and must return
    the ``role -> filename`` map it populated. The manifest is then
    written and validated; any mismatch raises ``ManifestError`` before
    the commit succeeds.
    """
    with bentoml.models.create(
        name=name,
        module="intelligence.ml.artifact",
        api_version="v1",
        signatures={},
        context=_model_context(),
        metadata={"kind": kind},
    ) as model:
        path = Path(model.path)
        files = write_fn(path)
        manifest = write_manifest(path, kind, files)
        validate_artifact_directory(path, manifest)
        tag = str(model.tag)

    fetched = get_artifact_by_tag(tag)
    if fetched is None:  # pragma: no cover — defence in depth
        raise RuntimeError(f"saved artifact {tag} not retrievable")
    return fetched


def get_artifact_by_tag(tag: str) -> SavedArtifact | None:
    """Resolve a tag (``name:version`` or ``name:latest``) to a
    ``SavedArtifact``, or ``None`` if not found. Callers translate
    ``None`` to HTTP 404 / 503 themselves.
    """
    try:
        m = bentoml.models.get(tag)
    except NotFound:
        return None
    path = Path(m.path)
    manifest = read_manifest(path)
    return SavedArtifact(
        tag=str(m.tag),
        name=m.tag.name,
        version=m.tag.version,
        path=path,
        manifest=manifest,
        created_at=m.info.creation_time.isoformat(),
    )


def list_artifacts_by_name(name: str) -> list[SavedArtifact]:
    """Return every artifact matching ``name``, newest first. Models
    with an unreadable manifest are skipped — they were not written
    by this codebase.
    """
    artifacts: list[SavedArtifact] = []
    for m in bentoml.models.list():
        if m.tag.name != name:
            continue
        path = Path(m.path)
        try:
            manifest = read_manifest(path)
        except Exception:
            logger.warning("skipping artifact %s with unreadable manifest", m.tag)
            continue
        artifacts.append(
            SavedArtifact(
                tag=str(m.tag),
                name=m.tag.name,
                version=m.tag.version,
                path=path,
                manifest=manifest,
                created_at=m.info.creation_time.isoformat(),
            )
        )
    artifacts.sort(key=lambda a: a.created_at, reverse=True)
    return artifacts


def import_artifact(name: str, source_dir: Path) -> SavedArtifact:
    """Copy a vetted artifact directory into the local store under
    ``name``. Re-validates the manifest defensively, even though the
    caller is expected to have done so. ``model.yaml`` is skipped:
    bentoml writes its own when the new bento commits.
    """
    manifest = read_manifest(source_dir)
    validate_artifact_directory(source_dir, manifest)

    def _copy(dest: Path) -> dict[str, str]:
        for entry in source_dir.iterdir():
            if entry.name in {"model.yaml", "manifest.json"}:
                # ``model.yaml`` is bentoml's; ``manifest.json`` is rewritten
                # by ``save_artifact`` (fresh timestamp on the new tag).
                continue
            shutil.copy2(entry, dest / entry.name)
        return dict(manifest.files)

    return save_artifact(name, manifest.kind, _copy)
