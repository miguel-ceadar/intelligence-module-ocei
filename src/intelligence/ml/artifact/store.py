"""Artefact store — thin wrapper over bentoml's tag/version layer.

We ride bentoml's local store for tag/version semantics (``:latest``,
immutable versions, ``creation_time``) but bypass ``picklable_model``
entirely: ``save_artifact`` opens a fresh model directory via
``bentoml.models.create``, hands it to a caller-supplied ``write_fn``
that populates files, then commits a manifest and validates the
directory before the bento closes.

The bentoml-specific types (``Tag``, ``NotFound``, ``ModelContext``)
stay inside this module — call-sites work in terms of
:class:`SavedArtifact` + string tags so swapping the underlying store
later is a one-file change.
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
    """Read-only handle to a committed artefact in the local store."""

    tag: str  # "name:version" string — pass back to ``get_artifact_by_tag``
    name: str
    version: str
    path: Path  # flat directory containing manifest.json + native files
    manifest: Manifest
    created_at: str  # ISO 8601 timestamp from bentoml.info.creation_time


def _model_context() -> ModelContext:
    # bentoml requires a context even though we don't ship a framework
    # backend; ``intelligence`` here is a marker so an operator inspecting
    # the bentoml CLI sees our models distinctly.
    return ModelContext(framework_name="intelligence", framework_versions={})


def save_artifact(
    name: str,
    kind: str,
    write_fn: Callable[[Path], dict[str, str]],
) -> SavedArtifact:
    """Open a fresh artefact directory and commit it to the local store.

    ``write_fn(path)`` receives a writable directory and must return the
    ``role -> filename`` map it populated. ``save_artifact`` then writes
    ``manifest.json`` and validates the directory against the manifest
    — stowaway files, declared-but-missing files, or unsafe extensions
    raise ``ManifestError`` before the bento commits.
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
        raise RuntimeError(f"saved artefact {tag} not retrievable")
    return fetched


def get_artifact_by_tag(tag: str) -> SavedArtifact | None:
    """Resolve a tag (``name:version`` or ``name:latest``) to a
    :class:`SavedArtifact`. Returns ``None`` if no artefact is found —
    callers translate that to HTTP 404 / 503 themselves.
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
    """Return every artefact matching ``name``, newest first.

    Models with an unreadable manifest are silently skipped — they were
    written by something other than this codebase (e.g. a stale
    picklable_model bento from before the migration) and we have no way
    to interpret them.
    """
    artefacts: list[SavedArtifact] = []
    for m in bentoml.models.list():
        if m.tag.name != name:
            continue
        path = Path(m.path)
        try:
            manifest = read_manifest(path)
        except Exception:
            logger.warning("skipping artefact %s with unreadable manifest", m.tag)
            continue
        artefacts.append(
            SavedArtifact(
                tag=str(m.tag),
                name=m.tag.name,
                version=m.tag.version,
                path=path,
                manifest=manifest,
                created_at=m.info.creation_time.isoformat(),
            )
        )
    artefacts.sort(key=lambda a: a.created_at, reverse=True)
    return artefacts


def import_artifact(name: str, source_dir: Path) -> SavedArtifact:
    """Copy a vetted artefact directory into the local store under
    ``name``. Used by the HF pull path: the caller is expected to have
    validated the manifest already, but this function re-validates so a
    bug in the pull path can't smuggle a stowaway in.

    ``model.yaml`` from the source is intentionally skipped — bentoml
    writes its own when the new bento commits, and the source's copy
    would be redundant and might disagree with the new tag.
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
