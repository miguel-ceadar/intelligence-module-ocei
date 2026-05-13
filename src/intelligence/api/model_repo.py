"""Hugging Face Hub push/pull for local manifest-described artifacts.

The HF token is read from ``HF_TOKEN`` at call time — never persisted
in config files or class state. A missing token raises
``PermissionError`` so the API translates it to HTTP 401.

Pulled artifacts are validated against the manifest schema and the
per-directory filename allowlist *before any file is opened*. There
is no ``pickle.load`` on the pull path — pulled artifacts are
framework-native files (``.ubj`` / ``.safetensors`` / ``.parquet``)
plus typed JSON sidecars. A pulled artifact that lacks ``input_spec``
is still refused at predict time by ``BaseTask._verify_artifact``
unless ``allow_unverified_models=True``.

Push goes the other way: the local artifact directory is re-validated
against its manifest before upload, so a corrupted local artifact
won't propagate to the remote repo.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from intelligence.ml.artifact import import_artifact
from intelligence.ml.artifact.manifest import (
    read_manifest,
    validate_artifact_directory,
)

logger = logging.getLogger(__name__)


def _require_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise PermissionError("HF_TOKEN not set in environment")
    return token


def push_to_hf(
    model_tag: str,
    repo_id: str,
    commit_message: str | None = None,
) -> str:
    """Upload a local manifest-described artifact directory to ``repo_id``.

    The whole artifact directory (manifest + native model file +
    typed sidecars) is uploaded under
    ``{tag.name}/{tag.version}/``. Before upload we re-validate the
    directory against its manifest — a corrupt local artifact won't
    propagate to the remote repo.

    Returns the local tag (unchanged by push).
    """
    token = _require_token()

    import bentoml

    bento = bentoml.models.get(model_tag)
    folder = Path(bento.path)
    if not folder.exists():
        raise FileNotFoundError(f"local artifact folder missing: {folder}")

    # Re-validate before publishing — refuses to upload a corrupt artifact.
    manifest = read_manifest(folder)
    validate_artifact_directory(folder, manifest)

    api = HfApi(token=token)
    api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type="model",
        path_in_repo=f"{bento.tag.name}/{bento.tag.version}",
        commit_message=commit_message or f"upload {bento.tag}",
    )
    logger.info("pushed %s to %s", bento.tag, repo_id)
    return str(bento.tag)


def pull_from_hf(model_tag: str, repo_id: str) -> str:
    """Download ``{name}/{version}/`` from ``repo_id`` and import into
    the local store under a fresh tag.

    The manifest is parsed and the artifact directory is validated
    against the per-directory filename allowlist *before any file is
    opened*. Manifests with the wrong schema version, hostile filenames
    (traversal, hidden, executable extensions), or stowaway files in
    the directory are refused before ``import_artifact`` runs. There
    is no ``pickle.load`` on this path — by design.
    """
    token = _require_token()

    if ":" not in model_tag:
        raise ValueError(f"model_tag must be 'name:version', got {model_tag!r}")
    name, version = model_tag.split(":", 1)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            allow_patterns=f"{name}/{version}/**",
            local_dir=str(tmp_path),
            token=token,
        )
        artifact_dir = tmp_path / name / version
        if not artifact_dir.exists():
            raise FileNotFoundError(
                f"snapshot for {model_tag} missing in repo (looked in {artifact_dir})"
            )

        # The manifest is the security boundary. Parse + validate
        # before any file is opened for processing.
        manifest = read_manifest(artifact_dir)
        validate_artifact_directory(artifact_dir, manifest)

        imported = import_artifact(name, artifact_dir)
        logger.info("pulled %s → local tag %s", model_tag, imported.tag)
        return imported.tag
