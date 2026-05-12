"""Hugging Face Hub push/pull for local Bento models.

The HF token is read from ``HF_TOKEN`` at call time — never persisted
in config files or class state. A missing token raises
``PermissionError`` so the API translates it to HTTP 401.

Pulled artifacts are re-saved via ``bentoml.picklable_model`` regardless
of their original framework. The new tree saves models through
``picklable_model`` consistently (see ``ml/models/*``), so this keeps
the local store homogeneous and lets ``BaseTask._load_bento`` find them
without framework-specific dispatch.

Operationally, a pulled Bento that lacks ``input_spec`` in its
``custom_objects`` is treated as **unverified** by ``BaseTask._verify_bento``
(see plan §3.5) — the predict path refuses it unless
``allow_unverified_models=True``.
"""

from __future__ import annotations

import logging
import os
import pickle
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

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
    """Upload the local Bento's folder to ``repo_id`` on Hugging Face.

    The whole Bento directory (model file, ``custom_objects.pkl``,
    metadata) is uploaded under the model's tag-named subdirectory.

    Returns the local Bento tag (unchanged by push).
    """
    token = _require_token()

    import bentoml
    bento = bentoml.models.get(model_tag)
    folder = Path(bento.path)
    if not folder.exists():
        raise FileNotFoundError(f"local Bento folder missing: {folder}")

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
    """Download ``{name}/{version}/`` from ``repo_id`` and re-save locally
    via ``bentoml.picklable_model``. Returns the new local tag.

    The model itself is loaded from ``saved_model.pkl``; any
    ``custom_objects.pkl`` alongside is unpickled and re-attached.
    Frameworks other than the pickle-able family are out of scope —
    they were dropped along with the NKUA tasks in phase 2 §3.1.
    """
    token = _require_token()

    if ":" not in model_tag:
        raise ValueError(f"model_tag must be 'name:version', got {model_tag!r}")
    name, version = model_tag.split(":", 1)

    import bentoml

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
        saved = artifact_dir / "saved_model.pkl"
        if not saved.exists():
            raise FileNotFoundError(
                f"snapshot for {model_tag} missing saved_model.pkl "
                f"(looked in {artifact_dir})"
            )

        with saved.open("rb") as f:
            obj = pickle.load(f)

        custom_objects: dict = {}
        custom_path = artifact_dir / "custom_objects.pkl"
        if custom_path.exists():
            with custom_path.open("rb") as f:
                custom_objects = pickle.load(f)

        bento = bentoml.picklable_model.save_model(
            name, obj, custom_objects=custom_objects,
        )
        logger.info("pulled %s → local tag %s", model_tag, bento.tag)
        return str(bento.tag)
