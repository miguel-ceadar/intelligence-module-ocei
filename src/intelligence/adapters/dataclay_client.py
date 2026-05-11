"""Centralised DataClay client.

The legacy code instantiated ``dataclay.Client(...)`` and called
``start()`` / ``stop()`` in five places (``processing/process.py``, the
three ``models/*_compiler.py`` files, and ``api_service.train``). This
module is the single place that imports ``dataclay`` and the single
place that knows the connection params.

Phase 2 deletes this file along with everything else DataClay-related.

``dataclay`` is imported lazily inside the context manager so the module
itself is importable without the ``dataclay`` package installed —
useful for a Mac dev loop that doesn't want the gRPC stack.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

logger = logging.getLogger(__name__)


@dataclass
class DataClayConfig:
    """Connection params for a DataClay client.

    Build via ``from_legacy_args`` from the existing argparse / pydantic
    shape, or construct directly.
    """

    proxy_host: str = "127.0.0.1"
    username: str = "testuser"
    password: str = "s3cret"
    dataset: str = "testdata"
    enabled: bool = False


def from_legacy_args(args: Any) -> DataClayConfig:
    """Build a ``DataClayConfig`` from the legacy ``args.dataclay*`` shape
    (argparse Namespace or a pydantic model with the same attribute names)."""
    return DataClayConfig(
        enabled=bool(getattr(args, "dataclay", False)),
        proxy_host=getattr(args, "dataclay_host", "127.0.0.1"),
        username=getattr(args, "dataclay_hostname", "testuser"),
        password=getattr(args, "dataclay_password", "s3cret"),
        dataset=getattr(args, "dataclay_dataset", "testdata"),
    )


@contextmanager
def dataclay_client(config: DataClayConfig | Any) -> Iterator[Any | None]:
    """Open a DataClay client for the body of a ``with`` block.

    Yields the live client when ``config.enabled`` is true, otherwise
    yields ``None`` — so callers can write::

        with dataclay_client(args) as client:
            ...

    and unconditionally enter the block. The ``dataclay`` package is
    imported lazily on first call; importing this module does not
    require it.
    """
    if not isinstance(config, DataClayConfig):
        config = from_legacy_args(config)

    if not config.enabled:
        logger.debug("DataClay disabled; skipping client connection")
        yield None
        return

    from dataclay import Client  # lazy: only required when actually connecting

    logger.info("Connecting to DataClay at %s", config.proxy_host)
    client = Client(
        proxy_host=config.proxy_host,
        username=config.username,
        password=config.password,
        dataset=config.dataset,
    )
    client.start()
    try:
        yield client
    finally:
        logger.info("Disconnecting DataClay client")
        try:
            client.stop()
        except Exception:
            logger.exception("Error stopping DataClay client")
