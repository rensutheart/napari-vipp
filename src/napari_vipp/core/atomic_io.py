"""Atomic UTF-8 persistence helpers for headless VIPP code.

Temporary files are written next to their targets so the final replacement
stays on the same filesystem.  A failed serialization or replacement removes
the temporary file and leaves any existing target untouched.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Callable
from pathlib import Path

JsonNormalizer = Callable[[object], object]

_ATOMIC_REPLACE_RETRY_DELAYS = (
    0.02,
    0.04,
    0.08,
    0.16,
    0.32,
    0.64,
    1.28,
    2.0,
)
_WINDOWS_TRANSIENT_LOCK_ERRORS = {5, 32, 33}


def atomic_write_json(
    path: str | Path,
    document: object,
    *,
    normalizer: JsonNormalizer | None = None,
    ensure_ascii: bool = False,
    trailing_newline: bool = True,
) -> Path:
    """Atomically replace a standards-compliant UTF-8 JSON document.

    ``allow_nan=False`` deliberately rejects non-finite floats rather than
    emitting the non-standard JavaScript tokens accepted by Python's default
    JSON encoder.  Callers that intentionally coerce domain values may provide
    an explicit ``normalizer``.
    """
    target = _validated_target(path, kind="JSON output")
    payload = normalizer(document) if normalizer is not None else document
    temporary = _temporary_path(target)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                payload,
                stream,
                indent=2,
                ensure_ascii=ensure_ascii,
                allow_nan=False,
            )
            if trailing_newline:
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        atomic_replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def atomic_write_text(path: str | Path, content: str) -> Path:
    """Atomically replace a UTF-8 text artifact."""
    target = _validated_target(path, kind="Text output")
    temporary = _temporary_path(target)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        atomic_replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _validated_target(path: str | Path, *, kind: str) -> Path:
    raw = str(path).strip()
    if not raw:
        raise ValueError(f"{kind} path cannot be blank.")
    target = Path(raw).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _temporary_path(target: Path) -> Path:
    return target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")


def atomic_replace(source: Path, target: Path) -> None:
    """Replace ``target`` atomically, retrying transient permission locks."""
    for attempt in range(len(_ATOMIC_REPLACE_RETRY_DELAYS) + 1):
        try:
            os.replace(source, target)
            return
        except OSError as exc:
            retryable = isinstance(exc, PermissionError) or (
                getattr(exc, "winerror", None) in _WINDOWS_TRANSIENT_LOCK_ERRORS
            )
            if not retryable or attempt == len(_ATOMIC_REPLACE_RETRY_DELAYS):
                raise
            # Virus scanners, file indexers, and network filesystems can hold
            # a newly created or replaced file for several seconds. Keep the
            # retry window bounded so a genuinely unwritable path still fails.
            time.sleep(_ATOMIC_REPLACE_RETRY_DELAYS[attempt])


__all__ = ["atomic_replace", "atomic_write_json", "atomic_write_text"]
