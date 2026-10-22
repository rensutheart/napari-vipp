"""Bound brief Windows sharing locks after caller-owned path validation."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path

ATOMIC_REPLACE_RETRY_DELAYS = (0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.0, 1.0)
_WINDOWS_TRANSIENT_ERRORS = frozenset({5, 32, 33})


def retry_windows_file_operation(
    operation: Callable[[], None],
    *,
    delays: tuple[float, ...] = ATOMIC_REPLACE_RETRY_DELAYS,
) -> None:
    for attempt in range(len(delays) + 1):
        try:
            operation()
            return
        except OSError as exc:
            transient = os.name == "nt" and (
                getattr(exc, "winerror", None) in _WINDOWS_TRANSIENT_ERRORS
                or exc.errno in {13, 16}
            )
            if not transient or attempt == len(delays):
                raise
            time.sleep(delays[attempt])


def replace_with_retry(source: Path, destination: Path) -> None:
    """Atomically replace without a delete-first fallback."""

    retry_windows_file_operation(lambda: os.replace(source, destination))


def unlink_with_retry(path: Path, *, missing_ok: bool = False) -> None:
    retry_windows_file_operation(lambda: path.unlink(missing_ok=missing_ok))
