"""Exact, Qt-free identities for local scientific source files and stores."""

from __future__ import annotations

import hashlib
import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from napari_vipp.core.progress import OperationCancelled

_HASH_CHUNK_BYTES = 1024 * 1024
_IDENTITY_DOMAIN = b"napari-vipp-local-source-v1\0"


class SourceChangedError(RuntimeError):
    """A local source no longer has the content captured before it was read."""


@dataclass(frozen=True, slots=True)
class LocalSourceIdentity:
    """Content identity for one ordinary file or directory-backed store."""

    kind: str
    sha256: str
    regular_file_count: int
    size_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "sha256": self.sha256,
            "regular_file_count": self.regular_file_count,
            "size_bytes": self.size_bytes,
        }


def capture_local_source_identity(
    path: str | Path,
    *,
    cancel_callback: Callable[[], bool] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> LocalSourceIdentity:
    """Hash every scientific byte and relative regular-file path at ``path``."""
    source = Path(path).expanduser()
    _check_cancelled(cancel_callback)
    if source.is_dir():
        records = _directory_file_records(
            source,
            cancel_callback=cancel_callback,
            progress_callback=progress_callback,
        )
        kind = "directory"
    elif _is_regular_file(source):
        records = ((".", source, source.stat().st_size),)
        kind = "file"
    elif not source.exists():
        raise FileNotFoundError(f"Local source not found: {source}")
    else:
        raise ValueError(
            f"Local source must be an ordinary file or directory: {source}"
        )

    identity_hasher = hashlib.sha256()
    identity_hasher.update(_IDENTITY_DOMAIN)
    identity_hasher.update(kind.encode("ascii"))
    total_size = 0
    file_count = 0
    expected_size = sum(size_bytes for _relative, _path, size_bytes in records)
    processed_size = 0
    _report_progress(
        progress_callback,
        0,
        expected_size,
        f"Hashing source bytes: {source}",
    )
    for relative_path, file_path, _recorded_size in records:
        _check_cancelled(cancel_callback)

        def file_progress(
            current: int,
            _total: int,
            message: str,
            processed_offset: int = processed_size,
        ) -> None:
            _report_progress(
                progress_callback,
                processed_offset + current,
                expected_size,
                message,
            )

        file_sha256, size_bytes = _hash_regular_file(
            file_path,
            cancel_callback=cancel_callback,
            progress_callback=file_progress,
        )
        relative_bytes = relative_path.encode("utf-8", errors="surrogateescape")
        identity_hasher.update(len(relative_bytes).to_bytes(8, "big"))
        identity_hasher.update(relative_bytes)
        identity_hasher.update(size_bytes.to_bytes(16, "big"))
        identity_hasher.update(bytes.fromhex(file_sha256))
        total_size += size_bytes
        processed_size += size_bytes
        file_count += 1
    _report_progress(
        progress_callback,
        total_size,
        expected_size,
        f"Source identity complete: {source}",
    )
    return LocalSourceIdentity(
        kind=kind,
        sha256=identity_hasher.hexdigest(),
        regular_file_count=file_count,
        size_bytes=total_size,
    )


def verify_local_source_identity(
    path: str | Path,
    expected: LocalSourceIdentity,
    *,
    cancel_callback: Callable[[], bool] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> LocalSourceIdentity:
    """Raise explicitly when a source differs from its pre-read identity."""
    source = Path(path).expanduser()
    try:
        observed = capture_local_source_identity(
            source,
            cancel_callback=cancel_callback,
            progress_callback=progress_callback,
        )
    except (OSError, ValueError) as exc:
        raise SourceChangedError(
            "Local scientific source changed or became unreadable during "
            f"execution: {source}"
        ) from exc
    if observed != expected:
        raise SourceChangedError(
            "Local scientific source changed during execution: "
            f"{source} (expected {expected.sha256}, observed {observed.sha256})."
        )
    return observed


def _directory_file_records(
    root: Path,
    *,
    cancel_callback: Callable[[], bool] | None,
    progress_callback: Callable[[int, int, str], None] | None,
) -> tuple[tuple[str, Path, int], ...]:
    records = []
    inspected = 0
    for candidate in root.rglob("*"):
        _check_cancelled(cancel_callback)
        try:
            candidate_stat = candidate.stat()
        except OSError as exc:
            raise OSError(
                f"Could not inspect local source entry: {candidate}"
            ) from exc
        inspected += 1
        _report_progress(
            progress_callback,
            inspected,
            0,
            f"Discovering source files: {root}",
        )
        if not stat.S_ISREG(candidate_stat.st_mode):
            continue
        relative = candidate.relative_to(root).as_posix()
        records.append((relative, candidate, candidate_stat.st_size))
    records.sort(key=lambda item: item[0])
    return tuple(records)


def _is_regular_file(path: Path) -> bool:
    try:
        return stat.S_ISREG(path.stat().st_mode)
    except OSError:
        return False


def _hash_regular_file(
    path: Path,
    *,
    cancel_callback: Callable[[], bool] | None,
    progress_callback: Callable[[int, int, str], None] | None,
) -> tuple[str, int]:
    digest = hashlib.sha256()
    size_bytes = 0
    try:
        expected_size = path.stat().st_size
    except OSError:
        expected_size = 0
    with path.open("rb") as stream:
        while True:
            _check_cancelled(cancel_callback)
            chunk = stream.read(_HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            size_bytes += len(chunk)
            _report_progress(
                progress_callback,
                size_bytes,
                expected_size,
                f"Hashing source file: {path}",
            )
    return digest.hexdigest(), size_bytes


def _check_cancelled(cancel_callback: Callable[[], bool] | None) -> None:
    if cancel_callback is not None and cancel_callback():
        raise OperationCancelled("Cancelled while validating a source identity.")


def _report_progress(
    callback: Callable[[int, int, str], None] | None,
    current: int,
    total: int,
    message: str,
) -> None:
    if callback is None:
        return
    try:
        callback(int(current), int(total), str(message))
    except Exception:
        # Presentation hooks never invalidate source identity verification.
        return


__all__ = [
    "LocalSourceIdentity",
    "SourceChangedError",
    "capture_local_source_identity",
    "verify_local_source_identity",
]
