"""Exact, Qt-free identities for local scientific source files and stores."""

from __future__ import annotations

import hashlib
import stat
from dataclasses import dataclass
from pathlib import Path

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


def capture_local_source_identity(path: str | Path) -> LocalSourceIdentity:
    """Hash every scientific byte and relative regular-file path at ``path``."""
    source = Path(path).expanduser()
    if source.is_dir():
        records = _directory_file_records(source)
        kind = "directory"
    elif _is_regular_file(source):
        records = ((".", source),)
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
    for relative_path, file_path in records:
        file_sha256, size_bytes = _hash_regular_file(file_path)
        relative_bytes = relative_path.encode("utf-8", errors="surrogateescape")
        identity_hasher.update(len(relative_bytes).to_bytes(8, "big"))
        identity_hasher.update(relative_bytes)
        identity_hasher.update(size_bytes.to_bytes(16, "big"))
        identity_hasher.update(bytes.fromhex(file_sha256))
        total_size += size_bytes
        file_count += 1
    return LocalSourceIdentity(
        kind=kind,
        sha256=identity_hasher.hexdigest(),
        regular_file_count=file_count,
        size_bytes=total_size,
    )


def verify_local_source_identity(
    path: str | Path,
    expected: LocalSourceIdentity,
) -> LocalSourceIdentity:
    """Raise explicitly when a source differs from its pre-read identity."""
    source = Path(path).expanduser()
    try:
        observed = capture_local_source_identity(source)
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


def _directory_file_records(root: Path) -> tuple[tuple[str, Path], ...]:
    records = []
    for candidate in root.rglob("*"):
        try:
            mode = candidate.stat().st_mode
        except OSError as exc:
            raise OSError(
                f"Could not inspect local source entry: {candidate}"
            ) from exc
        if not stat.S_ISREG(mode):
            continue
        relative = candidate.relative_to(root).as_posix()
        records.append((relative, candidate))
    records.sort(key=lambda item: item[0])
    return tuple(records)


def _is_regular_file(path: Path) -> bool:
    try:
        return stat.S_ISREG(path.stat().st_mode)
    except OSError:
        return False


def _hash_regular_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size_bytes = 0
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
            size_bytes += len(chunk)
    return digest.hexdigest(), size_bytes


__all__ = [
    "LocalSourceIdentity",
    "SourceChangedError",
    "capture_local_source_identity",
    "verify_local_source_identity",
]
