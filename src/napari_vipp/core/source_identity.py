"""Exact, Qt-free identities for local scientific source files and stores."""

from __future__ import annotations

import hashlib
import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from napari_vipp.core.progress import OperationCancelled
from napari_vipp.core.source_items import (
    SourceContainerBundle,
    SourceContainerMember,
    SourceRevisionProof,
)

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


@dataclass(frozen=True, slots=True)
class SourceRevisionToken:
    """Identity of one VIPP-owned snapshot of a live viewer layer."""

    layer_id: int
    revision: int


@dataclass(frozen=True, slots=True)
class BundledSampleRevisionToken:
    """Stable identity for one immutable sample from VIPP's bundled catalog."""

    name: str
    catalog_schema: str = "vipp-synthetic-samples-v1"

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        catalog_schema = str(self.catalog_schema).strip()
        if not name or not catalog_schema:
            raise ValueError(
                "Bundled sample name and catalog schema must not be empty."
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "catalog_schema", catalog_schema)


def is_vipp_owned_immutable_source_revision(value: object) -> bool:
    """Return whether ``value`` is an exact, recognized VIPP revision type.

    Exact type checks deliberately reject arbitrary user tokens and subclasses.
    These three revisions are paired with arrays that VIPP owns and marks
    read-only before detached execution begins.
    """

    return type(value) in {
        SourceRevisionToken,
        LocalSourceIdentity,
        BundledSampleRevisionToken,
    }


def capture_local_source_identity(
    path: str | Path,
    *,
    cancel_callback: Callable[[], bool] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> LocalSourceIdentity:
    """Hash every scientific byte and relative regular-file path at ``path``."""
    bundle = capture_local_source_bundle(
        path,
        cancel_callback=cancel_callback,
        progress_callback=progress_callback,
    )
    return local_source_identity_from_bundle(bundle)


def capture_local_source_bundle(
    path: str | Path,
    *,
    source_format: str = "local-source",
    cancel_callback: Callable[[], bool] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> SourceContainerBundle:
    """Hash the exact file inventory that forms one local source container.

    Ordinary files contain one ``.`` member, directory stores contain every
    relative regular file, and known multifile microscope containers include
    their required companion tree.  The selected URI remains private evidence;
    public SourceItem representations omit it.
    """
    source = Path(path).expanduser()
    _check_cancelled(cancel_callback)
    if source.is_dir():
        raw_records = _directory_file_records(
            source,
            cancel_callback=cancel_callback,
            progress_callback=progress_callback,
        )
        records = tuple(
            (relative, member_path, size_bytes, "data")
            for relative, member_path, size_bytes in raw_records
        )
        kind = "directory"
    elif _is_regular_file(source):
        records = _file_container_records(
            source,
            cancel_callback=cancel_callback,
            progress_callback=progress_callback,
        )
        kind = "multifile" if len(records) > 1 else "file"
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
    expected_size = sum(
        size_bytes for _relative, _path, size_bytes, _role in records
    )
    processed_size = 0
    members: list[SourceContainerMember] = []
    _report_progress(
        progress_callback,
        0,
        expected_size,
        f"Hashing source bytes: {source}",
    )
    for relative_path, file_path, _recorded_size, role in records:
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
        members.append(
            SourceContainerMember(
                key=relative_path,
                sha256=file_sha256,
                size_bytes=size_bytes,
                role=role,
            )
        )
    _report_progress(
        progress_callback,
        total_size,
        expected_size,
        f"Source identity complete: {source}",
    )
    revision = SourceRevisionProof(
        kind=kind,
        sha256=identity_hasher.hexdigest(),
        regular_file_count=file_count,
        size_bytes=total_size,
    )
    return SourceContainerBundle(
        uri=str(source.resolve(strict=False)),
        format=_normalized_source_format(source_format),
        revision=revision,
        members=tuple(members),
    )


def local_source_identity_from_bundle(
    bundle: SourceContainerBundle,
) -> LocalSourceIdentity:
    """Return the legacy-compatible exact identity for a captured bundle."""
    revision = bundle.revision
    return LocalSourceIdentity(
        kind=revision.kind,
        sha256=revision.sha256,
        regular_file_count=revision.regular_file_count,
        size_bytes=revision.size_bytes,
    )


def _normalized_source_format(value: str) -> str:
    normalized = str(value or "local-source").strip().lower()
    normalized = "-".join(normalized.replace("_", "-").split())
    return normalized or "local-source"


def _file_container_records(
    source: Path,
    *,
    cancel_callback: Callable[[], bool] | None,
    progress_callback: Callable[[int, int, str], None] | None,
) -> tuple[tuple[str, Path, int, str], ...]:
    records: list[tuple[str, Path, int, str]] = [
        (".", source, source.stat().st_size, "primary")
    ]
    companion = _required_companion_directory(source)
    if companion is None:
        return tuple(records)
    companion_records = _directory_file_records(
        companion,
        cancel_callback=cancel_callback,
        progress_callback=progress_callback,
    )
    if not companion_records:
        raise FileNotFoundError(
            f"Required companion directory contains no readable files: {companion}"
        )
    records.extend(
        (
            f"{companion.name}/{relative}",
            member_path,
            size_bytes,
            "companion",
        )
        for relative, member_path, size_bytes in companion_records
    )
    records.sort(key=lambda item: item[0])
    return tuple(records)


def _required_companion_directory(source: Path) -> Path | None:
    suffix = source.suffix.casefold()
    expected_name = ""
    if suffix == ".vsi":
        expected_name = f"_{source.stem}_"
    elif suffix == ".oif":
        expected_name = f"{source.stem}.files"
    else:
        return None

    expected = source.parent / expected_name
    if expected.is_dir():
        return expected
    try:
        casefold_match = next(
            (
                candidate
                for candidate in source.parent.iterdir()
                if candidate.name.casefold() == expected_name.casefold()
                and candidate.is_dir()
            ),
            None,
        )
    except OSError:
        casefold_match = None
    if casefold_match is not None:
        return casefold_match
    raise FileNotFoundError(
        "Required microscope companion directory is missing: "
        f"{expected}. Restore the companion files beside {source.name}."
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
            raise OSError(f"Could not inspect local source entry: {candidate}") from exc
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
    "BundledSampleRevisionToken",
    "LocalSourceIdentity",
    "SourceChangedError",
    "SourceRevisionToken",
    "capture_local_source_bundle",
    "capture_local_source_identity",
    "is_vipp_owned_immutable_source_revision",
    "local_source_identity_from_bundle",
    "verify_local_source_identity",
]
