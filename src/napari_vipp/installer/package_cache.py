"""Reusable, content-addressed wheels without trusting pip's mutable cache.

Only an explicit cache root is changed. Existing invalid entries are preserved,
not repaired or overwritten. The caller owns transport policy: ``fetcher`` must
return a context-managed binary response with a bounded blocking-read timeout.
This module validates source/final URLs, bounds bytes and elapsed work between
reads, and never follows a response redirect itself.
"""

from __future__ import annotations

import hashlib
import os
import re
import secrets
import stat
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO
from urllib.parse import unquote, urlsplit

from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

_APPROVED_HOSTS = frozenset({"files.pythonhosted.org", "pypi.org"})
_CHUNK_BYTES = 1024 * 1024
_DEFAULT_MAX_BYTES = 4 * 1024**3
_SHA256 = re.compile(r"[a-f0-9]{64}\Z")
_FILENAME = re.compile(r"[A-Za-z0-9_.+-]+\.whl\Z")


class PackageCacheError(RuntimeError):
    """A wheel cache boundary could not be verified safely."""


class PackageCacheCancelled(PackageCacheError):
    """The caller cancelled before a complete artifact was published."""


@dataclass(frozen=True)
class PackageArtifact:
    """One exact wheel identity from the caller's reviewed package plan."""

    name: str
    version: str
    source_url: str
    sha256: str

    def __post_init__(self) -> None:
        if not _SHA256.fullmatch(self.sha256):
            raise ValueError("An exact lowercase SHA-256 is required.")
        filename = _url_filename(self.source_url)
        try:
            wheel_name, wheel_version, _, _ = parse_wheel_filename(filename)
            expected_version = Version(self.version)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                "The package must identify a valid wheel version."
            ) from exc
        if (
            not self.name
            or wheel_name != canonicalize_name(self.name)
            or wheel_version != expected_version
        ):
            raise ValueError("The wheel filename differs from the reviewed package.")

    @property
    def filename(self) -> str:
        return _url_filename(self.source_url)


def package_cache_path(cache_root: Path, artifact: PackageArtifact) -> Path:
    """Return the exact content-addressed path, without creating it."""

    root = Path(cache_root)
    if not root.is_absolute() or ".." in root.parts:
        raise PackageCacheError("The package cache root must be an absolute path.")
    _assert_direct_path(root)
    return root / artifact.sha256 / artifact.filename


def get_cached_package(
    cache_root: Path,
    artifact: PackageArtifact,
    *,
    cancellation: object | None = None,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    max_io_seconds: float = 600,
) -> Path | None:
    """Return a fully rehashed wheel, or None for absence; preserve bad entries."""

    _validate_limits(max_bytes, max_io_seconds)
    path = package_cache_path(cache_root, artifact)
    deadline = time.monotonic() + max_io_seconds
    return _verified_entry(path, artifact, cancellation, max_bytes, deadline)


def cache_package(
    cache_root: Path,
    artifact: PackageArtifact,
    *,
    fetcher: Callable[[str], AbstractContextManager[BinaryIO]],
    cancellation: object | None = None,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    max_io_seconds: float = 600,
    lock_timeout_seconds: float = 30,
) -> Path:
    """Reuse verified bytes or fetch/publish once, never clobbering an entry.

    The per-digest lock is cancellable and bounded; a foreign or stale lock is
    never removed. Transport must impose its own blocking-read timeout. Our
    elapsed budget and byte cap apply to cache checks and every fetched chunk.
    A response exposing ``geturl`` is rejected if its final URL is not official
    HTTPS or if it names a different wheel. Fetcher errors remain failures.
    """

    _validate_limits(max_bytes, max_io_seconds)
    if not 0 < lock_timeout_seconds <= 300:
        raise ValueError("The cache lock timeout must be in (0, 300] seconds.")
    path = package_cache_path(cache_root, artifact)
    deadline = time.monotonic() + max_io_seconds
    cached = _verified_entry(path, artifact, cancellation, max_bytes, deadline)
    if cached is not None:
        return cached
    _ensure_directory(path.parent)
    with _artifact_lock(
        path.parent / ".download.lock",
        cancellation,
        min(deadline, time.monotonic() + lock_timeout_seconds),
    ):
        cached = _verified_entry(path, artifact, cancellation, max_bytes, deadline)
        if cached is not None:
            return cached
        temporary = path.with_name(f".{path.name}.{secrets.token_hex(16)}.partial")
        temporary_identity = None
        try:
            _checkpoint(cancellation, deadline)
            _assert_direct_path(temporary)
            with temporary.open("xb") as output:
                temporary_identity = _identity(os.fstat(output.fileno()))
                digest = hashlib.sha256()
                received = 0
                with fetcher(artifact.source_url) as response:
                    geturl = getattr(response, "geturl", None)
                    if geturl is not None:
                        final_filename = _url_filename(geturl())
                        if final_filename != artifact.filename:
                            raise PackageCacheError(
                                "The download changed wheel identity."
                            )
                    while True:
                        _checkpoint(cancellation, deadline)
                        block = response.read(_CHUNK_BYTES)
                        _checkpoint(cancellation, deadline)
                        if not isinstance(block, bytes) or len(block) > _CHUNK_BYTES:
                            raise PackageCacheError(
                                "The wheel response is not bounded binary data."
                            )
                        if not block:
                            break
                        received += len(block)
                        if received > max_bytes:
                            raise PackageCacheError(
                                "The wheel exceeds the cache size limit."
                            )
                        digest.update(block)
                        output.write(block)
                if not received or digest.hexdigest() != artifact.sha256:
                    raise PackageCacheError(
                        "The downloaded wheel failed its SHA-256 check."
                    )
                output.flush()
                os.fsync(output.fileno())
            _checkpoint(cancellation, deadline)
            _assert_direct_path(path)
            _assert_identity(temporary, temporary_identity)
            try:
                # A hard link publishes a complete file atomically and refuses
                # an existing destination on both Windows and POSIX. Unlike
                # replace(), a racing foreign file can never be overwritten.
                os.link(temporary, path, follow_symlinks=False)
            except FileExistsError:
                pass
            cached = _verified_entry(path, artifact, cancellation, max_bytes, deadline)
            if cached is None:
                raise PackageCacheError("The completed wheel was not published.")
            return cached
        finally:
            if temporary_identity is not None:
                _remove_owned_file(temporary, temporary_identity)


def materialize_packages(
    packages,
    cache_root: Path,
    download_directory: Path,
    *,
    downloader: Callable[[Path, Path], object],
    cancellation: object | None = None,
) -> tuple[Path, ...]:
    """Return verified local wheels in reviewed package order.

    ``packages`` supplies name/version/source_url/sha256 attributes. Only the
    missing official HTTPS artifacts enter one direct-URL, hash-locked batch.
    The caller's downloader must impose its network/child process bounds and
    return only after actual successful completion, raising on any failure.
    It receives the lock path and a new empty destination, both inside a new
    operation directory. Its files remain available as diagnostic evidence.
    Local bundled VIPP artifacts must be validated separately by the caller.
    """

    artifacts = tuple(
        PackageArtifact(
            package.name, package.version, package.source_url, package.sha256
        )
        for package in packages
    )
    names = [canonicalize_name(artifact.name) for artifact in artifacts]
    filenames = [artifact.filename for artifact in artifacts]
    if len(names) != len(set(names)) or len(filenames) != len(set(filenames)):
        raise PackageCacheError("The package batch contains duplicate identities.")
    paths = [
        get_cached_package(cache_root, artifact, cancellation=cancellation)
        for artifact in artifacts
    ]
    missing = [
        artifact
        for artifact, path in zip(artifacts, paths, strict=True)
        if path is None
    ]
    if not missing:
        return tuple(paths)

    root = Path(download_directory)
    if not root.is_absolute() or ".." in root.parts:
        raise PackageCacheError("The download directory must be an absolute path.")
    _ensure_directory(root)
    operation = root / f"wheel-download-{secrets.token_hex(16)}"
    operation.mkdir(exist_ok=False)
    destination = operation / "wheels"
    destination.mkdir()
    lock_path = operation / "requirements.lock"
    lock = "".join(
        f"{artifact.name} @ {artifact.source_url} --hash=sha256:{artifact.sha256}\n"
        for artifact in missing
    ).encode("utf-8")
    with lock_path.open("xb") as stream:
        stream.write(lock)
        stream.flush()
        os.fsync(stream.fileno())
        lock_identity = _identity(os.fstat(stream.fileno()))
    _check_cancelled(cancellation)
    downloader(lock_path, destination)
    # There is no installer mutation after a cancelled or failed download.
    _check_cancelled(cancellation)
    _assert_identity(lock_path, lock_identity)
    with lock_path.open("rb") as stream:
        if stream.read(len(lock) + 1) != lock:
            raise PackageCacheError("The reviewed download lock changed.")
    _assert_direct_path(destination)
    expected = {artifact.filename for artifact in missing}
    entries = tuple(destination.iterdir())
    if {entry.name for entry in entries} != expected or len(entries) != len(expected):
        raise PackageCacheError(
            "The download did not contain exactly the reviewed wheels."
        )
    for entry in entries:
        _assert_direct_path(entry)
        if not stat.S_ISREG(entry.lstat().st_mode):
            raise PackageCacheError("A downloaded wheel is not a regular file.")

    for index, (artifact, path) in enumerate(zip(artifacts, paths, strict=True)):
        if path is not None:
            continue
        source = destination / artifact.filename

        @contextmanager
        def local_fetcher(_url, source=source):
            _assert_direct_path(source)
            before = source.lstat()
            if not stat.S_ISREG(before.st_mode):
                raise PackageCacheError("A downloaded wheel is not a regular file.")
            with source.open("rb") as stream:
                if _identity(os.fstat(stream.fileno())) != _identity(before):
                    raise PackageCacheError("A downloaded wheel changed identity.")
                yield stream
            _assert_identity(source, _identity(before))

        paths[index] = cache_package(
            cache_root,
            artifact,
            fetcher=local_fetcher,
            cancellation=cancellation,
        )
    # Downloading may take a while. Do not return an earlier cache hit that
    # changed while the caller's child process was running.
    for artifact in artifacts:
        if get_cached_package(cache_root, artifact, cancellation=cancellation) is None:
            raise PackageCacheError("A reviewed cached wheel disappeared.")
    return tuple(paths)


def _url_filename(url: str) -> str:
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except (TypeError, ValueError) as exc:
        raise ValueError("The package artifact URL is invalid.") from exc
    if (
        not isinstance(url, str)
        or any(ord(character) < 32 for character in url)
        or "\\" in url
        or parsed.scheme != "https"
        or parsed.hostname not in _APPROVED_HOSTS
        or port not in (None, 443)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Only credential-free official HTTPS wheel URLs are allowed.")
    filename = unquote(parsed.path.rsplit("/", 1)[-1])
    if not _FILENAME.fullmatch(filename):
        raise ValueError("The package URL must end in a safe wheel filename.")
    return filename


def _validate_limits(max_bytes: int, max_io_seconds: float) -> None:
    if not isinstance(max_bytes, int) or not 0 < max_bytes <= 16 * 1024**3:
        raise ValueError("The cache byte limit must be in (0, 16 GiB].")
    if not 0 < max_io_seconds <= 3600:
        raise ValueError("The cache I/O budget must be in (0, 3600] seconds.")


def _checkpoint(cancellation: object | None, deadline: float) -> None:
    _check_cancelled(cancellation)
    if time.monotonic() >= deadline:
        raise PackageCacheError("The package cache operation exceeded its time budget.")


def _check_cancelled(cancellation: object | None) -> None:
    if cancellation is not None:
        if callable(cancellation):
            predicate = cancellation
        else:
            predicate = getattr(cancellation, "is_set", None)
            if predicate is None:
                predicate = getattr(cancellation, "is_cancelled", None)
        if not callable(predicate):
            raise TypeError("Cancellation requires a predicate or cancellation token.")
        if predicate():
            raise PackageCacheCancelled("Package caching was cancelled.")


def _assert_direct_path(path: Path) -> None:
    for part in (*reversed(path.parents), path):
        try:
            metadata = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode) or (
            getattr(metadata, "st_file_attributes", 0) & 0x0400
        ):
            raise PackageCacheError("The package cache path is redirected.")
        if part != path and not stat.S_ISDIR(metadata.st_mode):
            raise PackageCacheError("A package cache parent is not a directory.")


def _identity(metadata: os.stat_result) -> tuple[int, int]:
    return metadata.st_dev, metadata.st_ino


def _assert_identity(path: Path, identity: tuple[int, int]) -> os.stat_result:
    _assert_direct_path(path)
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or _identity(metadata) != identity:
        raise PackageCacheError(
            "An owned cache file changed identity; it was preserved."
        )
    return metadata


def _verified_entry(
    path: Path,
    artifact: PackageArtifact,
    cancellation: object | None,
    max_bytes: int,
    deadline: float,
) -> Path | None:
    _checkpoint(cancellation, deadline)
    _assert_direct_path(path)
    try:
        before = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= max_bytes:
        raise PackageCacheError("An invalid package cache entry was preserved.")
    digest = hashlib.sha256()
    received = 0
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        if _identity(opened) != _identity(before):
            raise PackageCacheError("The cached wheel changed before verification.")
        while block := stream.read(_CHUNK_BYTES):
            _checkpoint(cancellation, deadline)
            received += len(block)
            if received > max_bytes:
                raise PackageCacheError("The cached wheel exceeds its size limit.")
            digest.update(block)
        after = os.fstat(stream.fileno())
    _checkpoint(cancellation, deadline)
    current = _assert_identity(path, _identity(before))
    def stamp(value):
        # Windows hard-link creation times differ between stat(path) and
        # fstat(handle). Identity, size, last-write time and the complete hash
        # must agree; a platform-dependent creation time is not an invariant.
        return value.st_size, value.st_mtime_ns
    if (
        stamp(before) != stamp(opened)
        or stamp(before) != stamp(after)
        or stamp(before) != stamp(current)
        or received != before.st_size
        or digest.hexdigest() != artifact.sha256
    ):
        raise PackageCacheError(
            "The cached wheel failed verification and was preserved."
        )
    return path


def _ensure_directory(path: Path) -> None:
    _assert_direct_path(path)
    path.mkdir(parents=True, exist_ok=True)
    _assert_direct_path(path)
    if not path.is_dir():
        raise PackageCacheError("The package cache location is not a directory.")


def _remove_owned_file(path: Path, identity: tuple[int, int]) -> None:
    try:
        _assert_identity(path, identity)
    except FileNotFoundError:
        return
    path.unlink()


@contextmanager
def _artifact_lock(
    path: Path, cancellation: object | None, deadline: float
) -> Iterator[None]:
    identity = None
    while identity is None:
        _checkpoint(cancellation, deadline)
        _assert_direct_path(path)
        try:
            with path.open("xb") as stream:
                identity = _identity(os.fstat(stream.fileno()))
        except FileExistsError:
            if not stat.S_ISREG(path.lstat().st_mode):
                raise PackageCacheError("A foreign cache lock was preserved.") from None
            time.sleep(min(0.05, max(0, deadline - time.monotonic())))
    try:
        yield
    finally:
        _remove_owned_file(path, identity)
