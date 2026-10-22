"""Retain exact VIPP wheels after an installer or supplied bundle is removed."""

from __future__ import annotations

import hashlib
import os
import secrets
import stat
from dataclasses import replace
from pathlib import Path

from napari_vipp.installer.file_ops import replace_with_retry, unlink_with_retry
from napari_vipp.installer.models import ReleaseSpec


def wheel_cache_path(local_app_data: Path, digest: str, filename: str) -> Path:
    """Derive one artifact path under the verified Windows Known Folder."""
    if (
        len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or Path(filename).name != filename
        or not filename.endswith(".whl")
        or "/" in filename
        or "\\" in filename
    ):
        raise ValueError("The cached wheel identity is invalid.")
    return Path(local_app_data) / "VIPP" / "installer" / "wheels" / digest / filename


def verify_cached_wheel(path: Path, digest: str) -> bool:
    """Verify bytes and reject links/reparse points throughout the cache path."""
    try:
        _assert_direct_path(path)
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest() == digest
    except FileNotFoundError:
        return False


def cache_release_wheel(release: ReleaseSpec, local_app_data: Path) -> ReleaseSpec:
    """Copy a verified wheel atomically; existing mismatched content is preserved."""
    source = release.wheel_path
    if source is None or not release.wheel_sha256:
        raise ValueError("An exact wheel and SHA-256 are required for caching.")
    target = wheel_cache_path(local_app_data, release.wheel_sha256, source.name)
    _assert_direct_path(target)
    if target.exists():
        if not verify_cached_wheel(target, release.wheel_sha256):
            raise RuntimeError("The retained VIPP wheel failed its integrity check.")
        return replace(release, wheel_path=target)
    target.parent.mkdir(parents=True, exist_ok=True)
    _assert_direct_path(target.parent)
    temporary = target.with_name(f".{target.name}.{secrets.token_hex(8)}.tmp")
    try:
        with source.open("rb") as source_stream, temporary.open("xb") as output:
            digest = hashlib.sha256()
            for block in iter(lambda: source_stream.read(1024 * 1024), b""):
                digest.update(block)
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
        if digest.hexdigest() != release.wheel_sha256:
            raise RuntimeError(
                "The original VIPP wheel changed before it was retained."
            )
        _assert_direct_path(target)
        replace_with_retry(temporary, target)
    finally:
        unlink_with_retry(temporary, missing_ok=True)
    return replace(release, wheel_path=target)


def _assert_direct_path(path: Path) -> None:
    for part in (path, *path.parents):
        try:
            metadata = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode) or (
            getattr(metadata, "st_file_attributes", 0) & 0x0400
        ):
            raise RuntimeError("The VIPP wheel cache path is redirected.")
