"""Verified installer download and guided handoff, never in-place updating.

Only an active, owned Windows desktop installation can use this path. A
downloaded program is opened only after a fresh checksum and ownership check.
The existing installer still owns its review, authorization and transaction.
"""

from __future__ import annotations

import hashlib
import os
import platform
import re
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from packaging.version import InvalidVersion, Version

from napari_vipp.core.updates import PublishedRelease, installer_asset
from napari_vipp.installer.ownership import OwnershipState, inspect_ownership

MAX_INSTALLER_BYTES = 2 * 1024**3
MAX_CHECKSUM_BYTES = 256 * 1024
NETWORK_TIMEOUT_SECONDS = 20
DOWNLOAD_DEADLINE_SECONDS = 30 * 60
CHUNK_BYTES = 256 * 1024
_ASSET_HOSTS = frozenset(
    {
        "github.com",
        "release-assets.githubusercontent.com",
        "objects.githubusercontent.com",
    }
)


class UpdateInstallError(RuntimeError):
    """An actionable failure that leaves the installed application unchanged."""


class UpdateDownloadCancelled(UpdateInstallError):
    """The user cancelled before installer handoff."""


@dataclass(frozen=True)
class ManagedUpdateTarget:
    managed_root: Path
    environment_root: Path
    track: str
    base_python: Path | None
    installation_id: str
    ownership_sha256: str
    current_version: str
    package_path: Path


@dataclass(frozen=True)
class InstallerDownloadRequest:
    release: PublishedRelease
    target: ManagedUpdateTarget
    installer_name: str
    checksum_name: str


@dataclass(frozen=True)
class DownloadProgress:
    phase: str
    received_bytes: int = 0
    total_bytes: int | None = None


@dataclass(frozen=True)
class VerifiedInstaller:
    request: InstallerDownloadRequest
    path: Path
    sha256: str
    size: int
    file_identity: tuple[int, int]
    directory_identity: tuple[int, int]


def _file_identity(path: Path) -> tuple[int, int]:
    metadata = path.stat()
    return metadata.st_dev, metadata.st_ino


def _normal_path(path: Path, *, directory: bool = False) -> bool:
    """Reject redirected components, including Windows junctions."""
    try:
        for component in (path, *path.parents):
            metadata = component.lstat()
            if stat.S_ISLNK(metadata.st_mode) or (
                getattr(metadata, "st_file_attributes", 0) & 0x0400
            ):
                return False
        return path.is_dir() if directory else path.is_file()
    except OSError:
        return False


def _same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(os.path.abspath(left)) == os.path.normcase(
        os.path.abspath(right)
    )


def managed_update_target(
    *,
    prefix: Path | None = None,
    package_path: Path | None = None,
    system: str | None = None,
    machine: str | None = None,
    desktop: bool | None = None,
    current_version: str | None = None,
) -> ManagedUpdateTarget | None:
    """Recognize this active managed desktop, not a nearby or retired install.

    macOS constructor packages do not yet expose an ownership-bound updater;
    plugin, manual and source environments must use their documented route.
    """
    if (system or platform.system()) != "Windows" or (
        (machine or platform.machine()).lower() not in {"amd64", "x86_64"}
    ):
        return None
    if not ("--desktop" in sys.argv if desktop is None else desktop):
        return None
    environment = Path(os.path.abspath(prefix or sys.prefix))
    package = Path(os.path.abspath(package_path or __file__))
    if not _normal_path(environment, directory=True) or not _normal_path(package):
        return None
    try:
        package.relative_to(environment)
        root = environment.parents[2]
    except (ValueError, IndexError):
        return None
    inspection = inspect_ownership(root)
    record = inspection.record
    if (
        inspection.state is not OwnershipState.VALID
        or record is None
        or record.distribution.lower().replace("_", "-") != "napari-vipp"
        or not _same_path(record.environment_root, environment)
        or record.track.value not in {"cpu", "cuda13"}
    ):
        return None
    try:
        installed = Version(current_version or distribution_version("napari-vipp"))
        if installed != Version(record.version):
            return None
    except (InvalidVersion, PackageNotFoundError):
        return None
    marker = environment / ".vipp-install-candidate.json"
    try:
        if not _normal_path(marker) or marker.stat().st_size > MAX_CHECKSUM_BYTES:
            return None
        if (
            hashlib.sha256(marker.read_bytes()).hexdigest()
            != record.environment_marker_sha256
        ):
            return None
    except OSError:
        return None
    base = record.base_python if _normal_path(record.base_python) else None
    return ManagedUpdateTarget(
        root,
        environment,
        record.track.value,
        base,
        record.installation_id,
        inspection.manifest_sha256,
        record.version,
        package,
    )


def installer_download_request(
    release: PublishedRelease, target: ManagedUpdateTarget
) -> InstallerDownloadRequest:
    """Capture exact published assets and current installation identity."""
    names = installer_asset(release, "Windows", "x86_64")
    if (
        names is None
        or release.tag != f"v{release.version}"
        or release.version.is_devrelease
        or release.version.local
        or release.version <= Version(target.current_version)
    ):
        raise UpdateInstallError("No newer matching Windows installer is available.")
    return InstallerDownloadRequest(release, target, *names)


def _validate_request(request: InstallerDownloadRequest) -> None:
    if installer_download_request(request.release, request.target) != request:
        raise UpdateInstallError(
            "The selected update changed. Check for updates again."
        )


def _validate_target(target: ManagedUpdateTarget) -> None:
    current = managed_update_target()
    if current != target:
        raise UpdateInstallError(
            "This installation changed while the update was prepared. "
            "Close this update and check again; nothing was installed."
        )


def _safe_asset_url(url: str) -> None:
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise UpdateInstallError("The official download URL is malformed.") from exc
    if (
        parsed.scheme != "https"
        or parsed.hostname not in _ASSET_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or port not in {None, 443}
        or parsed.fragment
    ):
        raise UpdateInstallError(
            "The download redirected outside official HTTPS assets."
        )


class _OfficialRedirects(HTTPRedirectHandler):
    max_redirections = 5
    max_repeats = 2

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _safe_asset_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _checkpoint(cancelled: Callable[[], bool], deadline: float) -> None:
    if cancelled():
        raise UpdateDownloadCancelled(
            "Update download cancelled. Nothing was installed."
        )
    if time.monotonic() > deadline:
        raise UpdateInstallError(
            "The update download timed out. Please try again later."
        )


def _transfer(
    url: str,
    *,
    limit: int,
    opener,
    cancelled: Callable[[], bool],
    deadline: float,
    write: Callable[[bytes], object],
    progress: Callable[[DownloadProgress], object] | None = None,
) -> tuple[str, int]:
    _safe_asset_url(url)
    _checkpoint(cancelled, deadline)
    request = Request(url, headers={"User-Agent": "napari-vipp-update-download"})
    digest, received = hashlib.sha256(), 0
    with opener.open(request, timeout=NETWORK_TIMEOUT_SECONDS) as response:
        _safe_asset_url(response.geturl())
        if response.status != 200:
            raise UpdateInstallError("The official installer download was unavailable.")
        declared = response.headers.get("Content-Length")
        try:
            total = int(declared) if declared is not None else None
        except ValueError as exc:
            raise UpdateInstallError(
                "The download returned an invalid file size."
            ) from exc
        if total is not None and not 0 < total <= limit:
            raise UpdateInstallError("The update download exceeds its safe size limit.")
        if progress:
            progress(DownloadProgress("downloading", 0, total))
        while True:
            _checkpoint(cancelled, deadline)
            chunk = response.read(min(CHUNK_BYTES, limit - received + 1))
            if not chunk:
                break
            received += len(chunk)
            if received > limit:
                raise UpdateInstallError(
                    "The update download exceeds its safe size limit."
                )
            digest.update(chunk)
            write(chunk)
            if progress:
                progress(DownloadProgress("downloading", received, total))
        _checkpoint(cancelled, deadline)
        if received == 0 or (total is not None and received != total):
            raise UpdateInstallError(
                "The update download was incomplete. Please try again."
            )
    return digest.hexdigest(), received


def _expected_checksum(payload: bytes, name: str) -> str:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeError as exc:
        raise UpdateInstallError(
            "The official checksum file could not be read."
        ) from exc
    matches = []
    for line in text.splitlines():
        match = re.fullmatch(r"([0-9A-Fa-f]{64}) [ *](.+)", line)
        if match and match[2] == name:
            matches.append(match[1].lower())
    if len(matches) != 1:
        raise UpdateInstallError(
            "The official checksum does not uniquely identify this installer."
        )
    return matches[0]


def download_installer(
    request: InstallerDownloadRequest,
    *,
    progress: Callable[[DownloadProgress], object] | None = None,
    cancelled: Callable[[], bool] | None = None,
    cache_parent: Path | None = None,
    opener=None,
) -> VerifiedInstaller:
    """Download into a new private directory; never run or replace anything.

    Cancellation is checked between bounded reads (at most the socket timeout).
    Failures remove only this attempt's files. Successful bytes remain available
    for explicit handoff and are rehashed immediately before launching setup.
    """
    _validate_request(request)
    _validate_target(request.target)
    cancel = cancelled or (lambda: False)
    deadline = time.monotonic() + DOWNLOAD_DEADLINE_SECONDS
    _checkpoint(cancel, deadline)
    transfer = opener or build_opener(_OfficialRedirects())
    if cache_parent is not None and not _normal_path(
        Path(cache_parent), directory=True
    ):
        raise UpdateInstallError(
            "The update download folder is unavailable or redirected."
        )
    folder = Path(tempfile.mkdtemp(prefix="vipp-update-", dir=cache_parent))
    partial = folder / (request.installer_name + ".part")
    final = folder / request.installer_name
    created_final = False
    try:
        if not _normal_path(folder, directory=True):
            raise UpdateInstallError("The update download folder is redirected.")
        checksum = bytearray()
        if progress:
            progress(DownloadProgress("downloading"))
        _transfer(
            request.release.asset_url(request.checksum_name),
            limit=MAX_CHECKSUM_BYTES,
            opener=transfer,
            cancelled=cancel,
            deadline=deadline,
            write=checksum.extend,
        )
        expected = _expected_checksum(bytes(checksum), request.installer_name)
        with partial.open("xb") as stream:
            actual, size = _transfer(
                request.release.asset_url(request.installer_name),
                limit=MAX_INSTALLER_BYTES,
                opener=transfer,
                cancelled=cancel,
                deadline=deadline,
                write=stream.write,
                progress=progress,
            )
            stream.flush()
            os.fsync(stream.fileno())
        if progress:
            progress(DownloadProgress("verifying", size, size))
        _checkpoint(cancel, deadline)
        if actual != expected:
            raise UpdateInstallError(
                "The installer checksum did not match. The download was discarded; "
                "nothing was installed. Please try again."
            )
        _validate_target(request.target)
        # Atomic no-clobber publication, unlike replace/rename on some platforms.
        os.link(partial, final)
        created_final = True
        partial.unlink()
        _checkpoint(cancel, deadline)
        return VerifiedInstaller(
            request,
            final,
            expected,
            size,
            _file_identity(final),
            _file_identity(folder),
        )
    except Exception as exc:
        for owned in (partial, final if created_final else partial):
            try:
                owned.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            folder.rmdir()
        except OSError:
            pass
        if isinstance(exc, UpdateInstallError):
            raise
        raise UpdateInstallError(
            "Could not download the official update. Check your connection, free "
            "disk space and access to GitHub, then try again. Nothing was installed."
        ) from exc


def discard_verified_installer(verified: VerifiedInstaller) -> bool:
    """Remove only unchanged bytes and the empty private folder from this attempt.

    No broad cleanup, recursion or installation mutation. A replaced file,
    renamed folder, extra user file or redirected path is preserved.
    """
    if not isinstance(verified, VerifiedInstaller):
        return False
    path = verified.path
    try:
        if (
            path.name != verified.request.installer_name
            or not path.parent.name.startswith("vipp-update-")
            or not _normal_path(path)
            or _file_identity(path) != verified.file_identity
            or _file_identity(path.parent) != verified.directory_identity
            or path.stat().st_size != verified.size
        ):
            return False
        digest = hashlib.sha256()
        with _locked_installer(path) as stream:
            while chunk := stream.read(CHUNK_BYTES):
                digest.update(chunk)
        if digest.hexdigest() != verified.sha256:
            return False
        path.unlink()
        try:
            path.parent.rmdir()
        except OSError:
            pass  # Preserve any other files added to this folder.
        return True
    except OSError:
        return False


@contextmanager
def _locked_installer(path: Path):
    """Deny Windows writes/deletion from rehash until process creation."""
    if os.name != "nt":
        with path.open("rb") as stream:
            yield stream
        return
    import ctypes
    import msvcrt
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.CreateFileW(
        str(path), 0x80000000, 0x00000001, None, 3, 0x08000000, None
    )
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except Exception:
        kernel32.CloseHandle(handle)
        raise
    with os.fdopen(fd, "rb") as stream:
        yield stream


def launch_verified_installer(verified: VerifiedInstaller, *, popen=None) -> None:
    """Open the external guided installer; never quit or change this session."""
    _validate_request(verified.request)
    _validate_target(verified.request.target)
    path = verified.path
    if (
        path.name != verified.request.installer_name
        or not _normal_path(path)
        or not 0 < verified.size <= MAX_INSTALLER_BYTES
        or path.stat().st_size != verified.size
        or _file_identity(path) != verified.file_identity
        or _file_identity(path.parent) != verified.directory_identity
    ):
        raise UpdateInstallError("The verified installer changed. Download it again.")
    target = verified.request.target
    argv = [
        str(path),
        "--track",
        target.track,
        "--install-root",
        str(target.managed_root),
    ]
    if target.base_python is not None:
        if not _normal_path(target.base_python):
            raise UpdateInstallError(
                "The installation's base Python changed. Check again."
            )
        argv.extend(("--base-python", str(target.base_python)))
    try:
        with _locked_installer(path) as stream:
            digest = hashlib.sha256()
            while chunk := stream.read(CHUNK_BYTES):
                digest.update(chunk)
            if digest.hexdigest() != verified.sha256:
                raise UpdateInstallError(
                    "The verified installer changed. Download it again."
                )
            _validate_target(target)
            (popen or subprocess.Popen)(
                argv,
                cwd=str(path.parent),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=False,
                close_fds=True,
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )
    except OSError as exc:
        raise UpdateInstallError(
            "Windows could not open the verified installer. It may be blocked by "
            "your security policy. Keep those protections enabled and use the "
            "official installation guide. Nothing was installed by VIPP."
        ) from exc
