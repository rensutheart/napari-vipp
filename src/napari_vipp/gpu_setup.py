"""Install the running VIPP build in a separate, verified CUDA environment.

This adapter deliberately never installs packages in the running application.
The ordinary Windows installer retains ownership, rollback, cancellation and
GPU acceptance responsibilities. Its release is bound to the exact local
wheel recorded by pip, rather than a same-version package from an index.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
import re
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import url2pathname

import psutil

from napari_vipp.installer.discovery import _windows_known_folder
from napari_vipp.installer.engine import (
    CancellationToken,
    InstallCancelled,
    SubprocessCommandRunner,
    validate_build_target_identity,
)
from napari_vipp.installer.frontend import (
    BlockedAction,
    InstallationCancelled,
    InstallerSelection,
    PreparedInstall,
    ProgressUpdate,
    TargetKind,
    TrackChoice,
)
from napari_vipp.installer.models import (
    ComputeTrack,
    ReleaseSpec,
    ShortcutScope,
    managed_environment_name,
)
from napari_vipp.installer.ownership import OwnershipState, inspect_ownership
from napari_vipp.installer.payload import (
    InstallerPayloadError,
    _wheel_identity,
    bundled_release_spec,
)
from napari_vipp.installer.wheel_cache import (
    _assert_direct_path,
    cache_release_wheel,
    verify_cached_wheel,
    wheel_cache_path,
)
from napari_vipp.installer.windows_backend import WindowsInstallerBackend


class GpuSetupError(RuntimeError):
    """The running build cannot be reproduced safely by automatic GPU setup."""


def installed_wheel_release() -> ReleaseSpec:
    """Resolve and verify the exact installed wheel without contacting a server.

    Editable checkouts and ordinary index installs do not retain this artifact
    identity. They must use their original installer or a wheel installation;
    silently replacing either with a same-version PyPI build is not equivalent.
    """

    if getattr(sys, "frozen", False):
        try:
            release = bundled_release_spec(frozen=True)
        except InstallerPayloadError as exc:
            raise GpuSetupError(str(exc)) from exc
        if release.wheel_path is None or not release.wheel_sha256:
            raise GpuSetupError("The bundled VIPP wheel could not be verified.")
        return release

    try:
        distribution = metadata.distribution("napari-vipp")
    except metadata.PackageNotFoundError as exc:
        raise GpuSetupError(
            "VIPP's installation record is missing. Run VIPP from its supplied "
            "installer or workflow bundle, then retry GPU setup."
        ) from exc
    return _release_from_distribution(distribution)


def _release_from_distribution(distribution) -> ReleaseSpec:
    raw = distribution.read_text("direct_url.json")
    if not raw:
        raise GpuSetupError(
            "This installation does not retain its exact VIPP wheel. Use the "
            "original VIPP installer or workflow bundle to set up the GPU version. "
            "Automatic setup will not substitute a different build."
        )
    try:
        document = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise GpuSetupError("VIPP's wheel installation record is unreadable.") from exc
    if not isinstance(document, dict):
        raise GpuSetupError("VIPP's wheel installation record is invalid.")
    if document.get("dir_info") is not None or document.get("vcs_info") is not None:
        raise GpuSetupError(
            "This VIPP copy was installed from source or an editable checkout. "
            "Build and install its wheel first, then retry GPU setup so the "
            "same code is used in both environments."
        )
    try:
        url = urlsplit(document.get("url", ""))
        if (
            url.scheme != "file"
            or url.netloc not in {"", "localhost"}
            or url.query
            or url.fragment
            or not url.path
        ):
            raise ValueError("not a direct local wheel")
        path = Path(url2pathname(url.path))
        if not path.is_absolute() or path.suffix.lower() != ".whl":
            raise ValueError("not an absolute wheel path")
        archive = document.get("archive_info", {})
        hashes = archive.get("hashes", {})
        expected = str(hashes.get("sha256", "")).lower()
        legacy = str(archive.get("hash", ""))
        if not expected and legacy.startswith("sha256="):
            expected = legacy.removeprefix("sha256=").lower()
        if not re.fullmatch(r"[a-f0-9]{64}", expected):
            raise ValueError("missing SHA-256")
        if legacy.startswith("sha256=") and legacy[7:].lower() != expected:
            raise ValueError("conflicting SHA-256 records")
    except (AttributeError, TypeError, ValueError) as exc:
        raise GpuSetupError(
            "VIPP's installation record does not identify a hash-verified local "
            "wheel. Run VIPP from its supplied installer or workflow bundle."
        ) from exc
    observed = ""
    try:
        with path.open("rb") as stream:
            observed = hashlib.file_digest(stream, "sha256").hexdigest()
    except OSError:
        pass
    if observed != expected:
        local_app_data = _windows_known_folder("local_app_data")
        if local_app_data is not None:
            cached = wheel_cache_path(local_app_data, expected, path.name)
            if verify_cached_wheel(cached, expected):
                path, observed = cached, expected
    if not observed:
        raise GpuSetupError(
            f"The original VIPP wheel is unavailable: {path}. Restore the "
            "supplied installer/workflow bundle, then retry GPU setup."
        )
    if observed != expected:
        raise GpuSetupError(
            "The original VIPP wheel has changed since this application was "
            "installed. Restore the matching bundle or reinstall its updated "
            "wheel before retrying GPU setup."
        )
    try:
        name, version = _wheel_identity(path)
    except InstallerPayloadError as exc:
        raise GpuSetupError(str(exc)) from exc
    if name.lower().replace("_", "-").replace(".", "-") != "napari-vipp":
        raise GpuSetupError("The recorded wheel does not contain VIPP.")
    if version != distribution.version:
        raise GpuSetupError(
            "The recorded wheel version does not match this VIPP installation."
        )
    return ReleaseSpec(
        distribution="napari-vipp",
        version=version,
        wheel_path=path.resolve(),
        wheel_sha256=observed,
    )


class GpuSetupBackend(WindowsInstallerBackend):
    """A CUDA-only installer adapter that preserves the current CPU session."""

    def __init__(
        self,
        *,
        release: ReleaseSpec,
        open_gpu_callback: Callable[[Path], None] | None = None,
        **kwargs,
    ) -> None:
        if release.wheel_path is None or not release.wheel_sha256:
            raise GpuSetupError(
                "GPU setup requires the exact, hash-verified VIPP wheel."
            )
        super().__init__(
            release=release,
            shortcut_scope=ShortcutScope.NONE,
            managed_build_sha256=release.wheel_sha256,
            **kwargs,
        )
        self._open_gpu_callback = open_gpu_callback

    def inspect(
        self,
        selection: InstallerSelection,
        *,
        progress,
        cancellation,
        repair: bool = False,
    ) -> PreparedInstall:
        try:
            return self._inspect_gpu_setup(
                selection,
                progress=progress,
                cancellation=cancellation,
                repair=repair,
            )
        except InstallCancelled as exc:
            raise InstallationCancelled(details=str(exc)) from exc

    def _inspect_gpu_setup(self, selection, *, progress, cancellation, repair):
        if (
            selection.track is not TrackChoice.CUDA13
            or selection.install_root is not None
            or selection.existing_python is not None
        ):
            raise GpuSetupError(
                "GPU setup uses VIPP's separate managed CUDA 13 environment. "
                "It cannot change the running environment or install a CPU fallback."
            )
        self._assert_target_not_running()
        self._assert_build_identity()
        if self._local_app_data is not None:
            self._release = cache_release_wheel(self._release, self._local_app_data)
        prepared = super().inspect(
            selection,
            progress=progress,
            cancellation=cancellation,
            repair=repair,
        )
        if prepared.kind is TargetKind.CURRENT:
            progress(
                ProgressUpdate("verifying-gpu", "Checking the installed GPU setup…")
            )
            python = _installed_runtime(prepared.target, self._release.wheel_sha256)
            if python is None or not _gpu_check(python, cancellation):
                # Repair is still a reviewed action. No package mutation occurs
                # until the controller's explicit confirmation of this new plan.
                prepared = super().inspect(
                    selection,
                    progress=progress,
                    cancellation=cancellation,
                    repair=True,
                )
        if prepared.kind in {TargetKind.NEW, TargetKind.UPDATE, TargetKind.REPAIR}:
            prepared = replace(
                prepared,
                plain_summary=(
                    "Install this exact VIPP build with NVIDIA GPU support in "
                    "a separate environment. Your current VIPP session, open "
                    "workflows, and CPU shortcuts stay unchanged. The GPU version "
                    "will be available after its checks pass."
                ),
            )
        elif prepared.kind is TargetKind.CURRENT:
            prepared = replace(
                prepared,
                plain_summary=(
                    "This exact VIPP build is ready and its GPU checks passed. "
                    "Open VIPP with GPU to use it in a new window."
                ),
            )
        if prepared.blocked_action is BlockedAction.USE_CPU:
            prepared = replace(prepared, blocked_action=BlockedAction.RETRY)
        return prepared

    def apply(self, prepared, *, confirmed, progress, cancellation):
        if prepared.track is not ComputeTrack.CUDA13:
            raise GpuSetupError("The reviewed plan is not a GPU installation.")
        self._assert_target_not_running()
        self._assert_build_identity()
        outcome = super().apply(
            prepared,
            confirmed=confirmed,
            progress=progress,
            cancellation=cancellation,
        )
        return replace(
            outcome,
            message=(
                "GPU support passed verification. Open VIPP with GPU to use it "
                "in a new window. Your current VIPP session stays open."
            ),
        )

    def open_vipp(self, launcher: Path) -> None:
        if self._open_gpu_callback is not None:
            self._open_gpu_callback(Path(launcher))
        else:
            super().open_vipp(launcher)

    def _assert_target_not_running(self) -> None:
        if self._local_app_data is None:
            return
        root = self._managed_install_root(ComputeTrack.CUDA13).resolve()
        if Path(sys.prefix).resolve().is_relative_to(root):
            raise GpuSetupError(
                "This GPU environment is running. Close it and open your CPU "
                "VIPP session before repairing GPU support."
            )
        for process in psutil.process_iter(["exe"]):
            executable = process.info.get("exe")
            if executable and Path(executable).resolve().is_relative_to(root):
                raise GpuSetupError(
                    "Another VIPP session is using this GPU environment. Close "
                    "that session normally, then retry GPU setup here."
                )

    def _assert_build_identity(self) -> None:
        if self._local_app_data is None:
            return
        validate_build_target_identity(
            self._managed_install_root(ComputeTrack.CUDA13),
            self._release.wheel_sha256,
            state_root=self._local_app_data / "VIPP" / "installer",
        )


def _gpu_check(python: Path, cancellation) -> bool:
    result = SubprocessCommandRunner().run(
        (
            str(python),
            "-I",
            "-m",
            "napari_vipp.core.compute_diagnostics",
            "--track",
            "cuda13",
            "--json",
            "--refresh",
        ),
        cancellation=CancellationToken(cancellation),
    )
    try:
        report = json.loads(result.stdout)
    except (ValueError, TypeError):
        return False
    return (
        result.returncode == 0
        and isinstance(report, dict)
        and report.get("available") is True
    )


def _installed_runtime(root: Path, wheel_sha256: str) -> Path | None:
    inspection = inspect_ownership(root)
    if inspection.state is not OwnershipState.VALID or inspection.record is None:
        return None
    record = inspection.record
    if record.track is not ComputeTrack.CUDA13:
        return None
    packages = [p for p in record.packages if p.normalized_name == "napari-vipp"]
    if len(packages) != 1 or packages[0].sha256 != wheel_sha256:
        return None
    if packages[0].version != record.version:
        return None
    python = record.environment_root / "Scripts" / "python.exe"
    launcher = python.with_name("vipp-app.exe")
    try:
        for path in (python, launcher, record.environment_root / "pyvenv.cfg"):
            _assert_direct_path(path)
            if not path.is_file():
                return None
        marker = record.environment_root / ".vipp-install-candidate.json"
        if not verify_cached_wheel(marker, record.environment_marker_sha256):
            return None
    except (OSError, RuntimeError):
        return None
    return python


def find_installed_gpu_runtime(wheel_sha256: str) -> Path | None:
    """Find a previously accepted exact-build GPU runtime for a launch shortcut.

    This checks its ownership and artifact identity, not current GPU hardware.
    Runtime diagnostics still control GPU eligibility and CPU fallback at launch.
    """
    if not re.fullmatch(r"[a-f0-9]{64}", wheel_sha256):
        raise ValueError("An exact VIPP wheel SHA-256 is required.")
    local_app_data = _windows_known_folder("local_app_data")
    if local_app_data is None:
        return None
    root = (
        local_app_data
        / "VIPP"
        / "environments"
        / managed_environment_name(ComputeTrack.CUDA13, wheel_sha256)
    )
    return _installed_runtime(root, wheel_sha256)


def create_gpu_setup_backend(
    *,
    open_gpu_callback: Callable[[Path], None] | None = None,
) -> GpuSetupBackend:
    """Create the in-app installer, preserving the running local wheel identity."""

    if sys.platform != "win32":
        raise GpuSetupError(
            "Automatic GPU setup currently supports Windows with a compatible "
            "NVIDIA GPU. Use the platform's GPU installation instructions."
        )
    return GpuSetupBackend(
        release=installed_wheel_release(),
        preferred_python=Path(getattr(sys, "_base_executable", sys.executable)),
        open_gpu_callback=open_gpu_callback,
    )


__all__ = [
    "GpuSetupBackend",
    "GpuSetupError",
    "create_gpu_setup_backend",
    "find_installed_gpu_runtime",
    "installed_wheel_release",
]
