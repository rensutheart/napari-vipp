"""Pure, deterministic planning for a future VIPP installation executor."""

from __future__ import annotations

import importlib.metadata
import os
import re
from collections.abc import Iterable
from pathlib import Path

from napari_vipp.core.compute_policy_artifact import (
    PlatformAdmissionPolicy,
    load_phase1_compute_policy,
)
from napari_vipp.installer.models import (
    ComputeTrack,
    DiscoverySnapshot,
    InstalledPackage,
    InstallMode,
    InstallPlan,
    InstallRequest,
    IssueSeverity,
    PackageChange,
    PlanIssue,
    PlannedAction,
    ReleaseSpec,
    RollbackBoundary,
    ShortcutPlan,
    ShortcutScope,
    installation_request_fingerprint,
)

INSTALLER_DISTRIBUTION = "napari-vipp"
SUPPORTED_CPU_PYTHON_MINORS = frozenset({(3, 12), (3, 13)})
SUPPORTED_QT_BINDING = "pyqt6"
MINIMUM_NAPARI_VERSION = (0, 6)


class PlannerError(RuntimeError):
    """Unexpected planner configuration or packaged-release failure."""


def current_release_spec() -> ReleaseSpec:
    """Return the immutable version carried by the installed planner wheel."""

    try:
        version = importlib.metadata.version(INSTALLER_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError as exc:
        raise PlannerError(
            "The VIPP distribution metadata is unavailable; run the planner "
            "from an installed wheel or editable environment."
        ) from exc
    if not re.fullmatch(r"[0-9]+(?:\.[0-9A-Za-z]+)+", version):
        raise PlannerError(f"The installed VIPP version is invalid: {version!r}.")
    return ReleaseSpec(distribution=INSTALLER_DISTRIBUTION, version=version)


def create_install_plan(
    request: InstallRequest,
    *,
    discovery: DiscoverySnapshot,
    release: ReleaseSpec | None = None,
    gpu_policy: PlatformAdmissionPolicy | None = None,
) -> InstallPlan:
    """Create a plan without running commands or writing to the filesystem."""

    selected_release = release or current_release_spec()
    policy = gpu_policy or load_phase1_compute_policy().platform_admission
    issues: list[PlanIssue] = []
    _validate_request_binding(request, discovery, issues)
    _validate_request(request, issues)
    _validate_host(discovery, issues)
    _validate_python(request, discovery, issues)
    _validate_target(request, discovery, selected_release, issues)
    if request.mode is InstallMode.EXISTING and discovery.python.probe_succeeded:
        _validate_existing_environment(request, discovery, policy, issues)
    if request.track is ComputeTrack.CUDA13:
        _validate_cuda(discovery, policy, issues)

    required_free = selected_release.minimum_free_bytes(
        mode=request.mode,
        track=request.track,
    )
    _validate_disk(discovery, required_free, issues)
    _validate_shortcuts(request, discovery, issues)
    issues.extend(
        (
            PlanIssue(
                code="dependency_resolution_deferred",
                severity=IssueSeverity.INFO,
                subject="packages",
                message=(
                    "This slice records the exact top-level release requirement; "
                    "transitive dependency resolution is deferred."
                ),
                remediation=(
                    "The later executor must resolve and display concrete package "
                    "changes before Apply is enabled."
                ),
            ),
            PlanIssue(
                code="cucim_separate_installer",
                severity=IssueSeverity.INFO,
                subject="cucim",
                message="cuCIM is not included in the standard CPU/CUDA plan.",
                remediation=(
                    "Use the separate verified VIPP cuCIM local-build installer "
                    "after the CUDA environment is accepted."
                ),
            ),
        )
    )

    target_python = _target_python(request, discovery)
    shortcuts = _shortcut_plans(request, discovery, target_python)
    actions, acceptance = _planned_actions(
        request,
        discovery,
        selected_release,
        target_python,
    )
    rollback = _rollback_boundary(request, discovery, shortcuts)
    package_changes = _package_changes(request, discovery, selected_release, policy)
    return InstallPlan(
        request=request,
        release=selected_release,
        discovery=discovery,
        target_python=target_python,
        required_free_bytes=required_free,
        package_changes=package_changes,
        actions=actions,
        acceptance=acceptance,
        shortcuts=shortcuts,
        rollback=rollback,
        issues=tuple(issues),
    )


def _validate_request_binding(
    request: InstallRequest,
    discovery: DiscoverySnapshot,
    issues: list[PlanIssue],
) -> None:
    expected = installation_request_fingerprint(request)
    if discovery.request_fingerprint != expected:
        issues.append(
            PlanIssue(
                code="discovery_request_mismatch",
                severity=IssueSeverity.ERROR,
                subject="request",
                message=(
                    "The discovery snapshot was produced for different installer "
                    "inputs."
                ),
                remediation="Run discovery again for the current request.",
                details=(
                    ("expected_fingerprint", expected),
                    ("observed_fingerprint", discovery.request_fingerprint),
                ),
            )
        )


def _validate_request(request: InstallRequest, issues: list[PlanIssue]) -> None:
    if request.mode is InstallMode.EXISTING and request.install_root is not None:
        issues.append(
            PlanIssue(
                code="existing_mode_install_root_forbidden",
                severity=IssueSeverity.ERROR,
                subject="install_root",
                message=(
                    "Existing-environment mode derives its root from the selected "
                    "interpreter."
                ),
                remediation="Remove --install-root or select managed mode.",
            )
        )
    if (
        request.shortcut_scope is ShortcutScope.NONE
        and request.shortcut_directory is not None
    ):
        issues.append(
            PlanIssue(
                code="unused_shortcut_directory",
                severity=IssueSeverity.WARNING,
                subject="shortcut_directory",
                message=(
                    "A shortcut directory was supplied while shortcuts are disabled."
                ),
                remediation="Remove the directory or choose a shortcut scope.",
            )
        )
    if (
        request.shortcut_scope is ShortcutScope.BOTH
        and request.shortcut_directory is not None
    ):
        issues.append(
            PlanIssue(
                code="ambiguous_shortcut_directory",
                severity=IssueSeverity.ERROR,
                subject="shortcut_directory",
                message=(
                    "One explicit directory cannot represent both Desktop and "
                    "Start Menu shortcut locations."
                ),
                remediation=(
                    "Choose desktop or start-menu, or omit the override when "
                    "using both."
                ),
            )
        )


def _validate_host(discovery: DiscoverySnapshot, issues: list[PlanIssue]) -> None:
    host = discovery.host
    if host.sys_platform != "win32" or host.platform_system.casefold() != "windows":
        issues.append(
            PlanIssue(
                code="platform_not_yet_supported",
                severity=IssueSeverity.ERROR,
                subject="platform",
                message=(
                    "This production installer slice supports native Windows only; "
                    f"found {host.platform_system or host.sys_platform}."
                ),
                remediation=(
                    "Use the documented manual CPU installation for this platform "
                    "until its installer stage is released."
                ),
            )
        )
    if host.machine.casefold() not in {"amd64", "x86_64"}:
        issues.append(
            PlanIssue(
                code="host_architecture_unsupported",
                severity=IssueSeverity.ERROR,
                subject="platform",
                message=f"The Windows installer requires x86-64; found {host.machine}.",
                remediation="Select a 64-bit x86 Windows machine.",
            )
        )


def _validate_python(
    request: InstallRequest,
    discovery: DiscoverySnapshot,
    issues: list[PlanIssue],
) -> None:
    python = discovery.python
    if python.environment_path_unsafe:
        issues.append(
            PlanIssue(
                code="existing_environment_redirected",
                severity=IssueSeverity.ERROR,
                subject="environment",
                message=(
                    "The selected environment configuration or package path is "
                    "remote, symbolic, junctioned, or otherwise redirected."
                ),
                remediation="Choose a direct local virtual environment.",
            )
        )
        return
    if python.selected_path_remote:
        issues.append(
            PlanIssue(
                code="selected_python_remote",
                severity=IssueSeverity.ERROR,
                subject="python",
                message="A UNC or remote Python executable cannot be inspected.",
                remediation="Select a Python installation on a local volume.",
            )
        )
        return
    if python.selected_path_reparse_point:
        issues.append(
            PlanIssue(
                code="selected_python_redirected",
                severity=IssueSeverity.ERROR,
                subject="python",
                message=(
                    "The selected Python executable or one of its parents is a "
                    "symbolic link, junction, or other reparse point."
                ),
                remediation="Select a direct Python installation path.",
            )
        )
        return
    if not python.probe_succeeded or python.executable is None:
        issues.append(
            PlanIssue(
                code="python_probe_failed",
                severity=IssueSeverity.ERROR,
                subject="python",
                message=(
                    python.error or "The selected Python executable is unavailable."
                ),
                remediation="Select an installed 64-bit CPython interpreter.",
            )
        )
        return
    if python.implementation.casefold() != "cpython":
        issues.append(
            PlanIssue(
                code="python_implementation_unsupported",
                severity=IssueSeverity.ERROR,
                subject="python",
                message=(
                    "VIPP installation requires CPython; found "
                    f"{python.implementation or 'unknown'}."
                ),
                remediation="Install and select CPython.",
            )
        )
    if python.pointer_bits != 64:
        issues.append(
            PlanIssue(
                code="python_bitness_unsupported",
                severity=IssueSeverity.ERROR,
                subject="python",
                message=(
                    f"VIPP requires 64-bit Python; found {python.pointer_bits}-bit."
                ),
                remediation="Install and select a 64-bit CPython interpreter.",
            )
        )
    minor = python.version[:2]
    supported = (
        {(3, 12)}
        if request.track is ComputeTrack.CUDA13
        else SUPPORTED_CPU_PYTHON_MINORS
    )
    if minor not in supported:
        rendered = "3.12" if request.track is ComputeTrack.CUDA13 else "3.12 or 3.13"
        issues.append(
            PlanIssue(
                code="python_version_unsupported",
                severity=IssueSeverity.ERROR,
                subject="python",
                message=(
                    f"The {request.track.value} route requires Python {rendered}; "
                    f"found {python.version_text or 'unknown'}."
                ),
                remediation=f"Install and select 64-bit CPython {rendered}.",
            )
        )


def _validate_target(
    request: InstallRequest,
    discovery: DiscoverySnapshot,
    release: ReleaseSpec,
    issues: list[PlanIssue],
) -> None:
    filesystem = discovery.filesystem
    windows_managed = (
        request.mode is InstallMode.MANAGED
        and discovery.host.sys_platform == "win32"
        and discovery.host.platform_system.casefold() == "windows"
    )
    if windows_managed and filesystem.canonical_managed_root is None:
        issues.append(
            PlanIssue(
                code="managed_root_canonical_unavailable",
                severity=IssueSeverity.ERROR,
                subject="install_root",
                message=(
                    filesystem.canonical_managed_root_error
                    or "Windows did not return the canonical LocalAppData folder."
                ),
                remediation=(
                    "Restore this account's Windows Local App Data known folder "
                    "before using one-click setup. A custom managed substitute is "
                    "not accepted."
                ),
            )
        )
    elif (
        windows_managed
        and filesystem.canonical_managed_root is not None
        and not _same_path(
            filesystem.target,
            filesystem.canonical_managed_root,
        )
    ):
        issues.append(
            PlanIssue(
                code="managed_root_not_canonical",
                severity=IssueSeverity.ERROR,
                subject="install_root",
                message=(
                    "Windows one-click setup accepts only VIPP's exact canonical "
                    "per-account managed folder for this CPU/GPU option."
                ),
                remediation=(
                    "Remove the custom install-root selection and use the canonical "
                    "Windows Local App Data location."
                ),
                details=(
                    ("selected", str(filesystem.target)),
                    ("required", str(filesystem.canonical_managed_root)),
                ),
            )
        )
    if (
        request.track is ComputeTrack.CUDA13
        and discovery.host.sys_platform == "win32"
        and discovery.host.platform_system.casefold() == "windows"
        and not str(filesystem.target).isascii()
    ):
        managed = request.mode is InstallMode.MANAGED
        issues.append(
            PlanIssue(
                code="cuda13_environment_root_non_ascii",
                severity=IssueSeverity.ERROR,
                subject="install_root" if managed else "environment",
                message=(
                    "The CUDA environment location contains a non-ASCII "
                    "character. CuPy 14.1.1 cannot reliably compile CUDA kernels "
                    "from that Windows path."
                ),
                remediation=(
                    (
                        "Use the CPU one-click option on this account, or follow "
                        "the expert existing-environment instructions for a CUDA "
                        "environment whose complete path is ASCII-only."
                    )
                    if managed
                    else (
                        "Create a fresh CUDA environment in a path using standard "
                        "English letters, numbers, and punctuation. Do not move or "
                        "rename the existing virtual environment; spaces are "
                        "supported."
                    )
                ),
            )
        )
    if filesystem.target_protected:
        issues.append(
            PlanIssue(
                code="install_target_protected",
                severity=IssueSeverity.ERROR,
                subject="install_root",
                message=filesystem.target_protection_reason,
                remediation=(
                    "Restore the canonical VIPP managed-folder boundary. Expert "
                    "existing environments remain a separate workflow."
                ),
            )
        )
    if filesystem.target_reparse_point:
        issues.append(
            PlanIssue(
                code="install_target_redirected",
                severity=IssueSeverity.ERROR,
                subject="install_root",
                message=(
                    "The installation target or one of its existing parents is a "
                    "symbolic link, junction, or other reparse point."
                ),
                remediation=(
                    "Remove the redirection from the canonical managed path, or use "
                    "a direct local environment through the expert workflow."
                ),
            )
        )
    if (
        not filesystem.target_protected
        and not filesystem.nearest_existing_ancestor_is_directory
    ):
        issues.append(
            PlanIssue(
                code="install_target_parent_invalid",
                severity=IssueSeverity.ERROR,
                subject="install_root",
                message=(
                    "The nearest existing installation-target ancestor is not an "
                    "accessible directory."
                ),
                remediation=(
                    "Repair the canonical Windows Local App Data path before using "
                    "one-click setup."
                ),
                details=(
                    (
                        "nearest_existing_ancestor",
                        str(filesystem.nearest_existing_ancestor),
                    ),
                ),
            )
        )
    if request.mode is InstallMode.MANAGED:
        if filesystem.target_exists:
            ownership = filesystem.managed_ownership
            if ownership is None and filesystem.target_empty is not True:
                detail = filesystem.managed_ownership_error
                issues.append(
                    PlanIssue(
                        code="managed_target_already_exists",
                        severity=IssueSeverity.ERROR,
                        subject="install_root",
                        message=(
                            "The managed target already exists but does not have a "
                            "valid VIPP installer ownership record."
                        ),
                        remediation=(
                            "Move or remove that folder yourself only if appropriate, "
                            "then retry. One-click setup will not overwrite it or "
                            "switch to another managed root."
                        ),
                        details=(("ownership_error", detail),) if detail else (),
                    )
                )
            elif ownership is not None and (
                ownership.distribution.casefold() != release.distribution.casefold()
                or ownership.track is not request.track
            ):
                issues.append(
                    PlanIssue(
                        code="managed_target_ownership_mismatch",
                        severity=IssueSeverity.ERROR,
                        subject="install_root",
                        message=(
                            "The selected folder is owned by a different managed "
                            "VIPP installation route."
                        ),
                        remediation=(
                            "Use the location recorded for this compute option or "
                            "choose a new dedicated folder."
                        ),
                        details=(
                            ("owned_distribution", ownership.distribution),
                            ("owned_track", ownership.track.value),
                        ),
                    )
                )
    elif filesystem.target_kind != "directory":
        issues.append(
            PlanIssue(
                code="existing_environment_root_invalid",
                severity=IssueSeverity.ERROR,
                subject="environment",
                message="The selected environment root is not an existing directory.",
                remediation="Select python.exe from an existing virtual environment.",
            )
        )


def _validate_existing_environment(
    request: InstallRequest,
    discovery: DiscoverySnapshot,
    policy: PlatformAdmissionPolicy,
    issues: list[PlanIssue],
) -> None:
    python = discovery.python
    if python.pyvenv_cfg_error:
        issues.append(
            PlanIssue(
                code="pyvenv_configuration_invalid",
                severity=IssueSeverity.ERROR,
                subject="environment",
                message=python.pyvenv_cfg_error,
                remediation=(
                    "Repair the virtual environment or use managed installation."
                ),
            )
        )
    elif python.pyvenv_cfg_present and python.include_system_site_packages is not False:
        issues.append(
            PlanIssue(
                code="system_site_packages_not_supported",
                severity=IssueSeverity.ERROR,
                subject="environment",
                message=(
                    "The selected virtual environment exposes system site-packages, "
                    "so package conflicts cannot be bounded safely."
                ),
                remediation=(
                    "Choose a venv with include-system-site-packages = false or use "
                    "managed installation."
                ),
            )
        )
    if not python.is_virtual_environment or not python.pyvenv_cfg_present:
        issues.append(
            PlanIssue(
                code="existing_python_not_venv",
                severity=IssueSeverity.ERROR,
                subject="environment",
                message=(
                    "The selected interpreter must be Scripts\\python.exe inside a "
                    "virtual environment with pyvenv.cfg."
                ),
                remediation=(
                    "Select the interpreter from the intended napari venv or use "
                    "managed installation."
                ),
            )
        )
    if python.executable is not None and (
        python.base_executable is None
        or _same_path(python.executable, python.base_executable)
    ):
        issues.append(
            PlanIssue(
                code="existing_python_not_isolated",
                severity=IssueSeverity.ERROR,
                subject="environment",
                message=(
                    "The selected interpreter did not report a distinct base "
                    "interpreter and cannot be verified as an isolated venv."
                ),
                remediation=(
                    "Select Scripts\\python.exe from a normal virtual environment "
                    "or use managed installation."
                ),
            )
        )
    if python.environment_root != discovery.filesystem.target:
        issues.append(
            PlanIssue(
                code="existing_environment_prefix_mismatch",
                severity=IssueSeverity.ERROR,
                subject="environment",
                message="The selected interpreter does not belong to the target root.",
                remediation="Select the environment's own Scripts\\python.exe.",
            )
        )
    if python.executable is not None and python.environment_root is not None:
        expected_python = python.environment_root / "Scripts" / "python.exe"
        if not _same_path(python.executable, expected_python):
            issues.append(
                PlanIssue(
                    code="existing_environment_executable_mismatch",
                    severity=IssueSeverity.ERROR,
                    subject="environment",
                    message=(
                        "The probed interpreter is not the selected environment's "
                        "Scripts\\python.exe."
                    ),
                    remediation=(
                        "Select the exact python.exe inside the intended napari venv."
                    ),
                )
            )
    if python.package_probe_error:
        issues.append(
            PlanIssue(
                code="package_metadata_probe_failed",
                severity=IssueSeverity.ERROR,
                subject="packages",
                message=python.package_probe_error,
                remediation=(
                    "Repair the selected environment or use managed installation."
                ),
            )
        )
        return

    packages = _packages_by_name(python.packages)
    duplicates = tuple(
        sorted(name for name, values in packages.items() if len(values) > 1)
    )
    if duplicates:
        issues.append(
            PlanIssue(
                code="duplicate_distribution_metadata",
                severity=IssueSeverity.ERROR,
                subject="packages",
                message=(
                    "Duplicate installed distribution metadata was found for: "
                    + ", ".join(duplicates)
                ),
                remediation="Repair the environment or use managed installation.",
            )
        )
    napari_versions = packages.get("napari", ())
    if not napari_versions:
        issues.append(
            PlanIssue(
                code="napari_missing",
                severity=IssueSeverity.ERROR,
                subject="packages",
                message="The selected environment does not contain napari.",
                remediation="Choose a napari environment or use managed installation.",
            )
        )
    elif not all(
        _stable_version_at_least(package.version, MINIMUM_NAPARI_VERSION)
        for package in napari_versions
    ):
        issues.append(
            PlanIssue(
                code="napari_version_unsupported",
                severity=IssueSeverity.ERROR,
                subject="packages",
                message="The selected environment requires napari 0.6 or newer.",
                remediation="Upgrade napari separately or use managed installation.",
            )
        )

    binding_names = tuple(
        name for name in ("pyqt6", "pyside6", "pyqt5", "pyside2") if packages.get(name)
    )
    if SUPPORTED_QT_BINDING not in binding_names:
        issues.append(
            PlanIssue(
                code="pyqt6_missing",
                severity=IssueSeverity.ERROR,
                subject="packages",
                message="The selected napari environment does not contain PyQt6.",
                remediation=(
                    "Use a PyQt6 napari environment or choose managed installation."
                ),
            )
        )
    if len(binding_names) > 1:
        issues.append(
            PlanIssue(
                code="multiple_qt_bindings",
                severity=IssueSeverity.ERROR,
                subject="packages",
                message=(
                    "Multiple Qt bindings are installed: " + ", ".join(binding_names)
                ),
                remediation=(
                    "Use a clean environment with one PyQt6 binding or choose managed "
                    "installation."
                ),
            )
        )

    for package in packages.get(INSTALLER_DISTRIBUTION, ()):
        if package.editable:
            issues.append(
                PlanIssue(
                    code="editable_vipp_not_supported",
                    severity=IssueSeverity.ERROR,
                    subject="packages",
                    message=(
                        "The selected environment contains an editable VIPP install."
                    ),
                    remediation=(
                        "Remove the editable development installation or choose a "
                        "managed release environment."
                    ),
                )
            )

    if request.track is ComputeTrack.CUDA13:
        expected_cupy = _canonical_name(policy.cupy_distribution)
        cupy_packages = tuple(
            package
            for name, values in packages.items()
            if name == "cupy" or name.startswith("cupy-cuda")
            for package in values
        )
        incompatible = tuple(
            package
            for package in cupy_packages
            if package.normalized_name != expected_cupy
            or package.version not in policy.cupy_versions
        )
        if incompatible or len(cupy_packages) > 1:
            rendered = ", ".join(
                f"{package.name}=={package.version}" for package in cupy_packages
            )
            issues.append(
                PlanIssue(
                    code="cupy_environment_conflict",
                    severity=IssueSeverity.ERROR,
                    subject="packages",
                    message=(
                        "The selected environment contains a mixed or incompatible "
                        f"CuPy stack: {rendered or 'unknown'}."
                    ),
                    remediation=(
                        "Use a clean CUDA 13 environment or choose managed "
                        "installation."
                    ),
                )
            )


def _validate_cuda(
    discovery: DiscoverySnapshot,
    policy: PlatformAdmissionPolicy,
    issues: list[PlanIssue],
) -> None:
    nvidia = discovery.nvidia
    if nvidia is None or not nvidia.probe_succeeded:
        issues.append(
            PlanIssue(
                code="nvidia_driver_probe_failed",
                severity=IssueSeverity.ERROR,
                subject="gpu",
                message=(
                    nvidia.error
                    if nvidia is not None and nvidia.error
                    else "No usable NVIDIA CUDA driver was detected."
                ),
                remediation=(
                    "Install a current NVIDIA display driver, restart Windows, and "
                    "run planning again."
                ),
            )
        )
        return
    minimum_driver = int(policy.minimum_driver_version)
    driver = nvidia.driver_api_version
    if driver is None or driver < minimum_driver:
        issues.append(
            PlanIssue(
                code="nvidia_driver_too_old",
                severity=IssueSeverity.ERROR,
                subject="gpu",
                message=(
                    f"CUDA driver API {minimum_driver} or newer is required; "
                    f"found {driver if driver is not None else 'unknown'}."
                ),
                remediation="Install a current NVIDIA display driver and restart.",
                details=(
                    ("found", driver),
                    ("minimum", minimum_driver),
                ),
            )
        )
    if not nvidia.devices:
        issues.append(
            PlanIssue(
                code="nvidia_device_missing",
                severity=IssueSeverity.ERROR,
                subject="gpu",
                message="The NVIDIA driver reported no CUDA devices.",
                remediation="Use the CPU route or select a machine with an NVIDIA GPU.",
            )
        )
        return
    if not any(device.ordinal == 0 for device in nvidia.devices):
        issues.append(
            PlanIssue(
                code="nvidia_default_device_missing",
                severity=IssueSeverity.ERROR,
                subject="gpu",
                message="The NVIDIA driver report has no CUDA device ordinal 0.",
                remediation="Repair the NVIDIA driver installation and plan again.",
            )
        )
    if any(not device.name.strip() for device in nvidia.devices):
        issues.append(
            PlanIssue(
                code="nvidia_device_name_missing",
                severity=IssueSeverity.ERROR,
                subject="gpu",
                message="The NVIDIA driver returned an unnamed CUDA device.",
                remediation="Repair or update the NVIDIA display driver.",
            )
        )
    minimum_cc = _compute_capability(policy.minimum_nvidia_compute_capability)
    unsupported = tuple(
        device for device in nvidia.devices if device.compute_capability < minimum_cc
    )
    if unsupported:
        found = ", ".join(
            f"ordinal {device.ordinal}: {device.name} "
            f"(CC {device.compute_capability_text})"
            for device in unsupported
        )
        issues.append(
            PlanIssue(
                code="nvidia_compute_capability_unsupported",
                severity=IssueSeverity.ERROR,
                subject="gpu",
                message=(
                    "Every visible CUDA device must have compute capability "
                    f"{minimum_cc[0]}.{minimum_cc[1]} or newer because the runtime "
                    f"probes all devices; unsupported devices: {found}."
                ),
                remediation=(
                    "Use the CPU route or expose only Turing-class-or-newer NVIDIA "
                    "GPUs before planning."
                ),
            )
        )


def _validate_disk(
    discovery: DiscoverySnapshot,
    required_free: int,
    issues: list[PlanIssue],
) -> None:
    filesystem = discovery.filesystem
    if filesystem.target_protected:
        return
    if filesystem.free_bytes is None:
        issues.append(
            PlanIssue(
                code="disk_space_probe_failed",
                severity=IssueSeverity.ERROR,
                subject="disk",
                message=filesystem.disk_probe_error or "Free disk space is unknown.",
                remediation="Choose an accessible local installation volume.",
            )
        )
        return
    if filesystem.free_bytes < required_free:
        issues.append(
            PlanIssue(
                code="insufficient_disk_space",
                severity=IssueSeverity.ERROR,
                subject="disk",
                message="The selected volume does not have enough free space.",
                remediation="Free disk space or choose another installation directory.",
                details=(
                    ("free_bytes", filesystem.free_bytes),
                    ("required_bytes", required_free),
                    ("shortfall_bytes", required_free - filesystem.free_bytes),
                ),
            )
        )


def _validate_shortcuts(
    request: InstallRequest,
    discovery: DiscoverySnapshot,
    issues: list[PlanIssue],
) -> None:
    filesystem = discovery.filesystem
    if request.shortcut_scope in {ShortcutScope.DESKTOP, ShortcutScope.BOTH}:
        if filesystem.desktop_directory is None:
            issues.append(
                PlanIssue(
                    code="desktop_directory_unavailable",
                    severity=IssueSeverity.ERROR,
                    subject="shortcuts",
                    message="The Windows Desktop directory could not be determined.",
                    remediation="Select an explicit shortcut directory or disable it.",
                )
            )
    if request.shortcut_scope in {ShortcutScope.START_MENU, ShortcutScope.BOTH}:
        if filesystem.start_menu_directory is None:
            issues.append(
                PlanIssue(
                    code="start_menu_directory_unavailable",
                    severity=IssueSeverity.ERROR,
                    subject="shortcuts",
                    message="The Windows Start Menu directory could not be determined.",
                    remediation="Select an explicit shortcut directory or disable it.",
                )
            )
    selected_directories: list[Path | None] = []
    if request.shortcut_scope in {ShortcutScope.DESKTOP, ShortcutScope.BOTH}:
        selected_directories.append(filesystem.desktop_directory)
    if request.shortcut_scope in {ShortcutScope.START_MENU, ShortcutScope.BOTH}:
        selected_directories.append(filesystem.start_menu_directory)
    for directory in selected_directories:
        if directory is not None and directory == Path(directory.anchor):
            issues.append(
                PlanIssue(
                    code="shortcut_directory_is_root",
                    severity=IssueSeverity.ERROR,
                    subject="shortcuts",
                    message="A filesystem root cannot be a shortcut directory.",
                    remediation="Choose Desktop, Start Menu, or a dedicated directory.",
                )
            )
    if discovery.filesystem.shortcut_conflicts:
        issues.append(
            PlanIssue(
                code="shortcut_collision_unowned",
                severity=IssueSeverity.ERROR,
                subject="shortcuts",
                message=(
                    "One or more requested shortcut paths already exist without a "
                    "VIPP installer ownership record."
                ),
                remediation=(
                    "Choose another shortcut location or remove the foreign shortcut "
                    "after reviewing it."
                ),
                details=(
                    (
                        "paths",
                        [str(path) for path in discovery.filesystem.shortcut_conflicts],
                    ),
                ),
            )
        )
    if discovery.filesystem.unsafe_shortcut_directories:
        issues.append(
            PlanIssue(
                code="shortcut_directory_unsafe",
                severity=IssueSeverity.ERROR,
                subject="shortcuts",
                message=(
                    "A shortcut directory is redirected, remote, or inside "
                    "a protected system location."
                ),
                remediation="Choose a direct user-owned local directory.",
                details=(
                    (
                        "paths",
                        [
                            str(path)
                            for path in discovery.filesystem.unsafe_shortcut_directories
                        ],
                    ),
                ),
            )
        )


def _planned_actions(
    request: InstallRequest,
    discovery: DiscoverySnapshot,
    release: ReleaseSpec,
    target_python: Path,
) -> tuple[tuple[PlannedAction, ...], tuple[PlannedAction, ...]]:
    target = discovery.filesystem.target
    actions: list[PlannedAction] = []
    if request.mode is InstallMode.MANAGED:
        base_python = discovery.python.executable or request.python
        actions.append(
            PlannedAction(
                action_id="create_managed_environment",
                description="Create the dedicated VIPP virtual environment.",
                argv=(str(base_python), "-m", "venv", str(target)),
                mutation_scopes=(target,),
            )
        )
        actions.append(
            PlannedAction(
                action_id="ensure_managed_pip",
                description="Ensure pip exists inside the new environment.",
                argv=(str(target_python), "-m", "ensurepip", "--upgrade"),
                mutation_scopes=(target,),
            )
        )
    actions.append(
        PlannedAction(
            action_id="install_vipp_release",
            description="Install the exact VIPP release and selected dependency extra.",
            argv=(
                str(target_python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-input",
                "--upgrade-strategy",
                "only-if-needed",
                release.requirement(request),
            ),
            mutation_scopes=(target,),
        )
    )
    acceptance = [
        PlannedAction(
            action_id="check_dependencies",
            description="Verify installed dependency consistency.",
            argv=(str(target_python), "-m", "pip", "check"),
        ),
        PlannedAction(
            action_id="verify_vipp_release",
            description="Verify VIPP metadata and packaged launcher entry points.",
            argv=(
                str(target_python),
                "-I",
                "-B",
                "-c",
                _release_acceptance_code(release.version),
            ),
        ),
    ]
    if request.track is ComputeTrack.CUDA13:
        acceptance.append(
            PlannedAction(
                action_id="verify_cuda13",
                description="Run the released CUDA 13 compute acceptance doctor.",
                argv=(
                    str(target_python),
                    "-m",
                    "napari_vipp.core.compute_diagnostics",
                    "--track",
                    "cuda13",
                    "--json",
                    "--refresh",
                ),
            )
        )
    return tuple(actions), tuple(acceptance)


def _shortcut_plans(
    request: InstallRequest,
    discovery: DiscoverySnapshot,
    target_python: Path,
) -> tuple[ShortcutPlan, ...]:
    if request.shortcut_scope is ShortcutScope.NONE:
        return ()
    scripts = target_python.parent
    profiles = (
        (
            ("VIPP Automatic", "auto", scripts / "vipp-app.exe"),
            ("VIPP CPU", "cpu", scripts / "vipp-cpu.exe"),
            (
                "VIPP Prefer GPU",
                "prefer_gpu",
                scripts / "vipp-prefer-gpu.exe",
            ),
        )
        if request.track is ComputeTrack.CUDA13
        else (("VIPP", "cpu", scripts / "vipp-cpu.exe"),)
    )
    directories: list[Path] = []
    filesystem = discovery.filesystem
    if request.shortcut_scope in {ShortcutScope.DESKTOP, ShortcutScope.BOTH}:
        if filesystem.desktop_directory is not None:
            directories.append(filesystem.desktop_directory)
    if request.shortcut_scope in {ShortcutScope.START_MENU, ShortcutScope.BOTH}:
        if filesystem.start_menu_directory is not None:
            directories.append(filesystem.start_menu_directory)
    unique_directories = tuple(dict.fromkeys(directories))
    return tuple(
        ShortcutPlan(
            label=label,
            profile=profile,
            executable=executable,
            destination=directory / f"{label}.lnk",
        )
        for directory in unique_directories
        for label, profile, executable in profiles
    )


def _rollback_boundary(
    request: InstallRequest,
    discovery: DiscoverySnapshot,
    shortcuts: tuple[ShortcutPlan, ...],
) -> RollbackBoundary:
    shortcut_paths = tuple(shortcut.destination for shortcut in shortcuts)
    target = discovery.filesystem.target
    if request.mode is InstallMode.MANAGED:
        return RollbackBoundary(
            kind="owned-managed-environment",
            owned_paths=(target, *shortcut_paths),
            preserved_paths=(target.parent,),
            message=(
                "A later executor may roll back only the newly created environment "
                "and shortcuts recorded in its ownership manifest."
            ),
        )
    return RollbackBoundary(
        kind="existing-environment-package-snapshot-required",
        owned_paths=shortcut_paths,
        preserved_paths=(target,),
        message=(
            "The selected environment is user-owned. A later executor must capture "
            "before/after package inventories and may never delete the environment."
        ),
    )


def _package_changes(
    request: InstallRequest,
    discovery: DiscoverySnapshot,
    release: ReleaseSpec,
    policy: PlatformAdmissionPolicy,
) -> tuple[PackageChange, ...]:
    installed = (
        {}
        if request.mode is InstallMode.MANAGED
        else _packages_by_name(discovery.python.packages)
    )
    existing_vipp = _one_version(installed.get(INSTALLER_DISTRIBUTION, ()))
    disposition = (
        "install"
        if existing_vipp is None
        else "retain"
        if existing_vipp == release.version
        else "replace"
    )
    changes = [
        PackageChange(
            name=INSTALLER_DISTRIBUTION,
            installed_version=existing_vipp,
            requested=f"=={release.version}",
            disposition=disposition,
        )
    ]
    if request.track is ComputeTrack.CUDA13:
        exact = {
            "numpy": policy.numpy_versions[0],
            "scipy": policy.scipy_versions[0],
            "scikit-image": policy.scikit_image_versions[0],
            _canonical_name(policy.cupy_distribution): policy.cupy_versions[0],
        }
        for name, version in exact.items():
            current = _one_version(installed.get(name, ()))
            changes.append(
                PackageChange(
                    name=name,
                    installed_version=current,
                    requested=f"=={version}",
                    disposition=(
                        "install"
                        if current is None
                        else "retain"
                        if current == version
                        else "replace"
                    ),
                )
            )
    return tuple(changes)


def _target_python(
    request: InstallRequest,
    discovery: DiscoverySnapshot,
) -> Path:
    if request.mode is InstallMode.EXISTING:
        return discovery.python.executable or request.python
    ownership = discovery.filesystem.managed_ownership
    if ownership is not None:
        return ownership.environment_root / "Scripts" / "python.exe"
    return discovery.filesystem.target / "Scripts" / "python.exe"


def _packages_by_name(
    packages: Iterable[InstalledPackage],
) -> dict[str, tuple[InstalledPackage, ...]]:
    grouped: dict[str, list[InstalledPackage]] = {}
    for package in packages:
        grouped.setdefault(package.normalized_name, []).append(package)
    return {
        name: tuple(sorted(values, key=lambda package: package.version))
        for name, values in grouped.items()
    }


def _one_version(packages: tuple[InstalledPackage, ...]) -> str | None:
    versions = sorted({package.version for package in packages})
    return versions[0] if len(versions) == 1 else None


def _stable_version_at_least(value: str, minimum: tuple[int, int]) -> bool:
    """Fail closed to stable numeric versions without importing packaging."""

    match = re.fullmatch(
        r"\s*([0-9]+)\.([0-9]+)(?:\.([0-9]+))?(?:\.post[0-9]+)?\s*",
        value,
    )
    return bool(match and (int(match.group(1)), int(match.group(2))) >= minimum)


def _compute_capability(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"([0-9]+)\.([0-9]+)", value.strip())
    if match is None:
        raise PlannerError(
            "The packaged GPU policy has an invalid compute-capability floor."
        )
    return int(match.group(1)), int(match.group(2))


def _canonical_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value.strip().lower())


def _same_path(first: Path, second: Path) -> bool:
    return os.path.normcase(os.path.abspath(first)) == os.path.normcase(
        os.path.abspath(second)
    )


def _release_acceptance_code(version: str) -> str:
    return (
        "from importlib.metadata import distribution, version; "
        f"expected={version!r}; "
        "assert version('napari-vipp') == expected; "
        "points={(p.group,p.name) for p in distribution('napari-vipp').entry_points}; "
        "required={('gui_scripts','vipp-app'),('gui_scripts','vipp-cpu'),"
        "('gui_scripts','vipp-prefer-gpu')}; "
        "assert required <= points"
    )


__all__ = [
    "INSTALLER_DISTRIBUTION",
    "PlannerError",
    "create_install_plan",
    "current_release_spec",
]
