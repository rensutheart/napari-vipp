"""Immutable contracts for the non-mutating VIPP installation planner."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType


class InstallMode(StrEnum):
    """Whether VIPP owns a new environment or joins a selected one."""

    MANAGED = "managed"
    EXISTING = "existing"


class ComputeTrack(StrEnum):
    """Compute dependency family requested for the installation."""

    CPU = "cpu"
    CUDA13 = "cuda13"


class ShortcutScope(StrEnum):
    """Where the later executor should create launcher shortcuts."""

    NONE = "none"
    DESKTOP = "desktop"
    START_MENU = "start-menu"
    BOTH = "both"


class IssueSeverity(StrEnum):
    """Stable issue severity rendered by installer front ends."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class PlanStatus(StrEnum):
    """Top-level readiness of a reviewed installation plan."""

    READY = "ready"
    READY_WITH_WARNINGS = "ready_with_warnings"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class InstallRequest:
    """User intent before any environment or shortcut is changed."""

    mode: InstallMode
    track: ComputeTrack
    python: Path
    install_root: Path | None = None
    shortcut_scope: ShortcutScope = ShortcutScope.DESKTOP
    shortcut_directory: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", InstallMode(self.mode))
        object.__setattr__(self, "track", ComputeTrack(self.track))
        object.__setattr__(self, "shortcut_scope", ShortcutScope(self.shortcut_scope))
        object.__setattr__(self, "python", Path(self.python))
        if self.install_root is not None:
            object.__setattr__(self, "install_root", Path(self.install_root))
        if self.shortcut_directory is not None:
            object.__setattr__(
                self,
                "shortcut_directory",
                Path(self.shortcut_directory),
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode.value,
            "track": self.track.value,
            "python": str(self.python),
            "install_root": (
                str(self.install_root) if self.install_root is not None else None
            ),
            "shortcut_scope": self.shortcut_scope.value,
            "shortcut_directory": (
                str(self.shortcut_directory)
                if self.shortcut_directory is not None
                else None
            ),
        }


def installation_request_fingerprint(request: InstallRequest) -> str:
    """Return the deterministic identity bound to a discovery snapshot."""

    payload = json.dumps(
        request.as_dict(),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class ReleaseSpec:
    """Immutable release inputs bundled into an installer build."""

    distribution: str
    version: str
    managed_cpu_min_free_bytes: int = 5 * 1024**3
    managed_cuda_min_free_bytes: int = 15 * 1024**3
    existing_cpu_min_free_bytes: int = 2 * 1024**3
    existing_cuda_min_free_bytes: int = 12 * 1024**3
    wheel_path: Path | None = None
    wheel_sha256: str = ""

    def __post_init__(self) -> None:
        if self.wheel_path is not None:
            object.__setattr__(self, "wheel_path", Path(self.wheel_path))
        digest = self.wheel_sha256.strip().lower()
        if digest and (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("wheel_sha256 must be an empty string or 64 hex digits.")
        object.__setattr__(self, "wheel_sha256", digest)

    def minimum_free_bytes(
        self,
        *,
        mode: InstallMode,
        track: ComputeTrack,
    ) -> int:
        if mode is InstallMode.MANAGED:
            if track is ComputeTrack.CUDA13:
                return self.managed_cuda_min_free_bytes
            return self.managed_cpu_min_free_bytes
        if track is ComputeTrack.CUDA13:
            return self.existing_cuda_min_free_bytes
        return self.existing_cpu_min_free_bytes

    def requirement(self, request: InstallRequest) -> str:
        extras: tuple[str, ...]
        if request.mode is InstallMode.MANAGED:
            extras = (
                ("app", "gpu-cuda13")
                if request.track is ComputeTrack.CUDA13
                else ("app",)
            )
        else:
            extras = (
                ("gpu-cuda13",)
                if request.track is ComputeTrack.CUDA13
                else ()
        )
        rendered_extras = f"[{','.join(extras)}]" if extras else ""
        if self.wheel_path is not None:
            return f"{self.wheel_path}{rendered_extras}"
        return f"{self.distribution}{rendered_extras}=={self.version}"

    def as_dict(self, request: InstallRequest) -> dict[str, object]:
        return {
            "distribution": self.distribution,
            "version": self.version,
            "requirement": self.requirement(request),
            "wheel_path": (
                str(self.wheel_path) if self.wheel_path is not None else None
            ),
            "wheel_sha256": self.wheel_sha256 or None,
        }


@dataclass(frozen=True, slots=True)
class ManagedOwnershipSnapshot:
    """Validated read-only summary of one managed installation record."""

    installation_id: str
    managed_root: Path
    environment_root: Path
    distribution: str
    version: str
    track: ComputeTrack
    base_python: Path
    resolved_plan_id: str
    manifest_sha256: str
    shortcuts: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "managed_root", Path(self.managed_root))
        object.__setattr__(self, "environment_root", Path(self.environment_root))
        object.__setattr__(self, "base_python", Path(self.base_python))
        object.__setattr__(self, "track", ComputeTrack(self.track))
        object.__setattr__(
            self,
            "shortcuts",
            tuple(sorted((Path(path) for path in self.shortcuts), key=str)),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "installation_id": self.installation_id,
            "managed_root": str(self.managed_root),
            "environment_root": str(self.environment_root),
            "distribution": self.distribution,
            "version": self.version,
            "track": self.track.value,
            "base_python": str(self.base_python),
            "resolved_plan_id": self.resolved_plan_id,
            "manifest_sha256": self.manifest_sha256,
            "shortcuts": [str(path) for path in self.shortcuts],
        }


@dataclass(frozen=True, slots=True)
class HostSnapshot:
    """Read-only host identity relevant to this Windows delivery slice."""

    sys_platform: str
    platform_system: str
    machine: str

    def as_dict(self) -> dict[str, str]:
        return {
            "sys_platform": self.sys_platform,
            "platform_system": self.platform_system,
            "machine": self.machine,
        }


@dataclass(frozen=True, slots=True)
class InstalledPackage:
    """Relevant distribution metadata read without importing the package."""

    name: str
    version: str
    editable: bool = False

    @property
    def normalized_name(self) -> str:
        return self.name.strip().lower().replace("_", "-").replace(".", "-")

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "editable": self.editable,
        }


@dataclass(frozen=True, slots=True)
class PythonSnapshot:
    """Interpreter and selected-environment facts from a read-only probe."""

    requested_executable: Path
    executable: Path | None
    probe_succeeded: bool
    base_executable: Path | None = None
    selected_path_remote: bool = False
    selected_path_reparse_point: bool = False
    selected_path_invalid: bool = False
    environment_path_unsafe: bool = False
    implementation: str = ""
    version: tuple[int, int, int] = ()
    pointer_bits: int = 0
    error: str = ""
    environment_root: Path | None = None
    is_virtual_environment: bool = False
    pyvenv_cfg_present: bool = False
    include_system_site_packages: bool | None = None
    pyvenv_cfg_error: str = ""
    site_packages: Path | None = None
    packages: tuple[InstalledPackage, ...] = ()
    package_probe_error: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "requested_executable",
            Path(self.requested_executable),
        )
        if self.executable is not None:
            object.__setattr__(self, "executable", Path(self.executable))
        if self.base_executable is not None:
            object.__setattr__(self, "base_executable", Path(self.base_executable))
        if self.environment_root is not None:
            object.__setattr__(self, "environment_root", Path(self.environment_root))
        if self.site_packages is not None:
            object.__setattr__(self, "site_packages", Path(self.site_packages))
        ordered = tuple(
            sorted(
                self.packages,
                key=lambda package: (
                    package.normalized_name,
                    package.version,
                    package.name,
                ),
            )
        )
        object.__setattr__(self, "packages", ordered)

    @property
    def version_text(self) -> str:
        return ".".join(str(part) for part in self.version)

    def as_dict(self) -> dict[str, object]:
        return {
            "requested_executable": str(self.requested_executable),
            "executable": str(self.executable) if self.executable else None,
            "base_executable": (
                str(self.base_executable) if self.base_executable else None
            ),
            "selected_path_remote": self.selected_path_remote,
            "selected_path_reparse_point": self.selected_path_reparse_point,
            "selected_path_invalid": self.selected_path_invalid,
            "environment_path_unsafe": self.environment_path_unsafe,
            "probe_succeeded": self.probe_succeeded,
            "implementation": self.implementation,
            "version": self.version_text,
            "pointer_bits": self.pointer_bits,
            "error": self.error,
            "environment_root": (
                str(self.environment_root) if self.environment_root else None
            ),
            "is_virtual_environment": self.is_virtual_environment,
            "pyvenv_cfg_present": self.pyvenv_cfg_present,
            "include_system_site_packages": self.include_system_site_packages,
            "pyvenv_cfg_error": self.pyvenv_cfg_error,
            "site_packages": str(self.site_packages) if self.site_packages else None,
            "packages": [package.as_dict() for package in self.packages],
            "package_probe_error": self.package_probe_error,
        }


@dataclass(frozen=True, slots=True)
class GpuDeviceSnapshot:
    """One CUDA device reported by the NVIDIA driver API."""

    name: str
    compute_capability: tuple[int, int]
    total_memory_bytes: int | None = None
    ordinal: int = 0

    def __post_init__(self) -> None:
        if self.ordinal < 0:
            raise ValueError("CUDA device ordinals must be non-negative.")

    @property
    def compute_capability_text(self) -> str:
        return ".".join(str(part) for part in self.compute_capability)

    def as_dict(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "name": self.name,
            "device_class": "nvidia-cuda",
            "compute_capability": self.compute_capability_text,
            "total_memory_bytes": self.total_memory_bytes,
        }


@dataclass(frozen=True, slots=True)
class NvidiaSnapshot:
    """Driver-level CUDA discovery that does not require CuPy."""

    probe_succeeded: bool
    driver_api_version: int | None = None
    devices: tuple[GpuDeviceSnapshot, ...] = ()
    error: str = ""

    def __post_init__(self) -> None:
        ordinals = [device.ordinal for device in self.devices]
        if len(ordinals) != len(set(ordinals)):
            raise ValueError("CUDA device ordinals must be unique.")
        ordered = tuple(
            sorted(
                self.devices,
                key=lambda device: (
                    device.ordinal,
                    device.name.casefold(),
                    device.compute_capability,
                    device.total_memory_bytes or 0,
                ),
            )
        )
        object.__setattr__(self, "devices", ordered)

    def as_dict(self) -> dict[str, object]:
        default_ordinal = (
            0 if any(device.ordinal == 0 for device in self.devices) else None
        )
        return {
            "probe_succeeded": self.probe_succeeded,
            "driver_api_version": self.driver_api_version,
            "runtime_default_device_ordinal": default_ordinal,
            "devices": [device.as_dict() for device in self.devices],
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class FilesystemSnapshot:
    """Resolved target, shortcut, ownership, and capacity facts."""

    target: Path
    target_exists: bool
    target_kind: str
    target_empty: bool | None
    target_reparse_point: bool
    target_protected: bool
    target_protection_reason: str
    nearest_existing_ancestor: Path | None
    nearest_existing_ancestor_is_directory: bool
    free_bytes: int | None
    disk_probe_error: str
    desktop_directory: Path | None
    start_menu_directory: Path | None
    shortcut_conflicts: tuple[Path, ...] = ()
    unsafe_shortcut_directories: tuple[Path, ...] = ()
    managed_ownership: ManagedOwnershipSnapshot | None = None
    managed_ownership_error: str = ""
    ownership_manifest_exists: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "target", Path(self.target))
        if self.nearest_existing_ancestor is not None:
            object.__setattr__(
                self,
                "nearest_existing_ancestor",
                Path(self.nearest_existing_ancestor),
            )
        if self.desktop_directory is not None:
            object.__setattr__(
                self,
                "desktop_directory",
                Path(self.desktop_directory),
            )
        if self.start_menu_directory is not None:
            object.__setattr__(
                self,
                "start_menu_directory",
                Path(self.start_menu_directory),
            )
        object.__setattr__(
            self,
            "shortcut_conflicts",
            tuple(sorted((Path(path) for path in self.shortcut_conflicts), key=str)),
        )
        object.__setattr__(
            self,
            "unsafe_shortcut_directories",
            tuple(
                sorted(
                    (Path(path) for path in self.unsafe_shortcut_directories),
                    key=str,
                )
            ),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "target": str(self.target),
            "target_exists": self.target_exists,
            "target_kind": self.target_kind,
            "target_empty": self.target_empty,
            "target_reparse_point": self.target_reparse_point,
            "target_protected": self.target_protected,
            "target_protection_reason": self.target_protection_reason,
            "nearest_existing_ancestor": (
                str(self.nearest_existing_ancestor)
                if self.nearest_existing_ancestor is not None
                else None
            ),
            "nearest_existing_ancestor_is_directory": (
                self.nearest_existing_ancestor_is_directory
            ),
            "free_bytes": self.free_bytes,
            "disk_probe_error": self.disk_probe_error,
            "desktop_directory": (
                str(self.desktop_directory) if self.desktop_directory else None
            ),
            "start_menu_directory": (
                str(self.start_menu_directory)
                if self.start_menu_directory
                else None
            ),
            "shortcut_conflicts": [str(path) for path in self.shortcut_conflicts],
            "unsafe_shortcut_directories": [
                str(path) for path in self.unsafe_shortcut_directories
            ],
            "managed_ownership": (
                self.managed_ownership.as_dict()
                if self.managed_ownership is not None
                else None
            ),
            "managed_ownership_error": self.managed_ownership_error,
            "ownership_manifest_exists": self.ownership_manifest_exists,
        }


@dataclass(frozen=True, slots=True)
class DiscoverySnapshot:
    """Complete read-only input consumed by the pure planner."""

    request_fingerprint: str
    host: HostSnapshot
    python: PythonSnapshot
    filesystem: FilesystemSnapshot
    nvidia: NvidiaSnapshot | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "request_fingerprint": self.request_fingerprint,
            "host": self.host.as_dict(),
            "python": self.python.as_dict(),
            "filesystem": self.filesystem.as_dict(),
            "nvidia": self.nvidia.as_dict() if self.nvidia else None,
        }


def _freeze_json_value(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        frozen = {
            str(key): _freeze_json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json_value(item) for item in value)
    raise TypeError(
        f"Plan issue details must contain JSON values, not {type(value)!r}."
    )


def _copy_json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _copy_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_copy_json_value(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class PlanIssue:
    """One stable, actionable result of planning validation."""

    code: str
    severity: IssueSeverity
    subject: str
    message: str
    remediation: str = ""
    details: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "severity", IssueSeverity(self.severity))
        object.__setattr__(
            self,
            "details",
            tuple(
                (key, _freeze_json_value(value))
                for key, value in sorted(self.details, key=lambda item: item[0])
            ),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "subject": self.subject,
            "message": self.message,
            "remediation": self.remediation,
            "details": {
                key: _copy_json_value(value) for key, value in self.details
            },
        }


@dataclass(frozen=True, slots=True)
class PlannedAction:
    """A future executor action; the 2A planner never runs it."""

    action_id: str
    description: str
    argv: tuple[str, ...]
    mutation_scopes: tuple[Path, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.action_id,
            "description": self.description,
            "argv": list(self.argv),
            "mutation_scopes": [str(path) for path in self.mutation_scopes],
        }


@dataclass(frozen=True, slots=True)
class ShortcutPlan:
    """One shortcut that a later platform front end may create."""

    label: str
    profile: str
    executable: Path
    destination: Path

    def as_dict(self) -> dict[str, str]:
        return {
            "label": self.label,
            "profile": self.profile,
            "executable": str(self.executable),
            "destination": str(self.destination),
        }


@dataclass(frozen=True, slots=True)
class PackageChange:
    """Top-level package intent without claiming resolver output."""

    name: str
    installed_version: str | None
    requested: str
    disposition: str

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "installed_version": self.installed_version,
            "requested": self.requested,
            "disposition": self.disposition,
        }


@dataclass(frozen=True, slots=True)
class RollbackBoundary:
    """Ownership boundary that the future executor must preserve."""

    kind: str
    owned_paths: tuple[Path, ...]
    preserved_paths: tuple[Path, ...]
    message: str

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "owned_paths": [str(path) for path in self.owned_paths],
            "preserved_paths": [str(path) for path in self.preserved_paths],
            "message": self.message,
        }


_SEVERITY_ORDER = {
    IssueSeverity.ERROR: 0,
    IssueSeverity.WARNING: 1,
    IssueSeverity.INFO: 2,
}


@dataclass(frozen=True, slots=True)
class InstallPlan:
    """Stable, reviewable plan produced without filesystem mutation."""

    request: InstallRequest
    release: ReleaseSpec
    discovery: DiscoverySnapshot
    target_python: Path
    required_free_bytes: int
    package_changes: tuple[PackageChange, ...]
    actions: tuple[PlannedAction, ...]
    acceptance: tuple[PlannedAction, ...]
    shortcuts: tuple[ShortcutPlan, ...]
    rollback: RollbackBoundary
    issues: tuple[PlanIssue, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        ordered_issues = tuple(
            sorted(
                self.issues,
                key=lambda issue: (
                    _SEVERITY_ORDER[issue.severity],
                    issue.code,
                    issue.subject,
                    issue.message,
                ),
            )
        )
        object.__setattr__(self, "issues", ordered_issues)
        object.__setattr__(
            self,
            "package_changes",
            tuple(sorted(self.package_changes, key=lambda change: change.name)),
        )
        object.__setattr__(
            self,
            "shortcuts",
            tuple(sorted(self.shortcuts, key=lambda item: str(item.destination))),
        )

    @property
    def ready(self) -> bool:
        return not any(
            issue.severity is IssueSeverity.ERROR for issue in self.issues
        )

    @property
    def status(self) -> PlanStatus:
        if not self.ready:
            return PlanStatus.BLOCKED
        if any(issue.severity is IssueSeverity.WARNING for issue in self.issues):
            return PlanStatus.READY_WITH_WARNINGS
        return PlanStatus.READY

    def as_dict(self) -> dict[str, object]:
        free = self.discovery.filesystem.free_bytes
        shortfall = (
            max(0, self.required_free_bytes - free) if free is not None else None
        )
        return {
            "schema": "napari-vipp-install-plan",
            "schema_version": 1,
            "plan_only": True,
            "mutation_performed": False,
            "status": self.status.value,
            "ready": self.ready,
            "ready_for_resolution": self.ready,
            "resolution_required": True,
            "ready_for_apply": False,
            "execution_authorized": False,
            "request": self.request.as_dict(),
            "release": self.release.as_dict(self.request),
            "discovery": self.discovery.as_dict(),
            "target": {
                "environment_root": str(self.discovery.filesystem.target),
                "python": str(self.target_python),
            },
            "package_plan": {
                "resolution": "deferred",
                "requirement": self.release.requirement(self.request),
                "changes": [change.as_dict() for change in self.package_changes],
                "note": (
                    "Transitive dependency resolution occurs only after the user "
                    "reviews this plan and a later executor is authorized."
                ),
            },
            "disk": {
                "required_free_bytes": self.required_free_bytes,
                "observed_free_bytes": free,
                "shortfall_bytes": shortfall,
            },
            "cucim": {
                "included": False,
                "note": (
                    "cuCIM remains an optional separate Windows local-build "
                    "installer."
                ),
            },
            "shortcuts": [shortcut.as_dict() for shortcut in self.shortcuts],
            "rollback": self.rollback.as_dict(),
            "proposed_actions": [action.as_dict() for action in self.actions],
            "acceptance": [action.as_dict() for action in self.acceptance],
            "issues": [issue.as_dict() for issue in self.issues],
        }

    def to_json(self) -> str:
        """Return canonical, human-readable JSON with no volatile fields."""

        return (
            json.dumps(
                self.as_dict(),
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )


__all__ = [
    "ComputeTrack",
    "DiscoverySnapshot",
    "FilesystemSnapshot",
    "GpuDeviceSnapshot",
    "HostSnapshot",
    "InstallMode",
    "InstallPlan",
    "InstallRequest",
    "InstalledPackage",
    "IssueSeverity",
    "NvidiaSnapshot",
    "ManagedOwnershipSnapshot",
    "PackageChange",
    "PlanIssue",
    "PlannedAction",
    "PlanStatus",
    "PythonSnapshot",
    "ReleaseSpec",
    "RollbackBoundary",
    "ShortcutPlan",
    "ShortcutScope",
    "installation_request_fingerprint",
]
