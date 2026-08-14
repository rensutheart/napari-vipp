"""Headless dependency resolution and transactional managed installation.

Resolution is read-only with respect to the selected installation target.  An
apply requires a one-use authorization bound to the complete normalized pip
report, the target ownership snapshot, and any bundled wheel digest.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol
from urllib.parse import unquote, urlsplit, urlunsplit
from urllib.request import url2pathname

from napari_vipp.installer.models import (
    ComputeTrack,
    InstallMode,
    InstallPlan,
    PlannedAction,
    ReleaseSpec,
)
from napari_vipp.installer.ownership import (
    OWNERSHIP_DIRECTORY,
    OwnedEnvironment,
    OwnedPackage,
    OwnedShortcut,
    OwnershipRecord,
    OwnershipState,
    inspect_ownership,
    managed_environments_root,
    ownership_path,
    write_ownership_record,
)

_HEX_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_PACKAGE_NAME = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?\Z")
_SAFE_PATH_COMPONENT = re.compile(r"[^A-Za-z0-9._-]+")
_CANDIDATE_MARKER = ".vipp-install-candidate.json"
_FILE_ATTRIBUTE_REPARSE_POINT = 0x0400
_RESOLVED_PLAN_FILENAME = "resolved-plan.json"
_RESULT_FILENAME = "result.json"
_LOG_FILENAME = "install.jsonl"
_DEFAULT_INDEX_URL = "https://pypi.org/simple"
_DEFAULT_ARTIFACT_HOSTS = frozenset({"files.pythonhosted.org", "pypi.org"})
# pip's default socket timeout is short enough for a busy university connection
# to abandon otherwise healthy large wheels.  Keep the policy explicit and
# bounded in argv so it also applies under ``--isolated`` and cannot be changed
# by an inherited pip configuration.  A timeout is the maximum *idle socket*
# interval, not a limit on the duration of a complete download.
_PIP_NETWORK_TIMEOUT_SECONDS = 120
_PIP_NETWORK_RETRIES = 8
_CPU_RESOLUTION_TEMP_MIN_FREE_BYTES = 1024**3
_CUDA_RESOLUTION_TEMP_MIN_FREE_BYTES = 5 * 1024**3
_MAX_RUN_DIRECTORIES = 25
_MAX_STALE_LOCKS = 10
_WINDOWS_TRANSIENT_REPLACE_ERRORS = frozenset({5, 32, 33})
_ATOMIC_REPLACE_RETRY_DELAYS = (0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.0, 1.0)
_SHORTCUT_SCRIPT = r"""param(
    [Parameter(Mandatory=$true)][string]$Destination,
    [Parameter(Mandatory=$true)][string]$Target,
    [Parameter(Mandatory=$true)][string]$WorkingDirectory,
    [Parameter(Mandatory=$true)][string]$Description
)
$ErrorActionPreference = "Stop"
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($Destination)
$shortcut.TargetPath = $Target
$shortcut.WorkingDirectory = $WorkingDirectory
$shortcut.Description = $Description
$shortcut.Save()
if (-not (Test-Path -LiteralPath $Destination -PathType Leaf)) {
    throw "Windows did not create the requested VIPP shortcut."
}
"""


class InstallerEngineError(RuntimeError):
    """Base class for stable headless-installer failures."""


class PreparationError(InstallerEngineError):
    """The selected plan could not safely reach package review."""


class ResolutionError(PreparationError):
    """pip did not produce one complete, hash-bound candidate set."""


class AuthorizationError(InstallerEngineError):
    """Apply was attempted without a matching explicit authorization."""


class StalePreparedTransaction(InstallerEngineError):
    """The wheel, plan, or target changed after it was reviewed."""


class ConcurrentInstallationError(InstallerEngineError):
    """Another installer transaction currently owns the selected target."""


class RecoveryError(InstallerEngineError):
    """An interrupted transaction could not be recovered without guessing."""


class InstallCancelled(InstallerEngineError):
    """Cooperative cancellation observed at a safe transaction checkpoint."""


class CommandFailed(InstallerEngineError):
    """A child process returned a non-zero status."""

    def __init__(self, argv: Sequence[str], result: CommandResult):
        self.argv = tuple(argv)
        self.result = result
        detail = _redact_text(result.stderr.strip() or result.stdout.strip())
        if len(detail) > 500:
            detail = detail[-500:]
        super().__init__(
            f"Command exited with status {result.returncode}"
            + (f": {detail}" if detail else ".")
        )


class ManagedTargetKind(StrEnum):
    """Plain install choice inferred from the ownership and release state."""

    NEW = "new"
    UPDATE = "update"
    CURRENT = "current"
    REPAIR = "repair"
    NEWER = "newer"
    FOREIGN = "foreign"


_ACTION_LABELS = {
    ManagedTargetKind.NEW: "Install VIPP",
    ManagedTargetKind.UPDATE: "Update VIPP",
    ManagedTargetKind.CURRENT: "Open VIPP",
    ManagedTargetKind.REPAIR: "Repair VIPP",
    ManagedTargetKind.NEWER: "Keep newer VIPP",
    ManagedTargetKind.FOREIGN: "Choose another folder",
}


@dataclass(frozen=True, slots=True)
class ManagedTargetInspection:
    """Non-mutating classification consumed directly by installer front ends."""

    kind: ManagedTargetKind
    target: Path
    desired_version: str
    current_version: str | None
    reason: str
    fingerprint: str
    target_preexisting: bool
    record: OwnershipRecord | None = field(default=None, repr=False)

    @property
    def action_label(self) -> str:
        return _ACTION_LABELS[self.kind]

    @property
    def can_apply(self) -> bool:
        return self.kind in {
            ManagedTargetKind.NEW,
            ManagedTargetKind.UPDATE,
            ManagedTargetKind.REPAIR,
        }

    @property
    def installed_version(self) -> str | None:
        return self.current_version

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "target": str(self.target),
            "desired_version": self.desired_version,
            "current_version": self.current_version,
            "reason": self.reason,
            "fingerprint": self.fingerprint,
            "target_preexisting": self.target_preexisting,
            "action_label": self.action_label,
            "can_apply": self.can_apply,
        }


class PackageDisposition(StrEnum):
    INSTALL = "install"
    KEEP = "keep"
    UPGRADE = "upgrade"
    DOWNGRADE = "downgrade"
    REINSTALL = "reinstall"
    REMOVE = "remove"


@dataclass(frozen=True, slots=True)
class ResolvedPackage:
    """One exact artifact selected by pip's non-mutating resolver report."""

    name: str
    version: str
    source_url: str
    sha256: str
    requested: bool = False

    def __post_init__(self) -> None:
        name = self.name.strip()
        version = self.version.strip()
        digest = self.sha256.strip().lower()
        if not _PACKAGE_NAME.fullmatch(name):
            raise ValueError(f"Invalid resolved package name: {name!r}.")
        if not version:
            raise ValueError("Resolved package versions cannot be empty.")
        if "\n" in self.source_url or "\r" in self.source_url:
            raise ValueError("Resolved package URLs cannot contain newlines.")
        if not _HEX_DIGEST.fullmatch(digest):
            raise ValueError("Every resolved artifact must have a SHA-256 digest.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "sha256", digest)

    @property
    def normalized_name(self) -> str:
        return _normalize_name(self.name)

    def as_dict(self) -> dict[str, object]:
        source = _public_source_identity(self.source_url)
        return {
            "name": self.name,
            "version": self.version,
            "source": source,
            "sha256": self.sha256,
            "requested": self.requested,
        }


@dataclass(frozen=True, slots=True)
class PackageReviewChange:
    """Plain before/after package change shown during confirmation."""

    name: str
    installed_version: str | None
    resolved_version: str | None
    disposition: PackageDisposition

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "installed_version": self.installed_version,
            "resolved_version": self.resolved_version,
            "disposition": self.disposition.value,
        }


@dataclass(frozen=True, slots=True)
class ReviewSummary:
    """Small nontechnical summary; the full package set remains available."""

    headline: str
    detail: str
    packages_to_install: int
    packages_to_change: int
    packages_to_remove: int

    def as_dict(self) -> dict[str, object]:
        return {
            "headline": self.headline,
            "detail": self.detail,
            "packages_to_install": self.packages_to_install,
            "packages_to_change": self.packages_to_change,
            "packages_to_remove": self.packages_to_remove,
        }


@dataclass(frozen=True, slots=True)
class PreparedShortcut:
    """One reviewed shortcut destination bound to its prior exact bytes."""

    label: str
    profile: str
    destination: Path
    existed: bool
    prior_sha256: str = ""
    remove: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "destination", Path(self.destination))
        if self.remove and (not self.existed or not self.prior_sha256):
            raise ValueError("Shortcut removal requires an exact existing baseline.")

    def as_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "profile": self.profile,
            "destination": str(self.destination),
            "existed": self.existed,
            "prior_sha256": self.prior_sha256 or None,
            "remove": self.remove,
        }


@dataclass(slots=True)
class _ShortcutMutation:
    prepared: PreparedShortcut
    temporary: Path
    backup: Path | None
    new_sha256: str
    committed: bool = False


@dataclass(slots=True)
class _SetupMutation:
    destination: Path
    temporary: Path
    backup: Path | None
    prior_sha256: str
    new_sha256: str
    committed: bool = False


@dataclass(frozen=True, slots=True)
class PreparedTransaction:
    """Immutable review object bound to the target and exact artifact set."""

    plan: InstallPlan
    target_inspection: ManagedTargetInspection
    packages: tuple[ResolvedPackage, ...]
    changes: tuple[PackageReviewChange, ...]
    shortcuts: tuple[PreparedShortcut, ...]
    review_summary: ReviewSummary
    resolution_id: str
    plan_fingerprint: str
    wheel_sha256: str
    index_url: str
    persistent_setup_path: Path | None
    persistent_setup_sha256: str
    persistent_setup_prior_sha256: str
    working_directory: Path
    resolution_complete: bool
    applicable: bool
    blocking_reason: str = ""

    @property
    def action_label(self) -> str:
        return self.target_inspection.action_label

    @property
    def installed_version(self) -> str | None:
        return self.target_inspection.current_version

    @property
    def target(self) -> Path:
        return self.target_inspection.target

    @property
    def technical_details(self) -> tuple[ResolvedPackage, ...]:
        return self.packages

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": "napari-vipp-resolved-install-plan",
            "schema_version": 1,
            "resolution_id": self.resolution_id,
            "plan_fingerprint": self.plan_fingerprint,
            "wheel_sha256": self.wheel_sha256 or None,
            "index": _public_source_identity(self.index_url),
            "persistent_setup": (
                {
                    "path": str(self.persistent_setup_path),
                    "sha256": self.persistent_setup_sha256,
                    "prior_sha256": self.persistent_setup_prior_sha256 or None,
                }
                if self.persistent_setup_path is not None
                else None
            ),
            "working_directory": str(self.working_directory),
            "resolution_complete": self.resolution_complete,
            "applicable": self.applicable,
            "blocking_reason": self.blocking_reason,
            "action_label": self.action_label,
            "target_inspection": self.target_inspection.as_dict(),
            "review_summary": self.review_summary.as_dict(),
            "packages": [package.as_dict() for package in self.packages],
            "changes": [change.as_dict() for change in self.changes],
            "shortcuts": [shortcut.as_dict() for shortcut in self.shortcuts],
            "release": self.plan.release.as_dict(self.plan.request),
        }


@dataclass(frozen=True, slots=True)
class ExecutionAuthorization:
    """Opaque, one-use proof that the reviewed resolution was confirmed."""

    authorization_id: str
    resolution_id: str
    confirmation_label: str


class ProgressStage(StrEnum):
    INSPECTING = "inspecting"
    RESOLVING = "resolving"
    READY = "ready"
    PREPARING = "preparing"
    CREATING_ENVIRONMENT = "creating_environment"
    INSTALLING = "installing"
    VERIFYING = "verifying"
    CREATING_SHORTCUTS = "creating_shortcuts"
    COMMITTING = "committing"
    ROLLING_BACK = "rolling_back"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class InstallProgress:
    """One immutable progress event suitable for GUI or JSON front ends."""

    stage: ProgressStage
    message: str
    completed: int
    total: int


ProgressCallback = Callable[[InstallProgress], None]
CancellationSource = "CancellationToken | Callable[[], bool] | object | None"


class CancellationToken:
    """Thread-safe cancellation token shared by prepare and apply callers."""

    def __init__(self, event: threading.Event | None = None) -> None:
        self._event = event or threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def is_set(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled():
            raise InstallCancelled("The VIPP installation was cancelled.")


@dataclass(frozen=True, slots=True)
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


class CommandRunner(Protocol):
    def run(
        self,
        argv: Sequence[str],
        *,
        cancellation: object | None = None,
        env: Mapping[str, str] | None = None,
        cwd: Path | None = None,
    ) -> CommandResult: ...


class _RegistryPlanLike(Protocol):
    key: str
    values: tuple[tuple[str, str | int], ...]
    managed_root: Path
    installation_id: str
    manifest_sha256: str
    uninstaller_path: Path
    uninstaller_sha256: str


class SubprocessCommandRunner:
    """Cancellation-aware argv-only subprocess runner (never invokes a shell)."""

    def run(
        self,
        argv: Sequence[str],
        *,
        cancellation: object | None = None,
        env: Mapping[str, str] | None = None,
        cwd: Path | None = None,
    ) -> CommandResult:
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            creationflags |= getattr(subprocess, "CREATE_SUSPENDED", 0x00000004)
            creationflags |= getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        process = subprocess.Popen(
            tuple(str(value) for value in argv),
            cwd=str(cwd) if cwd is not None else None,
            env=dict(env) if env is not None else None,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=creationflags,
        )
        job: _WindowsKillOnCloseJob | None = None
        try:
            if os.name == "nt":
                try:
                    job = _WindowsKillOnCloseJob(process)
                    _resume_windows_process(process.pid)
                except Exception:
                    _cancel_process(process)
                    raise
            while True:
                try:
                    stdout, stderr = process.communicate(timeout=0.2)
                    return CommandResult(process.returncode, stdout, stderr)
                except subprocess.TimeoutExpired:
                    if _is_cancelled(cancellation):
                        if job is not None:
                            job.close()
                        else:
                            _cancel_process(process)
                        stdout, stderr = process.communicate()
                        raise InstallCancelled(
                            "The VIPP installation was cancelled."
                        ) from _CancelledCommandOutput(stdout, stderr)
        finally:
            if job is not None:
                job.close()


class _CancelledCommandOutput(Exception):
    def __init__(self, stdout: str, stderr: str) -> None:
        self.stdout = stdout
        self.stderr = stderr


class InstallStatus(StrEnum):
    SUCCEEDED = "succeeded"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class RollbackReport:
    attempted: bool
    completed: bool
    removed_paths: tuple[Path, ...] = ()
    preserved_paths: tuple[Path, ...] = ()
    errors: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "attempted": self.attempted,
            "completed": self.completed,
            "removed_paths": [str(path) for path in self.removed_paths],
            "preserved_paths": [str(path) for path in self.preserved_paths],
            "errors": list(self.errors),
        }


@dataclass(frozen=True, slots=True)
class InstallResult:
    """Terminal apply outcome with a retained log and launchable environment."""

    status: InstallStatus
    resolution_id: str
    operation: ManagedTargetKind
    managed_root: Path
    environment_root: Path | None
    launcher_path: Path | None
    log_path: Path
    ownership_record_path: Path | None
    rollback: RollbackReport
    message: str
    technical_error: str = ""
    retirement_cleanup: RollbackReport = field(
        default_factory=lambda: RollbackReport(False, True)
    )
    registration_warning: str = ""

    @property
    def succeeded(self) -> bool:
        return self.status is InstallStatus.SUCCEEDED

    @property
    def cancelled(self) -> bool:
        return self.status is InstallStatus.CANCELLED

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": "napari-vipp-install-result",
            "schema_version": 1,
            "status": self.status.value,
            "resolution_id": self.resolution_id,
            "operation": self.operation.value,
            "managed_root": str(self.managed_root),
            "environment_root": (
                str(self.environment_root) if self.environment_root else None
            ),
            "launcher_path": str(self.launcher_path) if self.launcher_path else None,
            "log_path": str(self.log_path),
            "ownership_record_path": (
                str(self.ownership_record_path) if self.ownership_record_path else None
            ),
            "rollback": self.rollback.as_dict(),
            "retirement_cleanup": self.retirement_cleanup.as_dict(),
            "message": self.message,
            "technical_error": self.technical_error,
            "registration_warning": self.registration_warning,
        }


def inspect_managed_target(
    target: str | Path,
    *,
    release: ReleaseSpec,
    track: ComputeTrack,
) -> ManagedTargetInspection:
    """Classify new, update, repair, newer, or foreign managed targets."""

    selected = Path(target)
    try:
        metadata = selected.lstat()
    except FileNotFoundError:
        return _inspection(
            ManagedTargetKind.NEW,
            selected,
            release,
            None,
            "VIPP is not installed in this location yet.",
        )
    except OSError as exc:
        return _inspection(
            ManagedTargetKind.FOREIGN,
            selected,
            release,
            None,
            f"The selected folder cannot be inspected safely: {exc}",
            target_preexisting=True,
        )
    if (
        not selected.is_dir()
        or selected.is_symlink()
        or _metadata_is_reparse(metadata)
        or _path_has_reparse_component(selected)
    ):
        return _inspection(
            ManagedTargetKind.FOREIGN,
            selected,
            release,
            None,
            "The selected location is not a normal local folder.",
            target_preexisting=True,
            extra=(metadata.st_size, metadata.st_mtime_ns),
        )
    ownership = inspect_ownership(selected)
    if ownership.state is not OwnershipState.VALID or ownership.record is None:
        if ownership.state is OwnershipState.ABSENT:
            try:
                empty = next(selected.iterdir(), None) is None or (
                    _is_empty_managed_residue(selected)
                )
            except OSError:
                empty = False
            if empty:
                return _inspection(
                    ManagedTargetKind.NEW,
                    selected,
                    release,
                    None,
                    "This empty folder is ready for a new VIPP installation.",
                    target_preexisting=True,
                    extra=("empty-existing-directory",),
                )
        reason = ownership.error or (
            "This folder was not created by the VIPP installer, so it will not be "
            "overwritten."
        )
        return _inspection(
            ManagedTargetKind.FOREIGN,
            selected,
            release,
            None,
            reason,
            target_preexisting=True,
            extra=(ownership.state.value,),
        )
    record = ownership.record
    if _normalize_name(record.distribution) != _normalize_name(
        release.distribution
    ) or record.track is not ComputeTrack(track):
        return _inspection(
            ManagedTargetKind.FOREIGN,
            selected,
            release,
            record.version,
            "This location belongs to a different VIPP installation option.",
            target_preexisting=True,
            record=record,
            extra=(ownership.manifest_sha256,),
        )
    if _path_has_reparse_component(record.environment_root) or (
        record.environment_root.is_dir() and _tree_has_reparse(record.environment_root)
    ):
        return _inspection(
            ManagedTargetKind.FOREIGN,
            selected,
            release,
            record.version,
            "The managed VIPP environment is redirected and cannot be trusted.",
            target_preexisting=True,
            record=record,
            extra=(ownership.manifest_sha256, "reparse-environment"),
        )
    comparison = _compare_versions(record.version, release.version)
    structural_reason = _structural_health_error(record)
    if comparison < 0:
        kind = ManagedTargetKind.UPDATE
        reason = (
            f"VIPP {record.version} is installed and can be updated to "
            f"{release.version}."
        )
    elif comparison > 0:
        kind = ManagedTargetKind.NEWER
        reason = (
            f"VIPP {record.version} is newer than this installer "
            f"({release.version}); it will not be downgraded."
        )
    elif structural_reason:
        kind = ManagedTargetKind.REPAIR
        reason = f"VIPP needs repair: {structural_reason}"
    else:
        kind = ManagedTargetKind.CURRENT
        reason = f"VIPP {release.version} is already installed and ready to open."
    return _inspection(
        kind,
        selected,
        release,
        record.version,
        reason,
        target_preexisting=True,
        record=record,
        extra=(ownership.manifest_sha256, structural_reason),
    )


class ManagedInstallerEngine:
    """Prepare, authorize, and apply managed installation transactions."""

    def __init__(
        self,
        *,
        runner: CommandRunner | None = None,
        state_root: str | Path | None = None,
        now: Callable[[], datetime] | None = None,
        identifier: Callable[[], str] | None = None,
        index_url: str = _DEFAULT_INDEX_URL,
        approved_artifact_hosts: Sequence[str] = tuple(_DEFAULT_ARTIFACT_HOSTS),
        require_embedded_wheel: bool | None = None,
        setup_source: str | Path | None = None,
        persistent_setup_path: str | Path | None = None,
        registry_backend: object | None = None,
        known_folder_probe: Callable[[str], Path | None] | None = None,
    ) -> None:
        self._runner = runner or SubprocessCommandRunner()
        self._state_root = Path(state_root) if state_root is not None else None
        self._now = now or (lambda: datetime.now(UTC))
        self._identifier = identifier or (lambda: str(uuid.uuid4()))
        self._approved_artifact_hosts = frozenset(
            host.strip().casefold() for host in approved_artifact_hosts if host.strip()
        )
        if not self._approved_artifact_hosts:
            raise ValueError("At least one approved artifact host is required.")
        self._index_url = _validated_index_url(
            index_url,
            approved_hosts=self._approved_artifact_hosts,
        )
        self._require_embedded_wheel = (
            bool(getattr(sys, "frozen", False))
            if require_embedded_wheel is None
            else bool(require_embedded_wheel)
        )
        frozen = bool(getattr(sys, "frozen", False))
        source = (
            Path(sys.executable) if setup_source is None and frozen else setup_source
        )
        self._setup_source = (
            Path(os.path.abspath(source)) if source is not None else None
        )
        if persistent_setup_path is not None:
            self._persistent_setup_path = Path(persistent_setup_path)
        else:
            self._persistent_setup_path = None
        if self._setup_source is None and persistent_setup_path is not None:
            raise ValueError("persistent_setup_path requires a setup_source.")
        if registry_backend is None and frozen and os.name == "nt":
            from napari_vipp.installer.uninstall import WindowsRegistryBackend

            registry_backend = WindowsRegistryBackend()
        self._registry_backend = registry_backend
        if known_folder_probe is None:
            from napari_vipp.installer.discovery import _windows_known_folder

            known_folder_probe = _windows_known_folder
        self._known_folder_probe = known_folder_probe
        self._authorizations: dict[str, str] = {}
        self._authorization_lock = threading.Lock()

    def inspect(self, plan: InstallPlan) -> ManagedTargetInspection:
        self._validate_managed_plan(plan)
        return inspect_managed_target(
            plan.discovery.filesystem.target,
            release=plan.release,
            track=plan.request.track,
        )

    def recover_interrupted(
        self,
        target: str | Path,
        *,
        shortcut_roots: Sequence[str | Path] = (),
    ) -> RollbackReport:
        """Recover one durable pre-commit journal, if present."""

        managed_root = Path(target)
        state_root = self._selected_state_root(managed_root)
        allowed_shortcut_roots = tuple(Path(root) for root in shortcut_roots)
        if not allowed_shortcut_roots:
            allowed_shortcut_roots = _default_shortcut_roots(self._known_folder_probe)
        for root in allowed_shortcut_roots:
            _assert_direct_path(root, "shortcut recovery root")
        journal = _transaction_journal_path(state_root, managed_root)
        if not journal.is_file():
            return RollbackReport(False, True)
        recovery_id = str(uuid.uuid4())
        with _TargetLock(state_root, managed_root, recovery_id):
            return _recover_transaction_journal(
                journal,
                managed_root=managed_root,
                state_root=state_root,
                allowed_shortcut_roots=allowed_shortcut_roots,
                registry_backend=self._registry_backend,
            )

    def prepare(
        self,
        plan: InstallPlan,
        *,
        progress: ProgressCallback | None = None,
        cancellation: object | None = None,
        repair: bool = False,
    ) -> PreparedTransaction:
        """Resolve exact artifacts without changing the managed target."""

        self._validate_managed_plan(plan)
        filesystem = plan.discovery.filesystem
        recovery = self.recover_interrupted(
            filesystem.target,
            shortcut_roots=tuple(
                root
                for root in (
                    filesystem.desktop_directory,
                    filesystem.start_menu_directory,
                )
                if root is not None
            ),
        )
        if not recovery.completed:
            raise RecoveryError(
                "An interrupted VIPP setup could not be cleaned up safely: "
                + "; ".join(recovery.errors)
            )
        _checkpoint(cancellation)
        _emit(
            progress,
            ProgressStage.INSPECTING,
            "Checking the selected VIPP location…",
            0,
            3,
        )
        inspection = self.inspect(plan)
        _validate_snapshot_binding(plan, inspection)
        if inspection.kind is ManagedTargetKind.CURRENT and repair:
            inspection = replace(
                inspection,
                kind=ManagedTargetKind.REPAIR,
                reason=(
                    f"VIPP {plan.release.version} will be reinstalled in a fresh "
                    "managed environment."
                ),
            )
        if inspection.kind in {
            ManagedTargetKind.CURRENT,
            ManagedTargetKind.NEWER,
            ManagedTargetKind.FOREIGN,
        }:
            return _blocked_preparation(
                plan,
                inspection,
                index_url=self._index_url,
            )
        blocking = [
            issue
            for issue in plan.issues
            if issue.severity.value == "error"
            and not (
                inspection.kind is ManagedTargetKind.NEW
                and issue.code == "managed_target_already_exists"
                and _is_empty_managed_residue(inspection.target)
            )
        ]
        if blocking:
            rendered = "; ".join(issue.message for issue in blocking)
            raise PreparationError(f"The installation plan is blocked: {rendered}")
        if self._require_embedded_wheel and plan.release.wheel_path is None:
            raise PreparationError(
                "This signed installer does not contain its exact VIPP release wheel."
            )
        installer_state_root = self._selected_state_root(
            plan.discovery.filesystem.target
        )
        _validate_resolution_temp_capacity(
            plan,
            state_root=installer_state_root,
        )
        recovery_roots = _default_shortcut_roots(self._known_folder_probe)
        reviewed_shortcut_roots = tuple(
            sorted(
                dict.fromkeys(
                    (
                        *recovery_roots,
                        *(
                            root
                            for root in (
                                plan.discovery.filesystem.desktop_directory,
                                plan.discovery.filesystem.start_menu_directory,
                            )
                            if root is not None
                        ),
                    ),
                ),
                key=_path_key,
            )
        )
        shortcuts = _prepare_shortcuts(
            plan,
            inspection,
            cancellation,
            allowed_roots=reviewed_shortcut_roots,
        )
        (
            persistent_path,
            persistent_digest,
            persistent_prior_digest,
        ) = _prepare_persistent_setup(
            inspection,
            track=plan.request.track,
            version=plan.release.version,
            source=self._setup_source,
            destination=self._persistent_setup_path,
            state_root=installer_state_root,
            shortcut_roots=reviewed_shortcut_roots,
            registry_backend=self._registry_backend,
            cancellation=cancellation,
        )
        working_directory = _prepare_working_directory(
            self._known_folder_probe,
            managed_root=inspection.target,
        )
        wheel_digest = _release_wheel_digest(plan.release, cancellation)
        _emit(
            progress,
            ProgressStage.RESOLVING,
            (
                "Reviewing exact packages from PyPI — the first check can take "
                "several minutes."
            ),
            1,
            3,
        )
        result = self._run(
            _resolution_argv(plan, index_url=self._index_url),
            cancellation=cancellation,
            env=_pip_environment(),
        )
        if result.returncode:
            detail = _redact_text(result.stderr.strip() or result.stdout.strip())
            raise ResolutionError(
                "VIPP could not determine the required components. " + detail
            )
        packages = _parse_pip_report(
            result.stdout,
            release=plan.release,
            wheel_sha256=wheel_digest,
            approved_hosts=self._approved_artifact_hosts,
        )
        changes = _package_changes(packages, inspection)
        summary = _review_summary(inspection, changes)
        plan_fingerprint = _plan_fingerprint(plan)
        resolution_id = _resolution_fingerprint(
            plan_fingerprint=plan_fingerprint,
            inspection=inspection,
            packages=packages,
            shortcuts=shortcuts,
            wheel_sha256=wheel_digest,
            index_url=self._index_url,
            persistent_setup_path=persistent_path,
            persistent_setup_sha256=persistent_digest,
            persistent_setup_prior_sha256=persistent_prior_digest,
            working_directory=working_directory,
        )
        prepared = PreparedTransaction(
            plan=plan,
            target_inspection=inspection,
            packages=packages,
            changes=changes,
            shortcuts=shortcuts,
            review_summary=summary,
            resolution_id=resolution_id,
            plan_fingerprint=plan_fingerprint,
            wheel_sha256=wheel_digest,
            index_url=self._index_url,
            persistent_setup_path=persistent_path,
            persistent_setup_sha256=persistent_digest,
            persistent_setup_prior_sha256=persistent_prior_digest,
            working_directory=working_directory,
            resolution_complete=True,
            applicable=True,
        )
        _checkpoint(cancellation)
        _emit(
            progress,
            ProgressStage.READY,
            f"Checks finished. {inspection.action_label} is ready for review.",
            3,
            3,
        )
        return prepared

    def authorize(
        self,
        prepared: PreparedTransaction,
        *,
        confirmed: bool,
    ) -> ExecutionAuthorization:
        """Create a one-use exact-plan authorization after explicit confirmation."""

        if confirmed is not True:
            raise AuthorizationError("Install VIPP was not explicitly confirmed.")
        if not prepared.applicable or not prepared.resolution_complete:
            raise AuthorizationError(
                prepared.blocking_reason or "This prepared plan cannot be applied."
            )
        token = secrets.token_urlsafe(32)
        with self._authorization_lock:
            self._authorizations[token] = prepared.resolution_id
        return ExecutionAuthorization(
            authorization_id=token,
            resolution_id=prepared.resolution_id,
            confirmation_label=prepared.action_label,
        )

    def apply(
        self,
        prepared: PreparedTransaction,
        authorization: ExecutionAuthorization,
        *,
        progress: ProgressCallback | None = None,
        cancellation: object | None = None,
    ) -> InstallResult:
        """Build at a permanent path, accept, then atomically publish ownership."""

        self._consume_authorization(prepared, authorization)
        if _plan_fingerprint(prepared.plan) != prepared.plan_fingerprint:
            raise StalePreparedTransaction(
                "The installation plan changed after review."
            )
        if prepared.index_url != self._index_url:
            raise StalePreparedTransaction(
                "The approved package source changed after review."
            )
        rebound_resolution = _resolution_fingerprint(
            plan_fingerprint=prepared.plan_fingerprint,
            inspection=prepared.target_inspection,
            packages=prepared.packages,
            shortcuts=prepared.shortcuts,
            wheel_sha256=prepared.wheel_sha256,
            index_url=prepared.index_url,
            persistent_setup_path=prepared.persistent_setup_path,
            persistent_setup_sha256=prepared.persistent_setup_sha256,
            persistent_setup_prior_sha256=(prepared.persistent_setup_prior_sha256),
            working_directory=prepared.working_directory,
        )
        if rebound_resolution != prepared.resolution_id:
            raise StalePreparedTransaction(
                "The resolved package set changed after review."
            )
        for package in prepared.packages:
            _validate_artifact_url(
                package.source_url,
                package_name=package.name,
                release=prepared.plan.release,
                approved_hosts=self._approved_artifact_hosts,
            )
        state_root = self._selected_state_root(prepared.target)
        _assert_direct_path(state_root, "installer state")
        _assert_direct_path(prepared.target, "managed installation")
        run_id = self._identifier()
        try:
            uuid.UUID(run_id)
        except (TypeError, ValueError) as exc:
            raise InstallerEngineError(
                "The generated run identifier is invalid."
            ) from exc
        _prune_state_history(state_root)
        run_directory = state_root / "runs" / run_id
        run_directory.mkdir(parents=True, exist_ok=False)
        log_path = run_directory / _LOG_FILENAME
        log = _RunLog(log_path, self._now)
        _atomic_json(run_directory / _RESOLVED_PLAN_FILENAME, prepared.as_dict())
        journal_path = _transaction_journal_path(state_root, prepared.target)
        lock = _TargetLock(state_root, prepared.target, run_id)
        candidate: Path | None = None
        created_directories: tuple[Path, ...] = ()
        shortcut_mutations: tuple[_ShortcutMutation, ...] = ()
        created_shortcut_directories: tuple[Path, ...] = ()
        setup_mutation: _SetupMutation | None = None
        created_setup_directories: tuple[Path, ...] = ()
        committed = False
        retirement_cleanup = RollbackReport(False, True)
        retained_retired: tuple[OwnedEnvironment, ...] = ()
        old_record = prepared.target_inspection.record
        old_registry_plan = None
        if old_record is not None and old_record.registry_key:
            try:
                from napari_vipp.installer.uninstall import (
                    registry_plan_from_record,
                )

                old_inspection = inspect_ownership(prepared.target)
                old_registry_plan = registry_plan_from_record(
                    old_record,
                    old_inspection.manifest_sha256,
                )
            except Exception:
                old_registry_plan = None
        try:
            with lock, log:
                log.write("apply_started", resolution_id=prepared.resolution_id)
                _emit(progress, ProgressStage.PREPARING, "Preparing VIPP…", 0, 6)
                _checkpoint(cancellation)
                current = self.inspect(prepared.plan)
                if current.fingerprint != prepared.target_inspection.fingerprint:
                    raise StalePreparedTransaction(
                        "The installation folder changed after it was reviewed."
                    )
                try:
                    actual_wheel = _release_wheel_digest(
                        prepared.plan.release,
                        cancellation,
                    )
                except ResolutionError as exc:
                    raise StalePreparedTransaction(
                        "The bundled VIPP wheel changed after it was reviewed."
                    ) from exc
                if actual_wheel != prepared.wheel_sha256:
                    raise StalePreparedTransaction(
                        "The bundled VIPP wheel changed after it was reviewed."
                    )
                created_directories = _ensure_managed_directories(prepared.target)
                candidate = _candidate_environment(
                    prepared.target,
                    prepared.plan.release.version,
                    run_id,
                )
                candidate.mkdir(parents=False, exist_ok=False)
                _atomic_json(
                    candidate / _CANDIDATE_MARKER,
                    {
                        "schema": "napari-vipp-install-candidate",
                        "schema_version": 1,
                        "run_id": run_id,
                        "resolution_id": prepared.resolution_id,
                    },
                )
                _write_transaction_journal(
                    journal_path,
                    phase="candidate_created",
                    prepared=prepared,
                    run_id=run_id,
                    run_directory=run_directory,
                    candidate=candidate,
                    mutations=(),
                    setup_mutation=None,
                    previous_registry_plan=old_registry_plan,
                )
                pip_temp = candidate / ".installer-tmp"
                pip_temp.mkdir()
                child_environment = _pip_environment(temp_directory=pip_temp)
                target_python = candidate / "Scripts" / "python.exe"
                base_python = (
                    prepared.plan.discovery.python.executable
                    or prepared.plan.request.python
                )
                _emit(
                    progress,
                    ProgressStage.CREATING_ENVIRONMENT,
                    "Creating a private VIPP workspace…",
                    1,
                    6,
                )
                self._run_checked(
                    (str(base_python), "-m", "venv", str(candidate)),
                    log=log,
                    cancellation=cancellation,
                    env=child_environment,
                )
                self._run_checked(
                    (str(target_python), "-m", "ensurepip", "--upgrade"),
                    log=log,
                    cancellation=cancellation,
                    env=child_environment,
                )
                lock_path = run_directory / "requirements.lock"
                _write_lock_file(lock_path, prepared.packages)
                _emit(
                    progress,
                    ProgressStage.INSTALLING,
                    "Installing VIPP and its required components…",
                    2,
                    6,
                )
                self._run_checked(
                    (
                        str(target_python),
                        "-m",
                        "pip",
                        "--isolated",
                        "install",
                        "--disable-pip-version-check",
                        "--no-input",
                        "--no-cache-dir",
                        "--timeout",
                        str(_PIP_NETWORK_TIMEOUT_SECONDS),
                        "--retries",
                        str(_PIP_NETWORK_RETRIES),
                        "--only-binary=:all:",
                        "--index-url",
                        prepared.index_url,
                        "--require-hashes",
                        "--no-deps",
                        "--requirement",
                        str(lock_path),
                    ),
                    log=log,
                    cancellation=cancellation,
                    env=child_environment,
                )
                _emit(
                    progress,
                    ProgressStage.VERIFYING,
                    "Making sure VIPP is ready…",
                    3,
                    6,
                )
                for action in prepared.plan.acceptance:
                    self._run_checked(
                        _acceptance_argv(action, target_python),
                        log=log,
                        cancellation=cancellation,
                        env=child_environment,
                    )
                try:
                    pip_temp.rmdir()
                except OSError:
                    # pip may leave private temporary data; it remains contained in
                    # this owned environment and is removed with that environment.
                    pass
                launcher = _launcher_path(candidate, prepared.plan.request.track)
                if not launcher.is_file():
                    raise InstallerEngineError(
                        f"The installed VIPP launcher is missing: {launcher}"
                    )
                (
                    setup_mutation,
                    created_setup_directories,
                ) = self._stage_persistent_setup(
                    prepared,
                    run_directory=run_directory,
                    cancellation=cancellation,
                )
                if setup_mutation is not None:
                    _write_transaction_journal(
                        journal_path,
                        phase="setup_staged",
                        prepared=prepared,
                        run_id=run_id,
                        run_directory=run_directory,
                        candidate=candidate,
                        mutations=(),
                        setup_mutation=setup_mutation,
                        previous_registry_plan=old_registry_plan,
                    )
                    _commit_persistent_setup(setup_mutation)
                    _write_transaction_journal(
                        journal_path,
                        phase="setup_committed",
                        prepared=prepared,
                        run_id=run_id,
                        run_directory=run_directory,
                        candidate=candidate,
                        mutations=(),
                        setup_mutation=setup_mutation,
                        previous_registry_plan=old_registry_plan,
                    )
                retirement_cleanup, retained_retired = _cleanup_retired_environments(
                    old_record,
                    cancellation=cancellation,
                )
                _checkpoint(cancellation)
                _emit(
                    progress,
                    ProgressStage.CREATING_SHORTCUTS,
                    "Creating the VIPP shortcut…",
                    4,
                    6,
                )
                (
                    shortcut_mutations,
                    created_shortcut_directories,
                ) = self._stage_shortcuts(
                    prepared.shortcuts,
                    environment_root=candidate,
                    working_directory=prepared.working_directory,
                    run_directory=run_directory,
                    run_id=run_id,
                    log=log,
                    cancellation=cancellation,
                )
                _write_transaction_journal(
                    journal_path,
                    phase="shortcuts_staged",
                    prepared=prepared,
                    run_id=run_id,
                    run_directory=run_directory,
                    candidate=candidate,
                    mutations=shortcut_mutations,
                    setup_mutation=setup_mutation,
                    previous_registry_plan=old_registry_plan,
                )
                _checkpoint(cancellation)
                _commit_shortcuts(shortcut_mutations)
                _write_transaction_journal(
                    journal_path,
                    phase="shortcuts_committed",
                    prepared=prepared,
                    run_id=run_id,
                    run_directory=run_directory,
                    candidate=candidate,
                    mutations=shortcut_mutations,
                    setup_mutation=setup_mutation,
                    previous_registry_plan=old_registry_plan,
                )
                _emit(
                    progress,
                    ProgressStage.COMMITTING,
                    "Finishing the installation…",
                    5,
                    6,
                )
                now = self._now().astimezone(UTC).isoformat()
                installation_id = (
                    old_record.installation_id if old_record is not None else run_id
                )
                created_at = old_record.created_at if old_record is not None else now
                retired = _retired_environments(old_record, retained_retired)
                if prepared.persistent_setup_path is not None:
                    from napari_vipp.installer.uninstall import (
                        registry_key_for_track,
                    )

                    registry_key = registry_key_for_track(prepared.plan.request.track)
                else:
                    registry_key = ""
                record = OwnershipRecord(
                    installation_id=installation_id,
                    managed_root=prepared.target,
                    environment_root=candidate,
                    distribution=prepared.plan.release.distribution,
                    version=prepared.plan.release.version,
                    track=prepared.plan.request.track,
                    base_python=Path(base_python),
                    resolved_plan_id=prepared.resolution_id,
                    packages=tuple(
                        OwnedPackage(package.name, package.version, package.sha256)
                        for package in prepared.packages
                    ),
                    environment_marker_sha256=_sha256_file(
                        candidate / _CANDIDATE_MARKER,
                        None,
                    ),
                    managed_root_preexisting=(
                        old_record.managed_root_preexisting
                        if old_record is not None
                        else prepared.target_inspection.target_preexisting
                    ),
                    shortcuts=tuple(
                        OwnedShortcut(
                            mutation.prepared.destination,
                            mutation.new_sha256,
                            _profile_launcher(
                                candidate,
                                mutation.prepared.profile,
                            ),
                        )
                        for mutation in shortcut_mutations
                        if not mutation.prepared.remove
                    ),
                    retired_environments=retired,
                    uninstaller_path=(prepared.persistent_setup_path),
                    uninstaller_sha256=(prepared.persistent_setup_sha256),
                    registry_key=registry_key,
                    created_at=created_at,
                    updated_at=now,
                )
                record_path = write_ownership_record(prepared.target, record)
                committed = True
                post_commit_warnings = _complete_committed_shortcuts(
                    shortcut_mutations,
                    record=record,
                    candidate=candidate,
                )
                registry_committed = not bool(record.registry_key)
                if record.registry_key:
                    try:
                        from napari_vipp.installer.uninstall import (
                            register_apps_and_features,
                            registry_plan_from_record,
                        )

                        accepted = inspect_ownership(prepared.target)
                        new_registry_plan = registry_plan_from_record(
                            record,
                            accepted.manifest_sha256,
                        )
                        if self._registry_backend is None:
                            raise InstallerEngineError(
                                "Windows registry integration is unavailable."
                            )
                        register_apps_and_features(
                            self._registry_backend,
                            new_registry_plan,
                            previous_plan=old_registry_plan,
                        )
                        registry_committed = True
                    except Exception as exc:
                        post_commit_warnings.append(
                            "Apps & Features registration is incomplete: "
                            f"{_redact_text(str(exc))}"
                        )
                if registry_committed:
                    cleanup_error = _cleanup_previous_persistent_setup(
                        old_registry_plan,
                        record,
                        state_root=state_root,
                    )
                    if cleanup_error:
                        post_commit_warnings.append(cleanup_error)
                    try:
                        from napari_vipp.installer.uninstall import (
                            remove_superseded_uninstall_recoveries,
                        )

                        accepted_current = inspect_ownership(prepared.target)
                        cleanup_roots = tuple(
                            sorted(
                                dict.fromkeys(
                                    (
                                        *_default_shortcut_roots(
                                            self._known_folder_probe
                                        ),
                                        *(
                                            root
                                            for root in (
                                                prepared.plan.discovery.filesystem.desktop_directory,
                                                prepared.plan.discovery.filesystem.start_menu_directory,
                                            )
                                            if root is not None
                                        ),
                                    ),
                                ),
                                key=_path_key,
                            )
                        )
                        remove_superseded_uninstall_recoveries(
                            record,
                            manifest_sha256=accepted_current.manifest_sha256,
                            shortcut_roots=cleanup_roots,
                            registry=self._registry_backend,
                        )
                    except Exception as exc:
                        post_commit_warnings.append(
                            "Old uninstall recovery cleanup is incomplete: "
                            f"{_redact_text(str(exc))}"
                        )
                if not post_commit_warnings:
                    try:
                        journal_path.unlink()
                    except OSError:
                        # A durable accepted journal is safe to replay. Failure to
                        # remove it is therefore a warning, never a false failure.
                        post_commit_warnings.append(
                            "Setup could not remove its completed recovery record."
                        )
                registration_warning = "; ".join(post_commit_warnings)
                try:
                    log.write(
                        "ownership_committed",
                        ownership_record=str(record_path),
                        environment_root=str(candidate),
                    )
                except OSError:
                    pass
                rollback = RollbackReport(
                    attempted=False,
                    completed=True,
                    preserved_paths=tuple(
                        path
                        for path in (
                            old_record.environment_root if old_record else None,
                        )
                        if path is not None
                    ),
                )
                result = InstallResult(
                    status=InstallStatus.SUCCEEDED,
                    resolution_id=prepared.resolution_id,
                    operation=prepared.target_inspection.kind,
                    managed_root=prepared.target,
                    environment_root=candidate,
                    launcher_path=launcher,
                    log_path=log_path,
                    ownership_record_path=record_path,
                    rollback=rollback,
                    message=(
                        "VIPP is ready."
                        if not registration_warning
                        else (
                            "VIPP is ready, but Windows could not finish all repair "
                            "and removal details. Run VIPP Setup again to finish."
                        )
                    ),
                    retirement_cleanup=retirement_cleanup,
                    registration_warning=registration_warning,
                )
                try:
                    _atomic_json(run_directory / _RESULT_FILENAME, result.as_dict())
                except OSError:
                    pass
                try:
                    _emit(
                        progress,
                        ProgressStage.COMPLETED,
                        "VIPP is ready — Open VIPP",
                        6,
                        6,
                    )
                except Exception:
                    # Ownership is the commit point. A late Cancel click or a
                    # failed UI observer cannot turn a live installation into a
                    # reported cancellation/failure.
                    pass
                return result
        except InstallCancelled as exc:
            rollback = _rollback_candidate(
                candidate,
                run_id=run_id,
                created_directories=created_directories,
                shortcut_mutations=shortcut_mutations,
                created_shortcut_directories=created_shortcut_directories,
                setup_mutation=setup_mutation,
                created_setup_directories=created_setup_directories,
                managed_root=prepared.target,
                old_record=old_record,
                committed=committed,
                progress=progress,
            )
            if rollback.completed:
                try:
                    journal_path.unlink()
                except OSError:
                    pass
            result = InstallResult(
                status=InstallStatus.CANCELLED,
                resolution_id=prepared.resolution_id,
                operation=prepared.target_inspection.kind,
                managed_root=prepared.target,
                environment_root=None,
                launcher_path=None,
                log_path=log_path,
                ownership_record_path=(
                    ownership_path(prepared.target) if old_record else None
                ),
                rollback=rollback,
                message=_terminal_failure_message(rollback, cancelled=True),
                technical_error=_redact_text(f"{type(exc).__name__}: {exc}"),
                retirement_cleanup=retirement_cleanup,
            )
            try:
                _atomic_json(run_directory / _RESULT_FILENAME, result.as_dict())
            except OSError:
                pass
            _emit(progress, ProgressStage.CANCELLED, result.message, 6, 6)
            return result
        except Exception as exc:
            rollback = _rollback_candidate(
                candidate,
                run_id=run_id,
                created_directories=created_directories,
                shortcut_mutations=shortcut_mutations,
                created_shortcut_directories=created_shortcut_directories,
                setup_mutation=setup_mutation,
                created_setup_directories=created_setup_directories,
                managed_root=prepared.target,
                old_record=old_record,
                committed=committed,
                progress=progress,
            )
            if rollback.completed:
                try:
                    journal_path.unlink()
                except OSError:
                    pass
            result = InstallResult(
                status=InstallStatus.FAILED,
                resolution_id=prepared.resolution_id,
                operation=prepared.target_inspection.kind,
                managed_root=prepared.target,
                environment_root=None,
                launcher_path=None,
                log_path=log_path,
                ownership_record_path=(
                    ownership_path(prepared.target) if old_record else None
                ),
                rollback=rollback,
                message=_terminal_failure_message(rollback, cancelled=False),
                technical_error=_redact_text(f"{type(exc).__name__}: {exc}"),
                retirement_cleanup=retirement_cleanup,
            )
            try:
                with log:
                    log.write("apply_failed", error=result.technical_error)
            except OSError:
                pass
            try:
                _atomic_json(run_directory / _RESULT_FILENAME, result.as_dict())
            except OSError:
                pass
            _emit(progress, ProgressStage.FAILED, result.message, 6, 6)
            return result

    def _stage_persistent_setup(
        self,
        prepared: PreparedTransaction,
        *,
        run_directory: Path,
        cancellation: object | None,
    ) -> tuple[_SetupMutation | None, tuple[Path, ...]]:
        destination = prepared.persistent_setup_path
        if destination is None:
            if (
                prepared.persistent_setup_sha256
                or prepared.persistent_setup_prior_sha256
            ):
                raise StalePreparedTransaction(
                    "The persistent setup review data is inconsistent."
                )
            return None, ()
        source = self._setup_source
        if source is None:
            raise StalePreparedTransaction(
                "The signed setup program is no longer available."
            )
        state_root = self._selected_state_root(prepared.target)
        if not _is_relative_to(destination, state_root):
            raise StalePreparedTransaction(
                "The persistent setup destination is outside installer state."
            )
        _assert_direct_path(source, "release setup program")
        _assert_direct_path(destination, "persistent setup program")
        if _sha256_file(source, cancellation) != prepared.persistent_setup_sha256:
            raise StalePreparedTransaction(
                "The signed setup program changed after it was reviewed."
            )
        recovery_roots = _default_shortcut_roots(self._known_folder_probe)
        reviewed_shortcut_roots = tuple(
            sorted(
                dict.fromkeys(
                    (
                        *recovery_roots,
                        *(
                            root
                            for root in (
                                prepared.plan.discovery.filesystem.desktop_directory,
                                prepared.plan.discovery.filesystem.start_menu_directory,
                            )
                            if root is not None
                        ),
                    ),
                ),
                key=_path_key,
            )
        )
        try:
            from napari_vipp.installer.uninstall import (
                UninstallPreparationError,
                reap_completed_uninstall_recovery,
            )

            reap_completed_uninstall_recovery(
                destination,
                managed_root=prepared.target,
                expected_sha256=prepared.persistent_setup_sha256,
                shortcut_roots=reviewed_shortcut_roots,
                registry=self._registry_backend,
                expected_track=prepared.plan.request.track,
                keep_executable=True,
                perform_cleanup=False,
            )
        except UninstallPreparationError as exc:
            raise StalePreparedTransaction(str(exc)) from exc
        try:
            destination_metadata = destination.lstat()
        except FileNotFoundError:
            if prepared.persistent_setup_prior_sha256:
                raise StalePreparedTransaction(
                    "The persistent setup program disappeared after review."
                ) from None
        else:
            if (
                not stat.S_ISREG(destination_metadata.st_mode)
                or _metadata_is_reparse(destination_metadata)
                or _sha256_file(destination, cancellation)
                != prepared.persistent_setup_prior_sha256
                or prepared.persistent_setup_prior_sha256
                != prepared.persistent_setup_sha256
            ):
                raise StalePreparedTransaction(
                    "The persistent setup program changed after review."
                )
            return None, ()
        created_directories = _ensure_directory_chain(destination.parent)
        temporary = destination.parent / (
            f".{destination.name}.{run_directory.name}.tmp"
        )
        try:
            with (
                source.open("rb") as source_stream,
                temporary.open("xb") as destination_stream,
            ):
                while True:
                    _checkpoint(cancellation)
                    block = source_stream.read(1024 * 1024)
                    if not block:
                        break
                    destination_stream.write(block)
                destination_stream.flush()
                os.fsync(destination_stream.fileno())
            if (
                _sha256_file(temporary, cancellation)
                != prepared.persistent_setup_sha256
            ):
                raise InstallerEngineError(
                    "The persistent setup copy failed its integrity check."
                )
            return (
                _SetupMutation(
                    destination=destination,
                    temporary=temporary,
                    backup=None,
                    prior_sha256="",
                    new_sha256=prepared.persistent_setup_sha256,
                ),
                created_directories,
            )
        except Exception:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            for directory in reversed(created_directories):
                try:
                    directory.rmdir()
                except (FileNotFoundError, OSError):
                    pass
            raise

    def _stage_shortcuts(
        self,
        shortcuts: tuple[PreparedShortcut, ...],
        *,
        environment_root: Path,
        working_directory: Path,
        run_directory: Path,
        run_id: str,
        log: _RunLog,
        cancellation: object | None,
    ) -> tuple[tuple[_ShortcutMutation, ...], tuple[Path, ...]]:
        if not shortcuts:
            return (), ()
        _validate_working_directory(
            working_directory,
            managed_root=environment_root.parents[2],
        )
        script = run_directory / "create-shortcut.ps1"
        _atomic_text(script, _SHORTCUT_SCRIPT)
        backup_directory = run_directory / "shortcut-backups"
        backup_directory.mkdir()
        mutations: list[_ShortcutMutation] = []
        created_directories: list[Path] = []
        active_temporary: Path | None = None
        try:
            for index, shortcut in enumerate(shortcuts):
                _checkpoint(cancellation)
                created_directories.extend(
                    _ensure_directory_chain(shortcut.destination.parent)
                )
                if shortcut.existed:
                    if (
                        not shortcut.destination.is_file()
                        or shortcut.destination.is_symlink()
                        or _sha256_file(shortcut.destination, cancellation)
                        != shortcut.prior_sha256
                    ):
                        raise StalePreparedTransaction(
                            "A VIPP shortcut changed after it was reviewed."
                        )
                    backup = backup_directory / f"{index}.lnk"
                    shutil.copy2(shortcut.destination, backup)
                else:
                    try:
                        shortcut.destination.lstat()
                    except FileNotFoundError:
                        pass
                    else:
                        raise StalePreparedTransaction(
                            "A shortcut appeared after the installation was reviewed."
                        )
                    backup = None
                temporary = shortcut.destination.parent / (
                    f".{shortcut.destination.stem}.{run_id}.tmp.lnk"
                )
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
                active_temporary = temporary
                if shortcut.remove:
                    mutations.append(
                        _ShortcutMutation(
                            prepared=shortcut,
                            temporary=temporary,
                            backup=backup,
                            new_sha256="",
                        )
                    )
                    active_temporary = None
                    continue
                target = _profile_launcher(environment_root, shortcut.profile)
                if not target.is_file():
                    raise InstallerEngineError(
                        f"The launcher for shortcut {shortcut.label!r} is missing: "
                        f"{target}"
                    )
                self._run_checked(
                    (
                        "powershell.exe",
                        "-NoLogo",
                        "-NoProfile",
                        "-NonInteractive",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-File",
                        str(script),
                        "-Destination",
                        str(temporary),
                        "-Target",
                        str(target),
                        "-WorkingDirectory",
                        str(working_directory),
                        "-Description",
                        "Open VIPP",
                    ),
                    log=log,
                    cancellation=cancellation,
                )
                if not temporary.is_file() or temporary.is_symlink():
                    raise InstallerEngineError(
                        f"Windows did not create the expected shortcut: {temporary}"
                    )
                mutations.append(
                    _ShortcutMutation(
                        prepared=shortcut,
                        temporary=temporary,
                        backup=backup,
                        new_sha256=_sha256_file(temporary, cancellation),
                    )
                )
                active_temporary = None
        except Exception:
            if active_temporary is not None:
                try:
                    active_temporary.unlink()
                except FileNotFoundError:
                    pass
            _rollback_shortcuts(tuple(mutations))
            for directory in reversed(tuple(dict.fromkeys(created_directories))):
                try:
                    directory.rmdir()
                except (FileNotFoundError, OSError):
                    pass
            raise
        return tuple(mutations), tuple(dict.fromkeys(created_directories))

    def _consume_authorization(
        self,
        prepared: PreparedTransaction,
        authorization: ExecutionAuthorization,
    ) -> None:
        if authorization.resolution_id != prepared.resolution_id:
            raise AuthorizationError("Authorization belongs to a different plan.")
        with self._authorization_lock:
            expected = self._authorizations.pop(authorization.authorization_id, None)
        if expected != prepared.resolution_id:
            raise AuthorizationError(
                "Authorization is missing, expired, or has already been used."
            )

    def _selected_state_root(self, target: Path) -> Path:
        if self._state_root is not None:
            selected = self._state_root
        else:
            try:
                local_app_data = self._known_folder_probe("local_app_data")
            except Exception as exc:
                raise InstallerEngineError(
                    "Windows LocalAppData Known Folder lookup failed."
                ) from exc
            if local_app_data is None:
                raise InstallerEngineError(
                    "Windows did not return FOLDERID_LocalAppData for installer state."
                )
            selected = Path(local_app_data) / "VIPP" / "installer"
        if _same_path(selected, target) or _is_relative_to(selected, target):
            raise InstallerEngineError(
                "Installer logs must be stored outside the managed environment."
            )
        _assert_direct_path(selected, "installer state")
        return selected

    @staticmethod
    def _validate_managed_plan(plan: InstallPlan) -> None:
        if plan.request.mode is not InstallMode.MANAGED:
            raise PreparationError(
                "This transactional slice supports managed environments only."
            )

    def _run(
        self,
        argv: Sequence[str],
        *,
        cancellation: object | None,
        env: Mapping[str, str] | None = None,
    ) -> CommandResult:
        _checkpoint(cancellation)
        result = self._runner.run(
            tuple(str(value) for value in argv),
            cancellation=cancellation,
            env=env,
            cwd=None,
        )
        _checkpoint(cancellation)
        return result

    def _run_checked(
        self,
        argv: Sequence[str],
        *,
        log: _RunLog,
        cancellation: object | None,
        env: Mapping[str, str] | None = None,
    ) -> CommandResult:
        log.write("command_started", argv=_redacted_argv(argv))
        result = self._run(argv, cancellation=cancellation, env=env)
        log.write(
            "command_finished",
            argv=_redacted_argv(argv),
            returncode=result.returncode,
            stdout=_redact_text(result.stdout),
            stderr=_redact_text(result.stderr),
        )
        if result.returncode:
            raise CommandFailed(argv, result)
        return result


def _blocked_preparation(
    plan: InstallPlan,
    inspection: ManagedTargetInspection,
    *,
    index_url: str,
) -> PreparedTransaction:
    fingerprint = _plan_fingerprint(plan)
    resolution_id = _resolution_fingerprint(
        plan_fingerprint=fingerprint,
        inspection=inspection,
        packages=(),
        shortcuts=(),
        wheel_sha256="",
        index_url=index_url,
        persistent_setup_path=None,
        persistent_setup_sha256="",
        persistent_setup_prior_sha256="",
        working_directory=_fallback_working_directory(plan),
    )
    return PreparedTransaction(
        plan=plan,
        target_inspection=inspection,
        packages=(),
        changes=(),
        shortcuts=(),
        review_summary=ReviewSummary(
            headline=inspection.action_label,
            detail=inspection.reason,
            packages_to_install=0,
            packages_to_change=0,
            packages_to_remove=0,
        ),
        resolution_id=resolution_id,
        plan_fingerprint=fingerprint,
        wheel_sha256="",
        index_url=index_url,
        persistent_setup_path=None,
        persistent_setup_sha256="",
        persistent_setup_prior_sha256="",
        working_directory=_fallback_working_directory(plan),
        resolution_complete=False,
        applicable=False,
        blocking_reason=inspection.reason,
    )


def _resolution_argv(
    plan: InstallPlan,
    *,
    index_url: str,
) -> tuple[str, ...]:
    base_python = plan.discovery.python.executable or plan.request.python
    return (
        str(base_python),
        "-m",
        "pip",
        "--isolated",
        "install",
        "--dry-run",
        "--ignore-installed",
        "--report",
        "-",
        "--quiet",
        "--disable-pip-version-check",
        "--no-input",
        "--no-cache-dir",
        "--timeout",
        str(_PIP_NETWORK_TIMEOUT_SECONDS),
        "--retries",
        str(_PIP_NETWORK_RETRIES),
        "--only-binary=:all:",
        "--index-url",
        index_url,
        "--upgrade-strategy",
        "only-if-needed",
        plan.release.requirement(plan.request),
    )


def _parse_pip_report(
    text: str,
    *,
    release: ReleaseSpec,
    wheel_sha256: str,
    approved_hosts: frozenset[str],
) -> tuple[ResolvedPackage, ...]:
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ResolutionError("pip returned an invalid dependency report.") from exc
    if not isinstance(document, dict) or not isinstance(document.get("install"), list):
        raise ResolutionError("pip's dependency report does not contain install data.")
    packages: list[ResolvedPackage] = []
    for item in document["install"]:
        if not isinstance(item, dict):
            raise ResolutionError("pip reported an invalid package entry.")
        metadata = item.get("metadata")
        download = item.get("download_info")
        if not isinstance(metadata, dict) or not isinstance(download, dict):
            raise ResolutionError("pip omitted package metadata or download identity.")
        name = metadata.get("name")
        version = metadata.get("version")
        url = download.get("url")
        if not all(isinstance(value, str) and value for value in (name, version, url)):
            raise ResolutionError("pip reported incomplete package identity fields.")
        digest = _report_sha256(download)
        try:
            _validate_artifact_url(
                url,
                package_name=name,
                release=release,
                approved_hosts=approved_hosts,
            )
            package = ResolvedPackage(
                name=name,
                version=version,
                source_url=url,
                sha256=digest,
                requested=item.get("requested") is True,
            )
        except ValueError as exc:
            raise ResolutionError(str(exc)) from exc
        packages.append(package)
    packages.sort(key=lambda package: package.normalized_name)
    normalized = [package.normalized_name for package in packages]
    if len(normalized) != len(set(normalized)):
        raise ResolutionError("pip resolved the same project more than once.")
    selected = [
        package
        for package in packages
        if package.normalized_name == _normalize_name(release.distribution)
    ]
    if len(selected) != 1 or selected[0].version != release.version:
        raise ResolutionError(
            "pip did not resolve the exact VIPP release carried by this installer."
        )
    if wheel_sha256 and selected[0].sha256 != wheel_sha256:
        raise ResolutionError(
            "pip's VIPP artifact does not match the bundled wheel digest."
        )
    return tuple(packages)


def _report_sha256(download: dict[object, object]) -> str:
    archive = download.get("archive_info")
    if not isinstance(archive, dict):
        return ""
    hashes = archive.get("hashes")
    if isinstance(hashes, dict):
        value = hashes.get("sha256")
        if isinstance(value, str):
            return value.strip().lower()
    value = archive.get("hash")
    if isinstance(value, str) and value.lower().startswith("sha256="):
        return value.partition("=")[2].strip().lower()
    return ""


def _validated_index_url(
    value: str,
    *,
    approved_hosts: frozenset[str],
) -> str:
    """Return a canonical, credential-free approved HTTPS simple index."""

    try:
        parsed = urlsplit(value.strip())
        port = parsed.port
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("The package index URL is invalid.") from exc
    hostname = (parsed.hostname or "").casefold()
    if (
        parsed.scheme.casefold() != "https"
        or not hostname
        or hostname not in approved_hosts
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or port not in {None, 443}
    ):
        raise ValueError(
            "The package index must be an approved credential-free HTTPS URL."
        )
    path = parsed.path.rstrip("/")
    if path.casefold() != "/simple":
        raise ValueError("The approved package index must use its /simple endpoint.")
    return urlunsplit(("https", hostname, "/simple", "", ""))


def _validate_artifact_url(
    value: str,
    *,
    package_name: str,
    release: ReleaseSpec,
    approved_hosts: frozenset[str],
) -> None:
    """Reject resolver artifacts outside the reviewed trust boundary."""

    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise ResolutionError("pip reported an invalid package source URL.") from exc
    scheme = parsed.scheme.casefold()
    if parsed.username is not None or parsed.password is not None:
        raise ResolutionError("pip reported a package source containing credentials.")
    if parsed.query or parsed.fragment:
        raise ResolutionError(
            "pip reported a package source with an unreviewable query or fragment."
        )
    if scheme == "https":
        hostname = (parsed.hostname or "").casefold()
        if not hostname or hostname not in approved_hosts or port not in {None, 443}:
            raise ResolutionError(
                "pip selected a package artifact from an unapproved HTTPS host."
            )
        return
    if scheme != "file":
        raise ResolutionError("pip selected an unsupported package source scheme.")
    if parsed.netloc not in {"", "localhost"}:
        raise ResolutionError("pip selected a remote file package source.")
    if release.wheel_path is None or _normalize_name(package_name) != _normalize_name(
        release.distribution
    ):
        raise ResolutionError(
            "Only the exact bundled VIPP wheel may use a local file source."
        )
    source_path = Path(url2pathname(unquote(parsed.path)))
    if not _same_path(source_path, release.wheel_path):
        raise ResolutionError(
            "The local VIPP package is not the exact wheel carried by this installer."
        )


def _package_changes(
    packages: tuple[ResolvedPackage, ...],
    inspection: ManagedTargetInspection,
) -> tuple[PackageReviewChange, ...]:
    old = {
        package.normalized_name: package
        for package in (inspection.record.packages if inspection.record else ())
    }
    new = {package.normalized_name: package for package in packages}
    changes: list[PackageReviewChange] = []
    for name in sorted(set(old) | set(new)):
        before = old.get(name)
        after = new.get(name)
        if before is None:
            disposition = PackageDisposition.INSTALL
        elif after is None:
            disposition = PackageDisposition.REMOVE
        elif before.version == after.version:
            disposition = (
                PackageDisposition.REINSTALL
                if inspection.kind is ManagedTargetKind.REPAIR
                else PackageDisposition.KEEP
            )
        elif _compare_versions(before.version, after.version) < 0:
            disposition = PackageDisposition.UPGRADE
        else:
            disposition = PackageDisposition.DOWNGRADE
        changes.append(
            PackageReviewChange(
                name=after.name if after is not None else before.name,
                installed_version=before.version if before is not None else None,
                resolved_version=after.version if after is not None else None,
                disposition=disposition,
            )
        )
    return tuple(changes)


def _review_summary(
    inspection: ManagedTargetInspection,
    changes: tuple[PackageReviewChange, ...],
) -> ReviewSummary:
    installs = sum(
        change.disposition is PackageDisposition.INSTALL for change in changes
    )
    removals = sum(
        change.disposition is PackageDisposition.REMOVE for change in changes
    )
    changed = sum(
        change.disposition
        in {
            PackageDisposition.UPGRADE,
            PackageDisposition.DOWNGRADE,
            PackageDisposition.REINSTALL,
        }
        for change in changes
    )
    if inspection.kind is ManagedTargetKind.UPDATE:
        headline = "Ready to update VIPP"
        detail = "The old version stays available until the update has been checked."
    elif inspection.kind is ManagedTargetKind.REPAIR:
        headline = "Ready to repair VIPP"
        detail = "A clean replacement will be checked before it becomes current."
    else:
        headline = "Ready to install VIPP"
        detail = "VIPP will be installed in its own private workspace."
    return ReviewSummary(headline, detail, installs, changed, removals)


def _prepare_shortcuts(
    plan: InstallPlan,
    inspection: ManagedTargetInspection,
    cancellation: object | None,
    *,
    allowed_roots: tuple[Path, ...],
) -> tuple[PreparedShortcut, ...]:
    planned: list[tuple[str, str, Path]]
    if plan.request.track is ComputeTrack.CPU:
        directories = tuple(
            dict.fromkeys(shortcut.destination.parent for shortcut in plan.shortcuts)
        )
        planned = [("VIPP", "cpu", directory / "VIPP.lnk") for directory in directories]
    else:
        planned = [
            (shortcut.label, shortcut.profile, shortcut.destination)
            for shortcut in plan.shortcuts
        ]
    owned = inspection.record.shortcuts if inspection.record is not None else ()
    prepared: list[PreparedShortcut] = []
    for label, profile, destination in planned:
        _checkpoint(cancellation)
        try:
            _assert_direct_path(destination.parent, "shortcut destination")
        except InstallerEngineError as exc:
            raise PreparationError(str(exc)) from exc
        try:
            destination_metadata = destination.lstat()
        except FileNotFoundError:
            prepared.append(
                PreparedShortcut(label, profile, destination, existed=False)
            )
            continue
        except OSError as exc:
            raise PreparationError(
                f"The shortcut destination cannot be inspected: {destination}: {exc}"
            ) from exc
        if (
            not destination.is_file()
            or destination.is_symlink()
            or _metadata_is_reparse(destination_metadata)
        ):
            raise PreparationError(
                f"The shortcut destination is not a normal file: {destination}"
            )
        digest = _sha256_file(destination, cancellation)
        authority = next(
            (shortcut for shortcut in owned if _same_path(shortcut.path, destination)),
            None,
        )
        if authority is None or authority.sha256 != digest:
            raise PreparationError(
                "A requested shortcut already exists but is not the exact shortcut "
                f"owned by this VIPP installation: {destination}"
            )
        prepared.append(
            PreparedShortcut(
                label,
                profile,
                destination,
                existed=True,
                prior_sha256=digest,
            )
        )
    desired_paths = {_path_key(shortcut.destination) for shortcut in prepared}
    for authority in owned:
        if _path_key(authority.path) in desired_paths:
            continue
        if not any(_same_path(authority.path.parent, root) for root in allowed_roots):
            raise PreparationError(
                "An old VIPP shortcut is outside the reviewed Desktop or Start "
                f"Menu folder and was preserved: {authority.path}"
            )
        try:
            metadata = authority.path.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise PreparationError(
                f"An old VIPP shortcut could not be inspected: {authority.path}"
            ) from exc
        if (
            not stat.S_ISREG(metadata.st_mode)
            or _metadata_is_reparse(metadata)
            or _sha256_file(authority.path, cancellation) != authority.sha256
        ):
            raise PreparationError(
                f"An old VIPP shortcut changed and was preserved: {authority.path}"
            )
        prepared.append(
            PreparedShortcut(
                label=authority.path.stem,
                profile="remove",
                destination=authority.path,
                existed=True,
                prior_sha256=authority.sha256,
                remove=True,
            )
        )
    return tuple(prepared)


def _prepare_persistent_setup(
    inspection: ManagedTargetInspection,
    *,
    track: ComputeTrack,
    version: str,
    source: Path | None,
    destination: Path | None,
    state_root: Path,
    shortcut_roots: tuple[Path, ...],
    registry_backend: object | None,
    cancellation: object | None,
) -> tuple[Path | None, str, str]:
    if source is None:
        if destination is not None:
            raise PreparationError(
                "The persistent repair and uninstall program is not configured."
            )
        return None, "", ""
    source = Path(os.path.abspath(source))
    try:
        metadata = source.lstat()
        _assert_direct_path(source, "release setup program")
    except (InstallerEngineError, OSError) as exc:
        raise PreparationError(
            "The signed setup program cannot be copied for repair and uninstall."
        ) from exc
    if not stat.S_ISREG(metadata.st_mode) or _metadata_is_reparse(metadata):
        raise PreparationError("The signed setup program is not a normal local file.")
    digest = _sha256_file(source, cancellation)
    root_digest = hashlib.sha256(
        _path_key(inspection.target).encode("utf-8")
    ).hexdigest()
    if destination is None:
        base_destination = state_root / "cache" / version / digest / "VIPP-Setup.exe"
        cache_root = base_destination.parents[2]
        destination = (
            cache_root
            / track.value
            / root_digest
            / version
            / digest
            / base_destination.name
        )
    else:
        base_destination = Path(os.path.abspath(destination))
        destination = (
            base_destination.parent
            / track.value
            / root_digest
            / version
            / digest
            / base_destination.name
        )
    # Track and digest scoping lets CPU and GPU coexist, and means updating one
    # installation can never invalidate the other's recorded uninstaller hash.
    if _same_path(destination, inspection.target) or _is_relative_to(
        destination,
        inspection.target,
    ):
        raise PreparationError(
            "The repair and uninstall program must be outside the VIPP workspace."
        )
    if not _is_relative_to(destination, state_root):
        raise PreparationError(
            "The repair and uninstall program must remain inside installer state."
        )
    completed_recovery = False
    if inspection.record is None:
        try:
            from napari_vipp.installer.uninstall import (
                UninstallPreparationError,
                reap_completed_uninstall_recovery,
            )

            completed_recovery = bool(
                reap_completed_uninstall_recovery(
                    destination,
                    managed_root=inspection.target,
                    expected_sha256=digest,
                    shortcut_roots=shortcut_roots,
                    registry=registry_backend,
                    expected_track=track,
                    keep_executable=True,
                    perform_cleanup=False,
                )
            )
        except UninstallPreparationError as exc:
            raise PreparationError(str(exc)) from exc
    try:
        _assert_direct_path(destination, "persistent setup program")
        destination_metadata = destination.lstat()
    except FileNotFoundError:
        return destination, digest, ""
    except (InstallerEngineError, OSError) as exc:
        raise PreparationError(
            "The persistent repair and uninstall location is unsafe."
        ) from exc
    if not stat.S_ISREG(destination_metadata.st_mode) or _metadata_is_reparse(
        destination_metadata
    ):
        raise PreparationError(
            "The persistent repair and uninstall location is not a normal file."
        )
    prior = _sha256_file(destination, cancellation)
    record = inspection.record
    owned = bool(
        record is not None
        and record.uninstaller_path is not None
        and _same_path(record.uninstaller_path, destination)
        and record.uninstaller_sha256 == prior
    )
    if prior != digest or (
        not owned and not completed_recovery and not _same_path(source, destination)
    ):
        raise PreparationError(
            "A different file already exists where setup keeps its repair and "
            "uninstall program. It was not overwritten."
        )
    return destination, digest, prior


def _prepare_working_directory(
    known_folder_probe: Callable[[str], Path | None],
    *,
    managed_root: Path,
) -> Path:
    try:
        documents = known_folder_probe("documents")
    except Exception:
        documents = None
    candidates = tuple(
        dict.fromkeys(
            Path(os.path.abspath(path))
            for path in (documents, Path.home())
            if path is not None
        )
    )
    for candidate in candidates:
        try:
            _validate_working_directory(candidate, managed_root=managed_root)
        except PreparationError:
            continue
        return candidate
    raise PreparationError(
        "Setup could not find a stable Documents or home folder for VIPP files."
    )


def _validate_working_directory(
    directory: Path,
    *,
    managed_root: Path,
) -> None:
    if (
        not directory.is_absolute()
        or not directory.is_dir()
        or _path_has_reparse_component(directory)
        or _same_path(directory, managed_root)
        or _is_relative_to(directory, managed_root)
    ):
        raise PreparationError(
            "VIPP needs a stable Documents or home folder outside its program files."
        )


def _fallback_working_directory(plan: InstallPlan) -> Path:
    home = Path(os.path.abspath(Path.home()))
    if _same_path(home, plan.discovery.filesystem.target) or _is_relative_to(
        home,
        plan.discovery.filesystem.target,
    ):
        return Path(os.path.abspath(plan.discovery.filesystem.target.parent))
    return home


def _write_lock_file(path: Path, packages: tuple[ResolvedPackage, ...]) -> None:
    lines = []
    for package in packages:
        url = _without_fragment(package.source_url)
        lines.append(f"{package.name} @ {url} --hash=sha256:{package.sha256}")
    _atomic_text(path, "\n".join(lines) + "\n")


def _acceptance_argv(action: PlannedAction, python: Path) -> tuple[str, ...]:
    if not action.argv:
        raise InstallerEngineError(f"Acceptance action {action.action_id} is empty.")
    return (str(python), *action.argv[1:])


def _launcher_path(environment: Path, track: ComputeTrack) -> Path:
    name = "vipp-app.exe" if track is ComputeTrack.CUDA13 else "vipp-cpu.exe"
    return environment / "Scripts" / name


def _profile_launcher(environment: Path, profile: str) -> Path:
    name = {
        "auto": "vipp-app.exe",
        "cpu": "vipp-cpu.exe",
        "prefer_gpu": "vipp-prefer-gpu.exe",
    }.get(profile)
    if name is None:
        raise InstallerEngineError(f"Unknown VIPP shortcut profile: {profile!r}.")
    return environment / "Scripts" / name


def _candidate_environment(target: Path, version: str, run_id: str) -> Path:
    safe_version = _SAFE_PATH_COMPONENT.sub("-", version).strip(".-") or "release"
    return managed_environments_root(target) / f"{safe_version}-{run_id}"


def _ensure_managed_directories(target: Path) -> tuple[Path, ...]:
    created = list(_ensure_directory_chain(target))
    created.extend(_ensure_directory_chain(target / OWNERSHIP_DIRECTORY))
    created.extend(_ensure_directory_chain(managed_environments_root(target)))
    return tuple(created)


def _ensure_directory_chain(directory: Path) -> tuple[Path, ...]:
    missing: list[Path] = []
    current = directory
    while True:
        try:
            current.lstat()
        except FileNotFoundError:
            missing.append(current)
            parent = current.parent
            if parent == current:
                raise InstallerEngineError(
                    f"No existing parent folder is available for {directory}."
                ) from None
            current = parent
            continue
        metadata = current.lstat()
        if (
            not current.is_dir()
            or current.is_symlink()
            or _metadata_is_reparse(metadata)
        ):
            raise InstallerEngineError(
                f"An installer path parent is not a normal folder: {current}"
            )
        break
    created: list[Path] = []
    for path in reversed(missing):
        path.mkdir()
        created.append(path)
    return tuple(created)


def _retired_environments(
    record: OwnershipRecord | None,
    retained: tuple[OwnedEnvironment, ...],
) -> tuple[OwnedEnvironment, ...]:
    if record is None:
        return ()
    return (
        *retained,
        OwnedEnvironment(
            record.environment_root,
            record.environment_marker_sha256,
        ),
    )


def _cleanup_retired_environments(
    record: OwnershipRecord | None,
    *,
    cancellation: object | None,
) -> tuple[RollbackReport, tuple[OwnedEnvironment, ...]]:
    if record is None or not record.retired_environments:
        return RollbackReport(False, True), ()
    removed: list[Path] = []
    errors: list[str] = []
    retained: list[OwnedEnvironment] = []
    for environment in record.retired_environments:
        _checkpoint(cancellation)
        error = _remove_owned_environment(environment, record.managed_root)
        if error:
            errors.append(error)
            retained.append(environment)
        else:
            removed.append(environment.path)
    return (
        RollbackReport(
            attempted=True,
            completed=not errors,
            removed_paths=tuple(removed),
            preserved_paths=tuple(environment.path for environment in retained),
            errors=tuple(errors),
        ),
        tuple(retained),
    )


def _remove_owned_environment(
    environment: OwnedEnvironment,
    managed_root: Path,
) -> str:
    path = environment.path
    store = managed_environments_root(managed_root)
    if not _is_relative_to(path, store) or _same_path(path, store):
        return f"Owned environment is outside the managed store: {path}"
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return ""
    except OSError as exc:
        return f"Owned environment could not be inspected and was preserved: {exc}"
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or _metadata_is_reparse(metadata)
        or _path_has_reparse_component(path)
        or _tree_has_reparse(path)
    ):
        return f"Owned environment is redirected and was preserved: {path}"
    marker = path / _CANDIDATE_MARKER
    if (
        not marker.is_file()
        or marker.is_symlink()
        or _sha256_file(marker, None) != environment.marker_sha256
    ):
        return f"Owned environment marker changed; the folder was preserved: {path}"
    try:
        shutil.rmtree(path)
    except OSError as exc:
        return f"Could not remove retired environment {path}: {exc}"
    return ""


def _commit_persistent_setup(mutation: _SetupMutation) -> None:
    _assert_direct_path(mutation.destination.parent, "persistent setup")
    try:
        mutation.destination.lstat()
    except FileNotFoundError:
        pass
    else:
        raise StalePreparedTransaction(
            "A file appeared at the persistent setup location after review."
        )
    if (
        not mutation.temporary.is_file()
        or mutation.temporary.is_symlink()
        or _sha256_file(mutation.temporary, None) != mutation.new_sha256
    ):
        raise StalePreparedTransaction(
            "The staged persistent setup program changed before publication."
        )
    os.replace(mutation.temporary, mutation.destination)
    mutation.committed = True


def _rollback_persistent_setup(
    mutation: _SetupMutation | None,
) -> tuple[list[Path], list[str]]:
    if mutation is None:
        return [], []
    removed: list[Path] = []
    errors: list[str] = []
    target = mutation.destination if mutation.committed else mutation.temporary
    try:
        _assert_direct_path(target.parent, "persistent setup rollback")
        if not target.exists():
            return removed, errors
        if (
            not target.is_file()
            or target.is_symlink()
            or _sha256_file(target, None) != mutation.new_sha256
        ):
            raise OSError("the setup program changed before it could be removed")
        target.unlink()
        removed.append(target)
    except (InstallerEngineError, OSError) as exc:
        errors.append(f"Could not roll back persistent setup {target}: {exc}")
    return removed, errors


def _commit_shortcuts(mutations: tuple[_ShortcutMutation, ...]) -> None:
    for mutation in mutations:
        _assert_direct_path(mutation.prepared.destination.parent, "shortcut")
        if mutation.prepared.remove:
            destination = mutation.prepared.destination
            if (
                not destination.is_file()
                or destination.is_symlink()
                or _sha256_file(destination, None) != mutation.prepared.prior_sha256
            ):
                raise StalePreparedTransaction(
                    "An old VIPP shortcut changed before it could be removed."
                )
            try:
                mutation.temporary.lstat()
            except FileNotFoundError:
                pass
            else:
                raise StalePreparedTransaction(
                    "The shortcut removal staging path is no longer empty."
                )
            os.replace(destination, mutation.temporary)
            mutation.committed = True
            continue
        if (
            not mutation.temporary.is_file()
            or mutation.temporary.is_symlink()
            or _sha256_file(mutation.temporary, None) != mutation.new_sha256
        ):
            raise StalePreparedTransaction(
                "A staged VIPP shortcut changed before it could be published."
            )
        os.replace(mutation.temporary, mutation.prepared.destination)
        mutation.committed = True


def _complete_committed_shortcuts(
    mutations: tuple[_ShortcutMutation, ...],
    *,
    record: OwnershipRecord,
    candidate: Path,
) -> list[str]:
    """Finish journalled shortcut swaps after ownership is authoritative."""

    errors: list[str] = []
    for mutation in mutations:
        destination = mutation.prepared.destination
        try:
            _assert_direct_path(destination.parent, "shortcut recovery")
            authority = next(
                (
                    shortcut
                    for shortcut in record.shortcuts
                    if _same_path(shortcut.path, destination)
                ),
                None,
            )
            if mutation.prepared.remove:
                if authority is not None:
                    raise OSError(
                        "the ownership record still authorizes this removed shortcut"
                    )
                try:
                    metadata = destination.lstat()
                except FileNotFoundError:
                    metadata = None
                if metadata is not None:
                    if (
                        not stat.S_ISREG(metadata.st_mode)
                        or _metadata_is_reparse(metadata)
                        or _sha256_file(destination, None)
                        != mutation.prepared.prior_sha256
                    ):
                        raise OSError(
                            "the old shortcut changed before removal could finish"
                        )
                    try:
                        mutation.temporary.lstat()
                    except FileNotFoundError:
                        os.replace(destination, mutation.temporary)
                    else:
                        raise OSError(
                            "both the old shortcut and its staged removal exist"
                        )
                try:
                    temporary_metadata = mutation.temporary.lstat()
                except FileNotFoundError:
                    continue
                if (
                    not stat.S_ISREG(temporary_metadata.st_mode)
                    or _metadata_is_reparse(temporary_metadata)
                    or _sha256_file(mutation.temporary, None)
                    != mutation.prepared.prior_sha256
                ):
                    raise OSError("the staged removed shortcut changed")
                mutation.temporary.unlink()
                continue
            if (
                authority is None
                or authority.sha256 != mutation.new_sha256
                or authority.target is None
                or not _same_path(
                    authority.target,
                    _profile_launcher(candidate, mutation.prepared.profile),
                )
            ):
                raise OSError("the ownership record does not authorize this shortcut")
            if (
                destination.is_file()
                and not destination.is_symlink()
                and _sha256_file(destination, None) == mutation.new_sha256
            ):
                continue
            if (
                mutation.temporary.is_file()
                and not mutation.temporary.is_symlink()
                and _sha256_file(mutation.temporary, None) == mutation.new_sha256
            ):
                os.replace(mutation.temporary, destination)
                continue
            raise OSError("neither the committed nor staged shortcut is intact")
        except (InstallerEngineError, OSError) as exc:
            errors.append(f"Could not finish shortcut {destination}: {exc}")
    return errors


def _rollback_shortcuts(
    mutations: tuple[_ShortcutMutation, ...],
) -> tuple[list[Path], list[str]]:
    removed: list[Path] = []
    errors: list[str] = []
    for mutation in reversed(mutations):
        destination = mutation.prepared.destination
        try:
            _assert_direct_path(destination.parent, "shortcut rollback")
            if mutation.committed:
                if mutation.prepared.remove:
                    try:
                        destination.lstat()
                    except FileNotFoundError:
                        pass
                    else:
                        raise OSError(
                            "the removed shortcut destination is no longer empty"
                        )
                    if (
                        not mutation.temporary.is_file()
                        or mutation.temporary.is_symlink()
                        or _sha256_file(mutation.temporary, None)
                        != mutation.prepared.prior_sha256
                    ):
                        raise OSError(
                            "the removed shortcut bytes changed before restoration"
                        )
                    os.replace(mutation.temporary, destination)
                elif mutation.backup is not None:
                    os.replace(mutation.backup, destination)
                elif (
                    destination.is_file()
                    and not destination.is_symlink()
                    and _sha256_file(destination, None) == mutation.new_sha256
                ):
                    destination.unlink()
                    removed.append(destination)
                else:
                    raise OSError("the new shortcut changed before it could be removed")
            elif mutation.temporary.exists():
                if (
                    mutation.temporary.is_file()
                    and not mutation.temporary.is_symlink()
                    and _sha256_file(mutation.temporary, None)
                    == (
                        mutation.prepared.prior_sha256
                        if mutation.prepared.remove
                        else mutation.new_sha256
                    )
                ):
                    if mutation.prepared.remove:
                        try:
                            destination.lstat()
                        except FileNotFoundError:
                            os.replace(mutation.temporary, destination)
                        else:
                            raise OSError(
                                "both the removed shortcut and its destination exist"
                            )
                    else:
                        mutation.temporary.unlink()
                        removed.append(mutation.temporary)
                else:
                    raise OSError("the staged shortcut changed unexpectedly")
        except (InstallerEngineError, OSError) as exc:
            errors.append(f"Could not roll back shortcut {destination}: {exc}")
    return removed, errors


def _rollback_candidate(
    candidate: Path | None,
    *,
    run_id: str,
    created_directories: tuple[Path, ...],
    shortcut_mutations: tuple[_ShortcutMutation, ...],
    created_shortcut_directories: tuple[Path, ...],
    setup_mutation: _SetupMutation | None,
    created_setup_directories: tuple[Path, ...],
    managed_root: Path,
    old_record: OwnershipRecord | None,
    committed: bool,
    progress: ProgressCallback | None,
) -> RollbackReport:
    if committed:
        return RollbackReport(
            attempted=False,
            completed=True,
            preserved_paths=(candidate,) if candidate is not None else (),
        )
    _emit(
        progress,
        ProgressStage.ROLLING_BACK,
        "Removing the incomplete installation…",
        5,
        6,
    )
    removed: list[Path] = []
    errors: list[str] = []
    incomplete_paths: list[Path] = []
    shortcut_removed, shortcut_errors = _rollback_shortcuts(shortcut_mutations)
    removed.extend(shortcut_removed)
    errors.extend(shortcut_errors)
    setup_removed, setup_errors = _rollback_persistent_setup(setup_mutation)
    removed.extend(setup_removed)
    errors.extend(setup_errors)
    if setup_errors and setup_mutation is not None:
        incomplete_paths.append(
            setup_mutation.destination
            if setup_mutation.committed
            else setup_mutation.temporary
        )
    if candidate is not None and candidate.exists():
        marker = candidate / _CANDIDATE_MARKER
        if not _is_relative_to(candidate, managed_environments_root(managed_root)):
            errors.append(f"Candidate is outside the managed store: {candidate}")
            incomplete_paths.append(candidate)
        elif _path_has_reparse_component(candidate) or _tree_has_reparse(candidate):
            errors.append(
                f"Candidate path is redirected; it was preserved: {candidate}"
            )
            incomplete_paths.append(candidate)
        elif _candidate_marker_matches(marker, run_id):
            try:
                shutil.rmtree(candidate)
                removed.append(candidate)
            except OSError as exc:
                errors.append(f"Could not remove candidate {candidate}: {exc}")
                incomplete_paths.append(candidate)
        else:
            errors.append(
                f"Candidate ownership marker is missing or mismatched: {candidate}"
            )
            incomplete_paths.append(candidate)
    for directory in reversed(created_directories):
        try:
            if _path_has_reparse_component(directory):
                continue
            directory.rmdir()
            removed.append(directory)
        except FileNotFoundError:
            continue
        except OSError:
            # A non-empty directory is outside this run's bounded removal set.
            continue
    for directory in reversed(created_shortcut_directories):
        try:
            if _path_has_reparse_component(directory):
                continue
            directory.rmdir()
            removed.append(directory)
        except (FileNotFoundError, OSError):
            continue
    for directory in reversed(created_setup_directories):
        try:
            if _path_has_reparse_component(directory):
                continue
            directory.rmdir()
            removed.append(directory)
        except (FileNotFoundError, OSError):
            continue
    baseline_preserved = (
        (old_record.environment_root, managed_root)
        if old_record is not None
        else (managed_root.parent,)
    )
    removed_keys = {_path_key(path) for path in removed}
    preserved = tuple(
        path
        for path in dict.fromkeys((*baseline_preserved, *incomplete_paths))
        if _path_key(path) not in removed_keys
    )
    return RollbackReport(
        attempted=True,
        completed=not errors,
        removed_paths=tuple(removed),
        preserved_paths=preserved,
        errors=tuple(errors),
    )


def _candidate_marker_matches(path: Path, run_id: str) -> bool:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return isinstance(document, dict) and document.get("run_id") == run_id


def _transaction_journal_path(state_root: Path, target: Path) -> Path:
    digest = hashlib.sha256(_path_key(target).encode("utf-8")).hexdigest()
    return state_root / "transactions" / f"{digest}.json"


def _default_shortcut_roots(
    known_folder_probe: Callable[[str], Path | None],
) -> tuple[Path, ...]:
    roots: list[Path] = []
    try:
        desktop = known_folder_probe("desktop")
    except Exception:
        desktop = None
    try:
        programs = known_folder_probe("programs")
    except Exception:
        programs = None
    if desktop is not None:
        roots.append(Path(desktop))
    if programs is not None:
        roots.append(Path(programs) / "VIPP")
    return tuple(dict.fromkeys(roots))


def _registry_plan_document(
    plan: _RegistryPlanLike | None,
) -> dict[str, object] | None:
    if plan is None:
        return None
    return {
        "key": str(plan.key),
        "values": [[str(name), value] for name, value in tuple(plan.values)],
        "managed_root": str(plan.managed_root),
        "installation_id": str(plan.installation_id),
        "manifest_sha256": str(plan.manifest_sha256),
        "uninstaller_path": str(plan.uninstaller_path),
        "uninstaller_sha256": str(plan.uninstaller_sha256),
    }


def _journal_registry_plan(
    value: object,
    *,
    managed_root: Path,
    state_root: Path,
) -> _RegistryPlanLike | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("journal previous_registry must be an object")
    raw_values = value.get("values")
    if not isinstance(raw_values, list):
        raise ValueError("journal registry values must be a list")
    values: list[tuple[str, str | int]] = []
    for item in raw_values:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not item[0].strip()
            or not isinstance(item[1], (str, int))
        ):
            raise ValueError("a journal registry value is invalid")
        values.append((item[0], item[1]))
    uninstaller_path = Path(_journal_text(value, "uninstaller_path"))
    if not _is_relative_to(uninstaller_path, state_root) or _path_has_reparse_component(
        uninstaller_path
    ):
        raise ValueError("the previous uninstaller is outside installer state")
    recorded_root = Path(_journal_text(value, "managed_root"))
    if not _same_path(recorded_root, managed_root):
        raise ValueError("the previous registry plan targets another location")
    key = _journal_text(value, "key")
    from napari_vipp.installer.uninstall import (
        CPU_REGISTRY_KEY,
        CUDA13_REGISTRY_KEY,
        RegistryRegistrationPlan,
    )

    if key.casefold() not in {
        CPU_REGISTRY_KEY.casefold(),
        CUDA13_REGISTRY_KEY.casefold(),
    }:
        raise ValueError("the previous registry key is not a VIPP track key")
    return RegistryRegistrationPlan(
        key=key,
        values=tuple(values),
        managed_root=recorded_root,
        installation_id=_journal_text(value, "installation_id"),
        manifest_sha256=_journal_digest(value, "manifest_sha256"),
        uninstaller_path=uninstaller_path,
        uninstaller_sha256=_journal_digest(value, "uninstaller_sha256"),
    )


def _write_transaction_journal(
    path: Path,
    *,
    phase: str,
    prepared: PreparedTransaction,
    run_id: str,
    run_directory: Path,
    candidate: Path,
    mutations: tuple[_ShortcutMutation, ...],
    setup_mutation: _SetupMutation | None,
    previous_registry_plan: _RegistryPlanLike | None = None,
) -> None:
    marker = candidate / _CANDIDATE_MARKER
    _atomic_json(
        path,
        {
            "schema": "napari-vipp-install-transaction",
            "schema_version": 1,
            "phase": phase,
            "run_id": run_id,
            "resolution_id": prepared.resolution_id,
            "old_resolution_id": (
                prepared.target_inspection.record.resolved_plan_id
                if prepared.target_inspection.record is not None
                else None
            ),
            "managed_root": str(prepared.target),
            "target_preexisting": prepared.target_inspection.target_preexisting,
            "run_directory": str(run_directory),
            "candidate": str(candidate),
            "candidate_marker_sha256": _sha256_file(marker, None),
            "previous_registry": _registry_plan_document(previous_registry_plan),
            "persistent_setup": (
                {
                    "destination": str(setup_mutation.destination),
                    "temporary": str(setup_mutation.temporary),
                    "backup": (
                        str(setup_mutation.backup)
                        if setup_mutation.backup is not None
                        else None
                    ),
                    "prior_sha256": setup_mutation.prior_sha256 or None,
                    "new_sha256": setup_mutation.new_sha256,
                }
                if setup_mutation is not None
                else None
            ),
            "shortcuts": [
                {
                    "label": mutation.prepared.label,
                    "profile": mutation.prepared.profile,
                    "destination": str(mutation.prepared.destination),
                    "existed": mutation.prepared.existed,
                    "prior_sha256": mutation.prepared.prior_sha256 or None,
                    "remove": mutation.prepared.remove,
                    "temporary": str(mutation.temporary),
                    "backup": (
                        str(mutation.backup) if mutation.backup is not None else None
                    ),
                    "new_sha256": mutation.new_sha256 or None,
                }
                for mutation in mutations
            ],
        },
    )


def _recover_transaction_journal(
    path: Path,
    *,
    managed_root: Path,
    state_root: Path,
    allowed_shortcut_roots: tuple[Path, ...],
    registry_backend: object | None = None,
) -> RollbackReport:
    errors: list[str] = []
    removed: list[Path] = []
    try:
        if path.is_symlink() or _path_has_reparse_component(path):
            raise ValueError("the transaction journal is redirected")
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError("the transaction journal must be an object")
        if document.get("schema") != "napari-vipp-install-transaction":
            raise ValueError("the transaction journal schema is not recognized")
        if document.get("schema_version") != 1:
            raise ValueError("the transaction journal version is not supported")
        if not _same_path(Path(_journal_text(document, "managed_root")), managed_root):
            raise ValueError("the transaction journal targets another location")
        resolution_id = _journal_digest(document, "resolution_id")
        old_resolution = document.get("old_resolution_id")
        if old_resolution is not None and (
            not isinstance(old_resolution, str)
            or not _HEX_DIGEST.fullmatch(old_resolution)
        ):
            raise ValueError("old_resolution_id is invalid")
        run_directory = Path(_journal_text(document, "run_directory"))
        if not _is_relative_to(run_directory, state_root / "runs"):
            raise ValueError("the transaction run directory is outside installer state")
        candidate = Path(_journal_text(document, "candidate"))
        if not _is_relative_to(candidate, managed_environments_root(managed_root)):
            raise ValueError("the candidate is outside the managed environment store")
        marker_sha = _journal_digest(document, "candidate_marker_sha256")
        target_preexisting = document.get("target_preexisting")
        if not isinstance(target_preexisting, bool):
            raise ValueError("target_preexisting must be a boolean")
        mutations = _journal_shortcut_mutations(
            document.get("shortcuts"),
            run_directory=run_directory,
            allowed_shortcut_roots=allowed_shortcut_roots,
        )
        setup_mutation = _journal_setup_mutation(
            document.get("persistent_setup"),
            state_root=state_root,
            run_directory=run_directory,
        )
        previous_registry_plan = _journal_registry_plan(
            document.get("previous_registry"),
            managed_root=managed_root,
            state_root=state_root,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return RollbackReport(
            attempted=True,
            completed=False,
            preserved_paths=(path, managed_root),
            errors=(f"Interrupted setup journal is invalid: {exc}",),
        )
    ownership = inspect_ownership(managed_root)
    current_resolution = (
        ownership.record.resolved_plan_id
        if ownership.state is OwnershipState.VALID and ownership.record is not None
        else None
    )
    if current_resolution == resolution_id:
        # Ownership is the commit point. Complete any shortcut whose atomic replace
        # was interrupted immediately before the ownership write returned.
        assert ownership.record is not None
        errors.extend(
            _complete_committed_shortcuts(
                mutations,
                record=ownership.record,
                candidate=candidate,
            )
        )
        if setup_mutation is not None:
            try:
                assert ownership.record is not None
                if (
                    ownership.record.uninstaller_path is None
                    or not _same_path(
                        ownership.record.uninstaller_path,
                        setup_mutation.destination,
                    )
                    or ownership.record.uninstaller_sha256 != setup_mutation.new_sha256
                ):
                    raise OSError(
                        "ownership does not authorize the persistent setup program"
                    )
                if (
                    setup_mutation.destination.is_file()
                    and not setup_mutation.destination.is_symlink()
                    and _sha256_file(setup_mutation.destination, None)
                    == setup_mutation.new_sha256
                ):
                    pass
                elif (
                    setup_mutation.temporary.is_file()
                    and not setup_mutation.temporary.is_symlink()
                    and _sha256_file(setup_mutation.temporary, None)
                    == setup_mutation.new_sha256
                ):
                    os.replace(
                        setup_mutation.temporary,
                        setup_mutation.destination,
                    )
                else:
                    raise OSError("the persistent setup program is not intact")
            except OSError as exc:
                errors.append(f"Could not finish persistent setup: {exc}")
        assert ownership.record is not None
        if ownership.record.registry_key:
            try:
                from napari_vipp.installer.uninstall import (
                    register_apps_and_features,
                    registry_plan_from_record,
                )

                if registry_backend is None:
                    raise InstallerEngineError(
                        "Windows registry integration is unavailable."
                    )
                current_plan = registry_plan_from_record(
                    ownership.record,
                    ownership.manifest_sha256,
                )
                if previous_registry_plan is not None and (
                    previous_registry_plan.key.casefold() != current_plan.key.casefold()
                    or previous_registry_plan.installation_id
                    != current_plan.installation_id
                    or not _same_path(
                        previous_registry_plan.managed_root,
                        current_plan.managed_root,
                    )
                ):
                    raise InstallerEngineError(
                        "The previous Apps & Features entry does not match this "
                        "installation."
                    )
                register_apps_and_features(
                    registry_backend,
                    current_plan,
                    previous_plan=previous_registry_plan,
                    recover_interrupted=True,
                )
                cleanup_error = _cleanup_previous_persistent_setup(
                    previous_registry_plan,
                    ownership.record,
                    state_root=state_root,
                )
                if cleanup_error:
                    raise OSError(cleanup_error)
            except Exception as exc:
                errors.append(
                    "Could not finish Apps & Features registration: "
                    f"{_redact_text(str(exc))}"
                )
        elif previous_registry_plan is not None:
            errors.append(
                "The interrupted setup dropped an owned Apps & Features entry."
            )
    elif current_resolution == old_resolution or (
        old_resolution is None and ownership.state is OwnershipState.ABSENT
    ):
        for mutation in mutations:
            destination = mutation.prepared.destination
            try:
                if mutation.prepared.remove:
                    destination_matches = bool(
                        destination.is_file()
                        and not destination.is_symlink()
                        and _sha256_file(destination, None)
                        == mutation.prepared.prior_sha256
                    )
                    staged_matches = bool(
                        mutation.temporary.is_file()
                        and not mutation.temporary.is_symlink()
                        and _sha256_file(mutation.temporary, None)
                        == mutation.prepared.prior_sha256
                    )
                    if destination_matches and staged_matches:
                        raise OSError(
                            "both the shortcut and its removal staging file exist"
                        )
                    if not destination_matches and not staged_matches:
                        raise OSError(
                            "neither the shortcut nor its removal staging file is "
                            "intact"
                        )
                    mutation.committed = staged_matches and not destination_matches
                else:
                    mutation.committed = bool(
                        destination.is_file()
                        and not destination.is_symlink()
                        and _sha256_file(destination, None) == mutation.new_sha256
                    )
            except OSError as exc:
                mutation.committed = False
                errors.append(
                    f"Could not inspect interrupted shortcut {destination}: {exc}"
                )
        shortcut_removed, shortcut_errors = _rollback_shortcuts(mutations)
        removed.extend(shortcut_removed)
        errors.extend(shortcut_errors)
        if setup_mutation is not None:
            try:
                setup_mutation.committed = bool(
                    setup_mutation.destination.is_file()
                    and not setup_mutation.destination.is_symlink()
                    and _sha256_file(setup_mutation.destination, None)
                    == setup_mutation.new_sha256
                )
            except OSError:
                setup_mutation.committed = False
            setup_removed, setup_errors = _rollback_persistent_setup(setup_mutation)
            removed.extend(setup_removed)
            errors.extend(setup_errors)
            if not setup_errors:
                _remove_empty_parents(
                    setup_mutation.destination.parent,
                    stop=state_root,
                    removed=removed,
                )
        environment = OwnedEnvironment(candidate, marker_sha)
        environment_error = _remove_owned_environment(environment, managed_root)
        if environment_error:
            errors.append(environment_error)
        elif not candidate.exists():
            removed.append(candidate)
        if not errors and ownership.state is OwnershipState.ABSENT:
            for directory in (
                managed_environments_root(managed_root),
                managed_root / OWNERSHIP_DIRECTORY,
            ):
                try:
                    if _path_has_reparse_component(directory):
                        continue
                    directory.rmdir()
                    removed.append(directory)
                except (FileNotFoundError, OSError):
                    pass
            if not target_preexisting:
                try:
                    if not _path_has_reparse_component(managed_root):
                        managed_root.rmdir()
                        removed.append(managed_root)
                except (FileNotFoundError, OSError):
                    pass
    else:
        errors.append(
            "The managed ownership record changed after the interrupted setup."
        )
    if not errors:
        try:
            path.unlink()
            removed.append(path)
        except OSError as exc:
            errors.append(f"Could not remove recovered transaction journal: {exc}")
    return RollbackReport(
        attempted=True,
        completed=not errors,
        removed_paths=tuple(removed),
        preserved_paths=(() if not errors else (path, managed_root)),
        errors=tuple(errors),
    )


def _journal_setup_mutation(
    value: object,
    *,
    state_root: Path,
    run_directory: Path,
) -> _SetupMutation | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("journal persistent_setup must be an object")
    destination = Path(_journal_text(value, "destination"))
    temporary = Path(_journal_text(value, "temporary"))
    if (
        destination.name != "VIPP-Setup.exe"
        or not _is_relative_to(destination, state_root)
        or temporary.parent != destination.parent
        or _path_has_reparse_component(destination.parent)
    ):
        raise ValueError("journal persistent setup paths are outside installer state")
    backup_value = value.get("backup")
    backup = Path(backup_value) if isinstance(backup_value, str) else None
    if backup is not None and not _is_relative_to(backup, run_directory):
        raise ValueError("journal persistent setup backup is outside the run")
    prior_value = value.get("prior_sha256")
    prior = "" if prior_value is None else _journal_digest(value, "prior_sha256")
    if bool(backup) != bool(prior):
        raise ValueError("journal persistent setup baseline is inconsistent")
    return _SetupMutation(
        destination=destination,
        temporary=temporary,
        backup=backup,
        prior_sha256=prior,
        new_sha256=_journal_digest(value, "new_sha256"),
    )


def _remove_empty_parents(
    directory: Path,
    *,
    stop: Path,
    removed: list[Path],
) -> None:
    current = directory
    while _is_relative_to(current, stop) and not _same_path(current, stop):
        try:
            if _path_has_reparse_component(current):
                return
            current.rmdir()
            removed.append(current)
        except (FileNotFoundError, OSError):
            return
        current = current.parent


def _cleanup_previous_persistent_setup(
    previous_plan: _RegistryPlanLike | None,
    new_record: OwnershipRecord,
    *,
    state_root: Path,
) -> str:
    if previous_plan is None or new_record.uninstaller_path is None:
        return ""
    old_path = Path(previous_plan.uninstaller_path)
    old_digest = str(previous_plan.uninstaller_sha256).lower()
    if _same_path(old_path, new_record.uninstaller_path):
        return ""
    # Delete only cache paths created by the track/digest-scoped contract. A
    # legacy shared bootstrapper is preserved because another track may use it.
    normalized_parts = {part.casefold() for part in old_path.parts}
    if (
        not _is_relative_to(old_path, state_root)
        or new_record.track.value.casefold() not in normalized_parts
        or old_digest not in normalized_parts
        or _path_has_reparse_component(old_path)
    ):
        return ""
    try:
        metadata = old_path.lstat()
    except FileNotFoundError:
        return ""
    except OSError as exc:
        return f"The previous repair program could not be inspected: {exc}"
    try:
        if (
            not stat.S_ISREG(metadata.st_mode)
            or _metadata_is_reparse(metadata)
            or _sha256_file(old_path, None) != old_digest
        ):
            return (
                f"The previous repair program changed and was preserved at {old_path}."
            )
        old_path.unlink()
        removed: list[Path] = []
        _remove_empty_parents(
            old_path.parent,
            stop=state_root,
            removed=removed,
        )
    except OSError as exc:
        return f"The previous repair program could not be removed: {exc}"
    return ""


def _journal_shortcut_mutations(
    value: object,
    *,
    run_directory: Path,
    allowed_shortcut_roots: tuple[Path, ...],
) -> tuple[_ShortcutMutation, ...]:
    if not isinstance(value, list):
        raise ValueError("journal shortcuts must be a list")
    mutations: list[_ShortcutMutation] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("each journal shortcut must be an object")
        destination = Path(_journal_text(item, "destination"))
        if destination.name not in {
            "VIPP.lnk",
            "VIPP Automatic.lnk",
            "VIPP CPU.lnk",
            "VIPP Prefer GPU.lnk",
        } or not any(
            _same_path(destination.parent, root) for root in allowed_shortcut_roots
        ):
            raise ValueError(
                "a journal shortcut is outside the reviewed Desktop/Start Menu roots"
            )
        if _path_has_reparse_component(destination.parent):
            raise ValueError("a journal shortcut root is redirected")
        temporary = Path(_journal_text(item, "temporary"))
        if temporary.parent != destination.parent:
            raise ValueError("a staged shortcut is outside its destination folder")
        backup_value = item.get("backup")
        backup = Path(backup_value) if isinstance(backup_value, str) else None
        if backup is not None and not _is_relative_to(backup, run_directory):
            raise ValueError("a shortcut backup is outside the run directory")
        existed = item.get("existed")
        if not isinstance(existed, bool):
            raise ValueError("shortcut existed must be a boolean")
        remove = item.get("remove", False)
        if not isinstance(remove, bool):
            raise ValueError("shortcut remove must be a boolean")
        prior = item.get("prior_sha256")
        prior_digest = "" if prior is None else _journal_digest(item, "prior_sha256")
        if existed != bool(prior_digest) or existed != (backup is not None):
            raise ValueError("shortcut baseline fields are inconsistent")
        new_value = item.get("new_sha256")
        new_digest = "" if new_value is None else _journal_digest(item, "new_sha256")
        if remove != (not bool(new_digest)):
            raise ValueError("shortcut removal fields are inconsistent")
        mutations.append(
            _ShortcutMutation(
                prepared=PreparedShortcut(
                    label=_journal_text(item, "label"),
                    profile=_journal_text(item, "profile"),
                    destination=destination,
                    existed=existed,
                    prior_sha256=prior_digest,
                    remove=remove,
                ),
                temporary=temporary,
                backup=backup,
                new_sha256=new_digest,
            )
        )
    return tuple(mutations)


def _journal_text(document: Mapping[object, object], field: str) -> str:
    value = document.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"journal field {field!r} must be non-empty text")
    return value


def _journal_digest(document: Mapping[object, object], field: str) -> str:
    value = _journal_text(document, field).lower()
    if not _HEX_DIGEST.fullmatch(value):
        raise ValueError(f"journal field {field!r} must be a SHA-256 digest")
    return value


def _terminal_failure_message(
    rollback: RollbackReport,
    *,
    cancelled: bool,
) -> str:
    if not rollback.completed:
        action = "was cancelled" if cancelled else "could not finish"
        return (
            f"Installation {action}, and some incomplete files could not be "
            "removed. Close VIPP and other programs, then choose Repair or try "
            "again. The setup details list the exact paths that were preserved."
        )
    if cancelled:
        return "Installation cancelled. No previous VIPP installation was changed."
    return (
        "VIPP could not be installed. Any previous VIPP installation was left "
        "unchanged."
    )


def _prune_state_history(state_root: Path) -> None:
    """Retain a bounded set of completed installer evidence directories."""

    runs_root = state_root / "runs"
    try:
        entries = tuple(runs_root.iterdir())
    except (FileNotFoundError, OSError):
        return
    protected = _journal_run_directories(state_root)
    candidates: list[tuple[int, Path]] = []
    for entry in entries:
        try:
            uuid.UUID(entry.name)
            metadata = entry.lstat()
        except (ValueError, OSError):
            continue
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or _metadata_is_reparse(metadata)
            or _path_key(entry) in protected
        ):
            continue
        candidates.append((metadata.st_mtime_ns, entry))
    candidates.sort(reverse=True)
    for _modified, entry in candidates[_MAX_RUN_DIRECTORIES:]:
        try:
            if _path_has_reparse_component(entry) or _tree_has_reparse(entry):
                continue
            shutil.rmtree(entry)
        except OSError:
            continue


def _journal_run_directories(state_root: Path) -> set[str]:
    protected: set[str] = set()
    directory = state_root / "transactions"
    try:
        entries = tuple(directory.glob("*.json"))
    except OSError:
        return protected
    for entry in entries:
        try:
            if entry.is_symlink() or _path_has_reparse_component(entry):
                continue
            document = json.loads(entry.read_text(encoding="utf-8"))
            value = (
                document.get("run_directory") if isinstance(document, dict) else None
            )
            if isinstance(value, str):
                run = Path(value)
                if _is_relative_to(run, state_root / "runs"):
                    protected.add(_path_key(run))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
    return protected


def _prune_stale_locks(directory: Path) -> None:
    try:
        entries = tuple(directory.glob("*.stale"))
    except OSError:
        return
    candidates: list[tuple[int, Path]] = []
    for entry in entries:
        try:
            metadata = entry.lstat()
        except OSError:
            continue
        if stat.S_ISREG(metadata.st_mode) and not _metadata_is_reparse(metadata):
            candidates.append((metadata.st_mtime_ns, entry))
    candidates.sort(reverse=True)
    for _modified, entry in candidates[_MAX_STALE_LOCKS:]:
        try:
            entry.unlink()
        except OSError:
            continue


class _TargetLock:
    def __init__(self, state_root: Path, target: Path, run_id: str) -> None:
        digest = hashlib.sha256(_path_key(target).encode("utf-8")).hexdigest()
        self.path = state_root / "locks" / f"{digest}.lock"
        self.run_id = run_id

    def __enter__(self) -> _TargetLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "schema": "napari-vipp-install-lock",
            "schema_version": 1,
            "run_id": self.run_id,
            "pid": os.getpid(),
            "created_at": datetime.now(UTC).isoformat(),
        }
        payload = json.dumps(document, sort_keys=True) + "\n"
        for _attempt in range(2):
            try:
                with self.path.open("x", encoding="utf-8") as stream:
                    stream.write(payload)
                return self
            except FileExistsError as exc:
                if self._recover_stale_lock():
                    continue
                raise ConcurrentInstallationError(
                    "Another VIPP installation is already using this location."
                ) from exc
        raise ConcurrentInstallationError(
            "Another VIPP installation claimed this location."
        )

    def __exit__(self, *_args: object) -> None:
        try:
            document = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(document, dict) and document.get("run_id") == self.run_id:
                self.path.unlink()
        except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
            pass

    def _recover_stale_lock(self) -> bool:
        try:
            original = self.path.read_bytes()
            document = json.loads(original.decode("utf-8"))
            if not isinstance(document, dict):
                return False
            pid = document.get("pid")
            stale_run = document.get("run_id")
            if (
                not isinstance(pid, int)
                or pid <= 0
                or not isinstance(stale_run, str)
                or not stale_run
                or _process_is_alive(pid)
            ):
                return False
            if self.path.read_bytes() != original:
                return False
            quarantine = self.path.with_name(
                f"{self.path.name}.{_SAFE_PATH_COMPONENT.sub('-', stale_run)}.stale"
            )
            os.replace(self.path, quarantine)
            _prune_stale_locks(self.path.parent)
            return True
        except (
            FileNotFoundError,
            OSError,
            UnicodeError,
            json.JSONDecodeError,
        ):
            return False


class _RunLog:
    def __init__(self, path: Path, now: Callable[[], datetime]) -> None:
        self.path = path
        self._now = now
        self._stream = None

    def __enter__(self) -> _RunLog:
        if self._stream is None or self._stream.closed:
            self._stream = self.path.open("a", encoding="utf-8", newline="\n")
        return self

    def __exit__(self, *_args: object) -> None:
        if self._stream is not None:
            try:
                self._stream.close()
            except OSError:
                pass

    def write(self, event: str, **details: object) -> None:
        if self._stream is None or self._stream.closed:
            raise RuntimeError("The installer log is not open.")
        document = {
            "time": self._now().astimezone(UTC).isoformat(),
            "event": event,
            **details,
        }
        self._stream.write(
            json.dumps(document, allow_nan=False, ensure_ascii=False, sort_keys=True)
            + "\n"
        )
        self._stream.flush()


def _inspection(
    kind: ManagedTargetKind,
    target: Path,
    release: ReleaseSpec,
    current_version: str | None,
    reason: str,
    *,
    target_preexisting: bool = False,
    record: OwnershipRecord | None = None,
    extra: tuple[object, ...] = (),
) -> ManagedTargetInspection:
    payload = {
        "kind": kind.value,
        "target": _path_key(target),
        "desired_version": release.version,
        "current_version": current_version,
        "target_preexisting": target_preexisting,
        "extra": extra,
    }
    fingerprint = _json_digest(payload)
    return ManagedTargetInspection(
        kind=kind,
        target=target,
        desired_version=release.version,
        current_version=current_version,
        reason=reason,
        fingerprint=fingerprint,
        target_preexisting=target_preexisting,
        record=record,
    )


def _structural_health_error(record: OwnershipRecord) -> str:
    environment = record.environment_root
    if not environment.is_dir() or environment.is_symlink():
        return "its private workspace is missing"
    if not (environment / "pyvenv.cfg").is_file():
        return "its private workspace is incomplete"
    python = environment / "Scripts" / "python.exe"
    if not python.is_file():
        return "its Python runtime is missing"
    launcher = _launcher_path(environment, record.track)
    if not launcher.is_file():
        return "its VIPP launcher is missing"
    package = next(
        (
            package
            for package in record.packages
            if package.normalized_name == _normalize_name(record.distribution)
        ),
        None,
    )
    if package is None or package.version != record.version:
        return "its recorded VIPP package does not match the installed version"
    return ""


def _validate_snapshot_binding(
    plan: InstallPlan,
    inspection: ManagedTargetInspection,
) -> None:
    snapshot = plan.discovery.filesystem
    if snapshot.target_exists != inspection.target_preexisting:
        raise PreparationError(
            "The installation folder changed after the plan was created. Check again."
        )
    if snapshot.managed_ownership is not None:
        record = inspection.record
        if (
            record is None
            or snapshot.managed_ownership.resolved_plan_id != record.resolved_plan_id
        ):
            raise PreparationError(
                "The managed installation changed after the plan was created."
            )


def _release_wheel_digest(
    release: ReleaseSpec,
    cancellation: object | None,
) -> str:
    if release.wheel_path is None:
        return ""
    path = release.wheel_path
    if not path.is_file():
        raise ResolutionError(f"The bundled VIPP wheel is missing: {path}")
    digest = _sha256_file(path, cancellation)
    if release.wheel_sha256 and digest != release.wheel_sha256:
        raise ResolutionError(
            "The bundled VIPP wheel failed its installer integrity check."
        )
    return digest


def _sha256_file(path: Path, cancellation: object | None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            _checkpoint(cancellation)
            block = stream.read(1024 * 1024)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def _plan_fingerprint(plan: InstallPlan) -> str:
    return hashlib.sha256(plan.to_json().encode("utf-8")).hexdigest()


def _resolution_fingerprint(
    *,
    plan_fingerprint: str,
    inspection: ManagedTargetInspection,
    packages: tuple[ResolvedPackage, ...],
    shortcuts: tuple[PreparedShortcut, ...],
    wheel_sha256: str,
    index_url: str,
    persistent_setup_path: Path | None,
    persistent_setup_sha256: str,
    persistent_setup_prior_sha256: str,
    working_directory: Path,
) -> str:
    return _json_digest(
        {
            "plan_fingerprint": plan_fingerprint,
            "target_fingerprint": inspection.fingerprint,
            "operation": inspection.kind.value,
            "wheel_sha256": wheel_sha256,
            "index_url": index_url,
            "persistent_setup": {
                "path": (
                    _path_key(persistent_setup_path)
                    if persistent_setup_path is not None
                    else None
                ),
                "sha256": persistent_setup_sha256,
                "prior_sha256": persistent_setup_prior_sha256,
            },
            "working_directory": _path_key(working_directory),
            "packages": [
                {
                    "name": package.name,
                    "version": package.version,
                    "source_url": package.source_url,
                    "sha256": package.sha256,
                    "requested": package.requested,
                }
                for package in packages
            ],
            "shortcuts": [shortcut.as_dict() for shortcut in shortcuts],
        }
    )


def _json_digest(document: object) -> str:
    payload = json.dumps(
        document,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, document: object) -> None:
    _atomic_text(
        path,
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{secrets.token_hex(8)}.tmp"
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        _replace_with_retry(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _replace_with_retry(source: Path, destination: Path) -> None:
    """Bound transient Windows scanner/share locks around an atomic replace."""

    for attempt in range(len(_ATOMIC_REPLACE_RETRY_DELAYS) + 1):
        try:
            os.replace(source, destination)
            return
        except OSError as exc:
            transient = os.name == "nt" and (
                getattr(exc, "winerror", None) in _WINDOWS_TRANSIENT_REPLACE_ERRORS
                or exc.errno in {13, 16}
            )
            if not transient or attempt == len(_ATOMIC_REPLACE_RETRY_DELAYS):
                raise
            time.sleep(_ATOMIC_REPLACE_RETRY_DELAYS[attempt])


def _validate_resolution_temp_capacity(
    plan: InstallPlan,
    *,
    state_root: Path,
) -> None:
    track_reserve = (
        _CUDA_RESOLUTION_TEMP_MIN_FREE_BYTES
        if plan.request.track is ComputeTrack.CUDA13
        else _CPU_RESOLUTION_TEMP_MIN_FREE_BYTES
    )
    required = min(plan.required_free_bytes, track_reserve)
    locations = (Path(tempfile.gettempdir()), state_root)
    checked_volumes: set[str] = set()
    for location in locations:
        probe = _nearest_existing_path(location)
        volume = os.path.normcase(os.path.splitdrive(os.path.abspath(probe))[0])
        volume_key = volume or _path_key(probe.anchor and Path(probe.anchor) or probe)
        if volume_key in checked_volumes:
            continue
        checked_volumes.add(volume_key)
        try:
            free = shutil.disk_usage(probe).free
        except OSError as exc:
            raise PreparationError(
                "Setup could not check free space for temporary downloads and "
                f"installer records at {probe}."
            ) from exc
        if free < required:
            required_gib = required / 1024**3
            # Round down so a near-threshold value can never be displayed as if
            # it met the requirement (for example, 4.999 GiB must not say 5.00).
            free_gib = (free * 100 // 1024**3) / 100
            raise PreparationError(
                f"Setup needs at least {required_gib:.0f} GiB of free disk space "
                f"on the volume containing {probe} for temporary downloads and "
                f"installer records; {free_gib:.2f} GiB is available."
            )


def _nearest_existing_path(path: Path) -> Path:
    current = Path(os.path.abspath(path))
    while not current.exists():
        parent = current.parent
        if parent == current:
            raise PreparationError(
                "Setup could not find a local drive for its temporary files."
            )
        current = parent
    _assert_direct_path(current, "temporary installer storage")
    return current


def _pip_environment(
    *,
    temp_directory: Path | None = None,
) -> dict[str, str]:
    excluded_python_variables = {
        "PYTHONHOME",
        "PYTHONIOENCODING",
        "PYTHONPATH",
        "PYTHONUTF8",
    }
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("PIP_")
        and key.upper() not in excluded_python_variables
    }
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    # pip renders ``--report -`` through Rich.  A windowed Windows installer
    # can otherwise leave the child on the legacy CP1252 code page, which makes
    # valid Unicode package metadata crash the resolver before it emits JSON.
    # Force both Python's UTF-8 mode and its standard-stream encoding; setting
    # both also makes this independent of inherited, case-varied environment
    # entries on Windows.
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    # PIP_CONFIG_FILE provides a second defence for pip versions that read a
    # site-level config even in isolated mode. All policy comes from argv.
    environment["PIP_CONFIG_FILE"] = os.devnull
    if temp_directory is not None:
        _assert_direct_path(temp_directory, "private package temporary")
        environment["TEMP"] = str(temp_directory)
        environment["TMP"] = str(temp_directory)
    return environment


def _checkpoint(cancellation: object | None) -> None:
    if _is_cancelled(cancellation):
        raise InstallCancelled("The VIPP installation was cancelled.")


def _is_cancelled(cancellation: object | None) -> bool:
    if cancellation is None:
        return False
    if isinstance(cancellation, CancellationToken):
        return cancellation.is_cancelled()
    if callable(cancellation):
        return bool(cancellation())
    is_set = getattr(cancellation, "is_set", None)
    if callable(is_set):
        return bool(is_set())
    is_cancelled = getattr(cancellation, "is_cancelled", None)
    if callable(is_cancelled):
        return bool(is_cancelled())
    raise TypeError("cancellation must be a token, Event-like object, or callback")


def _emit(
    callback: ProgressCallback | None,
    stage: ProgressStage,
    message: str,
    completed: int,
    total: int,
) -> None:
    if callback is None:
        return
    try:
        callback(InstallProgress(stage, message, completed, total))
    except Exception:
        # A display callback cannot corrupt an installation transaction.
        return


def _cancel_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt" and hasattr(signal, "CTRL_BREAK_EVENT"):
        try:
            process.send_signal(signal.CTRL_BREAK_EVENT)
            process.wait(timeout=2)
            return
        except (OSError, subprocess.TimeoutExpired):
            pass
    try:
        process.terminate()
        process.wait(timeout=2)
        return
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        process.kill()
        process.wait(timeout=2)
    except (OSError, subprocess.TimeoutExpired):
        pass


class _WindowsKillOnCloseJob:
    """Own a Windows process tree and terminate every descendant on close."""

    def __init__(self, process: subprocess.Popen[str]) -> None:
        if os.name != "nt":
            raise OSError("Windows Job Objects are only available on Windows.")
        import ctypes
        from ctypes import wintypes

        class _IoCounters(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class _BasicLimitInformation(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class _ExtendedLimitInformation(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _BasicLimitInformation),
                ("IoInfo", _IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
        kernel.CreateJobObjectW.restype = wintypes.HANDLE
        kernel.SetInformationJobObject.argtypes = (
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        )
        kernel.SetInformationJobObject.restype = wintypes.BOOL
        kernel.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        kernel.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel.CloseHandle.restype = wintypes.BOOL
        handle = kernel.CreateJobObjectW(None, None)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        self._kernel = kernel
        self._handle = handle
        try:
            information = _ExtendedLimitInformation()
            information.BasicLimitInformation.LimitFlags = 0x00002000
            if not kernel.SetInformationJobObject(
                handle,
                9,
                ctypes.byref(information),
                ctypes.sizeof(information),
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            if not kernel.AssignProcessToJobObject(
                handle,
                wintypes.HANDLE(int(process._handle)),  # type: ignore[attr-defined]
            ):
                raise ctypes.WinError(ctypes.get_last_error())
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle:
            self._handle = None
            self._kernel.CloseHandle(handle)


def _resume_windows_process(pid: int) -> None:
    """Resume a Popen process launched suspended after assigning its Job."""

    import ctypes
    from ctypes import wintypes

    class _ThreadEntry32(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ThreadID", wintypes.DWORD),
            ("th32OwnerProcessID", wintypes.DWORD),
            ("tpBasePri", wintypes.LONG),
            ("tpDeltaPri", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateToolhelp32Snapshot.argtypes = (wintypes.DWORD, wintypes.DWORD)
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel.Thread32First.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(_ThreadEntry32),
    )
    kernel.Thread32First.restype = wintypes.BOOL
    kernel.Thread32Next.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(_ThreadEntry32),
    )
    kernel.Thread32Next.restype = wintypes.BOOL
    kernel.OpenThread.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenThread.restype = wintypes.HANDLE
    kernel.ResumeThread.argtypes = (wintypes.HANDLE,)
    kernel.ResumeThread.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.CloseHandle.restype = wintypes.BOOL
    snapshot = kernel.CreateToolhelp32Snapshot(0x00000004, 0)
    invalid_handle = ctypes.c_void_p(-1).value
    if not snapshot or int(snapshot) == invalid_handle:
        raise ctypes.WinError(ctypes.get_last_error())
    resumed = False
    try:
        entry = _ThreadEntry32()
        entry.dwSize = ctypes.sizeof(entry)
        found = bool(kernel.Thread32First(snapshot, ctypes.byref(entry)))
        while found:
            if entry.th32OwnerProcessID == pid:
                thread = kernel.OpenThread(0x0002, False, entry.th32ThreadID)
                if not thread:
                    raise ctypes.WinError(ctypes.get_last_error())
                try:
                    if kernel.ResumeThread(thread) == 0xFFFFFFFF:
                        raise ctypes.WinError(ctypes.get_last_error())
                    resumed = True
                    break
                finally:
                    kernel.CloseHandle(thread)
            found = bool(kernel.Thread32Next(snapshot, ctypes.byref(entry)))
    finally:
        kernel.CloseHandle(snapshot)
    if not resumed:
        raise OSError("Windows could not find the suspended installer process thread.")


def _process_is_alive(pid: int) -> bool:
    if pid == os.getpid():
        return True
    if os.name == "nt":
        try:
            import ctypes

            process_query = 0x1000
            still_active = 259
            kernel = ctypes.windll.kernel32
            kernel.OpenProcess.argtypes = (
                ctypes.c_ulong,
                ctypes.c_int,
                ctypes.c_ulong,
            )
            kernel.OpenProcess.restype = ctypes.c_void_p
            kernel.GetExitCodeProcess.argtypes = (
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_ulong),
            )
            kernel.GetExitCodeProcess.restype = ctypes.c_int
            kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
            kernel.CloseHandle.restype = ctypes.c_int
            handle = kernel.OpenProcess(process_query, False, pid)
            if not handle:
                return False
            try:
                exit_code = ctypes.c_ulong()
                if not kernel.GetExitCodeProcess(
                    handle,
                    ctypes.byref(exit_code),
                ):
                    return True
                return exit_code.value == still_active
            finally:
                kernel.CloseHandle(handle)
        except (AttributeError, OSError, ValueError):
            return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        return True
    return True


def _redacted_argv(argv: Sequence[str]) -> list[str]:
    return [_redact_url(str(value)) for value in argv]


def _redact_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"}:
        return value
    hostname = parsed.hostname or "approved-host"
    try:
        port_value = parsed.port
    except ValueError:
        port_value = None
    port = f":{port_value}" if port_value else ""
    artifact = Path(unquote(parsed.path)).name
    safe_path = f"/{artifact}" if artifact else ""
    return urlunsplit((parsed.scheme, hostname + port, safe_path, "", ""))


def _redact_text(value: str) -> str:
    value = re.sub(
        r"https?://[^\s\"'<>]+",
        lambda match: _redact_url(match.group(0)),
        value,
        flags=re.IGNORECASE,
    )
    return re.sub(
        r"(?i)\b(token|password|secret|authorization)(\s*[:=]\s*)\S+",
        lambda match: f"{match.group(1)}{match.group(2)}[redacted]",
        value,
    )


def _public_source_identity(value: str) -> str:
    parsed = urlsplit(value)
    artifact = Path(unquote(parsed.path)).name or "package"
    if parsed.scheme.casefold() == "file":
        return f"Bundled file: {artifact}"
    if parsed.scheme.casefold() == "https":
        return f"Approved host {parsed.hostname or 'unknown'}: {artifact}"
    return "Unrecognized package source"


def _without_fragment(value: str) -> str:
    parsed = urlsplit(value)
    if not parsed.scheme:
        return value.partition("#")[0]
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, parsed.query, ""))


def _compare_versions(left: str, right: str) -> int:
    left_key = _version_key(left)
    right_key = _version_key(right)
    if left_key is None or right_key is None:
        return (left.casefold() > right.casefold()) - (
            left.casefold() < right.casefold()
        )
    return (left_key > right_key) - (left_key < right_key)


def _version_key(value: str) -> tuple[tuple[int, ...], int, int] | None:
    match = re.fullmatch(
        r"\s*(\d+(?:\.\d+)*)(?:(a|b|rc)(\d+))?(?:\.post(\d+))?\s*",
        value,
        re.IGNORECASE,
    )
    if match is None:
        return None
    release = tuple(int(part) for part in match.group(1).split("."))
    release = (*release, *(0 for _ in range(max(0, 4 - len(release)))))
    phase = match.group(2)
    number = int(match.group(3) or match.group(4) or 0)
    rank = {"a": -3, "b": -2, "rc": -1, None: 0}[phase.casefold() if phase else None]
    if match.group(4) is not None:
        rank = 1
    return release, rank, number


def _normalize_name(value: str) -> str:
    return value.strip().lower().replace("_", "-").replace(".", "-")


def _path_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))


def _assert_direct_path(path: Path, label: str) -> None:
    if _path_has_reparse_component(path):
        raise InstallerEngineError(
            f"The {label} path contains a symbolic link or Windows reparse point: "
            f"{path}"
        )


def _path_has_reparse_component(path: Path) -> bool:
    absolute = Path(os.path.abspath(path))
    parts = absolute.parts
    if not parts:
        return True
    current = Path(parts[0])
    for index, part in enumerate(parts):
        if index:
            current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            return False
        except OSError:
            return True
        if _metadata_is_reparse(metadata):
            return True
    return False


def _is_empty_managed_residue(target: Path) -> bool:
    """Admit only empty private directories left by a killed uninstall."""

    metadata = target / OWNERSHIP_DIRECTORY
    environments = managed_environments_root(target)
    try:
        root_entries = tuple(target.iterdir())
        if len(root_entries) != 1 or not _same_path(root_entries[0], metadata):
            return False
        for path in (metadata, environments):
            inspected = path.lstat()
            if (
                not stat.S_ISDIR(inspected.st_mode)
                or _metadata_is_reparse(inspected)
                or _path_has_reparse_component(path)
            ):
                return False
        metadata_entries = tuple(metadata.iterdir())
        return (
            len(metadata_entries) == 1
            and _same_path(metadata_entries[0], environments)
            and next(environments.iterdir(), None) is None
        )
    except (FileNotFoundError, OSError):
        return False


def _metadata_is_reparse(metadata: os.stat_result) -> bool:
    return stat.S_ISLNK(metadata.st_mode) or bool(
        getattr(metadata, "st_file_attributes", 0) & _FILE_ATTRIBUTE_REPARSE_POINT
    )


def _tree_has_reparse(directory: Path) -> bool:
    """Fail closed if an owned removal tree contains a redirecting entry."""

    pending = [directory]
    while pending:
        current = pending.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    metadata = entry.stat(follow_symlinks=False)
                    if _metadata_is_reparse(metadata):
                        return True
                    if stat.S_ISDIR(metadata.st_mode):
                        pending.append(Path(entry.path))
        except OSError:
            return True
    return False


def _same_path(left: Path, right: Path) -> bool:
    return _path_key(left) == _path_key(right)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        Path(os.path.abspath(path)).relative_to(Path(os.path.abspath(parent)))
    except ValueError:
        return False
    return True


__all__ = [
    "AuthorizationError",
    "CancellationToken",
    "CommandFailed",
    "CommandResult",
    "ConcurrentInstallationError",
    "ExecutionAuthorization",
    "InstallCancelled",
    "InstallProgress",
    "InstallResult",
    "InstallStatus",
    "InstallerEngineError",
    "ManagedInstallerEngine",
    "ManagedTargetInspection",
    "ManagedTargetKind",
    "PackageDisposition",
    "PackageReviewChange",
    "PreparationError",
    "PreparedShortcut",
    "PreparedTransaction",
    "ProgressStage",
    "ResolvedPackage",
    "ResolutionError",
    "ReviewSummary",
    "RollbackReport",
    "StalePreparedTransaction",
    "SubprocessCommandRunner",
    "inspect_managed_target",
]
