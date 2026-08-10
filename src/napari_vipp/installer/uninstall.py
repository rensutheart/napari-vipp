"""Ownership-safe Windows uninstall and Apps & Features support.

This module deliberately does not infer ownership from names or locations.  A
validated :class:`~napari_vipp.installer.ownership.OwnershipRecord`, the exact
bytes of its manifest, and the hashes recorded in it are the only authority to
delete an installed VIPP environment or shortcut.

The registry and shortcut-target interfaces are injectable so the complete
contract can be exercised on non-Windows build and test hosts.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import functools
import hashlib
import json
import os
import secrets
import shutil
import stat
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from napari_vipp.installer.ownership import (
    OWNERSHIP_DIRECTORY,
    OwnedEnvironment,
    OwnershipRecord,
    OwnershipState,
    inspect_ownership,
    managed_environments_root,
    ownership_path,
    parse_ownership_record_bytes,
)

_CANDIDATE_MARKER = ".vipp-install-candidate.json"
_FILE_ATTRIBUTE_REPARSE_POINT = 0x0400
_HEX_DIGITS = frozenset("0123456789abcdef")
_SAFE_VERSION_CHARACTERS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._+-"
)
_REGISTRY_ROOT = r"Software\Microsoft\Windows\CurrentVersion\Uninstall"
CPU_REGISTRY_KEY = rf"{_REGISTRY_ROOT}\VIPP-CPU"
CUDA13_REGISTRY_KEY = rf"{_REGISTRY_ROOT}\VIPP-CUDA13"
DEFAULT_REGISTRY_KEY = CPU_REGISTRY_KEY
_REGISTRY_BINDING_WRITE_ORDER = (
    "VippManagedRoot",
    "VippInstallationId",
    "VippManifestSha256",
    "VippUninstallerSha256",
)
_UNINSTALL_JOURNAL_SCHEMA = "napari-vipp-uninstall-transaction"
_UNINSTALL_JOURNAL_VERSION = 1
_UNINSTALL_JOURNAL_PHASES = (
    "prepared",
    "payload_removed",
    "manifest_removed",
    "registry_removed",
    "cache_removed",
)
_UNINSTALL_JOURNAL_PHASE_INDEX = {
    phase: index for index, phase in enumerate(_UNINSTALL_JOURNAL_PHASES)
}
_MAX_UNINSTALL_JOURNALS_PER_ROOT = 8
_WINDOWS_TRANSIENT_REPLACE_ERRORS = frozenset({5, 32, 33})
_ATOMIC_REPLACE_RETRY_DELAYS = (0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.0)
_SETUP_SINGLE_INSTANCE_MUTEX = r"Local\VIPP.Setup.SingleInstance"

type RegistryScalar = str | int


class UninstallError(RuntimeError):
    """Base error for a refused or failed managed uninstall operation."""


class UninstallPreparationError(UninstallError):
    """The current installation cannot be safely prepared for uninstall."""


class UninstallAuthorizationError(UninstallError):
    """The explicit uninstall confirmation is missing, stale, or reused."""


class RegistryOwnershipError(UninstallError):
    """The Apps & Features entry is not owned by this installation."""


class UninstallStatus(StrEnum):
    """Terminal state of an authorized uninstall attempt."""

    COMPLETED = "completed"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True, slots=True)
class RegistryRegistrationPlan:
    """Exact HKCU Apps & Features values bound to one ownership manifest."""

    key: str
    values: tuple[tuple[str, RegistryScalar], ...]
    managed_root: Path
    installation_id: str
    manifest_sha256: str
    uninstaller_path: Path
    uninstaller_sha256: str

    def __post_init__(self) -> None:
        key = _normalize_registry_key(self.key)
        if not key.casefold().startswith((_REGISTRY_ROOT + "\\").casefold()):
            raise ValueError("The uninstall registry key is outside HKCU Uninstall.")
        if not _is_sha256(self.manifest_sha256):
            raise ValueError("manifest_sha256 must contain 64 hexadecimal digits.")
        if not _is_sha256(self.uninstaller_sha256):
            raise ValueError("uninstaller_sha256 must contain 64 hexadecimal digits.")
        managed_root = Path(self.managed_root)
        uninstaller_path = Path(self.uninstaller_path)
        if not managed_root.is_absolute() or not uninstaller_path.is_absolute():
            raise ValueError(
                "Registry installation and uninstaller paths must be absolute."
            )
        if _same_path(uninstaller_path, managed_root) or _is_descendant(
            uninstaller_path,
            managed_root,
        ):
            raise ValueError("The persistent uninstaller must be outside managed_root.")
        names = [name.casefold() for name, _value in self.values]
        if len(names) != len(set(names)):
            raise ValueError("Registry value names must be unique.")
        value_map = {name.casefold(): value for name, value in self.values}
        required = {
            "vippmanagedroot": str(self.managed_root),
            "vippinstallationid": self.installation_id,
            "vippmanifestsha256": self.manifest_sha256.lower(),
            "vippuninstallersha256": self.uninstaller_sha256.lower(),
        }
        for name, expected in required.items():
            actual = value_map.get(name)
            if not isinstance(actual, str):
                raise ValueError(f"The registry binding value {name!r} is missing.")
            if name == "vippmanagedroot":
                if not _same_path(Path(actual), Path(expected)):
                    raise ValueError("The registry managed-root binding is invalid.")
            elif not secrets.compare_digest(actual.casefold(), expected.casefold()):
                raise ValueError(f"The registry binding value {name!r} is invalid.")
        uninstall_string = value_map.get("uninstallstring")
        expected_uninstall_string = subprocess.list2cmdline(
            (
                str(uninstaller_path),
                "--uninstall",
                "--managed-root",
                str(managed_root),
            )
        )
        if uninstall_string != expected_uninstall_string:
            raise ValueError(
                "UninstallString is not the exact hash-bound VIPP uninstall command."
            )
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "managed_root", managed_root)
        object.__setattr__(self, "uninstaller_path", uninstaller_path)
        object.__setattr__(self, "manifest_sha256", self.manifest_sha256.lower())
        object.__setattr__(
            self,
            "uninstaller_sha256",
            self.uninstaller_sha256.lower(),
        )

    @property
    def value_map(self) -> dict[str, RegistryScalar]:
        """Return a fresh mapping suitable for a registry backend."""

        return dict(self.values)

    @property
    def binding_values(self) -> dict[str, str]:
        """Return the values that prove ownership of the registry key."""

        return {
            "VippManagedRoot": str(self.managed_root),
            "VippInstallationId": self.installation_id,
            "VippManifestSha256": self.manifest_sha256,
            "VippUninstallerSha256": self.uninstaller_sha256,
        }


@dataclass(frozen=True, slots=True)
class PersistentUninstaller:
    """A hash-bound persistent uninstaller copied outside the managed root."""

    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class PreparedOwnedEnvironment:
    """One active or retired environment reviewed for removal."""

    path: Path
    marker_path: Path
    marker_sha256: str
    active: bool
    exists: bool
    quarantine_path: Path | None = None


@dataclass(frozen=True, slots=True)
class PreparedOwnedShortcut:
    """One shortcut reviewed against its hash, root, and optional target."""

    path: Path
    sha256: str
    expected_target: Path | None
    exists: bool


@dataclass(frozen=True, slots=True)
class PreparedUninstall:
    """Immutable review result that must be explicitly authorized."""

    managed_root: Path
    manifest_path: Path
    manifest_sha256: str
    installation_id: str
    environment_items: tuple[PreparedOwnedEnvironment, ...]
    shortcut_items: tuple[PreparedOwnedShortcut, ...]
    shortcut_roots: tuple[Path, ...]
    registry_plan: RegistryRegistrationPlan | None
    uninstaller_path: Path | None
    uninstaller_sha256: str
    fingerprint: str
    _record: OwnershipRecord = field(repr=False, compare=False, hash=False)
    journal_path: Path | None = None
    resume_phase: str = ""
    warnings: tuple[str, ...] = ()

    @property
    def record(self) -> OwnershipRecord:
        """Return the validated ownership record used to prepare the review."""

        return self._record


@dataclass(frozen=True, slots=True)
class UninstallIssue:
    """One exact path that could not be safely removed and why."""

    path: Path
    operation: str
    error: str


@dataclass(frozen=True, slots=True)
class DeferredSelfDelete:
    """Command the launcher starts only after displaying the final result."""

    target: Path
    expected_sha256: str
    wait_for_pid: int
    argv: tuple[str, ...]
    journal_path: Path | None = None
    journal_sha256: str = ""


@dataclass(frozen=True, slots=True)
class UninstallResult:
    """Complete, user-readable outcome of an uninstall attempt."""

    status: UninstallStatus
    managed_root: Path
    removed_paths: tuple[Path, ...]
    preserved_paths: tuple[Path, ...]
    issues: tuple[UninstallIssue, ...]
    deferred_self_delete: DeferredSelfDelete | None
    message: str
    retry_via_apps: bool = True

    @property
    def completed(self) -> bool:
        return self.status is UninstallStatus.COMPLETED


class RegistryBackend(Protocol):
    """Minimal HKCU registry surface required by this module."""

    def read_values(self, key: str) -> Mapping[str, RegistryScalar] | None:
        """Return all values, or ``None`` when the key does not exist."""

    def write_values(
        self,
        key: str,
        values: Mapping[str, RegistryScalar],
    ) -> None:
        """Create or replace values at ``key`` under HKCU."""

    def delete_key(self, key: str) -> None:
        """Delete one non-recursive key, ignoring an absent key."""


class ShortcutTargetReader(Protocol):
    """Read the executable target stored in a Windows shortcut."""

    def __call__(self, shortcut: Path) -> Path:
        """Return the shortcut target as an absolute path."""


class DeferredDeleteScheduler(Protocol):
    """Start a detached deferred-delete command."""

    def __call__(self, request: DeferredSelfDelete) -> None:
        """Schedule ``request`` and return without waiting."""


class UninstallAuthorization:
    """Single-use explicit permission bound to one prepared fingerprint."""

    __slots__ = ("_fingerprint", "_lock", "_used")

    def __init__(self, fingerprint: str) -> None:
        self._fingerprint = fingerprint
        self._used = False
        self._lock = threading.Lock()

    def consume(self, fingerprint: str) -> None:
        """Atomically consume this authorization for the matching review."""

        with self._lock:
            if self._used:
                raise UninstallAuthorizationError(
                    "This uninstall confirmation has already been used."
                )
            if not secrets.compare_digest(self._fingerprint, fingerprint):
                raise UninstallAuthorizationError(
                    "This uninstall confirmation belongs to a different review."
                )
            self._used = True


class WindowsRegistryBackend:
    """HKCU registry backend; importing the module remains cross-platform."""

    def read_values(self, key: str) -> Mapping[str, RegistryScalar] | None:
        import winreg

        normalized = _normalize_registry_key(key)
        try:
            handle = winreg.OpenKey(winreg.HKEY_CURRENT_USER, normalized)
        except FileNotFoundError:
            return None
        with handle:
            values: dict[str, RegistryScalar] = {}
            index = 0
            while True:
                try:
                    name, value, _kind = winreg.EnumValue(handle, index)
                except OSError:
                    break
                if isinstance(value, (str, int)):
                    values[name] = value
                index += 1
            return values

    def write_values(
        self,
        key: str,
        values: Mapping[str, RegistryScalar],
    ) -> None:
        normalized = _normalize_registry_key(key)
        previous = self.read_values(normalized)
        try:
            self._write_values_in_place(normalized, values)
        except Exception as exc:
            try:
                if previous is None:
                    self.delete_key(normalized)
                else:
                    self._write_values_in_place(normalized, previous)
            except Exception as rollback_exc:
                raise OSError(
                    "The Apps & Features update and its immediate rollback both "
                    f"failed: {rollback_exc}"
                ) from exc
            raise

    @staticmethod
    def _write_values_in_place(
        normalized: str,
        values: Mapping[str, RegistryScalar],
    ) -> None:
        import winreg

        with winreg.CreateKeyEx(
            winreg.HKEY_CURRENT_USER,
            normalized,
            access=winreg.KEY_SET_VALUE | winreg.KEY_QUERY_VALUE,
        ) as handle:
            existing_names: list[str] = []
            index = 0
            while True:
                try:
                    name, _value, _kind = winreg.EnumValue(handle, index)
                except OSError:
                    break
                existing_names.append(name)
                index += 1
            desired_names = {name.casefold() for name in values}
            ordered_names = [
                name
                for preferred in _REGISTRY_BINDING_WRITE_ORDER
                for name in values
                if name.casefold() == preferred.casefold()
            ]
            ordered_names.extend(
                name
                for name in values
                if name.casefold()
                not in {preferred.casefold() for preferred in ordered_names}
            )
            for name in ordered_names:
                value = values[name]
                kind = winreg.REG_DWORD if isinstance(value, int) else winreg.REG_SZ
                winreg.SetValueEx(handle, name, 0, kind, value)
            # Obsolete values are removed only after every desired value exists,
            # keeping the previous entry launchable throughout most of the swap.
            for name in existing_names:
                if name.casefold() not in desired_names:
                    winreg.DeleteValue(handle, name)

    def delete_key(self, key: str) -> None:
        import winreg

        try:
            winreg.DeleteKey(winreg.HKEY_CURRENT_USER, _normalize_registry_key(key))
        except FileNotFoundError:
            return


def _uninstall_journal_path(
    uninstaller_path: Path | None,
    managed_root: Path,
    installation_id: str,
) -> Path | None:
    if uninstaller_path is None:
        return None
    executable = Path(uninstaller_path)
    if not executable.is_absolute() or not managed_root.is_absolute():
        return None
    try:
        normalized_installation_id = str(uuid.UUID(installation_id))
    except (AttributeError, TypeError, ValueError):
        return None
    root_digest = hashlib.sha256(_path_key(managed_root).encode("utf-8")).hexdigest()
    return executable.parent / (
        f".vipp-uninstall-{root_digest}-{normalized_installation_id}.json"
    )


def _uninstall_journal_prefix(managed_root: Path) -> str:
    root_digest = hashlib.sha256(_path_key(managed_root).encode("utf-8")).hexdigest()
    return f".vipp-uninstall-{root_digest}-"


def _matching_uninstall_journals(
    uninstaller_path: Path,
    managed_root: Path,
) -> tuple[Path, ...]:
    """Find a bounded set of adjacent generation-bound recovery records."""

    executable = Path(uninstaller_path)
    if not executable.is_absolute() or not managed_root.is_absolute():
        return ()
    parent = executable.parent
    _require_direct_path(parent, "cached setup folder")
    prefix = _uninstall_journal_prefix(managed_root)
    matches: list[Path] = []
    try:
        with os.scandir(parent) as entries:
            for entry in entries:
                name = entry.name
                if not name.startswith(prefix) or not name.endswith(".json"):
                    continue
                metadata = entry.stat(follow_symlinks=False)
                if _metadata_is_reparse(metadata) or not stat.S_ISREG(metadata.st_mode):
                    raise UninstallPreparationError(
                        "An uninstall recovery record is redirected and was "
                        f"preserved: {entry.path}"
                    )
                matches.append(Path(entry.path))
                if len(matches) > _MAX_UNINSTALL_JOURNALS_PER_ROOT:
                    raise UninstallPreparationError(
                        "Too many uninstall recovery records target this folder. "
                        "Nothing was changed."
                    )
    except FileNotFoundError:
        return ()
    except OSError as exc:
        raise UninstallPreparationError(
            f"The uninstall recovery folder could not be inspected: {exc}"
        ) from exc
    return tuple(sorted(matches, key=_path_key))


def _find_uninstall_journal(
    uninstaller_path: Path,
    managed_root: Path,
) -> Path | None:
    matches = _matching_uninstall_journals(uninstaller_path, managed_root)
    if not matches:
        return None
    if len(matches) != 1:
        raise UninstallPreparationError(
            "More than one uninstall recovery generation targets this folder. "
            "Nothing was changed."
        )
    return matches[0]


def _registry_plan_document(
    plan: RegistryRegistrationPlan | None,
) -> dict[str, object] | None:
    if plan is None:
        return None
    return {
        "key": plan.key,
        "values": [[name, value] for name, value in plan.values],
        "managed_root": str(plan.managed_root),
        "installation_id": plan.installation_id,
        "manifest_sha256": plan.manifest_sha256,
        "uninstaller_path": str(plan.uninstaller_path),
        "uninstaller_sha256": plan.uninstaller_sha256,
    }


def _environment_plan_document(
    items: tuple[PreparedOwnedEnvironment, ...],
) -> list[dict[str, object]]:
    return [
        {
            "path": str(item.path),
            "marker_path": str(item.marker_path),
            "marker_sha256": item.marker_sha256,
            "active": item.active,
            "quarantine_path": (
                str(item.quarantine_path) if item.quarantine_path is not None else None
            ),
        }
        for item in items
    ]


def _environment_quarantine_path(
    record: OwnershipRecord,
    environment_path: Path,
) -> Path:
    installation_id = str(uuid.UUID(record.installation_id))
    path_digest = hashlib.sha256(
        _path_key(environment_path).encode("utf-8")
    ).hexdigest()[:24]
    return managed_environments_root(record.managed_root) / (
        f".vipp-uninstall-{installation_id}-{path_digest}"
    )


def _record_environment_plan_document(
    record: OwnershipRecord,
) -> list[dict[str, object]]:
    owned = (
        OwnedEnvironment(record.environment_root, record.environment_marker_sha256),
        *record.retired_environments,
    )
    return [
        {
            "path": str(environment.path),
            "marker_path": str(environment.path / _CANDIDATE_MARKER),
            "marker_sha256": environment.marker_sha256,
            "active": index == 0,
            "quarantine_path": str(
                _environment_quarantine_path(record, environment.path)
            ),
        }
        for index, environment in enumerate(owned)
    ]


def _shortcut_plan_document(
    items: tuple[PreparedOwnedShortcut, ...],
) -> list[dict[str, object]]:
    return [
        {
            "path": str(item.path),
            "sha256": item.sha256,
            "target": (
                str(item.expected_target) if item.expected_target is not None else None
            ),
        }
        for item in items
    ]


def _record_shortcut_plan_document(
    record: OwnershipRecord,
) -> list[dict[str, object]]:
    return [
        {
            "path": str(shortcut.path),
            "sha256": shortcut.sha256,
            "target": (str(shortcut.target) if shortcut.target is not None else None),
        }
        for shortcut in record.shortcuts
    ]


def _uninstall_journal_document(
    prepared: PreparedUninstall,
    *,
    phase: str,
    manifest_payload: bytes,
) -> dict[str, object]:
    if phase not in _UNINSTALL_JOURNAL_PHASES:
        raise ValueError(f"Unsupported uninstall journal phase: {phase}")
    if hashlib.sha256(manifest_payload).hexdigest() != prepared.manifest_sha256:
        raise UninstallPreparationError(
            "The ownership manifest changed before uninstall recovery was saved."
        )
    return {
        "schema": _UNINSTALL_JOURNAL_SCHEMA,
        "schema_version": _UNINSTALL_JOURNAL_VERSION,
        "phase": phase,
        "managed_root": str(prepared.managed_root),
        "installation_id": prepared.installation_id,
        "track": prepared.record.track.value,
        "manifest_path": str(prepared.manifest_path),
        "manifest_sha256": prepared.manifest_sha256,
        "manifest_payload_base64": base64.b64encode(manifest_payload).decode("ascii"),
        "shortcut_roots": [str(root) for root in prepared.shortcut_roots],
        "environments": _environment_plan_document(prepared.environment_items),
        "shortcuts": _shortcut_plan_document(prepared.shortcut_items),
        "registry": _registry_plan_document(prepared.registry_plan),
        "uninstaller_path": (
            str(prepared.uninstaller_path)
            if prepared.uninstaller_path is not None
            else None
        ),
        "uninstaller_sha256": prepared.uninstaller_sha256 or None,
    }


def _read_uninstall_journal(
    path: Path,
    *,
    managed_root: Path,
    shortcut_roots: tuple[Path, ...],
    expected_record: OwnershipRecord | None,
    expected_uninstaller: Path | None = None,
    allow_missing_uninstaller: bool = False,
) -> dict[str, object]:
    try:
        _require_direct_path(path, "uninstall recovery record")
        metadata = path.lstat()
        if _metadata_is_reparse(metadata) or not stat.S_ISREG(metadata.st_mode):
            raise ValueError("the uninstall recovery record is redirected")
        if metadata.st_size > 1024 * 1024:
            raise ValueError("the uninstall recovery record is unexpectedly large")
        document = _load_strict_json(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError("the uninstall recovery record must be an object")
        expected_keys = {
            "schema",
            "schema_version",
            "phase",
            "managed_root",
            "installation_id",
            "track",
            "manifest_path",
            "manifest_sha256",
            "manifest_payload_base64",
            "shortcut_roots",
            "environments",
            "shortcuts",
            "registry",
            "uninstaller_path",
            "uninstaller_sha256",
        }
        if set(document) != expected_keys:
            raise ValueError("the uninstall recovery fields are invalid")
        if document.get("schema") != _UNINSTALL_JOURNAL_SCHEMA:
            raise ValueError("the uninstall recovery schema is not recognized")
        if document.get("schema_version") != _UNINSTALL_JOURNAL_VERSION:
            raise ValueError("the uninstall recovery version is not supported")
        phase = document.get("phase")
        if phase not in _UNINSTALL_JOURNAL_PHASES:
            raise ValueError("the uninstall recovery phase is invalid")
        recorded_root = Path(_journal_text(document, "managed_root"))
        if not _same_path(recorded_root, managed_root):
            raise ValueError("the uninstall recovery record targets another folder")
        manifest_path = Path(_journal_text(document, "manifest_path"))
        if not _same_path(manifest_path, ownership_path(managed_root)):
            raise ValueError("the uninstall recovery manifest path is invalid")
        manifest_sha256 = _journal_digest(document, "manifest_sha256")
        payload_text = _journal_text(document, "manifest_payload_base64")
        manifest_payload = base64.b64decode(payload_text, validate=True)
        if hashlib.sha256(manifest_payload).hexdigest() != manifest_sha256:
            raise ValueError("the uninstall recovery manifest hash is invalid")
        record = parse_ownership_record_bytes(
            manifest_payload,
            managed_root=managed_root,
        )
        if record.installation_id != _journal_text(document, "installation_id"):
            raise ValueError("the uninstall recovery installation ID is invalid")
        if record.track.value != _journal_text(document, "track"):
            raise ValueError("the uninstall recovery CPU/GPU track is invalid")
        if expected_record is not None and record != expected_record:
            raise ValueError("the uninstall recovery ownership record changed")
        roots_value = document.get("shortcut_roots")
        if not isinstance(roots_value, list) or not all(
            isinstance(value, str) for value in roots_value
        ):
            raise ValueError("the uninstall recovery shortcut roots are invalid")
        recorded_roots = tuple(Path(value) for value in roots_value)
        if len(recorded_roots) != len(shortcut_roots) or any(
            not _same_path(actual, expected)
            for actual, expected in zip(
                recorded_roots,
                shortcut_roots,
                strict=True,
            )
        ):
            raise ValueError("the uninstall recovery shortcut roots changed")
        uninstaller_value = document.get("uninstaller_path")
        if not isinstance(uninstaller_value, str) or not uninstaller_value:
            raise ValueError("the uninstall recovery cached setup path is missing")
        uninstaller_path = Path(uninstaller_value)
        if (
            record.uninstaller_path is None
            or not _same_path(uninstaller_path, record.uninstaller_path)
            or (
                expected_uninstaller is not None
                and not _same_path(uninstaller_path, expected_uninstaller)
            )
            or _uninstall_journal_path(
                uninstaller_path,
                managed_root,
                record.installation_id,
            )
            != path
        ):
            raise ValueError("the uninstall recovery cached setup path is invalid")
        uninstaller_sha256 = _journal_digest(document, "uninstaller_sha256")
        if uninstaller_sha256 != record.uninstaller_sha256:
            raise ValueError("the uninstall recovery cached setup hash is invalid")
        if _path_exists(uninstaller_path):
            _verify_regular_file_hash(
                uninstaller_path,
                uninstaller_sha256,
                "uninstall recovery cached setup program",
            )
        elif not allow_missing_uninstaller:
            raise ValueError("the uninstall recovery cached setup program is missing")
        expected_registry = (
            registry_plan_from_record(record, manifest_sha256)
            if record.registry_key
            else None
        )
        if not _strict_json_equal(
            document.get("environments"),
            _record_environment_plan_document(record),
        ):
            raise ValueError("the uninstall recovery environment binding is invalid")
        if not _strict_json_equal(
            document.get("shortcuts"),
            _record_shortcut_plan_document(record),
        ):
            raise ValueError("the uninstall recovery shortcut binding is invalid")
        if not _strict_json_equal(
            document.get("registry"),
            _registry_plan_document(expected_registry),
        ):
            raise ValueError("the uninstall recovery registry binding is invalid")
        document["_record"] = record
        return document
    except (
        OSError,
        UnicodeError,
        binascii.Error,
        json.JSONDecodeError,
        TypeError,
        ValueError,
        UninstallPreparationError,
    ) as exc:
        raise UninstallPreparationError(
            f"The uninstall recovery record is invalid: {exc}"
        ) from exc


def _write_uninstall_journal(
    path: Path,
    prepared: PreparedUninstall,
    *,
    phase: str,
) -> None:
    if phase not in _UNINSTALL_JOURNAL_PHASE_INDEX:
        raise ValueError(f"Unsupported uninstall journal phase: {phase}")
    if _path_exists(path):
        existing = _read_uninstall_journal(
            path,
            managed_root=prepared.managed_root,
            shortcut_roots=prepared.shortcut_roots,
            expected_record=prepared.record,
            expected_uninstaller=prepared.uninstaller_path,
        )
        prior_phase = str(existing["phase"])
        if (
            _UNINSTALL_JOURNAL_PHASE_INDEX[phase]
            < (_UNINSTALL_JOURNAL_PHASE_INDEX[prior_phase])
        ):
            raise UninstallPreparationError(
                "The uninstall recovery record cannot move backwards."
            )
        manifest_payload = base64.b64decode(
            str(existing["manifest_payload_base64"]),
            validate=True,
        )
    else:
        if phase != "prepared":
            raise UninstallPreparationError(
                "The first uninstall recovery record must be written before any "
                "files are removed."
            )
        try:
            manifest_payload = prepared.manifest_path.read_bytes()
        except OSError as exc:
            raise UninstallPreparationError(
                f"The ownership manifest could not be saved for recovery: {exc}"
            ) from exc
    document = _uninstall_journal_document(
        prepared,
        phase=phase,
        manifest_payload=manifest_payload,
    )
    _atomic_json_file(path, document)


def _installer_state_root_from_cache(
    record: OwnershipRecord,
) -> Path | None:
    executable = record.uninstaller_path
    if executable is None:
        return None
    for candidate in executable.parents:
        if candidate.name.casefold() != "cache":
            continue
        try:
            relative = executable.relative_to(candidate)
        except ValueError:
            return None
        if not relative.parts or relative.parts[0].casefold() != record.track.value:
            raise UninstallPreparationError(
                "The cached setup program is outside its CPU/GPU cache."
            )
        state_root = candidate.parent
        _require_direct_path(state_root, "installer state folder")
        return state_root
    return None


def _uninstall_target_lock(
    prepared: PreparedUninstall,
) -> contextlib.AbstractContextManager[object]:
    state_root = _installer_state_root_from_cache(prepared.record)
    if state_root is None:
        # Injectable/non-Windows unit fixtures may deliberately use a standalone
        # cached executable. Production cache paths always identify state_root.
        return contextlib.nullcontext()
    from napari_vipp.installer.engine import _TargetLock

    return _TargetLock(
        state_root,
        prepared.managed_root,
        f"uninstall-{uuid.uuid4()}",
    )


def _with_uninstall_target_lock(method):
    @functools.wraps(method)
    def wrapped(self, prepared, *args, **kwargs):
        with _uninstall_target_lock(prepared):
            return method(self, prepared, *args, **kwargs)

    return wrapped


class ManagedUninstaller:
    """Prepare, authorize, and apply one ownership-safe uninstall."""

    def __init__(
        self,
        *,
        registry: RegistryBackend | None = None,
        shortcut_target_reader: ShortcutTargetReader | None = None,
        current_executable: str | Path | None = None,
    ) -> None:
        self._registry = registry
        self._shortcut_target_reader = shortcut_target_reader
        self._current_executable = Path(
            current_executable if current_executable is not None else sys.executable
        )

    def prepare(
        self,
        managed_root: str | Path,
        *,
        shortcut_roots: Sequence[str | Path],
    ) -> PreparedUninstall:
        """Review current owned objects without changing files or registry."""

        root = Path(managed_root)
        if not root.is_absolute():
            raise UninstallPreparationError(
                "The managed installation folder must be an explicit absolute path."
            )
        roots = _validated_shortcut_roots(shortcut_roots)
        _require_direct_path(root, "managed installation folder")
        inspection = inspect_ownership(root)
        if inspection.state is OwnershipState.ABSENT:
            return self._prepare_from_journal(root, roots)
        if inspection.state is not OwnershipState.VALID or inspection.record is None:
            detail = inspection.error or "No managed VIPP installation was found."
            raise UninstallPreparationError(detail)
        record = inspection.record
        journal_path = _uninstall_journal_path(
            record.uninstaller_path,
            root,
            record.installation_id,
        )
        resume_phase = ""
        if record.uninstaller_path is not None:
            matching_journals = _matching_uninstall_journals(
                record.uninstaller_path,
                root,
            )
            unexpected = tuple(
                path
                for path in matching_journals
                if journal_path is None or not _same_path(path, journal_path)
            )
            if unexpected:
                reviewed = remove_superseded_uninstall_recoveries(
                    record,
                    manifest_sha256=inspection.manifest_sha256,
                    shortcut_roots=roots,
                    registry=self._registry,
                    perform_cleanup=False,
                )
                if {_path_key(path) for path in reviewed} != {
                    _path_key(path) for path in unexpected
                }:
                    raise UninstallPreparationError(
                        "A recovery record from an older installation generation "
                        "still targets this folder. Nothing was changed."
                    )
        if journal_path is not None and _path_exists(journal_path):
            journal = _read_uninstall_journal(
                journal_path,
                managed_root=root,
                shortcut_roots=roots,
                expected_record=record,
            )
            resume_phase = str(journal["phase"])
            if journal["manifest_sha256"] != inspection.manifest_sha256:
                raise UninstallPreparationError(
                    "The uninstall recovery record belongs to another ownership "
                    "manifest."
                )
        prepared = self._prepare_from_record(
            record,
            manifest_path=inspection.path,
            manifest_sha256=inspection.manifest_sha256,
            roots=roots,
            journal_path=journal_path if resume_phase else None,
            resume_phase=resume_phase,
        )
        return self._validate_resume_state(prepared)

    def _prepare_from_record(
        self,
        record: OwnershipRecord,
        *,
        manifest_path: Path,
        manifest_sha256: str,
        roots: tuple[Path, ...],
        journal_path: Path | None,
        resume_phase: str,
    ) -> PreparedUninstall:
        environment_items = _prepare_environments(
            record,
            allow_quarantine=journal_path is not None,
        )
        shortcut_items = _prepare_shortcuts(
            record,
            roots,
            self._shortcut_target_reader,
        )
        uninstaller_path = record.uninstaller_path
        if uninstaller_path is not None:
            _require_direct_path(uninstaller_path, "persistent uninstaller")
            _verify_regular_file_hash(
                uninstaller_path,
                record.uninstaller_sha256,
                "persistent uninstaller",
            )
        registry_plan = (
            registry_plan_from_record(record, manifest_sha256)
            if record.registry_key
            else None
        )
        fingerprint = _uninstall_fingerprint(
            record,
            manifest_sha256,
            roots,
            environment_items,
            shortcut_items,
            registry_plan,
        )
        return PreparedUninstall(
            managed_root=record.managed_root,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            installation_id=record.installation_id,
            environment_items=environment_items,
            shortcut_items=shortcut_items,
            shortcut_roots=roots,
            registry_plan=registry_plan,
            uninstaller_path=uninstaller_path,
            uninstaller_sha256=record.uninstaller_sha256,
            fingerprint=fingerprint,
            _record=record,
            journal_path=journal_path,
            resume_phase=resume_phase,
        )

    def _prepare_from_journal(
        self,
        root: Path,
        roots: tuple[Path, ...],
    ) -> PreparedUninstall:
        journal_path = _find_uninstall_journal(self._current_executable, root)
        if journal_path is None or not _path_exists(journal_path):
            raise UninstallPreparationError(
                "No managed VIPP installation or recoverable uninstall record "
                "was found."
            )
        journal = _read_uninstall_journal(
            journal_path,
            managed_root=root,
            shortcut_roots=roots,
            expected_record=None,
            expected_uninstaller=self._current_executable,
        )
        payload = base64.b64decode(
            str(journal["manifest_payload_base64"]),
            validate=True,
        )
        try:
            record = parse_ownership_record_bytes(payload, managed_root=root)
        except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise UninstallPreparationError(
                f"The uninstall recovery ownership data is invalid: {exc}"
            ) from exc
        if (
            record.installation_id != journal["installation_id"]
            or record.track.value != journal["track"]
            or record.uninstaller_path is None
            or not _same_path(record.uninstaller_path, self._current_executable)
        ):
            raise UninstallPreparationError(
                "The uninstall recovery record does not match this cached setup "
                "program."
            )
        prepared = self._prepare_from_record(
            record,
            manifest_path=ownership_path(root),
            manifest_sha256=str(journal["manifest_sha256"]),
            roots=roots,
            journal_path=journal_path,
            resume_phase=str(journal["phase"]),
        )
        if prepared.registry_plan is not None:
            if self._registry is None:
                raise UninstallPreparationError(
                    "Windows registry access is required to resume this uninstall."
                )
            current = self._registry.read_values(prepared.registry_plan.key)
            if current is not None and not _registry_values_match(
                current,
                prepared.registry_plan,
            ):
                raise UninstallPreparationError(
                    "The Apps & Features entry changed after uninstall began and "
                    "was preserved."
                )
        return self._validate_resume_state(prepared)

    def _validate_resume_state(
        self,
        prepared: PreparedUninstall,
    ) -> PreparedUninstall:
        phase = prepared.resume_phase
        if not phase:
            return prepared
        phase_index = _UNINSTALL_JOURNAL_PHASE_INDEX[phase]
        payload_present = any(
            item.exists for item in prepared.environment_items
        ) or any(item.exists for item in prepared.shortcut_items)
        manifest_exists = _path_exists(prepared.manifest_path)
        if (
            phase_index >= _UNINSTALL_JOURNAL_PHASE_INDEX["payload_removed"]
            and payload_present
        ):
            raise UninstallPreparationError(
                "The uninstall recovery phase is ahead of the files on disk. "
                "Nothing further was removed."
            )
        if (
            phase_index >= _UNINSTALL_JOURNAL_PHASE_INDEX["manifest_removed"]
            and manifest_exists
        ):
            raise UninstallPreparationError(
                "The uninstall recovery phase is ahead of the ownership record. "
                "Nothing further was removed."
            )
        if prepared.registry_plan is not None:
            if self._registry is None:
                if phase_index >= _UNINSTALL_JOURNAL_PHASE_INDEX["registry_removed"]:
                    raise UninstallPreparationError(
                        "Windows registry access is required to validate this "
                        "uninstall recovery record."
                    )
                return prepared
            current = self._registry.read_values(prepared.registry_plan.key)
            if current is not None and not _registry_values_match(
                current,
                prepared.registry_plan,
            ):
                raise UninstallPreparationError(
                    "The Apps & Features entry changed after uninstall began and "
                    "was preserved."
                )
            if (
                phase_index >= _UNINSTALL_JOURNAL_PHASE_INDEX["registry_removed"]
                and current is not None
            ):
                raise UninstallPreparationError(
                    "The uninstall recovery phase is ahead of Windows' removal "
                    "entry. Nothing further was removed."
                )
            if (
                current is None
                and not payload_present
                and not manifest_exists
                and phase_index >= _UNINSTALL_JOURNAL_PHASE_INDEX["manifest_removed"]
                and phase_index < _UNINSTALL_JOURNAL_PHASE_INDEX["registry_removed"]
            ):
                return replace(prepared, resume_phase="registry_removed")
        return prepared

    def authorize(self, prepared: PreparedUninstall) -> UninstallAuthorization:
        """Create explicit, single-use permission for an already shown review."""

        return UninstallAuthorization(prepared.fingerprint)

    @_with_uninstall_target_lock
    def apply(
        self,
        prepared: PreparedUninstall,
        authorization: UninstallAuthorization,
        *,
        current_executable: str | Path | None = None,
        current_pid: int | None = None,
    ) -> UninstallResult:
        """Remove only objects still matching the authorized ownership review."""

        authorization.consume(prepared.fingerprint)
        self._revalidate_manifest(
            prepared,
            allow_missing=prepared.journal_path is not None,
        )
        _validate_prepared_contract(prepared)
        if _path_exists(prepared.manifest_path):
            remove_superseded_uninstall_recoveries(
                prepared.record,
                manifest_sha256=prepared.manifest_sha256,
                shortcut_roots=prepared.shortcut_roots,
                registry=self._registry,
            )
        removed: list[Path] = []
        issues: list[UninstallIssue] = []
        journal_path = prepared.journal_path or _uninstall_journal_path(
            prepared.uninstaller_path,
            prepared.managed_root,
            prepared.installation_id,
        )
        current_phase = prepared.resume_phase or "prepared"
        if journal_path is not None:
            try:
                _write_uninstall_journal(
                    journal_path,
                    prepared,
                    phase=prepared.resume_phase or "prepared",
                )
            except (OSError, UninstallPreparationError, ValueError) as exc:
                issues.append(
                    UninstallIssue(
                        journal_path,
                        "save uninstall recovery record",
                        str(exc),
                    )
                )
                return _incomplete_result(prepared, removed, issues)

        if _phase_before(current_phase, "payload_removed"):
            for item in prepared.shortcut_items:
                issue = self._remove_shortcut(item, prepared.shortcut_roots)
                if issue is None:
                    if item.exists:
                        removed.append(item.path)
                else:
                    issues.append(issue)

            for item in prepared.environment_items:
                issue = _remove_environment(item, prepared.managed_root)
                if issue is None:
                    if item.exists:
                        removed.append(item.path)
                else:
                    issues.append(issue)

            if issues:
                return _incomplete_result(prepared, removed, issues)

            if journal_path is not None:
                try:
                    _write_uninstall_journal(
                        journal_path,
                        prepared,
                        phase="payload_removed",
                    )
                except (OSError, UninstallPreparationError, ValueError) as exc:
                    issues.append(
                        UninstallIssue(
                            journal_path,
                            "update uninstall recovery record",
                            str(exc),
                        )
                    )
                    return _incomplete_result(prepared, removed, issues)
            current_phase = "payload_removed"

        if _phase_before(current_phase, "manifest_removed"):
            try:
                if _path_exists(prepared.manifest_path):
                    self._revalidate_manifest(prepared, allow_missing=False)
                    prepared.manifest_path.unlink()
                    removed.append(prepared.manifest_path)
                elif journal_path is None:
                    raise UninstallPreparationError(
                        "The ownership manifest disappeared before uninstall commit."
                    )
            except (OSError, UninstallPreparationError) as exc:
                issues.append(
                    UninstallIssue(
                        prepared.manifest_path,
                        "remove ownership record",
                        str(exc),
                    )
                )
                return _incomplete_result(prepared, removed, issues)

            if journal_path is not None:
                try:
                    _write_uninstall_journal(
                        journal_path,
                        prepared,
                        phase="manifest_removed",
                    )
                except (OSError, UninstallPreparationError, ValueError) as exc:
                    issues.append(
                        UninstallIssue(
                            journal_path,
                            "update uninstall recovery record",
                            str(exc),
                        )
                    )
                    return _incomplete_result(prepared, removed, issues)
            current_phase = "manifest_removed"

        registry_was_removed = not _phase_before(
            current_phase,
            "registry_removed",
        )
        if not registry_was_removed and prepared.registry_plan is not None:
            if self._registry is None:
                issues.append(
                    UninstallIssue(
                        Path(f"HKCU\\{prepared.registry_plan.key}"),
                        "remove Apps & Features entry",
                        "No Windows registry backend was supplied.",
                    )
                )
            else:
                try:
                    remove_apps_and_features(self._registry, prepared.registry_plan)
                    registry_was_removed = True
                except (OSError, RegistryOwnershipError) as exc:
                    issues.append(
                        UninstallIssue(
                            Path(f"HKCU\\{prepared.registry_plan.key}"),
                            "remove Apps & Features entry",
                            str(exc),
                        )
                    )
        elif prepared.registry_plan is None:
            registry_was_removed = True
        if issues:
            return _incomplete_result(prepared, removed, issues)

        if _phase_before(current_phase, "registry_removed"):
            if journal_path is not None:
                try:
                    _write_uninstall_journal(
                        journal_path,
                        prepared,
                        phase="registry_removed",
                    )
                except (OSError, UninstallPreparationError, ValueError) as exc:
                    issues.append(
                        UninstallIssue(
                            journal_path,
                            "finish cached uninstall cleanup",
                            str(exc),
                        )
                    )
                    return _post_registry_cleanup_result(
                        prepared,
                        removed,
                        issues,
                    )
            current_phase = "registry_removed"

        try:
            journal_sha256 = (
                _sha256_file(journal_path)
                if journal_path is not None and _path_exists(journal_path)
                else ""
            )
        except (OSError, UninstallPreparationError, ValueError) as exc:
            path = journal_path or prepared.uninstaller_path or prepared.managed_root
            issues.append(
                UninstallIssue(
                    path,
                    "finish cached uninstall cleanup",
                    str(exc),
                )
            )
            return _post_registry_cleanup_result(prepared, removed, issues)

        deferred = self._remove_or_defer_uninstaller(
            prepared,
            removed,
            issues,
            current_executable=current_executable,
            current_pid=current_pid,
            journal_path=journal_path,
            journal_sha256=journal_sha256,
        )
        if (
            not issues
            and deferred is None
            and journal_path is not None
            and _path_exists(journal_path)
        ):
            try:
                _remove_exact_file(
                    journal_path,
                    journal_sha256,
                    "uninstall recovery record",
                )
                removed.append(journal_path)
            except (OSError, UninstallPreparationError) as exc:
                issues.append(
                    UninstallIssue(
                        journal_path,
                        "remove uninstall recovery record",
                        str(exc),
                    )
                )
        _remove_empty_managed_directories(prepared, removed)
        if issues:
            return _post_registry_cleanup_result(
                prepared,
                removed,
                issues,
                deferred=deferred,
            )
        return UninstallResult(
            status=UninstallStatus.COMPLETED,
            managed_root=prepared.managed_root,
            removed_paths=tuple(removed),
            preserved_paths=(),
            issues=(),
            deferred_self_delete=deferred,
            message="VIPP was removed from this Windows account.",
        )

    @staticmethod
    def _revalidate_manifest(
        prepared: PreparedUninstall,
        *,
        allow_missing: bool,
    ) -> None:
        inspection = inspect_ownership(prepared.managed_root)
        if allow_missing and inspection.state is OwnershipState.ABSENT:
            return
        if (
            inspection.state is not OwnershipState.VALID
            or inspection.record is None
            or inspection.manifest_sha256 != prepared.manifest_sha256
            or inspection.record.installation_id != prepared.installation_id
            or inspection.record != prepared.record
        ):
            raise UninstallPreparationError(
                "The VIPP installation changed after the uninstall review. "
                "Nothing further was removed."
            )

    def _remove_shortcut(
        self,
        item: PreparedOwnedShortcut,
        roots: tuple[Path, ...],
    ) -> UninstallIssue | None:
        try:
            _validate_shortcut_path(item.path, roots)
            if not _path_exists(item.path):
                return None
            _verify_regular_file_hash(item.path, item.sha256, "shortcut")
            if item.expected_target is not None:
                if self._shortcut_target_reader is None:
                    raise UninstallPreparationError(
                        "The shortcut target cannot be verified on this system."
                    )
                actual = self._shortcut_target_reader(item.path)
                if not _same_path(actual, item.expected_target):
                    raise UninstallPreparationError(
                        f"The shortcut now targets {actual}, not the owned VIPP "
                        f"launcher {item.expected_target}."
                    )
            item.path.unlink()
            return None
        except (OSError, UninstallPreparationError) as exc:
            return UninstallIssue(item.path, "remove shortcut", str(exc))

    @staticmethod
    def _remove_or_defer_uninstaller(
        prepared: PreparedUninstall,
        removed: list[Path],
        issues: list[UninstallIssue],
        *,
        current_executable: str | Path | None,
        current_pid: int | None,
        journal_path: Path | None,
        journal_sha256: str,
    ) -> DeferredSelfDelete | None:
        path = prepared.uninstaller_path
        if path is None:
            return None
        try:
            _verify_regular_file_hash(
                path,
                prepared.uninstaller_sha256,
                "persistent uninstaller",
            )
            if current_executable is not None and _same_path(
                path,
                Path(current_executable),
            ):
                return build_deferred_self_delete(
                    path,
                    wait_for_pid=current_pid or os.getpid(),
                    expected_sha256=prepared.uninstaller_sha256,
                    journal_path=journal_path,
                    journal_sha256=journal_sha256,
                )
            path.unlink()
            removed.append(path)
            return None
        except (OSError, UninstallPreparationError, ValueError) as exc:
            issues.append(
                UninstallIssue(path, "remove persistent uninstaller", str(exc))
            )
            return None


def reap_completed_uninstall_recovery(
    cached_executable: str | Path,
    *,
    managed_root: str | Path,
    expected_sha256: str,
    shortcut_roots: Sequence[str | Path],
    registry: RegistryBackend | None,
    expected_track: object | None = None,
    keep_executable: bool = False,
    perform_cleanup: bool = True,
) -> tuple[Path, ...]:
    """Reap a terminal journal/cache pair before a later setup run.

    This is intentionally root- and executable-specific. It never scans an
    arbitrary tree and never recreates an ownership manifest.
    """

    executable = Path(cached_executable)
    root = Path(managed_root)
    roots = _validated_shortcut_roots(shortcut_roots)
    if not executable.is_absolute() or not root.is_absolute():
        raise UninstallPreparationError(
            "Completed uninstall cleanup requires absolute paths."
        )
    digest = expected_sha256.strip().lower()
    if not _is_sha256(digest):
        raise UninstallPreparationError(
            "Completed uninstall cleanup requires the cached setup SHA-256."
        )
    journal_path = _find_uninstall_journal(executable, root)
    if journal_path is None:
        return ()
    document = _read_uninstall_journal(
        journal_path,
        managed_root=root,
        shortcut_roots=roots,
        expected_record=None,
        expected_uninstaller=executable,
        allow_missing_uninstaller=True,
    )
    record = document.get("_record")
    if not isinstance(record, OwnershipRecord):
        raise UninstallPreparationError(
            "The completed uninstall recovery ownership data is invalid."
        )
    if record.uninstaller_sha256 != digest:
        raise UninstallPreparationError(
            "The completed uninstall recovery record belongs to another setup program."
        )
    if expected_track is not None:
        track_value = str(getattr(expected_track, "value", expected_track)).casefold()
        if record.track.value.casefold() != track_value:
            raise UninstallPreparationError(
                "The completed uninstall recovery record belongs to another "
                "CPU/GPU option."
            )
    inspection = inspect_ownership(root)
    if inspection.state is not OwnershipState.ABSENT:
        raise UninstallPreparationError(
            "A managed or unrecognized ownership record now exists at this "
            "location; stale uninstall cleanup was refused."
        )
    if (
        _UNINSTALL_JOURNAL_PHASE_INDEX[str(document["phase"])]
        < (_UNINSTALL_JOURNAL_PHASE_INDEX["manifest_removed"])
    ):
        raise UninstallPreparationError(
            "The prior uninstall did not reach terminal cleanup. Use its Windows "
            "Remove entry to continue."
        )
    environments = _prepare_environments(record, allow_quarantine=True)
    shortcuts = _prepare_shortcuts(record, roots, None)
    if any(item.exists for item in environments) or any(
        item.exists for item in shortcuts
    ):
        raise UninstallPreparationError(
            "The prior uninstall still owns files. Its cached setup program was "
            "preserved for a safe retry."
        )
    plan = (
        registry_plan_from_record(record, str(document["manifest_sha256"]))
        if record.registry_key
        else None
    )
    if plan is not None:
        if registry is None:
            raise UninstallPreparationError(
                "Windows registry access is required to finish prior uninstall cleanup."
            )
        current = registry.read_values(plan.key)
        if current is not None:
            if _registry_values_match(current, plan):
                raise UninstallPreparationError(
                    "The prior uninstall still has a working Windows Remove entry."
                )
            raise UninstallPreparationError(
                "The Windows Remove entry belongs to another installation and was "
                "preserved."
            )
    if not perform_cleanup:
        return (journal_path,)
    removed: list[Path] = []
    if _path_exists(executable):
        _verify_regular_file_hash(
            executable,
            digest,
            "completed uninstall cached setup program",
        )
        if not keep_executable:
            executable.unlink()
            removed.append(executable)
    journal_digest = _sha256_file(journal_path)
    _remove_exact_file(
        journal_path,
        journal_digest,
        "completed uninstall recovery record",
    )
    removed.append(journal_path)
    if not keep_executable:
        try:
            executable.parent.rmdir()
            removed.append(executable.parent)
        except OSError:
            pass
    return tuple(removed)


def remove_superseded_uninstall_recoveries(
    current_record: OwnershipRecord,
    *,
    manifest_sha256: str,
    shortcut_roots: Sequence[str | Path],
    registry: RegistryBackend | None,
    perform_cleanup: bool = True,
) -> tuple[Path, ...]:
    """Remove terminal journals superseded by an accepted installation.

    Only journal files are removed. Newly accepted environments and shortcuts
    are never interpreted through the older journal.
    """

    executable = current_record.uninstaller_path
    if executable is None:
        return ()
    roots = _validated_shortcut_roots(shortcut_roots)
    current_inspection = inspect_ownership(current_record.managed_root)
    if (
        current_inspection.state is not OwnershipState.VALID
        or current_inspection.record != current_record
        or current_inspection.manifest_sha256 != manifest_sha256
    ):
        raise UninstallPreparationError(
            "The accepted installation changed before old recovery cleanup."
        )
    exact_current = _uninstall_journal_path(
        executable,
        current_record.managed_root,
        current_record.installation_id,
    )
    candidates = tuple(
        path
        for path in _matching_uninstall_journals(
            executable,
            current_record.managed_root,
        )
        if exact_current is None or not _same_path(path, exact_current)
    )
    if not candidates:
        return ()
    reviewed: list[Path] = []
    for path in candidates:
        document = _read_uninstall_journal(
            path,
            managed_root=current_record.managed_root,
            shortcut_roots=roots,
            expected_record=None,
            expected_uninstaller=executable,
        )
        old_record = document.get("_record")
        if not isinstance(old_record, OwnershipRecord):
            raise UninstallPreparationError(
                "An older uninstall recovery ownership record is invalid."
            )
        if (
            old_record.installation_id == current_record.installation_id
            or old_record.track is not current_record.track
            or old_record.uninstaller_path is None
            or not _same_path(old_record.uninstaller_path, executable)
            or old_record.uninstaller_sha256 != current_record.uninstaller_sha256
            or _UNINSTALL_JOURNAL_PHASE_INDEX[str(document["phase"])]
            < _UNINSTALL_JOURNAL_PHASE_INDEX["manifest_removed"]
        ):
            raise UninstallPreparationError(
                "An older uninstall recovery record is not terminal or does not "
                "belong to this accepted setup cache."
            )
        reviewed.append(path)
    current_plan = (
        registry_plan_from_record(current_record, manifest_sha256)
        if current_record.registry_key
        else None
    )
    if current_plan is not None:
        if registry is None:
            raise UninstallPreparationError(
                "Windows registry access is required to validate old recovery cleanup."
            )
        current_values = registry.read_values(current_plan.key)
        if current_values is None or not _registry_values_match(
            current_values,
            current_plan,
        ):
            raise UninstallPreparationError(
                "The current Windows Remove entry is not accepted; old recovery "
                "cleanup was preserved."
            )
    if not perform_cleanup:
        return tuple(reviewed)
    removed: list[Path] = []
    for path in reviewed:
        digest = _sha256_file(path)
        _remove_exact_file(path, digest, "superseded uninstall recovery record")
        removed.append(path)
    return tuple(removed)


def stage_persistent_uninstaller(
    source: str | Path,
    destination: str | Path,
    *,
    managed_root: str | Path,
    expected_existing_sha256: str = "",
) -> PersistentUninstaller:
    """Atomically copy a release uninstaller outside the managed installation."""

    source_path = Path(source)
    destination_path = Path(destination)
    managed = Path(managed_root)
    if not destination_path.is_absolute() or not managed.is_absolute():
        raise UninstallPreparationError(
            "The managed folder and persistent uninstaller destination must be "
            "absolute paths."
        )
    _require_direct_path(source_path, "release uninstaller")
    _require_direct_path(destination_path, "persistent uninstaller destination")
    _require_direct_path(destination_path.parent, "uninstaller destination")
    if _same_path(destination_path, managed) or _is_descendant(
        destination_path,
        managed,
    ):
        raise UninstallPreparationError(
            "The persistent uninstaller must be stored outside the managed folder."
        )
    if not source_path.is_file() or source_path.is_symlink():
        raise UninstallPreparationError(
            f"The release uninstaller is missing or redirected: {source_path}"
        )
    source_digest = _sha256_file(source_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    _require_direct_path(destination_path.parent, "uninstaller destination")
    destination_exists = _path_exists(destination_path)
    if destination_exists:
        if not _is_sha256(expected_existing_sha256):
            raise UninstallPreparationError(
                "A file already exists at the persistent uninstaller location, "
                "but no prior ownership hash was supplied. It was preserved."
            )
        _verify_regular_file_hash(
            destination_path,
            expected_existing_sha256,
            "previous persistent uninstaller",
        )
        if not secrets.compare_digest(
            source_digest,
            expected_existing_sha256.lower(),
        ):
            raise UninstallPreparationError(
                "The new setup payload differs from the existing owned cached "
                "setup. Use its release-specific destination; the old copy was "
                "preserved for rollback."
            )
        return PersistentUninstaller(destination_path, source_digest)
    temporary = destination_path.parent / (
        f".{destination_path.name}.{secrets.token_hex(8)}.tmp"
    )
    try:
        with (
            source_path.open("rb") as source_stream,
            temporary.open("xb") as destination_stream,
        ):
            shutil.copyfileobj(source_stream, destination_stream, 1024 * 1024)
            destination_stream.flush()
            os.fsync(destination_stream.fileno())
        digest = _sha256_file(temporary)
        if digest != source_digest or digest != _sha256_file(source_path):
            raise UninstallPreparationError(
                "The persistent uninstaller copy failed its integrity check."
            )
        os.replace(temporary, destination_path)
        return PersistentUninstaller(destination_path, digest)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def registry_plan_from_record(
    record: OwnershipRecord,
    manifest_sha256: str,
    *,
    display_name: str | None = None,
    publisher: str = "VIPP",
) -> RegistryRegistrationPlan:
    """Build a hash-bound, per-user Apps & Features registration plan."""

    if not _is_sha256(manifest_sha256):
        raise UninstallPreparationError("The ownership manifest hash is invalid.")
    if not record.managed_root.is_absolute():
        raise UninstallPreparationError(
            "The managed installation folder must be an absolute path."
        )
    if record.uninstaller_path is None or not record.uninstaller_sha256:
        raise UninstallPreparationError(
            "The ownership record has no persistent uninstaller."
        )
    if not record.uninstaller_path.is_absolute():
        raise UninstallPreparationError(
            "The persistent uninstaller path must be absolute."
        )
    _verify_regular_file_hash(
        record.uninstaller_path,
        record.uninstaller_sha256,
        "persistent uninstaller",
    )
    track_key = registry_key_for_track(record.track)
    track_value = str(getattr(record.track, "value", record.track)).casefold()
    resolved_display_name = display_name or (
        "VIPP (GPU)" if track_value == "cuda13" else "VIPP (CPU)"
    )
    key = record.registry_key or track_key
    if _normalize_registry_key(key).casefold() != track_key.casefold():
        raise UninstallPreparationError(
            "The Apps & Features key does not match this installation's CPU/GPU track."
        )
    uninstall_argv = (
        str(record.uninstaller_path),
        "--uninstall",
        "--managed-root",
        str(record.managed_root),
    )
    values: tuple[tuple[str, RegistryScalar], ...] = (
        ("DisplayName", resolved_display_name),
        ("DisplayVersion", record.version),
        ("Publisher", publisher),
        ("InstallLocation", str(record.managed_root)),
        ("UninstallString", subprocess.list2cmdline(uninstall_argv)),
        ("NoModify", 1),
        ("NoRepair", 1),
        ("VippManagedRoot", str(record.managed_root)),
        ("VippInstallationId", record.installation_id),
        ("VippManifestSha256", manifest_sha256.lower()),
        ("VippUninstallerSha256", record.uninstaller_sha256.lower()),
    )
    return RegistryRegistrationPlan(
        key=key,
        values=values,
        managed_root=record.managed_root,
        installation_id=record.installation_id,
        manifest_sha256=manifest_sha256,
        uninstaller_path=record.uninstaller_path,
        uninstaller_sha256=record.uninstaller_sha256,
    )


def remove_superseded_persistent_uninstaller(
    path: str | Path,
    expected_sha256: str,
    *,
    current_path: str | Path,
) -> bool:
    """Remove an older cached setup only when its exact owned bytes remain."""

    previous = Path(path)
    current = Path(current_path)
    if not previous.is_absolute() or not current.is_absolute():
        raise UninstallPreparationError(
            "Cached uninstaller cleanup requires absolute paths."
        )
    if _same_path(previous, current):
        return False
    _require_direct_path(previous, "superseded persistent uninstaller")
    if not _path_exists(previous):
        return False
    _verify_regular_file_hash(
        previous,
        expected_sha256,
        "superseded persistent uninstaller",
    )
    try:
        previous.unlink()
    except OSError as exc:
        raise UninstallPreparationError(
            f"The superseded persistent uninstaller could not be removed at "
            f"{previous}: {exc}"
        ) from exc
    return True


def registry_key_for_track(track: object) -> str:
    """Return a distinct per-user Apps & Features key for CPU or CUDA 13."""

    value = getattr(track, "value", track)
    normalized = str(value).strip().casefold()
    if normalized == "cpu":
        return CPU_REGISTRY_KEY
    if normalized in {"cuda13", "gpu", "gpu-cuda13"}:
        return CUDA13_REGISTRY_KEY
    raise ValueError(f"Unsupported VIPP compute track: {value!r}")


def persistent_uninstaller_destination(
    local_app_data: str | Path,
    track: object,
    release_version: str,
) -> Path:
    """Return a track- and release-specific cached EXE path for safe rollback."""

    base = Path(local_app_data)
    if not base.is_absolute():
        raise ValueError("local_app_data must be an absolute path.")
    value = str(getattr(track, "value", track)).strip().casefold()
    if value == "cpu":
        suffix = "cpu"
    elif value in {"cuda13", "gpu", "gpu-cuda13"}:
        suffix = "cuda13"
    else:
        raise ValueError(f"Unsupported VIPP compute track: {track!r}")
    version = release_version.strip()
    if (
        not version
        or version in {".", ".."}
        or any(character not in _SAFE_VERSION_CHARACTERS for character in version)
    ):
        raise ValueError("release_version is not a safe path component.")
    return base / "VIPP" / "Uninstallers" / suffix / version / "VIPP-Setup.exe"


def register_apps_and_features(
    backend: RegistryBackend,
    plan: RegistryRegistrationPlan,
    *,
    previous_plan: RegistryRegistrationPlan | None = None,
    recover_interrupted: bool = False,
) -> None:
    """Register only when the manifest and persistent executable still match."""

    inspection = inspect_ownership(plan.managed_root)
    if (
        inspection.state is not OwnershipState.VALID
        or inspection.record is None
        or inspection.manifest_sha256 != plan.manifest_sha256
        or inspection.record.installation_id != plan.installation_id
    ):
        raise RegistryOwnershipError(
            "Apps & Features registration no longer matches the ownership record."
        )
    _verify_regular_file_hash(
        plan.uninstaller_path,
        plan.uninstaller_sha256,
        "persistent uninstaller",
    )
    current = backend.read_values(plan.key)
    if current is not None and not _registry_values_match(current, plan):
        previous_identity_matches = (
            previous_plan is not None
            and previous_plan.key.casefold() == plan.key.casefold()
            and previous_plan.installation_id == plan.installation_id
            and _same_path(previous_plan.managed_root, plan.managed_root)
        )
        previous_is_authorized = bool(
            previous_identity_matches
            and previous_plan is not None
            and _registry_values_match(current, previous_plan)
        )
        interrupted_transition_is_authorized = bool(
            recover_interrupted
            and _registry_transition_values_match(
                current,
                plan,
                previous_plan=(previous_plan if previous_identity_matches else None),
            )
        )
        if not previous_is_authorized and not interrupted_transition_is_authorized:
            raise RegistryOwnershipError(
                "The Apps & Features entry belongs to a different installation "
                "and was not overwritten."
            )
    backend.write_values(plan.key, plan.value_map)
    accepted = backend.read_values(plan.key)
    if accepted is None or not _registry_values_match(accepted, plan):
        raise RegistryOwnershipError(
            "The Apps & Features update was interrupted and remains queued for "
            "recovery."
        )


def remove_apps_and_features(
    backend: RegistryBackend,
    plan: RegistryRegistrationPlan,
) -> None:
    """Delete the per-user registry key only when every binding value matches."""

    current = backend.read_values(plan.key)
    if current is None:
        return
    if not _registry_values_match(current, plan):
        raise RegistryOwnershipError(
            "The Apps & Features entry changed and was preserved."
        )
    backend.delete_key(plan.key)
    remaining = backend.read_values(plan.key)
    if remaining is not None:
        if _registry_values_match(remaining, plan):
            raise OSError(
                "Windows did not confirm removal of the Apps & Features entry."
            )
        raise RegistryOwnershipError(
            "The Apps & Features entry changed while it was being removed and "
            "was preserved."
        )


def build_deferred_self_delete(
    target: str | Path,
    *,
    wait_for_pid: int,
    expected_sha256: str = "",
    powershell: str | Path = "powershell.exe",
    journal_path: str | Path | None = None,
    journal_sha256: str = "",
) -> DeferredSelfDelete:
    """Build a shell-injection-safe PowerShell self-delete request.

    The caller should start this command detached only after the uninstall UI is
    ready to exit.  The encoded script waits for this process, deletes the exact
    literal executable path, then removes its now-empty parent when possible.
    """

    path = Path(target)
    if wait_for_pid <= 0:
        raise ValueError("wait_for_pid must be positive.")
    _require_direct_path(path, "persistent uninstaller")
    digest = expected_sha256.strip().lower() or _sha256_file(path)
    _verify_regular_file_hash(path, digest, "persistent uninstaller")
    literal = str(path).replace("'", "''")
    parent = str(path.parent).replace("'", "''")
    # Get-FileHash is module-provided rather than a PowerShell language primitive.
    # A Python parent can pass Windows PowerShell a PSModulePath with PowerShell 7
    # modules first, preventing the compatible Desktop module from auto-loading.
    # Keep the detached helper independent of module discovery by hashing through
    # the .NET types that are built into every supported Windows PowerShell host.
    file_hash_function = (
        "function Get-VippSha256 { param([string]$Path) "
        "$stream=$null; $algorithm=$null; try { "
        "$stream=[System.IO.File]::OpenRead($Path); "
        "$algorithm=[System.Security.Cryptography.SHA256]::Create(); "
        "return ([System.BitConverter]::ToString("
        "$algorithm.ComputeHash($stream))).Replace('-','') "
        "} catch { return '' } finally { "
        "if ($algorithm -ne $null) { $algorithm.Dispose() }; "
        "if ($stream -ne $null) { $stream.Dispose() } } }; "
    )
    journal_cleanup = ""
    deletion_authorization = ""
    if journal_path is not None:
        journal = Path(journal_path)
        _require_direct_path(journal, "uninstall recovery record")
        expected_journal = journal_sha256.strip().lower()
        _verify_regular_file_hash(
            journal,
            expected_journal,
            "uninstall recovery record",
        )
        try:
            journal_document = _load_strict_json(journal.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise UninstallPreparationError(
                f"The uninstall recovery record cannot be deferred safely: {exc}"
            ) from exc
        if not isinstance(journal_document, dict) or (
            journal_document.get("schema") != _UNINSTALL_JOURNAL_SCHEMA
            or journal_document.get("schema_version") != _UNINSTALL_JOURNAL_VERSION
            or journal_document.get("phase") != "registry_removed"
        ):
            raise UninstallPreparationError(
                "The uninstall recovery record is not ready for final cleanup."
            )
        managed_root = Path(_journal_text(journal_document, "managed_root"))
        manifest_path = Path(_journal_text(journal_document, "manifest_path"))
        if (
            not managed_root.is_absolute()
            or not manifest_path.is_absolute()
            or not _same_path(manifest_path, ownership_path(managed_root))
        ):
            raise UninstallPreparationError(
                "The uninstall recovery ownership path is invalid."
            )
        _require_direct_path(manifest_path, "ownership manifest")
        journal_literal = str(journal).replace("'", "''")
        manifest_literal = str(manifest_path).replace("'", "''")
        deletion_authorization = (
            f"$j='{journal_literal}'; $m='{manifest_literal}'; "
            "function Test-VippDeleteAuthorized { "
            "if (Test-Path -LiteralPath $m) { return $false }; "
            "if (-not (Test-Path -LiteralPath $j -PathType Leaf)) { "
            "return $false }; "
            f"return ((Get-VippSha256 -Path $j) "
            f"-ieq '{expected_journal}') }}; "
        )
        journal_cleanup = (
            "if ((-not (Test-Path -LiteralPath $p)) -and "
            "(Test-VippDeleteAuthorized)) { "
            "Remove-Item -LiteralPath $j -Force -ErrorAction SilentlyContinue }; "
        )
    mutex_name = _SETUP_SINGLE_INSTANCE_MUTEX.replace("'", "''")
    authorization_checks = (
        "if (-not (Test-VippDeleteAuthorized)) { break }; "
        if journal_path is not None
        else ""
    )
    script = (
        f"{file_hash_function}"
        f"Wait-Process -Id {wait_for_pid} -ErrorAction SilentlyContinue; "
        f"$mx=[Threading.Mutex]::new($false,'{mutex_name}'); $held=$false; "
        "try { try { $held=$mx.WaitOne() } "
        "catch [Threading.AbandonedMutexException] { $held=$true }; "
        "if ($held) { "
        f"$p='{literal}'; {deletion_authorization}"
        "for ($i=0; $i -lt 120; $i++) { "
        "if (-not (Test-Path -LiteralPath $p -PathType Leaf)) { break }; "
        f"if ((Get-VippSha256 -Path $p) -ine "
        f"'{digest}') {{ break }}; "
        f"{authorization_checks}"
        f"try {{ {authorization_checks}"
        "Remove-Item -LiteralPath $p -Force -ErrorAction Stop; break } "
        "catch { Start-Sleep -Milliseconds 250 } }; "
        f"{journal_cleanup}"
        f"if (-not (Test-Path -LiteralPath $p)) {{ Remove-Item -LiteralPath "
        f"'{parent}' -Force -ErrorAction SilentlyContinue }} }} "
        "} finally { if ($held) { $mx.ReleaseMutex() }; $mx.Dispose() }"
    )
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    argv = (
        str(powershell),
        "-NoLogo",
        "-NoProfile",
        "-NonInteractive",
        "-WindowStyle",
        "Hidden",
        "-EncodedCommand",
        encoded,
    )
    return DeferredSelfDelete(
        path,
        digest,
        wait_for_pid,
        argv,
        Path(journal_path) if journal_path is not None else None,
        journal_sha256.strip().lower() if journal_path is not None else "",
    )


def schedule_deferred_self_delete(request: DeferredSelfDelete) -> None:
    """Start the reviewed deferred deletion without opening a console window."""

    creationflags = 0
    if os.name == "nt":
        creationflags = 0x00000008 | 0x08000000  # DETACHED_PROCESS | NO_WINDOW
    subprocess.Popen(  # noqa: S603 - argv is created internally, without a shell
        request.argv,
        close_fds=True,
        creationflags=creationflags,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def read_windows_shortcut_target(shortcut: str | Path) -> Path:
    """Read a ``.lnk`` target via Windows Script Host without loading the target."""

    path = Path(shortcut)
    _require_direct_path(path, "shortcut")
    literal = str(path).replace("'", "''")
    script = (
        "$w=New-Object -ComObject WScript.Shell;"
        f"$s=$w.CreateShortcut('{literal}');"
        "[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false);"
        "$s.TargetPath"
    )
    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    result = subprocess.run(  # noqa: S603 - fixed PowerShell argv, no shell
        (
            "powershell.exe",
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            script,
        ),
        check=False,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        creationflags=creationflags,
    )
    target = result.stdout.strip()
    if result.returncode != 0 or not target:
        detail = result.stderr.strip() or "Windows did not return a shortcut target."
        raise UninstallPreparationError(detail)
    return Path(target)


def _prepare_environments(
    record: OwnershipRecord,
    *,
    allow_quarantine: bool = False,
) -> tuple[PreparedOwnedEnvironment, ...]:
    store = managed_environments_root(record.managed_root)
    _require_direct_path(store, "managed environment store")
    owned = (
        OwnedEnvironment(record.environment_root, record.environment_marker_sha256),
        *record.retired_environments,
    )
    prepared: list[PreparedOwnedEnvironment] = []
    for index, environment in enumerate(owned):
        path = environment.path
        quarantine = _environment_quarantine_path(record, path)
        if not _is_strict_descendant(path, store):
            raise UninstallPreparationError(
                f"An owned environment is outside the private VIPP store: {path}"
            )
        _require_direct_path(path, "owned environment")
        if not _is_strict_descendant(quarantine, store):
            raise UninstallPreparationError(
                "An uninstall quarantine is outside the private VIPP store."
            )
        _require_direct_path(quarantine, "owned environment quarantine")
        original_exists = _path_exists(path)
        quarantine_exists = _path_exists(quarantine)
        if original_exists and quarantine_exists:
            raise UninstallPreparationError(
                "Both an owned environment and its uninstall quarantine exist. "
                "Nothing was removed."
            )
        if quarantine_exists and not allow_quarantine:
            raise UninstallPreparationError(
                "An uninstall quarantine exists without its recovery record. "
                "Nothing was removed."
            )
        if original_exists:
            if not path.is_dir() or path.is_symlink():
                raise UninstallPreparationError(
                    f"An owned environment is not a normal folder: {path}"
                )
            _validate_direct_tree(path)
            _verify_regular_file_hash(
                path / _CANDIDATE_MARKER,
                environment.marker_sha256,
                "environment ownership marker",
            )
        if quarantine_exists:
            if not quarantine.is_dir() or quarantine.is_symlink():
                raise UninstallPreparationError(
                    "An owned environment quarantine is not a normal folder: "
                    f"{quarantine}"
                )
            _validate_direct_tree(quarantine)
            quarantine_marker = quarantine / _CANDIDATE_MARKER
            if _path_exists(quarantine_marker):
                _verify_regular_file_hash(
                    quarantine_marker,
                    environment.marker_sha256,
                    "quarantined environment ownership marker",
                )
            elif any(quarantine.iterdir()):
                raise UninstallPreparationError(
                    "A partially removed environment lost its ownership marker "
                    f"and was preserved: {quarantine}"
                )
        prepared.append(
            PreparedOwnedEnvironment(
                path=path,
                marker_path=path / _CANDIDATE_MARKER,
                marker_sha256=environment.marker_sha256,
                active=index == 0,
                exists=original_exists or quarantine_exists,
                quarantine_path=quarantine,
            )
        )
    return tuple(prepared)


def _validate_prepared_contract(prepared: PreparedUninstall) -> None:
    """Reject a manually altered or mismatched prepared-removal object."""

    record = prepared.record
    expected_environments = (
        OwnedEnvironment(record.environment_root, record.environment_marker_sha256),
        *record.retired_environments,
    )
    if len(expected_environments) != len(prepared.environment_items):
        raise UninstallPreparationError(
            "The prepared environment list no longer matches the ownership record."
        )
    for index, (expected, actual) in enumerate(
        zip(expected_environments, prepared.environment_items, strict=True)
    ):
        if (
            not _same_path(expected.path, actual.path)
            or expected.marker_sha256 != actual.marker_sha256
            or actual.active != (index == 0)
            or actual.quarantine_path is None
            or not _same_path(
                actual.quarantine_path,
                _environment_quarantine_path(record, expected.path),
            )
        ):
            raise UninstallPreparationError(
                "The prepared environment list no longer matches the ownership record."
            )
    if len(record.shortcuts) != len(prepared.shortcut_items):
        raise UninstallPreparationError(
            "The prepared shortcut list no longer matches the ownership record."
        )
    for expected, actual in zip(
        record.shortcuts,
        prepared.shortcut_items,
        strict=True,
    ):
        target_value = getattr(expected, "target", None)
        expected_target = Path(target_value) if target_value is not None else None
        target_matches = (
            expected_target is None and actual.expected_target is None
        ) or (
            expected_target is not None
            and actual.expected_target is not None
            and _same_path(expected_target, actual.expected_target)
        )
        if (
            not _same_path(expected.path, actual.path)
            or expected.sha256 != actual.sha256
            or not target_matches
        ):
            raise UninstallPreparationError(
                "The prepared shortcut list no longer matches the ownership record."
            )
        _validate_shortcut_path(actual.path, prepared.shortcut_roots)
    if prepared.uninstaller_path != record.uninstaller_path or (
        prepared.uninstaller_sha256 != record.uninstaller_sha256
    ):
        raise UninstallPreparationError(
            "The prepared uninstaller no longer matches the ownership record."
        )
    expected_registry_key = (
        _normalize_registry_key(record.registry_key) if record.registry_key else ""
    )
    if bool(expected_registry_key) != bool(prepared.registry_plan):
        raise UninstallPreparationError(
            "The prepared registry operation no longer matches the ownership record."
        )
    if prepared.registry_plan is not None and (
        prepared.registry_plan.key.casefold() != expected_registry_key.casefold()
        or prepared.registry_plan.installation_id != record.installation_id
        or prepared.registry_plan.manifest_sha256 != prepared.manifest_sha256
    ):
        raise UninstallPreparationError(
            "The prepared registry operation no longer matches the ownership record."
        )
    expected_fingerprint = _uninstall_fingerprint(
        record,
        prepared.manifest_sha256,
        prepared.shortcut_roots,
        prepared.environment_items,
        prepared.shortcut_items,
        prepared.registry_plan,
    )
    if not secrets.compare_digest(expected_fingerprint, prepared.fingerprint):
        raise UninstallPreparationError(
            "The prepared uninstall review was altered and cannot be used."
        )


def _prepare_shortcuts(
    record: OwnershipRecord,
    roots: tuple[Path, ...],
    target_reader: ShortcutTargetReader | None,
) -> tuple[PreparedOwnedShortcut, ...]:
    prepared: list[PreparedOwnedShortcut] = []
    for shortcut in record.shortcuts:
        path = shortcut.path
        _validate_shortcut_path(path, roots)
        exists = _path_exists(path)
        if exists:
            _verify_regular_file_hash(path, shortcut.sha256, "shortcut")
        target_value = getattr(shortcut, "target", None)
        expected_target = Path(target_value) if target_value is not None else None
        if exists and expected_target is not None:
            if target_reader is None:
                raise UninstallPreparationError(
                    f"The target of the owned shortcut cannot be verified: {path}"
                )
            actual_target = target_reader(path)
            if not _same_path(actual_target, expected_target):
                raise UninstallPreparationError(
                    f"The shortcut target changed and will be preserved: {path}"
                )
        prepared.append(
            PreparedOwnedShortcut(
                path=path,
                sha256=shortcut.sha256,
                expected_target=expected_target,
                exists=exists,
            )
        )
    return tuple(prepared)


def _remove_environment(
    item: PreparedOwnedEnvironment,
    managed_root: Path,
) -> UninstallIssue | None:
    try:
        store = managed_environments_root(managed_root)
        quarantine = item.quarantine_path
        if quarantine is None or not _is_strict_descendant(
            quarantine,
            store,
        ):
            raise UninstallPreparationError(
                "The environment quarantine is outside the private VIPP store."
            )
        if not _is_strict_descendant(item.path, store):
            raise UninstallPreparationError(
                "The environment is outside the private VIPP store."
            )
        _require_direct_path(item.path, "owned environment")
        _require_direct_path(quarantine, "owned environment quarantine")
        original_exists = _path_exists(item.path)
        quarantine_exists = _path_exists(quarantine)
        if original_exists and quarantine_exists:
            raise UninstallPreparationError(
                "Both the environment and its quarantine exist; neither was removed."
            )
        if not original_exists and not quarantine_exists:
            return None
        if original_exists:
            if not item.path.is_dir() or item.path.is_symlink():
                raise UninstallPreparationError(
                    "The owned environment is no longer a normal folder."
                )
            _validate_direct_tree(item.path)
            _verify_regular_file_hash(
                item.marker_path,
                item.marker_sha256,
                "environment ownership marker",
            )
            _replace_file_with_retry(item.path, quarantine)
            _fsync_directory(store)
        _remove_quarantined_environment(
            quarantine,
            item.marker_sha256,
        )
        return None
    except (OSError, UninstallPreparationError) as exc:
        return UninstallIssue(item.path, "remove owned environment", str(exc))


def _remove_quarantined_environment(
    quarantine: Path,
    marker_sha256: str,
) -> None:
    """Remove a quarantined tree while retaining its marker until the end."""

    _require_direct_path(quarantine, "owned environment quarantine")
    if not _path_exists(quarantine):
        return
    if not quarantine.is_dir() or quarantine.is_symlink():
        raise UninstallPreparationError(
            "The owned environment quarantine is not a normal folder."
        )
    marker = quarantine / _CANDIDATE_MARKER
    if not _path_exists(marker):
        # A crash can occur after the marker unlink and before the final rmdir.
        # Only an already-empty folder is safe to finish; never recurse again.
        try:
            quarantine.rmdir()
        except OSError as exc:
            raise UninstallPreparationError(
                "The quarantined environment lost its ownership marker and was "
                f"preserved: {quarantine}: {exc}"
            ) from exc
        return
    _verify_regular_file_hash(
        marker,
        marker_sha256,
        "quarantined environment ownership marker",
    )
    _remove_direct_tree(quarantine, preserve_top_level=marker)
    _verify_regular_file_hash(
        marker,
        marker_sha256,
        "quarantined environment ownership marker",
    )
    marker.unlink()
    try:
        quarantine.rmdir()
    except OSError as exc:
        raise UninstallPreparationError(
            "The quarantined environment changed during removal and was preserved: "
            f"{quarantine}: {exc}"
        ) from exc


def _validated_shortcut_roots(
    shortcut_roots: Sequence[str | Path],
) -> tuple[Path, ...]:
    roots = tuple(Path(root) for root in shortcut_roots)
    if not roots:
        raise UninstallPreparationError(
            "Desktop and Start Menu shortcut folders must be supplied explicitly."
        )
    unique: dict[str, Path] = {}
    for root in roots:
        if not root.is_absolute():
            raise UninstallPreparationError(
                f"A shortcut folder is not an absolute path: {root}"
            )
        _require_direct_path(root, "shortcut folder")
        unique[_path_key(root)] = root
    return tuple(sorted(unique.values(), key=_path_key))


def _validate_shortcut_path(path: Path, roots: tuple[Path, ...]) -> None:
    if not path.is_absolute():
        raise UninstallPreparationError(
            f"An owned shortcut path is not absolute: {path}"
        )
    if path.suffix.casefold() != ".lnk":
        raise UninstallPreparationError(
            f"An owned shortcut is not a Windows .lnk file: {path}"
        )
    if not any(_is_strict_descendant(path, root) for root in roots):
        raise UninstallPreparationError(
            f"An owned shortcut is outside the reviewed Desktop/Start Menu "
            f"folders: {path}"
        )
    _require_direct_path(path, "owned shortcut")


def _remove_empty_managed_directories(
    prepared: PreparedUninstall,
    removed: list[Path],
) -> None:
    metadata = prepared.managed_root / OWNERSHIP_DIRECTORY
    candidates = (
        managed_environments_root(prepared.managed_root),
        metadata,
    )
    if not prepared.record.managed_root_preexisting:
        candidates = (*candidates, prepared.managed_root)
    for path in candidates:
        try:
            _require_direct_path(path, "empty managed folder")
            path.rmdir()
            removed.append(path)
        except (FileNotFoundError, OSError, UninstallPreparationError):
            # Non-empty or user-modified folders are deliberately preserved.
            continue


def _remove_direct_tree(
    path: Path,
    *,
    preserve_top_level: Path | None = None,
) -> None:
    """Remove a directory without ever following a reparse point or symlink."""

    _require_direct_path(path, "owned environment")
    with os.scandir(path) as entries:
        for entry in entries:
            child = Path(entry.path)
            if preserve_top_level is not None and _same_path(
                child,
                preserve_top_level,
            ):
                continue
            metadata = entry.stat(follow_symlinks=False)
            if _metadata_is_reparse(metadata):
                raise UninstallPreparationError(
                    "The owned environment contains a symbolic link or Windows "
                    f"reparse point and was preserved: {child}"
                )
            if stat.S_ISDIR(metadata.st_mode):
                _remove_direct_tree(child)
            elif stat.S_ISREG(metadata.st_mode):
                try:
                    child.unlink()
                except OSError as exc:
                    raise UninstallPreparationError(
                        f"Could not remove the owned file {child}: {exc}"
                    ) from exc
            else:
                raise UninstallPreparationError(
                    "The owned environment contains an unsupported filesystem "
                    f"object and was preserved: {child}"
                )
    if preserve_top_level is None:
        try:
            path.rmdir()
        except OSError as exc:
            raise UninstallPreparationError(
                f"Could not remove the owned folder {path}: {exc}"
            ) from exc


def _validate_direct_tree(path: Path) -> None:
    """Refuse any tree whose deletion could cross a filesystem redirection."""

    _require_direct_path(path, "owned environment")
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                child = Path(entry.path)
                metadata = entry.stat(follow_symlinks=False)
                if _metadata_is_reparse(metadata):
                    raise UninstallPreparationError(
                        "The owned environment contains a symbolic link or Windows "
                        f"reparse point and was preserved: {child}"
                    )
                if stat.S_ISDIR(metadata.st_mode):
                    _validate_direct_tree(child)
                elif not stat.S_ISREG(metadata.st_mode):
                    raise UninstallPreparationError(
                        "The owned environment contains an unsupported filesystem "
                        f"object and was preserved: {child}"
                    )
    except OSError as exc:
        raise UninstallPreparationError(
            f"The owned environment could not be inspected at {path}: {exc}"
        ) from exc


def _incomplete_result(
    prepared: PreparedUninstall,
    removed: list[Path],
    issues: list[UninstallIssue],
    *,
    deferred: DeferredSelfDelete | None = None,
) -> UninstallResult:
    preserved = tuple(dict.fromkeys(issue.path for issue in issues))
    details = "; ".join(
        f"{issue.operation} at {issue.path}: {issue.error}" for issue in issues
    )
    return UninstallResult(
        status=UninstallStatus.INCOMPLETE,
        managed_root=prepared.managed_root,
        removed_paths=tuple(removed),
        preserved_paths=preserved,
        issues=tuple(issues),
        deferred_self_delete=deferred,
        message=(
            "VIPP cleanup is incomplete. The listed items were preserved so no "
            f"unverified data was deleted. {details}"
        ),
    )


def _post_registry_cleanup_result(
    prepared: PreparedUninstall,
    removed: list[Path],
    issues: list[UninstallIssue],
    *,
    deferred: DeferredSelfDelete | None = None,
) -> UninstallResult:
    leftovers = [issue.path for issue in issues]
    journal_path = prepared.journal_path or _uninstall_journal_path(
        prepared.uninstaller_path,
        prepared.managed_root,
        prepared.installation_id,
    )
    for path in (prepared.uninstaller_path, journal_path):
        if path is not None and _path_exists(path):
            leftovers.append(path)
    preserved = tuple(dict.fromkeys(leftovers))
    details = "; ".join(
        f"{issue.operation} at {issue.path}: {issue.error}" for issue in issues
    )
    return UninstallResult(
        status=UninstallStatus.INCOMPLETE,
        managed_root=prepared.managed_root,
        removed_paths=tuple(removed),
        preserved_paths=preserved,
        issues=tuple(issues),
        deferred_self_delete=deferred,
        message=(
            "VIPP itself was removed, but cached setup cleanup is incomplete. "
            "The Windows Remove entry is already gone, so do not use it as a "
            f"retry route. Only the exact listed cached paths remain. {details}"
        ),
        retry_via_apps=False,
    )


def _phase_before(current: str, target: str) -> bool:
    return (
        _UNINSTALL_JOURNAL_PHASE_INDEX[current]
        < (_UNINSTALL_JOURNAL_PHASE_INDEX[target])
    )


def _strict_json_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            return False
        return all(_strict_json_equal(left[key], right[key]) for key in left)
    if isinstance(left, list):
        if not isinstance(right, list) or len(left) != len(right):
            return False
        return all(
            _strict_json_equal(actual, expected)
            for actual, expected in zip(left, right, strict=True)
        )
    return left == right


def _load_strict_json(payload: str) -> object:
    def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        document: dict[str, object] = {}
        for key, value in pairs:
            if key in document:
                raise ValueError(f"duplicate JSON field {key!r}")
            document[key] = value
        return document

    return json.loads(payload, object_pairs_hook=object_pairs)


def _journal_text(document: Mapping[str, object], field_name: str) -> str:
    value = document.get(field_name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"the uninstall recovery field {field_name!r} is invalid")
    return value


def _journal_digest(document: Mapping[str, object], field_name: str) -> str:
    value = _journal_text(document, field_name).strip().lower()
    if not _is_sha256(value):
        raise ValueError(f"the uninstall recovery digest {field_name!r} is invalid")
    return value


def _fsync_directory(directory: Path) -> None:
    """Best-effort directory durability; Windows may reject directory handles."""

    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _replace_file_with_retry(source: Path, destination: Path) -> None:
    for index, delay in enumerate((*_ATOMIC_REPLACE_RETRY_DELAYS, 0.0)):
        try:
            os.replace(source, destination)
            return
        except OSError as exc:
            transient = (
                os.name == "nt"
                and getattr(exc, "winerror", None) in _WINDOWS_TRANSIENT_REPLACE_ERRORS
            )
            if not transient or index == len(_ATOMIC_REPLACE_RETRY_DELAYS):
                raise
            time.sleep(delay)


def _atomic_json_file(path: Path, document: Mapping[str, object]) -> None:
    _require_direct_path(path, "uninstall recovery record")
    _require_direct_path(path.parent, "uninstall recovery folder")
    payload = (
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    temporary = path.parent / f".{path.name}.{secrets.token_hex(12)}.tmp"
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _replace_file_with_retry(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _remove_exact_file(path: Path, expected_sha256: str, label: str) -> None:
    _verify_regular_file_hash(path, expected_sha256, label)
    path.unlink()


def _registry_binding_matches(
    values: Mapping[str, RegistryScalar],
    plan: RegistryRegistrationPlan,
) -> bool:
    casefolded = {name.casefold(): value for name, value in values.items()}
    for name, expected in plan.binding_values.items():
        actual = casefolded.get(name.casefold())
        if name == "VippManagedRoot":
            if not isinstance(actual, str) or not _same_path(
                Path(actual),
                Path(expected),
            ):
                return False
        elif not isinstance(actual, str) or not secrets.compare_digest(
            actual.casefold(),
            expected.casefold(),
        ):
            return False
    return True


def _registry_values_match(
    values: Mapping[str, RegistryScalar],
    plan: RegistryRegistrationPlan,
) -> bool:
    """Require every owned value and reject unreviewed values at the key."""

    if not _registry_binding_matches(values, plan):
        return False
    observed = {name.casefold(): value for name, value in values.items()}
    expected = {name.casefold(): value for name, value in plan.values}
    if observed.keys() != expected.keys():
        return False
    return all(observed[name] == value for name, value in expected.items())


def _registry_transition_values_match(
    values: Mapping[str, RegistryScalar],
    plan: RegistryRegistrationPlan,
    *,
    previous_plan: RegistryRegistrationPlan | None,
) -> bool:
    """Recognize only a journal-authorized partial old-to-new value swap."""

    observed = {name.casefold(): value for name, value in values.items()}
    if not observed:
        return False
    expected = {name.casefold(): value for name, value in plan.values}
    previous = (
        {name.casefold(): value for name, value in previous_plan.values}
        if previous_plan is not None
        else {}
    )
    ownership_names = {name.casefold() for name in _REGISTRY_BINDING_WRITE_ORDER}
    if not ownership_names.intersection(observed):
        return False
    for name, value in observed.items():
        candidates = tuple(
            candidate[name] for candidate in (expected, previous) if name in candidate
        )
        if not candidates or value not in candidates:
            return False
    return True


def _uninstall_fingerprint(
    record: OwnershipRecord,
    manifest_sha256: str,
    roots: tuple[Path, ...],
    environments: tuple[PreparedOwnedEnvironment, ...],
    shortcuts: tuple[PreparedOwnedShortcut, ...],
    registry_plan: RegistryRegistrationPlan | None,
) -> str:
    digest = hashlib.sha256()
    fields = [
        record.installation_id,
        _path_key(record.managed_root),
        manifest_sha256,
        registry_plan.key if registry_plan else "",
    ]
    fields.extend(_path_key(root) for root in roots)
    fields.extend(
        f"{_path_key(item.path)}:{item.marker_sha256}:{int(item.exists)}:"
        f"{_path_key(item.quarantine_path) if item.quarantine_path else ''}"
        for item in environments
    )
    fields.extend(
        f"{_path_key(item.path)}:{item.sha256}:{int(item.exists)}:"
        f"{_path_key(item.expected_target) if item.expected_target else ''}"
        for item in shortcuts
    )
    for value in fields:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _verify_regular_file_hash(path: Path, expected: str, label: str) -> None:
    _require_direct_path(path, label)
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise UninstallPreparationError(f"The {label} is missing: {path}") from exc
    except OSError as exc:
        raise UninstallPreparationError(
            f"The {label} could not be inspected at {path}: {exc}"
        ) from exc
    if _metadata_is_reparse(metadata) or not stat.S_ISREG(metadata.st_mode):
        raise UninstallPreparationError(
            f"The {label} is redirected or is not a regular file: {path}"
        )
    if not _is_sha256(expected) or _sha256_file(path) != expected.lower():
        raise UninstallPreparationError(
            f"The {label} changed and was preserved: {path}"
        )


def _path_exists(path: Path) -> bool:
    """Distinguish a genuinely absent path from an uninspectable path."""

    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise UninstallPreparationError(
            f"The path could not be inspected and was preserved: {path}: {exc}"
        ) from exc
    return True


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _require_direct_path(path: Path, label: str) -> None:
    if _path_has_reparse_component(path):
        raise UninstallPreparationError(
            f"The {label} contains a symbolic link or Windows reparse point: {path}"
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


def _metadata_is_reparse(metadata: os.stat_result) -> bool:
    return stat.S_ISLNK(metadata.st_mode) or bool(
        getattr(metadata, "st_file_attributes", 0) & _FILE_ATTRIBUTE_REPARSE_POINT
    )


def _normalize_registry_key(key: str) -> str:
    normalized = key.strip().replace("/", "\\").strip("\\")
    prefix = "HKEY_CURRENT_USER\\"
    if normalized.upper().startswith(prefix):
        normalized = normalized[len(prefix) :]
    if not normalized or ".." in normalized.split("\\"):
        raise ValueError("The registry key is invalid.")
    return normalized


def _is_sha256(value: str) -> bool:
    normalized = value.strip().lower()
    return len(normalized) == 64 and all(
        character in _HEX_DIGITS for character in normalized
    )


def _path_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))


def _same_path(left: Path, right: Path) -> bool:
    return _path_key(left) == _path_key(right)


def _is_descendant(path: Path, parent: Path) -> bool:
    try:
        Path(os.path.abspath(path)).relative_to(Path(os.path.abspath(parent)))
    except ValueError:
        return False
    return True


def _is_strict_descendant(path: Path, parent: Path) -> bool:
    if _same_path(path, parent):
        return False
    return _is_descendant(path, parent)


__all__ = [
    "CPU_REGISTRY_KEY",
    "CUDA13_REGISTRY_KEY",
    "DEFAULT_REGISTRY_KEY",
    "DeferredDeleteScheduler",
    "DeferredSelfDelete",
    "ManagedUninstaller",
    "PersistentUninstaller",
    "PreparedOwnedEnvironment",
    "PreparedOwnedShortcut",
    "PreparedUninstall",
    "RegistryBackend",
    "RegistryOwnershipError",
    "RegistryRegistrationPlan",
    "ShortcutTargetReader",
    "UninstallAuthorization",
    "UninstallAuthorizationError",
    "UninstallError",
    "UninstallIssue",
    "UninstallPreparationError",
    "UninstallResult",
    "UninstallStatus",
    "WindowsRegistryBackend",
    "build_deferred_self_delete",
    "read_windows_shortcut_target",
    "reap_completed_uninstall_recovery",
    "register_apps_and_features",
    "registry_key_for_track",
    "registry_plan_from_record",
    "remove_superseded_persistent_uninstaller",
    "remove_superseded_uninstall_recoveries",
    "remove_apps_and_features",
    "schedule_deferred_self_delete",
    "stage_persistent_uninstaller",
    "persistent_uninstaller_destination",
]
