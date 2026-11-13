"""Explicit, narrowly scoped repair of a complete stale owned Windows entry.

Ordinary install/uninstall ownership rules remain unchanged. This operation
does not install, remove or launch software. Unknown values, kinds and subkeys
are refused, never adopted. Win32 value writes are not an atomic registry CAS;
we recheck before each changed-name patch and preserve raced foreign changes.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import stat
import subprocess
import threading
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

from packaging.version import Version

from napari_vipp.installer.models import ComputeTrack
from napari_vipp.installer.ownership import OwnershipState, inspect_ownership
from napari_vipp.installer.uninstall import (
    CPU_REGISTRY_KEY,
    CUDA13_REGISTRY_KEY,
    RegistryRegistrationPlan,
    WindowsRegistryBackend,
    register_apps_and_features,
    registry_plan_from_record,
)

REG_SZ = 1
REG_DWORD = 4
_KINDS = {"NoModify": REG_DWORD, "NoRepair": REG_DWORD}
_LIMIT = 1024 * 1024


class RegistrationRepairError(RuntimeError):
    """This registration cannot be safely reviewed or repaired."""


class RegistrationRepairCancelled(RegistrationRepairError):
    """Cancellation stopped repair; any owned partial change is rolled back."""


def _require(value: object, message: str) -> None:
    if not value:
        raise RegistrationRepairError(message)


def _checkpoint(cancellation: object | None) -> None:
    if cancellation is None:
        return
    check = getattr(cancellation, "is_set", None)
    if (callable(check) and check()) or (callable(cancellation) and cancellation()):
        raise RegistrationRepairCancelled("Windows entry repair was cancelled.")


def _path_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))


def _direct(path: Path, *, file: bool = False) -> None:
    _require(path.is_absolute(), "Repair paths must be absolute.")
    for item in (path, *path.parents):
        if os.path.lexists(item):
            metadata = item.lstat()
            _require(
                not stat.S_ISLNK(metadata.st_mode)
                and not getattr(metadata, "st_file_attributes", 0) & 0x400,
                "Redirected repair paths are not accepted.",
            )
    if file:
        _require(
            stat.S_ISREG(path.lstat().st_mode), "A regular cached EXE is required."
        )


def _sha(path: Path) -> str:
    _direct(path, file=True)
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass(frozen=True, slots=True)
class RegistryValue:
    name: str
    kind: int
    value: object


@dataclass(frozen=True, slots=True)
class RegistrationSnapshot:
    values: tuple[RegistryValue, ...]
    subkey_count: int = 0
    last_write: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "values": [
                {
                    "name": item.name,
                    "kind": item.kind,
                    "value": (
                        {"base64": base64.b64encode(item.value).decode("ascii")}
                        if isinstance(item.value, bytes)
                        else item.value
                    ),
                }
                for item in self.values
            ],
            "subkey_count": self.subkey_count,
            "last_write": self.last_write,
        }


def _values(snapshot: RegistrationSnapshot) -> dict[str, str | int]:
    _require(snapshot.subkey_count == 0, "An entry with subkeys cannot be repaired.")
    result = {}
    names = set()
    for item in snapshot.values:
        key = item.name.casefold()
        _require(key not in names, "Ambiguous Windows value names.")
        names.add(key)
        _require(
            (item.kind == REG_SZ and type(item.value) is str)
            or (item.kind == REG_DWORD and type(item.value) is int),
            "Unsupported Windows registry value kind; the entry was preserved.",
        )
        result[item.name] = item.value
    return result


def _typed(values: Mapping[str, str | int]) -> tuple[RegistryValue, ...]:
    return tuple(
        sorted(
            (
                RegistryValue(name, _KINDS.get(name, REG_SZ), value)
                for name, value in values.items()
            ),
            key=lambda value: value.name.casefold(),
        )
    )


def _same_data(first: RegistrationSnapshot, second: RegistrationSnapshot) -> bool:
    return first.values == second.values and first.subkey_count == second.subkey_count


class WindowsRegistrationRepairBackend:
    """Typed HKCU reader and conditional changed-name patch, never key replace."""

    @staticmethod
    def _snapshot(handle) -> RegistrationSnapshot:
        import winreg

        before = winreg.QueryInfoKey(handle)
        _require(before[1] <= 32, "Too many Windows entry values for safe repair.")
        rows = []
        for index in range(before[1]):
            # Unexpected errors are failures, not an inferred end-of-enumeration.
            name, value, kind = winreg.EnumValue(handle, index)
            _require(len(name) <= 512, "Oversized Windows registry value name.")
            if isinstance(value, list):
                value = tuple(value)
            rows.append(RegistryValue(name, kind, value))
        try:
            winreg.EnumValue(handle, before[1])
        except OSError as exc:
            _require(
                getattr(exc, "winerror", None) == 259,
                "Windows registry enumeration did not finish normally.",
            )
        else:
            raise RegistrationRepairError("Windows entry changed during enumeration.")
        _require(
            winreg.QueryInfoKey(handle) == before,
            "Windows entry changed during enumeration.",
        )
        snapshot = RegistrationSnapshot(
            tuple(sorted(rows, key=lambda value: value.name.casefold())),
            before[0],
            before[2],
        )
        _require(
            len(json.dumps(snapshot.as_dict()).encode()) <= _LIMIT,
            "Oversized Windows entry snapshot.",
        )
        return snapshot

    def read_snapshot(self, key: str) -> RegistrationSnapshot | None:
        import winreg

        _require(key in {CPU_REGISTRY_KEY, CUDA13_REGISTRY_KEY}, "Wrong repair key.")
        try:
            handle = winreg.OpenKey(winreg.HKEY_CURRENT_USER, key)
        except FileNotFoundError:
            return None
        with handle:
            return self._snapshot(handle)

    def write_repair(self, key, expected, values, *, cancellation=None) -> None:
        import winreg

        _require(key in {CPU_REGISTRY_KEY, CUDA13_REGISTRY_KEY}, "Wrong repair key.")
        desired = RegistrationSnapshot(_typed(values))
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            key,
            access=winreg.KEY_QUERY_VALUE | winreg.KEY_SET_VALUE,
        ) as handle:
            observed = self._snapshot(handle)
            _require(observed == expected, "Windows entry changed after confirmation.")
            original = {item.name: item for item in expected.values}
            current = observed
            transitions = [observed]
            changed = []
            preferred = (
                "VippManagedRoot",
                "VippInstallationId",
                "VippManifestSha256",
                "VippUninstallerSha256",
            )
            ordered = sorted(values, key=lambda name: (name not in preferred, name))
            try:
                for name in ordered:
                    previous = original[name]
                    if previous.value == values[name]:
                        continue
                    _checkpoint(cancellation)
                    _require(
                        self._snapshot(handle) == current,
                        "Windows entry changed during repair; preserved.",
                    )
                    planned = {item.name: item for item in current.values}
                    planned[name] = RegistryValue(
                        name, _KINDS.get(name, REG_SZ), values[name]
                    )
                    next_data = RegistrationSnapshot(
                        tuple(
                            sorted(
                                planned.values(),
                                key=lambda value: value.name.casefold(),
                            )
                        )
                    )
                    transitions.append(next_data)
                    changed.append(name)
                    winreg.SetValueEx(
                        handle, name, 0, _KINDS.get(name, REG_SZ), values[name]
                    )
                    current = self._snapshot(handle)
                    _require(
                        _same_data(current, next_data),
                        "Windows entry changed during repair; preserved.",
                    )
                _require(
                    _same_data(self._snapshot(handle), desired),
                    "Windows entry repair did not verify.",
                )
            except BaseException:
                # Restore only the exact known old/new transition, never new
                # foreign values, kinds or subkeys that appeared concurrently.
                current = self._snapshot(handle)
                if any(_same_data(current, item) for item in transitions):
                    for name in reversed(changed):
                        _require(
                            self._snapshot(handle) == current,
                            "Windows entry changed; rollback was refused.",
                        )
                        value = original[name]
                        planned = {item.name: item for item in current.values}
                        planned[name] = value
                        winreg.SetValueEx(handle, name, 0, value.kind, value.value)
                        current = self._snapshot(handle)
                        _require(
                            _same_data(
                                current,
                                RegistrationSnapshot(
                                    tuple(
                                        sorted(
                                            planned.values(),
                                            key=lambda value: value.name.casefold(),
                                        )
                                    )
                                ),
                            ),
                            "Windows entry changed during rollback; preserved.",
                        )
                raise


def _backend(backend):
    if backend is None or type(backend) is WindowsRegistryBackend:
        return WindowsRegistrationRepairBackend()
    _require(
        callable(getattr(backend, "read_snapshot", None))
        and callable(getattr(backend, "write_repair", None)),
        "A complete typed registry backend is required for entry repair.",
    )
    return backend


@dataclass(frozen=True, slots=True)
class PreparedRegistrationRepair:
    fingerprint: str
    managed_root: Path
    local_app_data: Path
    track: ComputeTrack
    registered_version: str
    installed_version: str
    changes: tuple[tuple[str, str | int, str | int], ...]
    backup_path: Path
    summary: str
    ownership_manifest_sha256: str
    before: RegistrationSnapshot = field(repr=False)
    previous_plan: RegistryRegistrationPlan = field(repr=False)
    current_plan: RegistryRegistrationPlan = field(repr=False)
    old_uninstaller_state: str = "verified-present"


@dataclass(frozen=True, slots=True)
class RegistrationRepairResult:
    succeeded: bool
    message: str
    backup_path: Path
    observed: RegistrationSnapshot | None = field(default=None, repr=False)


class RegistrationRepairAuthorization:
    def __init__(self, fingerprint):
        self._fingerprint, self._used = fingerprint, False
        self._lock = threading.Lock()

    def consume(self, fingerprint):
        with self._lock:
            _require(not self._used, "This repair confirmation was already used.")
            _require(
                secrets.compare_digest(fingerprint, self._fingerprint),
                "This repair confirmation is stale or belongs to another review.",
            )
            self._used = True


def _cache_path(local_app_data, root, track, version, digest):
    return (
        local_app_data
        / "VIPP"
        / "installer"
        / "cache"
        / track.value
        / hashlib.sha256(_path_key(root).encode()).hexdigest()
        / version
        / digest
        / "VIPP-Setup.exe"
    )


def _prepare(managed_root, local_app_data, backend, backup_path):
    root, local = Path(managed_root), Path(local_app_data)
    _direct(root)
    _direct(local)
    _require(
        root == Path(os.path.abspath(root))
        and local == Path(os.path.abspath(local))
        and local.is_dir(),
        "Use the direct managed root and current account's local app-data folder.",
    )
    _require(
        backup_path.parent == local / "VIPP" / "installer" / "registration-repairs"
        and backup_path.suffix == ".json",
        "Repair backup must remain in this account's private installer state.",
    )
    uuid.UUID(backup_path.stem)
    ownership = inspect_ownership(root)
    record = ownership.record
    _require(
        ownership.state is OwnershipState.VALID
        and record is not None
        and bool(record.registry_key),
        "A complete current owned installation is required.",
    )
    _require(
        _path_key(root)
        == _path_key(local / "VIPP" / "environments" / record.track.value),
        "Windows entry repair requires the canonical current CPU/GPU managed root.",
    )
    current = registry_plan_from_record(record, ownership.manifest_sha256)
    expected_cache = _cache_path(
        local, root, record.track, record.version, record.uninstaller_sha256
    )
    _require(
        _path_key(current.uninstaller_path) == _path_key(expected_cache),
        "The current uninstaller is outside its canonical trusted cache.",
    )
    snapshot = backend.read_snapshot(current.key)
    _require(
        snapshot is not None, "There is no complete stale Windows entry to repair."
    )
    observed = _values(snapshot)
    names = {name.casefold(): name for name in observed}
    expected_names = {name.casefold() for name, _ in current.values}
    _require(
        set(names) == expected_names,
        "Incomplete or additional Windows entry values were preserved.",
    )
    for item in snapshot.values:
        canonical = next(
            name
            for name, _ in current.values
            if name.casefold() == item.name.casefold()
        )
        _require(
            item.name == canonical and item.kind == _KINDS.get(canonical, REG_SZ),
            "Noncanonical Windows value names or kinds were preserved.",
        )
    old_version = observed["DisplayVersion"]
    _require(
        Version(old_version) < Version(record.version),
        "Only a complete strictly older Windows registration can be repaired.",
    )
    old_hash, old_uninstaller_hash = (
        observed["VippManifestSha256"],
        observed["VippUninstallerSha256"],
    )
    _require(
        all(
            isinstance(value, str)
            and len(value) == 64
            and all(char in "0123456789abcdef" for char in value)
            for value in (old_hash, old_uninstaller_hash)
        ),
        "The previous Windows entry does not carry exact hashes.",
    )
    # Derive the previous path instead of trusting an arbitrary command string.
    # The explicit state check below verifies present bytes or records absence;
    # RegistryRegistrationPlan subsequently validates only the command/schema.
    old_path = _cache_path(local, root, record.track, old_version, old_uninstaller_hash)
    _direct(old_path)
    old_observed_digest = None
    if os.path.lexists(old_path):
        # Existing-but-wrong, redirected and dangling paths are never absence.
        old_observed_digest = _sha(old_path)
        _require(
            old_observed_digest == old_uninstaller_hash,
            "The old cached uninstaller hash changed; entry preserved.",
        )
        old_state = "verified-present"
    else:
        old_state = "absent"
    old_values = current.value_map
    old_values.update(
        {
            "DisplayVersion": old_version,
            "VippManifestSha256": old_hash,
            "VippUninstallerSha256": old_uninstaller_hash,
            "UninstallString": subprocess.list2cmdline(
                (str(old_path), "--uninstall", "--managed-root", str(root))
            ),
        }
    )
    # This constructor validates the exact complete canonical command/schema
    # without claiming an absent historical EXE or manifest has been verified.
    # Current ownership/current verified EXE and explicit consent are authority;
    # the retired program is never executed, recreated or adopted as current.
    previous = replace(
        current,
        values=tuple(old_values.items()),
        manifest_sha256=old_hash,
        uninstaller_path=old_path,
        uninstaller_sha256=old_uninstaller_hash,
    )
    _require(
        observed == previous.value_map,
        "The stale entry is not this exact complete owned CPU/GPU registration.",
    )
    changes = tuple(
        (name, observed[name], value)
        for name, value in current.values
        if observed[name] != value
    )
    identity = {
        "root": str(root),
        "local_app_data": str(local),
        "backup_path": str(backup_path),
        "manifest": ownership.manifest_sha256,
        "before": snapshot.as_dict(),
        "desired": list(current.values),
        "old_exe_path": str(old_path),
        "old_exe_state": old_state,
        "old_exe_observed_sha256": old_observed_digest,
        "current_exe": _sha(current.uninstaller_path),
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()
    return PreparedRegistrationRepair(
        fingerprint,
        root,
        local,
        record.track,
        old_version,
        record.version,
        changes,
        backup_path,
        "Repair Windows Installed apps information only: "
        f"{old_version} → {record.version}. No packages or workflows will change."
        + (
            " The older cached uninstaller is unavailable and will not be run; "
            "its historical bytes are not verified."
            if old_state == "absent"
            else ""
        ),
        ownership.manifest_sha256,
        snapshot,
        previous,
        current,
        old_state,
    )


def prepare_registration_repair(managed_root, local_app_data, backend=None):
    """Read-only review of the narrow complete stale-registration case."""
    local = Path(local_app_data)
    backup = (
        local / "VIPP" / "installer" / "registration-repairs" / f"{uuid.uuid4()}.json"
    )
    _direct(backup)
    return _prepare(managed_root, local, _backend(backend), backup)


def authorize_registration_repair(prepared, *, confirmed: bool):
    _require(confirmed is True, "Windows entry repair must be explicitly confirmed.")
    return RegistrationRepairAuthorization(prepared.fingerprint)


def apply_registration_repair(
    prepared, authorization, backend=None, *, cancellation=None
):
    """Hold the shared install/uninstall lock throughout the repair transaction."""
    from napari_vipp.installer.engine import _TargetLock

    authorization.consume(prepared.fingerprint)
    selected = _backend(backend)
    _checkpoint(cancellation)
    # Refuse forged target/state fields before creating even an installer lock.
    # Repeat this full canonical validation after acquiring the cooperative lock.
    before_lock = _prepare(
        prepared.managed_root, prepared.local_app_data, selected, prepared.backup_path
    )
    _require(before_lock == prepared, "Repair review changed; check again.")
    state = prepared.local_app_data / "VIPP" / "installer"
    _direct(state)
    _direct(state / "locks")
    lock = _TargetLock(
        state, prepared.managed_root, f"registration-repair-{uuid.uuid4()}"
    )
    _direct(lock.path)
    with lock:
        return _apply_registration_repair_locked(prepared, selected, cancellation)


def _apply_registration_repair_locked(prepared, selected, cancellation):
    """Caller holds the ordinary managed-target transaction lock."""
    _checkpoint(cancellation)
    fresh = _prepare(
        prepared.managed_root, prepared.local_app_data, selected, prepared.backup_path
    )
    _require(fresh == prepared, "Repair review changed; check again.")
    _direct(prepared.backup_path)
    prepared.backup_path.parent.mkdir(parents=True, exist_ok=True)
    _direct(prepared.backup_path.parent)
    backup = {
        "schema": "napari-vipp-registration-repair-backup",
        "schema_version": 1,
        "prepared_fingerprint": prepared.fingerprint,
        "saved_utc": datetime.now(UTC).isoformat(),
        "managed_root": str(prepared.managed_root),
        "ownership_manifest_sha256": prepared.ownership_manifest_sha256,
        "key": prepared.current_plan.key,
        "retired_uninstaller": {
            "state": prepared.old_uninstaller_state,
            "path": str(prepared.previous_plan.uninstaller_path),
            "expected_sha256": prepared.previous_plan.uninstaller_sha256,
            "observed_sha256": (
                prepared.previous_plan.uninstaller_sha256
                if prepared.old_uninstaller_state == "verified-present"
                else None
            ),
            "historical_exe_verified": prepared.old_uninstaller_state
            == "verified-present",
            "will_be_executed": False,
        },
        "before": prepared.before.as_dict(),
        "desired": RegistrationSnapshot(
            _typed(prepared.current_plan.value_map)
        ).as_dict(),
    }
    with prepared.backup_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(backup, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    _checkpoint(cancellation)
    fresh = _prepare(
        prepared.managed_root, prepared.local_app_data, selected, prepared.backup_path
    )
    _require(
        fresh == prepared,
        "Windows entry, ownership or cached EXE changed before repair.",
    )

    class Adapter:
        def read_values(self, key):
            observed = selected.read_snapshot(key)
            _require(observed is not None, "Windows entry disappeared.")
            target = RegistrationSnapshot(_typed(prepared.current_plan.value_map))
            _require(
                observed == prepared.before or _same_data(observed, target),
                "Windows entry changed; repair was refused.",
            )
            return _values(observed)

        def write_values(self, key, values):
            _checkpoint(cancellation)
            checked = _prepare(
                prepared.managed_root,
                prepared.local_app_data,
                selected,
                prepared.backup_path,
            )
            _require(
                checked == prepared,
                "Windows entry or owned files changed immediately before repair.",
            )
            selected.write_repair(
                key, prepared.before, values, cancellation=cancellation
            )

    try:
        register_apps_and_features(
            Adapter(), prepared.current_plan, previous_plan=prepared.previous_plan
        )
        observed = selected.read_snapshot(prepared.current_plan.key)
        _require(
            observed is not None
            and _same_data(
                observed, RegistrationSnapshot(_typed(prepared.current_plan.value_map))
            ),
            "Windows entry did not verify after repair.",
        )
        final_ownership = inspect_ownership(prepared.managed_root)
        _require(
            final_ownership.state is OwnershipState.VALID
            and final_ownership.record is not None
            and final_ownership.manifest_sha256 == prepared.ownership_manifest_sha256,
            "Current ownership changed during Windows entry repair.",
        )
        # This reconstructs the plan and rehashes the current cached EXE; a
        # changed/noncooperative owner cannot turn a stale repair into success.
        _require(
            registry_plan_from_record(
                final_ownership.record, final_ownership.manifest_sha256
            )
            == prepared.current_plan,
            "Current owned uninstaller changed during Windows entry repair.",
        )
        return RegistrationRepairResult(
            True,
            "Windows Installed apps information was repaired. "
            "Check setup again to continue.",
            prepared.backup_path,
            observed,
        )
    except RegistrationRepairCancelled:
        raise
    except Exception as exc:
        return RegistrationRepairResult(
            False,
            f"Windows entry repair was incomplete; its backup was preserved. {exc}",
            prepared.backup_path,
            selected.read_snapshot(prepared.current_plan.key),
        )
