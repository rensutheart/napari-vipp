from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
import uuid
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

import napari_vipp.installer.uninstall as uninstall_module
from napari_vipp.installer.models import ComputeTrack
from napari_vipp.installer.ownership import (
    OwnedEnvironment,
    OwnedPackage,
    OwnedShortcut,
    OwnershipRecord,
    OwnershipState,
    inspect_ownership,
    managed_environments_root,
    write_ownership_record,
)
from napari_vipp.installer.uninstall import (
    CPU_REGISTRY_KEY,
    CUDA13_REGISTRY_KEY,
    ManagedUninstaller,
    RegistryOwnershipError,
    UninstallAuthorizationError,
    UninstallPreparationError,
    UninstallStatus,
    WindowsRegistryBackend,
    build_deferred_self_delete,
    persistent_uninstaller_destination,
    read_windows_shortcut_target,
    reap_completed_uninstall_recovery,
    register_apps_and_features,
    registry_key_for_track,
    registry_plan_from_record,
    remove_apps_and_features,
    remove_superseded_persistent_uninstaller,
    stage_persistent_uninstaller,
)

_NOW = datetime(2026, 8, 9, 12, 0, tzinfo=UTC).isoformat()
_MARKER_NAME = ".vipp-install-candidate.json"


@pytest.mark.skipif(os.name != "nt", reason="Windows shortcut inspection only")
def test_shortcut_inspection_does_not_open_a_console(tmp_path, monkeypatch):
    shortcut = tmp_path / "VIPP.lnk"
    shortcut.touch()
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(argv, 0, stdout="C:\\VIPP\\vipp.exe\n")

    monkeypatch.setattr(uninstall_module.subprocess, "run", fake_run)

    target = read_windows_shortcut_target(shortcut)

    assert target == Path("C:\\VIPP\\vipp.exe")
    assert captured["kwargs"]["stdin"] is subprocess.DEVNULL
    assert captured["kwargs"]["creationflags"] & getattr(
        subprocess,
        "CREATE_NO_WINDOW",
        0x08000000,
    )


class MemoryRegistry:
    def __init__(self) -> None:
        self.keys: dict[str, dict[str, str | int]] = {}

    def read_values(self, key: str) -> Mapping[str, str | int] | None:
        values = self.keys.get(key)
        return None if values is None else dict(values)

    def write_values(
        self,
        key: str,
        values: Mapping[str, str | int],
    ) -> None:
        self.keys[key] = dict(values)

    def delete_key(self, key: str) -> None:
        self.keys.pop(key, None)


def _environment(path: Path, token: str) -> str:
    path.mkdir(parents=True)
    marker = path / _MARKER_NAME
    marker.write_text(
        json.dumps(
            {
                "schema": "napari-vipp-install-candidate",
                "schema_version": 1,
                "run_id": token,
                "resolution_id": "a" * 64,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    scripts = path / "Scripts"
    scripts.mkdir()
    (scripts / "vipp-cpu.exe").write_bytes(b"launcher")
    return hashlib.sha256(marker.read_bytes()).hexdigest()


def _record(
    tmp_path: Path,
    *,
    shortcut: bool = False,
    shortcut_target: bool = False,
    retired: bool = False,
    uninstaller: bool = False,
    registry: bool = False,
    track: ComputeTrack = ComputeTrack.CPU,
) -> tuple[OwnershipRecord, Path, Path | None]:
    root = tmp_path / "managed"
    store = managed_environments_root(root)
    active = store / "0.13.0-active"
    active_digest = _environment(active, str(uuid.uuid4()))
    retired_items: tuple[OwnedEnvironment, ...] = ()
    if retired:
        retired_path = store / "0.12.0-retired"
        retired_digest = _environment(retired_path, str(uuid.uuid4()))
        retired_items = (OwnedEnvironment(retired_path, retired_digest),)
    shortcut_path: Path | None = None
    shortcut_items: tuple[OwnedShortcut, ...] = ()
    if shortcut:
        desktop = tmp_path / "Desktop"
        desktop.mkdir()
        shortcut_path = desktop / "VIPP.lnk"
        shortcut_path.write_bytes(b"owned shortcut")
        shortcut_items = (
            OwnedShortcut(
                shortcut_path,
                hashlib.sha256(shortcut_path.read_bytes()).hexdigest(),
                target=(active / "Scripts" / "vipp-cpu.exe")
                if shortcut_target
                else None,
            ),
        )
    uninstaller_path: Path | None = None
    uninstaller_digest = ""
    if uninstaller:
        directory = tmp_path / "persistent-uninstaller"
        directory.mkdir()
        uninstaller_path = directory / "VIPP-Setup.exe"
        uninstaller_path.write_bytes(b"frozen setup executable")
        uninstaller_digest = hashlib.sha256(uninstaller_path.read_bytes()).hexdigest()
    record = OwnershipRecord(
        installation_id=str(uuid.uuid4()),
        managed_root=root,
        environment_root=active,
        distribution="napari-vipp",
        version="0.13.0a4",
        track=track,
        base_python=tmp_path / "python.exe",
        resolved_plan_id="b" * 64,
        packages=(OwnedPackage("napari-vipp", "0.13.0a4", "c" * 64),),
        created_at=_NOW,
        updated_at=_NOW,
        environment_marker_sha256=active_digest,
        managed_root_preexisting=True,
        shortcuts=shortcut_items,
        retired_environments=retired_items,
        uninstaller_path=uninstaller_path,
        uninstaller_sha256=uninstaller_digest,
        registry_key=(registry_key_for_track(track) if registry else ""),
    )
    write_ownership_record(root, record)
    return record, root, shortcut_path


def _journal_files(record: OwnershipRecord) -> tuple[Path, ...]:
    assert record.uninstaller_path is not None
    return tuple(record.uninstaller_path.parent.glob(".vipp-uninstall-*.json"))


def _registered_case(
    tmp_path: Path,
    *,
    shortcut: bool = False,
    track: ComputeTrack = ComputeTrack.CPU,
) -> tuple[OwnershipRecord, Path, Path | None, MemoryRegistry]:
    record, root, shortcut_path = _record(
        tmp_path,
        shortcut=shortcut,
        uninstaller=True,
        registry=True,
        track=track,
    )
    registry = MemoryRegistry()
    plan = registry_plan_from_record(
        record,
        inspect_ownership(root).manifest_sha256,
    )
    register_apps_and_features(registry, plan)
    return record, root, shortcut_path, registry


class SimulatedPowerLoss(BaseException):
    pass


def test_prepare_and_apply_remove_only_hash_owned_objects(tmp_path: Path) -> None:
    record, root, shortcut = _record(tmp_path, shortcut=True, retired=True)
    unrelated = root / "research-notes.txt"
    unrelated.write_text("keep me", encoding="utf-8")
    uninstaller = ManagedUninstaller()

    prepared = uninstaller.prepare(root, shortcut_roots=(tmp_path / "Desktop",))
    result = uninstaller.apply(prepared, uninstaller.authorize(prepared))

    assert result.status is UninstallStatus.COMPLETED
    assert shortcut is not None and not shortcut.exists()
    assert not record.environment_root.exists()
    assert all(not item.path.exists() for item in record.retired_environments)
    assert inspect_ownership(root).state is OwnershipState.ABSENT
    assert unrelated.read_text(encoding="utf-8") == "keep me"
    assert root.exists()


def test_marker_mismatch_during_prepare_refuses_everything(tmp_path: Path) -> None:
    record, root, shortcut = _record(tmp_path, shortcut=True)
    (record.environment_root / _MARKER_NAME).write_text("tampered", encoding="utf-8")

    with pytest.raises(UninstallPreparationError, match="changed and was preserved"):
        ManagedUninstaller().prepare(
            root,
            shortcut_roots=(tmp_path / "Desktop",),
        )

    assert shortcut is not None and shortcut.exists()
    assert inspect_ownership(root).state is OwnershipState.VALID


def test_marker_race_returns_incomplete_with_exact_path(tmp_path: Path) -> None:
    record, root, _shortcut = _record(tmp_path)
    uninstaller = ManagedUninstaller()
    prepared = uninstaller.prepare(root, shortcut_roots=(tmp_path / "Desktop",))
    (record.environment_root / _MARKER_NAME).write_text("changed", encoding="utf-8")

    result = uninstaller.apply(prepared, uninstaller.authorize(prepared))

    assert result.status is UninstallStatus.INCOMPLETE
    assert result.preserved_paths == (record.environment_root,)
    assert str(record.environment_root) in result.message
    assert "cleanup is incomplete" in result.message
    assert record.environment_root.exists()
    assert inspect_ownership(root).state is OwnershipState.VALID


def test_shortcut_outside_reviewed_roots_is_never_removed(tmp_path: Path) -> None:
    _record_value, root, shortcut = _record(tmp_path, shortcut=True)
    wrong_root = tmp_path / "Start Menu"
    wrong_root.mkdir()

    with pytest.raises(UninstallPreparationError, match="outside the reviewed"):
        ManagedUninstaller().prepare(root, shortcut_roots=(wrong_root,))

    assert shortcut is not None and shortcut.exists()


def test_hash_owned_shortcut_target_is_rechecked_before_removal(
    tmp_path: Path,
) -> None:
    record, root, shortcut = _record(
        tmp_path,
        shortcut=True,
        shortcut_target=True,
    )
    expected = record.environment_root / "Scripts" / "vipp-cpu.exe"
    observed = [expected]
    uninstaller = ManagedUninstaller(shortcut_target_reader=lambda _path: observed[0])
    prepared = uninstaller.prepare(root, shortcut_roots=(tmp_path / "Desktop",))
    observed[0] = tmp_path / "foreign.exe"

    result = uninstaller.apply(prepared, uninstaller.authorize(prepared))

    assert result.status is UninstallStatus.INCOMPLETE
    assert shortcut is not None and shortcut.exists()
    assert any(issue.path == shortcut for issue in result.issues)


def test_tree_containing_symlink_is_refused(tmp_path: Path) -> None:
    record, root, _shortcut = _record(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "valuable.txt").write_text("keep", encoding="utf-8")
    link = record.environment_root / "redirected"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Creating a test symlink is not permitted on this Windows host.")

    with pytest.raises(UninstallPreparationError, match="reparse point"):
        ManagedUninstaller().prepare(root, shortcut_roots=(tmp_path / "Desktop",))

    assert (outside / "valuable.txt").exists()


def test_authorization_is_single_use(tmp_path: Path) -> None:
    _record_value, root, _shortcut = _record(tmp_path)
    uninstaller = ManagedUninstaller()
    prepared = uninstaller.prepare(root, shortcut_roots=(tmp_path / "Desktop",))
    authorization = uninstaller.authorize(prepared)

    first = uninstaller.apply(prepared, authorization)
    assert first.completed
    with pytest.raises(UninstallAuthorizationError, match="already been used"):
        uninstaller.apply(prepared, authorization)


def test_registry_registration_and_removal_are_manifest_bound(
    tmp_path: Path,
) -> None:
    record, root, _shortcut = _record(
        tmp_path,
        uninstaller=True,
        registry=True,
    )
    manifest_hash = inspect_ownership(root).manifest_sha256
    plan = registry_plan_from_record(record, manifest_hash)
    registry = MemoryRegistry()

    register_apps_and_features(registry, plan)

    values = registry.keys[plan.key]
    assert values["VippManagedRoot"] == str(root)
    assert values["VippInstallationId"] == record.installation_id
    assert values["VippManifestSha256"] == manifest_hash
    assert "--uninstall" in str(values["UninstallString"])
    assert str(root) in str(values["UninstallString"])
    assert "QuietUninstallString" not in values
    values["VippInstallationId"] = str(uuid.uuid4())
    with pytest.raises(RegistryOwnershipError, match="changed"):
        remove_apps_and_features(registry, plan)
    assert plan.key in registry.keys

    registry.keys[plan.key] = plan.value_map
    registry.keys[plan.key]["DisplayName"] = "Changed by another program"
    with pytest.raises(RegistryOwnershipError, match="changed"):
        remove_apps_and_features(registry, plan)
    assert plan.key in registry.keys

    registry.keys[plan.key] = plan.value_map
    remove_apps_and_features(registry, plan)
    assert plan.key not in registry.keys


def test_registration_never_overwrites_a_foreign_apps_entry(tmp_path: Path) -> None:
    record, root, _shortcut = _record(
        tmp_path,
        uninstaller=True,
        registry=True,
    )
    plan = registry_plan_from_record(record, inspect_ownership(root).manifest_sha256)
    registry = MemoryRegistry()
    registry.keys[plan.key] = {
        "DisplayName": "Someone else's VIPP",
        "VippManagedRoot": str(tmp_path / "other"),
        "VippInstallationId": str(uuid.uuid4()),
        "VippManifestSha256": "d" * 64,
        "VippUninstallerSha256": "e" * 64,
    }
    original = dict(registry.keys[plan.key])

    with pytest.raises(RegistryOwnershipError, match="different installation"):
        register_apps_and_features(registry, plan)

    assert registry.keys[plan.key] == original


def test_registration_can_replace_only_an_explicit_prior_owned_plan(
    tmp_path: Path,
) -> None:
    record, root, _shortcut = _record(
        tmp_path,
        uninstaller=True,
        registry=True,
    )
    inspection = inspect_ownership(root)
    previous = registry_plan_from_record(record, inspection.manifest_sha256)
    registry = MemoryRegistry()
    registry.keys[previous.key] = previous.value_map
    updated_record = replace(record, resolved_plan_id="e" * 64)
    write_ownership_record(root, updated_record)
    # Use the actual new manifest hash while retaining the old key binding as
    # the explicit authority that permits an in-place registry refresh.
    current_hash = inspect_ownership(root).manifest_sha256
    current = registry_plan_from_record(updated_record, current_hash)

    register_apps_and_features(registry, current, previous_plan=previous)

    assert registry.keys[current.key]["VippManifestSha256"] == current_hash


def test_interrupted_registry_value_swap_is_repaired_only_in_recovery(
    tmp_path: Path,
) -> None:
    record, root, _shortcut = _record(
        tmp_path,
        uninstaller=True,
        registry=True,
    )
    previous = registry_plan_from_record(
        record,
        inspect_ownership(root).manifest_sha256,
    )
    updated_record = replace(record, resolved_plan_id="e" * 64)
    write_ownership_record(root, updated_record)
    current = registry_plan_from_record(
        updated_record,
        inspect_ownership(root).manifest_sha256,
    )
    registry = MemoryRegistry()
    partial = previous.value_map
    partial["VippManifestSha256"] = current.manifest_sha256
    partial.pop("Publisher")
    registry.keys[current.key] = partial

    with pytest.raises(RegistryOwnershipError, match="different installation"):
        register_apps_and_features(
            registry,
            current,
            previous_plan=previous,
        )

    register_apps_and_features(
        registry,
        current,
        previous_plan=previous,
        recover_interrupted=True,
    )

    assert registry.keys[current.key] == current.value_map


def test_windows_registry_backend_restores_exact_snapshot_after_write_error(
    monkeypatch,
) -> None:
    store = {CPU_REGISTRY_KEY: {"LegacyValue": "keep exactly"}}
    set_calls = 0

    class _Handle:
        def __init__(self, key):
            self.key = key

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class _FakeWinreg:
        HKEY_CURRENT_USER = object()
        KEY_SET_VALUE = 1
        KEY_QUERY_VALUE = 2
        REG_DWORD = 4
        REG_SZ = 1

        @staticmethod
        def OpenKey(_root, key):
            if key not in store:
                raise FileNotFoundError(key)
            return _Handle(key)

        @staticmethod
        def CreateKeyEx(_root, key, access):
            assert access == 3
            store.setdefault(key, {})
            return _Handle(key)

        @staticmethod
        def EnumValue(handle, index):
            try:
                name, value = tuple(store[handle.key].items())[index]
            except IndexError as exc:
                raise OSError("end") from exc
            return name, value, 1

        @staticmethod
        def SetValueEx(handle, name, _reserved, _kind, value):
            nonlocal set_calls
            set_calls += 1
            if set_calls == 3:
                raise OSError("simulated mid-write registry failure")
            store[handle.key][name] = value

        @staticmethod
        def DeleteValue(handle, name):
            del store[handle.key][name]

        @staticmethod
        def DeleteKey(_root, key):
            try:
                del store[key]
            except KeyError as exc:
                raise FileNotFoundError(key) from exc

    monkeypatch.setitem(sys.modules, "winreg", _FakeWinreg)
    desired = {
        "DisplayName": "VIPP (CPU)",
        "VippManagedRoot": "C:/VIPP",
        "VippInstallationId": "installation-id",
        "VippManifestSha256": "a" * 64,
        "VippUninstallerSha256": "b" * 64,
    }

    with pytest.raises(OSError, match="mid-write"):
        WindowsRegistryBackend().write_values(CPU_REGISTRY_KEY, desired)

    assert store[CPU_REGISTRY_KEY] == {"LegacyValue": "keep exactly"}


def test_cpu_and_cuda_registry_keys_are_distinct() -> None:
    assert registry_key_for_track(ComputeTrack.CPU) == CPU_REGISTRY_KEY
    assert registry_key_for_track(ComputeTrack.CUDA13) == CUDA13_REGISTRY_KEY
    assert CPU_REGISTRY_KEY != CUDA13_REGISTRY_KEY


def test_cpu_and_cuda_use_distinct_novice_labels_and_cached_executables(
    tmp_path: Path,
) -> None:
    cpu, root, _shortcut = _record(
        tmp_path / "cpu-case",
        uninstaller=True,
        registry=True,
    )
    cpu_plan = registry_plan_from_record(
        cpu,
        inspect_ownership(root).manifest_sha256,
    )
    assert cpu_plan.value_map["DisplayName"] == "VIPP (CPU)"

    gpu, gpu_root, _shortcut = _record(
        tmp_path / "gpu-case",
        uninstaller=True,
        registry=True,
        track=ComputeTrack.CUDA13,
    )
    gpu_plan = registry_plan_from_record(
        gpu,
        inspect_ownership(gpu_root).manifest_sha256,
    )
    assert gpu_plan.key != cpu_plan.key
    assert gpu_plan.value_map["DisplayName"] == "VIPP (GPU)"

    cpu_path = persistent_uninstaller_destination(
        tmp_path,
        ComputeTrack.CPU,
        "0.13.0",
    )
    gpu_path = persistent_uninstaller_destination(
        tmp_path,
        ComputeTrack.CUDA13,
        "0.13.0",
    )
    assert cpu_path != gpu_path
    assert cpu_path.name == gpu_path.name == "VIPP-Setup.exe"
    assert "0.13.0" in cpu_path.parts


def test_managed_uninstall_removes_matching_registry_and_defers_self_delete(
    tmp_path: Path,
) -> None:
    record, root, _shortcut = _record(
        tmp_path,
        uninstaller=True,
        registry=True,
    )
    registry = MemoryRegistry()
    plan = registry_plan_from_record(record, inspect_ownership(root).manifest_sha256)
    register_apps_and_features(registry, plan)
    uninstaller = ManagedUninstaller(registry=registry)
    prepared = uninstaller.prepare(root, shortcut_roots=(tmp_path / "Desktop",))

    result = uninstaller.apply(
        prepared,
        uninstaller.authorize(prepared),
        current_executable=record.uninstaller_path,
        current_pid=1234,
    )

    assert result.completed
    assert plan.key not in registry.keys
    assert result.deferred_self_delete is not None
    assert result.deferred_self_delete.target == record.uninstaller_path
    assert result.deferred_self_delete.wait_for_pid == 1234
    assert record.uninstaller_path is not None and record.uninstaller_path.exists()


def test_missing_registry_backend_is_reported_as_incomplete(tmp_path: Path) -> None:
    _record_value, root, _shortcut = _record(
        tmp_path,
        uninstaller=True,
        registry=True,
    )
    uninstaller = ManagedUninstaller()
    prepared = uninstaller.prepare(root, shortcut_roots=(tmp_path / "Desktop",))

    result = uninstaller.apply(prepared, uninstaller.authorize(prepared))

    assert result.status is UninstallStatus.INCOMPLETE
    assert any("registry backend" in issue.error for issue in result.issues)
    assert inspect_ownership(root).state is OwnershipState.ABSENT
    assert _journal_files(_record_value)
    assert _record_value.uninstaller_path is not None
    assert _record_value.uninstaller_path.exists()


def test_persistent_uninstaller_is_staged_atomically_outside_managed_root(
    tmp_path: Path,
) -> None:
    source = tmp_path / "release" / "VIPP-Setup.exe"
    source.parent.mkdir()
    source.write_bytes(b"release bytes")
    managed = tmp_path / "managed"
    destination = tmp_path / "persistent" / "VIPP-Setup.exe"

    staged = stage_persistent_uninstaller(
        source,
        destination,
        managed_root=managed,
    )

    assert staged.path == destination
    assert staged.sha256 == hashlib.sha256(b"release bytes").hexdigest()
    assert destination.read_bytes() == b"release bytes"

    with pytest.raises(UninstallPreparationError, match="outside"):
        stage_persistent_uninstaller(
            source,
            managed / "uninstall.exe",
            managed_root=managed,
        )

    destination.write_bytes(b"foreign replacement")
    with pytest.raises(UninstallPreparationError, match="ownership hash"):
        stage_persistent_uninstaller(
            source,
            destination,
            managed_root=managed,
        )


def test_superseded_uninstaller_cleanup_is_hash_bound(tmp_path: Path) -> None:
    old = tmp_path / "Uninstallers" / "0.12" / "VIPP-Setup.exe"
    current = tmp_path / "Uninstallers" / "0.13" / "VIPP-Setup.exe"
    old.parent.mkdir(parents=True)
    current.parent.mkdir(parents=True)
    old.write_bytes(b"old owned setup")
    current.write_bytes(b"current setup")
    digest = hashlib.sha256(old.read_bytes()).hexdigest()

    assert remove_superseded_persistent_uninstaller(
        old,
        digest,
        current_path=current,
    )
    assert not old.exists()
    assert current.exists()

    old.write_bytes(b"changed foreign bytes")
    with pytest.raises(UninstallPreparationError, match="changed and was preserved"):
        remove_superseded_persistent_uninstaller(
            old,
            digest,
            current_path=current,
        )
    assert old.exists()


def test_deferred_self_delete_uses_encoded_literal_command(tmp_path: Path) -> None:
    target = tmp_path / "Odd ' name & setup.exe"
    target.write_bytes(b"exe")

    request = build_deferred_self_delete(target, wait_for_pid=72)

    assert "-EncodedCommand" in request.argv
    encoded = request.argv[-1]
    script = base64.b64decode(encoded).decode("utf-16-le")
    assert "Wait-Process -Id 72" in script
    assert r"Local\VIPP.Setup.SingleInstance" in script
    assert "AbandonedMutexException" in script
    assert "$mx.ReleaseMutex()" in script
    assert "-LiteralPath" in script
    assert "Get-FileHash" not in script
    assert "function Get-VippSha256" in script
    assert "[System.Security.Cryptography.SHA256]::Create()" in script
    assert str(target).replace("'", "''") in script
    assert all(str(target) not in argument for argument in request.argv)


@pytest.mark.skipif(os.name != "nt", reason="exercises the Windows helper process")
def test_deferred_helper_removes_matching_cache_and_journal_without_modules(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    record, root, _shortcut, registry = _registered_case(tmp_path)

    def crash_before_cache_cleanup(*_args, **_kwargs):
        raise SimulatedPowerLoss("terminal uninstall")

    monkeypatch.setattr(
        ManagedUninstaller,
        "_remove_or_defer_uninstaller",
        staticmethod(crash_before_cache_cleanup),
    )
    uninstaller = ManagedUninstaller(registry=registry)
    prepared = uninstaller.prepare(root, shortcut_roots=(tmp_path / "Desktop",))
    with pytest.raises(SimulatedPowerLoss):
        uninstaller.apply(prepared, uninstaller.authorize(prepared))

    assert record.uninstaller_path is not None
    journal = _journal_files(record)[0]
    request = build_deferred_self_delete(
        record.uninstaller_path,
        wait_for_pid=2_147_483_647,
        expected_sha256=record.uninstaller_sha256,
        journal_path=journal,
        journal_sha256=hashlib.sha256(journal.read_bytes()).hexdigest(),
    )
    hostile_modules = tmp_path / "empty-powershell-modules"
    hostile_modules.mkdir()
    environment = dict(os.environ)
    environment["PSModulePath"] = str(hostile_modules)

    completed = subprocess.run(  # noqa: S603 - reviewed encoded helper argv
        request.argv,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        env=environment,
        timeout=60,
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )

    assert completed.returncode == 0, (completed.stdout, completed.stderr)
    assert not record.uninstaller_path.exists()
    assert not journal.exists()
    assert not record.uninstaller_path.parent.exists()


@pytest.mark.skipif(os.name != "nt", reason="exercises the Windows helper process")
def test_delayed_deferred_helper_preserves_same_version_reinstall(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    record, root, _shortcut, registry = _registered_case(tmp_path)

    def crash_before_cache_cleanup(*_args, **_kwargs):
        raise SimulatedPowerLoss("terminal uninstall")

    monkeypatch.setattr(
        ManagedUninstaller,
        "_remove_or_defer_uninstaller",
        staticmethod(crash_before_cache_cleanup),
    )
    first = ManagedUninstaller(registry=registry)
    prepared = first.prepare(root, shortcut_roots=(tmp_path / "Desktop",))
    with pytest.raises(SimulatedPowerLoss):
        first.apply(prepared, first.authorize(prepared))

    assert record.uninstaller_path is not None
    journal = _journal_files(record)[0]
    request = build_deferred_self_delete(
        record.uninstaller_path,
        wait_for_pid=2_147_483_647,
        expected_sha256=record.uninstaller_sha256,
        journal_path=journal,
        journal_sha256=hashlib.sha256(journal.read_bytes()).hexdigest(),
    )
    script = base64.b64decode(request.argv[-1]).decode("utf-16-le")
    manifest_literal = str(root / ".vipp-installer" / "ownership.json").replace(
        "'",
        "''",
    )
    assert script.count("Test-VippDeleteAuthorized") >= 4
    assert manifest_literal in script
    assert script.index("$mx.WaitOne()") < script.index(
        "Remove-Item -LiteralPath $p"
    )
    assert script.index("Test-VippDeleteAuthorized") < script.index(
        "Remove-Item -LiteralPath $p"
    )

    # Model a new frozen Setup winning the global gate. The old helper must
    # wait; after the same-version install commits at the identical cache path
    # and hash, its manifest predicate must preserve that current uninstaller.
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.argtypes = (
        ctypes.c_void_p,
        wintypes.BOOL,
        wintypes.LPCWSTR,
    )
    kernel32.CreateMutexW.restype = wintypes.HANDLE
    kernel32.ReleaseMutex.argtypes = (wintypes.HANDLE,)
    kernel32.ReleaseMutex.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    gate = kernel32.CreateMutexW(
        None,
        True,
        r"Local\VIPP.Setup.SingleInstance",
    )
    if not gate or ctypes.get_last_error() == 183:
        if gate:
            kernel32.CloseHandle(gate)
        pytest.skip("The VIPP Setup mutex is already in use.")
    helper = subprocess.Popen(  # noqa: S603 - reviewed encoded helper argv
        request.argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        active = managed_environments_root(root) / "0.13.0-reinstalled"
        marker_sha256 = _environment(active, str(uuid.uuid4()))
        current = replace(
            record,
            installation_id=str(uuid.uuid4()),
            environment_root=active,
            environment_marker_sha256=marker_sha256,
            shortcuts=(),
            retired_environments=(),
            updated_at=datetime.now(UTC).isoformat(),
        )
        write_ownership_record(root, current)
        current_plan = registry_plan_from_record(
            current,
            inspect_ownership(root).manifest_sha256,
        )
        register_apps_and_features(registry, current_plan)
    finally:
        kernel32.ReleaseMutex(gate)
        kernel32.CloseHandle(gate)
    stdout, stderr = helper.communicate(timeout=20)

    assert helper.returncode == 0, (stdout, stderr)
    assert record.uninstaller_path.is_file()
    assert journal.is_file()
    assert inspect_ownership(root).record == current
    assert registry.read_values(current_plan.key) == current_plan.value_map


def test_manifest_change_after_review_stops_before_removal(tmp_path: Path) -> None:
    record, root, _shortcut = _record(tmp_path)
    uninstaller = ManagedUninstaller()
    prepared = uninstaller.prepare(root, shortcut_roots=(tmp_path / "Desktop",))
    updated = replace(record, updated_at=datetime.now(UTC).isoformat())
    write_ownership_record(root, updated)

    with pytest.raises(UninstallPreparationError, match="changed after"):
        uninstaller.apply(prepared, uninstaller.authorize(prepared))

    assert record.environment_root.exists()


@pytest.mark.parametrize(
    "crash_phase",
    ("prepared", "payload_removed", "manifest_removed", "registry_removed"),
)
def test_power_loss_at_each_journal_boundary_resumes_safely(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    crash_phase: str,
) -> None:
    record, root, _shortcut, registry = _registered_case(tmp_path)
    plan = registry_plan_from_record(record, inspect_ownership(root).manifest_sha256)
    original_write = uninstall_module._write_uninstall_journal
    original_remove = uninstall_module._remove_environment
    observed_phases: list[str] = []
    detached_payload = tmp_path / "simulated-removed-payload"

    def quarantine_without_recursive_delete(item, managed_root):
        assert managed_root == root
        if not item.exists:
            return None
        detached_payload.parent.mkdir(parents=True, exist_ok=True)
        uninstall_module._replace_file_with_retry(item.path, detached_payload)
        return None

    def crash_at_phase(path, prepared, *, phase):
        observed_phases.append(phase)
        if phase == crash_phase:
            raise SimulatedPowerLoss(phase)
        return original_write(path, prepared, phase=phase)

    monkeypatch.setattr(
        uninstall_module,
        "_write_uninstall_journal",
        crash_at_phase,
    )
    # Recursive deletion has its own power-loss and partial-tree tests.  Keep
    # this test about journal boundaries: an atomic move makes the payload
    # absent without allowing a transient Windows file lock to pre-empt the
    # injected crash.
    monkeypatch.setattr(
        uninstall_module,
        "_remove_environment",
        quarantine_without_recursive_delete,
    )
    first = ManagedUninstaller(
        registry=registry,
        current_executable=record.uninstaller_path,
    )
    prepared = first.prepare(root, shortcut_roots=(tmp_path / "Desktop",))
    try:
        result = first.apply(prepared, first.authorize(prepared))
    except SimulatedPowerLoss as exc:
        assert crash_phase in str(exc)
    else:
        pytest.fail(
            f"journal phase {crash_phase!r} was not reached; "
            f"observed phases={observed_phases!r}; result={result!r}"
        )

    crash_index = uninstall_module._UNINSTALL_JOURNAL_PHASE_INDEX[crash_phase]
    assert observed_phases == list(
        uninstall_module._UNINSTALL_JOURNAL_PHASES[: crash_index + 1]
    )

    if crash_phase == "registry_removed":
        assert plan.key not in registry.keys
    else:
        assert plan.key in registry.keys
    assert record.uninstaller_path is not None
    assert record.uninstaller_path.exists()

    monkeypatch.setattr(
        uninstall_module,
        "_write_uninstall_journal",
        original_write,
    )
    monkeypatch.setattr(
        uninstall_module,
        "_remove_environment",
        original_remove,
    )
    resumed = ManagedUninstaller(
        registry=registry,
        current_executable=record.uninstaller_path,
    )
    retry = resumed.prepare(root, shortcut_roots=(tmp_path / "Desktop",))
    result = resumed.apply(
        retry,
        resumed.authorize(retry),
        current_executable=tmp_path / "downloaded-setup.exe",
    )

    assert result.completed
    assert plan.key not in registry.keys
    assert not record.environment_root.exists()
    assert not record.uninstaller_path.exists()
    assert not _journal_files(record)


def test_power_loss_after_environment_quarantine_resumes_partial_tree(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    record, root, _shortcut, registry = _registered_case(tmp_path)
    original_remove = uninstall_module._remove_quarantined_environment

    def partially_remove_then_crash(quarantine: Path, marker_sha256: str) -> None:
        launcher = quarantine / "Scripts" / "vipp-cpu.exe"
        launcher.unlink()
        raise SimulatedPowerLoss(marker_sha256)

    monkeypatch.setattr(
        uninstall_module,
        "_remove_quarantined_environment",
        partially_remove_then_crash,
    )
    first = ManagedUninstaller(
        registry=registry,
        current_executable=record.uninstaller_path,
    )
    prepared = first.prepare(root, shortcut_roots=(tmp_path / "Desktop",))
    quarantine = prepared.environment_items[0].quarantine_path
    assert quarantine is not None
    with pytest.raises(SimulatedPowerLoss):
        first.apply(prepared, first.authorize(prepared))
    assert not record.environment_root.exists()
    assert (quarantine / _MARKER_NAME).exists()

    monkeypatch.setattr(
        uninstall_module,
        "_remove_quarantined_environment",
        original_remove,
    )
    resumed = ManagedUninstaller(
        registry=registry,
        current_executable=record.uninstaller_path,
    )
    retry = resumed.prepare(root, shortcut_roots=(tmp_path / "Desktop",))
    result = resumed.apply(
        retry,
        resumed.authorize(retry),
        current_executable=tmp_path / "downloaded-setup.exe",
    )

    assert result.completed
    assert not quarantine.exists()


def test_foreign_file_after_marker_removal_is_never_inherited(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    record, root, _shortcut, registry = _registered_case(tmp_path)

    def crash_after_marker(quarantine: Path, marker_sha256: str) -> None:
        marker = quarantine / _MARKER_NAME
        uninstall_module._remove_direct_tree(
            quarantine,
            preserve_top_level=marker,
        )
        assert hashlib.sha256(marker.read_bytes()).hexdigest() == marker_sha256
        marker.unlink()
        raise SimulatedPowerLoss("marker removed")

    monkeypatch.setattr(
        uninstall_module,
        "_remove_quarantined_environment",
        crash_after_marker,
    )
    first = ManagedUninstaller(
        registry=registry,
        current_executable=record.uninstaller_path,
    )
    prepared = first.prepare(root, shortcut_roots=(tmp_path / "Desktop",))
    quarantine = prepared.environment_items[0].quarantine_path
    assert quarantine is not None
    with pytest.raises(SimulatedPowerLoss):
        first.apply(prepared, first.authorize(prepared))
    foreign = quarantine / "new-user-file.txt"
    foreign.write_text("preserve", encoding="utf-8")

    resumed = ManagedUninstaller(
        registry=registry,
        current_executable=record.uninstaller_path,
    )
    with pytest.raises(UninstallPreparationError, match="lost its ownership marker"):
        resumed.prepare(root, shortcut_roots=(tmp_path / "Desktop",))
    assert foreign.read_text(encoding="utf-8") == "preserve"


def test_registry_delete_requires_absence_readback_and_retains_retry(
    tmp_path: Path,
) -> None:
    class RefusingDeleteRegistry(MemoryRegistry):
        def delete_key(self, key: str) -> None:
            del key

    record, root, _shortcut = _record(
        tmp_path,
        uninstaller=True,
        registry=True,
    )
    registry = RefusingDeleteRegistry()
    plan = registry_plan_from_record(record, inspect_ownership(root).manifest_sha256)
    register_apps_and_features(registry, plan)
    uninstaller = ManagedUninstaller(registry=registry)
    prepared = uninstaller.prepare(root, shortcut_roots=(tmp_path / "Desktop",))

    result = uninstaller.apply(prepared, uninstaller.authorize(prepared))

    assert not result.completed
    assert result.retry_via_apps
    assert plan.key in registry.keys
    assert record.uninstaller_path is not None and record.uninstaller_path.exists()
    assert _journal_files(record)


def test_terminal_journal_can_be_reviewed_then_reaped_for_same_setup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    record, root, _shortcut, registry = _registered_case(tmp_path)
    unrelated = root / "research-data.txt"
    unrelated.write_text("keep", encoding="utf-8")
    original_remove = uninstall_module._remove_environment
    detached_payload = tmp_path / "simulated-removed-payload"

    def quarantine_without_recursive_delete(item, managed_root):
        assert managed_root == root
        if not item.exists:
            return None
        uninstall_module._replace_file_with_retry(item.path, detached_payload)
        return None

    def crash_before_cache(*_args, **_kwargs):
        raise SimulatedPowerLoss("before cached setup cleanup")

    monkeypatch.setattr(
        ManagedUninstaller,
        "_remove_or_defer_uninstaller",
        staticmethod(crash_before_cache),
    )
    monkeypatch.setattr(
        uninstall_module,
        "_remove_environment",
        quarantine_without_recursive_delete,
    )
    first = ManagedUninstaller(registry=registry)
    prepared = first.prepare(root, shortcut_roots=(tmp_path / "Desktop",))
    try:
        result = first.apply(prepared, first.authorize(prepared))
    except SimulatedPowerLoss:
        pass
    else:
        pytest.fail(
            "terminal cleanup boundary was not reached; "
            f"result={result!r}"
        )
    monkeypatch.setattr(
        uninstall_module,
        "_remove_environment",
        original_remove,
    )

    assert record.uninstaller_path is not None
    reviewed = reap_completed_uninstall_recovery(
        record.uninstaller_path,
        managed_root=root,
        expected_sha256=record.uninstaller_sha256,
        shortcut_roots=(tmp_path / "Desktop",),
        registry=registry,
        expected_track=record.track,
        keep_executable=True,
        perform_cleanup=False,
    )
    assert reviewed == _journal_files(record)
    removed = reap_completed_uninstall_recovery(
        record.uninstaller_path,
        managed_root=root,
        expected_sha256=record.uninstaller_sha256,
        shortcut_roots=(tmp_path / "Desktop",),
        registry=registry,
        expected_track=record.track,
        keep_executable=True,
    )

    assert removed == reviewed
    assert record.uninstaller_path.exists()
    assert not _journal_files(record)
    assert unrelated.read_text(encoding="utf-8") == "keep"


def test_tampered_and_stale_generation_journals_are_refused(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    record, root, _shortcut, registry = _registered_case(tmp_path)
    original_write = uninstall_module._write_uninstall_journal

    def crash_after_payload(path, prepared, *, phase):
        if phase == "payload_removed":
            raise SimulatedPowerLoss(phase)
        return original_write(path, prepared, phase=phase)

    monkeypatch.setattr(
        uninstall_module,
        "_write_uninstall_journal",
        crash_after_payload,
    )
    first = ManagedUninstaller(registry=registry)
    prepared = first.prepare(root, shortcut_roots=(tmp_path / "Desktop",))
    with pytest.raises(SimulatedPowerLoss):
        first.apply(prepared, first.authorize(prepared))
    journal = _journal_files(record)[0]
    document = json.loads(journal.read_text(encoding="utf-8"))
    document["track"] = "cuda13"
    journal.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(UninstallPreparationError, match="CPU/GPU track"):
        ManagedUninstaller(
            registry=registry,
            current_executable=record.uninstaller_path,
        ).prepare(root, shortcut_roots=(tmp_path / "Desktop",))

    document["track"] = "cpu"
    journal.write_text(json.dumps(document), encoding="utf-8")
    newer = replace(record, installation_id=str(uuid.uuid4()))
    write_ownership_record(root, newer)
    with pytest.raises(UninstallPreparationError, match="not terminal"):
        ManagedUninstaller(registry=registry).prepare(
            root,
            shortcut_roots=(tmp_path / "Desktop",),
        )


def test_cpu_recovery_never_touches_gpu_installation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cpu, cpu_root, _shortcut, registry = _registered_case(tmp_path / "cpu")
    gpu, gpu_root, _shortcut, gpu_registry = _registered_case(
        tmp_path / "gpu",
        track=ComputeTrack.CUDA13,
    )
    registry.keys.update(gpu_registry.keys)

    def crash_before_cache(*_args, **_kwargs):
        raise SimulatedPowerLoss("cpu terminal")

    monkeypatch.setattr(
        ManagedUninstaller,
        "_remove_or_defer_uninstaller",
        staticmethod(crash_before_cache),
    )
    first = ManagedUninstaller(registry=registry)
    prepared = first.prepare(cpu_root, shortcut_roots=(tmp_path / "cpu" / "Desktop",))
    with pytest.raises(SimulatedPowerLoss):
        first.apply(prepared, first.authorize(prepared))

    assert cpu.uninstaller_path is not None
    reap_completed_uninstall_recovery(
        cpu.uninstaller_path,
        managed_root=cpu_root,
        expected_sha256=cpu.uninstaller_sha256,
        shortcut_roots=(tmp_path / "cpu" / "Desktop",),
        registry=registry,
        expected_track=ComputeTrack.CPU,
        keep_executable=True,
    )

    gpu_plan = registry_plan_from_record(
        gpu,
        inspect_ownership(gpu_root).manifest_sha256,
    )
    assert gpu_plan.key in registry.keys
    assert gpu.environment_root.exists()
    assert gpu.uninstaller_path is not None and gpu.uninstaller_path.exists()
