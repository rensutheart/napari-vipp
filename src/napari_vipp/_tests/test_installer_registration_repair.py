from __future__ import annotations

import hashlib
import json
import sys
import threading
import types
import uuid
from dataclasses import replace

import pytest

from napari_vipp.installer import registration_repair as repair
from napari_vipp.installer.models import ComputeTrack
from napari_vipp.installer.ownership import (
    OwnedPackage,
    OwnershipRecord,
    inspect_ownership,
    write_ownership_record,
)
from napari_vipp.installer.uninstall import (
    registry_key_for_track,
    registry_plan_from_record,
)


class MemoryRegistry:
    def __init__(self, key, snapshot):
        self.key, self.snapshot = key, snapshot
        self.writes = []

    def read_snapshot(self, key):
        assert key == self.key
        return self.snapshot

    def write_repair(self, key, expected, values, *, cancellation=None):
        repair._checkpoint(cancellation)
        assert self.snapshot == expected
        self.writes.append((key, dict(values)))
        self.snapshot = repair.RegistrationSnapshot(
            repair._typed(values), last_write=expected.last_write + 1
        )

    def read_values(self, _key):
        raise AssertionError("Lossy registry backend must not be used")

    def write_values(self, *_args):
        raise AssertionError("Whole-key registry replacement must not be used")


@pytest.fixture
def owned(tmp_path):
    local = tmp_path / "Local"
    root = local / "VIPP" / "environments" / "cuda13"
    environment = root / ".vipp-installer" / "environments" / "current"
    environment.mkdir(parents=True)
    marker = environment / ".vipp-install-candidate.json"
    marker.write_text("owned synthetic marker")
    current_hash = hashlib.sha256(b"current cached setup").hexdigest()
    old_hash = hashlib.sha256(b"old cached setup").hexdigest()
    current_path = repair._cache_path(
        local, root, ComputeTrack.CUDA13, "0.16.0a6", current_hash
    )
    old_path = repair._cache_path(local, root, ComputeTrack.CUDA13, "0.15a5", old_hash)
    for path, content in (
        (current_path, b"current cached setup"),
        (old_path, b"old cached setup"),
    ):
        path.parent.mkdir(parents=True)
        path.write_bytes(content)
    record = OwnershipRecord(
        str(uuid.uuid4()),
        root,
        environment,
        "napari-vipp",
        "0.16.0a6",
        ComputeTrack.CUDA13,
        tmp_path / "python.exe",
        "a" * 64,
        (OwnedPackage("napari-vipp", "0.16.0a6", "b" * 64),),
        "before",
        "before",
        environment_marker_sha256=hashlib.sha256(marker.read_bytes()).hexdigest(),
        uninstaller_path=current_path,
        uninstaller_sha256=current_hash,
        registry_key=registry_key_for_track(ComputeTrack.CUDA13),
    )
    write_ownership_record(root, record)
    previous = registry_plan_from_record(
        replace(
            record,
            version="0.15a5",
            uninstaller_path=old_path,
            uninstaller_sha256=old_hash,
        ),
        "c" * 64,
    )
    backend = MemoryRegistry(
        previous.key,
        repair.RegistrationSnapshot(repair._typed(previous.value_map), last_write=1),
    )
    return record, local, previous, backend


def _review(owned):
    record, local, _previous, backend = owned
    return repair.prepare_registration_repair(record.managed_root, local, backend)


def test_complete_stale_review_is_readonly_and_explicit_repair_only(owned):
    record, local, previous, backend = owned
    before = backend.snapshot
    prepared = _review(owned)
    assert backend.snapshot == before and backend.writes == []
    assert not prepared.backup_path.exists()
    assert (
        prepared.registered_version == "0.15a5"
        and prepared.installed_version == record.version
    )
    assert {item[0] for item in prepared.changes} == {
        "DisplayVersion",
        "VippManifestSha256",
        "VippUninstallerSha256",
        "UninstallString",
    }
    with pytest.raises(repair.RegistrationRepairError, match="explicitly confirmed"):
        repair.authorize_registration_repair(prepared, confirmed=False)
    authorization = repair.authorize_registration_repair(prepared, confirmed=True)
    result = repair.apply_registration_repair(prepared, authorization, backend)
    assert result.succeeded
    backup = json.loads(prepared.backup_path.read_text())
    assert backup["before"] == before.as_dict()
    assert backend.snapshot.values == repair._typed(prepared.current_plan.value_map)
    assert inspect_ownership(record.managed_root).record == record
    assert previous.uninstaller_path.read_bytes() == b"old cached setup"
    assert local.is_dir()
    with pytest.raises(repair.RegistrationRepairError, match="already used"):
        repair.apply_registration_repair(prepared, authorization, backend)


@pytest.mark.parametrize(
    "change",
    [
        "root",
        "id",
        "publisher",
        "missing",
        "extra",
        "binary",
        "kind",
        "subkey",
        "newer",
        "command",
    ],
)
def test_foreign_partial_additional_or_wrongly_typed_entries_refused(owned, change):
    record, _local, previous, backend = owned
    values = previous.value_map
    if change == "root":
        values["VippManagedRoot"] = str(record.managed_root.parent / "foreign")
    elif change == "id":
        values["VippInstallationId"] = str(uuid.uuid4())
    elif change == "publisher":
        values["Publisher"] = "foreign"
    elif change == "missing":
        values.pop("NoRepair")
    elif change == "extra":
        values["ForeignValue"] = "preserve"
    elif change == "newer":
        values["DisplayVersion"] = "99.0"
    elif change == "command":
        values["UninstallString"] = '"C:/foreign.exe" --uninstall'
    snapshot = repair.RegistrationSnapshot(repair._typed(values), last_write=1)
    if change == "binary":
        snapshot = replace(
            snapshot,
            values=(
                *snapshot.values,
                repair.RegistryValue("HiddenBinary", 3, b"preserve"),
            ),
        )
    elif change == "kind":
        snapshot = replace(
            snapshot,
            values=tuple(
                replace(item, kind=2) if item.name == "DisplayVersion" else item
                for item in snapshot.values
            ),
        )
    elif change == "subkey":
        snapshot = replace(snapshot, subkey_count=1)
    backend.snapshot = snapshot
    with pytest.raises((repair.RegistrationRepairError, ValueError)):
        _review(owned)
    assert backend.snapshot == snapshot and backend.writes == []


@pytest.mark.parametrize("which", ["old", "current"])
def test_cached_executable_must_match_exact_reviewed_bytes(owned, which):
    record, _local, previous, backend = owned
    path = previous.uninstaller_path if which == "old" else record.uninstaller_path
    path.write_bytes(b"foreign changed bytes")
    with pytest.raises(Exception, match="changed|match|hash"):
        _review(owned)
    assert backend.writes == []


def test_absent_retired_uninstaller_has_explicit_readonly_review_and_backup(owned):
    record, _local, previous, backend = owned
    previous.uninstaller_path.unlink()
    prepared = _review(owned)
    assert prepared.old_uninstaller_state == "absent"
    assert "unavailable" in prepared.summary and "will not be run" in prepared.summary
    assert not prepared.backup_path.exists() and backend.writes == []
    result = repair.apply_registration_repair(
        prepared,
        repair.authorize_registration_repair(prepared, confirmed=True),
        backend,
    )
    assert result.succeeded
    saved = json.loads(prepared.backup_path.read_text())
    assert saved["retired_uninstaller"]["state"] == "absent"
    assert saved["retired_uninstaller"]["historical_exe_verified"] is False
    assert saved["retired_uninstaller"]["observed_sha256"] is None
    assert not previous.uninstaller_path.exists()
    assert inspect_ownership(record.managed_root).record == record


def test_retired_exe_appearing_after_absence_review_requires_new_confirmation(owned):
    _record, _local, previous, backend = owned
    previous.uninstaller_path.unlink()
    prepared = _review(owned)
    previous.uninstaller_path.write_bytes(b"old cached setup")
    with pytest.raises(repair.RegistrationRepairError, match="changed"):
        repair.apply_registration_repair(
            prepared,
            repair.authorize_registration_repair(prepared, confirmed=True),
            backend,
        )
    assert backend.writes == [] and not prepared.backup_path.exists()


def test_retired_exe_dangling_or_redirected_path_is_not_absence(owned, monkeypatch):
    _record, _local, previous, backend = owned
    original = type(previous.uninstaller_path).lstat

    def redirected(path, *args, **kwargs):
        result = original(path, *args, **kwargs)
        if path == previous.uninstaller_path:
            return types.SimpleNamespace(st_mode=0o120777, st_file_attributes=0x400)
        return result

    monkeypatch.setattr(type(previous.uninstaller_path), "lstat", redirected)
    with pytest.raises(repair.RegistrationRepairError, match="Redirected"):
        _review(owned)
    assert backend.writes == []


def test_retired_exe_state_is_bound_and_absent_case_authorization_is_single_use(owned):
    _record, _local, previous, backend = owned
    previous.uninstaller_path.unlink()
    prepared = _review(owned)
    forged = replace(prepared, old_uninstaller_state="verified-present")
    with pytest.raises(repair.RegistrationRepairError, match="changed"):
        repair.apply_registration_repair(
            forged,
            repair.authorize_registration_repair(forged, confirmed=True),
            backend,
        )
    authorization = repair.authorize_registration_repair(prepared, confirmed=True)
    result = repair.apply_registration_repair(prepared, authorization, backend)
    assert result.succeeded
    with pytest.raises(repair.RegistrationRepairError, match="already used"):
        repair.apply_registration_repair(prepared, authorization, backend)


def test_changed_typed_snapshot_and_tampered_payload_refused(owned):
    _record, _local, _previous, backend = owned
    prepared = _review(owned)
    authorization = repair.authorize_registration_repair(prepared, confirmed=True)
    backend.snapshot = replace(backend.snapshot, last_write=2)
    with pytest.raises(repair.RegistrationRepairError, match="changed"):
        repair.apply_registration_repair(prepared, authorization, backend)
    assert not prepared.backup_path.exists() and backend.writes == []
    backend.snapshot = prepared.before
    tampered = replace(prepared, current_plan=prepared.previous_plan)
    authorization = repair.authorize_registration_repair(tampered, confirmed=True)
    with pytest.raises(repair.RegistrationRepairError, match="changed"):
        repair.apply_registration_repair(tampered, authorization, backend)
    assert backend.writes == []


@pytest.mark.parametrize("field", ["backup_path", "changes", "summary"])
def test_all_displayed_or_material_review_fields_are_bound(owned, field, tmp_path):
    _record, _local, _previous, backend = owned
    prepared = _review(owned)
    changes = {
        "backup_path": tmp_path / "unauthorized.json",
        "changes": (),
        "summary": "Misleading review",
    }
    tampered = replace(prepared, **{field: changes[field]})
    authorization = repair.authorize_registration_repair(tampered, confirmed=True)
    with pytest.raises(repair.RegistrationRepairError):
        repair.apply_registration_repair(tampered, authorization, backend)
    assert not prepared.backup_path.exists()
    assert not (tmp_path / "unauthorized.json").exists()
    assert backend.writes == []


@pytest.mark.parametrize("confirmation", [False, None, 1, "yes", "true"])
def test_confirmation_must_be_literal_true(owned, confirmation):
    prepared = _review(owned)
    with pytest.raises(repair.RegistrationRepairError, match="explicitly confirmed"):
        repair.authorize_registration_repair(prepared, confirmed=confirmation)


def test_local_app_data_cannot_be_substituted_to_adopt_an_owned_foreign_root(
    owned, tmp_path
):
    record, _local, _previous, backend = owned
    different_local = tmp_path / "OtherLocal"
    different_local.mkdir()
    with pytest.raises(repair.RegistrationRepairError, match="canonical"):
        repair.prepare_registration_repair(
            record.managed_root, different_local, backend
        )
    assert backend.writes == []


def test_repair_refuses_an_already_held_shared_installer_target_lock(owned):
    from napari_vipp.installer.engine import ConcurrentInstallationError, _TargetLock

    record, local, _previous, backend = owned
    prepared = _review(owned)
    authorization = repair.authorize_registration_repair(prepared, confirmed=True)
    state = local / "VIPP" / "installer"
    with _TargetLock(state, record.managed_root, "other-owned-operation"):
        with pytest.raises(ConcurrentInstallationError, match="already using"):
            repair.apply_registration_repair(prepared, authorization, backend)
        assert not prepared.backup_path.exists()
        assert backend.writes == []


def test_noncooperative_ownership_change_during_write_cannot_report_success(
    owned, monkeypatch
):
    record, _local, _previous, backend = owned
    prepared = _review(owned)
    original_write = backend.write_repair

    def racing_write(*args, **kwargs):
        original_write(*args, **kwargs)
        write_ownership_record(
            record.managed_root, replace(record, updated_at="noncooperative-change")
        )

    monkeypatch.setattr(backend, "write_repair", racing_write)
    result = repair.apply_registration_repair(
        prepared,
        repair.authorize_registration_repair(prepared, confirmed=True),
        backend,
    )
    assert not result.succeeded and "ownership changed" in result.message
    assert prepared.backup_path.exists()
    assert (
        inspect_ownership(record.managed_root).record.updated_at
        == "noncooperative-change"
    )


def test_backup_failure_and_cancellation_prevent_primary_mutation(owned, monkeypatch):
    _record, _local, _previous, backend = owned
    prepared = _review(owned)
    cancellation = threading.Event()
    cancellation.set()
    authorization = repair.authorize_registration_repair(prepared, confirmed=True)
    with pytest.raises(repair.RegistrationRepairCancelled):
        repair.apply_registration_repair(
            prepared, authorization, backend, cancellation=cancellation
        )
    assert not prepared.backup_path.exists() and backend.writes == []
    original = type(prepared.backup_path).open

    def failed(path, *args, **kwargs):
        if path == prepared.backup_path:
            raise PermissionError("fixture backup failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(type(prepared.backup_path), "open", failed)
    authorization = repair.authorize_registration_repair(prepared, confirmed=True)
    with pytest.raises(PermissionError, match="backup failure"):
        repair.apply_registration_repair(prepared, authorization, backend)
    assert backend.writes == []


def _winreg_fixture(monkeypatch, snapshot, *, fail_call=None, foreign_after=None):
    state = {
        "rows": {item.name: item for item in snapshot.values},
        "stamp": 1,
        "calls": 0,
    }

    class Handle:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    def enumerate_value(_handle, index):
        rows = list(state["rows"].values())
        if index >= len(rows):
            error = OSError("no more items")
            error.winerror = 259
            raise error
        item = rows[index]
        return item.name, item.value, item.kind

    def write(_handle, name, _reserved, kind, value):
        state["calls"] += 1
        state["rows"][name] = repair.RegistryValue(name, kind, value)
        state["stamp"] += 1
        if state["calls"] == foreign_after:
            state["rows"]["ForeignBinary"] = repair.RegistryValue(
                "ForeignBinary", 3, b"preserve"
            )
            state["stamp"] += 1
        if state["calls"] == fail_call:
            raise OSError("fixture partial native write failure")

    fake = types.SimpleNamespace(
        HKEY_CURRENT_USER=object(),
        KEY_QUERY_VALUE=1,
        KEY_SET_VALUE=2,
        OpenKey=lambda *_args, **_kwargs: Handle(),
        QueryInfoKey=lambda _handle: (0, len(state["rows"]), state["stamp"]),
        EnumValue=enumerate_value,
        SetValueEx=write,
    )
    monkeypatch.setitem(sys.modules, "winreg", fake)
    return state


def test_native_typed_patch_rolls_back_only_its_exact_partial_transition(
    owned, monkeypatch
):
    record, local, previous, backend = owned
    state = _winreg_fixture(monkeypatch, backend.snapshot, fail_call=2)
    native = repair.WindowsRegistrationRepairBackend()
    prepared = repair.prepare_registration_repair(record.managed_root, local, native)
    result = repair.apply_registration_repair(
        prepared, repair.authorize_registration_repair(prepared, confirmed=True), native
    )
    assert not result.succeeded and prepared.backup_path.exists()
    assert {
        name: item.value for name, item in state["rows"].items()
    } == previous.value_map


def test_native_typed_patch_preserves_raced_foreign_values_without_rollback(
    owned, monkeypatch
):
    record, local, _previous, backend = owned
    state = _winreg_fixture(monkeypatch, backend.snapshot, foreign_after=1)
    native = repair.WindowsRegistrationRepairBackend()
    prepared = repair.prepare_registration_repair(record.managed_root, local, native)
    result = repair.apply_registration_repair(
        prepared, repair.authorize_registration_repair(prepared, confirmed=True), native
    )
    assert not result.succeeded and prepared.backup_path.exists()
    assert state["rows"]["ForeignBinary"].value == b"preserve"
    assert state["calls"] == 1
