from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from napari_vipp.installer import engine as engine_module
from napari_vipp.installer import uninstall as uninstall_module
from napari_vipp.installer.engine import (
    AuthorizationError,
    CancellationToken,
    CommandResult,
    InstallStatus,
    ManagedInstallerEngine,
    ManagedTargetKind,
    PreparationError,
    ResolutionError,
    StalePreparedTransaction,
    inspect_managed_target,
)
from napari_vipp.installer.models import (
    ComputeTrack,
    DiscoverySnapshot,
    FilesystemSnapshot,
    HostSnapshot,
    InstallMode,
    InstallRequest,
    PythonSnapshot,
    ReleaseSpec,
    ShortcutScope,
    installation_request_fingerprint,
)
from napari_vipp.installer.ownership import (
    OwnedEnvironment,
    OwnedPackage,
    OwnedShortcut,
    OwnershipRecord,
    inspect_ownership,
    managed_environments_root,
    write_ownership_record,
)
from napari_vipp.installer.planner import create_install_plan
from napari_vipp.installer.uninstall import (
    CPU_REGISTRY_KEY,
    CUDA13_REGISTRY_KEY,
    ManagedUninstaller,
    UninstallStatus,
)

_NOW = datetime(2026, 8, 9, 12, 0, tzinfo=UTC)
_RUN_ID = "12345678-1234-4234-9234-123456789abc"
_VIPP_HASH = "a" * 64
_DEPENDENCY_HASH = "b" * 64


@pytest.fixture(autouse=True)
def _isolate_installer_temporary_directory(monkeypatch, tmp_path):
    """Keep Windows-installer tests off redirected host temp directories."""

    monkeypatch.setattr(
        engine_module.tempfile,
        "gettempdir",
        lambda: str(tmp_path.resolve()),
    )


def _release(
    version: str = "0.13.0a5",
    *,
    wheel_path: Path | None = None,
    wheel_sha256: str = "",
) -> ReleaseSpec:
    return ReleaseSpec(
        distribution="napari-vipp",
        version=version,
        wheel_path=wheel_path,
        wheel_sha256=wheel_sha256,
    )


def _record(
    target: Path,
    *,
    version: str = "0.13.0a4",
    healthy: bool = True,
    shortcut: Path | None = None,
) -> OwnershipRecord:
    environment = managed_environments_root(target) / f"{version}-old"
    environment.mkdir(parents=True)
    marker = environment / ".vipp-install-candidate.json"
    marker.write_text(
        json.dumps(
            {
                "schema": "napari-vipp-install-candidate",
                "schema_version": 1,
                "run_id": str(uuid.uuid4()),
                "resolution_id": "c" * 64,
            }
        ),
        encoding="utf-8",
    )
    if healthy:
        scripts = environment / "Scripts"
        scripts.mkdir()
        (environment / "pyvenv.cfg").write_text("home = test\n", encoding="utf-8")
        (scripts / "python.exe").touch()
        (scripts / "vipp-cpu.exe").touch()
    record = OwnershipRecord(
        installation_id=str(uuid.uuid4()),
        managed_root=target,
        environment_root=environment,
        distribution="napari-vipp",
        version=version,
        track=ComputeTrack.CPU,
        base_python=target.parent / "base-python.exe",
        resolved_plan_id="c" * 64,
        packages=(OwnedPackage("napari-vipp", version, _VIPP_HASH),),
        environment_marker_sha256=hashlib.sha256(marker.read_bytes()).hexdigest(),
        shortcuts=(
            (
                OwnedShortcut(
                    shortcut,
                    hashlib.sha256(shortcut.read_bytes()).hexdigest(),
                ),
            )
            if shortcut is not None
            else ()
        ),
        created_at=_NOW.isoformat(),
        updated_at=_NOW.isoformat(),
    )
    write_ownership_record(target, record)
    return record


def _plan(
    target: Path,
    release: ReleaseSpec,
    *,
    shortcut_directory: Path | None = None,
):
    scope = (
        ShortcutScope.DESKTOP if shortcut_directory is not None else ShortcutScope.NONE
    )
    request = InstallRequest(
        mode=InstallMode.MANAGED,
        track=ComputeTrack.CPU,
        python=target.parent / "base-python.exe",
        install_root=target,
        shortcut_scope=scope,
        shortcut_directory=shortcut_directory,
    )
    ownership = inspect_ownership(target)
    exists = target.exists()
    snapshot = DiscoverySnapshot(
        request_fingerprint=installation_request_fingerprint(request),
        host=HostSnapshot("win32", "Windows", "AMD64"),
        python=PythonSnapshot(
            requested_executable=request.python,
            executable=request.python,
            base_executable=request.python,
            probe_succeeded=True,
            implementation="cpython",
            version=(3, 12, 10),
            pointer_bits=64,
        ),
        filesystem=FilesystemSnapshot(
            target=target,
            target_exists=exists,
            target_kind="directory" if exists else "missing",
            target_empty=(next(target.iterdir(), None) is None if exists else None),
            target_reparse_point=False,
            target_protected=False,
            target_protection_reason="",
            nearest_existing_ancestor=target if exists else target.parent,
            nearest_existing_ancestor_is_directory=True,
            free_bytes=20 * 1024**3,
            disk_probe_error="",
            desktop_directory=shortcut_directory,
            start_menu_directory=None,
            managed_ownership=(
                ownership.record.to_snapshot(ownership.manifest_sha256)
                if ownership.record is not None
                else None
            ),
            managed_ownership_error=ownership.error,
            ownership_manifest_exists=ownership.record is not None,
        ),
    )
    return create_install_plan(request, discovery=snapshot, release=release)


def _report(
    *,
    vipp_hash: str = _VIPP_HASH,
    vipp_url: str = "https://packages.example/vipp.whl",
) -> str:
    return json.dumps(
        {
            "version": "1",
            "pip_version": "26.1",
            "install": [
                {
                    "download_info": {
                        "url": vipp_url,
                        "archive_info": {"hashes": {"sha256": vipp_hash}},
                    },
                    "requested": True,
                    "metadata": {"name": "napari-vipp", "version": "0.13.0a5"},
                },
                {
                    "download_info": {
                        "url": "https://packages.example/dependency.whl",
                        "archive_info": {"hashes": {"sha256": _DEPENDENCY_HASH}},
                    },
                    "requested": False,
                    "metadata": {"name": "example-dependency", "version": "2.0"},
                },
            ],
        }
    )


class _FakeRunner:
    def __init__(
        self,
        report: str | None = None,
        *,
        cancel_on_install: CancellationToken | None = None,
        fail_acceptance: bool = False,
    ) -> None:
        self.report = report or _report()
        self.cancel_on_install = cancel_on_install
        self.fail_acceptance = fail_acceptance
        self.calls: list[tuple[str, ...]] = []
        self.environments: list[dict[str, str]] = []

    def run(self, argv, *, cancellation=None, env=None, cwd=None):
        call = tuple(str(value) for value in argv)
        self.calls.append(call)
        self.environments.append(dict(env or {}))
        if "--dry-run" in call:
            return CommandResult(0, self.report, "")
        if call[1:4] == ("-m", "venv", call[-1]):
            environment = Path(call[-1])
            scripts = environment / "Scripts"
            scripts.mkdir(parents=True, exist_ok=True)
            (environment / "pyvenv.cfg").write_text("home = test\n", encoding="utf-8")
            (scripts / "python.exe").touch()
            return CommandResult(0)
        if "--requirement" in call:
            if self.cancel_on_install is not None:
                self.cancel_on_install.cancel()
            (Path(call[0]).parent / "vipp-cpu.exe").touch()
            (Path(call[0]).parent / "vipp-app.exe").touch()
            (Path(call[0]).parent / "vipp-prefer-gpu.exe").touch()
            return CommandResult(0)
        if call[0].casefold() == "powershell.exe":
            destination = Path(call[call.index("-Destination") + 1])
            target = call[call.index("-Target") + 1]
            destination.write_bytes(f"shortcut:{target}".encode())
            return CommandResult(0)
        if self.fail_acceptance and call[-2:] == ("pip", "check"):
            return CommandResult(1, "", "broken dependency")
        return CommandResult(0)


def _engine(tmp_path: Path, runner: _FakeRunner) -> ManagedInstallerEngine:
    documents = tmp_path / "Documents"
    documents.mkdir(exist_ok=True)
    return ManagedInstallerEngine(
        runner=runner,
        state_root=tmp_path / "installer-state",
        now=lambda: _NOW,
        identifier=lambda: _RUN_ID,
        approved_artifact_hosts=(
            "pypi.org",
            "files.pythonhosted.org",
            "packages.example",
        ),
        known_folder_probe=lambda name: {
            "documents": documents,
            "desktop": tmp_path / "Desktop",
            "programs": tmp_path / "Programs",
        }.get(name),
    )


class _MemoryRegistry:
    def __init__(self, *, fail_writes: bool = False):
        self.values: dict[str, dict[str, str | int]] = {}
        self.fail_writes = fail_writes
        self.partial_write_count: int | None = None

    def read_values(self, key):
        values = self.values.get(key.casefold())
        return dict(values) if values is not None else None

    def write_values(self, key, values):
        if self.fail_writes:
            if self.partial_write_count is not None:
                current = self.values.setdefault(key.casefold(), {})
                for name, value in tuple(values.items())[: self.partial_write_count]:
                    current[name] = value
            raise OSError("simulated registry write failure")
        self.values[key.casefold()] = dict(values)

    def delete_key(self, key):
        self.values.pop(key.casefold(), None)


def _engine_with_setup(
    tmp_path: Path,
    runner: _FakeRunner,
    setup_source: Path,
    registry: _MemoryRegistry,
    *,
    run_id: str = _RUN_ID,
) -> ManagedInstallerEngine:
    documents = tmp_path / "Documents"
    documents.mkdir(exist_ok=True)
    return ManagedInstallerEngine(
        runner=runner,
        state_root=tmp_path / "installer-state",
        now=lambda: _NOW,
        identifier=lambda: run_id,
        approved_artifact_hosts=(
            "pypi.org",
            "files.pythonhosted.org",
            "packages.example",
        ),
        setup_source=setup_source,
        persistent_setup_path=(tmp_path / "installer-state" / "VIPP-Setup.exe"),
        registry_backend=registry,
        known_folder_probe=lambda name: {
            "documents": documents,
            "desktop": tmp_path / "Desktop",
            "programs": tmp_path / "Programs",
        }.get(name),
    )


def test_classifies_new_update_current_repair_newer_and_foreign(tmp_path):
    release = _release()
    new = inspect_managed_target(
        tmp_path / "new",
        release=release,
        track=ComputeTrack.CPU,
    )
    update_root = tmp_path / "update"
    _record(update_root, version="0.13.0a4")
    update = inspect_managed_target(
        update_root,
        release=release,
        track=ComputeTrack.CPU,
    )
    current_root = tmp_path / "current"
    _record(current_root, version="0.13.0a5")
    current = inspect_managed_target(
        current_root,
        release=release,
        track=ComputeTrack.CPU,
    )
    repair_root = tmp_path / "repair"
    _record(repair_root, version="0.13.0a5", healthy=False)
    repair = inspect_managed_target(
        repair_root,
        release=release,
        track=ComputeTrack.CPU,
    )
    newer_root = tmp_path / "newer"
    _record(newer_root, version="0.13.0a6")
    newer = inspect_managed_target(
        newer_root,
        release=release,
        track=ComputeTrack.CPU,
    )
    foreign_root = tmp_path / "foreign"
    foreign_root.mkdir()
    (foreign_root / "research-data.txt").write_text("keep", encoding="utf-8")
    foreign = inspect_managed_target(
        foreign_root,
        release=release,
        track=ComputeTrack.CPU,
    )

    assert [
        new.kind,
        update.kind,
        current.kind,
        repair.kind,
        newer.kind,
        foreign.kind,
    ] == [
        ManagedTargetKind.NEW,
        ManagedTargetKind.UPDATE,
        ManagedTargetKind.CURRENT,
        ManagedTargetKind.REPAIR,
        ManagedTargetKind.NEWER,
        ManagedTargetKind.FOREIGN,
    ]
    assert new.action_label == "Install VIPP"
    assert current.action_label == "Open VIPP"
    assert newer.can_apply is False
    assert foreign.can_apply is False


def test_prepare_is_non_mutating_and_binds_exact_hashed_resolution(tmp_path):
    target = tmp_path / "VIPP Managed"
    state = tmp_path / "installer-state"
    runner = _FakeRunner()
    engine = _engine(tmp_path, runner)

    prepared = engine.prepare(_plan(target, _release()))

    assert prepared.applicable
    assert prepared.resolution_complete
    assert len(prepared.resolution_id) == 64
    assert [package.name for package in prepared.packages] == [
        "example-dependency",
        "napari-vipp",
    ]
    assert not target.exists()
    assert not state.exists()
    resolution = runner.calls[0]
    assert "--dry-run" in resolution
    assert "--ignore-installed" in resolution
    assert "--report" in resolution
    assert "--no-cache-dir" in resolution
    assert "--only-binary=:all:" in resolution


def test_apply_requires_one_use_explicit_authorization(tmp_path):
    target = tmp_path / "managed"
    runner = _FakeRunner()
    engine = _engine(tmp_path, runner)
    prepared = engine.prepare(_plan(target, _release()))

    with pytest.raises(AuthorizationError, match="explicitly confirmed"):
        engine.authorize(prepared, confirmed=False)

    authorization = engine.authorize(prepared, confirmed=True)
    result = engine.apply(prepared, authorization)
    assert result.status is InstallStatus.SUCCEEDED
    assert result.launcher_path is not None and result.launcher_path.is_file()
    with pytest.raises(AuthorizationError, match="already been used"):
        engine.apply(prepared, authorization)


def test_apply_uses_permanent_versioned_environment_and_exact_lock(tmp_path):
    target = tmp_path / "managed"
    runner = _FakeRunner()
    engine = _engine(tmp_path, runner)
    plan = _plan(target, _release())
    prepared = engine.prepare(plan)

    result = engine.apply(
        prepared,
        engine.authorize(prepared, confirmed=True),
    )

    assert result.succeeded
    assert result.environment_root is not None
    assert result.environment_root.parent == managed_environments_root(target)
    assert result.environment_root != target
    venv_call = next(call for call in runner.calls if call[1:3] == ("-m", "venv"))
    assert Path(venv_call[-1]) == result.environment_root
    install_call = next(call for call in runner.calls if "--requirement" in call)
    assert Path(install_call[0]).is_relative_to(result.environment_root)
    assert "--require-hashes" in install_call
    assert "--no-deps" in install_call
    assert "--only-binary=:all:" in install_call
    lock_path = Path(install_call[install_call.index("--requirement") + 1])
    lock = lock_path.read_text(encoding="utf-8")
    assert f"sha256:{_VIPP_HASH}" in lock
    assert f"sha256:{_DEPENDENCY_HASH}" in lock
    assert plan.release.requirement(plan.request) not in install_call
    ownership = inspect_ownership(target)
    assert ownership.record is not None
    assert ownership.record.environment_root == result.environment_root
    assert ownership.record.resolved_plan_id == prepared.resolution_id
    assert result.log_path.is_file()


def test_failed_update_preserves_active_environment_and_manifest(tmp_path):
    target = tmp_path / "managed"
    old = _record(target, version="0.13.0a4")
    before = (target / ".vipp-installer" / "ownership.json").read_bytes()
    runner = _FakeRunner(fail_acceptance=True)
    engine = _engine(tmp_path, runner)
    prepared = engine.prepare(_plan(target, _release()))
    assert prepared.target_inspection.kind is ManagedTargetKind.UPDATE

    result = engine.apply(
        prepared,
        engine.authorize(prepared, confirmed=True),
    )

    assert result.status is InstallStatus.FAILED
    assert old.environment_root.is_dir()
    assert (target / ".vipp-installer" / "ownership.json").read_bytes() == before
    assert inspect_ownership(target).record == old
    venv_call = next(call for call in runner.calls if call[1:3] == ("-m", "venv"))
    assert Path(venv_call[-1]) != old.environment_root
    assert not Path(venv_call[-1]).exists()
    assert old.environment_root in result.rollback.preserved_paths


def test_cancelled_apply_removes_only_marked_candidate(tmp_path):
    target = tmp_path / "managed"
    old = _record(target, version="0.13.0a4")
    token = CancellationToken()
    runner = _FakeRunner(cancel_on_install=token)
    engine = _engine(tmp_path, runner)
    prepared = engine.prepare(_plan(target, _release()))

    result = engine.apply(
        prepared,
        engine.authorize(prepared, confirmed=True),
        cancellation=token,
    )

    assert result.status is InstallStatus.CANCELLED
    assert result.rollback.completed
    assert old.environment_root.is_dir()
    assert inspect_ownership(target).record == old
    assert all(not path.exists() for path in result.rollback.removed_paths)


def test_foreign_and_newer_targets_are_not_resolved_or_authorized(tmp_path):
    runner = _FakeRunner()
    engine = _engine(tmp_path, runner)
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    sentinel = foreign / "experiment.txt"
    sentinel.write_text("preserve", encoding="utf-8")

    blocked = engine.prepare(_plan(foreign, _release()))

    assert blocked.target_inspection.kind is ManagedTargetKind.FOREIGN
    assert not blocked.applicable
    assert runner.calls == []
    with pytest.raises(AuthorizationError):
        engine.authorize(blocked, confirmed=True)
    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_bundled_wheel_hash_is_verified_for_resolution_and_apply(tmp_path):
    target = tmp_path / "managed"
    wheel = tmp_path / "napari_vipp-0.13.0a5-py3-none-any.whl"
    wheel.write_bytes(b"exact tagged wheel")
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    release = _release(wheel_path=wheel, wheel_sha256=digest)
    runner = _FakeRunner(_report(vipp_hash=digest, vipp_url=wheel.resolve().as_uri()))
    engine = _engine(tmp_path, runner)

    prepared = engine.prepare(_plan(target, release))
    assert prepared.wheel_sha256 == digest
    wheel.write_bytes(b"changed after confirmation")
    result = engine.apply(
        prepared,
        engine.authorize(prepared, confirmed=True),
    )

    assert result.status is InstallStatus.FAILED
    assert "changed after it was reviewed" in result.technical_error
    assert not target.exists()


def test_resolution_rejects_unhashed_artifact_and_unicode_is_not_mangled(tmp_path):
    report = json.loads(_report())
    report["install"][1]["download_info"]["archive_info"] = {}
    runner = _FakeRunner(json.dumps(report))
    engine = _engine(tmp_path, runner)
    messages: list[str] = []

    with pytest.raises(ResolutionError, match="SHA-256"):
        engine.prepare(
            _plan(tmp_path / "managed", _release()),
            progress=lambda event: messages.append(event.message),
        )

    assert "Checking the selected VIPP location…" in messages
    assert all("â" not in message for message in messages)


def test_preexisting_empty_destination_is_new_and_root_survives_rollback(tmp_path):
    target = tmp_path / "user-selected-empty-folder"
    target.mkdir()
    inspection = inspect_managed_target(
        target,
        release=_release(),
        track=ComputeTrack.CPU,
    )
    assert inspection.kind is ManagedTargetKind.NEW
    assert inspection.target_preexisting
    runner = _FakeRunner(fail_acceptance=True)
    engine = _engine(tmp_path, runner)
    prepared = engine.prepare(_plan(target, _release()))

    result = engine.apply(
        prepared,
        engine.authorize(prepared, confirmed=True),
    )

    assert result.status is InstallStatus.FAILED
    assert target.is_dir()
    assert next(target.iterdir(), None) is None
    assert target not in result.rollback.removed_paths


def test_failed_new_install_does_not_report_removed_parent_as_preserved(tmp_path):
    created_parent = tmp_path / "installer-created-parent"
    target = created_parent / "managed"
    engine = _engine(tmp_path, _FakeRunner(fail_acceptance=True))
    prepared = engine.prepare(_plan(target, _release()))

    result = engine.apply(
        prepared,
        engine.authorize(prepared, confirmed=True),
    )

    assert result.status is InstallStatus.FAILED
    assert result.rollback.completed
    assert not created_parent.exists()
    assert created_parent in result.rollback.removed_paths
    assert created_parent not in result.rollback.preserved_paths
    assert set(result.rollback.removed_paths).isdisjoint(
        result.rollback.preserved_paths
    )


def test_cpu_shortcut_is_staged_after_acceptance_and_atomically_owned(tmp_path):
    target = tmp_path / "managed"
    desktop = tmp_path / "Desktop"
    desktop.mkdir()
    runner = _FakeRunner()
    engine = _engine(tmp_path, runner)
    prepared = engine.prepare(_plan(target, _release(), shortcut_directory=desktop))
    assert [shortcut.destination for shortcut in prepared.shortcuts] == [
        desktop / "VIPP.lnk"
    ]

    result = engine.apply(
        prepared,
        engine.authorize(prepared, confirmed=True),
    )

    shortcut = desktop / "VIPP.lnk"
    assert result.succeeded
    assert shortcut.is_file()
    calls = runner.calls
    acceptance_index = max(
        index
        for index, call in enumerate(calls)
        if call[0].endswith("python.exe") and "--requirement" not in call
    )
    powershell_index = next(
        index for index, call in enumerate(calls) if call[0] == "powershell.exe"
    )
    assert powershell_index > acceptance_index
    shortcut_call = calls[powershell_index]
    assert Path(shortcut_call[shortcut_call.index("-WorkingDirectory") + 1]) == (
        tmp_path / "Documents"
    )
    assert (
        Path(shortcut_call[shortcut_call.index("-WorkingDirectory") + 1])
        != result.environment_root
    )
    ownership = inspect_ownership(target).record
    assert ownership is not None
    assert len(ownership.shortcuts) == 1
    assert ownership.shortcuts[0].path == shortcut
    assert (
        ownership.shortcuts[0].sha256
        == hashlib.sha256(shortcut.read_bytes()).hexdigest()
    )


def test_foreign_shortcut_is_never_overwritten(tmp_path):
    target = tmp_path / "managed"
    desktop = tmp_path / "Desktop"
    desktop.mkdir()
    shortcut = desktop / "VIPP.lnk"
    shortcut.write_bytes(b"a researcher's existing shortcut")
    engine = _engine(tmp_path, _FakeRunner())

    with pytest.raises(PreparationError, match="not the exact shortcut owned"):
        engine.prepare(_plan(target, _release(), shortcut_directory=desktop))

    assert shortcut.read_bytes() == b"a researcher's existing shortcut"
    assert not target.exists()


def test_shortcut_is_rolled_back_if_ownership_commit_fails(tmp_path, monkeypatch):
    target = tmp_path / "managed"
    desktop = tmp_path / "Desktop"
    desktop.mkdir()
    runner = _FakeRunner()
    engine = _engine(tmp_path, runner)
    prepared = engine.prepare(_plan(target, _release(), shortcut_directory=desktop))

    def fail_commit(_target, _record):
        raise OSError("simulated ownership commit failure")

    monkeypatch.setattr(engine_module, "write_ownership_record", fail_commit)
    result = engine.apply(
        prepared,
        engine.authorize(prepared, confirmed=True),
    )

    assert result.status is InstallStatus.FAILED
    assert not (desktop / "VIPP.lnk").exists()
    assert not target.exists()
    assert result.rollback.completed


def test_owned_shortcut_update_is_restored_if_commit_fails(tmp_path, monkeypatch):
    target = tmp_path / "managed"
    desktop = tmp_path / "Desktop"
    desktop.mkdir()
    shortcut = desktop / "VIPP.lnk"
    old_bytes = b"shortcut to the accepted old VIPP"
    shortcut.write_bytes(old_bytes)
    old_record = _record(target, version="0.13.0a4", shortcut=shortcut)
    manifest_before = (target / ".vipp-installer" / "ownership.json").read_bytes()
    runner = _FakeRunner()
    engine = _engine(tmp_path, runner)
    prepared = engine.prepare(_plan(target, _release(), shortcut_directory=desktop))

    def fail_commit(_target, _record):
        raise OSError("simulated ownership commit failure")

    monkeypatch.setattr(engine_module, "write_ownership_record", fail_commit)
    result = engine.apply(
        prepared,
        engine.authorize(prepared, confirmed=True),
    )

    assert result.status is InstallStatus.FAILED
    assert shortcut.read_bytes() == old_bytes
    assert (
        target / ".vipp-installer" / "ownership.json"
    ).read_bytes() == manifest_before
    assert inspect_ownership(target).record == old_record


def test_disabling_owned_desktop_shortcut_removes_it_after_acceptance(tmp_path):
    target = tmp_path / "managed"
    desktop = tmp_path / "Desktop"
    desktop.mkdir()
    shortcut = desktop / "VIPP.lnk"
    shortcut.write_bytes(b"owned old desktop shortcut")
    _record(target, version="0.13.0a4", shortcut=shortcut)
    engine = _engine(tmp_path, _FakeRunner())

    prepared = engine.prepare(_plan(target, _release()))

    assert len(prepared.shortcuts) == 1
    assert prepared.shortcuts[0].remove
    result = engine.apply(
        prepared,
        engine.authorize(prepared, confirmed=True),
    )

    assert result.succeeded
    assert not shortcut.exists()
    current = inspect_ownership(target).record
    assert current is not None
    assert current.shortcuts == ()


def test_owned_desktop_shortcut_removal_rolls_back_before_ownership(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "managed"
    desktop = tmp_path / "Desktop"
    desktop.mkdir()
    shortcut = desktop / "VIPP.lnk"
    old_bytes = b"owned old desktop shortcut"
    shortcut.write_bytes(old_bytes)
    old_record = _record(target, version="0.13.0a4", shortcut=shortcut)
    engine = _engine(tmp_path, _FakeRunner())
    prepared = engine.prepare(_plan(target, _release()))

    monkeypatch.setattr(
        engine_module,
        "write_ownership_record",
        lambda _target, _record: (_ for _ in ()).throw(
            OSError("simulated ownership commit failure")
        ),
    )
    result = engine.apply(
        prepared,
        engine.authorize(prepared, confirmed=True),
    )

    assert result.status is InstallStatus.FAILED
    assert shortcut.read_bytes() == old_bytes
    assert inspect_ownership(target).record == old_record


def test_dead_process_lock_is_quarantined_and_recovered(tmp_path):
    target = tmp_path / "managed"
    runner = _FakeRunner()
    engine = _engine(tmp_path, runner)
    prepared = engine.prepare(_plan(target, _release()))
    state = tmp_path / "installer-state"
    digest = hashlib.sha256(
        os.path.normcase(os.path.abspath(target)).encode("utf-8")
    ).hexdigest()
    lock = state / "locks" / f"{digest}.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text(
        json.dumps(
            {
                "schema": "napari-vipp-install-lock",
                "schema_version": 1,
                "run_id": "killed-run",
                "pid": 2_147_483_647,
                "created_at": _NOW.isoformat(),
            }
        ),
        encoding="utf-8",
    )

    result = engine.apply(
        prepared,
        engine.authorize(prepared, confirmed=True),
    )

    assert result.succeeded
    assert not lock.exists()
    assert list(lock.parent.glob("*.stale"))


def test_crash_recovery_shortcut_roots_come_from_known_folders(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "WrongProfile"))
    desktop = tmp_path / "Redirected Desktop"
    programs = tmp_path / "Redirected Programs"
    calls = []

    def probe(name):
        calls.append(name)
        return {"desktop": desktop, "programs": programs}.get(name)

    roots = engine_module._default_shortcut_roots(probe)

    assert roots == (desktop, programs / "VIPP")
    assert calls == ["desktop", "programs"]
    assert all("WrongProfile" not in str(root) for root in roots)


def test_result_journal_failure_after_commit_still_returns_success(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "managed"
    runner = _FakeRunner()
    engine = _engine(tmp_path, runner)
    prepared = engine.prepare(_plan(target, _release()))
    original = engine_module._atomic_json

    def fail_result_only(path, document):
        if Path(path).name == "result.json":
            raise OSError("simulated full disk after commit")
        return original(path, document)

    monkeypatch.setattr(engine_module, "_atomic_json", fail_result_only)
    result = engine.apply(
        prepared,
        engine.authorize(prepared, confirmed=True),
    )

    assert result.status is InstallStatus.SUCCEEDED
    assert inspect_ownership(target).record is not None


def test_completed_progress_observer_failure_cannot_reverse_commit(tmp_path):
    target = tmp_path / "managed"
    runner = _FakeRunner()
    engine = _engine(tmp_path, runner)
    prepared = engine.prepare(_plan(target, _release()))

    def observer(event):
        if event.stage is engine_module.ProgressStage.COMPLETED:
            raise RuntimeError("simulated closed GUI")

    result = engine.apply(
        prepared,
        engine.authorize(prepared, confirmed=True),
        progress=observer,
    )

    assert result.status is InstallStatus.SUCCEEDED
    assert inspect_ownership(target).record is not None


def test_late_cancel_observer_cannot_reverse_committed_success(tmp_path):
    target = tmp_path / "managed"
    token = CancellationToken()
    engine = _engine(tmp_path, _FakeRunner())
    prepared = engine.prepare(_plan(target, _release()))

    def observer(event):
        if event.stage is engine_module.ProgressStage.COMPLETED:
            token.cancel()
            token.raise_if_cancelled()

    result = engine.apply(
        prepared,
        engine.authorize(prepared, confirmed=True),
        progress=observer,
        cancellation=token,
    )

    assert token.is_cancelled()
    assert result.status is InstallStatus.SUCCEEDED
    assert inspect_ownership(target).record is not None


def test_pip_is_isolated_and_apply_uses_candidate_temp(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "PIP_EXTRA_INDEX_URL",
        "https://researcher:secret-token@untrusted.example/simple",
    )
    monkeypatch.setenv("PIP_TRUSTED_HOST", "untrusted.example")
    monkeypatch.setenv("PIP_DEFAULT_TIMEOUT", "1")
    monkeypatch.setenv("PIP_RETRIES", "0")
    target = tmp_path / "managed"
    runner = _FakeRunner()
    engine = _engine(tmp_path, runner)

    prepared = engine.prepare(_plan(target, _release()))
    result = engine.apply(
        prepared,
        engine.authorize(prepared, confirmed=True),
    )

    resolution_call = runner.calls[0]
    assert "--isolated" in resolution_call
    assert resolution_call[resolution_call.index("--index-url") + 1] == (
        "https://pypi.org/simple"
    )
    assert resolution_call[resolution_call.index("--timeout") + 1] == "120"
    assert resolution_call[resolution_call.index("--retries") + 1] == "8"
    assert "--trusted-host" not in resolution_call
    assert "PIP_EXTRA_INDEX_URL" not in runner.environments[0]
    assert "PIP_TRUSTED_HOST" not in runner.environments[0]
    assert "PIP_DEFAULT_TIMEOUT" not in runner.environments[0]
    assert "PIP_RETRIES" not in runner.environments[0]
    install_index = next(
        index for index, call in enumerate(runner.calls) if "--requirement" in call
    )
    install_call = runner.calls[install_index]
    assert "--isolated" in install_call
    assert "--only-binary=:all:" in install_call
    assert install_call[install_call.index("--timeout") + 1] == "120"
    assert install_call[install_call.index("--retries") + 1] == "8"
    assert "--trusted-host" not in install_call
    assert install_call[install_call.index("--index-url") + 1] == (
        "https://pypi.org/simple"
    )
    assert result.environment_root is not None
    expected_temp = result.environment_root / ".installer-tmp"
    assert Path(runner.environments[install_index]["TEMP"]) == expected_temp
    assert Path(runner.environments[install_index]["TMP"]) == expected_temp


def test_pip_report_subprocess_is_utf8_and_parses_unicode_metadata(monkeypatch):
    monkeypatch.setenv("pythonutf8", "0")
    monkeypatch.setenv("pythonioencoding", "cp1252")
    environment = engine_module._pip_environment()
    report_script = """
import json
import sys

document = {
    "version": "1",
    "pip_version": "26.1",
    "stdout_encoding": sys.stdout.encoding,
    "install": [{
        "download_info": {
            "url": "https://packages.example/vipp.whl",
            "archive_info": {"hashes": {"sha256": "a" * 64}},
        },
        "requested": True,
        "metadata": {
            "name": "napari-vipp",
            "version": "0.13.0a5",
            "summary": "Resolver progress \\u23f1",
        },
    }],
}
print(json.dumps(document, ensure_ascii=False))
"""

    result = engine_module.SubprocessCommandRunner().run(
        (sys.executable, "-c", report_script),
        env=environment,
    )

    assert environment["PYTHONUTF8"] == "1"
    assert environment["PYTHONIOENCODING"] == "utf-8"
    assert not any(
        key.casefold() in {"pythonutf8", "pythonioencoding"}
        and key not in {"PYTHONUTF8", "PYTHONIOENCODING"}
        for key in environment
    )
    assert result.returncode == 0, result.stderr
    assert "\ufffd" not in result.stdout
    document = json.loads(result.stdout)
    assert document["stdout_encoding"].replace("-", "").casefold() == "utf8"
    packages = engine_module._parse_pip_report(
        result.stdout,
        release=_release(),
        wheel_sha256=_VIPP_HASH,
        approved_hosts=frozenset({"packages.example"}),
    )
    assert [(package.name, package.version) for package in packages] == [
        ("napari-vipp", "0.13.0a5")
    ]


@pytest.mark.skipif(os.name != "nt", reason="Windows creation flags only")
def test_installer_subprocesses_do_not_open_a_console(monkeypatch):
    real_popen = engine_module.subprocess.Popen
    captured: dict[str, int] = {}

    def recording_popen(*args, **kwargs):
        captured["creationflags"] = kwargs.get("creationflags", 0)
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(engine_module.subprocess, "Popen", recording_popen)

    result = engine_module.SubprocessCommandRunner().run(
        (sys.executable, "-c", "print('ok')"),
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
    flags = captured["creationflags"]
    assert flags & getattr(
        engine_module.subprocess,
        "CREATE_NEW_PROCESS_GROUP",
        0x00000200,
    )
    assert flags & getattr(
        engine_module.subprocess,
        "CREATE_SUSPENDED",
        0x00000004,
    )
    assert flags & getattr(
        engine_module.subprocess,
        "CREATE_NO_WINDOW",
        0x08000000,
    )


@pytest.mark.parametrize(
    "source",
    [
        "http://files.pythonhosted.org/dependency.whl",
        "https://user:secret@files.pythonhosted.org/dependency.whl",
        "https://files.pythonhosted.org/dependency.whl?token=secret",
        "https://untrusted.example/dependency.whl",
        "ftp://files.pythonhosted.org/dependency.whl",
    ],
)
def test_resolution_rejects_unapproved_artifact_sources(tmp_path, source):
    report = json.loads(_report())
    report["install"][1]["download_info"]["url"] = source
    runner = _FakeRunner(json.dumps(report))
    engine = ManagedInstallerEngine(
        runner=runner,
        state_root=tmp_path / "state",
    )

    with pytest.raises(ResolutionError) as caught:
        engine.prepare(_plan(tmp_path / "managed", _release()))

    assert "secret" not in str(caught.value)
    assert "token=" not in str(caught.value)


def test_apply_rejects_mutated_resolved_package_set(tmp_path):
    target = tmp_path / "managed"
    engine = _engine(tmp_path, _FakeRunner())
    prepared = engine.prepare(_plan(target, _release()))
    authorization = engine.authorize(prepared, confirmed=True)
    changed_package = replace(
        prepared.packages[0],
        source_url="https://packages.example/another-dependency.whl",
    )
    tampered = replace(
        prepared,
        packages=(changed_package, *prepared.packages[1:]),
    )

    with pytest.raises(StalePreparedTransaction, match="changed after review"):
        engine.apply(tampered, authorization)

    assert not target.exists()


class _SimulatedProcessDeath(BaseException):
    pass


@pytest.mark.parametrize(
    "crash_phase",
    ["candidate_created", "shortcuts_staged", "shortcuts_committed"],
)
def test_crash_journal_rolls_back_precommit_phases(
    tmp_path,
    monkeypatch,
    crash_phase,
):
    target = tmp_path / "managed"
    desktop = tmp_path / "Desktop"
    desktop.mkdir()
    engine = _engine(tmp_path, _FakeRunner())
    prepared = engine.prepare(_plan(target, _release(), shortcut_directory=desktop))
    original = engine_module._write_transaction_journal

    def crash_at_phase(path, *, phase, **kwargs):
        if phase == crash_phase:
            if phase != "shortcuts_committed":
                original(path, phase=phase, **kwargs)
            raise _SimulatedProcessDeath(phase)
        original(path, phase=phase, **kwargs)

    monkeypatch.setattr(
        engine_module,
        "_write_transaction_journal",
        crash_at_phase,
    )
    with pytest.raises(_SimulatedProcessDeath):
        engine.apply(
            prepared,
            engine.authorize(prepared, confirmed=True),
        )
    monkeypatch.setattr(engine_module, "_write_transaction_journal", original)

    recovery = engine.recover_interrupted(
        target,
        shortcut_roots=(desktop,),
    )

    assert recovery.completed
    assert not (desktop / "VIPP.lnk").exists()
    assert inspect_ownership(target).record is None
    assert not target.exists()


def test_crash_after_ownership_commit_recovers_as_successful_install(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "managed"
    desktop = tmp_path / "Desktop"
    desktop.mkdir()
    engine = _engine(tmp_path, _FakeRunner())
    prepared = engine.prepare(_plan(target, _release(), shortcut_directory=desktop))
    original = engine_module.write_ownership_record

    def commit_then_die(selected, record):
        original(selected, record)
        raise _SimulatedProcessDeath("ownership committed")

    monkeypatch.setattr(engine_module, "write_ownership_record", commit_then_die)
    with pytest.raises(_SimulatedProcessDeath):
        engine.apply(
            prepared,
            engine.authorize(prepared, confirmed=True),
        )
    monkeypatch.setattr(engine_module, "write_ownership_record", original)

    recovery = engine.recover_interrupted(
        target,
        shortcut_roots=(desktop,),
    )

    assert recovery.completed
    assert (desktop / "VIPP.lnk").is_file()
    ownership = inspect_ownership(target)
    assert ownership.record is not None
    assert ownership.record.resolved_plan_id == prepared.resolution_id


def test_crash_during_owned_shortcut_removal_restores_old_install(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "managed"
    desktop = tmp_path / "Desktop"
    desktop.mkdir()
    shortcut = desktop / "VIPP.lnk"
    old_bytes = b"owned shortcut before update"
    shortcut.write_bytes(old_bytes)
    old_record = _record(target, version="0.13.0a4", shortcut=shortcut)
    engine = _engine(tmp_path, _FakeRunner())
    prepared = engine.prepare(_plan(target, _release()))
    original = engine_module._write_transaction_journal

    def die_after_shortcut_swap(path, *, phase, **kwargs):
        if phase == "shortcuts_committed":
            raise _SimulatedProcessDeath(phase)
        original(path, phase=phase, **kwargs)

    monkeypatch.setattr(
        engine_module,
        "_write_transaction_journal",
        die_after_shortcut_swap,
    )
    with pytest.raises(_SimulatedProcessDeath):
        engine.apply(
            prepared,
            engine.authorize(prepared, confirmed=True),
        )
    monkeypatch.setattr(engine_module, "_write_transaction_journal", original)

    recovery = engine.recover_interrupted(target)

    assert recovery.completed
    assert shortcut.read_bytes() == old_bytes
    assert inspect_ownership(target).record == old_record


def test_crash_after_ownership_finishes_owned_shortcut_removal(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "managed"
    desktop = tmp_path / "Desktop"
    desktop.mkdir()
    shortcut = desktop / "VIPP.lnk"
    shortcut.write_bytes(b"owned shortcut before update")
    _record(target, version="0.13.0a4", shortcut=shortcut)
    engine = _engine(tmp_path, _FakeRunner())
    prepared = engine.prepare(_plan(target, _release()))
    original = engine_module.write_ownership_record

    def commit_then_die(selected, record):
        original(selected, record)
        raise _SimulatedProcessDeath("ownership committed")

    monkeypatch.setattr(engine_module, "write_ownership_record", commit_then_die)
    with pytest.raises(_SimulatedProcessDeath):
        engine.apply(
            prepared,
            engine.authorize(prepared, confirmed=True),
        )
    monkeypatch.setattr(engine_module, "write_ownership_record", original)

    recovery = engine.recover_interrupted(target)

    assert recovery.completed
    assert not shortcut.exists()
    current = inspect_ownership(target).record
    assert current is not None
    assert current.shortcuts == ()


def test_old_retired_environments_are_bounded_after_update(tmp_path):
    target = tmp_path / "managed"
    old = _record(target, version="0.13.0a4")
    retired: list[OwnedEnvironment] = []
    for index in range(3):
        environment = managed_environments_root(target) / f"retired-{index}"
        environment.mkdir()
        marker = environment / ".vipp-install-candidate.json"
        marker.write_text(f"retired marker {index}", encoding="utf-8")
        retired.append(
            OwnedEnvironment(
                environment,
                hashlib.sha256(marker.read_bytes()).hexdigest(),
            )
        )
    write_ownership_record(
        target,
        replace(old, retired_environments=tuple(retired)),
    )
    engine = _engine(tmp_path, _FakeRunner())
    prepared = engine.prepare(_plan(target, _release()))

    result = engine.apply(
        prepared,
        engine.authorize(prepared, confirmed=True),
    )

    assert result.succeeded
    assert result.retirement_cleanup.completed
    assert all(not environment.path.exists() for environment in retired)
    current = inspect_ownership(target).record
    assert current is not None
    assert current.retired_environment_roots == (old.environment_root,)


def test_incomplete_rollback_names_preserved_candidate(tmp_path, monkeypatch):
    target = tmp_path / "managed"
    engine = _engine(tmp_path, _FakeRunner(fail_acceptance=True))
    prepared = engine.prepare(_plan(target, _release()))
    original = engine_module.shutil.rmtree

    def refuse_candidate(path, *args, **kwargs):
        candidate = Path(path)
        if candidate.parent == managed_environments_root(target):
            raise OSError("candidate is held open")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(engine_module.shutil, "rmtree", refuse_candidate)
    result = engine.apply(
        prepared,
        engine.authorize(prepared, confirmed=True),
    )

    assert result.status is InstallStatus.FAILED
    assert not result.rollback.completed
    assert result.rollback.errors
    assert any(
        path.parent == managed_environments_root(target)
        for path in result.rollback.preserved_paths
    )
    assert "could not be removed" in result.message


def test_run_history_is_bounded_and_active_journal_is_preserved(tmp_path):
    state = tmp_path / "state"
    runs = state / "runs"
    runs.mkdir(parents=True)
    created: list[Path] = []
    for _index in range(30):
        run = runs / str(uuid.uuid4())
        run.mkdir()
        (run / "result.json").write_text("{}", encoding="utf-8")
        created.append(run)
    protected = created[0]
    transactions = state / "transactions"
    transactions.mkdir()
    (transactions / "active.json").write_text(
        json.dumps({"run_directory": str(protected)}),
        encoding="utf-8",
    )

    engine_module._prune_state_history(state)

    assert protected.is_dir()
    assert len(tuple(runs.iterdir())) <= 26


def test_atomic_journal_replace_retries_transient_windows_scanner_lock(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "journal.tmp"
    destination = tmp_path / "journal.json"
    source.write_text("durable journal", encoding="utf-8")
    original = engine_module.os.replace
    attempts = []
    delays = []

    def transient_then_replace(selected_source, selected_destination):
        attempts.append((selected_source, selected_destination))
        if len(attempts) <= 3:
            error = PermissionError(
                13,
                "simulated Windows scanner lock",
                str(selected_source),
                str(selected_destination),
            )
            error.winerror = 5
            raise error
        original(selected_source, selected_destination)

    monkeypatch.setattr(engine_module.os, "name", "nt")
    monkeypatch.setattr(engine_module.os, "replace", transient_then_replace)
    monkeypatch.setattr(engine_module.time, "sleep", delays.append)

    engine_module._replace_with_retry(source, destination)

    assert destination.read_text(encoding="utf-8") == "durable journal"
    assert len(attempts) == 4
    assert delays == list(engine_module._ATOMIC_REPLACE_RETRY_DELAYS[:3])


def test_resolution_temp_reserve_is_track_aware_on_low_system_drive(
    tmp_path,
    monkeypatch,
):
    cpu_plan = _plan(tmp_path / "managed", _release())
    cuda_plan = replace(
        cpu_plan,
        request=replace(cpu_plan.request, track=ComputeTrack.CUDA13),
        required_free_bytes=15 * 1024**3,
    )
    monkeypatch.setattr(
        engine_module.tempfile,
        "gettempdir",
        lambda: "C:/Windows/Temp",
    )
    monkeypatch.setattr(
        engine_module,
        "_nearest_existing_path",
        lambda path: Path(path),
    )

    class _Usage:
        def __init__(self, free):
            self.free = free

    def disk_usage(path):
        normalized = str(path).replace("\\", "/").casefold()
        return _Usage(2 * 1024**3 if normalized.startswith("c:/") else 20 * 1024**3)

    monkeypatch.setattr(engine_module.shutil, "disk_usage", disk_usage)
    engine_module._validate_resolution_temp_capacity(
        cpu_plan,
        state_root=Path("D:/VIPP/installer"),
    )
    with pytest.raises(PreparationError, match="temporary downloads"):
        engine_module._validate_resolution_temp_capacity(
            cuda_plan,
            state_root=Path("D:/VIPP/installer"),
        )


def test_install_registers_and_uninstall_removes_only_owned_objects(tmp_path):
    target = tmp_path / "managed"
    shortcut_root = tmp_path / "Desktop"
    shortcut_root.mkdir()
    setup_source = tmp_path / "signed-release-setup.exe"
    setup_source.write_bytes(b"signed setup release one")
    registry = _MemoryRegistry()
    engine = _engine_with_setup(
        tmp_path,
        _FakeRunner(),
        setup_source,
        registry,
    )
    prepared = engine.prepare(_plan(target, _release()))

    result = engine.apply(
        prepared,
        engine.authorize(prepared, confirmed=True),
    )

    assert result.succeeded
    assert not result.registration_warning
    record = inspect_ownership(target).record
    assert record is not None
    assert record.uninstaller_path is not None
    assert record.uninstaller_path.is_file()
    assert "cpu" in {part.casefold() for part in record.uninstaller_path.parts}
    assert (
        record.uninstaller_sha256
        == hashlib.sha256(setup_source.read_bytes()).hexdigest()
    )
    assert CPU_REGISTRY_KEY.casefold() in registry.values

    uninstaller = ManagedUninstaller(registry=registry)
    removal = uninstaller.prepare(
        target,
        shortcut_roots=(shortcut_root,),
    )
    outcome = uninstaller.apply(
        removal,
        uninstaller.authorize(removal),
        current_executable=tmp_path / "another-process.exe",
    )

    assert outcome.status is UninstallStatus.COMPLETED
    assert not target.exists()
    assert not record.uninstaller_path.exists()
    assert CPU_REGISTRY_KEY.casefold() not in registry.values
    assert setup_source.is_file()


def test_cpu_and_cuda_cached_setup_and_registry_entries_do_not_collide(tmp_path):
    shortcut_root = tmp_path / "Desktop"
    shortcut_root.mkdir()
    setup_source = tmp_path / "signed-release-setup.exe"
    setup_source.write_bytes(b"one immutable signed setup")
    registry = _MemoryRegistry()
    cpu_target = tmp_path / "cpu-managed"
    cuda_target = tmp_path / "cuda-managed"
    cpu_engine = _engine_with_setup(
        tmp_path,
        _FakeRunner(),
        setup_source,
        registry,
        run_id="11111111-1111-4111-8111-111111111111",
    )
    cpu_prepared = cpu_engine.prepare(_plan(cpu_target, _release()))
    assert cpu_engine.apply(
        cpu_prepared,
        cpu_engine.authorize(cpu_prepared, confirmed=True),
    ).succeeded
    cuda_plan = _plan(cuda_target, _release())
    cuda_plan = replace(
        cuda_plan,
        request=replace(cuda_plan.request, track=ComputeTrack.CUDA13),
    )
    cuda_engine = _engine_with_setup(
        tmp_path,
        _FakeRunner(),
        setup_source,
        registry,
        run_id="22222222-2222-4222-8222-222222222222",
    )
    cuda_prepared = cuda_engine.prepare(cuda_plan)
    assert cuda_engine.apply(
        cuda_prepared,
        cuda_engine.authorize(cuda_prepared, confirmed=True),
    ).succeeded
    cpu_record = inspect_ownership(cpu_target).record
    cuda_record = inspect_ownership(cuda_target).record
    assert cpu_record is not None and cuda_record is not None
    assert cpu_record.uninstaller_path != cuda_record.uninstaller_path
    assert CPU_REGISTRY_KEY.casefold() in registry.values
    assert CUDA13_REGISTRY_KEY.casefold() in registry.values

    uninstaller = ManagedUninstaller(registry=registry)
    removal = uninstaller.prepare(
        cpu_target,
        shortcut_roots=(shortcut_root,),
    )
    outcome = uninstaller.apply(
        removal,
        uninstaller.authorize(removal),
        current_executable=tmp_path / "another-process.exe",
    )

    assert outcome.status is UninstallStatus.COMPLETED
    assert cuda_record.uninstaller_path is not None
    assert cuda_record.uninstaller_path.is_file()
    assert inspect_ownership(cuda_target).record == cuda_record
    assert CUDA13_REGISTRY_KEY.casefold() in registry.values


def test_repair_updates_cached_setup_and_registry_then_retires_old_copy(tmp_path):
    target = tmp_path / "managed"
    registry = _MemoryRegistry()
    first_source = tmp_path / "setup-one.exe"
    first_source.write_bytes(b"signed setup one")
    first_engine = _engine_with_setup(
        tmp_path,
        _FakeRunner(),
        first_source,
        registry,
        run_id="33333333-3333-4333-8333-333333333333",
    )
    first_prepared = first_engine.prepare(_plan(target, _release()))
    assert first_engine.apply(
        first_prepared,
        first_engine.authorize(first_prepared, confirmed=True),
    ).succeeded
    first_record = inspect_ownership(target).record
    assert first_record is not None and first_record.uninstaller_path is not None
    first_cached = first_record.uninstaller_path

    second_source = tmp_path / "setup-two.exe"
    second_source.write_bytes(b"signed setup two")
    second_engine = _engine_with_setup(
        tmp_path,
        _FakeRunner(),
        second_source,
        registry,
        run_id="44444444-4444-4444-8444-444444444444",
    )
    second_prepared = second_engine.prepare(
        _plan(target, _release()),
        repair=True,
    )
    result = second_engine.apply(
        second_prepared,
        second_engine.authorize(second_prepared, confirmed=True),
    )

    assert result.succeeded
    assert not result.registration_warning
    second_record = inspect_ownership(target).record
    assert second_record is not None and second_record.uninstaller_path is not None
    assert second_record.uninstaller_path != first_cached
    assert second_record.uninstaller_path.is_file()
    assert not first_cached.exists()
    values = registry.values[CPU_REGISTRY_KEY.casefold()]
    assert values["VippUninstallerSha256"] == second_record.uninstaller_sha256


@pytest.mark.parametrize("partial_write_count", [None, 8])
def test_failed_registry_update_retains_old_exe_and_journal_recovers(
    tmp_path,
    partial_write_count,
):
    target = tmp_path / "managed"
    registry = _MemoryRegistry()
    first_source = tmp_path / "setup-one.exe"
    first_source.write_bytes(b"signed setup one")
    first_engine = _engine_with_setup(
        tmp_path,
        _FakeRunner(),
        first_source,
        registry,
        run_id="55555555-5555-4555-8555-555555555555",
    )
    first_prepared = first_engine.prepare(_plan(target, _release()))
    assert first_engine.apply(
        first_prepared,
        first_engine.authorize(first_prepared, confirmed=True),
    ).succeeded
    first_record = inspect_ownership(target).record
    assert first_record is not None and first_record.uninstaller_path is not None
    first_cached = first_record.uninstaller_path
    old_registry = dict(registry.values[CPU_REGISTRY_KEY.casefold()])

    second_source = tmp_path / "setup-two.exe"
    second_source.write_bytes(b"signed setup two")
    registry.fail_writes = True
    registry.partial_write_count = partial_write_count
    second_engine = _engine_with_setup(
        tmp_path,
        _FakeRunner(),
        second_source,
        registry,
        run_id="66666666-6666-4666-8666-666666666666",
    )
    second_prepared = second_engine.prepare(
        _plan(target, _release()),
        repair=True,
    )
    result = second_engine.apply(
        second_prepared,
        second_engine.authorize(second_prepared, confirmed=True),
    )

    assert result.succeeded
    assert result.registration_warning
    assert first_cached.is_file()
    if partial_write_count is None:
        assert registry.values[CPU_REGISTRY_KEY.casefold()] == old_registry
    else:
        assert registry.values[CPU_REGISTRY_KEY.casefold()] != old_registry
    new_record = inspect_ownership(target).record
    assert new_record is not None and new_record.uninstaller_path is not None
    assert new_record.uninstaller_path.is_file()
    journals = tuple((tmp_path / "installer-state" / "transactions").glob("*.json"))
    assert len(journals) == 1

    registry.fail_writes = False
    registry.partial_write_count = None
    recovery = second_engine.recover_interrupted(target)

    assert recovery.completed
    assert not first_cached.exists()
    assert not journals[0].exists()
    values = registry.values[CPU_REGISTRY_KEY.casefold()]
    assert values["VippUninstallerSha256"] == new_record.uninstaller_sha256
    assert str(new_record.uninstaller_path) in str(values["UninstallString"])


def test_crash_after_registry_swap_replays_before_old_cache_cleanup(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "managed"
    registry = _MemoryRegistry()
    first_source = tmp_path / "setup-one.exe"
    first_source.write_bytes(b"signed setup one")
    first_engine = _engine_with_setup(
        tmp_path,
        _FakeRunner(),
        first_source,
        registry,
        run_id="77777777-7777-4777-8777-777777777777",
    )
    first_prepared = first_engine.prepare(_plan(target, _release()))
    assert first_engine.apply(
        first_prepared,
        first_engine.authorize(first_prepared, confirmed=True),
    ).succeeded
    old_record = inspect_ownership(target).record
    assert old_record is not None and old_record.uninstaller_path is not None

    second_source = tmp_path / "setup-two.exe"
    second_source.write_bytes(b"signed setup two")
    second_engine = _engine_with_setup(
        tmp_path,
        _FakeRunner(),
        second_source,
        registry,
        run_id="88888888-8888-4888-8888-888888888888",
    )
    prepared = second_engine.prepare(_plan(target, _release()), repair=True)
    original = uninstall_module.register_apps_and_features

    def register_then_die(*args, **kwargs):
        original(*args, **kwargs)
        raise _SimulatedProcessDeath("registry committed")

    monkeypatch.setattr(
        uninstall_module,
        "register_apps_and_features",
        register_then_die,
    )
    with pytest.raises(_SimulatedProcessDeath):
        second_engine.apply(
            prepared,
            second_engine.authorize(prepared, confirmed=True),
        )
    monkeypatch.setattr(
        uninstall_module,
        "register_apps_and_features",
        original,
    )

    new_record = inspect_ownership(target).record
    assert new_record is not None and new_record.uninstaller_path is not None
    assert new_record.uninstaller_path.is_file()
    assert old_record.uninstaller_path.is_file()
    values = registry.values[CPU_REGISTRY_KEY.casefold()]
    assert values["VippUninstallerSha256"] == new_record.uninstaller_sha256

    recovery = second_engine.recover_interrupted(target)

    assert recovery.completed
    assert not old_record.uninstaller_path.exists()
    assert new_record.uninstaller_path.is_file()


def test_crash_after_cached_setup_swap_is_recovered(tmp_path, monkeypatch):
    target = tmp_path / "managed"
    setup_source = tmp_path / "signed-release-setup.exe"
    setup_source.write_bytes(b"signed setup crash recovery")
    registry = _MemoryRegistry()
    engine = _engine_with_setup(
        tmp_path,
        _FakeRunner(),
        setup_source,
        registry,
    )
    prepared = engine.prepare(_plan(target, _release()))
    original = engine_module._write_transaction_journal

    def die_after_setup_swap(path, *, phase, **kwargs):
        if phase == "setup_committed":
            raise _SimulatedProcessDeath(phase)
        original(path, phase=phase, **kwargs)

    monkeypatch.setattr(
        engine_module,
        "_write_transaction_journal",
        die_after_setup_swap,
    )
    with pytest.raises(_SimulatedProcessDeath):
        engine.apply(
            prepared,
            engine.authorize(prepared, confirmed=True),
        )
    assert prepared.persistent_setup_path is not None
    assert prepared.persistent_setup_path.is_file()
    monkeypatch.setattr(engine_module, "_write_transaction_journal", original)

    recovery = engine.recover_interrupted(target)

    assert recovery.completed
    assert not prepared.persistent_setup_path.exists()
    assert not target.exists()


@pytest.mark.parametrize("fail_postcommit_cleanup", [False, True])
def test_same_version_reinstall_keeps_terminal_journal_until_commit(
    tmp_path,
    monkeypatch,
    fail_postcommit_cleanup,
):
    target = tmp_path / "managed"
    desktop = tmp_path / "Desktop"
    programs = tmp_path / "Programs"
    desktop.mkdir()
    programs.mkdir()
    start_menu = programs / "VIPP"
    start_menu.mkdir()
    setup_source = tmp_path / "signed-release-setup.exe"
    setup_source.write_bytes(b"same signed setup")
    registry = _MemoryRegistry()
    first_engine = _engine_with_setup(
        tmp_path,
        _FakeRunner(),
        setup_source,
        registry,
        run_id="11111111-1111-4111-8111-111111111111",
    )
    first_prepared = first_engine.prepare(_plan(target, _release()))
    first_result = first_engine.apply(
        first_prepared,
        first_engine.authorize(first_prepared, confirmed=True),
    )
    assert first_result.succeeded
    old_record = inspect_ownership(target).record
    assert old_record is not None and old_record.uninstaller_path is not None

    original_uninstaller_cleanup = ManagedUninstaller._remove_or_defer_uninstaller

    def die_before_cache_cleanup(*_args, **_kwargs):
        raise _SimulatedProcessDeath("terminal uninstall")

    monkeypatch.setattr(
        ManagedUninstaller,
        "_remove_or_defer_uninstaller",
        staticmethod(die_before_cache_cleanup),
    )
    uninstaller = ManagedUninstaller(
        registry=registry,
        current_executable=old_record.uninstaller_path,
    )
    removal = uninstaller.prepare(
        target,
        shortcut_roots=(desktop, start_menu),
    )
    with pytest.raises(_SimulatedProcessDeath):
        uninstaller.apply(removal, uninstaller.authorize(removal))
    terminal_journals = tuple(
        old_record.uninstaller_path.parent.glob(".vipp-uninstall-*.json")
    )
    assert len(terminal_journals) == 1
    monkeypatch.setattr(
        ManagedUninstaller,
        "_remove_or_defer_uninstaller",
        staticmethod(original_uninstaller_cleanup),
    )

    second_engine = _engine_with_setup(
        tmp_path,
        _FakeRunner(),
        setup_source,
        registry,
        run_id="22222222-2222-4222-8222-222222222222",
    )
    second_prepared = second_engine.prepare(_plan(target, _release()))
    original_stage = second_engine._stage_persistent_setup

    def review_then_die(*args, **kwargs):
        original_stage(*args, **kwargs)
        raise _SimulatedProcessDeath("after cache adoption review")

    monkeypatch.setattr(
        second_engine,
        "_stage_persistent_setup",
        review_then_die,
    )
    with pytest.raises(_SimulatedProcessDeath):
        second_engine.apply(
            second_prepared,
            second_engine.authorize(second_prepared, confirmed=True),
        )
    assert terminal_journals[0].exists()
    monkeypatch.setattr(
        second_engine,
        "_stage_persistent_setup",
        original_stage,
    )
    recovery = second_engine.recover_interrupted(target)
    assert recovery.completed

    third_engine = _engine_with_setup(
        tmp_path,
        _FakeRunner(),
        setup_source,
        registry,
        run_id="33333333-3333-4333-8333-333333333333",
    )
    third_prepared = third_engine.prepare(_plan(target, _release()))
    assert third_prepared.applicable
    assert terminal_journals[0].exists()
    original_retirement = uninstall_module.remove_superseded_uninstall_recoveries
    if fail_postcommit_cleanup:

        def fail_retirement(*_args, **_kwargs):
            raise OSError("simulated old journal cleanup failure")

        monkeypatch.setattr(
            uninstall_module,
            "remove_superseded_uninstall_recoveries",
            fail_retirement,
        )
    third_result = third_engine.apply(
        third_prepared,
        third_engine.authorize(third_prepared, confirmed=True),
    )
    assert third_result.succeeded
    if fail_postcommit_cleanup:
        assert terminal_journals[0].exists()
        assert "Old uninstall recovery cleanup" in third_result.registration_warning
        monkeypatch.setattr(
            uninstall_module,
            "remove_superseded_uninstall_recoveries",
            original_retirement,
        )
    else:
        assert not terminal_journals[0].exists()
    installed = inspect_ownership(target).record
    assert installed is not None
    removal_review = ManagedUninstaller(registry=registry).prepare(
        target,
        shortcut_roots=(desktop, start_menu),
    )
    assert removal_review.installation_id == installed.installation_id
    if fail_postcommit_cleanup:
        remover = ManagedUninstaller(registry=registry)
        removal_result = remover.apply(
            removal_review,
            remover.authorize(removal_review),
            current_executable=tmp_path / "downloaded-setup.exe",
        )
        assert removal_result.completed
        assert not terminal_journals[0].exists()


def test_environment_with_reparse_descendant_is_foreign(tmp_path):
    target = tmp_path / "managed"
    record = _record(target, version="0.13.0a5")
    outside = tmp_path / "outside"
    outside.mkdir()
    link = record.environment_root / "redirected-data"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("This Windows account cannot create a test symbolic link.")

    inspection = inspect_managed_target(
        target,
        release=_release(),
        track=ComputeTrack.CPU,
    )

    assert inspection.kind is ManagedTargetKind.FOREIGN
