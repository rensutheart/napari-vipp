from __future__ import annotations

import json
import os
import subprocess
import uuid
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from napari_vipp.installer import discovery as discovery_module
from napari_vipp.installer.discovery import (
    DiscoveryServices,
    InterpreterProbe,
)
from napari_vipp.installer.frontend import (
    BlockedAction,
    InstallerSelection,
    TargetKind,
    TrackChoice,
)
from napari_vipp.installer.models import (
    ComputeTrack,
    GpuDeviceSnapshot,
    NvidiaSnapshot,
    ReleaseSpec,
    ShortcutScope,
)
from napari_vipp.installer.ownership import (
    OwnedEnvironment,
    OwnedShortcut,
    OwnershipRecord,
    inspect_ownership,
    write_ownership_record,
)
from napari_vipp.installer.python_discovery import PythonCandidate
from napari_vipp.installer.uninstall import registry_plan_from_record
from napari_vipp.installer.windows_backend import WindowsInstallerBackend


@pytest.fixture(autouse=True)
def _simulate_windows_host(monkeypatch):
    """Exercise the Windows backend consistently on every CI host."""

    monkeypatch.setattr(discovery_module.sys, "platform", "win32")
    monkeypatch.setattr(discovery_module.platform, "system", lambda: "Windows")
    monkeypatch.setattr(discovery_module.platform, "machine", lambda: "AMD64")


class _Engine:
    def __init__(self, kind="new"):
        self.kind = kind
        self.prepared_plans = []
        self.authorizations = []
        self.applied = []

    def prepare(self, plan, *, progress, cancellation, repair=False):
        self.prepared_plans.append((plan, repair))
        progress(
            SimpleNamespace(
                stage=SimpleNamespace(value="resolve"),
                message="Resolved packages.",
                completed=4,
                total=5,
            )
        )
        return SimpleNamespace(
            target_inspection=SimpleNamespace(
                kind=SimpleNamespace(value=("repair" if repair else self.kind)),
                installed_version=None,
                launcher_path=None,
                reason="",
            ),
            operation=SimpleNamespace(value=("repair" if repair else self.kind)),
            as_dict=lambda: {"resolution": "reviewed"},
        )

    def authorize(self, prepared, *, confirmed):
        self.authorizations.append((prepared, confirmed))
        return "authorization"

    def apply(self, prepared, authorization, *, progress, cancellation):
        self.applied.append((prepared, authorization))
        return SimpleNamespace(
            status=SimpleNamespace(value="success"),
            launcher_path=Path("C:/VIPP/current/Scripts/vipp-app.exe"),
            as_dict=lambda: {"status": "success"},
        )


class _RecoveryEngine(_Engine):
    def __init__(self):
        super().__init__()
        self.recovered_roots = []

    def recover_interrupted(self, root):
        self.recovered_roots.append(Path(root))
        return SimpleNamespace(completed=True, errors=())


class _RemovingRecoveryEngine(_RecoveryEngine):
    def recover_interrupted(self, root):
        result = super().recover_interrupted(root)
        (Path(root) / ".vipp-installer" / "ownership.json").unlink()
        return result


class _Registry:
    def __init__(self, values=None):
        self.values = values

    def read_values(self, _key):
        return self.values


def _services(tmp_path, *, gpu_ok=False, local_app_data=None):
    desktop = tmp_path / "Desktop"
    desktop.mkdir(exist_ok=True)
    programs = tmp_path / "Programs"
    programs.mkdir(exist_ok=True)
    documents = tmp_path / "Documents"
    documents.mkdir(exist_ok=True)
    return DiscoveryServices(
        interpreter_probe=lambda path: InterpreterProbe(
            executable=path,
            base_executable=path,
            implementation="cpython",
            version=(3, 12, 10),
            pointer_bits=64,
        ),
        package_probe=lambda _path: (),
        nvidia_probe=lambda _python: NvidiaSnapshot(
            probe_succeeded=gpu_ok,
            driver_api_version=13030 if gpu_ok else None,
            devices=(
                (
                    GpuDeviceSnapshot(
                        "NVIDIA GeForce RTX 4050 Laptop GPU",
                        (8, 9),
                        6 * 1024**3,
                    ),
                )
                if gpu_ok
                else ()
            ),
            error="No qualifying NVIDIA driver" if not gpu_ok else "",
        ),
        disk_probe=lambda _path: 30 * 1024**3,
        reparse_probe=lambda _path: False,
        remote_path_probe=lambda _path: False,
        known_folder_probe=lambda name: {
            "desktop": desktop,
            "programs": programs,
            "documents": documents,
            "local_app_data": local_app_data or tmp_path,
        }.get(name),
    )


def _candidate_finder(python, **_kwargs):
    return (
        PythonCandidate(
            executable=python,
            version=(3, 12, 10),
            source="test",
        ),
    )


def _write_owned_cuda_installation(
    target: Path,
    *,
    base_python: Path,
    version: str = "0.13.0a6",
    shortcuts: tuple[OwnedShortcut, ...] = (),
    retired_environments: tuple[OwnedEnvironment, ...] = (),
    uninstaller_path: Path | None = None,
    track: ComputeTrack = ComputeTrack.CUDA13,
) -> None:
    environment = target / ".vipp-installer" / "environments" / f"{version}-current"
    uninstaller_sha256 = ""
    if uninstaller_path is not None:
        uninstaller_path.parent.mkdir(parents=True, exist_ok=True)
        uninstaller_bytes = b"signed cached VIPP setup"
        uninstaller_path.write_bytes(uninstaller_bytes)
        uninstaller_sha256 = sha256(uninstaller_bytes).hexdigest()
    record = OwnershipRecord(
        installation_id=str(uuid.uuid4()),
        managed_root=target,
        environment_root=environment,
        distribution="napari-vipp",
        version=version,
        track=track,
        base_python=base_python,
        resolved_plan_id="a" * 64,
        packages=(),
        environment_marker_sha256="b" * 64,
        shortcuts=shortcuts,
        retired_environments=retired_environments,
        uninstaller_path=uninstaller_path,
        uninstaller_sha256=uninstaller_sha256,
        created_at="2026-08-13T00:00:00+00:00",
        updated_at="2026-08-13T00:00:00+00:00",
    )
    write_ownership_record(target, record)


def test_missing_python_is_a_guided_non_mutating_screen(tmp_path):
    engine = _Engine()
    backend = WindowsInstallerBackend(
        engine=engine,
        release=ReleaseSpec("napari-vipp", "0.13.0a4"),
        services=_services(tmp_path),
        environ={"LOCALAPPDATA": str(tmp_path)},
        candidate_finder=lambda **_kwargs: (),
    )

    prepared = backend.inspect(
        InstallerSelection(),
        progress=lambda _event: None,
        cancellation=__import__("threading").Event(),
    )

    assert prepared.kind is TargetKind.BLOCKED
    assert "Python 3.12 or 3.13" in prepared.plain_summary
    assert prepared.help_url.endswith("/downloads/release/python-31210/")
    assert engine.prepared_plans == []
    assert not prepared.target.exists()


def test_known_folder_not_spoofed_localappdata_controls_default_and_recovery(
    tmp_path,
):
    trusted = tmp_path / "trusted-local-app-data"
    spoofed = tmp_path / "spoofed-local-app-data"
    engine = _RecoveryEngine()
    backend = WindowsInstallerBackend(
        engine=engine,
        release=ReleaseSpec("napari-vipp", "0.13.0a7"),
        services=_services(tmp_path, local_app_data=trusted),
        environ={"LOCALAPPDATA": str(spoofed)},
        candidate_finder=lambda **_kwargs: (),
    )

    prepared = backend.inspect(
        InstallerSelection(track=TrackChoice.CPU),
        progress=lambda _event: None,
        cancellation=__import__("threading").Event(),
    )
    expected = trusted / "VIPP" / "environments" / "cpu"

    assert prepared.target == expected
    assert engine.recovered_roots == [expected]
    assert str(spoofed) not in prepared.technical_details


def test_backend_created_engine_uses_same_known_folder_state_authority(tmp_path):
    trusted = tmp_path / "trusted-local-app-data"
    spoofed = tmp_path / "spoofed-local-app-data"
    backend = WindowsInstallerBackend(
        release=ReleaseSpec("napari-vipp", "0.13.0a7"),
        services=_services(tmp_path, local_app_data=trusted),
        environ={"LOCALAPPDATA": str(spoofed)},
        candidate_finder=lambda **_kwargs: (),
    )

    engine = backend._selected_engine()

    assert engine._state_root == trusted / "VIPP" / "installer"


def test_automatic_existing_napari_route_uses_location_chooser_without_mutation(
    tmp_path,
):
    python = tmp_path / "napari-env" / "Scripts" / "python.exe"
    engine = _RecoveryEngine()
    backend = WindowsInstallerBackend(
        engine=engine,
        release=ReleaseSpec("napari-vipp", "0.13.0a4"),
        services=_services(tmp_path),
        environ={"LOCALAPPDATA": str(tmp_path)},
        candidate_finder=lambda **_kwargs: (),
    )

    prepared = backend.inspect(
        InstallerSelection(existing_python=python),
        progress=lambda _event: None,
        cancellation=__import__("threading").Event(),
    )

    assert prepared.kind is TargetKind.BLOCKED
    assert prepared.track is None
    assert prepared.blocked_action is BlockedAction.USE_DEFAULT_LOCATION
    assert "keeps existing napari environments unchanged" in prepared.plain_summary
    assert engine.recovered_roots == []
    assert engine.prepared_plans == []


def test_managed_looking_user_environment_does_not_receive_owned_recovery(tmp_path):
    managed_root = tmp_path / "user-owned"
    python = (
        managed_root
        / ".vipp-installer"
        / "environments"
        / "personal"
        / "Scripts"
        / "python.exe"
    )
    python.parent.mkdir(parents=True)
    python.touch()
    sentinel = managed_root / "personal-notes.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    engine = _RecoveryEngine()
    backend = WindowsInstallerBackend(
        engine=engine,
        release=ReleaseSpec("napari-vipp", "0.13.0a7"),
        services=_services(tmp_path),
        environ={"LOCALAPPDATA": str(tmp_path)},
        candidate_finder=lambda **_kwargs: (),
    )

    prepared = backend.inspect(
        InstallerSelection(existing_python=python),
        progress=lambda _event: None,
        cancellation=__import__("threading").Event(),
    )

    assert prepared.kind is TargetKind.BLOCKED
    assert prepared.blocked_action is BlockedAction.USE_DEFAULT_LOCATION
    assert engine.recovered_roots == []
    assert engine.prepared_plans == []
    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_existing_unicode_cuda_route_offers_fresh_managed_location(tmp_path):
    python = tmp_path / "napari-env-Ångström" / "Scripts" / "python.exe"
    backend = WindowsInstallerBackend(
        engine=_Engine(),
        release=ReleaseSpec("napari-vipp", "0.13.0a7"),
        services=_services(tmp_path),
        environ={"LOCALAPPDATA": str(tmp_path)},
        candidate_finder=lambda **_kwargs: (),
    )

    prepared = backend.inspect(
        InstallerSelection(
            track=TrackChoice.CUDA13,
            existing_python=python,
        ),
        progress=lambda _event: None,
        cancellation=__import__("threading").Event(),
    )

    assert prepared.kind is TargetKind.BLOCKED
    assert prepared.track is ComputeTrack.CUDA13
    assert prepared.blocked_action is BlockedAction.USE_CPU
    assert "left that environment completely unchanged" in prepared.plain_summary
    assert "Use CPU one-click setup" in prepared.reason
    assert "will not move, rename, or edit" in prepared.reason


@pytest.mark.parametrize("selected_environment", ["current", "retired"])
def test_existing_owned_unicode_cuda_route_recovers_then_requires_uninstall(
    tmp_path,
    selected_environment,
):
    base_python = tmp_path / "Python312" / "python.exe"
    target = tmp_path / "VIPP GPU Ångström"
    environments = target / ".vipp-installer" / "environments"
    current = environments / "0.13.0a6-current"
    retired = environments / "0.13.0a5-retired"
    _write_owned_cuda_installation(
        target,
        base_python=base_python,
        retired_environments=(OwnedEnvironment(retired, "c" * 64),),
    )
    selected = current if selected_environment == "current" else retired
    selected_python = selected / "Scripts" / "python.exe"
    selected_python.parent.mkdir(parents=True)
    selected_python.touch()
    ownership_path = target / ".vipp-installer" / "ownership.json"
    ownership_before = ownership_path.read_bytes()
    engine = _RecoveryEngine()
    backend = WindowsInstallerBackend(
        engine=engine,
        release=ReleaseSpec("napari-vipp", "0.13.0a7"),
        services=_services(tmp_path, gpu_ok=True),
        environ={"LOCALAPPDATA": str(tmp_path)},
        candidate_finder=lambda **_kwargs: (),
    )
    updates = []

    prepared = backend.inspect(
        InstallerSelection(existing_python=selected_python),
        progress=updates.append,
        cancellation=__import__("threading").Event(),
    )

    assert prepared.kind is TargetKind.BLOCKED
    assert prepared.target == target
    assert prepared.track is ComputeTrack.CUDA13
    assert prepared.blocked_action is BlockedAction.OPEN_INSTALLED_APPS
    assert "cannot safely update or repair in place" in prepared.plain_summary
    assert "uninstall VIPP (GPU) first" in prepared.reason
    assert engine.recovered_roots == [target]
    assert engine.prepared_plans == []
    assert "resolution" not in {update.stage for update in updates}
    assert ownership_path.read_bytes() == ownership_before


def test_existing_owned_route_handles_recovery_that_removes_incomplete_ownership(
    tmp_path,
):
    base_python = tmp_path / "Python312" / "python.exe"
    target = tmp_path / "legacy custom GPU"
    _write_owned_cuda_installation(target, base_python=base_python)
    selected_python = (
        target
        / ".vipp-installer"
        / "environments"
        / "0.13.0a6-current"
        / "Scripts"
        / "python.exe"
    )
    selected_python.parent.mkdir(parents=True)
    selected_python.touch()
    engine = _RemovingRecoveryEngine()
    backend = WindowsInstallerBackend(
        engine=engine,
        release=ReleaseSpec("napari-vipp", "0.13.0a7"),
        services=_services(tmp_path, gpu_ok=True),
        environ={"LOCALAPPDATA": str(tmp_path)},
        candidate_finder=lambda **_kwargs: (),
    )

    prepared = backend.inspect(
        InstallerSelection(existing_python=selected_python),
        progress=lambda _update: None,
        cancellation=__import__("threading").Event(),
    )

    assert prepared.kind is TargetKind.BLOCKED
    assert prepared.blocked_action is BlockedAction.USE_DEFAULT_LOCATION
    assert "no longer registered" in prepared.plain_summary
    assert engine.recovered_roots == [target]
    assert engine.prepared_plans == []


def test_unowned_existing_target_is_foreign_and_engine_is_not_called(tmp_path):
    python = tmp_path / "Python312" / "python312.exe"
    python.parent.mkdir()
    python.touch()
    target = tmp_path / "occupied"
    target.mkdir()
    (target / "student-data.txt").write_text("preserve", encoding="utf-8")
    engine = _Engine()
    backend = WindowsInstallerBackend(
        engine=engine,
        release=ReleaseSpec("napari-vipp", "0.13.0a4"),
        services=_services(tmp_path),
        environ={"LOCALAPPDATA": str(tmp_path)},
        candidate_finder=lambda **kwargs: _candidate_finder(python, **kwargs),
    )

    prepared = backend.inspect(
        InstallerSelection(
            track=TrackChoice.CPU,
            install_root=target,
        ),
        progress=lambda _event: None,
        cancellation=__import__("threading").Event(),
    )

    assert prepared.kind is TargetKind.BLOCKED
    assert prepared.blocked_action is BlockedAction.USE_DEFAULT_LOCATION
    assert "Custom managed folders are not accepted" in prepared.plain_summary
    assert engine.prepared_plans == []
    assert (target / "student-data.txt").read_text(encoding="utf-8") == "preserve"


def test_explicit_cuda_non_ascii_root_blocks_before_resolution(tmp_path):
    python = tmp_path / "Python312" / "python312.exe"
    python.parent.mkdir()
    python.touch()
    target = tmp_path / "VIPP GPU Ångström"
    engine = _Engine()
    backend = WindowsInstallerBackend(
        engine=engine,
        release=ReleaseSpec("napari-vipp", "0.13.0a7"),
        services=_services(tmp_path, gpu_ok=True),
        environ={"LOCALAPPDATA": str(tmp_path)},
        candidate_finder=lambda **kwargs: _candidate_finder(python, **kwargs),
    )
    updates = []

    prepared = backend.inspect(
        InstallerSelection(
            track=TrackChoice.CUDA13,
            install_root=target,
        ),
        progress=updates.append,
        cancellation=__import__("threading").Event(),
    )

    assert prepared.kind is TargetKind.BLOCKED
    assert "Custom managed folders are not accepted" in prepared.plain_summary
    assert "FOLDERID_LocalAppData" in prepared.technical_details
    assert prepared.blocked_action is BlockedAction.USE_DEFAULT_LOCATION
    assert engine.prepared_plans == []
    assert "resolution" not in {update.stage for update in updates}
    assert not target.exists()


def test_automatic_cuda_path_blocker_stays_visible_instead_of_falling_back(
    tmp_path,
):
    python = tmp_path / "Python312" / "python312.exe"
    python.parent.mkdir()
    python.touch()
    local_app_data = tmp_path / "Profile Ångström" / "AppData" / "Local"
    local_app_data.mkdir(parents=True)
    engine = _Engine()
    backend = WindowsInstallerBackend(
        engine=engine,
        release=ReleaseSpec("napari-vipp", "0.13.0a7"),
        services=_services(
            tmp_path,
            gpu_ok=True,
            local_app_data=local_app_data,
        ),
        environ={"LOCALAPPDATA": str(local_app_data)},
        candidate_finder=lambda **kwargs: _candidate_finder(python, **kwargs),
    )

    prepared = backend.inspect(
        InstallerSelection(),
        progress=lambda _update: None,
        cancellation=__import__("threading").Event(),
    )

    assert prepared.kind is TargetKind.BLOCKED
    assert prepared.track is ComputeTrack.CUDA13
    assert prepared.blocked_action is BlockedAction.USE_CPU
    assert "standard English letters" in prepared.plain_summary
    assert "Automatic selection used CPU" not in prepared.technical_details
    assert engine.prepared_plans == []


def test_unowned_custom_root_blocks_before_recovery_or_resolution(tmp_path):
    python = tmp_path / "Python312" / "python312.exe"
    python.parent.mkdir()
    python.touch()
    target = tmp_path / "VIPP GPU Ångström"
    engine = _RecoveryEngine()
    backend = WindowsInstallerBackend(
        engine=engine,
        release=ReleaseSpec("napari-vipp", "0.13.0a7"),
        services=_services(tmp_path, gpu_ok=True),
        environ={"LOCALAPPDATA": str(tmp_path)},
        candidate_finder=lambda **kwargs: _candidate_finder(python, **kwargs),
    )

    prepared = backend.inspect(
        InstallerSelection(track=TrackChoice.CUDA13, install_root=target),
        progress=lambda _update: None,
        cancellation=__import__("threading").Event(),
    )

    assert prepared.kind is TargetKind.BLOCKED
    assert prepared.blocked_action is BlockedAction.USE_DEFAULT_LOCATION
    assert engine.recovered_roots == []
    assert engine.prepared_plans == []
    assert not target.exists()


@pytest.mark.parametrize(
    ("installed_version", "repair"),
    [("0.13.0a6", False), ("0.13.0a7", True)],
)
def test_owned_unicode_cuda_update_or_repair_is_preserved_with_uninstall_guidance(
    tmp_path,
    installed_version,
    repair,
):
    python = tmp_path / "Python312" / "python312.exe"
    python.parent.mkdir()
    python.touch()
    target = tmp_path / "VIPP GPU Ångström"
    _write_owned_cuda_installation(
        target,
        base_python=python,
        version=installed_version,
    )
    ownership_before = (target / ".vipp-installer" / "ownership.json").read_bytes()
    engine = _Engine()
    backend = WindowsInstallerBackend(
        engine=engine,
        release=ReleaseSpec("napari-vipp", "0.13.0a7"),
        services=_services(tmp_path, gpu_ok=True),
        environ={"LOCALAPPDATA": str(tmp_path)},
        candidate_finder=lambda **kwargs: _candidate_finder(python, **kwargs),
    )

    prepared = backend.inspect(
        InstallerSelection(track=TrackChoice.CUDA13, install_root=target),
        progress=lambda _update: None,
        cancellation=__import__("threading").Event(),
        repair=repair,
    )

    assert prepared.kind is TargetKind.BLOCKED
    assert prepared.blocked_action is BlockedAction.OPEN_INSTALLED_APPS
    assert "cannot safely update or repair in place" in prepared.plain_summary
    assert "new migration selection made no change" in prepared.plain_summary
    assert "Do not move or rename" in prepared.reason
    assert "uninstall VIPP (GPU) first" in prepared.reason
    assert "run setup again" in prepared.reason
    assert "second managed copy for the same CPU/GPU option" in prepared.reason
    assert engine.prepared_plans == []
    assert (target / ".vipp-installer" / "ownership.json").read_bytes() == (
        ownership_before
    )


@pytest.mark.parametrize(
    ("track", "choice", "option"),
    [
        (ComputeTrack.CPU, TrackChoice.CPU, "VIPP (CPU)"),
        (ComputeTrack.CUDA13, TrackChoice.CUDA13, "VIPP (GPU)"),
    ],
)
def test_owned_ascii_custom_root_requires_same_track_uninstall_first(
    tmp_path,
    track,
    choice,
    option,
):
    python = tmp_path / "Python312" / "python.exe"
    target = tmp_path / f"legacy custom {track.value}"
    _write_owned_cuda_installation(
        target,
        base_python=python,
        track=track,
    )
    engine = _RecoveryEngine()
    backend = WindowsInstallerBackend(
        engine=engine,
        release=ReleaseSpec("napari-vipp", "0.13.0a7"),
        services=_services(tmp_path, gpu_ok=True),
        environ={"LOCALAPPDATA": str(tmp_path)},
        candidate_finder=lambda **_kwargs: (),
    )

    prepared = backend.inspect(
        InstallerSelection(track=choice, install_root=target),
        progress=lambda _update: None,
        cancellation=__import__("threading").Event(),
    )

    assert prepared.kind is TargetKind.BLOCKED
    assert prepared.track is track
    assert prepared.blocked_action is BlockedAction.OPEN_INSTALLED_APPS
    assert f"uninstall {option} first" in prepared.reason
    assert engine.recovered_roots == [target]
    assert engine.prepared_plans == []


@pytest.mark.parametrize("registry_values", [None, {"DisplayName": "Other app"}])
def test_owned_legacy_root_uses_hash_bound_uninstaller_when_apps_entry_is_unusable(
    tmp_path,
    registry_values,
):
    python = tmp_path / "Python312" / "python.exe"
    uninstaller = (
        tmp_path
        / "VIPP"
        / "installer"
        / "cache"
        / "0.13.0a6"
        / ("a" * 64)
        / "VIPP-Setup.exe"
    )
    target = tmp_path / "legacy custom GPU Ångström"
    _write_owned_cuda_installation(
        target,
        base_python=python,
        uninstaller_path=uninstaller,
    )
    backend = WindowsInstallerBackend(
        engine=_RecoveryEngine(),
        release=ReleaseSpec("napari-vipp", "0.13.0a7"),
        services=_services(tmp_path, gpu_ok=True),
        environ={"LOCALAPPDATA": str(tmp_path)},
        candidate_finder=lambda **_kwargs: (),
        registry_backend=_Registry(registry_values),
    )

    prepared = backend.inspect(
        InstallerSelection(track=TrackChoice.CUDA13, install_root=target),
        progress=lambda _update: None,
        cancellation=__import__("threading").Event(),
    )

    assert prepared.blocked_action is BlockedAction.RUN_OWNED_UNINSTALLER
    assert prepared.owned_uninstaller_path == uninstaller
    assert (
        prepared.owned_uninstaller_sha256
        == sha256(b"signed cached VIPP setup").hexdigest()
    )
    assert prepared.ownership_manifest_sha256
    assert "does not contain the exact ownership-bound" in prepared.reason


def test_owned_legacy_root_uses_installed_apps_when_registration_is_exact(tmp_path):
    python = tmp_path / "Python312" / "python.exe"
    uninstaller = tmp_path / "installer-cache" / "VIPP-Setup.exe"
    target = tmp_path / "legacy custom GPU Ångström"
    _write_owned_cuda_installation(
        target,
        base_python=python,
        uninstaller_path=uninstaller,
    )
    inspection = inspect_ownership(target)
    assert inspection.record is not None
    registry_plan = registry_plan_from_record(
        inspection.record,
        inspection.manifest_sha256,
    )
    backend = WindowsInstallerBackend(
        engine=_RecoveryEngine(),
        release=ReleaseSpec("napari-vipp", "0.13.0a7"),
        services=_services(tmp_path, gpu_ok=True),
        environ={"LOCALAPPDATA": str(tmp_path)},
        candidate_finder=lambda **_kwargs: (),
        registry_backend=_Registry(registry_plan.value_map),
    )

    prepared = backend.inspect(
        InstallerSelection(track=TrackChoice.CUDA13, install_root=target),
        progress=lambda _update: None,
        cancellation=__import__("threading").Event(),
    )

    assert prepared.blocked_action is BlockedAction.OPEN_INSTALLED_APPS
    assert prepared.owned_uninstaller_path is None


def test_owned_uninstaller_is_revalidated_immediately_before_launch(
    tmp_path,
    monkeypatch,
):
    python = tmp_path / "Python312" / "python.exe"
    uninstaller = (
        tmp_path
        / "VIPP"
        / "installer"
        / "cache"
        / "0.13.0a6"
        / ("a" * 64)
        / "VIPP-Setup.exe"
    )
    target = tmp_path / "legacy custom GPU Ångström"
    _write_owned_cuda_installation(
        target,
        base_python=python,
        uninstaller_path=uninstaller,
    )
    backend = WindowsInstallerBackend(
        engine=_RecoveryEngine(),
        release=ReleaseSpec("napari-vipp", "0.13.0a7"),
        services=_services(tmp_path, gpu_ok=True),
        environ={"LOCALAPPDATA": str(tmp_path)},
        candidate_finder=lambda **_kwargs: (),
        registry_backend=_Registry(None),
    )
    prepared = backend.inspect(
        InstallerSelection(track=TrackChoice.CUDA13, install_root=target),
        progress=lambda _update: None,
        cancellation=__import__("threading").Event(),
    )
    launched = []
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda argv, **kwargs: launched.append((argv, kwargs)),
    )

    backend.open_owned_uninstaller(prepared)

    assert launched[0][0] == (
        str(uninstaller),
        "--uninstall",
        "--managed-root",
        str(target),
    )
    uninstaller.write_bytes(b"changed")
    with pytest.raises(Exception, match="persistent uninstaller"):
        backend.open_owned_uninstaller(prepared)


@pytest.mark.parametrize("traversal", [False, True], ids=["outside", "dot-dot"])
def test_owned_legacy_root_never_runs_self_declared_uninstaller(
    tmp_path,
    monkeypatch,
    traversal,
):
    python = tmp_path / "Python312" / "python.exe"
    if traversal:
        cache = tmp_path / "VIPP" / "installer" / "cache"
        cache.mkdir(parents=True)
        uninstaller = (
            cache / ".." / ".." / ".." / "shared legacy folder" / "VIPP-Setup.exe"
        )
        assert ".." in uninstaller.parts
    else:
        uninstaller = tmp_path / "shared legacy folder" / "VIPP-Setup.exe"
    target = tmp_path / "legacy custom GPU Ångström"
    _write_owned_cuda_installation(
        target,
        base_python=python,
        uninstaller_path=uninstaller,
    )
    backend = WindowsInstallerBackend(
        engine=_RecoveryEngine(),
        release=ReleaseSpec("napari-vipp", "0.13.0a7"),
        services=_services(tmp_path, gpu_ok=True),
        environ={"LOCALAPPDATA": str(tmp_path)},
        candidate_finder=lambda **_kwargs: (),
        registry_backend=_Registry(None),
    )

    prepared = backend.inspect(
        InstallerSelection(track=TrackChoice.CUDA13, install_root=target),
        progress=lambda _update: None,
        cancellation=__import__("threading").Event(),
    )

    assert prepared.blocked_action is BlockedAction.OPEN_INSTALLED_APPS
    assert prepared.owned_uninstaller_path is None
    assert "outside this account's trusted VIPP installer cache" in prepared.reason
    inspection = inspect_ownership(target)
    assert inspection.record is not None
    forged = replace(
        prepared,
        blocked_action=BlockedAction.RUN_OWNED_UNINSTALLER,
        ownership_manifest_sha256=inspection.manifest_sha256,
        owned_uninstaller_path=uninstaller,
        owned_uninstaller_sha256=inspection.record.uninstaller_sha256,
    )
    launched = []
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda argv, **kwargs: launched.append((argv, kwargs)),
    )

    with pytest.raises(RuntimeError, match="outside this account's trusted"):
        backend.open_owned_uninstaller(forged)

    assert launched == []


def test_second_managed_cuda_root_is_blocked_by_old_owned_shortcut(tmp_path):
    python = tmp_path / "Python312" / "python312.exe"
    python.parent.mkdir()
    python.touch()
    services = _services(tmp_path, gpu_ok=True)
    old_target = tmp_path / "VIPP GPU Ångström"
    old_environment = (
        old_target / ".vipp-installer" / "environments" / "0.13.0a6-current"
    )
    shortcut = tmp_path / "Programs" / "VIPP" / "VIPP Automatic.lnk"
    shortcut.parent.mkdir()
    shortcut_bytes = b"old owned CUDA shortcut"
    shortcut.write_bytes(shortcut_bytes)
    _write_owned_cuda_installation(
        old_target,
        base_python=python,
        shortcuts=(
            OwnedShortcut(
                shortcut,
                sha256(shortcut_bytes).hexdigest(),
                old_environment / "Scripts" / "vipp-app.exe",
            ),
        ),
    )
    ownership_before = (old_target / ".vipp-installer" / "ownership.json").read_bytes()
    new_target = tmp_path / "VIPP GPU ASCII"
    engine = _Engine()
    backend = WindowsInstallerBackend(
        engine=engine,
        release=ReleaseSpec("napari-vipp", "0.13.0a7"),
        services=services,
        environ={"LOCALAPPDATA": str(tmp_path)},
        candidate_finder=lambda **kwargs: _candidate_finder(python, **kwargs),
    )

    prepared = backend.inspect(
        InstallerSelection(
            track=TrackChoice.CUDA13,
            install_root=new_target,
            create_desktop_shortcut=False,
        ),
        progress=lambda _update: None,
        cancellation=__import__("threading").Event(),
    )

    assert prepared.kind is TargetKind.BLOCKED
    assert prepared.blocked_action is BlockedAction.USE_DEFAULT_LOCATION
    assert "Custom managed folders are not accepted" in prepared.plain_summary
    assert engine.prepared_plans == []
    assert not new_target.exists()
    assert shortcut.read_bytes() == shortcut_bytes
    assert (old_target / ".vipp-installer" / "ownership.json").read_bytes() == (
        ownership_before
    )


def test_automatic_route_falls_back_to_cpu_and_prepares_exact_transaction(
    tmp_path,
    monkeypatch,
):
    python = tmp_path / "Python312" / "python312.exe"
    python.parent.mkdir()
    python.touch()
    engine = _Engine()
    backend = WindowsInstallerBackend(
        engine=engine,
        release=ReleaseSpec("napari-vipp", "0.13.0a4"),
        services=_services(tmp_path, gpu_ok=False),
        environ={"LOCALAPPDATA": str(tmp_path)},
        candidate_finder=lambda **kwargs: _candidate_finder(python, **kwargs),
    )
    monkeypatch.setattr(
        "napari_vipp.installer.windows_backend._engine_cancellation_token",
        lambda event: event,
    )

    prepared = backend.inspect(
        InstallerSelection(),
        progress=lambda _event: None,
        cancellation=__import__("threading").Event(),
    )

    assert prepared.kind is TargetKind.NEW
    assert prepared.track is ComputeTrack.CPU
    assert "Automatic selection used CPU" in prepared.technical_details
    assert len(engine.prepared_plans) == 1
    assert engine.prepared_plans[0][0].request.shortcut_scope is ShortcutScope.BOTH


def test_frozen_automatic_route_probes_gpu_with_selected_python(
    tmp_path,
    monkeypatch,
):
    selected_python = tmp_path / "Python312" / "python.exe"
    selected_python.parent.mkdir()
    selected_python.touch()
    setup_executable = tmp_path / "VIPP-Setup-0.13.0a4-Windows-x86_64.exe"
    calls: list[tuple[str, ...]] = []
    payload = {
        "driver_api_version": 13030,
        "devices": [
            {
                "ordinal": 0,
                "name": "NVIDIA GeForce RTX 4050 Laptop GPU",
                "compute_capability": [8, 9],
                "total_memory_bytes": 6 * 1024**3,
            }
        ],
    }

    def fake_run(argv, **_kwargs):
        calls.append(tuple(argv))
        return subprocess.CompletedProcess(argv, 0, json.dumps(payload), "")

    monkeypatch.setattr(discovery_module.sys, "platform", "win32")
    monkeypatch.setattr(discovery_module.sys, "executable", str(setup_executable))
    monkeypatch.setattr(discovery_module.sys, "frozen", True, raising=False)
    monkeypatch.setattr(discovery_module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        "napari_vipp.installer.windows_backend._engine_cancellation_token",
        lambda event: event,
    )
    services = replace(
        _services(tmp_path),
        nvidia_probe=discovery_module._probe_nvidia_driver,
    )
    engine = _Engine()
    backend = WindowsInstallerBackend(
        engine=engine,
        release=ReleaseSpec("napari-vipp", "0.13.0a4"),
        services=services,
        environ={"LOCALAPPDATA": str(tmp_path)},
        candidate_finder=lambda **kwargs: _candidate_finder(
            selected_python,
            **kwargs,
        ),
    )

    prepared = backend.inspect(
        InstallerSelection(),
        progress=lambda _event: None,
        cancellation=__import__("threading").Event(),
    )

    assert prepared.kind is TargetKind.NEW
    assert prepared.track is ComputeTrack.CUDA13
    assert len(engine.prepared_plans) == 1
    plan = engine.prepared_plans[0][0]
    assert plan.discovery.nvidia is not None
    assert plan.discovery.nvidia.devices[0].name.endswith("RTX 4050 Laptop GPU")
    assert calls == [
        (
            str(selected_python.resolve()),
            "-I",
            "-S",
            "-B",
            "-c",
            discovery_module._CUDA_DRIVER_PROBE,
        )
    ]
    assert str(setup_executable) not in calls[0]


def test_disabling_desktop_shortcut_keeps_start_menu_launcher(
    tmp_path,
    monkeypatch,
):
    python = tmp_path / "Python312" / "python312.exe"
    python.parent.mkdir()
    python.touch()
    engine = _Engine()
    backend = WindowsInstallerBackend(
        engine=engine,
        release=ReleaseSpec("napari-vipp", "0.13.0a4"),
        services=_services(tmp_path),
        environ={"LOCALAPPDATA": str(tmp_path)},
        candidate_finder=lambda **kwargs: _candidate_finder(python, **kwargs),
    )
    monkeypatch.setattr(
        "napari_vipp.installer.windows_backend._engine_cancellation_token",
        lambda event: event,
    )

    backend.inspect(
        InstallerSelection(
            track=TrackChoice.CPU,
            create_desktop_shortcut=False,
        ),
        progress=lambda _event: None,
        cancellation=__import__("threading").Event(),
    )

    assert len(engine.prepared_plans) == 1
    assert (
        engine.prepared_plans[0][0].request.shortcut_scope is ShortcutScope.START_MENU
    )


def test_apply_passes_explicit_confirmation_to_engine(tmp_path, monkeypatch):
    engine = _Engine()
    transaction = SimpleNamespace(name="transaction")
    backend = WindowsInstallerBackend(
        engine=engine,
        release=ReleaseSpec("napari-vipp", "0.13.0a4"),
        services=_services(tmp_path),
    )
    monkeypatch.setattr(
        "napari_vipp.installer.windows_backend._engine_cancellation_token",
        lambda event: event,
    )
    prepared = SimpleNamespace(kind=TargetKind.NEW, payload=transaction)

    outcome = backend.apply(
        prepared,
        confirmed=True,
        progress=lambda _event: None,
        cancellation=__import__("threading").Event(),
    )

    assert engine.authorizations == [(transaction, True)]
    assert engine.applied == [(transaction, "authorization")]
    assert outcome.launcher.name == "vipp-app.exe"


def test_late_cancel_click_does_not_override_committed_engine_result(
    tmp_path,
    monkeypatch,
):
    cancellation = __import__("threading").Event()

    class _LateCancelEngine(_Engine):
        def apply(self, prepared, authorization, *, progress, cancellation=None):
            result = super().apply(
                prepared,
                authorization,
                progress=progress,
                cancellation=cancellation,
            )
            cancellation.set()
            return result

    engine = _LateCancelEngine()
    backend = WindowsInstallerBackend(
        engine=engine,
        release=ReleaseSpec("napari-vipp", "0.13.0a4"),
        services=_services(tmp_path),
    )
    monkeypatch.setattr(
        "napari_vipp.installer.windows_backend._engine_cancellation_token",
        lambda event: event,
    )

    outcome = backend.apply(
        SimpleNamespace(kind=TargetKind.NEW, payload=SimpleNamespace()),
        confirmed=True,
        progress=lambda _event: None,
        cancellation=cancellation,
    )

    assert cancellation.is_set()
    assert outcome.launcher.name == "vipp-app.exe"


def test_committed_registration_warning_is_visible_in_novice_outcome(
    tmp_path,
    monkeypatch,
):
    class _WarningEngine(_Engine):
        def apply(self, prepared, authorization, *, progress, cancellation):
            self.applied.append((prepared, authorization))
            return SimpleNamespace(
                status=SimpleNamespace(value="succeeded"),
                launcher_path=Path("C:/VIPP/current/Scripts/vipp-app.exe"),
                message=(
                    "VIPP is ready, but Windows could not finish all repair and "
                    "removal details. Run VIPP Setup again to finish."
                ),
                registration_warning="simulated registry access failure",
                as_dict=lambda: {
                    "status": "succeeded",
                    "registration_warning": "simulated registry access failure",
                },
            )

    backend = WindowsInstallerBackend(
        engine=_WarningEngine(),
        release=ReleaseSpec("napari-vipp", "0.13.0a4"),
        services=_services(tmp_path),
    )
    monkeypatch.setattr(
        "napari_vipp.installer.windows_backend._engine_cancellation_token",
        lambda event: event,
    )

    outcome = backend.apply(
        SimpleNamespace(kind=TargetKind.REPAIR, payload=SimpleNamespace()),
        confirmed=True,
        progress=lambda _event: None,
        cancellation=__import__("threading").Event(),
    )

    assert "Run VIPP Setup again" in outcome.message
    assert outcome.message.casefold().count("run vipp setup again") == 1
    assert (
        "VIPP is ready, but Windows could not finish its Repair" not in outcome.message
    )
    assert "registration_warning" in outcome.technical_details


def test_open_vipp_uses_documents_not_the_replaceable_environment(
    tmp_path,
    monkeypatch,
):
    launcher = tmp_path / "managed" / ".vipp" / "environments" / "current"
    launcher = launcher / "Scripts" / "vipp-cpu.exe"
    launcher.parent.mkdir(parents=True)
    launcher.touch()
    calls = []
    monkeypatch.setattr(
        "napari_vipp.installer.windows_backend.subprocess.Popen",
        lambda argv, **kwargs: calls.append((argv, kwargs)),
    )
    backend = WindowsInstallerBackend(services=_services(tmp_path))

    backend.open_vipp(launcher)

    assert len(calls) == 1
    assert Path(calls[0][1]["cwd"]) == tmp_path / "Documents"
    assert Path(calls[0][1]["cwd"]) != launcher.parent.parent
    if os.name == "nt":
        assert calls[0][1]["creationflags"] & getattr(
            subprocess,
            "CREATE_NO_WINDOW",
            0x08000000,
        )
    else:
        assert calls[0][1]["creationflags"] == 0


def test_prepare_reports_plain_language_discovery_and_decision_milestones(
    tmp_path,
    monkeypatch,
):
    python = tmp_path / "Python312" / "python312.exe"
    python.parent.mkdir()
    python.touch()
    backend = WindowsInstallerBackend(
        engine=_Engine(),
        release=ReleaseSpec("napari-vipp", "0.13.0a4"),
        services=_services(tmp_path, gpu_ok=False),
        environ={"LOCALAPPDATA": str(tmp_path)},
        candidate_finder=lambda **kwargs: _candidate_finder(python, **kwargs),
    )
    monkeypatch.setattr(
        "napari_vipp.installer.windows_backend._engine_cancellation_token",
        lambda event: event,
    )
    updates = []

    prepared = backend.inspect(
        InstallerSelection(),
        progress=updates.append,
        cancellation=__import__("threading").Event(),
    )

    assert prepared.kind is TargetKind.NEW
    messages = [update.message for update in updates]
    assert "Found 64-bit Python 3.12.10." in messages
    assert any("CPU setup is recommended" in message for message in messages)
    assert any(
        "Reviewing exact packages from PyPI" in message and "several minutes" in message
        for message in messages
    )
    assert messages[-1] == "Checks finished. Setup recommends installing VIPP."


def test_prepare_foreign_race_keeps_fixed_root_guidance(tmp_path, monkeypatch):
    python = tmp_path / "Python312" / "python312.exe"
    python.parent.mkdir()
    python.touch()
    backend = WindowsInstallerBackend(
        engine=_Engine(kind="foreign"),
        release=ReleaseSpec("napari-vipp", "0.13.0a7"),
        services=_services(tmp_path, gpu_ok=False),
        environ={"LOCALAPPDATA": str(tmp_path)},
        candidate_finder=lambda **kwargs: _candidate_finder(python, **kwargs),
    )
    monkeypatch.setattr(
        "napari_vipp.installer.windows_backend._engine_cancellation_token",
        lambda event: event,
    )
    updates = []

    prepared = backend.inspect(
        InstallerSelection(track=TrackChoice.CPU),
        progress=updates.append,
        cancellation=__import__("threading").Event(),
    )

    assert prepared.kind is TargetKind.FOREIGN
    assert updates[-1].message == (
        "Checks finished. Move the unexpected files yourself if appropriate, "
        "then choose Check again."
    )
    assert "another installation folder" not in updates[-1].message


def test_prepare_names_eligible_nvidia_device_in_progress(tmp_path, monkeypatch):
    python = tmp_path / "Python312" / "python312.exe"
    python.parent.mkdir()
    python.touch()
    backend = WindowsInstallerBackend(
        engine=_Engine(),
        release=ReleaseSpec("napari-vipp", "0.13.0a4"),
        services=_services(tmp_path, gpu_ok=True),
        environ={"LOCALAPPDATA": str(tmp_path)},
        candidate_finder=lambda **kwargs: _candidate_finder(python, **kwargs),
    )
    monkeypatch.setattr(
        "napari_vipp.installer.windows_backend._engine_cancellation_token",
        lambda event: event,
    )
    updates = []

    prepared = backend.inspect(
        InstallerSelection(),
        progress=updates.append,
        cancellation=__import__("threading").Event(),
    )

    assert prepared.track is ComputeTrack.CUDA13
    assert any(
        update.message
        == "NVIDIA GeForce RTX 4050 Laptop GPU is eligible for the CUDA option."
        for update in updates
    )
