from __future__ import annotations

import hashlib
import json
import threading
import zipfile
from dataclasses import replace
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace

import pytest

from napari_vipp import gpu_setup
from napari_vipp._tests.test_installer_windows_backend import (
    _RecoveryEngine,
    _services,
    _write_owned_cuda_installation,
)
from napari_vipp.installer import discovery as discovery_module
from napari_vipp.installer.discovery import discover_installation
from napari_vipp.installer.engine import (
    ManagedInstallerEngine,
    PreparationError,
    _transaction_journal_path,
    validate_build_target_identity,
)
from napari_vipp.installer.frontend import (
    InstallationCancelled,
    InstallerSelection,
    PreparedInstall,
    TargetKind,
    TrackChoice,
)
from napari_vipp.installer.models import (
    ComputeTrack,
    InstallMode,
    InstallRequest,
    ReleaseSpec,
    ShortcutScope,
    managed_environment_name,
)
from napari_vipp.installer.ownership import (
    OwnedPackage,
    inspect_ownership,
    write_ownership_record,
)
from napari_vipp.installer.planner import create_install_plan
from napari_vipp.installer.python_discovery import PythonCandidate
from napari_vipp.installer.wheel_cache import cache_release_wheel, wheel_cache_path
from napari_vipp.installer.windows_backend import WindowsInstallerBackend


class _BuildRecoveryEngine(_RecoveryEngine):
    def recover_interrupted(self, root, **_kwargs):
        return super().recover_interrupted(root)


@pytest.fixture(autouse=True)
def _isolate_host(monkeypatch, tmp_path):
    monkeypatch.setattr(discovery_module.sys, "platform", "win32")
    monkeypatch.setattr(discovery_module.platform, "system", lambda: "Windows")
    monkeypatch.setattr(discovery_module.platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(gpu_setup, "_windows_known_folder", lambda _name: tmp_path)
    monkeypatch.setattr(gpu_setup.psutil, "process_iter", lambda _attrs: ())


def _wheel(tmp_path, *, name="napari-vipp", version="0.16.0a3"):
    path = tmp_path / "bundle with spaces" / "napari_vipp-0.16.0a3-py3-none-any.whl"
    path.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(path, "w") as stream:
        stream.writestr(
            "napari_vipp-0.16.0a3.dist-info/METADATA",
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    record = {"url": path.as_uri(), "archive_info": {"hashes": {"sha256": digest}}}
    distribution = SimpleNamespace(
        version="0.16.0a3", read_text=lambda _name: json.dumps(record)
    )
    release = ReleaseSpec(
        "napari-vipp", "0.16.0a3", wheel_path=path, wheel_sha256=digest
    )
    return release, record, distribution


def test_exact_local_wheel_is_verified_without_index_lookup(tmp_path):
    release, _record, distribution = _wheel(tmp_path)
    assert gpu_setup._release_from_distribution(distribution) == release


@pytest.mark.parametrize(
    "record", [None, {}, {"dir_info": {"editable": True}}, {"vcs_info": {}}]
)
def test_unrecoverable_install_identity_is_not_replaced_by_same_version(record):
    distribution = SimpleNamespace(
        version="0.16.0a3",
        read_text=lambda _name: None if record is None else json.dumps(record),
    )
    with pytest.raises(gpu_setup.GpuSetupError):
        gpu_setup._release_from_distribution(distribution)


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/vipp.whl",
        "file://server/share/vipp.whl",
        "file:///tmp/vipp.whl?x=y",
    ],
)
def test_nonlocal_or_ambiguous_wheel_source_is_rejected(tmp_path, url):
    _release, record, distribution = _wheel(tmp_path)
    record["url"] = url
    with pytest.raises(gpu_setup.GpuSetupError, match="hash-verified local"):
        gpu_setup._release_from_distribution(distribution)


def test_changed_wheel_does_not_install_same_version(tmp_path):
    release, _record, distribution = _wheel(tmp_path)
    release.wheel_path.write_bytes(b"changed build")
    with pytest.raises(gpu_setup.GpuSetupError, match="changed since"):
        gpu_setup._release_from_distribution(distribution)


@pytest.mark.parametrize("change", [{"name": "other"}, {"version": "0.16.0a4"}])
def test_hash_alone_is_insufficient_without_wheel_identity(tmp_path, change):
    _release, _record, distribution = _wheel(tmp_path, **change)
    with pytest.raises(gpu_setup.GpuSetupError, match="does not (contain|match)"):
        gpu_setup._release_from_distribution(distribution)


@pytest.mark.parametrize("missing", [True, False])
def test_retained_wheel_survives_bundle_removal_or_rebuild(tmp_path, missing):
    release, _record, distribution = _wheel(tmp_path)
    cached = cache_release_wheel(release, tmp_path)
    if missing:
        release.wheel_path.unlink()
    else:
        release.wheel_path.write_bytes(b"later bundle build")
    resolved = gpu_setup._release_from_distribution(distribution)
    assert resolved == cached
    assert resolved.wheel_path.is_relative_to(
        tmp_path / "VIPP" / "installer" / "wheels"
    )


def test_bad_existing_cache_is_preserved_and_rejected(tmp_path):
    release, _record, _distribution = _wheel(tmp_path)
    target = wheel_cache_path(tmp_path, release.wheel_sha256, release.wheel_path.name)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"preserve unexpected content")
    with pytest.raises(RuntimeError, match="integrity"):
        cache_release_wheel(release, tmp_path)
    assert target.read_bytes() == b"preserve unexpected content"


def _backend(tmp_path, release, engine):
    python = tmp_path / "Python312" / "python.exe"
    python.parent.mkdir(exist_ok=True)
    python.touch()
    return gpu_setup.GpuSetupBackend(
        release=release,
        engine=engine,
        services=_services(tmp_path, gpu_ok=True),
        candidate_finder=lambda **_kwargs: (
            PythonCandidate(python, (3, 12, 10), "test"),
        ),
    )


def test_build_specific_gpu_setup_preserves_newer_standard_install_and_shortcuts(
    tmp_path,
):
    release, _record, _distribution = _wheel(tmp_path)
    standard = tmp_path / "VIPP" / "environments" / "cuda13"
    _write_owned_cuda_installation(
        standard, base_python=tmp_path / "Python312" / "python.exe", version="0.16.0a4"
    )
    ownership_file = standard / ".vipp-installer" / "ownership.json"
    before = ownership_file.read_bytes()
    engine = _BuildRecoveryEngine()
    backend = _backend(tmp_path, release, engine)
    prepared = backend.inspect(
        InstallerSelection(track=TrackChoice.CUDA13),
        progress=lambda _event: None,
        cancellation=threading.Event(),
    )
    target = standard.with_name(
        managed_environment_name(ComputeTrack.CUDA13, release.wheel_sha256)
    )
    assert prepared.target == target
    assert engine.recovered_roots == [target]
    plan, repair = engine.prepared_plans[0]
    assert not repair
    assert plan.request.managed_build_sha256 == release.wheel_sha256
    assert plan.discovery.filesystem.canonical_managed_root == target
    assert plan.request.shortcut_scope is ShortcutScope.NONE
    assert plan.shortcuts == ()
    assert plan.release.wheel_path != release.wheel_path
    assert plan.release.wheel_sha256 == release.wheel_sha256
    assert ownership_file.read_bytes() == before


@pytest.mark.parametrize(
    "selection",
    [
        InstallerSelection(),
        InstallerSelection(track=TrackChoice.CPU),
        InstallerSelection(track=TrackChoice.CUDA13, existing_python=Path("elsewhere")),
    ],
)
def test_gpu_adapter_cannot_silently_fall_back_or_target_existing_environment(
    tmp_path, selection
):
    release, _record, _distribution = _wheel(tmp_path)
    engine = _BuildRecoveryEngine()
    backend = _backend(tmp_path, release, engine)
    with pytest.raises(gpu_setup.GpuSetupError, match="separate managed"):
        backend.inspect(
            selection, progress=lambda _event: None, cancellation=threading.Event()
        )
    assert not engine.recovered_roots


def test_running_gpu_environment_is_rejected_before_recovery(tmp_path, monkeypatch):
    release, _record, _distribution = _wheel(tmp_path)
    engine = _BuildRecoveryEngine()
    backend = _backend(tmp_path, release, engine)
    target = backend._managed_install_root(ComputeTrack.CUDA13)
    monkeypatch.setattr(
        gpu_setup.sys, "prefix", str(target / ".vipp-installer" / "running")
    )
    with pytest.raises(gpu_setup.GpuSetupError, match="environment is running"):
        backend.inspect(
            InstallerSelection(track=TrackChoice.CUDA13),
            progress=lambda _event: None,
            cancellation=threading.Event(),
        )
    assert not engine.recovered_roots


@pytest.mark.parametrize("healthy", [True, False])
def test_current_exact_build_is_checked_and_only_failed_probe_prepares_repair(
    tmp_path, monkeypatch, healthy
):
    release, _record, _distribution = _wheel(tmp_path)
    backend = _backend(tmp_path, release, _BuildRecoveryEngine())
    target = backend._managed_install_root(ComputeTrack.CUDA13)
    calls = []

    def inspect(_self, selection, *, repair, **_kwargs):
        calls.append(repair)
        return PreparedInstall(
            kind=TargetKind.REPAIR if repair else TargetKind.CURRENT,
            target=target,
            release_version=release.version,
            track=ComputeTrack.CUDA13,
            plain_summary="checked",
            technical_details="",
            payload=object() if repair else None,
        )

    monkeypatch.setattr(WindowsInstallerBackend, "inspect", inspect)
    monkeypatch.setattr(
        gpu_setup, "_installed_runtime", lambda *_args: target / "python.exe"
    )
    monkeypatch.setattr(gpu_setup, "_gpu_check", lambda *_args: healthy)
    prepared = backend.inspect(
        InstallerSelection(track=TrackChoice.CUDA13),
        progress=lambda _event: None,
        cancellation=threading.Event(),
    )
    assert calls == ([False] if healthy else [False, True])
    assert prepared.kind is (TargetKind.CURRENT if healthy else TargetKind.REPAIR)


def test_request_binds_canonical_location_to_actual_artifact(tmp_path):
    release, _record, _distribution = _wheel(tmp_path)
    request = InstallRequest(
        InstallMode.MANAGED,
        ComputeTrack.CUDA13,
        tmp_path / "python.exe",
        shortcut_scope=ShortcutScope.NONE,
        managed_build_sha256=release.wheel_sha256,
    )
    services = _services(tmp_path, gpu_ok=True)
    discovery = discover_installation(request, services=services)
    assert discovery.filesystem.canonical_managed_root.name == managed_environment_name(
        ComputeTrack.CUDA13, release.wheel_sha256
    )
    mismatch = replace(release, wheel_sha256="a" * 64)
    plan = create_install_plan(request, discovery=discovery, release=mismatch)
    assert any(issue.code == "managed_build_identity_mismatch" for issue in plan.issues)
    with pytest.raises(PreparationError, match="exact VIPP wheel"):
        ManagedInstallerEngine._validate_managed_plan(plan)
    arbitrary = replace(request, install_root=tmp_path / "arbitrary")
    plan = create_install_plan(
        arbitrary,
        discovery=discover_installation(arbitrary, services=services),
        release=release,
    )
    assert any(issue.code == "managed_root_not_canonical" for issue in plan.issues)


@pytest.mark.parametrize("digest", ["../bad", "0" * 63, "g" * 64])
def test_invalid_build_identity_cannot_construct_path(digest):
    with pytest.raises(ValueError, match="managed_build_sha256"):
        InstallRequest(
            InstallMode.MANAGED,
            ComputeTrack.CUDA13,
            Path("python.exe"),
            managed_build_sha256=digest,
        )


def test_launch_callback_receives_the_verified_launcher(tmp_path):
    release, _record, _distribution = _wheel(tmp_path)
    opened = []
    backend = gpu_setup.GpuSetupBackend(
        release=release, open_gpu_callback=opened.append, services=_services(tmp_path)
    )
    launcher = tmp_path / "checked" / "vipp-app.exe"
    backend.open_vipp(launcher)
    assert opened == [launcher]


def test_cancellation_during_gpu_check_is_a_cancelled_state(tmp_path, monkeypatch):
    release, _record, _distribution = _wheel(tmp_path)
    backend = _backend(tmp_path, release, _BuildRecoveryEngine())
    target = backend._managed_install_root(ComputeTrack.CUDA13)
    monkeypatch.setattr(
        WindowsInstallerBackend,
        "inspect",
        lambda *_args, **_kwargs: PreparedInstall(
            kind=TargetKind.CURRENT,
            target=target,
            release_version=release.version,
            track=ComputeTrack.CUDA13,
            plain_summary="checked",
            technical_details="",
        ),
    )
    monkeypatch.setattr(
        gpu_setup, "_installed_runtime", lambda *_args: target / "python.exe"
    )

    def cancel(*_args):
        raise gpu_setup.InstallCancelled("cancelled subprocess")

    monkeypatch.setattr(gpu_setup, "_gpu_check", cancel)
    with pytest.raises(InstallationCancelled):
        backend.inspect(
            InstallerSelection(track=TrackChoice.CUDA13),
            progress=lambda _event: None,
            cancellation=threading.Event(),
        )


def test_shortcut_runtime_lookup_requires_exact_build_and_valid_candidate_marker(
    tmp_path,
):
    digest = "b" * 64
    target = (
        tmp_path
        / "VIPP"
        / "environments"
        / managed_environment_name(ComputeTrack.CUDA13, digest)
    )
    _write_owned_cuda_installation(target, base_python=tmp_path / "python.exe")
    record = inspect_ownership(target).record
    assert record is not None
    environment = record.environment_root
    scripts = environment / "Scripts"
    scripts.mkdir(parents=True)
    (scripts / "python.exe").touch()
    (scripts / "vipp-app.exe").touch()
    (environment / "pyvenv.cfg").touch()
    marker = environment / ".vipp-install-candidate.json"
    marker.write_bytes(b"accepted candidate marker")
    record = replace(
        record,
        packages=(OwnedPackage("napari-vipp", record.version, digest),),
        environment_marker_sha256=hashlib.sha256(marker.read_bytes()).hexdigest(),
    )
    write_ownership_record(target, record)
    assert gpu_setup.find_installed_gpu_runtime(digest) == scripts / "python.exe"
    assert gpu_setup.find_installed_gpu_runtime("a" * 64) is None
    marker.write_bytes(b"different candidate")
    assert gpu_setup.find_installed_gpu_runtime(digest) is None


def test_shortened_locator_retains_full_identity_and_native_dll_path_budget():
    digest = "a" * 64
    root = PureWindowsPath(
        "C:/Users/rensu/AppData/Local/VIPP/environments"
    ) / managed_environment_name(ComputeTrack.CUDA13, digest)
    environment = (
        root
        / ".vipp-installer/environments/0.16.0a3-12345678-1234-4234-9234-123456789abc"
    )
    native = environment / "Lib/site-packages/nvidia/cu13/bin/x86_64/cublasLt64_13.dll"
    assert len(str(environment)) == 153
    assert len(str(native)) < 260
    request = InstallRequest(
        InstallMode.MANAGED,
        ComputeTrack.CUDA13,
        Path("python.exe"),
        managed_build_sha256=digest,
    )
    assert request.as_dict()["managed_build_sha256"] == digest


def test_prefix_collision_is_rejected_before_recovery_or_cache_write(tmp_path):
    release, _record, _distribution = _wheel(tmp_path)
    engine = _BuildRecoveryEngine()
    backend = _backend(tmp_path, release, engine)
    target = backend._managed_install_root(ComputeTrack.CUDA13)
    other_digest = release.wheel_sha256[:24] + (
        "0" * 40 if release.wheel_sha256[24:] != "0" * 40 else "1" * 40
    )
    assert managed_environment_name(ComputeTrack.CUDA13, other_digest) == target.name
    _write_owned_cuda_installation(target, base_python=tmp_path / "python.exe")
    record = inspect_ownership(target).record
    write_ownership_record(
        target,
        replace(
            record,
            packages=(OwnedPackage("napari-vipp", record.version, other_digest),),
        ),
    )
    ownership_file = target / ".vipp-installer/ownership.json"
    before = ownership_file.read_bytes()
    with pytest.raises(PreparationError, match="full wheel identity differs"):
        backend.inspect(
            InstallerSelection(track=TrackChoice.CUDA13),
            progress=lambda _event: None,
            cancellation=threading.Event(),
        )
    assert engine.recovered_roots == []
    assert ownership_file.read_bytes() == before
    assert not (tmp_path / "VIPP/installer/wheels").exists()
    assert gpu_setup.find_installed_gpu_runtime(release.wheel_sha256) is None


@pytest.mark.parametrize("recorded_digest", [None, "a" * 24 + "b" * 40])
def test_unfinished_collision_never_reaches_recovery(tmp_path, recorded_digest):
    digest = "a" * 64
    root = tmp_path / managed_environment_name(ComputeTrack.CUDA13, digest)
    state = tmp_path / "state"
    journal = _transaction_journal_path(state, root)
    journal.parent.mkdir(parents=True)
    journal.write_text(
        json.dumps(
            {
                "schema": "napari-vipp-install-transaction",
                "schema_version": 1,
                "wheel_sha256": recorded_digest,
                "managed_root": str(root),
            }
        ),
        encoding="utf-8",
    )
    before = journal.read_bytes()
    engine = ManagedInstallerEngine(
        state_root=state,
        known_folder_probe=lambda _name: None,
    )
    with pytest.raises(PreparationError, match="unfinished GPU setup"):
        engine.recover_interrupted(root, expected_wheel_sha256=digest)
    assert journal.read_bytes() == before
    assert not root.exists()
    assert not (state / "locks").exists()


def test_matching_full_identity_allows_pending_journal_check(tmp_path):
    digest = "a" * 64
    root, state = tmp_path / "target", tmp_path / "state"
    journal = _transaction_journal_path(state, root)
    journal.parent.mkdir(parents=True)
    journal.write_text(
        json.dumps(
            {
                "schema": "napari-vipp-install-transaction",
                "schema_version": 1,
                "wheel_sha256": digest,
                "managed_root": str(root),
            }
        ),
        encoding="utf-8",
    )
    validate_build_target_identity(root, digest, state_root=state)


def test_real_managed_current_without_shortcuts_retains_ownership_launcher(
    tmp_path, monkeypatch
):
    """Exercise the real CURRENT engine result that formerly lost its launcher."""
    release, _record, _distribution = _wheel(tmp_path)
    state = tmp_path / "VIPP/installer"
    engine = ManagedInstallerEngine(
        state_root=state, known_folder_probe=lambda _name: None
    )
    backend = _backend(tmp_path, release, engine)
    root = backend._managed_install_root(ComputeTrack.CUDA13)
    _write_owned_cuda_installation(
        root, base_python=tmp_path / "Python312/python.exe", version=release.version
    )
    record = inspect_ownership(root).record
    assert record is not None
    scripts = record.environment_root / "Scripts"
    scripts.mkdir(parents=True)
    (scripts / "python.exe").touch()
    launcher = scripts / "vipp-app.exe"
    launcher.touch()
    (record.environment_root / "pyvenv.cfg").touch()
    marker = record.environment_root / ".vipp-install-candidate.json"
    marker.write_bytes(b"accepted GPU environment")
    write_ownership_record(
        root,
        replace(
            record,
            packages=(
                OwnedPackage("napari-vipp", release.version, release.wheel_sha256),
            ),
            environment_marker_sha256=hashlib.sha256(marker.read_bytes()).hexdigest(),
        ),
    )
    checked = []
    monkeypatch.setattr(
        gpu_setup, "_gpu_check", lambda python, _cancel: checked.append(python) or True
    )
    prepared = backend.inspect(
        InstallerSelection(track=TrackChoice.CUDA13),
        progress=lambda _event: None,
        cancellation=threading.Event(),
    )
    assert prepared.kind is TargetKind.CURRENT
    assert prepared.launcher == launcher
    assert checked == [scripts / "python.exe"]
    assert prepared.launcher != root / "Scripts/vipp-app.exe"
    from napari_vipp.installer.frontend import _view_for_prepared

    assert _view_for_prepared(prepared).primary_enabled
