from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from napari_vipp.installer import discovery as discovery_module
from napari_vipp.installer.discovery import (
    DiscoveryServices,
    InterpreterProbe,
)
from napari_vipp.installer.frontend import (
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
from napari_vipp.installer.python_discovery import PythonCandidate
from napari_vipp.installer.windows_backend import WindowsInstallerBackend


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


def _services(tmp_path, *, gpu_ok=False):
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


def test_existing_napari_route_is_never_silently_mutated(tmp_path):
    python = tmp_path / "napari-env" / "Scripts" / "python.exe"
    backend = WindowsInstallerBackend(
        engine=_Engine(),
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
    assert "keeps existing napari environments unchanged" in prepared.plain_summary


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

    assert prepared.kind is TargetKind.FOREIGN
    assert "will not replace or remove" in prepared.plain_summary
    assert engine.prepared_plans == []
    assert (target / "student-data.txt").read_text(encoding="utf-8") == "preserve"


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
    assert calls[0][1]["creationflags"] & getattr(
        subprocess,
        "CREATE_NO_WINDOW",
        0x08000000,
    )


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
