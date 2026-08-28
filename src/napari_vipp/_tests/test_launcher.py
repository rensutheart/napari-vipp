from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
from qtpy.QtWidgets import QApplication

from napari_vipp import app, launcher
from napari_vipp.launcher import LauncherController, build_child_command
from napari_vipp.startup import (
    LaunchProfile,
    StartupChannel,
    StartupPhase,
    StatusEmitter,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_packaged_commands_expose_console_and_clickable_profile_launchers():
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]

    assert project["scripts"]["vipp"] == "napari_vipp.__main__:main"
    assert (
        project["scripts"]["vipp-install-plan"]
        == "napari_vipp.installer.cli:main"
    )
    assert project["gui-scripts"] == {
        "vipp-app": "napari_vipp.launcher:main_auto",
        "vipp-cpu": "napari_vipp.launcher:main_cpu",
        "vipp-prefer-gpu": "napari_vipp.launcher:main_prefer_gpu",
    }


@pytest.mark.parametrize(
    ("facade_name", "conflicting_profile", "expected_profile"),
    [
        ("main_auto", "prefer_gpu", "auto"),
        ("main_cpu", "prefer_gpu", "cpu"),
        ("main_prefer_gpu", "cpu", "prefer_gpu"),
    ],
)
def test_graphical_profile_commands_cannot_be_overridden(
    monkeypatch,
    facade_name,
    conflicting_profile,
    expected_profile,
):
    captured = []
    monkeypatch.setattr(
        sys,
        "argv",
        [facade_name, "--profile", conflicting_profile, "--no-splash"],
    )
    monkeypatch.setattr(
        launcher,
        "main",
        lambda arguments: captured.extend(arguments) or 0,
    )

    assert getattr(launcher, facade_name)() == 0
    parsed = launcher.build_parser().parse_args(captured)
    assert parsed.profile == expected_profile
    assert parsed.no_splash is True


def test_console_reporter_renders_default_milestone_messages(capsys):
    reporter = app._ConsoleReporter()
    reporter.progress("loading_napari")
    assert "Loading napari and its plugins" in capsys.readouterr().err


def test_launcher_boundary_reports_startup_window_failure(monkeypatch, capsys):
    monkeypatch.setattr(
        launcher,
        "run_splash",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("simulated launcher failure")
        ),
    )
    assert launcher.main([]) == 2
    error = capsys.readouterr().err
    assert "OSError: simulated launcher failure" in error
    assert "Traceback" not in error


def test_launcher_module_import_does_not_load_application_or_scientific_stack():
    source_root = str(Path(__file__).resolve().parents[2])
    environment = os.environ.copy()
    current_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_root
        if not current_pythonpath
        else os.pathsep.join((source_root, current_pythonpath))
    )
    code = """
import json
import sys
import napari_vipp.launcher
forbidden = ("napari", "numpy", "cupy", "cucim", "napari_vipp._widget")
print(json.dumps(sorted(name for name in forbidden if name in sys.modules)))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "[]"


def test_child_command_is_shell_free_and_preserves_space_containing_paths(tmp_path):
    parent = tmp_path / "private channel with spaces"
    parent.mkdir()
    channel = StartupChannel.create(parent=parent)
    executable = tmp_path / "Python Folder" / "python executable"
    command = build_child_command(
        executable=executable,
        profile="prefer-gpu",
        channel=channel,
    )
    assert command == [
        str(executable),
        "-m",
        "napari_vipp.app",
        "--profile",
        "prefer_gpu",
        "--startup-channel",
        str(channel.path),
        "--startup-token",
        channel.token,
    ]
    channel.cleanup()


class _FakeNode:
    def __init__(self) -> None:
        self.params: dict[str, str] = {}


class _FakeGraphView:
    def __init__(self) -> None:
        self.selected: list[str] = []

    def select_node(self, node_id: str) -> None:
        self.selected.append(node_id)


class _FakeVippWidget:
    instances: list[_FakeVippWidget] = []

    def __init__(self, viewer, parent=None, **kwargs) -> None:
        self.viewer = viewer
        self.parent = parent
        self.kwargs = kwargs
        self.pipeline = SimpleNamespace(nodes={"input": _FakeNode()})
        self.graph_view = _FakeGraphView()
        self.initial_run_count = 0
        self.instances.append(self)

    def run_initial_pipeline_once(self) -> bool:
        self.initial_run_count += 1
        return self.initial_run_count == 1


def test_standalone_app_defers_configures_and_runs_initial_workflow_once():
    _FakeVippWidget.instances.clear()
    viewer = object()
    widget = app._construct_vipp_widget(
        _FakeVippWidget,
        viewer,
        LaunchProfile.CPU,
    )
    app._configure_initial_workflow(widget)

    assert widget.kwargs == {
        "defer_initial_run": True,
        "initial_compute_mode": "cpu",
    }
    assert widget.pipeline.nodes["input"].params == {
        "source_mode": "sample",
        "sample_name": "VIPP synthetic volume",
    }
    assert widget.initial_run_count == 1
    assert widget.graph_view.selected == ["input"]


def test_application_module_is_also_safe_to_import_before_child_start():
    assert "napari" not in app.__dict__
    assert "VippWidget" not in app.__dict__


class _WorkflowLoader:
    def __init__(self) -> None:
        self.example_ids: list[str] = []
        self.workflow_paths: list[Path] = []

    def load_example_workflow(self, example_id: str) -> Path:
        self.example_ids.append(example_id)
        return Path(f"template-{example_id}.json")

    def load_workflow_file(self, path: Path) -> Path:
        self.workflow_paths.append(path)
        return path


def test_example_launcher_keeps_bundled_workflows_as_templates(tmp_path):
    from napari_vipp.ui.examples import (
        EXAMPLE_WORKFLOWS,
        _example_workflow_path,
    )
    from scripts.launch_vipp_intensity_workflow import _load_selected_workflow

    loader = _WorkflowLoader()
    example = EXAMPLE_WORKFLOWS[0]
    bundled_path = _example_workflow_path(example)

    assert _load_selected_workflow(loader, bundled_path) == Path(
        f"template-{example.id}.json"
    )
    assert loader.example_ids == [example.id]
    assert loader.workflow_paths == []

    external = tmp_path / example.filename
    external.write_text("{}", encoding="utf-8")
    assert _load_selected_workflow(loader, external) == external
    assert loader.example_ids == [example.id]
    assert loader.workflow_paths == [external]


class _FakeSignal:
    def __init__(self) -> None:
        self.callbacks = []

    def connect(self, callback) -> None:
        self.callbacks.append(callback)

    def emit(self) -> None:
        for callback in tuple(self.callbacks):
            callback()


class _FakeTimer:
    def __init__(self) -> None:
        self.timeout = _FakeSignal()
        self.interval = None
        self.active = False

    def setInterval(self, interval: int) -> None:  # noqa: N802
        self.interval = interval

    def start(self) -> None:
        self.active = True

    def stop(self) -> None:
        self.active = False


class _FakeApplication:
    def __init__(self) -> None:
        self.aboutToQuit = _FakeSignal()
        self.quit_called = False

    def quit(self) -> None:
        self.quit_called = True


class _FakeSplash:
    def __init__(self) -> None:
        self.keep_waiting_requested = _FakeSignal()
        self.hide_requested = _FakeSignal()
        self.close_requested = _FakeSignal()
        self.open_log_requested = _FakeSignal()
        self.visible = True
        self.close_permitted = False
        self.snapshots = []
        self.elapsed_values = []

    def update_elapsed(self, value: float) -> None:
        self.elapsed_values.append(value)

    def update_snapshot(self, snapshot) -> None:
        self.snapshots.append(snapshot)

    def isVisible(self) -> bool:  # noqa: N802
        return self.visible

    def show(self) -> None:
        self.visible = True

    def hide(self) -> None:
        self.visible = False

    def permit_close(self) -> None:
        self.close_permitted = True

    def close(self) -> None:
        self.visible = False


class _FakeProcess:
    def __init__(self, returncode=None) -> None:
        self.returncode = returncode
        self.terminate_called = False

    def poll(self):
        return self.returncode

    def terminate(self) -> None:
        self.terminate_called = True


def _controller(
    tmp_path,
    *,
    process_factory,
    schedule=None,
):
    channel_parent = tmp_path / "controller channel"
    channel_parent.mkdir()
    channel = StartupChannel.create(parent=channel_parent)
    application = _FakeApplication()
    splash = _FakeSplash()
    timer = _FakeTimer()
    controller = LauncherController(
        application=application,
        splash=splash,
        timer=timer,
        profile=LaunchProfile.AUTO,
        channel=channel,
        log_path=tmp_path / "logs" / "startup log.txt",
        timeout_seconds=30,
        executable=tmp_path / "Python Folder" / "python.exe",
        process_factory=process_factory,
        schedule=schedule,
    )
    return controller, application, splash, timer, channel


def test_controller_closes_splash_after_authenticated_ready_without_killing_child(
    tmp_path,
):
    child = _FakeProcess()

    def process_factory(command, **kwargs):
        del kwargs
        channel_path = command[command.index("--startup-channel") + 1]
        token = command[command.index("--startup-token") + 1]
        with StatusEmitter(channel_path, token) as emitter:
            emitter.progress("starting_python")
            emitter.progress("building_interface")
            emitter.ready()
        return child

    scheduled = []
    controller, application, splash, timer, channel = _controller(
        tmp_path,
        process_factory=process_factory,
        schedule=lambda delay, callback: scheduled.append((delay, callback)),
    )
    controller.start()
    controller.poll()

    assert controller.machine.snapshot.phase is StartupPhase.READY
    assert not timer.active
    assert child.terminate_called is False
    assert [item[0] for item in scheduled] == [350]

    scheduled[0][1]()
    assert application.quit_called
    assert splash.close_permitted
    assert not channel.directory.exists()


def test_controller_surfaces_child_spawn_failure(tmp_path):
    def fail_spawn(*_args, **_kwargs):
        raise OSError("simulated process creation failure")

    controller, _application, splash, timer, channel = _controller(
        tmp_path,
        process_factory=fail_spawn,
        schedule=lambda *_args: None,
    )
    controller.start()

    assert controller.machine.snapshot.phase is StartupPhase.FAILED
    assert "process creation failure" in controller.machine.snapshot.error
    assert splash.visible
    assert not timer.active
    controller._cleanup()
    assert not channel.directory.exists()


def test_controller_surfaces_log_directory_failure_before_spawning(tmp_path):
    blocking_path = tmp_path / "logs"
    blocking_path.write_text("not a directory", encoding="utf-8")
    spawn_called = False

    def process_factory(*_args, **_kwargs):
        nonlocal spawn_called
        spawn_called = True
        return _FakeProcess()

    controller, _application, splash, timer, channel = _controller(
        tmp_path,
        process_factory=process_factory,
        schedule=lambda *_args: None,
    )
    controller.start()

    assert controller.machine.snapshot.phase is StartupPhase.FAILED
    assert "FileExistsError" in controller.machine.snapshot.error
    assert spawn_called is False
    assert splash.visible
    assert not timer.active
    controller._cleanup()
    assert not channel.directory.exists()


def test_splash_setup_failure_cleans_channel_and_preserves_host_app_identity(
    qtbot,
    monkeypatch,
):
    del qtbot
    application = QApplication.instance()
    assert application is not None
    old_application_name = application.applicationName()
    old_organization_name = application.organizationName()
    application.setApplicationName("Host application")
    application.setOrganizationName("Host organization")

    channel = SimpleNamespace(cleanup_called=False)

    def cleanup():
        channel.cleanup_called = True

    channel.cleanup = cleanup
    channel_factory = SimpleNamespace(create=lambda: channel)
    monkeypatch.setattr(launcher, "StartupChannel", channel_factory)

    def fail_log_path(_profile):
        raise OSError("simulated state directory failure")

    monkeypatch.setattr(launcher, "create_startup_log_path", fail_log_path)
    try:
        with pytest.raises(OSError, match="state directory failure"):
            launcher.run_splash(LaunchProfile.AUTO, timeout_seconds=30)
        assert channel.cleanup_called is True
        assert application.applicationName() == "Host application"
        assert application.organizationName() == "Host organization"
    finally:
        application.setApplicationName(old_application_name)
        application.setOrganizationName(old_organization_name)


def test_controller_timeout_can_hide_while_child_keeps_running(tmp_path):
    child = _FakeProcess()
    controller, _application, splash, timer, channel = _controller(
        tmp_path,
        process_factory=lambda *_args, **_kwargs: child,
        schedule=lambda *_args: None,
    )
    controller.start()
    controller.machine.deadline = 0
    controller.poll()
    assert controller.machine.snapshot.phase is StartupPhase.TIMED_OUT
    assert timer.active

    splash.hide_requested.emit()
    assert controller.machine.snapshot.phase is StartupPhase.HIDDEN
    assert not splash.visible
    assert child.terminate_called is False
    controller._cleanup()
    assert not channel.directory.exists()


def test_controller_reshows_diagnostics_when_child_exits_after_hide(tmp_path):
    child = _FakeProcess(returncode=9)
    controller, _application, splash, timer, channel = _controller(
        tmp_path,
        process_factory=lambda *_args, **_kwargs: child,
        schedule=lambda *_args: None,
    )
    controller.start()
    controller.hide()
    controller.poll()
    assert controller.machine.snapshot.phase is StartupPhase.FAILED
    assert splash.visible
    assert not timer.active
    assert child.terminate_called is False
    controller._cleanup()
    assert not channel.directory.exists()
