from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
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


def test_packaged_commands_expose_one_graphical_vipp_launcher():
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]

    assert project["scripts"]["vipp"] == "napari_vipp.__main__:main"
    assert project["scripts"]["vipp-install-plan"] == "napari_vipp.installer.cli:main"
    assert project["gui-scripts"] == {
        "vipp-app": "napari_vipp.launcher:main_auto",
    }


@pytest.mark.parametrize(
    ("facade_name", "conflicting_profile", "expected_profile"),
    [
        ("main_auto", "prefer_gpu", "auto"),
        ("main_cpu", "prefer_gpu", "cpu"),
        ("main_prefer_gpu", "cpu", "prefer_gpu"),
    ],
)
def test_graphical_launcher_and_legacy_facades_keep_explicit_profile(
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


def test_argument_parsing_reads_version_only_for_explicit_request(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(
        launcher, "installed_version", lambda: calls.append(True) or "1.2.3"
    )
    parser = launcher.build_parser()
    parser.parse_args([])
    assert calls == []
    with pytest.raises(SystemExit) as exited:
        parser.parse_args(["--version"])
    assert exited.value.code == 0
    assert calls == [True]
    assert capsys.readouterr().out == "VIPP 1.2.3\n"


def test_timing_reporter_retains_stage_durations_and_forwards_protocol(
    monkeypatch, capsys
):
    wall_clock = iter((10.0, 10.25, 15.25, 15.5))
    cpu_clock = iter((2.0, 2.1, 3.1, 3.125))
    monkeypatch.setattr(app.time, "perf_counter", lambda: next(wall_clock))
    monkeypatch.setattr(app.time, "process_time", lambda: next(cpu_clock))
    calls = []
    delegate = SimpleNamespace(
        progress=lambda *args: calls.append(("progress", args)),
        ready=lambda *args: calls.append(("ready", args)),
        failure=lambda *args, **kwargs: calls.append(("failure", args, kwargs)),
        close=lambda: calls.append(("close",)),
    )
    reporter = app._TimingReporter(delegate)
    reporter.progress("loading_napari", "Loading")
    reporter.ready("Ready")
    reporter.failure("Failed", error="details")
    reporter.close()
    records = [
        json.loads(line.removeprefix("VIPP startup timing: "))
        for line in capsys.readouterr().err.splitlines()
    ]
    assert records == [
        {
            "stage": "loading_napari",
            "elapsed_seconds": 0.25,
            "process_cpu_seconds": 0.1,
            "previous_stage": None,
            "previous_stage_seconds": 0.25,
        },
        {
            "stage": "ready",
            "elapsed_seconds": 5.25,
            "process_cpu_seconds": 1.1,
            "previous_stage": "loading_napari",
            "previous_stage_seconds": 5.0,
        },
        {
            "stage": "failure",
            "elapsed_seconds": 5.5,
            "process_cpu_seconds": 1.125,
            "previous_stage": "ready",
            "previous_stage_seconds": 0.25,
        },
    ]
    assert calls == [
        ("progress", ("loading_napari", "Loading")),
        ("ready", ("Ready",)),
        ("failure", ("Failed",), {"error": "details"}),
        ("close",),
    ]


def test_timing_reporter_supports_pythonw_without_stderr(monkeypatch):
    monkeypatch.setattr(app.sys, "stderr", None)
    reporter = app._TimingReporter(app._ConsoleReporter())
    reporter.progress("loading_napari")
    reporter.ready()
    reporter.failure("Failed")
    reporter.close()


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
        f"--startup-token={channel.token}",
    ]
    channel.cleanup()


@pytest.mark.parametrize("desktop", [False, True])
@pytest.mark.parametrize("token", ["a" * 43, "-" + "a" * 42, "--desktop" + "a" * 34])
def test_desktop_flag_reaches_child_and_no_splash_launch(
    tmp_path, monkeypatch, desktop, token
):
    monkeypatch.setattr(
        "napari_vipp.startup.secrets.token_urlsafe", lambda _size: token
    )
    channel = StartupChannel.create(parent=tmp_path)
    try:
        command = build_child_command(
            executable="python", profile="cpu", channel=channel, desktop=desktop
        )
        parsed = app.build_parser().parse_args(command[3:])
        assert parsed.desktop is desktop
        assert parsed.startup_token == token == channel.token
        assert parsed.startup_channel == channel.path
        assert parsed.profile == "cpu"
    finally:
        channel.cleanup()
    calls = []
    monkeypatch.setattr(app, "main", lambda argv: calls.append(argv) or 0)
    assert launcher.main(["--no-splash", *(["--desktop"] if desktop else [])]) == 0
    assert app.build_parser().parse_args(calls[0]).desktop is desktop


@pytest.mark.parametrize("desktop", [False, True])
def test_application_brands_only_explicit_desktop_launch(qtbot, monkeypatch, desktop):
    import napari

    from napari_vipp.ui import desktop_branding, reader_warmup

    viewer_calls = []
    branding_calls = []
    dock_calls = []
    launch_events = []
    monkeypatch.setattr(
        reader_warmup, "schedule_reader_warmup", lambda: launch_events.append("warmup")
    )
    dock_window = object()

    def add_dock_widget(widget, **kwargs):
        dock_calls.append((widget, kwargs))
        return SimpleNamespace(window=lambda: dock_window)

    viewer = SimpleNamespace(
        window=SimpleNamespace(add_dock_widget=add_dock_widget),
        show=lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        napari, "Viewer", lambda **kwargs: viewer_calls.append(kwargs) or viewer
    )
    monkeypatch.setattr(napari, "run", lambda: None)
    monkeypatch.setattr("napari_vipp._widget.VippWidget", _FakeVippWidget)
    monkeypatch.setattr(
        _FakeVippWidget, "_apply_initial_dock_size", lambda _self: None, raising=False
    )
    monkeypatch.setattr(
        desktop_branding,
        "set_desktop_process_identity",
        lambda: branding_calls.append("identity"),
    )
    monkeypatch.setattr(
        desktop_branding,
        "apply_desktop_branding",
        lambda application, window: branding_calls.append(window),
    )
    reporter = app._ConsoleReporter()
    monkeypatch.setattr(
        reporter, "ready", lambda _message: launch_events.append("ready")
    )
    assert (
        app.run_application(LaunchProfile.CPU, reporter, desktop=desktop)
        == 0
    )
    assert viewer_calls == [{"title": "VIPP", "show": False}]
    # The standalone editor must be discoverable by napari's plugin command,
    # which reuses docks by this registry key rather than by their title.
    assert len(dock_calls) == 1
    assert dock_calls[0][0].viewer is viewer
    assert dock_calls[0][1] == {
        "area": "bottom",
        "name": "VIPP Workflow (napari-vipp)",
    }
    assert branding_calls == (["identity", dock_window] if desktop else [])
    assert launch_events == ["ready", "warmup"]


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


def _finish_spawn(controller):
    """Wait only in tests, then deliver the worker result on the caller thread."""
    controller._spawn_thread.join(timeout=5)
    assert not controller._spawn_thread.is_alive()
    controller.poll()


def test_splash_paints_and_keeps_polling_during_blocked_process_creation(
    qtbot, tmp_path
):
    from qtpy.QtCore import QTimer

    from napari_vipp.ui.startup_splash import StartupSplash

    channel = StartupChannel.create(parent=tmp_path)
    splash = StartupSplash(
        profile="auto", version="test", log_path=tmp_path / "startup.log"
    )
    qtbot.addWidget(splash)
    timer = QTimer()
    entered = threading.Event()
    release = threading.Event()
    spawn_observations = []
    main_thread = threading.get_ident()
    scheduled = []

    def slow_spawn(*_args, **_kwargs):
        spawn_observations.append((splash._paint_reported, threading.get_ident()))
        entered.set()
        if not release.wait(5):
            raise TimeoutError("test did not release process creation")
        with StatusEmitter(channel.path, channel.token) as emitter:
            emitter.ready()
        return _FakeProcess()

    controller = LauncherController(
        application=_FakeApplication(),
        splash=splash,
        timer=timer,
        profile=LaunchProfile.AUTO,
        channel=channel,
        log_path=tmp_path / "startup.log",
        timeout_seconds=30,
        process_factory=slow_spawn,
        schedule=lambda delay, callback: scheduled.append((delay, callback)),
    )
    splash.first_painted.connect(controller.start)
    heartbeat = QTimer()
    ticks = []
    heartbeat.timeout.connect(lambda: ticks.append(True))
    heartbeat.start(20)
    try:
        assert not entered.is_set()
        splash.show()
        qtbot.waitUntil(entered.is_set, timeout=3000)
        assert spawn_observations == [(True, controller._spawn_thread.ident)]
        assert controller._spawn_thread.ident != main_thread
        ticks.clear()
        qtbot.waitUntil(lambda: len(ticks) >= 4, timeout=1000)
        controller.machine.started_at -= 20
        qtbot.waitUntil(lambda: splash.elapsed_label.text() == "Elapsed 0:20")
        assert controller.machine.snapshot.phase is StartupPhase.STARTING
        splash.hide_requested.emit()
        assert not splash.isVisible()
        release.set()
        qtbot.waitUntil(lambda: bool(scheduled), timeout=3000)
        assert controller.machine.snapshot.phase is StartupPhase.READY
        assert controller.process.terminate_called is False
    finally:
        release.set()
        if controller._spawn_thread is not None:
            controller._spawn_thread.join(timeout=5)
        heartbeat.stop()
        timer.stop()
        controller._cleanup()
        splash.permit_close()


def test_quit_during_launch_preparation_skips_spawn_and_cleans_channel(
    tmp_path, monkeypatch
):
    entered = threading.Event()
    release = threading.Event()
    spawned = []

    def delayed_version():
        entered.set()
        if not release.wait(5):
            raise TimeoutError("test did not release preparation")
        return "test"

    monkeypatch.setattr(launcher, "installed_version", delayed_version)
    controller, _application, _splash, _timer, channel = _controller(
        tmp_path, process_factory=lambda *_a, **_k: spawned.append(True)
    )
    try:
        controller.start()
        controller.start()
        assert entered.wait(3)
        controller._cleanup()
        assert channel.directory.exists()
    finally:
        release.set()
        controller._spawn_thread.join(timeout=5)
    assert spawned == []
    assert not channel.directory.exists()


def test_controller_closes_splash_after_authenticated_ready_without_killing_child(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "napari_vipp.startup.secrets.token_urlsafe", lambda _size: "-" + "a" * 42
    )
    child = _FakeProcess()

    def process_factory(command, **kwargs):
        del kwargs
        arguments = app.build_parser().parse_args(command[3:])
        with StatusEmitter(
            arguments.startup_channel, arguments.startup_token
        ) as emitter:
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
    _finish_spawn(controller)
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
    _finish_spawn(controller)

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
    _finish_spawn(controller)

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
    _finish_spawn(controller)
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
    _finish_spawn(controller)
    controller.hide()
    controller.poll()
    assert controller.machine.snapshot.phase is StartupPhase.FAILED
    assert splash.visible
    assert not timer.active
    assert child.terminate_called is False
    controller._cleanup()
    assert not channel.directory.exists()
