from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
INSTALLER_PATH = REPO_ROOT / "scripts" / "install_cucim_windows.py"
POWERSHELL_WRAPPER = REPO_ROOT / "scripts" / "install_cucim_windows.ps1"
CMD_WRAPPER = REPO_ROOT / "scripts" / "install_cucim_windows.cmd"
ROOT_BUNDLE_WRAPPER = REPO_ROOT / "scripts" / "Install VIPP cuCIM.cmd"
INSTALLER_README = REPO_ROOT / "scripts" / "README-cucim-windows-installer.md"


def _load_installer():
    name = "_test_napari_vipp_cucim_windows_installer"
    spec = importlib.util.spec_from_file_location(name, INSTALLER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def installer():
    return _load_installer()


class _FakeSetupError(RuntimeError):
    pass


class _FakeExistingPlan:
    def __init__(self, wheel: Path, manifest: Path):
        self.wheel = wheel
        self.manifest = manifest

    def as_dict(self, *, plan_only: bool) -> dict[str, object]:
        return {
            "mode": "existing-environment",
            "plan_only": plan_only,
            "wheel": str(self.wheel),
            "manifest": str(self.manifest),
        }


def _fake_setup(target_python: Path, venv_root: Path):
    calls: list[tuple[Path, Path]] = []

    def create_existing_environment_plan(
        *,
        track_name,
        environment_python,
        cucim_wheel,
        cucim_manifest,
    ):
        assert track_name == "cuda13"
        assert Path(environment_python) == target_python
        wheel = Path(cucim_wheel)
        manifest = Path(cucim_manifest)
        calls.append((wheel, manifest))
        return _FakeExistingPlan(wheel, manifest)

    setup = SimpleNamespace(
        SetupError=_FakeSetupError,
        _resolve_executable=lambda value: Path(value).resolve(),
        _validate_existing_release_environment=lambda python: (
            venv_root
            if Path(python) == target_python
            else (_ for _ in ()).throw(_FakeSetupError("unexpected Python"))
        ),
        create_existing_environment_plan=create_existing_environment_plan,
    )
    return setup, calls


def _target_environment(tmp_path: Path) -> tuple[Path, Path]:
    venv = tmp_path / "VIPP release environment"
    python = venv / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"fixture")
    return venv.resolve(), python.resolve()


def _powershell(tmp_path: Path) -> Path:
    path = tmp_path / "Power Shell" / "powershell.exe"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(b"fixture")
    return path.resolve()


def _artifact_pair(installer, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    wheel = directory / installer.EXPECTED_WHEEL_PATTERN
    wheel.write_bytes(b"wheel")
    manifest = directory / f"{wheel.stem}.build-manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    return wheel.resolve(), manifest.resolve()


def test_new_build_plan_reuses_reviewed_helpers_and_writes_nothing(
    installer,
    tmp_path,
):
    venv, python = _target_environment(tmp_path)
    setup, calls = _fake_setup(python, venv)
    state_root = tmp_path / "retained state"
    powershell = _powershell(tmp_path)

    plan = installer.create_installer_plan(
        target_python=python,
        state_root=state_root,
        repository_root=REPO_ROOT,
        platform_name="win32",
        run_id="TEST-RUN",
        setup_module=setup,
        powershell=powershell,
    )

    assert plan.target_python == python
    assert plan.target_venv == venv
    assert plan.artifact_pair is None
    assert plan.build_command is not None
    assert plan.build_command[0] == str(powershell)
    assert str(REPO_ROOT / "scripts" / "build_cucim_windows.ps1") in (
        plan.build_command
    )
    assert plan.build_command[-6:] == (
        "-Python",
        str(python),
        "-WorkRoot",
        str((state_root / "builder-work").resolve()),
        "-OutputDirectory",
        str((state_root / "artifacts" / "TEST-RUN").resolve()),
    )
    assert calls == []
    assert not state_root.exists()

    document = plan.as_dict(plan_only=True)
    assert document["plan_only"] is True
    assert document["security_boundary"] == {
        "builder": str(REPO_ROOT / "scripts" / "build_cucim_windows.ps1"),
        "environment_installer": str(REPO_ROOT / "scripts" / "setup_gpu_dev.py"),
        "user_supplied_approval_hash": False,
    }
    assert "<builder-manifest-verified-wheel>" in document["install"]["argv_template"]


def test_reuse_plan_validates_artifacts_without_requiring_powershell(
    installer,
    tmp_path,
):
    venv, python = _target_environment(tmp_path)
    setup, calls = _fake_setup(python, venv)
    artifact_directory = tmp_path / "retained artifacts"
    wheel, manifest = _artifact_pair(installer, artifact_directory)

    plan = installer.create_installer_plan(
        target_python=python,
        state_root=tmp_path / "state",
        artifact_directory=artifact_directory,
        repository_root=REPO_ROOT,
        platform_name="win32",
        run_id="REUSE-RUN",
        setup_module=setup,
        powershell="missing-powershell-is-not-needed",
    )

    assert plan.reuses_artifacts is True
    assert plan.build_command is None
    assert plan.powershell is None
    assert plan.artifact_pair == installer.ArtifactPair(wheel, manifest)
    assert calls == [(wheel, manifest)]
    command = plan.install_command(plan.artifact_pair)
    assert command[0] == str(python)
    assert command[1] == str(REPO_ROOT / "scripts" / "setup_gpu_dev.py")
    assert "--existing-environment" in command
    assert command[-4:] == (
        "--cucim-wheel",
        str(wheel),
        "--cucim-manifest",
        str(manifest),
    )


def test_target_python_selection_is_explicit_then_environment_then_active_venv(
    installer,
):
    selected = installer._selected_target_python(
        "explicit.exe",
        environment={installer.TARGET_ENVIRONMENT_VARIABLE: "configured.exe"},
        current_executable="active.exe",
        current_prefix="C:/venv",
        base_prefix="C:/base",
    )
    assert selected == "explicit.exe"

    selected = installer._selected_target_python(
        None,
        environment={installer.TARGET_ENVIRONMENT_VARIABLE: "configured.exe"},
        current_executable="active.exe",
        current_prefix="C:/venv",
        base_prefix="C:/base",
    )
    assert selected == "configured.exe"

    selected = installer._selected_target_python(
        None,
        environment={},
        current_executable="active.exe",
        current_prefix="C:/venv",
        base_prefix="C:/base",
    )
    assert selected == "active.exe"

    with pytest.raises(installer.InstallerError, match="--target-python"):
        installer._selected_target_python(
            None,
            environment={},
            current_executable="base.exe",
            current_prefix="C:/base",
            base_prefix="C:/base",
        )


def test_plan_rejects_non_windows_before_loading_helpers(installer, tmp_path):
    with pytest.raises(installer.InstallerError, match="Windows only"):
        installer.create_installer_plan(
            target_python="python",
            repository_root=tmp_path / "does-not-matter",
            platform_name="linux",
        )


def test_safe_directory_rejects_filesystem_root_and_non_directory(
    installer,
    tmp_path,
):
    filesystem_root = Path(tmp_path.anchor)
    with pytest.raises(installer.InstallerError, match="filesystem root"):
        installer._safe_non_root_directory(
            filesystem_root,
            label="test root",
            must_exist=True,
        )

    file_path = tmp_path / "not-a-directory"
    file_path.write_text("fixture", encoding="utf-8")
    with pytest.raises(installer.InstallerError, match="not a directory"):
        installer._safe_non_root_directory(
            file_path,
            label="test path",
            must_exist=True,
        )


def test_plan_rejects_state_or_work_paths_inside_released_environment(
    installer,
    tmp_path,
):
    venv, python = _target_environment(tmp_path)
    setup, _calls = _fake_setup(python, venv)

    with pytest.raises(installer.InstallerError, match="outside the released"):
        installer.create_installer_plan(
            target_python=python,
            state_root=venv / "installer-state",
            repository_root=REPO_ROOT,
            platform_name="win32",
            run_id="UNSAFE-STATE",
            setup_module=setup,
            powershell=_powershell(tmp_path),
        )

    with pytest.raises(installer.InstallerError, match="outside the released"):
        installer.create_installer_plan(
            target_python=python,
            state_root=tmp_path / "safe-state",
            work_root=venv / "builder-work",
            repository_root=REPO_ROOT,
            platform_name="win32",
            run_id="UNSAFE-WORK",
            setup_module=setup,
            powershell=_powershell(tmp_path),
        )


def test_artifact_discovery_requires_exactly_one_matching_named_pair(
    installer,
    tmp_path,
):
    directory = tmp_path / "artifacts"
    wheel, manifest = _artifact_pair(installer, directory)
    assert installer._discover_artifact_pair(directory) == installer.ArtifactPair(
        wheel,
        manifest,
    )

    extra = directory / "unexpected.build-manifest.json"
    extra.write_text("{}", encoding="utf-8")
    with pytest.raises(installer.InstallerError, match="exactly one"):
        installer._discover_artifact_pair(directory)
    extra.unlink()

    manifest.rename(directory / "wrong-name.build-manifest.json")
    with pytest.raises(installer.InstallerError, match="does not match"):
        installer._discover_artifact_pair(directory)


def test_execute_builds_then_delegates_install_and_retains_complete_record(
    installer,
    tmp_path,
):
    venv, python = _target_environment(tmp_path)
    setup, calls = _fake_setup(python, venv)
    state_root = tmp_path / "state"
    plan = installer.create_installer_plan(
        target_python=python,
        state_root=state_root,
        repository_root=REPO_ROOT,
        platform_name="win32",
        run_id="EXECUTE-RUN",
        setup_module=setup,
        powershell=_powershell(tmp_path),
    )
    commands: list[tuple[str, ...]] = []

    def runner(argv, log):
        commands.append(argv)
        log.write(f"mocked command {len(commands)}")
        if argv == plan.build_command:
            _artifact_pair(installer, plan.artifact_directory)

    artifacts = installer.execute_installer(
        plan,
        setup_module=setup,
        command_runner=runner,
    )

    assert commands == [plan.build_command, plan.install_command(artifacts)]
    assert calls == [(artifacts.wheel, artifacts.manifest)]
    assert artifacts.wheel.is_file()
    assert artifacts.manifest.is_file()
    assert plan.log_path.is_file()
    log_text = plan.log_path.read_text(encoding="utf-8")
    assert "[1/5] Revalidating" in log_text
    assert "[5/5] cuCIM is installed and verified" in log_text

    journal = json.loads(plan.journal_path.read_text(encoding="utf-8"))
    assert journal["status"] == "complete"
    assert journal["error"] is None
    assert journal["artifacts"] == artifacts.as_dict()
    assert journal["setup_plan"]["mode"] == "existing-environment"
    assert journal["finished_utc"]


def test_failed_install_retains_artifacts_log_and_actionable_journal(
    installer,
    tmp_path,
):
    venv, python = _target_environment(tmp_path)
    setup, _calls = _fake_setup(python, venv)
    artifact_directory = tmp_path / "artifacts"
    wheel, manifest = _artifact_pair(installer, artifact_directory)
    plan = installer.create_installer_plan(
        target_python=python,
        state_root=tmp_path / "state",
        artifact_directory=artifact_directory,
        repository_root=REPO_ROOT,
        platform_name="win32",
        run_id="FAILED-RUN",
        setup_module=setup,
    )

    def fail_install(_argv, _log):
        raise installer.InstallerError("mocked install failure")

    with pytest.raises(installer.InstallerError, match="mocked install failure"):
        installer.execute_installer(
            plan,
            setup_module=setup,
            command_runner=fail_install,
        )

    assert wheel.is_file()
    assert manifest.is_file()
    assert plan.log_path.is_file()
    journal = json.loads(plan.journal_path.read_text(encoding="utf-8"))
    assert journal["status"] == "failed"
    assert journal["artifacts"] == {
        "wheel": str(wheel),
        "manifest": str(manifest),
    }
    assert journal["error"] == {
        "type": "InstallerError",
        "message": "mocked install failure",
    }


def test_logged_command_interrupt_stops_process_group_with_bounded_escalation(
    installer,
    monkeypatch,
    tmp_path,
):
    class InterruptingOutput:
        closed = False

        def __iter__(self):
            return self

        def __next__(self):
            raise KeyboardInterrupt

        def close(self):
            self.closed = True

    class Process:
        def __init__(self):
            self.stdout = InterruptingOutput()
            self.events = []
            self.wait_calls = 0

        def poll(self):
            return None

        def send_signal(self, value):
            self.events.append(("signal", value))

        def wait(self, *, timeout=None):
            self.events.append(("wait", timeout))
            self.wait_calls += 1
            if self.wait_calls < 3:
                raise subprocess.TimeoutExpired("mock", timeout)
            return 1

        def terminate(self):
            self.events.append(("terminate", None))

        def kill(self):
            self.events.append(("kill", None))

    process = Process()
    popen_kwargs = {}

    def fake_popen(_argv, **kwargs):
        popen_kwargs.update(kwargs)
        return process

    monkeypatch.setattr(installer.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        installer.subprocess,
        "CREATE_NEW_PROCESS_GROUP",
        0x00000200,
        raising=False,
    )
    monkeypatch.setattr(
        installer.signal,
        "CTRL_BREAK_EVENT",
        123,
        raising=False,
    )

    log_path = tmp_path / "interrupt.log"
    with installer.RunLog(log_path) as log:
        with pytest.raises(KeyboardInterrupt):
            installer._run_logged_command(("powershell.exe", "-File", "build.ps1"), log)

    assert popen_kwargs["creationflags"] == 0x00000200
    assert process.events == [
        ("signal", 123),
        ("wait", installer.CANCEL_GRACE_SECONDS),
        ("terminate", None),
        ("wait", installer.TERMINATE_GRACE_SECONDS),
        ("kill", None),
        ("wait", installer.TERMINATE_GRACE_SECONDS),
    ]
    assert process.stdout.closed is True
    text = log_path.read_text(encoding="utf-8")
    assert "sending CTRL_BREAK_EVENT" in text
    assert "terminating it" in text
    assert "killing it" in text


def test_second_interrupt_cannot_abandon_process_group_teardown(
    installer,
    monkeypatch,
    tmp_path,
):
    class Process:
        def __init__(self):
            self.events = []

        def poll(self):
            return None

        def send_signal(self, value):
            self.events.append(("signal", value))
            raise KeyboardInterrupt

        def terminate(self):
            self.events.append(("terminate", None))
            raise KeyboardInterrupt

        def kill(self):
            self.events.append(("kill", None))

        def wait(self, *, timeout=None):
            self.events.append(("wait", timeout))
            return 1

    process = Process()
    monkeypatch.setattr(
        installer.signal,
        "CTRL_BREAK_EVENT",
        123,
        raising=False,
    )
    log_path = tmp_path / "second-interrupt.log"
    with installer.RunLog(log_path) as log:
        installer._cancel_process_group(process, log)

    assert process.events == [
        ("signal", 123),
        ("terminate", None),
        ("kill", None),
        ("wait", installer.TERMINATE_GRACE_SECONDS),
    ]
    text = log_path.read_text(encoding="utf-8")
    assert "terminating it" in text
    assert "killing it" in text


@pytest.mark.parametrize("argument", ["--plan-only", "--dry-run"])
def test_plan_and_dry_run_print_without_executing_or_writing(
    installer,
    monkeypatch,
    tmp_path,
    capsys,
    argument,
):
    state_root = tmp_path / "must-not-be-created"
    plan = SimpleNamespace(
        as_dict=lambda *, plan_only: {
            "schema": installer.INSTALLER_SCHEMA,
            "plan_only": plan_only,
        }
    )
    monkeypatch.setattr(installer, "create_installer_plan", lambda **_kwargs: plan)
    monkeypatch.setattr(
        installer,
        "execute_installer",
        lambda _plan: pytest.fail("dry-run must not execute"),
    )

    exit_code = installer.main(
        ["--target-python", "selected.exe", "--state-root", str(state_root), argument]
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {
        "schema": installer.INSTALLER_SCHEMA,
        "plan_only": True,
    }
    assert not state_root.exists()


def test_double_click_boundary_reports_unexpected_failure_without_traceback(
    installer,
    monkeypatch,
    tmp_path,
    capsys,
):
    log_path = tmp_path / "retained.log"
    log_path.write_text("details\n", encoding="utf-8")
    plan = SimpleNamespace(log_path=log_path)
    monkeypatch.setattr(installer, "create_installer_plan", lambda **_kwargs: plan)
    monkeypatch.setattr(
        installer,
        "execute_installer",
        lambda _plan: (_ for _ in ()).throw(OSError("simulated disk failure")),
    )

    assert installer.main(["--target-python", "selected.exe"]) == 3
    error = capsys.readouterr().err
    assert "OSError: simulated disk failure" in error
    assert str(log_path) in error
    assert "Traceback" not in error


def test_thin_windows_entries_delegate_without_user_approval_hash():
    python_text = INSTALLER_PATH.read_text(encoding="utf-8")
    powershell = POWERSHELL_WRAPPER.read_text(encoding="utf-8")
    cmd = CMD_WRAPPER.read_text(encoding="utf-8")
    root_bundle_cmd = ROOT_BUNDLE_WRAPPER.read_text(encoding="utf-8")
    readme = INSTALLER_README.read_text(encoding="utf-8")

    assert "build_cucim_windows.ps1" in python_text
    assert "setup_gpu_dev.py" in python_text
    assert "CUCIM_WHEEL_PAYLOAD_SHA256" not in python_text
    assert re.search(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])", python_text) is None
    assert "OpenFileDialog" in powershell
    assert "--target-python" in powershell
    assert "install_cucim_windows.py" in powershell
    assert "install_cucim_windows.ps1" in cmd
    assert "scripts\\install_cucim_windows.cmd" in root_bundle_cmd
    assert "install_cucim_windows.cmd" in root_bundle_cmd
    assert "not yet a signed native Windows wizard" in readme
    assert "There is no checksum or approval value" in readme
