from __future__ import annotations

import runpy
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from napari_vipp.installer.uninstall import (
    DeferredSelfDelete,
    UninstallIssue,
    UninstallResult,
    UninstallStatus,
)
from napari_vipp.installer.uninstall_gui import UninstallExitCode, main


class FakeRoot:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def withdraw(self) -> None:
        self.events.append("withdraw")

    def destroy(self) -> None:
        self.events.append("destroy")


class TkLikeFakeRoot(FakeRoot):
    tk = object()


class FakeMessages:
    def __init__(self, events: list[str], *, confirm: bool = True) -> None:
        self.events = events
        self.confirm = confirm
        self.errors: list[str] = []
        self.result_thread_ids: list[int] = []

    def askyesno(self, _title: str, _message: str, **_options: object) -> bool:
        self.events.append("confirm")
        return self.confirm

    def showinfo(self, _title: str, message: str, **_options: object) -> None:
        self.events.append("result-visible")
        self.result_thread_ids.append(threading.get_ident())
        assert "images and analysis files were not changed" in message

    def showerror(self, _title: str, message: str, **_options: object) -> None:
        self.events.append("error-visible")
        self.result_thread_ids.append(threading.get_ident())
        self.errors.append(message)


class FakeUninstaller:
    def __init__(self, events: list[str], result: UninstallResult) -> None:
        self.events = events
        self.result = result
        self.prepared = SimpleNamespace(
            record=SimpleNamespace(version="0.13.0"),
            fingerprint="a" * 64,
        )
        self.apply_thread_id: int | None = None

    def prepare(self, managed_root: Path, *, shortcut_roots):
        self.events.append("prepare")
        assert managed_root.is_absolute()
        assert tuple(shortcut_roots)
        return self.prepared

    def authorize(self, prepared):
        self.events.append("authorize")
        assert prepared is self.prepared
        return object()

    def apply(
        self,
        prepared,
        authorization,
        *,
        current_executable: Path,
        current_pid: int,
    ) -> UninstallResult:
        self.events.append("apply")
        self.apply_thread_id = threading.get_ident()
        assert prepared is self.prepared
        assert authorization is not None
        assert current_executable.is_absolute()
        assert current_pid > 0
        return self.result


def _result(
    tmp_path: Path,
    *,
    completed: bool,
    deferred: DeferredSelfDelete | None = None,
) -> UninstallResult:
    root = tmp_path / "managed"
    if completed:
        return UninstallResult(
            status=UninstallStatus.COMPLETED,
            managed_root=root,
            removed_paths=(root / "owned",),
            preserved_paths=(),
            issues=(),
            deferred_self_delete=deferred,
            message="VIPP was removed.",
        )
    failed = root / "private-environment"
    issue = UninstallIssue(
        failed,
        "remove owned environment",
        "Access is denied because VIPP is still open.",
    )
    return UninstallResult(
        status=UninstallStatus.INCOMPLETE,
        managed_root=root,
        removed_paths=(),
        preserved_paths=(failed,),
        issues=(issue,),
        deferred_self_delete=None,
        message="Cleanup is incomplete.",
    )


def _run(
    tmp_path: Path,
    uninstaller: FakeUninstaller,
    messages: FakeMessages,
    events: list[str],
    *,
    scheduler=lambda _request: None,
    progress_runner=None,
    root_factory=None,
) -> int:
    return main(
        ["--uninstall", "--managed-root", str(tmp_path / "managed")],
        uninstaller=uninstaller,  # type: ignore[arg-type]
        root_factory=root_factory or (lambda: FakeRoot(events)),
        messageboxes=messages,
        shortcut_roots_provider=lambda: (
            tmp_path / "Desktop",
            tmp_path / "Start Menu" / "Programs",
        ),
        scheduler=scheduler,
        current_executable=tmp_path / "cached" / "VIPP-Setup.exe",
        current_pid=42,
        progress_runner=progress_runner,
    )


def test_novice_can_decline_the_single_confirmation(tmp_path: Path) -> None:
    events: list[str] = []
    messages = FakeMessages(events, confirm=False)
    uninstaller = FakeUninstaller(events, _result(tmp_path, completed=True))

    exit_code = _run(tmp_path, uninstaller, messages, events)

    assert exit_code == UninstallExitCode.CANCELLED
    assert events.count("confirm") == 1
    assert "apply" not in events
    assert events[-1] == "destroy"


def test_self_delete_is_scheduled_only_after_success_is_visible(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    target = tmp_path / "cached" / "VIPP-Setup.exe"
    deferred = DeferredSelfDelete(
        target=target,
        expected_sha256="a" * 64,
        wait_for_pid=42,
        argv=("powershell.exe", "-EncodedCommand", "encoded"),
    )
    messages = FakeMessages(events)
    uninstaller = FakeUninstaller(
        events,
        _result(tmp_path, completed=True, deferred=deferred),
    )

    exit_code = _run(
        tmp_path,
        uninstaller,
        messages,
        events,
        scheduler=lambda request: events.append(f"scheduled:{request.target}"),
    )

    assert exit_code == UninstallExitCode.COMPLETED
    assert events.count("confirm") == 1
    assert events.index("result-visible") < events.index(f"scheduled:{target}")
    assert events[-1] == "destroy"


def test_incomplete_cleanup_shows_exact_path_and_error(tmp_path: Path) -> None:
    events: list[str] = []
    messages = FakeMessages(events)
    uninstaller = FakeUninstaller(events, _result(tmp_path, completed=False))

    exit_code = _run(tmp_path, uninstaller, messages, events)

    failed = tmp_path / "managed" / "private-environment"
    assert exit_code == UninstallExitCode.INCOMPLETE
    assert str(failed) in messages.errors[-1]
    assert "Access is denied" in messages.errors[-1]
    assert "cleanup is incomplete" in messages.errors[-1]


@pytest.mark.parametrize("completed", [True, False])
def test_removal_runs_in_worker_while_result_ui_stays_on_gui_thread(
    tmp_path: Path,
    completed: bool,
) -> None:
    events: list[str] = []
    messages = FakeMessages(events)
    uninstaller = FakeUninstaller(events, _result(tmp_path, completed=completed))
    gui_thread_id = threading.get_ident()

    def progress_runner(_root, operation):
        events.append("progress-visible")
        outcomes = []
        errors = []

        def work() -> None:
            try:
                outcomes.append(operation())
            except BaseException as exc:
                errors.append(exc)

        worker = threading.Thread(target=work, name="test-uninstall-worker")
        worker.start()
        worker.join(timeout=2)
        assert not worker.is_alive()
        if errors:
            raise errors[0]
        return outcomes[0]

    exit_code = _run(
        tmp_path,
        uninstaller,
        messages,
        events,
        progress_runner=progress_runner,
    )

    expected = (
        UninstallExitCode.COMPLETED if completed else UninstallExitCode.INCOMPLETE
    )
    assert exit_code == expected
    assert events.index("progress-visible") < events.index("apply")
    result_event = "result-visible" if completed else "error-visible"
    assert events.index("apply") < events.index(result_event)
    assert uninstaller.apply_thread_id != gui_thread_id
    assert messages.result_thread_ids == [gui_thread_id]


def test_tk_root_uses_the_visible_progress_route_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    messages = FakeMessages(events)
    uninstaller = FakeUninstaller(events, _result(tmp_path, completed=True))

    def visible_progress(_root, operation):
        events.append("visible-removing-window")
        return operation()

    monkeypatch.setattr(
        "napari_vipp.installer.uninstall_gui._run_removal_with_progress",
        visible_progress,
    )

    exit_code = _run(
        tmp_path,
        uninstaller,
        messages,
        events,
        root_factory=lambda: TkLikeFakeRoot(events),
    )

    assert exit_code == UninstallExitCode.COMPLETED
    assert events.index("visible-removing-window") < events.index("apply")


def test_managed_root_must_be_explicit_and_absolute(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--uninstall", "--managed-root", "relative-folder"])

    assert caught.value.code == 2


def test_frozen_entry_dispatches_uninstall_before_normal_setup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    entry = (
        Path(__file__).resolve().parents[3]
        / "packaging"
        / "windows"
        / "vipp_installer_entry.py"
    )
    called: list[object] = []
    fake_uninstall_gui = SimpleNamespace(main=lambda: called.append("uninstall") or 37)
    fake_normal_gui = SimpleNamespace(main=lambda: called.append("install") or 38)
    fake_splash = SimpleNamespace(update_text=lambda text: called.append(text))
    monkeypatch.setitem(
        sys.modules,
        "napari_vipp.installer.uninstall_gui",
        fake_uninstall_gui,
    )
    monkeypatch.setitem(sys.modules, "napari_vipp.installer.gui", fake_normal_gui)
    monkeypatch.setitem(sys.modules, "pyi_splash", fake_splash)
    monkeypatch.setattr(sys, "platform", "test-host")
    monkeypatch.setattr(
        sys,
        "argv",
        [str(entry), "--uninstall", "--managed-root", str(tmp_path / "managed")],
    )

    with pytest.raises(SystemExit) as caught:
        runpy.run_path(str(entry), run_name="__main__")

    assert caught.value.code == 37
    assert "uninstall" in called
    assert "install" not in called
