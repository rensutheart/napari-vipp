from __future__ import annotations

import threading

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication, QWidget

from napari_vipp import _startup_widget
from napari_vipp._startup_widget import StartupState, VippStartupWidget


class _FakeVippWidget(QWidget):
    constructed_on: list[int] = []
    initial_runs = 0
    construction_options: list[tuple[bool, object]] = []
    states_during_initial_run: list[StartupState] = []
    splash_visible_during_initial_run: list[bool] = []

    def __init__(
        self,
        _viewer,
        parent=None,
        *,
        defer_initial_run=False,
        initial_compute_mode=None,
    ) -> None:
        super().__init__(parent)
        type(self).constructed_on.append(threading.get_ident())
        type(self).construction_options.append(
            (bool(defer_initial_run), initial_compute_mode)
        )

    def run_initial_pipeline_once(self) -> bool:
        type(self).initial_runs += 1
        host = self.parent()
        type(self).states_during_initial_run.append(host.startup_state)
        type(self).splash_visible_during_initial_run.append(
            host._stack.currentWidget() is host._splash
        )
        return True


class _FailingInitialWidget(_FakeVippWidget):
    discard_permitted = False

    def run_initial_pipeline_once(self) -> bool:
        assert self.parent().startup_state is StartupState.PREPARING
        raise RuntimeError("simulated initial workflow failure")

    def _permit_incomplete_startup_discard(self) -> None:
        type(self).discard_permitted = True


def _reset_fake_widget() -> None:
    _FakeVippWidget.constructed_on.clear()
    _FakeVippWidget.construction_options.clear()
    _FakeVippWidget.states_during_initial_run.clear()
    _FakeVippWidget.splash_visible_during_initial_run.clear()
    _FakeVippWidget.initial_runs = 0


def test_startup_host_preloads_off_thread_and_builds_widget_on_gui_thread(
    qtbot,
    monkeypatch,
):
    _reset_fake_widget()
    gui_thread_id = threading.get_ident()
    preload_thread_ids: list[int] = []

    def load_widget_class():
        preload_thread_ids.append(threading.get_ident())
        return _FakeVippWidget

    monkeypatch.setattr(
        _startup_widget,
        "_load_vipp_widget_class",
        load_widget_class,
    )
    viewer = object()
    host = VippStartupWidget(viewer, initial_compute_mode="prefer_gpu")
    qtbot.addWidget(host)

    assert host.startup_state is StartupState.IDLE
    assert host.real_widget is None

    host.show()
    qtbot.waitUntil(
        lambda: host.startup_state is StartupState.READY,
        timeout=5000,
    )
    qtbot.waitUntil(lambda: _FakeVippWidget.initial_runs == 1, timeout=1000)

    assert preload_thread_ids
    assert preload_thread_ids[0] != gui_thread_id
    assert _FakeVippWidget.constructed_on == [gui_thread_id]
    assert _FakeVippWidget.construction_options == [(True, "prefer_gpu")]
    assert _FakeVippWidget.states_during_initial_run == [StartupState.PREPARING]
    assert _FakeVippWidget.splash_visible_during_initial_run == [True]
    assert isinstance(host.real_widget, _FakeVippWidget)
    assert QApplication.instance().thread() == host.thread()


def test_startup_host_exposes_error_and_can_retry(qtbot, monkeypatch):
    _reset_fake_widget()
    attempts = 0

    def load_widget_class():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("simulated import failure")
        return _FakeVippWidget

    monkeypatch.setattr(
        _startup_widget,
        "_load_vipp_widget_class",
        load_widget_class,
    )
    host = VippStartupWidget(object())
    qtbot.addWidget(host)
    host.show()

    qtbot.waitUntil(
        lambda: host.startup_state is StartupState.ERROR,
        timeout=5000,
    )
    assert "simulated import failure" in host.detail_label.text()
    assert "RuntimeError" in host.error_details
    assert host.retry_button.isVisible()

    qtbot.mouseClick(host.retry_button, Qt.LeftButton)

    qtbot.waitUntil(
        lambda: host.startup_state is StartupState.READY,
        timeout=5000,
    )
    qtbot.waitUntil(lambda: _FakeVippWidget.initial_runs == 1, timeout=1000)
    assert attempts == 2


def test_initial_workflow_failure_keeps_splash_visible_and_can_retry(
    qtbot,
    monkeypatch,
):
    _reset_fake_widget()
    _FailingInitialWidget.discard_permitted = False
    attempts = 0

    def load_widget_class():
        nonlocal attempts
        attempts += 1
        return _FailingInitialWidget if attempts == 1 else _FakeVippWidget

    monkeypatch.setattr(
        _startup_widget,
        "_load_vipp_widget_class",
        load_widget_class,
    )
    host = VippStartupWidget(object())
    qtbot.addWidget(host)
    host.show()

    qtbot.waitUntil(
        lambda: host.startup_state is StartupState.ERROR,
        timeout=5000,
    )
    assert host._stack.currentWidget() is host._splash
    assert "simulated initial workflow failure" in host.detail_label.text()
    assert _FailingInitialWidget.discard_permitted is True

    qtbot.mouseClick(host.retry_button, Qt.LeftButton)
    qtbot.waitUntil(
        lambda: host.startup_state is StartupState.READY,
        timeout=5000,
    )
    assert attempts == 2


def test_closed_startup_host_ignores_late_preload_result(qtbot, monkeypatch):
    _reset_fake_widget()
    preload_started = threading.Event()
    release_preload = threading.Event()

    def load_widget_class():
        preload_started.set()
        assert release_preload.wait(timeout=5)
        return _FakeVippWidget

    monkeypatch.setattr(
        _startup_widget,
        "_load_vipp_widget_class",
        load_widget_class,
    )
    host = VippStartupWidget(object())
    qtbot.addWidget(host)

    assert host.start()
    assert preload_started.wait(timeout=2)
    assert host.close()
    release_preload.set()

    qtbot.waitUntil(lambda: not host._workers, timeout=5000)
    assert host.real_widget is None
    assert _FakeVippWidget.constructed_on == []
    assert _FakeVippWidget.initial_runs == 0
