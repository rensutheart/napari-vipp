from __future__ import annotations

import threading

import pytest
from qtpy.QtCore import QEvent, QPoint, QPointF, Qt
from qtpy.QtGui import QColor, QImage, QMouseEvent, QPainter, QPalette, QRegion
from qtpy.QtWidgets import QApplication, QLabel, QLCDNumber, QWidget

from napari_vipp.startup import LaunchProfile, StartupPhase, StartupSnapshot
from napari_vipp.ui.startup_splash import StartupSplash


def test_first_frame_clock_and_minimize_work_while_fonts_are_pending(
    qtbot, tmp_path, monkeypatch
):
    from napari_vipp.ui import startup_splash

    entered = threading.Event()
    release = threading.Event()

    def prepare(done, cancelled):
        entered.set()
        release.wait(5)
        done.set()

    monkeypatch.setattr(startup_splash, "_prepare_font_database", prepare)
    splash = StartupSplash(
        profile="auto",
        version="test",
        log_path=tmp_path / "startup.log",
        defer_font_loading=True,
    )
    qtbot.addWidget(splash)
    try:
        assert not splash._ui_ready
        assert not splash.findChildren(QLCDNumber)
        assert splash._bootstrap_version.text() == "VIPP test"
        assert splash._bootstrap_stage.text() == "Step 0 of 6"
        splash.show()
        qtbot.waitUntil(entered.is_set)
        assert splash._paint_reported
        before = splash._bootstrap_clock.grab().toImage()
        splash.update_elapsed(21)
        after = splash._bootstrap_clock.grab().toImage()
        assert before != after
        assert splash._bootstrap_clock.accessibleDescription() == "Elapsed 0:21"
        splash.update_snapshot(_snapshot(StartupPhase.STARTING))
        assert splash._bootstrap_status.text() == "Test status"
        assert splash._bootstrap_stage.text() == "Step 2 of 6"
        splash.update_snapshot(_snapshot(StartupPhase.FAILED, error="test failure"))
        qtbot.mouseClick(splash, Qt.LeftButton, pos=QPoint(615, 24))
        assert splash.isMinimized()
        splash.showNormal()
        release.set()
        qtbot.waitUntil(lambda: splash._ui_ready)
        assert splash.elapsed_label.text() == "Elapsed 0:21"
        assert splash.version_label.text() == "VIPP test"
        assert splash.version_label.font().pixelSize() == 12
        assert splash.elapsed_label.font().pixelSize() == 11
        assert splash.stage_label.font().pixelSize() == 11
        assert splash.detail_label.text() == "test failure"
        assert splash.close_button.isVisibleTo(splash)
    finally:
        release.set()
        splash.finish_font_worker()
        splash.permit_close()


@pytest.mark.parametrize("missing_asset", ["frame", "lettering"])
def test_missing_first_frame_asset_uses_the_ordinary_splash(
    qtbot, tmp_path, monkeypatch, missing_asset
):
    from napari_vipp.ui import startup_splash

    if missing_asset == "frame":
        monkeypatch.setattr(startup_splash, "_first_frame_pixmap", lambda: None)
    else:
        monkeypatch.setattr(startup_splash.PreparedTypeface, "load", lambda: None)
    splash = StartupSplash(
        profile="auto",
        version="test",
        log_path=tmp_path / "startup.log",
        defer_font_loading=True,
    )
    qtbot.addWidget(splash)
    assert splash._ui_ready
    assert splash.elapsed_label.text() == "Elapsed 0:00"
    assert splash._font_worker is None


@pytest.mark.parametrize("scale", [1, 1.25, 1.5, 2])
@pytest.mark.parametrize(
    "phase", [StartupPhase.STARTING, StartupPhase.READY, StartupPhase.FAILED]
)
def test_status_lettering_does_not_change_when_fonts_finish(
    qtbot, tmp_path, monkeypatch, scale, phase
):
    from napari_vipp.ui import startup_splash

    release = threading.Event()

    def prepare(done, cancelled):
        release.wait(5)
        done.set()

    monkeypatch.setattr(startup_splash, "_prepare_font_database", prepare)
    splash = StartupSplash(
        profile="auto",
        version="test",
        log_path=tmp_path / "log",
        defer_font_loading=True,
    )
    qtbot.addWidget(splash)

    def render(label):
        label.setFixedSize(556, 40)
        image = QImage(round(556 * scale), round(40 * scale), QImage.Format_RGBA8888)
        image.setDevicePixelRatio(scale)
        image.fill(QColor("#08151F"))
        painter = QPainter(image)
        label.render(painter, QPoint(), QRegion(), QWidget.DrawChildren)
        painter.end()
        return image

    try:
        splash.show()
        splash.update_snapshot(_snapshot(phase))
        before = render(splash._bootstrap_status)
        release.set()
        qtbot.waitUntil(lambda: splash._ui_ready)
        assert render(splash.status_label) == before
    finally:
        release.set()
        splash.finish_font_worker()
        splash.permit_close()


def _snapshot(phase: StartupPhase, *, error: str = "") -> StartupSnapshot:
    return StartupSnapshot(
        phase=phase,
        stage="loading_napari" if phase is not StartupPhase.FAILED else "failure",
        message="Test status",
        step=2,
        total_steps=6,
        error=error,
    )


def test_explicit_diagnostic_profiles_remain_visible(qtbot, tmp_path):
    splash = StartupSplash(
        profile=LaunchProfile.PREFER_GPU,
        version="1.2.3",
        log_path=tmp_path / "diagnostic log.txt",
    )
    qtbot.addWidget(splash)
    assert splash.windowTitle() == "Starting VIPP"
    assert splash.profile_spec.label == "Prefer GPU"
    assert splash.version_label.text() == "VIPP 1.2.3"
    assert "GPU" in splash.profile_description.text()
    assert "first CUDA start" in splash.note_label.text()

    cpu_splash = StartupSplash(
        profile=LaunchProfile.CPU,
        version="1.2.3",
        log_path=tmp_path / "cpu-startup.log",
    )
    qtbot.addWidget(cpu_splash)
    assert "CPU path" in cpu_splash.note_label.text()
    assert "CUDA" not in cpu_splash.note_label.text()


def test_default_splash_has_one_vipp_identity_without_automatic_badge(qtbot, tmp_path):
    splash = StartupSplash(
        profile="auto", version="1.2.3", log_path=tmp_path / "startup.log"
    )
    qtbot.addWidget(splash)
    splash.show()
    assert splash.version_label.text() == "VIPP 1.2.3"
    assert splash.findChild(QLabel, "VippProfileText") is None
    assert splash.profile_description.isHidden()
    assert not any(
        "automatic" in label.text().casefold()
        for label in splash.findChildren(QLabel)
        if label.isVisibleTo(splash)
    )


def test_first_paint_signal_runs_once_across_repaints_and_reshow(qtbot, tmp_path):
    splash = StartupSplash(
        profile="auto", version="test", log_path=tmp_path / "startup.log"
    )
    qtbot.addWidget(splash)
    paints = []
    splash.first_painted.connect(lambda: paints.append(True))
    assert paints == []
    splash.show()
    qtbot.waitUntil(lambda: paints == [True])
    splash.repaint()
    splash.hide()
    splash.show()
    QApplication.processEvents()
    assert paints == [True]


@pytest.mark.parametrize("profile", tuple(LaunchProfile))
def test_splash_stylesheet_parses_and_styles_profile_and_minimize_button(
    qtbot, qtlog, tmp_path, profile
):
    splash = StartupSplash(
        profile=profile, version="test", log_path=tmp_path / "startup.log"
    )
    qtbot.addWidget(splash)
    splash.show()
    QApplication.processEvents()

    stylesheet_errors = [
        record.message
        for record in qtlog.records
        if "stylesheet" in record.message.casefold()
        and "parse" in record.message.casefold()
    ]
    assert not stylesheet_errors, stylesheet_errors
    badge = splash.findChild(QLabel, "VippProfileText")
    if profile is LaunchProfile.AUTO:
        assert badge is None
    else:
        assert badge is not None
        assert badge.palette().color(QPalette.WindowText).name() == (
            splash.profile_spec.accent.lower()
        )
        assert badge.font().bold()
        assert badge.font().pixelSize() == 10
    # A malformed rule earlier in the shared sheet can also discard the new
    # minimize rule, even though the window still launches and minimizes.
    assert splash.minimize_button.font().pixelSize() == 22


def test_splash_timeout_and_failure_actions_are_explicit(qtbot, tmp_path):
    splash = StartupSplash(
        profile="cpu",
        version="development",
        log_path=tmp_path / "startup.log",
    )
    qtbot.addWidget(splash)

    splash.update_snapshot(_snapshot(StartupPhase.TIMED_OUT))
    assert splash.keep_waiting_button.isVisibleTo(splash)
    assert splash.hide_button.isVisibleTo(splash)
    assert not splash.close_button.isVisibleTo(splash)

    splash.update_snapshot(
        _snapshot(StartupPhase.FAILED, error="Example startup error")
    )
    assert splash.open_log_button.isVisibleTo(splash)
    assert splash.close_button.isVisibleTo(splash)
    assert splash.detail_label.text() == "Example startup error"


def test_splash_elapsed_time_and_real_milestone_progress(qtbot, tmp_path):
    splash = StartupSplash(
        profile="auto",
        version="development",
        log_path=tmp_path / "startup.log",
    )
    qtbot.addWidget(splash)
    splash.update_elapsed(3723)
    splash.update_snapshot(_snapshot(StartupPhase.STARTING))
    assert splash.elapsed_label.text() == "Elapsed 1:02:03"
    assert splash.stage_label.text() == "Step 2 of 6"
    assert splash.progress_bar.value() == 33


def test_splash_is_movable_minimizable_and_does_not_reposition_or_restore(
    qtbot, tmp_path
):
    splash = StartupSplash(profile="auto", version="test", log_path=tmp_path / "log")
    qtbot.addWidget(splash)
    assert splash.windowType() == Qt.Window
    assert not splash.windowFlags() & Qt.WindowStaysOnTopHint
    assert splash.windowFlags() & Qt.FramelessWindowHint
    assert not splash.windowFlags() & Qt.WindowTitleHint
    assert splash.minimize_button.isVisibleTo(splash)
    assert splash.minimize_button.accessibleName() == "Minimize startup window"
    assert "keep starting" in splash.minimize_button.toolTip()
    splash.show()
    assert not splash.minimize_button.geometry().intersects(
        splash.logo_label.geometry()
    )
    splash.move(splash.pos() + QPoint(30, 30))
    position = splash.pos()
    qtbot.mouseClick(splash.minimize_button, Qt.LeftButton)
    for phase in (
        StartupPhase.STARTING,
        StartupPhase.TIMED_OUT,
        StartupPhase.FAILED,
        StartupPhase.READY,
    ):
        splash.update_snapshot(_snapshot(phase))
        splash.update_elapsed(12)
        assert splash.isMinimized()
    splash.showNormal()
    assert splash.pos() == position
    splash.hide()
    splash.show()
    assert splash.pos() == position


def test_splash_window_close_requests_hide_not_startup_cancellation(qtbot, tmp_path):
    splash = StartupSplash(profile="auto", version="test", log_path=tmp_path / "log")
    qtbot.addWidget(splash)
    splash.show()
    with qtbot.waitSignal(splash.hide_requested):
        assert not splash.close()
    splash.permit_close()
    assert splash.close()


def _mouse_move(splash, local, global_position, buttons=Qt.LeftButton):
    event = QMouseEvent(
        QEvent.MouseMove,
        QPointF(local),
        QPointF(global_position),
        Qt.NoButton,
        buttons,
        Qt.NoModifier,
    )
    QApplication.sendEvent(splash, event)


@pytest.mark.parametrize(
    "surface", [None, "logo_label", "status_label", "progress_bar"]
)
def test_splash_drag_fallback_moves_content_without_recentring(
    qtbot, tmp_path, monkeypatch, surface
):
    splash = StartupSplash(profile="auto", version="test", log_path=tmp_path / "log")
    qtbot.addWidget(splash)
    monkeypatch.setattr(splash, "_start_system_move", lambda: False)
    splash.show()
    target = splash if surface is None else getattr(splash, surface)
    press = QPoint(12, 12) if surface is None else target.rect().center()
    global_press = target.mapToGlobal(press)
    initial_position = splash.pos()
    qtbot.mousePress(target, Qt.LeftButton, pos=press)
    assert splash._drag_offset == global_press - initial_position
    delta = QPoint(40, 25)
    _mouse_move(
        splash, splash.mapFromGlobal(global_press + delta), global_press + delta
    )
    assert splash.pos() == initial_position + delta
    qtbot.mouseRelease(splash, Qt.LeftButton)
    assert splash._drag_offset is None
    _mouse_move(splash, QPoint(100, 100), global_press + delta * 2, Qt.NoButton)
    splash.update_snapshot(_snapshot(StartupPhase.STARTING))
    assert splash.pos() == initial_position + delta


def test_splash_prefers_native_drag_and_ignores_non_left_drags(
    qtbot, tmp_path, monkeypatch
):
    splash = StartupSplash(profile="auto", version="test", log_path=tmp_path / "log")
    qtbot.addWidget(splash)
    calls = []
    monkeypatch.setattr(
        splash, "_start_system_move", lambda: calls.append(True) or True
    )
    splash.show()
    initial_position = splash.pos()
    qtbot.mousePress(splash, Qt.RightButton)
    assert not calls
    qtbot.mouseRelease(splash, Qt.RightButton)
    qtbot.mousePress(splash, Qt.LeftButton)
    assert calls == [True]
    assert splash._drag_offset is None
    _mouse_move(splash, QPoint(100, 100), initial_position + QPoint(100, 100))
    assert splash.pos() == initial_position
    qtbot.mouseRelease(splash, Qt.LeftButton)


def test_splash_controls_do_not_drag_or_cancel_startup(qtbot, tmp_path, monkeypatch):
    splash = StartupSplash(profile="auto", version="test", log_path=tmp_path / "log")
    qtbot.addWidget(splash)
    calls = []
    monkeypatch.setattr(
        splash, "_start_system_move", lambda: calls.append(True) or True
    )
    splash.hide_requested.connect(lambda: calls.append("hide"))
    splash.close_requested.connect(lambda: calls.append("close"))
    splash.show()
    splash.minimize_button.setFocus()
    qtbot.keyClick(splash.minimize_button, Qt.Key_Space)
    assert splash.isMinimized()
    assert not calls
    splash.showNormal()
    splash.update_snapshot(
        _snapshot(StartupPhase.FAILED, error="Copyable diagnostic text")
    )
    assert not splash.detail_label.testAttribute(Qt.WA_TransparentForMouseEvents)
    assert splash.detail_label.textInteractionFlags() & Qt.TextSelectableByMouse
    qtbot.mouseClick(splash.detail_label, Qt.LeftButton)
    with qtbot.waitSignal(splash.open_log_requested):
        qtbot.mouseClick(splash.open_log_button, Qt.LeftButton)
    assert not calls
