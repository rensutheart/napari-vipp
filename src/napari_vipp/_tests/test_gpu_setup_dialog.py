from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from qtpy.QtCore import QPoint, QRect, Qt
from qtpy.QtGui import QCloseEvent, QColor, QPalette
from qtpy.QtWidgets import QApplication, QMainWindow, QPlainTextEdit, QWidget

from napari_vipp.installer.frontend import (
    InstallationCancelled,
    InstallOutcome,
    InstallSizeEstimate,
    PreparedInstall,
    ProgressUpdate,
    TargetKind,
    TrackChoice,
)
from napari_vipp.installer.models import ComputeTrack
from napari_vipp.ui.gpu_setup_dialog import GpuSetupDialog
from napari_vipp.ui.palette_roles import theme_colors


class _QueuedWorker:
    def __init__(self, target, queue):
        self.target = target
        self.queue = queue

    def start(self):
        self.queue.append(self.target)


class _Backend:
    def __init__(self, target: Path):
        self.prepared = PreparedInstall(
            kind=TargetKind.NEW,
            target=target,
            release_version="0.16.0a3",
            track=ComputeTrack.CUDA13,
            plain_summary="Install the same VIPP build with NVIDIA GPU components.",
            technical_details="Exact local wheel verified. No active packages changed.",
            payload="transaction",
            size_estimate=InstallSizeEstimate(1536 * 1024**2, 5 * 1024**3, 7 * 1024**3),
        )
        self.inspected = []
        self.applied = []
        self.opened = []
        self.inspect_error = None
        self.apply_error = None

    def inspect(self, selection, *, progress, cancellation, repair=False):
        self.inspected.append(selection)
        progress(ProgressUpdate("checking", "Checking hardware…"))
        if self.inspect_error:
            raise self.inspect_error
        return self.prepared

    def apply(self, prepared, *, confirmed, progress, cancellation):
        self.applied.append((prepared, confirmed))
        progress(ProgressUpdate("installing", "Downloading GPU components…"))
        if cancellation.is_set():
            raise InstallationCancelled(details="Temporary environment removed.")
        if self.apply_error:
            raise self.apply_error
        progress(ProgressUpdate("verifying", "GPU verification passed.", 1, 1))
        return InstallOutcome(
            launcher=prepared.target / "Scripts" / "vipp-app.exe",
            message="GPU verification passed.",
            technical_details="CUDA available. Qualified execution passed.",
        )

    def open_vipp(self, launcher):
        self.opened.append(launcher)


def _make(qtbot, tmp_path, *, parent=None):
    backend = _Backend(tmp_path / "separate-gpu")
    queue = []
    dialog = GpuSetupDialog(
        parent,
        backend=backend,
        worker_factory=lambda target: _QueuedWorker(target, queue),
    )
    qtbot.addWidget(dialog)
    return dialog, backend, queue


def _finish_next(queue):
    queue.pop(0)()
    QApplication.processEvents()


def test_review_requires_explicit_install_and_launch_requires_verification(
    qtbot, tmp_path
):
    dialog, backend, queue = _make(qtbot, tmp_path)
    assert dialog.running
    assert not dialog.action_button.isEnabled()
    assert backend.applied == []
    _finish_next(queue)

    assert not dialog.running
    assert backend.inspected[0].track is TrackChoice.CUDA13
    assert not backend.inspected[0].create_desktop_shortcut
    assert str(backend.prepared.target) in dialog.location.text()
    assert "1.5 GiB" in dialog.size_label.text()
    assert "separate" in dialog.description.text()
    assert "new window" in dialog.work_note.text()
    assert "Save and reopen" in dialog.work_note.text()
    assert dialog.action_button.text() == "Install GPU support"
    assert backend.applied == []
    assert backend.opened == []

    dialog.action_button.click()
    dialog.action_button.click()
    assert len(queue) == 1  # Repeated click cannot start another transaction.
    assert dialog.running
    assert backend.opened == []
    _finish_next(queue)

    assert backend.applied == [(backend.prepared, True)]
    assert not dialog.running
    assert dialog.action_button.text() == "Open VIPP with GPU"
    assert backend.opened == []  # Successful setup does not launch automatically.
    dialog.action_button.click()
    assert backend.opened == [backend.prepared.target / "Scripts" / "vipp-app.exe"]


def test_cancel_read_only_check_never_installs_and_can_retry(qtbot, tmp_path):
    dialog, backend, queue = _make(qtbot, tmp_path)
    dialog.cancel_button.click()
    assert not dialog.cancel_button.isEnabled()
    _finish_next(queue)
    assert not dialog.running
    assert backend.applied == []
    assert dialog.action_button.text() == "Try again"
    dialog.action_button.click()
    _finish_next(queue)
    assert dialog.action_button.text() == "Install GPU support"
    assert len(backend.inspected) == 2


@pytest.mark.parametrize("kind", [TargetKind.CURRENT, TargetKind.NEWER])
def test_verified_current_can_open_without_reinstall_but_newer_cannot(
    qtbot, tmp_path, kind
):
    dialog, backend, queue = _make(qtbot, tmp_path)
    launcher = backend.prepared.target / "Scripts" / "vipp-app.exe"
    backend.prepared = replace(backend.prepared, kind=kind, launcher=launcher)
    _finish_next(queue)
    assert backend.applied == []
    assert backend.opened == []
    if kind is TargetKind.CURRENT:
        assert dialog.action_button.text() == "Open VIPP with GPU"
        dialog.action_button.click()
        assert backend.opened == [launcher]
    else:
        assert not dialog.action_button.isEnabled()
        dialog._action()
        assert backend.opened == []


def test_close_during_install_waits_for_cancellation_and_rollback(qtbot, tmp_path):
    dialog, backend, queue = _make(qtbot, tmp_path)
    dialog.show()
    _finish_next(queue)
    dialog.action_button.click()
    dialog.close()
    assert dialog.running
    assert dialog.isVisible()
    _finish_next(queue)
    qtbot.waitUntil(lambda: not dialog.isVisible())
    assert not dialog.running
    assert backend.opened == []
    assert "Temporary environment removed" in dialog.details.toPlainText()


def test_escape_uses_safe_cancel_instead_of_destroying_worker(qtbot, tmp_path):
    dialog, backend, queue = _make(qtbot, tmp_path)
    dialog.show()
    qtbot.keyClick(dialog, Qt.Key_Escape)
    assert dialog.running
    assert dialog.isVisible()
    _finish_next(queue)
    qtbot.waitUntil(lambda: not dialog.isVisible())
    assert backend.applied == []


def test_parent_viewer_cannot_exit_during_setup_but_can_after_cancel(qtbot, tmp_path):
    viewer = QMainWindow()
    dock_content = QWidget(viewer)
    qtbot.addWidget(viewer)
    dialog, backend, queue = _make(qtbot, tmp_path, parent=dock_content)
    event = QCloseEvent()
    QApplication.sendEvent(viewer, event)
    assert not event.isAccepted()
    assert dialog.isVisible()
    assert "before closing VIPP" in dialog.message.text()
    dialog.cancel_setup()
    _finish_next(queue)
    event = QCloseEvent()
    QApplication.sendEvent(viewer, event)
    assert event.isAccepted()
    assert backend.applied == []


@pytest.mark.parametrize("phase", ["inspect", "apply"])
def test_failure_keeps_retry_available_and_never_offers_unverified_launch(
    qtbot, tmp_path, phase
):
    dialog, backend, queue = _make(qtbot, tmp_path)
    if phase == "inspect":
        backend.inspect_error = RuntimeError("Network offline")
    else:
        _finish_next(queue)
        backend.apply_error = RuntimeError("GPU verification failed")
        dialog.action_button.click()
    _finish_next(queue)
    assert not dialog.running
    assert dialog.action_button.text() == "Try again"
    assert backend.opened == []
    assert "RuntimeError" in dialog.details.toPlainText()
    backend.inspect_error = None
    backend.apply_error = None
    dialog.action_button.click()
    _finish_next(queue)
    assert dialog.action_button.text() == "Install GPU support"


def test_factory_error_is_actionable_and_retryable_without_environment_changes(
    qtbot, tmp_path
):
    calls = []
    queue = []
    backend = _Backend(tmp_path / "gpu")

    def factory():
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError(
                "The exact VIPP wheel could not be found. Reinstall VIPP."
            )
        return backend

    dialog = GpuSetupDialog(
        backend_factory=factory,
        worker_factory=lambda target: _QueuedWorker(target, queue),
    )
    qtbot.addWidget(dialog)
    assert "exact VIPP wheel" in dialog.message.text()
    assert not dialog.running
    assert dialog.action_button.text() == "Try again"
    assert backend.inspected == []
    dialog.action_button.click()
    _finish_next(queue)
    assert len(calls) == 2
    assert dialog.action_button.text() == "Install GPU support"


def test_technical_disclosure_has_stable_label_and_no_setup_side_effects(
    qtbot, tmp_path
):
    dialog, backend, queue = _make(qtbot, tmp_path)
    _finish_next(queue)

    assert dialog.details_button.text() == "Technical details"
    assert dialog.details_button.isCheckable()
    assert not dialog.details_button.isChecked()
    assert dialog.details_button.accessibleName() == "Expand technical details"
    assert not dialog.details_button.icon().isNull()
    assert dialog.details_widget.isHidden()
    collapsed_icon = dialog.details_button.icon().cacheKey()

    dialog.details_button.click()

    assert dialog.details_button.text() == "Technical details"
    assert dialog.details_button.accessibleName() == "Collapse technical details"
    assert dialog.details_button.icon().cacheKey() != collapsed_icon
    assert dialog.details_widget.isVisibleTo(dialog)
    assert dialog.details.isVisibleTo(dialog)
    assert "Exact local wheel verified" in dialog.details.toPlainText()

    dialog.details_button.click()

    assert dialog.details_button.text() == "Technical details"
    assert dialog.details_widget.isHidden()
    assert dialog.details_button.accessibleName() == "Expand technical details"
    assert len(backend.inspected) == 1
    assert queue == []
    assert backend.applied == backend.opened == []


def test_technical_log_is_readonly_monospace_and_bounded(qtbot, tmp_path):
    dialog, backend, queue = _make(qtbot, tmp_path)
    _finish_next(queue)
    dialog.details_button.click()

    assert isinstance(dialog.details, QPlainTextEdit)
    assert dialog.details.isReadOnly()
    # Headless Windows may substitute fonts; fixedPitch verifies the intent.
    assert dialog.details.font().fixedPitch()
    assert dialog.details.maximumBlockCount() == 500
    assert dialog.details.maximumHeight() <= 300
    assert dialog.details.lineWrapMode() == QPlainTextEdit.WidgetWidth

    dialog.details.setPlainText("\n".join(str(index) for index in range(600)))

    assert dialog.details.blockCount() == 500
    assert dialog.details.toPlainText().splitlines()[0] == "100"
    assert backend.applied == backend.opened == []


def test_verified_existing_setup_has_ready_copy_without_download_estimate(
    qtbot, tmp_path
):
    dialog, backend, queue = _make(qtbot, tmp_path)
    launcher = backend.prepared.target / "Scripts" / "vipp-app.exe"
    backend.prepared = replace(
        backend.prepared, kind=TargetKind.CURRENT, launcher=launcher
    )
    _finish_next(queue)

    assert "ready" in dialog.heading.text().lower()
    assert dialog.action_button.text() == "Open VIPP with GPU"
    assert dialog.action_button.isEnabled()
    assert "Install NVIDIA GPU support" not in dialog.description.text()
    assert "new window" in dialog.description.text()
    assert str(backend.prepared.target) in dialog.location.text()
    assert dialog.location.isVisibleTo(dialog)
    assert not dialog.size_label.text()
    assert not dialog.size_label.isVisibleTo(dialog)
    assert backend.applied == backend.opened == []


def test_completed_setup_keeps_location_but_hides_future_download_estimate(
    qtbot, tmp_path
):
    dialog, backend, queue = _make(qtbot, tmp_path)
    _finish_next(queue)
    dialog.action_button.click()
    _finish_next(queue)

    assert dialog.action_button.text() == "Open VIPP with GPU"
    assert "Install NVIDIA GPU support" not in dialog.description.text()
    assert str(backend.prepared.target) in dialog.location.text()
    assert dialog.location.isVisibleTo(dialog)
    assert not dialog.size_label.text()
    assert not dialog.size_label.isVisibleTo(dialog)
    assert backend.applied == [(backend.prepared, True)]
    assert backend.opened == []


@pytest.mark.parametrize("failed", [False, True])
def test_long_target_and_error_scroll_without_hiding_fixed_footer(
    qtbot, tmp_path, failed
):
    dialog, backend, queue = _make(qtbot, tmp_path)
    backend.prepared = replace(
        backend.prepared,
        target=tmp_path / ("A long GPU installation directory " * 12) / "cuda13",
        technical_details="Diagnostic information for support. " * 200,
    )
    _finish_next(queue)
    if failed:
        backend.apply_error = RuntimeError("GPU verification failed. " * 80)
        dialog.action_button.click()
        _finish_next(queue)
    dialog.details_button.click()
    dialog.resize(560, 440)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.scroll_area.verticalScrollBar().maximum() > 0)

    footer_rect = QRect(
        dialog.footer_widget.mapTo(dialog, QPoint()), dialog.footer_widget.size()
    )
    assert dialog.height() <= 440
    assert dialog.width() <= 560
    assert dialog.rect().contains(footer_rect)
    assert dialog.scroll_area.geometry().bottom() < footer_rect.top()
    assert dialog.scroll_area.horizontalScrollBar().maximum() == 0
    for button in (dialog.action_button, dialog.cancel_button, dialog.close_button):
        assert dialog.footer_widget.isAncestorOf(button)
        assert not dialog.scroll_area.isAncestorOf(button)
        if button.isVisibleTo(dialog):
            rect = QRect(button.mapTo(dialog, QPoint()), button.size())
            assert footer_rect.contains(rect)
    assert backend.opened == []


def test_gpu_setup_styles_follow_live_palette_changes(qtbot, tmp_path):
    dialog, _backend, queue = _make(qtbot, tmp_path)
    _finish_next(queue)
    dark = QPalette()
    dark.setColor(QPalette.Base, QColor("#111827"))
    dark.setColor(QPalette.Window, QColor("#111827"))
    dark.setColor(QPalette.Text, QColor("#f8fafc"))
    dark.setColor(QPalette.WindowText, QColor("#f8fafc"))
    dialog.setPalette(dark)
    dark_style = dialog.scroll_area.styleSheet()

    assert theme_colors(dark).surface.name() in dark_style
    assert theme_colors(dark).text.name() in dark_style

    light = QPalette()
    light.setColor(QPalette.Base, QColor("#ffffff"))
    light.setColor(QPalette.Window, QColor("#ffffff"))
    light.setColor(QPalette.Text, QColor("#111827"))
    light.setColor(QPalette.WindowText, QColor("#111827"))
    dialog.setPalette(light)
    qtbot.waitUntil(lambda: dialog.scroll_area.styleSheet() != dark_style)

    assert theme_colors(light).surface.name() in dialog.scroll_area.styleSheet()
    assert theme_colors(light).text.name() in dialog.scroll_area.styleSheet()


def test_gpu_setup_styles_follow_parent_theme_changes(qtbot, tmp_path):
    parent = QWidget()
    qtbot.addWidget(parent)
    dark = QPalette()
    dark.setColor(QPalette.Base, QColor("#111827"))
    dark.setColor(QPalette.Window, QColor("#111827"))
    dark.setColor(QPalette.Text, QColor("#f8fafc"))
    dark.setColor(QPalette.WindowText, QColor("#f8fafc"))
    parent.setPalette(dark)
    dialog, backend, queue = _make(qtbot, tmp_path, parent=parent)
    _finish_next(queue)
    dark_style = dialog.scroll_area.styleSheet()
    assert theme_colors(dark).surface.name() in dark_style

    light = QPalette()
    light.setColor(QPalette.Base, QColor("#ffffff"))
    light.setColor(QPalette.Window, QColor("#ffffff"))
    light.setColor(QPalette.Text, QColor("#111827"))
    light.setColor(QPalette.WindowText, QColor("#111827"))
    parent.setPalette(light)
    qtbot.waitUntil(
        lambda: theme_colors(light).surface.name() in dialog.scroll_area.styleSheet()
    )

    assert dialog.scroll_area.styleSheet() != dark_style
    assert theme_colors(light).text.name() in dialog.scroll_area.styleSheet()
    assert theme_colors(light).surface.name() in dialog.footer_widget.styleSheet()
    assert len(backend.inspected) == 1
    assert backend.applied == backend.opened == []
    assert queue == []
