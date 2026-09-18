"""Less-frequent setup and error dialogs share the ordinary footer policy."""

from functools import partial

import pytest
from qtpy.QtWidgets import QApplication

from napari_vipp import _startup_widget
from napari_vipp._tests.test_batch_bulk_edit_presentation import _dialog
from napari_vipp._tests.test_startup_splash import _snapshot
from napari_vipp.startup import StartupPhase
from napari_vipp.ui import batch_bulk_edit, reader_setup, startup_splash
from napari_vipp.ui.dialog_buttons import add_dialog_buttons


def _policy(monkeypatch, module, platform):
    monkeypatch.setattr(
        module, "add_dialog_buttons", partial(add_dialog_buttons, platform=platform)
    )


def _assert_order(action, dismiss, platform):
    QApplication.processEvents()
    assert action.isVisible() and dismiss.isVisible()
    assert (action.x() < dismiss.x()) == (platform == "win32")
    assert not dismiss.autoDefault()


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_bulk_edit_order_preserves_safe_default(qtbot, monkeypatch, platform):
    _policy(monkeypatch, batch_bulk_edit, platform)
    owner, dialog = _dialog(qtbot)
    dialog.show()
    _assert_order(dialog.apply_button, dialog.cancel_button, platform)
    assert not dialog.apply_button.autoDefault()
    dialog.cancel_button.click()
    assert owner.overrides() == ()


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_reader_setup_order_does_not_start_setup(
    qtbot, monkeypatch, tmp_path, platform
):
    _policy(monkeypatch, reader_setup, platform)
    monkeypatch.setattr(reader_setup.setup, "setup_blocker", lambda: None)
    dialog = reader_setup.ReaderSetupDialog("tiff", tmp_path, 0, 0)
    qtbot.addWidget(dialog)
    dialog.show()
    _assert_order(dialog.action, dialog.cancel, platform)
    dialog.cancel.click()
    assert dialog.worker is None
    assert dialog.phase == "review"


@pytest.mark.parametrize("platform", ["win32", "darwin"])
@pytest.mark.parametrize("phase", [StartupPhase.TIMED_OUT, StartupPhase.FAILED])
def test_splash_visible_action_pair_order(
    qtbot, monkeypatch, tmp_path, platform, phase
):
    _policy(monkeypatch, startup_splash, platform)
    splash = startup_splash.StartupSplash(
        profile="auto", version="test", log_path=tmp_path / "startup.log"
    )
    qtbot.addWidget(splash)
    splash.update_snapshot(_snapshot(phase, error="Synthetic failure"))
    splash.show()
    if phase is StartupPhase.FAILED:
        pair = splash.open_log_button, splash.close_button
    else:
        pair = splash.keep_waiting_button, splash.hide_button
    _assert_order(*pair, platform)
    splash.permit_close()


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_startup_failure_order_does_not_retry(qtbot, monkeypatch, platform):
    _policy(monkeypatch, _startup_widget, platform)
    host = _startup_widget.VippStartupWidget(object())
    qtbot.addWidget(host)
    host._show_error("Synthetic error", "No startup worker was launched.")
    host.show()
    _assert_order(host.retry_button, host.close_button, platform)
    assert host.startup_state is _startup_widget.StartupState.ERROR
