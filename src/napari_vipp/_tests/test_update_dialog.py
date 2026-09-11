from __future__ import annotations

import pytest
from qtpy.QtCore import QObject, Signal
from qtpy.QtWidgets import QMessageBox

from napari_vipp._tests.test_updates import (
    controller as controller,
)
from napari_vipp._tests.test_updates import (
    deliver,
    release,
)
from napari_vipp._tests.test_updates import (
    native_fonts_for_offscreen as native_fonts_for_offscreen,
)
from napari_vipp.core.update_install import UpdateInstallError
from napari_vipp.ui import update_dialog as ui


class Transfer(QObject):
    changed = Signal()

    def __init__(self):
        super().__init__()
        self.busy = False
        self.phase = ""
        self.received_bytes = 0
        self.total_bytes = None
        self.error = ""
        self.starts = []
        self.launches = 0
        self.closed = False

    def start(self, request):
        self.starts.append(request)
        self.busy = True
        self.phase = "downloading"
        self.changed.emit()

    def complete(self):
        self.busy = False
        self.phase = "ready"
        self.changed.emit()

    def open_installer(self):
        self.launches += 1
        self.phase = "opened"
        self.changed.emit()

    def cancel(self):
        self.busy = False
        self.phase = "cancelled"
        self.changed.emit()

    def shutdown(self):
        self.closed = True


@pytest.fixture
def managed_dialog(controller, qtbot, monkeypatch):
    monkeypatch.setattr(ui.platform, "system", lambda: "Windows")
    monkeypatch.setattr(ui.platform, "machine", lambda: "AMD64")
    deliver(
        controller,
        [
            release(
                assets=[
                    "VIPP-Setup-0.16.0a1-Windows-x86_64-UNSIGNED.exe",
                    "SHA256SUMS-Windows-0.16.0a1.txt",
                ]
            )
        ],
    )
    target = object()
    monkeypatch.setattr(ui, "installer_download_request", lambda r, t: (r, t))
    transfer = Transfer()
    dialog = ui.UpdateDialog(
        controller,
        target_provider=lambda: target,
        download_controller=transfer,
    )
    qtbot.addWidget(dialog)
    dialog.show()
    yield dialog
    dialog.shutdown()


def test_explicit_download_click_opens_verified_result_once(managed_dialog):
    d = managed_dialog
    assert d.update_button.isVisible()
    assert d.download.isHidden()
    assert not d.transfer.starts and not d.transfer.launches
    d.update_button.click()
    assert len(d.transfer.starts) == 1 and d.transfer.busy
    assert not d.update_button.isEnabled()
    d.transfer.complete()
    d.transfer.changed.emit()
    assert d.transfer.launches == 1
    assert d.progress_label.text() == "Setup is open"
    d.update_button.click()
    assert len(d.transfer.starts) == 1


def test_cached_error_is_not_headlined_as_fresh_update(managed_dialog):
    d = managed_dialog
    deliver(d.controller, {}, status=403)
    assert d.controller.available
    assert d.heading.text() == "Couldn’t check for updates"
    assert d.latest_caption.text() == "Last known release"
    assert "previous check" in d.summary.text()
    assert not d.update_button.isEnabled()
    d._start_update()
    assert not d.transfer.starts
    d.technical_toggle.click()
    assert d.technical_details.isVisible()


def test_download_error_and_cancel_do_not_open_setup(managed_dialog):
    d = managed_dialog
    d.update_button.click()
    d.transfer.busy = False
    d.transfer.phase = "failed"
    d.transfer.error = "The installer checksum did not match."
    d.transfer.changed.emit()
    assert not d.transfer.launches
    assert "checksum" in d.transfer_detail.text()
    d.update_button.click()
    d.cancel_download.click()
    # A late ready notification cannot restore the cancelled user intention.
    d.transfer.complete()
    assert not d.transfer.launches


@pytest.mark.parametrize("answer", [QMessageBox.Yes, QMessageBox.No])
def test_completion_during_close_question_waits_for_answer(
    managed_dialog,
    monkeypatch,
    answer,
):
    d = managed_dialog
    d.update_button.click()

    def question(*args):
        d.transfer.complete()
        assert d.transfer.launches == 0
        return answer

    monkeypatch.setattr(ui.QMessageBox, "question", question)
    d.reject()
    assert d.transfer.launches == (0 if answer == QMessageBox.Yes else 1)
    assert d.isVisible() == (answer == QMessageBox.No)


def test_shutdown_prevents_late_handoff(managed_dialog):
    d = managed_dialog
    d.update_button.click()
    d.shutdown()
    d.transfer.complete()
    assert d.transfer.closed and not d.transfer.launches


def test_changed_installation_is_an_actionable_error(managed_dialog, monkeypatch):
    d = managed_dialog

    def stale(*args):
        raise UpdateInstallError("The installation changed. Check again.")

    monkeypatch.setattr(ui, "installer_download_request", stale)
    d.update_button.click()
    assert not d.transfer.starts
    assert "installation changed" in d.summary.text()


def test_browser_alternative_remains_explicit(managed_dialog):
    d = managed_dialog
    opened = []
    d.open_url = lambda url: opened.append(url) or True
    d.browser_toggle.click()
    assert d.download.isVisible()
    d.download.click()
    assert opened == [
        d.controller.latest.asset_url("VIPP-Setup-0.16.0a1-Windows-x86_64-UNSIGNED.exe")
    ]
    assert not d.transfer.starts
