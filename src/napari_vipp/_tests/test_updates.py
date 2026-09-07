from __future__ import annotations

import json

import pytest
from packaging.version import Version
from qtpy.QtCore import QObject, Signal
from qtpy.QtGui import QFont
from qtpy.QtNetwork import QNetworkReply, QNetworkRequest
from qtpy.QtWidgets import QApplication

from napari_vipp.core.updates import (
    RELEASES_URL,
    current_release_url,
    installer_asset,
    newest_release,
    parse_releases,
)
from napari_vipp.ui import updates
from napari_vipp.ui.updates import UpdateController, UpdateDialog, VersionBadge


def release(version="0.16.0a1", *, assets=(), **extra):
    return {
        "tag_name": f"v{version}",
        "draft": False,
        "prerelease": Version(version).is_prerelease,
        "published_at": "2026-09-07",
        "assets": [
            {
                "name": name,
                "state": "uploaded",
                "browser_download_url": f"{RELEASES_URL}/download/v{version}/{name}",
            }
            for name in assets
        ],
        **extra,
    }


class MemorySettings:
    def __init__(self, values=None):
        self.values = values or {}

    def value(self, key, default=None):
        return self.values.get(key, default)

    def setValue(self, key, value):  # noqa: N802
        self.values[key] = value

    def sync(self):
        pass


class Reply(QObject):
    readyRead = Signal()
    finished = Signal()

    def __init__(self):
        super().__init__()
        self.data = b""
        self.status = 200
        self.code = QNetworkReply.NetworkError.NoError
        self.aborted = False

    def read(self, limit):
        data, self.data = self.data[:limit], self.data[limit:]
        return data

    def attribute(self, _attribute):
        return self.status

    def error(self):
        return self.code

    def abort(self):
        self.aborted = True
        self.code = QNetworkReply.NetworkError.OperationCanceledError
        self.finished.emit()

    def complete(self, data, status=200):
        self.data = data
        self.status = status
        self.finished.emit()


class Manager:
    def __init__(self):
        self.requests = []
        self.reply = None

    def get(self, request):
        self.requests.append(request)
        self.reply = Reply()
        return self.reply


@pytest.fixture
def controller(qapp):
    result = UpdateController("0.15.0a1", settings=MemorySettings(), manager=Manager())
    yield result
    result.shutdown()


def deliver(controller, payload, status=200):
    controller.check()
    controller.manager.reply.complete(json.dumps(payload).encode(), status)


def test_semantic_order_and_release_channels():
    releases = parse_releases(
        [
            release("0.9.0"),
            release("0.15.0a9"),
            release("0.15.0a10"),
            release("0.15.0", prerelease=True),
        ]
    )
    assert str(newest_release(releases, include_prereleases=True).version) == "0.15.0"
    assert str(newest_release(releases, include_prereleases=False).version) == "0.9.0"
    assert str(releases[1].version) == "0.15.0a10"


@pytest.mark.parametrize(
    "entry",
    [
        release(draft=True),
        release(published_at=None),
        release(tag_name="main"),
        release(tag_name="vgarbage"),
        release(tag_name="v0.16.0/evil"),
        release(tag_name="v0.16.0.dev1"),
        release(tag_name="v0.16.0+local"),
        {},
        None,
    ],
)
def test_unpublished_or_invalid_releases_are_ignored(entry):
    assert parse_releases([entry]) == ()


@pytest.mark.parametrize(
    "system,machine,name,checksum",
    [
        (
            "Windows",
            "AMD64",
            "VIPP-Setup-0.16.0a1-Windows-x86_64-UNSIGNED.exe",
            "SHA256SUMS-Windows-0.16.0a1.txt",
        ),
        (
            "Darwin",
            "arm64",
            "VIPP-0.16.0a1-macOS-arm64-UNSIGNED.pkg",
            "SHA256SUMS-macOS-arm64-0.16.0a1.txt",
        ),
        (
            "Darwin",
            "x86_64",
            "VIPP-0.16.0a1-macOS-x86_64.pkg",
            "SHA256SUMS-macOS-x86_64-0.16.0a1.txt",
        ),
    ],
)
def test_installer_requires_architecture_and_checksums(system, machine, name, checksum):
    info = parse_releases([release(assets=[name, checksum])])[0]
    assert installer_asset(info, system, machine) == (name, checksum)
    assert installer_asset(info, "Linux", machine) is None
    assert installer_asset(info, "Windows", "arm64") is None
    missing = parse_releases([release(assets=[name])])[0]
    assert installer_asset(missing, system, machine) is None


def test_downloads_cannot_be_redirected_to_another_owner():
    entry = release(assets=["installer.exe"])
    entry["assets"][0]["browser_download_url"] = "https://evil.example/installer.exe"
    parsed = parse_releases([entry])[0]
    assert not parsed.assets
    with pytest.raises(ValueError):
        parsed.asset_url("installer.exe")
    assert current_release_url("0.15.0a1") == RELEASES_URL + "/tag/v0.15.0a1"
    assert current_release_url("0.15.0+editable") == RELEASES_URL


def test_auto_check_is_throttled_and_does_not_open_dialog(
    controller, qtbot, monkeypatch
):
    monkeypatch.delenv(updates.DISABLE_AUTO_ENV)
    badge = VersionBadge("0.15.0a1", controller=controller)
    qtbot.addWidget(badge)
    controller.check_if_due()
    assert controller.checking and len(controller.manager.requests) == 1
    controller.check_if_due()
    controller.manager.reply.complete(json.dumps([release()]).encode())
    assert controller.available
    assert badge.property("updateAvailable") is True
    assert "↑" in badge.text()
    assert badge.dialog is None
    controller.check_if_due()
    assert len(controller.manager.requests) == 1
    assert not controller.deadline.isActive()


def test_preference_and_admin_switch_only_disable_automatic_checks(
    controller, monkeypatch
):
    controller.check_if_due()
    assert not controller.manager.requests  # test-wide environment switch
    monkeypatch.delenv(updates.DISABLE_AUTO_ENV)
    controller.set_automatic(False)
    controller.check_if_due()
    assert not controller.manager.requests
    controller.check()
    assert controller.checking


def test_request_is_bounded_and_sends_no_workflow_information(controller):
    controller.check()
    request = controller.manager.requests[0]
    assert request.url().toString() == updates.RELEASES_API
    assert request.transferTimeout() == 15000
    assert (
        request.attribute(QNetworkRequest.Attribute.RedirectPolicyAttribute)
        == QNetworkRequest.RedirectPolicy.ManualRedirectPolicy
    )
    assert request.rawHeader(b"User-Agent") == b"napari-vipp-update-check"
    assert controller.deadline.interval() == 20000


def test_cached_badge_survives_restart_and_offline_failure(controller, qapp):
    deliver(controller, [release()])
    restored = UpdateController(
        "0.15.0a1", settings=controller.settings, manager=Manager()
    )
    try:
        assert restored.available and restored.checked
        deliver(restored, {"message": "rate limited"}, status=403)
        assert restored.available and "request limit" in restored.error
        assert not restored.checking
    finally:
        restored.shutdown()


@pytest.mark.parametrize(
    "data,status",
    [
        (b"not json", 200),
        (b"{}", 200),
        (b"[]", 200),
        (b"redirect", 302),
        (b"offline", None),
    ],
)
def test_failed_check_is_not_reported_as_up_to_date(controller, data, status):
    controller.check()
    controller.manager.reply.complete(data, status)
    assert controller.error and not controller.checked and not controller.checking


def test_response_size_and_deadline_abort_safely(controller, monkeypatch):
    monkeypatch.setattr(updates, "MAX_RESPONSE_BYTES", 30)
    controller.check()
    reply = controller.manager.reply
    reply.data = b"x" * 100
    reply.readyRead.emit()
    assert reply.aborted and not controller.checking
    assert "too large" in controller.error
    controller.check()
    reply = controller.manager.reply
    controller.deadline.timeout.emit()
    assert reply.aborted and not controller.checking and controller.error


def test_shutdown_aborts_without_emitting_ui_result(controller):
    controller.check()
    changed = []
    controller.changed.connect(lambda: changed.append(True))
    reply = controller.manager.reply
    controller.shutdown()
    assert reply.aborted and not changed
    assert not controller.timer.isActive()
    controller.check()
    assert len(controller.manager.requests) == 1


def test_native_reply_can_return_none_after_final_drain(controller, monkeypatch):
    controller.check()
    reply = controller.manager.reply
    reply.data = json.dumps([release()]).encode()
    reply.readyRead.emit()
    monkeypatch.setattr(reply, "read", lambda _limit: None)
    reply.finished.emit()
    assert controller.checked and controller.available and not controller.error


def test_badge_click_and_context_actions_are_explicit(controller, qtbot, monkeypatch):
    deliver(controller, [release()])
    badge = VersionBadge("0.15.0a1", controller=controller)
    qtbot.addWidget(badge)
    opened = []
    monkeypatch.setattr(
        updates.QDesktopServices,
        "openUrl",
        lambda url: opened.append(url.toString()) or True,
    )
    menu = badge.make_context_menu()
    assert [a.text() for a in menu.actions()] == [
        "Release notes on GitHub",
        "Check for updates…",
    ]
    menu.actions()[0].trigger()
    assert opened == [RELEASES_URL + "/tag/v0.15.0a1"]
    badge.click()
    assert badge.dialog.isVisible()
    assert len(controller.manager.requests) == 1
    menu.actions()[1].trigger()
    assert controller.checking and len(controller.manager.requests) == 2
    badge.shutdown()


def test_dialog_download_is_user_initiated_and_points_to_exact_asset(
    controller, qtbot, monkeypatch
):
    name = "VIPP-Setup-0.16.0a1-Windows-x86_64-UNSIGNED.exe"
    checksum = "SHA256SUMS-Windows-0.16.0a1.txt"
    monkeypatch.setattr(updates.platform, "system", lambda: "Windows")
    monkeypatch.setattr(updates.platform, "machine", lambda: "AMD64")
    deliver(controller, [release(assets=[name, checksum])])
    opened = []
    dialog = UpdateDialog(controller, open_url=lambda url: opened.append(url) or True)
    qtbot.addWidget(dialog)
    dialog.show()
    assert not opened
    assert "unsigned" in dialog.instructions.text()
    qtbot.waitUntil(
        lambda: dialog.instructions.height()
        >= dialog.instructions.heightForWidth(dialog.instructions.width())
    )
    dialog.resize(440, 360)
    qtbot.waitUntil(lambda: dialog.scroll.verticalScrollBar().maximum() > 0)
    assert dialog.check_button.isVisible()
    dialog.download.click()
    assert opened == [controller.latest.asset_url(name)]
    dialog.checksums.click()
    assert opened[-1] == controller.latest.asset_url(checksum)
    dialog.prereleases.setChecked(False)
    assert not controller.available and dialog.download.isHidden()
    assert "No stable" in dialog.heading.text()


def test_no_update_for_equal_or_older_release_and_final_defaults_to_stable(qapp):
    controller = UpdateController(
        "0.16.0", settings=MemorySettings(), manager=Manager()
    )
    try:
        assert not controller.include_prereleases
        deliver(controller, [release("0.16.0"), release("0.17.0a1")])
        assert not controller.available
        controller.set_prereleases(True)
        assert controller.available
    finally:
        controller.shutdown()


@pytest.mark.parametrize("theme", ("dark", "light"))
@pytest.mark.parametrize("state", ("current", "installer", "offline"))
@pytest.mark.parametrize("font_size", (10, 14))
def test_update_dialog_is_compact_branded_and_scrollable(
    controller, qtbot, qapp, monkeypatch, tmp_path, theme, state, font_size,
):
    from napari._qt.qt_resources import get_stylesheet

    previous_font = qapp.font()
    qapp.setFont(QFont("Segoe UI", font_size))
    try:
        if state == "installer":
            monkeypatch.setattr(updates.platform, "system", lambda: "Windows")
            monkeypatch.setattr(updates.platform, "machine", lambda: "AMD64")
            deliver(controller, [release(assets=[
                "VIPP-Setup-0.16.0a1-Windows-x86_64-UNSIGNED.exe",
                "SHA256SUMS-Windows-0.16.0a1.txt",
            ])])
        else:
            deliver(controller, [release("0.15.0a1")])
            if state == "offline":
                deliver(controller, {}, status=403)
        dialog = UpdateDialog(controller)
        qtbot.addWidget(dialog)
        dialog.setStyleSheet(
            get_stylesheet(theme, extra_variables={"font_size": f"{font_size}pt"})
        )
        dialog.show()
        QApplication.processEvents()
        assert dialog.logo.pixmap() is not None
        assert not dialog.logo.pixmap().isNull()
        assert dialog.logo.accessibleName() == "VIPP logo"
        assert dialog.width() == 560
        if state == "current" and font_size == 10:
            assert dialog.height() < 480
            assert dialog.scroll.verticalScrollBar().maximum() == 0
        assert 0 < dialog.prereleases.y() - dialog.manual.geometry().bottom() <= 28
        for label in dialog.findChildren(updates._WrappedLabel):
            assert label.height() >= label.heightForWidth(label.width())
        assert dialog.grab().save(str(tmp_path / "updates.png"))
        dialog.resize(440, 320)
        qtbot.waitUntil(lambda: dialog.scroll.verticalScrollBar().maximum() > 0)
        assert dialog.scroll.horizontalScrollBar().maximum() == 0
        assert dialog.rect().contains(dialog.buttons.geometry())
        assert dialog.check_button.isVisible()
        controller.set_automatic(False)
        QApplication.processEvents()
        assert dialog.height() == 320  # Status changes must not undo user resizing.
        assert dialog.grab().save(str(tmp_path / "updates-narrow.png"))
    finally:
        qapp.setFont(previous_font)
