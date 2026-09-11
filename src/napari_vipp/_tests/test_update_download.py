from __future__ import annotations

import threading
from types import SimpleNamespace

from napari_vipp.core.update_install import DownloadProgress, UpdateDownloadCancelled
from napari_vipp.ui.update_download import UpdateDownloadController


def test_ready_never_launches_without_explicit_handoff(qtbot):
    request = object()
    result = SimpleNamespace(request=request)
    launches = []

    def download(snapshot, *, progress, cancelled):
        assert snapshot is request
        progress(DownloadProgress("downloading", 10, 20))
        progress(DownloadProgress("verifying", 20, 20))
        return result

    controller = UpdateDownloadController(downloader=download, launcher=launches.append)
    controller.start(request)
    qtbot.waitUntil(lambda: controller.phase == "ready", timeout=2000)
    assert not controller.busy and controller.verified is result
    assert controller.received_bytes == controller.total_bytes == 20
    assert not launches
    controller.open_installer()
    controller.open_installer()
    assert controller.phase == "opened" and launches == [result]
    controller.shutdown()


def test_cancel_is_not_a_failure_or_handoff(qtbot):
    entered = threading.Event()

    def download(request, *, progress, cancelled):
        entered.set()
        while not cancelled():
            entered.wait(0.001)
        raise UpdateDownloadCancelled()

    controller = UpdateDownloadController(downloader=download)
    controller.start(object())
    qtbot.waitUntil(entered.is_set, timeout=2000)
    controller.cancel()
    qtbot.waitUntil(lambda: not controller.busy, timeout=2000)
    assert controller.phase == "cancelled"
    assert not controller.error and controller.verified is None
    controller.shutdown()


def test_failure_and_launch_failure_do_not_retry(qtbot):
    request = object()
    attempts = []

    def denied(verified):
        attempts.append(verified)
        raise OSError("Setup blocked")

    controller = UpdateDownloadController(
        downloader=lambda *args, **kwargs: SimpleNamespace(request=request),
        launcher=denied,
    )
    controller.start(request)
    qtbot.waitUntil(lambda: controller.phase == "ready", timeout=2000)
    controller.open_installer()
    controller.open_installer()
    assert controller.phase == "failed" and controller.error == "Setup blocked"
    assert len(attempts) == 1
    controller.shutdown()


def test_shutdown_ignores_late_completion(qtbot):
    proceed = threading.Event()
    finished = threading.Event()
    request = object()

    def download(*args, **kwargs):
        proceed.wait(2)
        finished.set()
        return SimpleNamespace(request=request)

    controller = UpdateDownloadController(downloader=download)
    changed = []
    controller.changed.connect(lambda: changed.append(controller.phase))
    controller.start(request)
    controller.shutdown()
    proceed.set()
    qtbot.waitUntil(finished.is_set, timeout=2000)
    qtbot.wait(20)
    assert changed == ["downloading"]
    assert controller.verified is None


def test_ready_request_mismatch_never_launches(qtbot):
    launches = []
    controller = UpdateDownloadController(
        downloader=lambda *args, **kwargs: SimpleNamespace(request=object()),
        launcher=launches.append,
    )
    controller.start(object())
    qtbot.waitUntil(lambda: controller.phase == "ready", timeout=2000)
    controller.open_installer()
    assert controller.phase == "failed" and not launches
    controller.shutdown()


def test_cancel_after_worker_return_before_queued_delivery_discards(qtbot):
    request = object()
    verified = SimpleNamespace(request=request)
    discarded = []
    controller = UpdateDownloadController(discard=discarded.append)
    controller.request = request
    controller._generation = 1
    controller.busy = True
    controller._artifacts[1] = verified
    controller.cancel()
    controller._on_finished((1, verified, "", False))
    assert controller.phase == "cancelled"
    assert controller.verified is None
    assert discarded == [verified]
    controller._on_finished((1, verified, "", False))
    assert controller.phase == "cancelled" and discarded == [verified]
    controller.shutdown()


def test_cancel_during_worker_return_discards_in_worker(qtbot):
    request = object()
    verified = SimpleNamespace(request=request)
    discarded = []
    controller = None

    def download(*args, **kwargs):
        controller._cancellation.set()
        return verified

    controller = UpdateDownloadController(downloader=download, discard=discarded.append)
    controller.start(request)
    qtbot.waitUntil(lambda: controller.phase == "cancelled", timeout=2000)
    assert discarded == [verified] and controller.verified is None
    controller.shutdown()


def test_owner_shutdown_discards_ready_but_not_opened_artifact(qtbot):
    from qtpy.QtCore import QObject

    request = object()
    verified = SimpleNamespace(request=request)
    discarded = []
    parent = QObject()
    controller = UpdateDownloadController(
        parent,
        downloader=lambda *a, **k: verified,
        launcher=lambda item: None,
        discard=discarded.append,
    )
    controller.start(request)
    qtbot.waitUntil(lambda: controller.phase == "ready", timeout=2000)
    controller.shutdown()
    parent.deleteLater()
    qtbot.waitUntil(lambda: bool(discarded), timeout=2000)
    assert discarded == [verified]

    controller = UpdateDownloadController(
        downloader=lambda *a, **k: verified,
        launcher=lambda item: None,
        discard=discarded.append,
    )
    controller.start(request)
    qtbot.waitUntil(lambda: controller.phase == "ready", timeout=2000)
    controller.open_installer()
    controller._on_finished((controller._generation, verified, "", False))
    assert controller.phase == "opened"
    controller.shutdown()
    assert discarded == [verified]
