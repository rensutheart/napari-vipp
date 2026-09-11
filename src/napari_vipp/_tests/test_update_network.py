"""Updater startup and transport regressions; no real network or user settings."""

from __future__ import annotations

import json

import pytest
from qtpy.QtCore import Signal
from qtpy.QtNetwork import QNetworkReply

from napari_vipp._tests.test_updates import MemorySettings, Reply, release
from napari_vipp.ui import updates


class DiagnosticReply(Reply):
    sslErrors = Signal(object)

    def __init__(self):
        super().__init__()
        self.description = ""
        self.headers = {}

    def errorString(self):  # noqa: N802
        return self.description

    def rawHeader(self, name):  # noqa: N802
        return self.headers.get(name, b"")

    def fail(self, code, *, status=None, data=b"", description=""):
        self.code, self.description = code, description
        self.complete(data, status)


class DiagnosticManager:
    def __init__(self):
        self.requests = []
        self.reply = None

    def get(self, request):
        self.requests.append(request)
        self.reply = DiagnosticReply()
        return self.reply


@pytest.fixture
def make_controller(qapp):
    controllers = []

    def make(settings=None):
        controller = updates.UpdateController(
            "0.15.0a2", settings=settings or MemorySettings(),
            manager=DiagnosticManager(),
        )
        controllers.append(controller)
        return controller

    yield make
    for controller in controllers:
        controller.shutdown()


def finish_retry(controller):
    controller.retry_timer.stop()
    controller.retry_timer.timeout.emit()


def test_each_launch_checks_even_after_recent_persisted_attempt(
    make_controller, monkeypatch,
):
    monkeypatch.delenv(updates.DISABLE_AUTO_ENV)
    settings = MemorySettings({
        "updates/last-attempt": updates.time.time(),
        "updates/last-success": updates.time.time() - 5,
        "updates/cache-v1": json.dumps([release("0.15.0a3")]),
    })
    first = make_controller(settings)
    assert first.checked and first.available and not first.checked_this_session
    first.start_timer.timeout.emit()
    first.start_timer.timeout.emit()
    assert len(first.manager.requests) == 1
    first.manager.reply.complete(json.dumps([release("0.15.0a4")]).encode())
    assert first.checked_this_session
    assert str(first.latest.version) == "0.15.0a4"
    assert first.last_success == settings.value("updates/last-success")
    first.timer.timeout.emit()
    assert len(first.manager.requests) == 1
    second = make_controller(settings)
    assert not second.checked_this_session
    second.start_timer.timeout.emit()
    assert len(second.manager.requests) == 1


@pytest.mark.parametrize("disabled_by", ["preference", "environment"])
def test_startup_honors_opt_out_but_explicit_checks_still_work(
    make_controller, monkeypatch, disabled_by,
):
    monkeypatch.delenv(updates.DISABLE_AUTO_ENV)
    controller = make_controller()
    if disabled_by == "preference":
        controller.set_automatic(False)
    else:
        monkeypatch.setenv(updates.DISABLE_AUTO_ENV, "1")
    controller.start_timer.timeout.emit()
    assert not controller.manager.requests
    controller.check()
    assert len(controller.manager.requests) == 1


def test_manual_check_before_startup_does_not_duplicate_request(
    make_controller, monkeypatch,
):
    monkeypatch.delenv(updates.DISABLE_AUTO_ENV)
    controller = make_controller()
    controller.check()
    controller.manager.reply.complete(json.dumps([release()]).encode())
    controller.start_timer.timeout.emit()
    assert len(controller.manager.requests) == 1


def test_session_throttle_uses_monotonic_clock(make_controller, monkeypatch):
    monkeypatch.delenv(updates.DISABLE_AUTO_ENV)
    monkeypatch.setattr(updates.time, "monotonic", lambda: 100)
    controller = make_controller()
    controller.check_if_due()
    controller.manager.reply.complete(json.dumps([release()]).encode())
    monkeypatch.setattr(updates.time, "monotonic", lambda: 100 + updates.CHECK_INTERVAL)
    controller.timer.timeout.emit()
    assert len(controller.manager.requests) == 2


@pytest.mark.parametrize(
    "code,status,kind",
    [
        (QNetworkReply.NetworkError.RemoteHostClosedError, None, "connection"),
        (QNetworkReply.NetworkError.TemporaryNetworkFailureError, None, "connection"),
        (QNetworkReply.NetworkError.HostNotFoundError, None, "dns"),
        (QNetworkReply.NetworkError.TimeoutError, None, "timeout"),
        (QNetworkReply.NetworkError.ServiceUnavailableError, 503, "server"),
    ],
)
def test_transient_failure_retries_once_with_same_total_deadline(
    make_controller, code, status, kind,
):
    controller = make_controller()
    controller.check()
    deadline_id = controller.deadline.timerId()
    old_reply = controller.manager.reply
    old_reply.fail(code, status=status, description="Network disconnected")
    assert controller.checking and controller.retry_timer.isActive()
    assert not controller.error
    controller.check()
    assert len(controller.manager.requests) == 1
    finish_retry(controller)
    assert controller.deadline.timerId() == deadline_id
    assert len(controller.manager.requests) == 2
    # Old signals must not finish or consume the new reply.
    old_reply.readyRead.emit()
    old_reply.finished.emit()
    assert controller.reply is controller.manager.reply
    controller.manager.reply.fail(code, status=status, description="Still unavailable")
    assert not controller.checking and not controller.retry_timer.isActive()
    assert controller.error_kind == kind
    assert "Still unavailable" in controller.error_details
    assert len(controller.manager.requests) == 2


def test_retry_success_is_fresh_and_keeps_response_bytes_separate(make_controller):
    controller = make_controller()
    controller.check()
    controller.manager.reply.fail(
        QNetworkReply.NetworkError.RemoteHostClosedError, data=b"partial JSON",
    )
    finish_retry(controller)
    controller.manager.reply.complete(json.dumps([release("0.15.0a4")]).encode())
    assert controller.checked_this_session and not controller.error
    assert str(controller.latest.version) == "0.15.0a4"
    assert not controller.deadline.isActive()


def test_total_deadline_during_retry_wait_does_not_start_another_request(
    make_controller,
):
    controller = make_controller()
    controller.check()
    controller.manager.reply.fail(QNetworkReply.NetworkError.RemoteHostClosedError)
    controller.deadline.timeout.emit()
    assert not controller.checking and not controller.retry_timer.isActive()
    assert controller.error_kind == "timeout"
    controller.retry_timer.timeout.emit()
    assert len(controller.manager.requests) == 1


def test_shutdown_during_retry_wait_is_silent_and_final(make_controller):
    controller = make_controller()
    controller.check()
    controller.manager.reply.fail(QNetworkReply.NetworkError.RemoteHostClosedError)
    changed = []
    controller.changed.connect(lambda: changed.append(True))
    controller.shutdown()
    controller.retry_timer.timeout.emit()
    assert not controller.checking and not changed
    assert len(controller.manager.requests) == 1


@pytest.mark.parametrize(
    "code,status,data,kind,phrase",
    [
        (QNetworkReply.NetworkError.SslHandshakeFailedError, None, b"", "tls", "HTTPS"),
        (
            QNetworkReply.NetworkError.ProxyAuthenticationRequiredError,
            None, b"", "proxy", "proxy",
        ),
        (
            QNetworkReply.NetworkError.ContentAccessDenied,
            403, b"{}", "access", "HTTP 403",
        ),
        (
            QNetworkReply.NetworkError.ContentAccessDenied,
            403, b'{"message":"API rate limit exceeded"}',
            "rate_limit", "request limit",
        ),
        (
            QNetworkReply.NetworkError.ContentAccessDenied,
            429, b"", "rate_limit", "request limit",
        ),
        (QNetworkReply.NetworkError.NoError, 302, b"", "redirect", "redirected"),
        (
            QNetworkReply.NetworkError.UnknownNetworkError,
            None, b"", "network", "GitHub",
        ),
    ],
)
def test_non_transient_errors_are_actionable_without_retry(
    make_controller, code, status, data, kind, phrase,
):
    settings = MemorySettings({"updates/cache-v1": json.dumps([release("0.15.0a3")])})
    controller = make_controller(settings)
    controller.check()
    controller.manager.reply.fail(
        code, status=status, data=data,
        description="Failed https://user:password@proxy.example\nconnect",
    )
    assert not controller.checking and not controller.retry_timer.isActive()
    assert controller.error_kind == kind and phrase in controller.error
    assert code.name in controller.error_details
    assert "password" not in controller.error_details
    assert "\n" not in controller.error_details
    assert controller.available and str(controller.latest.version) == "0.15.0a3"
    assert not controller.checked_this_session
    assert len(controller.manager.requests) == 1


def test_ssl_error_details_are_retained_without_ignoring_certificates(make_controller):
    class CertificateError:
        def errorString(self):  # noqa: N802
            return "The certificate has expired"

    controller = make_controller()
    controller.check()
    controller.manager.reply.sslErrors.emit([CertificateError()])
    controller.manager.reply.fail(QNetworkReply.NetworkError.SslHandshakeFailedError)
    assert controller.error_kind == "tls"
    assert "certificate has expired" in controller.error_details
    assert "system clock" in controller.error


def test_missing_qt_tls_has_installation_guidance(make_controller, monkeypatch):
    monkeypatch.setattr(updates.QSslSocket, "supportsSsl", lambda: False)
    controller = make_controller()
    controller.check()
    controller.manager.reply.fail(QNetworkReply.NetworkError.ProtocolUnknownError)
    assert controller.error_kind == "tls"
    assert "repair" in controller.error and "unavailable" in controller.error


def test_failed_refresh_preserves_last_success_but_clears_freshness(make_controller):
    controller = make_controller()
    controller.check()
    controller.manager.reply.complete(json.dumps([release("0.15.0a3")]).encode())
    success_time = controller.last_success
    assert controller.checked_this_session
    controller.check()
    controller.manager.reply.complete(b"not json")
    assert controller.checked and not controller.checked_this_session
    assert controller.last_success == success_time
    assert controller.error_kind == "invalid_response"
    assert str(controller.latest.version) == "0.15.0a3"


def test_size_abort_during_finished_is_not_reentrant(make_controller, monkeypatch):
    monkeypatch.setattr(updates, "MAX_RESPONSE_BYTES", 4)
    controller = make_controller()
    controller.check()
    changed = []
    controller.changed.connect(lambda: changed.append(True))
    controller.manager.reply.complete(b"too much data")
    assert controller.error_kind == "response_size"
    assert not controller.checking and not controller.retry_timer.isActive()
    assert changed == [True]
