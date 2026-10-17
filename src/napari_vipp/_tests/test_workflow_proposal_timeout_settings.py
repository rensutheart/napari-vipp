"""Planning deadlines can be longer without weakening cancellation or bounds."""

from __future__ import annotations

import queue
import threading
import time
from types import SimpleNamespace

import pytest

from napari_vipp.core import workflow_proposal_codex as codex
from napari_vipp.core import workflow_proposal_model as local


@pytest.mark.parametrize("provider", ["codex", "local"])
def test_planning_timeout_defaults_to_ten_minutes(provider):
    settings = (
        codex.CodexSettings()
        if provider == "codex"
        else local.ModelSettings(model="qwen-local")
    )
    assert local.DEFAULT_PROPOSAL_TIMEOUT_SECONDS == 600
    assert local.MAX_PROPOSAL_TIMEOUT_SECONDS == 600
    assert settings.timeout_seconds == local.DEFAULT_PROPOSAL_TIMEOUT_SECONDS
    settings.validate()


@pytest.mark.parametrize("provider,minimum", [("codex", 5), ("local", 1)])
@pytest.mark.parametrize("use_maximum", [False, True])
def test_planning_timeout_preserves_minimum_and_accepts_ten_minutes(
    provider, minimum, use_maximum
):
    timeout = 600 if use_maximum else minimum
    settings = (
        codex.CodexSettings(timeout_seconds=timeout)
        if provider == "codex"
        else local.ModelSettings(model="qwen-local", timeout_seconds=timeout)
    )
    settings.validate()


@pytest.mark.parametrize("provider", ["codex", "local"])
@pytest.mark.parametrize(
    "value",
    [601, 600.01, float("nan"), float("inf"), -float("inf"), True, False, "600", None],
)
def test_planning_timeout_rejects_above_limit_nonfinite_and_wrong_types(
    provider, value
):
    settings = (
        codex.CodexSettings(timeout_seconds=value)
        if provider == "codex"
        else local.ModelSettings(model="qwen-local", timeout_seconds=value)
    )
    with pytest.raises(local.ModelProposalError, match="[Tt]imeout"):
        settings.validate()


@pytest.mark.parametrize("provider,minimum", [("codex", 5), ("local", 1)])
def test_planning_timeout_still_rejects_below_existing_minimum(provider, minimum):
    settings = (
        codex.CodexSettings(timeout_seconds=minimum - 0.01)
        if provider == "codex"
        else local.ModelSettings(model="qwen-local", timeout_seconds=minimum - 0.01)
    )
    with pytest.raises(local.ModelProposalError, match="[Tt]imeout"):
        settings.validate()


@pytest.mark.parametrize("cancel", [False, True])
def test_local_inflight_deadline_or_cancel_remains_prompt_without_network(
    monkeypatch, cancel
):
    """Inject a blocked connection; a longer setting cannot retain a late reply."""
    cancelled = threading.Event()
    released = threading.Event()
    worker_done = threading.Event()

    class BlockingConnection:
        def __init__(self):
            self.sock = SimpleNamespace(shutdown=lambda _: released.set())
            self.closed = False

        def connect(self):
            pass

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            if cancel:
                cancelled.set()
            assert released.wait(1), "Cancellation/deadline did not release IO."
            worker_done.set()
            return SimpleNamespace(status=200, read=lambda _: b"{}")

        def close(self):
            self.closed = True

    connection = BlockingConnection()
    monkeypatch.setattr(
        local.http.client, "HTTPConnection", lambda *args, **kwargs: connection
    )
    started = time.monotonic()
    with pytest.raises(
        local.ModelCancelled if cancel else local.ModelProposalError
    ) as caught:
        local._request_json(
            "http://127.0.0.1:11434",
            "ollama",
            "/chat",
            {},
            local.DEFAULT_PROPOSAL_TIMEOUT_SECONDS if cancel else 0.01,
            cancelled.is_set,
        )
    assert released.is_set() and connection.closed
    assert worker_done.wait(1)
    assert time.monotonic() - started < 1
    if not cancel:
        assert caught.value.diagnostic == local.SafeModelDiagnostic(
            "local", "request", "timeout"
        )


@pytest.mark.parametrize("cancel", [False, True])
def test_codex_inflight_deadline_or_cancel_remains_prompt_without_process(cancel):
    """Exercise the existing receive loop with no child process or model call."""
    rpc = object.__new__(codex._Rpc)
    rpc.abort_error = None
    rpc.cancelled = lambda: cancel
    rpc.messages = queue.Queue()
    started = time.monotonic()
    rpc.deadline = started + (
        local.DEFAULT_PROPOSAL_TIMEOUT_SECONDS if cancel else 0.01
    )
    with pytest.raises(
        local.ModelCancelled if cancel else local.ModelProposalError
    ) as caught:
        rpc.receive()
    assert time.monotonic() - started < 1
    if not cancel:
        assert caught.value.diagnostic == local.SafeModelDiagnostic(
            "codex", "request", "timeout"
        )
