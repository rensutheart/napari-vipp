from __future__ import annotations

import importlib.util
import json
import threading
import time
from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from napari_vipp.core import workflow_proposal_model as model
from napari_vipp.core.metadata import (
    AxisMetadata,
    SourceMetadata,
    image_state_from_array,
)
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.workflow_proposals import (
    ProposalDialogue,
    capture_proposal_context,
    reference_proposal,
)


def context_fixture():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    data = np.ones((12, 16), dtype=np.float32)
    data.flags.writeable = False
    state = image_state_from_array(
        data,
        axes=tuple(AxisMetadata(name, "space", "um", scale=0.2) for name in "yx"),
        source_name="private subject",
        source=SourceMetadata(uri="C:/private/subject/image.tif"),
    )
    pipeline.outputs["input"] = data
    pipeline.output_states["input"] = state
    pipeline.nodes["input"].params["file_path"] = "C:/private/subject/image.tif"
    return pipeline, capture_proposal_context(pipeline, "input")


def ollama_reply(text):
    return {
        "message": {"content": text},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 100,
        "eval_count": 50,
    }


def fake_requests(monkeypatch, replies):
    calls = []
    replies = iter(replies)

    def request(base_url, kind, route, payload, timeout, cancelled=None):
        calls.append((route, deepcopy(payload)))
        if route == "/show":
            return {
                "details": {"format": "gguf"},
                "model_info": {"general.architecture": "qwen"},
            }
        return next(replies)

    monkeypatch.setattr(model, "_request_json", request)
    return calls


def test_local_connector_rejects_image_context_before_any_request(monkeypatch):
    pipeline, _ = context_fixture()
    context = capture_proposal_context(pipeline, "input", include_images=True)
    calls = fake_requests(monkeypatch, ())
    with pytest.raises(model.ModelProposalError, match="metadata only"):
        model.generate_proposal(
            context, "Inspect the structures.", model.ModelSettings(model="qwen-local")
        )
    assert calls == []


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1:11434",
        "http://example.com:11434",
        "http://192.168.1.2:11434",
        "http://0.0.0.0:11434",
        "http://user:password@localhost:11434",
        "http://localhost:11434?key=secret",
        "http://localhost:11434#fragment",
        "http://localhost:11434/evil",
        "http://127.0.0.1:0",
        "http://127.0.0.1:99999",
        "http://localhost :11434",
        "http://localhost./",
    ],
)
def test_non_loopback_or_decorated_urls_never_open_a_connection(url, monkeypatch):
    monkeypatch.setattr(
        model.http.client,
        "HTTPConnection",
        lambda *a, **k: pytest.fail("No connection allowed"),
    )
    with pytest.raises(model.ModelProposalError):
        model.discover_models(url, "ollama")


@pytest.mark.parametrize(
    "url,host,port",
    [
        ("http://localhost:11434", "127.0.0.1", 11434),
        ("http://127.0.0.2:11434", "127.0.0.2", 11434),
        ("http://[::1]:11434", "::1", 11434),
    ],
)
def test_loopback_urls_are_explicit_without_dns(url, host, port):
    assert model._endpoint(url, "ollama") == (host, port, "/api")


@pytest.mark.parametrize(
    "changes",
    [
        {"max_attempts": 3},
        {"max_attempts": True},
        {"timeout_seconds": 0},
        {"timeout_seconds": 601},
        {"temperature": float("nan")},
        {"temperature": 1.1},
        {"max_output_tokens": 999999},
        {"model": "qwen:cloud"},
        {"model": "../weights"},
        {"model": "https://model"},
    ],
)
def test_invalid_settings_are_rejected_without_transmission(changes, monkeypatch):
    _, context = context_fixture()
    calls = fake_requests(monkeypatch, [])
    settings = {"model": "qwen-local", **changes}
    with pytest.raises(model.ModelProposalError):
        model.generate_proposal(
            context, "Find bright objects.", model.ModelSettings(**settings)
        )
    assert calls == []


@pytest.mark.parametrize(
    "identifier", ["C:/private/model", "c:/weights/qwen:latest", "D:/qwen"]
)
def test_private_drive_paths_cannot_be_sent_as_model_identifiers(
    identifier, monkeypatch
):
    _, context = context_fixture()
    calls = fake_requests(monkeypatch, [])
    with pytest.raises(model.ModelProposalError, match="path-like"):
        model.generate_proposal(
            context,
            "Find bright objects.",
            model.ModelSettings(model=identifier),
        )
    assert calls == []


@pytest.mark.parametrize(
    "identifier", ["library/qwen3.8:27b", "organization/qwen-local:latest"]
)
def test_ordinary_ollama_namespace_model_tags_remain_supported(identifier):
    model.ModelSettings(model=identifier).validate()


def test_one_shot_preserves_graph_and_private_source_data(monkeypatch):
    pipeline, context = context_fixture()
    expected = reference_proposal(context)
    calls = fake_requests(monkeypatch, [ollama_reply(json.dumps(expected))])
    result = model.generate_proposal(
        context,
        "Find bright objects and review them.",
        model.ModelSettings(model="qwen-local"),
    )
    assert result.proposal == expected
    assert result.prepared.can_apply
    assert len(result.attempts) == 1 and result.attempts[0].valid
    assert result.attempts[0].input_tokens == 100
    assert len(pipeline.nodes) == 1
    assert not pipeline.outputs["input"].flags.writeable
    encoded = json.dumps(calls)
    for secret in (
        "subject",
        "private",
        "image.tif",
        "file_path",
        "source_name",
        "tool_calls",
        "Authorization",
    ):
        assert secret not in encoded
    payload = calls[-1][1]
    assert "images" not in payload and all(
        "images" not in message for message in payload["messages"]
    )
    assert payload["truncate"] is False and payload["think"] is False
    assert payload["options"]["num_ctx"] == 16384
    assert len(result.request_sha256) == 64


@pytest.mark.parametrize("kind", ["ollama", "openai-compatible"])
@pytest.mark.parametrize("repair", [False, True])
def test_local_dialogue_keeps_long_clarifications_separate_from_original_goal(
    monkeypatch, kind, repair
):
    pipeline, context = context_fixture()
    dialogue = ProposalDialogue(
        "  Find bright objects and review them.  ",
        tuple((f"Question {number}?", str(number) * 1800) for number in range(3)),
    )
    expected = reference_proposal(context)
    replies = (["not JSON"] if repair else []) + [json.dumps(expected)]
    if kind == "ollama":
        replies = [ollama_reply(text) for text in replies]
    else:
        replies = [
            {"choices": [{"message": {"content": text}, "finish_reason": "stop"}]}
            for text in replies
        ]
    calls = fake_requests(monkeypatch, replies)
    result = model.generate_proposal(
        context,
        dialogue,
        model.ModelSettings(kind=kind, model="qwen-local", max_attempts=2),
    )
    assert result.proposal == expected
    prompts = [payload for route, payload in calls if route != "/show"]
    assert len(prompts) == (2 if repair else 1)
    for payload in prompts:
        public = json.loads(payload["messages"][1]["content"])
        assert public["objective"] == dialogue.objective
        assert public["clarifications"] == [
            {"question": question, "answer": answer}
            for question, answer in dialogue.clarifications
        ]
        assert len(json.dumps(public["clarifications"])) > 4000
        system = payload["messages"][0]["content"]
        assert "clarification answers" in system and "untrusted data" in system
        assert all(answer not in system for _, answer in dialogue.clarifications)
        assert "C:/private" not in json.dumps(payload)
    assert len(pipeline.nodes) == 1 and not pipeline.outputs["input"].flags.writeable


@pytest.mark.parametrize("field", ["question", "answer"])
def test_dialogue_paths_are_rejected_before_any_local_request(monkeypatch, field):
    _, context = context_fixture()
    question = "Which structure?" if field == "answer" else "Open C:/private/input.tif"
    answer = (
        "Use C:/private/input.tif" if field == "answer" else "The bright structure."
    )
    dialogue = ProposalDialogue("Find objects.", ((question, answer),))
    calls = fake_requests(monkeypatch, [])
    with pytest.raises(model.ModelProposalError, match="Remove filesystem paths"):
        model.generate_proposal(
            context, dialogue, model.ModelSettings(model="qwen-local")
        )
    assert calls == []


def test_dialogue_request_byte_bound_still_precedes_any_local_request(monkeypatch):
    _, context = context_fixture()
    dialogue = ProposalDialogue(
        "Find objects.", (("Which structure?", "The bright structure. " * 50),)
    )
    monkeypatch.setattr(model, "MAX_HTTP_BYTES", 1024)
    calls = fake_requests(monkeypatch, [])
    with pytest.raises(model.ModelProposalError, match="bounded size limit"):
        model.generate_proposal(
            context, dialogue, model.ModelSettings(model="qwen-local")
        )
    assert calls == []


@pytest.mark.parametrize(
    "objective",
    [
        "Use C:\\private\\sample.tif",
        "Open /home/user/image.tif",
        "Use file:///secret/image.tif",
        "Read \\\\server\\share",
    ],
)
def test_paths_typed_into_objective_are_not_transmitted(objective, monkeypatch):
    _, context = context_fixture()
    calls = fake_requests(monkeypatch, [])
    with pytest.raises(model.ModelProposalError, match="Remove filesystem paths"):
        model.generate_proposal(
            context, objective, model.ModelSettings(model="qwen-local")
        )
    assert calls == []


def test_local_gguf_check_blocks_cloud_alias_before_any_prompt(monkeypatch):
    _, context = context_fixture()
    calls = []

    def request(*args):
        calls.append(args[2])
        return {
            "details": {"format": "gguf"},
            "model_info": {},
            "remote_host": "remote",
        }

    monkeypatch.setattr(model, "_request_json", request)
    with pytest.raises(model.ModelProposalError, match="locally installed GGUF"):
        model.generate_proposal(
            context, "Find bright objects.", model.ModelSettings(model="innocent-alias")
        )
    assert calls == ["/show"]


def test_one_validation_feedback_attempt_and_no_hidden_retry(monkeypatch):
    _, context = context_fixture()
    expected = reference_proposal(context)
    bad = deepcopy(expected)
    bad["nodes"][0]["operation_id"] = "run_arbitrary_code"
    calls = fake_requests(
        monkeypatch, [ollama_reply(json.dumps(bad)), ollama_reply(json.dumps(expected))]
    )
    result = model.generate_proposal(
        context,
        "Find bright objects.",
        model.ModelSettings(model="qwen-local", max_attempts=2),
    )
    assert [attempt.valid for attempt in result.attempts] == [False, True]
    assert [call[0] for call in calls] == ["/show", "/chat", "/chat"]
    assert len(calls[-1][1]["messages"]) == 4
    assert "final attempt" in calls[-1][1]["messages"][-1]["content"]


def test_feedback_does_not_echo_paths_from_untrusted_generated_text(monkeypatch):
    _, context = context_fixture()
    expected = reference_proposal(context)
    bad = deepcopy(expected)
    bad["nodes"][0]["operation_id"] = "/private/secret/image.tif"
    calls = fake_requests(
        monkeypatch, [ollama_reply(json.dumps(bad)), ollama_reply(json.dumps(expected))]
    )
    result = model.generate_proposal(
        context,
        "Find bright objects.",
        model.ModelSettings(model="qwen-local", max_attempts=2),
    )
    assert result.prepared.can_apply
    assert "/private/secret" not in json.dumps(calls[-1][1])


def test_invalid_final_attempt_preserves_logs_and_does_not_mutate_graph(monkeypatch):
    pipeline, context = context_fixture()
    calls = fake_requests(monkeypatch, [ollama_reply("{}"), ollama_reply("{}")])
    with pytest.raises(model.ModelProposalError) as caught:
        model.generate_proposal(
            context,
            "Find bright objects.",
            model.ModelSettings(model="qwen-local", max_attempts=2),
        )
    assert len(caught.value.attempts) == 2
    assert len(calls) == 3 and len(pipeline.nodes) == 1


def test_cancellation_before_show_and_after_progress_never_sends_prompt(monkeypatch):
    _, context = context_fixture()
    calls = fake_requests(monkeypatch, [])
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(model.ModelCancelled):
        model.generate_proposal(
            context,
            "Find bright objects.",
            model.ModelSettings(model="qwen-local"),
            cancelled.is_set,
        )
    assert calls == []
    cancelled.clear()
    with pytest.raises(model.ModelCancelled):
        model.generate_proposal(
            context,
            "Find bright objects.",
            model.ModelSettings(model="qwen-local"),
            cancelled.is_set,
            lambda _: cancelled.set(),
        )
    assert [call[0] for call in calls] == ["/show"]


def test_cancellation_while_response_arrives_discards_valid_candidate(monkeypatch):
    pipeline, context = context_fixture()
    expected = reference_proposal(context)
    cancelled = threading.Event()

    def request(base_url, kind, route, payload, timeout, cancel):
        if route == "/show":
            return {
                "details": {"format": "gguf"},
                "model_info": {"general.architecture": "qwen"},
            }
        cancelled.set()
        return ollama_reply(json.dumps(expected))

    monkeypatch.setattr(model, "_request_json", request)
    with pytest.raises(model.ModelCancelled):
        model.generate_proposal(
            context,
            "Find bright objects.",
            model.ModelSettings(model="qwen-local"),
            cancelled.is_set,
        )
    assert len(pipeline.nodes) == 1


def test_openai_compatible_provider_uses_same_contract_without_credentials(monkeypatch):
    _, context = context_fixture()
    expected = reference_proposal(context)
    replies = [
        {
            "choices": [
                {"message": {"content": json.dumps(expected)}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 90, "completion_tokens": 45},
        }
    ]
    calls = fake_requests(monkeypatch, replies)
    result = model.generate_proposal(
        context,
        "Find bright objects.",
        model.ModelSettings(
            kind="openai-compatible", base_url="http://localhost:1234/v1", model="qwen"
        ),
    )
    assert result.prepared.can_apply and calls[0][0] == "/chat/completions"
    assert "tools" not in calls[0][1] and calls[0][1]["response_format"] == {
        "type": "json_object"
    }


@pytest.mark.parametrize(
    "reply",
    [
        {"message": {"content": "{}"}, "done": False},
        {"message": {"content": "{}"}, "done": True, "done_reason": "length"},
        {"message": {"content": "{}", "tool_calls": [{}]}, "done": True},
        {"message": {"content": "\ud800"}, "done": True},
    ],
)
def test_incomplete_tool_or_invalid_unicode_response_never_validates(reply):
    with pytest.raises(model.ModelProposalError):
        model._response_text(reply, "ollama")


class DummySocket:
    def shutdown(self, how):
        pass


class DummyResponse:
    def __init__(self, status=200, raw=b"{}"):
        self.status, self.raw = status, raw

    def read(self, maximum):
        return self.raw[:maximum]


class DummyConnection:
    def __init__(self, response):
        self.response, self.sock, self.requests = response, DummySocket(), []

    def connect(self):
        pass

    def request(self, *args, **kwargs):
        self.requests.append((args, kwargs))

    def getresponse(self):
        return self.response

    def close(self):
        pass


@pytest.mark.parametrize(
    "status,raw,message",
    [
        (302, b"{}", "redirects"),
        (401, b"private secret error", "HTTP 401"),
        (200, b"[1]", "JSON object"),
        (200, b"not-json", "invalid JSON"),
        (200, b"x" * (model.MAX_HTTP_BYTES + 1), "size limit"),
    ],
    ids=["redirect", "status", "list", "invalid-json", "oversize"],
)
def test_direct_http_rejects_redirect_errors_and_unbounded_responses(
    status, raw, message, monkeypatch
):
    connection = DummyConnection(DummyResponse(status, raw))
    monkeypatch.setattr(model.http.client, "HTTPConnection", lambda *a, **k: connection)
    with pytest.raises(model.ModelProposalError, match=message):
        model._request_json("http://localhost:11434", "ollama", "/tags", None, 1)
    assert len(connection.requests) == 1


def test_inventory_is_bounded_and_filters_cloud_models(monkeypatch):
    monkeypatch.setattr(
        model,
        "_request_json",
        lambda *a: {
            "models": [
                {"name": "qwen-local"},
                {"name": "qwen:cloud"},
                {"name": "qwen-local"},
            ]
        },
    )
    assert model.discover_models("http://localhost:11434", "ollama") == ("qwen-local",)
    monkeypatch.setattr(
        model, "_request_json", lambda *a: {"models": [{"name": "qwen"}] * 201}
    )
    with pytest.raises(model.ModelProposalError, match="excessive"):
        model.discover_models("http://localhost:11434", "ollama")


def test_io_cancellation_shuts_socket_and_discards_late_reply(monkeypatch):
    cancelled, released = threading.Event(), threading.Event()
    connection = DummyConnection(DummyResponse())

    class BlockingSocket:
        def shutdown(self, how):
            released.set()

    connection.sock = BlockingSocket()

    def delayed_response():
        cancelled.set()
        assert released.wait(1)
        return DummyResponse()

    connection.getresponse = delayed_response
    monkeypatch.setattr(model.http.client, "HTTPConnection", lambda *a, **k: connection)
    with pytest.raises(model.ModelCancelled):
        model._request_json(
            "http://localhost:11434", "ollama", "/tags", None, 1, cancelled.is_set
        )
    assert released.is_set()


def test_io_deadline_discards_late_reply_and_closes_socket(monkeypatch):
    released = threading.Event()
    connection = DummyConnection(DummyResponse())

    class BlockingSocket:
        def shutdown(self, how):
            released.set()

    connection.sock = BlockingSocket()

    def delayed_response():
        assert released.wait(1)
        return DummyResponse()

    connection.getresponse = delayed_response
    monkeypatch.setattr(model.http.client, "HTTPConnection", lambda *a, **k: connection)
    started = time.monotonic()
    with pytest.raises(model.ModelProposalError, match="timed out"):
        model._request_json("http://localhost:11434", "ollama", "/tags", None, 0.01)
    assert released.is_set() and time.monotonic() - started < 0.5


def test_network_failure_after_validation_feedback_retains_attempt_history(monkeypatch):
    _, context = context_fixture()
    calls = []

    def request(base_url, kind, route, payload, timeout, cancelled):
        calls.append(route)
        if route == "/show":
            return {
                "details": {"format": "gguf"},
                "model_info": {"general.architecture": "qwen"},
            }
        if len(calls) == 2:
            return ollama_reply("{}")
        raise model.ModelProposalError("Local model server returned HTTP 503.")

    monkeypatch.setattr(model, "_request_json", request)
    with pytest.raises(model.ModelProposalError) as caught:
        model.generate_proposal(
            context,
            "Find bright objects.",
            model.ModelSettings(model="qwen-local", max_attempts=2),
        )
    assert len(caught.value.attempts) == 2
    assert caught.value.attempts[0].response_text == "{}"
    assert caught.value.attempts[0].phase == "proposal"
    assert caught.value.attempts[1].phase == "request"
    assert "503" in caught.value.attempts[1].validation_error
    assert calls == ["/show", "/chat", "/chat"]


def test_attempt_default_phase_preserves_existing_positional_constructor():
    attempt = model.ModelAttempt(1, 0.1, False, "Invalid JSON", "", 12, 3)
    assert attempt.phase == "proposal"
    assert (attempt.input_tokens, attempt.output_tokens) == (12, 3)
    assert (
        model.ModelAttempt(1, 0.1, False, "Unavailable", "", phase="request").phase
        == "request"
    )
    with pytest.raises(ValueError, match="attempt phase"):
        model.ModelAttempt(1, 0.1, False, "error", "", phase="path/private")


def test_safe_diagnostic_is_frozen_detached_and_has_only_allowlisted_fields():
    diagnostic = model.SafeModelDiagnostic(
        provider="local", stage="request", category="unauthorized", http_status=401
    )
    capsule = diagnostic.to_dict()
    assert capsule == {
        "provider": "local",
        "stage": "request",
        "category": "unauthorized",
        "code": None,
        "http_status": 401,
        "reason": "unknown",
    }
    capsule["reason"] = "private raw provider text"
    assert diagnostic.to_dict()["reason"] == "unknown"
    with pytest.raises(FrozenInstanceError):
        diagnostic.reason = "invalid_schema"
    error = model.ModelProposalError("A controlled failure.", diagnostic=diagnostic)
    assert error.diagnostic is diagnostic and error.attempts == ()
    with pytest.raises(TypeError, match="validated SafeModelDiagnostic"):
        model.ModelProposalError("Failure", diagnostic={"reason": "private"})


@pytest.mark.parametrize(
    "changes",
    [
        {"provider": "remote"},
        {"provider": "C:/private/token"},
        {"stage": "free provider message"},
        {"stage": []},
        {"category": "secret raw exception"},
        {"category": False},
        {"reason": "token=private"},
        {"reason": None},
        {"code": True},
        {"code": 1.2},
        {"code": 1_000_001},
        {"code": -1_000_001},
        {"http_status": True},
        {"http_status": 99},
        {"http_status": 600},
        {"http_status": "401"},
    ],
)
def test_safe_diagnostic_rejects_free_text_and_unbounded_values(changes):
    kwargs = {"provider": "codex", "stage": "turn", "category": "provider", **changes}
    with pytest.raises(ValueError) as caught:
        model.SafeModelDiagnostic(**kwargs)
    # Rejection itself never reflects the supplied arbitrary value.
    assert "private" not in str(caught.value)
    assert "secret" not in str(caught.value)


@pytest.mark.parametrize("blank", ["", " \n\t"])
def test_completed_blank_model_text_is_a_proposal_and_gets_one_repair(
    monkeypatch, blank
):
    _, context = context_fixture()
    expected = reference_proposal(context)
    calls = fake_requests(
        monkeypatch, [ollama_reply(blank), ollama_reply(json.dumps(expected))]
    )
    result = model.generate_proposal(
        context,
        "Find bright objects.",
        model.ModelSettings(model="qwen-local", max_attempts=2),
    )
    assert [attempt.phase for attempt in result.attempts] == ["proposal", "proposal"]
    assert [attempt.valid for attempt in result.attempts] == [False, True]
    assert result.attempts[0].response_text == blank
    assert [call[0] for call in calls] == ["/show", "/chat", "/chat"]


def test_transport_failure_is_one_request_attempt_and_preserves_safe_diagnostic(
    monkeypatch,
):
    _, context = context_fixture()
    diagnostic = model.SafeModelDiagnostic("local", "request", "network")
    calls = []

    def request(_url, _kind, route, *_args):
        calls.append(route)
        if route == "/show":
            return {"details": {"format": "gguf"}, "model_info": {"local": True}}
        raise model.ModelProposalError(
            "Cannot reach the local model server.", diagnostic=diagnostic
        )

    monkeypatch.setattr(model, "_request_json", request)
    with pytest.raises(model.ModelProposalError) as caught:
        model.generate_proposal(
            context,
            "Find objects.",
            model.ModelSettings(model="qwen-local", max_attempts=2),
        )
    assert calls == ["/show", "/chat"]
    assert caught.value.diagnostic is diagnostic
    assert len(caught.value.attempts) == 1
    assert caught.value.attempts[0].phase == "request"


def test_single_attempt_invalid_proposal_does_not_claim_repair_happened(monkeypatch):
    _, context = context_fixture()
    calls = fake_requests(monkeypatch, [ollama_reply("{}")])
    with pytest.raises(model.ModelProposalError, match="repair was disabled") as caught:
        model.generate_proposal(
            context,
            "Find objects.",
            model.ModelSettings(model="qwen-local", max_attempts=1),
        )
    assert [call[0] for call in calls] == ["/show", "/chat"]
    assert caught.value.attempts[0].phase == "proposal"


@pytest.mark.parametrize(
    "status, category",
    [
        (400, "bad_request"),
        (401, "unauthorized"),
        (403, "unauthorized"),
        (429, "quota"),
        (503, "provider"),
        (504, "timeout"),
    ],
)
def test_http_error_capsule_contains_status_not_raw_response(
    status, category, monkeypatch
):
    connection = DummyConnection(
        DummyResponse(status, b"secret provider body token=abc")
    )
    monkeypatch.setattr(
        model.http.client, "HTTPConnection", lambda *_a, **_k: connection
    )
    with pytest.raises(model.ModelProposalError) as caught:
        model._request_json("http://localhost:11434", "ollama", "/chat", {}, 1)
    assert caught.value.diagnostic.category == category
    assert caught.value.diagnostic.http_status == status
    assert caught.value.diagnostic.stage == "request"
    assert "secret" not in json.dumps(caught.value.diagnostic.to_dict())
    assert "secret" not in str(caught.value)


def test_reference_comparison_records_three_tasks_without_model_calls(monkeypatch):
    path = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "evaluate_workflow_proposals.py"
    )
    spec = importlib.util.spec_from_file_location("proposal_evaluation_test", path)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    monkeypatch.setattr(
        evaluator,
        "generate_proposal",
        lambda *a, **k: pytest.fail("No opt-in inference"),
    )
    report = evaluator.evaluate()
    assert len(report["tasks"]) == 3
    assert report["settings"] is None
    for row in report["tasks"]:
        assert row["reference"]["contract_valid"]
        assert all(row["reference"]["intent_checks"].values())
        assert "not run" in row["local"]["status"]
    assert report["tasks"][-1]["reference"]["can_apply"] is False
    assert "not cloud/API performance" in report["interpretation"]
