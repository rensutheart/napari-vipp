"""Provider-free action/vision/accounting tests; no real model or paid calls."""

from __future__ import annotations

import base64
import json
import threading
import time
from copy import deepcopy
from dataclasses import replace

import pytest

from napari_vipp._tests.test_workflow_investigation import session
from napari_vipp._tests.test_workflow_investigation_model import (
    action_text,
    public_request,
)
from napari_vipp.core import workflow_agent_provider as provider
from napari_vipp.core import workflow_proposal_codex as codex
from napari_vipp.core import workflow_proposal_model as local
from napari_vipp.core.workflow_agent_loop import WorkflowAgentLoop
from napari_vipp.core.workflow_investigation import InvestigationError
from napari_vipp.core.workflow_proposal_images import ImagePreview


def settings(kind="openai-compatible", **kwargs):
    if kind == "openai":
        return provider.OpenAISettings(model="test-model", **kwargs)
    return local.ModelSettings(kind=kind, model="test-model", **kwargs)


def response(settings, text=None):
    text = json.dumps({"action_json": action_text()}) if text is None else text
    if isinstance(settings, provider.OpenAISettings):
        return {
            "model": settings.model,
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                }
            ],
            "usage": {"input_tokens": 23, "output_tokens": 11},
        }
    if settings.kind == "ollama":
        return {
            "model": settings.model,
            "message": {"role": "assistant", "content": text},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 23,
            "eval_count": 11,
        }
    return {
        "model": settings.model,
        "choices": [
            {"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 23, "completion_tokens": 11},
    }


def fake_transport(monkeypatch, chosen, replies):
    sent = []
    answers = iter(replies)

    def send(config, payload, **kwargs):
        assert config == chosen
        kwargs["before_send"]()
        sent.append((deepcopy(payload), kwargs["deadline"]))
        result = next(answers)
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(provider, "_request_json", send)
    monkeypatch.setattr(provider, "_confirm_ollama", lambda *_: None)
    return sent


def images_request():
    image = ImagePreview("Source XY", b"\x89PNG\r\n\x1a\npixels", "Native plane YX")
    request = public_request()
    request["image_permissions"]["input_images"] = True
    request["image_previews"] = [image.manifest()]
    return request, (image,)


@pytest.mark.parametrize("kind", ["ollama", "openai-compatible", "openai"])
def test_same_typed_action_contract_with_actual_approved_pixels(monkeypatch, kind):
    chosen = settings(kind)
    sent = fake_transport(monkeypatch, chosen, [response(chosen)])
    request, previews = images_request()
    original = deepcopy(request)
    calls = []
    result = provider.generate_investigation_action(
        request, chosen, image_previews=previews, before_call=lambda: calls.append(True)
    )
    assert result.action.action == "question"
    assert result.provider == ("openai" if kind == "openai" else "local")
    assert result.model == "test-model" and len(calls) == 1
    assert result.attempts[0].input_tokens == 23
    assert result.attempts[0].output_tokens == 11
    payload = sent[0][0]
    encoded = base64.b64encode(previews[0].png_bytes).decode("ascii")
    if kind == "ollama":
        assert payload["messages"][1]["images"] == [encoded]
        assert payload["format"]["required"] == ["action_json"]
        assert "think" not in payload  # Respect the model's default reasoning.
        assert payload["options"]["num_ctx"] == 16384
        assert payload["truncate"] is payload["shift"] is False
    elif kind == "openai-compatible":
        assert (
            payload["messages"][1]["content"][1]["image_url"]["url"]
            == "data:image/png;base64," + encoded
        )
        assert payload["response_format"]["type"] == "json_schema"
    else:
        assert (
            payload["input"][1]["content"][1]["image_url"]
            == "data:image/png;base64," + encoded
        )
        assert payload["store"] is False
        assert payload["text"]["format"]["strict"] is True
        assert "tools" not in payload
        assert "api_key" not in json.dumps(payload)
    assert request == original


@pytest.mark.parametrize("kind", ["ollama", "openai-compatible", "openai"])
def test_format_repair_uses_same_deadline_and_counts_every_actual_send(
    monkeypatch, kind
):
    chosen = settings(kind, max_attempts=2)
    malformed = json.dumps({"action_json": "{}"})
    sent = fake_transport(
        monkeypatch, chosen, [response(chosen, malformed), response(chosen)]
    )
    reservations = []
    deadline = time.monotonic() + 15
    result = provider.generate_investigation_action(
        public_request(),
        chosen,
        deadline=deadline,
        before_call=lambda: reservations.append(True),
    )
    assert len(result.attempts) == len(reservations) == len(sent) == 2
    assert [row.valid for row in result.attempts] == [False, True]
    assert [row[1] for row in sent] == [deadline, deadline]
    messages = sent[1][0]["input" if kind == "openai" else "messages"]
    assert messages[2]["role"] == "assistant" and messages[2]["content"] == "{}"


def test_invalid_wrapper_receives_only_one_format_repair(monkeypatch):
    chosen = settings(max_attempts=2)
    sent = fake_transport(
        monkeypatch, chosen, [response(chosen, "{}"), response(chosen, "{}")]
    )
    with pytest.raises(
        local.ModelProposalError, match="did not pass validation"
    ) as caught:
        provider.generate_investigation_action(public_request(), chosen)
    assert len(sent) == len(caught.value.attempts) == 2


@pytest.mark.parametrize("kind", ["openai-compatible", "openai"])
def test_provider_failure_not_format_repair_or_error_text_echo(monkeypatch, kind):
    chosen = settings(kind, max_attempts=2)
    failure = provider._error(
        "openai" if kind == "openai" else "local", "quota", status=429
    )
    sent = fake_transport(monkeypatch, chosen, [failure])
    with pytest.raises(local.ModelProposalError) as caught:
        provider.generate_investigation_action(public_request(), chosen)
    assert len(sent) == len(caught.value.attempts) == 1
    assert caught.value.diagnostic.category == "quota"
    assert caught.value.attempts[0].phase == "request"
    assert caught.value.attempts[0].response_text == ""


def test_authority_value_error_before_send_not_repaired(monkeypatch):
    chosen = settings(max_attempts=2)
    sent = fake_transport(monkeypatch, chosen, [response(chosen)])

    def reject():
        raise InvestigationError("The source authority changed.")

    with pytest.raises(InvestigationError, match="authority"):
        provider.generate_investigation_action(
            public_request(), chosen, before_call=reject
        )
    assert sent == []


def test_source_request_mutation_before_send_is_terminal(monkeypatch):
    chosen = settings(max_attempts=2)
    sent = fake_transport(monkeypatch, chosen, [response(chosen)])
    request = public_request()

    def change(_message):
        request["objective"] = "Another goal"

    with pytest.raises(local.ModelProposalError, match="changed"):
        provider.generate_investigation_action(request, chosen, progress=change)
    assert sent == []


@pytest.mark.parametrize("kind", ["openai-compatible", "openai"])
def test_expired_deadline_or_cancellation_never_sends(monkeypatch, kind):
    chosen = settings(kind)
    sent = fake_transport(monkeypatch, chosen, [])
    with pytest.raises(local.ModelProposalError) as caught:
        provider.generate_investigation_action(
            public_request(), chosen, deadline=time.monotonic() - 1
        )
    assert caught.value.diagnostic.category == "timeout"
    with pytest.raises(local.ModelCancelled):
        provider.generate_investigation_action(
            public_request(), chosen, cancelled=lambda: True
        )
    assert not sent


def test_paths_in_action_omitted_from_repair(monkeypatch):
    chosen = settings(max_attempts=2)
    path_action = action_text(arguments={"question": "Read C:/private/image.tif"})
    sent = fake_transport(
        monkeypatch,
        chosen,
        [response(chosen, json.dumps({"action_json": path_action})), response(chosen)],
    )
    provider.generate_investigation_action(public_request(), chosen)
    assert "C:/private" not in json.dumps(sent[1][0]["messages"][2:])


@pytest.mark.parametrize("kind", ["openai-compatible", "openai"])
def test_wrong_session_and_general_tools_never_accepted(monkeypatch, kind):
    chosen = settings(kind)
    bad = action_text(session_id="b" * 32)
    fake_transport(
        monkeypatch, chosen, [response(chosen, json.dumps({"action_json": bad}))]
    )
    with pytest.raises(local.ModelProposalError, match="did not pass validation"):
        provider.generate_investigation_action(public_request(), chosen)


@pytest.mark.parametrize(
    "info",
    [
        {
            "details": {"format": "gguf"},
            "model_info": {"architecture": "test"},
            "capabilities": ["completion"],
        },
        {
            "details": {"format": "gguf"},
            "model_info": {"architecture": "test"},
            "capabilities": ["vision"],
            "remote_host": "remote",
        },
        {
            "details": {"format": "other"},
            "model_info": {"architecture": "test"},
            "capabilities": ["vision"],
        },
    ],
)
def test_ollama_requires_local_weights_and_confirmed_vision(monkeypatch, info):
    metadata = []

    def show(*args):
        metadata.append(args)
        return info

    monkeypatch.setattr(local, "_request_json", show)
    chosen = settings("ollama")
    request, previews = images_request()
    with pytest.raises(local.ModelProposalError):
        provider.generate_investigation_action(request, chosen, image_previews=previews)
    assert len(metadata) == 1 and metadata[0][2] == "/show"
    assert metadata[0][3] == {"model": "test-model"}
    assert "objective" not in json.dumps(metadata)


def test_ollama_one_metadata_check_no_hidden_inference(monkeypatch):
    metadata = []

    def show(*args):
        metadata.append(args)
        return {
            "details": {"format": "gguf"},
            "model_info": {"architecture": "test"},
            "capabilities": ["vision", "completion"],
        }

    chosen = settings("ollama")
    sent = fake_transport(monkeypatch, chosen, [response(chosen)])
    # Keep the actual safety/capability check instead of fake_transport's stub.
    monkeypatch.undo()
    monkeypatch.setattr(local, "_request_json", show)
    reservations = []

    def send(_settings, _payload, **kwargs):
        kwargs["before_send"]()
        sent.append(True)
        return response(chosen)

    monkeypatch.setattr(provider, "_request_json", send)
    request, previews = images_request()
    provider.generate_investigation_action(
        request,
        chosen,
        image_previews=previews,
        before_call=lambda: reservations.append(True),
    )
    assert len(metadata) == len(sent) == len(reservations) == 1


class WireResponse:
    def __init__(self, raw, status=200):
        self.raw, self.status = raw, status

    def read(self, count):
        return self.raw[:count]


def wire_fixture(monkeypatch, result, *, status=200, on_connect=None):
    calls = []

    class Connection:
        def __init__(self, host, port, **kwargs):
            calls.append(("connection", host, port, kwargs))
            self.sock = None

        def connect(self):
            calls.append(("connect",))
            if on_connect:
                on_connect()

        def request(self, method, route, **kwargs):
            calls.append(("request", method, route, kwargs))

        def getresponse(self):
            raw = json.dumps(result).encode() if isinstance(result, dict) else result
            return WireResponse(raw, status)

        def close(self):
            pass

    monkeypatch.setattr(provider.http.client, "HTTPConnection", Connection)
    monkeypatch.setattr(provider.http.client, "HTTPSConnection", Connection)
    return calls


def test_official_api_credential_only_to_fixed_tls_host_after_accounted_gate(
    monkeypatch,
):
    chosen = settings("openai", api_key_env="VIPP_TEST_API_KEY")
    secret = "test-only-not-a-real-key"
    monkeypatch.setenv("VIPP_TEST_API_KEY", secret)
    calls = wire_fixture(monkeypatch, response(chosen))
    order = []
    provider.generate_investigation_action(
        public_request(), chosen, before_call=lambda: order.append("gate")
    )
    assert calls[0][1:3] == ("api.openai.com", 443)
    request = next(row for row in calls if row[0] == "request")
    assert request[1:3] == ("POST", "/v1/responses")
    assert request[3]["headers"]["Authorization"] == "Bearer " + secret
    assert secret not in repr(chosen) and secret not in request[3]["body"].decode()
    assert order == ["gate"]


def test_missing_api_secret_never_connects_or_consumes_inference(monkeypatch):
    chosen = settings("openai", api_key_env="VIPP_MISSING_TEST_KEY")
    monkeypatch.delenv(chosen.api_key_env, raising=False)
    calls = wire_fixture(monkeypatch, response(chosen))
    reservations = []
    with pytest.raises(
        local.ModelProposalError, match="environment variable"
    ) as caught:
        provider.generate_investigation_action(
            public_request(), chosen, before_call=lambda: reservations.append(True)
        )
    assert not calls and not reservations
    assert caught.value.diagnostic.provider == "openai"


@pytest.mark.parametrize("status", [302, 400, 401, 429, 500])
def test_http_error_safe_and_never_retried(monkeypatch, status):
    chosen = settings(max_attempts=2)
    private = b'{"error":"password and C:/private/image.tif"}'
    calls = wire_fixture(monkeypatch, private, status=status)
    with pytest.raises(local.ModelProposalError) as caught:
        provider.generate_investigation_action(public_request(), chosen)
    assert len([row for row in calls if row[0] == "request"]) == 1
    assert caught.value.diagnostic.http_status == status
    assert "password" not in str(caught.value)
    assert "private" not in str(caught.value)


def test_unsupported_image_http_error_is_clear_and_terminal(monkeypatch):
    chosen = settings(max_attempts=2)
    calls = wire_fixture(
        monkeypatch, b'{"error":"model does not support images"}', status=400
    )
    request, previews = images_request()
    with pytest.raises(local.ModelProposalError, match="image-capable") as caught:
        provider.generate_investigation_action(request, chosen, image_previews=previews)
    assert len([row for row in calls if row[0] == "request"]) == 1
    assert caught.value.diagnostic.reason == "unsupported_image"


@pytest.mark.parametrize(
    "raw",
    [b"{}", b'{"a":1,"a":2}', b'{"a":NaN}', b"[]", b"x" * (local.MAX_HTTP_BYTES + 1)],
    ids=["empty", "duplicate", "nonfinite", "list", "oversized"],
)
def test_invalid_response_json_never_repaired(monkeypatch, raw):
    chosen = settings(max_attempts=2)
    calls = wire_fixture(monkeypatch, raw)
    with pytest.raises(local.ModelProposalError):
        provider.generate_investigation_action(public_request(), chosen)
    assert len([row for row in calls if row[0] == "request"]) == 1


def test_cancellation_after_connect_discards_before_request(monkeypatch):
    stopped = threading.Event()
    chosen = settings()
    calls = wire_fixture(monkeypatch, response(chosen), on_connect=stopped.set)
    with pytest.raises(local.ModelCancelled):
        provider.generate_investigation_action(
            public_request(), chosen, cancelled=stopped.is_set
        )
    assert not [row for row in calls if row[0] == "request"]


def test_native_loop_local_provider_really_dispatches_same_tools(monkeypatch):
    engine = session()
    agent = WorkflowAgentLoop(engine)
    chosen = settings()
    action = {
        "kind": "napari-vipp-investigation-action",
        "version": 1,
        "session_id": engine.session_id,
        "action": "find_nodes",
        "arguments": {"query": "threshold"},
    }
    fake_transport(
        monkeypatch,
        chosen,
        [response(chosen, json.dumps({"action_json": json.dumps(action)}))],
    )
    step = agent.step(chosen)
    assert step.model_result.provider == "local"
    assert step.model_result.action.action == "find_nodes"
    assert engine.model_calls_used == engine.actions_used == 1
    assert step.tool_result.public_result()["action"] == "find_nodes"


def test_discard_prepared_or_decided_turn_keeps_usage(monkeypatch):
    engine = session()
    agent = WorkflowAgentLoop(engine)
    first = agent.prepare_turn()
    agent.discard_pending_turn()
    assert engine.model_calls_used == engine.actions_used == 0
    with pytest.raises(InvestigationError):
        agent.choose_action(first, settings())
    chosen = settings()
    action = {
        "kind": "napari-vipp-investigation-action",
        "version": 1,
        "session_id": engine.session_id,
        "action": "find_nodes",
        "arguments": {"query": "threshold"},
    }
    fake_transport(
        monkeypatch,
        chosen,
        [response(chosen, json.dumps({"action_json": json.dumps(action)}))],
    )
    turn = agent.prepare_turn()
    result = agent.choose_action(turn, chosen)
    assert engine.model_calls_used == 1
    agent.discard_pending_turn()
    assert engine.model_calls_used == 1 and engine.actions_used == 0
    with pytest.raises(InvestigationError):
        agent.execute_action(result.action)
    assert agent.prepare_turn() is not turn


def test_discard_refuses_active_request():
    engine = session()
    agent = WorkflowAgentLoop(engine)
    agent._phase = "requesting"
    with pytest.raises(InvestigationError, match="active agent request"):
        agent.discard_pending_turn()
    assert agent._phase == "requesting"


def test_codex_adapter_is_delegated_unchanged(monkeypatch):
    calls = []
    expected = object()

    def dispatch(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(provider.model, "generate_codex_investigation_action", dispatch)
    chosen = codex.CodexSettings()
    assert provider.generate_investigation_action(public_request(), chosen) is expected
    assert calls[0][0][1] is chosen


@pytest.mark.parametrize(
    "change",
    [
        {"model": "https://bad"},
        {"api_key_env": "sk-test-secret"},
        {"max_attempts": 3},
        {"max_attempts": True},
        {"timeout_seconds": float("nan")},
        {"max_output_tokens": 0},
        {"reasoning_effort": "whatever"},
    ],
)
def test_api_settings_reject_unsafe_or_unbounded_choices(change):
    with pytest.raises(local.ModelProposalError):
        replace(settings("openai"), **change).validate()


@pytest.mark.parametrize("count", [True, 0, 4095, 65537])
def test_local_context_window_is_explicit_bounded_and_non_boolean(count):
    with pytest.raises(local.ModelProposalError, match="context size"):
        replace(settings("ollama"), context_tokens=count).validate()


def test_user_chosen_local_context_is_not_silently_increased(monkeypatch):
    chosen = settings("ollama", context_tokens=32768)
    sent = fake_transport(monkeypatch, chosen, [response(chosen)])
    provider.generate_investigation_action(public_request(), chosen)
    assert sent[0][0]["options"]["num_ctx"] == 32768


def test_shared_request_accepts_exact_continuation_but_rejects_fake_grants(monkeypatch):
    from napari_vipp.core.workflow_agent_budget import HARD_ALLOWANCE, TIME_POLICY

    engine = session()
    request = engine.public_request()
    request["continuation"] = {
        "version": 1,
        "allowance": dict(HARD_ALLOWANCE),
        "hard_limits": dict(HARD_ALLOWANCE),
        "grants_used": 0,
        "time_policy": TIME_POLICY,
    }
    chosen = settings()
    action = {
        "kind": "napari-vipp-investigation-action",
        "version": 1,
        "session_id": engine.session_id,
        "action": "find_nodes",
        "arguments": {"query": "threshold"},
    }
    sent = fake_transport(
        monkeypatch,
        chosen,
        [response(chosen, json.dumps({"action_json": json.dumps(action)}))],
    )
    provider.generate_investigation_action(request, chosen)
    assert len(sent) == 1
    request["continuation"]["hard_limits"]["max_actions"] += 1
    with pytest.raises(local.ModelProposalError, match="continuation"):
        provider.generate_investigation_action(request, chosen)
    assert len(sent) == 1


def test_actual_reported_model_is_not_inferred_from_requested_alias(monkeypatch):
    chosen = settings()
    wire = response(chosen)
    wire["model"] = "test-model-resolved"
    fake_transport(monkeypatch, chosen, [wire])
    result = provider.generate_investigation_action(public_request(), chosen)
    assert result.model == "test-model-resolved"
    assert result.requested_model == "test-model"


def test_missing_reported_model_stays_unknown_not_false_resolution(monkeypatch):
    chosen = settings()
    wire = response(chosen)
    del wire["model"]
    fake_transport(monkeypatch, chosen, [wire])
    result = provider.generate_investigation_action(public_request(), chosen)
    assert result.model == "" and result.requested_model == "test-model"


def test_local_context_above_declared_capacity_rejected_without_inference(monkeypatch):
    monkeypatch.setattr(
        local,
        "_request_json",
        lambda *_: {
            "details": {"format": "gguf"},
            "model_info": {"test.context_length": 8192},
            "capabilities": ["vision", "completion"],
        },
    )
    with pytest.raises(local.ModelProposalError, match="context size exceeds"):
        provider.generate_investigation_action(public_request(), settings("ollama"))


def test_continuation_contract_note_is_procedural_and_explicitly_scoped():
    ordinary = public_request()
    original_system, original_developer, original_schema = (
        provider.model.provider_contract(ordinary)
    )
    enabled = deepcopy(ordinary)
    enabled["continuation"] = {}
    system, developer, schema = provider.model.provider_contract(enabled)
    assert system == original_system and schema == original_schema
    assert "pauses this SAME investigation" in developer
    assert "cannot resume yourself" in developer
    assert "history is discarded" not in original_developer
    assert developer.startswith(original_developer)
