from __future__ import annotations

import json
import time
from copy import deepcopy

import pytest

from napari_vipp.core import workflow_proposal_codex as codex


def contract():
    return (
        "Choose only one bounded typed investigation action.",
        "Return the action_json wrapper only.",
        {
            "type": "object",
            "properties": {"action_json": {"type": "string"}},
            "required": ["action_json"],
            "additionalProperties": False,
        },
    )


def completed(text, *, item_type="agentMessage", thread_id="owned-thread"):
    return {
        "method": "turn/completed",
        "params": {
            "threadId": thread_id,
            "turn": {
                "id": "owned-turn",
                "status": "completed",
                "items": [{"id": "answer", "type": item_type, "text": text}],
            },
        },
    }


class FakeRpc:
    instances = []
    events = []

    def __init__(self, executable, profile, cwd, deadline, cancelled, profile_digest):
        self.executable = executable
        self.profile = profile
        self.cwd = cwd
        self.deadline = deadline
        self.cancelled = cancelled
        self.profile_digest = profile_digest
        self.closed = False
        self.calls = []
        self.events = iter(deepcopy(type(self).events))
        type(self).instances.append(self)

    def request(self, method, params):
        self.calls.append((method, deepcopy(params)))
        return {
            "initialize": {},
            "permissionProfile/list": {
                "data": [{"id": "inspect-plan", "allowed": True}]
            },
            "skills/list": {"data": []},
            "thread/start": {
                "thread": {"id": "owned-thread"},
                "model": "resolved-model",
                "modelProvider": "openai",
            },
            "turn/start": {"turn": {"id": "owned-turn"}},
        }[method]

    def send(self, message):
        self.calls.append((message["method"], deepcopy(message.get("params"))))

    def receive(self):
        return next(self.events)

    def close(self):
        self.closed = True


def transport(monkeypatch, events):
    FakeRpc.instances = []
    FakeRpc.events = events
    monkeypatch.setattr(codex, "_Rpc", FakeRpc)
    checks = []
    monkeypatch.setattr(
        codex, "_assert_profile", lambda *_args, **_kwargs: checks.append(True)
    )
    return codex.CodexRuntime(
        "fake-codex", "test", True, True, "Ready", "b" * 64
    ), checks


def call(runtime, **kwargs):
    return codex._call_codex(
        {"session_id": "a" * 32, "objective": "Review observed structures."},
        (),
        codex.CodexSettings(),
        runtime,
        None,
        None,
        None,
        investigation_contract=contract(),
        **kwargs,
    )


def test_action_envelope_uses_fresh_isolated_thread_without_tools(monkeypatch):
    text = '{"action":"inspect_source"}'
    runtime, checks = transport(
        monkeypatch, [completed(json.dumps({"action_json": text}))]
    )
    metadata = {}
    deadline = time.monotonic() + 30
    assert call(runtime, absolute_deadline=deadline, result_metadata=metadata) == text
    assert metadata == {"model": "resolved-model", "modelProvider": "openai"}
    assert call(runtime) == text
    assert len(FakeRpc.instances) == 2
    assert len(checks) == 2
    for rpc in FakeRpc.instances:
        assert rpc.closed
        assert rpc.profile_digest == runtime.profile_config_sha256
        thread = dict(rpc.calls)["thread/start"]
        turn = dict(rpc.calls)["turn/start"]
        assert thread["ephemeral"] is True
        assert thread["environments"] == thread["dynamicTools"] == []
        assert thread["selectedCapabilityRoots"] == []
        assert thread["runtimeWorkspaceRoots"] == [rpc.cwd]
        assert thread["approvalPolicy"] == turn["approvalPolicy"] == "never"
        assert thread["permissions"] == turn["permissions"] == "inspect-plan"
        assert turn["environments"] == []
        assert turn["outputSchema"] == contract()[2]
        assert thread["baseInstructions"] == contract()[0]
        assert thread["allowProviderModelFallback"] is False
        assert "thread/resume" not in dict(rpc.calls)
    assert FakeRpc.instances[0].deadline == deadline
    assert FakeRpc.instances[0].cwd != FakeRpc.instances[1].cwd


@pytest.mark.parametrize(
    "wire",
    [
        "{}",
        '{"action_json":"{}","extra":"secret"}',
        '{"action_json":"{}","action_json":"{}"}',
        '{"proposal_json":"{}"}',
        '{"action_json":NaN}',
        '{"action_json":{}}',
    ],
)
def test_investigation_rejects_wrong_or_malformed_native_wrapper(monkeypatch, wire):
    runtime, _checks = transport(monkeypatch, [completed(wire)])
    with pytest.raises(
        codex.ModelProposalError, match="investigation reply envelope"
    ) as caught:
        call(runtime)
    assert caught.value.diagnostic.category == "protocol"
    assert FakeRpc.instances[0].closed


@pytest.mark.parametrize(
    "item_type", ["commandExecution", "dynamicToolCall", "fileChange", "webSearch"]
)
def test_investigation_contract_does_not_relax_tool_guards(monkeypatch, item_type):
    runtime, _checks = transport(
        monkeypatch, [completed("secret", item_type=item_type)]
    )
    with pytest.raises(codex.ModelProposalError, match="disabled tool") as caught:
        call(runtime)
    assert caught.value.diagnostic.category == "capability"
    assert FakeRpc.instances[0].closed


def test_investigation_retains_exact_thread_scope_guard(monkeypatch):
    runtime, _checks = transport(
        monkeypatch,
        [completed(json.dumps({"action_json": "{}"}), thread_id="other-thread")],
    )
    with pytest.raises(codex.ModelProposalError, match="unrelated planning session"):
        call(runtime)
    assert FakeRpc.instances[0].closed


def test_elapsed_shared_deadline_does_not_start_transport(monkeypatch):
    runtime, _checks = transport(monkeypatch, [])
    with pytest.raises(codex.ModelProposalError, match="deadline"):
        call(runtime, absolute_deadline=time.monotonic() - 1)
    assert FakeRpc.instances == []


def test_conflicting_contracts_never_start_transport(monkeypatch):
    runtime, _checks = transport(monkeypatch, [])
    with pytest.raises(codex.ModelProposalError, match="Only one"):
        call(runtime, assessment_contract=contract())
    assert FakeRpc.instances == []


def test_legacy_native_envelopes_are_unchanged():
    assert codex._decode_codex_envelope('{"proposal_json":"{}"}') == "{}"
    assert (
        codex._decode_codex_envelope('{"assessment_json":"{}"}', assessment=True)
        == "{}"
    )
    with pytest.raises(codex.ModelProposalError, match="planning reply envelope"):
        codex._decode_codex_envelope('{"action_json":"{}"}')
