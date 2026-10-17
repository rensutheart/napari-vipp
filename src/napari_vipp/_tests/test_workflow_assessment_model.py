from __future__ import annotations

import json
import threading
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest

from napari_vipp.core import workflow_assessment_model as model
from napari_vipp.core import workflow_proposal_codex as codex
from napari_vipp.core import workflow_proposal_model as local
from napari_vipp.core.workflow_proposal_images import ImagePreview
from napari_vipp.core.workflow_proposals import ProposalDialogue


def context_fixture(*, images=False):
    context = SimpleNamespace(
        assessment_id="a" * 32,
        run_id="completed-run-7",
        source_id="private-source-id",
        candidate_node_ids=("private-node-id",),
        image_previews=(
            ImagePreview("Mask n1", b"\x89PNG\r\n\x1a\npreview", "XY slice"),
        )
        if images
        else (),
    )
    evidence = {
        "assessment_id": context.assessment_id,
        "run_id": context.run_id,
        "source_id": "input",
        "candidate_node_ids": ["n1"],
        "source": {"shape": [3, 8, 9], "axes": "ZYX"},
        "selected_outputs": [
            {"node_id": "n1", "components": 147, "foreground_voxels": 97}
        ],
        "candidate_graph": {
            "nodes": [{"id": "n1", "operation_id": "binary_threshold"}]
        },
        "image_previews": [preview.manifest() for preview in context.image_previews],
    }

    def request(objective):
        result = deepcopy(evidence)
        result["objective"] = (
            objective.objective
            if isinstance(objective, ProposalDialogue)
            else objective
        )
        if isinstance(objective, ProposalDialogue):
            result["clarifications"] = [
                {"question": question, "answer": answer}
                for question, answer in objective.clarifications
            ]
        return result

    context.public_request = request
    return context, evidence


def valid_report(context):
    return {
        "assessment_id": context.assessment_id,
        "run_id": context.run_id,
        "overall": "not_achieved",
        "summary": (
            "The mask is fragmented, so the two structures are not yet separated."
        ),
        "goal_checks": [
            {
                "criterion": "Two clean structures",
                "status": "not_met",
                "evidence": "Output n1 contains 147 connected components.",
            }
        ],
        "findings": ["Small disconnected fragments remain."],
        "suggested_changes": ["Review background correction before thresholding."],
        "limitations": ["No reference mask was supplied."],
    }


def ready(monkeypatch):
    runtime = codex.CodexRuntime("test-codex", "test", True, True, "Ready", "a" * 64)
    monkeypatch.setattr(codex, "discover_codex_runtime", lambda *_: runtime)
    return runtime


def fake_codex(monkeypatch, replies):
    ready(monkeypatch)
    calls = []
    replies = iter(replies)

    def call(*args, **kwargs):
        calls.append((deepcopy(args), deepcopy(kwargs)))
        result = next(replies)
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(codex, "_call_codex", call)
    return calls


def test_strict_report_preserves_run_and_adds_independent_accuracy_caveat():
    context, _ = context_fixture()
    report = model.validate_assessment(json.dumps(valid_report(context)), context)
    assert report.assessment_id == report.context_id == context.assessment_id
    assert report.run_id == context.run_id and report.overall == "not_achieved"
    assert (
        report.goal_checks[0].evidence == "Output n1 contains 147 connected components."
    )
    assert model.ASSESSMENT_CAVEAT in report.limitations
    with pytest.raises(FrozenInstanceError):
        report.summary = "Claim success"
    changed = report.to_dict()
    changed["goal_checks"][0]["criterion"] = "Changed"
    assert report.goal_checks[0].criterion == "Two clean structures"


@pytest.mark.parametrize(
    "changes",
    [
        {"assessment_id": "other"},
        {"run_id": "other-run"},
        {"run_id": 7},
        {"overall": "confident"},
        {"overall": "achieved"},
        {"confidence": 0.99},
        {"nodes": []},
        {"summary": ""},
        {"summary": "x" * 1201},
        {"summary": "Read C:/private/input.tif"},
        {"summary": "bad\x00text"},
        {"goal_checks": []},
        {"goal_checks": [{}]},
        {
            "goal_checks": [
                {"criterion": "Two objects", "status": "met", "evidence": ""}
            ]
        },
        {
            "goal_checks": [
                {"criterion": "Two objects", "status": "yes", "evidence": "No"}
            ]
        },
        {"findings": ["noise"] * 17},
        {"limitations": "none"},
    ],
)
def test_malformed_wrong_run_and_free_confidence_are_not_accepted(changes):
    context, _ = context_fixture()
    value = valid_report(context) | changes
    with pytest.raises(local.ModelProposalError):
        model.validate_assessment(json.dumps(value), context)


@pytest.mark.parametrize(
    "text",
    [
        "not JSON",
        "[]",
        "```json\n{}\n```",
        '{"summary":1,"summary":2}',
        '{"value":NaN}',
        " " * (model.MAX_ASSESSMENT_BYTES + 1),
    ],
    ids=["not-json", "array", "markdown", "duplicates", "non-finite", "oversize"],
)
def test_no_json_repair_coercion_or_unbounded_report(text):
    context, _ = context_fixture()
    with pytest.raises(local.ModelProposalError):
        model.validate_assessment(text, context)


def test_codex_receives_actual_run_measurements_graph_previews_and_original_goal(
    monkeypatch,
):
    context, evidence = context_fixture(images=True)
    calls = fake_codex(monkeypatch, [json.dumps(valid_report(context))])
    dialogue = ProposalDialogue(
        "Separate two structures.", (("Which?", "Use best judgment."),)
    )
    result = model.generate_codex_assessment(context, dialogue, codex.CodexSettings())
    args, kwargs = calls[0]
    request, previews = args[:2]
    assert request["selected_outputs"] == evidence["selected_outputs"]
    assert request["candidate_graph"] == evidence["candidate_graph"]
    assert request["objective"] == dialogue.objective
    assert request["clarifications"][0]["answer"] == "Use best judgment."
    assert previews == context.image_previews
    assert request["image_previews"] == [previews[0].manifest()]
    assert kwargs["assessment_contract"][2]["required"] == ["assessment_json"]
    assert "Two mesh groups" in kwargs["assessment_contract"][0]
    assert "private-source" not in json.dumps(request)
    assert result.context is context and result.provider == "codex"
    assert result.assessment.overall == "not_achieved" and result.attempts[0].valid


def test_goal_progress_guidance_keeps_tradeoffs_and_same_strict_report(monkeypatch):
    context, _ = context_fixture(images=True)
    calls = fake_codex(monkeypatch, [json.dumps(valid_report(context))])
    progress = []
    result = model.generate_codex_assessment(
        context,
        "Separate the intended structures without losing their thin parts.",
        codex.CodexSettings(),
        progress=progress.append,
    )
    request = calls[0][0][0]
    system = calls[0][1]["assessment_contract"][0]
    assert request["assessment_rules"] == system
    for required in (
        "coverage",
        "unwanted background",
        "false connecting bridges",
        "intended separation",
        "mesh quality/admissibility",
        "cleaner appearance alone is not progress",
        "Use mixed when gains have material losses",
        "mask-preserving mesh change",
        "coordinate smoothing is not a repair for invalid edge topology",
        "Without an independent target",
        "a failed calculation is not a new result",
    ):
        assert required in system
    assert set(request["response_schema"]["properties"]) == {
        "assessment_id",
        "run_id",
        "overall",
        "summary",
        "goal_checks",
        "findings",
        "suggested_changes",
        "limitations",
        "comparison",
    }
    assert request["response_schema"]["additionalProperties"] is False
    assert progress == ["The assistant is checking results against your goal…"]
    assert result.assessment.overall == "not_achieved"
    assert len(calls) == 1


def test_one_format_repair_never_relaxes_identity_or_adds_execution(monkeypatch):
    context, _ = context_fixture()
    wrong = valid_report(context) | {"run_id": "wrong-run"}
    calls = fake_codex(
        monkeypatch, [json.dumps(wrong), json.dumps(valid_report(context))]
    )
    result = model.generate_codex_assessment(
        context, "Separate structures.", codex.CodexSettings()
    )
    assert len(calls) == 2
    assert [attempt.valid for attempt in result.attempts] == [False, True]
    assert "wrong-run" in calls[1][0][4]
    assert "different captured run" in calls[1][0][5]
    assert calls[1][0][0]["run_id"] == context.run_id


def test_paths_in_rejected_report_are_omitted_from_repair(monkeypatch):
    context, _ = context_fixture()
    wrong = valid_report(context) | {"summary": "C:/private/secret.tif"}
    calls = fake_codex(
        monkeypatch, [json.dumps(wrong), json.dumps(valid_report(context))]
    )
    model.generate_codex_assessment(
        context, "Separate structures.", codex.CodexSettings()
    )
    assert "secret.tif" not in calls[1][0][4]
    assert "omitted" in calls[1][0][4]


def test_request_failure_not_retried_and_preserves_safe_diagnostic(monkeypatch):
    context, _ = context_fixture()
    diagnostic = local.SafeModelDiagnostic("codex", "turn", "timeout")
    calls = fake_codex(
        monkeypatch, [local.ModelProposalError("Timed out", diagnostic=diagnostic)]
    )
    with pytest.raises(local.ModelProposalError) as caught:
        model.generate_codex_assessment(
            context, "Separate structures.", codex.CodexSettings()
        )
    assert len(calls) == 1
    assert caught.value.diagnostic == diagnostic
    assert caught.value.attempts[0].phase == "request"


def test_cancelled_after_provider_reply_is_discarded(monkeypatch):
    context, _ = context_fixture()
    ready(monkeypatch)
    cancelled = threading.Event()

    def call(*args, **kwargs):
        cancelled.set()
        return json.dumps(valid_report(context))

    monkeypatch.setattr(codex, "_call_codex", call)
    with pytest.raises(local.ModelCancelled):
        model.generate_codex_assessment(
            context,
            "Separate structures.",
            codex.CodexSettings(),
            cancelled=cancelled.is_set,
        )


def test_changed_evidence_discards_reply_without_retry(monkeypatch):
    context, evidence = context_fixture()
    ready(monkeypatch)
    calls = []

    def call(*args, **kwargs):
        calls.append(args)
        evidence["selected_outputs"][0]["components"] = 1
        return json.dumps(valid_report(context))

    monkeypatch.setattr(codex, "_call_codex", call)
    with pytest.raises(local.ModelProposalError, match="evidence changed"):
        model.generate_codex_assessment(
            context, "Separate structures.", codex.CodexSettings()
        )
    assert len(calls) == 1


@pytest.mark.parametrize("field", ["assessment_id", "run_id", "objective"])
def test_invalid_identity_and_private_paths_precede_runtime_discovery(
    monkeypatch, field
):
    context, evidence = context_fixture()
    if field == "objective":
        objective = "Read C:/private/image.tif"
    else:
        objective = "Separate structures."
        evidence[field] = "other"
    monkeypatch.setattr(
        codex, "discover_codex_runtime", lambda *_: pytest.fail("No discovery")
    )
    with pytest.raises(local.ModelProposalError):
        model.generate_codex_assessment(context, objective, codex.CodexSettings())


def local_response(text, *, tool=False):
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": text,
                    **({"tool_calls": [{"name": "execute"}]} if tool else {}),
                },
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 32},
    }


def test_local_numeric_evidence_uses_same_loopback_transport_without_tools(monkeypatch):
    context, evidence = context_fixture()
    calls = []

    def request(*args):
        calls.append(deepcopy(args))
        return local_response(json.dumps(valid_report(context)))

    monkeypatch.setattr(local, "_request_json", request)
    result = model.generate_assessment(
        context,
        "Separate structures.",
        local.ModelSettings(
            kind="openai-compatible",
            model="qwen-test",
            base_url="http://127.0.0.1:1234",
        ),
    )
    assert len(calls) == 1 and calls[0][2] == "/chat/completions"
    payload = calls[0][3]
    assert "tools" not in payload
    sent = json.loads(payload["messages"][1]["content"])
    assert sent["selected_outputs"] == evidence["selected_outputs"]
    assert not sent["image_previews"]
    assert result.attempts[0].input_tokens == 12


def test_local_cannot_silently_drop_approved_images(monkeypatch):
    context, _ = context_fixture(images=True)
    monkeypatch.setattr(local, "_request_json", lambda *_: pytest.fail("No network"))
    with pytest.raises(local.ModelProposalError, match="measurements only"):
        model.generate_assessment(
            context, "Separate structures.", local.ModelSettings(model="qwen-test")
        )


def test_local_rejects_tool_calls_and_remote_endpoints_without_retry(monkeypatch):
    context, _ = context_fixture()
    calls = []

    def request(*args):
        calls.append(args)
        return local_response(json.dumps(valid_report(context)), tool=True)

    monkeypatch.setattr(local, "_request_json", request)
    settings = local.ModelSettings(
        kind="openai-compatible", model="qwen-test", max_attempts=2
    )
    with pytest.raises(local.ModelProposalError, match="Tool requests"):
        model.generate_assessment(context, "Separate structures.", settings)
    assert len(calls) == 1
    with pytest.raises(local.ModelProposalError, match="loopback"):
        model.generate_assessment(
            context,
            "Separate structures.",
            replace(settings, base_url="http://example.com"),
        )
    assert len(calls) == 1


def test_local_legacy_function_calls_remain_unavailable(monkeypatch):
    context, _ = context_fixture()
    response = local_response(json.dumps(valid_report(context)))
    response["choices"][0]["message"]["function_call"] = {
        "name": "execute",
        "arguments": "{}",
    }
    calls = []

    def request(*args):
        calls.append(args)
        return response

    monkeypatch.setattr(local, "_request_json", request)
    with pytest.raises(local.ModelProposalError, match="Function requests"):
        model.generate_assessment(
            context,
            "Separate structures.",
            local.ModelSettings(
                kind="openai-compatible", model="qwen-test", max_attempts=2
            ),
        )
    assert len(calls) == 1


class FakeRpc:
    events = []
    instances = []

    def __init__(self, *args):
        self.calls = []
        self.events_iter = iter(deepcopy(self.events))
        self.closed = False
        self.instances.append(self)

    def request(self, method, params):
        self.calls.append((method, deepcopy(params)))
        return {
            "initialize": {},
            "skills/list": {"data": []},
            "permissionProfile/list": {
                "data": [{"id": "inspect-plan", "allowed": True}]
            },
            "thread/start": {"thread": {"id": "thread-1"}},
            "turn/start": {"turn": {"id": "turn-1"}},
        }[method]

    def send(self, message):
        self.calls.append((message["method"], None))

    def receive(self):
        return next(self.events_iter)

    def close(self):
        self.closed = True


def transport(monkeypatch, events):
    ready(monkeypatch)
    FakeRpc.events = events
    FakeRpc.instances = []
    monkeypatch.setattr(codex, "_Rpc", FakeRpc)
    monkeypatch.setattr(codex, "_assert_profile", lambda *args, **kwargs: None)


def terminal(context, *, field="assessment_json", extra_items=()):
    return {
        "method": "turn/completed",
        "params": {
            "threadId": "thread-1",
            "turn": {
                "id": "turn-1",
                "status": "completed",
                "items": [
                    {
                        "type": "agentMessage",
                        "id": "answer",
                        "text": json.dumps({field: json.dumps(valid_report(context))}),
                    },
                    *extra_items,
                ],
            },
        },
    }


def test_actual_restricted_rpc_sends_evidence_images_and_assessment_schema(monkeypatch):
    context, evidence = context_fixture(images=True)
    transport(monkeypatch, [terminal(context)])
    model.generate_codex_assessment(
        context, "Separate structures.", codex.CodexSettings()
    )
    rpc = FakeRpc.instances[0]
    assert rpc.closed and len(FakeRpc.instances) == 1
    calls = dict(rpc.calls)
    thread = calls["thread/start"]
    assert thread["ephemeral"] and thread["dynamicTools"] == []
    assert thread["selectedCapabilityRoots"] == [] and thread["environments"] == []
    assert (
        thread["permissions"] == "inspect-plan" and thread["approvalPolicy"] == "never"
    )
    assert "completed VIPP workflow run" in thread["baseInstructions"]
    assert "assessment_json" in thread["developerInstructions"]
    turn = calls["turn/start"]
    assert turn["outputSchema"]["required"] == ["assessment_json"]
    assert turn["permissions"] == "inspect-plan"
    assert (
        json.loads(turn["input"][0]["text"])["selected_outputs"]
        == evidence["selected_outputs"]
    )
    assert turn["input"][1]["text"].startswith("Approved result evidence preview")
    assert turn["input"][2]["url"].startswith("data:image/png;base64,")


@pytest.mark.parametrize(
    "kind",
    ["commandExecution", "fileChange", "mcpToolCall", "webSearch", "dynamicToolCall"],
)
def test_assessment_transport_keeps_disabled_tool_fail_closed(monkeypatch, kind):
    context, _ = context_fixture()
    transport(monkeypatch, [terminal(context, extra_items=({"type": kind},))])
    with pytest.raises(local.ModelProposalError, match="disabled tool"):
        model.generate_codex_assessment(
            context, "Separate structures.", codex.CodexSettings()
        )
    assert len(FakeRpc.instances) == 1 and FakeRpc.instances[0].closed


def test_assessment_cannot_accept_a_proposal_wire_wrapper(monkeypatch):
    context, _ = context_fixture()
    transport(monkeypatch, [terminal(context, field="proposal_json")])
    with pytest.raises(local.ModelProposalError, match="reply envelope"):
        model.generate_codex_assessment(
            context, "Separate structures.", codex.CodexSettings()
        )
    assert len(FakeRpc.instances) == 1


def test_proposal_default_envelope_still_rejects_assessment_wrapper():
    with pytest.raises(local.ModelProposalError, match="reply envelope"):
        codex._decode_codex_envelope(json.dumps({"assessment_json": "{}"}))
    assert codex._decode_codex_envelope(json.dumps({"proposal_json": "{}"})) == "{}"
