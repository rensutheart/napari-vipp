from __future__ import annotations

import json
import time
from copy import deepcopy

import pytest

from napari_vipp.core import workflow_investigation_model as model
from napari_vipp.core import workflow_proposal_codex as codex
from napari_vipp.core import workflow_proposal_model as local
from napari_vipp.core.workflow_investigation import (
    action_schema,
    parse_investigation_action,
)
from napari_vipp.core.workflow_proposal_images import ImagePreview

SESSION = "a" * 32


def public_request():
    return {
        "kind": "napari-vipp-investigation-request",
        "version": 1,
        "session_id": SESSION,
        "proposal_context_id": "b" * 32,
        "objective": "Isolate the wanted structures while preserving their detail.",
        "clarifications": [],
        "deferred_questions": [],
        "source": {"shape": [8, 9], "axes": "YX"},
        "image_permissions": {"input_images": False, "output_images": False},
        "budgets": {},
        "retained_candidate_id": None,
        "response_schema": action_schema(),
        "history": [],
        "image_previews": [],
        "rules": [],
    }


def action_text(**changes):
    return json.dumps(
        {
            "kind": "napari-vipp-investigation-action",
            "version": 1,
            "session_id": SESSION,
            "action": "question",
            "arguments": {"question": "Which observed structures are wanted?"},
        }
        | changes
    )


def fake_provider(monkeypatch, replies):
    runtime = codex.CodexRuntime("fake-codex", "test", True, True, "Ready", "a" * 64)
    monkeypatch.setattr(codex, "discover_codex_runtime", lambda *_: runtime)
    answers = iter(replies)
    calls = []

    def call(*args, **kwargs):
        calls.append((deepcopy(args), deepcopy(kwargs)))
        reply = next(answers)
        if isinstance(reply, BaseException):
            raise reply
        if callable(reply):
            return reply(args, kwargs)
        kwargs["result_metadata"]["model"] = "resolved-model"
        return reply

    monkeypatch.setattr(codex, "_call_codex", call)
    return calls


def test_original_goal_typed_action_and_approved_images_no_execution(monkeypatch):
    calls = fake_provider(monkeypatch, [action_text()])
    image = ImagePreview("Source XY", b"\x89PNG\r\n\x1a\npixels", "Native plane YX")
    reservations = []
    request = public_request()
    request["image_permissions"]["input_images"] = True
    request["image_previews"] = [image.manifest()]
    original = deepcopy(request)
    result = model.generate_codex_investigation_action(
        request,
        codex.CodexSettings(),
        image_previews=(image,),
        before_call=lambda: reservations.append(True),
    )
    assert result.action.action == "question"
    assert result.action.session_id == SESSION
    assert result.provider == "codex" and result.model == "resolved-model"
    assert len(result.request_sha256) == 64
    assert len(result.attempts) == len(reservations) == 1
    sent, kwargs = calls[0]
    assert sent[0]["objective"] == request["objective"]
    assert sent[0]["image_previews"] == [image.manifest()]
    assert sent[1] == (image,)
    assert "reference" not in request and "previous_assessment" not in request
    assert kwargs["investigation_contract"][2]["required"] == ["action_json"]
    assert kwargs["absolute_deadline"] > time.monotonic()
    assert "shell" in kwargs["investigation_contract"][0]
    assert request == original


def test_one_format_repair_counts_each_turn_and_uses_same_deadline(monkeypatch):
    calls = fake_provider(monkeypatch, ["{}", action_text()])
    reservations = []
    limit = time.monotonic() + 20
    result = model.generate_codex_investigation_action(
        public_request(),
        codex.CodexSettings(max_attempts=2),
        deadline=limit,
        before_call=lambda: reservations.append(True),
    )
    assert len(result.attempts) == len(calls) == len(reservations) == 2
    assert [attempt.valid for attempt in result.attempts] == [False, True]
    assert calls[0][1]["absolute_deadline"] == calls[1][1]["absolute_deadline"] == limit
    assert calls[0][0][4:6] == (None, None)
    assert calls[1][0][4] == "{}"


def test_wrong_session_never_accepted_and_no_extra_attempt(monkeypatch):
    calls = fake_provider(monkeypatch, [action_text(session_id="b" * 32)])
    with pytest.raises(
        local.ModelProposalError, match="did not pass validation"
    ) as caught:
        model.generate_codex_investigation_action(
            public_request(), codex.CodexSettings(max_attempts=1)
        )
    assert len(calls) == len(caught.value.attempts) == 1
    assert caught.value.diagnostic is None


def test_invalid_path_reply_is_not_echoed_in_format_repair(monkeypatch):
    private = action_text(arguments={"question": "Read C:/private/image.tif"})
    calls = fake_provider(monkeypatch, [private, action_text()])
    model.generate_codex_investigation_action(
        public_request(), codex.CodexSettings(max_attempts=2)
    )
    assert "C:/private" not in json.dumps(calls[1][0][4:6])


def test_provider_failure_is_not_an_action_validation_failure_or_retried(monkeypatch):
    diagnostic = local.SafeModelDiagnostic("codex", "turn", "quota", http_status=429)
    calls = fake_provider(
        monkeypatch,
        [local.ModelProposalError("A safe provider failure.", diagnostic=diagnostic)],
    )
    with pytest.raises(
        local.ModelProposalError, match="safe provider failure"
    ) as caught:
        model.generate_codex_investigation_action(
            public_request(), codex.CodexSettings(max_attempts=2)
        )
    assert len(calls) == len(caught.value.attempts) == 1
    assert caught.value.attempts[0].phase == "request"
    assert caught.value.diagnostic == diagnostic


def test_cancelled_before_discovery_does_not_contact_provider(monkeypatch):
    monkeypatch.setattr(
        codex, "discover_codex_runtime", lambda *_: pytest.fail("discovery")
    )
    with pytest.raises(local.ModelCancelled):
        model.generate_codex_investigation_action(
            public_request(), codex.CodexSettings(), cancelled=lambda: True
        )


def test_cancelled_after_reply_discards_the_action(monkeypatch):
    cancelled = []

    def late_reply(_args, _kwargs):
        cancelled.append(True)
        return action_text()

    calls = fake_provider(monkeypatch, [late_reply])
    with pytest.raises(local.ModelCancelled):
        model.generate_codex_investigation_action(
            public_request(), codex.CodexSettings(), cancelled=lambda: bool(cancelled)
        )
    assert len(calls) == 1


def test_changed_request_during_reply_is_discarded_without_repair(monkeypatch):
    request = public_request()

    def mutate(_args, _kwargs):
        request["history"].append({"unexpected": "new evidence"})
        return action_text()

    calls = fake_provider(monkeypatch, [mutate])
    with pytest.raises(local.ModelProposalError, match="evidence changed") as caught:
        model.generate_codex_investigation_action(
            request, codex.CodexSettings(max_attempts=2)
        )
    assert len(calls) == 1 and caught.value.attempts[0].phase == "request"


def test_before_call_budget_gate_blocks_format_repair(monkeypatch):
    calls = fake_provider(monkeypatch, ["{}"])
    reservations = []

    def reserve():
        if reservations:
            raise ValueError("The model-call budget has been reached.")
        reservations.append(True)

    with pytest.raises(ValueError, match="model-call budget"):
        model.generate_codex_investigation_action(
            public_request(), codex.CodexSettings(max_attempts=2), before_call=reserve
        )
    assert len(calls) == 1


def test_elapsed_total_budget_blocks_format_repair(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(model.time, "monotonic", lambda: now[0])

    def consume(_args, _kwargs):
        now[0] += 6
        return "{}"

    calls = fake_provider(monkeypatch, [consume])
    with pytest.raises(local.ModelProposalError, match="deadline") as caught:
        model.generate_codex_investigation_action(
            public_request(), codex.CodexSettings(timeout_seconds=5, max_attempts=2)
        )
    assert len(calls) == 1 and caught.value.diagnostic.category == "timeout"


@pytest.mark.parametrize("deadline", [True, float("nan"), "later", -1.0])
def test_invalid_or_elapsed_deadline_prevents_discovery(monkeypatch, deadline):
    monkeypatch.setattr(
        codex, "discover_codex_runtime", lambda *_: pytest.fail("discovery")
    )
    with pytest.raises(local.ModelProposalError, match="deadline"):
        model.generate_codex_investigation_action(
            public_request(), codex.CodexSettings(), deadline=deadline
        )


@pytest.mark.parametrize(
    "input_request",
    [
        None,
        {},
        {"session_id": True},
        {"session_id": SESSION, "objective": "C:/private/image.tif"},
    ],
)
def test_private_or_malformed_request_prevents_runtime_discovery(
    monkeypatch, input_request
):
    monkeypatch.setattr(
        codex, "discover_codex_runtime", lambda *_: pytest.fail("discovery")
    )
    with pytest.raises(local.ModelProposalError):
        model.generate_codex_investigation_action(input_request, codex.CodexSettings())


def test_excessive_preview_is_rejected_before_discovery(monkeypatch):
    monkeypatch.setattr(
        codex, "discover_codex_runtime", lambda *_: pytest.fail("discovery")
    )
    image = ImagePreview("Source", b"\x89PNG\r\n\x1a\n" + b"x" * (1024 * 1024), "XY")
    with pytest.raises(local.ModelProposalError, match="preview exceeds"):
        model.generate_codex_investigation_action(
            public_request(), codex.CodexSettings(), image_previews=(image,)
        )


@pytest.mark.parametrize("change", ["path", "reference", "oversize", "version"])
def test_public_request_privacy_and_size_fail_before_discovery(monkeypatch, change):
    monkeypatch.setattr(
        codex, "discover_codex_runtime", lambda *_: pytest.fail("discovery")
    )
    value = public_request()
    if change == "path":
        value["objective"] = "Read C:/private/image.tif"
    elif change == "reference":
        value["reference_recipe"] = {"nodes": []}
    elif change == "oversize":
        value["history"] = ["x" * model.MAX_INVESTIGATION_REQUEST_BYTES]
    else:
        value["version"] = True
    with pytest.raises(local.ModelProposalError):
        model.generate_codex_investigation_action(value, codex.CodexSettings())


@pytest.mark.parametrize("change", ["disabled", "undeclared", "different", "nonbool"])
def test_images_require_matching_explicit_consent_disclosure(monkeypatch, change):
    monkeypatch.setattr(
        codex, "discover_codex_runtime", lambda *_: pytest.fail("discovery")
    )
    value = public_request()
    image = ImagePreview("Source XY", b"\x89PNG\r\n\x1a\npixels", "Native plane YX")
    value["image_permissions"]["input_images"] = True
    value["image_previews"] = [image.manifest()]
    if change == "disabled":
        value["image_permissions"]["input_images"] = False
    elif change == "undeclared":
        value["image_previews"] = []
    elif change == "different":
        value["image_previews"][0]["sha256"] = "b" * 64
    else:
        value["image_permissions"]["input_images"] = 1
    with pytest.raises(local.ModelProposalError, match="approved evidence disclosure"):
        model.generate_codex_investigation_action(
            value, codex.CodexSettings(), image_previews=(image,)
        )


def test_local_connector_explicitly_unsupported_without_network_or_fallback(
    monkeypatch,
):
    monkeypatch.setattr(local, "_request_json", lambda *_: pytest.fail("network"))
    monkeypatch.setattr(
        codex, "discover_codex_runtime", lambda *_: pytest.fail("fallback")
    )
    with pytest.raises(local.ModelProposalError, match="does not support") as caught:
        model.generate_investigation_action(public_request(), local.ModelSettings())
    assert caught.value.diagnostic.provider == "local"
    assert caught.value.diagnostic.category == "capability"


def test_unavailable_resolved_identity_is_not_inferred_from_requested_model(
    monkeypatch,
):
    calls = fake_provider(monkeypatch, [lambda _args, _kwargs: action_text()])
    result = model.generate_codex_investigation_action(
        public_request(), codex.CodexSettings(model="requested-model")
    )
    assert len(calls) == 1
    assert result.model == result.model_provider == ""
    assert result.requested_model == "requested-model"


def test_real_engine_request_and_actual_image_budget_match_transport(monkeypatch):
    from napari_vipp._tests.test_workflow_investigation import session

    engine = session(include_input_images=True)
    request = engine.public_request()
    images = engine.preview_images()
    calls = fake_provider(
        monkeypatch, ["{}", action_text(session_id=engine.session_id)]
    )
    used_before = request["budgets"]["image_bytes_remaining"]
    result = model.generate_codex_investigation_action(
        request,
        codex.CodexSettings(max_attempts=2),
        image_previews=images,
        before_call=lambda: engine.record_model_call(request=request),
        deadline=engine.deadline,
    )
    assert result.action.session_id == engine.session_id
    assert len(calls) == engine.model_calls_used == 2
    assert engine.public_request()["budgets"]["image_bytes_remaining"] == (
        used_before - 2 * sum(len(image.png_bytes) for image in images)
    )
    assert "history" not in calls[0][0][0]
    assert calls[0][0][0]["evidence_notebook"]["working_records"] == []
    assert calls[0][0][0]["image_previews"] == [image.manifest() for image in images]


def test_replay_first_live_sole_envelope_reply_without_a_repair_turn(monkeypatch):
    # Archived first live reply, not a solution/example sent to another model.
    # The ordinary action_json transport had already decoded this inner text.
    recorded = {
        "kind": "napari-vipp-investigation-action",
        "version": 1,
        "session_id": "8468110f57fa40cab16f3717516a36c9",
        "action": "record_observation",
        "arguments": {
            "evidence_refs": [
                {
                    "kind": "preview",
                    "sha256": (
                        "b18088b89e56ebd04d5e5e0e733d4ada"
                        "4307e8d2948de9e18ca6ef582c9347ad"
                    ),
                }
            ],
            "interpretation": (
                "The three sampled Z planes show a wavy bright band; the middle "
                "sampled plane also shows a brighter, approximately circular "
                "region overlapping the band's location. These previews do "
                "not establish complete 3D boundaries or separation at the overlap."
            ),
            "hypothesis": (
                "An intensity-based circular-core mask and a separately "
                "reconstructed wavy-band mask may distinguish the requested "
                "structures. This assumes brightness and spatial continuity "
                "are useful cues; any reconstruction beneath the circular "
                "region would be inferred, not directly observed, and could "
                "introduce false connections or omit signal."
            ),
            "next_check": (
                "Discover registered segmentation and morphology nodes, then "
                "inspect source intensity statistics and additional planes "
                "before choosing a finite baseline trial."
            ),
        },
    }
    raw = json.dumps({"envelope": recorded}, separators=(",", ":"))
    calls = fake_provider(monkeypatch, [raw])
    request = public_request()
    request["session_id"] = recorded["session_id"]
    result = model.generate_codex_investigation_action(
        request, codex.CodexSettings(max_attempts=1)
    )
    assert len(calls) == len(result.attempts) == 1
    assert result.action.to_dict() == recorded
    assert result.attempts[0].response_text == raw
    assert result.attempts[0].valid
    assert result.response_normalizations == ("sole_envelope_wrapper_unwrapped",)
    instructions = calls[0][1]["investigation_contract"][1]
    assert "DIRECTLY to the flat action object" in instructions
    assert "NOT another wrapper key" in instructions
    # Adaptation stays at the provider boundary; domain validation is unchanged.
    with pytest.raises(ValueError):
        parse_investigation_action(raw, expected_session_id=recorded["session_id"])


@pytest.mark.parametrize(
    "case",
    [
        "wrong-session",
        "extra-outer-key",
        "nested-envelope",
        "duplicate-outer-key",
        "duplicate-inner-key",
        "nonfinite",
        "not-object",
        "unknown-wrapper",
        "double-encoded-flat",
        "double-encoded-wrapped",
    ],
)
def test_known_wrapper_adapter_preserves_strict_rejections(monkeypatch, case):
    action = json.loads(action_text())
    value = {"envelope": action}
    if case == "wrong-session":
        action["session_id"] = "b" * 32
    elif case == "extra-outer-key":
        value["extra"] = True
    elif case == "nested-envelope":
        value = {"envelope": value}
    elif case == "nonfinite":
        action["arguments"]["question"] = float("nan")
    elif case == "not-object":
        value = {"envelope": action_text()}
    elif case == "unknown-wrapper":
        value = {"action_json": action}
    raw = json.dumps(value, separators=(",", ":"))
    if case == "duplicate-outer-key":
        raw = '{"envelope":{},"envelope":' + json.dumps(action) + "}"
    elif case == "duplicate-inner-key":
        raw = raw.replace('"version":1', '"version":1,"version":1')
    elif case == "double-encoded-flat":
        raw = json.dumps(json.dumps(action))
    elif case == "double-encoded-wrapped":
        raw = json.dumps(raw)
    calls = fake_provider(monkeypatch, [raw])
    with pytest.raises(local.ModelProposalError, match="did not pass validation"):
        model.generate_codex_investigation_action(
            public_request(), codex.CodexSettings(max_attempts=1)
        )
    assert len(calls) == 1


def test_wrapper_adapter_enforces_same_inner_reply_byte_limit():
    raw = json.dumps({"envelope": json.loads(action_text())})
    at_limit = raw + " " * (model.MAX_PROPOSAL_BYTES - len(raw.encode("utf-8")))
    action, normalizations = model._action_reply(at_limit, SESSION)
    assert action.action == "question"
    assert normalizations == ("sole_envelope_wrapper_unwrapped",)
    with pytest.raises(ValueError, match="bounded strict JSON"):
        model._action_reply(at_limit + " ", SESSION)


def test_flat_reply_remains_unmodified_and_has_no_normalization():
    raw = action_text()
    action, normalizations = model._action_reply(raw, SESSION)
    assert action.to_dict() == json.loads(raw)
    assert normalizations == ()
