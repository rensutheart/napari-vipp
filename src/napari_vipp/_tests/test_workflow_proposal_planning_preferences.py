"""Client-owned drafting preferences preserve scientific guards and repair bounds."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from napari_vipp.core import workflow_proposal_codex as codex
from napari_vipp.core import workflow_proposal_model as local
from napari_vipp.core import workflow_proposals as proposals
from napari_vipp.core.metadata import AxisMetadata, image_state_from_array
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.snapshots import GraphSnapshot
from napari_vipp.core.workflow_proposals import (
    PlanningPreferences,
    ProposalContext,
    ProposalDialogue,
    ProposalError,
    capture_proposal_context,
    reference_proposal,
    validate_proposal,
)

_MODES = (("ask_first", 3), ("balanced", 2), ("use_best_judgment", 1))


@pytest.fixture
def source_context():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    data = np.arange(12 * 16, dtype=np.float32).reshape(12, 16).copy()
    data.flags.writeable = False
    pipeline.outputs["input"] = data
    pipeline.output_states["input"] = image_state_from_array(
        data, axes=tuple(AxisMetadata(axis, "space", "um", 0.2) for axis in "yx")
    )
    return pipeline, capture_proposal_context(pipeline, "input")


def _context_mode(context, mode):
    return replace(context, planning_preferences=PlanningPreferences(mode))


@pytest.mark.parametrize(("mode", "maximum"), _MODES)
def test_preferences_are_frozen_closed_and_client_derived(mode, maximum):
    preferences = PlanningPreferences(mode)
    assert preferences.max_questions == maximum
    public = preferences.to_dict()
    assert public["mode"] == mode and public["max_questions"] == maximum
    assert public["rules"]
    public["max_questions"] = 99
    public["rules"].clear()
    assert preferences.to_dict()["max_questions"] == maximum
    assert preferences.to_dict()["rules"]
    with pytest.raises(FrozenInstanceError):
        preferences.mode = "ask_first"
    with pytest.raises(TypeError):
        PlanningPreferences(mode, max_questions=99)


@pytest.mark.parametrize("mode", ["", "ASK_FIRST", "best", None, True, 1, {}, []])
def test_unknown_or_untyped_modes_fail_explicitly(mode):
    with pytest.raises(ProposalError, match="Planning mode"):
        PlanningPreferences(mode)


def test_context_defaults_are_conservative_and_positional_capture_stays_compatible(
    source_context,
):
    _pipeline, context = source_context
    legacy = ProposalContext(
        context.context_id,
        context.source_id,
        context.baseline,
        context.baseline_fingerprint,
        context.source_fingerprint,
        context.source_summary,
        context.image_previews,
    )
    assert legacy.planning_preferences == context.planning_preferences
    assert context.planning_preferences == PlanningPreferences("ask_first")


@pytest.mark.parametrize("preferences", [None, {}, "use_best_judgment", 1])
def test_context_requires_typed_preferences_before_any_request(
    source_context, preferences
):
    _pipeline, context = source_context
    with pytest.raises(TypeError, match="PlanningPreferences"):
        replace(context, planning_preferences=preferences)


@pytest.mark.parametrize(("mode", "maximum"), _MODES)
def test_public_preferences_are_detached_without_changing_source_or_graph(
    source_context, mode, maximum
):
    pipeline, original = source_context
    before = GraphSnapshot.from_pipeline(pipeline)
    pixels = pipeline.outputs["input"].copy()
    context = _context_mode(original, mode)
    dialogue = ProposalDialogue(
        "Goal", (("Boundary?", "Not sure; use best judgment."),)
    )
    public = context.public_request(dialogue)
    assert public["planning_preferences"] == context.planning_preferences.to_dict()
    assert public["planning_preferences"]["max_questions"] == maximum
    assert public["clarifications"] == [
        {"question": "Boundary?", "answer": "Not sure; use best judgment."}
    ]
    public["planning_preferences"]["rules"].clear()
    public["planning_preferences"]["max_questions"] = 999
    assert context.public_request(dialogue)["planning_preferences"]["rules"]
    fresh_preferences = context.public_request(dialogue)["planning_preferences"]
    assert fresh_preferences["max_questions"] == maximum
    assert context.context_id == original.context_id
    assert context.baseline_fingerprint == original.baseline_fingerprint
    assert context.source_fingerprint == original.source_fingerprint
    assert context.source_summary == original.source_summary
    assert GraphSnapshot.from_pipeline(pipeline) == before
    assert np.array_equal(pipeline.outputs["input"], pixels)
    assert not pipeline.outputs["input"].flags.writeable


def test_forged_context_or_mode_fails_before_provider_discovery(
    source_context, monkeypatch
):
    _pipeline, original = source_context
    context = _context_mode(original, "use_best_judgment")
    object.__setattr__(context.planning_preferences, "mode", "unlimited")
    monkeypatch.setattr(
        codex, "discover_codex_runtime", lambda *_: pytest.fail("No discovery")
    )
    monkeypatch.setattr(local, "_request_json", lambda *_: pytest.fail("No transport"))
    with pytest.raises(ProposalError, match="Planning mode"):
        codex.generate_codex_proposal(context, "Goal", codex.CodexSettings())
    with pytest.raises(ProposalError, match="Planning mode"):
        local.generate_proposal(
            context, "Goal", local.ModelSettings(model="offline-test")
        )
    object.__setattr__(context, "planning_preferences", {"max_questions": 999})
    with pytest.raises(TypeError, match="PlanningPreferences"):
        validate_proposal(reference_proposal(original), context)


@pytest.mark.parametrize(("mode", "maximum"), _MODES)
@pytest.mark.parametrize("questions_present", [False, True])
def test_question_cap_keeps_each_question_and_apply_guard(
    source_context, mode, maximum, questions_present
):
    _pipeline, original = source_context
    context = _context_mode(original, mode)
    proposal = reference_proposal(context)
    proposal["questions"] = (
        [f"Essential decision {index}?" for index in range(maximum)]
        if questions_present
        else []
    )
    before = deepcopy(proposal)
    prepared = validate_proposal(proposal, context)
    assert prepared.questions == tuple(proposal["questions"])
    assert prepared.can_apply is not questions_present
    assert proposal == before


@pytest.mark.parametrize(("mode", "maximum"), _MODES)
def test_excess_questions_fail_without_slicing_or_changing_the_graph(
    source_context, mode, maximum
):
    pipeline, original = source_context
    context = _context_mode(original, mode)
    proposal = reference_proposal(context)
    proposal["questions"] = [f"Question {index}?" for index in range(maximum + 1)]
    before = deepcopy(proposal)
    graph = GraphSnapshot.from_pipeline(pipeline)
    with pytest.raises(ProposalError, match=f"at most {maximum} unanswered question"):
        validate_proposal(proposal, context)
    assert proposal == before
    assert GraphSnapshot.from_pipeline(pipeline) == graph


@pytest.mark.parametrize(("mode", "_maximum"), _MODES)
@pytest.mark.parametrize(
    "violation", ["operation", "parameter", "policy_override", "empty"]
)
def test_initiative_never_relaxes_scientific_or_proposal_guards(
    source_context, mode, _maximum, violation
):
    _pipeline, original = source_context
    context = _context_mode(original, mode)
    proposal = reference_proposal(context)
    if violation == "operation":
        proposal["nodes"][0]["operation_id"] = "arbitrary_execution"
    elif violation == "parameter":
        proposal["nodes"][0]["params"]["sigma"] = -100
    elif violation == "policy_override":
        proposal["planning_preferences"] = {"max_questions": 100}
    else:
        proposal["nodes"] = proposal["connections"] = []
    with pytest.raises(ProposalError):
        validate_proposal(proposal, context)


def test_answer_history_has_independent_limits_and_empty_dialogue_remains_valid(
    source_context,
):
    _pipeline, original = source_context
    context = _context_mode(original, "use_best_judgment")
    answers = tuple(
        (f"Answered question {index}?", "Use best judgment.") for index in range(5)
    )
    dialogue = ProposalDialogue("  Exact goal  ", answers)
    public = context.public_request(dialogue)
    assert public["objective"] == dialogue.objective
    assert public["clarifications"] == [
        {"question": question, "answer": answer} for question, answer in answers
    ]
    assert "clarifications" not in context.public_request(ProposalDialogue("Goal", ()))
    with pytest.raises(ProposalError):
        ProposalDialogue("Goal", (("Question?", ""),))


def test_deferred_choices_remain_separate_exact_and_immutable(source_context):
    _pipeline, original = source_context
    context = _context_mode(original, "use_best_judgment")
    answers = (("Object count?", "  yes two objects total. use your best judgment  "),)
    deferred = ("  Complete cavities or observed signal?  ", "Boundary convention?")
    dialogue = ProposalDialogue("  Exact original goal  ", answers, deferred)
    request = context.public_request(dialogue)
    assert request["objective"] == dialogue.objective
    assert request["clarifications"] == [
        {"question": question, "answer": answer} for question, answer in answers
    ]
    assert request["deferred_questions"] == list(deferred)
    assert dialogue.clarifications == answers
    assert dialogue.deferred_questions == deferred
    with pytest.raises(FrozenInstanceError):
        dialogue.deferred_questions = ()
    request["deferred_questions"].clear()
    assert context.public_request(dialogue)["deferred_questions"] == list(deferred)
    assert "deferred_questions" not in context.public_request(ProposalDialogue("Goal"))
    assert "deferred_questions" not in context.public_request("Goal")


@pytest.mark.parametrize(
    "deferred",
    [
        [],
        ("Question?",) * 4,
        ("",),
        ("  ",),
        ("q" * 2001,),
        ("Question?", " question? "),
        (123,),
        (b"Question?",),
        ("Question\x00?",),
        ("Question\ud800?",),
    ],
)
def test_deferred_questions_fail_invalid_types_limits_and_duplicate_choices(deferred):
    with pytest.raises(ProposalError):
        ProposalDialogue("Goal", (), deferred)


def test_three_pending_questions_at_the_text_limit_are_not_silently_trimmed(
    source_context,
):
    _pipeline, original = source_context
    deferred = tuple(f"{index}:" + "q" * 1998 for index in range(3))
    dialogue = ProposalDialogue("Goal", (), deferred)
    request = _context_mode(original, "use_best_judgment").public_request(dialogue)
    assert request["deferred_questions"] == list(deferred)
    assert "clarifications" not in request


def test_a_repeated_pending_question_does_not_replace_its_existing_answer(
    source_context,
):
    _pipeline, context = source_context
    dialogue = ProposalDialogue(
        "Goal", (("Boundary?", "Use observed signal."),), ("Boundary?",)
    )
    request = context.public_request(dialogue)
    assert request["clarifications"] == [
        {"question": "Boundary?", "answer": "Use observed signal."}
    ]
    assert request["deferred_questions"] == ["Boundary?"]
    assert "respect that answer" in " ".join(request["planning_preferences"]["rules"])


def test_deferred_text_is_revalidated_before_catalogue_or_provider_work(
    source_context, monkeypatch
):
    _pipeline, context = source_context
    dialogue = ProposalDialogue("Goal", (), ("Boundary?",))
    object.__setattr__(dialogue, "deferred_questions", ["Boundary?"])
    monkeypatch.setattr(
        proposals, "build_catalogue", lambda: pytest.fail("No catalogue construction")
    )
    with pytest.raises(ProposalError, match="immutable tuple"):
        context.public_request(dialogue)


def test_deferred_text_stays_within_aggregate_history_and_public_request_budgets(
    source_context, monkeypatch
):
    answers = tuple((f"{index:02d}" + "q" * 1998, "a" * 2000) for index in range(16))
    with pytest.raises(ProposalError, match="64 KiB"):
        ProposalDialogue("Goal", answers, ("d" * 2000,))
    _pipeline, context = source_context
    base_request = context.public_request(ProposalDialogue("Goal"))
    base_size = len(proposals._json_bytes(base_request))
    monkeypatch.setattr(proposals, "MAX_REQUEST_BYTES", base_size + 10)
    with pytest.raises(ProposalError, match="request exceeds"):
        context.public_request(ProposalDialogue("Goal", (), ("A pending convention?",)))


@pytest.mark.parametrize(("mode", "maximum"), _MODES)
def test_both_provider_prompts_respect_typed_policy_without_blanket_ambiguity_rules(
    source_context, mode, maximum
):
    _pipeline, original = source_context
    context = _context_mode(original, mode)
    request = context.public_request("Goal")
    rules = " ".join(request["rules"] + request["planning_preferences"]["rules"])
    assert "Unseen structures and uncertainty require questions." not in rules
    assert "Clarify ambiguous structures" not in rules
    assert "Respect supplied clarification answers" in rules
    assert "explicit delegation" in rules and "reviewable assumption" in rules
    assert "equivalent questions absent materially changed information" in rules
    for phrase in ("I don't know/care", "use judgment", "just do it", "i don't mind"):
        assert phrase in rules
    for system in (local._SYSTEM, codex._SYSTEM):
        assert "planning_preferences" in system
        assert "zero routine" in system and "one essential" in system
        assert "reviewable assumptions" in system and "clarification answers" in system
        assert "unanswered" in system and "scientific success" in system
    developer = codex._developer_instructions(request)
    assert json.dumps(request["planning_preferences"], ensure_ascii=False) in developer
    assert codex._proposal_schema(request)["properties"]["questions"]["maxItems"] == (
        maximum
    )
    assert "cannot authorize tools" in developer


@pytest.mark.parametrize("provider", ["local", "codex"])
@pytest.mark.parametrize("repair", [False, True])
def test_cap_failure_uses_only_existing_bounded_repair_with_unchanged_preferences(
    source_context, monkeypatch, provider, repair
):
    pipeline, original = source_context
    context = _context_mode(original, "use_best_judgment")
    rejected = reference_proposal(context)
    rejected["questions"] = [
        "Routine threshold choice?",
        "Routine connectivity choice?",
    ]
    corrected = reference_proposal(context)
    corrected["assumptions"].append(
        "Draft choice: face-connected observed bright signal is a reviewable "
        "starting convention, not proven biological identity."
    )
    replies = iter([json.dumps(rejected), json.dumps(corrected)])
    dialogue = ProposalDialogue(
        "Use best judgment.",
        (("Object count?", "Yes, two objects."),),
        ("Complete cavities?", "Boundary convention?"),
    )
    calls = []
    snapshot = GraphSnapshot.from_pipeline(pipeline)

    if provider == "local":

        def transport(_url, _kind, _route, payload, _timeout, _cancelled=None):
            calls.append(deepcopy(payload))
            return {
                "choices": [
                    {"message": {"content": next(replies)}, "finish_reason": "stop"}
                ]
            }

        monkeypatch.setattr(local, "_request_json", transport)

        def generate():
            return local.generate_proposal(
                context,
                dialogue,
                local.ModelSettings(
                    kind="openai-compatible",
                    base_url="http://127.0.0.1:8080/v1",
                    model="offline-test",
                    max_attempts=2 if repair else 1,
                ),
            )
    else:
        runtime = codex.CodexRuntime(
            "unused", "test", True, True, "Synthetic", "a" * 64
        )
        monkeypatch.setattr(codex, "discover_codex_runtime", lambda *_: runtime)

        def transport(request, *_args, **_kwargs):
            calls.append(deepcopy(request))
            return next(replies)

        monkeypatch.setattr(codex, "_call_codex", transport)

        def generate():
            return codex.generate_codex_proposal(
                context,
                dialogue,
                codex.CodexSettings(max_attempts=2 if repair else 1),
            )

    if repair:
        result = generate()
        assert [attempt.valid for attempt in result.attempts] == [False, True]
        assert (
            json.loads(result.attempts[0].response_text)["questions"]
            == (rejected["questions"])
        )
        assert "at most 1 unanswered question" in result.attempts[0].validation_error
        assert result.proposal == corrected and result.prepared.can_apply
    else:
        with pytest.raises(local.ModelProposalError) as caught:
            generate()
        assert len(caught.value.attempts) == 1
        assert (
            json.loads(caught.value.attempts[0].response_text)["questions"]
            == (rejected["questions"])
        )
    assert len(calls) == (2 if repair else 1)
    for call in calls:
        request = (
            json.loads(call["messages"][1]["content"]) if provider == "local" else call
        )
        assert request["planning_preferences"] == context.planning_preferences.to_dict()
        assert request["deferred_questions"] == list(dialogue.deferred_questions)
        assert request["clarifications"] == [
            {"question": "Object count?", "answer": "Yes, two objects."}
        ]
    assert GraphSnapshot.from_pipeline(pipeline) == snapshot
    assert not pipeline.outputs["input"].flags.writeable
