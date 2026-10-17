"""A first review preserves original capacity for a real continuation trial."""

import json
from copy import deepcopy

import pytest

from napari_vipp._tests.test_workflow_investigation import act, proposal, session
from napari_vipp._tests.test_workflow_investigation_experiment_session import (
    assessment,
    linked_trial,
    plan,
)
from napari_vipp._tests.test_workflow_investigation_finish_review import nominate
from napari_vipp._tests.test_workflow_investigation_warmstart import (
    accepted,
    engine_for,
)
from napari_vipp.core import workflow_investigation_model as model
from napari_vipp.core import workflow_proposal_model as local
from napari_vipp.core.workflow_investigation import (
    InvestigationError,
    InvestigationLimits,
    InvestigationStopped,
)
from napari_vipp.core.workflow_investigation_review_budget import (
    REVIEW_ACTION_RESERVE,
    REVIEW_CALL_RESERVE,
)


def sent(engine, calls=1):
    request = engine.public_request()
    model._request(request, engine.preview_images())
    for _ in range(calls):
        engine.record_model_call(request=request)
    return request


def warm(*, planned=True, limits=None):
    pipeline, snapshot, bundle = accepted(protection=True, mesh=True)
    engine = engine_for(
        pipeline,
        bundle,
        review_before_finish=True,
        require_experiment_plan=planned,
        limits=limits or InvestigationLimits(max_attempts=3),
    )
    return pipeline, snapshot, bundle, engine


def warm_cycle(engine, bundle, parent, threshold, calls=1):
    request = sent(engine, calls)
    reference = next(
        row["evidence_ref"]
        for row in request["decision_context"]["numeric_evidence"]
        if row["alias"] == f"{parent}/preservation"
    )
    planned = plan(engine, baseline=parent, refs=[reference]).public_result()
    assert planned["status"] == "ok", planned
    identifier = planned["result"]["experiment_id"]
    sent(engine, calls)
    trial = act(
        engine,
        "trial_edit",
        candidate_id=parent,
        experiment_id=identifier,
        edits=[
            {
                "kind": "set_parameters",
                "node": bundle.output_roles[0]["node"],
                "params": {"threshold": threshold},
            }
        ],
    ).public_result()
    assert trial["status"] == "ok", trial
    current = trial["result"]["candidate_id"]
    request = sent(engine, calls)
    reference = next(
        row["evidence_ref"]
        for row in request["decision_context"]["numeric_evidence"]
        if row["alias"] == f"{current}/preservation"
    )
    checked = assessment(engine, identifier, current, refs=[reference]).public_result()
    assert checked["status"] == "ok", checked
    return current


def test_three_original_slots_split_two_then_real_continued_third_assessed_trial():
    pipeline, snapshot, bundle, engine = warm()
    first = warm_cycle(engine, bundle, "candidate0", 90)
    second = warm_cycle(engine, bundle, first, 80)
    request = engine.public_request()
    allocation = request["completion_review"]["reservation"]
    assert allocation["state"] == "held" and allocation["initial_trial_limit"] == 2
    assert allocation["reserved_trial_attempts"] == 1
    assert request["closing_phase"]["active"]
    assert not {"plan_experiment", "trial_candidate", "trial_edit"} & set(
        request["response_schema"]["actions"]
    )
    before = engine.actions_used, engine.attempted_trials, engine.export_notebook()
    with pytest.raises(InvestigationError, match="reserved"):
        act(engine, "trial_edit", candidate_id=second, edits=[])
    with pytest.raises(InvestigationError, match="reserved"):
        engine._reserve_trial({}, [], None)
    assert before == (
        engine.actions_used,
        engine.attempted_trials,
        engine.export_notebook(),
    )
    sent(engine)
    nominate(engine, second)
    assert engine.pending_finish_review["continuation_allowed"]
    authority = (
        engine.session_id,
        engine.deadline,
        engine.limits,
        engine.actions_used,
        engine.model_calls_used,
        engine.attempted_trials,
        engine._image_bytes,
        engine._accepted_bundle,
        engine._preservation,
    )
    records = engine.export_notebook()["records"]
    engine.continue_finish_review("Consider the remaining unmet requirements.")
    assert authority == (
        engine.session_id,
        engine.deadline,
        engine.limits,
        engine.actions_used,
        engine.model_calls_used,
        engine.attempted_trials,
        engine._image_bytes,
        engine._accepted_bundle,
        engine._preservation,
    )
    assert engine.export_notebook()["records"][: len(records)] == records
    allocation = engine.public_request()["completion_review"]["reservation"]
    assert allocation["state"] == "released" and allocation["initial_trial_limit"] == 2
    assert [
        allocation[key]
        for key in (
            "reserved_trial_attempts",
            "reserved_actions",
            "reserved_model_calls",
        )
    ] == [0, 0, 0]
    third = warm_cycle(engine, bundle, second, 70)
    sent(engine)
    result = nominate(engine, third)
    engine.accept_finish_review()
    assert result["retained_candidate_id"] == third and engine.attempted_trials == 3
    assert all(row["state"] == "assessed" for row in engine.experiment_ledger)
    assert engine.candidates[-1].evidence["preservation_check"]["missing_count"] == 0
    snapshot.assert_current(pipeline)


@pytest.mark.parametrize("calls", [1, 2])
def test_advertised_closing_read_cannot_drain_future_room(calls):
    _, _, _, engine = warm()
    engine.actions_used = engine.limits.max_actions - 9
    engine.model_calls_used = engine.limits.max_model_calls - 20
    request = sent(engine, calls)
    assert "inspect_source" in request["closing_phase"]["allowed_actions"]
    assert act(engine, "inspect_source").public_result()["status"] == "ok"
    request = sent(engine, calls)
    assert "inspect_source" not in request["closing_phase"]["allowed_actions"]
    nominate(engine)
    remaining = engine.pending_finish_review["remaining_budgets"]
    assert remaining["actions_remaining"] >= REVIEW_ACTION_RESERVE
    assert remaining["model_calls_remaining"] >= REVIEW_CALL_RESERVE
    assert engine.pending_finish_review["continuation_allowed"]
    engine.continue_finish_review("Try a bounded useful follow-up.")
    assert "trial_edit" in engine.public_request()["response_schema"]["actions"]


@pytest.mark.parametrize("actions,calls", [(12, 23), (32, 23), (12, 40)])
def test_lowest_eligible_goal_start_reaches_real_trial_assessment_and_finish(
    actions, calls
):
    engine = session(
        review_before_finish=True,
        require_experiment_plan=True,
        limits=InvestigationLimits(
            max_attempts=3, max_actions=actions, max_model_calls=calls
        ),
    )
    assert (
        engine.public_request()["completion_review"]["reservation"]["state"] == "held"
    )
    sent(engine, 2)
    planned = plan(engine).public_result()
    assert planned["status"] == "ok", planned
    identifier = planned["result"]["experiment_id"]
    request = sent(engine, 2)
    assert "trial_candidate" in request["response_schema"]["actions"]
    trial = linked_trial(engine, identifier, threshold=90).public_result()
    assert trial["status"] == "ok", trial
    current = trial["result"]["candidate_id"]
    sent(engine, 2)
    assert assessment(engine, identifier, current).public_result()["status"] == "ok"
    sent(engine, 2)
    nominate(engine, current)
    assert engine.pending_finish_review["continuation_allowed"]
    assert engine.pending_finish_review["remaining_budgets"]["actions_remaining"] >= 7
    assert (
        engine.pending_finish_review["remaining_budgets"]["model_calls_remaining"] >= 14
    )


@pytest.mark.parametrize(
    "planned,actions,calls",
    [(True, 11, 40), (True, 32, 22), (False, 10, 40), (False, 32, 20)],
)
def test_small_original_limits_disclose_unavailable_without_blocking_sole_initial_work(
    planned, actions, calls
):
    engine = session(
        review_before_finish=True,
        require_experiment_plan=planned,
        limits=InvestigationLimits(
            max_attempts=2, max_actions=actions, max_model_calls=calls
        ),
    )
    reservation = engine.public_request()["completion_review"]["reservation"]
    assert reservation["state"] == "unavailable"
    assert (
        reservation["initial_trial_limit"] == 2
        and reservation["reserved_trial_attempts"] == 0
    )
    assert "no allowance is added" in reservation["reason"]
    identifier = None
    if planned:
        sent(engine, 2)
        identifier = plan(engine).public_result()["result"]["experiment_id"]
    sent(engine, 2)
    args = {"experiment_id": identifier} if planned else {}
    result = act(
        engine,
        "trial_candidate",
        proposal=proposal(engine),
        output_roles=[{"role": "wanted", "node": "mask", "port": 0}],
        **args,
    ).public_result()
    assert result["status"] == "ok" and engine.attempted_trials == 1


@pytest.mark.parametrize("warm_start", [False, True])
def test_single_trial_limit_remains_usable_and_explicitly_unreserved(warm_start):
    if warm_start:
        _, _, bundle, engine = warm(
            planned=False, limits=InvestigationLimits(max_attempts=1)
        )
        from napari_vipp._tests.test_workflow_investigation_warmstart import update

        result = update(engine, 90)
    else:
        engine = session(
            review_before_finish=True, limits=InvestigationLimits(max_attempts=1)
        )
        result = act(
            engine,
            "trial_candidate",
            proposal=proposal(engine),
            output_roles=[{"role": "wanted", "node": "mask", "port": 0}],
        ).public_result()
    assert result["status"] == "ok" and engine.attempted_trials == 1
    assert (
        engine.public_request()["completion_review"]["reservation"]["state"]
        == "unavailable"
    )


def test_failed_continue_keeps_held_allocation_and_original_notebook(monkeypatch):
    _, _, _, engine = warm()
    nominate(engine)
    before = engine.export_notebook(), engine._review_reservation(), engine.deadline
    original = engine._finish_continuation_status
    seen = []

    def deny_after_commit(**kwargs):
        seen.append(engine._continue_requests)
        return (
            (False, "Synthetic prospective context refusal.")
            if engine._continue_requests
            else original(**kwargs)
        )

    monkeypatch.setattr(engine, "_finish_continuation_status", deny_after_commit)
    with pytest.raises(InvestigationError, match="prospective"):
        engine.continue_finish_review("Keep evaluating remaining requirements.")
    assert seen == [0, 1]
    assert before == (
        engine.export_notebook(),
        engine._review_reservation(),
        engine.deadline,
    )
    assert (
        engine._continue_requests == 0
        and engine.pending_finish_review["candidate_id"] == "candidate0"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", True),
        ("state", "extra"),
        ("total_trial_limit", True),
        ("initial_trial_limit", 3),
        ("reserved_trial_attempts", 0),
        ("reserved_actions", 8),
        ("reserved_model_calls", 15),
        ("reason", ""),
    ],
)
def test_changed_reservation_wire_is_rejected_before_provider(
    field, value, monkeypatch
):
    _, _, _, engine = warm()
    request = engine.public_request()
    request["completion_review"]["reservation"][field] = value
    from napari_vipp.core import workflow_proposal_codex as codex

    monkeypatch.setattr(
        codex, "discover_codex_runtime", lambda *_: pytest.fail("provider")
    )
    with pytest.raises(local.ModelProposalError):
        model.generate_codex_investigation_action(request, codex.CodexSettings())
    assert (
        engine.actions_used == engine.model_calls_used == engine.attempted_trials == 0
    )


def test_reservation_optional_old_review_wire_roundtrip_and_legacy_capacity_unchanged():
    _, _, _, engine = warm()
    request = engine.public_request()
    detached, wire = model._request(request, engine.preview_images())
    assert detached == json.loads(local._json_bytes(request))
    assert wire == local._json_bytes(request)
    old = deepcopy(request)
    del old["completion_review"]["reservation"]
    assert model._request(old, engine.preview_images())[0] == json.loads(
        local._json_bytes(old)
    )
    legacy = session(limits=InvestigationLimits(max_attempts=3))
    assert "completion_review" not in legacy.public_request()
    for threshold in (90, 80, 70):
        result = act(
            legacy,
            "trial_candidate",
            proposal=proposal(legacy, threshold),
            output_roles=[{"role": "wanted", "node": "mask", "port": 0}],
        ).public_result()
        assert result["status"] == "ok", result
    assert legacy.attempted_trials == 3
    request = legacy.public_request()
    assert model._request(request, ())[1] == local._json_bytes(request)


def test_repeated_finish_without_new_trial_never_replenishes_released_allocation():
    _, _, _, engine = warm()
    nominate(engine)
    deadline = engine.deadline
    engine.continue_finish_review("Investigate unresolved requirements.")
    assert engine._review_reservation()["state"] == "released"
    sent(engine, 2)
    nominate(engine)
    before = engine.actions_used, engine.model_calls_used, engine.attempted_trials
    engine.continue_finish_review("Keep the remaining requirements explicit.")
    assert engine._continue_requests == 2
    assert engine._review_reservation()["state"] == "released"
    assert [
        engine._review_reservation()[key]
        for key in (
            "reserved_trial_attempts",
            "reserved_actions",
            "reserved_model_calls",
        )
    ] == [0, 0, 0]
    assert before == (
        engine.actions_used,
        engine.model_calls_used,
        engine.attempted_trials,
    )
    assert engine.deadline == deadline and engine.attempted_trials == 0


def test_stale_continue_does_not_release_held_trial_or_reset_authority():
    pipeline, _, bundle, engine = warm()
    nominate(engine)
    before = (
        engine.actions_used,
        engine.model_calls_used,
        engine.attempted_trials,
        engine.deadline,
        engine.export_notebook(),
    )
    node_id = bundle.key_to_node[bundle.output_roles[0]["node"]]
    pipeline.nodes[node_id].params["threshold"] = 101
    with pytest.raises(InvestigationStopped):
        engine.continue_finish_review("Do not continue stale work.")
    assert (
        engine._continue_requests == 0
        and engine._review_reservation()["state"] == "held"
    )
    assert before == (
        engine.actions_used,
        engine.model_calls_used,
        engine.attempted_trials,
        engine.deadline,
        engine.export_notebook(),
    )


@pytest.mark.parametrize(
    "budgets",
    [
        None,
        [],
        {"attempts_remaining": True},
        {"attempts_remaining": -1},
        {"attempts_remaining": 4},
    ],
)
def test_reservation_rejects_malformed_budget_context_before_provider(
    budgets, monkeypatch
):
    _, _, _, engine = warm()
    request = engine.public_request()
    request["budgets"] = budgets
    from napari_vipp.core import workflow_proposal_codex as codex

    monkeypatch.setattr(
        codex, "discover_codex_runtime", lambda *_: pytest.fail("provider")
    )
    with pytest.raises(local.ModelProposalError):
        model.generate_codex_investigation_action(request, codex.CodexSettings())
    assert (
        engine.actions_used == engine.model_calls_used == engine.attempted_trials == 0
    )


def test_invalid_finish_still_costs_original_allowance_without_releasing_trial():
    _, _, _, engine = warm()
    engine.actions_used = engine.limits.max_actions - 8
    engine.model_calls_used = engine.limits.max_model_calls - 16
    deadline = engine.deadline
    sent(engine, 2)
    result = act(
        engine, "finish", candidate_id="candidate0", summary="", limitations=[]
    ).public_result()
    assert result["status"] == "error"
    assert engine.limits.max_actions - engine.actions_used == 7
    assert engine.limits.max_model_calls - engine.model_calls_used == 14
    assert engine.pending_finish_review is None and not engine.finished
    assert (
        engine._continue_requests == 0
        and engine._review_reservation()["state"] == "held"
    )
    assert engine.deadline == deadline and engine.attempted_trials == 0
    assert (
        "Failed responses consume original allowance"
        in engine._review_reservation()["reason"]
    )


def test_minimum_warm_limits_can_nominate_baseline_and_release_without_new_trial():
    _, _, _, engine = warm(
        limits=InvestigationLimits(max_attempts=2, max_actions=8, max_model_calls=16)
    )
    request = sent(engine, 2)
    assert request["completion_review"]["reservation"]["state"] == "held"
    assert "finish" in request["response_schema"]["actions"]
    nominate(engine)
    assert engine.attempted_trials == 0
    assert engine.pending_finish_review["continuation_allowed"]
    assert engine.pending_finish_review["remaining_budgets"]["actions_remaining"] == 7
    assert (
        engine.pending_finish_review["remaining_budgets"]["model_calls_remaining"] == 14
    )
    deadline = engine.deadline
    engine.continue_finish_review("Please investigate the unresolved limitations.")
    assert engine._review_reservation()["state"] == "released"
    assert engine.deadline == deadline and engine.attempted_trials == 0

    request = sent(engine, 2)
    ref = next(
        row["evidence_ref"]
        for row in request["decision_context"]["numeric_evidence"]
        if row["alias"] == "candidate0/preservation"
    )
    identifier = plan(engine, baseline="candidate0", refs=[ref]).public_result()[
        "result"
    ]["experiment_id"]
    sent(engine, 2)
    trial = act(
        engine,
        "trial_edit",
        candidate_id="candidate0",
        experiment_id=identifier,
        edits=[
            {
                "kind": "set_parameters",
                "node": engine.candidates[0].output_roles[0]["node"],
                "params": {"threshold": 90},
            }
        ],
    ).public_result()
    assert trial["status"] == "ok", trial
    current = trial["result"]["candidate_id"]
    for _ in range(2):
        request = sent(engine, 2)
        assert "inspect_source" in request["response_schema"]["actions"]
        assert act(engine, "inspect_source").public_result()["status"] == "ok"
    request = sent(engine, 2)
    ref = next(
        row["evidence_ref"]
        for row in request["decision_context"]["numeric_evidence"]
        if row["alias"] == f"{current}/preservation"
    )
    assert (
        assessment(engine, identifier, current, refs=[ref]).public_result()["status"]
        == "ok"
    )
    sent(engine, 2)
    nominate(engine, current)
    assert engine.pending_finish_review["remaining_budgets"]["actions_remaining"] == 1
    assert (
        engine.pending_finish_review["remaining_budgets"]["model_calls_remaining"] == 2
    )
    assert engine.deadline == deadline and engine.attempted_trials == 1
    assert not engine.pending_finish_review["continuation_allowed"]


def test_failed_initial_attempts_consume_initial_slots_not_the_held_trial():
    engine = session(
        review_before_finish=True, limits=InvestigationLimits(max_attempts=3)
    )
    for attempt in (1, 2):
        sent(engine)
        invalid = proposal(engine)
        invalid["nodes"][0]["operation_id"] = "not_an_existing_operation"
        result = act(
            engine,
            "trial_candidate",
            proposal=invalid,
            output_roles=[{"role": "wanted", "node": "mask", "port": 0}],
        ).public_result()
        assert result["status"] == "error"
        assert engine.attempted_trials == attempt
        assert engine.candidates[-1].status == "failed"
    request = sent(engine)
    assert request["completion_review"]["reservation"]["state"] == "held"
    assert "trial_candidate" not in request["response_schema"]["actions"]
    with pytest.raises(InvestigationError, match="reserved"):
        engine._reserve_trial({}, [], None)
    assert engine.attempted_trials == 2
    assert engine.selected_candidate is None and engine.pending_finish_review is None


def test_genuine_question_at_held_floor_stops_without_fabricated_finish():
    engine = session(
        review_before_finish=True, limits=InvestigationLimits(max_attempts=3)
    )
    engine.actions_used = engine.limits.max_actions - 8
    engine.model_calls_used = engine.limits.max_model_calls - 16
    request = sent(engine, 2)
    assert request["completion_review"]["reservation"]["state"] == "held"
    assert "question" in request["response_schema"]["actions"]
    result = act(
        engine, "question", question="Which result is scientifically acceptable?"
    ).public_result()
    assert result["status"] == "ok" and engine.stop_reason
    assert engine.pending_finish_review is None and not engine.finished
    assert engine.selected_candidate is None and engine.attempted_trials == 0
    before = engine.actions_used, engine.model_calls_used, engine.export_notebook()
    with pytest.raises(InvestigationStopped):
        engine.record_model_call(request=request)
    with pytest.raises(InvestigationStopped):
        act(engine, "question", question="Must not repeat.")
    with pytest.raises(InvestigationError):
        engine.continue_finish_review("Must not fabricate a nomination.")
    assert before == (
        engine.actions_used,
        engine.model_calls_used,
        engine.export_notebook(),
    )


def test_lowest_eligible_unplanned_goal_reaches_real_trial_and_finish():
    engine = session(
        review_before_finish=True,
        limits=InvestigationLimits(max_attempts=2, max_actions=11, max_model_calls=21),
    )
    request = sent(engine, 2)
    assert request["completion_review"]["reservation"]["state"] == "held"
    result = act(
        engine,
        "trial_candidate",
        proposal=proposal(engine),
        output_roles=[{"role": "wanted", "node": "mask", "port": 0}],
    ).public_result()
    assert result["status"] == "ok", result
    sent(engine, 2)
    nominate(engine, result["result"]["candidate_id"])
    assert engine.pending_finish_review["continuation_allowed"]
    assert engine.pending_finish_review["remaining_budgets"]["actions_remaining"] >= 7
    assert (
        engine.pending_finish_review["remaining_budgets"]["model_calls_remaining"] >= 14
    )
