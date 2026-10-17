"""Bounded refinement state and whole-graph no-publication boundaries."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from napari_vipp.core.pipeline import NODE_LIBRARY_BY_ID, GraphConnection
from napari_vipp.core.snapshots import GraphSnapshot, NodeSnapshot
from napari_vipp.core.workflow_proposal_catalogue import AUTHORABLE_OPERATION_IDS
from napari_vipp.core.workflow_refinement import (
    MAX_CALCULATION_REPAIRS,
    MAX_REFINEMENT_SECONDS,
    REFINEMENT_SAFE_OPERATION_IDS,
    RefinementError,
    RefinementLoop,
    assert_refinement_graph_safe,
    candidate_signature,
)


def graph(*nodes, connections=()):
    return GraphSnapshot([NodeSnapshot("input", "input", {}), *nodes], connections)


def node(key="a", op="binary_threshold", **params):
    return NodeSnapshot(key, op, params)


def initial(loop, *, overall="partly_achieved", now=1):
    loop.record_assessment("assessment-0", "run-0", overall, "Initial", now)


def round_trip(loop, index, *, outcome="partly_achieved", now=2):
    loop.begin_request("revision", ("same", True), now)
    loop.accept_revision(f"new-{index}", now=now)
    loop.run_started(now)
    loop.results_ready(now)
    loop.begin_request("assessment", ("same", True), now)
    return loop.record_assessment(
        f"assessment-{index}", f"run-{index}", outcome, "Result", now
    )


def start_attempt(loop, index, *, now=2):
    loop.begin_request("revision", ("same", True), now)
    loop.accept_revision(f"new-{index}", now=now)
    loop.run_started(now)


@pytest.mark.parametrize("limit", [0, -1, 6, 1.5, True, None, "2"])
def test_only_explicit_bounded_positive_start(limit):
    with pytest.raises(RefinementError):
        RefinementLoop.start(limit, ())
    with pytest.raises(TypeError, match="explicit"):
        RefinementLoop()


def test_initial_assessment_and_exact_additional_round_limit():
    loop = RefinementLoop.start(2, ("same", True), "baseline", now=0)
    assert loop.begin_request("assessment", ("same", True), 1) == 1799
    initial(loop)
    assert loop.completed_rounds == 0
    assert loop.active
    assert round_trip(loop, 1)
    assert not round_trip(loop, 2)
    assert loop.completed_rounds == 2
    assert loop.attempted_runs == 2
    assert loop.model_requests == loop.max_model_requests == 5
    assert "round limit" in loop.stop_reason
    assert [record.round_number for record in loop.history] == [0, 1, 2]
    assert isinstance(loop.history, tuple)
    with pytest.raises(RefinementError, match="round limit"):
        loop.begin_request("revision", ("same", True), 5)


def test_reused_initial_report_does_not_spend_model_allowance():
    loop = RefinementLoop.start(1, ("same", True), now=0)
    initial(loop)
    assert loop.model_requests == 0
    round_trip(loop, 1)
    assert loop.model_requests == 2


@pytest.mark.parametrize("outcome", ["achieved", "cannot_assess"])
def test_initial_stop_outcomes_never_start_an_extra_round(outcome):
    loop = RefinementLoop.start(5, (), now=0)
    initial(loop, overall=outcome)
    assert not loop.active and loop.completed_rounds == 0
    if outcome == "achieved":
        assert "assistant thinks" in loop.stop_reason
        assert "yourself" in loop.stop_reason


def test_achieved_is_a_model_judgment_not_scientific_success():
    loop = RefinementLoop.start(5, ("same", True), now=0)
    initial(loop)
    assert not round_trip(loop, 1, outcome="achieved")
    assert loop.completed_rounds == 1
    assert "assistant thinks" in loop.stop_reason


@pytest.mark.parametrize("binding", [("changed", True), ("same", False), ()])
def test_settings_permissions_or_goal_change_revokes_loop(binding):
    loop = RefinementLoop.start(2, ("same", True), now=0)
    with pytest.raises(RefinementError, match="changed"):
        loop.begin_request("assessment", binding, 1)
    assert not loop.active and loop.model_requests == 0


def test_caller_cannot_mutate_saved_binding_to_keep_changed_permission():
    binding = {"input": "original", "outputs": True}
    loop = RefinementLoop.start(1, binding, now=0)
    binding["outputs"] = False
    with pytest.raises(RefinementError, match="changed"):
        loop.assert_current(binding, now=1)


@pytest.mark.parametrize("phase", ["before_request", "run", "after_run"])
def test_deadline_enforced_during_every_stage(phase):
    loop = RefinementLoop.start(1, ("same", True), now=0)
    if phase != "before_request":
        initial(loop)
        loop.begin_request("revision", ("same", True), 2)
        loop.accept_revision("different", now=2)
        if phase == "after_run":
            loop.run_started(2)
    with pytest.raises(RefinementError, match="30-minute"):
        if phase == "before_request":
            loop.begin_request("assessment", ("same", True), MAX_REFINEMENT_SECONDS)
        elif phase == "run":
            loop.run_started(MAX_REFINEMENT_SECONDS)
        else:
            loop.results_ready(MAX_REFINEMENT_SECONDS)
    assert not loop.active


@pytest.mark.parametrize("clock", [float("nan"), float("inf"), "now", True])
def test_invalid_clock_rejected(clock):
    with pytest.raises(RefinementError, match="clock"):
        RefinementLoop.start(1, (), now=clock)


def test_cancel_and_failure_keep_first_stop_reason_and_history():
    loop = RefinementLoop.start(2, (), now=0)
    initial(loop)
    loop.cancel()
    loop.stop("A later callback must not erase the cancellation.")
    assert "cancelled" in loop.stop_reason
    assert len(loop.history) == 1
    with pytest.raises(RefinementError, match="cancelled"):
        loop.accept_revision("new", now=3)


def test_questions_stop_without_being_silently_dropped():
    loop = RefinementLoop.start(1, ("same", True), now=0)
    initial(loop)
    loop.begin_request("revision", ("same", True), 2)
    with pytest.raises(RefinementError, match="answer"):
        loop.accept_revision("new", ("How should objects be separated?",), 2)
    assert loop.completed_rounds == 0


def test_duplicate_draft_stops_before_add_or_run():
    loop = RefinementLoop.start(2, ("same", True), "original", now=0)
    initial(loop)
    loop.begin_request("revision", ("same", True), 2)
    with pytest.raises(RefinementError, match="unchanged draft"):
        loop.accept_revision("original", now=2)
    assert loop.completed_rounds == 0 and loop.phase == "stopped"


def test_duplicate_or_out_of_order_reports_do_not_count_twice():
    loop = RefinementLoop.start(2, (), now=0)
    initial(loop)
    with pytest.raises(RefinementError, match="outside"):
        initial(loop)
    assert loop.completed_rounds == 0 and len(loop.history) == 1


def test_only_valid_sequence_can_reserve_requests():
    loop = RefinementLoop.start(2, (), now=0)
    with pytest.raises(RefinementError, match="sequence"):
        loop.begin_request("revision", (), 1)
    assert loop.model_requests == 0


def test_duplicate_request_reservation_cannot_retry_or_burn_allowance():
    loop = RefinementLoop.start(2, (), now=0)
    loop.begin_request("assessment", (), 1)
    with pytest.raises(RefinementError, match="sequence"):
        loop.begin_request("assessment", (), 2)
    assert loop.model_requests == 1


def test_a_previous_report_cannot_be_reused_for_new_outputs_without_request():
    loop = RefinementLoop.start(1, ("same", True), now=0)
    initial(loop)
    loop.begin_request("revision", ("same", True), 2)
    loop.accept_revision("changed", now=2)
    loop.run_started(2)
    loop.results_ready(2)
    with pytest.raises(RefinementError, match="explicit assessment request"):
        loop.record_assessment("unrequested", "run-1", "partly_achieved", now=2)
    assert loop.completed_rounds == 0


def test_failed_attempt_is_not_a_completed_round_or_output_assessment():
    loop = RefinementLoop.start(3, ("same", True), now=0)
    initial(loop)
    start_attempt(loop, 1)
    requests_before = loop.model_requests
    assert loop.record_failure("failed-1", "Invalid input mesh", now=3)
    assert loop.attempted_runs == 1
    assert loop.completed_rounds == 0
    assert loop.calculation_repairs == 1
    assert loop.model_requests == requests_before
    assert loop.phase == "awaiting_revision"
    assert len(loop.history) == 1
    assert isinstance(loop.failures, tuple)
    (failure,) = loop.failures
    assert (failure.attempted_round, failure.run_id, failure.reason) == (
        1,
        "failed-1",
        "Invalid input mesh",
    )
    with pytest.raises(RefinementError, match="outside"):
        loop.results_ready(now=3)
    assert loop.completed_rounds == 0


@pytest.mark.parametrize("limit", [1, 2, 3])
def test_failed_runs_consume_every_selected_attempt(limit):
    loop = RefinementLoop.start(limit, ("same", True), now=0)
    initial(loop)
    for attempt in range(1, limit + 1):
        start_attempt(loop, attempt)
        assert loop.record_failure(f"failed-{attempt}", "Invalid input", now=3) is (
            attempt < limit
        )
    assert loop.attempted_runs == limit
    assert loop.completed_rounds == 0
    assert len(loop.failures) == limit
    assert "round limit" in loop.stop_reason
    with pytest.raises(RefinementError, match="round limit"):
        loop.begin_request("revision", ("same", True), now=4)
    assert loop.attempted_runs == limit


def test_no_third_calculation_repair_even_with_rounds_and_requests_remaining():
    loop = RefinementLoop.start(5, ("same", True), now=0)
    initial(loop)
    for attempt in range(1, MAX_CALCULATION_REPAIRS + 2):
        start_attempt(loop, attempt)
        assert loop.record_failure(f"failed-{attempt}", "Invalid input", now=3) is (
            attempt <= MAX_CALCULATION_REPAIRS
        )
    assert loop.calculation_repairs == MAX_CALCULATION_REPAIRS == 2
    assert loop.attempted_runs == 3 < loop.limit
    assert loop.model_requests == 3 < loop.max_model_requests
    assert len(loop.failures) == 3
    assert "calculation-repair limit" in loop.stop_reason


def test_assessment_history_retains_attempt_gaps_and_separate_completed_count():
    loop = RefinementLoop.start(4, ("same", True), now=0)
    initial(loop)
    start_attempt(loop, 1)
    assert loop.record_failure("failed-1", "Invalid input", now=3)
    assert round_trip(loop, 2)
    start_attempt(loop, 3)
    assert loop.record_failure("failed-3", "Invalid input", now=3)
    assert not round_trip(loop, 4)
    assert loop.attempted_runs == 4
    assert loop.completed_rounds == 2
    assert [record.round_number for record in loop.history] == [0, 2, 4]
    assert [record.attempted_round for record in loop.failures] == [1, 3]
    assert "round limit" in loop.stop_reason


@pytest.mark.parametrize("run_id", ["", None, True, (), 1.5])
def test_invalid_failed_run_identity_does_not_grant_repair(run_id):
    loop = RefinementLoop.start(2, ("same", True), now=0)
    initial(loop)
    start_attempt(loop, 1)
    with pytest.raises(RefinementError, match="invalid failed run"):
        loop.record_failure(run_id, "Invalid input", now=3)
    assert not loop.active and not loop.failures
    assert loop.calculation_repairs == 0


@pytest.mark.parametrize("reason", ["", "  ", None, {}, False])
def test_invalid_failure_reason_does_not_grant_repair(reason):
    loop = RefinementLoop.start(2, ("same", True), now=0)
    initial(loop)
    start_attempt(loop, 1)
    with pytest.raises(RefinementError, match="valid reason"):
        loop.record_failure("failed-1", reason, now=3)
    assert not loop.active and not loop.failures
    assert loop.calculation_repairs == 0


def test_replayed_failure_during_later_run_stops_without_second_repair():
    loop = RefinementLoop.start(3, ("same", True), now=0)
    initial(loop)
    start_attempt(loop, 1)
    assert loop.record_failure("failed-1", "Invalid input", now=3)
    start_attempt(loop, 2)
    with pytest.raises(RefinementError, match="duplicate"):
        loop.record_failure("failed-1", "Late duplicate", now=3)
    assert len(loop.failures) == loop.calculation_repairs == 1
    assert loop.completed_rounds == 0


def test_successful_run_identity_cannot_be_reclassified_as_failure():
    loop = RefinementLoop.start(3, ("same", True), now=0)
    initial(loop)
    assert round_trip(loop, 1)
    start_attempt(loop, 2)
    with pytest.raises(RefinementError, match="duplicate"):
        loop.record_failure("run-1", "Late error", now=3)
    assert not loop.failures and loop.calculation_repairs == 0
    assert loop.completed_rounds == 1


def test_failed_run_identity_cannot_be_blessed_as_completed_assessment():
    loop = RefinementLoop.start(3, ("same", True), now=0)
    initial(loop)
    start_attempt(loop, 1)
    assert loop.record_failure("failed-1", "Invalid input", now=3)
    start_attempt(loop, 2)
    loop.results_ready(now=3)
    loop.begin_request("assessment", ("same", True), now=3)
    with pytest.raises(RefinementError, match="duplicate"):
        loop.record_assessment("bad-assessment", "failed-1", "achieved", now=3)
    assert loop.completed_rounds == 0
    assert len(loop.history) == 1


@pytest.mark.parametrize("phase", ["initial", "revision", "ready", "assessment"])
def test_failure_outside_an_owned_running_attempt_stops_without_record(phase):
    loop = RefinementLoop.start(2, ("same", True), now=0)
    if phase != "initial":
        initial(loop)
        loop.begin_request("revision", ("same", True), now=2)
        if phase != "revision":
            loop.accept_revision("different", now=2)
            if phase == "assessment":
                loop.run_started(now=2)
                loop.results_ready(now=2)
                loop.begin_request("assessment", ("same", True), now=2)
    with pytest.raises(RefinementError, match="outside"):
        loop.record_failure("failed-1", "Invalid input", now=3)
    assert not loop.failures and loop.calculation_repairs == 0


@pytest.mark.parametrize("stop", ["cancel", "deadline"])
def test_late_failure_cannot_reopen_cancelled_or_expired_loop(stop):
    loop = RefinementLoop.start(2, ("same", True), now=0)
    initial(loop)
    start_attempt(loop, 1)
    if stop == "cancel":
        loop.cancel()
    with pytest.raises(RefinementError, match="cancelled|30-minute"):
        loop.record_failure(
            "failed-1",
            "Late failure",
            now=MAX_REFINEMENT_SECONDS if stop == "deadline" else 3,
        )
    assert not loop.active and not loop.failures
    assert loop.calculation_repairs == 0


def test_failure_does_not_reset_model_budget_or_whole_loop_deadline():
    loop = RefinementLoop.start(3, ("same", True), now=0)
    loop.begin_request("assessment", ("same", True), now=1)
    initial(loop)
    start_attempt(loop, 1, now=600)
    assert loop.record_failure("failed-1", "Invalid input", now=1200)
    assert loop.deadline == MAX_REFINEMENT_SECONDS
    assert loop.max_model_requests == 7
    assert loop.model_requests == 2
    assert loop.begin_request("revision", ("same", True), now=1500) == 300
    assert loop.model_requests == 3
    with pytest.raises(RefinementError, match="30-minute"):
        loop.accept_revision("repaired", now=MAX_REFINEMENT_SECONDS)
    assert loop.attempted_runs == 1


def test_failure_repair_must_still_reserve_existing_request_allowance():
    loop = RefinementLoop.start(2, ("same", True), now=0)
    initial(loop)
    start_attempt(loop, 1)
    assert loop.record_failure("failed-1", "Invalid input", now=3)
    # Fail closed if the connector's reserved logical request budget has already
    # been exhausted; calculation repair grants no separate model allowance.
    loop.model_requests = loop.max_model_requests
    with pytest.raises(RefinementError, match="model-request limit"):
        loop.begin_request("revision", ("same", True), now=4)
    assert loop.model_requests == loop.max_model_requests == 5
    assert loop.attempted_runs == 1


def test_failed_draft_signature_cannot_be_retried_as_a_repair():
    loop = RefinementLoop.start(2, ("same", True), now=0)
    initial(loop)
    start_attempt(loop, 1)
    assert loop.record_failure("failed-1", "Invalid input", now=3)
    loop.begin_request("revision", ("same", True), now=4)
    with pytest.raises(RefinementError, match="unchanged draft"):
        loop.accept_revision("new-1", now=4)
    assert loop.attempted_runs == 1


def test_whole_graph_contains_all_reviewed_registry_science_but_no_writers():
    assert REFINEMENT_SAFE_OPERATION_IDS == set(AUTHORABLE_OPERATION_IDS) - {
        "save_output"
    }
    all_safe = graph(
        *(node(str(i), op) for i, op in enumerate(REFINEMENT_SAFE_OPERATION_IDS))
    )
    assert_refinement_graph_safe(all_safe, "input")


@pytest.mark.parametrize(
    "op",
    ["save_output", "batch_output", "table_source", "custom_code", "future_writer"],
)
@pytest.mark.parametrize("mode", ["run", "bypass"])
def test_baseline_or_disabled_writer_or_unknown_operation_refused(op, mode):
    unsafe = graph(NodeSnapshot("old", op, {"enabled": False}, mode), node("new"))
    with pytest.raises(RefinementError, match="whole workflow"):
        assert_refinement_graph_safe(unsafe, "input")


def test_other_sources_and_missing_selected_source_refused():
    with pytest.raises(RefinementError, match="another source"):
        assert_refinement_graph_safe(graph(node("other", "input")), "input")
    with pytest.raises(RefinementError, match="selected input"):
        assert_refinement_graph_safe(GraphSnapshot([node()]), "input")


def test_new_registry_operation_is_not_automatically_allowed(monkeypatch):
    monkeypatch.setitem(
        NODE_LIBRARY_BY_ID,
        "future_operation",
        replace(NODE_LIBRARY_BY_ID["binary_threshold"], id="future_operation"),
    )
    with pytest.raises(RefinementError, match="not reviewed"):
        assert_refinement_graph_safe(graph(node(op="future_operation")), "input")


def test_candidate_signature_ignores_id_and_order_but_keeps_ports_params_and_sharing():
    first = graph(
        node("a", low_threshold=10),
        node("b", "invert"),
        connections=(GraphConnection("input", "a"), GraphConnection("a", "b")),
    )
    renamed = graph(
        node("new_b", "invert"),
        node("new_a", low_threshold=10),
        connections=(
            GraphConnection("new_a", "new_b"),
            GraphConnection("input", "new_a"),
        ),
    )
    digest = candidate_signature(first, ("a", "b"), "input")
    assert digest == candidate_signature(renamed, ("new_b", "new_a"), "input")
    changed = graph(
        node("a", low_threshold=11),
        node("b", "invert"),
        connections=first.connections,
    )
    assert digest != candidate_signature(changed, ("a", "b"), "input")
    different_port = GraphSnapshot(
        first.nodes, (GraphConnection("input", "a"), GraphConnection("a", "b", 1))
    )
    assert digest != candidate_signature(different_port, ("a", "b"), "input")
    duplicate = GraphSnapshot(
        (*first.nodes, node("c", "invert")),
        (*first.connections, GraphConnection("a", "c")),
    )
    assert digest != candidate_signature(duplicate, ("a", "b", "c"), "input")


def test_only_actual_derived_spatial_cache_is_ignored():
    unresolved = graph(node("a", "remove_small_objects", resolved_spatial_ndim=None))
    resolved = graph(node("a", "remove_small_objects", resolved_spatial_ndim=3))
    assert candidate_signature(unresolved, ("a",), "input") == candidate_signature(
        resolved, ("a",), "input"
    )
    authored = graph(node("a", "remove_small_objects", spatial_mode="3D ZYX"))
    assert candidate_signature(authored, ("a",), "input") != candidate_signature(
        resolved, ("a",), "input"
    )


@pytest.mark.parametrize(
    "edge", [GraphConnection("unreviewed", "a"), GraphConnection("a", "a")]
)
def test_invalid_candidate_dependencies_refused(edge):
    with pytest.raises(RefinementError):
        candidate_signature(graph(node(), connections=(edge,)), ("a",), "input")


def test_widget_cancellation_only_owns_exact_active_dispatch():
    from napari_vipp._widget import VippWidget

    cancel = Mock()
    fake = SimpleNamespace(
        _workflow_assistant_owned_dispatch=(12, object()),
        _active_pipeline_run_id=13,
        _cancel_background_pipeline_run=cancel,
    )
    assert not VippWidget._cancel_workflow_assistant_run(fake)
    cancel.assert_not_called()
    fake._active_pipeline_run_id = 12
    assert VippWidget._cancel_workflow_assistant_run(fake)
    cancel.assert_called_once()
    assert fake._workflow_assistant_owned_dispatch is None
    assert not VippWidget._cancel_workflow_assistant_run(fake)
    cancel.assert_called_once()


def test_generic_dispatch_erases_old_assistant_cancellation_ticket():
    from napari_vipp._widget import VippWidget

    fake = SimpleNamespace(
        _workflow_assistant_assessment_armed=None,
        _workflow_assistant_owned_dispatch=(10, object()),
        _workflow_assistant_assessment_serial=0,
        _invalidate_workflow_assessment=Mock(),
    )
    VippWidget._begin_workflow_assessment_run(fake)
    assert fake._workflow_assistant_owned_dispatch is None


def test_assistant_dispatch_gets_ticket_independent_of_evidence_invalidation():
    from napari_vipp._widget import VippWidget

    revision = object()
    fake = SimpleNamespace(
        _workflow_assistant_assessment_armed=revision,
        _workflow_assistant_owned_dispatch=None,
        _workflow_assistant_assessment_serial=0,
        _active_pipeline_run_id=12,
        _invalidate_workflow_assessment=Mock(),
    )
    VippWidget._begin_workflow_assessment_run(fake)
    assert fake._workflow_assistant_owned_dispatch == (12, revision)
    assert fake._workflow_assistant_assessment_pending == (1, "background-12", revision)


@pytest.mark.parametrize(
    "accepted", ["denied", "queued", "pending", "completed", "other"]
)
def test_run_return_only_accepts_exact_completed_or_pending_revision(accepted):
    from napari_vipp._widget import VippWidget

    revision = SimpleNamespace(
        added_node_ids=("candidate",),
        context=SimpleNamespace(output_destination=None),
        owner=("session", "workflow"),
    )
    fake = SimpleNamespace(
        _workflow_assistant_run_revision=revision,
        _closing=False,
        _workflow_proposal_edit_guard=Mock(),
        node_name_editor=SimpleNamespace(commit_pending=Mock()),
        _finish_parameter_history_group=Mock(),
        _assert_workflow_proposal_current=Mock(),
        _workflow_assistant_output_assessment_enabled=lambda: True,
        _workflow_assistant_assessment_serial=0,
        _workflow_assistant_assessment_pending=None,
        _workflow_assistant_assessment_run=None,
        _workflow_assistant_failed_run=None,
    )

    def invalidate():
        fake._workflow_assistant_assessment_pending = None
        fake._workflow_assistant_assessment_run = None

    def calculate():
        if accepted == "pending":
            fake._workflow_assistant_assessment_pending = (1, "background-1", revision)
        elif accepted in {"completed", "other"}:
            fake._workflow_assistant_assessment_run = SimpleNamespace(
                revision=revision if accepted == "completed" else object()
            )

    fake._invalidate_workflow_assessment = invalidate
    fake._calculate_all_nodes = calculate
    assert VippWidget._run_workflow_assistant_plan(fake, ("candidate",)) == (
        accepted in {"pending", "completed"}
    )
    assert fake._workflow_assistant_assessment_armed is None


@pytest.mark.parametrize("dispatched", [True, False])
def test_manual_run_opt_out_reports_dispatch_without_assessment_authority(dispatched):
    from napari_vipp._widget import VippWidget

    revision = SimpleNamespace(
        added_node_ids=("candidate",),
        context=SimpleNamespace(output_destination=None),
        owner=("session", "workflow"),
    )
    fake = SimpleNamespace(
        _workflow_assistant_run_revision=revision,
        _closing=False,
        _workflow_proposal_edit_guard=Mock(),
        node_name_editor=SimpleNamespace(commit_pending=Mock()),
        _finish_parameter_history_group=Mock(),
        _assert_workflow_proposal_current=Mock(),
        _workflow_assistant_output_assessment_enabled=lambda: False,
        _workflow_assistant_assessment_serial=9,
        _workflow_assistant_assessment_armed=None,
        _workflow_assistant_assessment_pending=None,
        _workflow_assistant_assessment_run=None,
        _invalidate_workflow_assessment=Mock(),
    )

    def calculate():
        assert fake._workflow_assistant_assessment_armed is None
        if dispatched:
            fake._workflow_assistant_assessment_serial += 1

    fake._calculate_all_nodes = calculate
    assert VippWidget._run_workflow_assistant_plan(fake, ("candidate",)) is dispatched
    assert fake._workflow_assistant_assessment_pending is None
    assert fake._workflow_assistant_assessment_run is None
