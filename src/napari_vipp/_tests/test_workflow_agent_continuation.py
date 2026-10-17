"""Provider-free same-session owner continuation, never a budget reset."""

from __future__ import annotations

import threading
from dataclasses import replace

import pytest

from napari_vipp._tests.test_workflow_agent_tools import (
    make_session,
    probe,
    threshold_proposal,
)
from napari_vipp._tests.test_workflow_investigation import act
from napari_vipp._tests.test_workflow_investigation_finish_review import nominate
from napari_vipp.core import workflow_agent_probes as probe_engine
from napari_vipp.core import workflow_investigation as engine
from napari_vipp.core import workflow_investigation_model as model
from napari_vipp.core.workflow_agent_budget import (
    HARD_ALLOWANCE,
    validate_continuation_context,
)
from napari_vipp.core.workflow_investigation import (
    InvestigationError,
    InvestigationLimits,
    InvestigationPaused,
    InvestigationStopped,
)


def ready(**kwargs):
    ticks = [100.0]
    limits = kwargs.pop("limits", InvestigationLimits(max_seconds=600))
    pipeline, session, tools = make_session(
        continuation_enabled=True, clock=lambda: ticks[0], limits=limits, **kwargs
    )
    return pipeline, session, tools, ticks


def trial(session):
    proposal, roles = threshold_proposal(session)
    result = act(session, "trial_candidate", proposal=proposal, output_roles=roles)
    assert result.public_result()["status"] == "ok", result.public_result()
    return result.public_result()["result"]["candidate_id"]


def counters(session):
    return (
        session.actions_used,
        session.attempted_trials,
        session.model_calls_used,
        session._image_bytes,
        session.attempted_probes,
        session._probe_snapshot_bytes,
    )


def record(session):
    session.record_model_call(request=session.public_request())


def test_pause_grant_retains_actual_trials_probes_counters_png_grants_and_journal():
    pipeline, session, tools, ticks = ready(include_input_images=True)
    record(session)
    assert probe(tools).public_result()["status"] == "ok"
    candidate_id = trial(session)
    record(session)
    prior = counters(session)
    notebook = session.export_notebook()["records"]
    candidates, probes = session.candidates, session.diagnostic_probes
    evidence = (set(session._sent_numeric), set(session._sent_previews))
    source, dialogue, identifier = session._source, session.dialogue, session.session_id
    ticks[0] += 13
    assert session.request_pause("Review the current result.")
    assert session.paused and session.stop_reason == ""
    ticks[0] += 7200
    assert session.continuation_state["active_seconds_used"] == 13
    assert session.continuation_state["paused_seconds"] == 7200
    with pytest.raises(InvestigationPaused):
        session.public_request()
    approved = InvestigationLimits(
        max_attempts=5,
        max_actions=64,
        max_model_calls=80,
        max_seconds=900,
        max_image_bytes=20 * 1024 * 1024,
    )
    session.grant_allowance(approved, resume=True)
    assert not session.paused and session.session_id == identifier
    assert counters(session) == prior
    assert session.candidates == candidates and session.diagnostic_probes == probes
    assert session._source is source and session.dialogue is dialogue
    assert session._pipeline is pipeline
    assert evidence == (session._sent_numeric, session._sent_previews)
    assert session.export_notebook()["records"][: len(notebook)] == notebook
    assert session.deadline == 100 + 900 + 7200
    assert session.continuation_state["active_seconds_used"] == 13
    assert session._candidate(candidate_id).status == "completed"
    assert session._experiments.max_experiments == approved.max_attempts
    assert session._preview_cache.image_limit_bytes == approved.max_image_bytes
    assert validate_continuation_context(session.public_request()["continuation"])


def test_question_answer_resumes_exact_original_goal_and_history_in_same_session():
    _, session, _, ticks = ready()
    objective, identifier = session.dialogue.objective, session.session_id
    result = act(session, "question", question="Should touching regions stay separate?")
    assert result.public_result()["result"]["paused"] is True
    assert (
        session.paused
        and session.pending_question == "Should touching regions stay separate?"
    )
    assert session.continuation_state["question"] == session.pending_question
    assert session.stop_reason == "" and not session.finished
    before = counters(session)
    ticks[0] += 5000
    session.resume("Yes, use your judgment for the exact boundary.")
    assert session.session_id == identifier and session.dialogue.objective == objective
    assert session.dialogue.clarifications == (
        (
            "Should touching regions stay separate?",
            "Yes, use your judgment for the exact boundary.",
        ),
    )
    assert counters(session) == before and session.pending_question is None
    request = session.public_request()
    assert request["objective"] == objective and request["clarifications"]
    assert request["budgets"]["seconds_remaining"] == 600


@pytest.mark.parametrize("answer", [None, "", "x" * 2001, "C:/private/data"])
def test_bad_question_answers_and_failed_combined_grant_are_atomic(answer):
    _, session, _, _ = ready()
    act(session, "question", question="Keep the regions separate?")
    before = (
        session.export_notebook(),
        session.limits,
        session.deadline,
        session.dialogue,
    )
    with pytest.raises(InvestigationError):
        session.grant_allowance(
            replace(session.limits, max_actions=64), resume=True, answer=answer
        )
    assert (
        session.export_notebook(),
        session.limits,
        session.deadline,
        session.dialogue,
    ) == before
    assert session.paused and session.pending_question and session.stop_reason == ""


def test_non_question_resume_cannot_inject_unbound_answer():
    _, session, _, _ = ready()
    session.pause()
    with pytest.raises(InvestigationError, match="exact pending"):
        session.resume("An unsolicited answer.")
    assert session.paused and not session.dialogue.clarifications


def test_pause_during_action_is_nonblocking_and_follows_complete_action_receipt(
    monkeypatch,
):
    _, session, tools, ticks = ready()
    entered, release = threading.Event(), threading.Event()
    result, errors = [], []

    def step(query):
        entered.set()
        assert release.wait(5)
        ticks[0] += 3
        session._guard()
        return [query]

    def worker():
        try:
            result.append(tools.call_tool("vipp_find_nodes", {"query": "generic"}))
        except BaseException as exc:
            errors.append(exc)

    monkeypatch.setattr(session, "_do_find_nodes", step)
    thread = threading.Thread(target=worker)
    thread.start()
    try:
        assert entered.wait(5)
        assert session.request_pause("External controller disconnected.")
        assert not session.paused and session.continuation_state["state"] == "pausing"
        with pytest.raises(InvestigationError, match="busy"):
            session.resume()
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive() and not errors
    assert result[0].public_result()["status"] == "ok"
    assert session.paused and session.actions_used == 1 and session.stop_reason == ""
    kinds = [row["kind"] for row in session.export_notebook()["record_index"]]
    assert kinds.index("action_receipt") < kinds.index("session_pause")
    session.resume()
    assert not session.paused and session.continuation_state["active_seconds_used"] == 3


@pytest.mark.parametrize("kind", ["trial", "probe"])
def test_real_ordinary_execution_finishes_cleanup_and_captures_outputs_before_pause(
    monkeypatch, kind
):
    _, session, tools, _ = ready()
    binding = engine if kind == "trial" else probe_engine
    execute = binding.execute_pipeline_request
    reports = []

    def during_execution(request):
        session.request_pause("Owner requested review during the current step.")
        assert not session.paused and not request.cancel_event.is_set()
        result = execute(request)
        reports.append(result.execution_report)
        return result

    monkeypatch.setattr(binding, "execute_pipeline_request", during_execution)
    if kind == "trial":
        trial(session)
        assert session.candidates[0].status == "completed"
    else:
        assert probe(tools).public_result()["status"] == "ok"
        assert session.diagnostic_probes[0]["status"] == "completed"
    assert reports and reports[0].cleanup_succeeded is True
    assert session.paused and session.stop_reason == ""
    session.resume()
    assert not session.paused


def test_cleanup_failure_remains_terminal_even_when_owner_requested_soft_pause(
    monkeypatch,
):
    _, session, _, _ = ready()
    execute = engine.execute_pipeline_request

    def failed_cleanup(request):
        session.request_pause("Review after the current step.")
        result = execute(request)
        return replace(
            result,
            execution_report=replace(result.execution_report, cleanup_succeeded=False),
        )

    monkeypatch.setattr(engine, "execute_pipeline_request", failed_cleanup)
    with pytest.raises(InvestigationStopped, match="cleanup"):
        trial(session)
    assert not session.paused and session.candidates[0].status == "failed"
    with pytest.raises(InvestigationStopped):
        session.resume()


def test_owner_continuation_invalidates_prepared_context_but_grants_no_evidence():
    _, session, tools, _ = ready(include_input_images=True)
    transmission = tools.prepare_transmission()
    evidence = set(session._sent_numeric), set(session._sent_previews)
    session.pause()
    session.resume()
    with pytest.raises(InvestigationError):
        tools.assert_transmission_current(transmission)
    assert evidence == (session._sent_numeric, session._sent_previews)
    assert not {"vipp_resume", "vipp_grant_allowance", "vipp_pause"} & {
        tool["name"] for tool in tools.list_tools()
    }
    current = tools.prepare_transmission()
    model._request(current.request, current.previews)


def test_soft_time_checkpoint_reports_completed_step_overshoot_and_absolute_grants(
    monkeypatch,
):
    _, session, _, ticks = ready()

    def long_step(query):
        ticks[0] += 700
        session._guard()
        assert not session._cancel_check()
        return [query]

    monkeypatch.setattr(session, "_do_find_nodes", long_step)
    assert act(session, "find_nodes", query="generic").public_result()["status"] == "ok"
    assert session.paused and session.continuation_state["active_seconds_used"] == 700
    assert session._seconds_remaining() == 0
    ticks[0] += 10000
    with pytest.raises(InvestigationError, match="exhausted"):
        session.grant_allowance(replace(session.limits, max_seconds=650), resume=True)
    session.grant_allowance(replace(session.limits, max_seconds=900), resume=True)
    assert session._seconds_remaining() == 200
    assert session.continuation_state["active_seconds_used"] == 700


def test_idle_time_checkpoint_can_resume_after_lunch_only_with_additional_allowance():
    _, session, _, ticks = ready()
    ticks[0] += 605
    with pytest.raises(InvestigationPaused):
        session.public_request()
    ticks[0] += 4000
    with pytest.raises(InvestigationError, match="exhausted"):
        session.resume()
    session.grant_allowance(replace(session.limits, max_seconds=900), resume=True)
    assert session._seconds_remaining() == 295
    assert session.deadline == 100 + 900 + 4000


def test_action_allowance_extension_retains_all_usage_and_closing_reserves():
    _, session, _, _ = ready(limits=InvestigationLimits(max_actions=4, max_seconds=600))
    # Closing reserves the last slot for a genuine question/Finish, not a
    # fourth free read. Exhaust it honestly with a Question, then add allowance
    # and its answer without resetting any of the four consumed actions.
    for _ in range(3):
        assert act(session, "inspect_source").public_result()["status"] == "ok"
    act(session, "question", question="May I consider another bounded test?")
    assert session.actions_used == 4 and session.stop_reason == ""
    with pytest.raises(InvestigationError, match="exhausted"):
        session.resume("Yes.")
    session.grant_allowance(
        replace(session.limits, max_actions=8), resume=True, answer="Yes."
    )
    assert act(session, "inspect_source").public_result()["status"] == "ok"
    assert session.actions_used == 5


def test_trial_attempt_allowance_can_increase_without_reusing_ids_or_old_pixels():
    _, session, _, _ = ready(
        limits=InvestigationLimits(max_attempts=1, max_seconds=600)
    )
    first = trial(session)
    before = session.candidates[0]
    recipe, roles = threshold_proposal(session, thresholds=(150,))
    with pytest.raises(InvestigationPaused, match="attempt budget"):
        act(session, "trial_candidate", proposal=recipe, output_roles=roles)
    assert session.actions_used == 1 and session.attempted_trials == 1
    session.grant_allowance(replace(session.limits, max_attempts=2), resume=True)
    result = act(session, "trial_candidate", proposal=recipe, output_roles=roles)
    assert result.public_result()["status"] == "ok"
    assert first == "candidate1" and session.candidates[1].candidate_id == "candidate2"
    assert session.candidates[0] is before and session.attempted_trials == 2


@pytest.mark.parametrize("field,value", [("max_seconds", 1801), ("max_actions", 65)])
def test_forged_frozen_limits_are_revalidated_at_owner_grant(field, value):
    _, session, _, _ = ready()
    session.pause()
    forged = replace(session.limits, max_actions=64)
    object.__setattr__(forged, field, value)
    before = session.export_notebook(), session.limits
    with pytest.raises(InvestigationError):
        session.grant_allowance(forged, resume=True)
    assert (session.export_notebook(), session.limits) == before
    assert session.paused and session.stop_reason == ""


def test_direct_allowance_replacement_cannot_bypass_owner_grant():
    _, session, _, _ = ready()
    session.pause()
    session.limits = replace(session.limits, max_actions=64)
    with pytest.raises(InvestigationStopped, match="outside its grant"):
        session.resume()
    assert session.stop_reason and session._allowance_grants == 0


def test_actual_model_call_allowance_pauses_before_send_without_replenishment():
    _, session, _, _ = ready(
        limits=InvestigationLimits(max_model_calls=2, max_seconds=600)
    )
    record(session)
    record(session)
    with pytest.raises(InvestigationPaused, match="model-call"):
        record(session)
    assert session.model_calls_used == 2
    session.grant_allowance(replace(session.limits, max_model_calls=8), resume=True)
    record(session)
    assert session.model_calls_used == 3


def test_actual_png_transmissions_keep_cumulative_bytes_across_image_allowance_grant():
    _, session, _, _ = ready(
        include_input_images=True,
        limits=InvestigationLimits(max_seconds=600, max_image_bytes=1000),
    )
    board = sum(len(p.png_bytes) for p in session.preview_images())
    assert 0 < board <= 1000
    for _ in range(1000 // board):
        record(session)
    spent, calls = session._image_bytes, session.model_calls_used
    with pytest.raises(InvestigationPaused, match="transmitted-image"):
        record(session)
    assert (session._image_bytes, session.model_calls_used) == (spent, calls)
    session.grant_allowance(replace(session.limits, max_image_bytes=2000), resume=True)
    record(session)
    assert (
        session._image_bytes == spent + board and session.model_calls_used == calls + 1
    )


@pytest.mark.parametrize(
    "invalidate", ["cancel", "source", "permission", "clock", "hard_time"]
)
def test_terminal_safety_invalidations_cannot_be_resumed_or_extended(invalidate):
    pipeline, session, _, ticks = ready()
    if invalidate == "hard_time":
        ticks[0] += 1800
        with pytest.raises(InvestigationStopped):
            session.public_request()
    else:
        session.pause()
        if invalidate == "cancel":
            session.cancel()
        elif invalidate == "source":
            pipeline.outputs["input"] = pipeline.outputs["input"].copy() + 1
        elif invalidate == "permission":
            session.include_output_images = False
        else:
            ticks[0] = float("nan")
    with pytest.raises(InvestigationStopped):
        session.resume()
    with pytest.raises(InvestigationStopped):
        session.grant_allowance(replace(session.limits, max_actions=64))
    assert session.stop_reason


def test_finish_review_can_extend_then_continue_without_spending_human_review_time():
    _, session, _, ticks = ready(review_before_finish=True)
    candidate_id = trial(session)
    nominate(session, candidate_id)
    assert session.paused and session.pending_finish_review
    before = counters(session)
    ticks[0] += 8000
    assert (
        session.pending_finish_review["remaining_budgets"]["seconds_remaining"] == 600
    )
    session.grant_allowance(replace(session.limits, max_seconds=900))
    with pytest.raises(InvestigationError, match="Keep improving"):
        session.resume()
    session.continue_finish_review("Consider any remaining defects.")
    assert not session.paused and session.pending_finish_review is None
    assert counters(session) == before
    assert session.deadline == 100 + 900 + 8000
    assert session._seconds_remaining() == 900
    assert session._continue_requests == 1


def test_legacy_question_and_elapsed_guard_stay_terminal_without_opt_in():
    ticks = [0.0]
    _, session, _ = make_session(clock=lambda: ticks[0])
    assert not session.continuation_enabled
    act(session, "question", question="An unanswered legacy question?")
    assert session.stop_reason and not session.paused
    with pytest.raises(InvestigationError, match="not enabled"):
        session.resume("An answer.")
    with pytest.raises(InvestigationError, match="not enabled"):
        session.request_pause()


@pytest.mark.parametrize(
    "mutation", ["hard", "unlimited", "grant_count", "extra", "bool"]
)
def test_continuation_wire_metadata_cannot_expand_hard_resource_limits(mutation):
    _, session, _, _ = ready()
    context = session.public_request()["continuation"]
    assert context["hard_limits"] == HARD_ALLOWANCE
    if mutation == "hard":
        context["hard_limits"]["max_attempts"] = 6
    elif mutation == "unlimited":
        context["allowance"]["max_seconds"] = float("inf")
    elif mutation == "grant_count":
        context["grants_used"] = 65
    elif mutation == "extra":
        context["auto_resume"] = True
    else:
        context["allowance"]["max_actions"] = True
    with pytest.raises(ValueError):
        validate_continuation_context(context)
