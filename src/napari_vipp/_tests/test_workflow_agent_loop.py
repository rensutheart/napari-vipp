"""Offline continuing agent contracts; real shared CPU tools, no provider I/O."""

from __future__ import annotations

from copy import deepcopy
from threading import Event, Thread

import numpy as np
import pytest

from napari_vipp._tests.test_workflow_investigation import proposal, session
from napari_vipp._tests.test_workflow_investigation_warmstart import (
    accepted,
    engine_for,
)
from napari_vipp.core import workflow_investigation_model as provider
from napari_vipp.core.workflow_agent_loop import WorkflowAgentLoop
from napari_vipp.core.workflow_agent_tools import WorkflowAgentTools
from napari_vipp.core.workflow_investigation import (
    InvestigationAction,
    InvestigationError,
)
from napari_vipp.core.workflow_proposal_codex import CodexSettings


def install_model(monkeypatch, actions, *, reservations=1):
    requests = []
    iterator = iter(actions)

    def choose(request, _settings, **kwargs):
        requests.append(deepcopy(request))
        for _ in range(reservations):
            kwargs["before_call"]()
        name, arguments = next(iterator)
        if callable(arguments):
            arguments = arguments(request)
        return provider.ModelInvestigationResult(
            InvestigationAction(request["session_id"], name, arguments),
            (),
            "codex",
            "offline-fixture",
            "0" * 64,
        )

    monkeypatch.setattr(provider, "generate_codex_investigation_action", choose)
    return requests


def finish(candidate):
    return "finish", dict(
        candidate_id=candidate,
        summary="Offline orchestration test, not scientific validation.",
        limitations=["Synthetic test only."],
    )


def test_enabled_question_returns_control_then_resumes_same_headless_loop(monkeypatch):
    engine = session(continuation_enabled=True)
    agent = WorkflowAgentLoop(engine)
    requests = install_model(
        monkeypatch,
        [
            ("question", {"question": "Which foreground should be kept?"}),
            ("question", {"question": "What would you like to do next?"}),
        ],
    )
    assert agent.run_until_pause(CodexSettings()) is None
    assert engine.paused and not engine.stop_reason and len(requests) == 1
    engine.resume(answer="Keep brighter objects; use your judgment.")
    assert agent.run_until_pause(CodexSettings()) is None
    assert engine.paused and not engine.stop_reason and len(requests) == 2
    assert engine.actions_used == engine.model_calls_used == 2
    assert "Keep brighter objects" in str(requests[-1])


def test_goal_agent_runs_real_shared_trial_then_pauses_without_accepting(monkeypatch):
    engine = session(review_before_finish=True)
    agent = WorkflowAgentLoop(engine)
    original = engine._source.data.copy()
    calls = []
    dispatch = WorkflowAgentTools.call_tool

    def observe(self, name, arguments):
        calls.append((self, name, deepcopy(arguments)))
        return dispatch(self, name, arguments)

    monkeypatch.setattr(WorkflowAgentTools, "call_tool", observe)
    requests = install_model(
        monkeypatch,
        [
            (
                "trial_candidate",
                dict(
                    proposal=proposal(engine, 100),
                    output_roles=[dict(role="wanted", node="mask", port=0)],
                ),
            ),
            finish("candidate1"),
        ],
    )
    steps = []
    pending = agent.run_until_pause(CodexSettings(), on_step=steps.append)
    assert pending["candidate_id"] == "candidate1"
    assert not engine.finished and engine.attempted_trials == 1
    assert len(requests) == len(steps) == engine.model_calls_used == 2
    assert [name for _, name, _ in calls] == ["vipp_trial_candidate", "vipp_finish"]
    assert all(tool is agent.tools for tool, _, _ in calls)
    assert all(row["session_id"] == engine.session_id for row in requests)
    assert steps[0].tool_result.public_result()["result"]["status"] == "completed"
    np.testing.assert_array_equal(engine._source.data, original)
    assert agent.export_notebook() == engine.export_notebook()
    agent.accept_finish_review()
    assert engine.finished and engine.selected_candidate.candidate_id == "candidate1"


def test_warm_refinement_uses_same_loop_tools_notebook_and_review_continuation(
    monkeypatch,
):
    pipeline, snapshot, bundle = accepted(protection=True)
    engine = engine_for(pipeline, bundle, review_before_finish=True)
    agent = WorkflowAgentLoop(engine)
    original = pipeline.outputs[snapshot.candidate_node_ids[0]].copy()
    key = engine.candidates[0].output_roles[0]["node"]
    requests = install_model(
        monkeypatch,
        [
            finish("candidate0"),
            (
                "trial_edit",
                dict(
                    candidate_id="candidate0",
                    edits=[
                        dict(kind="set_parameters", node=key, params={"threshold": 80})
                    ],
                ),
            ),
            finish("candidate1"),
        ],
    )
    session_identity = (
        id(agent.session),
        engine.session_id,
        engine.deadline,
        engine.limits,
    )
    first = agent.run_until_pause(CodexSettings())
    assert first["candidate_id"] == "candidate0"
    before = engine.actions_used, engine.model_calls_used, engine.attempted_trials
    agent.continue_finish_review(
        feedback="Test a useful change while preserving my result."
    )
    assert before == (
        engine.actions_used,
        engine.model_calls_used,
        engine.attempted_trials,
    )
    second = agent.run_until_pause(CodexSettings())
    assert second["candidate_id"] == "candidate1"
    assert session_identity == (
        id(agent.session),
        engine.session_id,
        engine.deadline,
        engine.limits,
    )
    assert requests[1]["completion_review"]["continue_requests"] == 1
    assert requests[1]["accepted_candidate"]["candidate_id"] == "candidate0"
    assert second["continuation_allowed"]
    assert engine.candidates[-1].evidence["preservation_check"]["passed"]
    np.testing.assert_array_equal(
        pipeline.outputs[snapshot.candidate_node_ids[0]], original
    )
    snapshot.assert_current(pipeline)


def test_prepare_turn_is_passive_detached_and_cannot_overlap():
    engine = session()
    agent = WorkflowAgentLoop(engine)
    before = engine.export_notebook(), engine.actions_used, engine.model_calls_used
    turn = agent.prepare_turn()
    assert all(
        definition["name"].startswith("vipp_") for definition in turn.tool_definitions
    )
    assert {definition["name"] for definition in turn.tool_definitions} == {
        "vipp_" + name for name in turn.request["response_schema"]["actions"]
    }
    request = turn.request
    request["objective"] = "Modified external copy"
    definitions = turn.tool_definitions
    definitions.clear()
    assert turn.request["objective"] != "Modified external copy"
    assert turn.tool_definitions
    assert before == (
        engine.export_notebook(),
        engine.actions_used,
        engine.model_calls_used,
    )
    with pytest.raises(InvestigationError, match="already in progress"):
        agent.prepare_turn()


def test_interleaved_shared_tool_call_revokes_prepared_turn_before_provider(
    monkeypatch,
):
    engine = session()
    agent = WorkflowAgentLoop(engine)
    turn = agent.prepare_turn()
    # An external adapter has the same underlying session, not another copy.
    WorkflowAgentTools(engine).call_tool("vipp_find_nodes", {"query": "threshold"})
    monkeypatch.setattr(
        provider,
        "generate_codex_investigation_action",
        lambda *_a, **_k: pytest.fail("sent"),
    )
    with pytest.raises(InvestigationError, match="no longer belongs"):
        agent.choose_action(turn, CodexSettings())
    assert engine.model_calls_used == 0


def test_only_exact_decided_action_dispatches_once(monkeypatch):
    engine = session()
    agent = WorkflowAgentLoop(engine)
    install_model(monkeypatch, [("find_nodes", {"query": "threshold"})])
    turn = agent.prepare_turn()
    result = agent.choose_action(turn, CodexSettings())
    with pytest.raises(InvestigationError, match="not the agent's decided"):
        agent.execute_action(
            InvestigationAction(engine.session_id, "find_nodes", {"query": "mesh"})
        )
    assert engine.actions_used == 0
    receipt = agent.execute_action(result.action)
    assert receipt.public_result()["status"] == "ok" and engine.actions_used == 1
    with pytest.raises(InvestigationError, match="current decided"):
        agent.execute_action(result.action)
    assert engine.actions_used == 1


def test_bounded_repair_calls_are_counted_without_changing_turn_or_deadline(
    monkeypatch,
):
    engine = session()
    agent = WorkflowAgentLoop(engine)
    requests = install_model(
        monkeypatch, [("find_nodes", {"query": "threshold"})], reservations=2
    )
    deadline = engine.deadline
    step = agent.step(CodexSettings(max_attempts=2))
    assert engine.model_calls_used == 2 and engine.actions_used == 1
    assert engine.deadline == deadline and len(requests) == 1
    assert step.turn.request == requests[0]


def test_provider_error_does_not_retry_or_restore_consumed_call(monkeypatch):
    engine = session()
    agent = WorkflowAgentLoop(engine)
    calls = []

    def fail(_request, _settings, **kwargs):
        kwargs["before_call"]()
        calls.append(True)
        raise RuntimeError("Offline provider failure")

    monkeypatch.setattr(provider, "generate_codex_investigation_action", fail)
    with pytest.raises(RuntimeError, match="Offline provider failure"):
        agent.run_until_pause(CodexSettings())
    assert len(calls) == engine.model_calls_used == 1
    assert engine.actions_used == engine.attempted_trials == 0
    assert agent.prepare_turn().request["session_id"] == engine.session_id


def test_cancel_during_request_rejects_late_reply(monkeypatch):
    engine = session()
    agent = WorkflowAgentLoop(engine)

    def cancelled(request, _settings, **kwargs):
        kwargs["before_call"]()
        agent.cancel(user_decision=True)
        return provider.ModelInvestigationResult(
            InvestigationAction(
                request["session_id"], "find_nodes", {"query": "threshold"}
            ),
            (),
            "codex",
            "offline-fixture",
            "0" * 64,
        )

    monkeypatch.setattr(provider, "generate_codex_investigation_action", cancelled)
    with pytest.raises(InvestigationError, match="no longer belongs"):
        agent.step(CodexSettings())
    assert engine.stop_reason and engine.actions_used == 0
    assert engine.model_calls_used == 1


def test_question_stops_same_loop_without_automatic_new_session(monkeypatch):
    engine = session()
    agent = WorkflowAgentLoop(engine)
    requests = install_model(
        monkeypatch, [("question", {"question": "Which input is intended?"})]
    )
    assert agent.run_until_pause(CodexSettings()) is None
    assert len(requests) == 1 and engine.stop_reason
    assert not engine.finished
    assert agent.run_until_pause(CodexSettings()) is None and len(requests) == 1


def test_unaccounted_provider_reply_cannot_reach_any_tool(monkeypatch):
    engine = session()
    agent = WorkflowAgentLoop(engine)
    install_model(monkeypatch, [("find_nodes", {"query": "threshold"})], reservations=0)
    with pytest.raises(InvestigationError, match="no accounted outgoing"):
        agent.step(CodexSettings())
    assert engine.model_calls_used == engine.actions_used == 0


def test_preparing_during_review_does_not_strand_acceptance(monkeypatch):
    pipeline, _, bundle = accepted()
    engine = engine_for(pipeline, bundle, review_before_finish=True)
    agent = WorkflowAgentLoop(engine)
    install_model(monkeypatch, [finish("candidate0")])
    agent.run_until_pause(CodexSettings())
    with pytest.raises(InvestigationError, match="waiting for your review"):
        agent.prepare_turn()
    # No half-prepared turn prevents the separate explicit acceptance.
    agent.accept_finish_review()
    assert engine.finished and engine.model_calls_used == 1


def test_failed_shared_tool_invalidates_decision_without_restoring_budget(monkeypatch):
    engine = session()
    agent = WorkflowAgentLoop(engine)
    install_model(monkeypatch, [("find_nodes", {"query": "threshold"})])
    turn = agent.prepare_turn()
    result = agent.choose_action(turn, CodexSettings())

    def rejected(_name, _arguments):
        raise RuntimeError("Offline tool failure")

    monkeypatch.setattr(agent.tools, "call_tool", rejected)
    with pytest.raises(RuntimeError, match="Offline tool failure"):
        agent.execute_action(result.action)
    with pytest.raises(InvestigationError, match="current decided"):
        agent.execute_action(result.action)
    assert engine.model_calls_used == 1


@pytest.mark.parametrize("decision", ["continue", "accept"])
def test_review_transition_uses_same_session_lease_as_external_tools(
    monkeypatch, decision
):
    pipeline, _, bundle = accepted()
    engine = engine_for(pipeline, bundle, review_before_finish=True)
    agent = WorkflowAgentLoop(engine)
    install_model(monkeypatch, [finish("candidate0")])
    agent.run_until_pause(CodexSettings())
    entered, release = Event(), Event()

    def hold_shared_lease():
        with WorkflowAgentTools(engine).exclusive():
            entered.set()
            assert release.wait(5)

    worker = Thread(target=hold_shared_lease)
    worker.start()
    assert entered.wait(5)
    before = engine.actions_used, engine.model_calls_used, engine.attempted_trials
    try:
        with pytest.raises(InvestigationError, match="busy"):
            if decision == "continue":
                agent.continue_finish_review(feedback="Keep investigating.")
            else:
                agent.accept_finish_review()
        assert engine.pending_finish_review is not None
        assert not engine.finished
        assert before == (
            engine.actions_used,
            engine.model_calls_used,
            engine.attempted_trials,
        )
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive()
    agent.accept_finish_review()
    assert engine.finished
