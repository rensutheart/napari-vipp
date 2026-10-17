"""Current-test evidence priority and baseline-descriptor regressions, no models."""

from dataclasses import replace

import pytest

from napari_vipp._tests.test_workflow_investigation import act
from napari_vipp._tests.test_workflow_investigation_warmstart import (
    accepted,
    engine_for,
    update,
)
from napari_vipp.core.workflow_investigation import (
    InvestigationError,
    InvestigationStopped,
    _json,
)
from napari_vipp.core.workflow_investigation_context import build_decision_context


def aliases(context):
    return [row["alias"] for row in context["numeric_evidence"]]


@pytest.mark.parametrize("threshold,status", [(90, "completed"), (150, "rejected")])
def test_completed_or_rejected_trial_uses_exact_baseline_descriptor(threshold, status):
    pipeline, snapshot, bundle = accepted(protection=True)
    engine = engine_for(pipeline, bundle)
    initial = engine.public_request()
    assert initial["accepted_candidate"] == engine.candidates[0].evidence
    baseline_record = engine._candidate_records["candidate0"]
    result = update(engine, threshold)
    assert result["status"] == "ok" and engine.candidates[-1].status == status
    outgoing = engine.public_request()
    descriptor = outgoing["accepted_candidate"]
    assert set(descriptor) == {"candidate_id", "status", "full_record", "meaning"}
    assert descriptor["full_record"] == baseline_record
    assert engine._notebook.lookup(**baseline_record) == engine.candidates[0].evidence
    assert outgoing["decision_context"]["recipes"][0]["candidate_id"] == "candidate1"
    assert engine.retained_candidate_id == "candidate0"
    assert engine.selected_candidate.graph == snapshot.baseline
    assert engine.model_calls_used == 0 and engine.attempted_trials == 1


def test_invalid_trial_does_not_displace_full_accepted_evidence(monkeypatch):
    import napari_vipp.core.workflow_investigation as module

    pipeline, _, bundle = accepted()
    engine = engine_for(pipeline, bundle)
    monkeypatch.setattr(
        module,
        "execute_pipeline_request",
        lambda *_a: pytest.fail("invalid trial executed"),
    )
    result = act(
        engine,
        "trial_edit",
        candidate_id="candidate0",
        edits=[
            {"kind": "set_parameters", "node": "missing", "params": {"threshold": 90}}
        ],
    ).public_result()
    assert result["status"] == "error" and engine.candidates[-1].status == "failed"
    assert not engine._has_completed_trial()
    assert (
        engine.public_request()["accepted_candidate"] == engine.candidates[0].evidence
    )
    assert engine.attempted_trials == 1


def test_unqualified_partial_output_is_not_a_completed_trial(monkeypatch):
    import napari_vipp.core.workflow_investigation as module

    pipeline, _, bundle = accepted()
    engine = engine_for(pipeline, bundle)
    original = module.execute_pipeline_request

    def unsafe(request):
        result = original(request)
        return replace(
            result,
            execution_report=replace(result.execution_report, cleanup_succeeded=False),
        )

    monkeypatch.setattr(module, "execute_pipeline_request", unsafe)
    with pytest.raises(InvestigationStopped, match="cleanup"):
        update(engine, 90)
    assert not engine._has_completed_trial()
    assert (
        engine.candidates[-1].status == "failed"
        and engine.candidates[-1]._outputs == ()
    )
    assert engine.retained_candidate_id == "candidate0"
    with pytest.raises(InvestigationStopped):
        engine.public_request()


def test_current_protection_mesh_and_latest_comparison_precede_baseline():
    pipeline, _, bundle = accepted(protection=True, mesh=True)
    engine = engine_for(pipeline, bundle)
    assert update(engine, 90)["status"] == "ok"
    compared = act(
        engine,
        "compare_candidates",
        left="candidate0",
        right="candidate1",
        roles=["band_mask"],
    ).public_result()["result"]
    inspected = act(
        engine, "inspect_candidate", candidate_id="candidate1", role="band_mesh"
    ).public_result()["result"]
    context = engine._decision_context(byte_limit=16 * 1024)
    current = aliases(context)
    wanted = {
        "candidate1/preservation",
        "candidate1/measurements/band_mesh",
        compared["evidence_ref"]["alias"],
        inspected["evidence_ref"]["alias"],
    }
    assert wanted <= set(current)
    assert current.index("candidate1/preservation") < current.index(
        "candidate1/completed"
    )
    assert len(_json(context)) <= 16 * 1024
    assert any(
        row["kind"] in {"recipes", "numeric_evidence", "node_contracts"}
        for row in context["omitted"]
    )
    # Speculative packing does not send any of these measured facts.
    assert engine._sent_numeric == set()
    reference = next(
        row["evidence_ref"]
        for row in context["numeric_evidence"]
        if row["alias"] == "candidate1/preservation"
    )
    with pytest.raises(InvestigationError, match="not transmitted"):
        engine._validate_evidence_refs([reference], candidate_id="candidate1")
    outgoing = engine.public_request()
    engine.record_model_call(request=outgoing)
    engine._validate_evidence_refs([reference], candidate_id="candidate1")


def test_deferred_baseline_entry_cannot_displace_current_complete_payload():
    pipeline, _, bundle = accepted(protection=True, mesh=True)
    engine = engine_for(pipeline, bundle)
    assert update(engine, 90)["status"] == "ok"
    full = engine.public_request()["decision_context"]
    current = [
        row
        for row in full["numeric_evidence"]
        if row["alias"].startswith("candidate1/")
    ]
    baseline = [
        row
        for row in full["numeric_evidence"]
        if row["alias"].startswith("candidate0/")
    ]
    packed = build_decision_context(
        session_id=engine.session_id,
        recipes=[],
        node_contracts=[],
        numeric_evidence=current,
        deferred_numeric_evidence=baseline,
        byte_limit=len(_json(current)) + 3 * 1024,
    )
    assert set(aliases(packed)) >= set(aliases({"numeric_evidence": current}))
    assert aliases(packed)[: len(current)] == aliases({"numeric_evidence": current})
    assert engine._sent_numeric == set()
