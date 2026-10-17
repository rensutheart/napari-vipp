"""Synthetic counter/authority proof, not evidence of model or image quality."""

from napari_vipp._tests.test_workflow_investigation import act
from napari_vipp._tests.test_workflow_investigation_experiment_session import plan
from napari_vipp._tests.test_workflow_investigation_warmstart import (
    accepted,
    engine_for,
)


def send(engine):
    request = engine.public_request()
    engine.record_model_call(request=request)
    return request["decision_context"]


def evidence(context, alias):
    return next(
        row["evidence_ref"]
        for row in context["numeric_evidence"]
        if row["alias"] == alias
    )


def test_three_complete_synthetic_cycles_need_no_history_housekeeping():
    pipeline, snapshot, bundle = accepted()
    engine = engine_for(pipeline, bundle, require_experiment_plan=True)
    assert engine.limits.max_actions == 32
    assert engine.limits.max_model_calls == 40
    key = engine.candidates[0].output_roles[0]["node"]
    for threshold in (110, 120, 130):
        context = send(engine)
        assert context["recipes"][0]["scientific_recipe"]["nodes"]
        assert context["node_contracts"]
        planned = plan(
            engine,
            baseline="candidate0",
            refs=[evidence(context, "candidate0/measurements/band_mask")],
        ).public_result()
        assert planned["status"] == "ok", planned
        experiment_id = planned["result"]["experiment_id"]
        send(engine)
        trial = act(
            engine,
            "trial_edit",
            candidate_id="candidate0",
            edits=[
                {
                    "kind": "set_parameters",
                    "node": key,
                    "params": {"threshold": threshold},
                }
            ],
            experiment_id=experiment_id,
        ).public_result()
        assert trial["status"] == "ok", trial
        candidate_id = trial["result"]["candidate_id"]
        context = send(engine)
        assessed = act(
            engine,
            "assess_experiment",
            experiment_id=experiment_id,
            candidate_id=candidate_id,
            check_results=[
                {
                    "check_id": "regional_count",
                    "outcome": "unresolved",
                    "evidence_refs": [
                        evidence(context, f"{candidate_id}/measurements/band_mask")
                    ],
                    "assessment": (
                        "Exact full-array counts supplied; the prospective regional "
                        "criterion remains unresolved without its requested ROI."
                    ),
                }
            ],
            unresolved_criteria=["A mechanical test, not scientific validation."],
        ).public_result()
        assert assessed["status"] == "ok", assessed
    send(engine)
    finished = act(
        engine,
        "finish",
        candidate_id="candidate0",
        summary="Keep the accepted synthetic result; no accuracy claim.",
        limitations=["No provider, images or independent scientific validation."],
    ).public_result()
    assert finished["status"] == "ok", finished
    assert engine.finished and engine.attempted_trials == 3
    assert engine.actions_used == engine.model_calls_used == 10
    assert len(engine.experiment_ledger) == 3
    assert all(row["assessment"] is not None for row in engine.experiment_ledger)
    assert engine.selected_candidate.graph == snapshot.baseline
