"""Synthetic decision facts/automatic preview carry; no providers or user data."""

import copy

import pytest

from napari_vipp._tests.test_workflow_investigation import act, session
from napari_vipp._tests.test_workflow_investigation_warmstart import (
    accepted,
    engine_for,
    update,
)
from napari_vipp.core.workflow_investigation import InvestigationError, _json
from napari_vipp.core.workflow_investigation_context import (
    MAX_CONTEXT_BYTES,
    DecisionContextError,
    build_decision_context,
    validate_decision_context,
)


def refs(context):
    return {row["alias"]: row["evidence_ref"] for row in context["numeric_evidence"]}


def test_warm_context_directly_cites_actual_protection_and_each_mesh_role():
    pipeline, _, bundle = accepted(protection=True, mesh=True)
    engine = engine_for(pipeline, bundle)
    request = engine.public_request()
    context = request["decision_context"]
    assert context["recipes"] == [engine._do_describe_candidate("candidate0")]
    issued = refs(context)
    assert "candidate0/preservation" in issued
    assert "candidate0/measurements/band_mesh" in issued
    assert engine._sent_numeric == set()
    engine.export_notebook()
    with pytest.raises(InvestigationError, match="not transmitted"):
        engine._validate_evidence_refs([issued["candidate0/preservation"]])
    engine.record_model_call(request=request)
    for reference in issued.values():
        engine._validate_evidence_refs([reference], candidate_id="candidate0")
    for row in context["numeric_evidence"]:
        if row["alias"] == "candidate0/measurements/band_mesh":
            measured = {
                output["role"]: output["diagnostics"]
                for output in engine.candidates[0].evidence["outputs"]
            }
            assert row["payload"]["diagnostics"] == measured["band_mesh"]


def test_trial_keeps_exact_retained_and_latest_recipes_without_another_run():
    pipeline, _, bundle = accepted(protection=True, mesh=True)
    engine = engine_for(pipeline, bundle)
    result = update(engine, 90)
    assert result["status"] == "ok"
    assert result["result"]["preservation_check"]["passed"]
    context = engine.public_request()["decision_context"]
    assert [row["candidate_id"] for row in context["recipes"]] == [
        "candidate1",
        "candidate0",
    ]
    assert {"candidate0/preservation", "candidate1/preservation"} <= refs(
        context
    ).keys()
    assert engine.retained_candidate_id == "candidate0"
    assert engine.attempted_trials == 1


def test_optional_context_uses_headroom_not_premature_closing(monkeypatch):
    import napari_vipp.core.workflow_investigation as module

    pipeline, _, bundle = accepted(protection=True, mesh=True)
    engine = engine_for(pipeline, bundle)
    full = engine.public_request()
    baseline = dict(full)
    baseline.pop("decision_context")
    assert not full["closing_phase"]["active"]
    target = len(_json(baseline)) + 1024
    monkeypatch.setattr(module, "WORKING_REQUEST_TARGET_BYTES", target)
    bounded = engine.public_request()
    assert len(_json(bounded)) <= target
    assert not bounded["closing_phase"]["active"]
    assert bounded["decision_context"]["omitted"]
    assert module.MAX_REQUEST_BYTES == 192 * 1024


def test_matching_baseline_trial_source_and_mesh_views_survive_pin_bookkeeping(
    monkeypatch,
):
    pipeline, _, bundle = accepted(protection=True, mesh=True)
    engine = engine_for(pipeline, bundle, include_input_images=True)
    options = [{"axis": "Z", "indices": [0], "montage": True, "overlay_source": True}]
    act(
        engine, "inspect_source", views=[{"axis": "Z", "indices": [0], "montage": True}]
    )
    baseline = act(
        engine,
        "inspect_candidate",
        candidate_id="candidate0",
        role="band_mask",
        views=options,
    ).public_result()
    baseline_hashes = {p["sha256"] for p in baseline["image_previews"]}
    assert update(engine, 90)["status"] == "ok"
    trial = act(
        engine,
        "inspect_candidate",
        candidate_id="candidate1",
        role="band_mask",
        views=options,
    ).public_result()
    trial_id = trial["result"]["view_id"]
    geometry = act(
        engine,
        "inspect_candidate",
        candidate_id="candidate1",
        role="band_mesh",
        views=[{"view_index": 0}, {"view_index": 1}],
    ).public_result()
    geometry_id = geometry["result"]["view_id"]
    before = engine.preview_images()
    assert baseline_hashes <= {p.manifest()["sha256"] for p in before}
    assert len(before) <= 7
    monkeypatch.setattr(
        engine, "_render_views", lambda *_a, **_k: pytest.fail("unexpected render")
    )
    monkeypatch.setattr(
        engine,
        "_render_mesh_views",
        lambda *_a, **_k: pytest.fail("unexpected mesh render"),
    )
    act(engine, "pin_previews", view_ids=[trial_id, geometry_id])
    assert baseline_hashes <= {p.manifest()["sha256"] for p in engine.preview_images()}
    assert engine._pinned_view_ids == (trial_id, geometry_id)
    act(engine, "find_nodes", query="threshold")
    assert len(engine.preview_images()) <= 7
    assert engine.attempted_trials == 1


def test_untransmitted_older_measurement_not_granted_by_a_quote_or_omission():
    engine = session()
    old = act(
        engine,
        "inspect_source",
        roi={"axis_order": "ZYX", "bounds": [[0, 1], [0, 2], [0, 2]]},
    ).public_result()["result"]
    latest = act(
        engine,
        "inspect_source",
        roi={"axis_order": "ZYX", "bounds": [[1, 2], [0, 2], [0, 2]]},
    ).public_result()["result"]
    request = engine.public_request()
    assert refs(request["decision_context"])["input"] == latest["evidence_ref"]
    engine.record_model_call(request=request)
    with pytest.raises(InvestigationError, match="not transmitted"):
        engine._validate_evidence_refs([old["evidence_ref"]])
    engine._validate_evidence_refs([latest["evidence_ref"]])


def test_small_builder_budget_keeps_whole_payloads_and_explicit_aggregate_omissions():
    source = session()
    contracts = [source._do_describe_node("binary_threshold")]
    result = build_decision_context(
        session_id=source.session_id,
        recipes=[],
        node_contracts=contracts,
        numeric_evidence=[],
        byte_limit=512,
        omissions=[
            {
                "kind": "numeric_evidence",
                "identity": "x" * 256,
                "reason": "A whole cached entry was not inlined.",
            }
        ]
        * 100,
    )
    assert len(_json(result)) <= 512
    assert result["node_contracts"] == []
    assert result["omitted"][-1]["kind"] == "additional_entries"
    assert len(_json(result)) <= MAX_CONTEXT_BYTES
    changed = copy.deepcopy(result)
    changed["node_contracts"] = [
        {"operation_id": "binary_threshold", "parameters": [], "trial_allowed": True}
    ]
    with pytest.raises(DecisionContextError):
        validate_decision_context(changed, session_id=source.session_id)


def test_cached_inspection_recall_refreshes_latest_alias_without_issuing_it():
    engine = session()
    first = act(engine, "inspect_source", percentiles=[10]).public_result()["result"]
    second = act(engine, "inspect_source", percentiles=[90]).public_result()["result"]
    assert first["evidence_ref"] != second["evidence_ref"]
    recalled = act(engine, "inspect_source", percentiles=[10]).public_result()
    assert recalled["inspection_reused"]
    assert (
        refs(engine.public_request()["decision_context"])["input"]
        == first["evidence_ref"]
    )
    assert engine._sent_numeric == set()
    engine.record_model_call(request=engine.public_request())
    engine._validate_evidence_refs([first["evidence_ref"]])
    with pytest.raises(InvestigationError, match="not transmitted"):
        engine._validate_evidence_refs([second["evidence_ref"]])
