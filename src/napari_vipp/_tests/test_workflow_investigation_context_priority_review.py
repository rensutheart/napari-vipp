"""Real graph/context pressure regression, tiny synthetic provider-free data."""

from copy import deepcopy

import numpy as np
import pytest

from napari_vipp._tests.test_workflow_investigation import act
from napari_vipp.core.compute import ComputeRequest
from napari_vipp.core.metadata import AxisMetadata, image_state_from_array
from napari_vipp.core.pipeline import PrototypePipeline, SourcePayload
from napari_vipp.core.workflow_assessment import capture_assessment_snapshot
from napari_vipp.core.workflow_investigation import (
    InvestigationError,
    InvestigationSession,
    _json,
    capture_completed_candidate,
)
from napari_vipp.core.workflow_investigation_model import _request
from napari_vipp.core.workflow_refinement import capture_preservation_constraint


def completed_33_node_8_role_candidate():
    """Actually execute ordinary nodes; no recorded user/reference recipe."""
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    data = np.arange(210, dtype=np.float32).reshape(5, 6, 7).copy()
    data.setflags(write=False)
    state = image_state_from_array(
        data,
        axes=tuple(AxisMetadata(axis, "space", "um", scale=0.5) for axis in "zyx"),
    )
    nodes = []

    def add(operation, source, *, port=0, **params):
        node = pipeline.add_node(operation)
        node.params.update(params)
        assert pipeline.connect(source, node.id, target_port=port).success
        nodes.append(node.id)
        return node.id

    upstream = "input"
    for _ in range(21):
        upstream = add(
            "gaussian_blur_3d", upstream, sigma_x=0.0, sigma_y=0.0, sigma_z=0.0
        )
    threshold_a = add("binary_threshold", upstream, foreground="Above", threshold=100)
    mask_a = add(
        "remove_small_objects",
        threshold_a,
        min_size=1,
        spatial_mode="3D ZYX",
        connectivity="Face connected",
    )
    labels_a = add(
        "label_connected_components",
        mask_a,
        spatial_mode="3D ZYX",
        connectivity="Face connected",
    )
    threshold_b = add("binary_threshold", upstream, foreground="Above", threshold=150)
    mask_b = add(
        "remove_small_objects",
        threshold_b,
        min_size=1,
        spatial_mode="3D ZYX",
        connectivity="Face connected",
    )
    labels_b = add(
        "label_connected_components",
        mask_b,
        spatial_mode="3D ZYX",
        connectivity="Face connected",
    )
    mesh_a = add("mask_to_3d_mesh", mask_a, boundary="Close at image border")
    mesh_b = add("mask_to_3d_mesh", mask_b, boundary="Close at image border")
    merged = add("combine_meshes", mesh_a, input_count=2)
    assert pipeline.connect(mesh_b, merged, target_port=1).success
    coloured = add("color_mesh_objects", merged)
    background = add(
        "subtract_background",
        upstream,
        radius=2,
        spatial_mode="3D ZYX",
        disable_smoothing=True,
    )
    diagnostic = add(
        "gaussian_blur_3d", background, sigma_x=0.0, sigma_y=0.0, sigma_z=0.0
    )
    assert len(nodes) == 33
    pipeline.run(
        data,
        source_payloads={
            "input": SourcePayload(
                data, image_state=state, axis_semantics_resolved=True
            )
        },
        manual_node_ids=nodes,
    )
    assert all(pipeline.node_execution_states[node] == "ready" for node in nodes)
    snapshot = capture_assessment_snapshot(
        pipeline, "input", tuple(nodes), "synthetic-owned"
    )
    constraint = capture_preservation_constraint(
        snapshot, (mask_a, 0), (mask_b, 0), output_sharing=True
    )
    roles = [
        {"role": role, "node": node, "port": 0}
        for role, node in (
            ("mask_a", mask_a),
            ("mask_b", mask_b),
            ("labels_a", labels_a),
            ("labels_b", labels_b),
            ("mesh_a", mesh_a),
            ("mesh_b", mesh_b),
            ("combined_mesh", coloured),
            ("diagnostic", diagnostic),
        )
    ]
    bundle = capture_completed_candidate(
        pipeline,
        snapshot,
        roles,
        preservation_constraint=constraint,
        protected_role="mask_a",
    )
    engine = InvestigationSession.from_completed(
        pipeline,
        bundle,
        "Preserve the accepted region and assess an ordinary bounded edit.",
        compute_request=ComputeRequest(),
        include_input_images=True,
        include_output_images=True,
        assert_current=lambda: None,
        require_experiment_plan=True,
    )
    assert len(engine.candidates[0].recipe["nodes"]) == 33
    assert len(engine.candidates[0].output_roles) == 8
    threshold_key = next(
        key
        for key, node in engine.candidates[0].key_to_node.items()
        if node == threshold_a
    )
    return pipeline, snapshot, engine, threshold_key


def test_current_facts_survive_real_33_node_8_role_and_32k_explicit_pins(monkeypatch):
    import napari_vipp.core.workflow_investigation as module

    pipeline, snapshot, engine, threshold_key = completed_33_node_8_role_candidate()
    first = engine.public_request()
    assert first["accepted_candidate"] == engine.candidates[0].evidence
    engine.record_model_call(request=first)
    baseline_reference = next(
        row["evidence_ref"]
        for row in first["decision_context"]["numeric_evidence"]
        if row["alias"] == "candidate0/measurements/mesh_a"
    )
    planned = act(
        engine,
        "plan_experiment",
        baseline_candidate_id="candidate0",
        evidence_refs=[baseline_reference],
        observed_failure="Synthetic stress case, not a scientific judgment.",
        hypothesis=(
            "One explicit threshold edit can be checked without changing other nodes."
        ),
        prior_assumptions=["No biological truth is supplied."],
        predicted_geometric_change="Some foreground may change.",
        risks=["The proposed edit may be inaccurate."],
        intent="cleanup",
        evaluation_checks=[
            {
                "check_id": "retention",
                "kind": "numeric",
                "criterion": "Retain protected foreground.",
            },
            {
                "check_id": "closure",
                "kind": "numeric",
                "criterion": "Inspect exact mesh diagnostics.",
            },
        ],
    ).public_result()
    assert planned["status"] == "ok", planned
    plan = planned["result"]
    trial = act(
        engine,
        "trial_edit",
        candidate_id="candidate0",
        experiment_id=plan["experiment_id"],
        edits=[
            {
                "kind": "set_parameters",
                "node": threshold_key,
                "params": {"threshold": 90},
            }
        ],
    ).public_result()
    assert trial["status"] == "ok" and trial["result"]["preservation_check"]["passed"]
    inspection = act(
        engine, "inspect_candidate", candidate_id="candidate1", role="mesh_a"
    ).public_result()["result"]
    comparison = act(
        engine,
        "compare_candidates",
        left="candidate0",
        right="candidate1",
        roles=["mask_a", "mask_b"],
    ).public_result()["result"]
    # Housekeeping replaces the latest receipt: successful recall must now come
    # from complete typed context, not accidental reissuance of the latest result.
    act(engine, "find_nodes", query="threshold")
    pins = []
    for number in range(4):
        payload = {"note": number, "text": ""}
        payload["text"] = "x" * (8190 - len(_json(payload)))
        assert len(_json(payload)) == 8190
        descriptor = engine._notebook.append("review_note", payload)
        pins.append({key: descriptor[key] for key in ("record_id", "sha256")})
    assert act(engine, "pin_notebook", refs=pins).public_result()["status"] == "ok"
    before = engine.export_notebook()
    assert sum(len(_json(engine._notebook.lookup(**ref))) for ref in pins) == 32760
    monkeypatch.setattr(
        module, "execute_pipeline_request", lambda *a, **k: pytest.fail("extra trial")
    )
    monkeypatch.setattr(
        engine, "_render_views", lambda *a, **k: pytest.fail("extra render")
    )
    monkeypatch.setattr(
        engine, "_render_mesh_views", lambda *a, **k: pytest.fail("extra mesh render")
    )
    request = engine.public_request()
    _request(
        request, engine.preview_images()
    )  # Strict production wire; no provider call.
    assert len(_json(request)) <= module.WORKING_REQUEST_TARGET_BYTES <= 128 * 1024
    assert module.MAX_REQUEST_BYTES == 192 * 1024
    assert request["closing_phase"]["active"] is False
    assert engine.export_notebook() == before
    assert request["evidence_notebook"]["pinned_record_refs"] == pins
    assert "outputs" not in request["accepted_candidate"]
    full_ref = request["accepted_candidate"]["full_record"]
    assert engine._notebook.lookup(**full_ref) == engine.candidates[0].evidence
    context = request["decision_context"]
    typed = {row["alias"]: row for row in context["numeric_evidence"]}
    required = {
        "candidate1/preservation",
        "candidate1/measurements/mesh_a",
        "candidate1/measurements/mesh_b",
        "candidate1/measurements/combined_mesh",
        inspection["evidence_ref"]["alias"],
        comparison["evidence_ref"]["alias"],
    }
    assert required <= typed.keys()
    assert any(row["candidate_id"] == "candidate1" for row in context["recipes"])
    for alias in required:
        row = typed[alias]
        assert engine._numeric_payloads[
            (alias, row["evidence_ref"]["sha256"])
        ] == _json(row["payload"])
        with pytest.raises(InvestigationError, match="not transmitted"):
            engine._validate_evidence_refs([row["evidence_ref"]])
    altered = deepcopy(request)
    altered["decision_context"]["numeric_evidence"][0]["payload"]["forged"] = True
    with pytest.raises(InvestigationError, match="outgoing request"):
        engine.record_model_call(request=altered)
    assert engine.model_calls_used == 1
    engine.record_model_call(request=request)
    for alias in required:
        engine._validate_evidence_refs(
            [typed[alias]["evidence_ref"]], candidate_id="candidate1"
        )
    assert engine.attempted_trials == 1 and engine.retained_candidate_id == "candidate0"
    assert engine.export_notebook()["pinned_record_refs"] == pins
    snapshot.assert_current(pipeline)
