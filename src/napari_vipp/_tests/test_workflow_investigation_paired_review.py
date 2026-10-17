"""Independent paired-review guards using ordinary, tiny synthetic execution.

No recorded scientific recipe, human reference, provider, native viewer or
previously nominated algorithm is used here. Pairing is presentation evidence,
not an accuracy score or an automatic scientific candidate selector.
"""

from copy import deepcopy

import numpy as np
import pytest

from napari_vipp._tests.test_workflow_investigation import act, session
from napari_vipp._tests.test_workflow_investigation_context_priority_review import (
    completed_33_node_8_role_candidate,
)
from napari_vipp._tests.test_workflow_investigation_experiment_session import (
    assessment,
    plan,
)
from napari_vipp.core.workflow_investigation import (
    InvestigationError,
    InvestigationSession,
    InvestigationStopped,
    _json,
    capture_completed_candidate,
)
from napari_vipp.core.workflow_investigation_model import _request


@pytest.fixture
def completed_trial():
    pipeline, snapshot, engine, threshold = completed_33_node_8_role_candidate()
    first = engine.public_request()
    engine.record_model_call(request=first)
    reference = next(
        row["evidence_ref"]
        for row in first["decision_context"]["numeric_evidence"]
        if row["alias"] == "candidate0/measurements/mesh_a"
    )
    planned = plan(
        engine, baseline="candidate0", refs=[reference], visual=True
    ).public_result()
    assert planned["status"] == "ok", planned
    trial = act(
        engine,
        "trial_edit",
        candidate_id="candidate0",
        experiment_id=planned["result"]["experiment_id"],
        edits=[
            {
                "kind": "set_parameters",
                "node": threshold,
                "params": {"threshold": 90},
            }
        ],
    ).public_result()
    assert trial["status"] == "ok", trial
    assert trial["result"]["preservation_check"]["passed"]
    return pipeline, snapshot, engine


def inspect_pair(engine, *, views=None, roi=None):
    views = views or [
        {"axis": "Z", "indices": [0, 2, 4], "montage": True, "overlay_source": True},
        {"axis": "X", "indices": [3], "overlay_source": True},
    ]
    results = []
    for candidate in ("candidate0", "candidate1"):
        result = act(
            engine,
            "inspect_candidate",
            candidate_id=candidate,
            role="mask_a",
            views=views,
            **({"roi": roi} if roi is not None else {}),
        ).public_result()
        assert result["status"] == "ok", result
        results.append(result["result"])
    return results


def pair_rows(request):
    rows = request["comparison_previews"]
    assert len(rows) <= 4 and len(_json(rows)) <= 8 * 1024
    return [row for row in rows if row["role"] == "mask_a"]


def assert_exact_pairs(engine, request, *, require_supplied=True):
    board = {row["sha256"] for row in request["image_previews"]}
    rows = pair_rows(request)
    complete = [row for row in rows if len(row["views"]) == 2]
    assert complete
    if require_supplied:
        assert any(row["status"] == "supplied" for row in complete)
    for row in complete:
        hashes = {view["preview_sha256"] for view in row["views"]}
        assert len(hashes) == 2  # The real edit changes these synthetic views.
        assert len(hashes & board) in (0, 2), "An automatic pair was split."
        assert (row["status"] == "supplied") == (hashes <= board)
        assert row["candidate_ids"] == ["candidate0", "candidate1"]
        assert {view["candidate_id"] for view in row["views"]} == {
            "candidate0",
            "candidate1",
        }
        for view in row["views"]:
            key = engine._view_keys[view["view_id"]]
            original = engine._preview_cache.lookup(key)
            assert view["preview_sha256"] in {
                image.manifest()["sha256"] for image in original
            }
        assert row["selection"]["source_sha256"] == engine._source.fingerprint
        assert (
            row["selection"]["source_state_sha256"] == engine._source.state_fingerprint
        )
        assert row["selection"]["grid"]["shape"] == [5, 6, 7]
    assert len(request["image_previews"]) <= 7
    assert sum(row["byte_count"] for row in request["image_previews"]) <= 2 * 1024**2
    return complete


def test_pair_survives_later_mesh_inspection_pins_and_genuine_closing(
    completed_trial, monkeypatch
):
    import napari_vipp.core.workflow_investigation as module

    pipeline, snapshot, engine = completed_trial
    inspect_pair(engine)
    mesh = act(
        engine,
        "inspect_candidate",
        candidate_id="candidate1",
        role="combined_mesh",
        views=[{"view_index": 0}, {"view_index": 1}],
    ).public_result()
    assert mesh["status"] == "ok", mesh
    pins = []
    for number in range(4):
        payload = {"note": number, "text": ""}
        payload["text"] = "x" * (8190 - len(_json(payload)))
        descriptor = engine._notebook.append("review_note", payload)
        pins.append({key: descriptor[key] for key in ("record_id", "sha256")})
    assert act(engine, "pin_notebook", refs=pins).public_result()["status"] == "ok"
    before = engine.export_notebook()
    monkeypatch.setattr(
        module, "execute_pipeline_request", lambda *a, **k: pytest.fail("extra Run")
    )
    monkeypatch.setattr(engine, "_render_views", lambda *a, **k: pytest.fail("render"))
    monkeypatch.setattr(
        engine, "_render_mesh_views", lambda *a, **k: pytest.fail("mesh render")
    )
    # The ordinary clock budget, not a manufactured assessment, enters closing.
    monkeypatch.setattr(engine, "_clock", lambda: engine.deadline - 60)
    request = engine.public_request()
    assert request["closing_phase"]["active"]
    assert "trial_edit" not in request["closing_phase"]["allowed_actions"]
    assert "assess_experiment" in request["closing_phase"]["allowed_actions"]
    assert "finish" not in request["closing_phase"]["allowed_actions"]
    rows = assert_exact_pairs(engine, request)
    topology = next(
        row["payload"]
        for row in request["decision_context"]["numeric_evidence"]
        if row["alias"] == "candidate0:candidate1/mesh_topology_review"
    )
    combined = next(row for row in topology["meshes"] if row["role"] == "combined_mesh")
    assert combined["baseline"]["counts"]["surface_component_count"] == 2
    assert combined["candidate"]["counts"]["surface_component_count"] == 2
    assert "surface_components" in {warning["code"] for warning in combined["warnings"]}
    assert request["evidence_notebook"]["pinned_record_refs"] == pins
    assert any(
        image["label"].startswith("Protected prior foreground")
        for image in request["image_previews"]
    )
    _request(request, engine.preview_images())
    assert len(_json(request)) <= module.WORKING_REQUEST_TARGET_BYTES
    assert engine.export_notebook() == before
    supplied = next(row for row in rows if row["status"] == "supplied")
    visual_ref = {"kind": "preview", "sha256": supplied["views"][1]["preview_sha256"]}
    with pytest.raises(InvestigationError, match="not transmitted"):
        engine._validate_evidence_refs([visual_ref], candidate_id="candidate1")
    numeric_ref = next(
        row["evidence_ref"]
        for row in request["decision_context"]["numeric_evidence"]
        if row["alias"] == "candidate1/preservation"
    )
    engine.record_model_call(request=request)
    checked = assessment(
        engine,
        "experiment1",
        "candidate1",
        refs=[numeric_ref],
        visual_ref=visual_ref,
    ).public_result()
    assert checked["status"] == "ok", checked
    final_request = engine.public_request()
    assert final_request["closing_phase"]["pending_experiment_ids"] == []
    assert "finish" in final_request["closing_phase"]["allowed_actions"]
    assert_exact_pairs(engine, final_request)
    assert engine.retained_candidate_id == "candidate0"
    assert engine.attempted_trials == 1
    snapshot.assert_current(pipeline)


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_sha256", "1" * 64),
        ("source_state_sha256", "2" * 64),
        ("grid", {"axes": "ZYX", "shape": [5, 6, 8], "calibration": []}),
        ("roi", {"axis_order": "ZYX", "bounds": [[0, 2], [0, 6], [0, 7]]}),
        ("alias", "candidate1/mask_b"),
        ("views", [{"axis": "X", "indices": [2], "overlay_source": True}]),
        ("display_policy", {"renderer": "unknown", "range": [0, 1]}),
    ],
)
def test_selection_never_claims_mismatched_cached_metadata(
    completed_trial, monkeypatch, field, value
):
    _, _, engine = completed_trial
    inspect_pair(engine, views=[{"axis": "X", "indices": [3], "overlay_source": True}])
    identifier = engine._recent_view_ids[-1]
    key = deepcopy(engine._view_keys[identifier])
    original = engine._preview_cache.lookup(key)
    key[field] = value
    engine._view_keys[identifier] = key
    # Unit-test matching independently of the cache's stronger key rejection.
    # These are genuine rendered bytes; altered metadata is never sent as evidence.
    lookup = engine._preview_cache.lookup
    monkeypatch.setattr(
        engine._preview_cache,
        "lookup",
        lambda selected: original if selected is key else lookup(selected),
    )
    rows = engine._comparison_preview_context()
    assert not any(row["status"] == "supplied" for row in rows)
    assert not any(len(row["views"]) == 2 for row in rows)
    assert engine.retained_candidate_id == "candidate0"


def test_descriptor_cannot_supply_omitted_pixels_or_change_actual_request(
    completed_trial,
):
    _, _, engine = completed_trial
    inspect_pair(engine)
    request = engine.public_request()
    rows = assert_exact_pairs(engine, request)
    unsent = {"kind": "preview", "sha256": rows[0]["views"][0]["preview_sha256"]}
    assert engine._sent_previews.isdisjoint({unsent["sha256"]})
    engine.export_notebook()
    engine.public_request()
    with pytest.raises(InvestigationError, match="not transmitted"):
        engine._validate_evidence_refs([unsent])
    changed = deepcopy(request)
    changed["comparison_previews"][0]["reason"] = "A forged board receipt."
    with pytest.raises(InvestigationError, match="outgoing request"):
        engine.record_model_call(request=changed)
    assert engine.model_calls_used == 1
    engine.record_model_call(request=request)
    engine._validate_evidence_refs([unsent])


def test_cumulative_png_limit_rejects_repeated_cached_pair_without_authority(
    completed_trial,
):
    _, _, engine = completed_trial
    inspect_pair(engine)
    request = engine.public_request()
    rows = assert_exact_pairs(engine, request)
    assert engine.limits.max_image_bytes == 10 * 1024**2
    charged = sum(image["byte_count"] for image in request["image_previews"])
    engine._image_bytes = engine.limits.max_image_bytes - charged + 1
    request = engine.public_request()
    before = set(engine._sent_previews)
    with pytest.raises(InvestigationStopped, match="transmitted-image byte budget"):
        engine.record_model_call(request=request)
    assert engine._sent_previews == before
    assert engine.model_calls_used == 1
    assert any(row["status"] == "supplied" for row in rows)


def test_permission_change_revokes_pairs_and_actual_send(completed_trial):
    _, _, engine = completed_trial
    inspect_pair(engine)
    request = engine.public_request()
    assert_exact_pairs(engine, request)
    engine.include_output_images = False
    assert engine.preview_images() == ()
    with pytest.raises(InvestigationStopped, match="permissions changed"):
        engine.record_model_call(request=request)
    assert engine.model_calls_used == 1


@pytest.mark.parametrize("include_input_images", [False, True])
def test_independent_default_output_opt_out_has_no_pairs_or_output_authority(
    include_input_images,
):
    engine = session(
        include_input_images=include_input_images, include_output_images=False
    )
    request = engine.public_request()
    assert request["comparison_previews"] == []
    assert request["image_permissions"]["output_images"] is False
    assert all(
        image["label"].startswith("input") for image in request["image_previews"]
    )
    if not include_input_images:
        assert request["image_previews"] == [] and engine.preview_images() == ()
        denied = act(
            engine, "inspect_source", views=[{"axis": "Z", "indices": [0]}]
        ).public_result()
        assert denied["status"] == "error"
        assert engine.preview_images() == ()
    engine.record_model_call(request=engine.public_request())
    assert engine._sent_preview_owners == {} or all(
        owners == frozenset() for owners in engine._sent_preview_owners.values()
    )


def test_real_closed_cylinder_to_ring_keeps_euler_warning_not_automatic_success(
    monkeypatch,
):
    from napari_vipp.core.compute import ComputeRequest
    from napari_vipp.core.metadata import AxisMetadata, image_state_from_array
    from napari_vipp.core.pipeline import PrototypePipeline, SourcePayload
    from napari_vipp.core.workflow_assessment import capture_assessment_snapshot

    # An ordinary analytical synthetic cylinder and ring; neither is a desired
    # anatomy, reference solution, prior-run method or preferred output.
    z, y, x = np.mgrid[-4:5, -4:5, -4:5]
    radius = np.sqrt(x**2 + y**2)
    cylinder = (radius <= 3.5) & (np.abs(z) <= 2)
    ring = (radius - 2.5) ** 2 + z**2 <= 1.3**2
    data = np.where(cylinder, 1, 0).astype(np.float32)
    data[ring] = 2
    data.setflags(write=False)
    state = image_state_from_array(
        data, axes=tuple(AxisMetadata(axis, "space", "um") for axis in "zyx")
    )
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    mask = pipeline.add_node("binary_threshold")
    mask.params.update(threshold=0.5, foreground="Above")
    mesh = pipeline.add_node("mask_to_3d_mesh")
    mesh.params["boundary"] = "Close at image border"
    assert pipeline.connect("input", mask.id).success
    assert pipeline.connect(mask.id, mesh.id).success
    pipeline.run(
        data,
        source_payloads={
            "input": SourcePayload(
                data, image_state=state, axis_semantics_resolved=True
            )
        },
        manual_node_ids=[mask.id, mesh.id],
    )
    snapshot = capture_assessment_snapshot(
        pipeline, "input", (mask.id, mesh.id), "ordinary-synthetic-cylinder"
    )
    bundle = capture_completed_candidate(
        pipeline,
        snapshot,
        [
            {"role": "mask", "node": mask.id, "port": 0},
            {"role": "mesh", "node": mesh.id, "port": 0},
        ],
    )
    engine = InvestigationSession.from_completed(
        pipeline,
        bundle,
        "Compare existing exact synthetic topology facts; no preferred shape.",
        compute_request=ComputeRequest(),
        include_output_images=True,
        assert_current=lambda: None,
    )
    threshold = next(
        key for key, node in engine.candidates[0].key_to_node.items() if node == mask.id
    )
    result = act(
        engine,
        "trial_edit",
        candidate_id="candidate0",
        edits=[
            {"kind": "set_parameters", "node": threshold, "params": {"threshold": 1.5}}
        ],
    ).public_result()
    assert result["status"] == "ok", result
    monkeypatch.setattr(engine, "_summary", lambda *a, **k: pytest.fail("remeasure"))
    request = engine.public_request()
    wrapper = next(
        row
        for row in request["decision_context"]["numeric_evidence"]
        if row["alias"] == "candidate0:candidate1/mesh_topology_review"
    )
    payload = wrapper["payload"]
    assert payload["kind"] == "mesh-topology-review"
    row = next(row for row in payload["meshes"] if row["role"] == "mesh")
    before, after = row["baseline"]["counts"], row["candidate"]["counts"]
    assert before["boundary_edge_count"] == after["boundary_edge_count"] == 0
    assert before["euler_count_v_minus_e_plus_f"] == 2
    assert after["euler_count_v_minus_e_plus_f"] == 0
    assert "euler_count_v_minus_e_plus_f" in row["changed_fields"]
    assert "topology_changed" in {warning["code"] for warning in row["warnings"]}
    assert "genus" in payload["limitations"]
    assert "do not prove" in payload["limitations"]
    assert not {"winner", "accuracy", "success", "genus"} & payload.keys()
    for candidate, side in zip(
        engine.candidates, ("baseline", "candidate"), strict=True
    ):
        output = next(
            row for row in candidate.evidence["outputs"] if row["role"] == "mesh"
        )
        for field, value in row[side]["counts"].items():
            assert value == output["diagnostics"][field]
    with pytest.raises(InvestigationError, match="not transmitted"):
        engine._validate_evidence_refs([wrapper["evidence_ref"]])
    engine.record_model_call(request=request)
    engine._validate_evidence_refs([wrapper["evidence_ref"]], candidate_id="candidate1")
    assert engine.retained_candidate_id == "candidate0"
    snapshot.assert_current(pipeline)
