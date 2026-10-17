"""Provider-free contracts: no models, network, filesystem source or GUI."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from napari_vipp.core.compute import ComputeMode, ComputeRequest
from napari_vipp.core.metadata import AxisMetadata, image_state_from_array
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.workflow_investigation import (
    ACTION_KIND,
    InvestigationAction,
    InvestigationError,
    InvestigationLimits,
    InvestigationSession,
    InvestigationStopped,
    parse_investigation_action,
)


def source_pipeline(shape=(4, 8, 10)):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape).copy()
    data.flags.writeable = False
    axes = tuple(
        AxisMetadata(a.lower(), "space", "um", scale=2 if a == "Z" else 0.5)
        for a in ("ZYX" if len(shape) == 3 else "YX")
    )
    state = image_state_from_array(data, axes=axes, source_name="PRIVATE source")
    pipeline.outputs["input"] = data
    pipeline.output_states["input"] = state
    return pipeline


def session(pipeline=None, **kwargs):
    return InvestigationSession(
        pipeline or source_pipeline(),
        "input",
        "Separate the requested structures; retain uncertainty.",
        compute_request=kwargs.pop("compute_request", ComputeRequest()),
        assert_current=kwargs.pop("assert_current", lambda: None),
        include_output_images=kwargs.pop("include_output_images", True),
        **kwargs,
    )


def act(engine, action, **arguments):
    return engine.perform(InvestigationAction(engine.session_id, action, arguments))


def proposal(engine, threshold=100, *, manual=False):
    from napari_vipp.core.pipeline import NODE_LIBRARY_BY_ID

    nodes = [
        {
            "key": "mask",
            "operation_id": "binary_threshold",
            "params": {
                p.name: p.default
                for p in NODE_LIBRARY_BY_ID["binary_threshold"].parameters
            },
            "reason": "Test an explicit threshold.",
        }
    ]
    nodes[0]["params"].update(threshold=threshold)
    edges = [{"source": "input", "source_port": 0, "target": "mask", "target_port": 0}]
    if manual:
        for key, operation in (
            ("labels", "label_connected_components"),
            ("measure", "measure_objects"),
        ):
            nodes.append(
                {
                    "key": key,
                    "operation_id": operation,
                    "params": {
                        p.name: p.default
                        for p in NODE_LIBRARY_BY_ID[operation].parameters
                    },
                    "reason": "Measure the calculated labels.",
                }
            )
            nodes[-1]["params"]["spatial_mode"] = "3D ZYX"
        edges.extend(
            [
                {
                    "source": "mask",
                    "source_port": 0,
                    "target": "labels",
                    "target_port": 0,
                },
                {
                    "source": "labels",
                    "source_port": 0,
                    "target": "measure",
                    "target_port": 0,
                },
            ]
        )
    return {
        "kind": "napari-vipp-workflow-proposal",
        "version": 1,
        "context_id": engine.proposal_context_id,
        "summary": "A bounded synthetic test.",
        "assumptions": ["Native threshold is a starting choice."],
        "questions": [],
        "nodes": nodes,
        "connections": edges,
    }


def trial(engine, threshold=100, **kwargs):
    return act(
        engine,
        "trial_candidate",
        proposal=proposal(engine, threshold, **kwargs),
        output_roles=[{"role": "wanted", "node": "mask", "port": 0}],
    )


def test_strict_action_wire_and_defensive_arguments():
    raw = {
        "kind": ACTION_KIND,
        "version": 1,
        "session_id": "abc",
        "action": "find_nodes",
        "arguments": {"query": "threshold"},
    }
    action = parse_investigation_action(json.dumps(raw), expected_session_id="abc")
    raw["arguments"]["query"] = "modified"
    action.arguments["query"] = "modified again"
    assert action.arguments == {"query": "threshold"}
    for bad in (
        raw | {"extra": 1},
        raw | {"version": True},
        raw | {"action": "shell"},
        raw | {"arguments": {"query": "threshold", "path": "x"}},
    ):
        with pytest.raises(InvestigationError):
            parse_investigation_action(bad)
    with pytest.raises(InvestigationError):
        parse_investigation_action(raw, expected_session_id="another")
    with pytest.raises(InvestigationError):
        parse_investigation_action('{"kind":"x","kind":"y"}')


@pytest.mark.parametrize(
    "limits",
    [
        {"max_attempts": 6},
        {"max_seconds": 1801},
        {"max_actions": True},
        {"max_model_calls": 0},
        {"max_image_bytes": 0},
    ],
)
def test_limits_are_finite_and_non_boolean(limits):
    with pytest.raises(InvestigationError):
        InvestigationLimits(**limits)


def test_seed_has_only_goal_source_and_neutral_action_contract():
    engine = session()
    seed = engine.public_request()
    assert seed["version"] == 2
    assert "history" not in seed
    notebook = seed["evidence_notebook"]
    assert notebook["session_id"] == engine.session_id
    assert notebook["pinned_record_refs"] == []
    assert {row["kind"] for row in notebook["record_index"]} == {
        "source_evidence",
        "node_index",
    }
    for row in notebook["record_index"]:
        assert engine._notebook.lookup(row["record_id"], row["sha256"])
    assert seed["image_previews"] == []
    assert seed["source"]["axes"] == "ZYX"
    assert "PRIVATE" not in json.dumps(seed)
    node_record = next(
        row for row in notebook["record_index"] if row["kind"] == "node_index"
    )
    index = engine._notebook.lookup(node_record["record_id"], node_record["sha256"])
    assert 1 <= len(index) <= 256
    assert all(
        set(row) == {"operation_id", "title", "category", "trial_allowed"}
        and row["trial_allowed"]
        for row in index
    )
    assert "gaussian_blur" in {row["operation_id"] for row in index}
    assert "save_image" not in {row["operation_id"] for row in index}
    found = act(engine, "find_nodes", query="threshold").public_result()
    assert any(x["operation_id"] == "binary_threshold" for x in found["result"])
    details = act(
        engine, "describe_node", operation_id="binary_threshold"
    ).public_result()
    assert details["result"]["parameters"]
    assert details["result"]["trial_allowed"]


def test_actual_trial_is_detached_complete_frozen_and_not_latest_wins():
    pipeline = source_pipeline()
    before = dict(pipeline.nodes)
    engine = session(pipeline)
    first = trial(engine).public_result()
    assert first["status"] == "ok", first
    assert first["result"]["status"] == "completed", first
    assert set(pipeline.nodes) == set(before)
    record = engine.candidates[0]
    assert record.graph is not None
    assert record.evidence["outputs"][0]["diagnostics"]["foreground_count"] == 219
    record.recipe["nodes"].clear()
    assert record.recipe["nodes"]
    with pytest.raises(FrozenInstanceError):
        record.status = "failed"
    act(
        engine,
        "retain_candidate",
        candidate_id="candidate1",
        reason="Observed foreground is retained for comparison.",
    )
    second = trial(engine, 200).public_result()
    assert second["result"]["status"] == "completed", second
    assert engine.retained_candidate_id == "candidate1"
    assert engine.selected_candidate is record
    compared = act(
        engine,
        "compare_candidates",
        left="candidate1",
        right="candidate2",
        roles=["wanted"],
    ).public_result()
    assert compared["result"]["roles"][0]["left_only_foreground"] == 100
    finished = act(
        engine,
        "finish",
        candidate_id="candidate1",
        summary="This is the nominated tested candidate, not accuracy proof.",
        limitations=["No independent truth mask."],
    ).public_result()
    assert finished["result"]["evidence"] == record.evidence
    assert engine.final_add_proposal().can_apply
    assert set(pipeline.nodes) == set(before)
    with pytest.raises(InvestigationStopped):
        engine.public_request()


def test_source_stats_histogram_percentiles_are_exact_without_images():
    engine = session()
    result = act(
        engine, "inspect_source", percentiles=[0, 50, 100], histogram=True
    ).public_result()
    info = result["result"]["diagnostics"]
    assert sum(info["histogram"]["counts"]) == 320
    assert info["percentiles"][1]["value"] == 159.5
    assert info["sampling"] == "exact full array"
    assert engine.preview_images() == ()


def _compare_frozen_arrays(left, right):
    from napari_vipp.core.workflow_assessment import _capture_binding

    engine = session(source_pipeline(shape=left.shape))
    for data in (left, right):
        completed = trial(engine).public_result()
        assert completed["result"]["status"] == "completed"
        # Isolate comparison from operation-specific dtype conversion policies,
        # while retaining real completed candidates and immutable typed evidence.
        data.setflags(write=False)
        state = image_state_from_array(data, axes=engine._source.state.axes)
        binding = replace(
            _capture_binding("comparison", 0, data, state, [0]), live=None
        )
        engine._candidates[-1] = replace(
            engine.candidates[-1], _outputs=(("wanted", binding),)
        )
    return act(
        engine,
        "compare_candidates",
        left="candidate1",
        right="candidate2",
        roles=["wanted"],
    ).public_result()


@pytest.mark.parametrize(
    "left,right",
    [
        (
            np.array([[2**63 + 1]], dtype=np.uint64),
            np.array([[2**63 - 1]], dtype=np.int64),
        ),
        (
            np.array([[2**53 + 1]], dtype=np.uint64),
            np.array([[float(2**53)]], dtype=np.float64),
        ),
        (
            np.array([[2**53 + 1]], dtype=np.int64),
            np.array([[float(2**53)]], dtype=np.float64),
        ),
    ],
    ids=["uint64-int64", "uint64-float64", "int64-float64"],
)
def test_comparison_rejects_mixed_dtypes_before_lossy_numeric_promotion(left, right):
    result = _compare_frozen_arrays(left, right)
    assert result["status"] == "error"
    assert "identical output dtypes" in result["error"]
    assert "equal_value_count" not in json.dumps(result)
    assert not left.flags.writeable and not right.flags.writeable


def test_same_dtype_comparison_keeps_adjacent_wide_integer_values_exact():
    left = np.array([[2**63, 2**63 + 1]], dtype=np.uint64)
    right = np.array([[2**63, 2**63 + 2]], dtype=np.uint64)
    result = _compare_frozen_arrays(left, right)
    assert result["status"] == "ok"
    comparison = result["result"]["roles"][0]
    assert comparison["equal_value_count"] == 1
    assert comparison["foreground_intersection"] == 2
    assert comparison["foreground_union"] == 2
    assert "Exact numerical/foreground agreement" in comparison["meaning"]
    assert not left.flags.writeable and not right.flags.writeable


@pytest.mark.parametrize(
    "input_images,output_images",
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_image_permissions_independent_and_overlays_need_both(
    input_images, output_images
):
    engine = session(
        include_input_images=input_images, include_output_images=output_images
    )
    view = [{"axis": "Z", "indices": [0, 2], "crop": [1, 7, 2, 9], "montage": True}]
    source = act(engine, "inspect_source", views=view)
    assert bool(source.image_previews) == input_images
    result = trial(engine).public_result()
    if not output_images:
        assert result["status"] == "error"
        assert engine.candidates == ()
        assert "foreground_count" not in json.dumps(engine.public_request())
        return
    assert result["result"]["status"] == "completed"
    output = act(
        engine,
        "inspect_candidate",
        candidate_id="candidate1",
        role="wanted",
        views=view,
    )
    assert bool(output.image_previews) == output_images
    approved_board = engine.preview_images()
    overlay = act(
        engine,
        "inspect_candidate",
        candidate_id="candidate1",
        role="wanted",
        views=[view[0] | {"overlay_source": True}],
    )
    if not input_images:
        assert overlay.public_result()["status"] == "error"
        assert overlay.image_previews == approved_board
        assert engine.preview_images() == approved_board
    else:
        assert overlay.public_result()["status"] == "ok"
        assert overlay.image_previews
        image = Image.open(BytesIO(overlay.image_previews[0].png_bytes))
        assert image.size == (14, 6)
        assert "crop [1, 7, 2, 9]" in overlay.image_previews[0].description
        assert engine.public_request()["image_previews"] == [
            p.manifest() for p in engine.preview_images()
        ]


def test_preview_bounds_never_clamp_and_image_budget_shared():
    engine = session(include_input_images=True)
    for view in (
        {"axis": "Z", "indices": [4]},
        {"axis": "Z", "indices": [0], "crop": [-1, 7, 0, 9]},
        {"axis": "T", "indices": [0]},
    ):
        approved_board = engine.preview_images()
        result = act(engine, "inspect_source", views=[view])
        assert result.public_result()["status"] == "error"
        assert result.image_previews == approved_board
        assert engine.preview_images() == approved_board
    too_many = [
        {"axis": "Z", "indices": [0, 1, 2, 3]},
        {"axis": "Z", "indices": [0, 1, 2, 3]},
    ]
    assert (
        act(engine, "inspect_source", views=too_many).public_result()["status"]
        == "error"
    )
    with pytest.raises(InvestigationError, match="budget"):
        session(
            include_input_images=True, limits=InvestigationLimits(max_image_bytes=1)
        )


def test_invalid_candidates_count_and_cannot_finish_failed_candidate():
    engine = session(limits=InvestigationLimits(max_attempts=1))
    bad = proposal(engine)
    bad["nodes"][0]["operation_id"] = "save_output"
    assert (
        act(engine, "trial_candidate", proposal=bad, output_roles=[]).public_result()[
            "status"
        ]
        == "error"
    )
    assert len(engine.candidates) == 1
    assert engine.candidates[0].status == "failed"
    assert (
        act(
            engine,
            "finish",
            candidate_id="candidate1",
            summary="not real",
            limitations=[],
        ).public_result()["status"]
        == "error"
    )
    assert trial(engine).public_result()["status"] == "error"
    assert len(engine.candidates) == 1


def test_action_model_time_cancel_and_current_authority_guards():
    engine = session(limits=InvestigationLimits(max_model_calls=1))
    engine.record_model_call(request=engine.public_request())
    with pytest.raises(InvestigationStopped, match="model-call"):
        engine.record_model_call(request=engine.public_request())
    engine = session(limits=InvestigationLimits(max_actions=1))
    assert engine.public_request()["closing_phase"]["active"]
    with pytest.raises(InvestigationError, match="reserved"):
        act(engine, "find_nodes", query="threshold")
    assert engine.actions_used == 0
    ticks = [0]
    engine = session(clock=lambda: ticks[0], limits=InvestigationLimits(max_seconds=1))
    ticks[0] = 2
    with pytest.raises(InvestigationStopped, match="elapsed"):
        engine.public_request()
    engine = session()
    engine.cancel()
    with pytest.raises(InvestigationStopped, match="cancelled"):
        trial(engine)
    pipeline = source_pipeline()
    engine = session(pipeline)
    pipeline.outputs["input"] = np.zeros((4, 8, 10))
    with pytest.raises(InvestigationStopped, match="revision"):
        engine.public_request()


def test_question_stops_without_source_or_candidate_changes():
    engine = session()
    question = act(
        engine, "question", question="Which structure is the intended target?"
    ).public_result()
    assert question["result"]["question"].endswith("?")
    assert engine.candidates == ()
    with pytest.raises(InvestigationStopped):
        engine.public_request()


def test_manual_sink_and_compute_request_use_normal_execution(monkeypatch):
    import napari_vipp.core.workflow_investigation as module

    engine = session(compute_request=ComputeRequest(mode=ComputeMode.AUTO))
    original = module.execute_pipeline_request
    requests = []

    def observe(request):
        requests.append(request)
        # Exercise ordinary manual execution without hardware-discovery timing.
        return original(replace(request, compute_request=ComputeRequest()))

    monkeypatch.setattr(module, "execute_pipeline_request", observe)
    trial(engine, manual=True)
    assert requests
    request = requests[0]
    assert request.compute_request.mode is ComputeMode.AUTO
    assert any("measure" in n for n in request.manual_node_ids)
    assert request.manual_node_ids <= request.target_node_ids


def test_cleanup_failed_result_never_supplies_candidate_evidence(monkeypatch):
    import napari_vipp.core.workflow_investigation as module

    engine = session()
    original = module.execute_pipeline_request

    def unsafe(request):
        result = original(request)
        return replace(
            result,
            execution_report=replace(result.execution_report, cleanup_succeeded=False),
        )

    monkeypatch.setattr(module, "execute_pipeline_request", unsafe)
    with pytest.raises(InvestigationStopped, match="cleanup"):
        trial(engine)
    assert engine.candidates[0]._outputs == ()
    with pytest.raises(InvestigationStopped):
        engine.record_model_call(request=engine.public_request())


def test_real_manual_mesh_has_actual_output_diagnostics_and_requested_views():
    from napari_vipp.core.pipeline import NODE_LIBRARY_BY_ID

    pipeline = source_pipeline((8, 12, 14))
    data = np.zeros((8, 12, 14), dtype=np.float32)
    data[2:6, 3:9, 4:10] = 1
    data.flags.writeable = False
    pipeline.outputs["input"] = data
    engine = session(pipeline, include_output_images=True)
    raw = proposal(engine, 0.5)
    params = {
        p.name: p.default for p in NODE_LIBRARY_BY_ID["mask_to_3d_mesh"].parameters
    }
    raw["nodes"].append(
        {
            "key": "surface",
            "operation_id": "mask_to_3d_mesh",
            "params": params,
            "reason": "Inspect actual geometry of this calculated mask.",
        }
    )
    raw["connections"].append(
        {"source": "mask", "source_port": 0, "target": "surface", "target_port": 0}
    )
    result = act(
        engine,
        "trial_candidate",
        proposal=raw,
        output_roles=[
            {"role": "mask", "node": "mask", "port": 0},
            {"role": "surface", "node": "surface", "port": 0},
        ],
    ).public_result()
    assert result["status"] == "ok", result
    assert result["result"]["status"] == "completed", result
    diagnostics = result["result"]["outputs"][1]["diagnostics"]
    assert diagnostics["triangle_count"] > 0
    assert diagnostics["display_object_group_count"] == 1
    preview = act(
        engine,
        "inspect_candidate",
        candidate_id="candidate1",
        role="surface",
        views=[{"view_index": 0}, {"view_index": 1}],
    )
    assert len(preview.image_previews) == 2, preview.public_result()
    assert not engine.include_input_images
    assert "calibrat" in preview.image_previews[0].description
    assert "PRIVATE" not in json.dumps(result)


def test_initial_source_images_and_every_repair_transmission_charge_budget():
    engine = session(include_input_images=True)
    images = engine.preview_images()
    assert images
    assert engine.public_request()["image_previews"] == [p.manifest() for p in images]
    size = sum(len(p.png_bytes) for p in images)
    engine.limits = replace(engine.limits, max_image_bytes=size * 2)
    engine.record_model_call(request=engine.public_request())
    assert engine.public_request()["budgets"]["image_bytes_remaining"] == size
    engine.record_model_call(request=engine.public_request())
    with pytest.raises(InvestigationStopped, match="transmitted-image"):
        engine.record_model_call(request=engine.public_request())


def test_paths_and_invalid_clocks_are_rejected_before_actions():
    with pytest.raises(InvestigationError, match="paths"):
        InvestigationAction(
            "test", "question", {"question": "Read C:/private/image.tif"}
        )
    with pytest.raises(InvestigationError, match="finite"):
        session(clock=lambda: float("nan"))


def test_validation_error_is_actionable_without_source_paths_or_full_recipe():
    engine = session()
    raw = proposal(engine)
    del raw["nodes"][0]["params"]["threshold"]
    result = act(
        engine, "trial_candidate", proposal=raw, output_roles=[]
    ).public_result()
    assert result["status"] == "error"
    assert "parameter" in result["error"].lower()
    assert "PRIVATE" not in json.dumps(result)
    assert "recipe_sha256" in result["request"]
    assert "connections" not in result["request"]


def test_output_evidence_off_refuses_execution_and_diagnostics(monkeypatch):
    import napari_vipp.core.workflow_investigation as module

    engine = session(include_output_images=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("No calculation is authorized.")

    monkeypatch.setattr(module, "execute_pipeline_request", forbidden)
    assert trial(engine).public_result()["status"] == "error"
    for name, arguments in (
        ("inspect_candidate", {"candidate_id": "candidate1", "role": "wanted"}),
        (
            "compare_candidates",
            {"left": "candidate1", "right": "candidate2", "roles": ["wanted"]},
        ),
        ("retain_candidate", {"candidate_id": "candidate1", "reason": "test"}),
        (
            "finish",
            {"candidate_id": "candidate1", "summary": "test", "limitations": []},
        ),
    ):
        assert act(engine, name, **arguments).public_result()["status"] == "error"
    assert engine.candidates == ()
    assert "foreground_count" not in json.dumps(engine.public_request())


def test_observation_requires_previously_transmitted_exact_evidence():
    engine = session(include_input_images=True)
    preview = engine.preview_images()[0].manifest()["sha256"]
    arguments = {
        "evidence_refs": [{"kind": "preview", "sha256": preview}],
        "interpretation": "A bounded model interpretation of the supplied sections.",
        "hypothesis": "Test an explicit existing-node candidate.",
        "next_check": "Inspect the actual calculated output.",
    }
    assert (
        act(engine, "record_observation", **arguments).public_result()["status"]
        == "error"
    )
    # A failed action keeps the approved board, but does not pretend it was sent.
    assert engine.preview_images()[0].manifest()["sha256"] == preview
    engine.record_model_call(request=engine.public_request())
    result = act(engine, "record_observation", **arguments).public_result()
    assert result["status"] == "ok"
    assert "model_interpretation" in result["result"]
    sha = engine.public_request()["source"]["sha256"]
    arguments["evidence_refs"] = [{"kind": "numeric", "sha256": sha, "alias": "input"}]
    assert (
        act(engine, "record_observation", **arguments).public_result()["status"] == "ok"
    )
    arguments["evidence_refs"][0]["sha256"] = "0" * 64
    assert (
        act(engine, "record_observation", **arguments).public_result()["status"]
        == "error"
    )


@pytest.mark.parametrize(
    "values",
    [
        np.array([[2**63, 2**63 + 1]], dtype=np.uint64),
        np.array([[0.0, np.nextafter(0.0, 1.0)]]),
        np.array([[-np.finfo(float).max, np.finfo(float).max]]),
    ],
)
def test_display_scaling_preserves_extreme_finite_ranges_without_warnings(values):
    from napari_vipp.core.workflow_investigation import _grey

    with np.errstate(all="raise"):
        pixels = _grey(values, values.min().item(), values.max().item())
    np.testing.assert_array_equal(pixels, [[0, 255]])


def test_permission_changes_revoke_old_evidence_history():
    engine = session()
    assert trial(engine).public_result()["result"]["status"] == "completed"
    engine.include_output_images = False
    with pytest.raises(InvestigationStopped, match="permissions changed"):
        engine.public_request()


def test_describe_candidate_recovers_only_own_exact_recipe_and_roles():
    engine = session()
    raw = proposal(engine, 123)
    roles = [{"role": "wanted", "node": "mask", "port": 0}]
    act(engine, "trial_candidate", proposal=raw, output_roles=roles)
    result = act(
        engine, "describe_candidate", candidate_id="candidate1"
    ).public_result()
    assert result["status"] == "ok"
    scientific = result["result"]["scientific_recipe"]
    assert scientific["nodes"][0]["params"]["threshold"] == 123
    assert scientific["connections"] == raw["connections"]
    assert result["result"]["output_roles"] == roles
    assert "reason" not in scientific["nodes"][0]
    assert (
        act(
            engine, "describe_candidate", candidate_id="other_session_candidate"
        ).public_result()["status"]
        == "error"
    )


@pytest.mark.parametrize(
    "failure_kind", ["memory_preflight", "cancelled", "execution_error"]
)
def test_unclassified_runtime_failures_stop_and_keep_failed_receipt(
    monkeypatch, failure_kind
):
    import napari_vipp.core.workflow_investigation as module
    from napari_vipp.core.execution import PipelineExecutionFailure

    engine = session()
    original = module.execute_pipeline_request

    def failing(request):
        result = original(request)
        failure = PipelineExecutionFailure(
            kind=failure_kind,
            error_type="UnknownRuntimeError",
            message="C:/private/runtime detail",
            cleanup_succeeded=True,
        )
        return replace(
            result,
            error=failure.message,
            failure=failure,
            cancelled=failure_kind == "cancelled",
        )

    monkeypatch.setattr(module, "execute_pipeline_request", failing)
    with pytest.raises(InvestigationStopped):
        trial(engine)
    assert engine.candidates[0].status == "failed"
    assert engine.candidates[0]._outputs == ()
    assert "private" not in json.dumps(engine.candidates[0].evidence)
    with pytest.raises(InvestigationStopped):
        engine.record_model_call(request=engine.public_request())
