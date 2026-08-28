from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import napari_vipp.core.execution as execution_module
from napari_vipp.core import operations, presentation
from napari_vipp.core.batch import scientific_workflow_hash
from napari_vipp.core.compute import ComputeRequest, DecisionKind
from napari_vipp.core.execution import (
    PipelinePresentationShadowResult,
    PipelineRunRequest,
    _pipeline_timing_workload_fingerprint,
    execute_pipeline_request,
)
from napari_vipp.core.execution_provenance import (
    serialize_execution_provenance,
)
from napari_vipp.core.graph_fragments import (
    capture_graph_fragment,
    decode_graph_fragment,
    encode_graph_fragment,
)
from napari_vipp.core.pipeline import (
    NODE_LIBRARY,
    NODE_LIBRARY_BY_ID,
    PrototypePipeline,
    SourcePayload,
)
from napari_vipp.core.snapshots import GraphSnapshot
from napari_vipp.core.tables import TableData, TableState
from napari_vipp.core.workflow import (
    WORKFLOW_VERSION,
    deserialize_workflow,
    serialize_workflow,
)


def _crop_pipeline(*, bypassed: bool = True) -> tuple[PrototypePipeline, str]:
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    crop = pipeline.add_node("crop_stack")
    pipeline.set_param(crop.id, "left", 1)
    assert pipeline.connect("input", crop.id).success
    if bypassed:
        assert pipeline.set_node_execution_mode(crop.id, "bypass")
    return pipeline, crop.id


def test_bypass_capability_is_derived_from_operation_schema_and_live_graph() -> None:
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    gaussian = pipeline.add_node("gaussian_blur")
    split = pipeline.add_node("split_channels")
    writer = pipeline.add_node("save_output")

    assert NODE_LIBRARY_BY_ID["gaussian_blur"].supports_bypass
    assert not NODE_LIBRARY_BY_ID["input"].supports_bypass
    assert not NODE_LIBRARY_BY_ID["split_channels"].supports_bypass
    assert not NODE_LIBRARY_BY_ID["save_output"].supports_bypass
    assert not pipeline.node_supports_bypass(gaussian.id)
    assert "primary input" in pipeline.node_bypass_block_reason(gaussian.id)
    with pytest.raises(ValueError, match="primary input"):
        pipeline.set_node_execution_mode(gaussian.id, "bypass")
    with pytest.raises(ValueError, match="multi-output boundary"):
        pipeline.set_node_execution_mode(split.id, "bypass")
    with pytest.raises(ValueError, match="writer"):
        pipeline.set_node_execution_mode(writer.id, "bypass")


def test_every_single_output_operation_inherits_bypass_except_true_boundaries() -> None:
    assert len(NODE_LIBRARY) == 114
    assert {spec.id for spec in NODE_LIBRARY if not spec.supports_bypass} == {
        "input",
        "born_wolf_psf",
        "split_axis",
        "skeleton_graph_tables",
        "skeleton_keypoints",
        "split_channels",
        "save_output",
        "batch_output",
    }


def test_type_changing_node_rejects_bypass_when_consumer_needs_native_output() -> None:
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    threshold = pipeline.add_node("binary_threshold")
    fill = pipeline.add_node("fill_holes")
    assert pipeline.connect("input", threshold.id).success
    assert pipeline.connect(threshold.id, fill.id).success

    reason = pipeline.node_bypass_block_reason(threshold.id)
    assert "would forward image data" in reason
    assert "requires mask" in reason
    with pytest.raises(ValueError, match="would forward image data"):
        pipeline.set_node_execution_mode(threshold.id, "bypass")


def test_bypassed_output_exposes_forwarded_type_and_new_edges_stay_reversible() -> None:
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    threshold = pipeline.add_node("binary_threshold")
    fill = pipeline.add_node("fill_holes")
    gaussian = pipeline.add_node("gaussian_blur")
    assert pipeline.connect("input", threshold.id).success
    assert pipeline.set_node_execution_mode(threshold.id, "bypass")

    assert pipeline.output_ports(threshold.id)[0].output_type == "image"
    incompatible = pipeline.connect(threshold.id, fill.id)
    assert not incompatible.success
    assert "mask input" in incompatible.message
    assert pipeline.connect(threshold.id, gaussian.id).success
    assert pipeline.set_node_execution_mode(threshold.id, "run")
    assert pipeline.output_ports(threshold.id)[0].output_type == "mask"


def test_restore_fails_closed_for_an_incompatible_persisted_graph_splice() -> None:
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    threshold = pipeline.add_node("binary_threshold")
    fill = pipeline.add_node("fill_holes")
    assert pipeline.connect("input", threshold.id).success
    assert pipeline.connect(threshold.id, fill.id).success
    nodes = [
        replace(node, execution_mode="bypass")
        if node.id == threshold.id
        else node
        for node in pipeline.nodes.values()
    ]

    restored = PrototypePipeline()
    with pytest.raises(ValueError, match="normal mask.*mask input"):
        restored.restore_graph(nodes, pipeline.connections)


def test_disconnected_bypass_chain_restores_and_reconnects_atomically() -> None:
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    psf_source = pipeline.add_node("input")
    first = pipeline.add_node("binary_threshold")
    second = pipeline.add_node("gaussian_blur")
    restoration = pipeline.add_node("richardson_lucy_deconvolution")
    assert pipeline.connect("input", first.id).success
    assert pipeline.connect(first.id, second.id).success
    assert pipeline.connect(second.id, restoration.id, target_port=0).success
    assert pipeline.connect(psf_source.id, restoration.id, target_port=1).success
    assert pipeline.set_node_execution_mode(first.id, "bypass")
    assert pipeline.disconnect("input", first.id)
    assert pipeline.set_node_execution_mode(second.id, "bypass")

    assert pipeline.output_ports(first.id)[0].output_type == "any"
    assert pipeline.output_ports(second.id)[0].output_type == "any"
    snapshot = GraphSnapshot.from_pipeline(pipeline)
    restored = snapshot.to_pipeline()
    assert restored.node_is_bypassed(first.id)
    assert restored.node_is_bypassed(second.id)
    assert restored.set_node_execution_mode(first.id, "run")
    GraphSnapshot.from_pipeline(restored).to_pipeline()
    restored = snapshot.to_pipeline()

    mask_source = restored.add_node("binary_threshold")
    assert restored.connect("input", mask_source.id).success
    original_connections = tuple(restored.connections)
    rejected = restored.connect(mask_source.id, first.id)
    assert not rejected.success
    assert "would forward mask data" in rejected.message
    assert tuple(restored.connections) == original_connections

    trapped = restored.connect("input", first.id)
    assert not trapped.success
    assert "Clearing Bypass" in trapped.message
    assert tuple(restored.connections) == original_connections
    assert restored.set_node_execution_mode(second.id, "run")
    assert restored.connect("input", first.id).success
    assert restored.output_ports(first.id)[0].output_type == "image"
    assert restored.output_ports(second.id)[0].output_type == "image"


def test_upstream_mode_change_cannot_invalidate_a_downstream_bypass_chain() -> None:
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    first = pipeline.add_node("otsu_threshold")
    second = pipeline.add_node("binary_threshold")
    cleanup = pipeline.add_node("remove_binary_outliers")
    assert pipeline.connect("input", first.id).success
    assert pipeline.connect(first.id, second.id).success
    assert pipeline.connect(second.id, cleanup.id).success
    assert pipeline.set_node_execution_mode(second.id, "bypass")

    prospective_reason = pipeline.node_bypass_block_reason(first.id)
    assert "would invalidate the existing Bypass" in prospective_reason
    assert "would forward image data" in prospective_reason
    with pytest.raises(ValueError, match="would forward image data"):
        pipeline.set_node_execution_mode(first.id, "bypass")

    assert not pipeline.node_is_bypassed(first.id)
    assert pipeline.node_is_bypassed(second.id)
    assert pipeline.node_bypass_block_reason(second.id) == ""
    assert pipeline.output_ports(second.id)[0].output_type == "mask"


def test_clearing_bypass_cannot_break_a_type_preserving_downstream_chain() -> None:
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    psf_source = pipeline.add_node("input")
    threshold = pipeline.add_node("binary_threshold")
    crop = pipeline.add_node("crop_stack")
    reorder = pipeline.add_node("reorder_axes")
    restoration = pipeline.add_node("richardson_lucy_deconvolution")
    assert pipeline.connect("input", threshold.id).success
    assert pipeline.set_node_execution_mode(threshold.id, "bypass")
    assert pipeline.connect(threshold.id, crop.id).success
    assert pipeline.connect(crop.id, reorder.id).success
    assert pipeline.connect(psf_source.id, restoration.id, target_port=1).success
    original_connections = tuple(pipeline.connections)

    rejected = pipeline.connect(reorder.id, restoration.id, target_port=0)
    assert not rejected.success
    assert "Clearing Bypass" in rejected.message
    assert "mask data" in rejected.message
    assert tuple(pipeline.connections) == original_connections
    assert pipeline.set_node_execution_mode(threshold.id, "run")
    assert not pipeline.node_is_bypassed(threshold.id)
    GraphSnapshot.from_pipeline(pipeline).to_pipeline()


def test_tunnel_reroute_requires_bypass_source_to_remain_clearable() -> None:
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    gaussian = pipeline.add_node("gaussian_blur")
    threshold = pipeline.add_node("binary_threshold")
    restoration = pipeline.add_node("richardson_lucy_deconvolution")
    assert pipeline.connect("input", gaussian.id).success
    assert pipeline.connect("input", threshold.id).success
    original_tunnel = pipeline.add_output_tunnel("Reference", gaussian.id)
    assert pipeline.connect_to_tunnel("Reference", restoration.id, 0).success
    assert pipeline.set_node_execution_mode(threshold.id, "bypass")
    pipeline.add_output_tunnel("Bypassed mask", threshold.id)

    assert "Bypassed mask" not in {
        tunnel.name for tunnel in pipeline.compatible_output_tunnels(restoration.id, 0)
    }
    pipeline.remove_output_tunnel("Bypassed mask")
    original_connections = tuple(pipeline.connections)
    with pytest.raises(ValueError, match="clearing Bypass must remain valid"):
        pipeline.reroute_output_tunnel("Reference", threshold.id)

    assert pipeline.output_tunnel("Reference") == original_tunnel
    assert tuple(pipeline.connections) == original_connections
    assert pipeline.set_node_execution_mode(threshold.id, "run")


@pytest.mark.parametrize(
    "operation_id",
    ["richardson_lucy_deconvolution", "richardson_lucy_tv_deconvolution"],
)
def test_deconvolution_bypass_aliases_intensity_port_not_psf(
    operation_id: str,
) -> None:
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    psf_source = pipeline.add_node("input")
    restoration = pipeline.add_node(operation_id)
    downstream = pipeline.add_node("gaussian_blur")
    assert pipeline.connect("input", restoration.id, target_port=0).success
    assert pipeline.connect(psf_source.id, restoration.id, target_port=1).success
    assert pipeline.connect(restoration.id, downstream.id).success
    assert pipeline.node_supports_bypass(restoration.id)
    assert pipeline.set_node_execution_mode(restoration.id, "bypass")

    intensity = np.arange(42, dtype=np.float32).reshape(6, 7)
    psf = np.ones((3, 3), dtype=np.float32)
    result = execute_pipeline_request(
        PipelineRunRequest(
            run_id=8,
            workflow=serialize_workflow(pipeline),
            input_data=intensity,
            input_metadata={"axes": "YX"},
            input_name="intensity",
            source_payloads={
                psf_source.id: SourcePayload(
                    psf,
                    {"axes": "YX"},
                    "psf",
                )
            },
        )
    )

    assert result.error == ""
    completed = result.pipeline
    assert completed is not None
    assert completed.outputs[restoration.id] is intensity
    assert completed.outputs[restoration.id] is not psf
    assert (
        completed.output_states[restoration.id]
        is completed.output_states["input"]
    )
    assert completed.outputs[downstream.id].shape == intensity.shape


def test_deconvolution_bypass_does_not_require_a_psf_connection() -> None:
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    restoration = pipeline.add_node("richardson_lucy_tv_deconvolution")
    downstream = pipeline.add_node("gaussian_blur")
    assert pipeline.connect("input", restoration.id, target_port=0).success
    assert pipeline.connect(restoration.id, downstream.id).success
    assert pipeline.set_node_execution_mode(restoration.id, "bypass")

    intensity = np.arange(30, dtype=np.float32).reshape(5, 6)
    result = execute_pipeline_request(
        PipelineRunRequest(
            run_id=9,
            workflow=serialize_workflow(pipeline),
            input_data=intensity,
            input_metadata={"axes": "YX"},
            input_name="intensity-only",
            source_payloads={},
        )
    )

    assert result.error == ""
    assert result.pipeline is not None
    assert result.pipeline.outputs[restoration.id] is intensity


def test_bypass_timing_fingerprint_ignores_secondary_presentation_branch() -> None:
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    psf_source = pipeline.add_node("input")
    psf_blur = pipeline.add_node("gaussian_blur")
    restoration = pipeline.add_node("richardson_lucy_tv_deconvolution")
    assert pipeline.connect("input", restoration.id, target_port=0).success
    assert pipeline.connect(psf_source.id, psf_blur.id).success
    assert pipeline.connect(psf_blur.id, restoration.id, target_port=1).success
    assert pipeline.set_node_execution_mode(restoration.id, "bypass")

    def fingerprint(contexts: dict[str, str]) -> str:
        return _pipeline_timing_workload_fingerprint(
            pipeline,
            frozenset({restoration.id}),
            retain_node_ids=frozenset(),
            prune_unretained=False,
            manual_node_ids=None,
            target_node_ids=frozenset({restoration.id}),
            compute_request=ComputeRequest(),
            source_scientific_contexts=contexts,
            cancel_callback=None,
        )

    without_psf = fingerprint({"input": "intensity"})
    psf_a = fingerprint({"input": "intensity", psf_source.id: "psf-a"})
    psf_b = fingerprint({"input": "intensity", psf_source.id: "psf-b"})
    pipeline.set_param(psf_blur.id, "sigma", 7.0)
    changed_psf_parameters = fingerprint(
        {"input": "intensity", psf_source.id: "psf-c"}
    )

    assert without_psf
    assert without_psf == psf_a == psf_b == changed_psf_parameters


def test_targeted_bypass_does_not_fingerprint_secondary_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    psf_source = pipeline.add_node("input")
    restoration = pipeline.add_node("richardson_lucy_tv_deconvolution")
    downstream = pipeline.add_node("gaussian_blur")
    assert pipeline.connect("input", restoration.id, target_port=0).success
    assert pipeline.connect(psf_source.id, restoration.id, target_port=1).success
    assert pipeline.connect(restoration.id, downstream.id).success
    assert pipeline.set_node_execution_mode(restoration.id, "bypass")
    captured_sources: set[str] = set()
    original = execution_module._capture_source_scientific_contexts

    def capture(*args: object, **kwargs: object):
        contexts = original(*args, **kwargs)
        captured_sources.update(contexts)
        return contexts

    monkeypatch.setattr(
        execution_module,
        "_capture_source_scientific_contexts",
        capture,
    )
    data = np.arange(25, dtype=np.float32).reshape(5, 5)
    psf = np.ones((3, 3), dtype=np.float32) / 9
    result = execute_pipeline_request(
        PipelineRunRequest(
            run_id=11,
            workflow=serialize_workflow(pipeline),
            input_data=data,
            input_metadata={"axes": "YX"},
            input_name="intensity",
            source_payloads={
                psf_source.id: SourcePayload(psf, {"axes": "YX"}, "psf")
            },
            target_node_ids=frozenset({downstream.id}),
        )
    )

    assert result.error == ""
    assert captured_sources == {"input"}


def test_cpu_bypass_aliases_exact_value_and_state_without_calling_crop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline, crop_id = _crop_pipeline()
    original = NODE_LIBRARY_BY_ID["crop_stack"]

    def forbidden_crop(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("the bypassed Crop Stack operation was invoked")

    monkeypatch.setitem(
        NODE_LIBRARY_BY_ID,
        "crop_stack",
        replace(original, function=forbidden_crop),
    )
    data = np.arange(30, dtype=np.uint16).reshape(5, 6)

    result = execute_pipeline_request(
        PipelineRunRequest(
            run_id=1,
            workflow=serialize_workflow(pipeline),
            input_data=data,
            input_metadata={"axes": "YX"},
            input_name="bypass-test",
            source_payloads={},
        )
    )

    assert result.error == ""
    assert result.pipeline is not None
    assert result.pipeline.outputs[crop_id] is data
    assert (
        result.pipeline.output_states[crop_id]
        is result.pipeline.output_states["input"]
    )
    assert result.execution_report is not None
    decision = result.execution_report.actual_decisions[0]
    assert decision.node_id == crop_id
    assert decision.decision_kind is DecisionKind.BYPASSED
    assert decision.runtime_id == "vipp-bypass"

    # Durable provenance remains exact even when an intermediate host cache was
    # deliberately pruned and only the completed decision survives.
    result.pipeline.node_compute_provenance.clear()
    provenance = serialize_execution_provenance(
        result.execution_report.request,
        result.pipeline,
        result.execution_report,
    )
    record = next(
        item
        for item in provenance["nodes"]
        if item["node_id"] == crop_id
    )
    assert record["execution_mode"] == "bypass"
    assert record["bypassed"] is True
    assert record["decision_kind"] == "bypassed"
    assert record["actual_implementation"]["runtime_id"] == "vipp-bypass"
    assert record["actual_implementation"]["identity_complete"] is True


def test_opt_in_card_shadow_crops_pixels_without_entering_scientific_outputs() -> None:
    pipeline, crop_id = _crop_pipeline()
    pipeline.set_param(crop_id, "top", 1)
    downstream = pipeline.add_node("median_filter")
    pipeline.set_param(downstream.id, "size", 1)
    assert pipeline.connect(crop_id, downstream.id).success
    data = np.arange(6 * 8, dtype=np.uint16).reshape(6, 8)
    node_results = []
    shadows: list[PipelinePresentationShadowResult] = []

    result = execute_pipeline_request(
        PipelineRunRequest(
            run_id=4,
            workflow=serialize_workflow(pipeline),
            input_data=data,
            input_metadata={"axes": "YX"},
            input_name="bypass-shadow-test",
            source_payloads={},
            presentation_shadow_node_ids=frozenset({crop_id}),
        ),
        node_finished_callback=node_results.append,
        presentation_shadow_callback=shadows.append,
    )

    assert result.error == ""
    completed = result.pipeline
    assert completed is not None
    assert completed.outputs[crop_id] is data
    assert completed.output_states[crop_id] is completed.output_states["input"]
    assert completed.outputs[downstream.id].shape == data.shape
    assert len(shadows) == 1
    shadow = shadows[0]
    assert shadow.node_id == crop_id
    assert shadow.error == ""
    assert shadow.output.shape == (5, 7)
    assert shadow.output_state.shape == (5, 7)
    assert np.shares_memory(shadow.output, data)
    assert not shadow.output.flags.writeable
    assert not hasattr(node_results[1], "presentation_shadow_output")


def test_multi_input_card_shadow_runs_original_operation_with_secondary_input() -> None:
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    threshold = pipeline.add_node("binary_threshold")
    masked = pipeline.add_node("mask_image")
    downstream = pipeline.add_node("gaussian_blur")
    pipeline.set_param(threshold.id, "threshold", 4.0)
    assert pipeline.connect("input", threshold.id).success
    assert pipeline.connect("input", masked.id, target_port=0).success
    assert pipeline.connect(threshold.id, masked.id, target_port=1).success
    assert pipeline.connect(masked.id, downstream.id).success
    assert pipeline.set_node_execution_mode(masked.id, "bypass")
    data = np.arange(9, dtype=np.uint16).reshape(3, 3)
    shadows: list[PipelinePresentationShadowResult] = []
    authored_params = dict(pipeline.nodes[masked.id].params)

    result = execute_pipeline_request(
        PipelineRunRequest(
            run_id=6,
            workflow=serialize_workflow(pipeline),
            input_data=data,
            input_metadata={"axes": "YX"},
            input_name="multi-input-shadow",
            source_payloads={},
            presentation_shadow_node_ids=frozenset({masked.id}),
        ),
        presentation_shadow_callback=shadows.append,
    )

    assert result.error == ""
    assert result.pipeline is not None
    assert result.pipeline.outputs[masked.id] is data
    assert len(shadows) == 1
    assert shadows[0].error == ""
    expected = data.copy()
    expected[data <= 4] = 0
    np.testing.assert_array_equal(shadows[0].output, expected)
    assert shadows[0].output is not result.pipeline.outputs[masked.id]
    assert result.pipeline.nodes[masked.id].params == authored_params


def test_missing_secondary_source_makes_only_rl_shadow_unavailable() -> None:
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    psf_source = pipeline.add_node("input")
    restoration = pipeline.add_node("richardson_lucy_tv_deconvolution")
    downstream = pipeline.add_node("gaussian_blur")
    assert pipeline.connect("input", restoration.id, target_port=0).success
    assert pipeline.connect(psf_source.id, restoration.id, target_port=1).success
    assert pipeline.connect(restoration.id, downstream.id).success
    assert pipeline.set_node_execution_mode(restoration.id, "bypass")
    data = np.arange(25, dtype=np.float32).reshape(5, 5)
    shadows: list[PipelinePresentationShadowResult] = []
    authored_params = dict(pipeline.nodes[restoration.id].params)

    result = execute_pipeline_request(
        PipelineRunRequest(
            run_id=10,
            workflow=serialize_workflow(pipeline),
            input_data=data,
            input_metadata={"axes": "YX"},
            input_name="intensity-only",
            source_payloads={},
            target_node_ids=frozenset({downstream.id}),
            presentation_shadow_node_ids=frozenset({restoration.id}),
        ),
        presentation_shadow_callback=shadows.append,
    )

    assert result.error == ""
    assert result.pipeline is not None
    assert result.pipeline.outputs[restoration.id] is data
    assert len(shadows) == 1
    assert "source payloads are missing" in shadows[0].error
    assert psf_source.id in shadows[0].error
    assert result.pipeline.nodes[restoration.id].params == authored_params


def test_secondary_edit_dirties_only_its_branch_and_bypass_shadow() -> None:
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    threshold = pipeline.add_node("binary_threshold")
    masked = pipeline.add_node("mask_image")
    downstream = pipeline.add_node("gaussian_blur")
    assert pipeline.connect("input", threshold.id).success
    assert pipeline.connect("input", masked.id, target_port=0).success
    assert pipeline.connect(threshold.id, masked.id, target_port=1).success
    assert pipeline.connect(masked.id, downstream.id).success
    assert pipeline.set_node_execution_mode(masked.id, "bypass")
    pipeline.run(np.arange(9, dtype=np.uint16).reshape(3, 3))

    assert pipeline.descendants_inclusive({threshold.id}) == {threshold.id}
    assert pipeline.presentation_shadow_nodes_affected_by({threshold.id}) == {
        masked.id
    }
    plan = pipeline.plan_execution({threshold.id})
    assert plan.runnable_node_ids == frozenset({threshold.id})


def test_targeted_pruned_shadow_uses_detached_secondary_ancestry() -> None:
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    threshold = pipeline.add_node("binary_threshold")
    masked = pipeline.add_node("mask_image")
    downstream = pipeline.add_node("gaussian_blur")
    pipeline.set_param(threshold.id, "threshold", 4.0)
    assert pipeline.connect("input", threshold.id).success
    assert pipeline.connect("input", masked.id, target_port=0).success
    assert pipeline.connect(threshold.id, masked.id, target_port=1).success
    assert pipeline.connect(masked.id, downstream.id).success
    assert pipeline.set_node_execution_mode(masked.id, "bypass")
    data = np.arange(9, dtype=np.uint16).reshape(3, 3)
    shadows: list[PipelinePresentationShadowResult] = []

    result = execute_pipeline_request(
        PipelineRunRequest(
            run_id=10,
            workflow=serialize_workflow(pipeline),
            input_data=data,
            input_metadata={"axes": "YX"},
            input_name="targeted-shadow",
            source_payloads={},
            target_node_ids=frozenset({downstream.id}),
            retain_node_ids=frozenset({masked.id, downstream.id}),
            prune_unretained=True,
            presentation_shadow_node_ids=frozenset({masked.id}),
        ),
        presentation_shadow_callback=shadows.append,
    )

    assert result.error == ""
    assert result.pipeline is not None
    assert result.pipeline.outputs[masked.id] is data
    assert result.pipeline.outputs[threshold.id] is None
    assert threshold.id not in result.pipeline.node_compute_provenance
    assert len(shadows) == 1
    assert shadows[0].error == ""
    expected = data.copy()
    expected[data <= 4] = 0
    np.testing.assert_array_equal(shadows[0].output, expected)


def test_atomic_profile_shadows_are_isolated_from_downstream_bypasses() -> None:
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    psf_source = pipeline.add_node("input")
    threshold = pipeline.add_node("binary_threshold")
    overlay = pipeline.add_node("skeleton_graph_overlay")
    restoration = pipeline.add_node("richardson_lucy_tv_deconvolution")
    pipeline.set_param(threshold.id, "threshold", 4.0)
    assert pipeline.connect("input", threshold.id).success
    assert pipeline.connect(threshold.id, overlay.id).success
    assert pipeline.connect(overlay.id, restoration.id, target_port=0).success
    assert pipeline.connect(psf_source.id, restoration.id, target_port=1).success
    pipeline.apply_atomic_node_execution_profile(
        {threshold.id: "bypass", overlay.id: "bypass"}
    )
    data = np.arange(9, dtype=np.uint16).reshape(3, 3)
    shadows: list[PipelinePresentationShadowResult] = []

    result = execute_pipeline_request(
        PipelineRunRequest(
            run_id=15,
            workflow=serialize_workflow(pipeline),
            input_data=data,
            input_metadata={"axes": "YX"},
            input_name="atomic-shadow",
            source_payloads={},
            target_node_ids=frozenset({overlay.id}),
            presentation_shadow_node_ids=frozenset({threshold.id, overlay.id}),
            atomic_bypass_profile=True,
        ),
        presentation_shadow_callback=shadows.append,
    )

    assert result.error == ""
    assert result.pipeline is not None
    assert result.pipeline.outputs[threshold.id] is data
    assert result.pipeline.outputs[overlay.id] is data
    by_node = {shadow.node_id: shadow for shadow in shadows}
    assert by_node[threshold.id].error == ""
    np.testing.assert_array_equal(by_node[threshold.id].output, data > 4)
    assert "requires mask" in by_node[overlay.id].error


def test_card_shadow_failure_is_nonfatal_and_never_replaces_exact_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline, crop_id = _crop_pipeline()
    data = np.arange(30, dtype=np.uint16).reshape(5, 6)
    shadows: list[PipelinePresentationShadowResult] = []

    def fail_preview(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("preview-only failure")

    monkeypatch.setattr(
        execution_module,
        "crop_stack_presentation_view",
        fail_preview,
    )
    result = execute_pipeline_request(
        PipelineRunRequest(
            run_id=5,
            workflow=serialize_workflow(pipeline),
            input_data=data,
            input_metadata={"axes": "YX"},
            input_name="bypass-shadow-failure-test",
            source_payloads={},
            presentation_shadow_node_ids=frozenset({crop_id}),
        ),
        node_finished_callback=lambda _result: None,
        presentation_shadow_callback=shadows.append,
    )

    assert result.error == ""
    assert not result.cancelled
    assert result.pipeline is not None
    assert result.pipeline.outputs[crop_id] is data
    assert result.pipeline.output_states[crop_id] is (
        result.pipeline.output_states["input"]
    )
    assert len(shadows) == 1
    assert shadows[0].output is None
    assert shadows[0].error == "preview-only failure"


def test_crop_shadow_view_does_not_weaken_scientific_copy_contract() -> None:
    data = np.arange(20, dtype=np.uint16).reshape(4, 5)

    preview = presentation.crop_stack_presentation_view(data)
    scientific = operations.crop_stack(data)

    assert preview is not data
    assert np.shares_memory(preview, data)
    assert not preview.flags.writeable
    assert scientific is not data
    assert scientific.flags.c_contiguous
    assert scientific.flags.writeable
    np.testing.assert_array_equal(scientific, data)


def test_dirty_sibling_rebinds_valid_bypass_cache_copy_to_exact_upstream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline, crop_id = _crop_pipeline()
    sibling = pipeline.add_node("median_filter")
    assert pipeline.connect("input", sibling.id).success
    data = np.arange(42, dtype=np.uint16).reshape(6, 7)

    initial_result = execute_pipeline_request(
        PipelineRunRequest(
            run_id=2,
            workflow=serialize_workflow(pipeline),
            input_data=data,
            input_metadata={"axes": "YX"},
            input_name="bypass-cache-test",
            source_payloads={},
        )
    )
    assert initial_result.error == ""
    initial = initial_result.pipeline
    assert initial is not None
    assert initial.outputs[crop_id] is initial.outputs["input"]

    copied_crop = np.array(initial.outputs[crop_id], copy=True)
    copied_crop.setflags(write=False)
    assert copied_crop is not initial.outputs["input"]
    cached_outputs = dict(initial.outputs)
    cached_outputs[crop_id] = copied_crop
    cached_node_outputs = {
        node_id: list(outputs)
        for node_id, outputs in initial.node_outputs.items()
    }
    cached_node_outputs[crop_id] = [copied_crop]

    crop_calls = 0
    original = NODE_LIBRARY_BY_ID["crop_stack"]

    def forbidden_crop(*_args: object, **_kwargs: object) -> object:
        nonlocal crop_calls
        crop_calls += 1
        raise AssertionError("the bypassed Crop Stack operation was invoked")

    monkeypatch.setitem(
        NODE_LIBRARY_BY_ID,
        "crop_stack",
        replace(original, function=forbidden_crop),
    )
    started: list[str] = []
    result = execute_pipeline_request(
        PipelineRunRequest(
            run_id=3,
            workflow=serialize_workflow(initial),
            input_data=data,
            input_metadata={"axes": "YX"},
            input_name="bypass-cache-test",
            source_payloads={},
            dirty_node_ids=frozenset({sibling.id}),
            cached_outputs=cached_outputs,
            cached_output_states=dict(initial.output_states),
            cached_node_outputs=cached_node_outputs,
            cached_node_output_states={
                node_id: list(states)
                for node_id, states in initial.node_output_states.items()
            },
            completed_node_ids=frozenset(initial.completed_node_ids),
            cached_execution_states=dict(initial.node_execution_states),
            cached_execution_messages=dict(initial.node_execution_messages),
            cached_compute_provenance={
                **initial.node_cache_lineage,
                **initial.node_compute_provenance,
            },
        ),
        node_started_callback=started.append,
    )

    assert result.error == ""
    hydrated = result.pipeline
    assert hydrated is not None
    assert started == [sibling.id]
    assert crop_calls == 0
    assert hydrated.outputs[crop_id] is hydrated.outputs["input"]
    assert hydrated.node_outputs[crop_id][0] is hydrated.node_outputs["input"][0]
    assert hydrated.output_states[crop_id] is hydrated.output_states["input"]
    assert (
        hydrated.node_output_states[crop_id][0]
        is hydrated.node_output_states["input"][0]
    )
    assert hydrated.outputs[crop_id] is not copied_crop
    assert crop_id in hydrated.completed_node_ids
    assert (
        hydrated.node_compute_provenance[crop_id]
        .actual_implementation.runtime_id
        == "vipp-bypass"
    )


def test_labels_to_table_measurement_can_alias_labels_through_a_tunnel() -> None:
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    threshold = pipeline.add_node("binary_threshold")
    labels = pipeline.add_node("label_connected_components")
    measurement = pipeline.add_node("measure_objects")
    pipeline.set_param(threshold.id, "threshold", 0.5)
    assert pipeline.connect("input", threshold.id).success
    assert pipeline.connect(threshold.id, labels.id).success
    assert pipeline.connect(labels.id, measurement.id).success
    pipeline.add_output_tunnel("Labels alias", measurement.id)
    assert pipeline.set_node_execution_mode(measurement.id, "bypass")
    data = np.zeros((8, 8), dtype=np.float32)
    data[1:3, 1:3] = 1
    data[5:7, 5:7] = 1

    result = execute_pipeline_request(
        PipelineRunRequest(
            run_id=12,
            workflow=serialize_workflow(pipeline),
            input_data=data,
            input_metadata={"axes": "YX"},
            input_name="labels-alias",
            source_payloads={},
            target_node_ids=frozenset({measurement.id}),
        )
    )

    assert result.error == ""
    completed = result.pipeline
    assert completed is not None
    assert completed.output_ports(measurement.id)[0].output_type == "labels"
    assert completed.outputs[measurement.id] is completed.outputs[labels.id]
    assert (
        completed.output_states[measurement.id]
        is completed.output_states[labels.id]
    )
    assert not isinstance(completed.outputs[measurement.id], TableData)


def test_table_transform_bypass_preserves_exact_table_cache_and_batch_output() -> None:
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    threshold = pipeline.add_node("binary_threshold")
    labels = pipeline.add_node("label_connected_components")
    measurement = pipeline.add_node("measure_objects")
    transform = pipeline.add_node("add_metadata_columns")
    output = pipeline.add_node("batch_output")
    pipeline.set_param(threshold.id, "threshold", 0.5)
    assert pipeline.connect("input", threshold.id).success
    assert pipeline.connect(threshold.id, labels.id).success
    assert pipeline.connect(labels.id, measurement.id).success
    assert pipeline.connect(measurement.id, transform.id).success
    assert pipeline.connect(transform.id, output.id).success
    pipeline.add_output_tunnel("Table alias", transform.id)
    assert pipeline.set_node_execution_mode(transform.id, "bypass")
    data = np.zeros((8, 8), dtype=np.float32)
    data[1:3, 1:3] = 1
    data[5:7, 5:7] = 1
    request_kwargs = {
        "workflow": serialize_workflow(pipeline),
        "input_data": data,
        "input_metadata": {"axes": "YX"},
        "input_name": "table-alias",
        "source_payloads": {},
        "manual_node_ids": frozenset(pipeline.manual_node_ids()),
    }

    initial_result = execute_pipeline_request(
        PipelineRunRequest(run_id=13, **request_kwargs)
    )
    assert initial_result.error == ""
    initial = initial_result.pipeline
    assert initial is not None
    table = initial.outputs[measurement.id]
    table_state = initial.output_states[measurement.id]
    assert isinstance(table, TableData)
    assert isinstance(table_state, TableState)
    assert initial.outputs[transform.id] is table
    assert initial.output_states[transform.id] is table_state
    assert initial.outputs[output.id] is table
    assert (
        initial.node_compute_provenance[transform.id]
        .actual_implementation.runtime_id
        == "vipp-bypass"
    )

    copied_table = replace(table)
    copied_state = replace(table_state)
    cached_outputs = dict(initial.outputs)
    cached_outputs[transform.id] = copied_table
    cached_states = dict(initial.output_states)
    cached_states[transform.id] = copied_state
    cached_node_outputs = {
        node_id: list(values)
        for node_id, values in initial.node_outputs.items()
    }
    cached_node_outputs[transform.id] = [copied_table]
    cached_node_states = {
        node_id: list(states)
        for node_id, states in initial.node_output_states.items()
    }
    cached_node_states[transform.id] = [copied_state]

    hydrated_result = execute_pipeline_request(
        PipelineRunRequest(
            run_id=14,
            **request_kwargs,
            dirty_node_ids=frozenset({output.id}),
            target_node_ids=frozenset({output.id}),
            cached_outputs=cached_outputs,
            cached_output_states=cached_states,
            cached_node_outputs=cached_node_outputs,
            cached_node_output_states=cached_node_states,
            completed_node_ids=frozenset(initial.completed_node_ids),
            cached_execution_states=dict(initial.node_execution_states),
            cached_execution_messages=dict(initial.node_execution_messages),
            cached_compute_provenance={
                **initial.node_cache_lineage,
                **initial.node_compute_provenance,
            },
        )
    )

    assert hydrated_result.error == ""
    hydrated = hydrated_result.pipeline
    assert hydrated is not None
    assert hydrated.outputs[transform.id] is hydrated.outputs[measurement.id]
    assert (
        hydrated.output_states[transform.id]
        is hydrated.output_states[measurement.id]
    )
    assert hydrated.outputs[transform.id] is not copied_table
    assert hydrated.output_states[transform.id] is not copied_state
    assert hydrated.outputs[output.id] is hydrated.outputs[transform.id]
    assert (
        hydrated.node_compute_provenance[transform.id]
        .actual_implementation.runtime_id
        == "vipp-bypass"
    )


def test_workflow_snapshot_and_fragment_preserve_bypass_intent() -> None:
    pipeline, crop_id = _crop_pipeline()

    document = serialize_workflow(pipeline)
    assert document["version"] == WORKFLOW_VERSION
    crop_document = next(
        node for node in document["nodes"] if node["id"] == crop_id
    )
    assert crop_document["execution_mode"] == "bypass"
    restored = deserialize_workflow(document)
    restored_crop = next(node for node in restored["nodes"] if node.id == crop_id)
    assert restored_crop.execution_mode == "bypass"

    snapshotted = GraphSnapshot.from_pipeline(pipeline).to_pipeline()
    assert snapshotted.nodes[crop_id].execution_mode == "bypass"

    fragment = capture_graph_fragment(
        pipeline,
        {crop_id},
        positions={crop_id: (25.0, 30.0)},
    )
    decoded = decode_graph_fragment(encode_graph_fragment(fragment))
    assert decoded.nodes[0].execution_mode == "bypass"


def test_legacy_workflow_cannot_silently_gain_bypass_intent() -> None:
    pipeline, _crop_id = _crop_pipeline()
    document = serialize_workflow(pipeline)
    document["version"] = 5

    with pytest.raises(ValueError, match="unknown field.*execution_mode"):
        deserialize_workflow(document)


def test_run_only_hash_stays_legacy_compatible_but_bypass_changes_it() -> None:
    pipeline, crop_id = _crop_pipeline(bypassed=False)
    current_run = serialize_workflow(pipeline)
    legacy_run = {**current_run, "version": 5}

    assert scientific_workflow_hash(current_run) == scientific_workflow_hash(
        legacy_run
    )

    pipeline.set_node_execution_mode(crop_id, "bypass")
    bypassed = serialize_workflow(pipeline)
    assert scientific_workflow_hash(bypassed) != scientific_workflow_hash(
        current_run
    )


def test_switching_execution_mode_invalidates_only_the_affected_branch() -> None:
    pipeline, crop_id = _crop_pipeline(bypassed=False)
    sibling = pipeline.add_node("median_filter")
    downstream = pipeline.add_node("gaussian_blur")
    assert pipeline.connect("input", sibling.id).success
    assert pipeline.connect(crop_id, downstream.id).success
    pipeline.completed_node_ids.update(
        {"input", crop_id, sibling.id, downstream.id}
    )

    assert pipeline.set_node_execution_mode(crop_id, "bypass")

    assert "input" in pipeline.completed_node_ids
    assert sibling.id in pipeline.completed_node_ids
    assert crop_id not in pipeline.completed_node_ids
    assert downstream.id not in pipeline.completed_node_ids
