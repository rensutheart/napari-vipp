"""Planning must carry genuine metadata, never touch or fake numerical results."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from napari_vipp.core.metadata import (
    AxisMetadata,
    SourceMetadata,
    image_state_from_array,
)
from napari_vipp.core.node_execution import PreparedNodeCall
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.registration_planning import (
    REGISTRATION_PLANNING_OPERATIONS,
    TransformPlan,
    project_registration_outputs,
)
from napari_vipp.core.transforms import is_transform_data, save_transform_output


class UnreadablePixels:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = np.dtype(dtype)

    def __array__(self, *args, **kwargs):
        raise AssertionError("Metadata planning must not inspect any pixels")


def image(shape, names, *, frame="moving", dtype="float32", kind=None, spacing=1):
    axes = tuple(
        AxisMetadata(
            name,
            "time" if name == "t" else "channel" if name == "c" else "space",
            "s" if name == "t" else None if name == "c" else "um",
            spacing if name in "zyx" else 1,
        )
        for name in names
    )
    proxy = PrototypePipeline._axis_contract_proxy(shape, np.dtype(dtype))
    state = image_state_from_array(
        proxy,
        axes=axes,
        source_name=frame,
        source=SourceMetadata(source_uuid=frame),
        defer_statistics=True,
    )
    if kind is not None:
        state = replace(state, kind=kind)
    return UnreadablePixels(shape, dtype), state


def call(operation, pairs, **params):
    return PreparedNodeCall(
        "test",
        operation,
        lambda *args, **kwargs: None,
        tuple(pair[0] for pair in pairs),
        tuple(pair[1] for pair in pairs),
        kwargs=params,
        multiple_inputs=True,
        output_port_count=2,
    )


def project(operation, pairs, **params):
    return project_registration_outputs(
        PrototypePipeline(), call(operation, pairs, **params)
    )


def test_pair_plan_is_typed_immutable_and_cannot_be_exported(tmp_path):
    moving = image((12, 16), "yx")
    reference = image((18, 20), "yx", frame="reference", spacing=2)
    transform, diagnostics = project(
        "estimate_registration", [moving, reference], model="Rigid"
    )
    plan, state = transform
    assert isinstance(plan, TransformPlan) and not is_transform_data(plan)
    assert not hasattr(plan, "matrices")
    assert plan.reference_grid.shape == (18, 20)
    assert state.transform_count == 1 and not plan.is_time_series
    assert diagnostics == (None, None)
    with pytest.raises(FrozenInstanceError):
        plan.reference_time = 1
    with pytest.raises(TypeError):
        save_transform_output(plan, tmp_path / "not-a-result.json")
    aligned, coverage = project("apply_transform", [moving, transform])
    assert aligned[0].shape == (18, 20) and aligned[0].dtype == np.float64
    assert aligned[1].axes[0].scale == 2
    assert aligned[1].source.source_uuid == "reference"
    assert coverage[0].dtype == bool and coverage[1].kind == "binary mask"


def test_time_plan_reuses_xyz_transform_for_all_channels_and_label_ids():
    moving = image((5, 2, 8, 12, 16), "tczyx")
    transform, _ = project(
        "estimate_registration",
        [moving],
        mode="Time series",
        channel=1,
        reference_time=3,
    )
    plan = transform[0]
    assert plan.transform_count == 5 and plan.reference_time == 3
    labels = image((5, 8, 12, 16), "tzyx", dtype="uint64", kind="label image")
    aligned, coverage = project("apply_transform", [labels, transform])
    assert aligned[0].shape == labels[0].shape
    assert aligned[0].dtype == np.uint64
    assert tuple(axis.name for axis in aligned[1].axes) == tuple(
        axis.name for axis in labels[1].axes
    )
    assert coverage[0].shape == labels[0].shape


@pytest.mark.parametrize(
    "params,match",
    [
        ({"channel": 2}, "selected registration channel"),
        ({"reference_channel": 1}, "single-channel"),
        ({"precision": 0}, "Subpixel precision"),
        ({"max_shift": float("nan")}, "Maximum displacement"),
        ({"metric": "unknown"}, "Correlation or Mutual"),
    ],
)
def test_pair_plan_rejects_invalid_parameters_without_pixels(params, match):
    moving = image((2, 12, 16), "cyx")
    reference = image((12, 16), "yx", frame="reference")
    with pytest.raises(ValueError, match=match):
        project("estimate_registration", [moving, reference], **params)


@pytest.mark.parametrize(
    "shape,names,params,match",
    [
        ((12, 16), "yx", {}, "explicit T"),
        ((1, 12, 16), "tyx", {}, "at least two"),
        ((5, 12, 16), "tyx", {"reference_time": 5}, "valid reference time"),
        ((5, 3, 16), "tyx", {}, "at least four"),
    ],
)
def test_time_plan_validates_metadata(shape, names, params, match):
    with pytest.raises(ValueError, match=match):
        project(
            "estimate_registration", [image(shape, names)], mode="Time series", **params
        )


def test_pair_rejects_time_and_translation_mismatched_grid():
    with pytest.raises(ValueError, match="does not consume T"):
        project(
            "estimate_registration", [image((3, 12, 16), "tyx"), image((12, 16), "yx")]
        )
    with pytest.raises(ValueError, match="equal spatial shape"):
        project("estimate_registration", [image((12, 16), "yx"), image((16, 16), "yx")])


def test_apply_plan_rejects_wrong_frame_grid_time_and_linear_labels():
    moving = image((3, 12, 16), "tyx")
    transform, _ = project("estimate_registration", [moving], mode="Time series")
    cases = [
        (
            image((3, 12, 16), "tyx", frame="different"),
            {},
            "different coordinate frame",
        ),
        (image((3, 12, 16), "tyx", spacing=2), {}, "grid differs"),
        (image((2, 12, 16), "tyx"), {}, "same time-point count"),
        (
            image((3, 12, 16), "tyx", kind="label image"),
            {"interpolation": "Linear"},
            "preserve IDs",
        ),
        (
            image((3, 12, 16), "tyx", dtype="uint8"),
            {"interpolation": "Nearest", "outside_value": -1},
            "representable",
        ),
    ]
    for value, params, match in cases:
        with pytest.raises(ValueError, match=match):
            project("apply_transform", [value, transform], **params)
    wrong_time = replace(
        moving[1], axes=(replace(moving[1].axes[0], scale=2), *moving[1].axes[1:])
    )
    with pytest.raises(ValueError, match="Time calibration"):
        project("apply_transform", [(moving[0], wrong_time), transform])


def test_plan_rejects_inferred_axes_and_bad_descriptors():
    moving = image((12, 16), "yx")
    inferred = image_state_from_array(np.zeros((12, 16)), source_name="moving")
    with pytest.raises(ValueError, match="shape/dtype metadata disagree"):
        project("estimate_registration", [(moving[0], inferred), moving])
    inferred = replace(inferred, dtype="float32")
    with pytest.raises(ValueError, match="explicit axis"):
        project("estimate_registration", [(moving[0], inferred), moving])


def test_unknown_upstream_contract_stops_without_creating_identity():
    outputs = project("estimate_registration", [(None, None), (None, None)])
    assert outputs == ((None, None), (None, None))
    assert project_registration_outputs(PrototypePipeline(), None) is None


def test_workload_planning_carries_transform_without_inventing_array_facts():
    from napari_vipp.core.compute import OutputPortKey
    from napari_vipp.core.compute_policy import ArrayFacts, FactCompleteness
    from napari_vipp.core.compute_registry import ComputeRegistry
    from napari_vipp.core.execution import _assemble_workloads, _shape_and_dtype

    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    estimate = pipeline.add_node("estimate_registration")
    pipeline.set_param(estimate.id, "mode", "Time series")
    apply = pipeline.add_node("apply_transform")
    invert = pipeline.add_node("invert")
    diagnostics = pipeline.add_node("select_table_columns")
    assert pipeline.connect("input", estimate.id).success
    assert pipeline.connect("input", apply.id, target_port=0).success
    assert pipeline.connect(estimate.id, apply.id, target_port=1).success
    assert pipeline.connect(apply.id, invert.id).success
    assert pipeline.connect(estimate.id, diagnostics.id, source_port=1).success
    moving = image((3, 12, 16), "tyx")
    port = OutputPortKey("input", 0)
    facts = ArrayFacts(
        moving[0].shape,
        "float32",
        3 * 12 * 16,
        "test-source-revision",
        completeness=FactCompleteness.COMPLETE,
        finite_count=3 * 12 * 16,
        minimum=0.0,
        maximum=1.0,
    )
    with ComputeRegistry() as registry:
        workloads, facts_by_node, _lineage = _assemble_workloads(
            pipeline,
            frozenset((estimate.id, apply.id, invert.id, diagnostics.id)),
            {port: moving[0], OutputPortKey(estimate.id, 1): object()},
            {port: moving[1]},
            registry,
            False,
            seed_facts_by_port={port: facts},
        )
    by_node = {item.node_id: item for item in workloads}
    assert estimate.id in facts_by_node  # Exercises the formerly unsafe branch.
    assert by_node[apply.id].inputs_resolved
    assert by_node[apply.id].input_shapes == ((3, 12, 16), ())
    assert by_node[apply.id].input_dtypes == ("float32", "object")
    assert by_node[invert.id].inputs_resolved
    assert by_node[invert.id].input_shapes == ((3, 12, 16),)
    assert by_node[invert.id].input_dtypes == ("float64",)
    assert not by_node[diagnostics.id].inputs_resolved
    planned, _ = project("estimate_registration", [moving], mode="Time series")
    assert _shape_and_dtype(*planned) == ((), "object")


def test_compute_projection_preserves_typed_transform_and_reference_lattice():
    from napari_vipp.core.execution import _project_host_planning_outputs

    pipeline = PrototypePipeline()
    moving = image((12, 16), "yx")
    reference = image((18, 20), "yx", frame="reference", spacing=2)
    estimate_call = call("estimate_registration", [moving, reference], model="Rigid")
    projected = _project_host_planning_outputs(
        pipeline,
        "estimate_registration",
        estimate_call,
        ((12, 16), (18, 20)),
        ("float32", "float32"),
    )
    assert isinstance(projected[0][0], TransformPlan)
    apply_call = call("apply_transform", [moving, projected[0]])
    aligned, coverage = _project_host_planning_outputs(
        pipeline, "apply_transform", apply_call, ((12, 16), ()), ("float32", "object")
    )
    assert aligned[0].shape == (18, 20)
    assert aligned[0].dtype == np.float64
    assert aligned[1].axes[0].scale == 2
    assert coverage[0].dtype == bool


@pytest.mark.parametrize("operation", sorted(REGISTRATION_PLANNING_OPERATIONS))
def test_declared_registration_contracts_project_every_port_without_pixels(operation):
    from napari_vipp.core.execution import _project_host_planning_outputs

    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    node = pipeline.add_node(operation)
    moving = image((3, 2, 8, 12, 16), "tczyx")
    transform, _ = project("estimate_registration", [moving], mode="Time series")
    pairs = [moving] if operation == "estimate_registration" else [moving, transform]
    params = {"mode": "Time series"} if operation == "estimate_registration" else {}
    prepared = replace(call(operation, pairs, **params), node_id=node.id)
    scientific = pipeline._axis_contract_transform_results(prepared)
    compute = _project_host_planning_outputs(pipeline, operation, prepared, (), ())
    assert len(scientific) == len(compute) == prepared.output_port_count
    for (scientific_value, scientific_state), (compute_value, compute_state) in zip(
        scientific, compute, strict=True
    ):
        assert scientific_state == compute_state
        if operation == "estimate_registration":
            assert scientific_value == compute_value
        else:
            assert scientific_value.shape == compute_value.shape == moving[0].shape
            assert scientific_value.dtype == compute_value.dtype


def test_prefer_gpu_finalization_keeps_transform_but_releases_image_inputs(monkeypatch):
    from napari_vipp.core.compute import ComputeRequest
    from napari_vipp.core.execution import PipelineRunRequest, execute_pipeline_request
    from napari_vipp.core.pipeline import SourcePayload
    from napari_vipp.core.registration_samples import translation_pair
    from napari_vipp.core.transforms import TransformData
    from napari_vipp.core.workflow import serialize_workflow

    phantom = translation_pair(noisy=False)
    moving_meta = image(phantom.moving.shape, "yx", frame="moving")[1]
    reference_meta = image(phantom.reference.shape, "yx", frame="reference")[1]
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    reference = pipeline.add_node("input")
    estimate = pipeline.add_node("estimate_registration")
    apply = pipeline.add_node("apply_transform")
    assert pipeline.connect("input", estimate.id, target_port=0).success
    assert pipeline.connect(reference.id, estimate.id, target_port=1).success
    assert pipeline.connect("input", apply.id, target_port=0).success
    assert pipeline.connect(estimate.id, apply.id, target_port=1).success
    payloads = {
        "input": SourcePayload(phantom.moving, None, "moving", image_state=moving_meta),
        reference.id: SourcePayload(
            phantom.reference, None, "reference", image_state=reference_meta
        ),
    }
    captured_transforms = []
    original_finalize = PrototypePipeline.finalize_node_call

    def checked_finalize(self, prepared, output):
        if prepared.operation_id == "apply_transform" and prepared.inputs[0] is None:
            # Post-transaction metadata finalization must not retain opaque
            # image/device inputs, but does need this tuple-only host artifact.
            assert isinstance(prepared.inputs[1], TransformData)
            captured_transforms.append(prepared.inputs[1])
        return original_finalize(self, prepared, output)

    monkeypatch.setattr(PrototypePipeline, "finalize_node_call", checked_finalize)
    results = []
    for mode in ("cpu", "prefer_gpu"):
        request = PipelineRunRequest(
            run_id=len(results) + 1,
            workflow=serialize_workflow(pipeline),
            input_data=None,
            input_metadata=None,
            input_name="",
            source_payloads=payloads,
            compute_request=ComputeRequest(mode=mode),
            manual_node_ids=frozenset(pipeline.manual_node_ids()),
        )
        result = execute_pipeline_request(request, raise_errors=True)
        assert not result.error and result.execution_report.cleanup_succeeded
        results.append(result.pipeline)
    assert len(captured_transforms) == 1
    cpu, prefer_gpu = results
    for left, right in zip(
        cpu.node_outputs[apply.id], prefer_gpu.node_outputs[apply.id], strict=True
    ):
        np.testing.assert_array_equal(left, right)
    assert cpu.node_output_states[apply.id] == prefer_gpu.node_output_states[apply.id]
    assert cpu.outputs[estimate.id] == prefer_gpu.outputs[estimate.id]
