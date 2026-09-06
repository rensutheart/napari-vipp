from __future__ import annotations

import importlib.util
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import napari_vipp.core.execution as execution_module
from napari_vipp._tests.test_execution_array_facts import (
    _accelerated_request as _fact_planning_request,
)
from napari_vipp._tests.test_execution_array_facts import (
    _CapturingPlanner,
    _execute_accelerated,
)
from napari_vipp._tests.test_gpu_execution_integration import (
    _accelerated_request,
    _assert_private_cuda_scope_clean,
)
from napari_vipp.core.compute import (
    ComputeMode,
    ComputeRequest,
    DecisionKind,
    FallbackPolicy,
    OutputPortKey,
    WorkloadDescriptor,
)
from napari_vipp.core.compute_planning import plan_compute_decisions
from napari_vipp.core.compute_policy import ArrayFacts, FactCompleteness
from napari_vipp.core.compute_registry import ComputeRegistry
from napari_vipp.core.compute_specs import compute_specs_for
from napari_vipp.core.execution import execute_pipeline_request
from napari_vipp.core.metadata import AxisMetadata, image_state_from_array
from napari_vipp.core.operations import (
    label_connected_components as cpu_label_connected_components,
)
from napari_vipp.core.operations import otsu_threshold as cpu_otsu_threshold
from napari_vipp.core.pipeline import PrototypePipeline

CONNECTED_COMPONENTS_IMPLEMENTATION_ID = "cupyx-connected-components-v1"
OTSU_IMPLEMENTATION_ID = "cupy-otsu-threshold-exact-v1"


def _connected_components_pipeline() -> tuple[PrototypePipeline, str, str]:
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    otsu = pipeline.add_node("otsu_threshold")
    labels = pipeline.add_node("label_connected_components")
    pipeline.set_param(labels.id, "spatial_mode", "2D YX")
    pipeline.set_param(labels.id, "connectivity", "Face connected")
    assert pipeline.connect("input", otsu.id).success
    assert pipeline.connect(otsu.id, labels.id).success
    return pipeline, otsu.id, labels.id


def test_fixed_int32_metadata_and_facts_are_projected_without_data() -> None:
    pipeline, _otsu_id, node_id = _connected_components_pipeline()
    mask = np.zeros((3, 17, 19), dtype=bool)
    axes = (
        AxisMetadata("t", "time"),
        AxisMetadata("y", "space"),
        AxisMetadata("x", "space"),
    )
    input_state = image_state_from_array(mask, axes=axes)
    assert input_state is not None
    call = pipeline.prepare_node_call(node_id, (mask,), (input_state,))
    (spec,) = compute_specs_for(
        "label_connected_components",
        include_cpu=False,
    )

    (predicted_state,) = execution_module._predict_device_node_states(
        pipeline,
        call,
        spec,
        (SimpleNamespace(shape=mask.shape, dtype=np.dtype(np.int32)),),
    )

    assert predicted_state is not None
    assert predicted_state.shape == mask.shape
    assert predicted_state.dtype == "int32"
    assert predicted_state.bit_depth == "32-bit integer"
    assert predicted_state.kind == "label image"
    assert tuple(axis.name for axis in predicted_state.axes) == ("t", "y", "x")
    assert predicted_state.channels == ()

    source_facts = ArrayFacts(
        mask.shape,
        "bool",
        mask.size,
        "resident-mask-revision-v1",
        completeness=FactCompleteness.UNKNOWN,
    )
    propagated = execution_module._propagate_shape_preserving_facts(
        "label_connected_components",
        source_facts,
        {"spatial_mode": "2D YX", "connectivity": "Face connected"},
        output_port=OutputPortKey(node_id, 0),
        output_shape=mask.shape,
        output_dtype="int32",
    )

    assert propagated is not None
    assert propagated.shape == mask.shape
    assert propagated.dtype == "int32"
    assert propagated.element_count == mask.size
    assert propagated.completeness is FactCompleteness.COMPLETE
    assert propagated.finite_count == mask.size
    assert propagated.minimum is None
    assert propagated.maximum is None
    assert propagated.label_count is None
    assert propagated.label_maximum is None
    assert {"integer-labels", "nonnegative", "no-negative-zero"} <= set(
        propagated.guarantees
    )


def test_otsu_to_connected_components_planning_never_scans_source_values(
    monkeypatch,
) -> None:
    pipeline, otsu_id, node_id = _connected_components_pipeline()
    y, x = np.indices((31, 37), dtype=np.uint16)
    image = np.asarray(x * 977 + y * 613, dtype=np.uint16)
    compute_request = ComputeRequest(
        mode=ComputeMode.CUSTOM,
        node_preferences={
            otsu_id: f"implementation:{OTSU_IMPLEMENTATION_ID}",
            node_id: f"implementation:{CONNECTED_COMPONENTS_IMPLEMENTATION_ID}",
        },
        runtime_id="cuda-cupy",
        device_id="cuda:0",
        fallback_policy=FallbackPolicy.STRICT,
    )
    planner = _CapturingPlanner()

    def forbidden_scan(*_args, **_kwargs):
        raise AssertionError(
            "Otsu-to-connected-components planning must not scan data."
        )

    monkeypatch.setattr(execution_module, "_complete_array_facts", forbidden_scan)
    result = _execute_accelerated(
        _fact_planning_request(
            pipeline,
            image,
            compute_request=compute_request,
            performance_evidence={},
        ),
        planner,
    )

    assert result.error == ""
    assert set(planner.array_facts) == {node_id}
    (otsu_output_facts,) = planner.array_facts[node_id]
    assert otsu_output_facts.shape == image.shape
    assert otsu_output_facts.dtype == "bool"
    assert otsu_output_facts.element_count == image.size
    assert otsu_output_facts.completeness is FactCompleteness.COMPLETE
    assert otsu_output_facts.all_finite is True
    assert {"nonnegative", "no-negative-zero"} <= set(
        otsu_output_facts.guarantees
    )
    (workload,) = tuple(
        item
        for item in planner.workloads
        if item.operation_id == "label_connected_components"
    )
    assert workload.input_shapes == (image.shape,)
    assert workload.input_dtypes == ("bool",)
    assert workload.resolved_spatial_ndim == 2


def test_real_otsu_to_connected_components_is_one_resident_cuda_segment(
    monkeypatch,
) -> None:
    if importlib.util.find_spec("cupy") is None:
        pytest.skip("CuPy is not installed.")

    registry = ComputeRegistry()
    try:
        runtime_probe = registry.probe_runtime("cuda-cupy", refresh=True)
        if not runtime_probe.available or not runtime_probe.selected_device_id:
            pytest.skip(runtime_probe.message or "The CUDA runtime is unavailable.")
        library_probe = registry.probe_library("cupyx", refresh=True)
        if not library_probe.available:
            pytest.skip(library_probe.message or "CuPyX is unavailable.")

        pipeline = PrototypePipeline()
        pipeline.reset_empty_graph()
        otsu = pipeline.add_node("otsu_threshold")
        labels = pipeline.add_node("label_connected_components")
        pipeline.set_param(otsu.id, "threshold_scope", "Stack histogram")
        pipeline.set_param(otsu.id, "histogram_bins", 256)
        pipeline.set_param(labels.id, "spatial_mode", "2D YX")
        pipeline.set_param(labels.id, "connectivity", "Face connected")
        assert pipeline.connect("input", otsu.id).success
        assert pipeline.connect(otsu.id, labels.id).success

        time, y, x = np.indices((3, 73, 89), dtype=np.uint32)
        image = (
            x * 977
            + y * 613
            + time * 4093
            + ((x - 44) ** 2 + (y - 36) ** 2) * 17
        )
        image = np.asarray(image % 65_536, dtype=np.uint16)
        expected_mask = cpu_otsu_threshold(
            image,
            threshold_scope="Stack histogram",
            histogram_bins=256,
        )
        expected = cpu_label_connected_components(
            expected_mask,
            spatial_mode="2D YX",
            connectivity="Face connected",
        )

        compute_request = ComputeRequest(
            mode=ComputeMode.CUSTOM,
            node_preferences={
                otsu.id: f"implementation:{OTSU_IMPLEMENTATION_ID}",
                labels.id: (
                    f"implementation:{CONNECTED_COMPONENTS_IMPLEMENTATION_ID}"
                ),
            },
            runtime_id="cuda-cupy",
            device_id=runtime_probe.selected_device_id,
            fallback_policy=FallbackPolicy.STRICT,
        )
        planned_workloads: dict[str, WorkloadDescriptor] = {}

        def planner(request, workloads, **kwargs):
            planned_workloads.update(
                (workload.node_id, workload) for workload in workloads
            )
            return plan_compute_decisions(request, workloads, **kwargs)

        runtime = registry.runtime("cuda-cupy")
        transfers = {"host_to_device": 0, "device_to_host": 0}
        original_to_device = runtime.to_device
        original_to_host = runtime.to_host

        def counted_to_device(value, *, device_id=""):
            transfers["host_to_device"] += 1
            return original_to_device(value, device_id=device_id)

        def counted_to_host(value):
            transfers["device_to_host"] += 1
            return original_to_host(value)

        def forbidden_scan(*_args, **_kwargs):
            raise AssertionError(
                "The exact uint16 Otsu-to-labels chain must not scan host facts."
            )

        monkeypatch.setattr(runtime, "to_device", counted_to_device)
        monkeypatch.setattr(runtime, "to_host", counted_to_host)
        monkeypatch.setattr(execution_module, "_complete_array_facts", forbidden_scan)
        request = replace(
            _accelerated_request(
                pipeline,
                image,
                compute_request,
                retain_node_ids=frozenset({labels.id}),
                prune_unretained=True,
            ),
            input_metadata={"axes": "TYX"},
        )

        result = execute_pipeline_request(
            request,
            compute_registry=registry,
            compute_planner=planner,
        )

        assert result.error == ""
        assert result.pipeline is not None
        assert result.execution_report is not None
        assert result.execution_report.cleanup_succeeded
        np.testing.assert_array_equal(result.pipeline.outputs[labels.id], expected)

        decisions = {
            decision.node_id: decision
            for decision in result.execution_report.actual_decisions
            if decision.node_id in {otsu.id, labels.id}
        }
        assert decisions[otsu.id].decision_kind is DecisionKind.SELECTED
        assert decisions[otsu.id].implementation_id == OTSU_IMPLEMENTATION_ID
        assert decisions[labels.id].decision_kind is DecisionKind.SELECTED
        assert decisions[labels.id].implementation_id == (
            CONNECTED_COMPONENTS_IMPLEMENTATION_ID
        )

        assert len(result.execution_report.plan.segments) == 1
        (segment,) = result.execution_report.plan.segments
        assert segment.node_ids == (otsu.id, labels.id)
        assert len(segment.entry_ports) == 1
        assert len(segment.exit_ports) == 1
        assert transfers == {"host_to_device": 1, "device_to_host": 1}

        assert planned_workloads[otsu.id].input_dtypes == ("uint16",)
        assert planned_workloads[labels.id].input_shapes == (expected_mask.shape,)
        assert planned_workloads[labels.id].input_dtypes == ("bool",)
        assert planned_workloads[labels.id].resolved_spatial_ndim == 2

        state = result.pipeline.output_states[labels.id]
        assert state.shape == expected.shape
        assert state.axis_order == "TYX"
        assert tuple(axis.name for axis in state.axes) == ("t", "y", "x")
        assert state.dtype == "int32"
        assert state.kind == "label image"
        assert state.channels == ()

        _assert_private_cuda_scope_clean(
            runtime,
            runtime_probe.selected_device_id,
        )
    finally:
        registry.close()
