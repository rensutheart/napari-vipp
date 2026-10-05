"""Time-series label evidence survives CPU, device and fallback boundaries."""

import threading
from dataclasses import replace

import numpy as np
import pytest

import napari_vipp.core.device_execution as device_execution
import napari_vipp.core.measurements as measurements
from napari_vipp._tests.test_device_execution import _device_copy, _FakeOOM
from napari_vipp._tests.test_example_compute_matrix import (
    _VALIDATED_TEST_HOST,
    _UnavailableAcceleratorLibrariesRegistry,
)
from napari_vipp._tests.test_gpu_execution_integration import (
    _decision,
    _ShapeAwareRuntime,
    _StaticPlanner,
    _test_registry,
)
from napari_vipp._tests.test_tracking_integration import payload
from napari_vipp.core.compute import ComputeMode, ComputeRequest
from napari_vipp.core.detection import _revision
from napari_vipp.core.execution import PipelineRunRequest, execute_pipeline_request
from napari_vipp.core.host_memory import HostMemorySnapshot
from napari_vipp.core.observation_series import OBJECT_OBSERVATION_REVISION_KEY
from napari_vipp.core.operations import measure_objects, measure_objects_with_intensity
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.workflow import serialize_workflow


def _packed_cpu_reference_provider(values, **kwargs):
    """Emulate only the resident ABI; this is not GPU numerical validation."""
    intensity = isinstance(values, list)
    value = values[0] if intensity else values
    runtime = value.runtime
    runtime.operation_count += 1
    if runtime.oom_remaining:
        runtime.oom_remaining -= 1
        raise _FakeOOM("synthetic measurement allocation failure")
    labels = value.payload
    table = (
        measure_objects_with_intensity([v.payload for v in values], **kwargs)
        if intensity
        else measure_objects(labels, **kwargs)
    )
    layout = measurements.basic_measurement_layout(
        labels.shape,
        include_intensity=intensity,
        **{
            key: kwargs[key]
            for key in (
                "spatial_mode",
                "resolved_spatial_ndim",
                "axis_names",
                "axis_types",
                "axis_scales",
                "axis_units",
            )
            if key in kwargs
        },
    )
    indices = [table.columns.index(name) for name in layout.packed_columns]
    packed = np.array(
        [[row[index] for index in indices] for row in table.rows], dtype=np.float64
    ).reshape((-1, layout.packed_width))
    return runtime.allocate(packed)


def _case(*, intensity=False, volume=False, resident=False):
    data = np.zeros((3, 4, 8, 10) if volume else (3, 8, 10), dtype=np.int32)
    for frame in range(3):
        data[(frame, 1, 3, 3 + frame) if volume else (frame, 3, 3 + frame)] = 1
    data.setflags(write=False)
    source = payload(labels=True, volume=volume, data=data)
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    threshold = pipeline.add_node("binary_threshold")
    components = pipeline.add_node("label_connected_components")
    pipeline.set_param(threshold.id, "threshold", 0.0)
    pipeline.set_param(components.id, "spatial_mode", "3D ZYX" if volume else "2D YX")
    assert pipeline.connect("input", threshold.id).success
    assert pipeline.connect(threshold.id, components.id).success
    source_id = components.id
    if resident:
        upstream = pipeline.add_node("expand_labels")
        pipeline.set_param(upstream.id, "distance", 0)
        assert pipeline.connect(source_id, upstream.id).success
        source_id = upstream.id
    operation = "measure_objects_intensity" if intensity else "measure_objects"
    measurement = pipeline.add_node(operation)
    linker = pipeline.add_node("build_tracks")
    pipeline.set_param(measurement.id, "spatial_mode", "3D ZYX" if volume else "2D YX")
    pipeline.set_param(linker.id, "maximum_displacement", 3.0)
    assert pipeline.connect(source_id, measurement.id, target_port=0).success
    if intensity:
        assert pipeline.connect("input", measurement.id, target_port=1).success
    assert pipeline.connect(measurement.id, linker.id).success
    request = PipelineRunRequest(
        run_id=1,
        workflow=serialize_workflow(pipeline),
        input_data=None,
        input_metadata=None,
        input_name="",
        source_payloads={"input": source},
        compute_request=ComputeRequest(mode=ComputeMode.CPU),
        manual_node_ids=frozenset(pipeline.manual_node_ids()),
    )
    return pipeline, measurement, linker, source, request


@pytest.mark.parametrize("intensity", (False, True))
@pytest.mark.parametrize("volume", (False, True))
def test_prefer_gpu_cpu_plan_preserves_measurement_evidence(
    monkeypatch, intensity, volume
):
    _, measured, linked, source, request = _case(intensity=intensity, volume=volume)
    cpu = execute_pipeline_request(request)
    monkeypatch.setattr(
        "napari_vipp.core.compute_planning.ComputeEnvironment",
        lambda: _VALIDATED_TEST_HOST,
    )
    with _UnavailableAcceleratorLibrariesRegistry() as registry:
        preferred = execute_pipeline_request(
            replace(
                request, compute_request=ComputeRequest(mode=ComputeMode.PREFER_GPU)
            ),
            compute_registry=registry,
        )
    _assert_equivalent(cpu, preferred, measured.id, linked.id, source)


def _assert_equivalent(cpu, actual, measured_id, linked_id, source):
    assert cpu.error == actual.error == ""
    assert actual.pipeline is not None and cpu.pipeline is not None
    expected = cpu.pipeline.outputs[measured_id]
    observed = actual.pipeline.outputs[measured_id]
    assert observed.columns == expected.columns
    np.testing.assert_equal(observed.rows, expected.rows)
    assert observed.observation_metadata == expected.observation_metadata
    assert observed.observation_metadata.source_revision == _revision(source.data)
    assert observed.observation_metadata.frame_count == 3
    assert actual.pipeline.outputs[linked_id] == cpu.pipeline.outputs[linked_id]
    assert not source.data.flags.writeable
    assert np.count_nonzero(source.data) == 3


@pytest.mark.parametrize("intensity", (False, True))
@pytest.mark.parametrize("volume", (False, True))
@pytest.mark.parametrize("fallback", (False, True))
def test_native_and_oom_fallback_finalize_before_linking(
    monkeypatch, intensity, volume, fallback
):
    pipeline, measured, linked, source, request = _case(
        intensity=intensity, volume=volume
    )
    cpu = execute_pipeline_request(request)
    runtime = _ShapeAwareRuntime(free_bytes=10_000_000)
    runtime.oom_remaining = int(fallback)
    finalizer_calls = []
    authoritative_finalizer = measurements.finalize_basic_measurement_outputs

    def inspect_finalizer(outputs, *, call):
        assert runtime.live == {} and not runtime.scope_active
        assert call.inputs == (None,) * len(call.inputs)
        assert call.kwargs[OBJECT_OBSERVATION_REVISION_KEY] == _revision(source.data)
        finalizer_calls.append(call)
        return authoritative_finalizer(outputs, call=call)

    monkeypatch.setattr(
        measurements, "finalize_basic_measurement_outputs", inspect_finalizer
    )
    result = _native_run(pipeline, request, runtime)
    _assert_equivalent(cpu, result, measured.id, linked.id, source)
    assert len(finalizer_calls) == (0 if fallback else 1)
    assert runtime.device_to_host_count == (0 if fallback else 1)
    assert runtime.live == {}
    assert len(result.execution_report.fallback_records) == int(fallback)


def _native_run(pipeline, request, runtime, **request_changes):
    measurement_ops = {
        node.operation_id
        for node in pipeline.nodes.values()
        if node.operation_id.startswith("measure_objects")
    }
    implementations = [(op, _packed_cpu_reference_provider) for op in measurement_ops]
    if any(node.operation_id == "expand_labels" for node in pipeline.nodes.values()):
        implementations.append(("expand_labels", _device_copy))
    registry, specs = _test_registry(
        runtime,
        tuple(implementations),
        host_finalizer_refs={
            op: "napari_vipp.core.measurements:finalize_basic_measurement_outputs"
            for op in measurement_ops
        },
    )
    compute = ComputeRequest(
        mode=ComputeMode.PREFER_GPU, runtime_id="cuda-cupy", device_id="cuda:0"
    )
    planner = _StaticPlanner(
        compute,
        tuple(
            _decision(node.id, specs[node.operation_id])
            for node in pipeline.nodes.values()
            if node.operation_id in specs
        ),
    )
    with registry:
        return execute_pipeline_request(
            replace(request, compute_request=compute, **request_changes),
            compute_registry=registry,
            compute_planner=planner,
        )


@pytest.mark.parametrize("retain_labels", (False, True))
def test_resident_labels_get_one_guarded_revision_copy(monkeypatch, retain_labels):
    pipeline, measured, linked, source, request = _case(resident=True)
    cpu = execute_pipeline_request(request)
    admitted = []
    admission = device_execution.preflight_host_allocation

    def record_admission(snapshot, **kwargs):
        admitted.append(kwargs["required_bytes"])
        return admission(snapshot, **kwargs)

    monkeypatch.setattr(device_execution, "preflight_host_allocation", record_admission)
    runtime = _ShapeAwareRuntime(free_bytes=10_000_000)
    result = _native_run(
        pipeline,
        request,
        runtime,
        retain_node_ids=frozenset({measured.id, linked.id}),
        prune_unretained=not retain_labels,
    )
    _assert_equivalent(cpu, result, measured.id, linked.id, source)
    assert admitted == [source.data.nbytes]
    assert runtime.device_to_host_count == 2
    assert runtime.live == {}


def test_revision_transfer_refusal_cleans_device_without_publishing(monkeypatch):
    pipeline, measured, linked, _, request = _case(resident=True)
    monkeypatch.setattr(
        device_execution,
        "capture_host_memory",
        lambda: HostMemorySnapshot(
            platform="linux",
            source="unavailable",
            detail="test admission refusal",
        ),
    )
    runtime = _ShapeAwareRuntime(free_bytes=10_000_000)
    result = _native_run(pipeline, request, runtime)
    assert result.error and "source-revision transfer" in result.error
    assert runtime.device_to_host_count == 0
    assert runtime.live == {}
    assert not result.execution_report or not result.execution_report.fallback_records
    assert result.pipeline.outputs.get(measured.id) is None
    assert result.pipeline.outputs.get(linked.id) is None


def test_cancel_after_revision_transfer_cleans_device_without_publishing():
    pipeline, measured, linked, _, request = _case(resident=True)
    cancelled = threading.Event()

    class CancellingRuntime(_ShapeAwareRuntime):
        def to_host(self, value):
            output = super().to_host(value)
            cancelled.set()
            return output

    runtime = CancellingRuntime(free_bytes=10_000_000)
    result = _native_run(pipeline, request, runtime, cancel_event=cancelled)
    assert result.cancelled
    assert runtime.device_to_host_count == 1
    assert runtime.live == {}
    assert result.pipeline is None or result.pipeline.outputs.get(measured.id) is None
    assert result.pipeline is None or result.pipeline.outputs.get(linked.id) is None
