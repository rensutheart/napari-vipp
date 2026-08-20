from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import napari_vipp.core.execution as execution_module
from napari_vipp.core import metadata as _metadata
from napari_vipp.core.compute import (
    ComputeEnvironment,
    ComputeMode,
    ComputeRequest,
    DecisionKind,
    DecisionReason,
    ExecutionReport,
    NodeComputePreference,
    NodeExecutionDecision,
    NodePreferenceKind,
    OutputPortKey,
)
from napari_vipp.core.execution import (
    PipelineNodeResult,
    PipelineRunRequest,
    PipelineRunResult,
    ResidentThumbnailStatisticsCleanupError,
    ResidentThumbnailStatisticsObservation,
    ResidentThumbnailStatisticsRequest,
    execute_pipeline_request,
)
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.progress import OperationCancelled, ProgressContext
from napari_vipp.core.thumbnail_statistics import (
    EXACT_FLOAT32_PERCENTILE_GPU_ALGORITHM_ID,
    ThumbnailStatisticsBackend,
    ThumbnailStatisticsDecision,
    ThumbnailStatisticsResult,
)
from napari_vipp.core.workflow import serialize_workflow


class _OpaqueDeviceValue:
    def __init__(self, shape=(2, 4, 5), dtype=np.float32) -> None:
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.size = int(np.prod(self.shape, dtype=np.int64))
        self.nbytes = self.size * self.dtype.itemsize

    def __array__(self, _dtype=None, copy=None):
        del copy
        raise AssertionError("Resident device values must never be coerced to NumPy.")


class _CudaRuntime:
    runtime_id = "cuda-cupy"


def _image_state(shape=(2, 4, 5), dtype=np.float32):
    axes = (
        _metadata.AxisMetadata("c", "channel"),
        _metadata.AxisMetadata("y", "space"),
        _metadata.AxisMetadata("x", "space"),
    )
    return _metadata.image_state_from_array(
        np.zeros(shape, dtype=dtype),
        axes=axes,
        defer_statistics=True,
    )


def _request(**changes) -> ResidentThumbnailStatisticsRequest:
    values = {
        "node_id": "node-b",
        "output_port": 0,
        "contrast_mode": "Percentile",
        "minimum_scanned_bytes": 64,
        "gpu_contract_warm": True,
    }
    values.update(changes)
    return ResidentThumbnailStatisticsRequest(**values)


def _result() -> ThumbnailStatisticsResult:
    return ThumbnailStatisticsResult(
        limits=(1.0, 7.0),
        decision=ThumbnailStatisticsDecision(
            backend=ThumbnailStatisticsBackend.GPU_CUPY,
            reason_code="resident_float32_warm_contract",
            reason="test",
            scanned_values=10,
            scanned_bytes=40,
            threshold_bytes=0,
            gpu_warm=True,
        ),
        actual_backend=ThumbnailStatisticsBackend.GPU_CUPY,
        algorithm_id=EXACT_FLOAT32_PERCENTILE_GPU_ALGORITHM_ID,
        elapsed_seconds=0.1,
        runtime_id="cuda-cupy",
        device_id="cuda:0",
        requested_compute_mode=ComputeMode.PREFER_GPU,
        input_path="resident_borrow",
        logical_input_host_to_device_bytes=0,
    )


def _observation(node_id="node-b", output_port=0):
    return ResidentThumbnailStatisticsObservation(
        node_id=node_id,
        output_port=output_port,
        contrast_mode="percentile",
        result=_result(),
    )


def test_resident_request_validates_and_normalizes_its_exact_port():
    request = _request(
        node_id=" node-b ",
        output_port=np.int64(2),
        contrast_mode="minimum maximum",
        minimum_scanned_bytes=np.int64(128),
    )

    assert request.port == OutputPortKey("node-b", 2)
    assert request.contrast_mode == "Min-max"
    assert request.minimum_scanned_bytes == 128
    with pytest.raises(ValueError, match="minimum_scanned_bytes"):
        _request(minimum_scanned_bytes=-1)
    with pytest.raises(TypeError, match="gpu_contract_warm"):
        _request(gpu_contract_warm=1)


def test_resident_float32_observation_is_zero_upload_host_only(monkeypatch):
    value = _OpaqueDeviceValue()
    state = _image_state()
    calls = []

    def calculate(runtime, device_value, **kwargs):
        calls.append((runtime, device_value, kwargs))
        return SimpleNamespace(
            limits=np.asarray(((1.0, 9.0), (2.0, 8.0))),
            auxiliary_host_to_device_bytes=40,
            device_to_host_bytes=64,
            device_to_host_values=8,
        )

    monkeypatch.setattr(
        execution_module,
        "_exact_float32_thumbnail_limits_from_device",
        calculate,
    )
    request = _request(minimum_scanned_bytes=value.nbytes)

    observation = execution_module._resident_thumbnail_statistics_observation(
        request,
        compute_mode=ComputeMode.PREFER_GPU,
        output_type="any",
        output_state=state,
        device_value=value,
        runtime=_CudaRuntime(),
        device_id="cuda:0",
        progress=ProgressContext(),
    )

    assert observation is not None
    assert calls[0][1] is value
    assert calls[0][2]["channel_axis"] == 0
    assert calls[0][2]["contrast_mode"] == "Percentile"
    assert observation.contrast_mode == "Percentile"
    assert observation.result.limits == ((1.0, 9.0), (2.0, 8.0))
    assert isinstance(observation.result.limits, tuple)
    assert observation.result.input_path == "resident_borrow"
    assert observation.result.logical_input_host_to_device_bytes == 0
    assert observation.result.auxiliary_host_to_device_bytes == 40
    assert observation.result.device_to_host_bytes == 64
    assert observation.result.device_to_host_values == 8
    assert observation.result.decision.scanned_bytes == value.nbytes
    assert observation.result.decision.threshold_bytes == value.nbytes
    assert observation.result.decision.gpu_warm


def test_resident_channel_limit_count_mismatch_is_a_soft_miss(monkeypatch):
    value = _OpaqueDeviceValue(shape=(2, 4, 5))

    monkeypatch.setattr(
        execution_module,
        "_exact_float32_thumbnail_limits_from_device",
        lambda *_args, **_kwargs: SimpleNamespace(
            limits=np.asarray((1.0, 9.0)),
            auxiliary_host_to_device_bytes=0,
            device_to_host_bytes=16,
            device_to_host_values=2,
        ),
    )

    observation = execution_module._resident_thumbnail_statistics_observation(
        _request(),
        compute_mode=ComputeMode.PREFER_GPU,
        output_type="image",
        output_state=_image_state(),
        device_value=value,
        runtime=_CudaRuntime(),
        device_id="cuda:0",
        progress=ProgressContext(),
    )

    assert observation is None


@pytest.mark.parametrize(
    (
        "resident_request",
        "mode",
        "output_type",
        "runtime_id",
        "dtype",
        "minimum_delta",
    ),
    (
        (
            _request(gpu_contract_warm=False),
            ComputeMode.PREFER_GPU,
            "image",
            "cuda-cupy",
            np.float32,
            0,
        ),
        (_request(), ComputeMode.AUTO, "image", "cuda-cupy", np.float32, 0),
        (_request(), ComputeMode.PREFER_GPU, "labels", "cuda-cupy", np.float32, 0),
        (_request(), ComputeMode.PREFER_GPU, "image", "other-runtime", np.float32, 0),
        (_request(), ComputeMode.PREFER_GPU, "image", "cuda-cupy", np.uint16, 0),
        (_request(), ComputeMode.PREFER_GPU, "image", "cuda-cupy", np.float32, 1),
    ),
)
def test_resident_observation_requires_every_explicit_gate(
    monkeypatch,
    resident_request,
    mode,
    output_type,
    runtime_id,
    dtype,
    minimum_delta,
):
    value = _OpaqueDeviceValue(dtype=dtype)
    if minimum_delta:
        resident_request = _request(minimum_scanned_bytes=value.nbytes + minimum_delta)
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("Ineligible outputs must not enter the provider.")

    monkeypatch.setattr(
        execution_module,
        "_exact_float32_thumbnail_limits_from_device",
        forbidden,
    )
    runtime = _CudaRuntime()
    runtime.runtime_id = runtime_id

    observation = execution_module._resident_thumbnail_statistics_observation(
        resident_request,
        compute_mode=mode,
        output_type=output_type,
        output_state=_image_state(dtype=dtype),
        device_value=value,
        runtime=runtime,
        device_id="cuda:0",
        progress=ProgressContext(),
    )

    assert observation is None
    assert not called


def test_resident_provider_failure_is_soft_but_cancellation_propagates(monkeypatch):
    value = _OpaqueDeviceValue()
    kwargs = dict(
        request=_request(),
        compute_mode=ComputeMode.PREFER_GPU,
        output_type="image",
        output_state=_image_state(),
        device_value=value,
        runtime=_CudaRuntime(),
        device_id="cuda:0",
        progress=ProgressContext(),
    )

    monkeypatch.setattr(
        execution_module,
        "_exact_float32_thumbnail_limits_from_device",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("soft miss")),
    )
    assert execution_module._resident_thumbnail_statistics_observation(**kwargs) is None

    monkeypatch.setattr(
        execution_module,
        "_exact_float32_thumbnail_limits_from_device",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OperationCancelled("cancelled")
        ),
    )
    with pytest.raises(OperationCancelled):
        execution_module._resident_thumbnail_statistics_observation(**kwargs)


def test_resident_provider_cleanup_failure_is_fatal(monkeypatch):
    class ProviderCleanupError(RuntimeError):
        cleanup_succeeded = False

    monkeypatch.setattr(
        execution_module,
        "_exact_float32_thumbnail_limits_from_device",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ProviderCleanupError("scratch release failed")
        ),
    )

    with pytest.raises(ResidentThumbnailStatisticsCleanupError) as error:
        execution_module._resident_thumbnail_statistics_observation(
            _request(),
            compute_mode=ComputeMode.PREFER_GPU,
            output_type="image",
            output_state=_image_state(),
            device_value=_OpaqueDeviceValue(),
            runtime=_CudaRuntime(),
            device_id="cuda:0",
            progress=ProgressContext(),
        )

    assert error.value.cleanup_succeeded is False


def test_pipeline_results_carry_deterministic_port_keyed_observations():
    first = _observation("node-b", 1)
    second = _observation("node-a", 0)

    result = PipelineRunResult(
        run_id=1,
        workflow={},
        resident_thumbnail_statistics=(first, second),
    )
    node_result = PipelineNodeResult(
        run_id=1,
        node_id="node-b",
        operation_id="gaussian_blur",
        output=None,
        output_state=None,
        node_outputs=(),
        node_output_states=(),
        execution_state="ready",
        resident_thumbnail_statistics=(first,),
    )

    assert tuple(item.port for item in result.resident_thumbnail_statistics) == (
        OutputPortKey("node-a", 0),
        OutputPortKey("node-b", 1),
    )
    assert node_result.resident_thumbnail_statistics == (first,)
    with pytest.raises(ValueError, match="only for that node"):
        PipelineNodeResult(
            run_id=1,
            node_id="node-a",
            operation_id="gaussian_blur",
            output=None,
            output_state=None,
            node_outputs=(),
            node_output_states=(),
            execution_state="ready",
            resident_thumbnail_statistics=(first,),
        )


def test_execute_pipeline_attaches_observation_to_incremental_and_final_results(
    monkeypatch,
):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    gaussian = pipeline.add_node("gaussian_blur")
    pipeline.set_param(gaussian.id, "sigma", 0.0)
    assert pipeline.connect("input", gaussian.id).success
    observation = _observation(gaussian.id)

    def execute_on_cpu_for_carrier_test(detached, request, **kwargs):
        kwargs["resident_thumbnail_statistics"].append(observation)
        kwargs["resident_observer_seconds"][0] += 0.25
        detached.run(
            request.input_data,
            input_metadata=request.input_metadata,
            input_name=request.input_name,
            source_payloads=request.source_payloads,
            node_started_callback=kwargs["node_started_callback"],
            node_finished_callback=kwargs["node_finished_callback"],
            progress_callback=kwargs["progress_callback"],
            cancel_callback=kwargs["cancel_callback"],
        )
        return ExecutionReport(
            request=request.compute_request,
            environment=ComputeEnvironment(),
        )

    monkeypatch.setattr(
        execution_module,
        "_execute_accelerated_pipeline",
        execute_on_cpu_for_carrier_test,
    )
    finished = []
    result = execute_pipeline_request(
        PipelineRunRequest(
            run_id=9,
            workflow=serialize_workflow(pipeline),
            input_data=np.ones((4, 4), dtype=np.float32),
            input_metadata={"axes": "YX"},
            input_name="source",
            source_payloads={},
            compute_request=ComputeRequest(mode=ComputeMode.PREFER_GPU),
            resident_thumbnail_statistics_request=_request(node_id=gaussian.id),
        ),
        node_finished_callback=finished.append,
    )

    assert result.error == ""
    assert result.resident_thumbnail_statistics == (observation,)
    gaussian_result = next(item for item in finished if item.node_id == gaussian.id)
    assert gaussian_result.resident_thumbnail_statistics == (observation,)
    input_result = next(item for item in finished if item.node_id == "input")
    assert input_result.resident_thumbnail_statistics == ()


def test_cleanup_integrity_failure_reaches_terminal_quarantine_result(monkeypatch):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()

    def fail_with_cleanup(*_args, **_kwargs):
        raise ResidentThumbnailStatisticsCleanupError("cleanup failed")

    monkeypatch.setattr(
        execution_module,
        "_execute_accelerated_pipeline",
        fail_with_cleanup,
    )
    result = execute_pipeline_request(
        PipelineRunRequest(
            run_id=10,
            workflow=serialize_workflow(pipeline),
            input_data=np.ones((2, 2), dtype=np.float32),
            input_metadata={"axes": "YX"},
            input_name="source",
            source_payloads={},
            compute_request=ComputeRequest(mode=ComputeMode.PREFER_GPU),
        )
    )

    assert result.failure is not None
    assert result.failure.cleanup_succeeded is False
    assert result.resident_thumbnail_statistics == ()


def test_resident_callback_time_is_excluded_from_scientific_history(monkeypatch):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    gaussian = pipeline.add_node("gaussian_blur")
    assert pipeline.connect("input", gaussian.id).success
    captured_samples = []
    now = [100.0]

    class TimingStore:
        def __init__(self, _path):
            pass

        def append(self, sample):
            captured_samples.append(sample)

    def timed_accelerated_run(_pipeline, request, **kwargs):
        now[0] += 5.0
        kwargs["resident_observer_seconds"][0] += 2.0
        decision = NodeExecutionDecision(
            node_id=gaussian.id,
            operation_id="gaussian_blur",
            requested_preference=NodeComputePreference(NodePreferenceKind.BEST_GPU),
            runtime_id="cuda-cupy",
            implementation_library_id="cupyx",
            implementation_id="cupyx-gaussian_blur-v1",
            decision_kind=DecisionKind.SELECTED,
            reason=DecisionReason.SELECTED_IMPLEMENTATION,
            reason_text="test",
            implementation_version="1",
        )
        return ExecutionReport(
            request=request.compute_request,
            environment=ComputeEnvironment(
                runtime_ids=("cpu-numpy", "cuda-cupy"),
                implementation_libraries=("cpu", "cupyx"),
                device_id="cuda:0",
                device_name="Test GPU",
                device_class="gpu",
            ),
            actual_decisions=(decision,),
        )

    monkeypatch.setattr(execution_module, "JsonPipelineTimingStore", TimingStore)
    monkeypatch.setattr(
        execution_module,
        "_pipeline_timing_workload_fingerprint",
        lambda *_args, **_kwargs: "workload",
    )
    monkeypatch.setattr(
        execution_module,
        "host_performance_fingerprint",
        lambda: "host",
    )
    monkeypatch.setattr(execution_module, "perf_counter", lambda: now[0])
    monkeypatch.setattr(
        execution_module,
        "_execute_accelerated_pipeline",
        timed_accelerated_run,
    )

    result = execute_pipeline_request(
        PipelineRunRequest(
            run_id=11,
            workflow=serialize_workflow(pipeline),
            input_data=np.ones((4, 4), dtype=np.float32),
            input_metadata={"axes": "YX"},
            input_name="source",
            source_payloads={},
            compute_request=ComputeRequest(mode=ComputeMode.PREFER_GPU),
            performance_history_path="unused-history.json",
        )
    )

    assert result.error == ""
    assert len(captured_samples) == 1
    assert captured_samples[0].elapsed_seconds == pytest.approx(3.0)
