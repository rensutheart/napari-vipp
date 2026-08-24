from __future__ import annotations

import threading
from dataclasses import replace

import numpy as np

import napari_vipp.core.execution as execution_module
from napari_vipp.core.compute import (
    ComputeMode,
    ComputeRequest,
    DecisionReason,
    ExecutionFallbackRecord,
)
from napari_vipp.core.compute_planning import (
    ComputePreflightError,
    ComputePreflightFailure,
)
from napari_vipp.core.device_execution import (
    DeviceMemoryNodeEstimate,
    DeviceMemoryPreflightError,
)
from napari_vipp.core.execution import (
    PipelineExecutionFailure,
    PipelineNodeResult,
    PipelineRunRequest,
    execute_pipeline_request,
)
from napari_vipp.core.execution_telemetry import (
    DeviceExecutionTelemetryConfig,
    PipelinePreparationPhase,
)
from napari_vipp.core.host_memory import HostMemorySnapshot
from napari_vipp.core.metadata import image_state_from_array
from napari_vipp.core.pipeline import (
    EXECUTION_BLOCKED,
    EXECUTION_ERROR,
    EXECUTION_READY,
    EXECUTION_STALE,
    NODE_LIBRARY_BY_ID,
    PrototypePipeline,
    SourcePayload,
)
from napari_vipp.core.source_identity import (
    BundledSampleRevisionToken,
    LocalSourceIdentity,
    SourceRevisionToken,
)
from napari_vipp.core.workflow import serialize_workflow


def _input_only_workflow() -> dict:
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    return serialize_workflow(pipeline)


def _pipeline_cache_kwargs(pipeline: PrototypePipeline) -> dict[str, object]:
    return {
        "cached_outputs": dict(pipeline.outputs),
        "cached_output_states": dict(pipeline.output_states),
        "cached_node_outputs": {
            node_id: list(outputs) for node_id, outputs in pipeline.node_outputs.items()
        },
        "cached_node_output_states": {
            node_id: list(states)
            for node_id, states in pipeline.node_output_states.items()
        },
        "completed_node_ids": frozenset(pipeline.completed_node_ids),
        "cached_execution_states": dict(pipeline.node_execution_states),
        "cached_execution_messages": dict(pipeline.node_execution_messages),
        "cached_compute_provenance": {
            **pipeline.node_cache_lineage,
            **pipeline.node_compute_provenance,
        },
    }


def test_execute_pipeline_request_materializes_a_detached_graph():
    data = np.arange(12, dtype=np.uint16).reshape(3, 4)
    request = PipelineRunRequest(
        run_id=7,
        workflow=_input_only_workflow(),
        input_data=data,
        input_metadata={"axes": "YX"},
        input_name="source",
        source_payloads={},
        source_revisions=("revision-1",),
    )
    assert request.compute_request.mode is ComputeMode.CPU
    started: list[str] = []
    finished: list[PipelineNodeResult] = []

    result = execute_pipeline_request(
        request,
        node_started_callback=started.append,
        node_finished_callback=finished.append,
    )

    assert result.run_id == 7
    assert result.error == ""
    assert not result.cancelled
    assert result.source_revisions == ("revision-1",)
    assert result.execution_report is not None
    assert result.execution_report.request.mode is ComputeMode.CPU
    assert result.execution_report.actual_decisions == ()
    assert result.device_execution_telemetry is None
    assert result.pre_device_execution_telemetry is None
    assert result.pipeline is not None
    assert started == ["input"]
    assert [node.node_id for node in finished] == ["input"]
    assert finished[0].run_id == 7
    assert finished[0].operation_id == "input"
    assert finished[0].source_revisions == ("revision-1",)
    assert len(finished[0].node_outputs) == 1
    assert finished[0].node_outputs[0] is data
    assert finished[0].output_state is not None
    np.testing.assert_array_equal(result.pipeline.outputs["input"], data)


def test_cpu_request_reports_opt_in_detached_preparation_phases():
    ticks = iter(float(index) for index in range(100))
    request = PipelineRunRequest(
        run_id=70,
        workflow=_input_only_workflow(),
        input_data=np.arange(12, dtype=np.uint16).reshape(3, 4),
        input_metadata={"axes": "YX"},
        input_name="source",
        source_payloads={},
        device_execution_telemetry=DeviceExecutionTelemetryConfig(
            clock=lambda: next(ticks)
        ),
    )

    result = execute_pipeline_request(request)

    assert result.error == ""
    assert result.device_execution_telemetry is None
    observation = result.pre_device_execution_telemetry
    assert observation is not None
    assert observation.completed is True
    assert [span.phase for span in observation.spans] == [
        PipelinePreparationPhase.GRAPH_RESTORATION,
        PipelinePreparationPhase.CACHE_PREPARATION,
        PipelinePreparationPhase.WORKLOAD_PREPARATION,
        PipelinePreparationPhase.CACHE_PREPARATION,
    ]
    assert all(span.succeeded for span in observation.spans)


def test_invalid_graph_reports_partial_preparation_observation():
    ticks = iter(float(index) for index in range(100))
    result = execute_pipeline_request(
        PipelineRunRequest(
            run_id=72,
            workflow={"type": "not-a-vipp-workflow"},
            input_data=np.zeros((3, 4), dtype=np.float32),
            input_metadata={"axes": "YX"},
            input_name="source",
            source_payloads={},
            device_execution_telemetry=DeviceExecutionTelemetryConfig(
                clock=lambda: next(ticks)
            ),
        )
    )

    assert result.error
    observation = result.pre_device_execution_telemetry
    assert observation is not None
    assert observation.completed is False
    assert len(observation.spans) == 1
    assert observation.spans[-1].phase is (PipelinePreparationPhase.GRAPH_RESTORATION)
    assert observation.spans[-1].succeeded is False


def test_cpu_failure_reports_completed_sibling_prefix_with_provenance(
    monkeypatch,
):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    gaussian = pipeline.add_node("gaussian_blur")
    median = pipeline.add_node("median_filter")
    failing = pipeline.add_node("gamma_correction")
    pipeline.set_param(gaussian.id, "sigma", 0.0)
    pipeline.set_param(median.id, "size", 1)
    for node in (gaussian, median, failing):
        assert pipeline.connect("input", node.id).success

    original_gamma = NODE_LIBRARY_BY_ID["gamma_correction"]

    def fail_gamma(_image, **_kwargs):
        raise ValueError("sibling sentinel")

    monkeypatch.setitem(
        NODE_LIBRARY_BY_ID,
        "gamma_correction",
        replace(original_gamma, function=fail_gamma),
    )
    data = np.arange(20, dtype=np.float32).reshape(4, 5)

    result = execute_pipeline_request(
        PipelineRunRequest(
            run_id=71,
            workflow=serialize_workflow(pipeline),
            input_data=data,
            input_metadata={"axes": "YX"},
            input_name="source",
            source_payloads={},
        )
    )

    assert result.error == "sibling sentinel"
    assert result.pipeline is not None
    assert result.execution_report is not None
    assert [
        decision.node_id for decision in result.execution_report.actual_decisions
    ] == [gaussian.id, median.id]
    assert result.pipeline.node_execution_states[gaussian.id] == EXECUTION_READY
    assert result.pipeline.node_execution_states[median.id] == EXECUTION_READY
    assert result.pipeline.node_execution_states[failing.id] == EXECUTION_ERROR
    assert {"input", gaussian.id, median.id} <= set(
        result.pipeline.node_compute_provenance
    )
    assert failing.id not in result.pipeline.node_compute_provenance


def test_cpu_report_keeps_pruned_intermediate_implementation_decisions():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    gaussian = pipeline.add_node("gaussian_blur")
    median = pipeline.add_node("median_filter")
    pipeline.set_param(gaussian.id, "sigma", 0.0)
    pipeline.set_param(median.id, "size", 3)
    assert pipeline.connect("input", gaussian.id).success
    assert pipeline.connect(gaussian.id, median.id).success
    data = np.arange(42, dtype=np.float32).reshape(6, 7)

    result = execute_pipeline_request(
        PipelineRunRequest(
            run_id=8,
            workflow=serialize_workflow(pipeline),
            input_data=data,
            input_metadata={"axes": "YX"},
            input_name="source",
            source_payloads={},
            retain_node_ids=frozenset({median.id}),
            prune_unretained=True,
        )
    )

    assert result.error == ""
    assert result.pipeline is not None
    assert result.execution_report is not None
    assert tuple(
        (decision.node_id, decision.implementation_id)
        for decision in result.execution_report.actual_decisions
    ) == (
        (gaussian.id, "cpu-gaussian_blur-v1"),
        (median.id, "cpu-median_filter-v1"),
    )
    assert result.pipeline.outputs[gaussian.id] is None
    assert gaussian.id not in result.pipeline.completed_node_ids
    assert result.pipeline.outputs[median.id] is not None


def test_execute_pipeline_request_reports_invalid_workflow_without_raising():
    request = PipelineRunRequest(
        run_id=11,
        workflow={"not": "a workflow"},
        input_data=None,
        input_metadata=None,
        input_name="",
        source_payloads={},
    )

    result = execute_pipeline_request(request)

    assert result.run_id == 11
    assert result.pipeline is None
    assert result.error
    assert not result.cancelled
    assert result.failure is not None
    assert result.failure.kind == "execution_error"


def test_execute_pipeline_request_can_preserve_generated_api_exception_type():
    request = PipelineRunRequest(
        run_id=12,
        workflow={"not": "a workflow"},
        input_data=None,
        input_metadata=None,
        input_name="",
        source_payloads={},
    )

    with np.testing.assert_raises(ValueError) as raised:
        execute_pipeline_request(request, raise_errors=True)

    detail = raised.exception.vipp_execution_failure
    assert detail["kind"] == "execution_error"
    assert detail["reason_code"] == "unclassified_execution_error"


def test_cancelled_cpu_result_has_typed_cleanup_provenance():
    cancelled = threading.Event()
    cancelled.set()
    result = execute_pipeline_request(
        PipelineRunRequest(
            run_id=13,
            workflow=_input_only_workflow(),
            input_data=np.zeros((3, 4), dtype=np.float32),
            input_metadata={"axes": "YX"},
            input_name="source",
            source_payloads={},
            cancel_event=cancelled,
        )
    )

    assert result.cancelled
    assert result.failure is not None
    assert result.failure.kind == "cancelled"
    assert result.failure.cleanup_succeeded is True
    assert result.failure.as_dict()["reason_code"] == "operation_cancelled"


def test_cancellation_interrupts_initial_exact_source_hash(monkeypatch):
    cancel_event = threading.Event()
    original_chunks = execution_module._iter_exact_array_chunks

    def cancelling_chunks(*args, **kwargs):
        for chunk in original_chunks(*args, **kwargs):
            cancel_event.set()
            yield chunk

    monkeypatch.setattr(
        execution_module,
        "_iter_exact_array_chunks",
        cancelling_chunks,
    )
    data = np.arange(4096, dtype=np.float32).reshape(64, 64)
    result = execute_pipeline_request(
        PipelineRunRequest(
            run_id=131,
            workflow=_input_only_workflow(),
            input_data=data,
            input_metadata={"axes": "YX"},
            input_name="source",
            source_payloads={},
            cancel_event=cancel_event,
        )
    )

    assert result.cancelled
    assert result.failure is not None
    assert result.failure.error_type == "OperationCancelled"


def test_cancellation_interrupts_source_context_reuse_after_envelope(monkeypatch):
    data = np.arange(25, dtype=np.float32).reshape(5, 5)
    data.setflags(write=False)
    payload = SourcePayload(
        data,
        {},
        "owned",
        image_state_from_array(data, source_name="owned"),
        SourceRevisionToken(layer_id=41, revision=2),
    )
    pipeline = PrototypePipeline()
    pipeline.reset_starter_graph()
    initial = execute_pipeline_request(
        PipelineRunRequest(
            run_id=132,
            workflow=serialize_workflow(pipeline),
            input_data=None,
            input_metadata=None,
            input_name="",
            source_payloads={"input": payload},
        )
    ).pipeline
    assert initial is not None
    cancel_event = threading.Event()
    original_envelope = (
        execution_module._source_scientific_context_reuse_envelope_fingerprint
    )

    def cancel_after_envelope(*args, **kwargs):
        envelope = original_envelope(*args, **kwargs)
        cancel_event.set()
        return envelope

    monkeypatch.setattr(
        execution_module,
        "_source_scientific_context_reuse_envelope_fingerprint",
        cancel_after_envelope,
    )
    result = execute_pipeline_request(
        PipelineRunRequest(
            run_id=133,
            workflow=serialize_workflow(initial),
            input_data=None,
            input_metadata=None,
            input_name="",
            source_payloads={"input": payload},
            dirty_node_ids=frozenset({"threshold"}),
            cancel_event=cancel_event,
            **_pipeline_cache_kwargs(initial),
        )
    )

    assert result.cancelled
    assert result.failure is not None
    assert result.failure.error_type == "OperationCancelled"


def test_pre_cancelled_request_does_not_restore_graph(monkeypatch):
    cancelled = threading.Event()
    cancelled.set()

    def unexpected_deserialize(_workflow):
        raise AssertionError("a pre-cancelled request must not restore its graph")

    monkeypatch.setattr(
        execution_module,
        "deserialize_workflow",
        unexpected_deserialize,
    )
    result = execute_pipeline_request(
        PipelineRunRequest(
            run_id=130,
            workflow=_input_only_workflow(),
            input_data=np.zeros((3, 4), dtype=np.float32),
            input_metadata={"axes": "YX"},
            input_name="source",
            source_payloads={},
            cancel_event=cancelled,
            device_execution_telemetry=DeviceExecutionTelemetryConfig(),
        )
    )

    assert result.cancelled
    observation = result.pre_device_execution_telemetry
    assert observation is not None
    assert observation.completed is False
    assert len(observation.spans) == 1
    assert observation.spans[0].phase is (PipelinePreparationPhase.GRAPH_RESTORATION)
    assert observation.spans[0].succeeded is False


def test_accelerated_cancellation_before_setup_skips_registry_construction(
    monkeypatch,
):
    cancelled = threading.Event()

    class CountingRegistry:
        implementation_specs = ()
        construction_count = 0

        def __init__(self) -> None:
            type(self).construction_count += 1

        @staticmethod
        def close() -> None:
            raise AssertionError("an obsolete request must not own a registry")

    monkeypatch.setattr(
        "napari_vipp.core.compute_registry.ComputeRegistry",
        CountingRegistry,
    )
    monkeypatch.setattr(
        execution_module,
        "_capture_source_scientific_contexts",
        lambda *_args, **_kwargs: cancelled.set() or {},
    )
    result = execute_pipeline_request(
        PipelineRunRequest(
            run_id=14,
            workflow=_input_only_workflow(),
            input_data=np.zeros((3, 4), dtype=np.float32),
            input_metadata={"axes": "YX"},
            input_name="source",
            source_payloads={},
            compute_request=ComputeRequest(mode=ComputeMode.AUTO),
            cancel_event=cancelled,
            device_execution_telemetry=DeviceExecutionTelemetryConfig(),
        )
    )

    assert result.error
    assert result.cancelled
    assert result.failure is not None
    assert result.failure.kind == "cancelled"
    assert result.failure.error_type == "OperationCancelled"
    assert CountingRegistry.construction_count == 0
    observation = result.pre_device_execution_telemetry
    assert observation is not None
    assert observation.completed is False
    assert observation.spans[-1].phase is (
        PipelinePreparationPhase.WORKLOAD_PREPARATION
    )
    assert observation.spans[-1].succeeded is False


def test_accelerated_cancellation_after_setup_reports_registry_cleanup_failure(
    monkeypatch,
):
    cancelled = threading.Event()

    class FailingRegistry:
        implementation_specs = ()

        def __init__(self) -> None:
            cancelled.set()

        @staticmethod
        def close() -> None:
            raise RuntimeError("provider would not close")

    monkeypatch.setattr(
        "napari_vipp.core.compute_registry.ComputeRegistry",
        FailingRegistry,
    )
    result = execute_pipeline_request(
        PipelineRunRequest(
            run_id=140,
            workflow=_input_only_workflow(),
            input_data=np.zeros((3, 4), dtype=np.float32),
            input_metadata={"axes": "YX"},
            input_name="source",
            source_payloads={},
            compute_request=ComputeRequest(mode=ComputeMode.AUTO),
            cancel_event=cancelled,
        )
    )

    assert result.error
    assert result.cancelled
    assert result.failure is not None
    assert result.failure.kind == "cancelled"
    assert result.failure.error_type == "AcceleratorCleanupError"
    assert result.failure.cleanup_succeeded is False
    assert "provider would not close" in result.failure.message


def test_accelerated_planner_setup_failure_closes_owned_registry(monkeypatch):
    class ClosingRegistry:
        implementation_specs = ()
        close_count = 0

        @classmethod
        def close(cls) -> None:
            cls.close_count += 1

    monkeypatch.setattr(
        "napari_vipp.core.compute_registry.ComputeRegistry",
        ClosingRegistry,
    )

    def fail_planner_setup():
        raise RuntimeError("planner setup sentinel")

    monkeypatch.setattr(
        execution_module,
        "_default_compute_planner",
        fail_planner_setup,
    )
    result = execute_pipeline_request(
        PipelineRunRequest(
            run_id=141,
            workflow=_input_only_workflow(),
            input_data=np.zeros((3, 4), dtype=np.float32),
            input_metadata={"axes": "YX"},
            input_name="source",
            source_payloads={},
            compute_request=ComputeRequest(mode=ComputeMode.AUTO),
            device_execution_telemetry=DeviceExecutionTelemetryConfig(),
        )
    )

    assert "planner setup sentinel" in result.error
    assert ClosingRegistry.close_count == 1
    observation = result.pre_device_execution_telemetry
    assert observation is not None
    assert observation.completed is False
    assert observation.spans[-1].phase is (PipelinePreparationPhase.ACCELERATOR_SETUP)
    assert observation.spans[-1].succeeded is False


def test_terminal_failure_serializes_prior_oom_retry_attempts():
    record = ExecutionFallbackRecord(
        segment_id="segment-1",
        runtime_id="cuda-cupy",
        node_ids=("gaussian-1",),
        reason="out_of_memory",
        reason_code="cupy_oom",
        cpu_retry_succeeded=False,
    )
    failure = PipelineExecutionFailure(
        kind="execution_error",
        error_type="ValueError",
        message="CPU retry failed",
        reason_code="unclassified_execution_error",
        fallback_records=(record,),
    )

    assert failure.as_dict()["fallback_records"] == [record.as_dict()]


def test_compute_preflight_failure_proves_no_device_cleanup_was_required():
    failure = ComputePreflightFailure(
        node_id="median-1",
        operation_id="median_filter",
        preference=ComputeRequest(mode="custom").preference_for("median-1"),
        reason=DecisionReason.WORKLOAD_UNSUPPORTED,
        reason_text="The exact workload is not eligible.",
    )

    detail = execution_module._pipeline_execution_failure(
        ComputePreflightError((failure,))
    )

    assert detail.kind == "compute_preflight"
    assert detail.reason_code == "compute_preflight_rejected"
    assert detail.cleanup_succeeded is True


def test_host_memory_error_records_available_physical_and_commit_headroom(
    monkeypatch,
):
    monkeypatch.setattr(
        execution_module,
        "capture_host_memory",
        lambda: HostMemorySnapshot(
            platform="win32",
            source="windows_global_memory_status_ex",
            physical_total_bytes=32_000,
            physical_available_bytes=8_000,
            commit_limit_bytes=64_000,
            commit_available_bytes=2_000,
        ),
    )

    detail = execution_module._pipeline_execution_failure(MemoryError("allocation"))

    assert detail.kind == "host_memory_oom"
    assert detail.reason_code == "host_allocation_failed"
    assert detail.available_bytes == 2_000


def test_device_memory_preflight_failure_keeps_machine_readable_diagnostics():
    error = DeviceMemoryPreflightError(
        "device-segment-001",
        "cuda-cupy",
        7_176_934_402,
        5_314_183_168,
        device_id="cuda:0",
        device_name="NVIDIA Test GPU",
        device_total_bytes=8_000_000_000,
        device_free_bytes=6_387_925_992,
        safety_reserve_bytes=1_073_742_824,
        memory_cap_bytes=7_000_000_000,
        limiting_constraint="free_vram_minus_reserve",
        node_estimates=(
            DeviceMemoryNodeEstimate(
                "otsu-2",
                "otsu_threshold",
                "Otsu Threshold",
                2_573_388_802,
                "otsu-test-v1",
            ),
            DeviceMemoryNodeEstimate(
                "remove-4",
                "remove_small_objects",
                "Remove Small Objects",
                4_603_545_600,
                "remove-test-v1",
            ),
        ),
    )

    detail = execution_module._pipeline_execution_failure(error)
    document = detail.as_dict()

    assert detail.kind == "memory_preflight"
    assert detail.node_ids == ("otsu-2", "remove-4")
    assert detail.shortfall_bytes == 1_862_751_234
    assert detail.device_free_bytes == 6_387_925_992
    assert detail.safety_reserve_bytes == 1_073_742_824
    assert detail.limiting_constraint == "free_vram_minus_reserve"
    assert document["device_memory_node_estimates"][1] == {
        "node_id": "remove-4",
        "operation_id": "remove_small_objects",
        "title": "Remove Small Objects",
        "required_bytes": 4_603_545_600,
        "model_id": "remove-test-v1",
    }


def test_accelerated_planning_error_preserves_only_resolved_source_boundary():
    pipeline = PrototypePipeline()
    pipeline.reset_starter_graph()
    data = np.zeros((7, 9, 3), dtype=np.uint16)
    request = PipelineRunRequest(
        run_id=12,
        workflow=serialize_workflow(pipeline),
        input_data=data,
        input_metadata={"axes": "YXC"},
        input_name="multichannel source",
        source_payloads={},
        compute_request=ComputeRequest(mode="auto"),
    )
    started: list[str] = []
    finished: list[PipelineNodeResult] = []

    result = execute_pipeline_request(
        request,
        node_started_callback=started.append,
        node_finished_callback=finished.append,
    )

    assert result.error
    assert "effective axis order is YXC" in result.error
    assert result.pipeline is not None
    assert started[0] == "input"
    assert [item.node_id for item in finished] == ["input"]
    np.testing.assert_array_equal(result.pipeline.outputs["input"], data)
    assert result.pipeline.outputs["gaussian"] is None
    assert result.pipeline.outputs["threshold"] is None


def test_dirty_execution_hydrates_and_reuses_clean_cached_outputs():
    data = np.arange(25, dtype=np.float32).reshape(5, 5)
    initial = PrototypePipeline()
    initial.reset_starter_graph()
    initial.run(data, input_metadata={"axes": "YX"}, input_name="source")
    cached_input = initial.outputs["input"]
    initial.set_param("gaussian", "sigma", 0.0)

    request = PipelineRunRequest(
        run_id=13,
        workflow=serialize_workflow(initial),
        input_data=data,
        input_metadata={"axes": "YX"},
        input_name="source",
        source_payloads={},
        dirty_node_ids=frozenset({"gaussian"}),
        cached_outputs=dict(initial.outputs),
        cached_output_states=dict(initial.output_states),
        cached_node_outputs={
            node_id: list(outputs) for node_id, outputs in initial.node_outputs.items()
        },
        cached_node_output_states={
            node_id: list(states)
            for node_id, states in initial.node_output_states.items()
        },
        completed_node_ids=frozenset(initial.completed_node_ids),
        cached_execution_states=dict(initial.node_execution_states),
        cached_execution_messages=dict(initial.node_execution_messages),
    )

    result = execute_pipeline_request(request)

    assert result.pipeline is not None
    assert result.pipeline.outputs["input"] is cached_input
    np.testing.assert_array_equal(result.pipeline.outputs["gaussian"], data)


def test_detached_low_memory_run_preserves_manual_sibling_across_wrappers():
    data = np.zeros((9, 9), dtype=np.float32)
    data[1:4, 1:4] = 10
    data[6:8, 6:8] = 20
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    threshold = pipeline.add_node("binary_threshold")
    labels = pipeline.add_node("label_connected_components")
    morphology = pipeline.add_node("measure_objects")
    intensity = pipeline.add_node("measure_objects_intensity")
    merged = pipeline.add_node("merge_tables")
    pipeline.set_param(threshold.id, "threshold", 5)
    assert pipeline.connect("input", threshold.id).success
    assert pipeline.connect(threshold.id, labels.id).success
    assert pipeline.connect(labels.id, morphology.id).success
    assert pipeline.connect(labels.id, intensity.id, target_port=0).success
    assert pipeline.connect("input", intensity.id, target_port=1).success
    assert pipeline.connect(morphology.id, merged.id, target_port=0).success
    assert pipeline.connect(intensity.id, merged.id, target_port=1).success

    identity = LocalSourceIdentity("file", "a" * 64, 1, data.nbytes)
    retained = frozenset({morphology.id, intensity.id, merged.id})

    def rematerialized_payload() -> SourcePayload:
        current = np.array(data, copy=True)
        current.setflags(write=False)
        return SourcePayload(
            current,
            {
                "axes": "YX",
                "vipp_source_path": "C:/verified/source.tif",
                "vipp_source_series_index": 0,
            },
            "Verified series",
            revision_token=identity,
        )

    initial = execute_pipeline_request(
        PipelineRunRequest(
            run_id=130,
            workflow=serialize_workflow(pipeline),
            input_data=None,
            input_metadata=None,
            input_name="",
            source_payloads={"input": rematerialized_payload()},
            retain_node_ids=retained,
            prune_unretained=True,
        )
    ).pipeline
    assert initial is not None
    assert initial.outputs[labels.id] is None
    assert {"input", threshold.id, labels.id} <= set(
        initial.node_cache_lineage
    )

    current = initial
    cached_intensity = None
    for run_id, manual_node in (
        (131, morphology),
        (132, intensity),
        (133, morphology),
    ):
        result = execute_pipeline_request(
            PipelineRunRequest(
                run_id=run_id,
                workflow=serialize_workflow(current),
                input_data=None,
                input_metadata=None,
                input_name="",
                source_payloads={"input": rematerialized_payload()},
                dirty_node_ids=frozenset({manual_node.id}),
                manual_node_ids=frozenset({manual_node.id}),
                retain_node_ids=retained,
                prune_unretained=True,
                **_pipeline_cache_kwargs(current),
            )
        )
        assert result.error == ""
        assert result.pipeline is not None
        current = result.pipeline
        if run_id == 132:
            cached_intensity = current.outputs[intensity.id]

    assert current.node_execution_states[morphology.id] == EXECUTION_READY
    assert current.node_execution_states[intensity.id] == EXECUTION_READY
    assert current.node_execution_states[merged.id] == EXECUTION_READY
    assert current.outputs[intensity.id] is cached_intensity
    assert current.outputs[merged.id].row_count == 2


def test_successful_cpu_request_publishes_processing_node_provenance():
    data = np.arange(25, dtype=np.float32).reshape(5, 5)
    pipeline = PrototypePipeline()
    pipeline.reset_starter_graph()
    request = PipelineRunRequest(
        run_id=131,
        workflow=serialize_workflow(pipeline),
        input_data=data,
        input_metadata={"axes": "YX"},
        input_name="source",
        source_payloads={},
        compute_request=ComputeRequest(mode=ComputeMode.CPU),
    )

    result = execute_pipeline_request(request)

    assert result.error == ""
    assert result.pipeline is not None
    assert set(result.pipeline.node_compute_provenance) == {
        "input",
        "gaussian",
        "threshold",
    }
    gaussian = result.pipeline.node_compute_provenance["gaussian"]
    assert gaussian.actual_implementation.implementation_id == ("cpu-gaussian_blur-v1")
    assert not gaussian.produced_by_fallback


def test_dirty_cpu_request_reuses_only_provenance_admitted_upstream_cache():
    data = np.arange(25, dtype=np.float32).reshape(5, 5)
    pipeline = PrototypePipeline()
    pipeline.reset_starter_graph()
    compute_request = ComputeRequest(mode=ComputeMode.CPU)
    initial = execute_pipeline_request(
        PipelineRunRequest(
            run_id=132,
            workflow=serialize_workflow(pipeline),
            input_data=data,
            input_metadata={"axes": "YX"},
            input_name="source",
            source_payloads={},
            compute_request=compute_request,
        )
    ).pipeline
    assert initial is not None
    cached_gaussian = initial.outputs["gaussian"]
    started: list[str] = []

    result = execute_pipeline_request(
        PipelineRunRequest(
            run_id=133,
            workflow=serialize_workflow(initial),
            input_data=data,
            input_metadata={"axes": "YX"},
            input_name="source",
            source_payloads={},
            compute_request=compute_request,
            dirty_node_ids=frozenset({"threshold"}),
            **_pipeline_cache_kwargs(initial),
        ),
        node_started_callback=started.append,
    )

    assert result.error == ""
    assert result.pipeline is not None
    assert started == ["threshold"]
    assert result.pipeline.outputs["gaussian"] is cached_gaussian

    initial.node_compute_provenance.pop("gaussian")
    initial.node_cache_lineage.pop("gaussian")
    started.clear()
    rejected = execute_pipeline_request(
        PipelineRunRequest(
            run_id=134,
            workflow=serialize_workflow(initial),
            input_data=data,
            input_metadata={"axes": "YX"},
            input_name="source",
            source_payloads={},
            compute_request=compute_request,
            dirty_node_ids=frozenset({"threshold"}),
            **_pipeline_cache_kwargs(initial),
        ),
        node_started_callback=started.append,
    )

    assert rejected.error == ""
    assert started == ["gaussian", "threshold"]


def test_dirty_request_reuses_owned_exact_source_context_without_byte_scan(
    monkeypatch,
):
    data = np.arange(25, dtype=np.float32).reshape(5, 5)
    data.setflags(write=False)
    payload = SourcePayload(
        data,
        {"axes": "YX"},
        "owned live snapshot",
        image_state_from_array(
            data,
            layer_metadata={"axes": "YX"},
            source_name="owned live snapshot",
        ),
        SourceRevisionToken(layer_id=17, revision=4),
    )
    pipeline = PrototypePipeline()
    pipeline.reset_starter_graph()
    compute_request = ComputeRequest(mode=ComputeMode.CPU)
    initial = execute_pipeline_request(
        PipelineRunRequest(
            run_id=1321,
            workflow=serialize_workflow(pipeline),
            input_data=None,
            input_metadata=None,
            input_name="",
            source_payloads={"input": payload},
            compute_request=compute_request,
        )
    ).pipeline
    assert initial is not None
    initial_source_provenance = initial.node_compute_provenance["input"]
    assert initial_source_provenance.source_reuse_envelope_fingerprint

    identity_calls = 0
    original_identity = execution_module._scientific_array_identity

    def counted_identity(*args, **kwargs):
        nonlocal identity_calls
        identity_calls += 1
        return original_identity(*args, **kwargs)

    monkeypatch.setattr(
        execution_module,
        "_scientific_array_identity",
        counted_identity,
    )
    started: list[str] = []
    result = execute_pipeline_request(
        PipelineRunRequest(
            run_id=1322,
            workflow=serialize_workflow(initial),
            input_data=None,
            input_metadata=None,
            input_name="",
            source_payloads={"input": payload},
            compute_request=compute_request,
            dirty_node_ids=frozenset({"threshold"}),
            **_pipeline_cache_kwargs(initial),
        ),
        node_started_callback=started.append,
    )

    assert result.error == ""
    assert result.pipeline is not None
    assert identity_calls == 0
    assert started == ["threshold"]
    assert result.pipeline.node_compute_provenance["input"] == initial_source_provenance
    fresh = execute_pipeline_request(
        PipelineRunRequest(
            run_id=13221,
            workflow=serialize_workflow(initial),
            input_data=None,
            input_metadata=None,
            input_name="",
            source_payloads={"input": payload},
            compute_request=compute_request,
        )
    )
    assert fresh.error == ""
    assert fresh.pipeline is not None
    assert (
        result.pipeline.node_compute_provenance
        == fresh.pipeline.node_compute_provenance
    )
    for node_id in result.pipeline.nodes:
        np.testing.assert_array_equal(
            result.pipeline.outputs[node_id],
            fresh.pipeline.outputs[node_id],
        )


def test_source_context_reuse_misses_changed_revision_and_writable_array(
    monkeypatch,
):
    data = np.arange(25, dtype=np.float32).reshape(5, 5)
    data.setflags(write=False)
    state = image_state_from_array(data, source_name="owned source")
    payload = SourcePayload(
        data,
        {},
        "owned source",
        state,
        SourceRevisionToken(layer_id=23, revision=1),
    )
    pipeline = PrototypePipeline()
    pipeline.reset_starter_graph()
    compute_request = ComputeRequest(mode=ComputeMode.CPU)
    initial = execute_pipeline_request(
        PipelineRunRequest(
            run_id=1323,
            workflow=serialize_workflow(pipeline),
            input_data=None,
            input_metadata=None,
            input_name="",
            source_payloads={"input": payload},
            compute_request=compute_request,
        )
    ).pipeline
    assert initial is not None

    identity_calls = 0
    original_identity = execution_module._scientific_array_identity

    def counted_identity(*args, **kwargs):
        nonlocal identity_calls
        identity_calls += 1
        return original_identity(*args, **kwargs)

    monkeypatch.setattr(
        execution_module,
        "_scientific_array_identity",
        counted_identity,
    )
    changed_revision = replace(
        payload,
        revision_token=SourceRevisionToken(layer_id=23, revision=2),
    )
    changed = execute_pipeline_request(
        PipelineRunRequest(
            run_id=1324,
            workflow=serialize_workflow(initial),
            input_data=None,
            input_metadata=None,
            input_name="",
            source_payloads={"input": changed_revision},
            compute_request=compute_request,
            dirty_node_ids=frozenset({"threshold"}),
            **_pipeline_cache_kwargs(initial),
        )
    )

    assert changed.error == ""
    assert identity_calls >= 1

    identity_calls = 0
    data.setflags(write=True)
    writable = execute_pipeline_request(
        PipelineRunRequest(
            run_id=1325,
            workflow=serialize_workflow(initial),
            input_data=None,
            input_metadata=None,
            input_name="",
            source_payloads={"input": payload},
            compute_request=compute_request,
            dirty_node_ids=frozenset({"threshold"}),
            **_pipeline_cache_kwargs(initial),
        )
    )

    assert writable.error == ""
    assert identity_calls >= 1


def test_file_and_bundled_source_revisions_skip_repeat_byte_scans(monkeypatch):
    tokens = (
        LocalSourceIdentity("file", "b" * 64, 1, 4096),
        BundledSampleRevisionToken("sample"),
    )
    identity_calls = 0
    original_identity = execution_module._scientific_array_identity

    def counted_identity(*args, **kwargs):
        nonlocal identity_calls
        identity_calls += 1
        return original_identity(*args, **kwargs)

    for index, token in enumerate(tokens):
        data = np.arange(25, dtype=np.float32).reshape(5, 5)
        data.setflags(write=False)
        payload = SourcePayload(
            data,
            {"axes": "YX"},
            f"owned-{index}",
            image_state_from_array(
                data,
                layer_metadata={"axes": "YX"},
                source_name=f"owned-{index}",
            ),
            token,
        )
        pipeline = PrototypePipeline()
        pipeline.reset_starter_graph()
        initial = execute_pipeline_request(
            PipelineRunRequest(
                run_id=13260 + index * 2,
                workflow=serialize_workflow(pipeline),
                input_data=None,
                input_metadata=None,
                input_name="",
                source_payloads={"input": payload},
            )
        ).pipeline
        assert initial is not None
        monkeypatch.setattr(
            execution_module,
            "_scientific_array_identity",
            counted_identity,
        )
        before = identity_calls
        repeated = execute_pipeline_request(
            PipelineRunRequest(
                run_id=13261 + index * 2,
                workflow=serialize_workflow(initial),
                input_data=None,
                input_metadata=None,
                input_name="",
                source_payloads={"input": payload},
                dirty_node_ids=frozenset({"threshold"}),
                **_pipeline_cache_kwargs(initial),
            )
        )

        assert repeated.error == ""
        assert identity_calls == before


def test_source_context_reuse_rejects_same_file_revision_with_new_series_array(
    monkeypatch,
):
    first = np.zeros((5, 5), dtype=np.uint16)
    second = np.full((5, 5), 700, dtype=np.uint16)
    first.setflags(write=False)
    second.setflags(write=False)
    identity = LocalSourceIdentity("file", "a" * 64, 1, 4096)
    first_payload = SourcePayload(
        first,
        {"series": 0},
        "series",
        image_state_from_array(first, source_name="series"),
        identity,
    )
    second_payload = SourcePayload(
        second,
        {"series": 1},
        "series",
        image_state_from_array(second, source_name="series"),
        identity,
    )
    pipeline = PrototypePipeline()
    pipeline.reset_starter_graph()
    compute_request = ComputeRequest(mode=ComputeMode.CPU)
    initial = execute_pipeline_request(
        PipelineRunRequest(
            run_id=1326,
            workflow=serialize_workflow(pipeline),
            input_data=None,
            input_metadata=None,
            input_name="",
            source_payloads={"input": first_payload},
            compute_request=compute_request,
        )
    ).pipeline
    assert initial is not None

    identity_calls = 0
    original_identity = execution_module._scientific_array_identity

    def counted_identity(*args, **kwargs):
        nonlocal identity_calls
        identity_calls += 1
        return original_identity(*args, **kwargs)

    monkeypatch.setattr(
        execution_module,
        "_scientific_array_identity",
        counted_identity,
    )
    result = execute_pipeline_request(
        PipelineRunRequest(
            run_id=1327,
            workflow=serialize_workflow(initial),
            input_data=None,
            input_metadata=None,
            input_name="",
            source_payloads={"input": second_payload},
            compute_request=compute_request,
            dirty_node_ids=frozenset({"threshold"}),
            **_pipeline_cache_kwargs(initial),
        )
    )

    assert result.error == ""
    assert result.pipeline is not None
    assert identity_calls >= 1
    assert result.pipeline.outputs["input"] is second


def test_source_context_reuse_requires_partial_clean_source_and_envelope(
    monkeypatch,
):
    data = np.arange(25, dtype=np.float32).reshape(5, 5)
    data.setflags(write=False)
    payload = SourcePayload(
        data,
        {},
        "bundled",
        image_state_from_array(data, source_name="bundled"),
        BundledSampleRevisionToken("bundled"),
    )
    pipeline = PrototypePipeline()
    pipeline.reset_starter_graph()
    compute_request = ComputeRequest(mode=ComputeMode.CPU)
    initial = execute_pipeline_request(
        PipelineRunRequest(
            run_id=1328,
            workflow=serialize_workflow(pipeline),
            input_data=None,
            input_metadata=None,
            input_name="",
            source_payloads={"input": payload},
            compute_request=compute_request,
        )
    ).pipeline
    assert initial is not None

    identity_calls = 0
    original_identity = execution_module._scientific_array_identity

    def counted_identity(*args, **kwargs):
        nonlocal identity_calls
        identity_calls += 1
        return original_identity(*args, **kwargs)

    monkeypatch.setattr(
        execution_module,
        "_scientific_array_identity",
        counted_identity,
    )
    full = execute_pipeline_request(
        PipelineRunRequest(
            run_id=1329,
            workflow=serialize_workflow(initial),
            input_data=None,
            input_metadata=None,
            input_name="",
            source_payloads={"input": payload},
            compute_request=compute_request,
            **_pipeline_cache_kwargs(initial),
        )
    )
    assert full.error == ""
    assert identity_calls >= 1

    identity_calls = 0
    source_dirty = execute_pipeline_request(
        PipelineRunRequest(
            run_id=13291,
            workflow=serialize_workflow(initial),
            input_data=None,
            input_metadata=None,
            input_name="",
            source_payloads={"input": payload},
            compute_request=compute_request,
            dirty_node_ids=frozenset({"input"}),
            **_pipeline_cache_kwargs(initial),
        )
    )
    assert source_dirty.error == ""
    assert identity_calls >= 1

    identity_calls = 0
    no_envelope = dict(initial.node_compute_provenance)
    no_envelope["input"] = replace(
        no_envelope["input"],
        source_reuse_envelope_fingerprint="",
    )
    request_kwargs = _pipeline_cache_kwargs(initial)
    request_kwargs["cached_compute_provenance"] = no_envelope
    missing_proof = execute_pipeline_request(
        PipelineRunRequest(
            run_id=1330,
            workflow=serialize_workflow(initial),
            input_data=None,
            input_metadata=None,
            input_name="",
            source_payloads={"input": payload},
            compute_request=compute_request,
            dirty_node_ids=frozenset({"threshold"}),
            **request_kwargs,
        )
    )
    assert missing_proof.error == ""
    assert identity_calls >= 1
    assert missing_proof.pipeline is not None
    republished = missing_proof.pipeline.node_compute_provenance["input"]
    assert republished.source_reuse_envelope_fingerprint
    assert (
        republished.scientific_context_fingerprint
        == initial.node_compute_provenance["input"].scientific_context_fingerprint
    )


def test_source_context_reuse_fails_closed_for_envelope_and_cache_mismatches(
    monkeypatch,
):
    data = np.arange(25, dtype=np.float32).reshape(5, 5)
    data.setflags(write=False)
    payload = SourcePayload(
        data,
        {"axes": "YX"},
        "owned",
        image_state_from_array(
            data,
            layer_metadata={"axes": "YX"},
            source_name="owned",
        ),
        SourceRevisionToken(layer_id=31, revision=8),
    )
    pipeline = PrototypePipeline()
    pipeline.reset_starter_graph()
    compute_request = ComputeRequest(mode=ComputeMode.CPU)
    initial = execute_pipeline_request(
        PipelineRunRequest(
            run_id=1331,
            workflow=serialize_workflow(pipeline),
            input_data=None,
            input_metadata=None,
            input_name="",
            source_payloads={"input": payload},
            compute_request=compute_request,
        )
    ).pipeline
    assert initial is not None

    identity_calls = 0
    original_identity = execution_module._scientific_array_identity

    def counted_identity(*args, **kwargs):
        nonlocal identity_calls
        identity_calls += 1
        return original_identity(*args, **kwargs)

    monkeypatch.setattr(
        execution_module,
        "_scientific_array_identity",
        counted_identity,
    )

    def assert_miss(
        run_id: int,
        *,
        current_payload: SourcePayload = payload,
        workflow: dict | None = None,
        cache_changes=None,
    ) -> None:
        nonlocal identity_calls
        before = identity_calls
        cache_kwargs = _pipeline_cache_kwargs(initial)
        if cache_changes is not None:
            cache_changes(cache_kwargs)
        started: list[str] = []
        result = execute_pipeline_request(
            PipelineRunRequest(
                run_id=run_id,
                workflow=(
                    serialize_workflow(initial) if workflow is None else workflow
                ),
                input_data=None,
                input_metadata=None,
                input_name="",
                source_payloads={"input": current_payload},
                compute_request=compute_request,
                dirty_node_ids=frozenset({"threshold"}),
                **cache_kwargs,
            ),
            node_started_callback=started.append,
        )
        assert result.error == ""
        assert identity_calls > before
        assert "input" in started

    assert_miss(
        1332,
        current_payload=replace(
            payload,
            metadata={"axes": "YX", "changed": True},
        ),
    )
    source_param_workflow = serialize_workflow(initial)
    source_document = next(
        node for node in source_param_workflow["nodes"] if node["id"] == "input"
    )
    source_document["params"]["sample_name"] = "changed source binding"
    assert_miss(1333, workflow=source_param_workflow)
    assert_miss(
        1334,
        current_payload=replace(
            payload,
            revision_token=("arbitrary-token", 8),
        ),
    )

    def remove_cached_output(cache_kwargs) -> None:
        cache_kwargs["cached_node_outputs"] = dict(cache_kwargs["cached_node_outputs"])
        cache_kwargs["cached_node_outputs"].pop("input")

    assert_miss(1335, cache_changes=remove_cached_output)

    def remove_cached_provenance(cache_kwargs) -> None:
        cache_kwargs["cached_compute_provenance"] = dict(
            cache_kwargs["cached_compute_provenance"]
        )
        cache_kwargs["cached_compute_provenance"].pop("input")

    assert_miss(1336, cache_changes=remove_cached_provenance)


def test_dirty_request_rejects_cached_chain_when_source_bytes_change_in_place():
    data = np.arange(25, dtype=np.float32).reshape(5, 5)
    pipeline = PrototypePipeline()
    pipeline.reset_starter_graph()
    compute_request = ComputeRequest(mode=ComputeMode.CPU)
    initial = execute_pipeline_request(
        PipelineRunRequest(
            run_id=136,
            workflow=serialize_workflow(pipeline),
            input_data=data,
            input_metadata={"axes": "YX"},
            input_name="same-object",
            source_payloads={},
            compute_request=compute_request,
        )
    ).pipeline
    assert initial is not None
    old_gaussian = np.array(initial.outputs["gaussian"], copy=True)

    # The object and its absent revision token are unchanged. Exact bytes must
    # still prevent admission of the old source and every dependent result.
    data[...] += 100.0
    started: list[str] = []
    result = execute_pipeline_request(
        PipelineRunRequest(
            run_id=137,
            workflow=serialize_workflow(initial),
            input_data=data,
            input_metadata={"axes": "YX"},
            input_name="same-object",
            source_payloads={},
            compute_request=compute_request,
            dirty_node_ids=frozenset({"threshold"}),
            **_pipeline_cache_kwargs(initial),
        ),
        node_started_callback=started.append,
    )

    assert result.error == ""
    assert result.pipeline is not None
    assert started == ["input", "gaussian", "threshold"]
    np.testing.assert_array_equal(result.pipeline.outputs["input"], data)
    assert not np.array_equal(result.pipeline.outputs["gaussian"], old_gaussian)


def test_dirty_request_rejects_upstream_cache_when_node_parameters_change():
    data = np.zeros((9, 9), dtype=np.float32)
    data[4, 4] = 100.0
    pipeline = PrototypePipeline()
    pipeline.reset_starter_graph()
    compute_request = ComputeRequest(mode=ComputeMode.CPU)
    initial = execute_pipeline_request(
        PipelineRunRequest(
            run_id=138,
            workflow=serialize_workflow(pipeline),
            input_data=data,
            input_metadata={"axes": "YX"},
            input_name="source",
            source_payloads={},
            compute_request=compute_request,
        )
    ).pipeline
    assert initial is not None
    old_gaussian = np.array(initial.outputs["gaussian"], copy=True)
    initial.nodes["gaussian"].params["sigma"] = 4.0
    changed_workflow = serialize_workflow(initial)
    started: list[str] = []

    result = execute_pipeline_request(
        PipelineRunRequest(
            run_id=139,
            workflow=changed_workflow,
            input_data=data,
            input_metadata={"axes": "YX"},
            input_name="source",
            source_payloads={},
            compute_request=compute_request,
            dirty_node_ids=frozenset({"threshold"}),
            **_pipeline_cache_kwargs(initial),
        ),
        node_started_callback=started.append,
    )
    fresh = execute_pipeline_request(
        PipelineRunRequest(
            run_id=140,
            workflow=changed_workflow,
            input_data=data,
            input_metadata={"axes": "YX"},
            input_name="source",
            source_payloads={},
            compute_request=compute_request,
        )
    )

    assert result.error == fresh.error == ""
    assert result.pipeline is not None and fresh.pipeline is not None
    assert started == ["gaussian", "threshold"]
    assert not np.allclose(result.pipeline.outputs["gaussian"], old_gaussian)
    np.testing.assert_array_equal(
        result.pipeline.outputs["gaussian"],
        fresh.pipeline.outputs["gaussian"],
    )


def test_downstream_compute_preference_change_reuses_exact_upstream_cache():
    data = np.arange(25, dtype=np.float32).reshape(5, 5)
    pipeline = PrototypePipeline()
    pipeline.reset_starter_graph()
    initial_request = ComputeRequest(mode=ComputeMode.CPU)
    initial = execute_pipeline_request(
        PipelineRunRequest(
            run_id=141,
            workflow=serialize_workflow(pipeline),
            input_data=data,
            input_metadata={"axes": "YX"},
            input_name="source",
            source_payloads={},
            compute_request=initial_request,
        )
    ).pipeline
    assert initial is not None
    cached_gaussian = initial.outputs["gaussian"]
    downstream_changed = ComputeRequest(
        mode=ComputeMode.CPU,
        node_preferences={"threshold": "best_gpu"},
    )
    started: list[str] = []

    result = execute_pipeline_request(
        PipelineRunRequest(
            run_id=142,
            workflow=serialize_workflow(initial),
            input_data=data,
            input_metadata={"axes": "YX"},
            input_name="source",
            source_payloads={},
            compute_request=downstream_changed,
            dirty_node_ids=frozenset({"threshold"}),
            **_pipeline_cache_kwargs(initial),
        ),
        node_started_callback=started.append,
    )

    assert result.error == ""
    assert result.pipeline is not None
    assert started == ["threshold"]
    assert result.pipeline.outputs["gaussian"] is cached_gaussian


def test_pipeline_cache_lifecycle_removes_compute_provenance_with_outputs():
    data = np.arange(25, dtype=np.float32).reshape(5, 5)
    pipeline = PrototypePipeline()
    pipeline.reset_starter_graph()
    result = execute_pipeline_request(
        PipelineRunRequest(
            run_id=135,
            workflow=serialize_workflow(pipeline),
            input_data=data,
            input_metadata={"axes": "YX"},
            input_name="source",
            source_payloads={},
        )
    ).pipeline
    assert result is not None

    result.prune_cached_outputs({"gaussian"})
    assert set(result.node_compute_provenance) == {"gaussian"}
    assert result.remove_node("gaussian")
    assert result.node_compute_provenance == {}
    result.reset_empty_graph()
    assert result.node_compute_provenance == {}


def test_background_request_honors_isolated_tuning_target():
    data = np.arange(81, dtype=np.float32).reshape(9, 9)
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    gaussian = pipeline.add_node("gaussian_blur")
    threshold = pipeline.add_node("binary_threshold")
    pipeline.set_param(threshold.id, "threshold", 30.0)
    assert pipeline.connect("input", gaussian.id).success
    assert pipeline.connect(gaussian.id, threshold.id).success
    pipeline.run(data, input_metadata={"axes": "YX"})
    cached_threshold = pipeline.outputs[threshold.id]
    pipeline.set_param(gaussian.id, "sigma", 0.0)
    pipeline.mark_nodes_stale(
        {threshold.id},
        message="Downstream propagation is paused.",
    )
    request = PipelineRunRequest(
        run_id=14,
        workflow=serialize_workflow(pipeline),
        input_data=data,
        input_metadata={"axes": "YX"},
        input_name="source",
        source_payloads={},
        dirty_node_ids=frozenset({gaussian.id}),
        target_node_ids=frozenset({gaussian.id}),
        cached_outputs=dict(pipeline.outputs),
        cached_output_states=dict(pipeline.output_states),
        cached_node_outputs={
            node_id: list(outputs) for node_id, outputs in pipeline.node_outputs.items()
        },
        cached_node_output_states={
            node_id: list(states)
            for node_id, states in pipeline.node_output_states.items()
        },
        completed_node_ids=frozenset(pipeline.completed_node_ids),
        cached_execution_states=dict(pipeline.node_execution_states),
        cached_execution_messages=dict(pipeline.node_execution_messages),
    )
    started: list[str] = []

    result = execute_pipeline_request(request, node_started_callback=started.append)

    assert result.error == ""
    assert result.pipeline is not None
    assert started == ["input", gaussian.id]
    np.testing.assert_array_equal(result.pipeline.outputs[gaussian.id], data)
    assert result.pipeline.outputs[threshold.id] is cached_threshold
    assert result.pipeline.node_execution_states[threshold.id] == EXECUTION_STALE


def test_background_request_holds_descendants_behind_stale_manual_node():
    data = np.zeros((9, 9), dtype=np.float32)
    data[2:7, 2:7] = 0.1
    data[4, 4] = 1.0
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    psf = pipeline.add_node("gaussian_blur")
    deconvolution = pipeline.add_node("richardson_lucy_deconvolution")
    rescale = pipeline.add_node("rescale_intensity")
    otsu = pipeline.add_node("otsu_threshold")
    pipeline.set_param(deconvolution.id, "spatial_mode", "2D YX")
    pipeline.set_param(deconvolution.id, "iterations", 1)
    assert pipeline.connect("input", psf.id).success
    assert pipeline.connect("input", deconvolution.id, target_port=0).success
    assert pipeline.connect(psf.id, deconvolution.id, target_port=1).success
    assert pipeline.connect(deconvolution.id, rescale.id).success
    assert pipeline.connect(rescale.id, otsu.id).success
    pipeline.run(data, input_metadata={"axes": "YX"})
    pipeline.set_param(psf.id, "sigma", 2.0)
    pipeline.mark_manual_descendants_stale({psf.id})
    cached_deconvolution = pipeline.outputs[deconvolution.id]
    cached_rescale = pipeline.outputs[rescale.id]
    cached_otsu = pipeline.outputs[otsu.id]
    request = PipelineRunRequest(
        run_id=15,
        workflow=serialize_workflow(pipeline),
        input_data=data,
        input_metadata={"axes": "YX"},
        input_name="source",
        source_payloads={},
        dirty_node_ids=frozenset({psf.id}),
        cached_outputs=dict(pipeline.outputs),
        cached_output_states=dict(pipeline.output_states),
        cached_node_outputs={
            node_id: list(outputs) for node_id, outputs in pipeline.node_outputs.items()
        },
        cached_node_output_states={
            node_id: list(states)
            for node_id, states in pipeline.node_output_states.items()
        },
        completed_node_ids=frozenset(pipeline.completed_node_ids),
        cached_execution_states=dict(pipeline.node_execution_states),
        cached_execution_messages=dict(pipeline.node_execution_messages),
    )
    started: list[str] = []

    result = execute_pipeline_request(request, node_started_callback=started.append)

    assert result.error == ""
    assert result.pipeline is not None
    assert started == ["input", psf.id]
    assert result.pipeline.node_execution_states[deconvolution.id] == EXECUTION_STALE
    assert result.pipeline.node_execution_states[rescale.id] == EXECUTION_BLOCKED
    assert result.pipeline.node_execution_states[otsu.id] == EXECUTION_BLOCKED
    assert result.pipeline.outputs[deconvolution.id] is cached_deconvolution
    assert result.pipeline.outputs[rescale.id] is cached_rescale
    assert result.pipeline.outputs[otsu.id] is cached_otsu


def test_execute_pipeline_request_distinguishes_cooperative_cancellation():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    background = pipeline.add_node("rolling_ball_background")
    assert pipeline.connect("input", background.id).success
    cancel_event = threading.Event()
    cancel_event.set()
    request = PipelineRunRequest(
        run_id=17,
        workflow=serialize_workflow(pipeline),
        input_data=np.arange(16, dtype=np.uint8).reshape(4, 4),
        input_metadata={"axes": "YX"},
        input_name="source",
        source_payloads={},
        cancel_event=cancel_event,
    )

    result = execute_pipeline_request(request)

    assert result.pipeline is None
    assert result.cancelled
    assert "cancel" in result.error.lower()


def test_execute_pipeline_request_forwards_rescale_phase_progress():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    rescale = pipeline.add_node("rescale_intensity")
    assert pipeline.connect("input", rescale.id).success
    updates = []
    request = PipelineRunRequest(
        run_id=19,
        workflow=serialize_workflow(pipeline),
        input_data=np.linspace(0.0, 1.0, 512, dtype=np.float32),
        input_metadata=None,
        input_name="source",
        source_payloads={},
    )

    result = execute_pipeline_request(
        request,
        progress_callback=lambda *update: updates.append(update),
    )

    assert result.error == ""
    assert result.pipeline is not None
    assert updates
    assert all(update[0] == rescale.id for update in updates)
    assert updates[-1][1:3] == (100, 100)
    assert any("cutoff" in update[3].lower() for update in updates)
    assert any("rescal" in update[3].lower() for update in updates)
