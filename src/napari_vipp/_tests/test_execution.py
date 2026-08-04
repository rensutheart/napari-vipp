from __future__ import annotations

import threading

import numpy as np

from napari_vipp.core.compute import (
    ComputeMode,
    ComputeRequest,
    ExecutionFallbackRecord,
)
from napari_vipp.core.execution import (
    PipelineExecutionFailure,
    PipelineNodeResult,
    PipelineRunRequest,
    execute_pipeline_request,
)
from napari_vipp.core.pipeline import (
    EXECUTION_BLOCKED,
    EXECUTION_STALE,
    PrototypePipeline,
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
            node_id: list(outputs)
            for node_id, outputs in pipeline.node_outputs.items()
        },
        "cached_node_output_states": {
            node_id: list(states)
            for node_id, states in pipeline.node_output_states.items()
        },
        "completed_node_ids": frozenset(pipeline.completed_node_ids),
        "cached_execution_states": dict(pipeline.node_execution_states),
        "cached_execution_messages": dict(pipeline.node_execution_messages),
        "cached_compute_provenance": dict(pipeline.node_compute_provenance),
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
            node_id: list(outputs)
            for node_id, outputs in initial.node_outputs.items()
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
    assert gaussian.actual_implementation.implementation_id == (
        "cpu-gaussian_blur-v1"
    )
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
            node_id: list(outputs)
            for node_id, outputs in pipeline.node_outputs.items()
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
            node_id: list(outputs)
            for node_id, outputs in pipeline.node_outputs.items()
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
