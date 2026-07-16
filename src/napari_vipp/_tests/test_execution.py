from __future__ import annotations

import threading

import numpy as np

from napari_vipp.core.execution import (
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
    assert started == [psf.id]
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
