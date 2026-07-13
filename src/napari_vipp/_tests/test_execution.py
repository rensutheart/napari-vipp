from __future__ import annotations

import threading

import numpy as np

from napari_vipp.core.execution import (
    PipelineRunRequest,
    execute_pipeline_request,
)
from napari_vipp.core.pipeline import PrototypePipeline
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

    result = execute_pipeline_request(
        request,
        node_started_callback=started.append,
    )

    assert result.run_id == 7
    assert result.error == ""
    assert not result.cancelled
    assert result.source_revisions == ("revision-1",)
    assert result.pipeline is not None
    assert started == ["input"]
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
