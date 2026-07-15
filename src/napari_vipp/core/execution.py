"""Headless contracts and service for one isolated pipeline execution.

The service detaches and validates the graph document. Input ownership is an
upstream source-boundary responsibility: callers must supply stable snapshots,
not mutable viewer arrays or live lazy stores.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass

from napari_vipp.core.pipeline import (
    MANUAL_RUN_SKIP,
    PrototypePipeline,
    SourcePayload,
)
from napari_vipp.core.progress import OperationCancelled
from napari_vipp.core.workflow import deserialize_workflow

NodeStartedCallback = Callable[[str], None]
ProgressCallback = Callable[[str, int, int, str], None]


@dataclass(frozen=True, slots=True)
class PipelineRunRequest:
    """Graph document, stable inputs, caches, and one execution policy."""

    run_id: int
    workflow: dict
    input_data: object
    input_metadata: object
    input_name: str
    source_payloads: dict[str, SourcePayload]
    dirty_node_ids: frozenset[str] | None = None
    cached_outputs: dict[str, object] | None = None
    cached_output_states: dict[str, object] | None = None
    cached_node_outputs: dict[str, list[object]] | None = None
    cached_node_output_states: dict[str, list[object]] | None = None
    completed_node_ids: frozenset[str] = frozenset()
    cached_execution_states: dict[str, str] | None = None
    cached_execution_messages: dict[str, str] | None = None
    manual_node_ids: frozenset[str] | None = None
    retain_node_ids: frozenset[str] = frozenset()
    prune_unretained: bool = False
    cancel_event: threading.Event | None = None
    source_revisions: tuple[object, ...] = ()


@dataclass(frozen=True, slots=True)
class PipelineRunResult:
    """Success, cancellation, or explicit error from one execution attempt."""

    run_id: int
    workflow: dict
    pipeline: PrototypePipeline | None = None
    error: str = ""
    cancelled: bool = False
    source_revisions: tuple[object, ...] = ()


@dataclass(frozen=True, slots=True)
class PipelineNodeResult:
    """One completed node's presentation-safe result from an active run."""

    run_id: int
    node_id: str
    operation_id: str
    output: object
    output_state: object
    node_outputs: tuple[object, ...]
    node_output_states: tuple[object, ...]
    execution_state: str
    execution_message: str = ""
    source_revisions: tuple[object, ...] = ()


NodeFinishedCallback = Callable[[PipelineNodeResult], None]


def execute_pipeline_request(
    request: PipelineRunRequest,
    *,
    node_started_callback: NodeStartedCallback | None = None,
    node_finished_callback: NodeFinishedCallback | None = None,
    progress_callback: ProgressCallback | None = None,
) -> PipelineRunResult:
    """Execute ``request`` without Qt and return errors as typed results."""
    try:
        workflow = deserialize_workflow(deepcopy(request.workflow))
        pipeline = PrototypePipeline()
        pipeline.restore_graph(
            workflow["nodes"],
            workflow["connections"],
            workflow.get("output_tunnels", ()),
        )
        _hydrate_cached_pipeline_outputs(pipeline, request)

        def publish_node_result(node_id: str) -> None:
            if node_finished_callback is None:
                return
            node = pipeline.nodes[node_id]
            node_finished_callback(
                PipelineNodeResult(
                    run_id=request.run_id,
                    node_id=node_id,
                    operation_id=node.operation_id,
                    output=pipeline.outputs.get(node_id),
                    output_state=pipeline.output_states.get(node_id),
                    node_outputs=tuple(pipeline.node_outputs.get(node_id, ())),
                    node_output_states=tuple(
                        pipeline.node_output_states.get(node_id, ())
                    ),
                    execution_state=pipeline.node_execution_states.get(node_id, ""),
                    execution_message=pipeline.node_execution_messages.get(
                        node_id,
                        "",
                    ),
                    source_revisions=request.source_revisions,
                )
            )

        pipeline.run(
            request.input_data,
            input_metadata=request.input_metadata,
            input_name=request.input_name,
            source_payloads=request.source_payloads,
            dirty_node_ids=request.dirty_node_ids,
            node_started_callback=node_started_callback,
            node_finished_callback=publish_node_result,
            progress_callback=progress_callback,
            cancel_callback=(
                request.cancel_event.is_set
                if request.cancel_event is not None
                else None
            ),
            manual_mode=MANUAL_RUN_SKIP,
            manual_node_ids=request.manual_node_ids,
            retain_node_ids=request.retain_node_ids,
            prune_unretained=request.prune_unretained,
        )
    except OperationCancelled as exc:
        return PipelineRunResult(
            request.run_id,
            request.workflow,
            error=str(exc),
            cancelled=True,
            source_revisions=request.source_revisions,
        )
    except Exception as exc:
        return PipelineRunResult(
            request.run_id,
            request.workflow,
            error=str(exc),
            source_revisions=request.source_revisions,
        )
    return PipelineRunResult(
        request.run_id,
        request.workflow,
        pipeline,
        source_revisions=request.source_revisions,
    )


def _hydrate_cached_pipeline_outputs(
    pipeline: PrototypePipeline,
    request: PipelineRunRequest,
) -> None:
    """Restore reusable output state before a dirty-subgraph execution."""
    if request.dirty_node_ids is None:
        return
    if request.cached_outputs is not None:
        pipeline.outputs = dict(request.cached_outputs)
    if request.cached_output_states is not None:
        pipeline.output_states = dict(request.cached_output_states)
    if request.cached_node_outputs is not None:
        pipeline.node_outputs = {
            node_id: list(outputs)
            for node_id, outputs in request.cached_node_outputs.items()
        }
    if request.cached_node_output_states is not None:
        pipeline.node_output_states = {
            node_id: list(states)
            for node_id, states in request.cached_node_output_states.items()
        }
    if request.cached_execution_states is not None:
        pipeline.node_execution_states = dict(request.cached_execution_states)
    if request.cached_execution_messages is not None:
        pipeline.node_execution_messages = dict(request.cached_execution_messages)
    pipeline.completed_node_ids = set(request.completed_node_ids)


__all__ = [
    "NodeFinishedCallback",
    "NodeStartedCallback",
    "PipelineNodeResult",
    "PipelineRunRequest",
    "PipelineRunResult",
    "ProgressCallback",
    "execute_pipeline_request",
]
