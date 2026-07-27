"""Headless contracts and service for one isolated pipeline execution.

The service detaches and validates the graph document. Input ownership is an
upstream source-boundary responsibility: callers must supply stable snapshots,
not mutable viewer arrays or live lazy stores.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol

import numpy as np

from napari_vipp.core.compute import (
    ComputeMode,
    ComputeRequest,
    DecisionKind,
    DecisionReason,
    ExecutionPlan,
    ExecutionReport,
    FallbackReason,
    NodeExecutionDecision,
    OutputPortKey,
    WorkloadDescriptor,
    canonical_digest,
)
from napari_vipp.core.compute_policy import ArrayFacts, FactCompleteness
from napari_vipp.core.node_execution import PreparedNodeCall
from napari_vipp.core.pipeline import (
    EXECUTION_RUNNING,
    MANUAL_RUN_SKIP,
    PrototypePipeline,
    SourcePayload,
)
from napari_vipp.core.progress import OperationCancelled
from napari_vipp.core.workflow import deserialize_workflow

if TYPE_CHECKING:
    from napari_vipp.core.compute_registry import ComputeRegistry

NodeStartedCallback = Callable[[str], None]
ProgressCallback = Callable[[str, int, int, str], None]


class ComputePlanner(Protocol):
    """Injectable planning seam used only for non-CPU requests."""

    def __call__(
        self,
        request: ComputeRequest,
        workloads: Sequence[WorkloadDescriptor],
        *,
        registry: ComputeRegistry | None = None,
        environment: object | None = None,
        array_facts: Mapping[str, tuple[object, ...]] | None = None,
        performance_evidence: Mapping[tuple[str, str], object] | None = None,
    ) -> object: ...


@dataclass(frozen=True, slots=True)
class _ArrayDescription:
    """Shape/dtype-only value used while planning resident continuations."""

    shape: tuple[int, ...]
    dtype: np.dtype

    @property
    def ndim(self) -> int:
        return len(self.shape)


@dataclass(frozen=True, slots=True)
class PipelineRunRequest:
    """Graph document, stable inputs, caches, and one execution policy."""

    run_id: int
    workflow: dict
    input_data: object
    input_metadata: object
    input_name: str
    source_payloads: dict[str, SourcePayload]
    compute_request: ComputeRequest = field(default_factory=ComputeRequest)
    dirty_node_ids: frozenset[str] | None = None
    cached_outputs: dict[str, object] | None = None
    cached_output_states: dict[str, object] | None = None
    cached_node_outputs: dict[str, list[object]] | None = None
    cached_node_output_states: dict[str, list[object]] | None = None
    completed_node_ids: frozenset[str] = frozenset()
    cached_execution_states: dict[str, str] | None = None
    cached_execution_messages: dict[str, str] | None = None
    manual_node_ids: frozenset[str] | None = None
    target_node_ids: frozenset[str] | None = None
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
    execution_report: ExecutionReport | None = None


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
    compute_registry: ComputeRegistry | None = None,
    compute_planner: ComputePlanner | None = None,
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

        cancel_callback = (
            request.cancel_event.is_set
            if request.cancel_event is not None
            else None
        )
        if request.compute_request.mode is ComputeMode.CPU:
            pipeline.run(
                request.input_data,
                input_metadata=request.input_metadata,
                input_name=request.input_name,
                source_payloads=request.source_payloads,
                dirty_node_ids=request.dirty_node_ids,
                node_started_callback=node_started_callback,
                node_finished_callback=publish_node_result,
                progress_callback=progress_callback,
                cancel_callback=cancel_callback,
                manual_mode=MANUAL_RUN_SKIP,
                manual_node_ids=request.manual_node_ids,
                target_node_ids=request.target_node_ids,
                retain_node_ids=request.retain_node_ids,
                prune_unretained=request.prune_unretained,
            )
            execution_report = None
        else:
            execution_report = _execute_accelerated_pipeline(
                pipeline,
                request,
                node_started_callback=node_started_callback,
                node_finished_callback=publish_node_result,
                progress_callback=progress_callback,
                cancel_callback=cancel_callback,
                compute_registry=compute_registry,
                compute_planner=compute_planner,
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
        execution_report=execution_report,
    )


def _execute_accelerated_pipeline(
    pipeline: PrototypePipeline,
    request: PipelineRunRequest,
    *,
    node_started_callback: NodeStartedCallback | None,
    node_finished_callback: Callable[[str], None] | None,
    progress_callback: ProgressCallback | None,
    cancel_callback: Callable[[], bool] | None,
    compute_registry: ComputeRegistry | None,
    compute_planner: ComputePlanner | None,
) -> ExecutionReport:
    """Plan and atomically commit one non-CPU headless execution."""
    # Accelerator modules remain behind this branch so the default CPU path
    # neither constructs a registry nor imports any provider-facing executor.
    from napari_vipp.core.compute_registry import ComputeRegistry
    from napari_vipp.core.device_execution import (
        CPU_RUNTIME_ID,
        execute_device_plan,
        plan_device_execution,
    )

    owned_registry = compute_registry is None
    registry = ComputeRegistry() if owned_registry else compute_registry
    assert registry is not None
    planner = compute_planner or _default_compute_planner()
    closed_cleanly = True
    try:
        schedule = pipeline.plan_execution(
            request.dirty_node_ids,
            manual_mode=MANUAL_RUN_SKIP,
            manual_node_ids=request.manual_node_ids,
            target_node_ids=request.target_node_ids,
        )
        host_values, state_by_port, source_results = _initial_transaction_values(
            pipeline,
            request,
            schedule.runnable_node_ids,
        )
        workloads, array_facts = _build_workloads(
            pipeline,
            schedule.runnable_node_ids,
            host_values,
            state_by_port,
            registry,
            request.compute_request.allow_experimental,
        )
        planning = planner(
            request.compute_request,
            workloads,
            registry=registry,
            array_facts=array_facts,
        )
        decisions_by_node = _planning_decisions_by_node(planning)

        retained_node_ids = set(request.retain_node_ids)
        if not request.prune_unretained:
            # Preserve the established Keep-all CPU cache contract.  A low-
            # memory/pruned request can retain only exits and selected nodes,
            # allowing a connected device chain to use one H2D and one D2H.
            retained_node_ids.update(schedule.runnable_node_ids)
        retained_ports = tuple(
            OutputPortKey(node_id, port_index)
            for node_id in sorted(retained_node_ids)
            if node_id in pipeline.nodes
            for port_index in range(len(pipeline.output_ports(node_id)))
        )
        device_plan = plan_device_execution(
            pipeline,
            decisions_by_node,
            registry,
            request.compute_request,
            dirty_node_ids=request.dirty_node_ids,
            manual_mode=MANUAL_RUN_SKIP,
            manual_node_ids=request.manual_node_ids,
            target_node_ids=request.target_node_ids,
            retained_ports=retained_ports,
        )
        execution = pipeline.prepare_execution(
            request.dirty_node_ids,
            manual_mode=MANUAL_RUN_SKIP,
            manual_node_ids=request.manual_node_ids,
            target_node_ids=request.target_node_ids,
            retain_node_ids=retained_node_ids,
            prune_unretained=request.prune_unretained,
        )
        if (
            execution.execution_plan.runnable_node_ids
            != schedule.runnable_node_ids
        ):
            raise RuntimeError(
                "Pipeline execution changed between compute planning and commit."
            )

        calls_by_node: dict[str, PreparedNodeCall] = {}
        started_node_ids: set[str] = set()

        def mark_started(node_id: str) -> None:
            if node_id in started_node_ids:
                return
            started_node_ids.add(node_id)
            pipeline.node_execution_states[node_id] = EXECUTION_RUNNING
            pipeline.node_execution_messages[node_id] = ""
            if node_started_callback is not None:
                node_started_callback(node_id)

        def prepare_call(
            node_id: str,
            inputs: tuple[object, ...],
        ) -> PreparedNodeCall:
            mark_started(node_id)
            input_states = tuple(
                state_by_port.get(
                    OutputPortKey(connection.source_id, connection.source_port)
                )
                for connection in pipeline._input_connections(node_id)
            )
            call = pipeline.prepare_node_call(
                node_id,
                inputs,
                input_states,
                progress_callback=progress_callback,
                cancel_callback=cancel_callback,
            )
            if call is None:
                raise RuntimeError(f"Node {node_id!r} could not be prepared.")
            # The transaction keeps only host metadata.  Retaining ``call``
            # directly would retain its opaque device inputs beyond the
            # runtime scope and prevent private-pool cleanup.
            calls_by_node[node_id] = replace(
                call,
                inputs=(None,) * len(call.inputs),
            )
            return call

        def observe_outputs(
            node_id: str,
            call: PreparedNodeCall,
            outputs: tuple[object, ...],
            runtime_id: str,
        ) -> None:
            if runtime_id == CPU_RUNTIME_ID:
                raw_output: object = (
                    outputs[0] if call.output_port_count == 1 else outputs
                )
                results = pipeline.finalize_node_call(call, raw_output)
                states = tuple(state for _value, state in results)
            else:
                states = pipeline.predict_shape_preserving_node_states(call)
            for port_index, state in enumerate(states):
                state_by_port[OutputPortKey(node_id, port_index)] = state

        for node_id in source_results:
            mark_started(node_id)
        device_result = execute_device_plan(
            device_plan,
            pipeline,
            registry,
            request.compute_request,
            host_values=host_values,
            prepare_call=prepare_call,
            cancel_callback=cancel_callback,
            node_outputs_callback=observe_outputs,
        )

        for node_id in pipeline.topological_order():
            if node_id not in execution.remaining_node_ids:
                continue
            mark_started(node_id)
            if node_id in source_results:
                results = source_results[node_id]
            else:
                output_count = len(pipeline.output_ports(node_id))
                ports = tuple(
                    OutputPortKey(node_id, index) for index in range(output_count)
                )
                if not all(port in device_result.host_values for port in ports):
                    pipeline.commit_uncached_node(execution, node_id)
                    if node_finished_callback is not None:
                        node_finished_callback(node_id)
                    continue
                call = calls_by_node[node_id]
                outputs = tuple(device_result.host_values[port] for port in ports)
                raw_output = outputs[0] if output_count == 1 else outputs
                results = pipeline.finalize_node_call(call, raw_output)
            pipeline.commit_node_results(execution, node_id, results)
            if node_finished_callback is not None:
                node_finished_callback(node_id)
        pipeline.finish_execution(execution)

        actual_decisions = _actual_execution_decisions(
            tuple(decisions_by_node.values()),
            device_plan,
            device_result.fallback_segment_ids,
        )
        warnings = list(getattr(planning, "warnings", ()))
        if device_result.fallback_segment_ids:
            warnings.append(
                "Device out-of-memory fallback used for "
                + ", ".join(device_result.fallback_segment_ids)
                + "."
            )
        execution_plan = _planning_execution_plan(
            planning,
            device_plan.segments,
        )
        report = ExecutionReport(
            request=request.compute_request,
            environment=planning.environment,
            plan=execution_plan,
            actual_decisions=actual_decisions,
            warnings=tuple(warnings),
            cleanup_succeeded=device_result.cleanup_succeeded,
        )
    finally:
        if owned_registry:
            try:
                registry.close()
            except Exception:
                closed_cleanly = False
    if not closed_cleanly:
        report = replace(
            report,
            warnings=report.warnings
            + ("The accelerator registry did not close cleanly.",),
            cleanup_succeeded=False,
        )
    return report


def _default_compute_planner() -> ComputePlanner:
    from napari_vipp.core.compute_planning import plan_compute_decisions

    return plan_compute_decisions


def _initial_transaction_values(
    pipeline: PrototypePipeline,
    request: PipelineRunRequest,
    runnable_node_ids: frozenset[str],
) -> tuple[
    dict[OutputPortKey, object],
    dict[OutputPortKey, object],
    dict[str, list[tuple[object, object]]],
]:
    host_values: dict[OutputPortKey, object] = {}
    states: dict[OutputPortKey, object] = {}
    source_results: dict[str, list[tuple[object, object]]] = {}
    for node_id in pipeline.nodes:
        for port_index, value in enumerate(pipeline.node_outputs.get(node_id, ())):
            if value is not None:
                host_values[OutputPortKey(node_id, port_index)] = value
        for port_index, state in enumerate(
            pipeline.node_output_states.get(node_id, ())
        ):
            if state is not None:
                states[OutputPortKey(node_id, port_index)] = state
        if pipeline.node_outputs.get(node_id):
            continue
        value = pipeline.outputs.get(node_id)
        state = pipeline.output_states.get(node_id)
        if value is not None:
            host_values[OutputPortKey(node_id, 0)] = value
        if state is not None:
            states[OutputPortKey(node_id, 0)] = state

    for node_id in pipeline.topological_order():
        if node_id not in runnable_node_ids:
            continue
        operation = pipeline.operation_spec(pipeline.nodes[node_id].operation_id)
        if operation.has_input:
            continue
        results = pipeline.source_node_results(
            node_id,
            request.input_data,
            request.input_metadata,
            request.input_name,
            request.source_payloads,
        )
        source_results[node_id] = results
        for port_index, (value, state) in enumerate(results):
            host_values[OutputPortKey(node_id, port_index)] = value
            states[OutputPortKey(node_id, port_index)] = state
    return host_values, states, source_results


def _build_workloads(
    pipeline: PrototypePipeline,
    runnable_node_ids: frozenset[str],
    host_values: Mapping[OutputPortKey, object],
    initial_states: Mapping[OutputPortKey, object],
    registry: ComputeRegistry,
    allow_experimental: bool,
) -> tuple[
    tuple[WorkloadDescriptor, ...],
    Mapping[str, tuple[ArrayFacts, ...]],
]:
    values: dict[OutputPortKey, object] = dict(host_values)
    states: dict[OutputPortKey, object] = dict(initial_states)
    facts_by_port: dict[OutputPortKey, ArrayFacts] = {}
    facts_by_node: dict[str, tuple[ArrayFacts, ...]] = {}
    scanned_values: dict[int, ArrayFacts] = {}
    runnable = set(runnable_node_ids)
    workloads: list[WorkloadDescriptor] = []
    for node_id in pipeline.topological_order():
        if node_id not in runnable:
            continue
        node = pipeline.nodes[node_id]
        operation = pipeline.operation_spec(node.operation_id)
        connections = pipeline._input_connections(node_id)
        input_shapes: list[tuple[int, ...]] = []
        input_dtypes: list[str] = []
        input_values: list[object] = []
        input_states: list[object] = []
        input_facts: list[ArrayFacts | None] = []
        for connection in connections:
            port = OutputPortKey(connection.source_id, connection.source_port)
            state = states.get(port)
            value = values.get(port)
            shape, dtype = _shape_and_dtype(value, state)
            input_shapes.append(shape)
            input_dtypes.append(dtype)
            input_values.append(
                value
                if value is not None
                else _ArrayDescription(shape, np.dtype(dtype))
            )
            input_states.append(state)
            facts = facts_by_port.get(port)
            if facts is None and isinstance(value, np.ndarray):
                facts = scanned_values.get(id(value))
                if facts is None:
                    facts = _complete_array_facts(
                        value,
                        revision_fingerprint=(
                            f"transaction:{port.node_id}:{port.port_index}:"
                            f"{id(value)}"
                        ),
                    )
                    scanned_values[id(value)] = facts
                facts_by_port[port] = facts
            input_facts.append(facts)

        planning_call: PreparedNodeCall | None = None
        resolved_spatial_ndim: int | None = None
        if operation.has_input and input_values:
            try:
                planning_call = pipeline.prepare_node_call(
                    node_id,
                    tuple(input_values),
                    tuple(input_states),
                )
            except (TypeError, ValueError):
                planning_call = None
            if planning_call is not None:
                raw_spatial_ndim = planning_call.kwargs.get("resolved_spatial_ndim")
                if raw_spatial_ndim is not None:
                    resolved_spatial_ndim = int(raw_spatial_ndim)

        parameters = _workload_parameters(pipeline, node_id, planning_call)
        predecessors = tuple(
            connection.source_id
            for connection in connections
            if connection.source_id in runnable
        )
        successors = tuple(
            connection.target_id
            for connection in pipeline.connections
            if connection.source_id == node_id
            and connection.target_id in runnable
        )
        required_boundaries = int(
            not operation.has_input
            or any(
                connection.source_id not in runnable for connection in connections
            )
        ) + int(not successors)
        facts_fingerprint = ""
        if input_facts and all(facts is not None for facts in input_facts):
            complete_input_facts = tuple(
                facts for facts in input_facts if facts is not None
            )
            facts_by_node[node_id] = complete_input_facts
            facts_fingerprint = _support_facts_fingerprint(complete_input_facts)
        workload = WorkloadDescriptor(
                node_id=node_id,
                operation_id=node.operation_id,
                input_shapes=tuple(input_shapes),
                input_dtypes=tuple(input_dtypes),
                parameters=parameters,
                resolved_spatial_ndim=resolved_spatial_ndim,
                resident_predecessors=predecessors,
                resident_successors=successors,
                required_host_boundaries=required_boundaries,
                facts_fingerprint=facts_fingerprint,
            )
        workloads.append(workload)

        if (
            planning_call is not None
            and _has_shape_preserving_device_implementation(
                registry,
                node.operation_id,
                allow_experimental,
            )
        ):
            predicted_states = pipeline.predict_shape_preserving_node_states(
                planning_call
            )
            for port_index, predicted_state in enumerate(predicted_states):
                port = OutputPortKey(node_id, port_index)
                states[port] = predicted_state
                if input_shapes and input_dtypes:
                    values[port] = _ArrayDescription(
                        input_shapes[0],
                        np.dtype(input_dtypes[0]),
                    )
                if complete_input_facts := facts_by_node.get(node_id):
                    propagated = _propagate_shape_preserving_facts(
                        node.operation_id,
                        complete_input_facts[0],
                        dict(parameters),
                        output_port=port,
                    )
                    if propagated is not None:
                        facts_by_port[port] = propagated
    return tuple(workloads), MappingProxyType(facts_by_node)


def _shape_and_dtype(value: object, state: object) -> tuple[tuple[int, ...], str]:
    raw_shape = getattr(state, "shape", None)
    if raw_shape is None:
        raw_shape = getattr(value, "shape", ())
    try:
        shape = tuple(int(size) for size in raw_shape)
    except (TypeError, ValueError):
        shape = ()
    raw_dtype = getattr(state, "dtype", None)
    if raw_dtype is None:
        raw_dtype = getattr(value, "dtype", "object")
    try:
        dtype = np.dtype(raw_dtype).name
    except TypeError:
        dtype = "object"
    return shape, dtype


def _complete_array_facts(
    value: np.ndarray,
    *,
    revision_fingerprint: str,
) -> ArrayFacts:
    array = np.asarray(value)
    guarantees: list[str] = []
    finite_count: int | None = None
    minimum: float | int | None = None
    maximum: float | int | None = None
    completeness = FactCompleteness.UNKNOWN
    is_real_numeric = (
        np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.floating)
        or array.dtype == np.dtype(bool)
    )
    if is_real_numeric:
        completeness = FactCompleteness.COMPLETE
        negative_zero = False
        if not np.issubdtype(array.dtype, np.floating):
            finite_count = int(array.size)
            if array.size:
                minimum = array.min().item()
                maximum = array.max().item()
        else:
            finite_count = 0
            iterator = np.nditer(
                array,
                flags=["buffered", "external_loop", "zerosize_ok"],
                op_flags=[["readonly"]],
                order="K",
                buffersize=1_048_576,
            )
            for raw_chunk in iterator:
                chunk = np.asarray(raw_chunk)
                finite = np.isfinite(chunk)
                count = int(np.count_nonzero(finite))
                finite_count += count
                if not count:
                    continue
                values = chunk if count == chunk.size else chunk[finite]
                chunk_minimum = values.min().item()
                chunk_maximum = values.max().item()
                minimum = (
                    chunk_minimum
                    if minimum is None
                    else min(minimum, chunk_minimum)
                )
                maximum = (
                    chunk_maximum
                    if maximum is None
                    else max(maximum, chunk_maximum)
                )
                negative_zero = negative_zero or bool(
                    np.any((values == 0) & np.signbit(values))
                )
        if not negative_zero:
            guarantees.append("no-negative-zero")
        if minimum is not None and minimum >= 0:
            guarantees.append("nonnegative")
    return ArrayFacts(
        shape=tuple(int(size) for size in array.shape),
        dtype=array.dtype.name,
        element_count=int(array.size),
        revision_fingerprint=revision_fingerprint,
        completeness=completeness,
        finite_count=finite_count,
        minimum=minimum,
        maximum=maximum,
        strides=tuple(int(stride) for stride in array.strides),
        contiguous=bool(array.flags.c_contiguous),
        guarantees=tuple(guarantees),
    )


def _support_facts_fingerprint(facts: tuple[ArrayFacts, ...]) -> str:
    """Fingerprint only support-relevant regions, not ephemeral array identity."""
    return canonical_digest(
        tuple(
            {
                "shape": item.shape,
                "dtype": item.dtype,
                "completeness": item.completeness.value,
                "all_finite": item.all_finite,
                "label_maximum": item.label_maximum,
                "label_count": item.label_count,
                "foreground_density": item.foreground_density,
                "guarantees": item.guarantees,
            }
            for item in facts
        )
    )


def _propagate_shape_preserving_facts(
    operation_id: str,
    facts: ArrayFacts,
    parameters: Mapping[str, object],
    *,
    output_port: OutputPortKey,
) -> ArrayFacts | None:
    phase_one_operations = {
        "rolling_ball_background",
        "subtract_background",
        "gaussian_blur",
        "gaussian_blur_3d",
        "median_filter",
    }
    if operation_id not in phase_one_operations:
        return None

    guarantees = set(facts.guarantees)
    finite_count = facts.finite_count
    completeness = facts.completeness
    if operation_id == "rolling_ball_background":
        finite_count = facts.element_count
        completeness = FactCompleteness.COMPLETE
        if "nonnegative" in guarantees:
            guarantees.add("no-negative-zero")
    elif operation_id == "subtract_background":
        if bool(parameters.get("clip_negative", True)):
            guarantees.update(("nonnegative", "no-negative-zero"))
    elif operation_id in {"gaussian_blur", "gaussian_blur_3d"}:
        if facts.all_finite is not True:
            finite_count = None
            completeness = FactCompleteness.UNKNOWN
        if "nonnegative" in guarantees:
            guarantees.add("no-negative-zero")
        else:
            guarantees.discard("no-negative-zero")

    return ArrayFacts(
        shape=facts.shape,
        dtype=facts.dtype,
        element_count=facts.element_count,
        revision_fingerprint=(
            f"{facts.revision_fingerprint}>{operation_id}:"
            f"{output_port.port_index}"
        ),
        completeness=completeness,
        finite_count=finite_count,
        guarantees=tuple(sorted(guarantees)),
    )


def _workload_parameters(
    pipeline: PrototypePipeline,
    node_id: str,
    call: PreparedNodeCall | None,
) -> tuple[tuple[str, object], ...]:
    raw = (
        dict(call.kwargs)
        if call is not None
        else pipeline._public_params(pipeline.nodes[node_id].params)
    )
    parameters: list[tuple[str, object]] = []
    for name, value in raw.items():
        if name in {"progress", "image_state"}:
            continue
        try:
            parameters.append((name, _json_contract_value(value)))
        except TypeError:
            continue
    return tuple(parameters)


def _json_contract_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _json_contract_value(item) for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return tuple(_json_contract_value(item) for item in value)
    raise TypeError(f"Unsupported planning parameter {type(value).__name__}.")


def _has_shape_preserving_device_implementation(
    registry: ComputeRegistry,
    operation_id: str,
    allow_experimental: bool,
) -> bool:
    implementations = registry.implementations_for_operation(
        operation_id,
        allow_experimental=allow_experimental,
    )
    return any(
        implementation.supports_device_residency
        and not implementation.host_boundary
        and implementation.shape_policy_id == "shape-preserving-v1"
        and all(
            port.output_dtype_policy_id == "dtype-same-v1"
            for port in implementation.output_ports
        )
        for implementation in implementations
    )


def _planning_decisions_by_node(
    planning: object,
) -> Mapping[str, NodeExecutionDecision]:
    decisions = getattr(planning, "decisions_by_node", None)
    if not isinstance(decisions, Mapping):
        raise TypeError("Compute planning must provide decisions_by_node.")
    if any(
        not isinstance(decision, NodeExecutionDecision)
        for decision in decisions.values()
    ):
        raise TypeError("Compute planning returned an invalid node decision.")
    return decisions


def _planning_execution_plan(
    planning: object,
    segments: tuple[object, ...],
) -> ExecutionPlan:
    factory = getattr(planning, "as_execution_plan", None)
    if callable(factory):
        return factory(segments=segments)
    request = planning.request
    environment = planning.environment
    return ExecutionPlan(
        request.fingerprint,
        environment.fingerprint,
        segments,
        tuple(_planning_decisions_by_node(planning).values()),
        tuple(getattr(planning, "warnings", ())),
    )


def _actual_execution_decisions(
    planned: tuple[NodeExecutionDecision, ...],
    device_plan: object,
    fallback_segment_ids: tuple[str, ...],
) -> tuple[NodeExecutionDecision, ...]:
    fallback_ids = set(fallback_segment_ids)
    fallback_nodes = {
        node_id
        for segment in getattr(device_plan, "segments", ())
        if segment.segment_id in fallback_ids
        for node_id in segment.node_ids
    }
    device_nodes = {
        node_id
        for segment in getattr(device_plan, "segments", ())
        for node_id in segment.node_ids
    }
    host_forced_nodes = {
        decision.node_id
        for decision in planned
        if decision.runtime_id != "cpu-numpy"
        and decision.node_id not in device_nodes
    }
    if not fallback_nodes and not host_forced_nodes:
        return planned
    try:
        from napari_vipp.core.compute_planning import actual_cpu_fallback_decision
    except ImportError:
        actual_cpu_fallback_decision = _local_actual_cpu_fallback_decision
    actual: list[NodeExecutionDecision] = []
    for decision in planned:
        if decision.node_id in fallback_nodes:
            decision = actual_cpu_fallback_decision(
                decision,
                FallbackReason.OUT_OF_MEMORY,
                reason_text=(
                    "The complete device segment retried on the CPU after OOM."
                ),
            )
        elif decision.node_id in host_forced_nodes:
            decision = actual_cpu_fallback_decision(
                decision,
                FallbackReason.WORKLOAD_UNSUPPORTED,
                reason_text=(
                    "Execution used the authoritative CPU implementation at a "
                    "required host boundary."
                ),
            )
        actual.append(decision)
    return tuple(actual)


def _local_actual_cpu_fallback_decision(
    decision: NodeExecutionDecision,
    fallback_reason: FallbackReason,
    *,
    reason_text: str,
) -> NodeExecutionDecision:
    return replace(
        decision,
        runtime_id="cpu-numpy",
        implementation_library_id="cpu",
        implementation_id=f"cpu-{decision.operation_id}-v1",
        decision_kind=DecisionKind.FALLBACK_CPU,
        reason=DecisionReason.OUT_OF_MEMORY_FALLBACK,
        reason_text=reason_text,
        fallback_reason=fallback_reason,
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
    "ComputePlanner",
    "NodeFinishedCallback",
    "NodeStartedCallback",
    "PipelineNodeResult",
    "PipelineRunRequest",
    "PipelineRunResult",
    "ProgressCallback",
    "execute_pipeline_request",
]
