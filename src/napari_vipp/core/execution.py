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
from itertools import count
from numbers import Integral
from time import perf_counter
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol
from weakref import WeakKeyDictionary

import numpy as np

from napari_vipp.core.compute import (
    ComputeEnvironment,
    ComputeMode,
    ComputeRequest,
    DecisionKind,
    DecisionReason,
    ExecutionPlan,
    ExecutionReport,
    FallbackReason,
    NodeExecutionDecision,
    NodePreferenceKind,
    OutputPortKey,
    WorkloadDescriptor,
    canonical_digest,
)
from napari_vipp.core.compute_policy import (
    ArrayFacts,
    ArrayFactsCache,
    ArrayFactsKey,
    FactCompleteness,
    PerformanceEvidence,
    evaluate_auto_performance,
    evaluate_candidate_support,
    evaluate_candidate_workload_support,
)
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
    from napari_vipp.core.compute_specs import OperationComputeSpec

NodeStartedCallback = Callable[[str], None]
ProgressCallback = Callable[[str, int, int, str], None]
_FACT_SCAN_CHUNK_VALUES = 1_048_576
_FACT_TRANSACTION_IDS = count()
_FACT_CACHE_WAIT_SECONDS = 0.05
_FACT_CACHE_COORDINATORS_GUARD = threading.Lock()
_PHASE_ONE_FACT_OPERATIONS = frozenset(
    {
        "rolling_ball_background",
        "subtract_background",
        "gaussian_blur",
        "gaussian_blur_3d",
        "median_filter",
    }
)


@dataclass(slots=True)
class _ArrayFactsFlight:
    """One active fill for an exact cache key."""

    completed: threading.Event = field(default_factory=threading.Event)


@dataclass(slots=True)
class _ArrayFactsCoordinator:
    """Short-lived per-key fills for one externally owned facts cache."""

    lock: threading.RLock = field(default_factory=threading.RLock)
    in_flight: dict[ArrayFactsKey, _ArrayFactsFlight] = field(default_factory=dict)


_FACT_CACHE_COORDINATORS: WeakKeyDictionary[
    ArrayFactsCache,
    _ArrayFactsCoordinator,
] = WeakKeyDictionary()


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
    array_facts_cache: ArrayFactsCache | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    performance_evidence: (
        Mapping[
            tuple[str, str],
            PerformanceEvidence,
        ]
        | None
    ) = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        raw_evidence = self.performance_evidence
        if raw_evidence is None:
            normalized: dict[tuple[str, str], PerformanceEvidence] = {}
        else:
            if not isinstance(raw_evidence, Mapping):
                raise TypeError("performance_evidence must be a mapping or None.")
            normalized = {}
            for raw_key, evidence in raw_evidence.items():
                if not isinstance(raw_key, tuple) or len(raw_key) != 2:
                    raise TypeError(
                        "performance_evidence keys must be "
                        "(node_id, implementation_id) tuples."
                    )
                node_id = str(raw_key[0]).strip()
                implementation_id = str(raw_key[1]).strip()
                if not node_id or not implementation_id:
                    raise ValueError(
                        "performance_evidence key identifiers must not be empty."
                    )
                key = (node_id, implementation_id)
                if key in normalized:
                    raise ValueError(
                        "performance_evidence contains duplicate normalized keys."
                    )
                if not isinstance(evidence, PerformanceEvidence):
                    raise TypeError(
                        "performance_evidence values must be PerformanceEvidence."
                    )
                normalized[key] = evidence
        object.__setattr__(
            self,
            "performance_evidence",
            MappingProxyType(dict(sorted(normalized.items()))),
        )


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
    array_facts_cache: ArrayFactsCache | None = None,
) -> PipelineRunResult:
    """Execute ``request`` without Qt and return errors as typed results."""
    pipeline: PrototypePipeline | None = None
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
            request.cancel_event.is_set if request.cancel_event is not None else None
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
                array_facts_cache=(
                    request.array_facts_cache
                    if array_facts_cache is None
                    else array_facts_cache
                ),
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
            pipeline=pipeline,
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
    array_facts_cache: ArrayFactsCache | None,
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
        execution = pipeline.prepare_execution(
            request.dirty_node_ids,
            manual_mode=MANUAL_RUN_SKIP,
            manual_node_ids=request.manual_node_ids,
            target_node_ids=request.target_node_ids,
            retain_node_ids=request.retain_node_ids,
            prune_unretained=request.prune_unretained,
        )
        if execution.execution_plan.runnable_node_ids != schedule.runnable_node_ids:
            raise RuntimeError(
                "Pipeline execution changed between compute planning and commit."
            )
        host_values, state_by_port, source_results = _initial_transaction_values(
            pipeline,
            request,
            schedule.runnable_node_ids,
        )
        # Source boundaries are authoritative inputs rather than transformed
        # scientific results. Commit them before accelerator planning so a
        # downstream eligibility/axis error can still present the exact source
        # and let the user repair the graph. All operation nodes remain inside
        # the atomic device transaction below.
        for node_id, results in source_results.items():
            if node_id not in execution.remaining_node_ids:
                continue
            pipeline.node_execution_states[node_id] = EXECUTION_RUNNING
            pipeline.node_execution_messages[node_id] = ""
            if node_started_callback is not None:
                node_started_callback(node_id)
            pipeline.commit_node_results(execution, node_id, results)
            if node_finished_callback is not None:
                node_finished_callback(node_id)
        workloads, array_facts, preflight_environment = _build_workloads(
            pipeline,
            schedule.runnable_node_ids,
            host_values,
            state_by_port,
            registry,
            request,
            cancel_callback=cancel_callback,
            array_facts_cache=array_facts_cache,
        )
        planning = planner(
            request.compute_request,
            workloads,
            registry=registry,
            environment=preflight_environment,
            array_facts=array_facts,
            performance_evidence=request.performance_evidence,
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
    request: PipelineRunRequest,
    *,
    cancel_callback: Callable[[], bool] | None,
    array_facts_cache: ArrayFactsCache | None,
) -> tuple[
    tuple[WorkloadDescriptor, ...],
    Mapping[str, tuple[ArrayFacts, ...]],
    ComputeEnvironment,
]:
    """Build workloads after lazily scanning only required concrete inputs."""

    if array_facts_cache is not None and not isinstance(
        array_facts_cache,
        ArrayFactsCache,
    ):
        raise TypeError("array_facts_cache must be an ArrayFactsCache or None.")
    initial_workloads, _initial_facts, fact_lineage = _assemble_workloads(
        pipeline,
        runnable_node_ids,
        host_values,
        initial_states,
        registry,
        request.compute_request.allow_experimental,
        seed_facts_by_port={},
    )
    potential_specs = _potential_accelerator_specs(
        registry,
        request.compute_request,
        initial_workloads,
    )
    _check_fact_scan_cancelled(cancel_callback)
    from napari_vipp.core.compute_planning import probe_compute_environment

    preflight_environment, _probe_warnings = probe_compute_environment(
        registry,
        request.compute_request,
        potential_specs,
    )
    _check_fact_scan_cancelled(cancel_callback)
    required_ports = _required_concrete_fact_ports(
        pipeline,
        runnable_node_ids,
        host_values,
        registry,
        request.compute_request,
        preflight_environment,
        initial_workloads,
        fact_lineage,
        request.performance_evidence,
    )
    if not required_ports:
        return (
            initial_workloads,
            MappingProxyType({}),
            preflight_environment,
        )

    cache = array_facts_cache or ArrayFactsCache()
    transaction_id = next(_FACT_TRANSACTION_IDS)
    scientific_digests: dict[OutputPortKey, str | None] = {}
    facts_by_port: dict[OutputPortKey, ArrayFacts] = {}
    for port in sorted(
        required_ports,
        key=lambda item: (item.node_id, item.port_index),
    ):
        value = host_values.get(port)
        if not isinstance(value, np.ndarray):
            continue
        revision_fingerprint = _array_revision_fingerprint(
            pipeline,
            request,
            port,
            value,
            transaction_id=transaction_id,
            scientific_digests=scientific_digests,
        )
        cache_key = ArrayFactsKey(port, revision_fingerprint)
        facts = _cached_complete_array_facts(
            cache,
            cache_key,
            value,
            cancel_callback=cancel_callback,
        )
        facts_by_port[port] = facts

    workloads, facts_by_node, _lineage = _assemble_workloads(
        pipeline,
        runnable_node_ids,
        host_values,
        initial_states,
        registry,
        request.compute_request.allow_experimental,
        seed_facts_by_port=facts_by_port,
    )
    return workloads, facts_by_node, preflight_environment


def _assemble_workloads(
    pipeline: PrototypePipeline,
    runnable_node_ids: frozenset[str],
    host_values: Mapping[OutputPortKey, object],
    initial_states: Mapping[OutputPortKey, object],
    registry: ComputeRegistry,
    allow_experimental: bool,
    *,
    seed_facts_by_port: Mapping[OutputPortKey, ArrayFacts],
) -> tuple[
    tuple[WorkloadDescriptor, ...],
    Mapping[str, tuple[ArrayFacts, ...]],
    Mapping[OutputPortKey, OutputPortKey],
]:
    values: dict[OutputPortKey, object] = dict(host_values)
    states: dict[OutputPortKey, object] = dict(initial_states)
    facts_by_port: dict[OutputPortKey, ArrayFacts] = dict(seed_facts_by_port)
    facts_by_node: dict[str, tuple[ArrayFacts, ...]] = {}
    fact_lineage: dict[OutputPortKey, OutputPortKey] = {}
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
            if connection.source_id == node_id and connection.target_id in runnable
        )
        required_boundaries = int(
            not operation.has_input
            or any(connection.source_id not in runnable for connection in connections)
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

        projected_output = _project_host_planning_output(
            node.operation_id,
            planning_call,
            input_shapes,
            input_dtypes,
        )
        if projected_output is not None:
            projected_value, projected_state = projected_output
            port = OutputPortKey(node_id, 0)
            values[port] = projected_value
            states[port] = projected_state
        elif planning_call is not None and (
            _has_shape_preserving_device_implementation(
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
                if (
                    node.operation_id in _PHASE_ONE_FACT_OPERATIONS
                    and len(connections) == 1
                ):
                    connection = connections[0]
                    fact_lineage[port] = OutputPortKey(
                        connection.source_id,
                        connection.source_port,
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
    return (
        tuple(workloads),
        MappingProxyType(facts_by_node),
        MappingProxyType(fact_lineage),
    )


def _project_host_planning_output(
    operation_id: str,
    planning_call: PreparedNodeCall | None,
    input_shapes: Sequence[tuple[int, ...]],
    input_dtypes: Sequence[str],
) -> tuple[_ArrayDescription, object | None] | None:
    """Describe deterministic host transforms needed by downstream planning.

    Compute planning happens before any runnable CPU node executes.  A host
    transform that changes rank therefore needs an explicit shape/dtype
    projection when a later node may run on an accelerator.  Keep this list
    deliberately narrow: every projection must be exact and independently
    validated by the authoritative CPU operation at execution time.
    """
    if (
        operation_id != "extract_channel"
        or planning_call is None
        or planning_call.multiple_inputs
        or planning_call.output_port_count != 1
        or len(input_shapes) != 1
        or len(input_dtypes) != 1
    ):
        return None

    input_shape = tuple(input_shapes[0])
    if not input_shape:
        return None
    axis_types = tuple(planning_call.kwargs.get("axis_types", ()))
    axis_names = tuple(planning_call.kwargs.get("axis_names", ()))
    channel_axis = next(
        (
            index
            for index, axis_type in enumerate(axis_types[: len(input_shape)])
            if str(axis_type).strip().casefold() == "channel"
        ),
        None,
    )
    if channel_axis is None:
        channel_axis = next(
            (
                index
                for index, axis_name in enumerate(axis_names[: len(input_shape)])
                if str(axis_name).strip().casefold()
                in {"c", "channel", "rgb", "rgba"}
            ),
            None,
        )
    if channel_axis is None:
        return None

    raw_channel = planning_call.kwargs.get("channel", 0)
    if isinstance(raw_channel, (bool, np.bool_)) or not isinstance(
        raw_channel,
        Integral,
    ):
        return None
    channel = int(raw_channel)
    channel_count = int(input_shape[channel_axis])
    normalized_channel = channel + channel_count if channel < 0 else channel
    if not 0 <= normalized_channel < channel_count:
        return None

    output_shape = (
        input_shape[:channel_axis] + input_shape[channel_axis + 1 :]
    )
    description = _ArrayDescription(output_shape, np.dtype(input_dtypes[0]))
    input_state = planning_call.input_states[0] if planning_call.input_states else None
    projected_state = input_state
    axes = tuple(getattr(input_state, "axes", ()))
    if input_state is not None and len(axes) == len(input_shape):
        channels = tuple(getattr(input_state, "channels", ()))
        metadata_channel = channel + len(channels) if channel < 0 else channel
        selected_channels = (
            (channels[metadata_channel],)
            if 0 <= metadata_channel < len(channels)
            else ()
        )
        projected_state = replace(
            input_state,
            shape=output_shape,
            axes=axes[:channel_axis] + axes[channel_axis + 1 :],
            channels=selected_channels,
            kind="intensity image",
            value_pattern="",
        )
    return description, projected_state


def _required_concrete_fact_ports(
    pipeline: PrototypePipeline,
    runnable_node_ids: frozenset[str],
    host_values: Mapping[OutputPortKey, object],
    registry: ComputeRegistry,
    request: ComputeRequest,
    environment: ComputeEnvironment,
    workloads: tuple[WorkloadDescriptor, ...],
    fact_lineage: Mapping[OutputPortKey, OutputPortKey],
    performance_evidence: Mapping[
        tuple[str, str],
        PerformanceEvidence,
    ],
) -> frozenset[OutputPortKey]:
    required: set[OutputPortKey] = set()
    runnable = set(runnable_node_ids)
    for workload in workloads:
        indexes = _candidate_required_fact_indexes(
            registry,
            request,
            environment,
            workload,
            performance_evidence,
        )
        if not indexes:
            continue
        connections = pipeline._input_connections(workload.node_id)
        for index in indexes:
            if index >= len(connections):
                continue
            connection = connections[index]
            input_port = OutputPortKey(
                connection.source_id,
                connection.source_port,
            )
            concrete = _trace_concrete_fact_port(
                pipeline,
                input_port,
                runnable,
                host_values,
                fact_lineage,
            )
            if concrete is not None:
                required.add(concrete)
    return frozenset(required)


def _potential_accelerator_specs(
    registry: ComputeRegistry,
    request: ComputeRequest,
    workloads: tuple[WorkloadDescriptor, ...],
) -> tuple[OperationComputeSpec, ...]:
    """Collect provider candidates that preflight must probe exactly once."""

    selected: list[OperationComputeSpec] = []
    identities: set[tuple[str, str, str]] = set()
    for workload in workloads:
        for implementation in _candidate_specs_for_workload(
            registry,
            request,
            workload,
        ):
            if not _candidate_statically_matches(implementation, workload):
                continue
            workload_support = evaluate_candidate_workload_support(
                implementation,
                workload,
                array_facts=(),
            )
            if (
                not workload_support.supported
                and not workload_support.requires_complete_facts
            ):
                continue
            identity = (
                implementation.runtime_id,
                implementation.implementation_id,
                implementation.implementation_version,
            )
            if identity in identities:
                continue
            identities.add(identity)
            selected.append(implementation)
    return tuple(selected)


def _candidate_required_fact_indexes(
    registry: ComputeRegistry,
    request: ComputeRequest,
    environment: ComputeEnvironment,
    workload: WorkloadDescriptor,
    performance_evidence: Mapping[
        tuple[str, str],
        PerformanceEvidence,
    ],
) -> frozenset[int]:
    required: set[int] = set()
    for implementation in _candidate_specs_for_workload(
        registry,
        request,
        workload,
    ):
        if not _candidate_statically_matches(implementation, workload):
            continue
        if not _candidate_can_clear_performance_gate(
            request,
            workload,
            implementation,
            performance_evidence,
        ):
            continue
        support = evaluate_candidate_support(
            implementation,
            workload,
            environment,
            allow_experimental=request.allow_experimental,
            array_facts=(),
        )
        if not support.supported and not support.requires_complete_facts:
            continue
        required.update(_implementation_required_fact_indexes(implementation, workload))
    return frozenset(required)


def _candidate_can_clear_performance_gate(
    request: ComputeRequest,
    workload: WorkloadDescriptor,
    implementation: OperationComputeSpec,
    performance_evidence: Mapping[
        tuple[str, str],
        PerformanceEvidence,
    ],
) -> bool:
    preference = request.preference_for(workload.node_id)
    forced = request.mode is ComputeMode.SELECTIVE and preference.kind in {
        NodePreferenceKind.BEST_GPU,
        NodePreferenceKind.LIBRARY,
        NodePreferenceKind.IMPLEMENTATION,
    }
    if forced:
        return True
    evidence = performance_evidence.get(
        (workload.node_id, implementation.implementation_id)
    )
    return evidence is not None and evaluate_auto_performance(evidence).select_candidate


def _candidate_specs_for_workload(
    registry: ComputeRegistry,
    request: ComputeRequest,
    workload: WorkloadDescriptor,
) -> tuple[OperationComputeSpec, ...]:
    if request.mode is ComputeMode.CPU:
        return ()
    preference = request.preference_for(workload.node_id)
    if (
        request.mode is ComputeMode.SELECTIVE
        and preference.kind is NodePreferenceKind.CPU
    ):
        return ()
    implementations = registry.implementations_for_operation(
        workload.operation_id,
        allow_experimental=request.allow_experimental,
    )
    if request.runtime_id:
        implementations = tuple(
            item for item in implementations if item.runtime_id == request.runtime_id
        )
    if (
        request.mode is ComputeMode.SELECTIVE
        and preference.kind is NodePreferenceKind.LIBRARY
    ):
        implementations = tuple(
            item
            for item in implementations
            if item.implementation_library_id == preference.value
        )
    elif (
        request.mode is ComputeMode.SELECTIVE
        and preference.kind is NodePreferenceKind.IMPLEMENTATION
    ):
        implementations = tuple(
            item
            for item in implementations
            if item.implementation_id == preference.value
        )
    return tuple(implementations)


def _candidate_statically_matches(
    implementation: OperationComputeSpec,
    workload: WorkloadDescriptor,
) -> bool:
    if (
        not getattr(implementation, "is_gpu", False)
        or getattr(implementation, "host_boundary", True)
        or not getattr(implementation, "supports_device_residency", False)
    ):
        return False
    input_ports = tuple(getattr(implementation, "input_ports", ()))
    if len(input_ports) != len(workload.input_dtypes):
        return False
    for raw_dtype, port in zip(
        workload.input_dtypes,
        input_ports,
        strict=True,
    ):
        try:
            dtype = np.dtype(raw_dtype).name
        except (TypeError, ValueError):
            return False
        raw_public = tuple(port.public_dtypes)
        if "*" in raw_public:
            continue
        try:
            public = tuple(np.dtype(item).name for item in raw_public)
        except (TypeError, ValueError):
            return False
        if dtype not in public:
            return False
    spatial_ndim = workload.resolved_spatial_ndim
    supported = tuple(getattr(implementation, "supported_spatial_ndims", ()))
    return spatial_ndim is None or spatial_ndim in supported


def _implementation_required_fact_indexes(
    implementation: OperationComputeSpec,
    workload: WorkloadDescriptor,
) -> frozenset[int]:
    required = {
        index
        for index, port in enumerate(getattr(implementation, "input_ports", ()))
        if port.nonfinite_policy_id == "finite-only-v1"
    }
    if (
        workload.operation_id in {"gaussian_blur", "gaussian_blur_3d", "median_filter"}
        and workload.input_dtypes
        and np.dtype(workload.input_dtypes[0]) == np.dtype(np.float32)
    ):
        # These Phase-1 support regions are intrinsically facts-gated even for
        # injected test/plugin declarations that reuse the CPU port contract.
        required.add(0)
    if (
        getattr(implementation, "parameter_policy_id", "") == "median-parameters-v1"
        and workload.input_dtypes
        and np.dtype(workload.input_dtypes[0]) == np.dtype(np.float32)
    ):
        required.add(0)
    for limitation in getattr(implementation, "limitations", ()):
        if "requires-complete" not in limitation:
            continue
        if limitation.startswith("float32-") and (
            not workload.input_dtypes
            or np.dtype(workload.input_dtypes[0]) != np.dtype(np.float32)
        ):
            continue
        required.update(range(len(workload.input_dtypes)))
    return frozenset(required)


def _trace_concrete_fact_port(
    pipeline: PrototypePipeline,
    starting_port: OutputPortKey,
    runnable_node_ids: set[str],
    host_values: Mapping[OutputPortKey, object],
    fact_lineage: Mapping[OutputPortKey, OutputPortKey],
) -> OutputPortKey | None:
    port = starting_port
    seen: set[OutputPortKey] = set()
    while port not in seen:
        seen.add(port)
        predecessor = fact_lineage.get(port)
        if predecessor is not None:
            port = predecessor
            continue
        node = pipeline.nodes.get(port.node_id)
        if node is None:
            return None
        operation = pipeline.operation_spec(node.operation_id)
        concrete_boundary = (
            port.node_id not in runnable_node_ids or not operation.has_input
        )
        if concrete_boundary and isinstance(host_values.get(port), np.ndarray):
            return port
        # A runnable opaque operation will replace any stale cached value and
        # has no safe pre-execution fact propagation theorem.
        return None
    return None


def _array_facts_cache_coordinator(
    cache: ArrayFactsCache,
) -> _ArrayFactsCoordinator:
    with _FACT_CACHE_COORDINATORS_GUARD:
        coordinator = _FACT_CACHE_COORDINATORS.get(cache)
        if coordinator is None:
            coordinator = _ArrayFactsCoordinator()
            _FACT_CACHE_COORDINATORS[cache] = coordinator
        return coordinator


def _cached_complete_array_facts(
    cache: ArrayFactsCache,
    cache_key: ArrayFactsKey,
    value: np.ndarray,
    *,
    cancel_callback: Callable[[], bool] | None = None,
) -> ArrayFacts:
    """Return one complete record while coordinating only its exact key."""

    coordinator = _array_facts_cache_coordinator(cache)
    while True:
        _check_fact_scan_cancelled(cancel_callback)
        with coordinator.lock:
            cached = cache.get(cache_key)
            if cached is not None and _facts_describe_array(cached, value):
                _check_fact_scan_cancelled(cancel_callback)
                return replace(cached, scan_seconds=0.0)
            flight = coordinator.in_flight.get(cache_key)
            owns_fill = flight is None
            if flight is None:
                flight = _ArrayFactsFlight()
                coordinator.in_flight[cache_key] = flight
        if owns_fill:
            break
        _wait_for_array_facts_flight(
            flight,
            cancel_callback=cancel_callback,
        )

    try:
        facts = _complete_array_facts(
            value,
            revision_fingerprint=cache_key.revision_fingerprint,
            cancel_callback=cancel_callback,
        )
        _check_fact_scan_cancelled(cancel_callback)
        with coordinator.lock:
            # The completed record becomes visible in one cache operation.
            # Cancellation or scan failures take the finally path without a put.
            cache.put(cache_key, facts)
        return facts
    finally:
        with coordinator.lock:
            if coordinator.in_flight.get(cache_key) is flight:
                del coordinator.in_flight[cache_key]
            flight.completed.set()


def _wait_for_array_facts_flight(
    flight: _ArrayFactsFlight,
    *,
    cancel_callback: Callable[[], bool] | None,
) -> None:
    while not flight.completed.wait(timeout=_FACT_CACHE_WAIT_SECONDS):
        _check_fact_scan_cancelled(cancel_callback)
    _check_fact_scan_cancelled(cancel_callback)


def _array_revision_fingerprint(
    pipeline: PrototypePipeline,
    request: PipelineRunRequest,
    port: OutputPortKey,
    value: np.ndarray,
    *,
    transaction_id: int,
    scientific_digests: dict[OutputPortKey, str | None],
) -> str:
    scientific = _scientific_output_digest(
        pipeline,
        port,
        scientific_digests,
    )
    source_revisions = _ancestor_source_revision_digest(
        pipeline,
        request,
        port,
    )
    if scientific is not None and source_revisions is not None:
        return canonical_digest(
            {
                "fact_policy_id": "array-facts-v1",
                "scientific_output": scientific,
                "ancestor_source_revisions": source_revisions,
            }
        )
    return f"transaction:{transaction_id}:{port.node_id}:{port.port_index}:{id(value)}"


def _ancestor_source_revision_digest(
    pipeline: PrototypePipeline,
    request: PipelineRunRequest,
    port: OutputPortKey,
) -> str | None:
    """Return a persistent lineage key only when every source is revisioned."""

    revisions: dict[str, str] = {}
    visited: set[str] = set()
    active: set[str] = set()

    def visit(node_id: str) -> bool:
        if node_id in visited:
            return True
        if node_id in active:
            return False
        node = pipeline.nodes.get(node_id)
        if node is None:
            return False
        active.add(node_id)
        try:
            operation = pipeline.operation_spec(node.operation_id)
            if not operation.has_input:
                payload = request.source_payloads.get(node_id)
                if payload is None or payload.revision_token is None:
                    return False
                revision = _canonical_revision_digest(payload.revision_token)
                if revision is None:
                    return False
                revisions[node_id] = revision
            else:
                connections = pipeline._input_connections(node_id)
                if not connections:
                    return False
                if any(not visit(connection.source_id) for connection in connections):
                    return False
        finally:
            active.discard(node_id)
        visited.add(node_id)
        return True

    try:
        complete = visit(port.node_id)
    except RecursionError:
        return None
    if not complete or not revisions:
        return None
    return canonical_digest(
        {
            "source_revisions": tuple(sorted(revisions.items())),
        }
    )


def _canonical_revision_digest(value: object) -> str | None:
    try:
        return canonical_digest({"revision": value})
    except (TypeError, ValueError, OverflowError, RecursionError):
        return None


def _scientific_output_digest(
    pipeline: PrototypePipeline,
    port: OutputPortKey,
    memo: dict[OutputPortKey, str | None],
) -> str | None:
    if port in memo:
        return memo[port]
    node = pipeline.nodes.get(port.node_id)
    if node is None:
        memo[port] = None
        return None
    try:
        parameters = {
            str(name): _json_contract_value(value)
            for name, value in pipeline._public_params(node.params).items()
        }
        inputs = []
        for connection in pipeline._input_connections(node.id):
            source_port = OutputPortKey(
                connection.source_id,
                connection.source_port,
            )
            source_digest = _scientific_output_digest(
                pipeline,
                source_port,
                memo,
            )
            if source_digest is None:
                memo[port] = None
                return None
            inputs.append(
                {
                    "target_port": connection.target_port,
                    "source_output": source_digest,
                }
            )
        digest = canonical_digest(
            {
                "node_id": node.id,
                "operation_id": node.operation_id,
                "parameters": parameters,
                "output_port": port.port_index,
                "inputs": inputs,
            }
        )
    except (TypeError, ValueError, OverflowError):
        digest = None
    memo[port] = digest
    return digest


def _facts_describe_array(facts: ArrayFacts, value: np.ndarray) -> bool:
    array = np.asarray(value)
    return (
        facts.shape == tuple(int(size) for size in array.shape)
        and np.dtype(facts.dtype) == array.dtype
        and facts.element_count == int(array.size)
        and facts.strides == tuple(int(stride) for stride in array.strides)
        and facts.contiguous is bool(array.flags.c_contiguous)
    )


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
    cancel_callback: Callable[[], bool] | None = None,
) -> ArrayFacts:
    array = np.asarray(value)
    started = perf_counter()
    _check_fact_scan_cancelled(cancel_callback)
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
        finite_count = 0
        if array.size:
            iterator = np.nditer(
                array,
                flags=["buffered", "external_loop", "zerosize_ok"],
                op_flags=[["readonly"]],
                order="K",
                buffersize=_FACT_SCAN_CHUNK_VALUES,
            )
            for raw_chunk in iterator:
                _check_fact_scan_cancelled(cancel_callback)
                chunk = np.asarray(raw_chunk)
                if np.issubdtype(array.dtype, np.floating):
                    finite = np.isfinite(chunk)
                    count = int(np.count_nonzero(finite))
                    values = chunk if count == chunk.size else chunk[finite]
                else:
                    count = int(chunk.size)
                    values = chunk
                finite_count += count
                if count:
                    chunk_minimum = values.min().item()
                    chunk_maximum = values.max().item()
                    if array.dtype == np.dtype(bool):
                        chunk_minimum = int(chunk_minimum)
                        chunk_maximum = int(chunk_maximum)
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
                    if np.issubdtype(array.dtype, np.floating):
                        negative_zero = negative_zero or bool(
                            np.any((values == 0) & np.signbit(values))
                        )
                _check_fact_scan_cancelled(cancel_callback)
        if not negative_zero:
            guarantees.append("no-negative-zero")
        if minimum is not None and minimum >= 0:
            guarantees.append("nonnegative")
    _check_fact_scan_cancelled(cancel_callback)
    scan_seconds = max(0.0, float(perf_counter() - started))
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
        scan_seconds=scan_seconds,
    )


def _check_fact_scan_cancelled(
    cancel_callback: Callable[[], bool] | None,
) -> None:
    if cancel_callback is not None and cancel_callback():
        raise OperationCancelled("Operation cancelled during array-fact scanning.")


def _support_facts_fingerprint(facts: tuple[ArrayFacts, ...]) -> str:
    """Fingerprint only support-relevant regions, not ephemeral array identity."""
    return canonical_digest(
        tuple(
            {
                "shape": item.shape,
                "dtype": item.dtype,
                "completeness": item.completeness.value,
                "all_finite": item.all_finite,
                "minimum": item.minimum,
                "maximum": item.maximum,
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
    if operation_id not in _PHASE_ONE_FACT_OPERATIONS:
        return None

    guarantees = set(facts.guarantees)
    finite_count = facts.finite_count
    completeness = facts.completeness
    try:
        output_dtype = np.dtype(facts.dtype)
    except TypeError:
        output_dtype = None
    dtype_proves_finite = output_dtype is not None and (
        output_dtype == np.dtype(bool) or np.issubdtype(output_dtype, np.integer)
    )
    if operation_id in {"rolling_ball_background", "subtract_background"}:
        float_output_proven_finite = (
            output_dtype is not None
            and np.issubdtype(output_dtype, np.floating)
            and _background_float_output_proven_finite(
                operation_id,
                facts,
                parameters,
                output_dtype,
            )
        )
        if dtype_proves_finite or float_output_proven_finite:
            # An integer/bool output cannot encode NaN or infinity, regardless
            # of the internal floating workspace used by background removal.
            # Floating output is admitted only when complete extrema establish
            # that every relevant public/workspace arithmetic bound is finite.
            finite_count = facts.element_count
            completeness = FactCompleteness.COMPLETE
        else:
            # Finite float input alone is not a proof of finite float output:
            # background offset arithmetic can overflow near the dtype limit
            # and subsequently produce NaN.  Until executable output bounds
            # prove otherwise, downstream finite-only candidates must fail
            # closed rather than inheriting the source-array fact.
            finite_count = None
            completeness = FactCompleteness.UNKNOWN

    if operation_id == "rolling_ball_background":
        if "nonnegative" in guarantees:
            guarantees.add("no-negative-zero")
        else:
            # Rolling/interpolation arithmetic may synthesize signed zero even
            # when the source did not contain one.  Retain the guarantee only
            # in the nonnegative region proven by the public operation.
            guarantees.discard("no-negative-zero")
    elif operation_id == "subtract_background":
        if bool(parameters.get("clip_negative", True)):
            guarantees.update(("nonnegative", "no-negative-zero"))
        else:
            guarantees.discard("nonnegative")
            guarantees.discard("no-negative-zero")
    elif operation_id in {"gaussian_blur", "gaussian_blur_3d"}:
        if facts.all_finite is not True:
            finite_count = None
            completeness = FactCompleteness.UNKNOWN
        if "nonnegative" in guarantees:
            guarantees.add("no-negative-zero")
        else:
            guarantees.discard("no-negative-zero")

    # Exact extrema are intentionally not propagated: each operation can
    # change them.  Phase-1 downstream facts gates consume only completeness,
    # finiteness, and signed-zero/nonnegative guarantees; another magnitude-
    # sensitive transform therefore fails closed instead of using false bounds.
    return ArrayFacts(
        shape=facts.shape,
        dtype=facts.dtype,
        element_count=facts.element_count,
        revision_fingerprint=(
            f"{facts.revision_fingerprint}>{operation_id}:{output_port.port_index}"
        ),
        completeness=completeness,
        finite_count=finite_count,
        guarantees=tuple(sorted(guarantees)),
        scan_seconds=facts.scan_seconds,
    )


def _background_float_output_proven_finite(
    operation_id: str,
    facts: ArrayFacts,
    parameters: Mapping[str, object],
    output_dtype: np.dtype,
) -> bool:
    """Prove finite background output from complete extrema and parameters."""

    if (
        facts.completeness is not FactCompleteness.COMPLETE
        or facts.all_finite is not True
    ):
        return False
    if facts.element_count == 0:
        return True
    if facts.minimum is None or facts.maximum is None:
        return False

    low = float(facts.minimum)
    high = float(facts.maximum)
    light_background = bool(parameters.get("light_background", False))
    workspace_dtype = (
        np.dtype(np.float64)
        if output_dtype.itemsize >= np.dtype(np.float64).itemsize
        else np.dtype(np.float32)
    )
    workspace_limit = float(np.finfo(workspace_dtype).max)
    if light_background and abs(low + high) > workspace_limit:
        # Light-background inversion explicitly forms ``low + high`` in the
        # workspace dtype.  A finite source at the dtype extreme can overflow
        # here before the range-preserving rolling-ball operation runs.
        return False
    if operation_id == "subtract_background":
        output_limit = float(np.finfo(output_dtype).max)
        if high - low > output_limit:
            # Background and source values remain within the input extrema,
            # but their public-dtype difference need not be representable.
            return False
    return True


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
        return {str(key): _json_contract_value(item) for key, item in value.items()}
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
        if decision.runtime_id != "cpu-numpy" and decision.node_id not in device_nodes
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
