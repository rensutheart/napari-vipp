"""Port-granular planning and transactional execution for device runtimes.

This module is intentionally Qt-free and provider-neutral.  A runtime owns every
device value from the first host-to-device transfer until the corresponding
segment has either committed host outputs or failed.  Intermediate values never
enter the pipeline cache, and a failed segment cannot partially update the
caller-visible host store.

The integration with :class:`~napari_vipp.core.pipeline.PrototypePipeline` is a
narrow slice: the existing graph planner chooses runnable nodes, while callers
provide the same prepared-call information used by the authoritative CPU path.
No optional accelerator package is imported here.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field, fields, is_dataclass, replace
from types import MappingProxyType
from typing import Protocol

from napari_vipp.core.accelerator_lease import accelerator_lease
from napari_vipp.core.compute import (
    ComputeMode,
    ComputeRequest,
    ExecutionFallbackRecord,
    ExecutionSegment,
    FallbackPolicy,
    MemoryEstimate,
    NodeExecutionDecision,
    OutputPortKey,
)
from napari_vipp.core.compute_registry import (
    ComputeRegistry,
    RuntimeExceptionInfo,
    RuntimeExceptionKind,
    RuntimeProtocol,
)
from napari_vipp.core.compute_specs import OperationComputeSpec, compute_specs_for
from napari_vipp.core.host_finalization import (
    apply_host_finalizer,
    normalize_operation_outputs,
)
from napari_vipp.core.node_execution import (
    DEFAULT_CPU_NODE_EXECUTOR,
    PreparedNodeCall,
)
from napari_vipp.core.pipeline import (
    MANUAL_RUN_CALCULATE,
    GraphConnection,
    PipelineExecutionPlan,
    PrototypePipeline,
)
from napari_vipp.core.progress import OperationCancelled

CPU_RUNTIME_ID = "cpu-numpy"
_WRITER_OPERATION_IDS = frozenset({"save_output", "batch_output"})


class DevicePlanningError(RuntimeError):
    """Raised when decisions cannot form a safe executable graph plan."""


class DeviceMemoryPreflightError(DevicePlanningError):
    """Raised before scientific execution when a segment cannot fit in memory."""

    def __init__(
        self,
        segment_id: str,
        runtime_id: str,
        required_bytes: int,
        available_bytes: int,
    ) -> None:
        self.segment_id = segment_id
        self.runtime_id = runtime_id
        self.required_bytes = required_bytes
        self.available_bytes = available_bytes
        super().__init__(
            f"Device segment {segment_id!r} requires approximately "
            f"{required_bytes} bytes, but runtime {runtime_id!r} has only "
            f"{available_bytes} bytes available under the active limits."
        )


class DeviceExecutionError(RuntimeError):
    """Raised when a device segment fails without an allowed CPU retry."""

    def __init__(
        self,
        segment_id: str,
        failure: RuntimeExceptionInfo,
        *,
        runtime_id: str = "",
        cleanup_succeeded: bool | None = None,
        fallback_records: tuple[ExecutionFallbackRecord, ...] = (),
    ) -> None:
        self.segment_id = segment_id
        self.runtime_id = str(runtime_id).strip()
        self.failure = failure
        self.cleanup_succeeded = cleanup_succeeded
        self.fallback_records = tuple(fallback_records)
        detail = failure.message or failure.reason_code
        super().__init__(
            f"Device segment {segment_id!r} failed "
            f"({failure.kind.value}: {detail})."
        )


class DeviceExecutionCancelled(OperationCancelled):
    """Cancellation after the device executor has completed scoped cleanup."""

    def __init__(
        self,
        message: str,
        *,
        cleanup_succeeded: bool,
        fallback_records: tuple[ExecutionFallbackRecord, ...] = (),
    ) -> None:
        self.cleanup_succeeded = bool(cleanup_succeeded)
        self.fallback_records = tuple(fallback_records)
        super().__init__(message)


class _DetachedRuntimeFailure(RuntimeError):
    """Carry classified provider failure data without its device traceback.

    Provider exceptions may retain adapter frames whose locals still reference
    private-pool arrays.  Raising those exceptions through ``execution_scope``
    makes terminal cleanup observe allocations that are reachable only through
    the traceback.  This wrapper is created only after the scope has cleaned up
    and retains the immutable, provider-neutral classification instead.
    """

    def __init__(self, failure: RuntimeExceptionInfo) -> None:
        self.failure = failure
        super().__init__(failure.reason_code)


@dataclass(frozen=True, slots=True)
class HostExecutionUnit:
    """One source, CPU operation, or explicit host/writer boundary."""

    node_id: str
    source_boundary: bool = False
    writer_boundary: bool = False


@dataclass(frozen=True, slots=True)
class DeviceSegmentUnit:
    """One maximal connected sub-DAG sharing an array runtime/domain."""

    segment: ExecutionSegment
    implementation_specs: tuple[OperationComputeSpec, ...]

    def __post_init__(self) -> None:
        expected = self.segment.node_ids
        actual = tuple(spec.operation_id for spec in self.implementation_specs)
        if len(expected) != len(actual):
            raise ValueError("Every segment node requires one implementation spec.")
        if any(
            spec.runtime_id != self.segment.runtime_id
            for spec in self.implementation_specs
        ):
            raise ValueError("Segment implementations must share its runtime.")
        if any(
            spec.host_boundary or not spec.supports_device_residency
            for spec in self.implementation_specs
        ):
            raise ValueError(
                "Device segments require resident, non-boundary implementations."
            )


type ExecutionUnit = HostExecutionUnit | DeviceSegmentUnit


@dataclass(frozen=True, slots=True)
class DeviceGraphPlan:
    """Runnable graph partitioned into ordered host units and device segments."""

    schedule: PipelineExecutionPlan
    units: tuple[ExecutionUnit, ...]
    decisions: tuple[NodeExecutionDecision, ...]
    retained_ports: tuple[OutputPortKey, ...] = ()

    @property
    def segments(self) -> tuple[ExecutionSegment, ...]:
        return tuple(
            unit.segment
            for unit in self.units
            if isinstance(unit, DeviceSegmentUnit)
        )


@dataclass(frozen=True, slots=True)
class DeviceExecutionResult:
    """Committed host-only values and visible fallback/cleanup provenance."""

    host_values: Mapping[OutputPortKey, object] = field(repr=False)
    fallback_segment_ids: tuple[str, ...] = ()
    fallback_records: tuple[ExecutionFallbackRecord, ...] = ()
    cleanup_succeeded: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "host_values",
            MappingProxyType(dict(self.host_values)),
        )
        object.__setattr__(
            self,
            "fallback_segment_ids",
            tuple(self.fallback_segment_ids),
        )
        object.__setattr__(self, "fallback_records", tuple(self.fallback_records))


class PrepareNodeCall(Protocol):
    """Build a validated call using inputs already ordered by target port."""

    def __call__(
        self,
        node_id: str,
        inputs: tuple[object, ...],
        /,
    ) -> PreparedNodeCall: ...


class NodeOutputsCallback(Protocol):
    """Observe normalized transaction-local outputs before a downstream call."""

    def __call__(
        self,
        node_id: str,
        call: PreparedNodeCall,
        outputs: tuple[object, ...],
        runtime_id: str,
        /,
    ) -> None: ...


@dataclass(slots=True)
class _DeviceValue:
    runtime_id: str
    device_id: str
    opaque_value: object = field(repr=False)
    allocation_identity: Hashable = field(repr=False)
    port: OutputPortKey
    remaining_consumers: int
    persistent: bool = False


@dataclass(slots=True)
class _CleanupStatus:
    succeeded: bool = True


@dataclass(frozen=True, slots=True)
class _PendingHostFinalization:
    """Host-neutral metadata retained until runtime cleanup has completed."""

    node_id: str
    finalizer_ref: str
    call: PreparedNodeCall
    output_ports: tuple[OutputPortKey, ...]


class _SegmentStore:
    """Own runtime values by output port and release allocations exactly once."""

    def __init__(self, runtime: RuntimeProtocol, device_id: str) -> None:
        self.runtime = runtime
        self.device_id = device_id
        self._ports: dict[OutputPortKey, _DeviceValue] = {}
        self._allocation_refcounts: dict[Hashable, int] = {}
        self._allocations: dict[Hashable, object] = {}
        self.cleanup_succeeded = True

    def owns(self, value: object) -> bool:
        try:
            identity = self.runtime.allocation_identity(value)
        except (TypeError, ValueError, RuntimeError):
            return False
        return identity in self._allocations

    def add(
        self,
        port: OutputPortKey,
        value: object,
        *,
        remaining_consumers: int,
        persistent: bool,
    ) -> None:
        if port in self._ports:
            raise DevicePlanningError(f"Device port {port!r} was produced twice.")
        if not self.runtime.is_device_value(value):
            raise TypeError(
                f"Runtime {self.runtime.runtime_id!r} returned a non-device value "
                f"for {port.node_id!r} output {port.port_index}."
            )
        allocation_identity = self.runtime.allocation_identity(value)
        try:
            hash(allocation_identity)
        except TypeError as exc:
            raise TypeError(
                f"Runtime {self.runtime.runtime_id!r} returned an unhashable "
                "allocation identity."
            ) from exc
        wrapped = _DeviceValue(
            self.runtime.runtime_id,
            self.device_id,
            value,
            allocation_identity,
            port,
            remaining_consumers,
            persistent,
        )
        self._ports[port] = wrapped
        self._allocations.setdefault(allocation_identity, value)
        self._allocation_refcounts[allocation_identity] = (
            self._allocation_refcounts.get(allocation_identity, 0) + 1
        )

    def value(self, port: OutputPortKey) -> object:
        try:
            return self._ports[port].opaque_value
        except KeyError as exc:
            raise DevicePlanningError(
                f"Device segment has no live value for {port.node_id!r} "
                f"output {port.port_index}."
            ) from exc

    def consume(self, port: OutputPortKey) -> None:
        try:
            wrapped = self._ports[port]
        except KeyError as exc:
            raise DevicePlanningError(
                f"Device liveness consumed missing port {port!r}."
            ) from exc
        if wrapped.remaining_consumers < 1:
            raise DevicePlanningError(
                f"Device liveness over-consumed port {port!r}."
            )
        wrapped.remaining_consumers -= 1
        if wrapped.remaining_consumers == 0 and not wrapped.persistent:
            self._drop_port(port)

    def release_if_dead(self, port: OutputPortKey) -> None:
        wrapped = self._ports.get(port)
        if (
            wrapped is not None
            and wrapped.remaining_consumers == 0
            and not wrapped.persistent
        ):
            self._drop_port(port)

    def release_all(self) -> bool:
        for port in tuple(self._ports):
            self._drop_port(port)
        return self.cleanup_succeeded

    def _drop_port(self, port: OutputPortKey) -> None:
        wrapped = self._ports.pop(port, None)
        if wrapped is None:
            return
        identity = wrapped.allocation_identity
        remaining = self._allocation_refcounts[identity] - 1
        if remaining:
            self._allocation_refcounts[identity] = remaining
            return
        self._allocation_refcounts.pop(identity, None)
        representative = self._allocations.pop(identity)
        try:
            self.runtime.release(representative)
        except Exception:
            self.cleanup_succeeded = False


def plan_device_execution(
    pipeline: PrototypePipeline,
    decisions: Mapping[str, NodeExecutionDecision],
    registry: ComputeRegistry,
    request: ComputeRequest,
    *,
    dirty_node_ids: Iterable[str] | None = None,
    manual_mode: str = MANUAL_RUN_CALCULATE,
    manual_node_ids: Iterable[str] | None = None,
    target_node_ids: Iterable[str] | None = None,
    retained_ports: Iterable[OutputPortKey] = (),
) -> DeviceGraphPlan:
    """Partition the existing runnable plan at host/device output boundaries.

    Compatible directly-connected device nodes form maximal components.  A
    source, writer, selected CPU node, unsupported-residency declaration, or
    runtime/domain change creates a host boundary.
    """

    if not isinstance(decisions, Mapping):
        raise TypeError("decisions must be a mapping keyed by node ID.")
    schedule = pipeline.plan_execution(
        dirty_node_ids,
        manual_mode=manual_mode,
        manual_node_ids=manual_node_ids,
        target_node_ids=target_node_ids,
    )
    runnable = set(schedule.runnable_node_ids)
    graph_order = pipeline.topological_order()
    order = tuple(node_id for node_id in graph_order if node_id in runnable)
    order_index = {node_id: index for index, node_id in enumerate(graph_order)}
    retained = _validated_retained_ports(pipeline, retained_ports)

    eligible: dict[
        str,
        tuple[
            NodeExecutionDecision,
            OperationComputeSpec,
            tuple[str, str, str, str],
        ],
    ] = {}
    host_units: dict[str, HostExecutionUnit] = {}
    for node_id in order:
        node = pipeline.nodes[node_id]
        operation = pipeline.operation_spec(node.operation_id)
        cpu_declaration = compute_specs_for(node.operation_id)[0]
        source_boundary = not operation.has_input
        writer_boundary = (
            node.operation_id in _WRITER_OPERATION_IDS
            or cpu_declaration.side_effect_policy_id == "host-writer-v1"
        )
        decision = decisions.get(node_id)
        force_host = (
            request.mode is ComputeMode.CPU
            or source_boundary
            or writer_boundary
            or cpu_declaration.host_boundary
            or decision is None
            or decision.runtime_id == CPU_RUNTIME_ID
        )
        if force_host:
            host_units[node_id] = HostExecutionUnit(
                node_id,
                source_boundary=source_boundary,
                writer_boundary=writer_boundary,
            )
            continue
        try:
            implementation = registry.implementation_spec(
                decision.implementation_id,
                allow_experimental=request.allow_experimental,
            )
        except (KeyError, ValueError) as exc:
            raise DevicePlanningError(
                f"Node {node_id!r} selected unknown implementation "
                f"{decision.implementation_id!r}."
            ) from exc
        mismatches = []
        if implementation.operation_id != node.operation_id:
            mismatches.append("operation")
        if implementation.runtime_id != decision.runtime_id:
            mismatches.append("runtime")
        if (
            implementation.implementation_library_id
            != decision.implementation_library_id
        ):
            mismatches.append("implementation library")
        if mismatches:
            raise DevicePlanningError(
                f"Node {node_id!r} decision disagrees with its registered "
                f"implementation ({', '.join(mismatches)})."
            )
        if implementation.host_boundary or not implementation.supports_device_residency:
            host_units[node_id] = HostExecutionUnit(
                node_id,
                source_boundary=source_boundary,
                writer_boundary=writer_boundary,
            )
            continue
        key = (
            implementation.runtime_id,
            request.device_id,
            implementation.array_domain,
            implementation.implementation_library_id,
        )
        eligible[node_id] = (decision, implementation, key)

    components = _device_components(
        order,
        pipeline.connections,
        eligible,
        registry,
    )
    component_for_node: dict[str, int] = {}
    for index, component in enumerate(components):
        for node_id in component:
            component_for_node[node_id] = index

    device_units: dict[int, DeviceSegmentUnit] = {}
    for index, component in enumerate(components, start=1):
        segment_index = index - 1
        specs = tuple(eligible[node_id][1] for node_id in component)
        runtime_id = specs[0].runtime_id
        component_set = set(component)
        entry_ports = {
            OutputPortKey(connection.source_id, connection.source_port)
            for connection in pipeline.connections
            if connection.target_id in component_set
            and connection.source_id not in component_set
        }
        exit_ports = {
            OutputPortKey(connection.source_id, connection.source_port)
            for connection in pipeline.connections
            if connection.source_id in component_set
            and connection.target_id not in component_set
        }
        outgoing_nodes = {connection.source_id for connection in pipeline.connections}
        for node_id in component:
            if node_id not in outgoing_nodes:
                exit_ports.update(
                    OutputPortKey(node_id, port_index)
                    for port_index in range(len(pipeline.output_ports(node_id)))
                )
        for tunnel in pipeline.output_tunnels.values():
            if tunnel.source_id in component_set:
                exit_ports.add(OutputPortKey(tunnel.source_id, tunnel.source_port))
        component_retained = {
            port for port in retained if port.node_id in component_set
        }
        exit_ports.update(component_retained)

        consumer_counts: Counter[OutputPortKey] = Counter()
        for connection in pipeline.connections:
            if connection.target_id in component_set:
                source_port = OutputPortKey(
                    connection.source_id,
                    connection.source_port,
                )
                if (
                    connection.source_id in component_set
                    or source_port in entry_ports
                ):
                    consumer_counts[source_port] += 1

        decisions_for_component = tuple(eligible[node_id][0] for node_id in component)
        segment = ExecutionSegment(
            segment_id=f"device-segment-{index:03d}",
            runtime_id=runtime_id,
            node_ids=component,
            entry_ports=_sorted_ports(entry_ports, order_index),
            exit_ports=_sorted_ports(exit_ports, order_index),
            retained_ports=_sorted_ports(component_retained, order_index),
            remaining_consumers=tuple(
                (port, consumer_counts[port])
                for port in _sorted_ports(consumer_counts, order_index)
            ),
            memory_estimate=_sum_memory_estimates(
                decision.memory_estimate for decision in decisions_for_component
            ),
        )
        device_units[segment_index] = DeviceSegmentUnit(segment, specs)

    unit_keys: list[tuple[str, object]] = []
    node_to_unit: dict[str, tuple[str, object]] = {}
    for node_id in order:
        if node_id in eligible:
            key: tuple[str, object] = ("device", component_for_node[node_id])
        else:
            key = ("host", node_id)
        node_to_unit[node_id] = key
        if key not in unit_keys:
            unit_keys.append(key)
    dependencies: dict[tuple[str, object], set[tuple[str, object]]] = {
        key: set() for key in unit_keys
    }
    successors: dict[tuple[str, object], set[tuple[str, object]]] = {
        key: set() for key in unit_keys
    }
    for connection in pipeline.connections:
        source_unit = node_to_unit.get(connection.source_id)
        target_unit = node_to_unit.get(connection.target_id)
        if source_unit is None or target_unit is None or source_unit == target_unit:
            continue
        dependencies[target_unit].add(source_unit)
        successors[source_unit].add(target_unit)

    unit_rank = {
        key: min(
            order_index[node_id]
            for node_id, node_key in node_to_unit.items()
            if node_key == key
        )
        for key in unit_keys
    }
    ordered_keys = _stable_unit_topological_order(
        unit_keys,
        dependencies,
        successors,
        unit_rank,
    )
    units: list[ExecutionUnit] = []
    for kind, identifier in ordered_keys:
        if kind == "host":
            units.append(host_units[str(identifier)])
        else:
            units.append(device_units[int(identifier)])

    selected_decisions = tuple(
        decisions[node_id] for node_id in order if node_id in decisions
    )
    return DeviceGraphPlan(schedule, tuple(units), selected_decisions, retained)


def preflight_device_execution(
    plan: DeviceGraphPlan,
    registry: ComputeRegistry,
    request: ComputeRequest,
) -> None:
    """Reject impossible memory requests before any scientific node runs."""

    for unit in plan.units:
        if not isinstance(unit, DeviceSegmentUnit):
            continue
        segment = unit.segment
        runtime = registry.runtime(segment.runtime_id)
        try:
            snapshot = runtime.memory_snapshot(device_id=request.device_id)
        except Exception as exc:
            raise DevicePlanningError(
                f"Could not inspect memory for runtime {segment.runtime_id!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        required = (
            segment.memory_estimate.total_device_peak_bytes
            + segment.memory_estimate.uncertainty_bytes
        )
        available_candidates: list[int] = []
        if snapshot.device_free_bytes is not None:
            reserve = request.accelerator_safety_reserve_bytes or 0
            available_candidates.append(max(0, snapshot.device_free_bytes - reserve))
        if request.accelerator_memory_cap_bytes is not None:
            available_candidates.append(request.accelerator_memory_cap_bytes)
        if available_candidates:
            available = min(available_candidates)
            if required > available:
                raise DeviceMemoryPreflightError(
                    segment.segment_id,
                    segment.runtime_id,
                    required,
                    available,
                )


def execute_device_plan(
    plan: DeviceGraphPlan,
    pipeline: PrototypePipeline,
    registry: ComputeRegistry,
    request: ComputeRequest,
    *,
    host_values: Mapping[OutputPortKey, object],
    prepare_call: PrepareNodeCall,
    cancel_callback: Callable[[], bool] | None = None,
    node_outputs_callback: NodeOutputsCallback | None = None,
) -> DeviceExecutionResult:
    """Execute one plan while exclusively owning all of its accelerators."""

    runtime_ids = sorted(
        {
            unit.segment.runtime_id
            for unit in plan.units
            if isinstance(unit, DeviceSegmentUnit)
        }
    )
    if not runtime_ids:
        return _execute_device_plan_under_leases(
            plan,
            pipeline,
            registry,
            request,
            host_values=host_values,
            prepare_call=prepare_call,
            cancel_callback=cancel_callback,
            node_outputs_callback=node_outputs_callback,
        )
    try:
        _check_cancelled(cancel_callback)
        with ExitStack() as leases:
            for runtime_id in runtime_ids:
                runtime = registry.runtime(runtime_id)
                device_id = _accelerator_lease_device_id(runtime, request.device_id)
                leases.enter_context(
                    accelerator_lease(
                        runtime_id,
                        device_id,
                        cancelled=cancel_callback,
                    )
                )
            return _execute_device_plan_under_leases(
                plan,
                pipeline,
                registry,
                request,
                host_values=host_values,
                prepare_call=prepare_call,
                cancel_callback=cancel_callback,
                node_outputs_callback=node_outputs_callback,
            )
    except DeviceExecutionCancelled:
        raise
    except OperationCancelled as exc:
        raise DeviceExecutionCancelled(
            str(exc),
            cleanup_succeeded=True,
        ) from None


def _execute_device_plan_under_leases(
    plan: DeviceGraphPlan,
    pipeline: PrototypePipeline,
    registry: ComputeRegistry,
    request: ComputeRequest,
    *,
    host_values: Mapping[OutputPortKey, object],
    prepare_call: PrepareNodeCall,
    cancel_callback: Callable[[], bool] | None = None,
    node_outputs_callback: NodeOutputsCallback | None = None,
) -> DeviceExecutionResult:
    """Execute ``plan`` and return an atomic host-only result mapping.

    Device segments commit only their declared exits/retained ports.  A typed,
    retryable out-of-memory failure may retry that complete segment once on CPU
    under the visible fallback policy; no other failure is retried.
    """

    if not isinstance(host_values, Mapping):
        raise TypeError("host_values must be a mapping keyed by OutputPortKey.")
    if not callable(prepare_call):
        raise TypeError("prepare_call must be callable.")
    committed: dict[OutputPortKey, object] = {}
    for port, value in host_values.items():
        if not isinstance(port, OutputPortKey):
            raise TypeError("host_values keys must be OutputPortKey values.")
        committed[port] = value

    # This must happen before sources, CPU nodes, and especially writers.
    preflight_device_execution(plan, registry, request)
    runtimes = {
        unit.segment.runtime_id: registry.runtime(unit.segment.runtime_id)
        for unit in plan.units
        if isinstance(unit, DeviceSegmentUnit)
    }
    _ensure_host_only(committed, tuple(runtimes.values()))

    fallback_segments: list[str] = []
    fallback_records: list[ExecutionFallbackRecord] = []
    cleanup_succeeded = True
    for unit in plan.units:
        try:
            _check_cancelled(cancel_callback)
        except OperationCancelled as exc:
            raise DeviceExecutionCancelled(
                str(exc),
                cleanup_succeeded=cleanup_succeeded,
                fallback_records=tuple(fallback_records),
            ) from None
        if isinstance(unit, HostExecutionUnit):
            try:
                _execute_host_unit(
                    unit,
                    pipeline,
                    committed,
                    prepare_call,
                    cancel_callback,
                    node_outputs_callback,
                )
            except OperationCancelled as exc:
                raise DeviceExecutionCancelled(
                    str(exc),
                    cleanup_succeeded=cleanup_succeeded,
                    fallback_records=tuple(fallback_records),
                ) from None
            except Exception as exc:
                _attach_fallback_records(exc, fallback_records)
                raise
            _ensure_host_only(committed, tuple(runtimes.values()))
            continue

        runtime = runtimes[unit.segment.runtime_id]
        segment_cleanup = _CleanupStatus()
        failure: RuntimeExceptionInfo | None = None
        failure_cause: BaseException | None = None
        try:
            provisional = _execute_device_segment(
                unit,
                pipeline,
                registry,
                runtime,
                request,
                committed,
                prepare_call,
                cancel_callback,
                segment_cleanup,
                node_outputs_callback,
            )
        except OperationCancelled as exc:
            cleanup_succeeded = cleanup_succeeded and segment_cleanup.succeeded
            raise DeviceExecutionCancelled(
                str(exc),
                cleanup_succeeded=cleanup_succeeded,
                fallback_records=tuple(fallback_records),
            ) from None
        except _DetachedRuntimeFailure as exc:
            failure = exc.failure
        except Exception as exc:
            failure = _classify_runtime_exception(runtime, exc)
            failure_cause = exc
        finally:
            cleanup_succeeded = (
                cleanup_succeeded and segment_cleanup.succeeded
            )
        if failure is not None:
            if (
                failure.kind is RuntimeExceptionKind.OUT_OF_MEMORY
                and failure.retryable
                and request.fallback_policy is FallbackPolicy.VISIBLE
            ):
                _best_effort_synchronize(runtime, request.device_id)
                snapshot = _best_effort_memory_snapshot(runtime, request.device_id)
                fallback_attempt = ExecutionFallbackRecord(
                    segment_id=unit.segment.segment_id,
                    runtime_id=unit.segment.runtime_id,
                    node_ids=unit.segment.node_ids,
                    reason="out_of_memory",
                    reason_code=failure.reason_code,
                    exception_type=failure.exception_type,
                    message=failure.message,
                    retryable=failure.retryable,
                    cleanup_succeeded=segment_cleanup.succeeded,
                    memory_estimate=unit.segment.memory_estimate,
                    memory_topology=(
                        snapshot.topology if snapshot is not None else None
                    ),
                    device_total_bytes=(
                        snapshot.device_total_bytes if snapshot is not None else None
                    ),
                    device_free_bytes=(
                        snapshot.device_free_bytes if snapshot is not None else None
                    ),
                    runtime_live_bytes=(
                        snapshot.runtime_live_bytes if snapshot is not None else 0
                    ),
                    runtime_reserved_bytes=(
                        snapshot.runtime_reserved_bytes if snapshot is not None else 0
                    ),
                    out_of_pool_bytes=(
                        snapshot.out_of_pool_bytes if snapshot is not None else 0
                    ),
                )
                try:
                    provisional = _execute_cpu_segment_fallback(
                        unit,
                        pipeline,
                        runtime,
                        committed,
                        prepare_call,
                        cancel_callback,
                        node_outputs_callback,
                    )
                except OperationCancelled as exc:
                    fallback_attempt = replace(
                        fallback_attempt,
                        cpu_retry_succeeded=False,
                    )
                    fallback_records.append(fallback_attempt)
                    raise DeviceExecutionCancelled(
                        str(exc),
                        cleanup_succeeded=cleanup_succeeded,
                        fallback_records=tuple(fallback_records),
                    ) from None
                except Exception as exc:
                    fallback_attempt = replace(
                        fallback_attempt,
                        cpu_retry_succeeded=False,
                    )
                    fallback_records.append(fallback_attempt)
                    _attach_fallback_records(exc, fallback_records)
                    raise
                fallback_segments.append(unit.segment.segment_id)
                fallback_records.append(
                    replace(fallback_attempt, cpu_retry_succeeded=True)
                )
            else:
                error = DeviceExecutionError(
                    unit.segment.segment_id,
                    failure,
                    runtime_id=unit.segment.runtime_id,
                    cleanup_succeeded=cleanup_succeeded,
                    fallback_records=tuple(fallback_records),
                )
                if failure_cause is None:
                    raise error from None
                raise error from failure_cause
        committed.update(provisional)

    _ensure_host_only(committed, tuple(runtimes.values()))
    return DeviceExecutionResult(
        host_values=committed,
        fallback_segment_ids=tuple(fallback_segments),
        fallback_records=tuple(fallback_records),
        cleanup_succeeded=cleanup_succeeded,
    )


def _execute_host_unit(
    unit: HostExecutionUnit,
    pipeline: PrototypePipeline,
    committed: dict[OutputPortKey, object],
    prepare_call: PrepareNodeCall,
    cancel_callback: Callable[[], bool] | None,
    node_outputs_callback: NodeOutputsCallback | None,
) -> None:
    node_id = unit.node_id
    output_count = len(pipeline.output_ports(node_id))
    if unit.source_boundary:
        missing = [
            OutputPortKey(node_id, index)
            for index in range(output_count)
            if OutputPortKey(node_id, index) not in committed
        ]
        if missing:
            raise DevicePlanningError(
                f"Source node {node_id!r} requires caller-provided host values "
                f"for {missing!r}."
            )
        return

    inputs, _ports = _host_inputs_for_node(pipeline, node_id, committed)
    call = prepare_call(node_id, inputs)
    _validate_prepared_call(
        call,
        node_id,
        pipeline.nodes[node_id].operation_id,
        output_count,
    )
    _check_cancelled(cancel_callback)
    raw = DEFAULT_CPU_NODE_EXECUTOR.execute(call)
    outputs = _normalized_outputs(raw, call.output_port_count)
    _check_cancelled(cancel_callback)
    if node_outputs_callback is not None:
        node_outputs_callback(node_id, call, outputs, CPU_RUNTIME_ID)
    provisional = {
        OutputPortKey(node_id, index): value
        for index, value in enumerate(outputs)
    }
    committed.update(provisional)


def _execute_device_segment(
    unit: DeviceSegmentUnit,
    pipeline: PrototypePipeline,
    registry: ComputeRegistry,
    runtime: RuntimeProtocol,
    request: ComputeRequest,
    committed: Mapping[OutputPortKey, object],
    prepare_call: PrepareNodeCall,
    cancel_callback: Callable[[], bool] | None,
    cleanup_status: _CleanupStatus,
    node_outputs_callback: NodeOutputsCallback | None,
) -> dict[OutputPortKey, object]:
    lease_device_id = _accelerator_lease_device_id(runtime, request.device_id)
    with accelerator_lease(
        unit.segment.runtime_id,
        lease_device_id,
        cancelled=cancel_callback,
    ):
        return _execute_device_segment_under_lease(
            unit,
            pipeline,
            registry,
            runtime,
            request,
            committed,
            prepare_call,
            cancel_callback,
            cleanup_status,
            node_outputs_callback,
        )


def _execute_device_segment_under_lease(
    unit: DeviceSegmentUnit,
    pipeline: PrototypePipeline,
    registry: ComputeRegistry,
    runtime: RuntimeProtocol,
    request: ComputeRequest,
    committed: Mapping[OutputPortKey, object],
    prepare_call: PrepareNodeCall,
    cancel_callback: Callable[[], bool] | None,
    cleanup_status: _CleanupStatus,
    node_outputs_callback: NodeOutputsCallback | None,
) -> dict[OutputPortKey, object]:
    segment = unit.segment
    counts = dict(segment.remaining_consumers)
    finalizer_output_ports = tuple(
        OutputPortKey(node_id, port_index)
        for node_id, implementation in zip(
            segment.node_ids,
            unit.implementation_specs,
            strict=True,
        )
        if _host_finalizer_ref(implementation)
        for port_index in range(len(pipeline.output_ports(node_id)))
    )
    persistent = (
        set(segment.exit_ports)
        | set(segment.retained_ports)
        | set(finalizer_output_ports)
    )
    store = _SegmentStore(runtime, request.device_id)
    materialized: dict[OutputPortKey, object] = {}
    pending_finalizations: list[_PendingHostFinalization] = []
    detached_failure: RuntimeExceptionInfo | None = None
    with runtime.execution_scope(
        device_id=request.device_id,
        memory_limit_bytes=request.accelerator_memory_cap_bytes,
        safety_reserve_bytes=request.accelerator_safety_reserve_bytes,
    ):
        try:
            try:
                for port in segment.entry_ports:
                    _check_cancelled(cancel_callback)
                    try:
                        host_value = committed[port]
                    except KeyError as exc:
                        raise DevicePlanningError(
                            f"Segment {segment.segment_id!r} is missing host entry "
                            f"{port.node_id!r} output {port.port_index}."
                        ) from exc
                    device_value = runtime.to_device(
                        host_value,
                        device_id=request.device_id,
                    )
                    try:
                        store.add(
                            port,
                            device_value,
                            remaining_consumers=counts.get(port, 0),
                            persistent=False,
                        )
                    except Exception:
                        if runtime.is_device_value(device_value) and not store.owns(
                            device_value
                        ):
                            _release_quietly(runtime, device_value)
                        raise

                for node_id, implementation in zip(
                    segment.node_ids,
                    unit.implementation_specs,
                    strict=True,
                ):
                    _check_cancelled(cancel_callback)
                    connections = pipeline._input_connections(node_id)
                    input_ports = tuple(
                        OutputPortKey(
                            connection.source_id,
                            connection.source_port,
                        )
                        for connection in connections
                    )
                    inputs = tuple(store.value(port) for port in input_ports)
                    call = prepare_call(node_id, inputs)
                    _validate_prepared_call(
                        call,
                        node_id,
                        pipeline.nodes[node_id].operation_id,
                        len(pipeline.output_ports(node_id)),
                    )
                    # Resolution remains lazy and occurs only after preflight and
                    # after the segment's runtime scope has been entered.
                    implementation_callable = registry.implementation_callable(
                        implementation,
                        allow_experimental=request.allow_experimental,
                    )
                    raw = implementation_callable(
                        call.positional_input(),
                        **_provider_keyword_arguments(call),
                    )
                    try:
                        outputs = _normalized_outputs(raw, call.output_port_count)
                    except Exception:
                        _release_orphan_outputs(runtime, raw, store)
                        raise
                    invalid = tuple(
                        value
                        for value in outputs
                        if not runtime.is_device_value(value)
                    )
                    if invalid:
                        _release_orphan_outputs(runtime, outputs, store)
                        raise TypeError(
                            f"Implementation {implementation.implementation_id!r} "
                            "returned a host value inside a device segment."
                        )
                    try:
                        for index, value in enumerate(outputs):
                            port = OutputPortKey(node_id, index)
                            store.add(
                                port,
                                value,
                                remaining_consumers=counts.get(port, 0),
                                persistent=port in persistent,
                            )
                    except Exception:
                        # Earlier outputs may already be owned by ``store`` while
                        # the value which failed registration is still orphaned.
                        # Release only the latter; the transactional ``finally``
                        # block owns cleanup of every successfully registered one.
                        _release_orphan_outputs(runtime, outputs, store)
                        raise
                    host_finalizer_ref = _host_finalizer_ref(implementation)
                    if host_finalizer_ref:
                        # The prepared call's inputs are opaque runtime values.
                        # Preserve only host metadata for the post-cleanup
                        # finalizer/callback phase.
                        pending_finalizations.append(
                            _PendingHostFinalization(
                                node_id,
                                host_finalizer_ref,
                                replace(
                                    call,
                                    inputs=(None,) * len(call.inputs),
                                ),
                                tuple(
                                    OutputPortKey(node_id, index)
                                    for index in range(len(outputs))
                                ),
                            )
                        )
                    elif node_outputs_callback is not None:
                        node_outputs_callback(
                            node_id,
                            call,
                            outputs,
                            segment.runtime_id,
                        )
                    for port in input_ports:
                        store.consume(port)
                    for index in range(len(outputs)):
                        store.release_if_dead(OutputPortKey(node_id, index))

                for port in _unique_ports(
                    (
                        *segment.exit_ports,
                        *segment.retained_ports,
                        *finalizer_output_ports,
                    )
                ):
                    _check_cancelled(cancel_callback)
                    materialized[port] = runtime.to_host(store.value(port))
                _check_cancelled(cancel_callback)
                runtime.synchronize(device_id=request.device_id)
            except OperationCancelled:
                raise
            except Exception as exc:
                # Classify while provider types are available, then suppress
                # the provider exception before leaving the private allocator
                # scope.  Once this handler ends, its traceback -- including
                # adapter-local scratch arrays -- is no longer retained.
                detached_failure = _classify_runtime_exception(runtime, exc)
        finally:
            # Release while the runtime's private allocator/device scope still
            # owns these arrays.  Releasing after ``__exit__`` can strand pool
            # allocations and violates runtimes that enforce scoped ownership.
            cleanup_status.succeeded = (
                cleanup_status.succeeded and store.release_all()
            )
            # ``release`` relinquishes VIPP's ownership; it must not forcibly
            # recycle storage while a Python alias can still reach it.  Clear
            # this frame's transient references before the runtime validates
            # that no private allocation escaped the scope.  Assignments are
            # intentionally unconditional because any preceding step may have
            # raised before all of these locals were bound.
            device_value = None
            inputs = ()
            call = None
            raw = None
            outputs = ()
            invalid = ()
            value = None

    if detached_failure is not None:
        raise _DetachedRuntimeFailure(detached_failure) from None
    # Finalizers are deliberately outside the runtime scope: every payload has
    # completed D2H, all private device values have been released, and provider
    # cleanup has succeeded before host table/scalar construction can begin.
    finalized_by_port: dict[OutputPortKey, object] = {}
    finalized_callbacks: list[tuple[_PendingHostFinalization, tuple[object, ...]]] = []
    for pending in pending_finalizations:
        _check_cancelled(cancel_callback)
        payloads = tuple(materialized[port] for port in pending.output_ports)
        finalized = apply_host_finalizer(
            pending.finalizer_ref,
            payloads,
            pending.call,
        )
        public_values = {
            port: value
            for port, value in zip(
                pending.output_ports,
                finalized,
                strict=True,
            )
        }
        _ensure_host_only(public_values, (runtime,))
        finalized_by_port.update(public_values)
        finalized_callbacks.append((pending, finalized))
        _check_cancelled(cancel_callback)

    # Do not publish callback-visible table/scalar metadata until every
    # finalizer in this segment has succeeded.  A later failure therefore
    # cannot expose a partially finalized segment.
    if node_outputs_callback is not None:
        for pending, outputs in finalized_callbacks:
            _check_cancelled(cancel_callback)
            node_outputs_callback(
                pending.node_id,
                pending.call,
                outputs,
                segment.runtime_id,
            )
    _check_cancelled(cancel_callback)
    provisional = {
        port: finalized_by_port.get(port, materialized[port])
        for port in _unique_ports((*segment.exit_ports, *segment.retained_ports))
    }
    return provisional


def _accelerator_lease_device_id(
    runtime: RuntimeProtocol,
    requested_device_id: str,
) -> str:
    requested = str(requested_device_id).strip()
    if requested:
        return requested
    probe = runtime.probe()
    selected = str(probe.selected_device_id).strip()
    return selected or "default"


def _execute_cpu_segment_fallback(
    unit: DeviceSegmentUnit,
    pipeline: PrototypePipeline,
    runtime: RuntimeProtocol,
    committed: Mapping[OutputPortKey, object],
    prepare_call: PrepareNodeCall,
    cancel_callback: Callable[[], bool] | None,
    node_outputs_callback: NodeOutputsCallback | None,
) -> dict[OutputPortKey, object]:
    """Retry one complete failed segment once with authoritative CPU calls."""

    segment = unit.segment
    local: dict[OutputPortKey, object] = {
        port: committed[port] for port in segment.entry_ports
    }
    for node_id in segment.node_ids:
        _check_cancelled(cancel_callback)
        inputs, _ports = _host_inputs_for_node(pipeline, node_id, local)
        call = prepare_call(node_id, inputs)
        _validate_prepared_call(
            call,
            node_id,
            pipeline.nodes[node_id].operation_id,
            len(pipeline.output_ports(node_id)),
        )
        raw = DEFAULT_CPU_NODE_EXECUTOR.execute(call)
        outputs = _normalized_outputs(raw, call.output_port_count)
        for value in outputs:
            if runtime.is_device_value(value):
                raise DevicePlanningError(
                    "CPU fallback returned a device-owned value."
                )
        if node_outputs_callback is not None:
            node_outputs_callback(node_id, call, outputs, CPU_RUNTIME_ID)
        for index, value in enumerate(outputs):
            local[OutputPortKey(node_id, index)] = value
    _check_cancelled(cancel_callback)
    return {
        port: local[port]
        for port in _unique_ports((*segment.exit_ports, *segment.retained_ports))
    }


def _host_finalizer_ref(implementation: OperationComputeSpec) -> str:
    return str(getattr(implementation, "host_finalizer_ref", "")).strip()


def _device_components(
    order: Sequence[str],
    connections: Sequence[GraphConnection],
    eligible: Mapping[
        str,
        tuple[
            NodeExecutionDecision,
            OperationComputeSpec,
            tuple[str, str, str, str],
        ],
    ],
    registry: ComputeRegistry,
) -> tuple[tuple[str, ...], ...]:
    def compatible(source: str, target: str) -> bool:
        source_key = eligible[source][2]
        target_key = eligible[target][2]
        if source_key[:3] != target_key[:3]:
            return False
        source_library = source_key[3]
        target_library = target_key[3]
        return source_library == target_library or bool(
            registry.interoperability_contract(
                source_key[0],
                (source_library, target_library),
            )
        )

    # A direct host-finalizer edge advances the downstream residency epoch.  An
    # alternate compatible path must not accidentally reunite both sides of the
    # boundary in the same undirected component (for example, a branch/join
    # where the join consumes both an upstream image and a finalized table).
    residency_epoch = {node_id: 0 for node_id in eligible}
    incoming: dict[str, list[str]] = {node_id: [] for node_id in eligible}
    for connection in connections:
        if (
            connection.source_id in eligible
            and connection.target_id in eligible
            and compatible(connection.source_id, connection.target_id)
        ):
            incoming[connection.target_id].append(connection.source_id)
    for node_id in order:
        if node_id not in eligible:
            continue
        predecessors = incoming[node_id]
        if predecessors:
            residency_epoch[node_id] = max(
                residency_epoch[source] + bool(_host_finalizer_ref(eligible[source][1]))
                for source in predecessors
            )

    adjacency: dict[str, set[str]] = {node_id: set() for node_id in eligible}
    for connection in connections:
        source = connection.source_id
        target = connection.target_id
        if source not in eligible or target not in eligible:
            continue
        # A finalizer converts the source node's transferred private payload to
        # a public host scalar/table.  Upstream device nodes may share its
        # segment, but no downstream node may observe the private payload.
        if _host_finalizer_ref(eligible[source][1]):
            continue
        if residency_epoch[source] != residency_epoch[target]:
            continue
        if not compatible(source, target):
            continue
        adjacency[source].add(target)
        adjacency[target].add(source)
    seen: set[str] = set()
    components: list[tuple[str, ...]] = []
    for node_id in order:
        if node_id not in eligible or node_id in seen:
            continue
        pending = [node_id]
        members: set[str] = set()
        while pending:
            candidate = pending.pop()
            if candidate in seen:
                continue
            seen.add(candidate)
            members.add(candidate)
            pending.extend(adjacency[candidate] - seen)
        components.append(tuple(item for item in order if item in members))
    return tuple(components)


def _stable_unit_topological_order(
    unit_keys: Sequence[tuple[str, object]],
    dependencies: Mapping[tuple[str, object], set[tuple[str, object]]],
    successors: Mapping[tuple[str, object], set[tuple[str, object]]],
    ranks: Mapping[tuple[str, object], int],
) -> tuple[tuple[str, object], ...]:
    remaining = {key: set(values) for key, values in dependencies.items()}
    ready = sorted(
        (key for key in unit_keys if not remaining[key]),
        key=ranks.__getitem__,
    )
    result: list[tuple[str, object]] = []
    while ready:
        key = ready.pop(0)
        result.append(key)
        for successor in sorted(successors[key], key=ranks.__getitem__):
            remaining[successor].discard(key)
            if (
                not remaining[successor]
                and successor not in result
                and successor not in ready
            ):
                ready.append(successor)
        ready.sort(key=ranks.__getitem__)
    if len(result) != len(unit_keys):
        raise DevicePlanningError("Execution-unit graph unexpectedly contains a cycle.")
    return tuple(result)


def _validated_retained_ports(
    pipeline: PrototypePipeline,
    retained_ports: Iterable[OutputPortKey],
) -> tuple[OutputPortKey, ...]:
    result: list[OutputPortKey] = []
    for port in retained_ports:
        if not isinstance(port, OutputPortKey):
            raise TypeError("retained_ports must contain OutputPortKey values.")
        if port.node_id not in pipeline.nodes:
            raise ValueError(f"Retained port references missing node {port.node_id!r}.")
        if port.port_index >= len(pipeline.output_ports(port.node_id)):
            raise ValueError(
                f"Retained port {port.node_id!r}:{port.port_index} does not exist."
            )
        if port not in result:
            result.append(port)
    return tuple(result)


def _sum_memory_estimates(estimates: Iterable[MemoryEstimate]) -> MemoryEstimate:
    values = tuple(estimates)
    model_ids = tuple(dict.fromkeys(value.model_id for value in values))
    return MemoryEstimate(
        runtime_managed_peak_bytes=sum(
            value.runtime_managed_peak_bytes for value in values
        ),
        total_device_peak_bytes=sum(value.total_device_peak_bytes for value in values),
        host_materialization_peak_bytes=sum(
            value.host_materialization_peak_bytes for value in values
        ),
        uncertainty_bytes=sum(value.uncertainty_bytes for value in values),
        model_id=(
            "segment-sum-v1"
            if len(model_ids) != 1
            else f"segment-sum-v1:{model_ids[0]}"
        ),
    )


def _sorted_ports(
    ports: Iterable[OutputPortKey],
    order_index: Mapping[str, int],
) -> tuple[OutputPortKey, ...]:
    return tuple(
        sorted(
            set(ports),
            key=lambda port: (
                order_index.get(port.node_id, len(order_index)),
                port.port_index,
                port.node_id,
            ),
        )
    )


def _unique_ports(ports: Iterable[OutputPortKey]) -> tuple[OutputPortKey, ...]:
    return tuple(dict.fromkeys(ports))


def _host_inputs_for_node(
    pipeline: PrototypePipeline,
    node_id: str,
    values: Mapping[OutputPortKey, object],
) -> tuple[tuple[object, ...], tuple[OutputPortKey, ...]]:
    connections = pipeline._input_connections(node_id)
    ports = tuple(
        OutputPortKey(connection.source_id, connection.source_port)
        for connection in connections
    )
    try:
        inputs = tuple(values[port] for port in ports)
    except KeyError as exc:
        missing = exc.args[0]
        raise DevicePlanningError(
            f"Node {node_id!r} is missing input {missing!r}."
        ) from exc
    return inputs, ports


def _validate_prepared_call(
    call: PreparedNodeCall,
    node_id: str,
    operation_id: str,
    output_port_count: int,
) -> None:
    if not isinstance(call, PreparedNodeCall):
        raise TypeError("prepare_call must return PreparedNodeCall.")
    if call.node_id != node_id or call.operation_id != operation_id:
        raise ValueError(
            "Prepared call identity does not match the scheduled pipeline node."
        )
    if call.output_port_count != output_port_count:
        raise ValueError(
            "Prepared call output count does not match the scheduled pipeline node."
        )


def _provider_keyword_arguments(call: PreparedNodeCall) -> dict[str, object]:
    """Detach the public operation contract for an external provider."""
    return {
        name: value
        for name, value in call.keyword_arguments().items()
        if not name.startswith("_vipp_")
    }


def _normalized_outputs(raw: object, output_count: int) -> tuple[object, ...]:
    return normalize_operation_outputs(raw, output_count)


def _release_orphan_outputs(
    runtime: RuntimeProtocol,
    raw: object,
    store: _SegmentStore,
) -> None:
    candidates = raw if isinstance(raw, (tuple, list)) else (raw,)
    seen: set[int] = set()
    for candidate in candidates:
        identity = id(candidate)
        if identity in seen or store.owns(candidate):
            continue
        seen.add(identity)
        try:
            is_device = runtime.is_device_value(candidate)
        except Exception:
            is_device = False
        if is_device:
            _release_quietly(runtime, candidate)


def _release_quietly(runtime: RuntimeProtocol, value: object) -> None:
    try:
        runtime.release(value)
    except Exception:
        pass


def _classify_runtime_exception(
    runtime: RuntimeProtocol,
    exc: BaseException,
) -> RuntimeExceptionInfo:
    try:
        return runtime.classify_exception(exc)
    except Exception as classification_error:
        return RuntimeExceptionInfo(
            RuntimeExceptionKind.UNKNOWN,
            "runtime_exception_classification_failed",
            f"{type(exc).__name__}: {exc}; classifier failed with "
            f"{type(classification_error).__name__}: {classification_error}",
            exception_type=type(exc).__name__,
            retryable=False,
        )


def _best_effort_synchronize(runtime: RuntimeProtocol, device_id: str) -> None:
    try:
        runtime.synchronize(device_id=device_id)
    except Exception:
        pass


def _best_effort_memory_snapshot(runtime: RuntimeProtocol, device_id: str):
    try:
        return runtime.memory_snapshot(device_id=device_id)
    except Exception:
        return None


def _attach_fallback_records(
    exc: BaseException,
    records: Sequence[ExecutionFallbackRecord],
) -> None:
    """Preserve earlier fallback attempts without replacing exception taxonomy."""

    if not records:
        return
    try:
        exc.vipp_fallback_records = tuple(records)
    except Exception:
        return


def _check_cancelled(callback: Callable[[], bool] | None) -> None:
    if callback is not None and callback():
        raise OperationCancelled("Operation cancelled.")


def _ensure_host_only(
    values: Mapping[OutputPortKey, object],
    runtimes: Sequence[RuntimeProtocol],
) -> None:
    for port, value in values.items():
        if _contains_device_value(value, runtimes, set()):
            raise DevicePlanningError(
                f"Device-owned value escaped through {port.node_id!r} "
                f"output {port.port_index}."
            )


def _contains_device_value(
    value: object,
    runtimes: Sequence[RuntimeProtocol],
    seen: set[int],
) -> bool:
    for runtime in runtimes:
        if runtime.is_device_value(value):
            return True
    identity = id(value)
    if identity in seen:
        return False
    seen.add(identity)
    if isinstance(value, Mapping):
        return any(
            _contains_device_value(item, runtimes, seen)
            for item in (*value.keys(), *value.values())
        )
    if isinstance(value, (tuple, list, set, frozenset)):
        return any(_contains_device_value(item, runtimes, seen) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        return any(
            _contains_device_value(getattr(value, item.name), runtimes, seen)
            for item in fields(value)
        )
    return False


__all__ = [
    "DeviceExecutionCancelled",
    "DeviceExecutionError",
    "DeviceExecutionResult",
    "DeviceGraphPlan",
    "DeviceMemoryPreflightError",
    "DevicePlanningError",
    "DeviceSegmentUnit",
    "ExecutionUnit",
    "HostExecutionUnit",
    "NodeOutputsCallback",
    "PrepareNodeCall",
    "execute_device_plan",
    "plan_device_execution",
    "preflight_device_execution",
]
