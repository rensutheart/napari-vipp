"""Production-node adapters for the generic transactional benchmark service.

This module is provider-neutral and import-safe: importing it does not import
CuPy, CuPyX, cuCIM, or initialize a device.  A registered runtime and callable
are not touched at module import. The explicit builder probes the runtime once
to resolve the exact device identity; the implementation callable remains lazy
until a candidate executes. Every invocation owns a fresh detached host call
and a private runtime scope; only host values cross back into the generic
parity/timing service.
"""

from __future__ import annotations

import copy
import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass, replace
from hashlib import sha256

import numpy as np

from napari_vipp.core.accelerator_lease import accelerator_lease
from napari_vipp.core.compute import WorkloadDescriptor, canonical_digest
from napari_vipp.core.compute_benchmark import (
    ADAPTIVE_WARM_ROUNDS,
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_CONFIDENCE_LEVEL,
    MINIMUM_WARM_ROUNDS,
    SCREENING_MINIMUM_WARM_ROUNDS,
    BenchmarkBudgetExceeded,
    BenchmarkCancelled,
    BenchmarkImplementation,
    BenchmarkInvocationObservation,
    BenchmarkProgressError,
    NodeBenchmarkRequest,
    ParityResult,
)
from napari_vipp.core.compute_registry import (
    ComputeRegistry,
    RuntimeExceptionInfo,
    RuntimeExceptionKind,
    RuntimeMemorySnapshot,
    RuntimeProtocol,
)
from napari_vipp.core.compute_specs import AdmissionTier, OperationComputeSpec
from napari_vipp.core.host_finalization import apply_host_finalizer
from napari_vipp.core.measurements import (
    MEASUREMENT_TABLE_PARITY_OPERATION_IDS,
    MEASUREMENT_TABLE_PARITY_POLICY_ID,
    measurement_table_parity,
)
from napari_vipp.core.node_execution import PreparedNodeCall
from napari_vipp.core.progress import ProgressContext, ProgressUpdate
from napari_vipp.core.richardson_lucy_parity import (
    RICHARDSON_LUCY_FLOAT32_ABSOLUTE_FLOOR,
    RICHARDSON_LUCY_FLOAT32_MAX_ABS_BASE,
    RICHARDSON_LUCY_FLOAT32_MAX_ABS_PEAK_FACTOR,
    RICHARDSON_LUCY_FLOAT32_NEAR_IDENTITY_NRMSE_LIMIT,
    RICHARDSON_LUCY_FLOAT32_NRMSE_LIMIT,
    RICHARDSON_LUCY_PARITY_OPERATION_IDS,
    RICHARDSON_LUCY_TV_FLOAT32_MAX_ABS_BASE,
    RICHARDSON_LUCY_TV_FLOAT32_MAX_ABS_PEAK_FACTOR,
    RICHARDSON_LUCY_TV_FLOAT32_NRMSE_LIMIT,
    RICHARDSON_LUCY_TV_PARITY_OPERATION_IDS,
    richardson_lucy_float32_parity,
    richardson_lucy_tv_float32_parity,
)

PRODUCTION_BENCHMARK_POLICY_ID = "production-node-paired-adaptive-bootstrap-v3"
PIPELINE_SCREENING_BENCHMARK_POLICY_ID = (
    "pipeline-node-progressive-screening-bootstrap-v2"
)
CUSTOM_BENCHMARK_POLICY_ID = "custom-node-paired-adaptive-bootstrap-v3"
EXACT_PARITY_OPERATION_IDS = frozenset(
    {
        "canny_edges",
        "binary_threshold",
        "convert_dtype",
        "extract_channel",
        "median_filter",
        "otsu_threshold",
        "label_connected_components",
    }
)
EXACT_MASK_PARITY_OPERATION_IDS = frozenset(
    {"binary_threshold", "canny_edges", "otsu_threshold"}
)
EXACT_LABEL_PARITY_OPERATION_IDS = frozenset({"label_connected_components"})
BACKGROUND_PARITY_OPERATION_IDS = frozenset(
    {"rolling_ball_background", "subtract_background"}
)
GAUSSIAN_PARITY_OPERATION_IDS = frozenset({"gaussian_blur", "gaussian_blur_3d"})
SIGMA_FILTER_PARITY_OPERATION_IDS = frozenset({"sigma_filter"})
BENCHMARK_WRITER_OPERATION_IDS = frozenset({"save_output", "batch_output"})
GAUSSIAN_FLOAT32_NRMSE_LIMIT = 2e-6
GAUSSIAN_FLOAT32_ABSOLUTE_FLOOR = 1e-12
BACKGROUND_FLOAT32_NRMSE_LIMIT = 2e-6
BACKGROUND_FLOAT32_MAX_ABS_BASE = 1e-6
BACKGROUND_FLOAT32_SCALE_ULPS = 2.0
SIGMA_FILTER_FLOAT32_NRMSE_LIMIT = 2e-6
SIGMA_FILTER_FLOAT32_MAX_ABS_BASE = 1e-6
SIGMA_FILTER_FLOAT32_SCALE_ULPS = 4.0
_BENCHMARK_ABORT_CALLBACK_KWARG = "__vipp_benchmark_abort_callback"


@dataclass(frozen=True, slots=True)
class ProductionInvocationObservation:
    """Rich sidecar evidence retained outside scientific/workflow caches."""

    implementation_id: str
    invocation_index: int
    measurement: BenchmarkInvocationObservation
    snapshots: tuple[tuple[str, RuntimeMemorySnapshot], ...]
    terminal_snapshot: RuntimeMemorySnapshot
    cleanup_succeeded: bool


class ProductionBenchmarkObservationLog:
    """Thread-safe machine-local observation log for one benchmark request."""

    def __init__(self) -> None:
        self._runs: dict[str, list[ProductionInvocationObservation]] = {}
        self._lock = threading.RLock()

    def record(
        self,
        implementation_id: str,
        measurement: BenchmarkInvocationObservation,
        snapshots: Sequence[tuple[str, RuntimeMemorySnapshot]],
        terminal_snapshot: RuntimeMemorySnapshot,
        *,
        cleanup_succeeded: bool,
    ) -> ProductionInvocationObservation:
        with self._lock:
            runs = self._runs.setdefault(str(implementation_id), [])
            observation = ProductionInvocationObservation(
                implementation_id=str(implementation_id),
                invocation_index=len(runs),
                measurement=measurement,
                snapshots=tuple(snapshots),
                terminal_snapshot=terminal_snapshot,
                cleanup_succeeded=bool(cleanup_succeeded),
            )
            runs.append(observation)
            return observation

    def latest_measurement(
        self,
        implementation_id: str,
    ) -> BenchmarkInvocationObservation | None:
        with self._lock:
            runs = self._runs.get(str(implementation_id), ())
            return runs[-1].measurement if runs else None

    def runs(
        self,
        implementation_id: str | None = None,
    ) -> tuple[ProductionInvocationObservation, ...]:
        with self._lock:
            if implementation_id is not None:
                return tuple(self._runs.get(str(implementation_id), ()))
            return tuple(
                observation
                for key in sorted(self._runs)
                for observation in self._runs[key]
            )


@dataclass(frozen=True, slots=True)
class RegisteredNodeBenchmark:
    """A generic request plus its detached call and rich local observations."""

    request: NodeBenchmarkRequest
    detached_call: PreparedNodeCall = field(repr=False, compare=False)
    observations: ProductionBenchmarkObservationLog = field(
        repr=False,
        compare=False,
    )


def build_registered_node_benchmark(
    call: PreparedNodeCall,
    *,
    admitted_specs: Sequence[OperationComputeSpec],
    registry: ComputeRegistry,
    environment_fingerprint: str,
    device_id: str = "",
    memory_limit_bytes: int | None = None,
    safety_reserve_bytes: int | None = None,
    warm_rounds: int = MINIMUM_WARM_ROUNDS,
    max_warm_rounds: int = ADAPTIVE_WARM_ROUNDS[-1],
    time_budget_seconds: float | None = None,
    allow_experimental: bool = False,
    clock: Callable[[], float] = time.perf_counter,
    paired_bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    paired_bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    paired_confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    check_abort: Callable[[], None] | None = None,
    call_is_detached: bool = False,
) -> RegisteredNodeBenchmark:
    """Build a production-faithful request from an already prepared node call.

    ``admitted_specs`` is deliberately explicit.  This adapter does not expand
    scientific support from a successful one-off benchmark; planning must first
    admit each exact implementation for the call's dtype/shape/parameters.
    """

    if not isinstance(call, PreparedNodeCall):
        raise TypeError("call must be a PreparedNodeCall.")
    if not callable(clock):
        raise TypeError("clock must be callable.")
    if check_abort is not None and not callable(check_abort):
        raise TypeError("check_abort must be callable or None.")
    if not isinstance(call_is_detached, bool):
        raise TypeError("call_is_detached must be a boolean.")
    environment = str(environment_fingerprint).strip()
    if not environment:
        raise ValueError("environment_fingerprint must not be empty.")
    specs = tuple(admitted_specs)
    if not specs:
        raise ValueError("admitted_specs must contain at least one GPU candidate.")
    if any(not isinstance(spec, OperationComputeSpec) for spec in specs):
        raise TypeError("admitted_specs must contain OperationComputeSpec values.")
    if len({spec.implementation_id for spec in specs}) != len(specs):
        raise ValueError("admitted_specs implementation IDs must be unique.")
    if call.operation_id in BENCHMARK_WRITER_OPERATION_IDS:
        raise ValueError("Registered node benchmarking refuses writer operations.")
    if call.output_port_count != 1:
        raise ValueError(
            "Registered node benchmarking requires exactly one output; "
            "multiple ordered inputs are supported."
        )
    if _BENCHMARK_ABORT_CALLBACK_KWARG in call.kwargs:
        raise ValueError("Prepared calls may not define benchmark-private keywords.")

    for spec in specs:
        _validate_admitted_spec(
            call,
            spec,
            registry,
            allow_experimental=allow_experimental,
        )

    resolved_device_id = _resolve_device_id(registry, specs, device_id)
    _validate_memory_scope(
        memory_limit_bytes=memory_limit_bytes,
        safety_reserve_bytes=safety_reserve_bytes,
    )

    _run_abort_check(check_abort)
    detached = (
        call
        if call_is_detached
        else detach_prepared_node_call(call, check_abort=check_abort)
    )
    if call_is_detached:
        _validate_detached_call(detached)
    workload = workload_from_prepared_node_call(
        detached,
        check_abort=check_abort,
    )
    observations = ProductionBenchmarkObservationLog()
    candidates = tuple(
        _candidate_implementation(
            detached,
            spec,
            registry,
            observations,
            device_id=resolved_device_id,
            memory_limit_bytes=memory_limit_bytes,
            safety_reserve_bytes=safety_reserve_bytes,
            allow_experimental=allow_experimental,
            clock=clock,
        )
        for spec in specs
    )
    reference = BenchmarkImplementation(
        implementation_id=f"cpu-{call.operation_id}-v1",
        execute=_execute_cpu_reference,
        implementation_version="1",
    )
    request = NodeBenchmarkRequest(
        workload=workload,
        environment_fingerprint=environment,
        reference=reference,
        candidates=candidates,
        private_input_factory=lambda: _clone_detached_call(
            detached,
            check_abort=check_abort,
        ),
        parity=lambda expected, actual: operation_parity(
            call.operation_id,
            expected,
            actual,
            input_peak=_finite_input_peak(detached.inputs[0]),
            input_dtypes=tuple(np.asarray(value).dtype for value in detached.inputs),
            parameters=detached.kwargs,
        ),
        benchmark_policy_id=_benchmark_policy_id(
            warm_rounds=warm_rounds,
            max_warm_rounds=max_warm_rounds,
            paired_bootstrap_samples=paired_bootstrap_samples,
            paired_bootstrap_seed=paired_bootstrap_seed,
            paired_confidence_level=paired_confidence_level,
        ),
        scientific_contract_digest=_benchmark_scientific_contract_digest(
            call.operation_id,
            specs,
        ),
        warm_rounds=warm_rounds,
        time_budget_seconds=time_budget_seconds,
        time_parity_as_cold=True,
        warmup_rounds=1,
        adaptive_rounds=True,
        max_warm_rounds=max_warm_rounds,
        paired_bootstrap_samples=paired_bootstrap_samples,
        paired_bootstrap_seed=paired_bootstrap_seed,
        paired_confidence_level=paired_confidence_level,
        device_id=resolved_device_id,
        memory_limit_bytes=memory_limit_bytes,
        safety_reserve_bytes=safety_reserve_bytes,
        bind_operation_progress=_bind_benchmark_operation_progress,
    )
    return RegisteredNodeBenchmark(request, detached, observations)


def detach_prepared_node_call(
    call: PreparedNodeCall,
    *,
    check_abort: Callable[[], None] | None = None,
) -> PreparedNodeCall:
    """Copy runtime data and remove live progress state from a prepared call."""

    if not isinstance(call, PreparedNodeCall):
        raise TypeError("call must be a PreparedNodeCall.")
    if check_abort is not None and not callable(check_abort):
        raise TypeError("check_abort must be callable or None.")
    _run_abort_check(check_abort)
    kwargs = call.keyword_arguments()
    if "progress" in kwargs:
        kwargs["progress"] = None
    kwargs = copy.deepcopy(kwargs)
    return PreparedNodeCall(
        node_id=call.node_id,
        operation_id=call.operation_id,
        cpu_function=call.cpu_function,
        inputs=tuple(
            _detached_host_value(value, check_abort=check_abort)
            for value in call.inputs
        ),
        input_states=copy.deepcopy(call.input_states),
        kwargs=kwargs,
        multiple_inputs=call.multiple_inputs,
        output_port_count=call.output_port_count,
    )


def _bind_benchmark_operation_progress(
    private_input: object,
    reporter,
    abort,
) -> object:
    """Attach fresh benchmark-only progress after private input cloning."""

    if not isinstance(private_input, PreparedNodeCall):
        raise TypeError("private benchmark input must be a PreparedNodeCall.")
    kwargs = private_input.keyword_arguments()

    def forward(update: ProgressUpdate) -> None:
        if reporter is None:
            return
        reporter(
            update.current,
            update.total,
            _benchmark_operation_progress_message(
                private_input,
                update.message,
            ),
        )

    if "progress" in kwargs:
        kwargs["progress"] = ProgressContext(
            cancelled=abort,
            reporter=forward if reporter is not None else None,
        )
    kwargs[_BENCHMARK_ABORT_CALLBACK_KWARG] = abort
    return PreparedNodeCall(
        node_id=private_input.node_id,
        operation_id=private_input.operation_id,
        cpu_function=private_input.cpu_function,
        inputs=private_input.inputs,
        input_states=private_input.input_states,
        kwargs=kwargs,
        multiple_inputs=private_input.multiple_inputs,
        output_port_count=private_input.output_port_count,
    )


def _benchmark_operation_progress_message(
    call: PreparedNodeCall,
    message: str,
) -> str:
    if call.operation_id in BACKGROUND_PARITY_OPERATION_IDS:
        resolved = call.kwargs.get("resolved_spatial_ndim")
        if resolved == 2:
            return "Rolling-ball YX plane"
        if resolved == 3:
            return "Rolling-ball spatial volume"
        return "Rolling-ball spatial block"
    return str(message).strip() or call.operation_id.replace("_", " ")


def _validate_detached_call(call: PreparedNodeCall) -> None:
    """Reject mutable arrays passed through the trusted detached fast path."""

    for value in call.inputs:
        if isinstance(value, np.ndarray) and value.flags.writeable:
            raise ValueError(
                "call_is_detached=True requires read-only NumPy input arrays."
            )


def operation_parity(
    operation_id: str,
    reference: object,
    candidate: object,
    *,
    input_peak: float | None = None,
    input_dtypes: Sequence[object] = (),
    parameters: Mapping[str, object] | None = None,
) -> ParityResult:
    """Apply the registered operation's production scientific parity gate."""

    operation = str(operation_id).strip()
    if operation in BACKGROUND_PARITY_OPERATION_IDS:
        return _background_dtype_parity(
            reference,
            candidate,
            input_peak=input_peak,
        )
    if operation in SIGMA_FILTER_PARITY_OPERATION_IDS:
        return _sigma_filter_dtype_parity(
            reference,
            candidate,
            input_peak=input_peak,
        )
    if operation in EXACT_PARITY_OPERATION_IDS:
        return _exact_array_parity(reference, candidate)
    if operation in GAUSSIAN_PARITY_OPERATION_IDS:
        return _gaussian_float32_parity(reference, candidate)
    if operation in RICHARDSON_LUCY_PARITY_OPERATION_IDS:
        return richardson_lucy_float32_parity(reference, candidate)
    if operation in RICHARDSON_LUCY_TV_PARITY_OPERATION_IDS:
        regularization = (parameters or {}).get("tv_regularization", 0.002)
        try:
            lambda_zero = float(regularization) == 0.0
        except (TypeError, ValueError):
            lambda_zero = False
        if lambda_zero:
            return richardson_lucy_float32_parity(reference, candidate)
        return richardson_lucy_tv_float32_parity(reference, candidate)
    if operation in MEASUREMENT_TABLE_PARITY_OPERATION_IDS:
        intensity_dtype = input_dtypes[1] if len(input_dtypes) > 1 else None
        return measurement_table_parity(
            reference,
            candidate,
            intensity_dtype=intensity_dtype,
        )
    raise ValueError(f"No production benchmark parity policy for {operation!r}.")


def _resolve_device_id(
    registry: ComputeRegistry,
    specs: Sequence[OperationComputeSpec],
    requested_device_id: str,
) -> str:
    runtime_ids = {spec.runtime_id for spec in specs}
    if len(runtime_ids) != 1:
        raise ValueError(
            "One node benchmark request must use one resolved array runtime."
        )
    runtime_id = next(iter(runtime_ids))
    probe = registry.probe_runtime(runtime_id)
    if not probe.available:
        raise ValueError(
            probe.message or f"Runtime {runtime_id!r} is unavailable for benchmarking."
        )
    requested = str(requested_device_id).strip()
    known_ids = {device.device_id for device in probe.devices}
    if requested:
        if requested not in known_ids:
            raise ValueError(
                f"Device {requested!r} is not reported by runtime {runtime_id!r}."
            )
        return requested
    selected = probe.selected_device_id
    if not selected:
        raise ValueError(f"Runtime {runtime_id!r} did not resolve a benchmark device.")
    return selected


def _validate_memory_scope(
    *,
    memory_limit_bytes: int | None,
    safety_reserve_bytes: int | None,
) -> None:
    if memory_limit_bytes is not None and (
        isinstance(memory_limit_bytes, bool)
        or not isinstance(memory_limit_bytes, int)
        or memory_limit_bytes <= 0
    ):
        raise ValueError("memory_limit_bytes must be positive or None.")
    if safety_reserve_bytes is not None and (
        isinstance(safety_reserve_bytes, bool)
        or not isinstance(safety_reserve_bytes, int)
        or safety_reserve_bytes < 0
    ):
        raise ValueError("safety_reserve_bytes must be non-negative or None.")


def _benchmark_policy_id(
    *,
    warm_rounds: int,
    max_warm_rounds: int,
    paired_bootstrap_samples: int,
    paired_bootstrap_seed: int,
    paired_confidence_level: float,
) -> str:
    production_profile = (
        warm_rounds == MINIMUM_WARM_ROUNDS
        and max_warm_rounds == ADAPTIVE_WARM_ROUNDS[-1]
        and paired_bootstrap_samples == DEFAULT_BOOTSTRAP_SAMPLES
        and paired_bootstrap_seed == DEFAULT_BOOTSTRAP_SEED
        and paired_confidence_level == DEFAULT_CONFIDENCE_LEVEL
    )
    pipeline_screening_profile = (
        warm_rounds == SCREENING_MINIMUM_WARM_ROUNDS
        and max_warm_rounds == 15
        and paired_bootstrap_samples == DEFAULT_BOOTSTRAP_SAMPLES
        and paired_bootstrap_seed == DEFAULT_BOOTSTRAP_SEED
        and paired_confidence_level == DEFAULT_CONFIDENCE_LEVEL
    )
    if production_profile:
        return PRODUCTION_BENCHMARK_POLICY_ID
    if pipeline_screening_profile:
        return PIPELINE_SCREENING_BENCHMARK_POLICY_ID
    return CUSTOM_BENCHMARK_POLICY_ID


def _benchmark_scientific_contract_digest(
    operation_id: str,
    specs: Sequence[OperationComputeSpec],
) -> str:
    """Bind reusable evidence to every declared scientific policy surface."""

    return canonical_digest(
        {
            "identity_policy_id": "production-node-scientific-contract-v1",
            "operation_id": operation_id,
            "reference": {
                "implementation_id": f"cpu-{operation_id}-v1",
                "implementation_version": "1",
            },
            "ordered_candidates": tuple(
                {
                    "operation_id": spec.operation_id,
                    "implementation_id": spec.implementation_id,
                    "implementation_version": spec.implementation_version,
                    "runtime_id": spec.runtime_id,
                    "array_domain": spec.array_domain,
                    "implementation_library_id": (
                        spec.implementation_library_id
                    ),
                    "validated_environment_policy_id": (
                        spec.validated_environment_policy_id
                    ),
                    "input_ports": spec.input_ports,
                    "output_ports": spec.output_ports,
                    "parameter_policy_id": spec.parameter_policy_id,
                    "workload_policy_id": spec.workload_policy_id,
                    "parity_policy_id": spec.parity_policy_id,
                    "shape_policy_id": spec.shape_policy_id,
                    "boundary_policy_id": spec.boundary_policy_id,
                    "precision_policy_id": spec.precision_policy_id,
                    "side_effect_policy_id": spec.side_effect_policy_id,
                    "dynamic_output_policy_id": spec.dynamic_output_policy_id,
                    "supported_spatial_ndims": spec.supported_spatial_ndims,
                    "limitations": spec.limitations,
                    "host_finalizer_ref": spec.host_finalizer_ref,
                }
                for spec in specs
            ),
        }
    )


def _validate_admitted_spec(
    call: PreparedNodeCall,
    spec: OperationComputeSpec,
    registry: ComputeRegistry,
    *,
    allow_experimental: bool,
) -> None:
    if spec.operation_id != call.operation_id:
        raise ValueError(
            f"Implementation {spec.implementation_id!r} belongs to "
            f"{spec.operation_id!r}, not {call.operation_id!r}."
        )
    if not spec.is_gpu or spec.host_boundary or not spec.supports_device_residency:
        raise ValueError(
            f"Implementation {spec.implementation_id!r} is not a resident GPU "
            "operation candidate."
        )
    if spec.side_effect_policy_id != "pure-v1":
        raise ValueError("Only declared pure operations may be benchmarked.")
    if len(spec.input_ports) != len(call.inputs):
        raise ValueError("Implementation input contracts do not match the call.")
    for index, port in enumerate(spec.input_ports):
        if port.port_index != index:
            raise ValueError(
                "Implementation input contracts must match ordered call ports."
            )
    if len(spec.output_ports) != call.output_port_count:
        raise ValueError("Implementation output contracts do not match the call.")
    if spec.admission_tier is AdmissionTier.DEVELOPER_HIDDEN and not allow_experimental:
        raise ValueError(
            f"Implementation {spec.implementation_id!r} is developer-hidden."
        )
    registered = registry.implementation_spec(
        spec.implementation_id,
        spec.implementation_version,
        allow_experimental=allow_experimental,
    )
    if registered != spec:
        raise ValueError(
            f"Implementation {spec.implementation_id!r} does not match its "
            "registered declaration."
        )
    expected_parity = (
        "gaussian-float32-tolerance-v1"
        if call.operation_id in GAUSSIAN_PARITY_OPERATION_IDS
        else None
    )
    if call.operation_id in EXACT_PARITY_OPERATION_IDS:
        expected_parity = (
            "labels-bitwise-int32-v1"
            if call.operation_id in EXACT_LABEL_PARITY_OPERATION_IDS
            else (
                "mask-bitwise-v1"
                if call.operation_id in EXACT_MASK_PARITY_OPERATION_IDS
                else (
                    "median-production-bitwise-v1"
                    if call.operation_id == "median_filter"
                    else "array-bitwise-v1"
                )
            )
        )
    elif call.operation_id in BACKGROUND_PARITY_OPERATION_IDS:
        expected_parity = "background-dtype-parity-v2"
    elif call.operation_id in RICHARDSON_LUCY_PARITY_OPERATION_IDS:
        expected_parity = "rl-scientific-equivalence-v2"
    elif call.operation_id in RICHARDSON_LUCY_TV_PARITY_OPERATION_IDS:
        expected_parity = "rl-tv-scientific-equivalence-v2"
    elif call.operation_id in SIGMA_FILTER_PARITY_OPERATION_IDS:
        expected_parity = "sigma-dtype-parity-v1"
    elif call.operation_id in MEASUREMENT_TABLE_PARITY_OPERATION_IDS:
        expected_parity = MEASUREMENT_TABLE_PARITY_POLICY_ID
    if expected_parity is None or spec.parity_policy_id != expected_parity:
        raise ValueError(
            f"Implementation {spec.implementation_id!r} has unsupported parity "
            f"policy {spec.parity_policy_id!r}."
        )


def _candidate_implementation(
    call: PreparedNodeCall,
    spec: OperationComputeSpec,
    registry: ComputeRegistry,
    observations: ProductionBenchmarkObservationLog,
    *,
    device_id: str,
    memory_limit_bytes: int | None,
    safety_reserve_bytes: int | None,
    allow_experimental: bool,
    clock: Callable[[], float],
) -> BenchmarkImplementation:
    runner = _RegisteredCandidateRunner(
        call,
        spec,
        registry,
        observations,
        device_id=str(device_id).strip(),
        memory_limit_bytes=memory_limit_bytes,
        safety_reserve_bytes=safety_reserve_bytes,
        allow_experimental=allow_experimental,
        clock=clock,
    )
    return BenchmarkImplementation(
        implementation_id=spec.implementation_id,
        execute=runner.execute,
        # execute returns only after kernel, D2H, cleanup, and terminal memory
        # synchronization; a second callback would be redundant and untimed.
        synchronize=lambda: None,
        peak_memory_bytes=runner.latest_peak_memory_bytes,
        observation=runner.latest_measurement,
        implementation_version=spec.implementation_version,
    )


class _DetachedBenchmarkCandidateFailure(RuntimeError):
    """Provider-neutral failure raised only after private-scope cleanup."""

    def __init__(self, failure: RuntimeExceptionInfo) -> None:
        self.failure = failure
        detail = failure.message or failure.reason_code
        super().__init__(f"{failure.kind.value}: {failure.reason_code}: {detail}")


@dataclass(slots=True)
class _RegisteredCandidateRunner:
    call: PreparedNodeCall
    spec: OperationComputeSpec
    registry: ComputeRegistry
    observations: ProductionBenchmarkObservationLog
    device_id: str
    memory_limit_bytes: int | None
    safety_reserve_bytes: int | None
    allow_experimental: bool
    clock: Callable[[], float] = field(repr=False)

    def latest_measurement(self) -> BenchmarkInvocationObservation | None:
        return self.observations.latest_measurement(self.spec.implementation_id)

    def latest_peak_memory_bytes(self) -> int:
        measurement = self.latest_measurement()
        return measurement.peak_memory_bytes if measurement is not None else 0

    def execute(self, private_call: object) -> object:
        if not isinstance(private_call, PreparedNodeCall):
            raise TypeError("private benchmark input must be a PreparedNodeCall.")
        if (
            private_call.node_id != self.call.node_id
            or private_call.operation_id != self.call.operation_id
        ):
            raise ValueError("private benchmark call does not match its template.")
        runtime = self.registry.runtime(self.spec.runtime_id)
        with accelerator_lease(
            self.spec.runtime_id,
            self.device_id,
            cancelled=_benchmark_abort_callback(private_call),
        ):
            implementation = self.registry.implementation_callable(
                self.spec,
                allow_experimental=self.allow_experimental,
            )
            return self._execute_in_scope(private_call, runtime, implementation)

    def _execute_in_scope(
        self,
        private_call: PreparedNodeCall,
        runtime: RuntimeProtocol,
        implementation: Callable[..., object],
    ) -> object:
        tracker = _AllocationTracker(runtime)
        snapshots: list[tuple[str, RuntimeMemorySnapshot]] = []
        transfer_seconds = 0.0
        resident_seconds = 0.0
        host_materialization_seconds = 0.0
        host_result: object | None = None
        host_payloads: tuple[object, ...] = ()
        raw: object | None = None
        outputs: tuple[object, ...] = ()
        device_inputs: tuple[object, ...] = ()
        positional: object | None = None
        output: object | None = None
        device_value: object | None = None
        transferred: list[object] = []
        detached_failure: RuntimeExceptionInfo | None = None
        control_failure: (
            tuple[
                type[BenchmarkCancelled]
                | type[BenchmarkBudgetExceeded]
                | type[BenchmarkProgressError],
                str,
            ]
            | None
        ) = None
        cleanup_failure: RuntimeExceptionInfo | None = None
        scope_failure: RuntimeExceptionInfo | None = None
        terminal_failure: RuntimeExceptionInfo | None = None
        terminal: RuntimeMemorySnapshot | None = None

        try:
            with runtime.execution_scope(
                device_id=self.device_id,
                memory_limit_bytes=self.memory_limit_bytes,
                safety_reserve_bytes=self.safety_reserve_bytes,
            ):
                try:
                    try:
                        snapshots.append(("scope_enter", self._snapshot(runtime)))

                        transfer_started = _read_clock(self.clock)
                        for host_value in private_call.inputs:
                            device_value = runtime.to_device(
                                host_value,
                                device_id=self.device_id,
                            )
                            try:
                                tracker.add(device_value)
                            except Exception:
                                _release_untracked(runtime, device_value, tracker)
                                raise
                            transferred.append(device_value)
                        device_inputs = tuple(transferred)
                        transferred = []
                        device_value = None
                        runtime.synchronize(device_id=self.device_id)
                        transfer_seconds += _elapsed(self.clock, transfer_started)
                        snapshots.append(("post_h2d", self._snapshot(runtime)))

                        resident_started = _read_clock(self.clock)
                        positional = (
                            list(device_inputs)
                            if private_call.multiple_inputs
                            else device_inputs[0]
                        )
                        raw = implementation(
                            positional,
                            **_provider_keyword_arguments(private_call),
                        )
                        try:
                            outputs = _normalized_outputs(
                                raw,
                                private_call.output_port_count,
                            )
                        except Exception:
                            _release_orphan_outputs(runtime, raw, tracker)
                            raise
                        try:
                            invalid = tuple(
                                value
                                for value in outputs
                                if not runtime.is_device_value(value)
                            )
                        except Exception:
                            _release_orphan_outputs(runtime, outputs, tracker)
                            raise
                        if invalid:
                            _release_orphan_outputs(runtime, outputs, tracker)
                            raise TypeError(
                                f"Implementation {self.spec.implementation_id!r} "
                                "returned a host value inside its device scope."
                            )
                        try:
                            for output in outputs:
                                tracker.add(output)
                        except Exception:
                            _release_orphan_outputs(runtime, outputs, tracker)
                            raise
                        output = None
                        runtime.synchronize(device_id=self.device_id)
                        resident_seconds = _elapsed(self.clock, resident_started)
                        snapshots.append(("post_kernel", self._snapshot(runtime)))

                        transfer_started = _read_clock(self.clock)
                        host_payloads = tuple(
                            runtime.to_host(output) for output in outputs
                        )
                        runtime.synchronize(device_id=self.device_id)
                        transfer_seconds += _elapsed(self.clock, transfer_started)
                        snapshots.append(("post_d2h", self._snapshot(runtime)))
                        _ensure_host_only(host_payloads, runtime)
                    except (
                        BenchmarkCancelled,
                        BenchmarkBudgetExceeded,
                        BenchmarkProgressError,
                    ) as exc:
                        # Do not retain the original exception: its traceback
                        # can hold provider frames and device temporaries alive
                        # across private-scope cleanup and memory validation.
                        control_failure = (type(exc), str(exc))
                    except Exception as exc:
                        # Provider exceptions may retain traceback-local device
                        # arrays. Classify and suppress them before the private
                        # allocator validates cleanup on scope exit.
                        detached_failure = _classify_runtime_exception(
                            runtime,
                            exc,
                        )
                finally:
                    release_error = tracker.release_all()
                    if release_error is not None:
                        cleanup_failure = _classify_runtime_exception(
                            runtime,
                            release_error,
                        )
                    release_error = None
                    raw = None
                    outputs = ()
                    device_inputs = ()
                    positional = None
                    output = None
                    device_value = None
                    transferred = []
                    invalid = ()
                    try:
                        runtime.synchronize(device_id=self.device_id)
                        snapshots.append(("post_release", self._snapshot(runtime)))
                    except Exception as exc:
                        cleanup_failure = _combine_runtime_failures(
                            cleanup_failure,
                            _classify_runtime_exception(runtime, exc),
                        )
        except Exception as exc:
            # Scope-exit failures are already outside the provider allocator,
            # but are still sanitized before crossing the adapter boundary.
            scope_failure = _classify_runtime_exception(runtime, exc)

        try:
            terminal = self._snapshot(runtime)
        except Exception as exc:
            terminal_failure = _classify_runtime_exception(runtime, exc)

        cleanup_succeeded = (
            cleanup_failure is None
            and scope_failure is None
            and terminal_failure is None
            and terminal is not None
            and terminal.runtime_live_bytes == 0
            and terminal.runtime_reserved_bytes == 0
        )
        if terminal is not None and not cleanup_succeeded:
            cleanup_failure = _combine_runtime_failures(
                cleanup_failure,
                RuntimeExceptionInfo(
                    RuntimeExceptionKind.KERNEL_FAILURE,
                    "benchmark_cleanup_incomplete",
                    "Benchmark candidate left runtime-managed allocations or "
                    "its private runtime scope failed cleanup.",
                    retryable=False,
                ),
            )
        failures = tuple(
            failure
            for failure in (
                detached_failure,
                cleanup_failure,
                scope_failure,
                terminal_failure,
            )
            if failure is not None
        )
        final_failure = _combined_runtime_failure(failures)

        if terminal is None:
            if final_failure is None:
                raise RuntimeError("Benchmark terminal memory state is unavailable.")
            raise _DetachedBenchmarkCandidateFailure(final_failure) from None
        if control_failure is not None:
            if final_failure is not None:
                raise _DetachedBenchmarkCandidateFailure(final_failure) from None
            failure_type, failure_message = control_failure
            raise failure_type(failure_message) from None
        if final_failure is None:
            abort = _benchmark_abort_callback(private_call)
            if abort is not None and abort():
                raise BenchmarkCancelled(
                    "Benchmark cancelled before host output finalization."
                )
            finalizer_ref = str(
                getattr(self.spec, "host_finalizer_ref", "")
            ).strip()
            boundary_started = _read_clock(self.clock)
            if finalizer_ref:
                finalized = apply_host_finalizer(
                    finalizer_ref,
                    host_payloads,
                    replace(
                        private_call,
                        inputs=(None,) * len(private_call.inputs),
                    ),
                )
                host_result = (
                    finalized[0]
                    if private_call.output_port_count == 1
                    else finalized
                )
            else:
                host_result = (
                    host_payloads[0]
                    if private_call.output_port_count == 1
                    else host_payloads
                )
            _ensure_host_only(host_result, runtime)
            if finalizer_ref:
                # The typed host conversion is a mandatory output-boundary
                # cost, never resident compute or a directional transfer.
                host_materialization_seconds = _elapsed(
                    self.clock,
                    boundary_started,
                )
            if abort is not None and abort():
                raise BenchmarkCancelled(
                    "Benchmark cancelled after host output finalization."
                )
        observed = tuple(snapshot for _stage, snapshot in snapshots) + (terminal,)
        measurement = BenchmarkInvocationObservation(
            timing_scope=(
                "synchronized-end-to-end-host-finalized-v1"
                if str(getattr(self.spec, "host_finalizer_ref", "")).strip()
                else "synchronized-end-to-end-v1"
            ),
            synchronized=True,
            transfers_included=True,
            transfer_seconds=transfer_seconds,
            resident_seconds=resident_seconds,
            host_materialization_seconds=host_materialization_seconds,
            runtime_live_bytes=max(
                (snapshot.runtime_live_bytes for snapshot in observed),
                default=0,
            ),
            runtime_reserved_bytes=max(
                (snapshot.runtime_reserved_bytes for snapshot in observed),
                default=0,
            ),
            out_of_pool_bytes=max(
                (snapshot.out_of_pool_bytes for snapshot in observed),
                default=0,
            ),
        )
        self.observations.record(
            self.spec.implementation_id,
            measurement,
            snapshots,
            terminal,
            cleanup_succeeded=cleanup_succeeded,
        )
        if final_failure is not None:
            raise _DetachedBenchmarkCandidateFailure(final_failure) from None
        return host_result

    def _snapshot(self, runtime: RuntimeProtocol) -> RuntimeMemorySnapshot:
        snapshot = runtime.memory_snapshot(device_id=self.device_id)
        if not isinstance(snapshot, RuntimeMemorySnapshot):
            raise TypeError("runtime memory_snapshot returned an invalid value.")
        return snapshot


class _AllocationTracker:
    """Release one representative of each private allocation exactly once."""

    def __init__(self, runtime: RuntimeProtocol) -> None:
        self.runtime = runtime
        self._values: dict[object, object] = {}

    def add(self, value: object) -> None:
        identity = self.runtime.allocation_identity(value)
        try:
            hash(identity)
        except TypeError as exc:
            raise TypeError("runtime allocation identity must be hashable.") from exc
        self._values.setdefault(identity, value)

    def owns(self, value: object) -> bool:
        try:
            identity = self.runtime.allocation_identity(value)
        except Exception:
            return False
        return identity in self._values

    def release_all(self) -> Exception | None:
        first_error = None
        values = tuple(self._values.values())
        self._values.clear()
        for value in reversed(values):
            try:
                self.runtime.release(value)
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        value = None
        values = ()
        return first_error


def _release_untracked(
    runtime: RuntimeProtocol,
    value: object,
    tracker: _AllocationTracker,
) -> None:
    if tracker.owns(value):
        return
    try:
        if runtime.is_device_value(value):
            runtime.release(value)
    except Exception:
        pass


def _release_orphan_outputs(
    runtime: RuntimeProtocol,
    raw: object,
    tracker: _AllocationTracker,
) -> None:
    values = tuple(raw) if isinstance(raw, (tuple, list)) else (raw,)
    released: set[object] = set()
    for value in values:
        if tracker.owns(value):
            continue
        try:
            if not runtime.is_device_value(value):
                continue
            try:
                identity = runtime.allocation_identity(value)
            except Exception:
                identity = ("python-object", id(value))
            if identity in released:
                continue
            released.add(identity)
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
            "benchmark_runtime_exception_classification_failed",
            f"{type(exc).__name__}: {exc}; classifier failed with "
            f"{type(classification_error).__name__}: {classification_error}",
            exception_type=type(exc).__name__,
            retryable=False,
        )


def _combine_runtime_failures(
    first: RuntimeExceptionInfo | None,
    second: RuntimeExceptionInfo,
) -> RuntimeExceptionInfo:
    if first is None:
        return second
    return _combined_runtime_failure((first, second))


def _combined_runtime_failure(
    failures: Sequence[RuntimeExceptionInfo],
) -> RuntimeExceptionInfo | None:
    values = tuple(failures)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    detail = "; ".join(
        f"{failure.kind.value}/{failure.reason_code}: "
        f"{failure.message or failure.reason_code}"
        for failure in values
    )
    return RuntimeExceptionInfo(
        RuntimeExceptionKind.KERNEL_FAILURE,
        "benchmark_candidate_multiple_failures",
        detail,
        exception_type="",
        retryable=False,
    )


def _execute_cpu_reference(private_call: object) -> object:
    if not isinstance(private_call, PreparedNodeCall):
        raise TypeError("private benchmark input must be a PreparedNodeCall.")
    raw = private_call.cpu_function(
        private_call.positional_input(),
        **_cpu_reference_keyword_arguments(private_call),
    )
    outputs = _normalized_outputs(raw, private_call.output_port_count)
    return outputs[0] if private_call.output_port_count == 1 else outputs


def _benchmark_abort_callback(
    call: PreparedNodeCall,
) -> Callable[[], bool] | None:
    callback = call.kwargs.get(_BENCHMARK_ABORT_CALLBACK_KWARG)
    if callback is None:
        return None
    if not callable(callback):
        raise TypeError("benchmark-private abort callback must be callable.")
    return callback


def _provider_keyword_arguments(call: PreparedNodeCall) -> dict[str, object]:
    kwargs = {
        name: value
        for name, value in call.keyword_arguments().items()
        if not name.startswith("_vipp_")
    }
    kwargs.pop(_BENCHMARK_ABORT_CALLBACK_KWARG, None)
    return kwargs


def _cpu_reference_keyword_arguments(
    call: PreparedNodeCall,
) -> dict[str, object]:
    kwargs = call.keyword_arguments()
    kwargs.pop(_BENCHMARK_ABORT_CALLBACK_KWARG, None)
    return kwargs


def _normalized_outputs(raw: object, output_count: int) -> tuple[object, ...]:
    if output_count == 1:
        return (raw,)
    if not isinstance(raw, (tuple, list)):
        raise TypeError("multi-output operations must return a tuple or list.")
    outputs = tuple(raw)
    if len(outputs) != output_count:
        raise ValueError(
            f"Operation returned {len(outputs)} outputs; expected {output_count}."
        )
    return outputs


def _ensure_host_only(value: object, runtime: RuntimeProtocol) -> None:
    if runtime.is_device_value(value):
        raise TypeError("A device value escaped a benchmark candidate scope.")
    if isinstance(value, Mapping):
        for item in value.values():
            _ensure_host_only(item, runtime)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _ensure_host_only(item, runtime)
    elif is_dataclass(value) and not isinstance(value, type):
        for descriptor in fields(value):
            _ensure_host_only(getattr(value, descriptor.name), runtime)


def workload_from_prepared_node_call(
    call: PreparedNodeCall,
    *,
    check_abort: Callable[[], None] | None = None,
) -> WorkloadDescriptor:
    """Return the exact benchmark identity for one prepared production call.

    The facts fingerprint covers every input byte in addition to shape, dtype,
    layout, resolved parameters, and operation identity.  Application-facing
    coordinators use this helper before runtime probing so scientific/workload
    eligibility is evaluated against the same identity ultimately stored by
    :class:`~napari_vipp.core.compute_benchmark.NodeBenchmarkService`.
    """

    if not isinstance(call, PreparedNodeCall):
        raise TypeError("call must be a PreparedNodeCall.")
    if check_abort is not None and not callable(check_abort):
        raise TypeError("check_abort must be callable or None.")
    _run_abort_check(check_abort)
    arrays = tuple(np.asarray(value) for value in call.inputs)
    parameters = tuple(
        (name, _json_parameter(value))
        for name, value in call.kwargs.items()
        if name != "progress" and not name.startswith("_vipp_")
    )
    resolved = call.kwargs.get("resolved_spatial_ndim")
    if isinstance(resolved, bool) or resolved not in {1, 2, 3}:
        resolved = None
    return WorkloadDescriptor(
        node_id=call.node_id,
        operation_id=call.operation_id,
        input_shapes=tuple(
            tuple(int(size) for size in array.shape) for array in arrays
        ),
        input_dtypes=tuple(
            array.dtype.name if array.dtype.isnative else array.dtype.str
            for array in arrays
        ),
        parameters=parameters,
        resolved_spatial_ndim=resolved,
        facts_fingerprint=_call_facts_fingerprint(
            call,
            check_abort=check_abort,
        ),
    )


def _call_facts_fingerprint(
    call: PreparedNodeCall,
    *,
    check_abort: Callable[[], None] | None = None,
) -> str:
    digest = sha256()
    digest.update(call.operation_id.encode("utf-8"))
    digest.update(str(call.output_port_count).encode("ascii"))
    for value in call.inputs:
        _run_abort_check(check_abort)
        array = np.asarray(value)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(repr(tuple(array.shape)).encode("ascii"))
        digest.update(repr(tuple(array.strides)).encode("ascii"))
        digest.update(b"C" if array.flags.c_contiguous else b"-")
        digest.update(b"F" if array.flags.f_contiguous else b"-")
        _update_digest_from_array(
            digest,
            array,
            check_abort=check_abort,
        )
    return digest.hexdigest()


def _json_parameter(value: object) -> object:
    if isinstance(value, np.generic):
        return _json_parameter(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("benchmark parameters must not contain NaN or infinity.")
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_parameter(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_json_parameter(item) for item in value)
    raise TypeError(
        "benchmark parameters must be JSON-safe after progress is detached; "
        f"received {type(value).__name__}."
    )


def _clone_detached_call(
    call: PreparedNodeCall,
    *,
    check_abort: Callable[[], None] | None = None,
) -> PreparedNodeCall:
    return PreparedNodeCall(
        node_id=call.node_id,
        operation_id=call.operation_id,
        cpu_function=call.cpu_function,
        inputs=tuple(
            _detached_host_value(value, check_abort=check_abort)
            for value in call.inputs
        ),
        input_states=copy.deepcopy(call.input_states),
        kwargs=copy.deepcopy(call.keyword_arguments()),
        multiple_inputs=call.multiple_inputs,
        output_port_count=call.output_port_count,
    )


def _detached_host_value(
    value: object,
    *,
    check_abort: Callable[[], None] | None = None,
) -> object:
    if isinstance(value, np.ndarray):
        detached = np.empty_like(value, order="K", subok=False)
        iterator = np.nditer(
            (value, detached),
            flags=["buffered", "external_loop", "refs_ok", "zerosize_ok"],
            op_flags=(("readonly",), ("writeonly", "no_broadcast")),
            order="K",
            buffersize=262_144,
        )
        for source_chunk, destination_chunk in iterator:
            _run_abort_check(check_abort)
            destination_chunk[...] = source_chunk
        _run_abort_check(check_abort)
        detached.setflags(write=False)
        return detached
    return copy.deepcopy(value)


def _update_digest_from_array(
    digest,
    array: np.ndarray,
    *,
    check_abort: Callable[[], None] | None,
) -> None:
    iterator = np.nditer(
        array,
        flags=["buffered", "external_loop", "refs_ok", "zerosize_ok"],
        op_flags=(("readonly",),),
        order="C",
        buffersize=262_144,
    )
    for raw_chunk in iterator:
        _run_abort_check(check_abort)
        chunk = np.ascontiguousarray(raw_chunk).view(np.uint8).reshape(-1)
        digest.update(memoryview(chunk))
    _run_abort_check(check_abort)


def _run_abort_check(check_abort: Callable[[], None] | None) -> None:
    if check_abort is not None:
        check_abort()


def _exact_array_parity(reference: object, candidate: object) -> ParityResult:
    try:
        expected = np.asarray(reference)
        actual = np.asarray(candidate)
    except Exception as exc:
        return ParityResult(False, f"outputs are not host arrays: {exc}")
    mismatch = _array_contract_mismatch(expected, actual)
    if mismatch:
        return ParityResult(False, mismatch)
    expected_bytes = np.ascontiguousarray(expected).view(np.uint8).reshape(-1)
    actual_bytes = np.ascontiguousarray(actual).view(np.uint8).reshape(-1)
    byte_mismatches = int(np.count_nonzero(expected_bytes != actual_bytes))
    if byte_mismatches:
        signed_zero_mismatches = 0
        if np.issubdtype(expected.dtype, np.floating):
            both_zero = (expected == 0) & (actual == 0)
            signed_zero_mismatches = int(
                np.count_nonzero(
                    both_zero & (np.signbit(expected) != np.signbit(actual))
                )
            )
        return ParityResult(
            False,
            f"bitwise mismatch: {byte_mismatches} bytes differ; "
            f"signed_zero_mismatches={signed_zero_mismatches}",
        )
    return ParityResult(True, "bitwise exact, including signed-zero bits")


def _background_dtype_parity(
    reference: object,
    candidate: object,
    *,
    input_peak: float | None,
) -> ParityResult:
    try:
        expected = np.asarray(reference)
        actual = np.asarray(candidate)
    except Exception as exc:
        return ParityResult(False, f"outputs are not host arrays: {exc}")
    mismatch = _array_contract_mismatch(expected, actual)
    if mismatch:
        return ParityResult(False, mismatch)
    if np.issubdtype(expected.dtype, np.integer):
        return _exact_array_parity(expected, actual)
    if expected.dtype != np.dtype(np.float32):
        return ParityResult(
            False,
            f"Background GPU parity has no policy for {expected.dtype}",
        )

    expected_finite = np.isfinite(expected)
    actual_finite = np.isfinite(actual)
    if not np.array_equal(expected_finite, actual_finite):
        return ParityResult(False, "finite/non-finite masks differ")
    if not np.array_equal(np.isnan(expected), np.isnan(actual)):
        return ParityResult(False, "NaN masks differ")
    if not np.array_equal(np.isposinf(expected), np.isposinf(actual)):
        return ParityResult(False, "positive-infinity masks differ")
    if not np.array_equal(np.isneginf(expected), np.isneginf(actual)):
        return ParityResult(False, "negative-infinity masks differ")
    expected_zero = expected == 0
    actual_zero = actual == 0
    if not np.array_equal(expected_zero, actual_zero):
        return ParityResult(False, "zero masks differ")
    both_zero = expected_zero & actual_zero
    signed_zero_mismatches = int(
        np.count_nonzero(both_zero & (np.signbit(expected) != np.signbit(actual)))
    )
    if signed_zero_mismatches:
        return ParityResult(
            False,
            f"signed-zero bits differ at {signed_zero_mismatches} values",
        )

    expected_values = expected[expected_finite].astype(np.float64)
    actual_values = actual[actual_finite].astype(np.float64)
    difference = actual_values - expected_values
    max_abs = float(np.max(np.abs(difference))) if difference.size else 0.0
    reference_peak = (
        float(np.max(np.abs(expected_values))) if expected_values.size else 0.0
    )
    normalized_input_peak = _validated_optional_peak(input_peak)
    scale = max(1.0, normalized_input_peak, reference_peak)
    max_abs_limit = (
        BACKGROUND_FLOAT32_MAX_ABS_BASE
        + BACKGROUND_FLOAT32_SCALE_ULPS * np.finfo(np.float32).eps * scale
    )
    denominator = max(
        float(np.linalg.norm(expected_values)),
        math.sqrt(expected_values.size) * GAUSSIAN_FLOAT32_ABSOLUTE_FLOOR,
    )
    numerator = float(np.linalg.norm(difference))
    nrmse = numerator / denominator if denominator else 0.0
    max_ulp = _maximum_float32_ulp_distance(
        expected[expected_finite],
        actual[actual_finite],
    )
    passed = bool(nrmse <= BACKGROUND_FLOAT32_NRMSE_LIMIT and max_abs <= max_abs_limit)
    return ParityResult(
        passed,
        f"nrmse={nrmse:.9g} (limit={BACKGROUND_FLOAT32_NRMSE_LIMIT:.9g}); "
        f"max_abs={max_abs:.9g} (limit={max_abs_limit:.9g}); "
        f"max_ulp={max_ulp} (diagnostic; near-zero cancellation is "
        "absolute-error gated)",
    )


def _sigma_filter_dtype_parity(
    reference: object,
    candidate: object,
    *,
    input_peak: float | None,
) -> ParityResult:
    """Gate Sigma Filter output without hiding threshold-branch differences.

    Unsigned restoration is part of the scientific contract, so integer output
    must be bitwise exact.  Float32 output is allowed only a small accumulated
    arithmetic difference.  Adversarial selection and fallback decisions are
    validated separately by the implementation evidence suite; this aggregate
    gate is intentionally too tight to excuse a materially different branch.
    """

    try:
        expected = np.asarray(reference)
        actual = np.asarray(candidate)
    except Exception as exc:
        return ParityResult(False, f"outputs are not host arrays: {exc}")
    mismatch = _array_contract_mismatch(expected, actual)
    if mismatch:
        return ParityResult(False, mismatch)
    if expected.dtype in {np.dtype(np.uint8), np.dtype(np.uint16)}:
        return _exact_array_parity(expected, actual)
    if expected.dtype != np.dtype(np.float32):
        return ParityResult(
            False,
            f"Sigma Filter GPU parity has no policy for {expected.dtype}",
        )

    expected_finite = np.isfinite(expected)
    actual_finite = np.isfinite(actual)
    if not np.array_equal(expected_finite, actual_finite):
        return ParityResult(False, "finite/non-finite masks differ")
    if not bool(np.all(expected_finite)):
        return ParityResult(
            False,
            "Sigma Filter admitted float32 output must be completely finite",
        )
    expected_zero = expected == 0
    actual_zero = actual == 0
    if not np.array_equal(expected_zero, actual_zero):
        return ParityResult(False, "zero masks differ")
    both_zero = expected_zero & actual_zero
    signed_zero_mismatches = int(
        np.count_nonzero(both_zero & (np.signbit(expected) != np.signbit(actual)))
    )
    if signed_zero_mismatches:
        return ParityResult(
            False,
            f"signed-zero bits differ at {signed_zero_mismatches} values",
        )

    expected64 = expected.astype(np.float64)
    actual64 = actual.astype(np.float64)
    difference = actual64 - expected64
    max_abs = float(np.max(np.abs(difference))) if difference.size else 0.0
    reference_peak = float(np.max(np.abs(expected64))) if expected64.size else 0.0
    normalized_input_peak = _validated_optional_peak(input_peak)
    scale = max(1.0, normalized_input_peak, reference_peak)
    max_abs_limit = (
        SIGMA_FILTER_FLOAT32_MAX_ABS_BASE
        + SIGMA_FILTER_FLOAT32_SCALE_ULPS * np.finfo(np.float32).eps * scale
    )
    denominator = max(
        float(np.linalg.norm(expected64.ravel())),
        math.sqrt(expected64.size) * GAUSSIAN_FLOAT32_ABSOLUTE_FLOOR,
    )
    numerator = float(np.linalg.norm(difference.ravel()))
    nrmse = numerator / denominator if denominator else 0.0
    max_ulp = _maximum_float32_ulp_distance(expected, actual)
    passed = bool(
        nrmse <= SIGMA_FILTER_FLOAT32_NRMSE_LIMIT and max_abs <= max_abs_limit
    )
    return ParityResult(
        passed,
        f"nrmse={nrmse:.9g} "
        f"(limit={SIGMA_FILTER_FLOAT32_NRMSE_LIMIT:.9g}); "
        f"max_abs={max_abs:.9g} (limit={max_abs_limit:.9g}); "
        f"max_ulp={max_ulp} (diagnostic; threshold/fallback branch parity is "
        "validated separately)",
    )


def _gaussian_float32_parity(
    reference: object,
    candidate: object,
) -> ParityResult:
    try:
        expected = np.asarray(reference)
        actual = np.asarray(candidate)
    except Exception as exc:
        return ParityResult(False, f"outputs are not host arrays: {exc}")
    mismatch = _array_contract_mismatch(expected, actual)
    if mismatch:
        return ParityResult(False, mismatch)
    if expected.dtype != np.dtype(np.float32):
        return ParityResult(
            False,
            f"Gaussian production benchmark requires float32, got {expected.dtype}",
        )
    expected_finite = np.isfinite(expected)
    actual_finite = np.isfinite(actual)
    if not np.array_equal(expected_finite, actual_finite):
        return ParityResult(False, "finite/non-finite masks differ")
    if not bool(np.all(expected_finite)):
        return ParityResult(False, "Gaussian admitted region must be completely finite")
    expected64 = expected.astype(np.float64)
    actual64 = actual.astype(np.float64)
    difference = actual64 - expected64
    max_abs = float(np.max(np.abs(difference))) if difference.size else 0.0
    peak = float(np.max(np.abs(expected64))) if expected64.size else 0.0
    max_abs_limit = 1e-6 + 5e-6 * peak
    denominator = max(
        float(np.linalg.norm(expected64.ravel())),
        float(math.sqrt(expected64.size) * GAUSSIAN_FLOAT32_ABSOLUTE_FLOOR),
    )
    numerator = float(np.linalg.norm(difference.ravel()))
    nrmse = numerator / denominator if denominator else 0.0
    passed = bool(nrmse <= GAUSSIAN_FLOAT32_NRMSE_LIMIT and max_abs <= max_abs_limit)
    return ParityResult(
        passed,
        f"nrmse={nrmse:.9g} (limit={GAUSSIAN_FLOAT32_NRMSE_LIMIT:.9g}); "
        f"max_abs={max_abs:.9g} (limit={max_abs_limit:.9g})",
    )


def _array_contract_mismatch(expected: np.ndarray, actual: np.ndarray) -> str:
    if expected.shape != actual.shape:
        return f"shape differs: CPU {expected.shape}, candidate {actual.shape}"
    if expected.dtype != actual.dtype:
        return f"dtype differs: CPU {expected.dtype}, candidate {actual.dtype}"
    return ""


def _finite_input_peak(value: object) -> float:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or not array.size:
        return 0.0
    finite = np.isfinite(array)
    if not bool(np.any(finite)):
        return 0.0
    return float(np.max(np.abs(array[finite].astype(np.float64))))


def _validated_optional_peak(value: float | None) -> float:
    if value is None:
        return 0.0
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise ValueError("input_peak must be finite and non-negative or None.")
    return float(value)


def _maximum_float32_ulp_distance(
    expected: np.ndarray,
    actual: np.ndarray,
) -> int:
    if not expected.size:
        return 0

    def ordered(values: np.ndarray) -> np.ndarray:
        bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
        negative = (bits & np.uint32(0x80000000)) != 0
        return np.where(
            negative,
            np.uint64(0xFFFFFFFF) - bits.astype(np.uint64),
            np.uint64(0x80000000) + bits.astype(np.uint64),
        )

    expected_ordered = ordered(expected)
    actual_ordered = ordered(actual)
    distance = np.maximum(expected_ordered, actual_ordered) - np.minimum(
        expected_ordered,
        actual_ordered,
    )
    return int(np.max(distance))


def _read_clock(clock: Callable[[], float]) -> float:
    value = clock()
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError("benchmark adapter clock must return a finite number.")
    return float(value)


def _elapsed(clock: Callable[[], float], started: float) -> float:
    elapsed = _read_clock(clock) - started
    if elapsed < 0 or not math.isfinite(elapsed):
        raise ValueError("benchmark adapter clock must be monotonic.")
    return elapsed


__all__ = [
    "BACKGROUND_FLOAT32_MAX_ABS_BASE",
    "BACKGROUND_FLOAT32_NRMSE_LIMIT",
    "BACKGROUND_FLOAT32_SCALE_ULPS",
    "BACKGROUND_PARITY_OPERATION_IDS",
    "BENCHMARK_WRITER_OPERATION_IDS",
    "CUSTOM_BENCHMARK_POLICY_ID",
    "EXACT_PARITY_OPERATION_IDS",
    "EXACT_LABEL_PARITY_OPERATION_IDS",
    "GAUSSIAN_FLOAT32_ABSOLUTE_FLOOR",
    "GAUSSIAN_FLOAT32_NRMSE_LIMIT",
    "GAUSSIAN_PARITY_OPERATION_IDS",
    "PRODUCTION_BENCHMARK_POLICY_ID",
    "PIPELINE_SCREENING_BENCHMARK_POLICY_ID",
    "ProductionBenchmarkObservationLog",
    "ProductionInvocationObservation",
    "RegisteredNodeBenchmark",
    "RICHARDSON_LUCY_FLOAT32_ABSOLUTE_FLOOR",
    "RICHARDSON_LUCY_FLOAT32_MAX_ABS_BASE",
    "RICHARDSON_LUCY_FLOAT32_MAX_ABS_PEAK_FACTOR",
    "RICHARDSON_LUCY_FLOAT32_NEAR_IDENTITY_NRMSE_LIMIT",
    "RICHARDSON_LUCY_FLOAT32_NRMSE_LIMIT",
    "RICHARDSON_LUCY_PARITY_OPERATION_IDS",
    "RICHARDSON_LUCY_TV_FLOAT32_MAX_ABS_BASE",
    "RICHARDSON_LUCY_TV_FLOAT32_MAX_ABS_PEAK_FACTOR",
    "RICHARDSON_LUCY_TV_FLOAT32_NRMSE_LIMIT",
    "RICHARDSON_LUCY_TV_PARITY_OPERATION_IDS",
    "SIGMA_FILTER_FLOAT32_MAX_ABS_BASE",
    "SIGMA_FILTER_FLOAT32_NRMSE_LIMIT",
    "SIGMA_FILTER_FLOAT32_SCALE_ULPS",
    "SIGMA_FILTER_PARITY_OPERATION_IDS",
    "build_registered_node_benchmark",
    "detach_prepared_node_call",
    "operation_parity",
    "workload_from_prepared_node_call",
]
