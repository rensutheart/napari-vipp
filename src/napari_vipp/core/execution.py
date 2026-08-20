"""Headless contracts and service for one isolated pipeline execution.

The service detaches and validates the graph document. Input ownership is an
upstream source-boundary responsibility: callers must supply stable snapshots,
not mutable viewer arrays or live lazy stores.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import Enum
from hashlib import sha256
from itertools import count
from numbers import Integral
from os import PathLike, fspath
from time import perf_counter
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol
from weakref import WeakKeyDictionary

import numpy as np

from napari_vipp.core import metadata as _metadata
from napari_vipp.core.compute import (
    ComputeEnvironment,
    ComputeMode,
    ComputeRequest,
    DecisionKind,
    DecisionReason,
    ExactWorkloadCandidateQualification,
    ExecutionFallbackRecord,
    ExecutionPlan,
    ExecutionReport,
    FallbackReason,
    NodeComputePreference,
    NodeExecutionDecision,
    NodePreferenceKind,
    OutputPortKey,
    WorkloadDescriptor,
    canonical_digest,
)
from napari_vipp.core.compute_cache import (
    CachedNodeComputeProvenance,
    build_cached_node_compute_provenance,
    build_cached_source_provenance,
    cached_node_provenance_matches,
    cached_source_provenance_matches,
)
from napari_vipp.core.compute_history import (
    JsonPipelineTimingStore,
    PipelineTimingChoice,
    PipelineTimingSample,
    host_performance_fingerprint,
)
from napari_vipp.core.compute_policy import (
    OTSU_MAXIMUM_NATIVE_INTEGER_LEVELS,
    ArrayFacts,
    ArrayFactsCache,
    ArrayFactsKey,
    FactCompleteness,
    PerformanceEvidence,
    ValueDescriptor,
    evaluate_auto_performance,
    evaluate_candidate_support,
    evaluate_candidate_workload_support,
    propagate_output_descriptors,
)
from napari_vipp.core.host_memory import (
    capture_host_memory,
    preflight_host_allocation,
)
from napari_vipp.core.node_execution import PreparedNodeCall
from napari_vipp.core.pipeline import (
    EXECUTION_RUNNING,
    MANUAL_RUN_SKIP,
    PrototypePipeline,
    SourcePayload,
)
from napari_vipp.core.progress import OperationCancelled, ProgressContext
from napari_vipp.core.thumbnail_statistics import (
    EXACT_FLOAT32_MINMAX_GPU_ALGORITHM_ID,
    EXACT_FLOAT32_PERCENTILE_GPU_ALGORITHM_ID,
    ThumbnailStatisticsBackend,
    ThumbnailStatisticsCleanupError,
    ThumbnailStatisticsDecision,
    ThumbnailStatisticsResult,
)
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
_SCIENTIFIC_CONTEXT_CHUNK_BYTES = 8 * 1_048_576
_PHASE_ONE_FACT_OPERATIONS = frozenset(
    {
        "rolling_ball_background",
        "subtract_background",
        "gaussian_blur",
        "gaussian_blur_3d",
        "convert_dtype",
        "binary_threshold",
        "extract_channel",
        "median_filter",
        "sigma_filter",
        "canny_edges",
        "otsu_threshold",
        "fill_holes",
        "remove_small_objects",
        "label_connected_components",
        "prepare_validate_psf",
        "richardson_lucy_deconvolution",
        "richardson_lucy_tv_deconvolution",
    }
)
_EXACT_HOST_AXIS_CONTRACT_OPERATIONS = frozenset(
    {
        "split_channels",
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
        exact_workload_qualifications: frozenset[
            ExactWorkloadCandidateQualification
        ] = frozenset(),
        exact_workload_qualification_scope_digest: str = "",
    ) -> object: ...


@dataclass(frozen=True, slots=True)
class _ArrayDescription:
    """Shape/dtype-only value used while planning resident continuations."""

    shape: tuple[int, ...]
    dtype: np.dtype

    @property
    def ndim(self) -> int:
        return len(self.shape)


def _normalized_resident_thumbnail_contrast_mode(value: object) -> str:
    """Return the public thumbnail mode spelling used by observations."""

    text = str(value or "").strip().casefold()
    if text == "percentile":
        return "Percentile"
    if text in {
        "min-max",
        "minmax",
        "minimum-maximum",
        "minimum maximum",
    }:
        return "Min-max"
    if text == "raw":
        return "Raw"
    raise ValueError(
        "Resident thumbnail contrast_mode must be Percentile, Min-max, or Raw."
    )


@dataclass(frozen=True, slots=True)
class ResidentThumbnailStatisticsRequest:
    """One explicitly budgeted presentation request for a resident output."""

    node_id: str
    output_port: int = 0
    contrast_mode: str = "Percentile"
    minimum_scanned_bytes: int = 0
    gpu_contract_warm: bool = False

    def __post_init__(self) -> None:
        node_id = str(self.node_id).strip()
        if not node_id:
            raise ValueError("Resident thumbnail node_id must not be empty.")
        if (
            isinstance(self.output_port, bool)
            or not isinstance(self.output_port, Integral)
            or int(self.output_port) < 0
        ):
            raise ValueError(
                "Resident thumbnail output_port must be a non-negative integer."
            )
        if (
            isinstance(self.minimum_scanned_bytes, bool)
            or not isinstance(self.minimum_scanned_bytes, Integral)
            or int(self.minimum_scanned_bytes) < 0
        ):
            raise ValueError(
                "Resident thumbnail minimum_scanned_bytes must be a "
                "non-negative integer."
            )
        if not isinstance(self.gpu_contract_warm, bool):
            raise TypeError("Resident thumbnail gpu_contract_warm must be a boolean.")
        contrast_mode = _normalized_resident_thumbnail_contrast_mode(self.contrast_mode)
        object.__setattr__(self, "node_id", node_id)
        object.__setattr__(self, "output_port", int(self.output_port))
        object.__setattr__(
            self,
            "minimum_scanned_bytes",
            int(self.minimum_scanned_bytes),
        )
        object.__setattr__(self, "contrast_mode", contrast_mode)

    @property
    def port(self) -> OutputPortKey:
        """Return the exact graph output requested by the presentation layer."""

        return OutputPortKey(self.node_id, self.output_port)


@dataclass(frozen=True, slots=True)
class ResidentThumbnailStatisticsObservation:
    """Host-only thumbnail statistics captured while one output was resident."""

    node_id: str
    output_port: int
    contrast_mode: str
    result: ThumbnailStatisticsResult

    def __post_init__(self) -> None:
        node_id = str(self.node_id).strip()
        if not node_id:
            raise ValueError("Resident thumbnail observation node_id is required.")
        if (
            isinstance(self.output_port, bool)
            or not isinstance(self.output_port, Integral)
            or int(self.output_port) < 0
        ):
            raise ValueError(
                "Resident thumbnail observation output_port must be non-negative."
            )
        if not isinstance(self.result, ThumbnailStatisticsResult):
            raise TypeError(
                "Resident thumbnail observation result must be a "
                "ThumbnailStatisticsResult."
            )
        contrast_mode = _normalized_resident_thumbnail_contrast_mode(self.contrast_mode)
        if (
            self.result.input_path != "resident_borrow"
            or self.result.logical_input_host_to_device_bytes != 0
        ):
            raise ValueError(
                "Resident thumbnail observations must describe a zero-upload "
                "resident borrow."
            )
        immutable_limits = _immutable_resident_thumbnail_limits(self.result.limits)
        object.__setattr__(self, "node_id", node_id)
        object.__setattr__(self, "output_port", int(self.output_port))
        object.__setattr__(self, "contrast_mode", contrast_mode)
        object.__setattr__(
            self,
            "result",
            replace(self.result, limits=immutable_limits),
        )

    @property
    def port(self) -> OutputPortKey:
        """Return the exact graph output described by this observation."""

        return OutputPortKey(self.node_id, self.output_port)


def _immutable_resident_thumbnail_limits(limits):
    """Detach provider limits into immutable host scalar tuples."""

    if limits is None:
        return None
    values = np.asarray(limits, dtype=np.float64)
    if values.shape == (2,):
        return (float(values[0]), float(values[1]))
    if values.ndim == 2 and values.shape[1:] == (2,):
        return tuple((float(row[0]), float(row[1])) for row in values)
    raise ValueError(
        "Resident thumbnail limits must be one pair or a sequence of pairs."
    )


def _normalized_resident_thumbnail_observations(
    observations: Sequence[ResidentThumbnailStatisticsObservation],
    *,
    node_id: str | None = None,
) -> tuple[ResidentThumbnailStatisticsObservation, ...]:
    normalized = tuple(observations)
    if any(
        not isinstance(item, ResidentThumbnailStatisticsObservation)
        for item in normalized
    ):
        raise TypeError(
            "resident_thumbnail_statistics must contain "
            "ResidentThumbnailStatisticsObservation values."
        )
    if node_id is not None and any(item.node_id != node_id for item in normalized):
        raise ValueError(
            "A node result may carry resident thumbnail observations only for "
            "that node."
        )
    ordered = tuple(
        sorted(
            normalized,
            key=lambda item: (item.node_id, item.output_port),
        )
    )
    ports = tuple(item.port for item in ordered)
    if len(set(ports)) != len(ports):
        raise ValueError(
            "resident_thumbnail_statistics contains duplicate output ports."
        )
    return ordered


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
    cached_compute_provenance: Mapping[str, CachedNodeComputeProvenance] | None = field(
        default=None, repr=False, compare=False
    )
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
    exact_workload_qualifications: frozenset[ExactWorkloadCandidateQualification] = (
        field(default=frozenset(), repr=False, compare=False)
    )
    exact_workload_qualification_scope_digest: str = field(
        default="",
        repr=False,
        compare=False,
    )
    performance_history_path: str | PathLike[str] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    resident_thumbnail_statistics_request: ResidentThumbnailStatisticsRequest | None = (
        field(default=None, repr=False, compare=False)
    )

    def __post_init__(self) -> None:
        raw_provenance = self.cached_compute_provenance
        if raw_provenance is None:
            normalized_provenance: dict[str, CachedNodeComputeProvenance] = {}
        else:
            if not isinstance(raw_provenance, Mapping):
                raise TypeError("cached_compute_provenance must be a mapping or None.")
            normalized_provenance = {}
            for raw_node_id, provenance in raw_provenance.items():
                node_id = str(raw_node_id).strip()
                if not node_id:
                    raise ValueError(
                        "cached_compute_provenance node IDs must not be empty."
                    )
                if not isinstance(provenance, CachedNodeComputeProvenance):
                    raise TypeError(
                        "cached_compute_provenance values must be "
                        "CachedNodeComputeProvenance."
                    )
                if provenance.node_id != node_id:
                    raise ValueError(
                        "cached_compute_provenance keys must match record node IDs."
                    )
                if node_id in normalized_provenance:
                    raise ValueError(
                        "cached_compute_provenance contains duplicate normalized "
                        "node IDs."
                    )
                normalized_provenance[node_id] = provenance
        object.__setattr__(
            self,
            "cached_compute_provenance",
            MappingProxyType(dict(sorted(normalized_provenance.items()))),
        )

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
        qualifications = frozenset(self.exact_workload_qualifications)
        if any(
            not isinstance(item, ExactWorkloadCandidateQualification)
            for item in qualifications
        ):
            raise TypeError(
                "exact_workload_qualifications must contain "
                "ExactWorkloadCandidateQualification values."
            )
        qualification_scope = str(
            self.exact_workload_qualification_scope_digest
        ).strip()
        if qualifications and not qualification_scope:
            raise ValueError(
                "exact_workload_qualification_scope_digest is required when "
                "exact workload qualifications are supplied."
            )
        if any(
            item.qualification_scope_digest != qualification_scope
            for item in qualifications
        ):
            raise ValueError(
                "exact workload qualifications must match the request scope."
            )
        qualification_keys = tuple(item.candidate_key for item in qualifications)
        if len(set(qualification_keys)) != len(qualification_keys):
            raise ValueError(
                "exact_workload_qualifications contain duplicate candidate identities."
            )
        object.__setattr__(self, "exact_workload_qualifications", qualifications)
        object.__setattr__(
            self,
            "exact_workload_qualification_scope_digest",
            qualification_scope,
        )
        history_path = self.performance_history_path
        if history_path is not None:
            history_path = fspath(history_path).strip()
            if not history_path:
                raise ValueError("performance_history_path must not be blank.")
        object.__setattr__(self, "performance_history_path", history_path)
        resident_request = self.resident_thumbnail_statistics_request
        if resident_request is not None and not isinstance(
            resident_request,
            ResidentThumbnailStatisticsRequest,
        ):
            raise TypeError(
                "resident_thumbnail_statistics_request must be a "
                "ResidentThumbnailStatisticsRequest or None."
            )


@dataclass(frozen=True, slots=True)
class PipelineExecutionFailure:
    """Provider-neutral terminal details for a failed or cancelled run."""

    kind: str
    error_type: str
    message: str
    reason_code: str = ""
    segment_id: str = ""
    runtime_id: str = ""
    retryable: bool = False
    required_bytes: int | None = None
    available_bytes: int | None = None
    cleanup_succeeded: bool | None = None
    fallback_records: tuple[ExecutionFallbackRecord, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "kind",
            "error_type",
            "message",
            "reason_code",
            "segment_id",
            "runtime_id",
        ):
            object.__setattr__(self, name, str(getattr(self, name)).strip())
        if not self.kind or not self.error_type or not self.message:
            raise ValueError("Failure kind, error_type, and message must not be empty.")
        if not isinstance(self.retryable, bool):
            raise TypeError("retryable must be a boolean.")
        for name in ("required_bytes", "available_bytes"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer or None.")
        if self.cleanup_succeeded is not None and not isinstance(
            self.cleanup_succeeded,
            bool,
        ):
            raise TypeError("cleanup_succeeded must be a boolean or None.")
        object.__setattr__(self, "fallback_records", tuple(self.fallback_records))
        if any(
            not isinstance(record, ExecutionFallbackRecord)
            for record in self.fallback_records
        ):
            raise TypeError(
                "fallback_records must contain ExecutionFallbackRecord values."
            )

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "kind": self.kind,
            "error_type": self.error_type,
            "message": self.message,
            "retryable": self.retryable,
        }
        for name in ("reason_code", "segment_id", "runtime_id"):
            value = getattr(self, name)
            if value:
                result[name] = value
        for name in ("required_bytes", "available_bytes", "cleanup_succeeded"):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        if self.fallback_records:
            result["fallback_records"] = [
                record.as_dict() for record in self.fallback_records
            ]
        return result


class AcceleratorCleanupError(RuntimeError):
    """A primary accelerated failure followed by failed registry cleanup."""

    cleanup_succeeded = False

    def __init__(
        self,
        primary_error: BaseException,
        cleanup_error: BaseException,
    ) -> None:
        self.primary_error = primary_error
        self.primary_error_type = type(primary_error).__name__
        self.cleanup_error_type = type(cleanup_error).__name__
        self.fallback_records = tuple(
            getattr(primary_error, "fallback_records", ())
            or getattr(primary_error, "vipp_fallback_records", ())
        )
        super().__init__(
            "Accelerator cleanup failed while handling "
            f"{self.primary_error_type}: {cleanup_error}"
        )


class ResidentThumbnailStatisticsCleanupError(RuntimeError):
    """A resident presentation scan could not release its GPU scratch."""

    cleanup_succeeded = False


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
    failure: PipelineExecutionFailure | None = None
    resident_thumbnail_statistics: tuple[
        ResidentThumbnailStatisticsObservation,
        ...,
    ] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "resident_thumbnail_statistics",
            _normalized_resident_thumbnail_observations(
                self.resident_thumbnail_statistics
            ),
        )


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
    resident_thumbnail_statistics: tuple[
        ResidentThumbnailStatisticsObservation,
        ...,
    ] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "resident_thumbnail_statistics",
            _normalized_resident_thumbnail_observations(
                self.resident_thumbnail_statistics,
                node_id=self.node_id,
            ),
        )


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
    raise_errors: bool = False,
) -> PipelineRunResult:
    """Execute ``request`` without Qt and return errors as typed results.

    ``raise_errors`` exists for generated-library compatibility: it re-raises
    the same scientific exception only after the shared execution/cleanup
    boundary has classified it. Interactive and batch callers retain the
    default detached-result contract.
    """
    if not isinstance(raise_errors, bool):
        raise TypeError("raise_errors must be a boolean.")
    run_started: float | None = None
    observer_seconds = 0.0
    pipeline: PrototypePipeline | None = None
    execution_report: ExecutionReport | None = None
    timing_store: JsonPipelineTimingStore | None = None
    timing_workload_fingerprint = ""
    timing_host_environment_fingerprint = ""
    timing_execution_surface = ""
    timing_runnable_node_ids: frozenset[str] = frozenset()
    timing_warnings: list[str] = []
    resident_thumbnail_statistics: list[ResidentThumbnailStatisticsObservation] = []
    resident_thumbnail_observer_seconds = [0.0]

    def call_observer(callback, *args) -> None:
        nonlocal observer_seconds
        if callback is None:
            return
        observer_started = perf_counter()
        try:
            callback(*args)
        finally:
            observer_seconds += max(0.0, perf_counter() - observer_started)

    observed_node_started_callback = (
        None
        if node_started_callback is None
        else lambda node_id: call_observer(node_started_callback, node_id)
    )
    observed_node_finished_callback = (
        None
        if node_finished_callback is None
        else lambda result: call_observer(node_finished_callback, result)
    )
    observed_progress_callback = (
        None
        if progress_callback is None
        else lambda operation_id, completed, total, message: call_observer(
            progress_callback,
            operation_id,
            completed,
            total,
            message,
        )
    )
    try:
        workflow = deserialize_workflow(deepcopy(request.workflow))
        pipeline = PrototypePipeline()
        pipeline.restore_graph(
            workflow["nodes"],
            workflow["connections"],
            workflow.get("output_tunnels", ()),
        )
        cancel_callback = (
            request.cancel_event.is_set if request.cancel_event is not None else None
        )
        source_scientific_contexts = _capture_source_scientific_contexts(
            pipeline,
            request,
            cancel_callback=cancel_callback,
        )
        registered_specs = tuple(getattr(compute_registry, "implementation_specs", ()))
        _hydrate_cached_pipeline_outputs(
            pipeline,
            request,
            implementation_specs=registered_specs,
            source_scientific_contexts=source_scientific_contexts,
            cancel_callback=cancel_callback,
        )
        timing_schedule = pipeline.plan_execution(
            request.dirty_node_ids,
            manual_mode=MANUAL_RUN_SKIP,
            manual_node_ids=request.manual_node_ids,
            target_node_ids=request.target_node_ids,
        )
        timing_runnable_node_ids = frozenset(
            node_id
            for node_id in timing_schedule.runnable_node_ids
            if pipeline.operation_spec(pipeline.nodes[node_id].operation_id).has_input
        )
        if request.performance_history_path is not None:
            try:
                timing_workload_fingerprint = _pipeline_timing_workload_fingerprint(
                    pipeline,
                    timing_runnable_node_ids,
                    retain_node_ids=request.retain_node_ids,
                    prune_unretained=request.prune_unretained,
                    manual_node_ids=request.manual_node_ids,
                    target_node_ids=request.target_node_ids,
                    compute_request=request.compute_request,
                    source_scientific_contexts=source_scientific_contexts,
                    cancel_callback=cancel_callback,
                )
                if timing_workload_fingerprint:
                    timing_store = JsonPipelineTimingStore(
                        request.performance_history_path
                    )
                    timing_host_environment_fingerprint = host_performance_fingerprint()
                    timing_execution_surface = (
                        "direct-cpu-v1"
                        if request.compute_request.mode is ComputeMode.CPU
                        else (
                            "planned-borrowed-registry-v1"
                            if compute_registry is not None
                            else "planned-owned-registry-v1"
                        )
                    )
            except OperationCancelled:
                raise
            except Exception as exc:
                timing_store = None
                timing_workload_fingerprint = ""
                timing_host_environment_fingerprint = ""
                timing_execution_surface = ""
                timing_warnings.append(
                    "VIPP could not prepare local completed-run timing history; "
                    f"this run will continue without it ({exc})."
                )

        def publish_node_result(node_id: str) -> None:
            if observed_node_finished_callback is None:
                return
            node = pipeline.nodes[node_id]
            observed_node_finished_callback(
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
                    resident_thumbnail_statistics=tuple(
                        item
                        for item in resident_thumbnail_statistics
                        if item.node_id == node_id
                    ),
                )
            )

        run_started = perf_counter() if timing_store is not None else None
        if request.compute_request.mode is ComputeMode.CPU:
            try:
                pipeline.run(
                    request.input_data,
                    input_metadata=request.input_metadata,
                    input_name=request.input_name,
                    source_payloads=request.source_payloads,
                    dirty_node_ids=request.dirty_node_ids,
                    node_started_callback=observed_node_started_callback,
                    node_finished_callback=publish_node_result,
                    progress_callback=observed_progress_callback,
                    cancel_callback=cancel_callback,
                    manual_mode=MANUAL_RUN_SKIP,
                    manual_node_ids=request.manual_node_ids,
                    target_node_ids=request.target_node_ids,
                    retain_node_ids=request.retain_node_ids,
                    prune_unretained=request.prune_unretained,
                )
            except OperationCancelled:
                raise
            except Exception:
                # CPU nodes commit atomically one at a time.  Preserve exact
                # provenance and backend decisions for the prefix that really
                # completed so an interactive caller can safely publish those
                # sibling results even though a later node failed.
                completed_cpu_node_ids = set(timing_schedule.runnable_node_ids) & set(
                    pipeline.completed_node_ids
                )
                try:
                    partial_decisions = _publish_cpu_compute_provenance(
                        pipeline,
                        request,
                        completed_cpu_node_ids,
                        source_scientific_contexts=source_scientific_contexts,
                    )
                except Exception:
                    # Provenance publication is presentation/cache metadata and
                    # must never replace the authoritative scientific failure.
                    execution_report = None
                else:
                    execution_report = ExecutionReport(
                        request=request.compute_request,
                        environment=ComputeEnvironment(),
                        actual_decisions=partial_decisions,
                    )
                raise
            actual_decisions = _publish_cpu_compute_provenance(
                pipeline,
                request,
                timing_schedule.runnable_node_ids,
                source_scientific_contexts=source_scientific_contexts,
            )
            execution_report = ExecutionReport(
                request=request.compute_request,
                environment=ComputeEnvironment(),
                actual_decisions=actual_decisions,
            )
        else:
            execution_report = _execute_accelerated_pipeline(
                pipeline,
                request,
                node_started_callback=observed_node_started_callback,
                node_finished_callback=publish_node_result,
                progress_callback=observed_progress_callback,
                resident_progress_callback=progress_callback,
                cancel_callback=cancel_callback,
                compute_registry=compute_registry,
                compute_planner=compute_planner,
                array_facts_cache=(
                    request.array_facts_cache
                    if array_facts_cache is None
                    else array_facts_cache
                ),
                source_scientific_contexts=source_scientific_contexts,
                timing_store=timing_store,
                timing_workload_fingerprint=timing_workload_fingerprint,
                timing_host_environment_fingerprint=(
                    timing_host_environment_fingerprint
                ),
                timing_execution_surface=timing_execution_surface,
                resident_thumbnail_statistics=resident_thumbnail_statistics,
                resident_observer_seconds=resident_thumbnail_observer_seconds,
            )
            observer_seconds += resident_thumbnail_observer_seconds[0]
    except OperationCancelled as exc:
        failure = _pipeline_execution_failure(
            exc,
            cancelled=True,
            cpu_only=request.compute_request.mode is ComputeMode.CPU,
        )
        if raise_errors:
            _attach_pipeline_execution_failure(exc, failure)
            raise
        return PipelineRunResult(
            request.run_id,
            request.workflow,
            error=str(exc),
            cancelled=True,
            source_revisions=request.source_revisions,
            failure=failure,
        )
    except Exception as exc:
        failure = _pipeline_execution_failure(exc)
        if raise_errors:
            _attach_pipeline_execution_failure(exc, failure)
            raise
        return PipelineRunResult(
            request.run_id,
            request.workflow,
            pipeline=pipeline,
            error=str(exc),
            cancelled=failure.kind == "cancelled",
            source_revisions=request.source_revisions,
            execution_report=execution_report,
            failure=failure,
        )
    assert execution_report is not None
    if timing_warnings:
        execution_report = replace(
            execution_report,
            warnings=execution_report.warnings + tuple(timing_warnings),
        )
    timing_decisions = tuple(
        decision
        for decision in execution_report.actual_decisions
        if decision.node_id in timing_runnable_node_ids
    )
    if (
        timing_store is not None
        and timing_workload_fingerprint
        and timing_runnable_node_ids
        and run_started is not None
        and execution_report.cleanup_succeeded
        and not execution_report.fallback_records
        and not any(decision.fallback_used for decision in timing_decisions)
        and not any(
            decision.reason is DecisionReason.HISTORICAL_PERFORMANCE
            for decision in timing_decisions
        )
        and {decision.node_id for decision in timing_decisions}
        == set(timing_runnable_node_ids)
    ):
        try:
            timing_store.append(
                PipelineTimingSample.completed_run(
                    workload_fingerprint=timing_workload_fingerprint,
                    host_environment_fingerprint=(timing_host_environment_fingerprint),
                    environment=execution_report.environment,
                    decisions=timing_decisions,
                    elapsed_seconds=max(
                        perf_counter() - run_started - observer_seconds,
                        1e-12,
                    ),
                    requested_mode=request.compute_request.mode,
                    execution_surface=timing_execution_surface,
                )
            )
            if any(
                decision.reason is DecisionReason.PERFORMANCE_EXPLORATION
                for decision in timing_decisions
            ):
                learned_choice = timing_store.choose(
                    workload_fingerprint=timing_workload_fingerprint,
                    host_environment_fingerprint=(timing_host_environment_fingerprint),
                    accelerator_environment_fingerprint=(
                        execution_report.environment.fingerprint
                    ),
                    execution_surface=timing_execution_surface,
                )
                if learned_choice is not None:
                    learned_label = (
                        "an accelerated assignment"
                        if learned_choice.uses_accelerator
                        else "CPU"
                    )
                    execution_report = replace(
                        execution_report,
                        warnings=execution_report.warnings
                        + (
                            "Auto completed its one-time CPU comparison and "
                            f"learned {learned_label} for the next exact matching "
                            "run.",
                        ),
                    )
        except Exception as exc:
            execution_report = replace(
                execution_report,
                warnings=execution_report.warnings
                + (
                    "VIPP completed the run but could not save its local timing "
                    f"history ({exc}).",
                ),
            )
    return PipelineRunResult(
        request.run_id,
        request.workflow,
        pipeline,
        source_revisions=request.source_revisions,
        execution_report=execution_report,
        resident_thumbnail_statistics=tuple(resident_thumbnail_statistics),
    )


def _pipeline_execution_failure(
    exc: BaseException,
    *,
    cancelled: bool = False,
    cpu_only: bool = False,
) -> PipelineExecutionFailure:
    """Detach stable terminal facts without importing an optional provider."""

    error_type = type(exc).__name__
    message = str(exc).strip() or error_type
    cleanup = getattr(exc, "cleanup_succeeded", None)
    if cleanup is not None:
        cleanup = bool(cleanup)
    fallback_records = tuple(
        getattr(exc, "fallback_records", ())
        or getattr(exc, "vipp_fallback_records", ())
    )
    if isinstance(exc, AcceleratorCleanupError):
        primary = _pipeline_execution_failure(
            exc.primary_error,
            cancelled=isinstance(exc.primary_error, OperationCancelled),
            cpu_only=False,
        )
        return replace(
            primary,
            error_type=type(exc).__name__,
            message=message,
            cleanup_succeeded=False,
            fallback_records=fallback_records or primary.fallback_records,
        )
    if cancelled:
        return PipelineExecutionFailure(
            kind="cancelled",
            error_type=error_type,
            message=message,
            reason_code="operation_cancelled",
            cleanup_succeeded=True if cpu_only else cleanup,
            fallback_records=fallback_records,
        )

    if error_type == "ComputePreflightError":
        return PipelineExecutionFailure(
            kind="compute_preflight",
            error_type=error_type,
            message=message,
            reason_code="compute_preflight_rejected",
            cleanup_succeeded=True,
            fallback_records=fallback_records,
        )

    if isinstance(exc, MemoryError):
        snapshot = capture_host_memory()
        available_candidates = tuple(
            value
            for value in (
                snapshot.physical_available_bytes,
                snapshot.commit_available_bytes,
            )
            if value is not None
        )
        return PipelineExecutionFailure(
            kind="host_memory_oom",
            error_type=error_type,
            message=message,
            reason_code="host_allocation_failed",
            available_bytes=(
                min(available_candidates) if available_candidates else None
            ),
            cleanup_succeeded=True if cpu_only else cleanup,
            fallback_records=fallback_records,
        )

    failure = getattr(exc, "failure", None)
    failure_kind = getattr(getattr(failure, "kind", None), "value", "")
    if failure is not None and failure_kind:
        return PipelineExecutionFailure(
            kind=str(failure_kind),
            error_type=str(getattr(failure, "exception_type", "")).strip()
            or error_type,
            message=str(getattr(failure, "message", "")).strip() or message,
            reason_code=str(getattr(failure, "reason_code", "")).strip(),
            segment_id=str(getattr(exc, "segment_id", "")).strip(),
            runtime_id=str(getattr(exc, "runtime_id", "")).strip(),
            retryable=bool(getattr(failure, "retryable", False)),
            cleanup_succeeded=cleanup,
            fallback_records=fallback_records,
        )

    required = getattr(exc, "required_bytes", None)
    available = getattr(exc, "available_bytes", None)
    runtime_id = str(getattr(exc, "runtime_id", "")).strip()
    segment_id = str(getattr(exc, "segment_id", "")).strip()
    if required is not None or available is not None:
        return PipelineExecutionFailure(
            kind="memory_preflight",
            error_type=error_type,
            message=message,
            reason_code="device_memory_preflight",
            segment_id=segment_id,
            runtime_id=runtime_id,
            required_bytes=required,
            available_bytes=available,
            cleanup_succeeded=True,
            fallback_records=fallback_records,
        )
    return PipelineExecutionFailure(
        kind="execution_error",
        error_type=error_type,
        message=message,
        reason_code="unclassified_execution_error",
        cleanup_succeeded=cleanup,
        fallback_records=fallback_records,
    )


def _attach_pipeline_execution_failure(
    exc: BaseException,
    failure: PipelineExecutionFailure,
) -> None:
    """Best-effort JSON-safe detail on a compatibility re-raised exception."""

    try:
        exc.vipp_execution_failure = failure.as_dict()
    except Exception:
        return


def _execute_accelerated_pipeline(
    pipeline: PrototypePipeline,
    request: PipelineRunRequest,
    *,
    node_started_callback: NodeStartedCallback | None,
    node_finished_callback: Callable[[str], None] | None,
    progress_callback: ProgressCallback | None,
    resident_progress_callback: ProgressCallback | None,
    cancel_callback: Callable[[], bool] | None,
    compute_registry: ComputeRegistry | None,
    compute_planner: ComputePlanner | None,
    array_facts_cache: ArrayFactsCache | None,
    source_scientific_contexts: Mapping[str, str],
    timing_store: JsonPipelineTimingStore | None,
    timing_workload_fingerprint: str,
    timing_host_environment_fingerprint: str,
    timing_execution_surface: str,
    resident_thumbnail_statistics: list[ResidentThumbnailStatisticsObservation],
    resident_observer_seconds: list[float],
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
    active_error: BaseException | None = None
    history_warnings: list[str] = []
    try:
        # A request cancelled before execution starts must not publish even its
        # source boundary through the incremental-result callback.  Source
        # publication remains intentionally early for real planning failures.
        _check_fact_scan_cancelled(cancel_callback)
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
        effective_compute_request = request.compute_request
        timing_choice: PipelineTimingChoice | None = None
        timing_cpu_exploration = False
        if (
            request.compute_request.mode is ComputeMode.AUTO
            and timing_store is not None
            and timing_workload_fingerprint
            and timing_host_environment_fingerprint
            and timing_execution_surface
        ):
            try:
                candidate_choice = timing_store.choose(
                    workload_fingerprint=timing_workload_fingerprint,
                    host_environment_fingerprint=(timing_host_environment_fingerprint),
                    accelerator_environment_fingerprint=(
                        preflight_environment.fingerprint
                    ),
                    execution_surface=timing_execution_surface,
                )
                coverage = timing_store.coverage(
                    workload_fingerprint=timing_workload_fingerprint,
                    host_environment_fingerprint=(timing_host_environment_fingerprint),
                    accelerator_environment_fingerprint=(
                        preflight_environment.fingerprint
                    ),
                    execution_surface=timing_execution_surface,
                )
            except Exception as exc:
                history_warnings.append(
                    "VIPP could not reuse local completed-run timing history; "
                    f"reviewed Auto defaults remain active ({exc})."
                )
            else:
                historical_request = _historical_auto_compute_request(
                    request.compute_request,
                    candidate_choice,
                    workloads,
                    registry,
                )
                if candidate_choice is not None:
                    timing_choice = candidate_choice
                    effective_compute_request = historical_request
                elif coverage.needs_cpu_exploration:
                    required_host_bytes = _estimate_auto_cpu_exploration_peak_bytes(
                        workloads
                    )
                    memory_preflight = preflight_host_allocation(
                        capture_host_memory(),
                        required_bytes=required_host_bytes,
                        purpose="Auto CPU timing comparison",
                    )
                    if memory_preflight.allowed:
                        effective_compute_request = (
                            _auto_cpu_exploration_compute_request(
                                request.compute_request,
                                workloads,
                            )
                        )
                        timing_cpu_exploration = True
                    else:
                        history_warnings.append(
                            f"{memory_preflight.reason} Auto kept its reviewed "
                            "safe assignment; the missing CPU timing can be "
                            "collected later when memory headroom is sufficient."
                        )
        planning = planner(
            effective_compute_request,
            workloads,
            registry=registry,
            environment=preflight_environment,
            array_facts=array_facts,
            performance_evidence=request.performance_evidence,
            exact_workload_qualifications=request.exact_workload_qualifications,
            exact_workload_qualification_scope_digest=(
                request.exact_workload_qualification_scope_digest
            ),
        )
        if timing_choice is not None:
            historical_planning = _mark_historical_auto_planning(
                planning,
                original_request=request.compute_request,
                choice=timing_choice,
            )
            if historical_planning is None:
                timing_choice = None
                effective_compute_request = request.compute_request
                planning = planner(
                    effective_compute_request,
                    workloads,
                    registry=registry,
                    environment=preflight_environment,
                    array_facts=array_facts,
                    performance_evidence=request.performance_evidence,
                    exact_workload_qualifications=(
                        request.exact_workload_qualifications
                    ),
                    exact_workload_qualification_scope_digest=(
                        request.exact_workload_qualification_scope_digest
                    ),
                )
            else:
                planning = historical_planning
        elif timing_cpu_exploration:
            planning = _mark_auto_cpu_exploration_planning(
                planning,
                original_request=request.compute_request,
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
            effective_compute_request,
            dirty_node_ids=request.dirty_node_ids,
            manual_mode=MANUAL_RUN_SKIP,
            manual_node_ids=request.manual_node_ids,
            target_node_ids=request.target_node_ids,
            retained_ports=retained_ports,
        )
        resident_request = request.resident_thumbnail_statistics_request
        if (
            request.compute_request.mode is not ComputeMode.PREFER_GPU
            or resident_request is None
            or not resident_request.gpu_contract_warm
        ):
            resident_request = None
        pending_resident_statistics: dict[
            OutputPortKey,
            ResidentThumbnailStatisticsObservation,
        ] = {}
        resident_cleanup_failure_message: str | None = None
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
            implementation = None
            if runtime_id != CPU_RUNTIME_ID:
                decision = decisions_by_node.get(node_id)
                if decision is None:
                    raise RuntimeError(
                        f"Device node {node_id!r} has no captured compute decision."
                    )
                implementation = registry.implementation_spec(
                    decision.implementation_id,
                    allow_experimental=request.compute_request.allow_experimental,
                )
            if runtime_id == CPU_RUNTIME_ID or (
                implementation is not None
                and str(getattr(implementation, "host_finalizer_ref", "")).strip()
            ):
                # Device host-finalizer callbacks arrive only after D2H and
                # complete runtime cleanup, and already carry the public host
                # value (for example TableData).  Use the same authoritative
                # metadata finalization path as CPU output.
                raw_output: object = (
                    outputs[0] if call.output_port_count == 1 else outputs
                )
                results = pipeline.finalize_node_call(call, raw_output)
                states = tuple(state for _value, state in results)
            else:
                assert implementation is not None
                states = _predict_device_node_states(
                    pipeline,
                    call,
                    implementation,
                    outputs,
                )
            for port_index, state in enumerate(states):
                state_by_port[OutputPortKey(node_id, port_index)] = state

        def observe_resident_output(
            port: OutputPortKey,
            device_value: object,
            runtime: object,
            device_id: str,
        ) -> None:
            nonlocal resident_cleanup_failure_message
            resident_started = perf_counter()
            try:
                if resident_request is None or port != resident_request.port:
                    return
                output_ports = pipeline.output_ports(port.node_id)
                if not 0 <= port.port_index < len(output_ports):
                    return
                operation_id = pipeline.nodes[port.node_id].operation_id
                reporter = (
                    None
                    if resident_progress_callback is None
                    else lambda update: resident_progress_callback(
                        operation_id,
                        update.current,
                        update.total,
                        update.message,
                    )
                )
                try:
                    observation = _resident_thumbnail_statistics_observation(
                        resident_request,
                        compute_mode=request.compute_request.mode,
                        output_type=output_ports[port.port_index].output_type,
                        output_state=state_by_port.get(port),
                        device_value=device_value,
                        runtime=runtime,
                        device_id=device_id,
                        progress=ProgressContext(
                            cancelled=cancel_callback,
                            reporter=reporter,
                        ),
                    )
                except ResidentThumbnailStatisticsCleanupError as exc:
                    # Store only detached text.  The provider exception/traceback
                    # must not outlive the active private allocator scope.
                    resident_cleanup_failure_message = str(exc)
                    raise
                if observation is not None:
                    pending_resident_statistics[port] = observation
            finally:
                resident_observer_seconds[0] += max(
                    0.0,
                    perf_counter() - resident_started,
                )

        try:
            device_result = execute_device_plan(
                device_plan,
                pipeline,
                registry,
                effective_compute_request,
                host_values=host_values,
                prepare_call=prepare_call,
                cancel_callback=cancel_callback,
                node_outputs_callback=observe_outputs,
                resident_output_callback=(
                    observe_resident_output if resident_request is not None else None
                ),
            )
        except BaseException:
            if resident_cleanup_failure_message is not None:
                raise ResidentThumbnailStatisticsCleanupError(
                    resident_cleanup_failure_message
                ) from None
            raise

        fallback_node_ids = {
            node_id
            for segment in device_plan.segments
            if segment.segment_id in device_result.fallback_segment_ids
            for node_id in segment.node_ids
        }
        resident_thumbnail_statistics.extend(
            observation
            for port, observation in sorted(
                pending_resident_statistics.items(),
                key=lambda item: (item[0].node_id, item[0].port_index),
            )
            if port.node_id not in fallback_node_ids
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
        warnings.extend(history_warnings)
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
            fallback_records=device_result.fallback_records,
            warnings=tuple(warnings),
            cleanup_succeeded=device_result.cleanup_succeeded,
        )
        _publish_actual_compute_provenance(
            pipeline,
            request.compute_request,
            actual_decisions,
            implementation_specs=registry.implementation_specs,
            source_scientific_contexts=source_scientific_contexts,
            cancel_callback=cancel_callback,
        )
    except BaseException as exc:
        active_error = exc
        raise
    finally:
        if owned_registry:
            try:
                registry.close()
            except Exception as cleanup_error:
                closed_cleanly = False
                if active_error is not None:
                    raise AcceleratorCleanupError(
                        active_error,
                        cleanup_error,
                    ) from active_error
    if not closed_cleanly:
        report = replace(
            report,
            warnings=report.warnings
            + ("The accelerator registry did not close cleanly.",),
            cleanup_succeeded=False,
        )
    return report


def _resident_thumbnail_statistics_observation(
    request: ResidentThumbnailStatisticsRequest,
    *,
    compute_mode: ComputeMode,
    output_type: str,
    output_state: object,
    device_value: object,
    runtime: object,
    device_id: str,
    progress: ProgressContext,
) -> ResidentThumbnailStatisticsObservation | None:
    """Softly observe one eligible borrowed float32 output on CUDA/CuPy."""

    if (
        compute_mode is not ComputeMode.PREFER_GPU
        or not request.gpu_contract_warm
        or request.contrast_mode == "Raw"
        or str(output_type).strip().casefold() not in {"image", "array", "any"}
        or str(getattr(runtime, "runtime_id", "")).strip() != "cuda-cupy"
        or not isinstance(output_state, _metadata.ImageState)
    ):
        return None
    shape = _resident_device_shape(device_value)
    if shape is None or tuple(output_state.shape) != shape:
        return None
    try:
        device_dtype = np.dtype(device_value.dtype)
        state_dtype = np.dtype(output_state.dtype)
        actual_nbytes = int(device_value.nbytes)
        scanned_values = int(device_value.size)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
    if (
        device_dtype != np.dtype(np.float32)
        or state_dtype != device_dtype
        or actual_nbytes < request.minimum_scanned_bytes
        or actual_nbytes < 0
        or scanned_values < 0
        or actual_nbytes != scanned_values * device_dtype.itemsize
        or not _resident_image_state_is_presentable(output_state)
    ):
        return None
    channel_contract = _explicit_resident_channel_axis(output_state, len(shape))
    if channel_contract is None:
        return None
    channel_axis = channel_contract[0]
    started = perf_counter()
    try:
        provider_result = _exact_float32_thumbnail_limits_from_device(
            runtime,
            device_value,
            device_id=str(device_id).strip(),
            channel_axis=channel_axis,
            contrast_mode=request.contrast_mode,
            progress=progress,
        )
        auxiliary_host_to_device_bytes = _nonnegative_resident_count(
            provider_result.auxiliary_host_to_device_bytes,
            "auxiliary_host_to_device_bytes",
        )
        device_to_host_bytes = _nonnegative_resident_count(
            provider_result.device_to_host_bytes,
            "device_to_host_bytes",
        )
        device_to_host_values = _nonnegative_resident_count(
            provider_result.device_to_host_values,
            "device_to_host_values",
        )
        limits = _immutable_resident_thumbnail_limits(provider_result.limits)
        _validate_resident_thumbnail_limit_channels(
            limits,
            channel_axis=channel_axis,
            shape=shape,
        )
    except OperationCancelled:
        raise
    except Exception as exc:
        if (
            isinstance(exc, ThumbnailStatisticsCleanupError)
            or getattr(
                exc,
                "cleanup_succeeded",
                None,
            )
            is False
        ):
            raise ResidentThumbnailStatisticsCleanupError(
                "Resident thumbnail GPU scratch cleanup failed; restart "
                "accelerator work before retrying."
            ) from None
        # Presentation is optional.  Dependency/import/provider failures are
        # a soft miss and must not replace a valid scientific result.
        return None
    decision = ThumbnailStatisticsDecision(
        backend=ThumbnailStatisticsBackend.GPU_CUPY,
        reason_code="resident_float32_warm_contract",
        reason=(
            "A requested warm float32 thumbnail scan reused the current "
            "CUDA/CuPy output without uploading it again."
        ),
        scanned_values=scanned_values,
        scanned_bytes=actual_nbytes,
        threshold_bytes=request.minimum_scanned_bytes,
        gpu_warm=request.gpu_contract_warm,
        host_staging_bytes=0,
    )
    result = ThumbnailStatisticsResult(
        limits=limits,
        decision=decision,
        actual_backend=ThumbnailStatisticsBackend.GPU_CUPY,
        algorithm_id=(
            EXACT_FLOAT32_MINMAX_GPU_ALGORITHM_ID
            if request.contrast_mode == "Min-max"
            else EXACT_FLOAT32_PERCENTILE_GPU_ALGORITHM_ID
        ),
        elapsed_seconds=max(0.0, perf_counter() - started),
        runtime_id="cuda-cupy",
        device_id=str(device_id).strip(),
        requested_compute_mode=ComputeMode.PREFER_GPU,
        input_path="resident_borrow",
        logical_input_host_to_device_bytes=0,
        auxiliary_host_to_device_bytes=auxiliary_host_to_device_bytes,
        device_to_host_bytes=device_to_host_bytes,
        device_to_host_values=device_to_host_values,
    )
    return ResidentThumbnailStatisticsObservation(
        node_id=request.node_id,
        output_port=request.output_port,
        contrast_mode=request.contrast_mode,
        result=result,
    )


def _resident_device_shape(value: object) -> tuple[int, ...] | None:
    try:
        shape = tuple(int(size) for size in value.shape)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
    if any(size < 0 for size in shape):
        return None
    return shape


def _resident_image_state_is_presentable(state: _metadata.ImageState) -> bool:
    kind = str(state.kind).strip().casefold()
    return kind.endswith("image") and "label" not in kind


def _explicit_resident_channel_axis(
    state: _metadata.ImageState,
    ndim: int,
) -> tuple[int | None] | None:
    axes = tuple(state.axes)
    if len(axes) != ndim:
        return None
    candidates = tuple(
        index
        for index, axis in enumerate(axes)
        if axis.is_explicit
        and (
            str(axis.type).strip().casefold() == "channel"
            or str(axis.name).strip().casefold() in {"c", "channel", "rgb", "rgba"}
        )
    )
    if len(candidates) > 1:
        return None
    return (candidates[0] if candidates else None,)


def _nonnegative_resident_count(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a non-negative integer.")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return normalized


def _validate_resident_thumbnail_limit_channels(
    limits: object,
    *,
    channel_axis: int | None,
    shape: tuple[int, ...],
) -> None:
    """Require one exact limit pair per declared presentation channel."""

    values = np.asarray(limits, dtype=np.float64)
    expected_shape = (2,) if channel_axis is None else (int(shape[channel_axis]), 2)
    if values.shape != expected_shape:
        raise ValueError(
            "Resident thumbnail statistics returned a channel-limit shape "
            f"of {values.shape}; expected {expected_shape}."
        )


def _exact_float32_thumbnail_limits_from_device(
    runtime: object,
    device_value: object,
    *,
    device_id: str,
    channel_axis: int | None,
    contrast_mode: str,
    progress: ProgressContext,
):
    """Late import seam for the production CuPy resident-statistics adapter."""

    from napari_vipp.core.gpu.cupy_thumbnail_statistics import (
        exact_float32_thumbnail_limits_from_device,
    )

    return exact_float32_thumbnail_limits_from_device(
        runtime,
        device_value,
        device_id=device_id,
        channel_axis=channel_axis,
        contrast_mode=contrast_mode,
        progress=progress,
    )


def _default_compute_planner() -> ComputePlanner:
    from napari_vipp.core.compute_planning import plan_compute_decisions

    return plan_compute_decisions


def _historical_auto_compute_request(
    request: ComputeRequest,
    choice: PipelineTimingChoice | None,
    workloads: Sequence[WorkloadDescriptor],
    registry: ComputeRegistry,
) -> ComputeRequest:
    """Translate one compatible whole-pipeline choice into private exact pins."""

    if choice is None or request.mode is not ComputeMode.AUTO:
        return request
    workloads_by_node = {item.node_id: item for item in workloads}
    decisions_by_node = {item.node_id: item for item in choice.assignment.decisions}
    if not decisions_by_node or not set(decisions_by_node).issubset(workloads_by_node):
        return request
    specs_by_id = {
        item.implementation_id: item for item in registry.implementation_specs
    }
    preferences: dict[str, NodeComputePreference] = {}
    for node_id, decision in decisions_by_node.items():
        workload = workloads_by_node[node_id]
        if decision.operation_id != workload.operation_id:
            return request
        if decision.runtime_id == "cpu-numpy":
            if (
                decision.implementation_library_id != "cpu"
                or decision.implementation_id != f"cpu-{workload.operation_id}-v1"
                or decision.implementation_version != "1"
            ):
                return request
            preferences[node_id] = NodeComputePreference(NodePreferenceKind.CPU)
            continue
        implementation = specs_by_id.get(decision.implementation_id)
        if implementation is None:
            return request
        if (
            implementation.operation_id != workload.operation_id
            or implementation.runtime_id != decision.runtime_id
            or implementation.implementation_library_id
            != decision.implementation_library_id
            or implementation.implementation_version != decision.implementation_version
            or not implementation.eligible_for_auto(
                allow_experimental=request.allow_experimental
            )
        ):
            return request
        preferences[node_id] = NodeComputePreference(
            NodePreferenceKind.IMPLEMENTATION,
            decision.implementation_id,
        )
    return replace(
        request,
        mode=ComputeMode.CUSTOM,
        node_preferences=preferences,
    )


def _auto_cpu_exploration_compute_request(
    request: ComputeRequest,
    workloads: Sequence[WorkloadDescriptor],
) -> ComputeRequest:
    """Author one private all-CPU run to complete a GPU-only timing pair."""

    return replace(
        request,
        mode=ComputeMode.CUSTOM,
        node_preferences={
            item.node_id: NodeComputePreference(NodePreferenceKind.CPU)
            for item in workloads
        },
    )


def _estimate_auto_cpu_exploration_peak_bytes(
    workloads: Sequence[WorkloadDescriptor],
) -> int:
    """Conservatively estimate additional host peak for an optional CPU run.

    CPU filters frequently promote integer microscopy data to floating-point and
    allocate several same-sized work arrays. Keep-all execution also retains
    node outputs. This estimate intentionally favors skipping optional timing
    evidence over risking process/system commit exhaustion.
    """

    retained_output_bytes = 0
    largest_workspace_bytes = 0
    high_workspace_operations = {
        "richardson_lucy_deconvolution",
        "richardson_lucy_tv_deconvolution",
        "prepare_validate_psf",
    }
    for workload in workloads:
        largest_elements = 0
        largest_itemsize = 1
        for shape, dtype_name in zip(
            workload.input_shapes,
            workload.input_dtypes,
            strict=True,
        ):
            elements = int(math.prod(shape)) if shape else 1
            largest_elements = max(largest_elements, elements)
            try:
                largest_itemsize = max(
                    largest_itemsize,
                    int(np.dtype(dtype_name).itemsize),
                )
            except (TypeError, ValueError):
                largest_itemsize = max(largest_itemsize, 8)
        if largest_elements <= 0:
            continue
        promoted_output_bytes = largest_elements * max(largest_itemsize, 8)
        retained_output_bytes += promoted_output_bytes
        workspace_multiplier = (
            10 if workload.operation_id in high_workspace_operations else 4
        )
        largest_workspace_bytes = max(
            largest_workspace_bytes,
            promoted_output_bytes * workspace_multiplier,
        )
    return retained_output_bytes + largest_workspace_bytes


def _mark_historical_auto_planning(
    planning: object,
    *,
    original_request: ComputeRequest,
    choice: PipelineTimingChoice,
) -> object | None:
    """Restore public Auto intent and attach exact timing evidence provenance."""

    expected = {item.node_id: item for item in choice.assignment.decisions}
    decisions_by_node = {item.node_id: item for item in planning.decisions}
    if not set(expected).issubset(decisions_by_node):
        return None
    mismatches = [
        node_id
        for node_id, decision in decisions_by_node.items()
        if node_id in expected
        if decision.fallback_used
        or decision.runtime_id != expected[node_id].runtime_id
        or decision.implementation_library_id
        != expected[node_id].implementation_library_id
        or decision.implementation_id != expected[node_id].implementation_id
        or decision.implementation_version != expected[node_id].implementation_version
    ]
    if mismatches:
        return None
    decisions = []
    for decision in planning.decisions:
        if decision.node_id not in expected:
            decisions.append(decision)
            continue
        decisions.append(
            replace(
                decision,
                requested_preference=NodeComputePreference(NodePreferenceKind.AUTO),
                reason=DecisionReason.HISTORICAL_PERFORMANCE,
                reason_text=choice.reason,
                performance_evidence_kind="completed_pipeline_timing",
                performance_evidence_digest=choice.evidence_digest,
            )
        )
    return replace(
        planning,
        request=original_request,
        decisions=tuple(decisions),
    )


def _mark_auto_cpu_exploration_planning(
    planning: object,
    *,
    original_request: ComputeRequest,
):
    """Expose a private CPU comparison run as the user's public Auto intent."""

    decisions = []
    for decision in planning.decisions:
        if decision.runtime_id != "cpu-numpy" or decision.fallback_used:
            return replace(planning, request=original_request)
        decisions.append(
            replace(
                decision,
                requested_preference=NodeComputePreference(NodePreferenceKind.AUTO),
                reason=DecisionReason.PERFORMANCE_EXPLORATION,
                reason_text=(
                    "Auto already has a compatible accelerated timing but needs "
                    "one CPU comparison. This run measures CPU once; the next "
                    "exact matching run can use the faster assignment."
                ),
            )
        )
    return replace(
        planning,
        request=original_request,
        decisions=tuple(decisions),
    )


def _pipeline_timing_workload_fingerprint(
    pipeline: PrototypePipeline,
    runnable_node_ids: frozenset[str],
    *,
    retain_node_ids: frozenset[str],
    prune_unretained: bool,
    manual_node_ids: frozenset[str] | None,
    target_node_ids: frozenset[str] | None,
    compute_request: ComputeRequest,
    source_scientific_contexts: Mapping[str, str],
    cancel_callback: Callable[[], bool] | None,
) -> str:
    """Identify an exact scientific graph/source/scope without compute intent."""

    source_node_ids = {
        node_id
        for node_id, node in pipeline.nodes.items()
        if not pipeline.operation_spec(node.operation_id).has_input
    }
    if not source_node_ids.issubset(source_scientific_contexts):
        # Opaque metadata/source values must disable learning rather than permit
        # fuzzy evidence reuse.
        return ""
    try:
        runnable = set(runnable_node_ids)
        cached_boundaries = []
        for connection in pipeline.connections:
            if connection.target_id not in runnable or connection.source_id in runnable:
                continue
            source_node = pipeline.nodes[connection.source_id]
            if pipeline.operation_spec(source_node.operation_id).has_input:
                provenance = pipeline.node_compute_provenance.get(connection.source_id)
                if provenance is None:
                    return ""
                result_context = provenance.result_context_fingerprint
            else:
                result_context = source_scientific_contexts.get(
                    connection.source_id,
                    "",
                )
                if not result_context:
                    return ""
            cached_boundaries.append(
                {
                    "source_id": connection.source_id,
                    "source_port": connection.source_port,
                    "target_id": connection.target_id,
                    "target_port": connection.target_port,
                    "result_context": result_context,
                }
            )
        nodes = [
            _node_structural_identity(
                pipeline,
                node_id,
                cancel_callback=cancel_callback,
            )
            for node_id in pipeline.topological_order()
        ]
        tunnels = [
            {
                "name": item.name,
                "source_id": item.source_id,
                "source_port": item.source_port,
            }
            for item in pipeline.output_tunnel_list()
        ]
        return canonical_digest(
            {
                "schema_id": "vipp-completed-pipeline-workload-v1",
                "nodes": nodes,
                "tunnels": tunnels,
                "source_scientific_contexts": dict(
                    sorted(source_scientific_contexts.items())
                ),
                "runnable_node_ids": sorted(runnable_node_ids),
                "manual_node_ids": sorted(manual_node_ids or ()),
                "target_node_ids": sorted(target_node_ids or ()),
                "retain_node_ids": sorted(retain_node_ids),
                "prune_unretained": bool(prune_unretained),
                "cached_boundaries": cached_boundaries,
                "compute_contract": {
                    "precision_policy_id": compute_request.precision_policy_id,
                    "workload_policy_id": compute_request.workload_policy_id,
                    "runtime_id": compute_request.runtime_id,
                    "device_id": compute_request.device_id,
                    "accelerator_memory_cap_bytes": (
                        compute_request.accelerator_memory_cap_bytes
                    ),
                    "accelerator_safety_reserve_bytes": (
                        compute_request.accelerator_safety_reserve_bytes
                    ),
                },
            }
        )
    except (OverflowError, TypeError, ValueError, RecursionError):
        return ""


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
        cancel_callback=cancel_callback,
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
        cancel_callback=cancel_callback,
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
    cancel_callback: Callable[[], bool] | None = None,
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
        if operation.has_input:
            if not connections:
                # Disconnected processing nodes are valid graph-editing state,
                # but they have no scientific workload to preflight or plan.
                # PrototypePipeline.run() will leave them uncalculated.
                continue
            if pipeline._node_accepts_multiple_inputs(node):
                required = pipeline._required_inputs_for(node)
                connected_ports = {connection.target_port for connection in connections}
                if any(port not in connected_ports for port in range(required)):
                    continue
        input_shapes: list[tuple[int, ...]] = []
        input_dtypes: list[str] = []
        input_values: list[object] = []
        input_states: list[object] = []
        input_facts: list[ArrayFacts | None] = []
        inputs_resolved = True
        for connection in connections:
            port = OutputPortKey(connection.source_id, connection.source_port)
            inputs_resolved = inputs_resolved and port in values
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

        if (
            node.operation_id == "measure_objects_intensity"
            and any(facts is not None for facts in input_facts)
            and any(facts is None for facts in input_facts)
        ):
            # Multi-input support policy consumes an ordered facts tuple.  Do
            # not force an image-sized scan for an intrinsically finite integer
            # intensity input merely because the labels need a non-negativity
            # proof.  A metadata-only UNKNOWN record preserves port alignment;
            # float32 intensity is still explicitly requested and scanned by
            # _implementation_required_fact_indexes below.
            for index, facts in enumerate(input_facts):
                if facts is not None:
                    continue
                connection = connections[index]
                input_facts[index] = ArrayFacts(
                    shape=input_shapes[index],
                    dtype=input_dtypes[index],
                    element_count=int(math.prod(input_shapes[index])),
                    revision_fingerprint=(
                        "unscanned-input:"
                        f"{connection.source_id}:{connection.source_port}"
                    ),
                    completeness=FactCompleteness.UNKNOWN,
                )

        planning_call: PreparedNodeCall | None = None
        resolved_spatial_ndim: int | None = None
        if operation.has_input and input_values:
            try:
                planning_call = pipeline.prepare_node_call(
                    node_id,
                    tuple(input_values),
                    tuple(input_states),
                    cancel_callback=cancel_callback,
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
            inputs_resolved=inputs_resolved,
        )
        workloads.append(workload)
        if not inputs_resolved:
            # A descriptor projected from placeholders is not exact.  Keep the
            # entire unresolved branch absent from ``values`` so every
            # descendant remains unresolved and compute planning defers it to
            # CPU until authoritative upstream execution supplies real data.
            continue

        projected_outputs = _project_host_planning_outputs(
            pipeline,
            node.operation_id,
            planning_call,
            input_shapes,
            input_dtypes,
        )
        if projected_outputs is not None:
            for port_index, (projected_value, projected_state) in enumerate(
                projected_outputs
            ):
                port = OutputPortKey(node_id, port_index)
                values[port] = projected_value
                states[port] = projected_state
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
                        output_shape=projected_value.shape,
                        output_dtype=projected_value.dtype.name,
                    )
                    if propagated is not None:
                        facts_by_port[port] = propagated
        elif (
            planning_call is not None
            and (
                projection_spec := _shape_preserving_device_projection(
                    registry,
                    node.operation_id,
                    allow_experimental,
                )
            )
            is not None
        ):
            projected_descriptors = propagate_output_descriptors(
                projection_spec,
                tuple(
                    ValueDescriptor(shape, dtype)
                    for shape, dtype in zip(
                        input_shapes,
                        input_dtypes,
                        strict=True,
                    )
                ),
            )
            if projected_descriptors is None:
                continue
            predicted_states = pipeline.predict_shape_preserving_node_states(
                planning_call,
                output_dtype_policy_ids=tuple(
                    port.output_dtype_policy_id for port in projection_spec.output_ports
                ),
            )
            for port_index, (predicted_state, descriptor) in enumerate(
                zip(predicted_states, projected_descriptors, strict=True)
            ):
                port = OutputPortKey(node_id, port_index)
                states[port] = predicted_state
                values[port] = _ArrayDescription(
                    descriptor.shape,
                    np.dtype(descriptor.dtype),
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
                        output_shape=descriptor.shape,
                        output_dtype=descriptor.dtype,
                    )
                    if propagated is not None:
                        facts_by_port[port] = propagated
    return (
        tuple(workloads),
        MappingProxyType(facts_by_node),
        MappingProxyType(fact_lineage),
    )


def _project_host_planning_outputs(
    pipeline: PrototypePipeline,
    operation_id: str,
    planning_call: PreparedNodeCall | None,
    input_shapes: Sequence[tuple[int, ...]],
    input_dtypes: Sequence[str],
) -> tuple[tuple[_ArrayDescription, object | None], ...] | None:
    """Describe exact deterministic host outputs for downstream planning.

    Compute planning happens before any runnable CPU node executes.  A host
    transform that changes rank therefore needs an explicit shape/dtype
    projection when a later node may run on an accelerator.  Reuse the
    pipeline's constant-memory axis contracts, including dynamic multi-output
    transforms, before considering the deliberately narrow explicit contracts
    below.  Every published port must be exact and is independently validated
    by the authoritative CPU operation at execution time.
    """
    if planning_call is None:
        return None

    contract_results = None
    if operation_id in _EXACT_HOST_AXIS_CONTRACT_OPERATIONS:
        try:
            contract_results = pipeline._axis_contract_transform_results(planning_call)
        except (TypeError, ValueError):
            return None
        if contract_results is None:
            return None
        if len(contract_results) != planning_call.output_port_count:
            return None
        projected: list[tuple[_ArrayDescription, object | None]] = []
        for value, state in contract_results:
            raw_shape = getattr(value, "shape", None)
            raw_dtype = getattr(value, "dtype", None)
            if raw_shape is None or raw_dtype is None:
                return None
            try:
                shape = tuple(int(size) for size in raw_shape)
                dtype = np.dtype(raw_dtype)
            except (TypeError, ValueError):
                return None
            state_shape = getattr(state, "shape", None)
            if state_shape is not None:
                try:
                    exact_state_shape = tuple(int(size) for size in state_shape)
                except (TypeError, ValueError):
                    return None
                if exact_state_shape != shape:
                    return None
            projected.append((_ArrayDescription(shape, dtype), state))
        return tuple(projected)

    if (
        planning_call.multiple_inputs
        or planning_call.output_port_count != 1
        or len(input_shapes) != 1
        or len(input_dtypes) != 1
    ):
        return None

    input_shape = tuple(input_shapes[0])
    if not input_shape:
        return None
    if operation_id == "prepare_validate_psf":
        if bool(planning_call.kwargs.get("crop_empty_border", False)):
            # Cropping depends on exact pixel support and has no static shape
            # theorem. A cached concrete output can still be benchmarked.
            return None
        output_shape = tuple(
            size + 1
            if bool(planning_call.kwargs.get("force_odd_shape", True)) and size % 2 == 0
            else size
            for size in input_shape
        )
        description = _ArrayDescription(output_shape, np.dtype(np.float32))
        projected_state = None
        if planning_call.input_states and planning_call.input_states[0] is not None:
            (base_state,) = pipeline.predict_shape_preserving_node_states(
                planning_call,
                output_dtype_policy_ids=("fixed:float32",),
            )
            if base_state is not None:
                projected_state = replace(
                    base_state,
                    shape=output_shape,
                    memory=_metadata._memory_label(
                        int(np.prod(output_shape, dtype=np.int64))
                        * np.dtype(np.float32).itemsize
                    ),
                )
        return ((description, projected_state),)
    if operation_id in {"canny_edges", "otsu_threshold"}:
        projection = _scalar_plane_luma_output_shape(
            input_shape,
            planning_call.kwargs.get("channel_axis"),
        )
        if projection is None:
            return None
        output_shape, channel_axis = projection

        description = _ArrayDescription(output_shape, np.dtype(bool))
        projected_state = None
        if planning_call.input_states and planning_call.input_states[0] is not None:
            (base_state,) = pipeline.predict_shape_preserving_node_states(
                planning_call,
                output_dtype_policy_ids=("fixed:bool",),
            )
            if base_state is not None:
                axes = tuple(getattr(base_state, "axes", ()))
                if channel_axis is not None and len(axes) == len(input_shape):
                    axes = axes[:channel_axis] + axes[channel_axis + 1 :]
                projected_state = replace(
                    base_state,
                    shape=output_shape,
                    axes=axes,
                    channels=(() if channel_axis is not None else base_state.channels),
                    memory=_metadata._memory_label(
                        int(np.prod(output_shape, dtype=np.int64))
                        * np.dtype(bool).itemsize
                    ),
                    value_pattern="",
                )
        return ((description, projected_state),)
    if operation_id != "extract_channel":
        return None
    projection = _extract_channel_output_projection(
        input_shape,
        axis_types=tuple(planning_call.kwargs.get("axis_types", ())),
        axis_names=tuple(planning_call.kwargs.get("axis_names", ())),
        raw_channel=planning_call.kwargs.get("channel", 0),
    )
    if projection is None:
        return None
    output_shape, channel_axis = projection
    output_dtype = np.dtype(input_dtypes[0])
    description = _ArrayDescription(output_shape, output_dtype)
    projected_state = None
    if planning_call.input_states and planning_call.input_states[0] is not None:
        (base_state,) = pipeline.predict_shape_preserving_node_states(
            planning_call,
            output_dtype_policy_ids=("dtype-same-v1",),
        )
        if base_state is not None:
            axes = tuple(getattr(base_state, "axes", ()))
            if len(axes) != len(input_shape):
                return None
            output_axes = axes[:channel_axis] + axes[channel_axis + 1 :]
            projected_state = replace(
                base_state,
                shape=output_shape,
                axes=output_axes,
                kind=_metadata._lazy_kind_label(
                    output_dtype,
                    output_shape,
                    output_axes,
                ),
                memory=_metadata._memory_label(
                    int(np.prod(output_shape, dtype=np.int64)) * output_dtype.itemsize
                ),
                value_pattern="",
            )
    return ((description, projected_state),)


def _extract_channel_output_projection(
    input_shape: tuple[int, ...],
    *,
    axis_types: Sequence[object],
    axis_names: Sequence[object],
    raw_channel: object,
) -> tuple[tuple[int, ...], int] | None:
    """Return the exact rank-reducing semantic channel selection."""

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
                if str(axis_name).strip().casefold() in {"c", "channel", "rgb", "rgba"}
            ),
            None,
        )
    if channel_axis is None:
        return None

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
    return (
        input_shape[:channel_axis] + input_shape[channel_axis + 1 :],
        channel_axis,
    )


def _scalar_plane_luma_output_shape(
    input_shape: tuple[int, ...],
    raw_channel_axis: object,
) -> tuple[tuple[int, ...], int | None] | None:
    """Return the exact mask shape and removed encoded-colour axis."""

    if raw_channel_axis is None:
        return input_shape, None
    if (
        isinstance(raw_channel_axis, (bool, np.bool_))
        or not isinstance(raw_channel_axis, Integral)
        or len(input_shape) < 3
        or int(raw_channel_axis) < -len(input_shape)
        or int(raw_channel_axis) >= len(input_shape)
    ):
        return None
    channel_axis = int(raw_channel_axis) % len(input_shape)
    if input_shape[channel_axis] not in {3, 4}:
        return None
    return (
        input_shape[:channel_axis] + input_shape[channel_axis + 1 :],
        channel_axis,
    )


def _predict_device_node_states(
    pipeline: PrototypePipeline,
    call: PreparedNodeCall,
    implementation: OperationComputeSpec,
    outputs: tuple[object, ...],
) -> tuple[object | None, ...]:
    """Project resident output metadata without materializing device arrays."""

    if implementation.shape_policy_id == "extract-channel-semantic-axis-v1":
        if len(outputs) != 1 or len(call.inputs) != 1:
            raise RuntimeError(
                "Resident Extract Channel metadata requires one input and one output."
            )
        input_shape = tuple(int(size) for size in call.inputs[0].shape)
        input_dtype = np.dtype(call.inputs[0].dtype)
        projected = _project_host_planning_outputs(
            pipeline,
            "extract_channel",
            call,
            (input_shape,),
            (input_dtype.name,),
        )
        if projected is None:
            raise RuntimeError(
                "The admitted Extract Channel semantic-axis projection is invalid."
            )
        ((description, projected_state),) = projected
        output = outputs[0]
        actual_shape = tuple(int(size) for size in output.shape)
        actual_dtype = np.dtype(output.dtype)
        if actual_shape != description.shape or actual_dtype != input_dtype:
            raise RuntimeError(
                "The resident Extract Channel output violated its declared "
                "semantic-axis shape or dtype contract."
            )
        return (projected_state,)

    output_dtype_policy_ids = tuple(
        port.output_dtype_policy_id for port in implementation.output_ports
    )
    states = pipeline.predict_shape_preserving_node_states(
        call,
        output_dtype_policy_ids=output_dtype_policy_ids,
    )
    if implementation.shape_policy_id == "shape-preserving-v1":
        if output_dtype_policy_ids == ("fixed:bool",):
            if len(outputs) != 1 or len(call.inputs) != 1:
                raise RuntimeError(
                    "Resident boolean-mask metadata requires one input and one output."
                )
            expected_shape = tuple(int(size) for size in call.inputs[0].shape)
            actual_shape = tuple(int(size) for size in outputs[0].shape)
            if actual_shape != expected_shape or np.dtype(outputs[0].dtype) != np.dtype(
                bool
            ):
                raise RuntimeError(
                    f"The resident {implementation.operation_id!r} output violated "
                    "its declared shape-preserving bool-mask contract."
                )
        return states
    if implementation.shape_policy_id != "scalar-plane-luma-mask-v1":
        raise RuntimeError(
            "Resident metadata projection is unavailable for shape policy "
            f"{implementation.shape_policy_id!r}."
        )
    if len(outputs) != 1 or len(states) != 1 or len(call.inputs) != 1:
        raise RuntimeError(
            "Scalar-plane mask metadata requires one input and one output."
        )
    input_shape = tuple(int(size) for size in call.inputs[0].shape)
    projection = _scalar_plane_luma_output_shape(
        input_shape,
        call.kwargs.get("channel_axis"),
    )
    if projection is None:
        raise RuntimeError("The admitted scalar-plane luma shape is invalid.")
    expected_shape, channel_axis = projection
    output = outputs[0]
    actual_shape = tuple(int(size) for size in output.shape)
    actual_dtype = np.dtype(output.dtype)
    if actual_shape != expected_shape or actual_dtype != np.dtype(bool):
        raise RuntimeError(
            "The resident segmentation output violated its declared bool-mask "
            "shape contract."
        )
    state = states[0]
    if state is None:
        return (None,)
    axes = tuple(getattr(state, "axes", ()))
    if channel_axis is not None and len(axes) == len(input_shape):
        axes = axes[:channel_axis] + axes[channel_axis + 1 :]
    return (
        replace(
            state,
            shape=actual_shape,
            axes=axes,
            channels=(() if channel_axis is not None else state.channels),
            memory=_metadata._memory_label(
                int(np.prod(actual_shape, dtype=np.int64)) * actual_dtype.itemsize
            ),
            value_pattern="",
        ),
    )


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

    from napari_vipp.core.compute_repairs import potential_compute_repair_specs

    selected: list[OperationComputeSpec] = []
    identities: set[tuple[str, str, str]] = set()
    for workload in workloads:
        repair_identities = {
            (
                implementation.runtime_id,
                implementation.implementation_id,
                implementation.implementation_version,
            )
            for implementation in potential_compute_repair_specs(
                request,
                workload,
                registry,
            )
        }
        for implementation in _candidate_specs_for_workload(
            registry,
            request,
            workload,
        ):
            identity = (
                implementation.runtime_id,
                implementation.implementation_id,
                implementation.implementation_version,
            )
            static_match = _candidate_statically_matches(implementation, workload)
            if not static_match and identity not in repair_identities:
                continue
            if static_match:
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
    preference = (
        request.preference_for(workload.node_id)
        if request.mode is ComputeMode.CUSTOM
        else NodeComputePreference(NodePreferenceKind.AUTO)
    )
    forced = request.mode is ComputeMode.CUSTOM and preference.kind in {
        NodePreferenceKind.BEST_GPU,
        NodePreferenceKind.LIBRARY,
        NodePreferenceKind.IMPLEMENTATION,
    }
    if forced or request.mode is ComputeMode.PREFER_GPU:
        return True
    evidence = performance_evidence.get(
        (workload.node_id, implementation.implementation_id)
    )
    return evidence is None or evaluate_auto_performance(evidence).select_candidate


def _candidate_specs_for_workload(
    registry: ComputeRegistry,
    request: ComputeRequest,
    workload: WorkloadDescriptor,
) -> tuple[OperationComputeSpec, ...]:
    if request.mode is ComputeMode.CPU or not workload.inputs_resolved:
        return ()
    preference = (
        request.preference_for(workload.node_id)
        if request.mode is ComputeMode.CUSTOM
        else NodeComputePreference(NodePreferenceKind.AUTO)
    )
    if request.mode is ComputeMode.CUSTOM and preference.kind is NodePreferenceKind.CPU:
        return ()
    implementations = registry.implementations_for_operation(
        workload.operation_id,
        allow_experimental=request.allow_experimental,
    )
    automatic_intent = request.mode is ComputeMode.AUTO or (
        request.mode is ComputeMode.CUSTOM
        and preference.kind is NodePreferenceKind.AUTO
    )
    if automatic_intent:
        implementations = tuple(
            item
            for item in implementations
            if item.eligible_for_auto(
                allow_experimental=request.allow_experimental,
            )
        )
    if request.runtime_id:
        implementations = tuple(
            item for item in implementations if item.runtime_id == request.runtime_id
        )
    if (
        request.mode is ComputeMode.CUSTOM
        and preference.kind is NodePreferenceKind.LIBRARY
    ):
        implementations = tuple(
            item
            for item in implementations
            if item.implementation_library_id == preference.value
        )
    elif (
        request.mode is ComputeMode.CUSTOM
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
        and index < len(workload.input_dtypes)
        and not _dtype_intrinsically_finite(workload.input_dtypes[index])
    }
    if getattr(implementation, "parameter_policy_id", "") == (
        "basic-measurements-parameters-v1"
    ):
        # Labels always need a complete non-negativity proof unless an upstream
        # label operation supplied the theorem. Integer intensity dtypes are
        # intrinsically finite; float32 requires its own complete scan.
        required.add(0)
        if (
            workload.operation_id == "measure_objects_intensity"
            and len(workload.input_dtypes) > 1
            and np.dtype(workload.input_dtypes[1]) == np.dtype(np.float32)
        ):
            required.add(1)
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
        if limitation.startswith("integer-span-") and (
            not workload.input_dtypes
            or not np.issubdtype(np.dtype(workload.input_dtypes[0]), np.integer)
            or np.dtype(workload.input_dtypes[0]) == np.dtype(bool)
        ):
            continue
        if (
            limitation.startswith("integer-span-")
            and workload.operation_id == "otsu_threshold"
            and workload.input_dtypes
        ):
            dtype = np.dtype(workload.input_dtypes[0])
            dtype_limits = np.iinfo(dtype)
            type_span = int(dtype_limits.max) - int(dtype_limits.min) + 1
            if type_span <= OTSU_MAXIMUM_NATIVE_INTEGER_LEVELS:
                # Admission is provable from the <=16-bit dtype itself; a
                # complete image extrema scan would not refine the decision.
                continue
        if limitation.startswith("float32-") and (
            not workload.input_dtypes
            or np.dtype(workload.input_dtypes[0]) != np.dtype(np.float32)
        ):
            continue
        required.update(range(len(workload.input_dtypes)))
    return frozenset(required)


def _dtype_intrinsically_finite(dtype: object) -> bool:
    """Return whether the public dtype cannot represent NaN or infinity."""

    resolved = np.dtype(dtype)
    return resolved == np.dtype(bool) or np.issubdtype(resolved, np.integer)


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
    # Prefer the concrete value so byte order cannot be erased by ImageState's
    # user-facing dtype name. Native arrays retain the historical short name;
    # a non-native array keeps its NumPy descriptor and therefore fails closed
    # in operation-specific accelerator policy.
    raw_dtype = getattr(value, "dtype", None)
    if raw_dtype is None:
        raw_dtype = getattr(state, "dtype", "object")
    try:
        resolved_dtype = np.dtype(raw_dtype)
        dtype = resolved_dtype.name if resolved_dtype.isnative else resolved_dtype.str
    except (TypeError, ValueError):
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
    output_shape: tuple[int, ...] | None = None,
    output_dtype: str | None = None,
) -> ArrayFacts | None:
    if operation_id not in _PHASE_ONE_FACT_OPERATIONS:
        return None

    guarantees = set(facts.guarantees)
    finite_count = facts.finite_count
    completeness = facts.completeness
    minimum = None
    maximum = None
    resolved_shape = facts.shape if output_shape is None else tuple(output_shape)
    resolved_dtype_name = facts.dtype if output_dtype is None else str(output_dtype)
    output_elements = int(math.prod(resolved_shape))
    try:
        resolved_dtype = np.dtype(resolved_dtype_name)
    except (TypeError, ValueError):
        resolved_dtype = None
    dtype_proves_finite = resolved_dtype is not None and (
        resolved_dtype == np.dtype(bool) or np.issubdtype(resolved_dtype, np.integer)
    )
    if operation_id in {"rolling_ball_background", "subtract_background"}:
        float_output_proven_finite = (
            resolved_dtype is not None
            and np.issubdtype(resolved_dtype, np.floating)
            and _background_float_output_proven_finite(
                operation_id,
                facts,
                parameters,
                resolved_dtype,
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
    elif operation_id == "convert_dtype":
        source_dtype = np.dtype(facts.dtype)
        output_dtype_parameter = (
            str(parameters.get("output_dtype", "uint8")).strip().casefold()
        )
        scaling = str(parameters.get("scaling", "rescale")).strip().casefold()
        if (
            source_dtype not in {np.dtype(np.uint8), np.dtype(np.uint16)}
            or resolved_dtype != np.dtype(np.float32)
            or output_dtype_parameter != "float32"
            or scaling != "preserve"
        ):
            return None
        # Every uint8/uint16 value is represented exactly in float32. This is a
        # conversion theorem, not a sample-based assumption, so downstream
        # finite-only providers can consume the projected resident result.
        finite_count = output_elements
        completeness = FactCompleteness.COMPLETE
        minimum = facts.minimum
        maximum = facts.maximum
        guarantees.update(("nonnegative", "no-negative-zero"))
    elif operation_id == "extract_channel":
        # Selecting one semantic channel is an exact subset view. Whole-array
        # finiteness and sign theorems therefore remain valid for the selected
        # channel, but source extrema need not occur in that channel.
        guarantees.discard("extrema-conservative-enclosure")
        if dtype_proves_finite or facts.all_finite is True:
            finite_count = output_elements
            completeness = FactCompleteness.COMPLETE
        else:
            finite_count = None
            completeness = FactCompleteness.UNKNOWN
    elif operation_id == "sigma_filter":
        # A successful Sigma Filter result is a bounded mean of finite source
        # samples restored to the authored dtype. The operation rejects the
        # float32 square-overflow region before calculation.
        finite_count = output_elements
        completeness = FactCompleteness.COMPLETE
        if (
            facts.completeness is FactCompleteness.COMPLETE
            and facts.all_finite is True
            and facts.minimum is not None
            and facts.maximum is not None
        ):
            # Every result branch is a mean of source samples.  The inherited
            # extrema are therefore a conservative enclosure (not a claim that
            # the output still attains both values), sufficient for downstream
            # magnitude admission without a device-to-host scan.
            minimum = facts.minimum
            maximum = facts.maximum
            guarantees.add("extrema-conservative-enclosure")
        if "nonnegative" in guarantees:
            guarantees.add("no-negative-zero")
        else:
            guarantees.discard("no-negative-zero")
    elif operation_id == "prepare_validate_psf":
        normalize = bool(parameters.get("normalize_sum", True))
        clip = bool(parameters.get("clip_negatives", True))
        if normalize and not clip:
            # Positive/negative cancellation can make a finite float32 sum
            # arbitrarily small and overflow normalization. Fail closed for
            # downstream finite-only candidates in that advanced region.
            return None
        finite_count = output_elements
        completeness = FactCompleteness.COMPLETE
        if clip:
            guarantees.update(("nonnegative", "no-negative-zero"))
        else:
            guarantees.discard("nonnegative")
            guarantees.discard("no-negative-zero")
    elif operation_id in {
        "richardson_lucy_deconvolution",
        "richardson_lucy_tv_deconvolution",
    }:
        # The reviewed RL/RL-TV contracts sanitize every iteration and their public
        # output. This is an output theorem, not an assumption inherited from
        # either input, and lets downstream finite-only GPU nodes be planned
        # without rescanning a resident result.
        finite_count = output_elements
        completeness = FactCompleteness.COMPLETE
        if bool(parameters.get("clip_output_negative", True)):
            guarantees.update(("nonnegative", "no-negative-zero"))
        else:
            guarantees.discard("nonnegative")
            guarantees.discard("no-negative-zero")
    elif operation_id in {
        "binary_threshold",
        "canny_edges",
        "otsu_threshold",
        "fill_holes",
    }:
        # These reviewed segmentation providers return an exact boolean mask.
        # Boolean output is finite by construction and cannot contain negative
        # values or a signed zero, independent of the source's numeric range.
        finite_count = output_elements
        completeness = FactCompleteness.COMPLETE
        guarantees.update(("nonnegative", "no-negative-zero"))
    elif operation_id == "remove_small_objects":
        # The operation preserves a bool mask or a non-negative integer label
        # dtype. The promoted GPU region is bool-only, while authoritative CPU
        # label cleanup still benefits from the exact kind/finiteness theorem.
        if resolved_dtype is None or not (
            resolved_dtype == np.dtype(bool)
            or np.issubdtype(resolved_dtype, np.integer)
        ):
            return None
        finite_count = output_elements
        completeness = FactCompleteness.COMPLETE
        guarantees.update(("nonnegative", "no-negative-zero"))
        if resolved_dtype != np.dtype(bool):
            guarantees.add("integer-labels")
    elif operation_id == "label_connected_components":
        # Successful CPU and GPU paths return an exact int32 label image.
        # Label extrema and counts remain data-dependent, but finiteness,
        # integrality, and sign are output theorems that require no host scan.
        finite_count = output_elements
        completeness = FactCompleteness.COMPLETE
        guarantees.update(("integer-labels", "nonnegative", "no-negative-zero"))

    # Exact extrema are generally not propagated because each operation can
    # change them. Sigma Filter is the narrow exception above: its branch
    # formulas prove the inherited range is a safe conservative enclosure.
    return ArrayFacts(
        shape=resolved_shape,
        dtype=resolved_dtype_name,
        element_count=output_elements,
        revision_fingerprint=(
            f"{facts.revision_fingerprint}>{operation_id}:{output_port.port_index}"
        ),
        completeness=completeness,
        finite_count=finite_count,
        minimum=minimum,
        maximum=maximum,
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
        if name in {"progress", "image_state"} or name.startswith("_vipp_"):
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


def _shape_preserving_device_projection(
    registry: ComputeRegistry,
    operation_id: str,
    allow_experimental: bool,
) -> OperationComputeSpec | None:
    implementations = registry.implementations_for_operation(
        operation_id,
        allow_experimental=allow_experimental,
    )
    compatible = tuple(
        implementation.supports_device_residency
        and not implementation.host_boundary
        and implementation.shape_policy_id == "shape-preserving-v1"
        for implementation in implementations
    )
    candidates = tuple(
        implementation
        for implementation, supported in zip(
            implementations,
            compatible,
            strict=True,
        )
        if supported
    )
    if not candidates:
        return None
    output_policies = {
        tuple(port.output_dtype_policy_id for port in implementation.output_ports)
        for implementation in candidates
    }
    if len(output_policies) != 1:
        # Planning must not guess between incompatible typed projections.
        return None
    return candidates[0]


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
        tuple(getattr(planning, "repair_suggestions", ())),
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


def _capture_source_scientific_contexts(
    pipeline: PrototypePipeline,
    request: PipelineRunRequest,
    *,
    cancel_callback: Callable[[], bool] | None,
) -> dict[str, str]:
    """Fingerprint the exact source snapshots used by this request."""

    contexts: dict[str, str] = {}
    for node_id in pipeline.topological_order():
        node = pipeline.nodes[node_id]
        if pipeline.operation_spec(node.operation_id).has_input:
            continue
        _check_scientific_context_cancelled(cancel_callback)
        results = pipeline.source_node_results(
            node_id,
            request.input_data,
            request.input_metadata,
            request.input_name,
            request.source_payloads,
        )
        payload = request.source_payloads.get(node_id)
        if payload is None:
            payload = SourcePayload(
                request.input_data,
                request.input_metadata,
                request.input_name,
            )
        try:
            contexts[node_id] = _source_scientific_context_fingerprint(
                pipeline,
                node_id,
                payload,
                results,
                cancel_callback=cancel_callback,
            )
        except (OverflowError, TypeError, ValueError):
            # Unsupported or opaque source metadata must disable reuse without
            # preventing the authoritative operation path from running.
            continue
    return contexts


def _source_scientific_context_fingerprint(
    pipeline: PrototypePipeline,
    node_id: str,
    payload: SourcePayload,
    results: Sequence[tuple[object, object]],
    *,
    cancel_callback: Callable[[], bool] | None,
) -> str:
    normalized_results = []
    for value, state in results:
        normalized_results.append(
            {
                "array": _scientific_array_identity(
                    value,
                    cancel_callback=cancel_callback,
                ),
                "state": _scientific_identity_value(
                    state,
                    cancel_callback=cancel_callback,
                ),
            }
        )
    document = {
        "schema_id": "vipp-source-scientific-context-v1",
        "node": _node_structural_identity(
            pipeline,
            node_id,
            cancel_callback=cancel_callback,
        ),
        "outputs": normalized_results,
        "payload_metadata": _scientific_identity_value(
            payload.metadata,
            cancel_callback=cancel_callback,
        ),
        "payload_name": str(payload.name),
        "payload_image_state": _scientific_identity_value(
            payload.image_state,
            cancel_callback=cancel_callback,
        ),
        "revision_token": _scientific_identity_value(
            payload.revision_token,
            cancel_callback=cancel_callback,
        ),
    }
    _check_scientific_context_cancelled(cancel_callback)
    return canonical_digest(document)


def _processing_scientific_context_fingerprint(
    pipeline: PrototypePipeline,
    node_id: str,
    upstream_provenance: Mapping[str, CachedNodeComputeProvenance],
    *,
    cancel_callback: Callable[[], bool] | None = None,
) -> str:
    upstream_contexts = []
    for connection in pipeline._input_connections(node_id):
        provenance = upstream_provenance[connection.source_id]
        upstream_contexts.append(
            {
                "source_id": connection.source_id,
                "source_port": connection.source_port,
                "target_port": connection.target_port,
                "result_context": provenance.result_context_fingerprint,
            }
        )
    if not upstream_contexts:
        raise ValueError("Processing-node context requires an upstream result.")
    return canonical_digest(
        {
            "schema_id": "vipp-processing-scientific-context-v1",
            "node": _node_structural_identity(
                pipeline,
                node_id,
                cancel_callback=cancel_callback,
            ),
            "upstream": upstream_contexts,
        }
    )


def _node_structural_identity(
    pipeline: PrototypePipeline,
    node_id: str,
    *,
    cancel_callback: Callable[[], bool] | None = None,
) -> object:
    node = pipeline.nodes[node_id]
    connections = pipeline._input_connections(node_id)
    return {
        "node_id": node_id,
        "operation_id": node.operation_id,
        "input_type": node.input_type,
        "output_type": node.output_type,
        "max_inputs": node.max_inputs,
        "public_params": _scientific_identity_value(
            pipeline._public_params(node.params),
            cancel_callback=cancel_callback,
        ),
        "input_ports": _scientific_identity_value(
            pipeline.input_ports(node_id),
            cancel_callback=cancel_callback,
        ),
        "output_ports": _scientific_identity_value(
            pipeline.output_ports(node_id),
            cancel_callback=cancel_callback,
        ),
        "incoming_topology": [
            {
                "source_id": connection.source_id,
                "source_port": connection.source_port,
                "target_port": connection.target_port,
                "tunnel_name": connection.tunnel_name,
            }
            for connection in connections
        ],
    }


def _scientific_array_identity(
    value: object,
    *,
    cancel_callback: Callable[[], bool] | None,
) -> object:
    if getattr(type(value), "__cuda_array_interface__", None) is not None:
        raise TypeError("Scientific source contexts accept host arrays only.")
    try:
        shape = tuple(int(size) for size in value.shape)
        dtype = np.dtype(value.dtype)
    except (AttributeError, TypeError, ValueError):
        array = np.asarray(value)
        shape = tuple(int(size) for size in array.shape)
        dtype = array.dtype
        value = array
    if any(size < 0 for size in shape):
        raise ValueError("Scientific source array shapes must be non-negative.")
    if dtype.hasobject:
        raise TypeError("Scientific source contexts reject object arrays.")
    digest = sha256()
    values_per_chunk = max(
        1,
        _SCIENTIFIC_CONTEXT_CHUNK_BYTES // max(1, int(dtype.itemsize)),
    )
    for raw_chunk in _iter_exact_array_chunks(
        value,
        shape=shape,
        values_per_chunk=values_per_chunk,
    ):
        _check_scientific_context_cancelled(cancel_callback)
        chunk = np.ascontiguousarray(np.asarray(raw_chunk)).view(np.uint8)
        digest.update(memoryview(chunk.reshape(-1)))
    _check_scientific_context_cancelled(cancel_callback)
    return {
        "schema_id": "vipp-exact-host-array-v1",
        "shape": list(shape),
        "dtype": dtype.str,
        "dtype_descriptor": str(dtype.descr),
        "bytes_sha256": digest.hexdigest(),
    }


def _iter_exact_array_chunks(
    value: object,
    *,
    shape: tuple[int, ...],
    values_per_chunk: int,
):
    """Yield bounded chunks in canonical C order without eager lazy loading."""

    if isinstance(value, np.ndarray):
        yield from np.nditer(
            value,
            flags=["buffered", "external_loop", "zerosize_ok"],
            op_flags=(("readonly",),),
            order="C",
            buffersize=values_per_chunk,
        )
        return
    total_values = math.prod(shape)
    if total_values == 0:
        return
    if not shape:
        try:
            yield value[()]
        except (IndexError, TypeError):
            yield np.asarray(value)
        return
    reshape = getattr(value, "reshape", None)
    if callable(reshape):
        try:
            flattened = reshape((-1,))
        except (TypeError, ValueError):
            flattened = None
        if flattened is not None:
            for start in range(0, total_values, values_per_chunk):
                yield flattened[start : min(total_values, start + values_per_chunk)]
            return

    trailing_values = 1
    split_axis = len(shape) - 1
    for axis in range(len(shape) - 1, -1, -1):
        candidate = trailing_values * shape[axis]
        if candidate > values_per_chunk:
            split_axis = axis
            break
        trailing_values = candidate
        split_axis = axis
    axis_chunk = max(1, values_per_chunk // max(1, trailing_values))
    prefix_shape = shape[:split_axis]
    prefixes = np.ndindex(prefix_shape) if prefix_shape else ((),)
    for prefix in prefixes:
        for start in range(0, shape[split_axis], axis_chunk):
            selector = (
                *prefix,
                slice(start, min(shape[split_axis], start + axis_chunk)),
                *(slice(None),) * (len(shape) - split_axis - 1),
            )
            yield value[selector]


def _scientific_identity_value(
    value: object,
    *,
    cancel_callback: Callable[[], bool] | None,
    active: set[int] | None = None,
) -> object:
    """Return an exact JSON-safe identity or fail closed for opaque values."""

    _check_scientific_context_cancelled(cancel_callback)
    if value is None:
        return {"type": "none"}
    if isinstance(value, Enum):
        return {
            "type": f"enum:{type(value).__module__}.{type(value).__qualname__}",
            "value": _scientific_identity_value(
                value.value,
                cancel_callback=cancel_callback,
                active=active,
            ),
        }
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": str(value)}
    if isinstance(value, float):
        return {"type": "float", "value": value.hex()}
    if isinstance(value, complex):
        return {
            "type": "complex",
            "real": value.real.hex(),
            "imag": value.imag.hex(),
        }
    if isinstance(value, str):
        return {"type": "str", "value": value}
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        return {
            "type": type(value).__name__,
            "length": len(raw),
            "sha256": sha256(raw).hexdigest(),
        }
    if isinstance(value, np.ndarray):
        return _scientific_array_identity(
            value,
            cancel_callback=cancel_callback,
        )
    if isinstance(value, np.generic):
        return {
            "type": "numpy-scalar",
            "array": _scientific_array_identity(
                np.asarray(value),
                cancel_callback=cancel_callback,
            ),
        }
    if isinstance(value, np.dtype):
        return {
            "type": "numpy-dtype",
            "dtype": value.str,
            "descriptor": str(value.descr),
        }
    if isinstance(value, PathLike):
        return {
            "type": f"path:{type(value).__module__}.{type(value).__qualname__}",
            "value": str(fspath(value)),
        }
    if isinstance(value, slice):
        return {
            "type": "slice",
            "start": _scientific_identity_value(
                value.start,
                cancel_callback=cancel_callback,
                active=active,
            ),
            "stop": _scientific_identity_value(
                value.stop,
                cancel_callback=cancel_callback,
                active=active,
            ),
            "step": _scientific_identity_value(
                value.step,
                cancel_callback=cancel_callback,
                active=active,
            ),
        }

    if active is None:
        active = set()
    identity = id(value)
    if identity in active:
        raise ValueError("Scientific context values must not contain cycles.")
    active.add(identity)
    try:
        if is_dataclass(value) and not isinstance(value, type):
            return {
                "type": (
                    f"dataclass:{type(value).__module__}.{type(value).__qualname__}"
                ),
                "fields": [
                    {
                        "name": item.name,
                        "value": _scientific_identity_value(
                            getattr(value, item.name),
                            cancel_callback=cancel_callback,
                            active=active,
                        ),
                    }
                    for item in fields(value)
                ],
            }
        if isinstance(value, Mapping):
            entries = [
                (
                    _scientific_identity_value(
                        key,
                        cancel_callback=cancel_callback,
                        active=active,
                    ),
                    _scientific_identity_value(
                        item,
                        cancel_callback=cancel_callback,
                        active=active,
                    ),
                )
                for key, item in value.items()
            ]
            entries.sort(key=lambda pair: canonical_digest(pair[0]))
            return {
                "type": f"mapping:{type(value).__module__}.{type(value).__qualname__}",
                "entries": [{"key": key, "value": item} for key, item in entries],
            }
        if isinstance(value, (tuple, list)):
            return {
                "type": type(value).__name__,
                "items": [
                    _scientific_identity_value(
                        item,
                        cancel_callback=cancel_callback,
                        active=active,
                    )
                    for item in value
                ],
            }
        if isinstance(value, (set, frozenset)):
            items = [
                _scientific_identity_value(
                    item,
                    cancel_callback=cancel_callback,
                    active=active,
                )
                for item in value
            ]
            items.sort(key=canonical_digest)
            return {"type": type(value).__name__, "items": items}
    finally:
        active.remove(identity)
    raise TypeError(
        "Scientific context contains an opaque value of type "
        f"{type(value).__module__}.{type(value).__qualname__}."
    )


def _check_scientific_context_cancelled(
    cancel_callback: Callable[[], bool] | None,
) -> None:
    if cancel_callback is not None and cancel_callback():
        raise OperationCancelled(
            "Operation cancelled while fingerprinting scientific cache context."
        )


def _hydrate_cached_pipeline_outputs(
    pipeline: PrototypePipeline,
    request: PipelineRunRequest,
    *,
    implementation_specs: Sequence[OperationComputeSpec] = (),
    source_scientific_contexts: Mapping[str, str],
    cancel_callback: Callable[[], bool] | None,
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
    requested_completed = set(request.completed_node_ids) & set(pipeline.nodes)
    accepted_completed: set[str] = set()
    accepted_provenance: dict[str, CachedNodeComputeProvenance] = {}
    cached_provenance = request.cached_compute_provenance
    for node_id in pipeline.topological_order():
        if node_id not in requested_completed or not pipeline._has_cached_output(
            node_id
        ):
            continue
        node = pipeline.nodes[node_id]
        operation = pipeline.operation_spec(node.operation_id)
        if not operation.has_input:
            source_context = source_scientific_contexts.get(node_id)
            provenance = cached_provenance.get(node_id)
            if source_context is None or not cached_source_provenance_matches(
                provenance,
                node_id=node_id,
                operation_id=node.operation_id,
                scientific_context_fingerprint=source_context,
            ):
                continue
            accepted_completed.add(node_id)
            accepted_provenance[node_id] = provenance
            continue
        if any(
            connection.source_id not in accepted_completed
            for connection in pipeline._input_connections(node_id)
        ):
            continue
        try:
            scientific_context = _processing_scientific_context_fingerprint(
                pipeline,
                node_id,
                accepted_provenance,
                cancel_callback=cancel_callback,
            )
        except (KeyError, TypeError, ValueError):
            continue
        provenance = cached_provenance.get(node_id)
        if provenance is None or not cached_node_provenance_matches(
            provenance,
            request=request.compute_request,
            node_id=node_id,
            operation_id=node.operation_id,
            scientific_context_fingerprint=scientific_context,
            implementation_specs=implementation_specs,
        ):
            continue
        accepted_completed.add(node_id)
        accepted_provenance[node_id] = provenance
    pipeline.completed_node_ids = accepted_completed
    pipeline.node_compute_provenance = accepted_provenance


def _publish_cpu_compute_provenance(
    pipeline: PrototypePipeline,
    request: PipelineRunRequest,
    node_ids: Sequence[str] | frozenset[str],
    *,
    source_scientific_contexts: Mapping[str, str],
) -> tuple[NodeExecutionDecision, ...]:
    decisions: list[NodeExecutionDecision] = []
    scheduled_node_ids = frozenset(node_ids)
    for node_id in pipeline.topological_order():
        if node_id not in scheduled_node_ids:
            continue
        node = pipeline.nodes.get(node_id)
        if node is None or not pipeline.operation_spec(node.operation_id).has_input:
            continue
        preference = request.compute_request.preference_for(node_id)
        decisions.append(
            NodeExecutionDecision(
                node_id=node_id,
                operation_id=node.operation_id,
                requested_preference=preference,
                runtime_id="cpu-numpy",
                implementation_library_id="cpu",
                implementation_id=f"cpu-{node.operation_id}-v1",
                decision_kind=DecisionKind.POLICY_CPU,
                reason=DecisionReason.EXPLICIT_CPU,
                reason_text=(
                    "The CPU policy selected the authoritative host implementation."
                ),
                implementation_version="1",
            )
        )
    _publish_actual_compute_provenance(
        pipeline,
        request.compute_request,
        decisions,
        source_scientific_contexts=source_scientific_contexts,
        cancel_callback=(
            request.cancel_event.is_set if request.cancel_event is not None else None
        ),
    )
    return tuple(decisions)


def _publish_actual_compute_provenance(
    pipeline: PrototypePipeline,
    request: ComputeRequest,
    decisions: Sequence[NodeExecutionDecision],
    *,
    implementation_specs: Sequence[OperationComputeSpec] = (),
    source_scientific_contexts: Mapping[str, str],
    cancel_callback: Callable[[], bool] | None,
) -> None:
    """Attach exact chained provenance to materialized host node caches."""

    decisions_by_node = {decision.node_id: decision for decision in decisions}
    prior_provenance = dict(pipeline.node_compute_provenance)
    resolved_provenance: dict[str, CachedNodeComputeProvenance] = {}
    published_provenance: dict[str, CachedNodeComputeProvenance] = {}
    for node_id in pipeline.topological_order():
        node = pipeline.nodes[node_id]
        operation = pipeline.operation_spec(node.operation_id)
        if not operation.has_input:
            scientific_context = source_scientific_contexts.get(node_id)
            if scientific_context is None:
                continue
            try:
                provenance = build_cached_source_provenance(
                    node_id=node_id,
                    operation_id=node.operation_id,
                    scientific_context_fingerprint=scientific_context,
                )
            except (TypeError, ValueError):
                continue
        else:
            try:
                scientific_context = _processing_scientific_context_fingerprint(
                    pipeline,
                    node_id,
                    resolved_provenance,
                    cancel_callback=cancel_callback,
                )
            except (KeyError, TypeError, ValueError):
                continue
            decision = decisions_by_node.get(node_id)
            if decision is None:
                provenance = prior_provenance.get(node_id)
                if provenance is None or not cached_node_provenance_matches(
                    provenance,
                    request=request,
                    node_id=node_id,
                    operation_id=node.operation_id,
                    scientific_context_fingerprint=scientific_context,
                    implementation_specs=implementation_specs,
                ):
                    continue
            else:
                if node.operation_id != decision.operation_id:
                    continue
                try:
                    implementation_spec = next(
                        (
                            spec
                            for spec in implementation_specs
                            if spec.implementation_id == decision.implementation_id
                            and spec.operation_id == decision.operation_id
                            and spec.runtime_id == decision.runtime_id
                            and spec.implementation_library_id
                            == decision.implementation_library_id
                        ),
                        None,
                    )
                    provenance = build_cached_node_compute_provenance(
                        decision,
                        request,
                        scientific_context_fingerprint=scientific_context,
                        implementation_spec=implementation_spec,
                    )
                except (KeyError, TypeError, ValueError):
                    continue
        resolved_provenance[node_id] = provenance
        if node_id in pipeline.completed_node_ids and pipeline._has_cached_output(
            node_id
        ):
            published_provenance[node_id] = provenance
    pipeline.node_compute_provenance = published_provenance


__all__ = [
    "AcceleratorCleanupError",
    "ComputePlanner",
    "NodeFinishedCallback",
    "NodeStartedCallback",
    "PipelineNodeResult",
    "PipelineExecutionFailure",
    "PipelineRunRequest",
    "PipelineRunResult",
    "ProgressCallback",
    "ResidentThumbnailStatisticsCleanupError",
    "ResidentThumbnailStatisticsObservation",
    "ResidentThumbnailStatisticsRequest",
    "execute_pipeline_request",
]
