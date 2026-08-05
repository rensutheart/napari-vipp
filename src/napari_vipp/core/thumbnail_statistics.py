"""Fast, exact, and provider-aware thumbnail contrast statistics.

Thumbnail statistics are presentation data, not scientific pipeline outputs.
This module therefore keeps their backend selection and provenance separate
from operation compute badges while still honoring the user's global compute
intent.  CUDA remains completely lazy: CPU-only and below-threshold Auto
requests neither construct a compute registry nor probe an accelerator.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from napari_vipp.core.compute import ComputeMode
from napari_vipp.core.progress import OperationCancelled, ProgressContext

THUMBNAIL_PERCENTILE_RANGE = (0.5, 99.9)
EXACT_UINT_HISTOGRAM_ALGORITHM_ID = "exact-native-uint-histogram-numpy-linear-v1"
EXACT_NATIVE_MINMAX_ALGORITHM_ID = "exact-native-minmax-chunked-v1"
NUMPY_PERCENTILE_ALGORITHM_ID = "numpy-float32-percentile-v1"
# Kept as a compatibility alias for exported provenance readers.  New results
# use the dtype-preserving, chunked implementation above.
NUMPY_MINMAX_ALGORITHM_ID = EXACT_NATIVE_MINMAX_ALGORITHM_ID
SCAN_FREE_ALGORITHM_ID = "thumbnail-scan-free-contract-v1"

DEFAULT_CPU_CHUNK_ELEMENTS = 1_048_576
# Cold CuPy startup and the first dtype-specific RawKernel compilation are
# substantial.  These conservative crossovers were measured on the project's
# RTX 5090 Windows validation host; they are intentionally not presented as a
# universal hardware guarantee.  Once a dtype's kernel has completed, the
# process-local warm crossover can be much lower.
DEFAULT_COLD_UINT8_GPU_THRESHOLD_BYTES = 384 * 1024**2
DEFAULT_COLD_UINT16_GPU_THRESHOLD_BYTES = 512 * 1024**2
DEFAULT_COLD_GPU_THRESHOLD_BYTES = DEFAULT_COLD_UINT16_GPU_THRESHOLD_BYTES
DEFAULT_WARM_GPU_THRESHOLD_BYTES = 32 * 1024**2
_GPU_RUNTIME_ID = "cuda-cupy"
_GPU_MEMORY_OVERHEAD_BYTES = 8 * 1024**2
_MAX_GPU_HISTOGRAM_COUNTER_BYTES = 64 * 1024**2
_STABLE_GPU_UNAVAILABLE_REASON_CODES = frozenset(
    {
        "runtime_load_failed",
        "runtime_unavailable",
        "dependency_missing",
        "runtime_component_missing",
        "invalid_device",
        "cuda_runtime_failure",
        "cuda_kernel_compile_failure",
        "cupy_missing",
        "cupy_import_failed",
        "cupyx_ndimage_missing",
        "cupyx_ndimage_import_failed",
        "no_cuda_device",
    }
)

type ScalarContrastLimits = tuple[float, float] | None
type ThumbnailContrastLimits = ScalarContrastLimits | tuple[ScalarContrastLimits, ...]


class ThumbnailStatisticsBackend(StrEnum):
    """Actual array backend used for one thumbnail-statistics result."""

    CPU_NUMPY = "cpu-numpy"
    GPU_CUPY = "gpu-cupy"


@dataclass(frozen=True, slots=True)
class ThumbnailStatisticsRequest:
    """One immutable request for reusable thumbnail contrast limits."""

    data: object = field(repr=False, compare=False)
    contrast_mode: str = "Percentile"
    data_kind: str = "image"
    channel_axis: int | None = None
    compute_mode: ComputeMode | str = ComputeMode.AUTO
    device_id: str = ""
    accelerator_memory_cap_bytes: int | None = None
    accelerator_safety_reserve_bytes: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "compute_mode", ComputeMode.parse(self.compute_mode))
        object.__setattr__(self, "contrast_mode", str(self.contrast_mode))
        object.__setattr__(self, "data_kind", str(self.data_kind))
        object.__setattr__(self, "device_id", str(self.device_id).strip())
        if self.channel_axis is not None:
            if isinstance(self.channel_axis, bool):
                raise TypeError("channel_axis must be an integer or None.")
            object.__setattr__(self, "channel_axis", int(self.channel_axis))
        for name in (
            "accelerator_memory_cap_bytes",
            "accelerator_safety_reserve_bytes",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be non-negative or None.")


@dataclass(frozen=True, slots=True)
class ThumbnailStatisticsDecision:
    """Deterministic backend selection made before optional runtime probing."""

    backend: ThumbnailStatisticsBackend
    reason_code: str
    reason: str
    scanned_values: int
    scanned_bytes: int
    threshold_bytes: int
    gpu_warm: bool
    host_staging_bytes: int = 0


@dataclass(frozen=True, slots=True)
class ThumbnailStatisticsResult:
    """Contrast limits plus exact presentation-compute provenance."""

    limits: ThumbnailContrastLimits
    decision: ThumbnailStatisticsDecision
    actual_backend: ThumbnailStatisticsBackend
    algorithm_id: str
    elapsed_seconds: float
    runtime_id: str = ""
    device_id: str = ""
    fallback_reason_code: str = ""
    fallback_message: str = ""

    @property
    def used_fallback(self) -> bool:
        """Return whether an intended GPU calculation visibly used the CPU."""

        return bool(self.fallback_reason_code)


class ThumbnailStatisticsCleanupError(RuntimeError):
    """Raised when accelerator ownership could not be released safely."""


class ThumbnailStatisticsGPUError(RuntimeError):
    """Raised for an unclassified GPU failure that cannot produce a result."""


class ThumbnailStatisticsMemoryError(MemoryError):
    """Raised when an exact float percentile workspace is not safely admitted."""

    def __init__(self, message: str, *, required_bytes: int) -> None:
        super().__init__(message)
        self.required_bytes = max(int(required_bytes), 0)


class _TrackingProgress(ProgressContext):
    """Forward progress while remembering the furthest completed fraction."""

    def __init__(self, delegate: ProgressContext | None) -> None:
        self._delegate = delegate
        self.fraction = 0.0

    def is_cancelled(self) -> bool:
        return bool(self._delegate and self._delegate.is_cancelled())

    def check_cancelled(self) -> None:
        if self._delegate is not None:
            self._delegate.check_cancelled()

    def report(self, current: int, total: int, message: str = "") -> None:
        total = max(int(total), 0)
        current = max(min(int(current), total), 0) if total else max(int(current), 0)
        if total > 0:
            self.fraction = max(self.fraction, min(float(current) / total, 1.0))
        if self._delegate is not None:
            if total > 0:
                self._delegate.report(
                    int(round(self.fraction * _FallbackProgress._SCALE)),
                    _FallbackProgress._SCALE,
                    message,
                )
            else:
                self._delegate.report(current, total, message)


class _FallbackProgress(ProgressContext):
    """Map a CPU fallback onto the unfinished tail of a GPU attempt."""

    _SCALE = 1_000_000

    def __init__(
        self,
        delegate: ProgressContext | None,
        *,
        start_fraction: float,
    ) -> None:
        self._delegate = delegate
        self._start_fraction = max(0.0, min(float(start_fraction), 1.0))

    def is_cancelled(self) -> bool:
        return bool(self._delegate and self._delegate.is_cancelled())

    def check_cancelled(self) -> None:
        if self._delegate is not None:
            self._delegate.check_cancelled()

    def report(self, current: int, total: int, message: str = "") -> None:
        total = max(int(total), 0)
        current = max(min(int(current), total), 0) if total else max(int(current), 0)
        fraction = float(current) / total if total > 0 else 0.0
        mapped = self._start_fraction + (1.0 - self._start_fraction) * fraction
        prefix = "CPU fallback"
        detail = str(message).strip()
        if detail:
            prefix += f" · {detail}"
        if self._delegate is not None:
            self._delegate.report(
                int(round(mapped * self._SCALE)),
                self._SCALE,
                prefix,
            )


class _ProgressRange(ProgressContext):
    """Map one scalar/channel calculation onto its share of a batch scan."""

    def __init__(
        self,
        delegate: ProgressContext | None,
        *,
        start: int,
        span: int,
        total: int,
    ) -> None:
        self._delegate = delegate
        self._start = max(int(start), 0)
        self._span = max(int(span), 0)
        self._total = max(int(total), 0)

    def is_cancelled(self) -> bool:
        return bool(self._delegate and self._delegate.is_cancelled())

    def check_cancelled(self) -> None:
        if self._delegate is not None:
            self._delegate.check_cancelled()

    def report(self, current: int, total: int, message: str = "") -> None:
        if self._delegate is None:
            return
        local_total = max(int(total), 0)
        fraction = (
            min(max(float(current) / local_total, 0.0), 1.0)
            if local_total > 0
            else 0.0
        )
        mapped = self._start + int(round(self._span * fraction))
        self._delegate.report(mapped, self._total, message)


@dataclass(frozen=True, slots=True)
class _GPUAttempt:
    counts: np.ndarray | None
    runtime_id: str = ""
    device_id: str = ""
    failure_reason_code: str = ""
    failure_message: str = ""


class ThumbnailStatisticsEngine:
    """Select and execute exact thumbnail statistics without retaining a runtime.

    ``gpu_warm`` is process-session evidence that this engine has completed the
    thumbnail histogram kernel at least once.  It changes only the conservative
    Auto crossover; every request still receives its own private runtime scope,
    and every registry is closed before :meth:`calculate` returns.
    """

    def __init__(
        self,
        *,
        cold_gpu_threshold_bytes: int | None = None,
        cold_uint8_gpu_threshold_bytes: int = (
            DEFAULT_COLD_UINT8_GPU_THRESHOLD_BYTES
        ),
        cold_uint16_gpu_threshold_bytes: int = (
            DEFAULT_COLD_UINT16_GPU_THRESHOLD_BYTES
        ),
        warm_gpu_threshold_bytes: int = DEFAULT_WARM_GPU_THRESHOLD_BYTES,
        cpu_chunk_elements: int = DEFAULT_CPU_CHUNK_ELEMENTS,
        registry_factory: Callable[[], object] | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if cold_gpu_threshold_bytes is not None:
            shared_cold_threshold = _positive_integer(
                cold_gpu_threshold_bytes,
                "cold_gpu_threshold_bytes",
            )
            cold_uint8_gpu_threshold_bytes = shared_cold_threshold
            cold_uint16_gpu_threshold_bytes = shared_cold_threshold
        self._cold_gpu_threshold_bytes = {
            np.dtype(np.uint8): _positive_integer(
                cold_uint8_gpu_threshold_bytes,
                "cold_uint8_gpu_threshold_bytes",
            ),
            np.dtype(np.uint16): _positive_integer(
                cold_uint16_gpu_threshold_bytes,
                "cold_uint16_gpu_threshold_bytes",
            ),
        }
        self._warm_gpu_threshold_bytes = _positive_integer(
            warm_gpu_threshold_bytes,
            "warm_gpu_threshold_bytes",
        )
        if any(
            self._warm_gpu_threshold_bytes > threshold
            for threshold in self._cold_gpu_threshold_bytes.values()
        ):
            raise ValueError(
                "warm_gpu_threshold_bytes must not exceed either cold threshold."
            )
        self._cpu_chunk_elements = _positive_integer(
            cpu_chunk_elements,
            "cpu_chunk_elements",
        )
        if registry_factory is not None and not callable(registry_factory):
            raise TypeError("registry_factory must be callable or None.")
        if not callable(clock):
            raise TypeError("clock must be callable.")
        self._registry_factory = registry_factory
        self._clock = clock
        self._warm_gpu_dtypes: set[np.dtype] = set()
        self._stable_gpu_unavailable: tuple[str, str] | None = None

    @property
    def gpu_warm(self) -> bool:
        """Return whether this engine has completed any CuPy histogram kernel."""

        return bool(self._warm_gpu_dtypes)

    def reset_accelerator_capability(self) -> None:
        """Retry a previously unavailable CUDA capability on the next request.

        Stable negative probes are cached for this engine session so a machine
        without CuPy or a CUDA device is not repeatedly probed once per node.
        Explicit presentation/compute-policy changes call this method and act
        as the user's retry boundary.
        """

        self._stable_gpu_unavailable = None

    def select(
        self,
        request: ThumbnailStatisticsRequest,
    ) -> ThumbnailStatisticsDecision:
        """Select a backend without importing, constructing, or probing CUDA."""

        if not isinstance(request, ThumbnailStatisticsRequest):
            raise TypeError("request must be a ThumbnailStatisticsRequest.")
        arr = np.asarray(request.data)
        normalized_axis = _normalized_channel_axis_or_none(
            request.channel_axis,
            arr.ndim,
        )
        invalid_channel_axis = (
            request.channel_axis is not None and normalized_axis is None
        )
        scan_required = (
            request.data is not None
            and not invalid_channel_axis
            and _scan_required(
                arr,
                request.contrast_mode,
                request.data_kind,
            )
        )
        scanned_values = int(arr.size) if scan_required else 0
        scanned_bytes = int(arr.nbytes) if scan_required else 0
        host_staging_bytes = (
            int(arr.nbytes)
            if scan_required and not bool(arr.flags.c_contiguous)
            else 0
        )
        mode = request.compute_mode
        effective_auto = mode in {ComputeMode.AUTO, ComputeMode.CUSTOM}
        gpu_warm = arr.dtype in self._warm_gpu_dtypes
        threshold = (
            self._warm_gpu_threshold_bytes
            if gpu_warm
            else self._cold_gpu_threshold_bytes.get(
                arr.dtype,
                DEFAULT_COLD_GPU_THRESHOLD_BYTES,
            )
        )

        if not scan_required:
            return ThumbnailStatisticsDecision(
                ThumbnailStatisticsBackend.CPU_NUMPY,
                "scan_free",
                "The requested thumbnail contract does not need an array scan.",
                scanned_values,
                scanned_bytes,
                threshold,
                gpu_warm,
                0,
            )
        if mode is ComputeMode.CPU:
            return ThumbnailStatisticsDecision(
                ThumbnailStatisticsBackend.CPU_NUMPY,
                "cpu_requested",
                "CPU compute mode applies to thumbnail presentation work.",
                scanned_values,
                scanned_bytes,
                threshold,
                gpu_warm,
                0,
            )
        gpu_ineligible_reason = _gpu_histogram_ineligibility(
            arr,
            request.contrast_mode,
            request.data_kind,
            normalized_axis,
        )
        if gpu_ineligible_reason is not None:
            reason_code, reason = gpu_ineligible_reason
            return ThumbnailStatisticsDecision(
                ThumbnailStatisticsBackend.CPU_NUMPY,
                reason_code,
                reason,
                scanned_values,
                scanned_bytes,
                threshold,
                gpu_warm,
                host_staging_bytes,
            )

        required_device_bytes = _estimated_gpu_bytes(arr, normalized_axis)
        memory_cap = request.accelerator_memory_cap_bytes
        if memory_cap is not None and required_device_bytes > memory_cap:
            return ThumbnailStatisticsDecision(
                ThumbnailStatisticsBackend.CPU_NUMPY,
                "gpu_memory_cap_insufficient",
                "The configured accelerator memory cap is smaller than the "
                "bounded thumbnail histogram allocation estimate.",
                scanned_values,
                scanned_bytes,
                threshold,
                gpu_warm,
                host_staging_bytes,
            )
        if effective_auto and scanned_bytes < threshold:
            reason_code = (
                "auto_below_warm_gpu_threshold"
                if gpu_warm
                else "auto_below_cold_gpu_threshold"
            )
            return ThumbnailStatisticsDecision(
                ThumbnailStatisticsBackend.CPU_NUMPY,
                reason_code,
                "The exact host workload is below the conservative GPU "
                "transfer and startup crossover.",
                scanned_values,
                scanned_bytes,
                threshold,
                gpu_warm,
                host_staging_bytes,
            )
        if effective_auto and host_staging_bytes:
            return ThumbnailStatisticsDecision(
                ThumbnailStatisticsBackend.CPU_NUMPY,
                "auto_noncontiguous_host_staging",
                "Auto kept this non-contiguous result on CPU because CuPy "
                "would first need a full contiguous host staging copy. Prefer "
                "GPU remains available as an explicit override.",
                scanned_values,
                scanned_bytes,
                threshold,
                gpu_warm,
                host_staging_bytes,
            )
        if host_staging_bytes:
            staging_rejection = _host_allocation_rejection(
                host_staging_bytes,
                purpose="contiguous thumbnail GPU upload staging",
            )
            if staging_rejection:
                return ThumbnailStatisticsDecision(
                    ThumbnailStatisticsBackend.CPU_NUMPY,
                    "gpu_host_staging_memory_insufficient",
                    staging_rejection,
                    scanned_values,
                    scanned_bytes,
                    threshold,
                    gpu_warm,
                    host_staging_bytes,
                )
        if self._stable_gpu_unavailable is not None:
            unavailable_code, unavailable_message = self._stable_gpu_unavailable
            return ThumbnailStatisticsDecision(
                ThumbnailStatisticsBackend.CPU_NUMPY,
                "gpu_session_unavailable",
                "GPU thumbnail statistics are unavailable for this session "
                f"({unavailable_code}): {unavailable_message}",
                scanned_values,
                scanned_bytes,
                threshold,
                gpu_warm,
                host_staging_bytes,
            )
        return ThumbnailStatisticsDecision(
            ThumbnailStatisticsBackend.GPU_CUPY,
            (
                "prefer_gpu_eligible"
                if mode is ComputeMode.PREFER_GPU
                else "auto_gpu_threshold_met"
            ),
            (
                "Prefer GPU selected every eligible thumbnail implementation."
                if mode is ComputeMode.PREFER_GPU
                else "The exact host workload meets the conservative GPU crossover."
            ),
            scanned_values,
            scanned_bytes,
            0 if mode is ComputeMode.PREFER_GPU else threshold,
            gpu_warm,
            host_staging_bytes,
        )

    def calculate(
        self,
        request: ThumbnailStatisticsRequest,
        *,
        progress: ProgressContext | None = None,
    ) -> ThumbnailStatisticsResult:
        """Calculate exact limits using the selected backend and safe fallback."""

        decision = self.select(request)
        arr = np.asarray(request.data)
        started = self._clock()
        if decision.backend is ThumbnailStatisticsBackend.CPU_NUMPY:
            limits, algorithm_id = _calculate_cpu_limits(
                arr,
                contrast_mode=request.contrast_mode,
                data_kind=request.data_kind,
                channel_axis=request.channel_axis,
                progress=progress,
                chunk_elements=self._cpu_chunk_elements,
            )
            fallback_reason_code = ""
            fallback_message = ""
            if (
                request.compute_mode is ComputeMode.PREFER_GPU
                and decision.reason_code not in {"cpu_requested", "scan_free"}
            ) or decision.reason_code == "gpu_session_unavailable":
                fallback_reason_code = decision.reason_code
                fallback_message = decision.reason
            return ThumbnailStatisticsResult(
                limits,
                decision,
                ThumbnailStatisticsBackend.CPU_NUMPY,
                algorithm_id,
                _elapsed(self._clock, started),
                fallback_reason_code=fallback_reason_code,
                fallback_message=fallback_message,
            )

        gpu_progress = _TrackingProgress(progress)
        attempt = self._calculate_gpu_counts(
            request,
            arr,
            progress=gpu_progress,
        )
        if attempt.counts is not None:
            limits = _limits_from_counts(
                attempt.counts,
                contrast_mode=request.contrast_mode,
            )
            self._warm_gpu_dtypes.add(arr.dtype)
            return ThumbnailStatisticsResult(
                limits,
                decision,
                ThumbnailStatisticsBackend.GPU_CUPY,
                EXACT_UINT_HISTOGRAM_ALGORITHM_ID,
                _elapsed(self._clock, started),
                runtime_id=attempt.runtime_id,
                device_id=attempt.device_id,
            )

        fallback_progress = _FallbackProgress(
            progress,
            start_fraction=gpu_progress.fraction,
        )
        limits, algorithm_id = _calculate_cpu_limits(
            arr,
            contrast_mode=request.contrast_mode,
            data_kind=request.data_kind,
            channel_axis=request.channel_axis,
            progress=fallback_progress,
            chunk_elements=self._cpu_chunk_elements,
        )
        return ThumbnailStatisticsResult(
            limits,
            decision,
            ThumbnailStatisticsBackend.CPU_NUMPY,
            algorithm_id,
            _elapsed(self._clock, started),
            runtime_id=attempt.runtime_id,
            device_id=attempt.device_id,
            fallback_reason_code=attempt.failure_reason_code,
            fallback_message=attempt.failure_message,
        )

    def _calculate_gpu_counts(
        self,
        request: ThumbnailStatisticsRequest,
        arr: np.ndarray,
        *,
        progress: ProgressContext | None,
    ) -> _GPUAttempt:
        registry = None
        runtime = None
        runtime_id = _GPU_RUNTIME_ID
        device_id = request.device_id
        counts = None
        failure_info = None
        pending_error: BaseException | None = None
        try:
            _check_cancelled(progress)
            _report(
                progress,
                0,
                0,
                "Probing the CuPy runtime · this probe may not be interruptible",
            )
            factory = self._registry_factory or _default_registry_factory
            registry = factory()
            _check_cancelled(progress)
            probe = registry.probe_runtime(runtime_id)
            _check_cancelled(progress)
            if not probe.available:
                reason_code = probe.reason_code or "runtime_unavailable"
                message = probe.message or "CuPy GPU execution is unavailable."
                if reason_code in _STABLE_GPU_UNAVAILABLE_REASON_CODES:
                    self._stable_gpu_unavailable = (reason_code, message)
                return _GPUAttempt(
                    None,
                    runtime_id,
                    device_id,
                    reason_code,
                    message,
                )
            device_id = device_id or probe.selected_device_id
            runtime = registry.runtime(runtime_id)
            _check_cancelled(progress)
            try:
                with runtime.execution_scope(
                    device_id=device_id,
                    memory_limit_bytes=request.accelerator_memory_cap_bytes,
                    safety_reserve_bytes=request.accelerator_safety_reserve_bytes,
                ):
                    counts = _calculate_cupy_counts(
                        runtime,
                        arr,
                        device_id=device_id,
                        channel_axis=request.channel_axis,
                        progress=progress,
                    )
            except BaseException as exc:
                pending_error = exc
                if not isinstance(exc, OperationCancelled):
                    try:
                        failure_info = runtime.classify_exception(exc)
                    except Exception:
                        failure_info = None
        except BaseException as exc:
            pending_error = exc
        finally:
            if registry is not None:
                try:
                    registry.close()
                except BaseException as cleanup_exc:
                    raise ThumbnailStatisticsCleanupError(
                        "Thumbnail GPU runtime cleanup failed; no CPU fallback "
                        "was accepted. Restart accelerator work before retrying: "
                        f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                    ) from (pending_error or cleanup_exc)

        if pending_error is None:
            return _GPUAttempt(np.asarray(counts), runtime_id, device_id)
        if isinstance(pending_error, OperationCancelled):
            raise pending_error
        if failure_info is not None and (
            failure_info.reason_code == "cuda_cleanup_incomplete"
            or (
                failure_info.kind.value == "kernel_failure"
                and "cleanup" in failure_info.reason_code
            )
        ):
            raise ThumbnailStatisticsCleanupError(
                "Thumbnail GPU runtime cleanup failed; no CPU fallback was "
                f"accepted: {failure_info.message}"
            ) from pending_error
        if failure_info is not None:
            return _GPUAttempt(
                None,
                runtime_id,
                device_id,
                failure_info.reason_code,
                failure_info.message,
            )
        if isinstance(pending_error, Exception):
            return _GPUAttempt(
                None,
                runtime_id,
                device_id,
                "gpu_execution_failed",
                f"{type(pending_error).__name__}: {pending_error}",
            )
        raise pending_error


def exact_uint_thumbnail_contrast_limits(
    data,
    *,
    contrast_mode: str = "Percentile",
    data_kind: str = "image",
    channel_axis: int | None = None,
    progress: ProgressContext | None = None,
    chunk_elements: int = DEFAULT_CPU_CHUNK_ELEMENTS,
) -> ThumbnailContrastLimits:
    """Return exact current-semantics limits for native uint8/uint16 data.

    Percentiles use bounded exact histograms.  Min-max uses a cheaper native
    chunked reduction and never allocates the 65,536-bin uint16 table.
    """

    arr = np.asarray(data)
    if arr.dtype not in {np.dtype(np.uint8), np.dtype(np.uint16)}:
        raise TypeError("Exact thumbnail histograms require native uint8 or uint16.")
    mode = _contrast_mode_key(contrast_mode)
    normalized_kind = _data_kind_key(data_kind)
    if channel_axis is not None:
        if arr.size == 0 or arr.ndim == 0:
            return ()
        axis = _normalized_channel_axis_or_none(channel_axis, arr.ndim)
        if axis is None:
            return ()
        channel_count = int(arr.shape[axis])
        if normalized_kind in {"label", "labels", "label image", "table"}:
            return tuple(None for _channel in range(channel_count))
        if normalized_kind == "mask":
            return tuple((0.0, 1.0) for _channel in range(channel_count))
        if mode == "raw":
            return tuple(None for _channel in range(channel_count))
    else:
        axis = None
        if normalized_kind in {"label", "labels", "label image", "table"}:
            return None
        if normalized_kind == "mask":
            return (0.0, 1.0)
        if mode == "raw":
            return None
    if arr.size == 0:
        return (0.0, 0.0)
    if mode == "minmax":
        return _exact_native_minmax_limits(
            arr,
            contrast_mode=mode,
            data_kind=normalized_kind,
            channel_axis=axis,
            progress=progress,
            chunk_elements=_positive_integer(chunk_elements, "chunk_elements"),
        )
    chunk_elements = _positive_integer(chunk_elements, "chunk_elements")
    if axis is not None:
        return _exact_uint_channel_histogram_limits(
            arr,
            channel_axis=axis,
            progress=progress,
            chunk_elements=chunk_elements,
        )
    counts = _exact_uint_histogram_counts(
        arr,
        channel_axis=None,
        progress=progress,
        chunk_elements=chunk_elements,
    )
    return _limits_from_counts(counts, contrast_mode=mode)


def _calculate_cpu_limits(
    arr: np.ndarray,
    *,
    contrast_mode: str,
    data_kind: str,
    channel_axis: int | None,
    progress: ProgressContext | None,
    chunk_elements: int,
) -> tuple[ThumbnailContrastLimits, str]:
    scan_required = _scan_required(arr, contrast_mode, data_kind)
    if (
        channel_axis is not None
        and _normalized_channel_axis_or_none(channel_axis, arr.ndim) is None
    ):
        scan_required = False
    mode = _contrast_mode_key(contrast_mode)
    if scan_required and (mode == "minmax" or mode == "raw"):
        limits = _exact_native_minmax_limits(
            arr,
            contrast_mode=mode,
            data_kind=data_kind,
            channel_axis=channel_axis,
            progress=progress,
            chunk_elements=chunk_elements,
        )
        return limits, EXACT_NATIVE_MINMAX_ALGORITHM_ID
    if arr.dtype in {np.dtype(np.uint8), np.dtype(np.uint16)}:
        limits = exact_uint_thumbnail_contrast_limits(
            arr,
            contrast_mode=contrast_mode,
            data_kind=data_kind,
            channel_axis=channel_axis,
            progress=progress,
            chunk_elements=chunk_elements,
        )
        algorithm_id = (
            SCAN_FREE_ALGORITHM_ID
            if not scan_required
            else EXACT_UINT_HISTOGRAM_ALGORITHM_ID
        )
        return limits, algorithm_id
    limits = _numpy_compatible_limits(
        arr,
        contrast_mode=contrast_mode,
        data_kind=data_kind,
        channel_axis=channel_axis,
        progress=progress,
        chunk_elements=chunk_elements,
    )
    if not scan_required:
        algorithm_id = SCAN_FREE_ALGORITHM_ID
    else:
        algorithm_id = NUMPY_PERCENTILE_ALGORITHM_ID
    return limits, algorithm_id


def _numpy_compatible_limits(
    arr: np.ndarray,
    *,
    contrast_mode: str,
    data_kind: str,
    channel_axis: int | None,
    progress: ProgressContext | None,
    chunk_elements: int,
) -> ThumbnailContrastLimits:
    if channel_axis is not None:
        if arr.size == 0 or arr.ndim == 0:
            return ()
        axis = _normalized_channel_axis_or_none(channel_axis, arr.ndim)
        if axis is None:
            return ()
        limits = []
        completed = 0
        overall_total = int(arr.size)
        for channel in range(arr.shape[axis]):
            _check_cancelled(progress)
            channel_values = _axis_view(arr, axis, channel)
            limits.append(
                _numpy_scalar_limits(
                    channel_values,
                    contrast_mode=contrast_mode,
                    data_kind=data_kind,
                    progress=_ProgressRange(
                        progress,
                        start=completed,
                        span=int(channel_values.size),
                        total=overall_total,
                    ),
                    chunk_elements=chunk_elements,
                )
            )
            completed += int(channel_values.size)
        return tuple(limits)
    limits = _numpy_scalar_limits(
        arr,
        contrast_mode=contrast_mode,
        data_kind=data_kind,
        progress=progress,
        chunk_elements=chunk_elements,
    )
    return limits


def _numpy_scalar_limits(
    arr: np.ndarray,
    *,
    contrast_mode: str,
    data_kind: str,
    progress: ProgressContext | None,
    chunk_elements: int,
) -> ScalarContrastLimits:
    if arr.ndim == 0 and arr.dtype == np.dtype(object) and arr.item() is None:
        return None
    normalized_kind = _data_kind_key(data_kind)
    if normalized_kind in {"label", "labels", "label image", "table"}:
        return None
    if normalized_kind == "mask":
        return (0.0, 1.0)
    if arr.size == 0:
        return (0.0, 0.0)

    mode = _contrast_mode_key(contrast_mode)
    if mode == "raw" and (
        arr.dtype == bool or np.issubdtype(arr.dtype, np.integer)
    ):
        return None
    workspace = _build_float_percentile_workspace(
        arr,
        progress=progress,
        chunk_elements=chunk_elements,
        clip_negative=mode != "raw",
    )
    if workspace.size == 0:
        return (0.0, 0.0)
    _check_cancelled(progress)
    _report(
        progress,
        9,
        10,
        "Exact NumPy percentile selection · cancel applies after this pass",
    )
    lo, hi = (
        float(value)
        for value in np.percentile(
            workspace,
            THUMBNAIL_PERCENTILE_RANGE,
            overwrite_input=True,
        )
    )
    _report(progress, 10, 10, "Exact thumbnail percentile ready")
    if hi <= lo:
        hi = float(workspace.max())
        lo = float(workspace.min())
    return (lo, hi)


def _build_float_percentile_workspace(
    arr: np.ndarray,
    *,
    progress: ProgressContext | None,
    chunk_elements: int,
    clip_negative: bool,
) -> np.ndarray:
    """Build one compact float32 workspace without a second full finite copy."""

    chunk_elements = _positive_integer(chunk_elements, "chunk_elements")
    total = int(arr.size)
    required_bytes = total * np.dtype(np.float32).itemsize
    # One converted chunk plus its finite mask is the only material temporary
    # beyond the compact workspace.  NumPy may use a small selection workspace
    # even with overwrite_input=True, so include the bounded chunk explicitly.
    required_peak = required_bytes + min(total, chunk_elements) * (
        np.dtype(np.float32).itemsize + np.dtype(bool).itemsize
    )
    _admit_float_percentile_workspace(required_peak)

    workspace = np.empty(total, dtype=np.float32)
    completed = 0
    retained = 0
    has_negative_finite = False
    negative_infinity_count = 0
    _report(
        progress,
        0,
        max(total * 10, 1),
        "Preparing bounded exact float percentile workspace",
    )
    for chunk in _iter_chunks(arr, chunk_elements):
        _check_cancelled(progress)
        values = np.asarray(chunk).astype(np.float32, copy=False)
        finite_values = values[np.isfinite(values)]
        if clip_negative and finite_values.size:
            # finite_values is already the bounded per-chunk copy produced by
            # boolean indexing, so clipping it in place cannot mutate pipeline
            # output data.
            chunk_has_negative = bool(np.any(finite_values < np.float32(0.0)))
            has_negative_finite = has_negative_finite or chunk_has_negative
            np.maximum(finite_values, np.float32(0.0), out=finite_values)
        if clip_negative:
            negative_infinity_count += int(np.count_nonzero(np.isneginf(values)))
        next_retained = retained + int(finite_values.size)
        workspace[retained:next_retained] = finite_values
        retained = next_retained
        completed += int(values.size)
        _report(
            progress,
            completed * 9,
            max(total * 10, 1),
            "Preparing bounded exact float percentile workspace",
        )
    if has_negative_finite and negative_infinity_count:
        next_retained = retained + negative_infinity_count
        workspace[retained:next_retained] = np.float32(0.0)
        retained = next_retained
    return workspace[:retained]


def _admit_float_percentile_workspace(required_bytes: int) -> None:
    """Reject only measured low-headroom workspaces; tolerate unknown probes."""

    # Small workspaces are bounded by the ordinary process reserve and do not
    # justify a native memory probe for every tiny node.
    if required_bytes < 64 * 1024**2:
        return
    rejection = _host_allocation_rejection(
        required_bytes,
        purpose="exact thumbnail percentile workspace",
    )
    if not rejection:
        return
    raise ThumbnailStatisticsMemoryError(
        rejection,
        required_bytes=required_bytes,
    )


def _host_allocation_rejection(required_bytes: int, *, purpose: str) -> str:
    """Return a measured low-headroom reason, or allow unknown observations."""

    from napari_vipp.core.host_memory import (
        HostMemoryPreflightReason,
        capture_host_memory,
        preflight_host_allocation,
    )

    snapshot = capture_host_memory()
    admission = preflight_host_allocation(
        snapshot,
        required_bytes=max(int(required_bytes), 0),
        purpose=purpose,
    )
    if admission.allowed or admission.reason_code in {
        HostMemoryPreflightReason.SNAPSHOT_UNAVAILABLE,
        HostMemoryPreflightReason.COMMIT_HEADROOM_UNAVAILABLE,
    }:
        return ""
    return admission.reason


def _exact_native_minmax_limits(
    arr: np.ndarray,
    *,
    contrast_mode: str,
    data_kind: str,
    channel_axis: int | None,
    progress: ProgressContext | None,
    chunk_elements: int,
) -> ThumbnailContrastLimits:
    """Return float32-compatible finite min/max with bounded cancellation."""

    normalized_kind = _data_kind_key(data_kind)
    if channel_axis is not None:
        if arr.size == 0 or arr.ndim == 0:
            return ()
        axis = _normalized_channel_axis_or_none(channel_axis, arr.ndim)
        if axis is None:
            return ()
        if normalized_kind in {"label", "labels", "label image", "table"}:
            return tuple(None for _channel in range(arr.shape[axis]))
        if normalized_kind == "mask":
            return tuple((0.0, 1.0) for _channel in range(arr.shape[axis]))
    else:
        axis = None
        if normalized_kind in {"label", "labels", "label image", "table"}:
            return None
        if normalized_kind == "mask":
            return (0.0, 1.0)
    if arr.size == 0:
        return (0.0, 0.0)

    mode = _contrast_mode_key(contrast_mode)
    clip_negative = mode != "raw"
    channel_count = 1 if axis is None else int(arr.shape[axis])
    total = int(arr.size)
    completed = 0
    limits: list[tuple[float, float]] = []
    native_integer = arr.dtype == bool or np.issubdtype(arr.dtype, np.integer)
    _report(progress, 0, total, "Reducing exact thumbnail min-max in chunks")
    for channel in range(channel_count):
        values = arr if axis is None else _axis_view(arr, axis, channel)
        minimum = np.float32(np.inf)
        maximum = np.float32(-np.inf)
        for chunk in _iter_chunks(values, chunk_elements):
            _check_cancelled(progress)
            chunk_values = np.asarray(chunk)
            if native_integer:
                # Integer-to-float32 conversion is monotonic across every NumPy
                # integer dtype.  Reducing native extrema first therefore
                # preserves the existing float32-compatible endpoints while
                # avoiding a conversion and finite-mask copy of every value.
                chunk_min = np.float32(chunk_values.min())
                chunk_max = np.float32(chunk_values.max())
                has_finite = True
            else:
                converted = chunk_values.astype(np.float32, copy=False)
                finite = converted[np.isfinite(converted)]
                has_finite = bool(finite.size)
                if has_finite:
                    chunk_min = np.float32(finite.min())
                    chunk_max = np.float32(finite.max())
            if has_finite:
                if clip_negative:
                    chunk_min = np.maximum(chunk_min, np.float32(0.0))
                    chunk_max = np.maximum(chunk_max, np.float32(0.0))
                minimum = np.minimum(minimum, chunk_min)
                maximum = np.maximum(maximum, chunk_max)
            completed += int(chunk_values.size)
            message = "Reducing exact thumbnail min-max in chunks"
            if channel_count > 1:
                message += f" · channel {channel + 1}/{channel_count}"
            _report(progress, completed, total, message)
        if not native_integer and (
            not np.isfinite(minimum) or not np.isfinite(maximum)
        ):
            limits.append((0.0, 0.0))
        else:
            limits.append((float(minimum), float(maximum)))
    return limits[0] if axis is None else tuple(limits)


def _exact_uint_channel_histogram_limits(
    arr: np.ndarray,
    *,
    channel_axis: int,
    progress: ProgressContext | None,
    chunk_elements: int,
) -> tuple[ScalarContrastLimits, ...]:
    """Stream channel histograms so spectral data cannot multiply RAM use."""

    levels = 256 if arr.dtype == np.dtype(np.uint8) else 65_536
    channel_count = int(arr.shape[channel_axis])
    total = int(arr.size)
    completed = 0
    limits: list[ScalarContrastLimits] = []
    _report(progress, 0, total, "Counting exact thumbnail intensity levels")
    for channel in range(channel_count):
        counts = np.zeros(levels, dtype=np.uint64)
        values = _axis_view(arr, channel_axis, channel)
        for chunk in _iter_chunks(values, chunk_elements):
            _check_cancelled(progress)
            indices = np.asarray(chunk).astype(np.intp, copy=False)
            chunk_counts = np.bincount(indices, minlength=levels)
            counts += chunk_counts.astype(np.uint64, copy=False)
            completed += int(indices.size)
            _report(
                progress,
                completed,
                total,
                "Counting exact thumbnail intensity levels · "
                f"channel {channel + 1}/{channel_count}",
            )
        limits.append(_scalar_limits_from_counts(counts, contrast_mode="percentile"))
    return tuple(limits)


def _exact_uint_histogram_counts(
    arr: np.ndarray,
    *,
    channel_axis: int | None,
    progress: ProgressContext | None,
    chunk_elements: int,
) -> np.ndarray:
    levels = 256 if arr.dtype == np.dtype(np.uint8) else 65_536
    channel_count = 1 if channel_axis is None else int(arr.shape[channel_axis])
    counts = np.zeros((channel_count, levels), dtype=np.uint64)
    total = int(arr.size)
    completed = 0
    _report(progress, 0, total, "Counting exact thumbnail intensity levels")
    for channel in range(channel_count):
        values = arr if channel_axis is None else _axis_view(arr, channel_axis, channel)
        for chunk in _iter_chunks(values, chunk_elements):
            _check_cancelled(progress)
            indices = np.asarray(chunk).astype(np.intp, copy=False)
            chunk_counts = np.bincount(indices, minlength=levels)
            counts[channel] += chunk_counts.astype(np.uint64, copy=False)
            completed += int(indices.size)
            message = "Counting exact thumbnail intensity levels"
            if channel_count > 1:
                message += f" · channel {channel + 1}/{channel_count}"
            _report(progress, completed, total, message)
    return counts[0] if channel_axis is None else counts


def _limits_from_counts(
    counts: np.ndarray,
    *,
    contrast_mode: str,
) -> ThumbnailContrastLimits:
    values = np.asarray(counts)
    if values.ndim == 1:
        return _scalar_limits_from_counts(values, contrast_mode=contrast_mode)
    if values.ndim != 2:
        raise ValueError("Thumbnail histogram counts must be one- or two-dimensional.")
    return tuple(
        _scalar_limits_from_counts(row, contrast_mode=contrast_mode) for row in values
    )


def _scalar_limits_from_counts(
    counts: np.ndarray,
    *,
    contrast_mode: str,
) -> tuple[float, float]:
    counts = np.asarray(counts)
    nonzero = np.flatnonzero(counts)
    if not nonzero.size:
        return (0.0, 0.0)
    minimum = float(nonzero[0])
    maximum = float(nonzero[-1])
    if _contrast_mode_key(contrast_mode) == "minmax":
        return (minimum, maximum)

    total = int(counts.sum(dtype=np.uint64))
    cdf = np.cumsum(counts, dtype=np.uint64)
    quantiles = np.true_divide(
        np.asarray(THUMBNAIL_PERCENTILE_RANGE, dtype=np.float64),
        100.0,
    )
    virtual = (total - 1) * quantiles
    previous_indices = np.floor(virtual).astype(np.intp)
    next_indices = np.minimum(previous_indices + 1, total - 1)
    gamma = virtual - previous_indices
    previous = np.searchsorted(cdf, previous_indices, side="right").astype(np.float32)
    following = np.searchsorted(cdf, next_indices, side="right").astype(np.float32)

    # NumPy's current linear percentile uses a precision-preserving upper-side
    # interpolation when gamma >= 0.5.  Mirroring both arithmetic order and the
    # float32 selected-value dtype makes uint8/uint16 results bit-for-bit equal
    # to np.percentile(arr.astype(float32), ..., method="linear").
    difference = np.subtract(following, previous)
    interpolated = np.add(previous, difference * gamma)
    np.subtract(
        following,
        difference * (1.0 - gamma),
        out=interpolated,
        where=gamma >= 0.5,
        casting="unsafe",
        dtype=interpolated.dtype,
    )
    lo, hi = (float(value) for value in interpolated)
    if hi <= lo:
        return (minimum, maximum)
    return (lo, hi)


def _calculate_cupy_counts(
    runtime,
    arr: np.ndarray,
    *,
    device_id: str,
    channel_axis: int | None,
    progress: ProgressContext | None,
) -> np.ndarray:
    # Importing this adapter is itself deferred until the selector has chosen
    # GPU and the registry probe has succeeded.
    from napari_vipp.core.gpu.cupy_thumbnail_statistics import (
        exact_uint_histogram_counts,
    )

    return exact_uint_histogram_counts(
        runtime,
        arr,
        device_id=device_id,
        channel_axis=channel_axis,
        progress=progress,
    )


def _default_registry_factory():
    from napari_vipp.core.compute_registry import ComputeRegistry

    return ComputeRegistry()


def _gpu_histogram_ineligibility(
    arr: np.ndarray,
    contrast_mode: str,
    data_kind: str,
    channel_axis: int | None,
) -> tuple[str, str] | None:
    if _contrast_mode_key(contrast_mode) != "percentile":
        return (
            "gpu_ineligible",
            "Min-max uses the faster exact chunked CPU reduction; the CuPy "
            "histogram is reserved for percentile statistics.",
        )
    if arr.dtype not in {np.dtype(np.uint8), np.dtype(np.uint16)}:
        return (
            "gpu_ineligible",
            "Exact GPU thumbnail percentiles currently support native uint8 "
            "and uint16 image data.",
        )
    if _data_kind_key(data_kind) in {
        "label",
        "labels",
        "label image",
        "mask",
        "table",
    }:
        return (
            "gpu_ineligible",
            "This thumbnail data kind does not use GPU percentile statistics.",
        )
    counter_bytes = _estimated_histogram_counter_bytes(arr, channel_axis)
    if counter_bytes > _MAX_GPU_HISTOGRAM_COUNTER_BYTES:
        return (
            "gpu_counter_allocation_too_large",
            "The channel histogram counter matrix would exceed the bounded "
            "GPU presentation limit; CPU streams one channel at a time.",
        )
    return None


def _scan_required(arr: np.ndarray, contrast_mode: str, data_kind: str) -> bool:
    if arr.ndim == 0 and arr.dtype == np.dtype(object) and arr.item() is None:
        return False
    kind = _data_kind_key(data_kind)
    if kind in {"label", "labels", "label image", "mask", "table"}:
        return False
    if arr.size == 0:
        return False
    mode = _contrast_mode_key(contrast_mode)
    if mode == "raw" and (arr.dtype == bool or np.issubdtype(arr.dtype, np.integer)):
        return False
    return True


def _estimated_gpu_bytes(arr: np.ndarray, channel_axis: int | None) -> int:
    return (
        int(arr.nbytes)
        + _estimated_histogram_counter_bytes(arr, channel_axis)
        + _GPU_MEMORY_OVERHEAD_BYTES
    )


def _estimated_histogram_counter_bytes(
    arr: np.ndarray,
    channel_axis: int | None,
) -> int:
    channel_count = 1 if channel_axis is None else int(arr.shape[channel_axis])
    level_count = 256 if arr.dtype == np.dtype(np.uint8) else 65_536
    return channel_count * level_count * np.dtype(np.uint64).itemsize


def _normalized_channel_axis_or_none(
    channel_axis: int | None,
    ndim: int,
) -> int | None:
    if channel_axis is None:
        return None
    if ndim <= 0:
        return None
    axis = int(channel_axis)
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        # This intentionally matches the existing thumbnail helper, which
        # treats unresolved/invalid presentation metadata as no channel limits.
        return None
    return axis


def _axis_view(arr: np.ndarray, axis: int, index: int) -> np.ndarray:
    selection = [slice(None)] * arr.ndim
    selection[axis] = index
    return arr[tuple(selection)]


def _iter_chunks(arr: np.ndarray, chunk_elements: int):
    iterator = np.nditer(
        arr,
        flags=["buffered", "external_loop", "refs_ok", "zerosize_ok"],
        op_flags=[["readonly"]],
        order="K",
        buffersize=chunk_elements,
    )
    for chunk in iterator:
        yield np.asarray(chunk)


def _contrast_mode_key(contrast_mode: str) -> str:
    text = str(contrast_mode or "").strip().lower()
    if text in {"min-max", "minmax", "minimum-maximum", "minimum maximum"}:
        return "minmax"
    if text == "raw":
        return "raw"
    return "percentile"


def _data_kind_key(data_kind: str) -> str:
    return str(data_kind or "").strip().lower()


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _report(
    progress: ProgressContext | None,
    current: int,
    total: int,
    message: str,
) -> None:
    if progress is not None:
        progress.report(current, total, message)


def _check_cancelled(progress: ProgressContext | None) -> None:
    if progress is not None:
        progress.check_cancelled()


def _elapsed(clock: Callable[[], float], started: float) -> float:
    return max(0.0, float(clock()) - float(started))


__all__ = [
    "DEFAULT_COLD_GPU_THRESHOLD_BYTES",
    "DEFAULT_COLD_UINT8_GPU_THRESHOLD_BYTES",
    "DEFAULT_COLD_UINT16_GPU_THRESHOLD_BYTES",
    "DEFAULT_CPU_CHUNK_ELEMENTS",
    "DEFAULT_WARM_GPU_THRESHOLD_BYTES",
    "EXACT_NATIVE_MINMAX_ALGORITHM_ID",
    "EXACT_UINT_HISTOGRAM_ALGORITHM_ID",
    "NUMPY_MINMAX_ALGORITHM_ID",
    "NUMPY_PERCENTILE_ALGORITHM_ID",
    "SCAN_FREE_ALGORITHM_ID",
    "ScalarContrastLimits",
    "THUMBNAIL_PERCENTILE_RANGE",
    "ThumbnailContrastLimits",
    "ThumbnailStatisticsBackend",
    "ThumbnailStatisticsCleanupError",
    "ThumbnailStatisticsDecision",
    "ThumbnailStatisticsEngine",
    "ThumbnailStatisticsGPUError",
    "ThumbnailStatisticsMemoryError",
    "ThumbnailStatisticsRequest",
    "ThumbnailStatisticsResult",
    "exact_uint_thumbnail_contrast_limits",
]
