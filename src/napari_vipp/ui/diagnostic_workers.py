"""Qt worker adapters for inspector and presentation diagnostics.

The workers in this module know how to schedule calculations and report typed
results, but they do not know about :class:`VippWidget`.  Calculations that
still depend on widget-composed view state are supplied through narrow
callables at construction time.
"""

from __future__ import annotations

import threading
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from inspect import Parameter, signature

import numpy as np
from qtpy.QtCore import QObject, QRunnable, Signal

from napari_vipp.core.compute import ComputeMode
from napari_vipp.core.host_memory import (
    DEFAULT_HOST_MEMORY_SAFETY_RESERVE_BYTES,
    HostMemoryPreflightReason,
    capture_host_memory,
    preflight_host_allocation,
)
from napari_vipp.core.progress import OperationCancelled, ProgressContext
from napari_vipp.core.thumbnail_statistics import (
    ThumbnailStatisticsCleanupError,
    ThumbnailStatisticsRequest,
)
from napari_vipp.ui.plots import (
    COLOCALIZATION_SCATTER_BINS,
    COLOCALIZATION_SCATTER_DISPLAY_MAX_BINS,
    cap_colocalization_scatter_density_for_display,
    colocalization_scatter_peak_bytes,
)


@dataclass(frozen=True)
class ThumbnailContrastLimitRequest:
    key: tuple
    node_id: str
    data: object
    channel_axis: int | None
    contrast_mode: str
    data_kind: str


@dataclass(frozen=True)
class ThumbnailContrastLimitResult:
    run_id: int
    keys: frozenset[tuple]
    limits: dict[tuple, object]
    error: str = ""
    statistics: dict[tuple, object] = field(default_factory=dict)
    errors: dict[tuple, str] = field(default_factory=dict)
    cancelled: bool = False
    cleanup_failed: bool = False


@dataclass(frozen=True)
class ThumbnailContrastProgress:
    """Chunk-aware progress for one presentation-only statistics batch."""

    run_id: int
    node_id: str
    node_index: int
    node_total: int
    current: int
    total: int
    overall_current: int
    overall_total: int
    backend: str = ""
    message: str = ""
    indeterminate: bool = False


@dataclass(frozen=True, slots=True)
class ThumbnailContrastStarted:
    """Worker-thread start time sampled only for opted-in diagnostics."""

    run_id: int
    started_monotonic_seconds: float


@dataclass(frozen=True, slots=True)
class ThumbnailContrastTerminal:
    """Worker-thread terminal time before queued UI result delivery."""

    run_id: int
    terminal_monotonic_seconds: float


@dataclass(frozen=True)
class InputHistogramDistribution:
    counts: object = None
    x_range: tuple[float, float] | None = None
    colors: object = None
    total_values: int = 0
    finite_values: int = 0
    display_bins: int = 0
    identity_ref: object = None


@dataclass(frozen=True)
class InputHistogramRequest:
    run_id: int
    key: tuple
    node_id: str
    operation_id: str
    data: object
    state: object
    scope: str
    current_step: tuple | None
    current_step_nsteps: tuple | None
    params: dict
    title: str
    cancel_event: threading.Event | None = None
    distribution_key: tuple = ()
    distribution: InputHistogramDistribution | None = None


@dataclass(frozen=True)
class InputHistogramResult:
    run_id: int
    key: tuple
    node_id: str
    counts: object = None
    x_range: tuple[float, float] | None = None
    colors: object = None
    markers: object = None
    title: str = "Input Histogram"
    error: str = ""
    marker_error: str = ""
    total_values: int = 0
    finite_values: int = 0
    display_bins: int = 0
    distribution_key: tuple = ()
    distribution: InputHistogramDistribution | None = None


@dataclass(frozen=True)
class LabelVolumeRequest:
    """Exact label/component-size calculation requested by the inspector."""

    run_id: int
    key: tuple
    node_id: str
    data: object
    spatial_ndim: int
    cancel_event: threading.Event | None = None
    connectivity: str = "Face connected"


@dataclass(frozen=True)
class LabelVolumeResult:
    """Terminal result from one exact label-volume calculation."""

    run_id: int
    key: tuple
    node_id: str
    volumes: object = None
    error: str = ""
    cancelled: bool = False


@dataclass(frozen=True)
class ColocalizationScatterDensity:
    """Threshold-independent density data shared by inspector results."""

    density_key: tuple
    density_counts: object
    channel_1_min: float
    channel_1_max: float
    channel_2_min: float
    channel_2_max: float
    range_percentile: float = 100.0
    full_channel_1_min: float | None = None
    full_channel_1_max: float | None = None
    full_channel_2_min: float | None = None
    full_channel_2_max: float | None = None
    presentation_density_counts: object = None

    @property
    def nbytes(self) -> int:
        """Return the retained ndarray footprint used by the cache budget."""
        arrays = [np.asarray(self.density_counts)]
        if self.presentation_density_counts is not None:
            arrays.append(np.asarray(self.presentation_density_counts))
        unique_arrays: list[np.ndarray] = []
        retained_bytes = 0
        for array in arrays:
            if any(np.shares_memory(array, existing) for existing in unique_arrays):
                continue
            unique_arrays.append(array)
            retained_bytes += int(array.nbytes)
        return retained_bytes


@dataclass(frozen=True)
class ColocalizationScatterRequest:
    run_id: int
    key: tuple
    node_id: str
    inputs: tuple[object, ...]
    threshold_mode: str
    threshold_1: float
    threshold_2: float
    intensity_max: float = 255.0
    bins: int = COLOCALIZATION_SCATTER_BINS
    range_percentile: float = 100.0
    cancel_event: threading.Event | None = None
    density_key: tuple = ()
    reusable_density: ColocalizationScatterDensity | None = None
    thresholds_resolved: bool = False


@dataclass(frozen=True)
class ColocalizationScatterResult:
    run_id: int
    key: tuple
    node_id: str
    threshold_mode: str
    threshold_1: float
    threshold_2: float
    intensity_min: float = 0.0
    intensity_max: float = 255.0
    density_counts: object = None
    roi_voxels: int = 0
    colocalized_voxels: int = 0
    warnings: tuple[str, ...] = ()
    error: str = ""
    density_key: tuple = ()
    channel_1_min: float | None = None
    channel_1_max: float | None = None
    channel_2_min: float | None = None
    channel_2_max: float | None = None
    range_percentile: float = 100.0
    density_reused: bool = False
    full_channel_1_min: float | None = None
    full_channel_1_max: float | None = None
    full_channel_2_min: float | None = None
    full_channel_2_max: float | None = None
    presentation_density_counts: object = None


@dataclass(frozen=True)
class AutoContrastRequest:
    run_id: int
    key: tuple
    node_id: str
    data: object
    saturation_percent: float


@dataclass(frozen=True)
class AutoContrastResult:
    run_id: int
    key: tuple
    node_id: str
    saturation_percent: float
    scale_offset: tuple[float, float, float, float] | None = None
    error: str = ""


@dataclass(frozen=True)
class GeneratedLayerContrastRequest:
    key: tuple
    layer_name: str
    data: object
    identity: object


@dataclass(frozen=True)
class GeneratedLayerContrastResult:
    key: tuple
    layer_name: str
    limits: tuple[float, float] | None = None
    error: str = ""
    identity: object = None


@dataclass(frozen=True)
class GeneratedLayerContrastPlan:
    key: tuple
    limits: tuple[float, float]
    pending: bool
    exact: bool


class _ThumbnailContrastLimitSignals(QObject):
    started = Signal(object)
    terminal = Signal(object)
    progress = Signal(object)
    finished = Signal(object)


class ThumbnailContrastLimitWorker(QRunnable):
    """Compute stack thumbnail contrast limits off the GUI thread."""

    def __init__(
        self,
        run_id: int,
        requests: tuple[ThumbnailContrastLimitRequest, ...],
        *,
        calculate_scalar: Callable[..., object] | None = None,
        calculate_channel: Callable[..., object] | None = None,
        statistics_engine: object | None = None,
        compute_mode: ComputeMode = ComputeMode.AUTO,
        device_id: str = "",
        cancel_event: threading.Event | None = None,
        started_clock: Callable[[], float] | None = None,
    ):
        super().__init__()
        self.run_id = int(run_id)
        self.requests = tuple(requests)
        if statistics_engine is None and (
            calculate_scalar is None or calculate_channel is None
        ):
            raise TypeError(
                "Thumbnail worker requires a statistics engine or both legacy "
                "calculation callables."
            )
        self._calculate_scalar = calculate_scalar
        self._calculate_channel = calculate_channel
        self._statistics_engine = statistics_engine
        self._compute_mode = ComputeMode.parse(compute_mode)
        self._device_id = str(device_id).strip()
        self._cancel_event = cancel_event or threading.Event()
        self._started_clock = started_clock
        self.signals = _ThumbnailContrastLimitSignals()

    def run(self) -> None:
        if self._started_clock is not None:
            try:
                started = self._started_clock()
                if (
                    not isinstance(started, bool)
                    and isinstance(started, (int, float))
                    and np.isfinite(float(started))
                ):
                    self.signals.started.emit(
                        ThumbnailContrastStarted(self.run_id, float(started))
                    )
            except Exception:
                pass
        keys = frozenset(request.key for request in self.requests)
        limits: dict[tuple, object] = {}
        statistics: dict[tuple, object] = {}
        errors: dict[tuple, str] = {}
        request_values: list[int] = []
        for item in self.requests:
            decision = None
            if self._statistics_engine is not None:
                try:
                    decision = self._statistics_engine.select(
                        ThumbnailStatisticsRequest(
                            item.data,
                            contrast_mode=item.contrast_mode,
                            data_kind=item.data_kind,
                            channel_axis=item.channel_axis,
                            compute_mode=self._compute_mode,
                            device_id=self._device_id,
                        )
                    )
                except Exception:
                    decision = None
            scanned_values = (
                int(getattr(decision, "scanned_values", 0) or 0)
                if decision is not None and hasattr(decision, "scanned_values")
                else _thumbnail_request_values(item)
            )
            request_values.append(
                max(
                    scanned_values
                    if decision is not None
                    else _thumbnail_request_values(item),
                    1,
                )
            )
        request_values_tuple = tuple(request_values)
        overall_total = sum(request_values_tuple)
        node_total = len(self.requests)
        overall_completed = 0
        cancelled = False
        cleanup_failed = False

        if not self._emit_progress(
            ThumbnailContrastProgress(
                self.run_id,
                "",
                0,
                node_total,
                0,
                0,
                0,
                overall_total,
                "",
                "Preparing thumbnail statistics",
            )
        ):
            return

        for index, (request, value_total) in enumerate(
            zip(self.requests, request_values_tuple, strict=True),
            start=1,
        ):
            if self._cancel_event.is_set():
                cancelled = True
                break

            statistics_request = None
            selected_backend = "CPU"
            selection_message = "Selecting CPU statistics backend"
            try:
                if self._statistics_engine is not None:
                    statistics_request = ThumbnailStatisticsRequest(
                        request.data,
                        contrast_mode=request.contrast_mode,
                        data_kind=request.data_kind,
                        channel_axis=request.channel_axis,
                        compute_mode=self._compute_mode,
                        device_id=self._device_id,
                    )
                    decision = self._statistics_engine.select(statistics_request)
                    backend_value = str(
                        getattr(decision.backend, "value", decision.backend)
                    )
                    selected_backend = "GPU" if "gpu" in backend_value else "CPU"
                    selection_message = (
                        f"{selected_backend} selected: {decision.reason}"
                    )
            except Exception as exc:
                errors[request.key] = f"{type(exc).__name__}: {exc}"
                overall_completed += value_total
                self._emit_progress(
                    ThumbnailContrastProgress(
                        self.run_id,
                        request.node_id,
                        index,
                        node_total,
                        1,
                        1,
                        overall_completed,
                        overall_total,
                        "CPU/GPU",
                        "Thumbnail statistics selection failed",
                    )
                )
                continue

            progress_state: dict[str, object] = {
                "backend": selected_backend,
                "scaled_current": 0,
            }

            def report(
                update,
                *,
                base=overall_completed,
                item=request,
                item_index=index,
                item_values=value_total,
                item_state=progress_state,
            ):
                operation_current = int(update.current)
                operation_total = int(update.total)
                fraction = (
                    min(max(float(operation_current) / operation_total, 0.0), 1.0)
                    if operation_total > 0
                    else 0.0
                )
                scaled_current = int(round(item_values * fraction))
                scaled_current = max(
                    int(item_state["scaled_current"]),
                    min(scaled_current, item_values),
                )
                item_state["scaled_current"] = scaled_current
                message = str(update.message)
                indeterminate = (
                    "cancel applies after this pass" in message.casefold()
                    or "probe may not be interruptible" in message.casefold()
                )
                if message.casefold().startswith("cpu fallback"):
                    item_state["backend"] = "CPU fallback"
                self._emit_progress(
                    ThumbnailContrastProgress(
                        self.run_id,
                        item.node_id,
                        item_index,
                        node_total,
                        operation_current,
                        operation_total,
                        base + scaled_current,
                        overall_total,
                        str(item_state["backend"]),
                        message,
                        indeterminate,
                    )
                )

            progress = ProgressContext(
                cancelled=self._cancel_event.is_set,
                reporter=report,
            )
            try:
                progress.report(0, value_total, selection_message)
                if self._statistics_engine is not None:
                    assert statistics_request is not None
                    result = self._statistics_engine.calculate(
                        statistics_request,
                        progress=progress,
                    )
                    limits[request.key] = result.limits
                    statistics[request.key] = result
                    backend_value = str(
                        getattr(result.actual_backend, "value", result.actual_backend)
                    ).casefold()
                    progress_state["backend"] = (
                        "CPU fallback"
                        if bool(getattr(result, "used_fallback", False))
                        else "GPU"
                        if "gpu" in backend_value
                        else "CPU"
                    )
                elif request.channel_axis is None:
                    assert self._calculate_scalar is not None
                    limits[request.key] = self._calculate_scalar(
                        request.data,
                        contrast_mode=request.contrast_mode,
                        data_kind=request.data_kind,
                    )
                else:
                    assert self._calculate_channel is not None
                    limits[request.key] = self._calculate_channel(
                        request.data,
                        channel_axis=request.channel_axis,
                        contrast_mode=request.contrast_mode,
                        data_kind=request.data_kind,
                    )
                progress.report(value_total, value_total, "Thumbnail statistics ready")
            except OperationCancelled:
                cancelled = True
                break
            except ThumbnailStatisticsCleanupError as exc:
                errors[request.key] = str(exc)
                cleanup_failed = True
                self._emit_progress(
                    ThumbnailContrastProgress(
                        self.run_id,
                        request.node_id,
                        index,
                        node_total,
                        1,
                        1,
                        overall_completed + value_total,
                        overall_total,
                        str(progress_state["backend"]),
                        "Thumbnail statistics cleanup failed",
                    )
                )
                break
            except Exception as exc:
                # One failed presentation refinement must not discard exact
                # limits already completed for other node cards.
                errors[request.key] = f"{type(exc).__name__}: {exc}"
                self._emit_progress(
                    ThumbnailContrastProgress(
                        self.run_id,
                        request.node_id,
                        index,
                        node_total,
                        1,
                        1,
                        overall_completed + value_total,
                        overall_total,
                        str(progress_state["backend"]),
                        "Thumbnail statistics failed; provisional preview retained",
                    )
                )
            finally:
                overall_completed += value_total

        error = "; ".join(errors.values())
        result = ThumbnailContrastLimitResult(
            self.run_id,
            keys,
            limits,
            error=error,
            statistics=statistics,
            errors=errors,
            cancelled=cancelled,
            cleanup_failed=cleanup_failed,
        )
        if self._started_clock is not None:
            try:
                terminal = self._started_clock()
                if (
                    not isinstance(terminal, bool)
                    and isinstance(terminal, (int, float))
                    and np.isfinite(float(terminal))
                ):
                    _emit_if_alive(
                        self.signals,
                        "terminal",
                        ThumbnailContrastTerminal(self.run_id, float(terminal)),
                    )
            except Exception:
                pass
        _emit_if_alive(
            self.signals,
            "finished",
            result,
        )

    def _emit_progress(self, progress: ThumbnailContrastProgress) -> bool:
        return _emit_if_alive(self.signals, "progress", progress)


def _thumbnail_request_values(request: ThumbnailContrastLimitRequest) -> int:
    try:
        return max(int(np.asarray(request.data).size), 0)
    except Exception:
        return 0


def _emit_if_alive(signals, name: str, payload: object) -> bool:
    """Emit unless the receiving widget has already destroyed its QObject."""
    try:
        getattr(signals, name).emit(payload)
    except RuntimeError:
        return False
    return True


class _InputHistogramSignals(QObject):
    finished = Signal(object)


class InputHistogramWorker(QRunnable):
    """Build a large input histogram without blocking Qt's event loop."""

    def __init__(
        self,
        request: InputHistogramRequest,
        *,
        histogram_summary: Callable[..., tuple],
        histogram_source: Callable[..., object],
        histogram_markers: Callable[..., object],
    ):
        super().__init__()
        self.request = request
        self._histogram_summary = histogram_summary
        self._histogram_source = histogram_source
        self._histogram_markers = histogram_markers
        self.signals = _InputHistogramSignals()

    def run(self) -> None:
        request = self.request
        distribution = request.distribution
        if distribution is None:
            try:
                counts, x_range, colors = self._histogram_summary(
                    request.data,
                    state=request.state,
                    scope=request.scope,
                    current_step=request.current_step,
                    current_step_nsteps=request.current_step_nsteps,
                )
                source = self._histogram_source(
                    request.data,
                    state=request.state,
                    scope=request.scope,
                    current_step=request.current_step,
                    current_step_nsteps=request.current_step_nsteps,
                )
                try:
                    identity_ref = weakref.ref(request.data)
                except TypeError:
                    identity_ref = None
                distribution = InputHistogramDistribution(
                    counts=counts,
                    x_range=x_range,
                    colors=colors,
                    total_values=int(source[0].size) if source is not None else 0,
                    finite_values=(
                        int(np.asarray(counts).sum()) if counts is not None else 0
                    ),
                    display_bins=(
                        int(np.asarray(counts).shape[-1]) if counts is not None else 0
                    ),
                    identity_ref=identity_ref,
                )
            except Exception as exc:
                self.signals.finished.emit(
                    InputHistogramResult(
                        request.run_id,
                        request.key,
                        request.node_id,
                        title=request.title,
                        error=str(exc),
                        distribution_key=request.distribution_key,
                    )
                )
                return
        marker_error = ""
        try:
            markers = self._histogram_markers(
                request.operation_id,
                request.data,
                state=request.state,
                scope=request.scope,
                current_step=request.current_step,
                current_step_nsteps=request.current_step_nsteps,
                params=request.params,
                progress=(
                    ProgressContext(cancelled=request.cancel_event.is_set)
                    if request.cancel_event is not None
                    else None
                ),
            )
        except OperationCancelled as exc:
            self.signals.finished.emit(
                InputHistogramResult(
                    request.run_id,
                    request.key,
                    request.node_id,
                    counts=distribution.counts,
                    x_range=distribution.x_range,
                    colors=distribution.colors,
                    title=request.title,
                    error=str(exc),
                    total_values=distribution.total_values,
                    finite_values=distribution.finite_values,
                    display_bins=distribution.display_bins,
                    distribution_key=request.distribution_key,
                    distribution=distribution,
                )
            )
            return
        except Exception as exc:
            markers = []
            marker_error = str(exc)
        self.signals.finished.emit(
            InputHistogramResult(
                request.run_id,
                request.key,
                request.node_id,
                counts=distribution.counts,
                x_range=distribution.x_range,
                colors=distribution.colors,
                markers=markers,
                title=request.title,
                marker_error=marker_error,
                total_values=distribution.total_values,
                finite_values=distribution.finite_values,
                display_bins=distribution.display_bins,
                distribution_key=request.distribution_key,
                distribution=distribution,
            )
        )


class _LabelVolumeSignals(QObject):
    finished = Signal(object)


class LabelVolumeWorker(QRunnable):
    """Calculate exact per-object label/component sizes off the GUI thread.

    The injected calculation owns the scientific definition.  Current
    :func:`napari_vipp.core.diagnostics.label_volumes` calls are cancellable
    at the worker boundary; a future or test implementation that accepts a
    ``progress`` keyword also receives a :class:`ProgressContext` so it can
    stop during its inner calculation.
    """

    def __init__(
        self,
        request: LabelVolumeRequest,
        *,
        label_volumes: Callable[..., object],
    ):
        super().__init__()
        self.request = request
        self._label_volumes = label_volumes
        self._accepts_progress = _callable_accepts_keyword(
            label_volumes,
            "progress",
        )
        self._accepts_connectivity = _callable_accepts_keyword(
            label_volumes,
            "connectivity",
        )
        self.signals = _LabelVolumeSignals()

    def run(self) -> None:
        request = self.request
        cancel_event = request.cancel_event
        if cancel_event is not None and cancel_event.is_set():
            self._emit_result(cancelled=True)
            return

        progress = (
            ProgressContext(cancelled=cancel_event.is_set)
            if cancel_event is not None
            else None
        )
        try:
            if progress is not None:
                progress.check_cancelled()
            kwargs = (
                {"progress": progress}
                if progress is not None and self._accepts_progress
                else {}
            )
            if self._accepts_connectivity:
                kwargs["connectivity"] = request.connectivity
            volumes = self._label_volumes(
                request.data,
                int(request.spatial_ndim),
                **kwargs,
            )
            if progress is not None:
                progress.check_cancelled()
        except OperationCancelled:
            self._emit_result(cancelled=True)
            return
        except Exception as exc:
            self._emit_result(error=str(exc))
            return
        self._emit_result(volumes=volumes)

    def _emit_result(
        self,
        *,
        volumes: object = None,
        error: str = "",
        cancelled: bool = False,
    ) -> None:
        request = self.request
        _emit_if_alive(
            self.signals,
            "finished",
            LabelVolumeResult(
                request.run_id,
                request.key,
                request.node_id,
                volumes=volumes,
                error=error,
                cancelled=cancelled,
            ),
        )


def _callable_accepts_keyword(function: Callable[..., object], name: str) -> bool:
    """Return whether ``function`` explicitly or generically accepts a keyword."""
    try:
        parameters = signature(function).parameters
    except (TypeError, ValueError):
        return False
    parameter = parameters.get(name)
    if parameter is not None and parameter.kind is not Parameter.POSITIONAL_ONLY:
        return True
    return any(item.kind is Parameter.VAR_KEYWORD for item in parameters.values())


class _ColocalizationScatterSignals(QObject):
    finished = Signal(object)


_UNAVAILABLE_HOST_MEMORY_REASONS = frozenset(
    {
        HostMemoryPreflightReason.SNAPSHOT_UNAVAILABLE,
        HostMemoryPreflightReason.COMMIT_HEADROOM_UNAVAILABLE,
    }
)


def _preflight_colocalization_scatter_density(
    bins: int,
    *,
    host_memory_provider: Callable[[], object],
) -> None:
    """Reject a measured unsafe high-resolution scatter allocation."""
    bins = int(bins)
    if bins <= COLOCALIZATION_SCATTER_DISPLAY_MAX_BINS:
        return
    try:
        snapshot = host_memory_provider()
    except Exception:
        # A diagnostic probe must never make the scientific operation unusable.
        return
    if snapshot is None:
        return
    required_bytes = colocalization_scatter_peak_bytes(bins)
    decision = preflight_host_allocation(
        snapshot,
        required_bytes=required_bytes,
        purpose=f"{bins:,}-bin colocalization scatter density",
    )
    if decision.allowed:
        return
    if decision.reason_code in _UNAVAILABLE_HOST_MEMORY_REASONS:
        observations = (
            (
                "physical-memory",
                snapshot.physical_available_bytes,
                decision.physical_reserve_bytes,
            ),
            (
                "commit",
                snapshot.commit_available_bytes,
                decision.commit_reserve_bytes,
            ),
        )
        for resource, available_bytes, reserve_bytes in observations:
            if available_bytes is None:
                continue
            reserve = (
                DEFAULT_HOST_MEMORY_SAFETY_RESERVE_BYTES
                if reserve_bytes is None
                else reserve_bytes
            )
            if available_bytes >= required_bytes + reserve:
                continue
            raise MemoryError(
                "Colocalization scatter memory preflight rejected the requested "
                f"{bins:,} × {bins:,} density: measured {resource} headroom "
                f"({available_bytes:,} bytes) cannot cover the estimated peak "
                f"({required_bytes:,} bytes) plus the safety reserve "
                f"({reserve:,} bytes)."
            )
        return
    raise MemoryError(
        "Colocalization scatter memory preflight rejected the requested "
        f"{bins:,} × {bins:,} density. {decision.reason}"
    )


class ColocalizationScatterWorker(QRunnable):
    """Prepare a large colocalization inspector without blocking Qt."""

    def __init__(
        self,
        request: ColocalizationScatterRequest,
        *,
        normalized_inputs: Callable[..., object],
        threshold_values: Callable[..., object],
        scatter_density: Callable[..., object],
        scatter_counts: Callable[..., object],
        host_memory_provider: Callable[[], object] = capture_host_memory,
    ):
        super().__init__()
        self.request = request
        self._normalized_inputs = normalized_inputs
        self._threshold_values = threshold_values
        self._scatter_density = scatter_density
        self._scatter_counts = scatter_counts
        self._host_memory_provider = host_memory_provider
        self.signals = _ColocalizationScatterSignals()

    def run(self) -> None:
        request = self.request
        threshold_1 = float(request.threshold_1)
        threshold_2 = float(request.threshold_2)
        display_min = 0.0
        display_max = float(request.intensity_max)
        channel_1_min = display_min
        channel_1_max = display_max
        channel_2_min = display_min
        channel_2_max = display_max
        progress = (
            ProgressContext(cancelled=request.cancel_event.is_set)
            if request.cancel_event is not None
            else None
        )
        try:
            if progress is not None:
                progress.check_cancelled()
            if request.reusable_density is None:
                _preflight_colocalization_scatter_density(
                    request.bins,
                    host_memory_provider=self._host_memory_provider,
                )
            if (
                str(request.threshold_mode).lower().startswith("costes")
                and not request.thresholds_resolved
            ):
                threshold_1, threshold_2 = self._threshold_values(
                    request.inputs,
                    threshold_mode=request.threshold_mode,
                    channel_1_threshold=threshold_1,
                    channel_2_threshold=threshold_2,
                    intensity_max=request.intensity_max,
                    progress=progress,
                )
            if progress is not None:
                progress.check_cancelled()
            ch1, ch2, roi_mask, warnings = self._normalized_inputs(
                request.inputs,
                intensity_max=request.intensity_max,
            )
            if progress is not None:
                progress.check_cancelled()
            reusable = request.reusable_density
            density_reused = reusable is not None
            if reusable is not None:
                density = np.asarray(reusable.density_counts)
                if reusable.density_key != request.density_key:
                    raise ValueError("Reusable scatter density key does not match.")
                if density.shape != (int(request.bins), int(request.bins)):
                    raise ValueError(
                        "Reusable scatter density shape does not match the request."
                    )
                if not np.isclose(
                    reusable.range_percentile,
                    request.range_percentile,
                ):
                    raise ValueError(
                        "Reusable scatter density range does not match the request."
                    )
                roi_voxels, colocalized_voxels = self._scatter_counts(
                    ch1,
                    ch2,
                    threshold_1=threshold_1,
                    threshold_2=threshold_2,
                    roi_mask=roi_mask,
                    progress=progress,
                )
                density_counts = reusable.density_counts
                presentation_density_counts = reusable.presentation_density_counts
                if presentation_density_counts is None:
                    presentation_density_counts = (
                        cap_colocalization_scatter_density_for_display(density)
                    )
                channel_1_min = float(reusable.channel_1_min)
                channel_1_max = float(reusable.channel_1_max)
                channel_2_min = float(reusable.channel_2_min)
                channel_2_max = float(reusable.channel_2_max)
                full_channel_1_min = float(
                    reusable.channel_1_min
                    if reusable.full_channel_1_min is None
                    else reusable.full_channel_1_min
                )
                full_channel_1_max = float(
                    reusable.channel_1_max
                    if reusable.full_channel_1_max is None
                    else reusable.full_channel_1_max
                )
                full_channel_2_min = float(
                    reusable.channel_2_min
                    if reusable.full_channel_2_min is None
                    else reusable.full_channel_2_min
                )
                full_channel_2_max = float(
                    reusable.channel_2_max
                    if reusable.full_channel_2_max is None
                    else reusable.full_channel_2_max
                )
            else:
                (
                    density_counts,
                    roi_voxels,
                    colocalized_voxels,
                    channel_1_min,
                    channel_1_max,
                    channel_2_min,
                    channel_2_max,
                    full_channel_1_min,
                    full_channel_1_max,
                    full_channel_2_min,
                    full_channel_2_max,
                ) = self._scatter_density(
                    ch1,
                    ch2,
                    threshold_1=threshold_1,
                    threshold_2=threshold_2,
                    roi_mask=roi_mask,
                    intensity_max=request.intensity_max,
                    bins=request.bins,
                    range_percentile=request.range_percentile,
                    progress=progress,
                    include_full_ranges=True,
                )
                presentation_density_counts = (
                    cap_colocalization_scatter_density_for_display(
                        np.asarray(density_counts)
                    )
                )
            display_min = min(channel_1_min, channel_2_min)
            display_max = max(channel_1_max, channel_2_max)
        except Exception as exc:
            self.signals.finished.emit(
                ColocalizationScatterResult(
                    request.run_id,
                    request.key,
                    request.node_id,
                    request.threshold_mode,
                    threshold_1,
                    threshold_2,
                    intensity_max=request.intensity_max,
                    error=str(exc),
                    density_key=request.density_key,
                )
            )
            return
        self.signals.finished.emit(
            ColocalizationScatterResult(
                request.run_id,
                request.key,
                request.node_id,
                request.threshold_mode,
                threshold_1,
                threshold_2,
                intensity_min=display_min,
                intensity_max=display_max,
                density_counts=density_counts,
                roi_voxels=roi_voxels,
                colocalized_voxels=colocalized_voxels,
                warnings=tuple(warnings),
                density_key=request.density_key,
                channel_1_min=channel_1_min,
                channel_1_max=channel_1_max,
                channel_2_min=channel_2_min,
                channel_2_max=channel_2_max,
                range_percentile=request.range_percentile,
                density_reused=density_reused,
                full_channel_1_min=full_channel_1_min,
                full_channel_1_max=full_channel_1_max,
                full_channel_2_min=full_channel_2_min,
                full_channel_2_max=full_channel_2_max,
                presentation_density_counts=presentation_density_counts,
            )
        )


class _AutoContrastSignals(QObject):
    finished = Signal(object)


class AutoContrastWorker(QRunnable):
    """Calculate exact automatic scale/offset parameters off the GUI thread."""

    def __init__(
        self,
        request: AutoContrastRequest,
        *,
        calculate: Callable[[object, float], object],
    ):
        super().__init__()
        self.request = request
        self._calculate = calculate
        self.signals = _AutoContrastSignals()

    def run(self) -> None:
        request = self.request
        try:
            scale_offset = self._calculate(
                request.data,
                request.saturation_percent,
            )
        except Exception as exc:
            self.signals.finished.emit(
                AutoContrastResult(
                    request.run_id,
                    request.key,
                    request.node_id,
                    request.saturation_percent,
                    error=str(exc),
                )
            )
            return
        self.signals.finished.emit(
            AutoContrastResult(
                request.run_id,
                request.key,
                request.node_id,
                request.saturation_percent,
                scale_offset=scale_offset,
            )
        )


class _GeneratedLayerContrastSignals(QObject):
    finished = Signal(object)


class GeneratedLayerContrastWorker(QRunnable):
    """Calculate exact generated-layer display limits off the GUI thread."""

    def __init__(
        self,
        request: GeneratedLayerContrastRequest,
        *,
        calculate: Callable[[object], object],
    ):
        super().__init__()
        self.request = request
        self._calculate = calculate
        self.signals = _GeneratedLayerContrastSignals()

    def run(self) -> None:
        request = self.request
        try:
            limits = self._calculate(request.data)
        except Exception as exc:
            self.signals.finished.emit(
                GeneratedLayerContrastResult(
                    request.key,
                    request.layer_name,
                    error=str(exc),
                    identity=request.identity,
                )
            )
            return
        self.signals.finished.emit(
            GeneratedLayerContrastResult(
                request.key,
                request.layer_name,
                limits=limits,
                identity=request.identity,
            )
        )


__all__ = [
    "AutoContrastRequest",
    "AutoContrastResult",
    "AutoContrastWorker",
    "ColocalizationScatterDensity",
    "ColocalizationScatterRequest",
    "ColocalizationScatterResult",
    "ColocalizationScatterWorker",
    "GeneratedLayerContrastPlan",
    "GeneratedLayerContrastRequest",
    "GeneratedLayerContrastResult",
    "GeneratedLayerContrastWorker",
    "InputHistogramDistribution",
    "InputHistogramRequest",
    "InputHistogramResult",
    "InputHistogramWorker",
    "LabelVolumeRequest",
    "LabelVolumeResult",
    "LabelVolumeWorker",
    "ThumbnailContrastLimitRequest",
    "ThumbnailContrastLimitResult",
    "ThumbnailContrastLimitWorker",
    "ThumbnailContrastStarted",
    "ThumbnailContrastTerminal",
]
