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
from dataclasses import dataclass

import numpy as np
from qtpy.QtCore import QObject, QRunnable, Signal

from napari_vipp.core.progress import OperationCancelled, ProgressContext
from napari_vipp.ui.plots import (
    COLOCALIZATION_SCATTER_BINS,
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
    cancel_event: threading.Event | None = None


@dataclass(frozen=True)
class ColocalizationScatterResult:
    run_id: int
    key: tuple
    node_id: str
    threshold_mode: str
    threshold_1: float
    threshold_2: float
    intensity_max: float = 255.0
    density_counts: object = None
    roi_voxels: int = 0
    colocalized_voxels: int = 0
    warnings: tuple[str, ...] = ()
    error: str = ""


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
    progress = Signal(object)
    finished = Signal(object)


class ThumbnailContrastLimitWorker(QRunnable):
    """Compute stack thumbnail contrast limits off the GUI thread."""

    def __init__(
        self,
        run_id: int,
        requests: tuple[ThumbnailContrastLimitRequest, ...],
        *,
        calculate_scalar: Callable[..., object],
        calculate_channel: Callable[..., object],
    ):
        super().__init__()
        self.run_id = int(run_id)
        self.requests = tuple(requests)
        self._calculate_scalar = calculate_scalar
        self._calculate_channel = calculate_channel
        self.signals = _ThumbnailContrastLimitSignals()

    def run(self) -> None:
        keys = frozenset(request.key for request in self.requests)
        limits: dict[tuple, object] = {}
        total = len(self.requests)
        try:
            if not _emit_if_alive(
                self.signals,
                "progress",
                (self.run_id, 0, total),
            ):
                return
            for index, request in enumerate(self.requests, start=1):
                if request.channel_axis is None:
                    limits[request.key] = self._calculate_scalar(
                        request.data,
                        contrast_mode=request.contrast_mode,
                        data_kind=request.data_kind,
                    )
                else:
                    limits[request.key] = self._calculate_channel(
                        request.data,
                        channel_axis=request.channel_axis,
                        contrast_mode=request.contrast_mode,
                        data_kind=request.data_kind,
                    )
                if not _emit_if_alive(
                    self.signals,
                    "progress",
                    (self.run_id, index, total),
                ):
                    return
        except Exception as exc:
            _emit_if_alive(
                self.signals,
                "finished",
                ThumbnailContrastLimitResult(
                    self.run_id,
                    keys,
                    {},
                    error=str(exc),
                )
            )
            return
        _emit_if_alive(
            self.signals,
            "finished",
            ThumbnailContrastLimitResult(self.run_id, keys, limits),
        )


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
                        int(np.asarray(counts).shape[-1])
                        if counts is not None
                        else 0
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


class _ColocalizationScatterSignals(QObject):
    finished = Signal(object)


class ColocalizationScatterWorker(QRunnable):
    """Prepare a large colocalization inspector without blocking Qt."""

    def __init__(
        self,
        request: ColocalizationScatterRequest,
        *,
        normalized_inputs: Callable[..., object],
        threshold_values: Callable[..., object],
        scatter_density: Callable[..., object],
    ):
        super().__init__()
        self.request = request
        self._normalized_inputs = normalized_inputs
        self._threshold_values = threshold_values
        self._scatter_density = scatter_density
        self.signals = _ColocalizationScatterSignals()

    def run(self) -> None:
        request = self.request
        threshold_1 = float(request.threshold_1)
        threshold_2 = float(request.threshold_2)
        progress = (
            ProgressContext(cancelled=request.cancel_event.is_set)
            if request.cancel_event is not None
            else None
        )
        try:
            if progress is not None:
                progress.check_cancelled()
            if str(request.threshold_mode).lower().startswith("costes"):
                threshold_1, threshold_2 = self._threshold_values(
                    request.inputs,
                    threshold_mode=request.threshold_mode,
                    channel_1_threshold=threshold_1,
                    channel_2_threshold=threshold_2,
                    intensity_max=request.intensity_max,
                )
            if progress is not None:
                progress.check_cancelled()
            ch1, ch2, roi_mask, warnings = self._normalized_inputs(
                request.inputs,
                intensity_max=request.intensity_max,
            )
            if progress is not None:
                progress.check_cancelled()
            (
                density_counts,
                roi_voxels,
                colocalized_voxels,
            ) = self._scatter_density(
                ch1,
                ch2,
                threshold_1=threshold_1,
                threshold_2=threshold_2,
                roi_mask=roi_mask,
                intensity_max=request.intensity_max,
                bins=request.bins,
                progress=progress,
            )
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
                intensity_max=request.intensity_max,
                density_counts=density_counts,
                roi_voxels=roi_voxels,
                colocalized_voxels=colocalized_voxels,
                warnings=tuple(warnings),
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
    "ThumbnailContrastLimitRequest",
    "ThumbnailContrastLimitResult",
    "ThumbnailContrastLimitWorker",
]
