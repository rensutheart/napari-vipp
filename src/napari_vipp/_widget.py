"""napari dock widget for the VIPP workflow prototype."""

from __future__ import annotations

import ctypes
import inspect as py_inspect
import os
import re
import textwrap
import threading
import weakref
from collections.abc import Iterable
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, replace
from html import escape
from numbers import Rational
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtCore import (
    QEvent,
    QLocale,
    QMimeData,
    QObject,
    QPointF,
    QRect,
    QRunnable,
    QSignalBlocker,
    QSize,
    Qt,
    QThreadPool,
    QTimer,
    Signal,
)
from qtpy.QtGui import (
    QAction,
    QBrush,
    QColor,
    QFont,
    QIcon,
    QImage,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QSyntaxHighlighter,
    QTextCharFormat,
)
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)
from scipy import ndimage as ndi

from napari_vipp import __version__ as VIPP_VERSION
from napari_vipp._graph import OPERATION_MIME, PipelineGraphView
from napari_vipp._sample_data import make_sample_data
from napari_vipp._theme import category_color, category_tint
from napari_vipp.core.batch import (
    BATCH_CONFIG_FILENAME,
    BATCH_MANIFEST_FILENAME,
    BATCH_SCRIPT_FILENAME,
    BATCH_WORKFLOW_FILENAME,
    BatchConfig,
    BatchOutputConfig,
    BatchRunResult,
    BatchSourceConfig,
    ExistingFilePolicy,
    atomic_write_json,
    atomic_write_text,
    load_batch_config,
    plan_batch,
    preflight_batch,
    run_batch,
    safe_batch_filename,
    save_batch_config,
    scientific_workflow_hash,
    validate_batch_config,
)
from napari_vipp.core.batch_demo import (
    SYNTHETIC_BATCH_GROUND_TRUTH_FILENAME,
    SyntheticBatchDemo,
    create_synthetic_batch_demo,
    next_synthetic_batch_demo_root,
    validate_synthetic_batch_demo,
)
from napari_vipp.core.channel_colors import (
    CHANNEL_COLOR_CHOICES,
    CHANNEL_COLOR_HEX,
    FLUORESCENCE_COLORS,
    channel_color_labels_from_metadata,
    color_value_to_rgb,
)
from napari_vipp.core.export import (
    export_batch_runner_to_python,
    export_pipeline_to_python,
)
from napari_vipp.core.graph_layout import (
    LayoutEdge,
    LayoutNode,
    layout_layered_dag,
)
from napari_vipp.core.graph_search import (
    GraphSearchMatch,
    find_graph_matches,
)
from napari_vipp.core.io import (
    MICROSCOPE_FILE_FILTER,
    MICROSCOPE_SUFFIXES,
    AnalysisLabel,
    OptionalMicroscopeReaderError,
    SourceInspection,
    detect_deconvolution_metadata,
    inspect_image_source,
    read_image,
    write_ome_zarr_analysis_dataset,
)
from napari_vipp.core.metadata import (
    ImageState,
    MetadataRow,
    format_compact_metadata,
    image_state_from_array,
    metadata_history_items,
    metadata_table_rows,
)
from napari_vipp.core.operations import (
    BORN_WOLF_PSF_AUTO_PARAMETERS,
    BORN_WOLF_PSF_MANUAL_DEFAULTS,
    NO_TABLE_COLUMNS_VALUE,
    automatic_threshold_value,
    colocalization_normalized_inputs,
    colocalization_threshold_values,
    exact_integer_percentiles,
    resolve_born_wolf_psf_parameters,
    save_array_output,
)
from napari_vipp.core.pipeline import (
    DEFAULT_DYNAMIC_OUTPUT_PORTS,
    EXECUTION_ERROR,
    EXECUTION_NOT_CALCULATED,
    EXECUTION_READY,
    EXECUTION_RUNNING,
    EXECUTION_STALE,
    GLOBAL_THRESHOLD_OPERATIONS,
    MANUAL_RUN_SKIP,
    InputSpec,
    OperationSpec,
    ParameterSpec,
    PrototypePipeline,
    SourcePayload,
    grouped_palette_specs,
)
from napari_vipp.core.preview import (
    MONOCHROME_COLORMAPS,
    THUMBNAIL_CONTRAST_MODES,
    THUMBNAIL_CONTRAST_SCOPES,
    _apply_monochrome_colormap,
    make_preview,
    normalize_thumbnail_with_colormap,
    thumbnail_channel_contrast_limits,
    thumbnail_contrast_limits,
)
from napari_vipp.core.progress import OperationCancelled, ProgressContext
from napari_vipp.core.tables import is_table_data, save_table_output
from napari_vipp.core.workflow import (
    deserialize_workflow,
    load_workflow,
    save_workflow,
    serialize_workflow,
)

_RGB_VOLUME_CHANNELS = (
    (0, "Red", "red"),
    (1, "Green", "green"),
    (2, "Blue", "blue"),
)

ASYNC_SOURCE_FILE_BYTES = 32 * 1024 * 1024
AUTO_BACKGROUND_MIN_BYTES = 32 * 1024 * 1024
AUTO_BACKGROUND_MIN_ELEMENTS = 4_000_000
AUTO_CONTRAST_BACKGROUND_MIN_ELEMENTS = 1_000_000
INSPECTOR_STATISTICS_CHUNK_ELEMENTS = 1_048_576
INSPECTOR_DISPLAY_HISTOGRAM_BINS = 128
COLOCALIZATION_SCATTER_BINS = 255


@dataclass(frozen=True)
class ExampleWorkflowSpec:
    id: str
    category: str
    title: str
    filename: str
    samples: tuple[str, ...]
    description: str
    generated_batch_demo: bool = False


EXAMPLE_WORKFLOWS: tuple[ExampleWorkflowSpec, ...] = (
    ExampleWorkflowSpec(
        "batch-provenance",
        "Batch & Reproducibility",
        "Deterministic Batch & Provenance",
        "synthetic-batch-provenance.json",
        ("Ready-to-run paired .npy demo",),
        "Open a ready-to-run three-item batch demo with paired sources, "
        "collision-aware preview, explicit image/label/table outputs, saved "
        "config and runner files, manifests, archives, per-item provenance, "
        "and automatic ground-truth validation.",
        generated_batch_demo=True,
    ),
    ExampleWorkflowSpec(
        "label-cleanup",
        "Segmentation & Labels",
        "Red-Channel Label Cleanup",
        "otsu-red-channel-labels.json",
        ("VIPP synthetic multichannel volume",),
        "Split the red/TRITC-like channel, blur, threshold, fill holes, "
        "label objects, clear borders, and filter labels by volume.",
    ),
    ExampleWorkflowSpec(
        "object-intensity",
        "Measurements & Tables",
        "Object Intensity Measurements",
        "red-channel-object-intensity-measurements.json",
        ("VIPP synthetic multichannel volume",),
        "Measure cleaned object labels together with matching red-channel "
        "intensity values.",
    ),
    ExampleWorkflowSpec(
        "merged-measurements",
        "Measurements & Tables",
        "Merged Measurement Table",
        "red-channel-merged-measurement-table.json",
        ("VIPP synthetic multichannel volume",),
        "Build a PCA-oriented table from object morphology, object intensity, "
        "and metadata columns.",
    ),
    ExampleWorkflowSpec(
        "summary-table",
        "Measurements & Tables",
        "Grouped Measurement Summary",
        "synthetic-measurement-summary.json",
        ("VIPP synthetic measurement summary",),
        "Summarize known object counts and areas across timepoints.",
    ),
    ExampleWorkflowSpec(
        "derived-morphology",
        "Measurements & Tables",
        "Derived 2D Object Morphology",
        "synthetic-derived-object-morphology.json",
        ("VIPP synthetic object morphology",),
        "Calculate 2D morphology, circularity, perimeter-area ratio, and Hu "
        "moments.",
    ),
    ExampleWorkflowSpec(
        "mesh-morphology",
        "Measurements & Tables",
        "3D Mesh Morphology",
        "synthetic-3d-mesh-morphology.json",
        ("VIPP synthetic 3D mesh morphology",),
        "Measure anisotropic 3D objects with surface area, mesh volume, convex "
        "hull metrics, and sphericity.",
    ),
    ExampleWorkflowSpec(
        "skeleton-qc",
        "Skeletons & Networks",
        "Skeleton QC",
        "synthetic-skeleton-qc.json",
        ("VIPP synthetic skeleton network",),
        "Inspect skeleton keypoints, components, branches, pruning, graph "
        "tables, and network summaries.",
    ),
    ExampleWorkflowSpec(
        "advanced-skeleton",
        "Skeletons & Networks",
        "Advanced Skeleton Network",
        "synthetic-advanced-skeleton-network.json",
        ("VIPP synthetic advanced skeleton network",),
        "Review time-indexed 3D skeleton/network analysis with loops, fragments, "
        "spurs, and anisotropic calibration.",
    ),
    ExampleWorkflowSpec(
        "racc-colocalization",
        "Colocalization & Association",
        "RACC Colocalization",
        "synthetic-colocalization-racc.json",
        ("VIPP synthetic colocalization",),
        "Inspect red/green channel tunnels, ROI masks, scatter thresholds, "
        "Manders/Pearson metrics, and RACC output.",
    ),
    ExampleWorkflowSpec(
        "object-colocalization",
        "Colocalization & Association",
        "Object Colocalization Association",
        "synthetic-object-colocalization-association.json",
        ("VIPP synthetic colocalization",),
        "Combine thresholded channel labels, object colocalization rows, label "
        "overlap, nearest distances, and event localization.",
    ),
    ExampleWorkflowSpec(
        "deconvolution-2d",
        "Restoration & PSF",
        "2D Richardson-Lucy / TV Deconvolution",
        "synthetic-deconvolution-rl-tv.json",
        ("VIPP synthetic deconvolution image", "VIPP synthetic measured PSF"),
        "Prepare a measured PSF and compare ordinary Richardson-Lucy with "
        "Richardson-Lucy TV restoration.",
    ),
    ExampleWorkflowSpec(
        "deconvolution-3d",
        "Restoration & PSF",
        "3D Richardson-Lucy / TV Deconvolution",
        "synthetic-3d-deconvolution-rl-tv.json",
        (
            "VIPP synthetic 3D deconvolution volume",
            "VIPP synthetic 3D measured PSF",
        ),
        "Run volumetric PSF-aware restoration with matched 3D image and PSF "
        "sources.",
    ),
)


def _example_workflow_by_id(example_id: str) -> ExampleWorkflowSpec | None:
    for spec in EXAMPLE_WORKFLOWS:
        if spec.id == example_id:
            return spec
    return None


def _example_workflow_dir() -> Path:
    candidates = (
        Path(__file__).resolve().parent / "examples",
        Path(__file__).resolve().parents[2] / "examples",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[-1]


def _example_workflow_path(spec: ExampleWorkflowSpec) -> Path:
    return _example_workflow_dir() / spec.filename

if TYPE_CHECKING:
    import napari


@dataclass(frozen=True)
class ParameterBounds:
    minimum: float | int
    maximum: float | int
    step: float | int
    decimals: int
    expandable: bool = False
    logarithmic: bool = False
    entry_minimum: float | int | None = None
    entry_maximum: float | int | None = None


@dataclass(frozen=True)
class AxisSliceOption:
    index: int
    name: str
    axis_type: str
    size: int

    @property
    def title(self) -> str:
        return self.name.upper() if len(self.name) == 1 else self.name


@dataclass(frozen=True)
class ViewDimAxis:
    """One semantic non-XY axis exposed in the VIPP-local view controls."""

    name: str
    label: str
    step_axis: int
    size: int
    value: int


@dataclass(frozen=True)
class TunnelSummary:
    name: str
    source_id: str
    source_title: str
    source_port: int
    output_type: str
    subscribers: tuple[tuple[str, str, int], ...]

    @property
    def subscriber_count(self) -> int:
        return len(self.subscribers)


@dataclass(frozen=True)
class GraphNoteState:
    id: str
    text: str
    position: tuple[float, float]
    width: float = 240.0
    attached_node: str = ""

    def to_workflow_dict(self) -> dict:
        result = {
            "id": self.id,
            "text": self.text,
            "position": [float(self.position[0]), float(self.position[1])],
            "width": float(self.width),
        }
        if self.attached_node:
            result["attached_node"] = self.attached_node
        return result


@dataclass(frozen=True)
class BatchSourceBinding:
    node_id: str
    title: str
    input_dir: Path | None
    pattern: str


@dataclass(frozen=True)
class BatchPreviewRow:
    batch_index: int
    batch_id: str
    sources: dict[str, Path]
    outputs: list[Path]
    output_statuses: tuple[str, ...] = ()
    explicit_outputs: bool = True


@dataclass(frozen=True)
class BatchPreviewResult:
    rows: tuple[BatchPreviewRow, ...]
    total_items: int
    collision_count: int
    explicit_outputs: bool

    def __iter__(self):
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


@dataclass(frozen=True)
class WorkflowHistorySnapshot:
    workflow: dict
    selected_node_id: str
    preview_disabled_node_ids: tuple[str, ...] = ()
    active_pinned_node_id: str | None = None


@dataclass(frozen=True)
class PipelineRunRequest:
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


@dataclass(frozen=True)
class PipelineRunResult:
    run_id: int
    workflow: dict
    pipeline: PrototypePipeline | None = None
    error: str = ""
    cancelled: bool = False


@dataclass(frozen=True)
class SourceFileLoadSpec:
    node_id: str
    path: str
    series_index: int
    cache_key: tuple[object, ...]


@dataclass(frozen=True)
class SourceFileLoadResult:
    run_id: int
    payloads: dict[tuple[object, ...], SourcePayload]
    error: str = ""
    node_id: str = ""


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
class ExactFiniteStats:
    count: int
    minimum: int | float
    maximum: int | float


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


class SourceFileLoadSignals(QObject):
    finished = Signal(object)


class SourceFileLoadWorker(QRunnable):
    """Read selected file-path Image Source payloads off the GUI thread."""

    def __init__(self, run_id: int, specs: tuple[SourceFileLoadSpec, ...]):
        super().__init__()
        self.run_id = int(run_id)
        self.specs = tuple(specs)
        self.signals = SourceFileLoadSignals()

    def run(self) -> None:
        payloads: dict[tuple[object, ...], SourcePayload] = {}
        current_node_id = ""
        try:
            for spec in self.specs:
                current_node_id = spec.node_id
                dataset = read_image(spec.path, series_index=spec.series_index)
                payloads[spec.cache_key] = SourcePayload(
                    dataset.data,
                    {"vipp_source_path": str(Path(spec.path).expanduser())},
                    dataset.selected_series.name,
                    dataset.image_state,
                )
        except Exception as exc:
            self.signals.finished.emit(
                SourceFileLoadResult(
                    self.run_id,
                    {},
                    error=str(exc),
                    node_id=current_node_id,
                )
            )
            return
        self.signals.finished.emit(SourceFileLoadResult(self.run_id, payloads))


class ThumbnailContrastLimitSignals(QObject):
    progress = Signal(object)
    finished = Signal(object)


class ThumbnailContrastLimitWorker(QRunnable):
    """Compute stack thumbnail contrast limits off the GUI thread."""

    def __init__(
        self,
        run_id: int,
        requests: tuple[ThumbnailContrastLimitRequest, ...],
    ):
        super().__init__()
        self.run_id = int(run_id)
        self.requests = tuple(requests)
        self.signals = ThumbnailContrastLimitSignals()

    def run(self) -> None:
        keys = frozenset(request.key for request in self.requests)
        limits: dict[tuple, object] = {}
        total = len(self.requests)
        try:
            self.signals.progress.emit((self.run_id, 0, total))
            for index, request in enumerate(self.requests, start=1):
                if request.channel_axis is None:
                    limits[request.key] = thumbnail_contrast_limits(
                        request.data,
                        contrast_mode=request.contrast_mode,
                        data_kind=request.data_kind,
                    )
                else:
                    limits[request.key] = thumbnail_channel_contrast_limits(
                        request.data,
                        channel_axis=request.channel_axis,
                        contrast_mode=request.contrast_mode,
                        data_kind=request.data_kind,
                    )
                self.signals.progress.emit((self.run_id, index, total))
        except Exception as exc:
            self.signals.finished.emit(
                ThumbnailContrastLimitResult(
                    self.run_id,
                    keys,
                    {},
                    error=str(exc),
                )
            )
            return
        self.signals.finished.emit(
            ThumbnailContrastLimitResult(self.run_id, keys, limits)
        )


class InputHistogramSignals(QObject):
    finished = Signal(object)


class InputHistogramWorker(QRunnable):
    """Build a large input histogram without blocking Qt's event loop."""

    def __init__(self, request: InputHistogramRequest):
        super().__init__()
        self.request = request
        self.signals = InputHistogramSignals()

    def run(self) -> None:
        request = self.request
        distribution = request.distribution
        if distribution is None:
            try:
                counts, x_range, colors = _histogram_summary(
                    request.data,
                    state=request.state,
                    scope=request.scope,
                    current_step=request.current_step,
                    current_step_nsteps=request.current_step_nsteps,
                )
                source = _histogram_source(
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
                    total_values=(
                        int(source[0].size) if source is not None else 0
                    ),
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
            markers = _input_histogram_markers(
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


class ColocalizationScatterSignals(QObject):
    finished = Signal(object)


class ColocalizationScatterWorker(QRunnable):
    """Prepare a large colocalization inspector without blocking Qt."""

    def __init__(self, request: ColocalizationScatterRequest):
        super().__init__()
        self.request = request
        self.signals = ColocalizationScatterSignals()

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
                threshold_1, threshold_2 = colocalization_threshold_values(
                    request.inputs,
                    threshold_mode=request.threshold_mode,
                    channel_1_threshold=threshold_1,
                    channel_2_threshold=threshold_2,
                    intensity_max=request.intensity_max,
                )
            if progress is not None:
                progress.check_cancelled()
            ch1, ch2, roi_mask, warnings = colocalization_normalized_inputs(
                request.inputs,
                intensity_max=request.intensity_max,
            )
            if progress is not None:
                progress.check_cancelled()
            (
                density_counts,
                roi_voxels,
                colocalized_voxels,
            ) = _prepare_colocalization_scatter_density(
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


class AutoContrastSignals(QObject):
    finished = Signal(object)


class AutoContrastWorker(QRunnable):
    """Calculate exact automatic scale/offset parameters off the GUI thread."""

    def __init__(self, request: AutoContrastRequest):
        super().__init__()
        self.request = request
        self.signals = AutoContrastSignals()

    def run(self) -> None:
        request = self.request
        try:
            scale_offset = _auto_contrast_scale_offset(
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


class GeneratedLayerContrastSignals(QObject):
    finished = Signal(object)


class GeneratedLayerContrastWorker(QRunnable):
    """Calculate exact generated-layer display limits off the GUI thread."""

    def __init__(self, request: GeneratedLayerContrastRequest):
        super().__init__()
        self.request = request
        self.signals = GeneratedLayerContrastSignals()

    def run(self) -> None:
        request = self.request
        try:
            limits = _exact_generated_layer_contrast_limits(request.data)
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


class PipelineRunSignals(QObject):
    node_started = Signal(object)
    progress = Signal(object)
    finished = Signal(object)


class PipelineRunWorker(QRunnable):
    """Run the headless pipeline on a serialized graph snapshot."""

    def __init__(self, request: PipelineRunRequest):
        super().__init__()
        self.request = request
        self.signals = PipelineRunSignals()

    def run(self) -> None:
        try:
            workflow = deserialize_workflow(deepcopy(self.request.workflow))
            pipeline = PrototypePipeline()
            pipeline.restore_graph(
                workflow["nodes"],
                workflow["connections"],
                workflow.get("output_tunnels", ()),
            )
            self._hydrate_cached_pipeline_outputs(pipeline)
            pipeline.run(
                self.request.input_data,
                input_metadata=self.request.input_metadata,
                input_name=self.request.input_name,
                source_payloads=self.request.source_payloads,
                dirty_node_ids=self.request.dirty_node_ids,
                node_started_callback=self._emit_node_started,
                progress_callback=self._emit_progress,
                cancel_callback=self._is_cancelled,
                manual_mode=MANUAL_RUN_SKIP,
                manual_node_ids=self.request.manual_node_ids,
                retain_node_ids=self.request.retain_node_ids,
                prune_unretained=self.request.prune_unretained,
            )
        except OperationCancelled as exc:
            self.signals.finished.emit(
                PipelineRunResult(
                    self.request.run_id,
                    self.request.workflow,
                    error=str(exc),
                    cancelled=True,
                )
            )
            return
        except Exception as exc:
            self.signals.finished.emit(
                PipelineRunResult(
                    self.request.run_id,
                    self.request.workflow,
                    error=str(exc),
                )
            )
            return
        self.signals.finished.emit(
            PipelineRunResult(self.request.run_id, self.request.workflow, pipeline)
        )

    def _emit_node_started(self, node_id: str) -> None:
        self.signals.node_started.emit((self.request.run_id, node_id))

    def _emit_progress(
        self,
        node_id: str,
        current: int,
        total: int,
        message: str,
    ) -> None:
        self.signals.progress.emit(
            (self.request.run_id, node_id, int(current), int(total), str(message))
        )

    def _is_cancelled(self) -> bool:
        return bool(self.request.cancel_event and self.request.cancel_event.is_set())

    def _hydrate_cached_pipeline_outputs(self, pipeline: PrototypePipeline) -> None:
        if self.request.dirty_node_ids is None:
            return
        if self.request.cached_outputs is not None:
            pipeline.outputs = dict(self.request.cached_outputs)
        if self.request.cached_output_states is not None:
            pipeline.output_states = dict(self.request.cached_output_states)
        if self.request.cached_node_outputs is not None:
            pipeline.node_outputs = {
                node_id: list(outputs)
                for node_id, outputs in self.request.cached_node_outputs.items()
            }
        if self.request.cached_node_output_states is not None:
            pipeline.node_output_states = {
                node_id: list(states)
                for node_id, states in self.request.cached_node_output_states.items()
            }
        if self.request.cached_execution_states is not None:
            pipeline.node_execution_states = dict(self.request.cached_execution_states)
        if self.request.cached_execution_messages is not None:
            pipeline.node_execution_messages = dict(
                self.request.cached_execution_messages
            )
        pipeline.completed_node_ids = set(self.request.completed_node_ids)


RESCALE_VALUE_PARAMETERS = {"in_low_value", "in_high_value"}
RESCALE_PERCENTILE_PARAMETERS = {"in_low_percentile", "in_high_percentile"}
RESCALE_CUTOFF_PARAMETERS = RESCALE_VALUE_PARAMETERS | RESCALE_PERCENTILE_PARAMETERS
RESCALE_CUTOFF_MODE_PARAMETER = "cutoff_mode"
CLIP_CUTOFF_PARAMETERS = {"minimum", "maximum"}
INPUT_HISTOGRAM_OPERATIONS = {
    "binary_threshold",
    "clip_intensity",
    "hysteresis_threshold",
    "rescale_intensity",
} | GLOBAL_THRESHOLD_OPERATIONS
COLOCALIZATION_THRESHOLD_OPERATIONS = {
    "colocalization_metrics",
    "masked_colocalization_metrics",
    "colocalized_voxels",
    "masked_colocalized_voxels",
    "racc_index",
    "masked_racc_index",
}
COLOCALIZATION_SCATTER_OPERATIONS = COLOCALIZATION_THRESHOLD_OPERATIONS
COLOCALIZATION_SCATTER_COLORMAPS = (
    "Viridis",
    "Magma",
    "Inferno",
    "Plasma",
    "Cividis",
    "Gray",
)
BACKGROUND_PIPELINE_OPERATIONS = {
    "auto_watershed_from_mask",
    "born_wolf_psf",
    "euclidean_distance_transform",
    "gaussian_blur_3d",
    "h_maxima_markers",
    "marker_controlled_watershed",
    "minimum_threshold",
    "non_local_means_filter",
    "orthogonal_projection",
    "project_image",
    "rescale_axes",
    "rolling_ball_background",
    "subtract_background",
}
ROLLING_BALL_RADIUS_SLIDER_MAX = 100.0
MAX_CHANNEL_COLOR_CONTROLS = 12
CACHE_MODE_KEEP_ALL = "Keep all node outputs cached"
CACHE_MODE_SMART = "Smart interactive cache"
CACHE_MODE_LOW_MEMORY = "Low-memory mode"
CACHE_MODE_CHOICES = (
    CACHE_MODE_KEEP_ALL,
    CACHE_MODE_SMART,
    CACHE_MODE_LOW_MEMORY,
)
CACHE_MODE_STATUS_LABELS = {
    CACHE_MODE_KEEP_ALL: "Keep all",
    CACHE_MODE_SMART: "Smart interactive",
    CACHE_MODE_LOW_MEMORY: "Low memory",
}
CACHE_KEEP_NODE_PARAM = "_vipp_keep_cached"
DEFAULT_CACHE_MEMORY_LIMIT_PERCENT = 90
MEMORY_GUARD_MIN_FREE_BYTES = 512 * 1024 * 1024
EXPLICIT_OUTPUT_OPERATIONS = {"batch_output", "save_output"}
SMART_CACHE_RECENT_LIMIT = 6


def _axis_heading_text(option: AxisSliceOption, *, mode: str = "keep") -> str:
    accent = "#fbbf24" if mode == "remove" else "#93c5fd"
    return (
        f"<span style='font-weight: 700; color: {accent};'>"
        f"{escape(option.title)}</span>"
        f"<span style='color: #d1d5db;'>&nbsp;({escape(option.axis_type)})</span>"
        f"<span style='color: #94a3b8;'>&nbsp;-&nbsp;size {int(option.size)}</span>"
    )


def _toolbar_icon(kind: str) -> QIcon:
    icon = QIcon()
    icon.addPixmap(
        _toolbar_icon_pixmap(kind, "#d1d5db"),
        QIcon.Normal,
        QIcon.Off,
    )
    icon.addPixmap(
        _toolbar_icon_pixmap(kind, "#64748b"),
        QIcon.Disabled,
        QIcon.Off,
    )
    return icon


def _toolbar_icon_pixmap(kind: str, foreground: str) -> QPixmap:
    size = 24
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)

    pen = QPen(QColor(foreground), 2.2)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    path = QPainterPath()
    arrow = QPainterPath()
    if kind == "redo":
        path.moveTo(16, 8)
        path.cubicTo(13, 5.5, 5, 6.5, 5, 13)
        path.cubicTo(5, 18, 9.5, 20, 16, 20)
        arrow.moveTo(20.5, 8)
        arrow.lineTo(15, 4)
        arrow.lineTo(15, 12)
        arrow.closeSubpath()
    elif kind == "reset":
        path.moveTo(18.5, 8)
        path.cubicTo(15.8, 4.5, 9.8, 3.8, 6.5, 8)
        path.cubicTo(3.2, 12.2, 5.6, 19.5, 12.5, 19.5)
        path.cubicTo(16, 19.5, 18.8, 17.4, 19.8, 14.4)
        arrow.moveTo(20.4, 7.3)
        arrow.lineTo(15, 7.1)
        arrow.lineTo(18.4, 11.7)
        arrow.closeSubpath()
    else:
        path.moveTo(8, 8)
        path.cubicTo(11, 5.5, 19, 6.5, 19, 13)
        path.cubicTo(19, 18, 14.5, 20, 8, 20)
        arrow.moveTo(3.5, 8)
        arrow.lineTo(9, 4)
        arrow.lineTo(9, 12)
        arrow.closeSubpath()
    painter.drawPath(path)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(foreground))
    painter.drawPath(arrow)
    painter.end()
    return pixmap


def _toolbar_separator(width: int = 12) -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.VLine)
    line.setFrameShadow(QFrame.Sunken)
    line.setFixedWidth(int(width))
    return line


AUTO_CONTRAST_SATURATION_SPEC = ParameterSpec(
    "saturation_percent",
    "Saturation (%)",
    "float",
    0.35,
    0.0,
    20.0,
    0.05,
    2,
)


@dataclass(frozen=True)
class ConnectionInsertCandidate:
    operation_id: str
    title: str
    category: str
    subcategory: str
    mode: str
    detail: str
    search_text: str


@dataclass(frozen=True)
class ConnectionInsertPortMapping:
    input_port: int
    output_port: int
    input_label: str
    output_label: str
    detail: str
    params: tuple[tuple[str, object], ...] = ()


class ConnectionInsertDialog(QDialog):
    """Searchable picker for inserting a node on a specific connection."""

    def __init__(
        self,
        candidates: list[ConnectionInsertCandidate],
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Insert node on connection")
        self.resize(620, 460)
        self._candidates = candidates
        self._candidate_by_id = {
            candidate.operation_id: candidate for candidate in candidates
        }

        self.search = QLineEdit()
        self.search.setPlaceholderText("Search compatible nodes")
        self.search.setClearButtonEnabled(True)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(["Node", "Insertion", "Category"])
        self.tree.setRootIsDecorated(False)
        self.tree.setAlternatingRowColors(True)
        self.tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tree.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.tree.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.tree.itemDoubleClicked.connect(self._accept_item)
        self.tree.itemSelectionChanged.connect(self._sync_button_state)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self.ok_button = self.buttons.button(QDialogButtonBox.Ok)
        self.ok_button.setEnabled(False)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Choose a compatible node to insert on this wire."))
        layout.addWidget(self.search)
        layout.addWidget(self.tree, 1)
        layout.addWidget(self.buttons)

        self.search.textChanged.connect(self._populate)
        self._populate("")

    def selected_operation_id(self) -> str | None:
        item = self.tree.currentItem()
        if item is None:
            return None
        operation_id = item.data(0, Qt.UserRole)
        return str(operation_id) if operation_id else None

    def _populate(self, query: str) -> None:
        normalized = _normalize_search_text(query)
        self.tree.clear()
        first_item = None
        for candidate in self._candidates:
            if normalized and normalized not in candidate.search_text:
                continue
            item = QTreeWidgetItem(
                [
                    candidate.title,
                    self._mode_label(candidate.mode),
                    self._category_label(candidate),
                ]
            )
            item.setData(0, Qt.UserRole, candidate.operation_id)
            item.setToolTip(0, candidate.detail)
            item.setToolTip(1, candidate.detail)
            item.setForeground(0, QBrush(QColor(category_color(candidate.category))))
            item.setForeground(1, QBrush(QColor(self._mode_color(candidate.mode))))
            self.tree.addTopLevelItem(item)
            if first_item is None:
                first_item = item
        if first_item is not None:
            self.tree.setCurrentItem(first_item)
        self.tree.resizeColumnToContents(0)
        self.tree.resizeColumnToContents(1)
        self._sync_button_state()

    def _sync_button_state(self) -> None:
        self.ok_button.setEnabled(self.selected_operation_id() is not None)

    def _accept_item(self, item, _column) -> None:
        if item.data(0, Qt.UserRole):
            self.accept()

    @staticmethod
    def _mode_label(mode: str) -> str:
        return {
            "full": "Full splice",
            "choose": "Choose ports",
            "partial": "Partial insert",
            "place": "Place in gap",
        }.get(mode, mode)

    @staticmethod
    def _mode_color(mode: str) -> str:
        return {
            "full": "#22c55e",
            "choose": "#a78bfa",
            "partial": "#38bdf8",
            "place": "#f59e0b",
        }.get(mode, "#cbd5e1")

    @staticmethod
    def _category_label(candidate: ConnectionInsertCandidate) -> str:
        if candidate.subcategory:
            return f"{candidate.category} / {candidate.subcategory}"
        return candidate.category


class ConnectionInsertMappingDialog(QDialog):
    """Chooser for ambiguous insert-on-wire port mappings."""

    def __init__(
        self,
        mappings: list[ConnectionInsertPortMapping],
        node_title: str,
        source_title: str,
        target_title: str,
        axis_choices: list[tuple[str, str]] | None = None,
        mappings_by_axis: dict[str, list[ConnectionInsertPortMapping]] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Choose insert ports")
        self.resize(560, 360)
        self._mappings = list(mappings)
        self._mappings_by_axis = {
            str(axis): list(axis_mappings)
            for axis, axis_mappings in (mappings_by_axis or {}).items()
        }
        self.axis_combo: QComboBox | None = None

        self.tree = QTreeWidget()
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(["Upstream input", "Downstream output", "Mapping"])
        self.tree.setRootIsDecorated(False)
        self.tree.setAlternatingRowColors(True)
        self.tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tree.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.tree.itemDoubleClicked.connect(self._accept_item)
        self.tree.itemSelectionChanged.connect(self._sync_button_state)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self.ok_button = self.buttons.button(QDialogButtonBox.Ok)
        self.ok_button.setEnabled(False)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                f"Choose how '{node_title}' should connect "
                f"'{source_title}' to '{target_title}'."
            )
        )
        if axis_choices:
            axis_row = QHBoxLayout()
            axis_row.setContentsMargins(0, 0, 0, 0)
            axis_row.addWidget(QLabel("Axis"))
            self.axis_combo = QComboBox()
            for value, label in axis_choices:
                self.axis_combo.addItem(label, value)
            self.axis_combo.currentIndexChanged.connect(self._on_axis_changed)
            axis_row.addWidget(self.axis_combo, 1)
            layout.addLayout(axis_row)
        layout.addWidget(self.tree, 1)
        layout.addWidget(self.buttons)

        if self.axis_combo is not None:
            self._on_axis_changed(self.axis_combo.currentIndex())
        self._populate()

    def selected_mapping(self) -> ConnectionInsertPortMapping | None:
        item = self.tree.currentItem()
        if item is None:
            return None
        index = item.data(0, Qt.UserRole)
        try:
            return self._mappings[int(index)]
        except (TypeError, ValueError, IndexError):
            return None

    def _on_axis_changed(self, _index: int) -> None:
        if self.axis_combo is None:
            return
        axis = str(self.axis_combo.currentData())
        self._mappings = list(self._mappings_by_axis.get(axis, ()))
        self._populate()

    def _populate(self) -> None:
        self.tree.clear()
        first_item = None
        for index, mapping in enumerate(self._mappings):
            item = QTreeWidgetItem(
                [
                    mapping.input_label,
                    mapping.output_label,
                    mapping.detail,
                ]
            )
            item.setData(0, Qt.UserRole, index)
            item.setToolTip(0, mapping.detail)
            item.setToolTip(1, mapping.detail)
            item.setToolTip(2, mapping.detail)
            self.tree.addTopLevelItem(item)
            if first_item is None:
                first_item = item
        if first_item is not None:
            self.tree.setCurrentItem(first_item)
        for column in range(3):
            self.tree.resizeColumnToContents(column)
        self._sync_button_state()

    def _sync_button_state(self) -> None:
        self.ok_button.setEnabled(self.selected_mapping() is not None)

    def _accept_item(self, item, _column) -> None:
        if item.data(0, Qt.UserRole) is not None:
            self.accept()


class NodePalette(QTreeWidget):
    """Small categorized operation palette for adding nodes."""

    operation_requested = Signal(str)

    def __init__(
        self,
        groups: dict[str, dict[str, list[OperationSpec]]],
        parent=None,
    ):
        super().__init__(parent)
        self.setHeaderHidden(True)
        self.setRootIsDecorated(True)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragOnly)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.itemDoubleClicked.connect(self._on_item_double_clicked)
        self._category_items: list[QTreeWidgetItem] = []
        self._subcategory_items: list[QTreeWidgetItem] = []
        self._operation_items: list[QTreeWidgetItem] = []
        for category, subgroups in groups.items():
            category_item = QTreeWidgetItem([category])
            category_item.setFlags(category_item.flags() & ~Qt.ItemIsDragEnabled)
            self._style_category_item(category_item, category)
            self.addTopLevelItem(category_item)
            self._category_items.append(category_item)
            for subgroup, specs in subgroups.items():
                parent_item = category_item
                if subgroup:
                    subgroup_item = QTreeWidgetItem([subgroup])
                    subgroup_item.setFlags(
                        subgroup_item.flags() & ~Qt.ItemIsDragEnabled
                    )
                    subgroup_item.setData(
                        0,
                        Qt.UserRole + 1,
                        f"{category} {subgroup}",
                    )
                    self._style_subcategory_item(subgroup_item, category)
                    category_item.addChild(subgroup_item)
                    self._subcategory_items.append(subgroup_item)
                    parent_item = subgroup_item
                for spec in specs:
                    item = QTreeWidgetItem([spec.title])
                    item.setData(0, Qt.UserRole, spec.id)
                    item.setData(
                        0,
                        Qt.UserRole + 1,
                        f"{category} {subgroup} {spec.title} {spec.id}",
                    )
                    self._style_operation_item(item, category)
                    parent_item.addChild(item)
                    self._operation_items.append(item)
        self._no_results_item = QTreeWidgetItem(["No matching nodes"])
        self._no_results_item.setFlags(Qt.NoItemFlags)
        self._no_results_item.setHidden(True)
        self.addTopLevelItem(self._no_results_item)
        self._scroll_spacer = QTreeWidgetItem([""])
        self._scroll_spacer.setFlags(Qt.NoItemFlags)
        self._scroll_spacer.setSizeHint(0, QSize(1, 36))
        self.addTopLevelItem(self._scroll_spacer)
        self.expandAll()

    def _style_category_item(self, item: QTreeWidgetItem, category: str) -> None:
        font = item.font(0)
        font.setBold(True)
        item.setFont(0, font)
        item.setForeground(0, QBrush(QColor(category_color(category))))
        item.setBackground(0, QBrush(QColor(category_tint(category))))

    def _style_subcategory_item(self, item: QTreeWidgetItem, category: str) -> None:
        font = item.font(0)
        font.setItalic(True)
        item.setFont(0, font)
        item.setForeground(0, QBrush(QColor(category_color(category))))

    def _style_operation_item(self, item: QTreeWidgetItem, category: str) -> None:
        item.setForeground(0, QBrush(QColor(category_color(category))))

    def mimeData(self, items):  # noqa: N802
        mime = QMimeData()
        if not items:
            return mime
        operation_id = items[0].data(0, Qt.UserRole)
        if operation_id:
            mime.setData(OPERATION_MIME, str(operation_id).encode())
        return mime

    def _on_item_double_clicked(self, item, _column) -> None:
        operation_id = item.data(0, Qt.UserRole)
        if operation_id:
            self.operation_requested.emit(str(operation_id))

    def set_filter_text(self, text: str) -> None:
        query = _normalize_search_text(text)
        visible_count = 0
        for category_item in self._category_items:
            category_visible, category_count = self._apply_filter_to_children(
                category_item,
                query,
            )
            visible_count += category_count
            category_item.setHidden(not category_visible)
            if category_visible:
                category_item.setExpanded(True)
        self._no_results_item.setHidden(not query or visible_count > 0)

    def _apply_filter_to_children(
        self,
        parent: QTreeWidgetItem,
        query: str,
    ) -> tuple[bool, int]:
        parent_visible = False
        visible_count = 0
        for index in range(parent.childCount()):
            item = parent.child(index)
            if item.data(0, Qt.UserRole):
                haystack = _normalize_search_text(item.data(0, Qt.UserRole + 1))
                visible = not query or _fuzzy_match(query, haystack)
                item.setHidden(not visible)
                parent_visible = parent_visible or visible
                visible_count += int(visible)
            else:
                child_visible, child_count = self._apply_filter_to_children(
                    item,
                    query,
                )
                item.setHidden(not child_visible)
                if child_visible:
                    item.setExpanded(True)
                parent_visible = parent_visible or child_visible
                visible_count += child_count
        return parent_visible, visible_count


class FlexibleDoubleSpinBox(QDoubleSpinBox):
    """Locale-independent float entry accepting decimal points and commas."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setLocale(QLocale.c())
        self.setKeyboardTracking(False)

    @staticmethod
    def _normalized_text(text: str) -> str:
        return str(text).replace(",", ".")

    def validate(self, text: str, position: int):
        state, _normalized, validated_position = super().validate(
            self._normalized_text(text),
            position,
        )
        return state, text, validated_position

    def valueFromText(self, text: str) -> float:
        return super().valueFromText(self._normalized_text(text))

    def textFromValue(self, value: float) -> str:  # noqa: N802
        decimals = max(int(self.decimals()), 0)
        text = f"{float(value):.{decimals}f}" if decimals else f"{float(value):.0f}"
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return "0" if text == "-0" else text


def _configure_numeric_spin_box(box: QSpinBox | QDoubleSpinBox) -> None:
    box.setKeyboardTracking(False)
    editor = box.lineEdit()
    editor.setAlignment(Qt.AlignCenter)
    editor.setTextMargins(0, 0, 0, 0)


class ParameterControl(QWidget):
    """Slider with numeric entry for a single node parameter."""

    valueChanged = Signal(object)

    def __init__(self, spec, value, bounds: ParameterBounds, parent=None):
        super().__init__(parent)
        self.spec = spec
        self._is_integer = spec.kind == "int"
        self._bounds = bounds
        self._entry_minimum = bounds.minimum
        self._entry_maximum = bounds.maximum
        self._scale = self._scale_for(bounds)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimumWidth(120)
        if self._is_integer:
            self.value_box = QSpinBox()
        else:
            self.value_box = FlexibleDoubleSpinBox()
            self.value_box.setDecimals(bounds.decimals)
        _configure_numeric_spin_box(self.value_box)
        self.value_box.setMinimumWidth(74)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.value_box)

        self.set_bounds(bounds, value, emit=False)
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.value_box.valueChanged.connect(self._on_box_changed)

    def value(self):
        return self.value_box.value()

    def set_bounds(
        self,
        bounds: ParameterBounds,
        value=None,
        emit: bool = False,
    ) -> None:
        entry_minimum, entry_maximum = self._entry_bounds_for(bounds)
        current = self.value() if value is None else value
        current = self._clamped_value(current, entry_minimum, entry_maximum)
        bounds = self._expanded_bounds_for_value(bounds, current)
        bounds = _slider_safe_bounds(
            bounds.minimum,
            bounds.maximum,
            bounds.step,
            bounds.decimals,
            expandable=bounds.expandable,
            logarithmic=bounds.logarithmic,
            entry_minimum=bounds.entry_minimum,
            entry_maximum=bounds.entry_maximum,
        )
        self._bounds = bounds
        self._entry_minimum = entry_minimum
        self._entry_maximum = entry_maximum
        self._scale = self._scale_for(bounds)

        with QSignalBlocker(self.slider), QSignalBlocker(self.value_box):
            if self._is_integer:
                self.value_box.setRange(int(entry_minimum), int(entry_maximum))
                self.value_box.setSingleStep(max(int(bounds.step), 1))
                if bounds.logarithmic:
                    self.slider.setRange(0, 1000)
                    self.slider.setSingleStep(1)
                else:
                    self.slider.setRange(int(bounds.minimum), int(bounds.maximum))
                    self.slider.setSingleStep(max(int(bounds.step), 1))
                self.slider.setValue(self._to_slider(current))
                self.value_box.setValue(int(current))
            else:
                self.value_box.setDecimals(bounds.decimals)
                self.value_box.setRange(float(entry_minimum), float(entry_maximum))
                self.value_box.setSingleStep(float(bounds.step))
                if bounds.logarithmic:
                    self.slider.setRange(0, 1000)
                    self.slider.setSingleStep(1)
                else:
                    self.slider.setRange(
                        self._to_slider(bounds.minimum),
                        self._to_slider(bounds.maximum),
                    )
                    self.slider.setSingleStep(max(self._to_slider(bounds.step), 1))
                self.slider.setValue(self._to_slider(current))
                self.value_box.setValue(float(current))

        if emit:
            self.valueChanged.emit(self.value())

    def _on_slider_changed(self, value: int) -> None:
        mapped = self._from_slider(value)
        with QSignalBlocker(self.value_box):
            self.value_box.setValue(mapped)
        self.valueChanged.emit(self.value())

    def _on_box_changed(self, value) -> None:
        if self._bounds.expandable and not self._value_in_slider_bounds(value):
            self.set_bounds(self._bounds, value, emit=False)
            self.valueChanged.emit(self.value())
            return
        with QSignalBlocker(self.slider):
            self.slider.setValue(self._to_slider(value))
        self.valueChanged.emit(self.value())

    def _scale_for(self, bounds: ParameterBounds) -> int:
        if self._is_integer:
            return 1
        if bounds.decimals > 0:
            return 10**bounds.decimals
        step = float(bounds.step)
        return max(int(round(1 / step)), 1) if step > 0 else 100

    def _to_slider(self, value) -> int:
        if self._bounds.logarithmic:
            minimum = float(self._bounds.minimum)
            span = float(self._bounds.maximum) - minimum
            if span <= 0:
                return 0
            offset = float(np.clip(float(value) - minimum, 0.0, span))
            fraction = np.log1p(offset) / np.log1p(span)
            return int(round(fraction * 1000))
        if self._is_integer:
            return int(round(float(value)))
        return int(round(float(value) * self._scale))

    def _from_slider(self, value: int):
        if self._bounds.logarithmic:
            minimum = float(self._bounds.minimum)
            span = float(self._bounds.maximum) - minimum
            fraction = float(np.clip(value, 0, 1000)) / 1000.0
            mapped = minimum + np.expm1(fraction * np.log1p(max(span, 0.0)))
            return int(round(mapped)) if self._is_integer else mapped
        return int(value) if self._is_integer else value / self._scale

    def _clamped_value(self, value, minimum, maximum):
        if value is None:
            value = minimum
        return min(max(value, minimum), maximum)

    def _entry_bounds_for(
        self,
        bounds: ParameterBounds,
    ) -> tuple[float | int, float | int]:
        if bounds.entry_minimum is not None or bounds.entry_maximum is not None:
            minimum = (
                bounds.minimum if bounds.entry_minimum is None else bounds.entry_minimum
            )
            maximum = (
                bounds.maximum if bounds.entry_maximum is None else bounds.entry_maximum
            )
            return minimum, maximum
        if not bounds.expandable:
            return bounds.minimum, bounds.maximum
        minimum = bounds.minimum
        maximum = bounds.maximum
        if float(minimum) < 0:
            minimum = min(float(minimum), -1_000_000.0)
        maximum = max(float(maximum), 1_000_000.0)
        if self._is_integer:
            return int(round(minimum)), int(round(maximum))
        return float(minimum), float(maximum)

    def _expanded_bounds_for_value(
        self,
        bounds: ParameterBounds,
        value,
    ) -> ParameterBounds:
        if not bounds.expandable:
            return bounds
        minimum = float(bounds.minimum)
        maximum = float(bounds.maximum)
        value = float(value)
        span = max(
            maximum - minimum,
            abs(maximum),
            abs(minimum),
            float(bounds.step),
            1.0,
        )
        if value > maximum:
            maximum = value + max(span * 0.25, abs(value) * 0.25, float(bounds.step))
        if value < minimum and minimum < 0:
            minimum = value - max(span * 0.25, abs(value) * 0.25, float(bounds.step))
        if self._is_integer:
            return ParameterBounds(
                int(np.floor(minimum)),
                int(np.ceil(maximum)),
                bounds.step,
                bounds.decimals,
                bounds.expandable,
                bounds.logarithmic,
                bounds.entry_minimum,
                bounds.entry_maximum,
            )
        return ParameterBounds(
            minimum,
            maximum,
            bounds.step,
            bounds.decimals,
            bounds.expandable,
            bounds.logarithmic,
            bounds.entry_minimum,
            bounds.entry_maximum,
        )

    def _value_in_slider_bounds(self, value) -> bool:
        if self._bounds.logarithmic:
            return (
                float(self._bounds.minimum)
                <= float(value)
                <= float(self._bounds.maximum)
            )
        if self._is_integer:
            return self.slider.minimum() <= int(value) <= self.slider.maximum()
        slider_value = self._to_slider(value)
        return self.slider.minimum() <= slider_value <= self.slider.maximum()


class NumericEntryControl(QWidget):
    """Numeric entry without a slider for parameters where sliders are misleading."""

    valueChanged = Signal(object)

    def __init__(self, spec, value, bounds: ParameterBounds, parent=None):
        super().__init__(parent)
        self.spec = spec
        self._is_integer = spec.kind == "int"
        if self._is_integer:
            self.value_box = QSpinBox()
        else:
            self.value_box = FlexibleDoubleSpinBox()
            self.value_box.setDecimals(bounds.decimals)
        _configure_numeric_spin_box(self.value_box)
        self.value_box.setMinimumWidth(100)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.value_box, 1)

        self.set_bounds(bounds, value, emit=False)
        self.value_box.valueChanged.connect(self.valueChanged.emit)

    def value(self):
        return self.value_box.value()

    def set_bounds(
        self,
        bounds: ParameterBounds,
        value=None,
        emit: bool = False,
    ) -> None:
        minimum = (
            bounds.minimum if bounds.entry_minimum is None else bounds.entry_minimum
        )
        maximum = (
            bounds.maximum if bounds.entry_maximum is None else bounds.entry_maximum
        )
        current = minimum if value is None else value
        if self._is_integer:
            current = int(np.clip(int(current), int(minimum), int(maximum)))
            with QSignalBlocker(self.value_box):
                self.value_box.setRange(int(minimum), int(maximum))
                self.value_box.setSingleStep(max(int(bounds.step), 1))
                self.value_box.setValue(current)
        else:
            current = float(np.clip(float(current), float(minimum), float(maximum)))
            with QSignalBlocker(self.value_box):
                self.value_box.setDecimals(bounds.decimals)
                self.value_box.setRange(float(minimum), float(maximum))
                self.value_box.setSingleStep(float(bounds.step))
                self.value_box.setValue(current)
        if emit:
            self.valueChanged.emit(self.value())


class ChoiceControl(QWidget):
    """Dropdown control for categorical node parameters."""

    valueChanged = Signal(object)

    def __init__(self, spec, value, bounds: ParameterBounds, parent=None):
        super().__init__(parent)
        self.spec = spec
        self._bounds = bounds
        self.combo = QComboBox()
        self._set_combo_items(spec.choices, spec.choice_labels)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.combo, 1)

        self.set_bounds(bounds, value, emit=False)
        self.combo.currentIndexChanged.connect(self._emit_current_value)

    def value(self):
        value = self.combo.currentData()
        return self.combo.currentText() if value is None else value

    def _emit_current_value(self, _index: int) -> None:
        self.valueChanged.emit(self.value())

    def _set_combo_items(
        self,
        choices: tuple[str, ...],
        choice_labels: tuple[str, ...] = (),
    ) -> None:
        labels = self._choice_labels(choices, choice_labels)
        self.combo.clear()
        for value, label in zip(choices, labels, strict=True):
            self.combo.addItem(label, value)

    @staticmethod
    def _choice_labels(
        choices: tuple[str, ...],
        choice_labels: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        if len(choice_labels) == len(choices):
            return tuple(str(label) for label in choice_labels)
        return tuple(str(choice) for choice in choices)

    def set_choices(
        self,
        choices: tuple[str, ...],
        value=None,
        emit: bool = False,
        choice_labels: tuple[str, ...] = (),
    ) -> None:
        self.spec = replace(
            self.spec,
            choices=tuple(choices),
            choice_labels=tuple(choice_labels),
        )
        current = self.spec.default if value is None else str(value)
        with QSignalBlocker(self.combo):
            self._set_combo_items(self.spec.choices, self.spec.choice_labels)
            index = self.combo.findData(current)
            self.combo.setCurrentIndex(max(index, 0))
        if emit:
            self.valueChanged.emit(self.value())

    def set_bounds(
        self,
        bounds: ParameterBounds,
        value=None,
        emit: bool = False,
    ) -> None:
        self._bounds = bounds
        current = self.spec.default if value is None else value
        current = str(current)
        with QSignalBlocker(self.combo):
            index = self.combo.findData(current)
            if index < 0:
                index = 0
            self.combo.setCurrentIndex(index)
        if emit:
            self.valueChanged.emit(self.value())


class TextControl(QWidget):
    """Single-line text control for path-like and free text parameters."""

    valueChanged = Signal(object)

    def __init__(self, spec, value, _bounds: ParameterBounds, parent=None):
        super().__init__(parent)
        self.spec = spec
        self.edit = QLineEdit()
        self.edit.setText("" if value is None else str(value))

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.edit, 1)
        self.edit.textChanged.connect(self.valueChanged.emit)

    def value(self):
        return self.edit.text()

    def set_bounds(
        self,
        _bounds: ParameterBounds,
        value=None,
        emit: bool = False,
    ) -> None:
        current = "" if value is None else str(value)
        with QSignalBlocker(self.edit):
            self.edit.setText(current)
        if emit:
            self.valueChanged.emit(self.value())


class BoolControl(QWidget):
    """Checkbox control for boolean node parameters."""

    valueChanged = Signal(object)

    def __init__(self, spec, value, _bounds: ParameterBounds, parent=None):
        super().__init__(parent)
        self.spec = spec
        self.checkbox = QCheckBox()
        self.checkbox.setChecked(bool(spec.default if value is None else value))

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.checkbox)
        layout.addStretch(1)
        self.checkbox.toggled.connect(self.valueChanged.emit)

    def value(self):
        return self.checkbox.isChecked()

    def set_bounds(
        self,
        _bounds: ParameterBounds,
        value=None,
        emit: bool = False,
    ) -> None:
        current = self.spec.default if value is None else value
        with QSignalBlocker(self.checkbox):
            self.checkbox.setChecked(bool(current))
        if emit:
            self.valueChanged.emit(self.value())


class ImageSourceControl(QWidget):
    """Source selector for explicit graph input nodes."""

    valueChanged = Signal(object)

    def __init__(
        self,
        value: dict | None,
        *,
        layer_names: list[str],
        sample_names: list[str],
        series_options: list[tuple[int, str]] | None = None,
        source_summary: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["napari layer", "file path", "sample"])
        self.layer_combo = QComboBox()
        self.sample_combo = QComboBox()
        self.path_edit = QLineEdit()
        self.path_button = QPushButton("File...")
        self.path_button.setMaximumWidth(64)
        self.zarr_button = QPushButton("Zarr...")
        self.zarr_button.setMaximumWidth(64)
        self.series_combo = QComboBox()
        self.binding_combo = QComboBox()
        self.binding_combo.addItems(["single item", "collection"])
        self.source_summary = QLabel()
        self.source_summary.setWordWrap(True)
        self.source_summary.setStyleSheet("color: #94a3b8;")

        self.layer_row = QWidget()
        layer_layout = QHBoxLayout(self.layer_row)
        layer_layout.setContentsMargins(0, 0, 0, 0)
        layer_layout.addWidget(self.layer_combo, 1)

        self.file_row = QWidget()
        file_layout = QHBoxLayout(self.file_row)
        file_layout.setContentsMargins(0, 0, 0, 0)
        file_layout.addWidget(self.path_edit, 1)
        file_layout.addWidget(self.path_button)
        file_layout.addWidget(self.zarr_button)

        self.sample_row = QWidget()
        sample_layout = QHBoxLayout(self.sample_row)
        sample_layout.setContentsMargins(0, 0, 0, 0)
        sample_layout.addWidget(self.sample_combo, 1)

        self.series_row = QWidget()
        series_layout = QHBoxLayout(self.series_row)
        series_layout.setContentsMargins(0, 0, 0, 0)
        series_layout.addWidget(self.series_combo, 1)

        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addRow("Source", self.mode_combo)
        layout.addRow("Layer", self.layer_row)
        layout.addRow("File", self.file_row)
        layout.addRow("Series / image", self.series_row)
        layout.addRow("Binding", self.binding_combo)
        layout.addRow("Sample", self.sample_row)
        layout.addRow(self.source_summary)

        self.set_options(
            layer_names,
            sample_names,
            series_options=series_options,
            source_summary=source_summary,
            value=value,
            emit=False,
        )

        self.mode_combo.currentTextChanged.connect(self._on_changed)
        self.layer_combo.currentTextChanged.connect(self._on_changed)
        self.sample_combo.currentTextChanged.connect(self._on_changed)
        self.path_edit.textChanged.connect(self._on_changed)
        self.series_combo.currentIndexChanged.connect(self._on_changed)
        self.binding_combo.currentTextChanged.connect(self._on_changed)
        self.path_button.clicked.connect(self._browse_path)
        self.zarr_button.clicked.connect(self._browse_zarr_path)

    def value(self) -> dict[str, object]:
        return {
            "source_mode": self.mode_combo.currentText(),
            "layer_name": self.layer_combo.currentText(),
            "file_path": self.path_edit.text(),
            "sample_name": self.sample_combo.currentText(),
            "series_index": int(self.series_combo.currentData() or 0),
            "binding_mode": self.binding_combo.currentText(),
        }

    def set_options(
        self,
        layer_names: list[str],
        sample_names: list[str],
        *,
        series_options: list[tuple[int, str]] | None = None,
        source_summary: str = "",
        value: dict | None = None,
        emit: bool = False,
    ) -> None:
        current = value or self.value()
        self._set_combo_items(
            self.layer_combo,
            layer_names,
            str(current.get("layer_name", "")),
        )
        self._set_combo_items(
            self.sample_combo,
            sample_names,
            str(current.get("sample_name", "")),
        )
        mode = str(current.get("source_mode", "napari layer"))
        if self.mode_combo.findText(mode) < 0:
            mode = "napari layer"
        with QSignalBlocker(self.mode_combo), QSignalBlocker(self.path_edit):
            self.mode_combo.setCurrentText(mode)
            self.path_edit.setText(str(current.get("file_path", "")))
        self._set_series_items(
            series_options or [],
            int(current.get("series_index", 0) or 0),
        )
        binding = str(current.get("binding_mode", "single item"))
        if self.binding_combo.findText(binding) < 0:
            binding = "single item"
        with QSignalBlocker(self.binding_combo):
            self.binding_combo.setCurrentText(binding)
        self.source_summary.setText(source_summary)
        self._sync_rows()
        if emit:
            self.valueChanged.emit(self.value())

    def _set_combo_items(
        self,
        combo: QComboBox,
        values: list[str],
        current: str,
    ) -> None:
        with QSignalBlocker(combo):
            combo.clear()
            combo.addItem("")
            for value in values:
                combo.addItem(value)
            if current:
                index = combo.findText(current)
                if index < 0:
                    combo.addItem(current)
                    index = combo.findText(current)
                combo.setCurrentIndex(index)

    def _set_series_items(
        self,
        values: list[tuple[int, str]],
        current: int,
    ) -> None:
        with QSignalBlocker(self.series_combo):
            self.series_combo.clear()
            if not values:
                self.series_combo.addItem("Series 1", 0)
            else:
                for index, label in values:
                    self.series_combo.addItem(label, index)
            selected = self.series_combo.findData(current)
            self.series_combo.setCurrentIndex(max(selected, 0))

    def _browse_path(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self,
            "Select image source",
            self.path_edit.text(),
            "Images and arrays (*.ome.tif *.ome.tiff *.tif *.tiff *.png *.jpg "
            "*.jpeg *.jpe *.jfif *.bmp *.dib *.gif *.webp *.tga *.pbm *.pgm "
            "*.ppm *.pnm *.npy *.npz *.nd2 *.czi *.lsm *.lif *.lof *.xlif "
            "*.oir *.oib *.oif *.vsi);;"
            f"{MICROSCOPE_FILE_FILTER};;"
            "All files (*.*)",
        )
        if path:
            self.path_edit.setText(path)

    def _browse_zarr_path(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Select OME-Zarr source",
            self.path_edit.text(),
        )
        if path:
            self.path_edit.setText(path)

    def _on_changed(self, *_args) -> None:
        self._sync_rows()
        self.valueChanged.emit(self.value())

    def _sync_rows(self) -> None:
        mode = self.mode_combo.currentText()
        self.layer_row.setVisible(mode == "napari layer")
        self.file_row.setVisible(mode == "file path")
        file_mode = mode == "file path"
        self.series_row.setVisible(file_mode and self.series_combo.count() > 1)
        self.binding_combo.setVisible(file_mode)
        self.source_summary.setVisible(file_mode and bool(self.source_summary.text()))
        self.sample_row.setVisible(mode == "sample")


class ExampleWorkflowDialog(QDialog):
    """Searchable chooser for bundled example workflows."""

    def __init__(
        self,
        parent=None,
        examples: tuple[ExampleWorkflowSpec, ...] = EXAMPLE_WORKFLOWS,
    ):
        super().__init__(parent)
        self._examples = tuple(examples)
        self._selected_example_id = ""
        self.setWindowTitle("Open VIPP Example")
        self.resize(780, 520)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter examples, samples, or tasks")
        self.filter_edit.setClearButtonEnabled(True)
        layout.addWidget(self.filter_edit)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Example", "Input"])
        self.tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tree.setRootIsDecorated(True)
        layout.addWidget(self.tree, 1)

        self.details_label = QLabel("Select an example workflow.")
        self.details_label.setWordWrap(True)
        self.details_label.setStyleSheet(
            "color: #cbd5e1; padding: 6px; background: #1f2937; "
            "border: 1px solid #374151; border-radius: 4px;"
        )
        layout.addWidget(self.details_label)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        self.open_button = self.buttons.addButton("Open", QDialogButtonBox.AcceptRole)
        self.open_button.setEnabled(False)
        layout.addWidget(self.buttons)

        self.filter_edit.textChanged.connect(self._populate_tree)
        self.tree.itemSelectionChanged.connect(self._on_selection_changed)
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.open_button.clicked.connect(self._accept_if_selected)
        self.buttons.rejected.connect(self.reject)
        self._populate_tree()

    def selected_example(self) -> ExampleWorkflowSpec | None:
        return _example_workflow_by_id(self._selected_example_id)

    def select_example(self, example_id: str) -> None:
        target = str(example_id)
        for index in range(self.tree.topLevelItemCount()):
            category = self.tree.topLevelItem(index)
            for child_index in range(category.childCount()):
                child = category.child(child_index)
                if child.data(0, Qt.UserRole) == target:
                    self.tree.setCurrentItem(child)
                    return

    def _populate_tree(self) -> None:
        selected_id = self._selected_example_id
        query = _normalize_search_text(self.filter_edit.text())
        self.tree.clear()
        categories: dict[str, list[ExampleWorkflowSpec]] = {}
        for spec in self._examples:
            if query and not self._matches_query(spec, query):
                continue
            categories.setdefault(spec.category, []).append(spec)

        for category, specs in categories.items():
            category_item = QTreeWidgetItem([category, ""])
            category_item.setFlags(category_item.flags() & ~Qt.ItemIsSelectable)
            self.tree.addTopLevelItem(category_item)
            for spec in specs:
                sample_text = ", ".join(spec.samples)
                item = QTreeWidgetItem([spec.title, sample_text])
                item.setData(0, Qt.UserRole, spec.id)
                item.setToolTip(0, spec.description)
                item.setToolTip(1, sample_text)
                category_item.addChild(item)
        self.tree.expandAll()
        self.tree.resizeColumnToContents(0)
        self._restore_selection(selected_id)
        self._on_selection_changed()

    @staticmethod
    def _matches_query(spec: ExampleWorkflowSpec, query: str) -> bool:
        text = _normalize_search_text(
            " ".join(
                (
                    spec.category,
                    spec.title,
                    spec.filename,
                    " ".join(spec.samples),
                    spec.description,
                )
            )
        )
        return all(part in text for part in query.split())

    def _restore_selection(self, example_id: str) -> None:
        if not example_id:
            return
        self.select_example(example_id)

    def _on_selection_changed(self) -> None:
        item = self.tree.currentItem()
        example_id = str(item.data(0, Qt.UserRole) or "") if item is not None else ""
        spec = _example_workflow_by_id(example_id)
        self._selected_example_id = spec.id if spec is not None else ""
        self.open_button.setEnabled(spec is not None)
        if spec is None:
            self.open_button.setText("Open")
            self.details_label.setText("Select an example workflow.")
            return
        samples = ", ".join(spec.samples)
        self.open_button.setText(
            "Open batch demo..." if spec.generated_batch_demo else "Open"
        )
        source_label = "Demo data" if spec.generated_batch_demo else "Source sample"
        next_step = (
            "\n\nVIPP will ask where to save a small self-contained working "
            "copy, then open the batch window already configured and "
            "previewed. Click Run demo batch to process and validate it."
            if spec.generated_batch_demo
            else ""
        )
        self.details_label.setText(
            f"{spec.title}\n\n{spec.description}\n\n{source_label}: {samples}"
            f"{next_step}"
        )

    def _on_item_double_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        if item.data(0, Qt.UserRole):
            self._accept_if_selected()

    def _accept_if_selected(self) -> None:
        if self.selected_example() is not None:
            self.accept()


class CollectionBatchDialog(QDialog):
    """Front door for running a workflow over one or more local collections."""

    def __init__(self, parent=None, source_nodes: list[dict] | None = None):
        super().__init__(parent)
        self.setWindowTitle("Run collection batch")
        self.setMinimumWidth(880)
        self._source_rows: list[dict[str, object]] = []
        self._loaded_config_path: Path | None = None
        self._demo: SyntheticBatchDemo | None = None

        source_nodes = source_nodes or [
            {
                "node_id": "input",
                "title": "Image Source",
                "binding_mode": "collection",
            }
        ]

        self.output_edit = QLineEdit()
        self.format_combo = QComboBox()
        self.format_combo.addItems(["ome-tiff", "imagej-tiff", "tiff", "npy"])
        self.existing_policy_combo = QComboBox()
        self.existing_policy_combo.addItem("Error", ExistingFilePolicy.ERROR.value)
        self.existing_policy_combo.addItem("Skip", ExistingFilePolicy.SKIP.value)
        self.existing_policy_combo.addItem(
            "Overwrite", ExistingFilePolicy.OVERWRITE.value
        )
        self.workflow_checkbox = QCheckBox("Save workflow JSON")
        self.workflow_checkbox.setChecked(True)
        self.workflow_checkbox.setEnabled(False)
        self.workflow_checkbox.setToolTip(
            "The workflow companion is required for reproducible batch configs."
        )
        self.script_checkbox = QCheckBox("Save batch runner Python script")
        self.script_checkbox.setChecked(True)
        self.continue_checkbox = QCheckBox("Continue after item failures")
        self.continue_checkbox.setChecked(True)
        self.preview_button = QPushButton("Preview batch")
        self.preview_button.clicked.connect(self._preview_batch)
        self.preview_status = QLabel("")
        self.preview_status.setWordWrap(True)
        self.preview_status.setStyleSheet("color: #94a3b8;")
        self.preview_table = QTableWidget(0, 4)
        self.preview_table.setHorizontalHeaderLabels(
            ["#", "Batch item", "Outputs", "Preflight"]
        )
        self.preview_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.preview_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.preview_table.setMinimumHeight(170)
        self.preview_table.setMaximumHeight(260)

        source_group = QGroupBox("Batch sources")
        self.source_layout = QVBoxLayout(source_group)
        self._set_source_nodes(source_nodes)

        output_button = QPushButton("Folder...")
        output_button.clicked.connect(self._browse_output)
        output_row = QWidget()
        output_layout = QHBoxLayout(output_row)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.addWidget(self.output_edit, 1)
        output_layout.addWidget(output_button)

        form = QFormLayout()
        form.addRow("Output folder", output_row)
        form.addRow("Default image format", self.format_combo)
        form.addRow("Existing files", self.existing_policy_combo)
        form.addRow("", self.workflow_checkbox)
        form.addRow("", self.script_checkbox)
        form.addRow("", self.continue_checkbox)

        self.load_config_button = QPushButton("Load config...")
        self.load_config_button.clicked.connect(self._load_config)
        self.save_config_button = QPushButton("Save config...")
        self.save_config_button.clicked.connect(self._save_config)
        self.demo_config_button = QPushButton("Open batch demo...")
        self.demo_config_button.setToolTip(
            "Open a ready-to-run deterministic batch workspace with paired "
            "inputs, explicit outputs, provenance, and ground-truth validation."
        )
        self.demo_config_button.clicked.connect(self._create_demo)
        config_row = QWidget()
        config_layout = QHBoxLayout(config_row)
        config_layout.setContentsMargins(0, 0, 0, 0)
        config_layout.addWidget(self.load_config_button)
        config_layout.addWidget(self.save_config_button)
        config_layout.addWidget(self.demo_config_button)
        config_layout.addStretch(1)

        help_label = QLabel(
            "Bind each Image Source that should change per batch item to a "
            "folder and file pattern. VIPP zips bound sources by sorted file "
            "order, assigns each row a stable batch ID, and saves only Batch "
            "Output nodes when present. Without Batch Output nodes, terminal "
            "graph outputs are saved as a compatibility fallback."
        )
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color: #94a3b8;")

        self.demo_guide_label = QLabel("")
        self.demo_guide_label.setWordWrap(True)
        self.demo_guide_label.setTextFormat(Qt.RichText)
        self.demo_guide_label.setStyleSheet(
            "color: #dbeafe; padding: 9px; background: #172554; "
            "border: 1px solid #3b82f6; border-radius: 4px;"
        )
        self.demo_guide_label.hide()

        preview_row = QWidget()
        preview_layout = QHBoxLayout(preview_row)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.addWidget(self.preview_button)
        preview_layout.addWidget(self.preview_status, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.run_button = buttons.button(QDialogButtonBox.Ok)
        self.run_button.setText("Run batch")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(config_row)
        layout.addWidget(self.demo_guide_label)
        layout.addWidget(source_group)
        layout.addLayout(form)
        layout.addWidget(help_label)
        layout.addWidget(preview_row)
        layout.addWidget(self.preview_table)
        layout.addWidget(buttons)

    def set_demo_context(self, demo: SyntheticBatchDemo) -> None:
        """Present a generated bundle as a ready-to-run example workspace."""
        self._demo = demo
        self.run_button.setText("Run demo batch")
        self.demo_guide_label.setText(
            "<b>Ready-to-run batch demo</b><br>"
            "Two collection sources are paired by sorted position. The graph "
            "automatically shows the first paired field while the batch preview "
            "plans all three items and three explicit outputs per item: an NPY "
            "image, TIFF labels, and a TSV table. Explore source bindings, "
            "preview planning, Error/Skip/Overwrite replay policies, failure "
            "continuation, and portable config/runner controls below. Click "
            "<b>Run demo batch</b> to execute the workflow and validate its "
            "scientific outputs, manifests, archive, and per-item provenance "
            "against exact ground truth.<br>"
            f"Working copy: <code>{escape(str(demo.root))}</code>"
        )
        self.demo_guide_label.show()

    def _set_source_nodes(self, source_nodes: list[dict]) -> None:
        self.preview_table.setRowCount(0)
        self.preview_status.clear()
        for row in self._source_rows:
            widget = row["widget"]
            self.source_layout.removeWidget(widget)
            widget.deleteLater()
        self._source_rows.clear()
        for index, source in enumerate(source_nodes):
            row = self._make_source_row(
                str(source.get("node_id", f"source_{index + 1}")),
                str(source.get("title", f"Image Source {index + 1}")),
                str(source.get("binding_mode", "")),
                index=index,
            )
            self.source_layout.addWidget(row)
        if self._source_rows:
            self.input_edit = self._source_rows[0]["folder"]
            self.pattern_edit = self._source_rows[0]["pattern"]
        else:
            self.input_edit = QLineEdit()
            self.pattern_edit = QLineEdit("*.tif;*.tiff;*.ome.tif;*.ome.tiff")

    def _make_source_row(
        self,
        node_id: str,
        title: str,
        binding_mode: str,
        *,
        index: int,
    ) -> QWidget:
        folder_edit = QLineEdit()
        pattern_edit = QLineEdit("*.tif;*.tiff;*.ome.tif;*.ome.tiff")
        browse_button = QPushButton("Folder...")
        browse_button.clicked.connect(
            lambda _checked=False, edit=folder_edit: self._browse_source_input(edit)
        )
        title_label = QLabel(
            f"{title} ({node_id})"
            + ("  - collection" if binding_mode == "collection" else "")
        )
        title_label.setStyleSheet("font-weight: 650;")

        folder_row = QWidget()
        folder_layout = QHBoxLayout(folder_row)
        folder_layout.setContentsMargins(0, 0, 0, 0)
        folder_layout.addWidget(QLabel("Folder"))
        folder_layout.addWidget(folder_edit, 1)
        folder_layout.addWidget(browse_button)

        pattern_row = QWidget()
        pattern_layout = QHBoxLayout(pattern_row)
        pattern_layout.setContentsMargins(0, 0, 0, 0)
        pattern_layout.addWidget(QLabel("Pattern"))
        pattern_layout.addWidget(pattern_edit, 1)

        row = QFrame()
        row.setFrameShape(QFrame.StyledPanel)
        row.setStyleSheet(
            "QFrame { border: 1px solid #334155; border-radius: 4px; }"
            "QLabel { border: none; }"
            "QLineEdit { border: 1px solid #475569; }"
        )
        row_layout = QVBoxLayout(row)
        row_layout.setContentsMargins(8, 6, 8, 6)
        row_layout.addWidget(title_label)
        row_layout.addWidget(folder_row)
        row_layout.addWidget(pattern_row)
        self._source_rows.append(
            {
                "node_id": node_id,
                "title": title,
                "folder": folder_edit,
                "pattern": pattern_edit,
                "index": index,
                "widget": row,
                "title_label": title_label,
                "binding_mode": binding_mode,
            }
        )
        return row

    def values(self) -> dict[str, object]:
        bindings = []
        for row in self._source_rows:
            bindings.append(
                {
                    "node_id": row["node_id"],
                    "title": row["title"],
                    "input_dir": row["folder"].text(),
                    "pattern": row["pattern"].text(),
                }
            )
        return {
            "input_dir": self.input_edit.text(),
            "output_dir": self.output_edit.text(),
            "pattern": self.pattern_edit.text(),
            "source_bindings": bindings,
            "image_format": self.format_combo.currentText(),
            "existing_file_policy": str(self.existing_policy_combo.currentData()),
            "save_workflow_snapshot": self.workflow_checkbox.isChecked(),
            "save_python_script": self.script_checkbox.isChecked(),
            "continue_on_error": self.continue_checkbox.isChecked(),
        }

    def _preview_batch(self) -> bool:
        parent = self.parent()
        if parent is None or not hasattr(parent, "_preview_collection_batch"):
            self.preview_status.setText("Preview is available from the VIPP widget.")
            return False
        try:
            rows = parent._preview_collection_batch(
                **self.values(),
                preview_limit=25,
            )
        except Exception as exc:
            self.preview_table.setRowCount(0)
            self.preview_status.setText(f"Preview failed: {exc}")
            return False
        self.preview_table.setRowCount(len(rows))
        for row_index, item in enumerate(rows):
            self.preview_table.setItem(
                row_index,
                0,
                QTableWidgetItem(str(item.batch_index)),
            )
            source_text = "\n".join(
                f"{node_id}: {path.name}" for node_id, path in item.sources.items()
            )
            self.preview_table.setItem(
                row_index,
                1,
                QTableWidgetItem(f"{item.batch_id}\n{source_text}"),
            )
            output_text = "\n".join(str(path) for path in item.outputs)
            self.preview_table.setItem(row_index, 2, QTableWidgetItem(output_text))
            status_text = "\n".join(item.output_statuses)
            self.preview_table.setItem(row_index, 3, QTableWidgetItem(status_text))
        self.preview_table.resizeColumnsToContents()
        self.preview_table.resizeRowsToContents()
        total_items = getattr(rows, "total_items", len(rows))
        collision_count = getattr(
            rows,
            "collision_count",
            sum(
                status
                in {
                    "exists; collision",
                    "duplicate planned destination",
                    "destination overlaps an input",
                }
                for row in rows
                for status in row.output_statuses
            ),
        )
        explicit_outputs = getattr(
            rows,
            "explicit_outputs",
            bool(not rows or rows[0].explicit_outputs),
        )
        messages = [
            f"Showing {len(rows)} of {total_items} planned batch item(s)."
        ]
        if collision_count:
            messages.append(f"{collision_count} collision(s) need attention.")
        if not explicit_outputs:
            messages.append(
                "Compatibility fallback: terminal graph outputs will be saved; "
                "add Batch Output nodes to make the selection explicit."
            )
        if self._demo is not None:
            planned_outputs = sum(len(row.outputs) for row in rows)
            messages.append(
                "Demo ready - click Run demo batch to process "
                f"{total_items} paired items, write {planned_outputs} outputs, "
                "and validate the results and provenance."
            )
        self.preview_status.setText(" ".join(messages))
        return True

    def _create_demo(self) -> None:
        parent = self.parent()
        if parent is None or not hasattr(parent, "_choose_collection_batch_demo"):
            self.preview_status.setText(
                "Opening the synthetic demo is available from the VIPP widget."
            )
            return
        try:
            demo = parent._choose_collection_batch_demo(dialog_parent=self)
            if demo is None:
                return
            self._set_source_nodes(parent._batch_source_rows())
            config = parent._load_collection_batch_config(demo.config_path)
            self._apply_config(config)
            self._loaded_config_path = demo.config_path
            self.set_demo_context(demo)
            if not self._preview_batch():
                return
        except Exception as exc:
            self.preview_status.setText(f"Could not open batch demo: {exc}")
            return

    def _load_config(self) -> None:
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Load batch configuration",
            str(self._loaded_config_path or BATCH_CONFIG_FILENAME),
            "VIPP batch config (*.json);;JSON files (*.json)",
        )
        if not path:
            return
        try:
            parent = self.parent()
            if parent is not None and hasattr(parent, "_load_collection_batch_config"):
                config = parent._load_collection_batch_config(path)
            else:
                config = load_batch_config(path)
            self._apply_config(config)
        except Exception as exc:
            self.preview_status.setText(f"Could not load batch config: {exc}")
            return
        self._loaded_config_path = Path(path)
        self.preview_status.setText(f"Loaded {Path(path).name}.")

    def _save_config(self) -> None:
        default_dir = Path(self.output_edit.text()).expanduser()
        default_path = (
            default_dir / BATCH_CONFIG_FILENAME
            if str(default_dir).strip()
            else Path(BATCH_CONFIG_FILENAME)
        )
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save batch configuration",
            str(default_path),
            "VIPP batch config (*.json);;JSON files (*.json)",
        )
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        parent = self.parent()
        if parent is None or not hasattr(parent, "_save_collection_batch_config"):
            self.preview_status.setText(
                "Saving a batch config is available from the VIPP widget."
            )
            return
        try:
            saved = parent._save_collection_batch_config(path, **self.values())
        except Exception as exc:
            self.preview_status.setText(f"Could not save batch config: {exc}")
            return
        self._loaded_config_path = Path(path)
        names = ", ".join(item.name for item in saved)
        self.preview_status.setText(f"Saved {names}.")

    def _apply_config(self, config: BatchConfig) -> None:
        rows = {str(row["node_id"]): row for row in self._source_rows}
        missing = [
            source.node_id for source in config.sources if source.node_id not in rows
        ]
        if missing:
            raise ValueError(
                "Config references source nodes not present in this workflow: "
                + ", ".join(missing)
                + "."
            )
        configured_ids = [source.node_id for source in config.sources]
        ordered_rows = [rows[node_id] for node_id in configured_ids]
        ordered_rows.extend(
            row
            for row in self._source_rows
            if str(row["node_id"]) not in configured_ids
        )
        self._source_rows = ordered_rows
        for row in self._source_rows:
            row["folder"].clear()
            self.source_layout.removeWidget(row["widget"])
            self.source_layout.addWidget(row["widget"])
        for source in config.sources:
            row = rows[source.node_id]
            row["title"] = source.title
            suffix = (
                "  - collection" if row["binding_mode"] == "collection" else ""
            )
            row["title_label"].setText(
                f"{source.title} ({source.node_id}){suffix}"
            )
            row["folder"].setText(str(config.resolve_path(source.input_dir)))
            row["pattern"].setText(source.pattern)
        if self._source_rows:
            self.input_edit = self._source_rows[0]["folder"]
            self.pattern_edit = self._source_rows[0]["pattern"]
        self.output_edit.setText(str(config.resolve_path(config.output_dir)))
        format_index = self.format_combo.findText(config.default_image_format)
        if format_index >= 0:
            self.format_combo.setCurrentIndex(format_index)
        policy_index = self.existing_policy_combo.findData(
            config.existing_file_policy.value
        )
        if policy_index >= 0:
            self.existing_policy_combo.setCurrentIndex(policy_index)
        self.workflow_checkbox.setChecked(True)
        self.script_checkbox.setChecked(config.save_python_script)
        self.continue_checkbox.setChecked(config.continue_on_error)

    def _browse_input(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Select batch input folder",
            self.input_edit.text(),
        )
        if path:
            self.input_edit.setText(path)

    def _browse_source_input(self, edit: QLineEdit) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Select batch input folder",
            edit.text(),
        )
        if path:
            edit.setText(path)

    def _browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Select batch output folder",
            self.output_edit.text(),
        )
        if path:
            self.output_edit.setText(path)


class AxisIntervalSlider(QWidget):
    """Small two-handle integer range slider for axis slicing."""

    valueChanged = Signal(int, int)

    def __init__(self, minimum: int = 0, maximum: int = 0, parent=None):
        super().__init__(parent)
        self._minimum = int(minimum)
        self._maximum = int(maximum)
        self._start = self._minimum
        self._end = self._maximum
        self._active_handle: str | None = None
        self.setMinimumHeight(26)
        self.setMinimumWidth(150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def value(self) -> tuple[int, int]:
        return self._start, self._end

    def setRange(self, minimum: int, maximum: int) -> None:
        self._minimum = int(minimum)
        self._maximum = max(int(maximum), self._minimum)
        self.setValue(self._start, self._end, emit=False)

    def setValue(self, start: int, end: int, *, emit: bool = True) -> None:
        start, end = self._normalized_values(start, end)
        changed = start != self._start or end != self._end
        self._start = start
        self._end = end
        self.update()
        if emit and changed:
            self.valueChanged.emit(self._start, self._end)

    def sizeHint(self) -> QSize:
        return QSize(190, 26)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        left = 8
        right = max(left + 1, self.width() - 8)
        center_y = self.height() // 2
        start_x = self._x_for_value(self._start)
        end_x = self._x_for_value(self._end)

        painter.setPen(QPen(QColor("#4b5563"), 4, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(left, center_y, right, center_y)
        painter.setPen(QPen(QColor("#60a5fa"), 4, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(start_x, center_y, end_x, center_y)

        painter.setPen(QPen(QColor("#93c5fd"), 1))
        painter.setBrush(QBrush(QColor("#1d4ed8")))
        painter.drawEllipse(start_x - 6, center_y - 6, 12, 12)
        painter.drawEllipse(end_x - 6, center_y - 6, 12, 12)

    def mousePressEvent(self, event) -> None:
        x = self._event_x(event)
        start_x = self._x_for_value(self._start)
        end_x = self._x_for_value(self._end)
        self._active_handle = "start" if abs(x - start_x) <= abs(x - end_x) else "end"
        self._set_active_value_from_x(x)
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._active_handle is None:
            return
        self._set_active_value_from_x(self._event_x(event))
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        self._active_handle = None
        event.accept()

    def _set_active_value_from_x(self, x: float) -> None:
        value = self._value_for_x(x)
        if self._active_handle == "start":
            self.setValue(value, self._end)
        elif self._active_handle == "end":
            self.setValue(self._start, value)

    def _normalized_values(self, start: int, end: int) -> tuple[int, int]:
        start = int(np.clip(int(start), self._minimum, self._maximum))
        end = int(np.clip(int(end), self._minimum, self._maximum))
        if start > end:
            start, end = end, start
        return start, end

    def _x_for_value(self, value: int) -> int:
        if self._maximum <= self._minimum:
            return 8
        fraction = (int(value) - self._minimum) / (self._maximum - self._minimum)
        return int(round(8 + fraction * max(self.width() - 16, 1)))

    def _value_for_x(self, x: float) -> int:
        if self._maximum <= self._minimum:
            return self._minimum
        fraction = (float(x) - 8) / max(self.width() - 16, 1)
        value = self._minimum + fraction * (self._maximum - self._minimum)
        return int(np.clip(round(value), self._minimum, self._maximum))

    def _event_x(self, event) -> float:
        if hasattr(event, "position"):
            return float(event.position().x())
        return float(event.pos().x())


class AxisSelectionRow(QWidget):
    """Explicit keep-range/remove-index controls for one metadata axis."""

    valueChanged = Signal()

    def __init__(
        self,
        option: AxisSliceOption,
        *,
        mode: str = "keep",
        start: int = 0,
        end: int | None = None,
        index: int = 0,
        parent=None,
    ):
        super().__init__(parent)
        self.option = option
        self._updating = False
        maximum = max(option.size - 1, 0)

        self.title_label = QLabel()
        self.title_label.setTextFormat(Qt.RichText)
        self.title_label.setText(_axis_heading_text(option))
        self.title_label.setToolTip(
            f"{option.title}: {option.axis_type} axis, size {option.size}"
        )
        self.title_label.setMinimumWidth(0)
        self.title_label.setStyleSheet("padding: 0 2px;")

        self.keep_button = self._mode_button(
            "Keep",
            "Keep this axis and crop it to the selected range.",
        )
        self.remove_button = self._mode_button(
            "Remove",
            "Remove this axis by taking one selected index.",
        )

        self.range_slider = AxisIntervalSlider(0, maximum)
        self.start_box = QSpinBox()
        self.end_box = QSpinBox()
        self.index_slider = QSlider(Qt.Horizontal)
        self.index_box = QSpinBox()
        for box in (self.start_box, self.end_box, self.index_box):
            _configure_numeric_spin_box(box)
        for widget in (self.start_box, self.end_box, self.index_slider, self.index_box):
            widget.setRange(0, maximum)
        for box in (self.start_box, self.end_box, self.index_box):
            box.setFixedWidth(58)
            box.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.range_panel = QWidget()
        self.range_panel.setFixedHeight(58)
        range_layout = QVBoxLayout(self.range_panel)
        range_layout.setContentsMargins(0, 2, 0, 4)
        range_layout.setSpacing(2)
        range_slider_line = QHBoxLayout()
        range_slider_line.setContentsMargins(0, 0, 0, 0)
        range_slider_line.setSpacing(5)
        range_label = QLabel("Range")
        range_label.setMinimumWidth(42)
        range_slider_line.addWidget(range_label)
        range_slider_line.addWidget(self.range_slider, 1)
        range_layout.addLayout(range_slider_line)
        range_value_line = QHBoxLayout()
        range_value_line.setContentsMargins(42, 0, 0, 0)
        range_value_line.setSpacing(5)
        range_value_line.addWidget(QLabel("Start"))
        range_value_line.addWidget(self.start_box)
        range_value_line.addWidget(QLabel("End"))
        range_value_line.addWidget(self.end_box)
        range_value_line.addStretch(1)
        range_layout.addLayout(range_value_line)

        self.remove_panel = QWidget()
        self.remove_panel.setFixedHeight(58)
        remove_layout = QVBoxLayout(self.remove_panel)
        remove_layout.setContentsMargins(0, 2, 0, 4)
        remove_layout.setSpacing(2)
        index_slider_line = QHBoxLayout()
        index_slider_line.setContentsMargins(0, 0, 0, 0)
        index_slider_line.setSpacing(5)
        index_label = QLabel("Index")
        index_label.setMinimumWidth(42)
        index_slider_line.addWidget(index_label)
        index_slider_line.addWidget(self.index_slider, 1)
        remove_layout.addLayout(index_slider_line)
        index_value_line = QHBoxLayout()
        index_value_line.setContentsMargins(42, 0, 0, 0)
        index_value_line.setSpacing(5)
        index_value_line.addWidget(QLabel("Index"))
        index_value_line.addWidget(self.index_box)
        index_value_line.addStretch(1)
        remove_layout.addLayout(index_value_line)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)
        header.addWidget(self.title_label)
        header.addWidget(self.keep_button)
        header.addWidget(self.remove_button)
        header.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 8)
        layout.setSpacing(4)
        layout.addLayout(header)
        layout.addWidget(self.range_panel)
        layout.addWidget(self.remove_panel)
        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setStyleSheet("color: #334155; background: #334155; max-height: 1px;")
        layout.addWidget(divider)

        self.set_range(start, maximum if end is None else end, emit=False)
        self.set_index(index, emit=False)
        self.set_mode(mode, emit=False)

        self.keep_button.clicked.connect(lambda: self.set_mode("keep"))
        self.remove_button.clicked.connect(lambda: self.set_mode("remove"))
        self.range_slider.valueChanged.connect(self._on_range_slider_changed)
        self.start_box.valueChanged.connect(self._on_start_changed)
        self.end_box.valueChanged.connect(self._on_end_changed)
        self.index_slider.valueChanged.connect(self._on_index_changed)
        self.index_box.valueChanged.connect(self._on_index_changed)

    def mode(self) -> str:
        return "remove" if self.remove_button.isChecked() else "keep"

    def range_value(self) -> tuple[int, int]:
        return self.start_box.value(), self.end_box.value()

    def index_value(self) -> int:
        return self.index_box.value()

    def is_full_range(self) -> bool:
        start, end = self.range_value()
        return start == 0 and end == max(self.option.size - 1, 0)

    def set_mode(self, mode: str, emit: bool = True) -> None:
        mode = "remove" if mode == "remove" else "keep"
        self._updating = True
        with QSignalBlocker(self.keep_button), QSignalBlocker(self.remove_button):
            self.keep_button.setChecked(mode == "keep")
            self.remove_button.setChecked(mode == "remove")
        self.range_panel.setVisible(mode == "keep")
        self.remove_panel.setVisible(mode == "remove")
        self._updating = False
        self._refresh_mode_styles()
        if emit:
            self.valueChanged.emit()

    def set_range(self, start: int, end: int, emit: bool = True) -> None:
        maximum = max(self.option.size - 1, 0)
        start = int(np.clip(start, 0, maximum))
        end = int(np.clip(end, 0, maximum))
        if start > end:
            start, end = end, start
        self._updating = True
        with _control_signal_blockers(
            (self.range_slider, self.start_box, self.end_box)
        ):
            self.range_slider.setValue(start, end, emit=False)
            self.start_box.setValue(start)
            self.end_box.setValue(end)
        self._updating = False
        if emit:
            self.valueChanged.emit()

    def set_index(self, index: int, emit: bool = True) -> None:
        maximum = max(self.option.size - 1, 0)
        index = int(np.clip(index, 0, maximum))
        self._updating = True
        with _control_signal_blockers((self.index_slider, self.index_box)):
            self.index_slider.setValue(index)
            self.index_box.setValue(index)
        self._updating = False
        if emit:
            self.valueChanged.emit()

    def _mode_button(self, text: str, tooltip: str) -> QToolButton:
        button = QToolButton()
        button.setText(text)
        button.setToolTip(tooltip)
        button.setCheckable(True)
        button.setMinimumHeight(24)
        button.setMinimumWidth(58)
        button.setStyleSheet(
            "QToolButton { border: 1px solid #4b5563; border-radius: 4px; "
            "padding: 2px 7px; color: #d1d5db; background: #111827; }"
            "QToolButton:checked { border-color: #60a5fa; color: #ffffff; "
            "background: #2563eb; }"
        )
        return button

    def _refresh_mode_styles(self) -> None:
        self.title_label.setText(_axis_heading_text(self.option, mode=self.mode()))

    def _on_range_slider_changed(self, start: int, end: int) -> None:
        if self._updating:
            return
        self.set_range(start, end)

    def _on_start_changed(self, value: int) -> None:
        if self._updating:
            return
        start = int(value)
        end = max(self.end_box.value(), start)
        self.set_range(start, end)

    def _on_end_changed(self, value: int) -> None:
        if self._updating:
            return
        end = int(value)
        start = min(self.start_box.value(), end)
        self.set_range(start, end)

    def _on_index_changed(self, value: int) -> None:
        if self._updating:
            return
        self.set_index(value)


class AxisSliceControl(QWidget):
    """Metadata-aware selector for keeping ranges or removing axes."""

    valueChanged = Signal(object)

    def __init__(
        self,
        options: list[AxisSliceOption],
        value: dict | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._options: list[AxisSliceOption] = []
        self._rows: dict[int, AxisSelectionRow] = {}
        self._updating = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(7)
        title = QLabel("Slice axes")
        title.setStyleSheet("font-weight: 600;")
        layout.addWidget(title)
        self._row_layout = QVBoxLayout()
        self._row_layout.setContentsMargins(0, 0, 0, 0)
        self._row_layout.setSpacing(2)
        layout.addLayout(self._row_layout)

        self.set_options(options, value=value, emit=False)

    def value(self) -> dict[str, int | str | bool]:
        ranges = self._modified_ranges()
        removals = self._removed_axes()
        remove_axes = sorted(removals)
        first_axis = remove_axes[0] if remove_axes else 0
        first_index = removals[first_axis] if remove_axes else 0
        return {
            "axis": first_axis,
            "index": first_index,
            "axes": ",".join(str(axis) for axis in remove_axes),
            "indices": ",".join(str(removals[axis]) for axis in remove_axes),
            "ranges": ";".join(
                f"{axis}:{start}:{end}" for axis, (start, end) in sorted(ranges.items())
            ),
            "range_mode": True,
            "remove_axes": ",".join(str(axis) for axis in remove_axes),
            "remove_indices": ",".join(str(removals[axis]) for axis in remove_axes),
        }

    def set_options(
        self,
        options: list[AxisSliceOption],
        value: dict | None = None,
        emit: bool = False,
    ) -> None:
        self._updating = True
        self._options = options or [AxisSliceOption(0, "axis 0", "unknown", 1)]
        current = value or self.value()
        ranges, removals = _axis_slice_state_from_value(current, self._options)
        option_indices = {option.index for option in self._options}
        ranges = {
            axis: value for axis, value in ranges.items() if axis in option_indices
        }
        removals = {
            axis: value for axis, value in removals.items() if axis in option_indices
        }
        self._clear_rows()
        self._build_rows(ranges, removals)
        self._updating = False
        if emit:
            self.valueChanged.emit(self.value())

    def set_ranges(
        self,
        ranges_by_axis: dict[int, tuple[int, int]],
        emit: bool = True,
    ) -> None:
        option_indices = {option.index for option in self._options}
        ranges = {
            int(axis): value
            for axis, value in ranges_by_axis.items()
            if int(axis) in option_indices
        }
        for axis, row in self._rows.items():
            if axis in ranges:
                start, end = ranges[axis]
                row.set_range(start, end, emit=False)
                row.set_mode("keep", emit=False)
            else:
                row.set_range(0, row.option.size - 1, emit=False)
                row.set_mode("keep", emit=False)
        if emit:
            self.valueChanged.emit(self.value())

    def set_removed_axes(
        self,
        indices_by_axis: dict[int, int],
        emit: bool = True,
    ) -> None:
        option_indices = {option.index for option in self._options}
        removals = {
            int(axis): int(index)
            for axis, index in indices_by_axis.items()
            if int(axis) in option_indices
        }
        for axis, row in self._rows.items():
            if axis in removals:
                row.set_index(removals[axis], emit=False)
                row.set_mode("remove", emit=False)
            else:
                row.set_mode("keep", emit=False)
        if emit:
            self.valueChanged.emit(self.value())

    def _build_rows(
        self,
        ranges: dict[int, tuple[int, int]],
        removals: dict[int, int],
    ) -> None:
        for option in self._options:
            start, end = ranges.get(option.index, (0, option.size - 1))
            index = removals.get(option.index, 0)
            row = AxisSelectionRow(
                option,
                mode="remove" if option.index in removals else "keep",
                start=start,
                end=end,
                index=index,
            )
            row.valueChanged.connect(self._emit_value_changed)
            self._row_layout.addWidget(row)
            self._rows[option.index] = row

    def _clear_rows(self) -> None:
        self._rows.clear()
        while self._row_layout.count():
            item = self._row_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _modified_ranges(self) -> dict[int, tuple[int, int]]:
        return {
            axis: row.range_value()
            for axis, row in self._rows.items()
            if row.mode() == "keep" and not row.is_full_range()
        }

    def _removed_axes(self) -> dict[int, int]:
        return {
            axis: row.index_value()
            for axis, row in self._rows.items()
            if row.mode() == "remove"
        }

    def _emit_value_changed(self) -> None:
        if not self._updating:
            self.valueChanged.emit(self.value())


class AxisOrderListWidget(QListWidget):
    """QListWidget that emits after a drag/drop reorder."""

    orderChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._drag_row = -1

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            item = self.itemAt(self._event_pos(event))
            self._drag_row = self.row(item) if item is not None else -1
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if event.buttons() & Qt.LeftButton and self._drag_row >= 0:
            item = self.itemAt(self._event_pos(event))
            target = self.row(item) if item is not None else -1
            if target >= 0 and target != self._drag_row:
                moved = self.takeItem(self._drag_row)
                self.insertItem(target, moved)
                self.setCurrentRow(target)
                self._drag_row = target
                self.orderChanged.emit()
                event.accept()
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        self._drag_row = -1
        super().mouseReleaseEvent(event)

    def dropEvent(self, event) -> None:
        super().dropEvent(event)
        self.orderChanged.emit()

    @staticmethod
    def _event_pos(event):
        if hasattr(event, "position"):
            return event.position().toPoint()
        return event.pos()


class ReorderAxesControl(QWidget):
    """Drag-reorder control for transposing image axes."""

    valueChanged = Signal(object)

    def __init__(
        self,
        options: list[AxisSliceOption],
        value: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self._options: list[AxisSliceOption] = []
        self._updating = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(7)

        title = QLabel("Output axis order")
        title.setStyleSheet("font-weight: 600;")
        layout.addWidget(title)

        hint = QLabel("Drag axes up or down. The top item becomes axis 0.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #94a3b8;")
        layout.addWidget(hint)

        warning = QLabel(
            "This transposes the data and redefines spatial axes for downstream "
            "nodes. Treat it like rotating a volume: physical scale follows the "
            "moved data axis."
        )
        warning.setWordWrap(True)
        warning.setStyleSheet("color: #fbbf24;")
        layout.addWidget(warning)

        self.list_widget = AxisOrderListWidget()
        self.list_widget.setDragDropMode(QAbstractItemView.InternalMove)
        self.list_widget.setDragEnabled(True)
        self.list_widget.setAcceptDrops(True)
        self.list_widget.setDefaultDropAction(Qt.MoveAction)
        self.list_widget.setDropIndicatorShown(True)
        self.list_widget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list_widget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.list_widget.setAlternatingRowColors(True)
        self.list_widget.orderChanged.connect(self._emit_value_changed)
        self.list_widget.itemSelectionChanged.connect(self._sync_button_state)
        layout.addWidget(self.list_widget)

        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.setSpacing(6)
        self.move_up_button = QPushButton("Move up")
        self.move_down_button = QPushButton("Move down")
        self.reset_button = QPushButton("Reset")
        button_row.addWidget(self.move_up_button)
        button_row.addWidget(self.move_down_button)
        button_row.addWidget(self.reset_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        self.serialized_label = QLabel()
        self.serialized_label.setStyleSheet("color: #64748b;")
        layout.addWidget(self.serialized_label)

        self.move_up_button.clicked.connect(lambda: self._move_selected(-1))
        self.move_down_button.clicked.connect(lambda: self._move_selected(1))
        self.reset_button.clicked.connect(self.reset_order)

        self.set_options(options, value=value, emit=False)

    def value(self) -> str:
        ordered = self._ordered_options()
        if [option.index for option in ordered] == [
            option.index for option in self._options
        ]:
            return ""
        return _axis_order_value(ordered)

    def set_options(
        self,
        options: list[AxisSliceOption],
        value: str = "",
        emit: bool = False,
    ) -> None:
        self._updating = True
        self._options = options or [AxisSliceOption(0, "axis 0", "unknown", 1)]
        self._build_items(_axis_options_from_order(value, self._options))
        self._updating = False
        self._sync_button_state()
        self._sync_serialized_label()
        if emit:
            self.valueChanged.emit(self.value())

    def set_order(self, order, emit: bool = True) -> None:
        self._updating = True
        self._build_items(_axis_options_from_order(order, self._options))
        self._updating = False
        self._sync_button_state()
        self._sync_serialized_label()
        if emit:
            self.valueChanged.emit(self.value())

    def reset_order(self) -> None:
        self.set_order([option.index for option in self._options])

    def _build_items(self, ordered: list[AxisSliceOption]) -> None:
        with QSignalBlocker(self.list_widget):
            self.list_widget.clear()
            for option in ordered:
                item = QListWidgetItem(_axis_order_item_text(option))
                item.setData(Qt.UserRole, int(option.index))
                item.setSizeHint(QSize(160, 28))
                item.setToolTip(
                    f"{option.title}: {option.axis_type} axis, size {option.size}"
                )
                item.setFlags(
                    item.flags()
                    | Qt.ItemIsDragEnabled
                    | Qt.ItemIsDropEnabled
                    | Qt.ItemIsSelectable
                    | Qt.ItemIsEnabled
                )
                self.list_widget.addItem(item)
            if self.list_widget.count():
                self.list_widget.setCurrentRow(0)
        height = 28 * max(self.list_widget.count(), 1) + 8
        self.list_widget.setMinimumHeight(min(max(height, 92), 220))

    def _ordered_options(self) -> list[AxisSliceOption]:
        by_index = {option.index: option for option in self._options}
        ordered: list[AxisSliceOption] = []
        for row in range(self.list_widget.count()):
            index = self.list_widget.item(row).data(Qt.UserRole)
            option = by_index.get(int(index))
            if option is not None:
                ordered.append(option)
        return ordered

    def _move_selected(self, delta: int) -> None:
        row = self.list_widget.currentRow()
        target = row + int(delta)
        if row < 0 or target < 0 or target >= self.list_widget.count():
            return
        item = self.list_widget.takeItem(row)
        self.list_widget.insertItem(target, item)
        self.list_widget.setCurrentRow(target)
        self._emit_value_changed()

    def _emit_value_changed(self) -> None:
        self._sync_button_state()
        self._sync_serialized_label()
        if not self._updating:
            self.valueChanged.emit(self.value())

    def _sync_button_state(self) -> None:
        row = self.list_widget.currentRow()
        count = self.list_widget.count()
        self.move_up_button.setEnabled(count > 1 and row > 0)
        self.move_down_button.setEnabled(count > 1 and 0 <= row < count - 1)
        self.reset_button.setEnabled(bool(self.value()))

    def _sync_serialized_label(self) -> None:
        value = self.value()
        if value:
            self.serialized_label.setText(f"Workflow value: {value}")
        else:
            self.serialized_label.setText("Workflow value: input order")


class SelectTableColumnsControl(QWidget):
    """Checklist control for keeping and ordering table columns."""

    valueChanged = Signal(object)

    def __init__(
        self,
        columns: list[str] | tuple[str, ...],
        value: str = "auto",
        parent=None,
    ):
        super().__init__(parent)
        self._columns: tuple[str, ...] = ()
        self._updating = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(7)

        title = QLabel("Columns to keep")
        title.setStyleSheet("font-weight: 600;")
        layout.addWidget(title)

        hint = QLabel(
            "Tick columns to include in the output table. Drag or move rows to "
            "control output order."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #94a3b8;")
        layout.addWidget(hint)

        self.list_widget = AxisOrderListWidget()
        self.list_widget.setDragDropMode(QAbstractItemView.InternalMove)
        self.list_widget.setDragEnabled(True)
        self.list_widget.setAcceptDrops(True)
        self.list_widget.setDefaultDropAction(Qt.MoveAction)
        self.list_widget.setDropIndicatorShown(True)
        self.list_widget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list_widget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.list_widget.setAlternatingRowColors(True)
        self.list_widget.setMinimumHeight(130)
        self.list_widget.setMaximumHeight(260)
        self.list_widget.orderChanged.connect(self._emit_value_changed)
        self.list_widget.itemChanged.connect(self._emit_value_changed)
        self.list_widget.itemSelectionChanged.connect(self._sync_button_state)
        layout.addWidget(self.list_widget)

        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.setSpacing(6)
        self.select_all_button = QPushButton("Select all")
        self.deselect_all_button = QPushButton("Deselect all")
        self.move_up_button = QPushButton("Move up")
        self.move_down_button = QPushButton("Move down")
        self.reset_button = QPushButton("Reset order")
        button_row.addWidget(self.select_all_button)
        button_row.addWidget(self.deselect_all_button)
        button_row.addWidget(self.move_up_button)
        button_row.addWidget(self.move_down_button)
        button_row.addWidget(self.reset_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet("color: #64748b;")
        layout.addWidget(self.summary_label)

        self.select_all_button.clicked.connect(lambda: self._set_all_checked(True))
        self.deselect_all_button.clicked.connect(lambda: self._set_all_checked(False))
        self.move_up_button.clicked.connect(lambda: self._move_selected(-1))
        self.move_down_button.clicked.connect(lambda: self._move_selected(1))
        self.reset_button.clicked.connect(self.reset_order)

        self.set_options(columns, value=value, emit=False)

    def value(self) -> str:
        selected = self._selected_columns()
        if not selected:
            return NO_TABLE_COLUMNS_VALUE
        if tuple(selected) == self._columns:
            return "auto"
        return ",".join(selected)

    def set_options(
        self,
        columns: list[str] | tuple[str, ...],
        value: str = "auto",
        emit: bool = False,
    ) -> None:
        self._updating = True
        self._columns = tuple(str(column) for column in columns)
        ordered, selected = self._ordered_columns_from_value(value)
        self._build_items(ordered, selected)
        self._updating = False
        self._sync_button_state()
        self._sync_summary_label()
        if emit:
            self.valueChanged.emit(self.value())

    def reset_order(self) -> None:
        selected = set(self._selected_columns())
        self._updating = True
        self._build_items(list(self._columns), selected)
        self._updating = False
        self._sync_button_state()
        self._sync_summary_label()
        self.valueChanged.emit(self.value())

    def _build_items(self, ordered: list[str], selected: set[str]) -> None:
        with QSignalBlocker(self.list_widget):
            self.list_widget.clear()
            for column in ordered:
                item = QListWidgetItem(column)
                item.setData(Qt.UserRole, column)
                item.setCheckState(
                    Qt.Checked if column in selected else Qt.Unchecked
                )
                item.setSizeHint(QSize(180, 28))
                item.setToolTip(column)
                item.setFlags(
                    item.flags()
                    | Qt.ItemIsUserCheckable
                    | Qt.ItemIsDragEnabled
                    | Qt.ItemIsDropEnabled
                    | Qt.ItemIsSelectable
                    | Qt.ItemIsEnabled
                )
                self.list_widget.addItem(item)
            if self.list_widget.count():
                self.list_widget.setCurrentRow(0)

    def _ordered_columns_from_value(self, value: str) -> tuple[list[str], set[str]]:
        available = list(self._columns)
        available_set = set(available)
        raw = str(value or "auto").strip()
        if not available:
            return [], set()
        if raw.lower() in {"", "auto"}:
            return available, set(available)
        if raw.lower() in {NO_TABLE_COLUMNS_VALUE, "none", "<none>"}:
            return available, set()
        requested = [
            column.strip()
            for column in raw.split(",")
            if column.strip() and column.strip() in available_set
        ]
        selected = set(dict.fromkeys(requested))
        ordered = list(dict.fromkeys(requested))
        ordered.extend(column for column in available if column not in selected)
        return ordered, selected

    def _selected_columns(self) -> list[str]:
        selected: list[str] = []
        for row in range(self.list_widget.count()):
            item = self.list_widget.item(row)
            if item is not None and item.checkState() == Qt.Checked:
                selected.append(str(item.data(Qt.UserRole)))
        return selected

    def _set_all_checked(self, checked: bool) -> None:
        self._updating = True
        with QSignalBlocker(self.list_widget):
            for row in range(self.list_widget.count()):
                item = self.list_widget.item(row)
                if item is not None:
                    item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
        self._updating = False
        self._sync_summary_label()
        self.valueChanged.emit(self.value())

    def _move_selected(self, direction: int) -> None:
        row = self.list_widget.currentRow()
        if row < 0:
            return
        target = row + int(direction)
        if not 0 <= target < self.list_widget.count():
            return
        item = self.list_widget.takeItem(row)
        self.list_widget.insertItem(target, item)
        self.list_widget.setCurrentRow(target)
        self._emit_value_changed()

    def _sync_button_state(self) -> None:
        has_columns = self.list_widget.count() > 0
        row = self.list_widget.currentRow()
        self.select_all_button.setEnabled(has_columns)
        self.deselect_all_button.setEnabled(has_columns)
        self.reset_button.setEnabled(has_columns)
        self.move_up_button.setEnabled(has_columns and row > 0)
        self.move_down_button.setEnabled(
            has_columns and 0 <= row < self.list_widget.count() - 1
        )

    def _sync_summary_label(self) -> None:
        total = len(self._columns)
        selected = len(self._selected_columns())
        if total <= 0:
            self.summary_label.setText(
                "No upstream table columns are available yet. Calculate or connect "
                "an upstream table-producing node first."
            )
            return
        self.summary_label.setText(f"Selected {selected} of {total} columns.")

    def _emit_value_changed(self, *_args) -> None:
        self._sync_button_state()
        self._sync_summary_label()
        if not self._updating:
            self.valueChanged.emit(self.value())


class PythonSyntaxHighlighter(QSyntaxHighlighter):
    """Small Python syntax highlighter for node-code inspection dialogs."""

    _KEYWORDS = (
        "False",
        "None",
        "True",
        "and",
        "as",
        "assert",
        "break",
        "class",
        "continue",
        "def",
        "del",
        "elif",
        "else",
        "except",
        "finally",
        "for",
        "from",
        "global",
        "if",
        "import",
        "in",
        "is",
        "lambda",
        "nonlocal",
        "not",
        "or",
        "pass",
        "raise",
        "return",
        "try",
        "while",
        "with",
        "yield",
    )
    _BUILTINS = (
        "bool",
        "dict",
        "float",
        "int",
        "len",
        "list",
        "max",
        "min",
        "np",
        "range",
        "repr",
        "set",
        "str",
        "tuple",
    )

    def __init__(self, document):
        super().__init__(document)
        self._rules: list[tuple[re.Pattern[str], QTextCharFormat]] = []
        self._add_rule(
            rf"\b({'|'.join(self._KEYWORDS)})\b",
            "#60a5fa",
            bold=True,
        )
        self._add_rule(rf"\b({'|'.join(self._BUILTINS)})\b", "#c084fc")
        self._add_rule(r"\b[A-Za-z_]\w*(?=\()", "#fbbf24")
        self._add_rule(r"\b\d+(\.\d+)?\b", "#fca5a5")
        self._string_format = self._format("#86efac")
        self._comment_format = self._format("#94a3b8", italic=True)

    def highlightBlock(self, text: str) -> None:  # noqa: N802
        for pattern, text_format in self._rules:
            for match in pattern.finditer(text):
                self.setFormat(
                    match.start(),
                    match.end() - match.start(),
                    text_format,
                )
        for match in re.finditer(r"(['\"])(?:\\.|(?!\1).)*\1", text):
            self.setFormat(
                match.start(),
                match.end() - match.start(),
                self._string_format,
            )
        comment_start = text.find("#")
        if comment_start >= 0:
            self.setFormat(
                comment_start,
                len(text) - comment_start,
                self._comment_format,
            )

    def _add_rule(
        self,
        pattern: str,
        color: str,
        *,
        bold: bool = False,
        italic: bool = False,
    ) -> None:
        self._rules.append((re.compile(pattern), self._format(color, bold, italic)))

    @staticmethod
    def _format(
        color: str,
        bold: bool = False,
        italic: bool = False,
    ) -> QTextCharFormat:
        text_format = QTextCharFormat()
        text_format.setForeground(QColor(color))
        if bold:
            text_format.setFontWeight(QFont.Bold)
        if italic:
            text_format.setFontItalic(True)
        return text_format


class HistogramPlot(QWidget):
    """Compact histogram display for the selected node output."""

    markerChanged = Signal(str, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._counts = np.array([], dtype=np.float32)
        self._series_counts = np.empty((0, 0), dtype=np.float32)
        self._series_colors: list[QColor] = []
        self._log_scale = False
        self._x_min_label = ""
        self._x_max_label = ""
        self._x_range: tuple[float, float] | None = None
        self._x_scale = "linear"
        self._markers: list[tuple[str, float, QColor]] = []
        self._draggable_markers: set[str] = set()
        self._drag_marker: str | None = None
        self.setMinimumHeight(120)
        self.setMouseTracking(True)

    def set_histogram(
        self,
        counts: np.ndarray | None,
        log_scale: bool,
        x_range: tuple[float, float] | None = None,
        colors: list[QColor] | None = None,
        markers: list[tuple[str, float, QColor]] | None = None,
        x_scale: str = "linear",
        draggable_markers: set[str] | None = None,
    ) -> None:
        self._counts = (
            np.asarray(counts, dtype=np.float32)
            if counts is not None
            else np.array([], dtype=np.float32)
        )
        if self._counts.ndim == 1:
            self._series_counts = self._counts.reshape(1, -1)
        elif self._counts.ndim == 2:
            self._series_counts = self._counts
            self._counts = self._series_counts.sum(axis=0)
        else:
            self._series_counts = np.empty((0, 0), dtype=np.float32)
            self._counts = np.array([], dtype=np.float32)
        self._series_colors = colors or _histogram_series_colors(
            self._series_counts.shape[0]
        )
        self._log_scale = log_scale
        self._x_range = x_range
        self._x_scale = x_scale
        self._markers = markers or []
        marker_labels = {label for label, _value, _color in self._markers}
        self._draggable_markers = set(draggable_markers or set()) & marker_labels
        if self._drag_marker not in self._draggable_markers:
            self._drag_marker = None
        if x_range is None or self._series_counts.size == 0:
            self._x_min_label = ""
            self._x_max_label = ""
        else:
            self._x_min_label = _format_histogram_label(x_range[0])
            self._x_max_label = _format_histogram_label(x_range[1])
        self.update()

    def paintEvent(self, event):  # noqa: N802
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect().adjusted(8, 8, -8, -8)
        painter.fillRect(rect, QColor("#111827"))
        painter.setPen(QPen(QColor("#374151"), 1))
        painter.drawRect(rect)

        label_height = painter.fontMetrics().height() + 5
        plot_frame = rect.adjusted(0, 0, 0, -label_height)
        plot_rect = plot_frame.adjusted(10, 8, -8, -10)
        self._draw_axes(painter, plot_rect)

        if self._series_counts.size == 0:
            painter.setPen(QColor("#9ca3af"))
            painter.drawText(plot_frame, Qt.AlignCenter, "No data")
            painter.end()
            return

        values = self._series_counts
        if self._log_scale:
            values = np.log10(values + 1.0)
        maximum = float(values.max())
        if maximum <= 0:
            painter.end()
            return

        self._draw_histogram_series(painter, plot_rect, values, maximum)
        self._draw_markers(painter, plot_rect)
        if self._x_min_label or self._x_max_label:
            painter.setPen(QColor("#9ca3af"))
            metrics = painter.fontMetrics()
            baseline = min(rect.bottom() - 2, plot_rect.bottom() + metrics.ascent() + 3)
            painter.drawText(plot_rect.left(), baseline, self._x_min_label)
            right_width = metrics.horizontalAdvance(self._x_max_label)
            painter.drawText(
                plot_rect.right() - right_width,
                baseline,
                self._x_max_label,
            )
        painter.end()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        marker = self._marker_at_point(_event_position(event))
        if marker is None:
            return
        self._drag_marker = marker
        self._emit_marker_from_point(marker, _event_position(event))

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        point = _event_position(event)
        if self._drag_marker is not None:
            self._emit_marker_from_point(self._drag_marker, point)
            return
        if self._marker_at_point(point) is None:
            self.unsetCursor()
        else:
            self.setCursor(Qt.SizeHorCursor)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._drag_marker is not None:
            self._emit_marker_from_point(self._drag_marker, _event_position(event))
        self._drag_marker = None

    def _draw_axes(self, painter: QPainter, plot_rect: QRect) -> None:
        painter.setPen(QPen(QColor("#64748b"), 1.2))
        painter.drawLine(
            plot_rect.left(),
            plot_rect.bottom(),
            plot_rect.right(),
            plot_rect.bottom(),
        )
        painter.drawLine(
            plot_rect.left(),
            plot_rect.top(),
            plot_rect.left(),
            plot_rect.bottom(),
        )

    def _draw_histogram_series(
        self,
        painter: QPainter,
        plot_rect: QRect,
        values: np.ndarray,
        maximum: float,
    ) -> None:
        width = max(plot_rect.width(), 1)
        height = max(plot_rect.height(), 1)
        step = max(int(np.ceil(values.shape[1] / width)), 1)
        for series_index, series_values in enumerate(values):
            reduced = np.array(
                [
                    series_values[i : i + step].max()
                    for i in range(0, series_values.size, step)
                ],
                dtype=np.float32,
            )
            if reduced.size == 0:
                continue
            color = self._series_colors[series_index % len(self._series_colors)]
            if values.shape[0] > 1:
                color = QColor(color)
                color.setAlpha(175)
            painter.setPen(QPen(color, 1.2))
            for index, value in enumerate(reduced):
                x = plot_rect.left() + int(index * width / max(reduced.size - 1, 1))
                y = plot_rect.bottom() - int((float(value) / maximum) * height)
                painter.drawLine(x, plot_rect.bottom(), x, y)

    def _draw_markers(self, painter: QPainter, plot_rect: QRect) -> None:
        if not self._markers or self._x_range is None:
            return
        metrics = painter.fontMetrics()
        label_y = plot_rect.top() + metrics.ascent() + 2
        for index, (label, value, color) in enumerate(self._markers):
            fraction = self._x_fraction(value)
            x = plot_rect.left() + int(fraction * max(plot_rect.width(), 1))
            painter.setPen(QPen(color, 2.0, Qt.DashLine))
            painter.drawLine(x, plot_rect.top(), x, plot_rect.bottom())
            text = f"{label} {_format_histogram_label(value)}"
            text_width = metrics.horizontalAdvance(text)
            rightmost_text_x = max(
                plot_rect.left(),
                plot_rect.right() - text_width,
            )
            text_x = int(np.clip(x + 3, plot_rect.left(), rightmost_text_x))
            painter.setPen(color)
            painter.drawText(
                text_x,
                label_y + index * (metrics.height() + 1),
                text,
            )

    def _x_fraction(self, value: int | float | Rational) -> float:
        if self._x_range is None:
            return 0.0
        minimum, maximum = self._x_range
        if maximum <= minimum:
            return 0.0
        integer_range = all(
            isinstance(item, (int, np.integer)) for item in (minimum, maximum)
        )
        if integer_range and isinstance(value, Rational):
            shifted_maximum = int(maximum) - int(minimum)
            shifted_value = min(max(value - int(minimum), 0), shifted_maximum)
            if self._x_scale == "log":
                return float(
                    np.log1p(float(shifted_value))
                    / np.log1p(max(shifted_maximum, 1))
                )
            return float(shifted_value / shifted_maximum)
        value = float(np.clip(value, minimum, maximum))
        if self._x_scale == "log":
            shifted_value = max(value - minimum, 0.0)
            shifted_maximum = maximum - minimum
            return float(np.log1p(shifted_value) / np.log1p(max(shifted_maximum, 1.0)))
        return float((value - minimum) / (maximum - minimum))

    def _plot_rect(self) -> QRect:
        rect = self.rect().adjusted(8, 8, -8, -8)
        label_height = self.fontMetrics().height() + 5
        plot_frame = rect.adjusted(0, 0, 0, -label_height)
        return plot_frame.adjusted(10, 8, -8, -10)

    def _marker_at_point(self, point) -> str | None:
        if not self._draggable_markers or self._x_range is None:
            return None
        plot_rect = self._plot_rect()
        if not plot_rect.adjusted(-8, 0, 8, 0).contains(point):
            return None
        candidates: list[tuple[float, str]] = []
        for label, value, _color in self._markers:
            if label not in self._draggable_markers:
                continue
            x = plot_rect.left() + self._x_fraction(value) * max(plot_rect.width(), 1)
            distance = abs(float(point.x()) - float(x))
            if distance <= 8.0:
                candidates.append((distance, label))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def _emit_marker_from_point(self, label: str, point) -> None:
        if self._x_range is None:
            return
        value = self._value_from_x(float(point.x()), self._plot_rect())
        self._replace_marker_value(label, value)
        self.markerChanged.emit(label, value)
        self.update()

    def _replace_marker_value(self, label: str, value: float) -> None:
        self._markers = [
            (
                marker_label,
                float(value) if marker_label == label else marker_value,
                color,
            )
            for marker_label, marker_value, color in self._markers
        ]

    def _value_from_x(self, x: float, plot_rect: QRect) -> float:
        if self._x_range is None:
            return 0.0
        minimum, maximum = self._x_range
        if maximum <= minimum:
            return float(minimum)
        width = max(float(plot_rect.width()), 1.0)
        fraction = float(np.clip((float(x) - plot_rect.left()) / width, 0.0, 1.0))
        if self._x_scale == "log":
            shifted_maximum = maximum - minimum
            shifted = np.expm1(fraction * np.log1p(max(shifted_maximum, 1.0)))
            return float(np.clip(minimum + shifted, minimum, maximum))
        return float(minimum + fraction * (maximum - minimum))


def _prepare_colocalization_scatter_density(
    channel_1: np.ndarray,
    channel_2: np.ndarray,
    *,
    threshold_1: float,
    threshold_2: float,
    roi_mask: np.ndarray | None,
    intensity_max: float,
    bins: int,
    progress=None,
) -> tuple[np.ndarray, int, int]:
    """Return an exact bounded-memory density and exact summary counts."""
    ch1 = np.asarray(channel_1)
    ch2 = np.asarray(channel_2)
    if ch1.shape != ch2.shape or ch1.size == 0:
        raise ValueError("Scatter channels must be non-empty and have matching shapes.")
    flat_1 = ch1.reshape(-1)
    flat_2 = ch2.reshape(-1)
    flat_roi: np.ndarray | None = None
    if roi_mask is not None:
        roi = np.asarray(roi_mask, dtype=bool)
        if roi.shape != ch1.shape:
            raise ValueError(
                f"ROI mask shape {roi.shape} does not match channels {ch1.shape}."
            )
        flat_roi = roi.reshape(-1)

    bins = int(np.clip(int(bins), 32, 512))
    intensity_max = max(float(intensity_max), 1.0)
    edges = np.linspace(0.0, intensity_max, bins + 1)
    density_counts = np.zeros((bins, bins), dtype=np.float64)
    roi_voxels = 0
    colocalized_voxels = 0
    threshold_1 = float(threshold_1)
    threshold_2 = float(threshold_2)
    for start in range(0, int(flat_1.size), INSPECTOR_STATISTICS_CHUNK_ELEMENTS):
        if progress is not None:
            progress.check_cancelled()
        stop = min(start + INSPECTOR_STATISTICS_CHUNK_ELEMENTS, int(flat_1.size))
        values_1 = flat_1[start:stop]
        values_2 = flat_2[start:stop]
        positive = np.greater_equal(values_1, threshold_1)
        np.logical_and(positive, values_2 >= threshold_2, out=positive)
        if flat_roi is None:
            roi_voxels += stop - start
            density_values_1 = values_1
            density_values_2 = values_2
        else:
            chunk_roi = flat_roi[start:stop]
            roi_voxels += int(np.count_nonzero(chunk_roi))
            np.logical_and(positive, chunk_roi, out=positive)
            density_values_1 = values_1[chunk_roi]
            density_values_2 = values_2[chunk_roi]
        colocalized_voxels += int(np.count_nonzero(positive))
        if density_values_1.size:
            chunk_density, _x_edges, _y_edges = np.histogram2d(
                density_values_1,
                density_values_2,
                bins=(edges, edges),
            )
            density_counts += chunk_density

    return density_counts, roi_voxels, colocalized_voxels


class ColocalizationScatterPlot(QWidget):
    """Interactive two-channel scatter-density plot with threshold guides."""

    thresholdChanged = Signal(int, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image: QImage | None = None
        self._threshold_1 = 25.0
        self._threshold_2 = 25.0
        self._intensity_max = 255.0
        self._channel_1_color = QColor("#ef4444")
        self._channel_2_color = QColor("#22c55e")
        self._colormap = "Viridis"
        self._summary = ""
        self._drag_axis: int | None = None
        self.setMinimumHeight(300)
        self.setMouseTracking(True)

    def set_density(
        self,
        density_counts: np.ndarray | None,
        *,
        threshold_1: float,
        threshold_2: float,
        intensity_max: float = 255.0,
        channel_1_color: object = "Red",
        channel_2_color: object = "Green",
        colormap: str = "Viridis",
        log_counts: bool = True,
        summary: str = "",
    ) -> None:
        """Render worker-prepared density counts without touching source images."""
        self._threshold_1 = float(threshold_1)
        self._threshold_2 = float(threshold_2)
        self._intensity_max = max(float(intensity_max), 1.0)
        self._channel_1_color = _qcolor_from_channel_color(
            channel_1_color,
            fallback="#ef4444",
        )
        self._channel_2_color = _qcolor_from_channel_color(
            channel_2_color,
            fallback="#22c55e",
        )
        self._colormap = str(colormap or "Viridis")
        self._summary = str(summary)
        self._image = self._density_image(density_counts, log_counts=log_counts)
        self.update()

    def clear(self, message: str = "Connect two channel inputs.") -> None:
        self._image = None
        self._summary = message
        self.update()

    def paintEvent(self, event):  # noqa: N802
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect().adjusted(8, 8, -8, -8)
        painter.fillRect(rect, QColor("#111827"))
        painter.setPen(QPen(QColor("#374151"), 1))
        painter.drawRect(rect)

        plot_rect = self._plot_rect()
        if self._image is None:
            painter.setPen(QColor("#9ca3af"))
            painter.drawText(rect, Qt.AlignCenter, self._summary or "No data")
            painter.end()
            return

        painter.drawImage(plot_rect, self._image)
        painter.setPen(QPen(QColor("#64748b"), 1.2))
        painter.drawRect(plot_rect)
        self._draw_thresholds(painter, plot_rect)
        self._draw_labels(painter, rect, plot_rect)
        painter.end()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        plot_rect = self._plot_rect()
        point = _event_position(event)
        if not plot_rect.contains(point):
            return
        vertical_x = self._x_from_value(self._threshold_1, plot_rect)
        horizontal_y = self._y_from_value(self._threshold_2, plot_rect)
        dx = abs(point.x() - vertical_x)
        dy = abs(point.y() - horizontal_y)
        self._drag_axis = 1 if dx <= dy else 2
        self._emit_threshold_from_point(point, plot_rect)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_axis is None:
            return
        point = _event_position(event)
        self._emit_threshold_from_point(point, self._plot_rect())

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._drag_axis is not None:
            point = _event_position(event)
            self._emit_threshold_from_point(point, self._plot_rect())
        self._drag_axis = None

    def _emit_threshold_from_point(self, point, plot_rect: QRect) -> None:
        if self._drag_axis == 1:
            value = self._value_from_x(point.x(), plot_rect)
            self._threshold_1 = value
            self.thresholdChanged.emit(1, value)
        elif self._drag_axis == 2:
            value = self._value_from_y(point.y(), plot_rect)
            self._threshold_2 = value
            self.thresholdChanged.emit(2, value)
        self.update()

    def _density_image(
        self,
        density_counts: np.ndarray | None,
        *,
        log_counts: bool,
    ) -> QImage | None:
        if density_counts is None:
            return None
        hist = np.asarray(density_counts)
        if hist.ndim != 2 or hist.size == 0:
            return None
        values = hist.T
        if bool(log_counts):
            values = np.log1p(values)
        maximum = float(np.max(values))
        if maximum > 0:
            values = values / maximum
        values = np.flipud(values)
        gray = np.clip(np.rint(np.sqrt(values) * 255.0), 0, 255).astype(np.uint8)
        rgb = _apply_monochrome_colormap(gray, self._colormap)
        rgb[values <= 0] = (4, 7, 15)
        return QImage(
            rgb.data,
            rgb.shape[1],
            rgb.shape[0],
            int(rgb.strides[0]),
            QImage.Format_RGB888,
        ).copy()

    def _draw_thresholds(self, painter: QPainter, plot_rect: QRect) -> None:
        x = self._x_from_value(self._threshold_1, plot_rect)
        y = self._y_from_value(self._threshold_2, plot_rect)
        painter.setPen(QPen(self._channel_1_color, 2.0, Qt.DashLine))
        painter.drawLine(x, plot_rect.top(), x, plot_rect.bottom())
        painter.setPen(QPen(self._channel_2_color, 2.0, Qt.DashLine))
        painter.drawLine(plot_rect.left(), y, plot_rect.right(), y)
        painter.setPen(QPen(QColor("#f8fafc"), 1.5))
        painter.drawEllipse(QPointF(float(x), float(y)), 3.5, 3.5)

    def _draw_labels(self, painter: QPainter, rect: QRect, plot_rect: QRect) -> None:
        metrics = painter.fontMetrics()
        axis_color = QColor("#9ca3af")
        zero_label = _format_histogram_label(0.0)
        max_label = _format_histogram_label(self._intensity_max)

        painter.setPen(axis_color)
        axis_value_y = plot_rect.bottom() + metrics.ascent() + 6
        painter.drawText(plot_rect.left(), axis_value_y, zero_label)
        painter.drawText(
            plot_rect.right() - metrics.horizontalAdvance(max_label),
            axis_value_y,
            max_label,
        )

        x_label = "Ch 1 intensity"
        painter.drawText(
            plot_rect.center().x() - metrics.horizontalAdvance(x_label) // 2,
            axis_value_y + metrics.height(),
            x_label,
        )

        y_value_x = plot_rect.left() - metrics.horizontalAdvance(max_label) - 8
        painter.drawText(y_value_x, plot_rect.top() + metrics.ascent(), max_label)
        painter.drawText(
            plot_rect.left() - metrics.horizontalAdvance(zero_label) - 8,
            plot_rect.bottom(),
            zero_label,
        )

        y_label = "Ch 2 intensity"
        painter.save()
        painter.translate(
            plot_rect.left() - 42,
            plot_rect.center().y() + metrics.horizontalAdvance(y_label) // 2,
        )
        painter.rotate(-90)
        painter.setPen(axis_color)
        painter.drawText(0, 0, y_label)
        painter.restore()

        t1_text = f"T1 {_format_histogram_label(self._threshold_1)}"
        t1_width = metrics.horizontalAdvance(t1_text)
        t1_x = int(
            np.clip(
                self._x_from_value(self._threshold_1, plot_rect) + 4,
                plot_rect.left() + 2,
                plot_rect.right() - t1_width - 2,
            )
        )
        painter.setPen(self._channel_1_color)
        painter.drawText(t1_x, plot_rect.bottom() - 4, t1_text)

        t2_text = f"T2 {_format_histogram_label(self._threshold_2)}"
        t2_y = int(
            np.clip(
                self._y_from_value(self._threshold_2, plot_rect) - 4,
                plot_rect.top() + metrics.ascent() + 2,
                plot_rect.bottom() - 4,
            )
        )
        painter.setPen(self._channel_2_color)
        painter.drawText(
            plot_rect.left() + 3,
            t2_y,
            t2_text,
        )

        if self._summary:
            summary_width = metrics.horizontalAdvance(self._summary)
            painter.setPen(axis_color)
            painter.drawText(
                max(plot_rect.left(), plot_rect.right() - summary_width),
                plot_rect.top() + metrics.ascent() + 2,
                self._summary,
            )

    def _plot_rect(self) -> QRect:
        rect = self.rect().adjusted(8, 8, -8, -8)
        metrics = self.fontMetrics()
        max_label = _format_histogram_label(self._intensity_max)
        left_margin = max(56, metrics.horizontalAdvance(max_label) + 22)
        right_margin = 10
        top_margin = 8
        bottom_margin = metrics.height() * 2 + 12
        available_width = max(1, rect.width() - left_margin - right_margin)
        available_height = max(1, rect.height() - top_margin - bottom_margin)
        side = max(1, min(available_width, available_height))
        x = rect.left() + left_margin + (available_width - side) // 2
        y = rect.top() + top_margin + (available_height - side) // 2
        return QRect(x, y, side, side)

    def _x_from_value(self, value: float, plot_rect: QRect) -> int:
        fraction = float(np.clip(value / self._intensity_max, 0.0, 1.0))
        return plot_rect.left() + int(round(fraction * max(plot_rect.width(), 1)))

    def _y_from_value(self, value: float, plot_rect: QRect) -> int:
        fraction = float(np.clip(value / self._intensity_max, 0.0, 1.0))
        return plot_rect.bottom() - int(round(fraction * max(plot_rect.height(), 1)))

    def _value_from_x(self, x: int, plot_rect: QRect) -> float:
        fraction = (float(x) - plot_rect.left()) / max(plot_rect.width(), 1)
        return float(np.clip(fraction, 0.0, 1.0) * self._intensity_max)

    def _value_from_y(self, y: int, plot_rect: QRect) -> float:
        fraction = (plot_rect.bottom() - float(y)) / max(plot_rect.height(), 1)
        return float(np.clip(fraction, 0.0, 1.0) * self._intensity_max)


def _event_position(event):
    if hasattr(event, "position"):
        return event.position().toPoint()
    return event.pos()


def _qcolor_from_channel_color(value, *, fallback: str) -> QColor:
    if isinstance(value, QColor):
        return QColor(value)
    rgb = color_value_to_rgb(value)
    if rgb is None:
        rgb = color_value_to_rgb(CHANNEL_COLOR_HEX.get(str(value).strip().lower()))
    if rgb is None:
        return QColor(fallback)
    values = np.clip(np.rint(np.asarray(rgb, dtype=np.float32) * 255.0), 0, 255)
    return QColor(int(values[0]), int(values[1]), int(values[2]))


class SidePanelToggleButton(QToolButton):
    """Compact glyph button for showing or hiding a side panel."""

    def __init__(self, side: str, parent=None):
        super().__init__(parent)
        self._side = side
        self._expanded = True
        self.setAutoRaise(False)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(36, 26)

    def set_expanded(self, expanded: bool) -> None:
        self._expanded = expanded
        self.update()

    def paintEvent(self, event):  # noqa: N802
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect().adjusted(1, 1, -1, -1)
        painter.setPen(QPen(QColor("#94a3b8"), 1.2))
        painter.setBrush(QColor("#27303a"))
        painter.drawRoundedRect(rect, 4, 4)

        panel_x = rect.left() + 7 if self._side == "left" else rect.right() - 16
        panel_y = rect.top() + 6
        panel_w = 11
        panel_h = 12
        panel_rect = QRect(panel_x, panel_y, panel_w, panel_h)
        painter.setPen(QPen(QColor("#cbd5e1"), 1))
        painter.setBrush(QColor("#111827"))
        painter.drawRect(panel_rect)
        strip_x = panel_rect.left() if self._side == "left" else panel_rect.right() - 3
        painter.fillRect(
            strip_x,
            panel_rect.top(),
            4,
            panel_rect.height(),
            QColor("#60a5fa"),
        )

        direction = self._direction()
        cx = rect.right() - 9 if self._side == "left" else rect.left() + 9
        cy = rect.center().y()
        painter.setPen(QPen(QColor("#e5e7eb"), 2))
        if direction < 0:
            painter.drawLine(cx + 3, cy - 5, cx - 3, cy)
            painter.drawLine(cx - 3, cy, cx + 3, cy + 5)
        else:
            painter.drawLine(cx - 3, cy - 5, cx + 3, cy)
            painter.drawLine(cx + 3, cy, cx - 3, cy + 5)
        painter.end()

    def _direction(self) -> int:
        if self._side == "left":
            return -1 if self._expanded else 1
        return 1 if self._expanded else -1


class ViewDimAxisControl(QWidget):
    """Responsive slider/spin control for one semantic viewer dimension."""

    value_changed = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._axis: ViewDimAxis | None = None
        self._syncing = False
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.label = QLabel("")
        self.label.setMinimumWidth(18)
        self.label.setStyleSheet("font-weight: 650; color: #cbd5e1;")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimumWidth(120)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(1)
        self.spin = QSpinBox()
        self.spin.setMinimumWidth(54)
        self.spin.setButtonSymbols(QSpinBox.NoButtons)
        self.range_label = QLabel("/0")
        self.range_label.setStyleSheet("color: #94a3b8;")

        layout.addWidget(self.label)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.spin)
        layout.addWidget(self.range_label)

        self.slider.valueChanged.connect(self._on_slider_value_changed)
        self.spin.valueChanged.connect(self._on_spin_value_changed)

    def set_axis(self, axis: ViewDimAxis) -> None:
        self._axis = axis
        maximum = max(int(axis.size) - 1, 0)
        with _control_signal_blockers((self.slider, self.spin)):
            self.label.setText(axis.label)
            self.label.setToolTip(f"{axis.name} axis")
            self.slider.setRange(0, maximum)
            self.slider.setValue(int(np.clip(axis.value, 0, maximum)))
            self.spin.setRange(0, maximum)
            self.spin.setValue(int(np.clip(axis.value, 0, maximum)))
            self.range_label.setText(f"/{maximum}")
        self.setToolTip(f"{axis.name} axis, {int(axis.size)} positions")

    def set_display_mode(self, mode: str) -> None:
        compact = mode == "compact"
        self.slider.setVisible(not compact)
        self.range_label.setVisible(True)

    def _on_slider_value_changed(self, value: int) -> None:
        if self._syncing or self._axis is None:
            return
        with QSignalBlocker(self.spin):
            self.spin.setValue(int(value))
        self.value_changed.emit(int(self._axis.step_axis), int(value))

    def _on_spin_value_changed(self, value: int) -> None:
        if self._syncing or self._axis is None:
            return
        with QSignalBlocker(self.slider):
            self.slider.setValue(int(value))
        self.value_changed.emit(int(self._axis.step_axis), int(value))


class ViewDimsBar(QWidget):
    """VIPP-local T/Z/C-style navigation controls synchronized to napari dims."""

    value_changed = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._axes: tuple[ViewDimAxis, ...] = ()
        self._controls: list[ViewDimAxisControl] = []
        self._responsive_mode: str | None = None

        self.setObjectName("ViewDimsBar")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setStyleSheet(
            "#ViewDimsBar {"
            " background: #20262f;"
            " border: 1px solid #374151;"
            " border-radius: 5px;"
            " padding: 2px;"
            "}"
        )

        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(6, 3, 6, 3)
        self._layout.setSpacing(6)

        self.title_label = QLabel("View dims")
        self.title_label.setStyleSheet("font-weight: 650; color: #e5e7eb;")
        self._layout.addWidget(self.title_label)

        self.menu_button = QToolButton()
        self.menu_button.setText("View dims...")
        self.menu_button.setPopupMode(QToolButton.InstantPopup)
        self.menu_button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.menu_button.setToolTip("Open full T/Z/C view sliders.")
        self.menu = QMenu(self.menu_button)
        self.menu.aboutToShow.connect(self._populate_menu)
        self.menu_button.setMenu(self.menu)
        self._layout.addStretch(1)
        self._layout.addWidget(self.menu_button)
        self.setHidden(True)

    def set_axes(self, axes: list[ViewDimAxis] | tuple[ViewDimAxis, ...]) -> None:
        self._axes = tuple(axes)
        self.setVisible(bool(self._axes))
        self._ensure_control_count(len(self._axes))
        for control, axis in zip(self._controls, self._axes, strict=False):
            control.set_axis(axis)
            control.setVisible(True)
        for control in self._controls[len(self._axes) :]:
            control.setVisible(False)
        self.sync_responsive_mode()

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self.sync_responsive_mode()

    def sync_responsive_mode(self) -> None:
        if not self._axes:
            return
        width = max(int(self.width()), 1)
        full_width = 190 + 220 * len(self._axes)
        compact_width = 160 + 86 * len(self._axes)
        if width >= full_width:
            mode = "full"
        elif width >= compact_width:
            mode = "compact"
        else:
            mode = "menu"
        if mode == self._responsive_mode:
            return
        self._responsive_mode = mode
        self.title_label.setVisible(mode != "menu")
        active_controls = self._controls[: len(self._axes)]
        for control in self._controls:
            control.set_display_mode("compact" if mode == "compact" else "full")
            control.setVisible(mode != "menu" and control in active_controls)
        self.menu_button.setVisible(mode != "full")
        self.menu_button.setText("Sliders..." if mode == "compact" else "View dims...")

    def _ensure_control_count(self, count: int) -> None:
        while len(self._controls) < count:
            control = ViewDimAxisControl(self)
            control.value_changed.connect(self.value_changed.emit)
            self._controls.append(control)
            self._layout.insertWidget(max(self._layout.count() - 2, 1), control, 1)

    def _populate_menu(self) -> None:
        self.menu.clear()
        if not self._axes:
            action = QAction("No view dimensions", self.menu)
            action.setEnabled(False)
            self.menu.addAction(action)
            return
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        for axis in self._axes:
            control = ViewDimAxisControl(container)
            control.set_axis(axis)
            control.set_display_mode("full")
            control.value_changed.connect(self.value_changed.emit)
            layout.addWidget(control)
        action = QWidgetAction(self.menu)
        action.setDefaultWidget(container)
        self.menu.addAction(action)


class TunnelManagerDialog(QDialog):
    """Small non-modal panel for auditing and managing named tunnels."""

    tunnelSelected = Signal(str)
    focusSourceRequested = Signal(str)
    revealRequested = Signal(str)
    renameRequested = Signal(str)
    deleteRequested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: tuple[TunnelSummary, ...] = ()
        self.setWindowTitle("VIPP Tunnels")
        self.resize(740, 320)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter tunnels, sources, or subscribers")
        layout.addWidget(self.filter_edit)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Tunnel", "Source", "Type", "Subscribers", "Subscriber nodes"],
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table, 1)

        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.setSpacing(6)
        self.focus_button = QPushButton("Focus source")
        self.reveal_button = QPushButton("Reveal subscribers")
        self.rename_button = QPushButton("Rename...")
        self.delete_button = QPushButton("Delete")
        self.delete_button.setToolTip("Remove the tunnel and clear its subscribers.")
        button_row.addWidget(self.focus_button)
        button_row.addWidget(self.reveal_button)
        button_row.addStretch(1)
        button_row.addWidget(self.rename_button)
        button_row.addWidget(self.delete_button)
        layout.addLayout(button_row)

        close_buttons = QDialogButtonBox(QDialogButtonBox.Close)
        close_buttons.rejected.connect(self.close)
        layout.addWidget(close_buttons)

        self.filter_edit.textChanged.connect(self._on_filter_changed)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        self.table.itemDoubleClicked.connect(lambda _item: self._emit_reveal())
        self.focus_button.clicked.connect(self._emit_focus)
        self.reveal_button.clicked.connect(self._emit_reveal)
        self.rename_button.clicked.connect(self._emit_rename)
        self.delete_button.clicked.connect(self._emit_delete)
        self._sync_button_state()

    def set_tunnels(self, rows: tuple[TunnelSummary, ...]) -> None:
        selected = self.selected_tunnel_name()
        self._rows = tuple(rows)
        self._populate_table(selected)

    def _populate_table(self, selected: str = "") -> None:
        rows = self._filtered_rows()
        self.table.setRowCount(len(rows))
        for row, summary in enumerate(rows):
            subscribers = ", ".join(
                f"{title} input {port + 1}"
                for _node_id, title, port in summary.subscribers
            )
            values = [
                summary.name,
                f"{summary.source_title} output {summary.source_port + 1}",
                summary.output_type,
                str(summary.subscriber_count),
                subscribers or "none",
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.UserRole, summary.name)
                self.table.setItem(row, column, item)
        self.table.resizeColumnsToContents()
        self._restore_selection(selected)
        self._sync_button_state()

    def _filtered_rows(self) -> tuple[TunnelSummary, ...]:
        query = _normalize_search_text(self.filter_edit.text())
        if not query:
            return self._rows
        matches: list[TunnelSummary] = []
        for summary in self._rows:
            subscriber_text = " ".join(
                title for _node_id, title, _port in summary.subscribers
            )
            haystack = _normalize_search_text(
                f"{summary.name} {summary.source_title} {summary.output_type} "
                f"{subscriber_text}"
            )
            if _fuzzy_match(query, haystack):
                matches.append(summary)
        return tuple(matches)

    def selected_tunnel_name(self) -> str:
        selected = self.table.selectedItems()
        if not selected:
            return ""
        return str(selected[0].data(Qt.UserRole) or "")

    def select_tunnel(self, name: str) -> None:
        self._restore_selection(str(name or "").strip())
        self._sync_button_state()

    def _restore_selection(self, name: str) -> None:
        if self.table.rowCount() == 0:
            self.table.clearSelection()
            return
        target_row = 0 if not name else -1
        if name:
            for row in range(self.table.rowCount()):
                item = self.table.item(row, 0)
                if item is not None and item.data(Qt.UserRole) == name:
                    target_row = row
                    break
        if target_row < 0:
            self.table.clearSelection()
            return
        self.table.selectRow(target_row)

    def _on_filter_changed(self) -> None:
        self._populate_table(self.selected_tunnel_name())

    def _on_selection_changed(self) -> None:
        self._sync_button_state()
        name = self.selected_tunnel_name()
        if name:
            self.tunnelSelected.emit(name)

    def _sync_button_state(self) -> None:
        has_selection = bool(self.selected_tunnel_name())
        for button in (
            self.focus_button,
            self.reveal_button,
            self.rename_button,
            self.delete_button,
        ):
            button.setEnabled(has_selection)

    def _emit_focus(self) -> None:
        name = self.selected_tunnel_name()
        if name:
            self.focusSourceRequested.emit(name)

    def _emit_reveal(self) -> None:
        name = self.selected_tunnel_name()
        if name:
            self.revealRequested.emit(name)

    def _emit_rename(self) -> None:
        name = self.selected_tunnel_name()
        if name:
            self.renameRequested.emit(name)

    def _emit_delete(self) -> None:
        name = self.selected_tunnel_name()
        if name:
            self.deleteRequested.emit(name)


class VippWidget(QWidget):
    """Visual node workflow composer hosted inside napari."""

    HISTORY_LIMIT = 80
    INSERT_GAP_PADDING_X = 70.0
    INSERT_GAP_PADDING_Y = 55.0
    TOOLBAR_HIDE_CHECKBOXES_WIDTH = 1700
    TOOLBAR_HIDE_DROPDOWNS_WIDTH = 1500
    TOOLBAR_HIDE_ZOOM_WIDTH = 1050

    def __init__(self, viewer: napari.viewer.Viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.pipeline = PrototypePipeline()
        self._selected_node_id = "gaussian"
        self._active_pinned_node_id: str | None = None
        self._inspect_layer_name = "VIPP Inspect"
        self._preview_disabled_node_ids: set[str] = set()
        self._hidden_input_layer_states: dict[int, tuple[object, bool]] = {}
        self._sample_payload_cache: dict[str, SourcePayload] | None = None
        self._source_inspection_cache: dict[str, tuple[int, SourceInspection]] = {}
        self._file_source_payload_cache: dict[tuple[object, ...], SourcePayload] = {}
        self._interactive_collection_source_paths: dict[str, Path] = {}
        self._active_source_load_id: int | None = None
        self._source_load_serial = 0
        self._source_load_pending = False
        self._dock_chrome_configured = False
        self._dock_window_behavior_configured = False
        self._initial_dock_size_applied = False
        self._undo_stack: list[WorkflowHistorySnapshot] = []
        self._redo_stack: list[WorkflowHistorySnapshot] = []
        self._restoring_history = False
        self._pending_parameter_undo_key: tuple[str, str] | None = None
        self._rescale_auto_output_ranges: dict[str, tuple[float, float]] = {}
        self._code_dialogs: list[QDialog] = []
        self._pending_dirty_node_ids: set[str] = set()
        self._pending_manual_node_ids: set[str] = set()
        self._inflight_dirty_node_ids: set[str] | None = None
        self._last_pipeline_source_signature: tuple | None = None
        self._toolbar_compact_stage: tuple[bool, bool, bool] | None = None
        self._toolbar_checkbox_widgets: list[QWidget] = []
        self._toolbar_dropdown_widgets: list[QWidget] = []
        self._toolbar_zoom_widgets: list[QWidget] = []
        self._toolbar_settings_widgets: list[QWidget] = []
        self._recent_cache_node_ids: list[str] = []
        self._thumbnail_contrast_limit_cache: dict[tuple, object] = {}
        self._queued_thumbnail_contrast_limit_requests: dict[
            tuple,
            ThumbnailContrastLimitRequest,
        ] = {}
        self._pending_thumbnail_contrast_limit_keys: set[tuple] = set()
        self._active_thumbnail_contrast_run_id: int | None = None
        self._thumbnail_contrast_serial = 0
        self._thumbnail_contrast_busy_visible = False
        self._input_histogram_serial = 0
        self._active_input_histogram_run_id: int | None = None
        self._active_input_histogram_key: tuple | None = None
        self._active_input_histogram_cancel_event: threading.Event | None = None
        self._pending_input_histogram_request: InputHistogramRequest | None = None
        self._current_input_histogram_key: tuple | None = None
        self._input_histogram_cache: dict[tuple, InputHistogramResult] = {}
        self._input_histogram_distribution_cache: dict[
            tuple,
            InputHistogramDistribution,
        ] = {}
        self._label_volume_cache: dict[
            tuple,
            tuple[weakref.ReferenceType, np.ndarray],
        ] = {}
        self._output_histogram_serial = 0
        self._active_output_histogram_run_id: int | None = None
        self._active_output_histogram_key: tuple | None = None
        self._pending_output_histogram_request: InputHistogramRequest | None = None
        self._current_output_histogram_key: tuple | None = None
        self._output_histogram_cache: dict[tuple, InputHistogramResult] = {}
        self._colocalization_scatter_serial = 0
        self._active_colocalization_scatter_run_id: int | None = None
        self._active_colocalization_scatter_key: tuple | None = None
        self._active_colocalization_scatter_cancel_event: (
            threading.Event | None
        ) = None
        self._pending_colocalization_scatter_request: (
            ColocalizationScatterRequest | None
        ) = None
        self._current_colocalization_scatter_key: tuple | None = None
        self._colocalization_scatter_cache: dict[
            tuple,
            ColocalizationScatterResult,
        ] = {}
        self._auto_contrast_serial = 0
        self._active_auto_contrast_run_id: int | None = None
        self._active_auto_contrast_key: tuple | None = None
        self._auto_contrast_busy_visible = False
        self._generated_layer_contrast_generation = 0
        self._generated_layer_contrast_cache: dict[
            tuple,
            tuple[weakref.ReferenceType, tuple[float, float]],
        ] = {}
        self._generated_layer_contrast_pending: set[tuple] = set()
        self._generated_layer_contrast_keys: dict[str, tuple] = {}
        self._syncing_view_dims_bar = False
        self._vipp_current_step: tuple[int, ...] | None = None
        self._vipp_current_nsteps: tuple[int, ...] | None = None
        self._tunnel_manager_dialog: TunnelManagerDialog | None = None
        self._graph_notes: dict[str, GraphNoteState] = {}
        self._graph_search_matches: tuple[GraphSearchMatch, ...] = ()
        self._graph_search_index = -1
        self._graph_search_highlighted_tunnel = ""
        self.setMinimumSize(0, 0)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)

        self.save_thumbnail_visibility_checkbox = QCheckBox(
            "Save thumbnail visibility in workflows"
        )
        self.save_thumbnail_visibility_checkbox.setChecked(False)
        self.save_thumbnail_visibility_checkbox.setToolTip(
            "Store per-node thumbnail preview visibility in saved workflow JSON. "
            "Thumbnail image pixels are never saved."
        )
        self.preview_mode_combo = QComboBox()
        self.preview_mode_combo.addItems(["Slice", "MIP", "Off"])
        self.thumbnail_contrast_combo = QComboBox()
        self.thumbnail_contrast_combo.addItems(THUMBNAIL_CONTRAST_MODES)
        self.thumbnail_scope_combo = QComboBox()
        self.thumbnail_scope_combo.addItems(THUMBNAIL_CONTRAST_SCOPES)
        self.thumbnail_colormap_combo = QComboBox()
        self.thumbnail_colormap_combo.addItems(MONOCHROME_COLORMAPS)
        self.follow_dims_checkbox = QCheckBox("Link napari/VIPP sliders")
        self.follow_dims_checkbox.setChecked(True)
        self.follow_dims_checkbox.setToolTip(
            "Keep napari dimension sliders and VIPP thumbnail sliders linked. "
            "When disabled, napari scrubbing updates only the viewer; VIPP sliders "
            "control workflow thumbnails and inspector summaries."
        )
        self.graph_zoom_slider = QSlider(Qt.Horizontal)
        self.graph_zoom_slider.setRange(
            PipelineGraphView.SLIDER_MIN_ZOOM,
            PipelineGraphView.SLIDER_MAX_ZOOM,
        )
        self.graph_zoom_slider.setValue(PipelineGraphView.DEFAULT_ZOOM)
        self.graph_zoom_slider.setSingleStep(5)
        self.graph_zoom_slider.setPageStep(20)
        self.graph_zoom_slider.setTickInterval(20)
        self.graph_zoom_slider.setTickPosition(QSlider.TicksBelow)
        self.graph_zoom_slider.setFixedWidth(120)
        self.graph_zoom_slider.setToolTip(
            "Scale graph cards from 40% to 250%. 100% is the calibrated default. "
            "Ctrl/trackpad wheel zoom can go beyond this slider range."
        )
        self.graph_zoom_label = QLabel("100%")
        self.graph_zoom_label.setMinimumWidth(44)
        self.graph_zoom_reset_button = QToolButton()
        self.graph_zoom_reset_button.setIcon(_toolbar_icon("reset"))
        self.graph_zoom_reset_button.setIconSize(QSize(18, 18))
        self.graph_zoom_reset_button.setFixedSize(24, 24)
        self.graph_zoom_reset_button.setToolTip("Reset graph zoom to the default 100%.")

        self.new_workflow_button = QPushButton("New workflow...")
        self.tunnel_manager_button = QPushButton("Tunnels...")
        self.tunnel_manager_button.setToolTip(
            "Manage named graph tunnels and reveal their subscribers.",
        )
        self.auto_structure_button = QPushButton("Auto structure graph")
        self.auto_structure_button.setToolTip(
            "One-shot source-to-sink layout cleanup. Undo restores old positions."
        )
        self.refresh_button = QPushButton("Refresh")
        self.calculate_all_button = QPushButton("Calculate all")
        self.calculate_all_button.setToolTip(
            "Calculate every manual node that is not current."
        )
        self.undo_action = QAction(
            _toolbar_icon("undo"),
            "Undo",
            self,
        )
        self.undo_action.setShortcuts(QKeySequence.keyBindings(QKeySequence.Undo))
        self.undo_action.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self.undo_action.setToolTip("Undo last workflow edit")
        self.redo_action = QAction(
            _toolbar_icon("redo"),
            "Redo",
            self,
        )
        self.redo_action.setShortcuts(QKeySequence.keyBindings(QKeySequence.Redo))
        self.redo_action.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self.redo_action.setToolTip("Redo last undone workflow edit")
        self.undo_button = QToolButton()
        self.undo_button.setDefaultAction(self.undo_action)
        self.undo_button.setIconSize(QSize(18, 18))
        self.undo_button.setFixedSize(24, 24)
        self.redo_button = QToolButton()
        self.redo_button.setDefaultAction(self.redo_action)
        self.redo_button.setIconSize(QSize(18, 18))
        self.redo_button.setFixedSize(24, 24)
        self.addAction(self.undo_action)
        self.addAction(self.redo_action)
        self.save_workflow_button = QPushButton("Save workflow...")
        self.open_example_button = QPushButton("Open example...")
        self.open_example_button.setToolTip(
            "Open a bundled example workflow with its sample Image Source nodes."
        )
        self.load_workflow_button = QPushButton("Load workflow...")
        self.export_button = QPushButton("Export Python...")
        self.batch_button = QPushButton("Run batch...")
        self.batch_button.setToolTip(
            "Run the current workflow over a folder of image files."
        )
        self.export_ome_button = QPushButton("Export OME dataset...")
        self.settings_menu_button = QToolButton()
        self.settings_menu_button.setText("Settings")
        self.settings_menu_button.setMinimumWidth(96)
        self.settings_menu_button.setPopupMode(QToolButton.InstantPopup)
        self.settings_menu_button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.settings_menu_button.setStyleSheet(
            "QToolButton { padding: 3px 18px 3px 8px; }"
            "QToolButton::menu-indicator {"
            " subcontrol-origin: padding;"
            " subcontrol-position: right center;"
            " right: 4px;"
            " width: 10px;"
            "}"
        )
        self.settings_menu_button.setToolTip(
            "Collapsed thumbnail, preview, contrast, colormap, and zoom controls."
        )
        self.settings_menu = QMenu(self.settings_menu_button)
        self.settings_menu.aboutToShow.connect(self._populate_settings_toolbar_menu)
        self.settings_menu_button.setMenu(self.settings_menu)
        self.cache_mode_combo = QComboBox()
        self.cache_mode_combo.addItems(CACHE_MODE_CHOICES)
        self.cache_mode_combo.setCurrentText(CACHE_MODE_KEEP_ALL)
        self.cache_mode_combo.setToolTip(
            "Choose how aggressively VIPP keeps calculated node outputs in memory."
        )
        self.memory_guard_checkbox = QCheckBox("Auto memory guard")
        self.memory_guard_checkbox.setChecked(True)
        self.memory_guard_checkbox.setToolTip(
            "When keep-all caching uses too much of free-or-reclaimable RAM, "
            "switch to Smart interactive cache and prune optional outputs."
        )
        self.memory_limit_spin = QSpinBox()
        self.memory_limit_spin.setRange(5, 95)
        self.memory_limit_spin.setValue(DEFAULT_CACHE_MEMORY_LIMIT_PERCENT)
        self.memory_limit_spin.setSuffix("%")
        self.memory_limit_spin.setToolTip(
            "Maximum share of reclaimable memory that VIPP keep-all cache may "
            "occupy. Reclaimable memory is free RAM plus the current VIPP cache."
        )
        self._memory_guard_dialog_shown = False
        self.background_all_checkbox = QCheckBox("Run all in BG")
        self.background_all_checkbox.setChecked(False)
        self.background_all_checkbox.setToolTip(
            "Run all pipeline updates in background. "
            "When off, VIPP still backgrounds known-slower operations and "
            "updates that process large images."
        )
        self.view_dims_bar = ViewDimsBar()
        self.pipeline_busy_label = QLabel("Processing")
        self.pipeline_busy_label.setStyleSheet("color: #93c5fd; font-weight: 650;")
        self.pipeline_busy_bar = QProgressBar()
        self.pipeline_busy_bar.setRange(0, 0)
        self.pipeline_busy_bar.setTextVisible(False)
        self.pipeline_busy_bar.setFixedWidth(96)
        self.pipeline_busy_bar.setFixedHeight(12)
        self.pipeline_cancel_button = QPushButton("Cancel")
        self.pipeline_cancel_button.setToolTip(
            "Cancel queued background reruns and ignore the active result. "
            "The operation already running in a worker may finish in the background."
        )
        self.pipeline_cancel_button.setVisible(False)
        self.pipeline_busy_label.setVisible(False)
        self.pipeline_busy_bar.setVisible(False)
        self.cache_status_label = QLabel("Cache: --")
        self.cache_status_label.setStyleSheet(
            "color: #94a3b8; font-size: 11px; padding: 2px 4px;"
        )
        self.cache_status_label.setToolTip("Estimated VIPP cache and system memory.")
        self.version_label = QLabel(f"VIPP {VIPP_VERSION}")
        self.version_label.setStyleSheet(
            "color: #94a3b8; font-size: 11px; font-weight: 600;"
            "padding: 2px 8px; border: 1px solid #334155;"
            "border-radius: 999px; background: #1f2937;"
        )
        self.version_label.setToolTip(f"napari-vipp {VIPP_VERSION}")
        self.status_label = QLabel(
            "Select an Image Source node to choose data for the workflow."
        )
        self.status_label.setWordWrap(True)

        self.palette_search = QLineEdit()
        self.palette_search.setPlaceholderText("Search nodes")
        self.palette_search.setClearButtonEnabled(True)
        self.graph_search_edit = QLineEdit()
        self.graph_search_edit.setPlaceholderText("Search graph")
        self.graph_search_edit.setClearButtonEnabled(True)
        self.graph_search_edit.setToolTip(
            "Search node titles, operation IDs, tunnel names, and output tags."
        )
        self.graph_search_edit.setMaximumWidth(260)
        self.graph_search_focus_button = QPushButton("Focus")
        self.graph_search_focus_button.setToolTip(
            "Focus the next graph search match."
        )
        self.graph_search_focus_button.setEnabled(False)
        self.graph_search_status = QLabel("")
        self.graph_search_status.setMinimumWidth(72)
        self.graph_search_status.setStyleSheet("color: #94a3b8; font-size: 11px;")
        self.palette = NodePalette(grouped_palette_specs())
        self.palette.setMinimumWidth(190)
        self.palette.setMinimumHeight(0)
        self.palette_panel = self._build_palette_panel()
        self.graph_view = PipelineGraphView()
        self.graph_view.set_connection_insert_validator(
            self._connection_insert_preview_state
        )
        self.graph_view.setMinimumHeight(80)
        self.graph_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        self.left_panel_toggle = SidePanelToggleButton("left")
        self.left_panel_toggle.setObjectName("LeftPanelToggle")
        self.right_panel_toggle = SidePanelToggleButton("right")
        self.right_panel_toggle.setObjectName("RightPanelToggle")
        self._default_splitter_sizes = [210, 850, 260]
        self._left_panel_last_width = self._default_splitter_sizes[0]
        self._right_panel_last_width = self._default_splitter_sizes[2]
        self._pipeline_thread_pool = QThreadPool(self)
        self._pipeline_thread_pool.setMaxThreadCount(1)
        self._pipeline_run_serial = 0
        self._active_pipeline_run_id: int | None = None
        self._active_pipeline_node_id: str | None = None
        self._pipeline_run_pending = False
        self._pipeline_run_context: dict[int, tuple] = {}
        self._pipeline_run_manual_node_ids: dict[int, frozenset[str]] = {}
        self._pipeline_cancel_events: dict[int, threading.Event] = {}

        self.selected_title = QLabel("Gaussian Blur")
        self.selected_title.setStyleSheet("font-weight: 650;")
        self.thumbnail_checkbox = QCheckBox("Show thumbnail preview")
        self.thumbnail_checkbox.setChecked(True)
        self.keep_cached_checkbox = QCheckBox("Keep output cached")
        self.keep_cached_checkbox.setToolTip(
            "Retain this node output in Smart and Low-memory cache modes. "
            "Use this for expensive intermediate images or tables you inspect often."
        )
        self.execution_group = QGroupBox("Execution")
        self.execution_status_label = QLabel("Automatic")
        self.execution_status_label.setWordWrap(True)
        self.execution_status_label.setMinimumHeight(34)
        self.auto_recalculate_checkbox = QCheckBox("Auto Recalculate")
        self.auto_recalculate_notice = QLabel(
            "Auto Recalculate runs this node after upstream or parameter changes. "
            "This can be slow on large images."
        )
        self.auto_recalculate_notice.setWordWrap(True)
        self.auto_recalculate_notice.setStyleSheet("color: #f59e0b;")
        self.calculate_button = QPushButton("Calculate")
        self.parameter_group = QGroupBox("Parameters")
        self.parameter_form = QFormLayout(self.parameter_group)
        self._parameter_widgets: dict[str, QWidget] = {}
        self.auto_contrast_group = QGroupBox("Auto Contrast")
        self.auto_saturation_control = ParameterControl(
            AUTO_CONTRAST_SATURATION_SPEC,
            AUTO_CONTRAST_SATURATION_SPEC.default,
            ParameterBounds(
                AUTO_CONTRAST_SATURATION_SPEC.minimum,
                AUTO_CONTRAST_SATURATION_SPEC.maximum,
                AUTO_CONTRAST_SATURATION_SPEC.step,
                AUTO_CONTRAST_SATURATION_SPEC.decimals,
            ),
        )
        self.auto_contrast_button = QPushButton("Auto")
        self.auto_contrast_button.setToolTip(
            "Set scale and offset from exact full-input finite percentiles. "
            "RGB and RGBA inputs use weighted RGB luminance; alpha is ignored. "
            "Large inputs are calculated in the background."
        )
        self.metadata_group = QGroupBox("Output Metadata")
        self.table_group = QGroupBox("Table Preview")
        self.table_summary = QLabel("No table output.")
        self.table_summary.setWordWrap(True)
        self.table_preview = QTableWidget(0, 0)
        self.table_preview.verticalHeader().setVisible(False)
        self.table_preview.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_preview.setSelectionMode(QAbstractItemView.NoSelection)
        self.table_preview.setFocusPolicy(Qt.NoFocus)
        self.table_preview.setMinimumHeight(180)
        self.table_preview.setStyleSheet(
            "QTableWidget { background: #1f242c; color: #e5e7eb; "
            "gridline-color: #374151; }"
            "QHeaderView::section { background: #2b313b; color: #f3f4f6; "
            "padding: 4px; }"
        )
        self.table_group.setHidden(True)
        self.metadata_table = QTableWidget(0, 2)
        self.metadata_table.setHorizontalHeaderLabels(["Field", "Value"])
        self.metadata_table.verticalHeader().setVisible(False)
        self.metadata_table.horizontalHeader().setStretchLastSection(True)
        self.metadata_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.metadata_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.metadata_table.setFocusPolicy(Qt.NoFocus)
        self.metadata_table.setWordWrap(True)
        self.metadata_table.setMinimumHeight(260)
        self.metadata_table.setStyleSheet(
            "QTableWidget { background: #1f242c; color: #e5e7eb; "
            "gridline-color: #374151; }"
            "QHeaderView::section { background: #2b313b; color: #f3f4f6; "
            "padding: 4px; }"
        )
        self.history_title = QLabel("History")
        self.history_title.setStyleSheet("font-weight: 650;")
        self.history_label = QLabel("No history yet.")
        self.history_label.setWordWrap(True)
        self.history_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.histogram_group = QGroupBox("Output Histogram")
        self.histogram_scope_combo = QComboBox()
        self.histogram_scope_combo.addItems(["Slice", "Stack"])
        self.histogram_log_checkbox = QCheckBox("Log scale")
        self.histogram_plot = HistogramPlot()
        self.rescale_input_histogram_group = QGroupBox("Input Histogram")
        self.rescale_input_histogram_scope_combo = QComboBox()
        self.rescale_input_histogram_scope_combo.addItems(
            ["Slice histogram", "Stack histogram"]
        )
        self.rescale_input_histogram_log_checkbox = QCheckBox("Log scale")
        self.rescale_input_histogram_plot = HistogramPlot()
        self.rescale_input_histogram_group.setHidden(True)
        self.label_volume_group = QGroupBox("Label Volume Distribution")
        self.label_volume_summary = QLabel("No labeled objects.")
        self.label_volume_summary.setWordWrap(True)
        self.label_volume_log_checkbox = QCheckBox("Log volume axis")
        self.label_volume_log_checkbox.setChecked(True)
        self.label_volume_plot = HistogramPlot()
        self.label_volume_group.setHidden(True)
        self.colocalization_scatter_group = QGroupBox("Colocalization Scatter")
        self.colocalization_scatter_summary = QLabel(
            "Connect two channel inputs."
        )
        self.colocalization_scatter_summary.setWordWrap(True)
        self.colocalization_scatter_summary.setMinimumHeight(42)
        self.colocalization_scatter_colormap_combo = QComboBox()
        self.colocalization_scatter_colormap_combo.addItems(
            COLOCALIZATION_SCATTER_COLORMAPS
        )
        self.colocalization_scatter_log_checkbox = QCheckBox("Log count scale")
        self.colocalization_scatter_log_checkbox.setChecked(True)
        self.colocalization_scatter_plot = ColocalizationScatterPlot()
        self.colocalization_scatter_group.setHidden(True)

        self.pin_button = QPushButton("Pin selected")
        self.save_button = QPushButton("Save selected output...")
        self.inspector_panel = self._build_inspector()

        self._debounce_timer = QTimer(self)
        self._debounce_timer.setInterval(150)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.timeout.connect(self.run_pipeline)

        self._build_layout()
        self._connect_signals()
        self._sync_toolbar_responsive_mode()
        self._autobind_default_image_sources()
        self._build_graph_from_pipeline()
        self._select_node(self._selected_node_id)
        self.run_pipeline()
        self._sync_history_actions()

    def closeEvent(self, event):  # noqa: N802
        self._restore_hidden_input_layers()
        super().closeEvent(event)

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self._sync_toolbar_responsive_mode()
        self.view_dims_bar.sync_responsive_mode()

    def eventFilter(self, watched, event):  # noqa: N802
        dock = self._dock_widget()
        if (
            watched is dock
            and event.type() == QEvent.NonClientAreaMouseButtonDblClick
            and dock.isFloating()
        ):
            if dock.isMaximized():
                dock.showNormal()
            else:
                dock.showMaximized()
            event.accept()
            return True
        return super().eventFilter(watched, event)

    def showEvent(self, event):  # noqa: N802
        super().showEvent(event)
        self._sync_toolbar_responsive_mode()
        if not self._dock_chrome_configured:
            QTimer.singleShot(0, self._ensure_dock_widget_chrome)
        if not self._initial_dock_size_applied:
            QTimer.singleShot(80, self._apply_initial_dock_size)

    def minimumSizeHint(self):  # noqa: N802
        return QSize(420, 120)

    def sizeHint(self):  # noqa: N802
        return QSize(1180, 560)

    def _ensure_dock_widget_chrome(self) -> None:
        dock = self._dock_widget()
        if dock is None or self._dock_chrome_configured:
            return
        try:
            if not self._dock_window_behavior_configured:
                dock.installEventFilter(self)
                dock.topLevelChanged.connect(self._on_dock_top_level_changed)
                dock.visibilityChanged.connect(self._on_dock_visibility_changed)
                self._dock_window_behavior_configured = True
            desired_features = (
                QDockWidget.DockWidgetClosable
                | QDockWidget.DockWidgetMovable
                | QDockWidget.DockWidgetFloatable
            )
            if dock.windowTitle() != "VIPP Workflow":
                dock.setWindowTitle("VIPP Workflow")
            if not dock.isFloating() and dock.titleBarWidget() is not None:
                dock.setTitleBarWidget(None)
            if dock.features() != desired_features:
                dock.setFeatures(desired_features)
            if dock.allowedAreas() != Qt.AllDockWidgetAreas:
                dock.setAllowedAreas(Qt.AllDockWidgetAreas)
            self._dock_chrome_configured = True
            if dock.isFloating():
                QTimer.singleShot(0, self._configure_floating_dock_window)
        except Exception:
            pass

    def _on_dock_top_level_changed(self, floating: bool) -> None:
        if floating:
            QTimer.singleShot(0, self._configure_floating_dock_window)
        else:
            QTimer.singleShot(0, self._restore_docked_title_bar)

    def _on_dock_visibility_changed(self, visible: bool) -> None:
        if visible:
            QTimer.singleShot(0, self._configure_floating_dock_window)

    def _configure_floating_dock_window(self) -> None:
        dock = self._dock_widget()
        if dock is None or not dock.isFloating():
            return
        try:
            if dock.titleBarWidget() is not None:
                dock.setTitleBarWidget(None)

            flags = dock.windowFlags()
            desired_flags = (flags & ~Qt.WindowType_Mask) | Qt.Window
            desired_flags &= ~Qt.FramelessWindowHint
            desired_flags |= (
                Qt.WindowTitleHint
                | Qt.WindowSystemMenuHint
                | Qt.WindowMinimizeButtonHint
                | Qt.WindowMaximizeButtonHint
                | Qt.WindowCloseButtonHint
            )
            if desired_flags == flags:
                return

            geometry = dock.geometry()
            was_visible = dock.isVisible()
            was_maximized = dock.isMaximized()
            with QSignalBlocker(dock):
                dock.setWindowFlags(desired_flags)
                dock.setGeometry(geometry)
                if was_visible:
                    if was_maximized:
                        dock.showMaximized()
                    else:
                        dock.show()
        except Exception:
            pass

    def _restore_docked_title_bar(self) -> None:
        dock = self._dock_widget()
        if dock is None or dock.isFloating():
            return
        try:
            if dock.titleBarWidget() is not None:
                dock.setTitleBarWidget(None)
        except Exception:
            pass

    def _apply_initial_dock_size(self) -> None:
        if self._initial_dock_size_applied:
            return
        dock = self._dock_widget()
        if dock is None:
            return
        try:
            if dock.isFloating():
                return
        except Exception:
            return

        window = self._dock_main_window(dock)
        if window is None:
            return

        target_height = 380
        target_width = 760
        try:
            area = window.dockWidgetArea(dock)
            if area in (Qt.BottomDockWidgetArea, Qt.TopDockWidgetArea):
                window.resizeDocks([dock], [target_height], Qt.Vertical)
            elif area in (Qt.LeftDockWidgetArea, Qt.RightDockWidgetArea):
                window.resizeDocks([dock], [target_width], Qt.Horizontal)
            else:
                return
            self._initial_dock_size_applied = True
        except Exception:
            pass

    def _dock_widget(self):
        parent = self.parentWidget()
        while parent is not None:
            if isinstance(parent, QDockWidget):
                return parent
            parent = parent.parentWidget()
        return None

    def _dock_main_window(self, dock: QDockWidget) -> QMainWindow | None:
        parent = dock.parentWidget()
        while parent is not None:
            if isinstance(parent, QMainWindow):
                return parent
            parent = parent.parentWidget()
        return None

    def _build_layout(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        input_row = QHBoxLayout()
        input_row.setContentsMargins(0, 0, 0, 0)
        input_row.setSpacing(4)
        preview_label = QLabel("Preview")
        input_row.addWidget(preview_label)
        input_row.addWidget(self.preview_mode_combo)
        contrast_label = QLabel("Contrast")
        input_row.addWidget(contrast_label)
        input_row.addWidget(self.thumbnail_contrast_combo)
        scope_label = QLabel("Contrast Range")
        input_row.addWidget(scope_label)
        input_row.addWidget(self.thumbnail_scope_combo)
        mono_label = QLabel("Mono")
        input_row.addWidget(mono_label)
        input_row.addWidget(self.thumbnail_colormap_combo)
        zoom_separator = _toolbar_separator()
        input_row.addWidget(zoom_separator)
        zoom_label = QLabel("Zoom")
        input_row.addWidget(zoom_label)
        input_row.addWidget(self.graph_zoom_slider)
        input_row.addWidget(self.graph_zoom_reset_button)
        input_row.addWidget(self.graph_zoom_label)
        action_separator = _toolbar_separator()
        input_row.addWidget(action_separator)
        input_row.addWidget(self.refresh_button)
        input_row.addWidget(self.calculate_all_button)
        input_row.addWidget(self.auto_structure_button)
        input_row.addWidget(self.tunnel_manager_button)
        input_row.addWidget(self.undo_button)
        input_row.addWidget(self.redo_button)
        compact_separator = _toolbar_separator(6)
        input_row.addWidget(compact_separator)
        input_row.addWidget(self.settings_menu_button)
        root.addLayout(input_row)
        self._toolbar_checkbox_widgets = []
        for widget in (
            self.background_all_checkbox,
            self.follow_dims_checkbox,
        ):
            widget.setVisible(False)
        self._toolbar_dropdown_widgets = [
            preview_label,
            self.preview_mode_combo,
            contrast_label,
            self.thumbnail_contrast_combo,
            scope_label,
            self.thumbnail_scope_combo,
            mono_label,
            self.thumbnail_colormap_combo,
        ]
        self._toolbar_zoom_widgets = [
            zoom_separator,
            zoom_label,
            self.graph_zoom_slider,
            self.graph_zoom_reset_button,
            self.graph_zoom_label,
        ]
        self._toolbar_settings_widgets = [
            compact_separator,
            self.settings_menu_button,
        ]

        workflow_row = QHBoxLayout()
        workflow_row.setContentsMargins(0, 0, 0, 0)
        workflow_row.setSpacing(4)
        workflow_row.addWidget(self.new_workflow_button)
        workflow_row.addWidget(self.open_example_button)
        workflow_row.addWidget(self.save_workflow_button)
        workflow_row.addWidget(self.load_workflow_button)
        workflow_separator = _toolbar_separator()
        workflow_row.addWidget(workflow_separator)
        workflow_row.addWidget(self.export_button)
        workflow_row.addWidget(self.batch_button)
        workflow_row.addWidget(self.export_ome_button)
        export_separator = _toolbar_separator()
        workflow_row.addWidget(export_separator)
        workflow_row.addStretch(1)
        workflow_row.addWidget(self.pipeline_busy_label)
        workflow_row.addWidget(self.pipeline_busy_bar)
        workflow_row.addWidget(self.pipeline_cancel_button)
        workflow_row.addWidget(self.cache_status_label)
        workflow_row.addWidget(self.version_label)
        root.addLayout(workflow_row)
        root.addWidget(self.view_dims_bar)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.palette_panel)
        self.splitter.addWidget(self._build_graph_panel())
        self.splitter.addWidget(self.inspector_panel)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 5)
        self.splitter.setStretchFactor(2, 1)
        self.splitter.setSizes(self._default_splitter_sizes)
        self.splitter.setMinimumHeight(0)
        self.splitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        root.addWidget(self.splitter, 1)
        root.addWidget(self.status_label)
        self._sync_side_panel_toggles()

    def _sync_toolbar_responsive_mode(self) -> None:
        width = int(self.width())
        hide_dropdowns = 0 < width < self.TOOLBAR_HIDE_DROPDOWNS_WIDTH
        hide_zoom = 0 < width < self.TOOLBAR_HIDE_ZOOM_WIDTH
        hide_checkboxes = True
        stage = (hide_checkboxes, hide_dropdowns, hide_zoom)
        if stage == self._toolbar_compact_stage:
            return
        self._toolbar_compact_stage = stage
        for widget in self._toolbar_checkbox_widgets:
            widget.setVisible(not hide_checkboxes)
        for widget in self._toolbar_dropdown_widgets:
            widget.setVisible(not hide_dropdowns)
        for widget in self._toolbar_zoom_widgets:
            widget.setVisible(not hide_zoom)
        for widget in self._toolbar_settings_widgets:
            widget.setVisible(True)
        self.auto_structure_button.setText(
            "Structure" if hide_dropdowns or hide_zoom else "Auto structure graph"
        )

    def _populate_settings_toolbar_menu(self) -> None:
        menu = self.settings_menu
        menu.clear()
        hide_checkboxes, hide_dropdowns, hide_zoom = self._toolbar_compact_stage or (
            False,
            False,
            False,
        )
        added_section = False
        self._add_checkbox_menu_action(
            menu,
            "Save thumbnail visibility in workflows",
            self.save_thumbnail_visibility_checkbox,
        )
        self._add_checkbox_menu_action(
            menu,
            "Run all in background",
            self.background_all_checkbox,
        )
        self._add_checkbox_menu_action(
            menu,
            "Link napari/VIPP sliders",
            self.follow_dims_checkbox,
        )
        menu.addSeparator()
        self._add_combo_menu(menu, "Cache mode", self.cache_mode_combo)
        self._add_checkbox_menu_action(
            menu,
            "Auto memory guard",
            self.memory_guard_checkbox,
        )
        self._add_spinbox_menu_widget(
            menu,
            "Cache limit",
            self.memory_limit_spin,
        )
        added_section = True
        if hide_dropdowns:
            if added_section:
                menu.addSeparator()
            self._add_combo_menu(menu, "Preview mode", self.preview_mode_combo)
            self._add_combo_menu(
                menu,
                "Thumbnail contrast",
                self.thumbnail_contrast_combo,
            )
            self._add_combo_menu(
                menu,
                "Contrast range",
                self.thumbnail_scope_combo,
            )
            self._add_combo_menu(
                menu,
                "Monochrome colormap",
                self.thumbnail_colormap_combo,
            )
            added_section = True
        if hide_zoom:
            if added_section:
                menu.addSeparator()
            self._add_zoom_menu_widget(menu)

    def _add_checkbox_menu_action(
        self,
        menu: QMenu,
        label: str,
        checkbox: QCheckBox,
    ) -> QAction:
        action = menu.addAction(label)
        action.setCheckable(True)
        action.setChecked(checkbox.isChecked())
        action.triggered.connect(lambda checked: checkbox.setChecked(bool(checked)))
        return action

    def _add_combo_menu(
        self,
        menu: QMenu,
        label: str,
        combo: QComboBox,
    ) -> QMenu:
        submenu = menu.addMenu(label)
        current = combo.currentText()
        for index in range(combo.count()):
            value = combo.itemText(index)
            action = submenu.addAction(value)
            action.setCheckable(True)
            action.setChecked(value == current)
            action.triggered.connect(
                lambda _checked=False, selected=value: combo.setCurrentText(selected)
            )
        return submenu

    def _add_zoom_menu_widget(self, menu: QMenu) -> None:
        widget = QWidget(menu)
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(6)
        layout.addWidget(QLabel("Zoom"))
        slider = QSlider(Qt.Horizontal)
        slider.setRange(
            PipelineGraphView.SLIDER_MIN_ZOOM,
            PipelineGraphView.SLIDER_MAX_ZOOM,
        )
        slider.setSingleStep(5)
        slider.setPageStep(20)
        slider.setFixedWidth(150)
        slider.setValue(int(round(self.graph_view.zoom_percent)))
        label = QLabel(f"{int(round(self.graph_view.zoom_percent))}%")
        label.setMinimumWidth(44)
        reset_button = QToolButton()
        reset_button.setIcon(_toolbar_icon("reset"))
        reset_button.setIconSize(QSize(16, 16))
        reset_button.setToolTip("Reset graph zoom to the default 100%.")
        slider.valueChanged.connect(
            lambda value: self.graph_view.set_zoom_percent(float(value))
        )
        slider.valueChanged.connect(lambda value: label.setText(f"{int(value)}%"))
        reset_button.clicked.connect(
            lambda: slider.setValue(PipelineGraphView.DEFAULT_ZOOM)
        )
        layout.addWidget(slider)
        layout.addWidget(reset_button)
        layout.addWidget(label)

        action = QWidgetAction(menu)
        action.setDefaultWidget(widget)
        menu.addAction(action)

    def _add_spinbox_menu_widget(
        self,
        menu: QMenu,
        label: str,
        spinbox: QSpinBox,
    ) -> None:
        widget = QWidget(menu)
        widget.setFont(menu.font())
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(6)
        label_widget = QLabel(label, widget)
        label_widget.setFont(menu.font())
        layout.addWidget(label_widget)
        clone = QSpinBox(widget)
        clone.setFont(menu.font())
        clone.setRange(spinbox.minimum(), spinbox.maximum())
        clone.setSingleStep(spinbox.singleStep())
        clone.setSuffix(spinbox.suffix())
        clone.setValue(spinbox.value())
        clone.setToolTip(spinbox.toolTip())
        clone.valueChanged.connect(spinbox.setValue)
        spinbox.valueChanged.connect(clone.setValue)
        layout.addWidget(clone)
        action = QWidgetAction(menu)
        action.setDefaultWidget(widget)
        menu.addAction(action)

    def _build_graph_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumHeight(0)
        panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        panel_controls = QHBoxLayout()
        panel_controls.setContentsMargins(0, 0, 0, 0)
        panel_controls.setSpacing(4)
        panel_controls.addWidget(self.left_panel_toggle)
        panel_controls.addWidget(self.graph_search_edit)
        panel_controls.addWidget(self.graph_search_focus_button)
        panel_controls.addWidget(self.graph_search_status)
        panel_controls.addStretch(1)
        panel_controls.addWidget(self.right_panel_toggle)
        layout.addLayout(panel_controls)
        layout.addWidget(self.graph_view, 1)
        return panel

    def _build_palette_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(190)
        panel.setMinimumHeight(0)
        panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Ignored)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.palette_search)
        layout.addWidget(self.palette, 1)
        return panel

    def _build_inspector(self) -> QWidget:
        content = QWidget()
        content.setMinimumHeight(0)
        self.inspector_content = content
        layout = QVBoxLayout(content)
        layout.addWidget(self.selected_title)
        layout.addWidget(self.thumbnail_checkbox)
        layout.addWidget(self.keep_cached_checkbox)
        execution_layout = QVBoxLayout(self.execution_group)
        execution_layout.addWidget(self.execution_status_label)
        execution_layout.addWidget(self.auto_recalculate_checkbox)
        execution_layout.addWidget(self.auto_recalculate_notice)
        execution_layout.addWidget(self.calculate_button)
        layout.addWidget(self.execution_group)
        layout.addWidget(self.parameter_group)
        auto_layout = QVBoxLayout(self.auto_contrast_group)
        auto_form = QFormLayout()
        auto_form.addRow(
            AUTO_CONTRAST_SATURATION_SPEC.label,
            self.auto_saturation_control,
        )
        auto_layout.addLayout(auto_form)
        auto_layout.addWidget(self.auto_contrast_button)
        layout.addWidget(self.auto_contrast_group)
        colocalization_scatter_layout = QVBoxLayout(
            self.colocalization_scatter_group
        )
        colocalization_scatter_layout.addWidget(self.colocalization_scatter_summary)
        colocalization_scatter_controls = QWidget()
        colocalization_scatter_controls_layout = QHBoxLayout(
            colocalization_scatter_controls
        )
        colocalization_scatter_controls_layout.setContentsMargins(0, 0, 0, 0)
        colocalization_scatter_controls_layout.setSpacing(8)
        colocalization_scatter_controls_layout.addWidget(QLabel("Colormap"))
        colocalization_scatter_controls_layout.addWidget(
            self.colocalization_scatter_colormap_combo
        )
        colocalization_scatter_controls_layout.addWidget(
            self.colocalization_scatter_log_checkbox
        )
        colocalization_scatter_controls_layout.addStretch(1)
        colocalization_scatter_layout.addWidget(colocalization_scatter_controls)
        colocalization_scatter_layout.addWidget(self.colocalization_scatter_plot)
        layout.addWidget(self.colocalization_scatter_group)
        label_volume_layout = QVBoxLayout(self.label_volume_group)
        label_volume_layout.addWidget(self.label_volume_summary)
        label_volume_layout.addWidget(self.label_volume_log_checkbox)
        label_volume_layout.addWidget(self.label_volume_plot)
        layout.addWidget(self.label_volume_group)
        rescale_input_histogram_layout = QVBoxLayout(self.rescale_input_histogram_group)
        self.rescale_input_histogram_scope_row = QWidget()
        rescale_input_histogram_scope_layout = QHBoxLayout(
            self.rescale_input_histogram_scope_row
        )
        rescale_input_histogram_scope_layout.setContentsMargins(0, 0, 0, 0)
        rescale_input_histogram_scope_layout.addWidget(QLabel("Histogram uses"))
        rescale_input_histogram_scope_layout.addWidget(
            self.rescale_input_histogram_scope_combo,
            1,
        )
        rescale_input_histogram_layout.addWidget(self.rescale_input_histogram_scope_row)
        rescale_input_histogram_layout.addWidget(
            self.rescale_input_histogram_log_checkbox
        )
        rescale_input_histogram_layout.addWidget(self.rescale_input_histogram_plot)
        layout.addWidget(self.rescale_input_histogram_group)
        histogram_layout = QVBoxLayout(self.histogram_group)
        self.histogram_scope_row = QWidget()
        histogram_scope_layout = QHBoxLayout(self.histogram_scope_row)
        histogram_scope_layout.setContentsMargins(0, 0, 0, 0)
        histogram_scope_layout.addWidget(QLabel("Scope"))
        histogram_scope_layout.addWidget(self.histogram_scope_combo, 1)
        histogram_layout.addWidget(self.histogram_scope_row)
        histogram_layout.addWidget(self.histogram_log_checkbox)
        histogram_layout.addWidget(self.histogram_plot)
        layout.addWidget(self.histogram_group)
        table_layout = QVBoxLayout(self.table_group)
        table_layout.addWidget(self.table_summary)
        table_layout.addWidget(self.table_preview)
        layout.addWidget(self.table_group)
        metadata_layout = QVBoxLayout(self.metadata_group)
        metadata_layout.addWidget(self.metadata_table)
        metadata_layout.addWidget(self.history_title)
        metadata_layout.addWidget(self.history_label)
        layout.addWidget(self.metadata_group)

        actions = QHBoxLayout()
        actions.addWidget(self.pin_button)
        actions.addWidget(self.save_button)
        layout.addLayout(actions)
        layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(content)
        scroll.setMinimumWidth(230)
        scroll.setMinimumHeight(0)
        scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Ignored)
        return scroll

    def _connect_signals(self) -> None:
        self.new_workflow_button.clicked.connect(self._new_workflow_dialog)
        self.open_example_button.clicked.connect(self._open_example_workflow_dialog)
        self.auto_structure_button.clicked.connect(self._auto_structure_graph)
        self.refresh_button.clicked.connect(self._refresh_and_run)
        self.calculate_all_button.clicked.connect(self._calculate_all_nodes)
        self.undo_action.triggered.connect(self.undo)
        self.redo_action.triggered.connect(self.redo)
        self.save_workflow_button.clicked.connect(self._save_workflow_dialog)
        self.load_workflow_button.clicked.connect(self._load_workflow_dialog)
        self.export_button.clicked.connect(self._export_python_dialog)
        self.batch_button.clicked.connect(self._batch_collection_dialog)
        self.export_ome_button.clicked.connect(self._export_ome_dataset_dialog)
        self.tunnel_manager_button.clicked.connect(self._show_tunnel_manager)
        self.pipeline_cancel_button.clicked.connect(self._cancel_background_pipeline_run)
        self.cache_mode_combo.currentTextChanged.connect(self._on_cache_mode_changed)
        self.memory_guard_checkbox.toggled.connect(
            self._on_memory_guard_setting_changed
        )
        self.memory_limit_spin.valueChanged.connect(
            self._on_memory_guard_setting_changed
        )
        self.preview_mode_combo.currentTextChanged.connect(self._update_thumbnails)
        self.thumbnail_contrast_combo.currentTextChanged.connect(
            self._update_thumbnails,
        )
        self.thumbnail_scope_combo.currentTextChanged.connect(self._update_thumbnails)
        self.thumbnail_colormap_combo.currentTextChanged.connect(
            self._update_thumbnails
        )
        self.follow_dims_checkbox.toggled.connect(self._on_follow_dims_toggled)
        self.graph_zoom_slider.valueChanged.connect(self._on_graph_zoom_slider_changed)
        self.graph_zoom_reset_button.clicked.connect(self._reset_graph_zoom)
        self.view_dims_bar.value_changed.connect(self._on_view_dim_changed)
        self.calculate_button.clicked.connect(self._calculate_selected_node)
        self.auto_recalculate_checkbox.toggled.connect(
            self._on_auto_recalculate_toggled
        )
        self.pin_button.clicked.connect(lambda: self.pin_node(self._selected_node_id))
        self.thumbnail_checkbox.toggled.connect(
            self._on_selected_preview_toggled,
        )
        self.keep_cached_checkbox.toggled.connect(self._on_keep_cached_toggled)
        self.histogram_log_checkbox.toggled.connect(self._update_histogram)
        self.histogram_scope_combo.currentTextChanged.connect(self._update_histogram)
        self.rescale_input_histogram_scope_combo.currentTextChanged.connect(
            self._update_histogram
        )
        self.rescale_input_histogram_log_checkbox.toggled.connect(
            self._update_histogram
        )
        self.rescale_input_histogram_plot.markerChanged.connect(
            self._on_input_histogram_marker_changed
        )
        self.label_volume_log_checkbox.toggled.connect(
            self._update_label_volume_histogram
        )
        self.label_volume_plot.markerChanged.connect(
            self._on_label_volume_marker_changed
        )
        self.colocalization_scatter_log_checkbox.toggled.connect(
            self._update_colocalization_scatter
        )
        self.colocalization_scatter_colormap_combo.currentTextChanged.connect(
            self._update_colocalization_scatter
        )
        self.colocalization_scatter_plot.thresholdChanged.connect(
            self._on_colocalization_scatter_threshold_changed
        )
        self.auto_contrast_button.clicked.connect(self._apply_auto_contrast)
        self.save_button.clicked.connect(self._save_selected_output_dialog)
        self.left_panel_toggle.clicked.connect(self._toggle_left_panel)
        self.right_panel_toggle.clicked.connect(self._toggle_right_panel)
        self.palette_search.textChanged.connect(self.palette.set_filter_text)
        self.graph_search_edit.textChanged.connect(self._on_graph_search_changed)
        self.graph_search_edit.returnPressed.connect(self._focus_next_graph_search_match)
        self.graph_search_focus_button.clicked.connect(
            self._focus_next_graph_search_match
        )

        self.palette.operation_requested.connect(self.add_node_from_palette)
        self.graph_view.node_create_requested.connect(self._add_node_at)
        self.graph_view.node_insert_requested.connect(self._insert_node_on_connection)
        self.graph_view.connection_insert_requested.connect(
            self._insert_node_from_connection_menu
        )
        self.graph_view.node_selected.connect(self._select_node)
        self.graph_view.node_delete_requested.connect(self._delete_node)
        self.graph_view.node_duplicate_requested.connect(self._duplicate_node)
        self.graph_view.node_code_requested.connect(self._inspect_node_code)
        self.graph_view.node_note_requested.connect(self._add_graph_note_for_node)
        self.graph_view.node_moved.connect(self._on_node_moved)
        self.graph_view.node_splice_requested.connect(
            self._insert_existing_node_on_connection
        )
        self.graph_view.pin_requested.connect(self.pin_node)
        self.graph_view.node_calculate_requested.connect(self._calculate_node)
        self.graph_view.connection_requested.connect(self._connect_nodes)
        self.graph_view.connection_removed.connect(self._disconnect_nodes)
        self.graph_view.port_context_requested.connect(self._show_port_context_menu)
        self.graph_view.tunnel_selected.connect(self._on_graph_tunnel_selected)
        self.graph_view.note_moved.connect(self._on_graph_note_moved)
        self.graph_view.note_edit_requested.connect(self._edit_graph_note)
        self.graph_view.note_delete_requested.connect(self._delete_graph_note)
        self.graph_view.status_message.connect(self.status_label.setText)
        self.graph_view.zoom_changed.connect(self._sync_graph_zoom_controls)

        try:
            self.viewer.layers.events.inserted.connect(
                self._on_viewer_layers_changed
            )
            self.viewer.layers.events.removed.connect(
                self._on_viewer_layers_changed
            )
        except Exception:
            pass
        try:
            self.viewer.dims.events.current_step.connect(self._on_dims_changed)
            self.viewer.dims.events.point.connect(self._on_dims_changed)
        except Exception:
            pass
        self._debounce_timer.timeout.connect(self._finish_parameter_history_group)

    def _toggle_left_panel(self) -> None:
        self._set_left_panel_visible(self.palette_panel.isHidden())

    def _toggle_right_panel(self) -> None:
        self._set_right_panel_visible(self.inspector_panel.isHidden())

    def _on_graph_zoom_slider_changed(self, value: int) -> None:
        self.graph_view.set_zoom_percent(float(value))

    def _reset_graph_zoom(self) -> None:
        self.graph_view.reset_zoom()

    def _sync_graph_zoom_controls(self, value: float) -> None:
        percent = int(round(float(value)))
        self.graph_zoom_label.setText(f"{percent}%")
        clipped = int(
            np.clip(
                percent,
                PipelineGraphView.SLIDER_MIN_ZOOM,
                PipelineGraphView.SLIDER_MAX_ZOOM,
            )
        )
        with QSignalBlocker(self.graph_zoom_slider):
            self.graph_zoom_slider.setValue(clipped)

    def _on_graph_search_changed(self, _text: str) -> None:
        self._clear_graph_search_tunnel_highlight()
        self._refresh_graph_search_matches(reset_index=True)

    def _refresh_graph_search_matches(self, *, reset_index: bool = True) -> None:
        query = self.graph_search_edit.text()
        matches = find_graph_matches(
            query,
            self.pipeline.nodes.values(),
            self.pipeline.output_tunnel_list(),
            self.pipeline.connections,
        )
        self._graph_search_matches = matches
        if reset_index:
            self._graph_search_index = 0 if matches else -1
        elif matches and self._graph_search_index >= len(matches):
            self._graph_search_index = 0
        elif not matches:
            self._graph_search_index = -1

        node_ids = {
            node_id
            for match in matches
            for node_id in match.node_ids
            if node_id in self.pipeline.nodes
        }
        if node_ids:
            self.graph_view.set_search_matches(node_ids)
        else:
            self.graph_view.clear_search_matches()
        self._sync_graph_search_status()

    def _sync_graph_search_status(self) -> None:
        query = self.graph_search_edit.text()
        if not _normalize_search_text(query):
            text = ""
        elif not self._graph_search_matches:
            text = "No matches"
        else:
            count = len(self._graph_search_matches)
            text = f"{count} match" if count == 1 else f"{count} matches"
        self.graph_search_status.setText(text)
        self.graph_search_focus_button.setEnabled(bool(self._graph_search_matches))

    def _focus_next_graph_search_match(self) -> None:
        if not self._graph_search_matches:
            self._refresh_graph_search_matches(reset_index=True)
        if not self._graph_search_matches:
            query = self.graph_search_edit.text().strip()
            if query:
                self.status_label.setText(f"No graph match for '{query}'.")
            else:
                self.status_label.setText("Enter graph search text to focus a match.")
            return

        index = self._graph_search_index
        if index < 0 or index >= len(self._graph_search_matches):
            index = 0
        match = self._graph_search_matches[index]
        self._graph_search_index = (index + 1) % len(self._graph_search_matches)
        self._focus_graph_search_match(match)

    def _focus_graph_search_match(self, match: GraphSearchMatch) -> None:
        if match.kind == "tunnel" and match.tunnel_name:
            self._clear_graph_search_tunnel_highlight()
            self.graph_view.set_search_matches(match.node_ids)
            self.graph_view.reveal_tunnel(match.tunnel_name)
            self._graph_search_highlighted_tunnel = match.tunnel_name
            self.status_label.setText(self._tunnel_status_text(match.tunnel_name))
            return

        if match.node_id not in self.pipeline.nodes:
            self._refresh_graph_search_matches(reset_index=True)
            return
        self._clear_graph_search_tunnel_highlight()
        self.graph_view.clear_tunnel_highlight(sticky=True)
        self.graph_view.focus_node(match.node_id)
        node = self.pipeline.nodes[match.node_id]
        fields = ", ".join(match.matched_fields)
        suffix = f" via {fields}" if fields else ""
        self.status_label.setText(f"Focused '{node.title}'{suffix}.")

    def _clear_graph_search_tunnel_highlight(self) -> None:
        if not self._graph_search_highlighted_tunnel:
            return
        self.graph_view.clear_tunnel_highlight(sticky=True)
        self._graph_search_highlighted_tunnel = ""

    def _on_follow_dims_toggled(self, _checked: bool) -> None:
        self._capture_vipp_dims_from_viewer()
        self._update_thumbnails()
        self._update_metadata_panel()
        self._update_histogram()
        self._sync_view_dims_bar()

    def undo(self) -> None:
        """Restore the previous workflow graph snapshot."""
        self._finish_parameter_history_group()
        if not self._undo_stack:
            return
        current = self._current_history_snapshot()
        snapshot = self._undo_stack.pop()
        self._redo_stack.append(current)
        self._restore_history_snapshot(snapshot)
        self._sync_history_actions()
        self.status_label.setText("Undid last workflow edit.")

    def redo(self) -> None:
        """Reapply the most recently undone workflow graph snapshot."""
        self._finish_parameter_history_group()
        if not self._redo_stack:
            return
        current = self._current_history_snapshot()
        snapshot = self._redo_stack.pop()
        self._undo_stack.append(current)
        if len(self._undo_stack) > self.HISTORY_LIMIT:
            del self._undo_stack[: len(self._undo_stack) - self.HISTORY_LIMIT]
        self._restore_history_snapshot(snapshot)
        self._sync_history_actions()
        self.status_label.setText("Redid workflow edit.")

    def _current_history_snapshot(
        self,
        positions: dict[str, tuple[float, float]] | None = None,
        notes_override: list[dict] | None = None,
    ) -> WorkflowHistorySnapshot:
        if positions is None:
            positions = self.graph_view.node_positions()
        valid_node_ids = set(self.pipeline.nodes)
        active_pin = (
            self._active_pinned_node_id
            if self._active_pinned_node_id in valid_node_ids
            else None
        )
        return WorkflowHistorySnapshot(
            workflow=deepcopy(
                serialize_workflow(
                    self.pipeline,
                    positions,
                    self._graph_note_documents()
                    if notes_override is None
                    else notes_override,
                )
            ),
            selected_node_id=(
                self._selected_node_id
                if self._selected_node_id in valid_node_ids
                else ""
            ),
            preview_disabled_node_ids=tuple(
                sorted(self._preview_disabled_node_ids & valid_node_ids)
            ),
            active_pinned_node_id=active_pin,
        )

    def _push_undo_snapshot(
        self,
        snapshot: WorkflowHistorySnapshot | None = None,
    ) -> None:
        if self._restoring_history:
            return
        snapshot = snapshot or self._current_history_snapshot()
        if self._undo_stack and self._undo_stack[-1] == snapshot:
            return
        self._undo_stack.append(snapshot)
        if len(self._undo_stack) > self.HISTORY_LIMIT:
            del self._undo_stack[: len(self._undo_stack) - self.HISTORY_LIMIT]
        self._redo_stack.clear()
        self._sync_history_actions()

    def _push_undo_if_changed(self, before: WorkflowHistorySnapshot) -> None:
        if before != self._current_history_snapshot():
            self._push_undo_snapshot(before)

    def _record_parameter_undo(self, node_id: str, name: str) -> None:
        key = (node_id, name)
        if self._pending_parameter_undo_key == key:
            return
        self._push_undo_snapshot()
        self._pending_parameter_undo_key = key

    def _finish_parameter_history_group(self) -> None:
        self._pending_parameter_undo_key = None

    def _restore_history_snapshot(self, snapshot: WorkflowHistorySnapshot) -> None:
        self._restoring_history = True
        self._debounce_timer.stop()
        self._interactive_collection_source_paths.clear()
        try:
            workflow = self._deserialize_history_workflow(snapshot.workflow)
            pinned_layer = self._active_pinned_layer()
            if pinned_layer is not None:
                self._remove_layer(pinned_layer)
            self.pipeline.restore_graph(
                workflow["nodes"],
                workflow["connections"],
                workflow.get("output_tunnels", ()),
            )
            self._restore_graph_notes(workflow.get("notes", ()))
            valid_node_ids = set(self.pipeline.nodes)
            self._preview_disabled_node_ids = (
                set(snapshot.preview_disabled_node_ids) & valid_node_ids
            )
            self._active_pinned_node_id = (
                snapshot.active_pinned_node_id
                if snapshot.active_pinned_node_id in valid_node_ids
                else None
            )
            selected = (
                snapshot.selected_node_id
                if snapshot.selected_node_id in valid_node_ids
                else ""
            )
            self._selected_node_id = selected
            self.graph_view.build_graph(
                self.pipeline.nodes.values(),
                self.pipeline.connections,
                workflow["positions"],
                output_tunnels=self.pipeline.output_tunnel_list(),
                notes=self._graph_note_documents(use_view_positions=False),
                preserve_view=True,
            )
            self._sync_all_input_ports()
            self._sync_all_output_ports()
            self._refresh_graph_search_matches(reset_index=True)
            self._sync_pin_ui()
            self._invalidate_pipeline_cache()
            self.run_pipeline()
            if selected:
                self.graph_view.select_node(selected)
            else:
                self._select_first_available_node()
        finally:
            self._restoring_history = False

    @staticmethod
    def _deserialize_history_workflow(workflow: dict) -> dict[str, object]:
        if workflow.get("type") == "napari-vipp-workflow" and not workflow.get(
            "nodes",
        ):
            return {"nodes": [], "connections": [], "positions": {}}
        return deserialize_workflow(deepcopy(workflow))

    def _sync_history_actions(self) -> None:
        undo_enabled = bool(self._undo_stack)
        redo_enabled = bool(self._redo_stack)
        self.undo_action.setEnabled(undo_enabled)
        self.redo_action.setEnabled(redo_enabled)
        undo_shortcut = QKeySequence(QKeySequence.Undo).toString(
            QKeySequence.NativeText
        )
        redo_shortcut = QKeySequence(QKeySequence.Redo).toString(
            QKeySequence.NativeText
        )
        self.undo_action.setToolTip(f"Undo last workflow edit ({undo_shortcut})")
        self.redo_action.setToolTip(f"Redo workflow edit ({redo_shortcut})")

    def _set_left_panel_visible(self, visible: bool) -> None:
        self._set_side_panel_visible("left", visible)

    def _set_right_panel_visible(self, visible: bool) -> None:
        self._set_side_panel_visible("right", visible)

    def _set_side_panel_visible(self, side: str, visible: bool) -> None:
        sizes = self._current_splitter_sizes()
        if side == "left":
            widget = self.palette_panel
            index = 0
            if not visible and sizes[index] > 0:
                self._left_panel_last_width = sizes[index]
        else:
            widget = self.inspector_panel
            index = 2
            if not visible and sizes[index] > 0:
                self._right_panel_last_width = sizes[index]

        widget.setVisible(visible)
        self._apply_splitter_panel_sizes()
        self._sync_side_panel_toggles()
        action = "shown" if visible else "hidden"
        panel = "node library" if side == "left" else "inspector"
        self.status_label.setText(f"{panel.capitalize()} {action}.")

    def _current_splitter_sizes(self) -> list[int]:
        sizes = self.splitter.sizes()
        if len(sizes) != 3 or sum(sizes) <= 0:
            return list(self._default_splitter_sizes)
        return [max(int(size), 0) for size in sizes]

    def _apply_splitter_panel_sizes(self) -> None:
        current = self._current_splitter_sizes()
        total = max(sum(current), sum(self._default_splitter_sizes))
        left = 0 if self.palette_panel.isHidden() else self._left_panel_last_width
        right = 0 if self.inspector_panel.isHidden() else self._right_panel_last_width
        middle = max(total - left - right, 320)
        self.splitter.setSizes([left, middle, right])

    def _sync_side_panel_toggles(self) -> None:
        left_visible = not self.palette_panel.isHidden()
        right_visible = not self.inspector_panel.isHidden()
        self.left_panel_toggle.set_expanded(left_visible)
        self.left_panel_toggle.setToolTip(
            "Hide node library" if left_visible else "Show node library"
        )
        self.right_panel_toggle.set_expanded(right_visible)
        self.right_panel_toggle.setToolTip(
            "Hide inspector" if right_visible else "Show inspector"
        )

    def _auto_structure_graph(self) -> None:
        self._finish_parameter_history_group()
        if not self.pipeline.nodes:
            self.status_label.setText("No graph nodes to structure.")
            return

        before = self._current_history_snapshot()
        sizes = self.graph_view.node_card_sizes()
        layout_nodes = [
            LayoutNode(
                node_id,
                sizes.get(node_id, (220.0, 180.0))[0],
                sizes.get(node_id, (220.0, 180.0))[1],
            )
            for node_id in self.pipeline.nodes
        ]
        layout_edges = [
            LayoutEdge(connection.source_id, connection.target_id)
            for connection in self.pipeline.connections
        ]
        positions = layout_layered_dag(
            layout_nodes,
            layout_edges,
            current_positions=self.graph_view.node_positions(),
        )
        if not self.graph_view.apply_node_positions(positions):
            self.status_label.setText("Graph is already structured.")
            return

        self._sync_graph_note_positions_from_view()
        self._push_undo_if_changed(before)
        self.status_label.setText(
            f"Auto-structured graph layout for {len(positions)} nodes."
        )

    def add_node_from_palette(self, operation_id: str):
        return self._add_node_at(operation_id, self.graph_view.suggest_node_position())

    def _add_node_at(self, operation_id: str, position) -> object:
        self._finish_parameter_history_group()
        before = self._current_history_snapshot()
        node = self.pipeline.add_node(operation_id)
        self.graph_view.add_node(node, position)
        self._sync_node_input_ports(node.id)
        self._sync_node_output_ports(node.id)
        self._sync_pin_ui()
        self._refresh_graph_search_matches(reset_index=True)
        self.graph_view.select_node(node.id)
        if self._active_pipeline_run_id is not None:
            # A loose node does not need calculation yet, but it does change the
            # serialized workflow owned by the active worker. Register that
            # additive graph edit as independent pending work so the worker's
            # still-valid result can be applied before the new node is visited.
            self._mark_pipeline_dirty(node.id)
            self.run_pipeline()
        self._push_undo_if_changed(before)
        self.status_label.setText(f"Added '{node.title}'.")
        return node

    def _insert_node_on_connection(
        self,
        operation_id: str,
        connection_key,
        position,
    ) -> object | None:
        self._finish_parameter_history_group()
        before = self._current_history_snapshot()
        mode, reason = self._connection_insert_mode(operation_id, connection_key)
        if mode == "incompatible":
            self.status_label.setText(reason)
            return None
        mapping = None
        if mode == "choose":
            mapping = self._choose_connection_insert_mapping(
                operation_id,
                connection_key,
            )
            if mapping is None:
                self.status_label.setText("Insert canceled.")
                return None
        splice_mode = "full" if mapping is not None else mode

        try:
            source_id, target_id, target_port, source_port = (
                self._normalize_connection_key(connection_key)
            )
            downstream = self.pipeline.descendants_inclusive([target_id])
            node = self.pipeline.add_node(operation_id)
            self._apply_insert_mapping_params(node.id, mapping)
            self.graph_view.add_node(node, position)
            self._sync_node_input_ports(node.id)
            self._sync_node_output_ports(node.id)
            self.graph_view.center_node_on(node.id, position)
            self._make_room_for_inserted_node(source_id, target_id, node.id, downstream)

            changed_connections = False
            if splice_mode in {"full", "partial"}:
                if not self.pipeline.disconnect(source_id, target_id, target_port):
                    raise RuntimeError("Original connection was no longer available.")
                self.graph_view.remove_connection(
                    source_id,
                    target_id,
                    target_port=target_port,
                    notify=False,
                )
                changed_connections = True

                input_result = self.pipeline.connect(
                    source_id,
                    node.id,
                    target_port=mapping.input_port if mapping is not None else 0,
                    source_port=source_port,
                )
                if not input_result.success:
                    raise RuntimeError(input_result.message)
                self._apply_connection_result_to_graph(input_result)
                self._sync_node_output_ports(node.id)

                if splice_mode == "full":
                    output_result = self.pipeline.connect(
                        node.id,
                        target_id,
                        target_port=target_port,
                        source_port=(
                            mapping.output_port if mapping is not None else 0
                        ),
                    )
                    if not output_result.success:
                        raise RuntimeError(output_result.message)
                    self._apply_connection_result_to_graph(output_result)

            self.graph_view.select_node(node.id)
            self._sync_pin_ui()
            dirty_nodes = {node.id}
            if splice_mode == "partial":
                dirty_nodes.add(target_id)
            self._mark_pipeline_branches_dirty(dirty_nodes)
            self.run_pipeline()
            self._make_room_for_inserted_node(source_id, target_id, node.id, downstream)
            self._push_undo_if_changed(before)
        except Exception as exc:
            self._restore_history_snapshot(before)
            self.status_label.setText(f"Insert failed: {exc}")
            return None

        if mapping is not None:
            self.status_label.setText(
                f"Inserted '{node.title}' between "
                f"'{self._node_title(source_id)}' and '{self._node_title(target_id)}' "
                f"using {mapping.input_label} and {mapping.output_label}."
            )
        elif mode == "full":
            self.status_label.setText(
                f"Inserted '{node.title}' between "
                f"'{self._node_title(source_id)}' and '{self._node_title(target_id)}'."
            )
        elif mode == "partial":
            self.status_label.setText(
                f"Inserted '{node.title}' after '{self._node_title(source_id)}'. "
                f"Choose which output should feed '{self._node_title(target_id)}'."
            )
        else:
            connection_note = (
                " Connections were left unchanged."
                if not changed_connections
                else ""
            )
            self.status_label.setText(
                f"Placed '{node.title}' in the opened gap. "
                f"Connect ports manually.{connection_note}"
            )
        return node

    def _insert_node_from_connection_menu(self, connection_key, position) -> None:
        operation_id = self._choose_connection_insert_operation(connection_key)
        if not operation_id:
            return
        self._insert_node_on_connection(operation_id, connection_key, position)

    def _insert_existing_node_on_connection(
        self,
        node_id: str,
        connection_key,
        old_pos,
        _new_pos,
    ) -> object | None:
        self._finish_parameter_history_group()
        node = self.pipeline.nodes.get(node_id)
        if node is None:
            return None

        old_point = QPointF(old_pos)
        positions = self.graph_view.node_positions()
        positions[node_id] = (float(old_point.x()), float(old_point.y()))
        before = self._current_history_snapshot(positions)

        if self._node_has_connections(node_id):
            self._push_undo_if_changed(before)
            self.status_label.setText(
                f"Disconnect '{node.title}' before inserting it on a wire."
            )
            return None

        mode, reason = self._connection_insert_mode(
            node.operation_id,
            connection_key,
            inserted_node_id=node_id,
        )
        if mode == "incompatible":
            self._push_undo_if_changed(before)
            self.status_label.setText(reason)
            return None
        mapping = None
        if mode == "choose":
            mapping = self._choose_connection_insert_mapping(
                node.operation_id,
                connection_key,
                inserted_node_id=node_id,
            )
            if mapping is None:
                self._push_undo_if_changed(before)
                self.status_label.setText("Insert canceled.")
                return None
        splice_mode = "full" if mapping is not None else mode

        try:
            source_id, target_id, target_port, source_port = (
                self._normalize_connection_key(connection_key)
            )
            downstream = self.pipeline.descendants_inclusive([target_id])
            downstream.discard(node_id)
            self._apply_insert_mapping_params(node_id, mapping)
            self._make_room_for_inserted_node(source_id, target_id, node_id, downstream)

            changed_connections = False
            if splice_mode in {"full", "partial"}:
                if not self.pipeline.disconnect(source_id, target_id, target_port):
                    raise RuntimeError("Original connection was no longer available.")
                self.graph_view.remove_connection(
                    source_id,
                    target_id,
                    target_port=target_port,
                    notify=False,
                )
                changed_connections = True

                input_result = self.pipeline.connect(
                    source_id,
                    node_id,
                    target_port=mapping.input_port if mapping is not None else 0,
                    source_port=source_port,
                )
                if not input_result.success:
                    raise RuntimeError(input_result.message)
                self._apply_connection_result_to_graph(input_result)
                self._sync_node_output_ports(node_id)

                if splice_mode == "full":
                    output_result = self.pipeline.connect(
                        node_id,
                        target_id,
                        target_port=target_port,
                        source_port=(
                            mapping.output_port if mapping is not None else 0
                        ),
                    )
                    if not output_result.success:
                        raise RuntimeError(output_result.message)
                    self._apply_connection_result_to_graph(output_result)

            self.graph_view.select_node(node_id)
            self._sync_pin_ui()
            if changed_connections:
                dirty_nodes = {node_id}
                if splice_mode == "partial":
                    dirty_nodes.add(target_id)
                self._mark_pipeline_branches_dirty(dirty_nodes)
                self.run_pipeline()
                self._make_room_for_inserted_node(
                    source_id,
                    target_id,
                    node_id,
                    downstream,
                )
            self._push_undo_if_changed(before)
        except Exception as exc:
            self._restore_history_snapshot(before)
            self.status_label.setText(f"Insert failed: {exc}")
            return None

        if mapping is not None:
            self.status_label.setText(
                f"Inserted existing '{node.title}' between "
                f"'{self._node_title(source_id)}' and '{self._node_title(target_id)}' "
                f"using {mapping.input_label} and {mapping.output_label}."
            )
        elif mode == "full":
            self.status_label.setText(
                f"Inserted existing '{node.title}' between "
                f"'{self._node_title(source_id)}' and '{self._node_title(target_id)}'."
            )
        elif mode == "partial":
            self.status_label.setText(
                f"Inserted existing '{node.title}' after "
                f"'{self._node_title(source_id)}'. Choose which output should feed "
                f"'{self._node_title(target_id)}'."
            )
        else:
            self.status_label.setText(
                f"Placed existing '{node.title}' in the opened gap. "
                "Connect ports manually."
            )
        return node

    def _node_has_connections(self, node_id: str) -> bool:
        return any(
            connection.source_id == node_id or connection.target_id == node_id
            for connection in self.pipeline.connections
        )

    def _apply_insert_mapping_params(
        self,
        node_id: str,
        mapping: ConnectionInsertPortMapping | None,
    ) -> None:
        if mapping is None:
            return
        for name, value in mapping.params:
            self.pipeline.set_param(node_id, str(name), value)

    def _choose_connection_insert_operation(self, connection_key) -> str | None:
        candidates = self._connection_insert_candidates(connection_key)
        if not candidates:
            self.status_label.setText("No compatible nodes can be inserted here.")
            return None
        dialog = ConnectionInsertDialog(candidates, self)
        if dialog.exec() != QDialog.Accepted:
            return None
        return dialog.selected_operation_id()

    def _connection_insert_candidates(
        self,
        connection_key,
    ) -> list[ConnectionInsertCandidate]:
        candidates: list[ConnectionInsertCandidate] = []
        for category, subgroups in grouped_palette_specs().items():
            for subcategory, specs in subgroups.items():
                for spec in specs:
                    mode, reason = self._connection_insert_mode(
                        spec.id,
                        connection_key,
                    )
                    if mode == "incompatible":
                        continue
                    search_text = _normalize_search_text(
                        " ".join(
                            (
                                spec.id,
                                spec.title,
                                spec.category,
                                spec.subcategory,
                                mode,
                                self._connection_insert_mode_label(mode),
                            )
                        )
                    )
                    candidates.append(
                        ConnectionInsertCandidate(
                            operation_id=spec.id,
                            title=spec.title,
                            category=category,
                            subcategory=subcategory,
                            mode=mode,
                            detail=self._connection_insert_detail(mode, reason),
                            search_text=search_text,
                        )
                    )
        return candidates

    @staticmethod
    def _connection_insert_mode_label(mode: str) -> str:
        return {
            "full": "full splice",
            "choose": "choose ports",
            "partial": "partial insert",
            "place": "place in gap",
        }.get(mode, mode)

    def _connection_insert_detail(self, mode: str, reason: str) -> str:
        if mode == "full":
            return (
                "Full splice: replace the original wire and connect "
                "source -> inserted node -> target."
            )
        if mode == "choose":
            return (
                "Choose ports: select which inserted input receives the upstream "
                "output and which inserted output feeds the downstream node."
            )
        if mode == "partial":
            return (
                "Partial insert: replace the original wire and connect only "
                "source -> inserted node. Choose the downstream output manually."
            )
        if mode == "place":
            if reason:
                return f"Place in gap: {reason}"
            return "Place in gap: create the node here and connect ports manually."
        return reason

    def _apply_connection_result_to_graph(self, result) -> None:
        affected_sources = {
            connection.source_id for connection in result.removed
        }
        for connection in result.removed:
            self.graph_view.remove_connection(
                connection.source_id,
                connection.target_id,
                target_port=connection.target_port,
                notify=False,
            )
        if result.connection is not None:
            affected_sources.add(result.connection.source_id)
            if not getattr(result.connection, "tunnel_name", ""):
                self.graph_view.add_connection(
                    result.connection.source_id,
                    result.connection.target_id,
                    result.connection.target_port,
                    result.connection.source_port,
                )
        self._sync_port_tunnels()
        self._refresh_split_channel_display_surfaces(affected_sources)

    @staticmethod
    def _normalize_connection_key(connection_key) -> tuple[str, str, int, int]:
        source_id, target_id, target_port, source_port = tuple(connection_key)
        return str(source_id), str(target_id), int(target_port), int(source_port)

    def _choose_connection_insert_mapping(
        self,
        operation_id: str,
        connection_key,
        *,
        inserted_node_id: str | None = None,
    ) -> ConnectionInsertPortMapping | None:
        axis_choices = self._connection_insert_axis_choices(
            operation_id,
            connection_key,
        )
        mappings_by_axis: dict[str, list[ConnectionInsertPortMapping]] = {}
        if axis_choices:
            for axis_value, _axis_label in axis_choices:
                axis_mappings = self._connection_insert_mapping_options(
                    operation_id,
                    connection_key,
                    inserted_node_id=inserted_node_id,
                    params_override={"axis": axis_value},
                )
                if axis_mappings:
                    mappings_by_axis[axis_value] = axis_mappings
            axis_choices = [
                choice for choice in axis_choices if choice[0] in mappings_by_axis
            ]
            if not axis_choices:
                return None
            mappings = list(mappings_by_axis[axis_choices[0][0]])
        else:
            mappings = self._connection_insert_mapping_options(
                operation_id,
                connection_key,
                inserted_node_id=inserted_node_id,
            )
        if len(mappings) == 1 and len(axis_choices) <= 1:
            return mappings[0]
        if not mappings:
            return None
        try:
            source_id, target_id, _target_port, _source_port = (
                self._normalize_connection_key(connection_key)
            )
            title = self.pipeline.operation_spec(operation_id).title
            source_title = self._node_title(source_id)
            target_title = self._node_title(target_id)
        except Exception:
            return None
        dialog = ConnectionInsertMappingDialog(
            mappings,
            title,
            source_title,
            target_title,
            axis_choices=axis_choices,
            mappings_by_axis=mappings_by_axis,
            parent=self,
        )
        if dialog.exec() != QDialog.Accepted:
            return None
        return dialog.selected_mapping()

    def _connection_insert_mapping_options(
        self,
        operation_id: str,
        connection_key,
        *,
        inserted_node_id: str | None = None,
        params_override: dict[str, object] | None = None,
    ) -> list[ConnectionInsertPortMapping]:
        try:
            source_id, target_id, target_port, source_port = (
                self._normalize_connection_key(connection_key)
            )
            spec = self.pipeline.operation_spec(operation_id)
        except Exception:
            return []
        if source_id not in self.pipeline.nodes or target_id not in self.pipeline.nodes:
            return []

        source_ports = self.pipeline.output_ports(source_id)
        target_ports = self.pipeline.input_ports(target_id)
        if not 0 <= source_port < len(source_ports):
            return []
        if not 0 <= target_port < len(target_ports):
            return []

        source_type = source_ports[source_port].output_type
        target_type = target_ports[target_port].input_type
        input_ports = self._operation_insert_input_ports(
            spec,
            inserted_node_id=inserted_node_id,
        )
        output_ports = self._operation_insert_output_ports(
            spec,
            source_type,
            inserted_node_id=inserted_node_id,
            source_id=source_id,
            source_port=source_port,
            params_override=params_override,
        )
        compatible_inputs = [
            index
            for index, port in enumerate(input_ports)
            if self.pipeline._types_compatible(source_type, port.input_type)
        ]
        compatible_outputs = [
            index
            for index, port in enumerate(output_ports)
            if self.pipeline._types_compatible(port.output_type, target_type)
        ]

        source_title = self._node_title(source_id)
        target_title = self._node_title(target_id)
        mappings: list[ConnectionInsertPortMapping] = []
        for input_index in compatible_inputs:
            input_label = input_ports[input_index].label
            for output_index in compatible_outputs:
                output_label = output_ports[output_index].label
                mappings.append(
                    ConnectionInsertPortMapping(
                        input_port=input_index,
                        output_port=output_index,
                        input_label=f"input {input_index + 1}: {input_label}",
                        output_label=f"output {output_index + 1}: {output_label}",
                        detail=(
                            f"{source_title} -> {input_label}; "
                            f"{output_label} -> {target_title} "
                            f"input {target_port + 1}"
                        ),
                        params=tuple((params_override or {}).items()),
                    )
                )
        return mappings

    def _connection_insert_axis_choices(
        self,
        operation_id: str,
        connection_key,
    ) -> list[tuple[str, str]]:
        if operation_id != "split_axis":
            return []
        try:
            source_id, _target_id, _target_port, source_port = (
                self._normalize_connection_key(connection_key)
            )
        except Exception:
            return []
        options = self._split_axis_options_from(
            self._axis_slice_options_for_output(source_id, source_port),
            fallback_all=False,
        )
        return [
            (f"axis:{option.index}", self._split_axis_choice_label(option))
            for option in options
        ]

    def _labeled_split_axis_insert_ports(
        self,
        ports,
        source_id: str,
        source_port: int,
        params: dict[str, object],
    ):
        shape = self.pipeline._resolved_output_shape(source_id, source_port)
        if not shape:
            return ports
        axis_index = self._split_axis_index_from_params(params, len(shape))
        options_by_index = {
            option.index: option
            for option in self._axis_slice_options_for_output(source_id, source_port)
        }
        option = options_by_index.get(axis_index)
        label = option.title if option is not None else f"Axis {axis_index}"
        return tuple(
            replace(
                port,
                name=f"{label.lower()}_{index + 1}",
                title=f"{label} {index + 1}",
            )
            for index, port in enumerate(ports)
        )

    def _connection_insert_preview_state(
        self,
        operation_id: str,
        connection_key,
    ) -> tuple[str, str]:
        mode, reason = self._connection_insert_mode(operation_id, connection_key)
        try:
            source_id, target_id, _target_port, _source_port = (
                self._normalize_connection_key(connection_key)
            )
            title = self.pipeline.operation_spec(operation_id).title
            source_title = self._node_title(source_id)
            target_title = self._node_title(target_id)
        except Exception:
            return "incompatible", reason
        if mode == "full":
            return (
                "full",
                f"Drop to insert '{title}' between "
                f"'{source_title}' and '{target_title}'.",
            )
        if mode == "choose":
            return (
                "partial",
                f"Drop to insert '{title}' and choose ports between "
                f"'{source_title}' and '{target_title}'.",
            )
        if mode == "partial":
            return (
                "partial",
                f"Drop to insert '{title}' after '{source_title}'. "
                f"Reconnect the desired output to '{target_title}'.",
            )
        if mode == "place":
            return (
                "place",
                f"Drop to place '{title}' in the gap. Connect ports manually.",
            )
        return "incompatible", reason

    def _connection_insert_mode(
        self,
        operation_id: str,
        connection_key,
        *,
        inserted_node_id: str | None = None,
    ) -> tuple[str, str]:
        try:
            source_id, target_id, target_port, source_port = (
                self._normalize_connection_key(connection_key)
            )
            spec = self.pipeline.operation_spec(operation_id)
        except Exception as exc:
            return "incompatible", f"Cannot insert node here: {exc}"
        if source_id not in self.pipeline.nodes or target_id not in self.pipeline.nodes:
            return "incompatible", "Cannot insert on a missing connection."

        source_ports = self.pipeline.output_ports(source_id)
        if not 0 <= source_port < len(source_ports):
            return "incompatible", "The source output no longer exists."
        target_ports = self.pipeline.input_ports(target_id)
        if not 0 <= target_port < len(target_ports):
            return "incompatible", "The target input no longer exists."

        source_type = source_ports[source_port].output_type
        target_type = target_ports[target_port].input_type
        input_ports = self._operation_insert_input_ports(
            spec,
            inserted_node_id=inserted_node_id,
        )
        compatible_inputs = [
            index
            for index, port in enumerate(input_ports)
            if self.pipeline._types_compatible(source_type, port.input_type)
        ]
        output_ports = self._operation_insert_output_ports(
            spec,
            source_type,
            inserted_node_id=inserted_node_id,
            source_id=source_id,
            source_port=source_port,
        )
        compatible_outputs = [
            index
            for index, port in enumerate(output_ports)
            if self.pipeline._types_compatible(port.output_type, target_type)
        ]
        if operation_id == "split_axis":
            axis_choices = self._connection_insert_axis_choices(
                operation_id,
                connection_key,
            )
            option_count = 0
            for axis_value, _axis_label in axis_choices:
                option_count += len(
                    self._connection_insert_mapping_options(
                        operation_id,
                        connection_key,
                        inserted_node_id=inserted_node_id,
                        params_override={"axis": axis_value},
                    )
                )
            if option_count > 1:
                return "choose", ""

        if not input_ports:
            if compatible_outputs:
                return (
                    "place",
                    "Source-like nodes can be placed in the gap, but are not "
                    "auto-wired.",
                )
            return (
                "incompatible",
                f"'{spec.title}' does not accept the upstream output.",
            )
        if not compatible_inputs:
            return (
                "incompatible",
                f"Cannot feed {source_type} output into '{spec.title}'.",
            )

        if compatible_outputs and len(compatible_inputs) * len(compatible_outputs) > 1:
            return "choose", ""

        single_input = (
            len(input_ports) == 1
            and len(compatible_inputs) == 1
            and spec.max_inputs == 1
        )
        if not single_input:
            return (
                "place",
                f"'{spec.title}' has multiple possible inputs; connect it manually.",
            )

        if len(output_ports) == 1 and compatible_outputs:
            return "full", ""
        if len(output_ports) > 1 and compatible_outputs:
            return "partial", ""
        if len(output_ports) > 1:
            return (
                "place",
                f"'{spec.title}' has multiple outputs; connect the desired output "
                "manually.",
            )
        return (
            "incompatible",
            f"'{spec.title}' output cannot feed the downstream input.",
        )

    def _operation_insert_input_ports(
        self,
        spec: OperationSpec,
        *,
        inserted_node_id: str | None = None,
    ):
        if inserted_node_id and inserted_node_id in self.pipeline.nodes:
            return self.pipeline.input_ports(inserted_node_id)
        if spec.inputs:
            return spec.input_ports
        if not spec.has_input:
            return ()
        count = 1
        if spec.max_inputs is None or spec.max_inputs != 1:
            for param in spec.parameters:
                if param.name != "input_count":
                    continue
                count = max(int(param.default), 1)
                if spec.max_inputs is not None:
                    count = min(count, max(int(spec.max_inputs), 1))
                break
        input_type = spec.input_type or "any"
        return tuple(
            InputSpec(f"input_{index + 1}", input_type, f"Input {index + 1}")
            for index in range(count)
        )

    def _operation_insert_output_ports(
        self,
        spec: OperationSpec,
        source_type: str,
        *,
        inserted_node_id: str | None = None,
        source_id: str | None = None,
        source_port: int = 0,
        params_override: dict[str, object] | None = None,
    ):
        if spec.id == "split_axis" and source_id is not None and not params_override:
            options = self._split_axis_options_from(
                self._axis_slice_options_for_output(source_id, source_port),
                fallback_all=False,
            )
            if not options:
                return ()
            params_override = {"axis": f"axis:{options[0].index}"}
        if inserted_node_id and inserted_node_id in self.pipeline.nodes:
            ports = self.pipeline.output_ports(inserted_node_id)
            if (
                spec.output_factory is not None
                and not self.pipeline.node_outputs.get(inserted_node_id)
            ):
                count = self._inferred_insert_output_count(
                    spec,
                    source_id,
                    source_port,
                    params_override=params_override,
                )
                if count is not None:
                    ports = spec.output_factory(count)
            if spec.id == "split_axis" and source_id is not None:
                ports = self._labeled_split_axis_insert_ports(
                    ports,
                    source_id,
                    source_port,
                    params_override or {},
                )
            if spec.preserves_input_type:
                return tuple(replace(port, output_type=source_type) for port in ports)
            return ports
        if spec.output_factory is not None:
            count = self._inferred_insert_output_count(
                spec,
                source_id,
                source_port,
                params_override=params_override,
            )
            if count is None:
                count = DEFAULT_DYNAMIC_OUTPUT_PORTS
            ports = spec.output_factory(count)
        else:
            ports = spec.output_ports
        if spec.id == "split_axis" and source_id is not None:
            ports = self._labeled_split_axis_insert_ports(
                ports,
                source_id,
                source_port,
                params_override or {},
            )
        if spec.preserves_input_type:
            return tuple(replace(port, output_type=source_type) for port in ports)
        return ports

    def _inferred_insert_output_count(
        self,
        spec: OperationSpec,
        source_id: str | None,
        source_port: int,
        *,
        params_override: dict[str, object] | None = None,
    ) -> int | None:
        if source_id is None:
            return None
        return self.pipeline.inferred_dynamic_output_count(
            spec.id,
            source_id,
            source_port,
            params_override,
        )

    def _insert_make_room_delta(
        self,
        source_id: str,
        target_id: str,
        inserted_node_id: str,
    ) -> QPointF:
        source_rect = self.graph_view.node_scene_rect(source_id)
        target_rect = self.graph_view.node_scene_rect(target_id)
        inserted_rect = self.graph_view.node_scene_rect(inserted_node_id)
        if source_rect is None or target_rect is None or inserted_rect is None:
            return QPointF(280.0, 0.0)
        vector = target_rect.center() - source_rect.center()
        if abs(vector.x()) >= abs(vector.y()):
            sign = -1.0 if vector.x() < 0 else 1.0
            if sign > 0:
                gap = target_rect.left() - source_rect.right()
            else:
                gap = source_rect.left() - target_rect.right()
            needed = inserted_rect.width() + (2.0 * self.INSERT_GAP_PADDING_X)
            return QPointF(sign * max(needed - gap, 0.0), 0.0)
        sign = -1.0 if vector.y() < 0 else 1.0
        if sign > 0:
            gap = target_rect.top() - source_rect.bottom()
        else:
            gap = source_rect.top() - target_rect.bottom()
        needed = inserted_rect.height() + (2.0 * self.INSERT_GAP_PADDING_Y)
        return QPointF(0.0, sign * max(needed - gap, 0.0))

    def _make_room_for_inserted_node(
        self,
        source_id: str,
        target_id: str,
        inserted_node_id: str,
        downstream: set[str],
    ) -> None:
        self.graph_view.move_nodes_by(
            downstream,
            self._insert_make_room_delta(source_id, target_id, inserted_node_id),
        )
        self._center_inserted_node_in_open_gap(
            source_id,
            target_id,
            inserted_node_id,
        )
        self._sync_graph_note_positions_from_view()

    def _center_inserted_node_in_open_gap(
        self,
        source_id: str,
        target_id: str,
        inserted_node_id: str,
    ) -> None:
        source_rect = self.graph_view.node_scene_rect(source_id)
        target_rect = self.graph_view.node_scene_rect(target_id)
        inserted_rect = self.graph_view.node_scene_rect(inserted_node_id)
        if source_rect is None or target_rect is None or inserted_rect is None:
            return

        vector = target_rect.center() - source_rect.center()
        if abs(vector.x()) >= abs(vector.y()):
            if vector.x() >= 0:
                gap_center_x = (source_rect.right() + target_rect.left()) / 2.0
            else:
                gap_center_x = (source_rect.left() + target_rect.right()) / 2.0
            center = QPointF(
                gap_center_x,
                (source_rect.center().y() + target_rect.center().y()) / 2.0,
            )
        else:
            if vector.y() >= 0:
                gap_center_y = (source_rect.bottom() + target_rect.top()) / 2.0
            else:
                gap_center_y = (source_rect.top() + target_rect.bottom()) / 2.0
            center = QPointF(
                (source_rect.center().x() + target_rect.center().x()) / 2.0,
                gap_center_y,
            )
        self.graph_view.center_node_on(inserted_node_id, center)

    def _duplicate_node(self, node_id: str) -> None:
        original = self.pipeline.nodes.get(node_id)
        if original is None:
            return
        self._finish_parameter_history_group()
        before = self._current_history_snapshot()
        clone = self.pipeline.add_node(original.operation_id)
        clone.params = deepcopy(original.params)
        source_pos = self.graph_view.node_position(node_id)
        position = (
            source_pos + QPointF(42, 42)
            if source_pos is not None
            else self.graph_view.suggest_node_position()
        )
        self.graph_view.add_node(clone, position)
        self._sync_node_input_ports(clone.id)
        self._sync_node_output_ports(clone.id)
        self._sync_pin_ui()
        self.graph_view.select_node(clone.id)
        self._push_undo_if_changed(before)
        self.status_label.setText(f"Duplicated '{original.title}' as '{clone.title}'.")

    def _inspect_node_code(self, node_id: str) -> None:
        if node_id not in self.pipeline.nodes:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(f"VIPP node code: {self._node_title(node_id)}")
        dialog.setAttribute(Qt.WA_DeleteOnClose, True)
        layout = QVBoxLayout(dialog)
        editor = QPlainTextEdit()
        editor.setReadOnly(True)
        editor.setLineWrapMode(QPlainTextEdit.NoWrap)
        editor.setPlainText(self._node_code_text(node_id))
        editor.setStyleSheet(
            "font-family: Menlo, Monaco, Consolas, monospace; font-size: 12px;"
        )
        editor._vipp_python_highlighter = PythonSyntaxHighlighter(editor.document())
        layout.addWidget(editor)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.clicked.connect(lambda _button: dialog.accept())
        layout.addWidget(buttons)
        dialog.resize(900, 700)
        dialog.finished.connect(
            lambda _result, ref=dialog: self._forget_code_dialog(ref)
        )
        self._code_dialogs.append(dialog)
        dialog.show()
        self.status_label.setText(f"Opened code for '{self._node_title(node_id)}'.")

    def _forget_code_dialog(self, dialog: QDialog) -> None:
        if dialog in self._code_dialogs:
            self._code_dialogs.remove(dialog)

    def _node_code_text(self, node_id: str) -> str:
        node = self.pipeline.nodes[node_id]
        spec = self.pipeline.operation_spec(node.operation_id)
        lines = [
            f"# VIPP node: {node.title}",
            f"# Node id: {node.id}",
            f"# Operation id: {node.operation_id}",
            f"# Category: {node.category}",
            "",
        ]
        if node.params:
            lines.append("params = {")
            for key, value in node.params.items():
                lines.append(f"    {key!r}: {value!r},")
            lines.append("}")
            lines.append("")
        else:
            lines.extend(["params = {}", ""])

        connections = self.pipeline._input_connections(node_id)
        if connections:
            lines.append("# Connected inputs")
            for index, connection in enumerate(connections, start=1):
                lines.append(
                    f"# {index}. {connection.source_id} output "
                    f"{connection.source_port} -> input {connection.target_port}"
                )
            lines.append("")

        function = spec.function
        if function is None:
            lines.append(
                "# This node is handled by the VIPP runtime and does not map "
                "to a single pure operation function."
            )
            return "\n".join(lines).rstrip() + "\n"

        signature = py_inspect.signature(function)
        accepted = list(signature.parameters)
        first_arg = self._node_code_first_arg(node, connections)
        kwargs = {
            key: value
            for key, value in node.params.items()
            if key in accepted and (not accepted or key != accepted[0])
        }
        kwargs_text = ", ".join(f"{key}={value!r}" for key, value in kwargs.items())
        call_args = first_arg
        if kwargs_text:
            call_args = f"{first_arg}, {kwargs_text}"

        lines.extend(
            [
                f"from {function.__module__} import {function.__name__}",
                "",
                "# Single-node call shape",
                f"output = {function.__name__}({call_args})",
                "",
                "# Operation function source",
            ]
        )
        try:
            lines.append(textwrap.dedent(py_inspect.getsource(function)).rstrip())
        except OSError:
            lines.append("# Source is not available for this operation function.")
        return "\n".join(lines).rstrip() + "\n"

    def _node_code_first_arg(self, node, connections) -> str:
        if not connections:
            if node.max_inputs is None or node.max_inputs != 1:
                return "[]"
            return "input_data"
        if node.max_inputs is None or node.max_inputs != 1:
            refs = [
                self._node_code_source_ref(connection) for connection in connections
            ]
            return f"[{', '.join(refs)}]"
        return self._node_code_source_ref(connections[0])

    @staticmethod
    def _node_code_source_ref(connection) -> str:
        ref = f"{connection.source_id}_output"
        if int(getattr(connection, "source_port", 0)) != 0:
            ref = f"{ref}[{int(connection.source_port)}]"
        return ref

    def _new_workflow_dialog(self) -> None:
        answer = QMessageBox.question(
            self,
            "New workflow",
            "Create a new empty workflow? This will erase the current graph and "
            "any unsaved changes.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        self._new_workflow()

    def _new_workflow(self) -> None:
        self._finish_parameter_history_group()
        self._interactive_collection_source_paths.clear()
        before = self._current_history_snapshot()
        self.pipeline.reset_empty_graph()
        self._preview_disabled_node_ids.clear()
        self._rescale_auto_output_ranges.clear()
        self._graph_notes.clear()
        self._clear_active_pin(status=False)
        self._build_graph_from_pipeline()
        self._select_node("input")
        self._invalidate_pipeline_cache()
        self.run_pipeline()
        self._push_undo_if_changed(before)
        self.status_label.setText("New empty workflow created.")

    def _build_graph_from_pipeline(self) -> None:
        self.graph_view.build_graph(
            self.pipeline.nodes.values(),
            self.pipeline.connections,
            output_tunnels=self.pipeline.output_tunnel_list(),
            notes=self._graph_note_documents(use_view_positions=False),
        )
        self._sync_pin_ui()
        self._sync_all_input_ports()
        self._sync_all_output_ports()
        self._refresh_graph_search_matches(reset_index=True)

    def _workflow_metadata(self) -> dict:
        valid_node_ids = set(self.pipeline.nodes)
        inspector: dict[str, object] = {
            "right_panel_visible": (
                not self.inspector_panel.isHidden()
                if hasattr(self, "inspector_panel")
                else True
            ),
        }
        if self._selected_node_id in valid_node_ids:
            inspector["selected_node_id"] = self._selected_node_id

        vipp: dict[str, object] = {"inspector": inspector}
        if self.save_thumbnail_visibility_checkbox.isChecked():
            vipp["thumbnails"] = {
                "disabled_node_ids": sorted(
                    self._preview_disabled_node_ids & valid_node_ids
                )
            }
        return {"vipp": vipp}

    @staticmethod
    def _workflow_vipp_metadata(workflow: dict) -> dict:
        metadata = workflow.get("metadata", {})
        if not isinstance(metadata, dict):
            return {}
        vipp = metadata.get("vipp", {})
        return vipp if isinstance(vipp, dict) else {}

    def _save_workflow_dialog(self) -> None:
        path, _filter = QFileDialog.getSaveFileName(
            self,
            "Save VIPP workflow",
            "vipp_workflow.json",
            "VIPP workflow (*.json);;All files (*.*)",
        )
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        try:
            positions = self.graph_view.node_positions()
            saved = save_workflow(
                path,
                self.pipeline,
                positions,
                self._graph_note_documents(),
                self._workflow_metadata(),
            )
        except Exception as exc:
            self.status_label.setText(f"Save failed: {exc}")
            return
        self.status_label.setText(f"Workflow saved to {saved.name}.")

    def _load_workflow_dialog(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self,
            "Load VIPP workflow",
            "",
            "VIPP workflow (*.json);;All files (*.*)",
        )
        if not path:
            return
        try:
            loaded = self.load_workflow_file(path)
        except Exception as exc:
            self.status_label.setText(f"Load failed: {exc}")
            return
        self.status_label.setText(f"Loaded workflow from {loaded.name}.")

    def _open_example_workflow_dialog(self) -> None:
        dialog = ExampleWorkflowDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return
        example = dialog.selected_example()
        if example is None:
            return
        try:
            if example.generated_batch_demo:
                demo = self._choose_collection_batch_demo()
                if demo is not None:
                    self._batch_collection_dialog(config_path=demo.config_path)
            else:
                self.load_example_workflow(example.id)
        except Exception as exc:
            self.status_label.setText(f"Open example failed: {exc}")

    def load_example_workflow(self, example_id: str) -> Path:
        example = _example_workflow_by_id(example_id)
        if example is None:
            raise ValueError(f"Unknown example workflow: {example_id!r}.")
        path = _example_workflow_path(example)
        if not path.exists():
            raise FileNotFoundError(
                f"Example workflow file is not available: {path}"
            )
        loaded = self.load_workflow_file(path)
        self.status_label.setText(f"Opened example workflow: {example.title}.")
        return loaded

    def load_workflow_file(self, path: str | Path) -> Path:
        """Load a workflow file into the widget and recompute the graph."""
        self._finish_parameter_history_group()
        self._interactive_collection_source_paths.clear()
        before = self._current_history_snapshot()
        source = Path(path).expanduser()
        workflow = load_workflow(source)
        self.pipeline.restore_graph(
            workflow["nodes"],
            workflow["connections"],
            workflow.get("output_tunnels", ()),
        )
        valid_node_ids = set(self.pipeline.nodes)
        vipp_metadata = self._workflow_vipp_metadata(workflow)
        thumbnail_metadata = vipp_metadata.get("thumbnails")
        if isinstance(thumbnail_metadata, dict):
            self._preview_disabled_node_ids = set(
                thumbnail_metadata.get("disabled_node_ids", ())
            ) & valid_node_ids
        else:
            self._preview_disabled_node_ids.clear()
        inspector_metadata = vipp_metadata.get("inspector")
        if not isinstance(inspector_metadata, dict):
            inspector_metadata = {}
        selected_node_id = str(inspector_metadata.get("selected_node_id", "") or "")
        if selected_node_id not in valid_node_ids:
            selected_node_id = ""
        right_panel_visible = inspector_metadata.get("right_panel_visible", None)
        self._rescale_auto_output_ranges.clear()
        self._restore_graph_notes(workflow.get("notes", ()))
        self._clear_active_pin(status=False)
        self.graph_view.build_graph(
            self.pipeline.nodes.values(),
            self.pipeline.connections,
            workflow["positions"],
            output_tunnels=self.pipeline.output_tunnel_list(),
            notes=self._graph_note_documents(use_view_positions=False),
        )
        self._sync_pin_ui()
        self._sync_all_input_ports()
        self._sync_all_output_ports()
        self._refresh_graph_search_matches(reset_index=True)
        self._invalidate_pipeline_cache()
        self.run_pipeline()
        if selected_node_id:
            self.graph_view.select_node(selected_node_id)
        else:
            self._select_first_available_node()
        if isinstance(right_panel_visible, bool):
            self._set_right_panel_visible(right_panel_visible)
        self._push_undo_if_changed(before)
        return source

    def _export_python_dialog(self) -> None:
        path, _filter = QFileDialog.getSaveFileName(
            self,
            "Export pipeline to Python",
            "vipp_pipeline.py",
            "Python script (*.py);;All files (*.*)",
        )
        if not path:
            return
        if not path.lower().endswith(".py"):
            path += ".py"
        try:
            code = export_pipeline_to_python(self.pipeline)
            target = Path(path).expanduser()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(code, encoding="utf-8")
        except Exception as exc:
            self.status_label.setText(f"Export failed: {exc}")
            return
        self.status_label.setText(f"Pipeline exported to {target.name}.")

    def _batch_collection_dialog(
        self,
        *,
        config_path: str | Path | None = None,
    ) -> None:
        dialog = CollectionBatchDialog(self, source_nodes=self._batch_source_rows())
        if config_path is not None:
            try:
                config = self._load_collection_batch_config(config_path)
                dialog._apply_config(config)
                dialog._loaded_config_path = Path(config_path)
                config_root = Path(config_path).expanduser().resolve().parent
                if (
                    config_root / SYNTHETIC_BATCH_GROUND_TRUTH_FILENAME
                ).is_file():
                    dialog.set_demo_context(
                        SyntheticBatchDemo.from_root(config_root)
                    )
                dialog._preview_batch()
            except Exception as exc:
                self.status_label.setText(f"Could not open batch config: {exc}")
                return
        if dialog.exec() != QDialog.Accepted:
            return
        values = dialog.values()
        try:
            result = self._run_collection_batch(**values)
        except Exception as exc:
            self._set_pipeline_busy(False)
            self.status_label.setText(f"Batch failed: {exc}")
            return
        validation_text = self._validate_collection_batch_demo_result(
            dialog._loaded_config_path,
            result,
        )
        summary = result.summary
        self.status_label.setText(
            f"Batch finished: {summary['completed']} completed, "
            f"{summary['partial']} partial, {summary['skipped']} skipped, "
            f"{summary['failed']} failed; {len(result.saved_paths)} output(s) "
            f"saved. Manifest: {result.manifest_path}."
            + (f" {validation_text}" if validation_text else "")
        )
        self._show_collection_batch_summary(result, validation_text)

    def _choose_collection_batch_demo(
        self,
        *,
        dialog_parent: QWidget | None = None,
    ) -> SyntheticBatchDemo | None:
        owner = dialog_parent or self
        answer = QMessageBox.question(
            owner,
            "Open deterministic batch demo",
            "Open a ready-to-run batch demo? VIPP will prepare a small "
            "self-contained working copy, load its two-source workflow, and "
            "open the batch window already configured and previewed. This "
            "replaces the current graph even if you later cancel the batch "
            "dialog, so save any workflow changes you want to keep first.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return None
        parent_path = QFileDialog.getExistingDirectory(
            owner,
            "Choose where to save the batch demo working copy",
            "",
        )
        if not parent_path:
            return None
        target = next_synthetic_batch_demo_root(parent_path)
        return self._create_collection_batch_demo(target)

    def _create_collection_batch_demo(
        self,
        root: str | Path,
    ) -> SyntheticBatchDemo:
        demo = create_synthetic_batch_demo(root)
        self.load_workflow_file(demo.workflow_path)
        preview_paths = self._load_collection_batch_demo_preview(demo)
        preview_names = ", ".join(path.name for path in preview_paths)
        self.status_label.setText(
            f"Opened deterministic batch demo at {demo.root}. "
            f"The graph is previewing the first paired item ({preview_names})."
        )
        return demo

    def _load_collection_batch_demo_preview(
        self,
        demo: SyntheticBatchDemo,
    ) -> tuple[Path, ...]:
        """Load the first planned pair without changing collection bindings."""
        config = load_batch_config(demo.config_path)
        workflow = self._batch_workflow_document()
        plan = plan_batch(
            workflow,
            config,
            workflow_path=demo.workflow_path,
        )
        if not plan.items:
            raise ValueError("The batch demo does not contain any input items.")
        first_item = plan.items[0]
        self._interactive_collection_source_paths = {
            node_id: Path(path)
            for node_id, path in first_item.source_paths.items()
            if node_id in self.pipeline.nodes
        }
        if not self._interactive_collection_source_paths:
            raise ValueError(
                "The batch demo did not resolve any interactive Image Sources."
            )
        self._invalidate_pipeline_cache()
        self.run_pipeline(
            force_sync=True,
            manual_node_ids=self.pipeline.manual_node_ids(),
        )
        return tuple(self._interactive_collection_source_paths.values())

    def _validate_collection_batch_demo_result(
        self,
        config_path: Path | None,
        result: BatchRunResult,
    ) -> str:
        if config_path is None:
            return ""
        root = Path(config_path).expanduser().resolve().parent
        if not (root / SYNTHETIC_BATCH_GROUND_TRUTH_FILENAME).is_file():
            return ""
        try:
            validation = validate_synthetic_batch_demo(root, result=result)
        except Exception as exc:
            return f"Synthetic ground-truth validation failed: {exc}"
        return f"Synthetic ground truth passed ({len(validation.checks)} checks)."

    def _show_collection_batch_summary(
        self,
        result: BatchRunResult,
        validation_text: str = "",
    ) -> None:
        summary = result.summary
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Collection batch summary")
        validation_failed = "validation failed" in validation_text.lower()
        dialog.setIcon(
            QMessageBox.Warning
            if result.has_failures or validation_failed
            else QMessageBox.Information
        )
        dialog.setText(
            f"{summary['completed']} completed, {summary['partial']} partial, "
            f"{summary['skipped']} skipped, and {summary['failed']} failed."
        )
        dialog.setInformativeText(
            f"{len(result.saved_paths)} output(s) saved.\n"
            f"Manifest: {result.manifest_path}"
            + (f"\n{validation_text}" if validation_text else "")
        )
        exceptional = [
            item
            for item in result.manifest.items
            if item.status.value in {"partial", "failed", "skipped"}
        ]
        if exceptional:
            details = []
            for item in exceptional:
                details.append(
                    f"{item.batch_id}: {item.status.value}"
                    + (f" - {item.error_message}" if item.error_message else "")
                )
                details.extend(
                    f"  {output.tag}: {output.status.value} - {output.path}"
                    + (
                        f" - {output.error_message}"
                        if output.error_message
                        else ""
                    )
                    for output in item.outputs
                    if output.status.value != "completed"
                )
            dialog.setDetailedText("\n".join(details))
        dialog.exec()

    def _run_collection_batch(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        pattern: str = "*.tif",
        image_format: str = "ome-tiff",
        save_workflow_snapshot: bool = True,
        save_python_script: bool = True,
        source_bindings: list[dict] | None = None,
        existing_file_policy: str = ExistingFilePolicy.ERROR.value,
        continue_on_error: bool = True,
    ) -> BatchRunResult:
        del save_workflow_snapshot
        positions = self.graph_view.node_positions()
        workflow = self._batch_workflow_document(positions)
        config = self._collection_batch_config(
            input_dir=input_dir,
            output_dir=output_dir,
            pattern=pattern,
            image_format=image_format,
            save_workflow_snapshot=True,
            save_python_script=save_python_script,
            source_bindings=source_bindings,
            existing_file_policy=existing_file_policy,
            continue_on_error=continue_on_error,
            workflow=workflow,
        )
        output_path = config.resolve_path(config.output_dir)
        config_path = output_path / BATCH_CONFIG_FILENAME
        workflow_path = output_path / BATCH_WORKFLOW_FILENAME
        script_path = output_path / BATCH_SCRIPT_FILENAME
        plan = preflight_batch(workflow, config, workflow_path=workflow_path)
        output_path.mkdir(parents=True, exist_ok=True)
        artifact_paths: list[Path] = [
            atomic_write_json(workflow_path, workflow)
        ]
        if save_python_script:
            atomic_write_text(
                script_path,
                export_batch_runner_to_python(),
            )
            artifact_paths.append(script_path)
        save_batch_config(config_path, config)

        self._set_pipeline_busy(True, cancelable=False)
        try:
            result = run_batch(
                workflow,
                config,
                workflow_path=workflow_path,
                config_path=config_path,
                plan=plan,
                progress_callback=self._collection_batch_progress,
            )
        finally:
            self._set_pipeline_busy(False)
        return replace(
            result,
            artifact_paths=tuple(artifact_paths),
        )

    def _collection_batch_progress(
        self,
        index: int,
        total: int,
        batch_id: str,
        status: str,
    ) -> None:
        self.status_label.setText(f"Batch {index}/{total}: {batch_id} ({status}).")
        QApplication.processEvents()

    def _batch_workflow_document(
        self,
        positions: dict[str, tuple[float, float]] | None = None,
    ) -> dict:
        workflow = serialize_workflow(
            self.pipeline,
            positions or self.graph_view.node_positions(),
            self._graph_note_documents(),
            self._workflow_metadata(),
        )
        for node in workflow.get("nodes", ()):
            if node.get("operation_id") != "input":
                continue
            params = node.get("params", {})
            if str(params.get("source_mode", "napari layer")) != "file path":
                continue
            raw_path = str(params.get("file_path", "")).strip()
            if raw_path:
                params["file_path"] = str(Path(raw_path).expanduser().resolve())
        return workflow

    def _collection_batch_config(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        pattern: str = "*.tif",
        image_format: str = "ome-tiff",
        save_workflow_snapshot: bool = True,
        save_python_script: bool = True,
        source_bindings: list[dict] | None = None,
        existing_file_policy: str = ExistingFilePolicy.ERROR.value,
        continue_on_error: bool = True,
        workflow: dict | None = None,
    ) -> BatchConfig:
        input_text = str(input_dir).strip()
        output_text = str(output_dir).strip()
        if not output_text:
            raise ValueError("Batch output folder cannot be blank.")
        input_path = Path(input_text).expanduser()
        if workflow is None:
            workflow = self._batch_workflow_document()
        restored = deserialize_workflow(workflow)
        batch_pipeline = PrototypePipeline()
        batch_pipeline.restore_graph(
            restored["nodes"],
            restored["connections"],
            restored.get("output_tunnels", ()),
        )
        output_node_ids = self._batch_saved_node_ids(batch_pipeline)
        if not output_node_ids:
            raise ValueError("The workflow has no outputs to save.")
        bindings = self._normalize_batch_source_bindings(
            batch_pipeline,
            input_path,
            pattern,
            source_bindings,
        )
        sources = tuple(
            BatchSourceConfig(
                node_id=binding.node_id,
                title=binding.title,
                input_dir=Path(binding.input_dir or "").expanduser().resolve(),
                pattern=binding.pattern,
            )
            for binding in bindings
        )
        outputs = []
        for node_id in output_node_ids:
            node = batch_pipeline.nodes[node_id]
            params = node.params if node.operation_id == "batch_output" else {}
            ports = batch_pipeline.output_ports(node_id)
            output_type = ports[0].output_type if ports else "any"
            outputs.append(
                BatchOutputConfig(
                    node_id=node_id,
                    node_title=node.title,
                    tag=self._batch_output_tag(batch_pipeline, node_id),
                    kind="table" if output_type == "table" else "image",
                    format=str(params.get("format", "batch default")),
                    subfolder=str(params.get("subfolder", "")),
                    filename_template=str(
                        params.get(
                            "filename_template",
                            "{source_stem}__{tag}",
                        )
                    ),
                    overwrite=str(params.get("overwrite", "batch default")),
                )
            )
        try:
            policy = ExistingFilePolicy(str(existing_file_policy))
        except ValueError as exc:
            raise ValueError(
                f"Unsupported existing-file policy: {existing_file_policy!r}."
            ) from exc
        config = BatchConfig(
            workflow_file=Path(BATCH_WORKFLOW_FILENAME),
            workflow_sha256=scientific_workflow_hash(workflow),
            output_dir=Path(output_text).expanduser().resolve(),
            sources=sources,
            outputs=tuple(outputs),
            default_image_format=image_format,
            existing_file_policy=policy,
            save_workflow_snapshot=True,
            save_python_script=save_python_script,
            continue_on_error=continue_on_error,
        )
        validate_batch_config(
            workflow,
            config,
            workflow_path=(config.output_dir / BATCH_WORKFLOW_FILENAME),
        )
        return config

    def _save_collection_batch_config(
        self,
        path: str | Path,
        **values,
    ) -> tuple[Path, Path]:
        target = Path(path).expanduser()
        reserved = {
            BATCH_WORKFLOW_FILENAME.casefold(),
            BATCH_MANIFEST_FILENAME.casefold(),
        }
        if target.name.casefold() in reserved:
            raise ValueError(
                f"Choose a config filename other than {target.name!r}; that "
                "name is reserved for a batch companion artifact."
            )
        workflow = self._batch_workflow_document()
        config = self._collection_batch_config(**values, workflow=workflow)
        workflow_path = target.parent / BATCH_WORKFLOW_FILENAME
        validate_batch_config(workflow, config, workflow_path=workflow_path)
        saved_workflow = atomic_write_json(workflow_path, workflow)
        saved_config = save_batch_config(target, config)
        return saved_config, saved_workflow

    def _load_collection_batch_config(self, path: str | Path) -> BatchConfig:
        config = load_batch_config(path)
        workflow = self._batch_workflow_document()
        try:
            validate_batch_config(
                workflow,
                config,
                workflow_path=config.resolve_path(config.workflow_file),
            )
        except ValueError as exc:
            if "workflow hash" in str(exc):
                raise ValueError(
                    "This config belongs to a different workflow. Load its saved "
                    "workflow before applying the batch config."
                ) from exc
            raise
        return config

    def _preview_collection_batch(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        pattern: str = "*.tif",
        image_format: str = "ome-tiff",
        save_workflow_snapshot: bool = True,
        save_python_script: bool = True,
        source_bindings: list[dict] | None = None,
        preview_limit: int = 25,
        existing_file_policy: str = ExistingFilePolicy.ERROR.value,
        continue_on_error: bool = True,
    ) -> BatchPreviewResult:
        workflow = self._batch_workflow_document()
        config = self._collection_batch_config(
            input_dir=input_dir,
            output_dir=output_dir,
            pattern=pattern,
            image_format=image_format,
            save_workflow_snapshot=save_workflow_snapshot,
            save_python_script=save_python_script,
            source_bindings=source_bindings,
            existing_file_policy=existing_file_policy,
            continue_on_error=continue_on_error,
            workflow=workflow,
        )
        plan = plan_batch(
            workflow,
            config,
            workflow_path=(config.output_dir / BATCH_WORKFLOW_FILENAME),
        )
        explicit = bool(self._batch_output_node_ids(self.pipeline))
        rows = tuple(
            BatchPreviewRow(
                batch_index=item.index,
                batch_id=item.batch_id,
                sources=dict(item.source_paths),
                outputs=[output.path for output in item.outputs],
                output_statuses=tuple(output.status_text for output in item.outputs),
                explicit_outputs=explicit,
            )
            for item in plan.items[: max(int(preview_limit), 0)]
        )
        collision_count = sum(
            output.duplicate
            or output.input_collision
            or (
                output.exists
                and output.existing_file_policy == ExistingFilePolicy.ERROR
            )
            for item in plan.items
            for output in item.outputs
        )
        return BatchPreviewResult(
            rows=rows,
            total_items=len(plan.items),
            collision_count=collision_count,
            explicit_outputs=explicit,
        )

    def _batch_source_rows(self) -> list[dict[str, str]]:
        rows = []
        for node_id in self.pipeline.topological_order():
            node = self.pipeline.nodes[node_id]
            if node.operation_id != "input":
                continue
            rows.append(
                {
                    "node_id": node_id,
                    "title": node.title,
                    "binding_mode": str(
                        node.params.get("binding_mode", "single item")
                    ),
                }
            )
        return rows or [
            {
                "node_id": "input",
                "title": "Image Source",
                "binding_mode": "collection",
            }
        ]

    def _normalize_batch_source_bindings(
        self,
        pipeline: PrototypePipeline,
        input_dir: Path,
        pattern: str,
        source_bindings: list[dict] | None,
    ) -> list[BatchSourceBinding]:
        bindings: list[BatchSourceBinding] = []
        if source_bindings is not None:
            for row in source_bindings:
                node_id = str(row.get("node_id", "")).strip()
                if node_id not in pipeline.nodes:
                    continue
                node = pipeline.nodes[node_id]
                if node.operation_id != "input":
                    continue
                raw_dir = str(row.get("input_dir", "")).strip()
                if not raw_dir:
                    continue
                bindings.append(
                    BatchSourceBinding(
                        node_id=node_id,
                        title=str(row.get("title", node.title) or node.title),
                        input_dir=Path(raw_dir).expanduser(),
                        pattern=str(row.get("pattern", "") or pattern or "*.tif"),
                    )
                )
        if bindings:
            return bindings
        if source_bindings is not None:
            raise ValueError("At least one batch source needs an input folder.")

        source_ids = self._batch_collection_source_node_ids(pipeline)
        if not input_dir.is_dir():
            raise ValueError("Batch input folder does not exist.")
        return [
            BatchSourceBinding(
                node_id=node_id,
                title=pipeline.nodes[node_id].title,
                input_dir=input_dir,
                pattern=pattern or "*.tif",
            )
            for node_id in source_ids
        ]

    def _batch_collection_source_node_ids(
        self,
        pipeline: PrototypePipeline,
    ) -> set[str]:
        source_ids = [
            node_id
            for node_id in pipeline.topological_order()
            if pipeline.nodes[node_id].operation_id == "input"
        ]
        collection_ids = {
            node_id
            for node_id in source_ids
            if str(
                pipeline.nodes[node_id].params.get("binding_mode", "single item")
            )
            == "collection"
        }
        if collection_ids:
            return collection_ids
        return {source_ids[0]} if source_ids else set()

    def _terminal_node_ids(self, pipeline: PrototypePipeline) -> list[str]:
        order = pipeline.topological_order()
        consumed = {connection.source_id for connection in pipeline.connections}
        terminals = [node_id for node_id in order if node_id not in consumed]
        return terminals or list(order)

    def _batch_output_node_ids(self, pipeline: PrototypePipeline) -> list[str]:
        return [
            node_id
            for node_id in pipeline.topological_order()
            if pipeline.nodes[node_id].operation_id == "batch_output"
        ]

    def _batch_saved_node_ids(self, pipeline: PrototypePipeline) -> list[str]:
        explicit = self._batch_output_node_ids(pipeline)
        return explicit if explicit else self._terminal_node_ids(pipeline)

    def _batch_output_tag(self, pipeline: PrototypePipeline, node_id: str) -> str:
        node = pipeline.nodes[node_id]
        if node.operation_id == "batch_output":
            raw = str(node.params.get("tag", "")).strip()
            return safe_batch_filename(raw or node_id)
        return safe_batch_filename(f"{node.title}-{node_id}")

    def _export_ome_dataset_dialog(self) -> None:
        reference_id = self._default_analysis_reference_node()
        labels = self._analysis_label_outputs()
        if reference_id is None:
            self.status_label.setText(
                "OME dataset export needs a reference image output."
            )
            return
        if not labels:
            self.status_label.setText(
                "OME dataset export needs at least one label output."
            )
            return
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export OME analysis dataset",
            "vipp_analysis.ome.zarr",
            "OME-Zarr 0.4 (*.ome.zarr);;OME-Zarr 0.5 (*.ome.zarr)",
        )
        if not path:
            return
        if not path.lower().endswith(".zarr"):
            path += ".ome.zarr"
        version = "0.5" if selected_filter.startswith("OME-Zarr 0.5") else "0.4"
        try:
            output_path = write_ome_zarr_analysis_dataset(
                self.pipeline.outputs[reference_id],
                Path(path).expanduser(),
                labels=tuple(labels),
                version=version,
                image_state=self.pipeline.output_states.get(reference_id),
            )
        except Exception as exc:
            self.status_label.setText(f"OME dataset export failed: {exc}")
            return
        self.status_label.setText(
            f"Exported OME analysis dataset with {len(labels)} label output(s) "
            f"to {output_path.name}."
        )

    def _default_analysis_reference_node(self) -> str | None:
        if (
            self.pipeline.outputs.get("input") is not None
            and self._data_kind(self.pipeline.outputs["input"], "input") == "image"
        ):
            return "input"
        for node_id in self.pipeline.topological_order():
            data = self.pipeline.outputs.get(node_id)
            if data is not None and self._data_kind(data, node_id) == "image":
                return node_id
        return None

    def _analysis_label_outputs(self) -> list[AnalysisLabel]:
        labels: list[AnalysisLabel] = []
        used_names: set[str] = set()
        for node_id in self.pipeline.topological_order():
            data = self.pipeline.outputs.get(node_id)
            if data is None or self._data_kind(data, node_id) != "labels":
                continue
            name = self._unique_export_label_name(self._node_title(node_id), used_names)
            labels.append(
                AnalysisLabel(
                    name=name,
                    data=data,
                    image_state=self.pipeline.output_states.get(node_id),
                    source_node_id=node_id,
                )
            )
        return labels

    @staticmethod
    def _unique_export_label_name(name: str, used_names: set[str]) -> str:
        base = "".join(
            character if character.isalnum() else "_"
            for character in name.strip().lower()
        ).strip("_")
        base = base or "labels"
        candidate = base
        suffix = 2
        while candidate in used_names:
            candidate = f"{base}_{suffix}"
            suffix += 1
        used_names.add(candidate)
        return candidate

    def _refresh_and_run(self) -> None:
        self._invalidate_pipeline_cache()
        self._autobind_default_image_sources()
        self._refresh_image_source_controls()
        self.run_pipeline()

    def _mark_pipeline_dirty(self, node_id: str) -> bool:
        return self._mark_pipeline_branches_dirty({node_id})

    def _mark_pipeline_branches_dirty(self, node_ids) -> bool:
        valid_node_ids = {
            str(node_id) for node_id in node_ids if str(node_id) in self.pipeline.nodes
        }
        if not valid_node_ids:
            return False
        self._clear_colocalization_scatter_cache()
        self._pending_dirty_node_ids.update(valid_node_ids)
        self.pipeline.mark_manual_descendants_stale(valid_node_ids)
        self._sync_execution_ui()
        return True

    def _invalidate_pipeline_cache(self) -> None:
        self._pending_dirty_node_ids.clear()
        self._pending_manual_node_ids.clear()
        self._inflight_dirty_node_ids = None
        self._last_pipeline_source_signature = None
        self._clear_thumbnail_contrast_limit_state()
        self._clear_input_histogram_cache()
        self._clear_output_histogram_cache()
        self._label_volume_cache.clear()
        self._clear_colocalization_scatter_cache()
        self._clear_generated_layer_contrast_state()
        self.pipeline.completed_node_ids.clear()
        self.pipeline.mark_manual_descendants_stale(self.pipeline.nodes)
        self._sync_execution_ui()

    def _begin_pipeline_dispatch(self, dirty_node_ids: set[str] | None) -> None:
        """Mark a run as in flight and clear the dirty nodes it covers.

        Removing the dispatched dirty nodes from ``_pending_dirty_node_ids`` here
        (rather than on completion) means any edit that re-marks the same node
        while the run is in flight is preserved as genuinely pending work, and a
        later debounced run will not see an empty pending set and recompute the
        whole pipeline from the source.
        """
        self._inflight_dirty_node_ids = (
            None if dirty_node_ids is None else set(dirty_node_ids)
        )
        if dirty_node_ids is not None:
            self._pending_dirty_node_ids.difference_update(dirty_node_ids)

    def _requeue_inflight_dirty_nodes(self) -> None:
        """Return a discarded run's dirty nodes to the pending set for a rerun."""
        if self._inflight_dirty_node_ids:
            self._pending_dirty_node_ids.update(
                self._inflight_dirty_node_ids & set(self.pipeline.nodes)
            )
        self._inflight_dirty_node_ids = None

    def _pipeline_source_signature(
        self,
        input_data,
        input_metadata,
        input_name: str,
        source_payloads: dict[str, SourcePayload],
    ) -> tuple:
        if source_payloads:
            return (
                "sources",
                tuple(
                    sorted(
                        (
                            node_id,
                            payload.name,
                            id(payload.data),
                            id(payload.metadata),
                        )
                        for node_id, payload in source_payloads.items()
                    )
                ),
            )
        return ("sources", ())

    def _dirty_nodes_for_run(self, source_signature: tuple) -> set[str] | None:
        if source_signature != self._last_pipeline_source_signature:
            self._pending_dirty_node_ids.clear()
            return None
        dirty_nodes = self._pending_dirty_node_ids & set(self.pipeline.nodes)
        return set(dirty_nodes) if dirty_nodes else None

    def _complete_pipeline_run(
        self,
        source_signature: tuple,
        dirty_node_ids: set[str] | None,
    ) -> None:
        # The dispatched dirty nodes were already cleared from
        # ``_pending_dirty_node_ids`` when the run started (see
        # ``_begin_pipeline_dispatch``). Anything still pending here arrived
        # while this run was in flight and must survive for the next run.
        self._last_pipeline_source_signature = source_signature
        self._inflight_dirty_node_ids = None

    def _cache_mode(self) -> str:
        mode = self.cache_mode_combo.currentText()
        return mode if mode in CACHE_MODE_CHOICES else CACHE_MODE_KEEP_ALL

    def _cache_pruning_enabled(self) -> bool:
        return self._cache_mode() != CACHE_MODE_KEEP_ALL

    def _on_cache_mode_changed(self, _mode: str) -> None:
        self._memory_guard_dialog_shown = False
        self._apply_cache_retention()
        self._refresh_dynamic_output_ports()
        self._update_thumbnails()
        self._refresh_inspection_layer_if_active()
        self._refresh_pinned_layer_if_active()
        self._update_metadata_panel()
        self._update_histogram()
        self._sync_execution_ui()
        self._refresh_cache_status()
        self.status_label.setText(f"Cache mode set to {self._cache_mode()}.")

    def _on_memory_guard_setting_changed(self, *_args) -> None:
        self._memory_guard_dialog_shown = False
        message = self._enforce_memory_guard()
        self._refresh_cache_status()
        if message:
            self.status_label.setText(message)

    def _remember_cache_node(self, node_id: str) -> None:
        if node_id not in self.pipeline.nodes:
            return
        self._recent_cache_node_ids = [
            recent_id
            for recent_id in self._recent_cache_node_ids
            if recent_id != node_id and recent_id in self.pipeline.nodes
        ]
        self._recent_cache_node_ids.insert(0, node_id)
        del self._recent_cache_node_ids[SMART_CACHE_RECENT_LIMIT:]

    def _apply_cache_retention(self) -> None:
        if not self._cache_pruning_enabled():
            self._refresh_cache_status()
            return
        self.pipeline.prune_cached_outputs(self._cache_retention_node_ids())
        self._refresh_cache_status()

    def _cache_retention_node_ids(self, mode: str | None = None) -> set[str]:
        mode = mode or self._cache_mode()
        valid = set(self.pipeline.nodes)
        if mode == CACHE_MODE_KEEP_ALL:
            return set(valid)

        working_nodes = self._current_working_cache_nodes()
        retained = (
            working_nodes
            | self._direct_input_cache_nodes(working_nodes)
            | self._explicit_output_nodes()
            | self._keep_cached_node_ids()
        )
        if mode == CACHE_MODE_SMART:
            retained.update(self._source_cache_nodes())
            retained.update(self._branch_point_cache_nodes())
            recent_nodes = {
                node_id
                for node_id in self._recent_cache_node_ids
                if node_id in self.pipeline.nodes
            }
            retained.update(
                recent_nodes | self._direct_input_cache_nodes(recent_nodes)
            )
        return retained & valid

    def _current_working_cache_nodes(self) -> set[str]:
        nodes = set()
        if self._selected_node_id in self.pipeline.nodes:
            nodes.add(self._selected_node_id)
        if self._active_pinned_node_id in self.pipeline.nodes:
            nodes.add(str(self._active_pinned_node_id))
        return nodes

    def _direct_input_cache_nodes(self, node_ids: set[str]) -> set[str]:
        return {
            source_id
            for node_id in node_ids
            for source_id in self.pipeline._input_sources(node_id)
            if source_id in self.pipeline.nodes
        }

    def _source_cache_nodes(self) -> set[str]:
        return {
            node_id
            for node_id, node in self.pipeline.nodes.items()
            if node.operation_id == "input"
        }

    def _explicit_output_nodes(self) -> set[str]:
        return {
            node_id
            for node_id, node in self.pipeline.nodes.items()
            if node.operation_id in EXPLICIT_OUTPUT_OPERATIONS
        }

    def _keep_cached_node_ids(self) -> set[str]:
        return {
            node_id
            for node_id, node in self.pipeline.nodes.items()
            if bool(node.params.get(CACHE_KEEP_NODE_PARAM, False))
        }

    def _branch_point_cache_nodes(self) -> set[str]:
        outgoing_counts: dict[str, int] = {}
        for connection in self.pipeline.connections:
            outgoing_counts[connection.source_id] = (
                outgoing_counts.get(connection.source_id, 0) + 1
            )
        for tunnel in self.pipeline.output_tunnel_list():
            outgoing_counts[tunnel.source_id] = max(
                outgoing_counts.get(tunnel.source_id, 0),
                1,
            )
        return {
            node_id
            for node_id, count in outgoing_counts.items()
            if count > 1 and node_id in self.pipeline.nodes
        }

    def _enforce_memory_guard(self) -> str:
        if not self.memory_guard_checkbox.isChecked():
            return ""
        if self._cache_mode() != CACHE_MODE_KEEP_ALL:
            return ""
        cache_bytes = _pipeline_cache_nbytes(self.pipeline)
        if cache_bytes <= 0:
            return ""
        free_bytes, total_bytes = _system_memory_bytes()
        limit_percent = int(self.memory_limit_spin.value())
        reason = self._memory_guard_reason(
            cache_bytes,
            free_bytes,
            total_bytes,
            limit_percent,
        )
        if not reason:
            return ""

        with QSignalBlocker(self.cache_mode_combo):
            self.cache_mode_combo.setCurrentText(CACHE_MODE_SMART)
        self._clear_optional_memory_caches()
        self._apply_cache_retention()
        self._refresh_dynamic_output_ports()
        self._update_thumbnails()
        self._refresh_inspection_layer_if_active()
        self._refresh_pinned_layer_if_active()
        self._update_metadata_panel()
        self._update_histogram()
        self._sync_execution_ui()
        message = (
            "Memory guard switched cache mode to Smart interactive cache. "
            f"{reason} Mark critical intermediates with Keep output cached if "
            "they should survive pruning."
        )
        if not self._memory_guard_dialog_shown:
            self._memory_guard_dialog_shown = True
            QMessageBox.warning(self, "VIPP memory guard", message)
        return message

    def _memory_guard_reason(
        self,
        cache_bytes: int,
        free_bytes: int | None,
        total_bytes: int | None,
        limit_percent: int,
    ) -> str:
        if free_bytes is not None and free_bytes >= 0:
            reclaimable_bytes = cache_bytes + free_bytes
            limit_bytes = int(reclaimable_bytes * (limit_percent / 100.0))
            if cache_bytes > limit_bytes:
                return (
                    f"The VIPP cache reached {_format_byte_count(cache_bytes)}, "
                    f"above the configured {limit_percent}% reclaimable-memory "
                    f"limit ({_format_byte_count(limit_bytes)} of "
                    f"{_format_byte_count(reclaimable_bytes)} free-or-cached RAM)."
                )
            if (
                free_bytes < MEMORY_GUARD_MIN_FREE_BYTES
                and cache_bytes > MEMORY_GUARD_MIN_FREE_BYTES
            ):
                return (
                    f"System free RAM is down to {_format_byte_count(free_bytes)}, "
                    f"below VIPP's {_format_byte_count(MEMORY_GUARD_MIN_FREE_BYTES)} "
                    "safety reserve."
                )
        if free_bytes is None and total_bytes is not None and total_bytes > 0:
            limit_bytes = int(total_bytes * (limit_percent / 100.0))
            if cache_bytes > limit_bytes:
                return (
                    f"The VIPP cache reached {_format_byte_count(cache_bytes)}, "
                    f"above the fallback {limit_percent}% total-RAM limit "
                    f"({_format_byte_count(limit_bytes)})."
                )
        return ""

    def _clear_optional_memory_caches(self) -> None:
        self._sample_payload_cache = None
        self._source_inspection_cache.clear()

    def _refresh_cache_status(self) -> None:
        cache_bytes = _pipeline_cache_nbytes(self.pipeline)
        free_bytes, total_bytes = _system_memory_bytes()
        cache_text = _format_byte_count(cache_bytes)
        mode_label = CACHE_MODE_STATUS_LABELS.get(
            self._cache_mode(),
            self._cache_mode(),
        )
        if free_bytes is None:
            memory_text = "RAM n/a"
        elif total_bytes:
            memory_text = (
                f"RAM free {_format_byte_count(free_bytes)} / "
                f"{_format_byte_count(total_bytes)}"
            )
        else:
            memory_text = f"RAM free {_format_byte_count(free_bytes)}"
        text = f"Cache {cache_text} ({mode_label}) | {memory_text}"
        guard_state = "on" if self.memory_guard_checkbox.isChecked() else "off"
        self.cache_status_label.setText(text)
        self.cache_status_label.setToolTip(
            f"{text}\nMode: {self._cache_mode()}"
            f"\nMemory guard: {guard_state}"
            f"\nCache limit: {int(self.memory_limit_spin.value())}% of "
            "free RAM + VIPP cache"
        )

    def _dirty_nodes_affect_node(
        self,
        dirty_node_ids: set[str],
        node_id: str | None,
    ) -> bool:
        if not node_id:
            return True
        return node_id in self.pipeline.descendants_inclusive(dirty_node_ids)

    def _hide_input_layer_for_inspection(self, layer) -> None:
        if layer is None or self._is_vipp_generated_layer(layer):
            return
        key = id(layer)
        if key not in self._hidden_input_layer_states:
            self._hidden_input_layer_states[key] = (
                layer,
                bool(getattr(layer, "visible", True)),
            )
        try:
            layer.visible = False
        except Exception:
            pass

    def _restore_hidden_input_layers(self, except_layer=None) -> None:
        keep_key = id(except_layer) if except_layer is not None else None
        for key, (layer, visible) in list(self._hidden_input_layer_states.items()):
            if key == keep_key:
                continue
            if self._layer_is_present(layer):
                try:
                    layer.visible = visible
                except Exception:
                    pass
            self._hidden_input_layer_states.pop(key, None)

    def _layer_is_present(self, layer) -> bool:
        try:
            return any(candidate is layer for candidate in self.viewer.layers)
        except Exception:
            return False

    def _on_viewer_layers_changed(self, _event=None) -> None:
        changed_node_ids = self._autobind_default_image_sources()
        self._refresh_image_source_controls()
        if changed_node_ids and self._mark_pipeline_branches_dirty(changed_node_ids):
            self.run_pipeline()

    def _autobind_default_image_sources(self) -> set[str]:
        default_layer_name = self._default_input_layer_name()
        if not default_layer_name:
            return set()
        changed: set[str] = set()
        for node_id, node in self.pipeline.nodes.items():
            if node.operation_id != "input":
                continue
            mode = str(node.params.get("source_mode", "napari layer"))
            layer_name = str(node.params.get("layer_name", "")).strip()
            if mode == "napari layer" and not layer_name:
                node.params["layer_name"] = default_layer_name
                changed.add(node_id)
        return changed

    def _refresh_image_source_controls(self) -> None:
        node = self.pipeline.nodes.get(self._selected_node_id)
        control = self._parameter_widgets.get("image_source")
        if node is None or node.operation_id != "input":
            return
        if isinstance(control, ImageSourceControl):
            self._refresh_image_source_options()

    def _default_input_layer_name(self) -> str:
        preferred = self._preferred_input_layer_name()
        if preferred:
            return preferred
        names = self._available_layer_names()
        return names[0] if names else ""

    def _preferred_input_layer_name(self) -> str | None:
        fallback: tuple[int, str] | None = None
        for layer in self.viewer.layers:
            if self._is_vipp_generated_layer(layer) or not hasattr(layer, "data"):
                continue
            metadata = getattr(layer, "metadata", None)
            if not isinstance(metadata, dict):
                continue
            if not metadata.get("napari_vipp_sample"):
                continue
            score = self._sample_layer_score(layer, metadata)
            if metadata.get("napari_vipp_preferred_input"):
                score += 1000
            if fallback is None or score > fallback[0]:
                fallback = (score, str(getattr(layer, "name", "")))
        return fallback[1] if fallback is not None else None

    def _sample_layer_score(self, layer, metadata: dict) -> int:
        axis_order = str(metadata.get("vipp_axis_order", "")).upper()
        score = len(axis_order) * 10
        if "T" in axis_order:
            score += 100
        if "C" in axis_order:
            score += 50
        try:
            score += int(np.asarray(layer.data).ndim)
        except Exception:
            pass
        return score

    def _available_layer_names(self) -> list[str]:
        names: list[str] = []
        for layer in self.viewer.layers:
            if self._is_vipp_generated_layer(layer):
                continue
            if hasattr(layer, "data"):
                names.append(str(getattr(layer, "name", "")))
        return names

    def _sample_names(self) -> list[str]:
        return list(self._sample_payloads().keys())

    def _sample_payloads(self) -> dict[str, SourcePayload]:
        if self._sample_payload_cache is None:
            payloads: dict[str, SourcePayload] = {}
            for data, metadata, _layer_type in make_sample_data():
                name = str(metadata.get("name", "VIPP sample"))
                payloads[name] = SourcePayload(
                    data,
                    metadata.get("metadata", {}),
                    name,
                )
            self._sample_payload_cache = payloads
        return self._sample_payload_cache

    def _uncached_async_file_source_specs(self) -> tuple[SourceFileLoadSpec, ...]:
        specs: list[SourceFileLoadSpec] = []
        seen: set[tuple[object, ...]] = set()
        for node_id, node in self.pipeline.nodes.items():
            if node.operation_id != "input":
                continue
            if not self._file_source_should_load_async(node):
                continue
            key = self._file_source_cache_key(node)
            if key is None or key in self._file_source_payload_cache or key in seen:
                continue
            seen.add(key)
            specs.append(
                SourceFileLoadSpec(
                    node_id=node_id,
                    path=str(
                        Path(str(node.params.get("file_path", "")).strip())
                        .expanduser()
                    ),
                    series_index=int(node.params.get("series_index", 0) or 0),
                    cache_key=key,
                )
            )
        return tuple(specs)

    def _file_source_should_load_async(self, node) -> bool:
        if str(node.params.get("source_mode", "")) != "file path":
            return False
        path_text = str(node.params.get("file_path", "")).strip()
        if not path_text:
            return False
        source_path = Path(path_text).expanduser()
        suffix = source_path.suffix.lower()
        if suffix in MICROSCOPE_SUFFIXES:
            return True
        try:
            return source_path.stat().st_size >= ASYNC_SOURCE_FILE_BYTES
        except OSError:
            return False

    def _file_source_cache_key(self, node) -> tuple[object, ...] | None:
        if str(node.params.get("source_mode", "")) != "file path":
            return None
        path_text = str(node.params.get("file_path", "")).strip()
        if not path_text:
            return None
        source_path = Path(path_text).expanduser()
        try:
            stat = source_path.stat()
        except OSError:
            return None
        try:
            identity = str(source_path.resolve())
        except OSError:
            identity = str(source_path)
        return (
            identity,
            int(node.params.get("series_index", 0) or 0),
            int(stat.st_mtime_ns),
            int(stat.st_size),
        )

    def _cached_file_source_payload(self, node) -> SourcePayload | None:
        key = self._file_source_cache_key(node)
        if key is None:
            return None
        payload = self._file_source_payload_cache.get(key)
        if payload is None:
            return None
        return self._viewer_aligned_source_payload(payload)

    def _prune_file_source_payload_cache(self) -> None:
        active_keys = {
            key
            for node in self.pipeline.nodes.values()
            if node.operation_id == "input"
            for key in (self._file_source_cache_key(node),)
            if key is not None
        }
        if not active_keys:
            self._file_source_payload_cache.clear()
            return
        self._file_source_payload_cache = {
            key: payload
            for key, payload in self._file_source_payload_cache.items()
            if key in active_keys
        }

    def _source_payloads_for_pipeline(
        self,
    ) -> tuple[dict[str, SourcePayload], list[object]]:
        payloads: dict[str, SourcePayload] = {}
        layers: list[object] = []
        for node_id, node in self.pipeline.nodes.items():
            if node.operation_id != "input":
                continue
            payload, layer = self._resolve_source_payload(node)
            if payload is not None:
                payloads[node_id] = payload
            if layer is not None:
                layers.append(layer)
        return payloads, layers

    def _resolve_source_payload(
        self,
        node,
    ) -> tuple[SourcePayload | None, object | None]:
        mode = str(node.params.get("source_mode", "napari layer"))
        if mode == "file path":
            path = str(node.params.get("file_path", "")).strip()
            if not path:
                representative = self._interactive_collection_source_paths.get(
                    node.id
                )
                if representative is None:
                    return SourcePayload(None, {}, ""), None
                path = str(representative)
            cached = self._cached_file_source_payload(node)
            if cached is not None:
                return cached, None
            if self._file_source_should_load_async(node):
                return None, None
            dataset = read_image(
                path,
                series_index=int(node.params.get("series_index", 0)),
            )
            return (
                SourcePayload(
                    dataset.data,
                    {"vipp_source_path": str(Path(path).expanduser())},
                    dataset.selected_series.name,
                    self._viewer_aligned_state(dataset.image_state),
                ),
                None,
            )
        if mode == "sample":
            sample_name = str(node.params.get("sample_name", "")).strip()
            payloads = self._sample_payloads()
            if not sample_name and payloads:
                sample_name = next(iter(payloads))
            payload = payloads.get(sample_name)
            if payload is None:
                return None, None
            return self._viewer_aligned_source_payload(payload), None

        layer_name = str(node.params.get("layer_name", "")).strip()
        if not layer_name:
            layer_name = self._default_input_layer_name()
        layer = self._layer_by_name(layer_name) if layer_name else None
        if layer is None:
            return None, None
        data = layer.data
        metadata = getattr(layer, "metadata", None)
        name = getattr(layer, "name", "")
        return (
            SourcePayload(
                data,
                metadata,
                name,
                self._viewer_aligned_image_state(data, metadata, str(name)),
            ),
            layer,
        )

    @contextmanager
    def _preserve_interactive_collection_workflow_params(self):
        """Keep representative preview calculations out of workflow params."""
        if not self._interactive_collection_source_paths:
            yield
            return
        original_params = {
            node_id: dict(node.params)
            for node_id, node in self.pipeline.nodes.items()
        }
        try:
            yield
        finally:
            for node_id, params in original_params.items():
                node = self.pipeline.nodes.get(node_id)
                if node is not None:
                    node.params = params

    def _viewer_aligned_source_payload(
        self,
        payload: SourcePayload,
    ) -> SourcePayload:
        state = payload.image_state or image_state_from_array(
            payload.data,
            layer_metadata=payload.metadata,
            source_name=payload.name,
        )
        return SourcePayload(
            payload.data,
            payload.metadata,
            payload.name,
            self._viewer_aligned_state(state),
        )

    def _viewer_aligned_image_state(
        self,
        data,
        metadata,
        name: str,
    ):
        state = image_state_from_array(
            data,
            layer_metadata=metadata,
            source_name=name,
            defer_statistics=_should_auto_background_data(data),
        )
        return self._viewer_aligned_state(state)

    def _viewer_aligned_state(self, state):
        if state is None:
            return None
        current_step = self._current_step()
        if current_step is None:
            return state
        viewer_ndim = len(tuple(current_step))
        offset = max(viewer_ndim - len(state.axes), 0)
        if offset == 0:
            return state
        axes = tuple(
            replace(axis, source_axis=index + offset)
            for index, axis in enumerate(state.axes)
        )
        return replace(state, axes=axes)

    def _inspect_source_file(self, path: str) -> SourceInspection | None:
        source_path = Path(path).expanduser()
        if not source_path.exists():
            return None
        cache_key = str(source_path.resolve())
        modified = source_path.stat().st_mtime_ns
        cached = self._source_inspection_cache.get(cache_key)
        if cached is not None and cached[0] == modified:
            return cached[1]
        inspection = inspect_image_source(source_path)
        self._source_inspection_cache[cache_key] = (modified, inspection)
        return inspection

    def _graph_note_documents(self, *, use_view_positions: bool = True) -> list[dict]:
        positions = self.graph_view.note_positions() if use_view_positions else {}
        documents: list[dict] = []
        for note in self._graph_notes.values():
            position = positions.get(note.id, note.position)
            documents.append(replace(note, position=position).to_workflow_dict())
        return documents

    def _sync_graph_note_positions_from_view(self) -> None:
        positions = self.graph_view.note_positions()
        for note_id, position in positions.items():
            note = self._graph_notes.get(note_id)
            if note is None:
                continue
            self._graph_notes[note_id] = replace(note, position=position)

    def _restore_graph_notes(self, notes) -> None:
        restored: dict[str, GraphNoteState] = {}
        for raw in notes or ():
            note_id = str(raw.get("id", "")).strip()
            if not note_id:
                continue
            position = tuple(raw.get("position", (0.0, 0.0)))
            restored[note_id] = GraphNoteState(
                note_id,
                str(raw.get("text", "")),
                (float(position[0]), float(position[1])),
                float(raw.get("width", 240.0)),
                str(raw.get("attached_node", "") or ""),
            )
        self._graph_notes = restored

    def _next_graph_note_id(self) -> str:
        index = 1
        while f"note_{index}" in self._graph_notes:
            index += 1
        return f"note_{index}"

    def _add_graph_note(
        self,
        text: str = "Note",
        position: QPointF | None = None,
        *,
        width: float = 240.0,
        attached_node: str = "",
    ) -> str:
        self._finish_parameter_history_group()
        before = self._current_history_snapshot()
        note_id = self._next_graph_note_id()
        attached_node = str(attached_node or "")
        if position is not None:
            point = position
        elif attached_node:
            point = self.graph_view.suggest_note_position_for_node(attached_node)
        else:
            point = self.graph_view.suggest_note_position()
        note = GraphNoteState(
            note_id,
            str(text),
            (float(point.x()), float(point.y())),
            float(width),
            attached_node,
        )
        self._graph_notes[note_id] = note
        self.graph_view.add_note(
            note.id,
            note.text,
            point,
            width=note.width,
            attached_node=note.attached_node,
        )
        self.graph_view.select_note(note_id)
        self._push_undo_if_changed(before)
        if attached_node:
            self.status_label.setText(
                f"Added note to '{self._node_title(attached_node)}'."
            )
        else:
            self.status_label.setText("Added graph note.")
        return note_id

    def _add_graph_note_for_node(self, node_id: str) -> None:
        if node_id not in self.pipeline.nodes:
            return
        self._add_graph_note(attached_node=node_id)

    def _edit_graph_note(self, note_id: str) -> None:
        note = self._graph_notes.get(note_id)
        if note is None:
            return
        text, ok = QInputDialog.getMultiLineText(
            self,
            "Edit Graph Note",
            "Note:",
            note.text,
        )
        if ok:
            self._set_graph_note_text(note_id, str(text))

    def _set_graph_note_text(self, note_id: str, text: str) -> None:
        note = self._graph_notes.get(note_id)
        if note is None or note.text == text:
            return
        self._finish_parameter_history_group()
        before = self._current_history_snapshot()
        self._graph_notes[note_id] = replace(note, text=str(text))
        self.graph_view.set_note_text(note_id, text)
        self._push_undo_if_changed(before)
        self.status_label.setText("Updated graph note.")

    def _delete_graph_note(self, note_id: str) -> None:
        if note_id not in self._graph_notes:
            return
        self._finish_parameter_history_group()
        before = self._current_history_snapshot()
        self._graph_notes.pop(note_id, None)
        self.graph_view.remove_note(note_id)
        self._push_undo_if_changed(before)
        self.status_label.setText("Deleted graph note.")

    def _on_graph_note_moved(
        self,
        note_id: str,
        old_pos: QPointF,
        new_pos: QPointF,
    ) -> None:
        note = self._graph_notes.get(note_id)
        if note is None:
            return
        new_position = (float(new_pos.x()), float(new_pos.y()))
        if (
            abs(note.position[0] - new_position[0])
            + abs(note.position[1] - new_position[1])
            < 0.001
        ):
            return
        self._finish_parameter_history_group()
        before = self._current_history_snapshot(
            notes_override=[
                item.to_workflow_dict()
                for item in self._graph_notes.values()
                if item.id != note_id
            ]
            + [
                replace(
                    note,
                    position=(float(old_pos.x()), float(old_pos.y())),
                ).to_workflow_dict()
            ],
        )
        self._graph_notes[note_id] = replace(
            note,
            position=new_position,
        )
        self._push_undo_if_changed(before)
        self.status_label.setText("Moved graph note.")

    def _show_tunnel_manager(self) -> None:
        dialog = self._tunnel_manager_dialog
        if dialog is None:
            dialog = TunnelManagerDialog(self)
            dialog.setAttribute(Qt.WA_DeleteOnClose, True)
            dialog.destroyed.connect(
                lambda *_args: setattr(self, "_tunnel_manager_dialog", None)
            )
            dialog.tunnelSelected.connect(self._highlight_output_tunnel)
            dialog.focusSourceRequested.connect(self._focus_tunnel_source)
            dialog.revealRequested.connect(self._reveal_output_tunnel)
            dialog.renameRequested.connect(self._rename_output_tunnel)
            dialog.deleteRequested.connect(self._remove_output_tunnel)
            self._tunnel_manager_dialog = dialog
        self._refresh_tunnel_manager()
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _refresh_tunnel_manager(self) -> None:
        dialog = self._tunnel_manager_dialog
        if dialog is not None:
            dialog.set_tunnels(self._tunnel_summaries())

    def _tunnel_summaries(self) -> tuple[TunnelSummary, ...]:
        rows: list[TunnelSummary] = []
        for tunnel in self.pipeline.output_tunnel_list():
            source_node = self.pipeline.nodes.get(tunnel.source_id)
            if source_node is None:
                continue
            ports = self.pipeline.output_ports(tunnel.source_id)
            output_type = "unknown"
            if 0 <= tunnel.source_port < len(ports):
                output_type = ports[tunnel.source_port].output_type
            subscribers = tuple(
                (
                    connection.target_id,
                    self._node_title(connection.target_id),
                    int(connection.target_port),
                )
                for connection in self.pipeline.connections
                if connection.tunnel_name == tunnel.name
                and connection.target_id in self.pipeline.nodes
            )
            rows.append(
                TunnelSummary(
                    tunnel.name,
                    tunnel.source_id,
                    source_node.title,
                    int(tunnel.source_port),
                    output_type,
                    subscribers,
                )
            )
        return tuple(rows)

    def _on_graph_tunnel_selected(self, name: str) -> None:
        tunnel_name = str(name or "").strip()
        if not tunnel_name:
            return
        self.status_label.setText(self._tunnel_status_text(tunnel_name))
        if self._tunnel_manager_dialog is not None:
            self._tunnel_manager_dialog.select_tunnel(tunnel_name)

    def _highlight_output_tunnel(self, name: str) -> None:
        tunnel_name = str(name or "").strip()
        if not tunnel_name:
            return
        self.graph_view.highlight_tunnel(tunnel_name, sticky=True)
        self.status_label.setText(self._tunnel_status_text(tunnel_name))

    def _reveal_output_tunnel(self, name: str) -> None:
        tunnel_name = str(name or "").strip()
        if not tunnel_name:
            return
        self.graph_view.reveal_tunnel(tunnel_name)
        self.status_label.setText(self._tunnel_status_text(tunnel_name))

    def _focus_tunnel_source(self, name: str) -> None:
        tunnel = self.pipeline.output_tunnel(name)
        if tunnel is None:
            return
        self.graph_view.focus_node(tunnel.source_id)
        self.status_label.setText(
            f"Focused tunnel '{tunnel.name}' source "
            f"'{self._node_title(tunnel.source_id)}'."
        )

    def _tunnel_status_text(self, name: str) -> str:
        for summary in self._tunnel_summaries():
            if summary.name == name:
                plural = "input" if summary.subscriber_count == 1 else "inputs"
                return (
                    f"Tunnel '{summary.name}': {summary.source_title} output "
                    f"{summary.source_port + 1} -> "
                    f"{summary.subscriber_count} {plural}."
                )
        return f"Tunnel '{name}'."

    def _show_port_context_menu(
        self,
        kind: str,
        node_id: str,
        port_index: int,
        global_pos,
    ) -> None:
        if node_id not in self.pipeline.nodes:
            return
        if kind == "output":
            self._show_output_tunnel_menu(node_id, port_index, global_pos)
        elif kind == "input":
            self._show_input_tunnel_menu(node_id, port_index, global_pos)

    def _show_output_tunnel_menu(
        self,
        node_id: str,
        port_index: int,
        global_pos,
    ) -> None:
        tunnel = self.pipeline.output_tunnel_for_port(node_id, port_index)
        menu = QMenu(self)
        if tunnel is None:
            create_action = menu.addAction("Create output tunnel...")
            manage_action = menu.addAction("Manage tunnels...")
            action = menu.exec(global_pos)
            if action == create_action:
                self._create_output_tunnel(node_id, port_index)
            elif action == manage_action:
                self._show_tunnel_manager()
            return

        reveal_action = menu.addAction(f"Reveal tunnel '{tunnel.name}'")
        focus_action = menu.addAction("Focus source")
        manage_action = menu.addAction("Manage tunnels...")
        menu.addSeparator()
        rename_action = menu.addAction(f"Rename tunnel '{tunnel.name}'...")
        remove_action = menu.addAction(f"Remove tunnel '{tunnel.name}'")
        action = menu.exec(global_pos)
        if action == reveal_action:
            self._reveal_output_tunnel(tunnel.name)
        elif action == focus_action:
            self._focus_tunnel_source(tunnel.name)
        elif action == manage_action:
            self._show_tunnel_manager()
        elif action == rename_action:
            self._rename_output_tunnel(tunnel.name)
        elif action == remove_action:
            self._remove_output_tunnel(tunnel.name)

    def _show_input_tunnel_menu(
        self,
        node_id: str,
        port_index: int,
        global_pos,
    ) -> None:
        current = self.pipeline.tunnel_connection_for_input(node_id, port_index)
        compatible = self.pipeline.compatible_output_tunnels(node_id, port_index)
        menu = QMenu(self)
        tunnel_actions = {}
        reveal_action = None
        rename_action = None
        manage_action = None
        if current is not None:
            reveal_action = menu.addAction(f"Reveal tunnel '{current.tunnel_name}'")
            rename_action = menu.addAction(f"Rename tunnel '{current.tunnel_name}'...")
            manage_action = menu.addAction("Manage tunnels...")
            menu.addSeparator()
        if compatible:
            use_menu = menu.addMenu("Use tunnel")
            for tunnel in compatible:
                action = use_menu.addAction(tunnel.name)
                action.setCheckable(True)
                action.setChecked(
                    current is not None and current.tunnel_name == tunnel.name
                )
                tunnel_actions[action] = tunnel.name
        else:
            action = menu.addAction("No compatible output tunnels")
            action.setEnabled(False)

        clear_action = None
        if current is not None:
            menu.addSeparator()
            clear_action = menu.addAction(f"Clear tunnel '{current.tunnel_name}'")

        action = menu.exec(global_pos)
        if reveal_action is not None and action == reveal_action:
            self._reveal_output_tunnel(current.tunnel_name)
        elif rename_action is not None and action == rename_action:
            self._rename_output_tunnel(current.tunnel_name)
        elif manage_action is not None and action == manage_action:
            self._show_tunnel_manager()
        elif action in tunnel_actions:
            self._connect_input_to_tunnel(
                tunnel_actions[action],
                node_id,
                port_index,
            )
        elif clear_action is not None and action == clear_action:
            self._clear_input_tunnel(node_id, port_index)

    def _create_output_tunnel(self, node_id: str, port_index: int) -> None:
        default = self._suggest_tunnel_name(node_id, port_index)
        name = self._prompt_tunnel_name("Create Output Tunnel", default)
        if not name:
            return
        self._finish_parameter_history_group()
        before = self._current_history_snapshot()
        try:
            tunnel = self.pipeline.add_output_tunnel(name, node_id, port_index)
        except ValueError as exc:
            self.status_label.setText(str(exc))
            return
        self._sync_port_tunnels()
        self._push_undo_if_changed(before)
        self.status_label.setText(
            f"Created tunnel '{tunnel.name}' from "
            f"'{self._node_title(node_id)}' output {port_index + 1}."
        )

    def _rename_output_tunnel(self, old_name: str) -> None:
        name = self._prompt_tunnel_name("Rename Output Tunnel", old_name)
        if not name or name == old_name:
            return
        self._rename_output_tunnel_to(old_name, name)

    def _rename_output_tunnel_to(self, old_name: str, name: str) -> bool:
        self._finish_parameter_history_group()
        before = self._current_history_snapshot()
        try:
            tunnel = self.pipeline.rename_output_tunnel(old_name, name)
        except ValueError as exc:
            self.status_label.setText(str(exc))
            return False
        self._sync_port_tunnels()
        self._push_undo_if_changed(before)
        self._highlight_output_tunnel(tunnel.name)
        self.status_label.setText(f"Renamed tunnel '{old_name}' to '{tunnel.name}'.")
        return True

    def _remove_output_tunnel(self, name: str) -> None:
        self._finish_parameter_history_group()
        before = self._current_history_snapshot()
        removed = self.pipeline.remove_output_tunnel(name)
        self._sync_port_tunnels()
        self._refresh_split_channel_display_surfaces(
            {connection.source_id for connection in removed}
        )
        self.graph_view.clear_tunnel_highlight(sticky=True)
        if removed:
            self._mark_pipeline_branches_dirty(
                {connection.target_id for connection in removed}
            )
            self.run_pipeline()
        self._push_undo_if_changed(before)
        self.status_label.setText(f"Removed tunnel '{name}'.")

    def _connect_input_to_tunnel(
        self,
        name: str,
        node_id: str,
        port_index: int,
    ) -> None:
        self._finish_parameter_history_group()
        before = self._current_history_snapshot()
        result = self.pipeline.connect_to_tunnel(name, node_id, port_index)
        if not result.success:
            self.status_label.setText(result.message)
            return
        self._apply_connection_result_to_graph(result)
        if self._mark_pipeline_dirty(node_id):
            self.run_pipeline()
        self._push_undo_if_changed(before)
        self.status_label.setText(result.message)

    def _clear_input_tunnel(self, node_id: str, port_index: int) -> None:
        connection = self.pipeline.tunnel_connection_for_input(node_id, port_index)
        if connection is None:
            return
        self._finish_parameter_history_group()
        before = self._current_history_snapshot()
        if self.pipeline.disconnect(
            connection.source_id,
            connection.target_id,
            connection.target_port,
        ):
            self._sync_port_tunnels()
            self._refresh_split_channel_display_surfaces({connection.source_id})
            if self._mark_pipeline_dirty(node_id):
                self.run_pipeline()
            self._push_undo_if_changed(before)
            self.status_label.setText(
                f"Cleared tunnel '{connection.tunnel_name}' from "
                f"'{self._node_title(node_id)}' input {port_index + 1}."
            )

    def _prompt_tunnel_name(self, title: str, default: str) -> str:
        text, ok = QInputDialog.getText(
            self,
            title,
            "Tunnel name:",
            QLineEdit.Normal,
            default,
        )
        return str(text or "").strip() if ok else ""

    def _suggest_tunnel_name(self, node_id: str, port_index: int) -> str:
        node = self.pipeline.nodes[node_id]
        ports = self.pipeline.output_ports(node_id)
        label = ""
        if 0 <= port_index < len(ports):
            label = str(ports[port_index].label or "").strip()
        base = label if label and label.lower() != "out" else node.title
        existing = {
            tunnel.name.casefold() for tunnel in self.pipeline.output_tunnel_list()
        }
        candidate = base
        suffix = 2
        while candidate.casefold() in existing:
            candidate = f"{base} {suffix}"
            suffix += 1
        return candidate

    def _sync_port_tunnels(self) -> None:
        self.graph_view.set_port_tunnels(
            self.pipeline.output_tunnel_list(),
            self.pipeline.connections,
        )
        self._refresh_tunnel_manager()
        self._refresh_graph_search_matches(reset_index=False)

    def _connect_nodes(
        self,
        source_id: str,
        target_id: str,
        target_port: int | None = None,
        source_port: int = 0,
    ) -> None:
        self._finish_parameter_history_group()
        before = self._current_history_snapshot()
        result = self.pipeline.connect(source_id, target_id, target_port, source_port)
        if not result.success:
            self.status_label.setText(result.message)
            return
        self._apply_connection_result_to_graph(result)
        self._sync_node_output_ports(target_id)
        if self._selected_node_id == target_id:
            self._refresh_selected_parameter_controls()
        if self._mark_pipeline_dirty(target_id):
            self.run_pipeline()
        self._push_undo_if_changed(before)
        self.status_label.setText(result.message)

    def _disconnect_nodes(
        self,
        source_id: str,
        target_id: str,
        target_port: int | None = None,
    ) -> None:
        self._finish_parameter_history_group()
        before = self._current_history_snapshot()
        if self.pipeline.disconnect(source_id, target_id, target_port):
            self._sync_port_tunnels()
            self._refresh_split_channel_display_surfaces({source_id})
            if self._mark_pipeline_dirty(target_id):
                self.run_pipeline()
            self._push_undo_if_changed(before)
            port_text = "" if target_port is None else f" input {int(target_port) + 1}"
            self.status_label.setText(
                f"Disconnected {source_id} -> {target_id}{port_text}."
            )

    def _delete_node(self, node_id: str) -> None:
        node = self.pipeline.nodes.get(node_id)
        if node is None:
            return
        self._finish_parameter_history_group()
        before = self._current_history_snapshot()
        title = node.title
        dirty_targets = {
            connection.target_id
            for connection in self.pipeline.connections
            if connection.source_id == node_id
        }
        affected_split_sources = {
            connection.source_id
            for connection in self.pipeline.connections
            if connection.target_id == node_id
        }
        if not self.pipeline.remove_node(node_id):
            return
        self._interactive_collection_source_paths.pop(node_id, None)
        attached_note_ids = [
            note_id
            for note_id, note in self._graph_notes.items()
            if note.attached_node == node_id
        ]
        for note_id in attached_note_ids:
            self._graph_notes.pop(note_id, None)
            self.graph_view.remove_note(note_id)
        self.graph_view.remove_node(node_id)
        self._sync_port_tunnels()
        self._refresh_split_channel_display_surfaces(affected_split_sources)
        self._preview_disabled_node_ids.discard(node_id)
        self._recent_cache_node_ids = [
            recent_id
            for recent_id in self._recent_cache_node_ids
            if recent_id != node_id
        ]
        self._rescale_auto_output_ranges.pop(node_id, None)
        if self._active_pinned_node_id == node_id:
            self._clear_active_pin(status=False)
        if self._selected_node_id == node_id:
            self._select_first_available_node()
        if self._mark_pipeline_branches_dirty(dirty_targets):
            self.run_pipeline()
        self._push_undo_if_changed(before)
        self.status_label.setText(f"Deleted '{title}'.")

    def _on_node_moved(self, node_id: str, old_pos, _new_pos) -> None:
        if node_id not in self.pipeline.nodes:
            return
        self._finish_parameter_history_group()
        positions = self.graph_view.node_positions()
        positions[node_id] = (float(old_pos.x()), float(old_pos.y()))
        self._push_undo_snapshot(
            self._current_history_snapshot(
                positions,
                notes_override=self._graph_note_documents(use_view_positions=False),
            )
        )
        self._sync_graph_note_positions_from_view()
        self.status_label.setText(f"Moved '{self._node_title(node_id)}'.")

    def _select_first_available_node(self) -> None:
        if self.pipeline.nodes:
            node_id = next(iter(self.pipeline.nodes))
            self.graph_view.select_node(node_id)
            return
        self._selected_node_id = ""
        self.selected_title.setText("No node selected")
        self._clear_parameter_form()
        self.parameter_group.setHidden(True)
        self.auto_contrast_group.setHidden(True)
        self.pin_button.setHidden(True)
        with QSignalBlocker(self.thumbnail_checkbox):
            self.thumbnail_checkbox.setChecked(False)
        self._clear_empty_inspector()

    def _clear_empty_inspector(self) -> None:
        self.execution_group.setHidden(True)
        self.metadata_table.setRowCount(0)
        self.table_group.setHidden(True)
        self.table_preview.setRowCount(0)
        self.table_preview.setColumnCount(0)
        self.history_label.setText("No history yet.")
        self.label_volume_group.setHidden(True)
        self.label_volume_plot.set_histogram(None, log_scale=False)
        self.colocalization_scatter_group.setHidden(True)
        self.colocalization_scatter_summary.setText("Connect two channel inputs.")
        self.colocalization_scatter_plot.clear()
        self.rescale_input_histogram_group.setHidden(True)
        self.rescale_input_histogram_scope_row.setHidden(True)
        self.rescale_input_histogram_plot.set_histogram(None, log_scale=False)
        self.histogram_group.setTitle("Output Histogram")
        self.histogram_scope_row.setHidden(True)
        self.histogram_plot.set_histogram(None, log_scale=False)
        self.keep_cached_checkbox.setVisible(False)
        self.keep_cached_checkbox.setEnabled(False)
        with QSignalBlocker(self.keep_cached_checkbox):
            self.keep_cached_checkbox.setChecked(False)

    def _select_node(self, node_id: str) -> None:
        if node_id not in self.pipeline.nodes:
            return
        self._selected_node_id = node_id
        self._remember_cache_node(node_id)
        node = self.pipeline.nodes[node_id]
        self.selected_title.setText(node.title)
        self._sync_preview_ui()
        self._sync_keep_cached_ui()
        self._render_parameters(node_id)
        self._sync_auto_contrast_ui()
        self._sync_pin_ui()
        self._inspect_selected_node()
        self._keep_active_pin_on_top()
        self._sync_view_dims_bar()
        self._update_metadata_panel()
        self._restore_selected_output_for_interactive_cache(node_id)
        self._update_histogram()
        self._sync_execution_ui()

    def _restore_selected_output_for_interactive_cache(self, node_id: str) -> None:
        if self._cache_mode() != CACHE_MODE_SMART:
            return
        if node_id not in self.pipeline.nodes:
            return
        if self.pipeline.outputs.get(node_id) is not None:
            self._apply_cache_retention()
            return
        if not self._node_has_required_inputs_for_restore(node_id):
            self._refresh_cache_status()
            return
        if self._active_pipeline_run_id is not None:
            self._mark_pipeline_dirty(node_id)
            self._pipeline_run_pending = True
            self._set_pipeline_busy(
                True,
                self._active_pipeline_node_id,
                queued=True,
                preserve_progress=True,
            )
            self.status_label.setText(
                f"Queued '{self._node_title(node_id)}' for cache restore."
            )
            self._refresh_cache_status()
            return
        if not self._mark_pipeline_dirty(node_id):
            return
        self.status_label.setText(
            f"Restoring '{self._node_title(node_id)}' for interactive preview..."
        )
        self.run_pipeline()

    def _node_has_required_inputs_for_restore(self, node_id: str) -> bool:
        node = self.pipeline.nodes.get(node_id)
        if node is None:
            return False
        spec = self.pipeline.operation_spec(node.operation_id)
        if not spec.has_input:
            return True
        connections = self.pipeline._input_connections(node_id)
        if not connections:
            return False
        if node.max_inputs is not None and node.max_inputs == 1:
            return True
        required_inputs = self.pipeline._required_inputs_for(node)
        connected_ports = {connection.target_port for connection in connections}
        return all(port in connected_ports for port in range(required_inputs))

    def _calculate_selected_node(self) -> None:
        self._calculate_node(self._selected_node_id)

    def _calculate_node(self, node_id: str) -> None:
        if not self.pipeline.is_manual_node(node_id):
            return
        self.pipeline.node_execution_states[node_id] = EXECUTION_RUNNING
        self.pipeline.node_execution_messages[node_id] = ""
        self._sync_execution_ui()
        self.run_pipeline(manual_node_ids={node_id})

    def _calculate_all_nodes(self) -> None:
        node_ids = self._manual_node_ids_needing_calculation()
        if not node_ids:
            self.status_label.setText("No manual nodes need calculation.")
            return
        for node_id in node_ids:
            self.pipeline.node_execution_states[node_id] = EXECUTION_RUNNING
            self.pipeline.node_execution_messages[node_id] = ""
        self._sync_execution_ui()
        self.status_label.setText(
            f"Calculating {len(node_ids)} manual node"
            f"{'' if len(node_ids) == 1 else 's'}..."
        )
        self.run_pipeline(manual_node_ids=node_ids)

    def _manual_node_ids_needing_calculation(self) -> set[str]:
        return {
            node_id
            for node_id in self.pipeline.manual_node_ids()
            if self.pipeline.node_execution_states.get(
                node_id,
                EXECUTION_NOT_CALCULATED,
            )
            not in {EXECUTION_READY, EXECUTION_RUNNING}
        }

    def _on_auto_recalculate_toggled(self, checked: bool) -> None:
        node_id = self._selected_node_id
        if not self.pipeline.is_manual_node(node_id):
            return
        if self.pipeline.node_auto_recalculate(node_id) == bool(checked):
            self._sync_execution_ui()
            return
        self._finish_parameter_history_group()
        before = self._current_history_snapshot()
        self.pipeline.set_node_auto_recalculate(node_id, checked)
        self._push_undo_if_changed(before)
        self._sync_execution_ui()
        if checked:
            state = self.pipeline.node_execution_states.get(
                node_id,
                EXECUTION_NOT_CALCULATED,
            )
            if state != EXECUTION_READY:
                self.run_pipeline(manual_node_ids={node_id})
            self.status_label.setText(
                f"Auto Recalculate enabled for '{self._node_title(node_id)}'."
            )
        else:
            self.status_label.setText(
                f"Auto Recalculate disabled for '{self._node_title(node_id)}'."
            )

    def _sync_execution_ui(self) -> None:
        for node_id in self.pipeline.nodes:
            manual = self.pipeline.is_manual_node(node_id)
            state = self.pipeline.node_execution_states.get(
                node_id,
                EXECUTION_NOT_CALCULATED,
            )
            message = self.pipeline.node_execution_messages.get(node_id, "")
            auto_recalculate = self.pipeline.node_auto_recalculate(node_id)
            self.graph_view.set_node_execution_state(
                node_id,
                state,
                manual=manual,
                message=message,
                auto_recalculate=auto_recalculate,
            )

        node_id = self._selected_node_id
        if node_id not in self.pipeline.nodes or not self.pipeline.is_manual_node(
            node_id
        ):
            self.execution_group.setHidden(True)
            return
        state = self.pipeline.node_execution_states.get(
            node_id,
            EXECUTION_NOT_CALCULATED,
        )
        message = self.pipeline.node_execution_messages.get(node_id, "")
        auto_recalculate = self.pipeline.node_auto_recalculate(node_id)
        self.execution_group.setHidden(False)
        self.execution_status_label.setText(
            self._execution_status_text(state, message),
        )
        with QSignalBlocker(self.auto_recalculate_checkbox):
            self.auto_recalculate_checkbox.setChecked(auto_recalculate)
        self.calculate_button.setEnabled(state != EXECUTION_RUNNING)
        self.calculate_button.setHidden(auto_recalculate)
        self.calculate_button.setText(
            "Calculate" if state == EXECUTION_NOT_CALCULATED else "Recalculate",
        )

    def _show_optional_reader_error(
        self,
        error: OptionalMicroscopeReaderError,
    ) -> None:
        command = error.install_command
        fallback = error.fallback_install_command
        suffix = error.suffix or "this file"
        lines = [
            f"VIPP can open {suffix} files after the optional reader is installed.",
        ]
        if command:
            lines.extend(("", "Install command:", command))
        if fallback and fallback != command:
            lines.extend(("", "Broader fallback command:", fallback))
        if error.restart_required:
            lines.extend(
                ("", "After installation, restart napari and reopen the file.")
            )

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Optional Image Reader Missing")
        box.setText(f"{error.reader_label} is not installed.")
        box.setInformativeText("\n".join(lines))
        box.setDetailedText(str(error))
        copy_button = None
        if command:
            copy_button = box.addButton(
                "Copy install command",
                QMessageBox.ActionRole,
            )
        box.addButton(QMessageBox.Close)
        box.exec()

        if copy_button is not None and box.clickedButton() == copy_button:
            clipboard = QApplication.clipboard()
            copied = False
            for _attempt in range(3):
                clipboard.setText(command)
                if clipboard.text() == command:
                    copied = True
                    break
                QApplication.processEvents()
            if copied:
                self.status_label.setText(
                    f"Copied reader install command: {command}"
                )
            else:
                self.status_label.setText(
                    "Could not access the system clipboard. Install with: "
                    f"{command}"
                )
            return
        if command:
            self.status_label.setText(
                f"Optional image reader missing. Install with: {command}"
            )
        else:
            self.status_label.setText(f"Image source error: {error}")

    @staticmethod
    def _execution_status_text(state: str, message: str = "") -> str:
        if state == EXECUTION_READY:
            return "Cached result ready."
        if state == EXECUTION_RUNNING:
            return "Calculating..."
        if state == EXECUTION_STALE:
            return message or "Cached result is stale. Recalculate to refresh it."
        if state == EXECUTION_ERROR:
            return message or "Calculation failed."
        return message or "This node calculates only when requested."

    def _render_parameters(self, node_id: str) -> None:
        self._clear_parameter_form()
        node = self.pipeline.nodes[node_id]
        if node.operation_id == "input":
            self.parameter_group.setHidden(False)
            self._render_image_source_parameters(node_id)
            return
        specs = self.pipeline.node_parameter_specs(node_id)
        stack_note = self._stack_processing_note(node_id)
        help_note = self._operation_help_note(node_id)
        self.parameter_group.setHidden(not specs and not stack_note and not help_note)
        if not specs:
            if stack_note:
                self._add_operation_note(stack_note)
                self.parameter_group.setHidden(False)
            if help_note:
                self._add_operation_note(help_note)
                self.parameter_group.setHidden(False)
            return
        if node.operation_id == "select_axis_slice":
            self._render_select_axis_slice_parameters(node_id)
            return
        if node.operation_id == "reorder_axes":
            self._render_reorder_axes_parameters(node_id)
            return
        if node.operation_id == "select_table_columns":
            self._render_select_table_columns_parameters(node_id)
            return
        if node.operation_id == "rescale_axes":
            self._render_rescale_axes_parameters(node_id)
            return
        if node.operation_id == "combine_channels":
            self._render_combine_channels_parameters(node_id)
            return
        if node.operation_id == "assign_channel_colors":
            self._render_assign_channel_colors_parameters(node_id)
            return
        if node.operation_id == "born_wolf_psf":
            self._render_born_wolf_psf_parameters(node_id)
            return

        self._sync_rescale_output_range_defaults(node_id)
        rendered = False
        for spec in specs:
            if self._parameter_spec_hidden(node_id, spec):
                continue
            spec = self._effective_parameter_spec(node_id, spec)
            bounds = self._parameter_bounds_for(node_id, spec)
            locked_split_channel = (
                node.operation_id == "split_channels"
                and spec.name == "preview_channel"
                and self._single_used_split_channel_port(node.id) is not None
            )
            presented_value = (
                self._single_used_split_channel_port(node.id)
                if locked_split_channel
                else node.params.get(spec.name)
            )
            if self._parameter_uses_numeric_entry_only(node_id, spec):
                control_class = NumericEntryControl
            elif spec.kind == "choice":
                control_class = ChoiceControl
            elif spec.kind == "text":
                control_class = TextControl
            elif spec.kind == "bool":
                control_class = BoolControl
            else:
                control_class = ParameterControl
            widget = control_class(spec, presented_value, bounds)
            if not locked_split_channel:
                node.params[spec.name] = widget.value()
            widget.valueChanged.connect(
                lambda value, name=spec.name: self._on_param_changed(name, value)
            )
            self.parameter_form.addRow(spec.label, widget)
            self._parameter_widgets[spec.name] = widget
            rendered = True
        if node.operation_id == "fill_holes":
            note = QLabel()
            note.setWordWrap(True)
            self.parameter_form.addRow(note)
            self._parameter_widgets["fill_holes_scope_note"] = note
            self._update_fill_holes_scope_note()
            rendered = True
        if stack_note:
            self._add_operation_note(stack_note)
            rendered = True
        if help_note:
            self._add_operation_note(help_note)
            rendered = True
        self.parameter_group.setHidden(not rendered)

    def _render_rescale_axes_parameters(self, node_id: str) -> None:
        self._sync_rescale_axes_representations(node_id)
        node = self.pipeline.nodes[node_id]
        for spec in self._rescale_axes_visible_specs(node_id):
            bounds = self._parameter_bounds_for(node_id, spec)
            if spec.kind == "choice":
                control_class = ChoiceControl
            elif spec.kind == "bool":
                control_class = BoolControl
            else:
                control_class = NumericEntryControl
            widget = control_class(spec, node.params.get(spec.name), bounds)
            node.params[spec.name] = widget.value()
            widget.valueChanged.connect(
                lambda value, name=spec.name: self._on_param_changed(name, value)
            )
            if isinstance(widget, NumericEntryControl):
                self._configure_rescale_axis_control(widget)
            if spec.name in {
                "x_scale",
                "y_scale",
                "z_scale",
                "x_size",
                "y_size",
                "z_size",
            }:
                self._add_rescale_axis_reset_button(node_id, spec.name, widget)
            self.parameter_form.addRow(spec.label, widget)
            self._parameter_widgets[spec.name] = widget
        self.parameter_group.setHidden(False)

    @staticmethod
    def _configure_rescale_axis_control(widget: NumericEntryControl) -> None:
        widget.layout().setSpacing(3)
        widget.value_box.setMinimumWidth(112)
        widget.value_box.setMaximumWidth(122)
        widget.value_box.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        widget.value_box.lineEdit().setTextMargins(0, 0, 0, 0)
        widget.value_box.setStyleSheet(
            "QSpinBox, QDoubleSpinBox { padding-left: 1px; padding-right: 1px; }"
            "QSpinBox::up-button, QDoubleSpinBox::up-button { width: 18px; }"
            "QSpinBox::down-button, QDoubleSpinBox::down-button { width: 18px; }"
        )

    def _rescale_axes_visible_specs(self, node_id: str) -> tuple[ParameterSpec, ...]:
        node = self.pipeline.nodes[node_id]
        base_specs = {
            spec.name: spec for spec in self.pipeline.node_parameter_specs(node_id)
        }
        mode = str(node.params.get("resize_mode", "Scale factor"))
        specs = [
            ParameterSpec(
                "resize_mode",
                "Resize using",
                "choice",
                "Scale factor",
                0,
                0,
                1,
                choices=("Scale factor", "Output size"),
                choice_labels=("Scale factor", "Output size (pixels)"),
            )
        ]
        options = self._rescale_axis_options_by_role(node_id)
        for role in ("x", "y", "z"):
            option = options.get(role)
            if option is None:
                continue
            if mode == "Output size":
                current = max(int(node.params.get(f"{role}_size", option.size)), 1)
                specs.append(
                    ParameterSpec(
                        f"{role}_size",
                        self._rescale_axis_label(node_id, role, mode),
                        "int",
                        int(option.size),
                        1,
                        max(int(option.size) * 4, current, 32),
                        1,
                    )
                )
            else:
                specs.append(
                    self._effective_parameter_spec(
                        node_id,
                        base_specs[f"{role}_scale"],
                    )
                )
        specs.append(base_specs["lock_xy"])
        specs.append(
            self._effective_parameter_spec(node_id, base_specs["interpolation"])
        )
        specs.append(base_specs["anti_aliasing"])
        return tuple(specs)

    def _add_rescale_axis_reset_button(
        self,
        node_id: str,
        name: str,
        widget: NumericEntryControl,
    ) -> None:
        role = name.split("_", maxsplit=1)[0]
        option = self._rescale_axis_options_by_role(node_id).get(role)
        reset_value = int(option.size) if name.endswith("_size") and option else 1.0
        button = QToolButton(widget)
        button.setIcon(_toolbar_icon("reset"))
        button.setIconSize(QSize(14, 14))
        button.setFixedSize(20, 20)
        if name.endswith("_size"):
            button.setToolTip(f"Reset {role.upper()} to its input size.")
        else:
            button.setToolTip(f"Reset {role.upper()} scale to 1.")
        button.clicked.connect(
            lambda _checked=False, value=reset_value: widget.value_box.setValue(value)
        )
        widget.layout().addWidget(button)
        self._parameter_widgets[f"{name}_reset"] = button

    def _render_born_wolf_psf_parameters(self, node_id: str) -> None:
        node = self.pipeline.nodes[node_id]
        specs = {
            spec.name: self._effective_parameter_spec(node_id, spec)
            for spec in self.pipeline.node_parameter_specs(node_id)
        }
        auto = bool(node.params.get("auto_parameters", True))
        resolution = self._born_wolf_psf_resolution(node_id)
        managed_names = set(BORN_WOLF_PSF_AUTO_PARAMETERS) | {"channel"}
        auto_channel_count = self._born_wolf_psf_auto_channel_count(node_id)
        if not auto:
            self._initialize_manual_born_wolf_psf_params(node_id, resolution)

        for name, spec in specs.items():
            bounds = self._born_wolf_psf_bounds(
                spec,
                auto=auto,
                resolution=resolution,
            )
            if spec.kind == "choice":
                widget = ChoiceControl(spec, node.params.get(name), bounds)
            elif spec.kind == "bool":
                widget = BoolControl(spec, node.params.get(name), bounds)
            else:
                value = node.params.get(name)
                if auto and name in managed_names:
                    result = resolution.parameters.get(name)
                    if (
                        name == "channel"
                        and self._born_wolf_psf_requests_all_channels(node_id)
                        and auto_channel_count > 1
                    ):
                        value = -1
                    else:
                        value = result.value if result is not None else None
                    if value is None:
                        value = 0
                widget = NumericEntryControl(spec, value, bounds)
                widget.setEnabled(not (auto and name in managed_names))

            if name == "auto_parameters":
                widget.valueChanged.connect(self._on_born_wolf_auto_changed)
            elif name in {"spatial_mode", "channel"}:
                widget.valueChanged.connect(
                    lambda value, param=name: self._on_born_wolf_rerender_param_changed(
                        param,
                        value,
                    )
                )
            else:
                widget.valueChanged.connect(
                    lambda value, param=name: self._on_param_changed(param, value)
                )

            label_widget = QLabel(spec.label)
            field_widget: QWidget = widget
            if name in managed_names:
                result = resolution.parameters.get(name)
                unresolved = bool(
                    auto
                    and result is not None
                    and result.required
                    and not result.resolved
                )
                status = QLabel(self._born_wolf_psf_status_text(result, auto=auto))
                status.setWordWrap(True)
                status.setStyleSheet(
                    "color: #f87171;" if unresolved else "color: #94a3b8;"
                )
                if unresolved:
                    label_widget.setStyleSheet("color: #f87171;")
                if (
                    name == "channel"
                    and auto
                    and self._born_wolf_psf_requests_all_channels(node_id)
                    and auto_channel_count > 1
                ):
                    status.setText(f"all channels ({auto_channel_count})")
                row = QWidget()
                layout = QHBoxLayout(row)
                layout.setContentsMargins(0, 0, 0, 0)
                layout.setSpacing(6)
                layout.addWidget(widget, 0)
                layout.addWidget(status, 1)
                field_widget = row
                self._parameter_widgets[f"{name}_status"] = status
                self._parameter_widgets[f"{name}_label"] = label_widget

            self.parameter_form.addRow(label_widget, field_widget)
            self._parameter_widgets[name] = widget
        self.parameter_group.setHidden(False)

    def _born_wolf_psf_resolution(self, node_id: str):
        node = self.pipeline.nodes[node_id]
        data = self.pipeline.input_data_for_node(node_id)
        state = self.pipeline.input_state_for_node(node_id)
        shape = tuple(np.asarray(data).shape) if data is not None else ()
        axis_names = (
            tuple(axis.name for axis in state.axes) if state is not None else ()
        )
        axis_types = (
            tuple(axis.type for axis in state.axes) if state is not None else ()
        )
        axis_scales = (
            tuple(axis.scale for axis in state.axes) if state is not None else ()
        )
        axis_units = (
            tuple(axis.unit for axis in state.axes) if state is not None else ()
        )
        channels = state.channels if state is not None else ()
        acquisition = state.acquisition if state is not None else None
        return resolve_born_wolf_psf_parameters(
            shape,
            node.params.get("spatial_mode", "Auto from axes"),
            auto_parameters=bool(node.params.get("auto_parameters", True)),
            wavelength_nm=node.params.get("wavelength_nm", 0.0),
            numerical_aperture=node.params.get("numerical_aperture", 0.0),
            refractive_index=node.params.get("refractive_index", 0.0),
            pixel_size_xy_um=node.params.get("pixel_size_xy_um", 0.0),
            z_step_um=node.params.get("z_step_um", 0.0),
            channel=node.params.get("channel", -1),
            resolved_spatial_ndim=node.params.get("resolved_spatial_ndim"),
            axis_types=axis_types,
            axis_names=axis_names,
            axis_scales=axis_scales,
            axis_units=axis_units,
            channel_emission_wavelengths=tuple(
                channel.emission_wavelength for channel in channels
            ),
            channel_emission_wavelength_units=tuple(
                channel.emission_wavelength_unit for channel in channels
            ),
            channel_excitation_wavelengths=tuple(
                channel.excitation_wavelength for channel in channels
            ),
            channel_excitation_wavelength_units=tuple(
                channel.excitation_wavelength_unit for channel in channels
            ),
            objective_lens_na=(
                None if acquisition is None else acquisition.objective_na
            ),
            objective_refractive_index=(
                None if acquisition is None else acquisition.refractive_index
            ),
        )

    def _born_wolf_psf_requests_all_channels(self, node_id: str) -> bool:
        node = self.pipeline.nodes.get(node_id)
        if node is None:
            return False
        try:
            channel = int(node.params.get("channel", -1))
        except Exception:
            channel = -1
        return bool(node.params.get("auto_parameters", True)) and channel < 0

    def _born_wolf_psf_auto_channel_count(self, node_id: str) -> int:
        state = self.pipeline.input_state_for_node(node_id)
        if state is not None and getattr(state, "channels", None):
            return len(state.channels)
        data = self.pipeline.input_data_for_node(node_id)
        if data is None:
            return 0
        shape = tuple(getattr(data, "shape", ()) or ())
        if state is not None:
            for index, axis in enumerate(state.axes):
                if axis.type == "channel" or axis.name.lower() == "c":
                    if 0 <= index < len(shape):
                        return int(shape[index])
        if len(shape) >= 3 and shape[-1] in {3, 4}:
            return int(shape[-1])
        return 0

    def _born_wolf_psf_bounds(
        self,
        spec: ParameterSpec,
        *,
        auto: bool,
        resolution,
    ) -> ParameterBounds:
        if spec.kind == "choice":
            return ParameterBounds(0, max(len(spec.choices) - 1, 0), 1, 0)
        if spec.kind == "bool":
            return ParameterBounds(spec.minimum, spec.maximum, spec.step, spec.decimals)
        minimum = spec.minimum
        if not auto and spec.name in BORN_WOLF_PSF_AUTO_PARAMETERS:
            minimum = 0.0001 if spec.decimals else 1
        if not auto and spec.name == "channel":
            minimum = 0
        value = None
        result = resolution.parameters.get(spec.name)
        if result is not None:
            value = result.value
        try:
            current = float(value if value is not None else spec.default)
        except Exception:
            current = float(spec.default)
        maximum = max(float(spec.maximum), current * 1.25, float(minimum))
        return ParameterBounds(
            minimum,
            maximum,
            spec.step,
            spec.decimals,
            expandable=False,
            entry_minimum=minimum,
            entry_maximum=max(float(spec.maximum), maximum),
        )

    def _born_wolf_psf_status_text(self, result, *, auto: bool) -> str:
        if result is None:
            return ""
        if not auto:
            return "manual"
        if not result.resolved:
            return "Unresolved"
        if result.source == "metadata":
            return "auto: metadata"
        if result.source == "not used":
            return "not used for 2D"
        if result.source == "manual":
            return "manual override"
        return "auto"

    def _on_born_wolf_auto_changed(self, value) -> None:
        node_id = self._selected_node_id
        node = self.pipeline.nodes.get(node_id)
        if node is None or node.params.get("auto_parameters") == bool(value):
            return
        previous_resolution = self._born_wolf_psf_resolution(node_id)
        self._record_parameter_undo(node_id, "auto_parameters")
        self.pipeline.set_param(node_id, "auto_parameters", bool(value))
        if not bool(value):
            self._initialize_manual_born_wolf_psf_params(
                node_id,
                previous_resolution,
            )
        self._mark_pipeline_dirty(node_id)
        self._render_parameters(node_id)
        self._sync_node_output_ports(node_id)
        self._debounce_timer.start()

    def _on_born_wolf_rerender_param_changed(self, name: str, value) -> None:
        self._on_param_changed(name, value)
        if self._selected_node_id in self.pipeline.nodes:
            self._render_parameters(self._selected_node_id)

    def _initialize_manual_born_wolf_psf_params(self, node_id: str, resolution) -> None:
        node = self.pipeline.nodes[node_id]
        for name, default in BORN_WOLF_PSF_MANUAL_DEFAULTS.items():
            current = node.params.get(name)
            try:
                initialized = (
                    float(current) > 0 if name != "channel" else int(current) >= 0
                )
            except Exception:
                initialized = False
            if initialized:
                continue
            result = resolution.parameters.get(name)
            value = (
                result.value
                if result is not None
                and result.value is not None
                and (name == "channel" or float(result.value) > 0)
                else default
            )
            node.params[name] = value

    def _add_operation_note(self, text: str) -> None:
        note = QLabel(text)
        note.setWordWrap(True)
        note.setStyleSheet("color: #f59e0b;")
        self.parameter_form.addRow(note)
        self._parameter_widgets["operation_notice"] = note

    def _stack_processing_note(self, node_id: str) -> str:
        node = self.pipeline.nodes.get(node_id)
        if node is None:
            return ""
        spec = self.pipeline.operation_spec(node.operation_id)
        if not spec.stack_processing_note:
            return ""
        if (
            self.pipeline.input_state_for_node(node_id) is None
            and self.pipeline.input_data_for_node(node_id) is None
        ):
            return ""
        if self._input_spatial_count(node_id) < 3:
            return ""
        return spec.stack_processing_note

    def _operation_help_note(self, node_id: str) -> str:
        node = self.pipeline.nodes.get(node_id)
        if node is None:
            return ""
        if node.operation_id in {"h_maxima_markers", "auto_watershed_from_mask"}:
            return (
                "H tuning guide:\n"
                "- H is a peak-prominence threshold on the distance map, in "
                "pixels/voxels.\n"
                "- 0 uses all local maxima.\n"
                "- Around 0 to 2 is usually the useful range; larger values only "
                "matter for larger objects or deeper peak separations."
            )
        if node.operation_id == "marker_controlled_watershed":
            return (
                "Input guide:\n"
                "- Image / distance: elevation image or distance map.\n"
                "- Markers: non-negative integer seed labels.\n"
                "- Mask: foreground constraint region (>0 = inside)."
            )
        if node.operation_id == "measure_3d_mesh_morphology":
            return (
                "3D mesh morphology requires true 3D label input. It uses "
                "spatial scale metadata for anisotropic Z/Y/X spacing, skips "
                "tiny objects below the minimum voxel count, and reports failed "
                "mesh or convex-hull metrics as NaN with a status message."
            )
        return ""

    def _render_combine_channels_parameters(self, node_id: str) -> None:
        node = self.pipeline.nodes[node_id]
        specs = {
            spec.name: spec for spec in self.pipeline.node_parameter_specs(node_id)
        }
        count_spec = specs["input_count"]
        count = self._combine_channels_input_count(node)
        node.params["input_count"] = count
        node.params["channel_colors"] = ",".join(
            self._combine_channels_colors(node),
        )

        count_widget = ParameterControl(
            count_spec,
            count,
            self._parameter_bounds_for(node_id, count_spec),
        )
        count_widget.valueChanged.connect(self._on_combine_channels_input_count_changed)
        self.parameter_form.addRow(count_spec.label, count_widget)
        self._parameter_widgets["input_count"] = count_widget

        for index, color in enumerate(self._combine_channels_colors(node)):
            spec = ParameterSpec(
                f"channel_color_{index}",
                f"Channel {index + 1} colour",
                "choice",
                color,
                0,
                0,
                1,
                choices=CHANNEL_COLOR_CHOICES,
            )
            widget = ChoiceControl(
                spec,
                color,
                ParameterBounds(0, len(CHANNEL_COLOR_CHOICES) - 1, 1, 0),
            )
            widget.valueChanged.connect(
                lambda value, slot=index: self._on_channel_color_changed(slot, value)
            )
            self.parameter_form.addRow(spec.label, widget)
            self._parameter_widgets[spec.name] = widget
        self._sync_combine_channels_graph_ports(node_id)

    def _render_image_source_parameters(self, node_id: str) -> None:
        node = self.pipeline.nodes[node_id]
        inspection = self._source_inspection_for_node(node)
        control = ImageSourceControl(
            self._image_source_value(node),
            layer_names=self._available_layer_names(),
            sample_names=self._sample_names(),
            series_options=self._source_series_options(inspection),
            source_summary=self._source_summary(inspection, node),
        )
        self._apply_image_source_params(node_id, control.value())
        control.valueChanged.connect(self._on_image_source_changed)
        self.parameter_form.addRow(control)
        self._parameter_widgets["image_source"] = control
        self._render_channel_color_controls(node_id)

    def _render_assign_channel_colors_parameters(self, node_id: str) -> None:
        self.parameter_group.setHidden(False)
        self._render_channel_color_controls(node_id)

    def _image_source_value(self, node) -> dict[str, object]:
        mode = str(node.params.get("source_mode", "napari layer"))
        layer_name = str(node.params.get("layer_name", "")).strip()
        if mode == "napari layer" and not layer_name:
            layer_name = self._default_input_layer_name()
        return {
            "source_mode": mode,
            "layer_name": layer_name,
            "file_path": node.params.get("file_path", ""),
            "sample_name": node.params.get("sample_name", ""),
            "series_index": node.params.get("series_index", 0),
            "binding_mode": node.params.get("binding_mode", "single item"),
        }

    def _apply_image_source_params(
        self,
        node_id: str,
        value: dict[str, object],
    ) -> None:
        value = self._normalized_image_source_value(value)
        node = self.pipeline.nodes[node_id]
        for name in (
            "source_mode",
            "layer_name",
            "file_path",
            "sample_name",
            "series_index",
            "binding_mode",
        ):
            node.params[name] = value.get(name, "")

    def _render_channel_color_controls(self, node_id: str) -> None:
        count = self._channel_color_control_count(node_id)
        if count <= 0:
            if self.pipeline.nodes[node_id].operation_id == "input":
                return
            note = QLabel("No channel axis detected for colour assignment.")
            note.setWordWrap(True)
            note.setStyleSheet("color: #94a3b8;")
            self.parameter_form.addRow(note)
            return
        state = self._channel_color_reference_state(node_id)
        colors = self._node_channel_color_choices(node_id, count, state)
        labels = self._channel_color_control_labels(count, state)
        for index, color in enumerate(colors):
            spec = ParameterSpec(
                f"channel_color_{index}",
                f"{labels[index]} colour",
                "choice",
                color,
                0,
                0,
                1,
                choices=CHANNEL_COLOR_CHOICES,
            )
            widget = ChoiceControl(
                spec,
                color,
                ParameterBounds(0, len(CHANNEL_COLOR_CHOICES) - 1, 1, 0),
            )
            widget.valueChanged.connect(
                lambda value, slot=index: self._on_metadata_channel_color_changed(
                    slot,
                    value,
                )
            )
            self.parameter_form.addRow(spec.label, widget)
            self._parameter_widgets[spec.name] = widget

    def _channel_color_control_count(self, node_id: str) -> int:
        state = self._channel_color_reference_state(node_id)
        if state is not None:
            for index, axis in enumerate(state.axes):
                if (axis.type == "channel" or axis.name.lower() == "c") and (
                    axis.name.lower() not in {"rgb", "rgba"}
                ):
                    count = int(state.shape[index])
                    return min(count, MAX_CHANNEL_COLOR_CONTROLS) if count > 1 else 0
            return 0
        data = (
            self.pipeline.outputs.get(node_id)
            if self.pipeline.nodes[node_id].operation_id == "input"
            else self.pipeline.input_data_for_node(node_id)
        )
        if data is None:
            return 0
        axis = self._selected_channel_axis(node_id, np.asarray(data))
        if axis is None:
            return 0
        count = int(np.asarray(data).shape[axis])
        return min(count, MAX_CHANNEL_COLOR_CONTROLS) if count > 1 else 0

    def _channel_color_reference_state(self, node_id: str):
        node = self.pipeline.nodes.get(node_id)
        if node is None:
            return None
        if node.operation_id == "input":
            return self.pipeline.output_states.get(node_id)
        return self.pipeline.input_state_for_node(node_id)

    def _normalized_image_source_value(
        self,
        value: dict[str, object],
    ) -> dict[str, object]:
        normalized = dict(value)
        if (
            str(normalized.get("source_mode", "napari layer")) == "napari layer"
            and not str(normalized.get("layer_name", "")).strip()
        ):
            normalized["layer_name"] = self._default_input_layer_name()
        return normalized

    def _node_channel_color_choices(self, node_id: str, count: int, state) -> list[str]:
        node = self.pipeline.nodes[node_id]
        raw = str(node.params.get("channel_colors", "")).strip()
        if raw:
            colors = [part.strip().title() for part in raw.split(",") if part.strip()]
            defaults = self._metadata_or_default_channel_colors(state, count)
            while len(colors) < count:
                colors.append(defaults[len(colors)])
        else:
            colors = self._metadata_or_default_channel_colors(state, count)
        valid = {choice.lower(): choice for choice in CHANNEL_COLOR_CHOICES}
        fallback = self._metadata_or_default_channel_colors(state, count)
        return [
            valid.get(str(color).lower(), fallback[index])
            for index, color in enumerate(colors[:count])
        ]

    def _metadata_or_default_channel_colors(self, state, count: int) -> list[str]:
        metadata_colors = ()
        if state is not None:
            metadata_colors = tuple(channel.color for channel in state.channels)
        return channel_color_labels_from_metadata(metadata_colors, count)

    def _channel_color_control_labels(self, count: int, state) -> list[str]:
        labels: list[str] = []
        channels = tuple(getattr(state, "channels", ())) if state is not None else ()
        for index in range(count):
            label = ""
            if index < len(channels):
                label = str(getattr(channels[index], "name", "")).strip()
            labels.append(label or f"Channel {index + 1}")
        return labels

    def _on_metadata_channel_color_changed(self, slot: int, value) -> None:
        node_id = self._selected_node_id
        node = self.pipeline.nodes.get(node_id)
        if node is None or node.operation_id not in {"input", "assign_channel_colors"}:
            return
        count = self._channel_color_control_count(node_id)
        if slot >= count:
            return
        state = self._channel_color_reference_state(node_id)
        colors = self._node_channel_color_choices(node_id, count, state)
        if colors[slot] == str(value):
            return
        self._record_parameter_undo(node_id, f"channel_color_{slot}")
        colors[slot] = str(value)
        node.params["channel_colors"] = ",".join(colors)
        self._mark_pipeline_dirty(node_id)
        self._update_thumbnails()
        self._debounce_timer.start()

    def _on_image_source_changed(self, value: dict[str, object]) -> None:
        value = self._normalized_image_source_value(value)
        node = self.pipeline.nodes[self._selected_node_id]
        if self._image_source_value(node) == value:
            return
        self._record_parameter_undo(self._selected_node_id, "image_source")
        previous_path = str(node.params.get("file_path", ""))
        previous_binding = str(node.params.get("binding_mode", "single item"))
        self._apply_image_source_params(self._selected_node_id, value)
        if not (
            str(value.get("source_mode", "")) == "file path"
            and str(value.get("binding_mode", "")) == "collection"
            and not str(value.get("file_path", "")).strip()
        ):
            self._interactive_collection_source_paths.pop(
                self._selected_node_id,
                None,
            )
        if (
            str(value.get("file_path", "")) != previous_path
            or str(value.get("binding_mode", "")) != previous_binding
        ):
            QTimer.singleShot(0, self._refresh_image_source_options)
        self._mark_pipeline_dirty(self._selected_node_id)
        self._debounce_timer.start()

    def _refresh_image_source_options(self) -> None:
        node = self.pipeline.nodes.get(self._selected_node_id)
        control = self._parameter_widgets.get("image_source")
        if node is None or not isinstance(control, ImageSourceControl):
            return
        inspection = self._source_inspection_for_node(node)
        control.set_options(
            self._available_layer_names(),
            self._sample_names(),
            series_options=self._source_series_options(inspection),
            source_summary=self._source_summary(inspection, node),
            value=self._image_source_value(node),
            emit=False,
        )
        self._apply_image_source_params(self._selected_node_id, control.value())

    def _source_inspection_for_node(self, node) -> SourceInspection | None:
        if str(node.params.get("source_mode", "")) != "file path":
            return None
        path = str(node.params.get("file_path", "")).strip()
        if not path:
            return None
        try:
            return self._inspect_source_file(path)
        except Exception:
            return None

    @staticmethod
    def _source_series_options(
        inspection: SourceInspection | None,
    ) -> list[tuple[int, str]]:
        if inspection is None:
            return []
        return [(series.index, series.label) for series in inspection.series]

    @staticmethod
    def _source_summary(inspection: SourceInspection | None, node) -> str:
        if inspection is None:
            return ""
        mode = str(node.params.get("binding_mode", "single item"))
        prefix = (
            "Collection binding; the selected series is the interactive "
            "representative. "
            if mode == "collection"
            else ""
        )
        summary = (
            f"{prefix}{inspection.format}; "
            f"{len(inspection.series)} image series discovered."
        )
        deconvolved, method = detect_deconvolution_metadata(
            inspection.original_metadata
        )
        if deconvolved is True:
            detail = f" ({method})" if method else ""
            summary += f" Source metadata mentions deconvolution{detail}."
        return summary

    def _render_select_axis_slice_parameters(self, node_id: str) -> None:
        node = self.pipeline.nodes[node_id]
        control = AxisSliceControl(
            self._axis_slice_options_for(node_id),
            self._select_axis_slice_value(node),
        )
        self._apply_select_axis_slice_params(node_id, control.value())
        control.valueChanged.connect(self._on_select_axis_slice_changed)
        self.parameter_form.addRow(control)
        self._parameter_widgets["axis_slice"] = control

    def _render_reorder_axes_parameters(self, node_id: str) -> None:
        node = self.pipeline.nodes[node_id]
        control = ReorderAxesControl(
            self._axis_slice_options_for(node_id),
            str(node.params.get("order", "")),
        )
        self._apply_reorder_axes_params(node_id, control.value())
        control.valueChanged.connect(self._on_reorder_axes_changed)
        self.parameter_form.addRow(control)
        self._parameter_widgets["order"] = control

    def _render_select_table_columns_parameters(self, node_id: str) -> None:
        node = self.pipeline.nodes[node_id]
        control = SelectTableColumnsControl(
            self._input_table_columns_for(node_id),
            str(node.params.get("columns", "auto")),
        )
        control.valueChanged.connect(self._on_select_table_columns_changed)
        self.parameter_form.addRow(control)
        self._parameter_widgets["columns"] = control
        self.parameter_group.setHidden(False)

    def _parameter_value_changed(self, spec, previous, current) -> bool:
        """Return ``True`` only when a parameter value meaningfully changed.

        Numeric controls round-trip through ``QDoubleSpinBox``, so a value the
        user produced with the spinner (e.g. ``1.74 + 0.1`` stored as
        ``1.8399999999999999``) differs from the box's cleanly rounded
        ``value()`` (``1.84``) by floating-point noise alone. Treat such values
        as unchanged so refreshing the controls never forces a spurious
        recompute.
        """
        if previous is None:
            return current is not None
        if spec.kind in {"int", "float"}:
            try:
                previous_number = float(previous)
                current_number = float(current)
            except (TypeError, ValueError):
                return previous != current
            if spec.kind == "int":
                return int(round(previous_number)) != int(round(current_number))
            decimals = max(int(getattr(spec, "decimals", 2) or 0), 0)
            return round(previous_number, decimals) != round(current_number, decimals)
        return previous != current

    def _refresh_selected_parameter_controls(self) -> bool:
        if self._selected_node_id not in self.pipeline.nodes:
            return False
        changed = False
        node = self.pipeline.nodes[self._selected_node_id]
        if node.operation_id == "input":
            widget = self._parameter_widgets.get("image_source")
            if isinstance(widget, ImageSourceControl):
                previous = dict(node.params)
                widget.set_options(
                    self._available_layer_names(),
                    self._sample_names(),
                    series_options=self._source_series_options(
                        self._source_inspection_for_node(node)
                    ),
                    source_summary=self._source_summary(
                        self._source_inspection_for_node(node),
                        node,
                    ),
                    value=self._image_source_value(node),
                    emit=False,
                )
                self._apply_image_source_params(
                    self._selected_node_id,
                    widget.value(),
                )
                changed = previous != node.params
            visible_color_controls = sum(
                name.startswith("channel_color_") for name in self._parameter_widgets
            )
            if visible_color_controls != self._channel_color_control_count(node.id):
                self._render_parameters(node.id)
            return changed
        if node.operation_id == "select_axis_slice":
            widget = self._parameter_widgets.get("axis_slice")
            if isinstance(widget, AxisSliceControl):
                previous = dict(node.params)
                widget.set_options(
                    self._axis_slice_options_for(self._selected_node_id),
                    self._select_axis_slice_value(node),
                    emit=False,
                )
                self._apply_select_axis_slice_params(
                    self._selected_node_id,
                    widget.value(),
                )
                changed = previous != node.params
            return changed
        if node.operation_id == "reorder_axes":
            widget = self._parameter_widgets.get("order")
            if isinstance(widget, ReorderAxesControl):
                previous = dict(node.params)
                widget.set_options(
                    self._axis_slice_options_for(self._selected_node_id),
                    str(node.params.get("order", "")),
                    emit=False,
                )
                self._apply_reorder_axes_params(
                    self._selected_node_id,
                    widget.value(),
                )
                changed = previous != node.params
            return changed
        if node.operation_id == "select_table_columns":
            widget = self._parameter_widgets.get("columns")
            if isinstance(widget, SelectTableColumnsControl):
                widget.set_options(
                    self._input_table_columns_for(self._selected_node_id),
                    str(node.params.get("columns", "auto")),
                    emit=False,
                )
            return changed
        if node.operation_id == "rescale_axes":
            return self._refresh_rescale_axes_controls(self._selected_node_id)
        if node.operation_id == "born_wolf_psf":
            self._render_parameters(self._selected_node_id)
            self._sync_node_output_ports(self._selected_node_id)
            return False
        if self._sync_rescale_output_range_defaults(self._selected_node_id):
            changed = True
        for spec in self.pipeline.node_parameter_specs(self._selected_node_id):
            widget = self._parameter_widgets.get(spec.name)
            if widget is None:
                continue
            spec = self._effective_parameter_spec(self._selected_node_id, spec)
            previous = node.params.get(spec.name)
            if spec.kind == "choice" and previous not in spec.choices:
                previous = spec.default
                node.params[spec.name] = previous
                changed = True
            bounds = self._parameter_bounds_for(self._selected_node_id, spec)
            locked_split_channel = (
                node.operation_id == "split_channels"
                and spec.name == "preview_channel"
                and self._single_used_split_channel_port(node.id) is not None
            )
            presented_value = (
                self._single_used_split_channel_port(node.id)
                if locked_split_channel
                else previous
            )
            if isinstance(widget, ChoiceControl):
                widget.set_choices(
                    spec.choices,
                    presented_value,
                    emit=False,
                    choice_labels=spec.choice_labels,
                )
            widget.set_bounds(bounds, presented_value, emit=False)
            label = self.parameter_form.labelForField(widget)
            if isinstance(label, QLabel):
                label.setText(spec.label)
            if locked_split_channel:
                continue
            current = widget.value()
            if self._parameter_value_changed(spec, previous, current):
                changed = True
            node.params[spec.name] = current
        if node.operation_id == "fill_holes":
            self._update_fill_holes_scope_note()
        if node.operation_id == "gaussian_blur_3d":
            changed = (
                self._sync_gaussian_blur_3d_xy_lock(
                    self._selected_node_id,
                    source_name="sigma_y",
                )
                or changed
            )
        if node.operation_id == "rescale_axes":
            changed = (
                self._sync_rescale_axes_xy_lock(
                    self._selected_node_id,
                    source_name="x_scale",
                )
                or changed
            )
        return changed

    def _parameter_spec_hidden(self, node_id: str, spec) -> bool:
        node = self.pipeline.nodes.get(node_id)
        if (
            node is not None
            and node.operation_id == "auto_watershed_from_mask"
            and spec.name == "spatial_mode"
        ):
            return self._input_spatial_count(node_id) < 3
        if (
            node is not None
            and node.operation_id in GLOBAL_THRESHOLD_OPERATIONS
            and spec.name == "threshold_scope"
        ):
            data = self.pipeline.input_data_for_node(node_id)
            if data is None:
                return False
            state = self.pipeline.input_state_for_node(node_id)
            return not _histogram_has_stack_scope(data, state)
        if (
            node is not None
            and node.operation_id in {"measure_objects", "measure_objects_intensity"}
            and spec.name
            in {"include_2d_boundary_descriptors", "include_2d_shape_moments"}
        ):
            return self._input_spatial_count(node_id) >= 3
        if (
            node is not None
            and node.operation_id == "set_pixel_size"
            and spec.name == "z_size"
        ):
            return self._input_spatial_count(node_id) < 3
        if (
            node is not None
            and node.operation_id == "rescale_axes"
            and spec.name == "z_scale"
        ):
            return self._input_spatial_count(node_id) < 3
        if node is not None and node.operation_id == "rescale_intensity":
            percentile_mode = str(
                node.params.get(RESCALE_CUTOFF_MODE_PARAMETER, "Percentiles")
            ).lower().startswith("percent")
            if spec.name in RESCALE_VALUE_PARAMETERS:
                return percentile_mode
            if spec.name in RESCALE_PERCENTILE_PARAMETERS:
                return not percentile_mode
        if (
            node is not None
            and node.operation_id == "clip_intensity"
            and spec.name in CLIP_CUTOFF_PARAMETERS
        ):
            return str(node.params.get("cutoff_mode", "Data range")).lower().startswith(
                "data"
            )
        return False

    def _parameter_uses_numeric_entry_only(self, node_id: str, spec) -> bool:
        node = self.pipeline.nodes.get(node_id)
        if (
            node is not None
            and node.operation_id in COLOCALIZATION_THRESHOLD_OPERATIONS
            and spec.name in {"channel_1_threshold", "channel_2_threshold"}
        ):
            return True
        return (
            node is not None
            and node.operation_id == "set_pixel_size"
            and spec.name in {"x_size", "y_size", "z_size"}
        )

    def _effective_parameter_spec(self, node_id: str, spec):
        node = self.pipeline.nodes.get(node_id)
        if (
            node is not None
            and node.operation_id == "split_channels"
            and spec.name == "preview_channel"
            and self._single_used_split_channel_port(node_id) is not None
        ):
            return replace(spec, label=f"{spec.label} (only used output)")
        if spec.name == "spatial_mode":
            choices = self._available_spatial_modes(node_id)
            return replace(
                spec,
                choices=choices,
                choice_labels=self._spatial_mode_choice_labels(node_id, choices),
            )
        if (
            node is not None
            and node.operation_id == "clear_border_objects"
            and spec.name == "boundary_mode"
        ):
            return replace(
                spec,
                choices=self._available_clear_border_modes(node_id),
            )
        if (
            node is not None
            and node.operation_id == "fill_holes"
            and spec.name == "max_hole_size"
        ):
            spatial_ndim = self._selected_spatial_ndim(node_id)
            unit = "volume (voxels)" if spatial_ndim >= 3 else "area (pixels)"
            return replace(
                spec,
                label=f"Maximum hole {unit} (0 = fill all)",
            )
        if (
            node is not None
            and node.operation_id == "remove_small_objects"
            and spec.name == "min_size"
        ):
            spatial_ndim = self._selected_spatial_ndim(node_id)
            unit = "volume (voxels)" if spatial_ndim >= 3 else "area (pixels)"
            return replace(spec, label=f"Minimum object {unit}")
        if (
            node is not None
            and node.operation_id == "rescale_axes"
            and spec.name in {"x_scale", "y_scale", "z_scale"}
        ):
            return replace(
                spec,
                label=self._rescale_axis_scale_label(node_id, spec),
            )
        if (
            node is not None
            and node.operation_id == "rescale_axes"
            and spec.name == "interpolation"
        ):
            automatic = self._automatic_rescale_interpolation(node_id)
            labels = tuple(
                f"Auto - {automatic}" if choice == "Auto" else choice
                for choice in spec.choices
            )
            return replace(spec, choice_labels=labels)
        if (
            node is not None
            and node.operation_id == "project_image"
            and spec.name == "axes"
        ):
            choices, choice_labels = self._available_project_axis_choices(node_id)
            return replace(
                spec,
                choices=choices,
                choice_labels=choice_labels,
            )
        if (
            node is not None
            and node.operation_id == "born_wolf_psf"
            and spec.name == "channel"
        ):
            return replace(
                spec,
                label=(
                    "Channel (-1 = all channels)"
                    if bool(node.params.get("auto_parameters", True))
                    else "Channel"
                ),
            )
        if (
            node is not None
            and node.operation_id == "split_axis"
            and spec.name == "axis"
        ):
            choices, choice_labels = self._available_split_axis_choices(node_id)
            return replace(
                spec,
                choices=choices,
                choice_labels=choice_labels,
            )
        return spec

    def _spatial_mode_choice_labels(
        self,
        node_id: str,
        choices: tuple[str, ...],
    ) -> tuple[str, ...]:
        labels = list(choices)
        resolved = self._resolved_auto_spatial_mode_label(node_id)
        if resolved:
            for index, choice in enumerate(choices):
                if str(choice).startswith("Auto from axes"):
                    labels[index] = f"Auto from axes - using {resolved}"
                    break
        return tuple(labels)

    def _resolved_auto_spatial_mode_label(self, node_id: str) -> str:
        node = self.pipeline.nodes.get(node_id)
        if node is None:
            return ""
        mode = str(node.params.get("spatial_mode", "Auto from axes")).strip().lower()
        if not mode.startswith("auto"):
            return ""
        resolved = node.params.get("resolved_spatial_ndim")
        if resolved == 3:
            return "3D ZYX"
        if resolved == 2:
            return "2D YX"
        return "3D ZYX" if self._input_spatial_count(node_id) >= 3 else "2D YX"

    def _available_project_axis_choices(
        self,
        node_id: str,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        choices = ["auto"]
        labels = ["Auto (Z if present)"]
        options = self._axis_slice_options_for(node_id)
        if options:
            for option in options:
                choices.append(f"axis:{option.index}")
                labels.append(self._project_axis_choice_label(option))
        spatial_options = [
            option for option in options if option.axis_type.lower() == "space"
        ]
        if len(spatial_options) > 2:
            choices.append("non_yx_spatial")
            labels.append("All non-YX spatial axes")
        deduplicated: dict[str, str] = {}
        for value, label in zip(choices, labels, strict=True):
            deduplicated.setdefault(value, label)
        return tuple(deduplicated), tuple(deduplicated.values())

    @staticmethod
    def _project_axis_choice_label(option: AxisSliceOption) -> str:
        title = option.title
        if title.lower().startswith("axis "):
            return f"{title} (size {option.size})"
        detail = f"{option.axis_type}, size {option.size}"
        return f"{title} axis ({detail})"

    def _available_split_axis_choices(
        self,
        node_id: str,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        options = self._split_axis_options_from(
            self._axis_slice_options_for(node_id),
            fallback_all=True,
        )
        choices = tuple(f"axis:{option.index}" for option in options)
        labels = tuple(self._split_axis_choice_label(option) for option in options)
        if choices:
            return choices, labels
        return ("axis:0",), ("Axis 0",)

    @staticmethod
    def _split_axis_choice_label(option: AxisSliceOption) -> str:
        title = option.title
        if title.lower().startswith("axis "):
            return f"{title} (size {option.size})"
        detail = f"{option.axis_type}, size {option.size}"
        return f"{title} axis ({detail})"

    def _rescale_axis_scale_label(self, node_id: str, spec) -> str:
        role = spec.name.split("_", maxsplit=1)[0]
        return self._rescale_axis_label(node_id, role, "Scale factor")

    def _rescale_axis_label(self, node_id: str, role: str, mode: str) -> str:
        option = self._rescale_axis_options_by_role(node_id).get(role)
        if option is None:
            suffix = "output size" if mode == "Output size" else "scale factor"
            return f"{role.upper()} {suffix}"
        node = self.pipeline.nodes.get(node_id)
        if mode == "Output size":
            target_size = max(
                (
                    int(node.params.get(f"{role}_size", option.size))
                    if node
                    else option.size
                ),
                1,
            )
            return f"{role.upper()} output size ({option.size} -> {target_size})"
        scale = self._rescale_axis_scale_value(node, role)
        target_size = max(int(round(option.size * scale)), 1)
        return f"{role.upper()} scale factor ({option.size} -> {target_size})"

    def _automatic_rescale_interpolation(self, node_id: str) -> str:
        state = self.pipeline.input_state_for_node(node_id)
        kind = str(getattr(state, "kind", "")).strip().lower()
        data = self.pipeline.input_data_for_node(node_id)
        if (data is not None and np.asarray(data).dtype == bool) or any(
            token in kind for token in ("mask", "label")
        ):
            return "Nearest neighbor"
        return "Linear"

    def _rescale_axis_scale_value(self, node, role: str) -> float:
        if node is None:
            return 1.0
        if role == "x":
            return _positive_scale_float(node.params.get("x_scale"), 1.0)
        if role == "y" and bool(node.params.get("lock_xy", True)):
            return _positive_scale_float(node.params.get("x_scale"), 1.0)
        return _positive_scale_float(node.params.get(f"{role}_scale"), 1.0)

    def _rescale_axis_options_by_role(
        self,
        node_id: str,
    ) -> dict[str, AxisSliceOption]:
        options = self._axis_slice_options_for(node_id)
        role_map: dict[str, AxisSliceOption] = {}
        for option in options:
            name = option.name.strip().lower()
            if name in {"x", "y", "z"}:
                role_map[name] = option
        spatial_options = [
            option for option in options if option.axis_type.lower() == "space"
        ]
        if spatial_options:
            role_map.setdefault("x", spatial_options[-1])
        if len(spatial_options) >= 2:
            role_map.setdefault("y", spatial_options[-2])
        if len(spatial_options) >= 3:
            role_map.setdefault("z", spatial_options[-3])
        if self.pipeline.input_data_for_node(node_id) is None:
            spatial_count = self._input_spatial_count(node_id)
            fallback_roles = ("x", "y", "z")[:spatial_count]
            for fallback_index, role in enumerate(fallback_roles):
                role_map.setdefault(
                    role,
                    AxisSliceOption(
                        fallback_index,
                        role,
                        "space",
                        1,
                    ),
                )
        return role_map

    def _available_spatial_modes(self, node_id: str) -> tuple[str, ...]:
        node = self.pipeline.nodes.get(node_id)
        if node is not None and node.operation_id == "fill_holes":
            choices = (
                "Auto from axes",
                "2D per XY slice (advanced)",
                "3D ZYX volume",
            )
        elif node is not None and node.operation_id == "measure_3d_mesh_morphology":
            choices = ("Auto from axes", "3D ZYX")
        else:
            choices = ("Auto from axes", "2D YX", "3D ZYX")
        if self._input_spatial_count(node_id) >= 3:
            return choices
        if node is not None and node.operation_id == "measure_3d_mesh_morphology":
            return choices[:1]
        return choices[:2]

    def _available_clear_border_modes(self, node_id: str) -> tuple[str, ...]:
        choices = (
            "All spatial borders",
            "Lateral borders only (YX)",
        )
        return choices if self._input_spatial_count(node_id) >= 3 else choices[:1]

    def _input_spatial_count(self, node_id: str) -> int:
        state = self.pipeline.input_state_for_node(node_id)
        if state is not None:
            spatial_count = sum(axis.type == "space" for axis in state.axes)
            if spatial_count:
                return spatial_count
        data = self.pipeline.input_data_for_node(node_id)
        if data is None:
            return 3
        return min(max(np.asarray(data).ndim, 1), 3)

    def _selected_spatial_ndim(self, node_id: str) -> int:
        data = self.pipeline.input_data_for_node(node_id)
        if data is None:
            return 3 if self._input_spatial_count(node_id) >= 3 else 2
        return self._label_filter_spatial_ndim(node_id, np.asarray(data))

    def _parameter_bounds_for(self, node_id: str, spec) -> ParameterBounds:
        if spec.kind == "choice":
            return ParameterBounds(0, max(len(spec.choices) - 1, 0), 1, 0)
        node = self.pipeline.nodes.get(node_id)
        if (
            node is not None
            and node.operation_id in {"h_maxima_markers", "auto_watershed_from_mask"}
            and spec.name == "h"
        ):
            upper = min(float(spec.maximum), 5.0)
            return ParameterBounds(
                spec.minimum,
                upper,
                spec.step,
                spec.decimals,
                expandable=False,
                entry_minimum=min(float(spec.minimum), -1_000_000.0),
                entry_maximum=1_000_000.0,
            )
        if (
            node is not None
            and node.operation_id == "split_channels"
            and spec.name == "preview_channel"
        ):
            single_port = self._single_used_split_channel_port(node_id)
            if single_port is not None:
                return ParameterBounds(single_port, single_port, 1, 0)
            return self._channel_bounds(node_id, spec)
        if (
            node is not None
            and node.operation_id == "linear_scale_offset"
            and spec.name in {"alpha", "beta"}
        ):
            return self._contrast_parameter_bounds(node_id, spec)
        if (
            node is not None
            and node.operation_id == "rescale_intensity"
            and spec.name in RESCALE_VALUE_PARAMETERS
        ):
            return self._rescale_input_value_bounds(node_id, spec)
        if (
            node is not None
            and node.operation_id == "clip_intensity"
            and spec.name in CLIP_CUTOFF_PARAMETERS
        ):
            return self._intensity_input_value_bounds(node_id, spec)
        if (
            node is not None
            and node.operation_id in {"rolling_ball_background", "subtract_background"}
            and spec.name == "radius"
        ):
            return self._rolling_ball_radius_bounds(spec)
        if (
            node is not None
            and node.operation_id == "rescale_intensity"
            and spec.name in {"out_min", "out_max"}
        ):
            return self._rescale_output_bounds(node_id, spec)
        if (
            node is not None
            and node.operation_id == "rescale_axes"
            and spec.name in {"x_scale", "y_scale", "z_scale"}
        ):
            return ParameterBounds(
                spec.minimum,
                10.0,
                spec.step,
                spec.decimals,
                expandable=False,
                entry_minimum=spec.minimum,
                entry_maximum=1_000_000.0,
            )
        if spec.name in {"threshold", "low_threshold", "high_threshold"}:
            return self._threshold_bounds(node_id, spec)
        if spec.name == "axis":
            return self._axis_bounds(node_id, spec)
        if spec.name == "index":
            return self._slice_index_bounds(node_id, spec)
        if spec.name == "channel_axis":
            return self._channel_axis_bounds(node_id, spec)
        if spec.name in {"top", "bottom", "left", "right"}:
            return self._crop_bounds(node_id, spec)
        if spec.name in {"channel", "red_channel", "green_channel", "blue_channel"}:
            return self._channel_bounds(node_id, spec)
        if spec.name == "block_size":
            return self._block_size_bounds(node_id, spec)
        if (
            node is not None
            and node.operation_id == "filter_labels_by_volume"
            and spec.name in {"min_volume", "max_volume"}
        ):
            return self._label_volume_bounds(node_id, spec)
        if (
            node is not None
            and node.operation_id == "clear_border_objects"
            and spec.name == "border_buffer"
        ):
            return self._clear_border_buffer_bounds(node_id, spec)
        if (
            node is not None
            and node.operation_id == "fill_holes"
            and spec.name == "max_hole_size"
        ):
            return self._fill_hole_size_bounds(node_id, spec)
        if (
            node is not None
            and node.operation_id == "remove_small_objects"
            and spec.name == "min_size"
        ):
            return self._remove_small_object_bounds(node_id, spec)
        return ParameterBounds(
            spec.minimum,
            spec.maximum,
            spec.step,
            spec.decimals,
            expandable=True,
        )

    def _rolling_ball_radius_bounds(self, spec) -> ParameterBounds:
        slider_maximum = min(float(spec.maximum), ROLLING_BALL_RADIUS_SLIDER_MAX)
        return ParameterBounds(
            spec.minimum,
            slider_maximum,
            spec.step,
            spec.decimals,
            expandable=False,
            entry_minimum=spec.minimum,
            entry_maximum=spec.maximum,
        )

    def _axis_slice_options_for(self, node_id: str) -> list[AxisSliceOption]:
        data = self.pipeline.input_data_for_node(node_id)
        if data is None:
            return [AxisSliceOption(0, "axis 0", "unknown", 1)]
        arr = np.asarray(data)
        state = self.pipeline.input_state_for_node(node_id)
        options: list[AxisSliceOption] = []
        if state is not None and len(state.axes) == arr.ndim:
            for index, (axis, size) in enumerate(
                zip(state.axes, arr.shape, strict=True)
            ):
                options.append(
                    AxisSliceOption(
                        index,
                        axis.name,
                        axis.type,
                        int(size),
                    )
                )
            return options
        return [
            AxisSliceOption(index, f"axis {index}", "unknown", int(size))
            for index, size in enumerate(arr.shape)
        ]

    def _axis_slice_options_for_output(
        self,
        source_id: str,
        source_port: int,
    ) -> list[AxisSliceOption]:
        shape = self.pipeline._resolved_output_shape(source_id, source_port)
        if not shape:
            return []
        state = self.pipeline._resolved_output_state(source_id, source_port)
        if isinstance(state, ImageState) and len(state.axes) == len(shape):
            return [
                AxisSliceOption(index, axis.name, axis.type, int(size))
                for index, (axis, size) in enumerate(
                    zip(state.axes, shape, strict=True)
                )
            ]
        return [
            AxisSliceOption(index, f"axis {index}", "unknown", int(size))
            for index, size in enumerate(shape)
        ]

    def _split_axis_options_from(
        self,
        options: list[AxisSliceOption],
        *,
        fallback_all: bool,
    ) -> list[AxisSliceOption]:
        candidates = [
            option
            for option in options
            if self._is_split_axis_candidate(option, len(options))
        ]
        if candidates or not fallback_all:
            return candidates
        fallback = [option for option in options if option.size > 1]
        return fallback or options[:1]

    @staticmethod
    def _is_split_axis_candidate(
        option: AxisSliceOption,
        option_count: int,
    ) -> bool:
        if option.size <= 1:
            return False
        name = option.name.strip().lower()
        axis_type = option.axis_type.strip().lower()
        if name in {"series", "s"} or axis_type == "series":
            return False
        if name in {"c", "channel", "t", "time", "z"}:
            return True
        if axis_type in {"channel", "time"}:
            return True
        if axis_type == "space":
            return name not in {"x", "y"} and option.index < max(option_count - 1, 1)
        if axis_type == "unknown":
            return option.index < max(option_count - 2, 1)
        return name not in {"x", "y"}

    @staticmethod
    def _split_axis_index_from_params(
        params: dict[str, object],
        ndim: int,
    ) -> int:
        value = params.get("axis", "axis:0")
        text = str(value).strip().lower()
        if text.startswith("axis:"):
            text = text.split(":", 1)[1]
        try:
            axis = int(text)
        except ValueError:
            axis = 0
        if ndim <= 0:
            return 0
        if axis < 0:
            axis += ndim
        return max(0, min(axis, ndim - 1))

    def _select_axis_slice_value(self, node) -> dict:
        params = node.params
        return {
            "axis": params.get("axis", 0),
            "index": params.get("index", 0),
            "axes": params.get("axes", ""),
            "indices": params.get("indices", ""),
            "ranges": params.get("ranges", ""),
            "range_mode": params.get("range_mode", True),
            "remove_axes": params.get("remove_axes", ""),
            "remove_indices": params.get("remove_indices", ""),
        }

    def _apply_select_axis_slice_params(self, node_id: str, value: dict) -> None:
        node = self.pipeline.nodes[node_id]
        for name in (
            "axis",
            "index",
            "axes",
            "indices",
            "ranges",
            "range_mode",
            "remove_axes",
            "remove_indices",
        ):
            node.params[name] = value[name]

    def _on_select_axis_slice_changed(self, value: dict) -> None:
        node = self.pipeline.nodes[self._selected_node_id]
        if self._select_axis_slice_value(node) == value:
            return
        self._record_parameter_undo(self._selected_node_id, "axis_slice")
        self._apply_select_axis_slice_params(self._selected_node_id, value)
        self._mark_pipeline_dirty(self._selected_node_id)
        self._debounce_timer.start()

    def _apply_reorder_axes_params(self, node_id: str, value: str) -> None:
        self.pipeline.nodes[node_id].params["order"] = str(value)

    def _on_reorder_axes_changed(self, value: str) -> None:
        node = self.pipeline.nodes[self._selected_node_id]
        if str(node.params.get("order", "")) == str(value):
            return
        self._record_parameter_undo(self._selected_node_id, "order")
        self._apply_reorder_axes_params(self._selected_node_id, value)
        self._mark_pipeline_dirty(self._selected_node_id)
        self._debounce_timer.start()

    def _input_table_columns_for(self, node_id: str) -> list[str]:
        data = self.pipeline.input_data_for_node(node_id)
        if is_table_data(data):
            return [str(column) for column in data.columns]
        state = self.pipeline.input_state_for_node(node_id)
        columns = getattr(state, "columns", ())
        return [str(column) for column in columns]

    def _apply_select_table_columns_params(self, node_id: str, value: str) -> None:
        node = self.pipeline.nodes[node_id]
        node.params["columns"] = str(value)
        node.params.pop("selection_mode", None)
        node.params.pop("append_unlisted", None)

    def _on_select_table_columns_changed(self, value: str) -> None:
        node = self.pipeline.nodes[self._selected_node_id]
        if str(node.params.get("columns", "auto")) == str(value):
            return
        self._record_parameter_undo(self._selected_node_id, "columns")
        self._apply_select_table_columns_params(self._selected_node_id, value)
        self._mark_pipeline_dirty(self._selected_node_id)
        self._debounce_timer.start()

    def _on_combine_channels_input_count_changed(self, value) -> None:
        node_id = self._selected_node_id
        node = self.pipeline.nodes.get(node_id)
        if node is None or node.operation_id != "combine_channels":
            return
        if int(node.params.get("input_count", 2)) == int(value):
            return
        self._record_parameter_undo(node_id, "input_count")
        node.params["input_count"] = int(value)
        colors = self._combine_channels_colors(node)
        node.params["channel_colors"] = ",".join(colors)
        for connection in self.pipeline.trim_invalid_connections(node_id):
            self.graph_view.remove_connection(
                connection.source_id,
                connection.target_id,
                connection.target_port,
                notify=False,
            )
        self._sync_combine_channels_graph_ports(node_id)
        self._render_parameters(node_id)
        self._mark_pipeline_dirty(node_id)
        self._debounce_timer.start()

    def _on_channel_color_changed(self, slot: int, value) -> None:
        node_id = self._selected_node_id
        node = self.pipeline.nodes.get(node_id)
        if node is None or node.operation_id != "combine_channels":
            return
        colors = self._combine_channels_colors(node)
        if slot >= len(colors):
            return
        if colors[slot] == str(value):
            return
        self._record_parameter_undo(node_id, f"channel_color_{slot}")
        colors[slot] = str(value)
        node.params["channel_colors"] = ",".join(colors)
        self._sync_combine_channels_graph_ports(node_id)
        self._mark_pipeline_dirty(node_id)
        self._update_thumbnails()
        self._debounce_timer.start()

    def _sync_combine_channels_graph_ports(self, node_id: str) -> None:
        node = self.pipeline.nodes.get(node_id)
        if node is None or node.operation_id != "combine_channels":
            return
        colors = self._combine_channels_colors(node)
        labels = [f"Channel {index + 1}: {color}" for index, color in enumerate(colors)]
        graph_colors = [CHANNEL_COLOR_HEX.get(color.lower()) for color in colors]
        self.graph_view.set_node_input_ports(
            node_id,
            len(colors),
            labels,
            graph_colors,
            [node.input_type or "array"] * len(colors),
        )
        self._sync_port_tunnels()

    def _sync_node_input_ports(self, node_id: str) -> None:
        node = self.pipeline.nodes.get(node_id)
        if node is None or not node.has_input:
            return
        if node.operation_id == "combine_channels":
            self._sync_combine_channels_graph_ports(node_id)
            return
        input_ports = self.pipeline.input_ports(node_id)
        self.graph_view.set_node_input_ports(
            node_id,
            len(input_ports),
            [port.label for port in input_ports],
            [None for _port in input_ports],
            [port.input_type for port in input_ports],
        )
        self._sync_port_tunnels()

    def _sync_node_output_ports(self, node_id: str) -> None:
        spec = self.pipeline.operation_spec(self.pipeline.nodes[node_id].operation_id)
        ports = self.pipeline.output_ports(node_id)
        if not spec.is_multi_output:
            return
        labels = [port.label for port in ports]
        colors = [
            self._output_port_color(index, port) for index, port in enumerate(ports)
        ]
        data_types = [port.output_type for port in ports]
        self.graph_view.set_node_output_ports(
            node_id,
            len(ports),
            labels,
            colors,
            data_types,
        )
        self._sync_port_tunnels()

    @staticmethod
    def _output_port_color(index: int, port) -> str | None:
        """Pick a hex color for an output port.

        Named color ports (e.g. ``red``) use the matching palette entry;
        positional ports (e.g. ``channel_3``) cycle through the palette so
        each Split Channels output is visually distinct.
        """
        direct = CHANNEL_COLOR_HEX.get(port.name.lower())
        if direct is not None:
            return direct
        palette = list(CHANNEL_COLOR_HEX.values())
        return palette[index % len(palette)]

    def _sync_all_output_ports(self) -> None:
        for node_id in self.pipeline.nodes:
            self._sync_node_output_ports(node_id)
        self._sync_port_tunnels()

    def _sync_all_input_ports(self) -> None:
        for node_id in self.pipeline.nodes:
            self._sync_node_input_ports(node_id)
        self._sync_port_tunnels()

    def _refresh_dynamic_output_ports(self) -> None:
        """Resync graph ports for nodes whose output count varies per run.

        Dynamic multi-output nodes (e.g. Split Channels) only know their true
        channel count after processing an image. Once a run completes, drop any
        downstream wires bound to ports that no longer exist, then rebuild the
        node's output ports to match the latest count.
        """
        for node_id, node in list(self.pipeline.nodes.items()):
            spec = self.pipeline.operation_spec(node.operation_id)
            if spec.output_factory is None:
                continue
            for connection in self.pipeline.trim_invalid_output_connections(node_id):
                self.graph_view.remove_connection(
                    connection.source_id,
                    connection.target_id,
                    connection.target_port,
                    notify=False,
                )
            self._sync_node_output_ports(node_id)

    def _combine_channels_input_count(self, node) -> int:
        maximum = node.max_inputs if node.max_inputs is not None else 12
        try:
            count = int(node.params.get("input_count", 2))
        except Exception:
            count = 2
        return int(np.clip(count, 1, max(int(maximum), 1)))

    def _combine_channels_colors(self, node) -> list[str]:
        count = self._combine_channels_input_count(node)
        raw = str(node.params.get("channel_colors", "")).strip()
        colors = [part.strip().title() for part in raw.split(",") if part.strip()]
        defaults = list(CHANNEL_COLOR_CHOICES)
        while len(colors) < count:
            colors.append(defaults[len(colors) % len(defaults)])
        valid = {choice.lower(): choice for choice in CHANNEL_COLOR_CHOICES}
        normalized = [
            valid.get(color.lower(), defaults[index % len(defaults)])
            for index, color in enumerate(colors[:count])
        ]
        return normalized

    def _node_preview_channel_colors(self, node_id: str) -> list[str] | None:
        node = self.pipeline.nodes.get(node_id)
        if node is None or node.operation_id != "combine_channels":
            return None
        return self._combine_channels_colors(node)

    def _used_split_channel_ports(self, node_id: str) -> tuple[int, ...]:
        node = self.pipeline.nodes.get(node_id)
        if node is None or node.operation_id != "split_channels":
            return ()
        port_count = len(self.pipeline.output_ports(node_id))
        return tuple(
            sorted(
                {
                    int(connection.source_port)
                    for connection in self.pipeline.connections
                    if connection.source_id == node_id
                    and 0 <= int(connection.source_port) < port_count
                }
            )
        )

    def _single_used_split_channel_port(self, node_id: str) -> int | None:
        used_ports = self._used_split_channel_ports(node_id)
        return used_ports[0] if len(used_ports) == 1 else None

    def _split_channel_display_port(self, node_id: str) -> int:
        node = self.pipeline.nodes.get(node_id)
        if node is None or node.operation_id != "split_channels":
            return 0
        outputs = self.pipeline.node_outputs.get(node_id) or []
        port_count = max(len(outputs), len(self.pipeline.output_ports(node_id)), 1)
        single_used = self._single_used_split_channel_port(node_id)
        if single_used is not None:
            return int(np.clip(single_used, 0, port_count - 1))
        try:
            index = int(node.params.get("preview_channel", 0))
        except Exception:
            index = 0
        return int(np.clip(index, 0, port_count - 1))

    def _node_output_state(self, node_id: str, output_port: int = 0):
        states = self.pipeline.node_output_states.get(node_id) or []
        if 0 <= int(output_port) < len(states) and states[int(output_port)] is not None:
            return states[int(output_port)]
        if int(output_port) == 0:
            return self.pipeline.output_states.get(node_id)
        return None

    def _node_display_payload(self, node_id: str, data=None):
        primary_data = self.pipeline.outputs.get(node_id) if data is None else data
        primary_state = self.pipeline.output_states.get(node_id)
        node = self.pipeline.nodes.get(node_id)
        if node is None or node.operation_id != "split_channels":
            return primary_data, primary_state, 0
        outputs = self.pipeline.node_outputs.get(node_id) or []
        if not outputs:
            return primary_data, primary_state, 0
        index = self._split_channel_display_port(node_id)
        if 0 <= index < len(outputs) and outputs[index] is not None:
            return outputs[index], self._node_output_state(node_id, index), index

        # A connected port is an explicit presentation choice. If its cached
        # output is unavailable, leave the surface empty instead of silently
        # presenting a different channel with mismatched scientific meaning.
        if self._single_used_split_channel_port(node_id) is not None:
            return None, None, index

        available = [
            (output_index, output)
            for output_index, output in enumerate(outputs)
            if output is not None
        ]
        if len(available) == 1:
            output_index, output = available[0]
            return (
                output,
                self._node_output_state(node_id, output_index),
                output_index,
            )
        if index == 0:
            return primary_data, primary_state, 0
        return None, None, index

    def _thumbnail_payload_for_node(self, node_id: str, data):
        preview_data, preview_state, _output_port = self._node_display_payload(
            node_id,
            data,
        )
        return preview_data, preview_state

    def _refresh_split_channel_display_surfaces(
        self,
        node_ids: Iterable[str],
    ) -> None:
        affected = {
            node_id
            for node_id in node_ids
            if node_id in self.pipeline.nodes
            and self.pipeline.nodes[node_id].operation_id == "split_channels"
        }
        if not affected:
            return
        if self._selected_node_id in affected:
            self._refresh_selected_parameter_controls()
        self._discard_pending_thumbnail_contrast_limit_requests()
        self._update_thumbnails()
        inspection_layer = self._layer_by_name(self._inspect_layer_name)
        inspected_node_id = (
            getattr(inspection_layer, "metadata", {}).get("node_id")
            if inspection_layer is not None
            else None
        )
        if inspected_node_id in affected:
            self._refresh_inspection_layer_if_active()
        if self._selected_node_id in affected:
            self._clear_output_histogram_cache()
            self._inspect_selected_node()
            self._sync_view_dims_bar()
            self._update_metadata_panel()
            self._update_histogram()
        if self._active_pinned_node_id in affected:
            self._refresh_pinned_layer_if_active()

    def _thumbnail_contrast_limits_for_node(
        self,
        node_id: str,
        data,
        state: ImageState | None,
        contrast_mode: str,
        contrast_scope: str,
        data_kind: str,
    ):
        request = self._thumbnail_contrast_limit_request(
            node_id,
            data,
            state,
            contrast_mode,
            contrast_scope,
            data_kind,
        )
        if request is None:
            return None
        if request.key in self._thumbnail_contrast_limit_cache:
            return self._thumbnail_contrast_limit_cache[request.key]
        self._queue_thumbnail_contrast_limit_request(request)
        return None

    def _clear_thumbnail_contrast_limit_state(self) -> None:
        self._thumbnail_contrast_limit_cache.clear()
        self._discard_pending_thumbnail_contrast_limit_requests()

    def _clear_input_histogram_cache(self) -> None:
        if self._active_input_histogram_cancel_event is not None:
            self._active_input_histogram_cancel_event.set()
        self._input_histogram_serial += 1
        self._active_input_histogram_run_id = None
        self._active_input_histogram_key = None
        self._active_input_histogram_cancel_event = None
        self._input_histogram_cache.clear()
        self._input_histogram_distribution_cache.clear()
        self._current_input_histogram_key = None
        self._pending_input_histogram_request = None

    def _clear_output_histogram_cache(self) -> None:
        self._output_histogram_serial += 1
        self._active_output_histogram_run_id = None
        self._active_output_histogram_key = None
        self._output_histogram_cache.clear()
        self._current_output_histogram_key = None
        self._pending_output_histogram_request = None

    def _clear_colocalization_scatter_cache(self) -> None:
        """Invalidate cached and in-flight colocalization inspector results."""
        if self._active_colocalization_scatter_cancel_event is not None:
            self._active_colocalization_scatter_cancel_event.set()
        self._colocalization_scatter_serial += 1
        self._active_colocalization_scatter_run_id = None
        self._active_colocalization_scatter_key = None
        self._active_colocalization_scatter_cancel_event = None
        self._pending_colocalization_scatter_request = None
        self._current_colocalization_scatter_key = None
        self._colocalization_scatter_cache.clear()

    def _clear_generated_layer_contrast_state(self) -> None:
        """Invalidate cached and in-flight generated-layer display statistics."""
        self._generated_layer_contrast_generation += 1
        self._generated_layer_contrast_cache.clear()
        self._generated_layer_contrast_pending.clear()
        self._generated_layer_contrast_keys.clear()

    def _discard_pending_thumbnail_contrast_limit_requests(self) -> None:
        self._queued_thumbnail_contrast_limit_requests.clear()
        self._pending_thumbnail_contrast_limit_keys.clear()
        self._active_thumbnail_contrast_run_id = None
        if (
            self._thumbnail_contrast_busy_visible
            and self._active_pipeline_run_id is None
            and self._active_source_load_id is None
        ):
            self._set_pipeline_busy(False)
        self._thumbnail_contrast_busy_visible = False

    def _thumbnail_contrast_limit_request(
        self,
        node_id: str,
        data,
        state: ImageState | None,
        contrast_mode: str,
        contrast_scope: str,
        data_kind: str,
    ) -> ThumbnailContrastLimitRequest | None:
        if str(contrast_scope or "").strip().lower().startswith("slice"):
            return None
        if data is None or str(data_kind or "").lower() in {"labels", "table"}:
            return None
        try:
            arr = np.asarray(data)
        except Exception:
            return None
        channel_axis = self._thumbnail_channel_axis_for_contrast(arr, state)
        shape = tuple(int(size) for size in arr.shape)
        dtype = str(getattr(arr, "dtype", ""))
        mode_key = str(contrast_mode or "").strip().lower()
        if channel_axis is not None:
            key = (
                "channel",
                node_id,
                id(data),
                shape,
                dtype,
                mode_key,
                data_kind,
                int(channel_axis),
                int(arr.shape[channel_axis]),
            )
            return ThumbnailContrastLimitRequest(
                key,
                node_id,
                arr,
                int(channel_axis),
                contrast_mode,
                data_kind,
            )

        key = ("scalar", node_id, id(data), shape, dtype, mode_key, data_kind)
        return ThumbnailContrastLimitRequest(
            key,
            node_id,
            arr,
            None,
            contrast_mode,
            data_kind,
        )

    def _queue_thumbnail_contrast_limit_request(
        self,
        request: ThumbnailContrastLimitRequest,
    ) -> None:
        if request.key in self._thumbnail_contrast_limit_cache:
            return
        if request.key in self._pending_thumbnail_contrast_limit_keys:
            return
        if request.key in self._queued_thumbnail_contrast_limit_requests:
            return
        self._queued_thumbnail_contrast_limit_requests[request.key] = request
        if self._active_thumbnail_contrast_run_id is None:
            QTimer.singleShot(0, self._start_thumbnail_contrast_limit_run)

    def _start_thumbnail_contrast_limit_run(self) -> None:
        if self._active_thumbnail_contrast_run_id is not None:
            return
        if not self._queued_thumbnail_contrast_limit_requests:
            return
        self._thumbnail_contrast_serial += 1
        run_id = self._thumbnail_contrast_serial
        requests = tuple(self._queued_thumbnail_contrast_limit_requests.values())
        self._queued_thumbnail_contrast_limit_requests.clear()
        self._pending_thumbnail_contrast_limit_keys.update(
            request.key for request in requests
        )
        self._active_thumbnail_contrast_run_id = run_id
        self._show_thumbnail_contrast_busy(len(requests))
        worker = ThumbnailContrastLimitWorker(run_id, requests)
        worker.signals.progress.connect(self._on_thumbnail_contrast_limit_progress)
        worker.signals.finished.connect(self._on_thumbnail_contrast_limit_finished)
        self._pipeline_thread_pool.start(worker)

    def _show_thumbnail_contrast_busy(self, total: int) -> None:
        if self._active_pipeline_run_id is not None or self._active_source_load_id:
            self._thumbnail_contrast_busy_visible = False
            return
        self._thumbnail_contrast_busy_visible = True
        self._set_pipeline_busy(True, None, cancelable=False)
        self.pipeline_busy_label.setText("Calculating thumbnail contrast...")
        if total > 1:
            self.pipeline_busy_bar.setRange(0, total)
            self.pipeline_busy_bar.setValue(0)
            self.pipeline_busy_bar.setTextVisible(True)
            self.pipeline_busy_bar.setFormat("%v/%m")

    def _on_thumbnail_contrast_limit_progress(self, payload: object) -> None:
        try:
            run_id, current, total = payload
        except Exception:
            return
        if run_id != self._active_thumbnail_contrast_run_id:
            return
        if not self._thumbnail_contrast_busy_visible:
            return
        current = int(current)
        total = int(total)
        if total > 1:
            self.pipeline_busy_bar.setRange(0, total)
            self.pipeline_busy_bar.setValue(max(0, min(current, total)))
            self.pipeline_busy_bar.setTextVisible(True)
            self.pipeline_busy_bar.setFormat("%v/%m")
            self.pipeline_busy_label.setText(
                f"Calculating thumbnail contrast {current}/{total}..."
            )
        else:
            self.pipeline_busy_bar.setRange(0, 0)
            self.pipeline_busy_bar.setTextVisible(False)
            self.pipeline_busy_label.setText("Calculating thumbnail contrast...")

    def _on_thumbnail_contrast_limit_finished(
        self,
        result: ThumbnailContrastLimitResult,
    ) -> None:
        if result.run_id != self._active_thumbnail_contrast_run_id:
            return
        self._active_thumbnail_contrast_run_id = None
        self._pending_thumbnail_contrast_limit_keys.difference_update(result.keys)
        if not result.error:
            self._thumbnail_contrast_limit_cache.update(result.limits)
        if self._thumbnail_contrast_busy_visible:
            self._thumbnail_contrast_busy_visible = False
            if (
                self._active_pipeline_run_id is None
                and self._active_source_load_id is None
            ):
                self._set_pipeline_busy(False)
        if result.error:
            self.status_label.setText(
                f"Thumbnail contrast calculation failed: {result.error}"
            )
        else:
            self.status_label.setText("Thumbnail contrast ready.")
            self._update_thumbnails()
        if self._queued_thumbnail_contrast_limit_requests:
            QTimer.singleShot(0, self._start_thumbnail_contrast_limit_run)

    def _thumbnail_channel_axis_for_contrast(
        self,
        arr: np.ndarray,
        state: ImageState | None,
    ) -> int | None:
        if state is not None and len(state.axes) == arr.ndim:
            for index, axis in enumerate(state.axes):
                if axis.type == "channel" or axis.name.lower() in {"c", "rgb", "rgba"}:
                    return index
        if arr.ndim >= 3 and arr.shape[-1] in (3, 4):
            return arr.ndim - 1
        return None

    def _thumbnail_preview_consumes_contrast(
        self,
        node_id: str,
        data,
        state: ImageState | None,
    ) -> bool:
        if state is None:
            return False
        try:
            arr = np.asarray(data)
        except Exception:
            return False
        if len(state.axes) != arr.ndim:
            return False
        channel_axis = self._thumbnail_channel_axis_for_contrast(arr, state)
        if channel_axis is None:
            return False
        axis_name = state.axes[channel_axis].name.lower()
        if axis_name in {"rgb", "rgba"} and not self._node_preview_channel_colors(
            node_id
        ):
            return False
        return True

    def _contrast_parameter_bounds(self, node_id: str, spec) -> ParameterBounds:
        node = self.pipeline.nodes.get(node_id)
        if node is None:
            return ParameterBounds(
                spec.minimum,
                spec.maximum,
                spec.step,
                spec.decimals,
                expandable=True,
            )

        current = _safe_float(node.params.get(spec.name, spec.default), spec.default)
        minimum = float(spec.minimum)
        maximum = float(spec.maximum)
        if np.isfinite(current):
            if spec.name == "alpha":
                maximum = max(maximum, current * 1.25, float(spec.step))
            else:
                extent = max(abs(minimum), abs(maximum), abs(current) * 1.25, 1.0)
                minimum = -extent
                maximum = extent
        if minimum == maximum:
            minimum, maximum = _expanded_bounds(minimum)
        return _slider_safe_bounds(
            minimum,
            maximum,
            spec.step,
            spec.decimals,
            expandable=True,
        )

    def _resync_autodefault_nodes(self) -> set[str]:
        """Re-sync dtype-dependent output defaults and report changed nodes."""
        changed: set[str] = set()
        for node_id in tuple(self.pipeline.nodes):
            if self._sync_rescale_output_range_defaults(node_id):
                changed.add(node_id)
        return changed

    def _set_node_param_if_changed(
        self,
        node_id: str,
        name: str,
        value,
    ) -> bool:
        node = self.pipeline.nodes.get(node_id)
        if node is None:
            return False
        current = node.params.get(name)
        if isinstance(value, float) or isinstance(current, float):
            if np.isclose(
                _safe_float(current, np.nan),
                _safe_float(value, np.nan),
                rtol=0.0,
                atol=1e-9,
            ):
                return False
        elif current == value:
            return False
        node.params[name] = value
        return True

    def _sync_gaussian_blur_3d_xy_lock(
        self,
        node_id: str,
        *,
        source_name: str,
    ) -> bool:
        node = self.pipeline.nodes.get(node_id)
        if node is None or node.operation_id != "gaussian_blur_3d":
            return False
        if not bool(node.params.get("lock_xy", True)):
            return False
        if source_name not in {"sigma_y", "sigma_x"}:
            source_name = "sigma_y"
        target_name = "sigma_x" if source_name == "sigma_y" else "sigma_y"
        value = node.params.get(source_name)
        if value is None:
            return False
        changed = self._set_node_param_if_changed(node_id, target_name, value)
        if changed:
            widget = self._parameter_widgets.get(target_name)
            spec = self._parameter_spec_by_name(node_id, target_name)
            if widget is not None and spec is not None:
                spec = self._effective_parameter_spec(node_id, spec)
                widget.set_bounds(
                    self._parameter_bounds_for(node_id, spec),
                    value,
                    emit=False,
                )
        return changed

    def _sync_rescale_axes_xy_lock(
        self,
        node_id: str,
        *,
        source_name: str,
    ) -> bool:
        return self._sync_rescale_axes_representations(
            node_id,
            source_name=source_name,
        )

    def _sync_rescale_axes_representations(
        self,
        node_id: str,
        *,
        source_name: str = "",
    ) -> bool:
        node = self.pipeline.nodes.get(node_id)
        if node is None or node.operation_id != "rescale_axes":
            return False
        changed = False
        if "resize_mode" not in node.params:
            node.params["resize_mode"] = "Scale factor"
            changed = True
        mode = str(node.params.get("resize_mode", "Scale factor"))
        options = self._rescale_axis_options_by_role(node_id)

        for role, option in options.items():
            size_name = f"{role}_size"
            scale_name = f"{role}_scale"
            if size_name not in node.params:
                scale = _positive_scale_float(node.params.get(scale_name), 1.0)
                node.params[size_name] = max(int(round(option.size * scale)), 1)
                changed = True

        lock_xy = bool(node.params.get("lock_xy", True))
        if mode == "Output size":
            if lock_xy and "x" in options and "y" in options:
                source_role = "y" if source_name == "y_size" else "x"
                source_option = options[source_role]
                source_size = max(
                    int(node.params.get(f"{source_role}_size", source_option.size)),
                    1,
                )
                common_scale = source_size / max(source_option.size, 1)
                for role in ("x", "y"):
                    target = max(int(round(options[role].size * common_scale)), 1)
                    changed = (
                        self._set_node_param_if_changed(
                            node_id,
                            f"{role}_size",
                            target,
                        )
                        or changed
                    )
            for role, option in options.items():
                target = max(
                    int(node.params.get(f"{role}_size", option.size)),
                    1,
                )
                scale = target / max(option.size, 1)
                changed = (
                    self._set_node_param_if_changed(
                        node_id,
                        f"{role}_scale",
                        scale,
                    )
                    or changed
                )
            return changed

        if lock_xy and "x" in options and "y" in options:
            source_role = "y" if source_name == "y_scale" else "x"
            common_scale = _positive_scale_float(
                node.params.get(f"{source_role}_scale"),
                1.0,
            )
            for role in ("x", "y"):
                changed = (
                    self._set_node_param_if_changed(
                        node_id,
                        f"{role}_scale",
                        common_scale,
                    )
                    or changed
                )
        for role, option in options.items():
            scale = _positive_scale_float(node.params.get(f"{role}_scale"), 1.0)
            target = max(int(round(option.size * scale)), 1)
            changed = (
                self._set_node_param_if_changed(
                    node_id,
                    f"{role}_size",
                    target,
                )
                or changed
            )
        return changed

    def _refresh_rescale_axes_controls(self, node_id: str) -> bool:
        node = self.pipeline.nodes.get(node_id)
        if node is None or node.operation_id != "rescale_axes":
            return False
        changed = self._sync_rescale_axes_representations(node_id)
        for spec in self._rescale_axes_visible_specs(node_id):
            widget = self._parameter_widgets.get(spec.name)
            if widget is None:
                continue
            previous = node.params.get(spec.name, spec.default)
            bounds = self._parameter_bounds_for(node_id, spec)
            if isinstance(widget, ChoiceControl):
                widget.set_choices(
                    spec.choices,
                    previous,
                    emit=False,
                    choice_labels=spec.choice_labels,
                )
            widget.set_bounds(bounds, previous, emit=False)
            label = self.parameter_form.labelForField(widget)
            if isinstance(label, QLabel):
                label.setText(spec.label)
        return changed

    def _parameter_spec_by_name(self, node_id: str, name: str):
        for spec in self.pipeline.node_parameter_specs(node_id):
            if spec.name == name:
                return spec
        return None

    def _rescale_input_value_bounds(self, node_id: str, spec) -> ParameterBounds:
        return self._intensity_input_value_bounds(node_id, spec)

    def _intensity_input_value_bounds(self, node_id: str, spec) -> ParameterBounds:
        data = self.pipeline.input_data_for_node(node_id)
        if data is None:
            return ParameterBounds(
                spec.minimum,
                spec.maximum,
                spec.step,
                spec.decimals,
                expandable=True,
            )

        dtype = np.asarray(data).dtype
        if dtype == np.dtype(bool):
            return ParameterBounds(0, 1, 1, 0)
        if dtype == np.dtype(np.uint8):
            return ParameterBounds(0.0, 255.0, 0.1, 2, expandable=True)
        if _should_auto_background_data(data):
            if np.issubdtype(dtype, np.integer):
                info = np.iinfo(dtype)
                return _slider_safe_bounds(
                    float(info.min),
                    float(info.max),
                    max((float(info.max) - float(info.min)) / 500.0, 1.0),
                    2,
                    expandable=True,
                )
            current = _safe_float(
                self.pipeline.nodes[node_id].params.get(spec.name),
                spec.default,
            )
            extent = max(abs(current) * 1.25, 1.0)
            return _slider_safe_bounds(
                min(float(spec.minimum), -extent),
                max(float(spec.maximum), extent),
                max(extent / 500.0, 1e-6),
                spec.decimals,
                expandable=True,
            )

        stats = _exact_finite_stats(data)
        if stats.count == 0:
            return ParameterBounds(
                spec.minimum,
                spec.maximum,
                spec.step,
                spec.decimals,
                expandable=True,
            )
        minimum = stats.minimum
        maximum = stats.maximum
        if minimum == maximum:
            minimum, maximum = _expanded_bounds(minimum)
        if np.issubdtype(dtype, np.integer):
            step = max((maximum - minimum) / 500.0, 1.0)
            return _slider_safe_bounds(
                minimum,
                maximum,
                step,
                2,
                expandable=True,
            )

        span = maximum - minimum
        step = max(span / 500.0, 1e-6)
        decimals = 4 if 0.0 <= minimum and maximum <= 1.0 else 3
        return _slider_safe_bounds(
            minimum,
            maximum,
            step,
            decimals,
            expandable=True,
        )

    def _sync_rescale_output_range_defaults(self, node_id: str) -> bool:
        node = self.pipeline.nodes.get(node_id)
        if node is None or node.operation_id != "rescale_intensity":
            return False
        data = self.pipeline.input_data_for_node(node_id)
        if data is None:
            return False

        default_min, default_max, _step, _decimals = _rescale_dtype_output_range(
            np.asarray(data).dtype
        )
        current = (
            _safe_float(node.params.get("out_min"), 0.0),
            _safe_float(node.params.get("out_max"), 1.0),
        )
        previous_auto = self._rescale_auto_output_ranges.get(node_id, (0.0, 1.0))
        self._rescale_auto_output_ranges[node_id] = (default_min, default_max)
        if not _ranges_close(current, previous_auto):
            return False
        if _ranges_close(current, (default_min, default_max)):
            return False
        node.params["out_min"] = default_min
        node.params["out_max"] = default_max
        return True

    def _rescale_output_bounds(self, node_id: str, spec) -> ParameterBounds:
        data = self.pipeline.input_data_for_node(node_id)
        if data is None:
            return ParameterBounds(
                spec.minimum,
                spec.maximum,
                spec.step,
                spec.decimals,
                expandable=True,
            )
        minimum, maximum, step, decimals = _rescale_dtype_output_range(
            np.asarray(data).dtype
        )
        return ParameterBounds(
            minimum,
            maximum,
            step,
            decimals,
            expandable=True,
        )

    def _threshold_bounds(self, node_id: str, spec) -> ParameterBounds:
        data = self.pipeline.input_data_for_node(node_id)
        if data is None:
            return ParameterBounds(spec.minimum, spec.maximum, spec.step, spec.decimals)
        arr = np.asarray(data)
        if arr.dtype == bool:
            return ParameterBounds(0, 1, 1, 0)
        if _should_auto_background_data(arr):
            if np.issubdtype(arr.dtype, np.integer):
                info = np.iinfo(arr.dtype)
                return ParameterBounds(
                    float(info.min),
                    float(info.max),
                    1,
                    0,
                    expandable=True,
                )
            return ParameterBounds(
                spec.minimum,
                spec.maximum,
                spec.step,
                spec.decimals,
                expandable=True,
            )
        if np.issubdtype(arr.dtype, np.integer):
            if arr.dtype == np.uint8:
                return ParameterBounds(0, 255, 1, 0)
            finite = _finite_values(arr)
            if finite.size:
                return ParameterBounds(
                    int(finite.min()),
                    int(finite.max()),
                    1,
                    0,
                )
            return ParameterBounds(0, 255, 1, 0)

        finite = _finite_values(arr)
        if finite.size == 0:
            return ParameterBounds(spec.minimum, spec.maximum, spec.step, spec.decimals)
        minimum = float(finite.min())
        maximum = float(finite.max())
        if 0.0 <= minimum and maximum <= 1.0:
            return ParameterBounds(0.0, 1.0, 0.01, 3)
        if minimum == maximum:
            minimum, maximum = _expanded_bounds(minimum)
        step = max((maximum - minimum) / 200.0, 1e-6)
        return ParameterBounds(minimum, maximum, step, 3)

    def _axis_bounds(self, node_id: str, spec) -> ParameterBounds:
        data = self.pipeline.input_data_for_node(node_id)
        if data is None:
            return ParameterBounds(spec.minimum, spec.maximum, spec.step, spec.decimals)
        maximum = max(np.asarray(data).ndim - 1, 0)
        return ParameterBounds(0, maximum, 1, 0)

    def _slice_index_bounds(self, node_id: str, spec) -> ParameterBounds:
        data = self.pipeline.input_data_for_node(node_id)
        node = self.pipeline.nodes.get(node_id)
        if data is None or node is None:
            return ParameterBounds(spec.minimum, spec.maximum, spec.step, spec.decimals)
        arr = np.asarray(data)
        if arr.ndim == 0:
            return ParameterBounds(0, 0, 1, 0)
        axis = int(np.clip(int(node.params.get("axis", 0)), 0, arr.ndim - 1))
        return ParameterBounds(0, max(arr.shape[axis] - 1, 0), 1, 0)

    def _crop_bounds(self, node_id: str, spec) -> ParameterBounds:
        data = self.pipeline.input_data_for_node(node_id)
        if data is None:
            return ParameterBounds(spec.minimum, spec.maximum, spec.step, spec.decimals)
        height, width = _xy_shape(np.asarray(data))
        maximum = height - 1 if spec.name in {"top", "bottom"} else width - 1
        return ParameterBounds(0, max(maximum, 0), 1, 0)

    def _channel_bounds(self, node_id: str, spec) -> ParameterBounds:
        data = self.pipeline.input_data_for_node(node_id)
        if data is None:
            return ParameterBounds(spec.minimum, spec.maximum, spec.step, spec.decimals)
        arr = np.asarray(data)
        axis = self._selected_channel_axis(node_id, arr)
        maximum = arr.shape[axis] - 1 if axis is not None else 0
        return ParameterBounds(spec.minimum, max(maximum, spec.minimum), 1, 0)

    def _channel_axis_bounds(self, node_id: str, spec) -> ParameterBounds:
        data = self.pipeline.input_data_for_node(node_id)
        if data is None:
            return ParameterBounds(spec.minimum, spec.maximum, spec.step, spec.decimals)
        maximum = max(np.asarray(data).ndim - 1, 0)
        return ParameterBounds(spec.minimum, max(maximum, spec.minimum), 1, 0)

    def _selected_channel_axis(self, node_id: str, arr: np.ndarray) -> int | None:
        node = self.pipeline.nodes.get(node_id)
        if node is not None and "channel_axis" in node.params:
            try:
                stored = int(node.params.get("channel_axis", -1))
            except Exception:
                stored = -1
            if stored >= 0:
                return int(np.clip(stored, 0, arr.ndim - 1))
        preferred = self._preferred_channel_axis(node_id)
        if preferred is not None:
            return preferred
        if arr.ndim >= 3 and arr.shape[-1] in (3, 4):
            return arr.ndim - 1
        if arr.ndim >= 4:
            return 1 if arr.ndim >= 5 else 0
        if arr.ndim == 3 and arr.shape[0] <= 16:
            return 0
        return None

    def _preferred_channel_axis(self, node_id: str) -> int | None:
        state = self.pipeline.input_state_for_node(node_id)
        if state is None:
            return None
        for index, axis in enumerate(state.axes):
            if axis.type == "channel" or axis.name.lower() == "c":
                return index
        return None

    def _block_size_bounds(self, node_id: str, spec) -> ParameterBounds:
        data = self.pipeline.input_data_for_node(node_id)
        if data is None:
            return ParameterBounds(spec.minimum, spec.maximum, spec.step, spec.decimals)
        height, width = _xy_shape(np.asarray(data))
        maximum = max(min(height, width), 3)
        if maximum % 2 == 0:
            maximum -= 1
        return ParameterBounds(3, maximum, 2, 0)

    def _label_volume_bounds(self, node_id: str, spec) -> ParameterBounds:
        data = self.pipeline.input_data_for_node(node_id)
        maximum = max(int(spec.default) * 10, 100)
        if data is not None:
            arr = np.asarray(data)
            node = self.pipeline.nodes[node_id]
            spatial_ndim = int(
                np.clip(
                    node.params.get(
                        "resolved_spatial_ndim",
                        3 if arr.ndim >= 3 else 2,
                    ),
                    1,
                    max(arr.ndim, 1),
                )
            )
            volumes = self._cached_label_volumes(arr, spatial_ndim)
            maximum = max(
                int(volumes.max()) if volumes.size else 0,
                int(spec.default),
                1,
            )
        return ParameterBounds(
            0,
            maximum,
            1,
            0,
            logarithmic=True,
            entry_minimum=spec.minimum,
            entry_maximum=spec.maximum,
        )

    def _clear_border_buffer_bounds(self, node_id: str, spec) -> ParameterBounds:
        data = self.pipeline.input_data_for_node(node_id)
        if data is None:
            return ParameterBounds(
                spec.minimum,
                spec.maximum,
                spec.step,
                spec.decimals,
            )
        arr = np.asarray(data)
        if arr.ndim == 0 or arr.size == 0:
            return ParameterBounds(0, 0, 1, 0)
        spatial_ndim = self._selected_spatial_ndim(node_id)
        node = self.pipeline.nodes[node_id]
        mode = str(node.params.get("boundary_mode", "")).lower()
        boundary_ndim = 2 if mode.startswith("lateral") else spatial_ndim
        maximum = max(min(arr.shape[-boundary_ndim:]) - 1, 0)
        return ParameterBounds(0, maximum, 1, 0)

    def _fill_hole_size_bounds(self, node_id: str, spec) -> ParameterBounds:
        data = self.pipeline.input_data_for_node(node_id)
        maximum = max(int(spec.default), 1)
        if data is not None:
            arr = np.asarray(data)
            spatial_ndim = self._selected_spatial_ndim(node_id)
            maximum = max(int(np.prod(arr.shape[-spatial_ndim:])), 1)
        return ParameterBounds(
            0,
            maximum,
            1,
            0,
            logarithmic=True,
            entry_minimum=spec.minimum,
            entry_maximum=spec.maximum,
        )

    def _remove_small_object_bounds(
        self,
        node_id: str,
        spec,
    ) -> ParameterBounds:
        data = self.pipeline.input_data_for_node(node_id)
        maximum = max(int(spec.default) * 10, 100)
        if data is not None:
            arr = np.asarray(data)
            spatial_ndim = self._selected_spatial_ndim(node_id)
            connectivity = str(
                self.pipeline.nodes[node_id].params.get(
                    "connectivity",
                    "Face connected",
                )
            )
            maximum = max(
                self._largest_object_size(arr, spatial_ndim, connectivity),
                int(spec.default),
                1,
            )
        return ParameterBounds(
            0,
            maximum,
            1,
            0,
            logarithmic=True,
            entry_minimum=spec.minimum,
            entry_maximum=spec.maximum,
        )

    @staticmethod
    def _largest_object_size(
        objects: np.ndarray,
        spatial_ndim: int,
        connectivity: str,
    ) -> int:
        arr = np.asarray(objects)
        if arr.size == 0:
            return 0
        if arr.dtype != bool:
            return VippWidget._largest_label_volume(arr, spatial_ndim)

        spatial_ndim = int(np.clip(spatial_ndim, 1, max(arr.ndim, 1)))
        rank = 1 if str(connectivity).lower().startswith("face") else spatial_ndim
        structure = ndi.generate_binary_structure(spatial_ndim, rank)
        leading_shape = arr.shape[: arr.ndim - spatial_ndim]
        blocks = (
            (arr[index] for index in np.ndindex(leading_shape))
            if leading_shape
            else (arr,)
        )
        largest = 0
        for block in blocks:
            labels, count = ndi.label(block, structure=structure)
            if count:
                largest = max(
                    largest,
                    int(np.bincount(labels.ravel())[1:].max()),
                )
        return largest

    @staticmethod
    def _largest_label_volume(labels: np.ndarray, spatial_ndim: int) -> int:
        volumes = VippWidget._label_volumes(labels, spatial_ndim)
        return int(volumes.max()) if volumes.size else 0

    def _cached_label_volumes(
        self,
        labels: np.ndarray,
        spatial_ndim: int,
    ) -> np.ndarray:
        key = (
            id(labels),
            tuple(labels.shape),
            str(labels.dtype),
            int(spatial_ndim),
        )
        cached = self._label_volume_cache.get(key)
        if cached is not None:
            identity_ref, volumes = cached
            if identity_ref() is labels:
                return volumes
            self._label_volume_cache.pop(key, None)
        volumes = self._label_volumes(labels, spatial_ndim)
        try:
            identity_ref = weakref.ref(labels)
        except TypeError:
            return volumes
        self._label_volume_cache[key] = (identity_ref, volumes)
        while len(self._label_volume_cache) > 16:
            self._label_volume_cache.pop(next(iter(self._label_volume_cache)))
        return volumes

    @staticmethod
    def _label_volumes(labels: np.ndarray, spatial_ndim: int) -> np.ndarray:
        arr = np.asarray(labels)
        if arr.size == 0:
            return np.array([], dtype=np.int64)
        spatial_ndim = int(np.clip(spatial_ndim, 1, max(arr.ndim, 1)))
        leading_shape = arr.shape[: arr.ndim - spatial_ndim]
        blocks = (
            (arr[index] for index in np.ndindex(leading_shape))
            if leading_shape
            else (arr,)
        )
        volumes: list[np.ndarray] = []
        for block in blocks:
            foreground = np.asarray(block)
            foreground = foreground[foreground > 0]
            if foreground.size == 0:
                continue
            _labels, counts = np.unique(foreground, return_counts=True)
            volumes.append(counts.astype(np.int64, copy=False))
        if not volumes:
            return np.array([], dtype=np.int64)
        return np.concatenate(volumes)

    def _clear_parameter_form(self) -> None:
        self._parameter_widgets.clear()
        while self.parameter_form.count():
            item = self.parameter_form.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _on_param_changed(self, name: str, value) -> None:
        node = self.pipeline.nodes.get(self._selected_node_id)
        if node is None or node.params.get(name) == value:
            return
        self._record_parameter_undo(self._selected_node_id, name)
        self.pipeline.set_param(self._selected_node_id, name, value)
        if node.operation_id == "split_channels" and name == "preview_channel":
            self._refresh_split_channel_display_surfaces({node.id})
            self.status_label.setText(
                f"Showing channel {int(value) + 1} for '{node.title}'."
            )
            return
        if name == "tag":
            self._refresh_graph_search_matches(reset_index=True)
        self._mark_pipeline_dirty(self._selected_node_id)
        if node.operation_id == "gaussian_blur_3d":
            if name == "lock_xy" and bool(value):
                self._sync_gaussian_blur_3d_xy_lock(
                    self._selected_node_id,
                    source_name="sigma_y",
                )
            elif name in {"sigma_y", "sigma_x"}:
                self._sync_gaussian_blur_3d_xy_lock(
                    self._selected_node_id,
                    source_name=name,
                )
        if node.operation_id == "rescale_axes":
            if name == "resize_mode":
                self._sync_rescale_axes_representations(
                    self._selected_node_id,
                    source_name=("x_size" if value == "Output size" else "x_scale"),
                )
                self._render_parameters(self._selected_node_id)
            elif name == "lock_xy" and bool(value):
                mode = str(node.params.get("resize_mode", "Scale factor"))
                self._sync_rescale_axes_representations(
                    self._selected_node_id,
                    source_name="x_size" if mode == "Output size" else "x_scale",
                )
            elif name in {
                "x_scale",
                "y_scale",
                "z_scale",
                "x_size",
                "y_size",
                "z_size",
            }:
                self._sync_rescale_axes_representations(
                    self._selected_node_id,
                    source_name=name,
                )
            if name != "resize_mode" and name in {
                "x_scale",
                "y_scale",
                "z_scale",
                "x_size",
                "y_size",
                "z_size",
                "lock_xy",
            }:
                self._refresh_selected_parameter_controls()
        if name == "input_count":
            for connection in self.pipeline.trim_invalid_connections(
                self._selected_node_id
            ):
                self.graph_view.remove_connection(
                    connection.source_id,
                    connection.target_id,
                    target_port=connection.target_port,
                    notify=False,
                )
            self._sync_node_input_ports(self._selected_node_id)
        if name in {"axis", "boundary_mode", "channel_axis", "spatial_mode"}:
            self._refresh_selected_parameter_controls()
        if node.operation_id == "split_axis" and name == "axis":
            for connection in self.pipeline.trim_invalid_output_connections(
                self._selected_node_id
            ):
                self.graph_view.remove_connection(
                    connection.source_id,
                    connection.target_id,
                    target_port=connection.target_port,
                    notify=False,
                )
            self._sync_node_output_ports(self._selected_node_id)
        if node.operation_id in COLOCALIZATION_THRESHOLD_OPERATIONS:
            if name == "threshold_mode" and str(value).lower().startswith("costes"):
                self._sync_colocalization_costes_thresholds(
                    self._selected_node_id,
                    update_controls=True,
                )
            if name in {
                "threshold_mode",
                "channel_1_threshold",
                "channel_2_threshold",
            }:
                self._update_colocalization_scatter()
        if name == "spatial_mode":
            self._update_fill_holes_scope_note()
        if node.operation_id == "born_wolf_psf" and name in {
            "auto_parameters",
            "channel",
        }:
            self._sync_node_output_ports(self._selected_node_id)
        if name in {"min_volume", "max_volume", "spatial_mode"}:
            self._update_label_volume_histogram()
        if node.operation_id == "rescale_intensity" and name == (
            RESCALE_CUTOFF_MODE_PARAMETER
        ):
            self._render_parameters(self._selected_node_id)
            self._update_rescale_input_histogram(
                self._selected_node_id,
                self._current_step(),
            )
        elif node.operation_id == "clip_intensity" and name == "cutoff_mode":
            self._render_parameters(self._selected_node_id)
            self._update_rescale_input_histogram(
                self._selected_node_id,
                self._current_step(),
            )
        elif (
            node.operation_id == "rescale_intensity"
            and name in RESCALE_CUTOFF_PARAMETERS
        ):
            self._update_rescale_input_histogram(
                self._selected_node_id,
                self._current_step(),
            )
        if (
            (node.operation_id == "clip_intensity" and name in CLIP_CUTOFF_PARAMETERS)
            or (node.operation_id == "binary_threshold" and name == "threshold")
            or (
                node.operation_id == "hysteresis_threshold"
                and name in {"low_threshold", "high_threshold"}
            )
            or (
                node.operation_id in GLOBAL_THRESHOLD_OPERATIONS
                and name
                in {"threshold_scope", "histogram_bins", "max_iterations"}
            )
        ):
            self._update_rescale_input_histogram(
                self._selected_node_id,
                self._current_step(),
            )
        self._debounce_timer.start()

    def _update_fill_holes_scope_note(self) -> None:
        note = self._parameter_widgets.get("fill_holes_scope_note")
        node = self.pipeline.nodes.get(self._selected_node_id)
        if not isinstance(note, QLabel) or node is None:
            return
        mode = str(node.params.get("spatial_mode", "Auto from axes")).lower()
        if self._input_spatial_count(node.id) < 3:
            text = "Holes are evaluated in the connected YX image."
            color = "#94a3b8"
        elif mode.startswith("2d"):
            text = (
                "Advanced mode: each XY slice is filled independently. "
                "A region that is open to background along Z can still be filled."
            )
            color = "#f59e0b"
        else:
            text = (
                "Recommended for z-stacks: holes are enclosed cavities in the "
                "complete ZYX volume."
            )
            color = "#94a3b8"
        note.setText(text)
        note.setStyleSheet(f"color: {color};")

    def _apply_auto_contrast(self) -> None:
        node_id = self._selected_node_id
        node = self.pipeline.nodes.get(node_id)
        if node is None or node.operation_id != "linear_scale_offset":
            return

        if self._active_auto_contrast_run_id is not None:
            self.status_label.setText(
                "Exact auto-contrast calculation is already running."
            )
            return

        data = self.pipeline.input_data_for_node(node_id)
        saturation = float(self.auto_saturation_control.value())
        key = self._auto_contrast_request_key(node_id, data, saturation)
        if _should_background_auto_contrast(data):
            self._auto_contrast_serial += 1
            request = AutoContrastRequest(
                self._auto_contrast_serial,
                key,
                node_id,
                data,
                saturation,
            )
            self._active_auto_contrast_run_id = request.run_id
            self._active_auto_contrast_key = request.key
            self.auto_contrast_button.setEnabled(False)
            self._show_auto_contrast_busy(node_id)
            self.status_label.setText(
                "Calculating exact full-input auto-contrast percentiles..."
            )
            worker = AutoContrastWorker(request)
            worker.signals.finished.connect(self._on_auto_contrast_finished)
            self._pipeline_thread_pool.start(worker, -1)
            return

        result = _auto_contrast_scale_offset(data, saturation)
        self._commit_auto_contrast_result(
            node_id,
            saturation,
            result,
        )

    def _auto_contrast_request_key(
        self,
        node_id: str,
        data,
        saturation: float,
    ) -> tuple:
        node = self.pipeline.nodes.get(node_id)
        return (
            node_id,
            id(data),
            tuple(getattr(data, "shape", ())),
            str(getattr(data, "dtype", "")),
            float(saturation),
            repr(node.params.get("alpha")) if node is not None else "",
            repr(node.params.get("beta")) if node is not None else "",
        )

    def _show_auto_contrast_busy(self, node_id: str) -> None:
        if (
            self._active_pipeline_run_id is not None
            or self._active_source_load_id is not None
            or self._thumbnail_contrast_busy_visible
        ):
            self._auto_contrast_busy_visible = False
            return
        self._auto_contrast_busy_visible = True
        self._set_pipeline_busy(True, node_id, cancelable=False)
        self.pipeline_busy_label.setText("Calculating exact auto contrast...")

    def _clear_auto_contrast_busy(self) -> None:
        if not self._auto_contrast_busy_visible:
            return
        self._auto_contrast_busy_visible = False
        if (
            self._active_pipeline_run_id is None
            and self._active_source_load_id is None
            and not self._thumbnail_contrast_busy_visible
        ):
            self._set_pipeline_busy(False)

    def _on_auto_contrast_finished(self, result: AutoContrastResult) -> None:
        if result.run_id != self._active_auto_contrast_run_id:
            return
        active_key = self._active_auto_contrast_key
        self._active_auto_contrast_run_id = None
        self._active_auto_contrast_key = None
        self._clear_auto_contrast_busy()
        self._sync_auto_contrast_ui()

        if result.error:
            self.status_label.setText(
                f"Exact auto-contrast calculation failed: {result.error}"
            )
            return

        current_data = self.pipeline.input_data_for_node(result.node_id)
        current_key = self._auto_contrast_request_key(
            result.node_id,
            current_data,
            self.auto_saturation_control.value(),
        )
        if (
            result.key != active_key
            or result.key != current_key
            or result.node_id != self._selected_node_id
        ):
            if result.node_id == self._selected_node_id:
                self.status_label.setText(
                    "Auto-contrast input or settings changed; the stale result "
                    "was ignored."
                )
            return

        self._commit_auto_contrast_result(
            result.node_id,
            result.saturation_percent,
            result.scale_offset,
        )

    def _commit_auto_contrast_result(
        self,
        node_id: str,
        saturation: float,
        result: tuple[float, float, float, float] | None,
    ) -> None:
        node = self.pipeline.nodes.get(node_id)
        if node is None or node.operation_id != "linear_scale_offset":
            return
        if result is None:
            self.status_label.setText(
                "Auto contrast needs connected input with intensity variation."
            )
            return

        alpha, beta, lower, upper = result
        if node.params.get("alpha") == alpha and node.params.get("beta") == beta:
            self.status_label.setText("Auto contrast is already up to date.")
            return
        self._finish_parameter_history_group()
        before = self._current_history_snapshot()
        self.pipeline.set_param(node_id, "alpha", alpha)
        self.pipeline.set_param(node_id, "beta", beta)
        self._mark_pipeline_dirty(node_id)
        self._debounce_timer.stop()
        self._render_parameters(node_id)
        self.run_pipeline()
        self._push_undo_if_changed(before)
        self.status_label.setText(
            f"Auto contrast set '{node.title}' to {saturation:.2f}% saturation "
            f"({lower:.3g} to {upper:.3g})."
        )

    def run_pipeline(
        self,
        *,
        force_sync: bool = False,
        manual_node_ids: set[str] | None = None,
    ) -> None:
        manual_node_ids = {
            node_id
            for node_id in (manual_node_ids or set())
            if self.pipeline.is_manual_node(node_id)
        }
        if self._pending_manual_node_ids:
            manual_node_ids.update(
                node_id
                for node_id in self._pending_manual_node_ids
                if self.pipeline.is_manual_node(node_id)
            )
            self._pending_manual_node_ids.clear()
        source_load_specs = self._uncached_async_file_source_specs()
        if source_load_specs:
            self._pending_manual_node_ids.update(manual_node_ids)
            self._start_source_file_load(source_load_specs)
            return
        try:
            source_payloads, source_layers = self._source_payloads_for_pipeline()
        except OptionalMicroscopeReaderError as exc:
            self._abandon_background_pipeline_run()
            self._set_pipeline_busy(False)
            self._show_optional_reader_error(exc)
            return
        except Exception as exc:
            self._abandon_background_pipeline_run()
            self._set_pipeline_busy(False)
            self.status_label.setText(f"Image source error: {exc}")
            return

        if not source_payloads:
            self._abandon_background_pipeline_run()
            self._set_pipeline_busy(False)
            self._invalidate_pipeline_cache()
            self._restore_hidden_input_layers()
            self.pipeline.run(None)
            self._update_thumbnails()
            self._sync_view_dims_bar()
            self._update_metadata_panel()
            self._update_histogram()
            self._sync_execution_ui()
            self._refresh_cache_status()
            self.status_label.setText("No Image Source node has a selected source.")
            return

        primary_layer = source_layers[0] if source_layers else None
        self._restore_hidden_input_layers(except_layer=primary_layer)
        input_data = None
        input_metadata = None
        input_name = ""
        source_label = self._pipeline_source_label(source_payloads, input_name)
        source_signature = self._pipeline_source_signature(
            input_data,
            input_metadata,
            input_name,
            source_payloads,
        )
        source_unchanged = source_signature == self._last_pipeline_source_signature
        dirty_node_ids = self._dirty_nodes_for_run(source_signature)
        manual_node_ids = self._manual_node_ids_for_run(
            dirty_node_ids,
            manual_node_ids,
            source_unchanged=source_unchanged,
        )
        if manual_node_ids and source_unchanged:
            if dirty_node_ids is None:
                dirty_node_ids = set(manual_node_ids)
            else:
                dirty_node_ids.update(manual_node_ids)
        if force_sync and self._active_pipeline_run_id is not None:
            inflight_dirty = self._inflight_dirty_node_ids
            if inflight_dirty is None:
                dirty_node_ids = None
                self._pending_dirty_node_ids.clear()
            elif dirty_node_ids is not None:
                dirty_node_ids.update(inflight_dirty)
            manual_node_ids.update(
                self._pipeline_run_manual_node_ids.get(
                    self._active_pipeline_run_id,
                    frozenset(),
                )
            )
            self._abandon_background_pipeline_run()
        if self._active_pipeline_run_id is not None or (
            not force_sync
            and self._should_run_pipeline_in_background(
                dirty_node_ids,
                manual_node_ids,
                source_payloads,
            )
        ):
            self._start_background_pipeline_run(
                input_data,
                input_metadata,
                input_name,
                source_payloads,
                primary_layer,
                source_label,
                source_signature,
                dirty_node_ids,
                manual_node_ids,
            )
            return
        self._run_pipeline_synchronously(
            input_data,
            input_metadata,
            input_name,
            source_payloads,
            primary_layer,
            source_label,
            source_signature,
            dirty_node_ids,
            manual_node_ids,
        )

    def _abandon_background_pipeline_run(self) -> None:
        """Cancel and detach an in-flight clone whose result must be ignored."""
        run_id = self._active_pipeline_run_id
        if run_id is None:
            return
        cancel_event = self._pipeline_cancel_events.pop(run_id, None)
        if cancel_event is not None:
            cancel_event.set()
        self._pipeline_run_context.pop(run_id, None)
        self._pipeline_run_manual_node_ids.pop(run_id, None)
        self._active_pipeline_run_id = None
        self._active_pipeline_node_id = None
        self._pipeline_run_pending = False
        self._inflight_dirty_node_ids = None
        self._set_pipeline_busy(False)

    def _manual_node_ids_for_run(
        self,
        dirty_node_ids: set[str] | None,
        explicit_manual_node_ids: set[str],
        *,
        source_unchanged: bool,
    ) -> set[str]:
        manual_node_ids = set(explicit_manual_node_ids)
        auto_node_ids = self.pipeline.auto_recalculate_node_ids()
        if not auto_node_ids:
            return manual_node_ids
        if not source_unchanged:
            manual_node_ids.update(auto_node_ids)
            return manual_node_ids
        if dirty_node_ids:
            affected = self.pipeline.descendants_inclusive(dirty_node_ids)
            manual_node_ids.update(auto_node_ids & affected)
        return manual_node_ids

    def _run_pipeline_synchronously(
        self,
        input_data,
        input_metadata,
        input_name: str,
        source_payloads: dict[str, SourcePayload],
        primary_layer,
        source_label: str,
        source_signature: tuple,
        dirty_node_ids: set[str] | None,
        manual_node_ids: set[str] | None = None,
    ) -> None:
        self._set_pipeline_busy(False)
        self._begin_pipeline_dispatch(dirty_node_ids)
        try:
            with self._preserve_interactive_collection_workflow_params():
                self.pipeline.run(
                    input_data,
                    input_metadata=input_metadata,
                    input_name=input_name,
                    source_payloads=source_payloads,
                    dirty_node_ids=dirty_node_ids,
                    manual_mode=MANUAL_RUN_SKIP,
                    manual_node_ids=manual_node_ids,
                    retain_node_ids=self._cache_retention_node_ids(),
                    prune_unretained=self._cache_pruning_enabled(),
                )
        except Exception as exc:
            for node_id in manual_node_ids or ():
                self.pipeline.set_node_execution_error(node_id, str(exc))
            self._sync_execution_ui()
            self.status_label.setText(f"Pipeline error: {exc}")
            return
        autodefault_changed = self._resync_autodefault_nodes()
        refreshed_selected = self._refresh_selected_parameter_controls()
        if autodefault_changed or refreshed_selected:
            rerun_dirty = set(autodefault_changed)
            if refreshed_selected and self._selected_node_id in self.pipeline.nodes:
                rerun_dirty.add(self._selected_node_id)
            rerun_dirty &= set(self.pipeline.nodes)
            if rerun_dirty:
                with self._preserve_interactive_collection_workflow_params():
                    self.pipeline.run(
                        input_data,
                        input_metadata=input_metadata,
                        input_name=input_name,
                        source_payloads=source_payloads,
                        dirty_node_ids=rerun_dirty,
                        manual_mode=MANUAL_RUN_SKIP,
                        retain_node_ids=self._cache_retention_node_ids(),
                        prune_unretained=self._cache_pruning_enabled(),
                    )
        self._complete_pipeline_run(source_signature, dirty_node_ids)
        self._finish_pipeline_update(primary_layer, source_label)

    def _finish_pipeline_update(self, primary_layer, source_label: str) -> None:
        self._clear_output_histogram_cache()
        self._clear_colocalization_scatter_cache()
        self._hide_input_layer_for_inspection(primary_layer)
        self._apply_cache_retention()
        self._refresh_dynamic_output_ports()
        self._discard_pending_thumbnail_contrast_limit_requests()
        self._update_thumbnails()
        self._refresh_inspection_layer_if_active()
        self._inspect_selected_node()
        self._refresh_pinned_layer_if_active()
        self._sync_view_dims_bar()
        self._update_metadata_panel()
        self._update_histogram()
        self._sync_execution_ui()
        memory_guard_message = self._enforce_memory_guard()
        if memory_guard_message:
            self.status_label.setText(memory_guard_message)
        elif source_label:
            self.status_label.setText(
                f"Graph updated from '{source_label}'. "
                "Connect ports to build alternate paths."
            )
        else:
            self.status_label.setText("No image source selected.")
        self._refresh_cache_status()

    def _pipeline_source_label(
        self,
        source_payloads: dict[str, SourcePayload],
        input_name: str,
    ) -> str:
        source_names = [payload.name for payload in source_payloads.values()]
        return ", ".join(name for name in source_names if name) or input_name

    def _start_source_file_load(
        self,
        specs: tuple[SourceFileLoadSpec, ...],
    ) -> None:
        if not specs:
            return
        if self._active_source_load_id is not None:
            self._source_load_pending = True
            self._set_pipeline_busy(
                True,
                specs[0].node_id,
                queued=True,
                cancelable=False,
            )
            self.status_label.setText(
                "Loading image source in background; latest source edit queued."
            )
            return
        self._source_load_serial += 1
        run_id = self._source_load_serial
        self._active_source_load_id = run_id
        self._source_load_pending = False
        node_id = specs[0].node_id
        self._set_pipeline_busy(True, node_id, cancelable=False)
        path_name = Path(specs[0].path).name
        suffix = f" ({len(specs)} source(s))" if len(specs) > 1 else ""
        self.status_label.setText(f"Loading image source '{path_name}'{suffix}...")
        worker = SourceFileLoadWorker(run_id, specs)
        worker.signals.finished.connect(self._on_source_file_load_finished)
        self._pipeline_thread_pool.start(worker)

    def _on_source_file_load_finished(self, result: SourceFileLoadResult) -> None:
        if result.run_id != self._active_source_load_id:
            return
        self._active_source_load_id = None
        if result.error:
            if result.node_id:
                self.pipeline.set_node_execution_error(result.node_id, result.error)
            self._sync_execution_ui()
            self._set_pipeline_busy(False)
            self.status_label.setText(f"Image source error: {result.error}")
            if self._source_load_pending:
                self._source_load_pending = False
                QTimer.singleShot(0, self.run_pipeline)
            return
        self._file_source_payload_cache.update(result.payloads)
        self._prune_file_source_payload_cache()
        self._set_pipeline_busy(False)
        self._source_load_pending = False
        QTimer.singleShot(0, self.run_pipeline)

    def _should_run_pipeline_in_background(
        self,
        dirty_node_ids: set[str] | None = None,
        manual_node_ids: set[str] | None = None,
        source_payloads: dict[str, SourcePayload] | None = None,
    ) -> bool:
        return (
            self._background_processing_node_id(
                dirty_node_ids,
                manual_node_ids,
                source_payloads,
            )
            is not None
        )

    def _background_processing_node_id(
        self,
        dirty_node_ids: set[str] | None = None,
        manual_node_ids: set[str] | None = None,
        source_payloads: dict[str, SourcePayload] | None = None,
    ) -> str | None:
        manual_node_ids = set(manual_node_ids or set())
        if manual_node_ids:
            for node_id in self.pipeline.topological_order():
                if node_id in manual_node_ids:
                    return node_id
            return None

        if self.background_all_checkbox.isChecked():
            if dirty_node_ids:
                affected_node_ids = self.pipeline.descendants_inclusive(dirty_node_ids)
                for node_id in self.pipeline.topological_order():
                    if node_id in affected_node_ids:
                        return node_id
                return None
            order = self.pipeline.topological_order()
            return order[0] if order else None

        affected_node_ids = None
        if dirty_node_ids:
            affected_node_ids = self.pipeline.descendants_inclusive(dirty_node_ids)
        large_source_descendants: set[str] = set()
        for source_id, payload in (source_payloads or {}).items():
            if source_id in self.pipeline.nodes and _should_auto_background_data(
                payload.data
            ):
                large_source_descendants.update(
                    self.pipeline.descendants_inclusive({source_id})
                )
        for node_id in self.pipeline.topological_order():
            if affected_node_ids is not None and node_id not in affected_node_ids:
                continue
            node = self.pipeline.nodes.get(node_id)
            if node is not None and node.operation_id in BACKGROUND_PIPELINE_OPERATIONS:
                return node_id
            payload = (source_payloads or {}).get(node_id)
            if payload is not None and _should_auto_background_data(payload.data):
                return node_id
            inputs = self.pipeline.input_data_by_port_for_node(node_id)
            if any(_should_auto_background_data(data) for data in inputs.values()):
                return node_id
            if (
                inputs
                and any(data is None for data in inputs.values())
                and node_id in large_source_descendants
            ):
                return node_id
            if _should_auto_background_data(self.pipeline.outputs.get(node_id)):
                return node_id
        return None

    def _start_background_pipeline_run(
        self,
        input_data,
        input_metadata,
        input_name: str,
        source_payloads: dict[str, SourcePayload],
        primary_layer,
        source_label: str,
        source_signature: tuple,
        dirty_node_ids: set[str] | None,
        manual_node_ids: set[str] | None = None,
    ) -> None:
        manual_node_ids = set(manual_node_ids or set())
        processing_node_id = self._background_processing_node_id(
            dirty_node_ids,
            manual_node_ids,
            source_payloads,
        )
        if self._active_pipeline_run_id is not None:
            self._pipeline_run_pending = True
            if dirty_node_ids is None:
                self._pending_dirty_node_ids.update(self.pipeline.nodes)
            else:
                self._pending_dirty_node_ids.update(
                    node_id
                    for node_id in dirty_node_ids
                    if node_id in self.pipeline.nodes
                )
            self._pending_manual_node_ids.update(
                node_id
                for node_id in manual_node_ids
                if self.pipeline.is_manual_node(node_id)
            )
            active_node_id = self._active_pipeline_node_id
            pending_dirty = self._pending_dirty_node_ids & set(self.pipeline.nodes)
            cancel_active = (
                active_node_id not in self.pipeline.nodes
                or not pending_dirty
                or self._dirty_nodes_affect_node(pending_dirty, active_node_id)
            )
            if cancel_active:
                event = self._pipeline_cancel_events.get(self._active_pipeline_run_id)
                if event is not None:
                    event.set()
            self._set_pipeline_busy(
                True,
                self._active_pipeline_node_id or processing_node_id,
                queued=True,
                preserve_progress=not cancel_active,
            )
            title = (
                self._node_title(self._active_pipeline_node_id)
                if (self._active_pipeline_node_id in self.pipeline.nodes)
                else "graph"
            )
            if cancel_active:
                self.status_label.setText(
                    f"Canceling '{title}' and queuing the latest calculation."
                )
            else:
                self.status_label.setText(
                    f"Finishing '{title}'; independent graph edit queued."
                )
            return

        self._pipeline_run_serial += 1
        run_id = self._pipeline_run_serial
        workflow = deepcopy(serialize_workflow(self.pipeline))
        cancel_event = threading.Event()
        request = PipelineRunRequest(
            run_id=run_id,
            workflow=workflow,
            input_data=input_data,
            input_metadata=input_metadata,
            input_name=input_name,
            source_payloads=dict(source_payloads),
            dirty_node_ids=(
                frozenset(dirty_node_ids) if dirty_node_ids is not None else None
            ),
            cached_outputs=(
                dict(self.pipeline.outputs)
            ),
            cached_output_states=(
                dict(self.pipeline.output_states)
            ),
            cached_node_outputs=(
                {
                    node_id: list(outputs)
                    for node_id, outputs in self.pipeline.node_outputs.items()
                }
            ),
            cached_node_output_states=(
                {
                    node_id: list(states)
                    for node_id, states in self.pipeline.node_output_states.items()
                }
            ),
            cached_execution_states=(
                dict(self.pipeline.node_execution_states)
            ),
            cached_execution_messages=(
                dict(self.pipeline.node_execution_messages)
            ),
            completed_node_ids=frozenset(self.pipeline.completed_node_ids),
            manual_node_ids=(
                frozenset(manual_node_ids) if manual_node_ids else None
            ),
            retain_node_ids=frozenset(self._cache_retention_node_ids()),
            prune_unretained=self._cache_pruning_enabled(),
            cancel_event=cancel_event,
        )
        self._active_pipeline_run_id = run_id
        self._pipeline_cancel_events[run_id] = cancel_event
        self._begin_pipeline_dispatch(dirty_node_ids)
        self._pipeline_run_context[run_id] = (
            primary_layer,
            source_label,
            processing_node_id,
            source_signature,
            dirty_node_ids,
        )
        self._pipeline_run_manual_node_ids[run_id] = frozenset(manual_node_ids)
        self._set_pipeline_busy(True, processing_node_id)
        title = (
            self._node_title(processing_node_id)
            if (processing_node_id in self.pipeline.nodes)
            else "graph"
        )
        self.status_label.setText(f"Processing '{title}' in background...")
        worker = PipelineRunWorker(request)
        worker.signals.node_started.connect(self._on_background_pipeline_node_started)
        worker.signals.progress.connect(self._on_background_pipeline_progress)
        worker.signals.finished.connect(self._on_background_pipeline_finished)
        self._pipeline_thread_pool.start(worker)

    def _on_background_pipeline_node_started(self, payload: object) -> None:
        try:
            run_id, node_id = payload
        except Exception:
            return
        if run_id != self._active_pipeline_run_id:
            return
        node_id = str(node_id)
        self._set_pipeline_busy(True, node_id, queued=self._pipeline_run_pending)
        title = self._node_title(node_id) if node_id in self.pipeline.nodes else "graph"
        suffix = "; latest edit queued" if self._pipeline_run_pending else ""
        self.status_label.setText(f"Processing '{title}' in background{suffix}...")

    def _on_background_pipeline_progress(self, payload: object) -> None:
        try:
            run_id, node_id, current, total, message = payload
        except Exception:
            return
        if run_id != self._active_pipeline_run_id:
            return
        node_id = str(node_id)
        title = self._node_title(node_id) if node_id in self.pipeline.nodes else "graph"
        total = int(total)
        current = int(current)
        if total > 0:
            self.pipeline_busy_bar.setRange(0, total)
            self.pipeline_busy_bar.setValue(max(min(current, total), 0))
            self.pipeline_busy_bar.setTextVisible(True)
            self.pipeline_busy_bar.setFormat("%p%")
        else:
            self.pipeline_busy_bar.setRange(0, 0)
            self.pipeline_busy_bar.setTextVisible(False)
        detail_message = _progress_detail_message(title, message)
        detail = f": {detail_message}" if detail_message else ""
        suffix = "; latest edit queued" if self._pipeline_run_pending else ""
        self.pipeline_busy_label.setText(f"Processing: {title}{detail}{suffix}")

    def _on_background_pipeline_finished(self, result: PipelineRunResult) -> None:
        if result.run_id != self._active_pipeline_run_id:
            return
        self._active_pipeline_run_id = None
        self._pipeline_cancel_events.pop(result.run_id, None)
        inflight_manual_node_ids = set(
            self._pipeline_run_manual_node_ids.pop(result.run_id, frozenset())
        )
        (
            primary_layer,
            source_label,
            processing_node_id,
            source_signature,
            dirty_node_ids,
        ) = self._pipeline_run_context.pop(result.run_id, (None, "", None, None, None))
        pending_dirty = self._pending_dirty_node_ids & set(self.pipeline.nodes)
        can_apply_before_pending = bool(
            self._pipeline_run_pending
            and pending_dirty
            and not self._dirty_nodes_affect_node(pending_dirty, processing_node_id)
        )
        if self._pipeline_run_pending and not can_apply_before_pending:
            self._pipeline_run_pending = False
            self._requeue_inflight_dirty_nodes()
            self._pending_manual_node_ids.update(
                node_id
                for node_id in inflight_manual_node_ids
                if self.pipeline.is_manual_node(node_id)
            )
            self.status_label.setText(
                "Restarting background processing with latest edit."
            )
            QTimer.singleShot(0, self.run_pipeline)
            return
        if result.cancelled:
            if self._pipeline_run_pending:
                self._pipeline_run_pending = False
                self._requeue_inflight_dirty_nodes()
                self._pending_manual_node_ids.update(
                    node_id
                    for node_id in inflight_manual_node_ids
                    if self.pipeline.is_manual_node(node_id)
                )
                self.status_label.setText(
                    "Restarting background processing after cancellation."
                )
                QTimer.singleShot(0, self.run_pipeline)
                return
            self._set_pipeline_busy(False)
            self.status_label.setText("Background processing canceled.")
            return
        if result.error:
            self.pipeline.set_node_execution_error(processing_node_id, result.error)
            continue_pending = bool(self._pipeline_run_pending and pending_dirty)
            self._pipeline_run_pending = False
            self._inflight_dirty_node_ids = None
            self._set_pipeline_busy(False)
            suffix = "; continuing queued graph edit" if continue_pending else ""
            self.status_label.setText(f"Pipeline error: {result.error}{suffix}")
            if continue_pending:
                QTimer.singleShot(0, self.run_pipeline)
            return
        if result.pipeline is None:
            self.pipeline.set_node_execution_error(
                processing_node_id,
                "No result returned.",
            )
            continue_pending = bool(self._pipeline_run_pending and pending_dirty)
            self._pipeline_run_pending = False
            self._inflight_dirty_node_ids = None
            self._set_pipeline_busy(False)
            suffix = "; continuing queued graph edit" if continue_pending else ""
            self.status_label.setText(f"Pipeline error: no result returned{suffix}.")
            if continue_pending:
                QTimer.singleShot(0, self.run_pipeline)
            return
        if not self._workflow_matches_current_pipeline(result.workflow):
            if not can_apply_before_pending:
                self._requeue_inflight_dirty_nodes()
                self._pending_manual_node_ids.update(
                    node_id
                    for node_id in inflight_manual_node_ids
                    if self.pipeline.is_manual_node(node_id)
                )
                self.status_label.setText(
                    "Discarded stale background result; rerunning latest graph."
                )
                QTimer.singleShot(0, self.run_pipeline)
                return

        self._apply_pipeline_run_result(
            result.pipeline,
            update_params=(
                not can_apply_before_pending
                and not self._interactive_collection_source_paths
            ),
        )
        if can_apply_before_pending:
            self._last_pipeline_source_signature = source_signature
            self._inflight_dirty_node_ids = None
            self._pipeline_run_pending = False
        else:
            self._complete_pipeline_run(source_signature, dirty_node_ids)
        autodefault_changed = self._resync_autodefault_nodes()
        refreshed_selected = self._refresh_selected_parameter_controls()
        if autodefault_changed or refreshed_selected:
            # An auto-tracking node (Clip/Rescale range) or the selected node's
            # controls shifted because a descendant of the just-run dirty set
            # got new data. Re-queue only the nodes that actually changed so the
            # follow-up run starts there and reuses cached upstream output,
            # instead of recomputing the whole dirty subtree from its source.
            rerun_dirty = set(autodefault_changed)
            if refreshed_selected and self._selected_node_id in self.pipeline.nodes:
                rerun_dirty.add(self._selected_node_id)
            rerun_dirty &= set(self.pipeline.nodes)
            if rerun_dirty:
                self._pending_dirty_node_ids.update(rerun_dirty)
                QTimer.singleShot(0, self.run_pipeline)
                return
        self._set_pipeline_busy(False)
        if can_apply_before_pending:
            self.status_label.setText(
                "Cached upstream result; rerunning latest downstream edit."
            )
            QTimer.singleShot(0, self.run_pipeline)
            return
        self._finish_pipeline_update(primary_layer, source_label)

    def _workflow_matches_current_pipeline(self, workflow: dict) -> bool:
        return serialize_workflow(self.pipeline) == workflow

    def _apply_pipeline_run_result(
        self,
        result_pipeline: PrototypePipeline,
        *,
        update_params: bool = True,
    ) -> None:
        for node_id, node in self.pipeline.nodes.items():
            result_node = result_pipeline.nodes.get(node_id)
            if result_node is None or result_node.operation_id != node.operation_id:
                continue
            if update_params:
                node.params = dict(result_node.params)
        self._discard_pending_thumbnail_contrast_limit_requests()
        result_node_ids = set(result_pipeline.nodes)
        live_node_ids = tuple(self.pipeline.nodes)
        live_node_id_set = set(live_node_ids)
        self.pipeline.outputs = {
            node_id: (
                result_pipeline.outputs.get(node_id)
                if node_id in result_node_ids
                else self.pipeline.outputs.get(node_id)
            )
            for node_id in live_node_ids
        }
        self.pipeline.output_states = {
            node_id: (
                result_pipeline.output_states.get(node_id)
                if node_id in result_node_ids
                else self.pipeline.output_states.get(node_id)
            )
            for node_id in live_node_ids
        }
        self.pipeline.node_outputs = {
            node_id: list(
                result_pipeline.node_outputs.get(node_id, [])
                if node_id in result_node_ids
                else self.pipeline.node_outputs.get(node_id, [])
            )
            for node_id in live_node_ids
        }
        self.pipeline.node_output_states = {
            node_id: list(
                result_pipeline.node_output_states.get(node_id, [])
                if node_id in result_node_ids
                else self.pipeline.node_output_states.get(node_id, [])
            )
            for node_id in live_node_ids
        }
        self.pipeline.completed_node_ids = (
            set(result_pipeline.completed_node_ids)
            | (self.pipeline.completed_node_ids - result_node_ids)
        ) & live_node_id_set
        self.pipeline.node_execution_states = {
            node_id: (
                result_pipeline.node_execution_states.get(
                    node_id,
                    EXECUTION_NOT_CALCULATED,
                )
                if node_id in result_node_ids
                else self.pipeline.node_execution_states.get(
                    node_id,
                    EXECUTION_NOT_CALCULATED,
                )
            )
            for node_id in live_node_ids
        }
        self.pipeline.node_execution_messages = {
            node_id: (
                result_pipeline.node_execution_messages.get(node_id, "")
                if node_id in result_node_ids
                else self.pipeline.node_execution_messages.get(node_id, "")
            )
            for node_id in live_node_ids
        }

    def _cancel_background_pipeline_run(self) -> None:
        run_id = self._active_pipeline_run_id
        if run_id is None:
            self._pipeline_run_pending = False
            self._set_pipeline_busy(False)
            self.status_label.setText("No background run is active.")
            return

        event = self._pipeline_cancel_events.pop(run_id, None)
        if event is not None:
            event.set()
        self._active_pipeline_run_id = None
        self._pipeline_run_pending = False
        self._pending_manual_node_ids.clear()
        self._pipeline_run_context.pop(run_id, None)
        self._pipeline_run_manual_node_ids.pop(run_id, None)
        self._requeue_inflight_dirty_nodes()
        self._set_pipeline_busy(False)
        self.status_label.setText(
            "Canceled background processing. The current worker may finish in "
            "the background, but its result will be ignored."
        )

    def _set_pipeline_busy(
        self,
        busy: bool,
        node_id: str | None = None,
        *,
        queued: bool = False,
        cancelable: bool = True,
        preserve_progress: bool = False,
    ) -> None:
        keep_current_progress = bool(
            busy and preserve_progress and not self.pipeline_busy_bar.isHidden()
        )
        self.pipeline_busy_label.setVisible(busy)
        self.pipeline_busy_bar.setVisible(busy)
        self.pipeline_cancel_button.setVisible(busy and cancelable)
        if not busy:
            self.pipeline_busy_label.setText("Processing")
            self.pipeline_busy_bar.setRange(0, 0)
            self.pipeline_busy_bar.setTextVisible(False)
            self.graph_view.clear_node_processing()
            self._active_pipeline_node_id = None
            self._sync_execution_ui()
            return
        if not keep_current_progress:
            self.pipeline_busy_bar.setRange(0, 0)
            self.pipeline_busy_bar.setTextVisible(False)
        if node_id is None:
            node_id = self._active_pipeline_node_id
        if node_id not in self.pipeline.nodes:
            node_id = None
        if self._active_pipeline_node_id and self._active_pipeline_node_id != node_id:
            self.graph_view.set_node_processing(self._active_pipeline_node_id, False)
        self._active_pipeline_node_id = node_id
        if node_id is not None:
            if self.pipeline.is_manual_node(node_id):
                self.pipeline.node_execution_states[node_id] = EXECUTION_RUNNING
                self.pipeline.node_execution_messages[node_id] = ""
            title = self._node_title(node_id)
            suffix = " queued" if queued else ""
            self.pipeline_busy_label.setText(f"Processing: {title}{suffix}")
            self.graph_view.set_node_processing(node_id, True, queued=queued)
            self._sync_execution_ui()
        else:
            self.pipeline_busy_label.setText("Processing graph")

    def _sync_view_dims_bar(self) -> None:
        if self._syncing_view_dims_bar:
            return
        self._syncing_view_dims_bar = True
        try:
            self.view_dims_bar.set_axes(self._view_dim_axes())
        finally:
            self._syncing_view_dims_bar = False

    def _view_dim_axes(self) -> list[ViewDimAxis]:
        state = self._view_dims_state()
        if state is None or not getattr(state, "axes", None):
            return []
        current_step = self._current_step()
        if current_step is None:
            return []
        current_nsteps = self._current_step_nsteps()
        axes: list[ViewDimAxis] = []
        for axis_index, (axis, size) in enumerate(
            zip(state.axes, state.shape, strict=False)
        ):
            axis_name = str(getattr(axis, "name", "")).lower()
            if axis_name in {"x", "y", "rgb", "rgba"} or int(size) <= 1:
                continue
            step_axis = self._state_axis_to_step_axis(state, axis_index, current_step)
            if step_axis is None or step_axis < 0:
                continue
            step_size = (
                int(current_nsteps[step_axis])
                if current_nsteps is not None and step_axis < len(current_nsteps)
                else int(size)
            )
            step_value = (
                int(current_step[step_axis]) if step_axis < len(current_step) else 0
            )
            value = _local_dim_value_from_viewer(
                step_value,
                axis_size=int(size),
                viewer_axis_size=step_size,
            )
            label = axis_name.upper() if len(axis_name) == 1 else str(axis.name)
            axes.append(
                ViewDimAxis(
                    name=str(axis.name),
                    label=label,
                    step_axis=int(step_axis),
                    size=int(size),
                    value=value,
                )
            )
        return axes

    def _view_dims_state(self):
        if self._active_pinned_node_id in self.pipeline.output_states:
            data, state, _output_port = self._node_display_payload(
                self._active_pinned_node_id
            )
            if state is not None and data is not None and not is_table_data(data):
                return state
        data, state, _output_port = self._node_display_payload(
            self._selected_node_id
        )
        if state is not None and data is not None and not is_table_data(data):
            return state
        data = self.pipeline.outputs.get("input")
        state = self.pipeline.output_states.get("input")
        if state is not None and data is not None and not is_table_data(data):
            return state
        return None

    def _on_view_dim_changed(self, step_axis: int, local_value: int) -> None:
        if self._syncing_view_dims_bar:
            return
        matching_axis = next(
            (
                axis
                for axis in self._view_dim_axes()
                if int(axis.step_axis) == int(step_axis)
            ),
            None,
        )
        if matching_axis is None:
            return
        current_nsteps = self._current_step_nsteps()
        step_size = (
            int(current_nsteps[int(step_axis)])
            if current_nsteps is not None and int(step_axis) < len(current_nsteps)
            else int(matching_axis.size)
        )
        step_value = _viewer_dim_value_from_local(
            int(local_value),
            axis_size=int(matching_axis.size),
            viewer_axis_size=step_size,
        )
        if self._dims_linked():
            self._set_current_step_axis(int(step_axis), step_value)
        else:
            self._set_vipp_current_step_axis(int(step_axis), step_value)
            self._update_thumbnails()
            self._update_metadata_panel()
            self._update_histogram()
        self._sync_view_dims_bar()

    def _set_raw_current_step(self, axis: int, value: int) -> None:
        try:
            self.viewer.dims.set_current_step(int(axis), int(value))
            return
        except Exception:
            pass
        try:
            current = list(self.viewer.dims.current_step)
            while len(current) <= int(axis):
                current.append(0)
            current[int(axis)] = int(value)
            self.viewer.dims.current_step = tuple(current)
            try:
                self.viewer.dims.events.current_step.emit()
            except Exception:
                pass
        except Exception:
            return

    def _state_axis_to_step_axis(
        self,
        state,
        axis_index: int,
        current_step: tuple,
    ) -> int | None:
        try:
            source_axis = state.axes[axis_index].source_axis
        except Exception:
            source_axis = None
        if source_axis is not None:
            return int(source_axis)
        visible_indices = [
            index
            for index, axis in enumerate(getattr(state, "axes", ()))
            if str(getattr(axis, "name", "")).lower() not in {"rgb", "rgba"}
        ]
        if axis_index not in visible_indices:
            return None
        offset = max(len(tuple(current_step)) - len(visible_indices), 0)
        return offset + visible_indices.index(axis_index)

    def _set_current_step_axis(self, step_axis: int, value: int) -> None:
        raw_axis = self._raw_axis_for_current_step_axis(step_axis)
        self._set_raw_current_step(raw_axis, int(value))

    def _set_vipp_current_step_axis(self, step_axis: int, value: int) -> None:
        self._ensure_vipp_dims_state()
        values = list(self._vipp_current_step or ())
        while len(values) <= int(step_axis):
            values.append(0)
        nsteps = self._vipp_current_nsteps
        upper = (
            max(int(nsteps[int(step_axis)]) - 1, 0)
            if nsteps is not None and int(step_axis) < len(nsteps)
            else int(value)
        )
        values[int(step_axis)] = int(np.clip(int(value), 0, upper))
        self._vipp_current_step = self._normalized_vipp_current_step(
            tuple(values),
            nsteps,
        )

    def _raw_axis_for_current_step_axis(self, step_axis: int) -> int:
        raw_step = self._raw_current_step()
        if raw_step is None:
            return int(step_axis)
        layer = self._layer_by_name(self._inspect_layer_name)
        metadata = getattr(layer, "metadata", {}) if layer is not None else {}
        if not isinstance(metadata, dict):
            return int(step_axis)
        node_id = metadata.get("node_id")
        output_port = int(metadata.get("output_port", 0) or 0)
        state = self._node_output_state(node_id, output_port)
        axes = tuple(getattr(state, "axes", ()))
        if not axes:
            return int(step_axis)
        display_axis_indices = [
            index
            for index, axis in enumerate(axes)
            if not _state_axis_hidden_from_napari_dims(axis, metadata)
        ]
        if not display_axis_indices:
            return int(step_axis)
        offset = max(len(tuple(raw_step)) - len(display_axis_indices), 0)
        for display_position, state_axis_index in enumerate(display_axis_indices):
            raw_axis = offset + display_position
            if raw_axis < 0 or raw_axis >= len(raw_step):
                continue
            try:
                source_axis = axes[state_axis_index].source_axis
            except Exception:
                source_axis = None
            if source_axis is None:
                source_axis = state_axis_index
            if int(source_axis) == int(step_axis):
                return int(raw_axis)
        return int(step_axis)

    def _on_dims_changed(self, _event=None) -> None:
        if not self._dims_linked():
            return
        self._capture_vipp_dims_from_viewer()
        self._sync_view_dims_bar()
        self._update_thumbnails()
        self._update_metadata_panel()
        self._update_histogram()

    def _update_thumbnails(self) -> None:
        mode = self.preview_mode_combo.currentText()
        current_step = self._current_step()
        current_step_nsteps = self._current_step_nsteps()
        contrast_mode = self.thumbnail_contrast_combo.currentText()
        contrast_scope = self.thumbnail_scope_combo.currentText()
        previews_visible_globally = mode.lower() != "off"
        for node_id, data in self.pipeline.outputs.items():
            preview_data, preview_state, _output_port = self._node_display_payload(
                node_id,
                data,
            )
            node_output_type = self._node_output_type(node_id)
            self.graph_view.set_node_metadata(
                node_id, format_compact_metadata(preview_state)
            )
            self.graph_view.set_node_output_type(node_id, node_output_type)
            self.graph_view.set_node_can_pin(node_id, self._node_can_pin(node_id))
            preview_enabled = previews_visible_globally and self._node_preview_enabled(
                node_id
            )
            self.graph_view.set_node_preview_enabled(node_id, preview_enabled)
            if not preview_enabled:
                self.graph_view.set_thumbnail(node_id, None)
                continue
            contrast_limits = self._thumbnail_contrast_limits_for_node(
                node_id,
                preview_data,
                preview_state,
                contrast_mode,
                contrast_scope,
                node_output_type,
            )
            stack_scope = not str(contrast_scope).strip().lower().startswith("slice")
            effective_contrast_scope = (
                "Slice" if stack_scope and contrast_limits is None else contrast_scope
            )
            effective_scope_is_slice = (
                str(effective_contrast_scope).strip().lower().startswith("slice")
            )
            preview_consumes_contrast = self._thumbnail_preview_consumes_contrast(
                node_id,
                preview_data,
                preview_state,
            )
            preview = make_preview(
                preview_data,
                mode=mode,
                current_step=current_step,
                current_step_nsteps=current_step_nsteps,
                state=preview_state,
                channel_colors=self._node_preview_channel_colors(node_id),
                contrast_mode=contrast_mode,
                contrast_scope=effective_contrast_scope,
                contrast_limits=contrast_limits,
            )
            thumbnail = normalize_thumbnail_with_colormap(
                preview,
                colormap=self.thumbnail_colormap_combo.currentText(),
                contrast_mode=contrast_mode,
                contrast_reference=(
                    preview if effective_scope_is_slice else None
                ),
                contrast_limits=(
                    None if preview_consumes_contrast else contrast_limits
                ),
                data_kind=node_output_type,
            )
            self.graph_view.set_thumbnail(node_id, thumbnail)
        if self._active_pinned_node_id is not None and not self._node_can_pin(
            self._active_pinned_node_id
        ):
            self._clear_active_pin(status=False)
        else:
            self._sync_pin_ui()

    def _on_selected_preview_toggled(self, checked: bool) -> None:
        node_id = self._selected_node_id
        if self._node_preview_enabled(node_id) == bool(checked):
            return
        self._finish_parameter_history_group()
        before = self._current_history_snapshot()
        if checked:
            self._preview_disabled_node_ids.discard(node_id)
        else:
            self._preview_disabled_node_ids.add(node_id)
        self._update_thumbnails()
        self._push_undo_if_changed(before)
        state = "enabled" if checked else "disabled"
        self.status_label.setText(
            f"Thumbnail preview {state} for '{self._node_title(node_id)}'."
        )

    def _update_metadata_panel(self) -> None:
        if self._selected_node_id not in self.pipeline.nodes:
            self._clear_empty_inspector()
            return
        _data, state, _output_port = self._node_display_payload(
            self._selected_node_id
        )
        rows = metadata_table_rows(state)
        current_view = self._current_view_label(state)
        if current_view:
            rows.insert(4, MetadataRow("Current view", current_view))
        self.metadata_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            label_item = QTableWidgetItem(row.label)
            value_item = QTableWidgetItem(row.value)
            label_item.setFlags(label_item.flags() & ~Qt.ItemIsEditable)
            value_item.setFlags(value_item.flags() & ~Qt.ItemIsEditable)
            self.metadata_table.setItem(row_index, 0, label_item)
            self.metadata_table.setItem(row_index, 1, value_item)
        self.metadata_table.resizeRowsToContents()
        self.metadata_table.resizeColumnToContents(0)

        history = metadata_history_items(state)
        if history:
            self.history_label.setText(
                "\n".join(f"{index}. {entry}" for index, entry in enumerate(history, 1))
            )
        else:
            self.history_label.setText("No history yet.")
        self._update_table_preview()

    def _current_view_label(self, state) -> str:
        if state is None or not hasattr(state, "axes"):
            return ""
        try:
            current_step = tuple(self._current_step())
        except Exception:
            current_step = ()
        parts = []
        for axis_index, (axis, size) in enumerate(
            zip(state.axes, state.shape, strict=False)
        ):
            if axis.name.lower() in {"x", "y", "rgb"}:
                continue
            if int(size) <= 1:
                continue
            step = self._axis_step_label(axis_index, int(size), current_step, state)
            parts.append(f"{axis.name}={step}/{int(size) - 1}")
        return ", ".join(parts)

    def _axis_step_label(
        self,
        axis_index: int,
        axis_size: int,
        current_step: tuple,
        state,
    ) -> int:
        axis_index = _metadata_current_step_axis(state, axis_index, current_step)
        try:
            step = int(current_step[axis_index])
        except Exception:
            step = axis_size // 2
        return int(np.clip(step, 0, max(axis_size - 1, 0)))

    def _update_histogram(self) -> None:
        self._update_label_volume_histogram()
        self._update_colocalization_scatter()
        node = self.pipeline.nodes.get(self._selected_node_id)
        if node is None:
            self._current_output_histogram_key = None
            self._pending_output_histogram_request = None
            self.rescale_input_histogram_group.setHidden(True)
            self.rescale_input_histogram_scope_row.setHidden(True)
            self.rescale_input_histogram_plot.set_histogram(None, log_scale=False)
            self.histogram_plot.set_histogram(None, log_scale=False)
            return
        data, state, output_port = self._node_display_payload(
            self._selected_node_id
        )
        if is_table_data(data):
            self._current_output_histogram_key = None
            self._pending_output_histogram_request = None
            self.rescale_input_histogram_group.setHidden(True)
            self.rescale_input_histogram_scope_row.setHidden(True)
            self.rescale_input_histogram_plot.set_histogram(None, log_scale=False)
            self.histogram_group.setHidden(True)
            self.histogram_plot.set_histogram(None, log_scale=False)
            return
        current_step = self._current_step()
        current_step_nsteps = self._current_step_nsteps()
        self._update_rescale_input_histogram(node.id, current_step)
        self.histogram_group.setHidden(False)
        histogram_title = "Output Histogram"
        if node.operation_id == "split_channels":
            ports = self.pipeline.output_ports(node.id)
            if 0 <= output_port < len(ports):
                histogram_title = f"Output Histogram — {ports[output_port].label}"
        self.histogram_group.setTitle(histogram_title)
        scope_available = _histogram_has_stack_scope(data, state)
        self.histogram_scope_row.setHidden(not scope_available)
        scope = self.histogram_scope_combo.currentText() if scope_available else "Slice"
        histogram_source = _histogram_source(
            data,
            state=state,
            scope=scope,
            current_step=current_step,
            current_step_nsteps=current_step_nsteps,
        )
        if histogram_source is not None and _should_auto_background_data(
            histogram_source[0]
        ):
            self._queue_output_histogram(
                node_id=node.id,
                data=data,
                state=state,
                scope=scope,
                current_step=current_step,
                current_step_nsteps=current_step_nsteps,
                title=histogram_title,
            )
            return

        self._current_output_histogram_key = None
        self._pending_output_histogram_request = None
        counts, x_range, colors = _histogram_summary(
            data,
            state=state,
            scope=scope,
            current_step=current_step,
            current_step_nsteps=current_step_nsteps,
        )
        self.histogram_plot.set_histogram(
            counts,
            log_scale=self.histogram_log_checkbox.isChecked(),
            x_range=x_range,
            colors=colors,
        )
        self._set_histogram_explanation(
            self.histogram_group,
            total_values=(
                int(histogram_source[0].size) if histogram_source is not None else 0
            ),
            finite_values=(
                int(np.asarray(counts).sum()) if counts is not None else 0
            ),
            display_bins=(
                int(np.asarray(counts).shape[-1]) if counts is not None else 0
            ),
        )

    def _update_colocalization_scatter(self) -> None:
        node = self.pipeline.nodes.get(self._selected_node_id)
        visible = (
            node is not None
            and node.operation_id in COLOCALIZATION_SCATTER_OPERATIONS
        )
        self.colocalization_scatter_group.setHidden(not visible)
        if not visible or node is None:
            self._current_colocalization_scatter_key = None
            self._pending_colocalization_scatter_request = None
            self.colocalization_scatter_summary.setText("Connect two channel inputs.")
            self.colocalization_scatter_summary.setToolTip("")
            self.colocalization_scatter_plot.clear()
            self.colocalization_scatter_plot.setToolTip("")
            return

        inputs = self._colocalization_inputs_for_node(node.id)
        if inputs is None:
            self._current_colocalization_scatter_key = None
            self._pending_colocalization_scatter_request = None
            self.colocalization_scatter_summary.setText(
                "Connect all required channel and ROI inputs."
            )
            self.colocalization_scatter_summary.setToolTip("")
            self.colocalization_scatter_plot.clear(
                "Connect all required channel and ROI inputs."
            )
            self.colocalization_scatter_plot.setToolTip("")
            return
        mode = str(node.params.get("threshold_mode", "Manual"))
        threshold_1 = _safe_float(node.params.get("channel_1_threshold"), 25.0)
        threshold_2 = _safe_float(node.params.get("channel_2_threshold"), 25.0)
        intensity_max = max(
            _safe_float(node.params.get("intensity_max"), 255.0),
            1.0,
        )
        key = self._colocalization_scatter_key(
            node.id,
            inputs,
            threshold_mode=mode,
            threshold_1=threshold_1,
            threshold_2=threshold_2,
            intensity_max=intensity_max,
        )
        self._current_colocalization_scatter_key = key
        cached = self._colocalization_scatter_cache.get(key)
        if cached is not None:
            self._apply_colocalization_scatter_result(cached)
            return

        if any(_should_auto_background_data(value) for value in inputs):
            self._queue_colocalization_scatter(
                ColocalizationScatterRequest(
                    0,
                    key,
                    node.id,
                    tuple(inputs),
                    mode,
                    threshold_1,
                    threshold_2,
                    intensity_max=intensity_max,
                )
            )
            return

        try:
            ch1, ch2, roi_mask, warnings = colocalization_normalized_inputs(
                inputs,
                intensity_max=intensity_max,
            )
            if mode.lower().startswith("costes"):
                self._sync_colocalization_costes_thresholds(
                    node.id,
                    update_controls=True,
                )
                threshold_1 = _safe_float(
                    node.params.get("channel_1_threshold"),
                    threshold_1,
                )
                threshold_2 = _safe_float(
                    node.params.get("channel_2_threshold"),
                    threshold_2,
                )
            density_counts, roi_voxels, coloc_voxels = (
                _prepare_colocalization_scatter_density(
                    ch1,
                    ch2,
                    threshold_1=threshold_1,
                    threshold_2=threshold_2,
                    roi_mask=roi_mask,
                    intensity_max=intensity_max,
                    bins=COLOCALIZATION_SCATTER_BINS,
                )
            )
        except Exception as exc:
            message = f"Scatter unavailable: {exc}"
            self.colocalization_scatter_summary.setText(message)
            self.colocalization_scatter_summary.setToolTip(message)
            self.colocalization_scatter_plot.clear(message)
            self.colocalization_scatter_plot.setToolTip(message)
            return
        result = ColocalizationScatterResult(
            0,
            key,
            node.id,
            mode,
            threshold_1,
            threshold_2,
            intensity_max=intensity_max,
            density_counts=density_counts,
            roi_voxels=roi_voxels,
            colocalized_voxels=coloc_voxels,
            warnings=tuple(warnings),
        )
        self._cache_colocalization_scatter_result(result)
        self._apply_colocalization_scatter_result(result)

    @staticmethod
    def _colocalization_scatter_key(
        node_id: str,
        inputs: list[object],
        *,
        threshold_mode: str,
        threshold_1: float,
        threshold_2: float,
        intensity_max: float,
    ) -> tuple:
        identities = tuple(
            (
                id(value),
                tuple(getattr(value, "shape", ())),
                str(getattr(value, "dtype", "")),
            )
            for value in inputs
        )
        thresholds = (
            (None, None)
            if str(threshold_mode).lower().startswith("costes")
            else (float(threshold_1), float(threshold_2))
        )
        return (
            node_id,
            identities,
            str(threshold_mode).strip().lower(),
            thresholds,
            float(intensity_max),
            COLOCALIZATION_SCATTER_BINS,
        )

    def _queue_colocalization_scatter(
        self,
        request: ColocalizationScatterRequest,
    ) -> None:
        summary = (
            f"{request.threshold_mode} thresholds. Calculating the exact scatter "
            "density, ROI count, and colocalized count in the background..."
        )
        detail = (
            "Every ROI voxel contributes to the scatter density and summary counts. "
            "The exact calculation uses bounded-memory chunks."
        )
        self.colocalization_scatter_summary.setText(summary)
        self.colocalization_scatter_summary.setToolTip(detail)
        self.colocalization_scatter_plot.clear("Calculating exact counts...")
        self.colocalization_scatter_plot.setToolTip(detail)
        if self._active_colocalization_scatter_run_id is not None:
            if self._active_colocalization_scatter_key == request.key:
                if (
                    self._active_colocalization_scatter_cancel_event is None
                    or not self._active_colocalization_scatter_cancel_event.is_set()
                ):
                    self._pending_colocalization_scatter_request = None
                else:
                    self._pending_colocalization_scatter_request = request
                return
            if self._active_colocalization_scatter_cancel_event is not None:
                self._active_colocalization_scatter_cancel_event.set()
            self._pending_colocalization_scatter_request = request
            return
        self._start_colocalization_scatter_request(request)

    def _start_colocalization_scatter_request(
        self,
        request: ColocalizationScatterRequest,
    ) -> None:
        self._colocalization_scatter_serial += 1
        cancel_event = threading.Event()
        request = replace(
            request,
            run_id=self._colocalization_scatter_serial,
            cancel_event=cancel_event,
        )
        self._active_colocalization_scatter_run_id = request.run_id
        self._active_colocalization_scatter_key = request.key
        self._active_colocalization_scatter_cancel_event = cancel_event
        worker = ColocalizationScatterWorker(request)
        worker.signals.finished.connect(self._on_colocalization_scatter_finished)
        self._pipeline_thread_pool.start(worker, -1)

    def _on_colocalization_scatter_finished(
        self,
        result: ColocalizationScatterResult,
    ) -> None:
        if result.run_id != self._active_colocalization_scatter_run_id:
            return
        self._active_colocalization_scatter_run_id = None
        self._active_colocalization_scatter_key = None
        self._active_colocalization_scatter_cancel_event = None
        if not result.error:
            self._cache_colocalization_scatter_result(result)
        if (
            result.key == self._current_colocalization_scatter_key
            and result.node_id == self._selected_node_id
        ):
            self._apply_colocalization_scatter_result(result)

        pending = self._pending_colocalization_scatter_request
        self._pending_colocalization_scatter_request = None
        if (
            pending is not None
            and pending.key == self._current_colocalization_scatter_key
        ):
            self._start_colocalization_scatter_request(pending)

    def _cache_colocalization_scatter_result(
        self,
        result: ColocalizationScatterResult,
    ) -> None:
        self._colocalization_scatter_cache[result.key] = result
        while len(self._colocalization_scatter_cache) > 16:
            self._colocalization_scatter_cache.pop(
                next(iter(self._colocalization_scatter_cache))
            )

    def _apply_colocalization_scatter_result(
        self,
        result: ColocalizationScatterResult,
    ) -> None:
        if result.key != self._current_colocalization_scatter_key:
            return
        node = self.pipeline.nodes.get(result.node_id)
        if node is None or result.node_id != self._selected_node_id:
            return
        if result.error:
            message = f"Scatter unavailable: {result.error}"
            self.colocalization_scatter_summary.setText(message)
            self.colocalization_scatter_summary.setToolTip(message)
            self.colocalization_scatter_plot.clear(message)
            self.colocalization_scatter_plot.setToolTip(message)
            return
        if (
            str(result.threshold_mode).lower().startswith("costes")
            and str(node.params.get("threshold_mode", "")).lower().startswith(
                "costes"
            )
        ):
            for name, value in (
                ("channel_1_threshold", result.threshold_1),
                ("channel_2_threshold", result.threshold_2),
            ):
                node.params[name] = float(value)
                self._set_parameter_control_value(node.id, name, float(value))

        display_detail = (
            f"Exact scatter density from all {result.roi_voxels:,} ROI voxels."
        )
        colocalized_percentage = (
            f"{100.0 * result.colocalized_voxels / result.roi_voxels:.1f}%"
            if result.roi_voxels
            else "n/a"
        )
        count_detail = (
            f"{result.colocalized_voxels:,}/{result.roi_voxels:,} "
            f"({colocalized_percentage})"
        )
        self.colocalization_scatter_plot.set_density(
            result.density_counts,
            threshold_1=result.threshold_1,
            threshold_2=result.threshold_2,
            intensity_max=result.intensity_max,
            channel_1_color=node.params.get("channel_1_color", "Red"),
            channel_2_color=node.params.get("channel_2_color", "Green"),
            colormap=self.colocalization_scatter_colormap_combo.currentText(),
            log_counts=self.colocalization_scatter_log_checkbox.isChecked(),
            summary=f"Exact: {count_detail}",
        )
        summary = (
            f"{result.threshold_mode} thresholds. Exact colocalized count: "
            f"{count_detail}. {display_detail}"
        )
        if result.warnings:
            summary += " " + "; ".join(result.warnings)
        tooltip = (
            f"Exact summary and scatter density from all {result.roi_voxels:,} "
            "ROI voxels; "
            f"{count_detail} meet both thresholds. {display_detail} "
            "Every ROI voxel contributes. Drag a threshold line to switch to manual "
            "thresholds."
        )
        self.colocalization_scatter_summary.setText(summary)
        self.colocalization_scatter_summary.setToolTip(tooltip)
        self.colocalization_scatter_plot.setToolTip(tooltip)

    def _colocalization_inputs_for_node(self, node_id: str) -> list[object] | None:
        ports = self.pipeline.input_ports(node_id)
        if len(ports) < 2:
            return None
        data_by_port = self.pipeline.input_data_by_port_for_node(node_id)
        required = len(ports)
        inputs = [data_by_port.get(index) for index in range(required)]
        if any(value is None for value in inputs):
            return None
        return inputs

    def _sync_colocalization_costes_thresholds(
        self,
        node_id: str,
        *,
        update_controls: bool,
    ) -> bool:
        node = self.pipeline.nodes.get(node_id)
        if node is None or node.operation_id not in COLOCALIZATION_THRESHOLD_OPERATIONS:
            return False
        if not str(node.params.get("threshold_mode", "Manual")).lower().startswith(
            "costes"
        ):
            return False
        inputs = self._colocalization_inputs_for_node(node_id)
        if inputs is None:
            return False
        try:
            threshold_1, threshold_2 = colocalization_threshold_values(
                inputs,
                threshold_mode=node.params.get("threshold_mode", "Costes auto"),
                channel_1_threshold=node.params.get("channel_1_threshold", 25.0),
                channel_2_threshold=node.params.get("channel_2_threshold", 25.0),
            )
        except Exception:
            return False
        changed = False
        for name, value in (
            ("channel_1_threshold", threshold_1),
            ("channel_2_threshold", threshold_2),
        ):
            value = float(value)
            if not np.isclose(float(node.params.get(name, np.nan)), value):
                node.params[name] = value
                changed = True
            if update_controls:
                self._set_parameter_control_value(node_id, name, value)
        return changed

    def _set_parameter_control_value(
        self,
        node_id: str,
        name: str,
        value,
    ) -> None:
        widget = self._parameter_widgets.get(name)
        if widget is None:
            return
        specs = {
            spec.name: spec for spec in self.pipeline.node_parameter_specs(node_id)
        }
        spec = specs.get(name)
        if spec is None:
            return
        spec = self._effective_parameter_spec(node_id, spec)
        bounds = self._parameter_bounds_for(node_id, spec)
        widget.set_bounds(bounds, value, emit=False)

    def _on_colocalization_scatter_threshold_changed(
        self,
        channel_index: int,
        value: float,
    ) -> None:
        node = self.pipeline.nodes.get(self._selected_node_id)
        if node is None or node.operation_id not in COLOCALIZATION_THRESHOLD_OPERATIONS:
            return
        name = (
            "channel_1_threshold"
            if int(channel_index) == 1
            else "channel_2_threshold"
        )
        value = float(np.round(float(value), 2))
        changed = False
        if node.params.get("threshold_mode") != "Manual":
            self._record_parameter_undo(self._selected_node_id, "threshold_mode")
            self.pipeline.set_param(self._selected_node_id, "threshold_mode", "Manual")
            self._set_parameter_control_value(
                self._selected_node_id,
                "threshold_mode",
                "Manual",
            )
            changed = True
        if not np.isclose(float(node.params.get(name, np.nan)), value):
            self._record_parameter_undo(self._selected_node_id, name)
            self.pipeline.set_param(self._selected_node_id, name, value)
            self._set_parameter_control_value(self._selected_node_id, name, value)
            changed = True
        if not changed:
            return
        self._mark_pipeline_dirty(self._selected_node_id)
        self._update_colocalization_scatter()
        self._debounce_timer.start()

    def _on_input_histogram_marker_changed(self, label: str, value: float) -> None:
        node_id = self._selected_node_id
        node = self.pipeline.nodes.get(node_id)
        if node is None:
            return
        if (
            node.operation_id == "rescale_intensity"
            and not str(
                node.params.get(RESCALE_CUTOFF_MODE_PARAMETER, "Percentiles")
            ).lower().startswith("value")
        ):
            return
        if (
            node.operation_id == "clip_intensity"
            and not str(node.params.get("cutoff_mode", "Data range"))
            .lower()
            .startswith("value")
        ):
            return
        name = _input_histogram_marker_parameter(node.operation_id, label)
        if name is None:
            return
        value = self._paired_histogram_marker_value(node_id, name, value)
        value = self._coerce_histogram_parameter_value(node_id, name, value)
        if not self._set_histogram_parameter_value(node_id, name, value):
            return
        self._mark_pipeline_dirty(node_id)
        self._update_rescale_input_histogram(
            node_id,
            self._current_step(),
        )
        self._debounce_timer.start()

    def _on_label_volume_marker_changed(self, label: str, value: float) -> None:
        node_id = self._selected_node_id
        node = self.pipeline.nodes.get(node_id)
        if node is None or node.operation_id != "filter_labels_by_volume":
            return
        name = {"min": "min_volume", "max": "max_volume"}.get(str(label))
        if name is None:
            return
        value = self._paired_histogram_marker_value(node_id, name, value)
        value = self._coerce_histogram_parameter_value(node_id, name, value)
        if not self._set_histogram_parameter_value(node_id, name, value):
            return
        self._mark_pipeline_dirty(node_id)
        self._update_label_volume_histogram()
        self._debounce_timer.start()

    def _set_histogram_parameter_value(self, node_id: str, name: str, value) -> bool:
        node = self.pipeline.nodes.get(node_id)
        spec = self._parameter_spec_by_name(node_id, name)
        if node is None or spec is None:
            return False
        if not self._parameter_value_changed(spec, node.params.get(name), value):
            return False
        self._record_parameter_undo(node_id, name)
        self.pipeline.set_param(node_id, name, value)
        self._set_parameter_control_value(node_id, name, value)
        return True

    def _coerce_histogram_parameter_value(self, node_id: str, name: str, value):
        spec = self._parameter_spec_by_name(node_id, name)
        if spec is None:
            return float(value)
        spec = self._effective_parameter_spec(node_id, spec)
        bounds = self._parameter_bounds_for(node_id, spec)
        minimum = (
            bounds.minimum if bounds.entry_minimum is None else bounds.entry_minimum
        )
        maximum = (
            bounds.maximum if bounds.entry_maximum is None else bounds.entry_maximum
        )
        numeric = float(np.clip(float(value), float(minimum), float(maximum)))
        if spec.kind == "int":
            return int(round(numeric))
        decimals = max(int(getattr(spec, "decimals", 2) or 0), 0)
        return float(round(numeric, decimals))

    def _paired_histogram_marker_value(
        self,
        node_id: str,
        name: str,
        value: float,
    ) -> float:
        node = self.pipeline.nodes.get(node_id)
        if node is None:
            return float(value)
        pair = {
            "in_low_value": ("in_high_value", "low"),
            "in_high_value": ("in_low_value", "high"),
            "minimum": ("maximum", "low"),
            "maximum": ("minimum", "high"),
            "low_threshold": ("high_threshold", "low"),
            "high_threshold": ("low_threshold", "high"),
            "min_volume": ("max_volume", "low"),
            "max_volume": ("min_volume", "high"),
        }.get(name)
        if pair is None:
            return float(value)
        other_name, role = pair
        other = node.params.get(other_name)
        if other is None:
            return float(value)
        other_value = _safe_float(other, np.nan)
        if not np.isfinite(other_value):
            return float(value)
        if name == "min_volume" and int(round(other_value)) <= 0:
            return float(value)
        if role == "low":
            return float(min(float(value), other_value))
        return float(max(float(value), other_value))

    @staticmethod
    def _input_histogram_keys(
        node_id: str,
        operation_id: str,
        data,
        state,
        scope: str,
        current_step,
        current_step_nsteps,
        params: dict,
    ) -> tuple[tuple, tuple]:
        stack_scope = str(scope).strip().lower().startswith("stack")
        normalized_scope = "stack" if stack_scope else "slice"
        distribution_key = (
            id(data),
            tuple(getattr(data, "shape", ())),
            str(getattr(data, "dtype", "")),
            _histogram_state_signature(state),
            normalized_scope,
            (
                None
                if stack_scope
                else _histogram_slice_signature(
                    data,
                    state,
                    current_step,
                    current_step_nsteps,
                )
            ),
        )
        marker_key = _input_histogram_marker_key(operation_id, params)
        return distribution_key, (
            str(node_id),
            str(operation_id),
            distribution_key,
            marker_key,
        )

    def _cached_input_histogram_distribution(self, key: tuple, data):
        distribution = self._input_histogram_distribution_cache.get(key)
        if distribution is None:
            return None
        identity_ref = distribution.identity_ref
        if identity_ref is not None and identity_ref() is data:
            return distribution
        self._input_histogram_distribution_cache.pop(key, None)
        for result_key in tuple(self._input_histogram_cache):
            if self._input_histogram_cache[result_key].distribution_key == key:
                self._input_histogram_cache.pop(result_key, None)
        return None

    def _cache_input_histogram_distribution(
        self,
        key: tuple,
        distribution: InputHistogramDistribution,
    ) -> None:
        if distribution.identity_ref is None:
            return
        self._input_histogram_distribution_cache[key] = distribution
        while len(self._input_histogram_distribution_cache) > 16:
            expired_key = next(iter(self._input_histogram_distribution_cache))
            self._input_histogram_distribution_cache.pop(expired_key)
            for result_key in tuple(self._input_histogram_cache):
                if (
                    self._input_histogram_cache[result_key].distribution_key
                    == expired_key
                ):
                    self._input_histogram_cache.pop(result_key, None)

    @staticmethod
    def _calculate_input_histogram_distribution(
        data,
        *,
        state,
        scope: str,
        current_step,
        current_step_nsteps,
    ) -> InputHistogramDistribution:
        counts, x_range, colors = _histogram_summary(
            data,
            state=state,
            scope=scope,
            current_step=current_step,
            current_step_nsteps=current_step_nsteps,
        )
        source = _histogram_source(
            data,
            state=state,
            scope=scope,
            current_step=current_step,
            current_step_nsteps=current_step_nsteps,
        )
        try:
            identity_ref = weakref.ref(data)
        except TypeError:
            identity_ref = None
        return InputHistogramDistribution(
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

    @staticmethod
    def _input_histogram_result(
        *,
        key: tuple,
        distribution_key: tuple,
        node_id: str,
        operation_id: str,
        data,
        state,
        scope: str,
        current_step,
        current_step_nsteps,
        params: dict,
        title: str,
        distribution: InputHistogramDistribution,
    ) -> InputHistogramResult:
        marker_error = ""
        try:
            markers = _input_histogram_markers(
                operation_id,
                data,
                state=state,
                scope=scope,
                current_step=current_step,
                current_step_nsteps=current_step_nsteps,
                params=params,
            )
        except Exception as exc:
            markers = []
            marker_error = str(exc)
        return InputHistogramResult(
            0,
            key,
            node_id,
            counts=distribution.counts,
            x_range=distribution.x_range,
            colors=distribution.colors,
            markers=markers,
            title=title,
            marker_error=marker_error,
            total_values=distribution.total_values,
            finite_values=distribution.finite_values,
            display_bins=distribution.display_bins,
            distribution_key=distribution_key,
            distribution=distribution,
        )

    def _cache_input_histogram_result(self, result: InputHistogramResult) -> None:
        self._input_histogram_cache[result.key] = result
        while len(self._input_histogram_cache) > 32:
            self._input_histogram_cache.pop(next(iter(self._input_histogram_cache)))

    @staticmethod
    def _input_histogram_marker_requires_background(
        operation_id: str,
        params: dict,
        data,
        histogram_source,
    ) -> bool:
        if operation_id == "minimum_threshold":
            return True
        if operation_id == "rescale_intensity":
            percentile_mode = str(
                params.get(RESCALE_CUTOFF_MODE_PARAMETER, "Percentiles")
            ).casefold() == "percentiles"
            return percentile_mode and _should_auto_background_data(data)
        if operation_id == "clip_intensity":
            data_range_mode = (
                str(params.get("cutoff_mode", "Data range")).casefold()
                == "data range"
            )
            return data_range_mode and _should_auto_background_data(data)
        return (
            operation_id in GLOBAL_THRESHOLD_OPERATIONS
            and histogram_source is not None
            and _should_auto_background_data(histogram_source[0])
        )

    def _update_rescale_input_histogram(
        self,
        node_id: str,
        current_step,
    ) -> None:
        current_step_nsteps = (
            self._current_step_nsteps() if current_step is not None else None
        )
        node = self.pipeline.nodes.get(node_id)
        visible = node is not None and node.operation_id in INPUT_HISTOGRAM_OPERATIONS
        self.rescale_input_histogram_group.setHidden(not visible)
        if not visible:
            self._current_input_histogram_key = None
            self._pending_input_histogram_request = None
            self.rescale_input_histogram_group.setTitle("Input Histogram")
            self.rescale_input_histogram_scope_row.setHidden(True)
            self.rescale_input_histogram_plot.set_histogram(None, log_scale=False)
            return

        data = self.pipeline.input_data_for_node(node_id)
        if data is None or is_table_data(data):
            self._current_input_histogram_key = None
            self._pending_input_histogram_request = None
            self.rescale_input_histogram_group.setTitle("Input Histogram")
            self.rescale_input_histogram_scope_row.setHidden(True)
            self.rescale_input_histogram_plot.set_histogram(None, log_scale=False)
            return

        state = self.pipeline.input_state_for_node(node_id)
        scope_available = _histogram_has_stack_scope(data, state)
        if node.operation_id in GLOBAL_THRESHOLD_OPERATIONS:
            if scope_available:
                scope = str(node.params.get("threshold_scope", "Stack histogram"))
                scope_label = _threshold_histogram_scope_label(scope)
                title = f"Input Histogram ({scope_label})"
            else:
                scope = "Slice histogram"
                title = "Input Histogram"
            self.rescale_input_histogram_scope_row.setHidden(True)
        else:
            title = "Input Histogram"
            self.rescale_input_histogram_scope_row.setHidden(not scope_available)
            scope = (
                self.rescale_input_histogram_scope_combo.currentText()
                if scope_available
                else "Slice"
            )
        self.rescale_input_histogram_group.setTitle(title)
        histogram_source = _histogram_source(
            data,
            state=state,
            scope=scope,
            current_step=current_step,
            current_step_nsteps=current_step_nsteps,
        )
        distribution_key, key = self._input_histogram_keys(
            node.id,
            node.operation_id,
            data,
            state,
            scope,
            current_step,
            current_step_nsteps,
            node.params,
        )
        self._current_input_histogram_key = key
        distribution = self._cached_input_histogram_distribution(
            distribution_key,
            data,
        )
        cached = self._input_histogram_cache.get(key)
        if cached is not None and distribution is not None:
            self._apply_input_histogram_result(cached)
            return

        distribution_requires_background = (
            distribution is None
            and histogram_source is not None
            and _should_auto_background_data(histogram_source[0])
        )
        marker_requires_background = self._input_histogram_marker_requires_background(
            node.operation_id,
            node.params,
            data,
            histogram_source,
        )
        if distribution_requires_background or marker_requires_background:
            self._queue_input_histogram(
                node_id=node.id,
                operation_id=node.operation_id,
                data=data,
                state=state,
                scope=scope,
                current_step=current_step,
                current_step_nsteps=current_step_nsteps,
                params=node.params,
                title=title,
            )
            return

        self._pending_input_histogram_request = None
        if distribution is None:
            distribution = self._calculate_input_histogram_distribution(
                data,
                state=state,
                scope=scope,
                current_step=current_step,
                current_step_nsteps=current_step_nsteps,
            )
            self._cache_input_histogram_distribution(distribution_key, distribution)
        result = self._input_histogram_result(
            key=key,
            distribution_key=distribution_key,
            node_id=node.id,
            operation_id=node.operation_id,
            data=data,
            state=state,
            scope=scope,
            current_step=current_step,
            current_step_nsteps=current_step_nsteps,
            params=node.params,
            title=title,
            distribution=distribution,
        )
        self._cache_input_histogram_result(result)
        self._apply_input_histogram_result(result)

    def _queue_input_histogram(
        self,
        *,
        node_id: str,
        operation_id: str,
        data,
        state,
        scope: str,
        current_step,
        current_step_nsteps,
        params: dict,
        title: str,
    ) -> None:
        distribution_key, key = self._input_histogram_keys(
            node_id,
            operation_id,
            data,
            state,
            scope,
            current_step,
            current_step_nsteps,
            params,
        )
        self._current_input_histogram_key = key
        distribution = self._cached_input_histogram_distribution(
            distribution_key,
            data,
        )
        cached = self._input_histogram_cache.get(key)
        if cached is not None and distribution is not None:
            self._apply_input_histogram_result(cached)
            return

        if distribution is None:
            self.rescale_input_histogram_group.setTitle(f"{title} (calculating...)")
            self.rescale_input_histogram_plot.set_histogram(
                None,
                log_scale=self.rescale_input_histogram_log_checkbox.isChecked(),
            )
        else:
            self.rescale_input_histogram_group.setTitle(
                f"{title} (calculating marker...)"
            )
            self.rescale_input_histogram_plot.set_histogram(
                distribution.counts,
                log_scale=self.rescale_input_histogram_log_checkbox.isChecked(),
                x_range=distribution.x_range,
                colors=distribution.colors,
            )
        request = InputHistogramRequest(
            0,
            key,
            node_id,
            operation_id,
            data,
            state,
            scope,
            tuple(current_step) if current_step is not None else None,
            (
                tuple(current_step_nsteps)
                if current_step_nsteps is not None
                else None
            ),
            deepcopy(params),
            title,
            distribution_key=distribution_key,
            distribution=distribution,
        )
        if self._active_input_histogram_run_id is not None:
            if self._active_input_histogram_key == key:
                if (
                    self._active_input_histogram_cancel_event is None
                    or not self._active_input_histogram_cancel_event.is_set()
                ):
                    self._pending_input_histogram_request = None
                else:
                    self._pending_input_histogram_request = request
                return
            if self._active_input_histogram_cancel_event is not None:
                self._active_input_histogram_cancel_event.set()
            self._pending_input_histogram_request = request
            return
        self._start_input_histogram_request(request)

    def _start_input_histogram_request(
        self,
        request: InputHistogramRequest,
    ) -> None:
        self._input_histogram_serial += 1
        cancel_event = threading.Event()
        request = replace(
            request,
            run_id=self._input_histogram_serial,
            cancel_event=cancel_event,
        )
        self._active_input_histogram_run_id = request.run_id
        self._active_input_histogram_key = request.key
        self._active_input_histogram_cancel_event = cancel_event
        worker = InputHistogramWorker(request)
        worker.signals.finished.connect(self._on_input_histogram_finished)
        self._pipeline_thread_pool.start(worker, -1)

    def _on_input_histogram_finished(self, result: InputHistogramResult) -> None:
        if result.run_id != self._active_input_histogram_run_id:
            return
        self._active_input_histogram_run_id = None
        self._active_input_histogram_key = None
        self._active_input_histogram_cancel_event = None
        if result.distribution is not None and result.distribution_key:
            self._cache_input_histogram_distribution(
                result.distribution_key,
                result.distribution,
            )
        if not result.error:
            self._cache_input_histogram_result(result)
        if (
            result.key == self._current_input_histogram_key
            and result.node_id == self._selected_node_id
        ):
            self._apply_input_histogram_result(result)

        pending = self._pending_input_histogram_request
        self._pending_input_histogram_request = None
        if pending is not None and pending.key == self._current_input_histogram_key:
            distribution = self._cached_input_histogram_distribution(
                pending.distribution_key,
                pending.data,
            )
            if distribution is not None:
                pending = replace(pending, distribution=distribution)
            self._start_input_histogram_request(pending)

    def _apply_input_histogram_result(
        self,
        result: InputHistogramResult,
    ) -> None:
        if result.key != self._current_input_histogram_key:
            return
        node = self.pipeline.nodes.get(result.node_id)
        if node is None:
            return
        self.rescale_input_histogram_group.setTitle(
            f"{result.title} (marker unavailable)"
            if result.marker_error
            else result.title
        )
        if result.error:
            self.rescale_input_histogram_plot.set_histogram(None, log_scale=False)
            return
        self.rescale_input_histogram_plot.set_histogram(
            result.counts,
            log_scale=self.rescale_input_histogram_log_checkbox.isChecked(),
            x_range=result.x_range,
            colors=result.colors,
            markers=result.markers,
            draggable_markers=_input_histogram_draggable_markers(
                node.operation_id,
                node.params,
            ),
        )
        self._set_histogram_explanation(
            self.rescale_input_histogram_group,
            total_values=result.total_values,
            finite_values=result.finite_values,
            display_bins=result.display_bins,
            marker_error=result.marker_error,
        )

    @staticmethod
    def _set_histogram_explanation(
        group: QGroupBox,
        *,
        total_values: int,
        finite_values: int,
        display_bins: int,
        marker_error: str = "",
    ) -> None:
        ignored = max(int(total_values) - int(finite_values), 0)
        detail = (
            f"Exact counts from all {int(finite_values):,} finite pixels, grouped "
            f"into {int(display_bins):,} display bins."
        )
        if ignored:
            detail += f" {ignored:,} non-finite pixels are ignored."
        detail += " Display bins do not affect processing thresholds or cutoffs."
        if marker_error:
            detail += f" Processing marker unavailable: {marker_error}"
        group.setToolTip(detail)

    def _queue_output_histogram(
        self,
        *,
        node_id: str,
        data,
        state,
        scope: str,
        current_step,
        current_step_nsteps,
        title: str = "Output Histogram",
    ) -> None:
        stack_scope = str(scope).strip().lower().startswith("stack")
        key = (
            node_id,
            id(data),
            tuple(getattr(data, "shape", ())),
            str(getattr(data, "dtype", "")),
            str(scope).strip().lower(),
            None if stack_scope else tuple(current_step or ()),
            None if stack_scope else tuple(current_step_nsteps or ()),
            title,
        )
        self._current_output_histogram_key = key
        cached = self._output_histogram_cache.get(key)
        if cached is not None:
            self._apply_output_histogram_result(cached)
            return

        self.histogram_group.setTitle(f"{title} (calculating exact counts...)")
        self.histogram_plot.set_histogram(
            None,
            log_scale=self.histogram_log_checkbox.isChecked(),
        )
        request = InputHistogramRequest(
            0,
            key,
            node_id,
            "",
            data,
            state,
            scope,
            tuple(current_step) if current_step is not None else None,
            (
                tuple(current_step_nsteps)
                if current_step_nsteps is not None
                else None
            ),
            {},
            title,
        )
        if self._active_output_histogram_run_id is not None:
            if self._active_output_histogram_key == key:
                self._pending_output_histogram_request = None
                return
            self._pending_output_histogram_request = request
            return
        self._start_output_histogram_request(request)

    def _start_output_histogram_request(
        self,
        request: InputHistogramRequest,
    ) -> None:
        self._output_histogram_serial += 1
        request = replace(request, run_id=self._output_histogram_serial)
        self._active_output_histogram_run_id = request.run_id
        self._active_output_histogram_key = request.key
        worker = InputHistogramWorker(request)
        worker.signals.finished.connect(self._on_output_histogram_finished)
        self._pipeline_thread_pool.start(worker, -1)

    def _on_output_histogram_finished(self, result: InputHistogramResult) -> None:
        if result.run_id != self._active_output_histogram_run_id:
            return
        self._active_output_histogram_run_id = None
        self._active_output_histogram_key = None
        if not result.error:
            self._output_histogram_cache[result.key] = result
            while len(self._output_histogram_cache) > 16:
                self._output_histogram_cache.pop(next(iter(self._output_histogram_cache)))
        if (
            result.key == self._current_output_histogram_key
            and result.node_id == self._selected_node_id
        ):
            self._apply_output_histogram_result(result)

        pending = self._pending_output_histogram_request
        self._pending_output_histogram_request = None
        if pending is not None and pending.key == self._current_output_histogram_key:
            self._start_output_histogram_request(pending)

    def _apply_output_histogram_result(
        self,
        result: InputHistogramResult,
    ) -> None:
        if result.key != self._current_output_histogram_key:
            return
        self.histogram_group.setTitle(result.title)
        if result.error:
            self.histogram_plot.set_histogram(None, log_scale=False)
            return
        self.histogram_plot.set_histogram(
            result.counts,
            log_scale=self.histogram_log_checkbox.isChecked(),
            x_range=result.x_range,
            colors=result.colors,
        )
        self._set_histogram_explanation(
            self.histogram_group,
            total_values=result.total_values,
            finite_values=result.finite_values,
            display_bins=result.display_bins,
        )

    def _update_table_preview(self) -> None:
        data = self.pipeline.outputs.get(self._selected_node_id)
        if not is_table_data(data):
            self.table_group.setHidden(True)
            self.table_preview.setRowCount(0)
            self.table_preview.setColumnCount(0)
            self.table_summary.setText("No table output.")
            return

        self.table_group.setHidden(False)
        row_limit = 200
        shown_rows = min(data.row_count, row_limit)
        self.table_summary.setText(
            f"{data.row_count} rows x {data.column_count} columns"
            + (f" (showing first {shown_rows})" if data.row_count > row_limit else "")
        )
        self.table_preview.setColumnCount(data.column_count)
        self.table_preview.setRowCount(shown_rows)
        headers = [
            f"{column}\n({unit})" if (unit := data.unit_for(column)) else column
            for column in data.columns
        ]
        self.table_preview.setHorizontalHeaderLabels(headers)
        for row_index, row in enumerate(data.rows[:shown_rows]):
            for column_index, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.table_preview.setItem(row_index, column_index, item)
        self.table_preview.resizeColumnsToContents()
        self.table_preview.resizeRowsToContents()

    def _update_label_volume_histogram(self) -> None:
        node = self.pipeline.nodes.get(self._selected_node_id)
        visible = node is not None and node.operation_id == "filter_labels_by_volume"
        self.label_volume_group.setHidden(not visible)
        if not visible:
            self.label_volume_plot.set_histogram(None, log_scale=False)
            return

        data = self.pipeline.input_data_for_node(self._selected_node_id)
        if data is None:
            self.label_volume_summary.setText("No connected label input.")
            self.label_volume_plot.set_histogram(None, log_scale=False)
            return

        arr = np.asarray(data)
        spatial_ndim = self._label_filter_spatial_ndim(
            self._selected_node_id,
            arr,
        )
        volumes = self._cached_label_volumes(arr, spatial_ndim)
        if volumes.size == 0:
            self.label_volume_summary.setText("No labeled objects.")
            self.label_volume_plot.set_histogram(None, log_scale=False)
            return

        largest = int(volumes.max())
        median = float(np.median(volumes))
        unit = "voxels" if spatial_ndim >= 3 else "pixels"
        self.label_volume_summary.setText(
            f"{volumes.size} objects | median {_format_histogram_label(median)} "
            f"| largest {largest} {unit}"
        )
        bin_count = int(np.clip(np.ceil(np.sqrt(volumes.size)) * 2, 8, 64))
        logarithmic = self.label_volume_log_checkbox.isChecked()
        if logarithmic:
            histogram_values = np.log1p(volumes.astype(np.float64))
            histogram_range = (0.0, float(np.log1p(max(largest, 1))))
            x_scale = "log"
        else:
            histogram_values = volumes.astype(np.float64)
            histogram_range = (0.0, float(max(largest, 1)))
            x_scale = "linear"
        counts, _edges = np.histogram(
            histogram_values,
            bins=bin_count,
            range=histogram_range,
        )
        minimum = max(int(node.params.get("min_volume", 0)), 0)
        maximum = max(int(node.params.get("max_volume", 0)), 0)
        markers = [("min", float(minimum), QColor("#f59e0b"))]
        if maximum > 0:
            markers.append(("max", float(maximum), QColor("#38bdf8")))
        self.label_volume_plot.set_histogram(
            counts,
            log_scale=False,
            x_range=(0.0, float(max(largest, 1))),
            colors=[QColor("#f472b6")],
            markers=markers,
            x_scale=x_scale,
            draggable_markers={"min", "max"},
        )

    def _label_filter_spatial_ndim(
        self,
        node_id: str,
        data: np.ndarray,
    ) -> int:
        node = self.pipeline.nodes[node_id]
        mode = str(node.params.get("spatial_mode", "Auto from axes")).lower()
        if mode.startswith("2d"):
            requested = 2
        elif mode.startswith("3d"):
            requested = 3
        else:
            state = self.pipeline.input_state_for_node(node_id)
            spatial_count = (
                sum(axis.type == "space" for axis in state.axes)
                if state is not None
                else 0
            )
            if spatial_count >= 3:
                requested = 3
            elif spatial_count >= 2:
                requested = 2
            else:
                requested = 3 if data.ndim >= 3 else 2
        return int(np.clip(requested, 1, max(data.ndim, 1)))

    def _save_selected_output_dialog(self) -> None:
        node_id = self._selected_node_id
        selected_data, _selected_state, _output_port = self._node_display_payload(
            node_id
        )
        if is_table_data(selected_data):
            default_name = f"{self._node_title(node_id).replace(' ', '_')}.csv"
            path, selected_filter = QFileDialog.getSaveFileName(
                self,
                "Save selected table output",
                default_name,
                "CSV table (*.csv);;TSV table (*.tsv);;All files (*.*)",
            )
            if path:
                format = "tsv" if selected_filter.startswith("TSV") else "csv"
                self._save_node_output(node_id, path, format=format)
            return

        default_name = f"{self._node_title(node_id).replace(' ', '_')}.ome.tif"
        filters = (
            "OME-TIFF (*.ome.tif *.ome.tiff);;"
            "OME-Zarr (*.ome.zarr);;"
            "ImageJ TIFF (*.tif *.tiff);;"
            "TIFF (*.tif *.tiff);;"
            "NumPy array (*.npy);;"
        )
        if self._can_save_selected_output_as_raster(node_id):
            filters += (
                "PNG image (*.png);;"
                "JPEG image (*.jpg *.jpeg);;"
                "BMP image (*.bmp);;"
                "GIF image (*.gif);;"
                "WebP image (*.webp);;"
                "TGA image (*.tga);;"
                "PNM image (*.pnm *.pgm *.ppm *.pbm);;"
            )
        filters += "All files (*.*)"
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save selected node output",
            default_name,
            filters,
        )
        if path:
            selected_format = {
                "OME-TIFF": "ome-tiff",
                "OME-Zarr": "ome-zarr",
                "ImageJ TIFF": "imagej-tiff",
                "TIFF": "tiff",
                "NumPy array": "npy",
                "PNG image": "png",
                "JPEG image": "jpeg",
                "BMP image": "bmp",
                "GIF image": "gif",
                "WebP image": "webp",
                "TGA image": "tga",
                "PNM image": "pnm",
            }
            format = next(
                (
                    value
                    for label, value in selected_format.items()
                    if selected_filter.startswith(label)
                ),
                "auto",
            )
            self._save_node_output(node_id, path, format=format)

    def _can_save_selected_output_as_raster(self, node_id: str) -> bool:
        data, _state, _output_port = self._node_display_payload(node_id)
        if data is None or is_table_data(data):
            return False
        shape = getattr(data, "shape", None)
        if shape is None:
            try:
                shape = np.asarray(data).shape
            except Exception:
                return False
        shape = tuple(int(size) for size in shape)
        return len(shape) == 2 or (len(shape) == 3 and shape[-1] in {3, 4})

    def _save_node_output(
        self,
        node_id: str,
        path: str,
        *,
        format: str = "auto",
    ) -> Path | None:
        data, state, _output_port = self._node_display_payload(node_id)
        if data is None:
            self.status_label.setText("That node has no output to save yet.")
            return None
        try:
            if is_table_data(data):
                output_path = save_table_output(
                    data,
                    path,
                    format=format,
                    overwrite=True,
                )
            else:
                output_path = save_array_output(
                    data,
                    path,
                    format=format,
                    overwrite=True,
                    image_state=state,
                )
        except Exception as exc:
            self.status_label.setText(f"Save failed: {exc}")
            return None
        self.status_label.setText(
            f"Saved '{self._node_title(node_id)}' to {output_path}."
        )
        return output_path

    def inspect_node(self, node_id: str) -> None:
        data, state, output_port = self._node_display_payload(node_id)
        if data is None:
            self.status_label.setText("That node has no output to inspect yet.")
            return
        if is_table_data(data):
            self.status_label.setText(
                f"'{self._node_title(node_id)}' is shown in the table inspector."
            )
            return
        title = self._node_title(node_id)
        self._remember_cache_node(node_id)
        self._set_or_add_generated_layer(
            self._inspect_layer_name,
            data,
            metadata={
                "napari_vipp_kind": "inspect",
                "node_id": node_id,
                "output_port": output_port,
                "vipp_image_state": self._node_state_dict(
                    node_id,
                    output_port,
                    state=state,
                ),
            },
            role="inspect",
        )
        self._keep_active_pin_on_top()
        self.status_label.setText(f"Inspecting '{title}' in napari.")

    def _inspect_selected_node(self) -> None:
        data, _state, _output_port = self._node_display_payload(
            self._selected_node_id
        )
        if data is not None and not is_table_data(data):
            self.inspect_node(self._selected_node_id)
            return
        layer = self._layer_by_name(self._inspect_layer_name)
        metadata = getattr(layer, "metadata", {}) if layer is not None else {}
        if isinstance(metadata, dict) and metadata.get("node_id") == (
            self._selected_node_id
        ):
            self._remove_layer(layer)

    def pin_node(self, node_id: str) -> None:
        if not self._node_can_pin(node_id):
            self.status_label.setText(
                f"'{self._node_title(node_id)}' does not produce a displayable "
                "image output."
            )
            return
        self._finish_parameter_history_group()
        before = self._current_history_snapshot()
        if node_id == self._active_pinned_node_id:
            self._clear_active_pin(status=True)
            self._push_undo_if_changed(before)
            return
        data, _state, _output_port = self._node_display_payload(node_id)
        if data is None:
            self.status_label.setText("That node has no output to pin yet.")
            return
        self._remember_cache_node(node_id)
        self._set_active_pin_layer(node_id, data)
        self._push_undo_if_changed(before)
        self.status_label.setText(f"Pinned '{self._node_title(node_id)}'.")

    def _set_active_pin_layer(self, node_id: str, data) -> None:
        data, state, output_port = self._node_display_payload(node_id, data)
        title = self._node_title(node_id)
        layer_name = self._pinned_layer_name(title)
        for layer in self._active_pinned_layers():
            if layer.metadata.get("node_id") != node_id:
                self._remove_layer(layer)
        self._set_or_add_generated_layer(
            layer_name,
            data,
            metadata={
                "napari_vipp_kind": "pinned",
                "node_id": node_id,
                "output_port": output_port,
                "vipp_image_state": self._node_state_dict(
                    node_id,
                    output_port,
                    state=state,
                ),
            },
            role="pinned",
        )
        self._move_generated_layers_to_top(layer_name)
        self._active_pinned_node_id = node_id
        self._sync_pin_ui()
        self._sync_view_dims_bar()

    def _clear_active_pin(self, status: bool) -> None:
        title = (
            self._node_title(self._active_pinned_node_id)
            if self._active_pinned_node_id in self.pipeline.nodes
            else None
        )
        for layer in self._active_pinned_layers():
            self._remove_layer(layer)
        self._active_pinned_node_id = None
        self._sync_pin_ui()
        self._sync_view_dims_bar()
        if status and title is not None:
            self.status_label.setText(f"Unpinned '{title}'.")

    def _refresh_inspection_layer_if_active(self) -> None:
        layer = self._layer_by_name(self._inspect_layer_name)
        if layer is None:
            return
        node_id = getattr(layer, "metadata", {}).get("node_id")
        if node_id in self.pipeline.outputs:
            data, state, output_port = self._node_display_payload(node_id)
            if data is None or is_table_data(data):
                self._remove_layer(layer)
                return
            self._set_or_add_generated_layer(
                self._inspect_layer_name,
                data,
                metadata={
                    "napari_vipp_kind": "inspect",
                    "node_id": node_id,
                    "output_port": output_port,
                    "vipp_image_state": self._node_state_dict(
                        node_id,
                        output_port,
                        state=state,
                    ),
                },
                role="inspect",
            )
            self._keep_active_pin_on_top()

    def _refresh_pinned_layer_if_active(self) -> None:
        if self._active_pinned_node_id is None:
            return
        data, _state, _output_port = self._node_display_payload(
            self._active_pinned_node_id
        )
        if data is None:
            self._clear_active_pin(status=False)
            return
        self._set_active_pin_layer(self._active_pinned_node_id, data)

    def _layer_by_name(self, name: str):
        try:
            return self.viewer.layers[name]
        except Exception:
            for layer in self.viewer.layers:
                if layer.name == name:
                    return layer
        return None

    def _set_or_add_generated_layer(
        self,
        name: str,
        data,
        metadata: dict,
        role: str,
    ) -> None:
        saved_step = self._raw_current_step()
        saved_nsteps = self._viewer_nsteps()
        output_port = int(metadata.get("output_port", 0) or 0)
        data_kind = self._data_kind(
            data,
            metadata.get("node_id"),
            output_port,
        )
        display_kind = self._display_kind(data_kind, role)
        display_data = self._display_data(
            data,
            as_labels=display_kind == "labels",
        )
        metadata = {
            **metadata,
            "data_kind": data_kind,
            "display_kind": display_kind,
            "display_ndim": np.asarray(display_data).ndim,
            "display_shape": tuple(np.asarray(display_data).shape),
            "display_rgb": self._display_rgb(
                data,
                metadata.get("node_id"),
                output_port,
            ),
        }
        if self._display_rgb_as_channel_layers(display_data, metadata):
            self._set_or_add_rgb_channel_layers(name, display_data, metadata)
            self._restore_viewer_step(saved_step, saved_nsteps)
            return
        self._remove_rgb_channel_layers(name)
        layer = self._layer_by_name(name)
        if layer is None:
            self._add_image_or_labels(
                name,
                data,
                metadata=metadata,
                display_data=display_data,
            )
            self._restore_viewer_step(saved_step, saved_nsteps)
            return
        if self._generated_layer_needs_replacement(layer, metadata):
            self._remove_layer(layer)
            self._add_image_or_labels(
                name,
                data,
                metadata=metadata,
                display_data=display_data,
            )
            self._restore_viewer_step(saved_step, saved_nsteps)
            return
        layer.data = display_data
        layer.metadata.update(metadata)
        layer.visible = True
        self._configure_generated_layer(layer, data, metadata)
        self._restore_viewer_step(saved_step, saved_nsteps)

    def _viewer_nsteps(self) -> tuple[int, ...] | None:
        try:
            return tuple(int(value) for value in self.viewer.dims.nsteps)
        except Exception:
            return None

    def _restore_viewer_step(
        self,
        step: tuple | None,
        previous_nsteps: tuple[int, ...] | None,
    ) -> None:
        if step is None:
            return
        try:
            dims = self.viewer.dims
            current = tuple(dims.current_step)
        except Exception:
            return
        target_ndim = len(current)
        source = tuple(step)
        if target_ndim <= 0:
            return
        offset = max(target_ndim - len(source), 0)
        aligned = [0] * target_ndim
        for axis in range(target_ndim):
            source_axis = axis - offset
            if 0 <= source_axis < len(source):
                aligned[axis] = int(source[source_axis])
            else:
                aligned[axis] = int(current[axis])
        nsteps = tuple(int(value) for value in getattr(dims, "nsteps", ()))

        # Preserve relative position (for example through Z rescaling) instead
        # of forcing the old absolute index after layer replacement.
        if previous_nsteps is not None:
            previous = tuple(int(value) for value in previous_nsteps)
            for axis in range(target_ndim):
                source_axis = axis - offset
                if source_axis < 0 or source_axis >= len(source):
                    continue
                if source_axis >= len(previous) or axis >= len(nsteps):
                    continue
                old_max = max(previous[source_axis] - 1, 0)
                new_max = max(nsteps[axis] - 1, 0)
                if old_max <= 0 or new_max <= 0:
                    continue
                ratio = float(np.clip(source[source_axis], 0, old_max)) / float(old_max)
                aligned[axis] = int(round(ratio * new_max))

        for axis, value in enumerate(aligned):
            if axis < len(nsteps):
                upper = max(int(nsteps[axis]) - 1, 0)
                value = int(np.clip(value, 0, upper))
            if axis < len(current) and int(current[axis]) == int(value):
                continue
            try:
                dims.set_current_step(axis, int(value))
            except Exception:
                pass

    def _add_image_or_labels(
        self,
        name: str,
        data,
        metadata: dict,
        display_data=None,
    ):
        if display_data is None:
            display_data = self._display_data(
                data,
                as_labels=metadata.get("display_kind") == "labels",
            )
        scale = _layer_scale_from_metadata(metadata)
        if metadata["display_kind"] == "labels" and hasattr(self.viewer, "add_labels"):
            kwargs = {"name": name, "metadata": metadata}
            if scale is not None:
                kwargs["scale"] = scale
            return self.viewer.add_labels(display_data, **kwargs)
        kwargs = {"name": name, "metadata": metadata}
        if scale is not None:
            kwargs["scale"] = scale
        if metadata.get("display_rgb"):
            kwargs["rgb"] = True
        if metadata["data_kind"] == "mask":
            kwargs.update(
                {
                    "blending": "opaque",
                    "colormap": "gray",
                    "contrast_limits": (0, 1),
                }
            )
        else:
            plan = self._generated_layer_contrast_plan(name, data)
            metadata.update(self._generated_layer_contrast_metadata(plan))
            kwargs["contrast_limits"] = plan.limits
        return self.viewer.add_image(display_data, **kwargs)

    def _display_rgb_as_channel_layers(self, display_data, metadata: dict) -> bool:
        arr = np.asarray(display_data)
        return (
            bool(metadata.get("display_rgb"))
            and arr.ndim > 3
            and arr.shape[-1] in (3, 4)
        )

    def _set_or_add_rgb_channel_layers(
        self,
        name: str,
        display_data,
        metadata: dict,
    ) -> None:
        arr = np.asarray(display_data)
        base_layer = self._layer_by_name(name)
        if base_layer is not None and not bool(
            base_layer.metadata.get("display_rgb_as_channels")
        ):
            self._remove_layer(base_layer)
        for index, channel_name, colormap in _RGB_VOLUME_CHANNELS:
            channel_data = arr[..., index]
            layer_name = _rgb_channel_layer_name(name, index)
            channel_metadata = {
                **metadata,
                "display_rgb_as_channels": True,
                "display_rgb_group": name,
                "display_rgb_channel": channel_name,
                "display_rgb_channel_index": index,
                "display_ndim": channel_data.ndim,
                "display_shape": tuple(channel_data.shape),
            }
            layer = self._layer_by_name(layer_name)
            if layer is not None and self._generated_layer_needs_replacement(
                layer,
                channel_metadata,
            ):
                self._remove_layer(layer)
                layer = None
            if layer is None:
                self._add_rgb_channel_layer(
                    layer_name,
                    channel_data,
                    channel_metadata,
                    colormap,
                    identity_data=arr,
                    channel_index=index,
                )
                continue
            layer.data = channel_data
            layer.metadata.update(channel_metadata)
            layer.visible = True
            self._configure_rgb_channel_layer(
                layer,
                channel_data,
                channel_metadata,
                identity_data=arr,
                channel_index=index,
            )
        self._remove_extra_rgb_channel_layers(name)

    def _add_rgb_channel_layer(
        self,
        name: str,
        data,
        metadata: dict,
        colormap: str,
        *,
        identity_data,
        channel_index: int,
    ):
        kwargs = {
            "name": name,
            "metadata": metadata,
            "colormap": colormap,
            "blending": "additive",
        }
        scale = _layer_scale_from_metadata(metadata)
        if scale is not None:
            kwargs["scale"] = scale
        plan = self._generated_layer_contrast_plan(
            name,
            data,
            identity_data=identity_data,
            channel_index=channel_index,
        )
        metadata.update(self._generated_layer_contrast_metadata(plan))
        kwargs["contrast_limits"] = plan.limits
        return self.viewer.add_image(data, **kwargs)

    def _configure_rgb_channel_layer(
        self,
        layer,
        data,
        metadata: dict,
        *,
        identity_data,
        channel_index: int,
    ) -> None:
        channel_index = int(metadata["display_rgb_channel_index"])
        colormap = _RGB_VOLUME_CHANNELS[channel_index][2]
        for attr, value in (
            ("colormap", colormap),
            ("blending", "additive"),
        ):
            try:
                setattr(layer, attr, value)
            except Exception:
                pass
        scale = _layer_scale_from_metadata(metadata)
        if scale is not None:
            try:
                layer.scale = scale
            except Exception:
                pass
        plan = self._generated_layer_contrast_plan(
            layer.name,
            data,
            identity_data=identity_data,
            channel_index=channel_index,
        )
        layer.metadata.update(self._generated_layer_contrast_metadata(plan))
        try:
            layer.contrast_limits = plan.limits
        except Exception:
            pass

    def _rgb_channel_layers(self, group_name: str) -> list:
        layers = []
        for layer in list(self.viewer.layers):
            try:
                if layer.metadata.get("display_rgb_group") == group_name:
                    layers.append(layer)
            except Exception:
                continue
        return layers

    def _remove_rgb_channel_layers(self, group_name: str) -> None:
        for layer in self._rgb_channel_layers(group_name):
            self._remove_layer(layer)

    def _remove_extra_rgb_channel_layers(self, group_name: str) -> None:
        expected = {
            _rgb_channel_layer_name(group_name, index)
            for index, _channel_name, _colormap in _RGB_VOLUME_CHANNELS
        }
        for layer in self._rgb_channel_layers(group_name):
            if layer.name not in expected:
                self._remove_layer(layer)

    def _generated_layers_for_name(self, name: str) -> list:
        layers = self._rgb_channel_layers(name)
        if layers:
            ordered = {
                _rgb_channel_layer_name(name, index): index
                for index, _channel_name, _colormap in _RGB_VOLUME_CHANNELS
            }
            return sorted(layers, key=lambda layer: ordered.get(layer.name, 99))
        layer = self._layer_by_name(name)
        return [layer] if layer is not None else []

    def _move_generated_layers_to_top(self, name: str) -> None:
        for layer in self._generated_layers_for_name(name):
            self._move_layer_to_top(layer)

    def _generated_layer_needs_replacement(self, layer, metadata: dict) -> bool:
        return (
            layer.metadata.get("display_kind") != metadata["display_kind"]
            or layer.metadata.get("data_kind") != metadata["data_kind"]
            or layer.metadata.get("display_ndim") != metadata["display_ndim"]
            or tuple(layer.metadata.get("display_shape", ()))
            != tuple(metadata["display_shape"])
            or bool(layer.metadata.get("display_rgb"))
            != bool(metadata.get("display_rgb"))
            or bool(layer.metadata.get("display_rgb_as_channels"))
            != bool(metadata.get("display_rgb_as_channels"))
            or layer.metadata.get("display_rgb_channel_index")
            != metadata.get("display_rgb_channel_index")
        )

    def _configure_generated_layer(self, layer, data, metadata: dict) -> None:
        scale = _layer_scale_from_metadata(metadata)
        if scale is not None:
            try:
                layer.scale = scale
            except Exception:
                pass
        if metadata["display_kind"] != "image":
            return
        if metadata["data_kind"] == "mask":
            for attr, value in (
                ("blending", "opaque"),
                ("colormap", "gray"),
                ("contrast_limits", (0, 1)),
            ):
                try:
                    setattr(layer, attr, value)
                except Exception:
                    pass
        else:
            plan = self._generated_layer_contrast_plan(layer.name, data)
            layer.metadata.update(self._generated_layer_contrast_metadata(plan))
            try:
                layer.contrast_limits = plan.limits
            except Exception:
                pass

    def _generated_layer_contrast_plan(
        self,
        layer_name: str,
        data,
        *,
        identity_data=None,
        channel_index: int | None = None,
    ) -> GeneratedLayerContrastPlan:
        """Return explicit limits now and queue an exact scan when data is large.

        Supplying provisional limits is important: without them napari may scan
        the complete array synchronously while constructing the layer. The
        provisional values affect display only and are replaced by exact finite
        extrema calculated over the full data, never by a sample.
        """
        arr = np.asarray(data)
        identity = data if identity_data is None else identity_data
        identity_arr = np.asarray(identity)
        key = (
            self._generated_layer_contrast_generation,
            str(layer_name),
            id(identity),
            tuple(int(size) for size in identity_arr.shape),
            str(identity_arr.dtype),
            tuple(int(size) for size in arr.shape),
            str(arr.dtype),
            None if channel_index is None else int(channel_index),
        )
        self._generated_layer_contrast_keys[str(layer_name)] = key

        cached = self._generated_layer_contrast_cache.get(key)
        if cached is not None:
            identity_ref, cached_limits = cached
            if identity_ref() is identity:
                return GeneratedLayerContrastPlan(
                    key,
                    cached_limits,
                    False,
                    True,
                )
            self._generated_layer_contrast_cache.pop(key, None)

        provisional = _provisional_generated_layer_contrast_limits(arr)
        if arr.size == 0:
            return GeneratedLayerContrastPlan(key, provisional, False, False)

        if _should_auto_background_data(identity_arr):
            if key not in self._generated_layer_contrast_pending:
                self._generated_layer_contrast_pending.add(key)
                request = GeneratedLayerContrastRequest(
                    key,
                    str(layer_name),
                    arr,
                    identity,
                )
                worker = GeneratedLayerContrastWorker(request)
                worker.signals.finished.connect(
                    self._on_generated_layer_contrast_finished
                )
                self._pipeline_thread_pool.start(worker, -1)
            return GeneratedLayerContrastPlan(key, provisional, True, False)

        limits = _exact_generated_layer_contrast_limits(arr)
        if limits is None:
            return GeneratedLayerContrastPlan(key, provisional, False, False)
        self._cache_generated_layer_contrast(key, identity, limits)
        return GeneratedLayerContrastPlan(key, limits, False, True)

    @staticmethod
    def _generated_layer_contrast_metadata(
        plan: GeneratedLayerContrastPlan,
    ) -> dict[str, object]:
        if plan.exact:
            basis = "Exact full finite data range (display only)"
        elif plan.pending:
            basis = "Provisional dtype range; exact full-data scan pending"
        else:
            basis = "Provisional dtype range; no finite values available"
        metadata = {
            "vipp_display_contrast_basis": basis,
            "vipp_display_contrast_pending": bool(plan.pending),
            "vipp_display_contrast_adjustable": True,
            "_vipp_display_contrast_key": plan.key,
            "_vipp_display_contrast_initial_limits": plan.limits,
        }
        if plan.exact:
            metadata["vipp_exact_finite_data_range"] = plan.limits
        return metadata

    def _cache_generated_layer_contrast(
        self,
        key: tuple,
        identity,
        limits: tuple[float, float],
    ) -> None:
        try:
            identity_ref = weakref.ref(identity)
        except TypeError:
            return
        self._generated_layer_contrast_cache[key] = (identity_ref, limits)
        while len(self._generated_layer_contrast_cache) > 32:
            self._generated_layer_contrast_cache.pop(
                next(iter(self._generated_layer_contrast_cache))
            )

    def _on_generated_layer_contrast_finished(
        self,
        result: GeneratedLayerContrastResult,
    ) -> None:
        self._generated_layer_contrast_pending.discard(result.key)
        if not result.key or (
            result.key[0] != self._generated_layer_contrast_generation
        ):
            return
        if (
            not result.error
            and result.limits is not None
            and result.identity is not None
        ):
            self._cache_generated_layer_contrast(
                result.key,
                result.identity,
                result.limits,
            )
        if self._generated_layer_contrast_keys.get(result.layer_name) != result.key:
            return
        layer = self._layer_by_name(result.layer_name)
        if layer is None:
            return
        try:
            if layer.metadata.get("_vipp_display_contrast_key") != result.key:
                return
        except Exception:
            return

        if result.error:
            layer.metadata.update(
                {
                    "vipp_display_contrast_basis": (
                        "Provisional dtype range; exact full-data scan failed"
                    ),
                    "vipp_display_contrast_pending": False,
                }
            )
            self.status_label.setText(
                f"Display contrast calculation failed: {result.error}"
            )
            return
        if result.limits is None:
            layer.metadata.update(
                {
                    "vipp_display_contrast_basis": (
                        "Provisional dtype range; no finite values available"
                    ),
                    "vipp_display_contrast_pending": False,
                }
            )
            return
        try:
            initial_limits = tuple(
                float(value)
                for value in layer.metadata.get(
                    "_vipp_display_contrast_initial_limits",
                    (),
                )
            )
            current_limits = tuple(float(value) for value in layer.contrast_limits)
        except (TypeError, ValueError):
            initial_limits = ()
            current_limits = ()
        if (
            len(initial_limits) == 2
            and len(current_limits) == 2
            and current_limits != initial_limits
        ):
            layer.metadata.update(
                {
                    "vipp_display_contrast_basis": (
                        "User-adjusted display limits; exact full finite data "
                        "range retained in metadata"
                    ),
                    "vipp_display_contrast_pending": False,
                    "vipp_exact_finite_data_range": result.limits,
                }
            )
            return
        try:
            layer.contrast_limits = result.limits
        except Exception:
            return
        layer.metadata.update(
            {
                "vipp_display_contrast_basis": (
                    "Exact full finite data range (display only)"
                ),
                "vipp_display_contrast_pending": False,
                "vipp_exact_finite_data_range": result.limits,
            }
        )

    def _display_kind(self, data_kind: str, role: str) -> str:
        if data_kind == "labels":
            return "labels"
        if role == "pinned" and data_kind == "mask":
            return "labels"
        return "image"

    def _data_kind(
        self,
        data,
        node_id: str | None = None,
        output_port: int = 0,
    ) -> str:
        if is_table_data(data):
            return "table"
        if node_id is not None:
            ports = self.pipeline.output_ports(node_id)
            port_index = int(np.clip(output_port, 0, max(len(ports) - 1, 0)))
            if ports and ports[port_index].output_type == "table":
                return "table"
            if ports and ports[port_index].output_type == "labels":
                return "labels"
        return "mask" if np.asarray(data).dtype == bool else "image"

    def _display_rgb(
        self,
        data,
        node_id: str | None = None,
        output_port: int = 0,
    ) -> bool:
        if data is None or is_table_data(data):
            return False
        arr = np.asarray(data)
        if arr.ndim < 3 or arr.shape[-1] not in (3, 4):
            return False
        state = self._node_output_state(node_id, output_port) if node_id else None
        kind = str(getattr(state, "kind", "")).lower()
        return kind in {"rgb image", "rgba image"}

    def _display_data(self, data, *, as_labels: bool = False):
        if is_table_data(data):
            raise ValueError(
                "Table outputs cannot be displayed as napari image layers."
            )
        arr = np.asarray(data)
        if as_labels and arr.dtype == bool:
            arr = arr.astype(np.uint8)
        if arr.ndim == 0:
            return arr.reshape(1, 1)
        if arr.ndim == 1:
            return arr.reshape(1, arr.shape[0])
        return arr

    def _active_pinned_layer(self):
        layers = self._active_pinned_layers()
        return layers[0] if layers else None

    def _active_pinned_layers(self) -> list:
        layers = []
        for layer in self.viewer.layers:
            try:
                if layer.metadata.get("napari_vipp_kind") == "pinned":
                    layers.append(layer)
            except Exception:
                continue
        return layers

    def _move_layer_to_top(self, layer) -> None:
        layers = self.viewer.layers
        try:
            index = list(layers).index(layer)
            layers.move(index, len(layers))
            return
        except Exception:
            pass
        try:
            layers.remove(layer)
            layers.append(layer)
        except Exception:
            pass

    def _keep_active_pin_on_top(self) -> None:
        for layer in self._active_pinned_layers():
            self._move_layer_to_top(layer)

    def _sync_pin_ui(self) -> None:
        self.graph_view.set_pinned_node(self._active_pinned_node_id)
        can_pin = self._node_can_pin(self._selected_node_id)
        self.pin_button.setVisible(can_pin)
        if not can_pin:
            self.pin_button.setText("Pin selected")
        elif self._selected_node_id == self._active_pinned_node_id:
            self.pin_button.setText("Unpin selected")
        else:
            self.pin_button.setText("Pin selected")

    def _sync_preview_ui(self) -> None:
        previewable = self._node_output_type(self._selected_node_id) != "table"
        self.thumbnail_checkbox.setVisible(previewable)
        self.thumbnail_checkbox.setEnabled(previewable)
        with QSignalBlocker(self.thumbnail_checkbox):
            self.thumbnail_checkbox.setChecked(
                previewable and self._node_preview_enabled(self._selected_node_id)
            )

    def _sync_keep_cached_ui(self) -> None:
        node = self.pipeline.nodes.get(self._selected_node_id)
        available = node is not None
        self.keep_cached_checkbox.setVisible(available)
        self.keep_cached_checkbox.setEnabled(available)
        with QSignalBlocker(self.keep_cached_checkbox):
            self.keep_cached_checkbox.setChecked(
                bool(node.params.get(CACHE_KEEP_NODE_PARAM, False))
                if node is not None
                else False
            )

    def _on_keep_cached_toggled(self, checked: bool) -> None:
        node_id = self._selected_node_id
        node = self.pipeline.nodes.get(node_id)
        if node is None:
            return
        if bool(node.params.get(CACHE_KEEP_NODE_PARAM, False)) == bool(checked):
            return
        self._record_parameter_undo(node_id, CACHE_KEEP_NODE_PARAM)
        node.params[CACHE_KEEP_NODE_PARAM] = bool(checked)
        self._apply_cache_retention()
        self._update_thumbnails()
        state = "kept" if checked else "not forced"
        self.status_label.setText(
            f"Cache retention for '{node.title}' is {state}."
        )

    def _sync_auto_contrast_ui(self) -> None:
        node = self.pipeline.nodes.get(self._selected_node_id)
        self.auto_contrast_group.setVisible(
            node is not None and node.operation_id == "linear_scale_offset"
        )
        self.auto_contrast_button.setEnabled(
            self._active_auto_contrast_run_id is None
        )

    def _node_preview_enabled(self, node_id: str) -> bool:
        if self._node_output_type(node_id) == "table":
            return False
        return node_id not in self._preview_disabled_node_ids

    def _node_can_pin(self, node_id: str) -> bool:
        node = self.pipeline.nodes.get(node_id)
        if node is None:
            return False
        data, _state, output_port = self._node_display_payload(node_id)
        if data is not None:
            return not is_table_data(data) and self._data_kind(
                data,
                node_id,
                output_port,
            ) != "table"
        return node.output_type != "table"

    def _node_output_type(self, node_id: str) -> str:
        node = self.pipeline.nodes.get(node_id)
        ports = self.pipeline.output_ports(node_id)
        output_port = (
            self._split_channel_display_port(node_id)
            if node is not None and node.operation_id == "split_channels"
            else 0
        )
        output_port = int(np.clip(output_port, 0, max(len(ports) - 1, 0)))
        if (
            node is not None
            and ports
            and (
                self.pipeline.operation_spec(node.operation_id).preserves_input_type
                or ports[output_port].output_type == "labels"
            )
        ):
            return ports[output_port].output_type
        data, _state, output_port = self._node_display_payload(node_id)
        if data is not None:
            return self._data_kind(data, node_id, output_port)
        return node.output_type if node is not None else "image"

    def _remove_layer(self, layer) -> None:
        try:
            self.viewer.layers.remove(layer)
        except Exception:
            pass

    def _pinned_layer_name(self, title: str) -> str:
        return f"VIPP Pinned: {title}"

    def _current_step(self):
        if not self._dims_linked():
            self._ensure_vipp_dims_state()
            return self._vipp_current_step
        return self._viewer_current_step()

    def _current_step_nsteps(self):
        if not self._dims_linked():
            self._ensure_vipp_dims_state()
            return self._vipp_current_nsteps
        return self._viewer_current_nsteps()

    def _dims_linked(self) -> bool:
        checkbox = getattr(self, "follow_dims_checkbox", None)
        return checkbox is None or bool(checkbox.isChecked())

    def _capture_vipp_dims_from_viewer(self) -> None:
        nsteps = self._viewer_current_nsteps()
        self._vipp_current_nsteps = (
            tuple(max(int(value), 1) for value in nsteps)
            if nsteps is not None
            else None
        )
        self._vipp_current_step = self._normalized_vipp_current_step(
            self._viewer_current_step(),
            self._vipp_current_nsteps,
        )

    def _ensure_vipp_dims_state(self) -> None:
        nsteps = self._viewer_current_nsteps()
        if nsteps is not None:
            self._vipp_current_nsteps = tuple(max(int(value), 1) for value in nsteps)
        if self._vipp_current_step is None:
            self._vipp_current_step = self._viewer_current_step()
        self._vipp_current_step = self._normalized_vipp_current_step(
            self._vipp_current_step,
            self._vipp_current_nsteps,
        )

    def _normalized_vipp_current_step(
        self,
        step,
        nsteps: tuple[int, ...] | None,
    ) -> tuple[int, ...] | None:
        if step is None:
            return None
        values = [int(value) for value in tuple(step)]
        if nsteps is None:
            return tuple(values)
        while len(values) < len(nsteps):
            values.append(0)
        values = values[: len(nsteps)]
        return tuple(
            int(np.clip(value, 0, max(int(size) - 1, 0)))
            for value, size in zip(values, nsteps, strict=True)
        )

    def _viewer_current_step(self):
        raw_step = self._raw_current_step()
        if raw_step is None:
            return None
        return self._canonical_viewer_values(raw_step, fill_value=0)

    def _viewer_current_nsteps(self):
        raw_nsteps = self._viewer_nsteps()
        if raw_nsteps is None:
            return None
        return self._canonical_viewer_values(raw_nsteps, fill_value=1)

    def _raw_current_step(self):
        try:
            return tuple(self.viewer.dims.current_step)
        except Exception:
            return None

    def _canonical_viewer_values(
        self,
        values: tuple,
        *,
        fill_value: int,
    ) -> tuple:
        layer = self._layer_by_name(self._inspect_layer_name)
        metadata = getattr(layer, "metadata", {}) if layer is not None else {}
        if not isinstance(metadata, dict):
            return tuple(int(value) for value in values)
        node_id = metadata.get("node_id")
        output_port = int(metadata.get("output_port", 0) or 0)
        state = self._node_output_state(node_id, output_port)
        axes = tuple(getattr(state, "axes", ()))
        if not axes:
            return tuple(int(value) for value in values)
        display_axis_indices = [
            index
            for index, axis in enumerate(axes)
            if not _state_axis_hidden_from_napari_dims(axis, metadata)
        ]
        if not display_axis_indices:
            return tuple(int(value) for value in values)
        offset = max(len(values) - len(display_axis_indices), 0)
        source_axes = [
            int(axis.source_axis)
            for axis in axes
            if getattr(axis, "source_axis", None) is not None
        ]
        if not source_axes:
            return tuple(int(value) for value in values)
        canonical = [int(fill_value)] * (max(source_axes) + 1)
        for display_position, state_axis_index in enumerate(display_axis_indices):
            current_index = offset + display_position
            if current_index < 0 or current_index >= len(values):
                continue
            axis = axes[state_axis_index]
            source_axis = getattr(axis, "source_axis", None)
            if source_axis is None:
                continue
            source_axis = int(source_axis)
            if 0 <= source_axis < len(canonical):
                canonical[source_axis] = int(values[current_index])
        return tuple(canonical)

    def _node_title(self, node_id: str) -> str:
        return self.pipeline.nodes[node_id].title

    def _node_state_dict(
        self,
        node_id: str,
        output_port: int | None = None,
        *,
        state=None,
    ) -> dict | None:
        if state is None:
            if output_port is None:
                _data, state, output_port = self._node_display_payload(node_id)
            else:
                state = self._node_output_state(node_id, output_port)
        return state.to_dict() if state is not None else None

    def _is_vipp_generated_layer(self, layer) -> bool:
        try:
            return str(layer.metadata.get("napari_vipp_kind", "")) in {
                "inspect",
                "pinned",
            }
        except Exception:
            return False


def _finite_values(arr: np.ndarray) -> np.ndarray:
    if arr.ndim >= 3 and arr.shape[-1] in (3, 4):
        arr = (
            arr[..., 0].astype(np.float32) * 0.299
            + arr[..., 1].astype(np.float32) * 0.587
            + arr[..., 2].astype(np.float32) * 0.114
        )
    return arr[np.isfinite(arr)]


def _positive_scale_float(value, default: float) -> float:
    try:
        result = float(value)
    except Exception:
        return default
    if result <= 0 or not np.isfinite(result):
        return default
    return result


def _progress_detail_message(title: str, message) -> str:
    detail = str(message).strip()
    if not detail:
        return ""
    if _normalized_progress_text(detail) == _normalized_progress_text(title):
        return ""
    return detail


def _normalized_progress_text(value: str) -> str:
    return " ".join(str(value).strip().casefold().split())


def _auto_contrast_scale_offset(
    data,
    saturation_percent: float,
) -> tuple[float, float, float, float] | None:
    if data is None:
        return None

    saturation = min(max(float(saturation_percent), 0.0), 100.0)
    tail_percent = saturation / 2.0
    reference = np.asarray(data)
    if reference.ndim >= 3 and reference.shape[-1] in (3, 4):
        # Auto contrast applies one scale/offset pair to an RGB image, so retain
        # its established luminance reference and never let an RGBA alpha plane
        # distort the intensity range.
        reference = (
            reference[..., 0].astype(np.float32) * 0.299
            + reference[..., 1].astype(np.float32) * 0.587
            + reference[..., 2].astype(np.float32) * 0.114
        )
    percentiles = _exact_finite_percentiles(
        reference,
        (tail_percent, 100.0 - tail_percent),
    )
    if percentiles is None:
        return None
    lower, upper = percentiles
    lower = float(lower)
    upper = float(upper)
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        return None

    alpha = 255.0 / (upper - lower)
    beta = -lower * alpha
    if not np.isfinite(alpha) or not np.isfinite(beta):
        return None
    return float(alpha), float(beta), lower, upper


def _should_background_auto_contrast(data) -> bool:
    """Keep exact percentile selection for large arrays off the GUI thread."""
    try:
        element_count = int(getattr(data, "size", 0))
    except (TypeError, ValueError, OverflowError):
        element_count = 0
    return (
        element_count > AUTO_CONTRAST_BACKGROUND_MIN_ELEMENTS
        or _should_auto_background_data(data)
    )


def _safe_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _rescale_dtype_output_range(
    dtype: np.dtype,
) -> tuple[float, float, float, int]:
    dtype = np.dtype(dtype)
    if dtype == np.dtype(bool):
        return 0.0, 1.0, 1.0, 0
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        if info.min < -(2**53) or info.max > 2**53:
            # QDoubleSpinBox cannot distinguish adjacent int64/uint64 levels.
            # Keep an exact, safe normalized default instead of installing a
            # rounded value beyond the native dtype range.
            return 0.0, 1.0, 1.0, 0
        return 0.0, float(info.max), 1.0, 0
    if np.issubdtype(dtype, np.floating):
        return 0.0, 1.0, 0.01, 3
    return 0.0, 1.0, 0.01, 3


def _ranges_close(
    first: tuple[float, float],
    second: tuple[float, float],
) -> bool:
    return bool(
        np.isclose(first[0], second[0], rtol=0.0, atol=1e-9)
        and np.isclose(first[1], second[1], rtol=0.0, atol=1e-9)
    )


def _slider_safe_bounds(
    minimum: float,
    maximum: float,
    step: float | int,
    decimals: int,
    expandable: bool = False,
    logarithmic: bool = False,
    entry_minimum: float | int | None = None,
    entry_maximum: float | int | None = None,
) -> ParameterBounds:
    maximum_slider_units = 1_000_000_000
    decimals = int(decimals)
    extent = max(abs(float(minimum)), abs(float(maximum)), 1.0)
    while decimals > 0 and extent * (10**decimals) > maximum_slider_units:
        decimals -= 1
    if extent > maximum_slider_units:
        minimum = max(float(minimum), -maximum_slider_units)
        maximum = min(float(maximum), maximum_slider_units)

    smallest_step = 1.0 if decimals == 0 else 10 ** (-decimals)
    return ParameterBounds(
        float(minimum),
        float(maximum),
        max(float(step), smallest_step),
        decimals,
        expandable,
        logarithmic,
        entry_minimum,
        entry_maximum,
    )


def _expanded_bounds(value: float) -> tuple[float, float]:
    padding = abs(value) * 0.1 or 1.0
    return value - padding, value + padding


def _xy_shape(arr: np.ndarray) -> tuple[int, int]:
    if arr.ndim < 2:
        return 1, 1
    if arr.ndim >= 3 and arr.shape[-1] in (3, 4):
        return int(arr.shape[-3]), int(arr.shape[-2])
    return int(arr.shape[-2]), int(arr.shape[-1])


def _histogram_counts(
    data,
    state=None,
    scope: str = "Slice",
    current_step=None,
    current_step_nsteps=None,
) -> np.ndarray | None:
    counts, _x_range, _colors = _histogram_summary(
        data,
        state=state,
        scope=scope,
        current_step=current_step,
        current_step_nsteps=current_step_nsteps,
    )
    return counts


def _histogram_summary(
    data,
    state=None,
    scope: str = "Slice",
    current_step=None,
    current_step_nsteps=None,
) -> tuple[np.ndarray | None, tuple[float, float] | None, list[QColor] | None]:
    if data is None:
        return None, None, None

    source = _histogram_source(
        data,
        state=state,
        scope=scope,
        current_step=current_step,
        current_step_nsteps=current_step_nsteps,
    )
    if source is None:
        return None, None, None

    arr, channel_axis, channel_axis_name = source
    if arr.size == 0:
        return None, None, None
    if channel_axis is not None:
        counts, x_range = _multichannel_histogram(arr, channel_axis)
        if counts is None:
            return None, None, None
        colors = _histogram_series_colors(counts.shape[0], channel_axis_name)
        return counts, x_range, colors
    counts, x_range = _single_histogram(arr)
    return counts, x_range, _histogram_series_colors(1)


def _histogram_has_stack_scope(data, state=None) -> bool:
    if data is None:
        return False
    arr = np.asarray(data)
    if arr.ndim <= 2:
        return False
    axes = tuple(getattr(state, "axes", ()))
    if len(axes) == arr.ndim:
        for axis, size in zip(axes, arr.shape, strict=False):
            name = str(axis.name).lower()
            if int(size) <= 1:
                continue
            if name in {"x", "y", "rgb"} or axis.type == "channel":
                continue
            return True
        return False
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        return False
    return True


def _histogram_state_signature(state) -> tuple:
    axes = tuple(getattr(state, "axes", ()))
    return tuple(
        (
            str(getattr(axis, "name", "")).casefold(),
            str(getattr(axis, "type", "")).casefold(),
            getattr(axis, "source_axis", None),
        )
        for axis in axes
    )


def _histogram_slice_signature(
    data,
    state,
    current_step,
    current_step_nsteps,
) -> tuple:
    arr = np.asarray(data)
    axes = tuple(getattr(state, "axes", ()))
    if len(axes) == arr.ndim:
        y_axis = _metadata_axis_index_by_name(state, "y")
        x_axis = _metadata_axis_index_by_name(state, "x")
        if y_axis is not None and x_axis is not None:
            channel_axis, _channel_name = _histogram_channel_axis(arr, state)
            keep_axes = {y_axis, x_axis, channel_axis}
            return tuple(
                (
                    axis_index,
                    _histogram_axis_index(
                        _metadata_current_step_axis(
                            state,
                            axis_index,
                            current_step,
                        ),
                        int(arr.shape[axis_index]),
                        current_step,
                        current_step_nsteps=current_step_nsteps,
                    ),
                )
                for axis_index in range(arr.ndim)
                if axis_index not in keep_axes
            )
    return (
        tuple(current_step or ()),
        tuple(current_step_nsteps or ()),
    )


def _input_histogram_marker_key(operation_id: str, params: dict | None) -> tuple:
    values = params or {}

    def selected(*names: str) -> tuple:
        return tuple((name, repr(values.get(name))) for name in names)

    if operation_id == "rescale_intensity":
        mode = str(
            values.get(RESCALE_CUTOFF_MODE_PARAMETER, "Percentiles")
        ).casefold()
        names = (
            ("in_low_value", "in_high_value")
            if mode == "values"
            else ("in_low_percentile", "in_high_percentile")
        )
        return (mode, selected(*names))
    if operation_id == "clip_intensity":
        mode = str(values.get("cutoff_mode", "Data range")).casefold()
        names = ("minimum", "maximum") if mode == "values" else ()
        return (mode, selected(*names))
    if operation_id == "binary_threshold":
        return selected("threshold")
    if operation_id == "hysteresis_threshold":
        return selected("low_threshold", "high_threshold")
    if operation_id in GLOBAL_THRESHOLD_OPERATIONS:
        names = ["histogram_bins"] if "histogram_bins" in values else []
        if operation_id == "minimum_threshold":
            names.append("max_iterations")
        return selected(*names)
    return ()


def _input_histogram_markers(
    operation_id: str,
    data,
    *,
    state=None,
    scope: str = "Slice",
    current_step=None,
    current_step_nsteps=None,
    params: dict | None = None,
    progress=None,
) -> list[tuple[str, float, QColor]]:
    if operation_id == "rescale_intensity":
        return _rescale_percentile_markers(
            data,
            state=state,
            scope=scope,
            current_step=current_step,
            current_step_nsteps=current_step_nsteps,
            params=params,
        )
    if operation_id == "clip_intensity":
        mode = str((params or {}).get("cutoff_mode", "Data range")).casefold()
        if mode == "data range":
            # Data-range mode describes the complete connected input. The
            # inspector scope changes only the displayed histogram counts.
            stats = _exact_finite_stats(data)
            if stats.count == 0:
                return []
            low, high = stats.minimum, stats.maximum
        elif mode == "values":
            low = _finite_marker_value((params or {}).get("minimum"), "Clip minimum")
            high = _finite_marker_value(
                (params or {}).get("maximum"),
                "Clip maximum",
            )
        else:
            raise ValueError("Clip input cutoffs must be 'Data range' or 'Values'.")
        if low > high:
            raise ValueError("Clip minimum must not exceed the maximum.")
        markers = [("min", low, QColor("#f59e0b"))]
        if low != high:
            markers.append(("max", high, QColor("#38bdf8")))
        return markers
    if operation_id == "binary_threshold":
        threshold = _safe_float((params or {}).get("threshold"), 0.0)
        return [("threshold", float(threshold), QColor("#f59e0b"))]
    if operation_id == "hysteresis_threshold":
        low = _safe_float((params or {}).get("low_threshold"), 0.0)
        high = _safe_float((params or {}).get("high_threshold"), low)
        if low > high:
            low, high = high, low
        markers = [("low", float(low), QColor("#f59e0b"))]
        if not np.isclose(low, high):
            markers.append(("high", float(high), QColor("#38bdf8")))
        return markers
    if operation_id in GLOBAL_THRESHOLD_OPERATIONS:
        source = _histogram_source(
            data,
            state=state,
            scope=scope,
            current_step=current_step,
            current_step_nsteps=current_step_nsteps,
        )
        if source is None:
            return []
        value = automatic_threshold_value(
            source[0],
            operation_id,
            histogram_bins=(params or {}).get(
                "histogram_bins",
                256,
            ),
            max_iterations=(params or {}).get("max_iterations", 10_000),
            progress=progress,
        )
        if value is None or not np.isfinite(value):
            return []
        marker_value = (
            int(value) if isinstance(value, (int, np.integer)) else float(value)
        )
        return [("threshold", marker_value, QColor("#f59e0b"))]
    return []


def _input_histogram_draggable_markers(
    operation_id: str,
    params: dict | None = None,
) -> set[str]:
    if operation_id == "rescale_intensity":
        if str(
            (params or {}).get(RESCALE_CUTOFF_MODE_PARAMETER, "Percentiles")
        ).lower().startswith("value"):
            return {"low", "high"}
        return set()
    if operation_id == "clip_intensity":
        if str((params or {}).get("cutoff_mode", "Data range")).lower().startswith(
            "value"
        ):
            return {"min", "max"}
        return set()
    if operation_id == "binary_threshold":
        return {"threshold"}
    if operation_id == "hysteresis_threshold":
        return {"low", "high"}
    return set()


def _input_histogram_marker_parameter(operation_id: str, label: str) -> str | None:
    label = str(label)
    if operation_id == "rescale_intensity":
        return {"low": "in_low_value", "high": "in_high_value"}.get(label)
    if operation_id == "clip_intensity":
        return {"min": "minimum", "max": "maximum"}.get(label)
    if operation_id == "binary_threshold":
        return "threshold" if label == "threshold" else None
    if operation_id == "hysteresis_threshold":
        return {"low": "low_threshold", "high": "high_threshold"}.get(label)
    return None


def _threshold_histogram_scope_label(scope: str) -> str:
    return (
        "Stack histogram"
        if str(scope).strip().lower().startswith("stack")
        else "Slice histogram"
    )


def _rescale_percentile_markers(
    data,
    *,
    state=None,
    scope: str = "Slice",
    current_step=None,
    current_step_nsteps=None,
    params: dict | None = None,
) -> list[tuple[str, float, QColor]]:
    mode = str(
        (params or {}).get(RESCALE_CUTOFF_MODE_PARAMETER, "Percentiles")
    ).casefold()
    if mode == "values":
        low, high = _rescale_value_pair(params or {})
        markers = [("low", low, QColor("#f59e0b"))]
        if low != high:
            markers.append(("high", high, QColor("#22c55e")))
        return markers
    if mode != "percentiles":
        raise ValueError(
            "Rescale Intensity input cutoffs must be 'Percentiles' or 'Values'."
        )

    low_p, high_p = _rescale_percentile_pair(params or {})
    # Rescale Intensity applies percentiles to the complete connected input.
    # The inspector scope changes only the displayed histogram counts.
    cutoffs = _exact_finite_percentiles(data, (low_p, high_p))
    if cutoffs is None:
        return []
    low, high = cutoffs
    markers = [("low", low, QColor("#f59e0b"))]
    if low != high:
        markers.append(("high", high, QColor("#22c55e")))
    return markers


def _iter_finite_numeric_chunks(data):
    """Yield finite numeric values without a full-size temporary mask."""
    if data is None:
        return
    try:
        iterator = np.nditer(
            np.asarray(data),
            flags=["buffered", "external_loop", "refs_ok", "zerosize_ok"],
            op_flags=[["readonly"]],
            order="K",
            buffersize=INSPECTOR_STATISTICS_CHUNK_ELEMENTS,
        )
    except (TypeError, ValueError):
        return
    for chunk in iterator:
        values = np.asarray(chunk)
        try:
            finite = np.isfinite(values)
        except TypeError:
            return
        if not finite.all():
            values = values[finite]
        yield values


def _exact_finite_stats(data) -> ExactFiniteStats:
    """Return exact finite count and extrema using bounded temporaries."""
    arr = np.asarray(data)
    integer_data = np.issubdtype(arr.dtype, np.integer)
    count = 0
    minimum: int | float | None = None
    maximum: int | float | None = None
    for values in _iter_finite_numeric_chunks(arr):
        if values.size == 0:
            continue
        count += int(values.size)
        chunk_minimum = (
            int(values.min()) if integer_data else float(values.min())
        )
        chunk_maximum = (
            int(values.max()) if integer_data else float(values.max())
        )
        minimum = chunk_minimum if minimum is None else min(minimum, chunk_minimum)
        maximum = chunk_maximum if maximum is None else max(maximum, chunk_maximum)
    if count == 0:
        return ExactFiniteStats(0, 0.0, 0.0)
    assert minimum is not None and maximum is not None
    return ExactFiniteStats(count, minimum, maximum)


def _exact_generated_layer_contrast_limits(
    data,
) -> tuple[float, float] | None:
    """Return display limits containing every finite value and zero.

    Zero remains in the window because zero-valued background is common in
    bioimages. Unlike the old signed-image path, negative results are not
    clipped and no percentile approximation is used.
    """
    stats = _exact_finite_stats(data)
    if stats.count == 0:
        return None
    low = min(float(stats.minimum), 0.0)
    high = max(float(stats.maximum), 0.0)
    if low == high:
        if low == 0.0:
            return (0.0, 1.0)
        low, high = _expanded_bounds(low)
    return (float(low), float(high))


def _provisional_generated_layer_contrast_limits(
    data,
) -> tuple[float, float]:
    """Return scan-free temporary limits while exact extrema are calculated."""
    arr = np.asarray(data)
    dtype = arr.dtype
    if dtype == np.dtype(bool):
        return (0.0, 1.0)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        low = min(float(info.min), 0.0)
        high = max(float(info.max), 0.0)
        if low < high:
            return (low, high)
    return (0.0, 1.0)


def _exact_finite_percentiles(
    data,
    percentiles: tuple[float, ...],
) -> tuple[int | float | Rational, ...] | None:
    """Calculate exact NumPy percentiles over every finite input value.

    Extrema avoid a full-data copy. Interior percentiles necessarily retain
    the finite values because exact order statistics cannot be recovered from
    fixed display bins.
    """
    requested = tuple(float(np.clip(value, 0.0, 100.0)) for value in percentiles)
    stats = _exact_finite_stats(data)
    if stats.count == 0:
        return None
    if stats.minimum == stats.maximum:
        return tuple(stats.minimum for _value in requested)
    if all(value in {0.0, 100.0} for value in requested):
        return tuple(
            stats.minimum if value == 0.0 else stats.maximum
            for value in requested
        )

    arr = np.asarray(data)
    if np.issubdtype(arr.dtype, np.integer):
        return exact_integer_percentiles(arr, requested)
    values = np.empty(stats.count, dtype=arr.dtype)
    offset = 0
    for chunk in _iter_finite_numeric_chunks(arr):
        if chunk.size == 0:
            continue
        stop = offset + int(chunk.size)
        values[offset:stop] = chunk
        offset = stop
    if offset != stats.count:
        raise RuntimeError("Finite-value count changed during percentile calculation.")
    result = np.percentile(values, requested, overwrite_input=True)
    return tuple(float(value) for value in np.asarray(result).ravel())


def _rescale_percentile_pair(params: dict) -> tuple[float, float]:
    low = _finite_marker_value(params.get("in_low_percentile"), "Low percentile")
    high = _finite_marker_value(
        params.get("in_high_percentile"),
        "High percentile",
    )
    if not 0.0 <= low <= 100.0 or not 0.0 <= high <= 100.0:
        raise ValueError("Rescale percentiles must be between 0 and 100.")
    if low > high:
        raise ValueError("Rescale low percentile must not exceed the high percentile.")
    return low, high


def _rescale_value_pair(params: dict) -> tuple[int | float, int | float]:
    low = _finite_marker_value(params.get("in_low_value"), "Low input value")
    high = _finite_marker_value(params.get("in_high_value"), "High input value")
    if low > high:
        raise ValueError(
            "Rescale low input value must not exceed the high input value."
        )
    return low, high


def _finite_marker_value(value, label: str) -> int | float:
    if isinstance(value, (int, np.integer)) and not isinstance(
        value,
        (bool, np.bool_),
    ):
        return int(value)
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite number.") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"{label} must be a finite number.")
    return parsed


def _rgb_channel_layer_name(base_name: str, channel_index: int) -> str:
    if int(channel_index) == 0:
        return base_name
    channel_name = _RGB_VOLUME_CHANNELS[int(channel_index)][1]
    return f"{base_name} {channel_name}"


def _local_dim_value_from_viewer(
    viewer_value: int,
    *,
    axis_size: int,
    viewer_axis_size: int,
) -> int:
    axis_size = max(int(axis_size), 1)
    viewer_axis_size = max(int(viewer_axis_size), 1)
    if axis_size == viewer_axis_size:
        return int(np.clip(viewer_value, 0, axis_size - 1))
    source_max = max(viewer_axis_size - 1, 0)
    target_max = max(axis_size - 1, 0)
    if source_max <= 0 or target_max <= 0:
        return 0
    ratio = float(np.clip(viewer_value, 0, source_max)) / float(source_max)
    return int(np.clip(round(ratio * target_max), 0, target_max))


def _viewer_dim_value_from_local(
    local_value: int,
    *,
    axis_size: int,
    viewer_axis_size: int,
) -> int:
    axis_size = max(int(axis_size), 1)
    viewer_axis_size = max(int(viewer_axis_size), 1)
    if axis_size == viewer_axis_size:
        return int(np.clip(local_value, 0, viewer_axis_size - 1))
    source_max = max(axis_size - 1, 0)
    target_max = max(viewer_axis_size - 1, 0)
    if source_max <= 0 or target_max <= 0:
        return 0
    ratio = float(np.clip(local_value, 0, source_max)) / float(source_max)
    return int(np.clip(round(ratio * target_max), 0, target_max))


def _histogram_source(
    data,
    *,
    state=None,
    scope: str = "Slice",
    current_step=None,
    current_step_nsteps=None,
) -> tuple[np.ndarray, int | None, str] | None:
    arr = np.asarray(data)
    channel_axis, channel_axis_name = _histogram_channel_axis(arr, state)
    if str(scope).strip().lower().startswith("stack"):
        return arr, channel_axis, channel_axis_name

    if state is not None:
        source = _state_histogram_slice(
            arr,
            state,
            channel_axis,
            current_step=current_step,
            current_step_nsteps=current_step_nsteps,
        )
        if source is not None:
            return source[0], source[1], channel_axis_name

    preview = make_preview(
        data,
        mode="slice",
        current_step=current_step,
        current_step_nsteps=current_step_nsteps,
        state=state,
    )
    if preview is None:
        return None
    arr = np.asarray(preview)
    channel_axis, channel_axis_name = _histogram_channel_axis(arr, None)
    return arr, channel_axis, channel_axis_name


def _state_histogram_slice(
    arr: np.ndarray,
    state,
    channel_axis: int | None,
    *,
    current_step=None,
    current_step_nsteps=None,
) -> tuple[np.ndarray, int | None] | None:
    if len(state.axes) != arr.ndim:
        return None
    y_axis = _metadata_axis_index_by_name(state, "y")
    x_axis = _metadata_axis_index_by_name(state, "x")
    if y_axis is None or x_axis is None:
        return None

    keep_axes = {y_axis, x_axis}
    if channel_axis is not None:
        keep_axes.add(channel_axis)

    result = arr
    remaining = list(range(arr.ndim))
    for original_axis in reversed(range(arr.ndim)):
        if original_axis in keep_axes:
            continue
        local_axis = remaining.index(original_axis)
        step_axis = _metadata_current_step_axis(
            state,
            original_axis,
            current_step,
        )
        index = _histogram_axis_index(
            step_axis,
            result.shape[local_axis],
            current_step,
            current_step_nsteps=current_step_nsteps,
        )
        result = _axis_index_view(result, local_axis, index)
        remaining.pop(local_axis)

    if channel_axis is None or channel_axis not in remaining:
        return result, None
    return result, remaining.index(channel_axis)


def _histogram_channel_axis(arr: np.ndarray, state) -> tuple[int | None, str]:
    if state is not None and len(getattr(state, "axes", ())) == arr.ndim:
        for index, axis in enumerate(state.axes):
            if axis.type == "channel" and arr.shape[index] > 1:
                return index, axis.name.lower()
    if arr.ndim >= 3 and arr.shape[-1] in (3, 4):
        return arr.ndim - 1, "rgb"
    return None, ""


def _metadata_axis_index_by_name(state, name: str) -> int | None:
    for index, axis in enumerate(state.axes):
        if axis.name.lower() == name:
            return index
    return None


def _metadata_current_step_axis(state, axis_index: int, current_step=None) -> int:
    try:
        current_ndim = len(tuple(current_step))
    except Exception:
        current_ndim = 0
    axes = tuple(getattr(state, "axes", ()))
    state_ndim = len(axes)
    if current_ndim == state_ndim and current_ndim > 0:
        # When viewer dims already match this state, map positionally.
        return axis_index
    try:
        source_axis = axes[axis_index].source_axis
    except Exception:
        source_axis = None
    if source_axis is not None and 0 <= int(source_axis) < current_ndim:
        return int(source_axis)
    return axis_index


def _state_axis_hidden_from_napari_dims(axis, metadata: dict) -> bool:
    if not bool(metadata.get("display_rgb")):
        return False
    return str(getattr(axis, "name", "")).lower() in {"rgb", "rgba"}


def _layer_scale_from_metadata(metadata: dict) -> tuple[float, ...] | None:
    if not isinstance(metadata, dict):
        return None
    expected_ndim = _napari_layer_transform_ndim(metadata)
    default_scale = (
        tuple(1.0 for _ in range(expected_ndim)) if expected_ndim > 0 else None
    )
    carried = metadata.get("vipp_image_state")
    if not isinstance(carried, dict):
        return default_scale
    state = ImageState.from_dict(carried)
    if state is None or not state.axes:
        return default_scale
    scales = tuple(
        _positive_scale_float(axis.scale, 1.0)
        for axis in state.axes
        if not _state_axis_hidden_from_napari_dims(axis, metadata)
    )
    if expected_ndim <= 0 or len(scales) != expected_ndim:
        return default_scale
    return scales


def _napari_layer_transform_ndim(metadata: dict) -> int:
    shape = tuple(metadata.get("display_shape", ()))
    if shape:
        ndim = len(shape)
        if (
            bool(metadata.get("display_rgb"))
            and not bool(metadata.get("display_rgb_as_channels"))
            and shape[-1] in (3, 4)
        ):
            return max(ndim - 1, 0)
        return ndim
    try:
        ndim = int(metadata.get("display_ndim", 0))
    except Exception:
        return 0
    if (
        bool(metadata.get("display_rgb"))
        and not bool(metadata.get("display_rgb_as_channels"))
        and ndim > 0
    ):
        return ndim - 1
    return ndim


def _histogram_axis_index(
    axis: int,
    axis_size: int,
    current_step=None,
    *,
    current_step_nsteps=None,
) -> int:
    if current_step is None:
        return axis_size // 2
    try:
        step = int(tuple(current_step)[axis])
    except Exception:
        step = axis_size // 2
    if current_step_nsteps is not None:
        try:
            source_nsteps = int(tuple(current_step_nsteps)[axis])
        except Exception:
            source_nsteps = 0
        # Keep exact napari index when source and target axis sizes match.
        if source_nsteps == int(axis_size):
            return int(np.clip(step, 0, max(axis_size - 1, 0)))
        source_max = max(source_nsteps - 1, 0)
        target_max = max(int(axis_size) - 1, 0)
        if source_max > 0 and target_max > 0:
            ratio = float(np.clip(step, 0, source_max)) / float(source_max)
            step = int(round(ratio * target_max))
    return int(np.clip(step, 0, max(axis_size - 1, 0)))


@contextmanager
def _control_signal_blockers(widgets):
    blockers = [QSignalBlocker(widget) for widget in widgets]
    try:
        yield
    finally:
        del blockers


def _axis_order_item_text(option: AxisSliceOption) -> str:
    return f"{option.title}    {option.axis_type} axis, size {int(option.size)}"


def _axis_order_value(ordered: list[AxisSliceOption]) -> str:
    names = [str(option.name).strip() for option in ordered]
    normalized = [name.lower() for name in names]
    if all(len(name) == 1 for name in names) and len(set(normalized)) == len(
        normalized
    ):
        return "".join(name.upper() for name in names)
    if all(name and not any(char in name for char in ",; ") for name in names) and len(
        set(normalized)
    ) == len(normalized):
        return ",".join(names)
    return ",".join(str(option.index) for option in ordered)


def _axis_options_from_order(
    order,
    options: list[AxisSliceOption],
) -> list[AxisSliceOption]:
    if not options:
        return []
    tokens = _axis_order_tokens(order)
    if not tokens:
        return list(options)
    if len(tokens) != len(options):
        return list(options)

    by_index = {option.index: option for option in options}
    numeric = _numeric_axis_order(tokens, set(by_index))
    if numeric is not None:
        return [by_index[index] for index in numeric]

    named = _named_axis_order(tokens, options)
    if named is not None:
        return named
    return list(options)


def _axis_order_tokens(order) -> list[str]:
    if order is None:
        return []
    if isinstance(order, (list, tuple)):
        return [str(part).strip() for part in order if str(part).strip()]
    text = str(order).strip()
    if not text:
        return []
    if any(separator in text for separator in (",", ";", " ")):
        normalized = text.replace(";", ",").replace(" ", ",")
        return [part.strip() for part in normalized.split(",") if part.strip()]
    return list(text)


def _numeric_axis_order(
    tokens: list[str],
    valid_indices: set[int],
) -> list[int] | None:
    indices: list[int] = []
    try:
        for token in tokens:
            indices.append(int(token))
    except ValueError:
        return None
    if set(indices) != valid_indices or len(set(indices)) != len(indices):
        return None
    return indices


def _named_axis_order(
    tokens: list[str],
    options: list[AxisSliceOption],
) -> list[AxisSliceOption] | None:
    used: set[int] = set()
    ordered: list[AxisSliceOption] = []
    for token in tokens:
        target = str(token).strip().lower()
        matches = [
            option
            for option in options
            if option.name.lower() == target and option.index not in used
        ]
        if not matches:
            return None
        option = matches[0]
        ordered.append(option)
        used.add(option.index)
    return ordered


def _axis_slice_state_from_value(
    value: dict,
    options: list[AxisSliceOption],
) -> tuple[dict[int, tuple[int, int]], dict[int, int]]:
    option_sizes = {option.index: option.size for option in options}
    ranges = _parse_axis_ranges(value.get("ranges"), option_sizes)
    removals = _axis_indices_from_value(
        value.get("remove_axes"),
        value.get("remove_indices"),
        option_sizes,
    )
    if removals:
        return ranges, removals
    if value.get("range_mode", True):
        return ranges, {}
    removals = _axis_indices_from_value(
        value.get("axes"),
        value.get("indices"),
        option_sizes,
    )
    if removals:
        return {}, removals
    return ranges, _axis_indices_from_value(
        value.get("axis"),
        value.get("index"),
        option_sizes,
    )


def _axis_indices_from_value(
    axes_value,
    indices_value,
    option_sizes: dict[int, int],
) -> dict[int, int]:
    axes = _parse_int_list(axes_value)
    indices = _parse_int_list(indices_value)
    if not axes:
        return {}
    return {
        int(axis): _clamped_index(
            indices[position] if position < len(indices) else 0,
            option_sizes.get(int(axis), 1),
        )
        for position, axis in enumerate(axes)
    }


def _parse_axis_ranges(
    value,
    option_sizes: dict[int, int],
) -> dict[int, tuple[int, int]]:
    if not isinstance(value, str) or not value.strip():
        return {}
    ranges: dict[int, tuple[int, int]] = {}
    for part in value.split(";"):
        pieces = [piece.strip() for piece in part.split(":")]
        if len(pieces) != 3:
            continue
        try:
            axis = int(pieces[0])
            start = int(pieces[1])
            end = int(pieces[2])
        except ValueError:
            continue
        ranges[axis] = _clamped_range(start, end, option_sizes.get(axis, 1))
    return ranges


def _clamped_range(start: int, end: int, size: int) -> tuple[int, int]:
    maximum = max(int(size) - 1, 0)
    start = int(np.clip(start, 0, maximum))
    end = int(np.clip(end, 0, maximum))
    if start > end:
        start, end = end, start
    return start, end


def _clamped_index(index: int, size: int) -> int:
    maximum = max(int(size) - 1, 0)
    return int(np.clip(index, 0, maximum))


def _axis_index_view(arr: np.ndarray, axis: int, index: int) -> np.ndarray:
    selection = [slice(None)] * arr.ndim
    selection[int(axis)] = int(index)
    return arr[tuple(selection)]


def _parse_int_list(value) -> list[int]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, (list, tuple, set)):
        parts = list(value)
    else:
        parts = [value]
    parsed = []
    for part in parts:
        try:
            parsed.append(int(part))
        except (TypeError, ValueError):
            continue
    return parsed


def _single_histogram(
    arr: np.ndarray,
) -> tuple[np.ndarray | None, tuple[int | float, int | float] | None]:
    if arr.dtype == bool:
        true_count = int(np.count_nonzero(arr))
        return (
            np.array([arr.size - true_count, true_count], dtype=np.int64),
            (0.0, 1.0),
        )

    stats = _exact_finite_stats(arr)
    if stats.count == 0:
        return None, None
    if stats.minimum == stats.maximum:
        return np.array([stats.count], dtype=np.int64), (
            stats.minimum,
            stats.maximum,
        )
    if np.issubdtype(arr.dtype, np.integer):
        return _exact_integer_display_histogram(arr, stats)
    edges, x_range = _display_histogram_edges(stats)
    return _exact_histogram_counts(arr, edges), x_range


def _multichannel_histogram(
    arr: np.ndarray,
    channel_axis: int,
) -> tuple[np.ndarray | None, tuple[int | float, int | float] | None]:
    channel_axis = int(np.clip(channel_axis, 0, arr.ndim - 1))
    channels = [
        _axis_index_view(arr, channel_axis, channel)
        for channel in range(arr.shape[channel_axis])
    ]
    channel_stats = [_exact_finite_stats(values) for values in channels]
    valid_stats = [stats for stats in channel_stats if stats.count]
    if not valid_stats:
        return None, None

    if arr.dtype == bool:
        counts = []
        for values in channels:
            true_count = int(np.count_nonzero(values))
            counts.append(
                np.array(
                    [values.size - true_count, true_count],
                    dtype=np.int64,
                )
            )
        return np.vstack(counts), (0.0, 1.0)

    combined = ExactFiniteStats(
        sum(stats.count for stats in valid_stats),
        min(stats.minimum for stats in valid_stats),
        max(stats.maximum for stats in valid_stats),
    )
    if combined.minimum == combined.maximum:
        counts = np.array(
            [[stats.count] for stats in channel_stats],
            dtype=np.int64,
        )
        return counts, (combined.minimum, combined.maximum)
    if np.issubdtype(arr.dtype, np.integer):
        first_counts, x_range = _exact_integer_display_histogram(
            channels[0],
            combined,
        )
        histogram_minimum, histogram_maximum, bin_count = (
            _integer_display_histogram_configuration(arr.dtype, combined)
        )
        counts = [first_counts]
        counts.extend(
            _exact_integer_display_histogram_counts(
                values,
                histogram_minimum=histogram_minimum,
                histogram_maximum=histogram_maximum,
                bin_count=bin_count,
            )
            for values in channels[1:]
        )
        return np.vstack(counts), x_range
    edges, x_range = _display_histogram_edges(combined)
    counts = [
        (
            _exact_histogram_counts(values, edges)
            if stats.count
            else np.zeros(edges.size - 1, dtype=np.int64)
        )
        for values, stats in zip(channels, channel_stats, strict=True)
    ]
    return np.vstack(counts), x_range


def _exact_integer_display_histogram(
    data,
    stats: ExactFiniteStats,
) -> tuple[np.ndarray, tuple[int, int]]:
    histogram_minimum, histogram_maximum, bin_count = (
        _integer_display_histogram_configuration(np.asarray(data).dtype, stats)
    )
    counts = _exact_integer_display_histogram_counts(
        data,
        histogram_minimum=histogram_minimum,
        histogram_maximum=histogram_maximum,
        bin_count=bin_count,
    )
    return counts, (histogram_minimum, histogram_maximum)


def _integer_display_histogram_configuration(
    dtype: np.dtype,
    stats: ExactFiniteStats,
) -> tuple[int, int, int]:
    minimum = int(stats.minimum)
    maximum = int(stats.maximum)
    if 0 <= minimum and maximum <= 255:
        return 0, 255, 256
    level_span = maximum - minimum + 1
    return minimum, maximum, min(INSPECTOR_DISPLAY_HISTOGRAM_BINS, level_span)


def _exact_integer_display_histogram_counts(
    data,
    *,
    histogram_minimum: int,
    histogram_maximum: int,
    bin_count: int,
) -> np.ndarray:
    """Count integer levels exactly after subtracting a Python-int offset."""
    level_span = histogram_maximum - histogram_minimum + 1
    counts = np.zeros(bin_count, dtype=np.int64)
    if level_span <= 65_536:
        native_counts = np.zeros(level_span, dtype=np.int64)
        for values in _iter_finite_numeric_chunks(data):
            if values.size == 0:
                continue
            levels, level_counts = np.unique(values, return_counts=True)
            indices = np.fromiter(
                (int(level) - histogram_minimum for level in levels),
                dtype=np.intp,
                count=levels.size,
            )
            native_counts[indices] += level_counts.astype(np.int64, copy=False)
        if bin_count == level_span:
            return native_counts
        boundaries = (
            np.arange(bin_count + 1, dtype=np.int64) * level_span // bin_count
        )
        return np.asarray(
            [
                native_counts[start:stop].sum(dtype=np.int64)
                for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True)
            ],
            dtype=np.int64,
        )

    for values in _iter_finite_numeric_chunks(data):
        if values.size == 0:
            continue
        levels, level_counts = np.unique(values, return_counts=True)
        indices = np.fromiter(
            (
                min(
                    ((int(level) - histogram_minimum) * bin_count) // level_span,
                    bin_count - 1,
                )
                for level in levels
            ),
            dtype=np.intp,
            count=levels.size,
        )
        np.add.at(counts, indices, level_counts.astype(np.int64, copy=False))
    return counts


def _display_histogram_edges(
    stats: ExactFiniteStats,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Return floating-point display edges after dtype-specific handling."""
    return (
        np.linspace(
            stats.minimum,
            stats.maximum,
            INSPECTOR_DISPLAY_HISTOGRAM_BINS + 1,
        ),
        (stats.minimum, stats.maximum),
    )


def _exact_histogram_counts(data, edges: np.ndarray) -> np.ndarray:
    counts = np.zeros(int(edges.size) - 1, dtype=np.int64)
    for values in _iter_finite_numeric_chunks(data):
        if values.size:
            counts += np.histogram(values, bins=edges)[0]
    return counts


def _should_auto_background_data(data) -> bool:
    """Return whether image-sized work should leave the GUI thread."""
    if data is None or is_table_data(data):
        return False
    size = getattr(data, "size", None)
    if size is None:
        shape = getattr(data, "shape", None)
        if shape is not None:
            try:
                size = int(np.prod(tuple(shape), dtype=np.int64))
            except (TypeError, ValueError):
                size = None
    try:
        element_count = int(size) if size is not None else 0
    except (TypeError, ValueError, OverflowError):
        element_count = 0
    if element_count >= AUTO_BACKGROUND_MIN_ELEMENTS:
        return True

    nbytes = getattr(data, "nbytes", None)
    if nbytes is None and element_count:
        try:
            nbytes = element_count * int(np.dtype(data.dtype).itemsize)
        except (AttributeError, TypeError, ValueError):
            nbytes = 0
    try:
        return int(nbytes or 0) >= AUTO_BACKGROUND_MIN_BYTES
    except (TypeError, ValueError, OverflowError):
        return False


def _histogram_series_colors(count: int, channel_axis_name: str = "") -> list[QColor]:
    if count <= 0:
        return []
    if channel_axis_name == "rgb":
        base = [QColor("#ef4444"), QColor("#22c55e"), QColor("#60a5fa")]
    elif count > 1:
        base = [_qcolor_from_unit_rgb(color) for color in FLUORESCENCE_COLORS]
    else:
        base = [QColor("#60a5fa")]
    return [QColor(base[index % len(base)]) for index in range(count)]


def _qcolor_from_unit_rgb(color: np.ndarray) -> QColor:
    return QColor.fromRgbF(
        float(np.clip(color[0], 0, 1)),
        float(np.clip(color[1], 0, 1)),
        float(np.clip(color[2], 0, 1)),
    )


def _pipeline_cache_nbytes(pipeline: PrototypePipeline) -> int:
    seen: set[int] = set()
    total = 0
    for value in pipeline.outputs.values():
        total += _object_nbytes(value, seen)
    for values in pipeline.node_outputs.values():
        for value in values:
            total += _object_nbytes(value, seen)
    return int(total)


def _object_nbytes(value, seen: set[int]) -> int:
    if value is None:
        return 0
    value_id = id(value)
    if value_id in seen:
        return 0
    seen.add(value_id)

    nbytes = getattr(value, "nbytes", None)
    if nbytes is not None:
        try:
            return max(int(nbytes), 0)
        except (OverflowError, TypeError, ValueError):
            pass
    if is_table_data(value):
        return _table_nbytes(value)
    if isinstance(value, dict):
        return sum(
            _object_nbytes(item, seen)
            for pair in value.items()
            for item in pair
        )
    if isinstance(value, (list, tuple)):
        return sum(_object_nbytes(item, seen) for item in value)
    return 0


def _table_nbytes(table) -> int:
    total = 0
    for column in table.columns:
        total += len(str(column).encode("utf-8"))
    for row in table.rows:
        for item in row:
            total += len(str(item).encode("utf-8"))
    return total


def _format_byte_count(size: int | float | None) -> str:
    if size is None:
        return "n/a"
    value = max(float(size), 0.0)
    units = ("B", "KB", "MB", "GB", "TB")
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"


def _system_memory_bytes() -> tuple[int | None, int | None]:
    if os.name == "nt":
        return _windows_memory_bytes()
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        total_pages = int(os.sysconf("SC_PHYS_PAGES"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None, None
    return available_pages * page_size, total_pages * page_size


def _windows_memory_bytes() -> tuple[int | None, int | None]:
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    try:
        ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    except (AttributeError, OSError):
        return None, None
    if not ok:
        return None, None
    return int(status.ullAvailPhys), int(status.ullTotalPhys)


def _format_histogram_label(value: int | float | Rational) -> str:
    if isinstance(value, Rational):
        if value.denominator == 1:
            return str(int(value))
        whole = value.numerator // value.denominator
        remainder = value - whole
        return f"{whole} + {remainder.numerator}/{remainder.denominator}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if not np.isfinite(value):
        return ""
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.4g}"


def _normalize_search_text(value) -> str:
    return "".join(
        character.lower() if character.isalnum() else " "
        for character in str(value or "")
    ).strip()


def _fuzzy_match(query: str, haystack: str) -> bool:
    tokens = query.split()
    if not tokens:
        return True
    return all(_fuzzy_token_match(token, haystack) for token in tokens)


def _fuzzy_token_match(token: str, haystack: str) -> bool:
    if token in haystack:
        return True
    position = 0
    for character in token:
        position = haystack.find(character, position)
        if position < 0:
            return False
        position += 1
    return True
