"""napari dock widget for the VIPP workflow prototype."""

from __future__ import annotations

import ctypes
import html
import inspect as py_inspect
import os
import re
import sys
import textwrap
import threading
import weakref
from collections.abc import Iterable
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtCore import (
    QEvent,
    QEventLoop,
    QPointF,
    QRect,
    QSignalBlocker,
    QSize,
    Qt,
    QThreadPool,
    QTimer,
)
from qtpy.QtGui import (
    QAction,
    QColor,
    QFont,
    QIcon,
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
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
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
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from napari_vipp import __version__ as VIPP_VERSION
from napari_vipp._graph import (
    STALE_EXECUTION_ACCENT,
    PipelineGraphView,
    PortLabelMode,
)
from napari_vipp._sample_data import make_sample_data
from napari_vipp.core.batch import (
    BATCH_CONFIG_FILENAME,
    BATCH_SCRIPT_FILENAME,
    BATCH_WORKFLOW_FILENAME,
    BatchConfig,
    BatchItemPlan,
    BatchRunResult,
    ExistingFilePolicy,
    atomic_write_json,
    atomic_write_text,
    load_batch_config,
    plan_batch,
    preflight_batch,
    run_batch,
    save_batch_config,
    scientific_workflow_hash,
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
    channel_color_labels_from_metadata,
    color_value_to_rgb,
)
from napari_vipp.core.diagnostics import (
    PSF_EDGE_MASS_WARNING_FRACTION,
    PsfPreflightResult,
    WidefieldNyquistResult,
    exact_histogram,
    label_volumes,
    largest_label_volume,
    largest_object_size,
    provisional_generated_layer_contrast_limits,
    psf_preflight,
    widefield_nyquist_sampling,
)
from napari_vipp.core.diagnostics import (
    exact_finite_percentiles as _exact_finite_percentiles,
)
from napari_vipp.core.diagnostics import (
    exact_finite_stats as _exact_finite_stats,
)
from napari_vipp.core.diagnostics import (
    exact_generated_layer_contrast_limits as _exact_generated_layer_contrast_limits,
)
from napari_vipp.core.execution import (
    PipelineNodeResult as PipelineNodeResult,
)
from napari_vipp.core.execution import (
    PipelineRunRequest as PipelineRunRequest,
)
from napari_vipp.core.execution import (
    PipelineRunResult as PipelineRunResult,
)
from napari_vipp.core.export import (
    export_batch_runner_to_python,
    export_pipeline_to_python,
)
from napari_vipp.core.file_sources import (
    SourceFileSnapshot as SourceFileSnapshot,
)
from napari_vipp.core.file_sources import (
    VerifiedSourceInspection as VerifiedSourceInspection,
)
from napari_vipp.core.file_sources import load_frozen_file_source_snapshot
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
    automatic_threshold_value,
    colocalization_normalized_inputs,
    colocalization_threshold_values,
    resolve_born_wolf_psf_parameters,
    save_array_output,
)
from napari_vipp.core.pipeline import (
    DEFAULT_DYNAMIC_OUTPUT_PORTS,
    EXECUTION_BLOCKED,
    EXECUTION_ERROR,
    EXECUTION_NOT_CALCULATED,
    EXECUTION_READY,
    EXECUTION_RUNNING,
    EXECUTION_STALE,
    GLOBAL_THRESHOLD_OPERATIONS,
    MANUAL_RUN_SKIP,
    GraphNode,
    InputSpec,
    OperationSpec,
    ParameterSpec,
    PrototypePipeline,
    SourcePayload,
    grouped_palette_specs,
    operation_call_parameter_value,
    resolve_parameter_visibility,
)
from napari_vipp.core.preview import (
    MONOCHROME_COLORMAPS,
    THUMBNAIL_CONTRAST_MODES,
    THUMBNAIL_CONTRAST_SCOPES,
    make_preview,
    normalize_thumbnail_with_colormap,
    thumbnail_channel_contrast_limits,
    thumbnail_contrast_limits,
)
from napari_vipp.core.source_identity import (
    LocalSourceIdentity,
    SourceChangedError,
    capture_local_source_identity,
    verify_local_source_identity,
)
from napari_vipp.core.tables import is_table_data, save_table_output
from napari_vipp.core.workflow import (
    load_workflow,
    save_workflow,
    serialize_workflow,
    workflow_snapshot_from_pipeline,
)
from napari_vipp.ui.axis_controls import (
    AxisIntervalSlider as AxisIntervalSlider,
)
from napari_vipp.ui.axis_controls import (
    AxisOrderListWidget as AxisOrderListWidget,
)
from napari_vipp.ui.axis_controls import (
    AxisSelectionRow as AxisSelectionRow,
)
from napari_vipp.ui.axis_controls import (
    AxisSliceControl,
    AxisSliceOption,
    ReorderAxesControl,
    SelectTableColumnsControl,
)
from napari_vipp.ui.axis_controls import (
    _axis_heading_text as _axis_heading_text,
)
from napari_vipp.ui.axis_controls import (
    _axis_indices_from_value as _axis_indices_from_value,
)
from napari_vipp.ui.axis_controls import (
    _axis_options_from_order as _axis_options_from_order,
)
from napari_vipp.ui.axis_controls import (
    _axis_order_item_text as _axis_order_item_text,
)
from napari_vipp.ui.axis_controls import (
    _axis_order_tokens as _axis_order_tokens,
)
from napari_vipp.ui.axis_controls import (
    _axis_order_value as _axis_order_value,
)
from napari_vipp.ui.axis_controls import (
    _axis_slice_state_from_value as _axis_slice_state_from_value,
)
from napari_vipp.ui.axis_controls import (
    _clamped_index as _clamped_index,
)
from napari_vipp.ui.axis_controls import (
    _clamped_range as _clamped_range,
)
from napari_vipp.ui.axis_controls import (
    _control_signal_blockers as _control_signal_blockers,
)
from napari_vipp.ui.axis_controls import (
    _named_axis_order as _named_axis_order,
)
from napari_vipp.ui.axis_controls import (
    _numeric_axis_order as _numeric_axis_order,
)
from napari_vipp.ui.axis_controls import (
    _parse_axis_ranges as _parse_axis_ranges,
)
from napari_vipp.ui.axis_controls import (
    _parse_int_list as _parse_int_list,
)
from napari_vipp.ui.batch import (
    BatchPreviewResult as BatchPreviewResult,
)
from napari_vipp.ui.batch import BatchPreviewRow as BatchPreviewRow
from napari_vipp.ui.batch import CollectionBatchActions
from napari_vipp.ui.batch import CollectionBatchDialog as CollectionBatchDialog
from napari_vipp.ui.batch_controller import CollectionBatchController
from napari_vipp.ui.batch_navigator import BatchNavigator
from napari_vipp.ui.controls import (
    BoolControl,
    ChoiceControl,
    ImageSourceControl,
    NumericEntryControl,
    ParameterBounds,
    ParameterControl,
    TextControl,
    _slider_safe_bounds,
)
from napari_vipp.ui.controls import (
    FlexibleDoubleSpinBox as FlexibleDoubleSpinBox,
)
from napari_vipp.ui.controls import (
    _configure_numeric_spin_box as _configure_numeric_spin_box,
)
from napari_vipp.ui.diagnostic_workers import (
    AutoContrastRequest as AutoContrastRequest,
)
from napari_vipp.ui.diagnostic_workers import (
    AutoContrastResult as AutoContrastResult,
)
from napari_vipp.ui.diagnostic_workers import (
    AutoContrastWorker,
    ColocalizationScatterWorker,
    GeneratedLayerContrastWorker,
    InputHistogramWorker,
    ThumbnailContrastLimitWorker,
)
from napari_vipp.ui.diagnostic_workers import (
    ColocalizationScatterRequest as ColocalizationScatterRequest,
)
from napari_vipp.ui.diagnostic_workers import (
    ColocalizationScatterResult as ColocalizationScatterResult,
)
from napari_vipp.ui.diagnostic_workers import (
    GeneratedLayerContrastPlan as GeneratedLayerContrastPlan,
)
from napari_vipp.ui.diagnostic_workers import (
    GeneratedLayerContrastRequest as GeneratedLayerContrastRequest,
)
from napari_vipp.ui.diagnostic_workers import (
    GeneratedLayerContrastResult as GeneratedLayerContrastResult,
)
from napari_vipp.ui.diagnostic_workers import (
    InputHistogramDistribution as InputHistogramDistribution,
)
from napari_vipp.ui.diagnostic_workers import (
    InputHistogramRequest as InputHistogramRequest,
)
from napari_vipp.ui.diagnostic_workers import (
    InputHistogramResult as InputHistogramResult,
)
from napari_vipp.ui.diagnostic_workers import (
    ThumbnailContrastLimitRequest as ThumbnailContrastLimitRequest,
)
from napari_vipp.ui.diagnostic_workers import (
    ThumbnailContrastLimitResult as ThumbnailContrastLimitResult,
)
from napari_vipp.ui.dialogs import (
    ConnectionInsertCandidate,
    ConnectionInsertDialog,
    ConnectionInsertMappingDialog,
    ConnectionInsertPortMapping,
    ExampleWorkflowDialog,
    TunnelManagerDialog,
    TunnelSummary,
)
from napari_vipp.ui.examples import (
    EXAMPLE_WORKFLOWS as EXAMPLE_WORKFLOWS,
)
from napari_vipp.ui.examples import (
    ExampleWorkflowSpec as ExampleWorkflowSpec,
)
from napari_vipp.ui.examples import (
    _example_workflow_by_id,
    _example_workflow_path,
)
from napari_vipp.ui.examples import (
    _example_workflow_dir as _example_workflow_dir,
)
from napari_vipp.ui.file_sources import (
    SourceFileLoadResult as SourceFileLoadResult,
)
from napari_vipp.ui.file_sources import SourceFileLoadSpec as SourceFileLoadSpec
from napari_vipp.ui.file_sources import SourceFileLoadWorker as SourceFileLoadWorker
from napari_vipp.ui.history import WorkflowHistory, WorkflowHistorySnapshot
from napari_vipp.ui.lifecycle import WidgetLifecycle
from napari_vipp.ui.palette import NodePalette
from napari_vipp.ui.plots import (
    COLOCALIZATION_SCATTER_BINS as COLOCALIZATION_SCATTER_BINS,
)
from napari_vipp.ui.plots import (
    COLOCALIZATION_SCATTER_COLORMAPS as COLOCALIZATION_SCATTER_COLORMAPS,
)
from napari_vipp.ui.plots import (
    ColocalizationScatterPlot as ColocalizationScatterPlot,
)
from napari_vipp.ui.plots import HistogramPlot as HistogramPlot
from napari_vipp.ui.plots import _event_position as _event_position
from napari_vipp.ui.plots import (
    _format_histogram_label as _format_histogram_label,
)
from napari_vipp.ui.plots import (
    _histogram_series_colors as _histogram_series_colors,
)
from napari_vipp.ui.plots import (
    _prepare_colocalization_scatter_density as _prepare_scatter_density,
)
from napari_vipp.ui.plots import (
    _qcolor_from_channel_color as _qcolor_from_channel_color,
)
from napari_vipp.ui.search import (
    _fuzzy_match as _fuzzy_match,
)
from napari_vipp.ui.search import (
    _fuzzy_token_match as _fuzzy_token_match,
)
from napari_vipp.ui.search import _normalize_search_text
from napari_vipp.ui.source_adapter import (
    LiveLayerSnapshot,
    LiveLayerSourceAdapter,
    SourceRevisionToken,
    apply_live_layer_axis_transform,
)
from napari_vipp.ui.view_dims import ViewDimAxis as ViewDimAxis
from napari_vipp.ui.view_dims import ViewDimAxisControl as ViewDimAxisControl
from napari_vipp.ui.view_dims import ViewDimsBar as ViewDimsBar
from napari_vipp.ui.workers import PipelineRunWorker as PipelineRunWorker

_provisional_generated_layer_contrast_limits = (
    provisional_generated_layer_contrast_limits
)

_RGB_VOLUME_CHANNELS = (
    (0, "Red", "red"),
    (1, "Green", "green"),
    (2, "Blue", "blue"),
)

_GENERATED_LAYER_CONTRAST_METADATA_KEYS = (
    "vipp_display_contrast_basis",
    "vipp_display_contrast_pending",
    "vipp_display_contrast_adjustable",
    "vipp_exact_finite_data_range",
    "_vipp_display_contrast_key",
    "_vipp_display_contrast_initial_limits",
)

COMPOSITE_CHANNEL_ASSIGNMENT_CHOICES = (
    "Unassigned",
    *CHANNEL_COLOR_CHOICES,
)

ASYNC_SOURCE_FILE_BYTES = 32 * 1024 * 1024
AUTO_BACKGROUND_MIN_BYTES = 32 * 1024 * 1024
AUTO_BACKGROUND_MIN_ELEMENTS = 4_000_000
AUTO_CONTRAST_BACKGROUND_MIN_ELEMENTS = 1_000_000
INSPECTOR_STATISTICS_CHUNK_ELEMENTS = 1_048_576


if TYPE_CHECKING:
    import napari


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
class _IsolatedTuningSnapshot:
    node_id: str
    params: dict[str, object]
    output: object
    output_state: object
    node_outputs: tuple[object, ...]
    node_output_states: tuple[object, ...]
    execution_states: dict[str, str]
    execution_messages: dict[str, str]
    completed_node_ids: frozenset[str]
    undo_stack: tuple[WorkflowHistorySnapshot, ...]
    redo_stack: tuple[WorkflowHistorySnapshot, ...]
    batch_workflow_stale: bool

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
CALCULATE_ALL_ATTENTION_STYLE = (
    "QPushButton {"
    " background-color: #78350f;"
    " color: #fde68a;"
    f" border: 2px solid {STALE_EXECUTION_ACCENT};"
    " border-radius: 3px;"
    " font-weight: 650;"
    "}"
    "QPushButton:hover { background-color: #92400e; }"
    "QPushButton:pressed { background-color: #451a03; }"
)
DEFAULT_CACHE_MEMORY_LIMIT_PERCENT = 90
MEMORY_GUARD_MIN_FREE_BYTES = 512 * 1024 * 1024
EXPLICIT_OUTPUT_OPERATIONS = {"batch_output", "save_output"}
SMART_CACHE_RECENT_LIMIT = 6
COMPACT_DECONVOLUTION_INSPECTOR_OPERATIONS = {
    "richardson_lucy_deconvolution",
    "richardson_lucy_tv_deconvolution",
}




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


def _toolbar_field_pair(
    label_text: str,
    control: QWidget,
    *,
    parent: QWidget | None = None,
) -> tuple[QWidget, QLabel]:
    """Return a compact, indivisible toolbar label/control pair."""
    field = QWidget(parent)
    layout = QHBoxLayout(field)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)
    label = QLabel(label_text, field)
    label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    label.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
    layout.addWidget(label)
    layout.addWidget(control)
    field.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
    return field, label


def _configure_toolbar_combo(combo: QComboBox) -> None:
    """Keep toolbar choices readable without ambiguous compression."""
    longest_text = max(
        (len(combo.itemText(index)) for index in range(combo.count())),
        default=0,
    )
    combo.setMinimumContentsLength(longest_text)
    combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
    combo.setMinimumWidth(max(78, combo.minimumSizeHint().width()))
    combo.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)


class _InspectorNoteLabel(QLabel):
    """Wrapped inspector text that always reserves its rendered height."""

    def __init__(self, text: str = "", parent=None):
        super().__init__(parent)
        self.setWordWrap(True)
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Minimum)
        self.setText(text)

    def setText(self, text: str) -> None:  # noqa: N802
        super().setText(text)
        self._sync_wrapped_minimum_height()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._sync_wrapped_minimum_height()

    def _sync_wrapped_minimum_height(self) -> None:
        width = self.contentsRect().width()
        if width <= 0 or not self.hasHeightForWidth():
            return
        required_height = max(int(self.heightForWidth(width)), 0)
        if required_height != self.minimumHeight():
            self.setMinimumHeight(required_height)
            self.updateGeometry()


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
    """Compatibility facade for the extracted scatter-density calculation."""
    return _prepare_scatter_density(
        channel_1,
        channel_2,
        threshold_1=threshold_1,
        threshold_2=threshold_2,
        roi_mask=roi_mask,
        intensity_max=intensity_max,
        bins=bins,
        progress=progress,
        chunk_elements=INSPECTOR_STATISTICS_CHUNK_ELEMENTS,
    )


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
        self._closing = False
        self._lifecycle = WidgetLifecycle(self)
        self._live_source_adapter = LiveLayerSourceAdapter(
            self._on_live_source_invalidated
        )
        self._live_source_node_layers: dict[str, object] = {}
        self.pipeline = PrototypePipeline()
        self._selected_node_id = "gaussian"
        self._workflow_load_selection_in_progress = False
        self._active_pinned_node_id: str | None = None
        self._inspect_layer_name = "VIPP Inspect"
        self._preview_disabled_node_ids: set[str] = set()
        self._hidden_input_layer_states: dict[int, tuple[object, bool]] = {}
        self._sample_payload_cache: dict[str, SourcePayload] | None = None
        self._source_inspection_cache: dict[str, VerifiedSourceInspection] = {}
        self._psf_preflight_cache: dict[
            str,
            tuple[tuple[object, ...], PsfPreflightResult],
        ] = {}
        self._file_source_payload_cache: dict[
            tuple[object, ...],
            SourceFileSnapshot,
        ] = {}
        self._file_source_path_identities: dict[str, LocalSourceIdentity] = {}
        self._interactive_collection_source_paths: dict[str, Path] = {}
        self._interactive_collection_batch_items: tuple[BatchItemPlan, ...] = ()
        self._interactive_collection_batch_config: BatchConfig | None = None
        self._interactive_collection_batch_config_path: Path | None = None
        self._interactive_collection_batch_index = -1
        self._interactive_collection_batch_requested_index = -1
        self._interactive_collection_batch_failed_index = -1
        self._interactive_collection_batch_plan_stale = False
        self._interactive_collection_batch_workflow_stale = False
        self._active_collection_batch_dialog: CollectionBatchDialog | None = None
        self._collection_batch_running = False
        self._collection_batch_graph_refresh_pending = False
        self._active_source_load_id: int | None = None
        self._source_load_serial = 0
        self._source_load_pending = False
        self._dock_chrome_configured = False
        self._dock_window_behavior_configured = False
        self._initial_dock_size_applied = False
        self._history = WorkflowHistory(limit=self.HISTORY_LIMIT)
        # Kept as aliases while downstream tests and integrations transition to
        # the explicit session-history component.
        self._undo_stack = self._history.undo_stack
        self._redo_stack = self._history.redo_stack
        self._rescale_auto_output_ranges: dict[str, tuple[float, float]] = {}
        self._code_dialogs: list[QDialog] = []
        self._pending_dirty_node_ids: set[str] = set()
        self._pending_manual_node_ids: set[str] = set()
        self._inflight_dirty_node_ids: set[str] | None = None
        self._isolated_tuning_node_id: str | None = None
        self._isolated_tuning_snapshot: _IsolatedTuningSnapshot | None = None
        self._isolated_tuning_has_changes = False
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
        for combo in (
            self.preview_mode_combo,
            self.thumbnail_contrast_combo,
            self.thumbnail_scope_combo,
            self.thumbnail_colormap_combo,
        ):
            _configure_toolbar_combo(combo)
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
        self.refresh_button.setToolTip(
            "Reload file-path sources and recalculate the graph. File data is "
            "held as a frozen scientific snapshot until Refresh is pressed."
        )
        self.graph_focus_button = QPushButton("Focus")
        self.graph_focus_button.setToolTip(
            "Center the workflow graph in the canvas without changing zoom."
        )
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
        self.batch_button = QPushButton("Batch workspace...")
        self.batch_button.setToolTip(
            "Bind collections, preview paired representatives through the "
            "graph, run the full batch, and inspect progress."
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
            "Graph labels, cache behavior, and collapsed display controls."
        )
        self.settings_menu = QMenu(self.settings_menu_button)
        self.settings_menu.aboutToShow.connect(self._populate_settings_toolbar_menu)
        self.settings_menu_button.setMenu(self.settings_menu)
        self.port_label_mode_combo = QComboBox()
        self.port_label_mode_combo.addItems(
            [
                PortLabelMode.AMBIGUOUS_ONLY.value,
                PortLabelMode.SHOW_ALL.value,
                PortLabelMode.HIDE_ALL.value,
            ]
        )
        self.port_label_mode_combo.setCurrentText(
            PortLabelMode.AMBIGUOUS_ONLY.value
        )
        self.port_label_mode_combo.setToolTip(
            "Show port names only on ambiguous multi-port nodes, on every node, "
            "or nowhere on the graph."
        )
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
        self._background_execution_state_overrides: dict[
            str,
            tuple[int, str, str],
        ] = {}
        self._background_node_result_overrides: dict[str, PipelineNodeResult] = {}

        self.selected_title = QLabel("Gaussian Blur")
        self.selected_title.setStyleSheet("font-weight: 650;")
        self.selected_title.setWordWrap(True)
        self.selected_title.setSizePolicy(
            QSizePolicy.Ignored,
            QSizePolicy.Preferred,
        )
        self.thumbnail_checkbox = QCheckBox("Show thumbnail preview")
        self.thumbnail_checkbox.setChecked(True)
        self.keep_cached_checkbox = QCheckBox("Keep output cached")
        self.keep_cached_checkbox.setToolTip(
            "Retain this node output in Smart and Low-memory cache modes. "
            "Use this for expensive intermediate images or tables you inspect often."
        )
        self.isolated_tuning_checkbox = QCheckBox("Tune node in isolation")
        self.isolated_tuning_checkbox.setToolTip(
            "Recalculate only this node while you tune its parameters. "
            "Downstream nodes stay stale until Apply and continue."
        )
        self.isolated_tuning_panel = QFrame()
        self.isolated_tuning_panel.setObjectName("IsolatedTuningPanel")
        self.isolated_tuning_panel.setStyleSheet(
            "QFrame#IsolatedTuningPanel { background: #2a2416; "
            "border: 1px solid #f59e0b; border-radius: 5px; padding: 5px; }"
        )
        self.isolated_tuning_status = QLabel("Downstream paused")
        self.isolated_tuning_status.setWordWrap(True)
        self.isolated_tuning_status.setStyleSheet(
            "color: #fde68a; font-weight: 650;"
        )
        self.apply_isolated_tuning_button = QPushButton("Apply and continue")
        self.cancel_isolated_tuning_button = QPushButton("Cancel tuning")
        self.isolated_tuning_panel.setVisible(False)
        self.execution_group = QGroupBox("Execution")
        self.execution_status_label = QLabel("Automatic")
        self.execution_status_label.setWordWrap(True)
        self.execution_status_label.setMinimumHeight(34)
        self.execution_status_label.setSizePolicy(
            QSizePolicy.Ignored,
            QSizePolicy.Preferred,
        )
        self.auto_recalculate_checkbox = QCheckBox("Auto Recalculate")
        self.auto_recalculate_notice = QLabel(
            "Auto Recalculate runs this node after upstream or parameter changes. "
            "This can be slow on large images."
        )
        self.auto_recalculate_notice.setWordWrap(True)
        self.auto_recalculate_notice.setSizePolicy(
            QSizePolicy.Ignored,
            QSizePolicy.Preferred,
        )
        self.auto_recalculate_notice.setStyleSheet("color: #f59e0b;")
        self.calculate_button = QPushButton("Calculate")
        self.parameter_group = QGroupBox("Parameters")
        self.parameter_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
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
            "Explicit RGB and RGBA inputs use weighted RGB luminance; alpha is "
            "ignored. Unlabelled arrays are treated as scalar data. "
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
        self._collection_batch_controller = CollectionBatchController(
            workflow_document_provider=self._batch_workflow_document,
            pipeline_provider=lambda: self.pipeline,
        )
        self._select_node(self._selected_node_id)
        self.run_pipeline()
        self._sync_history_actions()

    def closeEvent(self, event):  # noqa: N802
        self._lifecycle.shutdown()
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
        if self._closing:
            return
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
        if self._closing:
            return
        dock = self._dock_widget()
        if dock is None or self._dock_chrome_configured:
            return
        try:
            if not self._dock_window_behavior_configured:
                self._lifecycle.install_event_filter(dock, self)
                self._lifecycle.connect(
                    dock.topLevelChanged,
                    self._on_dock_top_level_changed,
                )
                self._lifecycle.connect(
                    dock.visibilityChanged,
                    self._on_dock_visibility_changed,
                )
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
        input_row.setSpacing(6)

        self.thumbnail_toolbar_group = QWidget(self)
        thumbnail_layout = QHBoxLayout(self.thumbnail_toolbar_group)
        thumbnail_layout.setContentsMargins(0, 0, 0, 0)
        thumbnail_layout.setSpacing(10)
        (
            self.preview_toolbar_field,
            self.preview_toolbar_label,
        ) = _toolbar_field_pair(
            "Preview",
            self.preview_mode_combo,
            parent=self.thumbnail_toolbar_group,
        )
        (
            self.contrast_toolbar_field,
            self.contrast_toolbar_label,
        ) = _toolbar_field_pair(
            "Contrast",
            self.thumbnail_contrast_combo,
            parent=self.thumbnail_toolbar_group,
        )
        (
            self.contrast_range_toolbar_field,
            self.contrast_range_toolbar_label,
        ) = _toolbar_field_pair(
            "Contrast Range",
            self.thumbnail_scope_combo,
            parent=self.thumbnail_toolbar_group,
        )
        (
            self.mono_toolbar_field,
            self.mono_toolbar_label,
        ) = _toolbar_field_pair(
            "Mono",
            self.thumbnail_colormap_combo,
            parent=self.thumbnail_toolbar_group,
        )
        for field in (
            self.preview_toolbar_field,
            self.contrast_toolbar_field,
            self.contrast_range_toolbar_field,
            self.mono_toolbar_field,
        ):
            thumbnail_layout.addWidget(field)
        self.thumbnail_toolbar_group.setSizePolicy(
            QSizePolicy.Maximum,
            QSizePolicy.Preferred,
        )
        input_row.addWidget(self.thumbnail_toolbar_group)

        self._toolbar_zoom_separator = _toolbar_separator()
        input_row.addWidget(self._toolbar_zoom_separator)
        self.zoom_toolbar_controls = QWidget(self)
        zoom_controls_layout = QHBoxLayout(self.zoom_toolbar_controls)
        zoom_controls_layout.setContentsMargins(0, 0, 0, 0)
        zoom_controls_layout.setSpacing(4)
        zoom_controls_layout.addWidget(self.graph_zoom_slider)
        zoom_controls_layout.addWidget(self.graph_zoom_reset_button)
        zoom_controls_layout.addWidget(self.graph_zoom_label)
        (
            self.zoom_toolbar_field,
            self.zoom_toolbar_label,
        ) = _toolbar_field_pair(
            "Zoom",
            self.zoom_toolbar_controls,
            parent=self,
        )
        input_row.addWidget(self.zoom_toolbar_field)

        self._toolbar_action_separator = _toolbar_separator()
        input_row.addWidget(self._toolbar_action_separator)
        self.graph_actions_toolbar_group = QWidget(self)
        graph_actions_layout = QHBoxLayout(self.graph_actions_toolbar_group)
        graph_actions_layout.setContentsMargins(0, 0, 0, 0)
        graph_actions_layout.setSpacing(4)
        for button in (
            self.refresh_button,
            self.graph_focus_button,
            self.calculate_all_button,
            self.auto_structure_button,
            self.tunnel_manager_button,
            self.undo_button,
            self.redo_button,
        ):
            graph_actions_layout.addWidget(button)
        self.graph_actions_toolbar_group.setSizePolicy(
            QSizePolicy.Maximum,
            QSizePolicy.Preferred,
        )
        input_row.addWidget(self.graph_actions_toolbar_group)

        self._toolbar_settings_separator = _toolbar_separator(6)
        input_row.addWidget(self._toolbar_settings_separator)
        input_row.addWidget(self.settings_menu_button)
        input_row.addStretch(1)
        self.main_toolbar_layout = input_row
        root.addLayout(input_row)
        self._toolbar_checkbox_widgets = []
        for widget in (
            self.background_all_checkbox,
            self.follow_dims_checkbox,
        ):
            widget.setVisible(False)
        self._toolbar_dropdown_widgets = [self.thumbnail_toolbar_group]
        self._toolbar_zoom_widgets = [self.zoom_toolbar_field]
        self._toolbar_settings_widgets = [
            self._toolbar_settings_separator,
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
        self.batch_navigator = BatchNavigator(self)
        root.addWidget(self.batch_navigator)
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
        expanded_width = max(
            self.TOOLBAR_HIDE_DROPDOWNS_WIDTH,
            self._expanded_toolbar_required_width(),
        )
        hide_dropdowns = 0 < width < expanded_width
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
        self._toolbar_zoom_separator.setVisible(
            not hide_dropdowns and not hide_zoom
        )
        self._toolbar_action_separator.setVisible(
            not hide_dropdowns or not hide_zoom
        )
        self.auto_structure_button.setText(
            "Structure" if hide_dropdowns or hide_zoom else "Auto structure graph"
        )

    def _expanded_toolbar_required_width(self) -> int:
        """Return the width needed to show the complete first toolbar row."""
        original_text = self.auto_structure_button.text()
        self.auto_structure_button.setText("Auto structure graph")
        actions_layout = self.graph_actions_toolbar_group.layout()
        if actions_layout is not None:
            actions_layout.invalidate()
        widgets = (
            self.thumbnail_toolbar_group,
            self._toolbar_zoom_separator,
            self.zoom_toolbar_field,
            self._toolbar_action_separator,
            self.graph_actions_toolbar_group,
            self._toolbar_settings_separator,
            self.settings_menu_button,
        )
        required = sum(widget.sizeHint().width() for widget in widgets)
        required += self.main_toolbar_layout.spacing() * (len(widgets) - 1)
        required += 12
        self.auto_structure_button.setText(original_text)
        if actions_layout is not None:
            actions_layout.invalidate()
        return required

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
        self._add_combo_menu(menu, "Port labels", self.port_label_mode_combo)
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
        layout.addWidget(self.isolated_tuning_checkbox)
        isolated_tuning_layout = QVBoxLayout(self.isolated_tuning_panel)
        isolated_tuning_layout.setContentsMargins(7, 7, 7, 7)
        isolated_tuning_layout.addWidget(self.isolated_tuning_status)
        isolated_tuning_actions = QHBoxLayout()
        isolated_tuning_actions.addWidget(self.apply_isolated_tuning_button)
        isolated_tuning_actions.addWidget(self.cancel_isolated_tuning_button)
        isolated_tuning_layout.addLayout(isolated_tuning_actions)
        layout.addWidget(self.isolated_tuning_panel)
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
        self.graph_focus_button.clicked.connect(self._focus_graph)
        self.calculate_all_button.clicked.connect(self._calculate_all_nodes)
        self.undo_action.triggered.connect(self.undo)
        self.redo_action.triggered.connect(self.redo)
        self.save_workflow_button.clicked.connect(self._save_workflow_dialog)
        self.load_workflow_button.clicked.connect(self._load_workflow_dialog)
        self.export_button.clicked.connect(self._export_python_dialog)
        self.batch_button.clicked.connect(
            lambda _checked=False: self._batch_collection_dialog()
        )
        self.batch_navigator.itemSelected.connect(
            self._preview_interactive_collection_batch_item
        )
        self.batch_navigator.workspaceRequested.connect(
            self._open_active_collection_batch_workspace
        )
        self.export_ome_button.clicked.connect(self._export_ome_dataset_dialog)
        self.tunnel_manager_button.clicked.connect(self._show_tunnel_manager)
        self.pipeline_cancel_button.clicked.connect(self._cancel_background_pipeline_run)
        self.port_label_mode_combo.currentTextChanged.connect(
            self._on_port_label_mode_changed
        )
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
        self.isolated_tuning_checkbox.toggled.connect(
            self._on_isolated_tuning_toggled
        )
        self.apply_isolated_tuning_button.clicked.connect(
            self._apply_isolated_tuning
        )
        self.cancel_isolated_tuning_button.clicked.connect(
            self._cancel_isolated_tuning
        )
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
        self.graph_view.node_isolation_requested.connect(
            self._toggle_node_isolation_from_graph
        )
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
            self._lifecycle.connect(
                self.viewer.layers.events.inserted,
                self._on_viewer_layers_changed,
            )
            self._lifecycle.connect(
                self.viewer.layers.events.removed,
                self._on_viewer_layers_changed,
            )
        except Exception:
            pass
        try:
            self._lifecycle.connect(
                self.viewer.dims.events.current_step,
                self._on_dims_changed,
            )
            self._lifecycle.connect(
                self.viewer.dims.events.point,
                self._on_dims_changed,
            )
        except Exception:
            pass
        self._debounce_timer.timeout.connect(
            self._finish_debounced_parameter_history_group
        )

    def _toggle_left_panel(self) -> None:
        self._set_left_panel_visible(self.palette_panel.isHidden())

    def _toggle_right_panel(self) -> None:
        self._set_right_panel_visible(self.inspector_panel.isHidden())

    def _on_graph_zoom_slider_changed(self, value: int) -> None:
        self.graph_view.set_zoom_percent(float(value))

    def _reset_graph_zoom(self) -> None:
        self.graph_view.reset_zoom()

    def _focus_graph(self) -> None:
        if self.graph_view.center_graph():
            self.status_label.setText("Centered workflow graph.")
        else:
            self.status_label.setText("Workflow graph is empty; returned to origin.")

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
        if self._isolated_tuning_node_id is not None:
            self._apply_isolated_tuning(run=False, announce=False)
        self._finish_parameter_history_group()
        if not self._history.can_undo:
            return
        current = self._current_history_snapshot()
        snapshot = self._history.undo(current)
        if snapshot is None:
            return
        self._restore_history_snapshot(snapshot)
        self._sync_history_actions()
        self.status_label.setText("Undid last workflow edit.")

    def redo(self) -> None:
        """Reapply the most recently undone workflow graph snapshot."""
        self._finish_parameter_history_group()
        if not self._history.can_redo:
            return
        current = self._current_history_snapshot()
        snapshot = self._history.redo(current)
        if snapshot is None:
            return
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
            workflow=workflow_snapshot_from_pipeline(
                self.pipeline,
                positions,
                self._graph_note_documents()
                if notes_override is None
                else notes_override,
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
        snapshot = snapshot or self._current_history_snapshot()
        if self._history.push(snapshot):
            self._sync_history_actions()

    def _push_undo_if_changed(self, before: WorkflowHistorySnapshot) -> None:
        if before != self._current_history_snapshot():
            self._push_undo_snapshot(before)
            self._mark_collection_batch_workflow_stale_if_needed()

    def _record_parameter_undo(self, node_id: str, name: str) -> None:
        if (
            self._isolated_tuning_node_id is not None
            and node_id != self._isolated_tuning_node_id
        ):
            self._finish_parameter_history_group()
        key = (node_id, name)
        if not self._history.should_capture_group(key):
            return
        self._push_undo_snapshot()

    def _finish_debounced_parameter_history_group(self) -> None:
        self._finish_parameter_history_group(preserve_isolated_tuning=True)

    def _finish_parameter_history_group(
        self,
        *,
        preserve_isolated_tuning: bool = False,
    ) -> None:
        """Close a parameter group before another history-backed edit.

        A graph, layout, note, or other serialized workflow edit commits an
        active isolated-tuning session first. This keeps Cancel from restoring
        execution dictionaries and history stacks captured for an older graph.
        """
        if (
            not preserve_isolated_tuning
            and self._isolated_tuning_node_id is not None
        ):
            had_changes = self._apply_isolated_tuning(
                run=False,
                announce=False,
            )
            if had_changes:
                QTimer.singleShot(
                    0,
                    self._resume_pipeline_after_history_edit,
                )
            return
        self._history.finish_group()

    def _resume_pipeline_after_history_edit(self) -> None:
        """Continue work committed before a serialized workflow edit."""
        if self._debounce_timer.isActive():
            # A parameter edit made immediately after committing isolation owns
            # the combined rerun; its debounce will consume the same pending set.
            return
        if self._pending_dirty_node_ids & set(self.pipeline.nodes):
            self.run_pipeline()

    def _restore_history_snapshot(self, snapshot: WorkflowHistorySnapshot) -> None:
        if self._isolated_tuning_node_id is not None:
            self._apply_isolated_tuning(run=False, announce=False)
        self._debounce_timer.stop()
        self._clear_interactive_collection_batch_session()
        with self._history.suspend_recording():
            workflow = snapshot.workflow
            pinned_layer = self._active_pinned_layer()
            if pinned_layer is not None:
                self._remove_layer(pinned_layer)
            workflow.graph.restore_into(self.pipeline)
            self._restore_graph_notes(
                note.to_mapping() for note in workflow.notes
            )
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
                workflow.positions_dict(),
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

    def _sync_history_actions(self) -> None:
        undo_enabled = self._history.can_undo
        redo_enabled = self._history.can_redo
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
        if operation_id == "input" and (
            self._interactive_collection_batch_items
            or self._active_collection_batch_dialog is not None
        ):
            self._clear_interactive_collection_batch_session()
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
        if original.operation_id == "input" and (
            self._interactive_collection_batch_items
            or self._active_collection_batch_dialog is not None
        ):
            self._clear_interactive_collection_batch_session()
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
                call_value = operation_call_parameter_value(
                    node.operation_id,
                    key,
                    value,
                )
                lines.append(f"    {key!r}: {call_value!r},")
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
            key: operation_call_parameter_value(
                node.operation_id,
                key,
                value,
            )
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
        if self._isolated_tuning_node_id is not None:
            self._apply_isolated_tuning(run=False, announce=False)
        self._finish_parameter_history_group()
        self._clear_interactive_collection_batch_session()
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
        loaded = self.load_workflow_file(path, prefer_image_source=True)
        if (
            self._active_pipeline_run_id is None
            and self._active_source_load_id is None
        ):
            self.status_label.setText(f"Opened example workflow: {example.title}.")
        return loaded

    def load_workflow_file(
        self,
        path: str | Path,
        *,
        prefer_image_source: bool = False,
        preserve_batch_workspace: bool = False,
    ) -> Path:
        """Load a workflow file into the widget and recompute the graph.

        Ordinary workflows restore an explicitly saved inspector selection.
        Bundled examples request ``prefer_image_source`` so they open at the
        start of the scientific data flow instead of a saved terminal node.
        """
        if self._isolated_tuning_node_id is not None:
            self._apply_isolated_tuning(run=False, announce=False)
        self._finish_parameter_history_group()
        self._clear_interactive_collection_batch_session(
            close_workspace=not preserve_batch_workspace,
        )
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
        image_source_node_id = self._first_image_source_node_id()
        if prefer_image_source and image_source_node_id:
            selected_node_id = image_source_node_id
        elif not selected_node_id and image_source_node_id:
            selected_node_id = image_source_node_id
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
        self._workflow_load_selection_in_progress = True
        try:
            if selected_node_id:
                self.graph_view.select_node(selected_node_id)
            else:
                self._select_first_available_node()
        finally:
            self._workflow_load_selection_in_progress = False
        if isinstance(right_panel_visible, bool):
            self._set_right_panel_visible(right_panel_visible)
        self._invalidate_pipeline_cache()
        self.run_pipeline()
        self._push_undo_if_changed(before)
        return source

    def _first_image_source_node_id(self) -> str:
        """Return the first Image Source in stable graph order, if present."""
        for node_id in self.pipeline.topological_order():
            node = self.pipeline.nodes.get(node_id)
            if node is not None and node.operation_id == "input":
                return node_id
        return ""

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
        config: BatchConfig | None = None,
    ) -> None:
        if self._collection_batch_running:
            self.status_label.setText(
                "A collection batch is already running. Its progress remains "
                "visible in the batch workspace."
            )
            active_dialog = self._active_collection_batch_dialog
            if active_dialog is not None:
                active_dialog.show()
                active_dialog.raise_()
                active_dialog.activateWindow()
            return
        active_dialog = self._active_collection_batch_dialog
        if active_dialog is not None and config_path is None and config is None:
            active_dialog.show()
            active_dialog.raise_()
            active_dialog.activateWindow()
            return
        if active_dialog is not None:
            self._discard_collection_batch_dialog(active_dialog)
        dialog = CollectionBatchDialog(
            self,
            source_nodes=self._batch_source_rows(),
            actions=self._collection_batch_dialog_actions(),
        )
        self._active_collection_batch_dialog = dialog
        dialog.runRequested.connect(
            lambda values, active=dialog: self._run_collection_batch_from_workspace(
                active, values
            )
        )
        dialog.previewInvalidated.connect(
            lambda active=dialog: self._mark_interactive_collection_batch_stale(active)
        )
        if config_path is not None:
            try:
                loaded_config = self._load_collection_batch_config(config_path)
                dialog._apply_config(loaded_config)
                dialog._loaded_config_path = Path(config_path)
                config_root = Path(config_path).expanduser().resolve().parent
                if (config_root / SYNTHETIC_BATCH_GROUND_TRUTH_FILENAME).is_file():
                    dialog.set_demo_context(SyntheticBatchDemo.from_root(config_root))
                dialog._preview_batch()
            except Exception as exc:
                self._discard_collection_batch_dialog(dialog)
                self.status_label.setText(f"Could not open batch config: {exc}")
                return
        elif config is not None:
            try:
                dialog._apply_config(config)
                dialog._preview_batch()
            except Exception as exc:
                self._discard_collection_batch_dialog(dialog)
                self.status_label.setText(f"Could not open batch workspace: {exc}")
                return
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _discard_collection_batch_dialog(
        self,
        dialog: CollectionBatchDialog,
    ) -> None:
        """Permanently discard a replaced workspace instead of hiding it."""
        if dialog is self._active_collection_batch_dialog:
            self._active_collection_batch_dialog = None
        dialog.close()
        dialog.deleteLater()

    def _run_collection_batch_from_workspace(
        self,
        dialog: CollectionBatchDialog,
        values: object,
    ) -> None:
        """Execute a full batch while retaining its workspace and progress."""
        if dialog is not self._active_collection_batch_dialog:
            return
        if self._collection_batch_running:
            self.status_label.setText("A collection batch is already running.")
            return
        if not isinstance(values, dict):
            dialog.show_run_error("The batch workspace returned invalid settings.")
            return
        if self._interactive_collection_batch_items and (
            self._active_source_load_id is not None
            or self._active_pipeline_run_id is not None
            or self._source_load_pending
            or self._pipeline_run_pending
        ):
            dialog.show_plan_refresh_required(
                "Wait for the active source load or graph calculation to finish "
                "before running the full batch."
            )
            return
        if (
            self._interactive_collection_batch_items
            and self._interactive_collection_batch_requested_index >= 0
        ):
            dialog.show_plan_refresh_required(
                "Wait for the representative calculation to finish before "
                "running the full batch."
            )
            return
        if self._interactive_collection_batch_items and (
            self._interactive_collection_batch_index < 0
            or self._interactive_collection_batch_failed_index >= 0
        ):
            dialog.show_plan_refresh_required(
                "The representative graph preview is unavailable or failed. "
                "Retry it or select another sample and wait for a successful "
                "calculation before running the full batch."
            )
            return
        preview = dialog._preview_result
        if preview is None:
            if dialog._preview_batch():
                dialog.show_plan_refresh_required(
                    "The runnable plan was refreshed. Review its items, "
                    "destinations, and preflight statuses, then click Run batch "
                    "again."
                )
            return

        current_hash = scientific_workflow_hash(self._batch_workflow_document())
        if current_hash != preview.config.workflow_sha256:
            dialog.invalidate_for_workflow_change()
            self._mark_collection_batch_workflow_stale_if_needed()
            if dialog._preview_batch():
                dialog.show_plan_refresh_required(
                    "The workflow changed, so VIPP rebuilt the runnable plan. "
                    "Review it, then click Run batch again."
                )
            return

        source_change = self._reviewed_batch_source_change(preview.items)
        if source_change:
            dialog.invalidate_for_source_change(source_change)
            self._interactive_collection_batch_plan_stale = True
            self.batch_navigator.set_session_stale(
                True,
                message=(
                    "A reviewed source changed on disk. The graph keeps its "
                    "pinned earlier revision; press Refresh, then Preview batch "
                    "again before running."
                ),
            )
            self.status_label.setText(
                "Batch stopped because a reviewed source changed. Press "
                "Refresh, then preview the batch again."
            )
            return

        try:
            fresh_preview = self._collection_batch_controller.preview(
                **values,
                preview_limit=25,
            )
        except Exception:
            dialog._preview_batch()
            return
        if (
            fresh_preview.config != preview.config
            or fresh_preview.items != preview.items
        ):
            if dialog._preview_batch():
                dialog.show_plan_refresh_required(
                    "Batch inputs, destinations, or preflight statuses changed "
                    "since the displayed plan. VIPP refreshed the table; review "
                    "it, then click Run batch again."
                )
            return

        total = preview.total_items
        if total <= 0:
            dialog.show_run_error("The batch plan contains no items to run.")
            return

        # Batch execution owns the GUI-thread progress surface. Supersede any
        # representative calculation so late interactive results cannot update
        # the graph or toolbar while the detached headless batch is running.
        self._debounce_timer.stop()
        if self._interactive_collection_batch_requested_index >= 0:
            self._show_interactive_collection_batch_preview_error(
                self._interactive_collection_batch_requested_index,
                "the representative calculation was superseded by the full "
                "batch run",
            )
        self._abandon_background_pipeline_run()
        if self._active_source_load_id is not None:
            self._source_load_serial += 1
            self._active_source_load_id = None
            self._source_load_pending = False
            self._set_pipeline_busy(False)
        self._collection_batch_running = True
        dialog.begin_run(total)
        self.batch_navigator.set_navigation_enabled(False)
        self.batch_navigator.begin_batch_progress(
            total,
            "Preparing full batch run...",
        )
        try:
            result = self._run_collection_batch(
                **values,
                expected_items=preview.items,
            )
        except Exception as exc:
            self._set_pipeline_busy(False)
            if dialog is self._active_collection_batch_dialog:
                dialog.show_run_error(
                    str(exc),
                    defer_control_restore=True,
                )
            self.batch_navigator.fail_batch_progress(
                f"Batch failed: {exc}",
            )
            self.status_label.setText(f"Batch failed: {exc}")
            return
        finally:
            self._collection_batch_running = False
            self.batch_navigator.set_navigation_enabled(True)
            if self._collection_batch_graph_refresh_pending:
                self._collection_batch_graph_refresh_pending = False
                QTimer.singleShot(0, self.run_pipeline)
        if dialog is not self._active_collection_batch_dialog:
            self.status_label.setText(
                "Batch finished, but its workspace was replaced before the "
                "result could be displayed."
            )
            return
        validation_text = self._validate_collection_batch_demo_result(
            dialog._loaded_config_path,
            result,
        )
        summary = result.summary
        summary_text = (
            f"Batch finished: {summary['completed']} completed, "
            f"{summary['partial']} partial, {summary['skipped']} skipped, "
            f"{summary['failed']} failed; {len(result.saved_paths)} output(s) "
            f"saved. Manifest: {result.manifest_path}."
        )
        self.status_label.setText(
            summary_text + (f" {validation_text}" if validation_text else "")
        )
        dialog.finish_run(
            result,
            validation_text,
            defer_control_restore=True,
        )
        self.batch_navigator.finish_batch_progress(
            f"Batch finished: {summary['completed']} completed, "
            f"{summary['partial']} partial, {summary['skipped']} skipped, "
            f"{summary['failed']} failed.",
        )
        dialog.mark_plan_historical_after_run()
        if (
            self._interactive_collection_batch_items
            and self._interactive_collection_batch_index >= 0
        ):
            self._interactive_collection_batch_plan_stale = True
            self.batch_navigator.set_session_stale(
                True,
                message=(
                    "The displayed representative belongs to the completed "
                    "run. Preview batch again before replaying so current "
                    "inputs and destinations are rechecked."
                ),
            )

    def _reviewed_batch_source_change(
        self,
        items: Iterable[BatchItemPlan],
    ) -> str:
        """Return a message when any pinned batch source changed in place."""
        planned_items = tuple(items)
        reviewed_paths = {
            Path(raw_path).expanduser().resolve()
            for item in planned_items
            for raw_path in item.source_paths.values()
        }
        collection_node_ids = {
            node_id for item in planned_items for node_id in item.source_paths
        }
        for node_id, node in self.pipeline.nodes.items():
            if node.operation_id != "input" or node_id in collection_node_ids:
                continue
            fixed_path = self._file_source_path_for_node(node)
            if fixed_path is not None:
                reviewed_paths.add(fixed_path)
        checked: set[str] = set()
        for path in reviewed_paths:
            path_text = str(path)
            if path_text in checked:
                continue
            checked.add(path_text)
            expected = self._file_source_path_identities.get(path_text)
            if expected is None:
                continue
            try:
                verify_local_source_identity(path, expected)
            except SourceChangedError as exc:
                return str(exc)
        return ""

    def _collection_batch_dialog_actions(self) -> CollectionBatchActions:
        return CollectionBatchActions(
            preview_batch=lambda values, preview_limit: self._preview_collection_batch(
                **values,
                preview_limit=preview_limit,
            ),
            choose_demo=lambda dialog_parent: self._choose_collection_batch_demo(
                dialog_parent=dialog_parent
            ),
            source_rows=self._batch_source_rows,
            load_config=self._load_collection_batch_config,
            save_config=lambda path, values: self._save_collection_batch_config(
                path, **values
            ),
            preview_item=self._preview_interactive_collection_batch_item,
        )

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
        self.load_workflow_file(
            demo.workflow_path,
            preserve_batch_workspace=True,
        )
        preview_paths = self._load_collection_batch_demo_preview(demo)
        preview_names = ", ".join(path.name for path in preview_paths)
        self.status_label.setText(
            f"Opened deterministic batch demo at {demo.root}. "
            f"The graph is previewing the first paired item ({preview_names})."
        )
        return demo

    def _activate_interactive_collection_batch(
        self,
        items: Iterable[BatchItemPlan],
        config: BatchConfig,
        *,
        config_path: str | Path | None = None,
        initial_index: int = 0,
        force_sync: bool = False,
    ) -> None:
        """Activate a UI-only batch session and preview one paired item."""
        planned_items = tuple(items)
        if not planned_items:
            raise ValueError("The batch does not contain any input items.")
        index = int(initial_index)
        if not 0 <= index < len(planned_items):
            raise ValueError("The representative batch index is out of range.")
        representative_paths = {
            node_id: Path(path).expanduser().resolve()
            for node_id, path in planned_items[index].source_paths.items()
        }
        reuse_representative = bool(
            self._interactive_collection_batch_items
            and self._interactive_collection_batch_requested_index < 0
            and self._interactive_collection_batch_failed_index < 0
            and self._active_pipeline_run_id is None
            and self._active_source_load_id is None
            and not self._pipeline_run_pending
            and representative_paths == self._interactive_collection_source_paths
            and scientific_workflow_hash(self._batch_workflow_document())
            == config.workflow_sha256
        )
        if self._interactive_collection_batch_items and not reuse_representative:
            self._interactive_collection_source_paths.clear()
            self._prune_file_source_payload_cache()
        self._interactive_collection_batch_items = planned_items
        self._interactive_collection_batch_config = config
        self._interactive_collection_batch_config_path = (
            Path(config_path).expanduser().resolve()
            if config_path is not None
            else None
        )
        if not reuse_representative:
            self._interactive_collection_source_paths.clear()
        self._interactive_collection_batch_index = index if reuse_representative else -1
        self._interactive_collection_batch_requested_index = -1
        self._interactive_collection_batch_failed_index = -1
        self._interactive_collection_batch_plan_stale = False
        self._interactive_collection_batch_workflow_stale = False
        self.batch_navigator.set_session_stale(False)
        self.batch_navigator.reset_batch_progress()
        if reuse_representative:
            self._sync_interactive_collection_batch_navigator()
            return
        self._preview_interactive_collection_batch_item(
            index,
            force_sync=force_sync,
        )

    def _preview_interactive_collection_batch_item(
        self,
        index: int,
        *,
        force_sync: bool = False,
    ) -> bool:
        """Calculate one representative item through the complete live graph."""
        if self._collection_batch_running:
            self.status_label.setText(
                "The batch is running; representative navigation is temporarily "
                "disabled."
            )
            self._sync_interactive_collection_batch_navigator()
            return False
        items = self._interactive_collection_batch_items
        index = int(index)
        if not items or not 0 <= index < len(items):
            return False
        item = items[index]
        missing = sorted(set(item.source_paths) - set(self.pipeline.nodes))
        if missing:
            message = (
                "The active batch references Image Source nodes that are no "
                "longer present: " + ", ".join(missing) + "."
            )
            self._clear_interactive_collection_batch_session(
                close_workspace=False,
            )
            self.status_label.setText(message)
            active_dialog = self._active_collection_batch_dialog
            if active_dialog is not None:
                active_dialog.show_graph_preview_error(index, message)
            return False
        representative_paths = {
            node_id: Path(path).expanduser().resolve()
            for node_id, path in item.source_paths.items()
        }
        if not representative_paths:
            message = "The batch item did not resolve any collection Image Sources."
            self.status_label.setText(message)
            return False
        unavailable = [
            path for path in representative_paths.values() if not path.exists()
        ]
        if unavailable:
            message = "Representative source is no longer available: " + ", ".join(
                str(path) for path in unavailable
            )
            self._show_interactive_collection_batch_preview_error(index, message)
            return False

        if (
            index == self._interactive_collection_batch_index
            and self._interactive_collection_batch_requested_index < 0
            and self._interactive_collection_batch_failed_index != index
            and representative_paths == self._interactive_collection_source_paths
        ):
            self._sync_interactive_collection_batch_navigator()
            active_dialog = self._active_collection_batch_dialog
            if active_dialog is not None:
                active_dialog.select_preview_item(index)
                active_dialog.set_graph_preview_ready(index)
                active_dialog.set_representative_pending(False)
            return True
        if (
            index == self._interactive_collection_batch_requested_index
            and representative_paths == self._interactive_collection_source_paths
        ):
            active_dialog = self._active_collection_batch_dialog
            if active_dialog is not None:
                active_dialog.select_preview_item(index)
                active_dialog.set_graph_preview_loading(index)
                active_dialog.set_representative_pending(True)
            return True

        # Replace the complete paired mapping in one assignment. Downstream
        # execution can therefore never observe a half-switched source pair.
        self._interactive_collection_source_paths = representative_paths
        self._interactive_collection_batch_requested_index = index
        self._interactive_collection_batch_failed_index = -1
        self._sync_interactive_collection_batch_navigator(index=index)
        self.batch_navigator.show_representative_loading(
            f"Loading and calculating representative item {index + 1} of "
            f"{len(items)} ({item.batch_id}) through the graph..."
        )
        active_dialog = self._active_collection_batch_dialog
        if active_dialog is not None:
            active_dialog.select_preview_item(index)
            active_dialog.set_graph_preview_loading(index)
            active_dialog.set_representative_pending(True)
        self._invalidate_pipeline_cache()
        if self._selected_node_id in representative_paths:
            with self._preserve_interactive_collection_workflow_params():
                self._refresh_image_source_options()
        self.run_pipeline(
            force_sync=force_sync,
            manual_node_ids=self.pipeline.manual_node_ids(),
        )
        return True

    def _sync_interactive_collection_batch_navigator(
        self,
        *,
        index: int | None = None,
    ) -> None:
        items = self._interactive_collection_batch_items
        if index is None:
            index = self._interactive_collection_batch_index
        if not items or not 0 <= index < len(items):
            self.batch_navigator.clear_session()
            return
        item = items[index]
        config = self._interactive_collection_batch_config
        configured_titles = (
            {source.node_id: source.title for source in config.sources}
            if config is not None
            else {}
        )
        source_names: dict[str, str] = {}
        for node_id, path in item.source_paths.items():
            node = self.pipeline.nodes.get(node_id)
            if node is None:
                continue
            title = configured_titles.get(node_id, node.title)
            if title in source_names:
                title = f"{title} ({node_id})"
            source_names[title] = Path(path).name
        self.batch_navigator.set_session(
            len(items),
            index,
            item.batch_id,
            source_names,
        )

    def _complete_interactive_collection_batch_preview(self) -> None:
        """Commit only the latest representative whose calculation succeeded."""
        index = self._interactive_collection_batch_requested_index
        items = self._interactive_collection_batch_items
        if not items or not 0 <= index < len(items):
            return
        expected_paths = {
            node_id: Path(path).expanduser().resolve()
            for node_id, path in items[index].source_paths.items()
        }
        if expected_paths != self._interactive_collection_source_paths:
            return
        self._interactive_collection_batch_index = index
        self._interactive_collection_batch_requested_index = -1
        self._interactive_collection_batch_failed_index = -1
        self._sync_interactive_collection_batch_navigator()
        # The ordinary completion refresh ran while the prior representative
        # was still the committed one. Refresh card captions now that this
        # matching generation is accepted.
        self._update_thumbnails()
        if self._selected_node_id in expected_paths:
            with self._preserve_interactive_collection_workflow_params():
                self._refresh_image_source_options()
        dialog = self._active_collection_batch_dialog
        if dialog is not None:
            dialog.select_preview_item(index)
            dialog.set_graph_preview_ready(index)
            dialog.set_representative_pending(False)
        if self._interactive_collection_batch_plan_stale:
            message = (
                "The scientific workflow changed - this representative uses "
                "the previous source pairing through the edited graph. Preview "
                "batch again before running."
                if self._interactive_collection_batch_workflow_stale
                else None
            )
            self.batch_navigator.set_session_stale(True, message=message)
        else:
            self.batch_navigator.set_session_stale(False)
        item = items[index]
        self.status_label.setText(
            f"Representative batch item {index + 1}/{len(items)} "
            f"({item.batch_id}) is shown in the graph; the full batch has not "
            "been run or saved."
        )

    def _show_interactive_collection_batch_preview_error(
        self,
        index: int,
        message: str,
        *,
        graph_may_be_partial: bool = False,
    ) -> None:
        """Reject a requested representative without claiming it is displayed."""
        requested = self._interactive_collection_batch_requested_index
        if requested < 0 and int(index) < 0:
            return
        failed_index = int(index if requested < 0 else requested)
        self._interactive_collection_batch_requested_index = -1
        committed = self._interactive_collection_batch_index
        items = self._interactive_collection_batch_items
        if graph_may_be_partial and items and 0 <= failed_index < len(items):
            self._interactive_collection_batch_index = failed_index
            self._interactive_collection_batch_failed_index = failed_index
            self._sync_interactive_collection_batch_navigator()
        elif items and 0 <= committed < len(items):
            self._interactive_collection_source_paths = {
                node_id: Path(path).expanduser().resolve()
                for node_id, path in items[committed].source_paths.items()
            }
            self._interactive_collection_batch_failed_index = -1
            self._sync_interactive_collection_batch_navigator()
        elif items and 0 <= failed_index < len(items):
            self._interactive_collection_source_paths.clear()
            self._interactive_collection_batch_failed_index = -1
            self._sync_interactive_collection_batch_navigator(index=failed_index)
        self.batch_navigator.show_representative_error(message)
        dialog = self._active_collection_batch_dialog
        if dialog is not None:
            dialog.show_graph_preview_error(failed_index, message)
            representative_ready = bool(
                items
                and 0 <= self._interactive_collection_batch_index < len(items)
                and self._interactive_collection_batch_failed_index < 0
            )
            dialog.set_representative_pending(not representative_ready)
        self.status_label.setText(f"Representative preview failed: {message}")

    def _mark_interactive_collection_batch_stale(
        self,
        dialog: CollectionBatchDialog,
    ) -> None:
        """Require a fresh plan while retaining the last representative view."""
        if dialog is not self._active_collection_batch_dialog:
            return
        self._interactive_collection_batch_config_path = None
        self._interactive_collection_batch_plan_stale = True
        if self._interactive_collection_batch_items:
            self.batch_navigator.set_session_stale(True)

    def _mark_collection_batch_workflow_stale_if_needed(self) -> None:
        """Invalidate a retained runnable plan after a scientific graph edit."""
        config = self._interactive_collection_batch_config
        if (
            config is None
            or not self._interactive_collection_batch_items
            or self._interactive_collection_batch_workflow_stale
        ):
            return
        if scientific_workflow_hash(self._batch_workflow_document()) == (
            config.workflow_sha256
        ):
            return
        self._interactive_collection_batch_config_path = None
        self._interactive_collection_batch_plan_stale = True
        self._interactive_collection_batch_workflow_stale = True
        dialog = self._active_collection_batch_dialog
        if dialog is not None:
            dialog.invalidate_for_workflow_change()
        self.batch_navigator.set_session_stale(
            True,
            message=(
                "The scientific workflow changed - representative navigation "
                "still uses the previous source pairing through the edited "
                "graph. Preview batch again before running."
            ),
        )

    def _clear_interactive_collection_batch_session(
        self,
        *,
        close_workspace: bool = True,
    ) -> None:
        """Clear representative paths and all UI-only batch session state."""
        if self._collection_batch_running:
            close_workspace = False
        self._interactive_collection_source_paths.clear()
        self._prune_file_source_payload_cache()
        self._interactive_collection_batch_items = ()
        self._interactive_collection_batch_config = None
        self._interactive_collection_batch_config_path = None
        self._interactive_collection_batch_index = -1
        self._interactive_collection_batch_requested_index = -1
        self._interactive_collection_batch_failed_index = -1
        self._interactive_collection_batch_plan_stale = False
        self._interactive_collection_batch_workflow_stale = False
        self.batch_navigator.clear_session()
        if self._active_source_load_id is not None:
            self._source_load_serial += 1
            self._active_source_load_id = None
            self._source_load_pending = False
            self._set_pipeline_busy(False)
        if close_workspace and self._active_collection_batch_dialog is not None:
            self._discard_collection_batch_dialog(
                self._active_collection_batch_dialog,
            )

    def _open_active_collection_batch_workspace(self) -> None:
        """Open or focus the workspace associated with the current session."""
        dialog = self._active_collection_batch_dialog
        if dialog is not None:
            dialog.show()
            dialog.raise_()
            dialog.activateWindow()
            return
        self._batch_collection_dialog(
            config_path=self._interactive_collection_batch_config_path,
            config=self._interactive_collection_batch_config,
        )

    def _load_collection_batch_demo_preview(
        self,
        demo: SyntheticBatchDemo,
    ) -> tuple[Path, ...]:
        """Activate the demo plan and calculate its first representative pair."""
        config = load_batch_config(demo.config_path)
        workflow = self._batch_workflow_document()
        plan = plan_batch(
            workflow,
            config,
            workflow_path=demo.workflow_path,
        )
        if not plan.items:
            raise ValueError("The batch demo does not contain any input items.")
        self._activate_interactive_collection_batch(
            plan.items,
            config,
            config_path=demo.config_path,
            initial_index=0,
            force_sync=True,
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
        expected_items: tuple[BatchItemPlan, ...] | None = None,
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
        if expected_items is not None and plan.items != tuple(expected_items):
            raise RuntimeError(
                "The batch plan changed after it was reviewed. No batch item "
                "was run; preview the batch again before retrying."
            )
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
        normalized_status = str(status).strip().casefold()
        completed = index - 1 if normalized_status == "running" else index
        self.pipeline_busy_bar.setRange(0, max(int(total), 1))
        self.pipeline_busy_bar.setValue(max(0, min(int(completed), int(total))))
        self.pipeline_busy_bar.setTextVisible(True)
        self.pipeline_busy_bar.setFormat("%v/%m")
        self.pipeline_busy_label.setText(
            f"Batch {index}/{total}: {batch_id} ({status})"
        )
        if self._interactive_collection_batch_items:
            self.batch_navigator.update_batch_progress(
                index,
                total,
                batch_id,
                status,
            )
        dialog = self._active_collection_batch_dialog
        if dialog is not None:
            dialog.update_run_progress(index, total, batch_id, status)
        self.status_label.setText(f"Batch {index}/{total}: {batch_id} ({status}).")
        QApplication.processEvents(QEventLoop.ExcludeUserInputEvents)

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
        return self._collection_batch_controller.build_config(
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

    def _save_collection_batch_config(
        self,
        path: str | Path,
        **values,
    ) -> tuple[Path, Path]:
        return self._collection_batch_controller.save_config(path, **values)

    def _load_collection_batch_config(self, path: str | Path) -> BatchConfig:
        return self._collection_batch_controller.load_config(path)

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
        result = self._collection_batch_controller.preview(
            input_dir=input_dir,
            output_dir=output_dir,
            pattern=pattern,
            image_format=image_format,
            save_workflow_snapshot=save_workflow_snapshot,
            save_python_script=save_python_script,
            source_bindings=source_bindings,
            preview_limit=preview_limit,
            existing_file_policy=existing_file_policy,
            continue_on_error=continue_on_error,
        )
        config_path = None
        if self._active_collection_batch_dialog is not None:
            config_path = self._active_collection_batch_dialog._loaded_config_path
        if result.items and result.config is not None:
            self._activate_interactive_collection_batch(
                result.items,
                result.config,
                config_path=config_path,
            )
        return result

    def _batch_source_rows(self) -> list[dict[str, str]]:
        return self._collection_batch_controller.source_rows()

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
        items = self._interactive_collection_batch_items
        representative_index = self._interactive_collection_batch_requested_index
        if not 0 <= representative_index < len(items):
            representative_index = self._interactive_collection_batch_index
        if items and 0 <= representative_index < len(items):
            item = items[representative_index]
            self._interactive_collection_batch_index = -1
            self._interactive_collection_batch_requested_index = representative_index
            self._interactive_collection_batch_failed_index = -1
            self._interactive_collection_batch_plan_stale = True
            self._sync_interactive_collection_batch_navigator(
                index=representative_index,
            )
            self.batch_navigator.show_representative_loading(
                f"Refreshing and calculating representative item "
                f"{representative_index + 1} of {len(items)} ({item.batch_id}) "
                "through the graph..."
            )
            dialog = self._active_collection_batch_dialog
            if dialog is not None:
                dialog.begin_representative_source_refresh(
                    representative_index,
                    len(items),
                    item.batch_id,
                )
        self._clear_file_source_snapshots(invalidate_inflight=True)
        refreshed_layer_ids: set[int] = set()
        for layer in self._live_source_node_layers.values():
            if id(layer) in refreshed_layer_ids:
                continue
            refreshed_layer_ids.add(id(layer))
            self._live_source_adapter.invalidate(layer, notify=False)
        self._invalidate_pipeline_cache()
        self._autobind_default_image_sources()
        self._refresh_image_source_controls()
        self.run_pipeline()

    def _clear_file_source_snapshots(self, *, invalidate_inflight: bool) -> None:
        """Drop pinned file revisions; optionally make all older loads stale."""
        self._file_source_payload_cache.clear()
        self._file_source_path_identities.clear()
        self._source_inspection_cache.clear()
        if not invalidate_inflight:
            return
        self._source_load_serial += 1
        self._active_source_load_id = None
        self._source_load_pending = False

    def _mark_pipeline_dirty(self, node_id: str) -> bool:
        if node_id == self._isolated_tuning_node_id:
            return self._mark_isolated_tuning_dirty(node_id)
        return self._mark_pipeline_branches_dirty({node_id})

    def _mark_pipeline_branches_dirty(self, node_ids) -> bool:
        valid_node_ids = {
            str(node_id) for node_id in node_ids if str(node_id) in self.pipeline.nodes
        }
        if not valid_node_ids:
            return False
        active_node_id = self._isolated_tuning_node_id
        if active_node_id is not None and valid_node_ids != {active_node_id}:
            self._apply_isolated_tuning(run=False, announce=False)
        affected_node_ids = self.pipeline.descendants_inclusive(valid_node_ids)
        cleared_overrides = self._discard_background_node_result_overrides(
            affected_node_ids
        )
        self._clear_colocalization_scatter_cache()
        self._pending_dirty_node_ids.update(valid_node_ids)
        self.pipeline.mark_manual_descendants_stale(valid_node_ids)
        self._sync_execution_ui()
        self._refresh_node_presentation_surfaces(cleared_overrides)
        self._mark_collection_batch_workflow_stale_if_needed()
        return True

    def _mark_isolated_tuning_dirty(self, node_id: str) -> bool:
        if node_id not in self.pipeline.nodes:
            return False
        self._clear_colocalization_scatter_cache()
        self._pending_dirty_node_ids.add(node_id)
        self.pipeline.mark_nodes_stale(
            {node_id},
            message=(
                "Parameters changed. Recalculating this node while downstream "
                "propagation is paused."
            ),
        )
        descendants = self.pipeline.descendants_inclusive({node_id}) - {node_id}
        cleared_overrides = self._discard_background_node_result_overrides(
            descendants | {node_id}
        )
        self.pipeline.mark_nodes_blocked(
            descendants,
            message=(
                "Downstream result is stale because propagation is paused while "
                f"'{self._node_title(node_id)}' is tuned."
            ),
        )
        self._isolated_tuning_has_changes = True
        self._sync_execution_ui()
        self._refresh_node_presentation_surfaces(cleared_overrides)
        self._mark_collection_batch_workflow_stale_if_needed()
        return True

    def _on_isolated_tuning_toggled(self, checked: bool) -> None:
        node_id = self._selected_node_id
        if checked:
            self._start_isolated_tuning(node_id)
        elif node_id == self._isolated_tuning_node_id:
            self._apply_isolated_tuning()
        else:
            self._sync_isolated_tuning_ui()

    def _toggle_node_isolation_from_graph(self, node_id: str) -> None:
        if node_id == self._isolated_tuning_node_id:
            self._apply_isolated_tuning()
            return
        self._start_isolated_tuning(node_id)

    def _start_isolated_tuning(self, node_id: str) -> bool:
        if node_id not in self.pipeline.nodes:
            self._sync_isolated_tuning_ui()
            return False
        if self._isolated_tuning_node_id is not None:
            self.status_label.setText(
                "Finish the current isolated tuning session with Apply and "
                "continue or Cancel tuning first."
            )
            self._sync_isolated_tuning_ui()
            return False
        descendants = self.pipeline.descendants_inclusive({node_id}) - {node_id}
        if not descendants:
            self.status_label.setText(
                f"'{self._node_title(node_id)}' has no downstream nodes to pause."
            )
            self._sync_isolated_tuning_ui()
            return False
        if (
            self._active_pipeline_run_id is not None
            or self._active_source_load_id is not None
            or self._collection_batch_running
        ):
            self.status_label.setText(
                "Wait for the current calculation or source load to finish before "
                "starting isolated tuning."
            )
            self._sync_isolated_tuning_ui()
            return False
        if (
            self._last_pipeline_source_signature is None
            or self._pending_dirty_node_ids
            or self._inflight_dirty_node_ids is not None
            or not self.pipeline._has_cached_output(node_id)
        ):
            self.status_label.setText(
                "Calculate the current graph before starting isolated tuning so "
                "Cancel tuning has a coherent result to restore."
            )
            self._sync_isolated_tuning_ui()
            return False

        self._finish_parameter_history_group()
        node = self.pipeline.nodes[node_id]
        self._isolated_tuning_snapshot = _IsolatedTuningSnapshot(
            node_id=node_id,
            params=deepcopy(node.params),
            output=self.pipeline.outputs.get(node_id),
            output_state=self.pipeline.output_states.get(node_id),
            node_outputs=tuple(self.pipeline.node_outputs.get(node_id, ())),
            node_output_states=tuple(
                self.pipeline.node_output_states.get(node_id, ())
            ),
            execution_states=dict(self.pipeline.node_execution_states),
            execution_messages=dict(self.pipeline.node_execution_messages),
            completed_node_ids=frozenset(self.pipeline.completed_node_ids),
            undo_stack=tuple(self._history.undo_stack),
            redo_stack=tuple(self._history.redo_stack),
            batch_workflow_stale=self._interactive_collection_batch_workflow_stale,
        )
        self._isolated_tuning_node_id = node_id
        self._isolated_tuning_has_changes = False
        self._sync_isolated_tuning_ui()
        self.status_label.setText(
            f"Tuning '{self._node_title(node_id)}' in isolation. Downstream "
            "propagation will pause after the first parameter change."
        )
        return True

    def _apply_isolated_tuning(
        self,
        _checked: bool = False,
        *,
        run: bool = True,
        announce: bool = True,
    ) -> bool:
        node_id = self._isolated_tuning_node_id
        if node_id is None:
            self._sync_isolated_tuning_ui()
            return False
        self._debounce_timer.stop()
        had_changes = self._isolated_tuning_has_changes
        self._finish_parameter_history_group(preserve_isolated_tuning=True)
        self._isolated_tuning_node_id = None
        self._isolated_tuning_snapshot = None
        self._isolated_tuning_has_changes = False

        dirty_targets: set[str] = set()
        if had_changes and node_id in self.pipeline.nodes:
            state = self.pipeline.node_execution_states.get(node_id)
            isolated_result_in_flight = bool(
                self._active_pipeline_run_id is not None
                and self._inflight_dirty_node_ids is not None
                and node_id in self._inflight_dirty_node_ids
            )
            latest_result_available = bool(
                state == EXECUTION_READY or isolated_result_in_flight
            )
            if (
                not latest_result_available
                or not self.pipeline._has_cached_output(node_id)
            ):
                dirty_targets.add(node_id)
            else:
                dirty_targets.update(
                    connection.target_id
                    for connection in self.pipeline.connections
                    if connection.source_id == node_id
                )
                if not dirty_targets:
                    dirty_targets.add(node_id)
            self._pending_dirty_node_ids.update(dirty_targets)
            self.pipeline.mark_manual_descendants_stale(dirty_targets)
            self._mark_collection_batch_workflow_stale_if_needed()

        self._sync_execution_ui()
        if announce:
            self.status_label.setText(
                (
                    f"Applying the latest '{self._node_title(node_id)}' result "
                    "and continuing downstream..."
                )
                if dirty_targets
                else "Isolated tuning ended; no parameter changes to apply."
            )
        if run and dirty_targets:
            self.run_pipeline()
        return had_changes

    def _cancel_isolated_tuning(
        self,
        _checked: bool = False,
        *,
        announce: bool = True,
    ) -> bool:
        snapshot = self._isolated_tuning_snapshot
        node_id = self._isolated_tuning_node_id
        if snapshot is None or node_id is None:
            self._sync_isolated_tuning_ui()
            return False
        self._debounce_timer.stop()
        if self._active_pipeline_run_id is not None:
            self._abandon_background_pipeline_run()
        self._pending_dirty_node_ids.discard(node_id)
        self._pending_manual_node_ids.discard(node_id)
        node = self.pipeline.nodes.get(node_id)
        if node is not None:
            node.params = deepcopy(snapshot.params)
            self.pipeline.outputs[node_id] = snapshot.output
            self.pipeline.output_states[node_id] = snapshot.output_state
            self.pipeline.node_outputs[node_id] = list(snapshot.node_outputs)
            self.pipeline.node_output_states[node_id] = list(
                snapshot.node_output_states
            )
        self.pipeline.node_execution_states = dict(snapshot.execution_states)
        self.pipeline.node_execution_messages = dict(snapshot.execution_messages)
        self.pipeline.completed_node_ids = set(snapshot.completed_node_ids)
        self._history.undo_stack[:] = snapshot.undo_stack
        self._history.redo_stack[:] = snapshot.redo_stack
        self._history.finish_group()
        self._interactive_collection_batch_workflow_stale = (
            snapshot.batch_workflow_stale
        )
        self._isolated_tuning_node_id = None
        self._isolated_tuning_snapshot = None
        self._isolated_tuning_has_changes = False
        if node is not None:
            self._sync_node_output_ports(node_id)
            if self._selected_node_id == node_id:
                self._render_parameters(node_id)
        self._sync_history_actions()
        self._update_thumbnails()
        self._inspect_selected_node()
        self._update_metadata_panel()
        self._update_histogram()
        self._sync_execution_ui()
        self._refresh_cache_status()
        if announce:
            self.status_label.setText(
                f"Canceled isolated tuning for '{self._node_title(node_id)}'; "
                "restored its original parameters and cached result."
            )
        return True

    def _sync_isolated_tuning_ui(self) -> None:
        active_node_id = self._isolated_tuning_node_id
        if active_node_id not in self.pipeline.nodes:
            active_node_id = None
            self._isolated_tuning_node_id = None
            self._isolated_tuning_snapshot = None
            self._isolated_tuning_has_changes = False
        selected_node_id = self._selected_node_id
        has_downstream = bool(
            selected_node_id in self.pipeline.nodes
            and (
                self.pipeline.descendants_inclusive({selected_node_id})
                - {selected_node_id}
            )
        )
        with QSignalBlocker(self.isolated_tuning_checkbox):
            self.isolated_tuning_checkbox.setChecked(
                active_node_id is not None and selected_node_id == active_node_id
            )
        self.isolated_tuning_checkbox.setEnabled(
            bool(
                selected_node_id in self.pipeline.nodes
                and (
                    selected_node_id == active_node_id
                    or (active_node_id is None and has_downstream)
                )
            )
        )
        if active_node_id is not None and selected_node_id != active_node_id:
            self.isolated_tuning_checkbox.setToolTip(
                f"'{self._node_title(active_node_id)}' is already being tuned. "
                "Apply or cancel that session before isolating another node."
            )
        else:
            self.isolated_tuning_checkbox.setToolTip(
                "Recalculate only this node while you tune its parameters. "
                "Downstream nodes stay stale until Apply and continue."
            )
        self.isolated_tuning_panel.setVisible(active_node_id is not None)
        if active_node_id is not None:
            state, _message = self._node_execution_ui_state(active_node_id)
            result_in_flight = bool(
                self._active_pipeline_run_id is not None
                and self._inflight_dirty_node_ids is not None
                and active_node_id in self._inflight_dirty_node_ids
            )
            if not self._isolated_tuning_has_changes:
                suffix = " Change a parameter to begin local recalculation."
            elif result_in_flight or state == EXECUTION_RUNNING:
                suffix = " Recalculating this node; downstream remains held."
            elif state == EXECUTION_ERROR:
                suffix = " Local calculation failed; fix the node or cancel tuning."
            elif state == EXECUTION_READY:
                suffix = " Latest local result is ready to apply."
            else:
                suffix = " Local result is waiting to be recalculated."
            self.isolated_tuning_status.setText(
                f"Downstream paused after '{self._node_title(active_node_id)}'."
                f"{suffix}"
            )
        self.graph_view.set_isolated_tuning_node(active_node_id)

    def _invalidate_pipeline_cache(self) -> None:
        if self._isolated_tuning_node_id is not None:
            self._apply_isolated_tuning(run=False, announce=False)
        cleared_overrides = self._discard_background_node_result_overrides()
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
        self._refresh_node_presentation_surfaces(cleared_overrides)

    def _begin_pipeline_dispatch(self, dirty_node_ids: set[str] | None) -> None:
        """Mark a run as in flight and clear the dirty nodes it covers.

        Removing the dispatched dirty nodes from ``_pending_dirty_node_ids`` here
        (rather than on completion) means any edit that re-marks the same node
        while the run is in flight is preserved as genuinely pending work, and a
        later debounced run will not see an empty pending set and recompute the
        whole pipeline from the source.
        """
        cleared_overrides = self._discard_background_node_result_overrides()
        self._inflight_dirty_node_ids = (
            None if dirty_node_ids is None else set(dirty_node_ids)
        )
        if dirty_node_ids is not None:
            self._pending_dirty_node_ids.difference_update(dirty_node_ids)
        self._refresh_node_presentation_surfaces(cleared_overrides)

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
                            (
                                "revision",
                                payload.revision_token.layer_id,
                                payload.revision_token.revision,
                            )
                            if isinstance(
                                payload.revision_token,
                                SourceRevisionToken,
                            )
                            else (
                                "file-snapshot",
                                payload.revision_token.kind,
                                payload.revision_token.sha256,
                                payload.revision_token.regular_file_count,
                                payload.revision_token.size_bytes,
                            )
                            if isinstance(
                                payload.revision_token,
                                LocalSourceIdentity,
                            )
                            else (
                                "object",
                                id(payload.data),
                                id(payload.metadata),
                            ),
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

    def _on_port_label_mode_changed(self, mode: str) -> None:
        self.graph_view.set_port_label_mode(mode)
        overlaps = self.graph_view.overlapping_node_pairs()
        if overlaps:
            suffix = "pair overlaps" if len(overlaps) == 1 else "pairs overlap"
            self.status_label.setText(
                f"Port labels set to {mode}; {len(overlaps)} node {suffix}. "
                "Use Auto structure graph to create space."
            )
            return
        self.status_label.setText(f"Port labels set to {mode}.")

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
            # Manual results are explicitly requested and often expensive.
            # A downstream display edit must not silently discard them merely
            # because an intermediate node makes them more than one edge away
            # from the currently selected node.
            | self.pipeline.manual_node_ids()
            # A blocked branch is one coherent cached snapshot. Keep it intact
            # until its actionable upstream manual barrier is recalculated.
            | {
                node_id
                for node_id, state in self.pipeline.node_execution_states.items()
                if state == EXECUTION_BLOCKED
            }
        )
        if self._isolated_tuning_node_id in self.pipeline.nodes:
            retained.update(
                self.pipeline.descendants_inclusive(
                    {str(self._isolated_tuning_node_id)}
                )
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
        if self._closing:
            return
        changed_node_ids = self._autobind_default_image_sources()
        changed_node_ids.update(self._changed_live_source_bindings())
        self._refresh_image_source_controls()
        if changed_node_ids and self._mark_pipeline_branches_dirty(changed_node_ids):
            self.run_pipeline()

    def _changed_live_source_bindings(self) -> set[str]:
        changed: set[str] = set()
        for node_id, node in self.pipeline.nodes.items():
            if node.operation_id != "input":
                continue
            if str(node.params.get("source_mode", "napari layer")) != "napari layer":
                continue
            layer_name = str(node.params.get("layer_name", "")).strip()
            if not layer_name:
                layer_name = self._default_input_layer_name()
            current = self._layer_by_name(layer_name) if layer_name else None
            previous = self._live_source_node_layers.get(node_id)
            if current is not previous:
                changed.add(node_id)
        return changed

    def _on_live_source_invalidated(self, layer) -> None:
        if self._closing:
            return
        node_ids = {
            node_id
            for node_id, bound_layer in self._live_source_node_layers.items()
            if bound_layer is layer and node_id in self.pipeline.nodes
        }
        if not node_ids or not self._mark_pipeline_branches_dirty(node_ids):
            return
        self._clear_input_histogram_cache()
        self._clear_output_histogram_cache()
        self._label_volume_cache.clear()
        self._clear_generated_layer_contrast_state()
        active_run_id = self._active_pipeline_run_id
        if active_run_id is not None:
            self._pipeline_run_pending = True
            active_node_id = self._active_pipeline_node_id
            if self._dirty_nodes_affect_node(node_ids, active_node_id):
                cancel_event = self._pipeline_cancel_events.get(active_run_id)
                if cancel_event is not None:
                    cancel_event.set()
            self._set_pipeline_busy(
                True,
                active_node_id,
                queued=True,
            )
        self.status_label.setText(
            "Live source changed; queued a calculation from the new revision."
        )
        self._debounce_timer.start()

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
        self._prune_file_source_payload_cache()
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
            resolved_path = str(key[0])
            specs.append(
                SourceFileLoadSpec(
                    node_id=node_id,
                    path=resolved_path,
                    series_index=int(node.params.get("series_index", 0) or 0),
                    cache_key=key,
                    expected_identity=self._file_source_path_identities.get(
                        resolved_path
                    ),
                )
            )
        return tuple(specs)

    def _file_source_should_load_async(self, node) -> bool:
        source_path = self._file_source_path_for_node(node)
        if source_path is None:
            return False
        suffix = source_path.suffix.lower()
        if suffix == ".zarr" or suffix in MICROSCOPE_SUFFIXES:
            return True
        try:
            return source_path.stat().st_size >= ASYNC_SOURCE_FILE_BYTES
        except OSError:
            return False

    def _file_source_cache_key(self, node) -> tuple[object, ...] | None:
        source_path = self._file_source_path_for_node(node)
        if source_path is None:
            return None
        return (
            str(source_path),
            int(node.params.get("series_index", 0) or 0),
        )

    def _file_source_path_for_node(self, node) -> Path | None:
        representative = self._interactive_collection_source_paths.get(node.id)
        if representative is not None:
            path_text = str(representative)
        elif str(node.params.get("source_mode", "")) == "file path":
            path_text = str(node.params.get("file_path", "")).strip()
        else:
            return None
        if not path_text:
            return None
        return Path(path_text).expanduser().resolve(strict=False)

    def _cached_file_source_payload(self, node) -> SourcePayload | None:
        key = self._file_source_cache_key(node)
        if key is None:
            return None
        snapshot = self._file_source_payload_cache.get(key)
        if snapshot is None:
            return None
        return self._viewer_aligned_source_payload(snapshot.payload)

    def _cache_file_source_snapshot(
        self,
        key: tuple[object, ...],
        snapshot: SourceFileSnapshot,
    ) -> None:
        resolved_path = str(key[0])
        pinned_identity = self._file_source_path_identities.get(resolved_path)
        if pinned_identity is not None and pinned_identity != snapshot.identity:
            raise SourceChangedError(
                "Local scientific source changed after its interactive snapshot "
                f"was pinned: {resolved_path}. Press Refresh to load the new "
                "revision."
            )
        cached_inspection = self._source_inspection_cache.get(resolved_path)
        if (
            cached_inspection is not None
            and cached_inspection.identity != snapshot.identity
        ):
            raise SourceChangedError(
                "Local scientific source no longer matches the revision that "
                f"was inspected: {resolved_path}. Press Refresh to inspect and "
                "load the new revision."
            )
        self._file_source_payload_cache[key] = snapshot
        self._file_source_path_identities[resolved_path] = snapshot.identity
        self._source_inspection_cache[resolved_path] = VerifiedSourceInspection(
            snapshot.inspection,
            snapshot.identity,
        )

    def _prune_file_source_payload_cache(self) -> None:
        """Bound representative arrays while retaining pinned path identities.

        Outside a collection session, materialized snapshots remain pinned until
        Refresh. During slider browsing, only the active paired arrays (plus
        fixed non-batch sources) stay materialized; identity records for every
        visited path remain pinned so a changed file is still rejected later.
        """
        items = self._interactive_collection_batch_items
        if not items:
            return
        batch_paths = {
            str(Path(path).expanduser().resolve())
            for item in items
            for path in item.source_paths.values()
        }
        retained_keys: set[tuple[object, ...]] = set()
        for node in self.pipeline.nodes.values():
            if node.operation_id != "input":
                continue
            key = self._file_source_cache_key(node)
            if key is None:
                continue
            if (
                node.id in self._interactive_collection_source_paths
                or str(key[0]) not in batch_paths
            ):
                retained_keys.add(key)
        for key in tuple(self._file_source_payload_cache):
            if str(key[0]) in batch_paths and key not in retained_keys:
                self._file_source_payload_cache.pop(key, None)

    def _source_payloads_for_pipeline(
        self,
    ) -> tuple[dict[str, SourcePayload], list[object]]:
        payloads: dict[str, SourcePayload] = {}
        layers: list[object] = []
        live_bindings: dict[str, object] = {}
        for node_id, node in self.pipeline.nodes.items():
            if node.operation_id != "input":
                continue
            payload, layer = self._resolve_source_payload(node)
            if payload is not None:
                payloads[node_id] = payload
            if layer is not None:
                layers.append(layer)
                live_bindings[node_id] = layer
        self._live_source_adapter.sync_layers(layers)
        self._live_source_node_layers = live_bindings
        return payloads, layers

    def _resolve_source_payload(
        self,
        node,
    ) -> tuple[SourcePayload | None, object | None]:
        mode = str(node.params.get("source_mode", "napari layer"))
        if (
            mode == "file path"
            or node.id in self._interactive_collection_source_paths
        ):
            source_path = self._file_source_path_for_node(node)
            if source_path is None:
                return SourcePayload(None, {}, ""), None
            key = self._file_source_cache_key(node)
            if key is None:
                return SourcePayload(None, {}, ""), None
            cached = self._cached_file_source_payload(node)
            if cached is not None:
                return cached, None
            if self._file_source_should_load_async(node):
                return None, None
            resolved_path = str(source_path)
            snapshot = load_frozen_file_source_snapshot(
                source_path,
                int(node.params.get("series_index", 0)),
                expected_identity=self._file_source_path_identities.get(
                    resolved_path
                ),
                reader=read_image,
            )
            self._cache_file_source_snapshot(key, snapshot)
            self._prune_file_source_payload_cache()
            return self._viewer_aligned_source_payload(snapshot.payload), None
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
        snapshot = self._live_source_adapter.snapshot(layer)
        if not snapshot.data_is_detached:
            raise ValueError(
                f"Live napari source '{snapshot.name}' uses "
                f"{type(snapshot.data).__name__} data that VIPP cannot detach "
                "into a stable scientific revision. Materialize it as a NumPy "
                "layer or use an immutable file source before calculating."
            )
        return (
            SourcePayload(
                snapshot.data,
                snapshot.metadata,
                snapshot.name,
                self._viewer_aligned_live_layer_state(snapshot),
                snapshot.token,
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
        original_rescale_auto_ranges = dict(self._rescale_auto_output_ranges)
        try:
            yield
        finally:
            for node_id, params in original_params.items():
                node = self.pipeline.nodes.get(node_id)
                if node is not None:
                    node.params = params
            self._rescale_auto_output_ranges = original_rescale_auto_ranges

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
            payload.revision_token,
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

    def _viewer_aligned_live_layer_state(self, snapshot: LiveLayerSnapshot):
        state = image_state_from_array(
            snapshot.data,
            layer_metadata=snapshot.metadata,
            source_name=snapshot.name,
            defer_statistics=_should_auto_background_data(snapshot.data),
        )
        state = apply_live_layer_axis_transform(state, snapshot)
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
        source_path = Path(path).expanduser().resolve(strict=False)
        cache_key = str(source_path)
        cached = self._source_inspection_cache.get(cache_key)
        if cached is not None:
            return cached.inspection
        if not source_path.exists():
            return None
        identity = capture_local_source_identity(source_path)
        inspection = inspect_image_source(source_path)
        verify_local_source_identity(source_path, identity)
        pinned_identity = self._file_source_path_identities.get(cache_key)
        if pinned_identity is not None and pinned_identity != identity:
            raise SourceChangedError(
                "Local scientific source no longer matches its pinned revision: "
                f"{source_path}. Press Refresh to inspect the new revision."
            )
        self._file_source_path_identities[cache_key] = identity
        self._source_inspection_cache[cache_key] = VerifiedSourceInspection(
            inspection,
            identity,
        )
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
        affected = {node_id}
        if result.connection is not None:
            affected.add(result.connection.source_id)
        if self._selected_node_id in affected:
            self._refresh_selected_parameter_controls()
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
            if self._selected_node_id in {node_id, connection.source_id}:
                self._refresh_selected_parameter_controls()
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
        if self._selected_node_id in {source_id, target_id}:
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
            if self._selected_node_id in {source_id, target_id}:
                self._refresh_selected_parameter_controls()
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
        if node_id == self._isolated_tuning_node_id:
            self._apply_isolated_tuning(run=False, announce=False)
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
        if node.operation_id == "input" and (
            self._interactive_collection_batch_items
            or self._active_collection_batch_dialog is not None
        ):
            self._clear_interactive_collection_batch_session()
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
        self._discard_background_node_result_overrides({node_id})
        if self._active_pinned_node_id == node_id:
            self._clear_active_pin(status=False)
        if self._selected_node_id == node_id:
            self._select_first_available_node()
        if self._mark_pipeline_branches_dirty(dirty_targets):
            self.run_pipeline()
        self._sync_execution_ui()
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
        self._sync_isolated_tuning_ui()

    def _select_node(self, node_id: str) -> None:
        if node_id not in self.pipeline.nodes:
            return
        self._selected_node_id = node_id
        # Recompute at a deliberate selection boundary. Subsequent inspector
        # repaints reuse the result while the resolved input identities and
        # relevant metadata remain unchanged.
        self._psf_preflight_cache.pop(node_id, None)
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
        self._sync_isolated_tuning_ui()

    def _restore_selected_output_for_interactive_cache(self, node_id: str) -> None:
        if self._workflow_load_selection_in_progress:
            return
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
        had_isolation = self._isolated_tuning_node_id is not None
        if had_isolation:
            self._apply_isolated_tuning(run=False, announce=False)
        node_ids = self._manual_node_ids_needing_calculation()
        has_dirty_work = bool(
            self._pending_dirty_node_ids & set(self.pipeline.nodes)
        )
        if not node_ids and not has_dirty_work:
            self.status_label.setText(
                "Isolated tuning disabled; no nodes need calculation."
                if had_isolation
                else "No manual nodes need calculation."
            )
            return
        for node_id in node_ids:
            self.pipeline.node_execution_states[node_id] = EXECUTION_RUNNING
            self.pipeline.node_execution_messages[node_id] = ""
        self._sync_execution_ui()
        if had_isolation:
            self.status_label.setText(
                "Isolated tuning disabled; applying the latest result and "
                "calculating downstream nodes..."
            )
        else:
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

    def _manual_node_ids_requiring_attention(self) -> set[str]:
        """Return actionable manual frontiers represented in bright amber."""
        return {
            node_id
            for node_id in self.pipeline.manual_node_ids()
            if self.pipeline.node_execution_states.get(
                node_id,
                EXECUTION_NOT_CALCULATED,
            )
            in {EXECUTION_NOT_CALCULATED, EXECUTION_STALE}
            and not self.pipeline.node_auto_recalculate(node_id)
        }

    def _sync_calculate_all_attention(self) -> None:
        attention_required = bool(
            self._manual_node_ids_requiring_attention()
        )
        current = self.calculate_all_button.property("attentionRequired")
        if current is not None and bool(current) == attention_required:
            return
        self.calculate_all_button.setProperty(
            "attentionRequired",
            attention_required,
        )
        self.calculate_all_button.setStyleSheet(
            CALCULATE_ALL_ATTENTION_STYLE if attention_required else ""
        )
        self.calculate_all_button.setToolTip(
            (
                "Manual nodes need calculation. Calculate every manual node "
                "that is not current."
            )
            if attention_required
            else "Calculate every manual node that is not current."
        )

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
            state, message = self._node_execution_ui_state(node_id)
            auto_recalculate = self.pipeline.node_auto_recalculate(node_id)
            self.graph_view.set_node_execution_state(
                node_id,
                state,
                manual=manual,
                message=message,
                auto_recalculate=auto_recalculate,
            )

        self._sync_calculate_all_attention()
        self._sync_isolated_tuning_ui()

        node_id = self._selected_node_id
        if node_id not in self.pipeline.nodes or not self.pipeline.is_manual_node(
            node_id
        ):
            self.execution_group.setHidden(True)
            return
        state, message = self._node_execution_ui_state(node_id)
        auto_recalculate = self.pipeline.node_auto_recalculate(node_id)
        self.execution_group.setHidden(False)
        self.execution_status_label.setText(
            self._execution_status_text(state, message),
        )
        with QSignalBlocker(self.auto_recalculate_checkbox):
            self.auto_recalculate_checkbox.setChecked(auto_recalculate)
        self.calculate_button.setEnabled(
            state not in {EXECUTION_RUNNING, EXECUTION_BLOCKED}
        )
        self.calculate_button.setHidden(auto_recalculate)
        if state == EXECUTION_BLOCKED:
            self.calculate_button.setText("Waiting upstream")
        else:
            self.calculate_button.setText(
                "Calculate"
                if state == EXECUTION_NOT_CALCULATED
                else "Recalculate",
            )

    def _node_execution_ui_state(self, node_id: str) -> tuple[str, str]:
        """Return live state plus an accepted active-run presentation overlay."""
        state = self.pipeline.node_execution_states.get(
            node_id,
            EXECUTION_NOT_CALCULATED,
        )
        message = self.pipeline.node_execution_messages.get(node_id, "")
        override = self._background_execution_state_overrides.get(node_id)
        if override is None:
            return state, message
        run_id, override_state, override_message = override
        if run_id != self._active_pipeline_run_id:
            return state, message
        return override_state, override_message

    def _background_node_result_override(
        self,
        node_id: str,
    ) -> PipelineNodeResult | None:
        result = self._background_node_result_overrides.get(node_id)
        if result is None or result.run_id != self._active_pipeline_run_id:
            return None
        node = self.pipeline.nodes.get(node_id)
        if node is None or node.operation_id != result.operation_id:
            return None
        return result

    def _discard_background_node_result_overrides(
        self,
        node_ids: Iterable[str] | None = None,
    ) -> set[str]:
        if node_ids is None:
            removed = set(self._background_execution_state_overrides) | set(
                self._background_node_result_overrides
            )
        else:
            removed = set(node_ids) & (
                set(self._background_execution_state_overrides)
                | set(self._background_node_result_overrides)
            )
        for node_id in removed:
            self._background_execution_state_overrides.pop(node_id, None)
            self._background_node_result_overrides.pop(node_id, None)
        return removed

    def _refresh_node_presentation_surfaces(
        self,
        node_ids: Iterable[str],
        *,
        thumbnails: bool = True,
    ) -> None:
        affected = set(node_ids) & set(self.pipeline.nodes)
        if not affected:
            return
        self._discard_pending_thumbnail_contrast_limit_requests()
        if thumbnails:
            for node_id in affected:
                data, state, output_port = self._node_display_payload(node_id)
                self._update_node_thumbnail(
                    node_id,
                    data,
                    state,
                    output_port,
                    queue_stack_contrast=False,
                )
                self.graph_view.set_node_can_pin(
                    node_id,
                    self._node_can_pin(node_id),
                )

        inspection_layer = self._layer_by_name(self._inspect_layer_name)
        inspected_node_id = (
            getattr(inspection_layer, "metadata", {}).get("node_id")
            if inspection_layer is not None
            else None
        )
        selected_affected = self._selected_node_id in affected
        if selected_affected:
            self._clear_output_histogram_cache()
            self._inspect_selected_node()
            self._sync_view_dims_bar()
            self._update_metadata_panel()
            self._update_histogram()
        elif inspected_node_id in affected:
            self._refresh_inspection_layer_if_active()
        if self._active_pinned_node_id in affected:
            self._refresh_pinned_layer_if_active()
            if not selected_affected:
                self._sync_view_dims_bar()

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
        if state == EXECUTION_BLOCKED:
            return message or (
                "Downstream result is stale; waiting for an upstream manual "
                "result."
            )
        if state == EXECUTION_ERROR:
            return message or "Calculation failed."
        return message or "This node calculates only when requested."

    def _render_parameters(self, node_id: str) -> None:
        self._clear_parameter_form()
        node = self.pipeline.nodes[node_id]
        compact_deconvolution_form = (
            node.operation_id in COMPACT_DECONVOLUTION_INSPECTOR_OPERATIONS
        )
        self.parameter_form.setRowWrapPolicy(
            QFormLayout.WrapLongRows
            if compact_deconvolution_form
            else QFormLayout.DontWrapRows
        )
        if node.operation_id == "input":
            self.parameter_group.setHidden(False)
            self._render_image_source_parameters(node_id)
            return
        specs = self.pipeline.node_parameter_specs(node_id)
        stack_note = self._stack_processing_note(node_id)
        help_note = self._operation_help_note(node_id)
        help_status = self._operation_help_note_status(node_id)
        self.parameter_group.setHidden(not specs and not stack_note and not help_note)
        if not specs:
            if stack_note:
                self._add_operation_note(stack_note)
                self.parameter_group.setHidden(False)
            if help_note:
                self._add_operation_note(help_note, status=help_status)
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
        if node.operation_id == "composite_to_rgb":
            self._render_composite_to_rgb_parameters(node_id)
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
            if compact_deconvolution_form and isinstance(
                widget,
                (ChoiceControl, ParameterControl),
            ):
                widget.setSizePolicy(
                    QSizePolicy.Ignored,
                    widget.sizePolicy().verticalPolicy(),
                )
                if isinstance(widget, ChoiceControl):
                    widget.combo.setSizePolicy(
                        QSizePolicy.Ignored,
                        widget.combo.sizePolicy().verticalPolicy(),
                    )
                else:
                    widget.slider.setMinimumWidth(72)
            widget.valueChanged.connect(
                lambda value, name=spec.name: self._on_param_changed(name, value)
            )
            self.parameter_form.addRow(spec.label, widget)
            self._apply_parameter_tooltip(spec, widget)
            self._parameter_widgets[spec.name] = widget
            rendered = True
        if self._add_parameter_visibility_note(node_id, specs):
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
            self._add_operation_note(help_note, status=help_status)
            rendered = True
        self.parameter_group.setHidden(not rendered)

    def _apply_parameter_tooltip(self, spec, widget: QWidget) -> None:
        tooltip = str(getattr(spec, "tooltip", "")).strip()
        if not tooltip:
            return
        display_tooltip = f"<qt>{tooltip}</qt>"
        widget.setToolTip(display_tooltip)
        for child in widget.findChildren(QWidget):
            child.setToolTip(display_tooltip)
        label = self.parameter_form.labelForField(widget)
        if isinstance(label, QWidget):
            label.setToolTip(display_tooltip)

    def _add_parameter_visibility_note(self, node_id: str, specs) -> bool:
        """Add at most one explanation for contextual inspector rows."""
        results = [
            resolve_parameter_visibility(
                spec,
                context=self.pipeline.parameter_visibility_context(node_id),
            )
            for spec in specs
            if str(getattr(spec, "visibility", "always") or "always") != "always"
        ]
        if any(not result.visible for result in results):
            text = (
                "Some settings are hidden because explicit input metadata or "
                "the selected mode proves they have no effect. Their stored "
                "values are preserved."
            )
        elif any(
            result.visible and "unresolved" in result.reason.casefold()
            for result in results
        ):
            text = (
                "Input context is unresolved, so potentially relevant settings "
                "remain available until metadata is resolved."
            )
        else:
            return False
        note = _InspectorNoteLabel(text)
        note.setStyleSheet("color: #94a3b8;")
        self.parameter_form.addRow(note)
        return True

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
        self._add_parameter_visibility_note(
            node_id,
            self.pipeline.node_parameter_specs(node_id),
        )
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
            base_spec = base_specs[f"{role}_scale"]
            if self._parameter_spec_hidden(node_id, base_spec):
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
                        visibility=base_spec.visibility,
                        visibility_parameter=base_spec.visibility_parameter,
                        visibility_values=base_spec.visibility_values,
                        visibility_ports=base_spec.visibility_ports,
                    )
                )
            else:
                specs.append(
                    self._effective_parameter_spec(
                        node_id,
                        base_spec,
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
        support_names = {"xy_size", "z_size"}
        annotated_names = managed_names | support_names
        auto_channel_count = self._born_wolf_psf_auto_channel_count(node_id)
        if not auto:
            self._initialize_manual_born_wolf_psf_params(node_id, resolution)

        for name, spec in specs.items():
            if self._parameter_spec_hidden(node_id, spec):
                continue
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
            if name in annotated_names:
                result = resolution.parameters.get(name)
                unresolved = bool(
                    name in managed_names
                    and auto
                    and result is not None
                    and result.required
                    and not result.resolved
                )
                if name in support_names:
                    status_text = self._born_wolf_psf_support_status_text(
                        name,
                        resolution,
                        value=node.params.get(name, spec.default),
                    )
                else:
                    status_text = self._born_wolf_psf_status_text(result, auto=auto)
                status = QLabel(status_text)
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
            self._apply_parameter_tooltip(spec, field_widget)
            self._parameter_widgets[name] = widget
        self._add_parameter_visibility_note(node_id, tuple(specs.values()))
        guidance, status = self._born_wolf_psf_guidance(node_id, resolution)
        self._add_operation_note(guidance, status=status)
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
                if _axis_is_explicit(axis) and (
                    axis.type == "channel" or axis.name.lower() == "c"
                ):
                    if 0 <= index < len(shape):
                        return int(shape[index])
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

    def _born_wolf_psf_nyquist(
        self,
        resolution,
    ) -> WidefieldNyquistResult | None:
        if resolution is None or resolution.unresolved:
            return None
        values = resolution.values
        try:
            return widefield_nyquist_sampling(
                wavelength_nm=float(values["wavelength_nm"]),
                numerical_aperture=float(values["numerical_aperture"]),
                refractive_index=float(values["refractive_index"]),
                xy_step_um=float(values["pixel_size_xy_um"]),
                z_step_um=(
                    float(values["z_step_um"])
                    if resolution.spatial_ndim == 3
                    else None
                ),
                spatial_ndim=resolution.spatial_ndim,
            )
        except (KeyError, TypeError, ValueError):
            return None

    def _born_wolf_psf_support_status_text(
        self,
        name: str,
        resolution,
        *,
        value,
    ) -> str:
        if name == "z_size" and int(resolution.spatial_ndim) < 3:
            return "not used for 2D"
        try:
            support = max(int(value), 1)
            if name == "xy_size":
                step_um = float(resolution.values["pixel_size_xy_um"])
            else:
                step_um = float(resolution.values["z_step_um"])
            half_span_um = (support // 2) * step_um
            full_span_um = 2.0 * half_span_um
            if not np.isfinite(full_span_um) or full_span_um <= 0:
                return "user set"
        except (KeyError, TypeError, ValueError):
            return "user set"
        return f"user set; {full_span_um:.3g} um span"

    def _born_wolf_psf_tail_report(
        self,
        node_id: str,
        *,
        spatial_ndim: int,
    ) -> PsfPreflightResult | None:
        if (
            node_id in self._pending_dirty_node_ids
            or self.pipeline.node_execution_states.get(node_id) != EXECUTION_READY
        ):
            return None
        psf = self.pipeline.outputs.get(node_id)
        if psf is None:
            return None
        return psf_preflight(
            self.pipeline.input_data_for_node(node_id),
            psf,
            spatial_ndim=spatial_ndim,
            image_state=self.pipeline.input_state_for_node(node_id),
            psf_state=self.pipeline.output_states.get(node_id),
        )

    def _born_wolf_psf_guidance(self, node_id: str, resolution) -> tuple[str, str]:
        node = self.pipeline.nodes[node_id]
        data = self.pipeline.input_data_for_node(node_id)
        shape = tuple(int(size) for size in getattr(data, "shape", ()) or ())
        spatial_ndim = int(resolution.spatial_ndim)
        xy_support = int(node.params.get("xy_size", 65))
        z_support = int(node.params.get("z_size", 33))
        psf_shape = (
            (z_support, xy_support, xy_support)
            if spatial_ndim == 3
            else (xy_support, xy_support)
        )
        image_shape = shape[-spatial_ndim:] if len(shape) >= spatial_ndim else ()
        support_text = " x ".join(str(size) for size in psf_shape)
        physical_spans = []
        try:
            xy_span_um = (xy_support - 1) * float(
                resolution.values["pixel_size_xy_um"]
            )
            if spatial_ndim == 3:
                z_span_um = (z_support - 1) * float(
                    resolution.values["z_step_um"]
                )
                physical_spans.append(f"Z {z_span_um:.3g} um")
            physical_spans.append(f"YX {xy_span_um:.3g} um")
        except (KeyError, TypeError, ValueError):
            physical_spans = []
        physical_text = (
            " Physical span between outer sample centers: "
            + html.escape("; ".join(physical_spans))
            + "."
            if physical_spans
            else ""
        )
        sections = [
            '<p><span style="color:#60a5fa"><b>SUPPORT</b></span><br>'
            f"Requested PSF: {support_text} samples. Support is user-set."
            + physical_text
            + " Use the tail check after calculation to assess this window.</p>"
        ]

        nyquist = self._born_wolf_psf_nyquist(resolution)
        status = "Unknown" if resolution.unresolved else "Ready"
        if nyquist is None:
            sections.insert(
                0,
                '<p><span style="color:#94a3b8"><b>SAMPLING CHECK PENDING</b>'
                "</span><br>Resolve wavelength, NA, refractive index, and "
                "physical sample spacing to estimate Nyquist sampling.</p>",
            )
        else:
            sampling_text = self._widefield_nyquist_summary(nyquist)
            if nyquist.met:
                sections.insert(
                    0,
                    '<p><span style="color:#34d399"><b>&#10003; WIDEFIELD '
                    "NYQUIST ESTIMATE MET</b></span><br>"
                    + html.escape(sampling_text)
                    + ".</p>",
                )
            else:
                status = "Warning"
                sections.insert(
                    0,
                    '<p><span style="color:#f59e0b"><b>! WIDEFIELD NYQUIST '
                    "ESTIMATE NOT MET</b></span><br>"
                    + html.escape(sampling_text)
                    + ". Changing PSF support cannot recover frequencies missing "
                    "from the acquisition.</p>",
                )

        tail_report = self._born_wolf_psf_tail_report(
            node_id,
            spatial_ndim=spatial_ndim,
        )
        edge_mass = None if tail_report is None else tail_report.edge_mass_fraction
        if edge_mass is None:
            sections.append(
                '<p><span style="color:#94a3b8"><b>TAIL CHECK PENDING</b>'
                "</span><br>Calculate this node to evaluate tail containment.</p>"
            )
        elif edge_mass > PSF_EDGE_MASS_WARNING_FRACTION:
            status = "Warning"
            by_axis = tail_report.edge_mass_fraction_by_axis or ()
            labels = ("Z", "Y", "X") if spatial_ndim == 3 else ("Y", "X")
            axis_text = ", ".join(
                f"{label} {fraction:.1%}"
                for label, fraction in zip(labels, by_axis, strict=False)
            )
            sections.append(
                '<p><span style="color:#f59e0b"><b>! TAIL REACHES THE '
                "WINDOW EDGE</b></span><br>"
                f"The outermost samples contain {edge_mass:.1%} of normalized "
                "PSF intensity"
                + (f" ({html.escape(axis_text)})" if axis_text else "")
                + f", above the {PSF_EDGE_MASS_WARNING_FRACTION:.0%} practical "
                "limit. Increase the identified support dimension and "
                "recalculate.</p>"
            )
        else:
            sections.append(
                '<p><span style="color:#34d399"><b>&#10003; TAIL CONTAINMENT '
                "CHECK PASSED</b></span><br>"
                f"The outermost samples contain {edge_mass:.1%} of normalized "
                f"PSF intensity, at or below the "
                f"{PSF_EDGE_MASS_WARNING_FRACTION:.0%} practical limit. The "
                "current support passes this check.</p>"
            )

        oversized = []
        if image_shape and len(image_shape) == len(psf_shape):
            labels = ("Z", "Y", "X") if spatial_ndim == 3 else ("Y", "X")
            oversized = [
                (label, psf_size, image_size)
                for label, psf_size, image_size in zip(
                    labels,
                    psf_shape,
                    image_shape,
                    strict=True,
                )
                if psf_size >= image_size
            ]
        if oversized:
            status = "Warning"
            comparisons = "; ".join(
                f"{label}: PSF {psf_size}, image {image_size}"
                for label, psf_size, image_size in oversized
            )
            sections.append(
                '<p><span style="color:#f59e0b"><b>! IMAGE EXTENT '
                "WARNING</b></span><br>"
                + html.escape(comparisons)
                + ". Review image extent, boundary assumptions, and processing "
                "dimensionality; do not change support only to clear the "
                "warning.</p>"
            )
        sections.append(
            '<p><span style="color:#94a3b8"><b>MORE GUIDANCE</b></span><br>'
            "Confirm that the conventional-widefield model matches the "
            "acquisition. See the <a href=\"https://rensutheart.github.io/"
            "vipp-mkdocs/workflows/psf-deconvolution/"
            "#choose-born-wolf-support\">Born-Wolf support guide</a> for "
            "support selection, boundary interpretation, and validation.</p>"
        )
        return "".join(sections), status

    @staticmethod
    def _widefield_nyquist_summary(result: WidefieldNyquistResult) -> str:
        comparisons = [
            f"XY {result.xy_step_um:.4g} um "
            f"{'<=' if result.xy_met else '>'} {result.xy_limit_um:.4g} um"
        ]
        if (
            result.spatial_ndim == 3
            and result.z_step_um is not None
            and result.z_limit_um is not None
        ):
            comparisons.append(
                f"Z {result.z_step_um:.4g} um "
                f"{'<=' if result.z_met else '>'} {result.z_limit_um:.4g} um"
            )
        return "; ".join(comparisons)

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

    def _add_operation_note(self, text: str, *, status: str = "") -> None:
        note = _InspectorNoteLabel(text)
        note.setTextInteractionFlags(
            Qt.TextSelectableByMouse | Qt.LinksAccessibleByMouse
        )
        note.setOpenExternalLinks(True)
        if status:
            note.setTextFormat(Qt.RichText)
        self._style_operation_note(note, status)
        self.parameter_form.addRow(note)
        self._parameter_widgets["operation_notice"] = note

    @staticmethod
    def _style_operation_note(note: QLabel, status: str) -> None:
        color = "#cbd5e1" if status else "#f59e0b"
        note.setStyleSheet(f"color: {color};")
        if status:
            note.setProperty("preflightStatus", status.lower())

    def _update_deconvolution_help_note(self) -> None:
        node_id = self._selected_node_id
        node = self.pipeline.nodes.get(node_id)
        note = self._parameter_widgets.get("operation_notice")
        if (
            node is None
            or node.operation_id not in COMPACT_DECONVOLUTION_INSPECTOR_OPERATIONS
            or not isinstance(note, QLabel)
        ):
            return
        report = self._deconvolution_psf_preflight(node_id)
        note.setText(self._deconvolution_help_note(node_id))
        self._style_operation_note(
            note,
            self._deconvolution_psf_display_status(node_id, report),
        )

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
        if node.operation_id in COMPACT_DECONVOLUTION_INSPECTOR_OPERATIONS:
            return self._deconvolution_help_note(node_id)
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

    def _operation_help_note_status(self, node_id: str) -> str:
        node = self.pipeline.nodes.get(node_id)
        if (
            node is None
            or node.operation_id not in COMPACT_DECONVOLUTION_INSPECTOR_OPERATIONS
        ):
            return ""
        report = self._deconvolution_psf_preflight(node_id)
        return self._deconvolution_psf_display_status(node_id, report)

    def _deconvolution_help_note(self, node_id: str) -> str:
        node = self.pipeline.nodes[node_id]
        report = self._deconvolution_psf_preflight(node_id)
        nyquist = self._deconvolution_psf_nyquist(node_id)
        display_status = self._deconvolution_psf_display_status(node_id, report)
        sections: list[str] = []
        status_color = {
            "Ready": "#34d399",
            "Warning": "#f59e0b",
            "Invalid": "#f87171",
            "Unknown": "#94a3b8",
        }[display_status]
        status_summary = {
            "Ready": (
                "All available checks passed; confirm the PSF model matches "
                "the acquisition."
            ),
            "Warning": (
                "Calculation can run, but scientific reliability needs "
                "attention."
            ),
            "Invalid": "Fix the item(s) below before calculating.",
            "Unknown": "Some checks need resolved input data or metadata.",
        }[display_status]
        sections.append(
            f'<p><span style="color:{status_color}"><b>PSF preflight: '
            f"{display_status}</b></span> - {status_summary}</p>"
        )

        passed = self._deconvolution_psf_passed_checks(report, nyquist)
        if passed:
            items = "<br>".join(
                f"&#10003; {html.escape(item)}" for item in passed
            )
            sections.append(
                '<p><span style="color:#34d399"><b>CHECKS PASSED</b></span>'
                f"<br>{items}</p>"
            )

        attention = [
            (
                issue.severity,
                self._deconvolution_psf_issue_title(issue.code),
                self._deconvolution_psf_issue_detail(
                    issue.code,
                    issue.detail,
                    report,
                ),
            )
            for issue in report.issues
            if issue.severity in {"warning", "invalid"}
        ]
        if nyquist is not None and not nyquist.met:
            attention.append(
                (
                    "warning",
                    "Widefield Nyquist estimate not met",
                    self._widefield_nyquist_summary(nyquist)
                    + ". Deconvolution cannot recover frequencies that were "
                    "aliased during acquisition.",
                )
            )
        unknown = tuple(
            issue for issue in report.issues if issue.severity == "unknown"
        )
        if attention:
            items = "<br><br>".join(
                '<span style="color:'
                + ("#f87171" if severity == "invalid" else "#f59e0b")
                + f'"><b>! {html.escape(title)}</b></span><br>'
                + html.escape(detail)
                for severity, title, detail in attention
            )
            sections.append(
                '<p><span style="color:#f59e0b"><b>NEEDS ATTENTION</b></span>'
                f"<br>{items}</p>"
            )
        if unknown:
            items = "<br><br>".join(
                "<b>? "
                + html.escape(self._deconvolution_psf_issue_title(issue.code))
                + "</b><br>"
                + html.escape(issue.detail)
                for issue in unknown
            )
            sections.append(
                '<p><span style="color:#94a3b8"><b>COULD NOT CHECK</b></span>'
                f"<br>{items}</p>"
            )

        actions = self._deconvolution_psf_next_actions(node_id, report)
        if actions:
            items = "<br>".join(
                f"{index}. {html.escape(action)}"
                for index, action in enumerate(actions, start=1)
            )
            sections.append(
                '<p><span style="color:#60a5fa"><b>WHAT TO DO NEXT</b></span>'
                f"<br>{items}</p>"
            )
        if node.operation_id == "richardson_lucy_tv_deconvolution":
            sections.append(
                '<p><span style="color:#94a3b8"><b>SCIENTIFIC CAUTION</b>'
                "</span><br>Excessive TV regularization may remove fine or dim "
                "structures. Early low-iteration outputs may be under-converged. "
                "Validate PSF sampling and centering before tuning "
                "regularization.</p>"
            )
        return "".join(sections)

    @staticmethod
    def _deconvolution_psf_passed_checks(
        report: PsfPreflightResult,
        nyquist: WidefieldNyquistResult | None = None,
    ) -> tuple[str, ...]:
        codes = {issue.code for issue in report.issues}
        checks: list[str] = []
        if (
            report.spatial_ndim is not None
            and len(report.psf_shape) == report.spatial_ndim
            and len(report.image_shape) >= report.spatial_ndim
        ):
            checks.append(f"{report.spatial_ndim}D PSF rank matches")
        if report.physical_sampling_known and "sampling_mismatch" not in codes:
            checks.append("physical sampling matches the image")
        if nyquist is not None and nyquist.met:
            checks.append(
                "conventional-widefield Nyquist estimate is met ("
                + VippWidget._widefield_nyquist_summary(nyquist)
                + ")"
            )
        invalid_value_codes = {
            "empty",
            "negative",
            "non_numeric",
            "nonfinite",
            "nonpositive_sum",
        }
        if report.values_scanned and not (codes & invalid_value_codes):
            checks.append("values are finite and non-negative")
        if report.approximately_normalized is True and report.psf_sum is not None:
            checks.append(f"normalized (sum = {report.psf_sum:.6g})")
        if report.odd_shape is True:
            checks.append("dimensions are odd")
        if (
            report.peak_offset_voxels is not None
            and report.centroid_offset_voxels is not None
            and "off_center_peak" not in codes
            and "centroid_offset" not in codes
        ):
            checks.append(
                "centered (peak offset "
                f"{report.peak_offset_voxels:.3g}; centroid offset "
                f"{report.centroid_offset_voxels:.3g} voxel)"
            )
        if (
            report.support_fraction_of_image
            and "support_vs_image" not in codes
        ):
            checks.append("support is smaller than the image on every axis")
        if report.edge_mass_fraction is not None and "edge_mass" not in codes:
            checks.append("PSF tail is contained within its support")
        return tuple(checks)

    @staticmethod
    def _deconvolution_psf_issue_title(code: str) -> str:
        return {
            "support_vs_image": "PSF support reaches or exceeds the image extent",
            "edge_mass": "PSF tail reaches its support boundary",
            "missing_calibration": "Physical calibration is unavailable",
            "sampling_mismatch": "Image and PSF sampling differ",
            "even_shape": "PSF dimensions are even",
            "off_center_peak": "PSF peak is off center",
            "centroid_offset": "PSF centroid is off center",
            "not_normalized": "PSF is not normalized",
            "negative": "PSF contains negative values",
            "nonfinite": "PSF contains non-finite values",
            "psf_rank": "PSF dimensionality does not match",
            "image_rank": "Image dimensionality does not match",
            "unresolved_inputs": "Inputs are not resolved",
            "unresolved_spatial_rank": "Spatial processing is not resolved",
            "values_not_scanned": "PSF values were not scanned",
        }.get(code, "PSF check")

    @staticmethod
    def _deconvolution_psf_issue_detail(
        code: str,
        detail: str,
        report: PsfPreflightResult,
    ) -> str:
        retained_by_axis = report.centered_mass_within_image_support_by_axis
        if (
            code != "support_vs_image"
            or retained_by_axis is None
            or report.spatial_ndim is None
            or len(retained_by_axis) != report.spatial_ndim
        ):
            return detail
        image_shape = report.image_shape[-report.spatial_ndim :]
        if len(image_shape) != len(report.psf_shape):
            return detail
        consequences = []
        for label, psf_size, image_size, retained in zip(
            report.spatial_axis_labels,
            report.psf_shape,
            image_shape,
            retained_by_axis,
            strict=True,
        ):
            if (
                psf_size <= image_size
                or retained >= 0.9995
                or not detail.startswith(f"{label} support")
            ):
                continue
            consequences.append(
                f"Cropping {label} from {psf_size} to {image_size} centered "
                f"samples would retain {retained:.1%} and discard "
                f"{1.0 - retained:.1%} of the current PSF intensity."
            )
        if not consequences:
            return detail
        return detail + " " + " ".join(consequences)

    def _deconvolution_psf_display_status(
        self,
        node_id: str,
        report: PsfPreflightResult,
    ) -> str:
        nyquist = self._deconvolution_psf_nyquist(node_id)
        if report.status == "Invalid":
            return "Invalid"
        if report.status == "Warning" or (nyquist is not None and not nyquist.met):
            return "Warning"
        return report.status

    def _deconvolution_psf_next_actions(
        self,
        node_id: str,
        report: PsfPreflightResult,
    ) -> tuple[str, ...]:
        codes = {issue.code for issue in report.issues}
        source_operations = self._deconvolution_psf_source_operations(node_id)
        actions: list[str] = []
        if "support_vs_image" in codes:
            if "born_wolf_psf" in source_operations and report.spatial_ndim == 3:
                actions.append(
                    "For true 3D restoration, acquire guard planes above and "
                    "below the region of interest so the PSF fits well inside "
                    "the stack, then interpret or crop the restored margins. "
                    "With this existing stack, set Spatial processing to 2D YX "
                    "on both Born-Wolf PSF and RL/RL-TV only if plane-wise "
                    "restoration is scientifically acceptable."
                )
            else:
                actions.append(
                    "Use a larger image/stack or a scientifically justified "
                    "smaller PSF support. Do not crop the PSF only to clear the "
                    "warning, because that can truncate the optical model."
                )
            if "prepare_validate_psf" in source_operations:
                actions.append(
                    "Prepare / Validate PSF is working as intended: it fixes "
                    "sign, normalization, odd shape, and centering, but it does "
                    "not shrink a non-empty PSF support. No change from that "
                    "node is expected for this size warning."
                )
        if "edge_mass" in codes:
            actions.append(
                "Resolve the image-versus-PSF size choice first. Then regenerate "
                "or remeasure the PSF and recheck this tail warning; enlarge "
                "support only on axes where it still fits comfortably inside "
                "the image."
            )
        if codes & {
            "even_shape",
            "negative",
            "nonfinite",
            "not_normalized",
            "off_center_peak",
            "centroid_offset",
        }:
            actions.append(
                "Use Prepare / Validate PSF, then calculate it and return here "
                "to confirm the centering and normalization checks pass."
            )
        if "missing_calibration" in codes:
            actions.append(
                "Set physical pixel size on the image and PSF (or use metadata "
                "that supplies it) before trusting the sampling comparison."
            )
        if "sampling_mismatch" in codes:
            actions.append(
                "Generate or measure a PSF at the image's physical sampling; "
                "VIPP will not resample it implicitly."
            )
        return tuple(actions)

    def _deconvolution_psf_source_operations(self, node_id: str) -> set[str]:
        return {
            source.operation_id
            for source in self._deconvolution_psf_source_nodes(node_id)
        }

    def _deconvolution_psf_source_nodes(self, node_id: str) -> tuple[GraphNode, ...]:
        sources: list[GraphNode] = []
        target_id = node_id
        target_port = 1
        visited: set[str] = set()
        while target_id not in visited:
            visited.add(target_id)
            connection = next(
                (
                    item
                    for item in self.pipeline.connections
                    if item.target_id == target_id
                    and item.target_port == target_port
                ),
                None,
            )
            if connection is None:
                break
            source = self.pipeline.nodes.get(connection.source_id)
            if source is None:
                break
            sources.append(source)
            if source.operation_id != "prepare_validate_psf":
                break
            target_id = source.id
            target_port = 0
        return tuple(sources)

    def _deconvolution_psf_nyquist(
        self,
        node_id: str,
    ) -> WidefieldNyquistResult | None:
        born_wolf = next(
            (
                source
                for source in self._deconvolution_psf_source_nodes(node_id)
                if source.operation_id == "born_wolf_psf"
            ),
            None,
        )
        if born_wolf is None:
            return None
        try:
            resolution = self._born_wolf_psf_resolution(born_wolf.id)
        except (KeyError, TypeError, ValueError):
            return None
        return self._born_wolf_psf_nyquist(resolution)

    def _deconvolution_psf_preflight(self, node_id: str) -> PsfPreflightResult:
        data_by_port = self.pipeline.input_data_by_port_for_node(node_id)
        states_by_port = self.pipeline.input_states_by_port_for_node(node_id)
        image = data_by_port.get(0)
        psf = data_by_port.get(1)
        image_state = states_by_port.get(0)
        psf_state = states_by_port.get(1)
        spatial_ndim = self._deconvolution_inspector_spatial_ndim(
            node_id,
            image,
            image_state,
        )
        key = (
            id(image),
            id(psf),
            tuple(getattr(image, "shape", ()) or ()),
            tuple(getattr(psf, "shape", ()) or ()),
            spatial_ndim,
            self._psf_preflight_state_key(image_state),
            self._psf_preflight_state_key(psf_state),
        )
        cached = self._psf_preflight_cache.get(node_id)
        if cached is not None and cached[0] == key:
            return cached[1]
        result = psf_preflight(
            image,
            psf,
            spatial_ndim=spatial_ndim,
            image_state=image_state,
            psf_state=psf_state,
        )
        self._psf_preflight_cache[node_id] = (key, result)
        return result

    def _deconvolution_inspector_spatial_ndim(
        self,
        node_id: str,
        image,
        image_state: ImageState | None,
    ) -> int | None:
        mode = str(
            self.pipeline.nodes[node_id].params.get(
                "spatial_mode",
                "Auto from axes",
            )
        ).strip().lower()
        if mode.startswith("2d"):
            return 2
        if mode.startswith("3d"):
            return 3
        shape = tuple(getattr(image, "shape", ()) or ())
        if shape and len(shape) <= 2:
            return 2
        if image_state is not None and image_state.spatial_axes_explicit:
            count = sum(axis.type == "space" for axis in image_state.axes)
            if count >= 3:
                return 3
            if count >= 2:
                return 2
        return None

    @staticmethod
    def _psf_preflight_state_key(state: ImageState | None) -> tuple[object, ...]:
        if state is None:
            return ()
        return (
            tuple(state.shape),
            tuple(
                (
                    axis.name,
                    axis.type,
                    axis.unit,
                    axis.scale,
                    axis.confidence,
                )
                for axis in state.axes
            ),
        )

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

    def _render_composite_to_rgb_parameters(self, node_id: str) -> None:
        """Render metadata-aware axis and per-source RGB assignment controls."""
        node = self.pipeline.nodes[node_id]
        base_specs = {
            spec.name: spec for spec in self.pipeline.node_parameter_specs(node_id)
        }
        axis_mode = self._composite_channel_axis_mode(node)
        mapping_mode = self._composite_mapping_mode(node)
        axis = self._composite_resolved_channel_axis(node_id, axis_mode)
        count = self._composite_channel_count(node_id, axis)
        assignments = self._composite_channel_assignments(
            node_id,
            mapping_mode,
            axis,
            count,
        )
        mapping_warning = (
            self._composite_manual_mapping_warning(node_id, count)
            if mapping_mode == "Manual"
            else ""
        )

        axis_mode_spec = ParameterSpec(
            "channel_axis_mode",
            "Channel axis mode",
            "choice",
            "Auto",
            0,
            0,
            1,
            choices=("Auto", "Manual"),
            tooltip=(
                "Auto selects the one explicit metadata channel axis, normally "
                "C. If axes are unnamed or no unique C/channel axis exists, "
                "switch to Manual. Manual may deliberately use any dimension, "
                "for example Z as colour when no C axis exists; VIPP never "
                "guesses from a length of three or four alone."
            ),
        )
        axis_mode_widget = ChoiceControl(
            axis_mode_spec,
            axis_mode,
            ParameterBounds(0, 1, 1, 0),
        )
        axis_mode_widget.valueChanged.connect(
            self._on_composite_channel_axis_mode_changed
        )
        self.parameter_form.addRow(axis_mode_spec.label, axis_mode_widget)
        self._apply_parameter_tooltip(axis_mode_spec, axis_mode_widget)
        self._parameter_widgets[axis_mode_spec.name] = axis_mode_widget

        axis_options = self._axis_slice_options_for(node_id)
        axis_choices = tuple(str(option.index) for option in axis_options)
        axis_labels = tuple(
            self._composite_axis_choice_label(option) for option in axis_options
        )
        if axis is None:
            if axis_mode == "Manual":
                axis_choices = ("-1", *axis_choices)
                axis_labels = (
                    "Choose an axis (saved selection is invalid)",
                    *axis_labels,
                )
            else:
                axis_choices = ("-1",)
                axis_labels = ("No channel axis resolved",)
        axis_spec = ParameterSpec(
            "channel_axis",
            "Channel axis",
            "choice",
            str(axis if axis is not None else -1),
            0,
            max(len(axis_choices) - 1, 0),
            1,
            choices=axis_choices,
            choice_labels=axis_labels,
            tooltip=(
                "The array dimension containing the source channels. In Auto "
                "mode this shows the metadata-resolved axis and is read-only; "
                "switch to Manual to choose any array dimension deliberately."
            ),
        )
        axis_widget = ChoiceControl(
            axis_spec,
            str(axis if axis is not None else -1),
            ParameterBounds(0, max(len(axis_choices) - 1, 0), 1, 0),
        )
        axis_widget.setEnabled(axis_mode == "Manual" and bool(axis_options))
        axis_widget.valueChanged.connect(self._on_composite_channel_axis_changed)
        self.parameter_form.addRow(axis_spec.label, axis_widget)
        self._apply_parameter_tooltip(axis_spec, axis_widget)
        self._parameter_widgets[axis_spec.name] = axis_widget

        axis_status = QLabel(self._composite_axis_status(node_id, axis_mode, axis))
        axis_status.setWordWrap(True)
        axis_status.setStyleSheet(
            "color: #94a3b8;" if axis is not None else "color: #f59e0b;"
        )
        axis_status.setToolTip(axis_mode_spec.tooltip)
        self.parameter_form.addRow(axis_status)
        self._parameter_widgets["channel_axis_status"] = axis_status

        mapping_mode_spec = ParameterSpec(
            "mapping_mode",
            "RGB mapping mode",
            "choice",
            "Auto",
            0,
            0,
            1,
            choices=("Auto", "Manual"),
            tooltip=(
                "Auto carries exact channel colours from metadata, or uses "
                "VIPP's fluorescence palette when colours are absent. Manual "
                "lets you assign every source channel explicitly."
            ),
        )
        mapping_mode_widget = ChoiceControl(
            mapping_mode_spec,
            mapping_mode,
            ParameterBounds(0, 1, 1, 0),
        )
        mapping_mode_widget.valueChanged.connect(
            self._on_composite_mapping_mode_changed
        )
        self.parameter_form.addRow(mapping_mode_spec.label, mapping_mode_widget)
        self._apply_parameter_tooltip(mapping_mode_spec, mapping_mode_widget)
        self._parameter_widgets[mapping_mode_spec.name] = mapping_mode_widget

        channel_labels = self._composite_channel_labels(node_id, axis, count)
        assignment_tooltip = (
            "Assign this source channel to an additive RGB colour. Channels "
            "that share an RGB component are added. Unassigned excludes the "
            "source channel from the RGB output."
        )
        for index, assignment in enumerate(assignments):
            choices = list(COMPOSITE_CHANNEL_ASSIGNMENT_CHOICES)
            if assignment not in choices:
                # Preserve and truthfully display a nonstandard metadata colour
                # as exact #RRGGBB while Auto is active (and if Manual inherits
                # that resolved value).
                choices.append(assignment)
            spec = ParameterSpec(
                f"channel_color_{index}",
                f"{channel_labels[index]} assignment",
                "choice",
                assignment,
                0,
                0,
                1,
                choices=tuple(choices),
                tooltip=assignment_tooltip,
            )
            widget = ChoiceControl(
                spec,
                assignment,
                ParameterBounds(0, len(choices) - 1, 1, 0),
            )
            widget.setEnabled(mapping_mode == "Manual")
            widget.valueChanged.connect(
                lambda value, slot=index: self._on_composite_channel_color_changed(
                    slot,
                    value,
                )
            )
            self.parameter_form.addRow(spec.label, widget)
            self._apply_parameter_tooltip(spec, widget)
            self._parameter_widgets[spec.name] = widget

        mapping_status = QLabel(
            self._composite_mapping_status(
                mapping_mode,
                channel_labels,
                assignments,
                mapping_warning,
            )
        )
        mapping_status.setWordWrap(True)
        mapping_status.setStyleSheet(
            "color: #f59e0b;" if mapping_warning else "color: #94a3b8;"
        )
        mapping_status.setToolTip(mapping_mode_spec.tooltip)
        self.parameter_form.addRow(mapping_status)
        self._parameter_widgets["mapping_status"] = mapping_status

        intensity_spec = base_specs["intensity_mapping"]
        if not intensity_spec.tooltip:
            intensity_spec = replace(
                intensity_spec,
                tooltip=(
                    "Preserve numeric values keeps the native intensity scale "
                    "and does not normalize or clip additive mixtures. The "
                    "1st-99th percentile option independently rescales each "
                    "source channel and is intentionally lossy."
                ),
            )
        intensity_widget = ChoiceControl(
            intensity_spec,
            node.params.get(intensity_spec.name),
            self._parameter_bounds_for(node_id, intensity_spec),
        )
        intensity_widget.valueChanged.connect(
            lambda value: self._on_param_changed("intensity_mapping", value)
        )
        self.parameter_form.addRow(intensity_spec.label, intensity_widget)
        self._apply_parameter_tooltip(intensity_spec, intensity_widget)
        self._parameter_widgets[intensity_spec.name] = intensity_widget

        note = QLabel(
            "The legacy Red/Green/Blue index fields remain load-compatible but "
            "are represented here as one assignment per source channel."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #94a3b8;")
        self.parameter_form.addRow(note)
        self._parameter_widgets["composite_mapping_note"] = note
        self.parameter_group.setHidden(False)

    @staticmethod
    def _composite_axis_choice_label(option: AxisSliceOption) -> str:
        axis_type = str(option.axis_type).strip()
        detail = f", {axis_type}" if axis_type and axis_type != "unknown" else ""
        return f"{option.index}: {option.name} ({option.size}{detail})"

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
                if (
                    _axis_is_explicit(axis)
                    and (axis.type == "channel" or axis.name.lower() == "c")
                    and axis.name.lower() not in {"rgb", "rgba"}
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
            if self._selected_node_id in self._interactive_collection_source_paths:
                self._clear_interactive_collection_batch_session()
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
        if (
            node.id not in self._interactive_collection_source_paths
            and str(node.params.get("source_mode", "")) != "file path"
        ):
            return None
        path = self._file_source_path_for_node(node)
        if path is None:
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

    def _source_summary(self, inspection: SourceInspection | None, node) -> str:
        batch_prefix = self._interactive_collection_source_summary(node.id)
        if inspection is None:
            return batch_prefix
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
        summary += (
            " File data is loaded as a frozen snapshot and remains pinned "
            "until Refresh."
        )
        return f"{batch_prefix} {summary}".strip()

    def _interactive_collection_source_summary(self, node_id: str) -> str:
        items = self._interactive_collection_batch_items
        index = self._interactive_collection_batch_index
        if not items or not 0 <= index < len(items):
            return ""
        path = items[index].source_paths.get(node_id)
        if path is None:
            return ""
        return (
            f"Representative batch item {index + 1}/{len(items)}: "
            f"{Path(path).name}. The full batch has not been run."
        )

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
        if self._refresh_selected_parameter_visibility():
            # Visibility is presentation-only.  Do not pass freshly rendered
            # controls through value normalization or report a graph change.
            return False
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
        if node.operation_id == "composite_to_rgb":
            # Its controls depend on the resolved input axis, channel count,
            # names, and exact carried metadata colours.
            self._render_parameters(self._selected_node_id)
            return False
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
            # Refreshing bounds, labels, or visibility is presentation-only.
            # User signals are the sole authority for ordinary parameter edits;
            # never serialize a control's rounded/clamped representation here.
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
        if node.operation_id in COMPACT_DECONVOLUTION_INSPECTOR_OPERATIONS:
            self._update_deconvolution_help_note()
        return changed

    def _refresh_selected_parameter_visibility(self) -> bool:
        """Rebuild contextual rows without changing any serialized state."""
        node_id = self._selected_node_id
        node = self.pipeline.nodes.get(node_id)
        if node is None or not self._parameter_visibility_controls_changed(node_id):
            return False
        saved_params = deepcopy(node.params)
        try:
            self._render_parameters(node_id)
        finally:
            node.params.clear()
            node.params.update(saved_params)
        return True

    def _parameter_visibility_controls_changed(self, node_id: str) -> bool:
        """Return whether input-aware controls need the form to be rebuilt."""
        node = self.pipeline.nodes.get(node_id)
        if node is not None and node.operation_id == "rescale_axes":
            expected = {
                spec.name for spec in self._rescale_axes_visible_specs(node_id)
            }
            actual = {
                name
                for name in self._parameter_widgets
                if not name.endswith("_reset")
            }
            return actual != expected
        for spec in self.pipeline.node_parameter_specs(node_id):
            if str(getattr(spec, "visibility", "always") or "always") == "always":
                continue
            expected = not self._parameter_spec_hidden(node_id, spec)
            if (spec.name in self._parameter_widgets) != expected:
                return True
        return False

    def _parameter_spec_hidden(self, node_id: str, spec) -> bool:
        node = self.pipeline.nodes.get(node_id)
        if node is None:
            return False
        visibility = resolve_parameter_visibility(
            spec,
            context=self.pipeline.parameter_visibility_context(node_id),
        )
        return not visibility.visible

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
                    if resolved.startswith("unavailable"):
                        labels[index] = f"Auto from axes - {resolved}"
                    else:
                        labels[index] = f"Auto from axes - using {resolved}"
                    break
        state = self.pipeline.input_state_for_node(node_id)
        spatial_count = (
            sum(axis.type == "space" for axis in state.axes)
            if state is not None and getattr(state, "axes_explicit", False)
            else None
        )
        if spatial_count is not None and spatial_count < 3:
            for index, choice in enumerate(choices):
                if str(choice).startswith("3D"):
                    labels[index] = f"{choice} - unavailable for resolved input"
        return tuple(labels)

    def _resolved_auto_spatial_mode_label(self, node_id: str) -> str:
        state = self.pipeline.input_state_for_node(node_id)
        if state is None or not bool(
            getattr(state, "spatial_axes_explicit", False)
        ):
            return "unavailable (axes are inferred or missing)"
        spatial_count = sum(axis.type == "space" for axis in state.axes)
        if spatial_count >= 3:
            return "3D ZYX"
        if spatial_count >= 2:
            return "2D YX"
        return "unavailable (fewer than two explicit spatial axes)"

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
        state = self.pipeline.input_state_for_node(node_id)
        axes_resolved = bool(
            state is not None
            and getattr(state, "axes_explicit", False)
            and tuple(getattr(state, "axes", ()))
        )
        if not axes_resolved:
            data = self.pipeline.input_data_for_node(node_id)
            shape = tuple(getattr(data, "shape", ()))
            for fallback_index, role in enumerate(("x", "y", "z")):
                trailing_index = len(shape) - fallback_index - 1
                role_map.setdefault(
                    role,
                    AxisSliceOption(
                        max(trailing_index, 0),
                        role,
                        "unresolved",
                        int(shape[trailing_index]) if trailing_index >= 0 else 1,
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
        state = self.pipeline.input_state_for_node(node_id)
        if state is None or not getattr(state, "axes_explicit", False):
            return choices
        spatial_count = sum(axis.type == "space" for axis in state.axes)
        if spatial_count >= 3:
            return choices
        if node is not None and node.operation_id == "measure_3d_mesh_morphology":
            available = choices
        else:
            available = choices[:2]
        stored = str(node.params.get("spatial_mode", "")) if node is not None else ""
        if stored in choices and stored not in available:
            available = (*available, stored)
        return available

    def _available_clear_border_modes(self, node_id: str) -> tuple[str, ...]:
        choices = (
            "All spatial borders",
            "Lateral borders only (YX)",
        )
        state = self.pipeline.input_state_for_node(node_id)
        if state is None or not getattr(state, "axes_explicit", False):
            return choices
        spatial_count = sum(axis.type == "space" for axis in state.axes)
        return choices if spatial_count >= 3 else choices[:1]

    def _input_spatial_count(self, node_id: str) -> int:
        state = self.pipeline.input_state_for_node(node_id)
        if state is not None and bool(
            getattr(state, "spatial_axes_explicit", False)
        ):
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
        if (
            node is not None
            and node.operation_id == "richardson_lucy_tv_deconvolution"
            and spec.name
            in {
                "iterations",
                "tv_regularization",
                "tv_epsilon",
                "filter_epsilon",
                "denominator_floor",
            }
        ):
            return self._richardson_lucy_tv_parameter_bounds(spec)
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

    @staticmethod
    def _richardson_lucy_tv_parameter_bounds(spec) -> ParameterBounds:
        entry_maximum = 1_000_000.0
        if spec.name == "iterations":
            return ParameterBounds(
                1,
                100,
                1,
                0,
                expandable=False,
                entry_minimum=1,
                entry_maximum=2_147_483_647,
            )
        if spec.name == "tv_regularization":
            return ParameterBounds(
                1e-6,
                1e-1,
                1e-4,
                6,
                expandable=False,
                logarithmic=True,
                entry_minimum=0.0,
                entry_maximum=entry_maximum,
            )
        if spec.name == "tv_epsilon":
            return ParameterBounds(
                1e-12,
                1e-2,
                1e-7,
                12,
                expandable=False,
                logarithmic=True,
                entry_minimum=1e-12,
                entry_maximum=entry_maximum,
            )
        if spec.name == "filter_epsilon":
            return ParameterBounds(
                1e-15,
                1e-3,
                1e-12,
                15,
                expandable=False,
                logarithmic=True,
                entry_minimum=0.0,
                entry_maximum=entry_maximum,
            )
        if spec.name == "denominator_floor":
            return ParameterBounds(
                1e-3,
                1.0,
                1e-2,
                6,
                expandable=False,
                logarithmic=True,
                entry_minimum=1e-6,
                entry_maximum=entry_maximum,
            )
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

    @staticmethod
    def _composite_channel_axis_mode(node) -> str:
        stored = str(node.params.get("channel_axis_mode", "")).strip().title()
        if stored in {"Auto", "Manual"}:
            return stored
        try:
            channel_axis = int(node.params.get("channel_axis", -1))
        except (TypeError, ValueError):
            channel_axis = -1
        return "Manual" if channel_axis >= 0 else "Auto"

    @staticmethod
    def _composite_mapping_mode(node) -> str:
        stored = str(node.params.get("mapping_mode", "")).strip().title()
        if stored in {"Auto", "Manual"}:
            return stored
        selections = []
        for name in ("red_channel", "green_channel", "blue_channel"):
            try:
                selections.append(int(node.params.get(name, -1)))
            except (TypeError, ValueError):
                selections.append(-1)
        return "Manual" if any(value >= 0 for value in selections) else "Auto"

    def _composite_declared_channel_axes(self, node_id: str) -> tuple[int, ...]:
        state = self.pipeline.input_state_for_node(node_id)
        if state is None:
            return ()
        return tuple(
            index
            for index, axis in enumerate(state.axes)
            if _axis_is_explicit(axis)
            and (
                str(axis.type).strip().lower() == "channel"
                or str(axis.name).strip().lower() in {"c", "channel", "rgb", "rgba"}
            )
        )

    def _composite_resolved_channel_axis(
        self,
        node_id: str,
        mode: str,
    ) -> int | None:
        data = self.pipeline.input_data_for_node(node_id)
        if data is None:
            return None
        ndim = np.asarray(data).ndim
        if ndim <= 0:
            return None
        declared = self._composite_declared_channel_axes(node_id)
        if mode == "Auto":
            return declared[0] if len(declared) == 1 else None
        node = self.pipeline.nodes[node_id]
        try:
            stored = int(node.params.get("channel_axis", -1))
        except (TypeError, ValueError):
            stored = -1
        if 0 <= stored < ndim:
            return stored
        # A saved Manual selection must be shown as unresolved when it is
        # absent or invalid. Falling back here would make the inspector look
        # runnable while the operation itself correctly rejects the params.
        return None

    def _composite_channel_count(self, node_id: str, axis: int | None) -> int:
        data = self.pipeline.input_data_for_node(node_id)
        if data is None or axis is None:
            return 0
        shape = np.asarray(data).shape
        if not 0 <= int(axis) < len(shape):
            return 0
        return max(int(shape[int(axis)]), 0)

    @staticmethod
    def _composite_exact_metadata_color(color) -> str | None:
        if color is None:
            return None
        try:
            value = int(color) & 0xFFFFFF
        except (TypeError, ValueError):
            return None
        standard = {
            0xFF0000: "Red",
            0x00FF00: "Green",
            0x0000FF: "Blue",
            0xFF00FF: "Magenta",
            0x00FFFF: "Cyan",
            0xFFFF00: "Yellow",
        }
        return standard.get(value, f"#{value:06X}")

    def _composite_auto_assignments(
        self,
        node_id: str,
        axis: int | None,
        count: int,
    ) -> list[str]:
        if axis is None or count <= 0:
            return []
        state = self.pipeline.input_state_for_node(node_id)
        declared = self._composite_declared_channel_axes(node_id)
        uses_declared_channels = len(declared) == 1 and axis == declared[0]
        axis_name = ""
        if state is not None and 0 <= axis < len(state.axes):
            axis_name = str(state.axes[axis].name).strip().lower()
        if uses_declared_channels and axis_name in {"rgb", "rgba"}:
            encoded = ["Red", "Green", "Blue"]
            if count > 3:
                encoded.extend(["Unassigned"] * (count - 3))
            return encoded[:count]

        metadata_colors = ()
        if uses_declared_channels and state is not None:
            metadata_colors = tuple(channel.color for channel in state.channels)
        defaults = channel_color_labels_from_metadata((), count)
        assignments: list[str] = []
        for index in range(count):
            exact = (
                self._composite_exact_metadata_color(metadata_colors[index])
                if index < len(metadata_colors)
                else None
            )
            assignments.append(exact or defaults[index])
        return assignments

    @staticmethod
    def _merge_composite_assignments(current: str, added: str) -> str:
        components = {
            "Unassigned": frozenset(),
            "Red": frozenset({"r"}),
            "Green": frozenset({"g"}),
            "Blue": frozenset({"b"}),
            "Yellow": frozenset({"r", "g"}),
            "Magenta": frozenset({"r", "b"}),
            "Cyan": frozenset({"g", "b"}),
        }
        names = {value: key for key, value in components.items()}
        merged = components.get(current, frozenset()) | components.get(
            added,
            frozenset(),
        )
        return names.get(merged, "#FFFFFF")

    def _composite_legacy_assignments(self, node, count: int) -> list[str]:
        assignments = ["Unassigned"] * count
        for name, color in (
            ("red_channel", "Red"),
            ("green_channel", "Green"),
            ("blue_channel", "Blue"),
        ):
            try:
                selection = int(node.params.get(name, -1))
            except (TypeError, ValueError):
                selection = -1
            # In a mixed legacy/manual mapping, -1 is intentionally presented
            # as Unassigned rather than silently filling a positional default.
            if 0 <= selection < count:
                assignments[selection] = self._merge_composite_assignments(
                    assignments[selection],
                    color,
                )
        return assignments

    @staticmethod
    def _normalized_composite_assignment(value: object) -> str:
        text = str(value).strip()
        choices = {
            choice.lower(): choice for choice in COMPOSITE_CHANNEL_ASSIGNMENT_CHOICES
        }
        if text.lower() in choices:
            return choices[text.lower()]
        if re.fullmatch(r"#[0-9a-fA-F]{6}", text):
            return text.upper()
        rgb = color_value_to_rgb(text)
        if rgb is not None:
            values = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)
            return "#" + "".join(f"{int(component):02X}" for component in values)
        return "Unassigned"

    @staticmethod
    def _presented_composite_assignment(value: object) -> str:
        """Return a truthful, non-mutating presentation of one saved value."""
        text = str(value).strip()
        if not text or text.casefold() == "unassigned":
            return "Unassigned"
        choices = {choice.casefold(): choice for choice in CHANNEL_COLOR_CHOICES}
        if text.casefold() in choices:
            return choices[text.casefold()]
        if re.fullmatch(r"#[0-9a-fA-F]{6}", text):
            return text.upper()
        rgb = color_value_to_rgb(text)
        if rgb is not None:
            values = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)
            return "#" + "".join(f"{int(component):02X}" for component in values)
        # Keep malformed text visible in the combo box. Rendering the form is
        # deliberately read-only; a user edit is what repairs persistence.
        return text

    def _composite_channel_assignments(
        self,
        node_id: str,
        mode: str,
        axis: int | None,
        count: int,
    ) -> list[str]:
        node = self.pipeline.nodes[node_id]
        automatic = self._composite_auto_assignments(node_id, axis, count)
        if mode == "Auto":
            return automatic
        raw = str(node.params.get("channel_colors", "")).strip()
        if raw:
            assignments = [
                self._presented_composite_assignment(part) for part in raw.split(",")
            ]
            assignments = assignments[:count]
            assignments.extend(["Unassigned"] * (count - len(assignments)))
            return assignments
        legacy_explicit = False
        for name in ("red_channel", "green_channel", "blue_channel"):
            try:
                legacy_explicit = legacy_explicit or int(node.params.get(name, -1)) >= 0
            except (TypeError, ValueError):
                continue
        if legacy_explicit:
            return self._composite_legacy_assignments(node, count)
        # Explicit/derived Manual mode with no assignments and all legacy
        # selectors at -1 is an intentionally black composite, not Auto.
        return ["Unassigned"] * count

    def _composite_manual_mapping_warning(self, node_id: str, count: int) -> str:
        """Describe malformed saved Manual mapping text without repairing it."""
        node = self.pipeline.nodes[node_id]
        raw = str(node.params.get("channel_colors", ""))
        if not raw.strip():
            return ""
        parts = [part.strip() for part in raw.split(",")]
        problems: list[str] = []
        if len(parts) != count:
            problems.append(
                f"it has {len(parts)} entries but the selected axis has {count}"
            )
        invalid = [
            part
            for part in parts
            if part
            and part.casefold() != "unassigned"
            and color_value_to_rgb(part) is None
        ]
        if invalid:
            quoted = ", ".join(repr(part) for part in invalid)
            problems.append(f"unrecognized assignment(s): {quoted}")
        if not problems:
            return ""
        return (
            f"Saved Manual mapping {raw!r} is invalid: "
            + "; ".join(problems)
            + ". It has not been changed; choose an assignment to replace it "
            "with a canonical full mapping."
        )

    def _composite_channel_labels(
        self,
        node_id: str,
        axis: int | None,
        count: int,
    ) -> list[str]:
        if axis is None:
            return []
        state = self.pipeline.input_state_for_node(node_id)
        declared = self._composite_declared_channel_axes(node_id)
        if len(declared) == 1 and axis == declared[0] and state is not None:
            return self._channel_color_control_labels(count, state)
        axis_name = "Axis"
        if state is not None and 0 <= axis < len(state.axes):
            axis_name = str(state.axes[axis].name).strip().upper() or "Axis"
        return [f"{axis_name} {index + 1}" for index in range(count)]

    def _composite_axis_status(
        self,
        node_id: str,
        mode: str,
        axis: int | None,
    ) -> str:
        declared = self._composite_declared_channel_axes(node_id)
        if axis is None:
            if mode == "Auto" and len(declared) > 1:
                return (
                    "Auto is ambiguous because more than one explicit channel-like "
                    "axis is declared. Switch to Manual and choose the intended axis."
                )
            if mode == "Manual":
                return (
                    "The saved Manual channel axis is missing or invalid. Choose an "
                    "axis to repair it."
                )
            return (
                "Auto could not find one explicit C/channel axis. Switch to Manual "
                "to choose any dimension, such as Z for a colour projection."
            )
        option = next(
            (
                item
                for item in self._axis_slice_options_for(node_id)
                if item.index == axis
            ),
            None,
        )
        detail = self._composite_axis_choice_label(option) if option else str(axis)
        prefix = "Auto resolved" if mode == "Auto" else "Manual selection"
        return f"{prefix}: {detail}."

    @staticmethod
    def _composite_mapping_status(
        mode: str,
        labels: list[str],
        assignments: list[str],
        warning: str = "",
    ) -> str:
        if not assignments:
            mapping = (
                "No source channels are available until a channel axis is resolved."
            )
            return f"{warning} {mapping}".strip()
        mapping = "; ".join(
            f"{label} → {assignment}"
            for label, assignment in zip(labels, assignments, strict=True)
        )
        status = f"{mode} mapping: {mapping}."
        return f"{warning} Displayed mapping: {mapping}." if warning else status

    def _on_composite_channel_axis_mode_changed(self, value) -> None:
        node_id = self._selected_node_id
        node = self.pipeline.nodes.get(node_id)
        if node is None or node.operation_id != "composite_to_rgb":
            return
        mode = str(value).title()
        if mode not in {"Auto", "Manual"} or (
            self._composite_channel_axis_mode(node) == mode
        ):
            return
        self._record_parameter_undo(node_id, "channel_axis_mode")
        if mode == "Manual":
            resolved = self._composite_resolved_channel_axis(node_id, "Auto")
            if resolved is None:
                options = self._axis_slice_options_for(node_id)
                resolved = options[0].index if options else None
            if resolved is not None:
                self.pipeline.set_param(node_id, "channel_axis", int(resolved))
        node.params["channel_axis_mode"] = mode
        self._mark_pipeline_dirty(node_id)
        self._render_parameters(node_id)
        self._debounce_timer.start()

    def _on_composite_channel_axis_changed(self, value) -> None:
        node_id = self._selected_node_id
        node = self.pipeline.nodes.get(node_id)
        if node is None or node.operation_id != "composite_to_rgb":
            return
        try:
            axis = int(value)
        except (TypeError, ValueError):
            return
        try:
            saved_axis = int(node.params.get("channel_axis", -1))
        except (TypeError, ValueError):
            saved_axis = -1
        if saved_axis == axis:
            return
        self._record_parameter_undo(node_id, "channel_axis")
        self.pipeline.set_param(node_id, "channel_axis", axis)
        node.params["channel_axis_mode"] = "Manual"
        count = self._composite_channel_count(node_id, axis)
        if self._composite_mapping_mode(node) == "Manual":
            assignments = self._composite_channel_assignments(
                node_id,
                "Manual",
                axis,
                count,
            )
            node.params["channel_colors"] = ",".join(
                self._normalized_composite_assignment(assignment)
                for assignment in assignments
            )
        self._mark_pipeline_dirty(node_id)
        self._render_parameters(node_id)
        self._debounce_timer.start()

    def _on_composite_mapping_mode_changed(self, value) -> None:
        node_id = self._selected_node_id
        node = self.pipeline.nodes.get(node_id)
        if node is None or node.operation_id != "composite_to_rgb":
            return
        mode = str(value).title()
        if mode not in {"Auto", "Manual"} or self._composite_mapping_mode(node) == mode:
            return
        self._record_parameter_undo(node_id, "mapping_mode")
        axis_mode = self._composite_channel_axis_mode(node)
        axis = self._composite_resolved_channel_axis(node_id, axis_mode)
        count = self._composite_channel_count(node_id, axis)
        if mode == "Manual":
            if str(node.params.get("channel_colors", "")).strip():
                assignments = self._composite_channel_assignments(
                    node_id,
                    "Manual",
                    axis,
                    count,
                )
            else:
                assignments = self._composite_auto_assignments(
                    node_id,
                    axis,
                    count,
                )
            node.params["channel_colors"] = ",".join(
                self._normalized_composite_assignment(assignment)
                for assignment in assignments
            )
        node.params["mapping_mode"] = mode
        self._mark_pipeline_dirty(node_id)
        self._render_parameters(node_id)
        self._debounce_timer.start()

    def _on_composite_channel_color_changed(self, slot: int, value) -> None:
        node_id = self._selected_node_id
        node = self.pipeline.nodes.get(node_id)
        if node is None or node.operation_id != "composite_to_rgb":
            return
        axis = self._composite_resolved_channel_axis(
            node_id,
            self._composite_channel_axis_mode(node),
        )
        count = self._composite_channel_count(node_id, axis)
        assignments = self._composite_channel_assignments(
            node_id,
            "Manual",
            axis,
            count,
        )
        if not 0 <= slot < len(assignments):
            return
        normalized = self._normalized_composite_assignment(value)
        canonical = [
            self._normalized_composite_assignment(assignment)
            for assignment in assignments
        ]
        canonical[slot] = normalized
        serialized = ",".join(canonical)
        if (
            self._composite_mapping_mode(node) == "Manual"
            and str(node.params.get("channel_colors", "")) == serialized
        ):
            return
        self._record_parameter_undo(node_id, "channel_colors")
        node.params["mapping_mode"] = "Manual"
        node.params["channel_colors"] = serialized
        self._mark_pipeline_dirty(node_id)
        self._render_parameters(node_id)
        self._update_thumbnails()
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

    def _split_channel_display_port(
        self,
        node_id: str,
        output_count: int | None = None,
    ) -> int:
        node = self.pipeline.nodes.get(node_id)
        if node is None or node.operation_id != "split_channels":
            return 0
        outputs = self.pipeline.node_outputs.get(node_id) or []
        port_count = max(
            len(outputs) if output_count is None else int(output_count),
            len(self.pipeline.output_ports(node_id)),
            1,
        )
        single_used = self._single_used_split_channel_port(node_id)
        if single_used is not None:
            return int(np.clip(single_used, 0, port_count - 1))
        try:
            index = int(node.params.get("preview_channel", 0))
        except Exception:
            index = 0
        return int(np.clip(index, 0, port_count - 1))

    def _node_output_state(self, node_id: str, output_port: int = 0):
        result = self._background_node_result_override(node_id)
        states = (
            result.node_output_states
            if result is not None
            else self.pipeline.node_output_states.get(node_id) or []
        )
        if 0 <= int(output_port) < len(states) and states[int(output_port)] is not None:
            return states[int(output_port)]
        if int(output_port) == 0:
            return (
                result.output_state
                if result is not None
                else self.pipeline.output_states.get(node_id)
            )
        return None

    def _node_display_payload(self, node_id: str, data=None):
        result = self._background_node_result_override(node_id)
        if result is not None:
            return self._node_display_payload_from_values(
                node_id,
                result.output,
                result.output_state,
                result.node_outputs,
                result.node_output_states,
            )
        primary_data = self.pipeline.outputs.get(node_id) if data is None else data
        primary_state = self.pipeline.output_states.get(node_id)
        return self._node_display_payload_from_values(
            node_id,
            primary_data,
            primary_state,
            self.pipeline.node_outputs.get(node_id) or [],
            self.pipeline.node_output_states.get(node_id) or [],
        )

    def _node_display_payload_from_values(
        self,
        node_id: str,
        primary_data,
        primary_state,
        outputs,
        output_states,
    ):
        node = self.pipeline.nodes.get(node_id)
        if node is None or node.operation_id != "split_channels":
            return primary_data, primary_state, 0
        outputs = list(outputs or [])
        output_states = list(output_states or [])
        if not outputs:
            return primary_data, primary_state, 0
        index = self._split_channel_display_port(node_id, len(outputs))

        def state_for_port(output_port: int):
            if 0 <= int(output_port) < len(output_states):
                state = output_states[int(output_port)]
                if state is not None:
                    return state
            return primary_state if int(output_port) == 0 else None

        if 0 <= index < len(outputs) and outputs[index] is not None:
            return outputs[index], state_for_port(index), index

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
                state_for_port(output_index),
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
        worker = ThumbnailContrastLimitWorker(
            run_id,
            requests,
            calculate_scalar=thumbnail_contrast_limits,
            calculate_channel=thumbnail_channel_contrast_limits,
        )
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
                if _axis_is_explicit(axis) and (
                    axis.type == "channel"
                    or axis.name.lower() in {"c", "rgb", "rgba"}
                ):
                    return index
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

    def _provisional_thumbnail_contrast_limits(
        self,
        data,
        state: ImageState | None,
    ):
        """Return scan-free limits while exact stack contrast is pending."""
        if data is None:
            return None
        try:
            arr = np.asarray(data)
        except Exception:
            return None
        limits = _provisional_generated_layer_contrast_limits(arr)
        channel_axis = self._thumbnail_channel_axis_for_contrast(arr, state)
        if channel_axis is None:
            return limits
        return tuple(limits for _ in range(int(arr.shape[channel_axis])))

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
        node = self.pipeline.nodes.get(node_id)
        channel_axis = _threshold_marker_channel_axis(
            arr,
            params=node.params if node is not None else None,
        )
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
            finite = _finite_values(arr, channel_axis=channel_axis)
            if finite.size:
                return ParameterBounds(
                    int(finite.min()),
                    int(finite.max()),
                    1,
                    0,
                )
            return ParameterBounds(0, 255, 1, 0)

        finite = _finite_values(arr, channel_axis=channel_axis)
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
        height, width = _xy_shape(
            np.asarray(data),
            self.pipeline.input_state_for_node(node_id),
        )
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
        if arr.ndim <= 0:
            return None
        node = self.pipeline.nodes.get(node_id)
        use_stored_axis = not (
            node is not None
            and node.operation_id == "composite_to_rgb"
            and self._composite_channel_axis_mode(node) == "Auto"
        )
        if node is not None and use_stored_axis and "channel_axis" in node.params:
            try:
                stored = int(node.params.get("channel_axis", -1))
            except Exception:
                stored = -1
            if 0 <= stored < arr.ndim:
                return stored
        if node is not None and node.operation_id == "composite_to_rgb":
            declared = self._composite_declared_channel_axes(node_id)
            return declared[0] if len(declared) == 1 else None
        preferred = self._preferred_channel_axis(node_id)
        if preferred is not None and 0 <= preferred < arr.ndim:
            return preferred
        return None

    def _preferred_channel_axis(self, node_id: str) -> int | None:
        state = self.pipeline.input_state_for_node(node_id)
        if state is None:
            return None
        for index, axis in enumerate(state.axes):
            if _axis_is_explicit(axis) and (
                axis.type == "channel" or axis.name.lower() == "c"
            ):
                return index
        return None

    def _block_size_bounds(self, node_id: str, spec) -> ParameterBounds:
        data = self.pipeline.input_data_for_node(node_id)
        if data is None:
            return ParameterBounds(spec.minimum, spec.maximum, spec.step, spec.decimals)
        height, width = _xy_shape(
            np.asarray(data),
            self.pipeline.input_state_for_node(node_id),
        )
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
        return largest_object_size(objects, spatial_ndim, connectivity)

    @staticmethod
    def _largest_label_volume(labels: np.ndarray, spatial_ndim: int) -> int:
        return largest_label_volume(labels, spatial_ndim)

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
        return label_volumes(labels, spatial_ndim)

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
        if self._parameter_visibility_controls_changed(self._selected_node_id):
            self._render_parameters(self._selected_node_id)
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
        if node.operation_id == "clear_border_objects" and name == "boundary_mode":
            buffer_control = self._parameter_widgets.get("border_buffer")
            if isinstance(buffer_control, ParameterControl):
                clamped = buffer_control.value()
                if node.params.get("border_buffer") != clamped:
                    self.pipeline.set_param(node.id, "border_buffer", clamped)
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
            if node.operation_id in COMPACT_DECONVOLUTION_INSPECTOR_OPERATIONS:
                self._update_deconvolution_help_note()
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
            self._update_rescale_input_histogram(
                self._selected_node_id,
                self._current_step(),
            )
        elif node.operation_id == "clip_intensity" and name == "cutoff_mode":
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
                in {
                    "threshold_scope",
                    "histogram_bins",
                    "max_iterations",
                    "channel_axis",
                }
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
        state = self.pipeline.input_state_for_node(node.id)
        if mode.startswith("auto") and not bool(
            getattr(state, "spatial_axes_explicit", False)
        ):
            text = (
                "Auto from axes is unavailable because axis meaning is inferred. "
                "Choose the explicit 2D or 3D mode before calculating."
            )
            color = "#f59e0b"
        elif self._input_spatial_count(node.id) < 3:
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
        state = self.pipeline.input_state_for_node(node_id)
        saturation = float(self.auto_saturation_control.value())
        key = self._auto_contrast_request_key(node_id, data, state, saturation)
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
            worker = AutoContrastWorker(
                request,
                calculate=lambda values, percent: _auto_contrast_scale_offset(
                    values,
                    percent,
                    state=state,
                ),
            )
            worker.signals.finished.connect(self._on_auto_contrast_finished)
            self._pipeline_thread_pool.start(worker, -1)
            return

        result = _auto_contrast_scale_offset(data, saturation, state=state)
        self._commit_auto_contrast_result(
            node_id,
            saturation,
            result,
        )

    def _auto_contrast_request_key(
        self,
        node_id: str,
        data,
        state,
        saturation: float,
    ) -> tuple:
        node = self.pipeline.nodes.get(node_id)
        return (
            node_id,
            id(data),
            tuple(getattr(data, "shape", ())),
            str(getattr(data, "dtype", "")),
            _histogram_state_signature(state),
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
        current_state = self.pipeline.input_state_for_node(result.node_id)
        current_key = self._auto_contrast_request_key(
            result.node_id,
            current_data,
            current_state,
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
        if self._closing:
            return
        if self._collection_batch_running:
            self._collection_batch_graph_refresh_pending = True
            return
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
            self._show_interactive_collection_batch_preview_error(
                self._interactive_collection_batch_requested_index,
                str(exc),
            )
            return
        except Exception as exc:
            self._abandon_background_pipeline_run()
            self._set_pipeline_busy(False)
            self.status_label.setText(f"Image source error: {exc}")
            self._show_interactive_collection_batch_preview_error(
                self._interactive_collection_batch_requested_index,
                str(exc),
            )
            return

        if not source_payloads:
            self._abandon_background_pipeline_run()
            self._set_pipeline_busy(False)
            self._invalidate_pipeline_cache()
            self._restore_hidden_input_layers()
            self.pipeline.run(None)
            self._refresh_selected_parameter_controls()
            self._update_thumbnails()
            self._sync_view_dims_bar()
            self._update_metadata_panel()
            self._update_histogram()
            self._sync_execution_ui()
            self._refresh_cache_status()
            self.status_label.setText("No Image Source node has a selected source.")
            self._show_interactive_collection_batch_preview_error(
                self._interactive_collection_batch_requested_index,
                "no Image Source payload could be loaded",
            )
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
        if self._isolated_tuning_node_id is not None and not source_unchanged:
            self._apply_isolated_tuning(run=False, announce=False)
        dirty_node_ids = self._dirty_nodes_for_run(source_signature)
        target_node_ids: set[str] | None = None
        isolated_node_id = self._isolated_tuning_node_id
        if (
            isolated_node_id is not None
            and dirty_node_ids is not None
            and isolated_node_id in dirty_node_ids
        ):
            dirty_node_ids = {isolated_node_id}
            target_node_ids = {isolated_node_id}
            manual_node_ids = (
                {isolated_node_id}
                if self.pipeline.is_manual_node(isolated_node_id)
                else set()
            )
        else:
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
                target_node_ids,
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
                target_node_ids,
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
            target_node_ids,
        )

    def _abandon_background_pipeline_run(self) -> None:
        """Cancel and detach an in-flight clone whose result must be ignored."""
        run_id = self._active_pipeline_run_id
        if run_id is None:
            cleared_overrides = self._discard_background_node_result_overrides()
            self._refresh_node_presentation_surfaces(cleared_overrides)
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
        target_node_ids: set[str] | None = None,
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
                    target_node_ids=target_node_ids,
                    retain_node_ids=self._cache_retention_node_ids(),
                    prune_unretained=(
                        self._cache_pruning_enabled() and target_node_ids is None
                    ),
                )
        except Exception as exc:
            for node_id in manual_node_ids or ():
                self.pipeline.set_node_execution_error(node_id, str(exc))
            self._sync_execution_ui()
            self.status_label.setText(f"Pipeline error: {exc}")
            self._show_interactive_collection_batch_preview_error(
                self._interactive_collection_batch_requested_index,
                str(exc),
                graph_may_be_partial=True,
            )
            return
        if self._interactive_collection_source_paths:
            # Representative browsing is a transient source override. Keep
            # dtype/series-driven UI normalization from rewriting the exact
            # scientific workflow that the batch config hashes and executes.
            with self._preserve_interactive_collection_workflow_params():
                self._refresh_selected_parameter_controls()
            autodefault_changed: set[str] = set()
            refreshed_selected = False
        else:
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
                        target_node_ids=target_node_ids,
                        retain_node_ids=self._cache_retention_node_ids(),
                        prune_unretained=(
                            self._cache_pruning_enabled()
                            and target_node_ids is None
                        ),
                    )
        self._complete_pipeline_run(source_signature, dirty_node_ids)
        self._finish_pipeline_update(primary_layer, source_label)

    def _finish_pipeline_update(self, primary_layer, source_label: str) -> None:
        self._set_pipeline_busy(True, None, cancelable=False)
        self.pipeline_busy_label.setText("Preparing result display...")
        self._clear_output_histogram_cache()
        self._clear_colocalization_scatter_cache()
        self._hide_input_layer_for_inspection(primary_layer)
        self._apply_cache_retention()
        self._refresh_dynamic_output_ports()
        self._refresh_selected_parameter_visibility()
        self._discard_pending_thumbnail_contrast_limit_requests()
        self._update_thumbnails()
        selected_data, _selected_state, _selected_port = self._node_display_payload(
            self._selected_node_id
        )
        if selected_data is not None and not is_table_data(selected_data):
            # Inspecting the selected node replaces or refreshes this layer. Do
            # not first copy the previously inspected volume only to replace it.
            self._inspect_selected_node()
        else:
            self._refresh_inspection_layer_if_active()
            self._inspect_selected_node()
        self._refresh_pinned_layer_if_active()
        self._sync_view_dims_bar()
        self._update_metadata_panel()
        self._update_histogram()
        self._sync_execution_ui()
        memory_guard_message = self._enforce_memory_guard()
        snapshot_note = (
            " File source snapshots are pinned until Refresh."
            if self._has_active_file_source_snapshot()
            else ""
        )
        if memory_guard_message:
            self.status_label.setText(f"{memory_guard_message}{snapshot_note}")
        elif (
            self._isolated_tuning_node_id in self.pipeline.nodes
            and self._isolated_tuning_has_changes
        ):
            node_id = str(self._isolated_tuning_node_id)
            self.status_label.setText(
                f"Updated '{self._node_title(node_id)}' only. Downstream nodes "
                "remain stale until Apply and continue."
            )
        elif source_label:
            self.status_label.setText(
                f"Graph updated from '{source_label}'. "
                f"Connect ports to build alternate paths.{snapshot_note}"
            )
        else:
            self.status_label.setText("No image source selected.")
        self._refresh_cache_status()
        self._complete_interactive_collection_batch_preview()
        if (
            self._queued_thumbnail_contrast_limit_requests
            or self._pending_thumbnail_contrast_limit_keys
            or self._active_thumbnail_contrast_run_id is not None
        ):
            self._thumbnail_contrast_busy_visible = True
            self._set_pipeline_busy(True, None, cancelable=False)
            self.pipeline_busy_label.setText("Calculating thumbnail contrast...")
        else:
            self._set_pipeline_busy(False)

    def _has_active_file_source_snapshot(self) -> bool:
        return any(
            key in self._file_source_payload_cache
            for node in self.pipeline.nodes.values()
            if node.operation_id == "input"
            for key in (self._file_source_cache_key(node),)
            if key is not None
        )

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
        worker = SourceFileLoadWorker(run_id, specs, reader=read_image)
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
            else:
                self._show_interactive_collection_batch_preview_error(
                    self._interactive_collection_batch_requested_index,
                    result.error,
                )
            return
        try:
            for key, snapshot in result.snapshots.items():
                self._cache_file_source_snapshot(key, snapshot)
        except Exception as exc:
            self._sync_execution_ui()
            self._set_pipeline_busy(False)
            self.status_label.setText(f"Image source error: {exc}")
            if self._source_load_pending:
                self._source_load_pending = False
                QTimer.singleShot(0, self.run_pipeline)
            else:
                self._show_interactive_collection_batch_preview_error(
                    self._interactive_collection_batch_requested_index,
                    str(exc),
                )
            return
        self._prune_file_source_payload_cache()
        self._set_pipeline_busy(False)
        self._source_load_pending = False
        QTimer.singleShot(0, self.run_pipeline)

    def _should_run_pipeline_in_background(
        self,
        dirty_node_ids: set[str] | None = None,
        manual_node_ids: set[str] | None = None,
        source_payloads: dict[str, SourcePayload] | None = None,
        target_node_ids: set[str] | None = None,
    ) -> bool:
        return (
            self._background_processing_node_id(
                dirty_node_ids,
                manual_node_ids,
                source_payloads,
                target_node_ids,
            )
            is not None
        )

    def _background_processing_node_id(
        self,
        dirty_node_ids: set[str] | None = None,
        manual_node_ids: set[str] | None = None,
        source_payloads: dict[str, SourcePayload] | None = None,
        target_node_ids: set[str] | None = None,
    ) -> str | None:
        manual_node_ids = set(manual_node_ids or set())
        execution_plan = self.pipeline.plan_execution(
            dirty_node_ids,
            manual_mode=MANUAL_RUN_SKIP,
            manual_node_ids=manual_node_ids,
            target_node_ids=target_node_ids,
        )
        runnable_node_ids = set(execution_plan.runnable_node_ids)
        if not runnable_node_ids:
            return None
        if manual_node_ids:
            for node_id in self.pipeline.topological_order():
                if node_id in runnable_node_ids:
                    return node_id
            return None

        if self.background_all_checkbox.isChecked():
            for node_id in self.pipeline.topological_order():
                if node_id in runnable_node_ids:
                    return node_id
            return None

        large_source_descendants: set[str] = set()
        for source_id, payload in (source_payloads or {}).items():
            if source_id in self.pipeline.nodes and _should_auto_background_data(
                payload.data
            ):
                large_source_descendants.update(
                    self.pipeline.descendants_inclusive({source_id})
                )
        for node_id in self.pipeline.topological_order():
            if node_id not in runnable_node_ids:
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
        target_node_ids: set[str] | None = None,
    ) -> None:
        manual_node_ids = set(manual_node_ids or set())
        processing_node_id = self._background_processing_node_id(
            dirty_node_ids,
            manual_node_ids,
            source_payloads,
            target_node_ids,
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
            target_node_ids=(
                frozenset(target_node_ids) if target_node_ids is not None else None
            ),
            retain_node_ids=frozenset(self._cache_retention_node_ids()),
            prune_unretained=(
                self._cache_pruning_enabled() and target_node_ids is None
            ),
            cancel_event=cancel_event,
            source_revisions=tuple(
                dict.fromkeys(
                    payload.revision_token
                    for node_id, payload in sorted(source_payloads.items())
                    if isinstance(
                        payload.revision_token,
                        SourceRevisionToken,
                    )
                )
            ),
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
        worker.signals.node_finished.connect(
            self._on_background_pipeline_node_finished
        )
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
        cleared_overrides = self._discard_background_node_result_overrides({node_id})
        self._refresh_node_presentation_surfaces(cleared_overrides)
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

    def _on_background_pipeline_node_finished(
        self,
        result: PipelineNodeResult,
    ) -> None:
        """Publish a completed node's sampled card preview during a graph run."""
        if result.run_id != self._active_pipeline_run_id:
            return
        if result.source_revisions and not self._live_source_adapter.tokens_are_current(
            result.source_revisions
        ):
            return
        node = self.pipeline.nodes.get(result.node_id)
        if node is None or node.operation_id != result.operation_id:
            return
        pending_dirty = self._pending_dirty_node_ids & set(self.pipeline.nodes)
        if (
            self._pipeline_run_pending
            and pending_dirty
            and self._dirty_nodes_affect_node(pending_dirty, result.node_id)
        ):
            return

        preview_data, preview_state, output_port = (
            self._node_display_payload_from_values(
                result.node_id,
                result.output,
                result.output_state,
                result.node_outputs,
                result.node_output_states,
            )
        )
        self._update_node_thumbnail(
            result.node_id,
            preview_data,
            preview_state,
            output_port,
            queue_stack_contrast=False,
        )
        self._background_execution_state_overrides[result.node_id] = (
            result.run_id,
            result.execution_state,
            result.execution_message,
        )
        if result.node_id in self._cache_retention_node_ids():
            self._background_node_result_overrides[result.node_id] = result
        else:
            self._background_node_result_overrides.pop(result.node_id, None)
        self._sync_execution_ui()
        self._refresh_node_presentation_surfaces(
            {result.node_id},
            thumbnails=False,
        )
        self.graph_view.set_node_processing(result.node_id, False)

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
        if result.source_revisions and not self._live_source_adapter.tokens_are_current(
            result.source_revisions
        ):
            self._pipeline_run_pending = False
            self._requeue_inflight_dirty_nodes()
            self._pending_manual_node_ids.update(
                node_id
                for node_id in inflight_manual_node_ids
                if self.pipeline.is_manual_node(node_id)
            )
            self._pending_dirty_node_ids.update(
                node_id
                for node_id in self._live_source_node_layers
                if node_id in self.pipeline.nodes
            )
            self._set_pipeline_busy(False)
            self.status_label.setText(
                "Discarded a result from an old live-source revision; "
                "calculating the current source."
            )
            QTimer.singleShot(0, self.run_pipeline)
            return
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
            self._show_interactive_collection_batch_preview_error(
                self._interactive_collection_batch_requested_index,
                "background processing was canceled",
            )
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
            else:
                self._show_interactive_collection_batch_preview_error(
                    self._interactive_collection_batch_requested_index,
                    result.error,
                )
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
            else:
                self._show_interactive_collection_batch_preview_error(
                    self._interactive_collection_batch_requested_index,
                    "no pipeline result was returned",
                )
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
        if self._interactive_collection_source_paths:
            with self._preserve_interactive_collection_workflow_params():
                self._refresh_selected_parameter_controls()
            autodefault_changed: set[str] = set()
            refreshed_selected = False
        else:
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
        self._discard_background_node_result_overrides()

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
        self._show_interactive_collection_batch_preview_error(
            self._interactive_collection_batch_requested_index,
            "background processing was canceled",
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
            cleared_overrides = self._discard_background_node_result_overrides()
            self.pipeline_busy_label.setText("Processing")
            self.pipeline_busy_bar.setRange(0, 0)
            self.pipeline_busy_bar.setTextVisible(False)
            self.graph_view.clear_node_processing()
            self._active_pipeline_node_id = None
            self._sync_execution_ui()
            self._refresh_node_presentation_surfaces(cleared_overrides)
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
        if self._closing:
            return
        if not self._dims_linked():
            return
        self._capture_vipp_dims_from_viewer()
        self._sync_view_dims_bar()
        self._update_thumbnails()
        self._update_metadata_panel()
        self._update_histogram()

    def _update_thumbnails(self) -> None:
        for node_id, data in self.pipeline.outputs.items():
            preview_data, preview_state, output_port = self._node_display_payload(
                node_id,
                data,
            )
            self._update_node_thumbnail(
                node_id,
                preview_data,
                preview_state,
                output_port,
                queue_stack_contrast=True,
            )
            self.graph_view.set_node_can_pin(node_id, self._node_can_pin(node_id))
        if self._active_pinned_node_id is not None and not self._node_can_pin(
            self._active_pinned_node_id
        ):
            self._clear_active_pin(status=False)
        else:
            self._sync_pin_ui()

    def _update_node_thumbnail(
        self,
        node_id: str,
        preview_data,
        preview_state,
        output_port: int,
        *,
        queue_stack_contrast: bool,
    ) -> None:
        """Render one card without requiring its result in the live pipeline cache."""
        mode = self.preview_mode_combo.currentText()
        contrast_mode = self.thumbnail_contrast_combo.currentText()
        contrast_scope = self.thumbnail_scope_combo.currentText()
        node_output_type = self._node_output_type_for_payload(
            node_id,
            preview_data,
            output_port,
        )
        metadata_text = format_compact_metadata(preview_state)
        batch_text = self._interactive_collection_card_metadata(node_id)
        if batch_text:
            metadata_text = (
                f"{batch_text}\n{metadata_text}"
                if metadata_text and metadata_text != "No output"
                else batch_text
            )
        self.graph_view.set_node_metadata(node_id, metadata_text)
        self.graph_view.set_node_output_type(node_id, node_output_type)
        preview_enabled = (
            mode.lower() != "off"
            and node_output_type != "table"
            and node_id not in self._preview_disabled_node_ids
        )
        self.graph_view.set_node_preview_enabled(node_id, preview_enabled)
        if not preview_enabled:
            self.graph_view.set_thumbnail(node_id, None)
            return

        contrast_limits = None
        if queue_stack_contrast:
            contrast_limits = self._thumbnail_contrast_limits_for_node(
                node_id,
                preview_data,
                preview_state,
                contrast_mode,
                contrast_scope,
                node_output_type,
            )
        stack_scope = not str(contrast_scope).strip().lower().startswith("slice")
        if stack_scope and contrast_limits is None:
            # Incremental results deliberately skip an exact full-stack display
            # scan. The final graph publication queues that presentation-only
            # work; dtype limits keep this first thumbnail immediate.
            contrast_limits = self._provisional_thumbnail_contrast_limits(
                preview_data,
                preview_state,
            )
        effective_scope_is_slice = str(contrast_scope).strip().lower().startswith(
            "slice"
        )
        preview_consumes_contrast = self._thumbnail_preview_consumes_contrast(
            node_id,
            preview_data,
            preview_state,
        )
        preview = make_preview(
            preview_data,
            mode=mode,
            current_step=self._current_step(),
            current_step_nsteps=self._current_step_nsteps(),
            state=preview_state,
            channel_colors=self._node_preview_channel_colors(node_id),
            contrast_mode=contrast_mode,
            contrast_scope=contrast_scope,
            contrast_limits=contrast_limits,
            preview_size=(180, 110),
        )
        thumbnail = normalize_thumbnail_with_colormap(
            preview,
            colormap=self.thumbnail_colormap_combo.currentText(),
            contrast_mode=contrast_mode,
            contrast_reference=(preview if effective_scope_is_slice else None),
            contrast_limits=(None if preview_consumes_contrast else contrast_limits),
            data_kind=node_output_type,
        )
        self.graph_view.set_thumbnail(node_id, thumbnail)

    def _interactive_collection_card_metadata(self, node_id: str) -> str:
        items = self._interactive_collection_batch_items
        index = self._interactive_collection_batch_index
        if not items or not 0 <= index < len(items):
            return ""
        path = items[index].source_paths.get(node_id)
        if path is None:
            return ""
        return f"Batch {index + 1}/{len(items)} · {Path(path).name}"

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
        worker = ColocalizationScatterWorker(
            request,
            normalized_inputs=colocalization_normalized_inputs,
            threshold_values=colocalization_threshold_values,
            scatter_density=_prepare_colocalization_scatter_density,
        )
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
            node.operation_id == "clip_intensity"
            and not str(node.params.get("cutoff_mode", "Data range"))
            .lower()
            .startswith("value")
        ):
            return
        name = _input_histogram_marker_parameter(node.operation_id, label)
        if name is None:
            return
        switched_rescale_mode = False
        if (
            node.operation_id == "rescale_intensity"
            and not str(
                node.params.get(RESCALE_CUTOFF_MODE_PARAMETER, "Percentiles")
            ).lower().startswith("value")
        ):
            switched_rescale_mode = (
                self._switch_rescale_histogram_to_value_cutoffs(
                    node_id,
                    history_name=name,
                )
            )
            if not switched_rescale_mode:
                return
        value = self._paired_histogram_marker_value(node_id, name, value)
        value = self._coerce_histogram_parameter_value(node_id, name, value)
        parameter_changed = self._set_histogram_parameter_value(
            node_id,
            name,
            value,
        )
        if not switched_rescale_mode and not parameter_changed:
            return
        if switched_rescale_mode:
            self._render_parameters(node_id)
            self.status_label.setText(
                "Rescale input cutoffs switched to explicit values for "
                "histogram dragging."
            )
        self._mark_pipeline_dirty(node_id)
        self._update_rescale_input_histogram(
            node_id,
            self._current_step(),
        )
        self._debounce_timer.start()

    def _switch_rescale_histogram_to_value_cutoffs(
        self,
        node_id: str,
        *,
        history_name: str,
    ) -> bool:
        """Make a percentile marker drag an exact explicit-value edit."""
        node = self.pipeline.nodes.get(node_id)
        if node is None or node.operation_id != "rescale_intensity":
            return False
        marker_values = self.rescale_input_histogram_plot.marker_values()
        low = marker_values.get("low")
        if low is None:
            return False
        high = marker_values.get("high", low)
        try:
            low_value = float(_finite_marker_value(low, "Low input value"))
            high_value = float(_finite_marker_value(high, "High input value"))
        except ValueError:
            return False

        self._record_parameter_undo(node_id, history_name)
        self.pipeline.set_param(
            node_id,
            RESCALE_CUTOFF_MODE_PARAMETER,
            "Values",
        )
        self.pipeline.set_param(node_id, "in_low_value", low_value)
        self.pipeline.set_param(node_id, "in_high_value", high_value)
        return True

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
        worker = InputHistogramWorker(
            request,
            histogram_summary=_histogram_summary,
            histogram_source=_histogram_source,
            histogram_markers=_input_histogram_markers,
        )
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
        worker = InputHistogramWorker(
            request,
            histogram_summary=_histogram_summary,
            histogram_source=_histogram_source,
            histogram_markers=_input_histogram_markers,
        )
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
        data, _state, _output_port = self._node_display_payload(
            self._selected_node_id
        )
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
                and bool(getattr(state, "spatial_axes_explicit", False))
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
            self._invalidate_generated_layer_contrast(layer)
            self._remove_layer(layer)
            self._add_image_or_labels(
                name,
                data,
                metadata=metadata,
                display_data=display_data,
            )
            self._restore_viewer_step(saved_step, saved_nsteps)
            return
        self._invalidate_generated_layer_contrast(layer)
        layer.data = _read_only_presentation_array(display_data)
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
        presentation_data = _read_only_presentation_array(display_data)
        scale = _layer_scale_from_metadata(metadata)
        if metadata["display_kind"] == "labels" and hasattr(self.viewer, "add_labels"):
            kwargs = {"name": name, "metadata": metadata}
            if scale is not None:
                kwargs["scale"] = scale
            layer = self.viewer.add_labels(presentation_data, **kwargs)
            self._make_generated_layer_noneditable(layer)
            return layer
        kwargs = {"name": name, "metadata": metadata}
        if scale is not None:
            kwargs["scale"] = scale
        if metadata.get("display_rgb"):
            kwargs["rgb"] = True
        kwargs["blending"] = "translucent"
        if not metadata.get("display_rgb"):
            kwargs["colormap"] = "gray"
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
        layer = self.viewer.add_image(presentation_data, **kwargs)
        self._make_generated_layer_noneditable(layer)
        return layer

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
            self._invalidate_generated_layer_contrast(base_layer)
            self._remove_layer(base_layer)
        for index, channel_name, colormap in _RGB_VOLUME_CHANNELS:
            channel_data = _read_only_presentation_array(arr[..., index])
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
                self._invalidate_generated_layer_contrast(layer)
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
            self._invalidate_generated_layer_contrast(layer)
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
        layer = self.viewer.add_image(data, **kwargs)
        self._make_generated_layer_noneditable(layer)
        return layer

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
        self._make_generated_layer_noneditable(layer)

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
            self._invalidate_generated_layer_contrast(layer)
            self._remove_layer(layer)

    def _remove_extra_rgb_channel_layers(self, group_name: str) -> None:
        expected = {
            _rgb_channel_layer_name(group_name, index)
            for index, _channel_name, _colormap in _RGB_VOLUME_CHANNELS
        }
        for layer in self._rgb_channel_layers(group_name):
            if layer.name not in expected:
                self._invalidate_generated_layer_contrast(layer)
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
            or layer.metadata.get("display_ndim") != metadata["display_ndim"]
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
            self._make_generated_layer_noneditable(layer)
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
            for attr, value in (("blending", "translucent"),):
                try:
                    setattr(layer, attr, value)
                except Exception:
                    pass
            if not metadata.get("display_rgb"):
                try:
                    layer.colormap = "gray"
                except Exception:
                    pass
            plan = self._generated_layer_contrast_plan(layer.name, data)
            layer.metadata.update(self._generated_layer_contrast_metadata(plan))
            try:
                layer.contrast_limits = plan.limits
            except Exception:
                pass
        self._make_generated_layer_noneditable(layer)

    @staticmethod
    def _make_generated_layer_noneditable(layer) -> None:
        try:
            layer.editable = False
        except Exception:
            # Older napari versions and lightweight test viewers may not expose
            # an editable property. The read-only NumPy view remains the safety
            # boundary in those environments.
            pass

    def _invalidate_generated_layer_contrast(self, layer) -> None:
        """Detach a layer from in-flight contrast results before reusing it."""
        layer_name = str(getattr(layer, "name", ""))
        if layer_name:
            self._generated_layer_contrast_keys.pop(layer_name, None)
        try:
            metadata = layer.metadata
        except Exception:
            return
        for key in _GENERATED_LAYER_CONTRAST_METADATA_KEYS:
            try:
                metadata.pop(key, None)
            except Exception:
                return

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
                worker = GeneratedLayerContrastWorker(
                    request,
                    calculate=_exact_generated_layer_contrast_limits,
                )
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
        if state is None or len(getattr(state, "axes", ())) != arr.ndim:
            return False
        channel_axis = state.axes[-1]
        if not _axis_is_explicit(channel_axis) or not (
            channel_axis.type == "channel"
            or channel_axis.name.lower() in {"c", "rgb", "rgba"}
        ):
            return False
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
        data, _state, output_port = self._node_display_payload(node_id)
        return self._node_output_type_for_payload(node_id, data, output_port)

    def _node_output_type_for_payload(
        self,
        node_id: str,
        data,
        output_port: int,
    ) -> str:
        node = self.pipeline.nodes.get(node_id)
        ports = self.pipeline.output_ports(node_id)
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


def _finite_values(
    arr: np.ndarray,
    *,
    channel_axis: int | None = None,
) -> np.ndarray:
    reference = _explicit_luminance_reference(
        np.asarray(arr),
        channel_axis=channel_axis,
        context="Threshold range",
    )
    return reference[np.isfinite(reference)]


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
    *,
    state=None,
) -> tuple[float, float, float, float] | None:
    if data is None:
        return None

    try:
        saturation = float(saturation_percent)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Auto-contrast saturation must be a finite number.") from exc
    if not np.isfinite(saturation):
        raise ValueError("Auto-contrast saturation must be a finite number.")
    if not 0.0 <= saturation <= 100.0:
        raise ValueError("Auto-contrast saturation must be between 0 and 100 percent.")
    tail_percent = saturation / 2.0
    reference = np.asarray(data)
    channel_axis = _explicit_encoded_color_axis(reference, state)
    reference = _explicit_luminance_reference(
        reference,
        channel_axis=channel_axis,
        context="Auto contrast",
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


def _explicit_encoded_color_axis(arr: np.ndarray, state) -> int | None:
    """Resolve an encoded RGB(A) axis only from explicit image semantics."""
    axes = tuple(getattr(state, "axes", ()))
    if len(axes) != arr.ndim:
        return None
    matches = [
        index
        for index, axis in enumerate(axes)
        if _axis_is_explicit(axis)
        and str(getattr(axis, "type", "")).lower() == "channel"
        and str(getattr(axis, "name", "")).lower() in {"rgb", "rgba"}
    ]
    if len(matches) > 1:
        raise ValueError("Image metadata declares more than one encoded RGB(A) axis.")
    if not matches:
        return None
    axis = matches[0]
    name = str(getattr(axes[axis], "name", "")).lower()
    expected = 3 if name == "rgb" else 4
    if int(arr.shape[axis]) != expected:
        raise ValueError(
            f"Explicit {name.upper()} axis must contain exactly {expected} channels, "
            f"not {int(arr.shape[axis])}."
        )
    return axis


def _explicit_luminance_reference(
    arr: np.ndarray,
    *,
    channel_axis: int | None,
    context: str,
) -> np.ndarray:
    """Return scalar data or BT.601 luma for a caller-declared RGB(A) axis."""
    if channel_axis is None:
        return arr
    if isinstance(channel_axis, (bool, np.bool_)) or not isinstance(
        channel_axis,
        (int, np.integer),
    ):
        raise ValueError(f"{context} channel axis must be an integer or None.")
    axis = int(channel_axis)
    if axis < -arr.ndim or axis >= arr.ndim:
        raise ValueError(
            f"{context} channel axis {axis} is outside an array with {arr.ndim} axes."
        )
    axis %= arr.ndim
    channel_count = int(arr.shape[axis])
    if channel_count not in {3, 4}:
        raise ValueError(
            f"{context} channel axis must contain exactly 3 RGB or 4 RGBA "
            f"channels, not {channel_count}."
        )
    if not (
        arr.dtype == bool
        or np.issubdtype(arr.dtype, np.integer)
        or np.issubdtype(arr.dtype, np.floating)
    ):
        raise ValueError(f"{context} requires real-valued image data.")
    moved = np.moveaxis(arr, axis, -1)
    work_dtype = np.result_type(arr.dtype, np.float32)
    rgb = moved[..., :3].astype(work_dtype, copy=False)
    coefficients = np.asarray((0.299, 0.587, 0.114), dtype=work_dtype)
    return np.sum(rgb * coefficients, axis=-1, dtype=work_dtype)


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


def _expanded_bounds(value: float) -> tuple[float, float]:
    padding = abs(value) * 0.1 or 1.0
    return value - padding, value + padding


def _axis_is_explicit(axis) -> bool:
    """Return whether carried axis semantics came from an explicit source."""
    return bool(getattr(axis, "is_explicit", False))


def _xy_shape(arr: np.ndarray, state=None) -> tuple[int, int]:
    """Resolve Y/X sizes without inferring RGB from an array dimension."""
    if arr.ndim < 2:
        return 1, 1
    axes = tuple(getattr(state, "axes", ()))
    if len(axes) == arr.ndim:
        y_axis = next(
            (
                index
                for index, axis in enumerate(axes)
                if _axis_is_explicit(axis) and axis.name.lower() == "y"
            ),
            None,
        )
        x_axis = next(
            (
                index
                for index, axis in enumerate(axes)
                if _axis_is_explicit(axis) and axis.name.lower() == "x"
            ),
            None,
        )
        if y_axis is not None and x_axis is not None and y_axis != x_axis:
            return int(arr.shape[y_axis]), int(arr.shape[x_axis])
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
    counts, x_range = exact_histogram(arr, channel_axis=channel_axis)
    if counts is None:
        return None, None, None
    series_count = counts.shape[0] if counts.ndim > 1 else 1
    colors = _histogram_series_colors(series_count, channel_axis_name)
    return counts, x_range, colors


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
            if name in {"x", "y"}:
                continue
            if _axis_is_explicit(axis) and (
                name in {"rgb", "rgba"} or axis.type == "channel"
            ):
                continue
            return True
        return False
    return True


def _histogram_state_signature(state) -> tuple:
    axes = tuple(getattr(state, "axes", ()))
    return tuple(
        (
            str(getattr(axis, "name", "")).casefold(),
            str(getattr(axis, "type", "")).casefold(),
            str(getattr(axis, "confidence", "")).casefold(),
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
        if "channel_axis" in values:
            names.append("channel_axis")
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
        threshold = _finite_marker_value(
            (params or {}).get("threshold"),
            "Binary threshold",
        )
        return [("threshold", float(threshold), QColor("#f59e0b"))]
    if operation_id == "hysteresis_threshold":
        low = _finite_marker_value(
            (params or {}).get("low_threshold"),
            "Hysteresis low threshold",
        )
        high = _finite_marker_value(
            (params or {}).get("high_threshold"),
            "Hysteresis high threshold",
        )
        if low > high:
            raise ValueError(
                "Hysteresis low threshold must not exceed the high threshold."
            )
        markers = [("low", float(low), QColor("#f59e0b"))]
        if not np.isclose(low, high):
            markers.append(("high", float(high), QColor("#38bdf8")))
        return markers
    if operation_id in GLOBAL_THRESHOLD_OPERATIONS:
        channel_axis = _threshold_marker_channel_axis(
            data,
            params=params,
        )
        source = _threshold_marker_source(
            data,
            channel_axis=channel_axis,
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
            channel_axis=source[1],
        )
        if value is None or not np.isfinite(value):
            return []
        marker_value = (
            int(value) if isinstance(value, (int, np.integer)) else float(value)
        )
        return [("threshold", marker_value, QColor("#f59e0b"))]
    return []


def _threshold_marker_channel_axis(
    data,
    *,
    params: dict | None,
) -> int | None:
    """Resolve only the numeric channel axis persisted by the threshold node."""
    value = (params or {}).get("channel_axis", -1)
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise ValueError("Threshold channel_axis must be an integer or -1.")
    axis = int(value)
    if axis == -1:
        return None
    ndim = np.asarray(data).ndim
    if not 0 <= axis < ndim:
        raise ValueError(
            f"Threshold channel_axis {axis} is outside an array with {ndim} axes."
        )
    return axis


def _threshold_marker_source(
    data,
    *,
    channel_axis: int | None,
    state=None,
    scope: str = "Slice",
    current_step=None,
    current_step_nsteps=None,
) -> tuple[np.ndarray, int | None]:
    """Return the exact positional plane used by a threshold operation marker."""
    arr = np.asarray(data)
    if str(scope).strip().lower().startswith("stack"):
        return arr, channel_axis

    scalar_axes = [
        axis for axis in range(arr.ndim) if axis != channel_axis
    ]
    if len(scalar_axes) <= 2:
        return arr, channel_axis

    keep_axes = set(scalar_axes[-2:])
    if channel_axis is not None:
        keep_axes.add(channel_axis)
    result = arr
    remaining_axes = list(range(arr.ndim))
    for original_axis in reversed(range(arr.ndim)):
        if original_axis in keep_axes:
            continue
        local_axis = remaining_axes.index(original_axis)
        step_axis = _metadata_current_step_axis(
            state,
            original_axis,
            current_step,
        )
        index = _histogram_axis_index(
            step_axis,
            int(result.shape[local_axis]),
            current_step,
            current_step_nsteps=current_step_nsteps,
        )
        result = _axis_index_view(result, local_axis, index)
        remaining_axes.pop(local_axis)

    if channel_axis is None:
        return result, None
    return result, remaining_axes.index(channel_axis)


def _input_histogram_draggable_markers(
    operation_id: str,
    params: dict | None = None,
) -> set[str]:
    if operation_id == "rescale_intensity":
        return {"low", "high"}
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


def _read_only_presentation_array(data) -> np.ndarray:
    """Return an exact zero-copy, non-writeable presentation view."""
    view = np.asarray(data).view()
    view.flags.writeable = False
    return view


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
            if (
                _axis_is_explicit(axis)
                and axis.type == "channel"
                and arr.shape[index] > 1
            ):
                return index, axis.name.lower()
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


def _axis_index_view(arr: np.ndarray, axis: int, index: int) -> np.ndarray:
    selection = [slice(None)] * arr.ndim
    selection[int(axis)] = int(index)
    return arr[tuple(selection)]




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
    """Return available and total memory using the current platform API."""
    if sys.platform == "win32":
        return _windows_memory_bytes()
    elif sys.platform == "darwin":
        return _macos_memory_bytes()

    sysconf = getattr(os, "sysconf", None)
    if sysconf is None:
        return None, None
    try:
        page_size = int(sysconf("SC_PAGE_SIZE"))
        total_pages = int(sysconf("SC_PHYS_PAGES"))
        available_pages = int(sysconf("SC_AVPHYS_PAGES"))
    except (AttributeError, OSError, TypeError, ValueError):
        return None, None
    return available_pages * page_size, total_pages * page_size


class _MacOSVMStatistics64(ctypes.Structure):
    _fields_ = [
        ("free_count", ctypes.c_uint32),
        ("active_count", ctypes.c_uint32),
        ("inactive_count", ctypes.c_uint32),
        ("wire_count", ctypes.c_uint32),
        ("zero_fill_count", ctypes.c_uint64),
        ("reactivations", ctypes.c_uint64),
        ("pageins", ctypes.c_uint64),
        ("pageouts", ctypes.c_uint64),
        ("faults", ctypes.c_uint64),
        ("cow_faults", ctypes.c_uint64),
        ("lookups", ctypes.c_uint64),
        ("hits", ctypes.c_uint64),
        ("purges", ctypes.c_uint64),
        ("purgeable_count", ctypes.c_uint32),
        ("speculative_count", ctypes.c_uint32),
        ("decompressions", ctypes.c_uint64),
        ("compressions", ctypes.c_uint64),
        ("swapins", ctypes.c_uint64),
        ("swapouts", ctypes.c_uint64),
        ("compressor_page_count", ctypes.c_uint32),
        ("throttled_count", ctypes.c_uint32),
        ("external_page_count", ctypes.c_uint32),
        ("internal_page_count", ctypes.c_uint32),
        ("total_uncompressed_pages_in_compressor", ctypes.c_uint64),
    ]


def _macos_memory_bytes() -> tuple[int | None, int | None]:
    """Return macOS available and total physical memory without subprocesses."""
    sysconf = getattr(os, "sysconf", None)
    if sysconf is None:
        return None, None
    try:
        page_size = int(sysconf("SC_PAGE_SIZE"))
        total_pages = int(sysconf("SC_PHYS_PAGES"))
        lib_system = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        lib_system.mach_host_self.restype = ctypes.c_uint32
        host = lib_system.mach_host_self()
        statistics = _MacOSVMStatistics64()
        count = ctypes.c_uint32(
            ctypes.sizeof(statistics) // ctypes.sizeof(ctypes.c_int)
        )
        result = lib_system.host_statistics64(
            host,
            4,  # HOST_VM_INFO64
            ctypes.byref(statistics),
            ctypes.byref(count),
        )
    except (AttributeError, OSError, TypeError, ValueError):
        return None, None
    if result != 0:
        return None, None
    available_pages = int(statistics.free_count) + int(statistics.inactive_count)
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
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    try:
        ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    except (AttributeError, OSError, TypeError, ValueError):
        return None, None
    if not ok:
        return None, None
    return int(status.ullAvailPhys), int(status.ullTotalPhys)
