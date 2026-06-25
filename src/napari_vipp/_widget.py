"""napari dock widget for the VIPP workflow prototype."""

from __future__ import annotations

import inspect as py_inspect
import re
import textwrap
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, replace
from html import escape
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
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
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
)
from scipy import ndimage as ndi

from napari_vipp._graph import OPERATION_MIME, PipelineGraphView
from napari_vipp._sample_data import make_sample_data
from napari_vipp._theme import category_color, category_tint
from napari_vipp.core.channel_colors import (
    CHANNEL_COLOR_CHOICES,
    CHANNEL_COLOR_HEX,
    FLUORESCENCE_COLORS,
    channel_color_labels_from_metadata,
)
from napari_vipp.core.export import export_pipeline_to_python
from napari_vipp.core.io import (
    AnalysisLabel,
    SourceInspection,
    inspect_image_source,
    read_image,
    write_ome_zarr_analysis_dataset,
)
from napari_vipp.core.metadata import (
    MetadataRow,
    format_compact_metadata,
    image_state_from_array,
    metadata_history_items,
    metadata_table_rows,
)
from napari_vipp.core.operations import automatic_threshold_value, save_array_output
from napari_vipp.core.pipeline import (
    GLOBAL_THRESHOLD_OPERATIONS,
    OperationSpec,
    ParameterSpec,
    PrototypePipeline,
    SourcePayload,
    grouped_palette_specs,
)
from napari_vipp.core.preview import (
    MONOCHROME_COLORMAPS,
    THUMBNAIL_CONTRAST_MODES,
    make_preview,
    normalize_thumbnail_with_colormap,
)
from napari_vipp.core.tables import is_table_data, save_table_output
from napari_vipp.core.workflow import (
    deserialize_workflow,
    load_workflow,
    save_workflow,
    serialize_workflow,
)

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


@dataclass(frozen=True)
class PipelineRunResult:
    run_id: int
    workflow: dict
    pipeline: PrototypePipeline | None = None
    error: str = ""


class PipelineRunSignals(QObject):
    node_started = Signal(object)
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
            pipeline.restore_graph(workflow["nodes"], workflow["connections"])
            self._hydrate_cached_pipeline_outputs(pipeline)
            pipeline.run(
                self.request.input_data,
                input_metadata=self.request.input_metadata,
                input_name=self.request.input_name,
                source_payloads=self.request.source_payloads,
                dirty_node_ids=self.request.dirty_node_ids,
                node_started_callback=self._emit_node_started,
            )
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
        pipeline.completed_node_ids = set(self.request.completed_node_ids)


RESCALE_VALUE_PARAMETERS = {"in_low_value", "in_high_value"}
RESCALE_PERCENTILE_PARAMETERS = {"in_low_percentile", "in_high_percentile"}
RESCALE_CUTOFF_PARAMETERS = RESCALE_VALUE_PARAMETERS | RESCALE_PERCENTILE_PARAMETERS
CLIP_CUTOFF_PARAMETERS = {"minimum", "maximum"}
INPUT_HISTOGRAM_OPERATIONS = {
    "binary_threshold",
    "clip_intensity",
    "hysteresis_threshold",
    "rescale_intensity",
} | GLOBAL_THRESHOLD_OPERATIONS
BACKGROUND_PIPELINE_OPERATIONS = {
    "euclidean_distance_transform",
    "gaussian_blur_3d",
    "h_maxima_markers",
    "marker_controlled_watershed",
    "non_local_means_filter",
    "orthogonal_projection",
    "project_image",
    "rescale_axes",
    "rolling_ball_background",
    "subtract_background",
}
ROLLING_BALL_RADIUS_SLIDER_MAX = 100.0
MAX_CHANNEL_COLOR_CONTROLS = 12


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


def _toolbar_separator() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.VLine)
    line.setFrameShadow(QFrame.Sunken)
    line.setFixedWidth(12)
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


def _configure_numeric_spin_box(box: QSpinBox | QDoubleSpinBox) -> None:
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
                bounds.minimum
                if bounds.entry_minimum is None
                else bounds.entry_minimum
            )
            maximum = (
                bounds.maximum
                if bounds.entry_maximum is None
                else bounds.entry_maximum
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
            "*.ppm *.pnm *.npy *.npz);;"
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
        self._active_handle = (
            "start" if abs(x - start_x) <= abs(x - end_x) else "end"
        )
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
                f"{axis}:{start}:{end}"
                for axis, (start, end) in sorted(ranges.items())
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
            axis: value
            for axis, value in ranges.items()
            if axis in option_indices
        }
        removals = {
            axis: value
            for axis, value in removals.items()
            if axis in option_indices
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
        self.setMinimumHeight(120)

    def set_histogram(
        self,
        counts: np.ndarray | None,
        log_scale: bool,
        x_range: tuple[float, float] | None = None,
        colors: list[QColor] | None = None,
        markers: list[tuple[str, float, QColor]] | None = None,
        x_scale: str = "linear",
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
            text_x = int(
                np.clip(x + 3, plot_rect.left(), rightmost_text_x)
            )
            painter.setPen(color)
            painter.drawText(
                text_x,
                label_y + index * (metrics.height() + 1),
                text,
            )

    def _x_fraction(self, value: float) -> float:
        if self._x_range is None:
            return 0.0
        minimum, maximum = self._x_range
        if maximum <= minimum:
            return 0.0
        value = float(np.clip(value, minimum, maximum))
        if self._x_scale == "log":
            shifted_value = max(value - minimum, 0.0)
            shifted_maximum = maximum - minimum
            return float(
                np.log1p(shifted_value) / np.log1p(max(shifted_maximum, 1.0))
            )
        return float((value - minimum) / (maximum - minimum))


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

    def __init__(self, viewer: napari.viewer.Viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.pipeline = PrototypePipeline()
        self._selected_node_id = "gaussian"
        self._active_pinned_node_id: str | None = None
        self._inspect_layer_name = "VIPP Inspect"
        self._last_input_layer_name: str | None = None
        self._preview_disabled_node_ids: set[str] = set()
        self._hidden_input_layer_states: dict[int, tuple[object, bool]] = {}
        self._sample_payload_cache: dict[str, SourcePayload] | None = None
        self._source_inspection_cache: dict[
            str, tuple[int, SourceInspection]
        ] = {}
        self._dock_chrome_configured = False
        self._dock_window_behavior_configured = False
        self._initial_dock_size_applied = False
        self._undo_stack: list[WorkflowHistorySnapshot] = []
        self._redo_stack: list[WorkflowHistorySnapshot] = []
        self._restoring_history = False
        self._pending_parameter_undo_key: tuple[str, str] | None = None
        self._clip_auto_input_ranges: dict[str, tuple[float, float]] = {}
        self._rescale_auto_input_cutoffs: dict[str, tuple[float, float]] = {}
        self._rescale_auto_output_ranges: dict[str, tuple[float, float]] = {}
        self._code_dialogs: list[QDialog] = []
        self._pending_dirty_node_ids: set[str] = set()
        self._last_pipeline_source_signature: tuple | None = None
        self.setMinimumSize(0, 0)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)

        self.layer_combo = QComboBox()
        self.layer_combo.setMinimumWidth(220)
        self.global_thumbnail_checkbox = QCheckBox("Thumbnails")
        self.global_thumbnail_checkbox.setChecked(True)
        self.preview_mode_combo = QComboBox()
        self.preview_mode_combo.addItems(["Slice", "MIP", "Off"])
        self.thumbnail_contrast_combo = QComboBox()
        self.thumbnail_contrast_combo.addItems(THUMBNAIL_CONTRAST_MODES)
        self.thumbnail_colormap_combo = QComboBox()
        self.thumbnail_colormap_combo.addItems(MONOCHROME_COLORMAPS)
        self.follow_dims_checkbox = QCheckBox("Follow napari dims")
        self.follow_dims_checkbox.setChecked(True)
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
        self.refresh_button = QPushButton("Refresh")
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
        self.load_workflow_button = QPushButton("Load workflow...")
        self.export_button = QPushButton("Export Python...")
        self.export_ome_button = QPushButton("Export OME dataset...")
        self.background_all_checkbox = QCheckBox("Run all in BG")
        self.background_all_checkbox.setChecked(False)
        self.background_all_checkbox.setToolTip(
            "Run all pipeline updates in background. "
            "When off, only known-slower operations use background processing."
        )
        self.pipeline_busy_label = QLabel("Processing")
        self.pipeline_busy_label.setStyleSheet("color: #93c5fd; font-weight: 650;")
        self.pipeline_busy_bar = QProgressBar()
        self.pipeline_busy_bar.setRange(0, 0)
        self.pipeline_busy_bar.setTextVisible(False)
        self.pipeline_busy_bar.setFixedWidth(96)
        self.pipeline_busy_bar.setFixedHeight(12)
        self.pipeline_busy_label.setVisible(False)
        self.pipeline_busy_bar.setVisible(False)
        self.status_label = QLabel(
            "Select an image layer and build the starter pipeline."
        )
        self.status_label.setWordWrap(True)

        self.palette_search = QLineEdit()
        self.palette_search.setPlaceholderText("Search nodes")
        self.palette_search.setClearButtonEnabled(True)
        self.palette = NodePalette(grouped_palette_specs())
        self.palette.setMinimumWidth(190)
        self.palette.setMinimumHeight(0)
        self.palette_panel = self._build_palette_panel()
        self.graph_view = PipelineGraphView()
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

        self.selected_title = QLabel("Gaussian Blur")
        self.selected_title.setStyleSheet("font-weight: 650;")
        self.thumbnail_checkbox = QCheckBox("Show thumbnail preview")
        self.thumbnail_checkbox.setChecked(True)
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

        self.pin_button = QPushButton("Pin selected")
        self.save_button = QPushButton("Save selected output...")
        self.inspector_panel = self._build_inspector()

        self._debounce_timer = QTimer(self)
        self._debounce_timer.setInterval(150)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.timeout.connect(self.run_pipeline)

        self._build_layout()
        self._connect_signals()
        self._refresh_layer_choices()
        self._build_graph_from_pipeline()
        self._select_node(self._selected_node_id)
        self.run_pipeline()
        self._sync_history_actions()

    def closeEvent(self, event):  # noqa: N802
        self._restore_hidden_input_layers()
        super().closeEvent(event)

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
        input_row.addWidget(QLabel("Input"))
        input_row.addWidget(self.layer_combo, 1)
        input_row.addWidget(_toolbar_separator())
        input_row.addWidget(self.global_thumbnail_checkbox)
        input_row.addWidget(self.background_all_checkbox)
        input_row.addWidget(QLabel("Preview"))
        input_row.addWidget(self.preview_mode_combo)
        input_row.addWidget(QLabel("Contrast"))
        input_row.addWidget(self.thumbnail_contrast_combo)
        input_row.addWidget(QLabel("Mono"))
        input_row.addWidget(self.thumbnail_colormap_combo)
        input_row.addWidget(self.follow_dims_checkbox)
        input_row.addWidget(_toolbar_separator())
        input_row.addWidget(QLabel("Zoom"))
        input_row.addWidget(self.graph_zoom_slider)
        input_row.addWidget(self.graph_zoom_reset_button)
        input_row.addWidget(self.graph_zoom_label)
        input_row.addWidget(_toolbar_separator())
        input_row.addWidget(self.refresh_button)
        input_row.addWidget(self.undo_button)
        input_row.addWidget(self.redo_button)
        root.addLayout(input_row)

        workflow_row = QHBoxLayout()
        workflow_row.setContentsMargins(0, 0, 0, 0)
        workflow_row.setSpacing(4)
        workflow_row.addWidget(self.new_workflow_button)
        workflow_row.addWidget(self.save_workflow_button)
        workflow_row.addWidget(self.load_workflow_button)
        workflow_row.addWidget(self.export_button)
        workflow_row.addWidget(self.export_ome_button)
        workflow_row.addStretch(1)
        workflow_row.addWidget(self.pipeline_busy_label)
        workflow_row.addWidget(self.pipeline_busy_bar)
        root.addLayout(workflow_row)

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

    def _build_graph_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumHeight(0)
        panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        panel_controls = QHBoxLayout()
        panel_controls.setContentsMargins(0, 0, 0, 0)
        panel_controls.addWidget(self.left_panel_toggle)
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
        label_volume_layout = QVBoxLayout(self.label_volume_group)
        label_volume_layout.addWidget(self.label_volume_summary)
        label_volume_layout.addWidget(self.label_volume_log_checkbox)
        label_volume_layout.addWidget(self.label_volume_plot)
        layout.addWidget(self.label_volume_group)
        rescale_input_histogram_layout = QVBoxLayout(
            self.rescale_input_histogram_group
        )
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
        rescale_input_histogram_layout.addWidget(
            self.rescale_input_histogram_scope_row
        )
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
        self.refresh_button.clicked.connect(self._refresh_and_run)
        self.undo_action.triggered.connect(self.undo)
        self.redo_action.triggered.connect(self.redo)
        self.save_workflow_button.clicked.connect(self._save_workflow_dialog)
        self.load_workflow_button.clicked.connect(self._load_workflow_dialog)
        self.export_button.clicked.connect(self._export_python_dialog)
        self.export_ome_button.clicked.connect(self._export_ome_dataset_dialog)
        self.layer_combo.currentTextChanged.connect(self.run_pipeline)
        self.global_thumbnail_checkbox.toggled.connect(self._update_thumbnails)
        self.preview_mode_combo.currentTextChanged.connect(self._update_thumbnails)
        self.thumbnail_contrast_combo.currentTextChanged.connect(
            self._update_thumbnails,
        )
        self.thumbnail_colormap_combo.currentTextChanged.connect(self._update_thumbnails)
        self.follow_dims_checkbox.toggled.connect(self._update_thumbnails)
        self.graph_zoom_slider.valueChanged.connect(self._on_graph_zoom_slider_changed)
        self.graph_zoom_reset_button.clicked.connect(self._reset_graph_zoom)
        self.pin_button.clicked.connect(lambda: self.pin_node(self._selected_node_id))
        self.thumbnail_checkbox.toggled.connect(
            self._on_selected_preview_toggled,
        )
        self.histogram_log_checkbox.toggled.connect(self._update_histogram)
        self.histogram_scope_combo.currentTextChanged.connect(self._update_histogram)
        self.rescale_input_histogram_scope_combo.currentTextChanged.connect(
            self._update_histogram
        )
        self.rescale_input_histogram_log_checkbox.toggled.connect(
            self._update_histogram
        )
        self.label_volume_log_checkbox.toggled.connect(
            self._update_label_volume_histogram
        )
        self.auto_contrast_button.clicked.connect(self._apply_auto_contrast)
        self.save_button.clicked.connect(self._save_selected_output_dialog)
        self.left_panel_toggle.clicked.connect(self._toggle_left_panel)
        self.right_panel_toggle.clicked.connect(self._toggle_right_panel)
        self.palette_search.textChanged.connect(self.palette.set_filter_text)

        self.palette.operation_requested.connect(self.add_node_from_palette)
        self.graph_view.node_create_requested.connect(self._add_node_at)
        self.graph_view.node_selected.connect(self._select_node)
        self.graph_view.node_delete_requested.connect(self._delete_node)
        self.graph_view.node_duplicate_requested.connect(self._duplicate_node)
        self.graph_view.node_code_requested.connect(self._inspect_node_code)
        self.graph_view.node_moved.connect(self._on_node_moved)
        self.graph_view.pin_requested.connect(self.pin_node)
        self.graph_view.connection_requested.connect(self._connect_nodes)
        self.graph_view.connection_removed.connect(self._disconnect_nodes)
        self.graph_view.status_message.connect(self.status_label.setText)
        self.graph_view.zoom_changed.connect(self._sync_graph_zoom_controls)

        try:
            self.viewer.layers.events.inserted.connect(
                lambda _=None: self._refresh_layer_choices()
            )
            self.viewer.layers.events.removed.connect(
                lambda _=None: self._refresh_layer_choices()
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
            workflow=deepcopy(serialize_workflow(self.pipeline, positions)),
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
        try:
            workflow = self._deserialize_history_workflow(snapshot.workflow)
            pinned_layer = self._active_pinned_layer()
            if pinned_layer is not None:
                self._remove_layer(pinned_layer)
            self.pipeline.restore_graph(
                workflow["nodes"],
                workflow["connections"],
            )
            valid_node_ids = set(self.pipeline.nodes)
            self._preview_disabled_node_ids = set(
                snapshot.preview_disabled_node_ids
            ) & valid_node_ids
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
                preserve_view=True,
            )
            self._sync_all_input_ports()
            self._sync_all_output_ports()
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
        self.graph_view.select_node(node.id)
        self._invalidate_pipeline_cache()
        self.run_pipeline()
        self._push_undo_if_changed(before)
        self.status_label.setText(f"Added '{node.title}'.")
        return node

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
        self._invalidate_pipeline_cache()
        self.run_pipeline()
        self._push_undo_if_changed(before)
        self.status_label.setText(
            f"Duplicated '{original.title}' as '{clone.title}'."
        )

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
            "font-family: Menlo, Monaco, Consolas, monospace; "
            "font-size: 12px;"
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
        self.status_label.setText(
            f"Opened code for '{self._node_title(node_id)}'."
        )

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
                self._node_code_source_ref(connection)
                for connection in connections
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
        before = self._current_history_snapshot()
        self.pipeline.reset_empty_graph()
        self._preview_disabled_node_ids.clear()
        self._clip_auto_input_ranges.clear()
        self._rescale_auto_input_cutoffs.clear()
        self._rescale_auto_output_ranges.clear()
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
        )
        self._sync_pin_ui()
        self._sync_all_input_ports()
        self._sync_all_output_ports()

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
            saved = save_workflow(path, self.pipeline, positions)
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

    def load_workflow_file(self, path: str | Path) -> Path:
        """Load a workflow file into the widget and recompute the graph."""
        self._finish_parameter_history_group()
        before = self._current_history_snapshot()
        source = Path(path).expanduser()
        workflow = load_workflow(source)
        self.pipeline.restore_graph(workflow["nodes"], workflow["connections"])
        self._preview_disabled_node_ids.clear()
        self._clip_auto_input_ranges.clear()
        self._rescale_auto_input_cutoffs.clear()
        self._rescale_auto_output_ranges.clear()
        self._clear_active_pin(status=False)
        self.graph_view.build_graph(
            self.pipeline.nodes.values(),
            self.pipeline.connections,
            workflow["positions"],
        )
        self._sync_pin_ui()
        self._sync_all_input_ports()
        self._sync_all_output_ports()
        self._invalidate_pipeline_cache()
        self.run_pipeline(force_sync=True)
        self._select_first_available_node()
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
        self._refresh_layer_choices()
        self.run_pipeline()

    def _mark_pipeline_dirty(self, node_id: str) -> None:
        if node_id in self.pipeline.nodes:
            self._pending_dirty_node_ids.add(node_id)

    def _invalidate_pipeline_cache(self) -> None:
        self._pending_dirty_node_ids.clear()
        self._last_pipeline_source_signature = None
        self.pipeline.completed_node_ids.clear()

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
        return ("toolbar", input_name, id(input_data), id(input_metadata))

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
        self._last_pipeline_source_signature = source_signature
        if dirty_node_ids is None:
            self._pending_dirty_node_ids.clear()
        else:
            self._pending_dirty_node_ids.difference_update(dirty_node_ids)

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

    def _refresh_layer_choices(self) -> None:
        current = self.layer_combo.currentText()
        with QSignalBlocker(self.layer_combo):
            self.layer_combo.clear()
            for layer in self.viewer.layers:
                if self._is_vipp_generated_layer(layer):
                    continue
                if hasattr(layer, "data"):
                    self.layer_combo.addItem(layer.name)
            if current:
                index = self.layer_combo.findText(current)
                if index >= 0:
                    self.layer_combo.setCurrentIndex(index)
                    return
            preferred = self._preferred_input_layer_name()
            if preferred:
                index = self.layer_combo.findText(preferred)
                if index >= 0:
                    self.layer_combo.setCurrentIndex(index)

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
                return SourcePayload(None, {}, ""), None
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
            layer_name = self.layer_combo.currentText()
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

    def _connect_nodes(
        self,
        source_id: str,
        target_id: str,
        target_port: int | None = None,
        source_port: int = 0,
    ) -> None:
        self._finish_parameter_history_group()
        before = self._current_history_snapshot()
        result = self.pipeline.connect(
            source_id, target_id, target_port, source_port
        )
        if not result.success:
            self.status_label.setText(result.message)
            return
        for connection in result.removed:
            self.graph_view.remove_connection(
                connection.source_id,
                connection.target_id,
                target_port=connection.target_port,
                notify=False,
            )
        if result.connection is not None:
            self.graph_view.add_connection(
                result.connection.source_id,
                result.connection.target_id,
                result.connection.target_port,
                result.connection.source_port,
            )
        self._invalidate_pipeline_cache()
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
            self._invalidate_pipeline_cache()
            self.run_pipeline()
            self._push_undo_if_changed(before)
            port_text = (
                ""
                if target_port is None
                else f" input {int(target_port) + 1}"
            )
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
        if not self.pipeline.remove_node(node_id):
            return
        self.graph_view.remove_node(node_id)
        self._preview_disabled_node_ids.discard(node_id)
        self._clip_auto_input_ranges.pop(node_id, None)
        self._rescale_auto_input_cutoffs.pop(node_id, None)
        self._rescale_auto_output_ranges.pop(node_id, None)
        if self._active_pinned_node_id == node_id:
            self._clear_active_pin(status=False)
        if self._selected_node_id == node_id:
            self._select_first_available_node()
        self._invalidate_pipeline_cache()
        self.run_pipeline()
        self._push_undo_if_changed(before)
        self.status_label.setText(f"Deleted '{title}'.")

    def _on_node_moved(self, node_id: str, old_pos, _new_pos) -> None:
        if node_id not in self.pipeline.nodes:
            return
        self._finish_parameter_history_group()
        positions = self.graph_view.node_positions()
        positions[node_id] = (float(old_pos.x()), float(old_pos.y()))
        self._push_undo_snapshot(self._current_history_snapshot(positions))
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
        self.metadata_table.setRowCount(0)
        self.table_group.setHidden(True)
        self.table_preview.setRowCount(0)
        self.table_preview.setColumnCount(0)
        self.history_label.setText("No history yet.")
        self.label_volume_group.setHidden(True)
        self.label_volume_plot.set_histogram(None, log_scale=False)
        self.rescale_input_histogram_group.setHidden(True)
        self.rescale_input_histogram_scope_row.setHidden(True)
        self.rescale_input_histogram_plot.set_histogram(None, log_scale=False)
        self.histogram_group.setTitle("Output Histogram")
        self.histogram_scope_row.setHidden(True)
        self.histogram_plot.set_histogram(None, log_scale=False)

    def _select_node(self, node_id: str) -> None:
        if node_id not in self.pipeline.nodes:
            return
        self._selected_node_id = node_id
        node = self.pipeline.nodes[node_id]
        self.selected_title.setText(node.title)
        self._sync_preview_ui()
        self._render_parameters(node_id)
        self._sync_auto_contrast_ui()
        self._sync_pin_ui()
        self._inspect_selected_node()
        self._keep_active_pin_on_top()
        self._update_metadata_panel()
        self._update_histogram()

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
        if node.operation_id == "rescale_axes":
            self._render_rescale_axes_parameters(node_id)
            return
        if node.operation_id == "combine_channels":
            self._render_combine_channels_parameters(node_id)
            return
        if node.operation_id == "assign_channel_colors":
            self._render_assign_channel_colors_parameters(node_id)
            return

        self._sync_clip_intensity_defaults(node_id)
        self._sync_rescale_input_cutoff_defaults(node_id)
        self._sync_rescale_output_range_defaults(node_id)
        rendered = False
        for spec in specs:
            if self._parameter_spec_hidden(node_id, spec):
                continue
            spec = self._effective_parameter_spec(node_id, spec)
            bounds = self._parameter_bounds_for(node_id, spec)
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
            widget = control_class(spec, node.params.get(spec.name), bounds)
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
                "- H is a peak-prominence threshold on the distance map, in pixels/voxels.\n"
                "- 0 uses all local maxima.\n"
                "- Around 0 to 2 is usually the useful range; larger values only matter for larger objects or deeper peak separations."
            )
        if node.operation_id == "marker_controlled_watershed":
            return (
                "Input guide:\n"
                "- Image / distance: elevation image or distance map.\n"
                "- Markers: non-negative integer seed labels.\n"
                "- Mask: foreground constraint region (>0 = inside)."
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
        count_widget.valueChanged.connect(
            self._on_combine_channels_input_count_changed
        )
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
        return {
            "source_mode": node.params.get("source_mode", "napari layer"),
            "layer_name": node.params.get("layer_name", ""),
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
        node = self.pipeline.nodes[self._selected_node_id]
        if self._image_source_value(node) == value:
            return
        self._record_parameter_undo(self._selected_node_id, "image_source")
        previous_path = str(node.params.get("file_path", ""))
        previous_binding = str(node.params.get("binding_mode", "single item"))
        self._apply_image_source_params(self._selected_node_id, value)
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
        return (
            f"{prefix}{inspection.format}; "
            f"{len(inspection.series)} image series discovered."
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
                name.startswith("channel_color_")
                for name in self._parameter_widgets
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
        if node.operation_id == "rescale_axes":
            return self._refresh_rescale_axes_controls(self._selected_node_id)
        if self._sync_rescale_output_range_defaults(self._selected_node_id):
            changed = True
        if self._sync_clip_intensity_defaults(self._selected_node_id):
            changed = True
        if self._sync_rescale_input_cutoff_defaults(self._selected_node_id):
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
            current = widget.value()
            if current != previous:
                node.params[spec.name] = current
                changed = True
        if node.operation_id == "fill_holes":
            self._update_fill_holes_scope_note()
        if node.operation_id == "gaussian_blur_3d":
            changed = self._sync_gaussian_blur_3d_xy_lock(
                self._selected_node_id,
                source_name="sigma_y",
            ) or changed
        if node.operation_id == "rescale_axes":
            changed = self._sync_rescale_axes_xy_lock(
                self._selected_node_id,
                source_name="x_scale",
            ) or changed
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
            and spec.name == "include_2d_boundary_descriptors"
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
        return False

    def _parameter_uses_numeric_entry_only(self, node_id: str, spec) -> bool:
        node = self.pipeline.nodes.get(node_id)
        return (
            node is not None
            and node.operation_id == "set_pixel_size"
            and spec.name in {"x_size", "y_size", "z_size"}
        )

    def _effective_parameter_spec(self, node_id: str, spec):
        node = self.pipeline.nodes.get(node_id)
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
        else:
            choices = ("Auto from axes", "2D YX", "3D ZYX")
        if self._input_spatial_count(node_id) >= 3:
            return choices
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
        labels = [
            f"Channel {index + 1}: {color}"
            for index, color in enumerate(colors)
        ]
        graph_colors = [
            CHANNEL_COLOR_HEX.get(color.lower())
            for color in colors
        ]
        self.graph_view.set_node_input_ports(
            node_id,
            len(colors),
            labels,
            graph_colors,
            [node.input_type or "array"] * len(colors),
        )

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

    def _sync_node_output_ports(self, node_id: str) -> None:
        spec = self.pipeline.operation_spec(
            self.pipeline.nodes[node_id].operation_id
        )
        ports = self.pipeline.output_ports(node_id)
        if not spec.is_multi_output:
            return
        labels = [port.label for port in ports]
        colors = [
            self._output_port_color(index, port)
            for index, port in enumerate(ports)
        ]
        data_types = [port.output_type for port in ports]
        self.graph_view.set_node_output_ports(
            node_id,
            len(ports),
            labels,
            colors,
            data_types,
        )

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

    def _sync_all_input_ports(self) -> None:
        for node_id in self.pipeline.nodes:
            self._sync_node_input_ports(node_id)

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

    def _thumbnail_payload_for_node(self, node_id: str, data):
        state = self.pipeline.output_states.get(node_id)
        node = self.pipeline.nodes.get(node_id)
        if node is None or node.operation_id != "split_channels":
            return data, state
        outputs = self.pipeline.node_outputs.get(node_id) or []
        if not outputs:
            return data, state
        states = self.pipeline.node_output_states.get(node_id) or []
        try:
            index = int(node.params.get("preview_channel", 0))
        except Exception:
            index = 0
        index = int(np.clip(index, 0, len(outputs) - 1))
        selected = outputs[index]
        if selected is None:
            return data, state
        selected_state = states[index] if 0 <= index < len(states) else state
        return selected, selected_state

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

    def _sync_all_rescale_output_range_defaults(self) -> bool:
        changed = False
        for node_id in tuple(self.pipeline.nodes):
            changed = self._sync_rescale_output_range_defaults(node_id) or changed
        return changed

    def _sync_all_rescale_input_cutoff_defaults(self) -> bool:
        changed = False
        for node_id in tuple(self.pipeline.nodes):
            changed = self._sync_rescale_input_cutoff_defaults(node_id) or changed
        return changed

    def _sync_all_clip_intensity_defaults(self) -> bool:
        changed = False
        for node_id in tuple(self.pipeline.nodes):
            changed = self._sync_clip_intensity_defaults(node_id) or changed
        return changed

    def _sync_clip_intensity_defaults(self, node_id: str) -> bool:
        node = self.pipeline.nodes.get(node_id)
        if node is None or node.operation_id != "clip_intensity":
            return False
        values = self._rescale_input_values(node_id)
        if values.size == 0:
            return False

        minimum = float(values.min())
        maximum = float(values.max())
        current = (
            _safe_float(node.params.get("minimum"), 0.0),
            _safe_float(node.params.get("maximum"), 255.0),
        )
        if current[0] > current[1]:
            current = (current[1], current[0])
        previous_auto = self._clip_auto_input_ranges.get(node_id, (0.0, 255.0))
        self._clip_auto_input_ranges[node_id] = (minimum, maximum)
        if not _ranges_close(current, previous_auto):
            return False
        if _ranges_close(current, (minimum, maximum)):
            return False
        node.params["minimum"] = minimum
        node.params["maximum"] = maximum
        return True

    def _sync_rescale_input_cutoff_defaults(self, node_id: str) -> bool:
        node = self.pipeline.nodes.get(node_id)
        if node is None or node.operation_id != "rescale_intensity":
            return False
        values = self._rescale_input_values(node_id)
        if values.size == 0:
            return False

        low_p, high_p = _rescale_percentile_pair(node.params)
        low_value, high_value = _percentile_cutoff_values(values, low_p, high_p)
        current = _rescale_value_pair(node.params)
        previous_auto = self._rescale_auto_input_cutoffs.get(node_id)
        has_values = (
            "in_low_value" in node.params
            and "in_high_value" in node.params
        )
        should_update = (
            not has_values
            or (
                previous_auto is None
                and _ranges_close(current, (0.0, 1.0))
                and not _ranges_close((low_value, high_value), (0.0, 1.0))
            )
            or (
                previous_auto is not None
                and _ranges_close(current, previous_auto)
            )
        )
        self._rescale_auto_input_cutoffs[node_id] = (low_value, high_value)
        if not should_update:
            return False
        if _ranges_close(current, (low_value, high_value)):
            return False
        node.params["in_low_value"] = low_value
        node.params["in_high_value"] = high_value
        return True

    def _sync_rescale_cutoff_parameters(
        self,
        node_id: str,
        driver: str,
    ) -> bool:
        node = self.pipeline.nodes.get(node_id)
        if node is None or node.operation_id != "rescale_intensity":
            return False
        values = self._rescale_input_values(node_id)
        if values.size == 0:
            return False

        changed = False
        if driver in RESCALE_PERCENTILE_PARAMETERS:
            low_p, high_p = _rescale_percentile_pair(node.params)
            low_value, high_value = _percentile_cutoff_values(values, low_p, high_p)
            changed = self._set_node_param_if_changed(
                node_id,
                "in_low_value",
                low_value,
            )
            changed = (
                self._set_node_param_if_changed(
                    node_id,
                    "in_high_value",
                    high_value,
                )
                or changed
            )
            self._rescale_auto_input_cutoffs[node_id] = (low_value, high_value)
            return changed

        low_value, high_value = _rescale_value_pair(node.params)
        if low_value > high_value:
            low_value, high_value = high_value, low_value
            changed = self._set_node_param_if_changed(
                node_id,
                "in_low_value",
                low_value,
            )
            changed = (
                self._set_node_param_if_changed(
                    node_id,
                    "in_high_value",
                    high_value,
                )
                or changed
            )
        low_p, high_p = _percentiles_for_cutoff_values(
            values,
            low_value,
            high_value,
        )
        changed = (
            self._set_node_param_if_changed(
                node_id,
                "in_low_percentile",
                low_p,
            )
            or changed
        )
        changed = (
            self._set_node_param_if_changed(
                node_id,
                "in_high_percentile",
                high_p,
            )
            or changed
        )
        self._rescale_auto_input_cutoffs[node_id] = (low_value, high_value)
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

    def _refresh_rescale_cutoff_widgets(self, node_id: str) -> None:
        for name in RESCALE_CUTOFF_PARAMETERS:
            widget = self._parameter_widgets.get(name)
            if widget is None or name not in self.pipeline.nodes[node_id].params:
                continue
            spec = self._parameter_spec_by_name(node_id, name)
            if spec is None:
                continue
            spec = self._effective_parameter_spec(node_id, spec)
            widget.set_bounds(
                self._parameter_bounds_for(node_id, spec),
                self.pipeline.nodes[node_id].params[name],
                emit=False,
            )

    def _parameter_spec_by_name(self, node_id: str, name: str):
        for spec in self.pipeline.node_parameter_specs(node_id):
            if spec.name == name:
                return spec
        return None

    def _rescale_input_values(self, node_id: str) -> np.ndarray:
        data = self.pipeline.input_data_for_node(node_id)
        if data is None:
            return np.array([], dtype=np.float64)
        return _rescale_reference_values(data)

    def _rescale_input_value_bounds(self, node_id: str, spec) -> ParameterBounds:
        return self._intensity_input_value_bounds(node_id, spec)

    def _intensity_input_value_bounds(self, node_id: str, spec) -> ParameterBounds:
        values = self._rescale_input_values(node_id)
        if values.size == 0:
            return ParameterBounds(
                spec.minimum,
                spec.maximum,
                spec.step,
                spec.decimals,
                expandable=True,
            )

        data = self.pipeline.input_data_for_node(node_id)
        dtype = np.asarray(data).dtype
        if dtype == np.dtype(bool):
            return ParameterBounds(0, 1, 1, 0)
        if dtype == np.dtype(np.uint8):
            return ParameterBounds(0.0, 255.0, 0.1, 2, expandable=True)

        minimum = float(values.min())
        maximum = float(values.max())
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
            maximum = max(
                self._largest_label_volume(arr, spatial_ndim),
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
        rank = (
            1
            if str(connectivity).lower().startswith("face")
            else spatial_ndim
        )
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
                    source_name=(
                        "x_size" if value == "Output size" else "x_scale"
                    ),
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
        if name == "spatial_mode":
            self._update_fill_holes_scope_note()
        if name in {"min_volume", "max_volume", "spatial_mode"}:
            self._update_label_volume_histogram()
        if (
            node.operation_id == "rescale_intensity"
            and name in RESCALE_CUTOFF_PARAMETERS
        ):
            if self._sync_rescale_cutoff_parameters(self._selected_node_id, name):
                self._refresh_rescale_cutoff_widgets(self._selected_node_id)
            self._update_rescale_input_histogram(
                self._selected_node_id,
                self._current_step() if self.follow_dims_checkbox.isChecked() else None,
            )
        if (
            (
                node.operation_id == "clip_intensity"
                and name in CLIP_CUTOFF_PARAMETERS
            )
            or (node.operation_id == "binary_threshold" and name == "threshold")
            or (
                node.operation_id == "hysteresis_threshold"
                and name in {"low_threshold", "high_threshold"}
            )
            or (
                node.operation_id in GLOBAL_THRESHOLD_OPERATIONS
                and name == "threshold_scope"
            )
        ):
            self._update_rescale_input_histogram(
                self._selected_node_id,
                self._current_step() if self.follow_dims_checkbox.isChecked() else None,
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

        data = self.pipeline.input_data_for_node(node_id)
        result = _auto_contrast_scale_offset(
            data,
            self.auto_saturation_control.value(),
        )
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
        saturation = self.auto_saturation_control.value()
        self.status_label.setText(
            f"Auto contrast set '{node.title}' to {saturation:.2f}% saturation "
            f"({lower:.3g} to {upper:.3g})."
        )

    def run_pipeline(self, *, force_sync: bool = False) -> None:
        toolbar_layer = self._selected_input_layer()
        try:
            source_payloads, source_layers = self._source_payloads_for_pipeline()
        except Exception as exc:
            self._set_pipeline_busy(False)
            self.status_label.setText(f"Image source error: {exc}")
            return

        if toolbar_layer is None and not source_payloads:
            self._set_pipeline_busy(False)
            self._invalidate_pipeline_cache()
            self._restore_hidden_input_layers()
            self.pipeline.run(None)
            self._update_thumbnails()
            self._update_metadata_panel()
            self._update_histogram()
            self.status_label.setText("No image layer selected.")
            return

        primary_layer = source_layers[0] if source_layers else (
            None if source_payloads else toolbar_layer
        )
        if primary_layer is not None:
            self._last_input_layer_name = getattr(primary_layer, "name", None)
        self._restore_hidden_input_layers(except_layer=primary_layer)
        input_data = None if source_payloads else getattr(toolbar_layer, "data", None)
        input_metadata = (
            None if source_payloads else getattr(toolbar_layer, "metadata", None)
        )
        input_name = "" if source_payloads else getattr(toolbar_layer, "name", "")
        source_label = self._pipeline_source_label(source_payloads, input_name)
        source_signature = self._pipeline_source_signature(
            input_data,
            input_metadata,
            input_name,
            source_payloads,
        )
        dirty_node_ids = self._dirty_nodes_for_run(source_signature)
        if (not force_sync) and self._should_run_pipeline_in_background(
            dirty_node_ids
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
        )

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
    ) -> None:
        self._set_pipeline_busy(False)
        try:
            self.pipeline.run(
                input_data,
                input_metadata=input_metadata,
                input_name=input_name,
                source_payloads=source_payloads,
                dirty_node_ids=dirty_node_ids,
            )
        except Exception as exc:
            self.status_label.setText(f"Pipeline error: {exc}")
            return
        if self._sync_all_clip_intensity_defaults():
            self.pipeline.run(
                input_data,
                input_metadata=input_metadata,
                input_name=input_name,
                source_payloads=source_payloads,
                dirty_node_ids=dirty_node_ids,
            )
        if self._sync_all_rescale_input_cutoff_defaults():
            self.pipeline.run(
                input_data,
                input_metadata=input_metadata,
                input_name=input_name,
                source_payloads=source_payloads,
                dirty_node_ids=dirty_node_ids,
            )
        if self._sync_all_rescale_output_range_defaults():
            self.pipeline.run(
                input_data,
                input_metadata=input_metadata,
                input_name=input_name,
                source_payloads=source_payloads,
                dirty_node_ids=dirty_node_ids,
            )
        if self._refresh_selected_parameter_controls():
            self.pipeline.run(
                input_data,
                input_metadata=input_metadata,
                input_name=input_name,
                source_payloads=source_payloads,
                dirty_node_ids=dirty_node_ids,
            )
        self._complete_pipeline_run(source_signature, dirty_node_ids)
        self._finish_pipeline_update(primary_layer, source_label)

    def _finish_pipeline_update(self, primary_layer, source_label: str) -> None:
        self._hide_input_layer_for_inspection(primary_layer)
        self._refresh_dynamic_output_ports()
        self._update_thumbnails()
        self._refresh_inspection_layer_if_active()
        self._inspect_selected_node()
        self._refresh_pinned_layer_if_active()
        self._update_metadata_panel()
        self._update_histogram()
        if source_label:
            self.status_label.setText(
                f"Graph updated from '{source_label}'. "
                "Connect ports to build alternate paths."
            )
        else:
            self.status_label.setText("No image source selected.")

    def _pipeline_source_label(
        self,
        source_payloads: dict[str, SourcePayload],
        input_name: str,
    ) -> str:
        source_names = [payload.name for payload in source_payloads.values()]
        return ", ".join(name for name in source_names if name) or input_name

    def _should_run_pipeline_in_background(
        self,
        dirty_node_ids: set[str] | None = None,
    ) -> bool:
        return self._background_processing_node_id(dirty_node_ids) is not None

    def _background_processing_node_id(
        self,
        dirty_node_ids: set[str] | None = None,
    ) -> str | None:
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
        for node_id in self.pipeline.topological_order():
            if affected_node_ids is not None and node_id not in affected_node_ids:
                continue
            node = self.pipeline.nodes.get(node_id)
            if node is not None and node.operation_id in BACKGROUND_PIPELINE_OPERATIONS:
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
    ) -> None:
        processing_node_id = self._background_processing_node_id(dirty_node_ids)
        if self._active_pipeline_run_id is not None:
            self._pipeline_run_pending = True
            self._set_pipeline_busy(
                True,
                self._active_pipeline_node_id or processing_node_id,
                queued=True,
            )
            title = self._node_title(self._active_pipeline_node_id) if (
                self._active_pipeline_node_id in self.pipeline.nodes
            ) else "graph"
            self.status_label.setText(
                f"Processing '{title}' in background; latest edit queued to rerun."
            )
            return

        self._pipeline_run_serial += 1
        run_id = self._pipeline_run_serial
        workflow = deepcopy(serialize_workflow(self.pipeline))
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
                dict(self.pipeline.outputs) if dirty_node_ids is not None else None
            ),
            cached_output_states=(
                dict(self.pipeline.output_states)
                if dirty_node_ids is not None
                else None
            ),
            cached_node_outputs=(
                {
                    node_id: list(outputs)
                    for node_id, outputs in self.pipeline.node_outputs.items()
                }
                if dirty_node_ids is not None
                else None
            ),
            cached_node_output_states=(
                {
                    node_id: list(states)
                    for node_id, states in self.pipeline.node_output_states.items()
                }
                if dirty_node_ids is not None
                else None
            ),
            completed_node_ids=frozenset(self.pipeline.completed_node_ids),
        )
        self._active_pipeline_run_id = run_id
        self._pipeline_run_context[run_id] = (
            primary_layer,
            source_label,
            processing_node_id,
            source_signature,
            dirty_node_ids,
        )
        self._set_pipeline_busy(True, processing_node_id)
        title = self._node_title(processing_node_id) if (
            processing_node_id in self.pipeline.nodes
        ) else "graph"
        self.status_label.setText(f"Processing '{title}' in background...")
        worker = PipelineRunWorker(request)
        worker.signals.node_started.connect(self._on_background_pipeline_node_started)
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

    def _on_background_pipeline_finished(self, result: PipelineRunResult) -> None:
        if result.run_id != self._active_pipeline_run_id:
            return
        self._active_pipeline_run_id = None
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
            self.status_label.setText(
                "Restarting background processing with latest edit."
            )
            QTimer.singleShot(0, self.run_pipeline)
            return
        if result.error:
            self._set_pipeline_busy(False)
            self.status_label.setText(f"Pipeline error: {result.error}")
            return
        if result.pipeline is None:
            self._set_pipeline_busy(False)
            self.status_label.setText("Pipeline error: no result returned.")
            return
        if not self._workflow_matches_current_pipeline(result.workflow):
            if not can_apply_before_pending:
                self.status_label.setText(
                    "Discarded stale background result; rerunning latest graph."
                )
                QTimer.singleShot(0, self.run_pipeline)
                return

        self._apply_pipeline_run_result(
            result.pipeline,
            update_params=not can_apply_before_pending,
        )
        if can_apply_before_pending:
            self._last_pipeline_source_signature = source_signature
        else:
            self._complete_pipeline_run(source_signature, dirty_node_ids)
        if (
            self._sync_all_clip_intensity_defaults()
            or self._sync_all_rescale_input_cutoff_defaults()
            or self._sync_all_rescale_output_range_defaults()
            or self._refresh_selected_parameter_controls()
        ):
            QTimer.singleShot(0, self.run_pipeline)
            return
        self._set_pipeline_busy(False)
        if can_apply_before_pending:
            self._pipeline_run_pending = False
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
        self.pipeline.outputs = dict(result_pipeline.outputs)
        self.pipeline.output_states = dict(result_pipeline.output_states)
        self.pipeline.node_outputs = {
            node_id: list(outputs)
            for node_id, outputs in result_pipeline.node_outputs.items()
        }
        self.pipeline.node_output_states = {
            node_id: list(states)
            for node_id, states in result_pipeline.node_output_states.items()
        }
        self.pipeline.completed_node_ids = set(result_pipeline.completed_node_ids)

    def _set_pipeline_busy(
        self,
        busy: bool,
        node_id: str | None = None,
        *,
        queued: bool = False,
    ) -> None:
        self.pipeline_busy_label.setVisible(busy)
        self.pipeline_busy_bar.setVisible(busy)
        if not busy:
            self.pipeline_busy_label.setText("Processing")
            self.graph_view.clear_node_processing()
            self._active_pipeline_node_id = None
            return
        if node_id is None:
            node_id = self._active_pipeline_node_id
        if node_id not in self.pipeline.nodes:
            node_id = None
        if self._active_pipeline_node_id and self._active_pipeline_node_id != node_id:
            self.graph_view.set_node_processing(self._active_pipeline_node_id, False)
        self._active_pipeline_node_id = node_id
        if node_id is not None:
            title = self._node_title(node_id)
            suffix = " queued" if queued else ""
            self.pipeline_busy_label.setText(f"Processing: {title}{suffix}")
            self.graph_view.set_node_processing(node_id, True, queued=queued)
        else:
            self.pipeline_busy_label.setText("Processing graph")

    def _on_dims_changed(self, _event=None) -> None:
        self._update_thumbnails()
        self._update_metadata_panel()
        self._update_histogram()

    def _update_thumbnails(self) -> None:
        mode = self.preview_mode_combo.currentText()
        current_step = (
            self._current_step() if self.follow_dims_checkbox.isChecked() else None
        )
        current_step_nsteps = (
            self._viewer_nsteps() if self.follow_dims_checkbox.isChecked() else None
        )
        contrast_mode = self.thumbnail_contrast_combo.currentText()
        previews_visible_globally = (
            self.global_thumbnail_checkbox.isChecked() and mode.lower() != "off"
        )
        for node_id, data in self.pipeline.outputs.items():
            node_output_type = self._node_output_type(node_id)
            self.graph_view.set_node_metadata(
                node_id,
                format_compact_metadata(self.pipeline.output_states.get(node_id)),
            )
            self.graph_view.set_node_output_type(node_id, node_output_type)
            self.graph_view.set_node_can_pin(node_id, self._node_can_pin(node_id))
            preview_enabled = (
                previews_visible_globally and self._node_preview_enabled(node_id)
            )
            self.graph_view.set_node_preview_enabled(node_id, preview_enabled)
            if not preview_enabled:
                self.graph_view.set_thumbnail(node_id, None)
                continue
            preview_data, preview_state = self._thumbnail_payload_for_node(
                node_id,
                data,
            )
            preview = make_preview(
                preview_data,
                mode=mode,
                current_step=current_step,
                current_step_nsteps=current_step_nsteps,
                state=preview_state,
                channel_colors=self._node_preview_channel_colors(node_id),
                contrast_mode=contrast_mode,
            )
            thumbnail = normalize_thumbnail_with_colormap(
                preview,
                colormap=self.thumbnail_colormap_combo.currentText(),
                contrast_mode=contrast_mode,
                data_kind=node_output_type,
            )
            self.graph_view.set_thumbnail(node_id, thumbnail)
        if (
            self._active_pinned_node_id is not None
            and not self._node_can_pin(self._active_pinned_node_id)
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
        state = self.pipeline.output_states.get(self._selected_node_id)
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
        if (
            state is None
            or not hasattr(state, "axes")
            or not self.follow_dims_checkbox.isChecked()
        ):
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
        node = self.pipeline.nodes.get(self._selected_node_id)
        if node is None:
            self.rescale_input_histogram_group.setHidden(True)
            self.rescale_input_histogram_scope_row.setHidden(True)
            self.rescale_input_histogram_plot.set_histogram(None, log_scale=False)
            self.histogram_plot.set_histogram(None, log_scale=False)
            return
        data = self.pipeline.outputs.get(self._selected_node_id)
        if is_table_data(data):
            self.rescale_input_histogram_group.setHidden(True)
            self.rescale_input_histogram_scope_row.setHidden(True)
            self.rescale_input_histogram_plot.set_histogram(None, log_scale=False)
            self.histogram_group.setHidden(True)
            self.histogram_plot.set_histogram(None, log_scale=False)
            return
        current_step = (
            self._current_step() if self.follow_dims_checkbox.isChecked() else None
        )
        current_step_nsteps = (
            self._viewer_nsteps() if self.follow_dims_checkbox.isChecked() else None
        )
        self._update_rescale_input_histogram(node.id, current_step)
        self.histogram_group.setHidden(False)
        self.histogram_group.setTitle("Output Histogram")
        state = self.pipeline.output_states.get(self._selected_node_id)
        scope_available = _histogram_has_stack_scope(data, state)
        self.histogram_scope_row.setHidden(not scope_available)
        scope = self.histogram_scope_combo.currentText() if scope_available else "Slice"
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

    def _update_rescale_input_histogram(
        self,
        node_id: str,
        current_step,
    ) -> None:
        current_step_nsteps = (
            self._viewer_nsteps()
            if self.follow_dims_checkbox.isChecked() and current_step is not None
            else None
        )
        node = self.pipeline.nodes.get(node_id)
        visible = node is not None and node.operation_id in INPUT_HISTOGRAM_OPERATIONS
        self.rescale_input_histogram_group.setHidden(not visible)
        if not visible:
            self.rescale_input_histogram_group.setTitle("Input Histogram")
            self.rescale_input_histogram_scope_row.setHidden(True)
            self.rescale_input_histogram_plot.set_histogram(None, log_scale=False)
            return

        data = self.pipeline.input_data_for_node(node_id)
        if data is None or is_table_data(data):
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
                self.rescale_input_histogram_group.setTitle(
                    f"Input Histogram ({scope_label})"
                )
            else:
                scope = "Slice histogram"
                self.rescale_input_histogram_group.setTitle("Input Histogram")
            self.rescale_input_histogram_scope_row.setHidden(True)
        else:
            self.rescale_input_histogram_group.setTitle("Input Histogram")
            self.rescale_input_histogram_scope_row.setHidden(not scope_available)
            scope = (
                self.rescale_input_histogram_scope_combo.currentText()
                if scope_available
                else "Slice"
            )
        counts, x_range, colors = _histogram_summary(
            data,
            state=state,
            scope=scope,
            current_step=current_step,
            current_step_nsteps=current_step_nsteps,
        )
        markers = _input_histogram_markers(
            node.operation_id,
            data,
            state=state,
            scope=scope,
            current_step=current_step,
            current_step_nsteps=current_step_nsteps,
            params=node.params,
        )
        self.rescale_input_histogram_plot.set_histogram(
            counts,
            log_scale=self.rescale_input_histogram_log_checkbox.isChecked(),
            x_range=x_range,
            colors=colors,
            markers=markers,
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
        visible = (
            node is not None
            and node.operation_id == "filter_labels_by_volume"
        )
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
        volumes = self._label_volumes(arr, spatial_ndim)
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
        if is_table_data(self.pipeline.outputs.get(node_id)):
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
        data = self.pipeline.outputs.get(node_id)
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
        data = self.pipeline.outputs.get(node_id)
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
                    image_state=self.pipeline.output_states.get(node_id),
                )
        except Exception as exc:
            self.status_label.setText(f"Save failed: {exc}")
            return None
        self.status_label.setText(
            f"Saved '{self._node_title(node_id)}' to {output_path}."
        )
        return output_path

    def inspect_node(self, node_id: str) -> None:
        data = self.pipeline.outputs.get(node_id)
        if data is None:
            self.status_label.setText("That node has no output to inspect yet.")
            return
        if is_table_data(data):
            self.status_label.setText(
                f"'{self._node_title(node_id)}' is shown in the table inspector."
            )
            return
        title = self._node_title(node_id)
        self._set_or_add_generated_layer(
            self._inspect_layer_name,
            data,
            metadata={
                "napari_vipp_kind": "inspect",
                "node_id": node_id,
                "vipp_image_state": self._node_state_dict(node_id),
            },
            role="inspect",
        )
        self._keep_active_pin_on_top()
        self.status_label.setText(f"Inspecting '{title}' in napari.")

    def _inspect_selected_node(self) -> None:
        data = self.pipeline.outputs.get(self._selected_node_id)
        if data is not None and not is_table_data(data):
            self.inspect_node(self._selected_node_id)

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
        data = self.pipeline.outputs.get(node_id)
        if data is None:
            self.status_label.setText("That node has no output to pin yet.")
            return
        self._set_active_pin_layer(node_id, data)
        self._push_undo_if_changed(before)
        self.status_label.setText(f"Pinned '{self._node_title(node_id)}'.")

    def _set_active_pin_layer(self, node_id: str, data) -> None:
        title = self._node_title(node_id)
        display_data = self._display_data(data)
        data_kind = self._data_kind(data, node_id)
        metadata = {
            "napari_vipp_kind": "pinned",
            "node_id": node_id,
            "data_kind": data_kind,
            "display_kind": self._display_kind(data_kind, "pinned"),
            "display_ndim": np.asarray(display_data).ndim,
            "display_shape": tuple(np.asarray(display_data).shape),
            "display_rgb": self._display_rgb(data, node_id),
            "vipp_image_state": self._node_state_dict(node_id),
        }
        layer = self._active_pinned_layer()
        if layer is not None and self._generated_layer_needs_replacement(
            layer,
            metadata,
        ):
            self._remove_layer(layer)
            layer = None

        if layer is None:
            layer = self._add_image_or_labels(
                self._pinned_layer_name(title),
                data,
                metadata,
            )
        else:
            layer.data = display_data
            layer.metadata.update(metadata)
            layer.name = self._pinned_layer_name(title)
            layer.visible = True

        self._move_layer_to_top(layer)
        self._active_pinned_node_id = node_id
        self._sync_pin_ui()

    def _clear_active_pin(self, status: bool) -> None:
        title = (
            self._node_title(self._active_pinned_node_id)
            if self._active_pinned_node_id in self.pipeline.nodes
            else None
        )
        layer = self._active_pinned_layer()
        if layer is not None:
            self._remove_layer(layer)
        self._active_pinned_node_id = None
        self._sync_pin_ui()
        if status and title is not None:
            self.status_label.setText(f"Unpinned '{title}'.")

    def _refresh_inspection_layer_if_active(self) -> None:
        layer = self._layer_by_name(self._inspect_layer_name)
        if layer is None:
            return
        node_id = getattr(layer, "metadata", {}).get("node_id")
        if (
            node_id in self.pipeline.outputs
            and self.pipeline.outputs[node_id] is not None
        ):
            if is_table_data(self.pipeline.outputs[node_id]):
                return
            self._set_or_add_generated_layer(
                self._inspect_layer_name,
                self.pipeline.outputs[node_id],
                metadata={
                    "napari_vipp_kind": "inspect",
                    "node_id": node_id,
                    "vipp_image_state": self._node_state_dict(node_id),
                },
                role="inspect",
            )
            self._keep_active_pin_on_top()

    def _refresh_pinned_layer_if_active(self) -> None:
        if self._active_pinned_node_id is None:
            return
        data = self.pipeline.outputs.get(self._active_pinned_node_id)
        if data is None:
            self._clear_active_pin(status=False)
            return
        self._set_active_pin_layer(self._active_pinned_node_id, data)

    def _selected_input_layer(self):
        name = self.layer_combo.currentText()
        if not name:
            return None
        return self._layer_by_name(name)

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
        display_data = self._display_data(data)
        data_kind = self._data_kind(data, metadata.get("node_id"))
        metadata = {
            **metadata,
            "data_kind": data_kind,
            "display_kind": self._display_kind(data_kind, role),
            "display_ndim": np.asarray(display_data).ndim,
            "display_shape": tuple(np.asarray(display_data).shape),
            "display_rgb": self._display_rgb(data, metadata.get("node_id")),
        }
        layer = self._layer_by_name(name)
        if layer is None:
            self._add_image_or_labels(name, data, metadata=metadata)
            self._restore_viewer_step(saved_step, saved_nsteps)
            return
        if self._generated_layer_needs_replacement(layer, metadata):
            self._remove_layer(layer)
            self._add_image_or_labels(name, data, metadata=metadata)
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
            try:
                dims.set_current_step(axis, int(value))
            except Exception:
                pass

    def _add_image_or_labels(self, name: str, data, metadata: dict):
        display_data = self._display_data(data)
        if metadata["display_kind"] == "labels" and hasattr(self.viewer, "add_labels"):
            return self.viewer.add_labels(display_data, name=name, metadata=metadata)
        kwargs = {"name": name, "metadata": metadata}
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
            limits = self._signed_image_contrast_limits(data)
            if limits is not None:
                kwargs["contrast_limits"] = limits
        return self.viewer.add_image(display_data, **kwargs)

    def _generated_layer_needs_replacement(self, layer, metadata: dict) -> bool:
        return (
            layer.metadata.get("display_kind") != metadata["display_kind"]
            or layer.metadata.get("data_kind") != metadata["data_kind"]
            or layer.metadata.get("display_ndim") != metadata["display_ndim"]
            or tuple(layer.metadata.get("display_shape", ()))
            != tuple(metadata["display_shape"])
            or bool(layer.metadata.get("display_rgb"))
            != bool(metadata.get("display_rgb"))
        )

    def _configure_generated_layer(self, layer, data, metadata: dict) -> None:
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
            limits = self._signed_image_contrast_limits(data)
            if limits is not None:
                try:
                    layer.contrast_limits = limits
                except Exception:
                    pass

    def _signed_image_contrast_limits(self, data) -> tuple[float, float] | None:
        """Anchor the display black point at zero for signed images.

        Bioimage intensities are non-negative, but arithmetic nodes such as
        Subtract can yield negative float values. Letting napari auto-scale from
        the negative minimum renders the zero background grey. Anchoring the
        black point at zero keeps the background black and matches the thumbnail.
        Non-negative images return ``None`` so napari's default contrast is kept.
        """
        arr = np.asarray(data)
        if arr.dtype == bool or arr.size == 0:
            return None
        finite = arr[np.isfinite(arr)]
        if finite.size == 0 or float(finite.min()) >= 0.0:
            return None
        non_negative = finite[finite >= 0.0]
        if non_negative.size:
            high = float(np.percentile(non_negative, 99))
        else:
            high = 0.0
        if high <= 0.0:
            high = float(finite.max())
        if high <= 0.0:
            return None
        return (0.0, high)

    def _display_kind(self, data_kind: str, role: str) -> str:
        if data_kind == "labels":
            return "labels"
        if role == "pinned" and data_kind == "mask":
            return "labels"
        return "image"

    def _data_kind(self, data, node_id: str | None = None) -> str:
        if is_table_data(data):
            return "table"
        if node_id is not None:
            ports = self.pipeline.output_ports(node_id)
            if ports and ports[0].output_type == "table":
                return "table"
            if ports and ports[0].output_type == "labels":
                return "labels"
        return "mask" if np.asarray(data).dtype == bool else "image"

    def _display_rgb(self, data, node_id: str | None = None) -> bool:
        if data is None or is_table_data(data):
            return False
        arr = np.asarray(data)
        if arr.ndim < 3 or arr.shape[-1] not in (3, 4):
            return False
        state = self.pipeline.output_states.get(node_id) if node_id else None
        kind = str(getattr(state, "kind", "")).lower()
        return kind in {"rgb image", "rgba image"}

    def _display_data(self, data):
        if is_table_data(data):
            raise ValueError(
                "Table outputs cannot be displayed as napari image layers."
            )
        arr = np.asarray(data)
        if arr.dtype == bool:
            arr = arr.astype(np.uint8)
        if arr.ndim == 0:
            return arr.reshape(1, 1)
        if arr.ndim == 1:
            return arr.reshape(1, arr.shape[0])
        return arr

    def _active_pinned_layer(self):
        for layer in self.viewer.layers:
            try:
                if layer.metadata.get("napari_vipp_kind") == "pinned":
                    return layer
            except Exception:
                continue
        return None

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
        layer = self._active_pinned_layer()
        if layer is not None:
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

    def _sync_auto_contrast_ui(self) -> None:
        node = self.pipeline.nodes.get(self._selected_node_id)
        self.auto_contrast_group.setVisible(
            node is not None and node.operation_id == "linear_scale_offset"
        )

    def _node_preview_enabled(self, node_id: str) -> bool:
        if self._node_output_type(node_id) == "table":
            return False
        return node_id not in self._preview_disabled_node_ids

    def _node_can_pin(self, node_id: str) -> bool:
        node = self.pipeline.nodes.get(node_id)
        if node is None:
            return False
        data = self.pipeline.outputs.get(node_id)
        if data is not None:
            return not is_table_data(data) and self._data_kind(data, node_id) != "table"
        return node.output_type != "table"

    def _node_output_type(self, node_id: str) -> str:
        node = self.pipeline.nodes.get(node_id)
        ports = self.pipeline.output_ports(node_id)
        if (
            node is not None
            and ports
            and (
                self.pipeline.operation_spec(node.operation_id).preserves_input_type
                or ports[0].output_type == "labels"
            )
        ):
            return ports[0].output_type
        data = self.pipeline.outputs.get(node_id)
        if data is not None:
            return self._data_kind(data, node_id)
        return node.output_type if node is not None else "image"

    def _remove_layer(self, layer) -> None:
        try:
            self.viewer.layers.remove(layer)
        except Exception:
            pass

    def _pinned_layer_name(self, title: str) -> str:
        return f"VIPP Pinned: {title}"

    def _current_step(self):
        raw_step = self._raw_current_step()
        if raw_step is None:
            return None
        return self._canonical_current_step(raw_step)

    def _raw_current_step(self):
        try:
            return tuple(self.viewer.dims.current_step)
        except Exception:
            return None

    def _canonical_current_step(self, current_step: tuple) -> tuple:
        layer = self._layer_by_name(self._inspect_layer_name)
        metadata = getattr(layer, "metadata", {}) if layer is not None else {}
        if not isinstance(metadata, dict):
            return current_step
        node_id = metadata.get("node_id")
        state = self.pipeline.output_states.get(node_id)
        axes = tuple(getattr(state, "axes", ()))
        if not axes:
            return current_step
        display_axis_indices = [
            index
            for index, axis in enumerate(axes)
            if not _state_axis_hidden_from_napari_dims(axis, metadata)
        ]
        if not display_axis_indices:
            return current_step
        offset = max(len(current_step) - len(display_axis_indices), 0)
        source_axes = [
            int(axis.source_axis)
            for axis in axes
            if getattr(axis, "source_axis", None) is not None
        ]
        if not source_axes:
            return current_step
        canonical = [0] * (max(source_axes) + 1)
        for display_position, state_axis_index in enumerate(display_axis_indices):
            current_index = offset + display_position
            if current_index < 0 or current_index >= len(current_step):
                continue
            axis = axes[state_axis_index]
            source_axis = getattr(axis, "source_axis", None)
            if source_axis is None:
                continue
            source_axis = int(source_axis)
            if 0 <= source_axis < len(canonical):
                canonical[source_axis] = int(current_step[current_index])
        return tuple(canonical)

    def _node_title(self, node_id: str) -> str:
        return self.pipeline.nodes[node_id].title

    def _node_state_dict(self, node_id: str) -> dict | None:
        state = self.pipeline.output_states.get(node_id)
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


def _auto_contrast_scale_offset(
    data,
    saturation_percent: float,
) -> tuple[float, float, float, float] | None:
    if data is None:
        return None

    values = _finite_values(np.asarray(data)).ravel()
    if values.size == 0:
        return None
    if values.size > 1_000_000:
        stride = int(np.ceil(values.size / 1_000_000))
        values = values[::stride]

    saturation = min(max(float(saturation_percent), 0.0), 100.0)
    tail_percent = saturation / 2.0
    lower, upper = np.percentile(
        values.astype(np.float64, copy=False),
        [tail_percent, 100.0 - tail_percent],
    )
    lower = float(lower)
    upper = float(upper)
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        return None

    alpha = 255.0 / (upper - lower)
    beta = -lower * alpha
    if not np.isfinite(alpha) or not np.isfinite(beta):
        return None
    return float(alpha), float(beta), lower, upper


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


def _input_histogram_markers(
    operation_id: str,
    data,
    *,
    state=None,
    scope: str = "Slice",
    current_step=None,
    current_step_nsteps=None,
    params: dict | None = None,
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
        low = _safe_float((params or {}).get("minimum"), 0.0)
        high = _safe_float((params or {}).get("maximum"), 255.0)
        if low > high:
            low, high = high, low
        markers = [("min", float(low), QColor("#f59e0b"))]
        if not np.isclose(low, high):
            markers.append(("max", float(high), QColor("#38bdf8")))
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
        value = automatic_threshold_value(source[0], operation_id)
        if value is None or not np.isfinite(value):
            return []
        return [("threshold", float(value), QColor("#f59e0b"))]
    return []


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
    if params is not None and {
        "in_low_value",
        "in_high_value",
    } <= set(params):
        low, high = _rescale_value_pair(params)
        markers = [("low", float(low), QColor("#f59e0b"))]
        if not np.isclose(low, high):
            markers.append(("high", float(high), QColor("#22c55e")))
        return markers

    source = _histogram_source(
        data,
        state=state,
        scope=scope,
        current_step=current_step,
        current_step_nsteps=current_step_nsteps,
    )
    if source is None:
        return []
    values = _sample_histogram_values(source[0])
    if values.size == 0:
        return []
    low_p, high_p = _rescale_percentile_pair(params or {})
    low, high = _percentile_cutoff_values(values, low_p, high_p)
    markers = [("low", float(low), QColor("#f59e0b"))]
    if not np.isclose(low, high):
        markers.append(("high", float(high), QColor("#22c55e")))
    return markers


def _rescale_reference_values(data) -> np.ndarray:
    arr = np.asarray(data)
    if arr.size == 0:
        return np.array([], dtype=np.float64)
    values = arr.ravel()
    try:
        values = values[np.isfinite(values)]
    except TypeError:
        return np.array([], dtype=np.float64)
    if values.size > 500_000:
        stride = int(np.ceil(values.size / 500_000))
        values = values[::stride]
    return values.astype(np.float64, copy=False)


def _rescale_percentile_pair(params: dict) -> tuple[float, float]:
    low = _safe_float(params.get("in_low_percentile"), 0.0)
    high = _safe_float(params.get("in_high_percentile"), 100.0)
    low = float(np.clip(low, 0.0, 100.0))
    high = float(np.clip(high, 0.0, 100.0))
    if low > high:
        low, high = high, low
    return low, high


def _rescale_value_pair(params: dict) -> tuple[float, float]:
    low = _safe_float(params.get("in_low_value"), 0.0)
    high = _safe_float(params.get("in_high_value"), 1.0)
    if low > high:
        low, high = high, low
    return low, high


def _percentile_cutoff_values(
    values: np.ndarray,
    low_percentile: float,
    high_percentile: float,
) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(
        values.astype(np.float64, copy=False),
        [low_percentile, high_percentile],
    )
    return float(low), float(high)


def _percentiles_for_cutoff_values(
    values: np.ndarray,
    low_value: float,
    high_value: float,
) -> tuple[float, float]:
    if values.size <= 1:
        return 0.0, 100.0
    sorted_values = np.sort(values.astype(np.float64, copy=False))
    low = _percentile_for_cutoff_value(sorted_values, low_value)
    high = _percentile_for_cutoff_value(sorted_values, high_value)
    if low > high:
        low, high = high, low
    return low, high


def _percentile_for_cutoff_value(
    sorted_values: np.ndarray,
    value: float,
) -> float:
    if sorted_values.size <= 1:
        return 0.0
    if value <= sorted_values[0]:
        return 0.0
    if value >= sorted_values[-1]:
        return 100.0
    index = int(np.searchsorted(sorted_values, float(value), side="left"))
    return float(np.clip((index / (sorted_values.size - 1)) * 100.0, 0.0, 100.0))


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
        result = np.take(result, index, axis=local_axis)
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
    if (
        all(len(name) == 1 for name in names)
        and len(set(normalized)) == len(normalized)
    ):
        return "".join(name.upper() for name in names)
    if (
        all(name and not any(char in name for char in ",; ") for name in names)
        and len(set(normalized)) == len(normalized)
    ):
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
) -> tuple[np.ndarray | None, tuple[float, float] | None]:
    if arr.dtype == bool:
        return np.bincount(arr.ravel().astype(np.uint8), minlength=2), (0.0, 1.0)

    values = _sample_histogram_values(arr)
    if values.size == 0:
        return None, None

    if np.issubdtype(values.dtype, np.integer):
        finite_min = int(values.min())
        finite_max = int(values.max())
        if finite_min == finite_max:
            return np.array([values.size], dtype=np.int64), (
                float(finite_min),
                float(finite_max),
            )
        if 0 <= finite_min and finite_max <= 255:
            counts, _edges = np.histogram(values, bins=256, range=(0, 255))
            x_range = (0.0, 255.0)
        else:
            counts, _edges = np.histogram(values, bins=128)
            x_range = (float(finite_min), float(finite_max))
    else:
        finite_min = float(values.min())
        finite_max = float(values.max())
        if finite_min == finite_max:
            return np.array([values.size], dtype=np.int64), (
                finite_min,
                finite_max,
            )
        counts, _edges = np.histogram(values, bins=128)
        x_range = (finite_min, finite_max)
    return counts, x_range


def _multichannel_histogram(
    arr: np.ndarray,
    channel_axis: int,
) -> tuple[np.ndarray | None, tuple[float, float] | None]:
    channel_axis = int(np.clip(channel_axis, 0, arr.ndim - 1))
    values_by_channel = [
        _sample_histogram_values(np.take(arr, channel, axis=channel_axis))
        for channel in range(arr.shape[channel_axis])
    ]
    values_by_channel = [values for values in values_by_channel if values.size]
    if not values_by_channel:
        return None, None

    if all(values.dtype == bool for values in values_by_channel):
        counts = [
            np.bincount(values.astype(np.uint8), minlength=2)
            for values in values_by_channel
        ]
        return np.vstack(counts), (0.0, 1.0)

    if all(np.issubdtype(values.dtype, np.integer) for values in values_by_channel):
        finite_min = min(int(values.min()) for values in values_by_channel)
        finite_max = max(int(values.max()) for values in values_by_channel)
        if finite_min == finite_max:
            counts = np.array(
                [[values.size] for values in values_by_channel],
                dtype=np.int64,
            )
            return counts, (float(finite_min), float(finite_max))
        if 0 <= finite_min and finite_max <= 255:
            bins = 256
            hist_range = (0.0, 255.0)
        else:
            bins = 128
            hist_range = (float(finite_min), float(finite_max))
    else:
        finite_min = min(float(values.min()) for values in values_by_channel)
        finite_max = max(float(values.max()) for values in values_by_channel)
        if finite_min == finite_max:
            counts = np.array(
                [[values.size] for values in values_by_channel],
                dtype=np.int64,
            )
            return counts, (finite_min, finite_max)
        bins = 128
        hist_range = (finite_min, finite_max)

    counts = [
        np.histogram(values, bins=bins, range=hist_range)[0]
        for values in values_by_channel
    ]
    return np.vstack(counts), hist_range


def _sample_histogram_values(arr: np.ndarray) -> np.ndarray:
    values = np.asarray(arr).ravel()
    if values.size == 0:
        return values
    values = values[np.isfinite(values)]
    if values.size > 500_000:
        stride = int(np.ceil(values.size / 500_000))
        values = values[::stride]
    return values


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


def _format_histogram_label(value: float) -> str:
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
