"""napari dock widget for the VIPP workflow prototype."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtCore import QMimeData, QRect, QSignalBlocker, QSize, Qt, QTimer, Signal
from qtpy.QtGui import QBrush, QColor, QPainter, QPen
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
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

from napari_vipp._graph import OPERATION_MIME, PipelineGraphView
from napari_vipp._theme import category_color, category_tint
from napari_vipp.core.metadata import (
    MetadataRow,
    format_compact_metadata,
    metadata_history_items,
    metadata_table_rows,
)
from napari_vipp.core.pipeline import (
    OperationSpec,
    ParameterSpec,
    PrototypePipeline,
    grouped_palette_specs,
)
from napari_vipp.core.preview import make_preview, normalize_thumbnail

if TYPE_CHECKING:
    import napari


@dataclass(frozen=True)
class ParameterBounds:
    minimum: float | int
    maximum: float | int
    step: float | int
    decimals: int
    expandable: bool = False


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

    def __init__(self, groups: dict[str, list[OperationSpec]], parent=None):
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
        self._operation_items: list[QTreeWidgetItem] = []
        for category, specs in groups.items():
            category_item = QTreeWidgetItem([category])
            category_item.setFlags(category_item.flags() & ~Qt.ItemIsDragEnabled)
            self._style_category_item(category_item, category)
            self.addTopLevelItem(category_item)
            self._category_items.append(category_item)
            for spec in specs:
                item = QTreeWidgetItem([spec.title])
                item.setData(0, Qt.UserRole, spec.id)
                item.setData(
                    0,
                    Qt.UserRole + 1,
                    f"{category} {spec.title} {spec.id}",
                )
                self._style_operation_item(item, category)
                category_item.addChild(item)
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
            category_visible = False
            for index in range(category_item.childCount()):
                item = category_item.child(index)
                haystack = _normalize_search_text(item.data(0, Qt.UserRole + 1))
                visible = not query or _fuzzy_match(query, haystack)
                item.setHidden(not visible)
                category_visible = category_visible or visible
                visible_count += int(visible)
            category_item.setHidden(not category_visible)
            if category_visible:
                category_item.setExpanded(True)
        self._no_results_item.setHidden(not query or visible_count > 0)


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
            self.value_box = QDoubleSpinBox()
            self.value_box.setDecimals(bounds.decimals)
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
        )
        self._bounds = bounds
        self._entry_minimum = entry_minimum
        self._entry_maximum = entry_maximum
        self._scale = self._scale_for(bounds)

        with QSignalBlocker(self.slider), QSignalBlocker(self.value_box):
            if self._is_integer:
                self.value_box.setRange(int(entry_minimum), int(entry_maximum))
                self.value_box.setSingleStep(max(int(bounds.step), 1))
                self.slider.setRange(int(bounds.minimum), int(bounds.maximum))
                self.slider.setSingleStep(max(int(bounds.step), 1))
                self.slider.setValue(int(current))
                self.value_box.setValue(int(current))
            else:
                self.value_box.setDecimals(bounds.decimals)
                self.value_box.setRange(float(entry_minimum), float(entry_maximum))
                self.value_box.setSingleStep(float(bounds.step))
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
        mapped = int(value) if self._is_integer else value / self._scale
        with QSignalBlocker(self.value_box):
            self.value_box.setValue(mapped)
        self.valueChanged.emit(self.value())

    def _on_box_changed(self, value) -> None:
        if self._bounds.expandable and not self._value_in_slider_bounds(value):
            self.set_bounds(self._bounds, value, emit=False)
            self.valueChanged.emit(self.value())
            return
        with QSignalBlocker(self.slider):
            slider_value = int(value) if self._is_integer else self._to_slider(value)
            self.slider.setValue(slider_value)
        self.valueChanged.emit(self.value())

    def _scale_for(self, bounds: ParameterBounds) -> int:
        if self._is_integer:
            return 1
        if bounds.decimals > 0:
            return 10**bounds.decimals
        step = float(bounds.step)
        return max(int(round(1 / step)), 1) if step > 0 else 100

    def _to_slider(self, value) -> int:
        return int(round(float(value) * self._scale))

    def _clamped_value(self, value, minimum, maximum):
        if value is None:
            value = minimum
        return min(max(value, minimum), maximum)

    def _entry_bounds_for(
        self,
        bounds: ParameterBounds,
    ) -> tuple[float | int, float | int]:
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
            )
        return ParameterBounds(
            minimum,
            maximum,
            bounds.step,
            bounds.decimals,
            bounds.expandable,
        )

    def _value_in_slider_bounds(self, value) -> bool:
        if self._is_integer:
            return self.slider.minimum() <= int(value) <= self.slider.maximum()
        slider_value = self._to_slider(value)
        return self.slider.minimum() <= slider_value <= self.slider.maximum()


class ChoiceControl(QWidget):
    """Dropdown control for categorical node parameters."""

    valueChanged = Signal(object)

    def __init__(self, spec, value, bounds: ParameterBounds, parent=None):
        super().__init__(parent)
        self.spec = spec
        self._bounds = bounds
        self.combo = QComboBox()
        self.combo.addItems(list(spec.choices))

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.combo, 1)

        self.set_bounds(bounds, value, emit=False)
        self.combo.currentTextChanged.connect(self.valueChanged.emit)

    def value(self):
        return self.combo.currentText()

    def set_bounds(
        self,
        bounds: ParameterBounds,
        value=None,
        emit: bool = False,
    ) -> None:
        self._bounds = bounds
        current = self.spec.default if value is None else value
        current = str(current)
        if current and self.combo.findText(current) < 0:
            self.combo.addItem(current)
        with QSignalBlocker(self.combo):
            index = self.combo.findText(current)
            if index < 0:
                index = 0
            self.combo.setCurrentIndex(index)
        if emit:
            self.valueChanged.emit(self.value())


class HistogramPlot(QWidget):
    """Compact histogram display for the selected node output."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._counts = np.array([], dtype=np.float32)
        self._log_scale = False
        self._x_min_label = ""
        self._x_max_label = ""
        self.setMinimumHeight(120)

    def set_histogram(
        self,
        counts: np.ndarray | None,
        log_scale: bool,
        x_range: tuple[float, float] | None = None,
    ) -> None:
        self._counts = (
            np.asarray(counts, dtype=np.float32)
            if counts is not None
            else np.array([], dtype=np.float32)
        )
        self._log_scale = log_scale
        if x_range is None or self._counts.size == 0:
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

        if self._counts.size == 0:
            painter.setPen(QColor("#9ca3af"))
            painter.drawText(plot_frame, Qt.AlignCenter, "No data")
            painter.end()
            return

        values = self._counts
        if self._log_scale:
            values = np.log10(values + 1.0)
        maximum = float(values.max())
        if maximum <= 0:
            painter.end()
            return

        width = max(plot_rect.width(), 1)
        height = max(plot_rect.height(), 1)
        step = max(int(np.ceil(values.size / width)), 1)
        reduced = np.array(
            [values[i : i + step].max() for i in range(0, values.size, step)],
            dtype=np.float32,
        )

        painter.setPen(QPen(QColor("#60a5fa"), 1.4))
        last_x = plot_rect.left()
        last_y = plot_rect.bottom()
        for index, value in enumerate(reduced):
            x = plot_rect.left() + int(index * width / max(reduced.size - 1, 1))
            y = plot_rect.bottom() - int((float(value) / maximum) * height)
            painter.drawLine(last_x, last_y, x, y)
            last_x, last_y = x, y
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
        self._dock_chrome_configured = False
        self._initial_dock_size_applied = False
        self.setMinimumSize(0, 0)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)

        self.layer_combo = QComboBox()
        self.layer_combo.setMinimumWidth(220)
        self.preview_mode_combo = QComboBox()
        self.preview_mode_combo.addItems(["Slice", "MIP", "Off"])
        self.follow_dims_checkbox = QCheckBox("Follow napari dims")
        self.follow_dims_checkbox.setChecked(True)

        self.build_button = QPushButton("Reset graph")
        self.refresh_button = QPushButton("Refresh")
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
        self.histogram_group = QGroupBox("Histogram")
        self.histogram_scope_combo = QComboBox()
        self.histogram_scope_combo.addItems(["Slice", "Stack"])
        self.histogram_log_checkbox = QCheckBox("Log scale")
        self.histogram_plot = HistogramPlot()

        self.pin_button = QPushButton("Pin selected")
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

    def closeEvent(self, event):  # noqa: N802
        self._restore_hidden_input_layers()
        super().closeEvent(event)

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
            if dock.isFloating():
                return
        except Exception:
            return
        try:
            desired_features = (
                QDockWidget.DockWidgetClosable
                | QDockWidget.DockWidgetMovable
                | QDockWidget.DockWidgetFloatable
            )
            if dock.windowTitle() != "VIPP Workflow":
                dock.setWindowTitle("VIPP Workflow")
            if dock.titleBarWidget() is not None:
                dock.setTitleBarWidget(None)
            if dock.features() != desired_features:
                dock.setFeatures(desired_features)
            if dock.allowedAreas() != Qt.AllDockWidgetAreas:
                dock.setAllowedAreas(Qt.AllDockWidgetAreas)
            self._dock_chrome_configured = True
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

        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Input"))
        toolbar.addWidget(self.layer_combo, 1)
        toolbar.addWidget(QLabel("Preview"))
        toolbar.addWidget(self.preview_mode_combo)
        toolbar.addWidget(self.follow_dims_checkbox)
        toolbar.addWidget(self.build_button)
        toolbar.addWidget(self.refresh_button)
        root.addLayout(toolbar)

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
        self._sync_side_panel_toggles()
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
        histogram_layout = QVBoxLayout(self.histogram_group)
        histogram_scope_layout = QHBoxLayout()
        histogram_scope_layout.addWidget(QLabel("Scope"))
        histogram_scope_layout.addWidget(self.histogram_scope_combo, 1)
        histogram_layout.addLayout(histogram_scope_layout)
        histogram_layout.addWidget(self.histogram_log_checkbox)
        histogram_layout.addWidget(self.histogram_plot)
        layout.addWidget(self.histogram_group)
        metadata_layout = QVBoxLayout(self.metadata_group)
        metadata_layout.addWidget(self.metadata_table)
        metadata_layout.addWidget(self.history_title)
        metadata_layout.addWidget(self.history_label)
        layout.addWidget(self.metadata_group)

        actions = QHBoxLayout()
        actions.addWidget(self.pin_button)
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
        self.build_button.clicked.connect(self._reset_graph)
        self.refresh_button.clicked.connect(self._refresh_and_run)
        self.layer_combo.currentTextChanged.connect(self.run_pipeline)
        self.preview_mode_combo.currentTextChanged.connect(self._update_thumbnails)
        self.follow_dims_checkbox.toggled.connect(self._update_thumbnails)
        self.pin_button.clicked.connect(lambda: self.pin_node(self._selected_node_id))
        self.thumbnail_checkbox.toggled.connect(
            self._on_selected_preview_toggled,
        )
        self.histogram_log_checkbox.toggled.connect(self._update_histogram)
        self.histogram_scope_combo.currentTextChanged.connect(self._update_histogram)
        self.auto_contrast_button.clicked.connect(self._apply_auto_contrast)
        self.left_panel_toggle.clicked.connect(self._toggle_left_panel)
        self.right_panel_toggle.clicked.connect(self._toggle_right_panel)
        self.palette_search.textChanged.connect(self.palette.set_filter_text)

        self.palette.operation_requested.connect(self.add_node_from_palette)
        self.graph_view.node_create_requested.connect(self._add_node_at)
        self.graph_view.node_selected.connect(self._select_node)
        self.graph_view.pin_requested.connect(self.pin_node)
        self.graph_view.connection_requested.connect(self._connect_nodes)
        self.graph_view.connection_removed.connect(self._disconnect_nodes)
        self.graph_view.status_message.connect(self.status_label.setText)

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
            self.viewer.dims.events.current_step.connect(
                lambda _=None: self._on_dims_changed()
            )
        except Exception:
            pass

    def _toggle_left_panel(self) -> None:
        self._set_left_panel_visible(self.palette_panel.isHidden())

    def _toggle_right_panel(self) -> None:
        self._set_right_panel_visible(self.inspector_panel.isHidden())

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
        node = self.pipeline.add_node(operation_id)
        self.graph_view.add_node(node, position)
        self._sync_pin_ui()
        self.graph_view.select_node(node.id)
        self.run_pipeline()
        self.status_label.setText(f"Added '{node.title}'.")
        return node

    def _reset_graph(self) -> None:
        self.pipeline.reset_starter_graph()
        self._preview_disabled_node_ids.clear()
        self._clear_active_pin(status=False)
        self._build_graph_from_pipeline()
        self._select_node("gaussian")
        self.run_pipeline()
        self.status_label.setText("Starter graph restored.")

    def _build_graph_from_pipeline(self) -> None:
        self.graph_view.build_graph(
            self.pipeline.nodes.values(),
            self.pipeline.connections,
        )
        self._sync_pin_ui()

    def _refresh_and_run(self) -> None:
        self._refresh_layer_choices()
        self.run_pipeline()

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

    def _connect_nodes(self, source_id: str, target_id: str) -> None:
        result = self.pipeline.connect(source_id, target_id)
        if not result.success:
            self.status_label.setText(result.message)
            return
        for connection in result.removed:
            self.graph_view.remove_connection(
                connection.source_id,
                connection.target_id,
                notify=False,
            )
        self.graph_view.add_connection(source_id, target_id)
        self.run_pipeline()
        self.status_label.setText(result.message)

    def _disconnect_nodes(self, source_id: str, target_id: str) -> None:
        if self.pipeline.disconnect(source_id, target_id):
            self.run_pipeline()
            self.status_label.setText(f"Disconnected {source_id} -> {target_id}.")

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
        specs = self.pipeline.node_parameter_specs(node_id)
        self.parameter_group.setHidden(not specs)
        if not specs:
            return

        node = self.pipeline.nodes[node_id]
        for spec in specs:
            bounds = self._parameter_bounds_for(node_id, spec)
            control_class = ChoiceControl if spec.kind == "choice" else ParameterControl
            widget = control_class(spec, node.params.get(spec.name), bounds)
            node.params[spec.name] = widget.value()
            widget.valueChanged.connect(
                lambda value, name=spec.name: self._on_param_changed(name, value)
            )
            self.parameter_form.addRow(spec.label, widget)
            self._parameter_widgets[spec.name] = widget

    def _refresh_selected_parameter_controls(self) -> bool:
        if self._selected_node_id not in self.pipeline.nodes:
            return False
        changed = False
        node = self.pipeline.nodes[self._selected_node_id]
        for spec in self.pipeline.node_parameter_specs(self._selected_node_id):
            widget = self._parameter_widgets.get(spec.name)
            if widget is None:
                continue
            previous = node.params.get(spec.name)
            if spec.name == "channel_axis":
                preferred = self._preferred_channel_axis(self._selected_node_id)
                if preferred is not None:
                    previous = preferred
                    if node.params.get(spec.name) != preferred:
                        node.params[spec.name] = preferred
                        changed = True
            widget.set_bounds(
                self._parameter_bounds_for(self._selected_node_id, spec),
                previous,
                emit=False,
            )
            current = widget.value()
            if current != previous:
                node.params[spec.name] = current
                changed = True
        return changed

    def _parameter_bounds_for(self, node_id: str, spec) -> ParameterBounds:
        if spec.kind == "choice":
            return ParameterBounds(0, max(len(spec.choices) - 1, 0), 1, 0)
        node = self.pipeline.nodes.get(node_id)
        if (
            node is not None
            and node.operation_id == "contrast_stretch"
            and spec.name in {"alpha", "beta"}
        ):
            return self._contrast_parameter_bounds(node_id, spec)
        if spec.name == "threshold":
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
        return ParameterBounds(
            spec.minimum,
            spec.maximum,
            spec.step,
            spec.decimals,
            expandable=True,
        )

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
        return ParameterBounds(0, maximum, 1, 0)

    def _channel_axis_bounds(self, node_id: str, spec) -> ParameterBounds:
        data = self.pipeline.input_data_for_node(node_id)
        if data is None:
            return ParameterBounds(spec.minimum, spec.maximum, spec.step, spec.decimals)
        maximum = max(np.asarray(data).ndim - 1, 0)
        return ParameterBounds(0, maximum, 1, 0)

    def _selected_channel_axis(self, node_id: str, arr: np.ndarray) -> int | None:
        node = self.pipeline.nodes.get(node_id)
        if node is not None and "channel_axis" in node.params:
            return int(
                np.clip(int(node.params.get("channel_axis", 0)), 0, arr.ndim - 1)
            )
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

    def _clear_parameter_form(self) -> None:
        self._parameter_widgets.clear()
        while self.parameter_form.count():
            item = self.parameter_form.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _on_param_changed(self, name: str, value) -> None:
        self.pipeline.set_param(self._selected_node_id, name, value)
        if name in {"axis", "channel_axis"}:
            self._refresh_selected_parameter_controls()
        self._debounce_timer.start()

    def _apply_auto_contrast(self) -> None:
        node_id = self._selected_node_id
        node = self.pipeline.nodes.get(node_id)
        if node is None or node.operation_id != "contrast_stretch":
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
        self.pipeline.set_param(node_id, "alpha", alpha)
        self.pipeline.set_param(node_id, "beta", beta)
        self._debounce_timer.stop()
        self._render_parameters(node_id)
        self.run_pipeline()
        saturation = self.auto_saturation_control.value()
        self.status_label.setText(
            f"Auto contrast set '{node.title}' to {saturation:.2f}% saturation "
            f"({lower:.3g} to {upper:.3g})."
        )

    def run_pipeline(self) -> None:
        layer = self._selected_input_layer()
        if layer is None:
            self._restore_hidden_input_layers()
            self.pipeline.run(None)
            self._update_thumbnails()
            self._update_metadata_panel()
            self._update_histogram()
            self.status_label.setText("No image layer selected.")
            return

        self._last_input_layer_name = layer.name
        self._restore_hidden_input_layers(except_layer=layer)
        try:
            self.pipeline.run(
                layer.data,
                input_metadata=getattr(layer, "metadata", None),
                input_name=getattr(layer, "name", ""),
            )
        except Exception as exc:
            self.status_label.setText(f"Pipeline error: {exc}")
            return
        if self._refresh_selected_parameter_controls():
            self.pipeline.run(
                layer.data,
                input_metadata=getattr(layer, "metadata", None),
                input_name=getattr(layer, "name", ""),
            )
        self._hide_input_layer_for_inspection(layer)

        self._update_thumbnails()
        self._refresh_inspection_layer_if_active()
        self._inspect_selected_node()
        self._refresh_pinned_layer_if_active()
        self._update_metadata_panel()
        self._update_histogram()
        self.status_label.setText(
            f"Graph updated from '{layer.name}'. "
            "Connect ports to build alternate paths."
        )

    def _on_dims_changed(self) -> None:
        self._update_thumbnails()
        self._update_metadata_panel()
        self._update_histogram()

    def _update_thumbnails(self) -> None:
        mode = self.preview_mode_combo.currentText()
        current_step = (
            self._current_step() if self.follow_dims_checkbox.isChecked() else None
        )
        previews_visible_globally = mode.lower() != "off"
        for node_id, data in self.pipeline.outputs.items():
            self.graph_view.set_node_metadata(
                node_id,
                format_compact_metadata(self.pipeline.output_states.get(node_id)),
            )
            self.graph_view.set_node_output_type(
                node_id,
                self._node_output_type(node_id),
            )
            self.graph_view.set_node_can_pin(node_id, self._node_can_pin(node_id))
            preview_enabled = (
                previews_visible_globally and self._node_preview_enabled(node_id)
            )
            self.graph_view.set_node_preview_enabled(node_id, preview_enabled)
            if not preview_enabled:
                continue
            preview = make_preview(
                data,
                mode=mode,
                current_step=current_step,
                state=self.pipeline.output_states.get(node_id),
            )
            thumbnail = normalize_thumbnail(preview)
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
        if checked:
            self._preview_disabled_node_ids.discard(node_id)
        else:
            self._preview_disabled_node_ids.add(node_id)
        self._update_thumbnails()
        state = "enabled" if checked else "disabled"
        self.status_label.setText(
            f"Thumbnail preview {state} for '{self._node_title(node_id)}'."
        )

    def _update_metadata_panel(self) -> None:
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

    def _current_view_label(self, state) -> str:
        if state is None or not self.follow_dims_checkbox.isChecked():
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
            step = self._axis_step_label(axis_index, int(size), current_step)
            parts.append(f"{axis.name}={step}/{int(size) - 1}")
        return ", ".join(parts)

    def _axis_step_label(
        self,
        axis_index: int,
        axis_size: int,
        current_step: tuple,
    ) -> int:
        try:
            step = int(current_step[axis_index])
        except Exception:
            step = axis_size // 2
        return int(np.clip(step, 0, max(axis_size - 1, 0)))

    def _update_histogram(self) -> None:
        data = self.pipeline.outputs.get(self._selected_node_id)
        counts, x_range = _histogram_summary(
            data,
            state=self.pipeline.output_states.get(self._selected_node_id),
            scope=self.histogram_scope_combo.currentText(),
            current_step=(
                self._current_step() if self.follow_dims_checkbox.isChecked() else None
            ),
        )
        self.histogram_plot.set_histogram(
            counts,
            log_scale=self.histogram_log_checkbox.isChecked(),
            x_range=x_range,
        )

    def inspect_node(self, node_id: str) -> None:
        data = self.pipeline.outputs.get(node_id)
        if data is None:
            self.status_label.setText("That node has no output to inspect yet.")
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
        if self.pipeline.outputs.get(self._selected_node_id) is not None:
            self.inspect_node(self._selected_node_id)

    def pin_node(self, node_id: str) -> None:
        if not self._node_can_pin(node_id):
            self.status_label.setText(
                f"'{self._node_title(node_id)}' does not produce a mask overlay."
            )
            return
        if node_id == self._active_pinned_node_id:
            self._clear_active_pin(status=True)
            return
        data = self.pipeline.outputs.get(node_id)
        if data is None:
            self.status_label.setText("That node has no output to pin yet.")
            return
        self._set_active_pin_layer(node_id, data)
        self.status_label.setText(f"Pinned '{self._node_title(node_id)}'.")

    def _set_active_pin_layer(self, node_id: str, data) -> None:
        title = self._node_title(node_id)
        display_data = self._display_data(data)
        metadata = {
            "napari_vipp_kind": "pinned",
            "node_id": node_id,
            "data_kind": self._data_kind(data),
            "display_kind": self._display_kind(data, "pinned"),
            "display_ndim": np.asarray(display_data).ndim,
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
        display_data = self._display_data(data)
        metadata = {
            **metadata,
            "data_kind": self._data_kind(data),
            "display_kind": self._display_kind(data, role),
            "display_ndim": np.asarray(display_data).ndim,
        }
        layer = self._layer_by_name(name)
        if layer is None:
            self._add_image_or_labels(name, data, metadata=metadata)
            return
        if self._generated_layer_needs_replacement(layer, metadata):
            self._remove_layer(layer)
            self._add_image_or_labels(name, data, metadata=metadata)
            return
        layer.data = display_data
        layer.metadata.update(metadata)
        layer.visible = True
        self._configure_generated_layer(layer, data, metadata)

    def _add_image_or_labels(self, name: str, data, metadata: dict):
        display_data = self._display_data(data)
        if metadata["display_kind"] == "labels" and hasattr(self.viewer, "add_labels"):
            return self.viewer.add_labels(display_data, name=name, metadata=metadata)
        kwargs = {"name": name, "metadata": metadata}
        if metadata["data_kind"] == "mask":
            kwargs.update(
                {
                    "blending": "opaque",
                    "colormap": "gray",
                    "contrast_limits": (0, 1),
                }
            )
        return self.viewer.add_image(display_data, **kwargs)

    def _generated_layer_needs_replacement(self, layer, metadata: dict) -> bool:
        return (
            layer.metadata.get("display_kind") != metadata["display_kind"]
            or layer.metadata.get("data_kind") != metadata["data_kind"]
            or layer.metadata.get("display_ndim") != metadata["display_ndim"]
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

    def _display_kind(self, data, role: str) -> str:
        if role == "pinned" and self._data_kind(data) == "mask":
            return "labels"
        return "image"

    def _data_kind(self, data) -> str:
        return "mask" if np.asarray(data).dtype == bool else "image"

    def _display_data(self, data):
        arr = np.asarray(data)
        if arr.dtype == bool:
            return arr.astype(np.uint8)
        return data

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
        with QSignalBlocker(self.thumbnail_checkbox):
            self.thumbnail_checkbox.setChecked(
                self._node_preview_enabled(self._selected_node_id)
            )

    def _sync_auto_contrast_ui(self) -> None:
        node = self.pipeline.nodes.get(self._selected_node_id)
        self.auto_contrast_group.setVisible(
            node is not None and node.operation_id == "contrast_stretch"
        )

    def _node_preview_enabled(self, node_id: str) -> bool:
        return node_id not in self._preview_disabled_node_ids

    def _node_can_pin(self, node_id: str) -> bool:
        node = self.pipeline.nodes.get(node_id)
        if node is None:
            return False
        data = self.pipeline.outputs.get(node_id)
        if data is not None:
            return self._data_kind(data) == "mask"
        return node.output_type == "mask"

    def _node_output_type(self, node_id: str) -> str:
        node = self.pipeline.nodes.get(node_id)
        data = self.pipeline.outputs.get(node_id)
        if data is not None:
            return self._data_kind(data)
        return node.output_type if node is not None else "image"

    def _remove_layer(self, layer) -> None:
        try:
            self.viewer.layers.remove(layer)
        except Exception:
            pass

    def _pinned_layer_name(self, title: str) -> str:
        return f"VIPP Pinned: {title}"

    def _current_step(self):
        try:
            return tuple(self.viewer.dims.current_step)
        except Exception:
            return None

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


def _slider_safe_bounds(
    minimum: float,
    maximum: float,
    step: float | int,
    decimals: int,
    expandable: bool = False,
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
) -> np.ndarray | None:
    counts, _x_range = _histogram_summary(
        data,
        state=state,
        scope=scope,
        current_step=current_step,
    )
    return counts


def _histogram_summary(
    data,
    state=None,
    scope: str = "Slice",
    current_step=None,
) -> tuple[np.ndarray | None, tuple[float, float] | None]:
    if data is None:
        return None, None

    if scope.lower() == "stack":
        arr = np.asarray(data)
    else:
        preview = make_preview(
            data,
            mode="slice",
            current_step=current_step,
            state=state,
        )
        if preview is None:
            return None, None
        arr = np.asarray(preview)
    if arr.size == 0:
        return None, None
    if arr.ndim >= 3 and arr.shape[-1] in (3, 4):
        arr = (
            arr[..., 0].astype(np.float32) * 0.299
            + arr[..., 1].astype(np.float32) * 0.587
            + arr[..., 2].astype(np.float32) * 0.114
        )
    if arr.dtype == bool:
        return np.bincount(arr.ravel().astype(np.uint8), minlength=2), (0.0, 1.0)

    values = arr[np.isfinite(arr)].ravel()
    if values.size == 0:
        return None, None
    if values.size > 500_000:
        stride = int(np.ceil(values.size / 500_000))
        values = values[::stride]

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
