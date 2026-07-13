"""Axis selection, slicing, reordering, and table-column UI controls."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from html import escape

import numpy as np
from qtpy.QtCore import QSignalBlocker, QSize, Qt, Signal
from qtpy.QtGui import QBrush, QColor, QPainter, QPen
from qtpy.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from napari_vipp.core.operations import NO_TABLE_COLUMNS_VALUE
from napari_vipp.ui.controls import _configure_numeric_spin_box


@dataclass(frozen=True)
class AxisSliceOption:
    index: int
    name: str
    axis_type: str
    size: int

    @property
    def title(self) -> str:
        return self.name.upper() if len(self.name) == 1 else self.name


def _axis_heading_text(option: AxisSliceOption, *, mode: str = "keep") -> str:
    accent = "#fbbf24" if mode == "remove" else "#93c5fd"
    return (
        f"<span style='font-weight: 700; color: {accent};'>"
        f"{escape(option.title)}</span>"
        f"<span style='color: #d1d5db;'>&nbsp;({escape(option.axis_type)})</span>"
        f"<span style='color: #94a3b8;'>&nbsp;-&nbsp;size {int(option.size)}</span>"
    )


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


__all__ = [
    "AxisIntervalSlider",
    "AxisOrderListWidget",
    "AxisSelectionRow",
    "AxisSliceControl",
    "AxisSliceOption",
    "ReorderAxesControl",
    "SelectTableColumnsControl",
    "_axis_heading_text",
    "_axis_indices_from_value",
    "_axis_options_from_order",
    "_axis_order_item_text",
    "_axis_order_tokens",
    "_axis_order_value",
    "_axis_slice_state_from_value",
    "_clamped_index",
    "_clamped_range",
    "_control_signal_blockers",
    "_named_axis_order",
    "_numeric_axis_order",
    "_parse_axis_ranges",
    "_parse_int_list",
]
