"""Resizable, sortable inspection window for VIPP table outputs."""

from __future__ import annotations

import numbers
import re
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from qtpy.QtCore import (
    QAbstractTableModel,
    QEvent,
    QObject,
    QPointF,
    QRunnable,
    Qt,
    QThreadPool,
    Signal,
)
from qtpy.QtGui import QPainter, QPalette, QPen
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProxyStyle,
    QPushButton,
    QStyle,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from napari_vipp.core.tables import TableData, save_table_output
from napari_vipp.ui.palette_roles import theme_colors

_NATURAL_TEXT_PART = re.compile(r"(\d+)")
_SORT_HINT = (
    "Click a column heading to sort ascending; click it again to sort descending."
)
_EXPORT_NOTE = (
    "Sorting changes this view only; export preserves the workflow's exact "
    "row order."
)
_BACKGROUND_TASK_ROW_THRESHOLD = 50_000


class _SortChevronStyle(QProxyStyle):
    """Render a compact, palette-aware header sort chevron.

    Qt's native header arrows can become nearly invisible when napari changes
    palette without changing the platform style.  Keep the standard
    ``QHeaderView`` sort-indicator state and replace only that primitive with
    two thin strokes, so assistive APIs and ordinary header behaviour remain
    conventional.
    """

    def drawPrimitive(self, element, option, painter, widget=None) -> None:  # noqa: N802
        if element != QStyle.PE_IndicatorHeaderArrow:
            super().drawPrimitive(element, option, painter, widget)
            return

        order = (
            widget.sortIndicatorOrder()
            if isinstance(widget, QHeaderView)
            else Qt.AscendingOrder
        )
        rect = option.rect
        center_x = float(rect.center().x())
        center_y = float(rect.center().y())
        half_width = max(2.5, min(4.0, (float(rect.width()) - 2.0) / 2.0))
        half_height = max(1.5, min(2.5, (float(rect.height()) - 2.0) / 2.0))
        peak_y = (
            center_y - half_height
            if order == Qt.AscendingOrder
            else center_y + half_height
        )
        edge_y = (
            center_y + half_height
            if order == Qt.AscendingOrder
            else center_y - half_height
        )
        colors = theme_colors(option.palette)

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        pen = QPen(colors.text)
        pen.setWidthF(1.4)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)
        peak = QPointF(center_x, peak_y)
        painter.drawLine(QPointF(center_x - half_width, edge_y), peak)
        painter.drawLine(peak, QPointF(center_x + half_width, edge_y))
        painter.restore()


def _display_value(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _is_missing_sort_value(value: object) -> bool:
    if value is None or (isinstance(value, str) and value == ""):
        return True
    if isinstance(value, (bool, np.bool_)):
        return False
    if isinstance(value, numbers.Number):
        try:
            differs_from_self = value != value
            if isinstance(differs_from_self, (bool, np.bool_)) and bool(
                differs_from_self
            ):
                return True
        except (TypeError, ValueError):
            pass
        try:
            return bool(np.isnan(value))
        except (TypeError, ValueError):
            return False
    return False


def _natural_text_key(value: object) -> tuple:
    text = _display_value(value)
    parts: list[tuple] = []
    for part in _NATURAL_TEXT_PART.split(text.casefold()):
        if not part:
            continue
        if part.isdigit():
            # Keep digit strings textual while ordering their numeric runs in
            # the familiar file-browser style. Length and source text make
            # identifiers such as ``1`` and ``001`` deterministic rather than
            # silently coercing them to the same number.
            parts.append((1, int(part), len(part), part))
        else:
            parts.append((0, part))
    return tuple(parts), text


def _value_sort_family(value: object) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "bool"
    if isinstance(value, numbers.Real):
        return "number"
    return "text"


def _column_sort_key(family: str):
    if family == "bool":
        return lambda value: bool(value)
    if family == "number":
        return lambda value: value.item() if isinstance(value, np.generic) else value
    return _natural_text_key


class _TableTaskCancelled(RuntimeError):
    pass


def _sorted_source_rows(
    table: TableData,
    column: int,
    order,
    *,
    cancel_event: threading.Event | None = None,
) -> list[int]:
    present: list[int] = []
    missing: list[int] = []
    family: str | None = None
    for source_row, row in enumerate(table.rows):
        if (
            cancel_event is not None
            and source_row % 4096 == 0
            and cancel_event.is_set()
        ):
            raise _TableTaskCancelled
        value = row[int(column)]
        if _is_missing_sort_value(value):
            missing.append(source_row)
            continue
        present.append(source_row)
        value_family = _value_sort_family(value)
        if family is None:
            family = value_family
        elif family != value_family:
            family = "text"
    key = _column_sort_key(family or "text")
    try:
        ordered_present = sorted(
            present,
            key=lambda row_index: key(table.rows[row_index][int(column)]),
            reverse=order == Qt.DescendingOrder,
        )
    except TypeError:
        # A permissive TableData column can hold otherwise numeric scalars that
        # Python cannot compare with one another (for example timedelta64 and
        # int). Fall back to the same natural display-text order used for a
        # mixed-type column instead of letting a header click fail.
        ordered_present = sorted(
            present,
            key=lambda row_index: _natural_text_key(
                table.rows[row_index][int(column)]
            ),
            reverse=order == Qt.DescendingOrder,
        )
    present = ordered_present
    if cancel_event is not None and cancel_event.is_set():
        raise _TableTaskCancelled
    present.extend(missing)
    return present


@dataclass(frozen=True)
class _SortOutcome:
    generation: int
    column: int
    order: object
    row_order: list[int] | None = None
    error: str = ""
    cancelled: bool = False


class _SortSignals(QObject):
    finished = Signal(object)


class _SortWorker(QRunnable):
    def __init__(
        self,
        table: TableData,
        *,
        generation: int,
        column: int,
        order,
        cancel_event: threading.Event,
    ) -> None:
        super().__init__()
        self.table = table
        self.generation = int(generation)
        self.column = int(column)
        self.order = order
        self.cancel_event = cancel_event
        self.signals = _SortSignals()

    def run(self) -> None:
        try:
            row_order = _sorted_source_rows(
                self.table,
                self.column,
                self.order,
                cancel_event=self.cancel_event,
            )
        except _TableTaskCancelled:
            outcome = _SortOutcome(
                self.generation,
                self.column,
                self.order,
                cancelled=True,
            )
        except Exception as exc:
            outcome = _SortOutcome(
                self.generation,
                self.column,
                self.order,
                error=str(exc),
            )
        else:
            outcome = _SortOutcome(
                self.generation,
                self.column,
                self.order,
                row_order=row_order,
            )
        self.signals.finished.emit(outcome)


@dataclass(frozen=True)
class _ExportOutcome:
    generation: int
    path: Path | None = None
    error: str = ""


class _ExportSignals(QObject):
    finished = Signal(object)


class _ExportWorker(QRunnable):
    def __init__(
        self,
        table: TableData,
        path: Path,
        *,
        format: str,
        generation: int,
    ) -> None:
        super().__init__()
        self.table = table
        self.path = Path(path)
        self.format = str(format)
        self.generation = int(generation)
        self.signals = _ExportSignals()

    def run(self) -> None:
        try:
            path = save_table_output(
                self.table,
                self.path,
                format=self.format,
                overwrite=True,
            )
        except Exception as exc:
            outcome = _ExportOutcome(self.generation, error=str(exc))
        else:
            outcome = _ExportOutcome(self.generation, path=Path(path))
        self.signals.finished.emit(outcome)


class ResultTableModel(QAbstractTableModel):
    """Read-only view model with stable, typed row sorting."""

    def __init__(self, table: TableData | None = None, parent=None) -> None:
        super().__init__(parent)
        self._table = table or TableData((), ())
        self._row_order: list[int] | None = None

    @property
    def table(self) -> TableData:
        return self._table

    def set_table(self, table: TableData) -> None:
        if not isinstance(table, TableData):
            raise TypeError("ResultTableModel expects a TableData object.")
        self.beginResetModel()
        self._table = table
        self._row_order = None
        self.endResetModel()

    def rowCount(self, parent=None) -> int:  # noqa: N802
        return (
            0
            if parent is not None and parent.isValid()
            else self._table.row_count
        )

    def columnCount(self, parent=None) -> int:  # noqa: N802
        return (
            0
            if parent is not None and parent.isValid()
            else self._table.column_count
        )

    def source_row(self, displayed_row: int) -> int:
        if not 0 <= int(displayed_row) < self._table.row_count:
            raise IndexError(displayed_row)
        if self._row_order is None:
            return int(displayed_row)
        return self._row_order[int(displayed_row)]

    def raw_value(self, displayed_row: int, column: int) -> object:
        source_row = self.source_row(displayed_row)
        return self._table.rows[source_row][int(column)]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        value = self.raw_value(index.row(), index.column())
        if role in (Qt.DisplayRole, Qt.ToolTipRole):
            return _display_value(value)
        if role == Qt.TextAlignmentRole:
            if isinstance(value, numbers.Number) and not isinstance(
                value,
                (bool, np.bool_),
            ):
                return Qt.AlignRight | Qt.AlignVCenter
            return Qt.AlignLeft | Qt.AlignVCenter
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):  # noqa: N802
        if orientation == Qt.Horizontal and 0 <= section < self._table.column_count:
            column = self._table.columns[section]
            unit = self._table.unit_for(column)
            if role == Qt.DisplayRole:
                return f"{column}\n({unit})" if unit else column
            if role == Qt.ToolTipRole:
                description = f"{column} ({unit})" if unit else column
                return f"{description}. Click to sort; click again to reverse."
        if orientation == Qt.Vertical and role == Qt.DisplayRole:
            return f"{section + 1:,}"
        return None

    def sort(self, column: int, order=Qt.AscendingOrder) -> None:
        if not 0 <= int(column) < self._table.column_count:
            return
        self.apply_row_order(
            _sorted_source_rows(self._table, int(column), order)
        )

    def apply_row_order(self, row_order: list[int]) -> None:
        if len(row_order) != self._table.row_count:
            raise ValueError("Sorted row order does not match the result table.")
        # A model reset deliberately clears cell selection. Without it, Qt
        # keeps the same displayed coordinates selected after sorting even
        # though those coordinates now refer to different scientific records.
        self.beginResetModel()
        self._row_order = list(row_order)
        self.endResetModel()


class ResultTableDialog(QDialog):
    """Nonmodal full-table viewer shared by every table-producing node."""

    exportCompleted = Signal(str)
    sortCompleted = Signal(int, object)
    recalculationRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("VippResultTableDialog")
        self.setAttribute(Qt.WA_WindowPropagation, True)
        self.setWindowTitle("Result table")
        self.setWindowModality(Qt.NonModal)
        self.setSizeGripEnabled(True)
        self.setMinimumSize(420, 280)
        self.resize(1000, 650)

        self.summary_label = QLabel("No table output is available.", self)
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.result_status_panel = QWidget(self)
        self.result_status_panel.setObjectName("ResultTableStatusPanel")
        result_status_layout = QHBoxLayout(self.result_status_panel)
        result_status_layout.setContentsMargins(8, 6, 8, 6)
        result_status_layout.setSpacing(8)
        self.result_status_label = QLabel("", self.result_status_panel)
        self.result_status_label.setObjectName("ResultTableStatusLabel")
        self.result_status_label.setWordWrap(True)
        self.result_status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.result_action_button = QPushButton("Recalculate", self.result_status_panel)
        self.result_action_button.setAccessibleName("Recalculate this result table")
        result_status_layout.addWidget(self.result_status_label, 1)
        result_status_layout.addWidget(self.result_action_button, 0)
        self.result_status_panel.hide()

        self.sort_hint = QLabel(_SORT_HINT, self)
        self.sort_hint.setWordWrap(True)

        self.model = ResultTableModel(parent=self)
        self.table_view = QTableView(self)
        self.table_view.setModel(self.model)
        self.table_view.setAccessibleName("Complete result table")
        self.table_view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_view.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.table_view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table_view.setAlternatingRowColors(True)
        self.table_view.setWordWrap(False)
        self.table_view.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.table_view.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        horizontal_header = self.table_view.horizontalHeader()
        self._sort_chevron_style = _SortChevronStyle()
        self._sort_chevron_style.setParent(horizontal_header)
        horizontal_header.setStyle(self._sort_chevron_style)
        horizontal_header.setSectionsClickable(True)
        horizontal_header.setSortIndicatorShown(False)
        horizontal_header.setSectionResizeMode(QHeaderView.Interactive)
        horizontal_header.setMinimumSectionSize(70)
        horizontal_header.setDefaultSectionSize(160)
        self.table_view.verticalHeader().setDefaultSectionSize(24)

        self.export_note = QLabel(_EXPORT_NOTE, self)
        self.export_note.setWordWrap(True)
        self.export_button = QPushButton("Export CSV/TSV…", self)
        self.export_button.setEnabled(False)
        self.export_button.setToolTip(
            "Export the complete table using the same CSV/TSV writer as the "
            "inspector."
        )
        self.close_button = QPushButton("Close", self)

        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.addStretch(1)
        buttons.addWidget(self.export_button)
        buttons.addWidget(self.close_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.result_status_panel)
        layout.addWidget(self.sort_hint)
        layout.addWidget(self.table_view, 1)
        layout.addWidget(self.export_note)
        layout.addLayout(buttons)

        self._table: TableData | None = None
        self._context_key: tuple[str, int] | None = None
        self._default_export_name = "result-table.csv"
        self._sort_column: int | None = None
        self._sort_order = Qt.AscendingOrder
        # Use the shared pool so destroying or hiding a dialog never waits for
        # an in-flight filesystem write. The immutable TableData snapshot is
        # owned by each worker for the duration of its task.
        self._worker_pool = QThreadPool.globalInstance()
        self._sort_generation = 0
        self._active_sort_cancel_event: threading.Event | None = None
        self._active_sort_worker: _SortWorker | None = None
        self._export_generation = 0
        self._active_export_worker: _ExportWorker | None = None
        self._theme_refresh_in_progress = False

        horizontal_header.sectionClicked.connect(self._sort_by_header)
        self.result_action_button.clicked.connect(self.recalculationRequested.emit)
        self.export_button.clicked.connect(self.request_export)
        self.close_button.clicked.connect(self.close)
        self.refresh_theme()

    @property
    def table(self) -> TableData | None:
        return self._table

    @property
    def context_key(self) -> tuple[str, int] | None:
        return self._context_key

    def set_table(
        self,
        table: TableData,
        *,
        title: str,
        default_export_name: str,
        context_key: tuple[str, int] | None = None,
    ) -> None:
        if not isinstance(table, TableData):
            raise TypeError("ResultTableDialog expects a TableData object.")
        changed = table is not self._table or context_key != self._context_key
        self._table = table
        self._context_key = context_key
        self._default_export_name = str(default_export_name or "result-table.csv")
        self.setWindowTitle(f"{title} — Result table")
        kind = str(table.table_kind or "").strip()
        kind_suffix = f" · {kind}" if kind and kind.casefold() != "table" else ""
        self.summary_label.setText(
            f"{table.row_count:,} rows × {table.column_count:,} fields{kind_suffix}"
        )
        self.table_view.setAccessibleName(f"Complete result table for {title}")
        if changed:
            self._cancel_active_sort()
            self.model.set_table(table)
            self._sort_column = None
            self._sort_order = Qt.AscendingOrder
            self._restore_committed_sort_indicator()
            self.table_view.scrollToTop()
        self.export_button.setEnabled(self._active_export_worker is None)

    def set_result_status(
        self,
        message: str = "",
        *,
        action_text: str = "",
        action_enabled: bool = False,
        attention_required: bool = False,
    ) -> None:
        """Present cached-result currentness without discarding retained rows."""

        message = str(message or "").strip()
        action_text = str(action_text or "").strip()
        visible = bool(message or action_text)
        self.result_status_label.setText(message)
        self.result_status_label.setVisible(bool(message))
        self.result_action_button.setText(action_text or "Recalculate")
        self.result_action_button.setVisible(bool(action_text))
        self.result_action_button.setEnabled(bool(action_text and action_enabled))
        self.result_action_button.setProperty(
            "attentionRequired",
            bool(attention_required),
        )
        self.result_status_panel.setVisible(visible)
        self.export_button.setToolTip(
            (
                "This retained table is not current. Review the warning before "
                "exporting the complete table."
            )
            if visible
            else "Export the complete table as CSV or TSV."
        )
        self._refresh_result_status_theme()

    def _refresh_result_status_theme(self) -> None:
        colors = theme_colors(self.palette())
        warning = colors.warning
        self.result_status_panel.setStyleSheet(
            "QWidget#ResultTableStatusPanel {"
            f" background-color: {warning.surface.name()};"
            f" border: 1px solid {warning.border.name()};"
            " border-radius: 3px;"
            "}"
            "QLabel#ResultTableStatusLabel {"
            f" color: {warning.foreground.name()};"
            " background: transparent; border: none;"
            "}"
        )
        attention_required = bool(
            self.result_action_button.property("attentionRequired")
        )
        self.result_action_button.setStyleSheet(
            (
                "QPushButton {"
                f" background-color: {warning.surface.name()};"
                f" color: {warning.foreground.name()};"
                f" border: 2px solid {warning.border.name()};"
                " border-radius: 3px; font-weight: 650; padding: 4px 8px;"
                "}"
                "QPushButton:hover {"
                f" background-color: {warning.surface.lighter(108).name()};"
                "}"
                "QPushButton:pressed {"
                f" background-color: {warning.surface.darker(108).name()};"
                "}"
            )
            if attention_required
            else ""
        )

    def _sort_by_header(self, column: int) -> None:
        if self._table is None:
            return
        if self._sort_column == int(column):
            order = (
                Qt.DescendingOrder
                if self._sort_order == Qt.AscendingOrder
                else Qt.AscendingOrder
            )
        else:
            order = Qt.AscendingOrder
        self._show_sort_indicator(int(column), order)
        if self._table.row_count >= _BACKGROUND_TASK_ROW_THRESHOLD:
            self._start_background_sort(int(column), order)
            return
        try:
            row_order = _sorted_source_rows(self._table, int(column), order)
        except Exception:
            self._restore_committed_sort_indicator()
            raise
        self._apply_sort_result(int(column), order, row_order)

    def _show_sort_indicator(self, column: int, order) -> None:
        header = self.table_view.horizontalHeader()
        header.setSortIndicator(int(column), order)
        header.setSortIndicatorShown(True)
        header.viewport().update()

    def _restore_committed_sort_indicator(self) -> None:
        header = self.table_view.horizontalHeader()
        if self._sort_column is None:
            header.setSortIndicatorShown(False)
            header.viewport().update()
            return
        self._show_sort_indicator(self._sort_column, self._sort_order)

    def _apply_sort_result(
        self,
        column: int,
        order,
        row_order: list[int],
    ) -> None:
        selection_model = self.table_view.selectionModel()
        if selection_model is not None:
            selection_model.clearSelection()
        self.model.apply_row_order(row_order)
        self._sort_column = int(column)
        self._sort_order = order
        self._show_sort_indicator(int(column), order)
        self.sort_hint.setText(_SORT_HINT)
        self.sortCompleted.emit(int(column), order)

    def _start_background_sort(self, column: int, order) -> None:
        table = self._table
        if table is None:
            return
        self._cancel_active_sort()
        self._show_sort_indicator(int(column), order)
        self._sort_generation += 1
        generation = self._sort_generation
        cancel_event = threading.Event()
        worker = _SortWorker(
            table,
            generation=generation,
            column=int(column),
            order=order,
            cancel_event=cancel_event,
        )
        worker.signals.finished.connect(self._on_sort_finished)
        self._active_sort_cancel_event = cancel_event
        self._active_sort_worker = worker
        self.table_view.horizontalHeader().setSectionsClickable(False)
        column_name = table.columns[int(column)]
        self.sort_hint.setText(
            f"Sorting {table.row_count:,} rows by “{column_name}”…"
        )
        self._worker_pool.start(worker)

    def _on_sort_finished(self, outcome: _SortOutcome) -> None:
        if outcome.generation != self._sort_generation:
            return
        self._active_sort_cancel_event = None
        self._active_sort_worker = None
        self.table_view.horizontalHeader().setSectionsClickable(True)
        if outcome.cancelled:
            self.sort_hint.setText(_SORT_HINT)
            self._restore_committed_sort_indicator()
            return
        if outcome.error:
            self.sort_hint.setText(f"Sort failed: {outcome.error}")
            self._restore_committed_sort_indicator()
            return
        if outcome.row_order is None:
            self.sort_hint.setText(_SORT_HINT)
            self._restore_committed_sort_indicator()
            return
        self._apply_sort_result(
            outcome.column,
            outcome.order,
            outcome.row_order,
        )

    def _cancel_active_sort(self) -> None:
        if self._active_sort_cancel_event is not None:
            self._active_sort_cancel_event.set()
        self._sort_generation += 1
        self._active_sort_cancel_event = None
        self._active_sort_worker = None
        self.table_view.horizontalHeader().setSectionsClickable(True)
        self.sort_hint.setText(_SORT_HINT)
        self._restore_committed_sort_indicator()

    def request_export(self) -> None:
        if self._table is None:
            return
        request = choose_table_export_target(
            self,
            default_name=self._default_export_name,
            caption="Save result table",
        )
        if request is None:
            return
        requested, format = request
        table = self._table
        self._start_background_export(table, requested, format=format)

    def _start_background_export(
        self,
        table: TableData,
        path: Path,
        *,
        format: str,
    ) -> None:
        if self._active_export_worker is not None:
            return
        self._export_generation += 1
        worker = _ExportWorker(
            table,
            Path(path),
            format=format,
            generation=self._export_generation,
        )
        worker.signals.finished.connect(self._on_export_finished)
        self._active_export_worker = worker
        self.export_button.setEnabled(False)
        self.export_button.setText("Exporting…")
        self.export_note.setText(
            f"Exporting {table.row_count:,} rows in the background…"
        )
        self._worker_pool.start(worker)

    def _on_export_finished(self, outcome: _ExportOutcome) -> None:
        if outcome.generation != self._export_generation:
            return
        self._active_export_worker = None
        self.export_button.setText("Export CSV/TSV…")
        self.export_button.setEnabled(self._table is not None)
        self.export_note.setText(_EXPORT_NOTE)
        if outcome.error:
            QMessageBox.critical(self, "Table export failed", outcome.error)
            return
        if outcome.path is not None:
            self.exportCompleted.emit(str(outcome.path))

    def export_table(self, path: str | Path, *, format: str = "auto") -> Path:
        if self._table is None:
            raise ValueError("No table output is available to export.")
        return save_table_output(
            self._table,
            path,
            format=format,
            overwrite=True,
        )

    def refresh_theme(self, palette: QPalette | None = None) -> None:
        if self._theme_refresh_in_progress:
            return
        self._theme_refresh_in_progress = True
        try:
            if palette is None:
                owner = self.parentWidget()
                palette = (
                    QWidget.palette(owner)
                    if owner is not None
                    else QApplication.palette()
                )
            self.setPalette(palette)
            colors = theme_colors(palette)
            highlight = palette.color(QPalette.Highlight)
            highlighted_text = palette.color(QPalette.HighlightedText)
            self.setStyleSheet(
                "QDialog#VippResultTableDialog {"
                f" background: {colors.surface.name()}; color: {colors.text.name()};"
                "}"
                "QTableView {"
                f" background: {colors.surface.name()};"
                f" alternate-background-color: {colors.alternate_surface.name()};"
                f" color: {colors.text.name()};"
                f" border: 1px solid {colors.border.name()};"
                f" gridline-color: {colors.border.name()};"
                f" selection-background-color: {highlight.name()};"
                f" selection-color: {highlighted_text.name()};"
                "}"
                "QHeaderView::section {"
                f" background: {colors.raised_surface.name()};"
                f" color: {colors.text.name()};"
                f" border: 1px solid {colors.border.name()};"
                " padding: 5px;"
                "}"
                "QLabel {"
                f" color: {colors.text.name()};"
                "}"
            )
            self._refresh_result_status_theme()
        finally:
            self._theme_refresh_in_progress = False

    def changeEvent(self, event: QEvent) -> None:  # noqa: N802
        super().changeEvent(event)
        if event.type() in {
            QEvent.ApplicationPaletteChange,
            QEvent.PaletteChange,
            QEvent.StyleChange,
        }:
            self.refresh_theme()

    def closeEvent(self, event) -> None:  # noqa: N802
        self._cancel_active_sort()
        super().closeEvent(event)


def choose_table_export_target(
    parent: QWidget,
    *,
    default_name: str,
    caption: str,
) -> tuple[Path, str] | None:
    """Return one conventional CSV/TSV save request without writing data."""

    path, selected_filter = QFileDialog.getSaveFileName(
        parent,
        caption,
        default_name,
        "CSV table (*.csv);;TSV table (*.tsv);;All files (*.*)",
    )
    if not path:
        return None
    requested = Path(path)
    suffix = requested.suffix.casefold()
    # The explicit file-type filter is authoritative, matching conventional
    # save dialogs. Canonicalize the filename with it so CSV bytes are never
    # written behind a .tsv name (or vice versa).
    if selected_filter.startswith("TSV"):
        format = "tsv"
        requested = requested.with_suffix(".tsv")
    elif selected_filter.startswith("CSV"):
        format = "csv"
        requested = requested.with_suffix(".csv")
    elif suffix == ".tsv":
        format = "tsv"
    elif suffix == ".csv":
        format = "csv"
    else:
        format = "auto"
    if not requested.suffix:
        requested = requested.with_suffix(".tsv" if format == "tsv" else ".csv")
    return requested, format


__all__ = [
    "ResultTableDialog",
    "ResultTableModel",
    "choose_table_export_target",
]
