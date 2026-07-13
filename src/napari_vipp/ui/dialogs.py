"""Standalone dialogs used by the VIPP workflow editor."""

from __future__ import annotations

from dataclasses import dataclass

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QBrush, QColor
from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from napari_vipp._theme import category_color
from napari_vipp.ui.examples import (
    EXAMPLE_WORKFLOWS,
    ExampleWorkflowSpec,
    _example_workflow_by_id,
)
from napari_vipp.ui.search import _fuzzy_match, _normalize_search_text


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


__all__ = [
    "ConnectionInsertCandidate",
    "ConnectionInsertDialog",
    "ConnectionInsertMappingDialog",
    "ConnectionInsertPortMapping",
    "ExampleWorkflowDialog",
    "TunnelManagerDialog",
    "TunnelSummary",
]
