"""Fail-closed editor for per-source numeric batch parameter overrides."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

from qtpy.QtCore import QEvent, QLocale, QSize, Qt, QTimer, Signal
from qtpy.QtGui import QDoubleValidator, QFontMetrics, QIntValidator
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QCompleter,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QSizePolicy,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from napari_vipp.core.batch_parameters import (
    BatchParameterOverride,
    BatchSourceParameterOverrides,
    batch_parameter_override_ineligibility_reason,
    batch_source_item_override_key,
    normalize_batch_parameter_overrides,
)
from napari_vipp.core.pipeline import ParameterSpec, validate_parameter_value
from napari_vipp.core.source_items import SourceItem
from napari_vipp.ui.batch_override_widgets import (
    BatchOverrideDelegate,
    BatchOverrideTable,
)
from napari_vipp.ui.batch_table_style import apply_batch_table_style
from napari_vipp.ui.palette_roles import custom_paint_colors, palette_is_dark
from napari_vipp.ui.toolbar_controls import ToolbarCommandButton, toolbar_icon


class BatchOverrideEditorError(ValueError):
    """The override editor cannot produce an unambiguous typed mapping."""


@dataclass(frozen=True, slots=True)
class BatchOverrideSourceItem:
    """One planned primary source and its human-facing batch item label."""

    source_node_id: str
    label: str
    source_item: SourceItem

    def __post_init__(self) -> None:
        source_node_id = str(self.source_node_id).strip()
        label = str(self.label).strip()
        if not source_node_id:
            raise BatchOverrideEditorError(
                "Every parameter-override source needs its primary source-node ID."
            )
        if not label:
            raise BatchOverrideEditorError(
                "Every parameter-override source needs a non-empty item label."
            )
        if not isinstance(self.source_item, SourceItem):
            raise TypeError("source_item must be a SourceItem.")
        object.__setattr__(self, "source_node_id", source_node_id)
        object.__setattr__(self, "label", label)


@dataclass(frozen=True, slots=True)
class BatchOverrideParameterSpec:
    """One eligible public numeric parameter on a scientific workflow node."""

    node_id: str
    node_label: str
    operation_id: str
    parameter: ParameterSpec
    workflow_value: int | float

    def __post_init__(self) -> None:
        node_id = str(self.node_id).strip()
        node_label = str(self.node_label).strip()
        operation_id = str(self.operation_id).strip()
        if not node_id or not node_label or not operation_id:
            raise BatchOverrideEditorError(
                "Override parameters need a node id, node label, and operation id."
            )
        if not isinstance(self.parameter, ParameterSpec):
            raise TypeError("parameter must be a ParameterSpec.")
        ineligible = batch_parameter_override_ineligibility_reason(
            operation_id,
            self.parameter,
        )
        if ineligible:
            raise BatchOverrideEditorError(
                f"{node_label} / {self.parameter.label} cannot vary per sample: "
                f"{ineligible}."
            )
        try:
            validate_parameter_value(
                self.parameter,
                self.workflow_value,
                context=f"Workflow value for {node_label}",
            )
        except (TypeError, ValueError) as exc:
            raise BatchOverrideEditorError(str(exc)) from exc
        workflow_value: int | float
        if self.parameter.kind == "int":
            workflow_value = int(self.workflow_value)
        else:
            workflow_value = float(self.workflow_value)
        object.__setattr__(self, "node_id", node_id)
        object.__setattr__(self, "node_label", node_label)
        object.__setattr__(self, "operation_id", operation_id)
        object.__setattr__(self, "workflow_value", workflow_value)

    @property
    def key(self) -> tuple[str, str]:
        return self.node_id, self.parameter.name

    @property
    def label(self) -> str:
        return f"{self.node_label}\n{self.parameter.label}"


class BatchParameterOverrideEditor(QWidget):
    """Matrix editor with blank cells meaning inheritance from the workflow."""

    overridesChanged = Signal()
    validityChanged = Signal(bool)
    sourceSelected = Signal(str, int)
    selectionChanged = Signal(object)

    PAGE_SIZE = 50

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._sources: tuple[BatchOverrideSourceItem, ...] = ()
        self._parameters: tuple[BatchOverrideParameterSpec, ...] = ()
        self._source_keys: tuple[str, ...] = ()
        self._editors: dict[tuple[str, str, str], QLineEdit] = {}
        # Raw text, including temporarily invalid edits, survives all view changes.
        self._values: dict[tuple[str, str, str], str] = {}
        self._checked_keys: set[str] = set()
        self._active_source_key = ""
        self._source_positions: dict[str, int] = {}
        self._visible_parameter_keys: set[tuple[str, str]] = set()
        self._visible_parameters: tuple[BatchOverrideParameterSpec, ...] = ()
        self._node_display_labels: dict[str, str] = {}
        self._page_keys: tuple[str, ...] = ()
        self._matching_keys: tuple[str, ...] = ()
        self._page = 0
        self._contract_error = ""
        self._configured = False
        self._updating = False
        self._status_tone = "neutral"
        self._filter_refresh_timer = QTimer(self)
        self._filter_refresh_timer.setSingleShot(True)
        self._filter_refresh_timer.setInterval(0)
        self._filter_refresh_timer.timeout.connect(
            self._refresh_filtered_rows_after_edit
        )

        self.help_label = QLabel(
            "Enter only values that differ for a sample. Each blank cell shows "
            "and inherits the node's authored workflow value. Overrides are bound "
            "to the exact primary source item. Clear a cell to use the workflow value."
        )
        self.help_label.setWordWrap(True)
        self.help_label.setMinimumWidth(0)
        self.help_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setMinimumWidth(0)
        self.status_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)

        self.sample_search = QLineEdit()
        self.sample_search.setPlaceholderText("Find samples by name or source path…")
        self.sample_search.setAccessibleName("Find samples")
        self.sample_search.setClearButtonEnabled(True)
        self.filter_combo = QComboBox()
        for label, value in (
            ("All samples", "all"),
            ("Show changed only", "changed"),
            ("Using workflow values", "workflow"),
            ("No override for parameter", "missing"),
        ):
            self.filter_combo.addItem(label, value)
        self.filter_combo.setAccessibleName("Show samples")
        self.missing_parameter_combo = QComboBox()
        self.missing_parameter_combo.setAccessibleName("Parameter without an override")
        self.missing_parameter_combo.hide()
        self.sample_search_label = QLabel("Find samples")
        self.sample_search_label.setBuddy(self.sample_search)
        self.filter_label = QLabel("Show samples")
        self.filter_label.setBuddy(self.filter_combo)
        self.missing_parameter_label = QLabel("Parameter without an override")
        self.missing_parameter_label.setBuddy(self.missing_parameter_combo)
        self.missing_parameter_label.hide()
        filters = QGridLayout()
        filters.addWidget(self.sample_search_label, 0, 0)
        filters.addWidget(self.filter_label, 0, 1)
        filters.addWidget(self.missing_parameter_label, 0, 2)
        filters.addWidget(self.sample_search, 1, 0)
        filters.addWidget(self.filter_combo, 1, 1)
        filters.addWidget(self.missing_parameter_combo, 1, 2)
        filters.setColumnStretch(0, 1)

        self.parameter_search = QLineEdit()
        self.parameter_search.setPlaceholderText(
            "Find node or parameter · jump to a column…"
        )
        self.parameter_search.setAccessibleName("Find node or parameter")
        self.parameter_search.setClearButtonEnabled(True)
        self.columns_button = ToolbarCommandButton("Show columns…")
        self.column_summary = QLabel()
        self.parameter_search_label = QLabel("Find node or parameter")
        self.parameter_search_label.setBuddy(self.parameter_search)
        self.parameter_search.setToolTip(
            "Enter a node or parameter name, then press Enter to jump to its column."
        )
        columns = QGridLayout()
        columns.addWidget(self.parameter_search_label, 0, 0)
        columns.addWidget(self.parameter_search, 1, 0)
        columns.addWidget(self.columns_button, 1, 1)
        columns.addWidget(self.column_summary, 1, 2)
        columns.setColumnStretch(0, 1)

        self.selection_label = QLabel("0 selected")
        self.selection_label.setWordWrap(True)
        self.select_page_checkbox = QCheckBox("Select this page")
        self.select_page_checkbox.setAccessibleName("Select this page of samples")
        self.select_page_checkbox.setToolTip(
            "Check or uncheck samples on this page only. "
            "Selections on other pages stay unchanged."
        )
        self.select_matching_button = ToolbarCommandButton("Select all matching")
        self.select_matching_button.setToolTip(
            "Select every sample matching the current filters, across all pages."
        )
        self.clear_selection_button = ToolbarCommandButton("Deselect all")
        self.clear_selection_button.setToolTip(
            "Uncheck all selected samples across all pages, "
            "including filtered-out samples. "
            "Parameter overrides stay unchanged."
        )
        self.reset_selected_button = ToolbarCommandButton("Reset selected…")
        self.reset_selected_button.setToolTip(
            "Restore every parameter override for the checked samples to its "
            "workflow value, including hidden columns and samples on other pages. "
            "Run or bypass settings are not changed."
        )
        self.edit_selected_button = ToolbarCommandButton("Edit selected…")
        self.selection_layout = QGridLayout()
        self.selection_layout.setHorizontalSpacing(8)
        self.selection_layout.setVerticalSpacing(6)
        self._selection_layout_mode = ""
        self._override_command_icons = (
            (self.columns_button, "columns"),
            (self.select_matching_button, "select_all"),
            (self.clear_selection_button, "deselect"),
            (self.reset_selected_button, "reset"),
            (self.edit_selected_button, "edit"),
        )
        self._layout_selection_commands()

        self.table = BatchOverrideTable()
        self.table.setHorizontalHeaderLabels(["Sample / item"])
        self.table.setItemDelegate(BatchOverrideDelegate(self))
        self.table.setEditTriggers(
            QAbstractItemView.DoubleClicked
            | QAbstractItemView.EditKeyPressed
            | QAbstractItemView.AnyKeyPressed
            | QAbstractItemView.SelectedClicked
        )
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setMinimumHeight(130)
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.empty_label = QLabel("Check batch in Setup to discover samples.")
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.previous_page_button = ToolbarCommandButton("Previous")
        self.next_page_button = ToolbarCommandButton("Next")
        self._override_command_icons += (
            (self.previous_page_button, "previous"),
            (self.next_page_button, "next"),
        )
        self.page_label = QLabel("0 samples")
        pagination = QHBoxLayout()
        pagination.addWidget(self.page_label, 1)
        pagination.addWidget(self.previous_page_button)
        pagination.addWidget(self.next_page_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.help_label)
        layout.addLayout(filters)
        layout.addLayout(columns)
        layout.addLayout(self.selection_layout)
        layout.addWidget(self.table, 1)
        layout.addWidget(self.empty_label)
        layout.addLayout(pagination)
        layout.addWidget(self.status_label)
        self.sample_search.textChanged.connect(self._filters_changed)
        self.filter_combo.currentIndexChanged.connect(self._filters_changed)
        self.missing_parameter_combo.currentIndexChanged.connect(self._filters_changed)
        self.columns_button.clicked.connect(self.show_column_chooser)
        self.parameter_search.returnPressed.connect(self._reveal_search_match)
        self.select_page_checkbox.clicked.connect(self._check_page)
        self.select_matching_button.clicked.connect(self.select_all_matching)
        self.clear_selection_button.clicked.connect(self.clear_selection)
        self.reset_selected_button.clicked.connect(
            lambda: self.reset_selected_overrides()
        )
        self.edit_selected_button.clicked.connect(self.edit_selected)
        self.previous_page_button.clicked.connect(lambda: self.set_page(self._page - 1))
        self.next_page_button.clicked.connect(lambda: self.set_page(self._page + 1))
        self.table.itemChanged.connect(self._item_changed)
        self.table.currentCellChanged.connect(self._current_cell_changed)
        self.table.customContextMenuRequested.connect(self._cell_context_menu)
        self._refresh_selection()
        self.previous_page_button.setEnabled(False)
        self.next_page_button.setEnabled(False)
        self.previous_page_button.hide()
        self.next_page_button.hide()
        self._apply_palette_styles()

    def changeEvent(self, event):  # noqa: N802
        super().changeEvent(event)
        if event.type() in (
            QEvent.PaletteChange,
            QEvent.ApplicationPaletteChange,
            QEvent.StyleChange,
            QEvent.FontChange,
        ):
            self._apply_palette_styles()

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self._layout_selection_commands()

    def _layout_selection_commands(self) -> None:
        """Keep complete command labels usable when the editor becomes narrow."""
        if not hasattr(self, "selection_layout"):
            return
        mode = "wide" if self.width() >= 1000 else (
            "medium" if self.width() >= 680 else "narrow"
        )
        if mode == self._selection_layout_mode:
            return
        self._selection_layout_mode = mode
        layout = self.selection_layout
        widgets = (
            self.select_page_checkbox,
            self.selection_label,
            self.select_matching_button,
            self.clear_selection_button,
            self.reset_selected_button,
            self.edit_selected_button,
        )
        for widget in widgets:
            layout.removeWidget(widget)
        for column in range(6):
            layout.setColumnStretch(column, 0)
        layout.addWidget(self.select_page_checkbox, 0, 0)
        layout.addWidget(self.selection_label, 0, 1, 1, 1 if mode == "wide" else 3)
        layout.setColumnStretch(1, 1)
        for index, widget in enumerate(widgets[2:]):
            if mode == "wide":
                layout.addWidget(widget, 0, index + 2)
            elif mode == "medium":
                layout.addWidget(widget, 1, index)
            else:
                layout.addWidget(widget, 1 + index // 2, (index % 2) * 2, 1, 2)

    def _apply_palette_styles(self) -> None:
        if not hasattr(self, "help_label"):
            return
        colors = custom_paint_colors(self.palette())
        for button, kind in getattr(self, "_override_command_icons", ()):
            button.setIcon(toolbar_icon(kind, self.palette()))
            button.setIconSize(QSize(18, 18))
        if hasattr(self, "table"):
            apply_batch_table_style(self.table, self.palette(), font=self.font())
            apply_batch_table_style(
                self.table.frozen_identity, self.palette(), font=self.font()
            )
        self.help_label.setStyleSheet(f"color: {colors.muted_text.name()};")
        if self._status_tone == "error":
            color = "#fca5a5" if palette_is_dark(self.palette()) else "#b91c1c"
        else:
            color = colors.muted_text.name()
        self.status_label.setStyleSheet(f"color: {color};")

    @property
    def configured(self) -> bool:
        return self._configured

    @property
    def error_message(self) -> str:
        return self._contract_error or self._first_cell_error()

    def configure(
        self,
        sources: Sequence[BatchOverrideSourceItem],
        parameters: Sequence[BatchOverrideParameterSpec],
        *,
        overrides: Sequence[BatchSourceParameterOverrides] = (),
    ) -> bool:
        """Replace the complete reviewed contract and populate saved values."""

        prior_checked = set(self._checked_keys)
        prior_active = self._active_source_key
        prior_visible = set(self._visible_parameter_keys)
        prior_parameter_keys = {binding.key for binding in self._parameters}
        self._clear_table()
        self._values.clear()
        self._configured = True
        self._updating = True
        try:
            normalized_sources = tuple(sources)
            normalized_parameters = tuple(parameters)
            self._validate_contract(normalized_sources, normalized_parameters)
            normalized_overrides = normalize_batch_parameter_overrides(tuple(overrides))
            self._sources = normalized_sources
            self._parameters = normalized_parameters
            self._source_keys = tuple(
                batch_source_item_override_key(
                    source.source_node_id,
                    source.source_item,
                )
                for source in self._sources
            )
            self._source_positions = {
                key: index for index, key in enumerate(self._source_keys)
            }
            self._checked_keys = prior_checked.intersection(self._source_keys)
            self._active_source_key = (
                prior_active if prior_active in self._source_positions else ""
            )
            self._visible_parameter_keys = {
                binding.key
                for binding in self._parameters
                if binding.key in prior_visible
                or binding.key not in prior_parameter_keys
            }
            self._update_parameter_catalog()
            self._populate_table(normalized_overrides)
            self._contract_error = ""
        except (TypeError, ValueError) as exc:
            self._updating = False
            self._sources = ()
            self._parameters = ()
            self._source_keys = ()
            self._source_positions.clear()
            self._values.clear()
            self._checked_keys.clear()
            self._clear_table()
            self._contract_error = str(exc)
            self._show_error(self._contract_error)
            self._refresh_selection()
            self.validityChanged.emit(False)
            return False
        self._updating = False
        self._refresh_validation()
        return True

    def clear_contract(self) -> None:
        self._clear_table()
        self._sources = ()
        self._parameters = ()
        self._source_keys = ()
        self._values.clear()
        self._checked_keys.clear()
        self._source_positions.clear()
        self._active_source_key = ""
        self._contract_error = ""
        self._configured = False
        self.status_label.clear()
        self._status_tone = "neutral"
        self._apply_palette_styles()
        self._refresh_selection()
        self.validityChanged.emit(True)

    def mark_saved_overrides_verifying(self, count: int) -> None:
        """Keep saved values quarantined while exact SourceItems are checked."""

        self._clear_table()
        self._sources = ()
        self._parameters = ()
        self._source_keys = ()
        self._values.clear()
        self._checked_keys.clear()
        self._source_positions.clear()
        self._configured = False
        self._contract_error = (
            "Saved per-sample values are still being matched to current exact "
            "source revisions."
        )
        entries = "entry" if int(count) == 1 else "entries"
        self.status_label.setText(
            f"Checking {int(count)} saved source override {entries} against "
            "the current collection..."
        )
        self._status_tone = "working"
        self._apply_palette_styles()
        self._refresh_selection()
        self.validityChanged.emit(False)

    def mark_saved_overrides_pending_review(
        self,
        count: int,
        *,
        reason: str = "",
    ) -> None:
        """Show that saved values need a newly planned SourceItem contract."""

        self._clear_table()
        self._sources = ()
        self._parameters = ()
        self._source_keys = ()
        self._values.clear()
        self._checked_keys.clear()
        self._source_positions.clear()
        self._configured = False
        self._contract_error = str(reason).strip() or (
            f"{int(count)} saved source override entr"
            f"{'y' if int(count) == 1 else 'ies'} must be matched against a fresh "
            "batch preview before VIPP can run them."
        )
        self._show_error(self._contract_error)
        self._refresh_selection()
        self.validityChanged.emit(False)

    def overrides(self) -> tuple[BatchSourceParameterOverrides, ...]:
        """Return the canonical typed mapping, or fail on ambiguous UI state."""

        if not self._configured:
            return ()
        error = self.error_message
        if error:
            raise BatchOverrideEditorError(error)
        result: list[BatchSourceParameterOverrides] = []
        for source_key in self._source_keys:
            values: list[BatchParameterOverride] = []
            for binding in self._parameters:
                text = self._values.get(
                    (source_key, binding.node_id, binding.parameter.name), ""
                ).strip()
                if not text:
                    continue
                values.append(
                    BatchParameterOverride(
                        node_id=binding.node_id,
                        parameter=binding.parameter.name,
                        value=self._parse_value(text, binding),
                    )
                )
            if values:
                result.append(BatchSourceParameterOverrides(source_key, tuple(values)))
        return normalize_batch_parameter_overrides(tuple(result))

    def editor_for(
        self,
        source_node_id: str,
        source_item: SourceItem,
        node_id: str,
        parameter: str,
    ) -> QLineEdit:
        """Reveal a configured cell and create its editor on the current page."""

        key = batch_source_item_override_key(source_node_id, source_item)
        parameter_key = (str(node_id), str(parameter))
        if key not in self._source_positions or parameter_key not in {
            binding.key for binding in self._parameters
        }:
            raise BatchOverrideEditorError(
                "The requested source/node/parameter is not in the current "
                "override contract."
            )
        self.select_source(key)
        self.reveal_parameter(*parameter_key)
        row = self._page_keys.index(key)
        column = next(
            index
            for index, binding in enumerate(self._visible_parameters, 1)
            if binding.key == parameter_key
        )
        item = self.table.item(row, column)
        self.table.openPersistentEditor(item)
        editor = self._editors[(key, *parameter_key)]
        self._first_cell_error()
        return editor

    def _validate_contract(
        self,
        sources: tuple[BatchOverrideSourceItem, ...],
        parameters: tuple[BatchOverrideParameterSpec, ...],
    ) -> None:
        if any(not isinstance(item, BatchOverrideSourceItem) for item in sources):
            raise TypeError(
                "Override sources must contain BatchOverrideSourceItem records."
            )
        if any(not isinstance(item, BatchOverrideParameterSpec) for item in parameters):
            raise TypeError(
                "Override parameters must contain BatchOverrideParameterSpec records."
            )
        labels = [source.label for source in sources]
        if len(labels) != len(set(labels)):
            raise BatchOverrideEditorError(
                "Duplicate sample/item labels make parameter-override rows "
                "ambiguous. Give every planned item a unique label."
            )
        source_keys = [
            batch_source_item_override_key(
                source.source_node_id,
                source.source_item,
            )
            for source in sources
        ]
        if len(source_keys) != len(set(source_keys)):
            raise BatchOverrideEditorError(
                "Duplicate primary SourceItems were found in the planned batch. "
                "VIPP will not guess which row an override belongs to."
            )
        parameter_keys = [parameter.key for parameter in parameters]
        if len(parameter_keys) != len(set(parameter_keys)):
            raise BatchOverrideEditorError(
                "Duplicate node/parameter columns were supplied to the override editor."
            )

    def _populate_table(
        self,
        overrides: tuple[BatchSourceParameterOverrides, ...],
    ) -> None:
        source_key_set = set(self._source_keys)
        parameter_by_key = {binding.key: binding for binding in self._parameters}
        for source_override in overrides:
            if source_override.source_item_key not in source_key_set:
                raise BatchOverrideEditorError(
                    "Saved parameter overrides are stale: at least one exact "
                    "primary SourceItem is no longer in this batch plan. Review "
                    "or remove those overrides before running."
                )
            for override in source_override.values:
                key = (override.node_id, override.parameter)
                binding = parameter_by_key.get(key)
                if binding is None:
                    raise BatchOverrideEditorError(
                        "Saved parameter overrides are stale: workflow parameter "
                        f"{override.node_id}.{override.parameter} is no longer an "
                        "eligible public numeric parameter."
                    )
                self._validated_value(override.value, binding)
                self._values[(source_override.source_item_key, *key)] = (
                    self._format_value(override.value, binding)
                )
        self._render_page()

    def _filtered_keys(self) -> tuple[str, ...]:
        query = self.sample_search.text().strip().casefold()
        mode = self.filter_combo.currentData()
        parameter_key = self.missing_parameter_combo.currentData()
        changed = {key[0] for key, value in self._values.items() if value.strip()}
        matched = []
        for source, source_key in zip(self._sources, self._source_keys, strict=True):
            searchable = " ".join(
                (
                    source.label,
                    source.source_item.container.uri,
                    source.source_item.resolved.name,
                    source.source_item.selector.key,
                )
            ).casefold()
            if query and query not in searchable:
                continue
            if mode == "changed" and source_key not in changed:
                continue
            if mode == "workflow" and source_key in changed:
                continue
            if (
                mode == "missing"
                and parameter_key
                and self._values.get((source_key, *parameter_key), "").strip()
            ):
                continue
            matched.append(source_key)
        return tuple(matched)

    def _render_page(self) -> None:
        updating = self._updating
        self._updating = True
        try:
            self._matching_keys = self._filtered_keys()
            page_count = max(
                1, (len(self._matching_keys) + self.PAGE_SIZE - 1) // self.PAGE_SIZE
            )
            self._page = min(max(0, self._page), page_count - 1)
            start = self._page * self.PAGE_SIZE
            self._page_keys = self._matching_keys[start : start + self.PAGE_SIZE]
            self._visible_parameters = tuple(
                binding
                for binding in self._parameters
                if binding.key in self._visible_parameter_keys
            )
            self._editors.clear()
            self.table.clearContents()
            self.table.setRowCount(len(self._page_keys))
            self.table.setColumnCount(1 + len(self._visible_parameters))
            self.table.setHorizontalHeaderLabels(
                [
                    "Sample / item",
                    *(
                        f"{self._display_node_label(binding)}\n"
                        f"{binding.parameter.label}\nWorkflow: "
                        f"{self._format_value(binding.workflow_value, binding)}"
                        for binding in self._visible_parameters
                    ),
                ]
            )
            self.table.horizontalHeader().setMinimumSectionSize(120)
            self.table.horizontalHeader().setStretchLastSection(False)
            self.table.setColumnWidth(0, 220)
            header_metrics = QFontMetrics(
                self.table.horizontalHeader().line_fonts()[1]
            )
            for column, binding in enumerate(self._visible_parameters, 1):
                header_item = self.table.horizontalHeaderItem(column)
                header_item.setToolTip(header_item.text())
                header_item.setTextAlignment(Qt.AlignCenter)
                label = self._display_node_label(binding)
                header_item.setData(
                    self.table.horizontalHeader().NODE_IDENTITY_ROLE,
                    (
                        binding.node_label,
                        binding.node_id if label != binding.node_label else "",
                    ),
                )
                width = max(
                    160, header_metrics.horizontalAdvance(binding.parameter.label) + 24
                )
                if label != binding.node_label:
                    # IDs have their own regular-weight line. Long authored IDs
                    # must not make a 30-node matrix impractically wide; middle
                    # elision preserves suffixes and the tooltip retains all text.
                    width = max(
                        width, header_metrics.horizontalAdvance(binding.node_id) + 24
                    )
                self.table.setColumnWidth(column, min(240, width))
            for row, source_key in enumerate(self._page_keys):
                source = self._sources[self._source_positions[source_key]]
                label_item = QTableWidgetItem(source.label)
                label_item.setToolTip(
                    f"Primary source node: {source.source_node_id}\n"
                    f"Source: {source.source_item.container.uri}\n"
                    f"Source item: {source.source_item.resolved.name}\n"
                    f"Selector: {source.source_item.selector.key}\n"
                    f"Exact override key: {source_key}"
                )
                label_item.setFlags(
                    (label_item.flags() & ~Qt.ItemIsEditable) | Qt.ItemIsUserCheckable
                )
                label_item.setCheckState(
                    Qt.Checked if source_key in self._checked_keys else Qt.Unchecked
                )
                self.table.setItem(row, 0, label_item)
                for column, binding in enumerate(self._visible_parameters, 1):
                    key = (source_key, binding.node_id, binding.parameter.name)
                    item = QTableWidgetItem(self._values.get(key, ""))
                    item.setTextAlignment(Qt.AlignCenter)
                    item.setToolTip(
                        f"{source.label} · {self._display_node_label(binding)} / "
                        f"{binding.parameter.label}\n"
                        f"Blank inherits the authored workflow value "
                        f"{self._format_value(binding.workflow_value, binding)}.\n"
                        "Double-click to edit. Clear the cell to use workflow value."
                    )
                    self.table.setItem(row, column, item)
                self.table.setRowHeight(row, 32)
            if self._active_source_key in self._page_keys:
                self.table.setCurrentCell(
                    self._page_keys.index(self._active_source_key), 0
                )
            self.table.sync_frozen_identity()
            count = len(self._matching_keys)
            self.page_label.setText(
                f"{start + 1:,}–{start + len(self._page_keys):,} of {count:,} "
                "samples · 50 per page"
                if count
                else "0 samples"
            )
            self.previous_page_button.setEnabled(self._page > 0)
            self.next_page_button.setEnabled(self._page + 1 < page_count)
            self.previous_page_button.setVisible(page_count > 1)
            self.next_page_button.setVisible(page_count > 1)
            self.empty_label.setVisible(not self._page_keys)
            self.empty_label.setText(
                "No samples match these filters."
                if self._sources
                else "Check batch in Setup to discover samples."
            )
            node_count = len({binding.node_id for binding in self._visible_parameters})
            self.column_summary.setText(
                f"{len(self._visible_parameters)} of {len(self._parameters)} "
                f"parameters · {node_count} nodes"
            )
            self.missing_parameter_combo.setVisible(
                self.filter_combo.currentData() == "missing"
            )
            self.missing_parameter_label.setVisible(
                self.filter_combo.currentData() == "missing"
            )
        finally:
            self._updating = updating
        self._refresh_selection()

    def _filters_changed(self, *_args) -> None:
        if not self._updating:
            self._page = 0
            self._render_page()

    def _refresh_filtered_rows_after_edit(self) -> None:
        if not self._updating and self._filtered_keys() != self._matching_keys:
            self._render_page()

    @property
    def current_page(self) -> int:
        """Current zero-based page within the filtered sample list."""
        return self._page

    def set_page(self, page: int) -> None:
        self._page = int(page)
        self._render_page()

    def selected_source_keys(self) -> tuple[str, ...]:
        """Checked samples in contract order, including hidden samples."""
        return tuple(key for key in self._source_keys if key in self._checked_keys)

    def set_selected_source_keys(self, keys: Sequence[str]) -> None:
        """Replace checked exact-source keys atomically with one view refresh.

        Selection includes filtered and off-page samples. This does not navigate,
        edit overrides, or change the active sample. Unknown keys fail before the
        existing selection is touched; duplicates are harmless.
        """
        if isinstance(keys, str):
            raise BatchOverrideEditorError(
                "Selected samples must be a sequence of exact source keys."
            )
        selected = set(keys)
        if not selected.issubset(self._source_positions):
            raise BatchOverrideEditorError(
                "Selected samples are no longer in this exact source contract."
            )
        self._checked_keys = selected
        self._render_page()
        self.selectionChanged.emit(self.selected_source_keys())

    def select_source(
        self,
        source_key: str | None = None,
        *,
        position: int | None = None,
        checked: bool | None = None,
        clear_selection: bool = False,
    ) -> bool:
        """Reveal a primary source by exact key or zero-based contract position."""
        if (
            source_key is None
            and position is not None
            and 0 <= position < len(self._source_keys)
        ):
            source_key = self._source_keys[position]
        if source_key not in self._source_positions:
            return False
        if source_key not in self._filtered_keys():
            self.sample_search.blockSignals(True)
            self.filter_combo.blockSignals(True)
            self.sample_search.clear()
            self.filter_combo.setCurrentIndex(0)
            self.sample_search.blockSignals(False)
            self.filter_combo.blockSignals(False)
        matching = self._filtered_keys()
        page = matching.index(source_key) // self.PAGE_SIZE
        self._active_source_key = source_key
        if clear_selection:
            self._checked_keys.clear()
        if checked is True:
            self._checked_keys.add(source_key)
        elif checked is False:
            self._checked_keys.discard(source_key)
        if (
            page != self._page
            or source_key not in self._page_keys
            or checked is not None
            or clear_selection
        ):
            self._page = page
            self._render_page()
        else:
            self.table.setCurrentCell(self._page_keys.index(source_key), 0)
        self.table.scrollToItem(self.table.item(self._page_keys.index(source_key), 0))
        self.sourceSelected.emit(source_key, self._source_positions[source_key])
        if checked is not None or clear_selection:
            self.selectionChanged.emit(self.selected_source_keys())
        return True

    def _current_cell_changed(self, row, _column, _previous_row, _previous_column):
        if self._updating or not 0 <= row < len(self._page_keys):
            return
        key = self._page_keys[row]
        self._active_source_key = key
        self.sourceSelected.emit(key, self._source_positions[key])

    def _item_changed(self, item):
        if self._updating or not 0 <= item.row() < len(self._page_keys):
            return
        source_key = self._page_keys[item.row()]
        if item.column() == 0:
            if item.checkState() == Qt.Checked:
                self._checked_keys.add(source_key)
            else:
                self._checked_keys.discard(source_key)
            self._refresh_selection()
            self.selectionChanged.emit(self.selected_source_keys())
        else:
            binding = self._visible_parameters[item.column() - 1]
            self._cell_changed(
                (source_key, binding.node_id, binding.parameter.name), item.text()
            )

    def _refresh_selection(self):
        if not hasattr(self, "selection_label"):
            return
        hidden = len(self._checked_keys.difference(self._matching_keys))
        elsewhere = len(self._checked_keys.difference(self._page_keys)) - hidden
        summary = f"{len(self._checked_keys):,} selected"
        if hidden:
            summary += f" · {hidden:,} hidden by filters (still included)"
        if elsewhere:
            summary += f" · {elsewhere:,} on other pages"
        self.selection_label.setText(summary)
        page_checked = len(self._checked_keys.intersection(self._page_keys))
        self.select_page_checkbox.blockSignals(True)
        self.select_page_checkbox.setCheckState(
            Qt.Checked
            if self._page_keys and page_checked == len(self._page_keys)
            else Qt.PartiallyChecked
            if page_checked
            else Qt.Unchecked
        )
        self.select_page_checkbox.blockSignals(False)
        self.select_page_checkbox.setEnabled(bool(self._page_keys))
        self.select_matching_button.setEnabled(bool(self._matching_keys))
        self.clear_selection_button.setEnabled(bool(self._checked_keys))
        self.reset_selected_button.setEnabled(
            bool(self._checked_keys)
            and bool(self.override_value_count(self._checked_keys))
        )
        self.edit_selected_button.setEnabled(
            bool(self._checked_keys and self._parameters) and not self.error_message
        )

    def _check_page(self, checked):
        if checked:
            self._checked_keys.update(self._page_keys)
        else:
            self._checked_keys.difference_update(self._page_keys)
        self._render_page()
        self.selectionChanged.emit(self.selected_source_keys())

    def select_all_matching(self):
        self._checked_keys.update(self._filtered_keys())
        self._render_page()
        self.selectionChanged.emit(self.selected_source_keys())

    def clear_selection(self):
        """Only uncheck samples; never modify any parameter values."""
        self._checked_keys.clear()
        self._render_page()
        self.selectionChanged.emit(())

    def override_value_count(self, source_keys: Sequence[str] | None = None) -> int:
        """Count nonblank stored cells, including invalid edits and hidden columns."""
        targets = None if source_keys is None else set(source_keys)
        return sum(
            bool(text.strip())
            for key, text in self._values.items()
            if targets is None or key[0] in targets
        )

    def reset_selected_overrides(self, *, confirm: bool = True) -> bool:
        """Restore every parameter for checked samples, preserving their checks.

        The scope is exact selected source keys, not merely visible rows or
        columns. Invalid raw edits can also be discarded to restore inheritance.
        Run/bypass overrides belong to the batch, not to a selected sample, and
        are deliberately outside this editor's reset scope.
        """
        targets = self.selected_source_keys()
        count = self.override_value_count(targets)
        if not count:
            return False
        if confirm and not self._confirm_override_reset(
            count, len(targets), all_=False
        ):
            return False
        return self._reset_override_values(set(targets))

    def reset_all_overrides(self, *, confirm: bool = True) -> bool:
        """Restore all per-sample cells; the caller owns batch-wide node settings.

        ``confirm=False`` allows a parent action to confirm a combined reset of
        sample parameters and node settings once. Scientific contract errors are
        preserved: discarding values must not make an unchecked contract valid.
        """
        count = self.override_value_count()
        if not count:
            return False
        if confirm and not self._confirm_override_reset(
            count, len(self._source_keys), all_=True
        ):
            return False
        return self._reset_override_values(None)

    def _confirm_override_reset(self, count: int, samples: int, *, all_: bool) -> bool:
        scope = "all samples" if all_ else f"{samples:,} selected samples"
        noun = "override" if count == 1 else "overrides"
        answer = QMessageBox.question(
            self,
            "Reset all parameter overrides?" if all_ else "Reset selected overrides?",
            f"Reset {count:,} parameter {noun} for {scope} "
            "to their workflow values?\n\n"
            "This includes hidden columns and samples on other pages. "
            "Run or bypass settings and the original workflow stay unchanged. "
            "Your sample selection will be kept.",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        return answer == QMessageBox.Yes

    def _reset_override_values(self, targets: set[str] | None) -> bool:
        keys = tuple(
            key for key in self._values if targets is None or key[0] in targets
        )
        if not keys:
            return False
        for key in keys:
            self._values.pop(key, None)
        self._filter_refresh_timer.stop()
        self._render_page()
        self._refresh_validation()
        self.overridesChanged.emit()
        return True

    def _display_node_label(self, binding: BatchOverrideParameterSpec) -> str:
        return self._node_display_labels.get(binding.node_id, binding.node_label)

    def _parameter_title(self, binding: BatchOverrideParameterSpec) -> str:
        return f"{self._display_node_label(binding)} / {binding.parameter.label}"

    def _parameter_search_text(self, binding: BatchOverrideParameterSpec) -> str:
        return (
            f"{self._parameter_title(binding)} {binding.node_id} "
            f"{binding.parameter.name}"
        ).casefold()

    def _update_parameter_catalog(self):
        nodes_by_label: dict[str, set[str]] = {}
        for binding in self._parameters:
            nodes_by_label.setdefault(binding.node_label, set()).add(binding.node_id)
        self._node_display_labels = {
            binding.node_id: (
                f"{binding.node_label} [{binding.node_id}]"
                if len(nodes_by_label[binding.node_label]) > 1
                else binding.node_label
            )
            for binding in self._parameters
        }
        previous = self.missing_parameter_combo.currentData()
        self.missing_parameter_combo.clear()
        self._parameter_search_keys = {}
        for binding in self._parameters:
            title = self._parameter_title(binding)
            self.missing_parameter_combo.addItem(title, binding.key)
            self._parameter_search_keys[title] = binding.key
        index = self.missing_parameter_combo.findData(previous)
        if index >= 0:
            self.missing_parameter_combo.setCurrentIndex(index)
        completer = QCompleter(list(self._parameter_search_keys), self.parameter_search)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        completer.setFilterMode(Qt.MatchContains)
        completer.activated[str].connect(self._reveal_search_match)
        self.parameter_search.setCompleter(completer)

    def _reveal_search_match(self, title=None):
        query = (
            title if isinstance(title, str) else self.parameter_search.text()
        ).strip()
        match = next(
            (
                binding
                for binding in self._parameters
                if query.casefold() in self._parameter_search_text(binding)
            ),
            None,
        )
        if match is not None:
            self.reveal_parameter(*match.key)

    def reveal_parameter(self, node_id: str, parameter: str) -> bool:
        """Show a hidden parameter and horizontally reveal its column."""
        key = (str(node_id), str(parameter))
        binding = next((item for item in self._parameters if item.key == key), None)
        if binding is None:
            return False
        if key not in self._visible_parameter_keys:
            self._visible_parameter_keys.add(key)
            self._render_page()
        column = self._visible_parameters.index(binding) + 1
        offset = self.table.horizontalHeader().sectionPosition(column)
        self.table.horizontalScrollBar().setValue(
            max(0, offset - self.table.columnWidth(0))
        )
        self.table.horizontalHeaderItem(column).setToolTip(
            f"Showing {self._parameter_title(binding)}"
        )
        return True

    def set_visible_parameters(self, keys: Sequence[tuple[str, str]]) -> None:
        """Choose display columns without changing any source override values."""
        known = {binding.key for binding in self._parameters}
        requested = {tuple(key) for key in keys}
        if not requested.issubset(known):
            raise BatchOverrideEditorError("An unknown override column was requested.")
        self._visible_parameter_keys = requested
        self._render_page()

    def _parameter_tree(self, parent, *, checked=()):
        tree = QTreeWidget(parent)
        tree.setHeaderHidden(True)
        tree.setAccessibleName("Parameters grouped by workflow node")
        groups = {}
        leaves = {}
        checked = set(checked)
        for binding in self._parameters:
            group = groups.get(binding.node_id)
            if group is None:
                group = QTreeWidgetItem(tree, [self._display_node_label(binding)])
                group.setFlags(
                    group.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsAutoTristate
                )
                group.setCheckState(0, Qt.Unchecked)
                groups[binding.node_id] = group
            item = QTreeWidgetItem(
                group,
                [
                    f"{binding.parameter.label} · workflow "
                    f"{self._format_value(binding.workflow_value, binding)}"
                ],
            )
            item.setData(0, Qt.UserRole, binding.key)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(
                0, Qt.Checked if binding.key in checked else Qt.Unchecked
            )
            leaves[binding.key] = item
        tree.expandAll()
        return tree, leaves

    def _filter_parameter_tree(self, leaves, query):
        query = query.strip().casefold()
        bindings = {binding.key: binding for binding in self._parameters}
        for key, item in leaves.items():
            binding = bindings[key]
            item.setHidden(query not in self._parameter_search_text(binding))
        for item in leaves.values():
            group = item.parent()
            group.setHidden(
                all(
                    group.child(index).isHidden() for index in range(group.childCount())
                )
            )

    def create_column_chooser(self) -> QDialog:
        dialog = QDialog(self)
        dialog.setWindowTitle("Show parameter columns")
        dialog.resize(520, 520)
        layout = QVBoxLayout(dialog)
        search = QLineEdit()
        search.setPlaceholderText("Find a node or parameter…")
        search.setAccessibleName("Find columns")
        layout.addWidget(search)
        tree, leaves = self._parameter_tree(
            dialog, checked=self._visible_parameter_keys
        )
        dialog.parameter_tree = tree
        dialog.parameter_items = leaves
        layout.addWidget(tree, 1)
        controls = QHBoxLayout()
        for title, state, kind in (
            ("Show all", Qt.Checked, "select_all"),
            ("Hide all", Qt.Unchecked, "deselect"),
        ):
            button = ToolbarCommandButton(title)
            button.setIcon(toolbar_icon(kind, self.palette()))
            button.setIconSize(QSize(18, 18))
            button.clicked.connect(
                lambda _checked=False, state=state: [
                    item.setCheckState(0, state) for item in leaves.values()
                ]
            )
            controls.addWidget(button)
        controls.addStretch()
        layout.addLayout(controls)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        search.textChanged.connect(
            lambda query: self._filter_parameter_tree(leaves, query)
        )
        dialog.accepted.connect(
            lambda: self.set_visible_parameters(
                [
                    key
                    for key, item in leaves.items()
                    if item.checkState(0) == Qt.Checked
                ]
            )
        )
        return dialog

    def show_column_chooser(self):
        dialog = self.create_column_chooser()
        dialog.exec()
        dialog.deleteLater()

    def _cell_context_menu(self, point):
        item = self.table.itemAt(point)
        if item is None or item.column() == 0:
            return
        key = self._page_keys[item.row()]
        binding = self._visible_parameters[item.column() - 1]
        menu = QMenu(self)
        action = menu.addAction("Use workflow value")
        action.setEnabled(bool(self._values.get((key, *binding.key), "").strip()))
        if menu.exec(self.table.viewport().mapToGlobal(point)) is action:
            self._cell_changed((key, *binding.key), "")
            self._render_page()

    def apply_selected_values(
        self,
        values: dict[tuple[str, str], int | float | None],
        *,
        source_keys: Sequence[str] | None = None,
    ) -> None:
        """Atomically update only chosen parameters; None restores inheritance."""
        if not self._configured or self.error_message:
            raise BatchOverrideEditorError(
                self.error_message or "Check batch before editing overrides."
            )
        targets = (
            self.selected_source_keys() if source_keys is None else tuple(source_keys)
        )
        if not set(targets).issubset(self._source_positions):
            raise BatchOverrideEditorError(
                "Selected samples are no longer in this exact source contract."
            )
        bindings = {binding.key: binding for binding in self._parameters}
        normalized = {}
        for key, value in values.items():
            if key not in bindings:
                raise BatchOverrideEditorError(
                    "An unknown override parameter was requested."
                )
            normalized[key] = (
                None
                if value is None
                else self._format_value(
                    self._validated_value(value, bindings[key]), bindings[key]
                )
            )
        changed = False
        for source_key in targets:
            for key, text in normalized.items():
                cell_key = (source_key, *key)
                previous = self._values.get(cell_key)
                if text is None:
                    self._values.pop(cell_key, None)
                else:
                    self._values[cell_key] = text
                changed |= previous != text
        self._render_page()
        self._refresh_validation()
        if changed:
            self.overridesChanged.emit()

    def create_edit_selected_dialog(self) -> QDialog:
        """Create a cancellable draft; opening never changes overrides."""
        from napari_vipp.ui.batch_bulk_edit import BatchBulkEditDialog

        return BatchBulkEditDialog(self)

    def edit_selected(self):
        if not self.selected_source_keys() or self.error_message:
            return
        dialog = self.create_edit_selected_dialog()
        dialog.exec()
        dialog.deleteLater()

    def _make_editor(self, binding: BatchOverrideParameterSpec) -> QLineEdit:
        spec = binding.parameter
        workflow_value = self._format_value(binding.workflow_value, binding)
        editor = QLineEdit()
        editor.setAlignment(Qt.AlignCenter)
        editor.setPlaceholderText(f"inherit {workflow_value}")
        # The typed serialization contract uses a decimal point, independently of
        # the desktop locale. Do not let Qt silently rewrite it to a comma.
        numeric_locale = QLocale.c()
        numeric_locale.setNumberOptions(QLocale.RejectGroupSeparator)
        if spec.data_dependent_bounds:
            editor.setToolTip(
                f"Blank inherits the authored workflow value {workflow_value}. Enter a "
                "finite value on the connected image's intensity scale; the "
                "valid working range depends on that sample."
            )
        else:
            editor.setToolTip(
                f"Blank inherits the authored workflow value {workflow_value}. "
                "Accepted "
                f"range: {spec.minimum!r} to {spec.maximum!r}."
            )
        if (
            not spec.data_dependent_bounds
            and spec.kind == "int"
            and all(
                isinstance(value, Integral) and not isinstance(value, bool)
                for value in (spec.minimum, spec.maximum)
            )
        ):
            minimum = int(spec.minimum)
            maximum = int(spec.maximum)
            if -(2**31) <= minimum <= maximum <= 2**31 - 1:
                validator = QIntValidator(minimum, maximum, editor)
                validator.setLocale(numeric_locale)
                editor.setValidator(validator)
        elif not spec.data_dependent_bounds and spec.kind == "float":
            validator = QDoubleValidator(
                float(spec.minimum),
                float(spec.maximum),
                max(0, int(spec.decimals)),
                editor,
            )
            validator.setLocale(numeric_locale)
            validator.setNotation(QDoubleValidator.ScientificNotation)
            editor.setValidator(validator)
        return editor

    def _cell_changed(self, key: tuple[str, str, str], text: str) -> None:
        if self._updating:
            return
        previous = self._values.get(key, "")
        if text.strip():
            self._values[key] = text
        else:
            self._values.pop(key, None)
        self._refresh_validation()
        if previous != self._values.get(key, ""):
            self.overridesChanged.emit()

    def _refresh_validation(self) -> None:
        if self._contract_error:
            self._show_error(self._contract_error)
            self.validityChanged.emit(False)
            return
        error = self._first_cell_error()
        if error:
            self._show_error(error)
            valid = False
        else:
            self.status_label.setText(
                "Ready. Blank cells use the shown workflow values; entered "
                "exceptions are saved by exact primary SourceItem."
            )
            self._status_tone = "neutral"
            self._apply_palette_styles()
            valid = True
        self.validityChanged.emit(valid)
        self._refresh_selection()

    def _first_cell_error(self) -> str:
        if not self._configured:
            return ""
        parameters = {binding.key: binding for binding in self._parameters}
        first_error = ""
        for editor in self._editors.values():
            editor.setStyleSheet("")
        for key, text in self._values.items():
            error = ""
            try:
                self._parse_value(text.strip(), parameters[key[1:]])
            except (TypeError, ValueError) as exc:
                error = str(exc)
            if error:
                editor = self._editors.get(key)
                if editor is not None:
                    editor.setStyleSheet("QLineEdit { border: 1px solid #ef4444; }")
                if not first_error:
                    source = self._sources[self._source_positions[key[0]]]
                    first_error = f"{source.label}: {error}"
        return first_error

    def _parse_value(
        self,
        text: str,
        binding: BatchOverrideParameterSpec,
    ) -> int | float:
        spec = binding.parameter
        label = f"{binding.node_label} / {spec.label}"
        try:
            if spec.kind == "int":
                value: int | float = int(text, 10)
            else:
                value = float(text)
        except ValueError as exc:
            kind = "whole number" if spec.kind == "int" else "number"
            raise BatchOverrideEditorError(f"{label} must be a {kind}.") from exc
        return self._validated_value(value, binding)

    @staticmethod
    def _validated_value(
        value: int | float,
        binding: BatchOverrideParameterSpec,
    ) -> int | float:
        spec = binding.parameter
        label = f"{binding.node_label} / {spec.label}"
        if isinstance(value, bool):
            raise BatchOverrideEditorError(f"{label} must be numeric, not Boolean.")
        if spec.kind == "int":
            if not isinstance(value, Integral):
                raise BatchOverrideEditorError(f"{label} must be a whole number.")
            normalized: int | float = int(value)
        else:
            normalized = float(value)
            if not math.isfinite(normalized):
                raise BatchOverrideEditorError(f"{label} must be finite.")
        try:
            validate_parameter_value(
                spec,
                normalized,
                context="Batch per-sample override",
            )
        except (TypeError, ValueError) as exc:
            raise BatchOverrideEditorError(str(exc)) from exc
        if (
            not spec.data_dependent_bounds
            and not spec.minimum <= normalized <= spec.maximum
        ):
            raise BatchOverrideEditorError(
                f"{label} must be between {spec.minimum!r} and {spec.maximum!r}."
            )
        return normalized

    @staticmethod
    def _format_value(
        value: int | float,
        binding: BatchOverrideParameterSpec,
    ) -> str:
        if binding.parameter.kind == "int":
            return str(int(value))
        return format(float(value), ".17g")

    def _clear_table(self) -> None:
        updating = self._updating
        self._updating = True
        self._editors.clear()
        self.table.clear()
        self.table.setColumnCount(1)
        self.table.setRowCount(0)
        self.table.setHorizontalHeaderLabels(["Sample / item"])
        self._page_keys = ()
        self._matching_keys = ()
        self._page = 0
        self.page_label.setText("0 samples")
        self.previous_page_button.setEnabled(False)
        self.next_page_button.setEnabled(False)
        self.previous_page_button.hide()
        self.next_page_button.hide()
        self.empty_label.show()
        self.table.sync_frozen_identity()
        self._updating = updating

    def _show_error(self, message: str) -> None:
        self.status_label.setText(f"Needs attention: {message}")
        self._status_tone = "error"
        self._apply_palette_styles()


__all__ = [
    "BatchOverrideEditorError",
    "BatchOverrideParameterSpec",
    "BatchOverrideSourceItem",
    "BatchParameterOverrideEditor",
]
