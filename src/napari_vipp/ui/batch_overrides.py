"""Fail-closed editor for per-source numeric batch parameter overrides."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QDoubleValidator, QIntValidator
from qtpy.QtWidgets import (
    QAbstractItemView,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
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

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._sources: tuple[BatchOverrideSourceItem, ...] = ()
        self._parameters: tuple[BatchOverrideParameterSpec, ...] = ()
        self._source_keys: tuple[str, ...] = ()
        self._editors: dict[tuple[str, str, str], QLineEdit] = {}
        self._contract_error = ""
        self._configured = False
        self._updating = False

        self.help_label = QLabel(
            "Enter only values that differ for a sample. Each blank cell shows "
            "and inherits the node's authored workflow value. Overrides are bound "
            "to the exact primary source item."
        )
        self.help_label.setWordWrap(True)
        self.help_label.setMinimumWidth(0)
        self.help_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.help_label.setStyleSheet("color: #94a3b8;")

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setMinimumWidth(0)
        self.status_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)

        self.table = QTableWidget(0, 1)
        self.table.setHorizontalHeaderLabels(["Sample / item"])
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setMinimumHeight(130)
        self.table.setMaximumHeight(300)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.help_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.table)

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

        self._clear_table()
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
            self._populate_table(normalized_overrides)
            self._contract_error = ""
        except (TypeError, ValueError) as exc:
            self._updating = False
            self._sources = ()
            self._parameters = ()
            self._source_keys = ()
            self._contract_error = str(exc)
            self._show_error(self._contract_error)
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
        self._contract_error = ""
        self._configured = False
        self.status_label.clear()
        self.validityChanged.emit(True)

    def mark_saved_overrides_verifying(self, count: int) -> None:
        """Keep saved values quarantined while exact SourceItems are checked."""

        self._clear_table()
        self._sources = ()
        self._parameters = ()
        self._source_keys = ()
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
        self.status_label.setStyleSheet("color: #94a3b8;")
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
        self._configured = False
        self._contract_error = str(reason).strip() or (
            f"{int(count)} saved source override entr"
            f"{'y' if int(count) == 1 else 'ies'} must be matched against a fresh "
            "batch preview before VIPP can run them."
        )
        self._show_error(self._contract_error)
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
                editor = self._editors[
                    (source_key, binding.node_id, binding.parameter.name)
                ]
                text = editor.text().strip()
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
        """Return one configured cell editor for integration tests/controllers."""

        key = batch_source_item_override_key(source_node_id, source_item)
        try:
            return self._editors[(key, str(node_id), str(parameter))]
        except KeyError as exc:
            raise BatchOverrideEditorError(
                "The requested source/node/parameter is not in the current "
                "override contract."
            ) from exc

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
        override_by_source: dict[str, dict[tuple[str, str], int | float]] = {}
        for source_override in overrides:
            if source_override.source_item_key not in source_key_set:
                raise BatchOverrideEditorError(
                    "Saved parameter overrides are stale: at least one exact "
                    "primary SourceItem is no longer in this batch plan. Review "
                    "or remove those overrides before running."
                )
            values: dict[tuple[str, str], int | float] = {}
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
                values[key] = override.value
            override_by_source[source_override.source_item_key] = values

        self.table.setColumnCount(1 + len(self._parameters))
        self.table.setRowCount(len(self._sources))
        self.table.setHorizontalHeaderLabels(
            ["Sample / item", *(binding.label for binding in self._parameters)]
        )
        header = self.table.horizontalHeader()
        header.setStretchLastSection(bool(self._parameters))
        for row_index, (source, source_key) in enumerate(
            zip(self._sources, self._source_keys, strict=True)
        ):
            label_item = QTableWidgetItem(source.label)
            label_item.setToolTip(
                f"Primary source node: {source.source_node_id}\n"
                f"Source item: {source.source_item.resolved.name}\n"
                f"Selector: {source.source_item.selector.key}\n"
                f"Exact override key: {source_key}"
            )
            label_item.setFlags(label_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row_index, 0, label_item)
            existing = override_by_source.get(source_key, {})
            for parameter_index, binding in enumerate(self._parameters, start=1):
                editor = self._make_editor(binding)
                value = existing.get(binding.key)
                if value is not None:
                    editor.setText(self._format_value(value, binding))
                self._editors[(source_key, binding.node_id, binding.parameter.name)] = (
                    editor
                )
                self.table.setCellWidget(row_index, parameter_index, editor)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setMinimumSectionSize(120)

    def _make_editor(self, binding: BatchOverrideParameterSpec) -> QLineEdit:
        spec = binding.parameter
        workflow_value = self._format_value(binding.workflow_value, binding)
        editor = QLineEdit()
        editor.setPlaceholderText(f"inherit {workflow_value}")
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
                editor.setValidator(QIntValidator(minimum, maximum, editor))
        elif not spec.data_dependent_bounds and spec.kind == "float":
            validator = QDoubleValidator(
                float(spec.minimum),
                float(spec.maximum),
                max(0, int(spec.decimals)),
                editor,
            )
            validator.setNotation(QDoubleValidator.ScientificNotation)
            editor.setValidator(validator)
        editor.textChanged.connect(self._cell_changed)
        return editor

    def _cell_changed(self, _text: str) -> None:
        if self._updating:
            return
        self._refresh_validation()
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
            self.status_label.setStyleSheet("color: #94a3b8;")
            valid = True
        self.validityChanged.emit(valid)

    def _first_cell_error(self) -> str:
        if not self._configured:
            return ""
        for source, source_key in zip(self._sources, self._source_keys, strict=True):
            for binding in self._parameters:
                editor = self._editors.get(
                    (source_key, binding.node_id, binding.parameter.name)
                )
                if editor is None:
                    continue
                text = editor.text().strip()
                error = ""
                if text:
                    try:
                        self._parse_value(text, binding)
                    except (TypeError, ValueError) as exc:
                        error = str(exc)
                editor.setStyleSheet(
                    "QLineEdit { border: 1px solid #ef4444; }" if error else ""
                )
                if error:
                    return f"{source.label}: {error}"
        return ""

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
        self._editors.clear()
        self.table.clear()
        self.table.setColumnCount(1)
        self.table.setRowCount(0)
        self.table.setHorizontalHeaderLabels(["Sample / item"])

    def _show_error(self, message: str) -> None:
        self.status_label.setText(f"Needs attention: {message}")
        self.status_label.setStyleSheet("color: #fca5a5;")


__all__ = [
    "BatchOverrideEditorError",
    "BatchOverrideParameterSpec",
    "BatchOverrideSourceItem",
    "BatchParameterOverrideEditor",
]
