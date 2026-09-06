"""Compact, read-only summaries of graph-connected node inputs."""

from __future__ import annotations

import html
from dataclasses import dataclass

import numpy as np
from qtpy.QtCore import QEvent, Qt
from qtpy.QtGui import QFont, QPalette
from qtpy.QtWidgets import (
    QApplication,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from napari_vipp.ui.iconography import data_type_icon
from napari_vipp.ui.palette_roles import theme_colors

_PHYSICAL_UNIT_LABELS = {
    "micrometer": "µm",
    "micrometre": "µm",
    "micrometers": "µm",
    "micrometres": "µm",
    "um": "µm",
    "nanometer": "nm",
    "nanometre": "nm",
    "nanometers": "nm",
    "nanometres": "nm",
    "millimeter": "mm",
    "millimetre": "mm",
    "millimeters": "mm",
    "millimetres": "mm",
}


def _dtype_bit_depth(dtype) -> str:
    try:
        normalized = np.dtype(dtype)
    except (TypeError, ValueError):
        return ""
    if normalized == np.dtype(bool):
        return "1-bit logical"
    if np.issubdtype(normalized, np.integer):
        return f"{np.iinfo(normalized).bits}-bit integer"
    if np.issubdtype(normalized, np.floating):
        return f"{normalized.itemsize * 8}-bit float"
    return f"{normalized.itemsize * 8}-bit"


def _physical_sampling_summary(state) -> str:
    axes = tuple(getattr(state, "axes", ()) or ())
    values: list[str] = []
    for axis in axes:
        if str(getattr(axis, "type", "")).casefold() != "space":
            continue
        unit = str(getattr(axis, "unit", "") or "").strip()
        if unit.casefold() in {"", "pixel", "pixels", "px"}:
            continue
        display_unit = _PHYSICAL_UNIT_LABELS.get(unit.casefold(), unit)
        label = str(
            getattr(axis, "short_label", "")
            or getattr(axis, "name", "axis")
        )
        try:
            scale = f"{float(getattr(axis, 'scale', 1.0)):.6g}"
        except (TypeError, ValueError):
            continue
        values.append(f"{label} {scale} {display_unit}")
    return f"sampling {', '.join(values)}" if values else ""


def connected_input_scientific_summary(data, state) -> str:
    """Return lazy-safe shape/type/sampling context for one graph binding."""

    table_source = state if state is not None else data
    row_count = getattr(table_source, "row_count", None)
    column_count = getattr(table_source, "column_count", None)
    if row_count is not None and column_count is not None:
        parts = [f"{int(row_count):,} rows", f"{int(column_count):,} fields"]
        table_kind = str(
            getattr(table_source, "table_kind", "")
            or getattr(table_source, "kind", "")
        ).strip()
        if table_kind and table_kind.casefold() != "table":
            parts.append(table_kind)
        return " · ".join(parts)

    shape_source = state if getattr(state, "shape", None) is not None else data
    raw_shape = getattr(shape_source, "shape", ()) if shape_source is not None else ()
    try:
        shape = tuple(int(size) for size in raw_shape)
    except (TypeError, ValueError):
        shape = ()
    dtype = getattr(state, "dtype", "") or getattr(data, "dtype", "")
    try:
        dtype_name = np.dtype(dtype).name if dtype else ""
    except (TypeError, ValueError):
        dtype_name = str(dtype or "")
    bit_depth = str(getattr(state, "bit_depth", "") or "")
    if not bit_depth and dtype_name:
        bit_depth = _dtype_bit_depth(dtype_name)

    parts: list[str] = []
    if shape:
        axis_order = str(getattr(state, "axis_order", "") or "")
        dimensions = " × ".join(str(size) for size in shape)
        parts.append(
            f"{axis_order}: {dimensions}"
            if axis_order and axis_order != "scalar"
            else f"Shape: {dimensions}"
        )
    if dtype_name:
        parts.append(
            f"{dtype_name} ({bit_depth})" if bit_depth else dtype_name
        )
    sampling = _physical_sampling_summary(state)
    if sampling:
        parts.append(sampling)
    if getattr(state, "axes_explicit", True) is False:
        parts.append("axes inferred")
    return " · ".join(parts)


@dataclass(frozen=True, slots=True)
class ConnectedInputBinding:
    """Presentation data for one declared graph input port."""

    port_label: str
    input_type: str
    source_title: str | None = None
    source_port_label: str | None = None
    scientific_summary: str = ""

    @property
    def is_connected(self) -> bool:
        return self.source_title is not None

    @property
    def source_summary(self) -> str:
        if not self.is_connected:
            expected = self.input_type.replace("_", " ")
            return f"Not connected · expects {expected}"
        if self.source_port_label:
            return f"{self.source_title} · {self.source_port_label}"
        return str(self.source_title)


class ConnectedInputRow(QFrame):
    """One icon-and-two-line binding row within a connected-input card."""

    def __init__(
        self,
        binding: ConnectedInputBinding,
        *,
        divider: bool,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.binding = binding
        self.setObjectName("InspectorConnectedInputRow")
        self.setProperty("hasDivider", divider)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(7)

        self.icon_label = QLabel(self)
        self.icon_label.setObjectName("InspectorConnectedInputIcon")
        self.icon_label.setFixedSize(18, 18)
        self.icon_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.icon_label, 0, Qt.AlignTop)

        text_layout = QVBoxLayout()
        text_layout.setSpacing(1)

        self.role_label = QLabel(binding.port_label, self)
        self.role_label.setObjectName("InspectorConnectedInputRole")
        self.role_label.setTextFormat(Qt.PlainText)
        self.role_label.setWordWrap(True)
        role_font = self.role_label.font()
        role_font.setWeight(QFont.DemiBold)
        self.role_label.setFont(role_font)
        text_layout.addWidget(self.role_label)

        self.source_label = QLabel(binding.source_summary, self)
        self.source_label.setObjectName("InspectorConnectedInputSource")
        self.source_label.setTextFormat(Qt.PlainText)
        self.source_label.setWordWrap(True)
        self.source_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        text_layout.addWidget(self.source_label)

        self.scientific_label = QLabel(binding.scientific_summary, self)
        self.scientific_label.setObjectName("InspectorConnectedInputScientific")
        self.scientific_label.setTextFormat(Qt.PlainText)
        self.scientific_label.setWordWrap(True)
        self.scientific_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.scientific_label.setVisible(bool(binding.scientific_summary))
        text_layout.addWidget(self.scientific_label)
        layout.addLayout(text_layout, 1)

        expected = binding.input_type.replace("_", " ")
        if binding.is_connected:
            tooltip = (
                f"{binding.port_label} expects {expected} data. Connected from "
                f"{binding.source_summary}."
            )
        else:
            tooltip = f"{binding.port_label} expects {expected} data."
        tooltip_html = f"<qt>{html.escape(tooltip)}</qt>"
        self.setToolTip(tooltip_html)
        self.role_label.setToolTip(tooltip_html)
        self.source_label.setToolTip(tooltip_html)
        self.scientific_label.setToolTip(tooltip_html)
        accessible_summary = f"{binding.port_label}: {binding.source_summary}"
        if binding.scientific_summary:
            accessible_summary += f"; {binding.scientific_summary}"
        self.setAccessibleName(accessible_summary)
        self._refresh_icon()

    def _refresh_icon(self, palette=None) -> None:
        icon = data_type_icon(
            self.binding.input_type,
            palette if palette is not None else self.palette(),
            16,
        )
        self.icon_label.setPixmap(icon.pixmap(16, 16))


class ConnectedInputsCard(QFrame):
    """One responsive card containing all graph-input bindings for a node."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme_refresh_in_progress = False
        self.rows: list[ConnectedInputRow] = []
        self.setObjectName("InspectorConnectedInputs")
        self.setAccessibleName("Connected node inputs")
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        card_layout = QVBoxLayout(self)
        card_layout.setContentsMargins(0, 0, 0, 0)
        card_layout.setSpacing(0)

        self.title_label = QLabel("Connected inputs", self)
        self.title_label.setObjectName("InspectorConnectedInputsTitle")
        self.title_label.setTextFormat(Qt.PlainText)
        self.title_label.setContentsMargins(8, 5, 8, 3)
        card_layout.addWidget(self.title_label)

        self.form_layout = QFormLayout()
        self.form_layout.setContentsMargins(0, 0, 0, 0)
        self.form_layout.setHorizontalSpacing(0)
        self.form_layout.setVerticalSpacing(0)
        self.form_layout.setRowWrapPolicy(QFormLayout.DontWrapRows)
        card_layout.addLayout(self.form_layout)
        self.refresh_theme()

    def set_bindings(self, bindings: list[ConnectedInputBinding]) -> None:
        while self.form_layout.rowCount():
            self.form_layout.removeRow(0)
        self.rows.clear()
        for index, binding in enumerate(bindings):
            row = ConnectedInputRow(
                binding,
                divider=index > 0,
                parent=self,
            )
            self.rows.append(row)
            self.form_layout.addRow(row)
        self.setVisible(bool(bindings))
        self.refresh_theme()

    def refresh_theme(self, palette: QPalette | None = None) -> None:
        if self._theme_refresh_in_progress:
            return
        self._theme_refresh_in_progress = True
        try:
            # Resolve from the owning surface (or an explicit palette) rather
            # than from this styled frame. Its previous QSS intentionally
            # changes background/text roles and must not feed into the next
            # live-theme refresh.
            if palette is None:
                palette_source = self.parentWidget()
                palette = (
                    palette_source.palette()
                    if palette_source is not None
                    else QApplication.palette()
                )
            colors = theme_colors(palette)
            self.setStyleSheet(
                "QFrame#InspectorConnectedInputs {"
                f" background: {colors.info.surface.name()};"
                f" border: 1px solid {colors.info.border.name()};"
                f" border-left: 3px solid {colors.info.accent.name()};"
                " border-radius: 4px;"
                "}"
                "QLabel#InspectorConnectedInputsTitle {"
                f" color: {colors.text.name()};"
                " font-size: 10px; font-weight: 600;"
                "}"
                "QFrame#InspectorConnectedInputRow {"
                " background: transparent; border: none;"
                "}"
                "QFrame#InspectorConnectedInputRow[hasDivider=\"true\"] {"
                f" border-top: 1px solid {colors.border.name()};"
                "}"
                "QLabel#InspectorConnectedInputRole {"
                f" color: {colors.text.name()};"
                " background: transparent;"
                "}"
                "QLabel#InspectorConnectedInputSource {"
                f" color: {colors.text.name()};"
                " background: transparent; font-size: 11px;"
                "}"
                "QLabel#InspectorConnectedInputScientific {"
                f" color: {colors.text.name()};"
                " background: transparent; font-size: 11px;"
                "}"
                "QLabel#InspectorConnectedInputIcon {"
                " background: transparent;"
                "}"
            )
            for row in self.rows:
                row._refresh_icon(palette)
        finally:
            self._theme_refresh_in_progress = False

    def changeEvent(self, event: QEvent) -> None:  # noqa: N802
        super().changeEvent(event)
        if event.type() in {
            QEvent.ApplicationPaletteChange,
            QEvent.PaletteChange,
        }:
            self.refresh_theme()


__all__ = [
    "ConnectedInputBinding",
    "ConnectedInputRow",
    "ConnectedInputsCard",
    "connected_input_scientific_summary",
]
