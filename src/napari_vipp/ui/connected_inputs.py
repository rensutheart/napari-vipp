"""Compact, read-only summaries of graph-connected node inputs."""

from __future__ import annotations

import html
from dataclasses import dataclass

import numpy as np
from qtpy.QtCore import QEvent, Qt, QTimer
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

from napari_vipp.core.transforms import TransformState, is_transform_data
from napari_vipp.ui.design import SPACE_SM, SPACE_XS, _font_host, type_style
from napari_vipp.ui.iconography import data_type_icon
from napari_vipp.ui.inspector import (
    InspectorSection,
    constrain_layout_minimum_height,
    release_layout_minimum_height,
)
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
        label = str(getattr(axis, "short_label", "") or getattr(axis, "name", "axis"))
        try:
            scale = f"{float(getattr(axis, 'scale', 1.0)):.6g}"
        except (TypeError, ValueError):
            continue
        values.append(f"{label} {scale} {display_unit}")
    return f"sampling {', '.join(values)}" if values else ""


def connected_input_scientific_summary(data, state) -> str:
    """Return lazy-safe shape/type/sampling context for one graph binding."""

    transform_state = data.state if is_transform_data(data) else state
    if isinstance(transform_state, TransformState):
        count = transform_state.transform_count
        axes = ", ".join(axis.upper() for axis in transform_state.spatial_axes)
        scope = "1" if count == 1 else f"{count:,} (one per time point)"
        return (
            f"Model: {transform_state.model}\nSpatial axes: {axes}\n"
            f"Transforms: {scope}\n"
            "Maps coordinates from the moving image to the reference."
        )
    table_source = state if state is not None else data
    row_count = getattr(table_source, "row_count", None)
    column_count = getattr(table_source, "column_count", None)
    if row_count is not None and column_count is not None:
        rows, columns = int(row_count), int(column_count)
        parts = [
            f"{rows:,} {'row' if rows == 1 else 'rows'}",
            f"{columns:,} {'field' if columns == 1 else 'fields'}",
        ]
        table_kind = str(
            getattr(table_source, "table_kind", "") or getattr(table_source, "kind", "")
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
        parts.append(f"{dtype_name} ({bit_depth})" if bit_depth else dtype_name)
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
    source_port_is_implicit: bool = False

    @property
    def is_connected(self) -> bool:
        return self.source_title is not None

    @property
    def source_summary(self) -> str:
        if not self.is_connected:
            expected = self.input_type.replace("_", " ")
            return f"Not connected · expects {expected}"
        source = f"From: {self.source_title}"
        if self.source_port_label and not self.source_port_is_implicit:
            return f"{source} · Output: {self.source_port_label}"
        return source


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
        layout.setContentsMargins(SPACE_SM, SPACE_XS, SPACE_SM, SPACE_XS)
        layout.setSpacing(SPACE_XS)

        self.icon_label = QLabel(self)
        self.icon_label.setObjectName("InspectorConnectedInputIcon")
        self.icon_label.setFixedSize(18, 18)
        self.icon_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.icon_label, 0, Qt.AlignTop)

        text_layout = QVBoxLayout()
        # Keep the receiving role and its upstream source visually paired.
        text_layout.setSpacing(SPACE_XS // 2)

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

        layout.addLayout(text_layout, 1)

        expected = binding.input_type.replace("_", " ")
        tooltip_lines = [f"Input: {binding.port_label}"]
        if binding.is_connected:
            tooltip_lines.extend((binding.source_summary, f"Accepts {expected} data."))
        else:
            tooltip_lines.append(binding.source_summary)
        tooltip_html = "<qt>" + "<br>".join(map(html.escape, tooltip_lines)) + "</qt>"
        self.setToolTip(tooltip_html)
        self.role_label.setToolTip(tooltip_html)
        self.source_label.setToolTip(tooltip_html)
        self.setAccessibleName(f"Input: {binding.port_label}; {binding.source_summary}")
        self._refresh_icon()
        constrain_layout_minimum_height(layout)

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
        self._observed_font_host = None
        self._font_refresh_timer = QTimer(self)
        self._font_refresh_timer.setSingleShot(True)
        self._font_refresh_timer.timeout.connect(self._refresh_host_font)
        self.installEventFilter(self)
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
        self.title_label.setContentsMargins(SPACE_SM, SPACE_XS, SPACE_SM, SPACE_XS)
        card_layout.addWidget(self.title_label)

        self.form_layout = QFormLayout()
        self.form_layout.setContentsMargins(0, 0, 0, 0)
        self.form_layout.setHorizontalSpacing(0)
        self.form_layout.setVerticalSpacing(0)
        self.form_layout.setRowWrapPolicy(QFormLayout.DontWrapRows)
        card_layout.addLayout(self.form_layout)
        self.refresh_theme()
        self._observe_font_host()

    def _observe_font_host(self) -> None:
        host = _font_host(self)
        if host is self._observed_font_host:
            return
        if self._observed_font_host is not None:
            try:
                self._observed_font_host.removeEventFilter(self)
            except RuntimeError:
                # Reparenting can outlive the old host's native QObject.
                pass
        self._observed_font_host = host
        if host is not None:
            host.installEventFilter(self)

    def _queue_host_font_refresh(self) -> None:
        if (
            not self._theme_refresh_in_progress
            and not self._font_refresh_timer.isActive()
        ):
            self._font_refresh_timer.start(0)

    def _refresh_host_font(self) -> None:
        self._observe_font_host()
        self.refresh_theme()

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        if event.type() == QEvent.ParentChange and (
            watched is self or watched is self._observed_font_host
        ):
            self._observe_font_host()
            self._queue_host_font_refresh()
        elif watched is self._observed_font_host and event.type() in {
            QEvent.FontChange,
            QEvent.ApplicationFontChange,
            QEvent.StyleChange,
        }:
            # A styled card need not inherit its host's FontChange. Observe
            # the same unroled authority used by the shared type resolver.
            self._queue_host_font_refresh()
        return super().eventFilter(watched, event)

    def set_bindings(self, bindings: list[ConnectedInputBinding]) -> None:
        if not bindings:
            # This persistent card is also reused by nodes without graph-input
            # context. Retire its queued ancestor commit before emptying it so
            # old registration geometry cannot relayout an unrelated form.
            release_layout_minimum_height(self.layout())
        # Parameter edits often refresh presentation twice without changing
        # any graph input. Removing/recreating these rows briefly collapses
        # the card and moves the controls in a scrolled inspector.
        if tuple(row.binding for row in self.rows) == tuple(bindings):
            self.setVisible(bool(bindings))
            if bindings:
                constrain_layout_minimum_height(self.layout())
            return
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
        if bindings:
            constrain_layout_minimum_height(self.layout())

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
                f" background: {colors.alternate_surface.name()};"
                f" border: 1px solid {colors.border.name()};"
                " border-radius: 4px;"
                "}"
                "QLabel#InspectorConnectedInputsTitle {"
                f" color: {colors.text.name()};"
                f" {type_style(self.title_label, 'body')} font-weight: 700;"
                "}"
                "QFrame#InspectorConnectedInputRow {"
                " background: transparent; border: none;"
                "}"
                'QFrame#InspectorConnectedInputRow[hasDivider="true"] {'
                f" border-top: 1px solid {colors.border.name()};"
                "}"
                "QLabel#InspectorConnectedInputRole {"
                f" color: {colors.text.name()};"
                f" background: transparent; {type_style(self, 'body')}"
                " font-weight: 600;"
                "}"
                "QLabel#InspectorConnectedInputSource {"
                f" color: {colors.text.name()};"
                f" background: transparent; {type_style(self, 'body')}"
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
        elif event.type() in {
            QEvent.FontChange,
            QEvent.ApplicationFontChange,
            QEvent.StyleChange,
        }:
            self._queue_host_font_refresh()


class InputMetadataSection(InspectorSection):
    """Optional, read-only details about incoming data, apart from wiring."""

    def __init__(self) -> None:
        super().__init__("Input metadata", expanded=False)
        self._entries: tuple[tuple[str, str], ...] = ()
        self._details_layout = QVBoxLayout(self.content_widget)
        self._details_layout.setContentsMargins(SPACE_SM, SPACE_XS, SPACE_SM, SPACE_SM)
        self._details_layout.setSpacing(SPACE_XS)
        self.hide()

    def set_entries(self, rows: list[tuple[str, str]]) -> None:
        entries = tuple(rows)
        if entries != self._entries:
            self._entries = entries
            while self._details_layout.count():
                widget = self._details_layout.takeAt(0).widget()
                widget.hide()
                widget.deleteLater()
            for role, summary in entries:
                title = QLabel(role, self.content_widget)
                title.setTextFormat(Qt.PlainText)
                title.setWordWrap(True)
                font = title.font()
                font.setWeight(QFont.DemiBold)
                title.setFont(font)
                details = QLabel(summary, self.content_widget)
                details.setTextFormat(Qt.PlainText)
                details.setWordWrap(True)
                details.setTextInteractionFlags(Qt.TextSelectableByMouse)
                details.setAccessibleName(f"{role}: {summary}")
                self._details_layout.addWidget(title)
                self._details_layout.addWidget(details)
            if entries:
                constrain_layout_minimum_height(self._details_layout)
            else:
                release_layout_minimum_height(self._details_layout)
        self.setVisible(bool(entries))


__all__ = [
    "ConnectedInputBinding",
    "ConnectedInputRow",
    "ConnectedInputsCard",
    "InputMetadataSection",
    "connected_input_scientific_summary",
]
