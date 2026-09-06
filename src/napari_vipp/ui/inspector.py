"""Capability-driven presentation helpers for the node inspector."""

from __future__ import annotations

from dataclasses import dataclass

from qtpy.QtCore import QEvent, QRectF, QSize, Qt, QTimer
from qtpy.QtGui import QPainter, QPalette, QPen
from qtpy.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from napari_vipp.core.pipeline import OperationSpec
from napari_vipp.ui.iconography import interface_icon
from napari_vipp.ui.palette_roles import theme_colors

PARAMETERS_SECTION = "parameters"
SOURCE_REPRESENTATION_SECTION = "source_representation"
OUTPUT_SELECTOR_SECTION = "output_selector"
COLOCALIZATION_SECTION = "colocalization"
LABEL_DISTRIBUTION_SECTION = "label_distribution"
MASK_SUMMARY_SECTION = "mask_summary"
TABLE_RESULTS_SECTION = "table_results"
HISTOGRAMS_SECTION = "histograms"
WRITER_STATUS_SECTION = "writer_status"
BEHAVIOR_SECTION = "behavior"
COMPUTE_SECTION = "compute"
METADATA_SECTION = "metadata"
HISTORY_SECTION = "history"

_WRITER_OPERATION_IDS = frozenset({"save_output", "batch_output"})
_COLOCALIZATION_SCATTER_OPERATION_IDS = frozenset(
    {
        "colocalization_metrics",
        "masked_colocalization_metrics",
        "colocalization_scatter_plot",
        "masked_colocalization_scatter_plot",
        "colocalized_voxels",
        "masked_colocalized_voxels",
        "racc_index",
        "masked_racc_index",
    }
)
_OBJECT_COLOCALIZATION_OPERATION_IDS = frozenset(
    {
        "object_colocalization_metrics",
    }
)
_MEASUREMENT_OPERATION_IDS = frozenset(
    {
        "intensity_histogram",
        "measure_objects",
        "measure_objects_intensity",
        "measure_3d_mesh_morphology",
        "analyze_skeleton",
        "measure_skeleton_branches",
        "summarize_skeleton_branches",
        "skeleton_graph_tables",
        "measure_overall_skeleton_network",
        "object_colocalization_metrics",
        "label_overlap_association",
        "nearest_object_distance",
        "event_localization",
        "summarize_measurements",
    }
)
_TABLE_TRANSFORM_OPERATION_IDS = frozenset(
    {
        "merge_tables",
        "add_metadata_columns",
        "select_table_columns",
    }
)
_METADATA_TRANSFORM_OPERATION_IDS = frozenset(
    {
        "assign_channel_colors",
        "reorder_axes",
        "set_microscope_metadata",
        "set_pixel_size",
    }
)
_CONNECTED_CONTEXT_OPERATION_IDS = frozenset({"born_wolf_psf"})
_THRESHOLD_DIAGNOSTIC_OPERATION_IDS = frozenset(
    {
        "otsu_threshold",
        "triangle_threshold",
        "imagej_auto_threshold",
        "li_threshold",
        "yen_threshold",
        "isodata_threshold",
        "minimum_threshold",
        "binary_threshold",
        "hysteresis_threshold",
        "adaptive_mean_threshold",
        "adaptive_gaussian_threshold",
        "sauvola_threshold",
        "niblack_threshold",
        "canny_edges",
    }
)


class _InspectorBusySpinner(QWidget):
    """Small palette-aware busy indicator for deferred inspector evidence."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._angle = 0
        self._busy = False
        self.setFixedSize(18, 18)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAccessibleName("")
        self._timer = QTimer(self)
        self._timer.setInterval(80)
        self._timer.timeout.connect(self._advance)

    def setBusy(self, busy: bool) -> None:  # noqa: N802 - Qt-compatible API
        busy = bool(busy)
        self._busy = busy
        if busy:
            self.setAccessibleName("Loading section")
            if not self._timer.isActive():
                self._timer.start()
        else:
            self.setAccessibleName("")
            self._timer.stop()
        self.update()

    def isBusy(self) -> bool:  # noqa: N802 - Qt-compatible API
        return self._busy

    def _advance(self) -> None:
        self._angle = (self._angle + 30) % 360
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        if not self._busy:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(QPen(theme_colors(self.palette()).info.accent, 2.0))
        painter.drawArc(
            QRectF(3.0, 3.0, 12.0, 12.0),
            int((90 - self._angle) * 16),
            int(275 * 16),
        )
        painter.end()


@dataclass(frozen=True, slots=True)
class InspectorProfile:
    """Stable semantic layout for one operation's inspector.

    The profile deliberately describes presentation only. Scientific inputs,
    parameters, outputs, and execution continue to be owned by the pipeline.
    """

    operation_id: str
    parameter_title: str
    primary_sections: tuple[str, ...]
    show_connected_inputs: bool
    output_action_kind: str
    supports_pin: bool
    execution_is_manual: bool
    show_output_selector: bool = False
    supports_all_outputs_action: bool = False
    distribution_kind: str = "none"

    @property
    def section_order(self) -> tuple[str, ...]:
        sections = (
            *self.primary_sections,
            BEHAVIOR_SECTION,
            COMPUTE_SECTION,
            METADATA_SECTION,
            HISTORY_SECTION,
        )
        return tuple(dict.fromkeys(sections))


def inspector_profile(
    spec: OperationSpec,
    *,
    effective_output_type: str | None = None,
) -> InspectorProfile:
    """Return a complete inspector presentation for ``spec``.

    Every registered operation reaches one of these semantic fallbacks, while
    science-specific surfaces retain explicit priority. In particular,
    colocalization scatter is primary evidence and therefore precedes channel
    histograms; label-producing operations use object distributions instead of
    misleading label-ID histograms.
    """

    operation_id = str(spec.id)
    is_source = operation_id == "input"
    is_writer = operation_id in _WRITER_OPERATION_IDS
    is_multi_output = bool(spec.is_multi_output)
    declared_output_types = frozenset(
        port.output_type for port in spec.output_ports
    )
    output_type = str(effective_output_type or spec.output_type)
    is_table = (
        output_type == "table"
        if effective_output_type is not None
        else bool(declared_output_types) and declared_output_types == {"table"}
    )

    if is_source:
        parameter_title = "Source & data representations"
    elif is_writer:
        parameter_title = "Output settings"
    elif operation_id in _COLOCALIZATION_SCATTER_OPERATION_IDS or (
        operation_id in _OBJECT_COLOCALIZATION_OPERATION_IDS
    ):
        parameter_title = "Colocalization"
    elif operation_id in _TABLE_TRANSFORM_OPERATION_IDS:
        parameter_title = "Table settings"
    elif operation_id in _MEASUREMENT_OPERATION_IDS:
        parameter_title = "Measurements"
    else:
        parameter_title = "Parameters"

    primary: list[str] = [PARAMETERS_SECTION]
    if is_source:
        primary.extend((SOURCE_REPRESENTATION_SECTION, HISTOGRAMS_SECTION))
        distribution_kind = "analysis_intensity"
    elif is_writer:
        primary.append(WRITER_STATUS_SECTION)
        distribution_kind = "none"
    else:
        if is_multi_output:
            primary.append(OUTPUT_SELECTOR_SECTION)

        if operation_id == "intensity_histogram":
            # The plot is the primary scientific result; the table remains
            # immediately available for exact bin inspection and export.
            primary.extend((HISTOGRAMS_SECTION, TABLE_RESULTS_SECTION))
            distribution_kind = "histogram_result"
        elif operation_id in _OBJECT_COLOCALIZATION_OPERATION_IDS:
            # Per-object results are the node's scientific output. The scatter
            # remains more informative than either channel histogram, but is a
            # secondary diagnostic for this table-producing operation.
            primary.extend(
                (
                    TABLE_RESULTS_SECTION,
                    COLOCALIZATION_SECTION,
                    HISTOGRAMS_SECTION,
                )
            )
            distribution_kind = "colocalization_inputs"
        elif operation_id in _COLOCALIZATION_SCATTER_OPERATION_IDS:
            # Threshold lines and joint density are the decision surface. Keep
            # them ahead of the per-channel input distributions for both table
            # and image-producing colocalization nodes.
            primary.append(COLOCALIZATION_SECTION)
            if is_table:
                primary.append(TABLE_RESULTS_SECTION)
            primary.append(HISTOGRAMS_SECTION)
            distribution_kind = "colocalization_inputs"
        elif is_table:
            primary.append(TABLE_RESULTS_SECTION)
            distribution_kind = "table"
        elif operation_id in _METADATA_TRANSFORM_OPERATION_IDS:
            # These nodes preserve pixel values. Repeating identical input and
            # output histograms obscures the axis, calibration, or channel
            # metadata that the node actually changes.
            primary.append(METADATA_SECTION)
            distribution_kind = "metadata"
        elif operation_id == "remove_small_objects":
            # The cutoff acts on input object sizes for both integer labels and
            # Boolean connected components. A Boolean result still benefits
            # from its separate foreground-occupancy summary.
            primary.append(LABEL_DISTRIBUTION_SECTION)
            if output_type == "mask":
                primary.append(MASK_SUMMARY_SECTION)
            distribution_kind = "object_sizes"
        elif operation_id == "filter_labels_by_property":
            # This node makes its decision from a connected measurement
            # table, not from label-ID frequency or object volume.  Reuse the
            # compact distribution surface, but give the widget a distinct
            # semantic kind so it renders the selected property and range
            # markers instead of scanning the label image.
            primary.append(LABEL_DISTRIBUTION_SECTION)
            distribution_kind = "property_filter"
        elif operation_id == "filter_labels_by_volume":
            primary.append(LABEL_DISTRIBUTION_SECTION)
            distribution_kind = "labels"
        elif (
            operation_id in _THRESHOLD_DIAGNOSTIC_OPERATION_IDS
            and output_type == "mask"
        ):
            # A threshold mask is best understood from the input distribution
            # (and, where applicable, its marker) plus an output occupancy
            # summary. The latter replaces a generic Boolean output histogram.
            primary.extend((HISTOGRAMS_SECTION, MASK_SUMMARY_SECTION))
            distribution_kind = "threshold"
        elif output_type == "labels":
            primary.append(LABEL_DISTRIBUTION_SECTION)
            distribution_kind = "labels"
        elif output_type == "mask":
            primary.append(MASK_SUMMARY_SECTION)
            distribution_kind = "mask_occupancy"
        else:
            primary.append(HISTOGRAMS_SECTION)
            distribution_kind = (
                "runtime" if spec.output_type == "any" else "intensity"
            )

    if is_source:
        action_kind = "source"
        supports_pin = True
    elif is_writer:
        action_kind = "none"
        supports_pin = False
    elif is_multi_output:
        if is_table:
            action_kind = "multi_table"
            supports_pin = False
        elif output_type in {"image", "mask", "labels"}:
            action_kind = f"multi_{output_type}"
            supports_pin = True
        else:
            action_kind = "multi_runtime"
            supports_pin = True
    elif is_table:
        action_kind = "table"
        supports_pin = False
    elif output_type in {"image", "mask", "labels"}:
        action_kind = output_type
        supports_pin = True
    else:
        action_kind = "runtime"
        supports_pin = True

    maximum_inputs = spec.max_inputs
    has_multiple_possible_inputs = bool(
        len(spec.input_ports) > 1
        or maximum_inputs is None
        or (maximum_inputs is not None and int(maximum_inputs) > 1)
    )
    # Measurement inputs are scientifically meaningful context even when a
    # node has just one port. They are graph connections, never parameter
    # choices, so keep the compact read-only connection row for these nodes.
    show_connected_inputs = bool(
        not is_source
        and not is_writer
        and (
            has_multiple_possible_inputs
            or operation_id in _MEASUREMENT_OPERATION_IDS
            or operation_id in _CONNECTED_CONTEXT_OPERATION_IDS
        )
    )

    return InspectorProfile(
        operation_id=operation_id,
        parameter_title=parameter_title,
        primary_sections=tuple(primary),
        show_connected_inputs=show_connected_inputs,
        output_action_kind=action_kind,
        supports_pin=supports_pin,
        execution_is_manual=spec.execution_policy == "manual",
        show_output_selector=is_multi_output,
        supports_all_outputs_action=is_multi_output,
        distribution_kind=distribution_kind,
    )


class _InspectorSectionTitleButton(QToolButton):
    """Natural-width section title that can still shrink in a narrow dock."""

    _ICON_TEXT_GAP = 4
    _HORIZONTAL_PADDING = 12

    def sizeHint(self) -> QSize:  # noqa: N802
        """Size to the visible icon and title rather than extra tool-button chrome."""

        hint = super().sizeHint()
        text_width = self.fontMetrics().horizontalAdvance(self.text())
        icon_width = 0 if self.icon().isNull() else self.iconSize().width()
        content_width = text_width + icon_width
        if text_width and icon_width:
            content_width += self._ICON_TEXT_GAP
        hint.setWidth(content_width + self._HORIZONTAL_PADDING)
        return hint

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        hint = super().minimumSizeHint()
        hint.setWidth(0)
        return hint


class _InspectorSectionSummaryLabel(QLabel):
    """Right-aligned section summary with clean, prefix-preserving elision."""

    def displayText(self) -> str:  # noqa: N802
        """Return the text painted at the label's current width."""

        text = self.text()
        if not text:
            return ""
        available_width = max(int(self.contentsRect().width()), 0)
        metrics = self.fontMetrics()
        if available_width < metrics.horizontalAdvance("…") + 2:
            return ""
        return metrics.elidedText(text, Qt.ElideRight, available_width)

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        text = self.displayText()
        if not text:
            return
        painter = QPainter(self)
        painter.setFont(self.font())
        self.style().drawItemText(
            painter,
            self.contentsRect(),
            self.alignment() | Qt.TextSingleLine,
            self.palette(),
            self.isEnabled(),
            text,
            QPalette.WindowText,
        )


class InspectorSection(QWidget):
    """Compact, accessible collapsible inspector section."""

    def __init__(
        self,
        title: str,
        *,
        expanded: bool = True,
        busy_capable: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._theme_refresh_in_progress = False
        self.setObjectName("InspectorSection")
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        self.header = QFrame(self)
        self.header.setObjectName("InspectorSectionHeader")
        self.header.setCursor(Qt.PointingHandCursor)
        self.header.setAttribute(Qt.WA_Hover, True)
        header_layout = QHBoxLayout(self.header)
        header_layout.setContentsMargins(6, 1, 4, 1)
        header_layout.setSpacing(6)

        self.title_button = _InspectorSectionTitleButton(self.header)
        self.title_button.setObjectName("InspectorSectionTitle")
        self.title_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.title_button.setAutoRaise(True)
        self.title_button.setCursor(Qt.PointingHandCursor)
        self.title_button.setIconSize(self.title_button.iconSize())
        # Use the title's natural width without turning it into a hard dock
        # minimum. The summary then receives every remaining flexible pixel.
        self.title_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.title_button.setMinimumWidth(0)
        title_font = self.title_button.font()
        title_font.setBold(True)
        self.title_button.setFont(title_font)
        header_layout.addWidget(self.title_button, 0)

        self.busy_indicator = _InspectorBusySpinner(self.header)
        self._busy_capable = bool(busy_capable)
        # An idle spinner paints nothing and must not permanently consume scarce
        # header width.  When active it sits before the flexible summary, keeping
        # the summary's right edge anchored beside the disclosure chevron.
        self.busy_indicator.hide()
        header_layout.addWidget(self.busy_indicator, 0, Qt.AlignVCenter)

        self.summary_label = _InspectorSectionSummaryLabel("", self.header)
        self.summary_label.setObjectName("InspectorSectionSummary")
        self.summary_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        # The summary consumes the flexible middle but contributes no hard
        # minimum width. At truly narrow widths it elides cleanly instead of
        # clipping the beginning of right-aligned text.
        self.summary_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.summary_label.setMinimumWidth(0)
        self.summary_label.setWordWrap(False)
        self.summary_label.setContentsMargins(0, 0, 0, 0)
        self.summary_label.setCursor(Qt.PointingHandCursor)
        header_layout.addWidget(self.summary_label, 1)

        # Keep the disclosure glyph at the far right, matching the visual
        # reading order of the approved inspector design.  ``toggle_button``
        # remains the public state-bearing control for API/test compatibility.
        self.toggle_button = QToolButton(self.header)
        self.toggle_button.setObjectName("InspectorSectionToggle")
        self.toggle_button.setCheckable(True)
        self.toggle_button.setAutoRaise(True)
        self.toggle_button.setCursor(Qt.PointingHandCursor)
        self.toggle_button.setFixedWidth(24)
        self.toggle_button.setIconSize(QSize(14, 14))
        self.toggle_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        header_layout.addWidget(self.toggle_button, 0)

        self.content_widget = QWidget(self)
        self.content_widget.setObjectName("InspectorSectionContent")
        self.content_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.header)
        layout.addWidget(self.content_widget)

        self._title = ""
        self.setTitle(title)
        self.title_button.clicked.connect(self._toggle_from_header)
        self.toggle_button.toggled.connect(self._set_expanded_from_toggle)
        self.header.installEventFilter(self)
        self.summary_label.installEventFilter(self)
        self.setExpanded(expanded)
        self.refresh_theme()

    def title(self) -> str:
        return self._title

    def setTitle(self, title: str) -> None:  # noqa: N802 - Qt-compatible API
        self._title = str(title)
        self.title_button.setText(self._title)
        self._refresh_semantic_icon()
        self._sync_accessibility()

    def setSummary(self, summary: str) -> None:  # noqa: N802
        text = str(summary or "").strip()
        self.summary_label.setText(text)
        self.summary_label.setToolTip(text)
        self.summary_label.setAccessibleDescription(text)
        self.summary_label.setVisible(bool(text))
        self._sync_accessibility()

    def setBusy(  # noqa: N802 - Qt-compatible API
        self,
        busy: bool,
        *,
        minimum_content_height: int = 0,
    ) -> None:
        """Show deferred-work feedback while reserving stable section geometry."""

        busy = bool(busy)
        self.busy_indicator.setVisible(busy)
        self.busy_indicator.setBusy(busy)
        self.content_widget.setMinimumHeight(
            max(int(minimum_content_height), 0) if busy else 0
        )
        self.setAccessibleDescription(
            f"{self._title} is loading" if busy else ""
        )

    def isBusy(self) -> bool:  # noqa: N802 - Qt-compatible API
        return self.busy_indicator.isBusy()

    def isExpanded(self) -> bool:  # noqa: N802
        return self.toggle_button.isChecked()

    def setExpanded(self, expanded: bool) -> None:  # noqa: N802
        expanded = bool(expanded)
        self.toggle_button.setChecked(expanded)
        self._set_expanded_from_toggle(expanded)

    def _set_expanded_from_toggle(self, expanded: bool) -> None:
        self.content_widget.setVisible(bool(expanded))
        self.toggle_button.setArrowType(Qt.NoArrow)
        self.toggle_button.setProperty(
            "vippDisclosureState",
            "expanded" if expanded else "collapsed",
        )
        self._refresh_disclosure_icon()
        self._sync_accessibility()

    def _toggle_from_header(self) -> None:
        self.setExpanded(not self.isExpanded())

    def _sync_accessibility(self) -> None:
        action = "Collapse" if self.isExpanded() else "Expand"
        summary = self.summary_label.text()
        self.toggle_button.setAccessibleName(f"{action} {self._title}")
        self.toggle_button.setToolTip(f"{action} {self._title.lower()}")
        self.toggle_button.setAccessibleDescription(summary)
        self.title_button.setAccessibleName(f"{action} {self._title}")
        self.title_button.setToolTip(f"{action} {self._title.lower()}")
        self.title_button.setAccessibleDescription(summary)

    def _semantic_icon_kind(self) -> str:
        title = self._title.casefold()
        if "histogram" in title or "distribution" in title:
            return "histogram"
        if "colocalization" in title or "scatter" in title:
            return "overlap"
        if "result" in title:
            return "table"
        if "metadata" in title:
            return "database"
        if "source" in title or "representation" in title:
            return "layers"
        if "displayed output" in title:
            return "image"
        if "mask" in title:
            return "regions"
        if "label" in title:
            return "tag"
        if "output" in title or "writer" in title:
            return "save"
        if "compute" in title:
            return "chip"
        if "history" in title:
            return "history"
        if "behavior" in title:
            return "eye"
        if any(
            word in title
            for word in ("parameter", "measurement", "setting")
        ):
            return "sliders"
        return "nodes"

    def _refresh_semantic_icon(self) -> None:
        if not hasattr(self, "title_button"):
            return
        colors = theme_colors(self.palette())
        icon_palette = QPalette(self.palette())
        icon_palette.setColor(QPalette.ButtonText, colors.info.accent)
        self.title_button.setIcon(
            interface_icon(self._semantic_icon_kind(), icon_palette, 16)
        )

    def _refresh_disclosure_icon(self) -> None:
        if not hasattr(self, "toggle_button"):
            return
        colors = theme_colors(self.palette())
        icon_palette = QPalette(self.palette())
        icon_palette.setColor(QPalette.ButtonText, colors.muted_text)
        icon_kind = "chevron-down" if self.isExpanded() else "chevron-right"
        self.toggle_button.setIcon(interface_icon(icon_kind, icon_palette, 14))

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        if (
            watched is self.header or watched is self.summary_label
        ) and event.type() == QEvent.MouseButtonRelease:
            if event.button() == Qt.LeftButton:
                self._toggle_from_header()
                return True
        return super().eventFilter(watched, event)

    def refresh_theme(self) -> None:
        if self._theme_refresh_in_progress:
            return
        colors = theme_colors(self.palette())
        style_sheet = (
            "QWidget#InspectorSection {"
            f" border: 1px solid {colors.border.name()};"
            " border-radius: 5px;"
            "}"
            "QFrame#InspectorSectionHeader {"
            f" background: {colors.alternate_surface.name()};"
            " border: none;"
            " border-radius: 4px;"
            "}"
            "QFrame#InspectorSectionHeader:hover {"
            f" background: {colors.raised_surface.name()};"
            "}"
            "QToolButton#InspectorSectionTitle {"
            f" color: {colors.text.name()};"
            " background: transparent; border: none;"
            " padding: 5px 3px; text-align: left;"
            " font-weight: 600;"
            "}"
            "QToolButton#InspectorSectionToggle {"
            f" color: {colors.text.name()};"
            " background: transparent; border: none; padding: 5px 2px;"
            "}"
            "QLabel#InspectorSectionSummary {"
            f" color: {colors.muted_text.name()};"
            " background: transparent; border: none;"
            " padding: 0; font-size: 10px;"
            "}"
            "QWidget#InspectorSectionContent { border: none; }"
        )
        if style_sheet == self.styleSheet():
            return
        self._theme_refresh_in_progress = True
        try:
            self.setStyleSheet(style_sheet)
            self._refresh_semantic_icon()
            self._refresh_disclosure_icon()
        finally:
            self._theme_refresh_in_progress = False

    def changeEvent(self, event) -> None:  # noqa: N802
        super().changeEvent(event)
        if event.type() in {
            QEvent.PaletteChange,
            QEvent.ApplicationPaletteChange,
            QEvent.StyleChange,
        }:
            self.refresh_theme()


__all__ = [
    "BEHAVIOR_SECTION",
    "COLOCALIZATION_SECTION",
    "COMPUTE_SECTION",
    "HISTOGRAMS_SECTION",
    "HISTORY_SECTION",
    "InspectorProfile",
    "InspectorSection",
    "LABEL_DISTRIBUTION_SECTION",
    "MASK_SUMMARY_SECTION",
    "METADATA_SECTION",
    "OUTPUT_SELECTOR_SECTION",
    "PARAMETERS_SECTION",
    "SOURCE_REPRESENTATION_SECTION",
    "TABLE_RESULTS_SECTION",
    "WRITER_STATUS_SECTION",
    "inspector_profile",
]
