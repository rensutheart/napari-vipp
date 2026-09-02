"""Resizable detailed histogram inspection and plot-image export."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import tifffile
from qtpy.QtCore import QEvent, QPoint, QSignalBlocker, Qt, Signal
from qtpy.QtGui import QImage, QPainter, QPalette
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from napari_vipp.core.pipeline import (
    NODE_LIBRARY_BY_ID,
    ParameterVisibilityContext,
    resolve_parameter_visibility,
    validate_parameter_value,
)
from napari_vipp.core.tables import TableData
from napari_vipp.ui.controls import (
    ChoiceControl,
    NumericEntryControl,
    ParameterBounds,
    ParameterControl,
)
from napari_vipp.ui.palette_roles import theme_colors
from napari_vipp.ui.plots import (
    DETAILED_HISTOGRAM_DEFAULT_GRID_DIVISIONS,
    DETAILED_HISTOGRAM_MAX_GRID_DIVISIONS,
    DETAILED_HISTOGRAM_MIN_GRID_DIVISIONS,
    DetailedHistogramPlot,
)

HISTOGRAM_Y_VALUE_OPTIONS = (
    "Count",
    "Fraction",
    "Probability density",
    "Cumulative count",
    "Cumulative fraction",
)

_Y_VALUE_COLUMN = {
    "Count": "count",
    "Fraction": "fraction",
    "Probability density": "density",
    "Cumulative count": "cumulative_count",
    "Cumulative fraction": "cumulative_fraction",
}


def _histogram_parameter_bounds(spec) -> ParameterBounds:
    """Translate authored histogram bounds into the shared control contract."""

    slider_minimum = (
        spec.minimum if spec.slider_minimum is None else spec.slider_minimum
    )
    slider_maximum = (
        spec.maximum if spec.slider_maximum is None else spec.slider_maximum
    )
    has_slider_window = (
        spec.slider_minimum is not None or spec.slider_maximum is not None
    )
    return ParameterBounds(
        slider_minimum,
        slider_maximum,
        spec.step,
        spec.decimals,
        expandable=not has_slider_window,
        entry_minimum=spec.minimum if has_slider_window else None,
        entry_maximum=spec.maximum if has_slider_window else None,
    )


class HistogramDialog(QDialog):
    """Nonmodal detailed view of one already-calculated histogram table.

    The dialog never scans a source image.  It accepts either VIPP's histogram
    ``TableData`` or the equivalent extracted bin/value arrays, then derives
    every presentation from those bounded results.  Axis and y-value controls
    therefore update immediately without invalidating workflow calculation.
    """

    presentationChanged = Signal(str, bool, bool)
    calculationParametersChanged = Signal(dict)
    exportCompleted = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("VippHistogramDialog")
        self.setAttribute(Qt.WA_WindowPropagation, True)
        self.setWindowTitle("Histogram")
        self.setWindowModality(Qt.NonModal)
        self.setSizeGripEnabled(True)
        self.setMinimumSize(520, 380)
        self.resize(960, 680)

        self.calculation_group = QGroupBox("Histogram calculation", self)
        self.calculation_form = QFormLayout(self.calculation_group)
        self.calculation_form.setContentsMargins(12, 10, 12, 10)
        self.calculation_form.setHorizontalSpacing(12)
        self.calculation_form.setVerticalSpacing(7)
        self.calculation_controls: dict[str, QWidget] = {}
        self._calculation_specs = NODE_LIBRARY_BY_ID[
            "intensity_histogram"
        ].parameters
        self._calculation_parameters = {
            spec.name: spec.default for spec in self._calculation_specs
        }
        self._build_calculation_controls()

        self.y_values_combo = QComboBox(self)
        self.y_values_combo.addItems(HISTOGRAM_Y_VALUE_OPTIONS)
        self.y_values_combo.setAccessibleName("Histogram y values")
        self.y_values_combo.setToolTip(
            "Count shows values per bin. Fraction sums to one. Probability "
            "density integrates to one after accounting for bin width. "
            "Cumulative options show the running total from low to high values."
        )
        self.log_x_checkbox = QCheckBox("Log X", self)
        self.log_x_checkbox.setAccessibleName("Logarithmic histogram x axis")
        self.log_y_checkbox = QCheckBox("Log Y", self)
        self.log_y_checkbox.setAccessibleName("Logarithmic histogram y axis")
        self.log_y_checkbox.setToolTip(
            "Use a true base-10 y axis. Zero-valued bins are omitted because "
            "zero has no logarithm."
        )

        self.x_grid_divisions_spinbox = QSpinBox(self)
        self.x_grid_divisions_spinbox.setRange(
            DETAILED_HISTOGRAM_MIN_GRID_DIVISIONS,
            DETAILED_HISTOGRAM_MAX_GRID_DIVISIONS,
        )
        self.x_grid_divisions_spinbox.setValue(
            DETAILED_HISTOGRAM_DEFAULT_GRID_DIVISIONS
        )
        self.x_grid_divisions_spinbox.setAccessibleName(
            "Histogram x axis grid divisions"
        )
        self.y_grid_divisions_spinbox = QSpinBox(self)
        self.y_grid_divisions_spinbox.setRange(
            DETAILED_HISTOGRAM_MIN_GRID_DIVISIONS,
            DETAILED_HISTOGRAM_MAX_GRID_DIVISIONS,
        )
        self.y_grid_divisions_spinbox.setValue(
            DETAILED_HISTOGRAM_DEFAULT_GRID_DIVISIONS
        )
        self.y_grid_divisions_spinbox.setAccessibleName(
            "Histogram y axis grid divisions"
        )
        grid_divisions_tooltip = (
            "Set the number of intervals between major grid lines. Endpoints "
            "are included, so 4 divisions draws 5 grid lines. Logarithmic "
            "axes divide the base-10 axis range evenly."
        )
        self.x_grid_divisions_spinbox.setToolTip(grid_divisions_tooltip)
        self.y_grid_divisions_spinbox.setToolTip(grid_divisions_tooltip)

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(8)
        controls.addWidget(QLabel("Y values", self))
        controls.addWidget(self.y_values_combo)
        controls.addWidget(self.log_x_checkbox)
        controls.addWidget(self.log_y_checkbox)
        controls.addStretch(1)

        grid_controls = QHBoxLayout()
        grid_controls.setContentsMargins(0, 0, 0, 0)
        grid_controls.setSpacing(8)
        self.grid_divisions_label = QLabel("Major grid divisions", self)
        self.x_grid_divisions_label = QLabel("X", self)
        self.y_grid_divisions_label = QLabel("Y", self)
        self.x_grid_divisions_label.setBuddy(self.x_grid_divisions_spinbox)
        self.y_grid_divisions_label.setBuddy(self.y_grid_divisions_spinbox)
        grid_controls.addWidget(self.grid_divisions_label)
        grid_controls.addWidget(self.x_grid_divisions_label)
        grid_controls.addWidget(self.x_grid_divisions_spinbox)
        grid_controls.addWidget(self.y_grid_divisions_label)
        grid_controls.addWidget(self.y_grid_divisions_spinbox)
        # Keep the explanatory label and both axis controls as one visual
        # group. Any surplus window width belongs after the group, not between
        # its label and the first spinner.
        grid_controls.addStretch(1)

        self.summary_label = QLabel("No histogram result is available.", self)
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.summary_label.setAccessibleName("Histogram result summary")

        self.plot = DetailedHistogramPlot(self)
        self.plot.setMinimumSize(440, 280)

        self.export_hint = QLabel(
            "Export uses the plot's current on-screen pixel dimensions.",
            self,
        )
        self.export_hint.setWordWrap(True)
        self.export_button = QPushButton("Export PNG/TIFF…", self)
        self.export_button.setEnabled(False)
        self.close_button = QPushButton("Close", self)

        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.addWidget(self.export_hint, 1)
        actions.addWidget(self.export_button)
        actions.addWidget(self.close_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.calculation_group)
        layout.addLayout(controls)
        layout.addLayout(grid_controls)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.plot, 1)
        layout.addLayout(actions)

        self._table: TableData | None = None
        self._bin_edges = np.array([], dtype=np.float64)
        self._values_by_column: dict[str, np.ndarray] = {}
        self._series_labels: tuple[str, ...] = ()
        self._series_colors: tuple[str, ...] = ()
        self._metadata: object | None = None
        self._title = "Histogram"
        self._x_axis_label = "Intensity (a.u.)"
        self._summary_override = ""
        self._theme_refresh_in_progress = False

        self.y_values_combo.currentTextChanged.connect(self._on_presentation_changed)
        self.log_x_checkbox.toggled.connect(self._on_presentation_changed)
        self.log_y_checkbox.toggled.connect(self._on_presentation_changed)
        self.x_grid_divisions_spinbox.valueChanged.connect(
            self._on_grid_divisions_changed
        )
        self.y_grid_divisions_spinbox.valueChanged.connect(
            self._on_grid_divisions_changed
        )
        self.export_button.clicked.connect(self.request_export)
        self.close_button.clicked.connect(self.close)
        self.refresh_theme()

    @property
    def table(self) -> TableData | None:
        return self._table

    @property
    def bin_edges(self) -> np.ndarray:
        view = self._bin_edges.view()
        view.flags.writeable = False
        return view

    @property
    def y_value_name(self) -> str:
        return self.y_values_combo.currentText()

    @property
    def calculation_parameters(self) -> dict[str, object]:
        """Return a defensive copy of the calculation controls' model state."""

        return dict(self._calculation_parameters)

    def set_calculation_parameters(
        self,
        parameters: Mapping[str, object],
    ) -> None:
        """Mirror histogram-node parameters without emitting an edit signal."""

        if not isinstance(parameters, Mapping):
            raise TypeError("Histogram calculation parameters must be a mapping.")
        expected = {spec.name for spec in self._calculation_specs}
        unknown = set(parameters) - expected
        if unknown:
            names = ", ".join(sorted(map(str, unknown)))
            raise ValueError(f"Unknown histogram calculation parameters: {names}.")
        merged = dict(self._calculation_parameters)
        merged.update(parameters)
        for spec in self._calculation_specs:
            validate_parameter_value(
                spec,
                merged[spec.name],
                context="Histogram calculation parameter",
            )

        self._calculation_parameters = merged
        for spec in self._calculation_specs:
            control = self.calculation_controls[spec.name]
            value = merged[spec.name]
            bounds = _histogram_parameter_bounds(spec)
            if isinstance(control, ChoiceControl):
                control.set_choices(
                    spec.choices,
                    value,
                    emit=False,
                    choice_labels=spec.choice_labels,
                )
            control.set_bounds(bounds, value, emit=False)
        self._refresh_calculation_row_visibility()

    def _build_calculation_controls(self) -> None:
        for spec in self._calculation_specs:
            bounds = _histogram_parameter_bounds(spec)
            if spec.kind == "choice":
                control_class = ChoiceControl
            elif spec.name in {"custom_min", "custom_max"}:
                control_class = NumericEntryControl
            else:
                control_class = ParameterControl
            control = control_class(
                spec,
                self._calculation_parameters[spec.name],
                bounds,
                self.calculation_group,
            )
            label = QLabel(spec.label, self.calculation_group)
            label.setBuddy(control)
            tooltip = str(spec.tooltip or "").strip()
            if tooltip:
                label.setToolTip(tooltip)
                control.setToolTip(tooltip)
                for child in control.findChildren(QWidget):
                    child.setToolTip(tooltip)
            self.calculation_form.addRow(label, control)
            self.calculation_controls[spec.name] = control
            control.valueChanged.connect(
                lambda value, name=spec.name: (
                    self._on_calculation_parameter_changed(name, value)
                )
            )
        self._refresh_calculation_row_visibility()

    def _on_calculation_parameter_changed(self, name: str, value: object) -> None:
        spec = next(
            candidate
            for candidate in self._calculation_specs
            if candidate.name == name
        )
        validate_parameter_value(
            spec,
            value,
            context="Histogram calculation parameter",
        )
        if self._calculation_parameters.get(name) == value:
            return
        self._calculation_parameters[name] = value
        self._refresh_calculation_row_visibility()
        self.calculationParametersChanged.emit(
            dict(self._calculation_parameters)
        )

    def _refresh_calculation_row_visibility(self) -> None:
        context = ParameterVisibilityContext(
            operation_id="intensity_histogram",
            parameter_values=dict(self._calculation_parameters),
        )
        for spec in self._calculation_specs:
            control = self.calculation_controls.get(spec.name)
            if control is None:
                continue
            visible = resolve_parameter_visibility(spec, context=context).visible
            control.setVisible(visible)
            label = self.calculation_form.labelForField(control)
            if label is not None:
                label.setVisible(visible)

    def clear(self, message: str = "No histogram result is available.") -> None:
        """Clear the bounded result without changing presentation choices."""

        self._table = None
        self._bin_edges = np.array([], dtype=np.float64)
        self._values_by_column.clear()
        self._series_labels = ()
        self._series_colors = ()
        self._metadata = None
        self._summary_override = ""
        self.summary_label.setText(str(message))
        self.export_button.setEnabled(False)
        self.log_x_checkbox.setEnabled(False)
        self.plot.clear(str(message))

    def set_histogram(
        self,
        table: TableData | None = None,
        *,
        bin_edges=None,
        count=None,
        fraction=None,
        density=None,
        cumulative_count=None,
        cumulative_fraction=None,
        metadata: object | None = None,
        title: str = "Histogram",
        x_axis_label: str = "Intensity (a.u.)",
        summary: str = "",
    ) -> None:
        """Display a histogram table or equivalent precomputed arrays.

        Passing ``table`` expects the histogram operation's columns:
        ``bin_left``, ``bin_right``, ``count``, ``fraction``, ``density``,
        ``cumulative_count``, and ``cumulative_fraction``.  Metadata is read
        from ``table.histogram_metadata`` when available.  Array callers must
        provide ``bin_edges`` and ``count``; omitted derived representations
        are calculated only from those small histogram arrays.
        """

        if table is not None:
            if any(
                value is not None
                for value in (
                    bin_edges,
                    count,
                    fraction,
                    density,
                    cumulative_count,
                    cumulative_fraction,
                )
            ):
                raise ValueError(
                    "Pass either a histogram table or extracted arrays, not both."
                )
            edges, values, series_labels, series_colors = (
                histogram_arrays_from_table(table)
            )
            resolved_metadata = (
                metadata
                if metadata is not None
                else getattr(table, "histogram_metadata", None)
            )
        else:
            edges, values = _histogram_arrays_from_values(
                bin_edges=bin_edges,
                count=count,
                fraction=fraction,
                density=density,
                cumulative_count=cumulative_count,
                cumulative_fraction=cumulative_fraction,
            )
            series_labels = ()
            series_colors = ()
            resolved_metadata = metadata

        # DetailedHistogramPlot performs the authoritative finite, monotonic,
        # and non-negative validation.  Store independent copies so neither the
        # workflow table nor caller arrays can mutate an open view.
        self._table = table
        self._bin_edges = np.asarray(edges, dtype=np.float64).copy()
        self._values_by_column = {
            name: np.asarray(column, dtype=np.float64).copy()
            for name, column in values.items()
        }
        self._series_labels = tuple(series_labels)
        self._series_colors = tuple(series_colors)
        self._metadata = resolved_metadata
        self._title = str(title).strip() or "Histogram"
        self._x_axis_label = str(x_axis_label).strip() or "Value"
        self._summary_override = str(summary).strip()
        self.setWindowTitle(f"{self._title} — Histogram")

        if self._bin_edges.size < 2:
            self.log_x_checkbox.setEnabled(False)
            if self.log_x_checkbox.isChecked():
                with QSignalBlocker(self.log_x_checkbox):
                    self.log_x_checkbox.setChecked(False)
            self.summary_label.setText(self._histogram_summary())
            self.export_button.setEnabled(False)
            self.plot.clear("No finite values are available to plot.")
            return

        supports_log_x = bool(self._bin_edges.size and np.all(self._bin_edges > 0.0))
        self.log_x_checkbox.setEnabled(supports_log_x)
        self.log_x_checkbox.setToolTip(
            "Use a true base-10 x axis."
            if supports_log_x
            else "Log X requires every bin edge to be greater than zero."
        )
        if not supports_log_x and self.log_x_checkbox.isChecked():
            with QSignalBlocker(self.log_x_checkbox):
                self.log_x_checkbox.setChecked(False)

        self.summary_label.setText(self._histogram_summary())
        self.export_button.setEnabled(True)
        self._refresh_plot(emit=False)

    def set_presentation(
        self,
        *,
        y_values: str | None = None,
        log_x: bool | None = None,
        log_y: bool | None = None,
    ) -> None:
        """Synchronize controls without emitting a presentation-change signal."""

        with QSignalBlocker(self.y_values_combo):
            if y_values is not None:
                index = self.y_values_combo.findText(str(y_values))
                if index < 0:
                    raise ValueError(
                        "Histogram y values must be one of: "
                        + ", ".join(HISTOGRAM_Y_VALUE_OPTIONS)
                    )
                self.y_values_combo.setCurrentIndex(index)
        with QSignalBlocker(self.log_x_checkbox):
            if log_x is not None:
                if bool(log_x) and not self.log_x_checkbox.isEnabled():
                    raise ValueError(
                        "Log X is unavailable because histogram edges are not positive."
                    )
                self.log_x_checkbox.setChecked(bool(log_x))
        with QSignalBlocker(self.log_y_checkbox):
            if log_y is not None:
                self.log_y_checkbox.setChecked(bool(log_y))
        self._refresh_plot(emit=False)

    def request_export(self) -> None:
        """Prompt for PNG/TIFF output and export the current rendered plot."""

        if self._bin_edges.size == 0:
            return
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export histogram plot",
            "histogram.png",
            "PNG image (*.png);;TIFF image (*.tif *.tiff)",
        )
        if not path:
            return
        requested = Path(path)
        if not requested.suffix:
            requested = requested.with_suffix(
                ".tif" if "TIFF" in selected_filter else ".png"
            )
        try:
            exported = self.export_image(requested)
        except Exception as exc:
            QMessageBox.critical(self, "Histogram export failed", str(exc))
            return
        self.exportCompleted.emit(str(exported))

    def export_image(self, path: str | Path) -> Path:
        """Export the visible detailed plot at its current pixel dimensions."""

        if self._bin_edges.size == 0:
            raise ValueError("No histogram result is available to export.")
        target = Path(path)
        suffix = target.suffix.casefold()
        if suffix not in {".png", ".tif", ".tiff"}:
            raise ValueError("Histogram export supports PNG, TIF, and TIFF files.")
        target.parent.mkdir(parents=True, exist_ok=True)
        image = render_widget_image(self.plot)
        if suffix == ".png":
            if not image.save(str(target), "PNG"):
                raise OSError(f"Qt could not write PNG output to {target}.")
        else:
            tifffile.imwrite(
                target,
                qimage_rgb_array(image),
                photometric="rgb",
            )
        return target

    def refresh_theme(self, palette: QPalette | None = None) -> None:
        """Apply the current napari palette to this top-level window."""

        if getattr(self, "_theme_refresh_in_progress", False):
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
            self.setStyleSheet(
                "QDialog#VippHistogramDialog {"
                f" background: {colors.surface.name()}; color: {colors.text.name()};"
                "}"
                "QLabel {"
                f" color: {colors.text.name()};"
                "}"
            )
            self.plot.update()
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

    def _on_presentation_changed(self, *_args) -> None:
        self._refresh_plot(emit=True)

    def _on_grid_divisions_changed(self, *_args) -> None:
        self.plot.set_grid_divisions(
            x=self.x_grid_divisions_spinbox.value(),
            y=self.y_grid_divisions_spinbox.value(),
        )

    def _refresh_plot(self, *, emit: bool) -> None:
        if self._bin_edges.size == 0:
            return
        y_name = self.y_values_combo.currentText()
        column_name = _Y_VALUE_COLUMN[y_name]
        values = self._values_by_column[column_name]
        matrix = values.reshape(1, -1) if values.ndim == 1 else values
        visible = np.all(np.isfinite(matrix), axis=1)
        if not np.any(visible):
            self.plot.clear(
                f"No {y_name.casefold()} values are defined for this result."
            )
            return
        visible_values = matrix[visible]
        labels = (
            tuple(
                label
                for index, label in enumerate(self._series_labels)
                if visible[index]
            )
            if self._series_labels
            else None
        )
        colors = (
            tuple(
                color
                for index, color in enumerate(self._series_colors)
                if visible[index]
            )
            if self._series_colors
            else None
        )
        hover_details = {}
        for option, column in _Y_VALUE_COLUMN.items():
            if option == y_name:
                continue
            detail = self._values_by_column[column]
            detail_matrix = detail.reshape(1, -1) if detail.ndim == 1 else detail
            detail_matrix = detail_matrix[visible]
            if np.all(np.isfinite(detail_matrix)):
                hover_details[option] = detail_matrix
        self.plot.set_histogram(
            self._bin_edges,
            visible_values,
            title=self._title,
            x_axis_label=self._x_axis_label,
            y_axis_label=y_name,
            x_scale="log10" if self.log_x_checkbox.isChecked() else "linear",
            y_scale="log10" if self.log_y_checkbox.isChecked() else "linear",
            series_labels=labels,
            colors=colors,
            hover_details=hover_details,
        )
        if emit:
            self.presentationChanged.emit(
                y_name,
                self.log_x_checkbox.isChecked(),
                self.log_y_checkbox.isChecked(),
            )

    def _histogram_summary(self) -> str:
        if self._summary_override:
            return self._summary_override
        metadata = self._metadata
        if metadata is None:
            binned = int(np.rint(float(np.sum(self._values_by_column["count"]))))
            bin_count = max(int(self._bin_edges.size) - 1, 0)
            return f"{bin_count:,} bins · {binned:,} binned values"

        bin_count = _metadata_int(metadata, "bin_count", self._bin_edges.size - 1)
        binned = _metadata_int(
            metadata,
            "binned_value_count",
            int(np.rint(float(np.sum(self._values_by_column["count"])))),
        )
        finite = _metadata_int(metadata, "finite_value_count", binned)
        input_count = _metadata_int(metadata, "input_value_count", finite)
        nan_count = _metadata_int(metadata, "nan_value_count", 0)
        positive_infinite = _metadata_int(
            metadata,
            "positive_infinite_value_count",
            0,
        )
        negative_infinite = _metadata_int(
            metadata,
            "negative_infinite_value_count",
            0,
        )
        accounted_nonfinite = nan_count + positive_infinite + negative_infinite
        other_nonfinite = max(input_count - finite - accounted_nonfinite, 0)
        underflow = _metadata_int(metadata, "underflow_count", 0)
        overflow = _metadata_int(metadata, "overflow_count", 0)
        nonpositive = _metadata_int(metadata, "nonpositive_excluded_count", 0)
        spacing = str(getattr(metadata, "bin_spacing", "")).strip()
        spacing_text = f" · {spacing} spacing" if spacing else ""
        excluded_parts = [
            ("NaN", nan_count),
            ("+Inf", positive_infinite),
            ("-Inf", negative_infinite),
            ("non-finite", other_nonfinite),
            ("underflow", underflow),
            ("overflow", overflow),
            ("nonpositive", nonpositive),
        ]
        excluded_text = ", ".join(
            f"{label} {value:,}" for label, value in excluded_parts if value
        )
        ignored_text = f" · Excluded: {excluded_text}" if excluded_text else ""
        return (
            f"{bin_count:,} bins{spacing_text} · {input_count:,} input values · "
            f"{finite:,} finite · {binned:,} binned{ignored_text}"
        )


def histogram_arrays_from_table(
    table: TableData,
) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    tuple[str, ...],
    tuple[str, ...],
]:
    if not isinstance(table, TableData):
        raise TypeError("HistogramDialog expects a TableData histogram result.")
    required = (
        "bin_left",
        "bin_right",
        "count",
        "fraction",
        "density",
        "cumulative_count",
        "cumulative_fraction",
    )
    missing = [name for name in required if name not in table.columns]
    if missing:
        raise ValueError(
            "Histogram table is missing required columns: " + ", ".join(missing)
        )
    indices = {name: table.columns.index(name) for name in required}
    if not table.rows:
        empty = np.array([], dtype=np.float64)
        values = {name: empty.copy() for name in _Y_VALUE_COLUMN.values()}
        return np.array([], dtype=np.float64), values, (), ()

    has_series = "series_index" in table.columns
    if has_series:
        series_fields = ("series_name", "series_color")
        missing_series = [name for name in series_fields if name not in table.columns]
        if missing_series:
            raise ValueError(
                "Histogram series table is missing required columns: "
                + ", ".join(missing_series)
            )
        series_index_column = table.columns.index("series_index")
        series_name_column = table.columns.index("series_name")
        series_color_column = table.columns.index("series_color")
        series_indices = tuple(
            dict.fromkeys(int(row[series_index_column]) for row in table.rows)
        )
        if series_indices != tuple(range(len(series_indices))):
            raise ValueError(
                "Histogram series rows must use contiguous zero-based indices."
            )
        grouped_rows = tuple(
            tuple(
                row
                for row in table.rows
                if int(row[series_index_column]) == series_index
            )
            for series_index in series_indices
        )
    else:
        grouped_rows = (tuple(table.rows),)

    edge_vectors: list[np.ndarray] = []
    value_vectors: dict[str, list[np.ndarray]] = {
        name: [] for name in _Y_VALUE_COLUMN.values()
    }
    labels: list[str] = []
    colors: list[str] = []
    for rows in grouped_rows:
        left = np.asarray(
            [row[indices["bin_left"]] for row in rows],
            dtype=np.float64,
        )
        right = np.asarray(
            [row[indices["bin_right"]] for row in rows],
            dtype=np.float64,
        )
        if left.size > 1 and not np.allclose(
            left[1:],
            right[:-1],
            rtol=1e-10,
            atol=0.0,
        ):
            raise ValueError("Histogram table bins must be contiguous and ordered.")
        edge_vectors.append(np.concatenate((left[:1], right)))
        for name in value_vectors:
            value_vectors[name].append(
                np.asarray([row[indices[name]] for row in rows], dtype=np.float64)
            )
        if has_series:
            series_names = {str(row[series_name_column]).strip() for row in rows}
            series_colors = {str(row[series_color_column]).strip() for row in rows}
            if len(series_names) != 1 or len(series_colors) != 1:
                raise ValueError(
                    "Histogram series identity must be constant within each series."
                )
            labels.append(next(iter(series_names)) or f"Series {len(labels) + 1}")
            colors.append(next(iter(series_colors)))

    edges = edge_vectors[0]
    if any(
        candidate.shape != edges.shape
        or not np.allclose(candidate, edges, rtol=1e-10, atol=0.0)
        for candidate in edge_vectors[1:]
    ):
        raise ValueError("Every histogram series must use the same bin edges.")
    values = {
        name: vectors[0] if not has_series else np.stack(vectors, axis=0)
        for name, vectors in value_vectors.items()
    }
    _validate_histogram_arrays(edges, values)
    return edges, values, tuple(labels), tuple(colors)


def _histogram_arrays_from_values(
    *,
    bin_edges,
    count,
    fraction,
    density,
    cumulative_count,
    cumulative_fraction,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if bin_edges is None or count is None:
        raise ValueError("Extracted histogram data requires bin_edges and count.")
    edges = np.asarray(bin_edges, dtype=np.float64)
    counts = np.asarray(count, dtype=np.float64)
    if edges.ndim != 1 or counts.ndim != 1 or edges.size != counts.size + 1:
        raise ValueError(
            "Histogram edges must contain exactly one more value than count."
        )
    widths = np.diff(edges)
    total = float(np.sum(counts))
    resolved_fraction = (
        np.asarray(fraction, dtype=np.float64)
        if fraction is not None
        else counts / total
        if total > 0.0
        else np.zeros_like(counts)
    )
    resolved_density = (
        np.asarray(density, dtype=np.float64)
        if density is not None
        else np.divide(
            resolved_fraction,
            widths,
            out=np.zeros_like(resolved_fraction),
            where=widths > 0.0,
        )
    )
    resolved_cumulative_count = (
        np.asarray(cumulative_count, dtype=np.float64)
        if cumulative_count is not None
        else np.cumsum(counts)
    )
    resolved_cumulative_fraction = (
        np.asarray(cumulative_fraction, dtype=np.float64)
        if cumulative_fraction is not None
        else np.cumsum(resolved_fraction)
    )
    values = {
        "count": counts,
        "fraction": resolved_fraction,
        "density": resolved_density,
        "cumulative_count": resolved_cumulative_count,
        "cumulative_fraction": resolved_cumulative_fraction,
    }
    if any(value.ndim != 1 or value.size != counts.size for value in values.values()):
        raise ValueError("Every histogram value column must match the count vector.")
    _validate_histogram_arrays(edges, values)
    return edges, values


def _validate_histogram_arrays(
    edges: np.ndarray,
    values: dict[str, np.ndarray],
) -> None:
    if not np.all(np.isfinite(edges)):
        raise ValueError("Histogram bin edges must all be finite.")
    if not np.all(np.diff(edges) > 0.0):
        raise ValueError("Histogram bin edges must be strictly increasing.")
    count_matrix = np.asarray(values["count"], dtype=np.float64)
    if count_matrix.ndim == 1:
        count_matrix = count_matrix.reshape(1, -1)
    for name, column in values.items():
        matrix = np.asarray(column, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2 or matrix.shape != count_matrix.shape:
            raise ValueError(
                f"Histogram {name} values must match the count series and bins."
            )
        if np.any(matrix[np.isfinite(matrix)] < 0.0):
            raise ValueError(
                f"Histogram {name} values must be finite and non-negative."
            )
        invalid_rows = ~np.all(np.isfinite(matrix), axis=1)
        if not np.any(invalid_rows):
            continue
        normalized = name in {"fraction", "density", "cumulative_fraction"}
        empty_rows = np.sum(count_matrix, axis=1) == 0.0
        all_nan_rows = np.all(np.isnan(matrix), axis=1)
        if not normalized or np.any(invalid_rows & ~(empty_rows & all_nan_rows)):
            raise ValueError(
                f"Histogram {name} values must be finite and non-negative."
            )


def _metadata_int(metadata: object, name: str, default: int) -> int:
    try:
        return max(int(getattr(metadata, name)), 0)
    except (AttributeError, TypeError, ValueError):
        return max(int(default), 0)


def render_widget_image(widget: QWidget) -> QImage:
    """Render one widget to an RGB image at its current pixel dimensions."""

    width = max(int(widget.width()), 1)
    height = max(int(widget.height()), 1)
    image = QImage(width, height, QImage.Format_RGB888)
    image.fill(theme_colors(QWidget.palette(widget)).surface)
    painter = QPainter(image)
    widget.render(painter, QPoint())
    painter.end()
    return image


def qimage_rgb_array(image: QImage) -> np.ndarray:
    """Copy a QImage into a binding-independent contiguous RGB array."""

    rgb = image.convertToFormat(QImage.Format_RGB888)
    pointer = rgb.bits()
    byte_count = int(rgb.sizeInBytes())
    if hasattr(pointer, "setsize"):
        pointer.setsize(byte_count)
    raw = np.frombuffer(pointer, dtype=np.uint8, count=byte_count)
    rows = raw.reshape(int(rgb.height()), int(rgb.bytesPerLine()))
    return (
        rows[:, : int(rgb.width()) * 3]
        .reshape(
            int(rgb.height()),
            int(rgb.width()),
            3,
        )
        .copy()
    )


__all__ = [
    "HISTOGRAM_Y_VALUE_OPTIONS",
    "HistogramDialog",
    "qimage_rgb_array",
    "render_widget_image",
]
