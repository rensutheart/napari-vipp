"""Measurement plotting controls shared by the inspector and nonmodal window.

The owning workflow remains the only recipe authority: edits emit complete
parameters, and set_state applies its prepared result without emitting edits.
"""

from __future__ import annotations

from pathlib import Path
from threading import Event

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from qtpy.QtCore import QAbstractTableModel, QObject, Qt, QThread, Signal, Slot
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from napari_vipp.core.measurement_export import measurement_export_destination_revisions
from napari_vipp.core.plot_rendering import (
    build_plot_figure,
    export_plot_result,
    plot_export_targets,
)
from napari_vipp.core.progress import OperationCancelled
from napari_vipp.core.result_plots import PlotRecipe, numeric_columns
from napari_vipp.ui.dialog_buttons import add_dialog_buttons
from napari_vipp.ui.palette_roles import theme_colors


def _label(text, parent=None):
    widget = QLabel(text, parent)
    widget.setWordWrap(True)
    widget.setTextFormat(Qt.PlainText)
    widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
    return widget


class PlotRecipeControls(QWidget):
    """Narrow-friendly mappings and appearance editor without calculations."""

    params_changed = Signal(dict)
    layout_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._updating = False
        self._params = PlotRecipe().to_params()
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Maximum)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.form = QFormLayout()
        self.form.setRowWrapPolicy(QFormLayout.WrapAllRows)
        self.form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        layout.addLayout(self.form)
        self.controls = {}
        self._labels = {}
        options = (
            ("plot_type", "Plot type", ("Compare groups", "Distribution", "Scatter")),
            ("y_column", "Measurement", ()),
            ("x_column", "X measurement", ()),
            ("group_column", "Group by", ()),
            ("point_unit", "Each point represents", ("Objects", "Mean per image")),
            ("image_column", "Image identity column", ()),
            ("summary", "Summary line", ("Mean", "Median", "None")),
            ("distribution", "Distribution view", ("Histogram", "Cumulative")),
            ("normalization", "Histogram height", ("Count", "Percent")),
        )
        for key, title, values in options:
            combo = QComboBox(self)
            combo.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
            combo.setMinimumContentsLength(1)
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setAccessibleName(title)
            for value in values:
                combo.addItem(value, value)
            self._add_control(key, title, combo)
            combo.currentIndexChanged.connect(self._changed)
        self.controls["point_unit"].setToolTip(
            "Objects uses one measurement row per point. Mean per image gives "
            "each image equal weight, after averaging its eligible rows. Neither "
            "choice automatically establishes independent biological samples."
        )
        bins = QSpinBox(self)
        bins.setRange(2, 512)
        bins.setKeyboardTracking(False)
        bins.valueChanged.connect(self._changed)
        self._add_control("bins", "Shared histogram bins", bins)
        self.note = _label(
            "Objects in one image are not automatically independent biological "
            "samples. These plots describe the connected measurements."
        )
        layout.addWidget(self.note)
        self.appearance_button = QPushButton("Appearance ▸", self)
        self.appearance_button.setCheckable(True)
        layout.addWidget(self.appearance_button)
        appearance = QGroupBox(self)
        appearance_layout = QFormLayout(appearance)
        appearance_layout.setRowWrapPolicy(QFormLayout.WrapAllRows)
        title_edit = QLineEdit(self)
        title_edit.setPlaceholderText("Automatic title")
        title_edit.setMaxLength(200)
        title_edit.editingFinished.connect(self._changed)
        self.controls["title"] = title_edit
        appearance_layout.addRow("Title", title_edit)
        size = QDoubleSpinBox(self)
        size.setRange(1, 20)
        size.setSingleStep(0.5)
        size.setDecimals(1)
        size.setKeyboardTracking(False)
        size.valueChanged.connect(self._changed)
        self.controls["point_size"] = size
        appearance_layout.addRow("Point size", size)
        for key, title in (
            ("log_x", "Log X axis"),
            ("log_y", "Log Y axis"),
            ("show_grid", "Show grid"),
        ):
            checkbox = QCheckBox(title, self)
            checkbox.toggled.connect(self._changed)
            self.controls[key] = checkbox
            appearance_layout.addRow(checkbox)
        self.controls["log_x"].setToolTip(
            "A log axis excludes non-positive values. Exclusions are reported."
        )
        self.controls["log_y"].setToolTip(
            "Log measurements exclude non-positive values. Exclusions are reported."
        )
        layout.addWidget(appearance)
        appearance.hide()
        self.appearance_button.toggled.connect(appearance.setVisible)
        self.appearance_button.toggled.connect(
            lambda expanded: self.appearance_button.setText(
                "Appearance ▾" if expanded else "Appearance ▸"
            )
        )
        self.appearance_button.toggled.connect(
            lambda _expanded: self.layout_changed.emit()
        )
        layout.addStretch(1)
        self.set_state(None, self._params)

    def _add_control(self, key, title, widget):
        label = _label(title)
        label.setWordWrap(False)
        self._labels[key] = label
        self.controls[key] = widget
        self.form.addRow(label, widget)

    def set_state(self, table, params):
        self._updating = True
        try:
            self._params = {**PlotRecipe().to_params(), **(params or {})}
            numeric = tuple(numeric_columns(table)) if table is not None else ()
            columns = table.columns if table is not None else ()
            for key in ("y_column", "x_column", "group_column", "image_column"):
                combo = self.controls[key]
                combo.clear()
                if key in {"y_column", "x_column"}:
                    combo.addItem("Automatic measurement", "auto")
                    candidates = numeric
                else:
                    combo.addItem(
                        "None" if key == "group_column" else "Choose image identity",
                        "",
                    )
                    candidates = columns
                for column in candidates:
                    unit = table.unit_for(column)
                    title = column.replace("_", " ").strip()
                    combo.addItem(f"{title} ({unit})" if unit else title, column)
                selected = self._params[key]
                if combo.findData(selected) < 0 and selected:
                    combo.addItem(f"Unavailable: {selected}", selected)
            for key, widget in self.controls.items():
                value = self._params.get(key)
                if isinstance(widget, QComboBox):
                    widget.setCurrentIndex(widget.findData(value))
                elif isinstance(widget, QCheckBox):
                    widget.setChecked(bool(value))
                elif isinstance(widget, QLineEdit):
                    widget.setText(str(value or ""))
                else:
                    widget.setValue(value)
            self._update_visibility()
        finally:
            self._updating = False

    def _changed(self, *_args):
        if self._updating:
            return
        params = dict(self._params)
        for key, widget in self.controls.items():
            if isinstance(widget, QComboBox):
                params[key] = widget.currentData()
            elif isinstance(widget, QCheckBox):
                params[key] = widget.isChecked()
            elif isinstance(widget, QLineEdit):
                params[key] = widget.text()
            else:
                params[key] = widget.value()
        self._params = params
        self._update_visibility()
        if params["point_unit"] == "Mean per image" and not params["image_column"]:
            self.note.setText(
                "Choose the image identity column to prepare one mean per image."
            )
            return
        self.note.setText(
            "Objects in one image are not automatically independent biological "
            "samples. These plots describe the connected measurements."
        )
        self.params_changed.emit(params)

    def _update_visibility(self):
        mode = self._params["plot_type"]
        histogram = (
            mode == "Distribution" and self._params["distribution"] == "Histogram"
        )
        visibility = {
            "x_column": mode == "Scatter",
            "image_column": self._params["point_unit"] == "Mean per image",
            "summary": mode == "Compare groups",
            "distribution": mode == "Distribution",
            "normalization": histogram,
            "bins": histogram,
        }
        for key, visible in visibility.items():
            self.controls[key].setVisible(visible)
            self._labels[key].setVisible(visible)
        self.controls["log_x"].setVisible(mode != "Compare groups")
        self.controls["point_size"].setEnabled(mode != "Distribution")
        self.layout_changed.emit()


class _PreparedTableModel(QAbstractTableModel):
    """Virtualized access to complete prepared rows; no QTableWidget copying."""

    def __init__(self, table, parent=None):
        super().__init__(parent)
        self.table = table

    def rowCount(self, parent=None):  # noqa: N802
        return 0 if parent is not None and parent.isValid() else self.table.row_count

    def columnCount(self, parent=None):  # noqa: N802
        return 0 if parent is not None and parent.isValid() else self.table.column_count

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid() and role == Qt.DisplayRole:
            value = self.table.rows[index.row()][index.column()]
            return "" if value is None else str(value)
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):  # noqa: N802
        if role != Qt.DisplayRole:
            return None
        return (
            self.table.columns[section]
            if orientation == Qt.Horizontal
            else str(section + 1)
        )


class ResultPlotCanvas(QWidget):
    point_selected = Signal(str)

    def __init__(self, parent=None, *, compact=False):
        super().__init__(parent)
        self.compact = compact
        self.result = None
        self.canvas = None
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.placeholder = _label(
            "Calculate the connected measurements to prepare this plot.", self
        )
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.placeholder)
        self.setMinimumHeight(260 if compact else 320)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)

    def set_result(self, result):
        if result is self.result and self.canvas is not None:
            return
        self.result = result
        if self.canvas is not None:
            old = self.canvas
            self.layout.removeWidget(old)
            old.figure.clear()
            old.close()
            old.deleteLater()
            self.canvas = None
        self.placeholder.setVisible(result is None)
        if result is None:
            return
        colors = theme_colors(self.palette())
        figure = build_plot_figure(
            result,
            display_only=True,
            compact=self.compact,
            size_inches=(4, 3) if self.compact else (7, 5),
            colors={
                "background": colors.surface.name(),
                "text": colors.text.name(),
                "grid": colors.border.name(),
            },
        )
        self.canvas = FigureCanvasQTAgg(figure)
        self.canvas.setMinimumWidth(0)
        self.canvas.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        self.canvas.mpl_connect("pick_event", self._picked)
        self.layout.addWidget(self.canvas)
        self.canvas.draw_idle()

    def _picked(self, event):
        if self.result is None or not len(event.ind):
            return
        source_rows = getattr(event.artist, "vipp_source_rows", ())
        index = int(event.ind[0])
        if index >= len(source_rows):
            return
        indices = source_rows[index]
        table = self.result.source_table
        if not indices:
            return
        row = table.rows[indices[0]]
        identities = [
            f"{name}: {row[position]}"
            for position, name in enumerate(table.columns)
            if "label" in name.casefold()
            or "image" in name.casefold()
            or "source" in name.casefold()
            or name.casefold() in {"object_id", "_vipp_item_key"}
        ]
        message = " · ".join(identities) or f"Measurement row {indices[0] + 1}"
        if len(indices) > 1:
            message = f"Image mean from {len(indices):,} rows · " + message
        self.point_selected.emit(message)


def _result_summary(result):
    count = result.counts
    unit = (
        "individual measurements"
        if result.recipe.point_unit == "Objects"
        else "image means"
    )
    parts = [
        f"{count.plotted_points:,} {unit}",
        f"{count.eligible_rows:,} of {count.input_rows:,} rows eligible",
    ]
    if count.displayed_points < count.plotted_points:
        parts.append(
            f"{count.displayed_points:,} points shown; summaries and exports use all"
        )
    exclusions = []
    for name, value in (
        ("missing", count.missing_rows),
        ("non-finite", count.nonfinite_rows),
        ("not numeric", count.nonnumeric_rows),
    ):
        if value:
            exclusions.append(f"{value:,} {name}")
    if count.nonpositive_rows:
        if result.recipe.point_unit == "Mean per image":
            exclusions.append(
                f"{count.nonpositive_points:,} non-positive image means "
                f"({count.nonpositive_rows:,} contributing rows)"
            )
        else:
            exclusions.append(f"{count.nonpositive_rows:,} non-positive for log axes")
    if exclusions:
        parts.append("Excluded: " + ", ".join(exclusions))
    return " · ".join(parts)


class PlotResultsPanel(QWidget):
    """Inspector plus an optional pop-out, bound to one workflow node."""

    params_changed = Signal(dict)
    export_completed = Signal(str)
    layout_changed = Signal()

    def __init__(self, table=None, params=None, result=None, parent=None):
        super().__init__(parent)
        self.table = table
        self.params = params or PlotRecipe().to_params()
        self.result = result
        self.stale = False
        self.dialog = None
        self.protected_paths = ()
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.controls = PlotRecipeControls(self)
        self.controls.params_changed.connect(self._edited)
        self.controls.layout_changed.connect(self.layout_changed.emit)
        layout.addWidget(self.controls)
        self.plot = ResultPlotCanvas(self, compact=True)
        self.plot.setFixedHeight(280)
        layout.addWidget(self.plot)
        self.summary = _label("")
        layout.addWidget(self.summary)
        self.open_button = QPushButton("Open plot…", self)
        self.open_button.clicked.connect(self.open_plot)
        layout.addWidget(self.open_button)
        layout.addStretch(1)
        self.set_state(table=table, params=params, result=result)

    def set_state(
        self,
        *,
        table=None,
        params=None,
        result=None,
        stale=False,
        message="",
        protected_paths=None,
    ):
        self.table = (
            table
            if table is not None
            else (result.source_table if result is not None else None)
        )
        self.params = params or (
            result.recipe.to_params() if result is not None else self.params
        )
        self.result = result
        self.stale = stale
        if protected_paths is not None:
            self.protected_paths = tuple(protected_paths)
        self.controls.set_state(self.table, self.params)
        self.plot.set_result(result)
        self.summary.setText(
            message
            or (
                "Plot is out of date. Calculate again before exporting."
                if stale
                else _result_summary(result)
                if result is not None
                else "Connect a measurement table and calculate this node."
            )
        )
        self.open_button.setEnabled(result is not None)
        if self.dialog is not None:
            self.dialog.set_state()

    def set_result(self, result, *, stale=False):
        self.set_state(table=self.table, params=self.params, result=result, stale=stale)

    def set_stale(
        self, message="Plot is out of date. Calculate again before exporting."
    ):
        self.set_state(
            table=self.table,
            params=self.params,
            result=self.result,
            stale=True,
            message=message,
        )

    def _edited(self, params):
        self.params = params
        self.set_stale()
        self.params_changed.emit(params)

    def open_plot(self):
        if self.result is None:
            return None
        if self.dialog is None:
            self.dialog = PlotResultsDialog(self)
        self.dialog.set_state()
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()
        return self.dialog

    def close_plot(self):
        if self.dialog is not None:
            self.dialog.close()


class PlotResultsDialog(QDialog):
    """Large live recipe editor with full data inspection and publication export."""

    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.setWindowTitle("Plot Results — VIPP")
        self.setModal(False)
        self.setSizeGripEnabled(True)
        self.resize(1120, 760)
        layout = QVBoxLayout(self)
        title = _label("Plot Results")
        font = title.font()
        font.setBold(True)
        font.setPointSize(font.pointSize() + 3)
        title.setFont(font)
        layout.addWidget(title)
        self.summary = _label("")
        layout.addWidget(self.summary)
        splitter = QSplitter(Qt.Horizontal, self)
        plot_side = QWidget(self)
        plot_layout = QVBoxLayout(plot_side)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        self.plot = ResultPlotCanvas(self)
        plot_layout.addWidget(self.plot, 1)
        self.point_label = _label(
            "Click a point to identify its object label or source image."
        )
        self.plot.point_selected.connect(self.point_label.setText)
        plot_layout.addWidget(self.point_label)
        self.data_view = QTableView(self)
        self.data_view.setMinimumHeight(150)
        self.data_view.setAlternatingRowColors(True)
        self.data_view.hide()
        plot_layout.addWidget(self.data_view)
        splitter.addWidget(plot_side)
        self.controls = PlotRecipeControls(self)
        self.controls.params_changed.connect(owner._edited)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.controls)
        scroll.setMinimumWidth(230)
        scroll.setMaximumWidth(400)
        splitter.addWidget(scroll)
        splitter.setSizes([790, 300])
        splitter.setStretchFactor(0, 1)
        layout.addWidget(splitter, 1)
        self.warning = _label("")
        layout.addWidget(self.warning)
        row = QHBoxLayout()
        self.data_button = QPushButton("View plotted data", self)
        self.data_button.setCheckable(True)
        self.data_button.toggled.connect(self.data_view.setVisible)
        row.addWidget(self.data_button)
        row.addStretch(1)
        self.export_button = QPushButton("Export figure…", self)
        self.export_button.clicked.connect(self.request_export)
        close = QPushButton("Close", self)
        close.clicked.connect(self.close)
        add_dialog_buttons(row, actions=(self.export_button,), dismiss=close)
        for button in (self.data_button, self.export_button, close):
            button.setAutoDefault(False)
        layout.addLayout(row)

    def set_state(self):
        owner = self.owner
        self.controls.set_state(owner.table, owner.params)
        self.plot.set_result(owner.result)
        self.summary.setText(owner.summary.text())
        self.export_button.setEnabled(owner.result is not None and not owner.stale)
        self.data_button.setEnabled(owner.result is not None)
        warnings = (
            tuple(
                warning
                for warning in owner.result.warnings
                if "independent biological sample" not in warning
            )
            if owner.result is not None
            else ()
        )
        # The unit caveat is already beside the point-unit selector.
        self.warning.setText("\n".join(warnings))
        self.warning.setVisible(bool(warnings))
        if owner.result is not None:
            old = self.data_view.model()
            self.data_view.setModel(
                _PreparedTableModel(owner.result.plotted_table, self.data_view)
            )
            if old is not None:
                old.deleteLater()
        self.point_label.setText(
            "Click a point to identify its object label or source image."
        )

    def request_export(self):
        if self.owner.result is None or self.owner.stale:
            return
        result = self.owner.result
        dialog = PlotExportDialog(
            result,
            self,
            protected_paths=self.owner.protected_paths,
            is_current=lambda: self.owner.result is result and not self.owner.stale,
        )
        if dialog.exec() == QDialog.Accepted and dialog.exported is not None:
            self.owner.export_completed.emit(str(dialog.exported.paths[0]))


class _FigureExportWorker(QObject):
    finished = Signal(object, object)

    def __init__(self, result, path, kwargs):
        super().__init__()
        self.result = result
        self.path = path
        self.kwargs = kwargs

    @Slot()
    def run(self):
        try:
            exported = export_plot_result(self.result, self.path, **self.kwargs)
        except Exception as exc:
            self.finished.emit(None, exc)
        else:
            self.finished.emit(exported, None)


class PlotExportDialog(QDialog):
    """Explicit physical dimensions, independent of window geometry."""

    def __init__(self, result, parent=None, *, protected_paths=(), is_current=None):
        super().__init__(parent)
        self.result = result
        self.protected_paths = protected_paths
        self.is_current = is_current
        self.exported = None
        self._thread = None
        self._worker = None
        self._cancel = Event()
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._application_quitting)
        self.setWindowTitle("Export figure")
        self.resize(450, 430)
        layout = QVBoxLayout(self)
        layout.addWidget(_label("Save a publication figure", self))
        form = QFormLayout()
        self.format_combo = QComboBox(self)
        self.format_combo.addItems(["PNG", "TIFF", "SVG", "PDF"])
        form.addRow("Format", self.format_combo)
        self.width_spin = QDoubleSpinBox(self)
        self.height_spin = QDoubleSpinBox(self)
        for widget, value in ((self.width_spin, 160), (self.height_spin, 110)):
            widget.setRange(20, 500)
            widget.setValue(value)
            widget.setSuffix(" mm")
            widget.setDecimals(1)
            widget.setKeyboardTracking(False)
        form.addRow("Width", self.width_spin)
        form.addRow("Height", self.height_spin)
        self.dpi = QSpinBox(self)
        self.dpi.setRange(72, 1200)
        self.dpi.setValue(300)
        form.addRow("Resolution", self.dpi)
        layout.addLayout(form)
        self.size_hint = _label("")
        layout.addWidget(self.size_hint)
        self.sidecars = QCheckBox("Include plotted data and settings", self)
        layout.addWidget(self.sidecars)
        layout.addWidget(
            _label(
                "Saves a CSV and a plot-settings JSON beside the figure. No source "
                "images are read or included. CSV is plain text; import identifier "
                "columns as text in spreadsheets."
            )
        )
        layout.addWidget(
            _label(
                "Figures use a light background and all eligible data. Output size "
                "is independent of this window. There are no statistical tests "
                "or significance annotations."
            )
        )
        self.progress = QProgressBar(self)
        self.progress.setRange(0, 0)
        self.progress.hide()
        layout.addWidget(self.progress)
        self.progress_label = _label("")
        self.progress_label.hide()
        layout.addWidget(self.progress_label)
        row = QHBoxLayout()
        row.addStretch(1)
        cancel = QPushButton("Cancel", self)
        cancel.clicked.connect(self.reject)
        self.cancel_button = cancel
        self.save_button = QPushButton("Choose file and export…", self)
        self.save_button.clicked.connect(self._save)
        self.save_button.setDefault(True)
        add_dialog_buttons(row, actions=(self.save_button,), dismiss=cancel)
        layout.addLayout(row)
        for control in (self.width_spin, self.height_spin, self.dpi):
            control.valueChanged.connect(self._size_changed)
        self.format_combo.currentIndexChanged.connect(self._size_changed)
        self._size_changed()

    def _size_changed(self, *_args):
        raster = self.format_combo.currentText() in {"PNG", "TIFF"}
        self.dpi.setEnabled(raster)
        self.size_hint.setText(
            f"{int(self.width_spin.value() / 25.4 * self.dpi.value()):,} × "
            f"{int(self.height_spin.value() / 25.4 * self.dpi.value()):,} pixels"
            if raster
            else "Vector figure: sharp at any display scale."
        )

    def _save(self):
        suffix = {"PNG": ".png", "TIFF": ".tiff", "SVG": ".svg", "PDF": ".pdf"}[
            self.format_combo.currentText()
        ]
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export measurement figure",
            f"measurements{suffix}",
            f"{self.format_combo.currentText()} figure (*{suffix})",
        )
        if not path:
            return
        target = Path(path)
        if not target.suffix:
            target = target.with_suffix(suffix)
        if target.suffix.lower() != suffix:
            QMessageBox.warning(
                self,
                "Choose a matching format",
                f"The selected format requires {suffix}.",
            )
            return
        try:
            if self.is_current is not None and not self.is_current():
                raise ValueError(
                    "The plot changed. Calculate it again before exporting."
                )
            targets = plot_export_targets(
                target, include_data=self.sidecars.isChecked()
            )
            revisions = measurement_export_destination_revisions(targets)
            existing = [path.name for path in targets if path.exists()]
            if (
                existing
                and QMessageBox.question(
                    self,
                    "Replace export files?",
                    "Replace these existing export files?\n" + "\n".join(existing),
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                != QMessageBox.Yes
            ):
                return
            kwargs = dict(
                width_mm=self.width_spin.value(),
                height_mm=self.height_spin.value(),
                dpi=self.dpi.value(),
                include_data=self.sidecars.isChecked(),
                protected_paths=self.protected_paths,
                overwrite=bool(existing),
                expected_destination_revisions=revisions,
                is_current=self.is_current,
                cancellation=self._cancel,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Figure export failed", str(exc))
            return
        self._start_export(target, kwargs)

    def _start_export(self, target, kwargs):
        self._cancel.clear()
        self._thread = QThread(self)
        self._worker = _FigureExportWorker(self.result, target, kwargs)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._export_finished)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread_stopped)
        self.progress.show()
        self.progress_label.setText("Rendering all eligible points at the chosen size…")
        self.progress_label.show()
        self.save_button.setEnabled(False)
        for control in (
            self.format_combo,
            self.width_spin,
            self.height_spin,
            self.dpi,
            self.sidecars,
        ):
            control.setEnabled(False)
        self._thread.start()

    def _export_finished(self, exported, error):
        self._outcome = (exported, error)

    def _thread_stopped(self):
        thread = self._thread
        self._thread = None
        self._worker = None
        thread.deleteLater()
        exported, error = self._outcome
        self.progress.hide()
        self.progress_label.hide()
        if exported is not None:
            self.exported = exported
            self.accept()
        elif isinstance(error, OperationCancelled):
            super().reject()
        else:
            self.save_button.setEnabled(True)
            for control in (
                self.format_combo,
                self.width_spin,
                self.height_spin,
                self.dpi,
                self.sidecars,
            ):
                control.setEnabled(True)
            self.cancel_button.setText("Cancel")
            self._size_changed()
            QMessageBox.critical(self, "Figure export failed", str(error))

    def reject(self):
        if self._thread is not None:
            self._cancel.set()
            self.cancel_button.setText("Cancelling…")
            self.progress_label.setText(
                "Cancelling at the next safe boundary; no partial figure "
                "will be published."
            )
            return
        super().reject()

    def closeEvent(self, event):  # noqa: N802
        if self._thread is not None:
            self.reject()
            event.ignore()
        else:
            super().closeEvent(event)

    def _application_quitting(self):
        if self._thread is not None:
            self._cancel.set()
            self._thread.quit()
            # Figure rendering itself is not interruptible. Drain it at app
            # shutdown so Qt cannot destroy a running QThread. The following
            # cancellation checkpoint prevents publication of a partial job.
            self._thread.wait()
