"""Measurement plotting controls shared by the inspector and nonmodal window.

The owning workflow remains the only recipe authority: edits emit complete
parameters, and set_state applies its prepared result without emitting edits.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import fill
from threading import Event

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.ticker import MaxNLocator
from qtpy.QtCore import (
    QAbstractTableModel,
    QEvent,
    QObject,
    QSize,
    Qt,
    QThread,
    QTimer,
    Signal,
    Slot,
)
from qtpy.QtGui import QPalette
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
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
from napari_vipp.core.result_plots import (
    PlotRecipe,
    is_summary_table,
    measurement_label,
    numeric_columns,
)
from napari_vipp.ui.dialog_buttons import add_dialog_buttons
from napari_vipp.ui.iconography import interface_icon, palette_branch_color
from napari_vipp.ui.palette_roles import theme_colors
from napari_vipp.ui.table_display import (
    DEFAULT_DECIMAL_PLACES,
    TableDecimalControls,
    format_table_value,
    validate_decimal_places,
)


def _label(text, parent=None):
    widget = QLabel(text, parent)
    widget.setWordWrap(True)
    widget.setTextFormat(Qt.PlainText)
    widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
    return widget


def _retain_hidden_space(widget):
    policy = widget.sizePolicy()
    policy.setRetainSizeWhenHidden(True)
    widget.setSizePolicy(policy)


def _summary_slot(layout):
    """Reserve the larger of the current and updating summaries, including wraps."""
    slot = QWidget()
    slot.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
    grid = QGridLayout(slot)
    grid.setContentsMargins(0, 0, 0, 0)
    current, alternate = _label(""), _label("")
    _retain_hidden_space(alternate)
    grid.addWidget(current, 0, 0)
    grid.addWidget(alternate, 0, 0)
    alternate.hide()
    layout.addWidget(slot)
    return current, alternate


def _set_plot_warnings(widget, result, *, stale=False):
    # A transient update should hide outdated advice without collapsing its row.
    # Keep its text too: an empty QLabel has a different hidden size hint.
    if stale:
        widget.hide()
        return
    # The biological-replication caveat is already beside the point-unit selector.
    warnings = (
        tuple(
            warning
            for warning in result.warnings
            if "independent biological sample" not in warning
        )
        if result is not None and not stale
        else ()
    )
    widget.setText("\n".join(warnings))
    policy = widget.sizePolicy()
    policy.setRetainSizeWhenHidden(bool(warnings))
    widget.setSizePolicy(policy)
    widget.setVisible(bool(warnings))
    widget.refresh_theme()


class _PlotWarningLabel(QLabel):
    """Readable wrapped warning, including inside the fixed-height inspector form."""

    layout_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._styling = False
        self._fitting = False
        self._theme_owner = None
        self.setWordWrap(True)
        self.setTextFormat(Qt.PlainText)
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.setMargin(8)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Minimum)
        self.refresh_theme()

    def refresh_theme(self):
        if self._styling:
            return
        self._styling = True
        try:
            # A stylesheet changes this label's palette. Blending that palette
            # again on every refresh eventually turns the whole box bright orange.
            owner = self.parentWidget()
            if owner is not self._theme_owner:
                if self._theme_owner is not None:
                    self._theme_owner.removeEventFilter(self)
                self._theme_owner = owner
                if owner is not None:
                    owner.installEventFilter(self)
            colors = theme_colors(owner.palette() if owner else QApplication.palette())
            tone = colors.warning
            style = (
                f"QLabel {{ background-color: {tone.surface.name()}; "
                f"color: {colors.text.name()}; "
                f"border-left: 3px solid {tone.accent.name()}; padding: 0; }}"
            )
            if self.styleSheet() != style:
                self.setStyleSheet(style)
        finally:
            self._styling = False
        self._fit_height()

    def _fit_height(self):
        if not self.wordWrap() or self._fitting:
            return
        previous = self.minimumHeight()
        self._fitting = True
        try:
            # QLabel's heightForWidth includes its existing minimum. Reset it
            # before measuring so widening the inspector can shrink the box too.
            self.setMinimumHeight(0)
            height = self.heightForWidth(max(1, self.width())) if self.text() else 0
            self.setMinimumHeight(max(0, height))
        finally:
            self._fitting = False
        if self.minimumHeight() != previous:
            self.updateGeometry()
            self.layout_changed.emit()

    def setText(self, text):  # noqa: N802
        super().setText(text)
        self.setAccessibleDescription(text)
        self._fit_height()

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        if event.size().width() != event.oldSize().width():
            self._fit_height()

    def eventFilter(self, watched, event):  # noqa: N802
        # Once styled, a QLabel may no longer inherit its parent's palette-change
        # event. Observe the unstyled owner explicitly for live theme changes.
        if watched is self._theme_owner and event.type() in (
            QEvent.PaletteChange,
            QEvent.ApplicationPaletteChange,
            QEvent.StyleChange,
        ):
            self.refresh_theme()
        return super().eventFilter(watched, event)

    def changeEvent(self, event):  # noqa: N802
        super().changeEvent(event)
        if getattr(self, "_styling", True):
            return
        if event.type() in (QEvent.PaletteChange, QEvent.ApplicationPaletteChange):
            self.refresh_theme()
        elif event.type() in (QEvent.FontChange, QEvent.StyleChange):
            self._fit_height()


class _PlotBusyIndicator(QWidget):
    """Small indeterminate progress row; never imply a completion percentage."""

    def __init__(self, parent=None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        self.progress = QProgressBar(self)
        self.progress.setTextVisible(False)
        self.progress.setFixedSize(56, 6)
        self.progress.setAccessibleName("Plot update in progress")
        row.addWidget(self.progress)
        self.label = _label("Updating plot…", self)
        self.label.setWordWrap(False)
        self.label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        row.addWidget(self.label, 1)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        _retain_hidden_space(self)
        self.set_busy(False)

    def set_busy(self, busy, message=""):
        self.label.setText(message or "Updating plot…")
        self.label.setToolTip(self.label.text())
        self.setAccessibleName(self.label.text() if busy else "")
        self.progress.setRange(0, 0 if busy else 1)
        if not busy:
            self.progress.setValue(0)
        self.setVisible(busy)


class _AxisIntervalControl(QWidget):
    """No spin-box clamping: preserve explicit decimal/scientific notation."""

    changed = Signal()

    def __init__(self, axis, parent=None):
        super().__init__(parent)
        self._updating = False
        self._manual_allowed = True
        self.axis = axis
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        row = QHBoxLayout()
        self.auto = QCheckBox("Auto", self)
        self.auto.setAccessibleName(f"Automatic {axis} axis label interval")
        self.edit = QLineEdit("1", self)
        self.edit.setMinimumWidth(0)
        self.edit.setPlaceholderText("e.g. 10 or 0.5")
        self.edit.setAccessibleName(f"{axis} axis label interval")
        row.addWidget(self.auto)
        row.addWidget(self.edit, 1)
        layout.addLayout(row)
        self.hint = _label("")
        layout.addWidget(self.hint)
        self.auto.toggled.connect(self._changed)
        self.edit.editingFinished.connect(self._changed)
        tooltip = (
            "Auto chooses readable axis labels for the current range. Turn it "
            "off to enter the numeric gap between labels and major grid lines, "
            "for example 10 gives 0, 10, 20, … . Decimal and scientific notation "
            "are accepted. Use a positive finite number; integer counts require "
            "whole-number intervals. This changes display only, not histogram "
            "bins or measurements, and is also used for exported figures."
        )
        self.setToolTip(tooltip)
        self.auto.setToolTip(tooltip)
        self.edit.setToolTip(tooltip)
        self.setValue("Auto")

    def value(self):
        return "Auto" if self.auto.isChecked() else self.edit.text().strip()

    def setValue(self, value):  # noqa: N802
        self._updating = True
        try:
            automatic = isinstance(value, str) and value.strip().lower() == "auto"
            self.auto.setChecked(automatic)
            if not automatic:
                self.edit.setText(str(value))
            self._enabled_state()
        finally:
            self._updating = False

    def set_axis_info(self, *, manual_allowed, hint):
        self._manual_allowed = manual_allowed
        self.hint.setText(hint)
        self._enabled_state()

    def _enabled_state(self):
        # Imported incompatible recipes remain visible and can be repaired by
        # explicitly checking Auto, rather than being silently rewritten.
        self.auto.setEnabled(self._manual_allowed or not self.auto.isChecked())
        self.edit.setEnabled(self._manual_allowed and not self.auto.isChecked())

    def _changed(self, *_args):
        self._enabled_state()
        if not self._updating:
            self.changed.emit()


class PlotRecipeControls(QWidget):
    """Narrow-friendly mappings and appearance editor without calculations."""

    params_changed = Signal(dict)
    layout_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._updating = False
        self._params = PlotRecipe().to_params()
        self._table = None
        self.setObjectName("PlotRecipeControls")
        # Let these controls sit naturally on the inspector or workspace
        # sidebar. Editors retain their normal themed input surfaces.
        self.setStyleSheet(
            "QWidget#PlotRecipeControls, QWidget#PlotAppearance, "
            "QWidget#PlotAxisInterval, QLabel, QCheckBox { background: transparent; }"
            "QPushButton#PlotAppearanceHeading { text-align: left; "
            "background: transparent; border: none; "
            "border-top: 1px solid palette(button); padding: 8px 0 2px 0; }"
            "QPushButton#PlotAppearanceHeading:hover { color: palette(highlight); }"
            "QPushButton#PlotAppearanceHeading:focus { "
            "border-bottom: 1px solid palette(highlight); }"
        )
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Maximum)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.settings_heading = _label("Plot settings", self)
        heading_font = self.settings_heading.font()
        heading_font.setBold(True)
        self.settings_heading.setFont(heading_font)
        layout.addWidget(self.settings_heading)
        # Each label uses the full sidebar width. QFormLayout constrains
        # wrapped labels to a short size hint even with WrapAllRows enabled.
        self.form = QVBoxLayout()
        self.form.setContentsMargins(0, 0, 0, 0)
        self.form.setSpacing(6)
        self.form.setAlignment(Qt.AlignTop)
        layout.addLayout(self.form)
        self.controls = {}
        self._labels = {}
        self._appearance_labels = {}
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
        self.controls["group_column"].setToolTip(
            "With Mean per image, every object in an image must have the same "
            "group value. Choose an image-level category such as treatment, "
            "or None. Use Objects to group individual object measurements."
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
        self.appearance_button = QPushButton("Appearance", self)
        self.appearance_button.setIconSize(QSize(18, 18))
        self.appearance_button.setObjectName("PlotAppearanceHeading")
        self.appearance_button.setFont(heading_font)
        self.appearance_button.setCheckable(True)
        self.appearance_button.setToolTip(
            "Show or hide the plot title, point size, grid and axis settings."
        )
        layout.addWidget(self.appearance_button)
        appearance = QWidget(self)
        appearance.setObjectName("PlotAppearance")
        appearance_layout = QVBoxLayout(appearance)
        appearance_layout.setContentsMargins(0, 0, 0, 0)
        appearance_layout.setSpacing(6)
        title_edit = QLineEdit(self)
        title_edit.setPlaceholderText("Automatic title")
        title_edit.setMaxLength(200)
        title_edit.editingFinished.connect(self._changed)
        self.controls["title"] = title_edit
        self._add_appearance_control(appearance_layout, "title", "Title", title_edit)
        size = QDoubleSpinBox(self)
        size.setRange(1, 20)
        size.setSingleStep(0.5)
        size.setDecimals(1)
        size.setKeyboardTracking(False)
        size.valueChanged.connect(self._changed)
        self.controls["point_size"] = size
        self._add_appearance_control(
            appearance_layout, "point_size", "Point size", size
        )
        for key, title in (
            ("log_x", "Log X axis"),
            ("log_y", "Log Y axis"),
            ("show_grid", "Show grid"),
        ):
            checkbox = QCheckBox(title, self)
            checkbox.toggled.connect(self._changed)
            self.controls[key] = checkbox
            appearance_layout.addWidget(checkbox)
        self.controls["log_x"].setToolTip(
            "A log axis excludes non-positive values. Exclusions are reported."
        )
        self.controls["log_y"].setToolTip(
            "Log measurements exclude non-positive values. Exclusions are reported."
        )
        for axis in ("x", "y"):
            interval = _AxisIntervalControl(axis.upper(), self)
            interval.setObjectName("PlotAxisInterval")
            interval.changed.connect(self._changed)
            self.controls[f"{axis}_tick_interval"] = interval
            self._add_appearance_control(
                appearance_layout,
                f"{axis}_tick_interval",
                f"{axis.upper()} axis label interval",
                interval,
            )
        self.controls["show_grid"].setToolTip(
            "Grid lines follow major axis labels. Use the X/Y axis label "
            "interval controls to set their spacing on linear numeric axes. "
            "Histogram bin width is a separate calculation setting."
        )
        layout.addWidget(appearance)
        appearance.hide()
        self.appearance_button.toggled.connect(appearance.setVisible)
        self.appearance_button.toggled.connect(self._refresh_appearance_icon)
        self.appearance_button.toggled.connect(
            lambda _expanded: self.layout_changed.emit()
        )
        layout.addStretch(1)
        self.set_state(None, self._params)
        self._refresh_appearance_icon()

    def _refresh_appearance_icon(self, *_args):
        if not hasattr(self, "appearance_button"):
            return
        expanded = self.appearance_button.isChecked()
        # Transparent Qt stylesheet backgrounds can report transparent black
        # as Base even in light mode. Resolve the actual surrounding surface.
        owner = self
        while owner is not None and owner.palette().color(QPalette.Base).alpha() == 0:
            owner = owner.parentWidget()
        palette = owner.palette() if owner is not None else QApplication.palette()
        icon_palette = QPalette(palette)
        icon_palette.setColor(
            QPalette.ButtonText, palette_branch_color("Image Data", palette)
        )
        self.appearance_button.setIcon(
            interface_icon(
                "chevron-down" if expanded else "chevron-right", icon_palette, 18
            )
        )
        self.appearance_button.setAccessibleName(
            "Collapse Appearance" if expanded else "Expand Appearance"
        )

    def changeEvent(self, event):  # noqa: N802
        super().changeEvent(event)
        if event.type() in (QEvent.PaletteChange, QEvent.StyleChange):
            self._refresh_appearance_icon()

    def _add_control(self, key, title, widget):
        label = _label(title)
        font = label.font()
        font.setBold(False)
        label.setFont(font)
        self._labels[key] = label
        self.controls[key] = widget
        self.form.addWidget(label)
        self.form.addWidget(widget)

    def _add_appearance_control(self, layout, key, title, widget):
        label = _label(title)
        font = label.font()
        font.setBold(False)
        label.setFont(font)
        self._appearance_labels[key] = label
        layout.addWidget(label)
        layout.addWidget(widget)

    def set_state(self, table, params):
        self._updating = True
        try:
            self._table = table
            self._params = {**PlotRecipe().to_params(), **(params or {})}
            numeric = tuple(numeric_columns(table)) if table is not None else ()
            columns = table.columns if table is not None else ()
            for key in ("y_column", "x_column", "group_column", "image_column"):
                combo = self.controls[key]
                combo.clear()
                if key in {"y_column", "x_column"}:
                    combo.addItem(
                        "Choose a summary measurement"
                        if is_summary_table(table)
                        else "Automatic measurement",
                        "auto",
                    )
                    candidates = numeric
                else:
                    combo.addItem(
                        "None" if key == "group_column" else "Choose image identity",
                        "",
                    )
                    candidates = columns
                for column in candidates:
                    combo.addItem(measurement_label(table, column), column)
                    combo.setItemData(combo.count() - 1, column, Qt.ToolTipRole)
                selected = self._params[key]
                if combo.findData(selected) < 0 and selected:
                    combo.addItem(f"Unavailable: {selected}", selected)
            point_unit = self.controls["point_unit"]
            summary_input = is_summary_table(table)
            point_unit.setItemText(0, "One summary row" if summary_input else "Objects")
            point_unit.model().item(1).setEnabled(not summary_input)
            point_unit.setItemData(
                1,
                "Already summarized: connect the original measurements for image means."
                if summary_input
                else "Average the eligible rows within each image.",
                Qt.ToolTipRole,
            )
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
            self._update_point_unit_note()
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
        # Switching plot/scale is an explicit edit: those axes now use Auto.
        # Keep the last numeric text in the control for an easy switch back.
        for axis in ("x", "y"):
            changed_scale = params[f"log_{axis}"] != self._params[f"log_{axis}"]
            changed_type = params["plot_type"] != self._params["plot_type"]
            incompatible = params[f"log_{axis}"] or (
                axis == "x" and params["plot_type"] == "Compare groups"
            )
            if incompatible and (changed_scale or changed_type):
                params[f"{axis}_tick_interval"] = "Auto"
                self.controls[f"{axis}_tick_interval"].setValue("Auto")
        self._params = params
        self._update_visibility()
        self._update_point_unit_note()
        if params["point_unit"] == "Mean per image" and not params["image_column"]:
            return
        self.params_changed.emit(params)

    def _update_point_unit_note(self):
        if is_summary_table(self._table):
            self.note.setText(
                "Input: summary table. Each point is one summary row, not an "
                "original object or independent sample. Counts do not recreate "
                "the original observations; SD is not automatically an error bar."
                if self._params["point_unit"] == "Objects"
                else "This table is already summarized. Choose One summary row, "
                "or connect original measurements to calculate image means."
            )
            self.controls["point_unit"].setToolTip(self.note.text())
            return
        self.controls["point_unit"].setToolTip(
            "Objects uses one measurement row per point. Mean per image gives "
            "each image equal weight, after averaging its eligible rows. Neither "
            "choice automatically establishes independent biological samples."
        )
        if self._params["point_unit"] == "Mean per image":
            self.note.setText(
                "Choose the image identity column to prepare one mean per image."
                if not self._params["image_column"]
                else "Mean per image makes one point per image. Use Group by: None "
                "or a category shared by all objects in that image, such as "
                "treatment. Images are not automatically independent "
                "biological samples."
            )
            return
        self.note.setText(
            "Objects in one image are not automatically independent biological "
            "samples. These plots describe the connected measurements."
        )

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
        self._update_interval_controls()
        self.layout_changed.emit()

    def _update_interval_controls(self):
        mode = self._params["plot_type"]
        for axis in ("x", "y"):
            categorical = axis == "x" and mode == "Compare groups"
            logarithmic = self._params[f"log_{axis}"]
            if categorical:
                hint = "Category labels are automatic; there is no numeric X interval."
            elif logarithmic:
                hint = (
                    "Log axis: automatic labels. Use a linear axis for fixed intervals."
                )
            elif mode == "Distribution" and axis == "y":
                count = (
                    self._params["distribution"] == "Histogram"
                    and self._params["normalization"] == "Count"
                )
                hint = (
                    "Count: whole-number intervals only."
                    if count
                    else "Interval in percent (%)."
                )
            else:
                column = self.controls[
                    "x_column" if axis == "x" and mode == "Scatter" else "y_column"
                ].currentData()
                table = self._table
                if (
                    table is not None
                    and column in {"", "auto"}
                    and not is_summary_table(table)
                ):
                    columns = numeric_columns(table)
                    if axis == "x" and mode == "Scatter" and columns:
                        y_column = self.controls["y_column"].currentData()
                        if y_column in {"", "auto"}:
                            y_column = columns[0]
                        columns = tuple(name for name in columns if name != y_column)
                    column = columns[0] if columns else ""
                units = table.unit_for(column) if table is not None and column else ""
                hint = (
                    f"Interval in {units}."
                    if units
                    else "Interval in measurement units."
                )
            self.controls[f"{axis}_tick_interval"].set_axis_info(
                manual_allowed=not categorical and not logarithmic, hint=hint
            )


class _PreparedTableModel(QAbstractTableModel):
    """Virtualized access to complete prepared rows; no QTableWidget copying."""

    def __init__(self, table, parent=None, *, decimal_places=DEFAULT_DECIMAL_PLACES):
        super().__init__(parent)
        self.table = table
        self.decimal_places = validate_decimal_places(decimal_places)

    def set_decimal_places(self, decimal_places):
        decimal_places = validate_decimal_places(decimal_places)
        if self.decimal_places == decimal_places:
            return
        self.decimal_places = decimal_places
        if self.rowCount() and self.columnCount():
            self.dataChanged.emit(
                self.index(0, 0),
                self.index(self.rowCount() - 1, self.columnCount() - 1),
                [Qt.DisplayRole],
            )

    def rowCount(self, parent=None):  # noqa: N802
        return 0 if parent is not None and parent.isValid() else self.table.row_count

    def columnCount(self, parent=None):  # noqa: N802
        return 0 if parent is not None and parent.isValid() else self.table.column_count

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid() and role in (Qt.DisplayRole, Qt.ToolTipRole):
            value = self.table.rows[index.row()][index.column()]
            if role == Qt.ToolTipRole:
                return "" if value is None else str(value)
            return format_table_value(value, self.decimal_places)
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

    def __init__(self, parent=None, *, compact=False, fit_viewport=False):
        super().__init__(parent)
        self.compact = compact
        self.fit_viewport = fit_viewport
        self._fit_timer = QTimer(self)
        self._fit_timer.setSingleShot(True)
        self._fit_timer.timeout.connect(self._fit_preview_text)
        self.result = None
        self.canvas = None
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.placeholder = _label(
            "Calculate the connected measurements to prepare this plot.", self
        )
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.placeholder)
        self.error_view = QScrollArea(self)
        self.error_view.setWidgetResizable(True)
        self.error_view.setFrameShape(QFrame.NoFrame)
        self.error_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        error_content = QWidget()
        error_layout = QVBoxLayout(error_content)
        error_layout.setContentsMargins(0, 0, 0, 0)
        self.error_card = QFrame()
        self.error_card.setObjectName("PlotCalculationProblem")
        card_layout = QVBoxLayout(self.error_card)
        card_layout.setContentsMargins(14, 12, 14, 12)
        card_layout.setSpacing(12)
        self.error_title = _label("Plot could not be created")
        font = self.error_title.font()
        font.setBold(True)
        self.error_title.setFont(font)
        self.error_detail = _label("")
        self.error_detail.setTextInteractionFlags(Qt.TextSelectableByMouse)
        for label in (self.error_title, self.error_detail):
            label.setMinimumWidth(0)
            label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            card_layout.addWidget(label)
        error_layout.addWidget(self.error_card)
        error_layout.addWidget(
            _label(
                "Change the plot settings to continue. The input measurements "
                "have not been changed."
            )
        )
        error_layout.addStretch(1)
        self.error_view.setWidget(error_content)
        self.error_view.hide()
        self.layout.addWidget(self.error_view, 1)
        self.setMinimumHeight(0 if fit_viewport else 260 if compact else 320)
        self.setMinimumWidth(0)
        self.setSizePolicy(
            QSizePolicy.Ignored,
            QSizePolicy.Ignored if fit_viewport else QSizePolicy.Expanding,
        )

    def set_result(self, result):
        self.error_view.hide()
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
        try:
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
        except ValueError as exc:
            self.set_error(str(exc))
            return
        self.canvas = FigureCanvasQTAgg(figure)
        self.canvas.setMinimumWidth(0)
        self.canvas.setSizePolicy(
            QSizePolicy.Ignored,
            QSizePolicy.Ignored if self.fit_viewport else QSizePolicy.Expanding,
        )
        self.canvas.mpl_connect("pick_event", self._picked)
        if self.fit_viewport:
            self.canvas.installEventFilter(self)
            self._preview_title = figure.axes[0].get_title()
        self.layout.addWidget(self.canvas)
        self.canvas.draw_idle()
        if self.fit_viewport:
            self._fit_timer.start(0)

    def eventFilter(self, watched, event):  # noqa: N802
        if watched is self.canvas and event.type() in (QEvent.Resize, QEvent.Show):
            self._fit_timer.start(0)
        return super().eventFilter(watched, event)

    def _fit_preview_text(self):
        """Adapt only on-screen typography; the exporter builds its own figure."""
        if not self.fit_viewport or self.canvas is None or self.result is None:
            return
        width, height = self.canvas.width(), self.canvas.height()
        scale = max(0.55, min(1.0, width / 620, height / 340))
        size = (8 if self.compact else 10) * scale
        axes = self.canvas.figure.axes[0]
        axes.tick_params(labelsize=size)
        # Wrap long labels rather than cutting them off at the figure edges.
        # Qt reports logical pixels; the preview uses Matplotlib's logical DPI.
        pixels_per_point = self.canvas.figure.dpi / self.canvas.devicePixelRatioF() / 72
        for text, original, extent, font_size in (
            (axes.title, self._preview_title, width * 0.9, size * 1.2),
            (axes.xaxis.label, self.result.x_label, width * 0.8, size),
            (axes.yaxis.label, self.result.y_label, height * 0.85, size),
        ):
            chars = max(12, int(extent / (font_size * pixels_per_point * 0.58)))
            text.set_text(
                "\n".join(
                    fill(line, width=chars, break_long_words=False)
                    for line in original.splitlines()
                )
            )
            text.set_fontsize(font_size)
        for axis, extent in ((axes.xaxis, width), (axes.yaxis, height)):
            locator = axis.get_major_locator()
            if isinstance(locator, MaxNLocator):
                # Do not touch explicit axis intervals, log locators or category
                # positions. Integer-count locator settings are preserved.
                locator.set_params(nbins=max(2, min(6, int(extent / 65))))
        legend = axes.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(size)
        for text in self.canvas.figure.texts:
            text.set_fontsize(max(5, size - 1))
        self.canvas.draw_idle()

    def set_error(self, message):
        """Replace a failed/stale drawing with a readable, scrollable explanation."""
        self.set_result(None)
        self.placeholder.hide()
        text = message.strip() or "Review the selected fields and try again."
        heading, separator, details = text.partition("\n\n")
        self.error_title.setText(heading if separator else "Plot could not be created")
        self.error_detail.setText(details if separator else text)
        colors = theme_colors(self.palette()).error
        self.error_card.setStyleSheet(
            f"QFrame#PlotCalculationProblem {{ background: {colors.surface.name()}; "
            f"border-left: 3px solid {colors.accent.name()}; }} "
            "QFrame#PlotCalculationProblem QLabel { "
            f"color: {colors.foreground.name()}; "
            "background: transparent; border: none; }"
        )
        self.error_view.setAccessibleName("Plot calculation problem")
        self.error_view.setAccessibleDescription(text)
        self.error_view.show()

    def set_busy(self, busy):
        self.placeholder.setText(
            "Preparing the updated plot…"
            if busy
            else "Calculate the connected measurements to prepare this plot."
        )

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
        recipe = self.result.recipe
        identities = [
            f"{name}: {row[position]}"
            for position, name in enumerate(table.columns)
            if "label" in name.casefold()
            or "image" in name.casefold()
            or "source" in name.casefold()
            or name.casefold() in {"object_id", "_vipp_item_key"}
            or name in {recipe.group_column, recipe.image_column}
        ]
        message = " · ".join(identities) or f"Measurement row {indices[0] + 1}"
        measurements = [recipe.y_column]
        if recipe.plot_type == "Scatter":
            measurements.insert(0, recipe.x_column)
        if recipe.point_unit == "Objects":
            details = [
                f"{measurement_label(table, column)}: "
                f"{row[table.columns.index(column)]}"
                for column in dict.fromkeys(measurements)
            ]
        else:
            # Use the prepared image means, never the first contributor's value.
            values = next(
                (
                    {recipe.y_column: series.y[position]}
                    | (
                        {recipe.x_column: series.x[position]}
                        if recipe.plot_type == "Scatter"
                        else {}
                    )
                    for series in self.result.series
                    for position, rows in enumerate(series.source_rows)
                    if tuple(rows) == tuple(indices)
                ),
                {},
            )
            details = [
                f"Mean {measurement_label(table, column)}: {values[column]}"
                for column in dict.fromkeys(measurements)
                if column in values
            ]
        if details:
            message += " · " + " · ".join(details)
        if len(indices) > 1:
            message = f"Image mean from {len(indices):,} rows · " + message
        self.point_selected.emit(message)


def _result_summary(result):
    count = result.counts
    unit = (
        "summary rows"
        if is_summary_table(result.source_table)
        else "individual measurements"
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
        self.failed = False
        self.error_message = ""
        self.busy = False
        self.busy_message = ""
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
        self.busy_indicator = _PlotBusyIndicator(self)
        layout.addWidget(self.busy_indicator)
        self.plot = ResultPlotCanvas(self, compact=True)
        self.plot.setFixedHeight(280)
        layout.addWidget(self.plot)
        self.summary, self._summary_reserve = _summary_slot(layout)
        self.warning = _PlotWarningLabel(self)
        self.warning.layout_changed.connect(self.layout_changed.emit)
        layout.addWidget(self.warning)
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
        failed=False,
        busy=False,
        busy_message="",
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
        self.failed = bool(failed)
        self.error_message = message if failed else ""
        self.busy = bool(busy) and not self.failed
        self.busy_message = busy_message if self.busy else ""
        self.stale = stale or self.failed or self.busy
        self.busy_indicator.set_busy(self.busy, self.busy_message)
        if protected_paths is not None:
            self.protected_paths = tuple(protected_paths)
        self.controls.set_state(self.table, self.params)
        if self.failed:
            self.plot.set_error(self.error_message)
        else:
            self.plot.set_result(result)
        self.plot.set_busy(self.busy)
        updating_summary = (
            "Previous plot shown; waiting for the updated result."
            if result is not None
            else "Preparing the plot from connected measurements."
        )
        ready_summary = (
            _result_summary(result)
            if result is not None
            else "Connect a measurement table and calculate this node."
        )
        self._summary_reserve.setText(ready_summary if self.busy else updating_summary)
        self.summary.setText(
            "Plot not created — review the settings."
            if self.failed
            else updating_summary
            if self.busy
            else message
            or (
                "Plot is out of date. Calculate again before exporting."
                if stale
                else _result_summary(result)
                if result is not None
                else "Connect a measurement table and calculate this node."
            )
        )
        self.open_button.setText("Review plot…" if self.failed else "Open plot…")
        self.open_button.setEnabled(
            result is not None or self.table is not None or self.failed or self.busy
        )
        _set_plot_warnings(self.warning, result, stale=self.stale)
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
        if (
            self.result is None
            and self.table is None
            and not self.failed
            and not self.busy
        ):
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
        self.summary, self._summary_reserve = _summary_slot(layout)
        self.busy_indicator = _PlotBusyIndicator(self)
        layout.addWidget(self.busy_indicator)
        self.warning = _PlotWarningLabel(self)
        layout.addWidget(self.warning)
        splitter = QSplitter(Qt.Horizontal, self)
        plot_side = QWidget(self)
        plot_layout = QVBoxLayout(plot_side)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        self.plot = ResultPlotCanvas(self, fit_viewport=True)
        plot_layout.addWidget(self.plot, 1)
        self.point_label = _label(
            "Click a point to identify its object label or source image."
        )
        _retain_hidden_space(self.point_label)
        self.plot.point_selected.connect(self.point_label.setText)
        plot_layout.addWidget(self.point_label)
        self.data_decimal_controls = TableDecimalControls(self)
        self.data_decimal_controls.decimals_changed.connect(self._set_data_decimals)
        self.data_decimal_controls.hide()
        plot_layout.addWidget(self.data_decimal_controls, 0, Qt.AlignRight)
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
        row = QHBoxLayout()
        self.data_button = QPushButton("View plotted data", self)
        self.data_button.setCheckable(True)
        self.data_button.toggled.connect(self._set_data_visible)
        self._data_before_busy = None
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

    def _set_data_visible(self, visible):
        self.data_view.setVisible(visible)
        self.data_decimal_controls.setVisible(visible)

    def _set_data_decimals(self, decimal_places):
        model = self.data_view.model()
        if model is not None:
            model.set_decimal_places(decimal_places)

    def set_state(self):
        owner = self.owner
        if owner.busy:
            if self._data_before_busy is None:
                self._data_before_busy = self.data_button.isChecked()
            for widget in (self.data_view, self.data_decimal_controls):
                policy = widget.sizePolicy()
                policy.setRetainSizeWhenHidden(self._data_before_busy)
                widget.setSizePolicy(policy)
        self.busy_indicator.set_busy(owner.busy, owner.busy_message)
        self.controls.set_state(owner.table, owner.params)
        if owner.failed:
            self.plot.set_error(owner.error_message)
        else:
            self.plot.set_result(owner.result)
        self.plot.set_busy(owner.busy)
        self.summary.setText(owner.summary.text())
        self._summary_reserve.setText(owner._summary_reserve.text())
        self.export_button.setEnabled(owner.result is not None and not owner.stale)
        current_data = owner.result is not None and not owner.failed and not owner.busy
        self.data_button.setEnabled(current_data)
        self.data_decimal_controls.setEnabled(current_data)
        _set_plot_warnings(self.warning, owner.result, stale=owner.stale)
        if current_data:
            old = self.data_view.model()
            self.data_view.setModel(
                _PreparedTableModel(
                    owner.result.plotted_table,
                    self.data_view,
                    decimal_places=self.data_decimal_controls.decimal_places,
                )
            )
            if old is not None:
                old.deleteLater()
        else:
            self.data_button.setChecked(False)
            old = self.data_view.model()
            self.data_view.setModel(None)
            if old is not None:
                old.deleteLater()
        if not owner.busy and self._data_before_busy is not None:
            self.data_button.setChecked(current_data and self._data_before_busy)
            self._data_before_busy = None
            for widget in (self.data_view, self.data_decimal_controls):
                policy = widget.sizePolicy()
                policy.setRetainSizeWhenHidden(False)
                widget.setSizePolicy(policy)
        self.point_label.setVisible(current_data)
        if not owner.busy:
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
