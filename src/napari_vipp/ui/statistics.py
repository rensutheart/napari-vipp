"""Descriptive Statistics inspector; author recipes, never calculate on render."""

from __future__ import annotations

import csv
import json
from html import escape

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from napari_vipp.core.result_plots import measurement_label
from napari_vipp.core.statistics import (
    AUTO_GROUP_COLUMNS,
    StatisticsRecipe,
    measurement_columns,
)
from napari_vipp.ui.statistics_overview import StatisticsOverview

STATISTIC_LABELS = {
    "count": "Count (valid observations)",
    "mean": "Mean",
    "median": "Median",
    "std": "Standard deviation (sample SD)",
    "min": "Minimum",
    "max": "Maximum",
    "q25": "25th percentile (Q1)",
    "q75": "75th percentile (Q3)",
    "iqr": "Interquartile range (IQR)",
    "sum": "Sum",
}

CONTROL_HELP = {
    "group_by": (
        "Create a separate summary for each value in this column, such as Control "
        "and Treated. None combines all rows into one summary. To split by a "
        "combination, use Group by several columns."
    ),
    "summary_level": (
        "Choose what is counted and summarized: individual object measurements, "
        "one average per image, or one average per declared sample. For example, "
        "100 objects from 6 images can contribute 100 object values or 6 image "
        "averages. Switching to Objects clears both identity choices; switching "
        "to Image averages clears sample identity. The change can be undone."
    ),
    "sample_weighting": (
        "Choose how to combine images within a sample. Equal images gives each "
        "image mean the same weight. Equal objects pools the objects, so images "
        "with more valid objects have more influence. Both options then give "
        "each contributing sample the same weight in the final summary."
    ),
    "missing_policy": (
        "Choose what happens to missing values, non-numeric values (including "
        "text and booleans), or non-finite values (NaN or infinity). "
        "Exclude and report leaves these values out and records "
        "how many were excluded, separately for each measurement. Stop and "
        "review stops this calculation so you can inspect them first. Values "
        "are never replaced with zero. Neither option ignores identity errors."
    ),
}

STATISTIC_HELP = {
    "count": "Number of valid values summarized: objects, image averages or sample "
    "averages, depending on Each observation represents. This is the valid n "
    "shown in the preview, not necessarily the number of objects.",
    "mean": "Arithmetic average of the valid values at the chosen observation level.",
    "median": "Middle value after sorting; halfway between the two middle values "
    "when their number is even. Less influenced by extreme values than the mean.",
    "std": "Spread of the values around their mean, using sample standard deviation "
    "(n − 1). Needs at least two valid values at the chosen observation level. "
    "Otherwise shown as undefined, not zero. This is not a confidence interval.",
    "min": "Smallest valid value at the chosen observation level.",
    "max": "Largest valid value at the chosen observation level.",
    "q25": "Lower quartile: the 25th percentile of the valid values, using linear "
    "interpolation between sorted values.",
    "q75": "Upper quartile: the 75th percentile of the valid values, using linear "
    "interpolation between sorted values.",
    "iqr": "Upper quartile minus lower quartile (Q3 − Q1). Describes the spread "
    "of the middle half of the values.",
    "sum": "Add the valid values at the chosen observation level. With Image "
    "averages or Sample averages, this adds those averages, not all object values.",
}


def _tooltip(text):
    """Let Qt wrap help text without interpreting column names as markup."""
    return "<qt>" + escape(text).replace("\n", "<br>") + "</qt>"


def _paragraph(text):
    return f'<p style="margin: 0 0 5px 0;">{text}</p>'


def statistics_preview_columns(table):
    """Project modern wide results for display only, retaining the full table."""
    if "summary_version" not in table.columns:
        return None
    boundary = table.columns.index("summary_version")
    if table.rows and table.rows[0][boundary] != 2:
        return None
    selected = list(range(boundary))
    for column in table.columns:
        if not column.endswith("_object_total"):
            continue
        measurement = column.removesuffix("_object_total")
        for suffix in ("n", *STATISTIC_LABELS):
            # The always-visible valid n is the selected Count statistic too.
            if suffix == "count":
                continue
            name = f"{measurement}_{suffix}"
            if name in table.columns:
                selected.append(table.columns.index(name))
    return tuple(selected)


def statistics_preview_header(table, column):
    for suffix, title in (("n", "valid n"), *STATISTIC_LABELS.items()):
        if column.endswith(f"_{suffix}"):
            measurement = column.removesuffix(f"_{suffix}")
            if f"{measurement}_object_total" in table.columns:
                readable = measurement.replace("_", " ").capitalize()
                unit = table.unit_for(column)
                title = {
                    "std": "SD",
                    "q25": "Q1",
                    "q75": "Q3",
                    "iqr": "IQR",
                }.get(suffix, title)
                return f"{readable} · {title}" + (f"\n({unit})" if unit else "")
    return measurement_label(table, column)


def _label(text=""):
    label = QLabel(text)
    label.setWordWrap(True)
    label.setTextFormat(Qt.PlainText)
    label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
    return label


def _rich_label(text=""):
    label = _label(text)
    label.setTextFormat(Qt.RichText)
    return label


def _names(value):
    """Decode saved selections for presentation without changing authored text."""
    value = str(value or "")
    if not value or value == "auto":
        return ()
    try:
        if value.lstrip().startswith("["):
            return tuple(json.loads(value))
        return tuple(name.strip() for name in next(csv.reader([value])) if name.strip())
    except (ValueError, TypeError, csv.Error):
        return (value,)


def _selection(names):
    return json.dumps(list(names), ensure_ascii=False) if names else ""


class StatisticsPanel(QWidget):
    """A narrow, connected-table editor with explicit observation identities."""

    params_changed = Signal(dict)
    upgrade_requested = Signal()
    layout_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._updating = False
        self.params = StatisticsRecipe().to_params()
        self.table = None
        self._column_table = None
        self._measurement_columns = ()
        self._available_measurements = ()
        self.result = None
        self.stale = True
        self.failed = False
        self.setObjectName("StatisticsPanel")
        # Napari styles QLabel/QCheckBox surfaces as well as their parent. Keep
        # these transparent on either the inspector or workspace sidebar; inputs
        # retain their normal themed surfaces and focus/selection states.
        self.setStyleSheet(
            "QWidget#StatisticsPanel, QWidget#StatisticsControls, "
            "QFrame#StatisticsSection, QLabel, QCheckBox { background: transparent; }"
            'QFrame#StatisticsSection[separated="true"] {'
            " border: none; border-top: 1px solid palette(button); }"
        )
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Maximum)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        self.description = _label(
            "Describe connected measurements with counts, averages and spread. "
            "This node does not perform hypothesis tests or infer independent samples."
        )
        layout.addWidget(self.description)
        self.input_note = _label()

        self.legacy = QGroupBox("Legacy summary (version 1)")
        legacy_layout = QVBoxLayout(self.legacy)
        legacy_layout.addWidget(
            _label(
                "This saved node keeps its original calculations. Upgrading uses "
                "explicit observation levels and reports exclusions; sample SD for "
                "one observation becomes undefined instead of zero. Results can change."
            )
        )
        legacy_form = QFormLayout()
        legacy_form.setRowWrapPolicy(QFormLayout.WrapAllRows)
        self.legacy_controls = {}
        for key, title in (
            ("group_by", "Group by (saved column names)"),
            ("value_columns", "Measurements (saved column names)"),
            ("statistics", "Statistics (saved names)"),
        ):
            edit = QLineEdit()
            edit.setAccessibleName(title)
            edit.setToolTip(
                _tooltip(
                    {
                        "group_by": "Saved comma-separated grouping columns. In legacy "
                        "summaries, auto or an empty field uses the original automatic "
                        "grouping rule. Upgrade explicitly to use the new controls.",
                        "value_columns": "Saved comma-separated measurement columns; "
                        "auto keeps the original automatic selection rule.",
                        "statistics": "Saved reductions, such as count, mean, median "
                        "and std, using the original legacy calculation rules.",
                    }[key]
                )
            )
            edit.editingFinished.connect(
                lambda name=key, control=edit: self._edit(name, control.text())
            )
            self.legacy_controls[key] = edit
            legacy_form.addRow(title, edit)
            legacy_form.labelForField(edit).setToolTip(edit.toolTip())
        legacy_layout.addLayout(legacy_form)
        self.upgrade_button = QPushButton("Upgrade to descriptive Statistics")
        self.upgrade_button.setToolTip(
            "Explicitly upgrade this node as one undoable edit. "
            "Keep the saved measurement, grouping and statistic selections."
        )
        self.upgrade_button.clicked.connect(self.upgrade_requested)
        legacy_layout.addWidget(self.upgrade_button)
        layout.addWidget(self.legacy)

        self.modern = QWidget()
        self.modern.setObjectName("StatisticsControls")
        self.modern.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Maximum)
        modern_layout = QVBoxLayout(self.modern)
        modern_layout.setContentsMargins(0, 0, 0, 0)
        modern_layout.setSpacing(14)
        self.sections = {}
        measurements_layout = self._section(modern_layout, "Measurements")
        measurements_layout.addWidget(self.input_note)
        self.auto_measurements = QCheckBox("Auto-select measurements")
        self.auto_measurements.setToolTip(
            _tooltip(
                "Automatically select numeric measurement columns, excluding known "
                "IDs, metadata and chosen grouping or identity fields. New measurement "
                "columns are included when the connected table changes. Uncheck to "
                "keep a fixed selection and choose columns from the list. "
                "This does not create or change measurements."
            )
        )
        self.auto_measurements.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.auto_measurements.toggled.connect(self._auto_changed)
        measurements_layout.addWidget(self.auto_measurements)
        self.measurement_selection_note = _label()
        measurements_layout.addWidget(self.measurement_selection_note)
        measurement_actions = QHBoxLayout()
        measurement_actions.setSpacing(6)
        self.select_all_measurements_button = QPushButton("Select all")
        self.select_no_measurements_button = QPushButton("Select none")
        for button, select_all in (
            (self.select_all_measurements_button, True),
            (self.select_no_measurements_button, False),
        ):
            button.setAutoDefault(False)
            button.setAccessibleName(
                "Select all available measurements"
                if select_all
                else "Deselect all measurements"
            )
            button.setToolTip(
                _tooltip(
                    "Select the currently available numeric measurements as a fixed "
                    "selection. Exclude known IDs, text, grouping and identity fields. "
                    "Turn off auto-selection; new columns will not be added "
                    "automatically."
                    if select_all
                    else "Clear every measurement selection and turn off "
                    "auto-selection. "
                    "Choose at least one measurement before calculating a summary."
                )
            )
            button.clicked.connect(
                lambda _checked=False, all_values=select_all: (
                    self._set_all_measurements_selected(all_values)
                )
            )
            measurement_actions.addWidget(button)
        measurement_actions.addStretch(1)
        measurements_layout.addLayout(measurement_actions)
        self.measurements = self._checklist("Measurements")
        self.measurements.setToolTip(
            _tooltip(
                "Choose which table columns to summarize. Turn off Auto-select "
                "measurements, or use Select all or Select none, to edit this list. "
                "Each selection keeps its units and valid-value count. Columns marked "
                "not automatic require individual selection; Select all skips them."
            )
        )
        self.measurements.itemChanged.connect(self._measurements_changed)
        measurements_layout.addWidget(self.measurements)
        grouping_layout = self._section(modern_layout, "Grouping")
        grouping_form = self._form()
        grouping_layout.addLayout(grouping_form)
        self.controls = {}
        self._labels = {}
        self._add_combo("group_by", "Group results by", (), form=grouping_form)
        self.multiple_groups = QCheckBox("Group by several columns")
        self.multiple_groups.setToolTip(
            _tooltip(
                "Check to choose several grouping columns. Each distinct combination "
                "gets a separate summary: for example, condition + timepoint separates "
                "Control and Treated at each timepoint. Checking keeps your current "
                "grouping until you change the list. Unchecking clears grouping "
                "to None (one overall summary)."
            )
        )
        self.multiple_groups.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.multiple_groups.toggled.connect(self._multiple_groups_changed)
        grouping_layout.addWidget(self.multiple_groups)
        self.groups = self._checklist("Grouping columns")
        self.groups.setToolTip(self.multiple_groups.toolTip())
        self.groups.itemChanged.connect(self._groups_changed)
        grouping_layout.addWidget(self.groups)

        observation_layout = self._section(modern_layout, "Observation unit")
        self.form = self._form()
        for key, title, options in (
            (
                "summary_level",
                "Each observation represents",
                (
                    "Objects",
                    "Image averages",
                    "Sample averages",
                ),
            ),
            ("image_column", "Image identity column", ()),
            ("sample_column", "Sample identity column", ()),
            (
                "sample_weighting",
                "Within each sample, weight",
                (
                    "Equal images",
                    "Equal objects",
                ),
            ),
        ):
            self._add_combo(key, title, options)
        observation_layout.addLayout(self.form)
        self.observation_note = _rich_label()
        observation_layout.addWidget(self.observation_note)
        statistics_layout = self._section(modern_layout, "Statistics to report")
        self.statistics = self._checklist("Statistics to report")
        self.statistics.setToolTip(
            _tooltip(
                "Choose at least one descriptive statistic. Hover over a statistic "
                "for its meaning. Every choice uses the observation level above."
            )
        )
        self.statistics.setMaximumHeight(180)
        self.statistics.itemChanged.connect(self._statistics_changed)
        statistics_layout.addWidget(self.statistics)
        self.sd_note = _label(
            "SD needs at least two valid observations; otherwise it is undefined."
        )
        self.sd_note.setToolTip(_tooltip(STATISTIC_HELP["std"]))
        statistics_layout.addWidget(self.sd_note)
        missing_form = self._form()
        self._add_combo(
            "missing_policy",
            "Missing or invalid measurements",
            ("Exclude and report", "Stop and review"),
            form=missing_form,
        )
        statistics_layout.addLayout(missing_form)
        self.missing_note = _label()
        self.missing_note.setToolTip(_tooltip(CONTROL_HELP["missing_policy"]))
        statistics_layout.addWidget(self.missing_note)
        layout.addWidget(self.modern)
        self.validation_note = _label()
        self.validation_note.hide()
        layout.addWidget(self.validation_note)
        self.result_group = StatisticsOverview()
        self.result_note = self.result_group.result_note
        self.inclusion_note = self.result_group.inclusion_note
        layout.addWidget(self.result_group)
        self.set_state(table=None, params=self.params)

    def _section(self, parent_layout, title):
        section = QFrame()
        section.setObjectName("StatisticsSection")
        section.setProperty("separated", bool(self.sections))
        section.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Maximum)
        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 10 if self.sections else 0, 0, 0)
        layout.setSpacing(7)
        heading = _label(title)
        font = heading.font()
        font.setBold(True)
        heading.setFont(font)
        layout.addWidget(heading)
        self.sections[title] = section
        parent_layout.addWidget(section)
        return layout

    @staticmethod
    def _form():
        # Each field already occupies its own row. A vertical layout lets its
        # word-wrapped label use the full sidebar width; QFormLayout instead
        # constrains wrapped labels to their short sizeHint and wraps too early.
        form = QVBoxLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setAlignment(Qt.AlignTop)
        form.setSpacing(6)
        return form

    @staticmethod
    def _checklist(title):
        widget = QListWidget()
        widget.setAccessibleName(title)
        widget.setMinimumWidth(0)
        widget.setMinimumHeight(80)
        widget.setMaximumHeight(130)
        widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        return widget

    def _add_combo(self, key, title, options, *, form=None):
        combo = QComboBox()
        combo.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        combo.setMinimumContentsLength(1)
        combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        combo.setAccessibleName(title)
        for value in options:
            combo.addItem(value, value)
        label = _label(title)
        self.controls[key] = combo
        self._labels[key] = label
        if key in CONTROL_HELP:
            combo.setToolTip(_tooltip(CONTROL_HELP[key]))
            label.setToolTip(combo.toolTip())
        form = self.form if form is None else form
        if form.count():
            form.addSpacing(4)
        form.addWidget(label)
        form.addWidget(combo)
        combo.currentIndexChanged.connect(
            lambda _index, name=key, control=combo: self._edit(
                name, control.currentData()
            )
        )

    @staticmethod
    def _populate_list(widget, choices, selected):
        widget.clear()
        for value, label in choices:
            item = QListWidgetItem(label, widget)
            item.setData(Qt.UserRole, value)
            item.setToolTip(_tooltip(f"{label}\nColumn: {value}"))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if value in selected else Qt.Unchecked)

    @staticmethod
    def _checked(widget):
        return tuple(
            widget.item(index).data(Qt.UserRole)
            for index in range(widget.count())
            if widget.item(index).checkState() == Qt.Checked
        )

    def set_state(
        self, *, table, params, result=None, stale=True, failed=False, message=""
    ):
        """Apply accepted state silently; never infer or overwrite stored choices."""
        self._updating = True
        try:
            self.setEnabled(True)
            self.params = {**StatisticsRecipe().to_params(), **params}
            self.table, self.result = table, result
            self.stale, self.failed = stale, failed
            legacy = self.params["summary_version"] == 1
            self.description.setVisible(legacy)
            self.setToolTip(_tooltip(self.description.text()))
            self.legacy.setVisible(legacy)
            self.modern.setVisible(not legacy)
            for key, control in self.legacy_controls.items():
                control.setText(str(self.params[key]))
            columns = () if table is None else table.columns
            if table is not self._column_table:
                self._column_table = table
                self._measurement_columns = (
                    () if table is None else tuple(measurement_columns(table))
                )
            grouping = (
                tuple(name for name in AUTO_GROUP_COLUMNS if name in columns)
                if str(self.params["group_by"]).strip().lower() == "auto"
                else _names(self.params["group_by"])
            )
            reserved = {
                *grouping,
                self.params["image_column"],
                self.params["sample_column"],
            }
            measured = tuple(
                name for name in self._measurement_columns if name not in reserved
            )
            self._available_measurements = measured
            self.input_note.setText(
                "Calculate the connected upstream table to see available measurements. "
                "Saved selections are retained."
                if table is None
                else f"{table.row_count:,} input rows · "
                f"{len(measured):,} available measurements"
            )
            automatic = str(self.params["value_columns"]).strip().lower() == "auto"
            selected = measured if automatic else _names(self.params["value_columns"])
            choices = [(name, measurement_label(table, name)) for name in measured]
            choices.extend(
                (name, f"{measurement_label(table, name)} (not automatic)")
                for name in columns
                if name not in measured
            )
            choices.extend(
                (name, f"Unavailable: {name}")
                for name in selected
                if name not in columns
            )
            self._populate_list(self.measurements, choices, selected)
            self.auto_measurements.setChecked(automatic)
            self.measurements.setEnabled(not automatic)
            for key in ("group_by", "image_column", "sample_column"):
                combo = self.controls[key]
                combo.clear()
                combo.addItem(
                    "None" if key == "group_by" else "Choose identity column", ""
                )
                if key == "group_by" and self.params[key] == "auto":
                    combo.addItem("Automatic grouping (saved rule)", "auto")
                for name in columns:
                    value = (
                        _selection((name,))
                        if key == "group_by" and ("," in name or name == "auto")
                        else name
                    )
                    combo.addItem(measurement_label(table, name), value)
                    combo.setItemData(
                        combo.count() - 1, _tooltip(f"Column: {name}"), Qt.ToolTipRole
                    )
                value = self.params[key]
                if combo.findData(value) < 0:
                    names = _names(value)
                    label = (
                        f"Saved grouping: {', '.join(names)}"
                        if names and all(name in columns for name in names)
                        else f"Unavailable: {value}"
                    )
                    combo.addItem(label, value)
            for key, combo in self.controls.items():
                value = self.params[key]
                if combo.findData(value) < 0:
                    combo.addItem(f"Unavailable: {value}", value)
                combo.setCurrentIndex(combo.findData(value))
            grouping = _names(self.params["group_by"])
            self.multiple_groups.setChecked(
                self.multiple_groups.isChecked() or len(grouping) > 1
            )
            group_choices = [(name, measurement_label(table, name)) for name in columns]
            group_choices.extend(
                (name, f"Unavailable: {name}")
                for name in grouping
                if name not in columns
            )
            self._populate_list(self.groups, group_choices, grouping)
            stats = _names(self.params["statistics"])
            stat_choices = list(STATISTIC_LABELS.items())
            stat_choices.extend(
                (name, f"Saved statistic: {name}")
                for name in stats
                if name not in STATISTIC_LABELS
            )
            self._populate_list(self.statistics, stat_choices, stats)
            for index in range(self.statistics.count()):
                item = self.statistics.item(index)
                name = item.data(Qt.UserRole)
                item.setToolTip(
                    _tooltip(
                        STATISTIC_HELP.get(
                            name,
                            f"Saved statistic: {name}. "
                            "Its original selection is retained.",
                        )
                    )
                )
            self.validation_note.clear()
            self.validation_note.hide()
            self._update_visibility()
            self._show_result(message)
        finally:
            self._updating = False

    def _update_visibility(self):
        self._update_measurement_actions()
        level = self.controls["summary_level"].currentData()
        image_required = level == "Image averages" or (
            level == "Sample averages"
            and self.controls["sample_weighting"].currentData() == "Equal images"
        )
        self._labels["image_column"].setText(
            "Image identity column"
            if image_required
            else "Image identity (optional counts)"
        )
        self._labels["sample_column"].setText(
            "Sample identity column"
            if level == "Sample averages"
            else "Sample identity (optional counts)"
        )
        image_help = (
            "Choose the column identifying which image each object came from, "
            "for example image_id. Rows from the same image must have the same ID; "
            "different images need different IDs across the whole table. Do not "
            "use object labels as image IDs. "
        )
        image_help += (
            "Required here: VIPP averages objects within each image. Each image "
            "must belong to a single selected group."
            if image_required
            else "Optional here: adds image counts but does not change the averages "
            "or their weighting. If selected, every row must still have a valid ID."
        )
        sample_help = (
            "Choose the column identifying the experimental sample, for example "
            "an animal or culture ID. Images from the same sample share a sample "
            "ID; different samples need different IDs. "
        )
        sample_help += (
            "Required here: VIPP first calculates one average per sample with valid "
            "values. Each sample must belong to one selected group. An ID does not "
            "prove that the samples are biologically independent."
            if level == "Sample averages"
            else "Optional here: adds sample counts but does not average by sample, "
            "change the averages or give samples equal weight. Every row must still "
            "have a valid ID, and each image can belong to only one sample."
        )
        for key, help_text in (
            ("image_column", image_help),
            ("sample_column", sample_help),
        ):
            self.controls[key].setToolTip(_tooltip(help_text))
            self._labels[key].setToolTip(self.controls[key].toolTip())
        for key, visible in (
            (
                "image_column",
                level != "Objects" or bool(self.controls["image_column"].currentData()),
            ),
            (
                "sample_column",
                level == "Sample averages"
                or bool(self.controls["sample_column"].currentData()),
            ),
            ("sample_weighting", level == "Sample averages"),
        ):
            self.controls[key].setVisible(visible)
            self._labels[key].setVisible(visible)
        self.groups.setVisible(self.multiple_groups.isChecked())
        self.controls["group_by"].setEnabled(not self.multiple_groups.isChecked())
        notes = {
            "Objects": _paragraph(
                "<b>One value per object.</b> Each valid object "
                "measurement contributes equally."
            )
            + _paragraph(
                "Objects in one image are not automatically independent "
                "biological samples."
            ),
            "Image averages": _paragraph(
                "<b>One value per image.</b> Average the "
                "objects within each image, then give "
                "each image average equal weight."
            )
            + _paragraph(
                "Choose the image identity column above. An image is not "
                "automatically an independent biological sample."
            ),
            "Sample averages": _paragraph(
                "<b>One value per sample.</b> Give each "
                "sample with valid values equal weight."
            )
            + _paragraph(
                "Choose sample IDs from your experimental design. "
                "An ID does not establish biological independence."
            ),
        }
        self.observation_note.setText(
            notes.get(level, "Review the saved observation level.")
        )
        self.sd_note.setVisible("std" in self._checked(self.statistics))
        self.missing_note.setText(
            "Invalid values are left out and counted separately for each measurement."
            if self.controls["missing_policy"].currentData() == "Exclude and report"
            else "Calculation stops if any measurement is missing or invalid. "
            "Review the input table before continuing."
        )
        self.layout_changed.emit()

    def _edit(self, key, value):
        if self._updating:
            return
        params = {**self.params, key: value}
        if key == "summary_level":
            if value == "Objects":
                params.update(image_column="", sample_column="")
            elif value == "Image averages":
                params["sample_column"] = ""
        try:
            StatisticsRecipe.from_params(params)
        except (TypeError, ValueError) as exc:
            self.validation_note.setText(str(exc))
            self.validation_note.show()
            return
        self.validation_note.clear()
        self.validation_note.hide()
        self._update_visibility()
        self.params_changed.emit(params)

    def _auto_changed(self, automatic):
        if self._updating:
            return
        self._apply_measurement_selection(
            self._available_measurements
            if automatic
            else self._checked(self.measurements),
            automatic=automatic,
        )

    def _measurements_changed(self, _item):
        if self._updating:
            return
        self._update_measurement_actions()
        self._edit("value_columns", _selection(self._checked(self.measurements)))

    def _update_measurement_actions(self):
        automatic = self.auto_measurements.isChecked()
        selected = self._checked(self.measurements)
        self.select_all_measurements_button.setEnabled(
            bool(self._available_measurements)
            and (automatic or set(selected) != set(self._available_measurements))
        )
        self.select_no_measurements_button.setEnabled(automatic or bool(selected))
        self.measurement_selection_note.setText(
            "Includes numeric measurements, including new columns. "
            "IDs, text and grouping fields are excluded."
            if automatic
            else "No measurements selected. Choose at least one to calculate a summary."
            if not selected
            else "Fixed selection: new columns are not added. "
            "Select all skips IDs, text and grouping fields."
        )

    def _apply_measurement_selection(self, names, *, automatic=False):
        # Batch checkbox changes before emitting exactly one authored recipe.
        # The owner may synchronously rebuild this list when handling the signal.
        selected = set(names)
        self._updating = True
        try:
            self.auto_measurements.setChecked(automatic)
            for index in range(self.measurements.count()):
                item = self.measurements.item(index)
                item.setCheckState(
                    Qt.Checked if item.data(Qt.UserRole) in selected else Qt.Unchecked
                )
            self.measurements.setEnabled(not automatic)
        finally:
            self._updating = False
        self._update_measurement_actions()
        self._edit("value_columns", "auto" if automatic else _selection(names))

    def _set_all_measurements_selected(self, select_all):
        if (
            self._updating
            or not self.isEnabled()
            or self.params["summary_version"] != 2
        ):
            return
        button = (
            self.select_all_measurements_button
            if select_all
            else self.select_no_measurements_button
        )
        if button.isEnabled():
            self._apply_measurement_selection(
                self._available_measurements if select_all else ()
            )

    def _statistics_changed(self, _item):
        self._edit("statistics", ",".join(self._checked(self.statistics)))

    def _multiple_groups_changed(self, enabled):
        if self._updating:
            return
        self._update_visibility()
        if not enabled:
            self._edit("group_by", "")

    def _groups_changed(self, _item):
        self._edit("group_by", _selection(self._checked(self.groups)))

    def set_unavailable(self, message):
        self.setEnabled(False)
        self.stale = True
        self.result_group.set_unavailable(message)

    def _show_result(self, message):
        self.result_group.set_state(
            table=self.table,
            params=self.params,
            result=self.result,
            stale=self.stale,
            failed=self.failed,
            message=message,
        )
