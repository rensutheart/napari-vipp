"""Compact, read-only inclusion evidence for descriptive Statistics results."""

from __future__ import annotations

from dataclasses import dataclass

from qtpy.QtCore import QEvent, Qt
from qtpy.QtWidgets import QGridLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from napari_vipp.core.result_plots import measurement_label
from napari_vipp.ui.palette_roles import theme_colors


@dataclass(frozen=True)
class InclusionRow:
    """Display totals across result groups; not a new scientific reduction."""

    label: str
    used: int
    total: int
    excluded: int
    observations: int
    unit: str
    undefined_sd: int

    def description(self):
        text = (
            f"{self.label}: {self.used:,} of {self.total:,} objects used; "
            f"{self.excluded:,} excluded; {self.observations:,} {self.unit} summarized."
        )
        if self.undefined_sd:
            text += (
                f" SD unavailable in {self.undefined_sd:,} summary row(s): "
                "fewer than two valid values."
            )
        return text


def _label(text="", *, bold=False):
    label = QLabel(text)
    label.setTextFormat(Qt.PlainText)
    label.setWordWrap(True)
    label.setMinimumWidth(0)
    label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Maximum)
    label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
    label.setStyleSheet("background: transparent;")
    if bold:
        font = label.font()
        font.setBold(True)
        label.setFont(font)
    return label


class StatisticsOverview(QWidget):
    """Keep the result status and inclusion counts readable at either panel width."""

    _wide_threshold = 600

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("statisticsOverview")
        self.setStyleSheet("#statisticsOverview { background: transparent; }")
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Maximum)
        self.measurement_rows: tuple[InclusionRow, ...] = ()
        self._wide = None
        self._tone = ""
        self._warnings = []
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(6)
        self._layout.setAlignment(Qt.AlignTop)
        self.heading = _label("Result overview", bold=True)
        font = self.heading.font()
        font.setPointSizeF(font.pointSizeF() + 1)
        self.heading.setFont(font)
        # A parent application's QLabel font-size stylesheet takes precedence
        # over setFont(). Give the section title an explicit, scaled size too.
        self.heading.setStyleSheet(
            "background: transparent; font-weight: bold; "
            f"font-size: {max(font.pointSizeF(), 11):g}pt;"
        )
        self._layout.addWidget(self.heading)
        self.result_note = _label()
        self._layout.addWidget(self.result_note)
        self.inclusion_note = _label()
        self.inclusion_note.setToolTip(
            "Counts are separate for each measurement and totalled across all groups. "
            "Missing values are never replaced with zero. The full Results table "
            "contains per-group counts, exclusion reasons and calculation settings."
        )
        self._layout.addWidget(self.inclusion_note)
        self.counts = QWidget()
        self.counts.setObjectName("statisticsInclusionCounts")
        self.counts.setStyleSheet(
            "#statisticsInclusionCounts { background: transparent; }"
        )
        self.counts.setMinimumWidth(0)
        self.counts.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Maximum)
        self._grid = QGridLayout(self.counts)
        self._grid.setContentsMargins(0, 2, 0, 0)
        self._grid.setHorizontalSpacing(12)
        self._grid.setVerticalSpacing(5)
        self._grid.setAlignment(Qt.AlignTop)
        self._layout.addWidget(self.counts)
        self.more_note = _label()
        self._layout.addWidget(self.more_note)
        self.set_state(table=None, params={})

    def set_state(
        self, *, table, params, result=None, stale=True, failed=False, message=""
    ):
        """Present accepted result evidence without changing any table or recipe."""
        self.measurement_rows = ()
        self.inclusion_note.clear()
        self.more_note.clear()
        self._tone = ""
        if failed:
            self._tone = "error"
            self.result_note.setText(
                "Statistics need review." + (f"\n{message}" if message else "")
            )
        elif stale or result is None:
            self._tone = "warning"
            self.result_note.setText(
                message or "No current result. Calculate this node to update the table."
            )
        else:
            count = result.row_count
            self.result_note.setText(
                f"Current result: {count:,} summary {'row' if count == 1 else 'rows'}."
            )
            if params.get("summary_version", 2) == 1:
                self.inclusion_note.setText(
                    "Legacy result: detailed exclusions are not reported."
                )
            else:
                self._read_inclusions(table, params, result)
        self.inclusion_note.setVisible(bool(self.inclusion_note.text()))
        self.more_note.setVisible(bool(self.more_note.text()))
        self.counts.setVisible(bool(self.measurement_rows))
        self._rebuild_counts()
        self._apply_tones()
        self.setAccessibleDescription(
            "\n".join(
                part
                for part in (
                    self.result_note.text(),
                    self.inclusion_note.text(),
                    *(row.description() for row in self.measurement_rows),
                    self.more_note.text(),
                )
                if part
            )
        )
        self.updateGeometry()

    def set_unavailable(self, message):
        self.set_state(table=None, params={}, message=message)

    def _read_inclusions(self, table, params, result):
        columns = {name: index for index, name in enumerate(result.columns)}
        measurements = [
            name.removesuffix("_object_total")
            for name in result.columns
            if name.endswith("_object_total")
        ]
        unit = {
            "Objects": "object values",
            "Image averages": "image averages",
            "Sample averages": "sample averages",
        }.get(params.get("summary_level", "Objects"), "values")
        rows = []
        for measurement in measurements[:3]:

            def total(suffix, measurement=measurement):
                index = columns.get(f"{measurement}_{suffix}")
                return (
                    0 if index is None else sum(int(row[index]) for row in result.rows)
                )

            std_index = columns.get(f"{measurement}_std")
            undefined_sd = (
                0
                if std_index is None
                else sum(row[std_index] is None for row in result.rows)
            )
            rows.append(
                InclusionRow(
                    measurement_label(table or result, measurement),
                    total("object_valid"),
                    total("object_total"),
                    total("object_excluded"),
                    total("n"),
                    unit,
                    undefined_sd,
                )
            )
        self.measurement_rows = tuple(rows)
        if rows:
            self.inclusion_note.setText("Values included · totals across all groups")
        if len(measurements) > 3:
            self.more_note.setText(
                f"{len(measurements) - 3:,} more measurements. "
                "The full Results table contains their inclusion counts."
            )

    def _rebuild_counts(self):
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget() is not None:
                item.widget().hide()
                item.widget().deleteLater()
        self._warnings = []
        self._wide = self.width() >= self._wide_threshold
        for column in range(4):
            self._grid.setColumnStretch(column, 2 if column == 0 else 1)
        if self._wide and self.measurement_rows:
            for column, title in enumerate(
                ("Measurement", "Objects used", "Excluded", "Summarized")
            ):
                self._grid.addWidget(_label(title, bold=True), 0, column)
        grid_row = 1 if self._wide else 0
        for entry in self.measurement_rows:
            name = _label(entry.label, bold=not self._wide)
            name.setToolTip(entry.description())
            self._grid.addWidget(name, grid_row, 0, 1, 1 if self._wide else 4)
            if self._wide:
                for column, text in enumerate(
                    (
                        f"{entry.used:,} / {entry.total:,}",
                        f"{entry.excluded:,}",
                        f"{entry.observations:,} {entry.unit}",
                    ),
                    start=1,
                ):
                    self._grid.addWidget(_label(text), grid_row, column)
                grid_row += 1
            else:
                grid_row += 1
                self._grid.addWidget(
                    _label(
                        f"{entry.used:,} of {entry.total:,} objects used · "
                        f"{entry.excluded:,} excluded\n"
                        f"{entry.observations:,} {entry.unit} summarized"
                    ),
                    grid_row,
                    0,
                    1,
                    4,
                )
                grid_row += 1
            if entry.undefined_sd:
                warning = _label(
                    f"SD unavailable in {entry.undefined_sd:,} summary row(s): "
                    "fewer than two valid values."
                )
                self._warnings.append(warning)
                self._grid.addWidget(warning, grid_row, 0, 1, 4)
                grid_row += 1
        self._apply_tones()
        self.updateGeometry()

    def _apply_tones(self):
        colors = theme_colors(self.palette())
        foreground = (
            getattr(colors, self._tone).foreground if self._tone else colors.text
        )
        self.result_note.setStyleSheet(
            f"background: transparent; color: {foreground.name()};"
        )
        for warning in self._warnings:
            warning.setStyleSheet(
                f"background: transparent; color: {colors.warning.foreground.name()};"
            )

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        if self._wide != (self.width() >= self._wide_threshold):
            self._rebuild_counts()

    def changeEvent(self, event):  # noqa: N802
        super().changeEvent(event)
        if hasattr(self, "_warnings") and event.type() in (
            QEvent.PaletteChange,
            QEvent.ApplicationPaletteChange,
        ):
            self._apply_tones()
