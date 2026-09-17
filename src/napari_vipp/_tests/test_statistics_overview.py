"""Result inclusion evidence stays compact, responsive, and scientifically literal."""

import pytest
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QPalette
from qtpy.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from napari_vipp.core.statistics import StatisticsRecipe, summarize_statistics
from napari_vipp.core.tables import TableData
from napari_vipp.ui.palette_roles import theme_colors
from napari_vipp.ui.statistics_overview import StatisticsOverview


def _data():
    return TableData(
        ("image_id", "sample_id", "condition", "area", "intensity"),
        (
            ("a", "s1", "control", 10.0, 40.0),
            ("a", "s1", "control", None, 60.0),
            ("b", "s2", "treated", 30.0, 80.0),
        ),
        column_units=(("area", "µm²"),),
    )


def _ready(qtbot, **overrides):
    widget = StatisticsOverview()
    qtbot.addWidget(widget)
    table = _data()
    params = {
        **StatisticsRecipe().to_params(),
        "value_columns": "area,intensity",
        "statistics": "count,mean,std",
        **overrides,
    }
    result = summarize_statistics(table, **params)
    widget.set_state(table=table, params=params, result=result, stale=False)
    return widget, table, params, result


def test_overview_separates_object_inclusion_from_image_observations(qtbot):
    widget, _, _, _ = _ready(
        qtbot, summary_level="Image averages", image_column="image_id"
    )
    assert widget.heading.text() == "Result overview"
    assert widget.heading.font().bold()
    assert widget.result_note.text() == "Current result: 1 summary row."
    assert "across all groups" in widget.inclusion_note.text()
    area, intensity = widget.measurement_rows
    assert area.label == "Area (µm²)"
    assert (area.used, area.total, area.excluded, area.observations) == (2, 3, 1, 2)
    assert area.unit == "image averages"
    assert (intensity.used, intensity.total, intensity.observations) == (3, 3, 2)
    assert "2 of 3 objects used; 1 excluded; 2 image averages" in (
        widget.accessibleDescription()
    )


def test_sample_counts_and_undefined_sd_are_explicit(qtbot):
    widget, _, _, _ = _ready(
        qtbot,
        summary_level="Sample averages",
        image_column="image_id",
        sample_column="sample_id",
        group_by="condition",
    )
    assert all(row.unit == "sample averages" for row in widget.measurement_rows)
    assert all(row.observations == 2 for row in widget.measurement_rows)
    assert all(row.undefined_sd == 2 for row in widget.measurement_rows)
    assert "fewer than two valid values" in widget.accessibleDescription()
    assert len(widget._warnings) == 2


@pytest.mark.parametrize("state", ["stale", "failed", "unavailable", "legacy"])
def test_noncurrent_or_legacy_state_clears_all_previous_counts(qtbot, state):
    widget, table, params, result = _ready(qtbot)
    assert widget.measurement_rows
    if state == "unavailable":
        widget.set_unavailable("Connect a current table.")
    else:
        widget.set_state(
            table=table,
            params={**params, "summary_version": 1 if state == "legacy" else 2},
            result=result,
            stale=state == "stale",
            failed=state == "failed",
            message="Choose an image identity." if state == "failed" else "",
        )
    assert widget.measurement_rows == ()
    assert widget._grid.count() == 0
    assert widget.counts.isHidden()
    assert "objects used" not in widget.accessibleDescription()
    assert widget.more_note.text() == ""
    if state == "failed":
        assert "Choose an image identity" in widget.result_note.text()
    if state == "legacy":
        assert "detailed exclusions are not reported" in widget.inclusion_note.text()


def test_preview_is_bounded_without_claiming_other_measurements_absent(qtbot):
    widget = StatisticsOverview()
    qtbot.addWidget(widget)
    table = TableData(tuple(f"measure_{i}" for i in range(62)), (tuple(range(62)),))
    params = StatisticsRecipe().to_params()
    result = summarize_statistics(table, **params)
    widget.set_state(table=table, params=params, result=result, stale=False)
    assert len(widget.measurement_rows) == 3
    assert "59 more measurements" in widget.more_note.text()
    assert "inclusion counts" in widget.more_note.text()


def test_column_names_and_messages_remain_plain_text(qtbot):
    widget = StatisticsOverview()
    qtbot.addWidget(widget)
    table = TableData(("<b>area</b>",), ((2.0,), (4.0,)))
    params = StatisticsRecipe().to_params()
    result = summarize_statistics(table, **params)
    widget.set_state(table=table, params=params, result=result, stale=False)
    assert widget.measurement_rows[0].label == "<b>area</b>"
    assert all(
        label.textFormat() == Qt.PlainText for label in widget.findChildren(QLabel)
    )
    widget.set_unavailable("Input <b>not found</b> & unchanged")
    assert widget.result_note.text() == "Input <b>not found</b> & unchanged"


def test_responsive_overview_uses_content_height_not_available_height(qtbot):
    widget, _, _, _ = _ready(qtbot)
    host = QWidget()
    qtbot.addWidget(host)
    layout = QVBoxLayout(host)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(widget)
    layout.addStretch(1)
    host.resize(1000, 700)
    host.show()
    qtbot.waitUntil(lambda: widget._wide is True)
    assert widget.height() < 210
    assert widget.sizePolicy().verticalPolicy() == QSizePolicy.Maximum
    assert widget._grid.itemAtPosition(0, 0).widget().text() == "Measurement"
    host.resize(260, 700)
    qtbot.waitUntil(lambda: widget._wide is False)
    assert widget.width() <= 260
    assert widget.height() < 350
    assert widget._grid.itemAtPosition(0, 0).widget().text() == "Area (µm²)"
    assert widget.layout().hasHeightForWidth()
    assert not widget.autoFillBackground()
    assert all(
        label.geometry().right() <= label.parentWidget().width()
        for label in widget.findChildren(QLabel)
        if not label.isHidden()
    )


def test_warning_treatment_tracks_theme_without_background_blocks(qtbot):
    widget = StatisticsOverview()
    qtbot.addWidget(widget)
    widget.show()
    for base, text in (("#222222", "#eeeeee"), ("#ffffff", "#222222")):
        palette = QPalette(widget.palette())
        palette.setColor(QPalette.Base, QColor(base))
        palette.setColor(QPalette.Text, QColor(text))
        widget.setPalette(palette)
        expected = theme_colors(palette).warning.foreground.name()
        assert expected in widget.result_note.styleSheet()
        assert "background: transparent" in widget.result_note.styleSheet()
