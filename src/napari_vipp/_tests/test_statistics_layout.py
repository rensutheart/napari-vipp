"""Statistics sections separate choices, explain active policies, and inherit theme."""

from copy import deepcopy

import pytest
from qtpy.QtCore import QPoint
from qtpy.QtWidgets import QFrame, QVBoxLayout

from napari_vipp.core.statistics import StatisticsRecipe, summarize_statistics
from napari_vipp.core.tables import TableData
from napari_vipp.ui.statistics import StatisticsPanel


def _table():
    return TableData(
        ("condition", "image_id", "sample_id", "area", "signal"),
        (
            ("Control", "image-1", "sample-1", 20.0, 10.0),
            ("Treated", "image-2", "sample-2", None, 30.0),
        ),
        column_units=(("area", "µm²"),),
    )


def _panel(qtbot, **params):
    panel = StatisticsPanel()
    qtbot.addWidget(panel)
    panel.set_state(table=_table(), params={**StatisticsRecipe().to_params(), **params})
    panel.resize(300, 1600)
    panel.show()
    return panel


def test_sections_follow_workflow_and_keep_related_fields_together(qtbot):
    panel = _panel(qtbot)
    assert tuple(panel.sections) == (
        "Measurements",
        "Grouping",
        "Observation unit",
        "Statistics to report",
    )
    expected = {
        "Measurements": (panel.input_note, panel.auto_measurements, panel.measurements),
        "Grouping": (panel.controls["group_by"], panel.multiple_groups, panel.groups),
        "Observation unit": (
            panel.controls["summary_level"],
            panel.controls["image_column"],
            panel.controls["sample_column"],
            panel.controls["sample_weighting"],
            panel.observation_note,
        ),
        "Statistics to report": (
            panel.statistics,
            panel.sd_note,
            panel.controls["missing_policy"],
            panel.missing_note,
        ),
    }
    prior_bottom = -1
    for index, (title, section) in enumerate(panel.sections.items()):
        assert all(section.isAncestorOf(control) for control in expected[title])
        heading = section.layout().itemAt(0).widget()
        assert heading.text() == title
        assert heading.font().bold()
        assert section.property("separated") is (index != 0)
        assert section.y() > prior_bottom
        prior_bottom = section.geometry().bottom()
    assert panel.description.isHidden()
    assert "hypothesis tests" in panel.toolTip()
    assert "experimental design" in panel.controls["sample_column"].toolTip() or (
        "experimental sample" in panel.controls["sample_column"].toolTip()
    )


@pytest.mark.parametrize("policy", ["Exclude and report", "Stop and review"])
def test_missing_value_help_describes_only_the_selected_policy(qtbot, policy):
    panel = _panel(qtbot, missing_policy=policy)
    if policy == "Exclude and report":
        assert "left out and counted" in panel.missing_note.text()
        assert "Calculation stops" not in panel.missing_note.text()
    else:
        assert "Calculation stops" in panel.missing_note.text()
        assert "left out and counted" not in panel.missing_note.text()
    assert "never replaced with zero" in panel.missing_note.toolTip()
    assert "Neither option ignores identity errors" in panel.missing_note.toolTip()


def test_sd_explanation_tracks_selection_without_emitting_refresh_edits(qtbot):
    panel = _panel(qtbot, statistics="mean")
    edits = []
    panel.params_changed.connect(edits.append)
    assert panel.sd_note.isHidden()
    panel.set_state(
        table=panel.table, params={**panel.params, "statistics": "mean,std"}
    )
    assert not panel.sd_note.isHidden()
    assert "at least two valid observations" in panel.sd_note.text()
    assert "undefined" in panel.sd_note.text()
    assert "not a confidence interval" in panel.sd_note.toolTip()
    assert edits == []
    panel.set_state(table=panel.table, params={**panel.params, "statistics": "median"})
    assert panel.sd_note.isHidden()
    assert edits == []


def test_new_layout_does_not_mutate_scientific_state_during_refresh(qtbot):
    panel = _panel(qtbot)
    table = _table()
    params = {
        **StatisticsRecipe().to_params(),
        "value_columns": "area,signal",
        "group_by": "condition",
        "summary_level": "Sample averages",
        "image_column": "image_id",
        "sample_column": "sample_id",
    }
    result = summarize_statistics(table, **params)
    before = deepcopy((params, table, result))
    edits, upgrades = [], []
    panel.params_changed.connect(edits.append)
    panel.upgrade_requested.connect(lambda: upgrades.append(True))
    panel.set_state(table=table, params=params, result=result, stale=False)
    assert panel.result_group.measurement_rows
    panel.resize(260, 1900)
    panel.set_state(table=table, params=params, result=result, stale=True)
    assert not panel.result_group.measurement_rows
    assert (params, table, result) == before
    assert edits == upgrades == []
    assert panel.params == params


def test_field_labels_use_available_width_instead_of_wrapping_prematurely(qtbot):
    panel = _panel(qtbot, summary_level="Sample averages")
    for key, label in panel._labels.items():
        if label.isVisible():
            assert abs(label.width() - panel.controls[key].width()) <= 2


def _sample(image, host, widget):
    point = widget.mapTo(host, QPoint(widget.width() - 3, widget.height() - 3))
    scale = image.devicePixelRatio()
    return image.pixelColor(round(point.x() * scale), round(point.y() * scale))


@pytest.mark.parametrize(
    "theme,background", [("dark", "#2d2e36"), ("light", "#e6e8ee")]
)
def test_actual_napari_styles_leave_labels_and_checkboxes_on_parent_surface(
    qtbot, theme, background
):
    from napari._qt.qt_resources import get_stylesheet

    host = QFrame()
    host.setObjectName("statisticsTestHost")
    qtbot.addWidget(host)
    host.setStyleSheet(
        get_stylesheet(theme, extra_variables={"font_size": "10pt"})
        + f"QFrame#statisticsTestHost {{ background: {background}; }}"
    )
    panel = StatisticsPanel()
    panel.set_state(
        table=_table(),
        params={**StatisticsRecipe().to_params(), "statistics": "mean,std"},
    )
    layout = QVBoxLayout(host)
    layout.setContentsMargins(12, 12, 12, 12)
    layout.addWidget(panel)
    host.resize(380, 1800)
    host.show()
    qtbot.waitUntil(lambda: panel.auto_measurements.isVisible())
    image = host.grab().toImage()
    for control in (
        panel.input_note,
        panel.auto_measurements,
        panel.multiple_groups,
        panel._labels["group_by"],
        panel.observation_note,
        panel.sd_note,
        panel.missing_note,
    ):
        assert _sample(image, host, control).name() == background, control.text()
    assert _sample(image, host, panel.controls["group_by"]).name() != background
