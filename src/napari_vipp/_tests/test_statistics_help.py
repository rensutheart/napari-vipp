"""Statistics guidance is discoverable, safely formatted and level-aware."""

import json
from html import escape

import pytest
from qtpy.QtCore import Qt
from qtpy.QtGui import QTextDocument
from qtpy.QtWidgets import QLabel

from napari_vipp._tests.test_statistics_ui import _item, _panel
from napari_vipp.core.statistics import summarize_statistics
from napari_vipp.core.tables import TableData


def _plain(text):
    document = QTextDocument()
    document.setHtml(text)
    return document.toPlainText()


def _useful_tooltip(control):
    text = _plain(control.toolTip()).strip()
    assert len(text.split()) >= 6, text
    return text.lower()


def test_every_statistics_control_and_its_label_share_useful_help(qtbot):
    panel = _panel(qtbot)
    for name, control in panel.controls.items():
        _useful_tooltip(control)
        assert panel._labels[name].toolTip() == control.toolTip(), name
    for control in (
        panel.auto_measurements,
        panel.multiple_groups,
        panel.groups,
        panel.measurements,
        panel.statistics,
        panel.upgrade_button,
        *panel.legacy_controls.values(),
    ):
        _useful_tooltip(control)


def test_statistic_items_explain_count_and_sample_standard_deviation(qtbot):
    panel = _panel(qtbot)
    for index in range(panel.statistics.count()):
        item = panel.statistics.item(index)
        assert len(_plain(item.toolTip()).split()) >= 6, item.data(Qt.UserRole)
    count = _plain(_item(panel.statistics, "count").toolTip()).lower()
    assert "valid" in count
    assert "observation" in count
    sd = _plain(_item(panel.statistics, "std").toolTip()).lower()
    assert "sample" in sd
    assert "undefined" in sd
    assert "two" in sd or "2" in sd


def test_multiple_group_help_explains_combination_and_return_to_none(qtbot):
    panel = _panel(qtbot)
    help_text = _useful_tooltip(panel.multiple_groups)
    assert "combin" in help_text
    assert "uncheck" in help_text or "clear" in help_text
    assert "none" in help_text
    assert "column" in _useful_tooltip(panel.groups)


@pytest.mark.parametrize("identity", ["image", "sample"])
def test_identity_help_changes_between_optional_counts_and_required_averages(
    qtbot,
    identity,
):
    panel = _panel(qtbot, image_column="image_id", sample_column="sample_id")
    control = panel.controls[f"{identity}_column"]
    optional = _useful_tooltip(control)
    assert "optional" in optional
    assert "count" in optional
    assert "average" in optional
    assert ("does not" in optional and "change the averages" in optional) or (
        "unchanged" in optional
    )
    assert f"same {identity}" in optional
    params = {
        **panel.params,
        "summary_level": "Image averages" if identity == "image" else "Sample averages",
    }
    panel.set_state(table=panel.table, params=params)
    required = _useful_tooltip(control)
    assert required != optional
    assert "required" in required
    assert f"same {identity}" in required
    assert panel._labels[f"{identity}_column"].toolTip() == control.toolTip()


def test_equal_objects_sample_help_marks_image_identity_optional(qtbot):
    panel = _panel(
        qtbot,
        summary_level="Sample averages",
        sample_weighting="Equal images",
        image_column="image_id",
        sample_column="sample_id",
    )
    assert "required" in _useful_tooltip(panel.controls["image_column"])
    panel.set_state(
        table=panel.table,
        params={**panel.params, "sample_weighting": "Equal objects"},
    )
    assert "optional" in _useful_tooltip(panel.controls["image_column"])
    assert "required" in _useful_tooltip(panel.controls["sample_column"])


@pytest.mark.parametrize("policy", ["Exclude and report", "Stop and review"])
def test_missing_policy_guidance_describes_only_the_selected_policy(qtbot, policy):
    panel = _panel(qtbot)
    panel.set_state(
        table=panel.table, params={**panel.params, "missing_policy": policy}
    )
    text = panel.missing_note.text().lower()
    assert "invalid" in text
    if policy == "Exclude and report":
        assert "left out" in text and "counted" in text
        assert "stops" not in text
    else:
        assert "stops" in text and "review" in text
        assert "left out" not in text
    explanation = _useful_tooltip(panel.controls["missing_policy"])
    assert "missing" in explanation
    assert "non-numeric" in explanation or "nonnumeric" in explanation
    assert "infinite" in explanation or "non-finite" in explanation


def test_observation_and_result_guidance_is_structurally_grouped(qtbot):
    panel = _panel(qtbot)
    assert panel.observation_note.textFormat() == Qt.RichText
    assert panel.sections["Observation unit"].isAncestorOf(panel.observation_note)
    assert panel.result_group.heading.text() == "Result overview"
    assert panel.result_group.heading.font().bold()
    assert panel.result_group.isAncestorOf(panel.result_note)
    assert panel.result_group.isAncestorOf(panel.inclusion_note)


def test_inclusion_displays_measurement_names_and_units_as_literal_text(qtbot):
    panel = _panel(qtbot)
    column = "<b>area & volume</b>"
    unit = "<i>unit & scale</i>"
    table = TableData(
        (column,), ((2.0,), (None,), (6.0,)), column_units=((column, unit),)
    )
    params = {
        **panel.params,
        "value_columns": json.dumps([column]),
        "statistics": "count,mean,std",
    }
    result = summarize_statistics(table, **params)
    panel.set_state(table=table, params=params, result=result, stale=False)
    text = panel.result_group.accessibleDescription()
    assert column in text
    assert unit in text
    assert all(
        label.textFormat() == Qt.PlainText
        for label in panel.result_group.findChildren(QLabel)
    )
    for name in ("group_by", "image_column", "sample_column"):
        combo = panel.controls[name]
        index = combo.findData(column)
        assert index >= 0
        tooltip = combo.itemData(index, Qt.ToolTipRole)
        assert escape(column) in tooltip
        assert column in _plain(tooltip)
        assert column not in tooltip


@pytest.mark.parametrize("state", ["stale", "failed", "unavailable"])
def test_external_status_text_is_literal_and_clears_previous_inclusion(qtbot, state):
    panel = _panel(qtbot, value_columns="area")
    result = summarize_statistics(panel.table, **panel.params)
    panel.set_state(table=panel.table, params=panel.params, result=result, stale=False)
    assert panel.inclusion_note.text()
    message = '<b>Review field & source</b> <a href="https://invalid.test">details</a>'
    if state == "unavailable":
        panel.set_unavailable(message)
    else:
        panel.set_state(
            table=panel.table,
            params=panel.params,
            result=result,
            stale=True,
            failed=state == "failed",
            message=message,
        )
    text = panel.result_note.text()
    assert panel.result_note.textFormat() == Qt.PlainText
    assert message in text
    assert panel.inclusion_note.text() == ""


@pytest.mark.parametrize("level", ["Image averages", "Sample averages"])
def test_inclusion_identifies_which_averages_are_counted(qtbot, level):
    panel = _panel(
        qtbot,
        value_columns="area",
        summary_level=level,
        image_column="image_id",
        sample_column="sample_id" if level == "Sample averages" else "",
    )
    result = summarize_statistics(panel.table, **panel.params)
    panel.set_state(table=panel.table, params=panel.params, result=result, stale=False)
    text = panel.result_group.accessibleDescription().lower()
    assert level.lower() in text
    assert "2 of 3 objects used" in text
    assert "1 excluded" in text
