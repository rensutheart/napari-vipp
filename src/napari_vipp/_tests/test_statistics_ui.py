"""Statistics controls retain authored choices without doing scientific work."""

import json

from qtpy.QtCore import Qt

from napari_vipp.core.statistics import StatisticsRecipe, summarize_statistics
from napari_vipp.core.tables import TableData
from napari_vipp.ui.statistics import StatisticsPanel, statistics_preview_columns


def _table():
    return TableData(
        ("label_id", "image_id", "sample_id", "condition", "area", "intensity"),
        (
            (1, "a", "s1", "control", 10.0, 40.0),
            (2, "a", "s1", "control", None, 60.0),
            (1, "b", "s2", "treated", 30.0, 80.0),
        ),
        column_units=(("area", "µm²"),),
    )


def _panel(qtbot, **params):
    panel = StatisticsPanel()
    qtbot.addWidget(panel)
    values = {**StatisticsRecipe().to_params(), **params}
    panel.set_state(table=_table(), params=values)
    panel.resize(280, 1000)
    panel.show()
    return panel


def _item(widget, value):
    return next(
        widget.item(index)
        for index in range(widget.count())
        if widget.item(index).data(Qt.UserRole) == value
    )


def test_connected_measurements_have_units_and_no_automatic_ids(qtbot):
    panel = _panel(qtbot)
    assert panel.auto_measurements.isChecked()
    assert panel._checked(panel.measurements) == ("area", "intensity")
    assert "µm²" in _item(panel.measurements, "area").text()
    assert "not automatic" in _item(panel.measurements, "label_id").text()
    assert panel.controls["group_by"].currentData() == ""
    assert panel.controls["group_by"].itemText(0) == "None"
    assert not panel.controls["image_column"].isVisible()
    assert not panel.controls["sample_weighting"].isVisible()
    assert panel.width() <= 280


def test_missing_input_retains_saved_measurement_and_identity(qtbot):
    panel = _panel(qtbot, value_columns="area,removed", image_column="old_image")
    params = dict(panel.params)
    emitted = []
    panel.params_changed.connect(emitted.append)
    panel.set_state(table=None, params=params)
    assert emitted == []
    assert panel.params == params
    assert panel._checked(panel.measurements) == ("area", "removed")
    assert "Unavailable" in _item(panel.measurements, "removed").text()
    assert panel.controls["image_column"].currentData() == "old_image"
    assert "Calculate the connected upstream table" in panel.input_note.text()


def test_observation_levels_commit_incomplete_explicit_identities(qtbot):
    panel = _panel(qtbot)
    emitted = []
    panel.params_changed.connect(emitted.append)
    panel.controls["summary_level"].setCurrentText("Sample averages")
    assert emitted[-1]["summary_level"] == "Sample averages"
    assert emitted[-1]["image_column"] == ""
    assert emitted[-1]["sample_column"] == ""
    assert panel.controls["image_column"].isVisible()
    assert panel.controls["sample_column"].isVisible()
    assert panel.controls["sample_weighting"].isVisible()
    assert "experimental design" in panel.observation_note.text()


def test_programmatic_refresh_never_emits_recipe_or_upgrade(qtbot):
    panel = _panel(qtbot)
    edits, upgrades = [], []
    panel.params_changed.connect(edits.append)
    panel.upgrade_requested.connect(lambda: upgrades.append(True))
    panel.set_state(table=_table(), params=panel.params)
    panel.set_state(table=_table(), params={**panel.params, "summary_version": 1})
    assert edits == upgrades == []
    assert panel.legacy.isVisible()
    assert not panel.modern.isVisible()
    assert panel.legacy_controls["value_columns"].text() == "auto"
    panel.upgrade_button.click()
    assert upgrades == [True]
    assert panel.params["summary_version"] == 1


def test_optional_saved_identities_remain_visible_and_clearable(qtbot):
    panel = _panel(qtbot, image_column="image_id", sample_column="sample_id")
    assert panel.controls["image_column"].isVisible()
    assert panel.controls["sample_column"].isVisible()
    assert "optional" in panel._labels["image_column"].text()
    emitted = []
    panel.params_changed.connect(emitted.append)
    panel.controls["image_column"].setCurrentIndex(0)
    assert emitted[-1]["image_column"] == ""


def test_user_level_change_clears_only_newly_inactive_identities(qtbot):
    panel = _panel(
        qtbot,
        summary_level="Sample averages",
        image_column="image_id",
        sample_column="sample_id",
    )
    emitted = []
    panel.params_changed.connect(emitted.append)
    panel.controls["summary_level"].setCurrentText("Image averages")
    assert emitted[-1]["image_column"] == "image_id"
    assert emitted[-1]["sample_column"] == ""
    panel.controls["summary_level"].setCurrentText("Objects")
    assert emitted[-1]["image_column"] == ""
    assert emitted[-1]["sample_column"] == ""


def test_same_immutable_table_does_not_rescan_measurements(qtbot, monkeypatch):
    panel = _panel(qtbot)
    scans = []
    monkeypatch.setattr(
        "napari_vipp.ui.statistics.measurement_columns",
        lambda table: scans.append(table) or (),
    )
    panel.set_state(table=panel.table, params=panel.params)
    assert scans == []


def test_automatic_measurements_exclude_selected_numeric_groups_and_identities(qtbot):
    panel = _panel(qtbot, group_by="dose", image_column="acquisition")
    table = TableData(("area", "dose", "acquisition"), ((4.0, 20.0, 6),))
    panel.set_state(table=table, params=panel.params)
    assert panel._checked(panel.measurements) == ("area",)
    assert "1 available measurements" in panel.input_note.text()
    assert "not automatic" in _item(panel.measurements, "dose").text()
    assert "not automatic" in _item(panel.measurements, "acquisition").text()


def test_invalid_empty_statistic_selection_is_visible_and_not_emitted(qtbot):
    panel = _panel(qtbot, statistics="mean")
    emitted = []
    panel.params_changed.connect(emitted.append)
    _item(panel.statistics, "mean").setCheckState(Qt.Unchecked)
    assert emitted == []
    assert "at least one" in panel.validation_note.text()


def test_measurement_selection_encodes_column_names_with_commas(qtbot):
    panel = _panel(qtbot, value_columns="area")
    table = TableData(("area", "signal, mean"), ((3.0, 4.0),))
    panel.set_state(table=table, params=panel.params)
    emitted = []
    panel.params_changed.connect(emitted.append)
    _item(panel.measurements, "signal, mean").setCheckState(Qt.Checked)
    assert json.loads(emitted[-1]["value_columns"]) == ["area", "signal, mean"]


def test_multiple_group_selection_stays_open_across_owner_refresh(qtbot):
    panel = _panel(qtbot)
    panel.params_changed.connect(
        lambda values: panel.set_state(table=panel.table, params=values)
    )
    panel.multiple_groups.setChecked(True)
    _item(panel.groups, "condition").setCheckState(Qt.Checked)
    assert panel.multiple_groups.isChecked()
    assert panel.groups.isVisible()
    _item(panel.groups, "image_id").setCheckState(Qt.Checked)
    assert set(json.loads(panel.params["group_by"])) == {"condition", "image_id"}


def test_ready_inclusion_is_hidden_when_stale_or_failed(qtbot):
    panel = _panel(qtbot, value_columns="area", statistics="count,mean,std")
    result = summarize_statistics(panel.table, **panel.params)
    panel.set_state(table=panel.table, params=panel.params, result=result, stale=False)
    assert "2 of 3 objects used" in panel.result_group.accessibleDescription()
    assert "1 excluded" in panel.result_group.accessibleDescription()
    assert "2 object values" in panel.result_group.accessibleDescription()
    assert "Current result" in panel.result_note.text()
    panel.set_state(table=panel.table, params=panel.params, result=result, stale=True)
    assert panel.inclusion_note.text() == ""
    assert "Current result" not in panel.result_note.text()
    panel.set_state(
        table=panel.table,
        params=panel.params,
        result=result,
        stale=True,
        failed=True,
        message="Choose image identity column.",
    )
    assert "Choose image identity column" in panel.result_note.text()
    assert panel.inclusion_note.text() == ""


def test_preview_keeps_group_summary_and_n_separate_from_evidence():
    result = summarize_statistics(
        _table(),
        value_columns="area",
        group_by="condition",
        statistics="count,mean,std",
    )
    projection = statistics_preview_columns(result)
    assert tuple(result.columns[index] for index in projection) == (
        "condition",
        "area_n",
        "area_mean",
        "area_std",
    )
    assert "area_object_excluded" in result.columns
    assert "statistics_recipe" in result.columns
    assert statistics_preview_columns(TableData(("area_mean",), ((2.0,),))) is None
