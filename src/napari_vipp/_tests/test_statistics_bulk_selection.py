"""Bulk measurement choices are explicit, atomic and safe across refreshes."""

import json

import pytest

from napari_vipp.core.statistics import StatisticsRecipe
from napari_vipp.core.tables import TableData
from napari_vipp.ui.statistics import StatisticsPanel


def _table():
    return TableData(
        ("label_id", "condition", "area", "intensity"),
        ((1, "control", 10.0, 40.0), (2, "treated", 20.0, 60.0)),
    )


def _panel(qtbot, table=None, **params):
    panel = StatisticsPanel()
    qtbot.addWidget(panel)
    panel.set_state(
        table=table if table is not None else _table(),
        params={**StatisticsRecipe().to_params(), **params},
    )
    panel.resize(380, 1000)
    panel.show()
    return panel


def _owner(panel):
    """Match the owner's synchronous control rebuild when a recipe is changed."""
    edits = []

    def accept(params):
        edits.append(params)
        panel.set_state(table=panel.table, params=params)

    panel.params_changed.connect(accept)
    return edits


@pytest.mark.parametrize("initial", ["auto", "area"])
def test_select_all_is_one_fixed_selection_even_from_auto(qtbot, initial):
    panel = _panel(qtbot, value_columns=initial, group_by="condition")
    before = dict(panel.params)
    edits = _owner(panel)
    assert panel.select_all_measurements_button.isEnabled()

    panel.select_all_measurements_button.click()

    assert len(edits) == 1
    assert json.loads(edits[0]["value_columns"]) == ["area", "intensity"]
    assert {**edits[0], "value_columns": initial} == before
    assert not panel.auto_measurements.isChecked()
    assert panel.measurements.isEnabled()
    assert not panel.select_all_measurements_button.isEnabled()
    assert panel.select_no_measurements_button.isEnabled()
    assert "new columns are not added" in panel.measurement_selection_note.text()


@pytest.mark.parametrize("initial", ["auto", "area"])
def test_select_none_is_explicit_empty_not_automatic(qtbot, initial):
    panel = _panel(qtbot, value_columns=initial)
    edits = _owner(panel)
    panel.select_no_measurements_button.click()

    assert len(edits) == 1
    assert edits[0]["value_columns"] == ""
    assert panel._checked(panel.measurements) == ()
    assert not panel.auto_measurements.isChecked()
    assert panel.measurements.isEnabled()
    assert "Choose at least one" in panel.measurement_selection_note.text()
    assert not panel.select_no_measurements_button.isEnabled()
    panel.select_no_measurements_button.click()
    assert len(edits) == 1


def test_select_all_uses_eligible_measurements_not_every_visible_field(qtbot):
    table = TableData(
        (
            "area",
            "dose",
            "acquisition",
            "replicate",
            "label_id",
            "flag",
            "description",
            "numeric_text",
            "missing",
            "_vipp_origin",
        ),
        ((4.0, 3, 7, 2, 1, True, "cell", "42", None, 5),),
    )
    panel = _panel(
        qtbot,
        table,
        value_columns='["area", "label_id"]',
        group_by="dose",
        image_column="acquisition",
        sample_column="replicate",
    )
    edits = _owner(panel)
    assert panel.measurements.count() == len(table.columns)
    panel.select_all_measurements_button.click()

    assert len(edits) == 1
    assert json.loads(edits[0]["value_columns"]) == ["area"]
    assert panel._checked(panel.measurements) == ("area",)
    assert (
        "skips IDs, text and grouping fields" in panel.measurement_selection_note.text()
    )


def test_bulk_selection_preserves_exact_raw_column_names(qtbot):
    names = ("auto", "signal, mean", 'quoted "intensity"', "μm² intensity")
    panel = _panel(qtbot, TableData(names, ((1.0, 2.0, 3.0, 4.0),)))
    edits = _owner(panel)
    panel.select_all_measurements_button.click()

    assert len(edits) == 1
    assert tuple(json.loads(edits[0]["value_columns"])) == names
    assert panel._checked(panel.measurements) == names
    assert not panel.auto_measurements.isChecked()


def test_bulk_and_auto_changes_do_not_rescan_cached_input_or_emit_storms(
    qtbot, monkeypatch
):
    panel = _panel(qtbot)
    edits = _owner(panel)

    def unexpected_scan(_table):
        pytest.fail("Selection edits must reuse the cached eligible measurements")

    monkeypatch.setattr(
        "napari_vipp.ui.statistics.measurement_columns", unexpected_scan
    )
    panel.select_all_measurements_button.click()
    panel.select_no_measurements_button.click()
    panel.auto_measurements.setChecked(True)
    panel.auto_measurements.setChecked(False)

    assert len(edits) == 4
    assert [entry["value_columns"] for entry in edits] == [
        '["area", "intensity"]',
        "",
        "auto",
        '["area", "intensity"]',
    ]
    assert panel._checked(panel.measurements) == ("area", "intensity")


def test_fixed_selection_does_not_grow_until_auto_is_explicitly_enabled(qtbot):
    panel = _panel(qtbot)
    edits = _owner(panel)
    panel.select_all_measurements_button.click()
    fixed = dict(panel.params)
    table = TableData(
        (*panel.table.columns, "volume"),
        tuple((*row, 100.0) for row in panel.table.rows),
    )
    panel.set_state(table=table, params=fixed)

    assert len(edits) == 1
    assert panel.params == fixed
    assert panel._checked(panel.measurements) == ("area", "intensity")
    assert not panel.auto_measurements.isChecked()
    panel.auto_measurements.setChecked(True)
    assert len(edits) == 2
    assert edits[-1]["value_columns"] == "auto"
    assert panel._checked(panel.measurements) == ("area", "intensity", "volume")
    assert not panel.measurements.isEnabled()
    assert panel.auto_measurements.text() == "Auto-select measurements"
    assert "including new columns" in panel.measurement_selection_note.text()


@pytest.mark.parametrize("initial", ["auto", "area,missing"])
def test_no_input_can_clear_saved_choices_but_cannot_select_all(qtbot, initial):
    panel = _panel(qtbot, value_columns=initial)
    panel.set_state(table=None, params=panel.params)
    edits = _owner(panel)
    assert not panel.select_all_measurements_button.isEnabled()
    assert panel.select_no_measurements_button.isEnabled()
    panel.select_all_measurements_button.click()
    assert edits == []
    panel.select_no_measurements_button.click()
    assert len(edits) == 1
    assert edits[0]["value_columns"] == ""
    assert panel._checked(panel.measurements) == ()
    assert not panel.auto_measurements.isChecked()


@pytest.mark.parametrize("rows", [(), ((1, "cell", True),)])
def test_no_eligible_measurements_disables_select_all(qtbot, rows):
    panel = _panel(qtbot, TableData(("label_id", "description", "flag"), rows))
    edits = _owner(panel)
    assert not panel.select_all_measurements_button.isEnabled()
    panel.select_no_measurements_button.click()
    assert len(edits) == 1
    assert edits[0]["value_columns"] == ""
    assert not panel.select_no_measurements_button.isEnabled()


@pytest.mark.parametrize("guard", ["legacy", "unavailable"])
def test_bulk_selection_cannot_edit_legacy_or_unavailable_panel(qtbot, guard):
    panel = _panel(qtbot)
    if guard == "legacy":
        panel.set_state(
            table=panel.table, params={**panel.params, "summary_version": 1}
        )
    else:
        panel.set_unavailable("The source was removed.")
    before = dict(panel.params)
    edits = _owner(panel)
    panel.select_all_measurements_button.click()
    panel.select_no_measurements_button.click()
    panel._set_all_measurements_selected(True)
    panel._set_all_measurements_selected(False)
    assert edits == []
    assert panel.params == before
