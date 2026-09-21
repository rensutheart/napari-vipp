"""Acknowledging chart advice never changes or retains scientific data."""

from __future__ import annotations

import gc
from dataclasses import replace
from weakref import ref

import pytest

from napari_vipp.core.result_plots import PlotRecipe, build_plot_result
from napari_vipp.core.tables import TableData
from napari_vipp.ui.plot_notice_state import (
    PlotNoticeState,
    plot_notes_are_caution,
    visible_plot_warnings,
)


def _table():
    return TableData(
        ("area", "intensity", "condition"),
        ((10.0, 20.0, "Control"), (30.0, 40.0, "Treated")),
    )


def _params():
    return PlotRecipe(
        plot_type="Scatter", x_column="intensity", y_column="area"
    ).to_params()


_WARNING = ("Excluded 1 row: 1 missing value.",)


@pytest.mark.parametrize(
    "change",
    [
        {"title": "A clearer title"},
        {"point_size": 9.0},
        {"show_grid": False},
        {"x_tick_interval": "10"},
        {"y_tick_interval": "5"},
    ],
)
def test_appearance_changes_keep_the_same_advice_acknowledged(change):
    state, table, params = PlotNoticeState(), _table(), _params()
    assert not state.update("plot", table, params, _WARNING)
    state.dismiss()
    assert state.update("plot", table, {**params, **change}, _WARNING)
    assert state.dismissed
    # Updating or reverting the display is not another scientific revision.
    assert state.update("plot", table, params, _WARNING)


@pytest.mark.parametrize(
    "change",
    [
        {"group_column": "condition"},
        {"y_column": "intensity"},
        {"x_column": "area"},
        {"log_x": True},
        {"log_y": True},
        {"summary": "Median"},
        {"plot_type": "Distribution"},
        {"distribution": "Cumulative"},
        {"normalization": "Percent"},
        {"bins": 40},
        {"point_unit": "Mean per image", "image_column": "image_id"},
        {"future_analysis_setting": "new policy"},
    ],
)
def test_analysis_changes_show_notes_again_even_if_warning_text_is_unchanged(change):
    state, table, params = PlotNoticeState(), _table(), _params()
    state.update("plot", table, params, _WARNING)
    state.dismiss()
    assert not state.update("plot", table, {**params, **change}, _WARNING)
    assert not state.dismissed


def test_new_equal_input_is_a_new_revision_and_old_inputs_are_not_retained():
    state, table, params = PlotNoticeState(), _table(), _params()
    state.update("plot", table, params, _WARNING)
    state.dismiss()
    new_table = _table()
    assert new_table == table and new_table is not table
    assert not state.update("plot", new_table, params, _WARNING)
    state.dismiss()
    tracked = ref(new_table)
    del new_table
    gc.collect()
    assert tracked() is None
    assert not state.update("plot", table, params, _WARNING)


def test_acknowledgements_belong_to_individual_plots_and_can_be_reopened():
    state, table, params = PlotNoticeState(), _table(), _params()
    state.update("first", table, params, _WARNING)
    state.dismiss()
    assert not state.update("second", table, params, _WARNING)
    state.dismiss()
    assert state.update("first", table, params, _WARNING)
    state.reopen()
    assert not state.dismissed
    assert state.update("second", table, params, _WARNING)
    assert not state.update("first", table, params, _WARNING)


def test_changed_warning_content_and_order_require_review_again():
    state, table, params = PlotNoticeState(), _table(), _params()
    warnings = (*_WARNING, "Display shows a sample of the available points.")
    state.update("plot", table, params, warnings)
    state.dismiss()
    assert not state.update("plot", table, params, tuple(reversed(warnings)))
    state.dismiss()
    assert not state.update("plot", table, params, (*warnings, "New warning."))


@pytest.mark.parametrize("empty_input", [False, True])
def test_missing_input_or_no_warnings_clears_only_that_plots_acknowledgement(
    empty_input,
):
    state, table, params = PlotNoticeState(), _table(), _params()
    for plot_id in ("first", "second"):
        state.update(plot_id, table, params, _WARNING)
        state.dismiss()
    assert not state.update(
        "first", None if empty_input else table, params, _WARNING if empty_input else ()
    )
    assert not state.dismissed
    state.dismiss()  # No currently visible advice to acknowledge.
    assert not state.dismissed
    assert state.update("second", table, params, _WARNING)
    assert not state.update("first", table, params, _WARNING)


def test_state_does_not_scan_rows_or_compare_tables():
    class UnscannableRows(tuple):
        def __iter__(self):
            raise AssertionError("UI acknowledgement must not scan measurements")

        def __eq__(self, other):
            raise AssertionError("UI acknowledgement must not compare measurements")

    table = TableData(("value",), UnscannableRows(((1,), (2,))))
    state, params = PlotNoticeState(), _params()
    state.update("plot", table, params, _WARNING)
    state.dismiss()
    assert state.update("plot", table, params, _WARNING)


def test_advice_filter_and_tone_match_displayed_warnings():
    result = build_plot_result(_table(), **_params())
    assert visible_plot_warnings(result) == ()
    summary_note = (
        "Each point represents one summary row, not an original object or "
        "independent sample."
    )
    result = replace(result, warnings=(*result.warnings, summary_note))
    assert visible_plot_warnings(result) == (summary_note,)
    assert not plot_notes_are_caution(visible_plot_warnings(result))
    result = replace(result, warnings=(*result.warnings, *_WARNING))
    assert plot_notes_are_caution(visible_plot_warnings(result))
    assert visible_plot_warnings(None) == ()
