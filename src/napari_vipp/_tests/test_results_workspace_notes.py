"""Collapsible plot advice stays separate from errors and scientific state."""

from __future__ import annotations

from dataclasses import replace

import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLabel

from napari_vipp._tests.test_results_workspace_dialog import _dialog, _table
from napari_vipp.core.result_plots import build_plot_result
from napari_vipp.core.statistics import summarize_statistics


def _summary(**changes):
    return summarize_statistics(
        _table(),
        **{
            "group_by": "condition",
            "value_columns": "area",
            "statistics": "mean,median",
            **changes,
        },
    )


def _result(table=None, **changes):
    return build_plot_result(
        table if table is not None else _summary(),
        **{"y_column": "area_mean", "group_column": "condition", **changes},
    )


def _publish(dialog, result, *, table, **state):
    dialog.set_plot(
        table=table,
        params=result.recipe.to_params(),
        result=result,
        source_kind="summary",
        **{"stale": False, **state},
    )


def _ready(qtbot):
    dialog = _dialog(qtbot)
    dialog.show_tab("plots")
    table = _summary()
    result = _result(table)
    _publish(dialog, result, table=table)
    return dialog, result, table


def test_input_details_start_collapsed_and_can_be_reopened_without_analysis(qtbot):
    dialog, result, _table = _ready(qtbot)
    signals = []
    dialog.plot_params_changed.connect(signals.append)
    assert not dialog.plot_input_toggle.isChecked()
    assert dialog.plot_source_card.isVisible()
    assert dialog.plot_source_card.isAncestorOf(dialog.plot_input_toggle)
    assert not dialog.plot_source_note.isVisible()
    assert not any(
        label.isVisible() for label in dialog.plot_source_card.findChildren(QLabel)
    )
    assert dialog.plot_source.currentText() in dialog.plot_input_toggle.text()
    assert "2 rows" in dialog.plot_input_toggle.text()
    assert "2 summary rows" in dialog.plot_input_toggle.toolTip()
    dialog.plot_input_toggle.click()
    assert dialog.plot_source_card.isVisible()
    assert dialog.plot_source_note.isVisible()
    assert "2 summary rows" in dialog.plot_source_note.text()
    dialog.plot_input_toggle.click()
    assert dialog.plot_source_card.isVisible()
    assert not dialog.plot_source_note.isVisible()
    assert dialog._plot_result is result
    assert signals == []


def test_input_header_supports_keyboard_disclosure_without_an_extra_row(qtbot):
    dialog, result, _table = _ready(qtbot)
    header = dialog.plot_input_toggle
    assert header.isCheckable()
    assert header.isVisible()
    assert dialog.plot_source_card.isAncestorOf(header)
    chevron = header.chevronRect()
    assert header.rect().contains(chevron)
    assert chevron.center().x() > header.width() * 0.8
    assert header.width() - chevron.right() <= 20
    header.setFocus(Qt.OtherFocusReason)
    qtbot.keyClick(header, Qt.Key_Space)
    assert header.isChecked()
    assert dialog.plot_source_note.isVisible()
    assert dialog.plot_source_card.isVisible()
    assert "Hide" in header.accessibleName()
    qtbot.keyClick(header, Qt.Key_Space)
    assert not header.isChecked()
    assert not dialog.plot_source_note.isVisible()
    assert dialog.plot_source_card.isVisible()
    assert "Show" in header.accessibleName()
    assert dialog._plot_result is result


def test_collapsing_input_keeps_header_but_returns_space_to_chart(qtbot):
    dialog, _result_data, _table = _ready(qtbot)
    dialog.resize(1280, 900)
    dialog.plot_notes_dismiss_button.click()
    dialog.plot_input_toggle.click()
    qtbot.waitUntil(lambda: dialog.plot_source_note.isVisible())
    qtbot.wait(30)
    expanded_card_height = dialog.plot_source_card.height()
    expanded_chart_height = dialog.plot_canvas.height()
    dialog.plot_input_toggle.click()
    qtbot.waitUntil(lambda: dialog.plot_source_card.height() < expanded_card_height)
    qtbot.waitUntil(lambda: dialog.plot_canvas.height() > expanded_chart_height)
    assert dialog.plot_source_card.isVisible()
    assert dialog.plot_input_toggle.isVisible()
    assert not dialog.plot_source_note.isVisible()


def test_dismissal_survives_refresh_resize_tabs_and_presentation_redraws(qtbot):
    dialog, result, table = _ready(qtbot)
    signals = []
    dialog.plot_params_changed.connect(signals.append)
    assert dialog.plot_notes_frame.isVisible()
    assert dialog.plot_warning.isVisible()
    warnings = result.warnings
    dialog.plot_notes_dismiss_button.click()
    assert not dialog.plot_notes_frame.isVisible()
    assert dialog.plot_notes_reopen_button.isVisible()

    for tab in ("data", "summary", "plots"):
        dialog.show_tab(tab)
        dialog.resize(960, 650)
        _publish(dialog, result, table=table)
        assert dialog._plot_notice_state.dismissed
    for change in (
        {"title": "A clearer figure title"},
        {"point_size": 8.0},
        {"show_grid": False},
        {"y_tick_interval": "10"},
    ):
        result = build_plot_result(table, **{**result.recipe.to_params(), **change})
        # PlotData freezes its own copy; the connected input stays the same.
        assert result.source_table is not table
        _publish(dialog, result, table=table)
        assert dialog._plot_notice_state.dismissed
        assert not dialog.plot_notes_frame.isVisible()
        assert dialog.plot_notes_reopen_button.isVisible()
    assert result.warnings == warnings
    assert signals == []
    dialog.plot_notes_reopen_button.click()
    assert dialog.plot_notes_frame.isVisible()
    assert not dialog.plot_notes_reopen_button.isVisible()
    assert not dialog._plot_notice_state.dismissed


@pytest.mark.parametrize("revision", ["input", "statistics", "grouping", "warning"])
def test_changed_analysis_or_advice_shows_notes_again(qtbot, revision):
    dialog, result, table = _ready(qtbot)
    dialog.plot_notes_dismiss_button.click()
    if revision == "input":
        table = _summary()  # Equal values still represent a new published input.
        updated = _result(table)
    elif revision == "statistics":
        table = _summary(statistics="mean,median,std")
        updated = _result(table)
    elif revision == "grouping":
        updated = _result(table, group_column="")
    else:
        updated = replace(result, warnings=(*result.warnings, "New analysis warning."))
    _publish(dialog, updated, table=table)
    assert not dialog._plot_notice_state.dismissed
    assert dialog.plot_notes_frame.isVisible()
    assert not dialog.plot_notes_reopen_button.isVisible()


@pytest.mark.parametrize(
    "transition",
    [
        {"stale": True, "message": "Inputs changed; calculate again."},
        {"busy": True, "message": "Updating plot…"},
        {"failed": True, "message": "The selected measurement is unavailable."},
    ],
)
def test_stale_busy_and_failed_results_do_not_replace_dismissal_revision(
    qtbot, transition
):
    dialog, result, table = _ready(qtbot)
    dialog.plot_notes_dismiss_button.click()
    candidate_table = _summary()
    outdated_candidate = _result(candidate_table)
    _publish(dialog, outdated_candidate, table=candidate_table, **transition)
    assert dialog._plot_notice_state.dismissed
    assert not dialog.plot_notes_frame.isVisible()
    assert not dialog.plot_notes_reopen_button.isVisible()
    assert not dialog.plot_notes_dismiss_button.isEnabled()
    assert not dialog.plot_notes_reopen_button.isEnabled()
    if transition.get("failed"):
        assert dialog.plot_canvas.error_view.isVisible()
        assert "measurement" in (
            dialog.plot_canvas.error_title.text()
            + dialog.plot_canvas.error_detail.text()
        )
    else:
        assert dialog.plot_status.isVisible()
        assert transition["message"] in dialog.plot_status.text()
    assert not dialog.export_button.isEnabled()
    # Restoring the same reviewed result remains acknowledged. Only publishing
    # the candidate as a valid current result establishes a new revision.
    _publish(dialog, result, table=table)
    assert dialog._plot_notice_state.dismissed
    _publish(dialog, outdated_candidate, table=candidate_table)
    assert not dialog._plot_notice_state.dismissed
    assert dialog.plot_notes_frame.isVisible()


def test_unavailable_or_globally_busy_workspace_hides_advice_without_forgetting(qtbot):
    dialog, result, _table = _ready(qtbot)
    dialog.plot_notes_dismiss_button.click()
    dialog.set_available(False, "Return to the source workflow.")
    assert not dialog.plot_notes_frame.isVisible()
    assert not dialog.plot_notes_reopen_button.isVisible()
    assert dialog.availability_label.isVisible()
    assert dialog._plot_notice_state.dismissed
    dialog.set_available(True)
    assert dialog.plot_notes_reopen_button.isVisible()
    dialog.set_busy(True, "Updating results…")
    assert not dialog.plot_notes_reopen_button.isVisible()
    assert dialog._plot_notice_state.dismissed
    dialog.set_busy(False)
    assert dialog.plot_notes_reopen_button.isVisible()
    assert dialog._plot_result is result


def test_summary_explanation_is_neutral_but_exclusions_keep_caution_tone(qtbot):
    dialog, result, table = _ready(qtbot)
    assert not dialog.plot_notes_frame.property("caution")
    assert "summary row" in dialog.plot_warning.text()
    updated = replace(result, warnings=(*result.warnings, "Excluded 2 invalid rows."))
    _publish(dialog, updated, table=table)
    assert dialog.plot_notes_frame.property("caution")
    assert "Excluded 2 invalid rows" in dialog.plot_warning.text()
