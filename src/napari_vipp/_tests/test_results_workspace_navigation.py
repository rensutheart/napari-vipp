"""Workspace navigation is distinct from editing a plot's graph connection."""

import pytest
from qtpy.QtCore import QPoint, Qt
from qtpy.QtWidgets import QComboBox, QLabel

from napari_vipp._tests.test_plot_warnings import _palette
from napari_vipp._tests.test_results_workspace_dialog import _dialog, _table
from napari_vipp.core.result_plots import build_plot_result
from napari_vipp.ui.palette_roles import theme_colors


def _choices(dialog, *, plot=True):
    dialog.set_choices(
        data_sources=[
            (("measure", "table"), "Cell measurements"),
            (("other", "table"), "Mitochondria measurements"),
        ],
        data_source=("measure", "table"),
        summaries=[("summary", "By condition")],
        summary_id="summary",
        plot_scopes=[
            ("measure", "Original measurements"),
            ("summary", "Summary: By condition"),
        ],
        plot_scope_id="summary",
        plots=[("plot", "Mean area")] if plot else [],
        plot_id="plot" if plot else "",
        plot_sources=[
            ("measure", "Original measurements"),
            ("summary", "Summary: By condition"),
        ],
        plot_source_id="summary",
    )


def test_populating_navigation_does_not_change_connections_or_emit_selections(qtbot):
    dialog = _dialog(qtbot)
    signals = []
    for signal in (
        dialog.data_selected,
        dialog.plot_scope_selected,
        dialog.summary_selected,
        dialog.plot_selected,
        dialog.plot_source_changed,
    ):
        signal.connect(signals.append)
    _choices(dialog)
    assert tuple(dialog.data_selector.currentData()) == ("measure", "table")
    assert dialog.plot_scope.currentData() == "summary"
    assert dialog.data_source_label.width() > 0
    assert signals == []


def test_python_tuple_source_choice_reliably_selects_nonfirst_output(qtbot):
    dialog = _dialog(qtbot)
    dialog.set_choices(
        data_sources=[
            (("measure", "table"), "Cell measurements"),
            (("other", "objects"), "Mitochondria measurements"),
        ],
        data_source=("other", "objects"),
    )
    assert dialog.data_selector.currentIndex() == 1
    assert dialog.data_selector.currentData() == ("other", "objects")
    assert dialog.data_selector.currentText() == "Mitochondria measurements"


def test_data_and_summary_navigation_are_separate_from_rewiring(qtbot):
    dialog = _dialog(qtbot)
    _choices(dialog)
    sources, summaries, connections = [], [], []
    dialog.data_selected.connect(sources.append)
    dialog.summary_selected.connect(summaries.append)
    dialog.plot_source_changed.connect(connections.append)
    dialog.data_selector.setCurrentIndex(1)
    dialog.summary_selector.setCurrentIndex(0)
    assert [tuple(source) for source in sources] == [("other", "table")]
    assert summaries == [""]
    assert connections == []
    assert "view only" in dialog.data_selector.toolTip()
    assert "does not change" in dialog.summary_selector.toolTip()
    assert "changes" in dialog.plot_source.toolTip()
    assert dialog.plot_source.accessibleName() == "Change plot input"
    dialog.plot_source.setCurrentIndex(0)
    assert connections == ["measure"]


def test_unplotted_summary_has_explicit_empty_state_and_cannot_show_old_node(qtbot):
    dialog = _dialog(qtbot)
    table = _table()
    result = build_plot_result(table, y_column="area")
    dialog.set_plot(
        table=table, params=result.recipe.to_params(), result=result, stale=False
    )
    dialog.show_tab("plots")
    assert dialog.plot_canvas.isVisible()
    _choices(dialog, plot=False)
    dialog.set_relationship(
        "Cell measurements → By condition → No connected plots",
        plot_context="By condition",
        has_plot=False,
    )
    dialog.set_plot()
    assert "By condition" in dialog.plot_setup_note.text()
    assert "No plots are connected" in dialog.plot_setup_note.text()
    assert "+ beside Plot" in dialog.plot_setup_note.text()
    assert "Statistics node above" in dialog.plot_setup_note.text()
    assert dialog.plot_setup_heading.text() == "No connected plots"
    assert dialog.plot_setup_card.isVisible()
    assert not dialog.plot_info.isVisible()
    assert not dialog.plot_controls.isVisible()
    assert not dialog.plot_canvas.isVisible()
    assert not dialog.choose_measurement_button.isVisible()
    assert not dialog.plotted_data_button.isVisible()
    assert not dialog.show_node_button.isEnabled()
    assert not dialog.recalculate_button.isEnabled()
    assert not dialog.export_button.isEnabled()
    assert dialog.current_node_id() == ""
    assert dialog.plot_canvas.result is None
    assert dialog._plot_result is None
    assert dialog.plot_status.text() == ""
    assert dialog.plot_warning.text() == ""
    added = []
    dialog.add_plot_requested.connect(added.append)
    dialog.add_plot_button.click()
    assert added == ["summary"]


def test_returning_to_connected_plot_restores_chart_without_stale_empty_state(qtbot):
    dialog = _dialog(qtbot)
    _choices(dialog, plot=False)
    dialog.set_relationship(
        "Cell measurements → By condition", plot_context="By condition"
    )
    dialog.show_tab("plots")
    _choices(dialog)
    table = _table()
    result = build_plot_result(table, y_column="area")
    dialog.set_plot(
        table=table, params=result.recipe.to_params(), result=result, stale=False
    )
    assert dialog.plot_info.isVisible()
    assert dialog.plot_canvas.isVisible()
    assert not dialog.plot_setup_card.isVisible()
    assert dialog.show_node_button.isEnabled()
    assert dialog.current_node_id() == "plot"
    assert dialog._plot_result is result


def test_deleted_selection_cannot_target_show_node_or_calculation(qtbot):
    dialog = _dialog(qtbot)
    dialog.set_choices(summary_id="deleted-summary", plot_id="deleted-plot")
    for tab in ("summary", "plots"):
        dialog.show_tab(tab)
        assert dialog.current_node_id() == ""
        assert not dialog.show_node_button.isEnabled()
        assert not dialog.recalculate_button.isEnabled()


def test_unavailable_selected_plot_does_not_claim_other_connected_plots_are_absent(
    qtbot,
):
    dialog = _dialog(qtbot)
    dialog.set_choices(
        plots=[("other-plot", "An existing connected plot")],
        plot_id="deleted-plot",
        plot_scopes=[("summary", "Summary: By condition")],
        plot_scope_id="summary",
    )
    dialog.set_relationship(
        "Cell measurements → By condition", plot_context="By condition"
    )
    dialog.show_tab("plots")
    assert dialog.plot_selector.currentData() == "deleted-plot"
    assert dialog.plot_selector.itemData(0) == "other-plot"
    assert dialog.plot_setup_heading.text() == "Plot unavailable"
    assert "removed or is no longer connected" in dialog.plot_setup_note.text()
    assert "Choose another connected plot" in dialog.plot_setup_note.text()
    assert "No plots are connected" not in dialog.plot_setup_note.text()
    assert not dialog.show_node_button.isEnabled()


def test_navigation_unavailable_without_active_workflow_choices(qtbot):
    dialog = _dialog(qtbot)
    dialog.set_choices()
    dialog.set_available(False)
    assert not dialog.data_selector.isEnabled()
    assert not dialog.plot_scope.isEnabled()
    assert not dialog.plot_source.isEnabled()


@pytest.mark.parametrize("width", [760, 1280])
@pytest.mark.parametrize("base,text", [("#23242b", "#f0f1f2"), ("#fafafa", "#20252d")])
def test_prominent_connection_bar_has_four_compact_controls_on_every_tab(
    qtbot, width, base, text
):
    dialog = _dialog(qtbot)
    dialog.setPalette(_palette(base, text))
    dialog.refresh_theme()
    dialog.set_workflows([("workflow", "Cell measurements")], "workflow")
    _choices(dialog)
    dialog.resize(width, 700)
    dialog.set_relationship("Cell measurements → By condition → Mean area")
    qtbot.wait(20)
    choices = (
        dialog.workflow_selector,
        dialog.data_selector,
        dialog.summary_selector,
        dialog.plot_selector,
    )
    assert dialog.connection_bar.findChildren(QComboBox) == list(choices)
    assert "border-radius: 0" in dialog.connection_bar.styleSheet()
    assert "border:" in dialog.connection_bar.styleSheet()
    assert (
        f"solid {theme_colors(dialog.palette()).info.accent.name()}"
        in dialog.connection_bar.styleSheet()
    )
    assert [label.text() for label in dialog.connection_labels] == [
        "Workflow",
        "Data source",
        "Statistics node",
        "Plot",
    ]
    assert all(label.font().bold() for label in dialog.connection_labels)
    assert len(dialog.connection_arrows) == 3
    assert all(not isinstance(arrow, QLabel) for arrow in dialog.connection_arrows)
    assert dialog.connection_bar.accessibleDescription().endswith("Mean area")
    assert not dialog.connection_label.isVisible()
    assert not dialog.plot_scope.isVisible()
    for tab in ("data", "summary", "plots"):
        dialog.show_tab(tab)
        qtbot.wait(10)
        for combo in choices:
            assert combo.isVisible()
            assert 95 <= combo.width() <= 340
            top = combo.mapTo(dialog, QPoint())
            assert top.y() < dialog.tabs.y()
        assert len({combo.mapTo(dialog, QPoint()).y() for combo in choices}) == 1
        for index, arrow in enumerate(dialog.connection_arrows):
            assert arrow.isVisible()
            assert arrow.width() >= 28
            center = arrow.mapTo(dialog, arrow.arrowCenter())
            target_center = choices[index + 1].mapTo(
                dialog, choices[index + 1].rect().center()
            )
            assert abs(center.y() - target_center.y()) <= 1
            previous_right = (
                dialog.connection_groups[index]
                .mapTo(dialog, dialog.connection_groups[index].rect().topRight())
                .x()
            )
            next_left = choices[index + 1].mapTo(dialog, QPoint()).x()
            assert previous_right < center.x() < next_left
        assert dialog.add_summary_button.isVisible()
        assert dialog.add_plot_button.isVisible()
        assert not dialog.plot_source.isVisible()


@pytest.mark.parametrize("button_height", [26, 31])
@pytest.mark.parametrize("width", [760, 1280])
def test_connection_selectors_stay_aligned_with_taller_native_action_buttons(
    qtbot, button_height, width
):
    dialog = _dialog(qtbot)
    _choices(dialog)
    choices = (
        dialog.workflow_selector,
        dialog.data_selector,
        dialog.summary_selector,
        dialog.plot_selector,
    )
    # Native styles need not give QPushButton and QComboBox the same height.
    # Reproduce both an odd/even 1px difference and a larger native-style gap
    # on every platform, without relying on the runner's installed styles.
    for combo in choices:
        combo.setFixedHeight(25)
    buttons = (dialog.add_summary_button, dialog.add_plot_button)
    for button in buttons:
        button.setFixedHeight(button_height)
    dialog.resize(width, 700)

    for tab in ("data", "summary", "plots"):
        dialog.show_tab(tab)
        qtbot.wait(10)
        assert len({combo.mapTo(dialog, QPoint()).y() for combo in choices}) == 1
        assert len({label.height() for label in dialog.connection_labels}) == 1
        for button, combo in zip(buttons, choices[2:], strict=True):
            assert button.height() == button_height
            button_center = button.mapTo(dialog, button.rect().center())
            combo_center = combo.mapTo(dialog, combo.rect().center())
            # QRect centers use integer coordinates; mixed odd/even heights
            # may round the same visual center to adjacent pixels.
            assert abs(button_center.y() - combo_center.y()) <= 1
        for arrow, combo in zip(dialog.connection_arrows, choices[1:], strict=True):
            assert arrow.mapTo(dialog, arrow.arrowCenter()).y() == combo.mapTo(
                dialog, combo.rect().center()
            ).y()


def test_none_middle_choice_is_explicit_and_add_plot_uses_data_not_old_scope(qtbot):
    dialog = _dialog(qtbot)
    _choices(dialog)
    dialog.set_choices(
        summaries=[("summary", "By condition")],
        summary_id="",
        plot_scopes=[("summary", "An old summary context")],
        plot_scope_id="summary",
        plot_sources=[("summary", "An old plot input")],
        plot_source_id="summary",
    )
    assert dialog.summary_selector.currentText() == "None — use input data"
    assert dialog.summary_selector.currentData() == ""
    added = []
    dialog.add_plot_requested.connect(added.append)
    dialog.add_plot_button.click()
    assert added == ["measure"]
    dialog.show_tab("summary")
    assert dialog.summary_empty.isVisible()
    assert "connection bar" in dialog.summary_empty.text()
    assert "+ beside Statistics node" in dialog.summary_empty.text()
    assert not dialog.statistics_panel.isVisible()
    assert not dialog.show_node_button.isEnabled()


def test_connection_change_is_a_collapsed_deliberate_action(qtbot):
    dialog = _dialog(qtbot)
    _choices(dialog)
    dialog.show_tab("plots")
    assert not dialog.plot_source.isVisible()
    assert not dialog.plot_connection_editor.isVisible()
    changes = []
    dialog.plot_source_changed.connect(changes.append)
    dialog.plot_connection_button.click()
    assert dialog.plot_connection_editor.isVisible()
    assert dialog.plot_source.isVisible()
    assert changes == []
    dialog.plot_source.setCurrentIndex(0)
    assert changes == ["measure"]
    dialog.set_busy(True)
    assert not dialog.add_summary_button.isEnabled()
    assert not dialog.add_plot_button.isEnabled()
    assert not dialog.plot_connection_button.isEnabled()


def test_long_selected_names_are_available_without_expanding_the_bar(qtbot):
    dialog = _dialog(qtbot)
    name = "Area and fluorescence measurements for the mitochondria experiment " * 4
    dialog.set_choices(
        data_sources=[(("measure", "table"), name)],
        data_source=("measure", "table"),
    )
    dialog.resize(760, 520)
    qtbot.wait(10)
    assert dialog.data_selector.width() <= 340
    assert name in dialog.data_selector.toolTip()
    assert dialog.data_selector.itemData(0, Qt.ToolTipRole) == name


def test_node_choice_details_update_without_changing_names_or_selection(qtbot):
    dialog = _dialog(qtbot)
    selections = []
    dialog.summary_selected.connect(selections.append)
    first = "Area by treatment\nOperation: Statistics\nMean, SD\nNode ID: stable-id"
    choices = dict(
        summaries=[("stable-id", "Area by treatment")],
        summary_id="stable-id",
    )
    dialog.set_choices(**choices, choice_tooltips={"stable-id": first})
    index = dialog.summary_selector.currentIndex()
    assert dialog.summary_selector.itemData(index, Qt.ToolTipRole) == first
    assert first in dialog.summary_selector.toolTip()
    revised = first.replace("Mean, SD", "Median")
    dialog.set_choices(**choices, choice_tooltips={"stable-id": revised})
    assert dialog.summary_selector.currentText() == "Area by treatment"
    assert revised in dialog.summary_selector.toolTip()
    assert "Mean, SD" not in dialog.summary_selector.toolTip()
    assert selections == []


def test_each_table_port_can_have_its_own_full_node_description(qtbot):
    dialog = _dialog(qtbot)
    outputs = [
        (("multi", 0), "Measures · Objects"),
        (("multi", 1), "Measures · Counts"),
    ]
    descriptions = {
        ("multi", 0): "Measures\nOperation: Measure\nOutput: Objects (port 1)",
        ("multi", 1): "Measures\nOperation: Measure\nOutput: Counts (port 2)",
    }
    dialog.set_choices(
        data_sources=outputs, data_source=("multi", 1), choice_tooltips=descriptions
    )
    assert dialog.data_selector.currentData() == ("multi", 1)
    assert descriptions[("multi", 1)] in dialog.data_selector.toolTip()
    assert (
        dialog.data_selector.itemData(0, Qt.ToolTipRole) == descriptions[("multi", 0)]
    )
