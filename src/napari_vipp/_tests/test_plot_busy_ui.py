"""Progress is visible for real work without making old plots exportable."""

import pytest

from napari_vipp.core.result_plots import build_plot_result
from napari_vipp.core.tables import TableData
from napari_vipp.ui.result_plots import PlotResultsPanel


@pytest.fixture
def panel(qtbot):
    table = TableData(("area",), ((12.5,), (18.0,), (21.5,)))
    result = build_plot_result(table)
    widget = PlotResultsPanel(table, result.recipe.to_params(), result)
    qtbot.addWidget(widget)
    widget.show()
    widget.open_plot()
    return widget


def test_busy_keeps_previous_drawing_but_blocks_stale_actions(panel):
    result = panel.result
    canvas = panel.dialog.plot.canvas
    panel.dialog.data_button.setChecked(True)
    panel.set_state(result=result, busy=True)
    assert panel.busy and panel.stale and not panel.failed
    assert panel.dialog.plot.canvas is canvas
    assert panel.result is result
    assert panel.table is result.source_table
    assert not panel.dialog.export_button.isEnabled()
    assert not panel.dialog.data_button.isEnabled()
    assert not panel.dialog.data_view.isVisible()
    assert not panel.dialog.point_label.isVisible()
    assert "Previous plot" in panel.summary.text()
    for indicator in (panel.busy_indicator, panel.dialog.busy_indicator):
        assert indicator.isVisible()
        assert indicator.label.text() == "Updating plot…"
        assert indicator.progress.minimum() == indicator.progress.maximum() == 0
        assert not indicator.progress.isTextVisible()
        assert "Updating plot" in indicator.accessibleName()

    panel.set_result(result)
    assert not panel.busy and not panel.stale
    assert panel.dialog.export_button.isEnabled()
    assert panel.dialog.data_button.isEnabled()
    for indicator in (panel.busy_indicator, panel.dialog.busy_indicator):
        assert indicator.isHidden()
        assert indicator.progress.maximum() == 1


@pytest.mark.parametrize("terminal", ["failure", "stale", "cancelled"])
def test_busy_stops_for_terminal_or_idle_stale_state(panel, terminal):
    panel.set_state(result=panel.result, busy=True)
    if terminal == "failure":
        # A failure wins even if an obsolete busy flag arrives with it.
        panel.set_state(result=panel.result, busy=True, failed=True, message="No data.")
        assert panel.dialog.plot.error_view.isVisible()
    elif terminal == "cancelled":
        panel.set_stale("Calculation cancelled. Calculate again when ready.")
    else:
        panel.set_stale()
    assert not panel.busy
    assert panel.busy_indicator.isHidden()
    assert panel.dialog.busy_indicator.isHidden()
    assert not panel.dialog.export_button.isEnabled()


def test_busy_can_open_before_measurements_arrive_and_clears_placeholder(qtbot):
    panel = PlotResultsPanel()
    qtbot.addWidget(panel)
    panel.set_state(busy=True, busy_message="Waiting for updated measurements…")
    assert panel.open_button.isEnabled()
    window = panel.open_plot()
    assert window is not None
    assert window.busy_indicator.isVisible()
    assert "Waiting for updated measurements" in window.busy_indicator.label.text()
    assert "Preparing the updated plot" in window.plot.placeholder.text()
    assert not window.export_button.isEnabled()
    panel.set_stale()
    assert window.busy_indicator.isHidden()
    assert "Calculate the connected measurements" in window.plot.placeholder.text()
