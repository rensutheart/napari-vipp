"""Transient plot progress must not move nearby controls or resize the canvas."""

import pytest
from qtpy.QtCore import QPoint
from qtpy.QtWidgets import QApplication

from napari_vipp.core.result_plots import build_plot_result
from napari_vipp.core.tables import TableData
from napari_vipp.ui.result_plots import PlotResultsPanel


def _result(with_warning):
    table = TableData(
        ("area", "minor_axis_length"),
        tuple((100 + index * 2, 10.123456 + index) for index in range(18)),
    )
    return build_plot_result(
        table,
        y_column="area",
        group_column="minor_axis_length" if with_warning else "",
    )


def _settle(qtbot):
    # Process queued layout requests and matplotlib's deferred first draw.
    for _ in range(3):
        QApplication.processEvents()
    qtbot.wait(1)


def _rect(widget, owner):
    point = widget.mapTo(owner, QPoint(0, 0))
    return point.x(), point.y(), widget.width(), widget.height()


def _geometry(panel, window):
    return {
        "inspector_plot": _rect(panel.plot, panel),
        "inspector_controls": _rect(panel.controls, panel),
        "inspector_footer": _rect(panel.open_button, panel),
        "inspector_progress_slot": _rect(panel.busy_indicator, panel),
        "dialog_plot": _rect(window.plot, window),
        "dialog_canvas": _rect(window.plot.canvas, window),
        "dialog_controls": _rect(window.controls, window),
        "dialog_footer": _rect(window.export_button, window),
        "dialog_progress_slot": _rect(window.busy_indicator, window),
    }


@pytest.mark.parametrize("inspector_width", [280, 440])
@pytest.mark.parametrize("dialog_width", [780, 1120])
@pytest.mark.parametrize("with_warning", [False, True])
def test_busy_transitions_preserve_plot_and_controls_geometry(
    qtbot, inspector_width, dialog_width, with_warning
):
    result = _result(with_warning)
    panel = PlotResultsPanel(result=result)
    qtbot.addWidget(panel)
    panel.setFixedWidth(inspector_width)
    panel.resize(inspector_width, 1300)
    panel.show()
    window = panel.open_plot()
    window.resize(dialog_width, 800)
    _settle(qtbot)
    assert panel.warning.isVisible() is with_warning
    assert window.warning.isVisible() is with_warning
    assert not panel.busy_indicator.isVisible()
    assert not window.busy_indicator.isVisible()
    before = _geometry(panel, window)

    for message in (
        "Plot update queued…",
        "Preparing plot measurements…",
        "Updating plot…",
    ):
        panel.set_state(result=result, busy=True, busy_message=message)
        _settle(qtbot)
        assert panel.busy_indicator.isVisible()
        assert window.busy_indicator.isVisible()
        assert not window.export_button.isEnabled()
        assert not window.point_label.isVisible()
        assert not window.warning.isVisible()
        assert _geometry(panel, window) == before, message

    panel.set_result(result)
    _settle(qtbot)
    assert not panel.busy_indicator.isVisible()
    assert not window.busy_indicator.isVisible()
    assert window.export_button.isEnabled()
    assert window.point_label.isVisible()
    assert window.warning.isVisible() is with_warning
    assert _geometry(panel, window) == before


def test_progress_slot_is_reserved_before_any_measurements_arrive(qtbot):
    panel = PlotResultsPanel()
    qtbot.addWidget(panel)
    panel.setFixedWidth(280)
    panel.resize(280, 1200)
    panel.show()
    _settle(qtbot)
    before = _rect(panel.plot, panel)
    slot = _rect(panel.busy_indicator, panel)
    assert slot[3] > 0
    for message in ("Plot update queued…", "Preparing plot measurements…"):
        panel.set_state(busy=True, busy_message=message)
        _settle(qtbot)
        assert _rect(panel.plot, panel) == before
        assert _rect(panel.busy_indicator, panel) == slot
    panel.set_state()
    _settle(qtbot)
    assert not panel.busy_indicator.isVisible()
    assert _rect(panel.plot, panel) == before
    assert _rect(panel.busy_indicator, panel) == slot


@pytest.mark.parametrize("dialog_width", [780, 1120])
def test_busy_preserves_open_data_view_space_and_restores_current_data(
    qtbot, dialog_width
):
    result = _result(with_warning=True)
    panel = PlotResultsPanel(result=result)
    qtbot.addWidget(panel)
    panel.setFixedWidth(280)
    panel.resize(280, 1300)
    panel.show()
    window = panel.open_plot()
    window.resize(dialog_width, 950)
    window.data_button.setChecked(True)
    _settle(qtbot)
    assert window.data_view.isVisible()
    before = _geometry(panel, window)
    data_rect = _rect(window.data_view, window)
    old_model = window.data_view.model()

    for message in ("Plot update queued…", "Updating plot…"):
        panel.set_state(result=result, busy=True, busy_message=message)
        _settle(qtbot)
        assert not window.data_view.isVisible()
        assert window.data_view.model() is None
        assert not window.data_button.isEnabled()
        assert _rect(window.data_view, window) == data_rect
        assert _geometry(panel, window) == before

    panel.set_result(result)
    _settle(qtbot)
    assert window.data_view.isVisible()
    assert window.data_button.isChecked()
    assert window.data_view.model() is not old_model
    assert window.data_view.model().rowCount() == len(result.plotted_table.rows)
    assert _rect(window.data_view, window) == data_rect
    assert _geometry(panel, window) == before
