"""Previous-frame controls expose scientific policy without authoring on render."""

import numpy as np
import pytest

from napari_vipp._tests.test_registration_previous_frame import translation_series
from napari_vipp.core.pipeline import EXECUTION_READY
from napari_vipp.core.registration import estimate_registration
from napari_vipp.ui.node_labels import build_node_presentations


@pytest.fixture
def widget_node(qtbot):
    from napari_vipp._tests.test_widget import _Viewer
    from napari_vipp._widget import VippWidget

    widget = VippWidget(_Viewer(np.zeros((16, 18), np.float32)), defer_initial_run=True)
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette("estimate_registration")
    widget._debounce_timer.stop()
    return widget, node


def change(widget, name, value):
    widget._on_param_changed(name, value)
    widget._debounce_timer.stop()


def test_previous_frame_controls_follow_mode_and_preserve_dormant_policy(widget_node):
    widget, node = widget_node
    assert "time_strategy" not in widget._parameter_widgets
    assert "cumulative_quality_policy" not in widget._parameter_widgets
    change(widget, "mode", "Time series")
    assert "time_strategy" in widget._parameter_widgets
    assert "cumulative_quality_policy" not in widget._parameter_widgets
    change(widget, "time_strategy", "Previous frame")
    assert "cumulative_quality_policy" in widget._parameter_widgets
    assert "original adjacent volumes" in widget._operation_help_note(node.id)
    assert "reported only" in widget._operation_help_note(node.id)
    assert widget._operation_help_note_status(node.id) == "Warning"
    assert (
        widget._parameter_widgets["operation_notice"].property("vippTextTone")
        == "warning"
    )
    change(widget, "cumulative_quality_policy", "Require local limits")
    assert "also gate" in widget._operation_help_note(node.id)
    change(widget, "mode", "Two images")
    assert "time_strategy" not in widget._parameter_widgets
    assert "cumulative_quality_policy" not in widget._parameter_widgets
    assert "Previous frame" not in widget._operation_help_note(node.id)
    assert (
        "Previous frame"
        not in build_node_presentations(widget.pipeline, {})[node.id].summary
    )
    assert node.params["cumulative_quality_policy"] == "Require local limits"
    change(widget, "mode", "Time series")
    assert "cumulative_quality_policy" in widget._parameter_widgets
    assert (
        "Previous frame"
        in build_node_presentations(widget.pipeline, {})[node.id].summary
    )


def test_render_legacy_fixed_reference_does_not_add_new_scientific_parameters(
    widget_node,
):
    widget, node = widget_node
    widget.pipeline.set_param(node.id, "mode", "Time series")
    node.params.pop("time_strategy", None)
    node.params.pop("cumulative_quality_policy", None)
    before = dict(node.params)
    widget._select_node(node.id)
    assert node.params == before
    assert "time_strategy" in widget._parameter_widgets
    assert widget._parameter_widgets["time_strategy"].value() == "Fixed reference"
    assert "cumulative_quality_policy" not in widget._parameter_widgets
    assert widget._operation_help_note_status(node.id) == "Info"


def test_result_summary_reads_completed_strategy_not_current_parameters(widget_node):
    widget, node = widget_node
    data, carried, _ = translation_series()
    transform, diagnostics = estimate_registration(
        data,
        moving_state=carried,
        mode="Time series",
        reference_time=2,
        time_strategy="Previous frame",
    )
    widget.pipeline.outputs[node.id] = transform
    widget.pipeline.output_states[node.id] = transform.state
    widget.pipeline.node_outputs[node.id] = [transform, diagnostics]
    widget.pipeline.node_output_states[node.id] = [transform.state, None]
    widget.pipeline.node_execution_states[node.id] = EXECUTION_READY
    widget._select_node(node.id)
    text = widget._registration_results.summary.text()
    assert "Reference time: 2" in text
    assert "Time-series strategy: Previous frame" in text
    assert "Cumulative quality: Report only" in text
    assert "errors can accumulate" in text
    assert widget._registration_results.export_transform.isEnabled()


@pytest.mark.parametrize("theme", ("dark", "light"))
def test_previous_frame_guidance_fits_narrow_inspector(
    widget_node, qtbot, monkeypatch, theme
):
    import os
    from pathlib import Path

    from napari.qt import get_stylesheet
    from qtpy.QtGui import QTextDocument
    from qtpy.QtWidgets import QMainWindow

    widget, _node = widget_node
    monkeypatch.setattr(widget, "_confirm_close_dirty_workflow_tabs", lambda: True)
    host = QMainWindow()
    host.setCentralWidget(widget)
    qtbot.addWidget(host)
    host.resize(1240, 1000)
    host.setStyleSheet(get_stylesheet(theme, extra_variables={"font_size": "10pt"}))
    host.show()
    qtbot.waitUntil(lambda: widget.property("vippColorScheme") == theme)
    change(widget, "mode", "Time series")
    change(widget, "time_strategy", "Previous frame")
    widget.splitter.setSizes([210, 650, 380])
    qtbot.wait(200)
    assert widget.inspector_panel.width() <= 400
    assert not widget.inspector_panel.horizontalScrollBar().isVisible()
    note = widget._parameter_widgets["operation_notice"]
    document = QTextDocument()
    document.setDocumentMargin(0)
    document.setDefaultFont(note.font())
    document.setHtml(note.text())
    document.setTextWidth(note.contentsRect().width())
    assert note.height() >= document.size().height()
    assert note.geometry().bottom() < widget.parameter_form_widget.height()
    child = note
    while child.parentWidget() is not widget.inspector_content:
        parent = child.parentWidget()
        assert child.geometry().bottom() < parent.height()
        child = parent
    screenshot_dir = os.environ.get("VIPP_REGISTRATION_UI_SCREENSHOTS")
    if screenshot_dir:
        target = Path(screenshot_dir)
        target.mkdir(parents=True, exist_ok=True)
        assert widget.inspector_content.grab().save(
            str(target / f"previous-frame-{theme}.png")
        )
