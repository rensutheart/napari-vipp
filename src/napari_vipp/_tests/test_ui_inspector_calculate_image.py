from __future__ import annotations

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLabel

from napari_vipp._tests.test_ui_inspector_widget_integration import (
    _select,
    _widget,
)
from napari_vipp.core.pipeline import NODE_EXECUTION_BYPASS


def _equation_preview(widget) -> QLabel:
    """Return the one visible Calculate New Image equation preview."""

    preview = widget.parameter_form_widget.findChild(
        QLabel,
        "ImageCalculatorEquationPreview",
    )
    assert preview is not None
    assert not preview.isHidden()
    return preview


def test_calculate_image_equation_shows_effective_weights_offset_and_sources(qtbot):
    data = np.zeros((8, 10), dtype=np.uint16)
    widget = _widget(qtbot, data, axes="YX")
    blur = widget.add_node_from_palette("gaussian_blur")
    calculator = widget.add_node_from_palette("calculate_weighted_image")
    widget._connect_nodes("input", blur.id)
    widget._connect_nodes("input", calculator.id, target_port=0)
    widget._connect_nodes(blur.id, calculator.id, target_port=1)
    widget.pipeline.set_param(calculator.id, "weights", "0.65,-0.35")
    widget.pipeline.set_param(calculator.id, "offset", 12.5)

    _select(widget, calculator.id)

    preview = _equation_preview(widget)
    assert preview.text().splitlines() == [
        "Calculation",
        "Output = 0.65 × I₁ − 0.35 × I₂ + 12.5",
        "Inputs: I₁ = Image Source · out; I₂ = Gaussian Blur · out",
    ]
    assert preview.textFormat() == Qt.PlainText


def test_calculate_image_equation_updates_immediately_with_parameter_controls(qtbot):
    data = np.zeros((2, 8, 10), dtype=np.uint16)
    widget = _widget(qtbot, data, axes="CYX")
    calculator = widget.add_node_from_palette("calculate_weighted_image")
    _select(widget, calculator.id)

    preview = _equation_preview(widget)
    preview_identity = id(preview)
    widget._parameter_widgets["weights"].edit.setText("0.2,0.8")
    widget._parameter_widgets["offset"].value_box.setValue(-7.0)
    widget._debounce_timer.stop()

    qtbot.waitUntil(
        lambda: preview.text().splitlines()[1]
        == "Output = 0.2 × I₁ + 0.8 × I₂ − 7"
    )
    assert id(_equation_preview(widget)) == preview_identity
    assert widget.pipeline.nodes[calculator.id].params["weights"] == "0.2,0.8"
    assert widget.pipeline.nodes[calculator.id].params["offset"] == -7.0


def test_calculate_image_equation_tracks_input_count_connections_and_errors(qtbot):
    data = np.zeros((8, 10), dtype=np.uint16)
    widget = _widget(qtbot, data, axes="YX")
    blur = widget.add_node_from_palette("gaussian_blur")
    calculator = widget.add_node_from_palette("calculate_weighted_image")
    _select(widget, calculator.id)
    preview = _equation_preview(widget)
    preview_identity = id(preview)

    widget._parameter_widgets["input_count"].value_box.setValue(3)
    widget._debounce_timer.stop()

    assert preview.text().splitlines()[0] == "Calculation"
    assert "Equation unavailable" in preview.text()
    assert "3" in preview.text()
    assert "Output =" not in preview.text()
    assert id(_equation_preview(widget)) == preview_identity

    widget._parameter_widgets["weights"].edit.setText("1,2,3")
    widget._connect_nodes("input", calculator.id, target_port=0)
    widget._connect_nodes(blur.id, calculator.id, target_port=1)
    widget._connect_nodes("input", calculator.id, target_port=2)
    widget._debounce_timer.stop()

    assert "Output = 1 × I₁ + 2 × I₂ + 3 × I₃ + 0" in (
        preview.text()
    )
    assert "I₁ = Image Source · out" in preview.text()
    assert "I₂ = Gaussian Blur · out" in preview.text()
    assert "I₃ = Image Source · out" in preview.text()
    assert id(_equation_preview(widget)) == preview_identity

    widget._parameter_widgets["weights"].edit.setText("1,bad,3")
    widget._debounce_timer.stop()

    assert "Equation unavailable" in preview.text()
    assert "Output =" not in preview.text()
    assert id(_equation_preview(widget)) == preview_identity

    widget._parameter_widgets["weights"].edit.setText("1,2,3,4")
    widget._debounce_timer.stop()

    assert "Equation unavailable" in preview.text()
    assert "4 were provided" in preview.text()
    assert "Output =" not in preview.text()
    assert id(_equation_preview(widget)) == preview_identity


def test_calculate_image_equation_maps_split_channel_output_symbols(qtbot):
    data = np.zeros((2, 8, 10), dtype=np.uint16)
    widget = _widget(qtbot, data, axes="CYX")
    split = widget.add_node_from_palette("split_channels")
    calculator = widget.add_node_from_palette("calculate_weighted_image")
    widget._connect_nodes("input", split.id)
    widget._connect_nodes(split.id, calculator.id, target_port=0, source_port=0)
    widget._connect_nodes(split.id, calculator.id, target_port=1, source_port=1)

    _select(widget, calculator.id)

    preview = _equation_preview(widget)
    assert "Inputs: I₁ = Split Channels · Ch 1; " in preview.text()
    assert "I₂ = Split Channels · Ch 2" in preview.text()


def test_calculate_image_equation_describes_bypass_without_claiming_stored_math(
    qtbot,
):
    widget = _widget(qtbot, np.zeros((8, 10), dtype=np.uint16), axes="YX")
    calculator = widget.add_node_from_palette("calculate_weighted_image")
    widget.pipeline.restore_node_execution_mode(
        calculator.id,
        NODE_EXECUTION_BYPASS,
    )

    _select(widget, calculator.id)

    preview = _equation_preview(widget)
    assert preview.text().splitlines() == [
        "Calculation",
        "Bypassed — Output = I₁; stored calculation is inactive.",
        "Inputs: I₁ = not connected",
    ]


def test_calculate_image_equation_bounds_malformed_restored_input_count(qtbot):
    widget = _widget(qtbot, np.zeros((8, 10), dtype=np.uint16), axes="YX")
    calculator = widget.add_node_from_palette("calculate_weighted_image")
    calculator.params["input_count"] = 10**9

    _select(widget, calculator.id)

    preview = _equation_preview(widget)
    assert "Equation unavailable" in preview.text()
    assert "input count must be between 1 and 12" in preview.text()
    assert preview.text().count("not connected") == 12
