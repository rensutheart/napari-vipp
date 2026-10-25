"""Shared presentation roles preserve inspector values and interaction safety."""

import re
from copy import deepcopy

import numpy as np
import pytest
from qtpy.QtGui import QColor, QFont, QPalette
from qtpy.QtWidgets import QWidget

from napari_vipp._graph import NodeCard
from napari_vipp._tests.test_ui_inspector_widget_integration import (
    _publish_array_output,
    _widget,
)
from napari_vipp._theme import graph_theme
from napari_vipp.core.pipeline import ParameterSpec
from napari_vipp.ui.axis_controls import AxisSelectionRow, AxisSliceOption
from napari_vipp.ui.connected_inputs import ConnectedInputBinding, ConnectedInputsCard
from napari_vipp.ui.controls import (
    ImageSourceControl,
    NumericEntryControl,
    ParameterBounds,
    ParameterControl,
)
from napari_vipp.ui.design import content_width
from napari_vipp.ui.inspector import InspectorSection
from napari_vipp.ui.status import MessageSeverity


def test_inspector_type_roles_scale_with_host_without_refresh_ratcheting(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    font = QFont(host.font())
    font.setPointSizeF(16)
    host.setFont(font)
    section = InspectorSection("A scientific section", parent=host)
    section.setSummary("Explicit ZYX · 0.25 µm")
    section.refresh_theme()
    first = section.styleSheet()
    for _ in range(5):
        section.refresh_theme()
    assert section.styleSheet() == first
    assert "font-size: 16pt" in first
    assert "font-size: 16.5pt" in first
    assert not re.search(r"font-size:\s*\d+(?:\.\d+)?px", first)
    assert section.title_button.minimumWidth() == 0
    assert section.summary_label.minimumWidth() == 0
    assert section.summary_label.toolTip() == "Explicit ZYX · 0.25 µm"


@pytest.mark.parametrize("control_type", [ParameterControl, NumericEntryControl])
def test_numeric_control_metrics_grow_but_value_bounds_and_precision_do_not(
    qtbot, control_type
):
    spec = ParameterSpec("count", "Exact count", "int", 123456789, 0, 2147483647, 1)
    control = control_type(spec, 123456789, ParameterBounds(0, 2147483647, 1, 0))
    qtbot.addWidget(control)
    box = control.value_box
    original = control.value(), box.minimum(), box.maximum(), box.singleStep()
    width = box.minimumWidth()
    font = QFont(control.font())
    font.setPointSizeF(max(font.pointSizeF(), 9) + 8)
    control.setFont(font)
    qtbot.waitUntil(lambda: box.minimumWidth() > width)
    assert box.minimumWidth() >= content_width(box, box.text(), padding=24)
    if control_type is ParameterControl:
        # Slider entries fit the current value, not every possible range value;
        # their fixed compact width must not stretch when the form wraps.
        assert box.minimumWidth() == box.maximumWidth()
    else:
        assert box.minimumWidth() >= box.sizeHint().width()
    assert (control.value(), box.minimum(), box.maximum(), box.singleStep()) == original


def test_axis_entries_and_mode_panels_are_not_fixed_to_58_pixels(qtbot):
    row = AxisSelectionRow(AxisSliceOption(0, "z", "space", 123456789))
    qtbot.addWidget(row)
    for box in (row.start_box, row.end_box, row.index_box):
        assert box.minimumWidth() >= box.sizeHint().width()
        assert box.maximumWidth() > box.minimumWidth()
        assert box.maximum() == 123456788
    assert row.range_panel.maximumHeight() > row.range_panel.minimumHeight()
    assert row.remove_panel.maximumHeight() > row.remove_panel.minimumHeight()
    assert row.range_panel.minimumHeight() == row.remove_panel.minimumHeight()


def test_source_warning_surface_is_scoped_and_actions_have_no_pixel_caps(qtbot):
    control = ImageSourceControl(None, layer_names=[], sample_names=[])
    qtbot.addWidget(control)
    assert "QWidget#VippSourceMemoryRepair" in control.memory_repair_panel.styleSheet()
    assert "QWidget {" not in control.memory_repair_panel.styleSheet()
    assert control.path_button.maximumWidth() > 64
    assert control.zarr_button.maximumWidth() > 64
    assert control.source_load_status.cancel_button.maximumWidth() > 104
    assert control.memory_repair_dismiss_button.maximumWidth() > 70


def test_every_graph_card_action_has_the_same_scalable_focus_and_disabled_rules(qtbot):
    card = NodeCard("synthetic", "Synthetic analysis", "Measurements", False)
    qtbot.addWidget(card)
    styles = {
        button.styleSheet()
        for button in (
            card.calculate_button,
            card.pin_button,
            card.plot_button,
            card.result_button,
            card.trajectory_review_button,
        )
    }
    assert len(styles) == 1
    style = styles.pop()
    assert "QPushButton:focus" in style and "QPushButton:disabled" in style
    assert "pt;" in style and "font-size: 9px" not in style
    for label in (
        card.category_label,
        card.subtitle_label,
        card.metadata_label,
        card.execution_label,
    ):
        assert "pt;" in label.styleSheet()


@pytest.mark.parametrize(
    "dark,surface,text",
    [(True, "#334155", "#f8fafc"), (False, "#e2e8f0", "#172033")],
)
def test_graph_action_colors_use_the_selected_theme_not_stale_native_roles(
    qtbot, dark, surface, text
):
    card = NodeCard("synthetic", "Synthetic analysis", "Measurements", False)
    qtbot.addWidget(card)
    selected = QPalette()
    selected.setColor(QPalette.Base, QColor("#111827" if dark else "#ffffff"))
    selected.setColor(QPalette.Window, QColor("#111827" if dark else "#ffffff"))
    theme = graph_theme(selected)
    stale = QPalette(card.palette())
    stale.setColor(QPalette.Base, QColor("#000000"))
    stale.setColor(QPalette.Text, QColor("#000000"))
    card.setPalette(stale)

    card._refresh_button_styles(theme)

    for button in (
        card.calculate_button,
        card.pin_button,
        card.plot_button,
        card.result_button,
        card.trajectory_review_button,
    ):
        assert f"background-color: {surface}" in button.styleSheet()
        assert f"color: {text}" in button.styleSheet()
    # Resolution is local presentation: no palette/global-state mutation.
    assert card.palette().color(QPalette.Base) == QColor("#000000")
    assert card.palette().color(QPalette.Text) == QColor("#000000")


def test_connected_input_card_keeps_scientific_context_at_body_size(qtbot):
    card = ConnectedInputsCard()
    qtbot.addWidget(card)
    summary = "Explicit ZYX · 0.25 µm XY · 1.0 µm Z"
    card.set_bindings(
        [
            ConnectedInputBinding(
                "Image", "image", "Synthetic source", scientific_summary=summary
            )
        ]
    )
    row = card.rows[0]
    assert row.scientific_label.text() == summary
    assert row.scientific_label.wordWrap()
    assert row.scientific_label.textInteractionFlags()
    assert "font-size: 11px" not in card.styleSheet()
    assert "border-left: 3px" not in card.styleSheet()


@pytest.mark.parametrize("operation_id", ["input", "gaussian_blur"])
def test_output_file_actions_use_export_without_mutating_authored_parameters(
    qtbot, operation_id
):
    widget = _widget(qtbot)
    node = next(
        node
        for node in widget.pipeline.nodes.values()
        if node.operation_id == operation_id
    )
    before = deepcopy(node.params)
    _publish_array_output(widget, node.id, np.zeros((8, 8), dtype=np.uint8), axes="YX")
    widget._selected_node_id = node.id
    widget._sync_output_actions_ui()
    assert widget.save_button.text().startswith("Export ")
    assert widget.save_button.text().endswith("…")
    assert widget.save_button.isEnabled()
    assert node.params == before


def test_important_status_exposes_details_without_erasing_node_failure(qtbot):
    widget = _widget(qtbot)
    widget.pipeline.set_node_execution_error("input", "Scientific failure retained")
    state = widget.pipeline.node_execution_states["input"]
    widget._set_status(
        "Original connection restored.",
        severity=MessageSeverity.ERROR,
        actionable=True,
        detail="Synthetic traceback",
    )
    assert widget.status_label.severity is MessageSeverity.ERROR
    assert widget.status_label.actionable
    assert widget.status_label.toolTip() == "Synthetic traceback"
    widget.status_actions.dismiss_button.click()
    assert (
        widget.pipeline.node_execution_messages["input"]
        == "Scientific failure retained"
    )
    assert widget.pipeline.node_execution_states["input"] == state
    assert not widget.status_label.text()
