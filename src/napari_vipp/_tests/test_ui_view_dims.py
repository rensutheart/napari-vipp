from __future__ import annotations

from napari_vipp import _widget
from napari_vipp.ui.view_dims import (
    ViewDimAxis,
    ViewDimAxisControl,
    ViewDimsBar,
)


def test_widget_module_reexports_extracted_view_dimension_controls():
    assert _widget.ViewDimAxis is ViewDimAxis
    assert _widget.ViewDimAxisControl is ViewDimAxisControl
    assert _widget.ViewDimsBar is ViewDimsBar


def test_view_dimension_control_synchronizes_values_and_emits_semantic_axis(qtbot):
    control = ViewDimAxisControl()
    qtbot.addWidget(control)
    captured: list[tuple[int, int]] = []
    control.value_changed.connect(
        lambda step_axis, value: captured.append((step_axis, value))
    )

    control.set_axis(ViewDimAxis("z", "Z", step_axis=2, size=5, value=9))

    assert captured == []
    assert control.slider.value() == 4
    assert control.spin.value() == 4
    assert control.range_label.text() == "/4"

    control.spin.setValue(1)

    assert control.slider.value() == 1
    assert captured == [(2, 1)]


def test_view_dimensions_menu_builds_full_controls_and_forwards_values(qtbot):
    bar = ViewDimsBar()
    qtbot.addWidget(bar)

    bar._populate_menu()
    empty_action = bar.menu.actions()[0]
    assert empty_action.text() == "No view dimensions"
    assert not empty_action.isEnabled()

    bar.set_axes((ViewDimAxis("time", "T", step_axis=0, size=3, value=1),))
    captured: list[tuple[int, int]] = []
    bar.value_changed.connect(
        lambda step_axis, value: captured.append((step_axis, value))
    )
    bar._populate_menu()

    action = bar.menu.actions()[0]
    container = action.defaultWidget()
    controls = container.findChildren(ViewDimAxisControl)
    assert len(controls) == 1
    assert not controls[0].slider.isHidden()

    controls[0].spin.setValue(2)

    assert captured == [(0, 2)]
