from __future__ import annotations

from qtpy.QtGui import QColor, QPalette
from qtpy.QtWidgets import QWidget

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


def test_view_dimensions_use_menu_before_compact_controls_overlap(qtbot):
    parent = QWidget()
    parent.resize(652, 80)
    bar = ViewDimsBar(parent)
    qtbot.addWidget(bar)
    bar.resize(640, 40)
    bar.set_axes(
        tuple(
            ViewDimAxis(name, label, step_axis=index, size=size, value=0)
            for index, (name, label, size) in enumerate(
                (
                    ("time", "T", 120),
                    ("channel", "C", 81),
                    ("depth", "Z", 12),
                    ("position", "P", 5),
                )
            )
        )
    )

    assert bar._responsive_mode == "menu"
    assert not bar.menu_button.isHidden()
    assert all(control.isHidden() for control in bar._controls)


def _theme_palette(*, base: str, alternate: str, text: str) -> QPalette:
    palette = QPalette()
    palette.setColor(QPalette.Base, QColor(base))
    palette.setColor(QPalette.AlternateBase, QColor(alternate))
    palette.setColor(QPalette.Text, QColor(text))
    return palette


def test_view_dimension_controls_follow_runtime_palette_changes(qtbot):
    control = ViewDimAxisControl()
    bar = ViewDimsBar()
    qtbot.addWidget(control)
    qtbot.addWidget(bar)

    light = _theme_palette(
        base="#ffffff", alternate="#f2f4f7", text="#111827"
    )
    control.setPalette(light)
    bar.setPalette(light)

    assert "#111827" in control.label.styleSheet()
    assert "#f2f4f7" in bar.styleSheet()
    assert "#111827" in bar.title_label.styleSheet()

    dark = _theme_palette(
        base="#111827", alternate="#1f2937", text="#f8fafc"
    )
    control.setPalette(dark)
    bar.setPalette(dark)

    assert "#f8fafc" in control.label.styleSheet()
    assert "#1f2937" in bar.styleSheet()
    assert "#f8fafc" in bar.title_label.styleSheet()
