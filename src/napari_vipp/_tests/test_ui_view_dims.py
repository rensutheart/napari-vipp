from __future__ import annotations

from qtpy.QtGui import QColor, QFont, QPalette
from qtpy.QtWidgets import QVBoxLayout, QWidget

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


def test_large_font_narrow_view_dimensions_keep_all_axes_in_the_full_menu(qtbot):
    parent = QWidget()
    qtbot.addWidget(parent)
    parent.setFont(QFont("Arial", 16))
    layout = QVBoxLayout(parent)
    bar = ViewDimsBar()
    bar.setMaximumWidth(360)
    layout.addWidget(bar)
    axes = (
        ViewDimAxis("time", "T", step_axis=0, size=12, value=3),
        ViewDimAxis("depth", "Z", step_axis=1, size=1024, value=512),
        ViewDimAxis("channel", "C", step_axis=2, size=4, value=0),
    )
    emitted = []
    bar.value_changed.connect(lambda *value: emitted.append(value))
    bar.set_axes(axes)
    parent.resize(950, 250)
    parent.show()

    for points in (16, 9, 16):
        parent.setFont(QFont("Arial", points))
        # Repeated presentation refreshes must not re-show any inline control
        # while the complete menu fallback is still the selected mode.
        bar.set_axes(axes)
        bar.set_axes(axes)
        qtbot.waitUntil(lambda: bar._responsive_mode == "menu")
        assert all(control.isHidden() for control in bar._controls)
        assert bar.title_label.isHidden()
        assert not bar.menu_button.isHidden()
        assert bar.menu_button.width() >= bar.menu_button.minimumSizeHint().width()

    bar._populate_menu()
    controls = bar.menu.actions()[0].defaultWidget().findChildren(ViewDimAxisControl)
    assert len(controls) == 3
    assert [control.spin.value() for control in controls] == [3, 512, 0]
    assert [control.spin.maximum() for control in controls] == [11, 1023, 3]
    assert [control.range_label.text() for control in controls] == [
        "/11",
        "/1023",
        "/3",
    ]
    assert all(not control.slider.isHidden() for control in controls)
    assert all(
        control.spin.minimumWidth()
        >= control.spin.fontMetrics().horizontalAdvance(str(control.spin.maximum()))
        + 24
        for control in controls
    )
    assert emitted == []


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

    light = _theme_palette(base="#ffffff", alternate="#f2f4f7", text="#111827")
    control.setPalette(light)
    bar.setPalette(light)

    assert "#111827" in control.label.styleSheet()
    assert "#f2f4f7" in bar.styleSheet()
    assert "#111827" in bar.title_label.styleSheet()

    dark = _theme_palette(base="#111827", alternate="#1f2937", text="#f8fafc")
    control.setPalette(dark)
    bar.setPalette(dark)

    assert "#f8fafc" in control.label.styleSheet()
    assert "#1f2937" in bar.styleSheet()
    assert "#f8fafc" in bar.title_label.styleSheet()
