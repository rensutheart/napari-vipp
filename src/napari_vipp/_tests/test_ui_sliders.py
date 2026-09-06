from __future__ import annotations

from qtpy.QtGui import QColor, QPalette
from qtpy.QtWidgets import (
    QStyle,
    QStyleOptionSlider,
    QVBoxLayout,
    QWidget,
)

from napari_vipp.core.pipeline import ParameterSpec
from napari_vipp.ui.axis_controls import AxisSelectionRow, AxisSliceOption
from napari_vipp.ui.batch_navigator import BatchNavigator
from napari_vipp.ui.controls import ParameterBounds, ParameterControl
from napari_vipp.ui.sliders import VippSlider, slider_colors
from napari_vipp.ui.view_dims import ViewDimAxis, ViewDimAxisControl


def _palette(*, base: str, text: str) -> QPalette:
    palette = QPalette()
    palette.setColor(QPalette.Base, QColor(base))
    palette.setColor(QPalette.Text, QColor(text))
    return palette


def _assert_handle_is_contained(slider: VippSlider) -> None:
    slider.resize(240, max(slider.minimumHeight(), slider.sizeHint().height()))
    midpoint = (slider.minimum() + slider.maximum()) // 2
    for value in (slider.minimum(), midpoint, slider.maximum()):
        slider.setValue(value)
        option = QStyleOptionSlider()
        slider.initStyleOption(option)
        handle_rect = slider.style().subControlRect(
            QStyle.CC_Slider,
            option,
            QStyle.SC_SliderHandle,
            slider,
        )
        assert slider.contentsRect().contains(handle_rect)


def test_slider_colors_match_dark_mockup_and_adapt_to_light_theme():
    dark = slider_colors(_palette(base="#111827", text="#f8fafc"))
    light = slider_colors(_palette(base="#ffffff", text="#111827"))

    assert dark.fill == QColor("#4560c4")
    assert dark.handle == QColor("#4560c4")
    assert light.fill == QColor("#a0b8ff")
    assert light.handle == QColor("#a0b8ff")
    assert dark.groove != light.groove
    assert dark.disabled != light.disabled


def test_vipp_slider_refreshes_its_track_and_handle_for_runtime_theme(qtbot):
    host = QWidget()
    layout = QVBoxLayout(host)
    slider = VippSlider()
    layout.addWidget(slider)
    qtbot.addWidget(host)

    host.setPalette(_palette(base="#111827", text="#f8fafc"))
    host.show()
    qtbot.waitExposed(host)
    dark_style = slider.styleSheet()

    assert "background: #4560c4" in dark_style
    assert "height: 6px" in dark_style
    assert "width: 12px" in dark_style
    assert "margin: -3px 0" in dark_style
    assert slider.minimumHeight() >= 14
    _assert_handle_is_contained(slider)

    host.setPalette(_palette(base="#ffffff", text="#111827"))
    light_style = slider.styleSheet()

    assert "background: #a0b8ff" in light_style
    assert light_style != dark_style


def test_shared_slider_class_covers_inspector_axes_dims_and_batch(qtbot):
    spec = ParameterSpec(
        name="amount",
        label="Amount",
        kind="int",
        default=5,
        minimum=0,
        maximum=10,
        step=1,
    )
    parameter = ParameterControl(
        spec,
        spec.default,
        ParameterBounds(0, 10, 1, 0),
    )
    axis_row = AxisSelectionRow(
        AxisSliceOption(0, "z", "space", 8),
        mode="remove",
    )
    dims = ViewDimAxisControl()
    batch = BatchNavigator()
    for control in (parameter, axis_row, dims, batch):
        qtbot.addWidget(control)

    axis_row.range_slider.setPalette(
        _palette(base="#111827", text="#f8fafc")
    )
    dims.set_axis(
        ViewDimAxis("depth", "Z", step_axis=0, size=9, value=4)
    )
    batch.set_session(3, 1, "sample", ["sample.tif"])

    assert isinstance(parameter.slider, VippSlider)
    assert isinstance(axis_row.index_slider, VippSlider)
    assert isinstance(dims.slider, VippSlider)
    assert isinstance(batch.slider, VippSlider)
    for slider in (
        parameter.slider,
        axis_row.index_slider,
        dims.slider,
        batch.slider,
    ):
        _assert_handle_is_contained(slider)
    range_colors = axis_row.range_slider._paint_colors()
    assert range_colors.fill == QColor("#4560c4")
    assert range_colors.handle == QColor("#4560c4")
