"""Slider number fields stay compact and consistent when form labels wrap."""

import pytest
from napari._qt.qt_event_loop import get_qapp
from napari._qt.qt_resources import get_stylesheet
from qtpy.QtGui import QFont, QFontDatabase
from qtpy.QtWidgets import QFormLayout, QStyle, QStyleOptionSpinBox, QWidget

from napari_vipp.core.pipeline import ParameterSpec
from napari_vipp.ui.controls import ParameterBounds, ParameterControl


@pytest.fixture
def styled_app(qapp):
    get_qapp()  # Register napari's actual theme/icon resources as in the app.
    previous_style, previous_font = qapp.styleSheet(), QFont(qapp.font())
    font_id = QFontDatabase.addApplicationFont("C:/Windows/Fonts/segoeui.ttf")
    if font_id < 0:
        # The system/default font still exercises geometry on non-Windows CI.
        family = previous_font.family()
    else:
        family = QFontDatabase.applicationFontFamilies(font_id)[0]
    qapp.setFont(QFont(family, 9))
    qapp.setStyleSheet(get_stylesheet("dark", extra_variables={"font_size": "9pt"}))
    yield qapp, family
    qapp.setFont(previous_font)
    qapp.setStyleSheet(previous_style)


def _form(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    form = QFormLayout(host)
    form.setRowWrapPolicy(QFormLayout.WrapLongRows)
    form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    controls = []
    for name, label, kind, value, high, decimals in (
        ("channel", "Estimation channel (zero-based)", "int", 0, 1000000, 0),
        ("factor", "Translation subpixel factor", "int", 10, 1000, 0),
        ("shift", "Maximum translation (fraction of image)", "float", 0.25, 1, 4),
        ("overlap", "Minimum valid overlap (fraction)", "float", 0.25, 1, 4),
    ):
        spec = ParameterSpec(name, label, kind, value, 0, high, 1)
        control = ParameterControl(spec, value, ParameterBounds(0, high, 1, decimals))
        form.addRow(label, control)
        # The real inspector permits long labels to wrap, not just whole rows.
        form.labelForField(control).setWordWrap(True)
        form.labelForField(control).setMinimumWidth(0)
        controls.append(control)
    return host, controls


@pytest.mark.parametrize("points", [9, 16])
def test_short_numeric_entries_share_fixed_compact_width_through_form_wrap(
    qtbot, styled_app, points
):
    app, family = styled_app
    app.setFont(QFont(family, points))
    app.setStyleSheet(
        get_stylesheet("dark", extra_variables={"font_size": f"{points}pt"})
    )
    host, controls = _form(qtbot)
    host.show()
    qtbot.waitUntil(lambda: controls[0].value_box.isVisible())
    original = [
        (
            c.value(),
            c.value_box.minimum(),
            c.value_box.maximum(),
            c.value_box.singleStep(),
        )
        for c in controls
    ]
    widths = []
    for width in (700, 480, 300, 480):
        host.resize(width, 700)
        qtbot.waitUntil(lambda width=width: host.width() == width)
        boxes = [control.value_box for control in controls]
        assert all(box.minimumWidth() == box.maximumWidth() for box in boxes)
        assert len({box.width() for box in boxes}) == 1
        assert max(box.width() for box in boxes) <= (110 if points == 9 else 155)
        widths.append(boxes[0].width())
        for box in boxes:
            option = QStyleOptionSpinBox()
            box.initStyleOption(option)
            editor = box.style().subControlRect(
                QStyle.CC_SpinBox, option, QStyle.SC_SpinBoxEditField, box
            )
            assert editor.width() >= box.fontMetrics().horizontalAdvance(box.text()) + 4
    assert len(set(widths)) == 1
    assert [
        (
            c.value(),
            c.value_box.minimum(),
            c.value_box.maximum(),
            c.value_box.singleStep(),
        )
        for c in controls
    ] == original


def test_compact_entry_can_grow_for_actual_long_value_without_changing_bounds(
    qtbot, styled_app
):
    spec = ParameterSpec("count", "Exact count", "int", 0, 0, 2147483647, 1)
    control = ParameterControl(spec, 0, ParameterBounds(0, 2147483647, 1, 0))
    qtbot.addWidget(control)
    control.show()
    qtbot.waitUntil(lambda: control.value_box.isVisible())
    box = control.value_box
    before = box.width()
    values = []
    control.valueChanged.connect(values.append)
    box.setValue(2147483647)
    qtbot.waitUntil(lambda: box.width() > before)
    option = QStyleOptionSpinBox()
    box.initStyleOption(option)
    editor = box.style().subControlRect(
        QStyle.CC_SpinBox, option, QStyle.SC_SpinBoxEditField, box
    )
    assert editor.width() >= box.fontMetrics().horizontalAdvance(box.text()) + 4
    assert (box.minimum(), box.maximum(), box.singleStep(), control.value()) == (
        0,
        2147483647,
        1,
        2147483647,
    )
    assert values == [2147483647]
