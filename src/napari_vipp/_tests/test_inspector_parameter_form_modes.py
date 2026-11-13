"""Editable inspector rows move together rather than wrapping independently."""

import pytest
from qtpy.QtCore import QPoint
from qtpy.QtWidgets import QApplication, QFormLayout, QLabel, QLineEdit, QWidget

from napari_vipp._tests.test_ui_inspector_widget_integration import _widget
from napari_vipp._widget import VippWidget, _InspectorParameterFormLayout
from napari_vipp.ui.controls import ParameterControl


def _visible_rows(form):
    for row in range(form.rowCount()):
        label_item = form.itemAt(row, QFormLayout.LabelRole)
        field_item = form.itemAt(row, QFormLayout.FieldRole)
        if label_item is None or field_item is None:
            continue
        label, field = label_item.widget(), field_item.widget()
        if isinstance(label, QLabel) and field is not None and field.isVisible():
            yield label, field


def _stacked_rows(form):
    return {
        label.geometry().bottom() < field.geometry().top()
        for label, field in _visible_rows(form)
        if label.isVisible()
    }


@pytest.mark.parametrize(
    "operation",
    ["estimate_registration", "richardson_lucy_tv_deconvolution"],
)
def test_inspector_resizes_visible_parameter_rows_as_one_group(qtbot, operation):
    widget = _widget(qtbot)
    node = widget.add_node_from_palette(operation)
    widget.graph_view.select_node(node.id)
    widget.parameter_group.setExpanded(True)
    widget._permit_incomplete_startup_discard()
    widget.resize(1740, 850)
    widget.show()
    qtbot.waitExposed(widget)
    form = widget.parameter_form
    original_params = dict(node.params)
    original_controls = dict(widget._parameter_widgets)

    widths = [340, 430, 510, 620, 760, 1050]
    modes = []
    for width in widths + widths[::-1]:
        widget.inspector_panel.setFixedWidth(width)
        QApplication.processEvents()
        assert form.rowWrapPolicy() in (
            QFormLayout.WrapAllRows,
            QFormLayout.DontWrapRows,
        )
        stacked = form.rowWrapPolicy() == QFormLayout.WrapAllRows
        assert _stacked_rows(form) == {stacked}
        modes.append(stacked)

        sliders = []
        for label, field in _visible_rows(form):
            control = (
                field if isinstance(field, ParameterControl)
                else field.findChild(ParameterControl)
            )
            if control is None:
                continue
            assert label.isVisible()
            sliders.append(control)
            slider, box = control.slider, control.value_box
            assert slider.geometry().right() < box.geometry().left()
            y_distance = abs(
                slider.geometry().center().y() - box.geometry().center().y()
            )
            assert y_distance <= 1
        assert len(sliders) >= 2
        starts = {
            control.mapTo(form.parentWidget(), QPoint()).x() for control in sliders
        }
        assert len(starts) == 1

    assert modes[: len(widths)] == modes[len(widths) :][::-1]
    assert modes[0] and not modes[len(widths) - 1]
    assert modes[: len(widths)] == sorted(modes[: len(widths)], reverse=True)
    assert node.params == original_params
    assert widget._parameter_widgets == original_controls


def test_hidden_parameter_rows_do_not_set_the_shared_breakpoint(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    form = _InspectorParameterFormLayout(host)
    short_field = QLineEdit()
    long_field = QLineEdit()
    form.addRow("Short", short_field)
    form.addRow("Longer scientific parameter " * 8, long_field)
    host.setFixedWidth(550)
    host.resize(550, 420)
    host.show()
    qtbot.waitExposed(host)

    def sync():
        VippWidget._set_inspector_form_stacked(form, False, available_width=550)
        QApplication.processEvents()

    sync()
    assert form.rowWrapPolicy() == QFormLayout.WrapAllRows
    assert _stacked_rows(form) == {True}

    form.setRowVisible(long_field, False)
    sync()
    assert form.rowWrapPolicy() == QFormLayout.DontWrapRows
    assert _stacked_rows(form) == {False}

    form.setRowVisible(long_field, True)
    sync()
    assert form.rowWrapPolicy() == QFormLayout.WrapAllRows
    assert _stacked_rows(form) == {True}


def test_inline_fit_accounts_for_both_shared_columns(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    form = _InspectorParameterFormLayout(host)
    wide_field = QLineEdit()
    wide_field.setMinimumWidth(330)
    short_field = QLineEdit()
    form.addRow("A deliberately long label in another row", short_field)
    form.addRow("Short", wide_field)
    host.show()
    qtbot.waitExposed(host)

    def sync(width):
        host.setFixedWidth(width)
        host.resize(width, 300)
        QApplication.processEvents()
        VippWidget._set_inspector_form_stacked(form, False, available_width=width)
        QApplication.processEvents()

    sync(460)
    assert form.rowWrapPolicy() == QFormLayout.WrapAllRows
    assert _stacked_rows(form) == {True}
    sync(1050)
    assert form.rowWrapPolicy() == QFormLayout.DontWrapRows
    assert _stacked_rows(form) == {False}
