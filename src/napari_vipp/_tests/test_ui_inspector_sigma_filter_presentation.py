from __future__ import annotations

import numpy as np
import pytest
from qtpy.QtWidgets import QApplication, QFormLayout, QLabel, QWidget

from napari_vipp._tests.test_ui_inspector_widget_integration import (
    _publish_array_output,
    _select,
    _widget,
)
from napari_vipp._widget import (
    DEFAULT_SLICE_WISE_PROCESSING_TOOLTIP,
    INSPECTOR_STACKED_FORM_BREAKPOINT,
    SLICE_WISE_PROCESSING_TOOLTIP,
)
from napari_vipp.core.pipeline import (
    DEFAULT_SLICE_WISE_STACK_NOTICE,
    SLICE_WISE_STACK_NOTICE,
)


def _selected_sigma_filter(qtbot):
    data = np.arange(4 * 24 * 32, dtype=np.uint16).reshape(4, 24, 32)
    widget = _widget(qtbot, data, axes="ZYX")
    sigma = widget.add_node_from_palette("sigma_filter")
    _publish_array_output(widget, "input", data, axes="ZYX")
    widget._connect_nodes("input", sigma.id)
    _select(widget, sigma.id)
    return widget


def _spanning_labels(form: QFormLayout) -> tuple[QLabel, ...]:
    labels: list[QLabel] = []
    for row in range(form.rowCount()):
        item = form.itemAt(row, QFormLayout.SpanningRole)
        if item is not None and isinstance(item.widget(), QLabel):
            labels.append(item.widget())
    return tuple(labels)


def _guidance_text(widgets: tuple[QWidget, ...]) -> str:
    parts: list[str] = []
    for widget in widgets:
        parts.extend((widget.toolTip(), widget.accessibleDescription()))
    return " ".join(part for part in parts if part).casefold()


def test_sigma_filter_keeps_one_concise_inline_note_and_tooltips_hold_details(qtbot):
    widget = _selected_sigma_filter(qtbot)

    notes = tuple(
        note for note in _spanning_labels(widget.parameter_form) if note.text().strip()
    )
    assert len(notes) == 1
    inline_text = notes[0].text()
    assert inline_text == SLICE_WISE_STACK_NOTICE
    assert notes[0].toolTip() == SLICE_WISE_PROCESSING_TOOLTIP
    assert notes[0].accessibleDescription() == SLICE_WISE_PROCESSING_TOOLTIP
    assert len(inline_text) <= 100
    assert "Fiji" not in inline_text
    assert "ROI/mask" not in inline_text
    assert "Hidden because" not in inline_text
    assert "Stored values" not in inline_text

    guidance = _guidance_text(
        (
            widget.selected_title,
            widget.parameter_group.header,
            widget.parameter_group.summary_label,
            notes[0],
        )
    )
    assert "fiji sigma filter plus" in guidance
    assert "clamped" in guidance
    assert "uint8" in guidance and "uint16" in guidance and "float32" in guidance
    assert "roi/mask" in guidance
    assert "3d" in guidance
    assert "reorder axes" in guidance
    summary_guidance = _guidance_text((widget.parameter_group.summary_label,))
    assert "hidden setting preserved" in summary_guidance
    assert "channel axis (-1 = scalar): -1" in summary_guidance


@pytest.mark.parametrize(
    "operation_id",
    (
        "gaussian_blur",
        "canny_edges",
        "adaptive_mean_threshold",
        "dilate",
        "imagej_auto_threshold",
    ),
)
def test_slice_wise_nodes_share_inline_vocabulary_and_guidance(
    qtbot,
    operation_id,
):
    data = np.arange(4 * 24 * 32, dtype=np.uint16).reshape(4, 24, 32)
    widget = _widget(qtbot, data, axes="ZYX")
    node = widget.add_node_from_palette(operation_id)
    _publish_array_output(widget, "input", data, axes="ZYX")
    widget._connect_nodes("input", node.id)
    _select(widget, node.id)

    notice = widget._parameter_widgets["operation_notice"]
    assert notice.text() == SLICE_WISE_STACK_NOTICE
    assert notice.toolTip() == SLICE_WISE_PROCESSING_TOOLTIP
    assert notice.accessibleDescription() == SLICE_WISE_PROCESSING_TOOLTIP


@pytest.mark.parametrize(
    "operation_id",
    ("rolling_ball_background", "subtract_background"),
)
def test_selectable_rolling_ball_nodes_show_default_2d_notice_only_in_2d(
    qtbot,
    operation_id,
):
    data = np.arange(4 * 24 * 32, dtype=np.uint16).reshape(4, 24, 32)
    widget = _widget(qtbot, data, axes="ZYX")
    node = widget.add_node_from_palette(operation_id)
    _publish_array_output(widget, "input", data, axes="ZYX")
    widget._connect_nodes("input", node.id)
    _select(widget, node.id)

    notice = widget._parameter_widgets["operation_notice"]
    assert notice.text() == DEFAULT_SLICE_WISE_STACK_NOTICE
    assert notice.toolTip() == DEFAULT_SLICE_WISE_PROCESSING_TOOLTIP
    assert "large radii" in notice.toolTip()

    for spatial_mode in ("3D ZYX", "Auto from axes"):
        widget.pipeline.set_param(node.id, "spatial_mode", spatial_mode)
        widget._render_parameters(node.id)
        assert "operation_notice" not in widget._parameter_widgets

    widget.pipeline.set_param(node.id, "spatial_mode", "2D YX")
    widget._render_parameters(node.id)
    notice = widget._parameter_widgets["operation_notice"]
    assert notice.text() == DEFAULT_SLICE_WISE_STACK_NOTICE
    assert notice.toolTip() == DEFAULT_SLICE_WISE_PROCESSING_TOOLTIP


def _set_inspector_width(widget, width: int) -> None:
    widget.parameter_group.content_widget.resize(width, 700)
    widget.parameter_form_widget.resize(max(width - 14, 1), 700)
    widget.inspector_content.resize(width, 700)
    widget.inspector_viewport.resize(width, 700)
    widget._sync_inspector_responsive_layout()
    QApplication.processEvents()


def test_stacked_parameter_label_uses_full_row_before_wrapping(qtbot):
    widget = _selected_sigma_filter(qtbot)

    control = widget._parameter_widgets["minimum_pixel_fraction"]
    label = widget.parameter_form.labelForField(control)
    assert isinstance(label, QLabel)
    natural_width = label.fontMetrics().horizontalAdvance(label.text())

    # This width uses stacked form rows, but still has ample room for the label.
    roomy_width = INSPECTOR_STACKED_FORM_BREAKPOINT - 10
    _set_inspector_width(widget, roomy_width)
    assert widget.parameter_form.rowWrapPolicy() == QFormLayout.WrapAllRows
    assert label.wordWrap()
    roomy_label_width = label.sizeHint().width()
    roomy_label_height = label.heightForWidth(roomy_label_width)
    assert roomy_label_width >= natural_width

    # At a genuinely narrow dock the same label may wrap, but it must still own
    # the complete stacked row instead of being limited to the slider column.
    narrow_width = max(natural_width - 30, 150)
    _set_inspector_width(widget, narrow_width)
    narrow_label_width = label.sizeHint().width()
    assert 0 < narrow_label_width < natural_width
    assert label.heightForWidth(narrow_label_width) > roomy_label_height
    form_width = widget.parameter_form_widget.contentsRect().width()
    form_margins = widget.parameter_form.contentsMargins()
    usable_form_width = form_width - form_margins.left() - form_margins.right()
    assert narrow_label_width >= usable_form_width - 2
