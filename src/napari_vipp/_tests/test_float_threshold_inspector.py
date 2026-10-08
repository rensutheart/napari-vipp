"""Raw float intensity cutoffs stay editable across deferred histogram work."""

from __future__ import annotations

import json

import numpy as np
import pytest
from qtpy.QtCore import QPoint, Qt

from napari_vipp._tests.test_widget import _Viewer
from napari_vipp._widget import VippWidget
from napari_vipp.core.workflow import deserialize_workflow, serialize_workflow


def _widget(qtbot, data, operation="binary_threshold"):
    data.setflags(write=False)
    widget = VippWidget(_Viewer(data, metadata={"axes": "ZYX"}))
    qtbot.addWidget(widget)
    widget._delete_nodes({"gaussian", "threshold"})
    node = widget.add_node_from_palette(operation)
    widget._connect_nodes("input", node.id)
    widget.graph_view.select_node(node.id)
    return widget, node


def _defer_histogram(qtbot, monkeypatch, widget, node):
    # Exercise the same deferred range branch as the user's hundred-million-
    # voxel volume, without allocating that volume in every regression case.
    monkeypatch.setattr("napari_vipp._widget.AUTO_BACKGROUND_MIN_BYTES", 1)
    monkeypatch.setattr("napari_vipp._widget.AUTO_BACKGROUND_MIN_ELEMENTS", 1)
    widget._clear_input_histogram_cache()
    widget._update_rescale_input_histogram(node.id, widget._current_step())
    qtbot.waitUntil(
        lambda: widget._active_input_histogram_run_id is None,
        timeout=5_000,
    )


@pytest.mark.parametrize("foreground", ("Above", "Below"))
@pytest.mark.parametrize("deferred", (False, True))
def test_raw_float_histogram_edit_updates_cutoff_control_mask_and_saved_workflow(
    qtbot,
    monkeypatch,
    foreground,
    deferred,
):
    data = np.array([0, 5_000, 23_670, 25_000, np.nan, np.inf], np.float32).reshape(
        1, 2, 3
    )
    before = data.copy()
    widget, node = _widget(qtbot, data)
    widget._parameter_widgets["foreground"].combo.setCurrentText(foreground)
    if deferred:
        _defer_histogram(qtbot, monkeypatch, widget, node)

        def forbidden_scan(*_args, **_kwargs):
            raise AssertionError("Threshold controls must not scan a large volume")

        monkeypatch.setattr("napari_vipp._widget._finite_values", forbidden_scan)

    control = widget._parameter_widgets["threshold"]
    assert control._bounds.maximum == 25_000
    original = node.params["threshold"]
    plot = widget.rescale_input_histogram_plot
    plot.markerChanged.emit("threshold", 23_670.0)
    widget._debounce_timer.stop()
    assert node.params["threshold"] == control.value() == 23_670.0
    assert plot.marker_values()["threshold"] == 23_670.0
    qtbot.waitUntil(
        lambda: (
            widget._active_pipeline_run_id is None
            and widget._active_thumbnail_contrast_run_id is None
        ),
        timeout=5_000,
    )
    widget.run_pipeline(force_sync=True)
    assert widget.pipeline.outputs[node.id] is not None, widget.status_label.text()
    expected = data > 23_670 if foreground == "Above" else data < 23_670
    np.testing.assert_array_equal(widget.pipeline.outputs[node.id], expected)
    np.testing.assert_array_equal(data, before)
    assert not data.flags.writeable
    restored = deserialize_workflow(
        json.loads(json.dumps(serialize_workflow(widget.pipeline)))
    )
    restored_node = next(record for record in restored["nodes"] if record.id == node.id)
    assert restored_node.params["threshold"] == 23_670.0
    widget.undo()
    assert widget.pipeline.nodes[node.id].params["threshold"] == original
    qtbot.waitUntil(
        lambda: not widget._workflow_tab_switch_block_reason(), timeout=5_000
    )
    widget.redo()
    assert widget.pipeline.nodes[node.id].params["threshold"] == 23_670.0


def test_deferred_signed_float_range_and_typed_cutoff(qtbot, monkeypatch):
    data = np.linspace(-25_000, 30_000, 24, dtype=np.float32).reshape(2, 3, 4)
    widget, node = _widget(qtbot, data)
    _defer_histogram(qtbot, monkeypatch, widget, node)
    control = widget._parameter_widgets["threshold"]
    assert control._bounds.minimum == -25_000
    control.value_box.setValue(-12_345)
    assert node.params["threshold"] == -12_345
    widget.rescale_input_histogram_plot.markerChanged.emit("threshold", -20_000)
    widget._debounce_timer.stop()
    assert node.params["threshold"] == control.value() == -20_000


def test_saved_cutoff_outside_slice_is_preserved_and_cache_is_identity_checked(
    qtbot,
    monkeypatch,
):
    data = np.array(
        [0, 100, 300, 500, 750, 1_000, 0, 10_000, 20_000, 30_000, 0, 0], np.float32
    ).reshape(2, 2, 3)
    widget, node = _widget(qtbot, data)
    widget.pipeline.set_param(node.id, "threshold", 25_000)
    _defer_histogram(qtbot, monkeypatch, widget, node)
    assert (
        node.params["threshold"]
        == widget._parameter_widgets["threshold"].value()
        == 25_000
    )
    source = widget.pipeline.input_data_for_node(node.id)
    assert widget._threshold_histogram_range(node.id, source) == (0, 1_000)
    widget.viewer.dims.set_current_step(0, 1)
    widget._update_rescale_input_histogram(node.id, widget._current_step())
    qtbot.waitUntil(
        lambda: widget._active_input_histogram_run_id is None, timeout=5_000
    )
    assert widget._parameter_widgets["threshold"]._bounds.maximum == 30_000
    assert node.params["threshold"] == 25_000
    replacement = np.full_like(source, 200)
    widget.pipeline.outputs["input"] = replacement
    assert widget._threshold_histogram_range(node.id, replacement) is None
    spec = widget._parameter_spec_by_name(node.id, "threshold")
    bounds = widget._threshold_bounds(node.id, spec)
    assert bounds.minimum <= 25_000 <= bounds.maximum


@pytest.mark.parametrize("foreground", ("In range", "Outside range"))
def test_float_range_guides_respect_linked_limits_and_restore_unchanged_marker(
    qtbot,
    monkeypatch,
    foreground,
):
    data = np.linspace(0, 30_000, 24, dtype=np.float32).reshape(1, 4, 6)
    widget, node = _widget(qtbot, data)
    widget._parameter_widgets["foreground"].combo.setCurrentText(foreground)
    widget._parameter_widgets["high_threshold"].value_box.setValue(25_000)
    widget._parameter_widgets["low_threshold"].value_box.setValue(5_000)
    _defer_histogram(qtbot, monkeypatch, widget, node)
    plot = widget.rescale_input_histogram_plot
    plot.markerChanged.emit("low", 26_000)
    widget._debounce_timer.stop()
    assert node.params["low_threshold"] == 25_000
    # Simulate the plot's immediate raw guide movement before its release signal.
    plot._markers = [("low", 27_000, "orange"), ("high", 25_000, "orange")]
    plot.markerChanged.emit("low", 27_000)
    assert node.params["low_threshold"] == 25_000
    assert plot.marker_values()["low"] == 25_000
    assert widget._parameter_widgets["low_threshold"].value() == 25_000


def test_unit_float_slider_and_fractional_drag_remain_synchronized(qtbot):
    widget, node = _widget(
        qtbot, np.linspace(0, 1, 12, dtype=np.float32).reshape(1, 3, 4)
    )
    control = widget._parameter_widgets["threshold"]
    assert (control._bounds.minimum, control._bounds.maximum) == (0, 1)
    assert control._bounds.step == 0.01
    plot = widget.rescale_input_histogram_plot
    plot.markerChanged.emit("threshold", 0.567)
    assert node.params["threshold"] == control.value() == 0.567
    plot._markers = [("threshold", 0.5671, "orange")]
    plot.markerChanged.emit("threshold", 0.5671)
    assert plot.marker_values()["threshold"] == 0.567


def test_integer_guide_uses_the_same_whole_number_as_the_numeric_control(qtbot):
    widget, node = _widget(qtbot, np.arange(24, dtype=np.uint8).reshape(1, 4, 6))
    widget.rescale_input_histogram_plot.markerChanged.emit("threshold", 12.3)
    assert (
        node.params["threshold"] == widget._parameter_widgets["threshold"].value() == 12
    )
    assert widget.rescale_input_histogram_plot.marker_values()["threshold"] == 12


def test_real_mouse_drag_updates_raw_float_cutoff_and_numeric_control(
    qtbot, monkeypatch
):
    data = np.linspace(0, 30_000, 256, dtype=np.float32).reshape(1, 16, 16)
    widget, node = _widget(qtbot, data)
    widget._parameter_widgets["threshold"].value_box.setValue(15_000)
    _defer_histogram(qtbot, monkeypatch, widget, node)
    plot = widget.rescale_input_histogram_plot
    plot.setParent(None)
    qtbot.addWidget(plot)
    plot.resize(600, 180)
    plot.show()
    qtbot.waitExposed(plot)
    rect = plot._plot_rect()
    start = QPoint(rect.left() + round(rect.width() * 0.5), rect.center().y())
    target = QPoint(rect.left() + round(rect.width() * 0.8), rect.center().y())
    expected = round(plot._value_from_x(target.x(), rect), 3)
    with qtbot.waitSignal(plot.markerChanged):
        qtbot.mousePress(plot, Qt.LeftButton, pos=start)
        qtbot.mouseMove(plot, target)
        qtbot.mouseRelease(plot, Qt.LeftButton, pos=target)
    widget._debounce_timer.stop()
    assert node.params["threshold"] == expected
    assert widget._parameter_widgets["threshold"].value() == expected
    assert plot.marker_values()["threshold"] == expected
    assert plot._drag_marker is None
