"""Scientific axis semantics, rather than storage order, drive 3D display."""

from __future__ import annotations

import numpy as np
import pytest
from napari.components import ViewerModel

from napari_vipp._tests.test_ui_inspector_widget_integration import (
    _publish_array_output,
)
from napari_vipp._widget import VippWidget, _sync_viewer_spatial_order_from_layer
from napari_vipp.core.metadata import image_state_from_array


@pytest.mark.parametrize("display_kind", ("image", "labels"))
def test_generated_layer_installs_cursor_guard_without_touching_user_layer(
    qtbot, display_kind
):
    viewer = ViewerModel()
    data = np.ones((4, 6, 8), dtype=np.uint16)
    add_layer = viewer.add_image if display_kind == "image" else viewer.add_labels
    user_layer = add_layer(data)
    ray = user_layer._get_value_ray
    if "slice_shape" not in ray.__func__.__code__.co_varnames:
        pytest.skip("This napari version does not use the affected 3D ray path.")
    widget = VippWidget(viewer, defer_initial_run=True)
    qtbot.addWidget(widget)
    with widget._suspend_viewer_layer_change_handling():
        inspect = widget._add_image_or_labels(
            "VIPP Inspect",
            data,
            {
                "napari_vipp_kind": "inspect",
                "display_kind": display_kind,
                "data_kind": display_kind,
            },
        )

    assert getattr(inspect._get_value_ray, "_vipp_safe_cursor_ray", False)
    assert user_layer._get_value_ray == ray
    assert not inspect.data.flags.writeable
    np.testing.assert_array_equal(inspect.data, data)
    with widget._suspend_viewer_layer_change_handling():
        viewer.layers.clear()


@pytest.mark.parametrize("initial_3d", (False, True))
def test_zcyx_inspect_volume_retains_channel_slider_and_syncs_both_ways(
    qtbot, initial_3d,
):
    # Match native Olympus OIR's Z,C,Y,X layout, with visibly different channels.
    data = np.zeros((5, 2, 8, 9), dtype=np.uint16)
    data[:, 0] = np.arange(5)[:, None, None] + 10
    data[:, 1] = np.arange(5)[:, None, None] + 100
    before = data.copy()
    viewer = ViewerModel()
    viewer.add_image(
        data, rgb=False, name="OIR-style source", metadata={"axes": "ZCYX"}
    )
    if initial_3d:
        viewer.dims.ndisplay = 3
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    widget._should_run_pipeline_in_background = lambda *_args, **_kwargs: False
    widget.run_pipeline(force_sync=True)
    widget.graph_view.select_node("input")
    widget.follow_dims_checkbox.setChecked(True)
    inspect = viewer.layers["VIPP Inspect"]
    scientific_state = widget.pipeline.output_states["input"].to_dict()
    viewer.dims.ndisplay = 3

    assert tuple(viewer.dims.axis_labels[i] for i in viewer.dims.displayed) == (
        "Z", "Y", "X",
    )
    assert tuple(
        viewer.dims.axis_labels[i] for i in viewer.dims.not_displayed
    ) == ("C",)
    assert inspect._data_view.shape == (5, 8, 9)
    channel = next(axis for axis in widget.view_dims_bar._axes if axis.label == "C")
    widget._on_view_dim_changed(channel.step_axis, 1)
    assert viewer.dims.current_step[1] == 1
    np.testing.assert_array_equal(inspect._data_view, data[:, 1])

    viewer.dims.set_current_step(1, 0)
    values = {axis.label: axis.value for axis in widget.view_dims_bar._axes}
    assert values["C"] == 0
    np.testing.assert_array_equal(inspect._data_view, data[:, 0])
    viewer.dims.ndisplay = 2
    assert set(
        viewer.dims.axis_labels[i] for i in viewer.dims.not_displayed
    ) == {"Z", "C"}
    viewer.dims.set_current_step(0, 3)
    values = {axis.label: axis.value for axis in widget.view_dims_bar._axes}
    assert values["Z"] == 3
    np.testing.assert_array_equal(inspect._data_view, data[3, 0])
    assert widget.pipeline.output_states["input"].to_dict() == scientific_state
    np.testing.assert_array_equal(data, before)
    assert inspect.data.shape == data.shape
    viewer.layers.clear()


def _select_cached_subtract_background(widget, data):
    node = widget.pipeline.add_node("subtract_background")
    widget.graph_view.add_node(node, widget.graph_view.suggest_append_position("input"))
    _publish_array_output(widget, node.id, data, axes="ZYX")
    widget.graph_view.select_node(node.id)
    return node


def test_cached_scalar_inspect_keeps_existing_channel_label_and_displays_zyx(qtbot):
    data = np.arange(4 * 7 * 8 * 9, dtype=np.uint16).reshape(4, 7, 8, 9)
    viewer = ViewerModel()
    source = viewer.add_image(data, rgb=False, axis_labels=tuple("CZYX"))
    viewer.dims.ndisplay = 3
    viewer.dims.set_current_step(0, 2)
    widget = VippWidget(viewer, defer_initial_run=True)
    qtbot.addWidget(widget)
    result = data[1].copy()
    node = _select_cached_subtract_background(widget, result)
    inspect = viewer.layers["VIPP Inspect"]

    assert viewer.layers.selection.active is inspect
    assert inspect.metadata["node_id"] == node.id
    assert inspect.ndim == 3 and viewer.dims.ndim == 4
    assert tuple(viewer.dims.axis_labels) == tuple("CZYX")
    assert tuple(viewer.dims.displayed) == (1, 2, 3)
    assert tuple(viewer.dims.not_displayed) == (0,)
    assert viewer.dims.current_step[0] == 2
    np.testing.assert_array_equal(inspect._data_view, result)
    assert source in viewer.layers
    assert source.data is data
    assert source.axis_labels == tuple("CZYX")
    assert source.visible


def test_rank_reduction_keeps_trailing_depth_step_in_2d(qtbot):
    data = np.arange(4 * 7 * 8 * 9, dtype=np.uint16).reshape(4, 7, 8, 9)
    viewer = ViewerModel()
    widget = VippWidget(viewer, defer_initial_run=True)
    qtbot.addWidget(widget)
    _publish_array_output(widget, "input", data, axes="CZYX")
    widget.inspect_node("input")
    source_inspect = viewer.layers["VIPP Inspect"]
    assert viewer.dims.ndim == 4 and viewer.dims.ndisplay == 2
    viewer.dims.set_current_step(0, 1)
    viewer.dims.set_current_step(1, 5)
    np.testing.assert_array_equal(source_inspect._data_view, data[1, 5])

    result = data[1].copy()
    _select_cached_subtract_background(widget, result)
    inspect = viewer.layers["VIPP Inspect"]

    assert inspect is not source_inspect
    assert len(viewer.layers) == 1 and viewer.dims.ndim == 3
    assert viewer.dims.ndisplay == 2
    assert tuple(viewer.dims.axis_labels) == tuple("ZYX")
    assert tuple(viewer.dims.displayed) == (1, 2)
    assert viewer.dims.current_step[0] == 5
    np.testing.assert_array_equal(inspect._data_view, result[5])


def _managed_layer(viewer, axes, *, explicit=True, kind="inspect"):
    data = np.zeros(tuple({"Y": 8, "X": 9}.get(axis, 2) for axis in axes))
    state = image_state_from_array(
        data, layer_metadata={"axes": axes} if explicit else {},
    )
    metadata = {
        "napari_vipp_kind": kind,
        "vipp_image_state": state.to_dict(),
        "display_ndim": data.ndim,
        "display_shape": data.shape,
        "display_rgb": False,
    }
    return viewer.add_image(
        data, rgb=False, metadata=metadata,
        axis_labels=tuple(axis.short_label for axis in state.axes),
        scale=tuple(axis.scale for axis in state.axes),
    )


@pytest.mark.parametrize("axes", ("ZCYX", "ZTCYX", "TCZYX", "CTZYX", "ZYXC"))
@pytest.mark.parametrize("kind", ("inspect", "pinned", "source_preview"))
def test_managed_spatial_order_supports_interleaved_axes_and_viewer_offsets(
    axes, kind,
):
    viewer = ViewerModel()
    viewer.add_image(np.zeros((2,) * 6), rgb=False)
    layer = _managed_layer(viewer, axes, kind=kind)
    viewer.dims.ndisplay = 3
    before_data, before_metadata = layer.data, layer.metadata.copy()
    _sync_viewer_spatial_order_from_layer(viewer, layer)
    offset = viewer.dims.ndim - layer.ndim
    assert tuple(axis - offset for axis in viewer.dims.displayed) == tuple(
        axes.index(name) for name in "ZYX"
    )
    assert layer.data is before_data
    assert layer.metadata == before_metadata
    # A second sync is a no-op, not a stream of recursive display events.
    assert not _sync_viewer_spatial_order_from_layer(viewer, layer)
    viewer.layers.clear()


@pytest.mark.parametrize(
    ("axes", "explicit", "kind", "order"),
    (
        ("ZCYX", True, "unmanaged", (0, 1, 2, 3)),
        ("ZCYX", False, "inspect", (0, 1, 2, 3)),
        ("TCYX", True, "inspect", (0, 1, 2, 3)),
        ("ZCYX", True, "inspect", (1, 2, 0, 3)),
    ),
)
def test_spatial_order_preserves_user_layers_unknown_axes_and_valid_orientation(
    axes, explicit, kind, order,
):
    viewer = ViewerModel()
    layer = _managed_layer(viewer, axes, explicit=explicit, kind=kind)
    viewer.dims.ndisplay = 3
    viewer.dims.order = order
    assert not _sync_viewer_spatial_order_from_layer(viewer, layer)
    assert viewer.dims.order == order
    viewer.layers.clear()
