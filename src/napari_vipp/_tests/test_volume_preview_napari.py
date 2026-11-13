"""Exercise the real napari model handoff without creating an OpenGL window."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import napari
import numpy as np
import pytest
from napari.components import ViewerModel
from napari.layers import Labels

from napari_vipp.ui import volume_preview_napari as adapter
from napari_vipp.ui.volume_preview import RendererVolumeLimits


@pytest.fixture
def viewer():
    model = ViewerModel(ndisplay=2)
    slicer = getattr(model, "_layer_slicer", None)
    force_sync = getattr(slicer, "force_sync", None)
    try:
        with force_sync() if callable(force_sync) else nullcontext():
            yield model
    finally:
        if callable(getattr(slicer, "shutdown", None)):
            slicer.shutdown()


def _require_owned_guard(layer):
    current = getattr(
        getattr(layer, "_slicing_state", None), "_make_slice_request_internal", None
    )
    if not getattr(current, "_vipp_volume_preview", False):
        if napari.__version__ == "0.9.0":
            pytest.fail("The napari 0.9.0 scalar slice boundary was not guarded.")
        pytest.skip("This napari version does not expose the recognized boundary.")


def test_owned_request_is_bounded_before_any_3d_set_data_and_restores_2d(
    viewer,
    monkeypatch,
):
    monkeypatch.setattr(
        adapter, "renderer_volume_limits", lambda: RendererVolumeLimits(6, 4096)
    )
    data = np.arange(5 * 2 * 11 * 13, dtype=np.uint16).reshape(5, 2, 11, 13)
    data.setflags(write=False)
    layer = adapter.add_managed_scalar_layer(
        viewer,
        data,
        _vipp_axis_labels=("Z", "C", "Y", "X"),
        metadata={"napari_vipp_kind": "inspect"},
        scale=(2, 3, 5, 7),
        translate=(11, 13, 17, 19),
        rgb=False,
    )
    _require_owned_guard(layer)
    untouched = viewer.add_image(
        data,
        rgb=False,
        name="Unrelated layer",
        scale=layer.scale,
        translate=layer.translate,
    )
    viewer.dims.order = (1, 0, 3, 2)
    viewer.dims.set_current_step(1, 1)
    observed = []
    layer.events.set_data.connect(
        lambda _event: observed.append(
            (layer._slice_input.ndisplay, layer._slice.image.raw.shape)
        )
    )

    viewer.dims.ndisplay = 3
    # Nontrivial displayed axis order (Z,X,Y), with C preserved at one index.
    shape_3d = (5, 5, 6)
    assert observed and all(shape == shape_3d for mode, shape in observed if mode == 3)
    assert observed[-1] == (3, shape_3d)
    np.testing.assert_array_equal(
        layer._slice.image.raw, data[:, 1, ::2, ::3].transpose(0, 2, 1)
    )
    assert np.shares_memory(layer._slice.image.raw, data)
    assert layer.data is data
    assert untouched._slice.image.raw.shape == (5, 13, 11)
    assert adapter.VOLUME_PREVIEW_METADATA not in untouched.metadata
    np.testing.assert_array_equal(layer.scale, (2, 3, 5, 7))
    np.testing.assert_array_equal(layer.translate, (11, 13, 17, 19))
    tile = layer._slice.tile_to_data
    np.testing.assert_array_equal(tile((2, 1, 3, 4)), (2, 1, 6, 12))
    np.testing.assert_array_equal(
        layer._transforms["data2physical"](tile((2, 1, 3, 4))), (15, 16, 47, 103)
    )
    assert layer.metadata[adapter.VOLUME_PREVIEW_METADATA]["strides"] == (1, 1, 2, 3)

    # The worker entry point receives the same bounded immutable request;
    # response calculation needs neither a live layer nor a GL context.
    request = layer._slicing_state._make_slice_request(viewer.dims)
    response = request()
    assert response.request_id == request.id
    np.testing.assert_array_equal(response.image.raw, layer._slice.image.raw)
    np.testing.assert_array_equal(response.tile_to_data.scale, (1, 1, 2, 3))

    # A transient C/Y/X order before the main widget corrects the dimensions
    # must never subsample the channel axis, even if the texture is rejected.
    viewer.dims.order = (0, 1, 2, 3)
    assert layer._slice.empty
    assert layer.metadata[adapter.VOLUME_PREVIEW_METADATA]["mode"] == "unavailable"
    assert layer.data is data
    viewer.dims.order = (1, 0, 3, 2)

    viewer.dims.ndisplay = 2
    assert layer._slice.image.raw.shape == (13, 11)
    assert np.shares_memory(layer._slice.image.raw, data)
    np.testing.assert_array_equal(layer._slice.tile_to_data.scale, (1, 1, 1, 1))
    assert adapter.VOLUME_PREVIEW_METADATA not in layer.metadata


@pytest.mark.parametrize("labels", (False, True))
def test_binary_mask_stays_exact_or_is_not_drawn(viewer, monkeypatch, labels):
    limits = RendererVolumeLimits(max_texture_axis=6, max_texture_bytes=4096)
    monkeypatch.setattr(adapter, "renderer_volume_limits", lambda: limits)
    # Exercise a layer inserted while 3D is already active, not just a toggle.
    viewer.dims.ndisplay = 3
    data = np.zeros((5, 11, 13), dtype=np.uint8 if labels else bool)
    data[2, 4, 6] = 1
    data.setflags(write=False)
    layer = adapter.add_managed_scalar_layer(
        viewer,
        data,
        labels=labels,
        metadata={"data_kind": "mask"},
    )
    _require_owned_guard(layer)
    assert isinstance(layer, Labels) == labels
    assert layer._slice.empty
    assert layer._slice.image.raw.shape == (1, 1, 1)
    assert layer.metadata[adapter.VOLUME_PREVIEW_METADATA]["mode"] == "unavailable"
    assert layer.data is data

    viewer.dims.ndisplay = 2
    viewer.dims.set_current_step(0, 2)
    assert not layer._slice.empty
    np.testing.assert_array_equal(layer._slice.image.raw, data[2])
    assert adapter.VOLUME_PREVIEW_METADATA not in layer.metadata

    limits = RendererVolumeLimits(max_texture_axis=16, max_texture_bytes=4096)
    viewer.dims.ndisplay = 3
    assert not layer._slice.empty
    np.testing.assert_array_equal(layer._slice.image.raw, data)
    assert layer._slice.image.raw.dtype == data.dtype
    assert adapter.VOLUME_PREVIEW_METADATA not in layer.metadata


@pytest.mark.parametrize(
    ("raw_dtype", "texture_dtype", "high_id"),
    [(np.int32, np.uint8, 100_003), (np.uint16, np.uint16, 50_003)],
)
def test_owned_labels_admit_the_native_texture_size_not_the_raw_id_size(
    viewer, monkeypatch, raw_dtype, texture_dtype, high_id
):
    data = np.zeros((5, 11, 13), dtype=raw_dtype)
    data[2, 3, 4], data[2, 7, 8] = 7, high_id
    data.setflags(write=False)
    texture_bytes = np.dtype(texture_dtype).itemsize
    calls = []

    def admit(whd, size):
        calls.append((whd, size))
        return True

    limits = RendererVolumeLimits(16, data.size * texture_bytes, admit)
    monkeypatch.setattr(adapter, "renderer_volume_limits", lambda: limits)
    layer = adapter.add_managed_scalar_layer(
        viewer,
        data,
        labels=True,
        _vipp_axis_labels=("Z", "Y", "X"),
        metadata={"napari_vipp_kind": "inspect", "data_kind": "labels"},
        scale=(2, 3, 5),
        translate=(11, 13, 17),
    )
    _require_owned_guard(layer)
    if layer._slice_dtype() != np.dtype(texture_dtype):
        if napari.__version__ == "0.9.0":
            pytest.fail("Unexpected native label texture encoding in napari 0.9.0.")
        pytest.skip("This napari version uses a different native label encoding.")
    viewer.dims.ndisplay = 3

    assert not layer._slice.empty
    assert calls and all(call == ((13, 11, 5), texture_bytes) for call in calls)
    assert adapter.VOLUME_PREVIEW_METADATA not in layer.metadata
    assert layer._slice.image.view.dtype == texture_dtype
    assert layer._slice.image.raw.dtype == raw_dtype
    np.testing.assert_array_equal(layer._slice.image.raw, data)
    assert layer.data is data and not layer.data.flags.writeable
    assert np.shares_memory(layer._slice.image.raw, data)
    np.testing.assert_array_equal(layer.scale, (2, 3, 5))
    np.testing.assert_array_equal(layer.translate, (11, 13, 17))
    np.testing.assert_array_equal(layer._slice.tile_to_data.scale, (1, 1, 1))

    # A renderer that really cannot fit the exact categorical texture must
    # still hide only the 3D presentation, never downsample labels.
    viewer.dims.ndisplay = 2
    limits = RendererVolumeLimits(16, data.size * texture_bytes - 1, admit)
    viewer.dims.ndisplay = 3
    assert layer._slice.empty
    note = layer.metadata[adapter.VOLUME_PREVIEW_METADATA]
    assert note["mode"] == "unavailable" and note["strides"] == (1, 1, 1)
    viewer.dims.ndisplay = 2
    viewer.dims.set_current_step(0, 2)
    np.testing.assert_array_equal(layer._slice.image.raw, data[2])
    assert adapter.VOLUME_PREVIEW_METADATA not in layer.metadata
    assert layer.data is data and int(layer.data[2, 7, 8]) == high_id


def test_main_inspect_and_pin_keep_int32_labels_visible_in_3d(
    qtbot, viewer, monkeypatch
):
    from napari_vipp._widget import VippWidget
    from napari_vipp.core.metadata import image_state_from_array

    data = np.zeros((5, 11, 13), dtype=np.int32)
    data[2, 3, 4], data[2, 7, 8] = 3, 100_003
    data.setflags(write=False)
    state = replace(
        image_state_from_array(data, layer_metadata={"axes": "ZYX"}), kind="label image"
    )
    monkeypatch.setattr(
        adapter, "renderer_volume_limits", lambda: RendererVolumeLimits(16, data.size)
    )
    widget = VippWidget(viewer, defer_initial_run=True, initial_compute_mode="cpu")
    qtbot.addWidget(widget)
    node = widget.pipeline.add_node("label_connected_components")
    widget._build_graph_from_pipeline()
    widget.pipeline.outputs[node.id] = data
    widget.pipeline.output_states[node.id] = state
    viewer.dims.ndisplay = 3

    widget.inspect_node(node.id)
    inspect = viewer.layers["VIPP Inspect"]
    assert isinstance(inspect, Labels)
    _require_owned_guard(inspect)
    if inspect._slice_dtype() != np.dtype(np.uint8):
        if napari.__version__ == "0.9.0":
            pytest.fail("Unexpected native label texture encoding in napari 0.9.0.")
        pytest.skip("This napari version uses a different native label encoding.")
    assert not inspect._slice.empty
    assert "Inspecting" in widget.status_label.text()
    widget.pin_node(node.id)
    pinned = widget._active_pinned_layer()
    assert isinstance(pinned, Labels) and pinned is not inspect
    assert "Pinned" in widget.status_label.text()

    for layer in (inspect, pinned):
        _require_owned_guard(layer)
        assert adapter.VOLUME_PREVIEW_METADATA not in layer.metadata
        assert not layer.editable and not layer.data.flags.writeable
        assert layer._slice.image.view.dtype == np.dtype(np.uint8)
        np.testing.assert_array_equal(layer._slice.image.raw, data)
        assert np.shares_memory(layer.data, data)
    assert widget.pipeline.outputs[node.id] is data
    assert widget.pipeline.output_states[node.id] is state

    viewer.dims.ndisplay = 2
    viewer.dims.set_current_step(0, 2)
    for layer in (inspect, pinned):
        np.testing.assert_array_equal(layer._slice.image.raw, data[2])


def test_main_inspect_add_in_3d_reports_preview_without_changing_output(
    qtbot,
    viewer,
    monkeypatch,
):
    from napari_vipp._widget import VippWidget
    from napari_vipp.core.metadata import image_state_from_array

    monkeypatch.setattr(
        adapter, "renderer_volume_limits", lambda: RendererVolumeLimits(6, 4096)
    )
    widget = VippWidget(viewer, defer_initial_run=True, initial_compute_mode="cpu")
    qtbot.addWidget(widget)
    data = np.arange(5 * 11 * 13, dtype=np.uint16).reshape(5, 11, 13)
    data.setflags(write=False)
    state = image_state_from_array(data, layer_metadata={"axes": "ZYX"})
    widget.pipeline.outputs["input"] = data
    widget.pipeline.output_states["input"] = state
    viewer.dims.ndisplay = 3

    widget.inspect_node("input")

    layer = viewer.layers["VIPP Inspect"]
    _require_owned_guard(layer)
    assert layer.metadata[adapter.VOLUME_PREVIEW_METADATA]["mode"] == "reduced"
    assert "reduced-resolution 3D preview" in widget.status_label.text()
    assert layer.data.shape == data.shape
    np.testing.assert_array_equal(layer._slice.image.raw, data[:, ::2, ::3])
    assert np.shares_memory(layer.data, data)
    assert widget.pipeline.outputs["input"] is data
    assert widget.pipeline.output_states["input"] is state

    viewer.dims.ndisplay = 2
    assert layer._slice.image.raw.shape == (11, 13)
    assert "reduced-resolution" not in widget.status_label.text()


def test_3d_refusal_notice_follows_visible_pinned_layer_not_workflow_status(
    qtbot, viewer, monkeypatch
):
    from napari_vipp._widget import VippWidget
    from napari_vipp.core.metadata import image_state_from_array

    data = np.zeros((5, 11, 13), dtype=np.int32)
    data[2, 3, 4] = 100_003
    state = replace(
        image_state_from_array(data, layer_metadata={"axes": "ZYX"}), kind="label image"
    )
    monkeypatch.setattr(
        adapter, "renderer_volume_limits", lambda: RendererVolumeLimits(6, 4096)
    )
    other = viewer.add_image(np.ones((5, 5, 5)), name="Other", rgb=False)
    widget = VippWidget(viewer, defer_initial_run=True, initial_compute_mode="cpu")
    qtbot.addWidget(widget)
    node = widget.pipeline.add_node("label_connected_components")
    widget._build_graph_from_pipeline()
    widget.pipeline.outputs[node.id] = data
    widget.pipeline.output_states[node.id] = state
    widget.pin_node(node.id)
    pinned = widget._active_pinned_layer()
    _require_owned_guard(pinned)
    viewer.layers.selection.select_only(other)
    assert viewer.layers.selection.active is other

    viewer.dims.ndisplay = 3
    notice = widget._volume_preview_notice
    assert pinned.visible and pinned._slice.empty
    assert pinned.metadata[adapter.VOLUME_PREVIEW_METADATA]["mode"] == "unavailable"
    qtbot.waitUntil(lambda: not notice.isHidden())
    assert "3D texture limit" in notice.text()
    assert "Switch to 2D" in notice.text()
    assert "unchanged" in notice.toolTip()
    widget._set_status("Workflow completed")
    assert not notice.isHidden()
    assert widget.status_label.text() == "Workflow completed"

    pinned.visible = False
    qtbot.waitUntil(notice.isHidden)
    pinned.visible = True
    qtbot.waitUntil(lambda: not notice.isHidden())
    full = pinned.data
    pinned.data = full[:, :5, :5]
    qtbot.waitUntil(notice.isHidden)
    pinned.data = full
    qtbot.waitUntil(lambda: not notice.isHidden())

    viewer.dims.ndisplay = 2
    qtbot.waitUntil(notice.isHidden)
    viewer.dims.set_current_step(0, 2)
    np.testing.assert_array_equal(pinned._slice.image.raw, data[2])
    assert widget.pipeline.outputs[node.id] is data
    assert widget.status_label.text() == "Workflow completed"


def test_main_3d_allocation_notice_ignores_other_canvas_and_clears_in_2d(
    qtbot, monkeypatch
):
    from napari.utils.events import EventEmitter

    from napari_vipp import _widget as ui
    from napari_vipp._tests.test_widget import _Viewer

    viewer = _Viewer()
    canvas = object()
    viewer.window = SimpleNamespace(
        _qt_viewer=SimpleNamespace(viewer=viewer, scene=canvas)
    )
    notifications = EventEmitter(type_name="render_error")
    monkeypatch.setattr(ui, "renderer_error_notifications", lambda: notifications)
    monkeypatch.setattr(ui, "scene_canvas", lambda qt_viewer: qt_viewer.scene)
    monkeypatch.setattr(
        ui,
        "is_3d_texture_allocation_failure",
        lambda notification, canvases: any(
            notification.canvas is candidate for candidate in canvases
        ),
    )
    widget = ui.VippWidget(viewer, defer_initial_run=True, initial_compute_mode="cpu")
    qtbot.addWidget(widget)
    notice = widget._volume_preview_notice
    viewer.dims.ndisplay = 3

    notifications(canvas=object())
    assert notice.isHidden() and not widget._volume_preview_allocation_failed
    notifications(canvas=canvas)
    qtbot.waitUntil(lambda: not notice.isHidden())
    assert "OpenGL could not allocate" in notice.text()
    assert viewer.dims.ndisplay == 3
    widget.status_label.setText("Still working")
    assert not notice.isHidden()

    viewer.dims.ndisplay = 2
    widget._on_dims_changed(SimpleNamespace(type="ndisplay"))
    qtbot.waitUntil(notice.isHidden)
    notifications(canvas=canvas)
    assert not widget._volume_preview_allocation_failed
    assert widget.status_label.text() == "Still working"
