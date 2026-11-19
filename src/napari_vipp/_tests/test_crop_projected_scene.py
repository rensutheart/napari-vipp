"""Calibrated Crop editing in slice, projection and 3D without requiring a GPU."""

from dataclasses import replace
from itertools import product

import numpy as np
import pytest
from napari.components import ViewerModel

from napari_vipp import _widget
from napari_vipp.core.batch import scientific_workflow_hash
from napari_vipp.core.metadata import AxisMetadata
from napari_vipp.core.source_window import ExactSourceWindowData, SourceWindowIdentity
from napari_vipp.core.workflow import serialize_workflow
from napari_vipp.ui.crop_slice_sampling import crop_intersects_slice


class _ShapeOnlyCropData:
    def __init__(self, shape):
        self.shape = shape

    def __array__(self, *args, **kwargs):
        raise AssertionError("Crop retention must not materialize image pixels")

    def __getitem__(self, key):
        raise AssertionError("Crop retention must not slice image pixels")


class _LegacyCropSliceLayer:
    """Expose the legacy factory, but never expose the new slicing-state API."""

    def __init__(self, layer, stale_slice):
        self._layer = layer
        self._stale_slice = stale_slice
        self.data = _ShapeOnlyCropData(layer.data.shape)
        self.cached_reads = 0
        self.factory_calls = 0

    @property
    def _data_slice(self):
        self.cached_reads += 1
        return self._stale_slice

    @property
    def _data_to_world(self):
        return self._layer._data_to_world

    @property
    def projection_mode(self):
        return self._layer.projection_mode

    def _make_slice_input(self, dims):
        self.factory_calls += 1
        factory = getattr(self._layer, "_make_slice_input", None)
        if callable(factory):
            return factory(dims)
        # Exercise the old API shape with real calibrated napari slice inputs
        # on current napari too; do not reimplement its transform/slice maths.
        return self._layer._slicing_state.make_slice_input(dims)


@pytest.mark.parametrize(
    "shape, translation, order, point, left, right, projection, hidden, expected",
    [
        pytest.param(
            (6, 8, 10),
            (10, 20, 30),
            (0, 1, 2),
            (14, 20, 30),
            (0, 0, 0),
            (0, 0, 0),
            "none",
            ((0, 2, 5),),
            True,
            id="retained-calibrated-point",
        ),
        pytest.param(
            (6, 8, 10),
            (10, 20, 30),
            (0, 1, 2),
            (10, 20, 30),
            (0, 0, 0),
            (0, 0, 0),
            "none",
            ((0, 2, 5),),
            False,
            id="excluded-calibrated-point",
        ),
        pytest.param(
            (6, 8, 10),
            (0, 20, 30),
            (0, 1, 2),
            (0, 20, 30),
            (0, 0, 0),
            (4, 0, 0),
            "max",
            ((0, 2, 5),),
            True,
            id="right-slab-includes-exact-upper-sample",
        ),
        pytest.param(
            (6, 8, 10),
            (0, 20, 30),
            (0, 1, 2),
            (0, 20, 30),
            (0, 0, 0),
            (3.2, 0, 0),
            "max",
            ((0, 2, 5),),
            False,
            id="right-slab-excludes-noninteger-upper-sample",
        ),
        pytest.param(
            (6, 8, 10),
            (0, 20, 30),
            (0, 1, 2),
            (8, 20, 30),
            (4, 0, 0),
            (0, 0, 0),
            "max",
            ((0, 2, 3),),
            True,
            id="left-slab-uses-calibrated-margin",
        ),
        pytest.param(
            (6, 8, 10),
            (10, 20, 30),
            (2, 0, 1),
            (10, 20, 38),
            (0, 0, 0),
            (0, 0, 0),
            "none",
            ((2, 2, 10),),
            True,
            id="orthogonal-retained-x-point",
        ),
        pytest.param(
            (6, 8, 10),
            (10, 20, 30),
            (2, 0, 1),
            (10, 20, 30),
            (0, 0, 0),
            (0, 0, 0),
            "none",
            ((2, 2, 10),),
            False,
            id="orthogonal-excluded-x-point",
        ),
        pytest.param(
            (4, 5, 6),
            (12, 26, 42),
            (0, 1, 2),
            (14, 26, 42),
            (0, 0, 0),
            (0, 0, 0),
            "none",
            ((0, 0, 4),),
            True,
            id="translated-exact-window-local-point",
        ),
    ],
)
def test_legacy_crop_sampling_uses_fresh_calibrated_request(
    shape, translation, order, point, left, right, projection, hidden, expected
):
    viewer = ViewerModel()
    layer = viewer.add_image(
        np.zeros(shape, dtype="uint8"), scale=(2, 3, 4), translate=translation
    )
    dims = viewer.dims
    dims.ndisplay = 3
    stale = layer._data_slice  # The completed volume request has no hidden plane.
    legacy = _LegacyCropSliceLayer(layer, stale)
    assert getattr(legacy, "_slicing_state", None) is None

    dims.ndisplay = 2
    dims.order = order
    dims.point = point
    dims.margin_left = left
    dims.margin_right = right
    layer.projection_mode = projection
    cached_input = layer._slice_input
    data = layer.data

    # Both the current API and the legacy factory must follow the same live
    # point/slab/order; neither may use the saved completed 3D slice request.
    assert crop_intersects_slice(layer, dims, hidden) is expected
    assert crop_intersects_slice(legacy, dims, hidden) is expected
    assert legacy.factory_calls == 1 and legacy.cached_reads == 0
    assert legacy._stale_slice is stale
    assert layer._slice_input is cached_input and layer.data is data


def test_translated_slab_retention_matches_actual_napari_render():
    viewer = ViewerModel()
    data = np.arange(6 * 8 * 10, dtype="uint16").reshape(6, 8, 10)
    layer = viewer.add_image(data, scale=(2, 3, 4), translate=(10, 20, 30))
    dims = viewer.dims
    dims.ndisplay = 3
    legacy = _LegacyCropSliceLayer(layer, layer._data_slice)
    dims.ndisplay = 2
    dims.point = (10, 20, 30)
    dims.margin_right = (4, 0, 0)
    layer.projection_mode = "max"

    slicing = getattr(layer, "_slicing_state", None)
    if hasattr(slicing, "_slice_indices") and hasattr(slicing, "make_slice_input"):
        sample = slicing._slice_indices(slicing.make_slice_input(dims), dims)
    else:
        sample = layer._make_slice_input(dims).data_slice(layer._data_to_world.inverse)
    measured = (sample.point[0], sample.margin_left[0], sample.margin_right[0])
    raw = layer._slice.image.raw
    if np.isclose(sample.margin_left[0], 0):
        np.testing.assert_allclose(measured, (0, 0, 2))
        assert not layer._slice.empty
        np.testing.assert_array_equal(raw, data[:3].max(axis=0))
        retained = True
    else:
        # Legacy napari translates slab widths too and renders an empty slice.
        np.testing.assert_allclose(measured, (0, -5, -3))
        assert layer._slice.empty
        np.testing.assert_array_equal(raw, np.zeros((1, 1), dtype=data.dtype))
        retained = False
    assert crop_intersects_slice(layer, dims, ((0, 2, 5),)) is retained
    assert crop_intersects_slice(legacy, dims, ((0, 2, 5),)) is retained
    assert legacy.factory_calls == 1 and legacy.cached_reads == 0


class _RecordingOutline:
    """Record the drawing requests; use napari's real layer and dims models."""

    available = True

    def __init__(self, viewer):
        self.box = None
        self.rectangle = None

    def set_box(self, layer, world_corners, *, line_width=2.0):
        self.rectangle = None
        self.box = (layer, np.array(world_corners, copy=True))
        return True

    def set_rectangle(self, layer, world_corners, *, retained=True, line_width=2.0):
        self.box = None
        self.rectangle = (layer, np.array(world_corners, copy=True), bool(retained))
        return True

    def clear(self):
        self.box = None
        self.rectangle = None

    close = clear


@pytest.fixture
def scene(qtbot, monkeypatch, request):
    monkeypatch.setattr(_widget, "CropCanvasOutline", _RecordingOutline)
    monkeypatch.setattr(_widget.VippWidget, "_crop_uses_bounds_preview", lambda _: True)
    data = np.arange(6 * 8 * 10, dtype=np.uint16).reshape(6, 8, 10)
    viewer = ViewerModel()
    viewer.add_image(
        data,
        name="Calibrated volume",
        metadata={"axes": "ZYX"},
        scale=(2, 3, 4),
        translate=getattr(request, "param", (10, 20, 30)),
    )
    widget = _widget.VippWidget(viewer)
    widget._should_run_pipeline_in_background = lambda *args, **kwargs: False
    qtbot.addWidget(widget)
    crop = widget.add_node_from_palette("crop_stack")
    widget._connect_nodes("input", crop.id)
    widget.pipeline.add_output_tunnel("Cropped volume", crop.id, 0)
    widget._sync_port_tunnels()
    widget._debounce_timer.stop()
    return widget, crop.id, data


def _world_box(bounds, order=(0, 1, 2)):
    # Source samples have integer centres, and Bounds stop is exclusive.
    scales, origins = (2, 3, 4), (10, 20, 30)
    intervals = [
        tuple(origins[a] + (n - 0.5) * scales[a] for n in bounds[a]) for a in order
    ]
    return set(product(*intervals))


def _corners(outline):
    return {tuple(row) for row in outline.box[1]}


def _rect(outline):
    return {tuple(row) for row in outline.rectangle[1]}


def test_live_draft_moves_box_not_pixels_or_scientific_result(scene, qtbot):
    widget, crop, data = scene
    original = data.copy()
    before_hash = scientific_workflow_hash(serialize_workflow(widget.pipeline))
    output = widget.pipeline.outputs[crop]
    assert output is not None
    widget.viewer.dims.ndisplay = 3  # Crop was already selected in 2D.
    outline = widget._crop_canvas_outline
    qtbot.waitUntil(lambda: outline.box is not None)
    context = outline.box[0]
    assert context.visible and context.metadata["vipp_crop_context"] == crop
    assert _corners(outline) == _world_box(((0, 6), (0, 8), (0, 10)))
    np.testing.assert_array_equal(context.data, original)
    # Observer on the layer itself: no swaps or reuploads during the drag.
    uploads = []
    context.events.data.connect(lambda event: uploads.append(event))
    top = widget._parameter_widgets["top"].slider
    top.setSliderDown(True)
    for value in (1, 3):
        top.setValue(value)
        assert outline.box[0] is context
        assert _corners(outline) == _world_box(((0, 6), (value, 8), (0, 10)))
        assert widget.pipeline.outputs[crop] is output
        assert widget.pipeline.nodes[crop].params["top"] == 0
        assert (
            scientific_workflow_hash(serialize_workflow(widget.pipeline)) == before_hash
        )
    assert not uploads
    assert not widget._owned_crop_presentation_layers()
    top.setSliderDown(False)
    widget._debounce_timer.stop()
    widget.run_pipeline()
    np.testing.assert_array_equal(widget.pipeline.outputs[crop], original[:, 3:, :])
    np.testing.assert_array_equal(context.data, original)
    assert outline.box[0] is context
    assert _corners(outline) == _world_box(((0, 6), (3, 8), (0, 10)))
    np.testing.assert_array_equal(data, original)

    widget.viewer.dims.ndisplay = 2
    qtbot.waitUntil(lambda: outline.rectangle is not None)
    assert outline.box is None and outline.rectangle[2]
    assert outline.rectangle[0] is context
    np.testing.assert_array_equal(context.data, original)
    assert _rect(outline) == _world_box(((0, 6), (3, 8), (0, 10)), (1, 2))

    # Widen the crop against the ORIGINAL input while the slider is held.
    top = widget._parameter_widgets["top"].slider
    top.setSliderDown(True)
    top.setValue(1)
    # In particular, allow queued view/inspector work and the 350 ms numeric
    # idle timer to run after the mode switch. A held slider is still a draft.
    qtbot.wait(550)
    assert top.isSliderDown()
    assert _rect(outline) == _world_box(((0, 6), (1, 8), (0, 10)), (1, 2))
    assert widget.pipeline.nodes[crop].params["top"] == 3
    np.testing.assert_array_equal(widget.pipeline.outputs[crop], original[:, 3:, :])
    assert not uploads
    top.setSliderDown(False)
    widget._debounce_timer.stop()
    widget.run_pipeline()
    np.testing.assert_array_equal(widget.pipeline.outputs[crop], original[:, 1:, :])

    widget.viewer.dims.ndisplay = 3
    qtbot.waitUntil(lambda: outline.box is not None)
    assert outline.rectangle is None and outline.box[0] is context
    assert _corners(outline) == _world_box(((0, 6), (1, 8), (0, 10)))


@pytest.mark.parametrize("scene", [(0, 20, 30)], indirect=True)
def test_excluded_slice_and_projected_slab_have_different_retention(scene, qtbot):
    widget, crop, data = scene
    outline, dims = widget._crop_canvas_outline, widget.viewer.dims
    qtbot.waitUntil(lambda: outline.rectangle is not None)
    source_layer = outline.rectangle[0]
    # Commit Z [2, 5), then show slice zero (world Z=0, scale=2).
    for parameter, value in (("z_start", 2), ("z_end", 1)):
        widget._parameter_widgets[parameter].slider.setValue(value)
    widget._commit_crop_draft(schedule_run=False)
    widget.run_pipeline()
    source_layer.projection_mode = "none"
    dims.set_point(0, 0)
    qtbot.waitUntil(lambda: not outline.rectangle[2])
    np.testing.assert_array_equal(source_layer.data, data)
    np.testing.assert_array_equal(widget.pipeline.outputs[crop], data[2:5])

    # VIPP's graph thumbnail selector is not the canvas projection selector.
    widget.preview_mode_combo.setCurrentText("MIP")
    assert str(source_layer.projection_mode) == "none"
    assert not outline.rectangle[2]
    # Same centre (Z=0), but the 2D projection includes retained Z=2.
    dims.margin_right = (4.0, 0.0, 0.0)
    assert not outline.rectangle[2]
    source_layer.projection_mode = "max"
    qtbot.waitUntil(lambda: outline.rectangle[2])
    np.testing.assert_array_equal(source_layer._slice.image.raw, data[0:3].max(axis=0))
    assert _rect(outline) == _world_box(((2, 5), (0, 8), (0, 10)), (1, 2))
    # 1.6 rounds to 2, but this noninteger upper endpoint is exclusive
    # in napari: only planes 0 and 1 contribute to this actual projection.
    dims.margin_right = (3.2, 0.0, 0.0)
    qtbot.waitUntil(lambda: not outline.rectangle[2])
    np.testing.assert_array_equal(source_layer._slice.image.raw, data[0:2].max(axis=0))
    dims.set_point(0, 4)
    qtbot.waitUntil(lambda: outline.rectangle[2])
    source_layer.projection_mode = "none"
    np.testing.assert_array_equal(widget.pipeline.outputs[crop], data[2:5])


def test_slider_press_value_change_before_pressed_is_a_single_draft(scene, qtbot):
    widget, crop, data = scene
    outline = widget._crop_canvas_outline
    qtbot.waitUntil(lambda: outline.rectangle is not None)
    widget._parameter_widgets["top"].slider.setValue(2)
    widget._commit_crop_draft(schedule_run=False)
    widget.run_pipeline()
    top = widget._parameter_widgets["top"].slider
    output = widget.pipeline.outputs[crop]
    undo_count = len(widget._undo_stack)
    before = scientific_workflow_hash(serialize_workflow(widget.pipeline))
    # On Windows, pressing a nonzero slider can move it by a pixel *before*
    # sliderPressed arrives. Both signals belong to the same physical gesture.
    top.setValue(3)
    top.setSliderDown(True)
    top.setValue(5)
    qtbot.wait(550)
    assert top.isSliderDown()
    assert widget.pipeline.nodes[crop].params["top"] == 2
    assert len(widget._undo_stack) == undo_count
    assert scientific_workflow_hash(serialize_workflow(widget.pipeline)) == before
    assert _rect(outline) == _world_box(((0, 6), (5, 8), (0, 10)), (1, 2))
    assert widget.pipeline.outputs[crop] is output
    top.setSliderDown(False)
    widget._debounce_timer.stop()
    widget.run_pipeline()
    assert len(widget._undo_stack) == undo_count + 1
    np.testing.assert_array_equal(widget.pipeline.outputs[crop], data[:, 5:])


def test_orthogonal_2d_plane_and_excluded_non_z_axis(scene, qtbot):
    widget, crop, _data = scene
    outline, dims = widget._crop_canvas_outline, widget.viewer.dims
    qtbot.waitUntil(lambda: outline.rectangle is not None)
    for parameter, value in (("left", 2), ("top", 1), ("z_start", 1)):
        widget._parameter_widgets[parameter].slider.setValue(value)
    widget._commit_crop_draft(schedule_run=False)
    widget.run_pipeline()
    dims.order = (2, 0, 1)  # ZY plane: the X coordinate controls retention.
    dims.set_point(2, 30)  # X=0 excluded by left=2.
    expected = _world_box(((1, 6), (1, 8), (2, 10)), (0, 1))
    qtbot.waitUntil(
        lambda: (
            outline.rectangle is not None
            and _rect(outline) == expected
            and not outline.rectangle[2]
        )
    )
    dims.set_point(2, 38)
    qtbot.waitUntil(lambda: outline.rectangle[2])


@pytest.mark.parametrize("axes", ["YX", "QYX"])
def test_xy_outline_does_not_require_or_invent_a_depth_axis(qtbot, monkeypatch, axes):
    monkeypatch.setattr(_widget, "CropCanvasOutline", _RecordingOutline)
    monkeypatch.setattr(_widget.VippWidget, "_crop_uses_bounds_preview", lambda _: True)
    dims = (8, 10) if axes == "YX" else (2, 8, 10)
    data = np.arange(np.prod(dims), dtype="uint16").reshape(dims)
    viewer = ViewerModel()
    viewer.add_image(data, name="Planar samples", metadata={"axes": axes})
    widget = _widget.VippWidget(viewer)
    widget._should_run_pipeline_in_background = lambda *args, **kwargs: False
    qtbot.addWidget(widget)
    crop = widget.add_node_from_palette("crop_stack")
    widget._connect_nodes("input", crop.id)
    widget._debounce_timer.stop()
    outline = widget._crop_canvas_outline
    qtbot.waitUntil(lambda: outline.rectangle is not None)
    right = widget._parameter_widgets["right"].slider
    right.setSliderDown(True)
    right.setValue(2)
    assert _rect(outline) == set(product((-0.5, 7.5), (-0.5, 7.5)))
    assert outline.rectangle[2]
    np.testing.assert_array_equal(outline.rectangle[0].data, data)
    np.testing.assert_array_equal(widget.pipeline.outputs[crop.id], data)
    if axes == "QYX":
        viewer.dims.set_current_step(0, 1)
        # This axis has no depth interpretation: all its planes remain valid.
        assert outline.rectangle[2]
    right.setSliderDown(False)
    widget._debounce_timer.stop()
    widget.run_pipeline()
    np.testing.assert_array_equal(widget.pipeline.outputs[crop.id], data[..., :-2])


def test_projected_box_updates_with_display_order_and_retires_on_bypass(scene, qtbot):
    widget, crop, data = scene
    widget.viewer.dims.ndisplay = 3
    outline = widget._crop_canvas_outline
    qtbot.waitUntil(lambda: outline.box is not None)
    widget.viewer.dims.order = (2, 0, 1)
    expected = _world_box(((0, 6), (0, 8), (0, 10)), order=(2, 0, 1))
    qtbot.waitUntil(lambda: outline.box is not None and _corners(outline) == expected)
    assert widget._set_node_bypassed(crop, True)
    qtbot.waitUntil(
        lambda: (
            outline.box is None
            and any(
                layer.visible and not layer.metadata.get("vipp_crop_context")
                for layer in widget._owned_inspect_layers()
            )
        )
    )
    np.testing.assert_array_equal(widget.pipeline.outputs[crop], data)
    assert widget._set_node_bypassed(crop, False)
    qtbot.waitUntil(lambda: outline.box is not None)
    widget._select_node("input")
    qtbot.waitUntil(
        lambda: (
            outline.box is None
            and any(
                layer.visible
                and layer.metadata.get("node_id") == "input"
                and not layer.metadata.get("vipp_crop_context")
                for layer in widget._owned_inspect_layers()
            )
        )
    )


@pytest.mark.parametrize("kind", ["label image", "RGB image", "RGBA image"])
def test_cold_crop_uses_input_kind_and_calibration(scene, qtbot, kind):
    widget, crop, data = scene
    state = widget.pipeline.input_state_for_node(crop)
    if kind == "label image":
        received = data
        state = replace(state, kind=kind)
    else:
        n = 3 if kind == "RGB image" else 4
        received = np.stack(
            [((data + i * 31) % 255).astype("uint8") for i in range(n)], axis=-1
        )
        state = replace(
            state,
            kind=kind,
            shape=received.shape,
            dtype=received.dtype.name,
            axes=(*state.axes, AxisMetadata("rgb" if n == 3 else "rgba", "channel")),
        )
    widget.pipeline.outputs["input"] = received
    widget.pipeline.node_outputs["input"] = [received]
    widget.pipeline.output_states["input"] = state
    widget.pipeline.node_output_states["input"] = [state]
    widget.pipeline._clear_cached_output(crop)
    widget.viewer.dims.ndisplay = 3
    outline = widget._crop_canvas_outline
    qtbot.waitUntil(
        lambda: (
            outline.box is not None
            and (
                outline.box[0]._type_string == "labels"
                if kind == "label image"
                else bool(outline.box[0].metadata.get("display_rgb_as_channels"))
            )
        )
    )
    layers = widget._owned_inspect_layers()
    assert all(layer.metadata["vipp_crop_context"] == crop for layer in layers)
    if kind == "label image":
        assert len(layers) == 1
        assert layers[0]._type_string == "labels"
        np.testing.assert_array_equal(layers[0].data, received)
    else:
        assert len(layers) == 3
        for layer in layers:
            channel = layer.metadata["display_rgb_channel_index"]
            np.testing.assert_array_equal(layer.data, received[..., channel])
            assert layer.metadata["display_rgb_as_channels"] and not layer.rgb
            assert tuple(layer.scale) == (2, 3, 4)
            assert tuple(layer.translate) == (10, 20, 30)
        assert {layer.metadata["display_rgb_channel_index"] for layer in layers} == {
            0,
            1,
            2,
        }
    widget._parameter_widgets["right"].slider.setValue(2)
    assert outline.box[0] in layers
    assert _corners(outline) == _world_box(((0, 6), (0, 8), (0, 8)))
    assert widget.pipeline.outputs[crop] is None
    assert not widget._owned_crop_presentation_layers()


def test_exact_window_context_draws_full_source_world_coordinates(scene, qtbot):
    widget, crop, data = scene
    bounds = ((1, 5), (2, 7), (3, 9))
    state = widget.pipeline.input_state_for_node(crop)
    window = np.array(data[1:5, 2:7, 3:9], copy=True)
    window.setflags(write=False)
    window_state = replace(
        state,
        shape=window.shape,
        axes=tuple(
            replace(axis, translation=axis.translation + start * axis.scale)
            for axis, (start, _) in zip(state.axes, bounds, strict=True)
        ),
    )
    exact = ExactSourceWindowData(
        window,
        window_state,
        SourceWindowIdentity(
            source_uri="C:/fixture/volume.ome.zarr",
            source_format="ome-zarr-0.5",
            series_index=0,
            item_key=".",
            reader_key="ome-zarr",
            reader_version="test",
            source_shape=data.shape,
            source_dtype="uint16",
            axis_names=("z", "y", "x"),
            bounds=bounds,
        ),
    )
    widget.pipeline.outputs["input"] = exact
    widget.pipeline.node_outputs["input"] = [exact]
    for name, value in dict(
        z_start=1, z_end=1, top=2, bottom=1, left=3, right=1
    ).items():
        widget.pipeline.set_param(crop, name, value)

    widget.viewer.dims.ndisplay = 3
    outline = widget._crop_canvas_outline
    qtbot.waitUntil(
        lambda: outline.box is not None and outline.box[0].data.shape == window.shape
    )
    layer = outline.box[0]
    assert layer.metadata["vipp_crop_context"] == crop
    np.testing.assert_array_equal(layer.data, window)
    np.testing.assert_allclose(layer.translate, [12, 26, 42])
    assert _corners(outline) == _world_box(bounds)
    assert widget.pipeline.input_data_for_node(crop) is exact
    widget.viewer.dims.ndisplay = 2
    widget.viewer.dims.set_point(0, 14)
    qtbot.waitUntil(lambda: outline.rectangle is not None)
    assert outline.rectangle[0] is layer and outline.rectangle[2]
    assert _rect(outline) == _world_box(bounds, (1, 2))
    np.testing.assert_array_equal(layer.data, window)
