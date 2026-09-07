from itertools import product

import numpy as np
import pytest
from skimage.morphology import convex_hull_image

from napari_vipp.core import convex_hull as hull_module
from napari_vipp.core.compute_specs import compute_specs_for
from napari_vipp.core.export import export_pipeline_to_python
from napari_vipp.core.metadata import AxisMetadata, image_state_from_array
from napari_vipp.core.operations import convex_hull
from napari_vipp.core.pipeline import NODE_LIBRARY_BY_ID, PrototypePipeline
from napari_vipp.core.progress import OperationCancelled, ProgressContext
from napari_vipp.core.workflow import deserialize_workflow, serialize_workflow


def _corners(ndim):
    mask = np.zeros((9,) * ndim, dtype=bool)
    for point in product((2, 6), repeat=ndim):
        mask[point] = True
    return mask


def _mode(ndim):
    return "2D YX" if ndim == 2 else "3D ZYX"


@pytest.mark.parametrize("ndim", [2, 3])
def test_box_hull_is_exact_and_preserves_readonly_input(ndim):
    mask = _corners(ndim)
    before = mask.copy()
    mask.setflags(write=False)
    expected = np.zeros_like(mask)
    expected[(slice(2, 7),) * ndim] = True
    actual = convex_hull(mask, spatial_mode=_mode(ndim))
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(mask, before)
    assert actual.dtype == np.dtype(bool)
    assert not np.shares_memory(actual, mask)


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("seed", range(5))
def test_matches_scikit_image_for_nondegenerate_masks(ndim, seed):
    mask = np.random.default_rng(seed).random((11,) * ndim) > 0.93
    # Guarantee full rank, including for a sparse random realization.
    mask[(1,) * ndim] = True
    for axis in range(ndim):
        point = [1] * ndim
        point[axis] = 8
        mask[tuple(point)] = True
    actual = convex_hull(mask, spatial_mode=_mode(ndim))
    np.testing.assert_array_equal(actual, convex_hull_image(mask))
    assert np.all(actual[mask])


@pytest.mark.parametrize("ndim", [2, 3])
def test_triangle_or_tetrahedron_matches_analytical_integer_simplex(ndim):
    mask = np.zeros((7,) * ndim, dtype=bool)
    mask[(1,) * ndim] = True
    for axis in range(ndim):
        point = [1] * ndim
        point[axis] = 5
        mask[tuple(point)] = True
    grid = np.indices(mask.shape)
    expected = np.all(grid >= 1, axis=0) & (np.sum(grid - 1, axis=0) <= 4)
    np.testing.assert_array_equal(convex_hull(mask, spatial_mode=_mode(ndim)), expected)


@pytest.mark.parametrize("shape", [(0, 5), (0, 4, 5), (3, 0, 4), (1, 1), (1, 1, 1)])
@pytest.mark.parametrize("value", [False, True])
def test_empty_and_singleton_shapes(shape, value):
    mask = np.full(shape, value, dtype=bool)
    result = convex_hull(mask, spatial_mode=_mode(len(shape)))
    np.testing.assert_array_equal(result, mask)
    assert result is not mask


@pytest.mark.parametrize("kind", ["point", "line", "plane", "oblique_plane"])
def test_degenerate_3d_hulls_keep_and_fill_foreground(kind):
    mask = np.zeros((9, 9, 9), dtype=bool)
    if kind == "point":
        mask[4, 4, 4] = True
        expected = mask.copy()
    elif kind == "line":
        mask[1, 1, 1] = mask[7, 7, 7] = True
        expected = np.zeros_like(mask)
        expected[np.arange(1, 8), np.arange(1, 8), np.arange(1, 8)] = True
    elif kind == "plane":
        for y, x in product((2, 6), repeat=2):
            mask[4, y, x] = True
        expected = np.zeros_like(mask)
        expected[4, 2:7, 2:7] = True
    else:
        for z, x in product((2, 6), repeat=2):
            mask[z, z, x] = True
        expected = np.zeros_like(mask)
        for z in range(2, 7):
            expected[z, z, 2:7] = True
    np.testing.assert_array_equal(convex_hull(mask, spatial_mode="3D ZYX"), expected)


def test_slice_and_volume_hulls_are_distinct_and_do_not_join_time_or_channels():
    volume = _corners(3)
    sliced = convex_hull(volume, spatial_mode="2D YX")
    assert not sliced[4].any()
    assert convex_hull(volume, spatial_mode="3D ZYX")[4, 4, 4]
    mask = np.zeros((2, 2, *volume.shape), dtype=bool)
    mask[0, 1] = volume
    result = convex_hull(mask, spatial_mode="3D ZYX")
    np.testing.assert_array_equal(
        result[0, 1], convex_hull(volume, spatial_mode="3D ZYX")
    )
    assert not result[1].any()
    assert not result[0, 0].any()


@pytest.mark.parametrize("value", [0, 1, 255, -1, 2**63, np.nan, np.inf, 1j])
def test_non_boolean_input_is_not_silently_thresholded(value):
    with pytest.raises(ValueError, match="Boolean mask.*threshold"):
        convex_hull(np.full((3, 3), value))


@pytest.mark.parametrize(
    ("shape", "kwargs", "message"),
    [
        ((3, 3, 3), {}, "explicit axis semantics"),
        ((3, 3), {"spatial_mode": "3D ZYX"}, "cannot be applied"),
        ((3, 3), {"spatial_mode": "guess"}, "Spatial mode"),
        ((3,), {}, "2D YX"),
        ((3, 3, 3), {"resolved_spatial_ndim": 1}, "2D YX"),
    ],
)
def test_invalid_or_ambiguous_spatial_requests_fail(shape, kwargs, message):
    with pytest.raises(ValueError, match=message):
        convex_hull(np.zeros(shape, dtype=bool), **kwargs)


def test_chunked_rasterization_matches_single_chunk_for_strided_input(monkeypatch):
    mask = _corners(3)[::-1, :, ::-1]
    expected = convex_hull(mask, spatial_mode="3D ZYX")
    monkeypatch.setattr(hull_module, "_RASTER_CHUNK_SIZE", 7)
    np.testing.assert_array_equal(convex_hull(mask, spatial_mode="3D ZYX"), expected)


def test_cancel_during_rasterization_does_not_publish_or_mutate_input(monkeypatch):
    mask = _corners(3)
    before = mask.copy()
    updates = []
    monkeypatch.setattr(hull_module, "_RASTER_CHUNK_SIZE", 7)
    progress = ProgressContext(
        reporter=updates.append,
        cancelled=lambda: bool(updates and updates[-1].current > 10),
    )
    with pytest.raises(OperationCancelled):
        convex_hull(mask, spatial_mode="3D ZYX", progress=progress)
    np.testing.assert_array_equal(mask, before)


def test_pre_cancelled_empty_input_still_observes_cancellation():
    with pytest.raises(OperationCancelled):
        convex_hull(
            np.zeros((0, 5), dtype=bool),
            progress=ProgressContext(cancelled=lambda: True),
        )


def test_geometric_failure_is_not_returned_as_an_empty_mask(monkeypatch):
    from scipy.spatial import QhullError

    def fail(*args, **kwargs):
        raise QhullError("forced geometry error")

    monkeypatch.setattr(hull_module, "ConvexHull", fail)
    with pytest.raises(ValueError, match="No hull was produced"):
        convex_hull(_corners(3), spatial_mode="3D ZYX")


def _pipeline(mode="Auto from axes"):
    pipeline = PrototypePipeline()
    threshold = pipeline.add_node("binary_threshold")
    hull = pipeline.add_node("convex_hull")
    pipeline.set_param(threshold.id, "threshold", 0.5)
    pipeline.set_param(hull.id, "spatial_mode", mode)
    pipeline.connect("input", threshold.id)
    pipeline.connect(threshold.id, hull.id)
    return pipeline, hull.id


@pytest.mark.parametrize(
    ("axes", "shape", "ndim"),
    [
        ("YX", (9, 9), 2),
        ("ZYX", (9, 9, 9), 3),
        ("TZYX", (2, 9, 9, 9), 3),
        ("CYX", (2, 9, 9), 2),
    ],
)
def test_pipeline_auto_axes_grid_and_history(axes, shape, ndim):
    data = np.zeros(shape, dtype=np.float32)
    data[...] = _corners(ndim)
    pipeline, node_id = _pipeline()
    outputs = pipeline.run(data, input_metadata={"axes": axes})
    state = pipeline.output_states[node_id]
    upstream = pipeline.output_states["binary_threshold_1"]
    assert state.axes == upstream.axes
    assert state.kind == "binary mask"
    assert pipeline.nodes[node_id].params["resolved_spatial_ndim"] == ndim
    assert f"{ndim}D" in state.history[-1]
    assert "half-pixel axis offsets" in state.history[-1]
    np.testing.assert_array_equal(
        outputs[node_id], convex_hull(data.astype(bool), spatial_mode=_mode(ndim))
    )


@pytest.mark.parametrize("mode", ["Auto from axes", "2D YX", "3D ZYX"])
def test_workflow_roundtrip_and_export_execution(mode):
    pipeline, node_id = _pipeline(mode)
    data = _corners(3).astype(np.float32)
    expected = pipeline.run(data, input_metadata={"axes": "ZYX"})[node_id]
    payload = deserialize_workflow(serialize_workflow(pipeline))
    restored = PrototypePipeline()
    restored.restore_graph(payload["nodes"], payload["connections"])
    np.testing.assert_array_equal(
        restored.run(data, input_metadata={"axes": "ZYX"})[node_id], expected
    )
    namespace = {"__name__": "exported_pipeline"}
    exec(compile(export_pipeline_to_python(pipeline), "<exported>", "exec"), namespace)
    np.testing.assert_array_equal(
        namespace["run_pipeline"](data, input_metadata={"axes": "ZYX"})[node_id],
        expected,
    )


def test_node_is_in_morphology_with_an_authoritative_cpu_provider():
    spec = NODE_LIBRARY_BY_ID["convex_hull"]
    assert spec.title == "Convex Hull"
    assert spec.category == "Morphology"
    assert spec.input_type == spec.output_type == "mask"
    assert spec.parameters[0].choices == ("Auto from axes", "2D YX", "3D ZYX")
    assert len(compute_specs_for("convex_hull")) == 1


def test_original_anisotropic_calibration_and_origin_are_preserved():
    data = _corners(3).astype(np.float32)
    axes = tuple(
        AxisMetadata(name, "space", "micrometer", scale, translation)
        for name, scale, translation in zip(
            "zyx", (2.0, 0.3, 0.2), (5.0, 8.0, -2.0), strict=True
        )
    )
    state = image_state_from_array(data, axes=axes)
    pipeline, node_id = _pipeline()
    pipeline.run(data, input_metadata={"vipp_image_state": state.to_dict()})
    assert pipeline.output_states[node_id].axes == state.axes


@pytest.mark.parametrize("axes", [None, "YZX", "CYX"])
def test_pipeline_rejects_ambiguous_or_non_zyx_volume_axes(axes):
    pipeline, _ = _pipeline("Auto from axes" if axes is None else "3D ZYX")
    with pytest.raises(ValueError):
        pipeline.run(
            _corners(3).astype(np.float32),
            input_metadata={"axes": axes} if axes else None,
        )


def test_progress_is_monotonic_across_multiple_volumes_and_empty_blocks():
    mask = np.zeros((3, 9, 9, 9), dtype=bool)
    mask[1] = _corners(3)
    updates = []
    convex_hull(
        mask, spatial_mode="3D ZYX", progress=ProgressContext(reporter=updates.append)
    )
    assert [u.current for u in updates] == sorted(u.current for u in updates)
    assert updates[-1].current == updates[-1].total == 300


@pytest.mark.parametrize("ndim", [2, 3])
def test_inspector_explains_hull_scope_and_exposes_volume_choice(qtbot, ndim):
    from qtpy.QtWidgets import QComboBox

    from napari_vipp._tests.test_ui_inspector_widget_integration import (
        _publish_array_output,
        _select,
        _widget,
    )
    from napari_vipp._tests.test_widget import _palette_item

    mask = _corners(ndim)
    axes = "YX" if ndim == 2 else "ZYX"
    widget = _widget(qtbot, mask.astype(np.float32), axes=axes)
    threshold = widget.add_node_from_palette("binary_threshold")
    node = widget.add_node_from_palette("convex_hull")
    widget._connect_nodes("input", threshold.id)
    _publish_array_output(widget, threshold.id, mask, axes=axes)
    widget._connect_nodes(threshold.id, node.id)
    _select(widget, node.id)
    assert _palette_item(widget, "convex_hull") is not None
    assert "Separate objects" in widget.selected_title.toolTip()
    assert "One hull" in widget._parameter_widgets["operation_notice"].text()
    if ndim == 3:
        control = widget._parameter_widgets["spatial_mode"].combo
        assert isinstance(control, QComboBox)
        assert "3D ZYX" in [control.itemText(i) for i in range(control.count())]
