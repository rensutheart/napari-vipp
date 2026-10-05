from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest
from napari.components import ViewerModel
from scipy.spatial.transform import Rotation

from napari_vipp.ui.napari_compat import viewer_camera
from napari_vipp.ui.workflow_view import (
    ViewerState,
    capture_viewer_state,
    fit_layer_view,
    restore_viewer_state,
)


def _fake_viewer(ndim=3):
    dims = SimpleNamespace(
        ndim=ndim,
        order=tuple(range(ndim)),
        ndisplay=min(ndim, 3),
        point=(0.0,) * ndim,
        range=tuple((0.0, 10.0, 1.0) for _ in range(ndim)),
    )
    camera = SimpleNamespace(
        center=(1.0, 2.0, 3.0),
        zoom=4.0,
        angles=(15.0, 25.0, 35.0),
        perspective=12.0,
    )
    return SimpleNamespace(dims=dims, camera=camera)


def test_view_snapshot_is_immutable_and_uses_scene_camera():
    viewer = _fake_viewer()
    viewer.scene = SimpleNamespace(
        camera=SimpleNamespace(
            center=(4, 5, 6), zoom=8, angles=(0, 0, 0), perspective=0
        )
    )
    state = capture_viewer_state(viewer)
    assert state.center == (4.0, 5.0, 6.0)
    assert state.zoom == 8.0
    viewer.scene.camera.center = (7, 8, 9)
    assert state.center == (4.0, 5.0, 6.0)
    with pytest.raises(FrozenInstanceError):
        state.zoom = 2


def test_missing_or_minimal_viewers_are_tolerated():
    assert capture_viewer_state(SimpleNamespace()) is None
    assert not restore_viewer_state(SimpleNamespace(), None)
    assert not restore_viewer_state(SimpleNamespace(), ViewerState())
    viewer = SimpleNamespace(dims=SimpleNamespace(ndisplay=2))
    assert capture_viewer_state(viewer) == ViewerState(ndisplay=2)
    assert restore_viewer_state(viewer, ViewerState(ndisplay=3))
    assert viewer.dims.ndisplay == 3
    assert not fit_layer_view(viewer, SimpleNamespace())


@pytest.mark.parametrize(
    ("incoming_ndim", "saved_order", "expected"),
    [(2, (2, 0, 1), (1, 0)), (5, (2, 0, 1), (0, 1, 4, 2, 3))],
)
def test_restore_adapts_dimension_order_by_right_alignment(
    incoming_ndim, saved_order, expected
):
    viewer = _fake_viewer(incoming_ndim)
    state = ViewerState(order=saved_order, ndisplay=3, point=(-5, 3, 50))
    assert restore_viewer_state(viewer, state)
    assert viewer.dims.order == expected
    assert viewer.dims.ndisplay == min(incoming_ndim, 3)
    assert viewer.dims.point == ((3, 10) if incoming_ndim == 2 else (0, 0, 0, 3, 10))


def test_restore_camera_runs_after_dim_callbacks_and_uses_world_points():
    viewer = _fake_viewer()
    camera = viewer.camera

    class CallbackDims:
        ndim = 3
        order = (0, 1, 2)
        point = (100.0, 200.0, 300.0)
        range = ((100, 110, 2), (200, 210, 2), (300, 310, 2))
        _ndisplay = 2

        @property
        def ndisplay(self):
            return self._ndisplay

        @ndisplay.setter
        def ndisplay(self, value):
            self._ndisplay = value
            camera.center = (999, 999, 999)
            camera.zoom = 999

    viewer.dims = CallbackDims()
    state = ViewerState(
        order=(0, 1, 2),
        ndisplay=3,
        point=(102, 212, 299),
        center=(1, 2, 3),
        zoom=2,
        angles=(4, 5, 6),
        perspective=7,
    )
    assert restore_viewer_state(viewer, state)
    assert viewer.dims.point == (102, 210, 300)
    assert capture_viewer_state(viewer) == ViewerState(
        order=(0, 1, 2),
        ndisplay=3,
        point=(102, 210, 300),
        center=(1, 2, 3),
        zoom=2,
        angles=(4, 5, 6),
        perspective=7,
    )


def test_real_viewer_state_roundtrip_restores_all_requested_fields():
    viewer = ViewerModel()
    viewer.add_image(np.zeros((3, 4, 5)), rgb=False)
    viewer.dims.ndisplay = 3
    viewer.dims.order = (1, 0, 2)
    viewer.dims.point = (1, 2, 3)
    camera = viewer_camera(viewer)
    camera.center, camera.zoom = (2, 3, 4), 5
    camera.angles, camera.perspective = (11, 23, 37), 8
    state = capture_viewer_state(viewer)
    viewer.dims.ndisplay = 2
    camera.center, camera.zoom = (0, 0, 0), 1
    camera.angles, camera.perspective = (0, 0, 0), 0
    assert restore_viewer_state(viewer, state)
    assert capture_viewer_state(viewer) == state


def test_fit_uses_only_target_calibration_and_preserves_layer_buffers():
    viewer = ViewerModel(ndisplay=3)
    unrelated = viewer.add_image(np.zeros((20, 100, 200)), rgb=False, visible=False)
    target = viewer.add_image(
        np.zeros((4, 6, 8)), rgb=False, scale=(2, 3, 4), translate=(10, 20, 30)
    )
    camera = viewer_camera(viewer)
    camera.angles, camera.perspective = (0, 0, 0), 12
    original_data = target.data
    original_extents = np.array(viewer.layers.extent.world, copy=True)
    assert fit_layer_view(viewer, target)
    assert camera.center == pytest.approx((13, 27.5, 44))
    # Full voxel-edge lengths are 8, 18, 32; zero angles show Y/X.
    assert camera.zoom == pytest.approx(0.95 * min(800 / 18, 600 / 32))
    assert camera.angles == (0, 0, 0)
    assert camera.perspective == 12
    assert viewer.dims.ndisplay == 3
    assert target.data is original_data
    assert unrelated.visible is False
    np.testing.assert_array_equal(viewer.layers.extent.world, original_extents)


def test_rotated_3d_fit_matches_projection_of_affine_pixel_edge_box():
    viewer = ViewerModel(ndisplay=3)
    target = viewer.add_image(
        np.zeros((4, 6, 8)), rgb=False, scale=(2, 3, 4), rotate=(17, 23, 31)
    )
    camera = viewer_camera(viewer)
    camera.angles = (19, 29, 41)
    angles = camera.angles
    transform = target.data_to_world
    corners = np.array(
        [
            transform((z, y, x))
            for z in (-0.5, 3.5)
            for y in (-0.5, 5.5)
            for x in (-0.5, 7.5)
        ]
    )
    size = np.max(corners, axis=0) - np.min(corners, axis=0)
    projected = np.array(
        [
            np.dot(np.abs(camera.up_direction), size),
            np.dot(np.abs(np.cross(camera.view_direction, camera.up_direction)), size),
        ]
    )
    assert fit_layer_view(viewer, target)
    assert camera.center == pytest.approx(np.mean([corners.min(0), corners.max(0)], 0))
    assert camera.zoom == pytest.approx(0.95 * min(np.array((800, 600)) / projected))
    np.testing.assert_allclose(
        Rotation.from_euler("xyz", camera.angles, degrees=True).as_matrix(),
        Rotation.from_euler("xyz", angles, degrees=True).as_matrix(),
    )


def test_fit_accounts_for_scene_unit_conversion():
    viewer = ViewerModel(ndisplay=3)
    viewer.add_image(np.zeros((4, 6, 8)), rgb=False, units="micrometer")
    target = viewer.add_image(
        np.zeros((4, 6, 8)),
        rgb=False,
        units="nanometer",
        scale=(2000, 3000, 4000),
        translate=(10000, 20000, 30000),
    )
    factor = (1 * target.units[0]).to(viewer.layers.units[0]).magnitude
    assert fit_layer_view(viewer, target)
    camera = viewer_camera(viewer)
    assert camera.center == pytest.approx(np.array((13000, 27500, 44000)) * factor)
    assert camera.zoom == pytest.approx(0.95 * min(800 / 18000, 600 / 32000) / factor)


def test_fit_adapts_2d_target_without_leaking_an_unrelated_leading_axis():
    viewer = ViewerModel(ndisplay=3)
    viewer.add_image(np.zeros((20, 100, 200)), rgb=False, visible=False)
    target = viewer.add_image(np.zeros((6, 8)), rgb=False, scale=(3, 4))
    viewer.dims.order = (1, 2, 0)
    assert fit_layer_view(viewer, target)
    assert viewer.dims.ndisplay == 2
    assert viewer.dims.order == (0, 1, 2)
    assert viewer_camera(viewer).center == pytest.approx((0, 7.5, 14))


def test_fit_clamps_only_target_slice_axes_to_sample_centres():
    viewer = ViewerModel()
    unrelated = viewer.add_image(np.zeros((20, 100, 200)), rgb=False, visible=False)
    target = viewer.add_image(
        np.zeros((4, 6, 8)), rgb=False, scale=(2, 3, 4), translate=(10, 20, 30)
    )
    viewer.dims.point = (19, 99, 199)
    unrelated_data = unrelated.data
    assert fit_layer_view(viewer, target)
    # Global Z range includes the hidden large layer, but the target ends at16.
    # Its outer voxel edge at17 is not a valid slice centre.
    assert viewer.dims.point == (16, 99, 199)
    assert unrelated.data is unrelated_data
    assert unrelated.visible is False


def test_restore_can_target_clamp_saved_slice_after_source_geometry_changes():
    viewer = ViewerModel()
    viewer.add_image(np.zeros((20, 100, 200)), rgb=False, visible=False)
    target = viewer.add_image(np.zeros((4, 6, 8)), rgb=False)
    state = ViewerState(
        order=(0, 1, 2),
        ndisplay=2,
        point=(19, 99, 199),
        center=(0, 4, 5),
        zoom=9,
    )
    assert restore_viewer_state(viewer, state, layer=target)
    assert viewer.dims.point == (3, 99, 199)
    assert viewer_camera(viewer).center == (0, 4, 5)
    assert viewer_camera(viewer).zoom == 9


@pytest.mark.parametrize("canvas_kind", ["public", "legacy", "absent"])
def test_fit_supports_canvas_fallbacks_and_preserves_tiny_calibrated_spans(canvas_kind):
    viewer = _fake_viewer(2)
    viewer.camera.angles = (0, 0, 0)
    if canvas_kind == "public":
        viewer.canvas = SimpleNamespace(size=(300, 500))
    elif canvas_kind == "legacy":
        viewer._canvas_size = (300, 500)
    target = SimpleNamespace(
        extent=SimpleNamespace(world=np.array([[0, 0], [2e-20, 4e-20]])), metadata={}
    )
    assert fit_layer_view(viewer, target)
    size = (800, 600) if canvas_kind == "absent" else (300, 500)
    assert viewer.camera.center == pytest.approx((0, 1e-20, 2e-20), abs=1e-30)
    assert viewer.camera.zoom == pytest.approx(
        0.95 * min(size[0] / 2e-20, size[1] / 4e-20)
    )


@pytest.mark.parametrize("margin", [-1, 1, float("nan"), float("inf")])
def test_fit_rejects_invalid_margin(margin):
    with pytest.raises(ValueError, match="margin"):
        fit_layer_view(SimpleNamespace(), SimpleNamespace(), margin=margin)


def test_invalid_extent_never_changes_camera():
    viewer = _fake_viewer()
    before = capture_viewer_state(viewer)
    layer = SimpleNamespace(extent=SimpleNamespace(world=[[0, 0, 0], [1, np.nan, 2]]))
    assert not fit_layer_view(viewer, layer)
    assert capture_viewer_state(viewer) == before
