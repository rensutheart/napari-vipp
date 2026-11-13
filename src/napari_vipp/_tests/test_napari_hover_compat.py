"""Cursor status regressions using napari's model and tiny synthetic arrays."""

from __future__ import annotations

from types import SimpleNamespace

import napari
import numpy as np
import pytest
from napari.components import ViewerModel

from napari_vipp.ui.napari_compat import install_layer_cursor_safety


@pytest.fixture
def cursor_viewer():
    viewer = ViewerModel(ndisplay=2)
    try:
        yield viewer
    finally:
        shutdown = getattr(getattr(viewer, "_layer_slicer", None), "shutdown", None)
        if callable(shutdown):
            shutdown()


def test_cursor_safety_is_scoped_and_preserves_normal_2d_and_3d_values(
    cursor_viewer, monkeypatch
):
    viewer = cursor_viewer
    data = np.full((5, 7, 9), 23, dtype=np.uint16)
    protected = viewer.add_image(
        data, rgb=False, metadata={"napari_vipp_kind": "inspect"}
    )
    untouched = viewer.add_image(data.copy(), rgb=False, name="User layer")
    read_status = getattr(viewer, "_calc_status_from_cursor", None)
    original_ray = getattr(type(protected), "_get_value_ray", None)
    ndisplay_event = getattr(viewer.dims.events, "ndisplay", None)
    if not (
        callable(read_status)
        and callable(original_ray)
        and hasattr(viewer.cursor, "_view_direction")
        and hasattr(ndisplay_event, "blocker")
        and hasattr(getattr(protected, "_slice", None), "image")
    ):
        pytest.skip("napari does not expose the affected cursor/slice layout")

    viewer.layers.selection.active = protected
    viewer.mouse_over_canvas = True
    position = (2.0, 3.0, 4.0)
    direction = np.array([-1.0, 0.0, 0.0])
    viewer.cursor.position = position
    viewer.cursor._view_direction = direction
    viewer.tooltip.visible = True
    status_2d, tooltip_2d = read_status()
    assert status_2d["value"] == "23"

    # Keep the old *real* 2D slice cached while the viewer starts requesting
    # 3D cursor values. This reproduces the reported transition without timing
    # dependence, a mocked array failure, or a real OpenGL window.
    with ndisplay_event.blocker():
        viewer.dims.ndisplay = 3
    assert len(viewer.dims.displayed) == 3
    assert protected._slice.image.raw.ndim == 2
    try:
        read_status()
    except ValueError as error:
        assert "operands could not be broadcast together" in str(error)
    else:
        if napari.__version__ == "0.9.0":
            pytest.fail("the expected napari 0.9.0 stale-slice failure did not occur")
        pytest.skip("this napari version already handles the stale cursor slice")

    assert install_layer_cursor_safety(protected)
    methods = ("get_status", "_get_tooltip_text", "_get_value_3d", "_get_value_ray")
    wrapped_methods = {
        name: vars(protected)[name] for name in methods if name in vars(protected)
    }
    assert wrapped_methods
    assert install_layer_cursor_safety(protected)
    assert all(
        vars(protected)[name] is method for name, method in wrapped_methods.items()
    )
    assert type(protected)._get_value_ray is original_ray
    assert not any(name in vars(untouched) for name in wrapped_methods)

    stale_status, _ = read_status()
    assert stale_status["value"] == ""
    # Selection itself refreshes hover status; isolate the explicit read below.
    with viewer.layers.selection.events.active.blocker():
        viewer.layers.selection.active = untouched
    with pytest.raises(ValueError, match="operands could not be broadcast together"):
        read_status()
    viewer.layers.selection.active = protected

    viewer.dims.ndisplay = 2
    # napari repositions the hidden coordinate to the selected slice.
    viewer.cursor.position = position
    viewer.cursor._view_direction = direction
    assert read_status() == (status_2d, tooltip_2d)
    viewer.dims.ndisplay = 3
    protected._slice_dims(viewer.dims, force=True)
    untouched._slice_dims(viewer.dims, force=True)
    viewer.cursor.position = position
    viewer.cursor._view_direction = direction
    assert protected._slice.image.raw.ndim == 3
    valid_status, _ = read_status()
    assert valid_status["value"] == "23"
    viewer.layers.selection.active = untouched
    assert read_status()[0]["value"] == valid_status["value"]
    viewer.layers.selection.active = protected

    def unrelated_failure(*_args, **_kwargs):
        raise ValueError("another ray-picking error")

    monkeypatch.setattr(protected, "_calculate_value_from_ray", unrelated_failure)
    with pytest.raises(ValueError, match="another ray-picking error"):
        read_status()
    np.testing.assert_array_equal(protected.data, data)


def test_cursor_safety_ignores_unfamiliar_layer_objects():
    assert not install_layer_cursor_safety(SimpleNamespace())
