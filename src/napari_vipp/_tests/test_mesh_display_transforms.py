"""Mesh publication must not invalidate unchanged napari transforms."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from napari.components import ViewerModel
from napari.layers import Surface

from napari_vipp._widget import VippWidget
from napari_vipp.core.meshes import MeshData, MeshState
from napari_vipp.core.metadata import AxisMetadata
from napari_vipp.ui.mesh_display import mesh_surface_display


def _mesh():
    return MeshData(
        np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]),
        np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]),
        MeshState(
            4,
            4,
            (
                AxisMetadata("z", "space", "nm", 2000, 10000),
                AxisMetadata("y", "space", "um", 0.5, 20),
                AxisMetadata("x", "space", "um", 0.25, -3),
            ),
            (2, 2, 2),
            "Close at image border",
        ),
    )


def _publish(viewer, mesh, *, role="inspect"):
    name = f"VIPP {role}"
    owner = SimpleNamespace(
        viewer=viewer,
        _generated_layers_for_name=lambda target: [
            layer for layer in viewer.layers if layer.name == target
        ],
        _remove_layer=viewer.layers.remove,
    )
    VippWidget._set_or_add_mesh_layer(
        owner, name, mesh, {"napari_vipp_kind": role}, role
    )
    return viewer.layers[name]


def _record_transform_writes(monkeypatch):
    writes = []
    for name in ("scale", "translate", "axis_labels"):
        descriptor = getattr(Surface, name)

        def record(layer, value, *, name=name, descriptor=descriptor):
            writes.append((name, tuple(value)))
            descriptor.fset(layer, value)

        monkeypatch.setattr(Surface, name, property(descriptor.fget, record))
    return writes


@pytest.mark.parametrize("role", ["inspect", "pinned"])
def test_mesh_creation_and_refresh_do_not_reassign_identical_transforms(
    monkeypatch, role
):
    viewer = ViewerModel()
    mesh = _mesh()
    writes = _record_transform_writes(monkeypatch)
    layer = _publish(viewer, mesh, role=role)

    # Initial transforms belong in the constructor, before listeners/caches.
    assert writes == []
    assert viewer.dims.ndisplay == 3
    np.testing.assert_array_equal(layer.scale, [2, 0.5, 0.25])
    np.testing.assert_array_equal(layer.translate, [10, 20, -3])
    assert layer.axis_labels == ("Z", "Y", "X")

    refreshed = replace(mesh, vertices=mesh.vertices * 2)
    for _ in range(3):
        viewer.layers.selection.active = layer
        layer.data_to_world((1, 1, 1))
        layer.world_to_data((12, 21, -2))
        assert _publish(viewer, refreshed, role=role) is layer

    assert writes == []
    expected, colors = mesh_surface_display(refreshed)
    np.testing.assert_array_equal(layer.data[0], expected[0])
    np.testing.assert_array_equal(layer.vertex_colors, colors)
    assert not np.shares_memory(layer.data[0], refreshed.vertices)
    assert not refreshed.vertices.flags.writeable


def test_image_to_surface_then_reselection_keeps_one_calibrated_layer(monkeypatch):
    viewer = ViewerModel()
    image = viewer.add_image(
        np.zeros((3, 4, 5)),
        name="VIPP inspect",
        metadata={"napari_vipp_kind": "inspect"},
    )
    writes = _record_transform_writes(monkeypatch)
    layer = _publish(viewer, _mesh())
    viewer.layers.selection.active = layer
    _publish(viewer, _mesh())

    assert image not in viewer.layers
    assert list(viewer.layers) == [layer]
    assert isinstance(layer, Surface)
    assert viewer.dims.ndisplay == 3
    assert writes == []


def test_changed_mesh_calibration_still_updates_exactly(monkeypatch):
    viewer = ViewerModel()
    mesh = _mesh()
    layer = _publish(viewer, mesh)
    writes = _record_transform_writes(monkeypatch)
    axes = list(mesh.state.spatial_axes)
    axes[0] = replace(axes[0], scale=3000, translation=15000)
    changed = replace(mesh, state=replace(mesh.state, spatial_axes=tuple(axes)))

    assert _publish(viewer, changed) is layer

    assert writes == [("scale", (3, 0.5, 0.25)), ("translate", (15, 20, -3))]
    np.testing.assert_array_equal(layer.data_to_world((1, 2, 4)), [18, 21, -2])
    assert layer.metadata["vipp_mesh_state"] == changed.state.to_dict()


def test_necessary_transform_failure_is_not_silenced(monkeypatch):
    viewer = ViewerModel()
    mesh = _mesh()
    _publish(viewer, mesh)
    descriptor = Surface.scale

    def fail(_layer, _value):
        raise ReferenceError("weakly-referenced object no longer exists")

    monkeypatch.setattr(Surface, "scale", property(descriptor.fget, fail))
    axes = list(mesh.state.spatial_axes)
    axes[0] = replace(axes[0], scale=3000)
    changed = replace(mesh, state=replace(mesh.state, spatial_axes=tuple(axes)))
    with pytest.raises(ReferenceError, match="weakly-referenced"):
        _publish(viewer, changed)


def test_small_but_real_calibration_change_is_not_treated_as_equal(monkeypatch):
    viewer = ViewerModel()
    mesh = _mesh()
    layer = _publish(viewer, mesh)
    writes = _record_transform_writes(monkeypatch)
    axes = list(mesh.state.spatial_axes)
    axes[0] = replace(axes[0], scale=axes[0].scale + 1e-8)
    changed = replace(mesh, state=replace(mesh.state, spatial_axes=tuple(axes)))

    _publish(viewer, changed)

    assert [name for name, _value in writes] == ["scale"]
    assert layer.scale[0] == axes[0].scale * 0.001
    assert layer.scale[0] > 2
