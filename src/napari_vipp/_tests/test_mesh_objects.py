"""Scientific object identity, calibration and non-destructive mesh contracts."""

from dataclasses import replace

import numpy as np
import pytest

from napari_vipp.core.mesh_objects import (
    color_mesh_objects,
    combine_meshes,
    filter_mesh_objects,
    iter_mesh_objects,
    object_id_color,
    split_mesh_objects,
)
from napari_vipp.core.meshes import (
    MeshData,
    MeshObject,
    MeshState,
    labels_to_3d_mesh,
    mask_to_3d_mesh,
    measure_mesh_geometry,
)
from napari_vipp.core.metadata import AxisMetadata, image_state_from_array


def _cube(object_id=1, *, scale=1.0, unit=None, origin=0.0, color=(0.1, 0.2, 0.3, 1.0)):
    vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ]
    )
    state = MeshState(
        8,
        12,
        tuple(AxisMetadata(n, "space", unit, scale, origin) for n in "zyx"),
        (2, 2, 2),
        "test",
        "cube",
    )
    return MeshData(
        vertices,
        faces,
        state,
        np.full(12, object_id, dtype=np.int64),
        (MeshObject(object_id, "Cube", color),),
    )


def _rows(table):
    return [dict(zip(table.columns, row, strict=True)) for row in table.rows]


def test_default_mesh_and_ids_are_immutable_and_metadata_truthful():
    mesh = _cube()
    result = MeshData(mesh.vertices, mesh.faces, mesh.state)
    assert result.object_count == result.state.object_count == 1
    assert result.objects[0].object_id == 1
    assert not result.vertices.flags.writeable
    assert not result.faces.flags.writeable
    assert not result.face_object_ids.flags.writeable
    with pytest.raises(ValueError):
        result.face_object_ids[0] = 2
    modified = replace(
        result.state, processing_history=("Smooth Mesh: Taubin 5 iterations",)
    )
    assert "Taubin" in modified.to_dict()["smoothing"]
    assert modified.to_dict()["object_count"] == 1


@pytest.mark.parametrize(
    "ids",
    [
        np.zeros(12, int),
        np.ones(12, float),
        np.ones(11, int),
        np.full(12, 2**63, dtype=np.uint64),
    ],
)
def test_invalid_face_object_ids_rejected(ids):
    mesh = _cube()
    with pytest.raises(ValueError, match="object IDs"):
        MeshData(mesh.vertices, mesh.faces, mesh.state, ids)


@pytest.mark.parametrize("object_id", [True, 0, -1, 2**63, 1.2])
def test_invalid_object_record_ids_rejected(object_id):
    with pytest.raises(ValueError, match="positive int64"):
        MeshObject(object_id)


def test_object_records_cannot_be_missing_duplicated_or_orphaned():
    mesh = _cube()
    for records in ((), (MeshObject(2),), (MeshObject(1), MeshObject(1))):
        with pytest.raises(ValueError, match="records"):
            MeshData(
                mesh.vertices, mesh.faces, mesh.state, mesh.face_object_ids, records
            )


def test_combine_calibration_position_colour_and_id_collision():
    first = _cube(7, scale=2, unit="um", origin=10)
    second = _cube(7, scale=1000, unit="nm", origin=12000, color=(1, 0, 0, 1))
    result = combine_meshes((first, second))
    assert [obj.object_id for obj in result.objects] == [7, 8]
    assert result.object_count == 2
    assert result.objects[1].source_object_id == 7
    assert result.objects[1].source_units == ("nm",) * 3
    assert [obj.color for obj in result.objects] == [
        first.objects[0].color,
        second.objects[0].color,
    ]
    parts = tuple(iter_mesh_objects(result))
    np.testing.assert_array_equal(parts[0].vertices, first.vertices)
    np.testing.assert_allclose(parts[1].vertices, second.vertices * 0.5 + 1)
    assert len(result.vertices) == 16 and len(result.faces) == 24  # No welding.
    rows = _rows(measure_mesh_geometry(result))
    assert [row["mesh_id"] for row in rows] == [7, 8]
    assert [row["mesh_volume_physical"] for row in rows] == pytest.approx([8, 1])
    assert "no welding" in result.state.history[-1]
    np.testing.assert_array_equal(first.vertices, _cube().vertices)


@pytest.mark.parametrize("unit", [None, "s", "custom"])
def test_combine_rejects_incompatible_or_uncalibrated_units(unit):
    with pytest.raises(ValueError, match="units|calibrated"):
        combine_meshes((_cube(unit="um"), _cube(unit=unit)))


def test_sparse_large_ids_never_round_and_collisions_remap_safely():
    first_id = 2**62
    mesh = combine_meshes((_cube(first_id), _cube(first_id + 1)))
    selected = filter_mesh_objects(mesh, "mesh_id", first_id + 1, first_id + 1)
    assert [item.object_id for item in selected.objects] == [first_id + 1]
    assert object_id_color(first_id) != object_id_color(first_id + 1)
    biggest = combine_meshes(
        (_cube(np.iinfo(np.int64).max), _cube(np.iinfo(np.int64).max))
    )
    assert {item.object_id for item in biggest.objects} == {1, np.iinfo(np.int64).max}


def test_split_uses_shared_edges_not_merely_a_vertex_and_preserves_parent_colour():
    first = _cube(4)
    second_vertices = first.vertices + 1
    # Cubes touch at exactly one coordinate, but even shared vertex index must
    # not join components that have no common triangle edge.
    vertices = np.concatenate((first.vertices, second_vertices[1:]))
    mapping = np.array([6, *range(8, 15)])
    faces = np.concatenate((first.faces, mapping[first.faces]))
    source = MeshData(
        vertices,
        faces,
        first.state,
        objects=first.objects,
        face_object_ids=np.full(len(faces), 4, dtype=np.int64),
    )
    split = split_mesh_objects(source)
    assert split.object_count == 2
    assert [len(part.faces) for part in iter_mesh_objects(split)] == [12, 12]
    assert all(item.parent_object_id == 4 for item in split.objects)
    assert all(item.color == first.objects[0].color for item in split.objects)
    np.testing.assert_array_equal(source.vertices, vertices)
    np.testing.assert_array_equal(source.faces, faces)
    again = split_mesh_objects(source)
    np.testing.assert_array_equal(split.face_object_ids, again.face_object_ids)


def test_connected_foreground_retains_cavity_as_one_object_then_explicit_split():
    mask = np.zeros((12, 13, 14), bool)
    mask[1:11, 1:12, 1:13] = True
    mask[4:8, 4:8, 4:8] = False
    mask.setflags(write=False)
    mesh = mask_to_3d_mesh(mask, object_mode="Connected objects")
    assert mesh.object_count == 1
    assert split_mesh_objects(mesh).object_count == 2
    whole = measure_mesh_geometry(mask_to_3d_mesh(mask)).rows
    assert measure_mesh_geometry(mesh).rows == whole


def test_label_extraction_preserves_sparse_ids_axes_units_and_disconnected_same_label():
    labels = np.zeros((12, 13, 14), np.int64)
    labels[1:3, 1:3, 1:3] = 2**62
    labels[7:9, 7:9, 7:9] = 2**62
    labels[4:6, 4:6, 4:6] = 17
    data = labels.transpose((2, 1, 0))
    axes = tuple(
        AxisMetadata(name, "space", "um", scale, 4)
        for name, scale in zip("xyz", (2, 3, 4), strict=True)
    )
    state = image_state_from_array(data, axes=axes, source_name="Labels")
    data.setflags(write=False)
    mesh = labels_to_3d_mesh(data, image_state=state)
    assert [item.object_id for item in mesh.objects] == [17, 2**62]
    assert tuple(axis.scale for axis in mesh.state.spatial_axes) == (4, 3, 2)
    assert split_mesh_objects(mesh).object_count == 3
    assert [row["mesh_id"] for row in _rows(measure_mesh_geometry(mesh))] == [17, 2**62]
    np.testing.assert_array_equal(data, labels.transpose((2, 1, 0)))


@pytest.mark.parametrize(
    "data",
    [
        np.zeros((3, 3, 3), float),
        np.ones((3, 3, 3), bool),
        np.full((3, 3, 3), -1),
        np.full((3, 3, 3), 2**63, np.uint64),
    ],
)
def test_labels_require_explicit_nonnegative_int64_values(data):
    with pytest.raises(TypeError, match="Label IDs"):
        labels_to_3d_mesh(data)


def test_colours_by_current_measurements_and_filter_bounds_are_non_destructive():
    small = _cube(11)
    large = replace(_cube(12), vertices=_cube().vertices * 2)
    mesh = combine_meshes((small, large))
    colored = color_mesh_objects(mesh, "mesh_volume_physical", "viridis")
    assert colored.objects[0].color != colored.objects[1].color
    assert [item.object_id for item in colored.objects] == [11, 12]
    np.testing.assert_array_equal(colored.vertices, mesh.vertices)
    np.testing.assert_array_equal(colored.faces, mesh.faces)
    kept = filter_mesh_objects(colored, minimum=1, maximum=1)
    assert [item.object_id for item in kept.objects] == [11]
    assert kept.objects[0].color == colored.objects[0].color
    outside = filter_mesh_objects(colored, minimum=1, maximum=1, keep="Outside range")
    assert [item.object_id for item in outside.objects] == [12]
    resized = replace(colored, vertices=colored.vertices * 2)
    assert filter_mesh_objects(resized, minimum=1, maximum=1).object_count == 0
    assert all(
        not buffer.flags.writeable
        for buffer in (mesh.vertices, mesh.faces, mesh.face_object_ids)
    )


def test_missing_volume_coloured_grey_and_excluded_for_both_range_choices():
    cube = _cube()
    open_mesh = MeshData(cube.vertices, cube.faces[:-1], cube.state)
    assert color_mesh_objects(open_mesh, "mesh_volume_physical").objects[0].color == (
        0.5,
        0.5,
        0.5,
        1.0,
    )
    for keep in ("In range", "Outside range"):
        result = filter_mesh_objects(open_mesh, keep=keep)
        assert result.object_count == 0 and result.faces.shape == (0, 3)
    assert (
        filter_mesh_objects(open_mesh, "mesh_surface_area_physical").object_count == 1
    )


def test_empty_collections_and_invalid_controls():
    empty = mask_to_3d_mesh(np.zeros((3, 3, 3), bool))
    assert combine_meshes((empty, empty)).object_count == 0
    assert split_mesh_objects(empty).object_count == 0
    assert color_mesh_objects(empty).object_count == 0
    assert filter_mesh_objects(empty).object_count == 0
    assert _rows(measure_mesh_geometry(empty))[0]["mesh_status"] == "empty_mesh"
    for kwargs in (
        dict(minimum=2, maximum=1),
        dict(minimum=np.nan),
        dict(keep="Maybe"),
    ):
        with pytest.raises(ValueError):
            filter_mesh_objects(_cube(), **kwargs)
    with pytest.raises(ValueError, match="Unknown"):
        color_mesh_objects(_cube(), "mesh_volume_physical", "not-a-map")
    for count in (True, 1.0, 3, 0):
        with pytest.raises(ValueError, match="supplied mesh count"):
            combine_meshes((_cube(), _cube()), input_count=count)
    assert combine_meshes((_cube(), _cube()), input_count=2).object_count == 2


def test_cancellation_is_observed_before_any_input_change():
    class Cancelled:
        def check_cancelled(self):
            raise RuntimeError("cancelled")

    mesh = _cube()
    for call in (
        lambda: combine_meshes((mesh,), progress=Cancelled()),
        lambda: split_mesh_objects(mesh, progress=Cancelled()),
        lambda: color_mesh_objects(mesh, progress=Cancelled()),
        lambda: filter_mesh_objects(mesh, progress=Cancelled()),
        lambda: labels_to_3d_mesh(np.ones((3, 3, 3), int), progress=Cancelled()),
    ):
        with pytest.raises(RuntimeError, match="cancelled"):
            call()
    assert not mesh.vertices.flags.writeable


def test_measurement_progress_advances_across_objects_without_resetting():
    from napari_vipp.core.progress import ProgressContext

    fractions = []
    progress = ProgressContext(
        reporter=lambda update: fractions.append(update.current / update.total)
    )
    measure_mesh_geometry(
        combine_meshes((_cube(1), _cube(2), _cube(3))), progress=progress
    )
    assert fractions == sorted(fractions)
    assert fractions[0] == 0 and fractions[-1] == 1


def test_combine_preserves_repeated_scientific_history_and_divergent_source_history():
    mesh = _cube()
    history = (
        "Threshold above 1",
        "Smooth Mesh: 10 iterations",
        "Smooth Mesh: 10 iterations",
    )
    mesh = replace(
        mesh,
        state=replace(mesh.state, history=history),
        objects=(replace(mesh.objects[0], source_history=history[:1]),),
    )
    combined = combine_meshes((mesh,))
    assert combined.objects[0].source_history == history
    previous = ("Different source", "Filter labels", "Filter labels")
    mesh = replace(mesh, objects=(replace(mesh.objects[0], source_history=previous),))
    combined = combine_meshes((mesh,))
    assert combined.objects[0].source_history == (
        *previous,
        "Combined input history: cube",
        *history,
    )


@pytest.mark.parametrize(
    "change", [{"type": "channel"}, {"type": "time"}, {"confidence": "inferred"}]
)
def test_combine_requires_declared_spatial_axes(change):
    mesh = _cube()
    axes = (replace(mesh.state.spatial_axes[0], **change), *mesh.state.spatial_axes[1:])
    malformed = replace(mesh, state=replace(mesh.state, spatial_axes=axes))
    with pytest.raises(ValueError, match="explicit Z/Y/X"):
        combine_meshes((malformed,))


def test_combine_identity_frame_preserves_bit_exact_vertices():
    mesh = _cube(scale=3, origin=19)
    vertices = mesh.vertices.copy()
    vertices[1, 0] = 0.1
    mesh = replace(mesh, vertices=vertices)
    combined = combine_meshes((mesh, mesh))
    np.testing.assert_array_equal(
        combined.vertices, np.concatenate((vertices, vertices))
    )
    assert not combined.vertices.flags.writeable


@pytest.mark.parametrize("field", ["minimum", "maximum"])
@pytest.mark.parametrize("value", [True, False, np.bool_(True)])
def test_filter_rejects_boolean_bounds(field, value):
    with pytest.raises(ValueError, match="not booleans"):
        filter_mesh_objects(_cube(), **{field: value})
