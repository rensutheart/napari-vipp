"""Non-destructive refinement, analytical surfaces and physical-frame contracts."""

from dataclasses import replace

import numpy as np
import pytest

from napari_vipp.core.mesh_objects import combine_meshes, iter_mesh_objects
from napari_vipp.core.mesh_refinement import simplify_mesh, smooth_mesh
from napari_vipp.core.meshes import MeshData, MeshObject, MeshState, mask_to_3d_mesh
from napari_vipp.core.metadata import AxisMetadata
from napari_vipp.core.progress import OperationCancelled, ProgressContext


def _sphere(radius=6):
    coordinates = np.indices((2 * radius + 5,) * 3) - (radius + 2)
    return mask_to_3d_mesh(np.sum(coordinates**2, axis=0) <= radius**2)


def _plane(size=9, bump=False):
    yy, xx = np.indices((size, size), dtype=float)
    zz = np.zeros_like(xx)
    if bump:
        zz[size // 2, size // 2] = 0.5
    vertices = np.column_stack((zz.ravel(), yy.ravel(), xx.ravel()))
    faces = []
    for y in range(size - 1):
        for x in range(size - 1):
            a = y * size + x
            faces.extend(((a, a + 1, a + size), (a + 1, a + size + 1, a + size)))
    state = MeshState(
        len(vertices),
        len(faces),
        tuple(AxisMetadata(n, "space") for n in "zyx"),
        (1, size, size),
        "test open plane",
    )
    return MeshData(vertices, np.array(faces, np.int64), state)


def _signed_volume(mesh):
    points = mesh.vertices - mesh.vertices.mean(axis=0)
    triangles = points[mesh.faces]
    return (
        np.einsum(
            "ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])
        ).sum()
        / 6
    )


def _boundary(mesh):
    directed = np.concatenate(
        (mesh.faces[:, [0, 1]], mesh.faces[:, [1, 2]], mesh.faces[:, [2, 0]])
    )
    edges, counts = np.unique(np.sort(directed, axis=1), axis=0, return_counts=True)
    return mesh.vertices[np.unique(edges[counts == 1])]


@pytest.mark.parametrize("refine", [smooth_mesh, simplify_mesh])
def test_refinement_preserves_readonly_original_ids_colours_and_frame(refine):
    original = _sphere()
    record = MeshObject(7, "Nucleus", (0.1, 0.7, 0.3, 0.8), source_object_id=29)
    mesh = replace(
        original, face_object_ids=np.full(len(original.faces), 7), objects=(record,)
    )
    before = mesh.vertices.copy(), mesh.faces.copy(), mesh.face_object_ids.copy()
    result = refine(mesh)
    assert result is not mesh and result.objects == (record,)
    assert result.state.spatial_axes == mesh.state.spatial_axes
    assert result.state.source_shape == mesh.state.source_shape
    assert (
        result.state.processing_history
        and "approximate" in result.state.processing_history[-1]
    )
    assert np.all(result.face_object_ids == 7)
    for current, previous in zip(
        (mesh.vertices, mesh.faces, mesh.face_object_ids), before, strict=True
    ):
        np.testing.assert_array_equal(current, previous)
        assert not current.flags.writeable
    assert not result.vertices.flags.writeable and not result.faces.flags.writeable
    assert _signed_volume(result) > 0
    if refine is smooth_mesh:
        assert len(result.faces) == len(mesh.faces)
        assert not np.array_equal(result.vertices, mesh.vertices)
        assert "Smooth Mesh" in result.state.to_dict()["smoothing"]
    else:
        assert len(result.faces) < len(mesh.faces)
        assert "Simplify Mesh" in result.state.to_dict()["simplification"]


def test_smoothing_matches_explicit_one_iteration_taubin_reference():
    mesh = _plane(size=4, bump=True)
    axes = tuple(
        AxisMetadata(name, "space", "um", scale)
        for name, scale in zip("zyx", (3, 0.5, 2), strict=True)
    )
    mesh = replace(mesh, state=replace(mesh.state, spatial_axes=axes))
    points = mesh.vertices * [3, 0.5, 2]
    neighbours = [set() for _ in points]
    for a, b, c in mesh.faces:
        neighbours[a].update((b, c))
        neighbours[b].update((a, c))
        neighbours[c].update((a, b))
    rows = []
    for vertex, adjacent in enumerate(neighbours):
        adjacent = sorted(adjacent)
        weights = 1 / np.linalg.norm(points[adjacent] - points[vertex], axis=1)
        rows.append((adjacent, weights / weights.sum()))
    reference = points.copy()
    for coefficient in (0.25, -0.25 / 0.975):
        means = np.array([weights @ reference[adjacent] for adjacent, weights in rows])
        reference += coefficient * (means - reference)
    result = smooth_mesh(mesh, iterations=1, strength=0.5, preserve_boundary=False)
    np.testing.assert_allclose(result.vertices * [3, 0.5, 2], reference, atol=1e-14)


def test_smoothing_reduces_bump_and_locks_open_boundary_exactly():
    mesh = _plane(bump=True)
    result = smooth_mesh(mesh, iterations=10)
    np.testing.assert_array_equal(_boundary(mesh), _boundary(result))
    assert 0 < np.max(result.vertices[:, 0]) < np.max(mesh.vertices[:, 0])
    flat = _plane()
    unchanged_plane = smooth_mesh(flat, iterations=10)
    assert np.all(unchanged_plane.vertices[:, 0] == 0)


def test_simplification_preserves_open_boundary_and_area():
    mesh = _plane(size=15)
    result = simplify_mesh(mesh, target_percent=35, preserve_boundary=True)
    assert len(result.faces) < len(mesh.faces)
    assert {tuple(point) for point in _boundary(result)} == {
        tuple(point) for point in _boundary(mesh)
    }
    assert np.all(result.vertices[:, 0] == 0)
    triangles = result.vertices[result.faces]
    area = (
        np.linalg.norm(
            np.cross(
                triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
            ),
            axis=1,
        ).sum()
        / 2
    )
    assert area == pytest.approx(14 * 14)


@pytest.mark.parametrize("refine", [smooth_mesh, simplify_mesh])
def test_equivalent_physical_geometry_in_anisotropic_and_baked_frames(refine):
    mesh = _sphere()
    # Binary-exact physical coordinates avoid testing provider tie-breaking
    # under a different order of floating-point multiplication/subtraction.
    scales = np.array([4.0, 0.5, 0.25])
    axes = (
        AxisMetadata("z", "space", "um", 4, 1e10),
        AxisMetadata("y", "space", "nm", 500, -3e6),
        AxisMetadata("x", "space", "um", 0.25, 10),
    )
    calibrated = replace(mesh, state=replace(mesh.state, spatial_axes=axes))
    baked = replace(
        mesh,
        vertices=mesh.vertices * scales,
        state=replace(
            mesh.state,
            spatial_axes=tuple(AxisMetadata(n, "space", "um") for n in "zyx"),
        ),
    )
    first, second = refine(calibrated), refine(baked)
    np.testing.assert_array_equal(first.faces, second.faces)
    np.testing.assert_allclose(first.vertices * scales, second.vertices, atol=1e-11)
    assert first.state.spatial_axes == axes


@pytest.mark.parametrize("refine", [smooth_mesh, simplify_mesh])
def test_overlapping_objects_remain_separate_with_ids_and_colours(refine):
    mesh = _sphere()
    first_record = MeshObject(10, "First", (1, 0, 0, 1))
    second_record = MeshObject(20, "Second", (0, 1, 0, 1))
    first = replace(
        mesh, face_object_ids=np.full(len(mesh.faces), 10), objects=(first_record,)
    )
    second = replace(
        mesh,
        vertices=mesh.vertices + 2,
        face_object_ids=np.full(len(mesh.faces), 20),
        objects=(second_record,),
    )
    combined = combine_meshes([first, second])
    result = refine(combined)
    assert result.objects == combined.objects
    actual = tuple(iter_mesh_objects(result))
    expected = tuple(refine(part) for part in iter_mesh_objects(combined))
    assert len(actual) == len(expected) == 2
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a.faces, b.faces)
        np.testing.assert_allclose(a.vertices, b.vertices)


def test_simplifier_keeps_a_small_disconnected_component_and_cavity_winding():
    large = _sphere(6)
    tiny = _sphere(1)
    offset = len(large.vertices)
    mesh = MeshData(
        np.concatenate((large.vertices, tiny.vertices + 50)),
        np.concatenate((large.faces, tiny.faces + offset)),
        large.state,
    )
    result = simplify_mesh(mesh, target_percent=5)
    assert np.any(result.vertices[:, 0] > 40) and np.any(result.vertices[:, 0] < 30)
    # An inward-wound shell remains inward. We do not independently reverse
    # all shells to make their enclosed volumes positive.
    inward = replace(tiny, faces=tiny.faces[:, ::-1])
    inward_result = simplify_mesh(inward, target_percent=50)
    assert _signed_volume(inward_result) < 0


def test_noop_parameters_and_empty_inputs():
    mesh = _sphere()
    for result in (
        smooth_mesh(mesh, strength=0),
        simplify_mesh(mesh, target_percent=100),
    ):
        np.testing.assert_array_equal(result.vertices, mesh.vertices)
        np.testing.assert_array_equal(result.faces, mesh.faces)
    empty = mask_to_3d_mesh(np.zeros((3, 3, 3), bool))
    for result in (smooth_mesh(empty), simplify_mesh(empty)):
        assert result.vertices.shape == result.faces.shape == (0, 3)
        assert result.object_count == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"iterations": 0},
        {"iterations": 1.5},
        {"iterations": True},
        {"iterations": 1001},
        {"strength": -0.1},
        {"strength": 1.1},
        {"strength": float("nan")},
        {"strength": True},
        {"preserve_boundary": "False"},
    ],
)
def test_smoothing_rejects_invalid_controls(kwargs):
    with pytest.raises(ValueError):
        smooth_mesh(_plane(), **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_percent": 0},
        {"target_percent": 101},
        {"target_percent": True},
        {"target_percent": float("nan")},
        {"aggressiveness": -1},
        {"aggressiveness": 11},
        {"aggressiveness": float("inf")},
        {"preserve_boundary": "False"},
    ],
)
def test_simplification_rejects_invalid_controls(kwargs):
    with pytest.raises(ValueError):
        simplify_mesh(_plane(), **kwargs)


@pytest.mark.parametrize("refine", [smooth_mesh, simplify_mesh])
def test_refinement_rejects_invalid_calibration_and_unsafe_surfaces(refine):
    mesh = _plane()
    axes = list(mesh.state.spatial_axes)
    axes[0] = replace(axes[0], unit="um")
    with pytest.raises(ValueError, match="compatible units"):
        refine(replace(mesh, state=replace(mesh.state, spatial_axes=tuple(axes))))
    with pytest.raises(TypeError, match="requires a 3D mesh"):
        refine(np.zeros((3, 3, 3)))
    duplicated = MeshData(
        mesh.vertices, np.concatenate((mesh.faces, mesh.faces[:1])), mesh.state
    )
    with pytest.raises(ValueError, match="duplicate"):
        refine(duplicated)
    degenerate = MeshData(mesh.vertices, np.array([[0, 0, 1]]), mesh.state)
    with pytest.raises(ValueError, match="non-degenerate"):
        refine(degenerate)
    reversed_faces = mesh.faces.copy()
    reversed_faces[0] = reversed_faces[0, ::-1]
    with pytest.raises(ValueError, match="consistently wound"):
        refine(MeshData(mesh.vertices, reversed_faces, mesh.state))


@pytest.mark.parametrize("refine", [smooth_mesh, simplify_mesh])
def test_cancelled_refinement_never_returns_a_partial_result(refine):
    mesh = _sphere()
    updates = []
    progress = ProgressContext(cancelled=lambda: bool(updates), reporter=updates.append)
    with pytest.raises(OperationCancelled):
        refine(mesh, progress=progress)
    assert len(updates) == 1
    assert not mesh.vertices.flags.writeable


def test_simplifier_rejects_broken_provider_output(monkeypatch):
    import napari_vipp.core.mesh_refinement as module

    monkeypatch.setattr(
        module,
        "_simplify_component",
        lambda *_args: (np.zeros((3, 3)), np.array([[0, 1, 2]], np.int64)),
    )
    with pytest.raises(ValueError, match="non-degenerate"):
        simplify_mesh(_sphere())


def test_native_provider_receives_owned_buffers_and_lock_releases(monkeypatch):
    import fast_simplification

    import napari_vipp.core.mesh_refinement as module

    observed = []

    def failing(points, faces, **kwargs):
        assert points.flags.writeable and faces.flags.writeable
        points[:] = 0
        faces[:] = 0
        observed.append(kwargs)
        raise RuntimeError("provider failed")

    monkeypatch.setattr(fast_simplification, "simplify", failing)
    mesh = _sphere()
    before = mesh.vertices.copy(), mesh.faces.copy()
    with pytest.raises(RuntimeError, match="provider failed"):
        simplify_mesh(mesh)
    np.testing.assert_array_equal(mesh.vertices, before[0])
    np.testing.assert_array_equal(mesh.faces, before[1])
    assert observed[0]["preserve_border"] is True
    assert not module._SIMPLIFICATION_LOCK.locked()


def test_simplification_cancellation_after_native_call_releases_lock(monkeypatch):
    import fast_simplification

    import napari_vipp.core.mesh_refinement as module

    cancelled = []
    original = fast_simplification.simplify

    def finish_then_cancel(*args, **kwargs):
        result = original(*args, **kwargs)
        cancelled.append(True)
        return result

    monkeypatch.setattr(fast_simplification, "simplify", finish_then_cancel)
    with pytest.raises(OperationCancelled):
        simplify_mesh(
            _sphere(), progress=ProgressContext(cancelled=lambda: bool(cancelled))
        )
    assert not module._SIMPLIFICATION_LOCK.locked()


def test_simplifier_rejects_orientation_loss_and_locked_boundary_motion(monkeypatch):
    import napari_vipp.core.mesh_refinement as module

    def inverted(points, faces, *_args):
        return points, faces[:, ::-1]

    monkeypatch.setattr(module, "_simplify_component", inverted)
    with pytest.raises(ValueError, match="collapsed or inverted"):
        simplify_mesh(_sphere())

    def translated(points, faces, *_args):
        return points + 1, faces

    monkeypatch.setattr(module, "_simplify_component", translated)
    with pytest.raises(ValueError, match="locked boundary"):
        simplify_mesh(_plane())


def test_provider_may_keep_more_triangles_and_records_achieved_count(monkeypatch):
    import napari_vipp.core.mesh_refinement as module

    monkeypatch.setattr(module, "_simplify_component", lambda p, f, *_args: (p, f))
    mesh = _sphere()
    result = simplify_mesh(mesh, target_percent=10)
    assert len(result.faces) == len(mesh.faces)
    message = result.state.processing_history[-1]
    assert "target keep 10%" in message
    assert f"{len(mesh.faces)} to {len(mesh.faces)} triangles achieved" in message


def test_anisotropic_simplification_keeps_boundary_in_original_frame_exactly():
    mesh = _plane(size=12)
    axes = tuple(
        AxisMetadata(n, "space", "um", scale)
        for n, scale in zip("zyx", (3.1, 0.37, 0.21), strict=True)
    )
    mesh = replace(
        mesh,
        vertices=mesh.vertices + 100.33,
        state=replace(mesh.state, spatial_axes=axes),
    )
    result = simplify_mesh(mesh, target_percent=40)
    assert {tuple(point) for point in _boundary(result)} == {
        tuple(point) for point in _boundary(mesh)
    }


def test_simplification_history_records_actual_provider_version():
    import importlib.metadata

    result = simplify_mesh(_sphere())
    version = importlib.metadata.version("fast-simplification")
    assert f"fast-simplification {version}" in result.state.processing_history[-1]
