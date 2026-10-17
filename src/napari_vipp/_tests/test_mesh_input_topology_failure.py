"""Typed safe diagnoses preserve the exact scientific refinement admission."""

from dataclasses import replace

import numpy as np
import pytest
from skimage.measure import marching_cubes

from napari_vipp._tests.test_mesh_refinement import _plane, _sphere
from napari_vipp.core.mesh_refinement import (
    MeshInputTopologyError,
    _input_surface_contract,
    _refinement_failure,
    safe_mesh_input_failure,
    simplify_mesh,
    smooth_mesh,
)
from napari_vipp.core.meshes import MeshData, mask_to_3d_mesh
from napari_vipp.core.metadata import AxisMetadata, image_state_from_array


@pytest.mark.parametrize("refine", [smooth_mesh, simplify_mesh])
def test_duplicate_input_is_typed_with_same_message_and_no_geometry_change(refine):
    original = _plane()
    mesh = MeshData(
        original.vertices,
        np.concatenate((original.faces, original.faces[:1, ::-1])),
        replace(original.state, source_name="private-path-not-for-provider"),
    )
    before = mesh.vertices.copy(), mesh.faces.copy(), mesh.state
    with pytest.raises(MeshInputTopologyError) as failure:
        refine(mesh)
    error = failure.value
    expected_operation = "Smooth Mesh" if refine is smooth_mesh else "Simplify Mesh"
    assert error.operation_id == refine.__name__
    assert error.code == "duplicate_faces"
    assert error.count == 1 and type(error.count) is int
    assert error.object_id == 1 and type(error.object_id) is int
    assert (
        str(error)
        == error.safe_message
        == str(
            _refinement_failure(
                expected_operation,
                1,
                "duplicate triangles (1 repeated face with the same three vertex IDs)",
            )
        )
    )
    assert "private-path-not-for-provider" not in error.safe_message
    assert safe_mesh_input_failure(
        error.operation_id, type(error).__name__, str(error)
    ) == {
        "code": "duplicate_faces",
        "safe_message": str(error),
        "object_id": 1,
        "count": 1,
    }
    np.testing.assert_array_equal(mesh.vertices, before[0])
    np.testing.assert_array_equal(mesh.faces, before[1])
    assert mesh.state == before[2]


@pytest.mark.parametrize(
    ("vertices", "faces", "code"),
    [
        ([[np.nan, 0, 0], [0, 1, 0], [0, 0, 1]], [[0, 1, 2]], "nonfinite_coordinates"),
        ([[0, 0, 0], [0, 1, 0], [0, 2, 0]], [[0, 1, 2]], "degenerate_faces"),
        (
            [[0, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]],
            [[0, 1, 2], [0, 1, 3]],
            "nonmanifold_or_inconsistent_winding",
        ),
        (
            [[0, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1]],
            [[0, 1, 2], [1, 0, 3], [0, 1, 4]],
            "nonmanifold_or_inconsistent_winding",
        ),
    ],
)
def test_other_existing_admission_criteria_are_known_and_roundtrip(
    vertices, faces, code
):
    with pytest.raises(MeshInputTopologyError) as failure:
        _input_surface_contract(
            np.array(vertices, float), np.array(faces, np.int64), "Smooth Mesh", 37
        )
    error = failure.value
    assert error.code == code and error.count is None and error.object_id == 37
    assert safe_mesh_input_failure(
        "smooth_mesh", "MeshInputTopologyError", str(error)
    ) == {
        "code": code,
        "safe_message": str(error),
        "object_id": 37,
        "count": None,
    }


@pytest.mark.parametrize("operation_id", ["smooth_mesh", "simplify_mesh"])
@pytest.mark.parametrize("count", [1, 2, 9223372036854775807])
def test_exact_canonical_duplicate_failure_roundtrip(operation_id, count):
    error = MeshInputTopologyError(
        operation_id, 9223372036854775807, "duplicate_faces", count
    )
    result = safe_mesh_input_failure(operation_id, "MeshInputTopologyError", str(error))
    assert result["count"] == count
    assert result["object_id"] == 9223372036854775807
    assert set(result) == {"code", "safe_message", "object_id", "count"}


@pytest.mark.parametrize(
    "mutate",
    [
        lambda text: text + " C:/private/source.tif",
        lambda text: "C:/private/source.tif " + text,
        lambda text: text + "\n",
        lambda text: text + " ",
        lambda text: text.replace("object 2", "object 02"),
        lambda text: text.replace("object 2", "object 0"),
        lambda text: text.replace("object 2", "object -2"),
        lambda text: text.replace("object 2", "object 9223372036854775808"),
        lambda text: text.replace("3 repeated faces", "03 repeated faces"),
        lambda text: text.replace("3 repeated faces", "0 repeated faces"),
        lambda text: text.replace("3 repeated faces", "3 repeated face"),
        lambda text: text.replace(
            "3 repeated faces", "9223372036854775808 repeated faces"
        ),
        lambda text: text.replace("duplicate triangles", "provider C:/private/file"),
        lambda text: text.replace("No new mesh", "Unexpected input; No new mesh"),
        lambda text: text.replace("input mesh", "input\nmesh"),
    ],
)
def test_serialized_failure_parser_rejects_noncanonical_or_injected_text(mutate):
    message = str(MeshInputTopologyError("smooth_mesh", 2, "duplicate_faces", 3))
    assert (
        safe_mesh_input_failure(
            "smooth_mesh", "MeshInputTopologyError", mutate(message)
        )
        is None
    )


@pytest.mark.parametrize(
    ("operation_id", "error_type", "message"),
    [
        ("smooth_mesh", "ValueError", None),
        ("simplify_mesh", "MeshInputTopologyError", None),
        ("not_reviewed", "MeshInputTopologyError", None),
        ([], "MeshInputTopologyError", None),
        ("smooth_mesh", None, None),
        ("smooth_mesh", "MeshInputTopologyError", []),
        ("smooth_mesh", "MeshInputTopologyError", "x" * 1025),
    ],
)
def test_failure_parser_requires_exact_type_operation_and_bounded_text(
    operation_id, error_type, message
):
    canonical = str(MeshInputTopologyError("smooth_mesh", 2, "duplicate_faces", 3))
    assert (
        safe_mesh_input_failure(
            operation_id, error_type, canonical if message is None else message
        )
        is None
    )


@pytest.mark.parametrize(
    ("operation", "object_id", "code", "count"),
    [
        ("other", 1, "duplicate_faces", 1),
        ([], 1, "duplicate_faces", 1),
        ("smooth_mesh", True, "duplicate_faces", 1),
        ("smooth_mesh", 0, "duplicate_faces", 1),
        ("smooth_mesh", 1.0, "duplicate_faces", 1),
        ("smooth_mesh", 2**63, "duplicate_faces", 1),
        ("smooth_mesh", 1, "unknown C:/private", None),
        ("smooth_mesh", 1, [], None),
        ("smooth_mesh", 1, "duplicate_faces", True),
        ("smooth_mesh", 1, "duplicate_faces", 0),
        ("smooth_mesh", 1, "duplicate_faces", 1.0),
        ("smooth_mesh", 1, "duplicate_faces", 2**63),
        ("smooth_mesh", 1, "degenerate_faces", 1),
    ],
)
def test_public_diagnosis_fields_are_only_known_codes_and_bounded_integer_facts(
    operation, object_id, code, count
):
    with pytest.raises(ValueError):
        MeshInputTopologyError(operation, object_id, code, count)


def test_rejected_refinement_result_is_not_an_input_failure(monkeypatch):
    import napari_vipp.core.mesh_refinement as module

    def invalid_generated(*_args):
        raise module._SurfaceContractViolation("duplicate_faces", 2)

    monkeypatch.setattr(module, "_check_refined_topology", invalid_generated)
    with pytest.raises(ValueError) as failure:
        smooth_mesh(_sphere())
    assert not isinstance(failure.value, MeshInputTopologyError)
    assert "passed input validation" in str(failure.value)
    assert (
        safe_mesh_input_failure(
            "smooth_mesh", type(failure.value).__name__, str(failure.value)
        )
        is None
    )


def test_unknown_provider_error_never_becomes_a_safe_topology_diagnosis(monkeypatch):
    import napari_vipp.core.mesh_refinement as module

    def unknown(*_args):
        raise ValueError("unreviewed provider C:/private/source.tif")

    monkeypatch.setattr(module, "_surface_contract", unknown)
    with pytest.raises(ValueError) as failure:
        module._input_surface_contract(
            np.zeros((3, 3)), np.array([[0, 1, 2]]), "Smooth Mesh", 1
        )
    assert not isinstance(failure.value, MeshInputTopologyError)
    assert (
        safe_mesh_input_failure(
            "smooth_mesh", type(failure.value).__name__, str(failure.value)
        )
        is None
    )


def test_exact_binary_saddle_remains_unrepaired_and_is_rejected_only_by_consumer():
    # The two coincident opposite-winding face pairs arise in Lewiner's binary
    # saddle handling, not from Combine Meshes or the subsequent smoothing.
    source = np.array(
        [
            [[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]],
            [[1, 1, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 0, 1]],
            [[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        ],
        dtype=bool,
    )
    source.setflags(write=False)
    axes = tuple(AxisMetadata(name, "space", "um", 0.45) for name in "zyx")
    state = image_state_from_array(source, axes=axes)
    expected_vertices, expected_faces, _, _ = marching_cubes(
        source.astype(np.float32),
        0.5,
        method="lewiner",
        step_size=1,
        allow_degenerate=False,
        gradient_direction="ascent",
    )
    mesh = mask_to_3d_mesh(source, "Leave open at image border", image_state=state)
    np.testing.assert_array_equal(mesh.vertices, expected_vertices)
    np.testing.assert_array_equal(mesh.faces, expected_faces)
    assert mesh.state.spatial_axes == state.axes
    before = mesh.vertices.copy(), mesh.faces.copy(), mesh.state
    count = len(expected_faces) - len(
        np.unique(np.sort(expected_faces, axis=1), axis=0)
    )
    # Accommodate a future provider fixing its own saddle triangulation; an
    # injected duplicate test above always exercises strict admission.
    if count:
        with pytest.raises(MeshInputTopologyError) as failure:
            smooth_mesh(mesh, iterations=3, strength=0.2)
        assert failure.value.code == "duplicate_faces" and failure.value.count == count
    np.testing.assert_array_equal(mesh.vertices, before[0])
    np.testing.assert_array_equal(mesh.faces, before[1])
    assert mesh.state == before[2]
    assert not source.flags.writeable and not mesh.vertices.flags.writeable
