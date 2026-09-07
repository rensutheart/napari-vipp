"""Refinement errors distinguish upstream defects from generated bad geometry."""

from dataclasses import replace

import numpy as np
import pytest

from napari_vipp._tests.test_mesh_refinement import _plane, _sphere
from napari_vipp.core.mesh_refinement import simplify_mesh, smooth_mesh
from napari_vipp.core.meshes import MeshData, MeshObject


@pytest.mark.parametrize("refine", [smooth_mesh, simplify_mesh])
def test_invalid_input_error_does_not_blame_refinement_settings(refine):
    mesh = _plane()
    mesh = MeshData(
        mesh.vertices, np.concatenate((mesh.faces, mesh.faces[:1, ::-1])), mesh.state
    )
    before = mesh.vertices.copy(), mesh.faces.copy(), mesh.state
    with pytest.raises(ValueError) as failure:
        refine(mesh)
    message = str(failure.value)
    assert "object 1 has an invalid input mesh" in message
    assert "duplicate triangles (1 repeated face" in message
    assert "Changing this node's settings will not repair the input" in message
    assert "upstream mesh" in message
    assert "No new mesh was produced" in message
    np.testing.assert_array_equal(mesh.vertices, before[0])
    np.testing.assert_array_equal(mesh.faces, before[1])
    assert mesh.state == before[2]


def test_generated_duplicates_identify_object_settings_and_remedy(monkeypatch):
    import napari_vipp.core.mesh_refinement as module

    mesh = _sphere()
    mesh = replace(
        mesh,
        face_object_ids=np.full(len(mesh.faces), 17, np.int64),
        objects=(MeshObject(17, "Test object"),),
    )
    before = mesh.vertices.copy(), mesh.faces.copy(), mesh.state

    def duplicates(points, faces, *_args):
        return points, np.concatenate((faces[:-1], faces[:1, ::-1]))

    monkeypatch.setattr(module, "_simplify_component", duplicates)
    with pytest.raises(ValueError) as failure:
        simplify_mesh(mesh, target_percent=10, aggressiveness=5)
    message = str(failure.value)
    assert "Simplify Mesh: object 17, component 1 passed input validation" in message
    assert "keeping 10% of triangles (Aggressiveness 5)" in message
    assert "produced an invalid result: duplicate triangles" in message
    assert "Increase Triangles to keep (%)" in message
    assert "lower Aggressiveness" in message
    assert "No new mesh was produced; the input is unchanged" in message
    np.testing.assert_array_equal(mesh.vertices, before[0])
    np.testing.assert_array_equal(mesh.faces, before[1])
    assert mesh.state == before[2]


def test_smoothing_output_failure_gives_smoothing_controls(monkeypatch):
    import napari_vipp.core.mesh_refinement as module

    def failed(*_args):
        raise ValueError("refinement collapsed or inverted an enclosed surface")

    monkeypatch.setattr(module, "_check_refined_topology", failed)
    with pytest.raises(ValueError) as failure:
        smooth_mesh(_sphere(), iterations=50, strength=0.75)
    message = str(failure.value)
    assert "Smooth Mesh: object 1, component 1 passed input validation" in message
    assert "Strength 0.75, Iterations 50" in message
    assert "Reduce Strength or Iterations" in message
    assert "Aggressiveness" not in message


def test_bundled_example_low_reduction_is_valid_or_reports_generated_defect():
    from napari_vipp._sample_data import _mesh_morphology_sample
    from napari_vipp._tests.test_mesh_objects_example import _pipeline
    from napari_vipp.core.compute import ComputeRequest
    from napari_vipp.core.execution import PipelineRunRequest, execute_pipeline_request
    from napari_vipp.core.mesh_objects import iter_mesh_objects
    from napari_vipp.core.mesh_refinement import _surface_contract
    from napari_vipp.core.workflow import serialize_workflow

    pipeline, _document = _pipeline()
    data, kwargs, _kind = _mesh_morphology_sample()
    run = execute_pipeline_request(
        PipelineRunRequest(
            run_id=1,
            workflow=serialize_workflow(pipeline),
            input_data=data,
            input_metadata=kwargs["metadata"],
            input_name=kwargs["name"],
            source_payloads={},
            compute_request=ComputeRequest(mode="cpu"),
            manual_node_ids=frozenset(pipeline.manual_node_ids()),
        ),
        raise_errors=True,
    )
    mesh = run.pipeline.outputs["smooth_mesh_1"]
    for part in iter_mesh_objects(mesh):
        _surface_contract(part.vertices, part.faces)
    # fast-simplification 0.2.x introduces repeated faces in object 4 at 10%.
    # Allow a future provider to succeed, but never accept an invalid surface
    # or misreport this as an upstream problem. Injection above covers the error
    # deterministically across provider versions.
    try:
        reduced = simplify_mesh(mesh, target_percent=10, aggressiveness=5)
    except ValueError as exc:
        assert "passed input validation" in str(exc)
        assert "keeping 10% of triangles (Aggressiveness 5)" in str(exc)
        assert "Increase Triangles to keep (%)" in str(exc)
    else:
        for part in iter_mesh_objects(reduced):
            _surface_contract(part.vertices, part.faces)
    safer = simplify_mesh(mesh, target_percent=20, aggressiveness=5)
    assert safer.object_count == mesh.object_count == 5
    for part in iter_mesh_objects(safer):
        _surface_contract(part.vertices, part.faces)
