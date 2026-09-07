"""Execute the shipped object-aware mesh example, including portable export."""

import importlib.util
import json
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest

from napari_vipp._sample_data import _mesh_morphology_sample
from napari_vipp.core.compute import ComputeRequest
from napari_vipp.core.execution import PipelineRunRequest, execute_pipeline_request
from napari_vipp.core.export import export_pipeline_to_python
from napari_vipp.core.meshes import MeshData, save_mesh_output
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.workflow import load_workflow, save_workflow, serialize_workflow
from napari_vipp.ui.examples import _example_workflow_by_id, _example_workflow_path

ROOT = Path(__file__).resolve().parents[3]


def _pipeline():
    spec = _example_workflow_by_id("mesh-objects")
    assert spec is not None
    path = _example_workflow_path(spec)
    document = load_workflow(path)
    pipeline = PrototypePipeline()
    pipeline.restore_graph(
        document["nodes"], document["connections"], document["output_tunnels"]
    )
    return pipeline, document


def test_mesh_objects_example_is_generated_and_readable_without_file_writers():
    generator_path = ROOT / "scripts" / "generate_mesh_objects_workflow.py"
    spec = importlib.util.spec_from_file_location(
        "mesh_example_generator", generator_path
    )
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    pipeline, positions, notes, metadata = generator.build_workflow()
    expected = serialize_workflow(
        pipeline, positions, notes, metadata, ComputeRequest(mode="cpu")
    )
    repository = ROOT / "examples" / generator.FILENAME
    packaged = ROOT / "src" / "napari_vipp" / "examples" / generator.FILENAME
    assert repository.read_bytes() == packaged.read_bytes()
    assert json.loads(repository.read_text()) == expected
    assert len(notes) == 3 and set(positions) == set(pipeline.nodes)
    assert not any(
        node.operation_id == "save_output" for node in pipeline.nodes.values()
    )
    for node in pipeline.nodes.values():
        if node.operation_id in {"input", "binary_threshold", "batch_output"}:
            continue
        assert pipeline.operation_spec(node.operation_id).execution_policy == "manual"


def test_mesh_objects_example_cpu_preserves_objects_and_exports_3mf(
    tmp_path, monkeypatch
):
    pipeline, _document = _pipeline()
    data, kwargs, _kind = _mesh_morphology_sample()
    before = data.copy()
    data.setflags(write=False)
    monkeypatch.chdir(tmp_path)
    result = execute_pipeline_request(
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
    assert not result.error and not result.cancelled
    outputs = result.pipeline.outputs
    original = outputs["mask_to_3d_mesh_1"]
    colored = outputs["color_mesh_objects_1"]
    large = outputs["filter_mesh_objects_1"]
    tiny = outputs["filter_mesh_objects_2"]
    combined = outputs["combine_meshes_1"]
    refined = outputs["simplify_mesh_1"]
    assert original.object_count == colored.object_count == combined.object_count == 5
    assert large.object_count == 4 and tiny.object_count == 1
    assert {item.object_id for item in large.objects}.isdisjoint(
        {item.object_id for item in tiny.objects}
    )
    assert [item.object_id for item in refined.objects] == [
        item.object_id for item in original.objects
    ]
    assert [item.color for item in combined.objects] == [
        item.color for item in colored.objects
    ]
    assert refined.objects == combined.objects
    assert len(refined.faces) < len(original.faces)
    assert all(axis.unit == "micrometer" for axis in refined.state.spatial_axes)
    assert [axis.scale for axis in refined.state.spatial_axes] == [2, 0.5, 0.5]
    table = outputs["measure_3d_mesh_morphology_1"]
    assert table.row_count == 5
    assert {row["mesh_id"] for row in table.records()} == {
        item.object_id for item in refined.objects
    }
    assert all(np.isfinite(row["mesh_volume_physical"]) for row in table.records())
    assert outputs["batch_output_1"] is refined
    np.testing.assert_array_equal(data, before)
    assert not list(tmp_path.iterdir())  # Batch Output declares; it does not write.
    path = save_mesh_output(refined, tmp_path / "five-objects.3mf", format="3mf")
    with zipfile.ZipFile(path) as archive:
        model = ElementTree.fromstring(archive.read("3D/3dmodel.model"))
    namespace = {"m": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}
    meshes = model.findall("m:resources/m:object/m:mesh", namespace)
    assert len(meshes) == 5
    assert model.attrib["unit"] == "micron"


def test_generated_python_runs_object_example_without_writing(tmp_path, monkeypatch):
    pipeline, _document = _pipeline()
    data, kwargs, _kind = _mesh_morphology_sample()
    namespace = {"__name__": "mesh_object_example_export"}
    exec(
        compile(export_pipeline_to_python(pipeline), "<mesh example>", "exec"),
        namespace,
    )
    monkeypatch.chdir(tmp_path)
    outputs = namespace["run_pipeline"](
        data, input_metadata=kwargs["metadata"], input_name=kwargs["name"]
    )
    assert isinstance(outputs["simplify_mesh_1"], MeshData)
    assert outputs["simplify_mesh_1"].object_count == 5
    assert outputs["measure_3d_mesh_morphology_1"].row_count == 5
    # Shared exported execution may persist timing history (isolated here by
    # the test fixture); no image, mesh, table or workflow output is published.
    assert {path.name for path in tmp_path.iterdir()} <= {".napari-vipp-test-state"}


@pytest.mark.parametrize("mode", ["cpu", "auto", "prefer_gpu"])
def test_labels_to_mesh_shared_execution_roundtrip_and_cached_measurement(
    mode,
    tmp_path,
    monkeypatch,
):
    """Exercise the label producer/mesh consumer, not just the direct helper."""
    from skimage import measure

    from napari_vipp.core import meshes
    from napari_vipp.core.meshes import MeshState

    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    pipeline.nodes["input"].params.update(
        source_mode="sample",
        sample_name="VIPP synthetic 3D mesh morphology",
    )
    threshold = pipeline.add_node("binary_threshold")
    threshold.params["threshold"] = 1000
    labels = pipeline.add_node("label_connected_components")
    labels.params["connectivity"] = "Face connected"
    surface = pipeline.add_node("labels_to_3d_mesh")
    measurements = pipeline.add_node("measure_3d_mesh_morphology")
    measurements.params["include_convex_hull_metrics"] = False
    for source, target in (
        ("input", threshold.id),
        (threshold.id, labels.id),
        (labels.id, surface.id),
        (surface.id, measurements.id),
    ):
        assert pipeline.connect(source, target).success
    data, kwargs, _kind = _mesh_morphology_sample()
    original_data = data.copy()
    data.setflags(write=False)
    request = dict(
        input_data=data,
        input_metadata=kwargs["metadata"],
        input_name=kwargs["name"],
        source_payloads={},
        compute_request=ComputeRequest(mode=mode),
    )
    first = execute_pipeline_request(
        PipelineRunRequest(
            run_id=1,
            workflow=serialize_workflow(pipeline),
            **request,
            manual_node_ids=frozenset(pipeline.manual_node_ids()) - {measurements.id},
        ),
        raise_errors=True,
    )
    assert not first.error and not first.cancelled
    cached = first.pipeline
    mesh = cached.outputs[surface.id]
    assert isinstance(mesh, MeshData)
    assert isinstance(cached.output_states[surface.id], MeshState)
    assert cached.output_states[surface.id] is mesh.state
    label_ids = {int(value) for value in np.unique(cached.outputs[labels.id]) if value}
    assert label_ids == set(range(1, 6))
    assert {item.object_id for item in mesh.objects} == label_ids
    assert {item.source_object_id for item in mesh.objects} == label_ids
    assert set(mesh.face_object_ids) == label_ids
    assert mesh.state.object_count == 5
    assert tuple(axis.scale for axis in mesh.state.spatial_axes) == (2.0, 0.5, 0.5)
    assert all(axis.unit == "micrometer" for axis in mesh.state.spatial_axes)
    assert cached.outputs[measurements.id] is None
    assert (
        next(
            decision
            for decision in first.execution_report.actual_decisions
            if decision.node_id == surface.id
        ).runtime_id
        == "cpu-numpy"
    )

    saved = save_workflow(
        tmp_path / "labels-to-mesh.json",
        cached,
        compute_request=ComputeRequest(mode=mode),
    )
    document = load_workflow(saved)
    restored = PrototypePipeline()
    restored.restore_graph(
        document["nodes"], document["connections"], document["output_tunnels"]
    )
    assert restored.nodes[surface.id].params == cached.nodes[surface.id].params
    assert restored.nodes[surface.id].operation_id == "labels_to_3d_mesh"

    def unexpected_remeshing(*args, **kwargs):
        raise AssertionError(
            "Cached mesh measurements must not run marching cubes again."
        )

    monkeypatch.setattr(meshes, "marching_cubes", unexpected_remeshing)
    monkeypatch.setattr(measure, "marching_cubes", unexpected_remeshing)
    started = []
    result = execute_pipeline_request(
        PipelineRunRequest(
            run_id=2,
            workflow=serialize_workflow(restored),
            **request,
            manual_node_ids=frozenset({measurements.id}),
            dirty_node_ids=frozenset({measurements.id}),
            completed_node_ids=frozenset(cached.completed_node_ids),
            cached_outputs=dict(cached.outputs),
            cached_output_states=dict(cached.output_states),
            cached_node_outputs=dict(cached.node_outputs),
            cached_node_output_states=dict(cached.node_output_states),
            cached_execution_states=dict(cached.node_execution_states),
            cached_execution_messages=dict(cached.node_execution_messages),
            cached_compute_provenance={
                **cached.node_cache_lineage,
                **cached.node_compute_provenance,
            },
        ),
        raise_errors=True,
        node_started_callback=started.append,
    )
    assert not result.error and not result.cancelled
    assert started == [measurements.id]
    assert result.pipeline.outputs[surface.id] is mesh
    table = result.pipeline.outputs[measurements.id]
    assert table.row_count == 5
    assert {row["mesh_id"] for row in table.records()} == label_ids
    assert all(row["mesh_status"] == "ok" for row in table.records())
    assert "no remeshing" in result.pipeline.output_states[measurements.id].history[-1]
    assert (
        next(
            decision
            for decision in result.execution_report.actual_decisions
            if decision.node_id == measurements.id
        ).runtime_id
        == "cpu-numpy"
    )
    np.testing.assert_array_equal(data, original_data)
