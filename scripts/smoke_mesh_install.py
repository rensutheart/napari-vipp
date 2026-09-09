"""Exercise the installed CPU mesh stack and packaged example without Qt or CUDA."""

from __future__ import annotations

import argparse
import json
import platform
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from importlib.metadata import version
from importlib.resources import as_file, files
from pathlib import Path

import numpy as np

import napari_vipp
from napari_vipp._sample_data import _mesh_morphology_sample
from napari_vipp.core.compute import ComputeRequest
from napari_vipp.core.execution import PipelineRunRequest, execute_pipeline_request
from napari_vipp.core.meshes import save_mesh_output
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.workflow import load_workflow


def smoke_mesh_install(*, require_installed=False):
    """Fail on missing native dependencies, altered object identity or bad exports."""
    installed = Path(napari_vipp.__file__).resolve()
    repository = Path(__file__).resolve().parents[1]
    if require_installed and repository in installed.parents:
        raise RuntimeError(f"Smoke imported the source checkout: {installed}")
    resource = files("napari_vipp").joinpath(
        "examples", "synthetic-mesh-objects.json"
    )
    with as_file(resource) as path:
        graph = load_workflow(path)
        document = json.loads(path.read_text(encoding="utf-8"))
    pipeline = PrototypePipeline()
    pipeline.restore_graph(
        graph["nodes"], graph["connections"], graph["output_tunnels"]
    )
    source, kwargs, _kind = _mesh_morphology_sample()
    original = source.copy()
    source.setflags(write=False)
    result = execute_pipeline_request(
        PipelineRunRequest(
            run_id=1,
            workflow=document,
            input_data=source,
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
    for node_id, count in (
        ("mask_to_3d_mesh_1", 1),
        ("split_mesh_objects_1", 5),
        ("filter_mesh_objects_1", 4),
        ("filter_mesh_objects_2", 1),
        ("combine_meshes_1", 5),
        ("simplify_mesh_1", 5),
    ):
        assert outputs[node_id].object_count == count, node_id
    combined = outputs["combine_meshes_1"]
    mesh = outputs["simplify_mesh_1"]
    assert mesh.objects == combined.objects
    assert 0 < len(mesh.faces) < len(combined.faces)
    assert len({obj.color for obj in mesh.objects}) > 1
    assert [axis.scale for axis in mesh.state.spatial_axes] == [2, 0.5, 0.5]
    assert all(axis.unit == "micrometer" for axis in mesh.state.spatial_axes)
    table = outputs["measure_3d_mesh_morphology_1"]
    assert table.row_count == 5
    assert {row["mesh_id"] for row in table.records()} == {
        obj.object_id for obj in mesh.objects
    }
    assert all(row["mesh_status"] == "ok" for row in table.records())
    assert outputs["batch_output_1"] is mesh
    np.testing.assert_array_equal(source, original)

    with tempfile.TemporaryDirectory(prefix="vipp-mesh-smoke-") as directory:
        root = Path(directory)
        obj_path = save_mesh_output(mesh, root / "sample.obj")
        obj_lines = obj_path.read_text(encoding="utf-8").splitlines()
        assert sum(line.startswith("v ") for line in obj_lines) == len(mesh.vertices)
        assert sum(line.startswith("f ") for line in obj_lines) == len(mesh.faces)
        package = save_mesh_output(mesh, root / "sample.3mf")
        with zipfile.ZipFile(package) as archive:
            assert archive.testzip() is None
            model = ET.fromstring(archive.read("3D/3dmodel.model"))
        ns = {"c": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}
        assert model.attrib["unit"] == "micron"
        objects = model.findall("c:resources/c:object[c:mesh]", ns)
        assert len(objects) == mesh.object_count
        assert {node.attrib["partnumber"] for node in objects} == {
            f"VIPP-{obj.object_id}" for obj in mesh.objects
        }
        materials = model.findall("c:resources/c:basematerials/c:base", ns)
        assert len(materials) == mesh.object_count
        assert all(node.attrib["displaycolor"].startswith("#") for node in materials)
        assert sum(
            len(node.findall("c:mesh/c:triangles/c:triangle", ns)) for node in objects
        ) == len(mesh.faces)

    return {
        "status": "passed",
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "installed_module": str(installed),
        "versions": {
            name: version(name)
            for name in ("napari-vipp", "numpy", "scikit-image", "fast-simplification")
        },
        "objects": mesh.object_count,
        "triangles_before": len(combined.faces),
        "triangles_after": len(mesh.faces),
        "measurement_rows": table.row_count,
        "exports": ["obj", "3mf"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-installed", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            smoke_mesh_install(require_installed=args.require_installed), indent=2
        )
    )
