"""Generated convenience runners must not discard authored 3MF output choices."""

from pathlib import Path
from zipfile import is_zipfile

import numpy as np
import pytest
import tifffile

from napari_vipp.core.export import export_pipeline_to_python
from napari_vipp.core.pipeline import PrototypePipeline


def _export(formats):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    threshold = pipeline.add_node("binary_threshold")
    threshold.params["threshold"] = 0.5
    mesh = pipeline.add_node("mask_to_3d_mesh")
    mesh.params["object_mode"] = "Connected objects"
    assert pipeline.connect("input", threshold.id).success
    assert pipeline.connect(threshold.id, mesh.id).success
    outputs = []
    for format in formats:
        output = pipeline.add_node("batch_output")
        output.params["format"] = format
        assert pipeline.connect(mesh.id, output.id).success
        outputs.append(output.id)
    namespace = {"__name__": "mesh_export_format_review"}
    exec(
        compile(export_pipeline_to_python(pipeline), "<mesh-export-format>", "exec"),
        namespace,
    )
    return namespace, outputs, pipeline


def _source(directory):
    directory.mkdir()
    data = np.zeros((7, 8, 9), np.uint8)
    data[1:3, 1:4, 1:4] = 1
    data[4:6, 4:7, 5:8] = 1
    source = directory / "sample.ome.tif"
    tifffile.imwrite(
        source,
        data,
        photometric="minisblack",
        metadata={
            "axes": "ZYX",
            "PhysicalSizeZ": 2,
            "PhysicalSizeZUnit": "µm",
            "PhysicalSizeY": 1,
            "PhysicalSizeYUnit": "µm",
            "PhysicalSizeX": 1,
            "PhysicalSizeXUnit": "µm",
        },
    )
    return source


@pytest.mark.parametrize(
    "authored,expected", [("3mf", ".3mf"), ("obj", ".obj"), ("batch default", ".obj")]
)
def test_generated_folder_loop_preserves_mesh_output_format(
    tmp_path, authored, expected
):
    namespace, outputs, _pipeline = _export((authored,))
    source = _source(tmp_path / "inputs")
    output_dir = tmp_path / "outputs"
    with pytest.warns(FutureWarning, match="convenience loop"):
        records = namespace["batch_process"](source.parent, output_dir)
    paths = [Path(path) for path in records[0]["saved_paths"]]
    assert paths == [output_dir / f"{source.stem}__{outputs[0]}{expected}"]
    assert is_zipfile(paths[0]) == (expected == ".3mf")
    assert len(list(output_dir.glob("*.vipp-provenance.json"))) == 1
    assert not list(output_dir.glob(".vipp-publish-*"))


def test_generated_multi_output_cli_keeps_each_mesh_format(tmp_path):
    namespace, outputs, _pipeline = _export(("3mf", "obj"))
    source = _source(tmp_path / "inputs")
    output_dir = tmp_path / "outputs"
    result = namespace["main"]([str(source), str(output_dir), "--no-provenance"])
    assert result == 0
    three_mf = output_dir / f"{source.stem}__{outputs[0]}.3mf"
    obj = output_dir / f"{source.stem}__{outputs[1]}.obj"
    assert is_zipfile(three_mf)
    assert "o VIPP_object_1" in obj.read_text(encoding="utf-8")
    assert sorted(output_dir.iterdir()) == sorted([three_mf, obj])


@pytest.mark.parametrize(
    "requested,expected", [("mesh", "mesh.3mf"), ("mesh.obj", "mesh.obj")]
)
def test_generated_single_cli_uses_authored_format_unless_path_is_explicit(
    tmp_path, requested, expected
):
    namespace, _outputs, _pipeline = _export(("3mf",))
    source = _source(tmp_path / "inputs")
    result = namespace["main"](
        [str(source), str(tmp_path / requested), "--no-provenance"]
    )
    assert result == 0
    assert (tmp_path / expected).is_file()
    assert is_zipfile(tmp_path / expected) == expected.endswith(".3mf")


def test_generated_mesh_auto_names_reject_an_incompatible_marker_format(tmp_path):
    namespace, outputs, _pipeline = _export(("tiff",))
    with pytest.raises(ValueError, match="requires OBJ or 3MF"):
        namespace["_mesh_output_format"](outputs[0])
    assert namespace["_mesh_output_format"](None) == "obj"


@pytest.mark.parametrize("name", ["_automatic_output_path", "_mesh_output_format"])
def test_generated_mesh_helpers_are_reserved_function_names(name):
    _namespace, _outputs, pipeline = _export(("3mf",))
    with pytest.raises(ValueError, match="Invalid exported function name"):
        export_pipeline_to_python(pipeline, function_name=name)
