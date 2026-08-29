from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import zarr

from napari_vipp.core.io import read_image_exact_window
from napari_vipp.core.io.ome_zarr import read_ome_zarr_presentation_preview
from napari_vipp.core.io.registry import inspect_image_source
from napari_vipp.core.source_window import SourceWindowRequest
from napari_vipp.core.workflow import WORKFLOW_VERSION, load_workflow

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_source_window_acceptance.py"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "source_window_acceptance_generator", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sparse_store_is_deterministic_valid_and_tiny_on_disk(tmp_path) -> None:
    script = _load_script()
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_manifest = script.write_acceptance_bundle(first)
    second_manifest = script.write_acceptance_bundle(second)

    assert first_manifest["decoded_bytes"] == 64 * 1024**3
    assert first_manifest["shape"] == [512, 8192, 8192]
    assert first_manifest["chunks"] == [1, 256, 256]
    assert first_manifest["dtype"] == "uint16"
    assert first_manifest["schema_version"] == 2
    assert first_manifest["written_chunk_count"] == 312
    assert first_manifest["store_bytes"] < 50 * 1024**2
    assert first_manifest["presentation_level"] == {
        "path": "1",
        "shape": [17, 513, 513],
        "chunks": [17, 513, 513],
        "scale_micrometers": [40.0, 4.0, 4.0],
        "decoded_bytes": 17 * 513 * 513 * 2,
    }
    assert first_manifest["store_tree_sha256"] == second_manifest["store_tree_sha256"]
    assert script.tree_sha256(first / script.DATASET_NAME) == script.tree_sha256(
        second / script.DATASET_NAME
    )

    root = zarr.open_group(str(first / script.DATASET_NAME), mode="r")
    array = root["0"]
    assert array.shape == (512, 8192, 8192)
    assert array.chunks == (1, 256, 256)
    assert array.dtype == np.dtype("uint16")
    assert int(array[0, 0, 0]) == 0
    assert int(array[256, 4096, 4096]) == 65000
    assert int(array[256, 4096, 4396]) == 32000
    # The analysis level must contain the same full-frame fibres promised by
    # the presentation level, including outside the old central 4x4 chunk box.
    assert int(array[256, 3722, 3265]) == 44000
    assert int(array[256, 4700, 3290]) == 36000
    preview = root["1"]
    assert preview.shape == (17, 513, 513)
    preview_data = np.asarray(preview[:])
    assert int(preview_data[8, 256, 256]) == 65000
    assert tuple(np.flatnonzero(np.any(preview_data != 0, axis=(1, 2))).tolist()) == (
        7,
        8,
        9,
    )
    assert np.count_nonzero(np.max(preview_data, axis=0)) == 1430


def test_generated_source_supports_bounded_exact_window_reads(tmp_path) -> None:
    script = _load_script()
    output = tmp_path / "bundle"
    script.write_acceptance_bundle(output)
    source = output / script.DATASET_NAME

    inspection = inspect_image_source(source)
    series = inspection.series[0]
    assert series.shape == (512, 8192, 8192)
    assert series.dtype == "uint16"
    assert series.axes == "ZYX"
    assert "exact_region_read" in series.capabilities

    request = SourceWindowRequest(
        (slice(254, 259), slice(3840, 4352), slice(3840, 4352))
    )
    result = read_image_exact_window(source, request=request)

    assert result.data.shape == (5, 512, 512)
    assert result.data.nbytes == 5 * 512 * 512 * 2
    assert result.data.flags.owndata
    assert not result.data.flags.writeable
    assert int(result.data.max()) == 65000
    assert result.image_state.axis_order == "ZYX"
    assert [axis.scale for axis in result.image_state.axes] == [1.25, 0.25, 0.25]
    assert [axis.translation for axis in result.image_state.axes] == [
        254 * 1.25,
        3840 * 0.25,
        3840 * 0.25,
    ]


def test_expanded_asymmetric_window_reveals_continuous_fibres(tmp_path) -> None:
    script = _load_script()
    output = tmp_path / "bundle"
    script.write_acceptance_bundle(output)
    source = output / script.DATASET_NAME

    request = SourceWindowRequest(
        (slice(255, 258), slice(3665, 4860), slice(3265, 4526))
    )
    result = read_image_exact_window(source, request=request)
    projection = np.max(result.data, axis=0)

    assert result.data.shape == (3, 1195, 1261)
    assert np.any(projection[:, 0] != 0)
    assert np.any(projection[943:, :] != 0)  # Global Y >= 4608.
    assert int(result.data.max()) == 65000


def test_generated_source_has_one_bounded_presentation_chunk(tmp_path) -> None:
    script = _load_script()
    output = tmp_path / "bundle"
    script.write_acceptance_bundle(output)

    result = read_ome_zarr_presentation_preview(output / script.DATASET_NAME)

    assert result.preview_level == 1
    assert result.data.shape == (17, 513, 513)
    assert result.data.nbytes < 9 * 1024**2
    assert int(result.data.max()) == 65000
    assert result.metrics.estimated_objects_read == 1
    assert "analysis remains full resolution" in result.message


def test_generated_workflow_is_crop_free_and_documents_one_click_repair(
    tmp_path,
) -> None:
    script = _load_script()
    output = tmp_path / "bundle"
    script.write_acceptance_bundle(output)

    workflow_path = output / script.WORKFLOW_NAME
    raw = json.loads(workflow_path.read_text(encoding="utf-8"))
    restored = load_workflow(workflow_path)
    by_id = {node.id: node for node in restored["nodes"]}

    assert raw["version"] == WORKFLOW_VERSION
    assert set(by_id) == {"input", "project_image_1"}
    assert by_id["input"].params["source_mode"] == "file path"
    assert Path(by_id["input"].params["file_path"]) == (output / script.DATASET_NAME)
    assert all(node.operation_id != "crop_stack" for node in by_id.values())
    assert [
        (connection.source_id, connection.target_id)
        for connection in restored["connections"]
    ] == [("input", "project_image_1")]
    assert by_id["project_image_1"].params == {
        "axes": "non_yx_spatial",
        "method": "Maximum",
    }

    note_text = " ".join(note["text"] for note in raw["notes"])
    assert "64 GiB" in note_text
    assert "add a fitted Crop Stack" in note_text
    assert "not a content-aware scientific choice" in note_text
    assert "instead of silently reading the complete level 0" in note_text
    readme = (output / script.README_NAME).read_text(encoding="utf-8")
    assert "intentionally contains no Crop Stack" in readme
    assert "level-1 preview is not blank" in readme
    assert "layer visibility/selection problem" in readme
    assert str(workflow_path) in readme


def test_existing_bundle_requires_explicit_force(tmp_path) -> None:
    script = _load_script()
    output = tmp_path / "bundle"
    first = script.write_acceptance_bundle(output)

    try:
        script.write_acceptance_bundle(output)
    except FileExistsError as error:
        assert "--force" in str(error)
    else:  # pragma: no cover - explicit assertion message
        raise AssertionError("Existing output was replaced without explicit force.")

    second = script.write_acceptance_bundle(output, force=True)
    assert first["store_tree_sha256"] == second["store_tree_sha256"]
