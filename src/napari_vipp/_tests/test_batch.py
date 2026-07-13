from __future__ import annotations

import errno
import json
import os
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import tifffile

import napari_vipp.core.batch as batch_module
from napari_vipp.core.batch import (
    BATCH_CONFIG_FILENAME,
    BATCH_CONFIG_TYPE,
    BATCH_CONFIG_VERSION,
    BATCH_MANIFEST_FILENAME,
    BATCH_MANIFEST_TYPE,
    BATCH_MANIFEST_VERSION,
    BatchConfig,
    BatchOutputConfig,
    BatchSourceConfig,
    BatchStatus,
    ExistingFilePolicy,
    atomic_write_json,
    batch_config_hash,
    build_batch_plan,
    load_batch_config,
    run_batch,
    run_batch_from_files,
    safe_batch_filename,
    save_batch_config,
    scientific_workflow_hash,
    validate_batch_config,
)
from napari_vipp.core.io import write_image
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.workflow import serialize_workflow


def _batch_workflow(
    output_specs: tuple[dict[str, str], ...] = (
        {
            "tag": "output",
            "format": "npy",
            "subfolder": "",
            "filename_template": "{source_stem}__{tag}",
            "overwrite": "batch default",
        },
    ),
) -> tuple[dict[str, object], tuple[str, ...]]:
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    output_ids: list[str] = []
    for spec in output_specs:
        output = pipeline.add_node("batch_output")
        assert pipeline.connect("input", output.id).success
        for name, value in spec.items():
            pipeline.set_param(output.id, name, value)
        output_ids.append(output.id)
    return serialize_workflow(pipeline), tuple(output_ids)


def _batch_config(
    workflow: dict[str, object],
    input_dir: Path,
    output_dir: Path,
    output_ids: tuple[str, ...],
    *,
    base_dir: Path | None = None,
    policy: ExistingFilePolicy = ExistingFilePolicy.ERROR,
    continue_on_error: bool = True,
) -> BatchConfig:
    nodes = {str(node["id"]): node for node in workflow["nodes"]}
    outputs = []
    for node_id in output_ids:
        params = nodes[node_id]["params"]
        outputs.append(
            BatchOutputConfig(
                node_id=node_id,
                node_title="Batch Output",
                tag=str(params["tag"]),
                kind="image",
                format=str(params["format"]),
                subfolder=str(params["subfolder"]),
                filename_template=str(params["filename_template"]),
                overwrite=str(params["overwrite"]),
            )
        )
    return BatchConfig(
        workflow_file=Path("workflow.json"),
        workflow_sha256=scientific_workflow_hash(workflow),
        output_dir=output_dir,
        sources=(
            BatchSourceConfig(
                node_id="input",
                title="Image Source",
                input_dir=input_dir,
                pattern="*.npy",
            ),
        ),
        outputs=tuple(outputs),
        default_image_format="npy",
        existing_file_policy=policy,
        save_workflow_snapshot=True,
        save_python_script=True,
        continue_on_error=continue_on_error,
        base_dir=base_dir,
    )


def _write_arrays(folder: Path, **arrays: np.ndarray) -> None:
    folder.mkdir(parents=True)
    for filename, data in arrays.items():
        np.save(folder / filename, data)


def test_batch_config_roundtrip_preserves_schema_and_resolves_relative_paths(
    tmp_path,
):
    workflow, output_ids = _batch_workflow()
    config = _batch_config(
        workflow,
        Path("inputs"),
        Path("outputs"),
        output_ids,
        base_dir=tmp_path,
        policy=ExistingFilePolicy.SKIP,
    )
    path = tmp_path / BATCH_CONFIG_FILENAME

    assert save_batch_config(path, config) == path
    loaded = load_batch_config(path)

    assert loaded.to_dict() == config.to_dict()
    assert loaded.base_dir == tmp_path.resolve()
    assert (
        loaded.resolve_path(loaded.sources[0].input_dir)
        == (tmp_path / "inputs").resolve()
    )
    assert loaded.resolve_path(loaded.output_dir) == (tmp_path / "outputs").resolve()
    assert loaded.existing_file_policy == ExistingFilePolicy.SKIP
    assert batch_config_hash(loaded) == batch_config_hash(config)
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["type"] == BATCH_CONFIG_TYPE
    assert document["version"] == BATCH_CONFIG_VERSION


def test_atomic_json_write_retries_transient_windows_replace_lock(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "record.json"
    original_replace = batch_module.os.replace
    attempts = 0

    def transient_replace(source, destination):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError("simulated transient lock")
        return original_replace(source, destination)

    monkeypatch.setattr(batch_module.os, "replace", transient_replace)
    monkeypatch.setattr(batch_module.time, "sleep", lambda _delay: None)

    assert atomic_write_json(target, {"ok": True}) == target
    assert attempts == 2
    assert json.loads(target.read_text(encoding="utf-8")) == {"ok": True}


def test_atomic_json_write_normalizes_non_finite_values(tmp_path):
    target = tmp_path / "non-finite.json"

    atomic_write_json(
        target,
        {"nan": float("nan"), "positive": float("inf"), "negative": -float("inf")},
    )

    assert json.loads(target.read_text(encoding="utf-8")) == {
        "nan": None,
        "positive": None,
        "negative": None,
    }


def test_saving_loaded_relative_config_elsewhere_preserves_resolved_paths(
    tmp_path,
):
    workflow, output_ids = _batch_workflow()
    original_dir = tmp_path / "original"
    moved_dir = tmp_path / "moved"
    original_dir.mkdir()
    config = _batch_config(
        workflow,
        Path("inputs"),
        Path("outputs"),
        output_ids,
        base_dir=original_dir,
    )
    original_path = original_dir / BATCH_CONFIG_FILENAME
    save_batch_config(original_path, config)
    loaded = load_batch_config(original_path)
    moved_path = moved_dir / BATCH_CONFIG_FILENAME

    save_batch_config(moved_path, loaded)
    moved = load_batch_config(moved_path)

    assert moved.resolve_path(moved.sources[0].input_dir) == (
        original_dir / "inputs"
    ).resolve()
    assert moved.resolve_path(moved.output_dir) == (
        original_dir / "outputs"
    ).resolve()


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("type", "not a napari-vipp batch config"),
        ("version", "Unsupported batch config version"),
        ("boolean version", "Unsupported batch config version"),
        ("hash", "lowercase SHA-256"),
        ("unknown", "unknown fields"),
        ("duplicate source", "Duplicate batch source node ids"),
        ("duplicate output", "Duplicate batch output node ids"),
    ],
)
def test_batch_config_strictly_rejects_invalid_documents(tmp_path, case, message):
    workflow, output_ids = _batch_workflow()
    document = _batch_config(
        workflow,
        Path("inputs"),
        Path("outputs"),
        output_ids,
    ).to_dict()
    if case == "type":
        document["type"] = "some-other-file"
    elif case == "version":
        document["version"] = BATCH_CONFIG_VERSION + 1
    elif case == "boolean version":
        document["version"] = True
    elif case == "hash":
        document["workflow"]["sha256"] = "A" * 64
    elif case == "unknown":
        document["unexpected"] = True
    elif case == "duplicate source":
        document["sources"].append(deepcopy(document["sources"][0]))
    else:
        document["outputs"].append(deepcopy(document["outputs"][0]))

    path = tmp_path / f"invalid-{case}.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_batch_config(path)


def test_scientific_workflow_hash_ignores_layout_notes_and_ui_metadata():
    pipeline = PrototypePipeline()
    plain = serialize_workflow(pipeline)
    decorated = serialize_workflow(
        pipeline,
        positions={
            node_id: (index * 10.0, index * -5.0)
            for index, node_id in enumerate(pipeline.nodes)
        },
        notes=(
            {
                "id": "note-1",
                "text": "presentation only",
                "position": (12.0, 34.0),
                "width": 280.0,
                "attached_node": "gaussian",
            },
        ),
        metadata={
            "vipp": {
                "inspector": {
                    "selected_node_id": "threshold",
                    "right_panel_visible": False,
                },
                "thumbnails": {"disabled_node_ids": ["gaussian"]},
            }
        },
    )

    assert scientific_workflow_hash(decorated) == scientific_workflow_hash(plain)

    pipeline.set_param("gaussian", "sigma", 3.25)
    changed = serialize_workflow(pipeline)
    assert scientific_workflow_hash(changed) != scientific_workflow_hash(plain)


@pytest.mark.parametrize("value", ["CON", "nul.txt", "LPT1", "com9.npy"])
def test_safe_batch_filename_avoids_windows_device_names(value):
    assert safe_batch_filename(value).startswith("_")


def test_build_batch_plan_pairs_sources_by_sorted_position_and_names_exactly(
    tmp_path,
):
    primary = tmp_path / "primary"
    secondary = tmp_path / "secondary"
    _write_arrays(
        primary,
        **{
            "field_b.npy": np.full((3, 4), 2, dtype=np.uint8),
            "field_a.npy": np.full((3, 4), 1, dtype=np.uint8),
        },
    )
    _write_arrays(
        secondary,
        **{
            "mask_z.npy": np.full((3, 4), 20, dtype=np.uint8),
            "mask_y.npy": np.full((3, 4), 10, dtype=np.uint8),
        },
    )
    config = BatchConfig(
        workflow_file=Path("workflow.json"),
        workflow_sha256="0" * 64,
        output_dir=Path("results"),
        sources=(
            BatchSourceConfig("primary", "Primary", Path("primary"), "*.npy"),
            BatchSourceConfig("secondary", "Secondary", Path("secondary"), "*.npy"),
        ),
        outputs=(
            BatchOutputConfig(
                "raw-output",
                "Raw Result",
                "raw image",
                "image",
                "batch default",
                "images/processed",
                "{batch_index}_{source_stem}_{tag}",
            ),
            BatchOutputConfig(
                "qc-output",
                "QC Result",
                "qc",
                "image",
                "tiff",
                "qc",
                "{batch_id}_{node_title}",
            ),
        ),
        default_image_format="npy",
        base_dir=tmp_path,
    )

    plan = build_batch_plan(config)

    assert plan.output_dir == (tmp_path / "results").resolve()
    assert [item.batch_id for item in plan.items] == [
        "0001_field_a",
        "0002_field_b",
    ]
    assert plan.items[0].source_paths == {
        "primary": (primary / "field_a.npy").resolve(),
        "secondary": (secondary / "mask_y.npy").resolve(),
    }
    assert [output.format for output in plan.items[0].outputs] == ["npy", "tiff"]
    assert [
        output.path.relative_to(plan.output_dir).as_posix()
        for output in plan.items[0].outputs
    ] == [
        "images/processed/0001_field_a_raw_image.npy",
        "qc/0001_field_a_QC_Result.tif",
    ]
    assert plan.output_count == 4
    assert not plan.has_collisions


def test_build_batch_plan_detects_duplicate_and_existing_destinations(tmp_path):
    inputs = tmp_path / "inputs"
    _write_arrays(inputs, field=np.ones((2, 3), dtype=np.uint8))
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    existing = output_dir / "already.npy"
    np.save(existing, np.zeros((1,), dtype=np.uint8))
    config = BatchConfig(
        workflow_file=Path("workflow.json"),
        workflow_sha256="0" * 64,
        output_dir=output_dir,
        sources=(BatchSourceConfig("input", "Input", inputs, "*.npy"),),
        outputs=(
            BatchOutputConfig(
                "first",
                "First",
                "first",
                "image",
                "npy",
                "",
                "already",
            ),
            BatchOutputConfig(
                "second",
                "Second",
                "second",
                "image",
                "npy",
                "",
                "already",
            ),
        ),
        default_image_format="npy",
        existing_file_policy=ExistingFilePolicy.ERROR,
    )

    plan = build_batch_plan(config)

    assert plan.has_collisions
    assert all(output.duplicate for output in plan.items[0].outputs)
    assert all(output.exists for output in plan.items[0].outputs)
    assert all(
        output.status_text == "duplicate planned destination"
        for output in plan.items[0].outputs
    )


def test_build_batch_plan_glob_union_is_globally_sorted(tmp_path):
    inputs = tmp_path / "inputs"
    _write_arrays(
        inputs,
        **{
            "B_field": np.ones((2, 3), dtype=np.uint8),
            "A_field": np.ones((2, 3), dtype=np.uint8),
        },
    )
    workflow, output_ids = _batch_workflow()
    config = _batch_config(workflow, inputs, tmp_path / "outputs", output_ids)
    config = replace(
        config,
        sources=(replace(config.sources[0], pattern="B*.npy;A*.npy"),),
    )

    plan = build_batch_plan(config)

    assert [item.primary_source.name for item in plan.items] == [
        "A_field.npy",
        "B_field.npy",
    ]


def test_build_batch_plan_rejects_output_overlapping_any_input(tmp_path):
    inputs = tmp_path / "inputs"
    _write_arrays(inputs, field=np.ones((2, 3), dtype=np.uint8))
    workflow, output_ids = _batch_workflow(
        (
            {
                "tag": "output",
                "format": "npy",
                "subfolder": "",
                "filename_template": "{source_name}",
                "overwrite": "yes",
            },
        )
    )
    config = _batch_config(workflow, inputs, inputs, output_ids)

    plan = build_batch_plan(config)

    assert plan.has_collisions
    assert plan.items[0].outputs[0].input_collision
    with pytest.raises(FileExistsError, match="preflight found output collisions"):
        run_batch(workflow, config)


def test_run_batch_writes_output_and_complete_provenance_manifest(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    source = np.arange(20, dtype=np.uint16).reshape(4, 5)
    tifffile.imwrite(inputs / "sample.tif", source, photometric="minisblack")
    workflow, output_ids = _batch_workflow()
    config = _batch_config(workflow, inputs, tmp_path / "outputs", output_ids)
    config = replace(
        config,
        sources=(replace(config.sources[0], pattern="*.tif"),),
    )

    result = run_batch(
        workflow,
        config,
        workflow_path=tmp_path / "workflow.json",
        config_path=tmp_path / BATCH_CONFIG_FILENAME,
    )

    output_path = tmp_path / "outputs" / "sample__output.npy"
    assert result.saved_paths == (output_path,)
    np.testing.assert_array_equal(np.load(output_path), source)
    assert result.summary == {
        "completed": 1,
        "partial": 0,
        "skipped": 0,
        "failed": 0,
    }
    assert not result.has_failures
    assert result.manifest_path == tmp_path / "outputs" / BATCH_MANIFEST_FILENAME

    document = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert document["type"] == BATCH_MANIFEST_TYPE
    assert document["version"] == BATCH_MANIFEST_VERSION
    assert document["workflow"]["sha256"] == scientific_workflow_hash(workflow)
    assert document["config"]["sha256"] == batch_config_hash(config)
    assert document["summary"] == result.summary
    assert document["finished_at"]
    assert document["runtime"]["python"]
    assert document["runtime"]["platform"]
    assert "napari-vipp" in document["runtime"]["packages"]

    item = document["items"][0]
    assert item["status"] == BatchStatus.COMPLETED.value
    assert item["started_at"]
    assert item["finished_at"]
    source_record = item["sources"][0]
    assert source_record["node_id"] == "input"
    assert source_record["path"] == str(inputs / "sample.tif")
    assert source_record["identity"]["size_bytes"] > source.nbytes
    assert source_record["identity"]["modified_ns"] > 0
    assert source_record["identity"]["kind"] == "file"
    assert source_record["identity"]["regular_file_count"] == 1
    assert len(source_record["identity"]["sha256"]) == 64
    assert source_record["series"]["shape"] == [4, 5]
    assert source_record["series"]["dtype"] == "uint16"
    assert [axis["name"] for axis in source_record["image_state"]["axes"]] == [
        "y",
        "x",
    ]
    assert source_record["provenance"]
    output = item["outputs"][0]
    assert output["status"] == BatchStatus.COMPLETED.value
    assert output["path"] == str(output_path)
    assert output["size_bytes"] == output_path.stat().st_size
    assert output["existing_file_policy"] == "error"
    assert output["existed_at_preflight"] is False
    assert output["overwrote_existing"] is False
    item_records_dir = result.manifest_path.parent / document["item_records_dir"]
    assert len(list(item_records_dir.glob("*.json"))) == 1


def test_zarr_chunk_change_fails_item_before_any_output_is_published(
    tmp_path,
    monkeypatch,
):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    source_store = inputs / "sample.ome.zarr"
    source = np.arange(4 * 6, dtype=np.uint16).reshape(4, 6)
    write_image(source, source_store, format="ome-zarr")
    chunk_files = [
        path
        for path in source_store.rglob("*")
        if path.is_file()
        and not any(
            part.startswith(".")
            for part in path.relative_to(source_store).parts
        )
    ]
    assert chunk_files
    chunk = chunk_files[0]
    root_stat = source_store.stat()

    workflow, output_ids = _batch_workflow(
        (
            {
                "tag": "first",
                "format": "npy",
                "subfolder": "",
                "filename_template": "{source_stem}__{tag}",
                "overwrite": "batch default",
            },
            {
                "tag": "second",
                "format": "npy",
                "subfolder": "",
                "filename_template": "{source_stem}__{tag}",
                "overwrite": "batch default",
            },
        )
    )
    config = _batch_config(workflow, inputs, tmp_path / "outputs", output_ids)
    config = replace(
        config,
        sources=(replace(config.sources[0], pattern="*.zarr"),),
    )
    planned_outputs = tuple(
        output.path for output in build_batch_plan(config).items[0].outputs
    )
    original_save = batch_module.save_array_output
    changed = False
    save_calls = 0

    def mutate_chunk_before_lazy_save(*args, **kwargs):
        nonlocal changed, save_calls
        save_calls += 1
        if save_calls == 2:
            assert not any(path.exists() for path in planned_outputs)
            payload = bytearray(chunk.read_bytes())
            payload[-1] ^= 0x01
            chunk.write_bytes(payload)
            os.utime(
                source_store,
                ns=(root_stat.st_atime_ns, root_stat.st_mtime_ns),
            )
            changed = True
        return original_save(*args, **kwargs)

    monkeypatch.setattr(
        batch_module,
        "save_array_output",
        mutate_chunk_before_lazy_save,
    )

    result = run_batch(workflow, config)

    assert changed
    assert save_calls == 2
    assert source_store.stat().st_mtime_ns == root_stat.st_mtime_ns
    assert not any(path.exists() for path in planned_outputs)
    assert result.saved_paths == ()
    assert result.summary == {
        "completed": 0,
        "partial": 0,
        "skipped": 0,
        "failed": 1,
    }
    item = result.manifest.items[0]
    assert item.status == BatchStatus.FAILED
    assert item.error_type == "SourceChangedError"
    assert "source changed" in item.error_message.lower()
    assert [output.status for output in item.outputs] == [
        BatchStatus.FAILED,
        BatchStatus.FAILED,
    ]
    assert all(output.error_type == "SourceChangedError" for output in item.outputs)
    assert item.sources[0]["identity"]["kind"] == "directory"
    assert item.sources[0]["identity"]["regular_file_count"] > 1
    assert not list((tmp_path / "outputs").glob(".*.tmp.npy"))


def test_batch_rejects_output_nested_inside_zarr_source_store(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    source_store = inputs / "sample.ome.zarr"
    write_image(
        np.arange(12, dtype=np.uint16).reshape(3, 4),
        source_store,
        format="ome-zarr",
    )
    workflow, output_ids = _batch_workflow()
    config = _batch_config(workflow, inputs, source_store, output_ids)
    config = replace(
        config,
        sources=(replace(config.sources[0], pattern="*.zarr"),),
    )

    plan = build_batch_plan(config)

    assert plan.has_collisions
    assert plan.items[0].outputs[0].input_collision
    with pytest.raises(FileExistsError, match="preflight found output collisions"):
        run_batch(workflow, config)


def test_batch_rejects_enabled_save_image_side_effect_before_execution(tmp_path):
    inputs = tmp_path / "inputs"
    _write_arrays(inputs, sample=np.arange(12, dtype=np.uint8).reshape(3, 4))
    unverified_output = tmp_path / "unverified.npy"
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    save_node = pipeline.add_node("save_output")
    assert pipeline.connect("input", save_node.id).success
    pipeline.set_param(save_node.id, "enabled", "on")
    pipeline.set_param(save_node.id, "path", str(unverified_output))
    pipeline.set_param(save_node.id, "format", "npy")
    output = pipeline.add_node("batch_output")
    assert pipeline.connect(save_node.id, output.id).success
    workflow = serialize_workflow(pipeline)
    config = _batch_config(
        workflow,
        inputs,
        tmp_path / "outputs",
        (output.id,),
    )

    with pytest.raises(ValueError, match="enabled Save Image"):
        run_batch(workflow, config)

    assert not unverified_output.exists()


def test_run_batch_executes_and_saves_table_batch_output(tmp_path):
    inputs = tmp_path / "inputs"
    labels_source = np.zeros((8, 9), dtype=np.uint8)
    labels_source[1:3, 1:4] = 200
    labels_source[5:7, 5:8] = 240
    _write_arrays(inputs, sample=labels_source)
    pipeline = PrototypePipeline()
    labels = pipeline.add_node("label_connected_components")
    assert pipeline.connect("threshold", labels.id).success
    measurements = pipeline.add_node("measure_objects")
    assert pipeline.connect(labels.id, measurements.id).success
    output = pipeline.add_node("batch_output")
    assert pipeline.connect(measurements.id, output.id).success
    pipeline.set_param(output.id, "tag", "measurements")
    workflow = serialize_workflow(pipeline)
    config = BatchConfig(
        workflow_file=Path("workflow.json"),
        workflow_sha256=scientific_workflow_hash(workflow),
        output_dir=tmp_path / "outputs",
        sources=(BatchSourceConfig("input", "Input", inputs, "*.npy"),),
        outputs=(
            BatchOutputConfig(
                output.id,
                "Batch Output",
                "measurements",
                "table",
                "batch default",
                "",
                "{source_stem}__{tag}",
            ),
        ),
        default_image_format="npy",
        save_python_script=False,
    )

    result = run_batch(workflow, config)

    table_path = tmp_path / "outputs" / "sample__measurements.csv"
    assert result.saved_paths == (table_path,)
    header = table_path.read_text(encoding="utf-8").splitlines()[0]
    assert "label" in header


def test_saved_workflow_and_relative_config_reproduce_output_paths(tmp_path):
    inputs = tmp_path / "inputs"
    source = np.arange(12, dtype=np.uint8).reshape(3, 4)
    _write_arrays(inputs, field_a=source)
    workflow, output_ids = _batch_workflow()
    config = _batch_config(
        workflow,
        Path("inputs"),
        Path("outputs"),
        output_ids,
        base_dir=tmp_path,
    )
    workflow_path = tmp_path / "workflow.json"
    config_path = tmp_path / BATCH_CONFIG_FILENAME
    atomic_write_json(workflow_path, workflow)
    save_batch_config(config_path, config)

    result = run_batch_from_files(None, config_path)

    expected = tmp_path / "outputs" / "field_a__output.npy"
    assert result.saved_paths == (expected,)
    np.testing.assert_array_equal(np.load(expected), source)
    assert result.manifest.workflow_sha256 == config.workflow_sha256


def test_saved_workflow_resolves_fixed_relative_source_from_workflow_dir(
    tmp_path,
    monkeypatch,
):
    bundle = tmp_path / "bundle"
    inputs = bundle / "inputs"
    _write_arrays(inputs, primary=np.zeros((2, 3), dtype=np.uint8))
    bundle.mkdir(exist_ok=True)
    fixed_data = np.arange(6, dtype=np.uint8).reshape(2, 3)
    np.save(bundle / "fixed.npy", fixed_data)
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    fixed = pipeline.add_node("input")
    pipeline.set_param(fixed.id, "source_mode", "file path")
    pipeline.set_param(fixed.id, "file_path", "fixed.npy")
    output = pipeline.add_node("batch_output")
    assert pipeline.connect(fixed.id, output.id).success
    pipeline.set_param(output.id, "tag", "fixed")
    pipeline.set_param(output.id, "format", "npy")
    workflow = serialize_workflow(pipeline)
    config = BatchConfig(
        workflow_file=Path("workflow.json"),
        workflow_sha256=scientific_workflow_hash(workflow),
        output_dir=Path("outputs"),
        sources=(
            BatchSourceConfig("input", "Primary", Path("inputs"), "*.npy"),
        ),
        outputs=(
            BatchOutputConfig(
                output.id,
                "Batch Output",
                "fixed",
                "image",
                "npy",
                "",
                "{source_stem}__{tag}",
            ),
        ),
        default_image_format="npy",
        save_python_script=False,
        base_dir=bundle,
    )
    atomic_write_json(bundle / "workflow.json", workflow)
    save_batch_config(bundle / BATCH_CONFIG_FILENAME, config)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    result = run_batch_from_files(None, bundle / BATCH_CONFIG_FILENAME)

    saved = bundle / "outputs" / "primary__fixed.npy"
    assert result.saved_paths == (saved,)
    np.testing.assert_array_equal(np.load(saved), fixed_data)


def test_run_batch_continues_after_middle_source_read_failure(tmp_path):
    inputs = tmp_path / "inputs"
    first = np.full((3, 4), 11, dtype=np.uint8)
    last = np.full((3, 4), 33, dtype=np.uint8)
    _write_arrays(inputs, **{"01_first": first, "03_last": last})
    (inputs / "02_broken.npy").write_bytes(b"not a NumPy file")
    workflow, output_ids = _batch_workflow()
    config = _batch_config(workflow, inputs, tmp_path / "outputs", output_ids)

    result = run_batch(workflow, config)

    assert result.summary == {
        "completed": 2,
        "partial": 0,
        "skipped": 0,
        "failed": 1,
    }
    assert result.has_failures
    assert [item.status for item in result.manifest.items] == [
        BatchStatus.COMPLETED,
        BatchStatus.FAILED,
        BatchStatus.COMPLETED,
    ]
    assert result.manifest.items[1].error_type
    assert result.manifest.items[1].error_message
    first_output = tmp_path / "outputs" / "01_first__output.npy"
    last_output = tmp_path / "outputs" / "03_last__output.npy"
    assert set(result.saved_paths) == {first_output, last_output}
    np.testing.assert_array_equal(np.load(first_output), first)
    np.testing.assert_array_equal(np.load(last_output), last)


def test_run_batch_preserves_run_manifest_archive_across_reruns(tmp_path):
    inputs = tmp_path / "inputs"
    _write_arrays(inputs, sample=np.ones((2, 3), dtype=np.uint8))
    workflow, output_ids = _batch_workflow()
    config = _batch_config(
        workflow,
        inputs,
        tmp_path / "outputs",
        output_ids,
        policy=ExistingFilePolicy.OVERWRITE,
    )

    first = run_batch(workflow, config)
    first_archive = first.manifest_archive_path
    assert first_archive is not None
    first_document = first_archive.read_bytes()
    second = run_batch(workflow, config)

    assert second.manifest_archive_path != first_archive
    assert first_archive.read_bytes() == first_document
    archived = json.loads(first_archive.read_text(encoding="utf-8"))
    assert archived["workflow"]["scientific_graph"]
    assert archived["config"]["document"]["type"] == BATCH_CONFIG_TYPE


def test_run_batch_records_partial_item_when_one_output_save_fails(
    tmp_path,
    monkeypatch,
):
    inputs = tmp_path / "inputs"
    source = np.arange(12, dtype=np.uint8).reshape(3, 4)
    _write_arrays(inputs, sample=source)
    workflow, output_ids = _batch_workflow(
        (
            {
                "tag": "good",
                "format": "npy",
                "subfolder": "",
                "filename_template": "{source_stem}__{tag}",
                "overwrite": "batch default",
            },
            {
                "tag": "bad",
                "format": "npy",
                "subfolder": "",
                "filename_template": "{source_stem}__{tag}",
                "overwrite": "batch default",
            },
        )
    )
    config = _batch_config(workflow, inputs, tmp_path / "outputs", output_ids)
    original_save = batch_module._save_planned_output

    def flaky_save(pipeline, output):
        if output.node_id == output_ids[1]:
            raise OSError("simulated destination failure")
        return original_save(pipeline, output)

    monkeypatch.setattr(batch_module, "_save_planned_output", flaky_save)

    result = run_batch(workflow, config)

    assert result.summary == {
        "completed": 0,
        "partial": 1,
        "skipped": 0,
        "failed": 0,
    }
    assert result.has_failures
    item = result.manifest.items[0]
    assert item.status == BatchStatus.PARTIAL
    assert [output.status for output in item.outputs] == [
        BatchStatus.COMPLETED,
        BatchStatus.FAILED,
    ]
    assert item.outputs[1].error_type == "OSError"
    assert item.outputs[1].error_message == "simulated destination failure"
    assert result.saved_paths == (tmp_path / "outputs" / "sample__good.npy",)


def test_pipeline_failure_still_saves_completed_independent_output(
    tmp_path,
    monkeypatch,
):
    inputs = tmp_path / "inputs"
    source = np.arange(12, dtype=np.uint8).reshape(3, 4)
    _write_arrays(inputs, sample=source)
    workflow, output_ids = _batch_workflow(
        (
            {
                "tag": "completed",
                "format": "npy",
                "subfolder": "",
                "filename_template": "{source_stem}__{tag}",
                "overwrite": "batch default",
            },
            {
                "tag": "failed",
                "format": "npy",
                "subfolder": "",
                "filename_template": "{source_stem}__{tag}",
                "overwrite": "batch default",
            },
        )
    )
    config = _batch_config(workflow, inputs, tmp_path / "outputs", output_ids)

    def partially_failing_run(pipeline, *_args, **_kwargs):
        pipeline.outputs[output_ids[0]] = source.copy()
        pipeline.outputs[output_ids[1]] = None
        raise RuntimeError("independent branch failed")

    monkeypatch.setattr(PrototypePipeline, "run", partially_failing_run)

    result = run_batch(workflow, config)

    assert result.manifest.items[0].status == BatchStatus.PARTIAL
    assert [record.status for record in result.manifest.items[0].outputs] == [
        BatchStatus.COMPLETED,
        BatchStatus.FAILED,
    ]
    assert result.saved_paths == (
        tmp_path / "outputs" / "sample__completed.npy",
    )


@pytest.mark.parametrize(
    ("policy", "expected_status", "expected_saved"),
    [
        (ExistingFilePolicy.ERROR, None, False),
        (ExistingFilePolicy.SKIP, BatchStatus.SKIPPED, False),
        (ExistingFilePolicy.OVERWRITE, BatchStatus.COMPLETED, True),
    ],
)
def test_run_batch_existing_file_policies(
    tmp_path,
    policy,
    expected_status,
    expected_saved,
):
    inputs = tmp_path / "inputs"
    source = np.arange(12, dtype=np.uint8).reshape(3, 4)
    _write_arrays(inputs, sample=source)
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    destination = output_dir / "sample__output.npy"
    original = np.full((2, 2), 99, dtype=np.uint8)
    np.save(destination, original)
    workflow, output_ids = _batch_workflow()
    config = _batch_config(
        workflow,
        inputs,
        output_dir,
        output_ids,
        policy=policy,
    )

    plan = build_batch_plan(config)
    assert plan.items[0].outputs[0].exists
    assert plan.items[0].outputs[0].existing_file_policy == policy
    if policy == ExistingFilePolicy.ERROR:
        with pytest.raises(FileExistsError, match="preflight found output collisions"):
            run_batch(workflow, config)
        np.testing.assert_array_equal(np.load(destination), original)
        return

    result = run_batch(workflow, config)

    assert result.manifest.items[0].status == expected_status
    assert bool(result.saved_paths) is expected_saved
    if policy == ExistingFilePolicy.OVERWRITE:
        np.testing.assert_array_equal(np.load(destination), source)
    else:
        np.testing.assert_array_equal(np.load(destination), original)


def test_mixed_completed_and_skipped_outputs_are_not_a_failure(tmp_path):
    inputs = tmp_path / "inputs"
    _write_arrays(
        inputs,
        first=np.ones((2, 3), dtype=np.uint8),
        second=np.full((2, 3), 2, dtype=np.uint8),
    )
    workflow, output_ids = _batch_workflow(
        (
            {
                "tag": "one",
                "format": "npy",
                "subfolder": "",
                "filename_template": "{source_stem}__{tag}",
                "overwrite": "batch default",
            },
            {
                "tag": "two",
                "format": "npy",
                "subfolder": "",
                "filename_template": "{source_stem}__{tag}",
                "overwrite": "batch default",
            },
        )
    )
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    np.save(output_dir / "first__one.npy", np.zeros((2, 3), dtype=np.uint8))
    config = _batch_config(
        workflow,
        inputs,
        output_dir,
        output_ids,
        policy=ExistingFilePolicy.SKIP,
        continue_on_error=False,
    )

    result = run_batch(workflow, config)

    assert result.summary == {
        "completed": 1,
        "partial": 1,
        "skipped": 0,
        "failed": 0,
    }
    assert not result.has_failures
    assert result.manifest.items[1].status == BatchStatus.COMPLETED


def test_atomic_output_failure_preserves_existing_destination(
    tmp_path,
    monkeypatch,
):
    inputs = tmp_path / "inputs"
    source = np.ones((2, 3), dtype=np.uint8)
    _write_arrays(inputs, sample=source)
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    destination = output_dir / "sample__output.npy"
    original = np.full((2, 3), 9, dtype=np.uint8)
    np.save(destination, original)
    workflow, output_ids = _batch_workflow()
    config = _batch_config(
        workflow,
        inputs,
        output_dir,
        output_ids,
        policy=ExistingFilePolicy.OVERWRITE,
    )
    original_replace = batch_module._replace_with_retry

    def fail_output_replace(source_path, target_path):
        if Path(target_path) == destination:
            raise OSError("simulated atomic replace failure")
        return original_replace(source_path, target_path)

    monkeypatch.setattr(batch_module, "_replace_with_retry", fail_output_replace)

    result = run_batch(workflow, config)

    assert result.manifest.items[0].status == BatchStatus.FAILED
    np.testing.assert_array_equal(np.load(destination), original)
    assert not list(output_dir.glob(".*.tmp.npy"))


@pytest.mark.parametrize("error_number", [errno.EPERM, errno.EOPNOTSUPP])
def test_no_replace_promotion_falls_back_when_hard_links_are_unsupported(
    tmp_path,
    monkeypatch,
    error_number,
):
    source = tmp_path / "temporary.npy"
    destination = tmp_path / "result.npy"
    payload = b"complete scientific output"
    source.write_bytes(payload)

    def unsupported_hard_link(_source, _destination):
        raise OSError(error_number, "hard links unavailable")

    monkeypatch.setattr(batch_module.os, "link", unsupported_hard_link)

    batch_module._promote_no_replace(source, destination)

    assert destination.read_bytes() == payload
    assert not source.exists()


def test_no_replace_fallback_preserves_destination_that_appeared(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "temporary.npy"
    destination = tmp_path / "result.npy"
    source.write_bytes(b"new")
    destination.write_bytes(b"existing")

    def unsupported_hard_link(_source, _destination):
        raise OSError(errno.EOPNOTSUPP, "hard links unavailable")

    monkeypatch.setattr(batch_module.os, "link", unsupported_hard_link)

    with pytest.raises(FileExistsError):
        batch_module._promote_no_replace(source, destination)

    assert destination.read_bytes() == b"existing"
    assert source.read_bytes() == b"new"


@pytest.mark.parametrize("failing_call", ["write", "fsync", "fstat"])
def test_no_replace_fallback_cleans_failed_exclusive_claim(
    tmp_path,
    monkeypatch,
    failing_call,
):
    source = tmp_path / "temporary.npy"
    destination = tmp_path / "result.npy"
    source.write_bytes(b"new")

    def unsupported_hard_link(_source, _destination):
        raise OSError(errno.EOPNOTSUPP, "hard links unavailable")

    def fail_claim_initialization(*_args, **_kwargs):
        raise OSError("simulated claim initialization failure")

    monkeypatch.setattr(batch_module.os, "link", unsupported_hard_link)
    monkeypatch.setattr(
        batch_module.os,
        failing_call,
        fail_claim_initialization,
    )

    with pytest.raises(OSError, match="claim initialization failure"):
        batch_module._promote_no_replace(source, destination)

    assert not destination.exists()
    assert source.read_bytes() == b"new"


def test_no_replace_promotion_does_not_fail_after_successful_link_cleanup(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "temporary.npy"
    destination = tmp_path / "result.npy"
    source.write_bytes(b"new")
    original_unlink = Path.unlink
    attempts = 0

    def transient_unlink(path, *args, **kwargs):
        nonlocal attempts
        if path == source and attempts == 0:
            attempts += 1
            raise PermissionError("simulated indexer lock")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", transient_unlink)

    batch_module._promote_no_replace(source, destination)

    assert destination.read_bytes() == b"new"
    assert not source.exists()
    assert attempts == 1


def test_run_batch_stops_after_failure_when_continue_on_error_is_disabled(
    tmp_path,
):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "01_broken.npy").write_bytes(b"invalid")
    np.save(inputs / "02_valid.npy", np.ones((2, 3), dtype=np.uint8))
    workflow, output_ids = _batch_workflow()
    config = _batch_config(
        workflow,
        inputs,
        tmp_path / "outputs",
        output_ids,
        continue_on_error=False,
    )

    result = run_batch(workflow, config)

    assert [item.status for item in result.manifest.items] == [
        BatchStatus.FAILED,
        BatchStatus.SKIPPED,
    ]
    assert not result.saved_paths


def test_run_batch_rejects_workflow_hash_mismatch_before_writing(tmp_path):
    inputs = tmp_path / "inputs"
    _write_arrays(inputs, sample=np.ones((2, 3), dtype=np.uint8))
    workflow, output_ids = _batch_workflow()
    config = replace(
        _batch_config(workflow, inputs, tmp_path / "outputs", output_ids),
        workflow_sha256="f" * 64,
    )

    with pytest.raises(ValueError, match="workflow hash does not match"):
        run_batch(workflow, config)

    assert not (tmp_path / "outputs").exists()


def test_run_batch_rejects_output_declaration_drift(tmp_path):
    inputs = tmp_path / "inputs"
    _write_arrays(inputs, sample=np.ones((2, 3), dtype=np.uint8))
    workflow, output_ids = _batch_workflow()
    config = _batch_config(workflow, inputs, tmp_path / "outputs", output_ids)
    config = replace(
        config,
        outputs=(replace(config.outputs[0], tag="edited-outside-workflow"),),
    )

    with pytest.raises(ValueError, match="declaration does not match"):
        run_batch(workflow, config)


def test_terminal_fallback_rejects_multi_output_node(tmp_path):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    split = pipeline.add_node("split_channels")
    assert pipeline.connect("input", split.id).success
    workflow = serialize_workflow(pipeline)
    config = BatchConfig(
        workflow_file=Path("workflow.json"),
        workflow_sha256=scientific_workflow_hash(workflow),
        output_dir=tmp_path / "outputs",
        sources=(
            BatchSourceConfig("input", "Input", tmp_path / "inputs", "*.npy"),
        ),
        outputs=(
            BatchOutputConfig(
                split.id,
                split.title,
                safe_batch_filename(f"{split.title}-{split.id}"),
                "image",
                "batch default",
                "",
                "{source_stem}__{tag}",
            ),
        ),
        save_python_script=False,
    )

    with pytest.raises(ValueError, match="multi-output nodes"):
        validate_batch_config(workflow, config)
