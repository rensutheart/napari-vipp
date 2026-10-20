"""Copied batches keep exact input evidence and per-sample scientific settings."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from napari_vipp.core.batch import (
    BatchConfig,
    BatchOutputConfig,
    BatchParameterOverride,
    BatchSourceConfig,
    BatchSourceParameterOverrides,
    atomic_write_json,
    batch_source_item_override_key,
    bind_batch_plan_source_items,
    build_batch_plan,
    load_batch_config,
    run_batch_from_files,
    save_batch_config,
    scientific_workflow_hash,
)
from napari_vipp.core.measurement_collection import (
    CollectionItem,
    MeasurementOutput,
    MeasurementPreview,
    collect_measurements,
    save_measurement_collection,
)
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.tables import TableData
from napari_vipp.core.workflow import serialize_workflow


def _bound_batch(root: Path):
    inputs = root / "data"
    inputs.mkdir(parents=True)
    array = np.arange(12, dtype=np.uint16).reshape(3, 4)
    np.save(inputs / "a.npy", array)
    np.save(inputs / "b.npy", array + 10)
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    operation = pipeline.add_node("linear_scale_offset")
    pipeline.set_param(operation.id, "alpha", 1.0)
    pipeline.set_param(operation.id, "beta", 0.0)
    assert pipeline.connect("input", operation.id).success
    output = pipeline.add_node("batch_output")
    pipeline.set_param(output.id, "format", "npy")
    assert pipeline.connect(operation.id, output.id).success
    document = serialize_workflow(pipeline)
    atomic_write_json(root / "workflow.json", document)
    config = BatchConfig(
        root / "workflow.json",
        scientific_workflow_hash(document),
        root / "outputs",
        (BatchSourceConfig("input", "Images", inputs, "*.npy"),),
        (
            BatchOutputConfig(
                output.id,
                "Batch Output",
                "output",
                "image",
                "npy",
                "",
                "{source_stem}__{tag}",
            ),
        ),
        default_image_format="npy",
        save_workflow_snapshot=False,
        save_python_script=False,
        base_dir=root,
    )
    plan = build_batch_plan(config)
    config = bind_batch_plan_source_items(config, plan)
    key = batch_source_item_override_key("input", plan.items[0].source_items["input"])
    config = replace(
        config,
        parameter_overrides=(
            BatchSourceParameterOverrides(
                key, (BatchParameterOverride(operation.id, "beta", 5.0),)
            ),
        ),
    )
    save_batch_config(root / "batch.json", config)
    return document, config, array


def test_copied_bound_batch_runs_from_unrelated_cwd_with_overrides(
    tmp_path, monkeypatch
):
    original = tmp_path / "original"
    document, config, array = _bound_batch(original)
    stored = json.loads((original / "batch.json").read_text(encoding="utf-8"))
    assert stored["workflow"]["file"] == "workflow.json"
    assert stored["output_dir"] == "outputs"
    assert stored["sources"][0]["input_dir"] == "data"
    assert [
        item["container"]["uri"] for item in stored["sources"][0]["source_items"]
    ] == ["data/a.npy", "data/b.npy"]
    for saved, expected in zip(
        stored["sources"][0]["source_items"],
        config.sources[0].source_items,
        strict=True,
    ):
        assert saved["container"]["revision"] == expected.container.revision.to_dict()
        assert saved["selector"] == expected.selector.to_dict()
    assert stored["parameter_overrides"] == config.to_dict()["parameter_overrides"]
    copied = tmp_path / "conference copy"
    shutil.copytree(original, copied)
    original.rename(tmp_path / "unavailable original")
    elsewhere = tmp_path / "unrelated working directory"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    loaded = load_batch_config(copied / "batch.json")
    plan = build_batch_plan(loaded)
    assert len(plan.items) == 2
    assert all(
        path.is_relative_to(copied)
        for item in plan.items
        for path in item.source_paths.values()
    )
    result = run_batch_from_files(None, copied / "batch.json")

    assert not result.has_failures, result.manifest.items[0].error_message
    np.testing.assert_array_equal(np.load(copied / "outputs/a__output.npy"), array + 5)
    np.testing.assert_array_equal(np.load(copied / "outputs/b__output.npy"), array + 10)
    assert result.manifest.workflow_sha256 == scientific_workflow_hash(document)


def test_save_as_rebases_every_source_item_without_mutating_original(tmp_path):
    root = tmp_path / "original"
    _bound_batch(root)
    loaded = load_batch_config(root / "batch.json")
    before = loaded.to_dict()
    target = tmp_path / "other" / "nested" / "batch.json"

    save_batch_config(target, loaded)

    assert loaded.to_dict() == before
    saved = json.loads(target.read_text(encoding="utf-8"))
    assert saved["workflow"]["file"] == "../../original/workflow.json"
    assert saved["output_dir"] == "../../original/outputs"
    assert saved["sources"][0]["input_dir"] == "../../original/data"
    assert (
        saved["sources"][0]["source_items"][0]["container"]["uri"]
        == "../../original/data/a.npy"
    )
    reloaded = load_batch_config(target)
    assert reloaded.resolve_path(reloaded.workflow_file) == root / "workflow.json"
    plan = build_batch_plan(reloaded)
    assert len(plan.items) == 2
    assert plan.items[0].source_paths["input"] == root / "data/a.npy"
    assert plan.items[0].parameter_overrides == loaded.parameter_overrides[0].values


def test_relocated_batch_still_rejects_modified_content(tmp_path):
    original = tmp_path / "original"
    _bound_batch(original)
    copied = tmp_path / "copy"
    shutil.copytree(original, copied)
    np.save(copied / "data/a.npy", np.zeros((3, 4), dtype=np.uint16))

    with pytest.raises(ValueError, match="SourceItem revisions or logical selectors"):
        build_batch_plan(load_batch_config(copied / "batch.json"))


def test_relocated_batch_still_rejects_changed_scientific_graph(tmp_path):
    original = tmp_path / "original"
    _bound_batch(original)
    copied = tmp_path / "copy"
    shutil.copytree(original, copied)
    workflow_path = copied / "workflow.json"
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    operation = next(
        node
        for node in workflow["nodes"]
        if node["operation_id"] == "linear_scale_offset"
    )
    operation["params"]["alpha"] = 2.0
    atomic_write_json(workflow_path, workflow)

    with pytest.raises(ValueError, match="workflow hash does not match"):
        run_batch_from_files(None, copied / "batch.json")
    assert not (copied / "outputs").exists()


def test_headless_batch_resolves_table_dataset_from_workflow_not_config_or_cwd(
    tmp_path, monkeypatch
):
    project = tmp_path / "project"
    workflows = project / "workflows"
    workflows.mkdir(parents=True)
    inputs = project / "images"
    inputs.mkdir()
    np.save(inputs / "sample.npy", np.zeros((2, 3), dtype=np.uint8))
    preview = MeasurementPreview(
        MeasurementOutput("output", "Objects", "objects"),
        (
            CollectionItem(
                "a",
                1,
                "image-a",
                "ready",
                row_count=1,
                result_sha256="c" * 64,
                table=TableData(("label", "area"), ((1, 3.5),)),
            ),
        ),
        "run-a",
        "a" * 64,
        "b" * 64,
    )
    dataset = project / "measurements.vipp-results.json"
    save_measurement_collection(collect_measurements(preview), dataset)
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    source = pipeline.add_node("table_source")
    pipeline.set_param(source.id, "dataset_path", "../measurements.vipp-results.json")
    pipeline.set_param(
        source.id, "dataset_sha256", hashlib.sha256(dataset.read_bytes()).hexdigest()
    )
    output = pipeline.add_node("batch_output")
    pipeline.set_param(output.id, "format", "csv")
    assert pipeline.connect(source.id, output.id).success
    workflow = serialize_workflow(pipeline)
    atomic_write_json(workflows / "workflow.json", workflow)
    config = BatchConfig(
        workflows / "workflow.json",
        scientific_workflow_hash(workflow),
        project / "outputs",
        (BatchSourceConfig("input", "Images", inputs, "*.npy"),),
        (
            BatchOutputConfig(
                output.id,
                "Batch Output",
                "output",
                "table",
                "csv",
                "",
                "{source_stem}__{tag}",
            ),
        ),
        save_workflow_snapshot=False,
        save_python_script=False,
        base_dir=project,
    )
    save_batch_config(project / "batch.json", config)
    monkeypatch.chdir(tmp_path)

    result = run_batch_from_files(None, project / "batch.json")

    assert not result.has_failures, result.manifest.items[0].error_message
    assert "3.5" in (project / "outputs/sample__output.csv").read_text(encoding="utf-8")
    assert result.manifest.workflow_sha256 == scientific_workflow_hash(workflow)
