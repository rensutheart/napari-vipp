"""Typed dataset sources never use image coercion or unverified caller data."""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import replace

import pytest

from napari_vipp.core.compute import ComputeMode, ComputeRequest
from napari_vipp.core.execution import PipelineRunRequest, execute_pipeline_request
from napari_vipp.core.export import export_pipeline_to_python
from napari_vipp.core.measurement_collection import (
    CollectionItem,
    MeasurementOutput,
    MeasurementPreview,
    collect_measurements,
    load_measurement_collection,
    save_measurement_collection,
)
from napari_vipp.core.pipeline import PrototypePipeline, SourcePayload
from napari_vipp.core.progress import OperationCancelled
from napari_vipp.core.reproducibility import build_reproducibility_package
from napari_vipp.core.table_source import load_table_source, resolve_table_source
from napari_vipp.core.tables import TableData, TableState
from napari_vipp.core.workflow import (
    deserialize_workflow,
    load_workflow,
    save_workflow,
    serialize_workflow,
)


@pytest.fixture
def dataset(tmp_path):
    table = TableData(
        ("label_id", "area", "sample", "valid"),
        ((2**63 + 3, 3.5, "001", True), (4, 6.5, "001", False)),
        name="Objects",
        table_kind="object measurements",
        column_units=(("area", "micrometer^2"),),
    )
    preview = MeasurementPreview(
        MeasurementOutput("output", "Objects", "objects"),
        (
            CollectionItem(
                "item-a", 1, "image-a", "ready", row_count=2,
                result_sha256="c" * 64, table=table,
            ),
        ),
        "run-a", "a" * 64, "b" * 64,
    )
    collection = collect_measurements(preview)
    path = tmp_path / "measurements.vipp-results.json"
    save_measurement_collection(collection, path)
    return path, hashlib.sha256(path.read_bytes()).hexdigest(), collection.table


def _pipeline(path, digest):
    pipeline = PrototypePipeline()
    pipeline.restore_graph([], [])
    source = pipeline.add_node("table_source")
    pipeline.set_param(source.id, "dataset_path", str(path))
    pipeline.set_param(source.id, "dataset_sha256", digest)
    select = pipeline.add_node("select_table_columns")
    pipeline.set_param(select.id, "columns", "label_id,area,sample,valid")
    assert pipeline.connect(source.id, select.id).success
    return pipeline, source.id, select.id


def test_load_source_preserves_typed_table_and_units(dataset):
    path, digest, table = dataset
    payload = load_table_source(path)
    assert payload.data == table
    assert isinstance(payload.image_state, TableState)
    assert payload.image_state.column_units == table.column_units
    assert payload.revision_token.file_sha256 == digest
    assert payload.data.rows[0][:4] == (2**63 + 3, 3.5, "001", True)
    assert type(payload.data.rows[0][0]) is int
    assert type(payload.data.rows[0][3]) is bool


def test_source_cancellation_reaches_background_read(dataset):
    path, digest, _ = dataset
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(OperationCancelled):
        load_table_source(path, digest, cancellation=cancelled)
    pipeline, _, _ = _pipeline(path, digest)
    with pytest.raises(OperationCancelled):
        pipeline.run(None, cancel_callback=cancelled.is_set)


def test_empty_collection_stays_a_zero_row_table(dataset, tmp_path):
    path, _, _ = dataset
    collection = load_measurement_collection(path)
    empty = replace(
        collection,
        table=replace(collection.table, rows=()),
        items=tuple(
            replace(item, status="empty", row_count=0) for item in collection.items
        ),
    )
    target = tmp_path / "empty.vipp-results.json"
    save_measurement_collection(empty, target)
    payload = load_table_source(target)
    pipeline, source, selected = _pipeline(target, payload.revision_token.file_sha256)
    outputs = pipeline.run(None)
    assert outputs[source].row_count == outputs[selected].row_count == 0
    assert outputs[selected].columns == ("label_id", "area", "sample", "valid")
    assert payload.image_state.row_count == 0


def test_preflight_reuses_only_loader_issued_snapshot_without_read(
    dataset, monkeypatch,
):
    path, digest, _ = dataset
    pipeline, source, _ = _pipeline(path, digest)
    payload = load_table_source(path, digest)

    def fail(*args, **kwargs):
        raise AssertionError("Metadata preflight must not read the dataset")

    monkeypatch.setattr("napari_vipp.core.table_source.load_table_source", fail)
    assert resolve_table_source(
        pipeline.nodes[source].params, payload, metadata_only=True
    ) == [(payload.data, payload.image_state)]
    assert resolve_table_source(
        pipeline.nodes[source].params, None, metadata_only=True
    ) == [(None, None)]
    with pytest.raises(ValueError, match="Reload Table Source"):
        resolve_table_source(
            pipeline.nodes[source].params,
            replace(payload, data=TableData(("fake",), ((999,),))),
            metadata_only=True,
        )


def test_execution_ignores_forged_payload_and_rechecks_file(dataset):
    path, digest, table = dataset
    pipeline, source, selected = _pipeline(path, digest)
    forged = SourcePayload(TableData(("fake",), ((999,),)))
    outputs = pipeline.run(None, source_payloads={source: forged})
    assert outputs[source] == table
    assert outputs[selected].rows[0] == table.rows[0][:4]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash does not match"):
        pipeline.run(None, source_payloads={source: forged})


@pytest.mark.parametrize("mode", [ComputeMode.CPU, ComputeMode.AUTO])
def test_shared_headless_execution_has_table_metadata_and_provenance(dataset, mode):
    path, digest, _ = dataset
    pipeline, source, selected = _pipeline(path, digest)
    result = execute_pipeline_request(
        PipelineRunRequest(
            run_id=1,
            workflow=serialize_workflow(pipeline),
            input_data=None,
            input_metadata=None,
            input_name="",
            source_payloads={},
            compute_request=ComputeRequest(mode=mode),
        ),
        raise_errors=True,
    )
    assert result.error == ""
    assert isinstance(result.pipeline.output_states[source], TableState)
    assert result.pipeline.outputs[selected].row_count == 2
    assert source in result.pipeline.node_compute_provenance


def test_missing_unbound_and_changed_preview_fail_clearly(dataset):
    path, digest, _ = dataset
    pipeline, source, _ = _pipeline(path, digest)
    payload = load_table_source(path, digest)
    with pytest.raises(ValueError, match="no recorded dataset hash"):
        resolve_table_source({"dataset_path": str(path), "dataset_sha256": ""})
    path.unlink()
    with pytest.raises(ValueError, match="missing"):
        resolve_table_source(pipeline.nodes[source].params, payload)
    with pytest.raises(ValueError, match="unavailable"):
        resolve_table_source(pipeline.nodes[source].params, payload, metadata_only=True)


def test_finish_rejects_dataset_changed_during_downstream_work(dataset):
    path, digest, _ = dataset
    pipeline, _, selected = _pipeline(path, digest)

    def changed(node_id):
        if node_id == selected:
            path.write_bytes(path.read_bytes() + b" ")

    with pytest.raises(ValueError, match="hash does not match"):
        pipeline.run(None, node_finished_callback=changed)


def test_workflow_reference_only_and_relative_reopen(dataset, tmp_path):
    path, digest, _ = dataset
    pipeline, source, _ = _pipeline(path.name, digest)
    saved = tmp_path / "results-workflow.json"
    save_workflow(saved, pipeline)
    raw = json.loads(saved.read_text())
    params = raw["nodes"][0]["params"]
    assert params == {"dataset_path": path.name, "dataset_sha256": digest}
    restored = load_workflow(saved)
    assert restored["nodes"][0].params["dataset_path"] == str(path.resolve())
    params["_vipp_table"] = {"rows": [["private data"]]}
    with pytest.raises(ValueError, match="embedded measurements"):
        deserialize_workflow(raw)
    pipeline.nodes[source].params["_vipp_table"] = [["private data"]]
    with pytest.raises(ValueError, match="embedded measurements"):
        serialize_workflow(pipeline)


def test_keep_cached_is_a_typed_supported_workflow_preference(dataset):
    path, digest, _ = dataset
    pipeline, source, _ = _pipeline(path, digest)
    pipeline.set_param(source, "_vipp_keep_cached", True)
    document = serialize_workflow(pipeline)
    restored = deserialize_workflow(document)
    assert restored["nodes"][0].params["_vipp_keep_cached"] is True
    assert resolve_table_source(pipeline.nodes[source].params)[0][0].row_count == 2
    document["nodes"][0]["params"]["_vipp_keep_cached"] = "false"
    with pytest.raises(ValueError, match="must be Boolean"):
        deserialize_workflow(document)


def test_export_limit_and_package_keep_recipe_without_reading_dataset(
    dataset, monkeypatch,
):
    path, digest, _ = dataset
    pipeline, _, _ = _pipeline(path, digest)
    with pytest.raises(ValueError, match="Python export for Table Source"):
        export_pipeline_to_python(pipeline)

    def fail(*args, **kwargs):
        raise AssertionError("Package export must not read result files")

    monkeypatch.setattr("napari_vipp.core.table_source.load_table_source", fail)
    path.unlink()
    package = build_reproducibility_package(serialize_workflow(pipeline))
    assert "runner.py" not in package.members
    document = json.loads(package.members["workflow.json"])
    params = document["nodes"][0]["params"]
    assert params["dataset_sha256"] == digest
    assert params["dataset_path"] != str(path)
    report = json.loads(package.members["report.json"])
    assert "not available" in report["python_runner_unavailable"]
    text = "\n".join(member.decode() for member in package.members.values())
    assert str(path.parent) not in text
    assert '"rows"' not in text
