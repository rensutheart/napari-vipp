from __future__ import annotations

import csv
import hashlib
import json
import math
import threading
from copy import deepcopy
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from napari_vipp.core import measurement_collection as module
from napari_vipp.core.batch_resume import (
    item_record_from_document,
    output_identity,
    seal_document,
)
from napari_vipp.core.measurement_collection import (
    CollectionLimits,
    MeasurementCollectionError,
    available_measurement_outputs,
    collect_measurements,
    inspect_collection,
    load_measurement_collection,
    save_measurement_collection,
    table_measurement_metadata,
)
from napari_vipp.core.progress import OperationCancelled
from napari_vipp.core.tables import TableData, save_table_output


def _table(rows=((1, 2.5),), *, units=(("volume", "micrometer^3"),)):
    return TableData(
        ("label", "volume"), tuple(rows), table_kind="objects", column_units=units
    )


def _manifest(tmp_path, tables=None, *, statuses=None, format="csv"):
    tables = tables or [_table(), _table()]
    output_dir = tmp_path / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for index, table in enumerate(tables, 1):
        path = output_dir / f"image-{index}.{format}"
        save_table_output(table, path, format=format)
        metadata = table_measurement_metadata(table)
        metadata["file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        status = statuses[index - 1] if statuses else "completed"
        items.append(
            {
                "index": index,
                "batch_id": f"image-{index}",
                "status": status,
                "sources": [{"path": str(tmp_path / f"unread-image-{index}.tif")}],
                "resumed_from_run_id": "previous-run",
                "outputs": [
                    {
                        "node_id": "batch_output_1",
                        "node_title": "Object measurements",
                        "tag": "objects",
                        "kind": "table",
                        "format": format,
                        "path": str(path),
                        "status": status,
                        "existing_file_policy": "error",
                        "existed_at_preflight": False,
                        "provenance_status": "verified_reused",
                        "content_identity": output_identity(path),
                        "table_metadata": metadata,
                    }
                ],
            }
        )
    return seal_document(
        {
            "type": "napari-vipp-batch-manifest",
            "version": 6,
            "run_id": "run-current",
            "finished_at": "2026-09-15T12:00:00Z",
            "workflow": {"sha256": "a" * 64},
            "output_dir": str(output_dir),
            "items": items,
        }
    )


def _reseal(document):
    return seal_document(document)


def test_append_keeps_image_identity_and_units_without_merging_local_labels(tmp_path):
    manifest = _manifest(tmp_path)
    (output,) = available_measurement_outputs(manifest)
    assert output.node_id == "batch_output_1"
    assert output.title == "Object measurements"
    events = []
    preview = inspect_collection(
        manifest, output.node_id, progress=lambda *event: events.append(event)
    )
    collection = collect_measurements(
        preview,
        annotations={
            "item-1": {"condition": "control", "sample": "mouse-a"},
            "item-2": {"condition": "treated", "sample": "mouse-b"},
        },
    )
    assert collection.table.row_count == 2
    rows = collection.table.records()
    assert [row["label"] for row in rows] == [1, 1]
    assert [row["_vipp_item_key"] for row in rows] == ["item-1", "item-2"]
    assert [row["_vipp_batch_id"] for row in rows] == ["image-1", "image-2"]
    assert [row["condition"] for row in rows] == ["control", "treated"]
    assert collection.table.unit_for("volume") == "micrometer^3"
    assert collection.provenance["manifest_sha256"] == manifest["integrity_sha256"]
    assert len(events) == 2
    with pytest.raises(TypeError):
        collection.annotations["item-1"]["condition"] = "changed"
    with pytest.raises(TypeError):
        collection.provenance["run_id"] = "changed"


@pytest.mark.parametrize("format", ["csv", "tsv"])
def test_exact_mixed_cells_big_integers_none_empty_unicode_and_nonfinite(
    tmp_path, format
):
    original = TableData(
        ("mixed", "text", "number"),
        (
            (None, "", 2**64 - 1),
            ("", "001", 1),
            ('a,\t"quoted"\nline', "μm", 3),
            (True, "true", 4),
            (False, "false", 5),
            (1.5, "nan", 6),
            (float("nan"), "inf", 7),
            (float("inf"), "-inf", 8),
            (-0.0, "null", 9),
        ),
    )
    manifest = _manifest(tmp_path, [original], format=format)
    metadata = manifest["items"][0]["outputs"][0]["table_metadata"]
    assert metadata["cell_types"][2] == [["int", 9]]
    assert "μm" not in json.dumps(metadata, ensure_ascii=False)
    preview = inspect_collection(manifest, "batch_output_1")
    restored = preview.items[0].table
    assert restored is not None, preview.items[0].message
    for expected_row, actual_row in zip(original.rows, restored.rows, strict=True):
        for expected, actual in zip(expected_row, actual_row, strict=True):
            assert type(actual) is type(expected)
            if isinstance(expected, float) and math.isnan(expected):
                assert math.isnan(actual)
            else:
                assert actual == expected
    assert restored.rows[0][2] == 18446744073709551615
    collection = collect_measurements(preview)
    path = save_measurement_collection(collection, tmp_path / "mixed.vipp-results.json")
    loaded = load_measurement_collection(path)
    assert loaded.table.rows[0][0] is None
    assert loaded.table.rows[1][0] == ""
    assert math.isnan(loaded.table.rows[6][0])
    assert math.isinf(loaded.table.rows[7][0])
    assert math.copysign(1, loaded.table.rows[8][0]) == -1


def test_empty_success_is_not_failed_or_missing_and_exclusions_are_reviewed(tmp_path):
    manifest = _manifest(
        tmp_path,
        [_table(()), _table(), _table(), _table()],
        statuses=["completed", "failed", "skipped", "partial"],
    )
    preview = inspect_collection(manifest, "batch_output_1")
    assert [item.status for item in preview.items] == [
        "empty",
        "failed",
        "skipped",
        "partial",
    ]
    assert [item.row_count for item in preview.items] == [0, None, None, None]
    with pytest.raises(MeasurementCollectionError, match="missing, failed"):
        collect_measurements(preview)
    with pytest.raises(MeasurementCollectionError, match="explicitly confirm"):
        collect_measurements(preview, included_ids=["item-1"])
    collection = collect_measurements(
        preview, included_ids=["item-1"], reviewed_exclusions=True
    )
    assert collection.table.rows == ()
    assert len(collection.items) == 4
    assert collection.items[0].included and not collection.items[1].included
    path = save_measurement_collection(collection, tmp_path / "empty.vipp-results.json")
    assert load_measurement_collection(path).table.rows == ()


@pytest.mark.parametrize(
    "change", ["content", "missing", "old", "unsupported", "no-output"]
)
def test_unavailable_outputs_are_never_silently_skipped(tmp_path, change):
    manifest = _manifest(tmp_path, [_table()])
    output = manifest["items"][0]["outputs"][0]
    if change == "content":
        Path(output["path"]).write_text("label,volume\n1,999.0\n", encoding="utf-8")
    elif change == "missing":
        Path(output["path"]).unlink()
    elif change == "old":
        output.pop("table_metadata")
    elif change == "unsupported":
        output["format"] = "npy"
    else:
        manifest["items"].append(
            {
                **deepcopy(manifest["items"][0]),
                "index": 2,
                "batch_id": "image-2",
                "outputs": [],
            }
        )
    preview = inspect_collection(_reseal(manifest), "batch_output_1")
    item = preview.items[-1]
    assert (
        item.status
        == {
            "content": "changed",
            "missing": "missing",
            "old": "unavailable",
            "unsupported": "unsupported",
            "no-output": "missing",
        }[change]
    )
    assert item.row_count is None and item.table is None
    with pytest.raises(MeasurementCollectionError):
        collect_measurements(preview)


@pytest.mark.parametrize("mismatch", ["columns", "units", "types", "kind"])
def test_schema_units_and_types_are_not_coerced_across_tables(tmp_path, mismatch):
    table = _table()
    alternate = {
        "columns": replace(table, columns=("object_id", "volume")),
        "units": replace(table, column_units=(("volume", "voxel^3"),)),
        "types": replace(table, rows=((1.0, 2.5),)),
        "kind": replace(table, table_kind="other"),
    }[mismatch]
    preview = inspect_collection(
        _manifest(tmp_path, [table, alternate]), "batch_output_1"
    )
    assert all(item.status == "ready" for item in preview.items)
    with pytest.raises(MeasurementCollectionError, match="differ|incompatible"):
        collect_measurements(preview)


@pytest.mark.parametrize("name", ["label", "_vipp_item_key", "_vipp_row"])
def test_annotations_cannot_overwrite_existing_fields(tmp_path, name):
    preview = inspect_collection(_manifest(tmp_path), "batch_output_1")
    with pytest.raises(MeasurementCollectionError, match="conflicts"):
        collect_measurements(preview, annotations={"item-1": {name: "changed"}})


def test_duplicate_resume_items_are_rejected_not_appended_twice(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["items"].append(deepcopy(manifest["items"][0]))
    with pytest.raises(MeasurementCollectionError, match="Duplicate"):
        inspect_collection(_reseal(manifest), "batch_output_1")


def test_resume_parser_preserves_new_schema_evidence(tmp_path):
    document = _manifest(tmp_path)["items"][0]
    restored = item_record_from_document(document)
    assert (
        restored.outputs[0].table_metadata == document["outputs"][0]["table_metadata"]
    )
    assert (
        restored.to_dict()["outputs"][0]["table_metadata"]
        == restored.outputs[0].table_metadata
    )


def test_snapshot_round_trip_does_not_reread_original_results_and_binds_exact_bytes(
    tmp_path, monkeypatch
):
    manifest = _manifest(tmp_path)
    preview = inspect_collection(manifest, "batch_output_1")
    collection = collect_measurements(preview)
    target = tmp_path / "review.vipp-results.json"
    assert save_measurement_collection(collection, target) == target
    original_digest = hashlib.sha256(target.read_bytes()).hexdigest()
    monkeypatch.setattr(
        module,
        "capture_local_source_identity",
        lambda *_args, **_kwargs: pytest.fail("Source/result files must not be reread"),
    )
    for item in preview.items:
        Path(item.result_path).unlink()
    loaded = load_measurement_collection(target, expected_sha256=original_digest)
    assert loaded.table == collection.table
    assert loaded.items == collection.items
    assert loaded.file_sha256 == original_digest
    assert loaded.items[0].result_sha256
    with pytest.raises(MeasurementCollectionError, match="file hash"):
        load_measurement_collection(target, expected_sha256="f" * 64)
    raw = json.loads(target.read_text(encoding="utf-8"))
    raw["table"]["rows"][0][0][1] = 99
    target.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(MeasurementCollectionError, match="integrity"):
        load_measurement_collection(target)


@pytest.mark.parametrize(
    "damage", ["row-id", "annotation", "type", "extra", "unit", "inventory"]
)
def test_resealed_invalid_snapshot_schema_is_rejected(tmp_path, damage):
    collection = collect_measurements(
        inspect_collection(_manifest(tmp_path), "batch_output_1"),
        annotations={"item-1": {"condition": "control"}},
    )
    path = save_measurement_collection(collection, tmp_path / "bad.vipp-results.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if damage == "row-id":
        raw["table"]["rows"][1][3][1] = "item-1"
    elif damage == "annotation":
        raw["annotations"]["item-1"]["condition"] = "treated"
    elif damage == "type":
        raw["table"]["rows"][0][0] = ["int", "1"]
    elif damage == "extra":
        raw["execute"] = "arbitrary code"
    elif damage == "unit":
        raw["table"]["column_units"] = [["unknown", "um"]]
    else:
        raw["items"][0]["row_count"] = 200
    path.write_text(json.dumps(_reseal(raw)), encoding="utf-8")
    with pytest.raises(MeasurementCollectionError):
        load_measurement_collection(path)


@pytest.mark.parametrize(
    "text", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', "[1,2,3]"]
)
def test_unsafe_json_is_rejected(tmp_path, text):
    path = tmp_path / "unsafe.vipp-results.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(MeasurementCollectionError):
        load_measurement_collection(path)


def test_paths_and_limits_and_cancellation_are_explicit(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["items"][0]["outputs"][0]["path"] = str(tmp_path / ".." / "outside.csv")
    preview = inspect_collection(_reseal(manifest), "batch_output_1")
    assert preview.items[0].status == "unavailable"
    assert "direct local" in preview.items[0].message
    preview = inspect_collection(
        _manifest(tmp_path),
        "batch_output_1",
        limits=replace(CollectionLimits(), max_file_bytes=2),
    )
    assert all(item.table is None for item in preview.items)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(OperationCancelled):
        inspect_collection(_manifest(tmp_path), "batch_output_1", cancellation=cancel)
    collection = collect_measurements(
        inspect_collection(_manifest(tmp_path), "batch_output_1")
    )
    target = tmp_path / "cancelled.vipp-results.json"
    with pytest.raises(OperationCancelled):
        save_measurement_collection(collection, target, cancellation=cancel)
    assert not target.exists()
    with pytest.raises(MeasurementCollectionError, match="extension"):
        save_measurement_collection(collection, tmp_path / "not-a-snapshot.json")


def test_batch_staging_records_schema_and_content_digest_before_publication(tmp_path):
    from napari_vipp.core import batch

    table = _table(((2**63 + 1, 3.5),))
    output = SimpleNamespace(
        path=tmp_path / "objects.csv",
        recovery_root=None,
        duplicate=False,
        input_collision=False,
        existing_file_policy=batch.ExistingFilePolicy.ERROR,
        node_id="out",
        format="csv",
    )
    pipeline = SimpleNamespace(outputs={"out": table}, output_states={})
    staged = batch._save_planned_output(pipeline, output)
    assert not output.path.exists()
    assert staged.table_metadata["column_types"] == [["int"], ["float"]]
    assert staged.table_metadata["column_units"] == [["volume", "micrometer^3"]]
    assert (
        staged.table_metadata["file_sha256"]
        == hashlib.sha256(staged.saved_temporary_path.read_bytes()).hexdigest()
    )
    assert "9223372036854775809" not in json.dumps(staged.table_metadata)
    batch._cleanup_staged_output(staged)


def test_changed_during_table_read_is_reported_and_not_collectable(
    tmp_path, monkeypatch
):
    manifest = _manifest(tmp_path, [_table()])
    original_read = module._read_bytes

    def read_then_change(path, *args):
        data = original_read(path, *args)
        Path(path).write_bytes(data.replace(b"2.5", b"9.5"))
        return data

    monkeypatch.setattr(module, "_read_bytes", read_then_change)
    preview = inspect_collection(manifest, "batch_output_1")
    assert preview.items[0].status == "changed"
    assert preview.items[0].table is None
    with pytest.raises(MeasurementCollectionError):
        collect_measurements(preview)


def test_cancelled_after_staging_never_publishes_and_keeps_existing_snapshot(
    tmp_path, monkeypatch
):
    collection = collect_measurements(
        inspect_collection(_manifest(tmp_path), "batch_output_1")
    )
    path = tmp_path / "existing.vipp-results.json"
    path.write_bytes(b"keep existing snapshot")
    token = threading.Event()
    real_fsync = module.os.fsync

    def finish_then_cancel(descriptor):
        real_fsync(descriptor)
        token.set()

    monkeypatch.setattr(module.os, "fsync", finish_then_cancel)
    with pytest.raises(OperationCancelled):
        save_measurement_collection(collection, path, cancellation=token)
    assert path.read_bytes() == b"keep existing snapshot"
    assert not list(tmp_path.glob(".existing.vipp-results.json.*.tmp"))


def test_multisource_selectors_hashes_and_authored_overrides_survive_snapshot(tmp_path):
    manifest = _manifest(tmp_path, [_table()])
    record = manifest["items"][0]
    selector = {"key": "series:1", "kind": "series", "axis_declaration": None}
    record["sources"] = [
        {
            "node_id": "input",
            "title": "Green",
            "role": "collection",
            "path": "green.tif",
            "identity": {
                "kind": "file",
                "sha256": "b" * 64,
                "regular_file_count": 1,
                "size_bytes": 10,
            },
            "series": {
                "index": 1,
                "key": "series:1",
                "name": "Green scene",
                "shape": [2, 3],
            },
            "source_item": {"selector": selector},
            "image_state": {"private": "omitted"},
        },
        {
            "node_id": "input_2",
            "title": "Red",
            "role": "fixed",
            "path": "red.tif",
            "identity": {
                "kind": "file",
                "sha256": "c" * 64,
                "regular_file_count": 1,
                "size_bytes": 20,
            },
        },
    ]
    record["effective_workflow_sha256"] = "d" * 64
    record["execution_provenance_sha256"] = "e" * 64
    record["parameter_overrides"] = {
        "identity": "batch-override",
        "source_item_key": "f" * 64,
        "values": [
            {
                "node_id": "threshold",
                "operation_id": "binary_threshold",
                "parameter": "threshold",
                "workflow_value": 10,
                "resolved_value": 20,
            }
        ],
    }
    collection = collect_measurements(
        inspect_collection(_reseal(manifest), "batch_output_1")
    )
    path = save_measurement_collection(collection, tmp_path / "multi.vipp-results.json")
    loaded = load_measurement_collection(path)
    item = loaded.items[0]
    assert [source["node_id"] for source in item.sources] == ["input", "input_2"]
    assert item.sources[0]["selector"] == selector
    assert item.sources[1]["identity"]["sha256"] == "c" * 64
    assert item.sources[0]["source_item_record_sha256"]
    assert "image_state" not in item.sources[0]
    assert "shape" not in item.sources[0]["series"]
    assert item.parameter_overrides["values"][0]["resolved_value"] == 20
    assert item.effective_workflow_sha256 == "d" * 64
    assert item.execution_provenance_sha256 == "e" * 64
    with pytest.raises(TypeError):
        item.sources[0]["selector"]["key"] = "changed"


@pytest.mark.parametrize(
    "damage", ["duplicate-header", "typed-cell", "row-count", "missing-unit-evidence"]
)
def test_recorded_csv_schema_is_enforced_even_when_document_resealed(tmp_path, damage):
    manifest = _manifest(tmp_path, [_table()])
    metadata = manifest["items"][0]["outputs"][0]["table_metadata"]
    if damage == "duplicate-header":
        metadata["columns"] = ["label", "label"]
    elif damage == "typed-cell":
        metadata["cell_types"][0] = [["bool", 1]]
        metadata["column_types"][0] = ["bool"]
    elif damage == "row-count":
        metadata["row_count"] = 0
    else:
        metadata.pop("column_units")
    preview = inspect_collection(_reseal(manifest), "batch_output_1")
    assert preview.items[0].table is None
    assert preview.items[0].status == "unavailable"


def test_manifest_path_seal_and_finality_are_checked(tmp_path):
    manifest = _manifest(tmp_path)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert len(inspect_collection(path, "batch_output_1").items) == 2
    manifest["finished_at"] = ""
    with pytest.raises(MeasurementCollectionError, match="finished"):
        inspect_collection(_reseal(manifest), "batch_output_1")
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(MeasurementCollectionError, match="integrity"):
        inspect_collection(path, "batch_output_1")


def test_real_batch_archive_collect_snapshot_and_verified_resume_without_duplicates(
    tmp_path, monkeypatch
):
    from napari_vipp.core import batch
    from napari_vipp.core.batch_resume import run_batch_resume_from_manifest
    from napari_vipp.core.pipeline import PrototypePipeline
    from napari_vipp.core.workflow import serialize_workflow

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    source = np.zeros((12, 12), dtype=np.uint16)
    source[1:3, 1:4] = 1000
    source[7:9, 7:10] = 1000
    np.save(inputs / "a.npy", source)
    source[7:9, 7:10] = 0
    np.save(inputs / "b.npy", source)
    np.save(inputs / "c.npy", np.zeros_like(source))
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    threshold = pipeline.add_node("binary_threshold")
    pipeline.set_param(threshold.id, "threshold", 100)
    labels = pipeline.add_node("label_connected_components")
    measure = pipeline.add_node("measure_objects")
    output = pipeline.add_node("batch_output")
    for source_id, target_id in (
        ("input", threshold.id),
        (threshold.id, labels.id),
        (labels.id, measure.id),
        (measure.id, output.id),
    ):
        assert pipeline.connect(source_id, target_id).success
    pipeline.set_param(output.id, "format", "csv")
    pipeline.set_param(output.id, "tag", "objects")
    workflow = serialize_workflow(pipeline)
    config = batch.BatchConfig(
        workflow_file=Path("workflow.json"),
        workflow_sha256=batch.scientific_workflow_hash(workflow),
        output_dir=tmp_path / "outputs",
        sources=(batch.BatchSourceConfig("input", "Input", inputs, "*.npy"),),
        outputs=(
            batch.BatchOutputConfig(
                output.id,
                output.title,
                "objects",
                "table",
                "csv",
                "",
                "{source_stem}__{tag}",
            ),
        ),
        save_python_script=False,
    )
    result = batch.run_batch(workflow, config)
    assert not result.has_failures
    assert result.summary["completed"] == 3
    assert result.manifest_archive_path is not None
    preview = inspect_collection(result.manifest_archive_path, output.id)
    assert [item.row_count for item in preview.items] == [2, 1, 0], [
        item.message for item in preview.items
    ]
    assert all(item.sources[0]["identity"]["sha256"] for item in preview.items)
    collection = collect_measurements(
        preview,
        annotations={
            "item-1": {"condition": "control"},
            "item-2": {"condition": "treated"},
            "item-3": {"condition": "empty control"},
        },
    )
    snapshot = save_measurement_collection(
        collection, tmp_path / "actual.vipp-results.json"
    )
    loaded = load_measurement_collection(snapshot)
    assert loaded.table.row_count == 3
    assert loaded.items[2].status == "empty" and loaded.items[2].included
    # Direct exports use the reviewed dataset, including annotations for images
    # with no object rows. They do not reopen inputs or require a results graph.
    from napari_vipp.core.measurement_export import export_measurement_collection

    with monkeypatch.context() as no_images:
        no_images.setattr(
            np, "load", lambda *_a, **_k: pytest.fail("Export must not load images")
        )
        for format_name, delimiter in (("csv", ","), ("tsv", "\t")):
            exported = export_measurement_collection(
                loaded, tmp_path / f"collected.{format_name}",
                include_image_summary=True,
            )
            with exported.paths[0].open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream, delimiter=delimiter))
            assert len(rows) == 3
            assert [row["condition"] for row in rows] == [
                "control", "control", "treated",
            ]
            with exported.paths[1].open(encoding="utf-8-sig", newline="") as stream:
                summary = list(csv.reader(stream, delimiter=delimiter))
            assert len(summary) == 4  # Header + all three original images.
            assert "empty control" in summary[-1]
        exported = export_measurement_collection(
            loaded, tmp_path / "collected.xlsx"
        )
        assert len(exported.paths) == 1
        assert exported.paths[0].is_file()
    assert load_measurement_collection(snapshot).table.rows == loaded.table.rows
    original_hashes = [item.result_sha256 for item in preview.items]
    monkeypatch.setattr(
        batch,
        "execute_pipeline_request",
        lambda *_args, **_kwargs: pytest.fail(
            "Verified resume must not recalculate completed images"
        ),
    )
    resumed = run_batch_resume_from_manifest(result.manifest_archive_path)
    assert resumed.summary["completed"] == 3
    assert resumed.saved_paths == ()
    repeated_preview = inspect_collection(resumed.manifest_archive_path, output.id)
    assert [item.result_sha256 for item in repeated_preview.items] == original_hashes
    assert all(
        item.resumed_from_run_id == result.manifest.run_id
        for item in repeated_preview.items
    )
    repeated = collect_measurements(repeated_preview)
    assert repeated.table.row_count == 3
    assert len(repeated.items) == 3
    assert (
        len(
            {
                (row["_vipp_item_key"], row["_vipp_row"])
                for row in repeated.table.records()
            }
        )
        == 3
    )


@pytest.mark.parametrize("location", ["input", "non-table-extension"])
def test_collector_never_opens_a_source_or_non_table_file(
    tmp_path, monkeypatch, location
):
    manifest = _manifest(tmp_path, [_table()])
    item = manifest["items"][0]
    output = item["outputs"][0]
    if location == "input":
        item["sources"][0]["path"] = output["path"]
    else:
        output["path"] = str(tmp_path / "image.npy")
    monkeypatch.setattr(
        module,
        "capture_local_source_identity",
        lambda *_args, **_kwargs: pytest.fail(
            "Source/non-table bytes must not be opened"
        ),
    )
    preview = inspect_collection(_reseal(manifest), "batch_output_1")
    assert preview.items[0].status == "unavailable"
    assert preview.items[0].table is None


@pytest.mark.parametrize("phase", ["metadata", "collect", "snapshot-decode"])
def test_resident_row_processing_observes_cancellation(tmp_path, phase):
    table = _table(tuple((index, float(index)) for index in range(100)))
    calls = 0

    def cancelled_after_some_rows():
        nonlocal calls
        calls += 1
        return calls > 15

    if phase == "metadata":
        operation = partial(
            table_measurement_metadata, table, cancellation=cancelled_after_some_rows
        )
    else:
        preview = inspect_collection(_manifest(tmp_path, [table]), "batch_output_1")
        if phase == "collect":
            operation = partial(
                collect_measurements, preview, cancellation=cancelled_after_some_rows
            )
        else:
            path = save_measurement_collection(
                collect_measurements(preview),
                tmp_path / "cancel-read.vipp-results.json",
            )
            operation = partial(
                load_measurement_collection,
                path,
                cancellation=cancelled_after_some_rows,
            )
    with pytest.raises(OperationCancelled):
        operation()
    assert calls == 16
