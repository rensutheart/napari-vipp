"""Detection contracts across execution, planning, export, batch and cache."""

import json
import threading
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import tifffile

from napari_vipp.core.batch import (
    BatchConfig,
    BatchOutputConfig,
    BatchSourceConfig,
    BatchStatus,
    run_batch,
    scientific_workflow_hash,
)
from napari_vipp.core.compute import ComputeRequest
from napari_vipp.core.execution import PipelineRunRequest, execute_pipeline_request
from napari_vipp.core.export import export_pipeline_to_python
from napari_vipp.core.measurement_collection import (
    CollectionLimits,
    MeasurementCollectionError,
    _read_table,
    collect_measurements,
    inspect_collection,
    load_measurement_collection,
    save_measurement_collection,
    table_measurement_metadata,
)
from napari_vipp.core.metadata import AxisMetadata, ImageState, image_state_from_array
from napari_vipp.core.pipeline import PrototypePipeline, SourcePayload
from napari_vipp.core.source_identity import SourceRevisionToken
from napari_vipp.core.tables import save_table_output
from napari_vipp.core.workflow import deserialize_workflow, serialize_workflow


def _image():
    rng = np.random.default_rng(4072026)
    image = rng.normal(0, 0.01, (40, 44))
    template = rng.normal(0, 1, (4, 6))
    image[5:9, 7:13] = template
    image[23:27, 29:35] = template * 2 + 0.3
    image.setflags(write=False)
    return image


def _payload(revision=1, data=None):
    data = _image() if data is None else data
    state = image_state_from_array(
        data,
        axes=(
            AxisMetadata("y", "space", "micrometer", 0.7, -5),
            AxisMetadata("x", "space", "micrometer", 0.4, 9),
        ),
        source_name="Known two-pattern image",
    )
    return SourcePayload(
        data,
        {"axes": "YX"},
        state.source_name,
        image_state=state,
        revision_token=SourceRevisionToken(layer_id=9407, revision=revision),
    )


def _pipeline():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    crop = pipeline.add_node("crop_stack")
    for key, value in {"top": 5, "bottom": 31, "left": 7, "right": 31}.items():
        pipeline.set_param(crop.id, key, value)
    match = pipeline.add_node("template_match")
    peaks = pipeline.add_node("find_peaks")
    pipeline.set_param(peaks.id, "use_mask", True)
    pipeline.set_param(peaks.id, "minimum_value", 0.9999)
    assert pipeline.connect("input", crop.id).success
    assert pipeline.connect("input", match.id, target_port=0).success
    assert pipeline.connect(crop.id, match.id, target_port=1).success
    assert pipeline.connect(match.id, peaks.id, source_port=0, target_port=0).success
    assert pipeline.connect(match.id, peaks.id, source_port=1, target_port=1).success
    return pipeline, match.id, peaks.id


def _request(pipeline, payload=None, *, mode="cpu", **kwargs):
    return PipelineRunRequest(
        run_id=1,
        workflow=serialize_workflow(pipeline),
        input_data=None,
        input_metadata=None,
        input_name="",
        source_payloads={"input": payload or _payload()},
        compute_request=ComputeRequest(mode=mode),
        manual_node_ids=frozenset(pipeline.manual_node_ids()),
        **kwargs,
    )


def _cache(pipeline):
    return dict(
        cached_outputs=dict(pipeline.outputs),
        cached_output_states=dict(pipeline.output_states),
        cached_node_outputs={k: list(v) for k, v in pipeline.node_outputs.items()},
        cached_node_output_states={
            k: list(v) for k, v in pipeline.node_output_states.items()
        },
        completed_node_ids=frozenset(pipeline.completed_node_ids),
        cached_execution_states=dict(pipeline.node_execution_states),
        cached_execution_messages=dict(pipeline.node_execution_messages),
        cached_compute_provenance={
            **pipeline.node_cache_lineage,
            **pipeline.node_compute_provenance,
        },
    )


@pytest.mark.parametrize("mode", ["cpu", "prefer_gpu"])
def test_even_template_execution_keeps_centres_calibration_and_population(mode):
    pipeline, match, peaks = _pipeline()
    payload = _payload()
    before = payload.data.copy()
    result = execute_pipeline_request(
        _request(pipeline, payload, mode=mode), raise_errors=True
    ).pipeline
    table = result.outputs[peaks]
    assert table.row_count == 2
    rows = table.records()
    assert {(r["y_index"], r["x_index"]) for r in rows} == {(6.5, 9.5), (24.5, 31.5)}
    for row in rows:
        assert row["y_physical"] == pytest.approx(-5 + row["y_index"] * 0.7)
        assert row["x_physical"] == pytest.approx(9 + row["x_index"] * 0.4)
        assert row["template_y_stop"] - row["template_y_start"] == 4
        assert row["template_x_stop"] - row["template_x_start"] == 6
        assert row["score"] == pytest.approx(1)
    score, valid = result.node_outputs[match]
    score_state, mask_state = result.node_output_states[match]
    assert score.shape == valid.shape == (37, 39)
    assert valid.dtype == bool
    assert score_state.axes[0].translation == pytest.approx(-3.95)
    assert score_state.axes[1].translation == pytest.approx(10)
    assert score_state.template_match_metadata == mask_state.template_match_metadata
    assert ImageState.from_dict(score_state.to_dict()) == score_state
    evidence = table.detection_metadata
    assert evidence.returned_count == evidence.accepted_count == 2
    assert not evidence.truncated
    assert result.output_states[peaks].detection_metadata == evidence
    assert deepcopy(table) == table
    np.testing.assert_array_equal(payload.data, before)
    assert not payload.data.flags.writeable


def test_workflow_roundtrip_dynamic_mask_ports():
    pipeline, match, peaks = _pipeline()
    restored = deserialize_workflow(serialize_workflow(pipeline))
    clone = PrototypePipeline()
    clone.restore_graph(
        restored["nodes"], restored["connections"], restored["output_tunnels"]
    )
    assert len(clone.input_ports(peaks)) == 2
    assert [p.output_type for p in clone.output_ports(match)] == ["image", "mask"]
    clone.set_param(peaks, "use_mask", False)
    assert len(clone.input_ports(peaks)) == 1
    assert not any(
        c.target_id == peaks and c.target_port == 1 for c in clone.connections
    )
    with pytest.raises(ValueError, match="valid-score mask"):
        execute_pipeline_request(_request(clone), raise_errors=True)


def test_metadata_only_planning_never_calls_detection_kernels(monkeypatch):
    from napari_vipp.core import detection

    def forbidden(*args, **kwargs):
        raise AssertionError("Scientific kernel used during metadata-only planning")

    monkeypatch.setattr(detection, "template_match", forbidden)
    monkeypatch.setattr(detection, "template_match_arrays", forbidden)
    monkeypatch.setattr(detection, "match_template", forbidden)
    monkeypatch.setattr(detection, "find_peaks", forbidden)
    pipeline, match, _peaks = _pipeline()
    pipeline.preflight_axis_contract({"input": _payload()})
    state = pipeline.node_output_states[match][0]
    assert state.shape == (37, 39)
    assert state.axes[0].translation == pytest.approx(-3.95)
    assert state.template_match_metadata is None  # no invented numerical evidence


def test_generated_python_exports_detection_table_with_carried_evidence(tmp_path):
    pipeline, _match, peaks = _pipeline()
    namespace = {"__name__": "detection_export"}
    exec(
        compile(export_pipeline_to_python(pipeline), "<detection export>", "exec"),
        namespace,
    )
    results = namespace["run_pipeline"](source_payloads={"input": _payload()})
    table = results[peaks]
    assert table.row_count == 2
    path = namespace["save_image"](
        table, tmp_path / "detections.csv", provenance=results, output_node_id=peaks
    )
    assert path.suffix == ".csv"
    sidecar = json.loads(
        Path(str(path) + ".vipp-provenance.json").read_text(encoding="utf-8")
    )
    assert sidecar["workflow"]["sha256"] == results.workflow_sha256
    assert sidecar["output"]["node_id"] == peaks
    # The table state, not rounded display strings, carries scientific evidence.
    assert results.output_states[peaks].detection_metadata == table.detection_metadata


def test_csv_manifest_roundtrip_retains_cap_and_empty_detection_evidence(tmp_path):
    pipeline, _match, peaks = _pipeline()
    for threshold, cap, expected in ((0.9999, 1, 1), (1.1, 2, 0)):
        pipeline.set_param(peaks, "minimum_value", threshold)
        pipeline.set_param(peaks, "maximum_detections", cap)
        result = execute_pipeline_request(
            _request(pipeline), raise_errors=True
        ).pipeline
        table = result.outputs[peaks]
        assert table.row_count == expected
        path = tmp_path / f"detections-{expected}.csv"
        save_table_output(table, path, format="csv")
        metadata = table_measurement_metadata(table)
        restored = _read_table(
            path.read_bytes(), metadata, "csv", CollectionLimits(), None
        )
        # The batch item supplies source display names separately; exact cells,
        # units and scientific identity/coordinate evidence belong to the table.
        assert replace(restored, source_name=table.source_name) == table
        assert restored.detection_metadata.truncated == (expected == 1)
        broken = deepcopy(metadata)
        broken["detection_metadata"]["returned_count"] = 999
        with pytest.raises(MeasurementCollectionError, match="detection"):
            _read_table(path.read_bytes(), broken, "csv", CollectionLimits(), None)


def test_peak_only_edit_reuses_scores_and_source_change_invalidates_them():
    pipeline, match, peaks = _pipeline()
    first = execute_pipeline_request(_request(pipeline), raise_errors=True).pipeline
    old_table = first.outputs[peaks]
    old_metadata = first.output_states[match].template_match_metadata
    first.set_param(peaks, "maximum_detections", 1)
    started = []
    changed = execute_pipeline_request(
        _request(first, dirty_node_ids=frozenset({peaks}), **_cache(first)),
        node_started_callback=started.append,
        raise_errors=True,
    ).pipeline
    assert match not in started
    assert changed.outputs[peaks].row_count == 1
    assert changed.outputs[peaks].detection_metadata.accepted_count == 2
    assert old_table.row_count == 2
    revised = _image().copy()
    revised[23:27, 29:35] = 0
    revised.setflags(write=False)
    started.clear()
    rerun = execute_pipeline_request(
        _request(
            changed,
            _payload(2, revised),
            dirty_node_ids=frozenset({peaks}),
            **_cache(changed),
        ),
        node_started_callback=started.append,
        raise_errors=True,
    ).pipeline
    assert match in started
    assert (
        rerun.output_states[match].template_match_metadata.search_revision
        != old_metadata.search_revision
    )
    assert rerun.outputs[peaks].detection_metadata.accepted_count == 1


@pytest.mark.parametrize(
    "missing", ["detection_id", "score", "y_index", "template_x_stop"]
)
def test_collection_rejects_detection_evidence_without_its_required_cells(
    tmp_path, missing
):
    pipeline, _match, peaks = _pipeline()
    table = execute_pipeline_request(
        _request(pipeline), raise_errors=True
    ).pipeline.outputs[peaks]
    index = table.columns.index(missing)
    incomplete = replace(
        table,
        columns=tuple(c for c in table.columns if c != missing),
        rows=tuple(row[:index] + row[index + 1 :] for row in table.rows),
        column_units=tuple((c, u) for c, u in table.column_units if c != missing),
    )
    path = save_table_output(incomplete, tmp_path / "incomplete.csv", format="csv")
    with pytest.raises(MeasurementCollectionError, match="Detection evidence"):
        _read_table(
            path.read_bytes(),
            table_measurement_metadata(incomplete),
            "csv",
            CollectionLimits(),
            None,
        )


def test_cancelled_matching_exposes_no_partial_score_or_detection():
    pipeline, match, peaks = _pipeline()
    cancelled = threading.Event()

    def progress(node_id, current, total, message):
        if node_id == match and current >= 1:
            cancelled.set()

    result = execute_pipeline_request(
        _request(pipeline, cancel_event=cancelled), progress_callback=progress
    )
    assert cancelled.is_set() and result.cancelled
    assert result.failure.kind == "cancelled"
    if result.pipeline is not None:
        assert result.pipeline.outputs.get(peaks) is None
        assert all(v is None for v in result.pipeline.node_outputs.get(match, ()))


def test_batch_exports_detection_coordinates_and_exact_population_manifest(tmp_path):
    pipeline, _match, peaks = _pipeline()
    pipeline.set_param(peaks, "maximum_detections", 1)
    output = pipeline.add_node("batch_output")
    pipeline.set_param(output.id, "tag", "detections")
    pipeline.set_param(output.id, "format", "csv")
    assert pipeline.connect(peaks, output.id).success
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    tifffile.imwrite(
        inputs / "patterns.ome.tif",
        _image(),
        ome=True,
        metadata={"axes": "YX", "PhysicalSizeY": 0.7, "PhysicalSizeX": 0.4},
        photometric="minisblack",
    )
    workflow = serialize_workflow(pipeline)
    config = BatchConfig(
        workflow_file=Path("workflow.json"),
        workflow_sha256=scientific_workflow_hash(workflow),
        output_dir=tmp_path / "outputs",
        sources=(BatchSourceConfig("input", "Pattern images", inputs, "*.ome.tif"),),
        outputs=(
            BatchOutputConfig(
                output.id,
                "Batch Output",
                "detections",
                "table",
                "csv",
                "",
                "{source_stem}__{tag}",
            ),
        ),
        compute_request=ComputeRequest(mode="cpu"),
        save_python_script=True,
    )
    result = run_batch(workflow, config)
    assert result.manifest.items[0].status is BatchStatus.COMPLETED
    assert len(result.saved_paths) == 1
    manifest = result.manifest.to_dict()
    evidence = manifest["items"][0]["outputs"][0]["table_metadata"][
        "detection_metadata"
    ]
    assert evidence["accepted_count"] == 2
    assert evidence["returned_count"] == 1 and evidence["truncated"]
    assert evidence["source_scale"] == [0.7, 0.4]
    assert result.saved_paths[0].suffix == ".csv"
    preview = inspect_collection(manifest, output.id)
    assert preview.items[0].detection_metadata["truncated"]
    collection = collect_measurements(preview)
    assert collection.table.detection_metadata is None  # not one image's grid
    path = save_measurement_collection(
        collection, tmp_path / "collected.vipp-results.json"
    )
    restored = load_measurement_collection(path)
    assert dict(restored.items[0].detection_metadata) == dict(
        collection.items[0].detection_metadata
    )
    assert restored.items[0].detection_metadata["accepted_count"] == 2
    assert restored.table.rows == collection.table.rows


@pytest.mark.parametrize(
    "format,suffix",
    [("tiff", ".tif"), ("ome-tiff", ".ome.tif"), ("ome-zarr-0.4", ".ome.zarr")],
)
def test_saved_paired_scores_keep_evidence_after_frozen_source_loading(
    tmp_path, format, suffix
):
    from napari_vipp.core.detection import find_peaks
    from napari_vipp.core.file_sources import load_frozen_file_source_snapshot
    from napari_vipp.core.io import write_image

    pipeline, match, peaks = _pipeline()
    result = execute_pipeline_request(_request(pipeline), raise_errors=True).pipeline
    snapshots = []
    for index, (data, state) in enumerate(
        zip(result.node_outputs[match], result.node_output_states[match], strict=True)
    ):
        path = write_image(
            data, tmp_path / f"port-{index}{suffix}", format=format, image_state=state
        )
        snapshot = load_frozen_file_source_snapshot(path, 0)
        assert snapshot.payload.image_state.template_match_metadata == (
            state.template_match_metadata
        )
        np.testing.assert_array_equal(snapshot.payload.data, data)
        snapshots.append(snapshot.payload)
    table = find_peaks(
        snapshots[0].data,
        snapshots[1].data,
        image_state=snapshots[0].image_state,
        mask_state=snapshots[1].image_state,
        minimum_value=0.9999,
    )
    assert table.records() == result.outputs[peaks].records()
    assert table.detection_metadata == result.outputs[peaks].detection_metadata
