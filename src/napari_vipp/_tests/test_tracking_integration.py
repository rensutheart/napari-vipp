"""Time-series evidence across shared execution, persistence and export."""

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
from napari_vipp.core.metadata import AxisMetadata, image_state_from_array
from napari_vipp.core.operations import add_metadata_columns, select_table_columns
from napari_vipp.core.pipeline import PrototypePipeline, SourcePayload
from napari_vipp.core.source_identity import SourceRevisionToken
from napari_vipp.core.tables import save_table_output
from napari_vipp.core.workflow import deserialize_workflow, serialize_workflow


def series(*, labels=False, volume=False):
    data = np.zeros(
        (4, 6, 24, 30) if volume else (4, 24, 30),
        dtype=np.uint16 if labels else np.float32,
    )
    for t in range(4):
        for n, (y, x) in enumerate(((5 + t, 6 + t), (17 - t, 22 - t)), 1):
            data[(t, 2, y, x) if volume else (t, y, x)] = n if labels else 10 + n
    data.setflags(write=False)
    return data


def payload(*, labels=False, volume=False, revision=1, data=None):
    data = series(labels=labels, volume=volume) if data is None else data
    spatial = ("z", "y", "x") if volume else ("y", "x")
    axes = (AxisMetadata("t", "time", "second", 2.0, 5.0),) + tuple(
        AxisMetadata(a, "space", "micrometer", s, origin)
        for a, s, origin in zip(
            spatial,
            (1.5, 0.7, 0.4)[-len(spatial) :],
            (-2, 10, -5)[-len(spatial) :],
            strict=True,
        )
    )
    state = image_state_from_array(data, axes=axes, source_name="Known time-series")
    return SourcePayload(
        data,
        {"axes": "T" + "".join(spatial).upper()},
        state.source_name,
        image_state=state,
        revision_token=SourceRevisionToken(layer_id=9471, revision=revision),
    )


def pipeline_for(*, labels=False):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    detector = pipeline.add_node(
        "measure_objects" if labels else "detect_spots_per_frame"
    )
    linker = pipeline.add_node("build_tracks")
    pipeline.set_param(linker.id, "maximum_displacement", 3.0)
    pipeline.set_param(linker.id, "distance_units", "Physical (micrometers)")
    if labels:
        threshold = pipeline.add_node("binary_threshold")
        objects = pipeline.add_node("label_connected_components")
        assert pipeline.connect("input", threshold.id).success
        assert pipeline.connect(threshold.id, objects.id).success
        assert pipeline.connect(objects.id, detector.id).success
    else:
        assert pipeline.connect("input", detector.id).success
    assert pipeline.connect(detector.id, linker.id).success
    return pipeline, detector.id, linker.id


def request(pipeline, source=None, **kwargs):
    return PipelineRunRequest(
        run_id=1,
        workflow=serialize_workflow(pipeline),
        input_data=None,
        input_metadata=None,
        input_name="",
        source_payloads={"input": source or payload()},
        compute_request=ComputeRequest(mode="cpu"),
        manual_node_ids=frozenset(pipeline.manual_node_ids()),
        **kwargs,
    )


def cache(pipeline):
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


@pytest.mark.parametrize("labels", [False, True])
@pytest.mark.parametrize("volume", [False, True])
def test_detection_or_labels_shared_executor_keeps_grid_and_exact_tracks(
    labels, volume
):
    pipeline, detector, linker = pipeline_for(labels=labels)
    source = payload(labels=labels, volume=volume)
    before = source.data.copy()
    result = execute_pipeline_request(
        request(pipeline, source), raise_errors=True
    ).pipeline
    observations = result.outputs[detector]
    evidence = observations.observation_metadata
    assert evidence.frame_count == 4
    assert [f.retained_count for f in evidence.frame_populations] == [2] * 4
    assert evidence.time_scale == 2 and evidence.time_origin == 5
    assert evidence.source_origin[-2:] == (10, -5)
    assert evidence.id_column == ("label_id" if labels else "detection_id")
    tracked, summary = result.node_outputs[linker]
    assert summary.row_count == 2 and tracked.row_count == 8
    assert sorted(r["observation_count"] for r in summary.records()) == [4, 4]
    assert all(r["duration"] == 6 for r in summary.records())
    expected_step = np.hypot(0.7, 0.4)
    assert all(
        r["path_length"] == pytest.approx(3 * expected_step) for r in summary.records()
    )
    assert tracked.observation_metadata == evidence
    assert result.node_output_states[linker][0].observation_metadata == evidence
    assert (
        result.node_output_states[linker][1].tracking_metadata
        == summary.tracking_metadata
    )
    assert (
        result.node_output_states[linker][1].to_dict()["tracking_metadata"][
            "source_observations"
        ]["time_origin"]
        == 5
    )
    np.testing.assert_array_equal(source.data, before)
    assert not source.data.flags.writeable


def test_series_template_ports_and_workflow_roundtrip():
    pipeline, detector, _ = pipeline_for()
    assert pipeline.input_port_count(detector) == 1
    pipeline.set_param(detector, "mode", "Template match")
    assert pipeline.input_port_count(detector) == 2
    assert pipeline.connect("input", detector, target_port=1).success
    document = serialize_workflow(pipeline)
    restored = deserialize_workflow(document)
    assert len([c for c in restored["connections"] if c.target_id == detector]) == 2
    pipeline.set_param(detector, "mode", "Local peaks")
    assert pipeline.input_port_count(detector) == 1
    assert not any(
        c.target_id == detector and c.target_port == 1 for c in pipeline.connections
    )


def test_generated_python_uses_same_linker_and_export_evidence(tmp_path):
    pipeline, _, linker = pipeline_for()
    namespace = {"__name__": "tracking_export"}
    exec(
        compile(export_pipeline_to_python(pipeline), "<tracking export>", "exec"),
        namespace,
    )
    results = namespace["run_pipeline"](source_payloads={"input": payload()})
    table = results[linker]
    assert table.row_count == 8 and table.tracking_metadata is not None
    path = namespace["save_image"](
        table, tmp_path / "tracks.csv", provenance=results, output_node_id=linker
    )
    sidecar = json.loads(
        Path(str(path) + ".vipp-provenance.json").read_text(encoding="utf-8")
    )
    assert sidecar["workflow"]["sha256"] == results.workflow_sha256
    assert results.output_states[linker].tracking_metadata == table.tracking_metadata


@pytest.mark.parametrize("which", ["detections", "capped", "tracks", "summary"])
def test_exact_csv_manifest_roundtrip_includes_time_and_tracking(tmp_path, which):
    pipeline, detector, linker = pipeline_for()
    if which == "capped":
        pipeline.remove_node(linker)
        pipeline.set_param(detector, "maximum_detections", 1)
    result = execute_pipeline_request(request(pipeline), raise_errors=True).pipeline
    table = (
        result.outputs[detector]
        if which in {"detections", "capped"}
        else result.node_outputs[linker][which == "summary"]
    )
    path = save_table_output(table, tmp_path / "table.csv", format="csv")
    meta = table_measurement_metadata(table)
    restored = _read_table(path.read_bytes(), meta, "csv", CollectionLimits(), None)
    assert replace(restored, source_name=table.source_name) == table
    if which != "summary":
        broken = deepcopy(meta)
        broken["observation_metadata"]["frame_populations"][0]["retained_count"] = 900
        with pytest.raises(MeasurementCollectionError, match="evidence|observation"):
            _read_table(path.read_bytes(), broken, "csv", CollectionLimits(), None)


def test_annotation_preserves_evidence_and_coordinate_edit_clears_it():
    pipeline, detector, linker = pipeline_for()
    result = execute_pipeline_request(request(pipeline), raise_errors=True).pipeline
    for table in (result.outputs[detector], result.outputs[linker]):
        annotated = add_metadata_columns(table, "condition=test")
        assert annotated.observation_metadata == table.observation_metadata
        assert annotated.tracking_metadata == table.tracking_metadata
        edited = add_metadata_columns(table, "t_index=2", overwrite="yes")
        assert edited.observation_metadata is edited.tracking_metadata is None
        assert "evidence cleared" in edited.table_kind
        selected = select_table_columns(
            table, columns="t_index", selection_mode="Drop listed columns"
        )
        assert selected.observation_metadata is None


def test_link_parameter_edit_reuses_detection_but_source_revision_invalidates_it():
    pipeline, detector, linker = pipeline_for()
    first = execute_pipeline_request(request(pipeline), raise_errors=True).pipeline
    first.set_param(linker, "maximum_displacement", 4.0)
    started = []
    second = execute_pipeline_request(
        request(first, dirty_node_ids=frozenset({linker}), **cache(first)),
        node_started_callback=started.append,
        raise_errors=True,
    ).pipeline
    assert detector not in started and linker in started
    changed = series().copy()
    changed[2, 7, 8] = 0
    changed.setflags(write=False)
    started.clear()
    third = execute_pipeline_request(
        request(
            second,
            payload(data=changed, revision=2),
            dirty_node_ids=frozenset({linker}),
            **cache(second),
        ),
        node_started_callback=started.append,
        raise_errors=True,
    ).pipeline
    assert detector in started
    assert (
        third.outputs[detector].observation_metadata.source_revision
        != first.outputs[detector].observation_metadata.source_revision
    )


def test_cancel_mid_series_has_no_partial_detector_or_tracks():
    pipeline, detector, linker = pipeline_for()
    cancelled = threading.Event()

    def progress(node_id, current, total, message):
        if node_id == detector and current >= 1000:
            cancelled.set()

    result = execute_pipeline_request(
        request(pipeline, cancel_event=cancelled), progress_callback=progress
    )
    assert result.cancelled
    if result.pipeline is not None:
        assert result.pipeline.outputs.get(detector) is None
        assert result.pipeline.outputs.get(linker) is None


def test_batch_tables_record_item_evidence_without_merging_source_frames(tmp_path):
    pipeline, _, linker = pipeline_for()
    output = pipeline.add_node("batch_output")
    pipeline.set_param(output.id, "tag", "tracks")
    pipeline.set_param(output.id, "format", "csv")
    assert pipeline.connect(linker, output.id).success
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    tifffile.imwrite(
        inputs / "series.ome.tif",
        series(),
        ome=True,
        photometric="minisblack",
        metadata={
            "axes": "TYX",
            "TimeIncrement": 2.0,
            "TimeIncrementUnit": "s",
            "PhysicalSizeY": 0.7,
            "PhysicalSizeX": 0.4,
        },
    )
    document = serialize_workflow(pipeline)
    config = BatchConfig(
        workflow_file=Path("workflow.json"),
        workflow_sha256=scientific_workflow_hash(document),
        output_dir=tmp_path / "outputs",
        sources=(BatchSourceConfig("input", "Series", inputs, "*.ome.tif"),),
        outputs=(
            BatchOutputConfig(
                output.id,
                "Batch Output",
                "tracks",
                "table",
                "csv",
                "",
                "{source_stem}__{tag}",
            ),
        ),
        compute_request=ComputeRequest(mode="cpu"),
        save_python_script=True,
    )
    result = run_batch(document, config)
    assert result.manifest.items[0].status is BatchStatus.COMPLETED
    preview = inspect_collection(result.manifest.to_dict(), output.id)
    assert preview.items[0].observation_metadata["frame_count"] == 4
    assert preview.items[0].tracking_metadata is not None
    collection = collect_measurements(preview)
    assert collection.table.observation_metadata is None
    path = save_measurement_collection(
        collection, tmp_path / "collected.vipp-results.json"
    )
    restored = load_measurement_collection(path)
    assert (
        restored.items[0].observation_metadata
        == collection.items[0].observation_metadata
    )
    assert restored.items[0].tracking_metadata == collection.items[0].tracking_metadata
    assert restored.table.rows == collection.table.rows


def test_metadata_only_planning_does_not_run_detection_or_linking(monkeypatch):
    from napari_vipp.core import time_detection, tracking

    def forbidden(*args, **kwargs):
        raise AssertionError("Scientific work during metadata planning")

    monkeypatch.setattr(time_detection, "detect_spots_per_frame", forbidden)
    monkeypatch.setattr(tracking, "build_tracks", forbidden)
    pipeline, _, _ = pipeline_for()
    pipeline.preflight_axis_contract({"input": payload()})


def test_intensity_measurement_tracks_preserve_index_centroids_and_grid():
    pipeline, detector, linker = pipeline_for(labels=True)
    intensity = pipeline.add_node("measure_objects_intensity")
    labels_connection = next(c for c in pipeline.connections if c.target_id == detector)
    assert pipeline.connect(
        labels_connection.source_id, intensity.id, target_port=0
    ).success
    assert pipeline.connect("input", intensity.id, target_port=1).success
    assert pipeline.connect(intensity.id, linker).success
    result = execute_pipeline_request(
        request(pipeline, payload(labels=True, volume=True)), raise_errors=True
    ).pipeline
    table = result.outputs[intensity.id]
    assert table.observation_metadata.coordinate_columns == (
        "centroid_z",
        "centroid_y",
        "centroid_x",
    )
    assert result.node_outputs[linker][1].row_count == 2


def test_arbitrary_columns_cannot_inherit_summary_evidence(tmp_path):
    pipeline, _, linker = pipeline_for()
    result = execute_pipeline_request(request(pipeline), raise_errors=True).pipeline
    broken = replace(
        result.node_outputs[linker][1],
        columns=("track_id",),
        rows=((-9,),),
        column_units=(),
    )
    path = save_table_output(broken, tmp_path / "broken.csv", format="csv")
    with pytest.raises(MeasurementCollectionError, match="tracking table"):
        _read_table(
            path.read_bytes(),
            table_measurement_metadata(broken),
            "csv",
            CollectionLimits(),
            None,
        )
