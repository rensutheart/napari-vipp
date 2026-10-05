"""Synthetic trajectory truth is authored independently of detection/linking."""

from pathlib import Path

import numpy as np
import pytest

from napari_vipp._sample_data import make_tracking_sample_data
from napari_vipp.core.compute import ComputeRequest
from napari_vipp.core.execution import PipelineRunRequest, execute_pipeline_request
from napari_vipp.core.export import export_pipeline_to_python
from napari_vipp.core.metadata import image_state_from_array
from napari_vipp.core.observation_series import attach_object_observations
from napari_vipp.core.operations import measure_objects
from napari_vipp.core.pipeline import PrototypePipeline, SourcePayload
from napari_vipp.core.tracking import build_tracks
from napari_vipp.core.workflow import load_workflow, serialize_workflow
from napari_vipp.ui.examples import _example_workflow_by_id, _example_workflow_path

ROOT = Path(__file__).resolve().parents[3]


def run_tracking_example(kind):
    """Execute a packaged graph and return independently authored source truth."""
    spec = _example_workflow_by_id(f"tracking-{kind}")
    document = load_workflow(_example_workflow_path(spec))
    pipeline = PrototypePipeline()
    pipeline.restore_graph(
        document["nodes"], document["connections"], document["output_tunnels"]
    )
    sample_index = 0 if kind == "spots-2d" else 1
    data, kwargs, _ = make_tracking_sample_data()[sample_index]
    before = data.copy()
    data.setflags(write=False)
    payload = SourcePayload(data, kwargs["metadata"], kwargs["name"])
    result = execute_pipeline_request(
        PipelineRunRequest(
            run_id=1,
            workflow=serialize_workflow(pipeline),
            input_data=data,
            input_metadata=payload.metadata,
            input_name=payload.name,
            source_payloads={"input": payload},
            compute_request=ComputeRequest(mode="cpu"),
            manual_node_ids=frozenset(pipeline.manual_node_ids()),
        ),
        raise_errors=True,
    )
    assert not result.error and not result.cancelled
    np.testing.assert_array_equal(data, before)
    assert not data.flags.writeable
    return (
        result.pipeline,
        document,
        {"input": payload},
        kwargs["metadata"]["tracking_ground_truth"],
    )


def test_tracking_samples_have_repeatable_independent_truth():
    first = make_tracking_sample_data()
    repeated = make_tracking_sample_data()
    changed = make_tracking_sample_data(seed=20260930)
    for sample, duplicate in zip(first, repeated, strict=True):
        np.testing.assert_array_equal(sample[0], duplicate[0])
        assert (
            sample[1]["metadata"]["tracking_ground_truth"]
            == duplicate[1]["metadata"]["tracking_ground_truth"]
        )
    assert not np.array_equal(first[0][0], changed[0][0])
    assert (
        first[0][1]["metadata"]["tracking_ground_truth"]["observations"]
        == changed[0][1]["metadata"]["tracking_ground_truth"]["observations"]
    )
    assert first[0][2] == "image" and first[1][2] == "labels"
    assert first[0][0].shape == (7, 2, 72, 96)
    assert first[1][0].shape == (6, 16, 32, 40)
    assert first[0][1]["metadata"]["vipp_axis_order"] == "TCYX"
    assert first[1][1]["metadata"]["vipp_axis_order"] == "TZYX"
    assert np.max(first[0][0][3, 1]) < 0.1
    assert np.max(first[0][0][:, 0]) < 0.1
    assert first[1][1]["metadata"]["vipp_image_state"]["kind"] == "label image"


@pytest.mark.parametrize("kind", ["spots-2d", "labels-3d"])
def test_examples_recover_exact_centers_populations_and_unambiguous_track_ids(
    kind, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    pipeline, document, payloads, truth = run_tracking_example(kind)
    observations, summary = pipeline.node_outputs["tracks"]
    coordinate_names = observations.observation_metadata.coordinate_columns
    records = observations.records()
    observed = {
        (record["t_index"], *(record[name] for name in coordinate_names)): record
        for record in records
    }
    expected = {
        (item["t_index"], *item["center"]): item for item in truth["observations"]
    }
    assert set(observed) == set(expected)
    assert observations.row_count == sum(truth["frame_counts"])
    assert summary.row_count == truth["expected_track_count"]
    evidence = observations.observation_metadata
    assert [item.retained_count for item in evidence.frame_populations] == truth[
        "frame_counts"
    ]
    assert all(
        item.eligible_count == item.retained_count and not item.truncated
        for item in evidence.frame_populations
    )
    assert evidence.source_scale == tuple(truth["spacing"])
    assert evidence.source_origin == tuple(truth["origin"])
    assert evidence.time_scale == truth["time_scale"]
    assert evidence.time_origin == truth["time_origin"]
    assert evidence.time_unit == "second"
    assert evidence.frame_count == truth["frame_count"]
    for key, item in expected.items():
        record = observed[key]
        if item["unambiguous_track_id"] is not None:
            assert record["track_id"] == item["unambiguous_track_id"]
            assert record["review_flag"] is False
        if kind == "labels-3d":
            assert record["label_id"] == item["label_id"]
    assert len({(row["t_index"], row["track_id"]) for row in records}) == len(records)
    assert set(document["positions"]) == set(pipeline.nodes)
    assert len(document["notes"]) >= 4
    assert not any(
        node.operation_id == "save_output" for node in pipeline.nodes.values()
    )
    assert {path.name for path in tmp_path.iterdir()} <= {".napari-vipp-test-state"}
    spec = _example_workflow_by_id(f"tracking-{kind}")
    assert (ROOT / "examples" / spec.filename).read_bytes() == _example_workflow_path(
        spec
    ).read_bytes()
    assert payloads["input"].data.flags.writeable is False


def test_spot_crossing_review_never_claims_biological_identity():
    pipeline, _, _, truth = run_tracking_example("spots-2d")
    observations, summary = pipeline.node_outputs["tracks"]
    records = observations.records()
    assert not any(record["t_index"] == 3 for record in records)
    assert sum(record["gap_frames"] == 1 for record in records) == 4
    assert all(
        record["gap_frames"] == 1 for record in records if record["t_index"] == 4
    )
    crossing = {
        (item["t_index"], *item["center"])
        for item in truth["observations"]
        if item["construction_identity"] in truth["review_construction_identities"]
    }
    reviewed = [record for record in records if record["review_flag"]]
    assert reviewed
    assert all(
        (record["t_index"], record["y_index"], record["x_index"]) in crossing
        for record in reviewed
    )
    assert all(record["observation_count"] == 6 for record in summary.records())
    assert all(record["missing_frame_count"] == 1 for record in summary.records())
    assert sum(record["review_flag"] for record in summary.records()) == 2


def test_anisotropic_object_paths_gap_and_speed_have_analytical_values():
    pipeline, _, _, _ = run_tracking_example("labels-3d")
    observations, summary = pipeline.node_outputs["tracks"]
    summaries = {record["track_id"]: record for record in summary.records()}
    assert [summaries[track]["observation_count"] for track in (1, 2, 3)] == [6, 5, 3]
    expected_step = (1.5**2 + 0.4**2) ** 0.5
    assert summaries[1]["path_length"] == pytest.approx(5 * expected_step)
    assert summaries[1]["duration"] == 12.5
    assert summaries[2]["path_length"] == pytest.approx(2.0)
    assert summaries[2]["gap_count"] == 1
    assert summaries[2]["missing_frame_count"] == 1
    assert summaries[3]["first_frame"] == 3
    assert summaries[3]["path_length"] == 0
    assert summary.unit_for("path_length") == "micrometers"
    assert summary.unit_for("duration") == "seconds"
    for record in observations.records():
        if record["previous_frame"] is None:
            continue
        expected_speed = {1: expected_step / 2.5, 2: 0.4 / 2.5, 3: 0}[
            record["track_id"]
        ]
        assert record["speed"] == pytest.approx(expected_speed)
    # At T=3 the new first-voxel object changes the measured local label ID,
    # while the isolated moving object's persistent track remains one.
    track_one = [record for record in observations.records() if record["track_id"] == 1]
    assert [record["label_id"] for record in track_one] == [1, 1, 1, 2, 2, 2]


def test_direct_supplied_label_measurements_preserve_changing_ids():
    labels, kwargs, _ = make_tracking_sample_data()[1]
    source_state = image_state_from_array(labels, layer_metadata=kwargs["metadata"])
    table = measure_objects(
        labels,
        spatial_mode="3D ZYX",
        axis_names=tuple(axis.name for axis in source_state.axes),
        axis_types=tuple(axis.type for axis in source_state.axes),
        axis_scales=tuple(axis.scale for axis in source_state.axes),
        axis_units=tuple(axis.unit for axis in source_state.axes),
    )
    table = attach_object_observations(table, labels, source_state)
    tracks, summary = build_tracks(
        table,
        maximum_displacement=2,
        distance_units="Physical (micrometers)",
        maximum_gap=1,
    )
    truth = kwargs["metadata"]["tracking_ground_truth"]["observations"]
    expected = {
        (item["t_index"], item["source_label_id"]): item["unambiguous_track_id"]
        for item in truth
    }
    assert {
        (row["t_index"], row["label_id"]): row["track_id"] for row in tracks.records()
    } == expected
    assert summary.row_count == 3


@pytest.mark.parametrize("kind", ["spots-2d", "labels-3d"])
def test_tracking_examples_generated_python_preserves_both_outputs(
    kind, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    pipeline, _, payloads, _ = run_tracking_example(kind)
    namespace = {"__name__": "tracking_example_export"}
    exec(
        compile(export_pipeline_to_python(pipeline), "<tracking example>", "exec"),
        namespace,
    )
    outputs = namespace["run_pipeline"](
        source_payloads=payloads, compute_request=ComputeRequest(mode="cpu")
    )
    assert outputs["tracks"].rows == pipeline.outputs["tracks"].rows
    # The all-column pass-through exposes output port 1 as a primary node
    # result under the existing generated-Python output contract.
    assert outputs["summary"].rows == pipeline.node_outputs["tracks"][1].rows
    assert (
        outputs["summary"].tracking_metadata
        == pipeline.node_outputs["tracks"][1].tracking_metadata
    )
