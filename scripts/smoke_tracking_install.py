"""Verify packaged tracking and previous-frame registration against known motion."""

from __future__ import annotations

import argparse
import json
import math
import platform
from importlib.metadata import distribution, version
from importlib.resources import files
from pathlib import Path


def previous_frame_known_motion():
    """Estimate both sides of a nonzero anchor using an analytical intensity field.

    Each frame is sampled independently from the same asymmetric Gaussian
    field at authored integer-shifted coordinates. No estimator or resampler
    constructs the source data or the expected moving-to-anchor matrices.
    """
    import numpy as np

    from napari_vipp.core.metadata import (
        AxisMetadata,
        SourceMetadata,
        image_state_from_array,
    )
    from napari_vipp.core.registration import apply_transform, estimate_registration

    count, anchor = 5, 2
    shape = (40, 52)
    step = np.asarray((2, -1))
    spacing = np.asarray((0.6, 1.4))
    origin = (10.0, -4.0)
    y, x = np.indices(shape, dtype=np.float64)

    def field(sample_y, sample_x):
        return (
            0.9 * np.exp(-((sample_y - 13) ** 2 / 12 + (sample_x - 17) ** 2 / 18))
            + 0.6 * np.exp(-((sample_y - 26) ** 2 / 22 + (sample_x - 35) ** 2 / 9))
            + 0.35 * np.exp(-((sample_y - 17) ** 2 / 5 + (sample_x - 39) ** 2 / 13))
        )

    data = np.stack(
        [
            field(y - (frame - anchor) * step[0], x - (frame - anchor) * step[1])
            for frame in range(count)
        ]
    )
    expected = np.repeat(np.eye(3)[None], count, axis=0)
    expected[:, :2, 2] = [(anchor - frame) * step * spacing for frame in range(count)]
    original = data.copy()
    data.setflags(write=False)
    state = image_state_from_array(
        data,
        axes=(
            AxisMetadata("t", "time", "second", 0.75, 3.0),
            AxisMetadata("y", "space", "micrometer", spacing[0], origin[0]),
            AxisMetadata("x", "space", "micrometer", spacing[1], origin[1]),
        ),
        source_name="Independent previous-frame analytical translation phantom",
        source=SourceMetadata(source_uuid="vipp-installed-previous-frame-oracle-v1"),
        defer_statistics=True,
    )
    transform, diagnostics = estimate_registration(
        data,
        moving_state=state,
        mode="Time series",
        model="Translation",
        reference_time=anchor,
        time_strategy="Previous frame",
        cumulative_quality_policy="Report only",
        precision=1,
        max_shift=0.15,
        minimum_overlap=0.6,
    )
    matrices = np.asarray(transform.matrices)
    np.testing.assert_allclose(matrices, expected, rtol=0, atol=1e-12)
    assert transform.reference_time == anchor
    assert transform.time_strategy == "Previous frame"
    expected_neighbours = [1, 2, 2, 2, 3]
    assert [row["pair_reference_time"] for row in diagnostics.records()] == (
        expected_neighbours
    )
    assert all(
        "errors accumulate" in row["review_note"] for row in diagnostics.records()
    )
    aligned, valid = apply_transform(
        data, transform, image_state=state, interpolation="Linear", outside_value=0
    )
    assert aligned.shape == valid.shape == data.shape
    assert valid.dtype == np.dtype(bool)
    alignment_errors = []
    for frame in range(count):
        assert valid[frame].any()
        np.testing.assert_allclose(
            aligned[frame][valid[frame]],
            original[anchor][valid[frame]],
            rtol=0,
            atol=1e-12,
        )
        alignment_errors.append(
            float(
                np.max(
                    np.abs(
                        aligned[frame][valid[frame]] - original[anchor][valid[frame]]
                    )
                )
            )
        )
    np.testing.assert_array_equal(data, original)
    assert not data.flags.writeable
    return {
        "case": "previous-frame-analytical-translation",
        "source_shape": list(data.shape),
        "anchor": anchor,
        "frames_before_anchor": anchor,
        "frames_after_anchor": count - anchor - 1,
        "spatial_scale": spacing.tolist(),
        "spatial_origin": list(origin),
        "time_scale": 0.75,
        "time_origin": 3.0,
        "expected_matrices": expected.tolist(),
        "actual_matrices": matrices.tolist(),
        "pair_reference_times": expected_neighbours,
        "maximum_matrix_absolute_error": float(np.max(np.abs(matrices - expected))),
        "maximum_valid_alignment_absolute_error": max(alignment_errors),
        "valid_count_by_frame": [int(np.count_nonzero(mask)) for mask in valid],
        "source_unchanged": True,
        "tolerance": 1e-12,
    }


def smoke_tracking_install(*, expected_package_root=None, require_installed=False):
    import numpy as np

    import napari_vipp
    from napari_vipp._sample_data import make_tracking_sample_data
    from napari_vipp.core.compute import ComputeRequest
    from napari_vipp.core.execution import PipelineRunRequest, execute_pipeline_request
    from napari_vipp.core.pipeline import PrototypePipeline, SourcePayload
    from napari_vipp.core.workflow import load_workflow, serialize_workflow

    imported = Path(napari_vipp.__file__).resolve()
    if expected_package_root is not None:
        root = Path(expected_package_root).resolve()
        if root not in imported.parents:
            raise RuntimeError(f"Wrong installed package: {imported}, expected {root}")
    distribution_root = Path(distribution("napari-vipp").locate_file("")).resolve()
    if require_installed and distribution_root not in imported.parents:
        raise RuntimeError(
            f"Imported code {imported} does not belong to installed distribution "
            f"{distribution_root}"
        )

    cases = []
    for case_index, (kind, sample) in enumerate(
        zip(("spots-2d", "labels-3d"), make_tracking_sample_data(), strict=True),
        start=1,
    ):
        resource = files("napari_vipp").joinpath(
            "examples", f"synthetic-tracking-{kind}.json"
        )
        if not resource.is_file():
            raise RuntimeError(f"Missing packaged example: {resource}")
        workflow = load_workflow(Path(str(resource)))
        pipeline = PrototypePipeline()
        pipeline.restore_graph(
            workflow["nodes"], workflow["connections"], workflow["output_tunnels"]
        )
        data, kwargs, _ = sample
        original = data.copy()
        data.setflags(write=False)
        truth = kwargs["metadata"]["tracking_ground_truth"]
        run = execute_pipeline_request(
            PipelineRunRequest(
                run_id=case_index,
                workflow=serialize_workflow(pipeline),
                input_data=None,
                input_metadata=None,
                input_name="",
                source_payloads={
                    "input": SourcePayload(data, kwargs["metadata"], kwargs["name"])
                },
                compute_request=ComputeRequest(mode="cpu"),
                manual_node_ids=frozenset(pipeline.manual_node_ids()),
            ),
            raise_errors=True,
        )
        if run.error or run.cancelled or run.pipeline is None:
            raise AssertionError(f"{kind} did not complete: {run.error}")
        result = run.pipeline
        observations, summary = result.node_outputs["tracks"]
        evidence = observations.observation_metadata
        records = observations.records()
        coordinates = evidence.coordinate_columns
        found = {
            (row["t_index"], *(row[column] for column in coordinates)): row
            for row in records
        }
        expected = {
            (row["t_index"], *row["center"]): row for row in truth["observations"]
        }
        assert set(found) == set(expected), f"{kind}: incorrect observation centers"
        assert observations.row_count == sum(truth["frame_counts"])
        assert summary.row_count == truth["expected_track_count"]
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
        assert len({(row["t_index"], row["track_id"]) for row in records}) == len(
            records
        )
        for key, known in expected.items():
            row = found[key]
            if known["unambiguous_track_id"] is not None:
                assert row["track_id"] == known["unambiguous_track_id"]
                assert not row["review_flag"]
            if kind == "labels-3d":
                assert row["label_id"] == known["label_id"]
        assert result.outputs["summary"].rows == summary.rows
        assert result.outputs["summary"].tracking_metadata == summary.tracking_metadata
        summaries = {row["track_id"]: row for row in summary.records()}
        if kind == "spots-2d":
            assert not any(row["t_index"] == 3 for row in records)
            gap_rows = [row for row in records if row["t_index"] == 4]
            assert len(gap_rows) == 4 and all(
                row["gap_frames"] == 1 for row in gap_rows
            )
            reviewed = [row for row in records if row["review_flag"]]
            assert reviewed
            assert all(row["y_index"] in (48, 52) for row in reviewed)
            assert sum(row["review_flag"] for row in summaries.values()) == 2
            assert all(row["observation_count"] == 6 for row in summaries.values())
            assert all(row["missing_frame_count"] == 1 for row in summaries.values())
            for row in records:
                if row["previous_frame"] is not None and row["track_id"] in (1, 2):
                    expected_speed = 6.0 if row["track_id"] == 1 else 4.0
                    np.testing.assert_allclose(row["speed"], expected_speed, rtol=1e-12)
        else:
            step = math.sqrt(1.5**2 + 0.4**2)
            assert [summaries[i]["observation_count"] for i in (1, 2, 3)] == [6, 5, 3]
            np.testing.assert_allclose(
                summaries[1]["path_length"], step * 5, rtol=1e-12
            )
            assert summaries[1]["duration"] == 12.5
            np.testing.assert_allclose(summaries[2]["path_length"], 2.0, rtol=1e-12)
            assert summaries[2]["gap_count"] == summaries[2]["missing_frame_count"] == 1
            assert summaries[3]["first_frame"] == 3
            assert summaries[3]["path_length"] == 0
            assert not any(row["review_flag"] for row in records)
            for row in records:
                if row["previous_frame"] is not None:
                    speed = {1: step / 2.5, 2: 0.4 / 2.5, 3: 0.0}[row["track_id"]]
                    np.testing.assert_allclose(row["speed"], speed, rtol=1e-12, atol=0)
        np.testing.assert_array_equal(data, original)
        assert not data.flags.writeable
        cases.append(
            {
                "case": kind,
                "input_shape": list(data.shape),
                "expected_and_found_observation_count": observations.row_count,
                "expected_and_found_track_count": summary.row_count,
                "frame_counts": truth["frame_counts"],
                "missed_observations": len(set(expected) - set(found)),
                "extra_observations": len(set(found) - set(expected)),
                "coordinate_index_error": 0,
                "review_observation_count": sum(row["review_flag"] for row in records),
                "source_unchanged": True,
                "summary": summary.records(),
                "observation_evidence": evidence.to_dict(),
                "tracking_evidence": observations.tracking_metadata.to_dict(),
            }
        )
    return {
        "status": "passed",
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "imported_package": str(imported),
        "distribution_metadata_root": str(distribution_root),
        "installed_distribution_verified": require_installed,
        "versions": {
            name: version(name)
            for name in ("napari-vipp", "numpy", "scipy", "scikit-image")
        },
        "cases": cases,
        "previous_frame_registration": previous_frame_known_motion(),
        "limits": [
            "Synthetic position and motion truth, not acquired-image tracking accuracy",
            "Crossing review does not claim biological identity recovery",
            "CPU-only native operating system shown above",
            "No split/merge lineage, interpolation or drift correction",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-package-root")
    parser.add_argument("--require-installed", action="store_true")
    parser.add_argument("--output-json", type=Path)
    options = parser.parse_args()
    evidence = smoke_tracking_install(
        expected_package_root=options.expected_package_root,
        require_installed=options.require_installed,
    )
    text = json.dumps(evidence, indent=2, allow_nan=False)
    if options.output_json is not None:
        options.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
