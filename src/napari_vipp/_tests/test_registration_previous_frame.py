"""Original-neighbour estimation, physical composition and explicit series QC."""

import weakref
from dataclasses import replace

import numpy as np
import pytest

from napari_vipp._tests.test_registration import state
from napari_vipp.core import registration
from napari_vipp.core.metadata import AxisMetadata
from napari_vipp.core.progress import OperationCancelled, ProgressContext
from napari_vipp.core.registration import apply_transform, estimate_registration
from napari_vipp.core.transforms import (
    TransformData,
    load_transform,
    save_transform_output,
)


def translation_series(rank=2, count=5, anchor=2):
    shape = (32, 40) if rank == 2 else (16, 24, 28)
    step = np.array((1, -2) if rank == 2 else (1, -1, 2))
    spacing = (0.7, 1.3) if rank == 2 else (0.4, 0.7, 1.1)
    origin = (11.0, -7.0) if rank == 2 else (11.0, -7.0, 5.0)
    base = np.random.default_rng(45).normal(size=shape)
    data = np.stack(
        [
            np.roll(base, tuple((t - anchor) * step), axis=tuple(range(rank)))
            for t in range(count)
        ]
    )
    data.setflags(write=False)
    carried = state(
        data, "TYX" if rank == 2 else "TZYX", spacing=spacing, origin=origin
    )
    expected = np.repeat(np.eye(rank + 1)[None], count, axis=0)
    expected[:, :-1, -1] = np.asarray(
        [(anchor - t) * step * spacing for t in range(count)]
    )
    return data, carried, expected


@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("anchor", [0, 2, 4])
def test_translation_original_neighbours_arbitrary_anchor_physical_units(rank, anchor):
    data, carried, expected = translation_series(rank, anchor=anchor)
    before = data.copy()
    transform, table = estimate_registration(
        data,
        moving_state=carried,
        mode="Time series",
        reference_time=anchor,
        time_strategy="Previous frame",
        precision=1,
        max_shift=0.1,
    )
    np.testing.assert_allclose(transform.matrices, expected, atol=1e-12)
    np.testing.assert_array_equal(data, before)
    assert not data.flags.writeable
    assert transform.time_strategy == "Previous frame"
    assert dict(transform.settings)["time_strategy_schema_version"] == 1
    assert "accumulate" in transform.state.history[-1]
    assert [row["pair_reference_time"] for row in table.records()] == [
        t + (1 if t < anchor else -1) if t != anchor else anchor
        for t in range(len(data))
    ]
    assert all("errors accumulate" in row["review_note"] for row in table.records())
    aligned, valid = apply_transform(data, transform, image_state=carried)
    for t in range(len(data)):
        np.testing.assert_allclose(
            aligned[t][valid[t]], data[anchor][valid[t]], atol=1e-12
        )


def rigid_series(rank):
    """Analytic continuous phantom, sampled independently into each physical frame."""
    shape = (40, 48) if rank == 2 else (20, 24, 28)
    spacing = np.array((0.7, 1.2) if rank == 2 else (0.6, 0.9, 1.3))
    origin = np.array((11.0, -7.0) if rank == 2 else (11.0, -7.0, 5.0))
    extent = (np.asarray(shape) - 1) * spacing
    center = origin + extent / 2
    points = np.indices(shape).reshape(rank, -1) * spacing[:, None] + origin[:, None]
    motions, frames = [], []
    for t in range(5):
        angle = (t - 2) * 0.018
        c, s = np.cos(angle), np.sin(angle)
        if rank == 2:
            rotation = np.array(((c, -s), (s, c)))
        else:
            c2, s2 = np.cos(angle * 0.7), np.sin(angle * 0.7)
            rotation = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1))) @ np.array(
                ((1, 0, 0), (0, c2, -s2), (0, s2, c2))
            )
        matrix = np.eye(rank + 1)
        matrix[:-1, :-1] = rotation
        matrix[:-1, -1] = center - rotation @ center + (t - 2) * spacing * 0.25
        motions.append(matrix)
        transformed = rotation @ points + matrix[:-1, -1, None]
        values = np.zeros(points.shape[1])
        for fractions, amplitude in (
            ((0.25, 0.30, 0.28), 1),
            ((0.62, 0.55, 0.72), 0.7),
            ((0.73, 0.25, 0.45), 0.4),
            ((0.45, 0.75, 0.20), 0.6),
        ):
            landmark = origin + extent * fractions[:rank]
            sigma = extent * (0.075 + amplitude * 0.02)
            values += amplitude * np.exp(
                -np.sum(
                    ((transformed - landmark[:, None]) / sigma[:, None]) ** 2, axis=0
                )
            )
        frames.append(values.reshape(shape))
    data = np.stack(frames)
    data.setflags(write=False)
    carried = state(
        data,
        "TYX" if rank == 2 else "TZYX",
        spacing=tuple(spacing),
        origin=tuple(origin),
    )
    return data, carried, np.asarray(motions)


@pytest.mark.parametrize("rank", [2, 3])
def test_noncommuting_rigid_composition_both_sides_no_registered_estimation(
    rank, monkeypatch
):
    data, carried, expected = rigid_series(rank)
    anchor = 2
    pairs = [(1, 2), (0, 1), (3, 2), (4, 3)]
    seen = []

    def exact_pair(fixed, moving, *args, **kwargs):
        t, neighbour = pairs[len(seen)]
        assert np.shares_memory(fixed, data)
        assert np.shares_memory(moving, data)
        np.testing.assert_array_equal(fixed, data[neighbour])
        np.testing.assert_array_equal(moving, data[t])
        seen.append((t, neighbour))
        return (
            np.linalg.inv(expected[neighbour]) @ expected[t],
            -1.0,
            "analytic pair oracle",
        )

    monkeypatch.setattr(registration, "_sitk_estimate", exact_pair)
    transform, table = estimate_registration(
        data,
        moving_state=carried,
        mode="Time series",
        model="Rigid",
        reference_time=anchor,
        time_strategy="Previous frame",
    )
    assert seen == pairs
    np.testing.assert_allclose(transform.matrices, expected, rtol=0, atol=4e-14)
    # A translated rotation distinguishes the wrong multiplication order.
    local = np.linalg.inv(expected[3]) @ expected[4]
    assert not np.allclose(local @ expected[3], expected[4], atol=1e-7)
    rows = table.records()
    assert rows[4]["pair_reference_time"] == 3
    assert rows[4]["pair_correlation"] > 0.99
    assert rows[4]["cumulative_correlation"] > 0.99
    assert all(unit == "micrometer" for _, unit in table.column_units)


@pytest.mark.parametrize("policy", ["Report only", "Require local limits"])
def test_local_and_cumulative_displacement_limits_are_distinct(policy):
    data, carried, _ = translation_series(anchor=0)
    kwargs = dict(
        moving_state=carried,
        mode="Time series",
        reference_time=0,
        time_strategy="Previous frame",
        precision=1,
        max_shift=0.08,
        cumulative_quality_policy=policy,
    )
    if policy == "Require local limits":
        with pytest.raises(ValueError, match="displacement exceeds"):
            estimate_registration(data, **kwargs)
    else:
        _, table = estimate_registration(data, **kwargs)
        row = table.records()[4]
        assert row["pair_center_displacement_x"] == pytest.approx(2.6)
        assert row["cumulative_center_displacement_x"] == pytest.approx(10.4)
        assert (
            row["pair_valid_overlap_fraction"]
            > row["cumulative_valid_overlap_fraction"]
        )


def test_report_only_allows_no_cumulative_anchor_coverage():
    data, carried, expected = translation_series(count=21, anchor=0)
    transform, table = estimate_registration(
        data,
        moving_state=carried,
        mode="Time series",
        time_strategy="Previous frame",
        precision=1,
    )
    np.testing.assert_allclose(transform.matrices, expected, atol=1e-12)
    assert table.records()[20]["pair_valid_overlap_fraction"] > 0.9
    assert table.records()[20]["cumulative_valid_overlap_fraction"] == 0
    assert table.records()[20]["cumulative_correlation"] is None


def test_cumulative_overlap_gate_is_explicit_and_independent_of_displacement():
    data, carried, _ = translation_series(anchor=0)
    with pytest.raises(ValueError, match="valid overlap"):
        estimate_registration(
            data,
            moving_state=carried,
            mode="Time series",
            reference_time=0,
            time_strategy="Previous frame",
            precision=1,
            max_shift=0.49,
            minimum_overlap=0.8,
            cumulative_quality_policy="Require local limits",
        )


@pytest.mark.parametrize("rank", [2, 3])
def test_rigid_estimator_chain_against_independently_sampled_physical_phantom(rank):
    pytest.importorskip("SimpleITK")
    data, carried, expected = rigid_series(rank)
    transform, table = estimate_registration(
        data,
        moving_state=carried,
        mode="Time series",
        model="Rigid",
        reference_time=2,
        time_strategy="Previous frame",
        iterations=100,
    )
    # Physical landmark error, not only similarity: includes anisotropic spacing,
    # nonzero origin and cumulative error from both directions around the anchor.
    grid = transform.moving_grid
    points = np.array(
        [
            np.asarray(grid.origin)
            + fraction * (np.asarray(grid.shape) - 1) * grid.spacing
            for fraction in (
                np.array((0.2, 0.3, 0.7)[:rank]),
                np.array((0.6, 0.2, 0.4)[:rank]),
                np.array((0.7, 0.8, 0.2)[:rank]),
                np.array((0.3, 0.6, 0.8)[:rank]),
            )
        ]
    ).T
    for actual, oracle in zip(transform.matrices, expected, strict=True):
        actual = np.asarray(actual)
        error = (actual[:-1, :-1] - oracle[:-1, :-1]) @ points + (
            actual[:-1, -1] - oracle[:-1, -1]
        )[:, None]
        assert np.max(np.linalg.norm(error, axis=0)) < 0.25
    assert min(row["pair_correlation"] for row in table.records()) > 0.99


@pytest.mark.parametrize(
    "failure", ["constant", "local displacement", "weak structure"]
)
def test_failed_local_pair_never_skipped_or_replaced_by_identity(failure, monkeypatch):
    data, carried, _ = translation_series()
    data = data.copy()
    if failure == "constant":
        data[0] = 1
        match = "constant image"
    elif failure == "local displacement":
        data[0] = np.roll(data[1], 12, axis=1)
        match = "No translation satisfies"
    else:

        def weak(*args, **kwargs):
            return np.eye(3), 0.1, "bad local match"

        monkeypatch.setattr(registration, "_translation", weak)
        match = "weak shared structure"
    data.setflags(write=False)
    with pytest.raises(ValueError, match=match):
        estimate_registration(
            data,
            moving_state=carried,
            mode="Time series",
            time_strategy="Previous frame",
            reference_time=2,
        )


def test_cancellation_between_pairs_publishes_no_transform(monkeypatch):
    data, carried, _ = translation_series()
    cancelled = False
    original = registration._translation
    calls = 0

    def estimate(*args, **kwargs):
        nonlocal cancelled, calls
        calls += 1
        result = original(*args, **kwargs)
        cancelled = True
        return result

    monkeypatch.setattr(registration, "_translation", estimate)
    monkeypatch.setattr(
        registration,
        "TransformData",
        lambda *a, **k: pytest.fail("partial transform published"),
    )
    with pytest.raises(OperationCancelled):
        estimate_registration(
            data,
            moving_state=carried,
            mode="Time series",
            time_strategy="Previous frame",
            reference_time=2,
            progress_context=ProgressContext(cancelled=lambda: cancelled),
        )
    assert calls == 1


def test_diagnostics_temporary_volumes_not_retained_across_pairs(monkeypatch):
    data, carried, _ = translation_series(count=9, anchor=4)
    original = registration._resample_volume
    references = []

    def bounded(*args, **kwargs):
        assert all(ref() is None for ref in references)
        result = original(*args, **kwargs)
        references.extend(weakref.ref(value) for value in result)
        return result

    monkeypatch.setattr(registration, "_resample_volume", bounded)
    transform, _ = estimate_registration(
        data,
        moving_state=carried,
        mode="Time series",
        reference_time=4,
        time_strategy="Previous frame",
    )
    assert all(ref() is None for ref in references)
    assert transform.nbytes == 9 * 3 * 3 * 8


def test_noncanonical_multichannel_whole_volume_application_once(monkeypatch):
    data, carried, expected = translation_series(rank=3)
    channels = np.stack((data, data * 3), axis=1)
    # C, Z, T, X, Y: neither time/channel nor spatial suffix can be assumed.
    channels = channels.transpose(1, 2, 0, 4, 3)
    channels.setflags(write=False)
    axes = (
        AxisMetadata("c", "channel"),
        carried.axes[1],
        carried.axes[0],
        carried.axes[3],
        carried.axes[2],
    )
    carried = replace(carried, shape=channels.shape, axes=axes)
    transform, _ = estimate_registration(
        channels,
        moving_state=carried,
        mode="Time series",
        reference_time=2,
        channel=1,
        time_strategy="Previous frame",
    )
    np.testing.assert_allclose(transform.matrices, expected, atol=1e-12)
    original = registration._resample_volume
    calls = []

    def once(volume, *args, **kwargs):
        assert np.shares_memory(volume, channels)
        calls.append(volume.shape)
        return original(volume, *args, **kwargs)

    monkeypatch.setattr(registration, "_resample_volume", once)
    aligned, valid = apply_transform(channels, transform, image_state=carried)
    assert calls == [(16, 24, 28)] * 10
    assert aligned.shape == valid.shape == channels.shape
    result = aligned.transpose(2, 0, 1, 4, 3)
    coverage = valid.transpose(2, 0, 1, 4, 3)
    for t in range(5):
        for channel in range(2):
            np.testing.assert_allclose(
                result[t, channel][coverage[t, channel]],
                data[2][coverage[t, channel]] * (1 if channel == 0 else 3),
                atol=1e-12,
            )


def test_legacy_transform_and_explicit_fixed_defaults_unchanged(tmp_path):
    data, carried, _ = translation_series()
    old, old_table = estimate_registration(
        data, moving_state=carried, mode="Time series"
    )
    explicit, explicit_table = estimate_registration(
        data,
        moving_state=carried,
        mode="Time series",
        time_strategy="Fixed reference",
        cumulative_quality_policy="Report only",
    )
    assert old.to_dict() == explicit.to_dict()
    assert old_table == explicit_table
    assert "time_strategy" not in dict(old.settings)
    path = tmp_path / "legacy.json"
    save_transform_output(old, path)
    restored = load_transform(path)
    assert restored.time_strategy == "Fixed reference"
    np.testing.assert_array_equal(
        apply_transform(data, restored, image_state=carried)[0],
        apply_transform(data, old, image_state=carried)[0],
    )


def test_previous_transform_json_workflow_and_generated_python_roundtrip(tmp_path):
    from napari_vipp._tests.test_registration_integration import _payload, _pipeline
    from napari_vipp.core.export import export_pipeline_to_python
    from napari_vipp.core.pipeline import PrototypePipeline
    from napari_vipp.core.workflow import deserialize_workflow, serialize_workflow

    pipeline, estimator, apply_id, _ = _pipeline()
    pipeline.set_param(estimator, "time_strategy", "Previous frame")
    pipeline.set_param(estimator, "reference_time", 1)
    saved = serialize_workflow(pipeline)
    document = deserialize_workflow(saved)
    restored = PrototypePipeline()
    restored.restore_graph(
        document["nodes"], document["connections"], document["output_tunnels"]
    )
    assert restored.nodes[estimator].params["time_strategy"] == "Previous frame"
    namespace = {"__name__": "previous_frame_export"}
    exec(
        compile(export_pipeline_to_python(restored), "<previous-frame export>", "exec"),
        namespace,
    )
    results = namespace["run_pipeline"](source_payloads={"input": _payload()})
    transform = results[estimator]
    assert transform.reference_time == 1 and transform.time_strategy == "Previous frame"
    assert results[apply_id].shape == _payload().data.shape
    path = tmp_path / "previous-frame.json"
    save_transform_output(transform, path)
    assert load_transform(path).to_dict() == transform.to_dict()


def test_legacy_workflow_keeps_exact_scientific_hash_and_missing_defaults():
    from napari_vipp._tests.test_registration_integration import _pipeline
    from napari_vipp.core.batch import scientific_workflow_hash
    from napari_vipp.core.pipeline import PrototypePipeline
    from napari_vipp.core.workflow import deserialize_workflow, serialize_workflow

    pipeline, estimator, _, _ = _pipeline()
    pipeline.nodes[estimator].params.pop("time_strategy")
    pipeline.nodes[estimator].params.pop("cumulative_quality_policy")
    saved = serialize_workflow(pipeline)
    digest = scientific_workflow_hash(saved)
    document = deserialize_workflow(saved)
    clone = PrototypePipeline()
    clone.restore_graph(
        document["nodes"], document["connections"], document["output_tunnels"]
    )
    assert "time_strategy" not in clone.nodes[estimator].params
    assert "cumulative_quality_policy" not in clone.nodes[estimator].params
    assert scientific_workflow_hash(serialize_workflow(clone)) == digest
    clone.set_param(estimator, "time_strategy", "Previous frame")
    assert scientific_workflow_hash(serialize_workflow(clone)) != digest


def test_previous_frame_batch_publishes_cumulative_transform_and_policy(tmp_path):
    from napari_vipp._tests.test_registration_integration import _batch
    from napari_vipp.core.batch import BatchStatus, run_batch, scientific_workflow_hash

    workflow, config, estimator = _batch(tmp_path)
    node = next(node for node in workflow["nodes"] if node["id"] == estimator)
    node["params"].update(
        time_strategy="Previous frame",
        reference_time=1,
        cumulative_quality_policy="Require local limits",
    )
    config = replace(config, workflow_sha256=scientific_workflow_hash(workflow))
    result = run_batch(workflow, config)
    assert result.manifest.items[0].status is BatchStatus.COMPLETED
    assert len(result.saved_paths) == 1
    transform = load_transform(result.saved_paths[0])
    assert transform.time_strategy == "Previous frame" and transform.reference_time == 1
    assert (
        dict(transform.settings)["cumulative_quality_policy"] == "Require local limits"
    )
    np.testing.assert_allclose(np.asarray(transform.matrices)[1], np.eye(3), atol=1e-12)


@pytest.mark.parametrize(
    "params,match",
    [
        ({"time_strategy": "Guess"}, "time strategy"),
        ({"cumulative_quality_policy": "Skip failures"}, "cumulative quality"),
    ],
)
def test_strategy_parameters_rejected_consistently_in_planning_and_execution(
    params, match
):
    from napari_vipp._tests.test_registration_planning import image, project

    data, carried, _ = translation_series()
    with pytest.raises(ValueError, match=match):
        estimate_registration(data, moving_state=carried, mode="Time series", **params)
    with pytest.raises(ValueError, match=match):
        project(
            "estimate_registration",
            [image(data.shape, "tyx")],
            mode="Time series",
            **params,
        )


def test_previous_frame_metadata_planning_never_reads_pixels():
    from napari_vipp._tests.test_registration_planning import image, project

    outputs = project(
        "estimate_registration",
        [image((5, 20, 24, 28), "tzyx")],
        mode="Time series",
        time_strategy="Previous frame",
        reference_time=2,
    )
    plan = outputs[0][0]
    assert plan.transform_count == 5 and plan.reference_time == 2


@pytest.mark.parametrize(
    "key,value",
    [
        ("time_strategy", "Unreviewed strategy"),
        ("time_strategy_schema_version", 2),
        ("time_strategy_schema_version", True),
        ("cumulative_quality_policy", "Skip failed frames"),
    ],
)
def test_previous_strategy_metadata_is_validated_on_import(key, value):
    data, carried, _ = translation_series()
    transform, _ = estimate_registration(
        data,
        moving_state=carried,
        mode="Time series",
        time_strategy="Previous frame",
    )
    document = transform.to_dict()
    document["settings"][key] = value
    with pytest.raises(ValueError, match="strategy|Previous-frame"):
        TransformData.from_dict(document)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_nonfinite_previous_frame_inputs_rejected_before_estimation(
    bad_value, monkeypatch
):
    data, carried, _ = translation_series()
    data = data.copy()
    data[0, 0, 0] = bad_value
    data.setflags(write=False)
    monkeypatch.setattr(
        registration, "_translation", lambda *a, **k: pytest.fail("estimator called")
    )
    with pytest.raises(ValueError, match="finite input values"):
        estimate_registration(
            data,
            moving_state=carried,
            mode="Time series",
            time_strategy="Previous frame",
        )


def test_previous_frame_boolean_estimation_and_wide_integer_rejection():
    data, _, expected = translation_series()
    boolean = data > 0
    boolean.setflags(write=False)
    carried = state(boolean, "TYX", spacing=(0.7, 1.3), origin=(11, -7))
    transform, _ = estimate_registration(
        boolean,
        moving_state=carried,
        mode="Time series",
        time_strategy="Previous frame",
        reference_time=2,
    )
    np.testing.assert_allclose(transform.matrices, expected, atol=1e-12)
    wide = boolean.astype(np.uint64) + np.uint64(2**63)
    wide.setflags(write=False)
    carried = state(wide, "TYX")
    with pytest.raises(ValueError, match="wide integer intensities"):
        estimate_registration(
            wide,
            moving_state=carried,
            mode="Time series",
            time_strategy="Previous frame",
        )
