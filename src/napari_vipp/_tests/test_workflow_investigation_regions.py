"""Synthetic exact diagnostic populations; no original-image recipe or data."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from napari_vipp._tests.test_workflow_investigation import (
    act,
    session,
    source_pipeline,
    trial,
)
from napari_vipp.core.workflow_assessment import _capture_binding
from napari_vipp.core.workflow_investigation import (
    InvestigationError,
    InvestigationLimits,
    InvestigationStopped,
)
from napari_vipp.core.workflow_proposals import MAX_REQUEST_BYTES


def test_batch_discovery_is_exact_ordered_bounded_and_not_a_trial():
    engine = session(limits=InvestigationLimits(max_attempts=1))
    ids = ["binary_threshold", "measure_objects", "label_connected_components"]
    result = act(engine, "describe_nodes", operation_ids=ids).public_result()
    assert result["status"] == "ok"
    assert [row["operation_id"] for row in result["result"]] == ids
    assert result["result"][0]["parameters"]
    assert engine.candidates == ()
    assert engine.actions_used == 1
    assert engine.public_request()["budgets"]["attempts_remaining"] == 1
    found = act(engine, "find_nodes", query="connected label").public_result()
    assert "label_connected_components" in {
        row["operation_id"] for row in found["result"]
    }
    assert (
        act(engine, "find_nodes", query="absent keyword").public_result()["result"]
        == []
    )
    all_rows = act(engine, "find_nodes", query="*").public_result()["result"]
    assert len(all_rows) <= 256
    assert all(len(json.dumps(row)) < 500 for row in all_rows)


@pytest.mark.parametrize(
    "ids",
    [
        [],
        ["binary_threshold"] * 2,
        ["binary_threshold"] * 9,
        ["binary_threshold", "absent"],
        [True],
        [{}],
    ],
)
def test_batch_discovery_rejects_invalid_ids_without_partial_contracts(ids):
    engine = session()
    result = act(engine, "describe_nodes", operation_ids=ids).public_result()
    assert result["status"] == "error"
    assert "result" not in result
    assert not engine.candidates


@pytest.mark.parametrize(
    "shape,bounds",
    [((4, 8, 10), [[1, 3], [2, 6], [3, 8]]), ((8, 10), [[2, 6], [3, 8]])],
)
def test_roi_exact_values_native_bounds_calibration_and_source_unchanged(shape, bounds):
    pipeline = source_pipeline(shape)
    original = pipeline.outputs["input"].copy()
    old_state = pipeline.output_states["input"]
    axes = tuple(
        replace(axis, translation=10.0 + i) for i, axis in enumerate(old_state.axes)
    )
    pipeline.output_states["input"] = replace(old_state, axes=axes)
    engine = session(pipeline)
    source_state = engine._source.state
    region = original[tuple(slice(*pair) for pair in bounds)]
    roi = {"axis_order": source_state.axis_order, "bounds": bounds}
    result = act(
        engine, "inspect_source", roi=roi, percentiles=[0, 50, 100], histogram=True
    ).public_result()
    assert result["status"] == "ok", result
    info = result["result"]["diagnostics"]
    assert info["element_count"] == region.size
    assert info["minimum"] == region.min()
    assert info["maximum"] == region.max()
    assert info["percentiles"][1]["value"] == np.median(region)
    assert sum(info["histogram"]["counts"]) == region.size
    population = info["population"]
    assert population["bounds"] == bounds
    assert population["parent_grid"]["shape"] == list(shape)
    assert population["region_grid"]["shape"] == list(region.shape)
    for i, coordinate in enumerate(population["sample_coordinates"]):
        axis = axes[i]
        assert (
            coordinate["first_sample"] == axis.translation + bounds[i][0] * axis.scale
        )
        assert (
            coordinate["last_sample"]
            == axis.translation + (bounds[i][1] - 1) * axis.scale
        )
        assert (
            coordinate["stop_exclusive_coordinate"]
            == axis.translation + bounds[i][1] * axis.scale
        )
        assert coordinate["unit"] == "um"
    assert "not a scientific output crop" in population["meaning"]
    assert "ROI" in info["sampling"]
    assert not engine.preview_images()
    np.testing.assert_array_equal(pipeline.outputs["input"], original)
    assert not pipeline.outputs["input"].flags.writeable
    assert pipeline.output_states["input"].shape == shape
    assert engine._source.state is source_state


@pytest.mark.parametrize(
    "roi",
    [
        {},
        {"axis_order": "YX", "bounds": [[0, 4], [0, 8], [0, 10]]},
        {"axis_order": "ZYX", "bounds": [[0, 4], [0, 8]]},
        {"axis_order": "ZYX", "bounds": [[-1, 4], [0, 8], [0, 10]]},
        {"axis_order": "ZYX", "bounds": [[0, 5], [0, 8], [0, 10]]},
        {"axis_order": "ZYX", "bounds": [[0, 0], [0, 8], [0, 10]]},
        {"axis_order": "ZYX", "bounds": [[3, 2], [0, 8], [0, 10]]},
        {"axis_order": "ZYX", "bounds": [[True, 4], [0, 8], [0, 10]]},
        {"axis_order": "ZYX", "bounds": [[0.0, 4], [0, 8], [0, 10]]},
        {"axis_order": "ZYX", "bounds": [[0, 4], [0, 8], [0, 10]], "clip": True},
    ],
)
def test_roi_invalid_population_never_clips_or_guesses(roi):
    engine = session()
    result = act(engine, "inspect_source", roi=roi).public_result()
    assert result["status"] == "error"
    assert "diagnostics" not in result
    assert not engine.preview_images()


def test_roi_candidates_and_comparison_measure_only_region_and_do_not_crop_views():
    engine = session(include_input_images=True)
    source_previews = engine.preview_images()
    for threshold in (50, 150):
        assert trial(engine, threshold).public_result()["status"] == "ok"
    roi = {"axis_order": "ZYX", "bounds": [[0, 2], [0, 8], [0, 10]]}
    result = act(
        engine,
        "inspect_candidate",
        candidate_id="candidate1",
        role="wanted",
        roi=roi,
        views=[{"axis": "Z", "indices": [3]}],
    )
    info = result.public_result()["result"]["diagnostics"]
    assert info["foreground_count"] == 109
    assert info["element_count"] == 160
    assert "cut by ROI bounds" in info["population"]["meaning"]
    # The requested view stays full-size, while approved source context now
    # remains on the board automatically. Neither diagnostic ROI crops pixels.
    requested = [
        preview
        for preview in result.image_previews
        if preview.label.startswith("candidate1/wanted ")
    ]
    assert len(requested) == 1
    assert len(result.image_previews) == len(source_previews) + 1
    assert all(preview in result.image_previews for preview in source_previews)
    assert "native index 3" in requested[0].description
    assert "crop [0, 8, 0, 10]" in requested[0].description
    compared = act(
        engine,
        "compare_candidates",
        left="candidate1",
        right="candidate2",
        roles=["wanted"],
        roi=roi,
    ).public_result()
    record = compared["result"]["roles"][0]
    assert record["element_count"] == 160
    assert record["left_only_foreground"] == 100
    assert record["right_only_foreground"] == 0
    assert record["foreground_intersection"] == 9
    assert record["equal_value_count"] == 60
    assert "not biological accuracy" in record["meaning"]
    for candidate in engine.candidates:
        assert candidate._outputs[0][1].data.shape == (4, 8, 10)
        assert not candidate._outputs[0][1].data.flags.writeable


@pytest.mark.parametrize(
    "change", ["scale", "origin", "unit", "axis", "dtype", "shape"]
)
def test_roi_comparison_preserves_full_grid_and_dtype_contracts(change):
    engine = session()
    trial(engine, 50)
    trial(engine, 150)
    candidate = engine.candidates[-1]
    binding = candidate._outputs[0][1]
    state, data = binding.state, binding.data
    if change == "dtype":
        data = data.astype(np.uint8)
        data.flags.writeable = False
        state = replace(state, dtype=data.dtype.name)
    elif change == "shape":
        state = replace(state, shape=(3, 8, 10))
    else:
        fields = {
            "scale": {"scale": 3},
            "origin": {"translation": 2},
            "unit": {"unit": "nm"},
            "axis": {"name": "t", "type": "time"},
        }
        state = replace(
            state, axes=(replace(state.axes[0], **fields[change]), *state.axes[1:])
        )
    engine._candidates[-1] = replace(
        candidate, _outputs=(("wanted", replace(binding, state=state, data=data)),)
    )
    result = act(
        engine,
        "compare_candidates",
        left="candidate1",
        right="candidate2",
        roles=["wanted"],
        roi={"axis_order": "ZYX", "bounds": [[1, 2], [1, 2], [1, 2]]},
    ).public_result()
    assert result["status"] == "error"
    assert "roles" not in result.get("result", {})


@pytest.mark.parametrize(
    "dtype,values",
    [
        (np.uint64, [2**63, 2**63 + 1, 2**63 + 2, 2**63 + 3]),
        (np.int64, [-(2**63), -(2**63) + 1, -(2**63) + 2, -(2**63) + 3]),
        (np.bool_, [False, True, True, False]),
    ],
)
def test_roi_extreme_dtypes_preserve_exact_integer_and_boolean_values(dtype, values):
    pipeline = source_pipeline((2, 2))
    data = np.array(values, dtype=dtype).reshape(2, 2).copy()
    data.flags.writeable = False
    engine = session(pipeline)
    trial(engine)
    state = replace(engine._source.state, dtype=data.dtype.name)
    binding = replace(_capture_binding("test", 0, data, state, [0]), live=None)
    engine._candidates[-1] = replace(
        engine.candidates[-1], _outputs=(("wanted", binding),)
    )
    result = act(
        engine,
        "inspect_candidate",
        candidate_id="candidate1",
        role="wanted",
        roi={"axis_order": "YX", "bounds": [[0, 1], [0, 2]]},
        percentiles=[0, 100] if dtype is np.bool_ else [0, 50, 100],
    ).public_result()
    assert result["status"] == "ok", result
    info = result["result"]["diagnostics"]
    assert info["minimum"] == min(values[:2])
    assert info["maximum"] == max(values[:2])
    if dtype is not np.bool_:
        assert info["percentiles"][1]["value"] == {
            "numerator": 2 * int(values[0]) + 1,
            "denominator": 2,
        }
    np.testing.assert_array_equal(data.ravel(), values)


@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_comparison_rejects_nonfinite_foreground_without_numeric_repair(nonfinite):
    engine = session()
    for _ in range(2):
        trial(engine)
    candidate = engine.candidates[0]
    binding = candidate._outputs[0][1]
    data = binding.data.astype(float)
    data[1, 2, 3] = nonfinite
    data.flags.writeable = False
    state = replace(binding.state, dtype=data.dtype.name, kind="intensity image")
    frozen = replace(_capture_binding("test", 0, data, state, [0]), live=None)
    engine._candidates = [
        replace(c, _outputs=(("wanted", frozen),)) for c in engine.candidates
    ]
    result = act(
        engine,
        "compare_candidates",
        left="candidate1",
        right="candidate2",
        roles=["wanted"],
    ).public_result()
    assert result["status"] == "error"
    assert "Non-finite" in result["error"]
    assert not data.flags.writeable
    # A separately explicit disjoint ROI has a finite exact population.
    finite = act(
        engine,
        "compare_candidates",
        left="candidate1",
        right="candidate2",
        roles=["wanted"],
        roi={"axis_order": "ZYX", "bounds": [[0, 1], [0, 1], [0, 1]]},
    ).public_result()
    assert finite["status"] == "ok", finite


def test_measurement_handles_bind_roi_content_and_require_transmission():
    engine = session()
    trial(engine)
    first = act(
        engine,
        "inspect_candidate",
        candidate_id="candidate1",
        role="wanted",
        roi={"axis_order": "ZYX", "bounds": [[0, 1], [0, 1], [0, 2]]},
    ).public_result()["result"]
    second = act(
        engine,
        "inspect_candidate",
        candidate_id="candidate1",
        role="wanted",
        roi={"axis_order": "ZYX", "bounds": [[0, 1], [0, 1], [2, 4]]},
    ).public_result()["result"]
    assert first["sha256"] == second["sha256"]
    assert first["evidence_ref"]["sha256"] != second["evidence_ref"]["sha256"]
    with pytest.raises(InvestigationError, match="not transmitted"):
        engine._validate_evidence_refs(
            [first["evidence_ref"]], candidate_id="candidate1"
        )
    engine.record_model_call(request=engine.public_request())
    # The outgoing working set contains only the second ROI's literal result;
    # neither its index nor a quoted old handle discloses the first ROI.
    with pytest.raises(InvestigationError, match="not transmitted"):
        engine._validate_evidence_refs(
            [first["evidence_ref"]], candidate_id="candidate1"
        )
    engine._validate_evidence_refs([second["evidence_ref"]], candidate_id="candidate1")
    reloaded = act(
        engine,
        "inspect_candidate",
        candidate_id="candidate1",
        role="wanted",
        roi={"axis_order": "ZYX", "bounds": [[0, 1], [0, 1], [0, 2]]},
    ).public_result()["result"]
    assert reloaded == first
    engine.record_model_call(request=engine.public_request())
    engine._validate_evidence_refs([first["evidence_ref"]], candidate_id="candidate1")
    with pytest.raises(InvestigationError, match="exact experiment candidate"):
        engine._validate_evidence_refs(
            [first["evidence_ref"]], candidate_id="candidate2"
        )
    forged = first["evidence_ref"] | {"alias": "candidate2/wanted"}
    with pytest.raises(InvestigationError, match="not transmitted"):
        engine._validate_evidence_refs([forged])


def test_new_actions_keep_cancel_permission_and_action_limits():
    engine = session(limits=InvestigationLimits(max_actions=1))
    assert engine.public_request()["closing_phase"]["active"]
    with pytest.raises(InvestigationError, match="reserved"):
        act(
            engine,
            "inspect_source",
            roi={"axis_order": "ZYX", "bounds": [[0, 1], [0, 1], [0, 1]]},
        )
    assert engine.actions_used == 0
    engine = session()
    engine.include_output_images = False
    with pytest.raises(InvestigationStopped, match="permissions changed"):
        act(engine, "describe_nodes", operation_ids=["binary_threshold"])
    engine = session()
    engine.cancel()
    with pytest.raises(InvestigationStopped, match="cancelled"):
        act(engine, "find_nodes", query="*")


def test_full_eight_contract_batch_and_five_trials_fit_request_budget():
    from napari_vipp.core.workflow_proposal_catalogue import build_catalogue

    engine = session(limits=InvestigationLimits(max_attempts=5, max_actions=64))
    ids = [
        row["operation_id"]
        for row in sorted(
            build_catalogue()["operations"], key=lambda row: len(json.dumps(row))
        )[-8:]
    ]
    assert (
        act(engine, "describe_nodes", operation_ids=ids).public_result()["status"]
        == "ok"
    )
    for index in range(5):
        assert trial(engine, 50 + index).public_result()["status"] == "ok"
        inspected = act(
            engine,
            "inspect_candidate",
            candidate_id=f"candidate{index + 1}",
            role="wanted",
            roi={"axis_order": "ZYX", "bounds": [[0, 2], [1, 4], [2, 7]]},
            histogram=True,
            percentiles=[0, 100],
        ).public_result()
        assert inspected["status"] == "ok"
    encoded = json.dumps(engine.public_request(), separators=(",", ":")).encode()
    assert len(encoded) < MAX_REQUEST_BYTES
    # All original receipts and large contracts survive in the private journal;
    # bounded v2 intentionally does not retransmit that entire journal each turn.
    exported = engine.export_notebook()
    receipts = [
        row["payload"]
        for row, descriptor in zip(
            exported["records"], exported["record_index"], strict=True
        )
        if descriptor["kind"] == "action_receipt"
    ]
    assert len(receipts) == 11
    assert receipts == [json.loads(item) for item in engine._history]
    contract = next(row for row in receipts if row["action"] == "describe_nodes")
    assert len(contract["result"]) == 8
    assert engine.public_request()["version"] == 2
