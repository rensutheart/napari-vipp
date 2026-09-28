"""Analytical complete-placement and deterministic detection contracts."""

import json
from dataclasses import replace

import numpy as np
import pytest

from napari_vipp.core import detection
from napari_vipp.core.detection import (
    find_peaks,
    template_match,
    template_match_grid_contract,
    template_match_required_bytes,
)
from napari_vipp.core.detection_metadata import DetectionMetadata, TemplateMatchMetadata
from napari_vipp.core.metadata import (
    AxisMetadata,
    SourceMetadata,
    image_state_from_array,
)
from napari_vipp.core.progress import OperationCancelled, ProgressContext


def state(array, *, spacing=None, origin=None, units=None, names=None, frame="source"):
    names = names or ("yx" if array.ndim == 2 else "zyx")
    spacing = spacing or (1.0,) * array.ndim
    origin = origin or (0.0,) * array.ndim
    units = units or ("pixel",) * array.ndim
    return image_state_from_array(
        array,
        axes=tuple(
            AxisMetadata(
                name, "space" if name in "zyx" else "channel", unit, scale, offset
            )
            for name, unit, scale, offset in zip(
                names, units, spacing, origin, strict=True
            )
        ),
        source=SourceMetadata(source_uuid=frame),
        source_name=frame,
        defer_statistics=True,
    )


def match(search, template, **states):
    return template_match(
        search,
        template,
        search_state=states.get("search_state", state(search)),
        template_state=states.get("template_state", state(template)),
    )


def peaks_from_match(result, **params):
    scores, valid, score_state, mask_state = result
    return find_peaks(
        scores, valid, image_state=score_state, mask_state=mask_state, **params
    )


@pytest.mark.parametrize("shape", [(3, 5), (4, 2), (3, 4, 2)])
@pytest.mark.parametrize("noise", [0.0, 0.03])
def test_known_independent_placements_even_odd_borders_and_noisy_volumes(shape, noise):
    rng = np.random.default_rng(2409)
    # Independent source construction, no detection/backend-based fixture truth.
    template = rng.normal(size=shape)
    search_shape = tuple(4 * size + 6 for size in shape)
    placements = [
        tuple(0 for _ in shape),
        tuple(s - t for s, t in zip(search_shape, shape, strict=True)),
    ]
    search = rng.normal(0, noise, search_shape)
    for placement in placements:
        slices = tuple(
            slice(start, start + width)
            for start, width in zip(placement, shape, strict=True)
        )
        search[slices] = 2.0 * template + 7.0 + rng.normal(0, noise, shape)
    original = search.copy()
    search.setflags(write=False)
    template.setflags(write=False)
    result = match(search, template)
    scores, valid, score_state, mask_state = result
    assert scores.shape == tuple(
        s - t + 1 for s, t in zip(search_shape, shape, strict=True)
    )
    assert scores.dtype == np.float64 and valid.dtype == bool
    assert score_state.axes == mask_state.axes
    assert tuple(axis.translation for axis in score_state.axes) == tuple(
        (size - 1) / 2 for size in shape
    )
    table = peaks_from_match(result, minimum_value=0.999 if not noise else 0.99)
    actual = {
        tuple(record[f"{axis}_index"] for axis in score_state.axis_order.lower())
        for record in table.records()
    }
    expected = {
        tuple(
            start + (size - 1) / 2 for start, size in zip(placement, shape, strict=True)
        )
        for placement in placements
    }
    assert actual == expected
    assert table.detection_metadata.accepted_count == 2
    for record in table.records():
        for axis, width in zip(score_state.axis_order.lower(), shape, strict=True):
            assert (
                record[f"template_{axis}_stop"] - record[f"template_{axis}_start"]
                == width
            )
        assert record["score"] > 0.99
    np.testing.assert_array_equal(search, original)
    assert not search.flags.writeable and not template.flags.writeable


def test_scores_match_direct_window_pearson_oracle_without_backend_oracle():
    rng = np.random.default_rng(519)
    search, template = rng.normal(size=(9, 10)), rng.normal(size=(4, 3))
    scores, valid, *_ = match(search, template)
    expected = np.empty(scores.shape)
    tc = template - template.mean()
    for index in np.ndindex(scores.shape):
        window = search[index[0] : index[0] + 4, index[1] : index[1] + 3]
        wc = window - window.mean()
        expected[index] = np.sum(wc * tc) / np.sqrt(np.sum(wc * wc) * np.sum(tc * tc))
    np.testing.assert_allclose(scores, expected, atol=1e-12, rtol=1e-12)
    assert valid.all()


def test_adjacent_templates_retained_at_exact_separation_and_suppressed_below():
    template = np.array([[0.0, 1.0, 3.0], [2.0, 8.0, -1.0], [4.0, 0.0, 5.0]])
    search = np.concatenate((template, template), axis=1)
    result = match(search, template)
    retained = peaks_from_match(result, minimum_value=0.999, minimum_separation=3)
    assert {(row[1], row[2]) for row in retained.rows} == {(1.0, 1.0), (1.0, 4.0)}
    suppressed = peaks_from_match(result, minimum_value=0.999, minimum_separation=3.01)
    assert suppressed.row_count == 1
    assert suppressed.detection_metadata.candidate_count == 2
    assert suppressed.detection_metadata.accepted_count == 1


def test_noncontiguous_readonly_inputs_and_owned_outputs():
    backing = np.random.default_rng(40).normal(size=(20, 24))
    search = backing[::2, ::2]
    template = search[2:5, 3:7]
    before = backing.copy()
    backing.setflags(write=False)
    search.setflags(write=False)
    template.setflags(write=False)
    scores, valid, ss, ms = match(search, template)
    assert not search.flags.c_contiguous and not template.flags.c_contiguous
    assert scores[2, 3] == pytest.approx(1, abs=1e-12)
    assert not np.shares_memory(scores, search)
    assert not np.shares_memory(valid, search)
    table = find_peaks(
        scores, valid, image_state=ss, mask_state=ms, minimum_value=0.999
    )
    assert table.rows[0][1:3] == (3.0, 4.5)
    np.testing.assert_array_equal(backing, before)


@pytest.mark.parametrize(
    "dtype", [np.bool_, np.int16, np.uint8, np.float32, np.float64]
)
def test_dtype_and_constant_window_mask(dtype):
    template = np.array([[0, 1], [1, 0]], dtype=dtype)
    search = np.zeros((8, 9), dtype=dtype)
    search[3:5, 4:6] = template
    result = match(search, template)
    scores, valid, *_ = result
    assert scores[3, 4] == pytest.approx(1, abs=1e-12)
    assert not valid[0, 0] and scores[0, 0] == 0
    # A non-positive threshold must never select undefined constant windows.
    table = peaks_from_match(result, minimum_value=-1)
    assert all(valid[int(row[1] - 0.5), int(row[2] - 0.5)] for row in table.rows)


def test_absent_and_inverted_template_signed_response():
    template = np.array([[1.0, 3.0, 2.0], [5.0, 0.0, -2.0]])
    empty = match(np.full((10, 12), 19.0), template)
    assert not empty[1].any()
    assert peaks_from_match(empty, minimum_value=-1).row_count == 0
    inverted = match(-template, template)
    assert inverted[0].item() == pytest.approx(-1)
    assert peaks_from_match(inverted).row_count == 0


def test_large_exact_integer_offset_and_tiny_float_amplitude():
    template = np.array([[0, 1], [3, 2]], dtype=np.int64) + (2**52)
    scores, valid, *_ = match(template, template)
    assert valid.item() and scores.item() == pytest.approx(1, abs=1e-12)


def test_low_contrast_window_next_to_huge_unrelated_outlier_is_not_placeholder_zero():
    template = np.array([[0.0, 1.0], [3.0, 2.0]])
    search = np.zeros((9, 10))
    search[2:4, 5:7] = template * 1e-150
    search[8, 9] = 1e150
    scores, valid, *_ = match(search, template)
    assert valid[2, 5]
    assert scores[2, 5] == pytest.approx(1, abs=1e-12)
    tiny = np.array([[0.0, 1.0], [3.0, 2.0]]) * 1e-250
    scores, valid, *_ = match(tiny, tiny)
    assert valid.item() and scores.item() == pytest.approx(1, abs=1e-12)


def test_physical_coordinates_compatible_units_and_even_template():
    template = np.array([[0.0, 1.0], [3.0, 2.0]])
    search = np.zeros((8, 9))
    search[2:4, 5:7] = template
    result = match(
        search,
        template,
        search_state=state(
            search, spacing=(2.0, 0.5), origin=(-7.0, 11.0), units=("um", "um")
        ),
        template_state=state(
            template, spacing=(2000.0, 500.0), origin=(123.0, 456.0), units=("nm", "nm")
        ),
    )
    row = peaks_from_match(result, minimum_value=0.99).records()[0]
    assert (row["y_index"], row["x_index"]) == (2.5, 5.5)
    assert (row["y_physical"], row["x_physical"]) == (-2.0, 13.75)
    assert (row["template_y_start"], row["template_x_stop"]) == (2, 7)


def test_peaks_anisotropic_euclidean_separation_and_exact_boundary():
    image = np.zeros((6, 10, 10))
    image[1, 2, 2], image[3, 2, 2], image[1, 5, 2] = 3, 2, 1
    metadata = state(image, spacing=(4, 1, 1), units=("um",) * 3)
    pixel = find_peaks(image, image_state=metadata, minimum_separation=3)
    physical = find_peaks(
        image,
        image_state=metadata,
        minimum_separation=3,
        separation_units="Physical (micrometers)",
    )
    assert pixel.row_count == 2  # 2-pixel Z pair suppressed, exact 3-pixel Y retained.
    assert physical.row_count == 3  # Z separation is 8 micrometers.


def test_plateaus_ties_inclusive_threshold_and_exact_cap_counts():
    image = np.zeros((12, 12))
    image[1:3, 1:3] = 4
    image[1, 8] = image[8, 1] = 4
    image[8, 8] = 2
    result = find_peaks(
        image, image_state=state(image), minimum_value=2, maximum_detections=2
    )
    assert result.rows == ((1, 1.0, 1.0, 4.0), (2, 1.0, 8.0, 4.0))
    evidence = result.detection_metadata
    assert (
        evidence.candidate_count,
        evidence.accepted_count,
        evidence.returned_count,
        evidence.truncated,
    ) == (4, 4, 2, True)
    assert (
        DetectionMetadata.from_dict(json.loads(json.dumps(evidence.to_dict())))
        == evidence
    )


def test_optional_mask_is_absent_from_neighborhood_not_zero_and_border_exclusion():
    image = np.full((5, 8), -10.0)
    image[0, 0], image[2, 2], image[2, 3] = -1, -3, 5
    valid = np.ones(image.shape, dtype=bool)
    valid[2, 3] = False
    result = find_peaks(
        image,
        valid,
        image_state=state(image),
        mask_state=state(valid),
        minimum_value=-3,
        border_exclusion=1,
    )
    assert result.rows == ((1, 2.0, 2.0, -3.0),)
    assert (
        find_peaks(image, image_state=state(image), border_exclusion=100).row_count == 0
    )


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_rejected_even_outside_optional_mask(value):
    array = np.arange(36.0).reshape(6, 6)
    array[0, 0] = value
    with pytest.raises(ValueError, match="finite"):
        match(array, np.eye(3))
    mask = np.ones(array.shape, dtype=bool)
    mask[0, 0] = False
    with pytest.raises(ValueError, match="finite"):
        find_peaks(array, mask, image_state=state(array), mask_state=state(mask))


@pytest.mark.parametrize(
    "array",
    [
        np.full((3, 4), 2**53 + 1, dtype=np.uint64),
        np.full((3, 4), -(2**53) - 1, dtype=np.int64),
        np.ones((3, 4), dtype=complex),
        np.empty((0, 4)),
    ],
)
def test_unsafe_types_ranges_and_empty_are_rejected(array):
    with pytest.raises(ValueError):
        match(array, np.eye(2))
    with pytest.raises(ValueError):
        find_peaks(array, image_state=state(array))


def test_grid_and_input_state_rejections():
    search, template = np.arange(100.0).reshape(10, 10), np.eye(3)
    for bad in (
        None,
        image_state_from_array(search),
        replace(state(search), shape=(11, 10)),
        replace(state(search), dtype="uint16"),
        state(search, names="xy"),
        state(search, names="cy"),
    ):
        with pytest.raises(ValueError):
            match(search, template, search_state=bad)
    with pytest.raises(ValueError, match="sampling"):
        match(search, template, template_state=state(template, spacing=(2, 1)))
    with pytest.raises(ValueError, match="units"):
        match(search, template, template_state=state(template, units=("um", "pixel")))
    with pytest.raises(ValueError, match="nonconstant"):
        match(search, np.ones((3, 3)))
    with pytest.raises(ValueError, match="equal rank"):
        match(search, np.ones((11, 11)))
    shape, axes = template_match_grid_contract(
        search.shape, template.shape, state(search), state(template)
    )
    assert shape == (8, 8) and tuple(axis.translation for axis in axes) == (1, 1)


def test_mask_and_score_exact_pair_required_and_immutable_schema():
    result = match(np.arange(100.0).reshape(10, 10), np.eye(3))
    scores, valid, ss, ms = result
    assert (
        TemplateMatchMetadata.from_dict(
            json.loads(json.dumps(ss.template_match_metadata.to_dict()))
        )
        == ss.template_match_metadata
    )
    with pytest.raises(ValueError, match="matching valid-score mask"):
        find_peaks(scores, image_state=ss)
    wrong = valid.copy()
    wrong[0, 0] = not wrong[0, 0]
    with pytest.raises(ValueError, match="same unchanged result"):
        find_peaks(scores, wrong, image_state=ss, mask_state=ms)
    changed = scores.copy()
    changed[0, 0] += 0.1
    with pytest.raises(ValueError, match="same unchanged result"):
        find_peaks(changed, valid, image_state=ss, mask_state=ms)
    with pytest.raises(ValueError, match="metadata is missing"):
        find_peaks(
            scores,
            valid,
            image_state=replace(ss, template_match_metadata=None),
            mask_state=ms,
        )
    payload = ss.template_match_metadata.to_dict()
    payload["search_scale"] = [float("nan"), 1]
    with pytest.raises(ValueError):
        TemplateMatchMetadata.from_dict(payload)
    with pytest.raises(ValueError):
        replace(peaks_from_match(result).detection_metadata, accepted_count=-1)


@pytest.mark.parametrize(
    "axis_change", [{"scale": 2}, {"translation": 3}, {"unit": "um"}]
)
def test_template_calibration_may_not_be_changed_on_both_outputs(axis_change):
    scores, valid, ss, ms = match(np.arange(100.0).reshape(10, 10), np.eye(3))
    axes = (replace(ss.axes[0], **axis_change), replace(ss.axes[1], **axis_change))
    with pytest.raises(ValueError, match="calibration changed"):
        find_peaks(
            scores,
            valid,
            image_state=replace(ss, axes=axes),
            mask_state=replace(ms, axes=axes),
        )


@pytest.mark.parametrize(
    "params",
    [
        {"minimum_value": True},
        {"minimum_value": np.nan},
        {"minimum_separation": -1},
        {"minimum_separation": np.inf},
        {"maximum_detections": 0},
        {"maximum_detections": 1.5},
        {"border_exclusion": -1},
        {"border_exclusion": True},
        {"separation_units": "um"},
        {"separation_units": "Physical (micrometers)"},
    ],
)
def test_peak_parameter_domains(params):
    image = np.eye(7)
    with pytest.raises(ValueError):
        find_peaks(image, image_state=state(image), **params)


def test_memory_refusal_and_cancellation_never_publish_partial_results(monkeypatch):
    search, template = np.arange(100.0).reshape(10, 10), np.eye(3)
    assert template_match_required_bytes((100, 100, 100), (9, 9, 9)) > 100**3 * 8 * 20

    def refuse(*args, **kwargs):
        raise MemoryError("test memory gate")

    monkeypatch.setattr(detection, "_guard_memory", refuse)
    with pytest.raises(MemoryError, match="gate"):
        match(search, template)
    with pytest.raises(MemoryError, match="gate"):
        find_peaks(search, image_state=state(search))
    progress = ProgressContext(cancelled=lambda: True)
    with pytest.raises(OperationCancelled):
        template_match(
            search,
            template,
            search_state=state(search),
            template_state=state(template),
            progress_context=progress,
        )
    with pytest.raises(OperationCancelled):
        find_peaks(search, image_state=state(search), progress_context=progress)


def test_cancellation_after_native_correlation_is_observed(monkeypatch):
    cancelled = False
    backend = detection.match_template

    def cancel_after(*args, **kwargs):
        nonlocal cancelled
        result = backend(*args, **kwargs)
        cancelled = True
        return result

    monkeypatch.setattr(detection, "match_template", cancel_after)
    search, template = np.arange(100.0).reshape(10, 10), np.eye(3)
    with pytest.raises(OperationCancelled):
        template_match(
            search,
            template,
            search_state=state(search),
            template_state=state(template),
            progress_context=ProgressContext(cancelled=lambda: cancelled),
        )
