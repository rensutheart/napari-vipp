from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def fixtures():
    name = "_vipp_background_fixture_tests"
    path = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "background_benchmark_fixtures.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)


@pytest.mark.parametrize("shape", [(32, 40), (7, 32, 40)])
@pytest.mark.parametrize(
    "dtype,scale",
    [("uint16", "native"), ("float32", "native"), ("float32", "normalized")],
)
@pytest.mark.parametrize("light", [False, True])
def test_deterministic_readonly_additive_phantoms(fixtures, shape, dtype, scale, light):
    a = fixtures.create_fixture(
        shape, dtype, intensity_scale=scale, light_background=light
    )
    b = fixtures.create_fixture(
        shape, dtype, intensity_scale=scale, light_background=light
    )
    assert a.input.dtype == np.dtype(dtype)
    for field in ("input", "background_truth", "signal_truth", "object_labels"):
        actual, expected = getattr(a, field), getattr(b, field)
        np.testing.assert_array_equal(actual, expected)
        assert actual.shape == shape
        assert not actual.flags.writeable
    ideal = (
        a.background_truth - a.signal_truth
        if light
        else a.background_truth + a.signal_truth
    )
    residual = a.input.astype(np.float64) - ideal
    assert np.sqrt(np.mean(residual**2)) < a.noise_sigma * 1.1
    assert a.input.min() >= 0
    assert a.input.max() <= a.intensity_ceiling
    assert a.signal_truth.max() > 0
    assert np.any(a.object_labels[0])  # edge-touching objects
    metadata = a.to_json_metadata()
    json.dumps(metadata, allow_nan=False)
    assert metadata["axes"] == ("YX" if len(shape) == 2 else "ZYX")
    if len(shape) == 3:
        assert not np.array_equal(a.signal_truth[0], a.signal_truth[shape[0] // 2])
        assert not np.array_equal(a.background_truth[0], a.background_truth[-1])


@pytest.mark.parametrize(
    "pattern", ["mixed", "sparse", "dense", "broad", "ramp", "vignette"]
)
def test_pattern_family_and_noise_seed(fixtures, pattern):
    a = fixtures.create_fixture((64, 64), "float32", pattern, seed=1)
    b = fixtures.create_fixture((64, 64), "float32", pattern, seed=2)
    assert not np.array_equal(a.input, b.input)
    np.testing.assert_array_equal(a.signal_truth, b.signal_truth)
    np.testing.assert_array_equal(a.background_truth, b.background_truth)


def test_quantization_and_intensity_scale_are_explicit(fixtures):
    floating = fixtures.create_fixture((32, 40), "float32", seed=9)
    integer = fixtures.create_fixture((32, 40), "uint16", seed=9)
    normalized = fixtures.create_fixture(
        (32, 40), "float32", intensity_scale="normalized", seed=9
    )
    np.testing.assert_array_equal(integer.input, np.rint(floating.input))
    np.testing.assert_array_equal(integer.signal_truth, floating.signal_truth)
    np.testing.assert_array_equal(integer.background_truth, floating.background_truth)
    np.testing.assert_allclose(normalized.input, floating.input / 60_000, atol=1e-7)
    assert floating.threshold == pytest.approx(normalized.threshold * 60_000)


def test_exact_and_known_numeric_errors(fixtures):
    a = np.array([[0, 1], [2, 3]], dtype=np.uint16)
    exact = fixtures.numerical_comparison(a, a.copy())
    assert exact["exact_equal_nan"]
    assert exact["mismatch_fraction"] == 0
    assert exact["rmse"] == 0
    assert exact["pearson_correlation"] == pytest.approx(1)
    changed = fixtures.numerical_comparison(a, a.astype(np.float32) + 2)
    assert not changed["exact_equal_nan"]
    assert not changed["dtype_equal"]
    assert changed["mismatch_fraction"] == 1
    assert changed["mae"] == 2
    assert changed["rmse"] == 2
    assert changed["max_absolute_error"] == 2
    assert changed["range_normalized_rmse"] == pytest.approx(2 / 3)


def test_correlation_is_stable_across_chunks_and_high_offsets(fixtures, monkeypatch):
    monkeypatch.setattr(fixtures, "_CHUNK", 2)
    a = np.arange(8, dtype=np.float64) + 1e12
    b = a + 2
    result = fixtures.numerical_comparison(a, b)
    assert result["pearson_correlation"] == pytest.approx(1)
    assert result["rmse"] == 2
    constant = fixtures.numerical_comparison(np.full(8, 2.4), np.full(8, 2.4))
    assert constant["pearson_correlation"] is None


def test_nonfinite_constant_and_empty_metrics_are_json_safe(fixtures):
    cases = [
        (np.array([np.nan, np.inf, -np.inf]), np.array([np.nan, np.inf, 0.0])),
        (np.ones(8), np.ones(8)),
        (np.array([]), np.array([])),
    ]
    for a, b in cases:
        result = fixtures.numerical_comparison(a, b)
        json.dumps(result, allow_nan=False)
        assert result["pearson_correlation"] is None
    assert fixtures.numerical_comparison(*cases[0])["nonfinite_mismatch_count"] == 1


def test_truth_metrics_and_fixed_threshold_do_not_fit_the_method(fixtures):
    fixture = fixtures.create_fixture((64, 80), "float32", intensity_scale="normalized")
    before = fixture.signal_truth.copy()
    result = fixtures.quality_comparison(
        fixture, fixture.signal_truth, fixture.signal_truth * 0
    )
    good = result["reference_vs_truth"]
    bad = result["candidate_vs_truth"]
    assert good["foreground_mean_bias"] == 0
    assert good["signal_retention_fraction"] == 1
    assert good["background_residual_rmse"] == 0
    assert good["fixed_threshold_dice"] == 1
    assert bad["signal_retention_fraction"] == 0
    assert bad["fixed_threshold_dice"] == 0
    assert bad["output_threshold_region_count"] == 0
    assert bad["truth_threshold_region_count"] == good["truth_threshold_region_count"]
    assert bad["fixed_threshold"] == good["fixed_threshold"] == fixture.threshold
    np.testing.assert_array_equal(fixture.signal_truth, before)
    json.dumps(result, allow_nan=False)


def test_positive_offset_detects_background_bias(fixtures):
    fixture = fixtures.create_fixture((32, 40), "float32", intensity_scale="normalized")
    result = fixtures.quality_against_truth(
        fixture, fixture.signal_truth + np.float32(0.1)
    )
    assert result["background_residual_mean"] == pytest.approx(0.1)
    assert result["foreground_mean_bias"] == pytest.approx(0.1)
    assert result["signal_retention_fraction"] > 1


def test_per_object_retention_exposes_one_lost_dim_structure(fixtures):
    fixture = fixtures.create_fixture((64, 80), "float32", intensity_scale="normalized")
    output = fixture.signal_truth.copy()
    lost_mask = fixture.object_labels == 3  # deliberately dim generated object
    assert lost_mask.any()
    output[lost_mask] = 0
    result = fixtures.quality_against_truth(fixture, output)
    per_object = {item["label"]: item for item in result["per_object_signal_retention"]}
    assert result["signal_retention_fraction"] > 0.95
    assert result["object_retention_min"] == 0
    assert result["object_retention_median"] == 1
    assert result["worst_retained_object_label"] == 3
    assert per_object[3]["signal_retention_fraction"] == 0
    assert per_object[3]["owned_voxel_count"] == np.count_nonzero(lost_mask)
    assert per_object[3]["truth_signal_sum"] > 0
    assert per_object[3]["output_signal_sum"] == 0
    output[lost_mask] = np.nan
    invalid = fixtures.quality_against_truth(fixture, output)
    invalid_object = next(
        obj for obj in invalid["per_object_signal_retention"] if obj["label"] == 3
    )
    assert invalid_object["signal_retention_fraction"] is None
    assert invalid_object["finite_voxel_count"] == 0


def test_invalid_fixture_requests_and_output_shapes_rejected(fixtures):
    for args, kwargs in [
        (((3, 4), "float32"), {}),
        (((32, 40), "float64"), {}),
        (((32, 40), "uint16"), {"intensity_scale": "normalized"}),
        (((32, 40), "float32"), {"pattern": "unknown"}),
    ]:
        with pytest.raises(ValueError):
            fixtures.create_fixture(*args, **kwargs)
    with pytest.raises(ValueError, match="shapes"):
        fixtures.numerical_comparison(np.zeros(3), np.zeros(4))
    fixture = fixtures.create_fixture((32, 40), "float32")
    with pytest.raises(ValueError, match="shape"):
        fixtures.quality_against_truth(fixture, np.zeros((40, 32)))
