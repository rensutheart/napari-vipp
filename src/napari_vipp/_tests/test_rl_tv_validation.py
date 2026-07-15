from __future__ import annotations

import numpy as np

from napari_vipp.core.operations import _tv_divergence
from scripts.validate_rl_tv_phantoms import (
    _production_tv,
    calculate_metrics,
    forward_backward_tv_divergence,
    make_phantom_2d,
    make_phantom_3d,
    observed_image,
    rl_tv_comparison_variant,
)


def test_phantoms_are_deterministic_and_cover_required_features() -> None:
    for factory in (make_phantom_2d, make_phantom_3d):
        first = factory()
        second = factory()

        np.testing.assert_array_equal(first.truth, second.truth)
        np.testing.assert_array_equal(first.psf, second.psf)
        assert set(first.masks) == {
            "points",
            "thin_line",
            "bright_structure",
            "dim_structure",
            "border_structure",
        }
        assert all(np.any(mask) for mask in first.masks.values())
        assert first.psf.ndim == first.truth.ndim
        assert all(size % 2 == 1 for size in first.psf.shape)
        assert np.min(first.psf) >= 0
        assert np.isclose(np.sum(first.psf, dtype=np.float64), 1.0, rtol=1e-5)


def test_seeded_observation_and_metrics_are_finite() -> None:
    phantom = make_phantom_2d()
    first = observed_image(phantom)
    second = observed_image(phantom)

    np.testing.assert_array_equal(first, second)
    metrics = calculate_metrics(first, phantom)
    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics["dim_structure_loss_fraction"] >= 0
    assert metrics["border_mse"] >= 0


def test_comparison_variant_matches_production_zero_boundary() -> None:
    phantom = make_phantom_2d()
    observed = observed_image(phantom)

    expected = _production_tv(observed, phantom, iterations=3)
    actual, diagnostics = rl_tv_comparison_variant(
        observed,
        phantom.psf,
        iterations=3,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    assert diagnostics.minimum_raw_denominator > 0
    assert diagnostics.maximum_floor_fraction == 0


def test_forward_backward_divergence_is_zero_for_constant_field() -> None:
    values = np.ones((5, 7, 9), dtype=np.float32)

    divergence = forward_backward_tv_divergence(values, epsilon=1e-6)

    np.testing.assert_array_equal(divergence, np.zeros_like(values))


def test_production_tv_divergence_has_smoothing_sign_convention() -> None:
    peak = np.zeros((9, 9), dtype=np.float32)
    peak[4, 4] = 1.0
    trough = np.ones((9, 9), dtype=np.float32)
    trough[4, 4] = 0.0

    peak_divergence = _tv_divergence(peak, epsilon=1e-6)
    trough_divergence = _tv_divergence(trough, epsilon=1e-6)

    assert peak_divergence[4, 4] < 0
    assert trough_divergence[4, 4] > 0
