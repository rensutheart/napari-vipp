"""Independent ImageJ 1.54p references and analytical/domain contracts."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from napari_vipp.core.imagej_gaussian import imagej_gaussian_blur
from napari_vipp.core.progress import OperationCancelled, ProgressContext


def _java_reference_cases():
    path = Path(__file__).parent / "fixtures" / "imagej_gaussian_reference_v1.json"
    reference = json.loads(path.read_text(encoding="utf-8"))
    assert reference["imagej_version"] == "1.54p"
    return reference["cases"]


@pytest.mark.parametrize("case", _java_reference_cases(), ids=lambda case: case["id"])
def test_independent_imagej_154p_java_references_are_bit_exact(case):
    dtype = np.dtype(case["dtype"])
    source = np.asarray(case["input_values"], dtype=dtype).reshape(case["shape"])
    expected = np.asarray(case["output_values"], dtype=dtype).reshape(case["shape"])
    before = source.copy()
    source.setflags(write=False)
    result = imagej_gaussian_blur(source, sigma=case["sigma"])
    assert result.dtype == dtype
    if dtype == np.dtype("float32"):
        np.testing.assert_array_equal(result.view(np.uint32), expected.view(np.uint32))
    else:
        np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(source, before)
    assert not np.shares_memory(result, source)


def test_negative_zero_java_reference_covers_interior_and_edge_arithmetic():
    case = next(
        item for item in _java_reference_cases()
        if item["id"] == "float32_41x41_1.5_negative_zero"
    )
    assert case["sigma"] == 1.5
    source = np.asarray(case["input_values"], dtype=np.float32).reshape(41, 41)
    expected = np.asarray(case["output_values"], dtype=np.float32).reshape(41, 41)
    assert np.signbit(source).all()
    assert np.count_nonzero(source) == 0
    assert np.signbit(expected[20, 20])
    assert not np.signbit(expected[0, 0])
    source.setflags(write=False)
    result = imagej_gaussian_blur(source, sigma=case["sigma"])
    np.testing.assert_array_equal(result.view(np.uint32), expected.view(np.uint32))
    assert np.signbit(source).all()


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_zero_sigma_returns_owned_copy_for_readonly_strided_input(dtype):
    source = np.arange(6 * 8, dtype=dtype).reshape(6, 8)[::2, ::-2]
    before = source.copy()
    source.setflags(write=False)
    result = imagej_gaussian_blur(source, sigma=0)
    np.testing.assert_array_equal(result, before)
    assert result.dtype == source.dtype
    assert not np.shares_memory(result, source)
    result[0, 0] = 0
    np.testing.assert_array_equal(source, before)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (7, 1), (11, 13)])
def test_constant_nearest_extension_and_small_axes(dtype, shape):
    source = np.full(shape, 37, dtype=dtype)
    source.setflags(write=False)
    result = imagej_gaussian_blur(source, sigma=8.5)
    np.testing.assert_allclose(result, source, rtol=2e-6, atol=0)
    assert result.dtype == source.dtype
    assert not np.shares_memory(result, source)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_full_unsigned_range_is_smoothed_without_wrap_or_input_mutation(dtype):
    maximum = np.iinfo(dtype).max
    source = np.zeros((13, 17), dtype=dtype)
    source[:, 8:] = maximum
    before = source.copy()
    source.setflags(write=False)
    result = imagej_gaussian_blur(source, sigma=1.5)
    assert result.dtype == source.dtype
    assert 0 < result[6, 7] < maximum
    assert 0 < result[6, 8] < maximum
    assert int(result.min()) >= 0
    assert int(result.max()) <= maximum
    np.testing.assert_array_equal(source, before)


def test_centered_impulse_has_gaussian_ratio_symmetry_and_unit_mass():
    source = np.zeros((41, 41), dtype=np.float32)
    source[20, 20] = 1
    result = imagej_gaussian_blur(source, sigma=1.5)
    np.testing.assert_allclose(result, result[::-1], rtol=0, atol=1e-8)
    np.testing.assert_allclose(result, result[:, ::-1], rtol=0, atol=1e-8)
    assert float(result.sum(dtype=np.float64)) == pytest.approx(1, abs=2e-7)
    expected_neighbor_ratio = math.exp(-0.5 / 1.5**2)
    assert float(result[20, 21] / result[20, 20]) == pytest.approx(
        expected_neighbor_ratio, rel=3e-7
    )


def test_leading_planes_remain_independent_and_signed_float_is_preserved():
    source = np.zeros((2, 3, 11, 13), dtype=np.float32)
    source[0, 0, 5, 6] = 1
    source[1, 2] = -37
    before = source.copy()
    source.setflags(write=False)
    result = imagej_gaussian_blur(source, sigma=1.5)
    np.testing.assert_array_equal(result[0, 1], 0)
    np.testing.assert_array_equal(result[1, 0], 0)
    np.testing.assert_allclose(result[1, 2], -37, rtol=2e-6)
    np.testing.assert_array_equal(source, before)
    assert result[0, 0, 5, 6] > 0


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_float_input_rejected_even_for_zero_sigma(value):
    source = np.zeros((3, 4), dtype=np.float32)
    source[1, 2] = value
    with pytest.raises(ValueError, match="finite input"):
        imagej_gaussian_blur(source, sigma=0)


@pytest.mark.parametrize(
    "sigma", [-1, 8.50001, np.nan, np.inf, -np.inf, True, "1.5", [1.5], 1j]
)
def test_invalid_sigma_rejected(sigma):
    with pytest.raises(ValueError, match="sigma"):
        imagej_gaussian_blur(np.zeros((3, 4), dtype=np.uint16), sigma=sigma)


@pytest.mark.parametrize("dtype", [bool, np.int16, np.uint32, np.uint64, np.float64])
def test_unsupported_dtype_requires_explicit_conversion(dtype):
    with pytest.raises(ValueError, match="explicit conversion"):
        imagej_gaussian_blur(np.zeros((3, 4), dtype=dtype))


@pytest.mark.parametrize("shape", [(4,), (0, 4), (4, 0), (0, 4, 5)])
def test_empty_or_non_yx_input_rejected(shape):
    with pytest.raises(ValueError, match="nonempty scalar YX"):
        imagej_gaussian_blur(np.zeros(shape, dtype=np.uint8))


def test_finite_float_overflow_fails_visibly():
    source = np.full((3, 4), np.finfo(np.float32).max, dtype=np.float32)
    with pytest.raises(ValueError, match="arithmetic overflowed"):
        imagej_gaussian_blur(source)


def test_immediate_cancellation_precedes_input_inspection_and_parameter_validation():
    class UnreadableInput:
        def __array__(self, *args, **kwargs):
            raise AssertionError("A cancelled operation must not inspect its input.")

    progress = ProgressContext(cancelled=lambda: True)
    with pytest.raises(OperationCancelled):
        imagej_gaussian_blur(UnreadableInput(), sigma=np.nan, progress=progress)


def test_cancellation_during_bounded_finite_validation_before_second_chunk(monkeypatch):
    source = np.zeros((3, 128, 512), dtype=np.float32)
    source.setflags(write=False)
    before = source.tobytes()
    cancelled = False
    inspected_sizes = []
    original_isfinite = np.isfinite

    def inspect_then_cancel(chunk):
        nonlocal cancelled
        inspected_sizes.append(chunk.size)
        result = original_isfinite(chunk)
        cancelled = True
        return result

    monkeypatch.setattr(np, "isfinite", inspect_then_cancel)
    progress = ProgressContext(cancelled=lambda: cancelled)
    with pytest.raises(OperationCancelled):
        imagej_gaussian_blur(source, sigma=0, progress=progress)
    assert inspected_sizes == [64 * 1024]
    assert source.tobytes() == before


def test_cancellation_inside_direction_before_second_kernel_product(monkeypatch):
    source = np.arange(31 * 37, dtype=np.uint16).reshape(31, 37)
    source.setflags(write=False)
    before = source.tobytes()
    cancelled = False
    products = 0
    original_multiply = np.multiply

    def product_then_cancel(*args, **kwargs):
        nonlocal cancelled, products
        result = original_multiply(*args, **kwargs)
        products += 1
        cancelled = True
        return result

    monkeypatch.setattr(np, "multiply", product_then_cancel)
    progress = ProgressContext(cancelled=lambda: cancelled)
    with pytest.raises(OperationCancelled):
        imagej_gaussian_blur(source, sigma=1.5, progress=progress)
    assert products == 1
    assert source.tobytes() == before


@pytest.mark.parametrize("sigma", [0, 1.5])
def test_all_leading_float_planes_validate_before_any_plane_is_filtered(
    monkeypatch, sigma
):
    import napari_vipp.core.imagej_gaussian as imagej_module

    source = np.zeros((2, 3, 128, 512), dtype=np.float32)
    source[-1, -1, -1, -1] = np.inf
    source.setflags(write=False)
    inspected_sizes = []
    original_isfinite = np.isfinite

    def bounded_inspection(chunk):
        inspected_sizes.append(chunk.size)
        return original_isfinite(chunk)

    def unexpected_filter(*args, **kwargs):
        raise AssertionError("Filtering began before every float value was validated.")

    monkeypatch.setattr(np, "isfinite", bounded_inspection)
    monkeypatch.setattr(imagej_module, "_convolve_direction", unexpected_filter)
    with pytest.raises(ValueError, match="finite input"):
        imagej_gaussian_blur(source, sigma=sigma)
    assert sum(inspected_sizes) == source.size
    assert max(inspected_sizes) <= 64 * 1024
