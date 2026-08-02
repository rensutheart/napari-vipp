from __future__ import annotations

import math

import numpy as np
import pytest

from napari_vipp.core import pipeline
from napari_vipp.core.operations import sigma_filter, sigma_filter_footprint
from napari_vipp.core.progress import OperationCancelled, ProgressContext


def _oracle_offsets(radius: float) -> tuple[tuple[int, int], ...]:
    normalized = float(radius)
    if 1.5 <= normalized < 1.75:
        normalized = 1.75
    elif 2.5 <= normalized < 2.85:
        normalized = 2.85
    r2 = math.floor(normalized * normalized) + 1
    extent = math.isqrt(r2)
    return tuple(
        (dy, dx)
        for dy in range(-extent, extent + 1)
        for dx in range(-extent, extent + 1)
        if dx * dx + dy * dy <= r2
    )


def _oracle_plane(
    plane: np.ndarray,
    *,
    radius: float,
    sigma_width: float,
    minimum_pixel_fraction: float,
    outlier_aware: bool,
) -> np.ndarray:
    source = np.asarray(plane, dtype=np.float32)
    offsets = _oracle_offsets(radius)
    minimum_count = math.ceil(len(offsets) * minimum_pixel_fraction)
    result = np.empty(source.shape, dtype=np.float64)
    height, width = source.shape
    for y in range(height):
        for x in range(width):
            samples: list[np.float32] = []
            total = 0.0
            total_squared = 0.0
            for dy, dx in offsets:
                yy = min(max(y + dy, 0), height - 1)
                xx = min(max(x + dx, 0), width - 1)
                value = np.float32(source[yy, xx])
                samples.append(value)
                total += float(value)
                total_squared += float(np.float32(value * value))
            mean = total / len(samples)
            raw_variance = total_squared / len(samples) - mean * mean
            variance = max(raw_variance, 0.0)
            spread = sigma_width * math.sqrt(variance)
            center = float(source[y, x])
            lower = center - spread
            upper = center + spread
            selected = [
                float(value) for value in samples if lower <= float(value) <= upper
            ]
            if len(selected) >= minimum_count:
                result[y, x] = sum(selected) / len(selected)
            elif outlier_aware:
                result[y, x] = (total - center) / (len(samples) - 1)
            else:
                result[y, x] = mean
    if plane.dtype in {np.dtype(np.uint8), np.dtype(np.uint16)}:
        maximum = np.iinfo(plane.dtype).max
        fiji_values = result.astype(np.float32)
        return np.clip(
            np.floor(fiji_values + np.float32(0.5)),
            0,
            maximum,
        ).astype(plane.dtype)
    return result.astype(np.float32)


@pytest.mark.parametrize(
    ("radius", "expected_r2", "expected_count"),
    (
        (0.5, 1, 5),
        (1.0, 2, 9),
        (1.499999, 3, 9),
        (1.5, 4, 13),
        (1.749999, 4, 13),
        (1.75, 4, 13),
        (2.0, 5, 21),
        (2.499999, 7, 21),
        (2.5, 9, 29),
        (2.849999, 9, 29),
        (2.85, 9, 29),
        (10.0, 101, 325),
    ),
)
def test_sigma_filter_footprint_matches_documented_discontinuities(
    radius: float,
    expected_r2: int,
    expected_count: int,
) -> None:
    r2, extent, offsets = sigma_filter_footprint(radius)

    assert r2 == expected_r2
    assert extent == math.isqrt(expected_r2)
    assert len(offsets) == expected_count
    assert offsets == _oracle_offsets(radius)


@pytest.mark.parametrize("boundary", (1.0, 1.5, 1.75, 2.0, 2.5, 2.85))
def test_sigma_filter_footprint_is_locked_at_adjacent_float64_radii(boundary) -> None:
    below = float(np.nextafter(np.float64(boundary), -np.inf))
    above = float(np.nextafter(np.float64(boundary), np.inf))

    for radius in (below, boundary, above):
        assert sigma_filter_footprint(radius)[2] == _oracle_offsets(radius)


def test_sigma_filter_radius_endpoints_reject_the_adjacent_outside_values() -> None:
    below = float(np.nextafter(np.float64(0.5), -np.inf))
    above = float(np.nextafter(np.float64(10.0), np.inf))

    with pytest.raises(ValueError, match="radius"):
        sigma_filter_footprint(below)
    with pytest.raises(ValueError, match="radius"):
        sigma_filter_footprint(above)


@pytest.mark.parametrize("dtype", (np.uint8, np.uint16, np.float32))
@pytest.mark.parametrize("sigma_width", (0.0, 1.0, 2.0, 3.0))
@pytest.mark.parametrize("minimum_fraction", (0.0, 0.2, 0.8, 1.0))
@pytest.mark.parametrize("outlier_aware", (False, True))
def test_sigma_filter_matches_independent_oracle_across_contract(
    dtype,
    sigma_width: float,
    minimum_fraction: float,
    outlier_aware: bool,
) -> None:
    values = np.array(
        [
            [0, 0, 1, 3, 8, 13, 21],
            [0, 2, 4, 6, 8, 10, 12],
            [1, 4, 9, 250, 9, 4, 1],
            [2, 6, 10, 14, 10, 6, 2],
            [50, 12, 8, 4, 0, 4, 8],
        ],
        dtype=dtype,
    )
    actual = sigma_filter(
        values,
        radius=2.0,
        sigma_width=sigma_width,
        minimum_pixel_fraction=minimum_fraction,
        outlier_aware=outlier_aware,
    )
    expected = _oracle_plane(
        values,
        radius=2.0,
        sigma_width=sigma_width,
        minimum_pixel_fraction=minimum_fraction,
        outlier_aware=outlier_aware,
    )

    if dtype is np.float32:
        np.testing.assert_array_equal(actual, expected)
    else:
        assert np.array_equal(actual, expected)


def test_sigma_filter_sigma_zero_interval_is_inclusive() -> None:
    values = np.zeros((3, 3), dtype=np.uint8)
    values[1, 1] = 10

    actual = sigma_filter(
        values,
        radius=0.5,
        sigma_width=0.0,
        minimum_pixel_fraction=0.2,
        outlier_aware=False,
    )

    assert actual[1, 1] == 10


def test_sigma_filter_threshold_interval_is_centered_on_pixel_not_mean() -> None:
    source = np.zeros((3, 3), dtype=np.float32)
    source[1, 2] = 1.0
    source[2, 1] = 2.0

    actual = sigma_filter(
        source,
        radius=0.5,
        sigma_width=1.0,
        minimum_pixel_fraction=0.8,
        outlier_aware=True,
    )

    # Centering the interval on the neighborhood mean would select four
    # samples and return 0.25. Fiji's documented center-pixel interval selects
    # only three, triggering the exclude-center fallback (3 / 4).
    assert actual[1, 1] == np.float32(0.75)


def test_sigma_filter_fallback_modes_and_half_up_restoration() -> None:
    values = np.zeros((3, 3), dtype=np.uint8)
    values[1, 1] = 10
    values[1, 0] = 1
    values[1, 2] = 1

    excluding_center = sigma_filter(
        values,
        radius=0.5,
        sigma_width=0.0,
        minimum_pixel_fraction=1.0,
        outlier_aware=True,
    )
    full_mean = sigma_filter(
        values,
        radius=0.5,
        sigma_width=0.0,
        minimum_pixel_fraction=1.0,
        outlier_aware=False,
    )

    assert excluding_center[1, 1] == 1  # 0.5 restored by half-up rounding.
    assert full_mean[1, 1] == 2


def test_sigma_filter_uses_clamped_boundaries_and_one_logical_center() -> None:
    values = np.zeros((2, 2), dtype=np.float32)
    values[0, 0] = 10.0

    actual = sigma_filter(
        values,
        radius=0.5,
        sigma_width=0.0,
        minimum_pixel_fraction=1.0,
        outlier_aware=True,
    )
    expected = _oracle_plane(
        values,
        radius=0.5,
        sigma_width=0.0,
        minimum_pixel_fraction=1.0,
        outlier_aware=True,
    )

    np.testing.assert_array_equal(actual, expected)
    assert actual[0, 0] == np.float32(5.0)


@pytest.mark.parametrize("dtype", (np.uint8, np.uint16, np.float32))
def test_sigma_filter_constant_and_tiny_planes_are_stable(dtype) -> None:
    value = 255 if dtype is np.uint8 else 65535 if dtype is np.uint16 else 65535.0
    for shape in ((1, 1), (1, 4), (4, 1), (2, 3)):
        source = np.full(shape, value, dtype=dtype)
        actual = sigma_filter(source, radius=10.0)
        assert np.array_equal(actual, source)


def test_sigma_filter_processes_leading_and_channel_axes_independently() -> None:
    base = np.arange(2 * 3 * 4 * 5 * 6, dtype=np.uint16).reshape(2, 3, 4, 5, 6)
    actual = sigma_filter(base, radius=1.0, channel_axis=2)

    assert actual.shape == base.shape
    assert actual.dtype == base.dtype
    for t in range(2):
        for z in range(3):
            for channel in range(4):
                expected = sigma_filter(base[t, z, channel], radius=1.0)
                assert np.array_equal(actual[t, z, channel], expected)


def test_sigma_filter_processes_explicit_tczyx_channel_axis_independently() -> None:
    source = np.arange(2 * 3 * 4 * 5 * 6, dtype=np.uint16).reshape(2, 3, 4, 5, 6)

    actual = sigma_filter(source, radius=1.0, channel_axis=1)

    for time in range(2):
        for channel in range(3):
            for z_index in range(4):
                expected = sigma_filter(source[time, channel, z_index], radius=1.0)
                np.testing.assert_array_equal(
                    actual[time, channel, z_index],
                    expected,
                )


@pytest.mark.parametrize(
    ("shape", "channel_axis"),
    (((4, 5), None), ((3, 4, 5), None), ((4, 5, 3), 2), ((3, 4, 5), 0)),
)
def test_sigma_filter_axis_forms(shape, channel_axis) -> None:
    source = np.arange(math.prod(shape), dtype=np.uint16).reshape(shape)
    actual = sigma_filter(source, radius=1.0, channel_axis=channel_axis)

    assert actual.shape == source.shape
    assert actual.dtype == source.dtype


def test_sigma_filter_handles_non_contiguous_input_without_mutating_it() -> None:
    owner = np.arange(3 * 8 * 10, dtype=np.float32).reshape(3, 8, 10)
    source = owner[:, ::2, 1::2].transpose(1, 0, 2)
    before = owner.copy()

    actual = sigma_filter(source, radius=1.75, channel_axis=1)

    assert not source.flags.c_contiguous
    assert actual.flags.c_contiguous
    assert actual.shape == source.shape
    np.testing.assert_array_equal(owner, before)


def test_sigma_filter_accepts_read_only_input_without_mutating_it() -> None:
    source = np.arange(35, dtype=np.uint8).reshape(5, 7)
    before = source.copy()
    source.setflags(write=False)

    actual = sigma_filter(source, radius=2.0)

    assert not source.flags.writeable
    np.testing.assert_array_equal(source, before)
    assert actual.flags.writeable


def test_sigma_filter_float32_negative_zero_is_canonicalized() -> None:
    source = np.full((3, 3), np.float32(-0.0), dtype=np.float32)

    actual = sigma_filter(source, radius=0.5)

    assert np.array_equal(actual, np.zeros_like(source))
    assert not np.signbit(actual).any()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    (
        ({"radius": 0.49}, "radius"),
        ({"radius": 10.01}, "radius"),
        ({"radius": math.nan}, "radius"),
        ({"sigma_width": -1.0}, "sigma_width"),
        ({"sigma_width": math.inf}, "sigma_width"),
        ({"minimum_pixel_fraction": -0.01}, "minimum_pixel_fraction"),
        ({"minimum_pixel_fraction": 1.01}, "minimum_pixel_fraction"),
        ({"outlier_aware": 1}, "outlier_aware"),
        ({"channel_axis": True}, "channel_axis"),
        ({"channel_axis": 4}, "channel_axis"),
    ),
)
def test_sigma_filter_rejects_invalid_parameters(kwargs, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        sigma_filter(np.ones((4, 4), dtype=np.uint8), **kwargs)


@pytest.mark.parametrize(
    "source",
    (
        np.ones((3, 3), dtype=bool),
        np.ones((3, 3), dtype=np.int16),
        np.ones((3, 3), dtype=np.float64),
        np.ones((3, 3), dtype=np.complex64),
        np.ones((3, 3), dtype=">u2"),
        np.ones((3, 3), dtype=">f4"),
    ),
)
def test_sigma_filter_rejects_unsupported_dtypes(source: np.ndarray) -> None:
    with pytest.raises(ValueError, match="uint8, uint16, and float32"):
        sigma_filter(source)


@pytest.mark.parametrize("nonfinite", (math.nan, math.inf, -math.inf))
def test_sigma_filter_rejects_nonfinite_input(nonfinite: float) -> None:
    source = np.ones((3, 3), dtype=np.float32)
    source[1, 1] = nonfinite

    with pytest.raises(ValueError, match="finite image intensities"):
        sigma_filter(source)


def test_sigma_filter_rejects_float32_square_overflow() -> None:
    safe_limit = np.float32(math.sqrt(float(np.finfo(np.float32).max)))
    unsafe = np.nextafter(safe_limit, np.float32(np.inf))
    source = np.full((3, 3), safe_limit, dtype=np.float32)

    np.testing.assert_array_equal(sigma_filter(source, radius=0.5), source)

    source[1, 1] = unsafe

    with pytest.raises(ValueError, match="square workspace"):
        sigma_filter(source)


class _RecordingProgress:
    def __init__(self) -> None:
        self.reports: list[tuple[int, int, str]] = []
        self.checks = 0

    def report(self, completed: int, total: int, message: str) -> None:
        self.reports.append((completed, total, message))

    def check_cancelled(self) -> None:
        self.checks += 1


def test_sigma_filter_reports_honest_row_block_progress() -> None:
    progress = _RecordingProgress()

    sigma_filter(
        np.ones((2, 130, 5), dtype=np.uint8),
        radius=0.5,
        progress=progress,
    )

    assert progress.reports[0] == (0, 12, "Sigma Filter validation")
    assert progress.reports[6] == (6, 12, "Sigma Filter validation")
    assert progress.reports[-1] == (12, 12, "Sigma Filter rows")
    # Initial, validation-row, calculation-row, and footprint-pass checkpoints.
    assert progress.checks == 25


def test_sigma_filter_progress_context_cancels_before_calculation() -> None:
    updates = []
    progress = ProgressContext(
        cancelled=lambda: len(updates) >= 2,
        reporter=updates.append,
    )

    with pytest.raises(OperationCancelled):
        sigma_filter(
            np.ones((130, 7), dtype=np.uint16),
            radius=10.0,
            progress=progress,
        )

    assert [(update.current, update.total) for update in updates] == [(0, 6), (1, 6)]
    assert {update.message for update in updates} == {"Sigma Filter validation"}


def test_sigma_filter_is_registered_as_scalar_default_positional_yx_node() -> None:
    spec = pipeline.NODE_LIBRARY_BY_ID["sigma_filter"]

    assert spec.title == "Sigma Filter"
    assert spec.category == "Filtering"
    assert spec.subcategory == "Smoothing & Denoising"
    assert spec.function is sigma_filter
    assert spec.stack_processing_note
    assert "sigma_filter" in pipeline._POSITIONAL_YX_OPERATIONS
    assert "sigma_filter" in pipeline.SCALAR_DEFAULT_CHANNEL_AXIS_OPERATIONS
    assert {parameter.name: parameter.default for parameter in spec.parameters} == {
        "radius": 2.0,
        "sigma_width": 2.0,
        "minimum_pixel_fraction": 0.2,
        "outlier_aware": True,
        "channel_axis": -1,
    }
    assert (
        pipeline.operation_call_parameter_value("sigma_filter", "channel_axis", -1)
        is None
    )
