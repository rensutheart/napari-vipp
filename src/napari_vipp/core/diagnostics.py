"""Exact, Qt-free diagnostic calculations for scientific image arrays.

These helpers inspect every relevant value. They do not sample arrays or infer
RGB/channel semantics from shape; callers must provide an explicit channel axis
when a per-channel histogram is required.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from numbers import Rational
from typing import TYPE_CHECKING

import numpy as np
from scipy import ndimage as ndi

from napari_vipp.core.grid import (
    ImageGrid,
    compare_psf_sampling,
    psf_physical_sampling_known,
)
from napari_vipp.core.operations import exact_integer_percentiles

if TYPE_CHECKING:
    from napari_vipp.core.metadata import ImageState

DIAGNOSTIC_CHUNK_ELEMENTS = 1_048_576
DISPLAY_HISTOGRAM_BINS = 128
PSF_PREFLIGHT_MAX_SCAN_ELEMENTS = 1_000_000
PSF_CENTER_OFFSET_WARNING_VOXELS = 0.25
PSF_NORMALIZATION_RTOL = 1e-3
PSF_EDGE_MASS_WARNING_FRACTION = 0.01

PSF_PREFLIGHT_READY = "Ready"
PSF_PREFLIGHT_WARNING = "Warning"
PSF_PREFLIGHT_INVALID = "Invalid"
PSF_PREFLIGHT_UNKNOWN = "Unknown"


@dataclass(frozen=True, slots=True)
class PsfPreflightIssue:
    """One conservative PSF-readiness finding."""

    severity: str
    code: str
    detail: str


@dataclass(frozen=True, slots=True)
class WidefieldNyquistResult:
    """Critical sampling distances for a conventional widefield PSF model."""

    spatial_ndim: int
    xy_step_um: float
    z_step_um: float | None
    xy_limit_um: float
    z_limit_um: float | None
    xy_met: bool
    z_met: bool | None

    @property
    def met(self) -> bool:
        return self.xy_met and self.z_met is not False


@dataclass(frozen=True, slots=True)
class PsfPreflightResult:
    """Qt-free image/PSF readiness summary for deconvolution."""

    status: str
    issues: tuple[PsfPreflightIssue, ...]
    spatial_ndim: int | None
    image_shape: tuple[int, ...]
    psf_shape: tuple[int, ...]
    spatial_axis_labels: tuple[str, ...] = ()
    values_scanned: bool = False
    psf_sum: float | None = None
    approximately_normalized: bool | None = None
    odd_shape: bool | None = None
    peak_index: tuple[int, ...] | None = None
    peak_offset_voxels: float | None = None
    centroid: tuple[float, ...] | None = None
    centroid_offset_voxels: float | None = None
    support_fraction_of_image: tuple[float, ...] | None = None
    edge_mass_fraction: float | None = None
    edge_mass_fraction_by_axis: tuple[float, ...] | None = None
    centered_mass_within_image_support_by_axis: tuple[float, ...] | None = None
    physical_sampling_known: bool = False


def psf_preflight(
    image,
    psf,
    *,
    spatial_ndim: int | None,
    image_state: ImageState | None = None,
    psf_state: ImageState | None = None,
    max_scan_elements: int = PSF_PREFLIGHT_MAX_SCAN_ELEMENTS,
) -> PsfPreflightResult:
    """Inspect PSF readiness without changing either input.

    Full value diagnostics are limited to already materialized NumPy arrays
    below ``max_scan_elements``. Array-like lazy data are never coerced merely
    to paint an inspector status.
    """
    issues: list[PsfPreflightIssue] = []
    image_shape = _preflight_shape(image, image_state)
    psf_shape = _preflight_shape(psf, psf_state)
    resolved_ndim = _preflight_spatial_ndim(spatial_ndim)
    spatial_axis_labels = _preflight_spatial_axis_labels(
        image_state,
        resolved_ndim,
    )

    if image is None or psf is None:
        missing = []
        if image is None:
            missing.append("Image")
        if psf is None:
            missing.append("PSF")
        issues.append(
            PsfPreflightIssue(
                "unknown",
                "unresolved_inputs",
                f"Connect and resolve the {' and '.join(missing)} input(s) to "
                "complete PSF preflight.",
            )
        )
    if resolved_ndim is None:
        issues.append(
            PsfPreflightIssue(
                "unknown",
                "unresolved_spatial_rank",
                "Resolve 2D YX or 3D ZYX spatial processing to check PSF rank.",
            )
        )

    rank_valid = bool(resolved_ndim is not None and image_shape and psf_shape)
    if rank_valid and len(image_shape) < resolved_ndim:
        issues.append(
            PsfPreflightIssue(
                "invalid",
                "image_rank",
                f"Image rank is {len(image_shape)}D but {resolved_ndim}D spatial "
                "processing was requested.",
            )
        )
        rank_valid = False
    if rank_valid and len(psf_shape) != resolved_ndim:
        issues.append(
            PsfPreflightIssue(
                "invalid",
                "psf_rank",
                f"PSF rank is {len(psf_shape)}D but deconvolution is resolved as "
                f"{resolved_ndim}D.",
            )
        )
        rank_valid = False

    support_fraction: tuple[float, ...] | None = None
    if rank_valid and resolved_ndim is not None:
        image_spatial_shape = image_shape[-resolved_ndim:]
        support_fraction = tuple(
            float(psf_size) / float(image_size)
            for psf_size, image_size in zip(
                psf_shape,
                image_spatial_shape,
                strict=True,
            )
            if image_size > 0
        )
        if len(support_fraction) != resolved_ndim:
            support_fraction = None
        else:
            for axis_label, psf_size, image_size in zip(
                spatial_axis_labels,
                psf_shape,
                image_spatial_shape,
                strict=True,
            ):
                if psf_size < image_size:
                    continue
                comparison = (
                    "the same size as" if psf_size == image_size else "larger than"
                )
                issues.append(
                    PsfPreflightIssue(
                        "warning",
                        "support_vs_image",
                        f"{axis_label} support is {comparison} the image "
                        f"({psf_size} PSF samples versus {image_size} image "
                        "samples). No output sample on this axis has full PSF "
                        "support on both sides. The calculation can complete, "
                        "but current same-size convolution treats signal beyond "
                        "the image as zero.",
                    )
                )

    physical_sampling_known = False
    if rank_valid and resolved_ndim is not None:
        if image_state is None or psf_state is None:
            issues.append(_missing_psf_calibration_issue())
        else:
            try:
                image_grid = ImageGrid.from_image_state(image_state)
                psf_grid = ImageGrid.from_image_state(psf_state)
                physical_sampling_known = psf_physical_sampling_known(
                    image_grid,
                    psf_grid,
                    spatial_ndim=resolved_ndim,
                )
                if not physical_sampling_known:
                    issues.append(_missing_psf_calibration_issue())
                else:
                    compatibility = compare_psf_sampling(
                        image_grid,
                        psf_grid,
                        spatial_ndim=resolved_ndim,
                    )
                    if not compatibility.compatible:
                        detail = "; ".join(
                            issue.detail for issue in compatibility.issues
                        )
                        issues.append(
                            PsfPreflightIssue(
                                "invalid",
                                "sampling_mismatch",
                                "Image and PSF physical sampling are incompatible: "
                                f"{detail}. VIPP will not resample the PSF "
                                "implicitly.",
                            )
                        )
            except (TypeError, ValueError) as exc:
                issues.append(
                    PsfPreflightIssue(
                        "unknown",
                        "invalid_metadata",
                        f"PSF sampling metadata could not be checked: {exc}",
                    )
                )

    array = _preflight_scan_array(psf, max_scan_elements=max_scan_elements)
    if psf is not None and array is None:
        issues.append(
            PsfPreflightIssue(
                "unknown",
                "values_not_scanned",
                "PSF values were not scanned because the data are lazy, "
                "non-array, or exceed the inspector scan limit.",
            )
        )

    values_scanned = array is not None
    psf_sum: float | None = None
    approximately_normalized: bool | None = None
    odd_shape: bool | None = None
    peak_index: tuple[int, ...] | None = None
    peak_offset: float | None = None
    centroid: tuple[float, ...] | None = None
    centroid_offset: float | None = None
    edge_mass_fraction: float | None = None
    edge_mass_fraction_by_axis: tuple[float, ...] | None = None
    centered_mass_within_image_support_by_axis: tuple[float, ...] | None = None
    if array is not None:
        (
            psf_sum,
            approximately_normalized,
            odd_shape,
            peak_index,
            peak_offset,
            centroid,
            centroid_offset,
            edge_mass_fraction,
            edge_mass_fraction_by_axis,
        ) = _inspect_psf_values(
            array,
            issues,
            axis_labels=spatial_axis_labels,
        )
        issue_codes = {issue.code for issue in issues}
        if (
            rank_valid
            and resolved_ndim is not None
            and psf_sum is not None
            and psf_sum > 0
            and not ({"negative", "nonfinite"} & issue_codes)
        ):
            centered_mass_within_image_support_by_axis = (
                _psf_centered_mass_within_axis_support(
                    array,
                    psf_sum,
                    image_shape[-resolved_ndim:],
                )
            )

    return PsfPreflightResult(
        status=_psf_preflight_status(issues),
        issues=tuple(issues),
        spatial_ndim=resolved_ndim,
        image_shape=image_shape,
        psf_shape=psf_shape,
        spatial_axis_labels=spatial_axis_labels,
        values_scanned=values_scanned,
        psf_sum=psf_sum,
        approximately_normalized=approximately_normalized,
        odd_shape=odd_shape,
        peak_index=peak_index,
        peak_offset_voxels=peak_offset,
        centroid=centroid,
        centroid_offset_voxels=centroid_offset,
        support_fraction_of_image=support_fraction,
        edge_mass_fraction=edge_mass_fraction,
        edge_mass_fraction_by_axis=edge_mass_fraction_by_axis,
        centered_mass_within_image_support_by_axis=(
            centered_mass_within_image_support_by_axis
        ),
        physical_sampling_known=physical_sampling_known,
    )


def _inspect_psf_values(
    array: np.ndarray,
    issues: list[PsfPreflightIssue],
    *,
    axis_labels: tuple[str, ...] = (),
) -> tuple[
    float | None,
    bool | None,
    bool | None,
    tuple[int, ...] | None,
    float | None,
    tuple[float, ...] | None,
    float | None,
    float | None,
    tuple[float, ...] | None,
]:
    if array.size == 0 or any(int(size) <= 0 for size in array.shape):
        issues.append(PsfPreflightIssue("invalid", "empty", "PSF is empty."))
        return (None, None, None, None, None, None, None, None, None)
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype,
        np.complexfloating,
    ):
        issues.append(
            PsfPreflightIssue(
                "invalid",
                "non_numeric",
                "PSF values must be real numeric samples.",
            )
        )
        return (None, None, None, None, None, None, None, None, None)

    finite = bool(np.all(np.isfinite(array)))
    if not finite:
        issues.append(
            PsfPreflightIssue(
                "invalid",
                "nonfinite",
                "PSF contains non-finite values; use Prepare / Validate PSF.",
            )
        )
    negative = bool(np.any(array < 0))
    if negative:
        issues.append(
            PsfPreflightIssue(
                "invalid",
                "negative",
                "PSF contains negative values; use Prepare / Validate PSF.",
            )
        )

    odd_shape = all(int(size) % 2 == 1 for size in array.shape)
    if not odd_shape:
        issues.append(
            PsfPreflightIssue(
                "warning",
                "even_shape",
                "PSF has an even dimension and no single central sample; use "
                "Prepare / Validate PSF.",
            )
        )

    if not finite:
        return (None, None, odd_shape, None, None, None, None, None, None)

    psf_sum = float(np.sum(array, dtype=np.float64))
    if not np.isfinite(psf_sum) or psf_sum <= 0:
        issues.append(
            PsfPreflightIssue(
                "invalid",
                "nonpositive_sum",
                "PSF sum must be finite and positive.",
            )
        )

    center = np.asarray([(int(size) - 1) / 2 for size in array.shape])
    peak_index = tuple(
        int(index)
        for index in np.unravel_index(int(np.argmax(array)), array.shape)
    )
    peak_offset = float(
        np.linalg.norm(np.asarray(peak_index, dtype=float) - center)
    )
    if peak_offset > PSF_CENTER_OFFSET_WARNING_VOXELS:
        issues.append(
            PsfPreflightIssue(
                "warning",
                "off_center_peak",
                f"PSF peak is {peak_offset:.3g} voxels from the geometric center; "
                "use Prepare / Validate PSF.",
            )
        )

    approximately_normalized = None
    centroid = None
    centroid_offset = None
    edge_mass_fraction = None
    edge_mass_fraction_by_axis = None
    if psf_sum > 0 and not negative:
        approximately_normalized = bool(
            np.isclose(
                psf_sum,
                1.0,
                rtol=PSF_NORMALIZATION_RTOL,
                atol=1e-6,
            )
        )
        if not approximately_normalized:
            issues.append(
                PsfPreflightIssue(
                    "warning",
                    "not_normalized",
                    f"PSF sum is {psf_sum:.6g}, not approximately one. Enable "
                    "Normalize PSF or use Prepare / Validate PSF.",
                )
            )
        centroid = _psf_centroid(array, psf_sum)
        centroid_offset = float(
            np.linalg.norm(np.asarray(centroid, dtype=float) - center)
        )
        if centroid_offset > PSF_CENTER_OFFSET_WARNING_VOXELS:
            issues.append(
                PsfPreflightIssue(
                    "warning",
                    "centroid_offset",
                    "PSF intensity-weighted centroid is "
                    f"{centroid_offset:.3g} voxels from the geometric center; "
                    "use Prepare / Validate PSF.",
                )
            )
        edge_mass_fraction = _psf_edge_mass_fraction(array, psf_sum)
        edge_mass_fraction_by_axis = _psf_edge_mass_fraction_by_axis(
            array,
            psf_sum,
        )
        if edge_mass_fraction > PSF_EDGE_MASS_WARNING_FRACTION:
            labels = (
                axis_labels
                if len(axis_labels) == array.ndim
                else tuple(f"axis {index}" for index in range(array.ndim))
            )
            by_axis = ", ".join(
                f"{label} {fraction:.1%}"
                for label, fraction in zip(
                    labels,
                    edge_mass_fraction_by_axis,
                    strict=True,
                )
            )
            issues.append(
                PsfPreflightIssue(
                    "warning",
                    "edge_mass",
                    f"{edge_mass_fraction:.1%} of PSF intensity is still present "
                    f"on the outermost samples ({by_axis}). This is not intensity "
                    "outside the array; it means the modeled PSF tail may be "
                    "truncated at the current support boundary.",
                )
            )

    return (
        psf_sum,
        approximately_normalized,
        odd_shape,
        peak_index,
        peak_offset,
        centroid,
        centroid_offset,
        edge_mass_fraction,
        edge_mass_fraction_by_axis,
    )


def _psf_centroid(array: np.ndarray, total: float) -> tuple[float, ...]:
    centroid: list[float] = []
    for axis, size in enumerate(array.shape):
        reduce_axes = tuple(index for index in range(array.ndim) if index != axis)
        marginal = np.sum(array, axis=reduce_axes, dtype=np.float64)
        coordinate = np.arange(int(size), dtype=np.float64)
        centroid.append(float(np.dot(coordinate, marginal) / total))
    return tuple(centroid)


def _psf_edge_mass_fraction(array: np.ndarray, total: float) -> float:
    if any(int(size) <= 2 for size in array.shape):
        return 1.0
    interior = array[tuple(slice(1, -1) for _axis in array.shape)]
    interior_sum = float(np.sum(interior, dtype=np.float64))
    return float(np.clip((total - interior_sum) / total, 0.0, 1.0))


def _psf_edge_mass_fraction_by_axis(
    array: np.ndarray,
    total: float,
) -> tuple[float, ...]:
    fractions: list[float] = []
    for axis in range(array.ndim):
        if int(array.shape[axis]) <= 1:
            fractions.append(1.0)
            continue
        lower = np.take(array, 0, axis=axis)
        upper = np.take(array, -1, axis=axis)
        boundary_sum = float(np.sum(lower, dtype=np.float64)) + float(
            np.sum(upper, dtype=np.float64)
        )
        fractions.append(float(np.clip(boundary_sum / total, 0.0, 1.0)))
    return tuple(fractions)


def _psf_centered_mass_within_axis_support(
    array: np.ndarray,
    total: float,
    support_shape: tuple[int, ...],
) -> tuple[float, ...]:
    fractions: list[float] = []
    for axis, requested_size in enumerate(support_shape):
        size = int(array.shape[axis])
        window = min(max(int(requested_size), 0), size)
        if window >= size:
            fractions.append(1.0)
            continue
        if window <= 0:
            fractions.append(0.0)
            continue
        reduce_axes = tuple(index for index in range(array.ndim) if index != axis)
        marginal = np.sum(array, axis=reduce_axes, dtype=np.float64)
        start = (size - window) // 2
        retained = float(np.sum(marginal[start : start + window], dtype=np.float64))
        fractions.append(float(np.clip(retained / total, 0.0, 1.0)))
    return tuple(fractions)


def _preflight_shape(data, state: ImageState | None) -> tuple[int, ...]:
    shape = getattr(data, "shape", None)
    if shape is not None:
        try:
            return tuple(int(size) for size in shape)
        except (TypeError, ValueError):
            pass
    if state is not None:
        return tuple(int(size) for size in state.shape)
    return ()


def _preflight_scan_array(data, *, max_scan_elements: int) -> np.ndarray | None:
    if not isinstance(data, np.ndarray):
        return None
    array = data
    limit = max(int(max_scan_elements), 0)
    if array.size > limit:
        return None
    return array


def _preflight_spatial_ndim(value: int | None) -> int | None:
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return None
    return resolved if resolved in {2, 3} else None


def _preflight_spatial_axis_labels(
    state: ImageState | None,
    spatial_ndim: int | None,
) -> tuple[str, ...]:
    if spatial_ndim not in {2, 3}:
        return ()
    fallback = ("Y", "X") if spatial_ndim == 2 else ("Z", "Y", "X")
    if state is None or len(state.axes) < spatial_ndim:
        return fallback
    labels = tuple(axis.short_label for axis in state.axes[-spatial_ndim:])
    return labels if all(label.strip() for label in labels) else fallback


def _missing_psf_calibration_issue() -> PsfPreflightIssue:
    return PsfPreflightIssue(
        "warning",
        "missing_calibration",
        "Physical image/PSF spacing cannot be compared because calibration is "
        "missing or only index/pixel units are available; no spacing was assumed.",
    )


def _psf_preflight_status(issues: list[PsfPreflightIssue]) -> str:
    severities = {issue.severity for issue in issues}
    if "invalid" in severities:
        return PSF_PREFLIGHT_INVALID
    if "warning" in severities:
        return PSF_PREFLIGHT_WARNING
    if "unknown" in severities:
        return PSF_PREFLIGHT_UNKNOWN
    return PSF_PREFLIGHT_READY


def widefield_nyquist_sampling(
    *,
    wavelength_nm: float,
    numerical_aperture: float,
    refractive_index: float,
    xy_step_um: float,
    z_step_um: float | None,
    spatial_ndim: int,
) -> WidefieldNyquistResult:
    """Estimate conventional-widefield Nyquist critical sample distances.

    The bandwidth-based distances are ``lambda / (4 NA)`` laterally and
    ``lambda / (2 n (1 - cos(alpha)))`` axially, where
    ``alpha = asin(NA / n)``. They match the conventional-widefield model used
    by the Born-Wolf generator; other modalities require different limits.
    """
    resolved_ndim = _preflight_spatial_ndim(spatial_ndim)
    if resolved_ndim is None:
        raise ValueError("Nyquist sampling requires 2D or 3D spatial rank.")
    values = {
        "wavelength": wavelength_nm,
        "numerical aperture": numerical_aperture,
        "refractive index": refractive_index,
        "XY step": xy_step_um,
    }
    if resolved_ndim == 3:
        values["Z step"] = z_step_um
    converted: dict[str, float] = {}
    for name, value in values.items():
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be finite and positive.") from exc
        if not np.isfinite(number) or number <= 0:
            raise ValueError(f"{name} must be finite and positive.")
        converted[name] = number
    wavelength_um = converted["wavelength"] / 1000.0
    na = converted["numerical aperture"]
    refractive_index = converted["refractive index"]
    if na >= refractive_index:
        raise ValueError(
            "Numerical aperture must be smaller than refractive index for the "
            "Born-Wolf widefield sampling estimate."
        )
    alpha = float(np.arcsin(na / refractive_index))
    xy_limit_um = wavelength_um / (4.0 * na)
    z_limit_um = None
    z_met = None
    resolved_z_step = None
    if resolved_ndim == 3:
        resolved_z_step = converted["Z step"]
        denominator = 2.0 * refractive_index * (1.0 - float(np.cos(alpha)))
        z_limit_um = wavelength_um / denominator
        z_met = resolved_z_step <= z_limit_um
    resolved_xy_step = converted["XY step"]
    return WidefieldNyquistResult(
        spatial_ndim=resolved_ndim,
        xy_step_um=resolved_xy_step,
        z_step_um=resolved_z_step,
        xy_limit_um=xy_limit_um,
        z_limit_um=z_limit_um,
        xy_met=resolved_xy_step <= xy_limit_um,
        z_met=z_met,
    )


@dataclass(frozen=True, slots=True)
class ExactFiniteStats:
    """Exact finite-value count and extrema for one array."""

    count: int
    minimum: int | float
    maximum: int | float


def exact_finite_stats(data) -> ExactFiniteStats:
    """Return exact finite count and extrema using bounded temporaries."""
    arr = np.asarray(data)
    integer_data = np.issubdtype(arr.dtype, np.integer)
    count = 0
    minimum: int | float | None = None
    maximum: int | float | None = None
    for values in _iter_finite_numeric_chunks(arr):
        if values.size == 0:
            continue
        count += int(values.size)
        chunk_minimum = (
            int(values.min()) if integer_data else float(values.min())
        )
        chunk_maximum = (
            int(values.max()) if integer_data else float(values.max())
        )
        minimum = chunk_minimum if minimum is None else min(minimum, chunk_minimum)
        maximum = chunk_maximum if maximum is None else max(maximum, chunk_maximum)
    if count == 0:
        return ExactFiniteStats(0, 0.0, 0.0)
    assert minimum is not None and maximum is not None
    return ExactFiniteStats(count, minimum, maximum)


def exact_finite_percentiles(
    data,
    percentiles: tuple[float, ...],
) -> tuple[int | float | Rational, ...] | None:
    """Calculate NumPy-linear percentiles over every finite input value.

    Extrema avoid a full-data copy. Interior percentiles necessarily retain
    all finite values because fixed display bins cannot recover exact order
    statistics. Integer inputs use exact rational interpolation so native
    int64/uint64 levels are never collapsed through float conversion.
    """
    requested = _validated_percentiles(percentiles)
    stats = exact_finite_stats(data)
    if stats.count == 0:
        return None
    if stats.minimum == stats.maximum:
        return tuple(stats.minimum for _value in requested)
    if all(value in {0.0, 100.0} for value in requested):
        return tuple(
            stats.minimum if value == 0.0 else stats.maximum
            for value in requested
        )

    arr = np.asarray(data)
    if arr.dtype == np.dtype(bool):
        raise ValueError(
            "Interior percentiles are undefined for boolean data; convert it "
            "to an explicit numeric dtype first."
        )
    if np.issubdtype(arr.dtype, np.integer) and arr.dtype != np.dtype(bool):
        return exact_integer_percentiles(arr, requested)
    values = np.empty(stats.count, dtype=arr.dtype)
    offset = 0
    for chunk in _iter_finite_numeric_chunks(arr):
        if chunk.size == 0:
            continue
        stop = offset + int(chunk.size)
        values[offset:stop] = chunk
        offset = stop
    if offset != stats.count:
        raise RuntimeError("Finite-value count changed during percentile calculation.")
    result = np.percentile(values, requested, overwrite_input=True)
    return tuple(float(value) for value in np.asarray(result).ravel())


def exact_histogram(
    data,
    *,
    channel_axis: int | None = None,
) -> tuple[
    np.ndarray | None,
    tuple[int | float, int | float] | None,
]:
    """Return an exact display histogram and its numeric range.

    ``channel_axis=None`` always treats the complete array as one scalar-value
    population. Supplying a channel axis returns one row per channel using a
    shared range. No trailing dimension is implicitly interpreted as RGB.
    """
    arr = np.asarray(data)
    if channel_axis is None:
        return _single_histogram(arr)
    axis = _normalized_channel_axis(channel_axis, arr.ndim)
    return _multichannel_histogram(arr, axis)


def exact_generated_layer_contrast_limits(
    data,
) -> tuple[float, float] | None:
    """Return display limits containing every finite value and zero."""
    stats = exact_finite_stats(data)
    if stats.count == 0:
        return None
    low = min(float(stats.minimum), 0.0)
    high = max(float(stats.maximum), 0.0)
    if low == high:
        return (0.0, 1.0)
    return (float(low), float(high))


def provisional_generated_layer_contrast_limits(
    data,
) -> tuple[float, float]:
    """Return scan-free temporary limits while exact extrema are calculated."""
    try:
        dtype = np.dtype(data.dtype)
    except (AttributeError, TypeError):
        dtype = np.asarray(data).dtype
    if dtype == np.dtype(bool):
        return (0.0, 1.0)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        low = min(float(info.min), 0.0)
        high = max(float(info.max), 0.0)
        if low < high:
            return (low, high)
    return (0.0, 1.0)


def label_volumes(labels, spatial_ndim: int) -> np.ndarray:
    """Count each positive label within each independent leading-axis block.

    The final ``spatial_ndim`` axes form one spatial block. Leading axes (for
    example time or channel) are evaluated independently, so the same numeric
    label ID in two leading-axis positions represents two separate objects.
    """
    arr = np.asarray(labels)
    spatial_ndim = _validated_spatial_ndim(arr, spatial_ndim)
    if arr.size == 0:
        return np.array([], dtype=np.int64)
    volumes: list[np.ndarray] = []
    for block in _spatial_blocks(arr, spatial_ndim):
        foreground = np.asarray(block)
        foreground = foreground[foreground > 0]
        if foreground.size == 0:
            continue
        _labels, counts = np.unique(foreground, return_counts=True)
        volumes.append(counts.astype(np.int64, copy=False))
    if not volumes:
        return np.array([], dtype=np.int64)
    return np.concatenate(volumes)


def largest_label_volume(labels, spatial_ndim: int) -> int:
    """Return the largest positive-label count across spatial blocks."""
    volumes = label_volumes(labels, spatial_ndim)
    return int(volumes.max()) if volumes.size else 0


def largest_object_size(
    objects,
    spatial_ndim: int,
    connectivity: str,
) -> int:
    """Return the largest labeled object or boolean connected component.

    Non-boolean inputs retain their existing positive label IDs. Boolean masks
    are connected-component labeled within each spatial block; ``Face
    connected`` uses rank-1 connectivity and ``Full connectivity`` uses the
    full spatial rank.
    """
    arr = np.asarray(objects)
    spatial_ndim = _validated_spatial_ndim(arr, spatial_ndim)
    if arr.size == 0:
        return 0
    if arr.dtype != np.dtype(bool):
        return largest_label_volume(arr, spatial_ndim)

    normalized_connectivity = str(connectivity).strip().lower()
    if normalized_connectivity == "face connected":
        rank = 1
    elif normalized_connectivity in {"full connectivity", "fully connected"}:
        rank = spatial_ndim
    else:
        raise ValueError(
            "Connectivity must be 'Face connected' or 'Full connectivity'."
        )
    structure = ndi.generate_binary_structure(spatial_ndim, rank)
    largest = 0
    for block in _spatial_blocks(arr, spatial_ndim):
        component_labels, count = ndi.label(block, structure=structure)
        if count:
            largest = max(
                largest,
                int(np.bincount(component_labels.ravel())[1:].max()),
            )
    return largest


def _iter_finite_numeric_chunks(data) -> Iterator[np.ndarray]:
    """Yield every finite numeric value without a full-size temporary mask."""
    if data is None:
        return
    try:
        iterator = np.nditer(
            np.asarray(data),
            flags=["buffered", "external_loop", "refs_ok", "zerosize_ok"],
            op_flags=[["readonly"]],
            order="K",
            buffersize=DIAGNOSTIC_CHUNK_ELEMENTS,
        )
    except (TypeError, ValueError):
        return
    for chunk in iterator:
        values = np.asarray(chunk)
        try:
            finite = np.isfinite(values)
        except TypeError:
            return
        if not finite.all():
            values = values[finite]
        yield values


def _normalized_channel_axis(channel_axis: int, ndim: int) -> int:
    if ndim <= 0:
        raise ValueError("A scalar array cannot have a channel axis.")
    axis = int(channel_axis)
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise ValueError(
            f"Channel axis {channel_axis} is outside an array with {ndim} axes."
        )
    return axis


def _validated_percentiles(
    percentiles: tuple[float, ...],
) -> tuple[float, ...]:
    requested: list[float] = []
    for value in percentiles:
        try:
            percentile = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Percentiles must be finite numeric values.") from exc
        if not np.isfinite(percentile):
            raise ValueError("Percentiles must be finite numeric values.")
        if not 0.0 <= percentile <= 100.0:
            raise ValueError("Percentiles must be between 0 and 100.")
        requested.append(percentile)
    return tuple(requested)


def _validated_spatial_ndim(arr: np.ndarray, spatial_ndim: int) -> int:
    if arr.ndim == 0:
        raise ValueError("Label diagnostics require at least one array axis.")
    try:
        resolved = int(spatial_ndim)
    except (TypeError, ValueError) as exc:
        raise ValueError("Spatial dimensionality must be an integer.") from exc
    if resolved < 1 or resolved > arr.ndim:
        raise ValueError(
            f"Spatial dimensionality {resolved} is outside an array with "
            f"{arr.ndim} axes."
        )
    return resolved


def _single_histogram(
    arr: np.ndarray,
) -> tuple[np.ndarray | None, tuple[int | float, int | float] | None]:
    if arr.dtype == np.dtype(bool):
        true_count = int(np.count_nonzero(arr))
        return (
            np.array([arr.size - true_count, true_count], dtype=np.int64),
            (0.0, 1.0),
        )

    stats = exact_finite_stats(arr)
    if stats.count == 0:
        return None, None
    if stats.minimum == stats.maximum:
        return np.array([stats.count], dtype=np.int64), (
            stats.minimum,
            stats.maximum,
        )
    if np.issubdtype(arr.dtype, np.integer):
        return _exact_integer_display_histogram(arr, stats)
    edges, value_range = _display_histogram_edges(stats)
    return _exact_histogram_counts(arr, edges), value_range


def _multichannel_histogram(
    arr: np.ndarray,
    channel_axis: int,
) -> tuple[np.ndarray | None, tuple[int | float, int | float] | None]:
    channels = [
        _axis_index_view(arr, channel_axis, channel)
        for channel in range(arr.shape[channel_axis])
    ]
    channel_stats = [exact_finite_stats(values) for values in channels]
    valid_stats = [stats for stats in channel_stats if stats.count]
    if not valid_stats:
        return None, None

    if arr.dtype == np.dtype(bool):
        counts = []
        for values in channels:
            true_count = int(np.count_nonzero(values))
            counts.append(
                np.array(
                    [values.size - true_count, true_count],
                    dtype=np.int64,
                )
            )
        return np.vstack(counts), (0.0, 1.0)

    combined = ExactFiniteStats(
        sum(stats.count for stats in valid_stats),
        min(stats.minimum for stats in valid_stats),
        max(stats.maximum for stats in valid_stats),
    )
    if combined.minimum == combined.maximum:
        counts = np.array(
            [[stats.count] for stats in channel_stats],
            dtype=np.int64,
        )
        return counts, (combined.minimum, combined.maximum)
    if np.issubdtype(arr.dtype, np.integer):
        histogram_minimum, histogram_maximum, bin_count = (
            _integer_display_histogram_configuration(arr.dtype, combined)
        )
        counts = [
            _exact_integer_display_histogram_counts(
                values,
                histogram_minimum=histogram_minimum,
                histogram_maximum=histogram_maximum,
                bin_count=bin_count,
            )
            for values in channels
        ]
        return np.vstack(counts), (histogram_minimum, histogram_maximum)

    edges, value_range = _display_histogram_edges(combined)
    counts = [
        (
            _exact_histogram_counts(values, edges)
            if stats.count
            else np.zeros(edges.size - 1, dtype=np.int64)
        )
        for values, stats in zip(channels, channel_stats, strict=True)
    ]
    return np.vstack(counts), value_range


def _exact_integer_display_histogram(
    data,
    stats: ExactFiniteStats,
) -> tuple[np.ndarray, tuple[int, int]]:
    histogram_minimum, histogram_maximum, bin_count = (
        _integer_display_histogram_configuration(np.asarray(data).dtype, stats)
    )
    counts = _exact_integer_display_histogram_counts(
        data,
        histogram_minimum=histogram_minimum,
        histogram_maximum=histogram_maximum,
        bin_count=bin_count,
    )
    return counts, (histogram_minimum, histogram_maximum)


def _integer_display_histogram_configuration(
    dtype: np.dtype,
    stats: ExactFiniteStats,
) -> tuple[int, int, int]:
    del dtype
    minimum = int(stats.minimum)
    maximum = int(stats.maximum)
    if 0 <= minimum and maximum <= 255:
        return 0, 255, 256
    level_span = maximum - minimum + 1
    return minimum, maximum, min(DISPLAY_HISTOGRAM_BINS, level_span)


def _exact_integer_display_histogram_counts(
    data,
    *,
    histogram_minimum: int,
    histogram_maximum: int,
    bin_count: int,
) -> np.ndarray:
    """Count integer levels exactly after subtracting a Python-int offset."""
    level_span = histogram_maximum - histogram_minimum + 1
    counts = np.zeros(bin_count, dtype=np.int64)
    if level_span <= 65_536:
        native_counts = np.zeros(level_span, dtype=np.int64)
        for values in _iter_finite_numeric_chunks(data):
            if values.size == 0:
                continue
            levels, level_counts = np.unique(values, return_counts=True)
            indices = np.fromiter(
                (int(level) - histogram_minimum for level in levels),
                dtype=np.intp,
                count=levels.size,
            )
            native_counts[indices] += level_counts.astype(np.int64, copy=False)
        if bin_count == level_span:
            return native_counts
        boundaries = (
            np.arange(bin_count + 1, dtype=np.int64) * level_span // bin_count
        )
        return np.asarray(
            [
                native_counts[start:stop].sum(dtype=np.int64)
                for start, stop in zip(
                    boundaries[:-1],
                    boundaries[1:],
                    strict=True,
                )
            ],
            dtype=np.int64,
        )

    for values in _iter_finite_numeric_chunks(data):
        if values.size == 0:
            continue
        levels, level_counts = np.unique(values, return_counts=True)
        indices = np.fromiter(
            (
                min(
                    ((int(level) - histogram_minimum) * bin_count) // level_span,
                    bin_count - 1,
                )
                for level in levels
            ),
            dtype=np.intp,
            count=levels.size,
        )
        np.add.at(counts, indices, level_counts.astype(np.int64, copy=False))
    return counts


def _display_histogram_edges(
    stats: ExactFiniteStats,
) -> tuple[np.ndarray, tuple[float, float]]:
    return (
        np.linspace(
            stats.minimum,
            stats.maximum,
            DISPLAY_HISTOGRAM_BINS + 1,
        ),
        (stats.minimum, stats.maximum),
    )


def _exact_histogram_counts(data, edges: np.ndarray) -> np.ndarray:
    counts = np.zeros(int(edges.size) - 1, dtype=np.int64)
    for values in _iter_finite_numeric_chunks(data):
        if values.size:
            counts += np.histogram(values, bins=edges)[0]
    return counts


def _axis_index_view(arr: np.ndarray, axis: int, index: int) -> np.ndarray:
    selection = [slice(None)] * arr.ndim
    selection[int(axis)] = int(index)
    return arr[tuple(selection)]


def _spatial_blocks(arr: np.ndarray, spatial_ndim: int) -> Iterator[np.ndarray]:
    leading_shape = arr.shape[: arr.ndim - spatial_ndim]
    if leading_shape:
        for index in np.ndindex(leading_shape):
            yield arr[index]
        return
    yield arr


__all__ = [
    "DIAGNOSTIC_CHUNK_ELEMENTS",
    "DISPLAY_HISTOGRAM_BINS",
    "ExactFiniteStats",
    "exact_finite_percentiles",
    "exact_finite_stats",
    "exact_generated_layer_contrast_limits",
    "exact_histogram",
    "label_volumes",
    "largest_label_volume",
    "largest_object_size",
    "provisional_generated_layer_contrast_limits",
]
