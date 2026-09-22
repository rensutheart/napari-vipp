"""Deterministic additive phantoms and shared quality metrics for background tests.

This module is benchmark support, not a new image-processing operation. The
foreground and smooth background are known before noise and input quantization.
All methods are compared to the same continuous truth and the same threshold;
threshold-mask agreement is a sensitivity check, not biological validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage as ndi

PATTERNS = ("mixed", "sparse", "dense", "broad", "ramp", "vignette")
_CHUNK = 262_144


@dataclass(frozen=True)
class BackgroundFixture:
    input: np.ndarray
    background_truth: np.ndarray
    signal_truth: np.ndarray
    object_labels: np.ndarray
    description: str
    pattern: str
    intensity_scale: str
    light_background: bool
    seed: int
    noise_sigma: float
    intensity_ceiling: float
    threshold: float

    def to_json_metadata(self) -> dict[str, Any]:
        """Small, JSON-safe evidence; arrays and private paths are excluded."""
        return {
            "schema": "vipp-background-phantom-v1",
            "shape": list(self.input.shape),
            "axes": "YX" if self.input.ndim == 2 else "ZYX",
            "dtype": str(self.input.dtype),
            "pattern": self.pattern,
            "intensity_scale": self.intensity_scale,
            "intensity_ceiling": self.intensity_ceiling,
            "light_background": self.light_background,
            "seed": self.seed,
            "noise_sigma": self.noise_sigma,
            "input_range": [float(self.input.min()), float(self.input.max())],
            "signal_range": [0.0, float(self.signal_truth.max())],
            "generated_object_count": int(
                np.count_nonzero(np.unique(self.object_labels))
            ),
            "fixed_detection_threshold": self.threshold,
            "threshold_rule": "15% of the known noiseless foreground maximum",
            "threshold_interpretation": (
                "One fixed threshold for all methods. Connected threshold regions "
                "are an algorithm-sensitivity diagnostic, not biological objects."
            ),
            "truth_policy": (
                "Continuous additive foreground/background before Gaussian noise "
                "and input rounding. Truth errors include a common noise and "
                "quantization floor; no method-specific normalization or fitting. "
                "Corrected foreground truth is positive for either background polarity."
            ),
            "description": self.description,
        }


def _add_object(
    signal: np.ndarray,
    labels: np.ndarray,
    center: tuple[float, ...],
    radii: tuple[float, ...],
    amplitude: float,
    object_id: int,
) -> None:
    """Bounded ellipsoidal ROI: no full-volume coordinate meshes."""
    slices = tuple(
        slice(max(0, int(np.floor(c - r))), min(n, int(np.ceil(c + r)) + 1))
        for n, c, r in zip(signal.shape, center, radii, strict=True)
    )
    shape = tuple(s.stop - s.start for s in slices)
    distance = np.zeros(shape, dtype=np.float32)
    for axis, (part, c, r) in enumerate(zip(slices, center, radii, strict=True)):
        coordinates = (np.arange(part.start, part.stop, dtype=np.float32) - c) / r
        view_shape = [1] * signal.ndim
        view_shape[axis] = len(coordinates)
        distance += (coordinates * coordinates).reshape(view_shape)
    mask = distance <= 1
    profile = np.exp(-2.5 * distance) * np.float32(amplitude)
    profile[~mask] = 0
    signal[slices] += profile
    # Overlap is intentionally represented by a single owner, not counted twice.
    labels[slices][mask] = object_id


def create_fixture(
    shape: tuple[int, ...],
    dtype: str | np.dtype,
    pattern: str = "mixed",
    intensity_scale: str = "native",
    light_background: bool = False,
    seed: int = 20260922,
) -> BackgroundFixture:
    """Create 2D or genuinely varying 3D data without mixing spatial dimensions.

    ``native`` means microscopy counts (ceiling 60,000), including for float32.
    ``normalized`` gives float32 values in [0, 1]. ``counts`` is an explicit alias
    for native units. No normalization is performed within benchmark methods.
    Arrays are read-only. Generation uses float32 arrays and bounded noise chunks.
    """
    shape = tuple(int(length) for length in shape)
    dtype = np.dtype(dtype)
    if len(shape) not in (2, 3) or any(length < 4 for length in shape):
        raise ValueError("Use a YX or ZYX shape with each dimension at least 4.")
    if dtype not in (np.dtype("uint16"), np.dtype("float32")):
        raise ValueError("Benchmark fixtures support uint16 and float32.")
    if pattern not in PATTERNS:
        raise ValueError(f"Unknown pattern {pattern!r}; choose from {PATTERNS}.")
    if intensity_scale not in ("native", "counts", "normalized"):
        raise ValueError("intensity_scale must be native, counts, or normalized.")
    if intensity_scale == "normalized" and dtype != np.dtype("float32"):
        raise ValueError("Normalized fixtures require float32 to preserve signal.")
    ceiling = 1.0 if intensity_scale == "normalized" else 60_000.0
    height, width = shape[-2:]
    y = np.linspace(-1, 1, height, dtype=np.float32)[:, None]
    x = np.linspace(-1, 1, width, dtype=np.float32)[None, :]
    background = np.empty(shape, dtype=np.float32)
    planes = background.reshape((-1, height, width))
    for z_index, plane in enumerate(planes):
        z = 0.0 if len(planes) == 1 else 2 * z_index / (len(planes) - 1) - 1
        plane[:] = 0.115 + 0.035 * x + 0.025 * y + 0.012 * z
        if pattern != "ramp":
            plane += 0.08 * np.exp(-1.5 * (x * x + y * y + 0.35 * z * z))
        if pattern not in ("ramp", "vignette"):
            plane += 0.012 * np.sin(3 * x + z) * np.cos(2 * y - z)
    signal = np.zeros(shape, dtype=np.float32)
    labels = np.zeros(shape, dtype=np.uint16)
    ndim = len(shape)
    objects: list[tuple[tuple[float, ...], tuple[float, ...], float]] = []

    def add(yx, yx_radii, amplitude, z=0.5, zr=0.22):
        center = tuple(c * (n - 1) for c, n in zip(yx, shape[-2:], strict=True))
        radii = tuple(
            max(1.2, r * n) for r, n in zip(yx_radii, shape[-2:], strict=True)
        )
        if ndim == 3:
            center = (z * (shape[0] - 1), *center)
            radii = (max(1.2, zr * shape[0]), *radii)
        objects.append((center, radii, amplitude))

    # Bright, dim and edge-touching structures occur in every pattern.
    add((0.21, 0.22), (0.028, 0.022), 0.36, z=0.25)
    add((0.36, 0.71), (0.038, 0.025), 0.21, z=0.7)
    add((0.69, 0.28), (0.025, 0.034), 0.055, z=0.55)
    add((0.02, 0.64), (0.065, 0.045), 0.30, z=0.05, zr=0.3)
    add((0.84, 0.99), (0.036, 0.067), 0.24, z=0.92)
    if pattern in ("mixed", "broad"):
        add((0.73, 0.70), (0.16, 0.19), 0.22, z=0.52, zr=0.33)
    if pattern in ("mixed", "dense"):
        for index, (cy, cx) in enumerate(
            (cy, cx)
            for cy in np.linspace(0.37, 0.64, 5)
            for cx in np.linspace(0.10, 0.43, 6)
        ):
            add(
                (cy, cx),
                (0.031, 0.032),
                0.10 + 0.012 * (index % 7),
                z=0.3 + 0.1 * (index % 5),
                zr=0.21,
            )
    for index, (center, radii, amplitude) in enumerate(objects, start=1):
        _add_object(signal, labels, center, radii, amplitude, index)
    signal *= np.float32(ceiling)
    background *= np.float32(ceiling)
    values = background + signal
    noise_sigma = ceiling * 0.0025
    rng = np.random.default_rng(seed)
    flat = values.ravel()
    for offset in range(0, flat.size, _CHUNK):
        block = flat[offset : offset + _CHUNK]
        block += rng.standard_normal(block.size, dtype=np.float32) * np.float32(
            noise_sigma
        )
    if light_background:
        values = np.float32(ceiling) - values
        background = np.float32(ceiling) - background
    np.clip(values, 0, ceiling, out=values)
    if np.issubdtype(dtype, np.integer):
        np.rint(values, out=values)
    image = values.astype(dtype, copy=False)
    for array in (image, background, signal, labels):
        array.flags.writeable = False
    return BackgroundFixture(
        input=image,
        background_truth=background,
        signal_truth=signal,
        object_labels=labels,
        description=(
            f"{pattern} additive phantom: smooth ramp/vignetting background; "
            "bright, dim and edge-touching ellipsoids; mixed/dense includes crowded "
            "objects and mixed/broad includes an extended foreground feature. "
            + (
                "True ZYX ellipsoids and Z-varying background."
                if ndim == 3
                else "YX spatial image."
            )
        ),
        pattern=pattern,
        intensity_scale=intensity_scale,
        light_background=bool(light_background),
        seed=int(seed),
        noise_sigma=noise_sigma,
        intensity_ceiling=ceiling,
        threshold=float(signal.max()) * 0.15,
    )


def numerical_comparison(
    reference: np.ndarray, candidate: np.ndarray
) -> dict[str, Any]:
    """Conversion-safe errors with bounded float64 temporaries and finite handling."""
    reference, candidate = np.asarray(reference), np.asarray(candidate)
    if reference.shape != candidate.shape:
        raise ValueError("Reference and candidate shapes must match.")
    count = mismatch = finite_count = nonfinite_mismatch = 0
    absolute = square = maximum = mx = my = sxx = syy = sxy = 0.0
    low, high = float("inf"), float("-inf")
    # flat slicing also supports non-contiguous source arrays without copying all.
    for offset in range(0, reference.size, _CHUNK):
        x = np.asarray(reference.flat[offset : offset + _CHUNK], dtype=np.float64)
        y = np.asarray(candidate.flat[offset : offset + _CHUNK], dtype=np.float64)
        finite = np.isfinite(x) & np.isfinite(y)
        equal = (x == y) | (np.isnan(x) & np.isnan(y))
        mismatch += int(np.count_nonzero(~equal))
        nonfinite_mismatch += int(np.count_nonzero(~equal & ~finite))
        count += x.size
        x, y = x[finite], y[finite]
        if not x.size:
            continue
        previous_count = finite_count
        finite_count += x.size
        difference = y - x
        absolute += float(np.abs(difference).sum())
        square += float(np.dot(difference, difference))
        maximum = max(maximum, float(np.abs(difference).max()))
        low, high = min(low, float(x.min())), max(high, float(x.max()))
        # Merge centered moments rather than subtracting two large sums; this
        # remains stable for high-valued images with very little variation.
        batch_mx, batch_my = float(x.mean()), float(y.mean())
        centered_x, centered_y = x - batch_mx, y - batch_my
        dx, dy = batch_mx - mx, batch_my - my
        weight = previous_count * x.size / finite_count
        sxx += float(np.dot(centered_x, centered_x)) + dx * dx * weight
        syy += float(np.dot(centered_y, centered_y)) + dy * dy * weight
        sxy += float(np.dot(centered_x, centered_y)) + dx * dy * weight
        mx += dx * x.size / finite_count
        my += dy * x.size / finite_count
    rmse = float(np.sqrt(square / finite_count)) if finite_count else None
    dynamic_range = high - low if finite_count else None
    correlation = None
    if sxx > 0 and syy > 0:
        correlation = float(np.clip(sxy / np.sqrt(sxx * syy), -1, 1))
    return {
        "exact_equal_nan": mismatch == 0,
        "dtype_equal": reference.dtype == candidate.dtype,
        "mismatch_fraction": mismatch / count if count else 0.0,
        "finite_pair_count": finite_count,
        "nonfinite_pair_count": count - finite_count,
        "nonfinite_mismatch_count": nonfinite_mismatch,
        "mae": absolute / finite_count if finite_count else None,
        "rmse": rmse,
        "max_absolute_error": maximum if finite_count else None,
        "reference_finite_range": dynamic_range,
        "range_normalized_rmse": rmse / dynamic_range
        if dynamic_range and rmse is not None
        else None,
        "pearson_correlation": correlation,
    }


def quality_against_truth(
    fixture: BackgroundFixture, corrected: np.ndarray
) -> dict[str, Any]:
    """Signal retention, background residual, and fixed-threshold sensitivity."""
    corrected = np.asarray(corrected)
    if corrected.shape != fixture.input.shape:
        raise ValueError("Corrected output must match the fixture shape.")
    foreground_n = background_n = excluded = 0
    foreground_bias = foreground_truth_sum = foreground_output_sum = 0.0
    background_sum = background_absolute = background_square = 0.0
    label_count = int(fixture.object_labels.max()) + 1
    object_valid_counts = np.zeros(label_count, dtype=np.int64)
    object_total_counts = np.zeros(label_count, dtype=np.int64)
    object_truth_sums = np.zeros(label_count, dtype=np.float64)
    object_output_sums = np.zeros(label_count, dtype=np.float64)
    for offset in range(0, corrected.size, _CHUNK):
        output = np.asarray(corrected.flat[offset : offset + _CHUNK], dtype=np.float64)
        truth = np.asarray(
            fixture.signal_truth.flat[offset : offset + _CHUNK], dtype=np.float64
        )
        finite = np.isfinite(output)
        excluded += int(np.count_nonzero(~finite))
        fg, bg = finite & (truth > 0), finite & (truth == 0)
        foreground_n += int(fg.sum())
        background_n += int(bg.sum())
        foreground_bias += float((output[fg] - truth[fg]).sum())
        foreground_output_sum += float(output[fg].sum())
        foreground_truth_sum += float(truth[fg].sum())
        background_sum += float(output[bg].sum())
        background_absolute += float(np.abs(output[bg]).sum())
        background_square += float(np.dot(output[bg], output[bg]))
        labels = fixture.object_labels.flat[offset : offset + _CHUNK]
        object_total_counts += np.bincount(labels, minlength=label_count)
        object_valid_counts += np.bincount(labels[fg], minlength=label_count)
        object_truth_sums += np.bincount(
            labels[fg], weights=truth[fg], minlength=label_count
        )
        object_output_sums += np.bincount(
            labels[fg], weights=output[fg], minlength=label_count
        )
    truth_mask = fixture.signal_truth >= fixture.threshold
    output_mask = np.isfinite(corrected) & (corrected >= fixture.threshold)
    intersection = int(np.count_nonzero(truth_mask & output_mask))
    denominator = int(truth_mask.sum()) + int(output_mask.sum())
    # Face connectivity is fixed (4-connected YX / 6-connected ZYX), matching
    # dimensions of the fixture even if a filtering method works slice-wise.
    component_labels, truth_count = ndi.label(truth_mask)
    del component_labels
    component_labels, output_count = ndi.label(output_mask)
    del component_labels
    per_object = []
    for label_id in range(1, label_count):
        if not object_total_counts[label_id]:
            continue
        truth_sum = float(object_truth_sums[label_id])
        output_sum = float(object_output_sums[label_id])
        per_object.append(
            {
                "label": label_id,
                "owned_voxel_count": int(object_total_counts[label_id]),
                "finite_voxel_count": int(object_valid_counts[label_id]),
                "truth_signal_sum": truth_sum,
                "output_signal_sum": output_sum,
                "signal_retention_fraction": output_sum / truth_sum
                if truth_sum
                else None,
            }
        )
    valid_objects = [
        obj for obj in per_object if obj["signal_retention_fraction"] is not None
    ]
    retentions = [obj["signal_retention_fraction"] for obj in valid_objects]
    worst_object = (
        min(valid_objects, key=lambda obj: obj["signal_retention_fraction"])
        if valid_objects
        else None
    )
    return {
        "nonfinite_output_count": excluded,
        "foreground_voxel_count": foreground_n,
        "background_voxel_count": background_n,
        "foreground_mean_bias": foreground_bias / foreground_n
        if foreground_n
        else None,
        "signal_retention_fraction": foreground_output_sum / foreground_truth_sum
        if foreground_truth_sum
        else None,
        "background_residual_mean": background_sum / background_n
        if background_n
        else None,
        "background_residual_mae": background_absolute / background_n
        if background_n
        else None,
        "background_residual_rmse": float(np.sqrt(background_square / background_n))
        if background_n
        else None,
        "fixed_threshold": fixture.threshold,
        "fixed_threshold_dice": 2 * intersection / denominator if denominator else 1.0,
        "truth_threshold_region_count": int(truth_count),
        "output_threshold_region_count": int(output_count),
        "threshold_connectivity": "face-connected in fixture spatial dimensions",
        "per_object_signal_retention": per_object,
        "object_retention_min": min(retentions) if retentions else None,
        "object_retention_median": float(np.median(retentions)) if retentions else None,
        "worst_retained_object_label": worst_object["label"] if worst_object else None,
        "object_retention_interpretation": (
            "Fixed generated-label regions, not detected objects. Each voxel is "
            "owned by the last generated ellipsoid covering it; overlapping "
            "signals are not deblended. Sums use finite output voxels only; "
            "finite/owned counts expose exclusions. Retention above 1 can mean "
            "background remains, not improved signal recovery."
        ),
    }


def quality_comparison(
    fixture: BackgroundFixture,
    reference: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, Any]:
    return {
        "numerical": numerical_comparison(reference, candidate),
        "reference_vs_truth": quality_against_truth(fixture, reference),
        "candidate_vs_truth": quality_against_truth(fixture, candidate),
        "interpretation": (
            "Identical outputs need not recover the true signal. Truth metrics "
            "include input noise/quantization; one truth-derived threshold is "
            "shared, and region counts are not a biological accuracy claim."
        ),
    }
