"""Reproducible scientific validation for VIPP Richardson-Lucy TV.

This script deliberately keeps all comparison variants outside production code.
It builds deterministic 2D and 3D microscopy phantoms, generates Born-Wolf PSFs,
applies controlled Poisson/Gaussian noise, and writes metric tables for the
production implementation plus boundary, initialization, and TV-discretization
comparisons.

Run from the repository root::

    python scripts/validate_rl_tv_phantoms.py

The default output directory is ``docs/validation/rl-tv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import skimage
from scipy import signal
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from napari_vipp.core.grid import validate_psf_image_states
from napari_vipp.core.metadata import AxisMetadata, image_state_from_array
from napari_vipp.core.operations import (
    _tv_divergence,
    born_wolf_psf,
    richardson_lucy_deconvolution,
    richardson_lucy_tv_deconvolution,
)

SEED_2D = 20260715
SEED_3D = 20260716
DEFAULT_ITERATIONS = 25
DEFAULT_TV_REGULARIZATION = 0.002
DEFAULT_TV_EPSILON = 1e-6
DEFAULT_FILTER_EPSILON = 1e-12
DEFAULT_DENOMINATOR_FLOOR = 0.05
EXAMPLE_WORKFLOW_SETTINGS = {
    "2D": {"iterations": 18, "tv_regularization": 0.012, "denominator_floor": 0.15},
    "3D": {"iterations": 8, "tv_regularization": 0.008, "denominator_floor": 0.15},
}


@dataclass(frozen=True)
class Phantom:
    """Ground truth, feature masks, optical PSF, and physical sampling."""

    name: str
    truth: np.ndarray
    masks: dict[str, np.ndarray]
    psf: np.ndarray
    spacing_um: tuple[float, ...]
    spatial_mode: str
    seed: int


@dataclass(frozen=True)
class VariantDiagnostics:
    """Numerical guard activity recorded by a comparison-only RL-TV run."""

    minimum_raw_denominator: float
    maximum_floor_fraction: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/validation/rl-tv"),
        help="Directory for CSV/JSON evidence (default: docs/validation/rl-tv).",
    )
    parser.add_argument(
        "--dimensions",
        nargs="+",
        choices=("2d", "3d"),
        default=("2d", "3d"),
        help="Phantom dimensions to analyze.",
    )
    return parser.parse_args()


def _born_wolf(
    shape: tuple[int, ...],
    *,
    pixel_size_xy_um: float,
    z_step_um: float,
    xy_size: int,
    z_size: int,
) -> np.ndarray:
    ndim = len(shape)
    return born_wolf_psf(
        np.zeros(shape, dtype=np.float32),
        spatial_mode="2D YX" if ndim == 2 else "3D ZYX",
        auto_parameters=False,
        wavelength_nm=520.0,
        numerical_aperture=1.2,
        refractive_index=1.33,
        pixel_size_xy_um=pixel_size_xy_um,
        z_step_um=z_step_um,
        xy_size=xy_size,
        z_size=z_size,
        pupil_samples=64,
        normalize=True,
    )


def make_phantom_2d() -> Phantom:
    """Return a 2D phantom with points, lines, dim detail, and border signal."""
    shape = (64, 64)
    truth = np.zeros(shape, dtype=np.float32)
    masks = {name: np.zeros(shape, dtype=bool) for name in _feature_names()}

    for y, x, value in ((18, 18, 1.0), (46, 47, 0.85)):
        truth[y, x] = value
        masks["points"][y, x] = True

    truth[16, 30:53] = 0.65
    masks["thin_line"][16, 30:53] = True

    truth[35, 14:35] = 1.0
    masks["bright_structure"][35, 14:35] = True
    truth[38, 14:35] = 0.20
    masks["dim_structure"][38, 14:35] = True

    truth[1:5, 2:9] = 0.75
    masks["border_structure"][1:5, 2:9] = True

    psf = _born_wolf(
        shape,
        pixel_size_xy_um=0.10,
        z_step_um=0.30,
        xy_size=15,
        z_size=1,
    )
    return Phantom("2D", truth, masks, psf, (0.10, 0.10), "2D YX", SEED_2D)


def make_phantom_3d() -> Phantom:
    """Return a small anisotropic 3D phantom with the same feature classes."""
    shape = (17, 48, 48)
    truth = np.zeros(shape, dtype=np.float32)
    masks = {name: np.zeros(shape, dtype=bool) for name in _feature_names()}

    for z, y, x, value in ((8, 14, 14, 1.0), (10, 37, 35, 0.85)):
        truth[z, y, x] = value
        masks["points"][z, y, x] = True

    truth[8, 25, 12:37] = 0.65
    masks["thin_line"][8, 25, 12:37] = True

    truth[8, 31, 12:37] = 1.0
    masks["bright_structure"][8, 31, 12:37] = True
    truth[9, 34, 12:37] = 0.18
    masks["dim_structure"][9, 34, 12:37] = True

    truth[1:4, 1:5, 2:9] = 0.75
    masks["border_structure"][1:4, 1:5, 2:9] = True

    psf = _born_wolf(
        shape,
        pixel_size_xy_um=0.10,
        z_step_um=0.30,
        xy_size=15,
        z_size=7,
    )
    return Phantom(
        "3D",
        truth,
        masks,
        psf,
        (0.30, 0.10, 0.10),
        "3D ZYX",
        SEED_3D,
    )


def _feature_names() -> tuple[str, ...]:
    return (
        "points",
        "thin_line",
        "bright_structure",
        "dim_structure",
        "border_structure",
    )


def convolve_boundary(
    values: np.ndarray,
    kernel: np.ndarray,
    *,
    boundary: str,
) -> np.ndarray:
    """Convolve with production zero-extension or reflect-pad/crop semantics."""
    if boundary == "zero":
        return signal.convolve(values, kernel, mode="same").astype(
            np.float32,
            copy=False,
        )
    if boundary != "reflect":
        raise ValueError(f"Unknown boundary mode: {boundary!r}")
    pad_width = tuple((size // 2, size // 2) for size in kernel.shape)
    padded = np.pad(values, pad_width, mode="reflect")
    return signal.convolve(padded, kernel, mode="valid").astype(
        np.float32,
        copy=False,
    )


def add_controlled_noise(
    values: np.ndarray,
    *,
    seed: int,
    photons_at_peak: float,
    gaussian_fraction: float = 0.004,
) -> np.ndarray:
    """Add seeded Poisson and Gaussian noise to a non-negative observation."""
    rng = np.random.default_rng(seed)
    peak = float(np.max(values))
    if peak <= 0:
        return np.zeros_like(values, dtype=np.float32)
    photons = max(float(photons_at_peak), 1.0)
    expected_counts = np.maximum(values, 0.0) / peak * photons
    poisson = rng.poisson(expected_counts).astype(np.float32) / np.float32(photons)
    noisy = poisson * np.float32(peak)
    noisy += rng.normal(0.0, gaussian_fraction * peak, size=values.shape).astype(
        np.float32
    )
    return np.maximum(noisy, 0.0).astype(np.float32, copy=False)


def observed_image(
    phantom: Phantom,
    *,
    photons_at_peak: float = 100.0,
) -> np.ndarray:
    """Generate a physically extended observation with reflect boundary blur."""
    blurred = convolve_boundary(phantom.truth, phantom.psf, boundary="reflect")
    return add_controlled_noise(
        blurred,
        seed=phantom.seed + int(photons_at_peak),
        photons_at_peak=photons_at_peak,
    )


def forward_backward_tv_divergence(
    values: np.ndarray,
    *,
    epsilon: float,
) -> np.ndarray:
    """Comparison discretization using forward differences and adjoint divergence."""
    data = np.asarray(values, dtype=np.float32)
    gradients: list[np.ndarray] = []
    for axis in range(data.ndim):
        difference = np.roll(data, -1, axis=axis) - data
        terminal = [slice(None)] * data.ndim
        terminal[axis] = -1
        difference[tuple(terminal)] = 0.0
        gradients.append(difference)
    norm = np.sqrt(
        np.sum(
            np.stack([gradient * gradient for gradient in gradients], axis=0),
            axis=0,
            dtype=np.float32,
        )
        + np.float32(epsilon) ** 2
    )
    divergence = np.zeros(data.shape, dtype=np.float32)
    for axis, gradient in enumerate(gradients):
        component = gradient / norm
        divergence += component - np.roll(component, 1, axis=axis)
    return np.nan_to_num(divergence, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32,
        copy=False,
    )


def rl_tv_comparison_variant(
    observed: np.ndarray,
    psf: np.ndarray,
    *,
    iterations: int,
    tv_regularization: float = DEFAULT_TV_REGULARIZATION,
    tv_epsilon: float = DEFAULT_TV_EPSILON,
    filter_epsilon: float = DEFAULT_FILTER_EPSILON,
    denominator_floor: float = DEFAULT_DENOMINATOR_FLOOR,
    initialization: str = "constant_0.5",
    boundary: str = "zero",
    tv_discretization: str = "production_central",
) -> tuple[np.ndarray, VariantDiagnostics]:
    """Run an instrumented comparison without changing production operations."""
    values = np.asarray(observed, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.maximum(values, 0.0)
    output_scale = float(np.max(values))
    if output_scale <= 0:
        return (
            np.zeros_like(values),
            VariantDiagnostics(1.0, 0.0),
        )
    values = values / np.float32(output_scale)

    kernel = np.asarray(psf, dtype=np.float32)
    kernel = np.maximum(np.nan_to_num(kernel), 0.0)
    kernel /= np.float32(np.sum(kernel, dtype=np.float64))

    if initialization == "constant_0.5":
        estimate = np.full(values.shape, 0.5, dtype=np.float32)
    elif initialization == "observed":
        estimate = values.copy()
    elif initialization == "positive_mean":
        positive = values[values > 0]
        mean = float(np.mean(positive)) if positive.size else 0.5
        estimate = np.full(values.shape, mean, dtype=np.float32)
    else:
        raise ValueError(f"Unknown initialization: {initialization!r}")

    mirror = np.flip(kernel)
    minimum_raw_denominator = float("inf")
    maximum_floor_fraction = 0.0
    for _ in range(max(int(iterations), 1)):
        blurred = convolve_boundary(estimate, kernel, boundary=boundary) + np.float32(
            1e-12
        )
        if filter_epsilon > 0:
            ratio = np.where(blurred < filter_epsilon, 0.0, values / blurred)
        else:
            ratio = values / blurred
        correction = convolve_boundary(ratio, mirror, boundary=boundary)
        if tv_regularization > 0:
            if tv_discretization == "production_central":
                divergence = _tv_divergence(estimate, epsilon=tv_epsilon)
            elif tv_discretization == "forward_backward":
                divergence = forward_backward_tv_divergence(
                    estimate,
                    epsilon=tv_epsilon,
                )
            else:
                raise ValueError(f"Unknown TV discretization: {tv_discretization!r}")
            raw_denominator = 1.0 - np.float32(tv_regularization) * divergence
            minimum_raw_denominator = min(
                minimum_raw_denominator,
                float(np.min(raw_denominator)),
            )
            maximum_floor_fraction = max(
                maximum_floor_fraction,
                float(np.mean(raw_denominator < denominator_floor)),
            )
            denominator = np.maximum(raw_denominator, denominator_floor)
            estimate = estimate * correction / denominator
        else:
            estimate *= correction
        estimate = np.maximum(
            np.nan_to_num(estimate, nan=0.0, posinf=0.0, neginf=0.0),
            0.0,
        ).astype(np.float32, copy=False)

    if not np.isfinite(minimum_raw_denominator):
        minimum_raw_denominator = 1.0
    restored = estimate * np.float32(output_scale)
    return restored, VariantDiagnostics(
        minimum_raw_denominator=minimum_raw_denominator,
        maximum_floor_fraction=maximum_floor_fraction,
    )


def total_variation(values: np.ndarray) -> float:
    gradients = np.gradient(np.asarray(values, dtype=np.float64))
    magnitude = np.sqrt(
        np.sum(np.stack([gradient**2 for gradient in gradients]), axis=0)
    )
    return float(np.sum(magnitude))


def _recovery_ratio(
    restored: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
) -> float:
    denominator = float(np.sum(truth[mask], dtype=np.float64))
    if denominator <= 0:
        return float("nan")
    return float(np.sum(restored[mask], dtype=np.float64) / denominator)


def calculate_metrics(
    restored: np.ndarray,
    phantom: Phantom,
) -> dict[str, float]:
    truth = phantom.truth.astype(np.float64, copy=False)
    candidate = np.asarray(restored, dtype=np.float64)
    data_range = max(float(np.max(truth) - np.min(truth)), 1e-12)
    mse = float(np.mean((candidate - truth) ** 2))
    psnr = float(peak_signal_noise_ratio(truth, candidate, data_range=data_range))
    win_size = min(7, *(size if size % 2 else size - 1 for size in truth.shape))
    ssim = float(
        structural_similarity(
            truth,
            candidate,
            data_range=data_range,
            win_size=max(win_size, 3),
        )
    )
    border_width = tuple(max(size // 2, 1) for size in phantom.psf.shape)
    border_mask = np.zeros(truth.shape, dtype=bool)
    for axis, width in enumerate(border_width):
        leading = [slice(None)] * truth.ndim
        leading[axis] = slice(0, width)
        border_mask[tuple(leading)] = True
        trailing = [slice(None)] * truth.ndim
        trailing[axis] = slice(-width, None)
        border_mask[tuple(trailing)] = True
    border_mse = float(np.mean((candidate[border_mask] - truth[border_mask]) ** 2))
    truth_flux = float(np.sum(truth, dtype=np.float64))
    metrics = {
        "mse": mse,
        "psnr_db": psnr,
        "ssim": ssim,
        "total_variation": total_variation(candidate),
        "flux_ratio": float(np.sum(candidate, dtype=np.float64) / truth_flux),
        "border_mse": border_mse,
    }
    for name, mask in phantom.masks.items():
        metrics[f"{name}_recovery"] = _recovery_ratio(candidate, truth, mask)
    metrics["dim_structure_loss_fraction"] = max(
        0.0,
        1.0 - metrics["dim_structure_recovery"],
    )
    return metrics


def _production_tv(
    observed: np.ndarray,
    phantom: Phantom,
    *,
    iterations: int = DEFAULT_ITERATIONS,
    tv_regularization: float = DEFAULT_TV_REGULARIZATION,
    tv_epsilon: float = DEFAULT_TV_EPSILON,
    filter_epsilon: float = DEFAULT_FILTER_EPSILON,
    denominator_floor: float = DEFAULT_DENOMINATOR_FLOOR,
    psf: np.ndarray | None = None,
    normalize_psf: bool = True,
) -> np.ndarray:
    return richardson_lucy_tv_deconvolution(
        [observed, phantom.psf if psf is None else psf],
        spatial_mode=phantom.spatial_mode,
        iterations=iterations,
        tv_regularization=tv_regularization,
        tv_epsilon=tv_epsilon,
        normalize_psf=normalize_psf,
        filter_epsilon=filter_epsilon,
        denominator_floor=denominator_floor,
    )


def _production_rl(
    observed: np.ndarray,
    phantom: Phantom,
    *,
    iterations: int = DEFAULT_ITERATIONS,
) -> np.ndarray:
    return richardson_lucy_deconvolution(
        [observed, phantom.psf],
        spatial_mode=phantom.spatial_mode,
        iterations=iterations,
    )


def _result_row(
    phantom: Phantom,
    experiment: str,
    level: str,
    restored: np.ndarray,
    *,
    implementation: str,
    iterations: int,
    tv_regularization: float,
    tv_epsilon: float,
    filter_epsilon: float,
    denominator_floor: float,
    boundary: str = "zero",
    initialization: str = "constant_0.5",
    tv_discretization: str = "production_central",
    diagnostics: VariantDiagnostics | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "phantom": phantom.name,
        "experiment": experiment,
        "level": level,
        "implementation": implementation,
        "iterations": iterations,
        "tv_regularization": tv_regularization,
        "tv_epsilon": tv_epsilon,
        "filter_epsilon": filter_epsilon,
        "denominator_floor": denominator_floor,
        "boundary": boundary,
        "initialization": initialization,
        "tv_discretization": tv_discretization,
        "minimum_raw_denominator": (
            diagnostics.minimum_raw_denominator if diagnostics else ""
        ),
        "maximum_floor_fraction": (
            diagnostics.maximum_floor_fraction if diagnostics else ""
        ),
    }
    row.update(calculate_metrics(restored, phantom))
    return row


def _shift_with_zero_fill(values: np.ndarray, offset: tuple[int, ...]) -> np.ndarray:
    shifted = np.zeros_like(values)
    source_slices: list[slice] = []
    target_slices: list[slice] = []
    for size, delta in zip(values.shape, offset, strict=True):
        if delta > 0:
            source_slices.append(slice(0, size - delta))
            target_slices.append(slice(delta, size))
        elif delta < 0:
            source_slices.append(slice(-delta, size))
            target_slices.append(slice(0, size + delta))
        else:
            source_slices.append(slice(None))
            target_slices.append(slice(None))
    shifted[tuple(target_slices)] = values[tuple(source_slices)]
    return shifted


def run_sweeps(phantom: Phantom) -> list[dict[str, Any]]:
    observed = observed_image(phantom)
    rows: list[dict[str, Any]] = []

    rows.append(
        _result_row(
            phantom,
            "baseline",
            "observed",
            observed,
            implementation="forward_model",
            iterations=0,
            tv_regularization=0.0,
            tv_epsilon=DEFAULT_TV_EPSILON,
            filter_epsilon=DEFAULT_FILTER_EPSILON,
            denominator_floor=DEFAULT_DENOMINATOR_FLOOR,
            boundary="reflect",
        )
    )
    rl = _production_rl(observed, phantom)
    rows.append(
        _result_row(
            phantom,
            "baseline",
            "ordinary_rl",
            rl,
            implementation="production_rl",
            iterations=DEFAULT_ITERATIONS,
            tv_regularization=0.0,
            tv_epsilon=DEFAULT_TV_EPSILON,
            filter_epsilon=DEFAULT_FILTER_EPSILON,
            denominator_floor=DEFAULT_DENOMINATOR_FLOOR,
        )
    )

    example_settings = EXAMPLE_WORKFLOW_SETTINGS[phantom.name]
    example_tv = _production_tv(observed, phantom, **example_settings)
    rows.append(
        _result_row(
            phantom,
            "example_workflow_parameters",
            "shipped_example",
            example_tv,
            implementation="production_rl_tv",
            iterations=example_settings["iterations"],
            tv_regularization=example_settings["tv_regularization"],
            tv_epsilon=DEFAULT_TV_EPSILON,
            filter_epsilon=DEFAULT_FILTER_EPSILON,
            denominator_floor=example_settings["denominator_floor"],
        )
    )
    default_tv = _production_tv(observed, phantom)
    rows.append(
        _result_row(
            phantom,
            "baseline",
            "default_rl_tv",
            default_tv,
            implementation="production_rl_tv",
            iterations=DEFAULT_ITERATIONS,
            tv_regularization=DEFAULT_TV_REGULARIZATION,
            tv_epsilon=DEFAULT_TV_EPSILON,
            filter_epsilon=DEFAULT_FILTER_EPSILON,
            denominator_floor=DEFAULT_DENOMINATOR_FLOOR,
        )
    )
    tv_zero = _production_tv(observed, phantom, tv_regularization=0.0)
    rows.append(
        _result_row(
            phantom,
            "lambda_zero_parity",
            "production_tv_lambda_0",
            tv_zero,
            implementation="production_rl_tv",
            iterations=DEFAULT_ITERATIONS,
            tv_regularization=0.0,
            tv_epsilon=DEFAULT_TV_EPSILON,
            filter_epsilon=DEFAULT_FILTER_EPSILON,
            denominator_floor=DEFAULT_DENOMINATOR_FLOOR,
        )
    )

    for value in (0.0, 0.0005, 0.002, 0.008, 0.012, 0.02):
        restored = _production_tv(observed, phantom, tv_regularization=value)
        rows.append(
            _result_row(
                phantom,
                "tv_regularization",
                f"{value:g}",
                restored,
                implementation="production_rl_tv",
                iterations=DEFAULT_ITERATIONS,
                tv_regularization=value,
                tv_epsilon=DEFAULT_TV_EPSILON,
                filter_epsilon=DEFAULT_FILTER_EPSILON,
                denominator_floor=DEFAULT_DENOMINATOR_FLOOR,
            )
        )

    for value in (5, 10, 25, 50):
        restored = _production_tv(observed, phantom, iterations=value)
        rows.append(
            _result_row(
                phantom,
                "iterations",
                str(value),
                restored,
                implementation="production_rl_tv",
                iterations=value,
                tv_regularization=DEFAULT_TV_REGULARIZATION,
                tv_epsilon=DEFAULT_TV_EPSILON,
                filter_epsilon=DEFAULT_FILTER_EPSILON,
                denominator_floor=DEFAULT_DENOMINATOR_FLOOR,
            )
        )

    for value in (1e-12, 1e-6, 1e-3):
        restored = _production_tv(observed, phantom, tv_epsilon=value)
        rows.append(
            _result_row(
                phantom,
                "tv_epsilon",
                f"{value:g}",
                restored,
                implementation="production_rl_tv",
                iterations=DEFAULT_ITERATIONS,
                tv_regularization=DEFAULT_TV_REGULARIZATION,
                tv_epsilon=value,
                filter_epsilon=DEFAULT_FILTER_EPSILON,
                denominator_floor=DEFAULT_DENOMINATOR_FLOOR,
            )
        )

    for value in (0.001, 0.05, 0.15, 0.2, 0.5):
        restored = _production_tv(observed, phantom, denominator_floor=value)
        rows.append(
            _result_row(
                phantom,
                "denominator_floor",
                f"{value:g}",
                restored,
                implementation="production_rl_tv",
                iterations=DEFAULT_ITERATIONS,
                tv_regularization=DEFAULT_TV_REGULARIZATION,
                tv_epsilon=DEFAULT_TV_EPSILON,
                filter_epsilon=DEFAULT_FILTER_EPSILON,
                denominator_floor=value,
            )
        )

    for value in (0.0, 1e-12, 1e-6, 1e-3):
        restored = _production_tv(observed, phantom, filter_epsilon=value)
        rows.append(
            _result_row(
                phantom,
                "filter_epsilon",
                f"{value:g}",
                restored,
                implementation="production_rl_tv",
                iterations=DEFAULT_ITERATIONS,
                tv_regularization=DEFAULT_TV_REGULARIZATION,
                tv_epsilon=DEFAULT_TV_EPSILON,
                filter_epsilon=value,
                denominator_floor=DEFAULT_DENOMINATOR_FLOOR,
            )
        )

    for iterations in (5, 25):
        for initialization in ("constant_0.5", "observed", "positive_mean"):
            restored, diagnostics = rl_tv_comparison_variant(
                observed,
                phantom.psf,
                iterations=iterations,
                initialization=initialization,
            )
            rows.append(
                _result_row(
                    phantom,
                    "initialization",
                    f"{initialization}_iter_{iterations}",
                    restored,
                    implementation="comparison_variant",
                    iterations=iterations,
                    tv_regularization=DEFAULT_TV_REGULARIZATION,
                    tv_epsilon=DEFAULT_TV_EPSILON,
                    filter_epsilon=DEFAULT_FILTER_EPSILON,
                    denominator_floor=DEFAULT_DENOMINATOR_FLOOR,
                    initialization=initialization,
                    diagnostics=diagnostics,
                )
            )

    for iterations in (5, 25):
        for boundary in ("zero", "reflect"):
            restored, diagnostics = rl_tv_comparison_variant(
                observed,
                phantom.psf,
                iterations=iterations,
                boundary=boundary,
            )
            rows.append(
                _result_row(
                    phantom,
                    "boundary",
                    f"{boundary}_iter_{iterations}",
                    restored,
                    implementation="comparison_variant",
                    iterations=iterations,
                    tv_regularization=DEFAULT_TV_REGULARIZATION,
                    tv_epsilon=DEFAULT_TV_EPSILON,
                    filter_epsilon=DEFAULT_FILTER_EPSILON,
                    denominator_floor=DEFAULT_DENOMINATOR_FLOOR,
                    boundary=boundary,
                    diagnostics=diagnostics,
                )
            )

    for discretization in ("production_central", "forward_backward"):
        restored, diagnostics = rl_tv_comparison_variant(
            observed,
            phantom.psf,
            iterations=DEFAULT_ITERATIONS,
            tv_discretization=discretization,
        )
        rows.append(
            _result_row(
                phantom,
                "tv_discretization",
                discretization,
                restored,
                implementation="comparison_variant",
                iterations=DEFAULT_ITERATIONS,
                tv_regularization=DEFAULT_TV_REGULARIZATION,
                tv_epsilon=DEFAULT_TV_EPSILON,
                filter_epsilon=DEFAULT_FILTER_EPSILON,
                denominator_floor=DEFAULT_DENOMINATOR_FLOOR,
                tv_discretization=discretization,
                diagnostics=diagnostics,
            )
        )

    scaled_psf = phantom.psf * np.float32(3.0)
    for normalize in (True, False):
        restored = _production_tv(
            observed,
            phantom,
            psf=scaled_psf,
            normalize_psf=normalize,
        )
        rows.append(
            _result_row(
                phantom,
                "psf_normalization",
                f"scaled_sum_3_normalize_{normalize}",
                restored,
                implementation="production_rl_tv",
                iterations=DEFAULT_ITERATIONS,
                tv_regularization=DEFAULT_TV_REGULARIZATION,
                tv_epsilon=DEFAULT_TV_EPSILON,
                filter_epsilon=DEFAULT_FILTER_EPSILON,
                denominator_floor=DEFAULT_DENOMINATOR_FLOOR,
            )
        )

    offset = (0,) * (phantom.psf.ndim - 1) + (1,)
    shifted_psf = _shift_with_zero_fill(phantom.psf, offset)
    shifted_psf /= np.float32(np.sum(shifted_psf, dtype=np.float64))
    restored = _production_tv(observed, phantom, psf=shifted_psf)
    rows.append(
        _result_row(
            phantom,
            "psf_centering",
            "one_pixel_x_shift",
            restored,
            implementation="production_rl_tv",
            iterations=DEFAULT_ITERATIONS,
            tv_regularization=DEFAULT_TV_REGULARIZATION,
            tv_epsilon=DEFAULT_TV_EPSILON,
            filter_epsilon=DEFAULT_FILTER_EPSILON,
            denominator_floor=DEFAULT_DENOMINATOR_FLOOR,
        )
    )

    for xy_size in (9, 15, 25):
        z_size = 1 if phantom.truth.ndim == 2 else {9: 5, 15: 7, 25: 9}[xy_size]
        candidate_psf = _born_wolf(
            phantom.truth.shape,
            pixel_size_xy_um=phantom.spacing_um[-1],
            z_step_um=phantom.spacing_um[0] if phantom.truth.ndim == 3 else 0.30,
            xy_size=xy_size,
            z_size=z_size,
        )
        restored = _production_tv(observed, phantom, psf=candidate_psf)
        rows.append(
            _result_row(
                phantom,
                "psf_support",
                f"shape_{'x'.join(str(size) for size in candidate_psf.shape)}",
                restored,
                implementation="production_rl_tv",
                iterations=DEFAULT_ITERATIONS,
                tv_regularization=DEFAULT_TV_REGULARIZATION,
                tv_epsilon=DEFAULT_TV_EPSILON,
                filter_epsilon=DEFAULT_FILTER_EPSILON,
                denominator_floor=DEFAULT_DENOMINATOR_FLOOR,
            )
        )

    mismatched_psf = _born_wolf(
        phantom.truth.shape,
        pixel_size_xy_um=phantom.spacing_um[-1] * 0.75,
        z_step_um=(phantom.spacing_um[0] * 0.75 if phantom.truth.ndim == 3 else 0.30),
        xy_size=15,
        z_size=1 if phantom.truth.ndim == 2 else 7,
    )
    restored = _production_tv(observed, phantom, psf=mismatched_psf)
    rows.append(
        _result_row(
            phantom,
            "psf_sampling",
            "direct_call_0.75x_spacing",
            restored,
            implementation="production_rl_tv_direct_without_metadata",
            iterations=DEFAULT_ITERATIONS,
            tv_regularization=DEFAULT_TV_REGULARIZATION,
            tv_epsilon=DEFAULT_TV_EPSILON,
            filter_epsilon=DEFAULT_FILTER_EPSILON,
            denominator_floor=DEFAULT_DENOMINATOR_FLOOR,
        )
    )

    for photons in (30.0, 100.0, 500.0):
        noisy = observed_image(phantom, photons_at_peak=photons)
        restored = _production_tv(noisy, phantom)
        rows.append(
            _result_row(
                phantom,
                "noise_regime",
                f"{photons:g}_photons_at_peak",
                restored,
                implementation="production_rl_tv",
                iterations=DEFAULT_ITERATIONS,
                tv_regularization=DEFAULT_TV_REGULARIZATION,
                tv_epsilon=DEFAULT_TV_EPSILON,
                filter_epsilon=DEFAULT_FILTER_EPSILON,
                denominator_floor=DEFAULT_DENOMINATOR_FLOOR,
            )
        )

    parity = np.abs(rl - tv_zero)
    rows.append(
        {
            **_result_row(
                phantom,
                "lambda_zero_parity",
                "absolute_difference",
                parity,
                implementation="derived_difference",
                iterations=DEFAULT_ITERATIONS,
                tv_regularization=0.0,
                tv_epsilon=DEFAULT_TV_EPSILON,
                filter_epsilon=DEFAULT_FILTER_EPSILON,
                denominator_floor=DEFAULT_DENOMINATOR_FLOOR,
            ),
            "parity_max_abs_difference": float(np.max(parity)),
            "parity_mean_abs_difference": float(np.mean(parity)),
        }
    )
    return rows


def _axis_metadata(
    phantom: Phantom, spacing: tuple[float, ...]
) -> tuple[AxisMetadata, ...]:
    names = ("y", "x") if phantom.truth.ndim == 2 else ("z", "y", "x")
    return tuple(
        AxisMetadata(name, "space", unit="micrometer", scale=scale)
        for name, scale in zip(names, spacing, strict=True)
    )


def psf_validation_rows(phantom: Phantom) -> list[dict[str, Any]]:
    """Validate kernel properties plus metadata-aware sampling behavior."""
    psf = phantom.psf
    center = tuple(size // 2 for size in psf.shape)
    coordinates = np.indices(psf.shape, dtype=np.float64)
    total = float(np.sum(psf, dtype=np.float64))
    centroid = tuple(
        float(np.sum(coordinates[axis] * psf, dtype=np.float64) / total)
        for axis in range(psf.ndim)
    )
    centroid_offset = float(
        np.linalg.norm(np.asarray(centroid) - np.asarray(center, dtype=float))
    )
    rows: list[dict[str, Any]] = [
        {
            "phantom": phantom.name,
            "check": "kernel_properties",
            "status": "pass"
            if (
                np.isclose(total, 1.0, rtol=1e-5)
                and float(np.min(psf)) >= 0.0
                and all(size % 2 == 1 for size in psf.shape)
                and psf[center] == np.max(psf)
            )
            else "fail",
            "detail": (
                f"shape={psf.shape}; sum={total:.9g}; min={float(np.min(psf)):.9g}; "
                f"peak_at_center={bool(psf[center] == np.max(psf))}; "
                f"centroid_offset_voxels={centroid_offset:.6g}"
            ),
        }
    ]

    image_state = image_state_from_array(
        phantom.truth,
        axes=_axis_metadata(phantom, phantom.spacing_um),
    )
    psf_state = image_state_from_array(
        phantom.psf,
        axes=_axis_metadata(phantom, phantom.spacing_um),
    )
    assert image_state is not None and psf_state is not None
    validate_psf_image_states(
        image_state,
        psf_state,
        spatial_ndim=phantom.truth.ndim,
        operation_title="Richardson-Lucy TV validation",
    )
    rows.append(
        {
            "phantom": phantom.name,
            "check": "matching_physical_sampling",
            "status": "pass",
            "detail": (
                "Explicit image and PSF grids with matching micrometer spacing "
                "accepted."
            ),
        }
    )

    mismatch_spacing = tuple(value * 0.75 for value in phantom.spacing_um)
    mismatch_state = image_state_from_array(
        phantom.psf,
        axes=_axis_metadata(phantom, mismatch_spacing),
    )
    assert mismatch_state is not None
    try:
        validate_psf_image_states(
            image_state,
            mismatch_state,
            spatial_ndim=phantom.truth.ndim,
            operation_title="Richardson-Lucy TV validation",
        )
    except ValueError as exc:
        rows.append(
            {
                "phantom": phantom.name,
                "check": "mismatched_physical_sampling",
                "status": "expected_rejection",
                "detail": str(exc),
            }
        )
    else:
        rows.append(
            {
                "phantom": phantom.name,
                "check": "mismatched_physical_sampling",
                "status": "unexpected_acceptance",
                "detail": "A 25% spacing mismatch was not rejected.",
            }
        )

    wrong_rank = np.zeros((3,) + phantom.psf.shape, dtype=np.float32)
    try:
        richardson_lucy_tv_deconvolution(
            [phantom.truth, wrong_rank],
            spatial_mode=phantom.spatial_mode,
            iterations=1,
        )
    except ValueError as exc:
        rows.append(
            {
                "phantom": phantom.name,
                "check": "mismatched_dimensionality",
                "status": "expected_rejection",
                "detail": str(exc),
            }
        )
    else:
        rows.append(
            {
                "phantom": phantom.name,
                "check": "mismatched_dimensionality",
                "status": "unexpected_acceptance",
                "detail": "A PSF with one extra dimension was not rejected.",
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0])
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summary(
    rows: list[dict[str, Any]],
    psf_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for phantom_name in sorted({str(row["phantom"]) for row in rows}):
        subset = [row for row in rows if row["phantom"] == phantom_name]
        baseline = {
            str(row["level"]): row for row in subset if row["experiment"] == "baseline"
        }
        lambda_rows = [
            row for row in subset if row["experiment"] == "tv_regularization"
        ]
        best_mse = min(lambda_rows, key=lambda row: float(row["mse"]))
        best_dim = max(
            lambda_rows,
            key=lambda row: float(row["dim_structure_recovery"]),
        )
        parity = next(
            row
            for row in subset
            if row["experiment"] == "lambda_zero_parity"
            and row["level"] == "absolute_difference"
        )
        summaries[phantom_name] = {
            "observed": _selected_metrics(baseline["observed"]),
            "ordinary_rl": _selected_metrics(baseline["ordinary_rl"]),
            "default_rl_tv": _selected_metrics(baseline["default_rl_tv"]),
            "best_lambda_by_mse": {
                "lambda": float(best_mse["tv_regularization"]),
                **_selected_metrics(best_mse),
            },
            "best_lambda_by_dim_structure_recovery": {
                "lambda": float(best_dim["tv_regularization"]),
                **_selected_metrics(best_dim),
            },
            "lambda_zero_parity": {
                "max_abs_difference": float(parity["parity_max_abs_difference"]),
                "mean_abs_difference": float(parity["parity_mean_abs_difference"]),
            },
        }
    return {
        "generated_by": "scripts/validate_rl_tv_phantoms.py",
        "seeds": {"2D": SEED_2D, "3D": SEED_3D},
        "defaults_evaluated": {
            "iterations": DEFAULT_ITERATIONS,
            "tv_regularization": DEFAULT_TV_REGULARIZATION,
            "tv_epsilon": DEFAULT_TV_EPSILON,
            "filter_epsilon": DEFAULT_FILTER_EPSILON,
            "denominator_floor": DEFAULT_DENOMINATOR_FLOOR,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_image": skimage.__version__,
            "platform": platform.platform(),
        },
        "phantoms": summaries,
        "psf_checks": [asdict_row(row) for row in psf_rows],
    }


def _selected_metrics(row: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(row[key])
        for key in (
            "mse",
            "psnr_db",
            "ssim",
            "flux_ratio",
            "border_mse",
            "points_recovery",
            "thin_line_recovery",
            "dim_structure_recovery",
            "dim_structure_loss_fraction",
        )
    }


def asdict_row(row: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe shallow copy for symmetry with dataclass output."""
    return dict(row)


def main() -> int:
    args = _parse_args()
    requested = set(args.dimensions)
    phantoms = []
    if "2d" in requested:
        phantoms.append(make_phantom_2d())
    if "3d" in requested:
        phantoms.append(make_phantom_3d())

    all_rows: list[dict[str, Any]] = []
    all_psf_rows: list[dict[str, Any]] = []
    for phantom in phantoms:
        print(f"Analyzing {phantom.name} phantom...", flush=True)
        all_psf_rows.extend(psf_validation_rows(phantom))
        all_rows.extend(run_sweeps(phantom))

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "results.csv", all_rows)
    _write_csv(output_dir / "psf-checks.csv", all_psf_rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(_summary(all_rows, all_psf_rows), stream, indent=2)
        stream.write("\n")
    print(f"Wrote validation evidence to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
