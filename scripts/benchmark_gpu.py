"""Reproducible CPU/CuPy benchmark spike for candidate VIPP operations.

This script is intentionally outside normal execution.  It imports optional GPU
packages only after capability detection, warms kernels before measurement, and
records host/device transfer time separately from resident compute time.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import importlib.metadata
import json
import platform
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from napari_vipp.core.compute import detect_compute_capabilities
from napari_vipp.core.operations import (
    gaussian_blur,
    gaussian_blur_3d,
    median_filter,
    richardson_lucy_deconvolution,
    richardson_lucy_tv_deconvolution,
    rolling_ball_background,
    subtract_background,
)


@dataclass(frozen=True, slots=True)
class Profile:
    image_2d: tuple[int, int]
    image_3d: tuple[int, int, int]
    rl_2d: tuple[int, int]
    rl_3d: tuple[int, int, int]
    batch_items: int
    rl_iterations: int
    repeats: int
    warmups: int


@dataclass(frozen=True, slots=True)
class Measurement:
    median_seconds: float
    minimum_seconds: float
    maximum_seconds: float
    output: Any

    def range_dict(self):
        return {
            "median": self.median_seconds,
            "minimum": self.minimum_seconds,
            "maximum": self.maximum_seconds,
        }


PROFILES = {
    "smoke": Profile(
        image_2d=(256, 256),
        image_3d=(16, 96, 96),
        rl_2d=(128, 128),
        rl_3d=(8, 48, 48),
        batch_items=3,
        rl_iterations=5,
        repeats=2,
        warmups=1,
    ),
    "standard": Profile(
        image_2d=(1024, 1024),
        image_3d=(32, 192, 192),
        rl_2d=(512, 512),
        rl_3d=(24, 128, 128),
        batch_items=6,
        rl_iterations=15,
        repeats=3,
        warmups=1,
    ),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILES),
        default="smoke",
        help="Workload sizes and repeat counts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path. JSON is always printed to stdout.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    profile = PROFILES[args.profile]
    capability = detect_compute_capabilities()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "profile": args.profile,
        "profile_definition": asdict(profile),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "capabilities": capability.as_dict(),
            "optional_providers": {
                name: _optional_provider_status(name)
                for name in ("cupy", "cucim", "torch")
            },
        },
        "methodology": {
            "time_statistic": "median wall-clock seconds after warm-up",
            "gpu_synchronization": "current CuPy stream synchronized",
            "gpu_memory": (
                "CuPy memory-pool total bytes after one clean execution; "
                "a conservative allocation high-water proxy, not device-wide peak"
            ),
            "promotion_speedup_threshold": 1.5,
        },
        "results": [],
    }

    if not capability.gpu.available:
        payload["status"] = "gpu_unavailable"
        payload["reason"] = capability.gpu.reason
        _emit(payload, args.output)
        return 0

    cupy = importlib.import_module("cupy")
    cupyx_ndi = importlib.import_module("cupyx.scipy.ndimage")
    payload["environment"]["cupy_runtime"] = {
        "version": str(cupy.__version__),
        "runtime_version": int(cupy.cuda.runtime.runtimeGetVersion()),
        "driver_version": int(cupy.cuda.runtime.driverGetVersion()),
    }
    rng = np.random.default_rng(20260715)
    results = payload["results"]

    image_2d = rng.random(profile.image_2d, dtype=np.float32)
    image_3d = rng.random(profile.image_3d, dtype=np.float32)

    results.append(
        _benchmark_resident_operation(
            "gaussian_blur_2d",
            image_2d,
            cpu_func=lambda value: gaussian_blur(value, sigma=2.0),
            gpu_func=lambda value: cupyx_ndi.gaussian_filter(
                value, sigma=(2.0, 2.0)
            ),
            cupy=cupy,
            profile=profile,
        )
    )
    results.append(
        _benchmark_resident_operation(
            "gaussian_blur_3d",
            image_3d,
            cpu_func=lambda value: gaussian_blur_3d(
                value, sigma_z=1.5, sigma_y=2.0, sigma_x=2.0
            ),
            gpu_func=lambda value: cupyx_ndi.gaussian_filter(
                value, sigma=(1.5, 2.0, 2.0)
            ),
            cupy=cupy,
            profile=profile,
        )
    )
    results.append(
        _benchmark_resident_operation(
            "median_filter_2d",
            image_2d,
            cpu_func=lambda value: median_filter(value, size=5),
            gpu_func=lambda value: cupyx_ndi.median_filter(value, size=(5, 5)),
            cupy=cupy,
            profile=profile,
        )
    )
    results.append(
        _benchmark_resident_operation(
            "median_filter_3d",
            image_3d,
            cpu_func=lambda value: median_filter(value, size=5),
            gpu_func=lambda value: cupyx_ndi.median_filter(
                value, size=(1, 5, 5)
            ),
            cupy=cupy,
            profile=profile,
        )
    )

    for dimensions, shape in ((2, profile.rl_2d), (3, profile.rl_3d)):
        image = _deconvolution_fixture(shape, rng)
        psf = _gaussian_psf(dimensions)
        spatial_mode = "2D YX" if dimensions == 2 else "3D ZYX"
        results.append(
            _benchmark_resident_operation(
                f"richardson_lucy_{dimensions}d",
                (image, psf),
                cpu_func=lambda pair, mode=spatial_mode: (
                    richardson_lucy_deconvolution(
                        pair,
                        spatial_mode=mode,
                        iterations=profile.rl_iterations,
                    )
                ),
                gpu_func=lambda pair: _cupy_rl(
                    pair[0],
                    pair[1],
                    iterations=profile.rl_iterations,
                    cupy=cupy,
                    cupyx_ndi=cupyx_ndi,
                ),
                cupy=cupy,
                profile=profile,
            )
        )
        results.append(
            _benchmark_resident_operation(
                f"richardson_lucy_tv_{dimensions}d",
                (image, psf),
                cpu_func=lambda pair, mode=spatial_mode: (
                    richardson_lucy_tv_deconvolution(
                        pair,
                        spatial_mode=mode,
                        iterations=profile.rl_iterations,
                    )
                ),
                gpu_func=lambda pair: _cupy_rl_tv(
                    pair[0],
                    pair[1],
                    iterations=profile.rl_iterations,
                    cupy=cupy,
                    cupyx_ndi=cupyx_ndi,
                ),
                cupy=cupy,
                profile=profile,
            )
        )

    results.extend(
        _rolling_ball_results(
            image_2d,
            image_3d,
            cupy=cupy,
            profile=profile,
        )
    )
    results.extend(
        _batch_residency_results(
            rng,
            profile,
            cupy=cupy,
            cupyx_ndi=cupyx_ndi,
        )
    )
    payload["status"] = "completed"
    _emit(payload, args.output)
    return 0


def _benchmark_resident_operation(
    workload: str,
    host_input,
    *,
    cpu_func: Callable,
    gpu_func: Callable,
    cupy,
    profile: Profile,
) -> dict[str, Any]:
    synchronize = cupy.cuda.get_current_stream().synchronize
    cpu_measurement = _measure(
        lambda: cpu_func(host_input),
        warmups=profile.warmups,
        repeats=profile.repeats,
    )
    cpu_seconds = cpu_measurement.median_seconds
    cpu_output = cpu_measurement.output

    device_input = _to_device(host_input, cupy)
    gpu_func(device_input)
    synchronize()

    transfer_in_measurement = _measure(
        lambda: _to_device(host_input, cupy),
        warmups=profile.warmups,
        repeats=profile.repeats,
        synchronize=synchronize,
    )
    gpu_compute_measurement = _measure(
        lambda: gpu_func(device_input),
        warmups=profile.warmups,
        repeats=profile.repeats,
        synchronize=synchronize,
    )
    device_output = gpu_compute_measurement.output
    transfer_out_measurement = _measure(
        lambda: _to_host(device_output, cupy),
        warmups=profile.warmups,
        repeats=profile.repeats,
        synchronize=synchronize,
    )

    def end_to_end():
        copied = _to_device(host_input, cupy)
        result = gpu_func(copied)
        return _to_host(result, cupy)

    gpu_e2e_measurement = _measure(
        end_to_end,
        warmups=profile.warmups,
        repeats=profile.repeats,
        synchronize=synchronize,
    )
    transfer_in_seconds = transfer_in_measurement.median_seconds
    gpu_compute_seconds = gpu_compute_measurement.median_seconds
    transfer_out_seconds = transfer_out_measurement.median_seconds
    gpu_e2e_seconds = gpu_e2e_measurement.median_seconds
    gpu_output = gpu_e2e_measurement.output
    pool = cupy.get_default_memory_pool()
    device_input = None
    device_output = None
    gc.collect()
    pool.free_all_blocks()
    clean_input = _to_device(host_input, cupy)
    clean_output = gpu_func(clean_input)
    synchronize()
    memory_pool_bytes = int(pool.total_bytes())
    del clean_input, clean_output

    comparison = _comparison(cpu_output, gpu_output)
    return {
        "workload": workload,
        "status": "completed",
        "shape": _shape_description(host_input),
        "dtype": _dtype_description(host_input),
        "cpu_seconds": cpu_seconds,
        "gpu_transfer_in_seconds": transfer_in_seconds,
        "gpu_compute_seconds": gpu_compute_seconds,
        "gpu_transfer_out_seconds": transfer_out_seconds,
        "gpu_end_to_end_seconds": gpu_e2e_seconds,
        "resident_compute_speedup": _ratio(cpu_seconds, gpu_compute_seconds),
        "end_to_end_speedup": _ratio(cpu_seconds, gpu_e2e_seconds),
        "gpu_memory_pool_bytes": memory_pool_bytes,
        "timing_ranges_seconds": {
            "cpu": cpu_measurement.range_dict(),
            "gpu_transfer_in": transfer_in_measurement.range_dict(),
            "gpu_compute": gpu_compute_measurement.range_dict(),
            "gpu_transfer_out": transfer_out_measurement.range_dict(),
            "gpu_end_to_end": gpu_e2e_measurement.range_dict(),
        },
        "comparison": comparison,
    }


def _rolling_ball_results(
    image_2d,
    image_3d,
    *,
    cupy,
    profile: Profile,
) -> list[dict[str, Any]]:
    try:
        restoration = importlib.import_module("cucim.skimage.restoration")
    except Exception as exc:
        reason = f"cuCIM unavailable: {type(exc).__name__}: {exc}"
        return [
            {"workload": name, "status": "unavailable", "reason": reason}
            for name in (
                "rolling_ball_background_2d",
                "subtract_background_2d",
                "rolling_ball_background_3d",
                "subtract_background_3d",
            )
        ]

    cases = []
    for dimensions, image in ((2, image_2d), (3, image_3d)):
        mode = "2D YX" if dimensions == 2 else "3D ZYX"
        radius = 15.0 if dimensions == 2 else 5.0
        cases.append(
            _benchmark_resident_operation(
                f"rolling_ball_background_{dimensions}d",
                image,
                cpu_func=lambda value, current_mode=mode, current_radius=radius: (
                    rolling_ball_background(
                        value,
                        radius=current_radius,
                        spatial_mode=current_mode,
                    )
                ),
                gpu_func=lambda value, current_radius=radius: restoration.rolling_ball(
                    value, radius=current_radius
                ),
                cupy=cupy,
                profile=profile,
            )
        )
        cases.append(
            _benchmark_resident_operation(
                f"subtract_background_{dimensions}d",
                image,
                cpu_func=lambda value, current_mode=mode, current_radius=radius: (
                    subtract_background(
                        value,
                        radius=current_radius,
                        spatial_mode=current_mode,
                    )
                ),
                gpu_func=lambda value, current_radius=radius: cupy.maximum(
                    value - restoration.rolling_ball(value, radius=current_radius),
                    0,
                ),
                cupy=cupy,
                profile=profile,
            )
        )
    return cases


def _batch_residency_results(rng, profile, *, cupy, cupyx_ndi):
    batch = [
        rng.random(profile.image_2d, dtype=np.float32)
        for _ in range(profile.batch_items)
    ]

    def cpu_chain():
        return [median_filter(gaussian_blur(item, sigma=2.0), size=5) for item in batch]

    def gpu_roundtrip_chain():
        outputs = []
        for item in batch:
            blurred_host = cupy.asnumpy(
                cupyx_ndi.gaussian_filter(cupy.asarray(item), sigma=(2.0, 2.0))
            )
            outputs.append(
                cupy.asnumpy(
                    cupyx_ndi.median_filter(
                        cupy.asarray(blurred_host), size=(5, 5)
                    )
                )
            )
        return outputs

    def gpu_resident_chain():
        outputs = []
        for item in batch:
            device = cupy.asarray(item)
            device = cupyx_ndi.gaussian_filter(device, sigma=(2.0, 2.0))
            device = cupyx_ndi.median_filter(device, size=(5, 5))
            outputs.append(cupy.asnumpy(device))
        return outputs

    synchronize = cupy.cuda.get_current_stream().synchronize
    cpu_measurement = _measure(
        cpu_chain, warmups=profile.warmups, repeats=profile.repeats
    )
    roundtrip_measurement = _measure(
        gpu_roundtrip_chain,
        warmups=profile.warmups,
        repeats=profile.repeats,
        synchronize=synchronize,
    )
    resident_measurement = _measure(
        gpu_resident_chain,
        warmups=profile.warmups,
        repeats=profile.repeats,
        synchronize=synchronize,
    )
    cpu_seconds = cpu_measurement.median_seconds
    cpu_output = cpu_measurement.output
    roundtrip_seconds = roundtrip_measurement.median_seconds
    roundtrip_output = roundtrip_measurement.output
    resident_seconds = resident_measurement.median_seconds
    resident_output = resident_measurement.output
    return [
        {
            "workload": "batch_gaussian_median_roundtrip_2d",
            "status": "completed",
            "shape": list(profile.image_2d),
            "items": profile.batch_items,
            "cpu_seconds": cpu_seconds,
            "gpu_end_to_end_seconds": roundtrip_seconds,
            "end_to_end_speedup": _ratio(cpu_seconds, roundtrip_seconds),
            "host_device_boundaries_per_item": 4,
            "timing_ranges_seconds": {
                "cpu": cpu_measurement.range_dict(),
                "gpu_end_to_end": roundtrip_measurement.range_dict(),
            },
            "comparison": _comparison(cpu_output, roundtrip_output),
        },
        {
            "workload": "batch_gaussian_median_resident_2d",
            "status": "completed",
            "shape": list(profile.image_2d),
            "items": profile.batch_items,
            "cpu_seconds": cpu_seconds,
            "gpu_end_to_end_seconds": resident_seconds,
            "end_to_end_speedup": _ratio(cpu_seconds, resident_seconds),
            "host_device_boundaries_per_item": 2,
            "residency_speedup_vs_roundtrip": _ratio(
                roundtrip_seconds, resident_seconds
            ),
            "timing_ranges_seconds": {
                "cpu": cpu_measurement.range_dict(),
                "gpu_end_to_end": resident_measurement.range_dict(),
            },
            "comparison": _comparison(cpu_output, resident_output),
        },
    ]


def _cupy_rl(image, psf, *, iterations, cupy, cupyx_ndi):
    estimate = cupy.full(image.shape, 0.5, dtype=cupy.float32)
    psf = psf.astype(cupy.float32, copy=False)
    psf = psf / psf.sum(dtype=cupy.float64)
    mirror = cupy.flip(psf)
    epsilon = cupy.float32(1e-12)
    for _ in range(iterations):
        blurred = cupyx_ndi.convolve(estimate, psf, mode="constant") + epsilon
        ratio = cupy.where(blurred < 1e-12, 0.0, image / blurred)
        estimate *= cupyx_ndi.convolve(ratio, mirror, mode="constant")
        estimate = cupy.nan_to_num(estimate, nan=0.0, posinf=0.0, neginf=0.0)
        estimate = cupy.maximum(estimate, 0.0).astype(cupy.float32, copy=False)
    return estimate


def _cupy_rl_tv(image, psf, *, iterations, cupy, cupyx_ndi):
    estimate = cupy.full(image.shape, 0.5, dtype=cupy.float32)
    psf = psf.astype(cupy.float32, copy=False)
    psf = psf / psf.sum(dtype=cupy.float64)
    mirror = cupy.flip(psf)
    epsilon = cupy.float32(1e-12)
    for _ in range(iterations):
        blurred = cupyx_ndi.convolve(estimate, psf, mode="constant") + epsilon
        ratio = cupy.where(blurred < 1e-12, 0.0, image / blurred)
        correction = cupyx_ndi.convolve(ratio, mirror, mode="constant")
        gradients = cupy.gradient(estimate)
        norm = cupy.sqrt(
            cupy.sum(
                cupy.stack([gradient * gradient for gradient in gradients]), axis=0
            )
            + cupy.float32(1e-6) ** 2
        )
        divergence = cupy.zeros(estimate.shape, dtype=cupy.float32)
        for axis, gradient in enumerate(gradients):
            divergence += cupy.gradient(gradient / norm, axis=axis)
        denominator = cupy.maximum(
            1.0 - cupy.float32(0.002) * divergence, cupy.float32(0.05)
        )
        estimate = estimate * correction / denominator
        estimate = cupy.nan_to_num(estimate, nan=0.0, posinf=0.0, neginf=0.0)
        estimate = cupy.maximum(estimate, 0.0).astype(cupy.float32, copy=False)
    return estimate


def _measure(func, *, warmups: int, repeats: int, synchronize=None):
    for _ in range(warmups):
        func()
        if synchronize is not None:
            synchronize()
    durations = []
    output = None
    for _ in range(repeats):
        start = time.perf_counter()
        output = func()
        if synchronize is not None:
            synchronize()
        durations.append(time.perf_counter() - start)
    return Measurement(
        median_seconds=float(statistics.median(durations)),
        minimum_seconds=float(min(durations)),
        maximum_seconds=float(max(durations)),
        output=output,
    )


def _to_device(value, cupy):
    if isinstance(value, tuple):
        return tuple(_to_device(item, cupy) for item in value)
    return cupy.asarray(value)


def _to_host(value, cupy):
    if isinstance(value, tuple):
        return tuple(_to_host(item, cupy) for item in value)
    if isinstance(value, list):
        return [_to_host(item, cupy) for item in value]
    return cupy.asnumpy(value)


def _comparison(expected, actual):
    expected_flat = _flatten_outputs(expected)
    actual_flat = _flatten_outputs(actual)
    if len(expected_flat) != len(actual_flat):
        return {"comparable": False, "reason": "output counts differ"}
    max_abs = 0.0
    abs_sum = 0.0
    squared_sum = 0.0
    expected_squared_sum = 0.0
    count = 0
    for left, right in zip(expected_flat, actual_flat, strict=True):
        left = np.asarray(left, dtype=np.float64)
        right = np.asarray(right, dtype=np.float64)
        if left.shape != right.shape:
            return {"comparable": False, "reason": "output shapes differ"}
        difference = np.abs(left - right)
        max_abs = max(max_abs, float(difference.max(initial=0.0)))
        abs_sum += float(difference.sum(dtype=np.float64))
        squared_sum += float(np.square(difference).sum(dtype=np.float64))
        expected_squared_sum += float(np.square(left).sum(dtype=np.float64))
        count += int(left.size)
    return {
        "comparable": True,
        "max_abs_error": max_abs,
        "mean_abs_error": abs_sum / max(count, 1),
        "normalized_rmse": (
            (squared_sum / max(expected_squared_sum, 1e-30)) ** 0.5
        ),
    }


def _flatten_outputs(value):
    if isinstance(value, (tuple, list)):
        flattened = []
        for item in value:
            flattened.extend(_flatten_outputs(item))
        return flattened
    return [value]


def _deconvolution_fixture(shape, rng):
    image = rng.random(shape, dtype=np.float32) * np.float32(0.03)
    center = tuple(size // 2 for size in shape)
    image[center] = 1.0
    slices = tuple(slice(max(value - 1, 0), value + 2) for value in center)
    image[slices] += np.float32(0.25)
    return np.clip(image, 0.0, 1.0).astype(np.float32, copy=False)


def _gaussian_psf(dimensions: int):
    shape = (9,) * dimensions
    coords = np.indices(shape, dtype=np.float32)
    center = np.asarray([(size - 1) / 2 for size in shape], dtype=np.float32)
    squared_radius = np.zeros(shape, dtype=np.float32)
    for axis in range(dimensions):
        squared_radius += (coords[axis] - center[axis]) ** 2
    psf = np.exp(-squared_radius / np.float32(2 * 1.5**2))
    return (psf / psf.sum(dtype=np.float64)).astype(np.float32)


def _shape_description(value):
    if isinstance(value, tuple):
        return [list(np.asarray(item).shape) for item in value]
    return list(np.asarray(value).shape)


def _dtype_description(value):
    if isinstance(value, tuple):
        return [str(np.asarray(item).dtype) for item in value]
    return str(np.asarray(value).dtype)


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else None


def _optional_provider_status(name: str):
    distributions = importlib.metadata.packages_distributions().get(name, ())
    for distribution in distributions:
        try:
            return {
                "installed": True,
                "distribution": distribution,
                "version": importlib.metadata.version(distribution),
            }
        except importlib.metadata.PackageNotFoundError:
            continue
    return {"installed": False, "distribution": "", "version": ""}


def _emit(payload, output_path: Path | None):
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized + "\n", encoding="utf-8")
    sys.stdout.write(serialized + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
