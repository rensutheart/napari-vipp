"""Benchmark a source-built cuCIM against CuPy and CPU scikit-image.

The benchmark deliberately distinguishes two questions:

* Does cuCIM improve an operation that can already be expressed with CuPy?
* Does cuCIM provide a useful GPU implementation where CuPy has no ready API?

GPU timings synchronize the current stream.  Resident timings exclude host/device
copies; end-to-end timings include one input and one output transfer.  The script
is standalone so it can run in an isolated source-build environment without
installing napari-vipp or its GUI dependencies.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class Profile:
    image_2d: tuple[int, int]
    image_3d: tuple[int, int, int]
    median_histogram_2d: tuple[int, int]
    labels_2d: tuple[int, int]
    labels_3d: tuple[int, int, int]
    rl_2d: tuple[int, int]
    rl_3d: tuple[int, int, int]
    rl_iterations: int
    rolling_radius_2d: int
    rolling_radius_3d: int
    warmups: int
    gpu_repeats: int
    cpu_repeats: int


@dataclass(frozen=True, slots=True)
class Measurement:
    median_seconds: float
    minimum_seconds: float
    maximum_seconds: float
    output: Any

    def timing_dict(self) -> dict[str, float]:
        return {
            "median": self.median_seconds,
            "minimum": self.minimum_seconds,
            "maximum": self.maximum_seconds,
        }


PROFILES = {
    "smoke": Profile(
        image_2d=(256, 256),
        image_3d=(8, 64, 64),
        median_histogram_2d=(256, 256),
        labels_2d=(512, 512),
        labels_3d=(12, 96, 96),
        rl_2d=(128, 128),
        rl_3d=(8, 48, 48),
        rl_iterations=5,
        rolling_radius_2d=8,
        rolling_radius_3d=3,
        warmups=1,
        gpu_repeats=2,
        cpu_repeats=1,
    ),
    "standard": Profile(
        image_2d=(1024, 1024),
        image_3d=(32, 192, 192),
        median_histogram_2d=(1024, 1024),
        labels_2d=(2048, 2048),
        labels_3d=(32, 192, 192),
        rl_2d=(512, 512),
        rl_3d=(24, 128, 128),
        rl_iterations=15,
        rolling_radius_2d=15,
        rolling_radius_3d=5,
        warmups=2,
        gpu_repeats=5,
        cpu_repeats=3,
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

    cupy = importlib.import_module("cupy")
    cupyx_ndi = importlib.import_module("cupyx.scipy.ndimage")
    scipy_ndi = importlib.import_module("scipy.ndimage")
    skimage_feature = importlib.import_module("skimage.feature")
    skimage_filters = importlib.import_module("skimage.filters")
    skimage_measure = importlib.import_module("skimage.measure")
    skimage_restoration = importlib.import_module("skimage.restoration")

    cucim = importlib.import_module("cucim")
    cucim_feature = importlib.import_module("cucim.skimage.feature")
    cucim_filters = importlib.import_module("cucim.skimage.filters")
    cucim_measure = importlib.import_module("cucim.skimage.measure")
    cucim_morphology = importlib.import_module("cucim.skimage.morphology")
    cucim_restoration = importlib.import_module("cucim.skimage.restoration")

    probe = cupy.arange(1, dtype=cupy.float32) + 1
    cupy.cuda.get_current_stream().synchronize()
    if int(probe.get()[0]) != 1:
        raise RuntimeError("CuPy real-kernel probe returned an unexpected result")

    rng = np.random.default_rng(20260715)
    image_2d = rng.random(profile.image_2d, dtype=np.float32)
    image_3d = rng.random(profile.image_3d, dtype=np.float32)
    binary_2d = image_2d > np.float32(0.62)
    footprint_5 = cupy.ones((5, 5), dtype=cupy.bool_)

    results: list[dict[str, Any]] = []
    results.append(
        _compare_gpu_implementations(
            "gaussian_2d",
            image_2d,
            baseline_name="cupyx.scipy.ndimage.gaussian_filter",
            baseline_func=lambda value: cupyx_ndi.gaussian_filter(
                value, sigma=2.0, mode="nearest", truncate=4.0
            ),
            candidate_func=lambda value: cucim_filters.gaussian(
                value,
                sigma=2.0,
                mode="nearest",
                preserve_range=True,
                truncate=4.0,
            ),
            cupy=cupy,
            profile=profile,
        )
    )
    results.append(
        _compare_gpu_implementations(
            "gaussian_3d",
            image_3d,
            baseline_name="cupyx.scipy.ndimage.gaussian_filter",
            baseline_func=lambda value: cupyx_ndi.gaussian_filter(
                value,
                sigma=(1.5, 2.0, 2.0),
                mode="nearest",
                truncate=4.0,
            ),
            candidate_func=lambda value: cucim_filters.gaussian(
                value,
                sigma=(1.5, 2.0, 2.0),
                mode="nearest",
                preserve_range=True,
                truncate=4.0,
            ),
            cupy=cupy,
            profile=profile,
        )
    )
    results.append(
        _compare_gpu_implementations(
            "median_sorting_float32_2d",
            image_2d,
            baseline_name="cupyx.scipy.ndimage.median_filter",
            baseline_func=lambda value: cupyx_ndi.median_filter(
                value, footprint=footprint_5, mode="nearest"
            ),
            candidate_func=lambda value: cucim_filters.median(
                value,
                footprint=footprint_5,
                mode="nearest",
                algorithm="sorting",
            ),
            cupy=cupy,
            profile=profile,
        )
    )

    histogram_image = rng.integers(
        0,
        4096,
        size=profile.median_histogram_2d,
        dtype=np.uint16,
    )
    # cuCIM's histogram kernel currently requires a dense rectangular footprint.
    # This matches VIPP's existing square median-filter size contract.
    histogram_footprint = cupy.ones((31, 31), dtype=cupy.bool_)
    results.append(
        _compare_gpu_implementations(
            "median_histogram_uint16_2d",
            histogram_image,
            baseline_name="cupyx.scipy.ndimage.median_filter",
            baseline_func=lambda value: cupyx_ndi.median_filter(
                value, footprint=histogram_footprint, mode="nearest"
            ),
            candidate_func=lambda value: cucim_filters.median(
                value,
                footprint=histogram_footprint,
                mode="nearest",
                algorithm="histogram",
            ),
            cupy=cupy,
            profile=profile,
        )
    )
    results.append(
        _compare_gpu_implementations(
            "sobel_2d",
            image_2d,
            baseline_name="CuPy composition over cupyx.scipy.ndimage.sobel",
            baseline_func=lambda value: _cupy_sobel(value, cupy, cupyx_ndi),
            candidate_func=lambda value: cucim_filters.sobel(
                value, mode="reflect"
            ),
            cupy=cupy,
            profile=profile,
        )
    )
    results.append(
        _compare_gpu_implementations(
            "binary_closing_2d",
            binary_2d,
            baseline_name="cupyx.scipy.ndimage.binary_closing",
            baseline_func=lambda value: cupyx_ndi.binary_closing(
                value, structure=footprint_5, border_value=0
            ),
            candidate_func=lambda value: cucim_morphology.binary_closing(
                value, footprint=footprint_5, mode="min"
            ),
            cupy=cupy,
            profile=profile,
        )
    )

    for dimensions, shape in ((2, profile.rl_2d), (3, profile.rl_3d)):
        rl_image = _deconvolution_fixture(shape, rng, scipy_ndi)
        psf = _gaussian_psf(dimensions)
        results.append(
            _compare_gpu_implementations(
                f"richardson_lucy_{dimensions}d",
                (rl_image, psf),
                baseline_name="CuPy composition over cupyx.scipy.ndimage.convolve",
                baseline_func=lambda pair: _cupy_richardson_lucy(
                    pair,
                    iterations=profile.rl_iterations,
                    cupy=cupy,
                    cupyx_ndi=cupyx_ndi,
                ),
                candidate_func=lambda pair: cucim_restoration.richardson_lucy(
                    pair[0],
                    pair[1],
                    num_iter=profile.rl_iterations,
                    clip=False,
                ),
                cupy=cupy,
                profile=profile,
            )
        )

    results.extend(
        _coverage_results(
            image_2d=image_2d,
            image_3d=image_3d,
            rng=rng,
            cupy=cupy,
            profile=profile,
            skimage_feature=skimage_feature,
            skimage_filters=skimage_filters,
            skimage_measure=skimage_measure,
            skimage_restoration=skimage_restoration,
            cucim_feature=cucim_feature,
            cucim_filters=cucim_filters,
            cucim_measure=cucim_measure,
            cucim_restoration=cucim_restoration,
        )
    )

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "completed",
        "profile": args.profile,
        "profile_definition": asdict(profile),
        "environment": _environment(cupy, cucim),
        "methodology": {
            "time_statistic": "median synchronized wall-clock seconds",
            "warmups": profile.warmups,
            "resident_gpu": "input already on device; output remains on device",
            "end_to_end_gpu": "one host-to-device and one device-to-host transfer",
            "cold_call": (
                "first synchronized call in the process; the persistent CuPy "
                "compiler cache is not cleared, so this is not clean-install JIT "
                "latency"
            ),
            "comparison_interpretation": (
                "speedup > 1 means cuCIM is faster than the named baseline"
            ),
            "random_seed": 20260715,
        },
        "results": results,
    }
    _emit(payload, args.output)
    return 0


def _coverage_results(
    *,
    image_2d,
    image_3d,
    rng,
    cupy,
    profile,
    skimage_feature,
    skimage_filters,
    skimage_measure,
    skimage_restoration,
    cucim_feature,
    cucim_filters,
    cucim_measure,
    cucim_restoration,
) -> list[dict[str, Any]]:
    results = []
    for dimensions, image, radius in (
        (2, image_2d, profile.rolling_radius_2d),
        (3, image_3d, profile.rolling_radius_3d),
    ):
        results.append(
            _compare_cpu_to_cucim(
                f"rolling_ball_{dimensions}d",
                image,
                baseline_name="skimage.restoration.rolling_ball",
                baseline_func=lambda value, r=radius: skimage_restoration.rolling_ball(
                    value, radius=r
                ),
                candidate_func=lambda value, r=radius: cucim_restoration.rolling_ball(
                    value, radius=r
                ),
                cupy=cupy,
                profile=profile,
            )
        )

    label_2d = rng.random(profile.labels_2d, dtype=np.float32) > np.float32(0.72)
    label_3d = rng.random(profile.labels_3d, dtype=np.float32) > np.float32(0.82)
    for dimensions, binary in ((2, label_2d), (3, label_3d)):
        results.append(
            _compare_cpu_to_cucim(
                f"connected_components_{dimensions}d",
                binary,
                baseline_name="skimage.measure.label",
                baseline_func=lambda value: skimage_measure.label(
                    value, connectivity=1
                ),
                candidate_func=lambda value: cucim_measure.label(
                    value, connectivity=1
                ),
                cupy=cupy,
                profile=profile,
            )
        )

    results.append(
        _compare_cpu_to_cucim(
            "canny_2d",
            image_2d,
            baseline_name="skimage.feature.canny",
            baseline_func=lambda value: skimage_feature.canny(
                value, sigma=2.0, mode="constant"
            ),
            candidate_func=lambda value: cucim_feature.canny(
                value, sigma=2.0, mode="constant"
            ),
            cupy=cupy,
            profile=profile,
        )
    )
    results.append(
        _compare_cpu_to_cucim(
            "threshold_otsu_2d",
            image_2d,
            baseline_name="skimage.filters.threshold_otsu",
            baseline_func=skimage_filters.threshold_otsu,
            candidate_func=cucim_filters.threshold_otsu,
            cupy=cupy,
            profile=profile,
        )
    )

    region_labels = _regionprops_labels(profile.labels_2d)
    properties = ("label", "area", "bbox", "centroid")
    results.append(
        _compare_cpu_to_cucim(
            "regionprops_table_2d",
            region_labels,
            baseline_name="skimage.measure.regionprops_table",
            baseline_func=lambda value: skimage_measure.regionprops_table(
                value, properties=properties
            ),
            candidate_func=lambda value: cucim_measure.regionprops_table(
                value, properties=properties
            ),
            cupy=cupy,
            profile=profile,
        )
    )
    return results


def _compare_gpu_implementations(
    workload: str,
    host_input,
    *,
    baseline_name: str,
    baseline_func: Callable,
    candidate_func: Callable,
    cupy,
    profile: Profile,
) -> dict[str, Any]:
    synchronize = cupy.cuda.get_current_stream().synchronize
    device_input = _to_device(host_input, cupy)

    baseline_cold = _measure(
        lambda: baseline_func(device_input),
        warmups=0,
        repeats=1,
        synchronize=synchronize,
    )
    candidate_cold = _measure(
        lambda: candidate_func(device_input),
        warmups=0,
        repeats=1,
        synchronize=synchronize,
    )
    baseline = _measure(
        lambda: baseline_func(device_input),
        warmups=profile.warmups,
        repeats=profile.gpu_repeats,
        synchronize=synchronize,
    )
    candidate = _measure(
        lambda: candidate_func(device_input),
        warmups=profile.warmups,
        repeats=profile.gpu_repeats,
        synchronize=synchronize,
    )

    def baseline_e2e():
        value = _to_device(host_input, cupy)
        return _to_host(baseline_func(value), cupy)

    def candidate_e2e():
        value = _to_device(host_input, cupy)
        return _to_host(candidate_func(value), cupy)

    baseline_end_to_end = _measure(
        baseline_e2e,
        warmups=profile.warmups,
        repeats=profile.gpu_repeats,
        synchronize=synchronize,
    )
    candidate_end_to_end = _measure(
        candidate_e2e,
        warmups=profile.warmups,
        repeats=profile.gpu_repeats,
        synchronize=synchronize,
    )
    return {
        "workload": workload,
        "comparison_class": "cuCIM optimization versus ready CuPy path",
        "baseline": baseline_name,
        "shape": _shape_description(host_input),
        "dtype": _dtype_description(host_input),
        "cold_seconds": {
            "baseline": baseline_cold.median_seconds,
            "cucim": candidate_cold.median_seconds,
        },
        "resident_seconds": {
            "baseline": baseline.median_seconds,
            "cucim": candidate.median_seconds,
        },
        "end_to_end_seconds": {
            "baseline": baseline_end_to_end.median_seconds,
            "cucim": candidate_end_to_end.median_seconds,
        },
        "cucim_speedup": {
            "resident": _ratio(baseline.median_seconds, candidate.median_seconds),
            "end_to_end": _ratio(
                baseline_end_to_end.median_seconds,
                candidate_end_to_end.median_seconds,
            ),
        },
        "timing_ranges_seconds": {
            "baseline_resident": baseline.timing_dict(),
            "cucim_resident": candidate.timing_dict(),
            "baseline_end_to_end": baseline_end_to_end.timing_dict(),
            "cucim_end_to_end": candidate_end_to_end.timing_dict(),
        },
        "comparison": _comparison(
            _to_host(baseline.output, cupy),
            _to_host(candidate.output, cupy),
        ),
    }


def _compare_cpu_to_cucim(
    workload: str,
    host_input,
    *,
    baseline_name: str,
    baseline_func: Callable,
    candidate_func: Callable,
    cupy,
    profile: Profile,
) -> dict[str, Any]:
    synchronize = cupy.cuda.get_current_stream().synchronize
    device_input = _to_device(host_input, cupy)
    candidate_cold = _measure(
        lambda: candidate_func(device_input),
        warmups=0,
        repeats=1,
        synchronize=synchronize,
    )
    baseline = _measure(
        lambda: baseline_func(host_input),
        warmups=min(profile.warmups, 1),
        repeats=profile.cpu_repeats,
    )
    candidate = _measure(
        lambda: candidate_func(device_input),
        warmups=profile.warmups,
        repeats=profile.gpu_repeats,
        synchronize=synchronize,
    )

    def candidate_e2e():
        value = _to_device(host_input, cupy)
        return _to_host(candidate_func(value), cupy)

    candidate_end_to_end = _measure(
        candidate_e2e,
        warmups=profile.warmups,
        repeats=profile.gpu_repeats,
        synchronize=synchronize,
    )
    return {
        "workload": workload,
        "comparison_class": "cuCIM coverage versus CPU; no ready CuPy API",
        "baseline": baseline_name,
        "shape": _shape_description(host_input),
        "dtype": _dtype_description(host_input),
        "cold_seconds": {"cucim": candidate_cold.median_seconds},
        "resident_seconds": {
            "baseline": baseline.median_seconds,
            "cucim": candidate.median_seconds,
        },
        "end_to_end_seconds": {
            "baseline": baseline.median_seconds,
            "cucim": candidate_end_to_end.median_seconds,
        },
        "cucim_speedup": {
            "resident": _ratio(baseline.median_seconds, candidate.median_seconds),
            "end_to_end": _ratio(
                baseline.median_seconds,
                candidate_end_to_end.median_seconds,
            ),
        },
        "timing_ranges_seconds": {
            "baseline": baseline.timing_dict(),
            "cucim_resident": candidate.timing_dict(),
            "cucim_end_to_end": candidate_end_to_end.timing_dict(),
        },
        "comparison": _comparison(
            baseline.output,
            _to_host(candidate.output, cupy),
        ),
    }


def _measure(
    func: Callable[[], Any],
    *,
    warmups: int,
    repeats: int,
    synchronize: Callable[[], None] | None = None,
) -> Measurement:
    output = None
    for _ in range(warmups):
        output = func()
        if synchronize is not None:
            synchronize()

    durations = []
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


def _cupy_sobel(image, cupy, cupyx_ndi):
    squared = cupy.zeros(image.shape, dtype=cupy.float32)
    for axis in range(image.ndim):
        # scipy.ndimage's raw Sobel kernel is not normalized; scikit-image and
        # cuCIM divide the per-axis response by 2**ndim.
        edge = (
            cupyx_ndi.sobel(image, axis=axis, mode="reflect") / 2**image.ndim
        )
        squared += edge * edge
    return cupy.sqrt(squared / image.ndim)


def _cupy_richardson_lucy(pair, *, iterations, cupy, cupyx_ndi):
    image, psf = pair
    image = image.astype(cupy.float32, copy=False)
    psf = psf.astype(cupy.float32, copy=False)
    estimate = cupy.full(image.shape, 0.5, dtype=cupy.float32)
    mirror = cupy.ascontiguousarray(psf[(slice(None, None, -1),) * psf.ndim])
    for _ in range(iterations):
        convolved = cupyx_ndi.convolve(estimate, psf, mode="constant") + 1e-12
        estimate *= cupyx_ndi.convolve(
            image / convolved, mirror, mode="constant"
        )
    return estimate


def _deconvolution_fixture(shape, rng, scipy_ndi):
    impulse = np.zeros(shape, dtype=np.float32)
    count = max(8, int(np.prod(shape) ** 0.25))
    for _ in range(count):
        index = tuple(int(rng.integers(0, size)) for size in shape)
        impulse[index] = np.float32(rng.uniform(0.3, 1.0))
    blurred = scipy_ndi.gaussian_filter(impulse, sigma=1.2, mode="constant")
    noise = rng.normal(0.0, 0.001, size=shape).astype(np.float32)
    return np.maximum(blurred + noise, 0.0).astype(np.float32)


def _gaussian_psf(dimensions: int) -> np.ndarray:
    shape = (9,) * dimensions if dimensions == 2 else (5,) * dimensions
    coordinates = np.meshgrid(
        *(np.arange(size, dtype=np.float32) - size // 2 for size in shape),
        indexing="ij",
    )
    squared_radius = sum(axis * axis for axis in coordinates)
    psf = np.exp(-squared_radius / np.float32(2.0 * 1.2**2))
    return (psf / psf.sum()).astype(np.float32)


def _regionprops_labels(shape: tuple[int, int]) -> np.ndarray:
    y, x = np.indices(shape, dtype=np.int32)
    cell = 32
    object_ids = (y // cell) * ((shape[1] + cell - 1) // cell) + (x // cell) + 1
    mask = ((y % cell) < 16) & ((x % cell) < 16)
    return np.where(mask, object_ids, 0).astype(np.int32)


def _to_device(value, cupy):
    if isinstance(value, tuple):
        return tuple(_to_device(item, cupy) for item in value)
    if isinstance(value, list):
        return [_to_device(item, cupy) for item in value]
    if isinstance(value, dict):
        return {key: _to_device(item, cupy) for key, item in value.items()}
    return cupy.asarray(value)


def _to_host(value, cupy):
    if isinstance(value, tuple):
        return tuple(_to_host(item, cupy) for item in value)
    if isinstance(value, list):
        return [_to_host(item, cupy) for item in value]
    if isinstance(value, dict):
        return {key: _to_host(item, cupy) for key, item in value.items()}
    if isinstance(value, (cupy.ndarray, cupy.generic)):
        return cupy.asnumpy(value)
    return value


def _comparison(reference, candidate) -> dict[str, Any]:
    reference_flat = _flatten_numeric(reference)
    candidate_flat = _flatten_numeric(candidate)
    schemas = {
        "reference_schema": _value_schema(reference),
        "candidate_schema": _value_schema(candidate),
    }
    if reference_flat.shape != candidate_flat.shape:
        return {
            **schemas,
            "shape_match": False,
            "reference_flat_shape": list(reference_flat.shape),
            "candidate_flat_shape": list(candidate_flat.shape),
        }
    if reference_flat.size == 0:
        return {**schemas, "shape_match": True, "value_count": 0}
    reference_float = reference_flat.astype(np.float64, copy=False)
    candidate_float = candidate_flat.astype(np.float64, copy=False)
    difference = candidate_float - reference_float
    data_range = float(np.ptp(reference_float))
    denominator = data_range if data_range > 0 else 1.0
    return {
        **schemas,
        "shape_match": True,
        "value_count": int(reference_flat.size),
        "exact": bool(np.array_equal(reference_flat, candidate_flat)),
        "mismatch_fraction": float(np.mean(reference_flat != candidate_flat)),
        "maximum_absolute_error": float(np.max(np.abs(difference))),
        "normalized_rmse": float(
            np.sqrt(np.mean(difference * difference)) / denominator
        ),
        "allclose_rtol_1e-4_atol_1e-6": bool(
            np.allclose(reference_float, candidate_float, rtol=1e-4, atol=1e-6)
        ),
    }


def _value_schema(value):
    if isinstance(value, dict):
        return {key: _value_schema(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_value_schema(item) for item in value]
    if isinstance(value, list):
        return [_value_schema(item) for item in value]
    array = np.asarray(value)
    return {"shape": list(array.shape), "dtype": str(array.dtype)}


def _flatten_numeric(value) -> np.ndarray:
    if isinstance(value, dict):
        arrays = [_flatten_numeric(value[key]) for key in sorted(value)]
    elif isinstance(value, (tuple, list)):
        arrays = [_flatten_numeric(item) for item in value]
    else:
        return np.asarray(value).reshape(-1)
    if not arrays:
        return np.asarray([], dtype=np.float64)
    return np.concatenate(arrays)


def _shape_description(value):
    if isinstance(value, tuple):
        return [_shape_description(item) for item in value]
    return list(np.asarray(value).shape)


def _dtype_description(value):
    if isinstance(value, tuple):
        return [_dtype_description(item) for item in value]
    return str(np.asarray(value).dtype)


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def _environment(cupy, cucim) -> dict[str, Any]:
    properties = cupy.cuda.runtime.getDeviceProperties(0)
    device_name = properties["name"]
    if isinstance(device_name, bytes):
        device_name = device_name.decode(errors="replace")
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "packages": {
            name: _package_version(name)
            for name in (
                "numpy",
                "scipy",
                "scikit-image",
                "cupy-cuda13x",
                "cucim-cu13",
            )
        },
        "cucim_version": str(cucim.__version__),
        "cucim_skimage_available": bool(cucim.is_available("skimage")),
        "cucim_clara_available": bool(cucim.is_available("clara")),
        "gpu": {
            "name": device_name,
            "compute_capability": cupy.cuda.Device(0).compute_capability,
            "memory_bytes": int(properties["totalGlobalMem"]),
            "driver_version": int(cupy.cuda.runtime.driverGetVersion()),
            "runtime_version": int(cupy.cuda.runtime.runtimeGetVersion()),
        },
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        check=False,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _emit(payload: dict[str, Any], output: Path | None) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    sys.exit(main())
