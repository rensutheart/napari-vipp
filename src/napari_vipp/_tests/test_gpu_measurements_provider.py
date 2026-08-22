from __future__ import annotations

import importlib
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from napari_vipp.core.gpu import cupy_measurements as provider
from napari_vipp.core.measurements import (
    basic_measurement_layout,
    finalize_basic_measurement_table,
    measurement_table_parity,
)
from napari_vipp.core.operations import (
    measure_objects as cpu_measure_objects,
)
from napari_vipp.core.operations import (
    measure_objects_with_intensity as cpu_measure_objects_with_intensity,
)

SOURCE_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def cupy_module():
    try:
        cupy = importlib.import_module("cupy")
        importlib.import_module("cupyx.scipy.ndimage")
        if int(cupy.cuda.runtime.getDeviceCount()) < 1:
            pytest.skip("No CUDA device is available.")
        cupy.zeros(1, dtype=cupy.uint8).sum().item()
    except Exception as exc:
        pytest.skip(f"A working CuPy CUDA environment is unavailable: {exc}")
    return cupy


def _labels_2d() -> np.ndarray:
    labels = np.zeros((2, 37, 41), dtype=np.int32)
    labels[0, 2:17, 3:19] = 17
    labels[0, 6:12, 8:15] = 0
    labels[0, 21:29, 27:36] = 9001
    labels[1, 1:10, 2:7] = 9001
    labels[1, 13:31, 12:34] = 17
    labels[1, 18:25, 18:27] = 0
    labels[1, 33:36, 36:40] = 40000
    return labels


def _labels_3d() -> np.ndarray:
    labels = np.zeros((2, 15, 17, 19), dtype=np.int32)
    labels[0, 1:10, 2:13, 3:15] = 71
    labels[0, 3:8, 5:10, 6:12] = 0
    labels[0, 11:14, 1:5, 13:18] = 9001
    labels[1, 1:8, 2:9, 2:10] = 9001
    labels[1, 3:6, 4:7, 4:8] = 0
    labels[1, 9:14, 10:16, 11:18] = 71
    return labels


def _finalize(packed, shape, *, include_intensity: bool, kwargs, cupy):
    layout = basic_measurement_layout(
        shape,
        include_intensity=include_intensity,
        **{
            name: value
            for name, value in kwargs.items()
            if name not in {"measurement_set", "source_name"}
        },
    )
    return finalize_basic_measurement_table(
        cupy.asnumpy(packed),
        layout=layout,
        measurement_set=kwargs.get("measurement_set"),
        source_name=kwargs.get("source_name", ""),
    )


def test_provider_module_import_is_cuda_lazy_in_a_fresh_process() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(SOURCE_ROOT), environment.get("PYTHONPATH", "")))
    )
    script = r"""
import sys
import napari_vipp.core.gpu.cupy_measurements as module
assert module.__all__ == ["measure_objects", "measure_objects_with_intensity"]
for name in ("cupy", "cupyx", "cucim"):
    assert name not in sys.modules, name
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    ("labels", "spatial_mode", "axis_names", "axis_types"),
    (
        (_labels_2d(), "2D YX", ("t", "y", "x"), ("time", "space", "space")),
        (
            _labels_3d(),
            "3D ZYX",
            ("t", "z", "y", "x"),
            ("time", "space", "space", "space"),
        ),
    ),
)
def test_real_gpu_basic_morphology_is_resident_and_cpu_equivalent(
    cupy_module,
    labels,
    spatial_mode,
    axis_names,
    axis_types,
) -> None:
    kwargs = {
        "spatial_mode": spatial_mode,
        "axis_names": axis_names,
        "axis_types": axis_types,
        "axis_scales": tuple(
            1.0 if name == "t" else 0.25 + index * 0.125
            for index, name in enumerate(axis_names)
        ),
        "axis_units": tuple(None if name == "t" else "um" for name in axis_names),
        "measurement_set": "Basic morphology",
        "source_name": "gpu-morphology-fixture",
    }
    expected = cpu_measure_objects(labels, **kwargs)
    device_labels = cupy_module.asarray(labels)
    before = device_labels.copy()

    packed = provider.measure_objects(device_labels, **kwargs)
    actual = _finalize(
        packed,
        labels.shape,
        include_intensity=False,
        kwargs=kwargs,
        cupy=cupy_module,
    )

    assert isinstance(packed, cupy_module.ndarray)
    assert packed.dtype == cupy_module.float64
    assert packed.flags.c_contiguous
    cupy_module.testing.assert_array_equal(device_labels, before)
    result = measurement_table_parity(expected, actual)
    assert result.passed, result.detail


@pytest.mark.parametrize("spatial_ndim", (2, 3))
def test_real_gpu_exhaustive_local_euler_configurations_match_cpu(
    cupy_module,
    spatial_ndim,
) -> None:
    spatial_shape = (2,) * spatial_ndim
    configuration_count = 2 ** (2**spatial_ndim)
    labels = np.zeros((configuration_count, *spatial_shape), dtype=np.int32)
    for configuration in range(configuration_count):
        bits = (
            (configuration >> np.arange(2**spatial_ndim, dtype=np.uint16)) & 1
        ).astype(bool)
        labels[configuration].reshape(-1)[bits] = 1
    spatial_mode = "2D YX" if spatial_ndim == 2 else "3D ZYX"
    kwargs = {"spatial_mode": spatial_mode}
    expected = cpu_measure_objects(labels, **kwargs)

    packed = provider.measure_objects(cupy_module.asarray(labels), **kwargs)
    actual = _finalize(
        packed,
        labels.shape,
        include_intensity=False,
        kwargs=kwargs,
        cupy=cupy_module,
    )

    result = measurement_table_parity(expected, actual)
    assert result.passed, result.detail


@pytest.mark.parametrize("spatial_ndim", (2, 3))
def test_real_gpu_touching_labels_have_independent_euler_numbers(
    cupy_module,
    spatial_ndim,
) -> None:
    if spatial_ndim == 2:
        coordinates = np.indices((9, 11))
        labels = ((coordinates[0] + coordinates[1]) % 3 + 1).astype(np.int32)
        labels[2:7, 3:8] = 0
        spatial_mode = "2D YX"
    else:
        coordinates = np.indices((7, 8, 9))
        labels = ((coordinates[0] + coordinates[1] + coordinates[2]) % 3 + 1).astype(
            np.int32
        )
        labels[2:5, 2:6, 3:7] = 0
        spatial_mode = "3D ZYX"
    kwargs = {"spatial_mode": spatial_mode}
    expected = cpu_measure_objects(labels, **kwargs)

    packed = provider.measure_objects(cupy_module.asarray(labels), **kwargs)
    actual = _finalize(
        packed,
        labels.shape,
        include_intensity=False,
        kwargs=kwargs,
        cupy=cupy_module,
    )

    result = measurement_table_parity(expected, actual)
    assert result.passed, result.detail


@pytest.mark.parametrize("dtype", (np.bool_, np.uint8, np.uint16, np.float32))
def test_real_gpu_intensity_types_and_sparse_ids_match_cpu(
    cupy_module,
    dtype,
) -> None:
    labels = _labels_2d()
    rng = np.random.default_rng(20260804)
    if dtype is np.bool_:
        intensity = rng.random(labels.shape) > 0.45
    elif dtype is np.float32:
        intensity = (1.0e7 + rng.normal(0.0, 64.0, labels.shape)).astype(dtype)
    else:
        intensity = rng.integers(
            0,
            np.iinfo(dtype).max + 1,
            size=labels.shape,
            dtype=dtype,
        )
    kwargs = {
        "spatial_mode": "2D YX",
        "axis_names": ("t", "y", "x"),
        "axis_types": ("time", "space", "space"),
        "measurement_set": "Basic morphology + intensity",
        "source_name": "gpu-intensity-fixture",
    }
    expected = cpu_measure_objects_with_intensity([labels, intensity], **kwargs)

    packed = provider.measure_objects_with_intensity(
        [cupy_module.asarray(labels), cupy_module.asarray(intensity)],
        **kwargs,
    )
    actual = _finalize(
        packed,
        labels.shape,
        include_intensity=True,
        kwargs=kwargs,
        cupy=cupy_module,
    )

    result = measurement_table_parity(
        expected,
        actual,
        intensity_dtype=np.dtype(dtype),
    )
    assert result.passed, result.detail


@pytest.mark.parametrize("fixture_name", ("high-offset", "dynamic-cancellation"))
def test_real_gpu_float32_two_pass_adversarial_reductions(
    cupy_module,
    fixture_name,
) -> None:
    size = 1024
    labels = np.ones((size, size), dtype=np.int32)
    if fixture_name == "high-offset":
        rng = np.random.default_rng(1947)
        intensity = np.float32(1.0e8) + rng.normal(0.0, 32.0, labels.shape).astype(
            np.float32
        )
    else:
        intensity = np.empty(labels.shape, dtype=np.float32)
        flat = intensity.reshape(-1)
        flat[0::4] = np.float32(1.0e20)
        flat[1::4] = np.float32(1.0)
        flat[2::4] = np.float32(-1.0e20)
        flat[3::4] = np.float32(-1.0)
    kwargs = {"spatial_mode": "2D YX"}
    expected = cpu_measure_objects_with_intensity([labels, intensity], **kwargs)

    packed = provider.measure_objects_with_intensity(
        [cupy_module.asarray(labels), cupy_module.asarray(intensity)],
        **kwargs,
    )
    actual = _finalize(
        packed,
        labels.shape,
        include_intensity=True,
        kwargs=kwargs,
        cupy=cupy_module,
    )

    result = measurement_table_parity(
        expected,
        actual,
        intensity_dtype="float32",
    )
    assert result.passed, result.detail


def test_real_gpu_singleton_population_standard_deviation_is_zero(
    cupy_module,
) -> None:
    labels = np.zeros((7, 9), dtype=np.int32)
    labels[1, 1] = 101
    labels[1, 2] = 202
    labels[1, 3] = 303
    labels[4:6, 5:8] = 404
    labels[6, 8] = 505
    intensity = np.arange(labels.size, dtype=np.uint16).reshape(labels.shape)
    kwargs = {"spatial_mode": "2D YX"}
    expected = cpu_measure_objects_with_intensity([labels, intensity], **kwargs)

    packed = provider.measure_objects_with_intensity(
        [cupy_module.asarray(labels), cupy_module.asarray(intensity)],
        **kwargs,
    )
    actual = _finalize(
        packed,
        labels.shape,
        include_intensity=True,
        kwargs=kwargs,
        cupy=cupy_module,
    )

    result = measurement_table_parity(
        expected,
        actual,
        intensity_dtype="uint16",
    )
    assert result.passed, result.detail
    label_index = actual.columns.index("label_id")
    std_index = actual.columns.index("intensity_std")
    std_by_label = {row[label_index]: row[std_index] for row in actual.rows}
    assert std_by_label[101] == 0.0
    assert std_by_label[202] == 0.0
    assert std_by_label[303] == 0.0
    assert std_by_label[505] == 0.0


def test_real_gpu_adjacent_short_group_variance_matches_cpu(cupy_module) -> None:
    group_count = 512
    labels = np.zeros((32, 64), dtype=np.int32)
    intensity = np.zeros(labels.shape, dtype=np.uint16)
    flat_labels = labels.reshape(-1)
    flat_intensity = intensity.reshape(-1)
    for label_id in range(1, group_count + 1):
        start = (label_id - 1) * 3
        flat_labels[start : start + 3] = label_id
        flat_intensity[start : start + 3] = (67, 67, 68)
    kwargs = {"spatial_mode": "2D YX"}
    expected = cpu_measure_objects_with_intensity([labels, intensity], **kwargs)

    packed = provider.measure_objects_with_intensity(
        [cupy_module.asarray(labels), cupy_module.asarray(intensity)],
        **kwargs,
    )
    actual = _finalize(
        packed,
        labels.shape,
        include_intensity=True,
        kwargs=kwargs,
        cupy=cupy_module,
    )

    result = measurement_table_parity(
        expected,
        actual,
        intensity_dtype="uint16",
    )
    assert result.passed, result.detail
    std_index = actual.columns.index("intensity_std")
    expected_std = float(np.std(np.asarray((67, 67, 68), dtype=np.float64)))
    assert all(row[std_index] == expected_std for row in actual.rows)


def test_real_gpu_compacts_the_full_positive_int32_id_range(cupy_module) -> None:
    labels = np.zeros((13, 17), dtype=np.int32)
    labels[1:4, 2:6] = np.iinfo(np.int32).max
    labels[8:12, 10:16] = 17
    packed = provider.measure_objects(
        cupy_module.asarray(labels),
        spatial_mode="2D YX",
    )
    actual = _finalize(
        packed,
        labels.shape,
        include_intensity=False,
        kwargs={"spatial_mode": "2D YX"},
        cupy=cupy_module,
    )

    label_index = actual.columns.index("label_id")
    assert [row[label_index] for row in actual.rows] == [17, np.iinfo(np.int32).max]


@dataclass
class _Progress:
    cancel_after: int | None = None
    cancelled: bool = False
    reports: list[tuple[int, int, str]] = field(default_factory=list)

    def report(self, current: int, total: int, message: str) -> None:
        self.reports.append((current, total, message))
        if current == self.cancel_after:
            self.cancelled = True

    def check_cancelled(self) -> None:
        if self.cancelled:
            raise RuntimeError("cancelled")


def test_real_gpu_progress_has_truthful_intra_block_stages_and_cancellation(
    cupy_module,
) -> None:
    labels = _labels_2d()
    intensity = np.arange(labels.size, dtype=np.uint16).reshape(labels.shape)
    progress = _Progress()

    provider.measure_objects_with_intensity(
        [cupy_module.asarray(labels), cupy_module.asarray(intensity)],
        spatial_mode="2D YX",
        progress=progress,
    )

    assert [(current, total) for current, total, _message in progress.reports] == [
        (current, 13) for current in range(14)
    ]
    messages = " ".join(message for _current, _total, message in progress.reports)
    assert "compacting labels" in messages
    assert "measuring morphology" in messages
    assert "measuring topology" in messages
    assert "intensity ranges and means" in messages
    assert "intensity variation" in messages
    assert "assembling packed table" in messages

    cancelled = _Progress(cancel_after=1)
    with pytest.raises(RuntimeError, match="cancelled"):
        provider.measure_objects(
            cupy_module.asarray(labels[:1]),
            spatial_mode="2D YX",
            progress=cancelled,
        )
    assert [current for current, _total, _message in cancelled.reports] == [0, 1]


def test_real_gpu_rejects_unsupported_regions_before_measurement(cupy_module) -> None:
    labels = np.zeros((9, 11), dtype=np.int32)
    with pytest.raises(ValueError, match="native int32"):
        provider.measure_objects(
            cupy_module.asarray(labels, dtype=cupy_module.uint16),
            spatial_mode="2D YX",
        )
    negative = labels.copy()
    negative[0, 0] = -1
    with pytest.raises(ValueError, match="non-negative"):
        provider.measure_objects(
            cupy_module.asarray(negative),
            spatial_mode="2D YX",
        )
    with pytest.raises(ValueError, match="finite"):
        provider.measure_objects_with_intensity(
            [
                cupy_module.asarray(labels),
                cupy_module.full(
                    labels.shape,
                    cupy_module.nan,
                    dtype=cupy_module.float32,
                ),
            ],
            spatial_mode="2D YX",
        )
    with pytest.raises(ValueError, match="same shape"):
        provider.measure_objects_with_intensity(
            [
                cupy_module.asarray(labels),
                cupy_module.zeros((3, 4), dtype=cupy_module.uint8),
            ],
            spatial_mode="2D YX",
        )
    with pytest.raises(ValueError, match="include_shape_descriptors"):
        provider.measure_objects(
            cupy_module.asarray(labels),
            spatial_mode="2D YX",
            include_shape_descriptors=True,
        )


def test_provider_has_no_cucim_runtime_hook() -> None:
    assert not hasattr(provider, "_validated_cucim_modules")
