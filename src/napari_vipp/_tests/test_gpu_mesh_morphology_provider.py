from __future__ import annotations

import importlib
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from napari_vipp.core.gpu import cupy_mesh_morphology as provider
from napari_vipp.core.measurements import measurement_table_parity
from napari_vipp.core.mesh_measurements import (
    MESH_ENCODING_SPARSE_UINT32,
    MESH_PAYLOAD_HEADER_BYTES,
    finalize_mesh_morphology_table,
    mesh_morphology_layout,
)
from napari_vipp.core.operations import measure_3d_mesh_morphology as cpu_measure

SOURCE_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def cupy_module():
    try:
        cupy = importlib.import_module("cupy")
        if int(cupy.cuda.runtime.getDeviceCount()) < 1:
            pytest.skip("No CUDA device is available.")
        cupy.zeros(1, dtype=cupy.uint8).sum().item()
    except Exception as exc:
        pytest.skip(f"A working CuPy CUDA environment is unavailable: {exc}")
    return cupy


def _labels() -> np.ndarray:
    labels = np.zeros((2, 16, 18, 20), dtype=np.int32)
    labels[0, 1:11, 2:14, 3:16] = 71
    labels[0, 4:8, 5:10, 7:12] = 0
    labels[0, 12:15, 13:17, 15:19] = 9001
    labels[1, 2:13, 3:15, 4:17] = np.iinfo(np.int32).max
    labels[1, 14:16, 16:18, 17:20] = 33
    return labels


def _kwargs() -> dict[str, object]:
    return {
        "spatial_mode": "3D ZYX",
        "axis_names": ("t", "z", "y", "x"),
        "axis_types": ("time", "space", "space", "space"),
        "axis_scales": (1.0, 2.0, 0.5, 0.25),
        "axis_units": (None, "um", "um", "um"),
        "minimum_voxel_count": 16,
        "include_convex_hull_metrics": True,
        "source_name": "gpu-mesh-fixture",
    }


def _finalize(packed, shape, kwargs, cupy):
    layout = mesh_morphology_layout(
        shape,
        **{
            name: value
            for name, value in kwargs.items()
            if name not in {"minimum_voxel_count", "source_name"}
        },
    )
    return finalize_mesh_morphology_table(
        cupy.asnumpy(packed),
        layout=layout,
        minimum_voxel_count=kwargs.get("minimum_voxel_count", 16),
        source_name=kwargs.get("source_name", ""),
    )


def test_provider_module_import_is_cuda_lazy_in_a_fresh_process() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(SOURCE_ROOT), environment.get("PYTHONPATH", "")))
    )
    script = r"""
import sys
import napari_vipp.core.gpu.cupy_mesh_morphology as module
assert module.__all__ == ["measure_3d_mesh_morphology"]
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


@pytest.mark.parametrize("include_hull", (False, True))
def test_real_gpu_payload_is_resident_input_preserving_and_cpu_equivalent(
    cupy_module,
    include_hull: bool,
) -> None:
    labels = _labels()
    kwargs = {**_kwargs(), "include_convex_hull_metrics": include_hull}
    expected = cpu_measure(labels, **kwargs)
    device_labels = cupy_module.asarray(labels)
    before = device_labels.copy()

    packed = provider.measure_3d_mesh_morphology(device_labels, **kwargs)
    actual = _finalize(packed, labels.shape, kwargs, cupy_module)

    assert isinstance(packed, cupy_module.ndarray)
    assert packed.dtype == cupy_module.uint8
    assert packed.ndim == 1
    assert packed.flags.c_contiguous
    cupy_module.testing.assert_array_equal(device_labels, before)
    parity = measurement_table_parity(expected, actual, exact_float_columns=True)
    assert parity.passed, parity.detail


def test_real_gpu_mask_payload_omits_every_below_threshold_object(cupy_module) -> None:
    labels = np.zeros((8, 9, 10), dtype=np.int32)
    labels[1:3, 1:3, 1:3] = 9
    labels[5:7, 6:8, 7:9] = 77
    kwargs = {
        "spatial_mode": "3D ZYX",
        "minimum_voxel_count": 9,
        "include_convex_hull_metrics": False,
    }

    packed = provider.measure_3d_mesh_morphology(
        cupy_module.asarray(labels),
        **kwargs,
    )
    layout = mesh_morphology_layout(
        labels.shape,
        spatial_mode="3D ZYX",
        include_convex_hull_metrics=False,
    )

    assert int(packed.size) == MESH_PAYLOAD_HEADER_BYTES + 2 * layout.record_words * 8
    actual = _finalize(packed, labels.shape, kwargs, cupy_module)
    expected = cpu_measure(labels, **kwargs)
    parity = measurement_table_parity(expected, actual, exact_float_columns=True)
    assert parity.passed, parity.detail


def test_real_gpu_threshold_above_uint64_preserves_cpu_skip_all(cupy_module) -> None:
    labels = np.zeros((6, 7, 8), dtype=np.int32)
    labels[1:5, 1:6, 1:7] = 77
    kwargs = {
        "spatial_mode": "3D ZYX",
        "minimum_voxel_count": 2**80,
        "include_convex_hull_metrics": False,
    }

    packed = provider.measure_3d_mesh_morphology(
        cupy_module.asarray(labels),
        **kwargs,
    )
    actual = _finalize(packed, labels.shape, kwargs, cupy_module)
    expected = cpu_measure(labels, **kwargs)

    parity = measurement_table_parity(expected, actual, exact_float_columns=True)
    assert parity.passed, parity.detail
    assert actual.rows[0][actual.columns.index("mesh_status")] == (
        "skipped_too_few_voxels"
    )


def test_real_gpu_pathological_sparse_bbox_uses_linear_uint32_encoding(
    cupy_module,
) -> None:
    labels = np.zeros((64, 65, 66), dtype=np.int32)
    coordinates = (
        (0, 0, 0),
        (0, 0, 65),
        (0, 64, 0),
        (0, 64, 65),
        (63, 0, 0),
        (63, 0, 65),
        (63, 64, 0),
        (63, 64, 65),
        (11, 13, 17),
        (19, 23, 29),
        (31, 37, 41),
        (43, 47, 53),
        (7, 59, 61),
        (57, 5, 3),
        (27, 33, 39),
        (49, 51, 55),
    )
    for coordinate in coordinates:
        labels[coordinate] = 77
    kwargs = {
        "spatial_mode": "3D ZYX",
        "minimum_voxel_count": len(coordinates),
        "include_convex_hull_metrics": False,
    }

    packed = provider.measure_3d_mesh_morphology(
        cupy_module.asarray(labels),
        **kwargs,
    )
    layout = mesh_morphology_layout(
        labels.shape,
        spatial_mode="3D ZYX",
        include_convex_hull_metrics=False,
    )
    host = cupy_module.asnumpy(packed)
    directory = np.frombuffer(
        host[
            MESH_PAYLOAD_HEADER_BYTES : MESH_PAYLOAD_HEADER_BYTES
            + layout.record_words * 8
        ],
        dtype="<u8",
    )

    assert int(directory[8]) == MESH_ENCODING_SPARSE_UINT32
    assert int(directory[10]) == len(coordinates) * 4
    assert int(packed.size) == (
        MESH_PAYLOAD_HEADER_BYTES + layout.record_words * 8 + len(coordinates) * 4
    )
    actual = _finalize(packed, labels.shape, kwargs, cupy_module)
    expected = cpu_measure(labels, **kwargs)
    parity = measurement_table_parity(expected, actual, exact_float_columns=True)
    assert parity.passed, parity.detail


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


def test_real_gpu_progress_reports_blocks_and_supports_cancellation(
    cupy_module,
) -> None:
    labels = _labels()
    progress = _Progress()

    provider.measure_3d_mesh_morphology(
        cupy_module.asarray(labels),
        spatial_mode="3D ZYX",
        minimum_voxel_count=16,
        progress=progress,
    )

    assert [(current, total) for current, total, _message in progress.reports] == [
        (current, 7) for current in range(8)
    ]
    messages = " ".join(message for _current, _total, message in progress.reports)
    assert "compacting labels" in messages
    assert "measuring bounds" in messages
    assert "packing masks" in messages
    assert "assembling payload" in messages

    cancelled = _Progress(cancel_after=1)
    with pytest.raises(RuntimeError, match="cancelled"):
        provider.measure_3d_mesh_morphology(
            cupy_module.asarray(labels[:1]),
            spatial_mode="3D ZYX",
            progress=cancelled,
        )
    assert [current for current, _total, _message in cancelled.reports] == [0, 1]


def test_real_gpu_rejects_unsupported_input_regions(cupy_module) -> None:
    labels = np.zeros((7, 8, 9), dtype=np.int32)
    with pytest.raises(ValueError, match="native int32"):
        provider.measure_3d_mesh_morphology(
            cupy_module.asarray(labels, dtype=cupy_module.uint16),
            spatial_mode="3D ZYX",
        )
    negative = labels.copy()
    negative[0, 0, 0] = -1
    with pytest.raises(ValueError, match="non-negative"):
        provider.measure_3d_mesh_morphology(
            cupy_module.asarray(negative),
            spatial_mode="3D ZYX",
        )
    with pytest.raises(ValueError, match="true 3D"):
        provider.measure_3d_mesh_morphology(
            cupy_module.asarray(labels),
            spatial_mode="2D YX",
        )
