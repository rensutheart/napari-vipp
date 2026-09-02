from __future__ import annotations

import importlib
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from napari_vipp.core.gpu import cupy_skeleton_measurements as provider
from napari_vipp.core.measurements import measurement_table_parity
from napari_vipp.core.metadata import AxisMetadata, image_state_from_array
from napari_vipp.core.operations import analyze_skeleton as cpu_analyze_skeleton
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.progress import OperationCancelled, ProgressContext
from napari_vipp.core.skeleton_measurements import (
    finalize_analyze_skeleton_table,
    skeleton_analysis_layout,
)

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


def _skeleton_2d() -> np.ndarray:
    skeleton = np.zeros((2, 18, 20), dtype=bool)
    skeleton[0, 1, 1] = True  # isolated component
    skeleton[0, 4, 2:9] = True  # open branch
    skeleton[0, 10, 11:16] = True  # pure cycle with shortcut filtering
    skeleton[0, 14, 11:16] = True
    skeleton[0, 10:15, 11] = True
    skeleton[0, 10:15, 15] = True
    skeleton[1, 3:14, 8] = True  # multi-arm junction
    skeleton[1, 8, 3:15] = True
    skeleton[1, 3:7, 16] = True  # diagonal-only branch
    for index in range(4):
        skeleton[1, 3 + index, 16 + index if 16 + index < 20 else 19] = True
    return skeleton


def _skeleton_3d() -> np.ndarray:
    skeleton = np.zeros((2, 10, 12, 14), dtype=bool)
    skeleton[0, 1, 1, 1] = True
    skeleton[0, 5, 2:10, 3] = True
    skeleton[0, 5, 6, 3:11] = True
    # A planar rectangular cycle exercises the lower-order shortcut rule in 3D.
    skeleton[1, 3, 2, 3:10] = True
    skeleton[1, 3, 8, 3:10] = True
    skeleton[1, 3, 2:9, 3] = True
    skeleton[1, 3, 2:9, 9] = True
    skeleton[1, 7, 2:10, 11] = True
    return skeleton


def _finalize(packed, shape, kwargs, cupy):
    layout = skeleton_analysis_layout(
        shape,
        **{
            name: value
            for name, value in kwargs.items()
            if name not in {"input_mode", "source_name"}
        },
    )
    return finalize_analyze_skeleton_table(
        cupy.asnumpy(packed),
        layout=layout,
        source_name=str(kwargs.get("source_name", "")),
    )


def test_provider_module_import_is_cuda_lazy_in_a_fresh_process() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(SOURCE_ROOT), environment.get("PYTHONPATH", "")))
    )
    script = r"""
import sys
import napari_vipp.core.gpu.cupy_skeleton_measurements as module
assert module.__all__ == ["analyze_skeleton"]
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
    ("skeleton_factory", "kwargs"),
    (
        (
            _skeleton_2d,
            {
                "spatial_mode": "2D YX",
                "axis_names": ("t", "y", "x"),
                "axis_types": ("time", "space", "space"),
                "axis_scales": (1.0, 2.0, 0.5),
                "axis_units": (None, "um", "um"),
            },
        ),
        (
            _skeleton_3d,
            {
                "spatial_mode": "3D ZYX",
                "axis_names": ("t", "z", "y", "x"),
                "axis_types": ("time", "space", "space", "space"),
                "axis_scales": (1.0, 3.0, 0.75, 0.25),
                "axis_units": (None, "um", "um", "um"),
            },
        ),
    ),
)
def test_real_gpu_cycles_branches_isolated_and_anisotropy_match_cpu(
    cupy_module,
    skeleton_factory,
    kwargs,
) -> None:
    skeleton = skeleton_factory()
    kwargs = {
        **kwargs,
        "input_mode": "Already skeletonized",
        "source_name": "gpu-skeleton-fixture",
    }
    expected = cpu_analyze_skeleton(skeleton, **kwargs)
    device_skeleton = cupy_module.asarray(skeleton)
    before = device_skeleton.copy()

    packed = provider.analyze_skeleton(device_skeleton, **kwargs)
    actual = _finalize(packed, skeleton.shape, kwargs, cupy_module)

    assert isinstance(packed, cupy_module.ndarray)
    assert packed.dtype == cupy_module.uint8
    assert packed.ndim == 1
    assert packed.flags.c_contiguous
    cupy_module.testing.assert_array_equal(device_skeleton, before)
    parity = measurement_table_parity(expected, actual)
    assert parity.passed, parity.detail


def test_real_gpu_nontrailing_spatial_axes_and_multiple_blocks_match_cpu(
    cupy_module,
) -> None:
    canonical = _skeleton_3d()
    skeleton = np.transpose(canonical, (2, 0, 3, 1))  # Y, T, X, Z
    kwargs = {
        "spatial_mode": "3D ZYX",
        "input_mode": "Already skeletonized",
        "axis_names": ("y", "t", "x", "z"),
        "axis_types": ("space", "time", "space", "space"),
        "axis_scales": (0.75, 1.0, 0.25, 3.0),
        "axis_units": ("um", None, "um", "um"),
    }
    expected = cpu_analyze_skeleton(skeleton, **kwargs)

    packed = provider.analyze_skeleton(cupy_module.asarray(skeleton), **kwargs)
    actual = _finalize(packed, skeleton.shape, kwargs, cupy_module)

    parity = measurement_table_parity(expected, actual)
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


def test_real_gpu_progress_and_cancellation(cupy_module) -> None:
    skeleton = _skeleton_2d()
    progress = _Progress()

    provider.analyze_skeleton(
        cupy_module.asarray(skeleton),
        spatial_mode="2D YX",
        progress=progress,
    )

    assert [(current, total) for current, total, _message in progress.reports] == [
        (current, 5) for current in range(6)
    ]
    messages = " ".join(message for _current, _total, message in progress.reports)
    assert "labeling components" in messages
    assert "measuring graph" in messages
    assert "assembling payload" in messages

    cancelled = _Progress(cancel_after=1)
    with pytest.raises(RuntimeError, match="cancelled"):
        provider.analyze_skeleton(
            cupy_module.asarray(skeleton[:1]),
            spatial_mode="2D YX",
            progress=cancelled,
        )
    assert [current for current, _total, _message in cancelled.reports] == [0, 1]


def test_real_gpu_uses_prepared_pipeline_progress_and_cancellation(cupy_module) -> None:
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    threshold = pipeline.add_node("binary_threshold")
    analysis = pipeline.add_node("analyze_skeleton")
    assert pipeline.connect("input", threshold.id).success
    assert pipeline.connect(threshold.id, analysis.id).success

    skeleton = _skeleton_2d()
    state = image_state_from_array(
        skeleton,
        axes=(
            AxisMetadata("t", "time"),
            AxisMetadata("y", "space"),
            AxisMetadata("x", "space"),
        ),
    )
    assert state is not None
    reports = []
    cancelled = False

    def report(node_id: str, current: int, total: int, message: str) -> None:
        nonlocal cancelled
        reports.append((node_id, current, total, message))
        if current == 1:
            cancelled = True

    call = pipeline.prepare_node_call(
        analysis.id,
        (skeleton,),
        (state,),
        progress_callback=report,
        cancel_callback=lambda: cancelled,
    )
    assert call is not None
    assert isinstance(call.kwargs.get("progress"), ProgressContext)

    with pytest.raises(OperationCancelled, match="Operation cancelled"):
        provider.analyze_skeleton(
            cupy_module.asarray(skeleton),
            **call.keyword_arguments(),
        )

    assert [(current, total) for _node, current, total, _message in reports] == [
        (0, 5),
        (1, 5),
    ]
    assert {node_id for node_id, *_rest in reports} == {analysis.id}


def test_real_gpu_rejects_cpu_only_input_regions(cupy_module) -> None:
    skeleton = _skeleton_2d()
    with pytest.raises(ValueError, match="Already skeletonized"):
        provider.analyze_skeleton(
            cupy_module.asarray(skeleton),
            spatial_mode="2D YX",
            input_mode="Skeletonize first",
        )
    with pytest.raises(ValueError, match="boolean skeleton mask"):
        provider.analyze_skeleton(
            cupy_module.asarray(skeleton, dtype=cupy_module.uint8),
            spatial_mode="2D YX",
        )
