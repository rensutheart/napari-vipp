from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import signal as scipy_signal

from napari_vipp.core.compute import WorkloadDescriptor
from napari_vipp.core.compute_policy import estimate_candidate_memory
from napari_vipp.core.compute_specs import compute_specs_for
from napari_vipp.core.gpu import cupy_rl, cupy_rl_tv
from napari_vipp.core.gpu.cupy_runtime import CuPyRuntime
from napari_vipp.core.progress import OperationCancelled, ProgressContext
from napari_vipp.core.richardson_lucy import (
    _tv_divergence as cpu_tv_divergence,
)
from napari_vipp.core.richardson_lucy import (
    richardson_lucy_tv_deconvolution as cpu_rl_tv,
)
from napari_vipp.core.richardson_lucy_parity import (
    richardson_lucy_tv_float32_parity,
)
from scripts.validate_rl_tv_phantoms import (
    calculate_metrics,
    make_phantom_2d,
    make_phantom_3d,
    observed_image,
)


class _FakeStream:
    def __init__(self) -> None:
        self.synchronizations = 0

    def synchronize(self) -> None:
        self.synchronizations += 1


class _FakeCupy:
    float32 = np.float32
    float64 = np.float64
    inf = np.inf

    def __init__(self) -> None:
        self.stream = _FakeStream()
        self.cuda = SimpleNamespace(get_current_stream=lambda: self.stream)

    asarray = staticmethod(np.asarray)
    ascontiguousarray = staticmethod(np.ascontiguousarray)
    nan_to_num = staticmethod(np.nan_to_num)
    maximum = staticmethod(np.maximum)
    sum = staticmethod(np.sum)
    max = staticmethod(np.max)
    min = staticmethod(np.min)
    mean = staticmethod(np.mean)
    isfinite = staticmethod(np.isfinite)
    where = staticmethod(np.where)
    any = staticmethod(np.any)
    zeros_like = staticmethod(np.zeros_like)
    zeros = staticmethod(np.zeros)
    empty = staticmethod(np.empty)
    full = staticmethod(np.full)
    flip = staticmethod(np.flip)
    gradient = staticmethod(np.gradient)
    sqrt = staticmethod(np.sqrt)
    stack = staticmethod(np.stack)


class _FakeSignal:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def convolve(self, first, second, *, mode, method):
        self.calls.append(
            {
                "first_shape": first.shape,
                "second_shape": second.shape,
                "mode": mode,
                "method": method,
            }
        )
        return scipy_signal.convolve(first, second, mode=mode)


@pytest.fixture
def fake_stack(monkeypatch):
    cupy = _FakeCupy()
    signal = _FakeSignal()
    real_import = importlib.import_module
    modules = {"cupy": cupy, "cupyx.scipy.signal": signal}

    def load(name: str):
        return modules[name] if name in modules else real_import(name)

    cupy_rl._cupy_modules.cache_clear()
    monkeypatch.setattr(cupy_rl.importlib, "import_module", load)
    yield cupy, signal
    cupy_rl._cupy_modules.cache_clear()


@pytest.fixture(scope="module")
def real_cupy():
    try:
        cupy = importlib.import_module("cupy")
        importlib.import_module("cupyx.scipy.signal")
        if int(cupy.cuda.runtime.getDeviceCount()) < 1:
            pytest.skip("CuPy reports no CUDA device.")
        probe = cupy.arange(4, dtype=cupy.float32)
        cupy.cuda.get_current_stream().synchronize()
        del probe
    except Exception as exc:
        pytest.skip(f"A working CuPy CUDA runtime is unavailable: {exc}")
    return cupy


def test_import_is_safe_without_cupy_or_cupyx():
    source_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(source_root), environment.get("PYTHONPATH", "")))
    )
    code = r"""
import builtins
import importlib
import sys

real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == "cupy" or name.startswith("cupyx"):
        raise AssertionError(f"optional CUDA import attempted: {name}")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
module = importlib.import_module("napari_vipp.core.gpu.cupy_rl_tv")
assert callable(module.richardson_lucy_tv_deconvolution)
assert module.__all__ == ["richardson_lucy_tv_deconvolution"]
assert "cupy" not in sys.modules
assert not any(name.startswith("cupyx") for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr


def _image(shape: tuple[int, ...], *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.uniform(0.0, 2048.0, size=shape).astype(np.float32)
    values.flat[0] = np.float32(-7.5)
    values.flat[1] = np.float32(0.0)
    values.setflags(write=False)
    return values


def _psf(shape: tuple[int, ...], *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.uniform(0.0, 1.0, size=shape).astype(np.float32)
    values[(0,) * len(shape)] = np.float32(-0.5)
    values.setflags(write=False)
    return values


def test_fake_tv_divergence_preserves_cpu_sign_stencil_and_epsilon(fake_stack):
    cupy, _signal = fake_stack
    rng = np.random.default_rng(101)
    values = rng.random((5, 7, 9), dtype=np.float32)

    expected = cpu_tv_divergence(values, epsilon=1e-6)
    actual = cupy_rl_tv._tv_divergence(values, epsilon=1e-6, cupy=cupy)

    np.testing.assert_array_equal(actual, expected)


def test_fake_validation_observer_records_denominator_floor_activity(fake_stack):
    cupy, signal = fake_stack
    diagnostics = []
    image = np.zeros((9, 11), dtype=np.float32)
    image[2, 3] = 1.0
    image[7, 8] = 0.5
    psf = np.ones((3, 3), dtype=np.float32) / np.float32(9)

    output = cupy_rl_tv._richardson_lucy_tv_block(
        image,
        psf,
        iterations=3,
        tv_regularization=0.25,
        tv_epsilon=1e-6,
        filter_epsilon=1e-12,
        denominator_floor=1.0,
        iteration_done=None,
        check_cancelled=None,
        diagnostics_observer=lambda *values: diagnostics.append(values),
        cupy=cupy,
        signal=signal,
    )

    assert output.shape == image.shape
    assert len(diagnostics) == 3
    assert [item[0] for item in diagnostics] == [0, 1, 2]
    assert any(item[2] > 0 for item in diagnostics[1:])


@pytest.mark.parametrize(
    ("image_shape", "psf_shape", "spatial_mode", "resolved_spatial_ndim"),
    (
        ((11, 13), (3, 5), "2D YX", None),
        ((2, 11, 13), (3, 5), "2D per XY slice (advanced)", None),
        ((2, 3, 11, 13), (3, 5), "Auto from axes", 2),
        ((5, 9, 11), (3, 3, 5), "3D ZYX", None),
        ((2, 5, 9, 11), (3, 3, 5), "Auto from axes", 3),
    ),
)
@pytest.mark.parametrize(
    ("tv_regularization", "tv_epsilon", "filter_epsilon", "denominator_floor"),
    (
        (0.0, 1e-6, 1e-12, 0.05),
        (0.002, 1e-6, 1e-12, 0.05),
        (0.02, 1e-3, 1e-8, 0.5),
    ),
)
def test_fake_provider_matches_cpu_across_blocks_and_tv_parameters(
    fake_stack,
    image_shape,
    psf_shape,
    spatial_mode,
    resolved_spatial_ndim,
    tv_regularization,
    tv_epsilon,
    filter_epsilon,
    denominator_floor,
):
    _cupy, signal = fake_stack
    image = _image(image_shape, seed=sum(image_shape) + 211)
    psf = _psf(psf_shape, seed=sum(psf_shape) + 307)
    image_before = image.copy()
    psf_before = psf.copy()
    kwargs = {
        "spatial_mode": spatial_mode,
        "iterations": 3,
        "tv_regularization": tv_regularization,
        "tv_epsilon": tv_epsilon,
        "filter_epsilon": filter_epsilon,
        "denominator_floor": denominator_floor,
        "resolved_spatial_ndim": resolved_spatial_ndim,
    }

    expected = cpu_rl_tv([image, psf], **kwargs)
    actual = cupy_rl_tv.richardson_lucy_tv_deconvolution([image, psf], **kwargs)

    assert actual.shape == expected.shape == image.shape
    assert actual.dtype == expected.dtype == np.dtype(np.float32)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(image, image_before)
    np.testing.assert_array_equal(psf, psf_before)
    assert image.flags.writeable is False
    assert psf.flags.writeable is False
    assert signal.calls
    assert all(call["mode"] == "same" for call in signal.calls)
    assert all(call["method"] == "fft" for call in signal.calls)


def test_fake_progress_matches_cpu_block_iteration_contract(fake_stack):
    cupy, _signal = fake_stack
    image = _image((2, 3, 7, 9), seed=401)
    psf = _psf((3, 3), seed=409)
    cpu_updates = []
    gpu_updates = []
    synchronization_at_report = []

    def gpu_report(update) -> None:
        gpu_updates.append(update)
        synchronization_at_report.append(cupy.stream.synchronizations)

    kwargs = {
        "spatial_mode": "2D YX",
        "iterations": 4,
        "progress": ProgressContext(reporter=cpu_updates.append),
    }
    expected = cpu_rl_tv([image, psf], **kwargs)
    kwargs["progress"] = ProgressContext(reporter=gpu_report)
    actual = cupy_rl_tv.richardson_lucy_tv_deconvolution([image, psf], **kwargs)

    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    assert gpu_updates == cpu_updates
    assert [(item.current, item.total) for item in gpu_updates] == [
        (current, 24) for current in range(25)
    ]
    assert {item.message for item in gpu_updates} == {
        "Richardson-Lucy TV deconvolution"
    }
    assert synchronization_at_report == list(range(25))
    assert cupy.stream.synchronizations == 24


def test_fake_cancellation_is_checked_between_synchronized_iterations(fake_stack):
    cupy, signal = fake_stack
    updates = []
    progress = ProgressContext(
        cancelled=lambda: len(updates) >= 2,
        reporter=updates.append,
    )

    with pytest.raises(OperationCancelled, match="Operation cancelled"):
        cupy_rl_tv.richardson_lucy_tv_deconvolution(
            [_image((2, 7, 9), seed=503), _psf((3, 3), seed=509)],
            spatial_mode="2D YX",
            iterations=5,
            progress=progress,
        )

    assert [(item.current, item.total) for item in updates] == [(0, 10), (1, 10)]
    assert cupy.stream.synchronizations == 1
    assert len(signal.calls) == 2


@pytest.mark.parametrize(
    ("image_shape", "psf_shape", "spatial_mode", "resolved_spatial_ndim"),
    (
        ((47, 53), (7, 9), "2D YX", None),
        ((2, 41, 45), (5, 7), "Auto from axes", 2),
        ((9, 31, 35), (5, 7, 7), "3D ZYX", None),
        ((2, 7, 27, 29), (3, 5, 5), "Auto from axes", 3),
    ),
)
def test_real_gpu_default_region_parity_and_residency(
    real_cupy,
    image_shape,
    psf_shape,
    spatial_mode,
    resolved_spatial_ndim,
):
    image = _image(image_shape, seed=601 + len(image_shape))
    psf = _psf(psf_shape, seed=607 + len(psf_shape))
    kwargs = {
        "spatial_mode": spatial_mode,
        "iterations": 25,
        "tv_regularization": 0.002,
        "tv_epsilon": 1e-6,
        "filter_epsilon": 1e-12,
        "denominator_floor": 0.05,
        "resolved_spatial_ndim": resolved_spatial_ndim,
    }
    expected = cpu_rl_tv([image, psf], **kwargs)
    device_image = real_cupy.asarray(image)
    device_psf = real_cupy.asarray(psf)
    image_before = device_image.copy()
    psf_before = device_psf.copy()

    output = cupy_rl_tv.richardson_lucy_tv_deconvolution(
        [device_image, device_psf],
        **kwargs,
    )
    real_cupy.cuda.get_current_stream().synchronize()

    assert isinstance(output, real_cupy.ndarray)
    assert output.device.id == device_image.device.id == device_psf.device.id
    assert output.shape == device_image.shape
    assert output.dtype == real_cupy.float32
    assert output.data.ptr != device_image.data.ptr
    real_cupy.testing.assert_array_equal(device_image, image_before)
    real_cupy.testing.assert_array_equal(device_psf, psf_before)
    _assert_float32_parity(expected, real_cupy.asnumpy(output))


def test_real_gpu_progress_is_synchronized_and_cancellable(real_cupy):
    image = real_cupy.asarray(_image((2, 13, 15), seed=641))
    psf = real_cupy.asarray(_psf((5, 5), seed=643))
    updates = []
    progress = ProgressContext(
        cancelled=lambda: len(updates) >= 3,
        reporter=updates.append,
    )

    with pytest.raises(OperationCancelled, match="Operation cancelled"):
        cupy_rl_tv.richardson_lucy_tv_deconvolution(
            [image, psf],
            spatial_mode="2D YX",
            iterations=5,
            tv_regularization=0.002,
            tv_epsilon=1e-6,
            filter_epsilon=1e-12,
            denominator_floor=0.05,
            progress=progress,
        )

    assert [(item.current, item.total) for item in updates] == [
        (0, 10),
        (1, 10),
        (2, 10),
    ]


@pytest.mark.parametrize("phantom_factory", (make_phantom_2d, make_phantom_3d))
def test_real_gpu_preserves_default_phantom_metrics(real_cupy, phantom_factory):
    phantom = phantom_factory()
    observed = observed_image(phantom)
    kwargs = {
        "spatial_mode": phantom.spatial_mode,
        "iterations": 25,
        "tv_regularization": 0.002,
        "tv_epsilon": 1e-6,
        "filter_epsilon": 1e-12,
        "denominator_floor": 0.05,
    }
    expected = cpu_rl_tv([observed, phantom.psf], **kwargs)
    output = cupy_rl_tv.richardson_lucy_tv_deconvolution(
        [real_cupy.asarray(observed), real_cupy.asarray(phantom.psf)],
        **kwargs,
    )
    actual = real_cupy.asnumpy(output)
    expected_metrics = calculate_metrics(expected, phantom)
    actual_metrics = calculate_metrics(actual, phantom)

    _assert_float32_parity(expected, actual)
    for name in ("points_recovery", "thin_line_recovery", "dim_structure_recovery"):
        assert abs(actual_metrics[name] - expected_metrics[name]) <= 0.005
    for name in ("mse", "border_mse", "flux_ratio"):
        denominator = max(abs(expected_metrics[name]), 1e-12)
        assert abs(actual_metrics[name] - expected_metrics[name]) / denominator <= 0.005


@pytest.mark.parametrize(
    ("shape", "psf_shape", "spatial_mode"),
    (
        ((43, 47), (5, 7), "2D YX"),
        ((7, 27, 31), (3, 5, 5), "3D ZYX"),
    ),
)
def test_real_gpu_lambda_zero_matches_ordinary_gpu(
    real_cupy,
    shape,
    psf_shape,
    spatial_mode,
):
    image = real_cupy.asarray(_image(shape, seed=701 + len(shape)))
    psf = real_cupy.asarray(_psf(psf_shape, seed=709 + len(psf_shape)))
    common = {
        "spatial_mode": spatial_mode,
        "iterations": 25,
        "filter_epsilon": 1e-8,
    }
    ordinary = cupy_rl.richardson_lucy_deconvolution([image, psf], **common)
    regularized = cupy_rl_tv.richardson_lucy_tv_deconvolution(
        [image, psf],
        tv_regularization=0.0,
        **common,
    )
    real_cupy.cuda.get_current_stream().synchronize()

    maximum = max(float(real_cupy.max(real_cupy.abs(image)).item()), 1.0)
    difference = float(real_cupy.max(real_cupy.abs(regularized - ordinary)).item())
    assert difference <= 1e-6 * maximum


def test_real_gpu_3d_peak_fits_versioned_tv_memory_estimate(real_cupy):
    del real_cupy  # The fixture provides the portable skip contract.
    rng = np.random.default_rng(811)
    image = rng.random((16, 256, 256), dtype=np.float32)
    z, y, x = np.mgrid[-3:4, -5:6, -5:6].astype(np.float32)
    psf = np.exp(-(z * z / np.float32(2.0 * 1.5**2) + (x * x + y * y) / 4.0)).astype(
        np.float32
    )
    psf /= np.float32(psf.sum(dtype=np.float64))
    spec = compute_specs_for(
        "richardson_lucy_tv_deconvolution",
        include_cpu=False,
        allow_experimental=True,
    )[0]
    workload = WorkloadDescriptor(
        "rl-tv-memory-regression",
        "richardson_lucy_tv_deconvolution",
        (image.shape, psf.shape),
        (image.dtype.name, psf.dtype.name),
        parameters=(
            ("spatial_mode", "3D ZYX"),
            ("iterations", 5),
            ("tv_regularization", 0.002),
            ("tv_epsilon", 1e-6),
            ("normalize_psf", True),
            ("clip_negative_input", True),
            ("clip_output_negative", True),
            ("preserve_input_scale", True),
            ("filter_epsilon", 1e-12),
            ("denominator_floor", 0.05),
        ),
        resolved_spatial_ndim=3,
    )
    estimate = estimate_candidate_memory(spec, workload)
    admitted_bytes = estimate.total_device_peak_bytes + estimate.uncertainty_bytes
    runtime = CuPyRuntime()
    try:
        probe = runtime.probe()
        if not probe.available or not probe.selected_device_id:
            pytest.skip(probe.message or "The CUDA runtime is unavailable.")
        observed_bytes = 0
        with runtime.execution_scope(
            device_id=probe.selected_device_id,
            safety_reserve_bytes=0,
        ):
            device_image = runtime.to_device(image, device_id=probe.selected_device_id)
            device_psf = runtime.to_device(psf, device_id=probe.selected_device_id)
            output = cupy_rl_tv.richardson_lucy_tv_deconvolution(
                [device_image, device_psf],
                spatial_mode="3D ZYX",
                iterations=5,
                resolved_spatial_ndim=3,
            )
            runtime.synchronize(device_id=probe.selected_device_id)
            assert runtime.is_device_value(output)
            snapshot = runtime.memory_snapshot(device_id=probe.selected_device_id)
            observed_bytes = (
                snapshot.runtime_reserved_bytes + snapshot.out_of_pool_bytes
            )
            output = None
            device_image = None
            device_psf = None

        terminal = runtime.memory_snapshot(device_id=probe.selected_device_id)
        assert terminal.runtime_live_bytes == 0
        assert terminal.runtime_reserved_bytes == 0
        assert observed_bytes <= admitted_bytes, (
            f"Observed RL-TV CUDA peak {observed_bytes} exceeds versioned memory "
            f"admission {admitted_bytes}."
        )
    finally:
        runtime.close()


def _assert_float32_parity(expected: np.ndarray, actual: np.ndarray) -> None:
    result = richardson_lucy_tv_float32_parity(expected, actual)
    assert result.passed, result.detail
