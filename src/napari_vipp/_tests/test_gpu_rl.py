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
from napari_vipp.core.gpu import cupy_rl
from napari_vipp.core.gpu.cupy_runtime import CuPyRuntime
from napari_vipp.core.operations import richardson_lucy_deconvolution as cpu_rl
from napari_vipp.core.progress import OperationCancelled, ProgressContext

RL_FLOAT32_NRMSE_LIMIT = 2e-6


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
    isfinite = staticmethod(np.isfinite)
    where = staticmethod(np.where)
    any = staticmethod(np.any)
    zeros_like = staticmethod(np.zeros_like)
    empty = staticmethod(np.empty)
    full = staticmethod(np.full)
    flip = staticmethod(np.flip)


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
module = importlib.import_module("napari_vipp.core.gpu.cupy_rl")
assert callable(module.richardson_lucy_deconvolution)
assert module.__all__ == ["richardson_lucy_deconvolution"]
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


@pytest.mark.parametrize(
    ("image_shape", "psf_shape", "spatial_mode", "resolved_spatial_ndim"),
    (
        ((9, 11), (3, 5), "2D YX", None),
        ((2, 9, 11), (3, 5), "2D per XY slice (advanced)", None),
        ((2, 3, 9, 11), (3, 5), "Auto from axes", 2),
        ((5, 7, 9), (3, 3, 5), "3D ZYX", None),
        ((2, 5, 7, 9), (3, 3, 5), "Auto from axes", 3),
    ),
)
@pytest.mark.parametrize(
    (
        "normalize_psf",
        "clip_negative_input",
        "clip_output_negative",
        "preserve_input_scale",
        "filter_epsilon",
    ),
    (
        (True, True, True, True, 1e-12),
        (False, True, False, False, 0.0),
        (True, False, False, True, 1e-5),
    ),
)
def test_fake_provider_matches_cpu_across_spatial_and_parameter_contract(
    fake_stack,
    image_shape,
    psf_shape,
    spatial_mode,
    resolved_spatial_ndim,
    normalize_psf,
    clip_negative_input,
    clip_output_negative,
    preserve_input_scale,
    filter_epsilon,
):
    _cupy, signal = fake_stack
    image = _image(image_shape, seed=sum(image_shape) + 101)
    psf = _psf(psf_shape, seed=sum(psf_shape) + 211)
    image_before = image.copy()
    psf_before = psf.copy()
    kwargs = {
        "spatial_mode": spatial_mode,
        "iterations": 3,
        "normalize_psf": normalize_psf,
        "clip_negative_input": clip_negative_input,
        "clip_output_negative": clip_output_negative,
        "preserve_input_scale": preserve_input_scale,
        "filter_epsilon": filter_epsilon,
        "resolved_spatial_ndim": resolved_spatial_ndim,
    }

    expected = cpu_rl([image, psf], **kwargs)
    actual = cupy_rl.richardson_lucy_deconvolution([image, psf], **kwargs)

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


def test_fake_progress_matches_cpu_block_times_iteration_contract(fake_stack):
    cupy, _signal = fake_stack
    image = _image((2, 3, 7, 9), seed=307)
    psf = _psf((3, 3), seed=311)
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
    expected = cpu_rl([image, psf], **kwargs)
    kwargs["progress"] = ProgressContext(reporter=gpu_report)
    actual = cupy_rl.richardson_lucy_deconvolution([image, psf], **kwargs)

    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    assert gpu_updates == cpu_updates
    assert [(item.current, item.total) for item in gpu_updates] == [
        (current, 24) for current in range(25)
    ]
    assert {item.message for item in gpu_updates} == {"Richardson-Lucy deconvolution"}
    assert synchronization_at_report == list(range(25))
    assert cupy.stream.synchronizations == 24


def test_fake_cancellation_is_checked_between_synchronized_iterations(fake_stack):
    cupy, signal = fake_stack
    image = _image((2, 7, 9), seed=401)
    psf = _psf((3, 3), seed=409)
    updates = []
    progress = ProgressContext(
        cancelled=lambda: len(updates) >= 2,
        reporter=updates.append,
    )

    with pytest.raises(OperationCancelled, match="Operation cancelled"):
        cupy_rl.richardson_lucy_deconvolution(
            [image, psf],
            spatial_mode="2D YX",
            iterations=5,
            progress=progress,
        )

    assert [(item.current, item.total) for item in updates] == [(0, 10), (1, 10)]
    assert cupy.stream.synchronizations == 1
    assert len(signal.calls) == 2


@pytest.mark.parametrize(
    ("inputs", "match"),
    (
        ([], "requires two inputs"),
        ([None, np.ones((3, 3), dtype=np.float32)], "connected Image and PSF"),
    ),
)
def test_fake_input_contract_errors_match_cpu(fake_stack, inputs, match):
    with pytest.raises(ValueError, match=match):
        cupy_rl.richardson_lucy_deconvolution(inputs, spatial_mode="2D YX")


def test_fake_rejects_mismatched_or_empty_psf(fake_stack):
    image = np.ones((7, 9), dtype=np.float32)

    with pytest.raises(ValueError, match="PSF dimensionality"):
        cupy_rl.richardson_lucy_deconvolution(
            [image, np.ones((3, 3, 3), dtype=np.float32)],
            spatial_mode="2D YX",
        )
    with pytest.raises(ValueError, match="sum is below"):
        cupy_rl.richardson_lucy_deconvolution(
            [image, np.zeros((3, 3), dtype=np.float32)],
            spatial_mode="2D YX",
        )


def test_fake_auto_mode_requires_resolved_axes_for_leading_dimensions(fake_stack):
    image = np.ones((2, 7, 9), dtype=np.float32)
    psf = np.ones((3, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="requires explicit axis semantics"):
        cupy_rl.richardson_lucy_deconvolution([image, psf])


@pytest.mark.parametrize(
    ("image_shape", "psf_shape", "spatial_mode", "resolved_spatial_ndim"),
    (
        ((19, 23), (5, 7), "2D YX", None),
        ((2, 17, 21), (5, 5), "Auto from axes", 2),
        ((7, 13, 15), (3, 5, 5), "3D ZYX", None),
        ((2, 5, 11, 13), (3, 3, 5), "Auto from axes", 3),
    ),
)
def test_real_gpu_float32_parity_and_residency(
    real_cupy,
    image_shape,
    psf_shape,
    spatial_mode,
    resolved_spatial_ndim,
):
    image = _image(image_shape, seed=503 + len(image_shape))
    psf = _psf(psf_shape, seed=509 + len(psf_shape))
    expected = cpu_rl(
        [image, psf],
        spatial_mode=spatial_mode,
        iterations=7,
        resolved_spatial_ndim=resolved_spatial_ndim,
    )
    device_image = real_cupy.asarray(image)
    device_psf = real_cupy.asarray(psf)
    image_before = device_image.copy()
    psf_before = device_psf.copy()

    output = cupy_rl.richardson_lucy_deconvolution(
        [device_image, device_psf],
        spatial_mode=spatial_mode,
        iterations=7,
        resolved_spatial_ndim=resolved_spatial_ndim,
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
    image = real_cupy.asarray(_image((2, 13, 15), seed=601))
    psf = real_cupy.asarray(_psf((5, 5), seed=607))
    updates = []
    progress = ProgressContext(
        cancelled=lambda: len(updates) >= 3,
        reporter=updates.append,
    )

    with pytest.raises(OperationCancelled, match="Operation cancelled"):
        cupy_rl.richardson_lucy_deconvolution(
            [image, psf],
            spatial_mode="2D YX",
            iterations=5,
            progress=progress,
        )

    assert [(item.current, item.total) for item in updates] == [
        (0, 10),
        (1, 10),
        (2, 10),
    ]


def test_real_gpu_progress_path_matches_cpu_and_reports_every_iteration(real_cupy):
    image = _image((2, 13, 15), seed=701)
    psf = _psf((5, 5), seed=709)
    cpu_updates = []
    gpu_updates = []
    kwargs = {"spatial_mode": "2D YX", "iterations": 5}
    expected = cpu_rl(
        [image, psf],
        progress=ProgressContext(reporter=cpu_updates.append),
        **kwargs,
    )

    output = cupy_rl.richardson_lucy_deconvolution(
        [real_cupy.asarray(image), real_cupy.asarray(psf)],
        progress=ProgressContext(reporter=gpu_updates.append),
        **kwargs,
    )
    real_cupy.cuda.get_current_stream().synchronize()

    assert gpu_updates == cpu_updates
    assert [(item.current, item.total) for item in gpu_updates] == [
        (current, 10) for current in range(11)
    ]
    _assert_float32_parity(expected, real_cupy.asnumpy(output))


def test_real_gpu_512_fft_peak_fits_versioned_memory_estimate(real_cupy):
    del real_cupy  # The fixture provides the portable skip contract.
    image = np.random.default_rng(811).random((512, 512), dtype=np.float32)
    y, x = np.mgrid[-6:7, -6:7].astype(np.float32)
    psf = np.exp(-(x * x + y * y) / np.float32(2.0 * 1.7**2)).astype(np.float32)
    psf /= np.float32(psf.sum(dtype=np.float64))
    spec = compute_specs_for(
        "richardson_lucy_deconvolution",
        include_cpu=False,
        allow_experimental=True,
    )[0]
    workload = WorkloadDescriptor(
        "rl-memory-regression",
        "richardson_lucy_deconvolution",
        (image.shape, psf.shape),
        (image.dtype.name, psf.dtype.name),
        parameters=(
            ("spatial_mode", "2D YX"),
            ("iterations", 10),
            ("normalize_psf", True),
            ("clip_negative_input", True),
            ("clip_output_negative", True),
            ("preserve_input_scale", True),
            ("filter_epsilon", 1e-8),
        ),
        resolved_spatial_ndim=2,
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
            device_image = runtime.to_device(
                image,
                device_id=probe.selected_device_id,
            )
            device_psf = runtime.to_device(
                psf,
                device_id=probe.selected_device_id,
            )
            output = cupy_rl.richardson_lucy_deconvolution(
                [device_image, device_psf],
                spatial_mode="2D YX",
                iterations=10,
                filter_epsilon=1e-8,
                resolved_spatial_ndim=2,
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
            f"Observed RL CUDA peak {observed_bytes} exceeds versioned memory "
            f"admission {admitted_bytes}."
        )
    finally:
        runtime.close()


def _assert_float32_parity(expected: np.ndarray, actual: np.ndarray) -> None:
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype == np.dtype(np.float32)
    assert np.isfinite(actual).all()
    expected64 = expected.astype(np.float64)
    actual64 = actual.astype(np.float64)
    difference = actual64 - expected64
    peak = float(np.max(np.abs(expected64), initial=0.0))
    max_abs = float(np.max(np.abs(difference), initial=0.0))
    max_abs_limit = 1e-6 + 5e-6 * peak
    denominator = max(
        float(np.linalg.norm(expected64.ravel())),
        float(np.sqrt(expected64.size) * 1e-12),
    )
    nrmse = float(np.linalg.norm(difference.ravel()) / denominator)
    assert nrmse <= RL_FLOAT32_NRMSE_LIMIT, (
        f"Richardson-Lucy NRMSE {nrmse:.9g} exceeds {RL_FLOAT32_NRMSE_LIMIT:.9g}."
    )
    assert max_abs <= max_abs_limit, (
        f"Richardson-Lucy max abs error {max_abs:.9g} exceeds {max_abs_limit:.9g}."
    )
