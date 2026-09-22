"""Scientific wrapper contracts for exploratory background benchmark methods."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import ndimage as ndi

from napari_vipp.core import operations as ops

CPU_METHODS = ("vipp_cpu", "itk_ball", "itk_box", "itk_reconstruction", "scipy_box")


@pytest.fixture(scope="module")
def methods():
    path = (
        Path(__file__).resolve().parents[3] / "scripts/background_benchmark_methods.py"
    )
    spec = importlib.util.spec_from_file_location("background_methods_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _provider(method):
    if method.startswith("itk_"):
        pytest.importorskip("SimpleITK")


@pytest.mark.parametrize("method", CPU_METHODS)
@pytest.mark.parametrize("dtype", [np.uint16, np.float32, np.float64])
def test_output_dtype_and_readonly_noncontiguous_input_unchanged(
    methods, method, dtype
):
    _provider(method)
    image = (
        np.random.default_rng(742).integers(0, 2000, (19, 23)).astype(dtype)[:, ::-1]
    )
    before = image.copy()
    image.setflags(write=False)
    result = methods.run_method(image, method=method, radius=3, threads=1)
    assert result.shape == image.shape
    assert result.dtype == image.dtype
    assert not np.shares_memory(image, result)
    np.testing.assert_array_equal(image, before)
    assert np.all(result >= 0)


@pytest.mark.parametrize("method", CPU_METHODS)
@pytest.mark.parametrize("smoothing", [False, True])
def test_raw_and_final_use_original_input_not_smoothed_subtraction(
    methods, method, smoothing
):
    _provider(method)
    image = np.full((17, 19), 11, dtype=np.float32)
    image[8, 9] = 90
    background = methods.estimate_background(
        image, method=method, radius=3, smoothing=smoothing, threads=1
    )
    raw = methods.raw_corrected(
        image, method=method, radius=3, smoothing=smoothing, threads=1
    )
    np.testing.assert_array_equal(raw, image - background)
    final = methods.run_method(
        image, method=method, radius=3, smoothing=smoothing, threads=1
    )
    np.testing.assert_array_equal(final, np.maximum(raw, 0))
    if smoothing:
        smoothed_minus_background = (
            ndi.uniform_filter(image, size=3, mode="nearest") - background
        )
        assert raw[8, 9] != smoothed_minus_background[8, 9]


@pytest.mark.parametrize("method", CPU_METHODS)
def test_light_background_inverts_estimate_and_subtraction(methods, method):
    _provider(method)
    image = np.full((17, 19), 100, dtype=np.float32)
    image[7:10, 8:11] = 20
    background = methods.estimate_background(
        image,
        method=method,
        radius=3,
        smoothing=False,
        light_background=True,
        threads=1,
    )
    raw = methods.raw_corrected(
        image,
        method=method,
        radius=3,
        smoothing=False,
        light_background=True,
        threads=1,
    )
    np.testing.assert_array_equal(raw, background - image)
    inverted = 120 - image
    dark_background = methods.estimate_background(
        inverted, method=method, radius=3, smoothing=False, threads=1
    )
    np.testing.assert_array_equal(background, 120 - dark_background)


@pytest.mark.parametrize("method", CPU_METHODS)
@pytest.mark.parametrize("spatial_ndim", [2, 3])
def test_leading_dimensions_are_independent_including_reconstruction(
    methods, method, spatial_ndim
):
    _provider(method)
    shape = (3, 13, 15) if spatial_ndim == 2 else (2, 3, 13, 15)
    image = np.random.default_rng(361).integers(0, 1000, shape).astype(np.float32)
    image[0] = 20
    result = methods.run_method(
        image, method=method, radius=2, spatial_ndim=spatial_ndim, threads=1
    )
    separate = np.stack(
        [
            methods.run_method(
                plane, method=method, radius=2, spatial_ndim=spatial_ndim, threads=1
            )
            for plane in image
        ]
    )
    np.testing.assert_array_equal(result, separate)


@pytest.mark.parametrize("method", CPU_METHODS)
def test_nonfinite_handling_matches_vipp_preparation_policy(methods, method):
    _provider(method)
    image = np.full((9, 11), 40, dtype=np.float32)
    image[1, :4] = [10, np.nan, np.inf, -np.inf]
    expected_input = np.nan_to_num(image, nan=10, posinf=40, neginf=10)
    estimate = methods.estimate_background(image, method=method, radius=2, threads=1)
    expected = methods.estimate_background(
        expected_input, method=method, radius=2, threads=1
    )
    np.testing.assert_array_equal(estimate, expected)
    all_invalid = np.full_like(image, np.nan)
    np.testing.assert_array_equal(
        methods.estimate_background(all_invalid, method=method, radius=2, threads=1),
        np.zeros_like(image),
    )
    raw = methods.raw_corrected(image, method=method, radius=2, threads=1)
    assert np.isnan(raw[1, 1])
    assert np.isposinf(raw[1, 2])
    assert np.isneginf(raw[1, 3])


@pytest.mark.parametrize("shape", [(17, 19), (5, 13, 15)])
@pytest.mark.parametrize("radius", [1, 3, 8])
def test_scipy_box_safe_border_control_matches_itk(methods, shape, radius):
    pytest.importorskip("SimpleITK")
    image = np.random.default_rng(830).uniform(-30, 100, shape).astype(np.float32)
    kwargs = {
        "radius": radius,
        "smoothing": False,
        "spatial_ndim": len(shape),
        "threads": 1,
    }
    actual = methods.estimate_background(image, method="scipy_box", **kwargs)
    expected = methods.estimate_background(image, method="itk_box", **kwargs)
    np.testing.assert_array_equal(actual, expected)


def test_cpu_wrapper_uses_public_operation_and_clipping_policy(methods):
    image = np.full((13, 15), 40, dtype=np.float32)
    image[6, 7] = 0
    actual = methods.run_method(image, method="vipp_cpu", radius=2, clip_negative=False)
    expected = ops.subtract_background(image, radius=2, clip_negative=False)
    np.testing.assert_array_equal(actual, expected)
    assert actual.min() < 0


@pytest.mark.parametrize("method", CPU_METHODS)
def test_integer_rounding_and_bool_match_vipp_policy(methods, method):
    _provider(method)
    image = np.random.default_rng(449).integers(0, 1000, (11, 13), dtype=np.uint16)
    raw = methods.raw_corrected(image, method=method, radius=2, threads=1)
    expected = np.rint(np.maximum(raw, 0)).astype(np.uint16)
    actual = methods.run_method(image, method=method, radius=2, threads=1)
    np.testing.assert_array_equal(actual, expected)
    mask = image > 500
    np.testing.assert_array_equal(
        methods.run_method(mask, method=method, radius=2), mask
    )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"method": "unknown", "radius": 3}, "Unknown"),
        ({"method": "itk_box", "radius": np.inf}, "finite"),
        ({"method": "itk_box", "radius": 3, "threads": 0}, "Thread"),
        ({"method": "itk_box", "radius": 3, "spatial_ndim": 3}, "spatial_ndim"),
    ],
)
def test_invalid_requests_are_rejected(methods, kwargs, match):
    with pytest.raises(ValueError, match=match):
        methods.run_method(np.zeros((7, 9)), **kwargs)


def test_module_import_does_not_load_numerical_providers(methods):
    script = (
        "import importlib.util, sys; "
        "s=importlib.util.spec_from_file_location('candidate_methods', "
        f"{methods.__file__!r}); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        "assert not any(x in sys.modules for x in "
        "('numpy','scipy','SimpleITK','cupy','napari_vipp'))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr


def test_gpu_call_uploads_computes_downloads_and_synchronizes(methods, monkeypatch):
    from napari_vipp.core.gpu import cupy_background

    events = []
    source = np.ones((7, 9), dtype=np.float32)
    device_input = object()
    device_output = object()
    expected_output = np.zeros_like(source)

    def upload(value):
        assert value is source
        events.append("upload")
        return device_input

    def subtract(value, **kwargs):
        assert value is device_input
        assert kwargs["disable_smoothing"] is False
        assert kwargs["spatial_mode"] == "2D YX"
        events.append("compute")
        return device_output

    def download(value, *, blocking):
        assert value is device_output and blocking
        events.append("download")
        return expected_output

    fake_cupy = SimpleNamespace(
        asarray=upload,
        asnumpy=download,
        cuda=SimpleNamespace(
            get_current_stream=lambda: SimpleNamespace(
                synchronize=lambda: events.append("synchronize")
            )
        ),
    )
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    monkeypatch.setattr(cupy_background, "subtract_background", subtract)
    result = methods.run_method(source, method="vipp_gpu", radius=3)
    assert result is expected_output
    assert events == ["upload", "compute", "download", "synchronize"]


def test_itk_instance_caps_do_not_change_process_global_thread_setting(methods):
    sitk = pytest.importorskip("SimpleITK")
    previous = sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()
    image = np.arange(63, dtype=np.float32).reshape(7, 9)
    methods.run_method(image, method="itk_box", radius=2, threads=1)
    assert sitk.ProcessObject.GetGlobalDefaultNumberOfThreads() == previous
