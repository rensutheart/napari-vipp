from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import ndimage as scipy_ndimage
from skimage import restoration as skimage_restoration

from napari_vipp.core.compute_benchmark_adapter import operation_parity
from napari_vipp.core.gpu import cucim_background as gpu_background
from napari_vipp.core.operations import (
    rolling_ball_background as cpu_rolling_ball_background,
)
from napari_vipp.core.operations import subtract_background as cpu_subtract_background
from napari_vipp.core.progress import OperationCancelled, ProgressContext


class _FakeStream:
    def __init__(self) -> None:
        self.synchronizations = 0

    def synchronize(self) -> None:
        self.synchronizations += 1


class _FakeCupy:
    bool_ = np.bool_
    float32 = np.float32
    float64 = np.float64
    inf = np.inf

    def __init__(self) -> None:
        self.stream = _FakeStream()
        self.cuda = SimpleNamespace(get_current_stream=lambda: self.stream)
        self.calls: list[dict[str, object]] = []
        self.kernel_constructions: list[dict[str, object]] = []

    def RawKernel(self, source, name, *, options=()):
        self.kernel_constructions.append(
            {
                "source": source,
                "name": name,
                "options": options,
            }
        )
        owner = self

        class FakeBackgroundKernel:
            def __call__(self, grid, block, arguments):
                del grid, block
                if name == "vipp_background_uniform_filter_axis_float32_v2":
                    (
                        values,
                        output,
                        line_count,
                        axis_length,
                        inner_stride,
                    ) = arguments
                    flat_values = np.asarray(values).reshape(-1)
                    flat_output = np.asarray(output).reshape(-1)
                    axis_length = int(axis_length)
                    inner_stride = int(inner_stride)
                    for line in range(int(line_count)):
                        outer, inner = divmod(line, inner_stride)
                        base = outer * axis_length * inner_stride + inner
                        second = 1 if axis_length > 1 else 0
                        total = 2.0 * float(flat_values[base]) + float(
                            flat_values[base + second * inner_stride]
                        )
                        flat_output[base] = np.float32(total / 3.0)
                        for coordinate in range(1, axis_length):
                            entering = min(coordinate + 1, axis_length - 1)
                            leaving = max(coordinate - 2, 0)
                            total += float(flat_values[base + entering * inner_stride])
                            total -= float(flat_values[base + leaving * inner_stride])
                            flat_output[base + coordinate * inner_stride] = np.float32(
                                total / 3.0
                            )
                    return
                if name == "vipp_background_light_transform_float32_v1":
                    values, output, _count, low, high = arguments
                    offset = np.float32(float(low) + float(high))
                    output[...] = (
                        np.float64(offset) - np.asarray(values, dtype=np.float64)
                    ).astype(np.float32)
                    return
                if name == "vipp_background_subtract_float32_v1":
                    (
                        values,
                        background,
                        output,
                        _count,
                        light_background,
                        clip_negative,
                    ) = arguments
                    left, right = (
                        (background, values)
                        if bool(light_background)
                        else (values, background)
                    )
                    result = np.asarray(left, dtype=np.float64) - np.asarray(
                        right,
                        dtype=np.float64,
                    )
                    if bool(clip_negative):
                        result = np.maximum(result, 0.0)
                    output[...] = result.astype(np.float32)
                    return
                if name == "vipp_background_float32_zero_bound_tie_v1":
                    values, _count, low, high = arguments
                    low_bits = int(np.asarray(low).view(np.uint32))
                    high_bits = int(np.asarray(high).view(np.uint32))
                    if not (low_bits & 0x7FFFFFFF) and not (high_bits & 0x7FFFFFFF):
                        finite = np.asarray(values).reshape(-1)
                        finite = finite[np.isfinite(finite)]
                        if finite.size:
                            last = finite[-1]
                            if not (int(np.asarray(last).view(np.uint32)) & 0x7FFFFFFF):
                                low[...] = last
                                high[...] = last
                    return
                values, output, value_count, *parameters = arguments
                del value_count
                extents = tuple(int(value) for value in parameters[:-1])
                radius = int(parameters[-1])
                copied = np.array(values, copy=True).reshape(extents)
                owner.calls.append(
                    {
                        "values": copied,
                        "shape": copied.shape,
                        "radius": radius,
                    }
                )
                expected = skimage_restoration.rolling_ball(copied, radius=radius)
                output[...] = expected

        return FakeBackgroundKernel()

    asarray = staticmethod(np.asarray)
    ascontiguousarray = staticmethod(np.ascontiguousarray)
    zeros_like = staticmethod(np.zeros_like)
    moveaxis = staticmethod(np.moveaxis)
    stack = staticmethod(np.stack)
    empty = staticmethod(np.empty)
    empty_like = staticmethod(np.empty_like)
    isfinite = staticmethod(np.isfinite)
    isposinf = staticmethod(np.isposinf)
    any = staticmethod(np.any)
    argmax = staticmethod(np.argmax)
    min = staticmethod(lambda values: np.asarray(np.min(values)))
    max = staticmethod(lambda values: np.asarray(np.max(values)))
    take = staticmethod(np.take)
    where = staticmethod(np.where)
    maximum = staticmethod(np.maximum)
    nan_to_num = staticmethod(np.nan_to_num)
    rint = staticmethod(np.rint)
    clip = staticmethod(np.clip)


class _FakeNdimage:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def uniform_filter1d(self, values, *, size, axis, output, mode):
        self.calls.append(
            {
                "shape": values.shape,
                "size": size,
                "axis": axis,
                "output": output,
                "mode": mode,
            }
        )
        return scipy_ndimage.uniform_filter1d(
            values,
            size=size,
            axis=axis,
            output=output,
            mode=mode,
        )


@pytest.fixture
def fake_stack(monkeypatch):
    cupy = _FakeCupy()
    ndimage = _FakeNdimage()
    real_import = importlib.import_module
    modules = {
        "cupy": cupy,
        "cupyx.scipy.ndimage": ndimage,
    }

    def load(name: str):
        return modules[name] if name in modules else real_import(name)

    gpu_background._gpu_modules.cache_clear()
    gpu_background._float32_uniform_filter_axis_kernel.cache_clear()
    gpu_background._float32_light_transform_kernel.cache_clear()
    gpu_background._float32_subtract_kernel.cache_clear()
    gpu_background._float32_zero_bound_tie_kernel.cache_clear()
    gpu_background._dynamic_rolling_ball_kernel.cache_clear()
    monkeypatch.setattr(gpu_background.importlib, "import_module", load)
    yield cupy, ndimage, cupy
    gpu_background._gpu_modules.cache_clear()
    gpu_background._float32_uniform_filter_axis_kernel.cache_clear()
    gpu_background._float32_light_transform_kernel.cache_clear()
    gpu_background._float32_subtract_kernel.cache_clear()
    gpu_background._float32_zero_bound_tie_kernel.cache_clear()
    gpu_background._dynamic_rolling_ball_kernel.cache_clear()


def test_import_is_safe_without_cupy_cupyx_or_cucim():
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
    if name == "cupy" or name.startswith("cupyx") or name.startswith("cucim"):
        raise AssertionError(f"optional GPU import attempted: {name}")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
module = importlib.import_module("napari_vipp.core.gpu.cucim_background")
assert callable(module.rolling_ball_background)
assert callable(module.subtract_background)
assert "cupy" not in sys.modules
assert not any(name.startswith("cupyx") for name in sys.modules)
assert not any(name.startswith("cucim") for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr


def test_missing_cupyx_is_raised_only_when_adapter_is_called(monkeypatch):
    cupy = _FakeCupy()
    imports = []

    def load(name: str):
        imports.append(name)
        if name == "cupy":
            return cupy
        raise ModuleNotFoundError(name)

    gpu_background._gpu_modules.cache_clear()
    monkeypatch.setattr(gpu_background.importlib, "import_module", load)

    with pytest.raises(ModuleNotFoundError, match="cupyx.scipy.ndimage"):
        gpu_background.rolling_ball_background(np.ones((3, 3), dtype=np.uint8))
    assert imports == ["cupy", "cupyx.scipy.ndimage"]
    gpu_background._gpu_modules.cache_clear()


@pytest.mark.parametrize("dtype", (np.uint8, np.uint16, np.float32))
@pytest.mark.parametrize("light_background", (False, True))
@pytest.mark.parametrize("disable_smoothing", (False, True))
@pytest.mark.parametrize("operation", ("background", "subtract"))
def test_fake_adapter_matches_complete_cpu_value_and_dtype_contract(
    fake_stack,
    dtype,
    light_background,
    disable_smoothing,
    operation,
):
    host = _fixture(dtype, (7, 9))
    common = {
        "radius": 2.0,
        "light_background": light_background,
        "disable_smoothing": disable_smoothing,
        "spatial_mode": "2D YX",
    }
    if operation == "background":
        expected = cpu_rolling_ball_background(host, **common)
        actual = gpu_background.rolling_ball_background(host, **common)
    else:
        expected = cpu_subtract_background(host, clip_negative=True, **common)
        actual = gpu_background.subtract_background(
            host,
            clip_negative=True,
            **common,
        )

    _assert_exact(expected, actual)


@pytest.mark.parametrize(
    ("shape", "spatial_mode", "channel_axis", "expected_blocks"),
    [
        ((2, 5, 7), "2D YX", None, [(5, 7)] * 2),
        ((2, 3, 5, 7), "3D ZYX", None, [(3, 5, 7)] * 2),
        ((2, 3, 5, 7), "2D YX", 1, [(5, 7)] * 6),
        ((2, 3, 4, 5, 7), "3D ZYX", 2, [(3, 5, 7)] * 8),
    ],
)
def test_fake_adapter_resolves_leading_spatial_and_channel_blocks(
    fake_stack,
    shape,
    spatial_mode,
    channel_axis,
    expected_blocks,
):
    _cupy, _ndimage, restoration = fake_stack
    host = _fixture(np.float32, shape)

    expected = cpu_rolling_ball_background(
        host,
        radius=2,
        disable_smoothing=True,
        spatial_mode=spatial_mode,
        channel_axis=channel_axis,
    )
    actual = gpu_background.rolling_ball_background(
        host,
        radius=2,
        disable_smoothing=True,
        spatial_mode=spatial_mode,
        channel_axis=channel_axis,
    )

    assert [call["shape"] for call in restoration.calls] == expected_blocks
    _assert_exact(expected, actual)


@pytest.mark.parametrize(
    ("resolved_spatial_ndim", "expected_blocks"),
    [
        (2, [(5, 7)] * 6),
        (3, [(3, 5, 7)] * 2),
    ],
)
def test_auto_mode_uses_explicit_resolved_spatial_dimension(
    fake_stack,
    resolved_spatial_ndim,
    expected_blocks,
):
    _cupy, _ndimage, restoration = fake_stack
    host = _fixture(np.float32, (2, 3, 5, 7))
    kwargs = {
        "radius": 2,
        "disable_smoothing": True,
        "spatial_mode": "Auto from axes",
        "resolved_spatial_ndim": resolved_spatial_ndim,
    }

    expected = cpu_rolling_ball_background(host, **kwargs)
    actual = gpu_background.rolling_ball_background(host, **kwargs)

    assert [call["shape"] for call in restoration.calls] == expected_blocks
    _assert_exact(expected, actual)


@pytest.mark.parametrize(
    ("requested", "canonical"),
    [(-5, 1), (0, 1), (1, 1), (1.5, 2), (2.5, 2), (2.6, 3)],
)
def test_radius_uses_cpu_rounding_and_lower_bound(fake_stack, requested, canonical):
    _cupy, _ndimage, restoration = fake_stack

    gpu_background.rolling_ball_background(
        np.arange(35, dtype=np.uint8).reshape(5, 7),
        radius=requested,
        disable_smoothing=True,
    )

    assert [call["radius"] for call in restoration.calls] == [canonical]


@pytest.mark.parametrize(
    ("operation", "dtype"),
    (
        (gpu_background.rolling_ball_background, ">u2"),
        (gpu_background.subtract_background, ">f4"),
    ),
)
def test_provider_rejects_non_native_endian_input(fake_stack, operation, dtype):
    del fake_stack
    host = np.arange(35, dtype=np.dtype(dtype)).reshape(5, 7)

    with pytest.raises(ValueError, match="native-endian input data"):
        operation(host, radius=2)


def test_finite_replacement_and_float_nonfinite_output_match_cpu(fake_stack):
    _cupy, _ndimage, restoration = fake_stack
    host = _fixture(np.float32, (7, 9))
    host.flat[:3] = (np.nan, np.inf, -np.inf)

    expected_background = cpu_rolling_ball_background(
        host,
        radius=2,
        light_background=True,
    )
    actual_background = gpu_background.rolling_ball_background(
        host,
        radius=2,
        light_background=True,
    )
    expected_subtracted = cpu_subtract_background(
        host,
        radius=2,
        light_background=False,
        clip_negative=True,
    )
    actual_subtracted = gpu_background.subtract_background(
        host,
        radius=2,
        light_background=False,
        clip_negative=True,
    )

    assert restoration.calls
    assert all(np.isfinite(call["values"]).all() for call in restoration.calls)
    _assert_exact(expected_background, actual_background)
    _assert_exact(expected_subtracted, actual_subtracted)


@pytest.mark.parametrize("clip_negative", (False, True))
def test_subtraction_clipping_and_light_inversion_match_cpu(
    fake_stack,
    clip_negative,
):
    host = _fixture(np.float32, (2, 7, 9))
    kwargs = {
        "radius": 2,
        "light_background": True,
        "disable_smoothing": False,
        "clip_negative": clip_negative,
        "spatial_mode": "2D per XY slice (advanced)",
    }
    expected = cpu_subtract_background(host, **kwargs)
    actual = gpu_background.subtract_background(host, **kwargs)
    _assert_exact(expected, actual)


def test_progress_reports_identical_completed_blocks(fake_stack):
    host = _fixture(np.uint16, (2, 3, 5, 7))
    cpu_updates = []
    gpu_updates = []
    kwargs = {
        "radius": 2,
        "disable_smoothing": True,
        "spatial_mode": "2D YX",
        "channel_axis": 1,
    }

    cpu_rolling_ball_background(
        host,
        progress=ProgressContext(reporter=cpu_updates.append),
        **kwargs,
    )
    gpu_background.rolling_ball_background(
        host,
        progress=ProgressContext(reporter=gpu_updates.append),
        **kwargs,
    )

    assert gpu_updates == cpu_updates


def test_completed_plane_is_written_and_synchronized_before_progress_is_reported():
    events: list[str] = []

    class TrackedResult:
        def __setitem__(self, index, value) -> None:
            del value
            events.append(f"write:{index!r}")

    class TrackedStream:
        def synchronize(self) -> None:
            events.append("synchronize")

    class TrackedProgress:
        def check_cancelled(self) -> None:
            pass

        def report(self, current, total, message="") -> None:
            events.append(f"report:{current}/{total}:{message}")

    call_count = 0

    def calculate_plane(plane):
        nonlocal call_count
        del plane
        call_count += 1
        events.append(f"calculate:{call_count}")
        return np.zeros((3, 5), dtype=np.float32)

    cupy = SimpleNamespace(
        empty=lambda _shape, dtype: TrackedResult(),
        cuda=SimpleNamespace(get_current_stream=lambda: TrackedStream()),
    )
    result = gpu_background._apply_spatial_blocks(
        np.zeros((2, 3, 5), dtype=np.float32),
        2,
        calculate_plane,
        dtype=np.float32,
        cupy=cupy,
        progress=TrackedProgress(),
        progress_message="Rolling-ball background",
    )

    assert isinstance(result, TrackedResult)
    assert events == [
        "report:0/2:Rolling-ball background",
        "calculate:1",
        "write:(0,)",
        "synchronize",
        "report:1/2:Rolling-ball background",
        "calculate:2",
        "write:(1,)",
        "synchronize",
        "report:2/2:Rolling-ball background",
    ]


@pytest.mark.parametrize(
    ("shape", "expected_synchronizations"),
    (((5, 7), 1), ((4, 5, 7), 4)),
)
def test_every_plane_is_synchronized_without_a_progress_reporter(
    fake_stack,
    shape,
    expected_synchronizations,
):
    cupy, _ndimage, _restoration = fake_stack
    host = _fixture(np.float32, shape)

    gpu_background.subtract_background(
        host,
        radius=2,
        spatial_mode="2D YX",
        progress=None,
    )

    assert cupy.stream.synchronizations == expected_synchronizations


def test_cancellation_is_checked_before_and_after_synchronized_block(fake_stack):
    cupy, _ndimage, restoration = fake_stack
    host = _fixture(np.float32, (5, 7))

    with pytest.raises(OperationCancelled):
        gpu_background.rolling_ball_background(
            host,
            radius=2,
            progress=ProgressContext(cancelled=lambda: True),
        )
    assert restoration.calls == []

    class CancelAfterKernel:
        def __init__(self) -> None:
            self.updates = []

        def check_cancelled(self) -> None:
            if cupy.stream.synchronizations:
                raise OperationCancelled("Operation cancelled.")

        def report(self, current, total, message="") -> None:
            self.check_cancelled()
            self.updates.append((current, total, message))

    progress = CancelAfterKernel()
    with pytest.raises(OperationCancelled):
        gpu_background.rolling_ball_background(
            host,
            radius=2,
            progress=progress,
        )
    assert len(restoration.calls) == 1
    assert progress.updates == [(0, 1, "Rolling-ball background")]


def test_bool_and_scalar_shortcuts_match_cpu_even_with_irrelevant_parameters(
    fake_stack,
):
    bool_image = np.array([[False, True], [True, False]])
    scalar = np.asarray(7, dtype=np.uint16)

    _assert_exact(
        cpu_rolling_ball_background(bool_image, radius="invalid"),
        gpu_background.rolling_ball_background(bool_image, radius="invalid"),
    )
    _assert_exact(
        cpu_subtract_background(bool_image, spatial_mode="invalid"),
        gpu_background.subtract_background(bool_image, spatial_mode="invalid"),
    )
    _assert_exact(
        cpu_rolling_ball_background(scalar, radius="invalid"),
        gpu_background.rolling_ball_background(scalar, radius="invalid"),
    )
    _assert_exact(
        cpu_subtract_background(scalar, spatial_mode="invalid"),
        gpu_background.subtract_background(scalar, spatial_mode="invalid"),
    )


@pytest.mark.parametrize(
    ("shape", "kwargs"),
    [
        ((5, 7, 3), {"channel_axis": True}),
        ((5, 7), {"channel_axis": 0}),
        ((5, 7, 3), {"channel_axis": 3}),
        ((2, 5, 7), {"spatial_mode": "Auto from axes"}),
        (
            (2, 5, 7),
            {"spatial_mode": "Auto from axes", "resolved_spatial_ndim": 4},
        ),
        ((5, 7), {"spatial_mode": "3D ZYX"}),
        ((5, 7), {"spatial_mode": "invalid"}),
    ],
)
def test_parameter_validation_messages_match_cpu(fake_stack, shape, kwargs):
    host = np.zeros(shape, dtype=np.uint8)
    with pytest.raises(ValueError) as cpu_error:
        cpu_rolling_ball_background(host, **kwargs)
    with pytest.raises(ValueError) as gpu_error:
        gpu_background.rolling_ball_background(host, **kwargs)
    assert str(gpu_error.value) == str(cpu_error.value)


def test_input_is_not_mutated_and_smoothing_is_size_three_nearest(fake_stack):
    cupy, ndimage, _restoration = fake_stack
    host = _fixture(np.float32, (2, 7, 9)).swapaxes(1, 2)
    original = host.copy()
    host.setflags(write=False)

    output = gpu_background.subtract_background(host, radius=2)

    np.testing.assert_array_equal(host, original)
    assert output.shape == host.shape
    assert not ndimage.calls
    assert any(
        item["name"] == "vipp_background_uniform_filter_axis_float32_v2"
        for item in cupy.kernel_constructions
    )


def test_dynamic_kernel_is_constructed_once_across_interactive_radius_edits(
    fake_stack,
):
    cupy, _ndimage, provider = fake_stack
    host = _fixture(np.float32, (7, 9))

    for radius in (2, 3, 4, 5, 3, 4):
        expected = cpu_subtract_background(
            host,
            radius=radius,
            disable_smoothing=True,
        )
        actual = gpu_background.subtract_background(
            host,
            radius=radius,
            disable_smoothing=True,
        )
        _assert_exact(expected, actual)

    assert [call["radius"] for call in provider.calls] == [2, 3, 4, 5, 3, 4]
    constructions = {item["name"]: item for item in cupy.kernel_constructions}
    assert set(constructions) == {
        "vipp_background_float32_zero_bound_tie_v1",
        "vipp_dynamic_rolling_ball_2d_float32_v1",
        "vipp_background_subtract_float32_v1",
    }
    construction = constructions["vipp_dynamic_rolling_ball_2d_float32_v1"]
    assert construction["name"] == "vipp_dynamic_rolling_ball_2d_float32_v1"
    assert "const int radius" in construction["source"]
    assert "for (int delta_0 = -radius" in construction["source"]
    assert construction["options"] == gpu_background._KERNEL_OPTIONS


@pytest.fixture(scope="module")
def real_gpu_stack():
    gpu_background._gpu_modules.cache_clear()
    if importlib.util.find_spec("cupy") is None:
        pytest.skip("CuPy is not installed.")
    try:
        cupy = importlib.import_module("cupy")
    except Exception as exc:
        pytest.fail(f"The installed CuPy stack could not import: {exc}")
    try:
        if int(cupy.cuda.runtime.getDeviceCount()) < 1:
            pytest.skip("No CUDA device is available.")
        cupy.zeros(1, dtype=cupy.float32).sum().item()
    except Exception as exc:
        pytest.skip(f"A working CUDA device is unavailable: {exc}")
    return cupy


_REAL_CASES = (
    (
        "background_dark_smoothed_2d",
        "background",
        (7, 9),
        {
            "radius": 2,
            "light_background": False,
            "disable_smoothing": False,
            "spatial_mode": "2D YX",
        },
    ),
    (
        "background_light_unsmoothed_3d",
        "background",
        (2, 5, 7),
        {
            "radius": 2,
            "light_background": True,
            "disable_smoothing": True,
            "spatial_mode": "3D ZYX",
        },
    ),
    (
        "background_dark_smoothed_leading_3d",
        "background",
        (2, 3, 5, 7),
        {
            "radius": 2,
            "light_background": False,
            "disable_smoothing": False,
            "spatial_mode": "3D ZYX",
        },
    ),
    (
        "subtract_dark_leading_2d",
        "subtract",
        (2, 5, 7),
        {
            "radius": 2,
            "light_background": False,
            "disable_smoothing": False,
            "clip_negative": True,
            "spatial_mode": "2D YX",
        },
    ),
    (
        "subtract_light_channel_axis",
        "subtract",
        (2, 3, 5, 7),
        {
            "radius": 2,
            "light_background": True,
            "disable_smoothing": True,
            "clip_negative": False,
            "spatial_mode": "2D YX",
            "channel_axis": 1,
        },
    ),
)


@pytest.mark.parametrize("dtype", (np.uint8, np.uint16, np.float32))
@pytest.mark.parametrize(
    ("region", "operation", "shape", "kwargs"),
    _REAL_CASES,
    ids=[case[0] for case in _REAL_CASES],
)
def test_real_rtx_cucim_exact_cpu_parity_for_advertised_regions(
    real_gpu_stack,
    dtype,
    region,
    operation,
    shape,
    kwargs,
):
    cupy = real_gpu_stack
    host = _fixture(dtype, shape)
    device = cupy.asarray(host)
    original = device.copy()

    if operation == "background":
        expected = cpu_rolling_ball_background(host, **kwargs)
        output = gpu_background.rolling_ball_background(device, **kwargs)
    else:
        expected = cpu_subtract_background(host, **kwargs)
        output = gpu_background.subtract_background(device, **kwargs)
    cupy.cuda.get_current_stream().synchronize()
    actual = cupy.asnumpy(output)

    assert isinstance(output, cupy.ndarray)
    assert output.data.ptr != device.data.ptr
    cupy.testing.assert_array_equal(device, original)
    _assert_exact_with_region(expected, actual, region=region, dtype=dtype)


@pytest.mark.parametrize("operation", ("background", "subtract"))
def test_real_rtx_cucim_exact_nonfinite_policy_parity(real_gpu_stack, operation):
    cupy = real_gpu_stack
    host = _fixture(np.float32, (7, 9))
    host.flat[:3] = (np.nan, np.inf, -np.inf)
    device = cupy.asarray(host)
    kwargs = {
        "radius": 2,
        "light_background": True,
        "disable_smoothing": False,
    }

    if operation == "background":
        expected = cpu_rolling_ball_background(host, **kwargs)
        output = gpu_background.rolling_ball_background(device, **kwargs)
    else:
        expected = cpu_subtract_background(
            host,
            clip_negative=True,
            **kwargs,
        )
        output = gpu_background.subtract_background(
            device,
            clip_negative=True,
            **kwargs,
        )
    actual = cupy.asnumpy(output)

    _assert_exact_with_region(
        expected,
        actual,
        region=f"nonfinite_{operation}",
        dtype=np.float32,
    )


@pytest.mark.parametrize("tiny_case", ("generated", "input"))
@pytest.mark.parametrize("light_background", (False, True))
@pytest.mark.parametrize("disable_smoothing", (False, True))
def test_real_rtx_float32_tiny_and_signed_zero_exact_parity(
    real_gpu_stack,
    tiny_case,
    light_background,
    disable_smoothing,
):
    """Keep strict CPU bits where CUDA normally flushes float32 subnormals."""

    cupy = real_gpu_stack
    smallest = np.nextafter(np.float32(0), np.float32(1), dtype=np.float32)
    if tiny_case == "generated":
        host = np.array(
            [[np.finfo(np.float32).tiny, 0.0], [0.0, 0.0]],
            dtype=np.float32,
        )
    else:
        host = np.array(
            [
                [np.finfo(np.float32).tiny, smallest],
                [np.float32(-0.0), np.float32(0.0)],
            ],
            dtype=np.float32,
        )
    common = {
        "radius": 2,
        "light_background": light_background,
        "disable_smoothing": disable_smoothing,
    }

    expected_background = cpu_rolling_ball_background(host, **common)
    actual_background = cupy.asnumpy(
        gpu_background.rolling_ball_background(cupy.asarray(host), **common)
    )
    _assert_exact_with_region(
        expected_background,
        actual_background,
        region=f"tiny_{tiny_case}_background",
        dtype=np.float32,
    )

    for clip_negative in (False, True):
        expected_subtracted = cpu_subtract_background(
            host,
            clip_negative=clip_negative,
            **common,
        )
        actual_subtracted = cupy.asnumpy(
            gpu_background.subtract_background(
                cupy.asarray(host),
                clip_negative=clip_negative,
                **common,
            )
        )
        _assert_exact_with_region(
            expected_subtracted,
            actual_subtracted,
            region=(f"tiny_{tiny_case}_subtract_clip_{int(clip_negative)}"),
            dtype=np.float32,
        )


def test_real_rtx_light_background_signed_zero_permutations(real_gpu_stack):
    """Match NumPy's last-equal-lane sign for all four-value zero layouts."""

    cupy = real_gpu_stack
    for sign_bits in range(16):
        host = np.array(
            [
                np.float32(-0.0 if sign_bits & (1 << index) else 0.0)
                for index in range(4)
            ],
            dtype=np.float32,
        ).reshape(2, 2)
        common = {
            "radius": 2,
            "light_background": True,
            "disable_smoothing": True,
        }
        expected_background = cpu_rolling_ball_background(host, **common)
        actual_background = cupy.asnumpy(
            gpu_background.rolling_ball_background(cupy.asarray(host), **common)
        )
        _assert_exact_with_region(
            expected_background,
            actual_background,
            region=f"signed_zero_background_{sign_bits:04b}",
            dtype=np.float32,
        )

        expected_subtracted = cpu_subtract_background(
            host,
            clip_negative=False,
            **common,
        )
        actual_subtracted = cupy.asnumpy(
            gpu_background.subtract_background(
                cupy.asarray(host),
                clip_negative=False,
                **common,
            )
        )
        _assert_exact_with_region(
            expected_subtracted,
            actual_subtracted,
            region=f"signed_zero_subtract_{sign_bits:04b}",
            dtype=np.float32,
        )


_RANDOMIZED_FLOAT32_PARITY_CASES = (
    ("2d-dark-smoothed", (31, 37), 5, False, False, "2D YX", None, 1.0),
    ("2d-light-smoothed-high", (17, 19), 2, True, False, "2D YX", None, 1e5),
    ("leading-2d-dark", (2, 9, 11), 2, False, False, "2D YX", None, 120.0),
    ("3d-dark-smoothed", (5, 7, 9), 2, False, False, "3D ZYX", None, 120.0),
    (
        "channel-blocks-light",
        (2, 3, 7, 9),
        2,
        True,
        False,
        "2D YX",
        1,
        1.0,
    ),
    ("2d-dark-unsmoothed", (17, 19), 5, False, True, "2D YX", None, 120.0),
)


@pytest.mark.parametrize("seed", (107, 811))
@pytest.mark.parametrize(
    (
        "region",
        "shape",
        "radius",
        "light_background",
        "disable_smoothing",
        "spatial_mode",
        "channel_axis",
        "scale",
    ),
    _RANDOMIZED_FLOAT32_PARITY_CASES,
    ids=[case[0] for case in _RANDOMIZED_FLOAT32_PARITY_CASES],
)
@pytest.mark.parametrize("operation", ("background", "subtract"))
def test_real_rtx_randomized_float32_background_v2_parity(
    real_gpu_stack,
    seed,
    region,
    shape,
    radius,
    light_background,
    disable_smoothing,
    spatial_mode,
    channel_axis,
    scale,
    operation,
):
    """Exercise the v2 bounded-float policy beyond bitwise-friendly fixtures."""

    cupy = real_gpu_stack
    rng = np.random.default_rng(seed + sum(shape))
    spread = max(abs(scale) * 0.2, 1e-5)
    host = rng.normal(scale, spread, size=shape).astype(np.float32)
    host.flat[:5] = np.asarray(
        (-spread, 0.0, scale + 4 * spread, 7.0, 7.0),
        dtype=np.float32,
    )
    if seed == 811 and region == "channel-blocks-light":
        host.flat[:3] = (np.nan, np.inf, -np.inf)
    kwargs = {
        "radius": radius,
        "light_background": light_background,
        "disable_smoothing": disable_smoothing,
        "spatial_mode": spatial_mode,
        "channel_axis": channel_axis,
    }
    device = cupy.asarray(host)
    if operation == "background":
        expected = cpu_rolling_ball_background(host, **kwargs)
        output = gpu_background.rolling_ball_background(device, **kwargs)
        operation_id = "rolling_ball_background"
    else:
        expected = cpu_subtract_background(
            host,
            clip_negative=bool(seed % 2),
            **kwargs,
        )
        output = gpu_background.subtract_background(
            device,
            clip_negative=bool(seed % 2),
            **kwargs,
        )
        operation_id = "subtract_background"
    actual = cupy.asnumpy(output)
    finite_input = np.asarray(host)[np.isfinite(host)]
    input_peak = (
        float(np.max(np.abs(finite_input.astype(np.float64))))
        if finite_input.size
        else 0.0
    )

    parity = operation_parity(
        operation_id,
        expected,
        actual,
        input_peak=input_peak,
    )

    assert parity.passed, f"{region}/{operation}/{seed}: {parity.detail}"


@pytest.mark.parametrize("operation", ("background", "subtract"))
@pytest.mark.parametrize(
    ("region", "shape", "kwargs"),
    (
        (
            "public_radius_500_2d",
            (17, 19),
            {"radius": 500, "spatial_mode": "2D YX"},
        ),
        (
            "admitted_radius_50_3d",
            (2, 3, 4),
            {"radius": 50, "spatial_mode": "3D ZYX"},
        ),
    ),
)
def test_real_rtx_cucim_exact_radius_boundaries(
    real_gpu_stack,
    operation,
    region,
    shape,
    kwargs,
):
    """Exercise the public 2D maximum and memory-safe admitted 3D maximum.

    A radius-500 3D footprint contains more than one billion elements and is
    intentionally rejected by policy; radius 50 is the reviewed 3D cap.
    """
    cupy = real_gpu_stack
    host = _fixture(np.uint16, shape)
    device = cupy.asarray(host)
    if operation == "background":
        expected = cpu_rolling_ball_background(host, **kwargs)
        output = gpu_background.rolling_ball_background(device, **kwargs)
    else:
        expected = cpu_subtract_background(host, **kwargs)
        output = gpu_background.subtract_background(device, **kwargs)
    actual = cupy.asnumpy(output)

    _assert_exact_with_region(
        expected,
        actual,
        region=f"{region}_{operation}",
        dtype=np.uint16,
    )


def test_real_cucim_uses_runtime_private_allocator_and_common_array_domain(
    real_gpu_stack,
):
    from napari_vipp.core.gpu.cupy_runtime import create_runtime

    cupy = real_gpu_stack
    host = _fixture(np.uint16, (7, 9))
    expected = cpu_subtract_background(host, radius=2)
    runtime = create_runtime()
    try:
        assert runtime.probe().available
        with runtime.execution_scope(
            memory_limit_bytes=512 * 1024**2,
            safety_reserve_bytes=64 * 1024**2,
        ):
            device = runtime.to_device(host)
            output = gpu_background.subtract_background(device, radius=2)
            actual = runtime.to_host(output)
            snapshot = runtime.memory_snapshot()
            assert isinstance(output, cupy.ndarray)
            assert snapshot.runtime_live_bytes >= device.nbytes + output.nbytes
            runtime.release(output)
            runtime.release(device)
            # ``release`` relinquishes VIPP ownership but never force-frees a
            # CuPy allocation while Python aliases are still live.  Dropping
            # the final references returns both allocations to the private pool.
            del output, device
            assert runtime.memory_snapshot().runtime_live_bytes == 0
    finally:
        runtime.close()

    _assert_exact_with_region(
        expected,
        actual,
        region="private_allocator_subtract",
        dtype=np.uint16,
    )


def _fixture(dtype, shape):
    seed = sum((axis + 1) * extent for axis, extent in enumerate(shape))
    seed += np.dtype(dtype).itemsize * 1009
    rng = np.random.default_rng(seed)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        values = rng.integers(0, info.max + 1, size=shape, dtype=dtype)
        values.flat[:5] = (0, info.max, info.max // 2, 7, 7)
        return values
    values = rng.normal(loc=120.0, scale=35.0, size=shape).astype(dtype)
    values.flat[:5] = (-20.5, 0.0, 255.25, 7.0, 7.0)
    return values


def _assert_exact(expected, actual):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)
    if np.issubdtype(expected.dtype, np.floating):
        finite = np.isfinite(expected) & np.isfinite(actual)
        unsigned = np.dtype(f"u{expected.dtype.itemsize}")
        np.testing.assert_array_equal(
            np.ascontiguousarray(actual[finite]).view(unsigned),
            np.ascontiguousarray(expected[finite]).view(unsigned),
        )


def _assert_exact_with_region(expected, actual, *, region: str, dtype) -> None:
    try:
        _assert_exact(expected, actual)
    except AssertionError as exc:
        finite = np.isfinite(expected) & np.isfinite(actual)
        maximum = (
            float(np.max(np.abs(expected[finite] - actual[finite])))
            if np.any(finite)
            else float("nan")
        )
        pytest.fail(
            f"Exact CPU/cuCIM parity failed in {region} for {np.dtype(dtype)}; "
            f"max finite absolute error={maximum}.\n{exc}",
            pytrace=False,
        )
