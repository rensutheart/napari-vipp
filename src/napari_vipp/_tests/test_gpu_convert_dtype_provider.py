from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from napari_vipp.core.gpu import cupy_convert_dtype as provider
from napari_vipp.core.operations import convert_dtype as cpu_convert_dtype

SOURCE_ROOT = Path(__file__).resolve().parents[2]


class _FakeStream:
    def __init__(self) -> None:
        self.synchronize_count = 0
        self.on_synchronize = None

    def synchronize(self) -> None:
        self.synchronize_count += 1
        if self.on_synchronize is not None:
            self.on_synchronize()


class _FakeCupy:
    def __init__(self, stream: _FakeStream) -> None:
        self.cuda = SimpleNamespace(get_current_stream=lambda: stream)

    def __getattr__(self, name):
        return getattr(np, name)


@dataclass
class _Progress:
    cancel_after_report: int | None = None
    cancelled: bool = False
    reports: list[tuple[int, int, str]] = field(default_factory=list)

    def report(self, current: int, total: int, message: str) -> None:
        self.reports.append((current, total, message))
        if current == self.cancel_after_report:
            self.cancelled = True

    def check_cancelled(self) -> None:
        if self.cancelled:
            raise RuntimeError("cancelled")


@pytest.fixture
def fake_cupy(monkeypatch):
    stream = _FakeStream()
    cupy = _FakeCupy(stream)
    monkeypatch.setattr(provider, "_cupy_module", lambda: cupy)
    return cupy, stream


def test_provider_module_import_is_gpu_lazy_in_a_fresh_process() -> None:
    script = f"""
import sys
sys.path.insert(0, {str(SOURCE_ROOT)!r})
import napari_vipp.core.gpu.cupy_convert_dtype as module
assert module.__all__ == ["convert_dtype"]
for name in ("cupy", "cupyx", "cucim"):
    assert name not in sys.modules, name
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("input_dtype", (bool, np.uint8, np.uint16, np.float32))
@pytest.mark.parametrize("output_dtype", ("bool", "uint8", "uint16", "float32"))
@pytest.mark.parametrize("scaling", ("preserve", "clip", "rescale"))
def test_fake_provider_matches_every_successful_cpu_conversion(
    fake_cupy,
    input_dtype,
    output_dtype,
    scaling,
) -> None:
    del fake_cupy
    values = np.asarray([0, 1, 7, 33, 127, 255], dtype=input_dtype)
    before = values.copy()
    try:
        expected = cpu_convert_dtype(
            values,
            output_dtype=output_dtype,
            scaling=scaling,
        )
    except ValueError as expected_error:
        with pytest.raises(ValueError, match=str(expected_error)):
            provider.convert_dtype(
                values,
                output_dtype=output_dtype,
                scaling=scaling,
            )
    else:
        actual = provider.convert_dtype(
            values,
            output_dtype=output_dtype,
            scaling=scaling,
        )
        np.testing.assert_array_equal(actual, expected, strict=True)
        assert not np.shares_memory(actual, values)
    np.testing.assert_array_equal(values, before)


@pytest.mark.parametrize("output_dtype", ("bool", "uint8", "uint16", "float32"))
@pytest.mark.parametrize("scaling", ("preserve", "clip", "rescale"))
def test_fake_provider_matches_cpu_nonfinite_and_signed_zero_semantics(
    fake_cupy,
    output_dtype,
    scaling,
) -> None:
    del fake_cupy
    values = np.asarray(
        [np.nan, -np.inf, -1.0, -0.0, 0.0, 0.5, 2.0, np.inf],
        dtype=np.float32,
    )
    before = values.view(np.uint32).copy()
    try:
        expected = cpu_convert_dtype(
            values,
            output_dtype=output_dtype,
            scaling=scaling,
        )
    except ValueError as expected_error:
        with pytest.raises(ValueError, match=str(expected_error)):
            provider.convert_dtype(
                values,
                output_dtype=output_dtype,
                scaling=scaling,
            )
    else:
        actual = provider.convert_dtype(
            values,
            output_dtype=output_dtype,
            scaling=scaling,
        )
        np.testing.assert_array_equal(actual, expected, strict=True)
        if np.issubdtype(expected.dtype, np.floating):
            np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))
    np.testing.assert_array_equal(values.view(np.uint32), before)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"output_dtype": "int8"}, "output_dtype must be"),
        ({"scaling": "automatic"}, "scaling must be"),
        (
            {"output_dtype": "uint8", "scaling": "preserve"},
            "exceed the output dtype range",
        ),
    ),
)
def test_fake_provider_preserves_authoritative_validation(fake_cupy, kwargs, message):
    del fake_cupy
    values = np.asarray([0.0, 256.0], dtype=np.float32)

    with pytest.raises(ValueError, match=message):
        provider.convert_dtype(values, **kwargs)


def test_fake_provider_handles_empty_constant_and_noncontiguous_inputs(fake_cupy):
    del fake_cupy
    base = np.arange(60, dtype=np.uint16).reshape(6, 10)
    values = base[::-2, 1::2]
    before = base.copy()

    actual = provider.convert_dtype(
        values,
        output_dtype="float32",
        scaling="preserve",
    )
    constant = provider.convert_dtype(
        np.asarray([5.0, 5.0, np.nan], dtype=np.float32),
        output_dtype="float32",
        scaling="rescale",
    )
    empty = provider.convert_dtype(
        np.empty((0, 3), dtype=np.uint16),
        output_dtype="float32",
        scaling="rescale",
    )

    np.testing.assert_array_equal(
        actual,
        cpu_convert_dtype(values, "float32", "preserve"),
    )
    np.testing.assert_array_equal(
        constant,
        cpu_convert_dtype(
            np.asarray([5.0, 5.0, np.nan], dtype=np.float32),
            "float32",
            "rescale",
        ),
    )
    assert empty.shape == (0, 3)
    assert empty.dtype == np.float32
    np.testing.assert_array_equal(base, before)


def test_progress_reports_only_synchronized_completion_and_honours_cancellation(
    fake_cupy,
) -> None:
    _cupy, stream = fake_cupy
    progress = _Progress()
    values = np.arange(8, dtype=np.uint16)

    output = provider.convert_dtype(
        values,
        output_dtype="float32",
        scaling="preserve",
        progress=progress,
    )

    np.testing.assert_array_equal(output, values.astype(np.float32))
    assert stream.synchronize_count == 1
    assert progress.reports == [
        (0, 1, "Converting dtype"),
        (1, 1, "Converting dtype"),
    ]

    cancelled_before = _Progress(cancel_after_report=0)
    with pytest.raises(RuntimeError, match="cancelled"):
        provider.convert_dtype(
            values,
            output_dtype="float32",
            scaling="preserve",
            progress=cancelled_before,
        )
    assert stream.synchronize_count == 1

    cancelled_after = _Progress()
    stream.on_synchronize = lambda: setattr(cancelled_after, "cancelled", True)
    with pytest.raises(RuntimeError, match="cancelled"):
        provider.convert_dtype(
            values,
            output_dtype="float32",
            scaling="preserve",
            progress=cancelled_after,
        )
    assert stream.synchronize_count == 2
    assert cancelled_after.reports == [(0, 1, "Converting dtype")]


def _real_cuda_or_skip():
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device is unavailable")
        cupy.cuda.runtime.getDevice()
    except Exception as exc:  # pragma: no cover - host-specific failure
        pytest.skip(f"CUDA runtime is unavailable: {exc}")
    return cupy


@pytest.mark.parametrize("input_dtype", (np.uint8, np.uint16))
def test_real_cuda_public_region_is_resident_bitwise_exact_and_nonmutating(
    input_dtype,
) -> None:
    cupy = _real_cuda_or_skip()
    host = np.arange(2 * 31 * 37, dtype=input_dtype).reshape(2, 31, 37)
    device = cupy.asarray(host)[:, :, ::2]
    before = device.copy()

    actual = provider.convert_dtype(
        device,
        output_dtype="float32",
        scaling="preserve",
    )

    assert isinstance(actual, cupy.ndarray)
    assert actual.dtype == cupy.float32
    assert actual.data.ptr != device.data.ptr
    cupy.testing.assert_array_equal(device, before)
    np.testing.assert_array_equal(
        cupy.asnumpy(actual),
        cpu_convert_dtype(host[:, :, ::2], "float32", "preserve"),
        strict=True,
    )


def test_real_cuda_all_existing_conversion_modes_match_cpu_output_bits() -> None:
    cupy = _real_cuda_or_skip()
    inputs = (
        np.asarray([False, True, False, True], dtype=bool),
        np.asarray([0, 1, 7, 33, 127, 255], dtype=np.uint8),
        np.asarray([0, 1, 255, 256, 32_768, 65_535], dtype=np.uint16),
        np.asarray(
            [np.nan, -np.inf, -7.25, -0.0, 0.0, 0.5, 7.25, np.inf],
            dtype=np.float32,
        ),
    )
    for values in inputs:
        for output_dtype in ("bool", "uint8", "uint16", "float32"):
            for scaling in ("preserve", "clip", "rescale"):
                try:
                    expected = cpu_convert_dtype(values, output_dtype, scaling)
                except ValueError as expected_error:
                    with pytest.raises(ValueError, match=str(expected_error)):
                        provider.convert_dtype(
                            cupy.asarray(values),
                            output_dtype,
                            scaling,
                        )
                    continue
                actual = cupy.asnumpy(
                    provider.convert_dtype(
                        cupy.asarray(values),
                        output_dtype,
                        scaling,
                    )
                )
                assert actual.dtype == expected.dtype
                np.testing.assert_array_equal(
                    np.ascontiguousarray(actual).view(np.uint8),
                    np.ascontiguousarray(expected).view(np.uint8),
                )


def test_real_cuda_rescale_preserves_numpy_nan_payload_and_quieting_bits() -> None:
    cupy = _real_cuda_or_skip()
    source_bits = np.asarray(
        [0x7FC00001, 0xFFC00001, 0x7FA12345, 0xFFA12345, 0, 0x3F800000],
        dtype=np.uint32,
    )
    values = source_bits.view(np.float32)

    with pytest.warns(RuntimeWarning, match="invalid value"):
        expected = cpu_convert_dtype(values, "float32", "rescale")
    actual = cupy.asnumpy(
        provider.convert_dtype(
            cupy.asarray(values),
            "float32",
            "rescale",
        )
    )

    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
