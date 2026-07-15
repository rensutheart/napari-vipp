from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from napari_vipp.core.compute import (
    BackendCapability,
    ComputeBackend,
    ComputeBackendUnavailable,
    ComputeCapabilityReport,
    detect_compute_capabilities,
    select_compute_backend,
)


def _report(*, gpu_available=True, operations=("gaussian_blur",)):
    return ComputeCapabilityReport(
        cpu=BackendCapability(
            ComputeBackend.CPU,
            True,
            "NumPy/SciPy/scikit-image",
            "runtime",
            "Test CPU",
            (),
        ),
        gpu=BackendCapability(
            ComputeBackend.GPU,
            gpu_available,
            "CuPy",
            "14.1.1",
            "Test GPU" if gpu_available else "",
            tuple(operations) if gpu_available else (),
            "" if gpu_available else "No CUDA device.",
        ),
    )


def test_compute_backend_parse_is_strict_and_case_insensitive():
    assert ComputeBackend.parse(" AUTO ") is ComputeBackend.AUTO
    assert ComputeBackend.parse(ComputeBackend.GPU) is ComputeBackend.GPU
    with pytest.raises(ValueError, match="Unsupported compute backend"):
        ComputeBackend.parse("fastest")


def test_auto_uses_gpu_only_for_a_validated_operation():
    supported = select_compute_backend(
        "auto", "gaussian_blur", capabilities=_report()
    )
    unsupported = select_compute_backend(
        "auto", "median_filter", capabilities=_report()
    )

    assert supported.resolved is ComputeBackend.GPU
    assert not supported.fell_back
    assert unsupported.resolved is ComputeBackend.CPU
    assert unsupported.fell_back
    assert "no validated GPU implementation" in unsupported.reason


def test_explicit_gpu_fails_closed_unless_fallback_is_enabled():
    with pytest.raises(ComputeBackendUnavailable, match="No CUDA device"):
        select_compute_backend(
            "gpu", "gaussian_blur", capabilities=_report(gpu_available=False)
        )

    selection = select_compute_backend(
        "gpu",
        "gaussian_blur",
        capabilities=_report(gpu_available=False),
        allow_explicit_gpu_fallback=True,
    )
    assert selection.resolved is ComputeBackend.CPU
    assert selection.fell_back


def test_cpu_selection_does_not_require_gpu_availability():
    selection = select_compute_backend(
        "cpu", "anything", capabilities=_report(gpu_available=False)
    )
    assert selection.resolved is ComputeBackend.CPU
    assert not selection.fell_back


def test_detection_handles_missing_cupy_without_import_time_failure(
    monkeypatch,
):
    def missing(_name):
        raise ModuleNotFoundError("No module named 'cupy'")

    monkeypatch.setattr("napari_vipp.core.compute.importlib.import_module", missing)
    report = detect_compute_capabilities()

    assert report.cpu.available
    assert not report.gpu.available
    assert report.gpu.supported_operation_ids == ()
    assert "not importable" in report.gpu.reason
    json.dumps(report.as_dict())


def test_detection_requires_a_real_cuda_probe(monkeypatch):
    runtime = SimpleNamespace(
        getDeviceCount=lambda: 1,
        getDeviceProperties=lambda _index: {"name": b"Mock GPU"},
    )
    stream = SimpleNamespace(synchronize=lambda: None)
    fake_cupy = SimpleNamespace(
        __version__="99.0",
        cuda=SimpleNamespace(
            runtime=runtime,
            get_current_stream=lambda: stream,
        ),
        arange=lambda *_args, **_kwargs: object(),
        float32="float32",
    )
    monkeypatch.setattr(
        "napari_vipp.core.compute.importlib.import_module",
        lambda name: fake_cupy if name == "cupy" else None,
    )

    report = detect_compute_capabilities(
        supported_gpu_operation_ids=("median_filter", "gaussian_blur")
    )

    assert report.gpu.available
    assert report.gpu.device_name == "Mock GPU"
    assert report.gpu.supported_operation_ids == (
        "gaussian_blur",
        "median_filter",
    )
    assert report.gpu.reason == ""


def test_importable_cupy_with_failed_probe_is_unavailable(monkeypatch):
    runtime = SimpleNamespace(
        getDeviceCount=lambda: 1,
        getDeviceProperties=lambda _index: {"name": b"Mock GPU"},
    )
    fake_cupy = SimpleNamespace(
        __version__="99.0",
        cuda=SimpleNamespace(runtime=runtime),
        arange=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("driver mismatch")
        ),
        float32="float32",
    )
    monkeypatch.setattr(
        "napari_vipp.core.compute.importlib.import_module", lambda _name: fake_cupy
    )

    report = detect_compute_capabilities(
        supported_gpu_operation_ids=("gaussian_blur",)
    )

    assert not report.gpu.available
    assert report.gpu.supported_operation_ids == ()
    assert "driver mismatch" in report.gpu.reason
