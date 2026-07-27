from __future__ import annotations

import json
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import replace

import numpy as np
import pytest

import napari_vipp.core.compute_registry as registry_module
from napari_vipp.core.compute_registry import (
    ComputeRegistry,
    ComputeRegistryClosed,
    ComputeRegistryLoadError,
    ImplementationLibraryDescriptor,
    RuntimeDescriptor,
    RuntimeDevice,
    RuntimeExceptionInfo,
    RuntimeExceptionKind,
    RuntimeMemorySnapshot,
    RuntimeProbeResult,
    RuntimeProtocol,
    validate_registry,
)
from napari_vipp.core.compute_specs import AdmissionTier, compute_specs_for


class OpaqueDeviceValue:
    """Fake device storage that fails if host code attempts NumPy coercion."""

    def __init__(self, value) -> None:
        self.value = value

    def __array__(self, _dtype=None, copy=None):
        del copy
        raise AssertionError("Opaque device values must remain runtime-owned.")


class FakeRuntime:
    runtime_id = "fake-device"
    array_domain = "fake-array"

    def __init__(self) -> None:
        self.events: list[object] = []
        self.probe_count = 0
        self.close_count = 0

    def probe(self, *, refresh: bool = False) -> RuntimeProbeResult:
        self.probe_count += 1
        self.events.append(("probe", refresh))
        return RuntimeProbeResult(
            self.runtime_id,
            True,
            version="1.2.3",
            devices=(RuntimeDevice("fake:0", "Opaque device", 8_000),),
            selected_device_id="fake:0",
            environment_fingerprint="fake-environment",
        )

    @contextmanager
    def execution_scope(
        self,
        *,
        device_id: str = "",
        memory_limit_bytes: int | None = None,
        safety_reserve_bytes: int | None = None,
    ):
        self.events.append(
            ("scope_enter", device_id, memory_limit_bytes, safety_reserve_bytes)
        )
        try:
            yield
        finally:
            self.events.append(("scope_exit", device_id))

    def is_device_value(self, value: object) -> bool:
        return isinstance(value, OpaqueDeviceValue)

    def to_device(self, value: object, *, device_id: str = "") -> object:
        self.events.append(("to_device", device_id))
        return OpaqueDeviceValue(value)

    def to_host(self, value: object) -> object:
        assert isinstance(value, OpaqueDeviceValue)
        self.events.append("to_host")
        return value.value

    def release(self, value: object) -> None:
        assert isinstance(value, OpaqueDeviceValue)
        self.events.append("release")

    def synchronize(self, *, device_id: str = "") -> None:
        self.events.append(("synchronize", device_id))

    def memory_snapshot(self, *, device_id: str = "") -> RuntimeMemorySnapshot:
        return RuntimeMemorySnapshot(
            self.runtime_id,
            device_id or "fake:0",
            "discrete",
            device_total_bytes=8_000,
            device_free_bytes=6_000,
            runtime_live_bytes=500,
            runtime_reserved_bytes=1_000,
            out_of_pool_bytes=250,
        )

    def classify_exception(self, exc: BaseException) -> RuntimeExceptionInfo:
        kind = (
            RuntimeExceptionKind.OUT_OF_MEMORY
            if isinstance(exc, MemoryError)
            else RuntimeExceptionKind.UNKNOWN
        )
        return RuntimeExceptionInfo(
            kind,
            "fake_oom" if kind is RuntimeExceptionKind.OUT_OF_MEMORY else "fake_error",
            str(exc),
            exception_type=type(exc).__name__,
            retryable=kind is RuntimeExceptionKind.OUT_OF_MEMORY,
        )

    def close(self) -> None:
        self.close_count += 1
        self.events.append("close")


def _runtime_descriptor(**updates) -> RuntimeDescriptor:
    values = {
        "runtime_id": "fake-device",
        "display_name": "Fake device",
        "factory_ref": "tests.fake_runtime:create_runtime",
        "array_domain": "fake-array",
        "device_domain": "fake-device-domain",
        "supported_os_families": ("Windows", "Linux", "macOS"),
        "interoperability_claims": ("fake-zero-copy-v1",),
    }
    values.update(updates)
    return RuntimeDescriptor(**values)


def _library_descriptor(**updates) -> ImplementationLibraryDescriptor:
    values = {
        "library_id": "fake-library",
        "display_name": "Fake library",
        "runtime_ids": ("fake-device",),
        "array_domain": "fake-array",
        "supported_os_families": ("Windows", "Linux", "macOS"),
    }
    values.update(updates)
    return ImplementationLibraryDescriptor(**values)


def _identity(value):
    return value


def _implementation_spec(**updates):
    values = {
        "implementation_id": "fake-gaussian-v1",
        "runtime_id": "fake-device",
        "array_domain": "fake-array",
        "implementation_library_id": "fake-library",
        "callable_ref": f"{__name__}:_identity",
        "host_boundary": False,
        "admission_tier": AdmissionTier.DEVELOPER_HIDDEN,
        "supports_device_residency": True,
    }
    values.update(updates)
    return replace(compute_specs_for("gaussian_blur")[0], **values)


def _registry(*, factory=None, spec=None) -> ComputeRegistry:
    factories = {} if factory is None else {"fake-device": factory}
    specs = () if spec is None else (spec,)
    return ComputeRegistry(
        runtime_descriptors=(_runtime_descriptor(),),
        library_descriptors=(_library_descriptor(),),
        implementation_specs=specs,
        runtime_factories=factories,
    )


def test_registry_import_and_descriptor_listing_do_not_import_accelerators():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from napari_vipp.core.compute_registry import ComputeRegistry; "
                "registry = ComputeRegistry(); "
                "assert registry.runtime_descriptors[0].runtime_id == 'cuda-cupy'; "
                "assert 'cupy' not in sys.modules; "
                "assert 'cupyx' not in sys.modules; "
                "assert 'cucim' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_runtime_factory_is_lazy_and_one_instance_is_reused():
    instances: list[FakeRuntime] = []

    def create_runtime():
        runtime = FakeRuntime()
        instances.append(runtime)
        return runtime

    registry = _registry(factory=create_runtime)

    assert registry.runtime_descriptor("fake-device").display_name == "Fake device"
    assert registry.library_descriptor("fake-library").runtime_ids == ("fake-device",)
    assert instances == []

    first = registry.runtime("fake-device")
    second = registry.runtime("fake-device")

    assert first is second
    assert instances == [first]
    assert isinstance(first, RuntimeProtocol)
    registry.close()


def test_opaque_runtime_values_cross_only_explicit_runtime_boundaries():
    runtime = FakeRuntime()
    registry = _registry(factory=lambda: runtime)

    with registry.runtime("fake-device").execution_scope(
        device_id="fake:0",
        memory_limit_bytes=4_000,
        safety_reserve_bytes=500,
    ):
        device = runtime.to_device([1, 2, 3], device_id="fake:0")
        assert runtime.is_device_value(device)
        with pytest.raises(AssertionError, match="runtime-owned"):
            np.asarray(device)
        assert runtime.to_host(device) == [1, 2, 3]
        runtime.release(device)

    memory = runtime.memory_snapshot(device_id="fake:0")
    failure = runtime.classify_exception(MemoryError("full"))
    json.dumps(memory.as_dict(), allow_nan=False)
    json.dumps(failure.as_dict(), allow_nan=False)
    assert failure.kind is RuntimeExceptionKind.OUT_OF_MEMORY
    assert failure.retryable
    registry.close()


def test_probe_results_are_cached_until_an_explicit_refresh():
    runtime = FakeRuntime()
    registry = _registry(factory=lambda: runtime)

    first = registry.probe_runtime("fake-device")
    second = registry.probe_runtime("fake-device")
    refreshed = registry.probe_runtime("fake-device", refresh=True)

    assert first is second
    assert refreshed.available
    assert runtime.probe_count == 2
    assert [event for event in runtime.events if event[0] == "probe"] == [
        ("probe", False),
        ("probe", True),
    ]
    json.dumps(refreshed.as_dict(), allow_nan=False)
    registry.close()


def test_probe_turns_lazy_load_failures_into_refreshable_diagnostics():
    attempts = []

    def unavailable_runtime():
        attempts.append("attempt")
        raise ModuleNotFoundError("optional provider is absent")

    registry = _registry(factory=unavailable_runtime)

    first = registry.probe_runtime("fake-device")
    cached = registry.probe_runtime("fake-device")
    refreshed = registry.probe_runtime("fake-device", refresh=True)

    assert first is cached
    assert not refreshed.available
    assert refreshed.reason_code == "runtime_load_failed"
    assert "optional provider is absent" in refreshed.message
    assert attempts == ["attempt", "attempt"]
    registry.close()


def test_runtime_eviction_recreates_and_registry_close_is_terminal():
    instances: list[FakeRuntime] = []

    def create_runtime():
        runtime = FakeRuntime()
        instances.append(runtime)
        return runtime

    registry = _registry(factory=create_runtime)
    assert not registry.release_runtime("fake-device")
    first = registry.runtime("fake-device")
    assert registry.release_runtime("fake-device")
    assert first.close_count == 1

    second = registry.runtime("fake-device")
    assert second is not first
    registry.close()
    registry.close()

    assert second.close_count == 1
    with pytest.raises(ComputeRegistryClosed, match="closed"):
        registry.runtime("fake-device")


def test_invalid_runtime_factory_result_fails_without_caching_it():
    runtime = FakeRuntime()
    runtime.runtime_id = "wrong-runtime"
    registry = _registry(factory=lambda: runtime)

    with pytest.raises(ComputeRegistryLoadError, match="does not match"):
        registry.runtime("fake-device")

    assert runtime.close_count == 1


def test_implementation_lookup_and_import_are_lazy_and_cached(monkeypatch):
    spec = _implementation_spec()
    registry = _registry(factory=FakeRuntime, spec=spec)
    imports = []
    real_import = registry_module.importlib.import_module

    def recording_import(name):
        imports.append(name)
        return real_import(name)

    monkeypatch.setattr(registry_module.importlib, "import_module", recording_import)

    assert registry.implementations_for_operation("gaussian_blur") == ()
    assert registry.implementations_for_operation(
        "gaussian_blur", allow_experimental=True
    ) == (spec,)
    assert imports == []
    with pytest.raises(KeyError, match="developer-hidden"):
        registry.implementation_spec(spec.implementation_id)

    first = registry.implementation_callable(spec, allow_experimental=True)
    second = registry.implementation_callable(spec, allow_experimental=True)

    assert first is _identity
    assert second is first
    assert imports == [__name__]
    registry.close()


def test_implementation_loader_rejects_an_unregistered_spec_variant():
    spec = _implementation_spec()
    registry = _registry(factory=FakeRuntime, spec=spec)

    with pytest.raises(KeyError, match="does not match"):
        registry.implementation_callable(
            replace(spec, callable_ref=f"{__name__}:pytest"),
            allow_experimental=True,
        )

    registry.close()


def test_registry_validation_rejects_broken_runtime_library_links():
    spec = _implementation_spec()
    runtime = _runtime_descriptor()
    library = _library_descriptor()

    validate_registry((runtime,), (library,), (spec,))
    with pytest.raises(ValueError, match="Duplicate runtime ID"):
        validate_registry((runtime, runtime), (library,), ())
    with pytest.raises(ValueError, match="unknown runtime"):
        validate_registry((), (library,), ())
    with pytest.raises(ValueError, match="array domain"):
        validate_registry(
            (runtime,),
            (library,),
            (replace(spec, array_domain="other-array"),),
        )


def test_probe_and_memory_shells_reject_inconsistent_values():
    with pytest.raises(ValueError, match="reported device"):
        RuntimeProbeResult(
            "fake-device",
            True,
            devices=(RuntimeDevice("fake:0", "Fake"),),
            selected_device_id="fake:1",
        )
    with pytest.raises(ValueError, match="must not exceed"):
        RuntimeMemorySnapshot(
            "fake-device",
            "fake:0",
            "discrete",
            device_total_bytes=100,
            device_free_bytes=101,
        )
    with pytest.raises(ValueError, match="live_bytes"):
        RuntimeMemorySnapshot(
            "fake-device",
            "fake:0",
            "discrete",
            runtime_live_bytes=2,
            runtime_reserved_bytes=1,
        )
