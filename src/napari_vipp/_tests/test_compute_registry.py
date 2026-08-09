from __future__ import annotations

import hashlib
import json
import re
import struct
import subprocess
import sys
import threading
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path, PurePosixPath

import numpy as np
import pytest

import napari_vipp.core.compute_registry as registry_module
from napari_vipp.core.accelerator_lease import AcceleratorLeaseManager
from napari_vipp.core.compute_policy import (
    PHASE1_CUCIM_BUILD_RECIPE_ID,
    PHASE1_CUCIM_SOURCE_COMMIT,
    PHASE1_CUCIM_SOURCE_TAG,
    PHASE1_CUCIM_WHEEL_PAYLOAD_SHA256,
)
from napari_vipp.core.compute_registry import (
    ComputeRegistry,
    ComputeRegistryClosed,
    ComputeRegistryLoadError,
    ImplementationLibraryDescriptor,
    ImplementationLibraryProbeResult,
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

    def allocation_identity(self, value: object):
        if not self.is_device_value(value):
            raise TypeError("not an opaque device allocation")
        return id(value)

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
                "assert {item.library_id for item in registry.library_descriptors} "
                "== {'cupy', 'cupyx', 'cucim'}; "
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


def test_builtin_libraries_declare_common_zero_copy_interoperability():
    registry = ComputeRegistry()

    assert registry.interoperability_contract("cuda-cupy", ("cupyx", "cucim")) == (
        "cupy-array-stream-device-lifetime-v1",
    )
    registry.close()


def test_library_probe_is_explicit_cached_and_refreshable():
    calls = []

    def probe():
        calls.append("probe")
        return ImplementationLibraryProbeResult(
            "fake-library",
            True,
            version="2.0",
            message="ready",
        )

    registry = ComputeRegistry(
        runtime_descriptors=(_runtime_descriptor(),),
        library_descriptors=(_library_descriptor(),),
        implementation_specs=(),
        runtime_factories={"fake-device": FakeRuntime},
        library_probes={"fake-library": probe},
    )

    first = registry.probe_library("fake-library")
    second = registry.probe_library("fake-library")
    refreshed = registry.probe_library("fake-library", refresh=True)

    assert first is second
    assert refreshed.available
    assert calls == ["probe", "probe"]
    registry.close()


@pytest.mark.parametrize("probe_kind", ("runtime", "library"))
def test_provider_probe_wait_does_not_hold_registry_lock(probe_kind) -> None:
    """Lease owner may resolve a callable while a provider probe is queued."""

    manager = AcceleratorLeaseManager()
    probe_attempted_lease = threading.Event()
    owner_has_lease = threading.Event()
    callable_resolved = threading.Event()
    probe_finished = threading.Event()
    failures: list[BaseException] = []

    def probe() -> ImplementationLibraryProbeResult:
        probe_attempted_lease.set()
        with manager.acquire("cuda-cupy", "cuda:0"):
            return ImplementationLibraryProbeResult(
                "fake-library",
                True,
                version="2.0",
                message="ready",
            )

    class LeaseProbeRuntime(FakeRuntime):
        def probe(self, *, refresh: bool = False) -> RuntimeProbeResult:
            probe_attempted_lease.set()
            with manager.acquire("cuda-cupy", "cuda:0"):
                return super().probe(refresh=refresh)

    spec = _implementation_spec()
    registry = ComputeRegistry(
        runtime_descriptors=(_runtime_descriptor(),),
        library_descriptors=(_library_descriptor(),),
        implementation_specs=(spec,),
        runtime_factories={"fake-device": LeaseProbeRuntime},
        library_probes={"fake-library": probe},
    )

    def own_device_then_resolve_callable() -> None:
        try:
            with manager.acquire("cuda-cupy", "cuda:0"):
                owner_has_lease.set()
                if not probe_attempted_lease.wait(timeout=5):
                    raise AssertionError("provider probe did not queue for the lease")
                registry.implementation_callable(spec, allow_experimental=True)
                callable_resolved.set()
        except BaseException as exc:
            failures.append(exc)

    def run_probe() -> None:
        try:
            if not owner_has_lease.wait(timeout=5):
                raise AssertionError("execution did not acquire the lease")
            result = (
                registry.probe_runtime("fake-device")
                if probe_kind == "runtime"
                else registry.probe_library("fake-library")
            )
            if not result.available:
                raise AssertionError(result.message)
            probe_finished.set()
        except BaseException as exc:
            failures.append(exc)

    owner_thread = threading.Thread(
        target=own_device_then_resolve_callable,
        daemon=True,
    )
    probe_thread = threading.Thread(target=run_probe, daemon=True)
    owner_thread.start()
    assert owner_has_lease.wait(timeout=5)
    probe_thread.start()
    owner_thread.join(timeout=5)
    probe_thread.join(timeout=5)

    assert callable_resolved.is_set(), "process owner deadlocked on registry lock"
    assert probe_finished.is_set(), "provider probe deadlocked on process lease"
    assert not owner_thread.is_alive()
    assert not probe_thread.is_alive()
    assert failures == []
    registry.close()


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


def _cucim_environment_record(
    path: Path,
    *,
    digest: str = "a" * 64,
    payload_digest: str = PHASE1_CUCIM_WHEEL_PAYLOAD_SHA256,
    source_tag: str = PHASE1_CUCIM_SOURCE_TAG,
    source_commit: str = PHASE1_CUCIM_SOURCE_COMMIT,
    build_recipe_id: str = PHASE1_CUCIM_BUILD_RECIPE_ID,
    cucim: bool = True,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": "napari-vipp-gpu-environment",
                "schema_version": 2,
                "track": "cuda13",
                "cupy_distribution": "cupy-cuda13x",
                "cucim": (
                    {
                        "distribution": "cucim-cu13",
                        "wheel_sha256": digest,
                        "wheel_payload_sha256": payload_digest,
                        "source_tag": source_tag,
                        "source_commit": source_commit,
                        "build_recipe_id": build_recipe_id,
                    }
                    if cucim
                    else None
                ),
            }
        ),
        encoding="utf-8",
    )
    return path


class _FakeDistribution:
    def __init__(
        self,
        name: str,
        *,
        version: str = "26.6.0",
        archive_sha256: str | None = None,
        direct_url_text: str | None = None,
    ) -> None:
        self.metadata = {"Name": name}
        self.version = version
        self._direct_url_text = direct_url_text
        if archive_sha256 is not None:
            self._direct_url_text = json.dumps(
                {
                    "archive_info": {
                        "hash": f"sha256={archive_sha256}",
                        "hashes": {"sha256": archive_sha256},
                    },
                    "url": "file:///verified-cucim.whl",
                }
            )

    def read_text(self, filename: str) -> str | None:
        assert filename == "direct_url.json"
        return self._direct_url_text


def _mock_installed_gpu_distributions(
    monkeypatch,
    *,
    cucim_digest: str = "a" * 64,
    cucim_payload_digest: str = PHASE1_CUCIM_WHEEL_PAYLOAD_SHA256,
    cupy_names: tuple[str, ...] = ("cupy-cuda13x",),
    cucim_direct_url: str | None = None,
    cucim_runtime_versions: dict[str, str] | None = None,
) -> None:
    cupy_distributions = [_FakeDistribution(name) for name in cupy_names]
    cucim_distribution = _FakeDistribution(
        "cucim-cu13",
        archive_sha256=(cucim_digest if cucim_direct_url is None else None),
        direct_url_text=cucim_direct_url,
    )
    runtime_versions = {
        "click": "8.4.2",
        "lazy-loader": "0.5",
        "nvidia-nvimgcodec-cu13": "0.8.0.22",
    }
    if cucim_runtime_versions is not None:
        runtime_versions.update(cucim_runtime_versions)
    runtime_distributions = {
        name: _FakeDistribution(name, version=version)
        for name, version in runtime_versions.items()
        if version
    }
    monkeypatch.setattr(
        registry_module.importlib.metadata,
        "distributions",
        lambda: iter(cupy_distributions),
    )
    monkeypatch.setattr(
        registry_module.importlib.metadata,
        "distribution",
        lambda name: (
            cucim_distribution
            if name == "cucim-cu13"
            else runtime_distributions.get(name)
            or (_ for _ in ()).throw(
                registry_module.importlib.metadata.PackageNotFoundError(name)
            )
        ),
    )
    monkeypatch.setattr(
        registry_module,
        "_installed_cucim_wheel_payload_sha256",
        lambda distribution: cucim_payload_digest,
    )


def test_builtin_library_probes_use_private_pools_and_expose_provenance(
    monkeypatch,
    tmp_path,
):
    events = []
    lease_calls: list[tuple[str, str]] = []

    @contextmanager
    def observed_lease(runtime_id: str, device_id: str):
        lease_calls.append((runtime_id, device_id))
        yield

    monkeypatch.setattr(registry_module, "accelerator_lease", observed_lease)

    class Pool:
        def __init__(self):
            events.append("pool-created")

        def malloc(self, _size):
            raise AssertionError("NumPy fixtures do not invoke the allocator")

        def free_all_blocks(self):
            events.append("free")

        @staticmethod
        def used_bytes():
            return 0

        @staticmethod
        def total_bytes():
            return 0

    class Stream:
        def synchronize(self):
            events.append("sync")

    class Cuda:
        MemoryPool = Pool

        @staticmethod
        def get_current_stream():
            return Stream()

        @staticmethod
        @contextmanager
        def using_allocator(allocator):
            assert getattr(allocator, "__self__", None).__class__ is Pool
            events.append("allocator-enter")
            try:
                yield
            finally:
                events.append("allocator-exit")

    class Cupy:
        __version__ = "test-cupy"
        float32 = np.float32
        cuda = Cuda()

        @staticmethod
        def arange(*args, **kwargs):
            return np.arange(*args, **kwargs)

        @staticmethod
        def asarray(*args, **kwargs):
            return np.asarray(*args, **kwargs)

        @staticmethod
        def get_default_memory_pool():
            raise AssertionError("A library probe touched the global default pool")

    class Ndimage:
        @staticmethod
        def gaussian_filter(values, **_kwargs):
            events.append("gaussian")
            return values.copy()

        @staticmethod
        def median_filter(values, **_kwargs):
            events.append("median")
            return values.copy()

        @staticmethod
        def label(_values, **_kwargs):
            events.append("label")
            return np.asarray([[1, 0], [0, 2]], dtype=np.int32), 2

    class Signal:
        @staticmethod
        def convolve(values, _kernel, **_kwargs):
            events.append("convolve")
            return values.copy()

    class Cucim:
        __version__ = "test-cucim"

        @staticmethod
        def is_available(component):
            return component == "skimage"

    class Restoration:
        @staticmethod
        def rolling_ball(values, **_kwargs):
            events.append("rolling-ball")
            return values.copy()

    class Measure:
        @staticmethod
        def regionprops_table(_labels, **_kwargs):
            events.append("regionprops")
            return {
                "label": np.asarray([1, 2]),
                "num_pixels": np.asarray([4, 2]),
                "bbox-0": np.asarray([1, 2]),
                "bbox-1": np.asarray([1, 4]),
                "bbox-2": np.asarray([3, 4]),
                "bbox-3": np.asarray([3, 5]),
                "centroid-0": np.asarray([1.5, 2.5]),
                "centroid-1": np.asarray([1.5, 4.0]),
            }

    class MeasurementKernels:
        @staticmethod
        def regionprops_euler(_labels, **_kwargs):
            events.append("euler")
            return np.asarray([1, 1])

    modules = {
        "cupy": Cupy(),
        "cupyx.scipy.ndimage": Ndimage(),
        "cupyx.scipy.signal": Signal(),
        "cucim": Cucim(),
        "cucim.skimage.restoration": Restoration(),
        "cucim.skimage.measure": Measure(),
        "cucim.skimage.measure._regionprops_gpu_misc_kernels": (
            MeasurementKernels()
        ),
    }
    monkeypatch.setattr(
        registry_module.importlib,
        "import_module",
        modules.__getitem__,
    )
    _mock_installed_gpu_distributions(monkeypatch)
    record_path = _cucim_environment_record(
        tmp_path / "gpu-environment.json",
        digest="A" * 64,
    )

    cupyx = registry_module._probe_cupyx_library()
    cucim = registry_module._probe_cucim_skimage_library(record_path=record_path)

    assert cupyx.available
    assert cucim.available
    assert dict(cucim.metadata) == {
        "environment_record_schema": "napari-vipp-gpu-environment",
        "environment_record_schema_version": "2",
        "environment_track": "cuda13",
        "cupy_distribution": "cupy-cuda13x",
        "cucim_distribution": "cucim-cu13",
        "cucim_distribution_version": "26.6.0",
        "cucim_artifact_sha256": "a" * 64,
        "cucim_wheel_payload_sha256": PHASE1_CUCIM_WHEEL_PAYLOAD_SHA256,
        "cucim_source_tag": PHASE1_CUCIM_SOURCE_TAG,
        "cucim_source_commit": PHASE1_CUCIM_SOURCE_COMMIT,
        "cucim_build_recipe_id": PHASE1_CUCIM_BUILD_RECIPE_ID,
    }
    assert events.count("pool-created") == 3
    assert events.count("allocator-enter") == 3
    assert events.count("allocator-exit") == 3
    assert events.count("free") == 3
    assert lease_calls == [
        ("cuda-cupy", "cuda:0"),
        ("cuda-cupy", "cuda:0"),
    ]
    assert "gaussian" in events
    assert "median" in events
    assert "label" in events
    assert "convolve" in events
    assert "rolling-ball" in events
    assert "regionprops" in events
    assert "euler" in events


@pytest.mark.parametrize("library_id", ("cupyx", "cucim"))
def test_builtin_gpu_library_probe_waits_for_process_device_lease(
    library_id,
    monkeypatch,
    tmp_path,
):
    manager = AcceleratorLeaseManager()
    lease_attempted = threading.Event()
    pool_created = threading.Event()
    holder_entered = threading.Event()
    release_holder = threading.Event()
    failures: list[BaseException] = []
    results: list[ImplementationLibraryProbeResult] = []

    @contextmanager
    def observed_lease(runtime_id: str, device_id: str):
        assert (runtime_id, device_id) == ("cuda-cupy", "cuda:0")
        lease_attempted.set()
        with manager.acquire(runtime_id, device_id):
            yield

    monkeypatch.setattr(registry_module, "accelerator_lease", observed_lease)

    class Pool:
        def __init__(self):
            pool_created.set()

        @staticmethod
        def malloc(_size):
            return object()

        @staticmethod
        def used_bytes():
            return 0

        @staticmethod
        def total_bytes():
            return 0

        @staticmethod
        def free_all_blocks():
            return None

    class Stream:
        @staticmethod
        def synchronize():
            return None

    class Runtime:
        @staticmethod
        def getDevice():
            return 0

    class Cuda:
        runtime = Runtime()
        MemoryPool = Pool

        @staticmethod
        @contextmanager
        def using_allocator(_allocator):
            yield

        @staticmethod
        def get_current_stream():
            return Stream()

    class Cupy:
        __version__ = "test-cupy"
        float32 = np.float32
        cuda = Cuda()

        @staticmethod
        def arange(*args, **kwargs):
            return np.arange(*args, **kwargs)

        @staticmethod
        def asarray(*args, **kwargs):
            return np.asarray(*args, **kwargs)

    class Ndimage:
        @staticmethod
        def gaussian_filter(values, **_kwargs):
            return values.copy()

        @staticmethod
        def median_filter(values, **_kwargs):
            return values.copy()

        @staticmethod
        def label(_values, **_kwargs):
            return np.asarray([[1, 0], [0, 2]], dtype=np.int32), 2

    class Signal:
        @staticmethod
        def convolve(values, _kernel, **_kwargs):
            return values.copy()

    class Cucim:
        __version__ = "test-cucim"

        @staticmethod
        def is_available(component):
            return component == "skimage"

    class Restoration:
        @staticmethod
        def rolling_ball(values, **_kwargs):
            return values.copy()

    class Measure:
        @staticmethod
        def regionprops_table(_labels, **_kwargs):
            return {
                "label": np.asarray([1, 2]),
                "num_pixels": np.asarray([4, 2]),
                "bbox-0": np.asarray([1, 2]),
                "bbox-1": np.asarray([1, 4]),
                "bbox-2": np.asarray([3, 4]),
                "bbox-3": np.asarray([3, 5]),
                "centroid-0": np.asarray([1.5, 2.5]),
                "centroid-1": np.asarray([1.5, 4.0]),
            }

    class MeasurementKernels:
        @staticmethod
        def regionprops_euler(_labels, **_kwargs):
            return np.asarray([1, 1])

    modules = {
        "cupy": Cupy(),
        "cupyx.scipy.ndimage": Ndimage(),
        "cupyx.scipy.signal": Signal(),
        "cucim": Cucim(),
        "cucim.skimage.restoration": Restoration(),
        "cucim.skimage.measure": Measure(),
        "cucim.skimage.measure._regionprops_gpu_misc_kernels": (
            MeasurementKernels()
        ),
    }
    monkeypatch.setattr(
        registry_module.importlib,
        "import_module",
        modules.__getitem__,
    )
    _mock_installed_gpu_distributions(monkeypatch)
    record_path = _cucim_environment_record(
        tmp_path / "gpu-environment.json",
        digest="A" * 64,
    )

    def holder() -> None:
        with manager.acquire("cuda-cupy", "cuda:0"):
            holder_entered.set()
            assert release_holder.wait(timeout=5)

    def run_probe() -> None:
        try:
            if library_id == "cupyx":
                result = registry_module._probe_cupyx_library()
            else:
                result = registry_module._probe_cucim_skimage_library(
                    record_path=record_path
                )
            results.append(result)
        except BaseException as exc:
            failures.append(exc)

    holder_thread = threading.Thread(target=holder)
    probe_thread = threading.Thread(target=run_probe)
    holder_thread.start()
    assert holder_entered.wait(timeout=5)
    probe_thread.start()
    assert lease_attempted.wait(timeout=5)
    assert not pool_created.is_set()
    release_holder.set()
    holder_thread.join(timeout=5)
    probe_thread.join(timeout=5)

    assert not holder_thread.is_alive()
    assert not probe_thread.is_alive()
    assert failures == []
    assert len(results) == 1 and results[0].available
    assert pool_created.is_set()


@pytest.mark.parametrize(
    ("document", "reason_code"),
    [
        ("{", "cucim_provenance_invalid"),
        (
            '{"schema":"napari-vipp-gpu-environment",'
            '"schema":"napari-vipp-gpu-environment",'
            '"schema_version":2,"track":"cuda13",'
            '"cupy_distribution":"cupy-cuda13x","cucim":null}',
            "cucim_provenance_invalid",
        ),
        (
            json.dumps(
                {
                    "schema": "napari-vipp-gpu-environment",
                    "schema_version": 1,
                    "track": "cuda13",
                    "cupy_distribution": "cupy-cuda13x",
                    "cucim": None,
                }
            ),
            "cucim_provenance_invalid",
        ),
        (
            json.dumps(
                {
                    "schema": "napari-vipp-gpu-environment",
                    "schema_version": 2.0,
                    "track": "cuda13",
                    "cupy_distribution": "cupy-cuda13x",
                    "cucim": None,
                }
            ),
            "cucim_provenance_invalid",
        ),
        (
            json.dumps(
                {
                    "schema": "napari-vipp-gpu-environment",
                    "schema_version": 2,
                    "track": "cuda12",
                    "cupy_distribution": "cupy-cuda12x",
                    "cucim": {
                        "distribution": "cucim-cu13",
                        "wheel_sha256": "a" * 64,
                        "wheel_payload_sha256": (
                            PHASE1_CUCIM_WHEEL_PAYLOAD_SHA256
                        ),
                        "source_tag": PHASE1_CUCIM_SOURCE_TAG,
                        "source_commit": PHASE1_CUCIM_SOURCE_COMMIT,
                        "build_recipe_id": PHASE1_CUCIM_BUILD_RECIPE_ID,
                    },
                }
            ),
            "cucim_provenance_invalid",
        ),
        (
            json.dumps(
                {
                    "schema": "napari-vipp-gpu-environment",
                    "schema_version": 2,
                    "track": "cuda13",
                    "cupy_distribution": "cupy-cuda12x",
                    "cucim": None,
                }
            ),
            "cucim_provenance_invalid",
        ),
        (
            json.dumps(
                {
                    "schema": "napari-vipp-gpu-environment",
                    "schema_version": 2,
                    "track": "cuda13",
                    "cupy_distribution": "cupy-cuda13x",
                    "cucim": {
                        "distribution": "cucim-cu13",
                        "wheel_sha256": "not-a-sha256",
                        "wheel_payload_sha256": (
                            PHASE1_CUCIM_WHEEL_PAYLOAD_SHA256
                        ),
                        "source_tag": PHASE1_CUCIM_SOURCE_TAG,
                        "source_commit": PHASE1_CUCIM_SOURCE_COMMIT,
                        "build_recipe_id": PHASE1_CUCIM_BUILD_RECIPE_ID,
                    },
                }
            ),
            "cucim_provenance_invalid",
        ),
        (
            json.dumps(
                {
                    "schema": "napari-vipp-gpu-environment",
                    "schema_version": 2,
                    "track": "cuda13",
                    "cupy_distribution": "cupy-cuda13x",
                    "cucim": None,
                    "unexpected": True,
                }
            ),
            "cucim_provenance_invalid",
        ),
        (
            json.dumps(
                {
                    "schema": "napari-vipp-gpu-environment",
                    "schema_version": 2,
                    "track": "cuda13",
                    "cupy_distribution": "cupy-cuda13x",
                    "cucim": None,
                }
            ),
            "cucim_provenance_unverified",
        ),
    ],
)
def test_cucim_probe_fails_closed_for_untrusted_records(
    tmp_path,
    monkeypatch,
    document,
    reason_code,
):
    path = tmp_path / "gpu-environment.json"
    path.write_text(document, encoding="utf-8")

    def unexpected_import(name):
        raise AssertionError(f"untrusted provenance imported {name}")

    monkeypatch.setattr(
        registry_module.importlib,
        "import_module",
        unexpected_import,
    )

    result = registry_module._probe_cucim_skimage_library(record_path=path)

    assert not result.available
    assert result.reason_code == reason_code
    assert "setup_gpu_dev.py" in result.message


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("payload_digest", "f" * 64),
        ("source_tag", "v0.0.0"),
        ("source_commit", "f" * 40),
        ("build_recipe_id", "other-recipe-v1"),
    ),
)
def test_cucim_probe_rejects_unapproved_source_or_payload_provenance(
    tmp_path,
    monkeypatch,
    field,
    value,
):
    path = _cucim_environment_record(
        tmp_path / "gpu-environment.json",
        **{field: value},
    )
    monkeypatch.setattr(
        registry_module.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(
            AssertionError(f"unapproved provenance imported {name}")
        ),
    )

    result = registry_module._probe_cucim_skimage_library(record_path=path)

    assert not result.available
    assert result.reason_code == "cucim_provenance_invalid"


def test_cucim_probe_missing_and_stale_records_are_actionable(
    tmp_path,
    monkeypatch,
):
    missing = registry_module._probe_cucim_skimage_library(
        record_path=tmp_path / "missing.json"
    )
    assert missing.reason_code == "cucim_provenance_missing"

    path = _cucim_environment_record(tmp_path / "gpu-environment.json")

    monkeypatch.setattr(
        registry_module.importlib.metadata,
        "distributions",
        lambda: iter([_FakeDistribution("cupy-cuda13x")]),
    )
    monkeypatch.setattr(
        registry_module.importlib.metadata,
        "distribution",
        lambda name: (_ for _ in ()).throw(
            registry_module.importlib.metadata.PackageNotFoundError(name)
        ),
    )
    monkeypatch.setattr(
        registry_module.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(
            AssertionError(f"stale provenance imported {name}")
        ),
    )

    stale = registry_module._probe_cucim_skimage_library(record_path=path)

    assert stale.reason_code == "cucim_provenance_stale"
    assert "not installed" in stale.message


def test_cucim_probe_rejects_same_version_reinstalled_from_another_wheel(
    tmp_path,
    monkeypatch,
):
    path = _cucim_environment_record(
        tmp_path / "gpu-environment.json",
        digest="a" * 64,
    )
    _mock_installed_gpu_distributions(monkeypatch, cucim_digest="b" * 64)
    monkeypatch.setattr(
        registry_module.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(
            AssertionError(f"mismatched provenance imported {name}")
        ),
    )

    result = registry_module._probe_cucim_skimage_library(record_path=path)

    assert not result.available
    assert result.reason_code == "cucim_artifact_mismatch"
    assert "expected " + "a" * 64 in result.message
    assert "found " + "b" * 64 in result.message


def test_cucim_probe_rejects_changed_installed_payload(
    tmp_path,
    monkeypatch,
):
    path = _cucim_environment_record(tmp_path / "gpu-environment.json")
    _mock_installed_gpu_distributions(
        monkeypatch,
        cucim_payload_digest="f" * 64,
    )
    monkeypatch.setattr(
        registry_module.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(
            AssertionError(f"mismatched payload imported {name}")
        ),
    )

    result = registry_module._probe_cucim_skimage_library(record_path=path)

    assert not result.available
    assert result.reason_code == "cucim_payload_mismatch"
    assert "expected " + PHASE1_CUCIM_WHEEL_PAYLOAD_SHA256 in result.message
    assert "found " + "f" * 64 in result.message


@pytest.mark.parametrize("digest", ("a" * 64, "b" * 64))
def test_cucim_provenance_accepts_each_matching_local_wheel_hash(
    tmp_path,
    monkeypatch,
    digest,
):
    path = _cucim_environment_record(
        tmp_path / "gpu-environment.json",
        digest=digest,
    )
    _mock_installed_gpu_distributions(monkeypatch, cucim_digest=digest)

    provenance = registry_module._read_cucim_environment_provenance(path)

    assert provenance is not None
    assert provenance.wheel_sha256 == digest
    assert registry_module._verify_installed_cucim_provenance(provenance) == "26.6.0"


@pytest.mark.parametrize(
    ("dependency", "version", "expected_text"),
    [
        ("click", "", "missing exact runtime dependency click==8.4.2"),
        ("lazy-loader", "0.6", "lazy-loader==0.5; found 0.6"),
        (
            "nvidia-nvimgcodec-cu13",
            "0.9.0",
            "nvidia-nvimgcodec-cu13==0.8.0.22; found 0.9.0",
        ),
    ],
)
def test_cucim_provenance_rejects_changed_exact_runtime_dependency(
    tmp_path,
    monkeypatch,
    dependency,
    version,
    expected_text,
):
    path = _cucim_environment_record(tmp_path / "gpu-environment.json")
    _mock_installed_gpu_distributions(
        monkeypatch,
        cucim_runtime_versions={dependency: version},
    )
    provenance = registry_module._read_cucim_environment_provenance(path)

    assert provenance is not None
    with pytest.raises(
        registry_module._InstalledProvenanceError,
        match=re.escape(expected_text),
    ) as error:
        registry_module._verify_installed_cucim_provenance(provenance)
    assert error.value.reason_code == "cucim_provenance_stale"


def test_installed_cucim_payload_digest_matches_canonical_wheel_stream(tmp_path):
    files = {
        "cucim/core.py": b"payload",
        "cucim_cu13-26.6.0.dist-info/METADATA": b"metadata",
        "cucim_cu13-26.6.0.dist-info/RECORD": b"pip may rewrite this",
        "cucim_cu13-26.6.0.dist-info/direct_url.json": b"pip-added",
        "cucim_cu13-26.6.0.dist-info/INSTALLER": b"pip",
        "cucim_cu13-26.6.0.dist-info/REQUESTED": b"",
        "cucim/__pycache__/core.cpython-312.pyc": b"generated",
    }
    for name, content in files.items():
        path = tmp_path.joinpath(*name.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    payload_paths = tuple(PurePosixPath(name) for name in files)

    class Distribution:
        files = payload_paths

        @staticmethod
        def locate_file(path):
            return tmp_path.joinpath(*str(path).split("/"))

    expected = hashlib.sha256()
    for name in sorted(
        ("cucim/core.py", "cucim_cu13-26.6.0.dist-info/METADATA"),
        key=lambda value: value.encode("utf-8"),
    ):
        name_bytes = name.encode("utf-8")
        content = files[name]
        expected.update(struct.pack(">Q", len(name_bytes)))
        expected.update(name_bytes)
        expected.update(struct.pack(">Q", len(content)))
        expected.update(content)

    observed = registry_module._installed_cucim_wheel_payload_sha256(Distribution())

    assert observed == expected.hexdigest()
    (tmp_path / "cucim_cu13-26.6.0.dist-info" / "RECORD").write_bytes(b"changed")
    assert (
        registry_module._installed_cucim_wheel_payload_sha256(Distribution())
        == observed
    )
    (tmp_path / "cucim" / "core.py").write_bytes(b"changed payload")
    assert (
        registry_module._installed_cucim_wheel_payload_sha256(Distribution())
        != observed
    )


@pytest.mark.parametrize("name", ("../escape.py", "missing.py", "directory"))
def test_installed_cucim_payload_digest_rejects_unsafe_or_nonfiles(tmp_path, name):
    if name == "directory":
        (tmp_path / name).mkdir()

    class Distribution:
        files = (PurePosixPath(name),)

        @staticmethod
        def locate_file(path):
            return tmp_path.joinpath(*str(path).split("/"))

    with pytest.raises(ValueError):
        registry_module._installed_cucim_wheel_payload_sha256(Distribution())


@pytest.mark.parametrize(
    ("cupy_names", "direct_url", "reason_code"),
    [
        (("cupy-cuda12x",), None, "cucim_provenance_stale"),
        (("cupy-cuda13x", "cupy"), None, "cucim_provenance_stale"),
        (("cupy-cuda13x",), "", "cucim_artifact_unverified"),
        (("cupy-cuda13x",), "{", "cucim_artifact_unverified"),
    ],
)
def test_cucim_probe_fails_closed_for_changed_cupy_or_missing_archive_provenance(
    tmp_path,
    monkeypatch,
    cupy_names,
    direct_url,
    reason_code,
):
    path = _cucim_environment_record(tmp_path / "gpu-environment.json")
    _mock_installed_gpu_distributions(
        monkeypatch,
        cupy_names=cupy_names,
        cucim_direct_url=direct_url,
    )
    monkeypatch.setattr(
        registry_module.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(
            AssertionError(f"unverified provenance imported {name}")
        ),
    )

    result = registry_module._probe_cucim_skimage_library(record_path=path)

    assert not result.available
    assert result.reason_code == reason_code


def test_private_probe_cleanup_releases_pool_after_sync_failure():
    events = []
    sync_error = RuntimeError("simulated synchronize failure")

    class Stream:
        @staticmethod
        def synchronize():
            events.append("sync")
            raise sync_error

    class Cuda:
        @staticmethod
        def get_current_stream():
            return Stream()

    class Pool:
        @staticmethod
        def free_all_blocks():
            events.append("free")

        @staticmethod
        def used_bytes():
            return 0

        @staticmethod
        def total_bytes():
            return 0

    with pytest.raises(RuntimeError, match="simulated synchronize") as exc_info:
        registry_module._drain_private_probe_pool(
            type("Cupy", (), {"cuda": Cuda()})(),
            Pool(),
            library_id="test",
        )

    assert exc_info.value is sync_error
    assert events == ["sync", "free"]


def test_library_probe_does_not_mask_operation_failure_with_cleanup_failure(
    monkeypatch,
):
    events = []

    class OperationFailure(RuntimeError):
        pass

    class Pool:
        @staticmethod
        def malloc(_size):
            raise AssertionError("NumPy fixture does not allocate through CuPy")

        @staticmethod
        def free_all_blocks():
            events.append("free")
            raise RuntimeError("simulated pool-release failure")

        @staticmethod
        def used_bytes():
            return 0

        @staticmethod
        def total_bytes():
            return 0

    class Stream:
        @staticmethod
        def synchronize():
            events.append("sync")

    class Cuda:
        MemoryPool = Pool

        @staticmethod
        @contextmanager
        def using_allocator(_allocator):
            yield

        @staticmethod
        def get_current_stream():
            return Stream()

    class Cupy:
        __version__ = "test-cupy"
        float32 = np.float32
        cuda = Cuda()

        @staticmethod
        def arange(*args, **kwargs):
            return np.arange(*args, **kwargs)

        @staticmethod
        def asarray(*args, **kwargs):
            return np.asarray(*args, **kwargs)

    class Ndimage:
        @staticmethod
        def gaussian_filter(_values, **_kwargs):
            raise OperationFailure("simulated Gaussian failure")

        @staticmethod
        def median_filter(values, **_kwargs):
            return values

        @staticmethod
        def label(_values, **_kwargs):
            return np.asarray([[1, 0], [0, 2]], dtype=np.int32), 2

    class Signal:
        @staticmethod
        def convolve(values, _kernel, **_kwargs):
            return values

    modules = {
        "cupy": Cupy(),
        "cupyx.scipy.ndimage": Ndimage(),
        "cupyx.scipy.signal": Signal(),
    }
    monkeypatch.setattr(
        registry_module.importlib,
        "import_module",
        modules.__getitem__,
    )

    with pytest.raises(OperationFailure, match="simulated Gaussian"):
        registry_module._probe_cupyx_library()

    assert events == ["sync", "free"]


def test_gpu_environment_record_path_is_relative_to_injected_prefix(tmp_path):
    assert registry_module._gpu_environment_record_path(prefix=tmp_path) == (
        tmp_path / "share" / "napari-vipp" / "gpu-environment.json"
    )


def test_real_cupyx_probe_preserves_external_default_pool_allocation():
    cupy = pytest.importorskip("cupy")
    try:
        if int(cupy.cuda.runtime.getDeviceCount()) < 1:
            pytest.skip("No CUDA device is available.")
        sentinel = cupy.full(1024, 17, dtype=cupy.float32)
        cupy.cuda.get_current_stream().synchronize()
    except Exception as exc:
        pytest.skip(f"A working CUDA device is unavailable: {exc}")

    pool = cupy.get_default_memory_pool()
    before = (int(pool.used_bytes()), int(pool.total_bytes()))
    try:
        try:
            result = registry_module._probe_cupyx_library()
        except Exception as exc:
            pytest.skip(f"A working CuPyX environment is unavailable: {exc}")
        after = (int(pool.used_bytes()), int(pool.total_bytes()))

        assert result.available
        assert after == before
        assert float(sentinel.sum().item()) == pytest.approx(17 * 1024)
    finally:
        sentinel = None
        pool.free_all_blocks()


def test_real_cucim_probe_preserves_external_default_pool_allocation(tmp_path):
    cupy = pytest.importorskip("cupy")
    pytest.importorskip("cucim")
    try:
        if int(cupy.cuda.runtime.getDeviceCount()) < 1:
            pytest.skip("No CUDA device is available.")
        sentinel = cupy.full(1024, 19, dtype=cupy.float32)
        cupy.cuda.get_current_stream().synchronize()
    except Exception as exc:
        pytest.skip(f"A working CUDA device is unavailable: {exc}")

    try:
        distribution = registry_module.importlib.metadata.distribution("cucim-cu13")
        direct_url = distribution.read_text("direct_url.json")
        if not direct_url:
            pytest.skip("Installed cuCIM has no PEP 610 wheel provenance.")
        digest = registry_module._pep610_archive_sha256(direct_url)
    except Exception as exc:
        pytest.skip(f"Installed cuCIM provenance is unavailable: {exc}")
    record_path = _cucim_environment_record(
        tmp_path / "gpu-environment.json",
        digest=digest,
    )
    pool = cupy.get_default_memory_pool()
    before = (int(pool.used_bytes()), int(pool.total_bytes()))
    try:
        result = registry_module._probe_cucim_skimage_library(record_path=record_path)
        after = (int(pool.used_bytes()), int(pool.total_bytes()))

        if not result.available and result.reason_code.startswith(
            ("cucim_provenance_", "cucim_artifact_", "cucim_payload_")
        ):
            pytest.skip(result.message)
        assert result.available
        assert after == before
        assert float(sentinel.sum().item()) == pytest.approx(19 * 1024)
    finally:
        sentinel = None
        pool.free_all_blocks()
