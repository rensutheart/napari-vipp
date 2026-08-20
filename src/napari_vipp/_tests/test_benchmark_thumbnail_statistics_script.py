from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "benchmark_thumbnail_statistics.py"


@pytest.fixture(scope="module")
def benchmark_script():
    module_name = "_vipp_test_benchmark_thumbnail_statistics"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(module_name, None)


def test_help_is_cupy_safe_in_a_fresh_process() -> None:
    code = "\n".join(
        (
            "import builtins, runpy, sys",
            "real_import = builtins.__import__",
            "def guarded_import(name, *args, **kwargs):",
            "    if name == 'cupy' or name.startswith('cupy.'):",
            "        raise RuntimeError('help imported CuPy')",
            "    return real_import(name, *args, **kwargs)",
            "builtins.__import__ = guarded_import",
            "sys.argv = [sys.argv[1], '--help']",
            "runpy.run_path(sys.argv[0], run_name='__main__')",
        )
    )
    completed = subprocess.run(
        [sys.executable, "-c", code, str(SCRIPT_PATH)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--sizes-mib" in completed.stdout
    assert "--nd2" in completed.stdout
    assert "--output" in completed.stdout


@pytest.mark.parametrize(
    ("dtype", "expected_shape"),
    (
        ("uint8", (3, 1024, 1024)),
        ("uint16", (3, 512, 1024)),
        ("float32", (3, 256, 1024)),
    ),
)
def test_synthetic_stack_is_exact_sized_deterministic_and_read_only(
    benchmark_script,
    dtype,
    expected_shape,
) -> None:
    first = benchmark_script._synthetic_stack(dtype, 3)
    second = benchmark_script._synthetic_stack(dtype, 3)

    assert first.shape == expected_shape
    assert first.dtype == np.dtype(dtype)
    assert first.nbytes == 3 * 1024**2
    assert first.flags.c_contiguous
    assert not first.flags.writeable
    np.testing.assert_array_equal(first, second)
    if np.issubdtype(first.dtype, np.integer):
        assert np.unique(first[:1]).size == np.iinfo(first.dtype).max + 1
    else:
        bits = first.reshape(-1).view(np.uint32)
        assert set(bits[:8]) == {
            0x00000000,
            0x80000000,
            0x00000001,
            0x80000001,
            0x00800000,
            0x7F800000,
            0xFF800000,
            0x7FC00001,
        }


def test_float32_is_explicitly_supported_but_not_in_the_default_matrix(
    benchmark_script,
) -> None:
    assert benchmark_script.DEFAULT_DTYPES == ("uint8", "uint16")
    assert benchmark_script._parse_dtypes("float32,uint8") == (
        "float32",
        "uint8",
    )
    with pytest.raises(ValueError, match="float32"):
        benchmark_script._parse_dtypes("float64")


def test_production_engine_timing_separates_cold_and_warm_calls(
    benchmark_script,
    monkeypatch,
) -> None:
    class Backend:
        CPU_NUMPY = SimpleNamespace(value="cpu-numpy")
        GPU_CUPY = SimpleNamespace(value="gpu-cupy")

    class ComputeMode:
        CPU = object()
        PREFER_GPU = object()
        AUTO = object()

    class Request:
        def __init__(self, **values):
            self.__dict__.update(values)

    class Engine:
        instances = []

        def __init__(self):
            self.gpu_calls = 0
            self.__class__.instances.append(self)

        def calculate(self, request):
            if request.compute_mode is ComputeMode.CPU:
                return SimpleNamespace(
                    actual_backend=Backend.CPU_NUMPY,
                    limits=(3.0, 900.0),
                    elapsed_seconds=0.2,
                    algorithm_id="cpu-exact",
                )
            self.gpu_calls += 1
            return SimpleNamespace(
                actual_backend=Backend.GPU_CUPY,
                limits=(3.0, 900.0),
                elapsed_seconds=0.5 if self.gpu_calls == 1 else 0.05,
                runtime_id="cuda-cupy",
                device_id="cuda:0",
                algorithm_id="gpu-exact",
                input_path="host_upload",
                logical_input_host_to_device_bytes=2048,
                auxiliary_host_to_device_bytes=0,
                device_to_host_bytes=65_536 * 8,
                device_to_host_values=65_536,
            )

        def select(self, request):
            del request
            return SimpleNamespace(
                backend=(Backend.GPU_CUPY if self.gpu_calls else Backend.CPU_NUMPY),
                reason_code=("auto_gpu_threshold_met" if self.gpu_calls else "below"),
                threshold_bytes=32 * 1024**2,
                gpu_warm=bool(self.gpu_calls),
            )

    monkeypatch.setattr(
        benchmark_script,
        "_production_api",
        lambda: (Engine, Request, Backend, ComputeMode),
    )
    result = benchmark_script._benchmark_array(
        np.arange(1024, dtype=np.uint16),
        case_id="synthetic-test",
        source={"source_kind": "deterministic-synthetic-stack"},
        cpu_rounds=3,
        warm_gpu_rounds=2,
        device_id="cuda:0",
    )

    assert len(Engine.instances) == 2
    assert result["cpu"]["samples_seconds"] == [0.2, 0.2, 0.2]
    assert result["gpu"]["cold_seconds"] == 0.5
    assert result["gpu"]["warm_samples_seconds"] == [0.05, 0.05]
    assert result["gpu"]["algorithm_id"] == "gpu-exact"
    expected_transfer = {
        "input_path": "host_upload",
        "logical_input_host_to_device_bytes": 2048,
        "auxiliary_host_to_device_bytes": 0,
        "device_to_host_bytes": 65_536 * 8,
        "device_to_host_values": 65_536,
    }
    assert result["gpu"]["cold_transfer"] == expected_transfer
    assert result["gpu"]["warm_transfers"] == [
        expected_transfer,
        expected_transfer,
    ]
    assert result["exact_parity"] is True
    assert result["speedup"]["cpu_over_cold_gpu"] == pytest.approx(0.4)
    assert result["speedup"]["cpu_over_warm_gpu"] == pytest.approx(4.0)
    assert result["production_auto_policy"] == {
        "before_gpu_evidence": {
            "backend": "cpu-numpy",
            "reason_code": "below",
            "threshold_bytes": 32 * 1024**2,
            "gpu_warm": False,
        },
        "after_this_engine_attempt": {
            "backend": "gpu-cupy",
            "reason_code": "auto_gpu_threshold_met",
            "threshold_bytes": 32 * 1024**2,
            "gpu_warm": True,
        },
    }
    assert "limits" not in result


def test_observed_crossover_requires_gpu_to_remain_faster(
    benchmark_script,
) -> None:
    cases = [
        _fake_result(2, cpu=1.0, cold=2.0, warm=0.5),
        _fake_result(4, cpu=1.0, cold=0.8, warm=0.8),
        _fake_result(8, cpu=1.0, cold=1.2, warm=0.6),
        _fake_result(16, cpu=1.0, cold=0.7, warm=0.5),
    ]

    observed = benchmark_script._observed_crossovers(cases)["uint8"]

    assert observed["cold_sustained_gpu_no_slower_from_bytes"] == 16 * 1024**2
    assert observed["warm_sustained_gpu_no_slower_from_bytes"] == 2 * 1024**2


def test_optional_nd2_path_and_filename_never_reach_document(
    benchmark_script,
    monkeypatch,
    tmp_path,
) -> None:
    private = tmp_path / "patient-identity-private.nd2"

    def fake_worker(spec, **kwargs):
        del kwargs
        if spec.source_kind == "synthetic":
            return _fake_result(spec.size_mib, dtype=spec.dtype)
        result = _fake_result(9)
        result["case_id"] = spec.case_id
        result["source"] = {
            "source_kind": "private-nd2-channel-stack",
            "direct_identifiers_omitted": True,
        }
        return result

    monkeypatch.setattr(benchmark_script, "_invoke_worker", fake_worker)
    monkeypatch.setattr(benchmark_script, "_source_provenance", lambda: {})
    document = benchmark_script.run_benchmark(
        sizes_mib=(2,),
        dtypes=("uint8",),
        cpu_rounds=1,
        warm_gpu_rounds=1,
        nd2_path=private,
    )
    serialized = json.dumps(document)

    assert str(private) not in serialized
    assert private.name not in serialized
    assert document["matrix"]["private_nd2_included"] is True
    assert document["results"][-1]["case_id"] == "private-nd2-channel-stack"
    benchmark_script._validate_document(document)

    tampered = deepcopy(document)
    tampered["results"][0]["exact_parity"] = False
    with pytest.raises(benchmark_script.BenchmarkError, match="exact CPU parity"):
        benchmark_script._validate_document(tampered)


def test_private_nd2_is_channel_sliced_before_compute(
    benchmark_script,
    monkeypatch,
    tmp_path,
) -> None:
    source_path = tmp_path / "private.nd2"
    source_path.write_bytes(b"test placeholder")
    selections = []

    class FakeSelected:
        def compute(self):
            return np.zeros((3, 4, 5, 6), dtype=np.uint16)

    class FakeLazy:
        def __getitem__(self, selection):
            selections.append(selection)
            return FakeSelected()

    class FakeND2File:
        shape = (3, 2, 4, 5, 6)
        sizes = {"T": 3, "C": 2, "Z": 4, "Y": 5, "X": 6}

        def __init__(self, path):
            assert path == str(source_path.resolve())

        def __enter__(self):
            return self

        def __exit__(self, *args):
            del args

    fake_nd2 = SimpleNamespace(
        ND2File=FakeND2File,
        imread=lambda path, dask: FakeLazy(),
    )
    monkeypatch.setitem(sys.modules, "nd2", fake_nd2)

    data, metadata = benchmark_script._load_private_nd2(
        source_path,
        channel_index=1,
        time_index=None,
    )

    assert selections == [(slice(None), 1, slice(None), slice(None), slice(None))]
    assert data.shape == (3, 4, 5, 6)
    assert metadata["selected_axes"] == "TZYX"
    assert source_path.name not in json.dumps(metadata)


def test_atomic_json_is_strict_and_confined_to_selected_directory(
    benchmark_script,
    tmp_path,
) -> None:
    output = tmp_path / "nested" / "calibration.json"
    written = benchmark_script._atomic_write_json(output, {"finite": 1.25})

    assert written == output.resolve()
    assert json.loads(output.read_text(encoding="utf-8")) == {"finite": 1.25}
    assert tuple(output.parent.iterdir()) == (output,)
    with pytest.raises(benchmark_script.BenchmarkError, match="strict JSON"):
        benchmark_script._atomic_write_json(output, {"invalid": float("nan")})
    assert json.loads(output.read_text(encoding="utf-8")) == {"finite": 1.25}


def _fake_result(
    size_mib: int,
    *,
    dtype: str = "uint8",
    cpu: float = 1.0,
    cold: float = 0.5,
    warm: float = 0.25,
):
    return {
        "case_id": f"synthetic-{dtype}-{size_mib:04d}mib",
        "source": {"source_kind": "deterministic-synthetic-stack"},
        "shape": [size_mib, 1024, 1024],
        "dtype": dtype,
        "element_count": size_mib * 1024**2,
        "input_bytes": size_mib * 1024**2,
        "cpu": {
            "samples_seconds": [cpu],
            "median_seconds": cpu,
            "algorithm_id": "cpu-exact",
        },
        "gpu": {
            "status": "available",
            "reason_code": "",
            "fallback_elapsed_seconds": None,
            "cold_seconds": cold,
            "cold_transfer": {
                "input_path": "host_upload",
                "logical_input_host_to_device_bytes": size_mib * 1024**2,
                "auxiliary_host_to_device_bytes": 0,
                "device_to_host_bytes": 2048,
                "device_to_host_values": 256,
            },
            "warm_samples_seconds": [warm],
            "warm_transfers": [
                {
                    "input_path": "host_upload",
                    "logical_input_host_to_device_bytes": size_mib * 1024**2,
                    "auxiliary_host_to_device_bytes": 0,
                    "device_to_host_bytes": 2048,
                    "device_to_host_values": 256,
                }
            ],
            "warm_median_seconds": warm,
            "runtime_id": "cuda-cupy",
            "device_id": "cuda:0",
            "algorithm_id": "gpu-exact",
        },
        "production_auto_policy": {
            "before_gpu_evidence": {
                "backend": "cpu-numpy",
                "reason_code": "auto_below_cold_gpu_threshold",
                "threshold_bytes": 384 * 1024**2,
                "gpu_warm": False,
            },
            "after_this_engine_attempt": {
                "backend": "cpu-numpy",
                "reason_code": "auto_below_warm_gpu_threshold",
                "threshold_bytes": 32 * 1024**2,
                "gpu_warm": True,
            },
        },
        "exact_parity": True,
        "speedup": {
            "cpu_over_cold_gpu": cpu / cold,
            "cpu_over_warm_gpu": cpu / warm,
        },
        "contrast_limits_omitted": True,
    }
