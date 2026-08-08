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
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "benchmark_gpu_rl_performance.py"


@pytest.fixture(scope="module")
def performance_script():
    module_name = "_vipp_test_benchmark_gpu_rl_performance"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(module_name, None)


def test_source_provenance_uses_operation_owners_not_shared_registries(
    performance_script,
) -> None:
    paths = tuple(performance_script.SOURCE_PROVENANCE_PATHS)

    for owner in (
        "src/napari_vipp/core/richardson_lucy.py",
        "src/napari_vipp/core/richardson_lucy_compute.py",
        "src/napari_vipp/core/richardson_lucy_parity.py",
    ):
        assert owner in paths
    for shared in (
        "src/napari_vipp/core/operations.py",
        "src/napari_vipp/core/compute_specs.py",
        "src/napari_vipp/core/compute_policy.py",
        "src/napari_vipp/core/compute_benchmark_adapter.py",
    ):
        assert shared not in paths


def test_source_provenance_detects_each_owner_but_ignores_shared_registries(
    performance_script,
    monkeypatch,
    tmp_path: Path,
) -> None:
    tracked = tuple(performance_script.SOURCE_PROVENANCE_PATHS)
    unrelated = Path("src/napari_vipp/core/compute_specs.py")
    for relative_path in (*tracked, str(unrelated)):
        destination = tmp_path / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((PROJECT_ROOT / relative_path).read_bytes())
    monkeypatch.setattr(performance_script, "PROJECT_ROOT", tmp_path)
    baseline = performance_script._source_provenance()

    unrelated_path = tmp_path / unrelated
    unrelated_path.write_bytes(unrelated_path.read_bytes() + b"\n# unrelated edit\n")
    assert performance_script._source_provenance() == baseline

    for relative_path in tracked:
        owner = tmp_path / relative_path
        original = owner.read_bytes()
        owner.write_bytes(original + b"\n# owner edit\n")
        assert performance_script._source_provenance() != baseline
        owner.write_bytes(original)


def test_help_is_cpu_safe_in_a_fresh_process():
    command = [sys.executable, str(SCRIPT_PATH), "--help"]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0
    assert "--validate-existing" in completed.stdout
    assert "--nd2" in completed.stdout
    assert "CuPy" in completed.stdout


def test_small_generator_and_psf_are_deterministic(performance_script):
    definition = performance_script._CaseDefinition(
        case_id="test",
        label="test",
        shape=(5, 7, 9),
        psf_shape=(3, 5, 5),
        psf_sigma=(1.0, 1.5, 1.5),
        seed=123,
        source_kind="deterministic-synthetic",
        source_metadata={},
    )

    first = performance_script._synthetic_image(definition)
    second = performance_script._synthetic_image(definition)
    psf = performance_script._gaussian_psf(
        definition.psf_shape,
        definition.psf_sigma,
    )

    assert first.shape == definition.shape
    assert first.dtype == np.float32
    assert first.flags.c_contiguous
    np.testing.assert_array_equal(first, second)
    assert psf.shape == definition.psf_shape
    assert psf.dtype == np.float32
    assert psf.flags.c_contiguous
    assert float(psf.sum(dtype=np.float64)) == pytest.approx(1.0, abs=1e-7)
    assert np.all(psf >= 0)


def test_resource_preflight_rejects_host_pressure(
    performance_script,
    monkeypatch,
):
    class FakeDevice:
        def __init__(self, index):
            self.index = index

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    fake_cupy = SimpleNamespace(
        cuda=SimpleNamespace(
            Device=FakeDevice,
            runtime=SimpleNamespace(memGetInfo=lambda: (16 << 30, 24 << 30)),
        )
    )
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    monkeypatch.setattr(
        performance_script,
        "_available_host_memory_bytes",
        lambda: 2 << 30,
    )
    image = np.zeros((4, 8, 8), dtype=np.float32)
    psf = np.ones((3, 3, 3), dtype=np.float32)

    with pytest.raises(
        performance_script.PerformanceBenchmarkError,
        match="Host-memory preflight",
    ):
        performance_script._resource_preflight(
            image=image,
            psf=psf,
            admitted_device_bytes=5 << 30,
            safety_reserve_bytes=1 << 30,
            device_id="cuda:0",
        )


def test_private_nd2_is_sliced_before_compute_and_redacted(
    performance_script,
    monkeypatch,
    tmp_path,
):
    selections = []

    class FakeND2File:
        shape = (2, 3, 2, 5, 7)
        sizes = {"T": 2, "Z": 3, "C": 2, "Y": 5, "X": 7}
        dtype = np.dtype(np.uint16)

        def __init__(self, _path):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class FakeSelected:
        def compute(self):
            return np.arange(3 * 5 * 7, dtype=np.uint16).reshape(3, 5, 7)

    class FakeLazy:
        def __getitem__(self, selection):
            selections.append(selection)
            return FakeSelected()

    fake_nd2 = SimpleNamespace(
        ND2File=FakeND2File,
        imread=lambda _path, dask: FakeLazy(),
    )
    monkeypatch.setitem(sys.modules, "nd2", fake_nd2)
    source = tmp_path / "private.nd2"
    source.touch()

    definition, volume = performance_script._load_private_nd2_volume(
        source,
        time_index=1,
        channel_index=1,
    )

    assert selections == [(1, slice(None), 1, slice(None), slice(None))]
    assert volume.shape == (3, 5, 7)
    assert volume.dtype == np.float32
    assert definition.psf_shape == (3, 5, 7)
    serialized = json.dumps(definition.source_metadata)
    assert str(source) not in serialized
    assert source.name not in serialized
    assert definition.private_source is True


def test_atomic_artifact_round_trip_and_currentness(
    performance_script,
    tmp_path,
):
    document = _example_document(performance_script)
    output = tmp_path / "evidence.json"
    markdown = tmp_path / "evidence.md"

    performance_script._atomic_write_artifacts(output, markdown, document)

    assert performance_script.validate_existing(output) == output.resolve()
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["schema"] == performance_script.SCHEMA
    assert markdown.read_text(encoding="utf-8") == (
        performance_script.render_markdown(loaded)
    )
    assert "GPU end-to-end" in markdown.read_text(encoding="utf-8")


def test_validate_existing_rejects_stale_source_hash(
    performance_script,
    tmp_path,
):
    document = _example_document(performance_script)
    document["source_provenance"][0]["sha256"] = "0" * 64
    output = tmp_path / "stale.json"
    output.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(
        performance_script.PerformanceBenchmarkError,
        match="source fingerprints",
    ):
        performance_script.validate_existing(output)


def test_contract_rejects_private_source_digest(performance_script):
    document = _example_document(performance_script)
    private = deepcopy(document["results"][0])
    private.update(
        {
            "case_id": "private-real-nd2-volume-3d",
            "source_kind": "private-nd2-volume",
            "source_metadata": {
                "original_axes": "TZCYX",
                "original_shape": [2, 3, 2, 5, 7],
                "original_dtype": "uint16",
                "selected_indices": {"T": 1, "C": 1},
                "direct_identifiers_omitted": True,
            },
            "input_sha256": "3" * 64,
            "workload_fingerprint": None,
            "benchmark_record_digest": None,
        }
    )
    document["results"].append(private)

    with pytest.raises(
        performance_script.PerformanceBenchmarkError,
        match="input_sha256",
    ):
        performance_script._validate_document_contract(document)


def test_checked_in_artifact_is_historical_and_internally_valid(
    performance_script,
):
    artifact = (
        PROJECT_ROOT
        / "docs"
        / "benchmarks"
        / "rl-cupy-performance-windows-rtx5090.json"
    )
    raw = artifact.read_text(encoding="utf-8")
    document = json.loads(raw)
    current_source_document = deepcopy(document)
    current_source_document["source_provenance"] = (
        performance_script._source_provenance()
    )

    with pytest.raises(
        performance_script.PerformanceBenchmarkError,
        match="source fingerprints",
    ):
        performance_script.validate_existing(artifact)
    performance_script._validate_document_contract(current_source_document)
    assert raw == json.dumps(document, indent=2, sort_keys=True) + "\n"
    assert artifact.with_suffix(".md").read_text(encoding="utf-8") == (
        performance_script.render_markdown(document)
    )


def _example_document(performance_script):
    cpu_times = [2.0, 2.1, 1.9]
    gpu_times = [0.5, 0.55, 0.45]
    resident_times = [0.4, 0.44, 0.36]
    transfer_times = [0.1, 0.11, 0.09]
    paired_speedups = [cpu / gpu for cpu, gpu in zip(cpu_times, gpu_times, strict=True)]
    terminal_snapshots = [
        {"runtime_live_bytes": 0, "runtime_reserved_bytes": 0} for _ in range(5)
    ]
    return {
        "schema": performance_script.SCHEMA,
        "schema_version": performance_script.SCHEMA_VERSION,
        "created_utc": "2026-07-29T00:00:00+00:00",
        "kind": "machine-local-production-path-screening-evidence",
        "portable_performance_claim": False,
        "profile": "medium",
        "method": {
            "operation_id": performance_script.OPERATION_ID,
            "implementation_id": performance_script.IMPLEMENTATION_ID,
            "iterations": 25,
            "filter_epsilon": 1e-8,
            "warmup_rounds": 1,
            "paired_warm_rounds": 3,
            "paired_bootstrap_samples": 0,
            "sampling_profile": "short-descriptive-3-paired-v1",
            "durable_optimizer_record": False,
            "gpu_timing_scope": "synchronized-end-to-end-v1",
            "disk_io_included": False,
            "input_generation_included": False,
            "exact_workload_parity_required_before_timing": True,
        },
        "platform": {
            "device_name": "Fake GPU",
            "processor": "Fake CPU",
        },
        "source_provenance": performance_script._source_provenance(),
        "results": [
            {
                "case_id": "synthetic-shape-stress-medium-3d",
                "label": "Medium 3D shape stress (synthetic)",
                "source_kind": "deterministic-synthetic",
                "source_metadata": {"generator": performance_script.GENERATOR_ID},
                "direct_private_identifiers_published": False,
                "shape": [4, 8, 8],
                "voxel_count": 256,
                "dtype": "float32",
                "image_bytes": 1024,
                "psf_shape": [3, 3, 3],
                "psf_sigma_voxels": [1.0, 1.0, 1.0],
                "psf_sha256": "2" * 64,
                "input_sha256": "1" * 64,
                "parameters": {
                    name: value
                    for name, value in performance_script._parameters().items()
                    if name != "progress"
                },
                "workload_fingerprint": "workload",
                "benchmark_record_digest": "record",
                "memory_estimate": {
                    "model_id": "cupyx-richardson-lucy-fft-memory-v2",
                    "total_device_peak_bytes": 100,
                    "uncertainty_bytes": 25,
                },
                "preflight": {"admitted_device_bytes": 125},
                "summary": {
                    "cpu_median_seconds": 2.0,
                    "gpu_end_to_end_median_seconds": 0.5,
                    "gpu_resident_median_seconds": 0.4,
                    "gpu_transfer_median_seconds": 0.1,
                    "paired_speedup_median": 4.0,
                    "paired_speedups": paired_speedups,
                    "screening_choice": "GPU-CuPy",
                },
                "candidates": [
                    {
                        "implementation_id": "cpu-richardson_lucy_deconvolution-v1",
                        "parity_passed": True,
                        "error": "",
                        "warm_seconds": cpu_times,
                    },
                    {
                        "implementation_id": performance_script.IMPLEMENTATION_ID,
                        "parity_passed": True,
                        "error": "",
                        "warm_seconds": gpu_times,
                        "warm_resident_seconds": resident_times,
                        "warm_transfer_seconds": transfer_times,
                        "timing_scope": "synchronized-end-to-end-v1",
                        "synchronized": True,
                        "transfers_included": True,
                    },
                ],
                "gpu_cleanup": {
                    "invocation_count": 5,
                    "all_cleanup_succeeded": True,
                    "all_runtime_pool_terminal_zero": True,
                    "terminal_snapshots": terminal_snapshots,
                },
            }
        ],
    }
