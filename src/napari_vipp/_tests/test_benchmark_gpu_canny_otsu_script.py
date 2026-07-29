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
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "benchmark_gpu_canny_otsu.py"


@pytest.fixture(scope="module")
def evidence_script():
    module_name = "_vipp_test_benchmark_gpu_canny_otsu"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(module_name, None)


def test_help_is_cuda_safe_in_a_fresh_process():
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
    assert "--validate-existing" in completed.stdout
    assert "--nd2" in completed.stdout
    assert "Canny" in completed.stdout
    assert "Otsu" in completed.stdout


def test_admission_manifest_is_deterministic_and_complete(evidence_script):
    first = evidence_script._admission_cases()
    second = evidence_script._admission_cases()
    assert [case.case_id for case in first] == [case.case_id for case in second]
    assert len({case.case_id for case in first}) == len(first)

    coverage = {operation: set() for operation in evidence_script.REQUIRED_COVERAGE}
    for first_case, second_case in zip(first, second, strict=True):
        np.testing.assert_array_equal(first_case.data, second_case.data)
        assert first_case.parameters == second_case.parameters
        coverage[first_case.operation_id].update(first_case.coverage)
    for operation, required in evidence_script.REQUIRED_COVERAGE.items():
        assert required <= coverage[operation]


def test_small_performance_generator_is_deterministic(
    evidence_script,
    monkeypatch,
):
    monkeypatch.setattr(evidence_script, "SYNTHETIC_SHAPE", (2, 64, 64))
    first = evidence_script._synthetic_performance_source()
    second = evidence_script._synthetic_performance_source()

    assert first.data.shape == (2, 64, 64)
    assert first.data.dtype == np.uint16
    assert first.data.flags.c_contiguous
    np.testing.assert_array_equal(first.data, second.data)
    assert np.unique(first.data).size > 100


def test_private_nd2_is_sliced_before_compute_and_redacted(
    evidence_script,
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
    source_path = tmp_path / "private.nd2"
    source_path.touch()

    source = evidence_script._load_private_nd2_volume(
        source_path,
        time_index=1,
        channel_index=1,
    )

    assert selections == [(1, slice(None), 1, slice(None), slice(None))]
    assert source.data.shape == (3, 5, 7)
    assert source.data.dtype == np.uint16
    assert source.private is True
    serialized = json.dumps(source.metadata)
    assert str(source_path) not in serialized
    assert source_path.name not in serialized
    assert source.metadata["selected_indices"] == {"T": 1, "C": 1}


def test_atomic_round_trip_and_cpu_safe_validation(
    evidence_script,
    tmp_path,
):
    document = _example_document(evidence_script)
    output = tmp_path / "evidence.json"
    markdown = tmp_path / "evidence.md"
    evidence_script._atomic_write_artifacts(output, markdown, document)

    assert evidence_script.validate_existing(output) == output.resolve()
    assert markdown.read_text(encoding="utf-8") == evidence_script.render_markdown(
        document
    )
    code = "\n".join(
        (
            "import builtins, runpy, sys",
            "real_import = builtins.__import__",
            "def guarded_import(name, *args, **kwargs):",
            "    if name == 'cupy' or name.startswith('cupy.'):",
            "        raise RuntimeError('validation imported CuPy')",
            "    return real_import(name, *args, **kwargs)",
            "builtins.__import__ = guarded_import",
            "sys.argv = [sys.argv[1], '--validate-existing', sys.argv[2]]",
            "runpy.run_path(sys.argv[0], run_name='__main__')",
        )
    )
    completed = subprocess.run(
        [sys.executable, "-c", code, str(SCRIPT_PATH), str(output)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "evidence is current" in completed.stdout


def test_validation_rejects_stale_source_and_private_markers(
    evidence_script,
    tmp_path,
):
    document = _example_document(evidence_script)
    stale = deepcopy(document)
    stale["source_provenance"][0]["sha256"] = "0" * 64
    output = tmp_path / "stale.json"
    output.write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(evidence_script.EvidenceError, match="fingerprints"):
        evidence_script.validate_existing(output)

    with pytest.raises(evidence_script.EvidenceError, match="private marker"):
        evidence_script._validate_privacy({"source": r"C:\Users\person\secret.nd2"})


def test_contract_rejects_mask_dtype_memory_and_lifecycle_claim_tampering(
    evidence_script,
):
    document = _example_document(evidence_script)

    wrong_dtype = deepcopy(document)
    wrong_dtype["admission"]["cases"][0]["gpu_output_dtype"] = "float32"
    with pytest.raises(evidence_script.EvidenceError, match="not exact"):
        evidence_script._validate_document_contract(wrong_dtype)

    insufficient_memory = deepcopy(document)
    memory = insufficient_memory["performance"]["sources"][0]["operations"][0]["memory"]
    memory["observed_private_pool_peak_bytes"] = (
        memory["admitted_device_peak_bytes"] + 1
    )
    with pytest.raises(evidence_script.EvidenceError, match="exceeds"):
        evidence_script._validate_document_contract(insufficient_memory)

    incomplete_cancel = deepcopy(document)
    incomplete_cancel["lifecycle"]["operations"][0]["reported_progress"][1][
        "current"
    ] = 0
    with pytest.raises(evidence_script.EvidenceError, match="progress"):
        evidence_script._validate_document_contract(incomplete_cancel)


def test_contract_rejects_unknown_fields_at_every_schema_layer(evidence_script):
    def nested(document, *keys):
        value = document
        for key in keys:
            value = value[key]
        return value

    targets = (
        (),
        ("method",),
        ("source_provenance", 0),
        ("admission",),
        ("admission", "cases", 0),
        ("performance",),
        ("performance", "sources", 0),
        ("performance", "sources", 0, "source_metadata"),
        ("performance", "sources", 0, "operations", 0),
        ("performance", "sources", 0, "operations", 0, "parity"),
        ("performance", "sources", 0, "operations", 0, "samples"),
        ("performance", "sources", 0, "operations", 0, "memory"),
        ("performance", "sources", 0, "operations", 0, "memory", "cleanup"),
        ("performance", "sources", 0, "operations", 0, "summary"),
        ("lifecycle",),
        ("lifecycle", "operations", 0),
        ("lifecycle", "operations", 0, "reported_progress", 0),
        ("lifecycle", "operations", 0, "cleanup"),
    )
    for index, path in enumerate(targets):
        document = _example_document(evidence_script)
        nested(document, *path)[f"private_field_{index}"] = "JOHN DOE"
        with pytest.raises(evidence_script.EvidenceError, match="privacy-safe"):
            evidence_script._validate_document_contract(document)


def test_contract_rejects_cross_field_integrity_tampering(evidence_script):
    mutations = []

    def add_mutation(name, mutate, message):
        mutations.append((name, mutate, message))

    add_mutation(
        "memory-model",
        lambda document: document["performance"]["sources"][0]["operations"][0][
            "memory"
        ].__setitem__("model_id", "fabricated-memory-proof"),
        "memory-model proof",
    )
    add_mutation(
        "memory-arithmetic",
        lambda document: document["performance"]["sources"][0]["operations"][0][
            "memory"
        ].__setitem__(
            "admitted_device_peak_bytes",
            document["performance"]["sources"][0]["operations"][0]["memory"][
                "admitted_device_peak_bytes"
            ]
            + 1,
        ),
        "policy admission",
    )
    add_mutation(
        "input-bytes",
        lambda document: document["performance"]["sources"][0].__setitem__(
            "input_bytes",
            document["performance"]["sources"][0]["input_bytes"] + 1,
        ),
        "input_bytes",
    )
    add_mutation(
        "parameters",
        lambda document: document["performance"]["sources"][0]["operations"][0][
            "parameters"
        ].__setitem__("sigma", 2.5),
        "parameters changed",
    )
    add_mutation(
        "parity-profile",
        lambda document: document["performance"]["sources"][0]["operations"][0][
            "parity"
        ].__setitem__("profile", "close-enough"),
        "exact parity",
    )
    add_mutation(
        "numeric-string",
        lambda document: document["performance"]["sources"][0]["operations"][0][
            "summary"
        ].__setitem__("cpu_median_seconds", "2.0"),
        "finite and positive",
    )
    add_mutation(
        "naive-time",
        lambda document: document.__setitem__(
            "created_utc",
            "2026-07-29T00:00:00",
        ),
        "timezone-aware UTC",
    )
    add_mutation(
        "duplicate-provenance",
        lambda document: document["source_provenance"].append(
            deepcopy(document["source_provenance"][0])
        ),
        "canonical source once",
    )
    add_mutation(
        "duplicate-lifecycle",
        lambda document: document["lifecycle"]["operations"].append(
            deepcopy(document["lifecycle"]["operations"][0])
        ),
        "Lifecycle evidence",
    )

    for _name, mutate, message in mutations:
        document = _example_document(evidence_script)
        mutate(document)
        with pytest.raises(evidence_script.EvidenceError, match=message):
            evidence_script._validate_document_contract(document)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("system", "Linux", "native Windows"),
        ("execution_mode", "wsl", "native execution"),
        ("python", "3.13.1", "Python 3.12"),
        ("python_implementation", "PyPy", "CPython"),
        ("python_abi", "cpython-313", "cpython-312"),
        ("cuda_device_index", True, "CUDA device index"),
        ("cuda_device_name", "unknown CUDA device", "exact reviewed device"),
        ("cuda_compute_capability", "twelve", "compute capability"),
        ("cuda_driver_version", 0, "driver version"),
        ("cuda_runtime_version", 14000, "runtime version"),
        ("total_accelerator_memory_bytes", 0, "memory must be positive"),
    ),
)
def test_environment_contract_rejects_host_and_device_tampering(
    evidence_script,
    field,
    value,
    message,
):
    document = _example_document(evidence_script)
    document["platform"][field] = value

    with pytest.raises(evidence_script.EvidenceError, match=message):
        evidence_script._validate_document_contract(document)


@pytest.mark.parametrize(
    ("distribution", "value", "message"),
    (
        ("numpy", "2.4.0", "numpy must be 2.5.1"),
        ("scipy", "1.17.0", "scipy must be 1.18.0"),
        ("scikit-image", "0.25.2", "scikit-image must be 0.26.0"),
        ("cupy-cuda13x", "14.0.0", "cupy-cuda13x 14.1.1"),
        ("napari-vipp", "not-installed", "napari-vipp package version"),
    ),
)
def test_environment_contract_rejects_package_tampering(
    evidence_script,
    distribution,
    value,
    message,
):
    document = _example_document(evidence_script)
    document["packages"][distribution] = value

    with pytest.raises(evidence_script.EvidenceError, match=message):
        evidence_script._validate_document_contract(document)


def test_environment_contract_rejects_schema_and_cuda_track_tampering(
    evidence_script,
):
    document = _example_document(evidence_script)

    missing_platform = deepcopy(document)
    missing_platform["platform"].pop("cuda_compute_capability")
    with pytest.raises(evidence_script.EvidenceError, match="platform/device metadata"):
        evidence_script._validate_document_contract(missing_platform)

    extra_package = deepcopy(document)
    extra_package["packages"]["private_path"] = "E:/private/source"
    with pytest.raises(evidence_script.EvidenceError, match="package metadata"):
        evidence_script._validate_document_contract(extra_package)

    dual_cupy = deepcopy(document)
    dual_cupy["packages"]["cupy-cuda12x"] = "14.1.1"
    with pytest.raises(evidence_script.EvidenceError, match="exactly one"):
        evidence_script._validate_document_contract(dual_cupy)

    mismatched_track = deepcopy(document)
    mismatched_track["platform"]["cuda_runtime_version"] = 12090
    with pytest.raises(evidence_script.EvidenceError, match="runtime version"):
        evidence_script._validate_document_contract(mismatched_track)


@pytest.mark.parametrize(
    "tamper",
    (
        "source_id",
        "label",
        "source_key",
        "metadata_key",
        "missing_metadata_key",
        "selected_index_key",
        "covert_axes",
        "duplicate_axes",
        "selected_index_range",
        "shape_mismatch",
        "position_not_singleton",
    ),
)
def test_private_source_contract_rejects_identifier_and_schema_tampering(
    evidence_script,
    tamper,
):
    document = _example_document_with_private_source(evidence_script)
    evidence_script._validate_document_contract(document)

    private = document["performance"]["sources"][-1]
    if tamper == "source_id":
        private["source_id"] = "private-gr535-control-sample"
    elif tamper == "label":
        private["label"] = "GR535 control acquisition"
    elif tamper == "source_key":
        private["filename"] = "private-image.tif"
    elif tamper == "metadata_key":
        private["source_metadata"]["file_path"] = "E:/research/private-image.tif"
    elif tamper == "missing_metadata_key":
        private["source_metadata"].pop("original_dtype")
    elif tamper == "selected_index_key":
        private["source_metadata"]["selected_indices"]["P"] = 0
    elif tamper == "covert_axes":
        private["source_metadata"]["original_axes"] = "SECRETCZYX"
        private["source_metadata"]["original_shape"] = [1] * 10
    elif tamper == "duplicate_axes":
        private["source_metadata"]["original_axes"] = "TZZCYX"
        private["source_metadata"]["original_shape"] = [2, 8, 1, 3, 1024, 1024]
    elif tamper == "selected_index_range":
        private["source_metadata"]["selected_indices"]["C"] = 3
    elif tamper == "shape_mismatch":
        private["source_metadata"]["original_shape"][1] = 7
    else:
        private["source_metadata"]["original_axes"] = "TPZCYX"
        private["source_metadata"]["original_shape"] = [2, 2, 8, 3, 1024, 1024]

    with pytest.raises(
        evidence_script.EvidenceError,
        match="privacy-safe|Performance source|Private source|Private selected",
    ):
        evidence_script._validate_document_contract(document)


def _example_document_with_private_source(evidence_script):
    document = _example_document(evidence_script)
    private = deepcopy(document["performance"]["sources"][0])
    private.update(
        {
            "source_id": evidence_script.PRIVATE_SOURCE_ID,
            "label": evidence_script.PRIVATE_SOURCE_LABEL,
            "source_kind": evidence_script.PRIVATE_SOURCE_KIND,
            "source_metadata": {
                "original_axes": "TZCYX",
                "original_shape": [2, 8, 3, 1024, 1024],
                "original_dtype": "uint16",
                "selected_indices": {"T": 1, "C": 1},
                "direct_identifiers_omitted": True,
            },
            "input_sha256": None,
        }
    )
    document["performance"]["sources"].append(private)
    document["performance"]["source_count"] = 2
    return document


def _example_document(evidence_script):
    cpu = [2.0, 2.1, 1.9]
    end_to_end = [0.5, 0.55, 0.45]
    resident = [0.4, 0.44, 0.36]
    shape = evidence_script.SYNTHETIC_SHAPE

    def operation(operation_id):
        contract = evidence_script._PERFORMANCE_OPERATION_CONTRACTS[operation_id]
        parameters = contract["parameters"]
        estimate = evidence_script._expected_memory_estimate(
            operation_id,
            shape,
            "uint16",
            parameters,
        )
        admitted = (
            estimate.runtime_managed_peak_bytes + estimate.uncertainty_bytes
        )
        return {
            "operation_id": operation_id,
            "implementation_id": contract["implementation_id"],
            "parameters": dict(parameters),
            "parity": {
                "profile": "bitwise-identical-boolean-mask",
                "passed": True,
                "mismatch_count": 0,
                "foreground_pixels": 17,
                "cpu_output_dtype": "bool",
                "gpu_output_dtype": "bool",
                "gpu_output_resident": True,
            },
            "samples": {
                "cpu_seconds": cpu,
                "gpu_end_to_end_seconds": end_to_end,
                "gpu_resident_seconds": resident,
            },
            "memory": {
                "model_id": contract["memory_model_id"],
                "runtime_managed_peak_bytes": estimate.runtime_managed_peak_bytes,
                "uncertainty_bytes": estimate.uncertainty_bytes,
                "admitted_device_peak_bytes": admitted,
                "observed_private_pool_peak_bytes": max(1, admitted // 2),
                "observed_within_admitted_peak": True,
                "cleanup": {
                    "passed": True,
                    "used_bytes_after_cleanup": 0,
                    "reserved_bytes_after_cleanup": 0,
                    "error": "",
                },
            },
            "summary": {
                "cpu_median_seconds": 2.0,
                "gpu_end_to_end_median_seconds": 0.5,
                "gpu_resident_median_seconds": 0.4,
                "gpu_end_to_end_speedup": 4.0,
                "gpu_resident_speedup": 5.0,
                "screening_choice": "GPU-CuPy",
            },
        }

    admission_cases = []
    admission_coverage = {
        operation_id: sorted(required)
        for operation_id, required in evidence_script.REQUIRED_COVERAGE.items()
    }
    for expected_case in evidence_script._admission_cases():
        input_shape = tuple(int(size) for size in expected_case.data.shape)
        output_shape = evidence_script._project_admission_output_shape(
            input_shape,
            expected_case.parameters,
        )
        admission_cases.append(
            {
                "case_id": expected_case.case_id,
                "operation_id": expected_case.operation_id,
                "input_shape": list(input_shape),
                "input_dtype": expected_case.data.dtype.name,
                "input_sha256": evidence_script._array_sha256(expected_case.data),
                "parameters": dict(expected_case.parameters),
                "coverage": list(expected_case.coverage),
                "output_shape": list(output_shape),
                "output_dtype": "bool",
                "gpu_output_dtype": "bool",
                "gpu_output_resident": True,
                "cpu_foreground_pixels": 0,
                "gpu_foreground_pixels": 0,
                "mismatch_count": 0,
                "exact_mask_match": True,
            }
        )

    return {
        "schema": evidence_script.SCHEMA,
        "schema_version": evidence_script.SCHEMA_VERSION,
        "created_utc": "2026-07-29T00:00:00+00:00",
        "kind": "scientific-admission-and-machine-local-performance-evidence",
        "portable_performance_claim": False,
        "durable_optimizer_record": False,
        "profile": evidence_script.PROFILE,
        "method": {
            "cpu_path": "production-napari-vipp-operations",
            "gpu_path": "production-exact-cupy-providers",
            "parity": "bitwise-identical-boolean-mask",
            "warmup_rounds": evidence_script.WARMUP_ROUNDS,
            "timed_rounds": evidence_script.BENCHMARK_ROUNDS,
            "cpu_timing_scope": "synchronous-operation-call-v1",
            "gpu_end_to_end_timing_scope": (
                "host-to-device-plus-synchronized-compute-plus-device-to-host-v1"
            ),
            "gpu_resident_timing_scope": "synchronized-resident-compute-v1",
            "disk_io_included": False,
            "input_generation_included": False,
            "exact_parity_required_before_timing": True,
        },
        "platform": {
            "system": "Windows",
            "release": "11",
            "machine": "AMD64",
            "processor": "Fake CPU",
            "python": "3.12.10",
            "python_implementation": "CPython",
            "python_abi": "cpython-312",
            "execution_mode": "native",
            "cuda_device_index": 0,
            "cuda_device_name": evidence_script.PUBLIC_V3_CUDA_DEVICE_NAME,
            "cuda_compute_capability": (
                evidence_script.PUBLIC_V3_CUDA_COMPUTE_CAPABILITY
            ),
            "cuda_driver_version": evidence_script.PUBLIC_V3_CUDA_DRIVER_VERSION,
            "cuda_runtime_version": evidence_script.PUBLIC_V3_CUDA_RUNTIME_VERSION,
            "total_accelerator_memory_bytes": 32 * 1024**3,
        },
        "packages": {
            "napari-vipp": "0.12.0a3",
            "numpy": "2.5.1",
            "scipy": "1.18.0",
            "scikit-image": "0.26.0",
            "cupy-cuda12x": "not-installed",
            "cupy-cuda13x": "14.1.1",
        },
        "source_provenance": evidence_script._source_provenance(),
        "admission": {
            "status": "pass",
            "case_count": len(admission_cases),
            "failure_count": 0,
            "coverage": admission_coverage,
            "cases": admission_cases,
        },
        "performance": {
            "status": "pass",
            "source_count": 1,
            "sources": [
                {
                    "source_id": "synthetic-structured-large-stack",
                    "label": (
                        "Structured synthetic "
                        + "x".join(str(size) for size in shape)
                        + " uint16 stack"
                    ),
                    "source_kind": "deterministic-synthetic",
                    "source_metadata": {"generator": evidence_script.GENERATOR_ID},
                    "direct_private_identifiers_published": False,
                    "shape": list(shape),
                    "dtype": "uint16",
                    "element_count": int(np.prod(shape)),
                    "input_bytes": int(np.prod(shape)) * 2,
                    "input_sha256": "2" * 64,
                    "operations": [
                        operation("canny_edges"),
                        operation("otsu_threshold"),
                    ],
                }
            ],
        },
        "lifecycle": {
            "status": "pass",
            "operations": [
                {
                    "operation_id": operation_id,
                    "implementation_id": contract["implementation_id"],
                    "parameters": dict(contract["parameters"]),
                    "reported_progress": [
                        dict(update) for update in contract["progress"]
                    ],
                    "cancellation_requested": True,
                    "cancellation_observed": True,
                    "cleanup": {
                        "passed": True,
                        "used_bytes_after_cleanup": 0,
                        "reserved_bytes_after_cleanup": 0,
                        "error": "",
                    },
                }
                for operation_id, contract in (
                    evidence_script._LIFECYCLE_OPERATION_CONTRACTS.items()
                )
            ],
        },
    }
