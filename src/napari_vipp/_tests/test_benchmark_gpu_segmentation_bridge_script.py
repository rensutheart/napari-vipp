from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "benchmark_gpu_segmentation_bridge.py"


@pytest.fixture(scope="module")
def evidence_script():
    module_name = "_vipp_test_gpu_segmentation_bridge_evidence"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(module_name, None)


@pytest.mark.parametrize("action", ("import", "--help", "--validate-existing"))
def test_cpu_safe_surfaces_do_not_import_or_initialize_cuda(action, tmp_path):
    guarded = """
import builtins, runpy, sys
real_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'cupy' or name.startswith(('cupy.', 'cupyx', 'cucim')):
        raise RuntimeError('CPU-safe evidence surface imported a GPU package')
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
"""
    if action == "import":
        body = f"runpy.run_path({str(SCRIPT_PATH)!r}, run_name='not_main')"
        expected_returncode = 0
    elif action == "--help":
        body = (
            f"sys.argv = [{str(SCRIPT_PATH)!r}, '--help']; "
            f"runpy.run_path({str(SCRIPT_PATH)!r}, run_name='__main__')"
        )
        expected_returncode = 0
    else:
        artifact = tmp_path / "invalid.json"
        artifact.write_text(json.dumps({}) + "\n", encoding="utf-8")
        body = (
            f"sys.argv = [{str(SCRIPT_PATH)!r}, '--validate-existing', "
            f"{str(artifact)!r}]; "
            f"runpy.run_path({str(SCRIPT_PATH)!r}, run_name='__main__')"
        )
        expected_returncode = 2
    completed = subprocess.run(
        [sys.executable, "-c", guarded + body],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == expected_returncode, completed.stderr
    assert "imported a GPU package" not in completed.stderr


def test_admission_manifest_covers_both_exact_public_regions(evidence_script):
    cases = evidence_script._admission_cases()
    assert [case.case_id for case in cases] == [
        "binary-boundaries-plane",
        "binary-nonfinite-strided",
        "binary-volume",
        "extract-bool-names",
        "extract-u8-types",
        "extract-u16-negative-strided",
        "extract-f32-type-precedence",
    ]
    assert len({case.case_id for case in cases}) == len(cases)
    for operation_id, required in evidence_script.REQUIRED_ADMISSION_COVERAGE.items():
        coverage = {
            item
            for case in cases
            if case.operation_id == operation_id
            for item in case.coverage
        } | {"repeat:deterministic"}
        assert required <= coverage


def test_adversarial_inputs_are_deterministic_and_semantically_exact(
    evidence_script,
):
    cpu_functions, _gpu_functions = evidence_script._operation_functions()
    for case in evidence_script._admission_cases():
        first = evidence_script._host_case(case.kind)
        second = evidence_script._host_case(case.kind)
        assert evidence_script._array_sha256(first) == evidence_script._array_sha256(
            second
        )
        parameters = evidence_script._case_parameters(case.kind)
        result = cpu_functions[case.operation_id](first, **parameters)
        if case.operation_id == "binary_threshold":
            assert first.dtype == np.float32
            assert result.dtype == bool
            assert result.shape == first.shape
        else:
            assert result.dtype == first.dtype
            assert result.ndim == first.ndim - 1


def test_quick_and_full_profiles_cover_each_implementation(evidence_script):
    quick = evidence_script._performance_cases("quick")
    full = evidence_script._performance_cases("full")

    assert quick
    assert tuple(full[: len(quick)]) == quick
    assert len(full) > len(quick)
    for cases in (quick, full):
        assert {case.operation_id for case in cases} == {
            "binary_threshold",
            "extract_channel",
        }
        assert all(
            case.shape and all(size > 0 for size in case.shape) for case in cases
        )
    odd_extract = next(
        case
        for case in quick
        if case.case_id == "extract-c3-plane-31x37-allocator-rounding"
    )
    assert np.prod(odd_extract.shape) * np.dtype(odd_extract.dtype).itemsize == 6_882
    staged_extract = next(
        case
        for case in quick
        if case.case_id == "extract-plane-31x37-c3-contiguous-staging"
    )
    assert staged_extract.shape[-1] == 3


def test_cpu_metadata_and_fallback_facets_are_executable(evidence_script):
    cpu_functions, _gpu_functions = evidence_script._operation_functions()

    metadata = evidence_script._run_metadata(cpu_functions)
    fallback = evidence_script._run_fallback(cpu_functions)

    assert {name: item["status"] for name, item in metadata.items()} == {
        "binary_threshold": "pass",
        "extract_channel": "pass",
    }
    assert fallback["status"] == "pass"
    assert fallback["safe_cpu_fallback_case_count"] == 4
    assert fallback["invalid_authored_case_count"] == 3
    nonfinite = next(
        item
        for item in fallback["safe_cpu_fallback_cases"]
        if item["case_id"] == "binary-nonfinite-threshold"
    )
    assert nonfinite["fallback_allowed"] is True
    assert all(
        item["fallback_allowed"] is False for item in fallback["invalid_authored_cases"]
    )


def test_contract_fingerprints_name_both_custom_public_implementations(
    evidence_script,
):
    contracts = evidence_script._operation_contracts()

    assert set(contracts) == {"binary_threshold", "extract_channel"}
    for operation_id, implementation_id in evidence_script.IMPLEMENTATION_IDS.items():
        contract = contracts[operation_id]
        assert len(contract["sha256"]) == 64
        assert contract["snapshot"]["operation_id"] == operation_id
        assert contract["snapshot"]["implementation_id"] == implementation_id
        assert contract["snapshot"]["admission_tier"] == "public_custom"
        assert (
            contract["snapshot"]["validated_environment_policy_id"]
            == evidence_script.ENVIRONMENT_POLICY_ID
        )


def test_source_provenance_tracks_both_providers_and_policy_owners(evidence_script):
    paths = {path.as_posix() for path in evidence_script.SOURCE_PROVENANCE_PATHS}

    assert {
        "src/napari_vipp/core/gpu/cupy_binary_threshold.py",
        "src/napari_vipp/core/gpu/cupy_extract_channel.py",
        "src/napari_vipp/core/compute_specs.py",
        "src/napari_vipp/core/compute_policy.py",
        "src/napari_vipp/core/compute_planning.py",
        "src/napari_vipp/core/execution.py",
        "src/napari_vipp/core/compute_benchmark_adapter.py",
        "scripts/benchmark_gpu_segmentation_bridge.py",
    } <= paths


def test_environment_and_package_claims_are_schema_checked_and_bound(
    evidence_script,
):
    environment = {
        "system": "Windows",
        "release": "10",
        "machine": "AMD64",
        "python_implementation": "CPython",
        "python_abi": "cpython-312",
        "python": "3.12.9",
        "python_executable": "python.exe",
        "device_index": 0,
        "device_name": "NVIDIA GeForce RTX 5090",
        "compute_capability": "12.0",
        "device_total_memory_bytes": 32 * 1024**3,
        "cuda_driver_version": 13030,
        "cuda_runtime_version": 13020,
    }
    packages = {
        "numpy": "2.5.1",
        "scipy": "1.18.0",
        "scikit-image": "0.26.0",
        "cupy": "14.1.1",
        "cupy-cuda13x": "14.1.1",
        "napari-vipp": evidence_script._project_version(),
    }
    document = {
        "environment": environment,
        "packages": packages,
        "environment_packages_sha256": (
            evidence_script._environment_packages_sha256(environment, packages)
        ),
    }

    evidence_script._validate_environment_and_packages(document)
    assert len(document["environment_packages_sha256"]) == (
        hashlib.sha256().digest_size * 2
    )

    document["environment"] = {**environment, "device_name": "tampered GPU"}
    with pytest.raises(evidence_script.EvidenceError, match="binding is invalid"):
        evidence_script._validate_environment_and_packages(document)

    document["environment"] = environment
    document["packages"] = {**packages, "cupy": "99.0.0"}
    document["environment_packages_sha256"] = (
        evidence_script._environment_packages_sha256(
            document["environment"], document["packages"]
        )
    )
    with pytest.raises(evidence_script.EvidenceError, match="exact admitted pins"):
        evidence_script._validate_environment_and_packages(document)


def test_metadata_and_admission_integrity_tampering_is_rejected(evidence_script):
    metadata = evidence_script._expected_metadata_records()
    evidence_script._validate_metadata_records(metadata)
    tampered_metadata = json.loads(json.dumps(metadata))
    tampered_metadata["extract_channel"][
        "selected_channel_metadata_preserved"
    ] = False
    with pytest.raises(evidence_script.EvidenceError, match="metadata evidence"):
        evidence_script._validate_metadata_records(tampered_metadata)

    admission = {}
    for operation_id in sorted(evidence_script.IMPLEMENTATION_IDS):
        definitions = [
            item
            for item in evidence_script._admission_cases()
            if item.operation_id == operation_id
        ]
        coverage = {
            item for definition in definitions for item in definition.coverage
        } | {"repeat:deterministic"}
        integrity = (
            "independent-bool-allocation-v1"
            if operation_id == "binary_threshold"
            else "shared-input-allocation-view-v1"
        )
        cases = [
            {
                "case_id": definition.case_id,
                "shape": list(evidence_script._host_case(definition.kind).shape),
                "input_dtype": str(
                    evidence_script._host_case(definition.kind).dtype
                ),
                "output_dtype": (
                    "bool"
                    if operation_id == "binary_threshold"
                    else str(evidence_script._host_case(definition.kind).dtype)
                ),
                "parameters": evidence_script._json_value(
                    evidence_script._case_parameters(definition.kind)
                ),
                "input_sha256": evidence_script._array_sha256(
                    evidence_script._host_case(definition.kind)
                ),
                "coverage": sorted(definition.coverage),
                "output_integrity_contract": integrity,
                "cpu_gpu_bitwise_equal": True,
                "gpu_output_resident": True,
                "input_immutable": True,
                "repeat_deterministic": True,
                "repeat_count": evidence_script.ADMISSION_REPEATS,
                "cpu_output_sha256": "a" * 64,
                "gpu_output_sha256": "a" * 64,
                "cleanup": {
                    "device_pool_used_bytes_after_cleanup": 0,
                    "device_pool_reserved_bytes_after_cleanup": 0,
                },
            }
            for definition in definitions
        ]
        admission[operation_id] = {
            "status": "pass",
            "case_count": len(cases),
            "repeat_count": evidence_script.ADMISSION_REPEATS,
            "coverage": sorted(coverage),
            "cases": cases,
        }

    evidence_script._validate_admission_records(admission)
    admission["extract_channel"]["cases"][0]["output_integrity_contract"] = (
        "not-a-view"
    )
    with pytest.raises(evidence_script.EvidenceError, match="integrity evidence"):
        evidence_script._validate_admission_records(admission)


def test_identity_claims_require_utc_timestamp_and_exact_kind(evidence_script):
    identity = {
        "schema": evidence_script.SCHEMA,
        "schema_version": evidence_script.SCHEMA_VERSION,
        "created_utc": "2026-08-13T12:30:00+00:00",
        "kind": evidence_script.EVIDENCE_KIND,
        "profile": "quick",
        "portable_performance_claim": False,
        "durable_optimizer_record": False,
        "implementations": evidence_script._implementation_records(),
    }

    evidence_script._validate_identity_claims(identity)
    identity["kind"] = "tampered"
    with pytest.raises(evidence_script.EvidenceError, match="identity/profile"):
        evidence_script._validate_identity_claims(identity)
    identity["kind"] = evidence_script.EVIDENCE_KIND
    identity["created_utc"] = "2026-08-13T12:30:00"
    with pytest.raises(evidence_script.EvidenceError, match="identity/profile"):
        evidence_script._validate_identity_claims(identity)


def test_fallback_record_is_bound_to_exact_authoritative_cases(evidence_script):
    expected = evidence_script._expected_fallback_record()

    assert [item["case_id"] for item in expected["safe_cpu_fallback_cases"]] == [
        "binary-uint16",
        "binary-luma",
        "binary-nonfinite-threshold",
        "extract-float64",
    ]
    assert [item["case_id"] for item in expected["invalid_authored_cases"]] == [
        "binary-invalid-axis",
        "extract-missing-axis",
        "extract-invalid-index",
    ]
    tampered = json.loads(json.dumps(expected))
    tampered["safe_cpu_fallback_cases"][0] = dict(
        tampered["safe_cpu_fallback_cases"][1]
    )
    assert tampered != evidence_script._expected_fallback_record()
