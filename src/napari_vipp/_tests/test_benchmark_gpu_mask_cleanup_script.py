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
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "benchmark_gpu_mask_cleanup.py"


@pytest.fixture(scope="module")
def evidence_script():
    module_name = "_vipp_test_gpu_mask_cleanup_evidence"
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


def test_admission_cases_cover_both_exact_regions(evidence_script):
    cases = evidence_script._admission_cases()

    assert [case.case_id for case in cases] == [
        "fill-2d-face-odd-leading",
        "fill-2d-full-strided-checker",
        "fill-3d-face-leading",
        "fill-empty-spatial-blocks",
        "remove-2d-face-threshold-odd",
        "remove-2d-full-strided-checker",
        "remove-3d-face-leading",
        "remove-empty-identity",
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


def test_adversarial_masks_are_deterministic_readonly_safe_and_semantic(
    evidence_script,
):
    cpu_functions, _gpu_functions = evidence_script._operation_functions()
    for case in evidence_script._admission_cases():
        first = evidence_script._host_case(case.kind)
        second = evidence_script._host_case(case.kind)
        assert evidence_script._array_sha256(first) == evidence_script._array_sha256(
            second
        )
        before = evidence_script._array_sha256(first)
        first.setflags(write=False)
        result = cpu_functions[case.operation_id](
            first,
            **evidence_script._case_parameters(case.kind),
        )
        assert result.shape == first.shape
        assert result.dtype == bool
        assert evidence_script._array_sha256(first) == before
    strided = [
        evidence_script._host_case(case.kind)
        for case in evidence_script._admission_cases()
        if "layout:noncontiguous" in case.coverage
    ]
    assert strided and all(not item.flags.c_contiguous for item in strided)


def test_quick_and_full_profiles_cover_odd_padded_checkerboards(evidence_script):
    quick = evidence_script._performance_cases("quick")
    full = evidence_script._performance_cases("full")

    assert tuple(full[: len(quick)]) == quick
    assert len(full) > len(quick)
    for cases in (quick, full):
        assert {case.operation_id for case in cases} == {
            "fill_holes",
            "remove_small_objects",
        }
    fill = next(case for case in quick if case.operation_id == "fill_holes")
    raw = np.prod(fill.shape[-fill.spatial_ndim :])
    padded = np.prod(
        [size + 2 for size in fill.shape[-fill.spatial_ndim :]],
    )
    assert fill.pattern == "checkerboard"
    assert all(size % 2 for size in fill.shape[-fill.spatial_ndim :])
    assert padded > raw


def test_memory_contract_uses_padded_fill_and_conservative_checker_allowance(
    evidence_script,
):
    cases = evidence_script._performance_cases("quick")
    fill = next(case for case in cases if case.operation_id == "fill_holes")
    remove = next(case for case in cases if case.operation_id == "remove_small_objects")
    fill_estimate = evidence_script._performance_memory_estimate(fill)
    remove_estimate = evidence_script._performance_memory_estimate(remove)
    full_elements = np.prod(fill.shape)
    padded = np.prod([size + 2 for size in fill.shape[-fill.spatial_ndim :]])

    assert fill_estimate.model_id == "cupyx-fill-holes-memory-v1"
    assert fill_estimate.runtime_managed_peak_bytes == (2 * full_elements + 13 * padded)
    assert remove_estimate.model_id == "cupyx-remove-small-objects-memory-v1"
    assert remove_estimate.runtime_managed_peak_bytes > 2 * np.prod(remove.shape)
    assert fill_estimate.uncertainty_bytes >= 8 * 1024**2
    assert remove_estimate.uncertainty_bytes >= 8 * 1024**2


def test_cpu_metadata_and_fallback_facets_are_executable(evidence_script):
    cpu_functions, _gpu_functions = evidence_script._operation_functions()

    metadata = evidence_script._run_metadata(cpu_functions)
    fallback = evidence_script._run_fallback(cpu_functions)

    assert {name: item["status"] for name, item in metadata.items()} == {
        "fill_holes": "pass",
        "remove_small_objects": "pass",
    }
    assert all(item["structural_metadata_equal"] for item in metadata.values())
    assert metadata["fill_holes"]["resolved_spatial_ndim"] == 2
    assert metadata["remove_small_objects"]["resolved_spatial_ndim"] == 3
    assert fallback["status"] == "pass"
    assert fallback["safe_cpu_fallback_case_count"] == 3
    assert fallback["invalid_authored_case_count"] == 3
    assert all(
        item["fallback_allowed"] is True for item in fallback["safe_cpu_fallback_cases"]
    )
    assert all(
        item["fallback_allowed"] is False for item in fallback["invalid_authored_cases"]
    )


def test_contract_fingerprints_name_both_custom_public_implementations(
    evidence_script,
):
    contracts = evidence_script._operation_contracts()

    assert set(contracts) == {"fill_holes", "remove_small_objects"}
    expected_memory = {
        "fill_holes": "cupyx-fill-holes-memory-v1",
        "remove_small_objects": "cupyx-remove-small-objects-memory-v1",
    }
    for operation_id, implementation_id in evidence_script.IMPLEMENTATION_IDS.items():
        contract = contracts[operation_id]
        snapshot = contract["snapshot"]
        assert len(contract["sha256"]) == 64
        assert snapshot["operation_id"] == operation_id
        assert snapshot["implementation_id"] == implementation_id
        assert snapshot["admission_tier"] == "public_custom"
        assert snapshot["parity_policy_id"] == "mask-bitwise-v1"
        assert snapshot["memory_model_id"] == expected_memory[operation_id]
        assert (
            snapshot["validated_environment_policy_id"]
            == evidence_script.ENVIRONMENT_POLICY_ID
        )


def test_source_provenance_tracks_both_providers_and_all_policy_owners(
    evidence_script,
):
    paths = {path.as_posix() for path in evidence_script.SOURCE_PROVENANCE_PATHS}

    assert {
        "src/napari_vipp/core/gpu/cupy_fill_holes.py",
        "src/napari_vipp/core/gpu/cupy_remove_small_objects.py",
        "src/napari_vipp/core/compute_specs.py",
        "src/napari_vipp/core/compute_policy.py",
        "src/napari_vipp/core/compute_planning.py",
        "src/napari_vipp/core/execution.py",
        "src/napari_vipp/core/compute_benchmark_adapter.py",
        "scripts/benchmark_gpu_mask_cleanup.py",
    } <= paths
    provenance = evidence_script._source_provenance()
    assert {item["path"] for item in provenance["files"]} == paths
    assert all(len(item["sha256"]) == 64 for item in provenance["files"])


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
            document["environment"],
            document["packages"],
        )
    )
    with pytest.raises(evidence_script.EvidenceError, match="exact admitted pins"):
        evidence_script._validate_environment_and_packages(document)


def test_metadata_and_admission_integrity_tampering_is_rejected(evidence_script):
    metadata = evidence_script._expected_metadata_records()
    evidence_script._validate_metadata_records(metadata)
    tampered_metadata = json.loads(json.dumps(metadata))
    tampered_metadata["fill_holes"]["structural_metadata_equal"] = False
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
        cases = [
            {
                "case_id": definition.case_id,
                "shape": list(evidence_script._host_case(definition.kind).shape),
                "input_dtype": "bool",
                "output_dtype": "bool",
                "parameters": evidence_script._json_value(
                    evidence_script._case_parameters(definition.kind)
                ),
                "coverage": sorted(definition.coverage),
                "input_sha256": evidence_script._array_sha256(
                    evidence_script._host_case(definition.kind)
                ),
                "output_integrity_contract": "independent-bool-allocation-v1",
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
    admission["fill_holes"]["cases"][0]["output_integrity_contract"] = "shared"
    with pytest.raises(evidence_script.EvidenceError, match="integrity evidence"):
        evidence_script._validate_admission_records(admission)


def test_identity_and_fallback_records_are_exactly_bound(evidence_script):
    identity = {
        "schema": evidence_script.SCHEMA,
        "schema_version": evidence_script.SCHEMA_VERSION,
        "created_utc": "2026-08-14T12:30:00+00:00",
        "kind": evidence_script.EVIDENCE_KIND,
        "profile": "quick",
        "portable_performance_claim": False,
        "durable_optimizer_record": False,
        "implementations": evidence_script._implementation_records(),
    }
    evidence_script._validate_identity_claims(identity)
    identity["created_utc"] = "2026-08-14T12:30:00"
    with pytest.raises(evidence_script.EvidenceError, match="identity/profile"):
        evidence_script._validate_identity_claims(identity)

    expected = evidence_script._expected_fallback_record()
    assert [item["case_id"] for item in expected["safe_cpu_fallback_cases"]] == [
        "fill-positive-size-limit",
        "fill-numeric-nonzero-mask",
        "remove-integer-label-preservation",
    ]
    assert [item["case_id"] for item in expected["invalid_authored_cases"]] == [
        "fill-invalid-connectivity",
        "remove-spatial-rank-disagreement",
        "remove-noninteger-size",
    ]


def test_atomic_writer_removes_temporary_file_on_replace_failure(
    evidence_script,
    monkeypatch,
    tmp_path,
):
    output = tmp_path / "evidence.json"

    def fail_replace(_source, _destination):
        raise OSError("replace failed")

    monkeypatch.setattr(evidence_script.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        evidence_script._atomic_write_json(output, {"status": "pass"})
    assert not output.exists()
    assert list(tmp_path.iterdir()) == []
