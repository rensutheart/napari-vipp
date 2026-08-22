from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "benchmark_gpu_measurements.py"
ARTIFACT_PATH = (
    PROJECT_ROOT / "docs" / "benchmarks" / "measurements-cupy-windows-rtx5090.json"
)


@pytest.fixture(scope="module")
def evidence_script():
    module_name = "_vipp_test_benchmark_gpu_measurements"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(module_name, None)


def _cleanup() -> dict[str, int]:
    return {
        "device_pool_used_bytes_after_cleanup": 0,
        "device_pool_reserved_bytes_after_cleanup": 0,
    }


def test_environment_record_uses_only_a_privacy_safe_executable_name(
    monkeypatch,
    evidence_script,
) -> None:
    runtime = SimpleNamespace(
        getDeviceProperties=lambda _index: {
            "name": b"Synthetic GPU",
            "major": 12,
            "minor": 0,
            "totalGlobalMem": 1024,
        },
        driverGetVersion=lambda: 13030,
        runtimeGetVersion=lambda: 13020,
        getDeviceCount=lambda: 1,
    )
    cp = SimpleNamespace(cuda=SimpleNamespace(runtime=runtime))
    monkeypatch.setattr(
        evidence_script.sys,
        "executable",
        "/home/researcher/private-worktree/.venv/bin/python",
    )

    environment = evidence_script._environment_record(cp, 0)

    assert environment["python_executable"] == "python"


def _synthetic_document(evidence_script, *, profile: str = "quick"):
    admission_cases = []
    for definition in evidence_script._admission_cases():
        admission_cases.append(
            {
                "case_id": definition.case_id,
                "parity": {"passed": True},
                "gpu_repeat_table_sha256": ["a" * 64] * 3,
                "input_immutable": True,
                "resident_output_contiguous": True,
                "cleanup": _cleanup(),
            }
        )
    rejection_cases = [
        {
            "case_id": definition.case_id,
            "rejected": True,
            "cleanup": _cleanup(),
        }
        for definition in evidence_script._rejection_cases()
    ]
    lifecycle_cases = []
    for operation_id, expected_total in (
        (evidence_script.MORPHOLOGY_OPERATION_ID, 13),
        (evidence_script.INTENSITY_OPERATION_ID, 19),
    ):
        lifecycle_cases.append(
            {
                "operation_id": operation_id,
                "cancellation_observed": True,
                "post_cancellation_reuse_parity": True,
                "complete_updates": [
                    {"current": current, "total": expected_total}
                    for current in range(expected_total + 1)
                ],
                "expected_total": expected_total,
                "cleanup": _cleanup(),
            }
        )
    rounds = evidence_script.BENCHMARK_ROUNDS if profile == "full" else 3
    samples = [0.5, 0.6, 0.7][:rounds]
    performance_cases = []
    for definition in evidence_script._performance_cases(profile):
        performance_cases.append(
            {
                "case_id": definition.case_id,
                "label": definition.label,
                "object_row_count": 7,
                "parity": {"passed": True},
                "samples": {
                    "cpu_seconds": list(samples),
                    "gpu_resident_compute_seconds": list(samples),
                    "gpu_resident_public_seconds": list(samples),
                    "gpu_transfer_inclusive_seconds": list(samples),
                },
                "summary": {
                    "cpu_median_seconds": 0.6,
                    "gpu_resident_compute_median_seconds": 0.6,
                    "gpu_resident_public_median_seconds": 0.6,
                    "gpu_transfer_inclusive_median_seconds": 0.6,
                    "gpu_transfer_inclusive_speedup": 1.0,
                    "screening_choice": "CPU",
                },
                "memory": {
                    "observed_reserved_bytes": 64,
                    "device_peak_with_uncertainty_bytes": 128,
                    "estimate_with_uncertainty_covers_observed": True,
                    **_cleanup(),
                },
                "cleanup": _cleanup(),
            }
        )
    return {
        "schema": evidence_script.SCHEMA,
        "schema_version": evidence_script.SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "kind": "scientific-admission-and-machine-local-performance-evidence",
        "portable_performance_claim": False,
        "durable_optimizer_record": False,
        "profile": profile,
        "method": evidence_script._method_record(profile, rounds),
        "environment": {
            "device_name": "Synthetic test device (not evidence)",
            "system": "test",
            "python": "test",
        },
        "packages": {},
        "source_provenance": evidence_script._source_provenance(),
        "operation_contracts": evidence_script._operation_contracts(),
        "admission": {
            "status": "pass",
            "case_count": len(admission_cases),
            "coverage": sorted(evidence_script.REQUIRED_ADMISSION_COVERAGE),
            "parity_policy_id": evidence_script.PARITY_POLICY_ID,
            "cases": admission_cases,
        },
        "rejections": {
            "status": "pass",
            "case_count": len(rejection_cases),
            "coverage": sorted(evidence_script.REQUIRED_REJECTION_COVERAGE),
            "cases": rejection_cases,
        },
        "lifecycle": {
            "status": "pass",
            "case_count": 2,
            "cases": lifecycle_cases,
        },
        "performance": {
            "status": "pass",
            "rounds": rounds,
            "case_count": len(performance_cases),
            "all_memory_estimates_cover_observed": True,
            "results": performance_cases,
        },
    }


def test_help_is_cuda_and_numpy_safe_in_a_fresh_process() -> None:
    code = "\n".join(
        (
            "import builtins, runpy, sys",
            "real_import = builtins.__import__",
            "def guarded_import(name, *args, **kwargs):",
            "    if name == 'numpy' or name.startswith(('numpy.', 'cupy')):",
            "        raise RuntimeError('help imported numeric or CUDA libraries')",
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
    assert "measurements" in completed.stdout.lower()


def test_admission_manifest_covers_the_complete_first_region(evidence_script) -> None:
    first = evidence_script._admission_cases()
    second = evidence_script._admission_cases()

    assert first == second
    assert len(first) == 11
    assert len({case.case_id for case in first}) == len(first)
    coverage = {tag for case in first for tag in case.coverage}
    assert evidence_script.REQUIRED_ADMISSION_COVERAGE <= coverage
    assert {case.operation_id for case in first} == {
        evidence_script.MORPHOLOGY_OPERATION_ID,
        evidence_script.INTENSITY_OPERATION_ID,
    }
    assert {case.intensity_dtype for case in first if case.intensity_dtype} == {
        "bool",
        "uint8",
        "uint16",
        "float32",
    }
    assert any(case.canonical_shape != case.shape for case in first)
    assert any(case.calibrated for case in first)


def test_rejection_manifest_names_every_scientific_fallback(evidence_script) -> None:
    definitions = evidence_script._rejection_cases()
    coverage = {tag for case in definitions for tag in case.coverage}

    assert len(definitions) == 11
    assert evidence_script.REQUIRED_REJECTION_COVERAGE <= coverage
    assert len({case.case_id for case in definitions}) == len(definitions)


def test_full_performance_manifest_has_crossovers_and_large_confocal_shapes(
    evidence_script,
) -> None:
    definitions = evidence_script._performance_cases("full")
    plane_matrix = {
        (definition.shape[0], definition.intensity_dtype)
        for definition in definitions
        if len(definition.shape) == 2
    }

    assert plane_matrix == {
        (extent, intensity_dtype)
        for extent in evidence_script.PLANE_EXTENTS
        for intensity_dtype in (None, "uint16")
    }
    assert any(
        definition.shape == (64, 512, 512)
        and definition.spatial_ndim == 3
        and definition.intensity_dtype == "uint16"
        for definition in definitions
    )
    assert any(
        definition.shape == (16, 1024, 1024) and definition.spatial_ndim == 2
        for definition in definitions
    )
    assert any(
        len(definition.shape) > definition.spatial_ndim for definition in definitions
    )
    assert any(
        definition.case_id == "many-objects-256x256-intensity-uint16"
        and definition.pattern == "many-objects"
        for definition in definitions
    )
    assert evidence_script.BENCHMARK_ROUNDS == 5


def test_source_provenance_is_strictly_operation_owned(evidence_script) -> None:
    paths = tuple(path.as_posix() for path in evidence_script.SOURCE_PROVENANCE_PATHS)

    assert paths == (
        "src/napari_vipp/core/measurements.py",
        "src/napari_vipp/core/operations.py",
        "src/napari_vipp/core/compute_policy.py",
        "src/napari_vipp/core/compute_specs.py",
        "src/napari_vipp/core/gpu/cupy_measurements.py",
        "scripts/benchmark_gpu_measurements.py",
    )
    assert "src/napari_vipp/core/execution.py" not in paths


def test_provenance_detects_each_scientific_and_policy_owner(
    evidence_script,
    monkeypatch,
    tmp_path: Path,
) -> None:
    for relative_path in evidence_script.SOURCE_PROVENANCE_PATHS:
        destination = tmp_path / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((PROJECT_ROOT / relative_path).read_bytes())
    monkeypatch.setattr(evidence_script, "PROJECT_ROOT", tmp_path)
    baseline = evidence_script._source_provenance()

    for relative_path in evidence_script.SOURCE_PROVENANCE_PATHS:
        owner = tmp_path / relative_path
        original = owner.read_bytes()
        owner.write_bytes(original + b"\n# provenance mutation\n")
        assert evidence_script._source_provenance() != baseline
        owner.write_bytes(original)
        assert evidence_script._source_provenance() == baseline


def test_operation_contracts_pin_both_operations_and_host_boundary(
    evidence_script,
) -> None:
    contracts = evidence_script._operation_contracts()
    snapshots = [item["snapshot"] for item in contracts["contracts"]]
    encoded = json.dumps(
        contracts["contracts"],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")

    assert {item["operation_id"] for item in snapshots} == {
        "measure_objects",
        "measure_objects_intensity",
    }
    assert {item["implementation_id"] for item in snapshots} == {
        "cupy-measure-objects-basic-v1",
        "cupy-measure-objects-intensity-basic-v1",
    }
    assert {item["memory_model_id"] for item in snapshots} == {
        "cupy-basic-measurements-memory-v1"
    }
    assert {item["implementation_library_id"] for item in snapshots} == {"cupy"}
    assert all(
        item["parity_policy_id"] == "basic-measurement-table-v1" for item in snapshots
    )
    assert all(
        item["table_boundary"] == "mandatory-typed-host-finalizer" for item in snapshots
    )
    assert contracts["sha256"] == hashlib.sha256(encoded).hexdigest()


def test_generators_are_deterministic_readonly_sparse_repeated_and_empty(
    evidence_script,
) -> None:
    np = pytest.importorskip("numpy")
    sparse = evidence_script._make_labels((3, 31, 33), 2, "sparse", 509)
    repeated = evidence_script._make_labels((17, 31, 33), 3, "repeated", 510)
    empty = evidence_script._make_labels((31, 33), 2, "empty", 511)

    np.testing.assert_array_equal(
        sparse,
        evidence_script._make_labels((3, 31, 33), 2, "sparse", 509),
    )
    assert sparse.dtype == np.int32 and sparse.flags.c_contiguous
    assert not sparse.flags.writeable
    positive = np.unique(sparse[sparse > 0])
    assert positive.size >= 2 and np.any(np.diff(positive) > 1)
    repeated_id = 700_001
    assert np.count_nonzero(repeated == repeated_id) > 1
    assert not np.any(empty)

    for dtype in ("bool", "uint8", "uint16", "float32"):
        first = evidence_script._make_intensity((19, 23), dtype, 812)
        second = evidence_script._make_intensity((19, 23), dtype, 812)
        np.testing.assert_array_equal(first, second)
        assert first.dtype == np.dtype(dtype)
        assert first.flags.c_contiguous and not first.flags.writeable
        if dtype == "float32":
            assert np.all(np.isfinite(first))


def test_memory_model_matches_registered_formula(evidence_script) -> None:
    from napari_vipp.core.compute import WorkloadDescriptor
    from napari_vipp.core.compute_policy import estimate_candidate_memory
    from napari_vipp.core.compute_specs import compute_specs_for

    morphology = evidence_script._estimated_memory(
        (8, 512, 512),
        2,
        8 * 512 * 512 * 4,
        include_intensity=False,
    )
    intensity = evidence_script._estimated_memory(
        (64, 512, 512),
        3,
        64 * 512 * 512 * 6,
        include_intensity=True,
    )

    # Morphology: one leading index + 3 fixed + 3*2 spatial packed columns.
    assert morphology["packed_output_upper_bound_bytes"] == 8 * 512 * 512 * 10 * 8
    assert morphology["active_block_workspace_bytes"] == 512 * 512 * 128
    # Intensity: no leading index + 3 fixed + 3*3 spatial + 5 intensity columns.
    assert intensity["packed_output_upper_bound_bytes"] == 64 * 512 * 512 * 17 * 8
    assert intensity["active_block_workspace_bytes"] == 64 * 512 * 512 * 224
    assert morphology["uncertainty_bytes"] >= 64 * 1024**2
    assert (
        intensity["device_peak_with_uncertainty_bytes"]
        > intensity["runtime_managed_peak_bytes"]
    )
    for operation_id, shape, spatial_ndim, benchmark_estimate in (
        ("measure_objects", (8, 512, 512), 2, morphology),
        ("measure_objects_intensity", (64, 512, 512), 3, intensity),
    ):
        (spec,) = compute_specs_for(operation_id, include_cpu=False)
        input_shapes = (shape, shape) if "intensity" in operation_id else (shape,)
        input_dtypes = (
            ("int32", "uint16") if "intensity" in operation_id else ("int32",)
        )
        workload = WorkloadDescriptor(
            "measurements",
            operation_id,
            input_shapes,
            input_dtypes,
            parameters=(
                (
                    "spatial_mode",
                    "2D YX" if spatial_ndim == 2 else "3D ZYX",
                ),
            ),
            resolved_spatial_ndim=spatial_ndim,
        )
        production = estimate_candidate_memory(spec, workload)
        assert benchmark_estimate["runtime_managed_peak_bytes"] == (
            production.runtime_managed_peak_bytes
        )
        assert benchmark_estimate["device_peak_with_uncertainty_bytes"] == (
            production.total_device_peak_bytes + production.uncertainty_bytes
        )
        assert benchmark_estimate["host_materialization_peak_bytes"] == (
            production.host_materialization_peak_bytes
        )


def test_synthetic_round_trip_validation_and_markdown_are_cuda_safe(
    evidence_script,
    tmp_path: Path,
) -> None:
    document = _synthetic_document(evidence_script)
    output = tmp_path / "synthetic-not-evidence.json"
    markdown = output.with_suffix(".md")
    evidence_script._atomic_write_artifacts(output, markdown, document)

    assert evidence_script.validate_existing(output) == output.resolve()
    raw = output.read_text(encoding="utf-8")
    assert raw == evidence_script._canonical_json(document)
    assert "mandatory packed-result transfer" in markdown.read_text(encoding="utf-8")

    code = "\n".join(
        (
            "import builtins, runpy, sys",
            "real_import = builtins.__import__",
            "def guarded_import(name, *args, **kwargs):",
            "    if name == 'numpy' or name.startswith(('numpy.', 'cupy')):",
            (
                "        raise RuntimeError('validation imported numeric or CUDA "
                "libraries')"
            ),
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


def test_checked_in_full_artifact_is_current_and_complete(evidence_script) -> None:
    assert evidence_script.validate_existing(ARTIFACT_PATH) == ARTIFACT_PATH.resolve()
    raw = ARTIFACT_PATH.read_text(encoding="utf-8")
    document = json.loads(raw)

    assert raw == evidence_script._canonical_json(document)
    assert document["profile"] == "full"
    assert document["admission"]["case_count"] == 11
    assert document["rejections"]["case_count"] == 11
    assert document["lifecycle"]["case_count"] == 2
    assert document["performance"]["case_count"] == 15
    assert document["performance"]["all_memory_estimates_cover_observed"] is True
    confocal = next(
        result
        for result in document["performance"]["results"]
        if result["case_id"] == "confocal-volume-64x512x512-intensity-uint16"
    )
    assert confocal["element_count"] == 16_777_216
    assert confocal["intensity_dtype"] == "uint16"
    assert confocal["parity"]["passed"] is True
    assert confocal["memory"]["estimate_with_uncertainty_covers_observed"] is True


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda value: value.update(unexpected=True), "root fields differ"),
        (
            lambda value: value["source_provenance"][0].update(sha256="0" * 64),
            "fingerprints are stale",
        ),
        (
            lambda value: value["operation_contracts"].update(sha256="0" * 64),
            "contracts are stale",
        ),
        (
            lambda value: value["admission"]["cases"][0][
                "gpu_repeat_table_sha256"
            ].__setitem__(1, "b" * 64),
            "not exactly deterministic",
        ),
        (
            lambda value: value["performance"]["results"][0]["summary"].update(
                cpu_median_seconds=99.0
            ),
            "summary is inconsistent",
        ),
        (
            lambda value: value["performance"]["results"][0]["memory"].update(
                device_peak_with_uncertainty_bytes=1
            ),
            "memory estimate does not cover",
        ),
    ),
)
def test_validator_rejects_tampering(
    evidence_script,
    mutation,
    message: str,
) -> None:
    document = deepcopy(_synthetic_document(evidence_script))
    mutation(document)

    with pytest.raises(evidence_script.EvidenceError, match=message):
        evidence_script._validate_document_contract(document)


def test_validate_existing_rejects_noncanonical_json_and_edited_markdown(
    evidence_script,
    tmp_path: Path,
) -> None:
    document = _synthetic_document(evidence_script)
    output = tmp_path / "synthetic-not-evidence.json"
    markdown = output.with_suffix(".md")
    evidence_script._atomic_write_artifacts(output, markdown, document)

    output.write_text(output.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(evidence_script.EvidenceError, match="canonical"):
        evidence_script.validate_existing(output)

    evidence_script._atomic_write_artifacts(output, markdown, document)
    markdown.write_text(
        markdown.read_text(encoding="utf-8") + "edited\n",
        encoding="utf-8",
    )
    with pytest.raises(evidence_script.EvidenceError, match="stale or edited"):
        evidence_script.validate_existing(output)
