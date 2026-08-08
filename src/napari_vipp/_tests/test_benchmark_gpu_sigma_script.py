from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "benchmark_gpu_sigma.py"


@pytest.fixture(scope="module")
def evidence_script():
    module_name = "_vipp_test_benchmark_gpu_sigma"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(module_name, None)


def test_help_is_cuda_safe_in_a_fresh_process() -> None:
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
    assert "--profile" in completed.stdout
    assert "Sigma Filter" in completed.stdout


def test_admission_and_rejection_manifests_are_complete_and_deterministic(
    evidence_script,
) -> None:
    first = evidence_script._admission_cases()
    second = evidence_script._admission_cases()
    assert [case.case_id for case in first] == [case.case_id for case in second]
    assert len({case.case_id for case in first}) == len(first)
    coverage = set()
    for first_case, second_case in zip(first, second, strict=True):
        np.testing.assert_array_equal(first_case.data, second_case.data)
        assert first_case.parameters == second_case.parameters
        assert not first_case.data.flags.writeable
        coverage.update(first_case.coverage)
    assert evidence_script.REQUIRED_ADMISSION_COVERAGE <= coverage

    rejections = evidence_script._rejection_cases()
    assert len({case.case_id for case in rejections}) == len(rejections)
    rejection_coverage = {tag for case in rejections for tag in case.coverage}
    assert evidence_script.REQUIRED_REJECTION_COVERAGE <= rejection_coverage


def test_full_profile_has_required_radius_size_grid_and_stacks(
    evidence_script,
) -> None:
    definitions = evidence_script._performance_cases("full")
    plane_cells = {
        (definition.shape[-1], definition.radius)
        for definition in definitions
        if len(definition.shape) == 2
    }
    expected_cells = {
        (extent, radius)
        for extent in evidence_script.PLANE_EXTENTS
        for radius in evidence_script.RADII
    }

    assert plane_cells == expected_cells
    stack_shapes = {
        definition.shape for definition in definitions if len(definition.shape) > 2
    }
    assert stack_shapes == {
        (8, 512, 512),
        (4, 1024, 1024),
    }
    assert evidence_script.BENCHMARK_ROUNDS == 7
    assert evidence_script.BOOTSTRAP_SAMPLES == 2_000


def test_source_provenance_allowlist_tracks_the_extracted_cpu_module(
    evidence_script,
) -> None:
    paths = tuple(path.as_posix() for path in evidence_script.SOURCE_PROVENANCE_PATHS)

    assert "src/napari_vipp/core/sigma_filter.py" in paths
    assert "src/napari_vipp/core/operations.py" not in paths


def test_source_provenance_ignores_unrelated_operations_but_tracks_every_owner(
    evidence_script,
    monkeypatch,
    tmp_path: Path,
) -> None:
    tracked_paths = tuple(evidence_script.SOURCE_PROVENANCE_PATHS)
    operations_path = Path("src/napari_vipp/core/operations.py")
    for relative_path in (*tracked_paths, operations_path):
        destination = tmp_path / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((PROJECT_ROOT / relative_path).read_bytes())

    monkeypatch.setattr(evidence_script, "PROJECT_ROOT", tmp_path)
    baseline = evidence_script._source_provenance()

    unrelated = tmp_path / operations_path
    unrelated.write_bytes(unrelated.read_bytes() + b"\n# unrelated operation\n")
    assert evidence_script._source_provenance() == baseline

    for relative_path in tracked_paths:
        owner = tmp_path / relative_path
        original = owner.read_bytes()
        owner.write_bytes(original + b"\n# provenance mutation\n")
        assert evidence_script._source_provenance() != baseline
        owner.write_bytes(original)
        assert evidence_script._source_provenance() == baseline


def test_synthetic_generator_is_deterministic_structured_and_read_only(
    evidence_script,
) -> None:
    evidence_script._synthetic_image.cache_clear()
    first = evidence_script._synthetic_image((2, 64, 72), 509)
    first_copy = first.copy()
    evidence_script._synthetic_image.cache_clear()
    second = evidence_script._synthetic_image((2, 64, 72), 509)

    np.testing.assert_array_equal(first_copy, second)
    assert second.dtype == np.uint16
    assert second.flags.c_contiguous
    assert not second.flags.writeable
    assert np.unique(second).size > 100
    assert int(second.min()) == 0
    assert int(second.max()) == 65_535


def test_bootstrap_and_reviewed_speed_gates_are_deterministic(
    evidence_script,
) -> None:
    cpu = [0.100, 0.102, 0.098, 0.101, 0.099, 0.103, 0.097]
    gpu = [0.020, 0.021, 0.019, 0.020, 0.021, 0.019, 0.020]
    resident = [0.010] * 7
    transfers = [0.004] * 7
    first = evidence_script._timing_summary(
        cpu,
        gpu,
        resident,
        transfers,
        cpu_cold_seconds=0.11,
        gpu_cold_seconds=0.03,
        bootstrap_seed=123,
    )
    second = evidence_script._timing_summary(
        cpu,
        gpu,
        resident,
        transfers,
        cpu_cold_seconds=0.11,
        gpu_cold_seconds=0.03,
        bootstrap_seed=123,
    )

    assert first == second
    assert first["paired_speedup_confidence_low"] >= 1.2
    assert first["auto_performance_gate_passed"] is True
    assert first["screening_choice"] == "GPU-CuPy"

    too_small = evidence_script._timing_summary(
        [0.010] * 7,
        [0.001] * 7,
        [0.0005] * 7,
        [0.0002] * 7,
        cpu_cold_seconds=0.011,
        gpu_cold_seconds=0.002,
        bootstrap_seed=123,
    )
    assert too_small["confidence_speed_gate_passed"] is True
    assert too_small["material_saving_gate_passed"] is False
    assert too_small["screening_choice"] == "CPU"


def test_atomic_round_trip_and_cpu_safe_validation(
    evidence_script,
    tmp_path,
) -> None:
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


def test_contract_rejects_parity_timing_gate_and_schema_tampering(
    evidence_script,
) -> None:
    document = _example_document(evidence_script)

    parity = deepcopy(document)
    parity["performance"]["results"][0]["parity"]["mismatch_count"] = 1
    with pytest.raises(evidence_script.EvidenceError, match="parity"):
        evidence_script._validate_document_contract(parity)

    timing = deepcopy(document)
    timing["performance"]["results"][0]["summary"]["cpu_median_seconds"] += 1.0
    with pytest.raises(evidence_script.EvidenceError, match="summary"):
        evidence_script._validate_document_contract(timing)

    gate = deepcopy(document)
    gate["performance"]["results"][0]["summary"]["auto_performance_gate_passed"] = False
    with pytest.raises(evidence_script.EvidenceError, match="summary"):
        evidence_script._validate_document_contract(gate)

    unknown = deepcopy(document)
    unknown["unexpected"] = True
    with pytest.raises(evidence_script.EvidenceError, match="fields differ"):
        evidence_script._validate_document_contract(unknown)


def test_validate_existing_rejects_stale_source_and_markdown(
    evidence_script,
    tmp_path,
) -> None:
    document = _example_document(evidence_script)
    stale = deepcopy(document)
    stale["source_provenance"][0]["sha256"] = "0" * 64
    output = tmp_path / "stale.json"
    output.write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(evidence_script.EvidenceError, match="fingerprints"):
        evidence_script.validate_existing(output)

    valid = tmp_path / "valid.json"
    markdown = valid.with_suffix(".md")
    evidence_script._atomic_write_artifacts(valid, markdown, document)
    markdown.write_text("edited", encoding="utf-8")
    with pytest.raises(evidence_script.EvidenceError, match="Markdown"):
        evidence_script.validate_existing(valid)


def test_checked_in_artifact_is_historical_and_internally_valid(
    evidence_script,
    monkeypatch,
) -> None:
    artifact = (
        PROJECT_ROOT / "docs" / "benchmarks" / "sigma-filter-cupy-windows-rtx5090.json"
    )
    raw = artifact.read_text(encoding="utf-8")
    document = json.loads(raw)
    current_source_document = deepcopy(document)
    current_source_document["source_provenance"] = (
        evidence_script._source_provenance()
    )

    with pytest.raises(evidence_script.EvidenceError, match="fingerprints are stale"):
        evidence_script.validate_existing(artifact)
    evidence_script._validate_document_contract(current_source_document)
    assert raw == json.dumps(document, indent=2, sort_keys=True) + "\n"
    historical_source_provenance = deepcopy(document["source_provenance"])
    monkeypatch.setattr(
        evidence_script,
        "_source_provenance",
        lambda: historical_source_provenance,
    )
    assert artifact.with_suffix(".md").read_text(encoding="utf-8") == (
        evidence_script.render_markdown(document)
    )


def _example_document(evidence_script):
    digest = "1" * 64
    admission = {
        "status": "pass",
        "parity_profile": "bitwise-identical-dtype-shape-and-signed-zero-v1",
        "case_count": 1,
        "coverage": sorted(evidence_script.REQUIRED_ADMISSION_COVERAGE),
        "cases": [
            {
                "case_id": "example-admission",
                "shape": [3, 3],
                "dtype": "uint16",
                "parameters": {"radius": 2.0},
                "coverage": sorted(evidence_script.REQUIRED_ADMISSION_COVERAGE),
                "exact": True,
                "mismatch_count": 0,
                "gpu_output_resident": True,
                "gpu_output_contiguous": True,
                "input_immutable": True,
                "cpu_output_sha256": digest,
                "gpu_output_sha256": digest,
            }
        ],
    }
    rejections = {
        "status": "pass",
        "case_count": 1,
        "coverage": sorted(evidence_script.REQUIRED_REJECTION_COVERAGE),
        "cases": [
            {
                "case_id": "example-rejection",
                "coverage": sorted(evidence_script.REQUIRED_REJECTION_COVERAGE),
                "error_type": "ValueError",
                "message": "rejected",
                "cpu_gpu_message_exact": True,
            }
        ],
    }
    lifecycle = {
        "status": "pass",
        "cancelled": True,
        "boundary": "synchronized-row-tile-v1",
        "reported_progress": [
            {"current": 0, "total": 3, "message": "Sigma Filter rows"},
            {"current": 1, "total": 3, "message": "Sigma Filter rows"},
        ],
        "device_pool_used_bytes_after_cleanup": 0,
        "device_pool_reserved_bytes_after_cleanup": 0,
    }
    results = []
    for index, definition in enumerate(
        evidence_script._performance_cases("quick"),
        start=1,
    ):
        cpu = [0.100 + offset * 0.001 for offset in range(7)]
        gpu = [0.020 + offset * 0.0001 for offset in range(7)]
        resident = [0.010 + offset * 0.0001 for offset in range(7)]
        transfers = [0.004 + offset * 0.00005 for offset in range(7)]
        seed = evidence_script.BOOTSTRAP_SEED + index
        summary = evidence_script._timing_summary(
            cpu,
            gpu,
            resident,
            transfers,
            cpu_cold_seconds=0.11,
            gpu_cold_seconds=0.03,
            bootstrap_seed=seed,
        )
        results.append(
            {
                "case_id": definition.case_id,
                "label": definition.label,
                "source_kind": definition.source_kind,
                "source_metadata": {
                    "generator": evidence_script.GENERATOR_ID,
                    "seed": definition.seed,
                },
                "shape": list(definition.shape),
                "element_count": int(np.prod(definition.shape)),
                "input_bytes": int(np.prod(definition.shape)) * 2,
                "dtype": "uint16",
                "radius": definition.radius,
                "parameters": {
                    "radius": definition.radius,
                    "sigma_width": 2.0,
                    "minimum_pixel_fraction": 0.2,
                    "outlier_aware": True,
                    "channel_axis": None,
                },
                "input_sha256": "2" * 64,
                "parity": {
                    "profile": "bitwise-identical-uint16-v1",
                    "passed": True,
                    "mismatch_count": 0,
                    "gpu_output_resident": True,
                    "cpu_output_sha256": digest,
                    "gpu_output_sha256": digest,
                },
                "samples": {
                    "cpu_seconds": cpu,
                    "gpu_end_to_end_seconds": gpu,
                    "gpu_resident_seconds": resident,
                    "gpu_transfer_seconds": transfers,
                    "cpu_case_cold_seconds": 0.11,
                    "gpu_case_cold_end_to_end_seconds": 0.03,
                },
                "summary": summary,
                "bootstrap_seed": seed,
                "cleanup": {
                    "used_bytes_before_cleanup": 0,
                    "used_bytes_after_cleanup": 0,
                    "reserved_bytes_after_cleanup": 0,
                },
            }
        )
    performance = {
        "status": "pass",
        "case_count": len(results),
        "results": results,
        "crossover": evidence_script._crossover_summary(results),
    }
    return {
        "schema": evidence_script.SCHEMA,
        "schema_version": evidence_script.SCHEMA_VERSION,
        "created_utc": "2026-08-02T00:00:00+00:00",
        "kind": "scientific-admission-and-machine-local-performance-evidence",
        "portable_performance_claim": False,
        "durable_optimizer_record": False,
        "profile": "quick",
        "method": evidence_script._method_record("quick"),
        "platform": {"device_name": "Fake GPU"},
        "packages": {"cupy-cuda13x": "14.1.1"},
        "source_provenance": evidence_script._source_provenance(),
        "admission": admission,
        "rejections": rejections,
        "lifecycle": lifecycle,
        "performance": performance,
    }
