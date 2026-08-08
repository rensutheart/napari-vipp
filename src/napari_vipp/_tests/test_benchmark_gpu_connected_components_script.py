from __future__ import annotations

import hashlib
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
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "benchmark_gpu_connected_components.py"
ARTIFACT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "benchmarks"
    / "connected-components-cupyx-windows-rtx5090.json"
)


@pytest.fixture(scope="module")
def evidence_script():
    module_name = "_vipp_test_benchmark_gpu_connected_components"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(module_name, None)


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
        r"C:\Users\researcher\private-worktree\.venv\Scripts\python.exe",
    )

    environment = evidence_script._environment_record(cp, 0)

    assert environment["python_executable"] == "python.exe"


def test_help_is_cuda_safe_in_a_fresh_process() -> None:
    code = "\n".join(
        (
            "import builtins, runpy, sys",
            "real_import = builtins.__import__",
            "def guarded_import(name, *args, **kwargs):",
            "    if name == 'cupy' or name.startswith(('cupy.', 'cupyx')):",
            "        raise RuntimeError('help imported CUDA libraries')",
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
    assert "connected-components" in completed.stdout.lower()


def test_admission_manifest_is_complete_and_deterministic(evidence_script) -> None:
    first = evidence_script._admission_cases()
    second = evidence_script._admission_cases()

    assert first == second
    assert len(first) == 16
    assert len({case.case_id for case in first}) == len(first)
    coverage = {tag for case in first for tag in case.coverage}
    assert evidence_script.REQUIRED_ADMISSION_COVERAGE <= coverage
    matrix = {(case.spatial_ndim, case.pattern, case.connectivity) for case in first}
    for spatial_ndim in (2, 3):
        for pattern in ("sparse", "dense", "checkerboard"):
            for connectivity in ("Face connected", "Full connectivity"):
                assert (spatial_ndim, pattern, connectivity) in matrix


def test_full_performance_manifest_has_crossovers_stacks_and_confocal_volume(
    evidence_script,
) -> None:
    definitions = evidence_script._performance_cases("full")
    plane_matrix = {
        (definition.shape[0], definition.pattern, definition.connectivity)
        for definition in definitions
        if len(definition.shape) == 2
    }
    expected_plane_matrix = {
        (extent, pattern, connectivity)
        for extent in evidence_script.PLANE_EXTENTS
        for pattern, connectivity in (
            ("sparse", "Face connected"),
            ("dense", "Full connectivity"),
            ("checkerboard", "Face connected"),
        )
    }

    assert plane_matrix == expected_plane_matrix
    assert any(
        definition.shape == (64, 512, 512) and definition.spatial_ndim == 3
        for definition in definitions
    )
    assert any(
        len(definition.shape) > definition.spatial_ndim for definition in definitions
    )
    assert evidence_script.BENCHMARK_ROUNDS == 5


def test_source_provenance_tracks_all_scientific_and_policy_owners(
    evidence_script,
) -> None:
    paths = tuple(path.as_posix() for path in evidence_script.SOURCE_PROVENANCE_PATHS)

    assert paths == (
        "src/napari_vipp/core/connected_components.py",
        "src/napari_vipp/core/gpu/cupy_connected_components.py",
        "src/napari_vipp/core/compute_specs.py",
        "src/napari_vipp/core/compute_policy.py",
        "src/napari_vipp/core/compute_benchmark_adapter.py",
        "scripts/benchmark_gpu_connected_components.py",
    )
    assert "src/napari_vipp/core/operations.py" not in paths


def test_source_provenance_detects_each_owner_but_ignores_operations_reexport(
    evidence_script,
    monkeypatch,
    tmp_path: Path,
) -> None:
    tracked = tuple(evidence_script.SOURCE_PROVENANCE_PATHS)
    operations_path = Path("src/napari_vipp/core/operations.py")
    for relative_path in (*tracked, operations_path):
        destination = tmp_path / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((PROJECT_ROOT / relative_path).read_bytes())
    monkeypatch.setattr(evidence_script, "PROJECT_ROOT", tmp_path)
    baseline = evidence_script._source_provenance()

    operations = tmp_path / operations_path
    operations.write_bytes(operations.read_bytes() + b"\n# unrelated reexport edit\n")
    assert evidence_script._source_provenance() == baseline

    for relative_path in tracked:
        owner = tmp_path / relative_path
        original = owner.read_bytes()
        owner.write_bytes(original + b"\n# provenance mutation\n")
        assert evidence_script._source_provenance() != baseline
        owner.write_bytes(original)
        assert evidence_script._source_provenance() == baseline


def test_operation_contract_snapshot_is_semantic_and_digest_checked(
    evidence_script,
) -> None:
    contract = evidence_script._operation_contract()
    snapshot = contract["snapshot"]
    encoded = json.dumps(
        snapshot,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")

    assert snapshot["operation_id"] == "label_connected_components"
    assert snapshot["implementation_id"] == "cupyx-connected-components-v1"
    assert snapshot["parity_policy_id"] == "labels-bitwise-int32-v1"
    assert snapshot["memory_model_id"] == "cupyx-connected-components-memory-v1"
    assert contract["sha256"] == hashlib.sha256(encoded).hexdigest()


def test_mask_generators_are_deterministic_readonly_and_reset_leading_ids(
    evidence_script,
) -> None:
    for pattern in ("sparse", "dense", "checkerboard"):
        first = evidence_script._make_mask((17, 19), pattern, 509, 2)
        second = evidence_script._make_mask((17, 19), pattern, 509, 2)
        np.testing.assert_array_equal(first, second)
        assert first.dtype == bool
        assert first.flags.c_contiguous
        assert not first.flags.writeable

    leading = evidence_script._make_mask((3, 17, 19), "leading-reset", 1, 2)
    assert [int(block.sum()) for block in leading] == [2, 2, 2]


def test_checked_in_artifact_is_historical_canonical_and_fully_covered(
    evidence_script,
) -> None:
    raw = ARTIFACT_PATH.read_text(encoding="utf-8")
    document = json.loads(raw)
    current_source_document = deepcopy(document)
    current_source_document["source_provenance"] = (
        evidence_script._source_provenance()
    )

    with pytest.raises(evidence_script.EvidenceError, match="fingerprints are stale"):
        evidence_script.validate_existing(ARTIFACT_PATH)
    evidence_script._validate_document_contract(current_source_document)

    assert raw == evidence_script._canonical_json(document)
    assert ARTIFACT_PATH.with_suffix(".md").read_text(encoding="utf-8") == (
        evidence_script.render_markdown(document)
    )
    assert document["profile"] == "full"
    assert document["admission"]["case_count"] == 16
    assert document["lifecycle"]["status"] == "pass"
    assert document["performance"]["case_count"] == 18
    assert document["performance"]["all_memory_estimates_cover_observed"] is True
    large = next(
        result
        for result in document["performance"]["results"]
        if result["shape"] == [64, 512, 512]
    )
    assert large["element_count"] == 16_777_216
    assert large["memory"]["estimated_peak_bytes"] == 192 * 1024**2


def test_existing_validation_is_cuda_safe_in_a_fresh_process() -> None:
    code = "\n".join(
        (
            "import builtins, runpy, sys",
            f"sys.path.insert(0, {str(PROJECT_ROOT / 'src')!r})",
            "real_import = builtins.__import__",
            "def guarded_import(name, *args, **kwargs):",
            "    if name == 'cupy' or name.startswith(('cupy.', 'cupyx')):",
            "        raise RuntimeError('validation imported CUDA libraries')",
            "    return real_import(name, *args, **kwargs)",
            "builtins.__import__ = guarded_import",
            "sys.argv = [sys.argv[1], '--validate-existing', sys.argv[2]]",
            "runpy.run_path(sys.argv[0], run_name='__main__')",
        )
    )
    completed = subprocess.run(
        [sys.executable, "-c", code, str(SCRIPT_PATH), str(ARTIFACT_PATH)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 2, completed.stderr
    assert "Source provenance fingerprints are stale" in completed.stderr
    assert "validation imported CUDA libraries" not in completed.stderr


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda document: document.update(unexpected=True),
            "root fields differ",
        ),
        (
            lambda document: document["source_provenance"][0].update(sha256="0" * 64),
            "fingerprints are stale",
        ),
        (
            lambda document: document["operation_contract"].update(sha256="0" * 64),
            "compute contract is stale",
        ),
        (
            lambda document: document["admission"]["cases"][0][
                "gpu_repeat_sha256"
            ].__setitem__(1, "0" * 64),
            "not exactly deterministic",
        ),
        (
            lambda document: document["performance"]["results"][0]["summary"].update(
                cpu_median_seconds=99.0
            ),
            "timing summary is inconsistent",
        ),
        (
            lambda document: document["performance"]["results"][0]["memory"].update(
                estimated_peak_bytes=1
            ),
            "memory evidence is inconsistent",
        ),
    ),
)
def test_validator_rejects_schema_provenance_parity_timing_and_memory_tampering(
    evidence_script,
    mutation,
    message,
) -> None:
    historical_document = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    document = deepcopy(historical_document)
    document["source_provenance"] = evidence_script._source_provenance()
    mutated = deepcopy(document)
    mutation(mutated)

    with pytest.raises(evidence_script.EvidenceError, match=message):
        evidence_script._validate_document_contract(mutated)


def test_validate_existing_rejects_noncanonical_json_and_edited_markdown(
    evidence_script,
    tmp_path: Path,
) -> None:
    historical_document = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    document = deepcopy(historical_document)
    document["source_provenance"] = evidence_script._source_provenance()
    output = tmp_path / "evidence.json"
    markdown = output.with_suffix(".md")
    evidence_script._atomic_write_artifacts(output, markdown, document)

    output.write_text(output.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(evidence_script.EvidenceError, match="canonical"):
        evidence_script.validate_existing(output)

    evidence_script._atomic_write_artifacts(output, markdown, document)
    markdown.write_text("edited", encoding="utf-8")
    with pytest.raises(evidence_script.EvidenceError, match="Markdown"):
        evidence_script.validate_existing(output)
