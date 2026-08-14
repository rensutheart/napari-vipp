from __future__ import annotations

import copy
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from napari_vipp.core.progress import ProgressContext
from napari_vipp.core.richardson_lucy import richardson_lucy_tv_deconvolution
from scripts import benchmark_gpu_rl_tv_admission as evidence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = (
    PROJECT_ROOT / "docs" / "benchmarks" / "rl-tv-cupy-admission-windows-rtx5090.json"
)
IS_AMD64 = platform.machine().lower() in {"amd64", "x86_64"}
amd64_evidence_validation = pytest.mark.skipif(
    not IS_AMD64,
    reason=(
        "The committed admission artifact contains reviewed Windows/AMD64 "
        "raw-byte fixture digests."
    ),
)


def test_source_provenance_is_operation_owned() -> None:
    paths = tuple(evidence.SOURCE_PROVENANCE_PATHS)

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
    monkeypatch,
    tmp_path: Path,
) -> None:
    tracked = tuple(evidence.SOURCE_PROVENANCE_PATHS)
    unrelated = Path("src/napari_vipp/core/operations.py")
    for relative_path in (*tracked, str(unrelated)):
        destination = tmp_path / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((PROJECT_ROOT / relative_path).read_bytes())
    monkeypatch.setattr(evidence, "PROJECT_ROOT", tmp_path)
    baseline = evidence._source_provenance()

    unrelated_path = tmp_path / unrelated
    unrelated_path.write_bytes(unrelated_path.read_bytes() + b"\n# unrelated edit\n")
    assert evidence._source_provenance() == baseline

    for relative_path in tracked:
        owner = tmp_path / relative_path
        original = owner.read_bytes()
        owner.write_bytes(original + b"\n# owner edit\n")
        assert evidence._source_provenance() != baseline
        owner.write_bytes(original)


def test_import_is_safe_without_cupy_or_cuda_initialization() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(
            None,
            (
                str(PROJECT_ROOT / "src"),
                str(PROJECT_ROOT),
                environment.get("PYTHONPATH", ""),
            ),
        )
    )
    code = r"""
import builtins
import importlib
import sys

real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == "cupy" or name.startswith("cupyx"):
        raise AssertionError(f"optional CUDA import attempted: {name}")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
module = importlib.import_module("scripts.benchmark_gpu_rl_tv_admission")
assert module.benchmark_contract_digest()
assert "cupy" not in sys.modules
assert not any(name.startswith("cupyx") for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_help_is_cpu_safe() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "benchmark_gpu_rl_tv_admission.py"),
            "--help",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--validate-existing" in completed.stdout
    assert "Richardson--Lucy TV admission evidence" in completed.stdout


def test_fixture_contract_is_deterministic_and_independent() -> None:
    inherited = tuple(evidence._inherited_fixtures())
    first = tuple(evidence._holdout_fixtures())
    second = tuple(evidence._holdout_fixtures())

    evidence._require_fixture_contract(inherited=inherited, holdout=first)
    assert len(inherited) == 164
    assert len(first) == 96
    assert sum(item.spatial_rank == 2 for item in first) == 48
    assert sum(item.spatial_rank == 3 for item in first) == 48
    assert {item.family for item in first} == set(evidence.HOLDOUT_FAMILIES)
    assert evidence._fixture_manifest_digest(
        first
    ) == evidence._fixture_manifest_digest(second)
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left.image, right.image)
        np.testing.assert_array_equal(left.psf, right.psf)


def test_positive_reference_diagnostics_preserve_production_output() -> None:
    fixture = next(iter(evidence._holdout_fixtures()))
    parameters = {
        **evidence._common_parameters(fixture, 2),
        "tv_regularization": evidence.POSITIVE_REGULARIZATION,
        "tv_epsilon": evidence.POSITIVE_TV_EPSILON,
        "filter_epsilon": evidence.POSITIVE_FILTER_EPSILON,
        "denominator_floor": evidence.POSITIVE_DENOMINATOR_FLOOR,
    }
    expected = richardson_lucy_tv_deconvolution(
        [fixture.image, fixture.psf],
        progress=ProgressContext(),
        **parameters,
    )
    actual, diagnostics = evidence._reference_positive_diagnostics(
        fixture.image,
        fixture.psf,
        iterations=2,
    )

    np.testing.assert_array_equal(actual, expected)
    assert diagnostics["total_voxel_iterations"] == fixture.image.size * 2
    assert diagnostics["reference_threshold_active_samples"] >= 0
    assert diagnostics["reference_floor_active_samples"] >= 0
    assert np.isfinite(diagnostics["minimum_raw_denominator"])


def test_official_parity_retains_near_identity_as_diagnostic_only() -> None:
    reference = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)
    close = reference.copy()
    close[3, 4] += np.float32(2e-4)

    result = evidence._parity_record(
        reference,
        close,
        nrmse_limit=evidence.SCIENTIFIC_NRMSE_LIMIT,
        max_abs_base=evidence.SCIENTIFIC_MAX_ABS_BASE,
        max_abs_peak_factor=evidence.SCIENTIFIC_MAX_ABS_PEAK_FACTOR,
    )

    assert result["passed"] is True
    assert result["near_identity"] is False
    assert (
        result["near_identity_nrmse_limit"]
        == evidence.NEAR_IDENTITY_NRMSE_LIMIT
    )


def test_positive_admission_contract_names_only_measured_iterations() -> None:
    contract = evidence.BENCHMARK_CONTRACT

    assert evidence.SCHEMA_VERSION == 2
    assert contract["generator"] == "vipp-rl-tv-admission-v2"
    assert contract["lambda_zero_profile"]["iterations"] == [10, 25, 26, 50, 100]
    assert (
        contract["lambda_zero_profile"]["nrmse_limit"]
        == evidence.SCIENTIFIC_NRMSE_LIMIT
    )
    assert (
        contract["lambda_zero_profile"]["near_identity_diagnostic"]
        ["affects_admission"]
        is False
    )
    assert (
        contract["lambda_zero_profile"]["parity_policy_id"]
        == "rl-scientific-equivalence-v2"
    )
    assert contract["positive_default_profile"]["iterations"] == [10, 25]
    assert (
        contract["positive_default_profile"]["parity_policy_id"]
        == "rl-tv-scientific-equivalence-v2"
    )
    assert contract["parameter_region"]["positive_iterations"] == [10, 25]
    assert contract["parameter_region"]["lambda_zero_maximum_iterations"] == 100


def test_git_provenance_preserves_the_first_porcelain_path(monkeypatch) -> None:
    outputs = {
        ("rev-parse", "HEAD"): "a" * 40 + "\n",
        ("rev-parse", "--abbrev-ref", "HEAD"): "gpu-branch\n",
        ("status", "--short"): " M MANIFEST.in\n?? new-file.py\n",
    }

    def fake_run(command, **_kwargs):
        return SimpleNamespace(stdout=outputs[tuple(command[1:])])

    monkeypatch.setattr(evidence.subprocess, "run", fake_run)

    provenance = evidence._git_provenance()

    assert provenance["dirty_relative_paths"] == ["MANIFEST.in", "new-file.py"]


def test_privacy_validator_rejects_private_paths() -> None:
    with pytest.raises(evidence.EvidenceError, match="private path"):
        evidence._validate_privacy({"source": r"C:\Users\example\sample.nd2"})


@amd64_evidence_validation
def test_committed_historical_evidence_is_structurally_valid_and_renders() -> None:
    document = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    evidence.validate_evidence_document(document, require_current_sources=False)
    with pytest.raises(evidence.EvidenceError, match="archival, not current"):
        evidence.validate_evidence_document(document, require_current_sources=True)
    markdown = evidence.render_markdown(document)

    assert "development-branch public exposure" in markdown
    assert "Positive shipped default" in markdown
    assert "Maintained microscopy phantoms" in markdown
    assert "historical_aggressive_profile" not in markdown
    assert "cross-platform support" in markdown
    assert "released-package" in markdown
    conclusion = document["conclusion"]
    assert conclusion["development_branch_public_exposure_justified"] is True
    assert conclusion["cross_platform_promotion_justified"] is False
    assert conclusion["released_package_promotion_justified"] is False


@amd64_evidence_validation
def test_validation_requires_unambiguous_publication_scope() -> None:
    document = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    conclusion = document["conclusion"]
    conclusion.pop("public_promotion_justified", None)
    conclusion.update(evidence.PUBLICATION_SCOPE)

    evidence.validate_evidence_document(document, require_current_sources=False)

    conclusion["released_package_promotion_justified"] = True
    with pytest.raises(evidence.EvidenceError, match="released_package"):
        evidence.validate_evidence_document(document, require_current_sources=False)


@amd64_evidence_validation
def test_historical_validation_subprocess_does_not_require_cupy() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(
            None,
            (
                str(PROJECT_ROOT / "src"),
                str(PROJECT_ROOT),
                environment.get("PYTHONPATH", ""),
            ),
        )
    )
    code = rf"""
import builtins

real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == "cupy" or name.startswith("cupyx"):
        raise AssertionError(f"optional CUDA import attempted: {{name}}")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
import json
from scripts import benchmark_gpu_rl_tv_admission as evidence
document = json.loads(open({str(ARTIFACT)!r}, encoding="utf-8").read())
evidence.validate_evidence_document(document, require_current_sources=False)
print("historical RL-TV evidence is structurally valid")
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "historical RL-TV evidence is structurally valid" in completed.stdout


def _current_validation_document() -> dict[str, object]:
    def parity(*, nrmse: float = 1e-4, max_abs: float = 2e-4):
        cpu_peak = 2.0
        max_abs_limit = evidence.SCIENTIFIC_MAX_ABS_BASE + (
            evidence.SCIENTIFIC_MAX_ABS_PEAK_FACTOR * cpu_peak
        )
        return {
            "cpu_shape": [2, 3],
            "gpu_shape": [2, 3],
            "cpu_dtype": "float32",
            "gpu_dtype": "float32",
            "cpu_nonfinite_count": 0,
            "gpu_nonfinite_count": 0,
            "finite_mask_mismatch_count": 0,
            "cpu_negative_count": 0,
            "gpu_negative_count": 0,
            "shape_equal": True,
            "dtype_equal": True,
            "finite_masks_equal": True,
            "completely_finite": True,
            "nonnegative": True,
            "cpu_peak": cpu_peak,
            "nrmse": nrmse,
            "nrmse_limit": evidence.SCIENTIFIC_NRMSE_LIMIT,
            "max_abs": max_abs,
            "max_abs_limit": max_abs_limit,
            "gate_score": max(
                nrmse / evidence.SCIENTIFIC_NRMSE_LIMIT,
                max_abs / max_abs_limit,
            ),
            "near_identity": nrmse <= evidence.NEAR_IDENTITY_NRMSE_LIMIT,
            "near_identity_nrmse_limit": evidence.NEAR_IDENTITY_NRMSE_LIMIT,
            "passed": True,
        }

    zero_records = []
    for iteration_count in evidence.LAMBDA_ZERO_ITERATIONS:
        for index in range(260):
            zero_records.append(
                {
                    "fixture_id": f"zero-{iteration_count}-{index}",
                    "spatial_rank": 2,
                    "image_shape": [2, 3],
                    "iterations": iteration_count,
                    "parity": parity(),
                    "cpu_tv_equals_cpu_rl_bitwise": True,
                    "gpu_tv_equals_gpu_rl_bitwise": True,
                    "passed": True,
                }
            )

    positive_records = []
    for iteration_count in evidence.POSITIVE_ITERATIONS:
        for index in range(260):
            positive_records.append(
                {
                    "fixture_id": f"positive-{iteration_count}-{index}",
                    "spatial_rank": 2,
                    "image_shape": [2, 3],
                    "iterations": iteration_count,
                    "parity": parity(),
                    "reference_diagnostics": {
                        "reference_threshold_active_samples": 0,
                        "reference_floor_active_samples": 0,
                        "total_voxel_iterations": 6 * iteration_count,
                        "minimum_raw_denominator": 0.9,
                    },
                    "passed": True,
                }
            )
    return {
        "matrices": {
            "lambda_zero_scientific_equivalence": evidence._matrix_document(
                zero_records,
                iterations=evidence.LAMBDA_ZERO_ITERATIONS,
            ),
            "positive_shipped_default": evidence._matrix_document(
                positive_records,
                iterations=evidence.POSITIVE_ITERATIONS,
                diagnostics=True,
            ),
        }
    }


def test_v2_matrix_validation_rejects_tampered_numeric_gate() -> None:
    document = _current_validation_document()
    matrix = document["matrices"]["positive_shipped_default"]
    record = matrix["records"][0]
    record["parity"]["nrmse"] = 999.0

    with pytest.raises(evidence.EvidenceError, match="gate score"):
        evidence._validate_matrix(
            matrix,
            expected_records=520,
            iterations=evidence.POSITIVE_ITERATIONS,
            lambda_zero=False,
            nrmse_limit=evidence.SCIENTIFIC_NRMSE_LIMIT,
            max_abs_base=evidence.SCIENTIFIC_MAX_ABS_BASE,
            max_abs_peak_factor=evidence.SCIENTIFIC_MAX_ABS_PEAK_FACTOR,
            require_near_identity_diagnostic=True,
            require_raw_observations=True,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("gpu_dtype", "float64", "dtype_equal"),
        ("gpu_negative_count", 1, "nonnegative"),
        ("gpu_nonfinite_count", 1, "completely_finite"),
        ("gpu_shape", [3, 2], "unequal shapes"),
    ),
)
def test_v2_parity_validation_rejects_tampered_output_contract(
    field: str,
    value: object,
    message: str,
) -> None:
    parity = copy.deepcopy(
        _current_validation_document()["matrices"]
        ["lambda_zero_scientific_equivalence"]["records"][0]["parity"]
    )
    parity[field] = value

    with pytest.raises(evidence.EvidenceError, match=message):
        evidence._validate_parity(
            parity,
            nrmse_limit=evidence.SCIENTIFIC_NRMSE_LIMIT,
            max_abs_base=evidence.SCIENTIFIC_MAX_ABS_BASE,
            max_abs_peak_factor=evidence.SCIENTIFIC_MAX_ABS_PEAK_FACTOR,
            require_near_identity_diagnostic=True,
            require_raw_observations=True,
        )


def test_v2_matrix_validation_rejects_tampered_summary() -> None:
    document = _current_validation_document()
    matrix = document["matrices"]["lambda_zero_scientific_equivalence"]
    matrix["summaries"][0]["worst_nrmse"] = 0.0

    with pytest.raises(evidence.EvidenceError, match="summaries"):
        evidence._validate_matrix(
            matrix,
            expected_records=1300,
            iterations=evidence.LAMBDA_ZERO_ITERATIONS,
            lambda_zero=True,
            nrmse_limit=evidence.SCIENTIFIC_NRMSE_LIMIT,
            max_abs_base=evidence.SCIENTIFIC_MAX_ABS_BASE,
            max_abs_peak_factor=evidence.SCIENTIFIC_MAX_ABS_PEAK_FACTOR,
            require_near_identity_diagnostic=True,
            require_raw_observations=True,
        )
