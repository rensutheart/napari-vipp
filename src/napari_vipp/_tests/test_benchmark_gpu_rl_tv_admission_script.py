from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from napari_vipp.core.operations import richardson_lucy_tv_deconvolution
from napari_vipp.core.progress import ProgressContext
from scripts import benchmark_gpu_rl_tv_admission as evidence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = (
    PROJECT_ROOT / "docs" / "benchmarks" / "rl-tv-cupy-admission-windows-rtx5090.json"
)


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


def test_versioned_parity_profiles_are_distinct() -> None:
    reference = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)
    close = reference.copy()
    close[3, 4] += np.float32(2e-4)

    strict = evidence._parity_record(
        reference,
        close,
        nrmse_limit=evidence.STRICT_NRMSE_LIMIT,
        max_abs_base=evidence.STRICT_MAX_ABS_BASE,
        max_abs_peak_factor=evidence.STRICT_MAX_ABS_PEAK_FACTOR,
    )
    positive = evidence._parity_record(
        reference,
        close,
        nrmse_limit=evidence.POSITIVE_NRMSE_LIMIT,
        max_abs_base=evidence.POSITIVE_MAX_ABS_BASE,
        max_abs_peak_factor=evidence.POSITIVE_MAX_ABS_PEAK_FACTOR,
    )

    assert strict["passed"] is False
    assert positive["passed"] is True


def test_positive_admission_contract_names_only_measured_iterations() -> None:
    contract = evidence.BENCHMARK_CONTRACT

    assert contract["positive_default_profile"]["iterations"] == [10, 25]
    assert contract["parameter_region"]["positive_iterations"] == [10, 25]
    assert contract["parameter_region"]["lambda_zero_maximum_iterations"] == 25


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


def test_committed_evidence_is_current_and_renders() -> None:
    document = json.loads(ARTIFACT.read_text(encoding="utf-8"))

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


def test_validation_requires_unambiguous_publication_scope() -> None:
    document = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    conclusion = document["conclusion"]
    conclusion.pop("public_promotion_justified", None)
    conclusion.update(evidence.PUBLICATION_SCOPE)

    evidence.validate_evidence_document(document, require_current_sources=False)

    conclusion["released_package_promotion_justified"] = True
    with pytest.raises(evidence.EvidenceError, match="released_package"):
        evidence.validate_evidence_document(document, require_current_sources=False)


def test_validate_existing_subprocess_does_not_require_cupy() -> None:
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
from scripts import benchmark_gpu_rl_tv_admission as evidence
raise SystemExit(evidence.main(["--validate-existing", {str(ARTIFACT)!r}]))
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
    assert "is current" in completed.stdout
