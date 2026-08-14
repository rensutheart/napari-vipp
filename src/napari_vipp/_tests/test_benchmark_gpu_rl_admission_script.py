from __future__ import annotations

import importlib.util
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_SCRIPT = PROJECT_ROOT / "scripts" / "benchmark_gpu_rl_admission.py"
EVIDENCE_PATH = (
    PROJECT_ROOT / "docs" / "benchmarks" / "rl-cupy-admission-windows-rtx5090.json"
)
SUMMARY_PATH = EVIDENCE_PATH.with_suffix(".md")
RL_TV_SUMMARY_PATH = (
    PROJECT_ROOT
    / "docs"
    / "benchmarks"
    / "rl-tv-cupy-admission-windows-rtx5090.md"
)


def _load_benchmark_module():
    name = "_napari_vipp_benchmark_gpu_rl_admission_test"
    spec = importlib.util.spec_from_file_location(name, BENCHMARK_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


benchmark_gpu_rl_admission = _load_benchmark_module()

IS_AMD64 = platform.machine().lower() in {"amd64", "x86_64"}


def _evidence() -> dict[str, object]:
    return json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))


def _summaries(document, suite_name):
    return {
        (item["filter_epsilon"], item["iterations"]): item
        for item in document["suites"][suite_name]["summaries"]
    }


def _synthetic_v2_evidence() -> dict[str, object]:
    def result(fixture_id, filter_epsilon, iterations):
        return {
            "fixture_id": fixture_id,
            "filter_epsilon": filter_epsilon,
            "iterations": iterations,
            "shape_equal": True,
            "dtype_equal": True,
            "finite_masks_equal": True,
            "completely_finite": True,
            "cpu_nonnegative": True,
            "gpu_nonnegative": True,
            "cpu_peak": 1.0,
            "nrmse": 0.004,
            "max_abs": 0.004,
            "max_abs_limit": 0.005001,
            "gate_score": 0.8,
            "passed": True,
            "max_ulp": 1,
            "legacy_v1_max_abs_limit": 6e-6,
            "legacy_v1_gate_score": 2000.0,
            "legacy_v1_gate_passed": False,
            "near_identity_nrmse": 0.004,
            "near_identity_passed": False,
        }

    def suite(fixture_count, filter_epsilons, iterations):
        results = [
            result(f"synthetic-{fixture_index}", epsilon, iteration_count)
            for epsilon in filter_epsilons
            for iteration_count in iterations
            for fixture_index in range(fixture_count)
        ]
        return {
            "fixture_count": fixture_count,
            "fixture_manifest_sha256": "0" * 64,
            "filter_epsilons": list(filter_epsilons),
            "iterations": list(iterations),
            "result_count": len(results),
            "results": results,
            "summaries": benchmark_gpu_rl_admission._derive_suite_summaries(
                results,
                filter_epsilons=filter_epsilons,
                iterations=iterations,
            ),
        }

    suites = {
        "default_epsilon_checkpoints_164": suite(
            164,
            (1e-12,),
            (10, 25, 26, 50, 100),
        ),
        "legacy_branch_characterization_164": suite(
            164,
            (1e-8, 1e-7, 1e-6),
            (10, 25, 50),
        ),
        "legacy_low_epsilon_characterization_36": suite(
            36,
            (1e-10,),
            (10, 25, 50, 100),
        ),
        "legacy_even_psf_characterization_40": suite(
            40,
            (1e-8, 1e-7, 1e-6),
            (5, 10, 25),
        ),
    }
    return {
        "schema": benchmark_gpu_rl_admission.EVIDENCE_SCHEMA,
        "schema_version": benchmark_gpu_rl_admission.EVIDENCE_SCHEMA_VERSION,
        "status": "complete",
        "contract": benchmark_gpu_rl_admission.BENCHMARK_CONTRACT,
        "contract_sha256": benchmark_gpu_rl_admission.benchmark_contract_digest(),
        "source_provenance": benchmark_gpu_rl_admission._source_provenance(),
        "parity_gate": {
            "policy_id": benchmark_gpu_rl_admission.PARITY_POLICY_ID,
            "scope": "CPU/GPU backend agreement only",
            "scientific_validity_claimed": False,
            "nonnegative_outputs_required": True,
            "nrmse_limit": benchmark_gpu_rl_admission.NRMSE_LIMIT,
            "max_abs_floor": benchmark_gpu_rl_admission.MAX_ABSOLUTE_FLOOR,
            "max_abs_peak_factor": (
                benchmark_gpu_rl_admission.MAX_ABSOLUTE_PEAK_FACTOR
            ),
        },
        "suites": suites,
        "conclusion": benchmark_gpu_rl_admission._derive_conclusion(suites),
    }


def test_source_provenance_is_operation_owned() -> None:
    paths = tuple(benchmark_gpu_rl_admission.SOURCE_PROVENANCE_PATHS)

    assert paths == (
        "scripts/benchmark_gpu_rl_admission.py",
        "src/napari_vipp/core/richardson_lucy.py",
        "src/napari_vipp/core/richardson_lucy_compute.py",
        "src/napari_vipp/core/richardson_lucy_parity.py",
        "src/napari_vipp/core/gpu/cupy_rl.py",
        "src/napari_vipp/core/progress.py",
    )
    assert "src/napari_vipp/core/operations.py" not in paths
    assert "src/napari_vipp/core/compute_specs.py" not in paths
    assert "src/napari_vipp/core/compute_policy.py" not in paths
    assert "src/napari_vipp/core/compute_benchmark_adapter.py" not in paths


def test_source_provenance_tracks_each_owner_but_ignores_shared_registries(
    monkeypatch,
    tmp_path: Path,
) -> None:
    tracked = tuple(benchmark_gpu_rl_admission.SOURCE_PROVENANCE_PATHS)
    unrelated_paths = (
        "src/napari_vipp/core/operations.py",
        "src/napari_vipp/core/compute_specs.py",
        "src/napari_vipp/core/compute_policy.py",
        "src/napari_vipp/core/compute_benchmark_adapter.py",
    )
    for relative_path in (*tracked, *unrelated_paths):
        destination = tmp_path / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((PROJECT_ROOT / relative_path).read_bytes())
    monkeypatch.setattr(benchmark_gpu_rl_admission, "PROJECT_ROOT", tmp_path)
    baseline = benchmark_gpu_rl_admission._source_provenance(project_root=tmp_path)

    for relative_path in unrelated_paths:
        unrelated = tmp_path / relative_path
        unrelated.write_bytes(unrelated.read_bytes() + b"\n# unrelated edit\n")
    assert (
        benchmark_gpu_rl_admission._source_provenance(project_root=tmp_path) == baseline
    )

    for relative_path in tracked:
        owner = tmp_path / relative_path
        original = owner.read_bytes()
        owner.write_bytes(original + b"\n# owner edit\n")
        assert (
            benchmark_gpu_rl_admission._source_provenance(project_root=tmp_path)
            != baseline
        )
        owner.write_bytes(original)


def test_help_does_not_import_or_probe_cupy(monkeypatch, capsys):
    def unexpected_load():
        raise AssertionError("--help unexpectedly loaded CuPy")

    monkeypatch.setattr(benchmark_gpu_rl_admission, "_load_cupy", unexpected_load)

    with pytest.raises(SystemExit) as stopped:
        benchmark_gpu_rl_admission.main(["--help"])

    assert stopped.value.code == 0
    output = capsys.readouterr().out
    assert "--validate-existing" in output
    assert "--device-index" in output


def test_fixture_contract_is_deterministic_and_preserves_reviewed_counts():
    gaussian = tuple(benchmark_gpu_rl_admission._gaussian_core_fixtures())
    asymmetric = tuple(benchmark_gpu_rl_admission._odd_asymmetric_fixtures())
    sparse = tuple(benchmark_gpu_rl_admission._sparse_seed_sweep_fixtures())
    even = tuple(benchmark_gpu_rl_admission._even_psf_comparison_fixtures())
    final = gaussian + asymmetric + sparse

    benchmark_gpu_rl_admission._require_fixture_contract(
        final_odd=final,
        gaussian_core=gaussian,
        odd_asymmetric=asymmetric,
        sparse_sweep=sparse,
        even_comparison=even,
    )

    assert (len(gaussian), len(asymmetric), len(sparse), len(final), len(even)) == (
        36,
        48,
        80,
        164,
        40,
    )
    if not IS_AMD64:
        pytest.skip(
            "Reviewed raw-byte fixture digests are Windows/AMD64 evidence; "
            "fixture structure is still checked on this architecture."
        )
    assert benchmark_gpu_rl_admission._fixture_manifest_digest(final) == (
        "3238340a510f933ebb0e5e12839e2bd4600b533f1c0326cd638cdd2ff6291641"
    )
    assert benchmark_gpu_rl_admission._fixture_manifest_digest(gaussian) == (
        "d2a4b96a5c58c8c1cd10bf8de3137ae12b4793e4b45ed2a4798dda8788fe26ce"
    )
    assert benchmark_gpu_rl_admission._fixture_manifest_digest(even) == (
        "54cec35b037f3de98829b6fc18e8646c0209d2afde9b2d6325aec898bebd8865"
    )


def test_v2_contract_adds_default_epsilon_checkpoints_through_100():
    contract = benchmark_gpu_rl_admission.BENCHMARK_CONTRACT

    assert benchmark_gpu_rl_admission.EVIDENCE_SCHEMA_VERSION == 2
    assert benchmark_gpu_rl_admission.PARITY_POLICY_ID == (
        "rl-scientific-equivalence-v2"
    )
    assert benchmark_gpu_rl_admission.NRMSE_LIMIT == 0.005
    assert benchmark_gpu_rl_admission.MAX_ABSOLUTE_PEAK_FACTOR == 0.005
    assert contract["reviewed_admission_filter_epsilon_range"] == [1e-12, 1e-6]
    assert contract["reviewed_admission_maximum_iterations"] == 100
    assert contract["default_epsilon_checkpoints_164"] == {
        "fixture_source": "final_odd_164",
        "filter_epsilons": [1e-12],
        "iterations": [10, 25, 26, 50, 100],
        "purpose": (
            "admission checkpoints for the authored CPU default through the "
            "expanded 100-iteration boundary"
        ),
    }
    assert "legacy_branch_characterization_164" in contract
    assert "legacy_low_epsilon_characterization_36" in contract
    assert "legacy_even_psf_characterization_40" in contract


def test_v2_gate_is_backend_agreement_and_v1_is_diagnostic_only():
    fixture = benchmark_gpu_rl_admission._Fixture(
        fixture_id="unit",
        group_id="unit",
        family="unit",
        image=np.ones((2, 2), dtype=np.float32),
        psf=np.ones((1, 1), dtype=np.float32),
    )
    expected = np.ones((2, 2), dtype=np.float32)
    actual = np.full((2, 2), 1.004, dtype=np.float32)

    record = benchmark_gpu_rl_admission._parity_record(
        fixture,
        expected,
        actual,
        filter_epsilon=1e-12,
        iterations=100,
    )

    assert record["passed"] is True
    assert record["legacy_v1_gate_passed"] is False
    assert record["near_identity_passed"] is False
    assert record["cpu_nonnegative"] is True
    assert record["gpu_nonnegative"] is True
    assert record["nrmse"] == pytest.approx(0.004, rel=1e-4)
    assert record["max_abs_limit"] == pytest.approx(0.005001)
    assert record["max_ulp"] > 0


def test_v2_evidence_validation_and_markdown_state_scope_clearly():
    document = _synthetic_v2_evidence()

    benchmark_gpu_rl_admission.validate_evidence_document(
        document,
        require_current_sources=True,
    )
    conclusion = document["conclusion"]
    assert conclusion["admitted_filter_epsilon_minimum"] == 1e-12
    assert conclusion["admitted_filter_epsilon_maximum"] == 1e-6
    assert conclusion["admitted_maximum_iterations"] == 100
    assert conclusion["default_epsilon_checkpoints"] == [10, 25, 26, 50, 100]
    assert conclusion["all_sampled_odd_psf_conditions_passed_v2"] is True
    assert conclusion["filter_epsilon_continuum_exhaustively_sampled"] is False
    assert conclusion["legacy_v1_branch_sensitivity_observed"] is True
    assert conclusion["agreement_scope"] == "CPU/GPU backend agreement only"
    assert conclusion["scientific_validity_claimed"] is False

    document["generated_at_utc"] = "2026-08-14T00:00:00+00:00"
    document["environment"] = {
        "cuda": {"device_name": "unit device"},
        "platform": "unit platform",
    }
    markdown = benchmark_gpu_rl_admission.render_markdown(document)
    assert "0.5%" in markdown
    assert "backend agreement" in markdown
    assert "scientifically valid" in markdown
    assert "diagnostics only" in markdown


def test_v2_validation_rejects_a_failed_odd_psf_checkpoint():
    document = _synthetic_v2_evidence()
    summaries = document["suites"]["legacy_branch_characterization_164"]["summaries"]
    summaries[0]["failure_count"] = 1

    with pytest.raises(
        benchmark_gpu_rl_admission.AdmissionEvidenceError,
        match="does not match raw results",
    ):
        benchmark_gpu_rl_admission.validate_evidence_document(
            document,
            require_current_sources=False,
        )


@pytest.mark.parametrize("metric", ["nrmse", "max_abs"])
def test_v2_validation_recomputes_gate_metrics_from_raw_results(metric):
    document = _synthetic_v2_evidence()
    result = document["suites"]["default_epsilon_checkpoints_164"]["results"][0]
    result[metric] = 1e9

    with pytest.raises(
        benchmark_gpu_rl_admission.AdmissionEvidenceError,
        match="stale or inconsistent",
    ):
        benchmark_gpu_rl_admission.validate_evidence_document(
            document,
            require_current_sources=False,
        )


def test_v2_validation_does_not_trust_stored_pass_boolean_or_summary():
    document = _synthetic_v2_evidence()
    suite = document["suites"]["default_epsilon_checkpoints_164"]
    result = suite["results"][0]
    result["gpu_nonnegative"] = False
    result["passed"] = False
    suite["summaries"][0]["failure_count"] = 1

    with pytest.raises(
        benchmark_gpu_rl_admission.AdmissionEvidenceError,
        match="checkpoint-backed odd-PSF envelope",
    ):
        benchmark_gpu_rl_admission.validate_evidence_document(
            document,
            require_current_sources=False,
        )


def test_v2_validation_recomputes_every_summary_from_raw_results():
    document = _synthetic_v2_evidence()
    summary = document["suites"]["default_epsilon_checkpoints_164"]["summaries"][0]
    summary["worst_gate_score"] = 0.0

    with pytest.raises(
        benchmark_gpu_rl_admission.AdmissionEvidenceError,
        match="does not match raw results",
    ):
        benchmark_gpu_rl_admission.validate_evidence_document(
            document,
            require_current_sources=False,
        )


def test_committed_v1_artifact_remains_identified_as_historical():
    document = _evidence()

    assert document["schema_version"] == 1
    assert document["parity_gate"]["policy_id"] == "rl-float32-tolerance-v1"
    assert SUMMARY_PATH.exists()
    with pytest.raises(
        benchmark_gpu_rl_admission.AdmissionEvidenceError,
        match="schema version",
    ):
        benchmark_gpu_rl_admission.validate_evidence_document(
            document,
            require_current_sources=False,
        )


def test_committed_readable_summaries_are_unmistakably_historical():
    rl_summary = SUMMARY_PATH.read_text(encoding="utf-8")
    rl_tv_summary = RL_TV_SUMMARY_PATH.read_text(encoding="utf-8")

    for summary in (rl_summary, rl_tv_summary):
        assert "HISTORICAL EVIDENCE" in summary
        assert "not been regenerated" in summary
        assert "current-policy admission evidence" in summary
    assert "uses CPU fallback" not in rl_summary
    assert "1e-12" in rl_summary
    assert "0.5%" in rl_summary


def test_validation_requires_unambiguous_publication_scope():
    document = _synthetic_v2_evidence()
    conclusion = document["conclusion"]

    benchmark_gpu_rl_admission.validate_evidence_document(
        document,
        require_current_sources=False,
    )

    conclusion["cross_platform_promotion_justified"] = True
    with pytest.raises(
        benchmark_gpu_rl_admission.AdmissionEvidenceError,
        match="cross_platform_promotion_justified",
    ):
        benchmark_gpu_rl_admission.validate_evidence_document(
            document,
            require_current_sources=False,
        )


def test_validate_existing_is_cpu_safe(monkeypatch, capsys, tmp_path: Path):
    def unexpected_load():
        raise AssertionError("validation unexpectedly loaded CuPy")

    monkeypatch.setattr(benchmark_gpu_rl_admission, "_load_cupy", unexpected_load)
    evidence_path = tmp_path / "v2-evidence.json"
    evidence_path.write_text(json.dumps(_synthetic_v2_evidence()), encoding="utf-8")

    assert (
        benchmark_gpu_rl_admission.main(["--validate-existing", str(evidence_path)])
        == 0
    )
    assert "is current" in capsys.readouterr().out
