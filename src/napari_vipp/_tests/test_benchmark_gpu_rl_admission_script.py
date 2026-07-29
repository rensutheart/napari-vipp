from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_SCRIPT = PROJECT_ROOT / "scripts" / "benchmark_gpu_rl_admission.py"
EVIDENCE_PATH = (
    PROJECT_ROOT / "docs" / "benchmarks" / "rl-cupy-admission-windows-rtx5090.json"
)
SUMMARY_PATH = EVIDENCE_PATH.with_suffix(".md")


def _load_benchmark_module():
    name = "_napari_vipp_benchmark_gpu_rl_admission_test"
    spec = importlib.util.spec_from_file_location(name, BENCHMARK_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


benchmark_gpu_rl_admission = _load_benchmark_module()


def _evidence() -> dict[str, object]:
    return json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))


def _summaries(document, suite_name):
    return {
        (item["filter_epsilon"], item["iterations"]): item
        for item in document["suites"][suite_name]["summaries"]
    }


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
    assert benchmark_gpu_rl_admission._fixture_manifest_digest(final) == (
        "3238340a510f933ebb0e5e12839e2bd4600b533f1c0326cd638cdd2ff6291641"
    )
    assert benchmark_gpu_rl_admission._fixture_manifest_digest(gaussian) == (
        "d2a4b96a5c58c8c1cd10bf8de3137ae12b4793e4b45ed2a4798dda8788fe26ce"
    )
    assert benchmark_gpu_rl_admission._fixture_manifest_digest(even) == (
        "54cec35b037f3de98829b6fc18e8646c0209d2afde9b2d6325aec898bebd8865"
    )


def test_committed_evidence_is_current_and_supports_only_the_exact_value():
    document = _evidence()

    benchmark_gpu_rl_admission.validate_evidence_document(
        document,
        require_current_sources=True,
    )

    conclusion = document["conclusion"]
    assert conclusion["required_filter_epsilon"] == 1e-8
    assert conclusion["recommended_maximum_iterations"] == 25
    assert conclusion["require_all_psf_extents_odd"] is True
    assert conclusion["higher_filter_epsilon_is_not_monotone"] is True
    assert conclusion["development_branch_public_exposure_justified"] is True
    assert conclusion["cross_platform_promotion_justified"] is False
    assert conclusion["released_package_promotion_justified"] is False

    final = _summaries(document, "final_odd_164")
    assert final[(1e-8, 10)]["failure_count"] == 0
    assert final[(1e-8, 25)]["failure_count"] == 0
    assert final[(1e-8, 25)]["worst_gate_score"] == pytest.approx(0.8643477385028496)
    assert final[(1e-8, 50)]["failure_count"] == 4
    assert final[(1e-7, 25)]["failure_count"] == 1
    assert final[(1e-6, 10)]["failure_count"] == 1

    provisional = _summaries(document, "provisional_floor_rejection_36")
    assert provisional[(1e-10, 25)]["failure_count"] == 1
    assert provisional[(1e-10, 100)]["failure_count"] == 7

    even = _summaries(document, "even_psf_comparison_40")
    assert even[(1e-8, 5)]["failure_count"] == 6
    assert even[(1e-8, 25)]["failure_count"] == 14


def test_committed_markdown_is_rendered_from_the_json_artifact():
    markdown = benchmark_gpu_rl_admission.render_markdown(_evidence())

    assert SUMMARY_PATH.read_text(encoding="utf-8") == markdown
    assert "public exposure on this development branch" in markdown
    assert "cross-platform" in markdown
    assert "released-package promotion" in markdown


def test_validation_requires_unambiguous_publication_scope():
    document = _evidence()
    conclusion = document["conclusion"]
    conclusion.pop("public_exposure", None)
    conclusion.update(benchmark_gpu_rl_admission.PUBLICATION_SCOPE)

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


def test_validate_existing_is_cpu_safe(monkeypatch, capsys):
    def unexpected_load():
        raise AssertionError("validation unexpectedly loaded CuPy")

    monkeypatch.setattr(benchmark_gpu_rl_admission, "_load_cupy", unexpected_load)

    assert (
        benchmark_gpu_rl_admission.main(["--validate-existing", str(EVIDENCE_PATH)])
        == 0
    )
    assert "is current" in capsys.readouterr().out
