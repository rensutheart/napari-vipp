from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_SCRIPT = PROJECT_ROOT / "scripts" / "benchmark_gpu_phase1.py"
EVIDENCE_PATH = (
    PROJECT_ROOT
    / "docs"
    / "benchmarks"
    / "phase1-production-node-benchmark-windows-rtx5090.json"
)


def _load_benchmark_module():
    name = "_napari_vipp_benchmark_gpu_phase1_test"
    spec = importlib.util.spec_from_file_location(name, BENCHMARK_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


benchmark_gpu_phase1 = _load_benchmark_module()


def test_platform_provenance_uses_only_a_privacy_safe_executable_name(monkeypatch):
    monkeypatch.setattr(
        benchmark_gpu_phase1.sys,
        "executable",
        r"C:\Users\researcher\private-worktree\.venv\Scripts\python.exe",
    )
    environment = SimpleNamespace(
        python_abi="cpython-312",
        execution_mode="native",
    )

    provenance = benchmark_gpu_phase1._platform_provenance(environment)

    assert provenance["executable"] == "python.exe"


def test_canonical_evidence_binds_the_privacy_safe_current_generator():
    document = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))

    assert document["platform"]["executable"] == "python.exe"
    assert document["source_provenance"]["files"][
        "scripts/benchmark_gpu_phase1.py"
    ] == hashlib.sha256(BENCHMARK_SCRIPT.read_bytes()).hexdigest()


def test_help_does_not_start_a_gpu_benchmark(monkeypatch, capsys):
    def unexpected_run(*, device_id):
        raise AssertionError(f"benchmark unexpectedly ran for {device_id!r}")

    monkeypatch.setattr(benchmark_gpu_phase1, "run_benchmarks", unexpected_run)

    with pytest.raises(SystemExit) as stopped:
        benchmark_gpu_phase1.main(["--help"])

    assert stopped.value.code == 0
    assert "--output" in capsys.readouterr().out


def test_main_writes_fake_evidence_without_gpu(monkeypatch, tmp_path, capsys):
    document = {
        "schema": benchmark_gpu_phase1.EVIDENCE_SCHEMA,
        "schema_version": benchmark_gpu_phase1.EVIDENCE_SCHEMA_VERSION,
        "results": [],
    }
    received = []

    def fake_run(*, device_id):
        received.append(device_id)
        return document

    monkeypatch.setattr(benchmark_gpu_phase1, "run_benchmarks", fake_run)
    output = tmp_path / "nested" / "evidence.json"

    result = benchmark_gpu_phase1.main(
        ["--output", str(output), "--device-id", "cuda:7"]
    )

    assert result == 0
    assert received == ["cuda:7"]
    assert json.loads(output.read_text(encoding="utf-8")) == document
    assert output.resolve().as_posix() in capsys.readouterr().out.replace("\\", "/")
    assert tuple(output.parent.iterdir()) == (output,)


def test_atomic_writer_serializes_strict_json_before_touching_output(tmp_path):
    output = tmp_path / "evidence.json"
    output.write_text("previous evidence\n", encoding="utf-8")

    with pytest.raises(
        benchmark_gpu_phase1.BenchmarkEvidenceError,
        match="strict JSON",
    ):
        benchmark_gpu_phase1._atomic_write_json(output, {"invalid": float("nan")})

    assert output.read_text(encoding="utf-8") == "previous evidence\n"
    assert tuple(tmp_path.iterdir()) == (output,)


def test_atomic_writer_confines_temporary_file_to_selected_parent(tmp_path):
    output = tmp_path / "chosen" / "phase1.json"
    written = benchmark_gpu_phase1._atomic_write_json(
        output,
        {"z": [3, 2, 1], "a": {"finite": 1.25}},
    )

    assert written == output.resolve()
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "a": {"finite": 1.25},
        "z": [3, 2, 1],
    }
    assert tuple(output.parent.iterdir()) == (output,)
    assert tuple(tmp_path.iterdir()) == (output.parent,)


def test_production_profile_requires_the_loaded_bootstrap_seed():
    workload = SimpleNamespace(operation_id="gaussian_blur")
    request = SimpleNamespace(
        workload=workload,
        benchmark_policy_id="production-policy",
        warm_rounds=7,
        max_warm_rounds=21,
        adaptive_rounds=True,
        time_parity_as_cold=True,
        warmup_rounds=1,
        paired_bootstrap_samples=2000,
        paired_bootstrap_seed=17029,
        paired_confidence_level=0.95,
        time_budget_seconds=None,
    )
    policy = SimpleNamespace(
        initial_warm_rounds=7,
        adaptive_warm_rounds=(15, 21),
        bootstrap_resamples=2000,
        bootstrap_seed=17029,
        confidence_level=0.95,
    )

    benchmark_gpu_phase1._require_production_profile(
        request,
        policy,
        policy_id="production-policy",
    )

    request.paired_bootstrap_seed = 99
    with pytest.raises(
        benchmark_gpu_phase1.BenchmarkEvidenceError,
        match="production benchmark sampling profile",
    ):
        benchmark_gpu_phase1._require_production_profile(
            request,
            policy,
            policy_id="production-policy",
        )


def test_directory_output_fails_without_creating_a_temporary_file(tmp_path):
    with pytest.raises(
        benchmark_gpu_phase1.BenchmarkEvidenceError,
        match="directory",
    ):
        benchmark_gpu_phase1._atomic_write_json(tmp_path, {"valid": True})

    assert tuple(tmp_path.iterdir()) == ()


def test_complete_samples_require_matching_production_round_target():
    built = SimpleNamespace(
        request=SimpleNamespace(
            reference=SimpleNamespace(implementation_id="cpu-gaussian_blur-v1")
        )
    )
    spec = SimpleNamespace(implementation_id="cupyx-gaussian-blur-v1")

    def record(cpu_rounds, gpu_rounds):
        return SimpleNamespace(
            candidates=(
                SimpleNamespace(
                    implementation_id="cpu-gaussian_blur-v1",
                    warm_seconds=(0.1,) * cpu_rounds,
                ),
                SimpleNamespace(
                    implementation_id="cupyx-gaussian-blur-v1",
                    warm_seconds=(0.05,) * gpu_rounds,
                ),
            )
        )

    assert (
        benchmark_gpu_phase1._require_complete_production_samples(
            record(7, 7),
            built,
            spec,
            production_warm_rounds=(7, 15, 21),
        )
        == 7
    )
    with pytest.raises(
        benchmark_gpu_phase1.BenchmarkEvidenceError,
        match="different warm-round counts",
    ):
        benchmark_gpu_phase1._require_complete_production_samples(
            record(7, 15),
            built,
            spec,
            production_warm_rounds=(7, 15, 21),
        )
    with pytest.raises(
        benchmark_gpu_phase1.BenchmarkEvidenceError,
        match="expected one of",
    ):
        benchmark_gpu_phase1._require_complete_production_samples(
            record(3, 3),
            built,
            spec,
            production_warm_rounds=(7, 15, 21),
        )


def test_source_provenance_keeps_hashes_when_git_is_unavailable(
    monkeypatch,
    tmp_path,
):
    source = tmp_path / "source.py"
    source.write_bytes(b"print('identified')\n")

    def missing_git(*args, **kwargs):
        del args, kwargs
        raise FileNotFoundError("git is unavailable")

    monkeypatch.setattr(benchmark_gpu_phase1.subprocess, "run", missing_git)
    provenance = benchmark_gpu_phase1._source_provenance(
        project_root=tmp_path,
        source_paths=("source.py",),
    )

    assert provenance["hash_algorithm"] == "sha256"
    assert provenance["files"] == {
        "source.py": hashlib.sha256(source.read_bytes()).hexdigest()
    }
    assert provenance["git"]["available"] is False
    assert "git is unavailable" in provenance["git"]["reason"]


def test_source_provenance_rejects_a_missing_required_file(tmp_path):
    with pytest.raises(
        benchmark_gpu_phase1.BenchmarkEvidenceError,
        match="Could not fingerprint required source",
    ):
        benchmark_gpu_phase1._source_provenance(
            project_root=tmp_path,
            source_paths=("missing.py",),
        )
