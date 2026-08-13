from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_gpu_admission.py"
MANIFEST_PATH = PROJECT_ROOT / "scripts" / "gpu_admission_suites.json"


@pytest.fixture(scope="module")
def harness():
    module_name = "_vipp_test_gpu_admission_harness"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(module_name, None)


@pytest.mark.parametrize("action", ("import", "--help", "--check", "--list"))
def test_cpu_safe_surfaces_do_not_import_optional_gpu_packages(action):
    guarded = """
import builtins, runpy, sys
real_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'cupy' or name.startswith(('cupy.', 'cupyx', 'cucim')):
        raise RuntimeError('CPU-safe harness surface imported a GPU package')
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
"""
    if action == "import":
        body = f"runpy.run_path({str(SCRIPT_PATH)!r}, run_name='not_main')"
    else:
        body = (
            f"sys.argv = [{str(SCRIPT_PATH)!r}, {action!r}]; "
            f"runpy.run_path({str(SCRIPT_PATH)!r}, run_name='__main__')"
        )
    completed = subprocess.run(
        [sys.executable, "-c", guarded + body],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "imported a GPU package" not in completed.stderr


def test_checked_in_manifest_maps_every_public_declaration_and_facet(harness):
    declarations = harness.public_accelerator_declarations()
    manifest = harness.load_suite_manifest(
        MANIFEST_PATH,
        declarations=declarations,
        project_root=PROJECT_ROOT,
    )
    owners = harness._facet_owner_map(manifest.runners)

    assert len(declarations) == 14
    assert {item.key for item in manifest.implementations} == {
        item.key for item in declarations
    }
    for declaration in declarations:
        assert set(owners[declaration.key]) == set(harness.REQUIRED_FACETS)
        assert all(owners[declaration.key].values())


def test_convert_dtype_runs_focused_evidence_and_contract_owners_in_both_profiles(
    harness,
):
    manifest = harness.load_suite_manifest(
        MANIFEST_PATH,
        declarations=harness.public_accelerator_declarations(),
        project_root=PROJECT_ROOT,
    )
    implementation = "convert_dtype::cupyx-convert-dtype-preserve-f32-v1"
    runners = {
        runner.runner_id: runner
        for runner in manifest.runners
        if implementation in runner.implementations
    }

    assert {
        "convert-dtype-evidence",
        "convert-dtype-provider-contracts",
        "public-fallback-policy-contracts",
    } <= set(runners)
    evidence = runners["convert-dtype-evidence"]
    contracts = runners["convert-dtype-provider-contracts"]
    assert set(evidence.facets) == set(harness.REQUIRED_FACETS)
    assert "scripts/benchmark_gpu_convert_dtype.py" in evidence.profile_commands[
        "quick"
    ]
    assert "scripts/benchmark_gpu_convert_dtype.py" in evidence.profile_commands[
        "full"
    ]
    for profile in harness.PROFILES:
        assert (
            "src/napari_vipp/_tests/test_gpu_convert_dtype_provider.py"
            in contracts.profile_commands[profile]
        )


def test_manifest_rejects_an_unmapped_public_declaration(harness, tmp_path):
    document = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    document["implementations"].pop()
    path = tmp_path / "unmapped.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(harness.AdmissionHarnessError, match="does not exactly map"):
        harness.load_suite_manifest(
            path,
            declarations=harness.public_accelerator_declarations(),
            project_root=PROJECT_ROOT,
        )


def test_manifest_rejects_a_required_facet_without_an_executable_owner(
    harness, tmp_path
):
    document = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    policy = next(
        runner
        for runner in document["runners"]
        if runner["id"] == "public-fallback-policy-contracts"
    )
    policy["implementations"].remove(
        "rolling_ball_background::cucim-rolling_ball_background-v2"
    )
    path = tmp_path / "missing-facet.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(harness.AdmissionHarnessError, match="unmapped required facets"):
        harness.load_suite_manifest(
            path,
            declarations=harness.public_accelerator_declarations(),
            project_root=PROJECT_ROOT,
        )


def test_manifest_rejects_a_stale_executable_declaration(harness, tmp_path):
    document = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    document["implementations"][0]["implementation_version"] = "999"
    path = tmp_path / "stale.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(harness.AdmissionHarnessError, match="is stale"):
        harness.load_suite_manifest(
            path,
            declarations=harness.public_accelerator_declarations(),
            project_root=PROJECT_ROOT,
        )


def test_manifest_rejects_a_facet_owner_not_executed_by_its_runner(
    harness, tmp_path
):
    document = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    runner = next(
        item
        for item in document["runners"]
        if item["id"] == "canny-otsu-provider-contracts"
    )
    for command in runner["profile_commands"].values():
        command.remove("src/napari_vipp/_tests/test_gpu_otsu.py")
    path = tmp_path / "unowned.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(harness.AdmissionHarnessError, match="does not execute"):
        harness.load_suite_manifest(
            path,
            declarations=harness.public_accelerator_declarations(),
            project_root=PROJECT_ROOT,
        )


@pytest.mark.parametrize("profile", ("quick", "full"))
def test_profile_writes_one_aggregate_only_after_evidence_passes(
    harness, monkeypatch, tmp_path, profile
):
    declaration = harness.AcceleratorDeclaration("op", "gpu-op-v1", "1", "cuda", "lib")
    runner = harness.RunnerSpec(
        runner_id="fake-evidence",
        kind="evidence",
        implementations=(declaration.key,),
        facets=harness.REQUIRED_FACETS,
        owner_paths=("fake.py",),
        profile_commands={
            "quick": ("{python}", "fake.py", "--output", "{artifact}"),
            "full": ("{python}", "fake.py", "--output", "{artifact}"),
        },
        artifact="fake.json",
        artifact_schema="fake-evidence",
        artifact_schema_version=1,
    )
    manifest = harness.SuiteManifest(
        path=tmp_path / "manifest.json",
        sha256="a" * 64,
        implementations=(declaration,),
        runners=(runner,),
    )

    def completed(command, **_kwargs):
        artifact = Path(command[-1])
        artifact.write_text(
            json.dumps({"schema": "fake-evidence", "schema_version": 1}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "passed\n", "")

    monkeypatch.setattr(harness.subprocess, "run", completed)
    monkeypatch.setattr(
        harness,
        "_git_identity",
        lambda _root: {
            "git_available": True,
            "commit": "b" * 40,
            "worktree_dirty": False,
        },
    )
    output = tmp_path / f"{profile}.json"

    written = harness.run_profile(
        manifest,
        profile=profile,
        output=output,
        artifacts=tmp_path / f"{profile}-artifacts",
        device_index=3,
        project_root=PROJECT_ROOT,
    )
    document = json.loads(written.read_text(encoding="utf-8"))

    assert document["status"] == "pass"
    assert document["profile"] == profile
    assert document["device_selection"] == {"device_id": "cuda:3", "device_index": 3}
    assert document["runners"][0]["artifact"]["schema"] == "fake-evidence"
    assert "<python>" in document["runners"][0]["command"]
    serialized = written.read_text(encoding="utf-8")
    assert str(PROJECT_ROOT) not in serialized
    assert str(tmp_path) not in serialized


def test_pytest_owner_cannot_pass_by_skipping_real_gpu_checks(
    harness, monkeypatch, tmp_path
):
    declaration = harness.AcceleratorDeclaration("op", "gpu-op-v1", "1", "cuda", "lib")
    runner = harness.RunnerSpec(
        runner_id="fake-pytest",
        kind="pytest",
        implementations=(declaration.key,),
        facets=harness.REQUIRED_FACETS,
        owner_paths=("test_fake.py",),
        profile_commands={
            "quick": ("{python}", "-m", "pytest", "test_fake.py"),
            "full": ("{python}", "-m", "pytest", "test_fake.py"),
        },
        artifact=None,
        artifact_schema=None,
        artifact_schema_version=None,
    )
    manifest = harness.SuiteManifest(
        path=tmp_path / "manifest.json",
        sha256="a" * 64,
        implementations=(declaration,),
        runners=(runner,),
    )
    monkeypatch.setattr(
        harness.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 0, "1 passed, 1 skipped in 0.01s\n", ""
        ),
    )
    output = tmp_path / "aggregate.json"

    with pytest.raises(harness.AdmissionHarnessError, match="skipped 1 tests"):
        harness.run_profile(
            manifest,
            profile="quick",
            output=output,
            artifacts=tmp_path / "artifacts",
            device_index=0,
            project_root=PROJECT_ROOT,
        )
    assert not output.exists()


def test_profile_rejects_an_existing_aggregate_before_running(
    harness, monkeypatch, tmp_path
):
    declaration = harness.AcceleratorDeclaration("op", "gpu-op-v1", "1", "cuda", "lib")
    runner = harness.RunnerSpec(
        runner_id="fake-pytest",
        kind="pytest",
        implementations=(declaration.key,),
        facets=harness.REQUIRED_FACETS,
        owner_paths=("test_fake.py",),
        profile_commands={
            "quick": ("{python}", "-m", "pytest", "test_fake.py"),
            "full": ("{python}", "-m", "pytest", "test_fake.py"),
        },
        artifact=None,
        artifact_schema=None,
        artifact_schema_version=None,
    )
    manifest = harness.SuiteManifest(
        path=tmp_path / "manifest.json",
        sha256="a" * 64,
        implementations=(declaration,),
        runners=(runner,),
    )
    output = tmp_path / "aggregate.json"
    output.write_text("old evidence\n", encoding="utf-8")
    called = False

    def completed(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("runner should not start")

    monkeypatch.setattr(harness.subprocess, "run", completed)

    with pytest.raises(harness.AdmissionHarnessError, match="fresh path"):
        harness.run_profile(
            manifest,
            profile="quick",
            output=output,
            artifacts=tmp_path / "artifacts",
            device_index=0,
            project_root=PROJECT_ROOT,
        )
    assert not called
    assert output.read_text(encoding="utf-8") == "old evidence\n"


def test_atomic_aggregate_write_removes_temporary_file_on_replace_failure(
    harness, monkeypatch, tmp_path
):
    output = tmp_path / "aggregate.json"

    def fail_replace(_source, _destination):
        raise OSError("replace failed")

    monkeypatch.setattr(harness.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        harness._atomic_write_json(output, {"status": "pass"})

    assert not output.exists()
    assert list(tmp_path.iterdir()) == []
