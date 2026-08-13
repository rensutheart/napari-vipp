from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from napari_vipp.core.operations import convert_dtype as cpu_convert_dtype

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "benchmark_gpu_convert_dtype.py"


@pytest.fixture(scope="module")
def evidence_script():
    module_name = "_vipp_test_gpu_convert_dtype_evidence"
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


def test_admission_manifest_covers_the_exact_public_region(evidence_script):
    cases = evidence_script._admission_cases()
    coverage = {
        item
        for case in cases
        for item in case.coverage
    } | {"repeat:deterministic"}

    assert [case.case_id for case in cases] == [
        "u8-empty-1d",
        "u8-boundaries-1d",
        "u8-pattern-2d",
        "u16-strided-2d",
        "u16-transposed-3d",
    ]
    assert evidence_script.REQUIRED_ADMISSION_COVERAGE <= coverage
    assert all(
        str(evidence_script._host_case(case.kind).dtype) in {"uint8", "uint16"}
        for case in cases
    )


def test_quick_and_full_profiles_execute_declared_timing_cases(evidence_script):
    quick = evidence_script._performance_cases("quick")
    full = evidence_script._performance_cases("full")

    assert quick
    assert tuple(full[: len(quick)]) == quick
    assert len(full) > len(quick)
    assert {case.dtype for case in quick} == {"uint8", "uint16"}
    assert all(case.shape and all(size > 0 for size in case.shape) for case in full)


def test_cpu_metadata_and_fallback_facets_are_executable(evidence_script):
    metadata = evidence_script._run_metadata(cpu_convert_dtype)
    fallback = evidence_script._run_fallback(cpu_convert_dtype)

    assert metadata["status"] == "pass"
    assert metadata["dtype_updated_to_float32"] is True
    assert metadata["axes_preserved"] is True
    assert fallback["status"] == "pass"
    assert fallback["safe_cpu_fallback_case_count"] == 3
    assert fallback["invalid_authored_conversion_fails_closed"] is True
    assert all(case["fallback_allowed"] is True for case in fallback["cases"])


def test_operation_fingerprint_names_the_public_implementation(evidence_script):
    contract = evidence_script._operation_contract()

    assert len(contract["sha256"]) == 64
    assert contract["snapshot"]["operation_id"] == "convert_dtype"
    assert (
        contract["snapshot"]["implementation_id"]
        == "cupyx-convert-dtype-preserve-f32-v1"
    )
    assert contract["snapshot"]["memory_model_id"] == "cupy-convert-dtype-memory-v1"
