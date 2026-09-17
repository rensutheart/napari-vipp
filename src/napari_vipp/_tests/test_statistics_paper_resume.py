"""Resume must retain the exact execution, data and artifact identity."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def driver(monkeypatch):
    script = (
        Path(__file__).resolve().parents[3] / "scripts/reproduce_statistics_paper.py"
    )
    if not script.is_file():
        pytest.skip("Reproduction driver is available in the source checkout")
    specification = importlib.util.spec_from_file_location(
        "statistics_resume_test", script
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    # The command-line driver prepends its checkout; keep test imports isolated.
    import sys

    monkeypatch.setattr(sys, "path", list(sys.path))
    specification.loader.exec_module(module)
    return module


@pytest.fixture
def saved_field(tmp_path, monkeypatch, driver):
    out = tmp_path / "field"
    out.mkdir()
    reference = tmp_path / "reference"
    reference.mkdir()
    source = tmp_path / "source.tif"
    source.write_bytes(b"original acquisition")
    cells = out / "cells.csv"
    cells.write_text("ObjectNumber,Ratio\n1,0.5\n", encoding="utf-8")
    workflow = out / "workflow.json"
    workflow.write_text('{"setting": 15}', encoding="utf-8")
    reference_array = reference / "Nuclei_segmented.npy"
    reference_array.write_bytes(b"independent reference bytes")
    case = {
        "Case": "idr0139",
        "protein": "Fascin",
        "Treatment": "untreated",
        "paths": [str(source)],
    }
    recipe = {"nodes": [{"operation": "nuclei", "minimum_diameter": 15}]}
    contract = {
        "script_sha256": "driver-revision",
        "source_sha256": {"core/pipeline.py": "source-revision"},
        "versions": {"numpy": "reference-runtime"},
        "python": "reference-python",
    }
    monkeypatch.setattr(driver, "build_workflow", lambda *_args: (recipe, {}))
    monkeypatch.setattr(driver, "reference_directory", lambda *_args: reference)
    record = {
        "execution_contract": copy.deepcopy(contract),
        "case_fingerprint": driver.document_digest(case),
        "recipe_sha256": driver.document_digest(recipe),
        "reference_directory": str(reference),
        "reference_sha256": {reference_array.name: driver.digest(reference_array)},
        "artifact_sha256": {
            path.name: driver.digest(path) for path in (cells, workflow)
        },
        "sources": [{"path": str(source), "sha256": driver.digest(source)}],
        "executed_utc": "2026-09-17T00:00:00Z",
    }
    driver.write_json(out / "result.json", record)
    return SimpleNamespace(
        out=out,
        reference=reference,
        source=source,
        cells=cells,
        reference_array=reference_array,
        case=case,
        recipe=recipe,
        contract=contract,
        record=record,
        receipt=out / "result.json",
    )


def _reuse(driver, field):
    return driver.reusable_record(
        field.case, field.out, field.contract, field.reference.parent
    )


def test_unchanged_field_reuses_original_receipt(driver, saved_field):
    assert _reuse(driver, saved_field) == saved_field.record


def test_legacy_receipt_requires_fresh_execution(driver, saved_field):
    legacy = {"sources": saved_field.record["sources"], "vipp_count": 1}
    driver.write_json(saved_field.receipt, legacy)
    assert _reuse(driver, saved_field) is None


@pytest.mark.parametrize("changed_file", ["cells", "source", "reference_array"])
def test_changed_output_source_or_reference_rejects_resume(
    driver, saved_field, changed_file
):
    getattr(saved_field, changed_file).write_bytes(b"changed after execution")
    assert _reuse(driver, saved_field) is None


@pytest.mark.parametrize(
    "changed_file", ["cells", "source", "reference_array", "receipt"]
)
def test_missing_required_file_rejects_resume(driver, saved_field, changed_file):
    getattr(saved_field, changed_file).unlink()
    assert _reuse(driver, saved_field) is None


def test_changed_current_case_rejects_resume(driver, saved_field):
    saved_field.case["Treatment"] = "different treatment"
    assert _reuse(driver, saved_field) is None


def test_changed_current_recipe_rejects_resume(driver, saved_field):
    saved_field.recipe["nodes"][0]["minimum_diameter"] = 20
    assert _reuse(driver, saved_field) is None


@pytest.mark.parametrize(
    "component", ["script_sha256", "source_sha256", "versions", "python"]
)
def test_changed_execution_contract_rejects_resume(driver, saved_field, component):
    saved_field.contract[component] = "different execution identity"
    assert _reuse(driver, saved_field) is None


def test_changed_reference_location_rejects_resume(driver, saved_field, monkeypatch):
    monkeypatch.setattr(driver, "reference_directory", lambda *_args: None)
    assert _reuse(driver, saved_field) is None


def test_incomplete_receipt_rejects_resume(driver, saved_field):
    saved_field.receipt.write_text('{"sources":', encoding="utf-8")
    assert _reuse(driver, saved_field) is None


def test_receipt_with_no_reference_can_reuse_without_claiming_parity(
    driver, saved_field, monkeypatch
):
    monkeypatch.setattr(driver, "reference_directory", lambda *_args: None)
    saved_field.record["reference_directory"] = None
    saved_field.record["reference_sha256"] = {}
    saved_field.record["independent_cellprofiler_comparison"] = {}
    driver.write_json(saved_field.receipt, saved_field.record)
    result = _reuse(driver, saved_field)
    assert result == saved_field.record
    assert result["independent_cellprofiler_comparison"] == {}
    assert json.loads(saved_field.receipt.read_text()) == saved_field.record
