"""The standalone acceptance runner must use real package code and known motion."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _smoke_module():
    spec = importlib.util.spec_from_file_location(
        "tracking_install_smoke", ROOT / "scripts" / "smoke_tracking_install.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_smoke_checks_tracking_and_both_sides_of_previous_frame_anchor():
    result = _smoke_module().smoke_tracking_install(expected_package_root=ROOT / "src")
    assert result["status"] == "passed"
    assert result["installed_distribution_verified"] is False
    assert [
        item["expected_and_found_observation_count"] for item in result["cases"]
    ] == [24, 14]
    registration = result["previous_frame_registration"]
    assert registration["anchor"] == 2
    assert registration["frames_before_anchor"] == 2
    assert registration["frames_after_anchor"] == 2
    assert registration["maximum_matrix_absolute_error"] <= 1e-12
    assert registration["maximum_valid_alignment_absolute_error"] <= 1e-12
    assert registration["source_unchanged"]


def test_source_smoke_rejects_a_wrong_expected_import_root(tmp_path):
    with pytest.raises(RuntimeError, match="Wrong installed package"):
        _smoke_module().smoke_tracking_install(expected_package_root=tmp_path)
