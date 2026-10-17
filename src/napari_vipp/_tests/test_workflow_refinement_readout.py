"""Synthetic own-output readout checks; no live run or user source reads."""

import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def reader(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "scripts"))
    return importlib.import_module("readout_workflow_refinement")


def test_default_prepares_only_without_reading_any_run(reader, monkeypatch):
    monkeypatch.setattr(reader, "load_terminal", lambda *_: pytest.fail("No run reads"))
    assert reader.main(["--run", "not-an-existing-run", "not-a-lock"]) == 0


def test_changes_separate_protected_and_other_voxels_without_quality_claim(reader):
    old = np.zeros((1, 3, 4), bool)
    old[0, 1, :3] = True
    new = old.copy()
    new[0, 1, 0] = False
    new[0, 2, 3] = True
    protected = np.zeros_like(old)
    protected[0, 1, 0] = True
    result = reader.mask_changes(old, new, protected)
    assert result["added_foreground"] == result["removed_foreground"] == 1
    assert result["protected_missing"] == 1
    assert result["removed_outside_exact_protected_voxels"] == 0
    assert result["added_outside_exact_protected_voxels"] == 1
    assert "not accuracy" in result["meaning"]


def test_vertical_extent_includes_gaps_and_excludes_empty_population_shift(reader):
    old = np.zeros((1, 7, 3), bool)
    old[0, [1, 5], 0] = True
    old[0, 2:4, 1] = True
    new = old.copy()
    new[0, 5, 0] = False
    new[0, :, 1] = False
    new[0, 2:6, 2] = True
    protected = np.zeros_like(old)
    result = reader.vertical_span_proxy(old, new, protected, 0.45)
    paired = result["paired_population"]["all_paired_columns"]
    assert paired["column_count"] == 1
    assert paired["baseline_mean_cells"] == 5
    assert paired["candidate_mean_cells"] == 1
    assert paired["mean_delta_micrometers"] == pytest.approx(-1.8)
    assert result["lost_nonempty_columns"] == result["new_nonempty_columns"] == 1
    assert "Not perpendicular" in result["not"]


def test_protected_foreground_excludes_whole_columns_from_exterior_proxy(reader):
    old = np.ones((1, 3, 2), bool)
    protected = np.zeros_like(old)
    protected[0, 1, 0] = True
    result = reader.vertical_span_proxy(old, old, protected, 1.0)
    assert result["paired_population"]["all_paired_columns"]["column_count"] == 2
    assert (
        result["paired_population"]["paired_columns_without_any_protected_voxel"][
            "column_count"
        ]
        == 1
    )


def test_empty_proxy_reports_missing_not_zero_as_observed_extent(reader):
    data = np.zeros((1, 3, 2), bool)
    result = reader.vertical_span_proxy(data, data, data, 1.0)
    assert (
        result["paired_population"]["all_paired_columns"]["candidate_mean_cells"]
        is None
    )


def test_raw_edge_check_closed_open_and_nonmanifold_are_distinct(reader):
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    assert reader.edge_check(vertices, faces)["boundary_edge_count"] == 0
    assert reader.edge_check(vertices, faces[:-1])["boundary_edge_count"] == 3
    duplicated = np.concatenate([faces, faces[:1]])
    result = reader.edge_check(vertices, duplicated)
    assert result["boundary_edge_count"] == 0
    assert result["nonmanifold_edge_count"] == 3
    assert "no solid" in result["meaning"]


def test_array_manifest_is_exact_and_readonly(reader, tmp_path):
    from workflow_investigation_evidence import EvidenceJournal

    journal = EvidenceJournal(tmp_path / "recorded")
    data = np.arange(12, dtype=np.uint8).reshape(1, 3, 4)
    manifest = journal.array("data.npy", data)
    result = reader.read_array(journal.directory, manifest)
    assert np.array_equal(result, data) and not result.flags.writeable
    manifest["raw_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="raw bytes"):
        reader.read_array(journal.directory, manifest)


def test_nonterminal_recording_rejected_before_arrays(reader, tmp_path, monkeypatch):
    from workflow_investigation_evidence import EvidenceJournal

    journal = EvidenceJournal(tmp_path / "recorded")
    journal.document("receipt.json", {"selected_result_locked": True})
    journal.document("session/terminal.json", {"finished": False})
    journal.document("protocol.json", {"reference_material_read": False})
    journal.lock_selection(
        {
            "candidate_id": "candidate0",
            "run_id": "synthetic",
            "role_mapping": [{"role": "band_mask"}],
        }
    )
    digest = reader.sha256(
        (journal.directory / "selected-result-lock.json").read_bytes()
    )
    monkeypatch.setattr(
        reader, "read_array", lambda *_: pytest.fail("No partial arrays")
    )
    with pytest.raises(ValueError, match="genuinely terminal"):
        reader.load_terminal(journal.directory, digest)


def test_grid_unit_spelling_only_alias_not_rescale(reader):
    def state(unit):
        return {
            "shape": [1, 3, 4],
            "axes": [
                {
                    "name": a,
                    "type": "space",
                    "unit": unit,
                    "scale": 0.45,
                    "translation": 0.0,
                }
                for a in "zyx"
            ],
        }

    assert reader.grid(state("micrometer")) == reader.grid(state("µm"))
    with pytest.raises(ValueError, match="ambiguous"):
        reader.grid(state("unknown"))


def test_signed_negative_or_float_intensity_not_silently_cast_to_mask(reader):
    with pytest.raises(ValueError):
        reader.foreground(np.array([[[-1, 0, 1]]]))
    with pytest.raises(ValueError):
        reader.foreground(np.zeros((1, 2, 3), float))


def test_actual_buffer_shape_must_match_carried_grid(reader):
    state = {
        "shape": [1, 3, 4],
        "axes": [
            {
                "name": a,
                "type": "space",
                "unit": "µm",
                "scale": 0.45,
                "translation": 0.0,
            }
            for a in "zyx"
        ],
    }
    reader.array_grid(np.zeros((1, 3, 4), bool), state)
    with pytest.raises(ValueError, match="actual buffer"):
        reader.array_grid(np.zeros((1, 4, 3), bool), state)
    state["shape"][0] = True
    with pytest.raises(ValueError, match="not explicit"):
        reader.grid(state)


def test_protected_foreground_must_belong_to_actual_accepted_band(reader):
    accepted = np.zeros((1, 3, 4), bool)
    accepted[0, 1, :3] = True
    protected = np.zeros_like(accepted)
    protected[0, 1, 0] = True
    reader.assert_protected_baseline(accepted, protected)
    protected[0, 1, 3] = True
    with pytest.raises(ValueError, match="accepted band"):
        reader.assert_protected_baseline(accepted, protected)


def test_actual_mesh_state_spatial_axes_matches_image_grid_without_conversion(reader):
    image = {
        "shape": [1, 3, 4],
        "axes": [
            {
                "name": a,
                "type": "space",
                "unit": "µm",
                "scale": 0.45,
                "translation": 0.0,
            }
            for a in "zyx"
        ],
    }
    mesh = {"source_shape": image["shape"], "spatial_axes": image["axes"]}
    assert reader.grid(mesh) == reader.grid(image)
    mesh["spatial_axes"] = [dict(row) for row in image["axes"]]
    mesh["spatial_axes"][1]["scale"] = 0.5
    assert reader.grid(mesh) != reader.grid(image)
