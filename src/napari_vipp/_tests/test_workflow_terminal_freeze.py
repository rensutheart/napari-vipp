"""Terminal no-nomination integrity fixtures, never scientific/reference data."""

import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def freeze(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "scripts"))
    return importlib.import_module("freeze_workflow_microedit_terminal")


def test_event_chain_and_explicit_finish_cannot_be_relabelled_terminal(
    freeze, tmp_path
):
    event = {
        "sequence": 1,
        "previous_event_sha256": "0" * 64,
        "kind": "product_action",
        "details": {"action": {"action": "inspect_candidate"}},
    }
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps(event) + "\n", encoding="utf-8")
    assert freeze.verified_events(tmp_path)["event_count"] == 1
    event["details"]["action"]["action"] = "finish"
    path.write_text(json.dumps(event) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must not contain a model FINISH"):
        freeze.verified_events(tmp_path)
    event["sequence"] = 2
    path.write_text(json.dumps(event) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="event chain"):
        freeze.verified_events(tmp_path)


def test_terminal_artifact_is_not_a_nomination(freeze, monkeypatch, tmp_path):
    document = {
        "kind": "napari-vipp-terminal-experiment-freeze",
        "terminal": True,
        "no_nomination": True,
        "selected_candidate_id": None,
        "reference_material_read": False,
        "no_further_inference_in_this_experiment": True,
        "candidates": [{"candidate_id": f"candidate{i}"} for i in range(3)],
        "recordings": [{"root": "one"}, {"root": "two"}],
    }
    path = tmp_path / "terminal.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(freeze, "load_terminal", lambda *_: document)
    assert (
        freeze.verify_terminal_freeze(path, freeze.sha256(path.read_bytes()))
        == document
    )
    changed = {**document, "selected_candidate_id": "candidate0"}
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="not a fake selection"):
        freeze.verify_terminal_freeze(path, freeze.sha256(path.read_bytes()))


def test_artifact_hash_precedes_any_original_or_reference_access(
    freeze, monkeypatch, tmp_path
):
    path = tmp_path / "terminal.json"
    path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        freeze, "load_terminal", lambda *_: pytest.fail("opened recordings")
    )
    with pytest.raises(ValueError, match="pinned terminal freeze changed"):
        freeze.verify_terminal_freeze(path, "0" * 64)
