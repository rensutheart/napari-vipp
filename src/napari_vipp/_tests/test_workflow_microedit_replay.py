"""Replay loader fixtures only; no native window, inference or calculations."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def replay(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "scripts"))
    return importlib.import_module("replay_workflow_microedit_nomination")


@pytest.mark.parametrize(
    "identity,expected",
    [("candidate0", False), ("candidate1", True), ("candidate2", True)],
)
def test_original_baseline_is_never_a_duplicate_run(replay, identity, expected):
    assert replay.selection_replay_required(identity) is expected


@pytest.mark.parametrize("identity", ["latest", "candidate3", "", None])
def test_no_unrecorded_or_latest_substitution(replay, identity):
    with pytest.raises(ValueError, match="locked original completed"):
        replay.selection_replay_required(identity)


def test_loader_uses_original_file_source_not_detached_trial_defaults(
    replay, monkeypatch, tmp_path
):
    source = tmp_path / "approved-wrapper.ome.tif"
    source.write_bytes(b"synthetic wrapper")
    original = tmp_path / "original"
    original.mkdir()
    selection_root = tmp_path / "selection"
    selection_root.mkdir()
    recipe, evidence = {"nodes": []}, {"run_id": "actual"}
    roles, keymap = (
        [{"role": "band_mask", "node": "protected_structure", "port": 0}],
        {},
    )
    pin = replay.sha256(replay.compact({}))
    selection = {
        "candidate_id": "candidate2",
        "original_recorded_run": str(original),
        "original_inventory_sha256": pin,
        "recorded_outputs_only_no_final_native_replay": True,
        "recipe": recipe,
        "evidence": evidence,
        "role_mapping": roles,
        "key_to_node": keymap,
    }
    monkeypatch.setattr(replay, "ORIGINAL_INVENTORY_SHA256", pin)
    monkeypatch.setattr(
        replay, "verified_lock", lambda *_: (selection_root, {"selection": selection})
    )
    monkeypatch.setattr(replay, "run_inventory", lambda _: {})
    monkeypatch.setattr(replay, "read_recording_json", lambda _: {"ports": []})
    monkeypatch.setattr(
        replay, "recorded_port_bundle", lambda *_: SimpleNamespace(expected=[])
    )

    def read(path):
        path = str(path).replace("\\", "/")
        if path.endswith("proposal.json"):
            return recipe
        if path.endswith("candidate.json"):
            return evidence
        if path.endswith("output-roles.json"):
            return roles
        if path.endswith("key-to-node.json"):
            return keymap
        if path.endswith("baseline/completed-workflow/workflow.json"):
            return {"nodes": [{"id": "input", "params": {"file_path": str(source)}}]}
        if path.endswith("trials/trial-002/workflow.json"):
            return {
                "nodes": [{"id": "input", "params": {"file_path": ""}}],
                "executed": True,
            }
        if path.endswith("candidate2/workflow.json"):
            return {"nodes": [], "prepared": True}
        raise AssertionError(path)

    monkeypatch.setattr(replay, "read_json", read)
    result = replay.load_nomination(
        selection_root / "selected-result-lock.json", "lockhash"
    )
    assert result.source == source
    assert result.source_params["file_path"] == str(source)
    assert result.prepared["prepared"] and result.executed["executed"]
