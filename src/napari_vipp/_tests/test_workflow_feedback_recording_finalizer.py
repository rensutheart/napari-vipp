"""Limited artifact indexing must never invent missing completion evidence."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def finalizer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "scripts"))
    return importlib.import_module("finalize_workflow_feedback_recording")


def test_buffer_index_uses_exact_model_identity_and_preserves_nan_payload(
    finalizer, tmp_path
):
    outputs = tmp_path / "final-workflow/outputs"
    outputs.mkdir(parents=True)
    data = np.array([1, np.nan], dtype="float64")
    np.save(outputs / "owned-port-0.npy", data, allow_pickle=False)
    np.save(outputs / "owned_other-port-0.npy", np.zeros(2), allow_pickle=False)
    before = (outputs / "owned-port-0.npy").read_bytes()
    indexed = finalizer.collect_node_buffers(tmp_path, {"model_choice": "owned"})
    assert len(indexed) == 1 and len(indexed[0]["files"]) == 1
    assert indexed[0]["model_key"] == "model_choice"
    assert indexed[0]["complete_output_state_available"] is False
    assert indexed[0]["files"][0]["raw_c_order_sha256"] == finalizer.sha256(
        data.tobytes()
    )
    assert (outputs / "owned-port-0.npy").read_bytes() == before
    finalizer.json_bytes(indexed)


def _bundle(finalizer, tmp_path):
    root = tmp_path / "original"
    root.mkdir()
    return {
        "root": root,
        "inventory": {},
        "inventory_sha256": finalizer.inventory_hash({}),
        "event_chain_sha256": "recorded-chain",
        "deadline_utc": finalizer.time.time() + 60,
        "recipe": {"nodes": []},
        "mapping": {"chosen": "recorded-node"},
        "buffers": [
            {
                "model_key": "chosen",
                "node_id": "recorded-node",
                "files": [{"path": "recorded.npy", "file_sha256": "recorded-hash"}],
                "complete_output_state_available": False,
            }
        ],
        "dispatch": {"run_id": "owned-recorded-run"},
        "report": {"summary": "Recorded interpretation"},
        "request": {"source": {"grid": "captured, not per-output metadata"}},
        "failure": {"status": "incomplete"},
        "model_calls_recorded": 3,
        "guards": {"calls": 0},
        "source_buffer": {"raw_c_order_sha256": "source-recorded"},
        "qualification": {"wheel_sha256": "qualified-recorded"},
    }


def test_limited_descriptor_explicitly_denies_full_mechanical_pass(finalizer, tmp_path):
    bundle = _bundle(finalizer, tmp_path)
    finalizer.write_descriptor(bundle, tmp_path / "limited")
    descriptor = json.loads(
        (tmp_path / "limited/limited-recording.json").read_text("utf-8")
    )
    lock = json.loads(
        (tmp_path / "limited/selected-result-lock.json").read_text("utf-8")
    )
    assert descriptor["mechanical_acceptance_pass"] is False
    assert descriptor["scientific_accuracy_established"] is False
    assert descriptor["new_model_calls"] == descriptor["new_scientific_runs"] == 0
    assert lock["selection"]["mechanical_acceptance_pass"] is False
    assert "not a model-ranked winner" in lock["selection"]["selection_basis"]
    assert any("No final full-state" in text for text in descriptor["limitations"])
    assert any("MeshObject" in text for text in descriptor["limitations"])
    assert any("UI adoption" in text for text in descriptor["limitations"])
    assert finalizer.run_inventory(bundle["root"]) == {}


def test_limited_write_rejects_expired_original_deadline(finalizer, tmp_path):
    bundle = _bundle(finalizer, tmp_path)
    bundle["deadline_utc"] = finalizer.time.time() - 1
    with pytest.raises(ValueError, match="deadline expired"):
        finalizer.write_descriptor(bundle, tmp_path / "limited")
    assert not (tmp_path / "limited").exists()


def test_limited_write_rejects_changed_original_inventory(finalizer, tmp_path):
    bundle = _bundle(finalizer, tmp_path)
    (bundle["root"] / "unexpected.txt").write_text("mutated", encoding="utf-8")
    with pytest.raises(ValueError, match="recorded run changed"):
        finalizer.write_descriptor(bundle, tmp_path / "limited")
    assert not (tmp_path / "limited/selected-result-lock.json").exists()


def _events(finalizer, root, extra_calls=0, late=False):
    records = []

    def add(kind, **details):
        event = {
            "sequence": len(records) + 1,
            "utc": "2026-10-16T19:00:00+00:00",
            "kind": kind,
            "details": details,
            "previous_event_sha256": finalizer.sha256(finalizer.json_bytes(records[-1]))
            if records
            else "0" * 64,
        }
        records.append(event)

    for phase in ("baseline", "refinement"):
        add("normal_product_add", phase=phase)
        add("normal_product_run", phase=phase, accepted=True)
    add("explicit_one_round_start")
    for index, kind in enumerate(("assessment", "revision", "assessment"), 1):
        add(
            "product_model_request",
            request_kind=kind,
            prefix=f"requests/request-{index:03d}",
        )
        add("actual_model_call", prefix=f"transport/call-{index:03d}")
        add(
            "product_model_response",
            request_kind=kind,
            prefix=f"requests/request-{index:03d}",
        )
    for index in range(extra_calls):
        add("actual_model_call", prefix=f"transport/extra-{index}")
    add(
        "acceptance_failed",
        message="The owned normal run failed or exceeded 300 seconds.",
    )
    if late:
        records[-2]["utc"] = "2026-10-16T19:31:00+00:00"
        records[-1]["previous_event_sha256"] = finalizer.sha256(
            finalizer.json_bytes(records[-2])
        )
    (root / "events.jsonl").write_text(
        "\n".join(json.dumps(item) for item in records), encoding="utf-8"
    )


def test_recorded_phase_and_actual_turn_bounds(finalizer, tmp_path):
    _events(finalizer, tmp_path)
    assert len(finalizer.verify_events(tmp_path)[0]) == 15
    _events(finalizer, tmp_path, extra_calls=4)
    with pytest.raises(ValueError, match="native turn bound"):
        finalizer.verify_events(tmp_path)


def test_recorded_model_response_after_deadline_is_rejected(finalizer, tmp_path):
    _events(finalizer, tmp_path, late=True)
    with pytest.raises(ValueError, match="original bounded deadline"):
        finalizer.verify_events(tmp_path)
