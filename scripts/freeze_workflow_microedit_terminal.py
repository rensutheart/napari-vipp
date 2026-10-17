"""Freeze a stopped unranked experiment, NEVER select or infer a winner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from replay_workflow_investigation_selection import read_json, run_inventory
from workflow_investigation_evidence import (
    RAW_PIXELS_SHA256,
    EvidenceJournal,
    json_bytes,
    read_recording_json,
    sha256,
)
from workflow_microedit_evidence import recording_bytes

ORIGINAL_PIN = "a50b4aa261531af8e2d31cacf39e05e26d92a29fd315251cda46b28d5ee0ccec"
CONTINUATION_PIN = "ca2f178b809c086ce17f00c38afee48b281e1e536cceec392ffe83294f178a4f"


def inventory_hash(inventory):
    return sha256(json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode())


def verified_events(root):
    data = (root / "events.jsonl").read_bytes()
    events = [json.loads(line) for line in data.decode("utf-8").splitlines()]
    previous = "0" * 64
    for number, event in enumerate(events, 1):
        if event["sequence"] != number or event["previous_event_sha256"] != previous:
            raise ValueError("The terminal experiment event chain changed.")
        previous = sha256(json_bytes(event))
    if any(
        e["kind"] == "product_action" and e["details"]["action"]["action"] == "finish"
        for e in events
    ):
        raise ValueError("This terminal boundary must not contain a model FINISH.")
    return {
        "event_count": len(events),
        "last_event_sha256": previous,
        "events_file_sha256": sha256(data),
    }


def load_terminal(original, continuation):
    roots = [
        (original.resolve(strict=True), ORIGINAL_PIN),
        (continuation.resolve(strict=True), CONTINUATION_PIN),
    ]
    records = []
    for root, pin in roots:
        inventory = run_inventory(root)
        if (
            inventory_hash(inventory) != pin
            or (root / "selected-result-lock.json").exists()
        ):
            raise ValueError("The exact incomplete terminal recording changed.")
        records.append(
            {
                "root": str(root),
                "inventory": inventory,
                "inventory_sha256": pin,
                **verified_events(root),
            }
        )
    original, continuation = (Path(record["root"]) for record in records)
    first, last = (
        read_json(original / "receipt.json"),
        read_json(continuation / "receipt.json"),
    )
    if (
        first["selected_result_locked"] is not False
        or last["selected_result_locked"] is not False
        or last["finished"] is not False
        or last["cumulative_actions"] != 29
        or last["cumulative_model_calls"] != 29
        or last["cumulative_image_bytes"] != 629889
        or last["new_scientific_runs"] != 0
        or last["new_counts"]
        != {
            "actions": 6,
            "logical_requests": 6,
            "model_calls": 6,
            "trial_executions": 0,
        }
        or last["stop_reason"] != "Investigation request history exceeds 192 KiB."
    ):
        raise ValueError(
            "The exact stopped budget/absence-of-nomination boundary changed."
        )
    protocol = read_json(original / "protocol.json")
    if protocol["reference_material_read"] is not False:
        raise ValueError("References must remain hidden until terminal freezing.")
    candidates = []
    for number in (0, 1, 2):
        folder = (
            original / "session/initial/candidates/candidate0"
            if number == 0
            else original / f"actions/action-023/candidates/candidate{number}"
        )
        prefix = (
            "baseline/completed-workflow"
            if number == 0
            else f"trials/trial-{number:03d}"
        )
        full = original / prefix / "full-state.json"
        state = read_recording_json(full)
        evidence = read_json(folder / "candidate.json")
        roles = read_json(folder / "output-roles.json")
        mapping = read_json(folder / "key-to-node.json")
        if (
            evidence["status"] != "completed"
            or evidence["candidate_id"] != f"candidate{number}"
        ):
            raise ValueError(
                "Every compared candidate must have actual completed outputs."
            )
        ports = {(row["node_id"], row["port"]): row for row in state["ports"]}
        if len(ports) != len(state["ports"]):
            raise ValueError(
                "Duplicate raw output records cannot define scientific roles."
            )
        for role in roles:
            if (mapping[role["node"]], role["port"]) not in ports:
                raise ValueError("An explicit role lacks its actual persisted output.")
        source = ports[("input", 0)]
        if source["array"]["raw_sha256"] != RAW_PIXELS_SHA256:
            raise ValueError("The recorded source is not the approved image.")
        candidates.append(
            {
                "candidate_id": f"candidate{number}",
                "evidence": evidence,
                "recipe": read_json(folder / "proposal.json"),
                "key_to_node": mapping,
                "output_roles": roles,
                "full_state": {"path": str(full), "sha256": sha256(full.read_bytes())},
                "workflow": {
                    "path": str(original / prefix / "workflow.json"),
                    "sha256": sha256(
                        (original / prefix / "workflow.json").read_bytes()
                    ),
                },
                "recorded_ports": state["ports"],
                "source": source,
                "ranked_or_nominated": False,
            }
        )
    return {
        "kind": "napari-vipp-terminal-experiment-freeze",
        "version": 1,
        "terminal": True,
        "status": "incomplete-no-model-nomination",
        "no_nomination": True,
        "selected_candidate_id": None,
        "reference_material_read": False,
        "no_further_inference_in_this_experiment": True,
        "original_accepted_result_unchanged": True,
        "original_accepted_candidate_id": "candidate0",
        "comparison_rule": (
            "ALL recorded completed candidates, unranked; agreement is not truth"
        ),
        "recordings": records,
        "original_receipt": first,
        "continuation_receipt": last,
        "final_experiment_ledger": last["experiment_ledger"],
        "candidate2_assessment_status": "awaiting_assessment",
        "goal": protocol["goal"],
        "feedback": protocol["feedback"],
        "approved_raw_pixels_sha256": RAW_PIXELS_SHA256,
        "actual_model_calls": 29,
        "actual_actions": 29,
        "scientific_trials": 2,
        "actual_transmitted_image_bytes": 629889,
        "candidates": candidates,
        "current_native_ui_adoption_claimed": False,
        "script_sha256": sha256(Path(__file__).read_bytes()),
    }


def verify_terminal_freeze(path, expected_sha256):
    if sha256(path.read_bytes()) != expected_sha256:
        raise ValueError("The pinned terminal freeze changed.")
    frozen = read_recording_json(path)
    if (
        frozen.get("kind") != "napari-vipp-terminal-experiment-freeze"
        or frozen.get("terminal") is not True
        or frozen.get("no_nomination") is not True
        or frozen.get("selected_candidate_id") is not None
        or frozen.get("reference_material_read") is not False
        or frozen.get("no_further_inference_in_this_experiment") is not True
        or [c["candidate_id"] for c in frozen["candidates"]]
        != ["candidate0", "candidate1", "candidate2"]
    ):
        raise ValueError(
            "A terminal no-nomination boundary is required, not a fake selection."
        )
    actual = load_terminal(*(Path(record["root"]) for record in frozen["recordings"]))
    if recording_bytes(actual) != recording_bytes(frozen):
        raise ValueError("Terminal evidence differs from its actual frozen recordings.")
    return frozen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--continuation", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--freeze-approved", action="store_true")
    args = parser.parse_args(argv)
    frozen = load_terminal(args.original, args.continuation)
    if not args.freeze_approved:
        print(
            json.dumps(
                {
                    "terminal_verified": True,
                    "nomination": None,
                    "reference_material_read": False,
                }
            )
        )
        return 0
    if (
        args.output is None
        or args.output.exists()
        or any(
            args.output.resolve().is_relative_to(p.resolve())
            for p in (args.original, args.continuation)
        )
    ):
        raise ValueError("Terminal freezing requires a fresh separate evidence root.")
    journal = EvidenceJournal(args.output)
    journal.document("terminal-experiment-freeze.json", frozen)
    journal.event(
        "terminal_experiment_frozen_before_reference_comparison",
        no_nomination=True,
        candidate_ids=[c["candidate_id"] for c in frozen["candidates"]],
        no_further_inference=True,
    )
    print(
        json.dumps(
            {
                "path": str(args.output / "terminal-experiment-freeze.json"),
                "sha256": sha256(
                    (args.output / "terminal-experiment-freeze.json").read_bytes()
                ),
                "is_selected_result_lock": False,
                "nomination": None,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
