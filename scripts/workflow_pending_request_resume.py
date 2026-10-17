"""Read-only provenance for resuming an unsent first refinement request.

This is not cached-response replay and grants no new phase, model or Run budget.
The original failed directory remains immutable; no original provider payload
can be claimed identical because the observer failed before recording one.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from time import monotonic

from replay_workflow_investigation_selection import read_json, run_inventory
from workflow_investigation_evidence import json_bytes, sha256


def strict_public_json(value):
    """Only JSON-equivalent representation changes; reject nonfinite values."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def anchor_continuation_loop(loop, continuation):
    """Called on the UI thread before ordinary settings/worker construction."""
    remaining = continuation["original_deadline_utc"] - time.time()
    if remaining <= 0 or loop is None:
        raise ValueError("The original pending-request deadline expired.")
    loop.deadline = min(loop.deadline, monotonic() + remaining)
    loop.started_at = loop.deadline - 1800


def load_pending_request(
    recorded_run,
    expected_inventory_sha256,
    *,
    qualified_source,
    qualification,
    bundle,
    objective,
    feedback,
    preservation,
):
    root = Path(recorded_run).resolve(strict=True)
    inventory = run_inventory(root)
    digest = sha256(strict_public_json(inventory))
    if digest != expected_inventory_sha256:
        raise ValueError("The pinned pending-request evidence inventory changed.")
    receipt = read_json(root / "receipt.json")
    protocol = read_json(root / "protocol.json")
    configuration = read_json(root / "preservation-configuration.json")
    if (
        receipt["status"] != "incomplete"
        or receipt["normal_runs"] != 1
        or receipt["additional_run_attempts"] != 0
        or receipt["completed_refinement_assessments"] != 0
        or receipt["provider_counts"]["logical_requests"] != 1
        or receipt["provider_counts"]["model_calls"] != 0
        or receipt["loop_history"]
        or receipt["preservation"]["product_result"] is not None
        or strict_public_json(receipt["preservation"]["configuration"])
        != strict_public_json(configuration)
        or any((root / "requests").rglob("*"))
        or any((root / "transport").rglob("*"))
        or (root / "refinement").exists()
    ):
        raise ValueError("Only the unsent first logical request may resume.")
    expected = {
        "requested_goal": objective,
        "new_user_feedback": feedback,
        "requested_preservation_selection": preservation,
        "include_input_previews": True,
        "include_output_previews": True,
        "maximum_logical_model_requests": 3,
        "maximum_actual_model_turns": 6,
        "maximum_additional_scientific_runs": 1,
        "maximum_refinement_seconds": 1800,
        "configured_per_attempt_timeout_seconds": 600,
        "qualified_wheel_sha256": qualification["wheel_sha256"],
        "own_model_baseline": bundle.own_model_baseline,
    }
    if any(
        strict_public_json(protocol.get(key)) != strict_public_json(value)
        for key, value in expected.items()
    ) or strict_public_json(configuration["authorized_model_key_selection"]) != (
        strict_public_json(preservation)
    ):
        raise ValueError("The pending goal, permissions, baseline or budgets changed.")
    if Path(protocol["qualified_source"]).resolve(strict=True) != Path(
        qualified_source
    ).resolve(strict=True) or strict_public_json(
        read_json(root / "qualification-receipt.json")
    ) != strict_public_json(qualification):
        raise ValueError("The exact pending-request package qualification changed.")
    recipe = read_json(root / "baseline/proposal.json")
    if strict_public_json({k: v for k, v in recipe.items() if k != "context_id"}) != (
        strict_public_json(
            {k: v for k, v in bundle.recipe.items() if k != "context_id"}
        )
    ):
        raise ValueError(
            "The pending baseline recipe or scientific parameters changed."
        )
    prior_source = next(
        node
        for node in read_json(root / "baseline/prepared-workflow.json")["nodes"]
        if node["id"] == "input"
    )
    expected_source = next(
        node for node in bundle.prepared_graph["nodes"] if node["id"] == "input"
    )
    if strict_public_json(prior_source) != strict_public_json(expected_source):
        raise ValueError("The pending baseline source binding changed.")
    verification = read_json(root / "baseline/exact-output-verification.json")
    if (
        len(verification) != len(bundle.expected)
        or not verification
        or any(
            item["persisted_scientific_buffers_exactly_verified"] is not True
            for item in verification
        )
        or bundle.own_model_baseline["persisted_buffers_bound_including_source"] != 41
    ):
        raise ValueError("The failed phase did not exactly reproduce its 41 buffers.")
    events = [
        json.loads(line)
        for line in (root / "events.jsonl").read_text("utf-8").splitlines()
    ]
    previous = "0" * 64
    for number, event in enumerate(events, 1):
        if event["sequence"] != number or event["previous_event_sha256"] != previous:
            raise ValueError("The pending-request event chain changed.")
        previous = sha256(json_bytes(event))
    if [event["kind"] for event in events] != [
        "normal_product_add",
        "normal_product_run",
        "explicit_one_round_start",
    ] or events[2]["details"]["user_feedback"] != feedback:
        raise ValueError(
            "The pending run performed more than baseline and first Start."
        )
    deadline = datetime.fromisoformat(events[2]["utc"]).timestamp() + 1800
    if not time.time() < deadline:
        raise ValueError("The original pending-request deadline has expired.")
    return {
        "root": root,
        "inventory": inventory,
        "inventory_sha256": digest,
        "original_deadline_utc": deadline,
        "original_start_utc": events[2]["utc"],
        "event_chain_sha256": previous,
        "configuration": configuration,
        "pending_logical_request": 1,
        "historical_logical_reservations": 1,
        "historical_actual_model_calls": 0,
        "no_original_provider_payload_or_response_recorded": True,
        "no_new_phase_model_or_run_allowance": True,
    }


def bind_resumed_preservation(original, current):
    """Only administrative fresh-owner identity may differ after exact replay."""
    allowed = {"constraint_id", "baseline_run_id"}
    old, new = original["product_contract"], current["product_contract"]
    if strict_public_json(original["authorized_model_key_selection"]) != (
        strict_public_json(current["authorized_model_key_selection"])
    ) or strict_public_json({k: v for k, v in old.items() if k not in allowed}) != (
        strict_public_json({k: v for k, v in new.items() if k not in allowed})
    ):
        raise ValueError("The resumed scientific preservation contract changed.")
    return {
        "original_product_contract": old,
        "fresh_product_contract": new,
        "allowed_administrative_contract_fields": sorted(allowed),
        "changed_contract_fields": sorted(key for key in old if old[key] != new[key]),
        "original_primary_ref": original["primary_ref"],
        "fresh_primary_ref": current["primary_ref"],
        "original_region_ref": original["region_ref"],
        "fresh_region_ref": current["region_ref"],
        "scientific_contract_identical_as_strict_json": True,
        "original_unsent_request_payload_equality_not_claimed": True,
    }
