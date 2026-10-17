"""Verify a recorded bundle; explicit approval permits ONE FINISH/question turn.

No Qt, calculations, trials, new images, auth repairs or default nomination.
The original tree is read-only and its original wall deadline is retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from qualify_private import (
    PrivatePackageImportGuard,
    assert_private_imports,
    verify_receipt,
)
from replay_workflow_investigation_selection import read_json, run_inventory
from workflow_investigation_evidence import (
    ORIGINAL_GOAL,
    EvidenceJournal,
    SampleGeneratorCallGuard,
    json_bytes,
    sha256,
)


def load_bundle(root, expected_inventory_sha256):
    root = Path(root).resolve(strict=True)
    inventory = run_inventory(root)
    inventory_hash = hashlib.sha256(
        json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if inventory_hash != expected_inventory_sha256:
        raise ValueError(
            "The original tree differs from its independently frozen inventory."
        )
    events = [
        json.loads(line)
        for line in (root / "events.jsonl").read_text("utf-8").splitlines()
    ]
    previous = "0" * 64
    for index, event in enumerate(events, 1):
        if event["sequence"] != index or event["previous_event_sha256"] != previous:
            raise ValueError("The original event chain changed.")
        previous = sha256(json_bytes(event))
    protocol, receipt = (
        read_json(root / "protocol.json"),
        read_json(root / "receipt.json"),
    )
    stopped = read_json(root / "session/stopped.json")
    if (
        protocol["goal"] != ORIGINAL_GOAL
        or protocol["reference_material_read"]
        or receipt["status"] != "incomplete"
        or receipt["selected_result_locked"]
        or (root / "selected-result-lock.json").exists()
    ):
        raise ValueError(
            "An unchanged incomplete original blinded investigation is required."
        )
    actions = [event for event in events if event["kind"] == "product_action"]
    if len(actions) != receipt["counts"]["actions"] or any(
        event["details"]["action"]["action"] == "finish" for event in actions
    ):
        raise ValueError(
            "The original action count or absence of FINISH did not verify."
        )
    history = [
        read_json(root / event["details"]["prefix"] / "result.json")
        for event in actions
    ]
    if (
        stopped["actions_used"] != receipt["counts"]["actions"]
        or stopped["model_calls_reserved"] != receipt["counts"]["model_calls"]
    ):
        raise ValueError("Recorded reserved/actual cumulative budgets do not agree.")
    if any(item.get("status") not in {"ok", "error"} for item in history):
        raise ValueError(
            "Every original action needs its complete returned public receipt."
        )
    requests = sorted((root / "requests").glob("request-*/public-request.json"))
    request = read_json(requests[-1])
    if (
        request["objective"] != ORIGINAL_GOAL
        or any(
            event["details"]["action"]["session_id"] != request["session_id"]
            for event in actions
        )
        or stopped["session_id"] != request["session_id"]
    ):
        raise ValueError("The original goal/session identity changed.")
    limit = stopped["limits"]
    deadline = (
        min(datetime.fromisoformat(event["utc"]).timestamp() for event in events)
        + limit["max_seconds"]
    )
    if limit["max_seconds"] != 1800 or time.time() >= deadline:
        raise ValueError("The original investigation wall deadline expired.")
    candidate_prefix = actions[-1]["details"]["prefix"]
    candidate_root = root / candidate_prefix / "candidates"
    candidates = {}
    for path in sorted(candidate_root.iterdir()):
        evidence = read_json(path / "candidate.json")
        recipe = read_json(path / "proposal.json")
        roles = read_json(path / "output-roles.json")
        identity = evidence["candidate_id"]
        if (
            identity != path.name
            or evidence["run_id"] != f"{request['session_id']}_{identity}"
            or recipe["context_id"] != request["proposal_context_id"]
            or evidence["source_sha256"] != request["source"]["sha256"]
            or not roles
        ):
            raise ValueError(
                "A recorded candidate identity/source/context/role binding changed."
            )
        candidates[identity] = {
            "recipe": recipe,
            "evidence": evidence,
            "output_roles": roles,
        }
        linked = [
            event["details"]["action"]["arguments"]
            for event in actions
            if event["details"]["action"]["action"] == "trial_candidate"
            and read_json(root / event["details"]["prefix"] / "result.json")
            .get("result", {})
            .get("candidate_id")
            == identity
        ]
        if (
            len(linked) != 1
            or linked[0]["proposal"] != recipe
            or linked[0]["output_roles"] != roles
        ):
            raise ValueError(
                "A candidate must bind its exact original trial recipe and roles."
            )
    if len(candidates) != receipt["counts"]["trial_executions"]:
        raise ValueError(
            "Every original completed trial must remain explicitly identified."
        )
    original_qualification = root / "qualification-receipt.json"
    if (
        sha256(original_qualification.read_bytes())
        != protocol["qualification_receipt_sha256"]
    ):
        raise ValueError("The original qualification receipt changed.")
    verify_receipt(original_qualification, protocol["qualified_source"])
    source = root / "source/input.ome.tif"
    wrapper = read_json(root / "source/wrapper-receipt.json")
    if sha256(source.read_bytes()) != wrapper["artifact"]["sha256"]:
        raise ValueError("The original source wrapper changed.")
    manifests = [
        read_json(path)
        for path in sorted((root / "transport").glob("call-*/manifest.json"))
    ]
    if len(manifests) != receipt["counts"]["model_calls"]:
        raise ValueError("Every actual original model call needs its image manifest.")
    image_bytes = sum(item["byte_count"] for manifest in manifests for item in manifest)
    if image_bytes > limit["max_image_bytes"]:
        raise ValueError("The original cumulative image budget exceeded its limit.")
    return dict(
        root=root,
        inventory=inventory,
        inventory_sha256=inventory_hash,
        events_sha256=previous,
        request=request,
        receipt=receipt,
        stopped=stopped,
        history=history,
        candidates=candidates,
        candidate_prefix=candidate_prefix,
        deadline=deadline,
        image_bytes=image_bytes,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recorded-run", type=Path, required=True)
    parser.add_argument("--expected-original-inventory-sha256", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--qualified-source", type=Path)
    parser.add_argument("--qualification-receipt", type=Path)
    parser.add_argument("--finalization-approved", action="store_true")
    args = parser.parse_args(argv)
    bundle = load_bundle(args.recorded_run, args.expected_original_inventory_sha256)
    if not args.finalization_approved:
        print(
            json.dumps(
                {
                    "verified": True,
                    "deadline_utc": datetime.fromtimestamp(
                        bundle["deadline"], UTC
                    ).isoformat(),
                    "model_calls": 0,
                }
            )
        )
        return 0
    if not args.output or not args.qualified_source or not args.qualification_receipt:
        parser.error(
            "Approved finalization requires fresh output and "
            "new qualified source/receipt."
        )
    if args.output.resolve().is_relative_to(bundle["root"]):
        parser.error("Finalization output must be outside the original evidence tree.")
    qualification = verify_receipt(args.qualification_receipt, args.qualified_source)
    assert_private_imports(args.qualified_source)
    package = args.qualified_source.resolve() / qualification["package_directory"]
    sys.path.insert(0, str(package.parent))
    importer = PrivatePackageImportGuard(
        args.qualified_source, module_file_sha256=qualification["module_file_sha256"]
    )
    sys.meta_path.insert(0, importer)
    guard = SampleGeneratorCallGuard(package / "_sample_data.py")
    try:
        guard.install()
        from run_workflow_investigation_acceptance import ProductObservers

        from napari_vipp.core.workflow_investigation import action_schema
        from napari_vipp.core.workflow_investigation_finalization import (
            RecordedFinalization,
        )
        from napari_vipp.core.workflow_investigation_history import (
            compact_history,
            expand_request_history,
        )
        from napari_vipp.core.workflow_investigation_model import (
            generate_codex_finalization_action,
        )
        from napari_vipp.core.workflow_proposal_codex import CodexSettings

        request = bundle["request"]
        ledger = bundle["stopped"]["experiment_ledger"]
        request.update(compact_history(bundle["history"], ledger))
        request["experiment_ledger"] = ledger
        request["retained_candidate_id"] = bundle["stopped"]["retained_candidate_id"]
        request["image_previews"] = []
        request["response_schema"] = action_schema()
        request["response_schema"]["actions"] = {
            key: value
            for key, value in request["response_schema"]["actions"].items()
            if key in {"finish", "question"}
        }
        counts, limits = bundle["receipt"]["counts"], bundle["stopped"]["limits"]
        request["budgets"].update(
            actions_remaining=limits["max_actions"] - counts["actions"],
            model_calls_remaining=limits["max_model_calls"] - counts["model_calls"],
            attempts_remaining=limits["max_attempts"] - len(bundle["candidates"]),
            seconds_remaining=min(
                request["budgets"]["seconds_remaining"],
                bundle["deadline"] - time.time(),
            ),
            image_bytes_remaining=limits["max_image_bytes"] - bundle["image_bytes"],
        )
        if expand_request_history(request) != bundle["history"]:
            raise ValueError(
                "The new encoding did not reconstruct the complete original journal."
            )
        journal = EvidenceJournal(args.output)
        journal.artifact(
            "new-qualification-receipt.json", args.qualification_receipt.read_bytes()
        )

        def authority():
            if run_inventory(bundle["root"]) != bundle["inventory"]:
                raise ValueError(
                    "The original evidence tree changed during finalization."
                )
            verify_receipt(args.qualification_receipt, args.qualified_source)
            verify_receipt(
                bundle["root"] / "qualification-receipt.json",
                read_json(bundle["root"] / "protocol.json")["qualified_source"],
            )
            assert_private_imports(args.qualified_source)

        boundary = RecordedFinalization(
            request,
            bundle["candidates"],
            actions_used=counts["actions"],
            model_calls_used=counts["model_calls"],
            limits=limits,
            deadline_utc=bundle["deadline"],
            assert_current=authority,
        )
        journal.document(
            "binding.json",
            {
                "original_run": str(bundle["root"]),
                "original_inventory": bundle["inventory"],
                "original_inventory_sha256": bundle["inventory_sha256"],
                "original_event_chain_sha256": bundle["events_sha256"],
                "original_deadline_utc": bundle["deadline"],
                "candidate_prefix": bundle["candidate_prefix"],
                "new_qualification": qualification,
                "cumulative_image_bytes": bundle["image_bytes"],
                "harness_file_sha256": {
                    name: sha256(Path(__file__).with_name(name).read_bytes())
                    for name in (
                        "finalize_workflow_investigation.py",
                        "replay_workflow_investigation_selection.py",
                        "run_workflow_investigation_acceptance.py",
                        "workflow_investigation_evidence.py",
                        "qualify_private.py",
                    )
                },
            },
        )
        journal.document("public-request.json", request)
        settings = read_json(
            sorted((bundle["root"] / "requests").glob("request-*/settings.json"))[-1]
        )
        settings["max_attempts"] = 1
        observers = ProductObservers(journal)
        observers.install()
        try:
            reply = generate_codex_finalization_action(
                request,
                CodexSettings(**settings),
                deadline=time.monotonic() + max(0, bundle["deadline"] - time.time()),
                before_call=boundary.reserve_model_call,
            )
        finally:
            observers.close()
        journal.document("validated-response.json", reply)
        result = boundary.accept(reply.action)
        if guard.calls:
            raise ValueError(
                "A forbidden sample call was attempted during finalization."
            )
        if observers.counts["model_calls"] != 1:
            raise ValueError(
                "Exactly one actual native finalization model turn is required."
            )
        journal.document("action.json", reply.action)
        journal.document("result.json", result)
        journal.event(
            "explicit_recorded_finalization_action", action=reply.action, result=result
        )
        authority()
        journal.document(
            "receipt.json",
            {
                "status": "explicit-model-finish"
                if reply.action.action == "finish"
                else "question",
                "actions_used": boundary.actions_used,
                "model_calls_used": boundary.model_calls_used,
                "actual_new_model_calls": 1,
                "source_guard": guard.receipt(),
                "scientific_success_established": False,
            },
        )
        print(
            json.dumps(
                {"status": result["action"], "scientific_success_established": False}
            )
        )
        return 0 if result["action"] == "finish" else 1
    finally:
        guard.close()
        sys.meta_path.remove(importer)


if __name__ == "__main__":
    raise SystemExit(main())
