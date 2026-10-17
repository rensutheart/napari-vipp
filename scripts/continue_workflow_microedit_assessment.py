"""Explicit same-budget recorded-cache inspection/assessment/FINISH continuation.

No Qt, source loading, scientific execution, plan, trial, edits or auth changes.
The original incomplete tree is immutable. Restored ownership means recorded
evidence identity, NEVER current native Run/adoption authority or a final replay.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from qualify_private import (
    PrivatePackageImportGuard,
    assert_private_imports,
    verify_receipt,
)
from replay_workflow_investigation_selection import read_json, run_inventory
from run_workflow_investigation_acceptance import ProductObservers
from run_workflow_protected_feedback_refinement import (
    PROTECTED_FEEDBACK,
    PROTECTED_GOAL,
)
from workflow_investigation_evidence import (
    EvidenceJournal,
    SampleGeneratorCallGuard,
    json_bytes,
    read_recording_json,
    recipe_node_mapping,
    sha256,
    verify_array_replay,
)
from workflow_microedit_evidence import (
    assert_selected_preservation,
    candidate_key_mapping,
    exact_baseline_replay,
    recorded_port_bundle,
    recording_bytes,
    restore_recorded_port,
)

ORIGINAL_INVENTORY_SHA256 = (
    "a50b4aa261531af8e2d31cacf39e05e26d92a29fd315251cda46b28d5ee0ccec"
)
ALLOWED_ACTIONS = frozenset(
    {
        "describe_candidate",
        "describe_node",
        "describe_nodes",
        "inspect_source",
        "inspect_candidate",
        "compare_candidates",
        "assess_experiment",
        "retain_candidate",
        "finish",
        "question",
    }
)


def compact(value):
    return json.dumps(
        value, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode()


def load_recorded(root):
    root = root.resolve(strict=True)
    inventory = run_inventory(root)
    if sha256(compact(inventory)) != ORIGINAL_INVENTORY_SHA256:
        raise ValueError("The entire frozen incomplete original tree changed.")
    events = [
        json.loads(line)
        for line in (root / "events.jsonl").read_text("utf-8").splitlines()
    ]
    previous = "0" * 64
    for i, event in enumerate(events, 1):
        if event["sequence"] != i or event["previous_event_sha256"] != previous:
            raise ValueError("The original complete event chain changed.")
        previous = sha256(json_bytes(event))
    protocol, receipt = (
        read_json(root / "protocol.json"),
        read_json(root / "receipt.json"),
    )
    sources = {
        **protocol["harness_file_sha256"],
        Path(__file__).name: sha256(Path(__file__).read_bytes()),
    }
    for name, digest in sources.items():
        if sha256(Path(__file__).with_name(name).read_bytes()) != digest:
            raise ValueError("A frozen recorder source changed.")
    terminal = read_json(root / "session/terminal.json")
    starts = [
        e for e in events if e["kind"] == "explicit_investigative_refinement_start"
    ]
    if len(starts) != 1:
        raise ValueError("One original explicit Start is required.")
    deadline = datetime.fromisoformat(starts[0]["utc"]).timestamp() + 1800
    actions = [
        read_json(root / f"actions/action-{i:03d}/action.json") for i in range(1, 24)
    ]
    history = [
        read_json(root / f"actions/action-{i:03d}/result.json") for i in range(1, 24)
    ]
    if (
        protocol["goal"] != PROTECTED_GOAL
        or protocol["feedback"] != PROTECTED_FEEDBACK
        or protocol["reference_material_read"] is not False
        or receipt["counts"]
        != {
            "actions": 23,
            "logical_requests": 23,
            "model_calls": 23,
            "trial_executions": 2,
        }
        or receipt["status"] != "incomplete"
        or receipt["selected_result_locked"]
        or terminal["finished"]
        or terminal["actions_used"] != 23
        or terminal["attempted_trials"] != 2
        or terminal["model_calls_reserved"] != 23
        or terminal["stop_reason"] != "Investigation request history exceeds 192 KiB."
        or any(a["action"] == "finish" for a in actions)
        or len(list((root / "transport").glob("call-*/native-response.json"))) != 23
        or len(list((root / "trials").glob("trial-*/full-state.json"))) != 2
        or (root / "selected-result-lock.json").exists()
    ):
        raise ValueError("The exact incomplete 23-call/two-trial boundary changed.")
    if any(a["session_id"] != terminal["session_id"] for a in actions):
        raise ValueError("The original actions belong to another session.")
    if time.time() >= deadline:
        raise ValueError("The ORIGINAL investigation deadline expired.")
    verify_receipt(root / "qualification-receipt.json", protocol["qualified_source"])
    return SimpleNamespace(
        root=root,
        inventory=inventory,
        protocol=protocol,
        receipt=receipt,
        terminal=terminal,
        actions=actions,
        history=history,
        deadline=deadline,
        event_chain=previous,
        sources=sources,
    )


def restore_pipeline(root, prefix):
    """Only exact saved data/metadata, with no execution or source loading."""
    from napari_vipp.core.pipeline import PrototypePipeline
    from napari_vipp.core.workflow import deserialize_workflow

    graph = read_json(root / prefix / "workflow.json")
    state = read_recording_json(root / prefix / "full-state.json")
    restored = deserialize_workflow(graph)
    pipeline = PrototypePipeline()
    pipeline.restore_graph(
        restored["nodes"], restored["connections"], restored["output_tunnels"]
    )
    bundle = recorded_port_bundle(root, state["ports"])
    for node, port, saved, expected in bundle.expected:
        data, carried = restore_recorded_port(saved, expected)
        if len(pipeline.node_outputs[node]) != port:
            raise ValueError("Recorded ports are not complete and ordered.")
        pipeline.node_outputs[node].append(data)
        pipeline.node_output_states[node].append(carried)
        if port == 0:
            pipeline.outputs[node], pipeline.output_states[node] = data, carried
    pipeline.completed_node_ids = set(state["completed_node_ids"])
    for field in (
        "node_execution_states",
        "node_execution_messages",
        "node_compute_provenance",
        "node_cache_lineage",
    ):
        setattr(pipeline, field, state[field])
    exact_baseline_replay(bundle, pipeline, {n: n for n in pipeline.nodes})
    return pipeline


def restore_transmitted_evidence(engine, bundle):
    """Register historical handles ONLY when actual native calls transmitted them."""
    from napari_vipp.core.workflow_proposal_images import ImagePreview

    engine._issued_numeric.clear()
    engine._sent_numeric.clear()
    engine._numeric_owners.clear()
    engine._sent_previews.clear()
    engine._sent_preview_owners.clear()
    engine._image_bytes = 0
    created = {"candidate0": 0}
    for index, (action, result) in enumerate(
        zip(bundle.actions, bundle.history, strict=True), 1
    ):
        engine._sent_numeric.add(("input", engine._source.fingerprint))
        engine._sent_numeric.update(engine._issued_numeric)
        for candidate in engine.candidates:
            if candidate.candidate_id not in created:
                continue
            for role, binding in candidate._outputs:
                key = (f"{candidate.candidate_id}/{role}", binding.fingerprint)
                engine._sent_numeric.add(key)
                engine._numeric_owners[key] = frozenset({candidate.candidate_id})
        manifest = read_json(bundle.root / f"transport/call-{index:03d}/manifest.json")
        previous = bundle.actions[index - 2] if index > 1 else None
        for item in manifest:
            artifact = item["artifact"]
            pixels = (bundle.root / artifact["path"]).read_bytes()
            if sha256(pixels) != item["sha256"] or len(pixels) != item["byte_count"]:
                raise ValueError("An actually transmitted historical PNG changed.")
            owners = set()
            if item["label"].startswith("Protected prior foreground · "):
                owners.add("candidate0")
            elif previous and previous["action"] == "inspect_candidate":
                owners.add(previous["arguments"]["candidate_id"])
            engine._sent_previews.add(item["sha256"])
            engine._sent_preview_owners[item["sha256"]] = (
                engine._sent_preview_owners.get(item["sha256"], frozenset()) | owners
            )
            engine._image_bytes += len(pixels)
        value = result.get("result", {})
        reference = value.get("evidence_ref") if isinstance(value, dict) else None
        if reference:
            if action["action"] == "inspect_candidate":
                owners = (action["arguments"]["candidate_id"],)
            elif action["action"] == "compare_candidates":
                owners = (action["arguments"]["left"], action["arguments"]["right"])
            elif action["action"] == "inspect_source":
                owners = ()
            else:
                raise ValueError("A historical numeric handle lacks its actual issuer.")
            payload = {k: v for k, v in value.items() if k != "evidence_ref"}
            if sha256(compact(payload)) != reference["sha256"]:
                raise ValueError(
                    "An issued numeric handle differs from its actual data."
                )
            key = (reference["alias"], reference["sha256"])
            engine._issued_numeric.add(key)
            engine._numeric_owners[key] = frozenset(owners)
        if action["action"] in {"trial_candidate", "trial_edit"}:
            created[value["candidate_id"]] = index
    if engine._image_bytes > engine.limits.max_image_bytes:
        raise ValueError("The original transmitted image budget was exceeded.")
    # Action23's protected PNG was recorded but not sent; no candidate2 frame exists.
    engine._images = tuple(
        ImagePreview(
            item["label"],
            (bundle.root / item["artifact"]["path"]).read_bytes(),
            item["description"],
        )
        for item in read_json(
            bundle.root / "actions/action-023/result-previews/manifest.json"
        )
    )
    engine._preview_owners = {id(p): frozenset({"candidate0"}) for p in engine._images}


def restore_engine(bundle, authority):
    from napari_vipp.core.compute import ComputeRequest
    from napari_vipp.core.snapshots import GraphSnapshot
    from napari_vipp.core.workflow_assessment import (
        _capture_binding,
        capture_assessment_snapshot,
    )
    from napari_vipp.core.workflow_investigation import (
        InvestigationCandidate,
        InvestigationLimits,
        InvestigationSession,
        capture_completed_candidate,
    )
    from napari_vipp.core.workflow_proposals import ProposalDialogue
    from napari_vipp.core.workflow_refinement import capture_preservation_constraint

    pipeline = restore_pipeline(bundle.root, "baseline/completed-workflow")
    initial = bundle.root / "session/initial/candidates/candidate0"
    keymap = read_json(initial / "key-to-node.json")
    saved0 = read_json(initial / "candidate.json")
    recipe0 = read_json(initial / "proposal.json")
    config = read_json(bundle.root / "preservation-configuration.json")
    original_ids = tuple(
        read_json(bundle.root / "baseline/binding.json")["added_node_ids"]
    )
    if set(original_ids) != set(keymap.values()):
        raise ValueError("Original ordered baseline Add identities changed.")
    snapshot = capture_assessment_snapshot(
        pipeline, "input", original_ids, saved0["run_id"]
    )
    constraint = capture_preservation_constraint(
        snapshot,
        tuple(config["primary_ref"]),
        tuple(config["region_ref"]),
        output_sharing=True,
    )
    if json_bytes(constraint.public_contract()) != json_bytes(
        config["product_contract"]
    ):
        raise ValueError("The actual recorded preservation source/grid/region changed.")
    roles0 = read_json(initial / "output-roles.json")
    actualroles = [{**role, "node": keymap[role["node"]]} for role in roles0]
    captured = capture_completed_candidate(
        pipeline,
        snapshot,
        actualroles,
        preservation_constraint=constraint,
        protected_role="band_mask",
    )
    detached = snapshot.baseline.to_pipeline()
    detached.outputs["input"], detached.output_states["input"] = (
        pipeline.outputs["input"],
        pipeline.output_states["input"],
    )
    engine = InvestigationSession.from_completed(
        detached,
        captured,
        ProposalDialogue(
            PROTECTED_GOAL, (("Additional instructions 1", PROTECTED_FEEDBACK),)
        ),
        compute_request=ComputeRequest(),
        limits=InvestigationLimits(max_attempts=3),
        include_input_images=True,
        include_output_images=True,
        require_experiment_plan=True,
        assert_current=authority,
    )
    engine.session_id = bundle.terminal["session_id"]
    engine._context = replace(engine._context, context_id=recipe0["context_id"])
    engine.deadline = time.monotonic() + max(0, bundle.deadline - time.time())
    if (
        engine.candidates[0].key_to_node != keymap
        or engine.candidates[0].output_roles != roles0
        or engine.candidates[0].evidence["role_bindings"] != saved0["role_bindings"]
    ):
        raise ValueError("The restored baseline's explicit role/key identity changed.")
    engine._candidates[0] = replace(
        engine._candidates[0], _recipe=compact(recipe0), _evidence=compact(saved0)
    )
    for number in (1, 2):
        folder = bundle.root / f"actions/action-023/candidates/candidate{number}"
        evidence, recipe = (
            read_json(folder / "candidate.json"),
            read_json(folder / "proposal.json"),
        )
        roles, keymap = (
            read_json(folder / "output-roles.json"),
            read_json(folder / "key-to-node.json"),
        )
        trial = restore_pipeline(bundle.root, f"trials/trial-{number:03d}")
        verify_array_replay(pipeline.outputs["input"], trial.outputs["input"])
        if recording_bytes(pipeline.output_states["input"]) != recording_bytes(
            trial.output_states["input"]
        ):
            raise ValueError("A recorded trial used different actual source metadata.")
        actual_graph = read_json(
            bundle.root / f"trials/trial-{number:03d}/workflow.json"
        )
        if recipe_node_mapping(recipe, actual_graph, recipe["context_id"]) != keymap:
            raise ValueError(
                "The recorded trial is not its exact model-authored graph."
            )
        execution = read_json(
            bundle.root / f"trials/trial-{number:03d}/execution-result.json"
        )
        if (
            execution["error"]
            or execution["cancelled"]
            or execution["failure"]
            or execution["execution_report"]["cleanup_succeeded"] is not True
            or evidence["run_id"] != f"{engine.session_id}_candidate{number}"
            or evidence["status"] != "completed"
        ):
            raise ValueError("The recorded candidate lacks successful owned execution.")
        graph = GraphSnapshot.from_pipeline(trial)
        if set(keymap.values()) != set(trial.nodes) - {"input"}:
            raise ValueError("The complete recorded trial graph/keymap differs.")
        outputs, budget = [], [engine._source.data.nbytes]
        for role in roles:
            node, port = keymap[role["node"]], role["port"]
            binding = replace(
                _capture_binding(
                    node,
                    port,
                    trial.node_outputs[node][port],
                    trial.node_output_states[node][port],
                    budget,
                ),
                live=None,
            )
            issued = next(
                row for row in evidence["outputs"] if row["role"] == role["role"]
            )
            if binding.fingerprint != issued["sha256"]:
                raise ValueError(
                    "Recorded role output bytes differ from issued diagnostics."
                )
            outputs.append((role["role"], binding))
        candidate = InvestigationCandidate(
            f"candidate{number}",
            evidence["run_id"],
            "completed",
            graph,
            compact(recipe),
            compact(evidence),
            tuple(outputs),
            engine._original.source_fingerprint,
            compact(roles),
            tuple(keymap.items()),
        )
        candidate_key_mapping(candidate)
        assert_selected_preservation(candidate, constraint.public_contract())
        engine._candidates.append(candidate)
    restore_transmitted_evidence(engine, bundle)
    # Revalidate exact original model notes/assessments through normal ledger APIs.
    for action, result in zip(bundle.actions, bundle.history, strict=True):
        kind, arguments = action["action"], action["arguments"]
        if kind == "plan_experiment":
            engine._do_plan_experiment(**arguments)
        elif kind in {"trial_candidate", "trial_edit"}:
            engine._experiments.bind_trial(
                arguments["experiment_id"], result["result"]["candidate_id"]
            )
        elif kind == "assess_experiment":
            engine._do_assess_experiment(**arguments)
    if json_bytes(engine.experiment_ledger) != json_bytes(
        bundle.terminal["experiment_ledger"]
    ):
        raise ValueError(
            "The complete recorded scientific ledger did not restore exactly."
        )
    engine._history = [compact(item) for item in bundle.history]
    engine.actions_used, engine.model_calls_used = 23, 23
    engine.retained_candidate_id = bundle.terminal["retained_candidate_id"]
    engine.stop_reason = ""
    return engine


class AssessmentOnlyObservers(ProductObservers):
    def install(self):
        super().install()
        from napari_vipp.core import workflow_investigation as core

        def forbid(_original):
            def call(*_a, **_k):
                raise RuntimeError(
                    "Recorded assessment continuation forbids execution/auth."
                )

            return call

        def action_guard(original):
            def call(session, action):
                if action.action not in ALLOWED_ACTIONS:
                    raise ValueError(
                        "Only recorded inspection, assessment and FINISH "
                        "are authorized."
                    )
                return original(session, action)

            return call

        from napari_vipp.core import workflow_proposal_codex as codex

        self._patch(core, "execute_pipeline_request", forbid)
        self._patch(core.InvestigationSession, "perform", action_guard)
        self._patch(codex, "start_codex_login", forbid)
        self._patch(codex, "repair_codex_profile", forbid)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recorded-run", type=Path, required=True)
    parser.add_argument("--qualified-source", type=Path, required=True)
    parser.add_argument("--qualification-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--continuation-approved", action="store_true")
    args = parser.parse_args(argv)
    bundle = load_recorded(args.recorded_run)
    qualification = verify_receipt(args.qualification_receipt, args.qualified_source)
    root = args.qualified_source.resolve(strict=True)
    assert_private_imports(root)
    package = root / qualification["package_directory"]
    sys.path.insert(0, str(package.parent))
    importer = PrivatePackageImportGuard(
        root, module_file_sha256=qualification["module_file_sha256"]
    )
    sys.meta_path.insert(0, importer)
    guard = SampleGeneratorCallGuard(package / "_sample_data.py")
    journal, observers, engine = None, None, None

    def authority():
        if (
            time.time() >= bundle.deadline
            or run_inventory(bundle.root) != bundle.inventory
        ):
            raise ValueError(
                "The original deadline expired or frozen evidence changed."
            )
        verify_receipt(args.qualification_receipt, root)
        assert_private_imports(root)
        if any(
            sha256(Path(__file__).with_name(name).read_bytes()) != digest
            for name, digest in bundle.sources.items()
        ):
            raise ValueError("A running continuation/helper source changed.")

    try:
        guard.install()
        engine = restore_engine(bundle, authority)
        request = engine.public_request()
        from napari_vipp.core.workflow_investigation_history import (
            expand_request_history,
        )

        if expand_request_history(request) != bundle.history:
            raise ValueError("The fixed history codec is not lossless.")
        if not args.continuation_approved:
            print(
                json.dumps(
                    {
                        "live_started": False,
                        "restored_candidates": 3,
                        "actions_used": 23,
                        "model_calls_used": 23,
                        "trial_attempts": 2,
                        "request_bytes": len(compact(request)),
                        "deadline_utc": bundle.deadline,
                        "candidate2_prior_images": 0,
                        "image_bytes_used": engine._image_bytes,
                    }
                )
            )
            return 0
        if (
            args.output is None
            or args.output.exists()
            or args.output.resolve().is_relative_to(bundle.root)
        ):
            raise ValueError("Continuation requires fresh separate evidence.")
        journal = EvidenceJournal(args.output)
        journal.document(
            "binding.json",
            {
                "original_root": str(bundle.root),
                "original_inventory_sha256": ORIGINAL_INVENTORY_SHA256,
                "original_inventory": bundle.inventory,
                "original_event_chain": bundle.event_chain,
                "original_deadline_utc": bundle.deadline,
                "original_counts": bundle.receipt["counts"],
                "allowed_actions": sorted(ALLOWED_ACTIONS),
                "new_scientific_runs_allowed": 0,
                "native_current_run_ownership_claimed": False,
                "new_qualification": qualification,
                "image_bytes_already_transmitted": engine._image_bytes,
                "same_session_identity_restored": engine.session_id,
                "script_sha256": sha256(Path(__file__).read_bytes()),
                "harness_sources": bundle.sources,
            },
        )
        observers = AssessmentOnlyObservers(journal)
        observers.install()
        from napari_vipp.core.workflow_investigation_model import (
            generate_codex_investigation_action,
        )
        from napari_vipp.core.workflow_proposal_codex import CodexSettings

        settings = read_json(bundle.root / "requests/request-023/settings.json")
        while not engine.finished and not engine.stop_reason:
            authority()
            request = engine.public_request()
            request["response_schema"]["actions"] = {
                k: v
                for k, v in request["response_schema"]["actions"].items()
                if k in ALLOWED_ACTIONS
            }
            request["budgets"]["attempts_remaining"] = 0
            request["rules"].append(
                "Recorded-evidence continuation only: inspect existing candidates, "
                "assess the outstanding experiment, then FINISH. No new trial, "
                "edits, plan, scientific execution or authority reset. Candidate2 "
                "images have not yet been inspected; use actual frozen outputs "
                "if you need new views, or disclose visual uncertainty."
            )
            current = {
                **settings,
                "timeout_seconds": min(600, max(1, int(bundle.deadline - time.time()))),
            }
            reply = generate_codex_investigation_action(
                request,
                CodexSettings(**current),
                image_previews=engine.preview_images(),
                deadline=engine.deadline,
                before_call=engine.record_model_call,
            )
            engine.perform(reply.action)
        authority()
        if observers.counts["trial_executions"] or engine.attempted_trials != 2:
            raise ValueError("No additional scientific trial is authorized.")
        selected = engine.selected_candidate
        journal.document(
            "receipt.json",
            {
                "finished": engine.finished,
                "stop_reason": engine.stop_reason,
                "new_counts": observers.counts,
                "cumulative_actions": engine.actions_used,
                "cumulative_model_calls": engine.model_calls_used,
                "cumulative_image_bytes": engine._image_bytes,
                "scientific_trials": 2,
                "native_ui_adoption_claimed": False,
                "new_scientific_runs": 0,
                "experiment_ledger": engine.experiment_ledger,
                "source_guard": guard.receipt(),
            },
        )
        if engine.finished:
            assert_selected_preservation(
                selected, engine._preservation.public_contract()
            )
            journal.lock_selection(
                {
                    "candidate_id": selected.candidate_id,
                    "run_id": selected.run_id,
                    "recipe": selected.recipe,
                    "evidence": selected.evidence,
                    "key_to_node": selected.key_to_node,
                    "role_mapping": selected.output_roles,
                    "original_recorded_run": str(bundle.root),
                    "original_inventory_sha256": ORIGINAL_INVENTORY_SHA256,
                    "selection_basis": (
                        "Genuine model FINISH after strict recorded "
                        "experiment assessment"
                    ),
                    "recorded_outputs_only_no_final_native_replay": True,
                }
            )
        print(
            json.dumps(
                {
                    "finished": engine.finished,
                    "selected_candidate": selected.candidate_id
                    if engine.finished
                    else None,
                    "actions": engine.actions_used,
                    "calls": engine.model_calls_used,
                }
            )
        )
        return 0 if engine.finished else 1
    except Exception as exc:
        if journal is not None:
            if not (journal.directory / "receipt.json").exists():
                journal.document(
                    "receipt.json",
                    {
                        "finished": False,
                        "status": "incomplete",
                        "stop_reason": str(exc),
                        "selected_result_locked": False,
                        "original_inventory_sha256": ORIGINAL_INVENTORY_SHA256,
                        "original_deadline_utc": bundle.deadline,
                        "new_counts": observers.counts if observers else {},
                        "cumulative_actions": engine.actions_used if engine else 23,
                        "cumulative_model_calls": engine.model_calls_used
                        if engine
                        else 23,
                        "cumulative_image_bytes": engine._image_bytes
                        if engine
                        else None,
                        "experiment_ledger": engine.experiment_ledger if engine else [],
                        "new_scientific_runs": 0,
                        "native_ui_adoption_claimed": False,
                    },
                )
            journal.document(
                "continuation-failure.json",
                {
                    "status": "incomplete",
                    "type": type(exc).__name__,
                    "reason": str(exc),
                    "new_counts": observers.counts if observers else {},
                    "selected_result_locked": False,
                },
            )
        raise
    finally:
        if observers:
            observers.close()
        guard.close()
        sys.meta_path.remove(importer)


if __name__ == "__main__":
    raise SystemExit(main())
