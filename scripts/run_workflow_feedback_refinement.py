"""One explicitly approved product refinement of its own locked model result.

Default mode verifies the immutable baseline and private qualification only.
Live mode first reproduces every selected baseline output exactly, then uses
the ordinary one-round Start refinement UI path. No analysis parameters are
chosen here, and no previous scientific interpretation is supplied to a model.
"""

from __future__ import annotations

import argparse
import ast
import functools
import importlib.metadata
import json
import os
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from time import monotonic
from types import SimpleNamespace

from qualify_private import (
    PrivatePackageImportGuard,
    assert_private_imports,
    verify_receipt,
)
from replay_workflow_investigation_selection import (
    load_recorded_selection,
    read_json,
    run_inventory,
)
from run_workflow_investigation_acceptance import (
    ProductObservers,
    assert_loaded_source,
    capture_pipeline,
)
from workflow_investigation_evidence import (
    ORIGINAL_GOAL,
    EvidenceJournal,
    OwnedReplayBinding,
    SampleGeneratorCallGuard,
    assert_owned_dispatch,
    freeze_owned_dispatch,
    freeze_replay_binding,
    json_bytes,
    owned_completion_details,
    owned_replay_completion,
    public_value,
    sha256,
    verify_array_replay,
    verify_executed_replay_graph,
    verify_mesh_objects_replay,
    verify_mesh_replay,
)
from workflow_pending_request_resume import (
    anchor_continuation_loop,
    bind_resumed_preservation,
    load_pending_request,
    strict_public_json,
)

EXPECTED_BASELINE_LOCK = (
    "640882cca3943c6c907a9f086fbd86c84b05bccc40dcd7a3a05c46ef3d349301"
)
USER_FEEDBACK = (
    "Go ahead and run that refinement pass: close the mesh ends and test "
    "background correction for the band thickness, preserving the reconstructed "
    "central section. Let the model choose the parameters and show me the comparison."
)


def assert_feedback_request(request, feedback=USER_FEEDBACK, objective=ORIGINAL_GOAL):
    if request.get("objective") != objective or request.get("clarifications") != [
        {"question": "Additional instructions 1", "answer": feedback}
    ]:
        raise ValueError("The original goal or exact new user feedback changed.")


def verify_locked_events(root, lock):
    """The pinned lock binds the entire ordered immutable event chain too."""
    data = (root / "events.jsonl").read_bytes()
    if sha256(data) != lock["events_file_sha256"]:
        raise ValueError("The locked baseline event file changed.")
    events = [json.loads(line) for line in data.decode("utf-8").splitlines()]
    previous = "0" * 64
    for number, event in enumerate(events, 1):
        if event["sequence"] != number or event["previous_event_sha256"] != previous:
            raise ValueError("The locked baseline event chain changed.")
        previous = sha256(json_bytes(event))
    if len(events) != lock["event_count"] or previous != lock["last_event_sha256"]:
        raise ValueError("The locked baseline event count or terminal hash changed.")


def provider_result_record(result):
    """Record an actual result without traversing private context/PNG buffers."""
    record = {
        "attempts": public_value(result.attempts),
        "provider": result.provider,
        "model": result.model,
    }
    for name in ("assessment", "proposal", "request_sha256"):
        if hasattr(result, name):
            record[name] = public_value(getattr(result, name))
    return record


def load_recorded_recovery(path, expected_inventory_sha256):
    """Freeze one failed observer run without granting another model allowance."""
    root = Path(path).resolve(strict=True)
    inventory = run_inventory(root)
    inventory_hash = sha256(
        json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    if inventory_hash != expected_inventory_sha256:
        raise ValueError("The independently frozen failed-run inventory changed.")
    receipt = read_json(root / "receipt.json")
    events = [
        json.loads(line)
        for line in (root / "events.jsonl").read_text("utf-8").splitlines()
    ]
    previous = "0" * 64
    for number, event in enumerate(events, 1):
        if event["sequence"] != number or event["previous_event_sha256"] != previous:
            raise ValueError("The failed-run ordered event chain changed.")
        previous = sha256(json_bytes(event))
    starts = [event for event in events if event["kind"] == "explicit_one_round_start"]
    if (
        receipt["status"] != "incomplete"
        or receipt["normal_runs"] != 1
        or receipt["additional_run_attempts"] != 0
        or receipt["provider_counts"]["logical_requests"] != 1
        or receipt["provider_counts"]["model_calls"] != 1
        or len(starts) != 1
        or len(list((root / "transport").glob("call-*/native-response.json"))) != 1
        or len(list((root / "requests").glob("request-*/public-request.json"))) != 1
    ):
        raise ValueError("Only the one recorded baseline assessment may be continued.")
    deadline = datetime.fromisoformat(starts[0]["utc"]).timestamp() + 1800
    if not time.time() < deadline:
        raise ValueError("The original refinement deadline has expired.")
    request = read_json(root / "requests/request-001/public-request.json")
    assert_feedback_request(request)
    text = read_json(root / "transport/call-001/native-response.json")
    model = next(
        event["details"]["result"]["model"]
        for event in events
        if event["kind"] == "native_rpc_response"
        and event["details"]["method"] == "thread/start"
    )
    return {
        "root": root,
        "inventory": inventory,
        "inventory_sha256": inventory_hash,
        "request": request,
        "text": text,
        "model": model,
        "original_deadline_utc": deadline,
        "event_chain_sha256": previous,
        "original_response_sha256": sha256(
            (root / "transport/call-001/native-response.json").read_bytes()
        ),
    }


def recorded_assessment_result(recovery, context, objective, journal):
    """Reuse exactly one valid report after exact evidence and PNG comparison."""
    from napari_vipp.core.workflow_assessment_model import (
        ModelAssessmentResult,
        _request,
        validate_assessment,
    )

    old = recovery["request"]
    original_context = SimpleNamespace(
        assessment_id=old["assessment_id"], run_id=old["run_id"], _evidence=old
    )
    original_report = validate_assessment(recovery["text"], original_context)
    current = context.public_request(objective)
    assert_feedback_request(current)
    excluded = {"assessment_id", "run_id"}
    if {key: value for key, value in old.items() if key not in excluded} != {
        key: value for key, value in current.items() if key not in excluded
    }:
        raise ValueError("Fresh evidence differs from the recorded assessment request.")
    manifests = read_json(recovery["root"] / "requests/request-001/manifest.json")
    if len(context.image_previews) != len(manifests):
        raise ValueError("The fresh approved preview count changed.")
    for preview, manifest in zip(context.image_previews, manifests, strict=True):
        path = (recovery["root"] / manifest["artifact"]["path"]).resolve(strict=True)
        if not path.is_relative_to(recovery["root"]):
            raise ValueError("A recorded preview path escaped its evidence root.")
        recorded_manifest = {
            key: value for key, value in manifest.items() if key != "artifact"
        }
        if (
            preview.manifest() != recorded_manifest
            or preview.png_bytes != path.read_bytes()
        ):
            raise ValueError("The fresh approved preview bytes or manifests changed.")
    report = replace(
        original_report, assessment_id=context.assessment_id, run_id=context.run_id
    )
    rebound = validate_assessment(json.dumps(report.to_dict()), context)
    before, after = original_report.to_dict(), rebound.to_dict()
    changed = {key for key in before if before[key] != after[key]}
    if not changed <= excluded:
        raise ValueError("Recorded scientific report content was changed.")
    _, request_bytes = _request(context, objective)
    journal.document(
        "recorded-assessment-reuse.json",
        {
            "original_response_sha256": recovery["original_response_sha256"],
            "original_request": old,
            "fresh_request": current,
            "original_assessment_id": original_report.assessment_id,
            "fresh_assessment_id": rebound.assessment_id,
            "original_run_id": original_report.run_id,
            "fresh_run_id": rebound.run_id,
            "changed_report_fields": sorted(changed),
            "allowed_request_differences": sorted(excluded),
            "current_png_bytes_identical": True,
            "scientific_report_content_unchanged": True,
            "is_new_logical_request": False,
            "is_new_model_call": False,
            "original_validated_report": before,
            "rebound_validated_report": after,
            "fresh_provider_request_sha256": sha256(request_bytes),
        },
    )
    return ModelAssessmentResult(
        rebound, (), "codex", recovery["model"], sha256(request_bytes), context
    )


def verify_baseline(args):
    """Bind the exact locked own-model result, not a latest-looking candidate."""
    baseline = args.baseline_replay.resolve(strict=True)
    lock_path = baseline / "selected-result-lock.json"
    if sha256(lock_path.read_bytes()) != EXPECTED_BASELINE_LOCK:
        raise ValueError("The explicitly approved baseline selection lock changed.")
    lock = read_json(lock_path)
    verify_locked_events(baseline, lock)
    for artifact in lock["artifacts"]:
        path = (baseline / artifact["path"]).resolve(strict=True)
        if not path.is_relative_to(baseline) or path.is_symlink():
            raise ValueError("A baseline artifact is redirected.")
        if sha256(path.read_bytes()) != artifact["sha256"]:
            raise ValueError("A baseline locked artifact changed.")
    bundle = load_recorded_selection(args.recorded_run, finalization=args.finalization)
    receipt = read_json(baseline / "receipt.json")
    selection = lock["selection"]
    if (
        receipt["status"] != "recorded-selection-exactly-replayed"
        or selection["candidate_id"] != bundle.candidate_id
        or selection["recipe"] != bundle.recipe
        or selection["original_finish_sha256"] != bundle.finish_sha256
        or selection["evidence"] != bundle.evidence
        or receipt["model_calls"] != 0
    ):
        raise ValueError("Baseline lock does not bind the exact prior model selection.")
    qualification = verify_receipt(args.qualification_receipt, args.qualified_source)
    return bundle, baseline, run_inventory(baseline), qualification


class FeedbackObservers(ProductObservers):
    """Record the unchanged ordinary assessment/proposal provider boundaries."""

    def __init__(
        self,
        journal,
        recovery=None,
        feedback=USER_FEEDBACK,
        objective=ORIGINAL_GOAL,
        *,
        preservation_required=False,
        pending=None,
    ):
        super().__init__(journal)
        self.recovery = recovery
        self.pending = pending
        self.pending_resume_used = False
        self.feedback = feedback
        self.objective = objective
        self.preservation_required = preservation_required
        self.preservation_contract = None
        self.preservation_result = None
        self.recovery_used = False
        self.historical_model_calls = 1 if recovery is not None else 0
        self.historical_logical_requests = (
            1 if recovery is not None or pending is not None else 0
        )
        self.counts["model_calls"] = self.historical_model_calls
        self.counts["logical_requests"] = self.historical_logical_requests

    def install(self):
        super().install()
        from napari_vipp.core import workflow_assessment_model as assessment
        from napari_vipp.core import workflow_proposal_codex as codex

        def observe(kind):
            def replacement(original):
                @functools.wraps(original)
                def call(context, objective, settings, **kwargs):
                    reuse = self.recovery is not None and not self.recovery_used
                    resume = self.pending is not None and not self.pending_resume_used
                    if resume:
                        if (
                            kind != "assessment"
                            or self.counts["logical_requests"] != 1
                            or self.counts["model_calls"] != 0
                        ):
                            raise RuntimeError("Only pending first request may resume.")
                        self.pending_resume_used = True
                    elif reuse:
                        if kind != "assessment" or self.counts["logical_requests"] != 1:
                            raise RuntimeError(
                                "Only the recorded first assessment may be reused."
                            )
                        self.recovery_used = True
                    else:
                        self.counts["logical_requests"] += 1
                    if self.counts["logical_requests"] > 3:
                        raise RuntimeError(
                            "The one-round three-request bound was exceeded."
                        )
                    prefix = f"requests/request-{self.counts['logical_requests']:03d}"
                    try:
                        request = context.public_request(objective)
                        self.journal.document(f"{prefix}/public-request.json", request)
                        self.journal.document(f"{prefix}/settings.json", settings)
                        assert_feedback_request(request, self.feedback, self.objective)
                        if self.preservation_required:
                            assert_preservation_request(
                                request,
                                kind,
                                self.counts["logical_requests"],
                                self.preservation_contract,
                                self.preservation_result,
                            )
                        self.journal.previews(prefix, context.image_previews)
                    except Exception as exc:
                        self.journal.document(
                            f"{prefix}/observer-validation-failure.json",
                            {"type": type(exc).__name__, "reason": str(exc)},
                        )
                        self.journal.event(
                            "product_model_request_validation_failed",
                            request_kind=kind,
                            prefix=prefix,
                            type=type(exc).__name__,
                            reason=str(exc),
                        )
                        raise
                    self.journal.event(
                        "product_model_request",
                        request_kind=kind,
                        prefix=prefix,
                        resumed_pending_logical_slot=resume,
                    )
                    result = (
                        recorded_assessment_result(
                            self.recovery, context, objective, self.journal
                        )
                        if reuse
                        else original(context, objective, settings, **kwargs)
                    )
                    self.journal.document(
                        f"{prefix}/validated-response.json",
                        provider_result_record(result),
                    )
                    self.journal.event(
                        "product_model_response", request_kind=kind, prefix=prefix
                    )
                    return result

                return call

            return replacement

        def forbidden(_original):
            def call(*_args, **_kwargs):
                raise RuntimeError(
                    "Authentication changes are outside this acceptance run."
                )

            return call

        self._patch(assessment, "generate_codex_assessment", observe("assessment"))
        self._patch(codex, "generate_codex_proposal", observe("revision"))

        def bound_transport(original):
            @functools.wraps(original)
            def call(*args, **kwargs):
                continuation = self.recovery or self.pending
                if (
                    continuation is not None
                    and not time.time() < continuation["original_deadline_utc"]
                ):
                    raise RuntimeError(
                        "The original refinement deadline expired before transport."
                    )
                if self.counts["model_calls"] >= 6:
                    raise RuntimeError(
                        "The six-turn bound was reached before transport."
                    )
                contract = kwargs.get("assessment_contract")
                if contract is not None:
                    prefix = f"transport/call-{self.counts['model_calls'] + 1:03d}"
                    self.journal.document(
                        f"{prefix}/assessment-contract.json", contract
                    )
                return original(*args, **kwargs)

            return call

        self._patch(codex, "_call_codex", bound_transport)
        self._patch(codex, "start_codex_login", forbidden)
        self._patch(codex, "repair_codex_profile", forbidden)


def exact_baseline_outputs(bundle, pipeline, binding):
    records = []
    for role, old_node, saved, expected in bundle.expected:
        node, port = binding.node_mapping[old_node], role["port"]
        actual = pipeline.node_outputs[node][port]
        if "persisted_mesh_arrays" in saved:
            for name in ("vertices", "faces", "face_object_ids"):
                verify_array_replay(getattr(expected, name), getattr(actual, name))
        elif "array" in saved:
            verify_array_replay(expected, actual)
        else:
            verify_mesh_replay(expected, actual)
            verify_mesh_objects_replay(saved["mesh"]["objects"], actual.objects)
        if not getattr(bundle, "recorded_buffers_only", False) and saved[
            "state"
        ] != public_value(pipeline.node_output_states[node][port]):
            raise ValueError("Baseline full output metadata did not reproduce exactly.")
        records.append(
            {
                **role,
                "old_node_id": old_node,
                "node_id": node,
                "prior_complete_metadata_exactly_verified": not getattr(
                    bundle, "recorded_buffers_only", False
                ),
                "persisted_scientific_buffers_exactly_verified": True,
            }
        )
    return records


def execution_completion_receipt(widget, ticket, binding):
    assert_owned_dispatch(widget, ticket, completed=True)
    active = widget._active_pipeline_run_id is not None
    pending = bool(
        widget._pipeline_run_pending
        or widget._pending_dirty_node_ids
        or widget._pending_manual_node_ids
    )
    details = owned_completion_details(
        binding, widget.pipeline, active=active, pending=pending
    )
    if details["status"] != "ready":
        raise ValueError("The owned scientific run is not complete and idle.")
    return {
        "owned_dispatch": public_value(ticket),
        "fresh_raw_execution_report": public_value(widget._last_execution_report),
        "cleanup_succeeded": True,
        "report_is_fresh_not_initial": True,
        "active_execution": active,
        "pending_execution": pending,
        **details,
        "node_execution_states": {
            node: widget.pipeline.node_execution_states.get(node)
            for node in binding.added_node_ids
        },
        "completed_node_ids": sorted(widget.pipeline.completed_node_ids),
    }


def final_assessment_adoption(dialog, widget, item):
    receipt = execution_completion_receipt(widget, item["ticket"], item["binding"])
    reply = dialog._assessment.reply
    loop = dialog._refinement.loop
    if (
        dialog._busy
        or reply is None
        or reply.result is None
        or loop is None
        or loop.active
        or loop.attempted_runs != 1
        or loop.completed_rounds != 1
    ):
        raise ValueError("The final model assessment was not adopted by the idle UI.")
    dialog._assessment.assert_current(reply.context, reply.owner)
    report = reply.result.assessment
    if (
        reply.choices != dialog._assessment._choices()
        or report.run_id != item["ticket"].run_id
        or report.run_id != dialog._assessment.run_id
        or report.assessment_id != reply.context.assessment_id
        or loop.history[-1].assessment_id != report.assessment_id
    ):
        raise ValueError("The final UI assessment does not bind this exact owned run.")
    return {
        "run_id": report.run_id,
        "assessment_id": report.assessment_id,
        "model_assessment": report.to_dict(),
        "guarded_current_owner": True,
        "sharing_choices_unchanged": True,
        "idle_ui_adoption_verified": True,
        "request_sha256": reply.result.request_sha256,
        "execution": receipt,
    }


def before_assessment_capture(original, capture):
    """UI-thread hook: durable owned completion before any assessment worker."""

    @functools.wraps(original)
    def call(*args, **kwargs):
        capture()
        return original(*args, **kwargs)

    return call


def validate_preservation_selection(selection, model_keys):
    """Validate references only; never derive a scientific region or policy."""
    if (
        set(selection)
        != {
            "mask_model_key",
            "mask_port",
            "region_model_key",
            "region_port",
            "policy",
            "proposal_key",
        }
        or selection["policy"] != "preserve_all_selected_positive_foreground"
        or selection["proposal_key"] != "protected_structure"
    ):
        raise ValueError("The explicitly authorized preservation policy changed.")
    for prefix in ("mask", "region"):
        key, port = selection[f"{prefix}_model_key"], selection[f"{prefix}_port"]
        authorized = (
            "refine_reference_labels" if prefix == "mask" else "refine_circle_labels"
        )
        if (
            key != authorized
            or key not in model_keys
            or type(port) is not int
            or port != 0
        ):
            raise ValueError("A selected prior output reference is not exact.")


def require_product_preservation_api(qualified_source, qualification):
    """Fail before baseline replay if qualification lacks the real product API."""
    source = (
        Path(qualified_source)
        / qualification["package_directory"]
        / "ui/workflow_assistant_refinement.py"
    )
    module = ast.parse(source.read_text("utf-8"))
    classes = [
        node
        for node in module.body
        if isinstance(node, ast.ClassDef)
        and node.name == "AssistantRefinementController"
    ]
    if len(classes) != 1 or not any(
        isinstance(node, ast.FunctionDef) and node.name == "configure_preservation"
        for node in classes[0].body
    ):
        raise ValueError("The qualified package lacks the product preservation API.")


def configure_product_preservation(controller, selection, model_key_to_node):
    """Select authorized prior outputs through the ordinary product API only."""
    validate_preservation_selection(selection, model_key_to_node)
    refs = []
    for prefix in ("mask", "region"):
        key, port = selection[f"{prefix}_model_key"], selection[f"{prefix}_port"]
        refs.append((model_key_to_node[key], port))
    constraint = controller.configure_preservation(*refs)
    contract = public_value(constraint.public_contract())
    if not contract:
        raise ValueError("The ordinary product did not issue a preservation contract.")
    return constraint, {
        "authorized_model_key_selection": dict(selection),
        "primary_ref": list(refs[0]),
        "region_ref": list(refs[1]),
        "product_contract": contract,
        "configured_via_normal_product_api": True,
        "harness_region_or_pixels_generated": False,
    }


def assert_preservation_request(request, kind, index, contract, result=None):
    """Observe, never inject, the product-owned constraint and positive result."""
    if (
        index not in {1, 2, 3}
        or kind != ("assessment", "revision", "assessment")[index - 1]
        or not contract
    ):
        raise ValueError("The protected request sequence or constraint is missing.")
    evidence = request
    if kind == "revision":
        evidence = request.get("result_feedback", {})
        permissions = evidence.get("evidence", {}).get("image_permissions")
    else:
        permissions = request.get("image_permissions")
    if strict_public_json(evidence.get("preservation_constraint")) != (
        strict_public_json(contract)
    ):
        raise ValueError("The exact product preservation contract was not shared.")
    if permissions != {"input_images": True, "output_images": True}:
        raise ValueError("Both approved preview permissions must remain enabled.")
    if index == 3 and (
        not result
        or result.get("passed") is not True
        or strict_public_json(request.get("preservation_check"))
        != strict_public_json(result)
    ):
        raise ValueError(
            "The final request lacks the passed product preservation check."
        )


def preservation_outcome(controller, constraint, configuration, *, run_id=None):
    """Bind the issued current product outcome; never repair or choose a mask."""
    if (
        constraint is None
        or not configuration
        or controller.preservation_constraint is not constraint
        or public_value(constraint.public_contract())
        != configuration["product_contract"]
    ):
        raise ValueError("The configured product preservation snapshot changed.")
    result = controller.preservation_result
    public = None if result is None else public_value(result.public_result())
    if public is not None and (
        public.get("constraint_id")
        != configuration["product_contract"].get("constraint_id")
        or public.get("candidate_key") != "protected_structure"
    ):
        raise ValueError(
            "The product preservation result refers to another constraint."
        )
    if run_id is not None and (
        result is None
        or result.passed is not True
        or public.get("passed") is not True
        or public.get("run_id") != run_id
        or public.get("missing_count") != 0
        or public.get("retained_count") != public.get("protected_count")
    ):
        raise ValueError("The owned refined run lacks a passed preservation result.")
    return {"configuration": configuration, "product_result": public}


def verify_authored_graph(expected, actual, spatial_operations):
    """Preserve the entire authored graph; ignore only declared derived rank."""

    def normalized(graph):
        graph = json.loads(json.dumps(graph))
        for node in graph["nodes"]:
            params = node["params"]
            rank = params.get("resolved_spatial_ndim")
            if (
                node["operation_id"] in spatial_operations
                and type(rank) is int
                and rank in {2, 3}
            ):
                params.pop("resolved_spatial_ndim")
        return graph

    if normalized(expected) != normalized(actual):
        raise ValueError("The executed full graph or authored parameters changed.")


def bind_additive_graph(recipe, expected, actual, added, spatial_operations):
    """Independently bind exact model keys, full additive graph and owned IDs."""
    verify_authored_graph(expected, actual, spatial_operations)
    context = recipe["context_id"]
    if type(context) is not str or len(context) < 12:
        raise ValueError("The proposal context identity is invalid.")
    mapping = {
        node["key"]: f"proposal_{context[:12]}_{node['key']}"
        for node in recipe["nodes"]
    }
    ids = tuple(added)
    if (
        not ids
        or len(ids) != len(set(ids))
        or len(mapping) != len(recipe["nodes"])
        or set(ids) != set(mapping.values())
    ):
        raise ValueError("The exact nonempty owned Add IDs do not match model keys.")
    nodes = {node["id"]: node for node in actual["nodes"]}
    for item in recipe["nodes"]:
        node = nodes[mapping[item["key"]]]
        if node["operation_id"] != item["operation_id"] or any(
            node["params"].get(key) != value for key, value in item["params"].items()
        ):
            raise ValueError("The added model operation or parameter changed.")
    edges = sorted(
        (
            "input" if edge["source"] == "input" else mapping[edge["source"]],
            mapping[edge["target"]],
            edge["source_port"],
            edge["target_port"],
        )
        for edge in recipe["connections"]
    )
    actual_edges = sorted(
        (edge["source"], edge["target"], edge["source_port"], edge["target_port"])
        for edge in actual["connections"]
        if edge["target"] in ids
    )
    if edges != actual_edges:
        raise ValueError("The added model wiring changed.")
    return OwnedReplayBinding(
        ids,
        tuple((node, node) for node in nodes if node != "input"),
        sha256(json.dumps(recipe, sort_keys=True).encode()),
    ), mapping


def run_live(
    args,
    bundle,
    baseline,
    baseline_inventory,
    qualification,
    recovery=None,
    pending_resume=None,
    *,
    feedback=USER_FEEDBACK,
    objective=ORIGINAL_GOAL,
    preservation=None,
    entry_source=None,
):
    root = args.qualified_source.resolve(strict=True)
    package = root / qualification["package_directory"]
    journal = EvidenceJournal(args.output)
    continuation = recovery or pending_resume
    state = journal.directory / "state"
    state.mkdir()
    os.environ.update(
        NAPARI_CONFIG=str(state / "napari.yaml"),
        XDG_CONFIG_HOME=str(state / "xdg"),
        NAPARI_VIPP_PIPELINE_TIMING_HISTORY_PATH=str(state / "timings.json"),
        VIPP_DISABLE_UPDATE_CHECKS="1",
    )
    journal.document("qualification-receipt.json", qualification)
    journal.document(
        "protocol.json",
        {
            "original_goal": ORIGINAL_GOAL,
            "requested_goal": objective,
            "goal_changed_from_original": objective != ORIGINAL_GOAL,
            "requested_goal_sha256": sha256(objective.encode("utf-8")),
            "requested_preservation_selection": preservation,
            "new_user_feedback": feedback,
            "own_model_baseline": getattr(bundle, "own_model_baseline", None),
            "baseline_replay": str(baseline),
            "baseline_lock_sha256": EXPECTED_BASELINE_LOCK,
            "baseline_inventory": baseline_inventory,
            "original_run_inventory": bundle.before,
            "original_finalization_inventory": bundle.finalization_before,
            "qualified_source": str(root),
            "qualified_wheel_sha256": qualification["wheel_sha256"],
            "interpreter": sys.executable,
            "python_version": sys.version,
            "dependency_versions": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "scipy", "scikit-image", "napari", "qtpy")
            },
            "maximum_additional_scientific_runs": 1,
            "maximum_logical_model_requests": 3,
            "maximum_actual_model_turns": 6,
            "maximum_refinement_seconds": 1800,
            "configured_per_attempt_timeout_seconds": 600,
            "include_input_previews": True,
            "include_output_previews": True,
            "harness_chooses_parameters": False,
            "recorded_assessment_continuation": None
            if recovery is None
            else {
                "root": str(recovery["root"]),
                "inventory": recovery["inventory"],
                "inventory_sha256": recovery["inventory_sha256"],
                "original_deadline_utc": recovery["original_deadline_utc"],
                "historical_logical_requests": 1,
                "historical_actual_model_calls": 1,
                "no_new_allowance": True,
            },
            "pending_first_request_resume": None
            if pending_resume is None
            else {**pending_resume, "root": str(pending_resume["root"])},
            "human_workflow_or_reference_read": False,
            "is_fresh_blind_investigation": False,
            "harness_file_sha256": {
                name: sha256(Path(__file__).with_name(name).read_bytes())
                for name in (
                    Path(__file__).name,
                    "replay_workflow_investigation_selection.py",
                    "run_workflow_investigation_acceptance.py",
                    "workflow_investigation_evidence.py",
                    "qualify_private.py",
                    "workflow_feedback_baseline.py",
                    "workflow_pending_request_resume.py",
                )
            },
            "entry_source_sha256": None
            if entry_source is None
            else {str(entry_source): sha256(Path(entry_source).read_bytes())},
        },
    )
    assert_private_imports(root)
    sys.path.insert(0, str(package.parent))
    importer = PrivatePackageImportGuard(
        root, module_file_sha256=qualification["module_file_sha256"]
    )
    sys.meta_path.insert(0, importer)
    sample_guard = SampleGeneratorCallGuard(package / "_sample_data.py")
    sample_guard.install()
    observers = FeedbackObservers(
        journal,
        recovery,
        feedback,
        objective,
        preservation_required=preservation is not None,
        pending=pending_resume,
    )
    try:
        import napari
        from qtpy.QtCore import QSettings, QTimer
        from qtpy.QtWidgets import QApplication

        from napari_vipp._widget import VippWidget
        from napari_vipp.core.compute import ComputeMode
        from napari_vipp.core.pipeline import SPATIAL_OPERATIONS
        from napari_vipp.core.workflow import serialize_workflow

        QSettings.setDefaultFormat(QSettings.IniFormat)
        QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(state))
        viewer = napari.Viewer(title="VIPP — one-round background refinement test")
        widget = VippWidget(
            viewer, defer_initial_run=True, initial_compute_mode=ComputeMode.CPU
        )
        widget.pipeline.reset_empty_graph()
        widget.pipeline.nodes["input"].params.update(bundle.source_params)
        widget._build_graph_from_pipeline()
        viewer.window.add_dock_widget(widget, area="bottom", name="VIPP Workflow")
        widget.run_pipeline(force_sync=True)
        assert_loaded_source(
            widget.pipeline.outputs["input"], widget.pipeline.output_states["input"]
        )
        if hasattr(bundle, "persisted_source"):
            verify_array_replay(
                bundle.persisted_source, widget.pipeline.outputs["input"]
            )
        widget._history.clear()
        widget._sync_current_workflow_tab_state()
        observers.install()
        runs, pending = [], None

        def observe_add(original):
            def call(prepared, owner):
                nonlocal pending
                if len(runs) >= 2 or pending is not None:
                    raise RuntimeError(
                        "Only baseline Add and one refinement Add are permitted."
                    )
                phase = "baseline" if not runs else "refinement"
                recipe = json.loads(dialog.json_edit.toPlainText())
                graph = serialize_workflow(prepared.candidate.to_pipeline())
                added = original(prepared, owner)
                if phase == "baseline":
                    binding = freeze_replay_binding(
                        bundle.recipe,
                        bundle.prepared_graph,
                        widget.pipeline,
                        added,
                        prepared._context.context_id,
                    )
                else:
                    binding, _mapping = bind_additive_graph(
                        recipe,
                        graph,
                        serialize_workflow(widget.pipeline),
                        added,
                        SPATIAL_OPERATIONS,
                    )
                pending = {
                    "phase": phase,
                    "recipe": recipe,
                    "graph": graph,
                    "binding": binding,
                    "key_to_node": {
                        node[
                            "key"
                        ]: f"proposal_{prepared._context.context_id[:12]}_{node['key']}"
                        for node in recipe["nodes"]
                    },
                }
                journal.document(f"{phase}/proposal.json", recipe)
                journal.document(f"{phase}/prepared-workflow.json", graph)
                journal.document(f"{phase}/binding.json", binding)
                journal.event("normal_product_add", phase=phase, added_node_ids=added)
                return added

            return call

        def observe_run(original):
            def call(added):
                nonlocal pending
                if pending is None or len(runs) >= 2:
                    raise RuntimeError("No additional scientific run is authorized.")
                item, pending = pending, None
                revision = widget._workflow_assistant_run_revision
                old_report = widget._last_execution_report
                before_serial = widget._workflow_assistant_assessment_serial
                before_pipeline = widget._pipeline_run_serial
                item["started"] = monotonic()
                runs.append(item)  # Failed attempts still consume the allowance.
                accepted = original(added)
                item["ticket"] = freeze_owned_dispatch(
                    widget, before_serial, before_pipeline, revision, old_report
                )
                journal.document(f"{item['phase']}/dispatch.json", item["ticket"])
                journal.event(
                    "normal_product_run", phase=item["phase"], accepted=accepted
                )
                return accepted

            return call

        observers._patch(widget, "_apply_workflow_proposal", observe_add)
        observers._patch(widget, "_run_workflow_assistant_plan", observe_run)
        widget._open_workflow_assistant()
        dialog = widget._workflow_assistant_dialog
        dialog.goal_edit.setPlainText(objective)
        dialog.provider_combo.setCurrentIndex(dialog.provider_combo.findData("codex"))
        dialog.codex_model_edit.setText(args.model)
        dialog.reasoning_combo.setCurrentText("high")
        dialog.images_checkbox.setChecked(True)
        dialog.output_assessment_checkbox.setChecked(True)
        dialog.initiative_slider.setValue(2)
        dialog.attempts_spin.setValue(2)
        dialog.timeout_spin.setValue(600)
        dialog._refinement.slider.setValue(1)
        dialog._investigation.checkbox.setChecked(False)
        # Settings changes intentionally invalidate prior dialogue. Attach the
        # literal new user instruction only after all such controls are set.
        dialog._clarifications = [("Additional instructions 1", feedback)]
        baseline_roles = None
        preservation_constraint, preservation_receipt = None, None
        phase, terminal, exit_code = "waiting_connection", False, 1
        started = monotonic()

        def originals_unchanged():
            own_baseline = getattr(bundle, "own_model_baseline", None)
            if own_baseline is not None:
                if (
                    run_inventory(Path(own_baseline["root"]))
                    != own_baseline["inventory"]
                    or run_inventory(Path(own_baseline["limited_root"]))
                    != own_baseline["limited_inventory"]
                ):
                    raise ValueError("The prior own-model baseline recording changed.")
            if (
                run_inventory(baseline) != baseline_inventory
                or run_inventory(bundle.root) != bundle.before
            ):
                raise ValueError("Prior locked evidence changed during refinement.")
            if run_inventory(bundle.finalization_root) != bundle.finalization_before:
                raise ValueError("Prior nomination evidence changed during refinement.")
            if (
                recovery is not None
                and run_inventory(recovery["root"]) != recovery["inventory"]
            ):
                raise ValueError("The failed original refinement evidence changed.")
            if (
                pending_resume is not None
                and run_inventory(pending_resume["root"]) != pending_resume["inventory"]
            ):
                raise ValueError("The original pending-request evidence changed.")

        def close():
            timer.stop()
            widget._discard_incomplete_startup_on_close = True
            dialog.close()
            viewer.close()
            QApplication.quit()

        def write_finish(success, reason):
            nonlocal exit_code
            exit_code = 0 if success else 1
            originals_unchanged()
            verify_receipt(args.qualification_receipt, root)
            assert_private_imports(root)
            if sample_guard.calls:
                raise ValueError("The sample-generator isolation guard was triggered.")
            preservation_record = None
            if preservation is not None and preservation_constraint is not None:
                preservation_record = preservation_outcome(
                    dialog._refinement,
                    preservation_constraint,
                    preservation_receipt,
                    run_id=runs[-1]["ticket"].run_id if success else None,
                )
                journal.document("preservation-outcome.json", preservation_record)
            if success:
                if preservation is not None and preservation_record is None:
                    raise ValueError(
                        "A protected pass has no issued product constraint."
                    )
                journal.document(
                    "ui-final-assessment-adoption.json",
                    final_assessment_adoption(dialog, widget, runs[-1]),
                )
            final_ports = capture_pipeline(journal, "final-workflow", widget.pipeline)
            journal.document(
                "final-workflow/full-workflow-ui-state.json",
                widget._batch_workflow_document(),
            )
            loop = dialog._refinement.loop
            journal.document(
                "receipt.json",
                {
                    "status": "one-product-refinement-completed"
                    if success
                    else "incomplete",
                    "reason": reason,
                    "normal_runs": len(runs),
                    "additional_run_attempts": loop.attempted_runs if loop else 0,
                    "completed_refinement_assessments": loop.completed_rounds
                    if loop
                    else 0,
                    "loop_history": loop.history if loop else [],
                    "provider_counts": observers.counts,
                    "historical_model_calls": observers.historical_model_calls,
                    "actual_new_model_calls": observers.counts["model_calls"]
                    - observers.historical_model_calls,
                    "historical_logical_requests": (
                        observers.historical_logical_requests
                    ),
                    "recorded_baseline_assessment_reused": observers.recovery_used,
                    "pending_first_logical_request_resumed": (
                        observers.pending_resume_used
                    ),
                    "baseline_role_mapping": baseline_roles,
                    "preservation": preservation_record,
                    "sample_guard": sample_guard.receipt(),
                    "prior_evidence_unchanged": True,
                    "human_workflow_or_reference_read": False,
                    "scientific_success_established": False,
                    "elapsed_seconds": monotonic() - started,
                },
            )
            if success:
                final = runs[-1]
                mapping = final["key_to_node"]
                by_node = {node: key for key, node in mapping.items()}
                roles = [
                    {
                        "role": (
                            f"model-node:{by_node[record['node_id']]}"
                            f":port:{record['port']}"
                        ),
                        "node_key": by_node[record["node_id"]],
                        "node_id": record["node_id"],
                        "port": record["port"],
                        "scientific_value_recorded": any(
                            name in record for name in ("array", "mesh", "data")
                        ),
                    }
                    for record in final_ports
                    if record["node_id"] in by_node
                ]
                journal.lock_selection(
                    {
                        "candidate_id": "refinement1",
                        "run_id": dialog._assessment.run_id,
                        "role_mapping": roles,
                        "recipe": final["recipe"],
                        "model_assessment": (
                            dialog._assessment.reply.result.assessment.to_dict()
                        ),
                        "baseline_lock_sha256": EXPECTED_BASELINE_LOCK,
                        "own_model_baseline_lock_sha256": (
                            getattr(bundle, "own_model_baseline", None) or {}
                        ).get("limited_lock_sha256"),
                        "actual_baseline_qualification": (
                            "prior own-model buffers exact; fresh full metadata"
                            if getattr(bundle, "recorded_buffers_only", False)
                            else "prior complete selected output exact replay"
                        ),
                        "preservation": preservation_record,
                        "pending_first_request_resume": None
                        if pending_resume is None
                        else {
                            "original_inventory_sha256": pending_resume[
                                "inventory_sha256"
                            ],
                            "original_deadline_utc": pending_resume[
                                "original_deadline_utc"
                            ],
                            "logical_slot_resumed": 1,
                            "new_phase_or_budget_granted": False,
                        },
                        "semantic_roles_not_inferred_by_harness": True,
                        "selection_basis": (
                            "single authorized refinement result, "
                            "not a model-ranked winner"
                        ),
                    }
                )

        def finish(success, reason):
            nonlocal terminal, exit_code
            if terminal:
                return
            terminal = True
            try:
                if (
                    continuation is not None
                    and not time.time() < continuation["original_deadline_utc"]
                ):
                    raise ValueError(
                        "The original refinement wall deadline was reached."
                    )
                write_finish(success, reason)
            except Exception as exc:
                exit_code = 1
                try:
                    journal.document(
                        "terminal-write-failure.json",
                        {
                            "status": "incomplete",
                            "type": type(exc).__name__,
                            "reason": str(exc),
                            "scientific_success_established": False,
                        },
                    )
                except Exception:
                    pass  # A locked/failed journal cannot authorize more work.
            finally:
                close()

        def complete_run(item):
            nonlocal baseline_roles
            if item.get("completed"):
                return True
            assert_owned_dispatch(widget, item["ticket"])
            status = owned_replay_completion(
                item["binding"],
                widget.pipeline,
                active=widget._active_pipeline_run_id is not None,
                pending=bool(
                    widget._pipeline_run_pending
                    or widget._pending_dirty_node_ids
                    or widget._pending_manual_node_ids
                ),
            )
            if status == "failed" or monotonic() - item["started"] > 300:
                raise ValueError("The owned normal run failed or exceeded 300 seconds.")
            if status != "ready":
                return False
            receipt = execution_completion_receipt(
                widget, item["ticket"], item["binding"]
            )
            journal.document(item["phase"] + "/owned-completion-receipt.json", receipt)
            journal.document(
                item["phase"] + "/execution-report.json", widget._last_execution_report
            )
            if item["phase"] == "baseline":
                verify_executed_replay_graph(
                    bundle.executed_graph, widget.pipeline, item["binding"]
                )
            else:
                verify_authored_graph(
                    item["graph"],
                    serialize_workflow(widget.pipeline),
                    SPATIAL_OPERATIONS,
                )
            capture_pipeline(
                journal, item["phase"] + "/completed-workflow", widget.pipeline
            )
            if item["phase"] == "baseline":
                baseline_roles = exact_baseline_outputs(
                    bundle, widget.pipeline, item["binding"]
                )
                journal.document(
                    "baseline/exact-output-verification.json", baseline_roles
                )
            item["completed"] = True
            return True

        def capture_before_assessment():
            if continuation is not None:
                anchor_continuation_loop(dialog._refinement.loop, continuation)
            if not runs or not complete_run(runs[-1]):
                raise ValueError(
                    "Cannot assess before durable owned completion recording."
                )
            if preservation is not None:
                record = preservation_outcome(
                    dialog._refinement,
                    preservation_constraint,
                    preservation_receipt,
                    run_id=runs[-1]["ticket"].run_id
                    if runs[-1]["phase"] == "refinement"
                    else None,
                )
                observers.preservation_result = record["product_result"]

        observers._patch(
            dialog._assessment,
            "start",
            lambda original: before_assessment_capture(
                original, capture_before_assessment
            ),
        )

        def tick():
            nonlocal \
                phase, \
                baseline_roles, \
                preservation_constraint, \
                preservation_receipt
            if terminal:
                return
            try:
                if (
                    continuation is not None
                    and not time.time() < continuation["original_deadline_utc"]
                ):
                    raise ValueError("The original refinement deadline was reached.")
                if monotonic() - started > 1860:
                    raise ValueError(
                        "The bounded acceptance wall deadline was reached."
                    )
                if observers.counts["model_calls"] > 6 or len(runs) > 2:
                    raise ValueError("The one-round model/run allowance was exceeded.")
                for item in runs:
                    if not complete_run(item):
                        return
                if phase == "waiting_connection":
                    if dialog._connection_busy:
                        return
                    if not dialog._codex_authenticated:
                        raise ValueError(
                            "The unchanged isolated Codex connection is unavailable."
                        )
                    request = dialog._ensure_context(dialog._answered_objective())
                    assert_feedback_request(request, feedback, objective)
                    recipe = json.loads(json.dumps(bundle.recipe))
                    recipe["context_id"] = dialog._context.context_id
                    dialog._accept_proposal(recipe)
                    if dialog._prepared is None or not dialog._prepared.can_apply:
                        raise ValueError(
                            "The unchanged baseline recipe did not pass normal checks."
                        )
                    dialog.apply_proposal()
                    if not dialog._run_added_plan():
                        raise ValueError("The normal baseline Run was denied.")
                    phase = "baseline_running"
                elif phase == "baseline_running" and runs[0].get("completed"):
                    if (
                        observers.counts["model_calls"]
                        != observers.historical_model_calls
                        or not dialog._assessment.run_id
                    ):
                        raise ValueError(
                            "Baseline reproduction lacked fresh assessment authority."
                        )
                    if preservation is not None and preservation_receipt is None:
                        preservation_constraint, preservation_receipt = (
                            configure_product_preservation(
                                dialog._refinement,
                                preservation,
                                runs[0]["key_to_node"],
                            )
                        )
                        journal.document(
                            "preservation-configuration.json", preservation_receipt
                        )
                        observers.preservation_contract = preservation_receipt[
                            "product_contract"
                        ]
                        if pending_resume is not None:
                            journal.document(
                                "pending-preservation-rebinding.json",
                                bind_resumed_preservation(
                                    pending_resume["configuration"],
                                    preservation_receipt,
                                ),
                            )
                    if not dialog._refinement.start_button.isEnabled():
                        raise ValueError(
                            "Normal one-round Start refinement is unavailable."
                        )
                    journal.event("explicit_one_round_start", user_feedback=feedback)
                    dialog._refinement.start_button.click()
                    if dialog._refinement.loop is None:
                        raise ValueError(
                            "The ordinary refinement controller did not start."
                        )
                    if continuation is not None:
                        remaining = continuation["original_deadline_utc"] - time.time()
                        loop = dialog._refinement.loop
                        loop.deadline = min(loop.deadline, monotonic() + remaining)
                        loop.started_at = loop.deadline - 1800
                        journal.document(
                            "continued-loop-budget.json",
                            {
                                "original_deadline_utc": continuation[
                                    "original_deadline_utc"
                                ],
                                "remaining_seconds": remaining,
                                "loop_deadline_monotonic": loop.deadline,
                                "model_requests_already_used": 1,
                                "actual_model_calls_already_used": (
                                    observers.historical_model_calls
                                ),
                                "maximum_additional_run_attempts": 1,
                                "maximum_total_logical_requests": 3,
                                "maximum_total_actual_model_calls": 6,
                            },
                        )
                    phase = "refining"
                elif (
                    phase == "refining"
                    and not dialog._busy
                    and not dialog._refinement.active
                ):
                    loop = dialog._refinement.loop
                    if (
                        loop.completed_rounds != 1
                        or len(runs) != 2
                        or not runs[-1].get("completed")
                    ):
                        finish(False, loop.stop_reason)
                    else:
                        finish(True, loop.stop_reason)
            except Exception as exc:
                journal.event(
                    "acceptance_failed", type=type(exc).__name__, message=str(exc)
                )
                finish(False, f"{type(exc).__name__}: {exc}")

        timer = QTimer(widget)
        timer.setInterval(100)
        timer.timeout.connect(tick)
        timer.start()
        QTimer.singleShot(0, tick)
        napari.run()
        originals_unchanged()
        return exit_code
    finally:
        observers.close()
        sample_guard.close()
        sys.meta_path.remove(importer)


def main(
    argv=None,
    *,
    feedback=USER_FEEDBACK,
    objective=ORIGINAL_GOAL,
    preservation=None,
    require_own_baseline=False,
    entry_source=None,
):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recorded-run", required=True, type=Path)
    parser.add_argument("--finalization", required=True, type=Path)
    parser.add_argument("--baseline-replay", required=True, type=Path)
    parser.add_argument("--qualified-source", required=True, type=Path)
    parser.add_argument("--qualification-receipt", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", default="")
    parser.add_argument("--continue-recorded-run", type=Path)
    parser.add_argument("--expected-continuation-inventory-sha256")
    parser.add_argument("--resume-pending-first-request", type=Path)
    parser.add_argument("--expected-pending-inventory-sha256")
    parser.add_argument("--live-approved", action="store_true")
    parser.add_argument("--own-model-baseline", type=Path)
    parser.add_argument("--own-model-baseline-lock", type=Path)
    args = parser.parse_args(argv)
    if args.resume_pending_first_request is not None and (
        args.continue_recorded_run is not None or preservation is None
    ):
        parser.error("Only this protected unsent first request can resume separately.")
    if bool(args.resume_pending_first_request) != bool(
        args.expected_pending_inventory_sha256
    ):
        parser.error("Pending-slot resumption requires its frozen inventory hash.")
    if preservation is not None and args.continue_recorded_run is not None:
        parser.error("A protected pass cannot reuse an old unprotected assessment.")
    bundle, baseline, inventory, qualification = verify_baseline(args)
    if require_own_baseline and (
        args.own_model_baseline is None or args.own_model_baseline_lock is None
    ):
        parser.error("This pass requires the pinned own-model result and limited lock.")
    if args.own_model_baseline is not None:
        if args.continue_recorded_run is not None:
            parser.error(
                "An own-model baseline is a new authorized pass, not old-turn recovery."
            )
        from workflow_feedback_baseline import adapt_own_model_baseline

        bundle = adapt_own_model_baseline(
            bundle, args.own_model_baseline, args.own_model_baseline_lock
        )
    if preservation is not None:
        if not getattr(bundle, "own_model_baseline", None):
            parser.error("A protected pass requires the pinned own-model baseline.")
        validate_preservation_selection(
            preservation, {node["key"] for node in bundle.recipe["nodes"]}
        )
        require_product_preservation_api(args.qualified_source, qualification)
    if bool(args.continue_recorded_run) != bool(
        args.expected_continuation_inventory_sha256
    ):
        parser.error(
            "Continuation requires the original run and independently frozen "
            "inventory hash."
        )
    recovery = (
        load_recorded_recovery(
            args.continue_recorded_run, args.expected_continuation_inventory_sha256
        )
        if args.continue_recorded_run is not None
        else None
    )
    pending = (
        load_pending_request(
            args.resume_pending_first_request,
            args.expected_pending_inventory_sha256,
            qualified_source=args.qualified_source,
            qualification=qualification,
            bundle=bundle,
            objective=objective,
            feedback=feedback,
            preservation=preservation,
        )
        if args.resume_pending_first_request is not None
        else None
    )
    if not args.live_approved:
        print(
            json.dumps(
                {
                    "baseline_verified": bundle.candidate_id,
                    "baseline_lock_sha256": EXPECTED_BASELINE_LOCK,
                    "qualified_wheel_sha256": qualification["wheel_sha256"],
                    "live_started": False,
                    "requested_goal": objective,
                    "goal_changed_from_original": objective != ORIGINAL_GOAL,
                    "requested_preservation_selection": preservation,
                    "continuation_verified": recovery is not None,
                    "pending_first_request_verified": pending is not None,
                    "original_deadline_utc": None
                    if pending is None
                    else pending["original_deadline_utc"],
                    "own_model_baseline_lock_sha256": (
                        getattr(bundle, "own_model_baseline", None) or {}
                    ).get("limited_lock_sha256"),
                    "persisted_baseline_buffers_including_source": (
                        getattr(bundle, "own_model_baseline", None) or {}
                    ).get("persisted_buffers_bound_including_source"),
                    "excluded_older_contextual_buffers": (
                        getattr(bundle, "own_model_baseline", None) or {}
                    ).get("older_contextual_branch_files_not_replayed_count"),
                }
            )
        )
        return 0
    if args.output is None:
        parser.error("Explicit live mode requires a fresh --output directory.")
    output = args.output.resolve()
    if any(
        output.is_relative_to(path)
        for path in (baseline, bundle.root, bundle.finalization_root)
    ):
        parser.error("New evidence must remain outside all prior frozen evidence.")
    if recovery is not None and output.is_relative_to(recovery["root"]):
        parser.error(
            "Continuation evidence must remain outside the failed original run."
        )
    if pending is not None and output.is_relative_to(pending["root"]):
        parser.error(
            "Pending-slot evidence must remain outside the original failed run."
        )
    own = getattr(bundle, "own_model_baseline", None)
    if own is not None and any(
        output.is_relative_to(Path(own[name])) for name in ("root", "limited_root")
    ):
        parser.error("New evidence must stay outside the prior own-model evidence.")
    return run_live(
        args,
        bundle,
        baseline,
        inventory,
        qualification,
        recovery,
        pending,
        feedback=feedback,
        objective=objective,
        preservation=preservation,
        entry_source=entry_source,
    )


if __name__ == "__main__":
    raise SystemExit(main())
