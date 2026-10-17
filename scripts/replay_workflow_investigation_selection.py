"""Replay one already recorded FINISH nomination; never request another model.

Default: verify the recorded bundle without starting Qt or calculating. An
explicit --replay-approved performs one normal checked Add/Run using the exact
original source path and qualified wheel. Existing evidence is read-only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from time import monotonic
from types import SimpleNamespace

import numpy as np
from qualify_private import (
    PrivatePackageImportGuard,
    assert_private_imports,
    verify_receipt,
)
from run_workflow_investigation_acceptance import assert_loaded_source, capture_pipeline
from workflow_investigation_evidence import (
    ORIGINAL_GOAL,
    RAW_PIXELS_SHA256,
    EvidenceJournal,
    InferenceCallGuard,
    SampleGeneratorCallGuard,
    assert_owned_dispatch,
    freeze_owned_dispatch,
    freeze_replay_binding,
    json_bytes,
    owned_replay_completion,
    public_value,
    read_recording_json,
    recipe_node_mapping,
    sha256,
    verify_executed_replay_graph,
    verify_mesh_replay,
)


def read_json(path):
    return json.loads(Path(path).read_text("utf-8"))


def run_inventory(root):
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("Recorded evidence cannot contain redirected paths.")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = sha256(path.read_bytes())
    return result


def artifact_path(root, relative):
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError("Recorded artifact path escapes its original run.")
    return path


def load_array(root, manifest):
    path = artifact_path(root, manifest["path"])
    if sha256(path.read_bytes()) != manifest["sha256"]:
        raise ValueError("A recorded scientific artifact file changed.")
    data = np.load(path, allow_pickle=False)
    if (
        str(data.dtype) != manifest["dtype"]
        or list(data.shape) != manifest["shape"]
        or sha256(data.tobytes(order="C")) != manifest["raw_sha256"]
    ):
        raise ValueError("A recorded scientific array payload changed.")
    return data


def load_recorded_selection(root, *, finalization=None):
    """Only the recorded successful FINISH chooses a candidate, never latest."""
    root = Path(root).resolve(strict=True)
    before = run_inventory(root)
    protocol = read_json(root / "protocol.json")
    if protocol["goal"] != ORIGINAL_GOAL or protocol["reference_material_read"]:
        raise ValueError("The original blinded goal/reference boundary changed.")
    events = [
        json.loads(line)
        for line in (root / "events.jsonl").read_text("utf-8").splitlines()
    ]
    previous = "0" * 64
    for number, event in enumerate(events, 1):
        if event["sequence"] != number or event["previous_event_sha256"] != previous:
            raise ValueError("The original event chain is invalid.")
        previous = sha256(json_bytes(event))
    finishes = [
        event
        for event in events
        if event["kind"] == "product_action"
        and event["details"]["action"]["action"] == "finish"
    ]
    finalization_root, finalization_before = None, None
    if finalization is not None:
        if finishes:
            raise ValueError("A recorded FINISH cannot be replaced by finalization.")
        finalization_root = Path(finalization).resolve(strict=True)
        if finalization_root.is_relative_to(root):
            raise ValueError("Finalization must remain outside the original tree.")
        finalization_before = run_inventory(finalization_root)
        binding = read_json(finalization_root / "binding.json")
        receipt = read_json(finalization_root / "receipt.json")
        original_receipt = read_json(root / "receipt.json")
        stopped = read_json(root / "session/stopped.json")
        action = read_json(finalization_root / "action.json")
        response = read_json(finalization_root / "validated-response.json")
        final_request = read_json(finalization_root / "public-request.json")
        final_events = [
            json.loads(line)
            for line in (finalization_root / "events.jsonl")
            .read_text("utf-8")
            .splitlines()
        ]
        final_previous = "0" * 64
        for number, event in enumerate(final_events, 1):
            if (
                event["sequence"] != number
                or event["previous_event_sha256"] != final_previous
            ):
                raise ValueError("The finalization event chain changed.")
            final_previous = sha256(json_bytes(event))
        nominations = [
            event
            for event in final_events
            if event["kind"] == "explicit_recorded_finalization_action"
        ]
        qualification = verify_receipt(
            finalization_root / "new-qualification-receipt.json",
            binding["new_qualification"]["qualified_source"],
        )
        calls = sorted((finalization_root / "transport").glob("call-*/request.json"))
        if (
            binding["original_run"] != str(root)
            or binding["original_inventory"] != before
            or binding["original_event_chain_sha256"] != previous
            or receipt["status"] != "explicit-model-finish"
            or receipt["actual_new_model_calls"] != 1
            or receipt["actions_used"] != 32
            or receipt["model_calls_used"] > 40
            or receipt["actions_used"] != original_receipt["counts"]["actions"] + 1
            or receipt["model_calls_used"]
            != original_receipt["counts"]["model_calls"] + 1
            or binding["original_deadline_utc"]
            != min(datetime.fromisoformat(event["utc"]).timestamp() for event in events)
            + stopped["limits"]["max_seconds"]
            or action["action"] != "finish"
            or response["action"] != action
            or qualification != binding["new_qualification"]
            or len(calls) != 1
            or read_json(calls[0])["request"] != final_request
            or final_request["objective"] != ORIGINAL_GOAL
            or final_request["session_id"] != action["session_id"]
            or final_request["session_id"] != stopped["session_id"]
            or final_request["experiment_ledger"] != stopped["experiment_ledger"]
            or final_request["image_previews"] != []
            or set(final_request["response_schema"]["actions"])
            != {"finish", "question"}
            or len(nominations) != 1
            or nominations[0]["details"]["action"] != action
            or nominations[0]["details"]["result"]
            != read_json(finalization_root / "result.json")
            or datetime.fromisoformat(nominations[0]["utc"]).timestamp()
            >= binding["original_deadline_utc"]
        ):
            raise ValueError(
                "The explicit finalization nomination/provenance did not verify."
            )
        finishes = [
            {"details": {"prefix": binding["candidate_prefix"], "action": action}}
        ]
        finish_path = finalization_root / "result.json"
    elif len(finishes) != 1:
        raise ValueError("Exactly one explicit recorded FINISH nomination is required.")
    prefix = finishes[0]["details"]["prefix"]
    if finalization is None:
        finish_path = artifact_path(root, prefix + "/result.json")
    finish = read_json(finish_path)
    candidate_id = finishes[0]["details"]["action"]["arguments"]["candidate_id"]
    if (
        finish["status"] != "ok"
        or finish["result"]["retained_candidate_id"] != candidate_id
    ):
        raise ValueError("The recorded FINISH did not nominate a completed candidate.")
    candidate_root = artifact_path(root, prefix + "/candidates/" + candidate_id)
    recipe = read_json(candidate_root / "proposal.json")
    prepared_graph = read_json(candidate_root / "workflow.json")
    evidence = read_json(candidate_root / "candidate.json")
    if (
        recipe != finish["result"]["recipe"]
        or evidence != finish["result"]["evidence"]
        or evidence["status"] != "completed"
        or evidence["candidate_id"] != candidate_id
    ):
        raise ValueError(
            "The selected recipe/evidence is not the exact completed FINISH."
        )
    trial_actions = []
    for event in events:
        if (
            event["kind"] != "product_action"
            or event["details"]["action"]["action"] != "trial_candidate"
        ):
            continue
        result = read_json(
            artifact_path(root, event["details"]["prefix"] + "/result.json")
        )
        if result.get("result", {}).get("candidate_id") == candidate_id:
            trial_actions.append(event["details"]["action"])
    if len(trial_actions) != 1 or trial_actions[0]["arguments"]["proposal"] != recipe:
        raise ValueError("The exact selected trial action/recipe is unavailable.")
    ordinal = int(candidate_id.removeprefix("candidate"))
    session_id = finishes[0]["details"]["action"]["session_id"]
    if evidence["run_id"] != f"{session_id}_{candidate_id}":
        raise ValueError("The selected candidate/execution run identity changed.")
    trials = [
        path.parent
        for path in (root / "trials").glob("trial-*/execution-result.json")
        if read_json(path)["run_id"] == ordinal
    ]
    if len(trials) != 1:
        raise ValueError(
            "The selected candidate's exact ordinary execution is ambiguous."
        )
    trial_root = trials[0]
    executed_graph = read_json(trial_root / "workflow.json")
    by_key = recipe_node_mapping(recipe, executed_graph, recipe["context_id"])
    recipe_node_mapping(recipe, prepared_graph, recipe["context_id"])
    full_state = read_recording_json(trial_root / "full-state.json")
    ports = {(item["node_id"], item["port"]): item for item in full_state["ports"]}
    roles = trial_actions[0]["arguments"]["output_roles"]
    if len({item["role"] for item in roles}) != len(roles) or {
        item["role"] for item in roles
    } != {item["role"] for item in evidence["outputs"]}:
        raise ValueError("Selected output roles are missing, duplicate or changed.")
    expected = []
    for role in roles:
        node_id = by_key[role["node"]]
        port = ports[(node_id, role["port"])]
        if full_state["node_execution_states"][node_id] != "ready":
            raise ValueError("A selected trial output was not completed.")
        if "array" in port:
            data = load_array(root, port["array"])
        elif "mesh" in port:
            mesh = port["mesh"]
            data = SimpleNamespace(
                vertices=load_array(root, mesh["vertices"]),
                faces=load_array(root, mesh["faces"]),
                face_object_ids=load_array(root, mesh["face_object_ids"]),
                object_ids=tuple(item["object_id"] for item in mesh["objects"]),
                colors=tuple(item["color"] for item in mesh["objects"]),
                state=mesh["state"],
            )
        else:
            raise ValueError(
                "This bounded recovery supports the recorded masks and meshes only."
            )
        expected.append((role, node_id, port, data))
    source_params = next(
        node["params"]
        for node in read_json(root / "starting-source/workflow.json")["nodes"]
        if node["id"] == "input"
    )
    source_literal = source_params["file_path"]
    source = Path(source_literal).resolve(strict=True)
    if source != (root / "source/input.ome.tif").resolve(strict=True):
        raise ValueError("Recovery must use the exact original source file/path.")
    wrapper = read_json(root / "source/wrapper-receipt.json")
    if (
        sha256(source.read_bytes()) != wrapper["artifact"]["sha256"]
        or wrapper["raw_pixels_sha256"] != RAW_PIXELS_SHA256
    ):
        raise ValueError("The original source file or raw pixel identity changed.")
    qualification = root / "qualification-receipt.json"
    if sha256(qualification.read_bytes()) != protocol["qualification_receipt_sha256"]:
        raise ValueError("The recorded qualified wheel receipt changed.")
    verify_receipt(qualification, protocol["qualified_source"])
    return SimpleNamespace(
        root=root,
        before=before,
        protocol=protocol,
        qualification=qualification,
        finish=finish,
        finish_sha256=sha256(finish_path.read_bytes()),
        event_chain_sha256=previous,
        candidate_id=candidate_id,
        recipe=recipe,
        evidence=evidence,
        prepared_graph=prepared_graph,
        executed_graph=executed_graph,
        expected=expected,
        source_literal=source_literal,
        source=source,
        source_params=source_params,
        finalization_root=finalization_root,
        finalization_before=finalization_before,
    )


def run_replay(bundle, output, hold_seconds):
    journal = EvidenceJournal(output)
    if bundle.finalization_root is not None:
        if run_inventory(bundle.finalization_root) != bundle.finalization_before:
            raise ValueError("The finalization nomination changed before replay.")
        journal.document(
            "finalization-provenance.json",
            {
                "root": str(bundle.finalization_root),
                "inventory": bundle.finalization_before,
            },
        )
    state = journal.directory / "state"
    state.mkdir()
    os.environ.update(
        NAPARI_CONFIG=str(state / "napari.yaml"),
        XDG_CONFIG_HOME=str(state / "xdg"),
        NAPARI_VIPP_PIPELINE_TIMING_HISTORY_PATH=str(state / "timings.json"),
        VIPP_DISABLE_UPDATE_CHECKS="1",
    )
    qualification = verify_receipt(
        bundle.qualification, bundle.protocol["qualified_source"]
    )
    root = Path(qualification["qualified_source"])
    package = root / qualification["package_directory"]
    assert_private_imports(root)
    sys.path.insert(0, str(package.parent))
    importer = PrivatePackageImportGuard(
        root, module_file_sha256=qualification["module_file_sha256"]
    )
    sample_guard, inference_guard = (
        SampleGeneratorCallGuard(package / "_sample_data.py"),
        InferenceCallGuard(package),
    )
    sys.meta_path.insert(0, importer)
    try:
        sample_guard.install()
        inference_guard.install()
        import napari
        from qtpy.QtCore import QSettings, QTimer
        from qtpy.QtWidgets import QApplication

        from napari_vipp._widget import VippWidget
        from napari_vipp.core.compute import ComputeMode
        from napari_vipp.core.workflow_proposals import validate_proposal

        QSettings.setDefaultFormat(QSettings.IniFormat)
        QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(state))
        viewer = napari.Viewer(
            title="VIPP — recorded selection replay (no AI requests)"
        )
        widget = VippWidget(
            viewer, defer_initial_run=True, initial_compute_mode=ComputeMode.CPU
        )
        widget.pipeline.reset_empty_graph()
        widget.pipeline.nodes["input"].params.update(
            source_mode="file path", file_path=bundle.source_literal, sample_name=""
        )
        widget._build_graph_from_pipeline()
        viewer.window.add_dock_widget(widget, area="bottom", name="VIPP Workflow")
        widget.run_pipeline(force_sync=True)
        assert_loaded_source(
            widget.pipeline.outputs["input"], widget.pipeline.output_states["input"]
        )
        if public_value(widget.pipeline.nodes["input"].params) != bundle.source_params:
            raise ValueError("The complete original File Source parameters changed.")
        if (
            widget._current_compute_request().as_dict()
            != bundle.evidence["execution"]["compute_request"]
        ):
            raise ValueError(
                "The replay compute request differs from the recorded trial."
            )
        widget._history.clear()
        widget._sync_current_workflow_tab_state()
        context, owner = widget._capture_workflow_proposal_context("input")
        adapted = json.loads(json.dumps(bundle.recipe))
        adapted["context_id"] = context.context_id
        prepared = validate_proposal(adapted, context)
        journal.document(
            "continuation-binding.json",
            {
                "original_run": str(bundle.root),
                "original_run_file_sha256": bundle.before,
                "finish_sha256": bundle.finish_sha256,
                "original_event_chain_sha256": bundle.event_chain_sha256,
                "source_file_sha256": sha256(bundle.source.read_bytes()),
                "raw_pixels_sha256": RAW_PIXELS_SHA256,
                "wheel_sha256": qualification["wheel_sha256"],
                "candidate_id": bundle.candidate_id,
                "recipe": bundle.recipe,
                "context_id_rebinding_only": context.context_id,
                "model_interpretation": bundle.finish["result"].get(
                    "model_interpretation"
                ),
                "limitations": bundle.finish["result"].get("limitations", []),
                "reference_material_read": False,
                "harness_file_sha256": {
                    name: sha256(Path(__file__).with_name(name).read_bytes())
                    for name in (
                        "replay_workflow_investigation_selection.py",
                        "run_workflow_investigation_acceptance.py",
                        "workflow_investigation_evidence.py",
                        "qualify_private.py",
                    )
                },
            },
        )
        journal.event("explicit_recorded_selection_add")
        added = widget._apply_workflow_proposal(prepared, owner)
        binding = freeze_replay_binding(
            bundle.recipe,
            bundle.prepared_graph,
            widget.pipeline,
            added,
            context.context_id,
        )
        journal.document("owned-replay-binding.json", binding)
        initial_report = widget._last_execution_report
        revision = widget._workflow_assistant_run_revision
        before_serial = widget._workflow_assistant_assessment_serial
        before_pipeline_serial = widget._pipeline_run_serial
        completed = False
        exit_code = 1
        started = monotonic()

        def close():
            timer.stop()
            widget._discard_incomplete_startup_on_close = True
            viewer.close()
            QApplication.quit()

        def check():
            nonlocal completed, exit_code
            if completed:
                return
            try:
                assert_owned_dispatch(widget, dispatch_ticket)
                status = owned_replay_completion(
                    binding,
                    widget.pipeline,
                    active=widget._active_pipeline_run_id is not None,
                    pending=bool(
                        widget._pipeline_run_pending
                        or widget._pending_dirty_node_ids
                        or widget._pending_manual_node_ids
                    ),
                )
                if status == "failed" or monotonic() - started > 300:
                    raise ValueError(
                        "The single normal replay failed or exceeded 300 seconds."
                    )
                if status != "ready":
                    return
                report = widget._last_execution_report
                assert_owned_dispatch(widget, dispatch_ticket, completed=True)
                journal.document("execution-report.json", report)
                verify_executed_replay_graph(
                    bundle.executed_graph, widget.pipeline, binding
                )
                role_records = []
                for role, old_node, saved, expected in bundle.expected:
                    node = binding.node_mapping[old_node]
                    port = role["port"]
                    actual = widget.pipeline.node_outputs[node][port]
                    actual_state = widget.pipeline.node_output_states[node][port]
                    if isinstance(expected, np.ndarray):
                        if (
                            expected.dtype.str != actual.dtype.str
                            or expected.shape != actual.shape
                            or expected.tobytes(order="C") != actual.tobytes(order="C")
                        ):
                            raise ValueError(
                                "Replay mask buffers differ from the selected trial."
                            )
                    else:
                        verify_mesh_replay(expected, actual)
                        if saved["mesh"]["objects"] != public_value(actual.objects):
                            raise ValueError(
                                "Replay mesh object metadata differs from "
                                "the selected trial."
                            )
                    if saved["state"] != public_value(actual_state):
                        raise ValueError(
                            "Replay complete output metadata differs from "
                            "the selected trial."
                        )
                    role_records.append(
                        {
                            "role": role["role"],
                            "trial_node_id": old_node,
                            "final_node_id": node,
                            "port": port,
                            "exact_replay": True,
                        }
                    )
                capture_pipeline(journal, "final-workflow", widget.pipeline)
                journal.document(
                    "final-workflow/full-workflow-ui-state.json",
                    widget._batch_workflow_document(),
                )
                if run_inventory(bundle.root) != bundle.before:
                    raise ValueError(
                        "The original run evidence changed during recovery."
                    )
                if (
                    bundle.finalization_root is not None
                    and run_inventory(bundle.finalization_root)
                    != bundle.finalization_before
                ):
                    raise ValueError(
                        "The finalization nomination changed during replay."
                    )
                verify_receipt(bundle.qualification, root)
                assert_private_imports(root)
                if sample_guard.calls or inference_guard.calls:
                    raise ValueError("Replay source/inference isolation was violated.")
                journal.document(
                    "isolation-receipt.json",
                    {
                        "sample_calls": sample_guard.receipt(),
                        "inference": inference_guard.receipt(),
                    },
                )
                journal.document(
                    "receipt.json",
                    {
                        "status": "recorded-selection-exactly-replayed",
                        "role_mapping": role_records,
                        "normal_runs": 1,
                        "model_calls": 0,
                        "owned_dispatch": dispatch_ticket,
                        "elapsed_seconds": monotonic() - started,
                        "original_run_unchanged": True,
                        "scientific_success_established": False,
                        "reference_material_read": False,
                    },
                )
                journal.lock_selection(
                    {
                        "candidate_id": bundle.candidate_id,
                        "run_id": bundle.evidence["run_id"],
                        "role_mapping": role_records,
                        "original_finish_sha256": bundle.finish_sha256,
                        "recipe": bundle.recipe,
                        "evidence": bundle.evidence,
                        "model_finish": bundle.finish["result"],
                    }
                )
                completed, exit_code = True, 0
                timer.stop()
                if hold_seconds:
                    viewer.title = "VIPP — verified locked replay (inspection only)"
                    QTimer.singleShot(hold_seconds * 1000, close)
                else:
                    close()
            except Exception as error:
                completed = True
                capture_pipeline(journal, "stopped-workflow", widget.pipeline)
                journal.document(
                    "failure-receipt.json",
                    {
                        "status": "incomplete",
                        "reason": f"{type(error).__name__}: {error}",
                        "normal_runs": 1,
                        "inference": inference_guard.receipt(),
                        "reference_material_read": False,
                    },
                )
                close()

        timer = QTimer(widget)
        timer.setInterval(100)
        timer.timeout.connect(check)
        journal.event(
            "explicit_single_normal_run", added_node_ids=binding.added_node_ids
        )
        if widget._run_workflow_assistant_plan(binding.added_node_ids) is False:
            raise ValueError("The ordinary checked Run was not accepted.")
        dispatch_ticket = freeze_owned_dispatch(
            widget, before_serial, before_pipeline_serial, revision, initial_report
        )
        journal.document("owned-dispatch-ticket.json", dispatch_ticket)
        timer.start()
        QTimer.singleShot(0, check)
        napari.run()
        if run_inventory(bundle.root) != bundle.before:
            raise ValueError("The original run changed after recovery.")
        verify_receipt(bundle.qualification, root)
        assert_private_imports(root)
        return exit_code
    finally:
        inference_guard.close()
        sample_guard.close()
        sys.meta_path.remove(importer)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recorded-run", type=Path, required=True)
    parser.add_argument("--finalization", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replay-approved", action="store_true")
    parser.add_argument("--hold-results-seconds", type=int, default=0)
    args = parser.parse_args(argv)
    if not 0 <= args.hold_results_seconds <= 120:
        parser.error("--hold-results-seconds must be between 0 and 120.")
    bundle = load_recorded_selection(args.recorded_run, finalization=args.finalization)
    if not args.replay_approved:
        print(
            json.dumps(
                {
                    "recorded_selection_verified": bundle.candidate_id,
                    "finish_sha256": bundle.finish_sha256,
                    "role_count": len(bundle.expected),
                    "replay_started": False,
                }
            )
        )
        return 0
    if args.output is None or args.output.resolve().is_relative_to(bundle.root):
        parser.error("A fresh --output outside the original recorded run is required.")
    if bundle.finalization_root is not None and args.output.resolve().is_relative_to(
        bundle.finalization_root
    ):
        parser.error("Replay output must also remain outside the finalization tree.")
    return run_replay(bundle, args.output, args.hold_results_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
