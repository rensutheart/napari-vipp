"""One explicit normal replay of a locked nomination; no model or new trial.

candidate0 is retained WITHOUT a duplicate Add/Run. Other nominations use a
fresh source-only native graph; no old accepted branch is claimed to exist there.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from time import monotonic
from types import SimpleNamespace

from continue_workflow_microedit_assessment import (
    ORIGINAL_INVENTORY_SHA256,
    compact,
    restore_pipeline,
)
from qualify_private import (
    PrivatePackageImportGuard,
    assert_private_imports,
    verify_receipt,
)
from replay_workflow_investigation_selection import (
    InferenceCallGuard,
    read_json,
    run_inventory,
    verify_executed_replay_graph,
)
from run_workflow_feedback_refinement import execution_completion_receipt
from run_workflow_investigation_acceptance import assert_loaded_source, capture_pipeline
from workflow_investigation_evidence import (
    EvidenceJournal,
    SampleGeneratorCallGuard,
    freeze_owned_dispatch,
    freeze_replay_binding,
    owned_replay_completion,
    read_recording_json,
    sha256,
)
from workflow_microedit_evidence import (
    assert_selected_preservation,
    exact_baseline_replay,
    recorded_port_bundle,
    verified_lock,
)


def selection_replay_required(candidate_id):
    if candidate_id not in {"candidate0", "candidate1", "candidate2"}:
        raise ValueError("Only a locked original completed candidate may be replayed.")
    return candidate_id != "candidate0"


def load_nomination(path, expected_sha):
    selection_root, lock = verified_lock(path.parent, expected_sha)
    selection = lock["selection"]
    required = selection_replay_required(selection["candidate_id"])
    original = Path(selection["original_recorded_run"]).resolve(strict=True)
    inventory = run_inventory(original)
    if (
        selection["original_inventory_sha256"] != ORIGINAL_INVENTORY_SHA256
        or sha256(compact(inventory)) != ORIGINAL_INVENTORY_SHA256
        or selection.get("recorded_outputs_only_no_final_native_replay") is not True
    ):
        raise ValueError(
            "The fixed original recording or nomination authority changed."
        )
    identity = selection["candidate_id"]
    candidate_dir = (
        original / "session/initial/candidates/candidate0"
        if not required
        else original / f"actions/action-023/candidates/{identity}"
    )
    recipe = read_json(candidate_dir / "proposal.json")
    evidence = read_json(candidate_dir / "candidate.json")
    roles = read_json(candidate_dir / "output-roles.json")
    keymap = read_json(candidate_dir / "key-to-node.json")
    if any(
        selection[key] != value
        for key, value in (
            ("recipe", recipe),
            ("evidence", evidence),
            ("key_to_node", keymap),
            ("role_mapping", roles),
        )
    ):
        raise ValueError("FINISH changed a recorded recipe, role or actual result.")
    prefix = (
        "baseline/completed-workflow"
        if not required
        else f"trials/trial-{identity[-1]:0>3}"
    )
    state = read_recording_json(original / prefix / "full-state.json")
    executed = read_json(original / prefix / "workflow.json")
    prepared = read_json(candidate_dir / "workflow.json")
    source_graph = read_json(original / "baseline/completed-workflow/workflow.json")
    params = next(n["params"] for n in source_graph["nodes"] if n["id"] == "input")
    source = Path(params["file_path"]).resolve(strict=True)
    return SimpleNamespace(
        original=original,
        inventory=inventory,
        selection_root=selection_root,
        selection_inventory=run_inventory(selection_root),
        lock_sha256=expected_sha,
        candidate_id=identity,
        recipe=recipe,
        evidence=evidence,
        roles=roles,
        keymap=keymap,
        prepared=prepared,
        executed=executed,
        prefix=prefix,
        expected=recorded_port_bundle(original, state["ports"]),
        source_params=params,
        source=source,
        source_sha256=sha256(source.read_bytes()),
        needs_replay=required,
    )


def original_constraint(bundle):
    """Recorded-cache identity only, not current native ownership."""
    from napari_vipp.core.workflow_assessment import capture_assessment_snapshot
    from napari_vipp.core.workflow_refinement import capture_preservation_constraint

    pipeline = restore_pipeline(bundle.original, "baseline/completed-workflow")
    config = read_json(bundle.original / "preservation-configuration.json")
    ids = tuple(read_json(bundle.original / "baseline/binding.json")["added_node_ids"])
    snapshot = capture_assessment_snapshot(
        pipeline, "input", ids, config["product_contract"]["baseline_run_id"]
    )
    constraint = capture_preservation_constraint(
        snapshot,
        tuple(config["primary_ref"]),
        tuple(config["region_ref"]),
        output_sharing=True,
    )
    if compact(constraint.public_contract()) != compact(config["product_contract"]):
        raise ValueError("Original protected source/grid/foreground contract changed.")
    return snapshot, constraint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--expected-lock-sha256", required=True)
    parser.add_argument("--qualified-source", type=Path, required=True)
    parser.add_argument("--qualification-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replay-approved", action="store_true")
    args = parser.parse_args(argv)
    bundle = load_nomination(args.selection_lock, args.expected_lock_sha256)
    qualification = verify_receipt(args.qualification_receipt, args.qualified_source)
    if not args.replay_approved:
        print(
            json.dumps(
                {
                    "native_started": False,
                    "candidate_id": bundle.candidate_id,
                    "needs_normal_replay": bundle.needs_replay,
                }
            )
        )
        return 0
    if (
        args.output is None
        or args.output.exists()
        or any(
            args.output.resolve().is_relative_to(p)
            for p in (bundle.original, bundle.selection_root)
        )
    ):
        raise ValueError("Final replay needs a fresh separate evidence directory.")
    journal = EvidenceJournal(args.output)
    journal.document(
        "binding.json",
        {
            "selected_lock_sha256": bundle.lock_sha256,
            "candidate_id": bundle.candidate_id,
            "original_root": str(bundle.original),
            "original_inventory": bundle.inventory,
            "original_inventory_sha256": ORIGINAL_INVENTORY_SHA256,
            "qualification": qualification,
            "script_sha256": sha256(Path(__file__).read_bytes()),
            "no_previous_branch_claimed_in_fresh_graph": True,
            "maximum_normal_runs": 1,
            "model_calls_permitted": 0,
        },
    )
    if not bundle.needs_replay:
        owned = read_json(bundle.original / "baseline/owned-completion.json")
        if owned.get("cleanup_succeeded") is not True:
            raise ValueError("The retained baseline lacks its successful original Run.")
        journal.document(
            "receipt.json",
            {
                "status": "retained-original-baseline",
                "normal_runs": 0,
                "model_calls": 0,
                "duplicate_baseline_run": False,
                "original_owned_completion": owned,
                "current_native_adoption_claimed": False,
            },
        )
        print(json.dumps({"retained_baseline": True, "normal_runs": 0}))
        return 0
    return native_replay(args, bundle, qualification, journal)


def native_replay(args, bundle, qualification, journal):
    root = args.qualified_source.resolve(strict=True)
    assert_private_imports(root)
    package = root / qualification["package_directory"]
    sys.path.insert(0, str(package.parent))
    importer = PrivatePackageImportGuard(
        root, module_file_sha256=qualification["module_file_sha256"]
    )
    sys.meta_path.insert(0, importer)
    sample, inference = (
        SampleGeneratorCallGuard(package / "_sample_data.py"),
        InferenceCallGuard(package),
    )
    state = journal.directory / "state"
    state.mkdir()
    os.environ.update(
        NAPARI_CONFIG=str(state / "napari.yaml"),
        XDG_CONFIG_HOME=str(state / "xdg"),
        NAPARI_VIPP_PIPELINE_TIMING_HISTORY_PATH=str(state / "timings.json"),
        VIPP_DISABLE_UPDATE_CHECKS="1",
    )

    def authority():
        if (
            run_inventory(bundle.original) != bundle.inventory
            or run_inventory(bundle.selection_root) != bundle.selection_inventory
            or sha256(bundle.source.read_bytes()) != bundle.source_sha256
        ):
            raise ValueError("Original selected recording, lock or source changed.")
        verify_receipt(args.qualification_receipt, root)
        assert_private_imports(root)

    try:
        sample.install()
        inference.install()
        import napari
        from qtpy.QtCore import QSettings, QTimer
        from qtpy.QtWidgets import QApplication

        from napari_vipp._widget import VippWidget
        from napari_vipp.core.compute import ComputeMode
        from napari_vipp.core.workflow_proposals import validate_proposal

        baseline_snapshot, constraint = original_constraint(bundle)
        assert_selected_preservation(
            SimpleNamespace(
                status="completed",
                evidence=bundle.evidence,
                run_id=bundle.evidence["run_id"],
            ),
            constraint.public_contract(),
        )
        QSettings.setDefaultFormat(QSettings.IniFormat)
        QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(state))
        viewer = napari.Viewer(title="VIPP — locked AI result replay (no AI requests)")
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
        if (
            widget._current_compute_request().as_dict()
            != (bundle.evidence["execution"]["compute_request"])
        ):
            raise ValueError("The final compute request differs from its actual trial.")
        widget._history.clear()
        widget._sync_current_workflow_tab_state()
        widget._open_workflow_assistant()
        dialog = widget._workflow_assistant_dialog
        dialog.provider_combo.setCurrentIndex(dialog.provider_combo.findData("example"))
        dialog.output_assessment_checkbox.setChecked(True)
        context, owner = widget._capture_workflow_proposal_context("input")
        proposal = {**bundle.recipe, "context_id": context.context_id}
        prepared = validate_proposal(proposal, context)
        added = widget._apply_workflow_proposal(prepared, owner)
        binding = freeze_replay_binding(
            bundle.recipe, bundle.prepared, widget.pipeline, added, context.context_id
        )
        journal.document("normal-add-binding.json", binding)
        initial, revision = (
            widget._last_execution_report,
            widget._workflow_assistant_run_revision,
        )
        before, pipe_before = (
            widget._workflow_assistant_assessment_serial,
            widget._pipeline_run_serial,
        )
        started, done, exit_code = monotonic(), False, 1

        def close():
            timer.stop()
            widget._discard_incomplete_startup_on_close = True
            dialog.close()
            viewer.close()
            QApplication.quit()

        def check():
            nonlocal done, exit_code
            if done:
                return
            try:
                if monotonic() - started > 300:
                    raise ValueError("The one normal replay exceeded 300 seconds.")
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
                if status == "failed":
                    raise ValueError("The normal selected-result replay failed.")
                if status != "ready":
                    return
                execution = execution_completion_receipt(widget, ticket, binding)
                verify_executed_replay_graph(bundle.executed, widget.pipeline, binding)
                verified = exact_baseline_replay(
                    bundle.expected, widget.pipeline, binding.node_mapping
                )
                snapshot, current_owner = widget._capture_workflow_assessment_snapshot(
                    added, "input"
                )
                widget._assert_workflow_assessment_current(snapshot, current_owner)
                selected_node = binding.node_mapping[
                    bundle.keymap["protected_structure"]
                ]
                ref = (selected_node, 0)
                if not any((o.node_id, o.port) == ref for o in snapshot._outputs):
                    raise ValueError(
                        "The actual bounded snapshot omitted the protected mesh input."
                    )
                protection = constraint.check_detached(baseline_snapshot, snapshot, ref)
                if not protection.passed:
                    raise ValueError("The actual final Run lost protected foreground.")
                authority()
                capture_pipeline(journal, "final-workflow", widget.pipeline)
                journal.document(
                    "execution-and-current-owner.json",
                    {
                        "execution": execution,
                        "current_snapshot_owner_verified": True,
                        "snapshot_run_id": snapshot.run_id,
                        "protection": protection.public_result(),
                        "exact_all_port_verification": verified,
                    },
                )
                role_records = [
                    {**role, "node": binding.node_mapping[bundle.keymap[role["node"]]]}
                    for role in bundle.roles
                ]
                journal.document(
                    "receipt.json",
                    {
                        "status": "exact-normal-native-replay",
                        "normal_runs": 1,
                        "model_calls": 0,
                        "roles": role_records,
                        "full_metadata_and_all_buffers_exact": True,
                        "source_guard": sample.receipt(),
                        "inference_guard": inference.receipt(),
                    },
                )
                journal.lock_selection(
                    {
                        "candidate_id": bundle.candidate_id,
                        "run_id": ticket.run_id,
                        "original_selected_lock_sha256": bundle.lock_sha256,
                        "role_mapping": role_records,
                        "recipe": bundle.recipe,
                        "selection_basis": (
                            "Replay of fixed model FINISH, no new selection"
                        ),
                        "protection": protection.public_result(),
                    }
                )
                done, exit_code = True, 0
                close()
            except Exception as exc:
                done = True
                journal.document(
                    "failure.json",
                    {
                        "type": type(exc).__name__,
                        "reason": str(exc),
                        "status": "incomplete",
                        "normal_runs": 1,
                        "model_calls": 0,
                    },
                )
                close()

        if not widget._run_workflow_assistant_plan(added):
            raise ValueError("The selected normal Run was denied.")
        ticket = freeze_owned_dispatch(widget, before, pipe_before, revision, initial)
        journal.document("normal-run-dispatch.json", ticket)
        timer = QTimer(widget)
        timer.setInterval(100)
        timer.timeout.connect(check)
        timer.start()
        QTimer.singleShot(0, check)
        napari.run()
        authority()
        return exit_code
    finally:
        inference.close()
        sample.close()
        sys.meta_path.remove(importer)


if __name__ == "__main__":
    raise SystemExit(main())
