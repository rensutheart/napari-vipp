"""Explicit bounded product investigation from its own accepted completed result.

Default mode verifies provenance only. Live authority is separate and explicit;
the harness supplies no scientific parameter, edit, trial, retention or FINISH.
"""

from __future__ import annotations

import argparse
import ast
import importlib.metadata
import json
import os
import sys
from pathlib import Path
from time import monotonic

from qualify_private import (
    PrivatePackageImportGuard,
    assert_private_imports,
    verify_receipt,
)
from replay_workflow_investigation_selection import run_inventory
from run_workflow_feedback_refinement import (
    configure_product_preservation,
    execution_completion_receipt,
    verify_authored_graph,
)
from run_workflow_investigation_acceptance import assert_loaded_source, capture_pipeline
from run_workflow_protected_feedback_refinement import (
    PROTECTED_FEEDBACK,
    PROTECTED_GOAL,
    PROTECTED_SELECTION,
)
from workflow_investigation_evidence import (
    EvidenceJournal,
    SampleGeneratorCallGuard,
    assert_owned_dispatch,
    freeze_owned_dispatch,
    freeze_replay_binding,
    owned_replay_completion,
    sha256,
    verify_input,
)
from workflow_microedit_evidence import (
    assert_selected_preservation,
    bind_selected_additive_replay,
    candidate_key_mapping,
    exact_baseline_replay,
    load_completed_baseline,
    offline_core_compatibility,
    recorded_port_bundle,
    retained_baseline_binding,
    runtime_baseline_roles,
)
from workflow_microedit_observers import MicroeditObservers

DEFAULT_BASELINE = Path(
    "C:/Users/rensu/Documents/VIPP/VIPP-local-tests/ai-assistant-protected-refinement-20261016/live/run2"
)
APPROVED_INPUT = Path(
    "C:/Users/rensu/Documents/VIPP/VIPP-local-tests/ai-assistant-independent-solver-20261016/input"
)
FROZEN_HELPERS = {
    "run_workflow_feedback_refinement.py": (
        "42d792df90717b79ea8d2c595a272fcf542584d9e6cd4b78df23c9c2946c8b35"
    ),
    "workflow_pending_request_resume.py": (
        "f6604590dbb858efc3633e49a14f4694e3dbfcd78259223b4f24bfac71ed05b6"
    ),
    "run_workflow_protected_feedback_refinement.py": (
        "b58809089fb0d0d44774d58fc0ed7a570d77f6ee65e0fbf27d1ff5b0de541755"
    ),
    "run_workflow_investigation_acceptance.py": (
        "528615634e1f3a6e9230897a73fae192ca55cd9d3b24a4342cb6c8fd5abc120a"
    ),
    "workflow_investigation_evidence.py": (
        "284d126c00cff96c81b2a7de4daf8b5c002d6ac19e54334556933115de1d411a"
    ),
    "replay_workflow_investigation_selection.py": (
        "b5744625c7559933a44edb46852b506754be68f32a2f7daa660b2122e8af75a2"
    ),
    "qualify_private.py": (
        "7017fb85e8856e6e7ca082118b683f5263514a8dc77ec8ebc50eb43225caec7d"
    ),
}


def source_inventory():
    paths = [
        Path(__file__),
        *(Path(__file__).with_name(name) for name in FROZEN_HELPERS),
        Path(__file__).with_name("workflow_microedit_evidence.py"),
        Path(__file__).with_name("workflow_microedit_observers.py"),
    ]
    result = {path.name: sha256(path.read_bytes()) for path in paths}
    if any(result[name] != expected for name, expected in FROZEN_HELPERS.items()):
        raise ValueError("An imported frozen recorder helper changed.")
    return result


def require_product_api(root, qualification):
    package = root / qualification["package_directory"]
    for relative, class_name, method in (
        (
            "ui/workflow_assistant_refinement.py",
            "AssistantRefinementController",
            "configure_investigative_outputs",
        ),
        (
            "ui/workflow_assistant_investigation.py",
            "AssistantInvestigationController",
            "start_refinement",
        ),
        ("core/workflow_investigation.py", "InvestigationSession", "from_completed"),
    ):
        module = ast.parse((package / relative).read_text("utf-8"))
        classes = [
            node
            for node in module.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        ]
        if len(classes) != 1 or not any(
            isinstance(node, ast.FunctionDef) and node.name == method
            for node in classes[0].body
        ):
            raise ValueError(
                "Qualification lacks the real warm-start product API: " + method
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--qualified-source", type=Path, required=True)
    parser.add_argument("--qualification-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", default="")
    parser.add_argument("--live-approved", action="store_true")
    args = parser.parse_args(argv)
    sources = source_inventory()
    input_identity = verify_input(APPROVED_INPUT)
    bundle = load_completed_baseline(args.baseline_run)
    root = args.qualified_source.resolve(strict=True)
    qualification = verify_receipt(args.qualification_receipt, root)
    require_product_api(root, qualification)
    if not args.live_approved:
        compatibility = qualified_core_preflight(
            root, qualification, bundle, args.qualification_receipt
        )
        print(
            json.dumps(
                {
                    "live_started": False,
                    "baseline_lock_sha256": bundle.lock_sha256,
                    "accepted_baseline_nodes": len(bundle.key_to_node),
                    "ports": len(bundle.expected),
                    "exact_recorded_buffers": bundle.buffer_count,
                    "full_metadata_available": True,
                    "qualified_wheel_sha256": qualification["wheel_sha256"],
                    "maximum_trial_attempts": 3,
                    "source_hashes": sources,
                    "offline_core_compatibility": compatibility,
                }
            )
        )
        return 0
    if args.output is None or args.output.exists():
        parser.error("Live mode requires a fresh explicit --output directory.")
    if any(
        args.output.resolve().is_relative_to(path)
        for path in (bundle.root, bundle.source_path.parent, APPROVED_INPUT)
    ):
        parser.error("New evidence cannot be written into an earlier scientific root.")
    return run_live(args, bundle, qualification, sources, input_identity)


def qualified_core_preflight(root, qualification, bundle, receipt_path):
    """Import only qualified bytes; rebind recorded caches without GUI or Run."""
    assert_private_imports(root)
    package = root / qualification["package_directory"]
    sys.path.insert(0, str(package.parent))
    importer = PrivatePackageImportGuard(
        root, module_file_sha256=qualification["module_file_sha256"]
    )
    sample_guard = SampleGeneratorCallGuard(package / "_sample_data.py")
    sys.meta_path.insert(0, importer)
    original_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        sample_guard.install()
        result = offline_core_compatibility(bundle, PROTECTED_GOAL, PROTECTED_FEEDBACK)
        assert_private_imports(root)
        verify_receipt(receipt_path, root)
        if sample_guard.calls:
            raise ValueError("A read-only compatibility check attempted sample access.")
        return {**result, "sample_guard": sample_guard.receipt()}
    finally:
        sample_guard.close()
        sys.dont_write_bytecode = original_bytecode
        if importer in sys.meta_path:
            sys.meta_path.remove(importer)
        sys.path.remove(str(package.parent))


def run_live(args, bundle, qualification, sources, input_identity):
    root = args.qualified_source.resolve(strict=True)
    package = root / qualification["package_directory"]
    journal = EvidenceJournal(args.output)
    state = journal.directory / "state"
    state.mkdir()
    os.environ.update(
        NAPARI_CONFIG=str(state / "napari.yaml"),
        XDG_CONFIG_HOME=str(state / "xdg"),
        NAPARI_VIPP_PIPELINE_TIMING_HISTORY_PATH=str(state / "timings.json"),
        VIPP_DISABLE_UPDATE_CHECKS="1",
    )
    journal.document("qualification-receipt.json", qualification)
    for name in sources:
        journal.artifact(
            "harness-sources/" + name, Path(__file__).with_name(name).read_bytes()
        )
    journal.document(
        "protocol.json",
        {
            "goal": PROTECTED_GOAL,
            "feedback": PROTECTED_FEEDBACK,
            "is_fresh_blind_investigation": False,
            "baseline_root": str(bundle.root),
            "baseline_lock_sha256": bundle.lock_sha256,
            "baseline_inventory": bundle.before,
            "baseline_artifact_scope": "baseline/ only; not final mixed revision",
            "baseline_full_metadata_exact_replay_required": True,
            "baseline_recorded_buffers": 41,
            "baseline_roles": bundle.roles,
            "protected_role": "band_mask",
            "protected_selection": PROTECTED_SELECTION,
            "source": input_identity,
            "source_wrapper_sha256": bundle.source_file_sha256,
            "include_input_images": True,
            "include_output_images": True,
            "limits": {
                "max_attempts": 3,
                "max_actions": 32,
                "max_model_calls": 40,
                "max_seconds": 1800,
            },
            "configured_timeout_seconds": 600,
            "model_format_attempts": 2,
            "baseline_selection_is_valid": True,
            "driver_supplies_algorithm_parameters": False,
            "reference_material_read": False,
            "harness_file_sha256": sources,
            "qualified_source": str(root),
            "qualified_wheel_sha256": qualification["wheel_sha256"],
            "interpreter": sys.executable,
            "python_version": sys.version,
            "dependency_versions": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "scipy", "scikit-image", "napari", "qtpy")
            },
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
    observers = MicroeditObservers(journal)
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
        viewer = napari.Viewer(title="VIPP — protected warm-start investigation")
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
        widget._history.clear()
        widget._sync_current_workflow_tab_state()
        observers.install()
        runs, pending_dispatch = [], None
        nominated = None
        final_item = None
        baseline_ports = None
        terminal, exit_code = False, 1
        phase, boot_started = "connection", monotonic()

        def observe_add(original):
            def call(prepared, owner):
                nonlocal pending_dispatch
                if pending_dispatch is not None or len(runs) >= 2:
                    raise RuntimeError(
                        "Only baseline and nominated final normal Adds are allowed."
                    )
                recipe = json.loads(dialog.json_edit.toPlainText())
                graph = serialize_workflow(prepared.candidate.to_pipeline())
                added = original(prepared, owner)
                if not runs:
                    binding = freeze_replay_binding(
                        bundle.recipe,
                        bundle.prepared_graph,
                        widget.pipeline,
                        added,
                        prepared._context.context_id,
                    )
                    name = "baseline"
                else:
                    if nominated is None:
                        raise ValueError("No model FINISH nominated this final Add.")
                    binding = bind_selected_additive_replay(
                        nominated,
                        graph,
                        serialize_workflow(widget.pipeline),
                        added,
                        prepared._context.context_id,
                    )
                    name = "final"
                pending_dispatch = {
                    "phase": name,
                    "recipe": recipe,
                    "prepared_graph": graph,
                    "binding": binding,
                    "key_to_node": {
                        n[
                            "key"
                        ]: f"proposal_{prepared._context.context_id[:12]}_{n['key']}"
                        for n in recipe["nodes"]
                    },
                }
                journal.document(f"{name}/proposal.json", recipe)
                journal.document(f"{name}/prepared-workflow.json", graph)
                journal.document(f"{name}/binding.json", binding)
                journal.event("normal_product_add", phase=name, added_node_ids=added)
                return added

            return call

        def observe_run(original):
            def call(added):
                nonlocal pending_dispatch
                if pending_dispatch is None or len(runs) >= 2:
                    raise ValueError("An extra normal Run has no acceptance authority.")
                item, pending_dispatch = pending_dispatch, None
                before, pipeline_before = (
                    widget._workflow_assistant_assessment_serial,
                    widget._pipeline_run_serial,
                )
                initial, revision = (
                    widget._last_execution_report,
                    widget._workflow_assistant_run_revision,
                )
                item["started"] = monotonic()
                runs.append(item)
                accepted = original(added)
                item["ticket"] = freeze_owned_dispatch(
                    widget, before, pipeline_before, revision, initial
                )
                journal.document(item["phase"] + "/dispatch.json", item["ticket"])
                journal.event(
                    "normal_product_run", phase=item["phase"], accepted=accepted
                )
                return accepted

            return call

        observers._patch(widget, "_apply_workflow_proposal", observe_add)
        observers._patch(widget, "_run_workflow_assistant_plan", observe_run)
        widget._open_workflow_assistant()
        dialog = widget._workflow_assistant_dialog
        controller = dialog._investigation
        dialog.goal_edit.setPlainText(PROTECTED_GOAL)
        dialog.provider_combo.setCurrentIndex(dialog.provider_combo.findData("codex"))
        dialog.codex_model_edit.setText(args.model)
        dialog.reasoning_combo.setCurrentText("high")
        dialog.images_checkbox.setChecked(True)
        dialog.output_assessment_checkbox.setChecked(True)
        dialog.initiative_slider.setValue(2)
        dialog.attempts_spin.setValue(2)
        dialog.timeout_spin.setValue(600)
        dialog._refinement.slider.setValue(3)
        controller.max_attempts.setValue(3)
        controller.checkbox.setChecked(False)
        dialog._clarifications = [("Additional instructions 1", PROTECTED_FEEDBACK)]

        def unchanged():
            if (
                run_inventory(bundle.root) != bundle.before
                or sha256(bundle.source_path.read_bytes()) != bundle.source_file_sha256
            ):
                raise ValueError("The locked earlier evidence/source changed.")
            if source_inventory() != sources:
                raise ValueError("A running recorder source changed.")
            verify_receipt(args.qualification_receipt, root)
            assert_private_imports(root)
            if sample_guard.calls:
                raise ValueError("Sample-generator access was attempted.")

        def close():
            watchdog.stop()
            widget._discard_incomplete_startup_on_close = True
            dialog.close()
            viewer.close()
            QApplication.quit()

        def session_record():
            session = controller.session
            return (
                None
                if session is None
                else {
                    "session_id": session.session_id,
                    "limits": session.limits,
                    "model_calls_reserved": session.model_calls_used,
                    "actions_used": session.actions_used,
                    "attempted_trials": session.attempted_trials,
                    "retained_candidate_id": session.retained_candidate_id,
                    "finished": session.finished,
                    "stop_reason": session.stop_reason,
                    "experiment_ledger": session.experiment_ledger,
                }
            )

        def finish(success, reason):
            nonlocal terminal, exit_code
            if terminal:
                return
            terminal = True
            try:
                unchanged()
                capture_pipeline(journal, "final-workflow", widget.pipeline)
                journal.document(
                    "final-workflow/full-workflow-ui-state.json",
                    widget._batch_workflow_document(),
                )
                journal.document(
                    "final-workflow/terminal-ui-status.json",
                    {
                        "status": dialog.status_label.text(),
                        "safe_attempt_trace": dialog.trace_edit.toPlainText(),
                        "investigation_active": controller.active,
                    },
                )
                session = controller.session
                role_mapping = []
                if success:
                    item = final_item
                    if item is None:
                        raise ValueError(
                            "The model nomination has no exact owned result binding."
                        )
                    execution = execution_completion_receipt(
                        widget, item["ticket"], item["binding"]
                    )
                    snapshot, owner = widget._capture_workflow_assessment_snapshot(
                        item["binding"].added_node_ids, "input"
                    )
                    widget._assert_workflow_assessment_current(snapshot, owner)
                    selected = session.selected_candidate
                    if (
                        selected is not nominated
                        or selected.status != "completed"
                        or not session.finished
                    ):
                        raise ValueError(
                            "The final replay is not the exact model nomination."
                        )
                    protection = assert_selected_preservation(
                        selected, observers.preservation_contract
                    )
                    journal.document(
                        "final/product-preservation-check.json", protection
                    )
                    manifest = observers.trial_ports_by_candidate.get(
                        selected.candidate_id
                    )
                    if not manifest:
                        raise ValueError(
                            "The selected candidate has no complete port recording."
                        )
                    verified = exact_baseline_replay(
                        recorded_port_bundle(journal.directory, manifest),
                        widget.pipeline,
                        item["binding"].node_mapping,
                    )
                    journal.document(
                        "final/exact-selected-output-verification.json", verified
                    )
                    for role, binding in selected._outputs:
                        role_mapping.append(
                            {
                                "role": role,
                                "trial_node_id": binding.node_id,
                                "final_node_id": item["binding"].node_mapping[
                                    binding.node_id
                                ],
                                "port": binding.port,
                                "selected_output_sha256": binding.fingerprint,
                                "selected_state_sha256": binding.state_fingerprint,
                                "exact_replay": True,
                            }
                        )
                    journal.document(
                        "final/owned-adoption-receipt.json",
                        {
                            "execution": execution,
                            "current_snapshot_owner_verified": True,
                            "snapshot_run_id": snapshot.run_id,
                            "selected_candidate_id": selected.candidate_id,
                            "final_normal_run_id": item["ticket"].run_id,
                        },
                    )
                journal.document("session/terminal.json", session_record())
                journal.document(
                    "receipt.json",
                    {
                        "status": "product-selected-and-exactly-replayed"
                        if success
                        else "incomplete",
                        "reason": str(reason),
                        "counts": observers.counts,
                        "normal_run_attempts": len(runs),
                        "session": session_record(),
                        "role_mapping": role_mapping,
                        "baseline_retention_is_valid": True,
                        "reference_material_read": False,
                        "baseline_retained_without_duplicate_add_run": success
                        and nominated.candidate_id == "candidate0",
                        "scientific_success_established": False,
                        "prior_evidence_unchanged": True,
                        "sample_guard": sample_guard.receipt(),
                        "selected_result_locked": success,
                    },
                )
                if success:
                    journal.lock_selection(
                        {
                            "candidate_id": nominated.candidate_id,
                            "run_id": nominated.run_id,
                            "final_normal_run_id": final_item["ticket"].run_id,
                            "recipe": nominated.recipe,
                            "evidence": nominated.evidence,
                            "key_to_node": candidate_key_mapping(nominated),
                            "role_mapping": role_mapping,
                            "output_roles": nominated.output_roles,
                            "baseline_lock_sha256": bundle.lock_sha256,
                            "selected_is_warm_start_baseline": nominated.candidate_id
                            == "candidate0",
                            "selection_basis": (
                                "explicit product model FINISH nomination, "
                                "including valid baseline retention"
                            ),
                            "semantic_roles_not_inferred_by_harness": True,
                        }
                    )
                exit_code = 0 if success else 1
            except Exception as exc:
                journal.document(
                    "terminal-write-failure.json",
                    {
                        "type": type(exc).__name__,
                        "reason": str(exc),
                        "status": "incomplete",
                    },
                )
                exit_code = 1
            finally:
                close()

        def complete(item):
            nonlocal baseline_ports
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
                raise ValueError("An owned normal Run failed or exceeded 300 seconds.")
            if status != "ready":
                return False
            journal.document(
                item["phase"] + "/owned-completion.json",
                execution_completion_receipt(widget, item["ticket"], item["binding"]),
            )
            verify_authored_graph(
                item["prepared_graph"],
                serialize_workflow(widget.pipeline),
                SPATIAL_OPERATIONS,
            )
            ports = capture_pipeline(
                journal, item["phase"] + "/completed-workflow", widget.pipeline
            )
            if item["phase"] == "baseline":
                journal.document(
                    "baseline/exact-replay.json",
                    exact_baseline_replay(
                        bundle, widget.pipeline, item["binding"].node_mapping
                    ),
                )
                baseline_ports = ports
            item["completed"] = True
            return True

        def tick():
            nonlocal phase, nominated, final_item
            if terminal:
                return
            try:
                if observers.deadline is not None and monotonic() >= observers.deadline:
                    raise ValueError("The same investigation wall deadline expired.")
                if observers.deadline is None and monotonic() - boot_started > 360:
                    raise ValueError("The bounded baseline/bootstrap deadline expired.")
                if any(
                    observers.counts[key] > maximum
                    for key, maximum in (
                        ("actions", 32),
                        ("model_calls", 40),
                        ("trial_executions", 3),
                    )
                ):
                    raise ValueError("The finite investigation allowance was exceeded.")
                for item in runs:
                    if not complete(item):
                        return
                if phase == "connection":
                    if dialog._connection_busy:
                        return
                    if not dialog._codex_authenticated:
                        raise ValueError(
                            "Existing isolated sign-in unavailable; "
                            "no repair attempted."
                        )
                    dialog._ensure_context(dialog._answered_objective())
                    recipe = {**bundle.recipe, "context_id": dialog._context.context_id}
                    dialog._accept_proposal(recipe)
                    if dialog._prepared is None or not dialog._prepared.can_apply:
                        raise ValueError(
                            "The unchanged accepted baseline failed normal checks."
                        )
                    dialog.apply_proposal()
                    if not dialog._run_added_plan():
                        raise ValueError("Baseline normal Run was denied.")
                    phase = "baseline"
                elif phase == "baseline":
                    if any(observers.counts.values()):
                        raise ValueError(
                            "A model/trial/action ran before "
                            "exact baseline verification."
                        )
                    mapping = runs[0]["binding"].node_mapping
                    roles = runtime_baseline_roles(bundle, mapping)
                    _constraint, config = configure_product_preservation(
                        dialog._refinement, PROTECTED_SELECTION, runs[0]["key_to_node"]
                    )
                    journal.document("preservation-configuration.json", config)
                    dialog._refinement.configure_investigative_outputs(
                        roles, protected_role="band_mask"
                    )
                    observers.preservation_contract = config["product_contract"]
                    observers.trial_ports_by_candidate["candidate0"] = baseline_ports
                    observers.deadline = monotonic() + 1800
                    journal.event(
                        "explicit_investigative_refinement_start",
                        goal=PROTECTED_GOAL,
                        output_roles=roles,
                        protected_role="band_mask",
                    )
                    dialog._refinement.start_button.click()
                    if controller.session is None:
                        raise ValueError(
                            "The ordinary warm-start product session did not start."
                        )
                    limits = controller.session.limits
                    if (
                        limits.max_attempts,
                        limits.max_actions,
                        limits.max_model_calls,
                        limits.max_seconds,
                    ) != (3, 32, 40, 1800):
                        raise ValueError(
                            "The product investigation limits differ "
                            "from authorization."
                        )
                    journal.document("session/started.json", session_record())
                    journal.document(
                        "session/warm-start-binding.json",
                        controller.session.warm_start_binding,
                    )
                    observers.capture_candidates(controller.session, "session/initial")
                    phase = "investigating"
                elif phase == "investigating" and not controller.active:
                    session = controller.session
                    if not session.finished:
                        finish(False, session.stop_reason or dialog.status_label.text())
                        return
                    nominated = session.selected_candidate
                    if nominated is None or nominated.status != "completed":
                        raise ValueError(
                            "FINISH did not produce a checked completed nomination."
                        )
                    journal.document(
                        "nomination-before-final-add.json",
                        {
                            "candidate_id": nominated.candidate_id,
                            "recipe": nominated.recipe,
                            "evidence": nominated.evidence,
                            "key_to_node": candidate_key_mapping(nominated),
                            "output_roles": nominated.output_roles,
                            "session": session_record(),
                        },
                    )
                    if nominated.candidate_id == "candidate0":
                        original = runs[0]
                        final_item = {
                            **original,
                            "binding": retained_baseline_binding(
                                nominated, original["binding"]
                            ),
                        }
                        phase = "retained-baseline"
                        return
                    if not dialog.apply_button.isEnabled():
                        raise ValueError(
                            "The new model nomination did not enable checked Add."
                        )
                    dialog.apply_proposal()
                    if not dialog._run_added_plan():
                        raise ValueError("The selected final normal Run was denied.")
                    phase = "final"
                    final_item = runs[-1]
                elif phase in {"final", "retained-baseline"}:
                    finish(
                        True,
                        "Explicit product nomination and complete exact normal replay.",
                    )
            except Exception as exc:
                journal.event(
                    "acceptance_failed", type=type(exc).__name__, reason=str(exc)
                )
                finish(False, f"{type(exc).__name__}: {exc}")

        watchdog = QTimer(widget)
        watchdog.setInterval(100)
        watchdog.timeout.connect(tick)
        watchdog.start()
        QTimer.singleShot(0, tick)
        napari.run()
        unchanged()
        return exit_code
    finally:
        observers.close()
        sample_guard.close()
        if importer in sys.meta_path:
            sys.meta_path.remove(importer)


if __name__ == "__main__":
    raise SystemExit(main())
