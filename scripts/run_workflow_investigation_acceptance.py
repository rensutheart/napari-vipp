"""Blinded acceptance of the PRODUCT investigation controller, not a solver.

Default mode verifies the two approved input files and exits without Qt or a
provider. A live run requires explicit approval and a qualified isolated source
receipt. This driver supplies no segmentation recipe or algorithm parameter.
See docs/workflow-investigation-acceptance.md for blinding and comparison gates.
"""

from __future__ import annotations

import argparse
import functools
import importlib
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
from workflow_investigation_evidence import (
    ORIGINAL_GOAL,
    RAW_PIXELS_SHA256,
    EvidenceJournal,
    SampleGeneratorCallGuard,
    assert_owned_dispatch,
    freeze_owned_dispatch,
    freeze_replay_binding,
    owned_replay_completion,
    public_value,
    sha256,
    verify_array_replay,
    verify_executed_replay_graph,
    verify_input,
    verify_mesh_objects_replay,
    verify_mesh_replay,
)

DEFAULT_INPUT = Path(
    "C:/Users/rensu/Documents/VIPP/VIPP-local-tests/"
    "ai-assistant-independent-solver-20261016/input"
)
DEFAULT_OUTPUT = Path(
    "C:/Users/rensu/Documents/VIPP/VIPP-local-tests/"
    "ai-assistant-investigation-loop-20261016"
)


def make_input_wrapper(input_directory, journal):
    """Format-only wrapper of approved pixels, with explicit OME calibration."""
    from io import BytesIO

    import numpy as np
    import tifffile

    from napari_vipp.core.io import read_image

    verify_input(input_directory)
    pixels = np.load(input_directory / "input.npy", allow_pickle=False)
    stream = BytesIO()
    tifffile.imwrite(
        stream,
        pixels,
        ome=True,
        photometric="minisblack",
        metadata={
            "axes": "ZYX",
            "PhysicalSizeX": 0.45,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": 0.45,
            "PhysicalSizeYUnit": "µm",
            "PhysicalSizeZ": 0.45,
            "PhysicalSizeZUnit": "µm",
        },
    )
    artifact = journal.artifact("source/input.ome.tif", stream.getvalue())
    source = journal.directory / artifact["path"]
    loaded = read_image(source)
    assert_loaded_source(loaded.data, loaded.image_state)
    if not np.array_equal(loaded.data, pixels):
        raise ValueError("The source wrapper did not preserve the approved raw pixels.")
    journal.document(
        "source/wrapper-receipt.json",
        {
            "artifact": artifact,
            "operation": "format-only NPY to OME-TIFF wrapper",
            "pixel_transform": "none",
            "metadata_source": "approved metadata.json only",
            "tifffile_version": tifffile.__version__,
            "ordinary_reader_state": loaded.image_state.to_dict(),
            "raw_pixels_sha256": sha256(loaded.data.tobytes(order="C")),
        },
    )
    return source


def assert_loaded_source(data, state):
    if (
        str(data.dtype) != "uint8"
        or data.shape != (12, 96, 128)
        or sha256(data.tobytes(order="C")) != RAW_PIXELS_SHA256
        or state is None
        or state.axis_order != "ZYX"
        or any(axis.confidence != "explicit" for axis in state.axes)
        or [axis.scale for axis in state.axes] != [0.45] * 3
        or [axis.translation for axis in state.axes] != [0.0] * 3
        or any(axis.unit not in {"micrometer", "µm", "um"} for axis in state.axes)
    ):
        raise ValueError(
            "The ordinary file source does not match the approved grid/pixels."
        )


def capture_pipeline(journal, prefix, pipeline):
    """Save unmodified local scientific results and their complete states."""
    import numpy as np

    from napari_vipp.core.meshes import MeshData
    from napari_vipp.core.workflow import serialize_workflow

    journal.document(f"{prefix}/workflow.json", serialize_workflow(pipeline))
    ports = []
    for node_id in pipeline.nodes:
        outputs = pipeline.node_outputs.get(node_id) or [pipeline.outputs.get(node_id)]
        states = pipeline.node_output_states.get(node_id) or [
            pipeline.output_states.get(node_id)
        ]
        for port, data in enumerate(outputs):
            state = states[port] if port < len(states) else None
            record = {"node_id": node_id, "port": port, "state": public_value(state)}
            stem = f"{prefix}/outputs/{node_id}-port-{port}"
            if isinstance(data, np.ndarray):
                record["array"] = journal.array(f"{stem}.npy", data)
            elif isinstance(data, MeshData):
                record["mesh"] = {
                    "vertices": journal.array(f"{stem}-vertices.npy", data.vertices),
                    "faces": journal.array(f"{stem}-faces.npy", data.faces),
                    "face_object_ids": journal.array(
                        f"{stem}-face-object-ids.npy", data.face_object_ids
                    ),
                    "objects": public_value(data.objects),
                    "state": data.state.to_dict(),
                }
            elif data is not None:
                record["data"] = public_value(data)
            ports.append(record)
    journal.document(
        f"{prefix}/full-state.json",
        {
            "ports": ports,
            "completed_node_ids": sorted(pipeline.completed_node_ids),
            "node_execution_states": pipeline.node_execution_states,
            "node_execution_messages": pipeline.node_execution_messages,
            "node_compute_provenance": pipeline.node_compute_provenance,
            "node_cache_lineage": pipeline.node_cache_lineage,
        },
    )
    return ports


class ProductObservers:
    """Observe existing boundaries without changing their arguments/results."""

    def __init__(self, journal):
        self.journal = journal
        self.counts = {
            "logical_requests": 0,
            "model_calls": 0,
            "actions": 0,
            "trial_executions": 0,
        }
        self._restore = []
        self.sessions = []
        self._latest_trial_ports = None
        self._latest_trial_graph = None
        self.trial_ports_by_candidate = {}
        self.trial_graphs_by_candidate = {}

    def _patch(self, owner, name, replacement):
        original = getattr(owner, name)
        self._restore.append((owner, name, original))
        setattr(owner, name, replacement(original))

    def install(self):
        from napari_vipp.core import workflow_investigation as engine
        from napari_vipp.core import workflow_investigation_model as provider
        from napari_vipp.core import workflow_proposal_codex as codex

        journal, counts = self.journal, self.counts

        def provider_observer(original):
            @functools.wraps(original)
            def call(request, settings, **kwargs):
                counts["logical_requests"] += 1
                prefix = f"requests/request-{counts['logical_requests']:03d}"
                journal.document(f"{prefix}/public-request.json", request)
                journal.document(f"{prefix}/settings.json", settings)
                previews = kwargs.get("image_previews", ())
                manifests = journal.previews(prefix, previews)
                journal.event("model_request", prefix=prefix, previews=manifests)
                try:
                    result = original(request, settings, **kwargs)
                except Exception as exc:
                    journal.event(
                        "model_request_failed",
                        type=type(exc).__name__,
                        message=str(exc),
                    )
                    raise
                journal.document(f"{prefix}/validated-response.json", result)
                return result

            return call

        def transport_observer(original):
            @functools.wraps(original)
            def call(
                request, previews, settings, runtime, previous, feedback, *a, **kw
            ):
                counts["model_calls"] += 1
                prefix = f"transport/call-{counts['model_calls']:03d}"
                journal.document(
                    f"{prefix}/request.json",
                    {"request": request, "previous": previous, "feedback": feedback},
                )
                journal.previews(prefix, previews)
                contract = kw.get("investigation_contract")
                if contract is not None:
                    journal.document(f"{prefix}/native-contract.json", contract)
                journal.event("actual_model_call", prefix=prefix)
                try:
                    result = original(
                        request,
                        previews,
                        settings,
                        runtime,
                        previous,
                        feedback,
                        *a,
                        **kw,
                    )
                except Exception as exc:
                    journal.event(
                        "actual_model_call_failed",
                        prefix=prefix,
                        type=type(exc).__name__,
                        message=str(exc),
                    )
                    raise
                journal.document(f"{prefix}/native-response.json", result)
                return result

            return call

        def rpc_observer(original):
            @functools.wraps(original)
            def call(rpc, method, params, *args, **kwargs):
                if method in {"thread/start", "turn/start"}:
                    journal.event("native_rpc_request", method=method, params=params)
                result = original(rpc, method, params, *args, **kwargs)
                if method in {"thread/start", "turn/start"}:
                    journal.event("native_rpc_response", method=method, result=result)
                return result

            return call

        def action_observer(original):
            @functools.wraps(original)
            def call(session, action):
                if not any(item is session for item in self.sessions):
                    self.sessions.append(session)
                counts["actions"] += 1
                prefix = f"actions/action-{counts['actions']:03d}"
                journal.document(f"{prefix}/action.json", action)
                journal.event("product_action", prefix=prefix, action=action)
                executions_before = counts["trial_executions"]
                try:
                    result = original(session, action)
                except Exception as exc:
                    journal.event(
                        "product_action_failed",
                        prefix=prefix,
                        type=type(exc).__name__,
                        message=str(exc),
                    )
                    self.capture_candidates(session, prefix)
                    raise
                public = result.public_result()
                journal.document(f"{prefix}/result.json", public)
                if counts["trial_executions"] == executions_before + 1:
                    candidate_id = public.get("result", {}).get("candidate_id")
                    if (
                        public.get("status") == "ok"
                        and public.get("result", {}).get("status") == "completed"
                        and candidate_id
                        and self._latest_trial_ports is not None
                        and self._latest_trial_graph is not None
                    ):
                        self.trial_ports_by_candidate[candidate_id] = (
                            self._latest_trial_ports
                        )
                        self.trial_graphs_by_candidate[candidate_id] = (
                            self._latest_trial_graph
                        )
                journal.previews(f"{prefix}/result-previews", result.image_previews)
                self.capture_candidates(session, prefix)
                journal.event(
                    "product_action_returned",
                    prefix=prefix,
                    status=public.get("status"),
                )
                if not session.finished and not session.stop_reason:
                    try:
                        next_request = session.public_request()
                    except Exception as exc:
                        journal.event(
                            "next_request_capture_failed",
                            prefix=prefix,
                            action_returned=True,
                            type=type(exc).__name__,
                            message=str(exc),
                        )
                        raise
                    journal.document(
                        f"{prefix}/session-request-after.json", next_request
                    )
                else:
                    journal.event(
                        "session_terminal",
                        finished=session.finished,
                        stop_reason=session.stop_reason,
                        retained_candidate_id=session.retained_candidate_id,
                    )
                return result

            return call

        def execution_observer(original):
            @functools.wraps(original)
            def call(request, *args, **kwargs):
                counts["trial_executions"] += 1
                self._latest_trial_ports = None
                self._latest_trial_graph = None
                prefix = f"trials/trial-{counts['trial_executions']:03d}"
                journal.document(f"{prefix}/workflow-request.json", request.workflow)
                try:
                    result = original(request, *args, **kwargs)
                except Exception as exc:
                    journal.event(
                        "trial_execution_failed",
                        prefix=prefix,
                        type=type(exc).__name__,
                        message=str(exc),
                    )
                    raise
                journal.document(
                    f"{prefix}/execution-result.json",
                    {
                        "run_id": result.run_id,
                        "workflow": result.workflow,
                        "error": result.error,
                        "cancelled": result.cancelled,
                        "source_revisions": result.source_revisions,
                        "execution_report": result.execution_report,
                        "failure": result.failure,
                    },
                )
                if result.pipeline is not None:
                    from napari_vipp.core.snapshots import GraphSnapshot

                    # Candidate.graph is the prepared PRE-Run recipe. Ordinary
                    # execution may materialize persisted runtime parameters
                    # such as resolved_spatial_ndim. Capture the actual POST-Run
                    # graph here, alongside these exact executed port buffers,
                    # without excluding or normalizing any parameter.
                    self._latest_trial_graph = GraphSnapshot.from_pipeline(
                        result.pipeline
                    )
                    self._latest_trial_ports = capture_pipeline(
                        journal, prefix, result.pipeline
                    )
                return result

            return call

        self._patch(provider, "generate_codex_investigation_action", provider_observer)
        self._patch(codex, "_call_codex", transport_observer)
        self._patch(codex._Rpc, "request", rpc_observer)
        self._patch(engine.InvestigationSession, "perform", action_observer)
        self._patch(engine, "execute_pipeline_request", execution_observer)

    def capture_candidates(self, session, prefix):
        """Implemented against the engine's explicit retained candidate API."""
        for candidate in session.candidates:
            identity = candidate.candidate_id
            marker = f"{prefix}/candidates/{identity}"
            self.journal.document(f"{marker}/candidate.json", candidate.evidence)
            self.journal.document(f"{marker}/proposal.json", candidate.recipe)
            self.journal.document(f"{marker}/output-roles.json", candidate.output_roles)
            if identity in self.trial_ports_by_candidate:
                self.journal.document(
                    f"{marker}/trial-output-manifest.json",
                    self.trial_ports_by_candidate[identity],
                )
            if candidate.graph is not None:
                from napari_vipp.core.workflow import serialize_workflow

                self.journal.document(
                    f"{marker}/workflow.json",
                    serialize_workflow(candidate.graph.to_pipeline()),
                )

    def trial_port(self, candidate_id, node_id, port):
        matches = [
            record
            for record in self.trial_ports_by_candidate.get(candidate_id, ())
            if record["node_id"] == node_id and record["port"] == port
        ]
        if len(matches) != 1:
            raise ValueError(
                "The selected candidate has no exact executed port manifest."
            )
        return matches[0]

    def trial_graph(self, candidate_id):
        """Return only the immutable graph captured for this completed trial."""
        graph = self.trial_graphs_by_candidate.get(candidate_id)
        if graph is None:
            raise ValueError(
                "The selected candidate has no exact executed graph snapshot."
            )
        return graph

    def close(self):
        for owner, name, original in reversed(self._restore):
            setattr(owner, name, original)
        self._restore.clear()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--qualified-source", type=Path)
    result.add_argument("--qualification-receipt", type=Path)
    result.add_argument("--live-approved", action="store_true")
    result.add_argument("--model", default="")
    result.add_argument("--reasoning", default="high")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    identity = verify_input(args.input.resolve())
    if not args.live_approved:
        print(json.dumps({"input_verified": identity, "live_run_started": False}))
        return 0
    if not args.qualified_source or not args.qualification_receipt:
        raise ValueError(
            "Live acceptance needs an isolated source and qualification receipt."
        )
    qualification = args.qualification_receipt.resolve().read_bytes()
    qualified_root = args.qualified_source.resolve()
    receipt = verify_receipt(args.qualification_receipt, qualified_root)
    assert_private_imports(qualified_root)
    package_root = (qualified_root / receipt["package_directory"]).parent
    sys.path.insert(0, str(package_root))
    import_guard = PrivatePackageImportGuard(
        qualified_root, module_file_sha256=receipt["module_file_sha256"]
    )
    sys.meta_path.insert(0, import_guard)
    sample_guard = SampleGeneratorCallGuard(
        qualified_root / receipt["package_directory"] / "_sample_data.py"
    )
    try:
        sample_guard.install()
        package = importlib.import_module("napari_vipp")
        package_path = Path(package.__file__).resolve()
        assert_private_imports(qualified_root)
        journal = EvidenceJournal(args.output.resolve())
        journal.artifact("qualification-receipt.json", qualification)
        journal.document(
            "protocol.json",
            {
                "input": identity,
                "qualified_source": str(qualified_root),
                "qualified_package_file": str(package_path),
                "qualification_receipt_sha256": sha256(qualification),
                "reference_material_read": False,
                "human_recipe_supplied": False,
                "sample_generator_call_guard": sample_guard.receipt(),
                "driver_supplies_algorithm_parameters": False,
                "goal": ORIGINAL_GOAL,
                "interpreter": sys.executable,
                "python_version": sys.version,
                "harness_file_sha256": {
                    name: sha256(Path(__file__).with_name(name).read_bytes())
                    for name in (
                        "run_workflow_investigation_acceptance.py",
                        "workflow_investigation_evidence.py",
                        "qualify_private.py",
                    )
                },
            },
        )
        exit_code, selection = run_live(args, journal)
        verify_receipt(args.qualification_receipt, qualified_root)
        assert_private_imports(qualified_root)
        journal.document(
            "source/sample-call-guard-receipt.json", sample_guard.receipt()
        )
        if sample_guard.calls:
            raise ValueError("A forbidden sample-module function call was attempted.")
        journal.event("post_run_private_package_reverified")
        if selection is not None:
            journal.lock_selection(selection)
        return exit_code
    finally:
        sample_guard.close()
        if import_guard in sys.meta_path:
            sys.meta_path.remove(import_guard)


def run_live(args, journal):
    """Drive Goal Send and the selected plan's ordinary final Add/Run only."""
    state_directory = journal.directory / "state"
    state_directory.mkdir()
    os.environ["NAPARI_CONFIG"] = str(state_directory / "napari.yaml")
    os.environ["XDG_CONFIG_HOME"] = str(state_directory / "xdg")
    os.environ["NAPARI_VIPP_PIPELINE_TIMING_HISTORY_PATH"] = str(
        state_directory / "timings.json"
    )
    os.environ["VIPP_DISABLE_UPDATE_CHECKS"] = "1"

    import napari
    import numpy as np
    from qtpy.QtCore import QSettings, QTimer
    from qtpy.QtWidgets import QApplication

    from napari_vipp._widget import VippWidget
    from napari_vipp.core.compute import ComputeMode

    source = make_input_wrapper(args.input.resolve(), journal)
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(state_directory))
    viewer = napari.Viewer(title="VIPP — blinded product investigation acceptance")
    widget = VippWidget(
        viewer, defer_initial_run=True, initial_compute_mode=ComputeMode.CPU
    )
    widget.pipeline.reset_empty_graph()
    widget.pipeline.nodes["input"].params.update(
        source_mode="file path", file_path=str(source), sample_name=""
    )
    widget._build_graph_from_pipeline()
    viewer.window.add_dock_widget(widget, area="bottom", name="VIPP Workflow")
    widget.run_pipeline(force_sync=True)
    assert_loaded_source(
        widget.pipeline.outputs["input"], widget.pipeline.output_states["input"]
    )
    capture_pipeline(journal, "starting-source", widget.pipeline)
    widget._history.clear()
    widget._sync_current_workflow_tab_state()
    observers = ProductObservers(journal)
    observers.install()
    assert_private_imports(args.qualified_source)
    widget._open_workflow_assistant()
    dialog = widget._workflow_assistant_dialog
    controller = dialog._investigation
    dialog.goal_edit.setPlainText(ORIGINAL_GOAL)
    dialog.provider_combo.setCurrentIndex(dialog.provider_combo.findData("codex"))
    dialog.codex_model_edit.setText(args.model)
    dialog.reasoning_combo.setCurrentText(args.reasoning)
    dialog.images_checkbox.setChecked(True)
    dialog.output_assessment_checkbox.setChecked(True)
    dialog.initiative_slider.setValue(2)
    controller.checkbox.setChecked(True)
    controller.max_attempts.setValue(5)
    started, terminal, final_running = monotonic(), False, False
    exit_code = 1
    pending_selection = None
    final_binding = None
    final_replay_run_id = None
    dispatch_ticket = None

    def capture_session(phase):
        session = controller.session
        if session is None:
            return
        journal.document(
            f"session/{phase}.json",
            {
                "session_id": session.session_id,
                "limits": session.limits,
                "request_timeout_seconds": dialog.timeout_spin.value(),
                "model_calls_reserved": session.model_calls_used,
                "actions_used": session.actions_used,
                "retained_candidate_id": session.retained_candidate_id,
                "finished": session.finished,
                "stop_reason": session.stop_reason,
                "experiment_ledger": session.experiment_ledger,
            },
        )

    def finish_error(message):
        nonlocal terminal
        if terminal:
            return
        terminal = True
        controller.cancel("The acceptance run stopped.")
        observers.close()
        capture_session("stopped")
        capture_pipeline(journal, "stopped-workflow", widget.pipeline)
        journal.document(
            "receipt.json",
            {
                "status": "incomplete",
                "reason": str(message),
                "counts": observers.counts,
                "elapsed_seconds": monotonic() - started,
                "reference_material_read": False,
                "selected_result_locked": False,
                "scientific_success_established": False,
            },
        )
        close()

    def close():
        watchdog.stop()
        dialog.close()
        widget._discard_incomplete_startup_on_close = True
        viewer.close()
        QApplication.quit()

    def finish_success():
        nonlocal terminal, exit_code, pending_selection
        if terminal:
            return
        try:
            session, selected = (
                controller.session,
                controller.session.selected_candidate,
            )
            role_mapping = []
            if (
                final_binding is None
                or final_replay_run_id is None
                or dispatch_ticket is None
                or selected is None
                or selected.status != "completed"
                or not selected._outputs
            ):
                raise ValueError("No exact owned completed replay run was accepted.")
            assert_owned_dispatch(widget, dispatch_ticket, completed=True)
            journal.document(
                "final-execution-report.json", widget._last_execution_report
            )
            journal.document(
                "final-owned-dispatch-receipt.json",
                {
                    "owned_dispatch": dispatch_ticket,
                    "fresh_cleanup_succeeded": True,
                    "active_execution": widget._active_pipeline_run_id is not None,
                    "pending_execution": bool(
                        widget._pipeline_run_pending
                        or widget._pending_dirty_node_ids
                        or widget._pending_manual_node_ids
                    ),
                },
            )
            verify_executed_replay_graph(
                observers.trial_graph(selected.candidate_id),
                widget.pipeline,
                final_binding,
            )
            replay_ids = final_binding.node_mapping
            for role, binding in selected._outputs:
                node_id = replay_ids[binding.node_id]
                actual = widget.pipeline.node_outputs[node_id][binding.port]
                actual_state = widget.pipeline.node_output_states[node_id][binding.port]
                expected = binding.data
                if isinstance(expected, np.ndarray):
                    verify_array_replay(expected, actual)
                elif hasattr(expected, "vertices") and hasattr(expected, "faces"):
                    verify_mesh_replay(expected, actual)
                    saved = observers.trial_port(
                        selected.candidate_id, binding.node_id, binding.port
                    )
                    verify_mesh_objects_replay(saved["mesh"]["objects"], actual.objects)
                elif public_value(expected) != public_value(actual):
                    raise ValueError(
                        "Final replay output differs from the selected trial."
                    )
                if public_value(binding.state) != public_value(actual_state):
                    raise ValueError(
                        "Final replay metadata differs from the selected trial."
                    )
                role_mapping.append(
                    {
                        "role": role,
                        "trial_node_id": binding.node_id,
                        "final_node_id": node_id,
                        "port": binding.port,
                        "selected_output_sha256": binding.fingerprint,
                        "selected_state_sha256": binding.state_fingerprint,
                        "exact_replay": True,
                    }
                )
            capture_pipeline(journal, "final-workflow", widget.pipeline)
            journal.document(
                "final-workflow/full-workflow-ui-state.json",
                widget._batch_workflow_document(),
            )
            observers.close()
            capture_session("finished")
            journal.document(
                "receipt.json",
                {
                    "status": "product-selected-and-replayed",
                    "counts": observers.counts,
                    "candidate_attempts": len(session.candidates),
                    "model_calls_reserved": session.model_calls_used,
                    "actions_used": session.actions_used,
                    "limits": session.limits,
                    "elapsed_seconds": monotonic() - started,
                    "role_mapping": role_mapping,
                    "final_replay_run_id": final_replay_run_id,
                    "owned_dispatch": dispatch_ticket,
                    "reference_material_read": False,
                    "scientific_success_established": False,
                    "scientific_limitations": (
                        "Model selection and exact replay do not establish "
                        "mask completeness or accuracy."
                    ),
                },
            )
            pending_selection = {
                "candidate_id": selected.candidate_id,
                "run_id": selected.run_id,
                "role_mapping": role_mapping,
                "recipe": selected.recipe,
                "evidence": selected.evidence,
                "stop_reason": session.stop_reason or "Explicit model finish action.",
            }
            terminal, exit_code = True, 0
            close()
        except Exception as error:
            finish_error(f"{type(error).__name__}: {error}")

    def final_add_and_run():
        nonlocal final_running, final_binding, dispatch_ticket
        if terminal or final_running:
            return
        if not dialog.apply_button.isEnabled():
            finish_error("The selected result did not produce an eligible checked Add.")
            return
        try:
            journal.event("explicit_final_add")
            prepared = dialog._prepared
            dialog.apply_button.click()
            if not dialog._added_ids or not dialog.run_button.isEnabled():
                raise ValueError("Final Add did not enable an ordinary explicit Run.")
            final_binding = freeze_replay_binding(
                controller.session.selected_candidate.recipe,
                controller.session.selected_candidate.graph,
                widget.pipeline,
                tuple(dialog._added_ids),
                prepared._context.context_id,
            )
            journal.document("final-owned-replay-binding.json", final_binding)
            initial_report = widget._last_execution_report
            revision = widget._workflow_assistant_run_revision
            before_serial = widget._workflow_assistant_assessment_serial
            before_pipeline_serial = widget._pipeline_run_serial
            final_running = True
            journal.event(
                "explicit_final_normal_run", added_node_ids=final_binding.added_node_ids
            )
            dialog.run_button.click()
            dispatch_ticket = freeze_owned_dispatch(
                widget, before_serial, before_pipeline_serial, revision, initial_report
            )
            journal.document("final-owned-dispatch-ticket.json", dispatch_ticket)
        except Exception as error:
            finish_error(f"Final Add/Run failed: {type(error).__name__}: {error}")

    def on_ready(_epoch, kind, _result):
        if kind == "action" and controller.session.finished and not controller.active:
            QTimer.singleShot(0, final_add_and_run)

    def tick():
        nonlocal final_replay_run_id
        if terminal:
            return
        if monotonic() - started > 1850:
            finish_error("The product acceptance safety deadline was reached.")
        elif final_running:
            try:
                assert_owned_dispatch(widget, dispatch_ticket)
                completion = owned_replay_completion(
                    final_binding,
                    widget.pipeline,
                    active=widget._active_pipeline_run_id is not None,
                    pending=bool(
                        widget._pipeline_run_pending
                        or widget._pending_dirty_node_ids
                        or widget._pending_manual_node_ids
                    ),
                )
                if completion == "ready":
                    snapshot, owner = widget._capture_workflow_assessment_snapshot(
                        final_binding.added_node_ids, "input"
                    )
                    widget._assert_workflow_assessment_current(snapshot, owner)
                    final_replay_run_id = snapshot.run_id
                    finish_success()
                elif completion == "failed":
                    finish_error("Final normal replay failed; see execution state.")
            except (ValueError, RuntimeError, TypeError) as error:
                finish_error("Final normal replay could not be verified: " + str(error))
        elif controller.session is not None and not controller.active:
            if not controller.session.finished:
                finish_error(dialog.status_label.text())

    def start():
        if dialog._connection_busy:
            QTimer.singleShot(100, start)
            return
        if not dialog._codex_authenticated:
            finish_error(
                "Existing ChatGPT connection was not authenticated; "
                "no repair attempted."
            )
            return
        journal.event("explicit_goal_start", goal=ORIGINAL_GOAL)
        dialog.prepare_button.click()
        if controller.session is None:
            finish_error(dialog.status_label.text())
        else:
            capture_session("started")

    controller.signals.ready.connect(on_ready)
    controller.signals.failed.connect(lambda _epoch, error: finish_error(str(error)))
    watchdog = QTimer(dialog)
    watchdog.setInterval(250)
    watchdog.timeout.connect(tick)
    watchdog.start()
    QTimer.singleShot(100, start)
    napari.run()
    return exit_code, pending_selection


if __name__ == "__main__":
    raise SystemExit(main())
