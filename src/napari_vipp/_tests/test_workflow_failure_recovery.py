"""Owned failed-run recovery uses ordinary execution, never partial results."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from napari_vipp._tests.test_workflow_assessment_ui import finish_worker
from napari_vipp._tests.test_workflow_assessment_ui import review as review
from napari_vipp._tests.test_workflow_assessment_widget import (
    assessment_editor as assessment_editor,
)
from napari_vipp.core import pipeline as pipeline_module
from napari_vipp.core import workflow_assessment_model, workflow_proposal_codex
from napari_vipp.core.execution import execute_pipeline_request
from napari_vipp.core.mesh_refinement import MeshInputTopologyError
from napari_vipp.core.metadata import AxisMetadata, image_state_from_array
from napari_vipp.core.pipeline import SourcePayload
from napari_vipp.core.workflow_assessment_model import (
    AssessmentComparison,
    AssessmentReport,
    GoalCheck,
)
from napari_vipp.core.workflow_proposals import ProposalError, reference_proposal
from napari_vipp.ui import workflow_assistant_refinement


def _mesh_proposal(context):
    raw = reference_proposal(context)
    raw.update(summary="Create and smooth a candidate surface.", questions=[])
    raw["nodes"] = []
    for key, operation in (
        ("mask", "binary_threshold"),
        ("mesh", "mask_to_3d_mesh"),
        ("smooth", "smooth_mesh"),
    ):
        parameters = {
            spec.name: spec.default
            for spec in pipeline_module.NODE_LIBRARY_BY_ID[operation].parameters
        }
        if operation == "binary_threshold":
            parameters["threshold"] = 0.5
        raw["nodes"].append(
            {
                "key": key,
                "operation_id": operation,
                "params": parameters,
                "reason": "A synthetic candidate for ordinary-run failure coverage.",
            }
        )
    raw["connections"] = [
        {"source": source, "target": target, "source_port": 0, "target_port": 0}
        for source, target in (("input", "mask"), ("mask", "mesh"), ("mesh", "smooth"))
    ]
    return raw


def _safe_proposal(context, sigma):
    raw = reference_proposal(context)
    raw["nodes"] = [node for node in raw["nodes"] if node["key"] != "measure"]
    raw["connections"] = [
        edge for edge in raw["connections"] if edge["target"] != "measure"
    ]
    next(node for node in raw["nodes"] if node["key"] == "blur")["params"].update(
        sigma_z=sigma,
        sigma_y=sigma,
        sigma_x=sigma,
    )
    return raw


@pytest.fixture(params=["synchronous", "background"])
def recovery(review, monkeypatch, request):
    """Actual source/kernel/executor paths, with only model replies held offline."""
    widget, dialog = review.widget, review.dialog
    # These held assessment/proposal workers cover the retained connector
    # recovery path. ChatGPT investigative warm starts have separate acceptance
    # coverage; do not accidentally route this legacy fixture to that worker.
    monkeypatch.setattr(dialog._investigation, "capture", None)
    z, y, x = np.indices((12, 24, 32))
    data = (((z - 6) ** 2 / 9 + (y - 12) ** 2 / 25 + (x - 16) ** 2 / 36) < 1).astype(
        np.float32
    )
    data.setflags(write=False)
    state = image_state_from_array(
        data,
        axes=tuple(AxisMetadata(name, "space", "um", 0.45) for name in "zyx"),
    )
    for name in ("outputs", "node_outputs", "output_states", "node_output_states"):
        getattr(widget.pipeline, name).clear()
    widget.pipeline.outputs["input"] = data
    widget.pipeline.node_outputs["input"] = [data]
    widget.pipeline.output_states["input"] = state
    widget.pipeline.node_output_states["input"] = [state]
    widget.pipeline.completed_node_ids.add("input")
    review.data = data
    dialog.refresh_sources()
    dialog.images_checkbox.setChecked(True)
    continuations, pipeline_workers, requests = [], [], []
    counts = {"assessment": 0, "proposal": 0, "kernel": 0}
    monkeypatch.setattr(
        workflow_assistant_refinement,
        "QTimer",
        SimpleNamespace(singleShot=lambda _delay, call: continuations.append(call)),
    )
    widget._pipeline_thread_pool = SimpleNamespace(
        start=pipeline_workers.append,
        waitForDone=lambda *_: True,
        activeThreadCount=lambda: 0,
        clear=lambda: None,
    )

    def ordinary_run(**kwargs):
        review.runs.append(kwargs)
        method = (
            widget._run_pipeline_synchronously
            if request.param == "synchronous"
            else widget._start_background_pipeline_run
        )
        method(
            data,
            None,
            "Synthetic test",
            {"input": SourcePayload(data, name="Synthetic test", image_state=state)},
            None,
            "Synthetic test",
            ("synthetic-test",),
            set(widget._pending_dirty_node_ids),
            manual_node_ids=kwargs.get("manual_node_ids"),
        )

    monkeypatch.setattr(widget, "run_pipeline", ordinary_run)
    original = pipeline_module.NODE_LIBRARY_BY_ID["smooth_mesh"]

    def known_failure(
        mesh, iterations=10, strength=0.5, preserve_boundary=True, *, progress=None
    ):
        counts["kernel"] += 1
        # Exercise the real registered CPU dispatch with a deterministic known
        # failure. No custom operation/provider or live scientific data is used.
        raise MeshInputTopologyError("smooth_mesh", 1, "duplicate_faces", 2)

    monkeypatch.setitem(
        pipeline_module.NODE_LIBRARY_BY_ID,
        "smooth_mesh",
        replace(original, function=known_failure),
    )

    def assess(context, objective, _settings, **_kwargs):
        counts["assessment"] += 1
        public = context.public_request(objective)
        requests.append(public)
        prior = public.get("previous_assessment")
        return SimpleNamespace(
            assessment=AssessmentReport(
                context.assessment_id,
                context.run_id,
                "partly_achieved",
                "A completed candidate still needs review.",
                (GoalCheck("Separate structures", "partly_met", "Review boundaries."),),
                ("One completed mask was measured.",),
                ("Try a fresh candidate.",),
                ("This is a synthetic control, not accuracy evidence.",),
                AssessmentComparison(
                    "mixed" if prior else "first_run",
                    prior["assessment_id"] if prior else None,
                    "A completed predecessor exists."
                    if prior
                    else "No earlier assessment.",
                ),
            )
        )

    def propose(context, objective, _settings, **_kwargs):
        counts["proposal"] += 1
        requests.append(context.public_request(objective))
        return SimpleNamespace(
            proposal=_mesh_proposal(context)
            if counts["proposal"] == 1
            else _safe_proposal(context, 1.7),
            attempts=(),
        )

    monkeypatch.setattr(workflow_assessment_model, "generate_codex_assessment", assess)
    monkeypatch.setattr(workflow_proposal_codex, "generate_codex_proposal", propose)
    review.counts, review.requests = counts, requests
    review.continuations, review.pipeline_workers = continuations, pipeline_workers
    review.mode = request.param
    try:
        yield review
    finally:
        if dialog._refinement.active:
            dialog._refinement.stop("Test complete.", cancel_work=True)


def _finish_run(recovery):
    if recovery.mode == "background":
        worker = recovery.pipeline_workers.pop(0)
        result = execute_pipeline_request(worker.request)
        recovery.widget._on_background_pipeline_finished(result)


def _advance(recovery):
    assert recovery.continuations
    recovery.continuations.pop(0)()


def _initial(recovery, limit=2):
    dialog = recovery.dialog
    dialog.output_assessment_checkbox.setChecked(True)
    context, owner = recovery.widget._capture_workflow_proposal_context(
        "input", include_images=True
    )
    dialog._context, dialog._owner = context, owner
    dialog._accept_proposal(_safe_proposal(context, 1.0))
    assert dialog.apply_button.isEnabled()
    dialog.apply_button.click()
    initial_ids = dialog._added_ids
    dialog.run_button.click()
    _finish_run(recovery)
    assert dialog._assessment.run_id
    dialog._refinement.slider.setValue(limit)
    dialog._refinement.start_button.click()
    finish_worker(recovery)
    _advance(recovery)  # Initial assessment accepted; request first revision.
    return initial_ids


def _failed(recovery, limit=2):
    initial_ids = _initial(recovery, limit)
    assert recovery.dialog._job_kind == "proposal"
    finish_worker(recovery)
    _advance(recovery)  # Normal checked Add/Run; kernel raises typed failure.
    _finish_run(recovery)
    return initial_ids


def test_owned_failure_rolls_back_only_failed_add_then_repairs_and_assesses(recovery):
    initial_ids = _failed(recovery)
    dialog, widget = recovery.dialog, recovery.widget
    original_goal = dialog.goal_edit.toPlainText()
    failed = widget._workflow_assistant_failed_run
    assert failed is not None, (
        dialog._refinement.note.text(),
        widget.status_label.text(),
    )
    failed_ids = failed.revision.added_node_ids
    assert set(initial_ids) <= set(widget.pipeline.nodes)
    assert set(failed_ids) <= set(widget.pipeline.nodes)
    assert widget._workflow_assistant_assessment_run is None
    assert not dialog._assessment.run_id and not dialog._assessment.reply
    assert recovery.counts == {"assessment": 1, "proposal": 1, "kernel": 1}
    assert not recovery.workers  # No assessment or model request of partial outputs.
    cached = {
        key: widget.pipeline.outputs[key]
        for key in ("input", *initial_ids)
        if key in widget.pipeline.outputs
    }
    failure_draft = dialog._drafts[-1]
    undo_count = len(widget._history.undo_stack)
    _advance(recovery)  # Guarded rollback and fresh repair request, no Run yet.
    assert not set(failed_ids) & set(widget.pipeline.nodes)
    assert set(initial_ids) <= set(widget.pipeline.nodes)
    assert all(widget.pipeline.outputs[key] is value for key, value in cached.items())
    assert failure_draft in dialog._drafts
    assert len(widget._history.undo_stack) > undo_count
    assert len(recovery.runs) == 2
    assert dialog._busy and dialog._job_kind == "proposal"
    assert widget._workflow_assistant_assessment_run is None
    assert not dialog._assessment.preview_button.isEnabled()
    finish_worker(recovery)
    feedback_request = recovery.requests[-1]
    feedback = feedback_request["calculation_failure"]
    assert feedback["status"] == "failed_no_assessment"
    assert feedback["run_id"] == failed.run_id
    assert "result_feedback" not in feedback_request
    assert "selected_outputs" not in str(feedback)
    assert feedback_request["objective"] == original_goal
    assert feedback["failed_nodes"][0]["code"] == "duplicate_faces"
    _advance(recovery)  # Add replacement; normal Run succeeds.
    _finish_run(recovery)
    assert len(recovery.runs) == 3
    assert dialog._assessment.run_id and widget._workflow_assistant_assessment_run
    _advance(recovery)  # Request assessment only of that accepted successful Run.
    finish_worker(recovery)
    _advance(recovery)
    assert not dialog._refinement.active
    assert recovery.counts == {"assessment": 2, "proposal": 2, "kernel": 1}
    assert dialog._refinement.loop.attempted_runs == 2
    assert dialog._refinement.loop.completed_rounds == 1
    assert len(dialog._refinement.loop.failures) == 1
    assert failed.run_id not in {
        record.run_id for record in dialog._refinement._history
    }
    assessments = [
        item
        for item in recovery.requests
        if item["kind"] == "napari-vipp-output-assessment-request"
    ]
    assert len(assessments) == 2
    assert (
        assessments[1]["previous_assessment"]["assessment_id"]
        == assessments[0]["assessment_id"]
    )
    assert not set(failed_ids) & set(widget.pipeline.nodes)
    assert dialog.goal_edit.toPlainText() == original_goal


@pytest.mark.parametrize(
    "action",
    ["stop", "output_off", "input_change", "goal", "deadline", "tab", "data", "graph"],
)
def test_change_before_queued_repair_sends_nothing_and_removes_nothing(
    recovery, monkeypatch, action
):
    initial_ids = _failed(recovery)
    dialog, widget = recovery.dialog, recovery.widget
    failed = widget._workflow_assistant_failed_run
    assert failed is not None
    failed_ids = failed.revision.added_node_ids
    original_nodes = set(widget.pipeline.nodes)
    if action == "stop":
        dialog._refinement.stop_button.click()
    elif action == "output_off":
        dialog.output_assessment_checkbox.setChecked(False)
    elif action == "input_change":
        dialog.images_checkbox.setChecked(False)
    elif action == "goal":
        dialog.goal_edit.setPlainText("A different objective")
    elif action == "deadline":
        dialog._refinement.loop.deadline = 0
    elif action == "tab":
        monkeypatch.setattr(widget, "run_pipeline", lambda **_kwargs: None)
        widget._new_workflow()
    elif action == "data":
        replacement = recovery.data.copy()
        replacement[0, 0, 0] += 1
        replacement.setflags(write=False)
        widget.pipeline.outputs["input"] = replacement
        widget.pipeline.node_outputs["input"] = [replacement]
    else:
        widget.pipeline.nodes[initial_ids[0]].params["sigma_z"] = 2.3
    _advance(recovery)
    assert not dialog._refinement.active
    assert not recovery.workers
    assert recovery.counts == {"assessment": 1, "proposal": 1, "kernel": 1}
    assert len(recovery.runs) == 2
    if action != "tab":
        assert set(widget.pipeline.nodes) == original_nodes
        assert set(failed_ids) <= set(widget.pipeline.nodes)


def test_failed_attempt_exhausts_selected_run_budget_without_repair_request(recovery):
    _failed(recovery, limit=1)
    dialog = recovery.dialog
    assert not dialog._refinement.active
    assert not recovery.workers and not recovery.continuations
    assert dialog._refinement.loop.attempted_runs == 1
    assert dialog._refinement.loop.completed_rounds == 0
    assert len(dialog._refinement.loop.failures) == 1
    assert recovery.counts == {"assessment": 1, "proposal": 1, "kernel": 1}
    assert len(recovery.runs) == 2


def test_unknown_runtime_error_does_not_rollback_retry_or_assess_partial_output(
    recovery, monkeypatch
):
    spec = pipeline_module.NODE_LIBRARY_BY_ID["smooth_mesh"]

    def unknown(*_args, **_kwargs):
        recovery.counts["kernel"] += 1
        raise ValueError("Unreviewed provider failure; C:/private/source.tif")

    monkeypatch.setitem(
        pipeline_module.NODE_LIBRARY_BY_ID,
        "smooth_mesh",
        replace(spec, function=unknown),
    )
    _failed(recovery)
    dialog, widget = recovery.dialog, recovery.widget
    assert widget._workflow_assistant_failed_run is None
    assert widget._workflow_assistant_assessment_run is None
    assert (
        not dialog._refinement.active
        and not recovery.continuations
        and not recovery.workers
    )
    assert "Calculation failed" in dialog._refinement.note.text()
    assert "Automatic repair is unavailable" in dialog._refinement.note.text()
    assert recovery.counts == {"assessment": 1, "proposal": 1, "kernel": 1}
    assert all("C:/private/source.tif" not in str(item) for item in recovery.requests)


def test_stale_failed_run_ticket_cannot_be_replayed_after_rollback(recovery):
    _failed(recovery)
    widget = recovery.widget
    failed = widget._workflow_assistant_failed_run
    _advance(recovery)
    nodes = set(widget.pipeline.nodes)
    with pytest.raises(ProposalError, match="changed"):
        widget._repair_workflow_assistant_failure(failed, include_images=True)
    assert set(widget.pipeline.nodes) == nodes
    assert len(recovery.runs) == 2


@pytest.mark.parametrize("action", ["stop", "output_off", "tab", "data", "deadline"])
def test_late_repair_reply_cannot_add_or_run_after_authority_changes(
    recovery,
    monkeypatch,
    action,
):
    _failed(recovery)
    dialog, widget = recovery.dialog, recovery.widget
    _advance(recovery)  # Exact failed Add removed; replacement worker held.
    worker = recovery.workers.pop(0)
    reply = worker.call()  # Already-approved offline reply, not a live call.
    nodes = set(widget.pipeline.nodes)
    if action == "stop":
        dialog._refinement.stop_button.click()
    elif action == "output_off":
        dialog.output_assessment_checkbox.setChecked(False)
    elif action == "tab":
        monkeypatch.setattr(widget, "run_pipeline", lambda **_kwargs: None)
        widget._new_workflow()
    elif action == "data":
        replacement = recovery.data.copy()
        replacement[0, 0, 0] += 1
        replacement.setflags(write=False)
        widget.pipeline.outputs["input"] = replacement
        widget.pipeline.node_outputs["input"] = [replacement]
    else:
        dialog._refinement.loop.deadline = 0
        dialog._refinement._tick()
    dialog._ready(worker.generation, reply)
    while recovery.continuations:
        _advance(recovery)
    assert not dialog._refinement.active
    assert not recovery.workers and not dialog._busy
    assert len(recovery.runs) == 2
    assert recovery.counts == {"assessment": 1, "proposal": 2, "kernel": 1}
    if action != "tab":
        assert set(widget.pipeline.nodes) == nodes


def test_failed_addition_removal_is_undoable_and_redoable_without_run(recovery, qtbot):
    initial_ids = _failed(recovery)
    widget, dialog = recovery.widget, recovery.dialog
    failed = widget._workflow_assistant_failed_run
    failed_ids = failed.revision.added_node_ids
    failed_workflow = widget._current_history_snapshot().workflow
    _advance(recovery)
    rolled_back_workflow = widget._current_history_snapshot().workflow
    assert not set(failed_ids) & set(widget.pipeline.nodes)
    dialog._refinement.stop_button.click()
    widget.undo()
    qtbot.wait(5)
    assert widget._current_history_snapshot().workflow == failed_workflow
    assert set(failed_ids) <= set(widget.pipeline.nodes)
    assert set(initial_ids) <= set(widget.pipeline.nodes)
    assert len(recovery.runs) == 2 and recovery.counts["kernel"] == 1
    assert widget._workflow_assistant_assessment_run is None
    assert widget._workflow_assistant_failed_run is None
    widget.redo()
    qtbot.wait(5)
    assert widget._current_history_snapshot().workflow == rolled_back_workflow
    assert not set(failed_ids) & set(widget.pipeline.nodes)
    assert set(initial_ids) <= set(widget.pipeline.nodes)
    assert len(recovery.runs) == 2 and recovery.counts["kernel"] == 1
    assert widget._workflow_assistant_assessment_run is None
