"""Run/session binding for output assessment; no model or live GUI requests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from napari.components import ViewerModel
from qtpy.QtWidgets import QCheckBox

from napari_vipp._widget import VippWidget
from napari_vipp.core import workflow_assessment
from napari_vipp.core.compute import ComputeMode
from napari_vipp.core.execution import PipelineRunResult, execute_pipeline_request
from napari_vipp.core.metadata import AxisMetadata, image_state_from_array
from napari_vipp.core.pipeline import SourcePayload
from napari_vipp.core.workflow_assessment import AssessmentError
from napari_vipp.core.workflow_proposals import (
    ProposalError,
    reference_proposal,
    validate_proposal,
)


@pytest.fixture
def assessment_editor(qtbot, monkeypatch):
    widget = VippWidget(
        ViewerModel(), defer_initial_run=True, initial_compute_mode=ComputeMode.CPU
    )
    qtbot.addWidget(widget)
    monkeypatch.setattr(widget, "_confirm_close_dirty_workflow_tabs", lambda: True)
    widget.pipeline.reset_empty_graph()
    widget._build_graph_from_pipeline()
    y, x = np.indices((24, 32))
    data = (((y - 12) ** 2 + (x - 16) ** 2) < 25).astype(np.float32).copy()
    data.flags.writeable = False
    state = image_state_from_array(
        data, axes=(AxisMetadata("y", "space"), AxisMetadata("x", "space"))
    )
    widget.pipeline.outputs["input"] = data
    widget.pipeline.node_outputs["input"] = [data]
    widget.pipeline.output_states["input"] = state
    widget.pipeline.node_output_states["input"] = [state]
    widget.pipeline.completed_node_ids.add("input")
    checkbox = QCheckBox(widget)
    dialog = SimpleNamespace(
        output_assessment_checkbox=checkbox,
        assessment_results_ready=Mock(),
        assessment_results_unavailable=Mock(),
        close=lambda: None,
    )
    widget._workflow_assistant_dialog = dialog
    checkbox.toggled.connect(widget._workflow_assessment_permission_changed)
    widget._history.clear()
    calls = []

    def ordinary_run(**kwargs):
        calls.append(kwargs)
        widget._run_pipeline_synchronously(
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

    # Exercise the ordinary synchronous executor/adoption path without needing
    # a real napari Canvas or triggering source selection or provider discovery.
    monkeypatch.setattr(widget, "run_pipeline", ordinary_run)
    monkeypatch.setattr(widget, "_finish_pipeline_update", lambda *_args: None)
    monkeypatch.setattr(
        widget, "_finish_interaction_without_preview_if_needed", lambda: None
    )
    monkeypatch.setattr(widget, "_refresh_selected_parameter_controls", lambda: False)
    monkeypatch.setattr(widget, "_resync_autodefault_nodes", set)
    yield widget, data, dialog, calls
    widget._workflow_assistant_dialog = None


def add_candidate(widget):
    context, owner = widget._capture_workflow_proposal_context("input")
    raw = reference_proposal(context)
    raw["nodes"] = [node for node in raw["nodes"] if node["key"] != "measure"]
    raw["connections"] = [
        edge for edge in raw["connections"] if edge["target"] != "measure"
    ]
    return widget._apply_workflow_proposal(validate_proposal(raw, context), owner)


def run_candidate(editor):
    widget, _data, dialog, _calls = editor
    dialog.output_assessment_checkbox.setChecked(True)
    ids = add_candidate(widget)
    widget._run_workflow_assistant_plan(ids)
    assert widget._workflow_assistant_assessment_run is not None, (
        widget._workflow_assistant_assessment_unavailable
    )
    return ids, widget._capture_workflow_assessment_snapshot(ids, "input")


def test_add_and_old_cache_never_grant_assessment(assessment_editor, monkeypatch):
    widget, _data, dialog, calls = assessment_editor
    capture = Mock(wraps=workflow_assessment.capture_assessment_snapshot)
    monkeypatch.setattr(workflow_assessment, "capture_assessment_snapshot", capture)
    dialog.output_assessment_checkbox.setChecked(True)
    ids = add_candidate(widget)
    assert calls == [] and capture.call_count == 0
    # Even a ready-looking, previously cached graph has no completed Run ticket.
    for node_id in ids:
        widget.pipeline.node_execution_states[node_id] = "ready"
        widget.pipeline.completed_node_ids.add(node_id)
    with pytest.raises(ProposalError, match="Run the latest added plan"):
        widget._capture_workflow_assessment_snapshot(ids, "input")
    assert capture.call_count == 0


def test_normal_run_opt_in_captures_exact_source_and_candidate(assessment_editor):
    widget, data, dialog, calls = assessment_editor
    original = data.tobytes()
    ids, (snapshot, owner) = run_candidate(assessment_editor)
    assert len(calls) == 1
    assert owner.run_id.startswith("synchronous-")
    assert owner.owner[0] == widget._current_workflow_session_id()
    assert owner.owner[1] == widget._current_history_snapshot().workflow
    assert snapshot.candidate_node_ids == ids
    assert snapshot.source_id == "input"
    widget._assert_workflow_assessment_current(snapshot, owner)
    dialog.assessment_results_ready.assert_called_once_with(ids, owner.run_id)
    assert original == data.tobytes() and not data.flags.writeable


@pytest.mark.parametrize("outcome", ["success", "cancelled", "error"])
def test_background_acceptance_not_partial_or_cancelled_cache(
    assessment_editor, monkeypatch, outcome
):
    widget, data, dialog, _calls = assessment_editor
    dialog.output_assessment_checkbox.setChecked(True)
    queued = []
    widget._pipeline_thread_pool = SimpleNamespace(
        start=queued.append,
        waitForDone=lambda *_args: True,
        activeThreadCount=lambda: 0,
        clear=lambda: None,
    )
    state = widget.pipeline.output_states["input"]

    def background_run(**kwargs):
        widget._start_background_pipeline_run(
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

    monkeypatch.setattr(widget, "run_pipeline", background_run)
    ids = add_candidate(widget)
    widget._run_workflow_assistant_plan(ids)
    assert widget._workflow_assistant_assessment_run is None
    with pytest.raises(ProposalError):
        widget._capture_workflow_assessment_snapshot(ids, "input")
    request = queued[0].request
    if outcome == "success":
        result = execute_pipeline_request(request)
        assert not result.error and not result.cancelled
    else:
        result = PipelineRunResult(
            request.run_id,
            request.workflow,
            cancelled=outcome == "cancelled",
            error="Deliberate guarded failure" if outcome == "error" else "",
        )
    widget._on_background_pipeline_finished(result)
    if outcome == "success":
        assert widget._workflow_assistant_assessment_run is not None, (
            widget._workflow_assistant_assessment_unavailable
        )
        snapshot, owner = widget._capture_workflow_assessment_snapshot(ids, "input")
        assert owner.run_id == f"background-{request.run_id}"
        widget._assert_workflow_assessment_current(snapshot, owner)
    else:
        assert widget._workflow_assistant_assessment_run is None
        assert widget._workflow_assistant_assessment_pending is None
        dialog.assessment_results_ready.assert_not_called()


def test_opt_out_does_not_copy_or_measure_results(assessment_editor, monkeypatch):
    widget, _data, dialog, _calls = assessment_editor
    capture = Mock(wraps=workflow_assessment.capture_assessment_snapshot)
    monkeypatch.setattr(workflow_assessment, "capture_assessment_snapshot", capture)
    ids = add_candidate(widget)
    widget._run_workflow_assistant_plan(ids)
    assert widget._workflow_assistant_assessment_run is None
    assert capture.call_count == 0
    dialog.output_assessment_checkbox.setChecked(True)
    with pytest.raises(ProposalError, match="Run the latest added plan"):
        widget._capture_workflow_assessment_snapshot(ids, "input")


@pytest.mark.parametrize(
    "change",
    [
        "graph",
        "source",
        "cache",
        "mutated_output",
        "mutated_intermediate",
        "session",
        "compute",
        "permission",
        "pending",
        "active",
        "closing",
        "new_dispatch",
        "undo",
    ],
)
def test_completed_checkpoint_rejects_stale_or_busy_results(
    assessment_editor, monkeypatch, change
):
    widget, data, dialog, _calls = assessment_editor
    ids, (snapshot, owner) = run_candidate(assessment_editor)
    scientific = next(
        node_id for node_id in ids if widget.pipeline.output_ports(node_id)
    )
    if change == "graph":
        widget.pipeline.nodes[scientific].params["sigma"] = 4.0
    elif change == "source":
        changed = data.copy()
        changed[0, 0] = 0.5
        changed.flags.writeable = False
        widget.pipeline.outputs["input"] = changed
        widget.pipeline.node_outputs["input"] = [changed]
    elif change == "cache":
        changed = widget.pipeline.outputs[scientific].copy()
        widget.pipeline.outputs[scientific] = changed
        widget.pipeline.node_outputs[scientific] = [changed]
    elif change == "mutated_output":
        assessed = next(out for out in snapshot._outputs if out.binding is not None)
        value = widget.pipeline.node_outputs[assessed.node_id][assessed.port]
        value.flags.writeable = True
        value.flat[0] += 1
    elif change == "mutated_intermediate":
        value = widget.pipeline.outputs[scientific]
        value.flags.writeable = True
        value.flat[0] += 0.25
    elif change == "session":
        monkeypatch.setattr(widget, "_current_workflow_session_id", lambda: "different")
    elif change == "compute":
        widget._compute_mode = ComputeMode.AUTO
    elif change == "permission":
        dialog.output_assessment_checkbox.setChecked(False)
    elif change == "pending":
        widget._pending_dirty_node_ids.add(scientific)
    elif change == "active":
        widget._active_pipeline_run_id = 99
    elif change == "closing":
        widget._closing = True
    elif change == "new_dispatch":
        widget._begin_pipeline_dispatch(None)
    else:
        widget.undo()
    try:
        with pytest.raises((ProposalError, AssessmentError)):
            widget._assert_workflow_assessment_current(snapshot, owner)
    finally:
        widget._active_pipeline_run_id = None
        widget._closing = False


def test_failed_run_never_assesses_retained_cache(assessment_editor, monkeypatch):
    widget, _data, dialog, _calls = assessment_editor
    dialog.output_assessment_checkbox.setChecked(True)
    ids = add_candidate(widget)
    monkeypatch.setattr(
        widget,
        "_execute_synchronous_cpu_pipeline",
        Mock(side_effect=ValueError("Deliberate guarded calculation failure")),
    )
    widget._run_workflow_assistant_plan(ids)
    assert widget._workflow_assistant_assessment_run is None
    assert widget._workflow_assistant_assessment_pending is None
    dialog.assessment_results_ready.assert_not_called()
    with pytest.raises(ProposalError):
        widget._capture_workflow_assessment_snapshot(ids, "input")


def test_cancel_and_recompute_invalidate_even_when_old_outputs_remain(
    assessment_editor,
):
    widget, _data, _dialog, _calls = assessment_editor
    _ids, (snapshot, owner) = run_candidate(assessment_editor)
    widget._cancel_background_pipeline_run()
    assert widget._workflow_assistant_assessment_run is None
    with pytest.raises(ProposalError):
        widget._assert_workflow_assessment_current(snapshot, owner)


def test_revoking_then_reenabling_cannot_resurrect_checkpoint(assessment_editor):
    widget, _data, dialog, _calls = assessment_editor
    ids, (snapshot, owner) = run_candidate(assessment_editor)
    dialog.output_assessment_checkbox.setChecked(False)
    dialog.output_assessment_checkbox.setChecked(True)
    with pytest.raises(ProposalError, match="latest assistant Run"):
        widget._assert_workflow_assessment_current(snapshot, owner)
    with pytest.raises(ProposalError, match="fresh plan"):
        widget._capture_workflow_assessment_snapshot(ids, "input")


def test_missing_assessment_diagnostics_do_not_fail_the_scientific_run(
    assessment_editor, monkeypatch
):
    widget, _data, dialog, _calls = assessment_editor
    dialog.output_assessment_checkbox.setChecked(True)
    ids = add_candidate(widget)
    monkeypatch.setattr(
        workflow_assessment,
        "capture_assessment_snapshot",
        Mock(side_effect=AssessmentError("No supported cached output is available.")),
    )
    widget._run_workflow_assistant_plan(ids)
    assert all(
        widget.pipeline.node_execution_states[node_id] == "ready"
        for node_id in ids
        if widget.pipeline.output_ports(node_id)
    )
    assert widget._workflow_assistant_assessment_run is None
    assert (
        "No supported cached output"
        in widget._workflow_assistant_assessment_unavailable
    )
    dialog.assessment_results_unavailable.assert_called_once()


@pytest.mark.parametrize("bad_ids", [(), ("old-plan",), ["old-plan"], None])
def test_other_or_mutable_candidate_identity_cannot_assess(assessment_editor, bad_ids):
    widget, _data, _dialog, _calls = assessment_editor
    run_candidate(assessment_editor)
    with pytest.raises(ProposalError, match="latest added and completed candidate"):
        widget._capture_workflow_assessment_snapshot(bad_ids, "input")


def test_run_without_actual_dispatch_does_not_arm_later_calculation(
    assessment_editor, monkeypatch
):
    widget, _data, dialog, _calls = assessment_editor
    dialog.output_assessment_checkbox.setChecked(True)
    ids = add_candidate(widget)
    monkeypatch.setattr(widget, "_calculate_all_nodes", lambda: None)
    widget._run_workflow_assistant_plan(ids)
    assert widget._workflow_assistant_assessment_armed is None
    widget._begin_pipeline_dispatch(None)
    assert widget._workflow_assistant_assessment_pending is None
