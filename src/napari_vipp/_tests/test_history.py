from __future__ import annotations

import pytest

from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.snapshots import GraphSnapshot, WorkflowSnapshot
from napari_vipp.ui.history import WorkflowHistory, WorkflowHistorySnapshot


def _snapshot(name: str) -> WorkflowHistorySnapshot:
    pipeline = PrototypePipeline()
    pipeline.reset_starter_graph()
    pipeline.set_param("gaussian", "sigma", float(len(name)))
    return WorkflowHistorySnapshot(
        WorkflowSnapshot(GraphSnapshot.from_pipeline(pipeline)),
        selected_node_id=name,
    )


def test_history_rejects_invalid_limits():
    for value in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            WorkflowHistory(limit=value)  # type: ignore[arg-type]


def test_history_undo_and_redo_move_the_current_snapshot():
    history = WorkflowHistory(limit=3)
    first = _snapshot("first")
    second = _snapshot("second")
    current = _snapshot("current")
    history.push(first)
    history.push(second)

    assert history.undo(current) == second
    assert history.undo(second) == first
    assert not history.can_undo
    assert history.redo(first) == second
    assert history.redo(second) == current
    assert not history.can_redo


def test_push_coalesces_duplicates_clears_redo_and_honors_limit():
    history = WorkflowHistory(limit=2)
    first = _snapshot("first")
    second = _snapshot("second")
    third = _snapshot("third")

    assert history.push(first)
    assert not history.push(first)
    assert history.push(second)
    assert history.undo(third) == second
    assert history.can_redo

    assert history.push(third)
    assert not history.can_redo
    assert history.undo_stack == [first, third]

    fourth = _snapshot("fourth")
    assert history.push(fourth)
    assert history.undo_stack == [third, fourth]


def test_parameter_groups_coalesce_until_finished():
    history = WorkflowHistory()

    assert history.should_capture_group(("node", "sigma"))
    assert not history.should_capture_group(("node", "sigma"))
    assert history.should_capture_group(("node", "truncate"))
    history.finish_group()
    assert history.should_capture_group(("node", "truncate"))


def test_suspended_recording_is_nested_and_exception_safe():
    history = WorkflowHistory()
    snapshot = _snapshot("saved")

    with pytest.raises(RuntimeError, match="stop"):
        with history.suspend_recording():
            with history.suspend_recording():
                assert not history.push(snapshot)
                raise RuntimeError("stop")

    assert history.recording_enabled
    assert history.push(snapshot)


def test_clear_discards_both_stacks_and_pending_group():
    history = WorkflowHistory()
    first = _snapshot("first")
    current = _snapshot("current")
    history.push(first)
    history.undo(current)
    history.should_capture_group("parameter")

    history.clear()

    assert not history.can_undo
    assert not history.can_redo
    assert history.should_capture_group("parameter")
