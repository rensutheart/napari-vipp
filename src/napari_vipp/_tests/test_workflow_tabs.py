from __future__ import annotations

from pathlib import Path

import pytest

from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.snapshots import GraphSnapshot, WorkflowSnapshot
from napari_vipp.ui.history import WorkflowHistorySnapshot
from napari_vipp.ui.workflow_tabs import (
    UnsavedWorkflowError,
    WorkflowTabBar,
    WorkflowTabModel,
    WorkflowTabSession,
)


def _editor_snapshot(
    pipeline: PrototypePipeline,
    *,
    selected: str = "gaussian",
    positions: dict[str, tuple[float, float]] | None = None,
) -> WorkflowHistorySnapshot:
    return WorkflowHistorySnapshot(
        workflow=WorkflowSnapshot(
            GraphSnapshot.from_pipeline(pipeline),
            positions or {},
        ),
        selected_node_id=selected,
    )


def _session(
    *,
    session_id: str,
    path: Path | None = None,
    title: str | None = None,
) -> WorkflowTabSession:
    pipeline = PrototypePipeline()
    return WorkflowTabSession(
        pipeline,
        _editor_snapshot(pipeline),
        path=path,
        title=title,
        session_id=session_id,
    )


def test_workflow_sessions_own_independent_pipeline_history_and_runtime_cache():
    model = WorkflowTabModel()

    first = model.create_blank()
    second = model.create_blank()

    assert first.title == "Untitled"
    assert second.title == "Untitled 2"
    assert first.pipeline is not second.pipeline
    assert first.history is not second.history
    assert first.runtime_cache is not second.runtime_cache
    first.pipeline.outputs["input"] = "calculated"
    first.runtime_cache["scatter"] = "density"
    assert second.pipeline.outputs == {}
    assert second.runtime_cache == {}


def test_capture_marks_only_persisted_editor_changes_dirty_and_save_resets_it(
    tmp_path,
):
    pipeline = PrototypePipeline()
    session = WorkflowTabSession(
        pipeline,
        _editor_snapshot(pipeline, selected="input"),
        session_id="session",
    )

    assert not session.dirty
    session.capture_editor_snapshot(_editor_snapshot(pipeline, selected="threshold"))
    assert not session.dirty

    pipeline.set_param("gaussian", "sigma", 4.5)
    session.capture_editor_snapshot(_editor_snapshot(pipeline, selected="gaussian"))
    assert session.dirty

    saved_path = tmp_path / "variant.json"
    session.mark_saved(saved_path)
    assert not session.dirty
    assert session.path == saved_path.resolve()
    assert session.title == "variant"

    session.rename("Reviewed variant")
    pipeline.set_param("gaussian", "sigma", 5.5)
    session.capture_editor_snapshot(_editor_snapshot(pipeline))
    assert session.dirty
    session.mark_saved(tmp_path / "renamed-on-disk.json")
    assert session.title == "Reviewed variant"
    assert not session.dirty


def test_save_persistent_state_outside_snapshot_participates_in_dirty_baseline(
    tmp_path,
):
    pipeline = PrototypePipeline()
    session = WorkflowTabSession(
        pipeline,
        _editor_snapshot(pipeline),
        session_id="batch-state",
        persistence_token="batch:none",
    )

    session.capture_editor_snapshot(
        _editor_snapshot(pipeline),
        persistence_token="batch:configured",
    )
    assert session.dirty

    session.mark_saved(
        tmp_path / "with-batch.json",
        persistence_token="batch:configured",
    )
    assert not session.dirty

    session.capture_editor_snapshot(
        _editor_snapshot(pipeline),
        persistence_token="batch:edited",
    )
    assert session.dirty


def test_session_rejects_empty_titles_and_snapshots_from_another_pipeline():
    first_pipeline = PrototypePipeline()
    second_pipeline = PrototypePipeline()
    session = WorkflowTabSession(
        first_pipeline,
        _editor_snapshot(first_pipeline),
        session_id="first",
    )
    second_pipeline.set_param("gaussian", "sigma", 9.0)

    with pytest.raises(ValueError, match="title cannot be empty"):
        session.rename("  ")
    with pytest.raises(ValueError, match="does not match its pipeline"):
        session.capture_editor_snapshot(_editor_snapshot(second_pipeline))


def test_model_preserves_active_session_during_insert_reorder_and_close(tmp_path):
    model = WorkflowTabModel()
    first = _session(session_id="first", path=tmp_path / "first.json")
    second = _session(session_id="second", path=tmp_path / "second.json")
    third = _session(session_id="third", path=tmp_path / "third.json")
    model.add(first)
    model.add(second)
    model.add(third, make_current=False, index=1)

    assert model.sessions == (first, third, second)
    assert model.current is second
    model.move(0, 2)
    assert model.sessions == (third, second, first)
    assert model.current is second
    assert model.current_index == 1

    first.mark_dirty()
    with pytest.raises(UnsavedWorkflowError) as error:
        model.close(2)
    assert error.value.session is first
    assert model.sessions == (third, second, first)

    assert model.close(2, discard_unsaved=True) is first
    assert model.current is second
    assert model.close(1) is second
    assert model.current is third
    assert model.current_index == 0
    assert model.close(0) is third
    assert model.current is None
    assert model.current_index == -1


def test_model_rename_and_index_lookup_do_not_change_workflow_dirty_state():
    model = WorkflowTabModel()
    session = model.create_blank()

    model.rename(0, "Reference")

    assert model.index_of(session.session_id) == 0
    assert session.title == "Reference"
    assert session.title_is_custom
    assert not session.dirty
    with pytest.raises(KeyError, match="not open"):
        model.index_of("missing")


def test_tab_bar_mirrors_titles_paths_dirty_state_and_active_index(qtbot, tmp_path):
    model = WorkflowTabModel()
    clean = _session(session_id="clean", path=tmp_path / "clean.json")
    dirty = _session(session_id="dirty", title="Experiment")
    dirty.mark_dirty()
    model.add(clean)
    model.add(dirty)
    bar = WorkflowTabBar()
    qtbot.addWidget(bar)

    bar.sync_from_model(model)

    assert bar.count() == 2
    assert bar.currentIndex() == 1
    assert bar.tabText(0) == "clean"
    assert bar.tabText(1) == "Experiment *"
    assert bar.tabData(0) == "clean"
    assert str((tmp_path / "clean.json").resolve()) in bar.tabToolTip(0)
    assert "Unsaved changes" in bar.tabToolTip(1)
    assert bar.isMovable()
    assert bar.tabsClosable()

    dirty.mark_clean()
    bar.refresh_session(1, dirty)
    assert bar.tabText(1) == "Experiment"
    assert "No unsaved changes" in bar.tabToolTip(1)


def test_tab_bar_emits_activation_close_rename_new_and_reorder_requests(
    qtbot,
    monkeypatch,
):
    model = WorkflowTabModel()
    model.create_blank()
    model.create_blank()
    bar = WorkflowTabBar()
    qtbot.addWidget(bar)
    bar.sync_from_model(model)

    with qtbot.waitSignal(bar.activateTabRequested) as activated:
        bar.setCurrentIndex(0)
    assert activated.args == [0]

    with qtbot.waitSignal(bar.closeTabRequested) as closed:
        bar.tabCloseRequested.emit(0)
    assert closed.args == [0]

    monkeypatch.setattr(
        "napari_vipp.ui.workflow_tabs.QInputDialog.getText",
        lambda *_args, **_kwargs: ("Comparison", True),
    )
    with qtbot.waitSignal(bar.renameTabRequested) as renamed:
        bar.request_rename(0)
    assert renamed.args == [0, "Comparison"]

    with qtbot.waitSignal(bar.newTabRequested):
        bar.request_new_tab()

    with qtbot.waitSignal(bar.tabsReordered) as reordered:
        bar.moveTab(0, 1)
    assert reordered.args == [0, 1]
