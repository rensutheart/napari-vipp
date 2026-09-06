"""Batch workflow identity stays tied to its owning retained workflow tab."""

from __future__ import annotations

import pytest

from napari_vipp._widget import VippWidget
from napari_vipp.ui.workflow_tabs import WorkflowTabModel


class _IdentityHost:
    _collection_batch_dialog_actions = VippWidget._collection_batch_dialog_actions
    _collection_batch_workflow_summary = VippWidget._collection_batch_workflow_summary
    _workflow_tab_session = VippWidget._workflow_tab_session

    def __init__(self):
        self._workflow_tabs = WorkflowTabModel()

    def _unused_action(self, *_args, **_kwargs):
        raise AssertionError("Reading workflow identity must not run host actions.")

    _preview_collection_batch = _unused_action
    _choose_collection_batch_demo = _unused_action
    _batch_source_rows = _unused_action
    _load_collection_batch_config = _unused_action
    _save_collection_batch_config = _unused_action
    _preview_collection_batch_plan_item = _unused_action
    _check_collection_batch = _unused_action
    _recheck_collection_batch_items = _unused_action
    _collection_batch_compute_summary = _unused_action
    _sync_current_workflow_tab_state = _unused_action


@pytest.mark.parametrize("dirty", [False, True])
def test_saved_workflow_identity_describes_live_workflow(tmp_path, dirty):
    host = _IdentityHost()
    session = host._workflow_tabs.create_blank()
    path = tmp_path / "vipp_workflow.json"
    session.mark_saved(path)
    if dirty:
        session.mark_dirty()

    label, context = host._collection_batch_workflow_summary()

    assert label == "vipp_workflow" + (" · modified" if dirty else "")
    assert str(path.resolve()) in context
    assert "current live workflow" in context
    assert "including any unsaved changes" in context
    assert "does not reload an older saved file" in context
    assert ("This workflow has unsaved changes." in context) is dirty
    assert not path.exists()


@pytest.mark.parametrize("dirty", [False, True])
def test_unsaved_workflow_identity_never_claims_a_saved_file(dirty):
    host = _IdentityHost()
    session = host._workflow_tabs.create_blank()
    if dirty:
        session.mark_dirty()

    label, context = host._collection_batch_workflow_summary()

    assert label == "Untitled" + (" · modified" if dirty else "")
    assert "Unsaved workflow: no workflow file path." in context
    assert "Workflow file:" not in context
    assert "current live workflow" in context


def test_workflow_action_retains_origin_and_reads_updated_identity(tmp_path):
    host = _IdentityHost()
    source = host._workflow_tabs.create_blank()
    source.mark_saved(tmp_path / "source.json")
    source_action = host._collection_batch_dialog_actions().workflow_summary
    target = host._workflow_tabs.create_blank()
    target.mark_saved(tmp_path / "target.json")
    target_action = host._collection_batch_dialog_actions().workflow_summary
    source.rename("Source workflow")
    source.mark_dirty()

    source_label, source_context = source_action()
    target_label, target_context = target_action()

    assert source_label == "Source workflow · modified"
    assert str(source.path) in source_context
    assert str(target.path) not in source_context
    assert target_label == "target"
    assert str(target.path) in target_context
    assert str(source.path) not in target_context
    assert host._workflow_tabs.current is target


def test_closed_origin_does_not_fall_back_to_another_workflow():
    host = _IdentityHost()
    source = host._workflow_tabs.create_blank()
    action = host._collection_batch_dialog_actions().workflow_summary
    target = host._workflow_tabs.create_blank()
    host._workflow_tabs.close(host._workflow_tabs.index_of(source.session_id))

    label, context = action()

    assert label == "Workflow unavailable"
    assert "no longer open" in context
    assert host._workflow_tabs.current is target


def test_unbound_actions_do_not_adopt_a_later_workflow():
    host = _IdentityHost()
    action = host._collection_batch_dialog_actions().workflow_summary
    host._workflow_tabs.create_blank()

    assert action()[0] == "Workflow unavailable"
