"""Existing-file choices reuse checks, without authorizing unreviewed writes."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from napari_vipp._tests.test_batch_redesign_host import batch_case as _case
from napari_vipp._tests.test_batch_results import _preview
from napari_vipp._tests.test_batch_run_startup import _request
from napari_vipp._tests.test_ui_batch import _actions
from napari_vipp._widget import VippWidget
from napari_vipp.core.batch import ExistingFilePolicy
from napari_vipp.ui.batch import CollectionBatchDialog
from napari_vipp.ui.batch_output_policy import (
    batch_work_counts,
    output_action,
    output_counts_text,
    planned_item_status,
    with_existing_file_policy,
)
from napari_vipp.ui.batch_workers import CollectionBatchWorker


def _existing_plan(tmp_path, count=14, existing=8):
    plan = _preview(tmp_path, count)
    items = tuple(
        replace(
            item,
            outputs=tuple(
                replace(output, exists=i < existing) for output in item.outputs
            ),
        )
        for i, item in enumerate(plan.items)
    )
    return replace(plan, items=items, collision_count=existing)


@pytest.mark.parametrize("surface", ["setup", "items", "results"])
def test_policy_only_edit_updates_all_review_surfaces_without_checking(
    qtbot, tmp_path, surface
):
    plan = _existing_plan(tmp_path)
    calls = []
    actions = replace(_actions(plan, []), check_batch=lambda *_a: calls.append("check"))
    dialog = CollectionBatchDialog(actions=actions)
    qtbot.addWidget(dialog)
    dialog.apply_preview_result(plan, preview_representative=False)
    dialog.tabs.setCurrentIndex(3)
    dialog._current_item = 5
    dialog._checked_items = {1, 5}
    dialog.results_panel.select_item(5)
    invalidations = []
    dialog.previewInvalidated.connect(lambda: invalidations.append(True))
    combo = {
        "setup": dialog.existing_policy_combo,
        "items": dialog.existing_files_controls.policy_combo,
        "results": dialog.results_panel.existing_files_controls.policy_combo,
    }[surface]
    assert "prompts when you press Run" in dialog.preview_status.text()
    combo.setCurrentIndex(combo.findData("skip"))

    updated = dialog._preview_result
    assert updated is not None and updated.collision_count == 0
    assert updated.config.existing_file_policy is ExistingFilePolicy.SKIP
    assert calls == invalidations == []
    assert dialog.tabs.currentIndex() == 3
    assert dialog._current_item == 5 and dialog._checked_items == {1, 5}
    assert dialog.results_panel._selected_position() == 5
    assert dialog.run_button.isEnabled() and dialog.run_button.text() == "Run 6 items"
    assert dialog.preview_table.item(0, 3).text() == "1 existing"
    assert dialog.preview_table.item(0, 5).text() == "Keep existing"
    assert dialog.preview_table.item(8, 3).text() == "1 to create"
    assert dialog.results_panel.items_table.item(0, 1).text() == "Keep existing"
    assert dialog.results_panel.items_table.item(0, 2).text() == "1 existing"
    assert (
        "6 to create · 8 existing to keep"
        in dialog.results_panel.review_labels["Batch"].text()
    )
    for original, changed in zip(plan.items, updated.items, strict=True):
        assert original.source_items is changed.source_items
        assert original.source_paths is changed.source_paths
        assert original.parameter_overrides is changed.parameter_overrides
    for mirror in (
        dialog.existing_policy_combo,
        dialog.existing_files_controls.policy_combo,
        dialog.results_panel.existing_files_controls.policy_combo,
    ):
        assert mirror.currentData() == "skip"

    combo.setCurrentIndex(combo.findData("overwrite"))
    assert dialog.run_button.text() == "Run 14 items"
    assert dialog.preview_table.item(0, 3).text() == "1 to overwrite"
    assert dialog.preview_table.item(0, 5).text() == "Will overwrite"
    assert calls == invalidations == []


def test_all_existing_can_be_kept_and_no_run_happens_until_clicked(qtbot, tmp_path):
    plan = _existing_plan(tmp_path, 3, 3)
    dialog = CollectionBatchDialog(actions=_actions(plan, []))
    qtbot.addWidget(dialog)
    dialog.apply_preview_result(plan, preview_representative=False)
    runs = []
    dialog.runRequested.connect(runs.append)
    dialog.tabs.setCurrentIndex(3)
    dialog._choose_existing_file_policy("skip")
    assert runs == []
    assert dialog.run_button.text() == "Keep existing files"
    assert dialog.run_button.isEnabled()
    dialog.run_button.click()
    assert len(runs) == 1 and runs[0]["existing_file_policy"] == "skip"


def test_policy_does_not_revive_stale_plan_or_override_active_work(qtbot, tmp_path):
    plan = _existing_plan(tmp_path)
    dialog = CollectionBatchDialog(actions=_actions(plan, []))
    qtbot.addWidget(dialog)
    dialog.apply_preview_result(plan, preview_representative=False)
    dialog.set_representative_pending(True)
    dialog._choose_existing_file_policy("skip")
    assert not dialog.run_button.isEnabled()
    dialog.set_representative_pending(False)
    assert dialog.run_button.isEnabled()
    dialog.pattern_edit.setText("*.tif")
    dialog._choose_existing_file_policy("overwrite")
    assert dialog._preview_result is None and not dialog.run_button.isEnabled()
    assert not dialog.existing_files_controls.policy_combo.isEnabled()


@pytest.mark.parametrize("policy", list(ExistingFilePolicy))
def test_policy_change_preserves_explicit_output_rules_and_real_collisions(
    tmp_path, policy
):
    plan = _existing_plan(tmp_path, 1, 1)
    output = plan.items[0].outputs[0]
    specs = tuple(
        replace(plan.config.outputs[0], node_id=str(i), overwrite=rule)
        for i, rule in enumerate(("yes", "no", "batch default", "batch default"))
    )
    outputs = tuple(
        replace(output, node_id=str(i), duplicate=i == 2, input_collision=i == 3)
        for i in range(4)
    )
    plan = replace(
        plan,
        config=replace(plan.config, outputs=specs),
        items=(replace(plan.items[0], outputs=outputs),),
    )
    updated = with_existing_file_policy(plan, policy)
    assert (
        updated.items[0].outputs[0].existing_file_policy is ExistingFilePolicy.OVERWRITE
    )
    assert updated.items[0].outputs[1].existing_file_policy is ExistingFilePolicy.ERROR
    assert [output_action(out, updated.config) for out in updated.items[0].outputs] == [
        "overwrite",
        "blocked",
        "blocked",
        "blocked",
    ]
    assert updated.collision_count == 3


def test_mixed_item_keeps_only_existing_outputs(tmp_path):
    plan = _existing_plan(tmp_path, 1, 1)
    first = plan.items[0].outputs[0]
    missing = replace(first, path=first.path.with_name("missing.npy"), exists=False)
    plan = replace(plan, items=(replace(plan.items[0], outputs=(first, missing)),))
    plan = with_existing_file_policy(plan, "skip")
    assert (
        output_counts_text(plan.items[0].outputs) == "1 to create · 1 existing to keep"
    )
    assert planned_item_status(plan.items[0].outputs) == "Create missing"
    assert batch_work_counts(plan) == (1, 0)


@pytest.mark.parametrize("existing_count", [1, 2])
def test_real_worker_skips_existing_and_runs_remaining_after_policy_only_edit(
    qtbot, tmp_path, monkeypatch, existing_count,
):
    case = _case.__wrapped__(tmp_path)
    controller, values, _ = case
    initial = controller.preview(**values)
    first = initial.items[0].outputs[0].path
    first.parent.mkdir(parents=True)
    first.write_bytes(b"existing result must be kept exactly")
    if existing_count == 2:
        initial.items[1].outputs[0].path.write_bytes(b"another existing output")
    checked = controller.preview(**values)
    updated = with_existing_file_policy(checked, "skip")
    values["existing_file_policy"] = "skip"
    assert updated == controller.preview(**values)
    request = replace(
        _request(case), config=updated.config, expected_items=updated.items
    )
    worker = CollectionBatchWorker(request)
    if existing_count == 2:
        # Keeping every output must not execute any image operation.
        monkeypatch.setattr(
            "napari_vipp.core.batch.execute_pipeline_request",
            lambda *_a, **_kw: pytest.fail("All-kept items must not be calculated"),
        )
    outcomes = []
    worker.signals.finished.connect(outcomes.append)
    worker.run()
    assert len(outcomes) == 1 and not outcomes[0].error
    assert outcomes[0].result.summary["completed"] == 2 - existing_count
    assert outcomes[0].result.summary["skipped"] == existing_count
    assert first.read_bytes() == b"existing result must be kept exactly"
    assert updated.items[1].outputs[0].path.exists()


@pytest.mark.parametrize("failed", [-1, 0])
def test_missing_or_failed_optional_preview_does_not_block_final_preflight(
    tmp_path, failed
):
    started = []
    dialog = SimpleNamespace(_preview_result=_preview(tmp_path))
    host = SimpleNamespace(
        _active_collection_batch_dialog=dialog,
        _commit_crop_draft=lambda **_kw: None,
        _engage_collection_batch_workspace=lambda _d: None,
        _collection_batch_running=False,
        _pending_collection_batch_start=None,
        _active_source_load_id=None,
        _active_pipeline_run_id=None,
        _source_load_pending=False,
        _pipeline_run_pending=False,
        _debounce_timer=SimpleNamespace(isActive=lambda: False),
        _interactive_collection_batch_items=dialog._preview_result.items,
        _interactive_collection_batch_requested_index=-1,
        _interactive_collection_batch_index=-1,
        _interactive_collection_batch_failed_index=failed,
        _active_thumbnail_contrast_run_id=None,
        _start_batch_run_preflight=lambda *args: started.append(args),
    )
    VippWidget._run_collection_batch_from_workspace(host, dialog, {})
    assert started == [(dialog, {})]


def test_failed_preview_clears_busy_flag_but_keeps_error_and_quarantine(
    qtbot, tmp_path
):
    plan = _preview(tmp_path)
    dialog = CollectionBatchDialog(actions=_actions(plan, []))
    qtbot.addWidget(dialog)
    dialog.apply_preview_result(plan, preview_representative=False)
    dialog.set_representative_pending(True)
    errors = []
    host = SimpleNamespace(
        _interactive_collection_batch_requested_index=0,
        _interactive_collection_batch_index=-1,
        _interactive_collection_batch_items=plan.items,
        _interactive_collection_source_paths={},
        _interactive_collection_source_series_indices={},
        _interactive_collection_source_node_ids=lambda: set(),
        _sync_interactive_collection_batch_navigator=lambda **_kw: None,
        _sync_input_node_subtitles=lambda _nodes: None,
        batch_navigator=SimpleNamespace(show_representative_error=errors.append),
        _active_collection_batch_dialog=dialog,
        _compute_runtime_quarantined_reason="Cleanup failed; restart required.",
        _set_status=lambda message, **_kw: errors.append(message),
    )
    VippWidget._show_interactive_collection_batch_preview_error(
        host, 0, "Preview failed"
    )
    assert not dialog._representative_pending
    assert host._interactive_collection_batch_requested_index == -1
    assert (
        host._compute_runtime_quarantined_reason == "Cleanup failed; restart required."
    )
    assert "Preview failed" in dialog.graph_preview_status.text()
    assert "Cleanup failed" in errors[-1]
