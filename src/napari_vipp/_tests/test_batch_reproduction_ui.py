from __future__ import annotations

from dataclasses import replace

import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QMessageBox

from napari_vipp._tests.test_batch_controller import _explicit_batch_pipeline
from napari_vipp._tests.test_ui_batch import _actions, _preview_result
from napari_vipp.core.batch import BatchPlan
from napari_vipp.core.reproduction import (
    ReproductionCheck,
    ReproductionReference,
    ReproductionRequest,
    ReproductionRow,
    current_vipp_version,
)
from napari_vipp.core.workflow import serialize_workflow
from napari_vipp.ui.batch import CollectionBatchDialog
from napari_vipp.ui.batch_controller import (
    CollectionBatchController,
    _batch_preview_result,
)


def _request(*, version=None, mode="reproduce"):
    return ReproductionRequest(
        reference=ReproductionReference.from_dict(
            {
                "type": "napari-vipp-reproduction-reference",
                "version": 1,
                "original_run_id": "a" * 32,
                "recorded_vipp_version": version or current_vipp_version(),
                "original_workflow_sha256": "b" * 64,
                "analysis_sha256": "c" * 64,
                "sources": [
                    {
                        "node_id": "input",
                        "role": "collection",
                        "items": [
                            {
                                "item_index": 1,
                                "container_format": "npy",
                                "revision": {
                                    "kind": "file",
                                    "sha256": "d" * 64,
                                    "regular_file_count": 1,
                                    "size_bytes": 1,
                                },
                                "selector_sha256": "e" * 64,
                                "scientific_sha256": "f" * 64,
                            }
                        ],
                    }
                ],
            }
        ),
        mode=mode,
    )


def _check(request, *, rows=None, can_run=True, status=None):
    return ReproductionCheck(
        status=status or ("verified" if can_run else "mismatch"),
        can_run=can_run,
        recorded_vipp_version=request.reference.recorded_vipp_version,
        current_vipp_version=current_vipp_version(),
        version_override_used=request.version_override is not None,
        rows=tuple(rows)
        if rows is not None
        else (ReproductionRow("input", 1, "matched", "Original input verified."),),
        problems=() if can_run else ("Resolve every original-reference difference.",),
        reference_sha256=request.reference.digest,
    )


def _dialog(qtbot, tmp_path, *, request=None, count=3):
    result = _preview_result(tmp_path, count=count)
    if request is not None:
        result = replace(
            result,
            config=replace(result.config, reproduction=request),
            items=tuple(
                replace(item, reproduction_status="matched") for item in result.items
            ),
        )
    previewed = []
    dialog = CollectionBatchDialog(actions=_actions(result, previewed))
    qtbot.addWidget(dialog)
    dialog._apply_config(result.config)
    return dialog, result, previewed


def test_ordinary_batch_has_no_reproduction_controls_or_form_field(qtbot, tmp_path):
    dialog, result, _ = _dialog(qtbot, tmp_path)
    dialog.apply_preview_result(result, preview_representative=False)
    assert dialog.reproduction_banner.isHidden()
    assert "reproduction" not in dialog.values()
    dialog.tabs.setCurrentIndex(3)
    assert dialog.run_button.isEnabled()


def test_reproduction_verification_is_required_and_settings_invalidate_it(
    qtbot, tmp_path
):
    request = _request()
    dialog, result, _ = _dialog(qtbot, tmp_path, request=request)
    emitted = []
    dialog.runRequested.connect(emitted.append)
    assert dialog.values()["reproduction"] == request
    assert "Not verified" in dialog.reproduction_status_label.text()
    dialog._request_run()
    assert not emitted
    result = replace(result, reproduction=_check(request))
    dialog.apply_preview_result(result, preview_representative=False)
    assert dialog.preview_table.item(0, 4).text() == "Verified"
    dialog.tabs.setCurrentIndex(3)
    assert dialog.run_button.isEnabled()
    assert "verified source checks" in dialog.reproduction_status_label.text()

    dialog.output_edit.setText(str(tmp_path / "changed-output"))
    assert dialog._reproduction_check() is None
    assert not dialog.run_button.isEnabled()
    assert "Not verified" in dialog.reproduction_status_label.text()
    assert dialog.preview_table.item(0, 4).text() == "Needs recheck"
    assert dialog.values()["reproduction"] == request


def test_mismatches_block_direct_run_continue_preview_and_subset_bypass(
    qtbot, tmp_path
):
    request = _request()
    dialog, result, previewed = _dialog(qtbot, tmp_path, request=request)
    rows = (
        ReproductionRow("input", 1, "matched", "Verified input"),
        ReproductionRow("input", 2, "changed", "Wrong contents", path="field.npy"),
    )
    result = replace(result, reproduction=_check(request, rows=rows, can_run=False))
    emitted = []
    dialog.runRequested.connect(emitted.append)
    dialog.apply_preview_result(result, preview_representative=True)
    assert not previewed
    assert dialog.tabs.currentIndex() == 1
    assert dialog.next_button.text() == "Continue to run"
    assert not dialog.next_button.isEnabled()
    dialog._next_batch_step()
    assert dialog.tabs.currentIndex() == 1
    assert dialog._reproduction_table_active
    assert dialog.preview_table.item(1, 4).text() == "Changed input"
    assert not dialog.preview_table.item(1, 0).flags() & Qt.ItemIsUserCheckable
    dialog.preview_table.selectRow(0)
    dialog._checked_items = {0}
    dialog._request_run()
    assert not emitted
    assert not dialog._preview_selected_item()
    assert not dialog.recheck_item_button.isEnabled()
    dialog.tabs.setCurrentIndex(3)
    assert not dialog.run_button.isEnabled()


def test_missing_unexpected_checks_paginate_without_a_runnable_plan(qtbot, tmp_path):
    request = _request()
    dialog, result, _ = _dialog(qtbot, tmp_path, request=request)
    rows = [
        ReproductionRow("input", i, "matched", "Verified", path=f"field-{i}.npy")
        for i in range(1, 1002)
    ]
    rows.extend(
        (
            ReproductionRow("input", 1002, "missing", "Select the original folder."),
            ReproductionRow(
                "input", None, "extra", "Not an original input.", path="extra.npy"
            ),
        )
    )
    result = replace(
        result,
        items=(),
        rows=(),
        total_items=0,
        reproduction=_check(request, rows=rows, can_run=False),
    )
    dialog.apply_preview_result(result, preview_representative=False)
    assert dialog.tabs.isTabEnabled(1)
    assert dialog.preview_table.rowCount() == dialog._ITEM_PAGE_SIZE
    assert "1,001 verified" in dialog.reproduction_status_label.text()
    assert "1 missing input" in dialog.reproduction_status_label.text()
    assert "1 unexpected input" in dialog.reproduction_status_label.text()
    assert "1,003 source checks" in dialog.item_range_label.text()
    dialog.item_filter.setCurrentIndex(1)
    assert dialog.preview_table.rowCount() == 2
    assert dialog.preview_table.item(0, 4).text() == "Missing input"
    assert dialog.preview_table.item(1, 4).text() == "Unexpected input"
    dialog.preview_table.selectRow(1)
    assert "extra.npy" in dialog.item_details.toPlainText()
    dialog.item_search.setText("extra.npy")
    assert dialog.preview_table.rowCount() == 1
    assert not dialog.run_button.isEnabled()


def test_new_data_switch_requires_confirmation_clears_reference_and_emits(
    qtbot, tmp_path, monkeypatch
):
    request = _request()
    dialog, result, _ = _dialog(qtbot, tmp_path, request=request)
    dialog.apply_preview_result(
        replace(result, reproduction=_check(request)), preview_representative=False
    )
    changes = []
    dialog.reproductionChanged.connect(changes.append)
    monkeypatch.setattr(QMessageBox, "question", lambda *_a, **_k: QMessageBox.No)
    dialog._use_reproduction_new_data()
    assert dialog.values()["reproduction"] == request
    assert dialog._reproduction_check() is not None
    assert not changes
    monkeypatch.setattr(QMessageBox, "question", lambda *_a, **_k: QMessageBox.Yes)
    dialog._use_reproduction_new_data()
    assert "reproduction" not in dialog.values()
    assert dialog._preview_result is None
    assert changes == [None]
    assert not dialog.reproduction_banner.isHidden()
    assert "comparison is off" in dialog.reproduction_mode_label.text()


def test_cancelled_config_reproduction_choice_keeps_current_workspace(
    qtbot,
    tmp_path,
    monkeypatch,
):
    from qtpy.QtWidgets import QFileDialog

    from napari_vipp.ui.reproduction import ReproductionOpenCancelled

    dialog, result, _ = _dialog(qtbot, tmp_path, request=_request())
    dialog.apply_preview_result(result, preview_representative=False)
    before_values = dialog.values()
    before_status = dialog.preview_status.text()
    demo = object()
    dialog._demo = demo

    def cancel(_path):
        raise ReproductionOpenCancelled()

    dialog._actions = replace(dialog._actions, load_config=cancel)
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *_a, **_k: (str(tmp_path / "different.json"), ""),
    )
    dialog._load_config()
    assert dialog.values() == before_values
    assert dialog.preview_status.text() == before_status
    assert dialog._preview_result is result
    assert dialog._demo is demo


def test_version_acknowledgement_stays_visible_and_requires_fresh_check(
    qtbot, tmp_path, monkeypatch
):
    request = _request(version="0.0.1")
    dialog, result, _ = _dialog(qtbot, tmp_path, request=request)
    dialog.apply_preview_result(
        replace(
            result,
            reproduction=_check(request, can_run=False, status="version-mismatch"),
        ),
        preview_representative=False,
    )
    assert "Version mismatch" in dialog.reproduction_mode_label.text()
    assert not dialog.reproduction_version_button.isHidden()
    changes = []
    dialog.reproductionChanged.connect(changes.append)
    monkeypatch.setattr(QMessageBox, "question", lambda *_a, **_k: QMessageBox.Yes)
    dialog._acknowledge_reproduction_version()
    updated = dialog.values()["reproduction"]
    assert updated.version_override.to_dict() == {
        "recorded_vipp_version": "0.0.1",
        "current_vipp_version": current_vipp_version(),
    }
    assert changes == [updated]
    assert "acknowledged (deviation)" in dialog.reproduction_mode_label.text()
    assert dialog._reproduction_check() is None
    assert not dialog.run_button.isEnabled()


def test_controller_form_and_attached_normalization_preserve_reproduction(tmp_path):
    pipeline, _ = _explicit_batch_pipeline()
    workflow = serialize_workflow(pipeline)
    controller = CollectionBatchController(
        workflow_document_provider=lambda: workflow,
        pipeline_provider=lambda: pipeline,
    )
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    request = _request()
    config = controller.build_config(
        input_dir=inputs,
        output_dir=tmp_path / "outputs",
        reproduction=request,
    )
    assert config.reproduction == request
    assert (
        controller.prepare_attached_config_preview(config).config.reproduction
        == request
    )
    assert (
        controller.prepare_preview(
            input_dir=inputs,
            output_dir=tmp_path / "outputs",
            reproduction=request,
        ).config.reproduction
        == request
    )
    assert (
        controller.build_config(
            input_dir=inputs, output_dir=tmp_path / "new"
        ).reproduction
        is None
    )


def test_controller_maps_full_reference_checks_even_when_preview_rows_are_limited(
    tmp_path,
):
    result = _preview_result(tmp_path)
    request = _request()
    config = replace(result.config, reproduction=request)
    check = _check(request)
    mapped = _batch_preview_result(
        BatchPlan(config, result.items, config.output_dir, reproduction=check),
        config,
        preview_limit=1,
        explicit_outputs=True,
    )
    assert len(mapped.rows) == 1
    assert mapped.reproduction is check


def test_blocked_comparison_keeps_all_diagnostics_before_reviewed_snapshot_checks(
    tmp_path,
    monkeypatch,
):
    from napari_vipp.ui import batch_controller

    result = _preview_result(tmp_path)
    request = _request()
    config = replace(result.config, reproduction=request)
    check = _check(request, can_run=False)
    plan = BatchPlan(config, (), config.output_dir, reproduction=check)
    prepared = batch_controller.PreparedCollectionBatchPreview(
        workflow={},
        config=config,
        workflow_path=tmp_path / "workflow.json",
        preview_limit=50,
        explicit_outputs=True,
    )
    monkeypatch.setattr(batch_controller, "preflight_batch", lambda *_a, **_k: plan)

    def must_not_discard_diagnostics(*_args, **_kwargs):
        raise AssertionError(
            "A blocked comparison must retain its complete diagnostics"
        )

    monkeypatch.setattr(
        batch_controller,
        "_verify_reviewed_source_identities",
        must_not_discard_diagnostics,
    )
    mapped = batch_controller.execute_prepared_collection_batch_preview(prepared)
    assert mapped.reproduction is check
    assert not mapped.items


@pytest.mark.parametrize(
    "recorded,current,mismatch",
    [("v0.15.0a2", "0.15.0a2", False), ("unknown", "unknown", True)],
)
def test_banner_uses_core_version_identity_rules(
    qtbot,
    tmp_path,
    monkeypatch,
    recorded,
    current,
    mismatch,
):
    monkeypatch.setattr(
        "napari_vipp.ui.batch_reproduction.current_vipp_version",
        lambda: current,
    )
    dialog, _, _ = _dialog(qtbot, tmp_path, request=_request(version=recorded))
    assert ("Version mismatch" in dialog.reproduction_mode_label.text()) is mismatch


@pytest.mark.parametrize(
    "case", ["mismatch", "awaiting-choice", "version-mismatch", "missing-check"]
)
def test_worker_reproduction_guard_precedes_any_artifact_write(
    tmp_path, monkeypatch, case
):
    from napari_vipp.ui import batch_workers

    request = _request(
        mode="awaiting-choice" if case == "awaiting-choice" else "reproduce",
        version="0.0.1" if case == "version-mismatch" else None,
    )
    result = _preview_result(tmp_path)
    config = replace(result.config, reproduction=request)
    check = (
        None if case == "missing-check" else _check(request, can_run=False, status=case)
    )
    plan = BatchPlan(config, result.items, config.output_dir, reproduction=check)
    run = batch_workers.CollectionBatchRunRequest(
        job_id=1,
        origin_session_id="test",
        workflow={},
        config=config,
        workflow_path=config.output_dir / "workflow.json",
        config_path=config.output_dir / "config.json",
        script_path=config.output_dir / "runner.py",
        preflight_plan=plan,
    )
    monkeypatch.setattr(batch_workers, "validate_batch_config", lambda *_a, **_k: None)
    writes = []
    monkeypatch.setattr(batch_workers, "atomic_write_json", lambda *a: writes.append(a))
    monkeypatch.setattr(batch_workers, "atomic_write_text", lambda *a: writes.append(a))
    monkeypatch.setattr(batch_workers, "save_batch_config", lambda *a: writes.append(a))
    with pytest.raises(ValueError, match="(?i)reproduction"):
        batch_workers.prepare_collection_batch_run(run)
    assert not writes
    assert not config.output_dir.exists()
