"""Recorded-run checks must survive the real GUI-to-worker Run handoff."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from napari_vipp._tests.test_reproduction_open import (
    _open,
)
from napari_vipp._tests.test_reproduction_open import (
    shared_run as shared_run,
)
from napari_vipp.core.batch import BATCH_MANIFEST_FILENAME
from napari_vipp.ui.palette_roles import theme_colors


def _check_inputs(qtbot, dialog, source):
    dialog.input_edit.setText(str(source))
    assert dialog._check_batch()
    qtbot.waitUntil(lambda: not dialog._checking_plan, timeout=10_000)
    plan = dialog._preview_result
    assert plan is not None, dialog.preview_status.text()
    assert plan.reproduction is not None
    return plan


def _record_run_requests(monkeypatch, widget):
    requests = []
    original_prepare = widget._prepare_collection_batch_run

    def prepare(*args, **kwargs):
        request = original_prepare(*args, **kwargs)
        requests.append(request)
        return request

    monkeypatch.setattr(widget, "_prepare_collection_batch_run", prepare)
    return requests


def _run_and_wait(qtbot, widget, dialog):
    dialog.tabs.setCurrentIndex(3)
    assert dialog.run_button.isEnabled()
    dialog.run_button.click()
    qtbot.waitUntil(
        lambda: (
            not dialog._run_preparing
            and not widget._batch_workspace_preview_workers
            and not widget._collection_batch_running
            and not widget._collection_batch_workers
        ),
        timeout=10_000,
    )


@pytest.mark.parametrize("check_changed_first", [False, True])
def test_verified_originals_run_through_gui_and_worker(
    qtbot, monkeypatch, tmp_path, shared_run, check_changed_first
):
    widget, dialog = _open(qtbot, monkeypatch, shared_run)
    output = tmp_path / "reproduced-results"
    dialog.output_edit.setText(str(output))
    requests = _record_run_requests(monkeypatch, widget)

    if check_changed_first:
        changed = _check_inputs(qtbot, dialog, shared_run[2])
        assert not changed.reproduction.can_run
        assert changed.reproduction.mismatch_count == 3
        assert not dialog.run_button.isEnabled()
        assert not output.exists()

    checked = _check_inputs(qtbot, dialog, shared_run[1])
    assert checked.reproduction.can_run
    assert checked.reproduction.matched_count == 3
    assert checked.reproduction.mismatch_count == 0
    assert dialog.run_button.isEnabled()
    assert not output.exists()

    _run_and_wait(qtbot, widget, dialog)

    manifest_path = output / BATCH_MANIFEST_FILENAME
    assert manifest_path.is_file(), (
        dialog.run_result_label.text(),
        dialog.preview_status.text(),
        widget.status_label.text(),
    )
    assert len(requests) == 1
    handed_off = requests[0].preflight_plan
    assert handed_off is not None
    assert handed_off.reproduction is not None
    assert handed_off.reproduction.can_run
    assert handed_off.reproduction.matched_count == 3

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["summary"] == {
        "completed": 3,
        "partial": 0,
        "skipped": 0,
        "cancelled": 0,
        "failed": 0,
    }
    assert manifest["reproduction"]["status"] == "verified"
    assert manifest["reproduction"]["matched_count"] == 3
    assert len(manifest["items"]) == 3
    for item in manifest["items"]:
        assert item["status"] == "completed"
        assert len(item["outputs"]) == 1
        saved = item["outputs"][0]
        assert saved["status"] == "completed"
        destination = Path(saved["path"])
        assert destination.is_relative_to(output)
        np.testing.assert_array_equal(
            np.load(destination), np.load(item["sources"][0]["path"])
        )
    assert "3 completed" in dialog.run_progress_label.text()
    qtbot.waitUntil(lambda: not dialog._run_in_progress, timeout=2_000)
    status = dialog.reproduction_status_label.text()
    assert "Run complete · Original inputs verified" in status
    assert "3 matched" in status
    assert "Not verified" not in status
    assert "Ready for reviewed run" not in status
    success = theme_colors(dialog.palette()).success
    assert f"background: {success.surface.name()}" in (
        dialog.reproduction_banner.styleSheet()
    )
    assert dialog._preview_result is None
    assert dialog._reproduction_check() is None
    assert not dialog.run_button.isEnabled()
    dialog._request_run()
    assert len(requests) == 1


def test_source_changed_after_successful_check_stops_before_worker_or_outputs(
    qtbot, monkeypatch, tmp_path, shared_run
):
    # Never mutate the shared fixture's originals or any user input files.
    source = tmp_path / "original-copies"
    shutil.copytree(shared_run[1], source)
    widget, dialog = _open(qtbot, monkeypatch, shared_run)
    output = tmp_path / "must-not-exist"
    dialog.output_edit.setText(str(output))
    requests = _record_run_requests(monkeypatch, widget)
    checked = _check_inputs(qtbot, dialog, source)
    assert checked.reproduction.can_run
    assert checked.reproduction.matched_count == 3

    changed_path = source / "sample-1.npy"
    np.save(changed_path, np.full((4, 5), 99, dtype=np.uint8))
    _run_and_wait(qtbot, widget, dialog)

    assert requests == []
    assert not output.exists()
    refreshed = dialog._preview_result
    assert refreshed is not None, dialog.preview_status.text()
    assert refreshed.reproduction is not None
    assert not refreshed.reproduction.can_run
    assert refreshed.reproduction.matched_count == 2
    assert refreshed.reproduction.mismatch_count == 1
    assert {row.status for row in refreshed.reproduction.rows} == {"matched", "changed"}
    assert not dialog.run_button.isEnabled()
