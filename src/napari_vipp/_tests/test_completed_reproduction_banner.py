"""Completed-run evidence is visible without authorizing another reproduction."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
from qtpy.QtWidgets import QMessageBox

from napari_vipp._tests.test_batch_reproduction_ui import _check, _dialog, _request
from napari_vipp.core.batch import BatchStatus
from napari_vipp.core.reproduction import ReproductionRow, current_vipp_version
from napari_vipp.ui.palette_roles import theme_colors


def _three_input_request(*, version=None):
    request = _request(version=version)
    source = request.reference.sources[0]
    source = replace(
        source,
        items=tuple(replace(source.items[0], item_index=i) for i in range(1, 4)),
    )
    return replace(request, reference=replace(request.reference, sources=(source,)))


def _verified_check(request):
    return _check(
        request,
        rows=tuple(
            ReproductionRow("input", i, "matched", "Original input verified.")
            for i in range(1, 4)
        ),
    )


def _run_result(tmp_path, request, *, statuses=None, has_failures=False):
    statuses = statuses or (BatchStatus.COMPLETED,) * 3
    return SimpleNamespace(
        manifest=SimpleNamespace(
            items=tuple(
                SimpleNamespace(index=i, status=status)
                for i, status in enumerate(statuses, start=1)
            ),
            reproduction=_verified_check(request).to_dict(),
            compute={"runtime_cleanup_succeeded": not has_failures},
        ),
        summary={
            status.value: sum(item == status for item in statuses)
            for status in (
                BatchStatus.COMPLETED,
                BatchStatus.PARTIAL,
                BatchStatus.SKIPPED,
                BatchStatus.CANCELLED,
                BatchStatus.FAILED,
            )
        },
        cancelled=BatchStatus.CANCELLED in statuses,
        has_failures=has_failures or BatchStatus.FAILED in statuses,
        saved_paths=(),
        manifest_path=tmp_path / "outputs" / "vipp_batch_manifest.json",
    )


def _finish_verified_run(qtbot, tmp_path, *, request=None, run_result=None):
    request = request or _three_input_request()
    dialog, preview, _ = _dialog(qtbot, tmp_path, request=request)
    preview = replace(preview, reproduction=_verified_check(request))
    dialog.apply_preview_result(preview, preview_representative=False)
    dialog.begin_run(3)
    dialog.finish_run(run_result or _run_result(tmp_path, request))
    return dialog, preview


def _assert_tone(dialog, tone_name):
    tone = getattr(theme_colors(dialog.palette()), tone_name)
    style = dialog.reproduction_banner.styleSheet()
    assert f"background: {tone.surface.name()}" in style
    assert f"border-left: 3px solid {tone.accent.name()}" in style


def _assert_historical_success(dialog):
    status = dialog.reproduction_status_label.text()
    assert "Run complete · Original inputs verified" in status
    assert "3 matched" in status
    assert "Not verified" not in status
    assert "Ready for reviewed run" not in status
    assert dialog.reproduction_status_label.toolTip().startswith(status)
    _assert_tone(dialog, "success")
    assert dialog._reproduction_check() is None
    assert not dialog.run_button.isEnabled()


def test_completed_banner_keeps_recorded_verification_without_rerun_authority(
    qtbot, tmp_path
):
    dialog, _ = _finish_verified_run(qtbot, tmp_path)
    _assert_historical_success(dialog)

    dialog.mark_plan_historical_after_run()
    dialog.tabs.setCurrentIndex(1)
    dialog.tabs.setCurrentIndex(3)
    dialog._sync_workspace()
    _assert_historical_success(dialog)

    requested = []
    dialog.runRequested.connect(requested.append)
    dialog._request_run()
    assert requested == []
    assert dialog._reproduction_block_reason()
    _assert_historical_success(dialog)


def test_completed_banner_uses_returned_manifest_not_a_retained_preflight(
    qtbot, tmp_path
):
    request = _three_input_request()
    dialog, preview, _ = _dialog(qtbot, tmp_path, request=request)
    dialog.apply_preview_result(
        replace(preview, reproduction=_verified_check(request)),
        preview_representative=False,
    )
    dialog.begin_run(3)
    dialog._preview_result = None
    dialog.finish_run(_run_result(tmp_path, request))
    _assert_historical_success(dialog)


@pytest.mark.parametrize(
    "change", ["input", "output", "workflow", "config", "request", "new-data"]
)
def test_completed_evidence_is_cleared_by_workspace_changes(
    qtbot, tmp_path, monkeypatch, change
):
    dialog, preview = _finish_verified_run(qtbot, tmp_path)
    if change == "input":
        dialog.input_edit.setText(str(tmp_path / "other-inputs"))
    elif change == "output":
        dialog.output_edit.setText(str(tmp_path / "other-outputs"))
    elif change == "workflow":
        dialog.invalidate_for_workflow_change()
    elif change == "config":
        dialog._apply_config(preview.config)
    elif change == "request":
        dialog.set_reproduction_request(_three_input_request(version="0.0.1"))
    else:
        monkeypatch.setattr(QMessageBox, "question", lambda *_a, **_k: QMessageBox.Yes)
        dialog._use_reproduction_new_data()

    assert "Run complete" not in dialog.reproduction_status_label.text()
    assert dialog._reproduction_check() is None
    if change == "new-data":
        assert "comparison is off" in dialog.reproduction_mode_label.text()
        _assert_tone(dialog, "info")
    else:
        assert "Not verified" in dialog.reproduction_status_label.text()
        assert not dialog.run_button.isEnabled()
        _assert_tone(dialog, "warning")


def test_fresh_check_clears_completed_evidence_before_async_result(qtbot, tmp_path):
    dialog, preview = _finish_verified_run(qtbot, tmp_path)
    observed = []

    def checking(_values, _limit):
        observed.append(dialog.reproduction_status_label.text())
        return True

    dialog._actions = replace(dialog._actions, check_batch=checking)
    assert dialog._check_batch()
    assert observed == ["Checking the complete collection…"]
    _assert_tone(dialog, "warning")
    dialog.apply_preview_result(preview, preview_representative=False)
    assert "Run complete" not in dialog.reproduction_status_label.text()
    assert "Ready for reviewed run" in dialog.reproduction_status_label.text()
    assert dialog._reproduction_check() is preview.reproduction
    assert dialog.run_button.isEnabled()


@pytest.mark.parametrize("transition", ["begin-run", "run-error", "preflight-error"])
def test_new_or_error_run_cannot_reuse_prior_completed_banner(
    qtbot, tmp_path, transition
):
    dialog, _ = _finish_verified_run(qtbot, tmp_path)
    if transition == "begin-run":
        dialog.begin_run(3)
    elif transition == "run-error":
        dialog.show_run_error("Execution failed before returning a manifest.")
    else:
        dialog.show_preflight_error("The source is missing.")
    assert "Run complete" not in dialog.reproduction_status_label.text()
    _assert_tone(dialog, "warning")
    assert dialog._reproduction_check() is None
    assert not dialog.run_button.isEnabled()


@pytest.mark.parametrize(
    "outcome",
    [
        BatchStatus.PARTIAL,
        BatchStatus.FAILED,
        BatchStatus.CANCELLED,
        BatchStatus.SKIPPED,
    ],
)
def test_incomplete_run_retains_input_evidence_but_never_a_success_banner(
    qtbot, tmp_path, outcome
):
    request = _three_input_request()
    result = _run_result(
        tmp_path, request, statuses=(BatchStatus.COMPLETED,) * 2 + (outcome,)
    )
    dialog, _ = _finish_verified_run(
        qtbot, tmp_path, request=request, run_result=result
    )
    status = dialog.reproduction_status_label.text()
    assert "Original inputs verified" in status
    assert "3 matched" in status
    if outcome in (BatchStatus.PARTIAL, BatchStatus.FAILED):
        assert "issues" in status.lower()
    else:
        assert outcome.value in status.lower()
    assert "Run complete · Original inputs verified" not in status
    assert "Ready for reviewed run" not in status
    _assert_tone(dialog, "warning")
    assert dialog._reproduction_check() is None
    assert not dialog.run_button.isEnabled()


def test_cleanup_failure_does_not_produce_a_green_completed_banner(qtbot, tmp_path):
    request = _three_input_request()
    result = _run_result(tmp_path, request, has_failures=True)
    dialog, _ = _finish_verified_run(
        qtbot, tmp_path, request=request, run_result=result
    )
    _assert_tone(dialog, "warning")
    assert "Run complete · Original inputs verified" not in (
        dialog.reproduction_status_label.text()
    )


def test_completed_version_deviation_remains_explicit_and_amber(qtbot, tmp_path):
    request = _three_input_request(version="0.0.1")
    request = replace(
        request,
        version_override={
            "recorded_vipp_version": "0.0.1",
            "current_vipp_version": current_vipp_version(),
        },
    )
    dialog, _ = _finish_verified_run(qtbot, tmp_path, request=request)
    assert "Version difference acknowledged (deviation)" in (
        dialog.reproduction_mode_label.text()
    )
    assert "Original inputs verified" in dialog.reproduction_status_label.text()
    assert "Not verified" not in dialog.reproduction_status_label.text()
    assert "Ready for reviewed run" not in dialog.reproduction_status_label.text()
    _assert_tone(dialog, "warning")
    assert dialog._reproduction_check() is None
    assert not dialog.run_button.isEnabled()


@pytest.mark.parametrize(
    "bad_evidence",
    [
        None,
        {},
        {"status": "verified"},
        {"reference_sha256": "0" * 64},
        {"can_run": False},
        {"matched_count": 0, "rows": []},
        {"matched_count": "not a count"},
        {"mismatch_count": 1},
        {"status": "mismatch"},
        {"current_vipp_version": "0.0.1"},
    ],
)
def test_missing_invalid_or_foreign_manifest_checks_do_not_reuse_green_preflight(
    qtbot, tmp_path, bad_evidence
):
    request = _three_input_request()
    result = _run_result(tmp_path, request)
    if bad_evidence in (None, {}, {"status": "verified"}):
        result.manifest.reproduction = bad_evidence
    else:
        result.manifest.reproduction.update(bad_evidence)
    dialog, _ = _finish_verified_run(
        qtbot, tmp_path, request=request, run_result=result
    )
    assert "Ready for reviewed run" not in dialog.reproduction_status_label.text()
    _assert_tone(dialog, "warning")
    assert dialog._reproduction_check() is None
    assert not dialog.run_button.isEnabled()
