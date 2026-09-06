from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QPalette

from napari_vipp.core.batch import (
    BatchConfig,
    BatchItemPlan,
    BatchItemRecord,
    BatchManifest,
    BatchOutputConfig,
    BatchOutputPlan,
    BatchOutputRecord,
    BatchRunResult,
    BatchSourceConfig,
    BatchStatus,
    ExistingFilePolicy,
)
from napari_vipp.ui import batch_results, file_reveal
from napari_vipp.ui.batch import BatchPreviewResult
from napari_vipp.ui.batch_results import BatchResultsPanel


@pytest.fixture(autouse=True)
def no_file_manager(monkeypatch):
    calls = []
    monkeypatch.setattr(file_reveal, "_launch", calls.append)
    return calls


@pytest.fixture
def file_reveal_requests(monkeypatch, request):
    """Keep exact requested files observable above the OS-specific fallback."""
    platform_name = request.param
    paths = []
    reveal = file_reveal.reveal_file

    def record_reveal(path):
        paths.append(Path(path))
        return reveal(path, platform_name=platform_name)

    monkeypatch.setattr(batch_results, "reveal_file", record_reveal)
    return platform_name, paths


def _expected_reveal_arguments(path, platform_name):
    if platform_name == "win32":
        return ["explorer.exe", "/select,", str(path)]
    if platform_name == "darwin":
        return ["open", "-R", str(path)]
    return ["xdg-open", str(path.parent)]


def _preview(tmp_path, count=3):
    config = BatchConfig(
        workflow_file=tmp_path / "workflow.json",
        workflow_sha256="a" * 64,
        output_dir=tmp_path / "outputs",
        sources=(BatchSourceConfig("input", "Input", tmp_path / "inputs", "*.npy"),),
        outputs=(
            BatchOutputConfig(
                "out",
                "Output",
                "result",
                "image",
                "npy",
                "",
                "{source_stem}",
            ),
        ),
    )
    items = tuple(
        BatchItemPlan(
            index=i,
            batch_id=f"{i:04d}_sample-{i}",
            primary_source=tmp_path / "inputs" / f"sample-{i}.npy",
            source_paths={"input": tmp_path / "inputs" / f"sample-{i}.npy"},
            outputs=(
                BatchOutputPlan(
                    "out",
                    "Output",
                    "result",
                    "image",
                    "npy",
                    config.output_dir / f"sample-{i}.npy",
                    ExistingFilePolicy.ERROR,
                ),
            ),
        )
        for i in range(1, count + 1)
    )
    return BatchPreviewResult((), count, 0, True, items, config)


def _record(item, status, *, output_status=None, error="", timing=True):
    output = item.outputs[0]
    return BatchItemRecord(
        index=item.index,
        batch_id=item.batch_id,
        sources=(),
        status=status,
        started_at="2026-09-05T10:00:00+00:00" if timing else "",
        finished_at="2026-09-05T10:00:12+00:00" if timing else "",
        error_message=error,
        outputs=(
            BatchOutputRecord(
                node_id=output.node_id,
                node_title=output.node_title,
                tag=output.tag,
                kind=output.kind,
                format=output.format,
                path=str(output.path),
                existing_file_policy=output.existing_file_policy,
                existed_at_preflight=False,
                status=output_status or status,
                error_message=error,
            ),
        ),
    )


def _result(preview, records, *, timing=True):
    manifest = BatchManifest(
        run_id="test-run",
        started_at="2026-09-05T10:00:00Z" if timing else "",
        finished_at="2026-09-05T10:01:02Z" if timing else "",
        workflow_sha256="a" * 64,
        config_sha256="b" * 64,
        effective_config_sha256="b" * 64,
        workflow_file="workflow.json",
        config_file="config.json",
        output_dir=str(preview.config.output_dir),
        runtime={},
        workflow_document={},
        config_document={},
        compute={},
        items=tuple(records),
    )
    return BatchRunResult(
        manifest=manifest,
        manifest_path=preview.config.output_dir / "vipp_batch_manifest.json",
        saved_paths=tuple(
            output.path
            for record in records
            for output in record.outputs
            if output.status == BatchStatus.COMPLETED
        ),
    )


def test_review_shows_real_plan_and_full_collection_with_paging(qtbot, tmp_path):
    preview = _preview(tmp_path, 1200)
    panel = BatchResultsPanel()
    qtbot.addWidget(panel)
    panel.set_plan(preview)
    assert "1,200 items · 1,200 to create" in panel.review_labels["Batch"].text()
    assert panel.items_table.rowCount() == panel.PAGE_SIZE
    assert panel.next_page_button.isEnabled()
    assert panel.select_item(1199)
    assert "1,200" in panel.page_label.text()
    requested = []
    panel.itemRequested.connect(requested.append)
    panel.items_table.linkActivated.emit(99, 0)
    assert requested == [1199]
    assert "sample-1200" in panel.selected_item_label.text()
    assert panel.output_table.item(0, 1).text() == "To create"
    assert not panel.reveal_button.isEnabled()


def test_paging_controls_only_appear_when_results_span_multiple_pages(qtbot, tmp_path):
    panel = BatchResultsPanel()
    qtbot.addWidget(panel)
    assert panel.previous_page_button.isHidden()
    assert panel.next_page_button.isHidden()

    for count in (3, panel.PAGE_SIZE, panel.PAGE_SIZE + 1, 3):
        panel.set_plan(_preview(tmp_path, count))
        multiple_pages = count > panel.PAGE_SIZE
        assert panel.previous_page_button.isHidden() is not multiple_pages
        assert panel.next_page_button.isHidden() is not multiple_pages
        assert not panel.previous_page_button.isEnabled()
        assert panel.next_page_button.isEnabled() is multiple_pages
        if multiple_pages:
            panel.select_item(count - 1)
            assert not panel.previous_page_button.isHidden()
            assert not panel.next_page_button.isHidden()
            assert panel.previous_page_button.isEnabled()
            assert not panel.next_page_button.isEnabled()


def test_reported_statuses_timing_and_file_existence_are_separate(qtbot, tmp_path):
    preview = _preview(tmp_path)
    preview.config.output_dir.mkdir()
    preview.items[0].outputs[0].path.touch()
    records = (
        _record(preview.items[0], BatchStatus.COMPLETED),
        _record(
            preview.items[1],
            BatchStatus.PARTIAL,
            output_status=BatchStatus.FAILED,
            error="Disk is full.",
        ),
        _record(preview.items[2], BatchStatus.CANCELLED),
    )
    result = _result(preview, records)
    result.manifest_path.touch()
    panel = BatchResultsPanel()
    qtbot.addWidget(panel)
    panel.set_plan(preview)
    panel.finish_run(result)

    assert "1 of 3 outputs saved" in panel.summary_label.text()
    assert "1 partial" in panel.summary_label.text()
    assert panel.elapsed_label.text() == "Total elapsed 01:02"
    assert panel.items_table.item(0, 3).text() == "00:12"
    assert panel.output_table.item(0, 1).text() == "Saved"
    assert panel.reveal_button.isEnabled()
    assert panel.has_run_report
    panel.select_item(1)
    assert panel.output_table.item(0, 1).text() == "Failed · not created"
    assert panel.item_error_label.text() == "Disk is full."
    assert not panel.reveal_button.isEnabled()
    panel.select_item(2)
    assert panel.output_table.item(0, 1).text() == "Not created · cancelled"
    assert not panel._timer.isActive()
    assert panel.run_progress_bar.maximum() == 3
    assert panel.run_progress_bar.value() == 2


@pytest.mark.parametrize(
    "file_reveal_requests", ["win32", "darwin", "linux"], indirect=True
)
def test_exact_file_and_report_actions_do_not_open_real_windows(
    qtbot,
    tmp_path,
    no_file_manager,
    file_reveal_requests,
):
    preview = _preview(tmp_path, 1)
    preview.config.output_dir.mkdir()
    path = preview.items[0].outputs[0].path
    path.touch()
    result = _result(preview, (_record(preview.items[0], BatchStatus.COMPLETED),))
    result.manifest_path.touch()
    panel = BatchResultsPanel()
    qtbot.addWidget(panel)
    panel.set_plan(preview)
    panel.finish_run(result)
    platform_name, requested_paths = file_reveal_requests
    qtbot.mouseClick(panel.reveal_button, Qt.LeftButton)
    assert requested_paths == [path]
    assert no_file_manager[-1] == _expected_reveal_arguments(path, platform_name)
    assert not panel.run_report.isHidden()
    assert len(no_file_manager) == 1
    qtbot.mouseClick(panel.run_report.manifest_button, Qt.LeftButton)
    assert requested_paths == [path, result.manifest_path]
    assert no_file_manager[-1] == _expected_reveal_arguments(
        result.manifest_path, platform_name
    )

    path.unlink()
    panel.output_table.linkActivated.emit(0, 0)
    assert len(no_file_manager) == 2
    assert "missing" in panel.file_action_label.text()
    assert panel.output_table.item(0, 1).text() == "Saved · file missing"
    assert not panel.reveal_button.isEnabled()


def test_running_keeps_file_review_read_only_and_does_not_invent_saves(
    qtbot,
    tmp_path,
    monkeypatch,
):
    now = [100.0]
    monkeypatch.setattr(batch_results.time, "monotonic", lambda: now[0])
    preview = _preview(tmp_path, 2)
    preview.config.output_dir.mkdir()
    preview.items[0].outputs[0].path.touch()
    panel = BatchResultsPanel()
    qtbot.addWidget(panel)
    panel.set_plan(preview)
    panel.begin_run(2)
    panel.update_item(1, 2, preview.items[0].batch_id, "running")
    assert panel.reveal_button.isEnabled()
    assert panel.review_item_button.isEnabled()
    assert panel.output_table.item(0, 1).text() == "Existing · awaiting report"
    now[0] = 109
    panel.update_item(1, 2, preview.items[0].batch_id, "completed")
    assert panel.items_table.item(0, 3).text() == "00:09"
    assert panel.run_progress_bar.value() == 1
    assert panel.elapsed_label.text() == "Elapsed 00:09"
    assert panel.items_table.item(0, 2).text() == "1 to create"
    panel.set_stopping()
    assert "Stopping safely" in panel.summary_label.text()
    panel.show_error("Source changed during processing.")
    assert not panel._timer.isActive()


def test_missing_timing_is_unknown_and_failed_existing_file_not_marked_saved(
    qtbot,
    tmp_path,
):
    preview = _preview(tmp_path, 1)
    preview.config.output_dir.mkdir()
    preview.items[0].outputs[0].path.touch()
    record = _record(preview.items[0], BatchStatus.FAILED, timing=False)
    panel = BatchResultsPanel()
    qtbot.addWidget(panel)
    panel.set_plan(preview)
    panel.finish_run(_result(preview, (record,), timing=False))
    assert panel.items_table.item(0, 3).text() == "—"
    assert panel.elapsed_label.text() == "Total elapsed —"
    assert panel.output_table.item(0, 1).text() == "Failed · existing file remains"
    assert "0 of 1 outputs saved" in panel.summary_label.text()
    assert panel.has_run_report
    assert not panel.run_report.manifest_button.isEnabled()


def test_historical_result_is_retained_on_invalidation(qtbot, tmp_path):
    preview = _preview(tmp_path, 1)
    panel = BatchResultsPanel()
    qtbot.addWidget(panel)
    panel.set_plan(preview)
    panel.finish_run(
        _result(preview, (_record(preview.items[0], BatchStatus.COMPLETED),))
    )
    panel.invalidate_plan()
    assert panel.items_table.item(0, 1).text() == "Completed"
    assert "before running again" in panel.summary_label.text()
    panel.set_plan(preview)
    assert panel.items_table.item(0, 1).text() == "Not run"
    assert not panel.has_run_report


@pytest.mark.parametrize(
    "file_reveal_requests", ["win32", "darwin", "linux"], indirect=True
)
def test_partial_item_uses_each_output_record_and_stable_run_report(
    qtbot,
    tmp_path,
    no_file_manager,
    file_reveal_requests,
):
    preview = _preview(tmp_path, 1)
    preview.config.output_dir.mkdir()
    saved_path = preview.items[0].outputs[0].path
    saved_path.touch()
    base = _record(preview.items[0], BatchStatus.PARTIAL)
    item = replace(
        base,
        outputs=(
            replace(base.outputs[0], status=BatchStatus.COMPLETED),
            replace(
                base.outputs[0],
                path=str(saved_path.with_suffix(".tsv")),
                status=BatchStatus.FAILED,
                error_message="Write failed.",
            ),
        ),
    )
    result = _result(preview, (item,))
    archive_path = preview.config.output_dir / "run-archive.json"
    archive_path.touch()
    result = replace(result, manifest_archive_path=archive_path)
    panel = BatchResultsPanel()
    qtbot.addWidget(panel)
    panel.set_plan(preview)
    panel.finish_run(result)
    assert panel.items_table.item(0, 2).text() == "1 / 2 saved"
    assert panel.output_table.item(0, 1).text() == "Saved"
    assert panel.output_table.item(1, 1).text() == "Failed · not created"
    assert "Write failed." in panel.output_table.item(1, 1).toolTip()
    assert "1 of 2 outputs saved" in panel.summary_label.text()
    qtbot.mouseClick(panel.run_report.manifest_button, Qt.LeftButton)
    platform_name, requested_paths = file_reveal_requests
    assert requested_paths == [archive_path]
    assert no_file_manager[-1] == _expected_reveal_arguments(
        archive_path, platform_name
    )


@pytest.mark.parametrize("dark", [False, True])
def test_panel_uses_palette_semantics_in_light_and_dark(qtbot, tmp_path, dark):
    panel = BatchResultsPanel()
    qtbot.addWidget(panel)
    palette = QPalette(panel.palette())
    palette.setColor(QPalette.Base, QColor("#181b20" if dark else "#ffffff"))
    palette.setColor(QPalette.Text, QColor("#ffffff" if dark else "#202124"))
    panel.setPalette(palette)
    panel.set_plan(replace(_preview(tmp_path, 1), collision_count=1))
    colors = batch_results.theme_colors(palette)
    assert colors.warning.surface.name() in panel.summary_label.styleSheet()
    assert colors.warning.foreground.name() in panel.summary_label.styleSheet()


@pytest.mark.parametrize(
    "policy, label",
    [
        (ExistingFilePolicy.ERROR, "Ask before overwrite"),
        (ExistingFilePolicy.OVERWRITE, "Overwrite without asking"),
        (ExistingFilePolicy.SKIP, "Skip existing files"),
    ],
)
def test_review_existing_file_policy_matches_interactive_batch(
    qtbot,
    tmp_path,
    policy,
    label,
):
    preview = _preview(tmp_path, 1)
    panel = BatchResultsPanel()
    qtbot.addWidget(panel)
    panel.set_plan(
        replace(preview, config=replace(preview.config, existing_file_policy=policy))
    )
    assert panel.review_labels["Existing files"].text() == label


def test_runtime_cleanup_failure_does_not_show_success(qtbot, tmp_path):
    preview = _preview(tmp_path, 1)
    result = _result(preview, (_record(preview.items[0], BatchStatus.COMPLETED),))
    result = replace(
        result,
        manifest=replace(
            result.manifest,
            compute={"runtime_cleanup_succeeded": False},
        ),
    )
    panel = BatchResultsPanel()
    qtbot.addWidget(panel)
    panel.set_plan(preview)
    panel.finish_run(result)
    assert "Finished with issues" in panel.summary_label.text()
    assert "Runtime cleanup did not finish successfully" in panel.summary_label.text()
    assert "1 completed" in panel.summary_label.text()
    assert "1 of 1 outputs saved" in panel.summary_label.text()
