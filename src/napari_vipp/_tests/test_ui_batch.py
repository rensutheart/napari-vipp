from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QScrollArea

from napari_vipp.core.batch import (
    BatchConfig,
    BatchItemPlan,
    BatchOutputConfig,
    BatchSourceConfig,
    BatchStatus,
)
from napari_vipp.ui.batch import (
    BatchPreviewResult,
    BatchPreviewRow,
    CollectionBatchActions,
    CollectionBatchDialog,
)


def _preview_result(tmp_path, *, count: int = 3) -> BatchPreviewResult:
    config = BatchConfig(
        workflow_file=tmp_path / "workflow.json",
        workflow_sha256="a" * 64,
        output_dir=tmp_path / "outputs",
        sources=(
            BatchSourceConfig(
                node_id="input",
                title="Image Source",
                input_dir=tmp_path / "inputs",
                pattern="*.npy",
            ),
        ),
        outputs=(
            BatchOutputConfig(
                node_id="output",
                node_title="Batch Output",
                tag="result",
                kind="image",
                format="npy",
                subfolder="",
                filename_template="{source_stem}__{tag}",
            ),
        ),
    )
    items = tuple(
        BatchItemPlan(
            index=index,
            batch_id=f"{index:04d}_field-{index}",
            primary_source=tmp_path / "inputs" / f"field-{index}.npy",
            source_paths={"input": tmp_path / "inputs" / f"field-{index}.npy"},
            outputs=(),
        )
        for index in range(1, count + 1)
    )
    rows = tuple(
        BatchPreviewRow(
            batch_index=item.index,
            batch_id=item.batch_id,
            sources=dict(item.source_paths),
            outputs=[tmp_path / "outputs" / f"field-{item.index}.npy"],
            output_statuses=("new",),
        )
        for item in items
    )
    return BatchPreviewResult(
        rows=rows,
        total_items=len(items),
        collision_count=0,
        explicit_outputs=True,
        items=items,
        config=config,
    )


def _actions(result, previewed: list[int]) -> CollectionBatchActions:
    return CollectionBatchActions(
        preview_batch=lambda _values, _limit: result,
        choose_demo=lambda _parent: None,
        source_rows=lambda: [],
        load_config=lambda _path: result.config,
        save_config=lambda _path, _values: (),
        preview_item=previewed.append,
    )


def test_batch_dialog_run_request_does_not_accept_workspace(qtbot, tmp_path):
    result = _preview_result(tmp_path)
    dialog = CollectionBatchDialog(actions=_actions(result, []))
    qtbot.addWidget(dialog)
    requests = []
    dialog.runRequested.connect(requests.append)

    qtbot.mouseClick(dialog.run_button, Qt.LeftButton)

    assert len(requests) == 1
    assert requests[0]["continue_on_error"] is True
    assert dialog.result() != dialog.Accepted
    assert dialog.close_button.text() == "Close"


def test_batch_preview_auto_loads_first_representative_and_selection_is_explicit(
    qtbot,
    tmp_path,
):
    result = _preview_result(tmp_path)
    previewed: list[int] = []
    dialog = CollectionBatchDialog(actions=_actions(result, previewed))
    qtbot.addWidget(dialog)

    assert dialog._preview_batch()

    assert previewed == [0]
    assert dialog.preview_table.columnCount() == 5
    assert dialog.preview_table.item(0, 3).text() == "new"
    assert dialog.preview_table.item(0, 4).text() == "Not run"
    assert "representative calculation" in dialog.graph_preview_status.text()
    assert dialog.select_preview_item(1)
    assert previewed == [0]

    qtbot.mouseClick(dialog.preview_item_button, Qt.LeftButton)
    assert previewed == [0, 1]

    dialog.preview_table.itemDoubleClicked.emit(dialog.preview_table.item(2, 1))
    assert previewed == [0, 1, 2]


def test_batch_preview_status_tracks_full_plan_item_beyond_table_limit(
    qtbot,
    tmp_path,
):
    full = _preview_result(tmp_path, count=26)
    limited = replace(full, rows=full.rows[:25])
    dialog = CollectionBatchDialog(actions=_actions(limited, []))
    qtbot.addWidget(dialog)
    assert dialog._preview_batch()

    assert dialog.preview_table.rowCount() == 25
    assert dialog.select_preview_item(25)
    assert not dialog.preview_table.selectionModel().selectedRows()
    assert "item 26 of 26" in dialog.graph_preview_status.text()

    dialog.begin_run(26)
    dialog.preview_table.selectRow(24)
    dialog.update_run_progress(26, 26, limited.items[25].batch_id, "running")
    assert not dialog.preview_table.selectionModel().selectedRows()
    assert "Item 26 of 26" in dialog.run_progress_label.text()


def test_failed_preview_emits_invalidation_and_clears_demo_identity(qtbot, tmp_path):
    result = _preview_result(tmp_path)

    def fail_preview(_values, _limit):
        raise ValueError("source disappeared")

    actions = replace(_actions(result, []), preview_batch=fail_preview)
    dialog = CollectionBatchDialog(actions=actions)
    qtbot.addWidget(dialog)
    dialog.set_demo_context(SimpleNamespace(root=tmp_path))
    invalidations: list[None] = []
    dialog.previewInvalidated.connect(lambda: invalidations.append(None))

    assert not dialog._preview_batch()
    assert invalidations == [None]
    assert dialog.run_button.text() == "Run batch"
    assert dialog.demo_guide_label.isHidden()


def test_completed_run_marks_preflight_historical_without_erasing_evidence(
    qtbot,
    tmp_path,
):
    result = _preview_result(tmp_path)
    dialog = CollectionBatchDialog(actions=_actions(result, []))
    qtbot.addWidget(dialog)
    assert dialog._preview_batch()
    dialog.begin_run(3)
    run_result = SimpleNamespace(
        manifest=SimpleNamespace(
            items=tuple(
                SimpleNamespace(index=index, status=BatchStatus.COMPLETED)
                for index in range(1, 4)
            )
        ),
        summary={"completed": 3, "partial": 0, "skipped": 0, "failed": 0},
        saved_paths=(),
        manifest_path=tmp_path / "manifest.json",
    )
    dialog.finish_run(run_result)

    dialog.mark_plan_historical_after_run()

    assert dialog._preview_result is None
    assert [dialog.preview_table.item(row, 4).text() for row in range(3)] == [
        "Completed",
        "Completed",
        "Completed",
    ]
    assert "historical" in dialog.preview_status.text().lower()
    assert "3 completed" in dialog.run_progress_label.text()


def test_batch_dialog_retains_determinate_progress_and_restores_controls(
    qtbot,
    tmp_path,
):
    result = _preview_result(tmp_path)
    dialog = CollectionBatchDialog(actions=_actions(result, []))
    qtbot.addWidget(dialog)
    assert dialog._preview_batch()
    assert dialog.preview_item_button.isEnabled()
    assert not dialog.workflow_checkbox.isEnabled()

    dialog.begin_run(3)

    assert not dialog.run_button.isEnabled()
    assert not dialog.preview_button.isEnabled()
    assert not dialog.source_group.isEnabled()
    assert dialog.close_button.isEnabled()
    assert dialog.preview_table.item(0, 3).text() == "new"
    assert dialog.preview_table.item(0, 4).text() == "Pending"

    dialog.update_run_progress(2, 3, result.items[1].batch_id, "running")
    assert dialog.run_progress_bar.maximum() == 3
    assert dialog.run_progress_bar.value() == 1
    assert dialog.preview_table.item(1, 4).text() == "Running"

    dialog.update_run_progress(2, 3, result.items[1].batch_id, "completed")
    assert dialog.run_progress_bar.value() == 2
    assert dialog.preview_table.item(1, 4).text() == "Completed"

    run_result = SimpleNamespace(
        manifest=SimpleNamespace(
            items=(
                SimpleNamespace(index=1, status=BatchStatus.COMPLETED),
                SimpleNamespace(index=2, status=BatchStatus.FAILED),
                SimpleNamespace(index=3, status=BatchStatus.SKIPPED),
            )
        ),
        summary={"completed": 1, "partial": 0, "skipped": 1, "failed": 1},
        saved_paths=(tmp_path / "outputs" / "field-1.npy",),
        manifest_path=tmp_path / "outputs" / "vipp_batch_manifest.json",
    )
    dialog.finish_run(run_result, "Ground truth passed.")

    assert dialog.run_progress_bar.value() == 3
    assert dialog.preview_table.item(0, 4).text() == "Completed"
    assert dialog.preview_table.item(1, 4).text() == "Failed"
    assert dialog.preview_table.item(2, 4).text() == "Skipped"
    assert dialog.preview_table.item(0, 3).text() == "new"
    assert "1 completed" in dialog.run_progress_label.text()
    assert "vipp_batch_manifest.json" in dialog.run_result_label.text()
    assert "Ground truth passed" in dialog.run_result_label.text()
    assert dialog.run_button.isEnabled()
    assert dialog.preview_button.isEnabled()
    assert dialog.source_group.isEnabled()
    assert dialog.preview_item_button.isEnabled()
    assert not dialog.workflow_checkbox.isEnabled()

    assert dialog._preview_batch()
    assert dialog.run_group.isHidden()
    assert dialog.run_progress_bar.format() == "Not run"
    assert dialog.preview_table.item(0, 4).text() == "Not run"


def test_batch_dialog_error_restores_exact_control_state(qtbot, tmp_path):
    result = _preview_result(tmp_path)
    dialog = CollectionBatchDialog(actions=_actions(result, []))
    qtbot.addWidget(dialog)
    assert dialog._preview_batch()
    dialog.save_config_button.setEnabled(False)

    dialog.begin_run(3)
    dialog.update_run_progress(1, 3, result.items[0].batch_id, "running")
    dialog.show_run_error("A source changed during execution.")

    assert dialog.preview_table.item(0, 4).text() == "Failed"
    assert dialog.run_progress_bar.format() == "Failed"
    assert "source changed" in dialog.run_result_label.text()
    assert dialog.run_button.isEnabled()
    assert not dialog.save_config_button.isEnabled()


def test_batch_dialog_scrolls_compact_content_and_keeps_footer_fixed(
    qtbot,
    tmp_path,
):
    result = _preview_result(tmp_path)
    previewed: list[int] = []
    source_nodes = [
        {
            "node_id": f"input_{index}_with_a_long_identifier",
            "title": f"Microscope source {index}",
            "binding_mode": "collection",
        }
        for index in range(1, 4)
    ]
    dialog = CollectionBatchDialog(
        source_nodes=source_nodes,
        actions=_actions(result, previewed),
    )
    qtbot.addWidget(dialog)
    dialog.set_demo_context(
        SimpleNamespace(root=tmp_path / "a" / "long" / "demo" / "working-copy")
    )
    assert dialog._preview_batch()
    requests: list[dict[str, object]] = []
    dialog.runRequested.connect(requests.append)

    assert isinstance(dialog.content_scroll, QScrollArea)
    assert dialog.content_scroll.widgetResizable()
    assert dialog.content_scroll.widget() is dialog.content_widget
    assert dialog.content_widget.isAncestorOf(dialog.source_group)
    assert dialog.content_widget.isAncestorOf(dialog.preview_table)
    assert not dialog.content_widget.isAncestorOf(dialog.button_box)
    assert (
        dialog.content_scroll.verticalScrollBarPolicy()
        == Qt.ScrollBarAsNeeded
    )

    dialog.resize(640, dialog.minimumHeight())
    dialog.show()
    qtbot.waitUntil(dialog.isVisible)
    qtbot.waitUntil(
        lambda: dialog.content_scroll.verticalScrollBar().maximum() > 0
    )

    assert dialog.width() <= 640
    scroll_bar = dialog.content_scroll.verticalScrollBar()
    for position in (scroll_bar.minimum(), scroll_bar.maximum()):
        scroll_bar.setValue(position)
        assert dialog.run_button.isVisibleTo(dialog)
        assert dialog.close_button.isVisibleTo(dialog)

    assert dialog.select_preview_item(1)
    dialog.content_scroll.ensureWidgetVisible(dialog.preview_item_button, 12, 12)
    viewport = dialog.content_scroll.viewport()
    button_center = dialog.preview_item_button.mapTo(
        viewport,
        dialog.preview_item_button.rect().center(),
    )
    assert viewport.rect().contains(button_center)
    qtbot.mouseClick(dialog.preview_item_button, Qt.LeftButton)
    assert previewed == [0, 1]

    output_item = dialog.preview_table.item(0, 2)
    assert output_item.text() == "field-1.npy"
    assert str(result.rows[0].outputs[0]) in output_item.toolTip()
    header = dialog.preview_table.horizontalHeader()
    last_column_right = header.sectionViewportPosition(4) + header.sectionSize(4)
    assert last_column_right <= dialog.preview_table.viewport().width() + 4

    qtbot.mouseClick(dialog.run_button, Qt.LeftButton)
    assert len(requests) == 1
    dialog.begin_run(3)
    dialog.content_scroll.ensureWidgetVisible(dialog.run_group, 12, 12)
    assert dialog.run_group.isVisibleTo(dialog.content_scroll.viewport())
    assert dialog.close_button.isVisibleTo(dialog)
