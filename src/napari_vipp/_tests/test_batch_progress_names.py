"""The nested bar names the executing node, not the sample again."""

from types import SimpleNamespace

import pytest
from qtpy.QtWidgets import QLabel, QProgressBar

from napari_vipp._widget import VippWidget
from napari_vipp.core.batch import BatchExecutionProgress, _batch_execution_callbacks
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.ui.batch import CollectionBatchDialog
from napari_vipp.ui.batch_progress import operation_progress_text
from napari_vipp.ui.batch_workers import CollectionBatchOperationProgress


@pytest.mark.parametrize("node_title", ["", "Channel colours → RGB"])
def test_nested_label_has_display_title_and_no_duplicate_sample(qtbot, node_title):
    dialog = CollectionBatchDialog()
    qtbot.addWidget(dialog)
    dialog.begin_run(14)
    dialog.update_run_progress(7, 14, "sample_007_Airyscan_Processing", "running")
    dialog.update_operation_progress(
        7,
        14,
        "sample_007_Airyscan_Processing",
        "composite_to_rgb_1",
        "composite_to_rgb",
        0,
        0,
        "Node started.",
        node_title=node_title,
    )
    expected = f"{node_title or 'Composite → RGB'} · Node started."
    for surface in (dialog, dialog.results_panel):
        assert surface.operation_progress_label.text() == expected
        assert "sample_007_Airyscan_Processing" in surface.run_progress_label.text()
        assert "composite_to_rgb_1" in surface.operation_progress_label.toolTip()
        assert "sample_007_Airyscan_Processing" in (
            surface.operation_progress_label.toolTip()
        )


def test_node_callbacks_carry_actual_node_title_for_all_progress_events():
    pipeline = PrototypePipeline()
    node = pipeline.add_node("composite_to_rgb")
    # Compatibility nodes can derive their title from authored parameters.
    # Use the exact node title, not a prettified operation ID or registry guess.
    node.title = "Channel colours → RGB"
    updates = []
    started, finished, progress, _ = _batch_execution_callbacks(
        item_index=7,
        item_total=14,
        batch_id="sample_007",
        pipeline=pipeline,
        callback=updates.append,
    )
    started(node.id)
    progress(node.operation_id, 1, 2, "Channel 1 of 2")
    finished(SimpleNamespace(node_id=node.id, operation_id=node.operation_id))
    assert len(updates) == 3
    assert all(update.node_title == node.title for update in updates)
    assert all(update.operation_id == "composite_to_rgb" for update in updates)


@pytest.mark.parametrize("active", [True, False])
def test_host_forwards_executing_node_title_even_when_another_tab_is_active(
    qtbot,
    active,
):
    calls = []
    dialog = SimpleNamespace(
        update_operation_progress=lambda *args, **kwargs: calls.append(kwargs),
    )
    label, bar = QLabel(), QProgressBar()
    qtbot.addWidget(label)
    qtbot.addWidget(bar)
    host = SimpleNamespace(
        _active_collection_batch_job=SimpleNamespace(
            job_id=9,
            origin_session_id="origin",
            dialog=dialog,
        ),
        _workflow_tab_is_active=lambda _id: active,
        pipeline_busy_label=label,
        pipeline_busy_bar=bar,
    )
    host._collection_batch_operation_progress = lambda progress, **kwargs: (
        VippWidget._collection_batch_operation_progress(host, progress, **kwargs)
    )
    event = BatchExecutionProgress(
        7,
        14,
        "sample_007",
        "composite_to_rgb_1",
        "composite_to_rgb",
        0,
        0,
        "Node started.",
        node_title="Channel colours → RGB",
        node_current=12,
        node_total=32,
    )
    VippWidget._on_collection_batch_worker_operation_progress(
        host,
        CollectionBatchOperationProgress(9, event),
    )
    assert calls == [
        {
            "node_title": "Channel colours → RGB",
            "node_current": 12,
            "node_total": 32,
        }
    ]
    if active:
        assert label.text() == "Batch 7/14: Channel colours → RGB · Node started."


def test_non_node_stages_and_missing_ids_have_readable_fallbacks():
    assert operation_progress_text("batch_publish_output") == "Save output file"
    assert operation_progress_text("third_party_filter") == "Third party filter"
    assert operation_progress_text("") == "Current operation"
