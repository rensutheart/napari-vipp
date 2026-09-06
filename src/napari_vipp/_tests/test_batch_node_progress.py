"""Sample-level node counts are independent of nested operation progress units."""

from types import SimpleNamespace

import pytest

from napari_vipp._tests.test_batch_redesign_host import batch_case as _case
from napari_vipp._tests.test_batch_run_startup import _request
from napari_vipp.core.batch import _batch_execution_callbacks
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.ui.batch import CollectionBatchDialog
from napari_vipp.ui.batch_workers import CollectionBatchWorker


@pytest.mark.parametrize("bypass", [False, True])
def test_counter_uses_runnable_nodes_and_distinguishes_repeated_operation_types(bypass):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    first = pipeline.add_node("gaussian_blur")
    second = pipeline.add_node("gaussian_blur")
    unused = pipeline.add_node("gaussian_blur")
    loose = pipeline.add_node("gaussian_blur")
    output = pipeline.add_node("batch_output")
    assert pipeline.connect("input", first.id).success
    assert pipeline.connect(first.id, second.id).success
    assert pipeline.connect(second.id, output.id).success
    assert pipeline.connect("input", unused.id).success
    if bypass:
        pipeline.set_node_execution_mode(first.id, "bypass")
    updates = []
    started, finished, progress, _ = _batch_execution_callbacks(
        item_index=1,
        item_total=14,
        batch_id="sample_01",
        pipeline=pipeline,
        callback=updates.append,
        target_node_ids=frozenset({output.id}),
    )
    for ordinal, node_id in enumerate(("input", first.id, second.id, output.id), 1):
        node = pipeline.nodes[node_id]
        started(node_id)
        started(node_id)  # Internal retries must not double-count a graph node.
        progress(node.operation_id, 800, 1000, "Tile 800 of 1000")
        finished(SimpleNamespace(node_id=node_id, operation_id=node.operation_id))
        assert all(event.node_current == ordinal for event in updates[-4:])
        assert all(event.node_total == 4 for event in updates[-4:])
    assert all(event.node_id not in {unused.id, loose.id} for event in updates)
    assert updates[-2].current == 800 and updates[-2].node_current == 4


def test_upper_count_updates_without_replacing_lower_operation_units(qtbot):
    dialog = CollectionBatchDialog()
    qtbot.addWidget(dialog)
    dialog.begin_run(14)
    dialog.update_run_progress(6, 14, "sample_06", "running")
    for current in (1, 12, 32):
        dialog.update_operation_progress(
            6,
            14,
            "sample_06",
            "blur",
            "gaussian_blur",
            4,
            9,
            "Tile 4 of 9",
            node_current=current,
            node_total=32,
        )
        for surface in (dialog, dialog.results_panel):
            assert surface.run_progress_label.text().endswith(
                f"Running (node {current}/32)"
            )
            assert "sample_06" in surface.run_progress_label.text()
            assert surface.operation_progress_bar.value() == 4
            assert surface.operation_progress_bar.maximum() == 9
            assert "node " not in surface.operation_progress_label.text()
    dialog.update_operation_progress(
        6,
        14,
        "sample_06",
        "output",
        "batch_stage_output",
        3,
        4,
        "Saving outputs",
    )
    assert dialog.results_panel.run_progress_label.text().endswith(
        "Running (node 32/32)"
    )
    dialog.update_run_progress(6, 14, "sample_06", "completed")
    assert "(node " not in dialog.results_panel.run_progress_label.text()
    dialog.update_run_progress(7, 14, "sample_07", "running")
    assert "(node " not in dialog.results_panel.run_progress_label.text()
    dialog.update_operation_progress(
        7,
        14,
        "sample_07",
        "input",
        "input",
        0,
        0,
        node_current=1,
        node_total=32,
    )
    assert dialog.results_panel.run_progress_label.text().endswith(
        "Running (node 1/32)"
    )
    dialog.update_run_progress(7, 14, "sample_07", "failed")
    dialog.begin_run(14)
    assert "(node " not in dialog.results_panel.run_progress_label.text()


def test_late_progress_does_not_move_counter_back_to_previous_item(qtbot):
    dialog = CollectionBatchDialog()
    qtbot.addWidget(dialog)
    dialog.begin_run(2)
    dialog.update_run_progress(1, 2, "first", "running")
    dialog.update_run_progress(1, 2, "first", "completed")
    dialog.update_run_progress(2, 2, "second", "running")
    dialog.update_operation_progress(
        2,
        2,
        "second",
        "input",
        "input",
        0,
        0,
        node_current=1,
        node_total=3,
    )
    expected = dialog.results_panel.run_progress_label.text()
    dialog.update_operation_progress(
        1,
        2,
        "first",
        "last",
        "gaussian_blur",
        1,
        1,
        node_current=3,
        node_total=3,
    )
    assert dialog.results_panel.run_progress_label.text() == expected


@pytest.mark.parametrize("skip_all", [False, True])
def test_real_worker_counts_each_sample_and_does_not_invent_nodes_for_kept_items(
    qtbot,
    tmp_path,
    skip_all,
):
    case = _case.__wrapped__(tmp_path)
    if skip_all:
        case[1]["existing_file_policy"] = "skip"
    request = _request(case)
    if skip_all:
        for item in request.expected_items:
            path = item.outputs[0].path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"kept result")
        request = _request(case)
    updates, outcomes = [], []
    worker = CollectionBatchWorker(request)
    worker.signals.operation_progress.connect(
        lambda update: updates.append(update.progress)
    )
    worker.signals.finished.connect(outcomes.append)
    worker.run()
    assert len(outcomes) == 1 and not outcomes[0].error
    starts = [update for update in updates if update.message == "Node started."]
    if skip_all:
        assert starts == []
        assert outcomes[0].result.summary["skipped"] == 2
    else:
        assert [
            (event.item_index, event.node_current, event.node_total) for event in starts
        ] == [
            (1, 1, 2),
            (1, 2, 2),
            (2, 1, 2),
            (2, 2, 2),
        ]
        assert outcomes[0].result.summary["completed"] == 2
