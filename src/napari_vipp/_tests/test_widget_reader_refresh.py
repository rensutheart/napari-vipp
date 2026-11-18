"""A changed TIFF reader must not be silently accepted by Calculate all."""

from dataclasses import replace

import numpy as np
import pytest
import tifffile

from napari_vipp._tests.test_widget import _Viewer
from napari_vipp._widget import VippWidget
from napari_vipp.core.batch import BatchItemPlan
from napari_vipp.core.file_sources import load_frozen_file_source_snapshot
from napari_vipp.core.io.errors import ImageSourceError, ImageSourceErrorCode
from napari_vipp.core.source_item_persistence import (
    SOURCE_ITEM_PARAMETER,
    params_with_source_item,
)
from napari_vipp.core.source_resolution import SourceReaderMismatchError
from napari_vipp.ui.file_sources import SourceFileLoadResult
from napari_vipp.ui.status import MessageSeverity


@pytest.mark.parametrize("background", [False, True], ids=["sync", "worker"])
def test_reader_mismatch_requires_explicit_refresh_to_load_tiff(
    qtbot, tmp_path, monkeypatch, background
):
    path = tmp_path / "field.ome.tif"
    pixels = np.arange(42, dtype=np.uint16).reshape(6, 7)
    tifffile.imwrite(path, pixels, metadata={"axes": "YX"})
    current = load_frozen_file_source_snapshot(path).source_item
    assert current is not None
    old = replace(current, reader=replace(current.reader, version="older-reader"))

    widget = VippWidget(_Viewer(), defer_initial_run=True)
    qtbot.addWidget(widget)
    node = widget.pipeline.nodes["input"]
    node.params.update(source_mode="file path", file_path=str(path), series_index=0)
    node.params = params_with_source_item(node.params, old)
    monkeypatch.setattr(widget, "_file_source_should_load_async", lambda n: background)

    widget.calculate_all_button.click()
    qtbot.waitUntil(
        lambda: widget.status_label.severity is MessageSeverity.ERROR, timeout=10_000
    )
    assert node.params[SOURCE_ITEM_PARAMETER] == old.to_dict()
    assert not widget._file_source_payload_cache
    assert widget.status_label.text() == (
        "The image reader has changed. Refresh file sources to reload with the "
        "current reader."
    )
    assert "older-reader" in widget.status_label.toolTip()
    assert current.reader.version in widget.status_label.toolTip()
    assert not widget.status_actions.refresh_button.isHidden()

    widget.status_actions.refresh_button.click()
    qtbot.waitUntil(
        lambda: np.array_equal(widget.pipeline.outputs.get("input"), pixels),
        timeout=10_000,
    )
    assert node.params[SOURCE_ITEM_PARAMETER] == current.to_dict()
    assert widget.status_actions.refresh_button.isHidden()


@pytest.mark.parametrize(
    ("code", "pending"),
    [
        (ImageSourceErrorCode.CORRUPT_SOURCE, False),
        (ImageSourceErrorCode.READER_MISMATCH, True),
    ],
)
def test_other_errors_and_superseded_worker_do_not_offer_rebinding(
    qtbot, monkeypatch, code, pending
):
    widget = VippWidget(_Viewer(), defer_initial_run=True)
    qtbot.addWidget(widget)
    widget._active_source_load_id = 17
    widget._source_load_pending = pending
    monkeypatch.setattr(widget, "_queue_source_work_continuation", lambda: None)
    error = ImageSourceError(code, "Source could not be loaded.")
    result = SourceFileLoadResult(
        17, {}, error=error.display_text, node_id="input", source_error=error
    )
    widget._on_source_file_load_finished(result)
    assert widget.status_label.severity is MessageSeverity.ERROR
    assert widget.status_actions.refresh_button.isHidden()


def test_rolled_back_batch_preview_does_not_offer_refresh_of_another_item(
    qtbot, tmp_path
):
    widget = VippWidget(_Viewer(), defer_initial_run=True)
    qtbot.addWidget(widget)
    items = tuple(
        BatchItemPlan(i, path.stem, path, {"input": path}, ())
        for i, path in enumerate((tmp_path / "first.tif", tmp_path / "next.tif"))
    )
    widget._interactive_collection_batch_items = items
    widget._interactive_collection_batch_index = 0
    widget._interactive_collection_batch_requested_index = 1
    widget._interactive_collection_source_paths = items[1].source_paths.copy()
    error = SourceReaderMismatchError("The next sample's reader has changed.")
    widget._show_interactive_collection_batch_preview_error(
        1, str(error), source_error=error
    )
    assert widget._interactive_collection_source_paths == items[0].source_paths
    assert "batch" in widget.status_label.text().lower()
    assert widget.status_actions.refresh_button.isHidden()


def test_old_warning_cannot_refresh_a_new_tab(qtbot):
    widget = VippWidget(_Viewer(), defer_initial_run=True)
    qtbot.addWidget(widget)
    error = SourceReaderMismatchError("Reader changed in the original workflow.")
    widget._set_source_error_status(str(error), error)
    stale_action = widget.status_label.refresh_sources
    assert stale_action is not None

    widget._new_workflow()
    assert widget.status_label.refresh_sources is None
    assert widget.status_actions.refresh_button.isHidden()
    with qtbot.assertNotEmitted(widget.refresh_button.clicked):
        stale_action()
