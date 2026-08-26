from __future__ import annotations

import threading
from dataclasses import replace

import numpy as np
import pytest
from qtpy.QtCore import QThreadPool

from napari_vipp.core.io.errors import ImageSourceErrorCode
from napari_vipp.core.io.model import ImageSeriesInfo, SourceInspection
from napari_vipp.core.metadata import AxisMetadata, image_state_from_array
from napari_vipp.core.source_identity import capture_local_source_identity
from napari_vipp.core.source_preview import (
    SourcePreviewReadMetrics,
    SourcePreviewRequest,
    SourcePreviewResult,
)
from napari_vipp.ui import source_preview as ui_source_preview
from napari_vipp.ui.source_preview import (
    SourcePreviewSeriesSelection,
    SourcePreviewWorker,
    SourcePreviewWorkerSpec,
)


def _inspection(*, selected_index: int = 2) -> SourceInspection:
    items = [
        ImageSeriesInfo(0, ".", "image", (8, 8), "uint8", "YX"),
        ImageSeriesInfo(1, "labels/other", "other", (8, 8), "uint8", "YX"),
        ImageSeriesInfo(
            selected_index,
            "labels/cells",
            "cells",
            (8, 8),
            "uint8",
            "YX",
            kind="labels",
        ),
    ]
    return SourceInspection("sample.ome.zarr", "ome-zarr-0.4", tuple(items))


def _preview(generation: int) -> SourcePreviewResult:
    data = np.zeros((8, 8), dtype=np.uint8)
    state = image_state_from_array(
        data,
        axes=(
            AxisMetadata("y", "space", source_axis=0),
            AxisMetadata("x", "space", source_axis=1),
        ),
    )
    assert state is not None
    return SourcePreviewResult(
        data=data,
        image_state=state,
        preview_level=1,
        level_count=2,
        message="Preview level 1 - analysis remains full resolution",
        metrics=SourcePreviewReadMetrics(requested_decoded_bytes=data.nbytes),
        generation=generation,
    )


def _spec(generation: int = 7) -> SourcePreviewWorkerSpec:
    return SourcePreviewWorkerSpec(
        generation=generation,
        path="sample.ome.zarr",
        selection=SourcePreviewSeriesSelection(
            item_key="labels/cells",
            series_index_hint=0,
        ),
        request=SourcePreviewRequest(display_shape_yx=(64, 64)),
        node_id="source-1",
    )


def test_preview_worker_runs_off_gui_thread_and_uses_stable_item_key(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspections = []
    read_threads = []
    read_indices = []
    gui_thread = threading.get_ident()

    def inspect(_path):
        inspections.append(True)
        return _inspection()

    def read(_path, series_index, *, request, control):
        read_threads.append(threading.get_ident())
        read_indices.append(series_index)
        assert request == _spec().request
        control.report(0, 2, "Selecting preview level")
        control.report(2, 2, "Preview ready")
        return _preview(control.generation)

    monkeypatch.setattr(ui_source_preview, "inspect_ome_zarr", inspect)
    monkeypatch.setattr(
        ui_source_preview,
        "read_ome_zarr_presentation_preview",
        read,
    )
    worker = SourcePreviewWorker(_spec(), current_generation=lambda: 7)
    progress = []
    outcomes = []
    worker.signals.progress.connect(progress.append)
    worker.signals.finished.connect(outcomes.append)

    QThreadPool.globalInstance().start(worker)
    qtbot.waitUntil(lambda: len(outcomes) == 1)

    assert inspections == [True, True]
    assert read_indices == [2]
    assert read_threads[0] != gui_thread
    assert [update.generation for update in progress] == [7, 7]
    assert outcomes[0].succeeded
    assert outcomes[0].preview is not None
    assert outcomes[0].preview.presentation_only
    assert outcomes[0].item_key == "labels/cells"


def test_preview_worker_cancel_before_start_is_one_typed_terminal(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        ui_source_preview,
        "inspect_ome_zarr",
        lambda _path: calls.append(True),
    )
    worker = SourcePreviewWorker(_spec())
    outcomes = []
    worker.signals.finished.connect(outcomes.append)

    worker.cancel()
    worker.run()
    worker.run()

    assert calls == []
    assert len(outcomes) == 1
    assert outcomes[0].cancelled
    assert not outcomes[0].stale
    assert outcomes[0].source_error.code is ImageSourceErrorCode.CANCELLED


def test_preview_worker_cooperatively_cancels_during_read(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ui_source_preview,
        "inspect_ome_zarr",
        lambda _path: _inspection(),
    )

    def read(_path, _series_index, *, request, control):
        del request
        control.report(1, 4, "Reading requested preview chunks")
        pytest.fail("cancelled progress should stop the preview reader")

    monkeypatch.setattr(
        ui_source_preview,
        "read_ome_zarr_presentation_preview",
        read,
    )
    worker = SourcePreviewWorker(_spec())
    progress = []
    outcomes = []

    def receive(update):
        progress.append(update)
        worker.cancel()

    worker.signals.progress.connect(receive)
    worker.signals.finished.connect(outcomes.append)

    worker.run()

    assert len(progress) == 1
    assert len(outcomes) == 1
    assert outcomes[0].cancelled
    assert not outcomes[0].stale
    assert outcomes[0].source_error.code is ImageSourceErrorCode.CANCELLED


def test_preview_worker_rejects_stale_generation_before_reader_call(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        ui_source_preview,
        "inspect_ome_zarr",
        lambda _path: calls.append(True),
    )
    worker = SourcePreviewWorker(_spec(), current_generation=lambda: 8)
    outcomes = []
    worker.signals.finished.connect(outcomes.append)

    worker.run()

    assert calls == []
    assert len(outcomes) == 1
    assert outcomes[0].cancelled
    assert outcomes[0].stale
    assert outcomes[0].source_error.code is ImageSourceErrorCode.CANCELLED


def test_preview_worker_stops_when_new_generation_starts_during_read(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active_generation = [7]
    monkeypatch.setattr(
        ui_source_preview,
        "inspect_ome_zarr",
        lambda _path: _inspection(),
    )

    def read(_path, _series_index, *, request, control):
        del request
        control.report(1, 4, "Reading requested preview chunks")
        pytest.fail("stale progress should stop the preview reader")

    monkeypatch.setattr(
        ui_source_preview,
        "read_ome_zarr_presentation_preview",
        read,
    )
    worker = SourcePreviewWorker(
        _spec(),
        current_generation=lambda: active_generation[0],
    )
    outcomes = []

    def supersede(_update):
        active_generation[0] = 8

    worker.signals.progress.connect(supersede)
    worker.signals.finished.connect(outcomes.append)

    worker.run()

    assert len(outcomes) == 1
    assert outcomes[0].cancelled
    assert outcomes[0].stale
    assert outcomes[0].source_error.code is ImageSourceErrorCode.CANCELLED


def test_preview_worker_discards_result_if_item_order_changes_mid_read(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspections = iter((_inspection(selected_index=2), _inspection(selected_index=1)))
    monkeypatch.setattr(
        ui_source_preview,
        "inspect_ome_zarr",
        lambda _path: next(inspections),
    )
    monkeypatch.setattr(
        ui_source_preview,
        "read_ome_zarr_presentation_preview",
        lambda _path, _index, *, request, control: _preview(control.generation),
    )
    worker = SourcePreviewWorker(_spec())
    outcomes = []
    worker.signals.finished.connect(outcomes.append)

    worker.run()

    assert len(outcomes) == 1
    assert not outcomes[0].succeeded
    assert outcomes[0].source_error.code is ImageSourceErrorCode.CONTRACT_MISMATCH
    assert "discarded" in outcomes[0].error


def test_preview_worker_missing_stable_key_fails_closed(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ui_source_preview,
        "inspect_ome_zarr",
        lambda _path: _inspection(),
    )
    worker = SourcePreviewWorker(
        SourcePreviewWorkerSpec(
            generation=3,
            path="sample.ome.zarr",
            selection=SourcePreviewSeriesSelection("labels/missing"),
            request=SourcePreviewRequest(),
        )
    )
    outcomes = []
    worker.signals.finished.connect(outcomes.append)

    worker.run()

    assert len(outcomes) == 1
    assert outcomes[0].source_error.code is ImageSourceErrorCode.CONTRACT_MISMATCH
    assert outcomes[0].source_error.item == "labels/missing"


def test_preview_worker_rejects_source_revision_changed_during_read(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "sample.ome.zarr"
    path.mkdir()
    member = path / "chunk"
    member.write_bytes(b"before")
    identity = capture_local_source_identity(path)
    monkeypatch.setattr(
        ui_source_preview,
        "inspect_ome_zarr",
        lambda _path: _inspection(),
    )

    def mutate_then_preview(_path, _index, *, request, control):
        del request
        member.write_bytes(b"after")
        return _preview(control.generation)

    monkeypatch.setattr(
        ui_source_preview,
        "read_ome_zarr_presentation_preview",
        mutate_then_preview,
    )
    worker = SourcePreviewWorker(
        replace(
            _spec(generation=9),
            path=str(path),
            expected_identity=identity,
        )
    )
    outcomes = []
    worker.signals.finished.connect(outcomes.append)

    worker.run()

    assert len(outcomes) == 1
    assert not outcomes[0].succeeded
    assert "changed" in outcomes[0].error.casefold()


@pytest.mark.parametrize(
    ("selection", "match"),
    [
        (lambda: SourcePreviewSeriesSelection(""), "stable item key"),
        (lambda: SourcePreviewSeriesSelection(".", -1), "non-negative"),
    ],
)
def test_preview_selection_rejects_unsafe_values(selection, match) -> None:
    with pytest.raises(ValueError, match=match):
        selection()
