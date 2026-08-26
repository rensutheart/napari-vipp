from __future__ import annotations

import threading

import numpy as np
import pytest
from qtpy.QtCore import QThreadPool

from napari_vipp.core.io.errors import ImageSourceErrorCode
from napari_vipp.core.source_inspection import (
    SourceInspectionPhase,
    SourceInspectionProgress,
    SourceInspectionProgressUnit,
)
from napari_vipp.core.source_resolution import SourceItemResolutionError
from napari_vipp.ui import source_inspection as ui_source_inspection
from napari_vipp.ui.source_inspection import (
    SourceInspectionWorker,
    SourceInspectionWorkerSpec,
)


def _spec(path, generation: int = 11) -> SourceInspectionWorkerSpec:
    return SourceInspectionWorkerSpec(
        generation=generation,
        path=str(path),
        node_id="source-node",
        series_index_hint=0,
        item_key="volume",
    )


def test_inspection_worker_runs_off_gui_thread_with_typed_progress(
    tmp_path,
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "worker.npz"
    np.savez_compressed(path, volume=np.zeros((2, 5, 6), dtype=np.uint16))
    original = ui_source_inspection.inspect_local_source_item
    worker_threads = []
    gui_thread = threading.get_ident()

    def tracked(*args, **kwargs):
        worker_threads.append(threading.get_ident())
        return original(*args, **kwargs)

    monkeypatch.setattr(
        ui_source_inspection,
        "inspect_local_source_item",
        tracked,
    )
    worker = SourceInspectionWorker(
        _spec(path),
        current_generation=lambda: 11,
    )
    progress = []
    outcomes = []
    worker.signals.progress.connect(progress.append)
    worker.signals.finished.connect(outcomes.append)

    QThreadPool.globalInstance().start(worker)
    qtbot.waitUntil(lambda: len(outcomes) == 1)

    assert worker_threads[0] != gui_thread
    assert len(outcomes) == 1
    assert outcomes[0].succeeded
    assert outcomes[0].resolved.source_item.selector.key == "volume"
    assert all(update.generation == 11 for update in progress)
    assert all(update.node_id == "source-node" for update in progress)
    assert {update.phase for update in progress} == {
        SourceInspectionPhase.IDENTITY,
        SourceInspectionPhase.HEADER,
        SourceInspectionPhase.NORMALIZE,
        SourceInspectionPhase.VERIFY,
    }
    assert any(update.determinate for update in progress)


def test_inspection_worker_cancel_before_start_emits_once(
    tmp_path,
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        ui_source_inspection,
        "inspect_local_source_item",
        lambda *_args, **_kwargs: calls.append(True),
    )
    worker = SourceInspectionWorker(_spec(tmp_path / "cancelled.npz"))
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


def test_inspection_worker_rejects_stale_generation_before_inspection(
    tmp_path,
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        ui_source_inspection,
        "inspect_local_source_item",
        lambda *_args, **_kwargs: calls.append(True),
    )
    worker = SourceInspectionWorker(
        _spec(tmp_path / "stale.npz"),
        current_generation=lambda: 12,
    )
    outcomes = []
    worker.signals.finished.connect(outcomes.append)

    worker.run()

    assert calls == []
    assert len(outcomes) == 1
    assert outcomes[0].cancelled
    assert outcomes[0].stale
    assert outcomes[0].source_error.code is ImageSourceErrorCode.CANCELLED


def test_inspection_worker_stops_when_progress_is_superseded(
    tmp_path,
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active_generation = [11]

    def inspect(*_args, progress_callback, **_kwargs):
        progress_callback(
            SourceInspectionProgress(
                phase=SourceInspectionPhase.HEADER,
                current=0,
                total=1,
                unit=SourceInspectionProgressUnit.STEPS,
                message="Opening metadata.",
            )
        )
        pytest.fail("superseded inspection continued after progress publication")

    monkeypatch.setattr(
        ui_source_inspection,
        "inspect_local_source_item",
        inspect,
    )
    worker = SourceInspectionWorker(
        _spec(tmp_path / "superseded.npz"),
        current_generation=lambda: active_generation[0],
    )
    outcomes = []

    def supersede(_update):
        active_generation[0] = 12

    worker.signals.progress.connect(supersede)
    worker.signals.finished.connect(outcomes.append)

    worker.run()

    assert len(outcomes) == 1
    assert outcomes[0].cancelled
    assert outcomes[0].stale


def test_inspection_worker_preserves_structured_contract_error(
    tmp_path,
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ui_source_inspection,
        "inspect_local_source_item",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            SourceItemResolutionError("Saved item is missing.")
        ),
    )
    worker = SourceInspectionWorker(_spec(tmp_path / "missing.npz"))
    outcomes = []
    worker.signals.finished.connect(outcomes.append)

    worker.run()

    assert len(outcomes) == 1
    assert not outcomes[0].cancelled
    assert outcomes[0].source_error.code is ImageSourceErrorCode.CONTRACT_MISMATCH
    assert outcomes[0].source_error.item == "volume"
    assert "Saved item is missing" in outcomes[0].error
