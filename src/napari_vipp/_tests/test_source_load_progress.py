from __future__ import annotations

from types import SimpleNamespace

import pytest
from qtpy.QtCore import Qt

from napari_vipp.core.io.errors import ImageSourceErrorCode
from napari_vipp.core.progress import OperationCancelled
from napari_vipp.ui import file_sources as ui_file_sources
from napari_vipp.ui.controls import ImageSourceControl, SourceLoadStatusControl
from napari_vipp.ui.file_sources import (
    SourceFileLoadSpec,
    SourceFileLoadWorker,
    SourceLoadPhase,
    SourceLoadProgress,
    SourceLoadProgressUnit,
)


def _spec() -> SourceFileLoadSpec:
    return SourceFileLoadSpec(
        node_id="input",
        path="sample.ome.tif",
        series_index=0,
        cache_key=("sample.ome.tif", 0),
    )


def test_source_worker_emits_typed_ordered_phase_progress(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = SimpleNamespace()

    def load(*_args, progress_callback, cancel_callback, **_kwargs):
        assert not cancel_callback()
        progress_callback(0, 10, "Source validation 1/3: hashing")
        progress_callback(10, 10, "Source validation 1/3: hashed")
        progress_callback(0, 64, "Source materialization 2/3: decoding")
        progress_callback(64, 64, "Source materialization 2/3: decoded")
        progress_callback(8, 8, "Source reverification 3/3: verified")
        return SimpleNamespace(identity=identity)

    monkeypatch.setattr(ui_file_sources, "load_frozen_file_source_snapshot", load)
    worker = SourceFileLoadWorker(12, (_spec(),), reader=lambda *_a, **_k: None)
    progress = []
    outcomes = []
    worker.signals.progress.connect(progress.append)
    worker.signals.finished.connect(outcomes.append)

    worker.run()

    phases = [update.phase for update in progress]
    assert phases[0] is SourceLoadPhase.INSPECT
    assert SourceLoadPhase.READ in phases
    assert SourceLoadPhase.DECODE in phases
    assert SourceLoadPhase.VERIFY in phases
    assert phases[-1] is SourceLoadPhase.NORMALIZE
    assert phases.index(SourceLoadPhase.READ) < phases.index(SourceLoadPhase.DECODE)
    assert all(update.run_id == 12 for update in progress)
    assert all(update.node_id == "input" for update in progress)
    decoded = next(
        update
        for update in progress
        if update.phase is SourceLoadPhase.DECODE and update.total == 64
    )
    assert decoded.unit is SourceLoadProgressUnit.BYTES
    assert len(outcomes) == 1
    assert not outcomes[0].cancelled
    assert outcomes[0].last_phase is SourceLoadPhase.NORMALIZE


def test_source_worker_cancel_before_start_is_one_terminal_result(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        ui_file_sources,
        "load_frozen_file_source_snapshot",
        lambda *_args, **_kwargs: calls.append(True),
    )
    worker = SourceFileLoadWorker(3, (_spec(),), reader=lambda *_a, **_k: None)
    outcomes = []
    worker.signals.finished.connect(outcomes.append)

    worker.cancel()
    worker.run()
    worker.run()

    assert calls == []
    assert len(outcomes) == 1
    assert outcomes[0].cancelled
    assert outcomes[0].source_error.code is ImageSourceErrorCode.CANCELLED
    assert outcomes[0].source_error.path == "sample.ome.tif"


def test_source_worker_cancel_during_decode_stops_at_core_checkpoint(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def load(*_args, progress_callback, cancel_callback, **_kwargs):
        progress_callback(1, 1, "Source validation 1/3: complete")
        progress_callback(1, 8, "Source materialization 2/3: decoding")
        if cancel_callback():
            raise OperationCancelled("Cancelled during materialization.")
        raise AssertionError("the progress receiver should have cancelled")

    monkeypatch.setattr(ui_file_sources, "load_frozen_file_source_snapshot", load)
    worker = SourceFileLoadWorker(4, (_spec(),), reader=lambda *_a, **_k: None)
    outcomes = []
    observed = []

    def receive(update: SourceLoadProgress) -> None:
        observed.append(update)
        if update.phase is SourceLoadPhase.DECODE:
            worker.cancel()

    worker.signals.progress.connect(receive)
    worker.signals.finished.connect(outcomes.append)
    worker.run()

    assert worker.cancellation_requested
    assert len(outcomes) == 1
    assert outcomes[0].cancelled
    assert outcomes[0].last_phase is SourceLoadPhase.DECODE
    assert [update.phase for update in observed][-1] is SourceLoadPhase.DECODE


def test_source_load_status_rejects_stale_generations_and_cancels_once(qtbot) -> None:
    control = SourceLoadStatusControl()
    qtbot.addWidget(control)
    cancelled = []
    control.cancelRequested.connect(cancelled.append)

    assert control.begin(8)
    assert not control.update_progress(
        SourceLoadProgress(7, SourceLoadPhase.DOWNLOAD, total=10, current=5)
    )
    current = SourceLoadProgress(
        8,
        SourceLoadPhase.DECODE,
        current=512,
        total=1024,
        unit=SourceLoadProgressUnit.BYTES,
        message="Decoding plane 1.",
    )
    assert control.update_progress(current)
    assert "Decoding image data" in control.label.text()
    assert control.progress_bar.format() == "512 B / 1.0 KiB"

    qtbot.mouseClick(control.cancel_button, Qt.LeftButton)
    control._request_cancel()

    assert cancelled == [8]
    assert not control.cancel_button.isEnabled()
    assert not control.update_progress(current)
    assert not control.finish(7, cancelled=True)
    assert control.finish(8, cancelled=True)
    assert "cancelled" in control.label.text().casefold()
    assert not control.begin(8)
    assert control.begin(9)


def test_image_source_control_exposes_progress_and_cancel_hook(qtbot) -> None:
    control = ImageSourceControl(
        {"source_mode": "file path", "file_path": "sample.zarr"},
        layer_names=[],
        sample_names=[],
    )
    qtbot.addWidget(control)
    cancelled = []
    control.sourceLoadCancelRequested.connect(cancelled.append)

    assert control.begin_source_load(11)
    assert control.update_source_load_progress(
        SourceLoadProgress(
            11,
            SourceLoadPhase.PREVIEW,
            current=1,
            total=2,
            unit=SourceLoadProgressUnit.STEPS,
            message="Preview level 2.",
        )
    )
    assert "presentation preview" in control.source_load_status.label.text()
    qtbot.mouseClick(control.source_load_status.cancel_button, Qt.LeftButton)
    assert cancelled == [11]
    assert control.finish_source_load(11, cancelled=True)
