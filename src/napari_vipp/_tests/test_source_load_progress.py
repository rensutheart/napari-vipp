from __future__ import annotations

from html import escape
from types import SimpleNamespace

import pytest
from qtpy.QtCore import QPoint, Qt
from qtpy.QtGui import QFont
from qtpy.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

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
    materialization = (
        r"Source materialization 2/3: Decoding C:\images\reverification copy.czi"
    )
    window_materialization = (
        r"Source window materialization 2/3: Decoding C:\images\hash preview.czi"
    )
    reverification = (
        r"Source reverification 3/3: Checking C:\images\decoding download.czi"
    )

    def load(*_args, progress_callback, cancel_callback, **_kwargs):
        assert not cancel_callback()
        progress_callback(0, 10, "Source validation 1/3: hashing")
        progress_callback(10, 10, "Source validation 1/3: hashed")
        progress_callback(0, 64, materialization)
        progress_callback(32, 64, window_materialization)
        progress_callback(64, 64, "Source materialization 2/3: decoded")
        progress_callback(8, 8, reverification)
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
    by_message = {update.message: update.phase for update in progress}
    assert by_message[materialization] is SourceLoadPhase.DECODE
    assert by_message[window_materialization] is SourceLoadPhase.DECODE
    assert by_message[reverification] is SourceLoadPhase.VERIFY
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
    assert control.phase_label.text() == "Inspecting source"
    assert not control.update_progress(
        SourceLoadProgress(7, SourceLoadPhase.DOWNLOAD, total=10, current=5)
    )
    assert control.phase_label.text() == "Inspecting source"
    current = SourceLoadProgress(
        8,
        SourceLoadPhase.DECODE,
        current=512,
        total=1024,
        unit=SourceLoadProgressUnit.BYTES,
        message="Decoding plane 1.",
    )
    assert control.update_progress(current)
    assert control.phase_label.text() == "Decoding image data"
    assert control.label.text() == "Decoding plane 1."
    assert control.progress_bar.format() == "512 B / 1.0 KiB"

    qtbot.mouseClick(control.cancel_button, Qt.LeftButton)
    control._request_cancel()

    assert cancelled == [8]
    assert not control.cancel_button.isEnabled()
    assert not control.update_progress(current)
    assert not control.finish(7, cancelled=True)
    assert control.finish(8, cancelled=True)
    assert "cancelled" in (
        control.phase_label.text() + " " + control.label.text()
    ).casefold()
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
    assert "presentation preview" in control.source_load_status.phase_label.text()
    assert control.source_load_status.label.text() == "Preview level 2."
    qtbot.mouseClick(control.source_load_status.cancel_button, Qt.LeftButton)
    assert cancelled == [11]
    assert control.finish_source_load(11, cancelled=True)


@pytest.mark.parametrize("name", ("sample.czi", "sample & raw.czi"))
def test_source_progress_compacts_known_validation_and_keeps_raw_diagnostic(
    qtbot, name
) -> None:
    control = SourceLoadStatusControl()
    qtbot.addWidget(control)
    assert control.begin(1)
    raw = rf"Source validation 1/3: Hashing source file: C:\some\very\long\{name}"

    assert control.update_progress(
        SourceLoadProgress(1, SourceLoadPhase.INSPECT, message=raw)
    )

    assert control.phase_label.text() == "Inspecting source"
    assert control.label.text() == f"Step 1 of 3 · Checking file contents · {name}"
    assert escape(raw) in control.label.toolTip()
    assert control.phase_label.textFormat() == Qt.PlainText
    assert control.label.textFormat() == Qt.PlainText


def test_source_progress_preview_start_and_unrecognized_detail(qtbot) -> None:
    control = SourceLoadStatusControl()
    qtbot.addWidget(control)
    assert control.begin(1, phase=SourceLoadPhase.PREVIEW)
    assert "presentation preview" in control.phase_label.text()

    detail = "Reader detail: <b>literal</b> & status"
    assert control.update_progress(
        SourceLoadProgress(
            1,
            SourceLoadPhase.INSPECT,
            item_index=2,
            item_total=4,
            message=detail,
        )
    )

    assert control.phase_label.text() == "Inspecting source · Item 2 of 4"
    assert control.label.text() == detail
    assert control.label.textFormat() == Qt.PlainText


def _bounds_within(control, widget):
    return widget.rect().translated(widget.mapTo(control, QPoint()))


def test_source_progress_fits_narrow_width_and_large_text(qtbot) -> None:
    host = QWidget()
    qtbot.addWidget(host)
    host.setFont(QFont("Arial", 15))
    layout = QVBoxLayout(host)
    layout.setContentsMargins(0, 0, 0, 0)
    control = SourceLoadStatusControl()
    layout.addWidget(control)
    host.setFixedWidth(300)
    host.show()

    def bounds(widget):
        return _bounds_within(control, widget)

    assert control.begin(1, title="Loading image source — Original fluorescence stack")
    assert control.amount_label.isHidden()
    assert control.progress_bar.maximum() == 0
    qtbot.waitUntil(lambda: control.progress_bar.width() >= 280)
    full_bar_width = control.progress_bar.width()
    assert control.update_progress(
        SourceLoadProgress(
            1,
            SourceLoadPhase.VERIFY,
            current=775 * 1024**2,
            total=2 * 1024**3,
            unit=SourceLoadProgressUnit.BYTES,
            message="Checking the source revision before reading any pixels.",
        )
    )
    labels = (control.heading, control.phase_label, control.label, control.amount_label)
    qtbot.waitUntil(
        lambda: (
            all(
                label.height() >= label.heightForWidth(label.width())
                for label in labels
            )
            and bounds(control.progress_bar).bottom() < control.height()
        )
    )
    for label in labels:
        assert control.rect().contains(bounds(label))
    assert bounds(control.heading).right() < bounds(control.cancel_button).left()
    assert bounds(control.heading).bottom() < bounds(control.phase_label).top()
    assert bounds(control.phase_label).bottom() < bounds(control.label).top()
    assert control.phase_label.font().weight() > control.label.font().weight()
    bar_rect = bounds(control.progress_bar)
    amount_rect = bounds(control.amount_label)
    assert control.rect().contains(bar_rect)
    assert bounds(control.label).bottom() < min(bar_rect.top(), amount_rect.top())
    assert bar_rect.right() < amount_rect.left()
    assert abs(bar_rect.center().y() - amount_rect.center().y()) <= 1
    assert 70 <= bar_rect.width() < full_bar_width
    amount_text = control.amount_label.text()
    assert amount_text == "775.0 MiB / 2.0 GiB"
    assert amount_rect.width() >= (
        control.amount_label.fontMetrics().horizontalAdvance(amount_text)
    )
    assert not control.progress_bar.isTextVisible()

    assert control.update_progress(
        SourceLoadProgress(1, SourceLoadPhase.NORMALIZE, message="Resolving axes.")
    )
    assert control.amount_label.isHidden()
    assert control.progress_bar.maximum() == 0
    qtbot.waitUntil(lambda: control.progress_bar.width() >= full_bar_width)


def test_source_progress_amount_stacks_only_until_there_is_room(qtbot) -> None:
    host = QWidget()
    qtbot.addWidget(host)
    host.setFont(QFont("Arial", 18))
    layout = QVBoxLayout(host)
    layout.setContentsMargins(0, 0, 0, 0)
    control = SourceLoadStatusControl()
    layout.addWidget(control)
    host.setFixedWidth(245)
    host.show()

    assert control.begin(1)
    assert control.update_progress(
        SourceLoadProgress(
            1,
            SourceLoadPhase.VERIFY,
            current=775 * 1024**2,
            total=2 * 1024**3,
            unit=SourceLoadProgressUnit.BYTES,
        )
    )

    def bar_bounds():
        return _bounds_within(control, control.progress_bar)

    def amount_bounds():
        return _bounds_within(control, control.amount_label)

    qtbot.waitUntil(
        lambda: bar_bounds().bottom() < amount_bounds().top()
        and control.rect().contains(amount_bounds())
    )
    assert control.rect().contains(bar_bounds())
    assert control.rect().contains(amount_bounds())
    assert bar_bounds().width() >= 220
    assert control.amount_label.text() == "775.0 MiB / 2.0 GiB"
    assert control.amount_label.minimumWidth() == 0

    host.setFixedWidth(420)
    qtbot.waitUntil(
        lambda: bar_bounds().right() < amount_bounds().left()
        and control.rect().contains(amount_bounds())
    )
    assert control.rect().contains(bar_bounds())
    assert control.rect().contains(amount_bounds())
    assert abs(bar_bounds().center().y() - amount_bounds().center().y()) <= 1
    assert control.amount_label.text() == "775.0 MiB / 2.0 GiB"
    assert control.amount_label.minimumWidth() >= (
        control.amount_label.fontMetrics().horizontalAdvance(control.amount_label.text())
    )


def test_source_progress_reclaims_graph_space_after_wrapping(qtbot) -> None:
    host = QWidget()
    qtbot.addWidget(host)
    host.setFont(QFont("Arial", 15))
    layout = QVBoxLayout(host)
    layout.setContentsMargins(0, 0, 0, 0)
    control = SourceLoadStatusControl()
    graph = QWidget()
    graph.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
    layout.addWidget(control)
    layout.addWidget(graph, 1)
    host.setFixedSize(296, 420)
    host.show()

    title = (
        "Loading image source — Long name for a fluorescence stack from the "
        "representative microscope acquisition"
    )
    long_detail = (
        "Checking the source revision before reading any pixels. "
        "The previously verified workflow results are still available."
    )
    short_detail = "Checking file contents."

    def update_detail(message: str) -> None:
        assert control.update_progress(
            SourceLoadProgress(
                1,
                SourceLoadPhase.INSPECT,
                current=775 * 1024**2,
                total=2 * 1024**3,
                unit=SourceLoadProgressUnit.BYTES,
                message=message,
            )
        )

    def wrapped() -> bool:
        return all(
            label.height() >= label.heightForWidth(label.width())
            > label.fontMetrics().lineSpacing()
            for label in (control.heading, control.label)
        ) and all(
            control.rect().contains(_bounds_within(control, widget))
            for widget in (control.heading, control.label, control.progress_bar)
        )

    assert control.begin(1, title=title)
    update_detail(long_detail)
    qtbot.waitUntil(wrapped)
    narrow_graph_height = graph.height()

    for _ in range(2):
        host.setFixedWidth(2560)
        update_detail(short_detail)
        qtbot.waitUntil(lambda: control.heading.width() > 2000)
        for label in (control.heading, control.phase_label, control.label):
            assert label.minimumHeight() <= label.fontMetrics().lineSpacing() + 2
        qtbot.waitUntil(lambda: graph.height() > narrow_graph_height + 30)

        host.setFixedWidth(296)
        update_detail(long_detail)
        qtbot.waitUntil(lambda: control.heading.width() < 296 and wrapped())

    # A shorter message should also release obsolete height without a resize.
    before_short_detail = graph.height()
    update_detail(short_detail)
    qtbot.waitUntil(lambda: graph.height() > before_short_detail)
    detail = control.label
    assert detail.minimumHeight() <= detail.fontMetrics().lineSpacing() + 2
