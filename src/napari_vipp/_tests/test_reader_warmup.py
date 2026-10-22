from __future__ import annotations

import importlib
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from qtpy.QtCore import QTimer

from napari_vipp.core.reader_warmup import ReaderLibraryWarmup
from napari_vipp.ui import reader_warmup as ui_warmup


def test_importing_warmup_does_not_import_any_scientific_or_qt_library():
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(
            None,
            (str(Path(__file__).resolve().parents[2]), environment.get("PYTHONPATH")),
        )
    )
    code = """
import sys
import napari_vipp.core.reader_warmup
forbidden = {'numpy', 'ome_types', 'ome_zarr', 'qtpy', 'napari', 'cupy'}
assert not forbidden & sys.modules.keys()
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_warmup_is_once_only_off_thread_and_cancel_skips_remaining_imports():
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def load(name):
        calls.append((name, threading.get_ident()))
        entered.set()
        assert release.wait(5)

    warmup = ReaderLibraryWarmup(modules=("first", "second"), loader=load)
    try:
        assert warmup.start()
        assert entered.wait(5)
        assert not warmup.start()
        assert calls == [("first", calls[0][1])]
        assert calls[0][1] != threading.get_ident()
        warmup.cancel()
    finally:
        release.set()
        assert warmup.wait(5)
    assert [result.module for result in warmup.results] == ["first"]
    assert not warmup.start()


def test_failed_warmup_is_recorded_and_normal_import_can_retry(
    tmp_path, monkeypatch, caplog
):
    name = "vipp_warmup_retry_fixture"
    (tmp_path / f"{name}.py").write_text("raise RuntimeError('reader unavailable')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    warmup = ReaderLibraryWarmup(modules=(name, "math"))
    with caplog.at_level(logging.WARNING):
        assert warmup.start()
        assert warmup.wait(5)
    assert [result.module for result in warmup.results] == [name, "math"]
    assert "reader unavailable" in warmup.results[0].error
    assert not warmup.results[1].error
    assert name not in sys.modules
    assert "Reader warm-up failed" in caplog.text
    (tmp_path / f"{name}.py").write_text("value = 42\n")
    importlib.invalidate_caches()
    try:
        assert importlib.import_module(name).value == 42
    finally:
        sys.modules.pop(name, None)


def test_file_request_during_warmup_shares_one_normal_module_import(
    tmp_path, monkeypatch
):
    name = "vipp_shared_reader_fixture"
    gate_name = "vipp_reader_warmup_gate"
    gate = SimpleNamespace(
        entered=threading.Event(), release=threading.Event(), calls=[]
    )
    monkeypatch.setitem(sys.modules, gate_name, gate)
    monkeypatch.syspath_prepend(str(tmp_path))
    (tmp_path / f"{name}.py").write_text(
        f"from {gate_name} import entered, release, calls\n"
        "calls.append(True)\nentered.set()\nassert release.wait(5)\nvalue = object()\n"
    )
    warmup = ReaderLibraryWarmup(modules=(name,))
    requested = []
    request_started = threading.Event()

    def file_request():
        request_started.set()
        requested.append(importlib.import_module(name))

    request = threading.Thread(target=file_request)
    try:
        assert warmup.start()
        assert gate.entered.wait(5)
        request.start()
        assert request_started.wait(5)
    finally:
        gate.release.set()
        assert warmup.wait(5)
        if request.ident is not None:
            request.join(5)
    try:
        assert not request.is_alive()
        assert gate.calls == [True]
        assert requested == [sys.modules[name]]
        assert importlib.import_module(name) is requested[0]
        assert not warmup.pending(name)
    finally:
        sys.modules.pop(name, None)


@pytest.fixture
def scheduler(qapp, monkeypatch):
    existing = getattr(qapp, ui_warmup._APPLICATION_ATTRIBUTE, None)
    if existing is not None:
        pytest.fail("A test leaked its reader warm-up scheduler")
    monkeypatch.setattr(ui_warmup, "_WARMUP_DELAY_MS", 20)
    created = []
    yield created
    for controller in created:
        controller.cancel()
        controller.deleteLater()
    if hasattr(qapp, ui_warmup._APPLICATION_ATTRIBUTE):
        delattr(qapp, ui_warmup._APPLICATION_ATTRIBUTE)


def test_schedule_returns_before_import_and_keeps_qt_heartbeat_running(
    qtbot, qapp, monkeypatch, scheduler
):
    entered = threading.Event()
    release = threading.Event()
    thread_ids = []

    def load(_name):
        thread_ids.append(threading.get_ident())
        entered.set()
        assert release.wait(5)

    warmup = ReaderLibraryWarmup(modules=("ome_types",), loader=load)
    monkeypatch.setattr(ui_warmup, "ReaderLibraryWarmup", lambda: warmup)
    ui_warmup.schedule_reader_warmup()
    controller = getattr(qapp, ui_warmup._APPLICATION_ATTRIBUTE)
    scheduler.append(controller)
    assert not entered.is_set()
    assert ui_warmup.reader_warmup_pending("ome_types")
    ui_warmup.schedule_reader_warmup()
    assert getattr(qapp, ui_warmup._APPLICATION_ATTRIBUTE) is controller
    ticks = []
    heartbeat = QTimer()
    heartbeat.timeout.connect(lambda: ticks.append(True))
    heartbeat.start(5)
    try:
        qtbot.waitUntil(entered.is_set, timeout=1000)
        initial_count = len(ticks)
        qtbot.waitUntil(lambda: len(ticks) >= initial_count + 3, timeout=1000)
        assert not warmup.wait(0)
        assert thread_ids[0] != threading.get_ident()
    finally:
        heartbeat.stop()
        release.set()
        assert warmup.wait(5)
    assert not ui_warmup.reader_warmup_pending("ome_types")


def test_quitting_before_delay_cancels_without_importing(qapp, monkeypatch, scheduler):
    calls = []
    warmup = ReaderLibraryWarmup(modules=("ome_types",), loader=calls.append)
    monkeypatch.setattr(ui_warmup, "ReaderLibraryWarmup", lambda: warmup)
    ui_warmup.schedule_reader_warmup()
    controller = getattr(qapp, ui_warmup._APPLICATION_ATTRIBUTE)
    scheduler.append(controller)
    controller.cancel()  # The same slot connected to QApplication.aboutToQuit.
    controller._start()
    assert not calls
    assert not controller._timer.isActive()
    assert not warmup.start()
    assert not ui_warmup.reader_warmup_pending("ome_types")


def test_small_tiff_uses_source_worker_while_ome_xml_warmup_is_pending(
    tmp_path, monkeypatch
):
    from napari_vipp._widget import VippWidget

    path = tmp_path / "small.ome.tif"
    path.write_bytes(b"tiny source")
    widget = SimpleNamespace(_file_source_path_for_node=lambda _node: path)
    monkeypatch.setattr(ui_warmup, "reader_warmup_pending", lambda _module: True)
    assert VippWidget._file_source_should_load_async(widget, object())
    monkeypatch.setattr(ui_warmup, "reader_warmup_pending", lambda _module: False)
    assert not VippWidget._file_source_should_load_async(widget, object())
