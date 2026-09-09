"""Actual filter-pair counts, bounded background scheduling and Qt lifetime."""

from __future__ import annotations

import gc
import threading
import weakref

import numpy as np
import pytest
from qtpy.QtCore import QCoreApplication, QEvent, QObject, QThread

from napari_vipp.core.object_filter_counts import ObjectFilterCounts
from napari_vipp.ui import object_filter_diagnostics as module
from napari_vipp.ui.object_filter_diagnostics import (
    ObjectFilterDiagnosticKey,
    ObjectFilterDiagnostics,
)


def _pair():
    before = np.array([[1, 1, 0, 9], [0, 0, 0, 9]], dtype=np.int32)
    after = np.where(before == 9, 0, before)
    before.setflags(write=False)
    after.setflags(write=False)
    return before, after


@pytest.fixture
def diagnostics(qapp):
    controller = ObjectFilterDiagnostics()
    yield controller
    controller.close()
    assert controller.wait(5000)


def test_exact_counts_use_actual_read_only_pair(diagnostics, qtbot):
    before, after = _pair()
    with qtbot.waitSignal(diagnostics.ready) as emitted:
        diagnostics.request(before, after, 2)
    assert emitted.args[0].matches(before, after, 2)
    assert diagnostics.cached(before, after, 2) == ObjectFilterCounts(2, 1, 1, 1)
    assert diagnostics.error(before, after, 2) == ""
    np.testing.assert_array_equal(before, [[1, 1, 0, 9], [0, 0, 0, 9]])
    np.testing.assert_array_equal(after, [[1, 1, 0, 0], [0, 0, 0, 0]])
    assert not before.flags.writeable and not after.flags.writeable


def test_duplicate_requests_count_once_off_gui_thread(
    diagnostics, qtbot, monkeypatch
):
    pair = _pair()
    entered, release = threading.Event(), threading.Event()
    calls = []
    gui_thread = QThread.currentThread()

    def calculate(*args, **kwargs):
        calls.append((args, kwargs, QThread.currentThread()))
        entered.set()
        assert release.wait(3)
        return ObjectFilterCounts(2, 1, 1, 1)

    monkeypatch.setattr(module, "object_filter_counts", calculate)
    diagnostics.request(*pair, 2)
    try:
        qtbot.waitUntil(entered.is_set)
        assert diagnostics.cached(*pair, 2) is None
        for _ in range(5):
            diagnostics.request(*pair, 2)
        assert len(calls) == 1
    finally:
        release.set()
    qtbot.waitUntil(lambda: diagnostics.cached(*pair, 2) is not None)
    diagnostics.request(*pair, 2)
    assert len(calls) == 1
    assert calls[0][0][0] is pair[0] and calls[0][0][1] is pair[1]
    assert calls[0][2] != gui_thread


def test_cache_keys_both_arrays_rank_connectivity_and_semantic_axes(
    diagnostics, qtbot, monkeypatch
):
    pair = _pair()
    calls = []

    def calculate(*_args, **_kwargs):
        calls.append(1)
        return ObjectFilterCounts(2, 1, 1, 1)

    monkeypatch.setattr(module, "object_filter_counts", calculate)
    with qtbot.waitSignal(diagnostics.ready):
        diagnostics.request(*pair, 2)
    assert diagnostics.cached(pair[0].copy(), pair[1], 2) is None
    assert diagnostics.cached(pair[0], pair[1].copy(), 2) is None
    assert diagnostics.cached(*pair, 3) is None
    assert diagnostics.cached(*pair, 2, "Full connectivity") is None
    assert diagnostics.cached(*pair, 2, spatial_axes=(0, 1)) is None
    with qtbot.waitSignal(diagnostics.ready):
        diagnostics.request(*pair, 2, "Full connectivity")
    assert len(calls) == 2


def test_semantic_axes_are_moved_in_worker_and_cache_uses_original_arrays(
    diagnostics, qtbot
):
    # Y,T,X: label 1 belongs to a separate object at each of three time points.
    before = np.ones((2, 3, 4), dtype=np.int32)
    after = before.copy()
    after[:, 1, :] = 0
    before.setflags(write=False)
    after.setflags(write=False)
    with qtbot.waitSignal(diagnostics.ready) as emitted:
        diagnostics.request(before, after, 2, spatial_axes=(0, 2))
    assert emitted.args[0].matches(before, after, 2, spatial_axes=(0, 2))
    assert diagnostics.cached(
        before, after, 2, spatial_axes=(0, 2)
    ) == ObjectFilterCounts(3, 2, 1, 3)
    assert diagnostics.cached(before, after, 2) is None


def test_clear_cancels_old_generation_and_allows_new_request(
    diagnostics, qtbot, monkeypatch
):
    pair = _pair()
    entered, release = threading.Event(), threading.Event()
    calls, delivered = [], []

    def calculate(*_args, cancel_callback, **_kwargs):
        calls.append(cancel_callback)
        if len(calls) == 1:
            entered.set()
            assert release.wait(3)
        # Simulate a native phase ignoring cancellation until it returns.
        return ObjectFilterCounts(2, 1, 1, 1)

    monkeypatch.setattr(module, "object_filter_counts", calculate)
    diagnostics.ready.connect(delivered.append)
    diagnostics.request(*pair, 2)
    try:
        qtbot.waitUntil(entered.is_set)
        diagnostics.clear()
        assert calls[0]()
        diagnostics.request(*pair, 2)
    finally:
        release.set()
    qtbot.waitUntil(lambda: diagnostics.cached(*pair, 2) is not None)
    assert len(calls) == 2
    assert len(delivered) == 1
    assert delivered[0].matches(*pair, 2)


def test_new_requests_keep_only_latest_pending_pair(
    diagnostics, qtbot, monkeypatch
):
    entered, release = threading.Event(), threading.Event()
    first, middle, latest = _pair(), _pair(), _pair()
    middle_refs = [weakref.ref(value) for value in middle]
    calls, delivered = [], []

    def calculate(before, after, **_kwargs):
        calls.append((id(before), id(after)))
        if len(calls) == 1:
            entered.set()
            assert release.wait(3)
        return ObjectFilterCounts(2, 1, 1, 1)

    monkeypatch.setattr(module, "object_filter_counts", calculate)
    diagnostics.ready.connect(delivered.append)
    diagnostics.request(*first, 2)
    try:
        qtbot.waitUntil(entered.is_set)
        diagnostics.request(*middle, 2)
        for _ in range(10):
            diagnostics.request(*_pair(), 2)
        diagnostics.request(*latest, 2)
        del middle
        gc.collect()
        assert all(reference() is None for reference in middle_refs)
        assert len(calls) == 1
    finally:
        release.set()
    qtbot.waitUntil(lambda: diagnostics.cached(*latest, 2) is not None)
    assert calls == [(id(a), id(b)) for a, b in (first, latest)]
    assert diagnostics.cached(*first, 2) is None
    assert len(delivered) == 1 and delivered[0].matches(*latest, 2)


def test_clear_discards_result_queued_before_gui_delivery(
    diagnostics, qtbot, monkeypatch
):
    pair = _pair()
    delivered = []
    monkeypatch.setattr(
        module, "object_filter_counts", lambda *_a, **_k: ObjectFilterCounts(2, 1, 1, 1)
    )
    diagnostics.ready.connect(delivered.append)
    diagnostics.request(*pair, 2)
    assert diagnostics.wait(5000)
    diagnostics.clear()
    qtbot.wait(10)
    assert delivered == []
    assert diagnostics.cached(*pair, 2) is None


def test_errors_cached_once_and_clear_allows_retry(diagnostics, qtbot, monkeypatch):
    pair, calls = _pair(), []

    def calculate(*_args, **_kwargs):
        calls.append(1)
        raise ValueError("Output is not from a whole-object filter")

    monkeypatch.setattr(module, "object_filter_counts", calculate)
    with qtbot.waitSignal(diagnostics.ready):
        diagnostics.request(*pair, 2)
    assert diagnostics.cached(*pair, 2) is None
    assert diagnostics.error(*pair, 2).startswith("ValueError: Output is not")
    diagnostics.request(*pair, 2)
    assert len(calls) == 1
    diagnostics.clear()
    with qtbot.waitSignal(diagnostics.ready):
        diagnostics.request(*pair, 2)
    assert len(calls) == 2


def test_lru_weak_identities_do_not_retain_arrays(qapp, qtbot):
    controller = ObjectFilterDiagnostics(max_entries=2)
    pairs = [_pair() for _ in range(3)]
    try:
        for pair in pairs[:2]:
            with qtbot.waitSignal(controller.ready):
                controller.request(*pair, 2)
        assert controller.cached(*pairs[0], 2) is not None
        with qtbot.waitSignal(controller.ready):
            controller.request(*pairs[2], 2)
        assert controller.cached(*pairs[0], 2) is not None
        assert controller.cached(*pairs[1], 2) is None
        assert controller.cached(*pairs[2], 2) is not None
        original_key = ObjectFilterDiagnosticKey(*pairs[0], 2, "Face connected")
        other_key = ObjectFilterDiagnosticKey(*pairs[1], 2, "Face connected")
        controller._cache[other_key.cache_key] = controller._cache[
            original_key.cache_key
        ]
        assert controller.cached(*pairs[1], 2) is None
        references = [weakref.ref(value) for value in pairs[0]]
        del original_key, pairs[0]
        gc.collect()
        assert all(reference() is None for reference in references)
    finally:
        controller.close()
        assert controller.wait(5000)


def test_close_is_terminal_and_parent_teardown_joins_cancelled_worker(
    qapp, qtbot, monkeypatch
):
    parent = QObject()
    controller = ObjectFilterDiagnostics(parent)
    entered, cancelled = threading.Event(), threading.Event()
    calls, delivered = [], []

    def calculate(*_args, cancel_callback, **_kwargs):
        calls.append(1)
        entered.set()
        for _ in range(100):
            if cancel_callback():
                cancelled.set()
                return ObjectFilterCounts(2, 1, 1, 1)
            cancelled.wait(0.01)
        raise AssertionError("Worker did not receive cancellation")

    monkeypatch.setattr(module, "object_filter_counts", calculate)
    controller.ready.connect(delivered.append)
    controller.request(*_pair(), 2)
    qtbot.waitUntil(entered.is_set)
    controller.close()
    controller.request(*_pair(), 2)
    parent.deleteLater()
    QCoreApplication.sendPostedEvents(parent, QEvent.Type.DeferredDelete)
    assert cancelled.is_set()
    assert calls == [1]
    assert delivered == []


def test_destroyed_child_pool_is_not_accessed_by_python_teardown(qapp):
    parent = QObject()
    controller = ObjectFilterDiagnostics(parent)
    pool = controller._pool
    pool.deleteLater()
    QCoreApplication.sendPostedEvents(pool, QEvent.Type.DeferredDelete)
    parent.deleteLater()
    QCoreApplication.sendPostedEvents(parent, QEvent.Type.DeferredDelete)
