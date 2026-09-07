"""Exact mesh diagnostic populations, caching, and Qt worker lifetime."""

from __future__ import annotations

import gc
import threading
import weakref

import numpy as np
import pytest
from qtpy.QtCore import QCoreApplication, QEvent, QObject, QThread

from napari_vipp.core.meshes import MeshData, MeshState, measure_mesh_geometry
from napari_vipp.core.metadata import AxisMetadata
from napari_vipp.core.progress import OperationCancelled, ProgressContext
from napari_vipp.core.tables import TableData
from napari_vipp.ui import mesh_diagnostics
from napari_vipp.ui.mesh_diagnostics import MeshMeasurementDiagnostics


def _mesh(*, open_surface=False, empty=False):
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    if open_surface:
        faces = faces[:-1]
    if empty:
        faces = np.empty((0, 3), dtype=np.int64)
    return MeshData(
        vertices,
        faces,
        MeshState(
            len(vertices),
            len(faces),
            (
                AxisMetadata("z", "space", "um", 2),
                AxisMetadata("y", "space", "nm", 500),
                AxisMetadata("x", "space", "um", 0.25),
            ),
            (2, 2, 2),
            "supplied triangles",
        ),
    )


@pytest.fixture
def diagnostics(qapp):
    controller = MeshMeasurementDiagnostics()
    yield controller
    controller.close()
    assert controller.wait(5000)


@pytest.mark.parametrize(
    "open_surface,empty", [(False, False), (True, False), (False, True)]
)
def test_diagnostics_use_actual_calibrated_geometry_and_keep_all_rows(
    diagnostics, qtbot, open_surface, empty
):
    mesh = _mesh(open_surface=open_surface, empty=empty)
    before = (mesh.vertices.copy(), mesh.faces.copy(), mesh.face_object_ids.copy())
    expected = measure_mesh_geometry(mesh, include_convex_hull_metrics=False)
    with qtbot.waitSignal(diagnostics.ready, timeout=10000) as emitted:
        diagnostics.request(mesh)
    assert emitted.args[0] is mesh
    table = diagnostics.cached(mesh)
    assert table is not None
    assert table.columns == expected.columns
    assert table.column_units == expected.column_units
    assert table.row_count == expected.row_count
    assert table.unit_for("mesh_volume_physical") == "um^3"
    for actual_row, expected_row in zip(table.rows, expected.rows, strict=True):
        for actual, reference in zip(actual_row, expected_row, strict=True):
            if isinstance(reference, float) and np.isnan(reference):
                assert np.isnan(actual)
            else:
                assert actual == reference
    volume = table.rows[0][table.columns.index("mesh_volume_physical")]
    if open_surface or empty:
        assert np.isnan(volume)
    else:
        assert volume == pytest.approx(2 * 0.5 * 0.25 / 6)
    assert diagnostics.error(mesh) == ""
    for actual, original in zip(
        (mesh.vertices, mesh.faces, mesh.face_object_ids), before, strict=True
    ):
        np.testing.assert_array_equal(actual, original)
        assert not actual.flags.writeable


def test_pending_requests_and_metric_edits_reuse_one_off_gui_measurement(
    diagnostics, qtbot, monkeypatch
):
    mesh = _mesh()
    entered, release = threading.Event(), threading.Event()
    calls = []
    table = TableData(("mesh_id", "mesh_volume_physical"), ((1, 4.0),))
    gui_thread = QThread.currentThread()

    def calculate(value, *, include_convex_hull_metrics, progress):
        calls.append((value, include_convex_hull_metrics, QThread.currentThread()))
        entered.set()
        assert release.wait(3)
        progress.check_cancelled()
        return table

    monkeypatch.setattr(mesh_diagnostics, "measure_mesh_geometry", calculate)
    diagnostics.request(mesh)
    try:
        qtbot.waitUntil(entered.is_set)
        assert diagnostics.cached(mesh) is None
        for _ in range(5):
            diagnostics.request(mesh)
        assert len(calls) == 1
    finally:
        release.set()
    qtbot.waitUntil(lambda: diagnostics.cached(mesh) is not None)
    for property_name in table.columns:
        diagnostics.request(mesh)
        assert diagnostics.cached(mesh) is table
        assert property_name in table.columns
    assert len(calls) == 1
    assert calls[0][0] is mesh
    assert calls[0][1] is False
    assert calls[0][2] != gui_thread


def test_clear_cancels_running_work_and_discards_stale_completion(
    diagnostics, qtbot, monkeypatch
):
    mesh = _mesh()
    entered, release = threading.Event(), threading.Event()
    calls = []
    results = []

    def calculate(_value, *, progress, **_kwargs):
        calls.append(progress)
        if len(calls) == 1:
            entered.set()
            assert release.wait(3)
            # Deliberately ignore inner cancellation: the worker boundary must
            # discard even a library call that completes after clear().
        return TableData(("mesh_id",), ((len(calls),),))

    monkeypatch.setattr(mesh_diagnostics, "measure_mesh_geometry", calculate)
    diagnostics.ready.connect(results.append)
    diagnostics.request(mesh)
    try:
        qtbot.waitUntil(entered.is_set)
        diagnostics.clear()
        assert calls[0].is_cancelled()
        diagnostics.request(mesh)
    finally:
        release.set()
    qtbot.waitUntil(lambda: diagnostics.cached(mesh) is not None)
    assert len(calls) == 2
    assert diagnostics.cached(mesh).rows == ((2,),)
    assert results == [mesh]


def test_clear_discards_result_already_queued_to_gui(diagnostics, qtbot, monkeypatch):
    mesh = _mesh()
    results = []
    monkeypatch.setattr(
        mesh_diagnostics,
        "measure_mesh_geometry",
        lambda *_args, **_kwargs: TableData(("mesh_id",), ((1,),)),
    )
    diagnostics.ready.connect(results.append)
    diagnostics.request(mesh)
    assert diagnostics.wait(5000)
    # The worker finished, but its queued GUI slot has not run yet.
    diagnostics.clear()
    qtbot.wait(10)
    assert results == []
    assert diagnostics.cached(mesh) is None


def test_errors_are_cached_and_clear_allows_retry(diagnostics, qtbot, monkeypatch):
    mesh = _mesh()
    calls = []

    def calculate(*_args, **_kwargs):
        calls.append(1)
        raise ValueError("Invalid calibration")

    monkeypatch.setattr(mesh_diagnostics, "measure_mesh_geometry", calculate)
    with qtbot.waitSignal(diagnostics.ready):
        diagnostics.request(mesh)
    assert diagnostics.cached(mesh) is None
    assert diagnostics.error(mesh) == "ValueError: Invalid calibration"
    diagnostics.request(mesh)
    assert calls == [1]
    diagnostics.clear()
    with qtbot.waitSignal(diagnostics.ready):
        diagnostics.request(mesh)
    assert calls == [1, 1]


def test_cache_is_identity_guarded_lru_and_does_not_retain_meshes(
    qapp, qtbot, monkeypatch
):
    controller = MeshMeasurementDiagnostics(max_entries=2)
    meshes = [_mesh() for _ in range(3)]
    monkeypatch.setattr(
        mesh_diagnostics,
        "measure_mesh_geometry",
        lambda *_args, **_kwargs: TableData(("mesh_id",), ((1,),)),
    )
    try:
        for mesh in meshes[:2]:
            with qtbot.waitSignal(controller.ready):
                controller.request(mesh)
        first_table = controller.cached(meshes[0])
        with qtbot.waitSignal(controller.ready):
            controller.request(meshes[2])
        assert controller.cached(meshes[0]) is first_table
        assert controller.cached(meshes[1]) is None
        assert controller.cached(meshes[2]) is not None
        assert controller.cached(_mesh()) is None
        # Simulate an integer object-ID collision; identity, not id() alone,
        # must determine whether a retained table belongs to this input.
        controller._cache[id(meshes[1])] = controller._cache[id(meshes[0])]
        assert controller.cached(meshes[1]) is None
        first_ref = weakref.ref(meshes[0])
        del meshes[0]
        gc.collect()
        assert first_ref() is None
    finally:
        controller.close()
        assert controller.wait(5000)


def test_oversized_table_fails_once_without_retaining_it(qapp, qtbot, monkeypatch):
    controller = MeshMeasurementDiagnostics(max_bytes=1)
    mesh = _mesh()
    calls = []

    def calculate(*_args, **_kwargs):
        calls.append(1)
        return TableData(("mesh_id",), ((1,),))

    monkeypatch.setattr(mesh_diagnostics, "measure_mesh_geometry", calculate)
    try:
        with qtbot.waitSignal(controller.ready):
            controller.request(mesh)
        assert controller.cached(mesh) is None
        assert "memory limit" in controller.error(mesh)
        controller.request(mesh)
        assert calls == [1]
    finally:
        controller.close()
        assert controller.wait(5000)


def test_retained_table_memory_budget_evicts_old_entries(qapp, qtbot, monkeypatch):
    table = TableData(("mesh_id",), ((1,),))
    table_bytes = mesh_diagnostics._table_retained_bytes(table, ProgressContext())
    controller = MeshMeasurementDiagnostics(max_bytes=2 * table_bytes - 1)
    meshes = [_mesh(), _mesh()]
    monkeypatch.setattr(
        mesh_diagnostics, "measure_mesh_geometry", lambda *_args, **_kwargs: table
    )
    try:
        for mesh in meshes:
            with qtbot.waitSignal(controller.ready):
                controller.request(mesh)
        assert controller.cached(meshes[0]) is None
        assert controller.cached(meshes[1]) is table
    finally:
        controller.close()
        assert controller.wait(5000)


def test_close_is_terminal_and_cancels_pending_work(diagnostics, qtbot, monkeypatch):
    mesh = _mesh()
    entered, release = threading.Event(), threading.Event()
    observed_progress, results = [], []

    def calculate(*_args, progress, **_kwargs):
        observed_progress.append(progress)
        entered.set()
        assert release.wait(3)
        progress.check_cancelled()

    monkeypatch.setattr(mesh_diagnostics, "measure_mesh_geometry", calculate)
    diagnostics.ready.connect(results.append)
    diagnostics.request(mesh)
    try:
        qtbot.waitUntil(entered.is_set)
        diagnostics.close()
        assert observed_progress[0].is_cancelled()
        assert not diagnostics.wait(0)
        diagnostics.request(_mesh())
    finally:
        release.set()
    assert diagnostics.wait(5000)
    qtbot.wait(10)
    assert diagnostics.cached(mesh) is None
    assert results == []
    assert len(observed_progress) == 1


def test_explicit_close_before_parent_destruction_cancels_and_joins_worker(
    qapp, qtbot, monkeypatch
):
    parent = QObject()
    controller = MeshMeasurementDiagnostics(parent)
    entered, cancelled = threading.Event(), threading.Event()

    def calculate(*_args, progress, **_kwargs):
        entered.set()
        try:
            # A cancellation checkpoint stands in for each geometry phase.
            for _ in range(100):
                progress.check_cancelled()
                if cancelled.wait(0.01):
                    break
        except OperationCancelled:
            cancelled.set()
            raise
        return TableData(("mesh_id",), ((1,),))

    monkeypatch.setattr(mesh_diagnostics, "measure_mesh_geometry", calculate)
    controller.request(_mesh())
    qtbot.waitUntil(entered.is_set)
    controller.close()
    parent.deleteLater()
    QCoreApplication.sendPostedEvents(parent, QEvent.Type.DeferredDelete)
    assert cancelled.is_set()


def test_controller_destruction_never_touches_already_destroyed_child_pool(qapp):
    parent = QObject()
    controller = MeshMeasurementDiagnostics(parent)
    pool = controller._pool
    # Bindings can tear down child wrappers before dispatching a parent's
    # Python destroyed callback. That callback must not call Qt child methods.
    pool.deleteLater()
    QCoreApplication.sendPostedEvents(pool, QEvent.Type.DeferredDelete)
    parent.deleteLater()
    QCoreApplication.sendPostedEvents(parent, QEvent.Type.DeferredDelete)
