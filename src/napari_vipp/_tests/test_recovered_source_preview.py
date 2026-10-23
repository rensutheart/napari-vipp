"""Recovery rebuilds automatic previews without authorizing manual or file writes."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
from napari.components import ViewerModel
from qtpy.QtCore import QTimer

from napari_vipp._tests.test_source_load_coalescing import (
    fake_czi_source as fake_czi_source,
)
from napari_vipp._widget import (
    CACHE_MODE_KEEP_ALL,
    CACHE_MODE_LOW_MEMORY,
    CACHE_MODE_SMART,
    VippWidget,
)
from napari_vipp.core.compute import ComputeMode
from napari_vipp.core.file_sources import load_frozen_file_source_snapshot
from napari_vipp.core.metadata import image_state_from_array
from napari_vipp.core.operations import gaussian_blur, otsu_threshold
from napari_vipp.core.pipeline import SourcePayload
from napari_vipp.core.source_item_persistence import params_with_source_item


@pytest.fixture
def preview_widget(qtbot):
    widget = VippWidget(
        ViewerModel(), defer_initial_run=True, initial_compute_mode=ComputeMode.CPU
    )
    qtbot.addWidget(widget)
    widget.pipeline.nodes["input"].params.update(
        source_mode="sample", sample_name="VIPP synthetic volume"
    )
    widget._selected_node_id = "input"
    widget._sync_current_workflow_tab_state()
    yield widget
    widget._discard_incomplete_startup_on_close = True


def _checkpoint(widget):
    widget._sync_current_workflow_tab_state()
    return widget._workflow_recovery.checkpoint_now(required=True)


def _wait_updated(widget, qtbot, node_id="threshold", *, thumbnail_node_ids=()):
    qtbot.waitUntil(
        lambda: (
            widget.pipeline.outputs.get(node_id) is not None
            or bool(widget._source_inspection_errors.get("input"))
        )
        and widget._active_source_load_id is None
        and widget._active_pipeline_run_id is None
        # Exact stack contrast is deferred presentation work. Scientific
        # completion alone can precede even its zero-turn start callback.
        and widget._active_thumbnail_contrast_run_id is None
        and not widget._queued_thumbnail_contrast_limit_requests
        and not widget._pending_thumbnail_contrast_limit_keys
        and all(
            widget.graph_view.node_has_thumbnail(expected_node_id)
            for expected_node_id in thumbnail_node_ids
        ),
        timeout=15_000,
    )
    assert not widget._source_inspection_errors.get("input"), (
        widget._source_inspection_errors.get("input")
    )
    assert widget.pipeline.outputs.get(node_id) is not None


def _wait_source_error(widget, qtbot):
    qtbot.waitUntil(
        lambda: bool(widget._source_inspection_errors.get("input"))
        and widget._active_source_load_id is None
        and widget._active_pipeline_run_id is None,
        timeout=15_000,
    )


def _assert_unexecuted(widget, node_ids):
    for node_id in node_ids:
        assert widget.pipeline.outputs.get(node_id) is None
        assert all(
            value is None for value in widget.pipeline.node_outputs.get(node_id, ())
        )
        assert node_id not in widget.pipeline.completed_node_ids
        assert node_id not in widget.pipeline.node_compute_provenance


def _add_armed_writer(widget, tmp_path, source_id="threshold"):
    writer = widget.pipeline.add_node("save_output")
    assert widget.pipeline.connect(source_id, writer.id).success
    target = tmp_path / "explicit-calculation-only" / "result.npy"
    writer.params.update(path=str(target), enabled="on", format="npy")
    return writer, target


def _add_auto_manual_measurement(widget):
    labels = widget.pipeline.add_node("label_connected_components")
    assert widget.pipeline.connect("threshold", labels.id).success
    manual = widget.pipeline.add_node("measure_objects")
    assert widget.pipeline.connect(labels.id, manual.id).success
    widget.pipeline.set_node_auto_recalculate(manual.id, True)
    return labels, manual


def test_recovery_wait_includes_queued_thumbnail_statistics_before_worker_start(
    preview_widget, qtbot, monkeypatch, tmp_path
):
    widget = preview_widget
    _labels, manual = _add_auto_manual_measurement(widget)
    writer, target = _add_armed_writer(widget, tmp_path)
    widget._build_graph_from_pipeline()
    original_start = widget._start_thumbnail_contrast_limit_run
    held_starts, observed_queued_gap = [], []
    released = False

    def hold_start():
        held_starts.append(True)

    monkeypatch.setattr(widget, "_start_thumbnail_contrast_limit_run", hold_start)
    original_wait = qtbot.waitUntil

    def observe_wait(predicate, **kwargs):
        def observed_predicate():
            nonlocal released
            ready = predicate()
            scientific_ready = (
                widget.pipeline.outputs.get("threshold") is not None
                and widget._active_source_load_id is None
                and widget._active_pipeline_run_id is None
            )
            if scientific_ready and not released:
                # Let the real queued callback reach the gate before checking
                # the wait predicate. No clock delay or production event pump.
                if not held_starts:
                    return False
                assert widget._active_thumbnail_contrast_run_id is None
                assert widget._queued_thumbnail_contrast_limit_requests
                assert not widget._pending_thumbnail_contrast_limit_keys
                assert all(
                    not widget.graph_view.node_has_thumbnail(node_id)
                    for node_id in ("input", "gaussian", "threshold")
                )
                observed_queued_gap.append(True)
                assert not ready, "Scientific completion is not thumbnail readiness"
                released = True
                monkeypatch.setattr(
                    widget, "_start_thumbnail_contrast_limit_run", original_start
                )
                # Replay only the presentation callback which actually reached
                # the gate; normal statistics/render publication remains real.
                QTimer.singleShot(0, original_start)
            return ready

        original_wait(observed_predicate, **kwargs)

    monkeypatch.setattr(qtbot, "waitUntil", observe_wait)
    widget._restore_recovery_workspace(_checkpoint(widget))
    _wait_updated(widget, qtbot)

    assert held_starts and observed_queued_gap and released
    assert all(
        widget.graph_view.node_has_thumbnail(node_id)
        for node_id in ("input", "gaussian", "threshold")
    )
    assert widget._recovery_preview_restricted
    _assert_unexecuted(widget, {manual.id, writer.id})
    assert not target.parent.exists()


def test_recovered_sample_rebuilds_gaussian_and_otsu_but_never_writes(
    preview_widget, qtbot, tmp_path
):
    widget = preview_widget
    writer, target = _add_armed_writer(widget, tmp_path)
    writer_params = deepcopy(writer.params)
    sample = widget._sample_payloads()["VIPP synthetic volume"].data
    original_sample = np.array(sample, copy=True)
    expected_gaussian = gaussian_blur(sample, sigma=1.2, channel_axis=None)
    expected_mask = otsu_threshold(expected_gaussian, channel_axis=None)
    widget._build_graph_from_pipeline()

    widget._restore_recovery_workspace(_checkpoint(widget))
    _wait_updated(
        widget, qtbot, thumbnail_node_ids=("input", "gaussian", "threshold")
    )

    np.testing.assert_array_equal(widget.pipeline.outputs["input"], original_sample)
    np.testing.assert_allclose(widget.pipeline.outputs["gaussian"], expected_gaussian)
    np.testing.assert_array_equal(widget.pipeline.outputs["threshold"], expected_mask)
    np.testing.assert_array_equal(sample, original_sample)
    assert not sample.flags.writeable
    assert {"input", "gaussian", "threshold"} <= widget.pipeline.completed_node_ids
    assert all(
        widget.graph_view.node_has_thumbnail(node_id)
        for node_id in ("input", "gaussian", "threshold")
    )
    inspect = widget.viewer.layers["VIPP Inspect"]
    np.testing.assert_array_equal(inspect.data, original_sample)
    assert not inspect.data.flags.writeable
    assert widget._recovery_preview_restricted
    _assert_unexecuted(widget, {writer.id})
    assert widget.pipeline.nodes[writer.id].params == writer_params
    assert not target.parent.exists()
    assert widget._workflow_tabs.current.path is None
    assert widget._workflow_tabs.current.dirty


def test_manual_auto_recalculate_does_not_authorize_recovery_execution(
    preview_widget, qtbot, tmp_path
):
    widget = preview_widget
    labels, manual = _add_auto_manual_measurement(widget)
    writer, target = _add_armed_writer(widget, tmp_path)
    manual_params, writer_params = deepcopy(manual.params), deepcopy(writer.params)
    widget._build_graph_from_pipeline()

    widget._restore_recovery_workspace(_checkpoint(widget))
    _wait_updated(widget, qtbot, labels.id)

    assert widget.pipeline.node_auto_recalculate(manual.id)
    _assert_unexecuted(widget, {manual.id, writer.id})
    assert widget.pipeline.nodes[manual.id].params == manual_params
    assert widget.pipeline.nodes[writer.id].params == writer_params
    assert not target.parent.exists()


def test_automatic_edit_keeps_manual_and_writer_barriers_until_explicit_calculate(
    preview_widget, qtbot, tmp_path
):
    widget = preview_widget
    _labels, manual = _add_auto_manual_measurement(widget)
    writer, target = _add_armed_writer(widget, tmp_path)
    widget._build_graph_from_pipeline()
    widget._restore_recovery_workspace(_checkpoint(widget))
    _wait_updated(widget, qtbot)

    widget.pipeline.set_param("gaussian", "sigma", 0.0)
    widget._mark_pipeline_dirty("gaussian")
    widget.run_pipeline(force_sync=True)
    _wait_updated(widget, qtbot)

    np.testing.assert_array_equal(
        widget.pipeline.outputs["gaussian"], widget.pipeline.outputs["input"]
    )
    assert widget._recovery_preview_restricted
    _assert_unexecuted(widget, {manual.id, writer.id})
    assert not target.parent.exists()


def test_explicit_calculate_all_releases_recovery_manual_and_writer_barriers(
    preview_widget, qtbot, tmp_path
):
    widget = preview_widget
    _labels, manual = _add_auto_manual_measurement(widget)
    writer, target = _add_armed_writer(widget, tmp_path)
    widget._build_graph_from_pipeline()
    widget._restore_recovery_workspace(_checkpoint(widget))
    _wait_updated(widget, qtbot)
    assert not target.exists()
    _assert_unexecuted(widget, {manual.id, writer.id})

    widget.calculate_all_button.click()
    qtbot.waitUntil(
        lambda: target.exists()
        and widget.pipeline.outputs.get(manual.id) is not None
        and widget._active_pipeline_run_id is None,
        timeout=15_000,
    )

    assert not widget._recovery_preview_restricted
    np.testing.assert_array_equal(np.load(target), widget.pipeline.outputs["threshold"])
    assert widget.pipeline.outputs[manual.id].row_count > 0
    assert {manual.id, writer.id} <= widget.pipeline.completed_node_ids


def test_writer_descendants_and_fanin_are_excluded_from_automatic_recovery(
    preview_widget, qtbot, tmp_path
):
    widget = preview_widget
    writer, target = _add_armed_writer(widget, tmp_path, source_id="gaussian")
    join = widget.pipeline.add_node("add_images")
    assert widget.pipeline.connect("input", join.id, target_port=0).success
    assert widget.pipeline.connect(writer.id, join.id, target_port=1).success
    tail = widget.pipeline.add_node("gaussian_blur")
    assert widget.pipeline.connect(join.id, tail.id).success
    widget._build_graph_from_pipeline()

    widget._restore_recovery_workspace(_checkpoint(widget))
    _wait_updated(widget, qtbot)

    _assert_unexecuted(widget, {writer.id, join.id, tail.id})
    assert widget.pipeline.outputs["threshold"].dtype == bool
    assert not target.parent.exists()


def test_recovered_file_is_verified_and_calculated_without_changing_source(
    preview_widget, qtbot, tmp_path
):
    widget = preview_widget
    data = np.arange(70, dtype=np.uint16).reshape(2, 5, 7)
    path = tmp_path / "source.npy"
    np.save(path, data)
    before = path.read_bytes()
    widget.pipeline.nodes["input"].params.update(
        source_mode="file path", file_path=str(path), axis_declaration="ZYX -> ZYX"
    )

    widget._restore_recovery_workspace(_checkpoint(widget))
    _wait_updated(widget, qtbot)

    np.testing.assert_array_equal(widget.pipeline.outputs["input"], data)
    np.testing.assert_array_equal(
        widget.pipeline.outputs["threshold"],
        otsu_threshold(gaussian_blur(data, sigma=1.2)),
    )
    assert path.read_bytes() == before
    assert [axis.name for axis in widget.pipeline.output_states["input"].axes] == [
        "z", "y", "x"
    ]


@pytest.mark.parametrize("saved_item", [False, True])
def test_async_recovered_czi_is_read_once_and_automatic_preview_runs(
    preview_widget, qtbot, fake_czi_source, saved_item
):
    widget = preview_widget
    source = widget.pipeline.nodes["input"]
    source.params.update(
        source_mode="file path", file_path=str(fake_czi_source.path)
    )
    if saved_item:
        snapshot = load_frozen_file_source_snapshot(fake_czi_source.path)
        source.params = params_with_source_item(
            source.params, snapshot.source_item, legacy_series_index=999
        )
    checkpoint = _checkpoint(widget)
    fake_czi_source.events.clear()

    widget._restore_recovery_workspace(checkpoint)
    _wait_updated(widget, qtbot)

    expected = np.full((2, 4, 5), 9, dtype=np.uint8)
    np.testing.assert_array_equal(widget.pipeline.outputs["input"], expected)
    np.testing.assert_array_equal(widget.pipeline.outputs["gaussian"], expected)
    assert widget.pipeline.outputs["threshold"].dtype == bool
    assert fake_czi_source.events.count("pixels") == 1
    assert widget._recovery_preview_restricted
    assert widget.pipeline.nodes["input"].params["_vipp_source_item"]["selector"][
        "key"
    ] == "9"


@pytest.mark.parametrize("rgb", [False, True])
def test_recovered_source_preserves_rgb_and_scalar_axis_semantics(
    preview_widget, qtbot, rgb
):
    widget = preview_widget
    widget.pipeline.reset_empty_graph()
    data = np.arange(60, dtype=np.uint8).reshape(4, 5, 3)
    data.setflags(write=False)
    metadata = {"axes": "Y,X,rgb" if rgb else "Z,Y,X", "rgb": rgb}
    state = image_state_from_array(data, layer_metadata=metadata, source_name="tiny")
    assert state is not None
    widget._sample_payload_cache = {
        "tiny": SourcePayload(data, metadata, "tiny", state)
    }
    widget.pipeline.nodes["input"].params.update(
        source_mode="sample", sample_name="tiny"
    )
    widget._selected_node_id = "input"
    widget._build_graph_from_pipeline()

    widget._restore_recovery_workspace(_checkpoint(widget))
    _wait_updated(widget, qtbot, "input")

    layer = widget.viewer.layers["VIPP Inspect"]
    assert layer.rgb is rgb
    np.testing.assert_array_equal(layer.data, data)
    assert tuple(layer.scale) == (1,) * (data.ndim - int(rgb))
    assert not layer.data.flags.writeable


def test_missing_recovered_source_reports_failure_without_sample_substitution(
    preview_widget, qtbot, tmp_path
):
    widget = preview_widget
    missing = tmp_path / "missing.npy"
    widget.pipeline.nodes["input"].params.update(
        source_mode="file path", file_path=str(missing)
    )

    widget._restore_recovery_workspace(_checkpoint(widget))
    _wait_source_error(widget, qtbot)

    _assert_unexecuted(widget, {"input", "gaussian", "threshold"})
    assert widget.pipeline.nodes["input"].params["file_path"] == str(missing)
    assert not missing.exists()


def test_recovered_preview_rejects_mismatched_axis_mapping(
    preview_widget, qtbot, tmp_path
):
    widget = preview_widget
    path = tmp_path / "source.npy"
    np.save(path, np.arange(70, dtype=np.uint16).reshape(2, 5, 7))
    widget.pipeline.nodes["input"].params.update(
        source_mode="file path", file_path=str(path), axis_declaration="QYX -> ZYX"
    )

    widget._restore_recovery_workspace(_checkpoint(widget))
    _wait_source_error(widget, qtbot)

    assert "Axis declaration expects QYX, but this source reports ZYX." in (
        widget._source_inspection_errors["input"]
    )
    _assert_unexecuted(widget, {"input", "gaussian", "threshold"})


def test_changed_saved_source_is_not_silently_accepted_for_recovery_calculation(
    preview_widget, qtbot, fake_czi_source
):
    widget = preview_widget
    snapshot = load_frozen_file_source_snapshot(fake_czi_source.path)
    source = widget.pipeline.nodes["input"]
    source.params.update(
        source_mode="file path", file_path=str(fake_czi_source.path)
    )
    source.params = params_with_source_item(source.params, snapshot.source_item)
    original_proof = deepcopy(source.params["_vipp_source_item"])
    checkpoint = _checkpoint(widget)
    fake_czi_source.path.write_bytes(b"a-new-source-revision")
    fake_czi_source.events.clear()

    widget._restore_recovery_workspace(checkpoint)
    _wait_source_error(widget, qtbot)

    assert fake_czi_source.events.count("pixels") == 0
    assert widget.pipeline.nodes["input"].params["_vipp_source_item"] == original_proof
    _assert_unexecuted(widget, {"input", "gaussian", "threshold"})


def test_inactive_recovered_tab_calculates_automatic_preview_only_when_activated(
    preview_widget, qtbot, tmp_path
):
    widget = preview_widget
    writer, target = _add_armed_writer(widget, tmp_path)
    widget._build_graph_from_pipeline()
    tabs, active = widget._capture_recovery_workspace()
    second = deepcopy(tabs[0])
    second["tab_id"] = "second-interrupted-tab"
    second["title"] = "Second interrupted analysis"
    checkpoint = widget._workflow_recovery.store.checkpoint([tabs[0], second], active)

    ids = widget._restore_recovery_workspace(checkpoint)
    _wait_updated(widget, qtbot)
    inactive = widget._workflow_tabs[widget._workflow_tabs.index_of(ids[1])]
    assert all(value is None for value in inactive.pipeline.outputs.values())
    assert not inactive.pipeline.completed_node_ids
    qtbot.waitUntil(
        lambda: not widget._workflow_tab_switch_block_reason(), timeout=10_000
    )

    assert widget._activate_workflow_tab(widget._workflow_tabs.index_of(ids[1]))
    _wait_updated(widget, qtbot)

    assert widget._workflow_tabs.current is inactive
    assert widget._recovery_preview_restricted
    _assert_unexecuted(widget, {writer.id})
    assert not target.parent.exists()


def test_queued_source_completion_cannot_calculate_another_tab(
    preview_widget, qtbot, monkeypatch
):
    widget = preview_widget
    original = widget._workflow_tabs.current
    widget._restore_recovery_workspace(_checkpoint(widget))
    _wait_updated(widget, qtbot)
    widget._queue_source_work_continuation()
    assert widget._activate_workflow_tab(
        widget._workflow_tabs.index_of(original.session_id), check_safety=False
    )
    runs, resumes = [], []
    monkeypatch.setattr(widget, "run_pipeline", lambda **kwargs: runs.append(kwargs))
    monkeypatch.setattr(
        widget, "_resume_pending_source_work", lambda: resumes.append("resume")
    )
    event_turn = []
    QTimer.singleShot(0, lambda: event_turn.append("processed"))
    qtbot.waitUntil(lambda: bool(event_turn), timeout=1000)

    assert widget._workflow_tabs.current is original
    assert runs == [] and resumes == []
    assert not widget._recovery_preview_restricted
    assert all(value is None for value in widget.pipeline.outputs.values())


@pytest.mark.parametrize(
    "cache_mode", [CACHE_MODE_KEEP_ALL, CACHE_MODE_SMART, CACHE_MODE_LOW_MEMORY]
)
def test_recovery_automatic_previews_obey_cache_mode_and_keep_writers_paused(
    preview_widget, qtbot, tmp_path, cache_mode
):
    widget = preview_widget
    writer, target = _add_armed_writer(widget, tmp_path)
    widget._selected_node_id = "threshold"
    widget._build_graph_from_pipeline()
    widget.cache_mode_combo.setCurrentText(cache_mode)

    widget._restore_recovery_workspace(_checkpoint(widget))
    _wait_updated(widget, qtbot, thumbnail_node_ids=("threshold",))

    assert widget.graph_view.node_has_thumbnail("threshold")
    assert widget.viewer.layers["VIPP Inspect"].metadata["node_id"] == "threshold"
    assert widget._recovery_preview_restricted
    _assert_unexecuted(widget, {writer.id})
    assert not target.parent.exists()
    assert widget._cache_mode() == cache_mode


def test_smart_selection_cannot_cross_a_recovered_manual_barrier(
    preview_widget, qtbot, tmp_path
):
    widget = preview_widget
    _labels, manual = _add_auto_manual_measurement(widget)
    writer, target = _add_armed_writer(widget, tmp_path)
    widget._build_graph_from_pipeline()
    widget.cache_mode_combo.setCurrentText(CACHE_MODE_SMART)
    widget._restore_recovery_workspace(_checkpoint(widget))
    _wait_updated(widget, qtbot)

    widget.graph_view.select_node(manual.id)
    event_turn = []
    QTimer.singleShot(0, lambda: event_turn.append("processed"))
    qtbot.waitUntil(lambda: bool(event_turn), timeout=1000)

    assert widget._recovery_preview_restricted
    _assert_unexecuted(widget, {manual.id, writer.id})
    assert not target.parent.exists()


def test_bypass_presentation_shadow_cannot_execute_a_recovered_writer(
    preview_widget, qtbot, tmp_path
):
    widget = preview_widget
    writer, target = _add_armed_writer(widget, tmp_path, source_id="input")
    bypassed = widget.pipeline.add_node("gaussian_blur")
    assert widget.pipeline.connect(writer.id, bypassed.id).success
    assert widget.pipeline.set_node_execution_mode(bypassed.id, "bypass")
    widget._build_graph_from_pipeline()

    widget._restore_recovery_workspace(_checkpoint(widget))
    _wait_updated(widget, qtbot)

    assert widget.pipeline.nodes[bypassed.id].execution_mode == "bypass"
    _assert_unexecuted(widget, {writer.id, bypassed.id})
    assert not target.parent.exists()
    assert widget._recovery_preview_restricted
    assert widget.pipeline.outputs["threshold"] is not None


def test_explicit_single_manual_calculation_keeps_unrelated_writers_paused(
    preview_widget, qtbot, tmp_path
):
    widget = preview_widget
    _labels, manual = _add_auto_manual_measurement(widget)
    writer, target = _add_armed_writer(widget, tmp_path)
    widget._build_graph_from_pipeline()
    widget._restore_recovery_workspace(_checkpoint(widget))
    _wait_updated(widget, qtbot)
    _assert_unexecuted(widget, {manual.id, writer.id})

    widget.graph_view.select_node(manual.id)
    assert not widget.calculate_button.isHidden()
    assert widget.calculate_button.isEnabled()
    widget.calculate_button.click()
    qtbot.waitUntil(
        lambda: widget.pipeline.outputs.get(manual.id) is not None
        and widget._active_pipeline_run_id is None,
        timeout=15_000,
    )

    assert widget.pipeline.outputs[manual.id].row_count > 0
    assert widget.pipeline.node_auto_recalculate(manual.id)
    assert widget._recovery_preview_restricted
    _assert_unexecuted(widget, {writer.id})
    assert not target.parent.exists()


def test_delayed_initial_run_cannot_release_recovered_writer_restrictions(
    preview_widget, qtbot, tmp_path
):
    widget = preview_widget
    writer, target = _add_armed_writer(widget, tmp_path)
    widget._build_graph_from_pipeline()
    widget._restore_recovery_workspace(_checkpoint(widget))
    _wait_updated(widget, qtbot)

    assert widget.run_initial_pipeline_once() is False
    event_turn = []
    QTimer.singleShot(0, lambda: event_turn.append("processed"))
    qtbot.waitUntil(lambda: bool(event_turn), timeout=1000)

    assert widget._recovery_preview_restricted
    _assert_unexecuted(widget, {writer.id})
    assert not target.parent.exists()
