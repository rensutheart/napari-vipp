"""Focused read-only Batch workflow host contracts."""

from __future__ import annotations

from dataclasses import replace
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from napari_vipp._widget import VippWidget
from napari_vipp.core.batch import BatchScientificPreflightError
from napari_vipp.core.compute import ComputeMode, ComputeRequest
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.source_identity import SourceChangedError
from napari_vipp.core.workflow import serialize_workflow
from napari_vipp.ui.batch_controller import (
    CollectionBatchController,
    execute_prepared_collection_batch_preview,
)


@pytest.fixture
def batch_case(tmp_path):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    pipeline.nodes["input"].params["binding_mode"] = "collection"
    output = pipeline.add_node("batch_output")
    output.params.update(tag="result", format="npy")
    assert pipeline.connect("input", output.id).success
    workflow = serialize_workflow(pipeline)
    controller = CollectionBatchController(
        workflow_document_provider=lambda: workflow,
        pipeline_provider=lambda: pipeline,
    )
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for index in range(2):
        np.save(inputs / f"field_{index}.npy", np.full((4, 5), index, dtype=np.uint8))
    values = {
        "input_dir": inputs,
        "output_dir": tmp_path / "outputs",
        "pattern": "*.npy",
        "image_format": "npy",
    }
    return controller, values, workflow


def test_selected_recheck_keeps_original_full_plan_and_skips_other_sources(batch_case):
    controller, values, _workflow = batch_case
    reviewed = controller.preview(**values)
    prepared = controller.prepare_item_recheck(reviewed, (0,), **values)
    np.save(values["input_dir"] / "field_1.npy", np.full((4, 5), 8, dtype=np.uint8))

    result = execute_prepared_collection_batch_preview(prepared)

    assert result is reviewed
    assert len(result.items) == 2
    assert not values["output_dir"].exists()


def test_selected_recheck_detects_exact_changed_content(batch_case):
    controller, values, _workflow = batch_case
    reviewed = controller.preview(**values)
    prepared = controller.prepare_item_recheck(reviewed, (0,), **values)
    np.save(values["input_dir"] / "field_0.npy", np.full((4, 5), 9, dtype=np.uint8))

    with pytest.raises(SourceChangedError):
        execute_prepared_collection_batch_preview(prepared)
    assert not values["output_dir"].exists()


def test_selected_recheck_detects_destination_presence_change(batch_case):
    controller, values, _workflow = batch_case
    reviewed = controller.preview(**values)
    prepared = controller.prepare_item_recheck(reviewed, (0,), **values)
    output = reviewed.items[0].outputs[0].path
    output.parent.mkdir(parents=True)
    output.write_bytes(b"external file")

    with pytest.raises(ValueError, match="Output presence changed"):
        execute_prepared_collection_batch_preview(prepared)
    assert output.read_bytes() == b"external file"


def test_selected_recheck_requires_exact_source_revision_and_unchanged_settings(
    batch_case,
):
    controller, values, _workflow = batch_case
    reviewed = controller.preview(**values)
    legacy = replace(reviewed, items=(replace(reviewed.items[0], source_items={}),))
    prepared = controller.prepare_item_recheck(legacy, (0,), **values)
    with pytest.raises(ValueError, match="no recorded exact source revision"):
        execute_prepared_collection_batch_preview(prepared)
    changed = dict(values, existing_file_policy="skip")
    with pytest.raises(ValueError, match="settings changed"):
        controller.prepare_item_recheck(reviewed, (0,), **changed)
    with pytest.raises(ValueError, match="Select current"):
        controller.prepare_item_recheck(reviewed, (4,), **values)


class _CheckHost:
    """Weak-referenceable receiver for real Qt worker signals."""

    def __init__(self, **values):
        self.__dict__.update(values)


def _check_host(batch_case):
    controller, values, workflow = batch_case
    published = []
    failures = []
    rechecks = []
    scheduled = []
    statuses = []
    session = SimpleNamespace(session_id="origin", runtime_cache={})
    dialog = SimpleNamespace(
        values=lambda: dict(values),
        _preview_result=None,
        _pending_parameter_overrides=(),
        _loaded_config_path=None,
        _checking_plan=False,
        _check_progress=[],
        parameter_override_editor=SimpleNamespace(error_message=""),
        apply_axis_suggestion=lambda _error: False,
        show_workspace_activity=lambda *_args, **_kwargs: None,
        _sync_workspace=lambda: None,
    )

    def publish(result, *, preview_representative):
        assert preview_representative is False
        dialog._preview_result = result
        dialog._checking_plan = False
        published.append(result)

    def fail(message, **_kwargs):
        dialog._checking_plan = False
        failures.append(message)

    def recheck(indices, message, *, unchanged):
        dialog._checking_plan = False
        rechecks.append((indices, message, unchanged))
        if not unchanged:
            dialog._preview_result = None

    dialog.apply_preview_result = publish
    dialog._show_preview_failure = fail
    dialog.show_item_recheck_result = recheck
    dialog.cancel_saved_workspace_discovery = fail

    def progress_update(progress, **kwargs):
        dialog._check_progress.append((progress, kwargs))

    dialog.show_check_progress = progress_update
    host = _CheckHost(
        _active_collection_batch_dialog=dialog,
        _workflow_tabs=SimpleNamespace(current=session),
        _collection_batch_running=False,
        _pending_collection_batch_start=None,
        _collection_batch_controller=controller,
        _batch_workspace_preview_serial=0,
        _batch_workspace_preview_contexts={},
        _batch_workspace_preview_workers={},
        _batch_workspace_preview_thread_pool=SimpleNamespace(start=scheduled.append),
        _closing=False,
        _interactive_collection_batch_items=(),
        _interactive_collection_batch_config=None,
        _interactive_collection_batch_plan_stale=False,
        _commit_crop_draft=lambda **_kwargs: None,
        _compute_request_for_batch_dialog=lambda _dialog: ComputeRequest(),
        _engage_collection_batch_workspace=lambda _dialog: None,
        _configure_batch_parameter_overrides=lambda *_args: True,
        _sync_current_workflow_tab_state=lambda: None,
        _workflow_tab_session=lambda _id: session,
        _workflow_tab_is_active=lambda _id: True,
        _batch_workflow_document=lambda: workflow,
        _set_status=lambda message, **_kwargs: statuses.append(message),
        _mark_interactive_collection_batch_stale=lambda _dialog: setattr(
            host, "_interactive_collection_batch_plan_stale", True
        ),
    )
    for name in (
        "_check_collection_batch",
        "_recheck_collection_batch_items",
        "_cancel_attached_batch_workspace_preview",
        "_on_batch_workspace_preview_progress",
        "_on_attached_batch_workspace_preview_finished",
        "_present_attached_batch_workspace_preview",
        "_present_collection_batch_check",
    ):
        setattr(host, name, MethodType(getattr(VippWidget, name), host))
    return host, dialog, scheduled, published, failures, rechecks


def test_check_schedules_read_only_work_and_publishes_without_graph_preview(
    qtbot,
    batch_case,
):
    del qtbot
    host, dialog, scheduled, published, failures, _rechecks = _check_host(batch_case)

    assert host._check_collection_batch(dialog.values())
    assert dialog._checking_plan
    assert published == []
    assert not batch_case[1]["output_dir"].exists()
    scheduled[0].run()

    assert len(published) == 1
    assert failures == []
    assert host._interactive_collection_batch_items == ()
    assert not dialog._checking_plan
    assert not host._batch_workspace_preview_workers
    assert not batch_case[1]["output_dir"].exists()


def test_superseded_check_never_publishes(qtbot, batch_case):
    del qtbot
    host, dialog, scheduled, published, failures, _rechecks = _check_host(batch_case)
    assert host._check_collection_batch(dialog.values())
    assert host._check_collection_batch(dialog.values())
    assert scheduled[0].cancellation_requested
    scheduled[0].run()
    assert published == []
    scheduled[1].run()
    assert len(published) == 1
    assert failures == []


def test_check_rejects_settings_changed_during_worker(qtbot, batch_case):
    del qtbot
    host, dialog, scheduled, published, failures, _rechecks = _check_host(batch_case)
    assert host._check_collection_batch(dialog.values())
    batch_case[1]["existing_file_policy"] = "skip"
    scheduled[0].run()
    assert published == []
    assert "settings or the workflow changed" in failures[0]


def test_check_does_not_publish_to_replaced_dialog(qtbot, batch_case):
    del qtbot
    host, dialog, scheduled, published, failures, _rechecks = _check_host(batch_case)
    assert host._check_collection_batch(dialog.values())
    host._active_collection_batch_dialog = SimpleNamespace()
    scheduled[0].run()
    assert published == failures == []


def test_check_rejects_changed_scientific_workflow(qtbot, batch_case):
    del qtbot
    host, dialog, scheduled, published, failures, _rechecks = _check_host(batch_case)
    assert host._check_collection_batch(dialog.values())
    batch_case[2]["nodes"][-1]["params"]["tag"] = "new-scientific-output"
    scheduled[0].run()
    assert published == []
    assert "settings or the workflow changed" in failures[0]


def test_check_result_waits_for_owning_workflow_tab(qtbot, batch_case):
    del qtbot
    host, dialog, scheduled, published, failures, _rechecks = _check_host(batch_case)
    assert host._check_collection_batch(dialog.values())
    host._workflow_tab_is_active = lambda _id: False
    scheduled[0].run()
    assert published == failures == []
    deferred = host._workflow_tabs.current.runtime_cache[
        "_batch_workspace_preview_outcome"
    ]
    host._workflow_tab_is_active = lambda _id: True
    assert host._present_attached_batch_workspace_preview(*deferred)
    assert len(published) == 1


def test_check_retries_reviewed_axis_suggestion_only_once(
    qtbot, monkeypatch, batch_case
):
    del qtbot
    host, dialog, scheduled, published, failures, _rechecks = _check_host(batch_case)
    suggestions = []
    dialog.apply_axis_suggestion = lambda error: suggestions.append(error) or True

    def fail(_prepared, **_kwargs):
        raise BatchScientificPreflightError("axis check failed")

    monkeypatch.setattr(
        "napari_vipp.ui.batch_workers.execute_prepared_collection_batch_preview", fail
    )
    assert host._check_collection_batch(dialog.values())
    scheduled[0].run()
    assert len(scheduled) == 2
    scheduled[1].run()
    assert len(scheduled) == 2
    assert len(suggestions) == 1
    assert failures == ["axis check failed"]
    assert published == []


def test_check_does_not_start_during_batch_execution(batch_case):
    host, dialog, scheduled, _published, _failures, _rechecks = _check_host(batch_case)
    host._collection_batch_running = True
    with pytest.raises(RuntimeError, match="active batch"):
        host._check_collection_batch(dialog.values())
    assert not scheduled


def test_item_recheck_preserves_plan_and_invalidates_on_discrepancy(qtbot, batch_case):
    del qtbot
    host, dialog, scheduled, published, _failures, rechecks = _check_host(batch_case)
    assert host._check_collection_batch(dialog.values())
    scheduled.pop().run()
    reviewed = dialog._preview_result
    assert host._recheck_collection_batch_items((1,))
    scheduled.pop().run()
    assert dialog._preview_result is reviewed
    assert len(published) == 1
    assert rechecks[-1][0] == (1,)
    assert rechecks[-1][2] is True
    assert host._recheck_collection_batch_items((1,))
    np.save(batch_case[1]["input_dir"] / "field_1.npy", np.zeros((8, 9)))
    scheduled.pop().run()
    assert rechecks[-1][2] is False
    assert dialog._preview_result is None


def test_full_check_marks_previous_host_plan_stale(qtbot, batch_case):
    del qtbot
    host, dialog, scheduled, _published, _failures, _rechecks = _check_host(batch_case)
    old = batch_case[0].preview(**batch_case[1])
    host._interactive_collection_batch_items = old.items
    host._interactive_collection_batch_config = old.config
    np.save(batch_case[1]["input_dir"] / "field_1.npy", np.zeros((7, 8)))
    host._check_collection_batch(dialog.values())
    scheduled.pop().run()
    assert host._interactive_collection_batch_plan_stale
    assert host._interactive_collection_batch_items is old.items


def test_graph_preview_uses_displayed_plan_instead_of_stale_host_items(batch_case):
    reviewed = batch_case[0].preview(**batch_case[1])
    activated = []
    host = SimpleNamespace(
        _active_collection_batch_dialog=SimpleNamespace(
            _preview_result=reviewed, _loaded_config_path=None
        ),
        _interactive_collection_batch_items=(object(),),
        _interactive_collection_batch_config=reviewed.config,
        _interactive_collection_batch_plan_stale=True,
        _activate_interactive_collection_batch=lambda *args, **kwargs: activated.append(
            (args, kwargs)
        ),
        _preview_interactive_collection_batch_item=lambda _index: pytest.fail(
            "Must not navigate the stale host plan"
        ),
    )

    assert VippWidget._preview_collection_batch_plan_item(host, 1)
    assert activated[0][0] == (reviewed.items, reviewed.config)
    assert activated[0][1]["initial_index"] == 1
    host._active_collection_batch_dialog._preview_result = None
    assert not VippWidget._preview_collection_batch_plan_item(host, 0)


@pytest.mark.parametrize(
    ("changed_source", "stale", "reuse"),
    [(False, False, True), (True, False, False), (False, True, False)],
)
def test_activation_reuses_only_current_exact_source_revision(
    batch_case, changed_source, stale, reuse
):
    controller, values, workflow = batch_case
    previous = controller.preview(**values)
    if changed_source:
        np.save(values["input_dir"] / "field_0.npy", np.zeros((8, 9), dtype=np.uint8))
    reviewed = controller.preview(**values)
    previews = []
    host = _CheckHost(
        _interactive_collection_batch_items=previous.items,
        _interactive_collection_batch_config=previous.config,
        _interactive_collection_batch_index=0,
        _interactive_collection_batch_requested_index=-1,
        _interactive_collection_batch_failed_index=-1,
        _interactive_collection_batch_plan_stale=stale,
        _interactive_collection_source_paths=dict(previous.items[0].source_paths),
        _interactive_collection_source_series_indices=dict(
            previous.items[0].source_series_indices
        ),
        _active_pipeline_run_id=None,
        _active_source_load_id=None,
        _pipeline_run_pending=False,
        _interactive_collection_source_node_ids=lambda: {"input"},
        _batch_workflow_document=lambda: workflow,
        _batch_item_parameter_override_signature=(
            VippWidget._batch_item_parameter_override_signature
        ),
        _batch_node_execution_override_signature=(
            VippWidget._batch_node_execution_override_signature
        ),
        _prune_file_source_payload_cache=lambda: None,
        _sync_input_node_subtitles=lambda _ids: None,
        _sync_interactive_collection_batch_navigator=lambda: None,
        _preview_interactive_collection_batch_item=lambda index, **_kwargs: (
            previews.append(index)
        ),
        batch_navigator=SimpleNamespace(
            set_session_stale=lambda _value: None, reset_batch_progress=lambda: None
        ),
    )

    VippWidget._activate_interactive_collection_batch(
        host, reviewed.items, reviewed.config
    )

    assert previews == ([] if reuse else [0])
    assert host._interactive_collection_batch_items is reviewed.items


def test_compute_summary_reports_saved_intent_until_toolbar_changes():
    current = [ComputeRequest(mode=ComputeMode.CPU)]
    saved = ComputeRequest(
        mode=ComputeMode.PREFER_GPU,
        runtime_id="cuda-cupy",
        device_id="cuda:0",
        accelerator_memory_cap_bytes=4_000_000_000,
    )
    dialog = SimpleNamespace(
        _loaded_compute_request=saved,
        _compute_toolbar_fingerprint_at_load=current[0].fingerprint,
    )
    host = SimpleNamespace(
        _active_collection_batch_dialog=dialog,
        _current_compute_request=lambda: current[0],
    )
    host._compute_request_for_batch_dialog = MethodType(
        VippWidget._compute_request_for_batch_dialog, host
    )
    label, tooltip = VippWidget._collection_batch_compute_summary(host)
    assert label == "Prefer GPU · saved batch"
    assert "Compute mode: Prefer GPU." in tooltip
    assert "Settings loaded from the saved batch configuration." in tooltip
    assert "main workflow toolbar replaces these saved settings" in tooltip
    assert "not a claim about actual GPU use" in tooltip
    assert "cuda-cupy" in tooltip
    assert "4,000,000,000 bytes" in tooltip
    current[0] = ComputeRequest(mode=ComputeMode.AUTO)
    label, tooltip = VippWidget._collection_batch_compute_summary(host)
    assert label == "Auto · inherited"
    assert "Compute mode: Auto." in tooltip
    assert "Settings inherited from the main workflow toolbar." in tooltip
    assert dialog._loaded_compute_request is None
    assert "cuda-cupy" not in tooltip


def test_real_widget_check_and_selected_recheck_are_read_only(
    qtbot, monkeypatch, tmp_path
):
    from napari_vipp._tests.test_widget import (
        _configure_workflow_attached_batch,
        _Viewer,
    )

    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    dialog, inputs, outputs = _configure_workflow_attached_batch(widget, tmp_path)
    monkeypatch.setattr(
        widget,
        "_activate_interactive_collection_batch",
        lambda *_args, **_kwargs: pytest.fail(
            "Read-only checks must not activate representative pixels."
        ),
    )

    assert dialog._check_batch()
    assert dialog._checking_plan
    assert not dialog.run_button.isEnabled()
    qtbot.waitUntil(lambda: not widget._batch_workspace_preview_workers, timeout=10_000)
    reviewed = dialog._preview_result
    assert reviewed is not None
    assert reviewed.total_items == 1
    assert dialog.tabs.currentIndex() == 1
    assert not dialog._checking_plan
    assert widget._interactive_collection_batch_items == ()
    assert not outputs.exists()

    dialog._recheck_selected_items()
    qtbot.waitUntil(lambda: not widget._batch_workspace_preview_workers, timeout=10_000)
    assert dialog._preview_result is reviewed
    assert "unchanged" in dialog.preview_status.text()
    np.save(inputs / "field.npy", np.zeros((4, 5), dtype=np.uint16))
    dialog._recheck_selected_items()
    qtbot.waitUntil(lambda: not widget._batch_workspace_preview_workers, timeout=10_000)
    assert dialog._preview_result is None
    assert not dialog.run_button.isEnabled()
    assert not outputs.exists()
