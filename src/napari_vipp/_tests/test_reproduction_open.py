"""Recorded package -> explicit opening choice -> full GUI input checking."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
from qtpy.QtWidgets import QDialog

from napari_vipp._tests.test_batch import _batch_config, _batch_workflow, _write_arrays
from napari_vipp._tests.test_widget import VippWidget, _Viewer
from napari_vipp.core.batch import (
    BATCH_MANIFEST_FILENAME,
    BatchConfig,
    run_batch,
    scientific_workflow_hash,
)
from napari_vipp.core.io import inspect_image_source
from napari_vipp.core.reproducibility import build_reproducibility_package
from napari_vipp.core.source_identity import capture_local_source_bundle
from napari_vipp.core.source_resolution import resolve_source_item
from napari_vipp.core.workflow import serialize_workflow
from napari_vipp.ui.reproduction import (
    ReproductionChoiceDialog,
    ReproductionOpenCancelled,
)


@pytest.fixture(scope="module", params=["napari layer", "file path"])
def shared_run(tmp_path_factory, request):
    root = tmp_path_factory.mktemp("reproduction-open")
    source = root / "originals"
    arrays = {f"sample-{i}": np.full((4, 5), i, dtype=np.uint8) for i in range(3)}
    _write_arrays(source, **arrays)
    workflow, outputs = _batch_workflow()
    if request.param == "file path":
        representative = source / "sample-0.npy"
        item = resolve_source_item(
            capture_local_source_bundle(representative),
            inspect_image_source(representative),
            series_index=0,
        )
        workflow["nodes"][0]["params"].update(
            source_mode="file path",
            file_path=str(representative),
            _vipp_source_item=item.to_dict(),
        )
    config = _batch_config(workflow, source, root / "original-results", outputs)
    run_batch(workflow, config, performance_history_path=root / "history.json")
    package = build_reproducibility_package(
        manifest_path=config.output_dir / BATCH_MANIFEST_FILENAME
    )
    path = root / "workflow.json"
    path.write_bytes(package.members["workflow.json"])
    assert package.report_data["reproduction"]["available"]
    exported = json.loads(path.read_text())
    reproduction = exported["batch_config"]["reproduction"]
    assert reproduction["mode"] == "awaiting-choice"
    assert "version_override" not in reproduction
    assert str(root) not in json.dumps(reproduction)
    if request.param == "file path":
        params = exported["nodes"][0]["params"]
        assert params["file_path"].startswith("relink/")
        assert "\\" not in params["file_path"]
        assert params["_vipp_source_item"]["container"]["uri"] == params["file_path"]
    changed = root / "changed-inputs"
    _write_arrays(changed, **{name: data + 10 for name, data in arrays.items()})
    return path, source, changed, package


def _choose(monkeypatch, *, new_data=False):
    def exec_choice(dialog):
        if new_data:
            dialog.new_data_radio.setChecked(True)
        assert dialog.open_button.isEnabled()
        return QDialog.Accepted

    monkeypatch.setattr(ReproductionChoiceDialog, "exec", exec_choice)


def test_dropped_recorded_workflow_uses_choice_and_opens_batch(
    qtbot, monkeypatch, shared_run
):
    from napari_vipp._tests.test_workflow_drop import _send_drop

    _choose(monkeypatch)
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    original = widget._workflow_tabs.current
    _send_drop(widget.graph_view.viewport(), shared_run[0])
    assert len(widget._workflow_tabs) == 1
    qtbot.waitUntil(lambda: len(widget._workflow_tabs) == 2)
    assert widget._workflow_tabs[0] is original
    assert widget._workflow_tabs.current.path == shared_run[0].resolve()
    dialog = widget._active_collection_batch_dialog
    assert dialog is not None
    assert dialog.isVisible()
    assert not dialog.run_button.isEnabled()
    exported = json.loads(shared_run[0].read_text())
    assert (
        scientific_workflow_hash(
            serialize_workflow(
                widget.pipeline, compute_request=widget._current_compute_request()
            )
        )
        == exported["batch_config"]["workflow"]["sha256"]
    )


def _open(qtbot, monkeypatch, shared_run, *, new_data=False):
    _choose(monkeypatch, new_data=new_data)
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    widget.load_workflow_file(shared_run[0])
    dialog = widget._active_collection_batch_dialog
    assert dialog is not None, widget._last_workflow_load_detail
    assert dialog.isVisible()
    assert not widget._batch_workspace_preview_workers
    assert dialog._preview_result is None
    assert not dialog.run_button.isEnabled()
    exported = json.loads(shared_run[0].read_text())
    assert (
        scientific_workflow_hash(
            serialize_workflow(
                widget.pipeline, compute_request=widget._current_compute_request()
            )
        )
        == exported["batch_config"]["workflow"]["sha256"]
    )
    assert widget.pipeline.nodes["input"].params == exported["nodes"][0]["params"]
    return widget, dialog


def test_cancel_does_not_create_tab_or_change_current_workflow(
    qtbot, monkeypatch, shared_run
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    before = widget._workflow_tabs.current
    count = len(widget._workflow_tabs)
    pipeline = widget.pipeline
    monkeypatch.setattr(ReproductionChoiceDialog, "exec", lambda self: QDialog.Rejected)
    with pytest.raises(ReproductionOpenCancelled):
        widget.load_workflow_file(shared_run[0])
    assert len(widget._workflow_tabs) == count
    assert widget._workflow_tabs.current is before
    assert widget.pipeline is pipeline


@pytest.mark.parametrize(
    "new_data,changed", [(False, False), (False, True), (True, True)]
)
def test_gui_check_obeys_explicit_mode(
    qtbot, monkeypatch, tmp_path, shared_run, new_data, changed
):
    widget, dialog = _open(qtbot, monkeypatch, shared_run, new_data=new_data)
    output = tmp_path / "new-results"
    dialog.input_edit.setText(str(shared_run[2 if changed else 1]))
    dialog.output_edit.setText(str(output))
    assert dialog._check_batch()
    qtbot.waitUntil(lambda: not dialog._checking_plan, timeout=10_000)
    plan = dialog._preview_result
    assert plan is not None, dialog.preview_status.text()
    if new_data:
        assert plan.config.reproduction is None
        assert "reproduction" not in dialog.values()
        saved = widget._workflow_document_with_batch_config(tmp_path / "saved.json", {})
        assert "reproduction" not in saved["batch_config"]
        assert dialog.run_button.isEnabled()
    else:
        assert plan.reproduction is not None
        assert plan.reproduction.can_run is (not changed)
        assert plan.reproduction.mismatch_count == (3 if changed else 0)
        assert dialog.run_button.isEnabled() is (not changed)
        assert len(plan.reproduction.rows) == 3
        if changed:
            assert {row.status for row in plan.reproduction.rows} == {"changed"}
            assert "3" in dialog.reproduction_status_label.text()
    assert not output.exists()


def test_open_does_not_automatically_calculate_or_discover_sources(
    qtbot, monkeypatch, shared_run
):
    _choose(monkeypatch)
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    calls = []
    monkeypatch.setattr(widget, "run_pipeline", lambda *a, **kw: calls.append("run"))
    monkeypatch.setattr(
        widget,
        "_start_attached_batch_workspace_preview",
        lambda *a, **kw: calls.append("discover"),
    )
    widget.load_workflow_file(shared_run[0])
    assert calls == []


def test_source_inspector_render_never_reapplies_authored_source_parameters(
    qtbot, monkeypatch, shared_run
):
    widget, _dialog = _open(qtbot, monkeypatch, shared_run)
    before = serialize_workflow(widget.pipeline)

    def reject_presentation_write(*_args, **_kwargs):
        raise AssertionError(
            "Rendering the source inspector must not normalize parameters."
        )

    # This guards the boundary on every OS, including platforms where replacing
    # POSIX separators with native separators would not change the path bytes.
    monkeypatch.setattr(widget, "_apply_image_source_params", reject_presentation_write)
    widget._render_parameters("input")
    assert serialize_workflow(widget.pipeline) == before


def test_recipe_export_does_not_keep_an_old_reproduction_reference(shared_run):
    original = json.loads(shared_run[0].read_text())
    recipe = build_reproducibility_package(original)
    saved = json.loads(recipe.members["workflow.json"])
    assert "reproduction" not in saved["batch_config"]
    assert "reproduction" not in json.loads(recipe.members["batch-config.json"])
    assert "reproduction" in original["batch_config"]


def test_saved_version_acknowledgement_is_not_reused_on_open(
    qtbot, monkeypatch, shared_run
):
    from napari_vipp.core.reproduction import (
        ReproductionRequest,
        ReproductionVersionOverride,
        current_vipp_version,
    )

    document = json.loads(shared_run[0].read_text())
    request = ReproductionRequest.from_dict(document["batch_config"]["reproduction"])
    request = replace(
        request,
        reference=replace(request.reference, recorded_vipp_version="0.14.0"),
        mode="reproduce",
        version_override=ReproductionVersionOverride("0.14.0", current_vipp_version()),
    )
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)

    def inspect(dialog):
        assert not dialog.override_checkbox.isChecked()
        assert not dialog.open_button.isEnabled()
        return QDialog.Rejected

    monkeypatch.setattr(ReproductionChoiceDialog, "exec", inspect)
    with pytest.raises(ReproductionOpenCancelled):
        widget._choose_reproduction_request(request)


def test_workflow_tabs_keep_independent_reproduction_choices(
    qtbot, monkeypatch, shared_run
):
    widget, reproducing = _open(qtbot, monkeypatch, shared_run)
    original_tab = widget._workflow_tabs.current
    original_request = reproducing.values()["reproduction"]
    _choose(monkeypatch, new_data=True)
    widget.load_workflow_file(shared_run[0])
    reused = widget._active_collection_batch_dialog
    assert reused is not reproducing
    assert "reproduction" not in reused.values()
    original_index = widget._workflow_tabs.index_of(original_tab.session_id)
    assert widget._activate_workflow_tab(original_index, check_safety=False)
    assert widget._active_collection_batch_dialog is reproducing
    assert reproducing.values()["reproduction"] == original_request


def test_accepted_version_difference_is_reported_and_fresh_export_resets_choice(
    tmp_path, shared_run
):
    from napari_vipp.core.reproduction import (
        ReproductionVersionOverride,
        current_vipp_version,
    )

    workflow = json.loads(shared_run[0].read_text())
    config = BatchConfig.from_dict(workflow["batch_config"])
    request = replace(
        config.reproduction,
        reference=replace(
            config.reproduction.reference, recorded_vipp_version="0.14.0"
        ),
        mode="reproduce",
        version_override=ReproductionVersionOverride("0.14.0", current_vipp_version()),
    )
    config = replace(
        config,
        reproduction=request,
        sources=(replace(config.sources[0], input_dir=shared_run[1]),),
        output_dir=tmp_path / "reproduced",
    )
    run_batch(workflow, config, performance_history_path=tmp_path / "history.json")
    package = build_reproducibility_package(
        manifest_path=config.output_dir / BATCH_MANIFEST_FILENAME
    )
    assert package.report_data["reproduction_check"]["version_override_used"]
    assert "author explicitly accepted" in package.members["report.html"].decode()
    fresh = json.loads(package.members["batch-config.json"])["reproduction"]
    assert fresh["mode"] == "awaiting-choice"
    assert "version_override" not in fresh
    assert fresh["reference"]["recorded_vipp_version"] == current_vipp_version()
    assert fresh["reference"]["original_run_id"] != request.reference.original_run_id
