from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QDialog,
    QFormLayout,
    QScrollArea,
    QSizePolicy,
)

from napari_vipp.core.batch import (
    DEFAULT_BATCH_SOURCE_PATTERN,
    BatchAxisSuggestion,
    BatchConfig,
    BatchItemPlan,
    BatchOutputConfig,
    BatchScientificPreflightError,
    BatchSourceConfig,
    BatchStatus,
)
from napari_vipp.core.batch_parameters import (
    BatchParameterOverride,
    BatchSourceParameterOverrides,
)
from napari_vipp.core.metadata import AxisDeclaration
from napari_vipp.ui import recent_paths
from napari_vipp.ui.batch import (
    AxisInterpretationControl,
    BatchPreviewResult,
    BatchPreviewRow,
    CollectionBatchActions,
    CollectionBatchDialog,
)


def _preview_result(tmp_path, *, count: int = 3) -> BatchPreviewResult:
    config = BatchConfig(
        workflow_file=tmp_path / "workflow.json",
        workflow_sha256="a" * 64,
        output_dir=tmp_path / "outputs",
        sources=(
            BatchSourceConfig(
                node_id="input",
                title="Image Source",
                input_dir=tmp_path / "inputs",
                pattern="*.npy",
            ),
        ),
        outputs=(
            BatchOutputConfig(
                node_id="output",
                node_title="Batch Output",
                tag="result",
                kind="image",
                format="npy",
                subfolder="",
                filename_template="{source_stem}__{tag}",
            ),
        ),
    )
    items = tuple(
        BatchItemPlan(
            index=index,
            batch_id=f"{index:04d}_field-{index}",
            primary_source=tmp_path / "inputs" / f"field-{index}.npy",
            source_paths={"input": tmp_path / "inputs" / f"field-{index}.npy"},
            outputs=(),
        )
        for index in range(1, count + 1)
    )
    rows = tuple(
        BatchPreviewRow(
            batch_index=item.index,
            batch_id=item.batch_id,
            sources=dict(item.source_paths),
            outputs=[tmp_path / "outputs" / f"field-{item.index}.npy"],
            output_statuses=("new",),
        )
        for item in items
    )
    return BatchPreviewResult(
        rows=rows,
        total_items=len(items),
        collision_count=0,
        explicit_outputs=True,
        items=items,
        config=config,
    )


def _actions(result, previewed: list[int]) -> CollectionBatchActions:
    return CollectionBatchActions(
        preview_batch=lambda _values, _limit: result,
        choose_demo=lambda _parent: None,
        source_rows=lambda: [],
        load_config=lambda _path: result.config,
        save_config=lambda _path, _values: (),
        preview_item=previewed.append,
    )


def test_saved_workspace_discovery_has_distinct_indeterminate_progress(
    qtbot,
    tmp_path,
):
    result = _preview_result(tmp_path)
    dialog = CollectionBatchDialog(actions=_actions(result, []))
    qtbot.addWidget(dialog)

    assert not dialog.batch_activity_strip.isHidden()
    assert "not checked" in dialog.batch_activity_status.text().lower()
    assert dialog.source_detection_progress.isHidden()

    dialog.begin_saved_workspace_discovery()

    assert not dialog.source_detection_progress.isHidden()
    assert dialog.source_detection_progress.minimum() == 0
    assert dialog.source_detection_progress.maximum() == 0
    assert "detecting batch items" in dialog.batch_activity_status.text().lower()
    assert dialog.source_detection_progress.format() == "Checking"
    assert "main vipp progress bar" in (
        dialog.source_detection_progress.toolTip().lower()
    )
    assert "no representative image" in dialog.preview_status.text().lower()
    assert not dialog.preview_button.isEnabled()
    assert not dialog.run_button.isEnabled()
    assert dialog.parameter_override_group.isHidden()

    dialog.show_saved_workspace_discovery_success(
        item_count=3,
        override_count=0,
    )

    assert dialog.source_detection_progress.isHidden()
    assert dialog.source_detection_progress.minimum() == 0
    assert dialog.source_detection_progress.maximum() == 1
    assert dialog.preview_button.isEnabled()
    assert dialog.batch_activity_status.text() == "Ready · 3 batch items."
    assert "restored and detected 3" in dialog.preview_status.text().lower()


def test_saved_workspace_discovery_progress_ends_on_failure_and_invalidation(
    qtbot,
    tmp_path,
):
    result = _preview_result(tmp_path)
    dialog = CollectionBatchDialog(actions=_actions(result, []))
    qtbot.addWidget(dialog)

    dialog.begin_saved_workspace_discovery()
    dialog.show_saved_workspace_discovery_failure(
        "The source folder is missing.",
        technical_detail="missing path",
    )

    assert dialog.source_detection_progress.isHidden()
    assert "needs attention" in dialog.batch_activity_status.text().lower()
    assert dialog.preview_button.isEnabled()
    assert not dialog.run_button.isEnabled()
    assert dialog.parameter_override_group.isHidden()
    assert "source folder is missing" in dialog.preview_status.text().lower()
    assert dialog.preview_status.toolTip() == "missing path"

    dialog.begin_saved_workspace_discovery()
    dialog._invalidate_preview_plan()

    assert dialog.source_detection_progress.isHidden()
    assert dialog.preview_button.isEnabled()
    assert "not checked" in dialog.batch_activity_status.text().lower()
    assert "settings changed" in dialog.preview_status.text().lower()


def test_fixed_activity_strip_summarizes_preview_and_run_lifecycle(qtbot, tmp_path):
    result = _preview_result(tmp_path)
    planning_states: list[str] = []
    dialog: CollectionBatchDialog

    def preview(_values, _limit):
        planning_states.append(dialog.batch_activity_status.text())
        return result

    actions = replace(_actions(result, []), preview_batch=preview)
    dialog = CollectionBatchDialog(actions=actions)
    qtbot.addWidget(dialog)

    assert dialog._preview_batch()
    assert planning_states == ["Checking batch inputs, outputs, and parameters…"]
    assert dialog.batch_activity_status.text() == (
        "Ready · item 1 of 3 shown in main VIPP."
    )
    assert dialog.source_detection_progress.isHidden()

    emitted_states: list[str] = []
    dialog.runRequested.connect(
        lambda _values: emitted_states.append(dialog.batch_activity_status.text())
    )
    dialog._request_run()
    assert emitted_states == ["Checking current inputs, destinations, and parameters…"]
    assert dialog.source_detection_progress.maximum() == 0

    dialog.begin_run(3)
    assert dialog.source_detection_progress.maximum() == 3
    assert dialog.source_detection_progress.value() == 0
    dialog.update_run_progress(2, 3, result.items[1].batch_id, "running")
    assert dialog.batch_activity_status.text() == "Running batch · item 2 of 3."
    assert dialog.source_detection_progress.value() == 1

    dialog._request_cancel()
    assert "cancelling safely" in dialog.batch_activity_status.text().lower()
    assert dialog.source_detection_progress.maximum() == 3
    assert dialog.source_detection_progress.value() == 1


def test_fixed_activity_strip_ends_when_preview_returns_no_plan(qtbot, tmp_path):
    result = _preview_result(tmp_path)
    actions = replace(_actions(result, []), preview_batch=lambda *_args: None)
    dialog = CollectionBatchDialog(actions=actions)
    qtbot.addWidget(dialog)

    assert not dialog._preview_batch()

    assert dialog.source_detection_progress.isHidden()
    assert "needs attention" in dialog.batch_activity_status.text().lower()
    assert "no batch plan" in dialog.preview_status.text().lower()


def test_saved_workspace_discovery_cancel_and_override_aliases(qtbot, tmp_path):
    result = _preview_result(tmp_path)
    dialog = CollectionBatchDialog(actions=_actions(result, []))
    qtbot.addWidget(dialog)

    dialog.begin_saved_workspace_discovery()

    assert not dialog.source_detection_progress.isHidden()
    assert dialog.parameter_override_group.isHidden()

    dialog.cancel_saved_workspace_discovery("The Batch settings changed.")

    assert dialog.source_detection_progress.isHidden()
    assert dialog.preview_button.isEnabled()
    assert dialog.run_button.isEnabled()
    assert dialog.preview_status.text() == "The Batch settings changed."

    dialog.begin_saved_override_verification(1)

    assert not dialog.parameter_override_group.isHidden()
    assert "1 saved source override entry" in (
        dialog.parameter_override_editor.status_label.text().lower()
    )

    dialog.show_saved_override_verification_success(
        item_count=3,
        override_count=1,
    )

    assert dialog.source_detection_progress.isHidden()
    assert "1 saved per-sample override entry" in dialog.preview_status.text()


def test_saved_workspace_discovery_cancel_keeps_overrides_quarantined(
    qtbot,
    tmp_path,
):
    result = _preview_result(tmp_path)
    dialog = CollectionBatchDialog(actions=_actions(result, []))
    qtbot.addWidget(dialog)
    dialog._pending_parameter_overrides = (
        BatchSourceParameterOverrides(
            "a" * 64,
            (BatchParameterOverride("threshold", "threshold", 5_000.0),),
        ),
    )

    dialog.begin_saved_workspace_discovery(override_count=1)
    dialog.cancel_saved_workspace_discovery("The Batch settings changed.")

    assert dialog.source_detection_progress.isHidden()
    assert not dialog.run_button.isEnabled()
    assert not dialog.parameter_override_group.isHidden()
    assert "batch settings changed" in (
        dialog.parameter_override_editor.status_label.text().lower()
    )


def test_batch_source_row_defaults_to_supported_image_discovery(qtbot):
    dialog = CollectionBatchDialog()
    qtbot.addWidget(dialog)

    pattern_edit = dialog._source_rows[0]["pattern"]

    assert pattern_edit.text() == DEFAULT_BATCH_SOURCE_PATTERN
    tooltip = pattern_edit.toolTip()
    assert "supported image" in tooltip
    assert "top-level OME-Zarr" in tooltip


def test_batch_output_defaults_to_primary_source_output_and_tracks_it(
    qtbot,
    tmp_path,
):
    result = _preview_result(tmp_path)
    dialog = CollectionBatchDialog(
        source_nodes=[
            {"node_id": "input", "title": "Primary"},
            {"node_id": "reference", "title": "Reference"},
        ],
        actions=_actions(result, []),
    )
    qtbot.addWidget(dialog)

    reference = tmp_path / "reference"
    dialog._source_rows[1]["folder"].setText(str(reference))
    assert dialog.output_edit.text() == str(reference / "output")

    primary = tmp_path / "primary"
    dialog._source_rows[0]["folder"].setText(str(primary))

    assert dialog.output_edit.text() == str(primary / "output")
    assert dialog.output_edit.property("suggestedDefault") is True
    assert "#f59e0b" in dialog.output_edit.styleSheet()
    assert "Suggested from the first bound batch source" in dialog.output_edit.toolTip()

    dialog._source_rows[1]["folder"].setText(str(tmp_path / "new-reference"))
    assert dialog.output_edit.text() == str(primary / "output")

    updated_primary = tmp_path / "updated-primary"
    dialog._source_rows[0]["folder"].setText(str(updated_primary))
    assert dialog.output_edit.text() == str(updated_primary / "output")


def test_batch_output_path_expands_with_dialog(qtbot, tmp_path):
    result = _preview_result(tmp_path)
    dialog = CollectionBatchDialog(actions=_actions(result, []))
    qtbot.addWidget(dialog)
    dialog.resize(640, 720)
    dialog.show()
    qtbot.waitUntil(dialog.isVisible)
    qtbot.waitUntil(lambda: dialog.output_edit.width() > 0)

    assert dialog.output_edit.sizePolicy().horizontalPolicy() == (QSizePolicy.Expanding)
    output_row = dialog.output_edit.parentWidget()
    assert output_row.sizePolicy().horizontalPolicy() == QSizePolicy.Expanding
    content_layout = dialog.content_widget.layout()
    form = next(
        item.layout()
        for index in range(content_layout.count())
        if isinstance((item := content_layout.itemAt(index)).layout(), QFormLayout)
    )
    assert isinstance(form, QFormLayout)
    assert form.fieldGrowthPolicy() == QFormLayout.AllNonFixedFieldsGrow

    compact_width = dialog.output_edit.width()
    dialog.resize(840, 720)
    qtbot.waitUntil(lambda: dialog.output_edit.width() >= compact_width + 150)

    assert dialog.output_edit.width() >= int(dialog.input_edit.width() * 0.65)


def test_batch_input_folder_reuses_recent_selection_on_empty_fields(
    qtbot,
    monkeypatch,
    tmp_path,
):
    result = _preview_result(tmp_path)
    selected = tmp_path / "recent-input"
    selected.mkdir()
    starts = []
    selections = iter((str(selected), ""))

    def choose_directory(_parent, _title, start):
        starts.append(start)
        return next(selections)

    monkeypatch.setattr(
        "napari_vipp.ui.batch.QFileDialog.getExistingDirectory",
        choose_directory,
    )
    first = CollectionBatchDialog(actions=_actions(result, []))
    qtbot.addWidget(first)
    first._browse_source_input(first.input_edit)
    second = CollectionBatchDialog(actions=_actions(result, []))
    qtbot.addWidget(second)
    second._browse_source_input(second.input_edit)

    assert starts == ["", str(selected.resolve())]
    assert recent_paths.recent_directory(recent_paths.INPUT_DIRECTORY) == str(
        selected.resolve()
    )


def test_batch_output_suggestion_stops_tracking_after_user_acknowledges(
    qtbot,
    tmp_path,
):
    result = _preview_result(tmp_path)
    dialog = CollectionBatchDialog(actions=_actions(result, []))
    qtbot.addWidget(dialog)
    dialog.input_edit.setText(str(tmp_path / "inputs"))
    suggested = dialog.output_edit.text()
    dialog.show()
    qtbot.waitUntil(dialog.isVisible)

    qtbot.mouseClick(dialog.output_edit, Qt.LeftButton)

    assert dialog.output_edit.property("suggestedDefault") is False
    assert dialog.output_edit.styleSheet() == ""
    dialog.input_edit.setText(str(tmp_path / "different-inputs"))
    assert dialog.output_edit.text() == suggested


def test_clicking_empty_batch_output_prevents_a_later_suggestion(qtbot, tmp_path):
    result = _preview_result(tmp_path)
    dialog = CollectionBatchDialog(actions=_actions(result, []))
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(dialog.isVisible)

    qtbot.mouseClick(dialog.output_edit, Qt.LeftButton)
    dialog.input_edit.setText(str(tmp_path / "inputs"))

    assert dialog.output_edit.text() == ""
    assert dialog.output_edit.property("suggestedDefault") is False


def test_tab_focus_acknowledges_batch_output_suggestion(qtbot, tmp_path):
    result = _preview_result(tmp_path)
    dialog = CollectionBatchDialog(actions=_actions(result, []))
    qtbot.addWidget(dialog)
    dialog.input_edit.setText(str(tmp_path / "inputs"))
    suggested = dialog.output_edit.text()
    dialog.show()
    qtbot.waitUntil(dialog.isVisible)
    assert dialog.output_edit.property("suggestedDefault") is True
    dialog.pattern_edit.setFocus(Qt.OtherFocusReason)
    qtbot.waitUntil(dialog.pattern_edit.hasFocus)

    dialog.output_edit.setFocus(Qt.TabFocusReason)
    qtbot.waitUntil(dialog.output_edit.hasFocus)

    assert dialog.output_edit.property("suggestedDefault") is False
    dialog.input_edit.setText(str(tmp_path / "different-inputs"))
    assert dialog.output_edit.text() == suggested


def test_editing_batch_output_replaces_and_acknowledges_suggestion(qtbot, tmp_path):
    result = _preview_result(tmp_path)
    dialog = CollectionBatchDialog(actions=_actions(result, []))
    qtbot.addWidget(dialog)
    dialog.input_edit.setText(str(tmp_path / "inputs"))
    custom_output = tmp_path / "custom-output"

    dialog.output_edit.setText(str(custom_output))

    assert dialog.output_edit.property("suggestedDefault") is False
    dialog.input_edit.setText(str(tmp_path / "different-inputs"))
    assert dialog.output_edit.text() == str(custom_output)


def test_batch_output_folder_choice_acknowledges_unchanged_suggestion(
    qtbot,
    monkeypatch,
    tmp_path,
):
    result = _preview_result(tmp_path)
    dialog = CollectionBatchDialog(actions=_actions(result, []))
    qtbot.addWidget(dialog)
    dialog.input_edit.setText(str(tmp_path / "inputs"))
    suggested = dialog.output_edit.text()
    monkeypatch.setattr(
        "napari_vipp.ui.batch.QFileDialog.getExistingDirectory",
        lambda *_args, **_kwargs: suggested,
    )

    dialog._browse_output()

    assert dialog.output_edit.text() == suggested
    assert dialog.output_edit.property("suggestedDefault") is False
    dialog.input_edit.setText(str(tmp_path / "different-inputs"))
    assert dialog.output_edit.text() == suggested


def test_loaded_batch_output_is_custom_and_does_not_follow_source(qtbot, tmp_path):
    result = _preview_result(tmp_path)
    dialog = CollectionBatchDialog(actions=_actions(result, []))
    qtbot.addWidget(dialog)
    dialog.input_edit.setText(str(tmp_path / "before-load"))

    dialog._apply_config(result.config)

    configured_output = str(result.config.resolve_path(result.config.output_dir))
    assert dialog.output_edit.text() == configured_output
    assert dialog.output_edit.property("suggestedDefault") is False
    dialog.input_edit.setText(str(tmp_path / "after-load"))
    assert dialog.output_edit.text() == configured_output


def test_batch_source_axis_declaration_roundtrips_through_dialog_values(
    qtbot,
    tmp_path,
):
    result = _preview_result(tmp_path)
    declared_source = replace(
        result.config.sources[0],
        axis_declaration=AxisDeclaration("QYX", "ZYX"),
    )
    config = replace(result.config, sources=(declared_source,))
    dialog = CollectionBatchDialog(actions=_actions(result, []))
    qtbot.addWidget(dialog)

    dialog._apply_config(config)

    control = dialog._source_rows[0]["axis_declaration"]
    assert control.mode_combo.currentData() == "z_stack"
    assert "Z stack" in control.mode_combo.currentText()
    assert control.text() == "QYX -> ZYX"
    assert dialog.values()["source_bindings"][0]["axis_declaration"] == ("QYX -> ZYX")
    assert "does not transpose pixels" in control.toolTip()

    dialog._apply_config(result.config)

    assert control.mode_combo.currentData() == "file"
    assert "file" in control.mode_combo.currentText().lower()
    assert control.text() == ""
    assert dialog.values()["source_bindings"][0]["axis_declaration"] == ""


def test_batch_source_row_inherits_workflow_axis_choice_and_allows_blank_opt_out(
    qtbot,
):
    dialog = CollectionBatchDialog(
        source_nodes=[
            {
                "node_id": "input",
                "title": "Image Source",
                "binding_mode": "collection",
                "axis_declaration": "QYX -> ZYX",
            }
        ]
    )
    qtbot.addWidget(dialog)
    control = dialog._source_rows[0]["axis_declaration"]

    assert control.mode_combo.currentData() == AxisInterpretationControl.Z_STACK
    assert dialog.values()["source_bindings"][0]["axis_declaration"] == ("QYX -> ZYX")

    file_index = control.mode_combo.findData(AxisInterpretationControl.FILE_METADATA)
    control.mode_combo.setCurrentIndex(file_index)

    assert dialog.values()["source_bindings"][0]["axis_declaration"] == ""


def test_advanced_axis_mapping_keeps_partial_text_as_uncommitted_draft(qtbot):
    control = AxisInterpretationControl(
        allow_automatic=False,
        save_target="workflow",
    )
    qtbot.addWidget(control)
    changes: list[str] = []
    control.textChanged.connect(changes.append)
    custom_index = control.mode_combo.findData(AxisInterpretationControl.CUSTOM)
    control.mode_combo.setCurrentIndex(custom_index)

    for partial in ("Z", "Z-", "Z ->", "QYX ->"):
        control.advanced_edit.setText(partial)
        assert control.text() == ""
    assert changes == []
    assert "nothing has been applied" in control.notice_label.text().casefold()

    control.advanced_edit.setText("QYX -> ZYX")
    assert control.text() == ""
    assert changes == []
    assert "ready to apply" in control.notice_label.text().casefold()

    control.advanced_edit.editingFinished.emit()

    assert control.text() == "QYX -> ZYX"
    assert changes == ["QYX -> ZYX"]

    control.advanced_edit.setText("ZY")
    control.advanced_edit.editingFinished.emit()

    assert control.text() == "QYX -> ZYX"
    assert changes == ["QYX -> ZYX"]
    assert "not applied" in control.notice_label.text().casefold()


def test_loaded_blank_axis_choice_is_strict_file_metadata_opt_out(qtbot, tmp_path):
    result = _preview_result(tmp_path)
    dialog = CollectionBatchDialog(actions=_actions(result, []))
    qtbot.addWidget(dialog)

    dialog._apply_config(result.config)

    control = dialog._source_rows[0]["axis_declaration"]
    assert control.mode_combo.currentData() == AxisInterpretationControl.FILE_METADATA
    assert control.text() == ""
    assert not control.apply_z_stack_suggestion(_qyx_z_stack_suggestion())


@pytest.mark.parametrize("changed_field", ["folder", "pattern"])
def test_source_change_resets_auto_axis_choice_but_retains_manual_z_stack(
    qtbot,
    tmp_path,
    changed_field,
):
    result = _preview_result(tmp_path)
    dialog = CollectionBatchDialog(actions=_actions(result, []))
    qtbot.addWidget(dialog)
    row = dialog._source_rows[0]
    control = row["axis_declaration"]
    suggestion = _qyx_z_stack_suggestion()

    assert control.mode_combo.currentData() == AxisInterpretationControl.AUTOMATIC
    assert control.apply_z_stack_suggestion(suggestion)
    assert control.mode_combo.currentData() == AxisInterpretationControl.Z_STACK

    if changed_field == "folder":
        row["folder"].setText(str(tmp_path / "replacement"))
    else:
        row["pattern"].setText("*.replacement.tif")

    assert control.mode_combo.currentData() == AxisInterpretationControl.AUTOMATIC
    assert control.text() == ""

    z_stack_index = control.mode_combo.findData(AxisInterpretationControl.Z_STACK)
    control.mode_combo.setCurrentIndex(z_stack_index)
    assert control.mode_combo.currentData() == AxisInterpretationControl.Z_STACK

    if changed_field == "folder":
        row["folder"].setText(str(tmp_path / "manual-replacement"))
    else:
        row["pattern"].setText("*.manual.tif")

    assert control.mode_combo.currentData() == AxisInterpretationControl.Z_STACK
    assert control.text() == AxisInterpretationControl.Z_STACK_DECLARATION


def test_loading_config_clears_axis_choice_from_omitted_source_row(qtbot, tmp_path):
    result = _preview_result(tmp_path)
    dialog = CollectionBatchDialog(
        source_nodes=[
            {"node_id": "input", "title": "Primary"},
            {"node_id": "omitted", "title": "Omitted"},
        ],
        actions=_actions(result, []),
    )
    qtbot.addWidget(dialog)
    omitted = dialog._source_rows[1]
    omitted["folder"].setText(str(tmp_path / "old-source"))
    omitted["axis_declaration"].setText("QYX -> ZYX")
    assert omitted["axis_declaration"].text() == "QYX -> ZYX"

    dialog._apply_config(result.config)

    control = omitted["axis_declaration"]
    assert omitted["folder"].text() == ""
    assert control.mode_combo.currentData() == AxisInterpretationControl.FILE_METADATA
    assert control.text() == ""


def test_scientific_preflight_error_disables_run_until_settings_change(
    qtbot,
    tmp_path,
):
    result = _preview_result(tmp_path)
    dialog = CollectionBatchDialog(actions=_actions(result, []))
    qtbot.addWidget(dialog)

    dialog.show_preflight_error("QYX needs a reviewed declaration.")

    assert not dialog.run_button.isEnabled()
    assert "QYX" in dialog.run_result_label.text()
    dialog._source_rows[0]["axis_declaration"].setText("QYX -> ZYX")
    assert dialog.run_button.isEnabled()


def _qyx_z_stack_suggestion() -> BatchAxisSuggestion:
    return BatchAxisSuggestion(
        source_node_id="input",
        source_title="Image Source",
        declaration=AxisDeclaration("QYX", "ZYX"),
    )


def _qyx_preflight_error() -> BatchScientificPreflightError:
    return BatchScientificPreflightError(
        "Batch scientific preflight failed before item processing, output "
        "creation, or CPU/GPU device setup. Representative source axes: "
        "Image Source: raw QYX, effective QYX (no declaration). Subtract "
        "Background, Gaussian Blur 3D, and Reorder Axes cannot continue.",
        user_message=(
            "This source has an unknown leading Q axis. Because the workflow "
            "requires 3D processing, VIPP can treat Q as depth Z."
        ),
        axis_suggestion=_qyx_z_stack_suggestion(),
    )


def test_qyx_preview_applies_visible_z_stack_default_without_typing(qtbot, tmp_path):
    raw_result = _preview_result(tmp_path)
    declared_source = replace(
        raw_result.config.sources[0],
        axis_declaration=AxisDeclaration("QYX", "ZYX"),
    )
    declared_result = replace(
        raw_result,
        config=replace(raw_result.config, sources=(declared_source,)),
    )
    calls: list[dict[str, object]] = []

    def preview(values, _limit):
        calls.append(values)
        declaration = values["source_bindings"][0]["axis_declaration"]
        if len(calls) == 1:
            assert declaration == ""
            raise _qyx_preflight_error()
        assert declaration == "QYX -> ZYX"
        return declared_result

    actions = replace(_actions(raw_result, []), preview_batch=preview)
    dialog = CollectionBatchDialog(actions=actions)
    qtbot.addWidget(dialog)

    assert dialog._preview_batch()

    assert len(calls) == 2
    control = dialog._source_rows[0]["axis_declaration"]
    assert control.mode_combo.currentData() == "z_stack"
    assert "Z stack" in control.mode_combo.currentText()
    assert control.text() == "QYX -> ZYX"
    assert dialog.values()["source_bindings"][0]["axis_declaration"] == ("QYX -> ZYX")
    assert not control.notice_label.isHidden()
    notice = control.notice_label.text().lower()
    assert "qyx -> zyx" in notice
    assert "automatic choice" in notice
    assert "pixel" in notice
    assert "unchanged" in notice or "not" in notice


def test_explicit_file_metadata_opt_out_is_not_auto_overridden(qtbot, tmp_path):
    raw_result = _preview_result(tmp_path)
    declared_source = replace(
        raw_result.config.sources[0],
        axis_declaration=AxisDeclaration("QYX", "ZYX"),
    )
    declared_result = replace(
        raw_result,
        config=replace(raw_result.config, sources=(declared_source,)),
    )
    calls: list[dict[str, object]] = []

    def initial_preview(values, _limit):
        calls.append(values)
        if len(calls) == 1:
            raise _qyx_preflight_error()
        return declared_result

    actions = replace(_actions(raw_result, []), preview_batch=initial_preview)
    dialog = CollectionBatchDialog(actions=actions)
    qtbot.addWidget(dialog)
    assert dialog._preview_batch()

    control = dialog._source_rows[0]["axis_declaration"]
    file_index = control.mode_combo.findData("file")
    assert file_index >= 0
    control.mode_combo.setCurrentIndex(file_index)
    assert control.text() == ""
    assert dialog.values()["source_bindings"][0]["axis_declaration"] == ""

    retry_calls: list[dict[str, object]] = []

    def rejected_preview(values, _limit):
        retry_calls.append(values)
        raise _qyx_preflight_error()

    dialog._actions = replace(actions, preview_batch=rejected_preview)

    assert not dialog._preview_batch()
    assert len(retry_calls) == 1
    assert control.mode_combo.currentData() == "file"
    assert control.text() == ""


def test_qyx_preview_failure_shows_one_concise_issue(qtbot, tmp_path):
    result = _preview_result(tmp_path)
    error = _qyx_preflight_error()

    def fail_preview(_values, _limit):
        raise error

    actions = replace(_actions(result, []), preview_batch=fail_preview)
    dialog = CollectionBatchDialog(actions=actions)
    qtbot.addWidget(dialog)
    control = dialog._source_rows[0]["axis_declaration"]
    suggestion = error.axis_suggestion
    assert suggestion is not None
    assert control.apply_z_stack_suggestion(suggestion)
    file_index = control.mode_combo.findData("file")
    assert file_index >= 0
    control.mode_combo.setCurrentIndex(file_index)

    assert not dialog._preview_batch()

    message = dialog.preview_status.text()
    assert error.user_message in message
    assert message.count(error.user_message) == 1
    assert len(message) < 240
    assert "Representative source axes" not in message
    assert "CPU/GPU device setup" not in message
    assert "Subtract Background" not in message
    assert "Gaussian Blur 3D" not in message
    assert "Reorder Axes" not in message


def test_batch_dialog_run_request_does_not_accept_workspace(qtbot, tmp_path):
    result = _preview_result(tmp_path)
    dialog = CollectionBatchDialog(actions=_actions(result, []))
    qtbot.addWidget(dialog)
    requests = []
    dialog.runRequested.connect(requests.append)

    qtbot.mouseClick(dialog.run_button, Qt.LeftButton)

    assert len(requests) == 1
    assert requests[0]["continue_on_error"] is True
    assert dialog.result() != QDialog.DialogCode.Accepted
    assert dialog.close_button.text() == "Close"


def test_batch_preview_auto_loads_first_representative_and_selection_is_explicit(
    qtbot,
    tmp_path,
):
    result = _preview_result(tmp_path)
    previewed: list[int] = []
    dialog = CollectionBatchDialog(actions=_actions(result, previewed))
    qtbot.addWidget(dialog)

    assert dialog._preview_batch()

    assert previewed == [0]
    assert dialog.preview_table.columnCount() == 5
    assert dialog.preview_table.item(0, 3).text() == "new"
    assert dialog.preview_table.item(0, 4).text() == "Not run"
    assert "representative calculation" in dialog.graph_preview_status.text()
    assert dialog.select_preview_item(1)
    assert previewed == [0]

    qtbot.mouseClick(dialog.preview_item_button, Qt.LeftButton)
    assert previewed == [0, 1]

    dialog.preview_table.itemDoubleClicked.emit(dialog.preview_table.item(2, 1))
    assert previewed == [0, 1, 2]


def test_plan_only_result_does_not_calculate_a_graph_representative(
    qtbot,
    tmp_path,
):
    result = _preview_result(tmp_path)
    previewed: list[int] = []
    dialog = CollectionBatchDialog(actions=_actions(result, previewed))
    qtbot.addWidget(dialog)

    dialog.apply_preview_result(result, preview_representative=False)

    assert dialog._preview_result is result
    assert previewed == []
    assert dialog.preview_table.rowCount() == 3
    assert dialog.preview_status.text() == (
        "Ready: 3 batch item(s) checked. Nothing was saved."
    )
    assert "No representative was loaded" in dialog.graph_preview_status.text()
    assert dialog.preview_item_button.isEnabled()


def test_batch_preview_status_tracks_full_plan_item_beyond_table_limit(
    qtbot,
    tmp_path,
):
    full = _preview_result(tmp_path, count=26)
    limited = replace(full, rows=full.rows[:25])
    dialog = CollectionBatchDialog(actions=_actions(limited, []))
    qtbot.addWidget(dialog)
    assert dialog._preview_batch()

    assert dialog.preview_table.rowCount() == 25
    assert dialog.select_preview_item(25)
    assert not dialog.preview_table.selectionModel().selectedRows()
    assert "item 26 of 26" in dialog.graph_preview_status.text()

    dialog.begin_run(26)
    dialog.preview_table.selectRow(24)
    dialog.update_run_progress(26, 26, limited.items[25].batch_id, "running")
    assert not dialog.preview_table.selectionModel().selectedRows()
    assert "Item 26 of 26" in dialog.run_progress_label.text()


def test_failed_preview_emits_invalidation_and_clears_demo_identity(qtbot, tmp_path):
    result = _preview_result(tmp_path)

    def fail_preview(_values, _limit):
        raise ValueError("source disappeared")

    actions = replace(_actions(result, []), preview_batch=fail_preview)
    dialog = CollectionBatchDialog(actions=actions)
    qtbot.addWidget(dialog)
    dialog.set_demo_context(SimpleNamespace(root=tmp_path))
    invalidations: list[None] = []
    dialog.previewInvalidated.connect(lambda: invalidations.append(None))

    assert not dialog._preview_batch()
    assert invalidations == [None]
    assert dialog.run_button.text() == "Run batch"
    assert dialog.demo_guide_label.isHidden()


def test_completed_run_marks_preflight_historical_without_erasing_evidence(
    qtbot,
    tmp_path,
):
    result = _preview_result(tmp_path)
    dialog = CollectionBatchDialog(actions=_actions(result, []))
    qtbot.addWidget(dialog)
    assert dialog._preview_batch()
    dialog.begin_run(3)
    run_result = SimpleNamespace(
        manifest=SimpleNamespace(
            items=tuple(
                SimpleNamespace(index=index, status=BatchStatus.COMPLETED)
                for index in range(1, 4)
            )
        ),
        summary={"completed": 3, "partial": 0, "skipped": 0, "failed": 0},
        saved_paths=(),
        manifest_path=tmp_path / "manifest.json",
    )
    dialog.finish_run(run_result)

    dialog.mark_plan_historical_after_run()

    assert dialog._preview_result is None
    assert [dialog.preview_table.item(row, 4).text() for row in range(3)] == [
        "Completed",
        "Completed",
        "Completed",
    ]
    assert "historical" in dialog.preview_status.text().lower()
    assert "3 completed" in dialog.run_progress_label.text()


def test_batch_dialog_retains_determinate_progress_and_restores_controls(
    qtbot,
    tmp_path,
):
    result = _preview_result(tmp_path)
    dialog = CollectionBatchDialog(actions=_actions(result, []))
    qtbot.addWidget(dialog)
    assert dialog._preview_batch()
    assert dialog.preview_item_button.isEnabled()
    assert not dialog.workflow_checkbox.isEnabled()

    dialog.begin_run(3)

    assert not dialog.run_button.isEnabled()
    assert not dialog.preview_button.isEnabled()
    assert not dialog.source_group.isEnabled()
    assert dialog.close_button.isEnabled()
    assert not dialog.cancel_run_button.isHidden()
    assert dialog.cancel_run_button.isEnabled()
    assert dialog.operation_progress_bar.minimum() == 0
    assert dialog.operation_progress_bar.maximum() == 0
    assert dialog.preview_table.item(0, 3).text() == "new"
    assert dialog.preview_table.item(0, 4).text() == "Pending"

    dialog.update_run_progress(2, 3, result.items[1].batch_id, "running")
    assert dialog.run_progress_bar.maximum() == 3
    assert dialog.run_progress_bar.value() == 1
    assert dialog.preview_table.item(1, 4).text() == "Running"

    dialog.update_operation_progress(
        2,
        3,
        result.items[1].batch_id,
        "gaussian",
        "gaussian_blur",
        4,
        10,
        "Gaussian planes",
    )
    assert dialog.operation_progress_bar.maximum() == 10
    assert dialog.operation_progress_bar.value() == 4
    assert "gaussian_blur" in dialog.operation_progress_label.text()
    assert "Gaussian planes" in dialog.operation_progress_label.text()

    dialog.update_run_progress(2, 3, result.items[1].batch_id, "completed")
    assert dialog.run_progress_bar.value() == 2
    assert dialog.preview_table.item(1, 4).text() == "Completed"

    run_result = SimpleNamespace(
        manifest=SimpleNamespace(
            items=(
                SimpleNamespace(index=1, status=BatchStatus.COMPLETED),
                SimpleNamespace(index=2, status=BatchStatus.FAILED),
                SimpleNamespace(index=3, status=BatchStatus.SKIPPED),
            )
        ),
        summary={"completed": 1, "partial": 0, "skipped": 1, "failed": 1},
        saved_paths=(tmp_path / "outputs" / "field-1.npy",),
        manifest_path=tmp_path / "outputs" / "vipp_batch_manifest.json",
    )
    dialog.finish_run(run_result, "Ground truth passed.")

    assert dialog.run_progress_bar.value() == 3
    assert dialog.preview_table.item(0, 4).text() == "Completed"
    assert dialog.preview_table.item(1, 4).text() == "Failed"
    assert dialog.preview_table.item(2, 4).text() == "Skipped"
    assert dialog.preview_table.item(0, 3).text() == "new"
    assert "1 completed" in dialog.run_progress_label.text()
    assert dialog.batch_activity_status.text() == ("Completed with 1 issue · 3 items.")
    assert dialog.source_detection_progress.value() == 3
    assert "vipp_batch_manifest.json" in dialog.run_result_label.text()
    assert "Ground truth passed" in dialog.run_result_label.text()
    assert dialog.run_button.isEnabled()
    assert dialog.preview_button.isEnabled()
    assert dialog.source_group.isEnabled()
    assert dialog.preview_item_button.isEnabled()
    assert not dialog.workflow_checkbox.isEnabled()
    assert dialog.cancel_run_button.isHidden()

    assert dialog._preview_batch()
    assert dialog.run_group.isHidden()
    assert dialog.run_progress_bar.format() == "Not run"
    assert dialog.preview_table.item(0, 4).text() == "Not run"


def test_batch_dialog_cancel_is_single_shot_and_waits_for_safe_checkpoint(
    qtbot,
    tmp_path,
):
    result = _preview_result(tmp_path)
    dialog = CollectionBatchDialog(actions=_actions(result, []))
    qtbot.addWidget(dialog)
    requested: list[bool] = []
    dialog.cancelRequested.connect(lambda: requested.append(True))
    dialog.begin_run(3)

    qtbot.mouseClick(dialog.cancel_run_button, Qt.LeftButton)
    qtbot.mouseClick(dialog.cancel_run_button, Qt.LeftButton)

    assert requested == [True]
    assert not dialog.cancel_run_button.isEnabled()
    assert dialog.cancel_run_button.text() == "Cancelling..."
    assert "safe checkpoint" in dialog.operation_progress_label.text()
    assert "cancelling safely" in dialog.batch_activity_status.text().lower()


def test_batch_dialog_error_restores_exact_control_state(qtbot, tmp_path):
    result = _preview_result(tmp_path)
    dialog = CollectionBatchDialog(actions=_actions(result, []))
    qtbot.addWidget(dialog)
    assert dialog._preview_batch()
    dialog.save_config_button.setEnabled(False)

    dialog.begin_run(3)
    dialog.update_run_progress(1, 3, result.items[0].batch_id, "running")
    dialog.show_run_error("A source changed during execution.")

    assert dialog.preview_table.item(0, 4).text() == "Failed"
    assert dialog.run_progress_bar.format() == "Failed"
    assert dialog.batch_activity_status.text() == "Failed · item 1 of 3."
    assert "source changed" in dialog.run_result_label.text()
    assert dialog.run_button.isEnabled()
    assert not dialog.save_config_button.isEnabled()


def test_batch_dialog_scrolls_compact_content_and_keeps_footer_fixed(
    qtbot,
    tmp_path,
):
    result = _preview_result(tmp_path)
    previewed: list[int] = []
    source_nodes = [
        {
            "node_id": f"input_{index}_with_a_long_identifier",
            "title": f"Microscope source {index}",
            "binding_mode": "collection",
        }
        for index in range(1, 4)
    ]
    dialog = CollectionBatchDialog(
        source_nodes=source_nodes,
        actions=_actions(result, previewed),
    )
    qtbot.addWidget(dialog)
    dialog.set_demo_context(
        SimpleNamespace(root=tmp_path / "a" / "long" / "demo" / "working-copy")
    )
    assert dialog._preview_batch()
    requests: list[dict[str, object]] = []
    dialog.runRequested.connect(requests.append)

    assert isinstance(dialog.content_scroll, QScrollArea)
    assert dialog.content_scroll.widgetResizable()
    assert dialog.content_scroll.widget() is dialog.content_widget
    assert dialog.content_widget.isAncestorOf(dialog.source_group)
    assert dialog.content_widget.isAncestorOf(dialog.preview_table)
    assert not dialog.content_widget.isAncestorOf(dialog.config_row)
    assert dialog.config_row.isAncestorOf(dialog.load_config_button)
    assert dialog.config_row.isAncestorOf(dialog.batch_activity_strip)
    assert not dialog.content_widget.isAncestorOf(dialog.batch_activity_strip)
    assert dialog.isAncestorOf(dialog.batch_activity_strip)
    assert not dialog.content_widget.isAncestorOf(dialog.button_box)
    assert dialog.content_scroll.verticalScrollBarPolicy() == Qt.ScrollBarAsNeeded

    dialog.resize(640, dialog.minimumHeight())
    dialog.show()
    qtbot.waitUntil(dialog.isVisible)
    qtbot.waitUntil(lambda: dialog.content_scroll.verticalScrollBar().maximum() > 0)

    assert dialog.width() <= 640
    toolbar_layout = dialog.config_row.layout()
    assert (
        toolbar_layout.indexOf(dialog.load_config_button)
        < toolbar_layout.indexOf(dialog.save_config_button)
        < toolbar_layout.indexOf(dialog.demo_config_button)
        < toolbar_layout.indexOf(dialog.batch_activity_strip)
    )
    assert dialog.batch_activity_status.width() > 0
    toolbar_button_center = dialog.load_config_button.mapTo(
        dialog,
        dialog.load_config_button.rect().center(),
    )
    activity_center = dialog.batch_activity_strip.mapTo(
        dialog,
        dialog.batch_activity_strip.rect().center(),
    )
    assert abs(toolbar_button_center.y() - activity_center.y()) <= 2
    dialog.show_workspace_activity(
        "Detecting batch items and checking saved sources...",
        state="working",
        indeterminate=True,
        progress_text="Checking",
    )
    QApplication.processEvents()
    assert dialog.batch_activity_status.width() >= 96
    assert dialog.source_detection_progress.isVisibleTo(dialog.config_row)
    for widget in (
        dialog.batch_activity_status,
        dialog.source_detection_progress,
    ):
        widget_right = widget.mapTo(
            dialog.config_row,
            widget.rect().topRight(),
        ).x()
        assert widget_right <= dialog.config_row.rect().right()
    scroll_bar = dialog.content_scroll.verticalScrollBar()
    for position in (scroll_bar.minimum(), scroll_bar.maximum()):
        scroll_bar.setValue(position)
        assert dialog.batch_activity_strip.isVisibleTo(dialog)
        assert dialog.run_button.isVisibleTo(dialog)
        assert dialog.close_button.isVisibleTo(dialog)

    assert dialog.select_preview_item(1)
    dialog.content_scroll.ensureWidgetVisible(dialog.preview_item_button, 12, 12)
    viewport = dialog.content_scroll.viewport()
    button_center = dialog.preview_item_button.mapTo(
        viewport,
        dialog.preview_item_button.rect().center(),
    )
    assert viewport.rect().contains(button_center)
    qtbot.mouseClick(dialog.preview_item_button, Qt.LeftButton)
    assert previewed == [0, 1]

    output_item = dialog.preview_table.item(0, 2)
    assert output_item.text() == "field-1.npy"
    assert str(result.rows[0].outputs[0]) in output_item.toolTip()
    header = dialog.preview_table.horizontalHeader()
    last_column_right = header.sectionViewportPosition(4) + header.sectionSize(4)
    assert last_column_right <= dialog.preview_table.viewport().width() + 4

    qtbot.mouseClick(dialog.run_button, Qt.LeftButton)
    assert len(requests) == 1
    dialog.begin_run(3)
    dialog.content_scroll.ensureWidgetVisible(dialog.run_group, 12, 12)
    assert dialog.run_group.isVisibleTo(dialog.content_scroll.viewport())
    assert dialog.close_button.isVisibleTo(dialog)
