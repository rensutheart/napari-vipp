from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QFont, QPalette
from qtpy.QtWidgets import QDialog, QLineEdit

from napari_vipp.core.batch_parameters import (
    BatchParameterOverride,
    BatchSourceParameterOverrides,
    batch_source_item_override_key,
)
from napari_vipp.core.pipeline import ParameterSpec
from napari_vipp.core.source_items import (
    ResolvedSourceItemIdentity,
    SourceCapabilities,
    SourceContainerBundle,
    SourceContainerMember,
    SourceItem,
    SourceItemSelector,
    SourceReaderDescriptor,
    SourceRevisionProof,
)
from napari_vipp.ui.batch_overrides import (
    BatchOverrideEditorError,
    BatchOverrideParameterSpec,
    BatchOverrideSourceItem,
    BatchParameterOverrideEditor,
)
from napari_vipp.ui.palette_roles import custom_paint_colors


def _theme_palette(*, base: str, alternate: str, text: str) -> QPalette:
    palette = QPalette()
    palette.setColor(QPalette.Base, QColor(base))
    palette.setColor(QPalette.AlternateBase, QColor(alternate))
    palette.setColor(QPalette.Text, QColor(text))
    return palette


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _source_item(seed: str, *, selector: str = "image") -> SourceItem:
    size = 16
    return SourceItem(
        SourceContainerBundle(
            uri=f"C:/private/{seed}.npy",
            format="npy",
            revision=SourceRevisionProof("file", _sha(seed), 1, size),
            members=(SourceContainerMember(".", _sha(f"{seed}-member"), size),),
        ),
        SourceItemSelector(
            selector,
            "image",
            source_axes=("Y", "X"),
            effective_axes=("Y", "X"),
        ),
        SourceReaderDescriptor("numpy", "numpy", "2.0"),
        SourceCapabilities(decoded_size_estimate=True),
        ResolvedSourceItemIdentity(
            key=selector,
            name=f"Image {seed}",
            kind="image",
            shape=(2, 4),
            dtype="uint16",
            axes=("Y", "X"),
            raw_axes=("Y", "X"),
            estimated_decoded_bytes=size,
        ),
    )


def _parameter(
    node_id: str,
    name: str,
    kind: str,
    minimum: int | float,
    maximum: int | float,
    *,
    data_dependent_bounds: bool = False,
    workflow_value: int | float | None = None,
) -> BatchOverrideParameterSpec:
    default: int | float = 1 if kind == "int" else 1.0
    return BatchOverrideParameterSpec(
        node_id=node_id,
        node_label=f"Node {node_id}",
        operation_id="gaussian_blur",
        parameter=ParameterSpec(
            name,
            name.title(),
            kind,
            default,
            minimum,
            maximum,
            1 if kind == "int" else 0.1,
            2,
            data_dependent_bounds=data_dependent_bounds,
        ),
        workflow_value=(default if workflow_value is None else workflow_value),
    )


def test_editor_emits_canonical_typed_overrides_and_blank_inherits(qtbot):
    first = _source_item("first")
    second = _source_item("second")
    int_parameter = _parameter("node-b", "iterations", "int", 1, 20)
    float_parameter = _parameter("node-a", "sigma", "float", 0.0, 12.0)
    editor = BatchParameterOverrideEditor()
    qtbot.addWidget(editor)

    assert editor.configure(
        [
            BatchOverrideSourceItem("input", "Field 1 / image", first),
            BatchOverrideSourceItem("input", "Field 2 / image", second),
        ],
        [int_parameter, float_parameter],
    )
    editor.editor_for("input", first, "node-b", "iterations").setText("3")
    editor.editor_for("input", first, "node-a", "sigma").setText("2.75")
    # Every second-source cell remains blank and therefore inherits.

    overrides = editor.overrides()

    assert len(overrides) == 1
    assert overrides[0].source_item_key == batch_source_item_override_key(
        "input",
        first,
    )
    assert overrides[0].values == (
        BatchParameterOverride("node-a", "sigma", 2.75),
        BatchParameterOverride("node-b", "iterations", 3),
    )
    assert isinstance(overrides[0].values[0].value, float)
    assert isinstance(overrides[0].values[1].value, int)
    assert (
        editor.editor_for("input", second, "node-b", "iterations").placeholderText()
        == "inherit 1"
    )
    assert "authored workflow value" in editor.help_label.text()


def test_editor_refuses_non_numeric_source_and_output_parameters():
    text = ParameterSpec("mode", "Mode", "text", "a", 0, 0, 1)
    with pytest.raises(BatchOverrideEditorError, match="only declared int and float"):
        BatchOverrideParameterSpec("node", "Node", "filter", text, 0)

    numeric = ParameterSpec("index", "Index", "int", 0, 0, 5, 1)
    with pytest.raises(BatchOverrideEditorError, match="source, output, selector"):
        BatchOverrideParameterSpec("input", "Source", "input", numeric, 0)
    with pytest.raises(BatchOverrideEditorError, match="source, output, selector"):
        BatchOverrideParameterSpec("output", "Output", "batch_output", numeric, 0)


@pytest.mark.parametrize("duplicate", ["label", "source", "parameter"])
def test_editor_flags_duplicate_contract_data_without_guessing(qtbot, duplicate):
    first = _source_item("first")
    second = _source_item("second")
    sources = [
        BatchOverrideSourceItem("input", "First", first),
        BatchOverrideSourceItem("input", "Second", second),
    ]
    parameters = [_parameter("node", "sigma", "float", 0.0, 10.0)]
    if duplicate == "label":
        sources[1] = BatchOverrideSourceItem("input", "First", second)
    elif duplicate == "source":
        sources[1] = BatchOverrideSourceItem("input", "Second", first)
    else:
        parameters.append(parameters[0])
    editor = BatchParameterOverrideEditor()
    qtbot.addWidget(editor)

    assert not editor.configure(sources, parameters)
    assert "Duplicate" in editor.status_label.text()
    with pytest.raises(BatchOverrideEditorError, match="Duplicate"):
        editor.overrides()


def test_editor_flags_stale_saved_source_and_parameter_records(qtbot):
    first = _source_item("first")
    missing = _source_item("missing")
    parameter = _parameter("node", "sigma", "float", 0.0, 10.0)
    editor = BatchParameterOverrideEditor()
    qtbot.addWidget(editor)
    stale_source = BatchSourceParameterOverrides(
        batch_source_item_override_key("input", missing),
        (BatchParameterOverride("node", "sigma", 2.0),),
    )

    assert not editor.configure(
        [BatchOverrideSourceItem("input", "First", first)],
        [parameter],
        overrides=(stale_source,),
    )
    assert "stale" in editor.status_label.text().lower()

    stale_parameter = BatchSourceParameterOverrides(
        batch_source_item_override_key("input", first),
        (BatchParameterOverride("removed-node", "sigma", 2.0),),
    )
    assert not editor.configure(
        [BatchOverrideSourceItem("input", "First", first)],
        [parameter],
        overrides=(stale_parameter,),
    )
    assert "no longer" in editor.status_label.text().lower()


def test_invalid_cell_is_visible_and_cannot_emit(qtbot):
    source = _source_item("first")
    parameter = _parameter("node", "iterations", "int", 1, 5)
    editor = BatchParameterOverrideEditor()
    qtbot.addWidget(editor)
    assert editor.configure(
        [BatchOverrideSourceItem("input", "First", source)],
        [parameter],
    )

    cell = editor.editor_for("input", source, "node", "iterations")
    # Programmatic text assignment exercises the same fail-closed read boundary
    # even when a platform validator declines an interactive keystroke.
    cell.setText("2.5")

    assert "whole number" in editor.status_label.text()
    assert "#ef4444" in cell.styleSheet()
    with pytest.raises(BatchOverrideEditorError, match="whole number"):
        editor.overrides()


def test_override_editor_status_follows_runtime_palette_and_keeps_error_tone(qtbot):
    editor = BatchParameterOverrideEditor()
    qtbot.addWidget(editor)
    light = _theme_palette(base="#ffffff", alternate="#f2f4f7", text="#111827")
    editor.setPalette(light)

    assert custom_paint_colors(light).muted_text.name() in (
        editor.help_label.styleSheet()
    )
    editor._show_error("Invalid override")
    assert "#b91c1c" in editor.status_label.styleSheet()

    dark = _theme_palette(base="#111827", alternate="#1f2937", text="#f8fafc")
    editor.setPalette(dark)

    assert "#fca5a5" in editor.status_label.styleSheet()


def test_data_dependent_threshold_accepts_raw_integer_intensity(qtbot):
    source = _source_item("uint16")
    parameter = _parameter(
        "threshold",
        "threshold",
        "float",
        0.0,
        1.0,
        data_dependent_bounds=True,
        workflow_value=5000.0,
    )
    editor = BatchParameterOverrideEditor()
    qtbot.addWidget(editor)
    assert editor.configure(
        [BatchOverrideSourceItem("input", "uint16 image", source)],
        [parameter],
    )
    cell = editor.editor_for(
        "input",
        source,
        "threshold",
        "threshold",
    )

    cell.setText("13000")

    assert "Ready" in editor.status_label.text()
    assert cell.validator() is None
    assert cell.placeholderText() == "inherit 5000"
    assert "authored workflow value 5000" in cell.toolTip()
    assert "authored workflow value 1" not in cell.toolTip()
    assert "connected image's intensity scale" in cell.toolTip()
    assert editor.overrides()[0].values == (
        BatchParameterOverride("threshold", "threshold", 13_000.0),
    )

    cell.setText("nan")
    assert "finite" in editor.status_label.text()


def test_collection_dialog_preserves_values_when_unused_and_exposes_hook(qtbot):
    from napari_vipp.ui.batch import CollectionBatchDialog

    dialog = CollectionBatchDialog()
    qtbot.addWidget(dialog)
    assert "parameter_overrides" not in dialog.values()
    assert dialog.parameter_override_group.isHidden()

    source = _source_item("first")
    parameter = _parameter("node", "sigma", "float", 0.0, 10.0)
    assert dialog.configure_parameter_overrides(
        [BatchOverrideSourceItem("input", "First", source)],
        [parameter],
        overrides=(),
    )
    dialog.parameter_override_editor.editor_for(
        "input", source, "node", "sigma"
    ).setText("4.5")

    assert dialog.values()["parameter_overrides"] == dialog.parameter_overrides()
    assert not dialog.parameter_override_group.isHidden()


def _many_sources(count):
    return [
        BatchOverrideSourceItem(
            "input", f"Field {index:04}", _source_item(f"field-{index:04}")
        )
        for index in range(count)
    ]


def test_large_matrix_pages_without_allocating_cell_editors(qtbot):
    editor = BatchParameterOverrideEditor()
    qtbot.addWidget(editor)
    sources = _many_sources(1200)
    parameters = [
        _parameter(f"node-{index}", "sigma", "float", 0.0, 10.0) for index in range(30)
    ]
    assert editor.configure(sources, parameters)
    assert editor.table.rowCount() == 50
    assert editor.table.columnCount() == 31
    assert not editor._editors
    assert len(editor.findChildren(QLineEdit)) < 10
    assert "1–50 of 1,200" in editor.page_label.text()

    source = sources[1199]
    editor.editor_for("input", source.source_item, "node-29", "sigma").setText("4.5")
    assert editor.current_page == 23
    assert len(editor._editors) == 1
    editor.set_page(0)
    assert not editor._editors
    assert editor.overrides()[0].values[0].value == 4.5
    assert (
        editor.editor_for("input", source.source_item, "node-29", "sigma").text()
        == "4.5"
    )


def test_checked_selection_survives_pages_and_filters(qtbot):
    editor = BatchParameterOverrideEditor()
    qtbot.addWidget(editor)
    sources = _many_sources(120)
    editor.configure(sources, [_parameter("node", "sigma", "float", 0.0, 10.0)])
    editor.table.item(0, 0).setCheckState(Qt.Checked)
    first_key = editor.selected_source_keys()[0]
    editor.set_page(1)
    editor.table.item(1, 0).setCheckState(Qt.Checked)
    assert len(editor.selected_source_keys()) == 2
    editor.sample_search.setText("Field 0119")
    assert editor.table.rowCount() == 1
    assert "2 hidden by filters (still included)" in editor.selection_label.text()
    editor.select_all_matching()
    assert len(editor.selected_source_keys()) == 3
    editor.sample_search.clear()
    editor.set_page(0)
    assert editor.table.item(0, 0).checkState() == Qt.Checked
    assert first_key in editor.selected_source_keys()
    editor.clear_selection()
    assert editor.selected_source_keys() == ()


def test_sample_filters_include_paths_and_missing_parameter(qtbot):
    editor = BatchParameterOverrideEditor()
    qtbot.addWidget(editor)
    sources = _many_sources(3)
    parameters = [
        _parameter("node", "sigma", "float", 0.0, 10.0),
        _parameter("node", "iterations", "int", 1, 10),
    ]
    editor.configure(sources, parameters)
    editor.editor_for("input", sources[0].source_item, "node", "sigma").setText("3")
    editor.editor_for("input", sources[1].source_item, "node", "iterations").setText(
        "2"
    )
    editor.filter_combo.setCurrentIndex(editor.filter_combo.findData("changed"))
    assert len(editor._page_keys) == 2
    editor.filter_combo.setCurrentIndex(editor.filter_combo.findData("workflow"))
    assert editor.table.item(0, 0).text() == "Field 0002"
    editor.filter_combo.setCurrentIndex(editor.filter_combo.findData("missing"))
    editor.missing_parameter_combo.setCurrentIndex(0)
    assert [editor.table.item(row, 0).text() for row in range(2)] == [
        "Field 0001",
        "Field 0002",
    ]
    editor.sample_search.setText("C:/private/field-0002.npy")
    assert editor.table.rowCount() == 1
    assert editor.table.item(0, 0).text() == "Field 0002"


def test_hidden_invalid_values_remain_fail_closed(qtbot):
    editor = BatchParameterOverrideEditor()
    qtbot.addWidget(editor)
    sources = _many_sources(75)
    editor.configure(sources, [_parameter("node", "iterations", "int", 1, 5)])
    editor.editor_for("input", sources[0].source_item, "node", "iterations").setText(
        "2.5"
    )
    editor.set_visible_parameters([])
    editor.set_page(1)
    assert "whole number" in editor.error_message
    with pytest.raises(BatchOverrideEditorError, match="whole number"):
        editor.overrides()
    cell = editor.editor_for("input", sources[0].source_item, "node", "iterations")
    assert cell.text() == "2.5"
    assert "#ef4444" in cell.styleSheet()
    cell.clear()
    assert editor.overrides() == ()


def test_column_chooser_groups_nodes_and_keeps_hidden_values(qtbot):
    editor = BatchParameterOverrideEditor()
    qtbot.addWidget(editor)
    source = _many_sources(1)[0]
    parameters = [
        _parameter("one", "sigma", "float", 0.0, 10.0),
        _parameter("one", "iterations", "int", 1, 5),
        _parameter("two", "sigma", "float", 0.0, 10.0),
    ]
    editor.configure([source], parameters)
    editor.editor_for("input", source.source_item, "one", "sigma").setText("3")
    chooser = editor.create_column_chooser()
    qtbot.addWidget(chooser)
    assert chooser.parameter_tree.topLevelItemCount() == 2
    chooser.parameter_items[("one", "sigma")].setCheckState(0, Qt.Unchecked)
    chooser.accept()
    assert editor.table.columnCount() == 3
    assert editor.overrides()[0].values[0].value == 3.0
    editor.parameter_search.setText("Node one / Sigma")
    editor.parameter_search.returnPressed.emit()
    assert ("one", "sigma") in editor._visible_parameter_keys
    assert editor.table.columnCount() == 4


def test_bulk_edit_changes_only_chosen_parameters_including_hidden_selection(qtbot):
    editor = BatchParameterOverrideEditor()
    qtbot.addWidget(editor)
    sources = _many_sources(3)
    parameters = [
        _parameter("node", "sigma", "float", 0.0, 10.0),
        _parameter("node", "iterations", "int", 1, 10),
    ]
    editor.configure(sources, parameters)
    editor.select_source(position=0, checked=True)
    editor.select_source(position=1, checked=True)
    editor.apply_selected_values({("node", "iterations"): 4})
    editor.sample_search.setText("Field 0002")
    dialog = editor.create_edit_selected_dialog()
    qtbot.addWidget(dialog)
    assert not dialog.apply_button.isEnabled()
    assert not dialog.draft_fields
    dialog.parameter_items[("node", "sigma")].setCheckState(0, Qt.Checked)
    _row, mode, field = dialog.draft_fields[("node", "sigma")]
    assert not dialog.apply_button.isEnabled()
    field.setText("2.5")
    assert dialog.apply_button.isEnabled()
    assert len(editor.overrides()[0].values) == 1  # Still a draft.
    dialog.apply_button.click()
    assert dialog.result() == QDialog.Accepted
    assert len(editor.overrides()) == 2
    assert all(
        override.values
        == (
            BatchParameterOverride("node", "iterations", 4),
            BatchParameterOverride("node", "sigma", 2.5),
        )
        for override in editor.overrides()
    )

    reset = editor.create_edit_selected_dialog()
    qtbot.addWidget(reset)
    reset.parameter_items[("node", "sigma")].setCheckState(0, Qt.Checked)
    _row, mode, field = reset.draft_fields[("node", "sigma")]
    mode.setCurrentIndex(mode.findData("inherit"))
    assert not field.isEnabled()
    assert reset.apply_button.isEnabled()
    reset.apply_button.click()
    assert all(
        override.values == (BatchParameterOverride("node", "iterations", 4),)
        for override in editor.overrides()
    )


def test_bulk_cancel_and_invalid_values_leave_every_sample_unchanged(qtbot):
    editor = BatchParameterOverrideEditor()
    qtbot.addWidget(editor)
    editor.configure(_many_sources(2), [_parameter("node", "iterations", "int", 1, 5)])
    editor.select_all_matching()
    dialog = editor.create_edit_selected_dialog()
    qtbot.addWidget(dialog)
    dialog.parameter_items[("node", "iterations")].setCheckState(0, Qt.Checked)
    dialog.draft_fields[("node", "iterations")][2].setText("2.5")
    assert not dialog.apply_button.isEnabled()
    assert "whole number" in dialog.feedback_label.text()
    dialog.reject()
    assert editor.overrides() == ()
    with pytest.raises(BatchOverrideEditorError):
        editor.apply_selected_values({("node", "iterations"): 8})
    assert editor.overrides() == ()


def test_source_selection_hook_uses_exact_identity_and_zero_based_position(qtbot):
    editor = BatchParameterOverrideEditor()
    qtbot.addWidget(editor)
    sources = _many_sources(80)
    editor.configure(sources, [_parameter("node", "sigma", "float", 0.0, 10.0)])
    editor.sample_search.setText("Field 0000")
    key = batch_source_item_override_key("input", sources[79].source_item)
    with qtbot.waitSignal(editor.sourceSelected) as received:
        assert editor.select_source(key, checked=True)
    assert received.args == [key, 79]
    assert editor.current_page == 1
    assert editor.sample_search.text() == ""
    assert editor.selected_source_keys() == (key,)
    assert not editor.select_source("unmatched")
    assert not editor.select_source(position=80)


def test_checking_and_clearing_quarantine_saved_values(qtbot):
    editor = BatchParameterOverrideEditor()
    qtbot.addWidget(editor)
    editor.configure(
        _many_sources(2), [_parameter("node", "sigma", "float", 0.0, 10.0)]
    )
    editor.select_all_matching()
    editor.apply_selected_values({("node", "sigma"): 2.0})
    editor.mark_saved_overrides_verifying(2)
    assert editor.table.rowCount() == 0
    assert editor.selected_source_keys() == ()
    assert not editor.edit_selected_button.isEnabled()
    assert editor.error_message
    editor.clear_contract()
    assert not editor.error_message
    assert editor.overrides() == ()


def test_inline_keyboard_edit_commits_and_blanking_refreshes_changed_filter(qtbot):
    editor = BatchParameterOverrideEditor()
    qtbot.addWidget(editor)
    sources = _many_sources(3)
    editor.configure(sources, [_parameter("node", "sigma", "float", 0.0, 10.0)])
    editor.resize(800, 500)
    editor.show()
    editor.table.setCurrentCell(0, 1)
    editor.table.editItem(editor.table.item(0, 1))
    cell = next(iter(editor._editors.values()))
    qtbot.keyClicks(cell, "3.75")
    qtbot.keyClick(cell, Qt.Key_Return)
    assert editor.overrides()[0].values[0].value == 3.75
    editor.filter_combo.setCurrentIndex(editor.filter_combo.findData("changed"))
    assert editor.table.rowCount() == 1
    cell = editor.editor_for("input", sources[0].source_item, "node", "sigma")
    cell.clear()
    cell.editingFinished.emit()
    qtbot.waitUntil(lambda: editor.table.rowCount() == 0)
    assert editor.overrides() == ()


def test_frozen_identity_tracks_scroll_and_checked_model(qtbot):
    editor = BatchParameterOverrideEditor()
    qtbot.addWidget(editor)
    editor.configure(
        _many_sources(60),
        [
            _parameter(f"node-{index}", "sigma", "float", 0.0, 10.0)
            for index in range(8)
        ],
    )
    editor.resize(800, 500)
    editor.show()
    frozen = editor.table.frozen_identity
    assert frozen.model() is editor.table.model()
    assert frozen.isColumnHidden(1)
    before = frozen.geometry().left()
    editor.reveal_parameter("node-7", "sigma")
    assert editor.table.horizontalScrollBar().value() > 0
    assert frozen.geometry().left() == before
    editor.table.verticalScrollBar().setValue(80)
    assert frozen.verticalScrollBar().value() == 80
    frozen.model().setData(frozen.model().index(0, 0), Qt.Checked, Qt.CheckStateRole)
    assert len(editor.selected_source_keys()) == 1


def test_reconfigure_retains_selection_and_columns_only_for_exact_sources(qtbot):
    editor = BatchParameterOverrideEditor()
    qtbot.addWidget(editor)
    sources = _many_sources(2)
    parameters = [
        _parameter("node", "sigma", "float", 0.0, 10.0),
        _parameter("node", "iterations", "int", 1, 5),
    ]
    editor.configure(sources, parameters)
    editor.select_source(position=0, checked=True)
    editor.apply_selected_values({("node", "sigma"): 2.0})
    editor.set_visible_parameters([("node", "iterations")])
    saved = editor.overrides()
    key = editor.selected_source_keys()[0]
    assert editor.configure(sources, parameters, overrides=saved)
    assert editor.selected_source_keys() == (key,)
    assert editor._visible_parameter_keys == {("node", "iterations")}
    replaced = BatchOverrideSourceItem(
        "input", sources[0].label, _source_item("new-revision")
    )
    assert not editor.configure([replaced, sources[1]], parameters, overrides=saved)
    assert "stale" in editor.error_message
    assert editor.selected_source_keys() == ()
    assert editor.table.rowCount() == 0


@pytest.mark.parametrize("count", [0, 1, 3, 50])
def test_override_navigation_hidden_for_one_page_or_less(qtbot, count):
    editor = BatchParameterOverrideEditor()
    qtbot.addWidget(editor)
    assert editor.previous_page_button.isHidden()
    assert editor.next_page_button.isHidden()
    editor.configure(
        _many_sources(count), [_parameter("node", "sigma", "float", 0.0, 10.0)]
    )
    assert editor.previous_page_button.isHidden()
    assert editor.next_page_button.isHidden()


def test_override_navigation_hides_when_filters_reduce_page_count(qtbot):
    editor = BatchParameterOverrideEditor()
    qtbot.addWidget(editor)
    editor.configure(
        _many_sources(51), [_parameter("node", "sigma", "float", 0.0, 10.0)]
    )
    assert not editor.previous_page_button.isHidden()
    assert not editor.next_page_button.isHidden()
    assert not editor.previous_page_button.isEnabled()
    assert editor.next_page_button.isEnabled()
    editor.sample_search.setText("Field 0050")
    assert editor.previous_page_button.isHidden()
    assert editor.next_page_button.isHidden()
    editor.sample_search.setText("no matching sample")
    assert editor.previous_page_button.isHidden()
    assert editor.next_page_button.isHidden()
    editor.sample_search.clear()
    assert not editor.previous_page_button.isHidden()
    assert not editor.next_page_button.isHidden()
    editor.clear_contract()
    assert editor.previous_page_button.isHidden()
    assert editor.next_page_button.isHidden()


def test_frozen_override_rows_align_after_hidden_configure_and_theme_change(qtbot):
    from napari._qt.qt_resources import get_stylesheet

    editor = BatchParameterOverrideEditor()
    qtbot.addWidget(editor)
    editor.setFont(QFont("Segoe UI", 10))
    editor.setStyleSheet(get_stylesheet("dark", extra_variables={"font_size": "10pt"}))
    editor.configure(
        _many_sources(3),
        [_parameter("threshold", "threshold", "float", 0.0, 100.0)],
    )
    editor.resize(900, 650)
    editor.show()
    table = editor.table
    frozen = table.frozen_identity

    def rows_aligned():
        if frozen.horizontalHeader().height() != table.horizontalHeader().height():
            return False
        for row in range(table.rowCount()):
            parameter = table.visualRect(table.model().index(row, 1))
            identity = frozen.visualRect(frozen.model().index(row, 0))
            parameter_top = table.viewport().mapTo(editor, parameter.topLeft()).y()
            identity_top = frozen.viewport().mapTo(editor, identity.topLeft()).y()
            if identity_top != parameter_top:
                return False
        return True

    qtbot.waitUntil(rows_aligned)
    assert table.horizontalHeader().height() > 30
    initial_header_height = table.horizontalHeader().height()
    editor.setFont(QFont("Segoe UI", 12))
    editor.setStyleSheet(get_stylesheet("light", extra_variables={"font_size": "12pt"}))
    qtbot.waitUntil(
        lambda: (
            table.horizontalHeader().height() != initial_header_height
            and rows_aligned()
        )
    )


def test_replace_checked_keys_refreshes_once_across_pages_and_filters(
    qtbot, monkeypatch
):
    editor = BatchParameterOverrideEditor()
    qtbot.addWidget(editor)
    sources = _many_sources(1200)
    editor.configure(sources, [_parameter("node", "sigma", "float", 0.0, 10.0)])
    editor.select_source(position=1, checked=True)
    editor.apply_selected_values({("node", "sigma"): 2.5})
    saved = editor.overrides()
    editor.sample_search.setText("Field 1199")
    active = editor._active_source_key
    keys = tuple(
        batch_source_item_override_key(source.source_node_id, source.source_item)
        for source in sources
    )
    rendered = []
    changed = []
    overrides_changed = []
    source_selected = []
    render_page = editor._render_page

    def track_render():
        rendered.append(True)
        render_page()

    monkeypatch.setattr(editor, "_render_page", track_render)
    editor.selectionChanged.connect(changed.append)
    editor.overridesChanged.connect(lambda: overrides_changed.append(True))
    editor.sourceSelected.connect(lambda *args: source_selected.append(args))
    editor.set_selected_source_keys(tuple(reversed(keys)) + (keys[0],))

    assert editor.selected_source_keys() == keys
    assert rendered == [True]
    assert changed == [keys]
    assert overrides_changed == []
    assert source_selected == []
    assert editor.overrides() == saved
    assert editor._active_source_key == active
    assert editor.sample_search.text() == "Field 1199"
    assert editor.table.rowCount() == 1
    assert editor.table.item(0, 0).checkState() == Qt.Checked
    assert "1,199 hidden by filters" in editor.selection_label.text()

    editor.set_selected_source_keys(())
    assert editor.selected_source_keys() == ()
    assert rendered == [True, True]
    assert changed == [keys, ()]
    assert editor.overrides() == saved


def test_replace_checked_keys_rejects_unknown_identity_before_mutation(qtbot):
    editor = BatchParameterOverrideEditor()
    qtbot.addWidget(editor)
    editor.configure(
        _many_sources(3), [_parameter("node", "sigma", "float", 0.0, 10.0)]
    )
    editor.select_source(position=1, checked=True)
    selected = editor.selected_source_keys()
    changes = []
    editor.selectionChanged.connect(changes.append)

    with pytest.raises(BatchOverrideEditorError, match="exact source contract"):
        editor.set_selected_source_keys((*selected, "unknown-source-key"))
    assert editor.selected_source_keys() == selected
    assert changes == []
    with pytest.raises(BatchOverrideEditorError, match="sequence"):
        editor.set_selected_source_keys(selected[0])
    assert editor.selected_source_keys() == selected
    assert changes == []


def test_duplicate_node_labels_are_disambiguated_but_unique_labels_stay_short(qtbot):
    editor = BatchParameterOverrideEditor()
    qtbot.addWidget(editor)
    parameters = [
        replace(
            _parameter(node_id, "threshold", "float", 0.0, 100.0),
            node_label="Binary Threshold",
        )
        for node_id in ("threshold-red", "threshold-green")
    ]
    parameters.append(
        replace(
            _parameter("gaussian", "sigma", "float", 0.0, 10.0),
            node_label="Gaussian Blur",
        )
    )
    editor.configure(_many_sources(1), parameters)
    expected = [
        "Binary Threshold [threshold-red]",
        "Binary Threshold [threshold-green]",
        "Gaussian Blur",
    ]
    for column, title in enumerate(expected, 1):
        assert editor.table.horizontalHeaderItem(column).text().splitlines()[0] == title
        assert editor.missing_parameter_combo.itemText(column - 1).startswith(
            title + " / "
        )
        if column < 3:
            assert editor.table.horizontalHeader().section_lines(column)[:2] == (
                "Binary Threshold",
                f"[{parameters[column - 1].node_id}]",
            )
            assert 160 <= editor.table.columnWidth(column) <= 240
            assert editor.table.horizontalHeaderItem(column).toolTip().startswith(title)

    chooser = editor.create_column_chooser()
    qtbot.addWidget(chooser)
    assert [
        chooser.parameter_tree.topLevelItem(index).text(0) for index in range(3)
    ] == expected

    editor.select_all_matching()
    bulk = editor.create_edit_selected_dialog()
    qtbot.addWidget(bulk)
    assert [
        bulk.parameter_tree.topLevelItem(index).text(0) for index in range(3)
    ] == expected
    for binding, title in zip(parameters, expected, strict=True):
        bulk.parameter_items[binding.key].setCheckState(0, Qt.Checked)
        field = bulk.draft_fields[binding.key][2]
        assert field.accessibleName() == f"{title} / {binding.parameter.label}"

    # Hiding the other matching label must not make the remaining node ambiguous.
    editor.set_visible_parameters([parameters[1].key])
    assert editor.table.horizontalHeaderItem(1).text().startswith(expected[1] + "\n")
