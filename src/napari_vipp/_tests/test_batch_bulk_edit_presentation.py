"""Bulk edit drafts explain their scope and fill the available dialog space."""

import pytest
from qtpy.QtCore import Qt
from qtpy.QtGui import QFont
from qtpy.QtWidgets import QLabel

from napari_vipp._tests.test_batch_table_theme import _palette
from napari_vipp._tests.test_ui_batch_parameter_overrides import (
    _many_sources,
    _parameter,
)
from napari_vipp.ui.batch_overrides import BatchParameterOverrideEditor


def _dialog(qtbot):
    owner = BatchParameterOverrideEditor()
    qtbot.addWidget(owner)
    owner.configure(
        _many_sources(120),
        [
            _parameter(f"node-{i}", name, "float", 0, 100, workflow_value=99.9)
            for i in range(30)
            for name in ("low", "high")
        ],
    )
    owner.set_selected_source_keys(owner._source_keys[:2])
    dialog = owner.create_edit_selected_dialog()
    qtbot.addWidget(dialog)
    return owner, dialog


@pytest.mark.parametrize("dark", [True, False])
@pytest.mark.parametrize("size", [(980, 720), (560, 680)])
def test_parameter_picker_uses_dialog_height_and_stacks_when_narrow(qtbot, dark, size):
    from napari.qt import get_stylesheet

    _owner, dialog = _dialog(qtbot)
    dialog.setPalette(_palette(dark))
    dialog.setStyleSheet(get_stylesheet("dark" if dark else "light"))
    dialog.setFont(QFont("Segoe UI", 10))
    dialog.resize(*size)
    dialog.show()
    qtbot.wait(20)
    assert dialog.width() == size[0]
    assert dialog.height() == size[1]
    assert dialog.parameter_tree.maximumHeight() > 1000
    assert dialog.parameter_tree.verticalScrollBar().maximum() > 0
    assert dialog.search_label.buddy() is dialog.search
    assert dialog.empty_state.isVisible()
    assert not dialog.apply_button.isEnabled()
    if size[0] > 780:
        assert dialog.splitter.orientation() == Qt.Horizontal
        assert dialog.parameter_tree.height() > 350
        assert abs(dialog.picker.height() - dialog.changes.height()) <= 2
    else:
        assert dialog.splitter.orientation() == Qt.Vertical
        assert dialog.parameter_tree.height() >= 120
        assert dialog.fields_scroll.height() >= 160
    assert dialog.apply_button.geometry().bottom() < dialog.height()


def test_checked_fields_remain_visible_and_included_when_search_hides_them(qtbot):
    owner, dialog = _dialog(qtbot)
    dialog.show()
    key = ("node-0", "low")
    dialog.parameter_items[key].setCheckState(0, Qt.Checked)
    qtbot.wait(10)
    row, mode, field = dialog.draft_fields[key]
    assert not dialog.empty_state.isVisible()
    assert field.placeholderText() == "Enter a value"
    field.setText("12.34567890123456")
    dialog.search.setText("node-29")
    assert "hidden by search" in dialog.selection_summary.text()
    assert row.isVisible()
    assert dialog.apply_button.isEnabled()
    assert owner.overrides() == ()
    mode.setCurrentIndex(mode.findData("inherit"))
    assert not field.isEnabled()
    assert not field.isVisible()
    assert "99.9" in row.inherit_notice.text()
    assert row.inherit_notice.isVisible()
    mode.setCurrentIndex(mode.findData("set"))
    assert field.text() == "12.34567890123456"
    assert field.isVisible()
    # The reference is readable, but editing/serialization retains precision.
    assert all(
        "99.900000000000006" not in label.text() for label in row.findChildren(QLabel)
    )
    dialog.apply_button.click()
    assert all(
        entry.values[0].value == 12.34567890123456 for entry in owner.overrides()
    )


def test_bulk_selection_scope_includes_other_pages_and_hidden_samples(qtbot):
    owner, initial = _dialog(qtbot)
    initial.reject()
    owner.set_selected_source_keys((owner._source_keys[0], owner._source_keys[99]))
    dialog = owner.create_edit_selected_dialog()
    qtbot.addWidget(dialog)
    assert "other pages are included" in dialog.scope_detail.text()
    dialog.reject()
    owner.sample_search.setText("Field 0000")
    filtered = owner.create_edit_selected_dialog()
    qtbot.addWidget(filtered)
    assert "hidden by filters are included" in filtered.scope_detail.text()
    assert filtered.apply_button.text() == "Apply to 2 samples"
    filtered.search.setText("not a parameter")
    assert not filtered.search_empty_label.isHidden()
    assert not filtered.apply_button.isEnabled()
