"""Explicit reset scopes preserve checked samples, hidden cells, and safety."""

import pytest
from qtpy.QtCore import QSize, Qt
from qtpy.QtGui import QColor, QIcon, QPalette
from qtpy.QtWidgets import QMessageBox

from napari_vipp._tests.test_ui_batch_parameter_overrides import (
    _parameter,
    _source_item,
)
from napari_vipp.ui.batch_overrides import (
    BatchOverrideSourceItem,
    BatchParameterOverrideEditor,
)
from napari_vipp.ui.toolbar_controls import ToolbarCommandButton, toolbar_icon


def _editor(qtbot, count=3):
    editor = BatchParameterOverrideEditor()
    qtbot.addWidget(editor)
    sources = tuple(
        BatchOverrideSourceItem(
            "input", f"Sample {index:03d}", _source_item(str(index))
        )
        for index in range(count)
    )
    parameters = (
        _parameter("blur", "sigma", "float", 0, 100),
        _parameter("background", "radius", "int", 0, 100),
    )
    assert editor.configure(sources, parameters)
    return editor


def test_reset_selected_covers_hidden_columns_filters_and_pages(qtbot):
    editor = _editor(qtbot, 110)
    first, second, last = (
        editor._source_keys[0],
        editor._source_keys[1],
        editor._source_keys[-1],
    )
    editor.apply_selected_values(
        {("blur", "sigma"): 2.5, ("background", "radius"): 12},
        source_keys=(first, second, last),
    )
    editor.set_selected_source_keys((first, last))
    editor.set_visible_parameters([("blur", "sigma")])
    editor.set_page(2)
    editor.sample_search.setText("109")
    changed = []
    selection = []
    editor.overridesChanged.connect(lambda: changed.append(True))
    editor.selectionChanged.connect(selection.append)

    assert editor.reset_selected_overrides(confirm=False)

    assert editor.selected_source_keys() == (first, last)
    assert editor.sample_search.text() == "109"
    assert editor._visible_parameter_keys == {("blur", "sigma")}
    assert editor.override_value_count() == 2
    assert {entry.source_item_key for entry in editor.overrides()} == {second}
    assert changed == [True]
    assert selection == []
    assert not editor.reset_selected_button.isEnabled()
    assert not editor.reset_selected_overrides(confirm=False)


def test_search_labels_persist_and_missing_parameter_label_tracks_filter(qtbot):
    editor = _editor(qtbot)
    editor.show()
    editor.sample_search.setText("Sample")
    editor.parameter_search.setText("blur")
    for label, field in (
        (editor.sample_search_label, editor.sample_search),
        (editor.parameter_search_label, editor.parameter_search),
        (editor.filter_label, editor.filter_combo),
    ):
        assert label.isVisible()
        assert label.text()
        assert label.buddy() is field
    assert not editor.missing_parameter_label.isVisible()
    editor.filter_combo.setCurrentIndex(editor.filter_combo.findData("missing"))
    assert editor.missing_parameter_label.isVisible()
    assert editor.missing_parameter_combo.isVisible()
    editor.filter_combo.setCurrentIndex(0)
    assert not editor.missing_parameter_label.isVisible()


def test_page_selection_and_deselect_all_have_distinct_scopes(qtbot):
    editor = _editor(qtbot, 110)
    last = editor._source_keys[-1]
    editor.set_selected_source_keys((last,))
    editor._check_page(True)
    assert last in editor.selected_source_keys()
    editor._check_page(False)
    assert editor.selected_source_keys() == (last,)
    assert editor.select_page_checkbox.text() == "Select this page"
    editor.clear_selection_button.click()
    assert editor.selected_source_keys() == ()


def test_reset_selected_can_discard_invalid_raw_edits(qtbot):
    editor = _editor(qtbot)
    key = editor._source_keys[0]
    editor.set_selected_source_keys((key,))
    editor._cell_changed((key, "blur", "sigma"), "not a number")
    assert editor.error_message
    assert editor.reset_selected_button.isEnabled()

    assert editor.reset_selected_overrides(confirm=False)

    assert not editor.error_message
    assert editor.overrides() == ()
    assert editor.selected_source_keys() == (key,)


@pytest.mark.parametrize("accepted", [False, True])
def test_reset_selected_confirmation_explains_scope_and_defaults_cancel(
    qtbot, monkeypatch, accepted
):
    editor = _editor(qtbot)
    keys = editor._source_keys[:2]
    editor.set_selected_source_keys(keys)
    editor.apply_selected_values({("blur", "sigma"): 2})
    calls = []

    def question(*args):
        calls.append(args)
        return QMessageBox.Yes if accepted else QMessageBox.Cancel

    monkeypatch.setattr(QMessageBox, "question", question)
    assert editor.reset_selected_overrides() is accepted
    assert len(calls) == 1
    assert "2 parameter overrides" in calls[0][2]
    assert "2 selected samples" in calls[0][2]
    assert "hidden columns" in calls[0][2]
    assert "Run or bypass settings" in calls[0][2]
    assert calls[0][-1] == QMessageBox.Cancel
    assert editor.override_value_count() == (0 if accepted else 2)
    assert editor.selected_source_keys() == keys


def test_reset_all_removes_all_values_but_preserves_selection_and_contract_error(qtbot):
    editor = _editor(qtbot)
    editor.apply_selected_values(
        {("blur", "sigma"): 2}, source_keys=editor._source_keys
    )
    editor.set_selected_source_keys((editor._source_keys[1],))
    editor._cell_changed((editor._source_keys[2], "background", "radius"), "bad")
    editor._contract_error = "The reviewed source revisions have changed."
    assert editor.override_value_count() == 4
    events = []
    editor.overridesChanged.connect(lambda: events.append(True))

    assert editor.reset_all_overrides(confirm=False)

    assert editor.override_value_count() == 0
    assert editor.error_message == "The reviewed source revisions have changed."
    assert editor.selected_source_keys() == (editor._source_keys[1],)
    assert events == [True]
    assert not editor.reset_all_overrides(confirm=False)
    assert events == [True]


def test_reset_all_default_confirmation_is_cancel_and_noop_never_prompts(
    qtbot, monkeypatch
):
    editor = _editor(qtbot)
    questions = []

    def cancel(*args):
        questions.append(args)
        return QMessageBox.Cancel

    monkeypatch.setattr(QMessageBox, "question", cancel)
    assert not editor.reset_all_overrides()
    assert not editor.reset_selected_overrides()
    assert questions == []
    editor.apply_selected_values(
        {("blur", "sigma"): 3}, source_keys=editor._source_keys
    )
    assert not editor.reset_all_overrides()
    assert editor.override_value_count() == 3
    assert "all samples" in questions[0][2]
    assert questions[0][-1] == QMessageBox.Cancel


def test_reset_selected_in_changed_filter_keeps_hidden_checked_samples(qtbot):
    editor = _editor(qtbot)
    editor.select_all_matching()
    editor.apply_selected_values({("blur", "sigma"): 4})
    selected = editor.selected_source_keys()
    editor.filter_combo.setCurrentIndex(editor.filter_combo.findData("changed"))

    assert editor.reset_selected_overrides(confirm=False)

    assert editor._matching_keys == ()
    assert editor.selected_source_keys() == selected
    assert "3 hidden by filters" in editor.selection_label.text()
    assert editor.clear_selection_button.isEnabled()
    assert not editor.reset_selected_button.isEnabled()


def test_deselect_only_clears_checks_and_never_values(qtbot):
    editor = _editor(qtbot)
    editor.select_all_matching()
    editor.apply_selected_values({("blur", "sigma"): 2})
    before = editor.overrides()
    assert editor.clear_selection_button.text() == "Deselect all"
    assert "overrides stay unchanged" in editor.clear_selection_button.toolTip()

    qtbot.mouseClick(editor.clear_selection_button, Qt.LeftButton)

    assert editor.selected_source_keys() == ()
    assert editor.overrides() == before
    assert not editor.reset_selected_button.isEnabled()


@pytest.mark.parametrize("dark", [False, True])
def test_editor_command_icons_share_toolbar_padding_and_theme(qtbot, dark):
    editor = _editor(qtbot)
    palette = QPalette(editor.palette())
    foreground = QColor("#eef2f6" if dark else "#202530")
    palette.setColor(QPalette.ButtonText, foreground)
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#7d8590"))
    editor.setPalette(palette)
    for button, kind in editor._override_command_icons:
        assert isinstance(button, ToolbarCommandButton)
        assert button.iconSize() == QSize(18, 18)
        assert button._toolbar_style_option().text == (
            ToolbarCommandButton._ICON_TEXT_SPACER + button.text()
        )
        expected = toolbar_icon(kind, palette)
        for mode in (QIcon.Normal, QIcon.Disabled):
            assert button.icon().pixmap(18, 18, mode).toImage() == (
                expected.pixmap(18, 18, mode).toImage()
            )


@pytest.mark.parametrize("width", [560, 736, 1080])
def test_selection_commands_reflow_without_clipping(qtbot, width):
    editor = _editor(qtbot)
    editor.resize(width, 700)
    editor.show()
    qtbot.wait(10)
    assert editor.width() == width
    commands = (
        editor.select_matching_button,
        editor.clear_selection_button,
        editor.reset_selected_button,
        editor.edit_selected_button,
    )
    for button in commands:
        assert button.width() >= button.minimumSizeHint().width()
        assert editor.rect().contains(button.geometry())
    for index, button in enumerate(commands):
        for other in commands[index + 1 :]:
            assert not button.geometry().intersects(other.geometry())
