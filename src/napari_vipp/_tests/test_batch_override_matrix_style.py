"""Readable override values and mixed-weight headers in the actual Qt matrix."""

from dataclasses import replace

import pytest
from qtpy.QtCore import QPoint, Qt
from qtpy.QtGui import QFont, QFontMetrics
from qtpy.QtWidgets import QApplication, QStyleOptionViewItem

from napari_vipp._tests.test_batch_table_theme import _palette
from napari_vipp._tests.test_ui_batch_parameter_overrides import (
    _many_sources,
    _parameter,
)
from napari_vipp.ui.batch_override_widgets import BatchOverrideParameterHeader
from napari_vipp.ui.batch_overrides import BatchParameterOverrideEditor


def _editor(qtbot):
    editor = BatchParameterOverrideEditor()
    qtbot.addWidget(editor)
    parameters = [
        replace(
            _parameter(node, "radius", "float", 0.0, 100.0, workflow_value=value),
            node_label="Subtract Background",
            parameter=replace(
                _parameter(node, "radius", "float", 0.0, 100.0).parameter,
                label="Radius (px)",
            ),
        )
        for node, value in (
            ("subtract_background_1", 15),
            ("subtract_background_2", 20),
        )
    ]
    assert editor.configure(_many_sources(3), parameters)
    return editor, parameters


@pytest.mark.parametrize("dark", [False, True])
def test_numeric_values_and_live_editors_are_centered_in_both_themes(qtbot, dark):
    from napari._qt.qt_resources import get_stylesheet

    editor, parameters = _editor(qtbot)
    editor.setPalette(_palette(dark))
    editor.setStyleSheet(
        get_stylesheet(
            "dark" if dark else "light", extra_variables={"font_size": "10pt"}
        )
    )
    editor.select_source(position=0, checked=True)
    editor.apply_selected_values({parameters[0].key: 25.5})
    editor.resize(900, 650)
    editor.show()
    QApplication.processEvents()
    table = editor.table
    for row in range(3):
        for column in (1, 2):
            assert table.item(row, column).textAlignment() == Qt.AlignCenter
    # Identity remains left aligned so long sample names are easy to scan.
    assert table.item(0, 0).textAlignment() != Qt.AlignCenter
    assert table.item(0, 1).text() == "25.5"
    assert table.item(1, 1).text() == ""
    for row in (0, 1):
        index = table.model().index(row, 1)
        delegate = table.itemDelegate()
        field = delegate.createEditor(table.viewport(), QStyleOptionViewItem(), index)
        delegate.setEditorData(field, index)
        assert field.alignment() == Qt.AlignCenter
        assert field.placeholderText() == "inherit 15"
        delegate.destroyEditor(field, index)
    assert not table.grab().isNull()


@pytest.mark.parametrize("dark", [False, True])
def test_header_bolds_only_node_name_and_preserves_accessible_metadata(qtbot, dark):
    editor, parameters = _editor(qtbot)
    # Even if the host theme supplies a bold header font, metadata stays regular.
    editor.setFont(QFont("Segoe UI", 10, QFont.Bold))
    editor.setPalette(_palette(dark))
    editor.resize(900, 650)
    editor.show()
    QApplication.processEvents()
    table = editor.table
    header = table.horizontalHeader()
    assert isinstance(header, BatchOverrideParameterHeader)
    title_font, detail_font = header.line_fonts()
    assert title_font.bold()
    assert detail_font.weight() == QFont.Normal
    for column, binding in enumerate(parameters, 1):
        assert header.section_lines(column) == (
            "Subtract Background",
            f"[{binding.node_id}]",
            "Radius (px)",
            f"Workflow: {binding.workflow_value:g}",
        )
        plain = table.horizontalHeaderItem(column).text()
        assert plain.startswith(f"Subtract Background [{binding.node_id}]\n")
        assert "Radius (px)\nWorkflow:" in plain
        assert table.horizontalHeaderItem(column).toolTip() == plain
        assert 160 <= table.columnWidth(column) <= 240
    content_height = (
        QFontMetrics(title_font).height() + 3 * QFontMetrics(detail_font).height()
    )
    assert header.height() >= content_height + 8
    assert not header.grab().isNull()


def test_rich_header_tracks_font_changes_and_frozen_identity_geometry(qtbot):
    editor, parameters = _editor(qtbot)
    editor.configure(_many_sources(70), parameters)
    editor.resize(720, 550)
    editor.show()
    table = editor.table
    frozen = table.frozen_identity

    def aligned():
        if frozen.horizontalHeader().height() != table.horizontalHeader().height():
            return False
        return (
            table.viewport().mapTo(editor, QPoint()).y()
            == frozen.viewport().mapTo(editor, QPoint()).y()
        )

    qtbot.waitUntil(aligned)
    before = table.horizontalHeader().height()
    editor.setFont(QFont("Segoe UI", 15))
    qtbot.waitUntil(lambda: table.horizontalHeader().height() != before and aligned())
    table.horizontalScrollBar().setValue(100)
    table.verticalScrollBar().setValue(70)
    QApplication.processEvents()
    assert aligned()
    assert frozen.verticalScrollBar().value() == table.verticalScrollBar().value()
    assert frozen.columnWidth(0) == table.columnWidth(0)


def test_long_node_ids_do_not_force_unbounded_matrix_columns(qtbot):
    editor, parameters = _editor(qtbot)
    parameters = [
        replace(binding, node_id="long_custom_node_identifier_" * 10 + str(index))
        for index, binding in enumerate(parameters)
    ]
    editor.configure(_many_sources(1), parameters)
    for column, binding in enumerate(parameters, 1):
        assert editor.table.columnWidth(column) == 240
        assert binding.node_id in editor.table.horizontalHeaderItem(column).toolTip()
        assert (
            editor.table.horizontalHeader()
            .section_lines(column)[1]
            .endswith(f"{column - 1}]")
        )
