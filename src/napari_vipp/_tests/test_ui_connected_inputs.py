from __future__ import annotations

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QPalette
from qtpy.QtWidgets import QFormLayout, QLabel, QWidget

from napari_vipp.core.metadata import AxisMetadata, image_state_from_array
from napari_vipp.core.tables import TableState
from napari_vipp.ui.connected_inputs import (
    ConnectedInputBinding,
    ConnectedInputsCard,
    connected_input_scientific_summary,
)
from napari_vipp.ui.palette_roles import theme_colors


def _palette(*, dark: bool) -> QPalette:
    palette = QPalette()
    surface = QColor("#171a21" if dark else "#f8fafc")
    alternate = QColor("#222631" if dark else "#eef2f7")
    text = QColor("#f8fafc" if dark else "#172033")
    palette.setColor(QPalette.Window, surface)
    palette.setColor(QPalette.Base, surface)
    palette.setColor(QPalette.AlternateBase, alternate)
    palette.setColor(QPalette.Button, alternate)
    palette.setColor(QPalette.WindowText, text)
    palette.setColor(QPalette.Text, text)
    palette.setColor(QPalette.ButtonText, text)
    return palette


def test_connected_inputs_card_keeps_each_role_paired_with_its_source(qtbot):
    card = ConnectedInputsCard()
    qtbot.addWidget(card)
    card.set_bindings(
        [
            ConnectedInputBinding(
                "Labels",
                "labels",
                "Filter Labels By Volume",
                "out",
            ),
            ConnectedInputBinding(
                "Intensity image",
                "image",
                "Split Channels",
                "Ch 3",
            ),
        ]
    )

    assert card.title_label.text() == "Connected inputs"
    assert card.form_layout.rowCount() == 2
    assert len(card.rows) == 2
    assert [row.role_label.text() for row in card.rows] == [
        "Labels",
        "Intensity image",
    ]
    assert [row.source_label.text() for row in card.rows] == [
        "Filter Labels By Volume · out",
        "Split Channels · Ch 3",
    ]
    for index, row in enumerate(card.rows):
        item = card.form_layout.itemAt(
            index,
            QFormLayout.ItemRole.SpanningRole,
        )
        assert item is not None
        assert item.widget() is row
        assert row.role_label.wordWrap()
        assert row.source_label.wordWrap()
        assert row.source_label.textInteractionFlags() & Qt.TextSelectableByMouse
        assert not row.icon_label.pixmap().isNull()
        assert row.role_label.text() in row.accessibleName()
        # Keep one genuinely uniform card surface: the text is laid out
        # directly in the row rather than inside a second QWidget surface.
        assert all(
            isinstance(child, QLabel)
            for child in row.children()
            if isinstance(child, QWidget)
        )


def test_connected_inputs_card_is_theme_safe_and_describes_empty_ports(qtbot):
    card = ConnectedInputsCard()
    qtbot.addWidget(card)
    card.set_bindings([ConnectedInputBinding("ROI mask", "mask_or_labels")])

    row = card.rows[0]
    assert row.source_label.text() == "Not connected · expects mask or labels"
    assert "expects mask or labels" in row.toolTip()

    for dark in (True, False):
        palette = _palette(dark=dark)
        card.setPalette(palette)
        card.refresh_theme(palette)
        colors = theme_colors(palette)
        style = card.styleSheet()
        assert colors.info.surface.name() in style
        assert colors.text.name() in style
        assert not row.icon_label.pixmap().isNull()

    card.set_bindings([])
    assert card.isHidden()
    assert card.form_layout.rowCount() == 0
    assert card.rows == []


def test_connected_input_text_and_tooltips_treat_metadata_as_plain_text(qtbot):
    card = ConnectedInputsCard()
    qtbot.addWidget(card)
    card.set_bindings(
        [
            ConnectedInputBinding(
                "Signal <required>",
                "image",
                "Source <b>literal</b>",
                "Ch <3>",
            )
        ]
    )

    row = card.rows[0]
    assert row.role_label.textFormat() == Qt.PlainText
    assert row.source_label.textFormat() == Qt.PlainText
    assert row.role_label.text() == "Signal <required>"
    assert row.source_label.text() == "Source <b>literal</b> · Ch <3>"
    assert "&lt;b&gt;literal&lt;/b&gt;" in row.toolTip()


def test_connected_input_scientific_summary_reports_grid_depth_and_sampling():
    data = np.zeros((12, 96, 128), dtype=np.uint16)
    state = image_state_from_array(
        data,
        axes=(
            AxisMetadata("z", "space", "micrometer", 0.5),
            AxisMetadata("y", "space", "micrometer", 0.1),
            AxisMetadata("x", "space", "micrometer", 0.1),
        ),
        defer_statistics=True,
    )

    summary = connected_input_scientific_summary(None, state)

    assert "ZYX: 12 × 96 × 128" in summary
    assert "uint16 (16-bit integer)" in summary
    assert "sampling Z 0.5 µm, Y 0.1 µm, X 0.1 µm" in summary


def test_connected_input_scientific_summary_uses_table_dimensions():
    state = TableState(
        row_count=184,
        column_count=14,
        columns=tuple(f"field_{index}" for index in range(14)),
        table_kind="object measurements",
    )

    assert connected_input_scientific_summary(None, state) == (
        "184 rows · 14 fields · object measurements"
    )
