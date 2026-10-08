from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
from qtpy.QtCore import QPoint
from qtpy.QtGui import QFont
from qtpy.QtWidgets import QLabel

from napari_vipp._tests.test_ui_inspector_widget_integration import _widget


@pytest.mark.parametrize("theme", ("dark", "light"))
@pytest.mark.parametrize("font_size", (10, 14))
def test_compute_repair_card_reserves_text_height_when_inspector_resizes(
    qtbot, qapp, tmp_path, theme, font_size,
):
    from napari._qt.qt_resources import get_stylesheet

    previous_font = qapp.font()
    qapp.setFont(QFont("Segoe UI", font_size))
    widget = _widget(qtbot)
    widget.setStyleSheet(
        get_stylesheet(theme, extra_variables={"font_size": f"{font_size}pt"})
    )
    try:
        widget._select_node("gaussian")
        suggestion = SimpleNamespace(
            message=(
                "This node could become eligible for GPU use if its image input "
                "is converted from uint16 to float32. The conversion will preserve "
                "every pixel value exactly, while that converted input uses 2× "
                "as much memory."
            ),
            target_dtype="float32",
        )
        widget._selected_compute_repair = lambda: suggestion
        widget._shared_compute_repairs = lambda _suggestion: (suggestion,)
        widget._sync_selected_compute_repair()
        widget._sync_inspector_presentation()
        widget.compute_section.setExpanded(True)
        before_params = {
            key: deepcopy(node.params) for key, node in widget.pipeline.nodes.items()
        }
        before_connections = tuple(widget.pipeline.connections)

        panel = widget.inspector_panel
        panel.setParent(None)
        qtbot.addWidget(panel)
        panel.setStyleSheet(widget.styleSheet())
        panel.show()
        label = widget.compute_repair_label
        card = widget.compute_repair_panel
        button = widget.add_compute_conversion_button
        heights = []
        for width in (300, 640, 300):
            panel.resize(width, 650)
            qapp.processEvents()
            # Measure an unconstrained QLabel with the same actual font/style;
            # do not trust a heightForWidth clamped by the subject's minimum.
            probe = QLabel(label.text(), card)
            probe.setWordWrap(True)
            probe.setFont(label.font())
            probe.setStyleSheet(label.styleSheet())
            probe.hide()
            probe.ensurePolished()

            def fully_visible(probe=probe):
                required = probe.heightForWidth(label.width())
                return (
                    required > 0
                    and label.height() >= required
                    and card.rect().contains(label.geometry())
                    and card.rect().contains(button.geometry())
                    and label.geometry().bottom() < button.geometry().top()
                )

            qtbot.waitUntil(fully_visible, timeout=2_000)
            heights.append(label.height())
            panel.ensureWidgetVisible(card)
            qtbot.waitUntil(
                lambda: panel.viewport().rect().contains(
                    button.mapTo(panel.viewport(), QPoint(0, button.height() - 1))
                ),
                timeout=2_000,
            )
            assert panel.horizontalScrollBar().maximum() == 0
            assert label.text() == suggestion.message
            assert card.grab().save(str(tmp_path / f"{theme}-{font_size}-{width}.png"))
            probe.deleteLater()

        assert heights[0] > heights[1]
        assert heights[2] == heights[0]
        assert {
            key: node.params for key, node in widget.pipeline.nodes.items()
        } == before_params
        assert tuple(widget.pipeline.connections) == before_connections
    finally:
        qapp.setFont(previous_font)
