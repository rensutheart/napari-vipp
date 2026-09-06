"""Source setup cards retain axis behavior in the mockup's compact layout."""

import pytest
from qtpy.QtCore import QPoint, QSize
from qtpy.QtGui import QColor, QPalette

from napari_vipp.ui.batch import CollectionBatchDialog
from napari_vipp.ui.toolbar_controls import ToolbarCommandButton, toolbar_icon


def _point_in_card(widget, card):
    return widget.mapTo(card, QPoint())


@pytest.mark.parametrize("width", [420, 600])
def test_source_card_places_full_width_folder_above_pattern_and_axes(qtbot, width):
    dialog = CollectionBatchDialog()
    qtbot.addWidget(dialog)
    source = dialog._source_rows[0]
    card = source["widget"]
    card.setParent(None)
    qtbot.addWidget(card)
    card.resize(width, card.sizeHint().height())
    card.show()
    qtbot.wait(10)

    folder = source["folder"]
    pattern = source["pattern"]
    axes = source["axis_declaration"].mode_combo
    folder_pos = _point_in_card(folder, card)
    pattern_pos = _point_in_card(pattern, card)
    axes_pos = _point_in_card(axes, card)
    assert _point_in_card(source["folder_label"], card).y() < folder_pos.y()
    assert folder_pos.y() < pattern_pos.y() == axes_pos.y()
    assert pattern_pos.x() + pattern.width() < axes_pos.x()
    assert folder.width() > pattern.width()
    assert axes_pos.x() + axes.width() <= card.width() - 10
    assert card.width() == width
    assert source["folder_label"].buddy() is folder
    assert source["pattern_label"].buddy() is pattern
    assert source["axis_label"].buddy() is axes
    assert source["axis_label"].text() == "Image axes"


def test_source_card_uses_the_shared_folder_icon_and_scoped_frame_style(qtbot):
    dialog = CollectionBatchDialog(
        source_nodes=[{"node_id": "image", "title": "My image"}]
    )
    qtbot.addWidget(dialog)
    source = dialog._source_rows[0]
    button = source["browse_button"]
    assert isinstance(button, ToolbarCommandButton)
    assert button.text() == ""
    assert button.iconSize() == QSize(18, 18)
    assert not button.autoDefault()
    assert button.accessibleName() == "Choose folder for My image"
    assert "My image" in button.toolTip()
    assert not source["title_icon"].pixmap().isNull()
    assert "QFrame#batchSourceCard" in source["widget"].styleSheet()
    assert "QFrame {" not in source["widget"].styleSheet()

    images = []
    for foreground, background in (("#eef2f6", "#25282f"), ("#202530", "#ffffff")):
        palette = QPalette(dialog.palette())
        palette.setColor(QPalette.ButtonText, QColor(foreground))
        palette.setColor(QPalette.Text, QColor(foreground))
        palette.setColor(QPalette.Base, QColor(background))
        dialog.setPalette(palette)
        image = button.icon().pixmap(18, 18).toImage()
        assert image == toolbar_icon("open", dialog.palette()).pixmap(18, 18).toImage()
        images.append(image)
    assert images[0] != images[1]


def test_source_card_keeps_advanced_axes_and_output_suggestion(qtbot, tmp_path):
    dialog = CollectionBatchDialog()
    qtbot.addWidget(dialog)
    source = dialog._source_rows[0]
    control = source["axis_declaration"]
    control.setText("QYX -> ZYX")
    assert control.text() == "QYX -> ZYX"
    control.setText("TYX -> ZYX")
    assert control.text() == "TYX -> ZYX"
    assert not control.advanced_edit.isHidden()
    source["folder"].setText(str(tmp_path))
    assert dialog.output_edit.text() == str(tmp_path / "output")
    assert control.text() == "TYX -> ZYX"
