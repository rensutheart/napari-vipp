"""Batch commands share the main toolbar's glyphs and spacing contract."""

import pytest
from qtpy.QtCore import QSize
from qtpy.QtGui import QColor, QIcon, QPalette

from napari_vipp.ui.batch import CollectionBatchDialog
from napari_vipp.ui.toolbar_controls import ToolbarCommandButton, toolbar_icon


@pytest.mark.parametrize("dark", [False, True])
@pytest.mark.parametrize("point_size", [10, 14, 18])
def test_batch_commands_match_shared_toolbar_at_larger_fonts(qtbot, dark, point_size):
    dialog = CollectionBatchDialog()
    qtbot.addWidget(dialog)
    palette = QPalette(dialog.palette())
    background = QColor("#25282f" if dark else "#ffffff")
    foreground = QColor("#eef2f6" if dark else "#202530")
    for role in (QPalette.Base, QPalette.Window, QPalette.AlternateBase):
        palette.setColor(role, background)
    for role in (QPalette.Text, QPalette.WindowText, QPalette.ButtonText):
        palette.setColor(role, foreground)
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#7d8590"))
    dialog.setPalette(palette)
    font = dialog.font()
    font.setPointSize(point_size)
    dialog.setFont(font)
    dialog.resize(1080, 800)
    dialog.show()
    qtbot.wait(10)

    assert "padding: 2px 5px" in dialog.config_row.styleSheet()
    for button, kind in (
        (dialog.load_config_button, "open"),
        (dialog.save_config_button, "save"),
        (dialog.more_button, "more"),
    ):
        assert isinstance(button, ToolbarCommandButton)
        assert button.iconSize() == QSize(18, 18)
        expected_icon = toolbar_icon(kind, dialog.palette())
        for mode in (QIcon.Normal, QIcon.Disabled):
            assert button.icon().pixmap(18, 18, mode).toImage() == (
                expected_icon.pixmap(18, 18, mode).toImage()
            )
        if button.text():
            assert button._toolbar_style_option().text == (
                ToolbarCommandButton._ICON_TEXT_SPACER + button.text()
            )
            assert button.property("text") == button.text()
            assert button.width() >= button.minimumSizeHint().width()
            assert button.height() >= button.minimumSizeHint().height()
        assert not button.autoDefault()

    assert dialog.more_button.text() == ""
    assert dialog.more_button.accessibleName() == "More batch options"
    assert dialog.more_button.menu() is dialog.more_menu
    assert "menu-indicator { image: none" in dialog.config_row.styleSheet()


def test_batch_icons_refresh_when_theme_changes(qtbot):
    dialog = CollectionBatchDialog()
    qtbot.addWidget(dialog)
    rendered = []
    for foreground, background in (("#eef2f6", "#25282f"), ("#202530", "#ffffff")):
        palette = QPalette(dialog.palette())
        palette.setColor(QPalette.ButtonText, QColor(foreground))
        palette.setColor(QPalette.Text, QColor(foreground))
        palette.setColor(QPalette.Base, QColor(background))
        dialog.setPalette(palette)
        rendered.append(dialog.load_config_button.icon().pixmap(18, 18).toImage())
    assert rendered[0] != rendered[1]
