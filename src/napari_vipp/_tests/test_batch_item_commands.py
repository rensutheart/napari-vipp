"""Batch item commands retain their groups, labels, and toolbar glyphs."""

from itertools import combinations

import pytest
from qtpy.QtCore import QPoint, QRect, QSize
from qtpy.QtGui import QColor, QIcon, QPalette
from qtpy.QtWidgets import QApplication, QPushButton

from napari_vipp._tests.test_ui_batch import _actions, _preview_result
from napari_vipp.ui.batch import CollectionBatchDialog
from napari_vipp.ui.palette_roles import theme_colors
from napari_vipp.ui.toolbar_controls import ToolbarCommandButton


def _rect_in(widget, parent):
    return QRect(widget.mapTo(parent, QPoint(0, 0)), widget.size())


def _item_commands(dialog):
    return (
        dialog.recheck_item_button,
        dialog.preview_item_button,
        dialog.load_overrides_button,
        dialog.preview_button,
        dialog.item_filter,
        dialog.item_search,
    )


def test_review_banner_details_and_status_keep_theme_semantics(qtbot, tmp_path):
    plan = _preview_result(tmp_path)
    dialog = CollectionBatchDialog(actions=_actions(plan, []))
    qtbot.addWidget(dialog)
    dialog.apply_preview_result(plan, preview_representative=False)
    colors = theme_colors(dialog.palette())
    assert colors.success.surface.name() in dialog.preview_status.styleSheet()
    assert colors.success.foreground.name() in dialog.preview_status.styleSheet()
    assert "border-left: 3px" in dialog.review_banner.styleSheet()
    assert "Nothing was saved" in dialog.preview_status.text()
    assert "Selected item" in dialog.item_details.toPlainText()
    assert "<h3" not in dialog.item_details.toHtml()
    assert (
        dialog.preview_table.item(1, 4).foreground().color()
        == colors.success.foreground
    )

    dialog.show_workspace_activity("Check failed", state="error")
    assert colors.error.surface.name() in dialog.preview_status.styleSheet()
    dialog._invalidate_preview_plan()
    assert "Check batch again" in dialog.preview_status.text()
    assert "Needs recheck" == dialog.preview_table.item(1, 4).text()
    assert (
        dialog.preview_table.item(1, 4).foreground().color()
        == colors.warning.foreground
    )


@pytest.mark.parametrize("width", [1080, 736, 560])
def test_item_command_groups_fit_without_clipping_or_overlap(qtbot, tmp_path, width):
    from napari._qt.qt_resources import get_stylesheet

    plan = _preview_result(tmp_path)
    dialog = CollectionBatchDialog(actions=_actions(plan, []))
    qtbot.addWidget(dialog)
    dialog.setStyleSheet(get_stylesheet("dark", extra_variables={"font_size": "10pt"}))
    dialog._check_batch()
    dialog.tabs.setCurrentIndex(1)
    dialog.resize(width, 800)
    dialog.show()
    qtbot.wait(20)

    assert dialog.width() == width
    selected = dialog.item_selection_commands
    collection = dialog.item_collection_commands
    selected_rect = _rect_in(selected, dialog.items_command_row)
    collection_rect = _rect_in(collection, dialog.items_command_row)
    if width == 1080:
        assert collection_rect.left() - selected_rect.right() - 1 >= 18
        assert abs(collection_rect.center().y() - selected_rect.center().y()) <= 2
    else:
        assert selected_rect.bottom() < collection_rect.top()
        assert selected_rect.left() == collection_rect.left()

    widgets = _item_commands(dialog)
    assert [button.text() for button in widgets[:4]] == [
        "Recheck selected",
        "Preview selected",
        "Load overrides",
        "Recheck all",
    ]
    viewport = dialog.items_page
    rectangles = [_rect_in(widget, viewport) for widget in widgets]
    for widget, rect in zip(widgets, rectangles, strict=True):
        assert widget.isVisible()
        assert viewport.rect().contains(rect), (widget.objectName(), rect)
        assert widget.width() >= widget.minimumSizeHint().width()
        assert widget.height() >= widget.minimumSizeHint().height()
        if isinstance(widget, QPushButton):
            assert widget.width() >= widget.sizeHint().width(), widget.text()
            assert widget.height() >= widget.sizeHint().height(), widget.text()
    assert all(
        not left.intersects(right) for left, right in combinations(rectangles, 2)
    )
    for group, members in ((selected, widgets[:3]), (collection, widgets[3:])):
        group_rectangles = [_rect_in(widget, group) for widget in members]
        assert all(group.isAncestorOf(widget) for widget in members)
        assert all(group.rect().contains(rect) for rect in group_rectangles)
        centers = [rect.center().y() for rect in group_rectangles]
        assert max(centers) - min(centers) <= 2
        assert all(
            left.right() < right.left()
            for left, right in zip(group_rectangles, group_rectangles[1:], strict=False)
        )


def test_item_footer_and_tab_icons_refresh_with_the_palette(qtbot, tmp_path):
    plan = _preview_result(tmp_path)
    dialog = CollectionBatchDialog(actions=_actions(plan, []))
    qtbot.addWidget(dialog)
    buttons = (
        *_item_commands(dialog)[:4],
        dialog.next_button,
        dialog.footer_overrides_button,
        dialog.run_button,
        dialog.cancel_run_button,
    )
    rendered = []
    for dark in (True, False):
        foreground = QColor("#eef2f6" if dark else "#202530")
        background = QColor("#25282f" if dark else "#ffffff")
        palette = QPalette(dialog.palette())
        for role in (QPalette.Text, QPalette.WindowText, QPalette.ButtonText):
            palette.setColor(role, foreground)
        for role in (QPalette.Base, QPalette.Window, QPalette.AlternateBase):
            palette.setColor(role, background)
        palette.setColor(
            QPalette.Disabled,
            QPalette.ButtonText,
            QColor("#8691a3" if dark else "#788291"),
        )
        dialog.setPalette(palette)
        QApplication.processEvents()
        icons = []
        for button in buttons:
            assert isinstance(button, ToolbarCommandButton)
            assert button.iconSize() == QSize(18, 18)
            assert button.text()
            assert not button.icon().isNull()
            assert button._toolbar_style_option().text == (
                ToolbarCommandButton._ICON_TEXT_SPACER + button.text()
            )
            icons.append(button.icon())
        assert dialog.tabs.count() == 4
        for index in range(4):
            icon = dialog.tabs.tabIcon(index)
            assert not icon.isNull(), dialog.tabs.tabText(index)
            icons.append(icon)
        images = [icon.pixmap(18, 18, QIcon.Normal).toImage() for icon in icons]
        assert all(
            any(
                image.pixelColor(x, y).alpha() > 0
                for y in range(image.height())
                for x in range(image.width())
            )
            for image in images
        )
        rendered.append(images)
    assert all(
        dark != light for dark, light in zip(rendered[0], rendered[1], strict=True)
    )


def test_item_commands_refresh_minimums_after_inherited_style_changes(qtbot):
    dialog = CollectionBatchDialog()
    qtbot.addWidget(dialog)
    dialog.tabs.setCurrentIndex(1)
    dialog.resize(1080, 800)
    dialog.show()
    widths = []
    for points in (10, 14, 10):
        dialog.setStyleSheet(f"QWidget {{ font-size: {points}pt; }}")
        dialog._layout_item_commands()
        qtbot.wait(10)
        buttons = _item_commands(dialog)[:4]
        for button in buttons:
            assert button.minimumSize() == button.sizeHint()
            assert button.width() >= button.sizeHint().width(), button.text()
            assert button.height() >= button.sizeHint().height(), button.text()
        widths.append(buttons[0].minimumWidth())
    assert widths[1] > widths[0]
    assert widths[2] == widths[0]


def test_item_filter_reserves_native_width_and_shrinks_after_style_changes(
    qtbot, tmp_path
):
    plan = _preview_result(tmp_path)
    dialog = CollectionBatchDialog(actions=_actions(plan, []))
    qtbot.addWidget(dialog)
    dialog._check_batch()
    dialog.tabs.setCurrentIndex(1)
    dialog.resize(1080, 800)
    dialog.show()
    widths = []
    for points in (10, 16, 10):
        dialog.setStyleSheet(
            f"QComboBox {{ font-size: {points}pt; padding: 5px 22px; }}"
        )
        for _ in range(3):
            dialog._layout_item_commands()
            qtbot.wait(10)
            combo = dialog.item_filter
            native_width = combo.minimumSizeHint().width()
            assert native_width > 120
            assert combo.minimumWidth() == max(120, native_width)
            assert combo.maximumWidth() == max(180, native_width)
            assert combo.isVisible()
            assert combo.width() >= native_width
            assert dialog.items_page.rect().contains(_rect_in(combo, dialog.items_page))
        widths.append(combo.minimumWidth())
    assert widths[1] > widths[0]
    assert widths[2] == widths[0]
