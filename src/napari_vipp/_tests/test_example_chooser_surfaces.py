"""Paint-level regression for chooser panes under inherited application QSS."""

import pytest
from qtpy.QtCore import QPoint, Qt
from qtpy.QtGui import QColor, QFont, QPalette
from qtpy.QtWidgets import QWidget

from napari_vipp.ui.dialogs import ExampleWorkflowDialog
from napari_vipp.ui.palette_roles import blend_colors, theme_colors


def _palette(light):
    palette = QPalette()
    background = QColor("#f2f5f9" if light else "#1b2433")
    text = QColor("#182438" if light else "#ecf1fa")
    for role in (QPalette.Window, QPalette.Base, QPalette.AlternateBase):
        palette.setColor(role, background)
    for role in (QPalette.Text, QPalette.WindowText):
        palette.setColor(role, text)
    return palette


def _inherited_style(palette):
    background = palette.color(QPalette.Base).name()
    foreground = palette.color(QPalette.Text).name()
    # Napari-like application rules also target anonymous child QWidget panels.
    # Local QLabel transparency alone cannot erase these subsection rectangles.
    return (
        f"QWidget {{ background-color: {background}; color: {foreground}; }}"
        f"QLabel {{ background-color: {background}; }}"
        f"QTreeWidget {{ background-color: {background}; border: 0; }}"
        f"QScrollArea {{ background-color: {background}; border: 0; }}"
        f"QSplitter::handle {{ background-color: {background}; }}"
    )


def _chooser(qtbot, light, inherited_style, width):
    host = QWidget()
    qtbot.addWidget(host)
    host.setFont(QFont("Segoe UI", 9))
    palette = _palette(light)
    host.setPalette(palette)
    if inherited_style:
        host.setStyleSheet(_inherited_style(palette))
    dialog = ExampleWorkflowDialog(host)
    qtbot.addWidget(dialog)
    dialog.filter_edit.setText("synthetic-colocalization-racc.json")
    dialog.select_example("racc-colocalization")
    # Leave a large blank viewport area, without changing the selected example.
    dialog.tree.collapseAll()
    dialog.resize(width, 900)
    dialog.show()
    return host, dialog


def _painted_pixel(dialog, widget, point):
    point = widget.mapTo(dialog, point)
    pixmap = dialog.grab()
    image = pixmap.toImage()
    ratio = pixmap.devicePixelRatio()
    x, y = round(point.x() * ratio), round(point.y() * ratio)
    assert 0 <= x < image.width() and 0 <= y < image.height()
    return image.pixelColor(x, y)


def _assert_painted_surfaces(dialog, qapp):
    qapp.processEvents()
    colors = theme_colors(dialog.palette())
    sidebar_color = colors.surface
    detail_color = blend_colors(colors.surface, colors.text, 0.055)
    assert sidebar_color != detail_color

    sidebar = _painted_pixel(
        dialog, dialog.sidebar, QPoint(2, dialog.sidebar.height() - 3)
    )
    viewport = dialog.tree.viewport()
    tree_blank = _painted_pixel(
        dialog, viewport, QPoint(viewport.width() - 5, viewport.height() - 5)
    )
    detail_margin = _painted_pixel(
        dialog, dialog.details_scroll.viewport(), QPoint(3, 8)
    )
    assert sidebar == sidebar_color
    assert tree_blank == sidebar_color
    assert detail_margin == detail_color

    # These are child QWidget section surfaces, not just their label colours.
    for section in (dialog.data_panel, dialog.method_panel, dialog.explore_panel):
        dialog.details_scroll.ensureWidgetVisible(section, 0, 0)
        qapp.processEvents()
        point = QPoint(section.width() - 2, 2)
        assert _painted_pixel(dialog, section, point) == detail_color

    dialog.details_scroll.ensureWidgetVisible(dialog.try_panel, 0, 0)
    qapp.processEvents()
    callout = _painted_pixel(dialog, dialog.try_panel, QPoint(8, 5))
    assert callout == colors.info.surface
    assert callout not in (sidebar_color, detail_color)

    handle = dialog.splitter.handle(1)
    points = (
        [QPoint(x, handle.height() // 2) for x in range(handle.width())]
        if dialog.splitter.orientation() == Qt.Horizontal
        else [QPoint(handle.width() // 2, y) for y in range(handle.height())]
    )
    painted_handle = [_painted_pixel(dialog, handle, point) for point in points]
    assert colors.border in painted_handle
    assert sidebar_color in painted_handle
    assert detail_color in painted_handle
    assert dialog.selected_example().id == "racc-colocalization"
    assert dialog.open_button.isEnabled()


@pytest.mark.parametrize("light", [False, True])
@pytest.mark.parametrize("inherited_style", [False, True])
@pytest.mark.parametrize("width", [720, 1180])
def test_sidebar_detail_and_splitter_have_distinct_painted_surfaces(
    qtbot, qapp, light, inherited_style, width
):
    _host, dialog = _chooser(qtbot, light, inherited_style, width)
    _assert_painted_surfaces(dialog, qapp)


@pytest.mark.parametrize("inherited_style", [False, True])
def test_painted_surfaces_refresh_with_runtime_palette_and_orientation_change(
    qtbot, qapp, inherited_style
):
    host, dialog = _chooser(qtbot, False, inherited_style, 1180)
    _assert_painted_surfaces(dialog, qapp)
    light = _palette(True)
    host.setPalette(light)
    if inherited_style:
        host.setStyleSheet(_inherited_style(light))
    dialog.setPalette(light)
    dialog.resize(720, 900)
    qapp.processEvents()
    assert dialog.splitter.orientation() == Qt.Vertical
    _assert_painted_surfaces(dialog, qapp)
