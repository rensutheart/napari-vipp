from __future__ import annotations

from collections.abc import Iterator

import pytest
from qtpy.compat import isalive
from qtpy.QtCore import QPoint, Qt
from qtpy.QtGui import QColor, QPalette
from qtpy.QtWidgets import (
    QApplication,
    QLineEdit,
    QMenu,
    QToolButton,
    QTreeWidgetItem,
)

from napari_vipp._graph import OPERATION_MIME
from napari_vipp.core.pipeline import grouped_palette_specs
from napari_vipp.ui.iconography import (
    operation_icon_kind,
    palette_branch_color,
    palette_category_colors,
)
from napari_vipp.ui.palette import (
    CATEGORY_ROLE,
    OPERATION_ROLE,
    NodeLibraryPanel,
    _exec_menu,
)
from napari_vipp.ui.panel_toggle import SidePanelToggleButton


def _walk(item: QTreeWidgetItem) -> Iterator[QTreeWidgetItem]:
    for index in range(item.childCount()):
        child = item.child(index)
        yield child
        yield from _walk(child)


def _category_item(panel: NodeLibraryPanel, category: str) -> QTreeWidgetItem:
    for index in range(panel.palette.topLevelItemCount()):
        item = panel.palette.topLevelItem(index)
        if item.data(0, CATEGORY_ROLE) == category:
            return item
    raise AssertionError(f"Node-library category not found: {category}")


def _operation_item(tree, operation_id: str) -> QTreeWidgetItem:
    for index in range(tree.topLevelItemCount()):
        root = tree.topLevelItem(index)
        for item in _walk(root):
            if item.data(0, OPERATION_ROLE) == operation_id:
                return item
    raise AssertionError(f"Node-library operation not found: {operation_id}")


def _contrast_ratio(first: QColor, second: QColor) -> float:
    """Return WCAG contrast without relying on production color helpers."""

    def relative_luminance(color: QColor) -> float:
        channels = []
        for value in (color.redF(), color.greenF(), color.blueF()):
            channels.append(
                value / 12.92
                if value <= 0.04045
                else ((value + 0.055) / 1.055) ** 2.4
            )
        return (
            0.2126 * channels[0]
            + 0.7152 * channels[1]
            + 0.0722 * channels[2]
        )

    bright, dark = sorted(
        (relative_luminance(first), relative_luminance(second)),
        reverse=True,
    )
    return (bright + 0.05) / (dark + 0.05)


def _theme_palette(*, base: str, text: str) -> QPalette:
    palette = QPalette()
    surface = QColor(base)
    foreground = QColor(text)
    alternate = (
        surface.lighter(108)
        if surface.lightness() < 128
        else surface.darker(104)
    )
    for role in (QPalette.Base, QPalette.Window, QPalette.Button):
        palette.setColor(role, surface)
    palette.setColor(QPalette.AlternateBase, alternate)
    for role in (QPalette.Text, QPalette.WindowText, QPalette.ButtonText):
        palette.setColor(role, foreground)
    return palette


@pytest.fixture
def node_library(qtbot) -> NodeLibraryPanel:
    panel = NodeLibraryPanel(grouped_palette_specs())
    qtbot.addWidget(panel)
    panel.resize(260, 640)
    panel.show()
    return panel


def test_category_rows_show_recursive_counts_theme_safe_text_and_icons(node_library):
    groups = grouped_palette_specs()

    for category, subgroups in groups.items():
        item = _category_item(node_library, category)
        expected_count = sum(len(specs) for specs in subgroups.values())
        text, tint, _accent = palette_category_colors(
            category,
            node_library.palette.palette(),
        )

        assert item.text(0) == category
        assert item.text(1) == str(expected_count)
        assert item.toolTip(0) == f"{category} — {expected_count} nodes"
        assert item.data(0, Qt.AccessibleTextRole) == (
            f"{category}, {expected_count} nodes"
        )
        assert item.foreground(0).color() == text
        assert item.foreground(1).color() == text
        assert item.background(0).color() == tint
        assert item.background(1).color() == tint
        assert not item.icon(0).isNull()


@pytest.mark.parametrize(
    ("base", "text"),
    [
        ("#171923", "#f1f5f9"),
        ("#f8fafc", "#111827"),
    ],
    ids=("dark", "light"),
)
def test_palette_branch_colors_keep_strong_contrast_across_themes(base, text):
    palette = QPalette()
    palette.setColor(QPalette.Base, QColor(base))
    palette.setColor(QPalette.Text, QColor(text))

    for category in grouped_palette_specs():
        _row_text, category_tint, _accent = palette_category_colors(
            category,
            palette,
        )
        branch = palette_branch_color(category, palette)

        assert branch.isValid()
        assert _contrast_ratio(branch, palette.color(QPalette.Base)) >= 4.5
        assert _contrast_ratio(branch, category_tint) >= 4.5


def test_node_library_and_open_popup_recolor_on_same_instance(qtbot, node_library):
    node_library.set_compact(True)
    filtering_button = next(
        button
        for button in node_library.compact_rail.category_buttons
        if button.property("vipp_category") == "Filtering"
    )

    dark = _theme_palette(base="#111827", text="#f8fafc")
    node_library.setPalette(dark)
    node_library.open_category("Filtering", filtering_button)
    qtbot.waitUntil(node_library.popup.isVisible)

    expanded_item = _category_item(node_library, "Filtering")
    popup_item = _category_item(node_library.popup, "Filtering")
    dark_text, dark_tint, _accent = palette_category_colors("Filtering", dark)
    dark_expanded_icon = expanded_item.icon(0).cacheKey()
    dark_rail_icon = filtering_button.icon().cacheKey()
    dark_popup_icon = popup_item.icon(0).cacheKey()
    dark_close_icon = node_library.popup.close_button.icon().cacheKey()
    assert expanded_item.foreground(0).color() == dark_text
    assert expanded_item.background(0).color() == dark_tint
    assert popup_item.foreground(0).color() == dark_text
    assert node_library.popup.palette.palette().color(QPalette.Base) == QColor(
        "#111827"
    )

    light = _theme_palette(base="#ffffff", text="#111827")
    node_library.setPalette(light)
    light_text, light_tint, _accent = palette_category_colors("Filtering", light)
    qtbot.waitUntil(
        lambda: expanded_item.foreground(0).color() == light_text
        and popup_item.foreground(0).color() == light_text
    )

    assert expanded_item.background(0).color() == light_tint
    assert node_library.popup.palette.palette().color(QPalette.Base) == QColor(
        "#ffffff"
    )
    assert expanded_item.icon(0).cacheKey() != dark_expanded_icon
    assert filtering_button.icon().cacheKey() != dark_rail_icon
    assert popup_item.icon(0).cacheKey() != dark_popup_icon
    assert node_library.popup.close_button.icon().cacheKey() != dark_close_icon


def test_global_search_restores_exact_expansion_state_after_query_changes(node_library):
    filtering = _category_item(node_library, "Filtering")
    segmentation = _category_item(node_library, "Segmentation")
    filtering_subgroup = filtering.child(0)
    segmentation_subgroup = segmentation.child(0)

    filtering.setExpanded(False)
    filtering_subgroup.setExpanded(False)
    segmentation.setExpanded(True)
    segmentation_subgroup.setExpanded(False)
    before = {
        id(item): item.isExpanded()
        for item in (
            filtering,
            filtering_subgroup,
            segmentation,
            segmentation_subgroup,
        )
    }

    node_library.search_edit.setText("gblr")
    assert filtering.isExpanded()
    assert filtering_subgroup.isExpanded()

    # Refining an active query must not replace the original snapshot.
    node_library.search_edit.setText("gaussian blur")
    node_library.search_edit.clear()

    after = {
        id(item): item.isExpanded()
        for item in (
            filtering,
            filtering_subgroup,
            segmentation,
            segmentation_subgroup,
        )
    }
    assert after == before


def test_compact_rail_has_full_accessible_labels_and_only_global_search(
    node_library,
    qtbot,
):
    groups = grouped_palette_specs()
    node_library.set_compact(True)

    assert node_library.is_compact
    assert node_library.search_edit.placeholderText() == "Find a node to add…"
    assert len(node_library.findChildren(QLineEdit)) == 1
    rail_layout = node_library.compact_rail.layout()
    assert rail_layout.itemAt(0).widget() is node_library.compact_rail.expand_button
    assert rail_layout.itemAt(1).widget() is node_library.compact_rail.search_button
    assert not node_library.compact_rail.findChildren(QLineEdit)
    assert node_library.compact_rail.search_button.accessibleName() == (
        "Search all nodes"
    )
    assert node_library.compact_rail.search_button.toolTip() == (
        "Search all available workflow nodes"
    )

    buttons = tuple(node_library.compact_rail.category_buttons)
    assert len(buttons) == len(groups)
    for button, (category, subgroups) in zip(buttons, groups.items(), strict=True):
        count = sum(len(specs) for specs in subgroups.values())
        assert isinstance(button, QToolButton)
        assert button.accessibleName() == f"{category}, {count} nodes"
        assert button.toolTip() == f"{category} — {count} nodes"
        assert not button.icon().isNull()

    filtering_button = next(
        button for button in buttons if button.property("vipp_category") == "Filtering"
    )
    node_library.open_category("Filtering", filtering_button)
    qtbot.waitUntil(node_library.popup.isVisible)

    assert node_library.popup.title_label.text() == "Filtering"
    assert not node_library.popup.search_host.isVisible()
    assert not node_library.popup.findChildren(QLineEdit)
    assert _operation_item(node_library.popup.palette, "gaussian_blur").text(0) == (
        "Gaussian Blur"
    )

    node_library.popup.close()
    qtbot.waitUntil(lambda: not node_library.popup.isVisible())
    node_library.open_global_search(node_library.compact_rail.search_button)
    qtbot.waitUntil(node_library.popup.isVisible)

    assert node_library.popup.title_label.text() == "Search all nodes"
    assert node_library.popup.search_host.isVisible()
    assert node_library.popup._search_edit is node_library.search_edit  # noqa: SLF001
    assert len(node_library.findChildren(QLineEdit)) == 1


def test_node_library_uses_shared_toggle_with_action_direction(node_library):
    collapse = node_library.collapse_button
    expand = node_library.compact_rail.expand_button

    assert isinstance(collapse, SidePanelToggleButton)
    assert isinstance(expand, SidePanelToggleButton)
    assert collapse.size() == expand.size()
    assert collapse._direction() == -1  # noqa: SLF001
    assert expand._direction() == 1  # noqa: SLF001
    assert collapse.accessibleName() == "Collapse node library"
    assert expand.accessibleName() == "Expand node library"


def test_operation_tooltips_explain_nodes_and_are_accessible(node_library):
    image_source = _operation_item(node_library.palette, "input")
    gaussian = _operation_item(node_library.palette, "gaussian_blur")

    assert image_source.toolTip(0) == (
        "Load an image, image collection, or supported microscopy dataset."
    )
    assert "Gaussian blur" in gaussian.toolTip(0)
    assert gaussian.data(0, Qt.AccessibleDescriptionRole) == gaussian.toolTip(0)
    assert not gaussian.toolTip(0).startswith("Add ")

    for item in node_library.palette._operation_items:  # noqa: SLF001
        assert item.toolTip(0)
        assert item.data(0, Qt.AccessibleDescriptionRole) == item.toolTip(0)
        assert item.toolTip(0) != f"Add {item.text(0)}"


def test_palette_context_menu_expands_collapses_and_adds_node(node_library):
    tree = node_library.palette
    gaussian = _operation_item(tree, "gaussian_blur")
    requested: list[str] = []
    node_library.operation_requested.connect(requested.append)

    node_menu = tree._context_menu_for_item(gaussian)  # noqa: SLF001
    node_actions = {
        action.text(): action
        for action in node_menu.actions()
        if not action.isSeparator()
    }
    assert tuple(node_actions) == (
        "Add to workflow",
        "Expand all",
        "Collapse all",
    )
    node_actions["Add to workflow"].trigger()
    assert requested == ["gaussian_blur"]

    blank_menu = tree._context_menu_for_item(None)  # noqa: SLF001
    blank_actions = {
        action.text(): action
        for action in blank_menu.actions()
        if not action.isSeparator()
    }
    assert tuple(blank_actions) == ("Expand all", "Collapse all")

    blank_actions["Collapse all"].trigger()
    assert all(
        not tree.topLevelItem(index).isExpanded()
        for index in range(tree.topLevelItemCount())
    )
    blank_actions["Expand all"].trigger()
    assert all(
        tree.topLevelItem(index).isExpanded()
        for index in range(tree.topLevelItemCount())
        if tree.topLevelItem(index).childCount()
    )


def test_context_menu_open_path_executes_and_releases_menu(
    node_library,
    qtbot,
    monkeypatch,
):
    captured: list[tuple[QMenu, QPoint]] = []
    monkeypatch.setattr(
        "napari_vipp.ui.palette._exec_menu",
        lambda menu, position: captured.append((menu, position)),
    )

    node_library.palette._open_context_menu(QPoint(-1, -1))  # noqa: SLF001

    assert len(captured) == 1
    menu, _position = captured[0]
    assert [action.text() for action in menu.actions()] == [
        "Expand all",
        "Collapse all",
    ]
    qtbot.waitUntil(lambda: not isalive(menu))


def test_context_menu_exec_supports_legacy_qt_spelling():
    class LegacyMenu:
        def exec_(self, position):
            return ("legacy", position)

    position = QPoint(4, 7)
    assert _exec_menu(LegacyMenu(), position) == ("legacy", position)


def test_expanded_and_compact_palette_emit_and_drag_the_same_operation(
    node_library,
    qtbot,
):
    operation_id = "gaussian_blur"
    requested: list[str] = []
    node_library.operation_requested.connect(requested.append)

    expanded_item = _operation_item(node_library.palette, operation_id)
    expanded_mime = node_library.palette.mimeData([expanded_item])
    node_library.palette._on_item_double_clicked(expanded_item, 0)  # noqa: SLF001

    node_library.set_compact(True)
    filtering_button = next(
        button
        for button in node_library.compact_rail.category_buttons
        if button.property("vipp_category") == "Filtering"
    )
    node_library.open_category("Filtering", filtering_button)
    qtbot.waitUntil(node_library.popup.isVisible)
    compact_item = _operation_item(node_library.popup.palette, operation_id)
    compact_mime = node_library.popup.palette.mimeData([compact_item])
    node_library.popup.palette._on_item_double_clicked(compact_item, 0)  # noqa: SLF001

    assert requested == [operation_id, operation_id]
    assert bytes(expanded_mime.data(OPERATION_MIME)) == operation_id.encode()
    assert bytes(compact_mime.data(OPERATION_MIME)) == operation_id.encode()


def test_compact_transition_moves_focus_and_hiding_closes_popup(
    node_library,
    qtbot,
):
    node_library.search_edit.setFocus()
    qtbot.waitUntil(lambda: QApplication.focusWidget() is node_library.search_edit)

    node_library.set_compact(True)
    assert QApplication.focusWidget() is node_library.compact_rail.search_button

    image_button = next(iter(node_library.compact_rail.category_buttons))
    node_library.open_category("Image Data", image_button)
    qtbot.waitUntil(node_library.popup.isVisible)
    node_library.hide()

    assert not node_library.popup.isVisible()
    assert node_library.search_edit.parent() is node_library.search_host


@pytest.mark.parametrize(
    ("operation_id", "expected_kind"),
    [
        ("morphological_gradient", "contours"),
        ("rescale_intensity", "sun"),
        ("add_metadata_columns", "table"),
        ("filter_labels_by_volume", "tag"),
        ("subtract_background", "waves"),
    ],
)
def test_operation_icons_use_stable_ids_then_category_fallback(
    operation_id,
    expected_kind,
):
    specs = {
        spec.id: spec
        for subgroups in grouped_palette_specs().values()
        for group_specs in subgroups.values()
        for spec in group_specs
    }

    assert operation_icon_kind(specs[operation_id]) == expected_kind
