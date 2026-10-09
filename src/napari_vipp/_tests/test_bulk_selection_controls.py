"""Paired selection actions share presentation without sharing selection policy."""

from __future__ import annotations

import pytest
from qtpy.QtCore import QEvent, QObject, QPoint, QRect, QSize, Qt
from qtpy.QtGui import QColor, QFont, QIcon, QPalette
from qtpy.QtWidgets import QApplication, QLabel, QLineEdit, QVBoxLayout, QWidget

from napari_vipp._tests.test_batch_override_reset_actions import _editor
from napari_vipp._tests.test_batch_table_theme import _palette
from napari_vipp._tests.test_results_workspace_dialog import _dialog as _results
from napari_vipp._tests.test_statistics_bulk_selection import _panel as _statistics
from napari_vipp._tests.test_ui_batch import _actions, _preview_result
from napari_vipp.ui.axis_controls import SelectTableColumnsControl
from napari_vipp.ui.batch import CollectionBatchDialog
from napari_vipp.ui.bulk_selection import BulkSelectionControls
from napari_vipp.ui.toolbar_controls import ToolbarCommandButton, toolbar_icon

SELECTORS = (
    "items", "samples", "parameters", "keep_columns", "measurements", "visible_columns"
)


def _selector(qtbot, tmp_path, case):
    if case == "items":
        plan = _preview_result(tmp_path, count=120)
        owner = CollectionBatchDialog(actions=_actions(plan, []))
        qtbot.addWidget(owner)
        owner.apply_preview_result(plan, preview_representative=False)
        owner.tabs.setCurrentIndex(1)
        pair = (owner.select_all_items_button, owner.deselect_all_items_button)
        target = owner.preview_table
    elif case in {"samples", "parameters"}:
        editor = _editor(qtbot, count=120)
        owner = editor if case == "samples" else editor.create_column_chooser()
        if owner is not editor:
            qtbot.addWidget(owner)
        if case == "samples":
            pair = (editor.select_matching_button, editor.clear_selection_button)
            target = editor.table
        else:
            controls = owner.findChildren(BulkSelectionControls)
            assert len(controls) == 1
            pair = (controls[0].select_all_button, controls[0].deselect_all_button)
            target = owner.parameter_tree
    elif case == "keep_columns":
        owner = SelectTableColumnsControl(["label_id", "area", "intensity"])
        qtbot.addWidget(owner)
        pair = (owner.select_all_button, owner.deselect_all_button)
        target = owner.list_widget
    elif case == "measurements":
        owner = _statistics(qtbot, value_columns="area", group_by="condition")
        pair = (
            owner.select_all_measurements_button, owner.select_no_measurements_button
        )
        target = owner.measurements
    else:
        owner = _results(qtbot)
        pair = (owner.select_all_columns_button, owner.select_no_columns_button)
        target = owner.column_list
    controls = next(
        controls for controls in owner.findChildren(BulkSelectionControls)
        if controls.select_all_button is pair[0]
    )
    return owner, controls, target


def _rectangle(widget, owner):
    return QRect(widget.mapTo(owner, QPoint()), widget.size())


def _assert_compact_measurement_scope(panel):
    controls = panel.measurement_selection_controls
    label = controls.scope_label
    assert label.hasHeightForWidth()
    assert label.height() == label.heightForWidth(label.width())
    pair_bottom = max(
        _rectangle(button, controls).bottom()
        for button in (controls.select_all_button, controls.deselect_all_button)
    )
    assert label.geometry().top() - pair_bottom - 1 == 6
    assert (
        _rectangle(panel.measurements, panel).top()
        - _rectangle(controls, panel).bottom() - 1
    ) == 7


def _selection_state(owner, case):
    if case == "items":
        return frozenset(owner._checked_items), owner._preview_result
    if case == "samples":
        return owner.selected_source_keys(), owner.overrides()
    if case == "parameters":
        return tuple(
            item.checkState(0) for item in owner.parameter_items.values()
        )
    if case == "keep_columns":
        return owner.value()
    if case == "measurements":
        return dict(owner.params), owner._checked(owner.measurements)
    return owner.data_panel.table, tuple(
        owner.column_list.item(index).checkState()
        for index in range(owner.column_list.count())
    )


@pytest.mark.parametrize("case", SELECTORS)
def test_six_selectors_share_full_labels_outline_icons_and_actions_above_choices(
    qtbot, tmp_path, case
):
    owner, controls, target = _selector(qtbot, tmp_path, case)
    owner.resize(1100 if case == "visible_columns" else 760, 1000)
    owner.show()
    qtbot.wait(20)
    buttons = (controls.select_all_button, controls.deselect_all_button)
    assert tuple(button.text() for button in buttons) == ("Select all", "Deselect all")
    for button, kind in zip(buttons, ("select_all", "deselect"), strict=True):
        assert isinstance(button, ToolbarCommandButton)
        assert button.iconSize() == QSize(18, 18)
        assert not button.icon().isNull()
        assert button.toolTip()
        assert button.icon().pixmap(18, 18, QIcon.Normal).toImage() == (
            toolbar_icon(kind, button.palette()).pixmap(18, 18, QIcon.Normal).toImage()
        )
        assert button.width() >= button.minimumSizeHint().width()
    assert not _rectangle(buttons[0], controls).intersects(
        _rectangle(buttons[1], controls)
    )
    assert _rectangle(controls, owner).bottom() < _rectangle(target, owner).top()
    assert controls.scope_label.text().strip()
    assert controls.scope_label.isVisible()


@pytest.mark.parametrize("case", SELECTORS)
def test_shared_selector_icons_follow_palette_changes_without_selection_edits(
    qtbot, tmp_path, case
):
    from napari.qt import get_stylesheet

    owner, controls, _target = _selector(qtbot, tmp_path, case)
    before = _selection_state(owner, case)
    owner.show()
    icons = []
    for dark in (False, True):
        palette = _palette(dark)
        palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#87909c"))
        owner.setPalette(palette)
        owner.setStyleSheet(get_stylesheet("dark" if dark else "light"))
        # Results applies descendant styles in a queued theme refresh; icons
        # then follow those palettes through their own coalesced refresh.
        qtbot.waitUntil(
            lambda: all(
                button.icon().pixmap(18, 18, mode).toImage()
                == toolbar_icon(kind, button.palette()).pixmap(18, 18, mode).toImage()
                for button, kind in (
                    (controls.select_all_button, "select_all"),
                    (controls.deselect_all_button, "deselect"),
                )
                for mode in (QIcon.Normal, QIcon.Disabled)
            ),
            timeout=3000,
        )
        pair = []
        for button, kind in (
            (controls.select_all_button, "select_all"),
            (controls.deselect_all_button, "deselect"),
        ):
            for mode in (QIcon.Normal, QIcon.Disabled):
                actual = button.icon().pixmap(18, 18, mode).toImage()
                expected = toolbar_icon(kind, button.palette()).pixmap(
                    18, 18, mode
                ).toImage()
                assert actual == expected
            pair.append(button.icon().pixmap(18, 18, QIcon.Normal).toImage())
        icons.append(pair)
        assert _selection_state(owner, case) == before
    assert icons[0] != icons[1]


@pytest.mark.parametrize("font_size", [10, 14])
@pytest.mark.parametrize("width", [260, 420])
def test_paired_column_actions_wrap_without_forcing_a_wider_inspector(
    qtbot, width, font_size
):
    from napari.qt import get_stylesheet

    owner = SelectTableColumnsControl(["label_id", "area", "intensity"])
    qtbot.addWidget(owner)
    owner.setFont(QFont("Segoe UI", font_size))
    owner.setStyleSheet(
        get_stylesheet("dark") + f"\nQWidget {{ font-size: {font_size}pt; }}"
    )
    owner.resize(width, 780)
    owner.show()
    qtbot.wait(20)
    assert owner.width() == width
    controls = owner.findChild(BulkSelectionControls)
    first = _rectangle(controls.select_all_button, controls)
    second = _rectangle(controls.deselect_all_button, controls)
    for button in (controls.select_all_button, controls.deselect_all_button):
        assert controls.rect().contains(_rectangle(button, controls))
        assert button.width() >= button.minimumSizeHint().width()
    assert not first.intersects(second)
    gap = second.left() - first.right() - 1 if first.y() == second.y() else (
        second.top() - first.bottom() - 1
    )
    assert gap == 6
    assert controls.scope_label.geometry().top() > max(first.bottom(), second.bottom())


def test_wrapped_pair_keeps_complete_scope_clear_of_buttons_and_choices(qtbot):
    owner = SelectTableColumnsControl(["label_id", "area", "intensity"])
    qtbot.addWidget(owner)
    owner.setStyleSheet("QWidget { font-size: 20px; } QPushButton { padding: 5px; }")
    owner.resize(owner.sizeHint().width() * 2, 780)
    owner.show()
    qtbot.wait(20)
    controls = owner.column_selection_controls
    outer_width = owner.width() - controls.button_row.width()
    pair_width = controls.button_layout.sizeHint().width()
    narrow = pair_width - 1 + outer_width
    wide = pair_width * 2 + outer_width
    assert narrow >= owner.minimumSizeHint().width()
    heights = []
    for width in (narrow, wide, narrow):
        owner.resize(width, 780)
        qtbot.wait(20)
        first = _rectangle(controls.select_all_button, controls)
        second = _rectangle(controls.deselect_all_button, controls)
        label = controls.scope_label
        if width == narrow:
            assert second.top() > first.bottom()
        else:
            assert first.top() == second.top()
        assert label.geometry().top() - max(first.bottom(), second.bottom()) - 1 == 6
        assert label.height() == label.heightForWidth(label.width())
        assert (
            _rectangle(controls, owner).bottom()
            < _rectangle(owner.list_widget, owner).top()
        )
        heights.append(controls.height())
    assert heights[0] > heights[1]
    assert heights[2] == heights[0]


@pytest.mark.parametrize("width", [260, 420])
def test_numeric_measurement_scope_wraps_completely_in_a_narrow_inspector(qtbot, width):
    from napari.qt import get_stylesheet

    panel = _statistics(qtbot, value_columns="area", group_by="condition")
    panel.resize(width, 1600)
    label = panel.measurement_selection_controls.scope_label
    heights = []
    for font_size in (10, 14, 10):
        panel.setStyleSheet(
            get_stylesheet("dark") + f"\nQWidget {{ font-size: {font_size}pt; }}"
        )
        qtbot.wait(20)
        _assert_compact_measurement_scope(panel)
        assert "Select: eligible numeric measurements" in label.text()
        assert "Deselect: all selected measurements" in label.text()
        heights.append(label.height())
    assert heights[1] >= heights[0]
    assert heights[2] == heights[0]


def test_scope_height_timer_commits_sibling_geometry_in_the_same_turn(qtbot):
    owner = QWidget()
    layout = QVBoxLayout(owner)
    layout.setAlignment(Qt.AlignTop)
    layout.setSpacing(7)
    controls = BulkSelectionControls(scope="Choose measurements.")
    choices = QLabel("Measurement choices")
    choices.setFixedHeight(40)
    layout.addWidget(controls)
    layout.addWidget(choices)
    qtbot.addWidget(owner)
    owner.resize(260, 700)
    owner.show()
    qtbot.waitUntil(lambda: controls.isVisible() and choices.y() > controls.y())
    controls._sync_height()
    layout.activate()
    previous_height = controls.height()

    controls.set_scope(
        "Select: eligible numeric measurements. "
        "Deselect: all selected measurements. "
        "Grouping and identity columns are excluded from automatic selection."
    )
    # Exercise the production timer callback without processing the parent's
    # queued LayoutRequest. The callback must not leave the next row at its
    # old y coordinate while its own fixed height has already grown.
    controls._sync_height()
    assert controls.height() > previous_height
    assert (
        _rectangle(choices, owner).top()
        - _rectangle(controls, owner).bottom() - 1
    ) == 7


@pytest.mark.parametrize("change", ("show", "scope", "font", "width"))
def test_bulk_geometry_timer_quiesces_after_layout_changes(
    qtbot, record_property, change,
):
    owner = QWidget()
    layout = QVBoxLayout(owner)
    layout.setAlignment(Qt.AlignTop)
    layout.setSpacing(7)
    controls = BulkSelectionControls(scope="Choose measurements.")
    following = QLabel("Measurement choices")
    following.setFixedHeight(40)
    layout.addWidget(controls)
    layout.addWidget(following)
    qtbot.addWidget(owner)

    class RequestCounter(QObject):
        requests = 0

        def eventFilter(self, watched, event):  # noqa: N802
            if watched is controls and event.type() == QEvent.LayoutRequest:
                self.requests += 1
            return False

    counter = RequestCounter(controls)
    controls.installEventFilter(counter)
    timeouts = []
    controls._geometry_timer.timeout.connect(lambda: timeouts.append(1))

    def dispatch_turns(count):
        # Drain a bounded number of posted-layout/timer turns, not a timed
        # sleep. A stable control must eventually stop creating more work.
        for _ in range(count):
            QApplication.sendPostedEvents(None, QEvent.LayoutRequest)
            QApplication.processEvents()

    owner.resize(340, 700)
    owner.show()
    dispatch_turns(12)
    if change == "scope":
        controls.set_scope(
            "Select: eligible numeric measurements. "
            "Deselect: all selected measurements. "
            "Grouping and identity columns are excluded from automatic selection."
        )
    elif change == "font":
        owner.setFont(QFont("Segoe UI", 14))
    elif change == "width":
        owner.resize(260, 700)
    dispatch_turns(12)
    before = (len(timeouts), counter.requests, controls.geometry())
    dispatch_turns(8)
    after = (len(timeouts), counter.requests, controls.geometry())
    record_property("idle_timeout_delta", after[0] - before[0])
    record_property("idle_layout_request_delta", after[1] - before[1])
    record_property("total_height_timeouts", after[0])
    record_property("total_layout_requests", after[1])
    assert after == before, (before, after)
    assert not controls._geometry_timer.isActive()
    assert controls.minimumHeight() == controls.maximumHeight() == controls.height()
    assert controls.scope_label.height() == controls.scope_label.heightForWidth(
        controls.scope_label.width()
    )
    assert (
        _rectangle(following, owner).top()
        - _rectangle(controls, owner).bottom() - 1
    ) == 7


def test_measurement_scope_reflows_and_shrinks_after_inspector_width_changes(qtbot):
    from napari.qt import get_stylesheet

    panel = _statistics(qtbot, value_columns="area", group_by="condition")
    panel.setStyleSheet(get_stylesheet("dark") + "\nQWidget { font-size: 14pt; }")
    controls = panel.measurement_selection_controls
    qtbot.wait(20)
    label = controls.scope_label
    outer_width = panel.width() - label.width()
    text_width = label.fontMetrics().horizontalAdvance(label.text())
    narrow_label_width = max(
        controls.button_layout.minimumSize().width(),
        panel.minimumSizeHint().width() - outer_width,
        panel.minimumWidth() - outer_width,
        text_width // 2,
        1,
    )
    wide_label_width = max(text_width, narrow_label_width + 1)
    for _attempt in range(10):
        if label.heightForWidth(wide_label_width) < label.heightForWidth(
            narrow_label_width
        ):
            break
        wide_label_width *= 2
    assert label.heightForWidth(wide_label_width) < label.heightForWidth(
        narrow_label_width
    )
    narrow = narrow_label_width + outer_width
    wide = wide_label_width + outer_width
    heights = []
    for width in (wide, narrow, wide):
        panel.resize(width, 1600)
        qtbot.wait(20)
        _assert_compact_measurement_scope(panel)
        heights.append(controls.height())
    assert heights[1] > heights[0]
    assert heights[2] == heights[0]


@pytest.mark.parametrize("dark", [False, True])
def test_measurement_scope_is_complete_when_opened_with_large_font(qtbot, dark):
    from napari.qt import get_stylesheet

    panel = _statistics(qtbot, value_columns="area", group_by="condition")
    panel.setFont(QFont("Segoe UI", 14))
    panel.setPalette(_palette(dark))
    panel.setStyleSheet(
        get_stylesheet("dark" if dark else "light")
        + "\nQWidget { font-family: Segoe UI; font-size: 14pt; }"
    )
    panel.resize(260, 1600)
    panel.show()
    qtbot.wait(40)
    _assert_compact_measurement_scope(panel)


def test_overrides_deselect_all_clears_filtered_out_checks_but_keeps_values(qtbot):
    editor = _editor(qtbot, count=120)
    checked = (editor._source_keys[0], editor._source_keys[-1])
    editor.set_selected_source_keys(checked)
    editor.apply_selected_values({("blur", "sigma"): 7.5}, source_keys=checked)
    before = editor.overrides()
    editor.sample_search.setText("119")
    controls = next(
        controls for controls in editor.findChildren(BulkSelectionControls)
        if controls.select_all_button is editor.select_matching_button
    )
    scope = controls.scope_label.text().lower()
    assert "matching" in scope
    assert "hidden" in scope
    assert controls.deselect_all_button.isEnabled()

    controls.deselect_all_button.click()

    assert editor.selected_source_keys() == ()
    assert editor.overrides() == before
    assert not controls.deselect_all_button.isEnabled()
    controls.select_all_button.click()
    assert editor.selected_source_keys() == (editor._source_keys[-1],)
    assert editor.overrides() == before


def test_parameter_column_pair_applies_all_columns_including_search_hidden(
    qtbot
):
    editor = _editor(qtbot)
    editor.select_all_matching()
    editor.apply_selected_values({("blur", "sigma"): 3.25})
    before = editor.overrides()
    chooser = editor.create_column_chooser()
    qtbot.addWidget(chooser)
    controls = chooser.findChild(BulkSelectionControls)
    search = chooser.findChild(QLineEdit)
    search.setText("sigma")
    assert chooser.parameter_items[("background", "radius")].isHidden()

    controls.deselect_all_button.click()
    assert all(
        item.checkState(0) == Qt.Unchecked for item in chooser.parameter_items.values()
    )
    controls.select_all_button.click()
    assert all(
        item.checkState(0) == Qt.Checked for item in chooser.parameter_items.values()
    )
    controls.deselect_all_button.click()
    chooser.accept()

    assert editor._visible_parameter_keys == set()
    assert editor.overrides() == before


def test_keep_columns_pair_retains_order_and_emits_one_atomic_edit(qtbot):
    owner = SelectTableColumnsControl(["label_id", "area", "intensity"])
    qtbot.addWidget(owner)
    owner.set_options(["label_id", "area", "intensity"], value="intensity,area")
    order = tuple(
        owner.list_widget.item(index).data(Qt.UserRole)
        for index in range(owner.list_widget.count())
    )
    edits = []
    owner.valueChanged.connect(edits.append)

    owner.deselect_all_button.click()
    owner.select_all_button.click()

    assert len(edits) == 2
    assert tuple(
        owner.list_widget.item(index).data(Qt.UserRole)
        for index in range(owner.list_widget.count())
    ) == order
    assert all(
        owner.list_widget.item(index).checkState() == Qt.Checked
        for index in range(owner.list_widget.count())
    )
