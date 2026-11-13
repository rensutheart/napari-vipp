from __future__ import annotations

import pytest
from qtpy.QtCore import QPoint, Qt, QTimer
from qtpy.QtGui import QColor, QFont, QPalette
from qtpy.QtWidgets import QLabel, QStyle, QStyleOptionSpinBox, QVBoxLayout, QWidget

from napari_vipp import _widget
from napari_vipp.ui.view_dims import (
    ViewDimAxis,
    ViewDimAxisControl,
    ViewDimsBar,
)


@pytest.mark.parametrize("points", [9, 16])
def test_view_dimensions_tracks_match_and_fields_fit_the_range(qtbot, points):
    parent = QWidget()
    qtbot.addWidget(parent)
    parent.resize(1412, 80)
    parent.setFont(QFont("Arial", points))
    # Exercise the app theme's general purpose input minimum as well as
    # deliberately unequal axis chrome (labels, numeric ranges and fields).
    parent.setStyleSheet("QSpinBox { min-width: 70px; padding: 4px 8px; }")
    bar = ViewDimsBar(parent)
    bar.resize(1400, 56)
    axes = (
        ViewDimAxis("time", "T", 0, 12, 3),
        ViewDimAxis("depth", "Z", 1, 1024, 512),
        ViewDimAxis("channel", "Ch", 2, 4, 0),
    )
    bar.set_axes(axes)
    parent.show()

    fields = [control.spin for control in bar._controls]

    def settled_layout(mode, expected_widths=None):
        assert bar._responsive_mode == mode
        for control in bar._controls:
            assert not control._metric_refresh.isActive()
            field = control.spin
            assert field.minimumWidth() == field.maximumWidth() == field.width()
        if mode == "full":
            lengths = [control.slider.width() for control in bar._controls]
            assert min(lengths) >= 120 and max(lengths) - min(lengths) <= 1, lengths
        if expected_widths is not None:
            assert [field.width() for field in fields] == expected_widths

    for width in (1400, 1250, 1400):
        bar.resize(width, 56)
        qtbot.waitUntil(lambda: settled_layout("full"))

    widths = [field.width() for field in fields]
    assert widths[2] < widths[0] < widths[1] < 80
    for field in fields:
        field.setValue(field.maximum())

    def assert_editors_fit():
        for field in fields:
            editor = field.lineEdit()
            margins = editor.textMargins()
            available = editor.contentsRect().width() - margins.left() - margins.right()
            text_width = editor.fontMetrics().horizontalAdvance(editor.text())
            assert available >= text_width + 4
            # Preserve the native edit field without adding arrow targets.
            option = QStyleOptionSpinBox()
            field.initStyleOption(option)
            for button in (QStyle.SC_SpinBoxUp, QStyle.SC_SpinBoxDown):
                assert field.style().subControlRect(
                    QStyle.CC_SpinBox, option, button, field
                ).isEmpty()

    qtbot.waitUntil(lambda: settled_layout("full", widths))
    assert_editors_fit()
    for width, mode in ((530, "compact"), (1400, "full")):
        bar.resize(width, 56)
        qtbot.waitUntil(lambda mode=mode: settled_layout(mode, widths))
        assert_editors_fit()


def test_widget_module_reexports_extracted_view_dimension_controls():
    assert _widget.ViewDimAxis is ViewDimAxis
    assert _widget.ViewDimAxisControl is ViewDimAxisControl
    assert _widget.ViewDimsBar is ViewDimsBar


def test_view_dimension_control_synchronizes_values_and_emits_semantic_axis(qtbot):
    control = ViewDimAxisControl()
    qtbot.addWidget(control)
    captured: list[tuple[int, int]] = []
    control.value_changed.connect(
        lambda step_axis, value: captured.append((step_axis, value))
    )

    control.set_axis(ViewDimAxis("z", "Z", step_axis=2, size=5, value=9))

    assert captured == []
    assert control.slider.value() == 4
    assert control.spin.value() == 4
    assert control.range_label.text() == "/4"
    assert control.slider.accessibleName() == "z view slider"
    assert control.spin.accessibleName() == "z view position"

    control.spin.setValue(1)

    assert control.slider.value() == 1
    assert captured == [(2, 1)]


def test_view_dimension_entry_stays_arrowless_and_keyboard_editable(qtbot):
    control = ViewDimAxisControl()
    qtbot.addWidget(control)
    control.set_axis(ViewDimAxis("depth", "Z", step_axis=2, size=1024, value=20))
    control.show()
    spin = control.spin
    editor = spin.lineEdit()
    option = QStyleOptionSpinBox()
    spin.initStyleOption(option)
    for edge in (0, spin.width() - 1):
        for height in (spin.height() // 4, 3 * spin.height() // 4):
            pos = QPoint(edge, height)
            assert spin.style().hitTestComplexControl(
                QStyle.CC_SpinBox, option, pos, spin
            ) not in (QStyle.SC_SpinBoxUp, QStyle.SC_SpinBoxDown)
            qtbot.mouseClick(spin, Qt.LeftButton, pos=pos)
    assert spin.value() == control.slider.value() == 20

    emitted = []
    control.value_changed.connect(lambda *args: emitted.append(args))
    editor.setFocus()
    editor.selectAll()
    qtbot.keyClicks(editor, "512")
    qtbot.keyClick(editor, Qt.Key_Return)
    assert spin.value() == control.slider.value() == 512
    assert emitted[-1] == (2, 512)
    qtbot.keyClick(editor, Qt.Key_Up)
    assert spin.value() == control.slider.value() == 513
    assert emitted[-1] == (2, 513)


def test_view_dimensions_menu_builds_full_controls_and_forwards_values(qtbot):
    bar = ViewDimsBar()
    qtbot.addWidget(bar)

    bar._populate_menu()
    empty_action = bar.menu.actions()[0]
    assert empty_action.text() == "No view dimensions"
    assert not empty_action.isEnabled()

    bar.set_axes((ViewDimAxis("time", "T", step_axis=0, size=3, value=1),))
    captured: list[tuple[int, int]] = []
    bar.value_changed.connect(
        lambda step_axis, value: captured.append((step_axis, value))
    )
    bar._populate_menu()

    action = bar.menu.actions()[0]
    container = action.defaultWidget()
    controls = container.findChildren(ViewDimAxisControl)
    assert len(controls) == 1
    assert not controls[0].slider.isHidden()
    labels = [label.text() for label in container.findChildren(QLabel)]
    assert "View dimensions" in labels
    assert any("changes the view, not node settings" in text for text in labels)

    controls[0].spin.setValue(2)

    assert captured == [(0, 2)]


def test_view_dimensions_label_stays_at_start_as_sliders_move_to_menu(qtbot):
    parent = QWidget()
    qtbot.addWidget(parent)
    parent.setFont(QFont("Arial", 9))
    bar = ViewDimsBar(parent)
    bar.set_axes(
        (
            ViewDimAxis("time", "T", step_axis=0, size=5, value=1),
            ViewDimAxis("channel", "C", step_axis=1, size=3, value=0),
            ViewDimAxis("z", "Z", step_axis=2, size=12, value=5),
        )
    )
    parent.show()

    for width, mode in ((1000, "full"), (480, "compact"), (190, "menu")):
        parent.resize(width + 12, 100)
        bar.resize(width, 40)
        bar.sync_responsive_mode()
        bar.layout().activate()

        assert bar._responsive_mode == mode
        heading = bar.title_label if mode == "full" else bar.menu_button
        caption = "Show sliders" if mode == "compact" else "View dimensions"
        assert heading.text() == caption
        assert not heading.isHidden()
        assert bar.title_label.isHidden() == (mode != "full")
        assert bar.menu_button.isHidden() == (mode == "full")
        if mode != "menu":
            assert heading.geometry().right() < bar._controls[0].geometry().left()
        if mode != "full":
            assert bar.menu_button.accessibleName() == caption
            assert "not node settings" in bar.menu_button.accessibleDescription()
        if mode == "compact":
            assert "same dimensions" in bar.menu_button.toolTip()
            assert all(
                not control.spin.isHidden() and control.slider.isHidden()
                for control in bar._controls
            )
            bar._populate_menu()
            popup = bar.menu.actions()[0].defaultWidget()
            popup_axes = popup.findChildren(ViewDimAxisControl)
            assert [
                (control.label.text(), control.spin.value(), control.spin.maximum())
                for control in popup_axes
            ] == [
                (control.label.text(), control.spin.value(), control.spin.maximum())
                for control in bar._controls
            ]
            assert all(not control.slider.isHidden() for control in popup_axes)

    opened = []

    def close_menu():
        opened.append(True)
        QTimer.singleShot(0, bar.menu.close)

    bar.menu.aboutToShow.connect(close_menu)
    bar.menu_button.setFocus()
    assert bar.menu_button.focusPolicy() == Qt.StrongFocus
    qtbot.keyClick(bar.menu_button, Qt.Key_Space)
    assert opened == [True]


def test_view_dimensions_use_menu_before_compact_controls_overlap(qtbot):
    parent = QWidget()
    parent.resize(272, 80)
    bar = ViewDimsBar(parent)
    qtbot.addWidget(bar)
    bar.resize(260, 40)
    bar.set_axes(
        tuple(
            ViewDimAxis(name, label, step_axis=index, size=size, value=0)
            for index, (name, label, size) in enumerate(
                (
                    ("time", "T", 120),
                    ("channel", "C", 81),
                    ("depth", "Z", 12),
                    ("position", "P", 5),
                )
            )
        )
    )

    assert bar._responsive_mode == "menu"
    assert not bar.menu_button.isHidden()
    assert all(control.isHidden() for control in bar._controls)


def test_large_font_narrow_view_dimensions_keep_all_axes_in_the_full_menu(qtbot):
    parent = QWidget()
    qtbot.addWidget(parent)
    parent.setFont(QFont("Arial", 16))
    layout = QVBoxLayout(parent)
    bar = ViewDimsBar()
    layout.addWidget(bar)
    axes = (
        ViewDimAxis("time", "T", step_axis=0, size=12, value=3),
        ViewDimAxis("depth", "Z", step_axis=1, size=1024, value=512),
        ViewDimAxis("channel", "C", step_axis=2, size=4, value=0),
    )
    emitted = []
    bar.value_changed.connect(lambda *value: emitted.append(value))
    bar.set_axes(axes)
    parent.resize(180, 250)
    parent.show()
    button = bar.menu_button

    def menu_caption_ready(points):
        assert bar._responsive_mode == "menu"
        assert button.text() == "View dimensions"
        assert button.font().pointSizeF() == points
        assert not bar._launcher_font_refresh.isActive()

    def required_button_width():
        return max(button.sizeHint().width(), button.minimumSizeHint().width())

    for points in (16, 9, 16):
        parent.setFixedWidth(180)
        parent.setFont(QFont("Arial", points))
        # Budget a real dock for the menu-only caption, not the shorter
        # compact caption or its size before the requested font takes effect.
        qtbot.waitUntil(lambda points=points: menu_caption_ready(points))
        margins = (layout.contentsMargins(), bar.layout().contentsMargins())
        insets = sum(margin.left() + margin.right() for margin in margins)
        parent.setFixedWidth(required_button_width() + insets + 12)
        # Repeated presentation refreshes must not re-show any inline control
        # while the complete menu fallback is still the selected mode.
        bar.set_axes(axes)
        bar.set_axes(axes)

        def launcher_fits(points=points):
            menu_caption_ready(points)
            assert all(control.isHidden() for control in bar._controls)
            assert bar.title_label.isHidden()
            assert button.isVisible()
            assert bar.contentsRect().contains(button.geometry())
            assert button.width() >= required_button_width()
            text_width = button.fontMetrics().horizontalAdvance(button.text())
            assert button.width() >= text_width + 26  # indicator and trailing space

        qtbot.waitUntil(launcher_fits)

    bar._populate_menu()
    controls = bar.menu.actions()[0].defaultWidget().findChildren(ViewDimAxisControl)
    assert len(controls) == 3
    assert [control.spin.value() for control in controls] == [3, 512, 0]
    assert [control.spin.maximum() for control in controls] == [11, 1023, 3]
    assert [control.range_label.text() for control in controls] == [
        "/11",
        "/1023",
        "/3",
    ]
    assert all(not control.slider.isHidden() for control in controls)
    assert controls[2].spin.maximumWidth() < controls[1].spin.maximumWidth()
    assert emitted == []


def test_view_dimension_launcher_tracks_new_font_host(qtbot, qapp):
    original = QFont(qapp.font())
    first, second = QWidget(), QWidget()
    for host in (first, second):
        qtbot.addWidget(host)
        host.resize(225, 100)
        host.show()
    bar = ViewDimsBar(first)
    bar.resize(213, 40)
    bar.set_axes((ViewDimAxis("z", "Z", 0, 12, 5),))
    button = bar.menu_button

    def font_is_current(points):
        assert button.font().pointSizeF() == points
        assert not bar._launcher_font_refresh.isActive()

    try:
        for points in (16, 9, 16):
            qapp.setFont(QFont("Arial", points))
            qtbot.waitUntil(lambda points=points: font_is_current(points))
        second.setFont(QFont("Arial", 13))
        bar.setParent(second)
        bar.show()
        qtbot.waitUntil(lambda: font_is_current(13))
        first.setFont(QFont("Arial", 11))
        assert not bar._launcher_font_refresh.isActive()
        font_is_current(13)
        second.setFont(QFont("Arial", 15))
        qtbot.waitUntil(lambda: font_is_current(15))
    finally:
        qapp.setFont(original)


def _theme_palette(*, base: str, alternate: str, text: str) -> QPalette:
    palette = QPalette()
    palette.setColor(QPalette.Base, QColor(base))
    palette.setColor(QPalette.AlternateBase, QColor(alternate))
    palette.setColor(QPalette.Text, QColor(text))
    return palette


def test_menu_button_draws_visible_chevron_in_light_and_dark_palettes(qtbot):
    bar = ViewDimsBar()
    qtbot.addWidget(bar)
    bar.resize(240, 40)
    bar.set_axes((ViewDimAxis("z", "Z", step_axis=0, size=12, value=5),))
    bar.show()
    button = bar.menu_button
    assert not button.isHidden()

    for base, alternate, text in (
        ("#ffffff", "#f2f4f7", "#111827"),
        ("#111827", "#1f2937", "#f8fafc"),
    ):
        bar.setPalette(_theme_palette(base=base, alternate=alternate, text=text))
        image = button.grab().toImage()
        scale = image.devicePixelRatio()
        midpoint = image.height() // 2
        background = image.pixelColor(round(8 * scale), midpoint)
        visible_pixels = []
        for x in range(
            image.width() - round(20 * scale), image.width() - round(6 * scale)
        ):
            for y in range(midpoint - round(5 * scale), midpoint + round(5 * scale)):
                pixel = image.pixelColor(x, y)
                if abs(pixel.lightness() - background.lightness()) >= 80:
                    visible_pixels.append((x, y))
        # Verify paint in the space beyond the label, not a glyph in button text.
        assert len(visible_pixels) >= 6
        columns, rows = zip(*visible_pixels, strict=True)
        assert max(columns) - min(columns) >= round(5 * scale)
        assert max(rows) - min(rows) >= round(2 * scale)


def test_view_dimension_controls_follow_runtime_palette_changes(qtbot):
    control = ViewDimAxisControl()
    bar = ViewDimsBar()
    qtbot.addWidget(control)
    qtbot.addWidget(bar)

    light = _theme_palette(base="#ffffff", alternate="#f2f4f7", text="#111827")
    control.setPalette(light)
    bar.setPalette(light)

    assert "#111827" in control.label.styleSheet()
    assert "#f2f4f7" in bar.styleSheet()
    assert "#111827" in bar.title_label.styleSheet()

    dark = _theme_palette(base="#111827", alternate="#1f2937", text="#f8fafc")
    control.setPalette(dark)
    bar.setPalette(dark)

    assert "#f8fafc" in control.label.styleSheet()
    assert "#1f2937" in bar.styleSheet()
    assert "#f8fafc" in bar.title_label.styleSheet()
