from __future__ import annotations

from qtpy.QtCore import QEvent
from qtpy.QtGui import QBrush, QColor, QPalette
from qtpy.QtWidgets import QApplication, QDialog

from napari_vipp._theme import category_color, category_foreground
from napari_vipp._widget import ExampleWorkflowDialog, ExampleWorkflowSpec
from napari_vipp.ui.dialogs import (
    ConnectionInsertCandidate,
    ConnectionInsertDialog,
)


def test_example_dialog_resolves_entries_from_its_supplied_catalog(qtbot):
    custom = ExampleWorkflowSpec(
        id="custom-example",
        category="Custom",
        title="Custom workflow",
        filename="custom.json",
        samples=("Custom sample",),
        description="A caller-supplied workflow entry.",
    )
    dialog = ExampleWorkflowDialog(examples=(custom,))
    qtbot.addWidget(dialog)

    dialog.select_example(custom.id)

    assert dialog.selected_example() is custom
    assert dialog.open_button.isEnabled()
    assert "Custom workflow" in dialog.details_label.text()

    dialog.open_button.click()

    assert dialog.result() == QDialog.Accepted


def _theme_palette(*, base: str, alternate: str, text: str) -> QPalette:
    palette = QPalette()
    palette.setColor(QPalette.Base, QColor(base))
    palette.setColor(QPalette.Window, QColor(base))
    palette.setColor(QPalette.AlternateBase, QColor(alternate))
    palette.setColor(QPalette.Text, QColor(text))
    palette.setColor(QPalette.WindowText, QColor(text))
    return palette


def test_insert_dialog_alternating_rows_follow_runtime_palette(qtbot):
    candidate = ConnectionInsertCandidate(
        operation_id="gaussian-blur",
        title="Gaussian Blur",
        category="Filtering",
        subcategory="",
        mode="full",
        detail="Smooth an image.",
        search_text="gaussian blur smooth image",
    )
    dialog = ConnectionInsertDialog([candidate])
    qtbot.addWidget(dialog)

    light = _theme_palette(
        base="#ffffff", alternate="#f2f4f7", text="#111827"
    )
    dialog.setPalette(light)

    assert dialog.tree.palette().color(QPalette.Base) == QColor("#ffffff")
    assert dialog.tree.palette().color(QPalette.AlternateBase) == QColor("#f2f4f7")

    dark = _theme_palette(
        base="#111827", alternate="#1f2937", text="#f8fafc"
    )
    dialog.setPalette(dark)

    assert dialog.tree.palette().color(QPalette.Base) == QColor("#111827")
    assert dialog.tree.palette().color(QPalette.AlternateBase) == QColor("#1f2937")


def test_insert_dialog_foregrounds_follow_runtime_palette(qtbot):
    candidate = ConnectionInsertCandidate(
        operation_id="gaussian-blur",
        title="Gaussian Blur",
        category="Filtering",
        subcategory="",
        mode="full",
        detail="Smooth an image.",
        search_text="gaussian blur smooth image",
    )
    dialog = ConnectionInsertDialog([candidate])
    qtbot.addWidget(dialog)

    light = _theme_palette(
        base="#ffffff", alternate="#f2f4f7", text="#111827"
    )
    dialog.setPalette(light)
    item = dialog.tree.topLevelItem(0)
    light_category = item.foreground(0).color()
    light_mode = item.foreground(1).color()

    assert light_category == QColor(category_foreground("Filtering", light))
    assert light_category != QColor(category_color("Filtering"))
    assert light_mode != QColor(dialog._mode_color("full"))

    dark = _theme_palette(
        base="#111827", alternate="#1f2937", text="#f8fafc"
    )
    dialog.setPalette(dark)

    assert item.foreground(0).color() == QColor(category_color("Filtering"))
    assert item.foreground(1).color() == QColor(dialog._mode_color("full"))


def test_insert_dialog_style_change_refreshes_existing_foregrounds(qtbot):
    candidate = ConnectionInsertCandidate(
        operation_id="gaussian-blur",
        title="Gaussian Blur",
        category="Filtering",
        subcategory="",
        mode="full",
        detail="Smooth an image.",
        search_text="gaussian blur smooth image",
    )
    dialog = ConnectionInsertDialog([candidate])
    qtbot.addWidget(dialog)
    light = _theme_palette(
        base="#ffffff", alternate="#f2f4f7", text="#111827"
    )
    dialog.setPalette(light)
    item = dialog.tree.topLevelItem(0)
    item.setForeground(0, QBrush(QColor("#000000")))
    item.setForeground(1, QBrush(QColor("#000000")))

    QApplication.sendEvent(dialog, QEvent(QEvent.StyleChange))

    assert item.foreground(0).color() == QColor(
        category_foreground("Filtering", light)
    )
    assert item.foreground(1).color() == QColor(
        dialog._mode_foreground("full", light)
    )


def test_example_details_follow_runtime_palette(qtbot):
    dialog = ExampleWorkflowDialog(examples=())
    qtbot.addWidget(dialog)

    light = _theme_palette(
        base="#ffffff", alternate="#f2f4f7", text="#111827"
    )
    dialog.setPalette(light)
    assert "#f2f4f7" in dialog.details_label.styleSheet()
    assert "#111827" in dialog.details_label.styleSheet()

    dark = _theme_palette(
        base="#111827", alternate="#1f2937", text="#f8fafc"
    )
    dialog.setPalette(dark)
    assert "#1f2937" in dialog.details_label.styleSheet()
    assert "#f8fafc" in dialog.details_label.styleSheet()
