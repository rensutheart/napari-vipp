from __future__ import annotations

from qtpy.QtWidgets import QDialog

from napari_vipp._widget import ExampleWorkflowDialog, ExampleWorkflowSpec


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
