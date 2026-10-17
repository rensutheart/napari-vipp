"""Explicit review/apply of tested local edits; no model or Run calls."""

from __future__ import annotations

import json

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from napari_vipp.ui.workflow_assistant_details import AssistantDetailsEditor
from napari_vipp.ui.workflow_assistant_style import AssistantButton


class AssistantPatchController:
    def __init__(self, dialog, prepare, apply):
        self.dialog, self.prepare, self.apply = dialog, prepare, apply
        self.button = AssistantButton("Review in-place changes…")
        self.button.setAutoDefault(False)
        self.button.setToolTip(
            "Review tested parameter or compatible-node edits to your existing "
            "workflow. Branch/topology changes still use Add to workflow. "
            "Applying saves a recovery checkpoint, supports Undo, and never runs."
        )
        self.button.clicked.connect(self.review)
        self.sync()

    def _baseline(self):
        controller = self.dialog._investigation
        return controller._existing_bundle or controller._warm_bundle

    def sync(self):
        dialog = self.dialog
        controller = getattr(dialog, "_investigation", None)
        session = getattr(controller, "session", None)
        visible = self.prepare is not None and self.apply is not None
        self.button.setVisible(visible)
        self.button.setEnabled(
            visible
            and not dialog._busy
            and session is not None
            and session.finished
            and self._baseline() is not None
            and dialog._prepared is not None
            and dialog._draft_authorized
            and dialog._latest_selected()
        )

    def review(self):
        dialog = self.dialog
        if not self.button.isEnabled():
            return
        try:
            prepared = self.prepare(
                dialog._context,
                dialog._owner,
                self._baseline(),
                dialog._investigation.session,
            )
        except (ValueError, RuntimeError) as error:
            dialog._set_status(
                "warning",
                f"Cannot edit this result in place: {error} "
                "Your current workflow is unchanged. Add can create a separate branch.",
            )
            return
        review = QDialog(dialog)
        review.setWindowTitle("Review changes to existing workflow")
        review.resize(660, 470)
        layout = QVBoxLayout(review)
        description = QLabel(
            "Only the changes below will be applied. Node locations and unrelated "
            "results stay in place. A recovery copy is saved first. You can Undo "
            "the whole edit; affected results need recalculation."
        )
        description.setWordWrap(True)
        layout.addWidget(description)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        rows = QVBoxLayout(body)
        diff = prepared.public_diff()
        for change in diff["changes"]:
            heading = QLabel(
                f"{change['node_id']} · {change['after_operation'].replace('_', ' ')}"
            )
            heading.setTextFormat(Qt.PlainText)
            font = heading.font()
            font.setBold(True)
            heading.setFont(font)
            rows.addWidget(heading)
            lines = []
            if change["before_operation"] != change["after_operation"]:
                lines.append(
                    "Processing step: "
                    + change["before_operation"].replace("_", " ")
                    + " → "
                    + change["after_operation"].replace("_", " ")
                )
            for param in change["parameters"]:
                before = str(param["before"]) if param["before_present"] else "not set"
                after = str(param["after"]) if param["after_present"] else "not set"
                lines.append(f"{param['name'].replace('_', ' ')}: {before} → {after}")
            if change["local_fields_reset"]:
                lines.append(
                    "Previously inferred settings will be checked again when run."
                )
            label = QLabel("\n".join("• " + line for line in lines))
            label.setTextFormat(Qt.PlainText)
            label.setWordWrap(True)
            rows.addWidget(label)
        rows.addStretch()
        scroll.setWidget(body)
        layout.addWidget(scroll, 1)
        from napari_vipp.ui.workflow_assistant_style import AssistantDisclosureButton

        advanced = AssistantDisclosureButton("Technical details")
        layout.addWidget(advanced)
        details = AssistantDetailsEditor(json_syntax=True)
        details.setReadOnly(True)
        details.setPlainText(json.dumps(diff, indent=2, ensure_ascii=False))
        details.setMaximumHeight(180)
        details.hide()
        advanced.toggled.connect(details.setVisible)
        layout.addWidget(details)
        row = QHBoxLayout()
        row.addStretch()
        cancel = AssistantButton("Cancel")
        confirm = AssistantButton("Apply changes")
        confirm.setProperty("assistantAction", "primary")
        cancel.clicked.connect(review.reject)
        confirm.clicked.connect(review.accept)
        row.addWidget(cancel)
        row.addWidget(confirm)
        layout.addLayout(row)
        if review.exec() != QDialog.Accepted:
            return
        try:
            ids = self.apply(prepared, dialog._context, dialog._owner)
        except (ValueError, RuntimeError) as error:
            dialog._set_status("error", f"No in-place edit was applied: {error}")
            self.sync()
            return
        dialog._context = dialog._owner = None
        dialog._invalidate_proposal()
        dialog.consent_checkbox.setChecked(False)
        dialog._added_ids = tuple(ids)
        dialog.json_edit.setReadOnly(True)
        message = (
            f"Updated {len(prepared.changed_node_ids)} existing nodes "
            "as one undoable edit. "
            "Nothing was calculated. Review the workflow, then Run when ready."
        )
        dialog._append_chat("VIPP", message)
        dialog.draft_note.setText("Updated in place · Run is a separate action.")
        dialog._set_status("success", message)
        dialog._sync_buttons()
