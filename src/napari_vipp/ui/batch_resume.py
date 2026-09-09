"""Explicit, non-destructive entry point for a saved batch continuation."""

from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QFileDialog, QMessageBox


class BatchResumeActions:
    """Keep checkpoint selection separate from ordinary existing-file policy."""

    def _request_resume(self) -> None:
        if self._run_in_progress or self._run_preparing or self._checking_plan:
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Resume a saved batch run",
            self.output_edit.text().strip(),
            "VIPP batch manifests (vipp_batch_manifest*.json);;JSON files (*.json)",
        )
        if not path:
            return
        box = QMessageBox(self)
        box.setWindowTitle("Verify and resume saved run?")
        box.setIcon(QMessageBox.Question)
        box.setTextFormat(Qt.PlainText)
        box.setText(f"Resume the saved run in {Path(path).name}?")
        box.setInformativeText(
            "Use this run's saved workflow and settings, not the currently open "
            "workflow. Your open workflow stays unchanged.\n\n"
            "VIPP first verifies the checkpoints, inputs, parameters, software "
            "and output contents. Only verified completed items are reused; "
            "unfinished items are calculated. Unverified existing files are "
            "never silently skipped or overwritten.\n\n"
            "Older runs without the required verification records cannot be resumed."
        )
        resume = box.addButton("Verify and resume", QMessageBox.AcceptRole)
        cancel = box.addButton("Cancel", QMessageBox.RejectRole)
        box.setDefaultButton(cancel)
        box.setEscapeButton(cancel)
        box.exec()
        if box.clickedButton() is resume:
            self.runRequested.emit({"resume_manifest_path": str(Path(path))})

    def begin_resume_run(self, path: Path) -> None:
        # No disk access or parsing on the GUI thread. The worker validates the
        # archive; totals arrive with its verified progress/result records.
        self.results_panel.set_plan(None)
        self.results_panel.set_item_review_available(False)
        self._resume_source_path = path
        self.begin_run(0)
        self.show_workspace_activity(
            "Verifying saved run · inputs, settings and output contents…",
            state="working",
            indeterminate=True,
            progress_text="Verifying",
        )
        self.cancel_run_button.setText("Cancel verification")
        self.run_progress_label.setText("Verifying saved checkpoints before resuming…")
