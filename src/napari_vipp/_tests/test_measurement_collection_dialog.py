"""Collector review, worker lifetime, and annotation safety contracts."""

from __future__ import annotations

import csv
import threading
import time
from dataclasses import replace
from functools import partial
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree
from zipfile import ZipFile

import pytest
from qtpy.QtCore import QItemSelectionModel, QPoint, Qt, QThread
from qtpy.QtGui import QColor, QFont, QPalette
from qtpy.QtWidgets import QMessageBox, QWidget

from napari_vipp.core.measurement_collection import (
    CollectionItem,
    MeasurementOutput,
    MeasurementPreview,
    collect_measurements,
    load_measurement_collection,
    save_measurement_collection,
)
from napari_vipp.core.tables import TableData
from napari_vipp.ui import measurement_collection as ui


def _preview(statuses=("ready", "empty"), *, metadata=False):
    columns = ("label", "area", "condition") if metadata else ("label", "area")
    rows = ((1, 25.0, "Control"), (2, 15.0, "Control")) if metadata else (
        (1, 25.0), (2, 15.0)
    )
    items = []
    for index, status in enumerate(statuses, 1):
        table = (
            TableData(columns, () if status == "empty" else rows,
                      column_units=(("area", "micrometer²"),))
            if status in {"ready", "empty"} else None
        )
        items.append(CollectionItem(
            key=f"item-{index}", index=index, batch_id=f"image-{index}",
            status=status, message=f"File check: {status}",
            row_count=table.row_count if table is not None else None,
            source_path=f"C:/private/source/image-{index}.tif",
            result_path=f"C:/results/image-{index}.json",
            result_sha256="c" * 64 if table is not None else "",
            table=table,
        ))
    return MeasurementPreview(
        MeasurementOutput("measure", "Object measurements", "objects"),
        tuple(items), "run-1", "a" * 64, "b" * 64,
    )


def _inspection(preview):
    preserved = tuple(
        (item.key, tuple(
            (field, value) for field in ui._ANNOTATIONS
            if (value := ui._preserved_annotation(item, field)) is not None
        )) for item in preview.items
    )
    return (preview.output,), preview, preserved


def _open(qtbot, monkeypatch, preview=None, **kwargs):
    preview = preview or _preview()
    monkeypatch.setattr(ui, "_inspect", lambda *_args: _inspection(preview))
    dialog = ui.MeasurementCollectionDialog("C:/batch/manifest.json", **kwargs)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.preview is not None)
    return dialog


def _select(dialog, *rows):
    selection = dialog.table.selectionModel()
    selection.clearSelection()
    for row in rows:
        selection.select(
            dialog.model.index(row, 1),
            QItemSelectionModel.Select | QItemSelectionModel.Rows,
        )


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_footer_orders_save_export_close_and_preserves_export_default(
    qtbot, monkeypatch, platform
):
    from napari_vipp.ui.dialog_buttons import add_dialog_buttons

    monkeypatch.setattr(
        ui, "add_dialog_buttons", partial(add_dialog_buttons, platform=platform)
    )
    dialog = _open(qtbot, monkeypatch)
    dialog.layout().activate()

    ordered = [dialog.save_button, dialog.export_button]
    if platform == "darwin":
        ordered.insert(0, dialog.close_button)
    else:
        ordered.append(dialog.close_button)
    assert ordered == sorted(ordered, key=lambda button: button.x())
    assert all(
        left.geometry().right() < right.x()
        for left, right in zip(ordered, ordered[1:], strict=False)
    )
    assert not dialog.close_button.autoDefault()
    assert not dialog.cancel_work_button.autoDefault()
    assert dialog.export_button.isDefault()


def test_review_distinguishes_zero_rows_and_unavailable_results(qtbot, monkeypatch):
    statuses = (
        "ready", "empty", "failed", "missing", "changed", "unsupported",
        "skipped", "partial",
    )
    dialog = _open(qtbot, monkeypatch, _preview(statuses))
    model = dialog.model
    assert model.rowCount() == 8
    assert model.included == {"item-1", "item-2"}
    assert model.data(model.index(1, 2)) == "Ready · no rows"
    assert model.data(model.index(1, 3)) == "0"
    assert model.data(model.index(2, 3)) == "—"
    assert [model.data(model.index(row, 2)) for row in range(2, 8)] == [
        "Failed", "Missing file", "File changed", "Unsupported table",
        "Skipped", "Incomplete",
    ]
    assert "2 images included · 6 excluded · 2 rows" in dialog.review_label.text()
    assert "1 successful image with no rows" in dialog.review_label.text()
    assert not dialog.save_button.isEnabled()
    assert not model.flags(model.index(2, 0)) & Qt.ItemIsUserCheckable
    assert not model.setData(model.index(2, 0), Qt.Checked, Qt.CheckStateRole)
    dialog.exclusions_check.setChecked(True)
    assert dialog.save_button.isEnabled()
    model.setData(model.index(0, 0), Qt.Unchecked, Qt.CheckStateRole)
    assert not dialog.exclusions_check.isChecked()
    assert not dialog.save_button.isEnabled()
    _select(dialog, 4)
    assert "File check: changed" in dialog.item_detail.text()
    assert "image-5.json" in dialog.item_detail.text()


def test_valid_empty_collection_can_be_saved(qtbot, monkeypatch, tmp_path):
    dialog = _open(qtbot, monkeypatch, _preview(("empty",)))
    assert dialog.save_button.isEnabled()
    assert "0 rows" in dialog.review_label.text()
    path = tmp_path / "empty.vipp-results.json"
    monkeypatch.setattr(ui.QFileDialog, "getSaveFileName", lambda *_a, **_k: (
        str(path), ""
    ))
    dialog.open_workflow_check.setChecked(True)
    with qtbot.waitSignal(dialog.collectionSaved, timeout=5000) as signal:
        dialog.save_button.click()
    assert signal.args == [str(path)]
    collection = load_measurement_collection(path)
    assert collection.table.rows == ()
    assert collection.items[0].included
    assert collection.items[0].status == "empty"
    assert dialog.isVisible()


def test_native_save_does_not_open_a_workflow_by_default(qtbot, monkeypatch, tmp_path):
    dialog = _open(qtbot, monkeypatch)
    assert not dialog.open_workflow_check.isChecked()
    assert dialog.save_button.text() == "Save VIPP collection…"
    assert dialog.export_button.text() == "Export results…"
    assert dialog.export_button.isDefault()
    opened = []
    dialog.collectionSaved.connect(opened.append)
    path = tmp_path / "saved.vipp-results.json"
    monkeypatch.setattr(ui.QFileDialog, "getSaveFileName", lambda *_a, **_k: (
        str(path), ""
    ))
    dialog.save_button.click()
    qtbot.waitUntil(lambda: dialog.status_label.text() == "Collection saved")
    assert path.exists()
    assert opened == []
    assert dialog.isVisible()
    assert dialog.export_button.isEnabled()
    assert dialog.save_button.isEnabled()


@pytest.mark.parametrize("already_exists", [False, True])
def test_native_save_binds_destination_revision_before_confirmation_or_work(
    qtbot, monkeypatch, tmp_path, already_exists,
):
    dialog = _open(qtbot, monkeypatch)
    path = tmp_path / "guarded.vipp-results.json"
    if already_exists:
        path.write_text("old snapshot", encoding="utf-8")

        def confirm(*_args, **_kwargs):
            path.write_text("Changed by another application", encoding="utf-8")
            return QMessageBox.Yes

        monkeypatch.setattr(QMessageBox, "question", confirm)
    else:
        save = ui._save

        def create_before_save(request, cancellation, progress):
            assert request.destination_revision is None
            path.write_text("Changed by another application", encoding="utf-8")
            return save(request, cancellation, progress)

        monkeypatch.setattr(ui, "_save", create_before_save)
    monkeypatch.setattr(ui.QFileDialog, "getSaveFileName", lambda *_a, **_k: (
        str(path), ""
    ))
    opened = []
    dialog.collectionSaved.connect(opened.append)
    dialog.open_workflow_check.setChecked(True)
    dialog.save_button.click()
    qtbot.waitUntil(lambda: dialog._task_id is None)
    assert dialog.status_label.text() == "Could not save the collection"
    assert path.read_text(encoding="utf-8") == "Changed by another application"
    assert opened == []
    assert dialog.save_button.isEnabled()


def _export_setup(monkeypatch, path, *, format="csv", summary=True):
    choice = ui._ExportChoice(format, summary)
    targets = (Path(path),)
    if summary and format != "xlsx":
        targets += (Path(path).with_stem(Path(path).stem + "-image-summary"),)
    monkeypatch.setattr(ui, "_choose_export_options", lambda _parent: choice)
    monkeypatch.setattr(ui, "_export_targets", lambda *_args: targets)
    monkeypatch.setattr(ui.QFileDialog, "getSaveFileName", lambda *_a, **_k: (
        str(path), ""
    ))
    return targets


@pytest.mark.parametrize("format", ["csv", "tsv", "xlsx"])
def test_direct_export_uses_reviewed_rows_keeps_window_and_never_opens_workflow(
    qtbot, monkeypatch, tmp_path, format,
):
    dialog = _open(qtbot, monkeypatch, _preview(("ready", "empty", "missing")))
    _select(dialog, 0, 1)
    dialog.annotation_edit.setText("Treatment")
    dialog.apply_button.click()
    dialog.open_workflow_check.setChecked(True)
    assert not dialog.export_button.isEnabled()
    dialog.exclusions_check.setChecked(True)
    targets = _export_setup(monkeypatch, tmp_path / f"results.{format}", format=format)
    collected = []
    opened = []
    dialog.collectionSaved.connect(opened.append)

    def export(request, cancellation, progress):
        assert not request.open_workflow
        assert request.export_format == format
        assert request.reviewed_exclusions
        collection = ui._collect(request, cancellation)
        collected.append(collection)
        progress(100, 100, "Export finished")
        return SimpleNamespace(paths=targets, notes=("Format-specific note.",))

    monkeypatch.setattr(ui, "_export", export)
    dialog.export_button.click()
    qtbot.waitUntil(lambda: dialog.status_label.text() == "Results exported")
    assert len(collected) == 1
    assert collected[0].table.row_count == 2  # Never invent a row for the empty image.
    assert collected[0].annotations["item-2"]["condition"] == "Treatment"
    assert [item.included for item in collected[0].items] == [True, True, False]
    assert opened == []
    assert dialog.isVisible()
    assert dialog.export_button.isEnabled()
    assert dialog.save_button.isEnabled()
    assert "Format-specific note." not in dialog.progress_label.text()
    assert "Format-specific note." in dialog.progress_label.toolTip()
    for path in targets:
        assert str(path) in dialog.progress_label.toolTip()


def test_companion_overwrite_requires_explicit_confirmation(
    qtbot, monkeypatch, tmp_path,
):
    dialog = _open(qtbot, monkeypatch)
    targets = _export_setup(monkeypatch, tmp_path / "measurements.csv")
    targets[1].write_text("Keep existing image summary", encoding="utf-8")
    questions = []
    requests = []

    def question(*args, **_kwargs):
        questions.append(args[2])
        return QMessageBox.Cancel

    monkeypatch.setattr(QMessageBox, "question", question)
    monkeypatch.setattr(ui, "_export", lambda *args: requests.append(args))
    dialog.export_button.click()
    assert dialog._task_id is None
    assert not requests
    assert not targets[0].exists()
    assert targets[1].read_text(encoding="utf-8") == "Keep existing image summary"
    assert all(str(path) in questions[0] for path in targets)

    def export(request, _cancel, _progress):
        requests.append(request)
        return SimpleNamespace(paths=targets, notes=())

    monkeypatch.setattr(QMessageBox, "question", lambda *_a, **_k: QMessageBox.Yes)
    monkeypatch.setattr(ui, "_export", export)
    dialog.export_button.click()
    qtbot.waitUntil(lambda: dialog.status_label.text() == "Results exported")
    assert requests[0].overwrite is True


def test_direct_export_companion_revision_is_bound_before_overwrite_prompt(
    qtbot, monkeypatch, tmp_path,
):
    dialog = _open(qtbot, monkeypatch)
    targets = _export_setup(monkeypatch, tmp_path / "guarded.csv")
    targets[1].write_text("old summary", encoding="utf-8")

    def confirm(*_args, **_kwargs):
        targets[1].write_text("New externally edited summary", encoding="utf-8")
        return QMessageBox.Yes

    monkeypatch.setattr(QMessageBox, "question", confirm)
    dialog.export_button.click()
    qtbot.waitUntil(lambda: dialog._task_id is None)
    assert dialog.status_label.text() == "Could not export the results"
    assert not targets[0].exists()
    assert targets[1].read_text(encoding="utf-8") == "New externally edited summary"


def test_cancelled_or_failed_export_keeps_review_and_never_opens_tab(
    qtbot, monkeypatch, tmp_path,
):
    dialog = _open(qtbot, monkeypatch)
    _export_setup(monkeypatch, tmp_path / "results.csv", summary=False)
    started = threading.Event()
    opened = []
    dialog.collectionSaved.connect(opened.append)
    original = dialog.preview

    def export(request, cancellation, _progress):
        assert not request.include_summary
        started.set()
        cancellation.wait(5)
        return None

    monkeypatch.setattr(ui, "_export", export)
    dialog.export_button.click()
    qtbot.waitUntil(started.is_set)
    assert dialog.cancel_work_button.text() == "Cancel export"
    assert not dialog.open_workflow_check.isEnabled()
    dialog.cancel_work()
    qtbot.waitUntil(lambda: dialog._task_id is None)
    assert dialog.status_label.text() == "Cancelled"
    assert opened == []
    assert dialog.preview is original
    assert dialog.model.included == {"item-1", "item-2"}

    def failure(*_args):
        raise ValueError("Could not publish the export; existing files were kept.")

    monkeypatch.setattr(ui, "_export", failure)
    dialog.export_button.click()
    qtbot.waitUntil(
        lambda: dialog.status_label.text() == "Could not export the results"
    )
    assert dialog.export_button.isEnabled()
    assert dialog.save_button.isEnabled()
    assert dialog.preview is original
    assert opened == []


def test_export_options_explain_formats_and_zero_image_accounting(qtbot):
    dialog = ui._ExportOptionsDialog()
    qtbot.addWidget(dialog)
    dialog.show()
    assert dialog.choice() == ui._ExportChoice("csv", True)
    assert "absent" in dialog.description_label.text()
    assert "types or units" in dialog.format_note.text()
    dialog.summary_check.setChecked(False)
    assert not dialog.choice().include_summary
    dialog.format_combo.setCurrentIndex(2)
    assert dialog.choice() == ui._ExportChoice("xlsx", True)
    assert not dialog.summary_check.isVisible()
    assert "About this collection" in dialog.description_label.text()
    assert "exact VIPP values" in dialog.format_note.text()
    dialog.format_combo.setCurrentIndex(1)
    assert dialog.choice() == ui._ExportChoice("tsv", False)
    assert dialog.summary_check.isVisible()


@pytest.mark.parametrize("format_index", [0, 2])
def test_export_options_large_font_wraps_without_clipped_explanations(
    qtbot, monkeypatch, format_index,
):
    parent = _open(qtbot, monkeypatch)
    parent.setFont(QFont("Segoe UI", 14))
    dialog = ui._ExportOptionsDialog(parent)
    qtbot.addWidget(dialog)
    dialog.format_combo.setCurrentIndex(format_index)
    dialog.show()
    dialog.grab()
    for label in (dialog.description_label, dialog.summary_help, dialog.format_note):
        assert label.height() >= label.heightForWidth(label.width())
        point = label.mapTo(dialog, QPoint(0, 0))
        assert point.x() + label.width() <= dialog.width()
        assert point.y() + label.height() <= dialog.height()
    assert dialog.buttons.geometry().bottom() < dialog.height()


def test_cancelled_format_or_file_picker_does_nothing(qtbot, monkeypatch):
    dialog = _open(qtbot, monkeypatch)
    monkeypatch.setattr(ui, "_choose_export_options", lambda _parent: None)
    dialog.export_button.click()
    assert dialog._task_id is None
    monkeypatch.setattr(
        ui, "_choose_export_options", lambda _parent: ui._ExportChoice()
    )
    monkeypatch.setattr(ui.QFileDialog, "getSaveFileName", lambda *_a, **_k: ("", ""))
    dialog.export_button.click()
    assert dialog._task_id is None


@pytest.mark.parametrize(
    "format,summary", [("csv", True), ("tsv", False), ("xlsx", True)]
)
def test_real_direct_export_roundtrip_does_not_reopen_originals_or_add_empty_rows(
    qtbot, monkeypatch, tmp_path, format, summary,
):
    from napari_vipp.core import measurement_collection as core

    dialog = _open(qtbot, monkeypatch, _preview(("ready", "empty", "missing")))
    _select(dialog, 0, 1)
    dialog.annotation_edit.setText("Condition A")
    dialog.apply_button.click()
    dialog.exclusions_check.setChecked(True)
    dialog.open_workflow_check.setChecked(True)  # Deliberately irrelevant to export.
    path = tmp_path / f"actual.{format}"
    monkeypatch.setattr(
        ui, "_choose_export_options", lambda _parent: ui._ExportChoice(format, summary)
    )
    monkeypatch.setattr(
        ui.QFileDialog, "getSaveFileName", lambda *_a, **_k: (str(path), "")
    )

    def forbidden_read(*_args, **_kwargs):
        raise AssertionError("Original files must not be read during export.")

    monkeypatch.setattr(core, "_read_bytes", forbidden_read)
    opened = []
    dialog.collectionSaved.connect(opened.append)
    dialog.export_button.click()
    qtbot.waitUntil(lambda: dialog._task_id is None, timeout=10_000)
    assert dialog.status_label.text() == "Results exported", (
        dialog.progress_label.text()
    )
    assert opened == []
    assert dialog.isVisible()
    if format == "xlsx":
        namespace = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        with ZipFile(path) as archive:
            workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
            sheets = workbook.findall("m:sheets/m:sheet", namespace)
            assert [sheet.attrib["name"] for sheet in sheets] == [
                "Measurements", "Image summary", "About this collection",
            ]
            measurements = ElementTree.fromstring(
                archive.read("xl/worksheets/sheet1.xml")
            )
            inventory = ElementTree.fromstring(archive.read("xl/worksheets/sheet2.xml"))
        assert len(measurements.findall("m:sheetData/m:row", namespace)) == 3
        assert len(inventory.findall("m:sheetData/m:row", namespace)) == 4
    else:
        delimiter = "\t" if format == "tsv" else ","
        with path.open(encoding="utf-8", newline="") as stream:
            measurements = list(csv.DictReader(stream, delimiter=delimiter))
        assert len(measurements) == 2
        assert measurements[0]["condition"] == "Condition A"
        summary_path = path.with_stem(path.stem + "-image-summary")
        assert summary_path.exists() == summary
        if summary:
            with summary_path.open(encoding="utf-8", newline="") as stream:
                inventory = list(csv.DictReader(stream, delimiter=delimiter))
            assert len(inventory) == 3
            assert [row["Measurement rows"] for row in inventory] == ["2", "0", ""]


def test_existing_metadata_is_visible_and_protected(qtbot, monkeypatch):
    preview = _preview(metadata=True)
    dialog = _open(qtbot, monkeypatch, preview)
    assert dialog.model.data(dialog.model.index(0, 5)) == "Control"
    assert dialog.model.data(dialog.model.index(1, 5)) == "Existing column (no rows)"
    assert not dialog.model.flags(dialog.model.index(0, 5)) & Qt.ItemIsEditable
    assert "preserved" in dialog.model.data(dialog.model.index(0, 5), Qt.ToolTipRole)
    _select(dialog, 0, 1)
    dialog.annotation_edit.setText("Treatment")
    dialog.apply_button.click()
    assert dialog.model.annotations == {}
    assert "Existing metadata kept for 2 images" in dialog.item_detail.text()
    assert preview.items[0].table.rows[0][-1] == "Control"


def test_varying_metadata_is_not_collapsed_or_replaced():
    item = _preview(metadata=True).items[0]
    item = replace(item, table=replace(item.table, rows=(
        (1, 25.0, "Control"), (2, 15.0, "Treatment"),
    )))
    assert ui._preserved_annotation(item, "condition") == "Multiple values (kept)"
    assert ui._preserved_annotation(item, "sample") is None


def test_bulk_annotations_only_target_selection_and_require_conflict_review(
    qtbot, monkeypatch,
):
    dialog = _open(qtbot, monkeypatch, _preview(("ready", "ready", "ready")))
    _select(dialog, 0, 2)
    dialog.annotation_edit.setText("Control")
    dialog.apply_button.click()
    assert dialog.model.annotations == {
        "item-1": {"condition": "Control"}, "item-3": {"condition": "Control"},
    }
    monkeypatch.setattr(QMessageBox, "question", lambda *_a, **_k: QMessageBox.Cancel)
    _select(dialog, 0, 1)
    dialog.annotation_edit.setText("Treatment")
    dialog.apply_button.click()
    assert "item-2" not in dialog.model.annotations
    assert dialog.model.annotations["item-1"]["condition"] == "Control"
    monkeypatch.setattr(QMessageBox, "question", lambda *_a, **_k: QMessageBox.Yes)
    dialog.apply_button.click()
    assert dialog.model.annotations == {
        "item-1": {"condition": "Treatment"},
        "item-2": {"condition": "Treatment"},
        "item-3": {"condition": "Control"},
    }


def test_save_snapshot_preserves_annotations_exclusions_and_provenance(
    qtbot, monkeypatch, tmp_path,
):
    dialog = _open(qtbot, monkeypatch, _preview(("ready", "empty", "missing")))
    _select(dialog, 0, 1)
    dialog.annotation_edit.setText("Treatment")
    dialog.apply_button.click()
    dialog.exclusions_check.setChecked(True)
    path = tmp_path / "review.vipp-results.json"
    monkeypatch.setattr(ui.QFileDialog, "getSaveFileName", lambda *_a, **_k: (
        str(tmp_path / "review"), ""
    ))
    dialog.open_workflow_check.setChecked(True)
    with qtbot.waitSignal(dialog.collectionSaved, timeout=5000):
        dialog.save_button.click()
    saved = load_measurement_collection(path)
    assert saved.annotations["item-1"]["condition"] == "Treatment"
    assert saved.annotations["item-2"]["condition"] == "Treatment"
    assert [item.included for item in saved.items] == [True, True, False]
    assert saved.provenance["reviewed_exclusions"] is True
    assert saved.provenance["manifest_sha256"] == "a" * 64
    assert saved.table.unit_for("area") == "micrometer²"
    assert saved.table.rows[0][:2] == (1, 25.0)


def test_suffix_normalization_cannot_bypass_overwrite_confirmation(
    qtbot, monkeypatch, tmp_path,
):
    dialog = _open(qtbot, monkeypatch)
    path = tmp_path / "existing.vipp-results.json"
    path.write_text("keep this existing file", encoding="utf-8")
    monkeypatch.setattr(ui.QFileDialog, "getSaveFileName", lambda *_a, **_k: (
        str(tmp_path / "existing"), ""
    ))
    monkeypatch.setattr(QMessageBox, "question", lambda *_a, **_k: QMessageBox.Cancel)
    dialog.save_button.click()
    assert dialog._task_id is None
    assert path.read_text(encoding="utf-8") == "keep this existing file"


def test_cancelled_output_change_preserves_actual_selected_output(qtbot, monkeypatch):
    dialog = _open(qtbot, monkeypatch)
    dialog.output_combo.addItem("Second output", "other")
    dialog.model.setData(dialog.model.index(0, 5), "Control")
    monkeypatch.setattr(QMessageBox, "question", lambda *_a, **_k: QMessageBox.Cancel)
    dialog.output_combo.setCurrentIndex(1)
    assert dialog.output_combo.currentData() == "measure"
    assert dialog.preview.output.node_id == "measure"
    assert dialog.model.annotations["item-1"]["condition"] == "Control"


def test_io_runs_off_gui_thread_and_progress_is_queued(qtbot, monkeypatch, qapp):
    thread_checks = []
    preview = _preview()

    def inspect(_request, _cancel, progress):
        thread_checks.append(QThread.currentThread() != qapp.thread())
        progress(1, 2, "Checked one table")
        return _inspection(preview)

    monkeypatch.setattr(ui, "_inspect", inspect)
    dialog = ui.MeasurementCollectionDialog("C:/batch/manifest.json")
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.preview is not None)
    assert thread_checks == [True]
    assert dialog.table.thread() == qapp.thread()


@pytest.mark.parametrize("action", ["cancel", "close", "delete_parent"])
def test_cancel_and_close_never_wait_or_deliver_stale_preview(
    qtbot, monkeypatch, action,
):
    started, cancelled = threading.Event(), threading.Event()

    def inspect(_request, cancellation, _progress):
        started.set()
        if cancellation.wait(5):
            cancelled.set()
        return _inspection(_preview())

    monkeypatch.setattr(ui, "_inspect", inspect)
    parent = QWidget()
    qtbot.addWidget(parent)
    dialog = ui.MeasurementCollectionDialog("C:/batch/manifest.json", parent)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(started.is_set)
    service = dialog._service
    started_at = time.monotonic()
    if action == "cancel":
        dialog.cancel_work()
    elif action == "close":
        dialog.close()
    else:
        parent.deleteLater()
    assert time.monotonic() - started_at < 0.5
    qtbot.waitUntil(cancelled.is_set)
    qtbot.waitUntil(lambda: not service.jobs)
    if action != "delete_parent":
        assert dialog.preview is None
        assert not dialog.save_button.isEnabled()


def test_stale_reply_and_progress_cannot_change_a_review(qtbot, monkeypatch):
    dialog = _open(qtbot, monkeypatch)
    before = dialog.status_label.text()
    preview = dialog.preview
    dialog._completed(ui._Reply("stale", -1, "inspect", error="Old failure"))
    dialog._progressed(("stale", -1, 1, 1, "Stale progress"))
    assert dialog.preview is preview
    assert dialog.status_label.text() == before
    assert "Stale progress" not in dialog.progress_label.text()


def test_cancel_save_does_not_emit_saved_path(qtbot, monkeypatch, tmp_path):
    dialog = _open(qtbot, monkeypatch)
    started = threading.Event()
    paths = []
    dialog.collectionSaved.connect(paths.append)

    def save(_request, cancellation, _progress):
        started.set()
        cancellation.wait(5)
        return None

    monkeypatch.setattr(ui, "_save", save)
    path = tmp_path / "cancel.vipp-results.json"
    monkeypatch.setattr(ui.QFileDialog, "getSaveFileName", lambda *_a, **_k: (
        str(path), ""
    ))
    dialog.save_button.click()
    qtbot.waitUntil(started.is_set)
    assert not dialog.table.isEnabled()
    dialog.cancel_work()
    qtbot.waitUntil(lambda: dialog._task_id is None)
    assert not path.exists()
    assert paths == []
    assert dialog.status_label.text() == "Cancelled"


def test_inspection_adapter_reads_only_core_table_apis_off_thread(
    qtbot, monkeypatch, qapp,
):
    from napari_vipp.core import measurement_collection as core

    preview = _preview(metadata=True)
    calls = []

    def available(manifest):
        calls.append(("available", manifest, QThread.currentThread() != qapp.thread()))
        return (preview.output,)

    def inspect(manifest, node_id, *, cancellation, progress):
        calls.append((node_id, manifest, QThread.currentThread() != qapp.thread()))
        assert isinstance(cancellation, threading.Event)
        progress(1, 2, "Checked first table")
        return preview

    monkeypatch.setattr(core, "available_measurement_outputs", available)
    monkeypatch.setattr(core, "inspect_collection", inspect)
    dialog = ui.MeasurementCollectionDialog("C:/batch/manifest.json")
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.preview is not None)
    assert calls == [
        ("available", "C:/batch/manifest.json", True),
        ("measure", "C:/batch/manifest.json", True),
    ]
    assert dialog.model.data(dialog.model.index(0, 5)) == "Control"


@pytest.mark.parametrize("failure", [False, True])
def test_no_outputs_or_failed_inspection_has_clear_disabled_save(
    qtbot, monkeypatch, failure,
):
    def inspect(*_args):
        if failure:
            raise ValueError("The batch record changed. Select the completed run.")
        return (), None, ()

    monkeypatch.setattr(ui, "_inspect", inspect)
    dialog = ui.MeasurementCollectionDialog("C:/batch/manifest.json")
    qtbot.addWidget(dialog)
    dialog.show()
    expected = (
        "Could not check the tables" if failure else "No saved measurement output"
    )
    qtbot.waitUntil(lambda: dialog.status_label.text() == expected)
    assert not dialog.save_button.isEnabled()
    assert dialog.model.rowCount() == 0
    assert dialog.inspect_button.isEnabled()


def test_saved_review_keeps_exact_inclusion_inventory_and_annotations(
    qtbot, tmp_path, monkeypatch,
):
    preview = _preview(("ready", "empty", "ready", "changed"), metadata=True)
    collection = collect_measurements(
        preview, included_ids=("item-1", "item-2"), reviewed_exclusions=True,
        annotations={
            "item-1": {"sample": "Sample A"},
            "item-2": {"sample": "Sample B"},
            "item-3": {"sample": "Excluded Sample"},
        },
    )
    path = tmp_path / "review.vipp-results.json"
    save_measurement_collection(collection, path)

    def no_original_files(*_args, **_kwargs):
        pytest.fail("Review must not inspect original tables or images.")

    monkeypatch.setattr(ui, "_inspect", no_original_files)
    dialog = ui.MeasurementCollectionReviewDialog(
        path, expected_sha256=sha256(path.read_bytes()).hexdigest()
    )
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.preview is not None)
    assert dialog.model.rowCount() == 4
    assert dialog.model.included == {"item-1", "item-2"}
    included_state = dialog.model.data(dialog.model.index(2, 0), Qt.CheckStateRole)
    assert included_state == Qt.Unchecked
    assert dialog.model.data(dialog.model.index(0, 5)) == "Control"
    assert dialog.model.data(dialog.model.index(0, 6)) == "Sample A"
    assert dialog.model.data(dialog.model.index(1, 6)) == "Sample B"
    assert dialog.model.data(dialog.model.index(2, 6)) == "Excluded Sample"
    assert not dialog.model.flags(dialog.model.index(0, 0)) & Qt.ItemIsUserCheckable
    assert not dialog.model.flags(dialog.model.index(0, 6)) & Qt.ItemIsEditable
    assert not dialog.save_button.isVisible()
    assert not dialog.export_button.isVisible()
    assert not dialog.open_workflow_check.isVisible()
    assert not dialog.exclusions_check.isVisible()
    assert "2 images included · 2 excluded" in dialog.review_label.text()
    assert "not reopened" in dialog.intro_label.text()


def test_saved_review_rejects_changed_bound_snapshot(qtbot, tmp_path):
    path = tmp_path / "review.vipp-results.json"
    save_measurement_collection(collect_measurements(_preview()), path)
    dialog = ui.MeasurementCollectionReviewDialog(path, expected_sha256="0" * 64)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: "Could not" in dialog.status_label.text())
    assert dialog.preview is None
    assert dialog.model.rowCount() == 0
    assert "hash does not match" in dialog.progress_label.text().casefold()


def test_parent_styles_cannot_shrink_fonts_and_theme_remains_live(qtbot, monkeypatch):
    parent = QWidget()
    qtbot.addWidget(parent)
    parent.setStyleSheet("QWidget {font-family: Arial; font-size: 6pt;}")
    dialog = _open(qtbot, monkeypatch, parent=parent)
    dialog.setFont(QFont("Segoe UI", 14))
    dialog.grab()
    for widget in (
        dialog.title_label, dialog.intro_label, dialog.status_label,
        dialog.output_combo, dialog.annotation_edit, dialog.save_button,
        dialog.export_button,
        dialog.table.horizontalHeader(),
    ):
        assert widget.font().pointSizeF() == pytest.approx(14), (
            type(widget).__name__, widget.styleSheet(), dialog.font().toString()
        )
        assert widget.font().family() == "Segoe UI"
    original_style = dialog.status_panel.styleSheet()
    palette = QPalette(dialog.palette())
    palette.setColor(QPalette.Base, QColor("#111827"))
    palette.setColor(QPalette.Text, QColor("#f5f6fa"))
    dialog.setPalette(palette)
    assert dialog.status_panel.styleSheet() != original_style
    assert ui.theme_colors(palette).success.surface.name() in (
        dialog.status_panel.styleSheet()
    )


def test_long_evidence_text_is_compact_with_full_details_in_tooltip(qtbot, monkeypatch):
    preview = _preview()
    item = replace(preview.items[0], message="Long evidence " * 1000)
    preview = replace(preview, items=(item, *preview.items[1:]))
    dialog = _open(qtbot, monkeypatch, preview)
    _select(dialog, 0)
    assert len(dialog.item_detail.text()) < 400
    assert item.message in dialog.item_detail.toolTip()
    dialog._status("Could not check", "Error " * 1000, "error")
    assert len(dialog.progress_label.text()) <= 400
    assert "Error " * 1000 == dialog.progress_label.toolTip()


@pytest.mark.parametrize("width", [640, 1100])
@pytest.mark.parametrize("point_size", [10, 14])
@pytest.mark.parametrize("dark", [False, True])
def test_responsive_theme_and_large_font_layout(
    qtbot, monkeypatch, width, point_size, dark,
):
    dialog = _open(qtbot, monkeypatch)
    palette = QPalette(dialog.palette())
    base = QColor("#252831" if dark else "#ffffff")
    foreground = QColor("#eeeeee" if dark else "#182432")
    for role in (QPalette.Base, QPalette.Window, QPalette.Button):
        palette.setColor(role, base)
    for role in (QPalette.Text, QPalette.WindowText, QPalette.ButtonText):
        palette.setColor(role, foreground)
    dialog.setPalette(palette)
    font = QFont(dialog.font())
    font.setPointSize(point_size)
    dialog.setFont(font)
    dialog.resize(width, 840)
    qtbot.waitUntil(lambda: dialog.table.viewport().width() > 100)
    dialog.grab()
    assert dialog.width() <= width + 10
    for widget in (
        dialog.intro_label, dialog.status_panel, dialog.annotation_help,
        dialog.review_label, dialog.save_button, dialog.close_button,
        dialog.export_button, dialog.open_workflow_check,
    ):
        location = widget.mapTo(dialog, QPoint(0, 0))
        assert location.x() >= 0
        assert location.x() + widget.width() <= dialog.width()
        assert location.y() + widget.height() <= dialog.height()
    assert dialog.intro_label.height() >= dialog.intro_label.heightForWidth(
        dialog.intro_label.width()
    )
    assert dialog.table.viewport().height() >= 130
    assert dialog.table.verticalHeader().defaultSectionSize() >= font.pointSize() + 16
    colors = ui.theme_colors(palette)
    assert colors.success.surface.name() in dialog.status_panel.styleSheet()
    assert dialog.table.palette().color(QPalette.Text).lightnessF() > 0.5 if dark else (
        dialog.table.palette().color(QPalette.Text).lightnessF() < 0.5
    )
