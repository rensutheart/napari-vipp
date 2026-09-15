from __future__ import annotations

import csv
import os
import threading
from dataclasses import replace
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

import pytest

from napari_vipp.core import measurement_export as module
from napari_vipp.core.measurement_collection import (
    CollectionItem,
    CollectionLimits,
    MeasurementCollectionError,
    MeasurementOutput,
    MeasurementPreview,
    collect_measurements,
)
from napari_vipp.core.measurement_export import (
    MeasurementExportError,
    export_measurement_collection,
    measurement_export_destination_revisions,
    measurement_export_targets,
)
from napari_vipp.core.progress import OperationCancelled
from napari_vipp.core.tables import TableData


def _collection(tmp_path, table=None):
    if table is None:
        table = TableData(
            ("label", "volume", "condition", "note"),
            (
                (2**64 - 1, 1.23456789123456789, "control", "=SUM(1,2)"),
                (2, float("-inf"), "treated", "https://example.invalid/a"),
            ),
            name="Object measurements",
            table_kind="objects",
            column_units=(("volume", "micrometer^3"),),
        )
    items = (
        CollectionItem(
            "item-1",
            1,
            "image-one",
            "ready",
            row_count=table.row_count,
            source_path=str(tmp_path / "never-read-one.tif"),
            result_path=str(tmp_path / "original-one.csv"),
            result_sha256="a" * 64,
            table=table,
        ),
        CollectionItem(
            "item-2",
            2,
            "empty-image",
            "empty",
            row_count=0,
            source_path=str(tmp_path / "never-read-two.tif"),
            result_path=str(tmp_path / "original-two.csv"),
            result_sha256="b" * 64,
            table=replace(table, rows=()),
        ),
        CollectionItem(
            "item-3",
            3,
            "failed-image",
            "failed",
            message="Missing result",
            source_path=str(tmp_path / "never-read-three.tif"),
            result_path=str(tmp_path / "original-three.csv"),
        ),
    )
    preview = MeasurementPreview(
        MeasurementOutput("batch_output_1", "Measurements", "objects"),
        items,
        "run-example",
        "c" * 64,
        "d" * 64,
    )
    return collect_measurements(
        preview,
        included_ids=("item-1", "item-2"),
        reviewed_exclusions=True,
        annotations={
            "item-1": {"replicate": "R1"},
            "item-2": {"replicate": "empty control"},
            "item-3": {"replicate": "excluded control"},
        },
    )


def _read_csv(path, delimiter=","):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def _xlsx_cells(path, sheet=1):
    with ZipFile(path) as archive:
        root = ElementTree.fromstring(archive.read(f"xl/worksheets/sheet{sheet}.xml"))
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    result = {}
    for cell in root.findall(".//m:c", ns):
        text = "".join(node.text or "" for node in cell.findall(".//m:t", ns))
        numeric = cell.find("m:v", ns)
        result[cell.attrib["r"]] = (
            cell.attrib.get("t", "n"),
            text if numeric is None else numeric.text,
        )
    return result


@pytest.mark.parametrize("format", ["csv", "tsv", "xlsx"])
def test_plan_targets_is_pure_and_names_the_companion(tmp_path, monkeypatch, format):
    monkeypatch.setattr(Path, "stat", lambda *a, **kw: pytest.fail("Planner read disk"))
    targets = measurement_export_targets(tmp_path / "out", format=format)
    expected = (tmp_path / f"out.{format}",)
    if format != "xlsx":
        expected += (tmp_path / f"out-image-summary.{format}",)
    assert targets == expected


@pytest.mark.parametrize(
    "path,options",
    [
        ("", {}),
        ("out.txt", {}),
        ("out.csv", {"format": "xlsx"}),
        ("out.csv", {"include_image_summary": "yes"}),
        ("../out.csv", {}),
        ("//server/share/out.csv", {}),
    ],
)
def test_reject_invalid_export_targets(path, options):
    with pytest.raises(MeasurementExportError):
        measurement_export_targets(path, **options)


@pytest.mark.parametrize("format", ["csv", "tsv"])
def test_delimited_export_preserves_rows_inventory_and_annotations(
    tmp_path, monkeypatch, format
):
    collection = _collection(tmp_path)
    original_open = Path.open
    recorded = {
        Path(path)
        for item in collection.items
        for path in (item.source_path, item.result_path)
    }

    def guarded_open(path, *args, **kwargs):
        assert path not in recorded, "Export opened original data"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    events = []
    result = export_measurement_collection(
        collection, tmp_path / f"export.{format}", progress=lambda *x: events.append(x)
    )
    delimiter = "\t" if format == "tsv" else ","
    rows = _read_csv(result.paths[0], delimiter)
    summary = _read_csv(result.paths[1], delimiter)
    assert len(rows) == 2
    assert rows[0]["label"] == str(2**64 - 1)
    assert rows[0]["note"] == "=SUM(1,2)"
    assert rows[1]["note"] == "https://example.invalid/a"
    assert rows[1]["volume"] == "-inf"
    assert [row["_vipp_item_key"] for row in rows] == ["item-1", "item-1"]
    assert [row["Measurement rows"] for row in summary] == ["2", "0", ""]
    assert [row["Included"] for row in summary] == ["True", "True", "False"]
    assert summary[0]["Annotation: condition"] == "Mixed (see Measurements)"
    assert summary[1]["Annotation: condition"] == ""
    assert summary[1]["Annotation: replicate"] == "empty control"
    assert summary[2]["Annotation: replicate"] == "excluded control"
    assert "formula-like" in " ".join(result.notes)
    assert "types or units" in " ".join(result.notes)
    assert events[0][0] == 0 and events[-1][0] == 1
    assert not list(tmp_path.glob(".vipp-export-*"))
    assert collection.table.rows[0][0] == 2**64 - 1


def test_csv_without_summary_has_truthful_inventory_warning(tmp_path):
    result = export_measurement_collection(
        _collection(tmp_path), tmp_path / "out.csv", include_image_summary=False
    )
    assert len(result.paths) == 1
    assert not (tmp_path / "out-image-summary.csv").exists()
    assert "No image-summary companion" in " ".join(result.notes)
    assert "Image summary retains" not in " ".join(result.notes)


def test_xlsx_has_three_sheets_literal_strings_exact_ids_units_and_inventory(tmp_path):
    result = export_measurement_collection(_collection(tmp_path), tmp_path / "out.xlsx")
    (path,) = result.paths
    with ZipFile(path) as archive:
        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        names = [
            node.attrib["name"]
            for node in workbook.iter()
            if node.tag.endswith("}sheet")
        ]
        assert names == ["Measurements", "Image summary", "About this collection"]
        for name in archive.namelist():
            if name.startswith("xl/worksheets/") and name.endswith(".xml"):
                xml = archive.read(name)
                assert b"<f>" not in xml and b"<hyperlink" not in xml
    cells = _xlsx_cells(path)
    assert cells["A2"] == ("inlineStr", str(2**64 - 1))
    assert cells["B2"][0] == "n"
    assert cells["B3"] == ("inlineStr", "-inf")
    assert cells["D2"] == ("inlineStr", "=SUM(1,2)")
    assert cells["D3"] == ("inlineStr", "https://example.invalid/a")
    summary = _xlsx_cells(path, 2)
    assert summary["F2"] == ("n", "2")
    assert summary["F3"] == ("n", "0")
    assert "F4" not in summary
    assert summary["O2"] == ("inlineStr", "Mixed (see Measurements)")
    assert summary["P3"] == ("inlineStr", "empty control")
    assert summary["P4"] == ("inlineStr", "excluded control")
    about = {value for _, value in _xlsx_cells(path, 3).values()}
    assert "micrometer^3" in about
    assert "c" * 64 in about and "d" * 64 in about
    assert any("15 significant" in text for text in about)
    assert not list(tmp_path.glob(".vipp-export-*"))


def test_xlsx_nonfinite_small_large_signedzero_and_scalar_types(tmp_path):
    special = [
        None,
        "",
        True,
        False,
        float("nan"),
        float("inf"),
        float("-inf"),
        -0.0,
        5e-324,
        1.79e308,
        10**15,
        -(10**15),
        999_999_999_999_999,
        "001",
        "+SUM(1)",
        "@SUM(A1)",
        "\t=1",
        "µm\nline",
    ]
    table = TableData(("value",), tuple((value,) for value in special))
    path = export_measurement_collection(
        _collection(tmp_path, table), tmp_path / "special.xlsx"
    ).paths[0]
    cells = _xlsx_cells(path)
    assert "A2" not in cells
    assert cells["A3"] == ("inlineStr", "")
    assert cells["A4"] == ("b", "1")
    assert cells["A5"] == ("b", "0")
    for row, text in enumerate(
        [
            "nan",
            "inf",
            "-inf",
            "-0.0",
            "5e-324",
            "1.79e+308",
            str(10**15),
            str(-(10**15)),
        ],
        6,
    ):
        assert cells[f"A{row}"] == ("inlineStr", text)
    assert cells["A14"] == ("n", "999999999999999")
    assert cells["A15"] == ("inlineStr", "001")


@pytest.mark.parametrize(
    "constant,value", [("_EXCEL_ROWS", 2), ("_EXCEL_COLUMNS", 3), ("_EXCEL_TEXT", 10)]
)
def test_xlsx_limits_fail_before_creating_any_output(
    tmp_path, monkeypatch, constant, value
):
    monkeypatch.setattr(module, constant, value)
    with pytest.raises(MeasurementExportError, match="Excel"):
        export_measurement_collection(_collection(tmp_path), tmp_path / "out.xlsx")
    assert list(tmp_path.iterdir()) == []


def test_excel_string_utf16_limit_and_csv_no_truncation(tmp_path):
    text = "🙂" * 16_384
    collection = _collection(tmp_path, TableData(("text",), ((text,),)))
    with pytest.raises(MeasurementExportError, match="not be truncated"):
        export_measurement_collection(collection, tmp_path / "out.xlsx")
    path = export_measurement_collection(collection, tmp_path / "out.csv").paths[0]
    assert _read_csv(path)[0]["text"] == text


def test_revalidate_resident_collection_before_export(tmp_path):
    collection = _collection(tmp_path)
    bad = replace(collection, table=replace(collection.table, rows=()))
    with pytest.raises(MeasurementCollectionError, match="inventory"):
        export_measurement_collection(bad, tmp_path / "out.csv")
    with pytest.raises(MeasurementCollectionError, match="columns"):
        export_measurement_collection(
            collection, tmp_path / "out.csv", limits=CollectionLimits(max_columns=2)
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "target", ["original-one.csv", "original-two.csv", "never-read-one.csv"]
)
def test_recorded_inputs_and_results_cannot_be_overwritten(tmp_path, target):
    collection = _collection(tmp_path)
    if target == "never-read-one.csv":
        collection = replace(
            collection,
            items=(
                replace(collection.items[0], source_path=str(tmp_path / target)),
                *collection.items[1:],
            ),
        )
    path = tmp_path / target
    path.write_bytes(b"original")
    with pytest.raises(MeasurementExportError, match="recorded input or original"):
        export_measurement_collection(collection, path, overwrite=True)
    assert path.read_bytes() == b"original"


def test_protects_companion_and_every_recorded_source_path(tmp_path):
    collection = _collection(tmp_path)
    companion = tmp_path / "out-image-summary.csv"
    collection = replace(
        collection,
        items=(
            replace(collection.items[0], sources=({"path": str(companion)},)),
            *collection.items[1:],
        ),
    )
    with pytest.raises(MeasurementExportError, match="recorded input or original"):
        export_measurement_collection(collection, tmp_path / "out.csv", overwrite=True)
    assert not (tmp_path / "out.csv").exists()


def test_overwrite_requires_permission_for_all_destinations(tmp_path):
    collection = _collection(tmp_path)
    first, second = measurement_export_targets(tmp_path / "out.csv")
    second.write_bytes(b"original summary")
    with pytest.raises(FileExistsError):
        export_measurement_collection(collection, first)
    assert not first.exists() and second.read_bytes() == b"original summary"
    export_measurement_collection(collection, first, overwrite=True)
    assert len(_read_csv(first)) == 2 and len(_read_csv(second)) == 3
    assert not list(tmp_path.glob(".vipp-export-*"))


@pytest.mark.parametrize("existing", [False, True])
def test_second_publish_failure_rolls_back_entire_pair(tmp_path, monkeypatch, existing):
    collection = _collection(tmp_path)
    first, second = measurement_export_targets(tmp_path / "out.csv")
    if existing:
        first.write_bytes(b"old measurements")
        second.write_bytes(b"old inventory")
    original_link = os.link
    original_replace = os.replace

    def fail_link(source, target, *args, **kwargs):
        if Path(target) == second:
            raise OSError("Injected second publication failure")
        return original_link(source, target, *args, **kwargs)

    def fail_replace(source, target):
        if Path(target) == second and Path(source).name == second.name:
            raise OSError("Injected second publication failure")
        return original_replace(source, target)

    monkeypatch.setattr(module.os, "link", fail_link)
    monkeypatch.setattr(module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="Injected"):
        export_measurement_collection(collection, first, overwrite=existing)
    if existing:
        assert first.read_bytes() == b"old measurements"
        assert second.read_bytes() == b"old inventory"
    else:
        assert not first.exists() and not second.exists()
    assert not list(tmp_path.glob(".vipp-export-*"))


@pytest.mark.parametrize("format", ["csv", "xlsx"])
def test_cancel_during_writing_or_before_publication_leaves_old_files(tmp_path, format):
    collection = _collection(tmp_path)
    target = tmp_path / f"out.{format}"
    paths = measurement_export_targets(target)
    for path in paths:
        path.write_bytes(b"original")
    cancelled = threading.Event()

    def progress(fraction, message):
        if fraction >= 0.9:
            cancelled.set()

    with pytest.raises(OperationCancelled):
        export_measurement_collection(
            collection,
            target,
            overwrite=True,
            cancellation=cancelled,
            progress=progress,
        )
    assert all(path.read_bytes() == b"original" for path in paths)
    assert not list(tmp_path.glob(".vipp-export-*"))


def test_cancellation_inside_excel_rows_closes_and_removes_all_temporary_files(
    tmp_path,
):
    collection = _collection(
        tmp_path, TableData(("value",), tuple((index,) for index in range(600)))
    )
    cancelled = threading.Event()

    def progress(fraction, message):
        if 0.1 < fraction < 0.9:
            cancelled.set()

    with pytest.raises(OperationCancelled):
        export_measurement_collection(
            collection, tmp_path / "out.xlsx", cancellation=cancelled, progress=progress
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("existing", [False, True])
def test_destination_change_during_write_is_not_overwritten(tmp_path, existing):
    collection = _collection(tmp_path)
    target = tmp_path / "out.csv"
    if existing:
        target.write_text("previous", encoding="utf-8")

    def change_destination(fraction, message):
        if fraction == 0.95:
            target.write_text("new external work", encoding="utf-8")

    with pytest.raises((MeasurementExportError, FileExistsError)):
        export_measurement_collection(
            collection, target, overwrite=True, progress=change_destination
        )
    assert target.read_text(encoding="utf-8") == "new external work"
    assert not (tmp_path / "out-image-summary.csv").exists()
    assert not list(tmp_path.glob(".vipp-export-*"))


def test_publication_boundary_completes_pair_even_if_cancel_arrives(
    tmp_path, monkeypatch
):
    collection = _collection(tmp_path)
    event = threading.Event()
    original_link = os.link

    def link_and_cancel(*args, **kwargs):
        result = original_link(*args, **kwargs)
        event.set()
        return result

    monkeypatch.setattr(module.os, "link", link_and_cancel)
    result = export_measurement_collection(
        collection, tmp_path / "out.csv", cancellation=event
    )
    assert event.is_set() and all(path.is_file() for path in result.paths)


def test_invalid_unicode_rejected_without_partial_output(tmp_path):
    collection = _collection(tmp_path, TableData(("text",), (("\ud800",),)))
    with pytest.raises((MeasurementExportError, UnicodeError)):
        export_measurement_collection(collection, tmp_path / "out.csv")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("existing", [False, True])
def test_ui_review_revision_detects_changes_before_worker_starts(tmp_path, existing):
    collection = _collection(tmp_path)
    targets = measurement_export_targets(tmp_path / "out.csv")
    if existing:
        targets[0].write_bytes(b"reviewed old file")
    expected = measurement_export_destination_revisions(targets)
    targets[0].write_bytes(b"new work after the overwrite confirmation")
    with pytest.raises(MeasurementExportError, match="changed after your review"):
        export_measurement_collection(
            collection,
            targets[0],
            overwrite=True,
            expected_destination_revisions=expected,
        )
    assert targets[0].read_bytes() == b"new work after the overwrite confirmation"
    assert not targets[1].exists()


def test_ui_review_revision_allows_unchanged_existing_pair(tmp_path):
    collection = _collection(tmp_path)
    targets = measurement_export_targets(tmp_path / "out.csv")
    for target in targets:
        target.write_bytes(b"reviewed previous file")
    expected = measurement_export_destination_revisions(targets)
    exported = export_measurement_collection(
        collection,
        targets[0],
        overwrite=True,
        expected_destination_revisions=expected,
    )
    assert exported.paths == targets
    assert len(_read_csv(targets[0])) == 2
    assert len(_read_csv(targets[1])) == 3
    assert not list(tmp_path.glob(".vipp-export-*"))


def test_failed_rollback_retains_original_backup_and_explains_recovery(
    tmp_path, monkeypatch
):
    collection = _collection(tmp_path)
    first, second = measurement_export_targets(tmp_path / "out.csv")
    first.write_bytes(b"original measurements")
    second.write_bytes(b"original inventory")
    original_replace = os.replace

    def fail_publish_and_rollback(source, target):
        if Path(source).name == "previous-0" or Path(target) == second:
            raise OSError("Injected write or recovery failure")
        return original_replace(source, target)

    monkeypatch.setattr(module.os, "replace", fail_publish_and_rollback)
    with pytest.raises(
        MeasurementExportError, match="automatic recovery was incomplete"
    ) as exc:
        export_measurement_collection(collection, first, overwrite=True)
    (directory,) = tmp_path.glob(".vipp-export-*")
    assert str(directory) in str(exc.value)
    assert (directory / "previous-0").read_bytes() == b"original measurements"
    assert second.read_bytes() == b"original inventory"


def test_atomic_overwrite_keeps_previous_target_present_until_replace(
    tmp_path, monkeypatch
):
    collection = _collection(tmp_path)
    target = tmp_path / "out.xlsx"
    target.write_bytes(b"old workbook")
    original_replace = os.replace
    replacements = []

    def checked_replace(source, destination):
        if Path(destination) == target:
            assert target.read_bytes() == b"old workbook"
            replacements.append(target)
        return original_replace(source, destination)

    monkeypatch.setattr(module.os, "replace", checked_replace)
    export_measurement_collection(collection, target, overwrite=True)
    assert replacements == [target]
    assert target.read_bytes().startswith(b"PK")


def test_locked_commit_does_not_retry_over_newer_unreviewed_destination(
    tmp_path, monkeypatch
):
    collection = _collection(tmp_path)
    target = tmp_path / "out.csv"
    target.write_bytes(b"reviewed original")
    original_replace = os.replace
    attempts = []

    def locked_replace(source, destination):
        if Path(destination) == target and Path(source).name == target.name:
            attempts.append(Path(source))
            if len(attempts) == 1:
                newer = tmp_path / "external-work.tmp"
                newer.write_bytes(b"new unreviewed work")
                original_replace(newer, target)
                raise PermissionError("File was locked during guarded commit")
        return original_replace(source, destination)

    monkeypatch.setattr(module.os, "replace", locked_replace)
    with pytest.raises(
        MeasurementExportError, match="automatic recovery was incomplete"
    ):
        export_measurement_collection(collection, target, overwrite=True)
    assert len(attempts) == 1
    assert target.read_bytes() == b"new unreviewed work"
    assert not (tmp_path / "out-image-summary.csv").exists()
    (directory,) = tmp_path.glob(".vipp-export-*")
    assert (directory / "previous-0").read_bytes() == b"reviewed original"
