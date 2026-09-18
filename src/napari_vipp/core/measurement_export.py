"""Direct, cancellable exports of a reviewed resident measurement collection.

No source image or original result file is opened. Delimited text is faithful
data, not a typed snapshot: spreadsheet software may interpret its strings or
numbers on import. XLSX stores strings literally and documents Excel precision.
Publication of a workbook is atomic; a CSV/TSV pair is staged together and rolled
back on ordinary publication failures, but is not a crash-atomic transaction.
"""

from __future__ import annotations

import csv
import math
import os
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from napari_vipp.core.measurement_collection import (
    CollectionLimits,
    MeasurementCollection,
    MeasurementCollectionError,
    _check,
    _ordinary_path,
    _snapshot_document,
    _validated_snapshot,
)
from napari_vipp.core.tables import TableData

_EXCEL_ROWS = 1_048_576
_EXCEL_COLUMNS = 16_384
_EXCEL_TEXT = 32_767
_EXCEL_INTEGER = 999_999_999_999_999
_EXCEL_MIN_NUMBER = 2.2250738585072014e-308
_EXCEL_MAX_NUMBER = 9.99999999999999e307
_FORMATS = frozenset({"csv", "tsv", "xlsx"})
_DEFAULT_LIMITS = CollectionLimits()
_SUMMARY_NAMES = (
    "Item key",
    "Item number",
    "Batch item",
    "Included",
    "Status",
    "Measurement rows",
    "Source file",
    "Source count",
    "Result file",
    "Result SHA256",
    "Run ID",
    "Effective workflow SHA256",
    "Resumed from run",
    "Notes",
)
_CSV_NOTES = (
    "CSV/TSV preserves column names and values as plain text. Import columns as "
    "text when needed: spreadsheet software can interpret formula-like text, "
    "URLs, dates or long identifiers. XLSX is safer for opening in Excel.",
    "CSV/TSV does not preserve scalar types or units, and both missing values "
    "and empty strings appear blank. Save a VIPP collection for an exact typed "
    "snapshot; XLSX includes recorded units on About this collection.",
    "Non-finite numbers are written explicitly as nan, inf and -inf.",
)
_XLSX_NOTES = (
    "Strings are literal text, never formulas or hyperlinks. Integers longer "
    "than 15 digits are text so that identifiers are not rounded.",
    "Finite measurements are Excel numbers, with approximately 15 significant "
    "digits. Save a VIPP collection for exact values and scalar types.",
    "nan, inf, -inf, negative zero and values outside Excel's numeric range "
    "are explicit text rather than blank cells, errors or silently rounded zeros.",
)
_COMMON_NOTES = (
    "Measurements contains only included rows. Image summary retains every "
    "image, including zero-row successes and reviewed exclusions. A blank row "
    "count means unavailable, not zero.",
    "Image annotations already present in measurement columns are summarized "
    "only when rows are available. Mixed values are marked, not replaced by "
    "the first value. No annotations are inferred for empty or excluded images.",
    "This exports the reviewed collection in memory. No images or original "
    "result files are read or recalculated; recorded hashes describe the "
    "evidence checked when the collection was created.",
)


class MeasurementExportError(MeasurementCollectionError):
    """An export cannot be represented or published safely."""


@dataclass(frozen=True)
class MeasurementExportResult:
    paths: tuple[Path, ...]
    notes: tuple[str, ...]


def measurement_export_targets(
    path: str | Path, *, format: str | None = None, include_image_summary: bool = True
) -> tuple[Path, ...]:
    """Plan all destinations without reading files, for overwrite confirmation."""
    if not str(path).strip():
        raise MeasurementExportError("Choose an export file name.")
    target = Path(path)
    chosen = format if format is not None else target.suffix.removeprefix(".").lower()
    if chosen not in _FORMATS or type(include_image_summary) is not bool:
        raise MeasurementExportError("Choose CSV, TSV or XLSX for the export.")
    if not target.suffix:
        target = target.with_suffix(f".{chosen}")
    elif target.suffix.lower() != f".{chosen}":
        raise MeasurementExportError("The file extension must match the export format.")
    if ".." in target.parts or str(target).startswith(("\\\\", "//")):
        raise MeasurementExportError("Choose a direct local export path.")
    target = target.absolute()
    if chosen == "xlsx" or not include_image_summary:
        return (target,)
    return (target, target.with_name(f"{target.stem}-image-summary{target.suffix}"))


def _path_key(path):
    # Lexical normalization only: recorded image/result paths are never opened.
    return os.path.normcase(os.path.abspath(os.path.normpath(str(path))))


def _validate_destinations(collection, targets, overwrite):
    protected = {
        _path_key(path)
        for item in collection.items
        for path in (
            item.source_path,
            item.result_path,
            *(source.get("path", "") for source in item.sources),
        )
        if path
    }
    for target in targets:
        _ordinary_path(target, may_be_missing=True)
        if _path_key(target) in protected:
            raise MeasurementExportError(
                "Choose a new file name: an export must not overwrite a recorded "
                f"input or original result file ({target.name})."
            )
        if target.exists():
            if not target.is_file():
                raise MeasurementExportError(
                    f"Export destination is not a file: {target}"
                )
            if not overwrite:
                raise FileExistsError(f"Export already exists: {target}")


def _value_key(value):
    # Distinguish None/empty, bool/int, negative zero and NaN consistently.
    return (type(value).__name__, value.hex() if type(value) is float else value)


def _image_summary(collection, cancellation):
    table = collection.table
    annotation_names = tuple(
        name
        for name in table.columns
        if name.casefold() in {"condition", "sample", "replicate"}
        or name in collection.provenance["annotation_columns"]
    )
    positions = {name: table.columns.index(name) for name in annotation_names}
    item_position = table.columns.index("_vipp_item_key")
    observed = {}
    for row in table.rows:
        _check(cancellation)
        item_values = observed.setdefault(row[item_position], {})
        for name, position in positions.items():
            value = row[position]
            previous = item_values.get(name)
            if previous is None:
                item_values[name] = (_value_key(value), value, False)
            elif previous[0] != _value_key(value):
                item_values[name] = (previous[0], previous[1], True)
    rows = []
    for item in collection.items:
        _check(cancellation)
        annotations = []
        for name in annotation_names:
            if name in collection.annotations.get(item.key, {}):
                value = collection.annotations[item.key][name]
            else:
                evidence = observed.get(item.key, {}).get(name)
                value = (
                    None
                    if evidence is None
                    else "Mixed (see Measurements)"
                    if evidence[2]
                    else evidence[1]
                )
            annotations.append(value)
        rows.append(
            (
                item.key,
                item.index,
                item.batch_id,
                item.included,
                item.status,
                item.row_count,
                item.source_path,
                len(item.sources) or int(bool(item.source_path)),
                item.result_path,
                item.result_sha256,
                collection.provenance["run_id"],
                item.effective_workflow_sha256,
                item.resumed_from_run_id,
                item.message,
                *annotations,
            )
        )
    return TableData(
        (*_SUMMARY_NAMES, *(f"Annotation: {name}" for name in annotation_names)),
        tuple(rows),
        name="Image summary",
    )


def _about(collection, notes):
    rows = [
        ("Collection", collection.table.name or "Batch measurements"),
        ("Measurement type", collection.table.table_kind),
        ("Measurement rows", collection.table.row_count),
        ("Images in inventory", len(collection.items)),
        ("Included images", sum(item.included for item in collection.items)),
        ("Excluded images", sum(not item.included for item in collection.items)),
        (
            "Zero-row successes",
            sum(item.status == "empty" for item in collection.items),
        ),
        ("Recorded run ID", collection.provenance["run_id"]),
        ("Output node", collection.provenance["output_node_id"]),
        ("Original workflow SHA256", collection.provenance["workflow_sha256"]),
        ("Original manifest SHA256", collection.provenance["manifest_sha256"]),
        (
            "Loaded VIPP snapshot SHA256",
            collection.file_sha256 or "Not loaded from a file",
        ),
        ("Exclusions reviewed", collection.provenance["reviewed_exclusions"]),
        (
            "Excel limits",
            "1,048,576 rows including headers; 16,384 columns; "
            "32,767 UTF-16 units per cell. Oversize exports are rejected, "
            "not truncated.",
        ),
    ]
    rows.extend(
        (f"Important note {index}", note) for index, note in enumerate(notes, 1)
    )
    rows.extend(
        (f"Unit — {name}", collection.table.unit_for(name) or "No unit recorded")
        for name in collection.table.columns
    )
    return TableData(("About this collection", "Details"), tuple(rows))


def _excel_text(value):
    if type(value) is int and abs(value) > _EXCEL_INTEGER:
        return str(value)
    if type(value) is float:
        if not math.isfinite(value):
            return str(value)
        if value == 0 and math.copysign(1, value) < 0:
            return "-0.0"
        if value and not _EXCEL_MIN_NUMBER <= abs(value) <= _EXCEL_MAX_NUMBER:
            return repr(value)
    return value


def _validate_tables(tables, *, excel, cancellation):
    for title, table in tables:
        if excel and (
            table.row_count + 1 > _EXCEL_ROWS or table.column_count > _EXCEL_COLUMNS
        ):
            raise MeasurementExportError(
                f"{title} exceeds Excel's row or column limit. Export CSV/TSV instead."
            )
        _validate_text_row(table.columns, title, excel=excel)
        for row in table.rows:
            _check(cancellation)
            _validate_text_row(row, title, excel=excel)


def _validate_text_row(row, title, *, excel):
    for value in row:
        value = _excel_text(value) if excel else value
        if isinstance(value, str):
            try:
                value.encode("utf-8")
                length = len(value.encode("utf-16-le")) // 2 if excel else len(value)
            except UnicodeError as exc:
                raise MeasurementExportError(
                    f"{title} contains invalid Unicode text."
                ) from exc
            if excel and length > _EXCEL_TEXT:
                raise MeasurementExportError(
                    f"{title} contains text longer than Excel's cell limit. "
                    "Export CSV/TSV instead; text will not be truncated."
                )


def _notify(progress, value, message):
    if progress is not None:
        progress(value, message)


def _write_delimited(path, table, separator, cancellation, tick):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter=separator, lineterminator="\n")
        writer.writerow(table.columns)
        for row in table.rows:
            _check(cancellation)
            writer.writerow(row)
            tick()
        stream.flush()
        os.fsync(stream.fileno())


def _write_xlsx(path, tables, cancellation, tick, temporary_dir):
    try:
        import xlsxwriter
    except ImportError as exc:
        raise MeasurementExportError(
            "Excel export requires the XlsxWriter package."
        ) from exc
    # Dedicated temp directory also removes worksheet fragments on errors or cancel.
    workbook = xlsxwriter.Workbook(
        path,
        {
            "constant_memory": True,
            "tmpdir": str(temporary_dir),
            "strings_to_formulas": False,
            "strings_to_urls": False,
        },
    )
    try:
        heading = workbook.add_format(
            {
                "bold": True,
                "font_color": "#FFFFFF",
                "bg_color": "#176B87",
                "text_wrap": True,
                "valign": "top",
            }
        )
        wrapped = workbook.add_format({"text_wrap": True, "valign": "top"})
        for title, table in tables:
            worksheet = workbook.add_worksheet(title)
            worksheet.freeze_panes(1, 0)
            worksheet.set_row(0, 30)
            for column, name in enumerate(table.columns):
                worksheet.write_string(0, column, name, heading)
                width = 58 if title == "About this collection" and column == 1 else 24
                worksheet.set_column(column, column, width, wrapped)
            if title != "About this collection":
                worksheet.autofilter(0, 0, table.row_count, table.column_count - 1)
            for index, row in enumerate(table.rows, 1):
                _check(cancellation)
                for column, value in enumerate(row):
                    value = _excel_text(value)
                    if value is None:
                        continue
                    if type(value) is str:
                        code = worksheet.write_string(index, column, value)
                    elif type(value) is bool:
                        code = worksheet.write_boolean(index, column, value)
                    else:
                        code = worksheet.write_number(index, column, value)
                    if code:
                        raise MeasurementExportError(
                            f"Excel could not write {title}, row {index + 1}, "
                            f"column {column + 1}; no output was published."
                        )
                tick()
    finally:
        workbook.close()
    _check(cancellation)
    with path.open("r+b") as stream:
        os.fsync(stream.fileno())


class _RecoveryFailed(MeasurementExportError):
    """Retain the staging directory because a backup needs manual recovery."""


def _fingerprint(path):
    try:
        info = path.stat()
    except FileNotFoundError:
        return None
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


def measurement_export_destination_revisions(paths):
    """Stat-only revisions for the exact destinations the user will confirm.

    Each revision is ``(device, inode, size_bytes, modified_nanoseconds)`` or
    ``None`` for a missing file. Pass the returned tuple to the export worker.
    """
    revisions = []
    for path in paths:
        target = Path(path)
        _ordinary_path(target, may_be_missing=True)
        revisions.append(_fingerprint(target))
    return tuple(revisions)


def _publish(staged, targets, directory, expected):
    backups = {}
    published = {}
    try:
        for index, (source, target) in enumerate(zip(staged, targets, strict=True)):
            _ordinary_path(target, may_be_missing=True)
            if _fingerprint(target) != expected[index]:
                raise MeasurementExportError(
                    f"Export destination changed while writing: {target}. Try again."
                )
            if expected[index] is not None:
                backup = directory / f"previous-{index}"
                # A same-filesystem link keeps the old inode available for
                # recovery without opening/copying its potentially large data.
                os.link(target, backup)
                backups[target] = backup
                # Do not retry after a lock: the reviewed destination revision
                # could change while waiting, making a delayed replace unsafe.
                os.replace(source, target)
            else:
                # No-clobber publication: a newly appearing file is never replaced.
                os.link(source, target)
            published[target] = _fingerprint(target)
    except BaseException as exc:
        failed_recovery = []
        for target in reversed(published):
            try:
                if _fingerprint(target) != published[target]:
                    raise OSError("Published file changed during recovery.")
                if target in backups:
                    os.replace(backups.pop(target), target)
                else:
                    target.unlink()
            except OSError:
                failed_recovery.append(str(target))
        for target, backup in backups.items():
            try:
                if not target.exists() or not os.path.samefile(backup, target):
                    raise OSError("Original file needs manual recovery.")
                backup.unlink()
            except OSError:
                failed_recovery.append(str(backup))
        if failed_recovery:
            raise _RecoveryFailed(
                "Export failed and automatic recovery was incomplete. Original "
                f"files are retained in {directory}. "
                f"Review: {', '.join(failed_recovery)}"
            ) from exc
        raise


def export_measurement_collection(
    collection: MeasurementCollection,
    path: str | Path,
    *,
    format: str | None = None,
    include_image_summary: bool = True,
    overwrite: bool = False,
    expected_destination_revisions=None,
    cancellation=None,
    progress: Callable[[float, str], None] | None = None,
    limits: CollectionLimits = _DEFAULT_LIMITS,
) -> MeasurementExportResult:
    """Export reviewed rows and inventory, without rerunning or reading inputs.

    All destinations are refused unless ``overwrite`` explicitly permits replacing
    existing files. Cancellation before the publication boundary leaves existing
    files untouched; publication itself is completed or rolled back as a unit.
    CSV/TSV companion files cannot be made crash-atomic as a pair. Recovery files
    are retained and identified in the exception if operating-system recovery fails.
    """
    _check(cancellation)
    if type(overwrite) is not bool:
        raise MeasurementExportError("Overwrite permission must be explicit.")
    targets = measurement_export_targets(
        path, format=format, include_image_summary=include_image_summary
    )
    chosen = targets[0].suffix.lower().removeprefix(".")
    _validate_destinations(collection, targets, overwrite)
    current = measurement_export_destination_revisions(targets)
    expected = (
        current
        if expected_destination_revisions is None
        else expected_destination_revisions
    )
    if (
        type(expected) is not tuple
        or len(expected) != len(targets)
        or any(
            revision is not None
            and (
                type(revision) is not tuple
                or len(revision) != 4
                or any(type(value) is not int for value in revision)
            )
            for revision in expected
        )
    ):
        raise MeasurementExportError("Invalid export destination review revisions.")
    if expected != current:
        raise MeasurementExportError(
            "An export destination changed after your review. Choose Export results "
            "again and review the current files."
        )
    _notify(progress, 0.0, "Validating the reviewed collection")
    # Reuse the native snapshot's exact schema/identity/inclusion checks. This
    # validates resident values only, never the paths recorded in the inventory.
    _validated_snapshot(
        _snapshot_document(collection, cancellation), limits, cancellation
    )
    notes = (*(_XLSX_NOTES if chosen == "xlsx" else _CSV_NOTES), *_COMMON_NOTES)
    if chosen != "xlsx" and not include_image_summary:
        notes = (
            *_CSV_NOTES,
            "No image-summary companion was requested. Only included measurement "
            "rows are exported: zero-row images, exclusions and their annotations "
            "are absent from this file. Save an image summary or VIPP collection "
            "to retain the complete image inventory.",
            _COMMON_NOTES[-1],
        )
    tables = [("Measurements", collection.table)]
    if chosen == "xlsx" or include_image_summary:
        tables.append(("Image summary", _image_summary(collection, cancellation)))
    if chosen == "xlsx":
        tables.append(("About this collection", _about(collection, notes)))
    _validate_tables(tables, excel=chosen == "xlsx", cancellation=cancellation)
    _check(cancellation)
    _validate_destinations(collection, targets, overwrite)
    targets[0].parent.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=".vipp-export-", dir=targets[0].parent))
    total_rows = max(1, sum(table.row_count for _, table in tables))
    written = 0

    def tick():
        nonlocal written
        written += 1
        if written == total_rows or written % 256 == 0:
            _notify(progress, 0.1 + 0.8 * written / total_rows, "Writing results")

    keep_recovery = False
    try:
        staged = tuple(directory / target.name for target in targets)
        if chosen == "xlsx":
            _write_xlsx(staged[0], tables, cancellation, tick, directory)
        else:
            for temporary, (_, table) in zip(staged, tables, strict=True):
                _write_delimited(
                    temporary,
                    table,
                    "\t" if chosen == "tsv" else ",",
                    cancellation,
                    tick,
                )
        _notify(progress, 0.95, "Publishing results")
        _validate_destinations(collection, targets, overwrite)
        _check(cancellation)
        _publish(staged, targets, directory, expected)
    except _RecoveryFailed:
        keep_recovery = True
        raise
    finally:
        # An unrestored backup is the user's original file, not disposable temp.
        if not keep_recovery:
            shutil.rmtree(directory)
    _notify(progress, 1.0, "Results exported")
    return MeasurementExportResult(targets, notes)


__all__ = [
    "MeasurementExportError",
    "MeasurementExportResult",
    "export_measurement_collection",
    "measurement_export_destination_revisions",
    "measurement_export_targets",
]
