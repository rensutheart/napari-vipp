"""Strict, immutable snapshots of manifest-backed batch measurement tables.

Checksums detect accidental changes, not a malicious author who reseals data.
Only ordinary, contained result files are read; source images are never read.
CSV values are decoded exclusively from recorded type evidence, never inferred.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType

from napari_vipp.core.atomic_io import atomic_replace
from napari_vipp.core.batch_resume import document_digest, seal_document
from napari_vipp.core.progress import OperationCancelled
from napari_vipp.core.source_identity import (
    SourceChangedError,
    capture_local_source_identity,
)
from napari_vipp.core.tables import TableData

_SCHEMA = "napari-vipp-measurement-collection"
_SUFFIX = ".vipp-results.json"
_IDENTITY_COLUMNS = ("_vipp_run_id", "_vipp_item_key", "_vipp_batch_id", "_vipp_row")
_TYPES = frozenset({"none", "bool", "int", "float", "str"})
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_UNSPECIFIED_DESTINATION_REVISION = object()


class MeasurementCollectionError(ValueError):
    """Evidence is incomplete, changed, incompatible, or unsafe to collect."""


class _ChangedResult(MeasurementCollectionError):
    """Verified bytes changed while collection was in progress."""


@dataclass(frozen=True)
class CollectionLimits:
    max_manifest_bytes: int = 64 * 1024**2
    max_file_bytes: int = 256 * 1024**2
    max_total_bytes: int = 512 * 1024**2
    max_items: int = 100_000
    max_rows: int = 1_000_000
    max_cells: int = 10_000_000
    max_columns: int = 1024
    max_text_chars: int = 1_000_000

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in vars(self).values()):
            raise ValueError("Collection limits must be positive integers.")


_DEFAULT_LIMITS = CollectionLimits()


@dataclass(frozen=True)
class MeasurementOutput:
    node_id: str
    title: str
    tag: str


@dataclass(frozen=True)
class CollectionItem:
    key: str
    index: int
    batch_id: str
    status: str
    message: str = ""
    row_count: int | None = None
    source_path: str = ""
    result_path: str = ""
    result_sha256: str = ""
    table: TableData | None = field(default=None, repr=False)
    included: bool = False
    sources: tuple[Mapping[str, object], ...] = ()
    parameter_overrides: Mapping[str, object] = field(default_factory=dict)
    effective_workflow_sha256: str = ""
    execution_provenance_sha256: str = ""
    resumed_from_run_id: str = ""

    def __post_init__(self):
        object.__setattr__(self, "sources", _freeze(self.sources))
        object.__setattr__(
            self, "parameter_overrides", _freeze(self.parameter_overrides)
        )


@dataclass(frozen=True)
class MeasurementPreview:
    output: MeasurementOutput
    items: tuple[CollectionItem, ...]
    run_id: str
    manifest_sha256: str
    workflow_sha256: str
    limits: CollectionLimits = field(default_factory=CollectionLimits, repr=False)


@dataclass(frozen=True)
class MeasurementCollection:
    table: TableData
    items: tuple[CollectionItem, ...]
    annotations: Mapping[str, Mapping[str, str]]
    provenance: Mapping[str, object]
    file_sha256: str = ""

    def __post_init__(self):
        object.__setattr__(self, "annotations", _freeze(self.annotations))
        object.__setattr__(self, "provenance", _freeze(self.provenance))


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value):
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_thaw(item) for item in value]
    return value


def _check(cancellation):
    cancelled = (
        cancellation()
        if callable(cancellation)
        else cancellation.is_set()
        if cancellation is not None
        else False
    )
    if cancelled:
        raise OperationCancelled("Measurement collection cancelled.")


def _text(value, label, *, empty=False, limit=1_000_000):
    if type(value) is not str or (not empty and not value) or len(value) > limit:
        raise MeasurementCollectionError(f"Invalid {label}.")
    return value


def _count(value, label, maximum):
    if type(value) is not int or not 0 <= value <= maximum:
        raise MeasurementCollectionError(f"Invalid or excessive {label}.")
    return value


def _ordinary_path(path: Path, *, may_be_missing=False):
    if ".." in path.parts or str(path).startswith(("\\\\", "//")):
        raise MeasurementCollectionError("Result paths must be direct local paths.")
    for component in (path, *path.parents):
        try:
            info = component.lstat()
        except FileNotFoundError:
            if may_be_missing:
                continue
            raise
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise MeasurementCollectionError(
                "Linked or redirected paths are not allowed."
            )
    if not may_be_missing and not path.is_file():
        raise MeasurementCollectionError("Expected an ordinary result file.")


def capture_measurement_collection_destination_revision(
    path: str | Path,
) -> tuple[int, int, int, int] | None:
    """Capture a local save target before approval, without reading file bytes.

    ``None`` records an absent destination. Otherwise the immutable fields are
    size, nanosecond modification time, device and inode. This is a bounded
    accidental-change check, not a content hash or an adversarial lock.
    """
    path = Path(path)
    _ordinary_path(path, may_be_missing=True)
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
        raise MeasurementCollectionError("Expected an ordinary collection file.")
    return (info.st_size, info.st_mtime_ns, info.st_dev, info.st_ino)


def _read_bytes(path, maximum, cancellation=None):
    path = Path(path)
    _ordinary_path(path)
    before = path.stat()
    if before.st_size > maximum:
        raise MeasurementCollectionError("File exceeds the collection size limit.")
    parts = []
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(min(1024**2, maximum + 1)):
            _check(cancellation)
            size += len(chunk)
            if size > maximum:
                raise MeasurementCollectionError(
                    "File exceeds the collection size limit."
                )
            parts.append(chunk)
        after_open = os.fstat(stream.fileno())
    after = path.stat()

    def fields(item):
        return (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)

    if fields(before) != fields(after_open) or fields(before) != fields(after):
        raise _ChangedResult("File changed while it was being read.")
    _check(cancellation)
    return b"".join(parts)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise MeasurementCollectionError(f"Duplicate JSON field {key!r}.")
        result[key] = value
    return result


def _json_document(data):
    def invalid_constant(value):
        raise MeasurementCollectionError(f"Unsafe JSON numeric literal {value}.")

    try:
        result = json.loads(
            data, object_pairs_hook=_unique, parse_constant=invalid_constant
        )
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise MeasurementCollectionError(f"Invalid JSON document: {exc}") from exc
    if not isinstance(result, dict):
        raise MeasurementCollectionError("Expected a JSON object.")
    return result


def _verified_document(document):
    try:
        expected = document.get("integrity_sha256")
        if not isinstance(expected, str) or not _HASH.fullmatch(expected):
            raise MeasurementCollectionError(
                "The document has no valid integrity seal."
            )
        if seal_document(document)["integrity_sha256"] != expected:
            raise MeasurementCollectionError("Document integrity check failed.")
    except (TypeError, ValueError, RecursionError) as exc:
        raise MeasurementCollectionError(f"Invalid document integrity: {exc}") from exc
    return document


def _manifest(manifest, limits, cancellation=None):
    if isinstance(manifest, (str, Path)):
        document = _json_document(
            _read_bytes(manifest, limits.max_manifest_bytes, cancellation)
        )
    elif hasattr(manifest, "to_dict"):
        document = manifest.to_dict()
    elif isinstance(manifest, Mapping):
        document = _thaw(manifest)
    else:
        raise MeasurementCollectionError("Select a completed batch manifest.")
    _verified_document(document)
    if (
        document.get("type") != "napari-vipp-batch-manifest"
        or document.get("version") != 6
    ):
        raise MeasurementCollectionError("Unsupported batch manifest version.")
    if not document.get("finished_at"):
        raise MeasurementCollectionError("Wait until the batch has finished.")
    _text(document["finished_at"], "batch completion time")
    items = document.get("items")
    if not isinstance(items, list) or len(items) > limits.max_items:
        raise MeasurementCollectionError("Invalid or excessive batch item inventory.")
    _text(document.get("run_id"), "run identity")
    seen_indices, seen_ids = set(), set()
    for item in items:
        if not isinstance(item, dict):
            raise MeasurementCollectionError("Invalid batch item record.")
        index = _count(item.get("index"), "item index", limits.max_items)
        batch_id = _text(item.get("batch_id"), "batch item identity")
        if index < 1 or index in seen_indices or batch_id in seen_ids:
            raise MeasurementCollectionError("Duplicate batch/resume item identity.")
        seen_indices.add(index)
        seen_ids.add(batch_id)
        if not isinstance(item.get("outputs"), list):
            raise MeasurementCollectionError("Invalid batch output inventory.")
    return document


def _optional_hash(value, label):
    if type(value) is not str or (value and not _HASH.fullmatch(value)):
        raise MeasurementCollectionError(f"Invalid {label} hash.")
    return value


def _source_evidence(sources, limits, *, snapshot=False):
    if not isinstance(sources, list) or len(sources) > limits.max_columns:
        raise MeasurementCollectionError("Invalid or excessive source inventory.")
    result = []
    allowed = {
        "node_id",
        "title",
        "role",
        "path",
        "identity",
        "series",
        "selector",
        "source_item_record_sha256",
    }
    for source in sources:
        if not isinstance(source, dict) or (snapshot and set(source) != allowed):
            raise MeasurementCollectionError("Invalid source evidence record.")
        item = {
            key: _text(
                source.get(key, ""),
                f"source {key}",
                empty=True,
                limit=limits.max_text_chars,
            )
            for key in ("node_id", "title", "role", "path")
        }
        raw_identity = source.get("identity", {})
        if not isinstance(raw_identity, dict):
            raise MeasurementCollectionError("Invalid source content identity.")
        identity = {}
        if raw_identity.get("sha256"):
            identity = {
                "sha256": _optional_hash(raw_identity["sha256"], "source content"),
                "kind": _text(raw_identity.get("kind"), "source identity kind"),
                "size_bytes": _count(
                    raw_identity.get("size_bytes"), "source size", 2**128
                ),
                "regular_file_count": _count(
                    raw_identity.get("regular_file_count"), "source members", 2**64
                ),
            }
        if snapshot and identity != raw_identity:
            raise MeasurementCollectionError("Unexpected source identity fields.")
        item["identity"] = identity
        raw_series = source.get("series", {})
        if not isinstance(raw_series, dict):
            raise MeasurementCollectionError("Invalid recorded source series.")
        series = {}
        if raw_series:
            series = {
                "index": _count(raw_series.get("index"), "series index", 2**32),
                "key": _text(raw_series.get("key", ""), "series key", empty=True),
                "name": _text(raw_series.get("name", ""), "series name", empty=True),
            }
        if snapshot and series != raw_series:
            raise MeasurementCollectionError("Unexpected source series fields.")
        item["series"] = series
        source_item = source.get("source_item", {})
        if not isinstance(source_item, dict):
            raise MeasurementCollectionError("Invalid source-item evidence.")
        raw_selector = (
            source.get("selector", {}) if snapshot else source_item.get("selector", {})
        )
        selector = {}
        if raw_selector:
            from napari_vipp.core.source_items import SourceItemSelector

            selector = SourceItemSelector.from_dict(raw_selector).to_dict()
        item["selector"] = selector
        item["source_item_record_sha256"] = (
            _optional_hash(
                source.get("source_item_record_sha256", ""), "source-item record"
            )
            if snapshot
            else document_digest(source_item)
            if source_item
            else ""
        )
        result.append(item)
    return tuple(result)


def _override_evidence(value, limits):
    if value == {}:
        return {}
    if not isinstance(value, dict) or set(value) != {
        "identity",
        "source_item_key",
        "values",
    }:
        raise MeasurementCollectionError("Invalid parameter override evidence.")
    values = value["values"]
    if not isinstance(values, list) or len(values) > limits.max_columns:
        raise MeasurementCollectionError("Excessive parameter override evidence.")
    records = []
    for entry in values:
        if not isinstance(entry, dict) or set(entry) != {
            "node_id",
            "operation_id",
            "parameter",
            "workflow_value",
            "resolved_value",
        }:
            raise MeasurementCollectionError("Invalid parameter override record.")
        record = {
            name: _text(entry[name], f"override {name}", limit=limits.max_text_chars)
            for name in ("node_id", "operation_id", "parameter")
        }
        for name in ("workflow_value", "resolved_value"):
            item = entry[name]
            _type_tag(item)
            if isinstance(item, float) and not (-float("inf") < item < float("inf")):
                raise MeasurementCollectionError("Parameter overrides must be finite.")
            if isinstance(item, str):
                _text(item, "override text", empty=True, limit=limits.max_text_chars)
            record[name] = item
        records.append(record)
    return {
        "identity": _text(value["identity"], "override identity"),
        "source_item_key": _optional_hash(
            value["source_item_key"], "override source item"
        ),
        "values": records,
    }


def available_measurement_outputs(manifest, *, limits=_DEFAULT_LIMITS):
    document = _manifest(manifest, limits)
    outputs = {}
    for item in document["items"]:
        for output in item["outputs"]:
            if output.get("kind") == "table":
                node_id = _text(output.get("node_id"), "output node identity")
                outputs.setdefault(
                    node_id,
                    MeasurementOutput(
                        node_id,
                        str(output.get("node_title", node_id)),
                        str(output.get("tag", "")),
                    ),
                )
    return tuple(outputs.values())


def table_measurement_metadata(table: TableData, *, cancellation=None) -> dict:
    """Capture lossless scalar type evidence before CSV/TSV publication.

    Column-major run-length encoding preserves mixed None/empty/int/string cells
    without embedding measurement values in the manifest. Unsupported values
    leave the normal export intact but explicitly disable later collection.
    """
    try:
        _check(cancellation)
        columns = _columns(table.columns, _DEFAULT_LIMITS)
        if (
            table.row_count > _DEFAULT_LIMITS.max_rows
            or table.row_count * len(columns) > _DEFAULT_LIMITS.max_cells
        ):
            raise MeasurementCollectionError(
                "Table exceeds exact collection evidence limits."
            )
        runs = [[] for _ in columns]
        types = [set() for _ in columns]
        run_count = 0
        for row in table.rows:
            _check(cancellation)
            if len(row) != len(columns):
                raise MeasurementCollectionError("Inconsistent table row width.")
            for index, value in enumerate(row):
                tag = _type_tag(value)
                types[index].add(tag)
                if runs[index] and runs[index][-1][0] == tag:
                    runs[index][-1][1] += 1
                else:
                    runs[index].append([tag, 1])
                    run_count += 1
                    if run_count > 100_000:
                        raise MeasurementCollectionError(
                            "Table type evidence exceeds its bounded manifest budget."
                        )
        units = _units(table.column_units, columns)
        return {
            "version": 1,
            "available": True,
            "columns": list(columns),
            "row_count": table.row_count,
            "column_units": [list(item) for item in units],
            "column_types": [sorted(tags) for tags in types],
            "cell_types": runs,
            "name": table.name,
            "table_kind": table.table_kind,
        }
    except (MeasurementCollectionError, TypeError) as exc:
        return {"version": 1, "available": False, "reason": str(exc)}


def _type_tag(value):
    if value is None:
        return "none"
    name = type(value).__name__
    if type(value) not in (bool, int, float, str):
        raise MeasurementCollectionError(
            f"Unsupported exact table scalar type: {name}."
        )
    return name


def _columns(value, limits):
    if not isinstance(value, (tuple, list)) or not 0 < len(value) <= limits.max_columns:
        raise MeasurementCollectionError("Invalid table columns.")
    columns = tuple(
        _text(name, "column name", limit=limits.max_text_chars) for name in value
    )
    if len(set(columns)) != len(columns):
        raise MeasurementCollectionError("Duplicate table columns.")
    return columns


def _units(value, columns):
    if not isinstance(value, (tuple, list)):
        raise MeasurementCollectionError("Missing exact column unit evidence.")
    result = []
    for pair in value:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise MeasurementCollectionError("Invalid column units.")
        name, unit = pair
        if name not in columns or type(unit) is not str or name in dict(result):
            raise MeasurementCollectionError("Invalid or conflicting column units.")
        result.append((name, unit))
    return tuple(result)


def _decode_cell(text, tag, limits):
    _text(text, "table cell", empty=True, limit=limits.max_text_chars)
    if tag == "str":
        return text
    if tag == "none" and text == "":
        return None
    if tag == "bool" and text in {"True", "False"}:
        return text == "True"
    if tag == "int" and re.fullmatch(r"-?(?:0|[1-9][0-9]*)", text):
        return int(text)
    if tag == "float":
        try:
            value = float(text)
        except ValueError:
            pass
        else:
            if str(value) == text:
                return value
    raise MeasurementCollectionError(f"Cell does not match its recorded {tag} type.")


def _read_table(data, metadata, format, limits, cancellation):
    if (
        not isinstance(metadata, dict)
        or metadata.get("version") != 1
        or metadata.get("available") is not True
    ):
        raise MeasurementCollectionError(
            "Exact table types and units were not recorded; "
            "rerun this batch to collect it."
        )
    columns = _columns(metadata.get("columns"), limits)
    count = _count(metadata.get("row_count"), "row count", limits.max_rows)
    if count * len(columns) > limits.max_cells:
        raise MeasurementCollectionError("Table exceeds the collection cell limit.")
    units = _units(metadata.get("column_units"), columns)
    all_runs = metadata.get("cell_types")
    declared_types = metadata.get("column_types")
    if not isinstance(all_runs, list) or len(all_runs) != len(columns):
        raise MeasurementCollectionError("Missing cell type evidence.")
    if not isinstance(declared_types, list) or len(declared_types) != len(columns):
        raise MeasurementCollectionError("Missing column type evidence.")
    tags = []
    for runs, declared in zip(all_runs, declared_types, strict=True):
        _check(cancellation)
        column_tags = []
        if not isinstance(runs, list):
            raise MeasurementCollectionError("Invalid cell type runs.")
        for run in runs:
            _check(cancellation)
            if not isinstance(run, list) or len(run) != 2 or run[0] not in _TYPES:
                raise MeasurementCollectionError("Unsupported cell type evidence.")
            length = _count(run[1], "cell type run", count)
            if length < 1 or len(column_tags) + length > count:
                raise MeasurementCollectionError("Invalid cell type run length.")
            column_tags.extend([run[0]] * length)
        if len(column_tags) != count or sorted(set(column_tags)) != declared:
            raise MeasurementCollectionError("Cell/column type evidence mismatch.")
        tags.append(column_tags)
    try:
        reader = csv.reader(
            io.StringIO(data.decode("utf-8"), newline=""),
            delimiter="\t" if format == "tsv" else ",",
            strict=True,
        )
        if tuple(next(reader)) != columns:
            raise MeasurementCollectionError(
                "Saved table header differs from recorded columns."
            )
        rows = []
        for index, row in enumerate(reader):
            _check(cancellation)
            if index >= count or len(row) != len(columns):
                raise MeasurementCollectionError(
                    "Saved table dimensions differ from the recorded table."
                )
            rows.append(
                tuple(
                    _decode_cell(text, tags[col][index], limits)
                    for col, text in enumerate(row)
                )
            )
        if len(rows) != count:
            raise MeasurementCollectionError("Saved table has missing rows.")
    except (UnicodeError, csv.Error, StopIteration, OverflowError, ValueError) as exc:
        raise MeasurementCollectionError(
            f"Cannot decode the exact saved table: {exc}"
        ) from exc
    return TableData(
        columns,
        tuple(rows),
        name=str(metadata.get("name", "")),
        table_kind=str(metadata.get("table_kind", "table")),
        column_units=units,
    )


def inspect_collection(
    manifest, node_id, *, cancellation=None, progress=None, limits=_DEFAULT_LIMITS
):
    document = _manifest(manifest, limits, cancellation)
    outputs = available_measurement_outputs(document, limits=limits)
    output = next((item for item in outputs if item.node_id == node_id), None)
    if output is None:
        raise MeasurementCollectionError("No such measurement output in this batch.")
    output_root = Path(_text(document.get("output_dir"), "output folder"))
    if not output_root.is_absolute():
        raise MeasurementCollectionError("The manifest output folder must be absolute.")
    items = []
    total_bytes = total_rows = total_cells = 0
    seen_paths = set()
    for position, record in enumerate(document["items"]):
        _check(cancellation)
        sources = record.get("sources", [])
        source = (
            str(sources[0].get("path", ""))
            if sources and isinstance(sources[0], dict)
            else ""
        )
        item = CollectionItem(
            f"item-{record['index']}",
            record["index"],
            record["batch_id"],
            str(record.get("status", "missing")),
            source_path=source,
            sources=_source_evidence(sources, limits),
            parameter_overrides=_override_evidence(
                record.get("parameter_overrides", {}), limits
            ),
            effective_workflow_sha256=_optional_hash(
                record.get("effective_workflow_sha256", ""), "item workflow"
            ),
            execution_provenance_sha256=_optional_hash(
                record.get("execution_provenance_sha256", ""), "item execution"
            ),
            resumed_from_run_id=_text(
                record.get("resumed_from_run_id", ""), "resumed run", empty=True
            ),
        )
        selected = [
            entry for entry in record["outputs"] if entry.get("node_id") == node_id
        ]
        if len(selected) > 1:
            raise MeasurementCollectionError(
                "Duplicate output identity in a batch item."
            )
        if not selected:
            items.append(
                replace(
                    item,
                    status="missing",
                    message="This item has no recorded output for the selected node.",
                )
            )
            continue
        entry = selected[0]
        identity = entry.get("content_identity", {})
        item = replace(
            item,
            result_path=str(entry.get("path", "")),
            result_sha256=str(identity.get("sha256", "")),
        )
        if record.get("status") != "completed" or entry.get("status") != "completed":
            status = (
                str(record.get("status"))
                if record.get("status") != "completed"
                else str(entry.get("status", "missing"))
            )
            items.append(
                replace(
                    item,
                    status=status,
                    message="Not a successfully completed measurement output.",
                )
            )
            continue
        if entry.get("kind") != "table" or entry.get("format") not in {"csv", "tsv"}:
            items.append(
                replace(
                    item,
                    status="unsupported",
                    message="Only recorded CSV/TSV table outputs can be collected.",
                )
            )
            continue
        if entry.get("provenance_status") not in {"produced", "verified_reused"}:
            items.append(
                replace(
                    item,
                    status="unavailable",
                    message="Output was not produced or verified by this run.",
                )
            )
            continue
        metadata = entry.get("table_metadata")
        if (
            not isinstance(metadata, dict)
            or not metadata.get("available")
            or not _HASH.fullmatch(str(metadata.get("file_sha256", "")))
        ):
            items.append(
                replace(
                    item,
                    status="unavailable",
                    message=(
                        "Exact types/units and table hash are unavailable; "
                        "rerun this batch."
                    ),
                )
            )
            continue
        try:
            path = Path(item.result_path)
            if not path.is_absolute():
                path = output_root / path
            if path.suffix.casefold() != f".{entry['format']}":
                raise MeasurementCollectionError(
                    "Result extension does not match its recorded table format."
                )
            source_paths = {
                os.path.normcase(os.path.abspath(source["path"]))
                for source in item.sources
                if source["path"]
            }
            if os.path.normcase(os.path.abspath(path)) in source_paths:
                raise MeasurementCollectionError(
                    "A result path cannot point to an input source."
                )
            _ordinary_path(path)
            if not path.resolve().is_relative_to(output_root.resolve()):
                raise MeasurementCollectionError(
                    "Result path is outside its batch output folder."
                )
            path_key = os.path.normcase(str(path.resolve()))
            if path_key in seen_paths:
                raise MeasurementCollectionError(
                    "Duplicate result path; a resumed table cannot be appended twice."
                )
            seen_paths.add(path_key)
            budget = min(limits.max_file_bytes, limits.max_total_bytes - total_bytes)
            if path.stat().st_size > budget:
                raise MeasurementCollectionError("Collection exceeds its byte limit.")
            before = capture_local_source_identity(
                path,
                cancel_callback=lambda path=path, budget=budget: _hash_checkpoint(
                    path,
                    budget,
                    cancellation,
                ),
            ).to_dict()
            if before != identity:
                items.append(
                    replace(
                        item,
                        status="changed",
                        message="Result content does not match the completed run.",
                    )
                )
                continue
            data = _read_bytes(path, budget, cancellation)
            total_bytes += len(data)
            if hashlib.sha256(data).hexdigest() != metadata["file_sha256"]:
                raise _ChangedResult(
                    "Result bytes changed or disagree with their table evidence."
                )
            table = _read_table(data, metadata, entry["format"], limits, cancellation)
            after = capture_local_source_identity(
                path,
                cancel_callback=lambda path=path, budget=budget: _hash_checkpoint(
                    path,
                    budget,
                    cancellation,
                ),
            ).to_dict()
            if after != before:
                raise _ChangedResult("Result changed while reading the table.")
            total_rows += table.row_count
            total_cells += table.row_count * table.column_count
            if total_rows > limits.max_rows or total_cells > limits.max_cells:
                raise MeasurementCollectionError(
                    "Collection exceeds its row/cell limit."
                )
            item = replace(
                item,
                status="ready" if table.row_count else "empty",
                row_count=table.row_count,
                table=table,
            )
        except FileNotFoundError:
            item = replace(
                item, status="missing", message="The recorded result file is missing."
            )
        except (_ChangedResult, SourceChangedError) as exc:
            item = replace(item, status="changed", message=str(exc))
        except (OSError, ValueError) as exc:
            item = replace(item, status="unavailable", message=str(exc))
        items.append(item)
        if progress is not None:
            progress(position + 1, len(document["items"]), f"Checked {item.batch_id}")
    return MeasurementPreview(
        output,
        tuple(items),
        document["run_id"],
        document["integrity_sha256"],
        str(document.get("workflow", {}).get("sha256", "")),
        limits,
    )


def _hash_checkpoint(path, budget, cancellation):
    _check(cancellation)
    if path.stat().st_size > budget:
        raise MeasurementCollectionError("Growing result exceeds the collection limit.")
    return False


def collect_measurements(
    preview,
    *,
    annotations=None,
    included_ids=None,
    reviewed_exclusions=False,
    cancellation=None,
):
    by_key = {item.key: item for item in preview.items}
    if len(by_key) != len(preview.items):
        raise MeasurementCollectionError("Duplicate collection item identity.")
    selected = set(by_key) if included_ids is None else set(included_ids)
    if not selected <= by_key.keys():
        raise MeasurementCollectionError("Unknown included item identity.")
    if selected != by_key.keys() and reviewed_exclusions is not True:
        raise MeasurementCollectionError(
            "Review and explicitly confirm every excluded item."
        )
    included = [item for item in preview.items if item.key in selected]
    if not included:
        raise MeasurementCollectionError(
            "Select at least one verified table, including a valid empty table."
        )
    if any(
        item.table is None or item.status not in {"ready", "empty"} for item in included
    ):
        raise MeasurementCollectionError(
            "Included items contain missing, failed, changed, or unsupported results. "
            "Review exclusions."
        )
    reference = included[0].table
    columns = reference.columns
    if set(columns) & set(_IDENTITY_COLUMNS):
        raise MeasurementCollectionError(
            "Measurement columns conflict with collection identity columns."
        )
    observed_types = [set() for _ in columns]
    for item in included:
        _check(cancellation)
        table = item.table
        if (
            table.columns != columns
            or dict(table.column_units) != dict(reference.column_units)
            or table.table_kind != reference.table_kind
        ):
            raise MeasurementCollectionError(
                "Measurement columns, units, or table kinds differ; "
                "no automatic conversion is performed."
            )
        item_types = [set() for _ in columns]
        for row in table.rows:
            _check(cancellation)
            for index, value in enumerate(row):
                if value is not None:
                    item_types[index].add(_type_tag(value))
        for index, types in enumerate(item_types):
            if observed_types[index] and types and observed_types[index] != types:
                raise MeasurementCollectionError(
                    f"Column {columns[index]!r} has incompatible recorded scalar types."
                )
            observed_types[index] |= types
    annotations = {} if annotations is None else annotations
    if not isinstance(annotations, Mapping) or not annotations.keys() <= by_key.keys():
        raise MeasurementCollectionError("Annotations refer to unknown batch items.")
    annotation_columns = []
    clean_annotations = {}
    for key, values in annotations.items():
        if not isinstance(values, Mapping):
            raise MeasurementCollectionError(
                "Annotations must map column names to text."
            )
        clean_annotations[key] = {}
        for name, text in values.items():
            _text(name, "annotation column", limit=preview.limits.max_text_chars)
            _text(
                text,
                "annotation value",
                empty=True,
                limit=preview.limits.max_text_chars,
            )
            if name in columns or name in _IDENTITY_COLUMNS:
                raise MeasurementCollectionError(
                    f"Annotation {name!r} conflicts with an existing measurement "
                    "or identity column."
                )
            if name not in annotation_columns:
                annotation_columns.append(name)
            clean_annotations[key][name] = text
    rows = []
    for item in included:
        for index, row in enumerate(item.table.rows):
            _check(cancellation)
            rows.append(
                (
                    *row,
                    preview.run_id,
                    item.key,
                    item.batch_id,
                    index + 1,
                    *(
                        clean_annotations.get(item.key, {}).get(name, "")
                        for name in annotation_columns
                    ),
                )
            )
    if (
        len(rows) * (len(columns) + 4 + len(annotation_columns))
        > preview.limits.max_cells
    ):
        raise MeasurementCollectionError("Annotated collection exceeds its cell limit.")
    table = TableData(
        (*columns, *_IDENTITY_COLUMNS, *annotation_columns),
        tuple(rows),
        name=preview.output.title,
        table_kind="batch measurement collection",
        column_units=reference.column_units,
    )
    inventory = tuple(
        replace(item, table=None, included=item.key in selected)
        for item in preview.items
    )
    return MeasurementCollection(
        table,
        inventory,
        clean_annotations,
        {
            "run_id": preview.run_id,
            "manifest_sha256": preview.manifest_sha256,
            "workflow_sha256": preview.workflow_sha256,
            "output_node_id": preview.output.node_id,
            "reviewed_exclusions": bool(reviewed_exclusions),
            "measurement_columns": list(columns),
            "annotation_columns": annotation_columns,
        },
    )


def _snapshot_cell(value):
    tag = _type_tag(value)
    if tag == "float":
        value = value.hex()  # JSON stays finite; NaN/Inf remain explicit typed values.
    return [tag, value]


def _restore_cell(cell, limits):
    if not isinstance(cell, list) or len(cell) != 2 or cell[0] not in _TYPES:
        raise MeasurementCollectionError("Invalid typed snapshot cell.")
    tag, value = cell
    if tag == "float":
        _text(value, "encoded float", limit=128)
        try:
            result = float.fromhex(value)
        except ValueError as exc:
            raise MeasurementCollectionError("Invalid encoded snapshot float.") from exc
        if result.hex() != value:
            raise MeasurementCollectionError("Noncanonical encoded snapshot float.")
        return result
    if _type_tag(value) != tag:
        raise MeasurementCollectionError("Snapshot cell scalar type mismatch.")
    if isinstance(value, str):
        _text(value, "snapshot text", empty=True, limit=limits.max_text_chars)
    return value


def _snapshot_document(collection, cancellation=None):
    rows = []
    for row in collection.table.rows:
        _check(cancellation)
        rows.append([_snapshot_cell(value) for value in row])
    return seal_document(
        {
            "type": _SCHEMA,
            "version": 1,
            "table": {
                "columns": list(collection.table.columns),
                "rows": rows,
                "column_units": [list(pair) for pair in collection.table.column_units],
                "name": collection.table.name,
                "table_kind": collection.table.table_kind,
            },
            "items": [
                {
                    name: _thaw(value)
                    for name, value in vars(item).items()
                    if name != "table"
                }
                for item in collection.items
            ],
            "annotations": _thaw(collection.annotations),
            "provenance": _thaw(collection.provenance),
        }
    )


def save_measurement_collection(
    collection, path, *, cancellation=None, limits=_DEFAULT_LIMITS,
    expected_destination_revision=_UNSPECIFIED_DESTINATION_REVISION,
):
    """Save a snapshot, optionally bound to the destination reviewed by the UI.

    Pass the capture helper's ``None`` to create only if absent, or its tuple
    to replace only a still-matching file. Omitting the keyword retains the
    existing core API's unconditional atomic-replacement behavior.
    """
    path = Path(path)
    guarded = expected_destination_revision is not _UNSPECIFIED_DESTINATION_REVISION
    if guarded and expected_destination_revision is not None and (
        type(expected_destination_revision) is not tuple
        or len(expected_destination_revision) != 4
        or any(type(value) is not int for value in expected_destination_revision)
    ):
        raise MeasurementCollectionError("Invalid expected destination revision.")
    if not path.name.endswith(_SUFFIX):
        raise MeasurementCollectionError(
            f"Save collections with the {_SUFFIX} extension."
        )
    _ordinary_path(path, may_be_missing=True)
    _check(cancellation)
    document = _snapshot_document(collection, cancellation)
    encoded = (json.dumps(document, ensure_ascii=False, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    if len(encoded) > limits.max_total_bytes:
        raise MeasurementCollectionError("Snapshot exceeds its size limit.")
    _validated_snapshot(document, limits, cancellation)
    _check(cancellation)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            for offset in range(0, len(encoded), 1024**2):
                _check(cancellation)
                stream.write(encoded[offset : offset + 1024**2])
            stream.flush()
            os.fsync(stream.fileno())
        _ordinary_path(path, may_be_missing=True)
        _check(cancellation)
        # This atomic replacement is the commit point. Earlier cancellation
        # leaves existing snapshots untouched and removes private staged bytes.
        if guarded:
            actual = capture_measurement_collection_destination_revision(path)
            if actual != expected_destination_revision:
                raise MeasurementCollectionError(
                    f"Collection destination changed since it was reviewed: {path}. "
                    "Choose the save file again before replacing it."
                )
            _check(cancellation)
            if actual is None:
                try:
                    # Atomic create-if-absent also protects the final check/link gap.
                    os.link(temporary, path)
                except FileExistsError as exc:
                    raise MeasurementCollectionError(
                        f"Collection destination appeared while saving: {path}. "
                        "Choose the save file again before replacing it."
                    ) from exc
            else:
                # A single atomic attempt: retrying after a Windows lock would
                # otherwise outlive the destination revision just checked.
                os.replace(temporary, path)
        else:
            atomic_replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def load_measurement_collection(
    path, *, expected_sha256="", cancellation=None, limits=_DEFAULT_LIMITS
):
    data = _read_bytes(path, limits.max_total_bytes, cancellation)
    digest = hashlib.sha256(data).hexdigest()
    if expected_sha256 and (
        not isinstance(expected_sha256, str)
        or not _HASH.fullmatch(expected_sha256)
        or digest != expected_sha256
    ):
        raise MeasurementCollectionError(
            "Measurement collection file hash does not match the selected snapshot."
        )
    document = _json_document(data)
    return replace(
        _validated_snapshot(document, limits, cancellation), file_sha256=digest
    )


def _validated_snapshot(document, limits, cancellation):
    try:
        return _load_snapshot(document, limits, cancellation)
    except (KeyError, TypeError, IndexError, OverflowError, RecursionError) as exc:
        raise MeasurementCollectionError(
            f"Malformed collection snapshot: {exc}"
        ) from exc


def _load_snapshot(document, limits, cancellation=None):
    _verified_document(document)
    if (
        set(document)
        != {
            "type",
            "version",
            "table",
            "items",
            "annotations",
            "provenance",
            "integrity_sha256",
        }
        or document.get("type") != _SCHEMA
        or document.get("version") != 1
    ):
        raise MeasurementCollectionError("Unsupported collection snapshot schema.")
    raw = document["table"]
    if not isinstance(raw, dict) or set(raw) != {
        "columns",
        "rows",
        "column_units",
        "name",
        "table_kind",
    }:
        raise MeasurementCollectionError("Invalid snapshot table schema.")
    columns = _columns(raw["columns"], limits)
    units = _units(raw["column_units"], columns)
    rows = raw["rows"]
    if (
        not isinstance(rows, list)
        or len(rows) > limits.max_rows
        or len(rows) * len(columns) > limits.max_cells
    ):
        raise MeasurementCollectionError("Snapshot exceeds table limits.")
    decoded = []
    for row in rows:
        _check(cancellation)
        if not isinstance(row, list) or len(row) != len(columns):
            raise MeasurementCollectionError(
                "Snapshot row width differs from its columns."
            )
        decoded.append(tuple(_restore_cell(cell, limits) for cell in row))
    items = document["items"]
    if not isinstance(items, list) or len(items) > limits.max_items:
        raise MeasurementCollectionError("Invalid snapshot item inventory.")
    inventory = []
    for item in items:
        if not isinstance(item, dict) or set(item) != set(
            CollectionItem.__dataclass_fields__
        ) - {"table"}:
            raise MeasurementCollectionError("Invalid snapshot inventory record.")
        for name in (
            "key",
            "batch_id",
            "status",
            "message",
            "source_path",
            "result_path",
            "result_sha256",
        ):
            _text(
                item[name],
                name,
                empty=name not in {"key", "batch_id", "status"},
                limit=limits.max_text_chars,
            )
        _count(item["index"], "item index", limits.max_items)
        if item["row_count"] is not None:
            _count(item["row_count"], "item row count", limits.max_rows)
        if type(item["included"]) is not bool:
            raise MeasurementCollectionError("Invalid included flag.")
        if item["included"] and (
            item["status"] not in {"ready", "empty"} or item["row_count"] is None
        ):
            raise MeasurementCollectionError("Invalid included result inventory.")
        _optional_hash(item["result_sha256"], "result content")
        if item["included"] and not item["result_sha256"]:
            raise MeasurementCollectionError("Missing included result hash.")
        if (item["status"] == "empty" and item["row_count"] != 0) or (
            item["status"] == "ready" and not item["row_count"]
        ):
            raise MeasurementCollectionError(
                "Empty/nonempty result status is inconsistent."
            )
        item = dict(item)
        item["sources"] = _source_evidence(item["sources"], limits, snapshot=True)
        if item["sources"] and item["source_path"] != item["sources"][0]["path"]:
            raise MeasurementCollectionError(
                "Source summary differs from source evidence."
            )
        item["parameter_overrides"] = _override_evidence(
            item["parameter_overrides"], limits
        )
        for name in ("effective_workflow_sha256", "execution_provenance_sha256"):
            _optional_hash(item[name], name)
        _text(item["resumed_from_run_id"], "resumed run", empty=True)
        inventory.append(CollectionItem(**item))
    by_key = {item.key: item for item in inventory}
    if (
        len(by_key) != len(inventory)
        or len({item.index for item in inventory}) != len(inventory)
        or len({item.batch_id for item in inventory}) != len(inventory)
    ):
        raise MeasurementCollectionError("Duplicate snapshot item identities.")
    provenance = document["provenance"]
    annotations = document["annotations"]
    expected_provenance = {
        "run_id",
        "manifest_sha256",
        "workflow_sha256",
        "output_node_id",
        "reviewed_exclusions",
        "measurement_columns",
        "annotation_columns",
    }
    if not isinstance(provenance, dict) or set(provenance) != expected_provenance:
        raise MeasurementCollectionError("Invalid snapshot provenance.")
    _text(provenance["run_id"], "original run identity")
    _text(provenance["output_node_id"], "original output node identity")
    if type(provenance["reviewed_exclusions"]) is not bool:
        raise MeasurementCollectionError("Invalid exclusion review flag.")
    for name in ("manifest_sha256", "workflow_sha256"):
        if not isinstance(provenance[name], str) or not _HASH.fullmatch(
            provenance[name]
        ):
            raise MeasurementCollectionError("Missing original provenance hash.")
    measurement_columns = _columns(provenance["measurement_columns"], limits)
    annotation_columns = provenance["annotation_columns"]
    if not isinstance(annotation_columns, list) or columns != (
        *measurement_columns,
        *_IDENTITY_COLUMNS,
        *annotation_columns,
    ):
        raise MeasurementCollectionError(
            "Snapshot identity/annotation columns do not match provenance."
        )
    if not isinstance(annotations, dict) or not annotations.keys() <= by_key.keys():
        raise MeasurementCollectionError("Unknown snapshot annotation item.")
    for values in annotations.values():
        if not isinstance(values, dict) or not values.keys() <= set(annotation_columns):
            raise MeasurementCollectionError("Unknown snapshot annotation column.")
        for text in values.values():
            _text(text, "annotation", empty=True, limit=limits.max_text_chars)
    offset = len(measurement_columns)
    observed = {item.key: 0 for item in inventory}
    for row in decoded:
        _check(cancellation)
        run_id, key, batch_id, row_index = row[offset : offset + 4]
        item = by_key.get(key)
        if (
            item is None
            or not item.included
            or run_id != provenance["run_id"]
            or batch_id != item.batch_id
            or type(row_index) is not int
            or row_index != observed[key] + 1
        ):
            raise MeasurementCollectionError(
                "Snapshot row identity is duplicated or inconsistent."
            )
        observed[key] += 1
        if row[offset + 4 :] != tuple(
            annotations.get(key, {}).get(name, "") for name in annotation_columns
        ):
            raise MeasurementCollectionError(
                "Snapshot annotations conflict with table rows."
            )
    if any(
        observed[item.key] != (item.row_count if item.included else 0)
        for item in inventory
    ):
        raise MeasurementCollectionError(
            "Snapshot rows do not match the complete item inventory."
        )
    if (
        any(not item.included for item in inventory)
        and provenance["reviewed_exclusions"] is not True
    ):
        raise MeasurementCollectionError(
            "Snapshot exclusions were not explicitly reviewed."
        )
    table = TableData(
        columns,
        tuple(decoded),
        name=_text(raw["name"], "table name", empty=True),
        table_kind=_text(raw["table_kind"], "table kind"),
        column_units=units,
    )
    return MeasurementCollection(table, tuple(inventory), annotations, provenance)
