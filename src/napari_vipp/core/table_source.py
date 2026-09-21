"""Verified, Qt-free source boundary for collected measurement datasets."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from weakref import WeakValueDictionary

from napari_vipp.core.tables import TableData, TableState, table_state_from_data

_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
_ISSUED_PAYLOADS: WeakValueDictionary = WeakValueDictionary()


@dataclass(frozen=True)
class TableSourceRevision:
    """Exact dataset revision; contains no measurements or runtime objects."""

    path: str
    file_sha256: str
    size: int
    mtime_ns: int
    device: int
    inode: int


def validate_table_source_reference(params) -> dict[str, str]:
    """Validate a reference only, without inspecting or reading the dataset."""
    allowed = {"dataset_path", "dataset_sha256"}
    ui_flags = {"_vipp_keep_cached"}
    if not allowed <= set(params) or set(params) - allowed - ui_flags:
        raise ValueError(
            "Table Source stores only dataset_path and dataset_sha256; "
            "apart from supported cache preferences, embedded measurements "
            "or extra metadata are not allowed."
        )
    for name in ui_flags & set(params):
        if type(params[name]) is not bool:
            raise ValueError(f"Table Source {name} preference must be Boolean.")
    path = params["dataset_path"]
    expected = params["dataset_sha256"]
    if not isinstance(path, str) or not isinstance(expected, str):
        raise ValueError("Table Source path and expected hash must be text.")
    if expected and not _SHA256.fullmatch(expected):
        raise ValueError(
            "Table Source expected SHA-256 must be 64 hexadecimal characters."
        )
    return {"dataset_path": path, "dataset_sha256": expected}


def _stat_key(path: Path) -> tuple[int, int, int, int]:
    info = path.stat()
    return info.st_size, info.st_mtime_ns, info.st_dev, info.st_ino


def load_table_source(path, expected_sha256="", *, cancellation=None):
    """Read a stable typed snapshot; blank hash is for explicit initial adoption.

    Executing a saved source requires an expected hash. The UI can use this
    loader with no hash when the user first chooses a dataset, then save the
    returned revision's hash in the node before execution.
    """
    from napari_vipp.core.measurement_collection import load_measurement_collection
    from napari_vipp.core.pipeline import SourcePayload

    raw = str(path).strip()
    if not raw:
        raise ValueError("Choose a .vipp-results.json dataset for Table Source.")
    validate_table_source_reference(
        {"dataset_path": raw, "dataset_sha256": expected_sha256}
    )
    source = Path(raw).expanduser().absolute()
    try:
        before = _stat_key(source)
        collection = load_measurement_collection(
            source, expected_sha256=expected_sha256.lower(), cancellation=cancellation
        )
        after = _stat_key(source)
    except FileNotFoundError as exc:
        raise ValueError(
            "The measurement dataset is missing. Locate the original "
            ".vipp-results.json file and reconnect Table Source."
        ) from exc
    if before != after:
        raise ValueError("The measurement dataset changed while it was being loaded.")
    table = collection.table
    if not isinstance(table, TableData):
        raise ValueError("The measurement dataset does not contain a typed table.")
    digest = collection.file_sha256
    state = table_state_from_data(
        table,
        metadata_source="VIPP measurement collection",
        source_name=table.source_name or source.name,
        history=(f"Loaded measurement dataset; SHA-256 {digest}",),
    )
    revision = TableSourceRevision(str(source), digest, *after)
    payload = SourcePayload(
        table,
        name=table.name or source.stem,
        image_state=state,
        revision_token=revision,
    )
    _ISSUED_PAYLOADS[id(payload)] = payload
    return payload


def resolve_table_source(
    params, payload=None, *, metadata_only=False, cancellation=None,
):
    """Reuse a loader-issued preview; never trust caller data for execution."""
    reference = validate_table_source_reference(params)
    path = reference["dataset_path"]
    expected = reference["dataset_sha256"].lower()
    if metadata_only:
        if payload is None or payload.data is None:
            return [(None, None)]
        revision = payload.revision_token
        if (
            _ISSUED_PAYLOADS.get(id(payload)) is not payload
            or not isinstance(revision, TableSourceRevision)
            or not isinstance(payload.data, TableData)
            or not isinstance(payload.image_state, TableState)
            or not path
            or not expected
            or revision.path != str(Path(path).expanduser().absolute())
            or revision.file_sha256 != expected
        ):
            raise ValueError("Reload Table Source to verify the selected dataset.")
        try:
            current = _stat_key(Path(revision.path))
        except OSError as exc:
            raise ValueError(
                "The measurement dataset is unavailable; reconnect it."
            ) from exc
        if current != (
            revision.size, revision.mtime_ns, revision.device, revision.inode
        ):
            raise ValueError("The measurement dataset changed; reload it for review.")
        return [(payload.data, payload.image_state)]
    if not expected:
        raise ValueError(
            "Table Source has no recorded dataset hash. Choose the dataset "
            "explicitly before calculating this workflow."
        )
    verified = load_table_source(path, expected, cancellation=cancellation)
    return [(verified.data, verified.image_state)]
