"""Deterministic, headless collection-batch planning and execution.

The batch configuration is intentionally independent from the workflow schema.
The workflow defines the scientific graph and its ``Batch Output`` nodes; the
configuration binds local collections to source nodes and freezes the resolved
save policy for one reproducible run.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import sys
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

from napari_vipp import __version__ as VIPP_VERSION
from napari_vipp.core.atomic_io import (
    atomic_replace as _replace_with_retry,
)
from napari_vipp.core.atomic_io import (
    atomic_write_json as _atomic_write_json,
)
from napari_vipp.core.atomic_io import (
    atomic_write_text,
)
from napari_vipp.core.io import read_image
from napari_vipp.core.operations import save_array_output
from napari_vipp.core.pipeline import PrototypePipeline, SourcePayload
from napari_vipp.core.source_identity import (
    LocalSourceIdentity,
    SourceChangedError,
    capture_local_source_identity,
    verify_local_source_identity,
)
from napari_vipp.core.tables import is_table_data, save_table_output
from napari_vipp.core.workflow import deserialize_workflow

BATCH_CONFIG_TYPE = "napari-vipp-batch-config"
BATCH_CONFIG_VERSION = 1
BATCH_MANIFEST_TYPE = "napari-vipp-batch-manifest"
BATCH_MANIFEST_VERSION = 1

BATCH_CONFIG_FILENAME = "vipp_batch_config.json"
BATCH_MANIFEST_FILENAME = "vipp_batch_manifest.json"
BATCH_WORKFLOW_FILENAME = "vipp_batch_workflow.json"
BATCH_SCRIPT_FILENAME = "vipp_batch_pipeline.py"

PAIRING_POLICY = "sorted-position"
_PATTERN_SEPARATORS = re.compile(r"[;,\n]+")
_KNOWN_SUFFIXES = (
    ".ome.tif",
    ".ome.tiff",
    ".tif",
    ".tiff",
    ".npy",
    ".csv",
    ".tsv",
)
_IMAGE_SUFFIXES = {
    "ome-tiff": ".ome.tif",
    "imagej-tiff": ".tif",
    "tiff": ".tif",
    "npy": ".npy",
}
_IMAGE_FORMATS = frozenset(_IMAGE_SUFFIXES)
_TABLE_FORMATS = frozenset(("csv", "tsv"))
_OUTPUT_FORMATS = frozenset(("batch default", *_IMAGE_FORMATS, *_TABLE_FORMATS))
_OVERWRITE_VALUES = frozenset(("batch default", "yes", "no"))
_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
_WINDOWS_RESERVED_STEMS = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }
)


class ExistingFilePolicy(StrEnum):
    """Action to take when a planned output already exists."""

    ERROR = "error"
    SKIP = "skip"
    OVERWRITE = "overwrite"


class BatchStatus(StrEnum):
    """Stable item/output status values written to manifests."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(frozen=True)
class BatchSourceConfig:
    """One workflow source bound to a local file collection."""

    node_id: str
    title: str
    input_dir: Path
    pattern: str

    def __post_init__(self) -> None:
        _require_text(self.node_id, "Batch source node_id")
        _require_text(self.title, "Batch source title")
        _require_text(str(self.input_dir), "Batch source input_dir")
        _require_text(self.pattern, "Batch source pattern")

    def to_dict(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "title": self.title,
            "input_dir": _config_path_text(self.input_dir),
            "pattern": self.pattern,
        }

    @classmethod
    def from_dict(cls, value: object, *, index: int) -> BatchSourceConfig:
        data = _require_object(value, f"Batch source {index}")
        _reject_unknown_keys(
            data,
            {"node_id", "title", "input_dir", "pattern"},
            f"Batch source {index}",
        )
        return cls(
            node_id=_required_text(data, "node_id", f"batch source {index}"),
            title=_required_text(data, "title", f"batch source {index}"),
            input_dir=Path(_required_text(data, "input_dir", f"batch source {index}")),
            pattern=_required_text(data, "pattern", f"batch source {index}"),
        )


@dataclass(frozen=True)
class BatchOutputConfig:
    """Resolved save declaration for one selected workflow output."""

    node_id: str
    node_title: str
    tag: str
    kind: str
    format: str
    subfolder: str
    filename_template: str
    overwrite: str = "batch default"

    def __post_init__(self) -> None:
        _require_text(self.node_id, "Batch output node_id")
        _require_text(self.node_title, "Batch output node_title")
        _require_text(self.tag, "Batch output tag")
        if self.kind not in {"image", "table"}:
            raise ValueError("Batch output kind must be 'image' or 'table'.")
        if self.format not in _OUTPUT_FORMATS:
            raise ValueError(f"Unsupported batch output format: {self.format!r}.")
        if self.kind == "table" and self.format in _IMAGE_FORMATS:
            raise ValueError("A table batch output cannot use an image format.")
        if self.kind == "image" and self.format in _TABLE_FORMATS:
            raise ValueError("An image batch output cannot use a table format.")
        _require_text(self.filename_template, "Batch output filename_template")
        if self.overwrite not in _OVERWRITE_VALUES:
            raise ValueError(
                "Batch output overwrite must be 'batch default', 'yes', or 'no'."
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "node_title": self.node_title,
            "tag": self.tag,
            "kind": self.kind,
            "format": self.format,
            "subfolder": self.subfolder,
            "filename_template": self.filename_template,
            "overwrite": self.overwrite,
        }

    @classmethod
    def from_dict(cls, value: object, *, index: int) -> BatchOutputConfig:
        data = _require_object(value, f"Batch output {index}")
        allowed = {
            "node_id",
            "node_title",
            "tag",
            "kind",
            "format",
            "subfolder",
            "filename_template",
            "overwrite",
        }
        _reject_unknown_keys(data, allowed, f"Batch output {index}")
        return cls(
            node_id=_required_text(data, "node_id", f"batch output {index}"),
            node_title=_required_text(data, "node_title", f"batch output {index}"),
            tag=_required_text(data, "tag", f"batch output {index}"),
            kind=_required_text(data, "kind", f"batch output {index}"),
            format=_required_text(data, "format", f"batch output {index}"),
            subfolder=_optional_text(data, "subfolder", f"batch output {index}"),
            filename_template=_required_text(
                data, "filename_template", f"batch output {index}"
            ),
            overwrite=_optional_text(
                data,
                "overwrite",
                f"batch output {index}",
                default="batch default",
            ),
        )


@dataclass(frozen=True)
class BatchConfig:
    """Versioned configuration for a reproducible local collection run."""

    workflow_file: Path
    workflow_sha256: str
    output_dir: Path
    sources: tuple[BatchSourceConfig, ...]
    outputs: tuple[BatchOutputConfig, ...]
    default_image_format: str = "ome-tiff"
    existing_file_policy: ExistingFilePolicy = ExistingFilePolicy.ERROR
    save_workflow_snapshot: bool = True
    save_python_script: bool = True
    continue_on_error: bool = True
    pairing_policy: str = PAIRING_POLICY
    base_dir: Path | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        _require_text(str(self.workflow_file), "Batch config workflow_file")
        if not _HASH_PATTERN.fullmatch(self.workflow_sha256):
            raise ValueError("Batch config workflow_sha256 must be lowercase SHA-256.")
        _require_text(str(self.output_dir), "Batch config output_dir")
        if not self.sources:
            raise ValueError("Batch config needs at least one source binding.")
        if not self.outputs:
            raise ValueError("Batch config needs at least one selected output.")
        _reject_duplicate_ids(
            (source.node_id for source in self.sources), "batch source"
        )
        _reject_duplicate_ids(
            (output.node_id for output in self.outputs), "batch output"
        )
        if self.default_image_format not in _IMAGE_FORMATS:
            raise ValueError(
                f"Unsupported default image format: {self.default_image_format!r}."
            )
        if not isinstance(self.existing_file_policy, ExistingFilePolicy):
            raise ValueError("Batch config existing_file_policy is invalid.")
        if self.pairing_policy != PAIRING_POLICY:
            raise ValueError(
                f"Unsupported batch pairing policy: {self.pairing_policy!r}."
            )
        for name in (
            "save_workflow_snapshot",
            "save_python_script",
            "continue_on_error",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"Batch config {name} must be a boolean.")
        if self.save_python_script and not self.save_workflow_snapshot:
            raise ValueError(
                "A saved batch runner requires save_workflow_snapshot to be true."
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "type": BATCH_CONFIG_TYPE,
            "version": BATCH_CONFIG_VERSION,
            "workflow": {
                "file": _config_path_text(self.workflow_file),
                "sha256": self.workflow_sha256,
            },
            "output_dir": _config_path_text(self.output_dir),
            "pairing_policy": self.pairing_policy,
            "sources": [source.to_dict() for source in self.sources],
            "outputs": [output.to_dict() for output in self.outputs],
            "defaults": {
                "image_format": self.default_image_format,
                "existing_file_policy": self.existing_file_policy.value,
            },
            "artifacts": {
                "save_workflow_snapshot": self.save_workflow_snapshot,
                "save_python_script": self.save_python_script,
            },
            "continue_on_error": self.continue_on_error,
        }

    @classmethod
    def from_dict(
        cls, value: object, *, base_dir: str | Path | None = None
    ) -> BatchConfig:
        data = _require_object(value, "Batch config")
        allowed = {
            "type",
            "version",
            "workflow",
            "output_dir",
            "pairing_policy",
            "sources",
            "outputs",
            "defaults",
            "artifacts",
            "continue_on_error",
        }
        _reject_unknown_keys(data, allowed, "Batch config")
        if data.get("type") != BATCH_CONFIG_TYPE:
            raise ValueError("File is not a napari-vipp batch config.")
        raw_version = data.get("version")
        if type(raw_version) is not int or raw_version != BATCH_CONFIG_VERSION:
            raise ValueError(
                f"Unsupported batch config version: {raw_version!r}. "
                f"Expected version {BATCH_CONFIG_VERSION}."
            )
        workflow = _require_object(data.get("workflow"), "Batch config workflow")
        _reject_unknown_keys(workflow, {"file", "sha256"}, "Batch config workflow")
        defaults = _require_object(data.get("defaults"), "Batch config defaults")
        _reject_unknown_keys(
            defaults,
            {"image_format", "existing_file_policy"},
            "Batch config defaults",
        )
        artifacts = _require_object(data.get("artifacts"), "Batch config artifacts")
        _reject_unknown_keys(
            artifacts,
            {"save_workflow_snapshot", "save_python_script"},
            "Batch config artifacts",
        )
        raw_sources = data.get("sources")
        raw_outputs = data.get("outputs")
        if not isinstance(raw_sources, list):
            raise ValueError("Batch config sources must be a list.")
        if not isinstance(raw_outputs, list):
            raise ValueError("Batch config outputs must be a list.")
        policy_text = _required_text(
            defaults, "existing_file_policy", "batch config defaults"
        )
        try:
            policy = ExistingFilePolicy(policy_text)
        except ValueError as exc:
            raise ValueError(
                f"Unsupported existing-file policy: {policy_text!r}."
            ) from exc
        return cls(
            workflow_file=Path(
                _required_text(workflow, "file", "batch config workflow")
            ),
            workflow_sha256=_required_text(workflow, "sha256", "batch config workflow"),
            output_dir=Path(_required_text(data, "output_dir", "batch config")),
            sources=tuple(
                BatchSourceConfig.from_dict(item, index=index)
                for index, item in enumerate(raw_sources)
            ),
            outputs=tuple(
                BatchOutputConfig.from_dict(item, index=index)
                for index, item in enumerate(raw_outputs)
            ),
            default_image_format=_required_text(
                defaults, "image_format", "batch config defaults"
            ),
            existing_file_policy=policy,
            save_workflow_snapshot=_required_bool(
                artifacts, "save_workflow_snapshot", "batch config artifacts"
            ),
            save_python_script=_required_bool(
                artifacts, "save_python_script", "batch config artifacts"
            ),
            continue_on_error=_required_bool(data, "continue_on_error", "batch config"),
            pairing_policy=_required_text(data, "pairing_policy", "batch config"),
            base_dir=(
                Path(base_dir).expanduser().resolve() if base_dir is not None else None
            ),
        )

    def resolve_path(self, value: Path) -> Path:
        value = value.expanduser()
        if value.is_absolute() or self.base_dir is None:
            return value
        return (self.base_dir / value).resolve()


@dataclass(frozen=True)
class BatchOutputPlan:
    node_id: str
    node_title: str
    tag: str
    kind: str
    format: str
    path: Path
    existing_file_policy: ExistingFilePolicy
    exists: bool = False
    duplicate: bool = False
    input_collision: bool = False

    @property
    def status_text(self) -> str:
        if self.duplicate:
            return "duplicate planned destination"
        if self.input_collision:
            return "destination overlaps an input"
        if not self.exists:
            return "new"
        if self.existing_file_policy == ExistingFilePolicy.OVERWRITE:
            return "exists; will overwrite"
        if self.existing_file_policy == ExistingFilePolicy.SKIP:
            return "exists; will skip"
        return "exists; collision"


@dataclass(frozen=True)
class _StagedBatchOutput:
    plan: BatchOutputPlan
    temporary_path: Path
    saved_temporary_path: Path


@dataclass(frozen=True)
class BatchItemPlan:
    index: int
    batch_id: str
    primary_source: Path
    source_paths: dict[str, Path]
    outputs: tuple[BatchOutputPlan, ...]


@dataclass(frozen=True)
class BatchPlan:
    config: BatchConfig
    items: tuple[BatchItemPlan, ...]
    output_dir: Path

    @property
    def output_count(self) -> int:
        return sum(len(item.outputs) for item in self.items)

    @property
    def has_collisions(self) -> bool:
        return any(
            output.duplicate
            or output.input_collision
            or (
                output.exists
                and output.existing_file_policy == ExistingFilePolicy.ERROR
            )
            for item in self.items
            for output in item.outputs
        )


@dataclass(frozen=True)
class BatchOutputRecord:
    node_id: str
    node_title: str
    tag: str
    kind: str
    format: str
    path: str
    existing_file_policy: ExistingFilePolicy
    existed_at_preflight: bool
    status: BatchStatus = BatchStatus.PENDING
    size_bytes: int | None = None
    overwrote_existing: bool = False
    existing_identity: dict[str, int] = field(default_factory=dict)
    error_type: str = ""
    error_message: str = ""

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "node_id": self.node_id,
            "node_title": self.node_title,
            "tag": self.tag,
            "kind": self.kind,
            "format": self.format,
            "path": self.path,
            "existing_file_policy": self.existing_file_policy.value,
            "existed_at_preflight": self.existed_at_preflight,
            "overwrote_existing": self.overwrote_existing,
            "status": self.status.value,
        }
        if self.existing_identity:
            result["existing_identity"] = dict(self.existing_identity)
        if self.size_bytes is not None:
            result["size_bytes"] = self.size_bytes
        if self.error_type:
            result["error"] = {
                "type": self.error_type,
                "message": self.error_message,
            }
        elif self.error_message:
            result["message"] = self.error_message
        return result


@dataclass(frozen=True)
class BatchItemRecord:
    index: int
    batch_id: str
    sources: tuple[dict[str, object], ...]
    outputs: tuple[BatchOutputRecord, ...]
    status: BatchStatus = BatchStatus.PENDING
    started_at: str = ""
    finished_at: str = ""
    error_type: str = ""
    error_message: str = ""

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "index": self.index,
            "batch_id": self.batch_id,
            "status": self.status.value,
            "sources": [_json_safe(source) for source in self.sources],
            "outputs": [output.to_dict() for output in self.outputs],
        }
        if self.started_at:
            result["started_at"] = self.started_at
        if self.finished_at:
            result["finished_at"] = self.finished_at
        if self.error_type:
            result["error"] = {
                "type": self.error_type,
                "message": self.error_message,
            }
        elif self.error_message:
            result["message"] = self.error_message
        return result


@dataclass(frozen=True)
class BatchManifest:
    run_id: str
    started_at: str
    workflow_sha256: str
    config_sha256: str
    workflow_file: str
    config_file: str
    output_dir: str
    runtime: dict[str, object]
    workflow_document: dict[str, object]
    config_document: dict[str, object]
    items: tuple[BatchItemRecord, ...]
    item_records_dir: str = ""
    finished_at: str = ""

    @property
    def summary(self) -> dict[str, int]:
        return {
            status.value: sum(item.status == status for item in self.items)
            for status in (
                BatchStatus.COMPLETED,
                BatchStatus.PARTIAL,
                BatchStatus.SKIPPED,
                BatchStatus.FAILED,
            )
        }

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "type": BATCH_MANIFEST_TYPE,
            "version": BATCH_MANIFEST_VERSION,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "workflow": {
                "file": self.workflow_file,
                "sha256": self.workflow_sha256,
                "scientific_graph": _json_safe(self.workflow_document),
            },
            "config": {
                "file": self.config_file,
                "sha256": self.config_sha256,
                "document": _json_safe(self.config_document),
            },
            "output_dir": self.output_dir,
            "runtime": _json_safe(self.runtime),
            "summary": self.summary,
            "items": [item.to_dict() for item in self.items],
        }
        if self.item_records_dir:
            result["item_records_dir"] = self.item_records_dir
        if self.finished_at:
            result["finished_at"] = self.finished_at
        return result

    def replace_item(self, item: BatchItemRecord) -> BatchManifest:
        items = list(self.items)
        if not 1 <= item.index <= len(items):
            raise ValueError(f"Manifest item index is out of range: {item.index}.")
        items[item.index - 1] = item
        return replace(self, items=tuple(items))


@dataclass(frozen=True)
class BatchRunResult:
    """Structured batch outcome; iteration yields completed output paths."""

    manifest: BatchManifest
    manifest_path: Path
    saved_paths: tuple[Path, ...]
    manifest_archive_path: Path | None = None
    artifact_paths: tuple[Path, ...] = ()

    @property
    def summary(self) -> dict[str, int]:
        return self.manifest.summary

    @property
    def has_failures(self) -> bool:
        return any(
            bool(item.error_type)
            or item.status == BatchStatus.FAILED
            or any(output.status == BatchStatus.FAILED for output in item.outputs)
            for item in self.manifest.items
        )

    @property
    def all_paths(self) -> tuple[Path, ...]:
        return (*self.artifact_paths, *self.saved_paths)

    def __iter__(self) -> Iterator[Path]:
        return iter(self.all_paths)

    def __len__(self) -> int:
        return len(self.all_paths)


def scientific_workflow_document(workflow: object) -> dict[str, object]:
    """Return the canonical scientific portion of a workflow document."""
    data = _require_object(workflow, "Workflow")
    # Full deserialization validates operation ids, params, ports, and references.
    deserialize_workflow(data)
    nodes = sorted(
        (_canonical_mapping(item) for item in data["nodes"]),
        key=lambda item: str(item.get("id", "")),
    )
    connections = sorted(
        (_canonical_mapping(item) for item in data["connections"]),
        key=lambda item: (
            str(item.get("source", "")),
            int(item.get("source_port", 0)),
            str(item.get("target", "")),
            int(item.get("target_port", 0)),
            str(item.get("tunnel", "")),
        ),
    )
    tunnels = sorted(
        (_canonical_mapping(item) for item in data.get("tunnels", [])),
        key=lambda item: (
            str(item.get("name", "")),
            str(item.get("source", "")),
            int(item.get("source_port", 0)),
        ),
    )
    return {
        "type": data.get("type"),
        "version": data.get("version"),
        "nodes": nodes,
        "connections": connections,
        "tunnels": tunnels,
    }


def scientific_workflow_hash(workflow: object) -> str:
    """Return a stable SHA-256 excluding layout, notes, and UI metadata."""
    return _document_hash(scientific_workflow_document(workflow))


def batch_config_hash(config: BatchConfig) -> str:
    return _document_hash(config.to_dict())


def load_batch_config(path: str | Path) -> BatchConfig:
    raw = str(path).strip()
    if not raw:
        raise ValueError("Batch config path cannot be blank.")
    source = Path(raw).expanduser().resolve()
    data = json.loads(source.read_text(encoding="utf-8"))
    return BatchConfig.from_dict(data, base_dir=source.parent)


def save_batch_config(path: str | Path, config: BatchConfig) -> Path:
    target = Path(str(path).strip()).expanduser()
    document = config.to_dict()
    if (
        config.base_dir is not None
        and target.parent.resolve() != config.base_dir.resolve()
    ):
        document["workflow"]["file"] = str(
            config.resolve_path(config.workflow_file)
        )
        document["output_dir"] = str(config.resolve_path(config.output_dir))
        for source_document, source in zip(
            document["sources"], config.sources, strict=True
        ):
            source_document["input_dir"] = str(
                config.resolve_path(source.input_dir)
            )
    return atomic_write_json(target, document)


def save_batch_manifest(path: str | Path, manifest: BatchManifest) -> Path:
    return atomic_write_json(path, manifest.to_dict())


def _save_run_manifest(
    latest_path: Path,
    archive_path: Path,
    manifest: BatchManifest,
) -> None:
    save_batch_manifest(archive_path, manifest)
    save_batch_manifest(latest_path, manifest)


def _save_item_record(directory: Path, item: BatchItemRecord) -> Path:
    filename = f"{item.index:04d}_{safe_batch_filename(item.batch_id)}.json"
    return atomic_write_json(directory / filename, item.to_dict())


def atomic_write_json(path: str | Path, document: object) -> Path:
    """Preserve the batch API's explicit JSON value normalization."""
    return _atomic_write_json(path, document, normalizer=_json_safe)


def _promote_no_replace(source: Path, target: Path) -> None:
    """Promote ``source`` without replacing a destination that appeared.

    A hard link gives us atomic create-if-absent semantics on the common local
    filesystems.  Some removable and network filesystems do not support hard
    links, so claim the destination exclusively before atomically replacing
    that private claim.  The fallback briefly exposes only a small sentinel,
    never a partially written scientific output.
    """
    try:
        os.link(source, target)
    except FileExistsError:
        raise
    except OSError:
        _promote_via_exclusive_claim(source, target)
    else:
        _best_effort_unlink(source)


def _promote_via_exclusive_claim(source: Path, target: Path) -> None:
    token = f"napari-vipp-claim:{uuid.uuid4().hex}\n".encode()
    descriptor = os.open(
        target,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    claim_stat = None
    try:
        claim_stat = os.fstat(descriptor)
        remaining = memoryview(token)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("Could not initialize the exclusive output claim.")
            remaining = remaining[written:]
        os.fsync(descriptor)
    except BaseException:
        if claim_stat is None:
            try:
                claim_stat = target.stat()
            except OSError:
                pass
        try:
            os.close(descriptor)
        except OSError:
            pass
        if claim_stat is not None:
            _remove_owned_claim(target, claim_stat.st_dev, claim_stat.st_ino)
        raise
    try:
        os.close(descriptor)
    except BaseException:
        _remove_owned_claim(target, claim_stat.st_dev, claim_stat.st_ino)
        raise
    try:
        _replace_with_retry(source, target)
    except BaseException:
        _remove_owned_claim(
            target,
            claim_stat.st_dev,
            claim_stat.st_ino,
            expected_content=token,
        )
        raise


def _remove_owned_claim(
    target: Path,
    device: int,
    inode: int,
    *,
    expected_content: bytes | None = None,
) -> None:
    """Remove a failed fallback claim only when it is still ours."""
    try:
        current = target.stat()
        if (current.st_dev, current.st_ino) != (device, inode):
            return
        if expected_content is not None and target.read_bytes() != expected_content:
            return
        _best_effort_unlink(target)
    except OSError:
        return


def _best_effort_unlink(path: Path) -> None:
    """Retry transient cleanup locks without changing the scientific result."""
    for attempt in range(6):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            if attempt < 5:
                time.sleep(0.01 * (2**attempt))
        except OSError:
            return


def build_batch_plan(config: BatchConfig) -> BatchPlan:
    """Resolve source pairing and every output path without loading image data."""
    source_lists: dict[str, list[Path]] = {}
    counts: dict[str, int] = {}
    for source in config.sources:
        input_dir = config.resolve_path(source.input_dir)
        if not input_dir.is_dir():
            raise ValueError(f"Batch source '{source.title}' folder does not exist.")
        paths = _iter_source_paths(input_dir, source.pattern)
        if not paths:
            raise ValueError(
                f"No files matched '{source.pattern}' for "
                f"batch source '{source.title}'."
            )
        source_lists[source.node_id] = paths
        counts[source.title] = len(paths)
    expected = len(next(iter(source_lists.values())))
    if any(len(paths) != expected for paths in source_lists.values()):
        summary = ", ".join(f"{title}={count}" for title, count in counts.items())
        raise ValueError(
            "Bound batch sources must contain the same number of matched files "
            f"so they can be paired by sorted order ({summary})."
        )

    output_dir = config.resolve_path(config.output_dir)
    primary_id = config.sources[0].node_id
    items: list[BatchItemPlan] = []
    for item_index in range(expected):
        source_paths = {
            source.node_id: source_lists[source.node_id][item_index]
            for source in config.sources
        }
        primary_source = source_paths[primary_id]
        batch_id = safe_batch_filename(
            f"{item_index + 1:04d}_{batch_source_stem(primary_source)}"
        )
        outputs = tuple(
            _plan_output(
                config,
                output_dir,
                output,
                item_index + 1,
                batch_id,
                primary_source,
            )
            for output in config.outputs
        )
        items.append(
            BatchItemPlan(
                index=item_index + 1,
                batch_id=batch_id,
                primary_source=primary_source,
                source_paths=source_paths,
                outputs=outputs,
            )
        )

    target_counts: dict[str, int] = {}
    for item in items:
        for output in item.outputs:
            key = os.path.normcase(str(output.path.resolve(strict=False)))
            target_counts[key] = target_counts.get(key, 0) + 1
    all_source_paths = tuple(
        path
        for item in items
        for path in item.source_paths.values()
    )
    resolved_items = []
    for item in items:
        resolved_outputs = []
        for output in item.outputs:
            key = os.path.normcase(str(output.path.resolve(strict=False)))
            resolved_outputs.append(
                replace(
                    output,
                    duplicate=target_counts.get(key, 0) > 1,
                    input_collision=any(
                        _output_overlaps_source(output.path, source_path)
                        for source_path in all_source_paths
                    ),
                )
            )
        resolved_items.append(replace(item, outputs=tuple(resolved_outputs)))
    return BatchPlan(config, tuple(resolved_items), output_dir)


def _with_fixed_source_collisions(
    plan: BatchPlan,
    fixed_source_paths,
) -> BatchPlan:
    fixed_paths = tuple(Path(path) for path in fixed_source_paths)
    if not fixed_paths:
        return plan
    items = []
    for item in plan.items:
        outputs = tuple(
            replace(
                output,
                input_collision=(
                    output.input_collision
                    or any(
                        _output_overlaps_source(output.path, source_path)
                        for source_path in fixed_paths
                    )
                ),
            )
            for output in item.outputs
        )
        items.append(replace(item, outputs=outputs))
    return replace(plan, items=tuple(items))


def _output_overlaps_source(output_path: Path, source_path: Path) -> bool:
    output_text = os.path.normcase(str(output_path.resolve(strict=False)))
    source_text = os.path.normcase(str(source_path.resolve(strict=False)))
    if output_text == source_text:
        return True
    if not source_path.is_dir():
        return False
    try:
        return os.path.commonpath((output_text, source_text)) == source_text
    except ValueError:
        return False


def _collision_paths(plan: BatchPlan) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for item in plan.items:
        for output in item.outputs:
            collision = output.duplicate or output.input_collision or (
                output.exists
                and output.existing_file_policy == ExistingFilePolicy.ERROR
            )
            text = str(output.path)
            if collision and text not in seen:
                seen.add(text)
                paths.append(text)
    return paths


def run_batch_from_files(
    workflow_path: str | Path | None,
    config_path: str | Path,
    *,
    progress_callback: Callable[[int, int, str, str], None] | None = None,
) -> BatchRunResult:
    """Load a saved workflow/config pair and execute it headlessly."""
    if not str(config_path).strip():
        raise ValueError("Batch config path cannot be blank.")
    config_source = Path(str(config_path).strip()).expanduser().resolve()
    config = load_batch_config(config_source)
    if workflow_path is None or not str(workflow_path).strip():
        workflow_source = config.resolve_path(config.workflow_file).resolve()
    else:
        workflow_source = Path(str(workflow_path).strip()).expanduser().resolve()
    workflow = json.loads(workflow_source.read_text(encoding="utf-8"))
    return run_batch(
        workflow,
        config,
        workflow_path=workflow_source,
        config_path=config_source,
        progress_callback=progress_callback,
    )


def run_batch(
    workflow: object,
    config: BatchConfig,
    *,
    workflow_path: str | Path | None = None,
    config_path: str | Path | None = None,
    plan: BatchPlan | None = None,
    progress_callback: Callable[[int, int, str, str], None] | None = None,
) -> BatchRunResult:
    """Execute a deterministic batch plan with checkpointed provenance."""
    workflow_sha256 = scientific_workflow_hash(workflow)
    pipeline, fixed_source_paths = _validated_batch_pipeline(
        workflow,
        config,
        workflow_sha256,
        workflow_path=workflow_path,
    )
    if plan is None:
        plan = build_batch_plan(config)
    elif plan.config is not config:
        raise ValueError(
            "A supplied batch plan must use the exact validated config instance."
        )
    plan = _with_fixed_source_collisions(plan, fixed_source_paths.values())
    if plan.has_collisions:
        collisions = _collision_paths(plan)
        preview = ", ".join(collisions[:3])
        suffix = "" if len(collisions) <= 3 else f" (+{len(collisions) - 3} more)"
        raise FileExistsError(
            "Batch preflight found output collisions: " + preview + suffix
        )
    plan.output_dir.mkdir(parents=True, exist_ok=True)

    workflow_label = str(workflow_path or config.workflow_file)
    config_label = str(config_path or BATCH_CONFIG_FILENAME)
    manifest_path = plan.output_dir / BATCH_MANIFEST_FILENAME
    manifest = _seed_manifest(
        plan,
        workflow_sha256,
        batch_config_hash(config),
        workflow_label,
        config_label,
        scientific_workflow_document(workflow),
        config.to_dict(),
        fixed_source_paths,
    )
    manifest_archive_path = plan.output_dir / (
        f"vipp_batch_manifest_{manifest.run_id}.json"
    )
    item_records_dir = plan.output_dir / f"vipp_batch_items_{manifest.run_id}"
    item_records_dir.mkdir(parents=True, exist_ok=False)
    manifest = replace(manifest, item_records_dir=item_records_dir.name)
    _save_run_manifest(manifest_path, manifest_archive_path, manifest)
    saved_paths: list[Path] = []
    output_node_ids = tuple(output.node_id for output in config.outputs)
    total = len(plan.items)

    for item_position, item_plan in enumerate(plan.items):
        item_record = manifest.items[item_position]
        item_record = replace(
            item_record,
            status=BatchStatus.RUNNING,
            started_at=_timestamp(),
        )
        manifest = manifest.replace_item(item_record)
        _save_item_record(item_records_dir, item_record)
        _report_progress(
            progress_callback,
            item_plan.index,
            total,
            item_plan.batch_id,
            "running",
        )

        item_error: Exception | None = None
        sources = list(item_record.sources)
        source_paths: dict[str, Path] = {}
        source_identities: dict[str, LocalSourceIdentity] = {}
        try:
            source_paths = _item_source_paths(
                pipeline,
                item_plan,
                fixed_source_paths,
            )
            source_identities = _capture_item_source_identities(source_paths)
            payloads, sources = _source_payloads_for_item(
                pipeline,
                item_plan,
                config,
                source_paths,
                source_identities,
            )
            pipeline.run(
                None,
                input_metadata=None,
                input_name="",
                source_payloads=payloads,
                retain_node_ids=output_node_ids,
                prune_unretained=True,
            )
        except Exception as exc:
            item_error = exc
            item_record = replace(item_record, sources=tuple(sources))
        finally:
            # Fully stage every available branch first. This forces lazy arrays
            # to finish reading their sources without publishing an output.
            output_records = list(item_record.outputs)
            staged_outputs: dict[int, _StagedBatchOutput] = {}
            for output_index, output_plan in enumerate(item_plan.outputs):
                output_record = output_records[output_index]
                if (
                    item_error is not None
                    and pipeline.outputs.get(output_plan.node_id) is None
                ):
                    output_record = replace(
                        output_record,
                        status=BatchStatus.FAILED,
                        error_type=type(item_error).__name__,
                        error_message=str(item_error),
                    )
                else:
                    try:
                        staged = _save_planned_output(pipeline, output_plan)
                    except _SkippedOutput as exc:
                        output_record = replace(
                            output_record,
                            status=BatchStatus.SKIPPED,
                            error_type="",
                            error_message=str(exc),
                        )
                    except Exception as exc:
                        output_record = replace(
                            output_record,
                            status=BatchStatus.FAILED,
                            error_type=type(exc).__name__,
                            error_message=str(exc),
                        )
                    else:
                        staged_outputs[output_index] = staged
                output_records[output_index] = output_record
                running_record = replace(
                    item_record,
                    sources=tuple(sources),
                    outputs=tuple(output_records),
                )
                manifest = manifest.replace_item(running_record)
                _save_item_record(item_records_dir, running_record)

            source_change_error: SourceChangedError | None = None
            if source_identities:
                try:
                    _verify_item_source_identities(
                        source_paths,
                        source_identities,
                    )
                except SourceChangedError as exc:
                    source_change_error = exc
                    item_error = exc

            if source_change_error is not None:
                for staged in staged_outputs.values():
                    _cleanup_staged_output(staged)
                staged_outputs.clear()
                output_records = [
                    replace(
                        output_record,
                        status=BatchStatus.FAILED,
                        size_bytes=None,
                        overwrote_existing=False,
                        error_type=type(source_change_error).__name__,
                        error_message=str(source_change_error),
                    )
                    for output_record in output_records
                ]
                running_record = replace(
                    item_record,
                    sources=tuple(sources),
                    outputs=tuple(output_records),
                )
                manifest = manifest.replace_item(running_record)
                _save_item_record(item_records_dir, running_record)
            else:
                # All source-dependent bytes are now private and stable. Only
                # atomic promotion remains; it cannot read a source again.
                for output_index, staged in staged_outputs.items():
                    output_record = output_records[output_index]
                    try:
                        saved = _promote_staged_output(staged)
                    except _SkippedOutput as exc:
                        output_record = replace(
                            output_record,
                            status=BatchStatus.SKIPPED,
                            error_type="",
                            error_message=str(exc),
                        )
                    except Exception as exc:
                        output_record = replace(
                            output_record,
                            status=BatchStatus.FAILED,
                            error_type=type(exc).__name__,
                            error_message=str(exc),
                        )
                    else:
                        saved_paths.append(saved)
                        try:
                            size = saved.stat().st_size if saved.is_file() else None
                        except OSError:
                            size = None
                        output_record = replace(
                            output_record,
                            status=BatchStatus.COMPLETED,
                            size_bytes=size,
                            overwrote_existing=(
                                output_record.existed_at_preflight
                                and output_record.existing_file_policy
                                == ExistingFilePolicy.OVERWRITE
                            ),
                        )
                    output_records[output_index] = output_record
                    running_record = replace(
                        item_record,
                        sources=tuple(sources),
                        outputs=tuple(output_records),
                    )
                    manifest = manifest.replace_item(running_record)
                    _save_item_record(item_records_dir, running_record)
            item_record = replace(
                item_record,
                sources=tuple(sources),
                outputs=tuple(output_records),
            )
            pipeline.prune_cached_outputs(())

        if item_error is not None:
            derived_status = _item_status(item_record.outputs)
            item_record = replace(
                item_record,
                status=(
                    BatchStatus.FAILED
                    if derived_status == BatchStatus.FAILED
                    else BatchStatus.PARTIAL
                ),
                error_type=type(item_error).__name__,
                error_message=str(item_error),
            )
        else:
            item_record = replace(
                item_record,
                status=_item_status(item_record.outputs),
            )
        item_record = replace(item_record, finished_at=_timestamp())
        manifest = manifest.replace_item(item_record)
        _save_item_record(item_records_dir, item_record)
        _report_progress(
            progress_callback,
            item_plan.index,
            total,
            item_plan.batch_id,
            item_record.status.value,
        )

        item_has_failure = bool(item_record.error_type) or (
            item_record.status == BatchStatus.FAILED
            or any(
                output.status == BatchStatus.FAILED
                for output in item_record.outputs
            )
        )
        if item_has_failure and not config.continue_on_error:
            manifest = _skip_remaining_items(
                manifest,
                start_index=item_position + 1,
                reason="Not run because continue_on_error is disabled.",
            )
            for skipped_item in manifest.items[item_position + 1 :]:
                _save_item_record(item_records_dir, skipped_item)
            break

    manifest = replace(manifest, finished_at=_timestamp())
    _save_run_manifest(manifest_path, manifest_archive_path, manifest)
    return BatchRunResult(
        manifest,
        manifest_path,
        tuple(saved_paths),
        manifest_archive_path=manifest_archive_path,
    )


def _report_progress(
    callback: Callable[[int, int, str, str], None] | None,
    index: int,
    total: int,
    batch_id: str,
    status: str,
) -> None:
    if callback is None:
        return
    try:
        callback(index, total, batch_id, status)
    except Exception:
        # Presentation hooks must never invalidate scientific execution or
        # provenance finalization.
        return


def preflight_batch(
    workflow: object,
    config: BatchConfig,
    *,
    workflow_path: str | Path | None = None,
) -> BatchPlan:
    """Validate and plan a batch, raising before any artifact is modified."""
    plan = plan_batch(workflow, config, workflow_path=workflow_path)
    if plan.has_collisions:
        collisions = _collision_paths(plan)
        preview = ", ".join(collisions[:3])
        suffix = "" if len(collisions) <= 3 else f" (+{len(collisions) - 3} more)"
        raise FileExistsError(
            "Batch preflight found output collisions: " + preview + suffix
        )
    return plan


def plan_batch(
    workflow: object,
    config: BatchConfig,
    *,
    workflow_path: str | Path | None = None,
) -> BatchPlan:
    """Return the fully validated plan, including fixed-source collisions."""
    _pipeline, fixed_source_paths = _validated_batch_pipeline(
        workflow,
        config,
        scientific_workflow_hash(workflow),
        workflow_path=workflow_path,
    )
    plan = _with_fixed_source_collisions(
        build_batch_plan(config),
        fixed_source_paths.values(),
    )
    return plan


def validate_batch_config(
    workflow: object,
    config: BatchConfig,
    *,
    workflow_path: str | Path | None = None,
) -> None:
    """Validate a config against a workflow without planning or execution."""
    _validated_batch_pipeline(
        workflow,
        config,
        scientific_workflow_hash(workflow),
        workflow_path=workflow_path,
    )


def _validated_batch_pipeline(
    workflow: object,
    config: BatchConfig,
    workflow_sha256: str,
    *,
    workflow_path: str | Path | None = None,
) -> tuple[PrototypePipeline, dict[str, Path]]:
    if workflow_sha256 != config.workflow_sha256:
        raise ValueError(
            "Batch config workflow hash does not match the selected workflow."
        )
    restored = deserialize_workflow(workflow)
    pipeline = PrototypePipeline()
    pipeline.restore_graph(
        restored["nodes"],
        restored["connections"],
        restored.get("output_tunnels", ()),
    )
    fixed_source_paths = _validate_pipeline_config(
        pipeline,
        config,
        workflow_path=workflow_path,
    )
    return pipeline, fixed_source_paths


def _plan_output(
    config: BatchConfig,
    output_dir: Path,
    output: BatchOutputConfig,
    index: int,
    batch_id: str,
    primary_source: Path,
) -> BatchOutputPlan:
    source_stem = batch_source_stem(primary_source)
    values = {
        "source_stem": source_stem,
        "tag": safe_batch_filename(output.tag),
        "node_id": safe_batch_filename(output.node_id),
        "node_title": safe_batch_filename(output.node_title),
        "batch_id": batch_id,
        "batch_index": f"{index:04d}",
        "source_name": safe_batch_filename(primary_source.name),
        "primary_source_stem": source_stem,
    }
    filename = format_batch_filename(output.filename_template, values)
    resolved_format = _resolved_output_format(config, output)
    suffix = (
        ".tsv"
        if resolved_format == "tsv"
        else ".csv"
        if resolved_format == "csv"
        else _IMAGE_SUFFIXES[resolved_format]
    )
    if filename.lower().endswith(_KNOWN_SUFFIXES):
        if not _filename_suffix_matches_format(filename, resolved_format):
            raise ValueError(
                f"Batch filename {filename!r} has an extension that conflicts "
                f"with format {resolved_format!r}."
            )
    else:
        filename += suffix
    folder = output_dir
    for part in re.split(r"[\\/]+", output.subfolder):
        safe = safe_batch_filename(part) if part else ""
        if safe:
            folder /= safe
    policy = _resolved_existing_file_policy(config, output)
    path = folder / filename
    return BatchOutputPlan(
        output.node_id,
        output.node_title,
        output.tag,
        output.kind,
        resolved_format,
        path,
        policy,
        exists=path.exists(),
    )


def _resolved_output_format(config: BatchConfig, output: BatchOutputConfig) -> str:
    if output.format != "batch default":
        return output.format
    return "csv" if output.kind == "table" else config.default_image_format


def _resolved_existing_file_policy(
    config: BatchConfig, output: BatchOutputConfig
) -> ExistingFilePolicy:
    if output.overwrite == "yes":
        return ExistingFilePolicy.OVERWRITE
    if output.overwrite == "no":
        return ExistingFilePolicy.ERROR
    return config.existing_file_policy


def _filename_suffix_matches_format(filename: str, output_format: str) -> bool:
    lower = filename.lower()
    if output_format == "ome-tiff":
        return lower.endswith((".ome.tif", ".ome.tiff", ".tif", ".tiff"))
    if output_format in {"imagej-tiff", "tiff"}:
        return lower.endswith((".tif", ".tiff")) and not lower.endswith(
            (".ome.tif", ".ome.tiff")
        )
    if output_format == "npy":
        return lower.endswith(".npy")
    if output_format == "csv":
        return lower.endswith(".csv")
    if output_format == "tsv":
        return lower.endswith(".tsv")
    return False


def _validate_pipeline_config(
    pipeline: PrototypePipeline,
    config: BatchConfig,
    *,
    workflow_path: str | Path | None,
) -> dict[str, Path]:
    source_ids = {
        node_id
        for node_id, node in pipeline.nodes.items()
        if node.operation_id == "input"
    }
    unknown_sources = {source.node_id for source in config.sources} - source_ids
    if unknown_sources:
        raise ValueError(
            "Batch config references missing source nodes: "
            + ", ".join(sorted(unknown_sources))
            + "."
        )
    unbound_sources = source_ids - {source.node_id for source in config.sources}
    fixed_base = _fixed_source_base_dir(config, workflow_path)
    fixed_source_paths: dict[str, Path] = {}
    for node_id in sorted(unbound_sources):
        node = pipeline.nodes[node_id]
        if str(node.params.get("source_mode", "napari layer")) != "file path":
            raise ValueError(
                f"Image Source {node_id!r} is not bound to a collection and "
                "does not use a reproducible file path."
            )
        path = Path(str(node.params.get("file_path", "")).strip()).expanduser()
        if not path.is_absolute():
            path = (fixed_base / path).resolve()
        else:
            path = path.resolve()
        if not _is_supported_local_image_source(path):
            raise ValueError(
                f"Fixed Image Source {node_id!r} path does not exist or is not "
                "a supported local image source."
            )
        fixed_source_paths[node_id] = path

    enabled_save_nodes = [
        node_id
        for node_id, node in pipeline.nodes.items()
        if node.operation_id == "save_output"
        and str(node.params.get("enabled", "off")).lower() == "on"
    ]
    if enabled_save_nodes:
        raise ValueError(
            "Batch workflows cannot run enabled Save Image nodes because they "
            "publish before batch source verification. Disable them and use "
            "Batch Output nodes instead: "
            + ", ".join(enabled_save_nodes)
            + "."
        )

    explicit = [
        node_id
        for node_id in pipeline.topological_order()
        if pipeline.nodes[node_id].operation_id == "batch_output"
    ]
    if explicit:
        expected_outputs = explicit
    else:
        consumed = {connection.source_id for connection in pipeline.connections}
        order = pipeline.topological_order()
        expected_outputs = [node_id for node_id in order if node_id not in consumed]
        expected_outputs = expected_outputs or order
        multi_output_terminals = [
            node_id
            for node_id in expected_outputs
            if len(pipeline.output_ports(node_id)) > 1
        ]
        if multi_output_terminals:
            raise ValueError(
                "Terminal-output compatibility fallback cannot save all ports "
                "from multi-output nodes. Add one Batch Output node for each "
                "desired port: "
                + ", ".join(multi_output_terminals)
                + "."
            )
    configured_outputs = [output.node_id for output in config.outputs]
    if configured_outputs != expected_outputs:
        raise ValueError(
            "Batch config selected outputs do not match the workflow's "
            "Batch Output selection."
        )
    for output in config.outputs:
        node = pipeline.nodes[output.node_id]
        ports = pipeline.output_ports(output.node_id)
        output_type = ports[0].output_type if ports else "any"
        expected_kind = "table" if output_type == "table" else "image"
        if output.kind != expected_kind:
            raise ValueError(
                f"Batch output {output.node_id!r} kind does not match the workflow."
            )
        if node.operation_id == "batch_output":
            params = node.params
            raw_tag = str(params.get("tag", "")).strip()
            expected_tag = safe_batch_filename(raw_tag or output.node_id)
            expected_format = str(params.get("format", "batch default"))
            expected_subfolder = str(params.get("subfolder", ""))
            expected_template = str(
                params.get("filename_template", "{source_stem}__{tag}")
            )
            expected_overwrite = str(params.get("overwrite", "batch default"))
        else:
            expected_tag = safe_batch_filename(f"{node.title}-{output.node_id}")
            expected_format = "batch default"
            expected_subfolder = ""
            expected_template = "{source_stem}__{tag}"
            expected_overwrite = "batch default"
        expected = (
            node.title,
            expected_tag,
            expected_kind,
            expected_format,
            expected_subfolder,
            expected_template,
            expected_overwrite,
        )
        actual = (
            output.node_title,
            output.tag,
            output.kind,
            output.format,
            output.subfolder,
            output.filename_template,
            output.overwrite,
        )
        if actual != expected:
            raise ValueError(
                f"Batch output {output.node_id!r} declaration does not match "
                "the workflow's Batch Output settings."
            )
    return fixed_source_paths


def _fixed_source_base_dir(
    config: BatchConfig,
    workflow_path: str | Path | None,
) -> Path:
    if workflow_path is None:
        workflow_path = config.resolve_path(config.workflow_file)
    path = Path(workflow_path).expanduser()
    if not path.is_absolute():
        path = config.resolve_path(path)
    return path.resolve(strict=False).parent


def _source_payloads_for_item(
    pipeline: PrototypePipeline,
    item: BatchItemPlan,
    config: BatchConfig,
    source_paths: dict[str, Path],
    source_identities: dict[str, LocalSourceIdentity],
) -> tuple[dict[str, SourcePayload], list[dict[str, object]]]:
    payloads: dict[str, SourcePayload] = {}
    records: list[dict[str, object]] = []
    for node_id, node in pipeline.nodes.items():
        if node.operation_id != "input":
            continue
        path = source_paths[node_id]
        binding = next(
            (source for source in config.sources if source.node_id == node_id),
            None,
        )
        if node_id not in item.source_paths:
            title = node.title
            role = "fixed"
        else:
            title = binding.title if binding is not None else node.title
            role = "collection"
        dataset = read_image(
            path,
            series_index=int(node.params.get("series_index", 0)),
        )
        provenance = _json_safe(dataset.provenance)
        payloads[node_id] = SourcePayload(
            dataset.data,
            {
                "vipp_source_path": str(path),
                "vipp_source_provenance": provenance,
            },
            dataset.selected_series.name or path.name,
            dataset.image_state,
        )
        identity = {
            **_path_identity(path),
            **source_identities[node_id].to_dict(),
        }
        records.append(
            {
                "node_id": node_id,
                "title": title,
                "role": role,
                "path": str(path),
                "identity": identity,
                "series": {
                    "index": dataset.selected_series.index,
                    "key": dataset.selected_series.key,
                    "name": dataset.selected_series.name,
                    "shape": list(dataset.selected_series.shape),
                    "dtype": dataset.selected_series.dtype,
                    "axes": dataset.selected_series.axes,
                    "kind": dataset.selected_series.kind,
                },
                "image_state": dataset.image_state.to_dict(),
                "provenance": provenance,
            }
        )
    return payloads, records


def _item_source_paths(
    pipeline: PrototypePipeline,
    item: BatchItemPlan,
    fixed_source_paths: dict[str, Path],
) -> dict[str, Path]:
    return {
        node_id: (
            item.source_paths[node_id]
            if node_id in item.source_paths
            else fixed_source_paths[node_id]
        )
        for node_id, node in pipeline.nodes.items()
        if node.operation_id == "input"
    }


def _capture_item_source_identities(
    source_paths: dict[str, Path],
) -> dict[str, LocalSourceIdentity]:
    return {
        node_id: capture_local_source_identity(path)
        for node_id, path in source_paths.items()
    }


def _verify_item_source_identities(
    source_paths: dict[str, Path],
    source_identities: dict[str, LocalSourceIdentity],
) -> None:
    for node_id, path in source_paths.items():
        verify_local_source_identity(path, source_identities[node_id])


def _save_planned_output(
    pipeline: PrototypePipeline,
    output: BatchOutputPlan,
) -> _StagedBatchOutput:
    """Fully write an output privately without publishing its destination."""
    if output.duplicate:
        raise FileExistsError(
            f"Multiple planned outputs use destination {output.path}."
        )
    if output.input_collision:
        raise FileExistsError(
            f"Output destination overlaps an input source: {output.path}."
        )
    if output.path.exists():
        if output.existing_file_policy == ExistingFilePolicy.SKIP:
            raise _SkippedOutput(
                f"Existing destination was left unchanged: {output.path}"
            )
        if output.existing_file_policy == ExistingFilePolicy.ERROR:
            raise FileExistsError(f"Output already exists: {output.path}")
    data = pipeline.outputs.get(output.node_id)
    if data is None:
        raise ValueError(f"Batch output {output.node_id!r} produced no data.")
    output.path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_output_path(output.path)
    saved_temporary = temporary
    try:
        if is_table_data(data):
            if output.format not in _TABLE_FORMATS:
                raise ValueError(
                    f"Table output {output.node_id!r} has invalid format "
                    f"{output.format!r}."
                )
            saved_temporary = save_table_output(
                data,
                temporary,
                format=output.format,
                overwrite=True,
            )
        else:
            if output.format not in _IMAGE_FORMATS:
                raise ValueError(
                    f"Image output {output.node_id!r} has invalid format "
                    f"{output.format!r}."
                )
            saved_temporary = save_array_output(
                data,
                temporary,
                format=output.format,
                overwrite=True,
                image_state=pipeline.output_states.get(output.node_id),
            )
        saved_temporary = Path(saved_temporary)
        with saved_temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
    except BaseException:
        _best_effort_unlink(Path(saved_temporary))
        _best_effort_unlink(temporary)
        raise
    return _StagedBatchOutput(output, temporary, saved_temporary)


def _promote_staged_output(staged: _StagedBatchOutput) -> Path:
    output = staged.plan
    saved_temporary = staged.saved_temporary_path
    try:
        if output.existing_file_policy == ExistingFilePolicy.OVERWRITE:
            _replace_with_retry(saved_temporary, output.path)
        else:
            try:
                _promote_no_replace(saved_temporary, output.path)
            except FileExistsError as exc:
                if output.existing_file_policy == ExistingFilePolicy.SKIP:
                    raise _SkippedOutput(
                        f"Destination appeared during execution and was left "
                        f"unchanged: {output.path}"
                    ) from exc
                raise FileExistsError(
                    f"Output appeared during execution: {output.path}"
                ) from exc
        return output.path
    finally:
        _cleanup_staged_output(staged)


def _cleanup_staged_output(staged: _StagedBatchOutput) -> None:
    _best_effort_unlink(staged.saved_temporary_path)
    _best_effort_unlink(staged.temporary_path)


def _temporary_output_path(path: Path) -> Path:
    suffix = "".join(path.suffixes)
    stem = path.name[: -len(suffix)] if suffix else path.name
    return path.with_name(f".{stem}.{uuid.uuid4().hex}.tmp{suffix}")


def _seed_manifest(
    plan: BatchPlan,
    workflow_sha256: str,
    config_sha256: str,
    workflow_file: str,
    config_file: str,
    workflow_document: dict[str, object],
    config_document: dict[str, object],
    fixed_source_paths: dict[str, Path],
) -> BatchManifest:
    items = []
    for item in plan.items:
        sources = tuple(
            _planned_source_record(node_id, path)
            for node_id, path in item.source_paths.items()
        )
        sources += tuple(
            _planned_source_record(node_id, path, role="fixed")
            for node_id, path in fixed_source_paths.items()
        )
        outputs = tuple(
            BatchOutputRecord(
                output.node_id,
                output.node_title,
                output.tag,
                output.kind,
                output.format,
                str(output.path),
                output.existing_file_policy,
                output.exists,
                existing_identity=_path_identity(output.path) if output.exists else {},
            )
            for output in item.outputs
        )
        items.append(BatchItemRecord(item.index, item.batch_id, sources, outputs))
    return BatchManifest(
        run_id=uuid.uuid4().hex,
        started_at=_timestamp(),
        workflow_sha256=workflow_sha256,
        config_sha256=config_sha256,
        workflow_file=workflow_file,
        config_file=config_file,
        output_dir=str(plan.output_dir),
        runtime=_runtime_versions(),
        workflow_document=workflow_document,
        config_document=config_document,
        items=tuple(items),
    )


def _planned_source_record(
    node_id: str,
    path: Path,
    *,
    role: str = "collection",
) -> dict[str, object]:
    record: dict[str, object] = {
        "node_id": node_id,
        "role": role,
        "path": str(path),
    }
    identity = _path_identity(path)
    if identity:
        record["identity"] = identity
    return record


def _path_identity(path: Path) -> dict[str, int]:
    try:
        stat = path.stat()
    except OSError:
        return {}
    return {
        "size_bytes": stat.st_size,
        "modified_ns": stat.st_mtime_ns,
    }


def _item_status(outputs: tuple[BatchOutputRecord, ...]) -> BatchStatus:
    statuses = {output.status for output in outputs}
    if statuses == {BatchStatus.COMPLETED}:
        return BatchStatus.COMPLETED
    if statuses == {BatchStatus.SKIPPED}:
        return BatchStatus.SKIPPED
    if statuses == {BatchStatus.FAILED}:
        return BatchStatus.FAILED
    return BatchStatus.PARTIAL


def _skip_remaining_items(
    manifest: BatchManifest, *, start_index: int, reason: str
) -> BatchManifest:
    items = list(manifest.items)
    for index in range(start_index, len(items)):
        item = items[index]
        outputs = tuple(
            replace(
                output,
                status=BatchStatus.SKIPPED,
                error_message=reason,
            )
            for output in item.outputs
        )
        items[index] = replace(
            item,
            status=BatchStatus.SKIPPED,
            outputs=outputs,
            finished_at=_timestamp(),
            error_message=reason,
        )
    return replace(manifest, items=tuple(items))


def _runtime_versions() -> dict[str, object]:
    distributions = (
        "napari-vipp",
        "numpy",
        "scipy",
        "scikit-image",
        "dask",
        "tifffile",
        "zarr",
        "ome-zarr",
        "ome-types",
        "imageio",
        "fsspec",
        "pillow",
        "qtpy",
        "napari",
        "bioio",
        "bioio-bioformats",
        "bioio-czi",
        "bioio-lif",
    )
    versions: dict[str, str] = {}
    for distribution in distributions:
        try:
            versions[distribution] = package_version(distribution)
        except PackageNotFoundError:
            continue
    versions.setdefault("napari-vipp", VIPP_VERSION)
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
    }


def _iter_source_paths(input_dir: Path, pattern: str) -> list[Path]:
    patterns = [
        item.strip() for item in _PATTERN_SEPARATORS.split(str(pattern)) if item.strip()
    ] or ["*.tif"]
    paths: list[Path] = []
    seen: set[Path] = set()
    for item in patterns:
        for path in input_dir.glob(item):
            if not _is_supported_local_image_source(path) or path in seen:
                continue
            seen.add(path)
            paths.append(path)
    return sorted(
        paths,
        key=lambda path: (
            path.name.casefold(),
            os.path.normcase(str(path.resolve(strict=False))),
        ),
    )


def format_batch_filename(template: str, values: dict[str, str]) -> str:
    try:
        filename = template.format(**values)
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(
            f"Invalid batch filename template {template!r}: {exc}"
        ) from exc
    parts = [
        safe_batch_filename(part) for part in re.split(r"[\\/]+", filename) if part
    ]
    return "_".join(part for part in parts if part) or values["tag"]


def batch_source_stem(path: Path) -> str:
    name = path.name
    lower = name.lower()
    for suffix in (
        ".ome.zarr",
        ".zarr",
        ".ome.tiff",
        ".ome.tif",
        ".tiff",
        ".tif",
    ):
        if lower.endswith(suffix):
            return safe_batch_filename(name[: -len(suffix)])
    return safe_batch_filename(path.stem)


def _is_supported_local_image_source(path: Path) -> bool:
    return path.is_file() or (
        path.is_dir() and path.suffix.lower() == ".zarr"
    )


def safe_batch_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    safe = safe or "output"
    if safe.split(".", 1)[0].upper() in _WINDOWS_RESERVED_STEMS:
        safe = f"_{safe}"
    return safe


def _document_hash(document: object) -> str:
    encoded = json.dumps(
        _json_safe(document),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _config_path_text(path: Path) -> str:
    """Serialize relative config paths portably across operating systems."""
    value = Path(path)
    return str(value) if value.is_absolute() else value.as_posix()


def _canonical_mapping(value: object) -> dict[str, object]:
    data = _require_object(value, "Canonical workflow record")
    return {str(key): _json_safe(data[key]) for key in sorted(data)}


def _json_safe(value: object) -> object:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def _timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _require_object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object.")
    return value


def _required_text(data: dict[str, Any], key: str, label: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} {key} must be non-empty text.")
    return value


def _optional_text(
    data: dict[str, Any], key: str, label: str, *, default: str = ""
) -> str:
    value = data.get(key, default)
    if not isinstance(value, str):
        raise ValueError(f"{label} {key} must be text.")
    return value


def _required_bool(data: dict[str, Any], key: str, label: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{label} {key} must be a boolean.")
    return value


def _require_text(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text.")


def _reject_unknown_keys(data: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        names = ", ".join(sorted(str(key) for key in unknown))
        raise ValueError(f"{label} contains unknown fields: {names}.")


def _reject_duplicate_ids(values, label: str) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        raise ValueError(
            f"Duplicate {label} node ids: {', '.join(sorted(duplicates))}."
        )


class _SkippedOutput(RuntimeError):
    pass


__all__ = [
    "BATCH_CONFIG_FILENAME",
    "BATCH_CONFIG_TYPE",
    "BATCH_CONFIG_VERSION",
    "BATCH_MANIFEST_FILENAME",
    "BATCH_MANIFEST_TYPE",
    "BATCH_MANIFEST_VERSION",
    "BATCH_SCRIPT_FILENAME",
    "BATCH_WORKFLOW_FILENAME",
    "BatchConfig",
    "BatchItemPlan",
    "BatchItemRecord",
    "BatchManifest",
    "BatchOutputConfig",
    "BatchOutputPlan",
    "BatchOutputRecord",
    "BatchPlan",
    "BatchRunResult",
    "BatchSourceConfig",
    "BatchStatus",
    "ExistingFilePolicy",
    "atomic_write_json",
    "atomic_write_text",
    "batch_config_hash",
    "build_batch_plan",
    "load_batch_config",
    "plan_batch",
    "preflight_batch",
    "run_batch",
    "run_batch_from_files",
    "safe_batch_filename",
    "save_batch_config",
    "save_batch_manifest",
    "scientific_workflow_document",
    "scientific_workflow_hash",
    "validate_batch_config",
]
