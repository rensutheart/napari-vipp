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
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

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
from napari_vipp.core.batch_execution import (
    BATCH_NODE_EXECUTION_OVERRIDE_IDENTITY,
    BatchNodeExecutionMode,
    BatchNodeExecutionOverride,
    batch_node_execution_overrides_document,
    node_execution_override_provenance,
    normalize_batch_node_execution_overrides,
    parse_batch_node_execution_overrides,
    validate_batch_node_execution_overrides,
    workflow_with_node_execution_overrides,
)
from napari_vipp.core.batch_parameters import (
    BATCH_PARAMETER_OVERRIDE_IDENTITY,
    BatchParameterOverride,
    BatchSourceParameterOverrides,
    batch_parameter_overrides_document,
    batch_source_item_override_key,
    normalize_batch_parameter_overrides,
    parameter_override_provenance,
    parse_batch_parameter_overrides,
    validate_batch_parameter_overrides,
    workflow_with_parameter_overrides,
)
from napari_vipp.core.compute import ComputeMode, ComputeRequest
from napari_vipp.core.execution import (
    PipelineExecutionFailure,
    PipelineRunRequest,
    execute_pipeline_request,
)
from napari_vipp.core.execution_provenance import (
    execution_provenance_digest,
    serialize_execution_provenance,
)
from napari_vipp.core.file_sources import load_frozen_file_source_snapshot
from napari_vipp.core.io import (
    MICROSCOPE_SUFFIXES,
    SourceInspection,
    inspect_image_source,
    inspect_image_state,
    read_image,
)
from napari_vipp.core.io.raster import RASTER_SUFFIXES
from napari_vipp.core.metadata import (
    AmbiguousAxisError,
    AxisDeclaration,
    apply_axis_declaration,
)
from napari_vipp.core.operations import save_array_output
from napari_vipp.core.pipeline import PrototypePipeline, SourcePayload
from napari_vipp.core.progress import OperationCancelled
from napari_vipp.core.source_identity import (
    LocalSourceIdentity,
    SourceChangedError,
    capture_local_source_bundle,
    capture_local_source_identity,
    local_source_identity_from_bundle,
    verify_local_source_identity,
)
from napari_vipp.core.source_item_persistence import (
    SOURCE_ITEM_PARAMETER,
    source_item_from_params,
)
from napari_vipp.core.source_items import SourceContainerBundle, SourceItem
from napari_vipp.core.source_resolution import (
    resolve_source_item,
    select_inspected_item,
    verify_saved_source_item,
)
from napari_vipp.core.source_window_planning import plan_exact_source_crop_window
from napari_vipp.core.tables import is_table_data, save_table_output
from napari_vipp.core.workflow import deserialize_workflow

if TYPE_CHECKING:
    from napari_vipp.core.compute_registry import ComputeRegistry
    from napari_vipp.core.execution import ComputePlanner

BATCH_CONFIG_TYPE = "napari-vipp-batch-config"
BATCH_CONFIG_VERSION = 5
BATCH_MANIFEST_TYPE = "napari-vipp-batch-manifest"
BATCH_MANIFEST_VERSION = 5

BATCH_CONFIG_FILENAME = "vipp_batch_config.json"
BATCH_MANIFEST_FILENAME = "vipp_batch_manifest.json"
BATCH_WORKFLOW_FILENAME = "vipp_batch_workflow.json"
BATCH_SCRIPT_FILENAME = "vipp_batch_pipeline.py"

PAIRING_POLICY = "sorted-position"
DEFAULT_BATCH_SOURCE_PATTERN = "*"
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
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class BatchExecutionProgress:
    """Nested operation progress tagged with its containing batch item."""

    item_index: int
    item_total: int
    batch_id: str
    node_id: str
    operation_id: str
    current: int
    total: int
    message: str = ""


class BatchExecutionError(RuntimeError):
    """One isolated execution service call failed for a batch item."""


class BatchRuntimeCleanupError(RuntimeError):
    """A batch item could not prove accelerator cleanup before publication."""


@dataclass(frozen=True, slots=True)
class BatchAxisSuggestion:
    """One safe, UI-actionable interpretation for a generic source axis."""

    source_node_id: str
    source_title: str
    declaration: AxisDeclaration


class BatchScientificPreflightError(ValueError):
    """A representative source cannot satisfy the workflow contract."""

    def __init__(
        self,
        message: str,
        *,
        user_message: str | None = None,
        axis_suggestion: BatchAxisSuggestion | None = None,
    ) -> None:
        super().__init__(message)
        self.technical_detail = str(message)
        self.user_message = str(user_message or message)
        self.axis_suggestion = axis_suggestion


@dataclass(frozen=True)
class BatchSourceConfig:
    """One workflow source bound to a local file collection."""

    node_id: str
    title: str
    input_dir: Path
    pattern: str
    axis_declaration: AxisDeclaration | None = None
    source_items: tuple[SourceItem, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.node_id, "Batch source node_id")
        _require_text(self.title, "Batch source title")
        _require_text(str(self.input_dir), "Batch source input_dir")
        _require_text(self.pattern, "Batch source pattern")
        object.__setattr__(
            self,
            "axis_declaration",
            AxisDeclaration.from_value(self.axis_declaration),
        )
        items = tuple(self.source_items)
        if any(not isinstance(item, SourceItem) for item in items):
            raise TypeError(
                "Batch source source_items must contain only SourceItem records."
            )
        digests = [item.digest for item in items]
        if len(digests) != len(set(digests)):
            raise ValueError("Batch source source_items must be unique.")
        items = tuple(
            sorted(
                items,
                key=lambda item: (
                    item.container.uri,
                    item.selector.key,
                    item.digest,
                ),
            )
        )
        object.__setattr__(self, "source_items", items)

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "node_id": self.node_id,
            "title": self.title,
            "input_dir": _config_path_text(self.input_dir),
            "pattern": self.pattern,
        }
        if self.axis_declaration is not None:
            result["axis_declaration"] = self.axis_declaration.to_dict()
        if self.source_items:
            result["source_items"] = [item.to_dict() for item in self.source_items]
        return result

    @property
    def source_item_documents(self) -> tuple[dict[str, object], ...]:
        """Detached canonical documents retained for transition callers."""

        return tuple(item.to_dict() for item in self.source_items)

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        index: int,
        config_version: int = BATCH_CONFIG_VERSION,
    ) -> BatchSourceConfig:
        data = _require_object(value, f"Batch source {index}")
        allowed = {"node_id", "title", "input_dir", "pattern"}
        if config_version >= 3:
            allowed.add("axis_declaration")
        if config_version >= 4:
            allowed.add("source_items")
        _reject_unknown_keys(
            data,
            allowed,
            f"Batch source {index}",
        )
        return cls(
            node_id=_required_text(data, "node_id", f"batch source {index}"),
            title=_required_text(data, "title", f"batch source {index}"),
            input_dir=Path(_required_text(data, "input_dir", f"batch source {index}")),
            pattern=_required_text(data, "pattern", f"batch source {index}"),
            axis_declaration=(
                AxisDeclaration.from_value(data.get("axis_declaration"))
                if config_version >= 3
                else None
            ),
            source_items=(
                _source_items_from_batch_source(data, index=index)
                if config_version >= 4
                else ()
            ),
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
    compute_request: ComputeRequest = field(default_factory=ComputeRequest)
    base_dir: Path | None = field(default=None, compare=False, repr=False)
    parameter_overrides: tuple[BatchSourceParameterOverrides, ...] = ()
    node_execution_overrides: tuple[BatchNodeExecutionOverride, ...] = ()

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
        if not isinstance(self.compute_request, ComputeRequest):
            raise TypeError("Batch config compute_request must be a ComputeRequest.")
        object.__setattr__(
            self,
            "parameter_overrides",
            normalize_batch_parameter_overrides(self.parameter_overrides),
        )
        object.__setattr__(
            self,
            "node_execution_overrides",
            normalize_batch_node_execution_overrides(self.node_execution_overrides),
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
        document: dict[str, object] = {
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
            "compute": self.compute_request.as_dict(),
        }
        # Keep the established document and hash byte-for-byte equivalent when
        # the explicit opt-in feature is unused.
        if self.parameter_overrides:
            document["parameter_overrides"] = batch_parameter_overrides_document(
                self.parameter_overrides
            )
        if self.node_execution_overrides:
            document["node_execution_overrides"] = (
                batch_node_execution_overrides_document(self.node_execution_overrides)
            )
        return document

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
            "compute",
        }
        if data.get("type") != BATCH_CONFIG_TYPE:
            raise ValueError("File is not a napari-vipp batch config.")
        raw_version = data.get("version")
        if type(raw_version) is not int or raw_version not in {
            1,
            2,
            3,
            4,
            BATCH_CONFIG_VERSION,
        }:
            raise ValueError(
                f"Unsupported batch config version: {raw_version!r}. "
                f"Expected version 1, 2, 3, 4, or {BATCH_CONFIG_VERSION}."
            )
        if raw_version == 1:
            allowed.remove("compute")
        if raw_version in {4, BATCH_CONFIG_VERSION}:
            allowed.add("parameter_overrides")
        if raw_version == BATCH_CONFIG_VERSION:
            allowed.add("node_execution_overrides")
        _reject_unknown_keys(data, allowed, "Batch config")
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
        if raw_version == 1:
            compute_request = ComputeRequest(mode=ComputeMode.CPU)
        else:
            compute_document = _require_object(
                data.get("compute"),
                "Batch config compute",
            )
            try:
                compute_request = ComputeRequest.from_dict(compute_document)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Batch config compute is invalid: {exc}") from exc
        return cls(
            workflow_file=Path(
                _required_text(workflow, "file", "batch config workflow")
            ),
            workflow_sha256=_required_text(workflow, "sha256", "batch config workflow"),
            output_dir=Path(_required_text(data, "output_dir", "batch config")),
            sources=tuple(
                BatchSourceConfig.from_dict(
                    item,
                    index=index,
                    config_version=raw_version,
                )
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
            compute_request=compute_request,
            parameter_overrides=(
                parse_batch_parameter_overrides(data["parameter_overrides"])
                if "parameter_overrides" in data
                else ()
            ),
            node_execution_overrides=(
                parse_batch_node_execution_overrides(data["node_execution_overrides"])
                if "node_execution_overrides" in data
                else ()
            ),
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
class _BatchSourceItem:
    """One deterministic selectable image within a matched source container."""

    path: Path
    series_index: int | None = None
    series_name: str = ""
    source_item: SourceItem | None = None


@dataclass(frozen=True)
class BatchItemPlan:
    index: int
    batch_id: str
    primary_source: Path
    source_paths: dict[str, Path]
    outputs: tuple[BatchOutputPlan, ...]
    source_series_indices: dict[str, int] = field(default_factory=dict)
    source_series_names: dict[str, str] = field(default_factory=dict)
    source_items: dict[str, SourceItem] = field(default_factory=dict)
    parameter_override_source_item_key: str = ""
    parameter_overrides: tuple[BatchParameterOverride, ...] = ()

    @property
    def source_item_documents(self) -> dict[str, dict[str, object]]:
        """Canonical SourceItem records keyed by Image Source node id."""

        return {
            node_id: source_item.to_dict()
            for node_id, source_item in self.source_items.items()
        }

    def source_label(self, node_id: str) -> str:
        path = self.source_paths[node_id]
        series_index = self.source_series_indices.get(node_id)
        if series_index is None:
            return path.name
        series_name = self.source_series_names.get(node_id) or (
            f"Series {series_index + 1}"
        )
        return f"{path.name} › {series_name}"


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
    provenance_status: str = "not_produced"
    execution_provenance_sha256: str = ""
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
            "provenance_status": self.provenance_status,
            "status": self.status.value,
        }
        if self.existing_identity:
            result["existing_identity"] = dict(self.existing_identity)
        if self.size_bytes is not None:
            result["size_bytes"] = self.size_bytes
        if self.execution_provenance_sha256:
            result["execution_provenance_sha256"] = self.execution_provenance_sha256
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
    execution: dict[str, object] = field(default_factory=dict)
    parameter_overrides: dict[str, object] = field(default_factory=dict)
    execution_provenance_sha256: str = ""
    error_type: str = ""
    error_message: str = ""
    effective_workflow_sha256: str = ""
    node_execution_overrides: dict[str, object] = field(default_factory=dict)

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
        if self.execution:
            result["execution"] = _json_safe(self.execution)
        if self.parameter_overrides:
            result["parameter_overrides"] = _json_safe(self.parameter_overrides)
        if self.effective_workflow_sha256:
            result["effective_workflow_sha256"] = self.effective_workflow_sha256
        if self.node_execution_overrides:
            result["node_execution_overrides"] = _json_safe(
                self.node_execution_overrides
            )
        if self.execution_provenance_sha256:
            result["execution_provenance_sha256"] = self.execution_provenance_sha256
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
    effective_config_sha256: str
    workflow_file: str
    config_file: str
    output_dir: str
    runtime: dict[str, object]
    workflow_document: dict[str, object]
    config_document: dict[str, object]
    compute: dict[str, object]
    items: tuple[BatchItemRecord, ...]
    profile_workflow_sha256: str = ""
    profile_workflow_document: dict[str, object] = field(default_factory=dict)
    node_execution_overrides: dict[str, object] = field(default_factory=dict)
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
                BatchStatus.CANCELLED,
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
                "effective_for_batch": {
                    "sha256": (self.profile_workflow_sha256 or self.workflow_sha256),
                    "scientific_graph": _json_safe(
                        self.profile_workflow_document or self.workflow_document
                    ),
                },
            },
            "config": {
                "file": self.config_file,
                "sha256": self.config_sha256,
                "effective_sha256": self.effective_config_sha256,
                "document": _json_safe(self.config_document),
            },
            "output_dir": self.output_dir,
            "runtime": _json_safe(self.runtime),
            "compute": _json_safe(self.compute),
            "summary": self.summary,
            "items": [item.to_dict() for item in self.items],
        }
        if self.node_execution_overrides:
            result["workflow"]["node_execution_overrides"] = _json_safe(
                self.node_execution_overrides
            )
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
            (bool(item.error_type) and item.status is not BatchStatus.CANCELLED)
            or item.status == BatchStatus.FAILED
            or any(output.status == BatchStatus.FAILED for output in item.outputs)
            for item in self.manifest.items
        ) or not bool(self.manifest.compute.get("runtime_cleanup_succeeded", True))

    @property
    def cancelled(self) -> bool:
        return any(item.status is BatchStatus.CANCELLED for item in self.manifest.items)

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
    restored = deserialize_workflow(data)
    nodes = sorted(
        (_canonical_scientific_node(item) for item in data["nodes"]),
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
    document: dict[str, object] = {
        "type": data.get("type"),
        "version": data.get("version"),
        "nodes": nodes,
        "connections": connections,
        "tunnels": tunnels,
    }
    source_item_bound = any(
        SOURCE_ITEM_PARAMETER
        in _require_object(node.get("params", {}), "Workflow node parameters")
        for node in nodes
        if node.get("operation_id") == "input"
    )
    bypass_authored = any(
        str(node.get("execution_mode", "run")).strip().casefold() == "bypass"
        for node in nodes
    )
    if data.get("version") in {4, 5, 6}:
        # Authored compute intent changes reproducibility and therefore belongs
        # in the scientific workflow hash. Machine-local benchmark evidence and
        # resolved devices are excluded by the portable workflow schema itself.
        execution = _canonical_compute_execution(restored["compute_request"])
        if bypass_authored:
            document["version"] = 6
            document["execution"] = execution
        elif source_item_bound:
            document["version"] = 5
            document["execution"] = execution
        elif execution == _implicit_v3_cpu_execution():
            # v3 already meant this exact CPU request. Preserve its established
            # scientific hash so attached batch configs remain valid after
            # lossless v3 -> v4 -> v5 save migrations.
            document["version"] = 3
        else:
            # A v5 document without SourceItem evidence has the same scientific
            # meaning as its v4 portable-compute representation.
            document["version"] = 4
            document["execution"] = execution
    return document


def scientific_workflow_hash(workflow: object) -> str:
    """Return a stable SHA-256 excluding layout, notes, and UI metadata."""
    return _document_hash(scientific_workflow_document(workflow))


def batch_config_hash(config: BatchConfig) -> str:
    return _document_hash(config.to_dict())


def effective_batch_compute_request(
    config: BatchConfig,
    override: ComputeRequest | None = None,
) -> ComputeRequest:
    """Resolve a run-scoped override without mutating durable config intent."""

    if not isinstance(config, BatchConfig):
        raise TypeError("config must be a BatchConfig.")
    if override is None:
        return config.compute_request
    if not isinstance(override, ComputeRequest):
        raise TypeError("compute_request override must be a ComputeRequest or None.")
    return override


def effective_batch_config_hash(
    config: BatchConfig,
    compute_request: ComputeRequest | None = None,
) -> str:
    """Hash the config as executed, including a non-mutating run override."""

    effective = effective_batch_compute_request(config, compute_request)
    document = config.to_dict()
    document["compute"] = effective.as_dict()
    return _document_hash(document)


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
        document["workflow"]["file"] = str(config.resolve_path(config.workflow_file))
        document["output_dir"] = str(config.resolve_path(config.output_dir))
        for source_document, source in zip(
            document["sources"], config.sources, strict=True
        ):
            source_document["input_dir"] = str(config.resolve_path(source.input_dir))
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


def _try_save_item_record(
    directory: Path,
    item: BatchItemRecord,
) -> OSError | None:
    """Persist a recoverable item checkpoint without aborting the whole run."""
    try:
        _save_item_record(directory, item)
    except OSError as exc:
        return exc
    return None


def _fully_skipped_item_record(
    item: BatchItemRecord,
    plan: BatchItemPlan,
) -> BatchItemRecord | None:
    """Return a terminal no-op record when every destination should be skipped."""
    if not plan.outputs or not all(
        output.existing_file_policy == ExistingFilePolicy.SKIP and output.path.exists()
        for output in plan.outputs
    ):
        return None
    started_at = _timestamp()
    outputs = tuple(
        replace(
            record,
            status=BatchStatus.SKIPPED,
            existing_identity=_path_identity(output.path),
            error_type="",
            error_message=(f"Existing destination was left unchanged: {output.path}"),
        )
        for record, output in zip(item.outputs, plan.outputs, strict=True)
    )
    return replace(
        item,
        outputs=outputs,
        status=BatchStatus.SKIPPED,
        started_at=started_at,
        finished_at=_timestamp(),
    )


def _with_item_record_write_failure(
    item: BatchItemRecord,
    error: OSError,
) -> BatchItemRecord:
    """Expose missing per-item provenance in the authoritative run manifest."""
    detail = (
        "Could not save the final per-item provenance record "
        f"({type(error).__name__}): {error}"
    )
    message = (
        f"{item.error_message} Additionally, {detail}" if item.error_message else detail
    )
    status = (
        item.status
        if item.status in {BatchStatus.PARTIAL, BatchStatus.FAILED}
        else BatchStatus.PARTIAL
    )
    return replace(
        item,
        status=status,
        error_type=item.error_type or type(error).__name__,
        error_message=message,
    )


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
    source_lists: dict[str, list[_BatchSourceItem]] = {}
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
        source_items = _expand_source_items(
            paths,
            axis_declaration=source.axis_declaration,
        )
        _verify_configured_source_items(source, source_items)
        source_lists[source.node_id] = source_items
        counts[source.title] = len(source_items)
    expected = len(next(iter(source_lists.values())))
    if any(len(paths) != expected for paths in source_lists.values()):
        summary = ", ".join(f"{title}={count}" for title, count in counts.items())
        raise ValueError(
            "Bound batch sources must contain the same number of image items "
            f"so they can be paired by sorted order ({summary})."
        )

    output_dir = config.resolve_path(config.output_dir)
    primary_id = config.sources[0].node_id
    items: list[BatchItemPlan] = []
    for item_index in range(expected):
        source_items = {
            source.node_id: source_lists[source.node_id][item_index]
            for source in config.sources
        }
        source_paths = {
            node_id: source_item.path for node_id, source_item in source_items.items()
        }
        source_series_indices = {
            node_id: source_item.series_index
            for node_id, source_item in source_items.items()
            if source_item.series_index is not None
        }
        source_series_names = {
            node_id: source_item.series_name
            for node_id, source_item in source_items.items()
            if source_item.series_index is not None
        }
        resolved_source_items = {
            node_id: source_item.source_item
            for node_id, source_item in source_items.items()
            if source_item.source_item is not None
        }
        primary_source = source_paths[primary_id]
        primary_item = source_items[primary_id]
        source_stem = _batch_source_item_stem(primary_item)
        source_name = (
            primary_source.name if primary_item.series_index is None else source_stem
        )
        batch_id = safe_batch_filename(f"{item_index + 1:04d}_{source_stem}")
        outputs = tuple(
            _plan_output(
                config,
                output_dir,
                output,
                item_index + 1,
                batch_id,
                source_stem,
                source_name,
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
                source_series_indices=source_series_indices,
                source_series_names=source_series_names,
                source_items=resolved_source_items,
            )
        )

    items = _bind_batch_parameter_overrides(
        config,
        items,
        primary_source_node_id=primary_id,
    )

    target_counts: dict[str, int] = {}
    for item in items:
        for output in item.outputs:
            key = os.path.normcase(str(output.path.resolve(strict=False)))
            target_counts[key] = target_counts.get(key, 0) + 1
    all_source_paths = tuple(
        path for item in items for path in item.source_paths.values()
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


def bind_batch_plan_source_items(
    config: BatchConfig,
    plan: BatchPlan,
) -> BatchConfig:
    """Freeze every resolved collection item into a replay-safe config."""

    if plan.config is not config:
        raise ValueError(
            "A batch plan can bind SourceItems only to the exact config used "
            "to create it."
        )
    bound_sources: list[BatchSourceConfig] = []
    for source in config.sources:
        source_items: list[SourceItem] = []
        for item in plan.items:
            source_item = item.source_items.get(source.node_id)
            if source_item is None:
                # Unreadable items have no scientific identity VIPP can invent.
                # A later readable replacement becomes an extra observed item
                # and is rejected by configured-set verification.
                continue
            source_items.append(source_item)
        bound_sources.append(replace(source, source_items=tuple(source_items)))
    return replace(config, sources=tuple(bound_sources))


def _bind_batch_parameter_overrides(
    config: BatchConfig,
    items: list[BatchItemPlan],
    *,
    primary_source_node_id: str,
) -> list[BatchItemPlan]:
    """Resolve opt-in override keys against exact planned SourceItems."""

    if not config.parameter_overrides:
        return items
    configured = {item.source_item_key: item for item in config.parameter_overrides}
    observed: dict[str, int] = {}
    resolved: list[BatchItemPlan] = []
    for item in items:
        source_item = item.source_items.get(primary_source_node_id)
        if source_item is None:
            raise ValueError(
                "Batch parameter overrides require canonical SourceItem "
                "evidence for every primary collection item. Refresh the "
                "collection before configuring per-sample values."
            )
        source_item_key = batch_source_item_override_key(
            primary_source_node_id,
            source_item,
        )
        if source_item_key in observed:
            first = observed[source_item_key]
            raise ValueError(
                "Batch primary SourceItem identity is not unique: items "
                f"{first} and {item.index} both resolve to {source_item_key}."
            )
        observed[source_item_key] = item.index
        override = configured.get(source_item_key)
        if override is None:
            resolved.append(item)
            continue
        resolved.append(
            replace(
                item,
                parameter_override_source_item_key=source_item_key,
                parameter_overrides=override.values,
            )
        )
    unmatched = sorted(set(configured) - set(observed))
    if unmatched:
        preview = ", ".join(unmatched[:3])
        suffix = "" if len(unmatched) <= 3 else f" (+{len(unmatched) - 3} more)"
        raise ValueError(
            "Batch parameter overrides reference source items that are not in "
            "the current primary collection: " + preview + suffix + ". "
            "The source content, logical item selection, or collection may "
            "have changed; refresh and review the mappings."
        )
    return resolved


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
            collision = (
                output.duplicate
                or output.input_collision
                or (
                    output.exists
                    and output.existing_file_policy == ExistingFilePolicy.ERROR
                )
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
    compute_request: ComputeRequest | None = None,
    cancel_event: threading.Event | None = None,
    progress_callback: Callable[[int, int, str, str], None] | None = None,
    execution_progress_callback: (
        Callable[[BatchExecutionProgress], None] | None
    ) = None,
    performance_history_path: str | Path | None = None,
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
        compute_request=compute_request,
        cancel_event=cancel_event,
        progress_callback=progress_callback,
        execution_progress_callback=execution_progress_callback,
        performance_history_path=performance_history_path,
    )


def run_batch(
    workflow: object,
    config: BatchConfig,
    *,
    workflow_path: str | Path | None = None,
    config_path: str | Path | None = None,
    plan: BatchPlan | None = None,
    compute_request: ComputeRequest | None = None,
    cancel_event: threading.Event | None = None,
    progress_callback: Callable[[int, int, str, str], None] | None = None,
    execution_progress_callback: (
        Callable[[BatchExecutionProgress], None] | None
    ) = None,
    compute_registry: ComputeRegistry | None = None,
    compute_planner: ComputePlanner | None = None,
    performance_history_path: str | Path | None = None,
) -> BatchRunResult:
    """Execute a deterministic batch plan with checkpointed provenance."""
    effective_request = effective_batch_compute_request(config, compute_request)
    workflow_sha256 = scientific_workflow_hash(workflow)
    _authored_pipeline, fixed_source_paths = _validated_batch_pipeline(
        workflow,
        config,
        workflow_sha256,
        workflow_path=workflow_path,
    )
    profile_workflow, pipeline = _effective_batch_pipeline(workflow, config)
    _validate_compute_request_node_ids(
        pipeline,
        effective_request,
        label="Effective batch compute request",
    )
    if plan is None:
        plan = build_batch_plan(config)
    elif plan.config is not config:
        raise ValueError(
            "A supplied batch plan must use the exact validated config instance."
        )
    plan = replace(
        plan,
        items=tuple(
            _bind_batch_parameter_overrides(
                config,
                [
                    replace(
                        item,
                        parameter_override_source_item_key="",
                        parameter_overrides=(),
                    )
                    for item in plan.items
                ],
                primary_source_node_id=config.sources[0].node_id,
            )
        ),
    )
    plan = _with_fixed_source_collisions(plan, fixed_source_paths.values())
    if plan.has_collisions:
        collisions = _collision_paths(plan)
        preview = ", ".join(collisions[:3])
        suffix = "" if len(collisions) <= 3 else f" (+{len(collisions) - 3} more)"
        raise FileExistsError(
            "Batch preflight found output collisions: " + preview + suffix
        )
    fixed_source_items = _preflight_representative_scientific_contract(
        pipeline,
        plan,
        config,
        fixed_source_paths,
    )
    # Resolve and deserialize every item-specific graph before creating any
    # run artifacts. Invalid substitutions therefore fail closed before the
    # first item can start or publish output.
    _validate_no_inert_parameter_overrides(pipeline, plan)
    item_workflows = {
        item.index: workflow_with_parameter_overrides(
            profile_workflow,
            item.parameter_overrides,
        )
        for item in plan.items
    }
    item_pipelines: dict[int, PrototypePipeline] = {}
    for index, document in item_workflows.items():
        restored_item = deserialize_workflow(document)
        item_pipeline = PrototypePipeline()
        item_pipeline.restore_graph(
            restored_item["nodes"],
            restored_item["connections"],
            restored_item.get("output_tunnels", ()),
            atomic_bypass_profile=bool(config.node_execution_overrides),
        )
        item_pipelines[index] = item_pipeline
    item_workflow_hashes = {
        index: scientific_workflow_hash(document)
        for index, document in item_workflows.items()
    }
    profile_workflow_sha256 = scientific_workflow_hash(profile_workflow)
    execution_override_record = node_execution_override_provenance(
        config.node_execution_overrides,
        workflow,
        profile_workflow,
    )
    plan.output_dir.mkdir(parents=True, exist_ok=True)

    workflow_label = str(workflow_path or config.workflow_file)
    config_label = str(config_path or BATCH_CONFIG_FILENAME)
    manifest_path = plan.output_dir / BATCH_MANIFEST_FILENAME
    manifest = _seed_manifest(
        plan,
        workflow_sha256,
        batch_config_hash(config),
        effective_batch_config_hash(config, effective_request),
        workflow_label,
        config_label,
        scientific_workflow_document(workflow),
        profile_workflow_sha256,
        scientific_workflow_document(profile_workflow),
        item_workflow_hashes,
        execution_override_record,
        config.to_dict(),
        fixed_source_paths,
        fixed_source_items,
        effective_request,
        override_used=compute_request is not None,
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
    active_registry = compute_registry
    owns_registry = False
    runtime_cleanup_error: Exception | None = None
    try:
        for item_position, item_plan in enumerate(plan.items):
            if _cancel_requested(cancel_event):
                manifest = _cancel_from_item(
                    manifest,
                    start_index=item_position,
                    reason="Batch cancelled before this item started.",
                    compute_request=effective_request,
                )
                for cancelled_item in manifest.items[item_position:]:
                    _try_save_item_record(item_records_dir, cancelled_item)
                break

            item_record = manifest.items[item_position]
            skipped_record = _fully_skipped_item_record(item_record, item_plan)
            if skipped_record is not None:
                item_record = skipped_record
                manifest = manifest.replace_item(item_record)
                _report_progress(
                    progress_callback,
                    item_plan.index,
                    total,
                    item_plan.batch_id,
                    "running",
                )
                record_error = _try_save_item_record(item_records_dir, item_record)
                if record_error is not None:
                    item_record = _with_item_record_write_failure(
                        item_record,
                        record_error,
                    )
                    manifest = manifest.replace_item(item_record)
                _report_progress(
                    progress_callback,
                    item_plan.index,
                    total,
                    item_plan.batch_id,
                    item_record.status.value,
                )
                if item_record.error_type and not config.continue_on_error:
                    manifest = _skip_remaining_items(
                        manifest,
                        start_index=item_position + 1,
                        reason="Not run because continue_on_error is disabled.",
                    )
                    for remaining_item in manifest.items[item_position + 1 :]:
                        _try_save_item_record(item_records_dir, remaining_item)
                    break
                continue

            item_record = replace(
                item_record,
                status=BatchStatus.RUNNING,
                started_at=_timestamp(),
            )
            manifest = manifest.replace_item(item_record)
            _try_save_item_record(item_records_dir, item_record)
            _report_progress(
                progress_callback,
                item_plan.index,
                total,
                item_plan.batch_id,
                "running",
            )

            item_error: Exception | None = None
            item_cancelled = False
            publication_blocked = False
            sources = list(item_record.sources)
            source_paths: dict[str, Path] = {}
            source_identities: dict[str, LocalSourceIdentity] = {}
            item_pipeline = item_pipelines[item_plan.index]
            (
                node_started,
                node_finished,
                operation_progress,
                completed_node_ids,
            ) = _batch_execution_callbacks(
                item_index=item_plan.index,
                item_total=total,
                batch_id=item_plan.batch_id,
                pipeline=pipeline,
                callback=execution_progress_callback,
            )
            try:
                source_paths = _item_source_paths(
                    pipeline,
                    item_plan,
                    fixed_source_paths,
                )
                source_identities = _capture_item_source_identities(
                    source_paths,
                    cancel_event=cancel_event,
                    progress_callback=execution_progress_callback,
                    item_index=item_plan.index,
                    item_total=total,
                    batch_id=item_plan.batch_id,
                )
                payloads, sources = _source_payloads_for_item(
                    item_pipeline,
                    item_plan,
                    config,
                    source_paths,
                    source_identities,
                    fixed_source_items,
                )
                if (
                    effective_request.mode is not ComputeMode.CPU
                    and active_registry is None
                ):
                    from napari_vipp.core.compute_registry import ComputeRegistry

                    active_registry = ComputeRegistry()
                    owns_registry = True

                execution_result = execute_pipeline_request(
                    PipelineRunRequest(
                        run_id=item_plan.index,
                        workflow=item_workflows[item_plan.index],
                        input_data=None,
                        input_metadata=None,
                        input_name="",
                        source_payloads=payloads,
                        compute_request=effective_request,
                        manual_node_ids=frozenset(pipeline.manual_node_ids()),
                        target_node_ids=frozenset(output_node_ids),
                        retain_node_ids=frozenset(output_node_ids),
                        prune_unretained=True,
                        cancel_event=cancel_event,
                        performance_history_path=performance_history_path,
                        atomic_bypass_profile=bool(
                            config.node_execution_overrides
                        ),
                    ),
                    node_started_callback=node_started,
                    node_finished_callback=node_finished,
                    progress_callback=operation_progress,
                    compute_registry=active_registry,
                    compute_planner=compute_planner,
                )
                if execution_result.pipeline is not None:
                    item_pipeline = execution_result.pipeline
                execution = serialize_execution_provenance(
                    effective_request,
                    execution_result.pipeline,
                    execution_result.execution_report,
                    completed_node_ids=completed_node_ids,
                    implementation_specs=(
                        ()
                        if active_registry is None
                        else active_registry.implementation_specs
                    ),
                    failure=execution_result.failure,
                )
                if item_record.parameter_overrides:
                    execution["parameter_overrides"] = _json_safe(
                        item_record.parameter_overrides
                    )
                _attach_item_workflow_provenance(execution, item_record)
                execution["status"] = execution["outcome"]
                provenance_sha256 = execution_provenance_digest(execution)
                item_record = replace(
                    item_record,
                    sources=tuple(sources),
                    execution=execution,
                    execution_provenance_sha256=provenance_sha256,
                )
                manifest = manifest.replace_item(item_record)
                # Checkpoint actual decisions or typed terminal failure before
                # any output staging or source-dependent publication begins.
                _try_save_item_record(item_records_dir, item_record)
                if execution.get("cleanup_succeeded") is not True:
                    publication_blocked = True
                if execution_result.cancelled:
                    raise OperationCancelled(
                        execution_result.error or "Batch execution cancelled."
                    )
                if execution_result.error:
                    raise BatchExecutionError(execution_result.error)
                if (
                    execution_result.execution_report is not None
                    and not execution_result.execution_report.cleanup_succeeded
                ):
                    publication_blocked = True
                    raise BatchRuntimeCleanupError(
                        "The accelerator runtime did not clean up completely; "
                        "outputs were not published."
                    )
            except OperationCancelled as exc:
                item_cancelled = True
                item_error = exc
                item_record = replace(item_record, sources=tuple(sources))
            except Exception as exc:
                item_error = exc
                item_record = replace(item_record, sources=tuple(sources))
            finally:
                output_records = list(item_record.outputs)
                staged_outputs: dict[int, _StagedBatchOutput] = {}
                if _cancel_requested(cancel_event):
                    item_cancelled = True
                    if item_error is None:
                        item_error = OperationCancelled("Batch execution cancelled.")

                if publication_blocked:
                    output_records = [
                        replace(
                            output_record,
                            status=BatchStatus.FAILED,
                            size_bytes=None,
                            overwrote_existing=False,
                            error_type="BatchRuntimeCleanupError",
                            error_message=str(item_error),
                        )
                        for output_record in output_records
                    ]
                elif item_cancelled:
                    output_records = [
                        replace(
                            output_record,
                            status=BatchStatus.CANCELLED,
                            error_type="OperationCancelled",
                            error_message=str(item_error or "Batch cancelled."),
                        )
                        for output_record in output_records
                    ]
                else:
                    # Fully stage every available branch first. This forces lazy
                    # arrays to finish reading without publishing an output.
                    for output_index, output_plan in enumerate(item_plan.outputs):
                        _report_batch_phase(
                            execution_progress_callback,
                            item_index=item_plan.index,
                            item_total=total,
                            batch_id=item_plan.batch_id,
                            node_id=output_plan.node_id,
                            operation_id="batch_stage_output",
                            current=output_index,
                            total=len(item_plan.outputs),
                            message=(
                                f"Staging output {output_index + 1}/"
                                f"{len(item_plan.outputs)} privately. The writer "
                                "may finish its current file before cancellation "
                                "is observed."
                            ),
                        )
                        output_record = output_records[output_index]
                        output_checkpoint_changed = False
                        if (
                            item_error is not None
                            and item_pipeline.outputs.get(output_plan.node_id) is None
                        ):
                            output_record = replace(
                                output_record,
                                status=BatchStatus.FAILED,
                                error_type=type(item_error).__name__,
                                error_message=str(item_error),
                            )
                            output_checkpoint_changed = True
                        else:
                            try:
                                staged = _save_planned_output(
                                    item_pipeline,
                                    output_plan,
                                )
                            except _SkippedOutput as exc:
                                output_record = replace(
                                    output_record,
                                    status=BatchStatus.SKIPPED,
                                    error_type="",
                                    error_message=str(exc),
                                )
                                output_checkpoint_changed = True
                            except Exception as exc:
                                output_record = replace(
                                    output_record,
                                    status=BatchStatus.FAILED,
                                    error_type=type(exc).__name__,
                                    error_message=str(exc),
                                )
                                output_checkpoint_changed = True
                            else:
                                staged_outputs[output_index] = staged
                        output_records[output_index] = output_record
                        running_record = replace(
                            item_record,
                            sources=tuple(sources),
                            outputs=tuple(output_records),
                        )
                        manifest = manifest.replace_item(running_record)
                        if output_checkpoint_changed:
                            _try_save_item_record(item_records_dir, running_record)
                        _report_batch_phase(
                            execution_progress_callback,
                            item_index=item_plan.index,
                            item_total=total,
                            batch_id=item_plan.batch_id,
                            node_id=output_plan.node_id,
                            operation_id="batch_stage_output",
                            current=output_index + 1,
                            total=len(item_plan.outputs),
                            message=(
                                f"Staged output {output_index + 1}/"
                                f"{len(item_plan.outputs)}."
                            ),
                        )
                        if _cancel_requested(cancel_event):
                            item_cancelled = True
                            item_error = OperationCancelled(
                                "Batch cancelled while staging private outputs."
                            )
                            break

                if item_cancelled and not publication_blocked:
                    for staged in staged_outputs.values():
                        _cleanup_staged_output(staged)
                    staged_outputs.clear()
                    output_records = [
                        output_record
                        if output_record.status is BatchStatus.SKIPPED
                        else replace(
                            output_record,
                            status=BatchStatus.CANCELLED,
                            size_bytes=None,
                            overwrote_existing=False,
                            error_type="OperationCancelled",
                            error_message=str(item_error or "Batch cancelled."),
                        )
                        for output_record in output_records
                    ]

                source_change_error: SourceChangedError | None = None
                if not item_cancelled and not publication_blocked and source_identities:
                    try:
                        _verify_item_source_identities(
                            source_paths,
                            source_identities,
                            cancel_event=cancel_event,
                            progress_callback=execution_progress_callback,
                            item_index=item_plan.index,
                            item_total=total,
                            batch_id=item_plan.batch_id,
                        )
                    except OperationCancelled as exc:
                        item_cancelled = True
                        item_error = exc
                    except SourceChangedError as exc:
                        source_change_error = exc
                        item_error = exc

                if item_cancelled:
                    for staged in staged_outputs.values():
                        _cleanup_staged_output(staged)
                    staged_outputs.clear()
                    output_records = [
                        output_record
                        if output_record.status is BatchStatus.SKIPPED
                        else replace(
                            output_record,
                            status=BatchStatus.CANCELLED,
                            size_bytes=None,
                            overwrote_existing=False,
                            error_type="OperationCancelled",
                            error_message=str(item_error or "Batch cancelled."),
                        )
                        for output_record in output_records
                    ]

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
                elif not item_cancelled and not publication_blocked:
                    # All source-dependent bytes are private and stable. Do not
                    # observe cancellation once multi-output atomic promotion
                    # begins: stopping there would create an avoidable partial
                    # publication set.
                    staged_items = tuple(staged_outputs.items())
                    for staged_position, (output_index, staged) in enumerate(
                        staged_items
                    ):
                        output_record = output_records[output_index]
                        _report_batch_phase(
                            execution_progress_callback,
                            item_index=item_plan.index,
                            item_total=total,
                            batch_id=item_plan.batch_id,
                            node_id=staged.plan.node_id,
                            operation_id="batch_publish_output",
                            current=staged_position,
                            total=len(staged_items),
                            message=(
                                f"Publishing output {staged_position + 1}/"
                                f"{len(staged_items)} atomically."
                            ),
                        )
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
                                provenance_status="produced",
                                execution_provenance_sha256=(
                                    item_record.execution_provenance_sha256
                                ),
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
                        if staged_position + 1 < len(staged_items):
                            _try_save_item_record(item_records_dir, running_record)
                        _report_batch_phase(
                            execution_progress_callback,
                            item_index=item_plan.index,
                            item_total=total,
                            batch_id=item_plan.batch_id,
                            node_id=staged.plan.node_id,
                            operation_id="batch_publish_output",
                            current=staged_position + 1,
                            total=len(staged_items),
                            message=(
                                f"Published output {staged_position + 1}/"
                                f"{len(staged_items)}."
                            ),
                        )
                item_record = replace(
                    item_record,
                    sources=tuple(sources),
                    outputs=tuple(output_records),
                )
                item_pipeline.prune_cached_outputs(())

            if item_error is not None and not item_record.execution:
                item_record = _with_synthetic_execution_failure(
                    item_record,
                    effective_request,
                    item_error,
                    cancelled=item_cancelled,
                )
            if publication_blocked:
                item_record = replace(
                    item_record,
                    status=BatchStatus.FAILED,
                    error_type="BatchRuntimeCleanupError",
                    error_message=str(item_error),
                )
            elif item_cancelled:
                item_record = replace(
                    item_record,
                    status=BatchStatus.CANCELLED,
                    error_type="OperationCancelled",
                    error_message=str(item_error or "Batch cancelled."),
                )
            elif item_error is not None:
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
            record_error = _try_save_item_record(item_records_dir, item_record)
            if record_error is not None:
                item_record = _with_item_record_write_failure(
                    item_record,
                    record_error,
                )
                manifest = manifest.replace_item(item_record)
            _report_progress(
                progress_callback,
                item_plan.index,
                total,
                item_plan.batch_id,
                item_record.status.value,
            )

            if item_cancelled and not publication_blocked:
                manifest = _skip_remaining_items(
                    manifest,
                    start_index=item_position + 1,
                    reason="Not run because the batch was cancelled.",
                )
                for skipped_item in manifest.items[item_position + 1 :]:
                    _try_save_item_record(item_records_dir, skipped_item)
                break

            if publication_blocked:
                manifest = _skip_remaining_items(
                    manifest,
                    start_index=item_position + 1,
                    reason=(
                        "Not run because accelerator cleanup failed and the "
                        "runtime is no longer trusted."
                    ),
                )
                for skipped_item in manifest.items[item_position + 1 :]:
                    _try_save_item_record(item_records_dir, skipped_item)
                break

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
                    _try_save_item_record(item_records_dir, skipped_item)
                break
    finally:
        if owns_registry and active_registry is not None:
            try:
                active_registry.close()
            except Exception as exc:
                runtime_cleanup_error = exc

    compute_record = dict(manifest.compute)
    actual_summary = _batch_actual_compute_summary(manifest.items)
    compute_record["actual_summary"] = actual_summary
    if not actual_summary["item_cleanup_succeeded"]:
        compute_record["runtime_cleanup_succeeded"] = False
    compute_record["item_execution_provenance"] = [
        {
            "index": item.index,
            "batch_id": item.batch_id,
            "sha256": item.execution_provenance_sha256,
        }
        for item in manifest.items
        if item.execution_provenance_sha256
    ]
    if runtime_cleanup_error is not None:
        compute_record["runtime_cleanup_succeeded"] = False
        warnings = list(compute_record.get("warnings", []))
        warnings.append(
            "The batch accelerator runtime did not close cleanly: "
            f"{type(runtime_cleanup_error).__name__}: {runtime_cleanup_error}"
        )
        compute_record["warnings"] = warnings
    manifest = replace(
        manifest,
        compute=compute_record,
        finished_at=_timestamp(),
    )
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


def _report_execution_progress(
    callback: Callable[[BatchExecutionProgress], None] | None,
    update: BatchExecutionProgress,
) -> None:
    if callback is None:
        return
    try:
        callback(update)
    except Exception:
        # Presentation hooks must not invalidate computation or publication.
        return


def _report_batch_phase(
    callback: Callable[[BatchExecutionProgress], None] | None,
    *,
    item_index: int,
    item_total: int,
    batch_id: str,
    node_id: str,
    operation_id: str,
    current: int,
    total: int,
    message: str,
) -> None:
    _report_execution_progress(
        callback,
        BatchExecutionProgress(
            item_index=item_index,
            item_total=item_total,
            batch_id=batch_id,
            node_id=node_id,
            operation_id=operation_id,
            current=current,
            total=total,
            message=message,
        ),
    )


def _batch_execution_callbacks(
    *,
    item_index: int,
    item_total: int,
    batch_id: str,
    pipeline: PrototypePipeline,
    callback: Callable[[BatchExecutionProgress], None] | None,
) -> tuple[
    Callable[[str], None],
    Callable[[Any], None],
    Callable[[str, int, int, str], None],
    list[str],
]:
    """Build item-bound callbacks without retaining loop variables."""

    state = {"node_id": ""}
    completed_node_ids: list[str] = []

    def node_started(node_id: str) -> None:
        state["node_id"] = str(node_id)
        node = pipeline.nodes.get(state["node_id"])
        _report_execution_progress(
            callback,
            BatchExecutionProgress(
                item_index=item_index,
                item_total=item_total,
                batch_id=batch_id,
                node_id=state["node_id"],
                operation_id="" if node is None else str(node.operation_id),
                current=0,
                total=0,
                message="Node started.",
            ),
        )

    def node_finished(result: Any) -> None:
        completed_node_ids.append(str(result.node_id))
        _report_execution_progress(
            callback,
            BatchExecutionProgress(
                item_index=item_index,
                item_total=item_total,
                batch_id=batch_id,
                node_id=str(result.node_id),
                operation_id=str(result.operation_id),
                current=1,
                total=1,
                message="Node completed.",
            ),
        )
        state["node_id"] = ""

    def operation_progress(
        operation_id: str,
        current: int,
        operation_total: int,
        message: str,
    ) -> None:
        _report_execution_progress(
            callback,
            BatchExecutionProgress(
                item_index=item_index,
                item_total=item_total,
                batch_id=batch_id,
                node_id=state["node_id"],
                operation_id=str(operation_id),
                current=int(current),
                total=int(operation_total),
                message=str(message),
            ),
        )

    return node_started, node_finished, operation_progress, completed_node_ids


def _cancel_requested(cancel_event: threading.Event | None) -> bool:
    return cancel_event is not None and cancel_event.is_set()


def preflight_batch(
    workflow: object,
    config: BatchConfig,
    *,
    workflow_path: str | Path | None = None,
    allow_collisions: bool = False,
) -> BatchPlan:
    """Validate and plan a batch, raising before any artifact is modified."""
    workflow_sha256 = scientific_workflow_hash(workflow)
    _authored_pipeline, fixed_source_paths = _validated_batch_pipeline(
        workflow,
        config,
        workflow_sha256,
        workflow_path=workflow_path,
    )
    _effective_workflow, pipeline = _effective_batch_pipeline(workflow, config)
    plan = _with_fixed_source_collisions(
        build_batch_plan(config),
        fixed_source_paths.values(),
    )
    _validate_no_inert_parameter_overrides(pipeline, plan)
    if plan.has_collisions and not allow_collisions:
        collisions = _collision_paths(plan)
        preview = ", ".join(collisions[:3])
        suffix = "" if len(collisions) <= 3 else f" (+{len(collisions) - 3} more)"
        raise FileExistsError(
            "Batch preflight found output collisions: " + preview + suffix
        )
    _preflight_representative_scientific_contract(
        pipeline,
        plan,
        config,
        fixed_source_paths,
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
    # The graph and the durable batch request are validated independently.  A
    # run-scoped override may change execution intent without mutating either
    # the workflow document or attached batch configuration.
    pipeline.restore_graph(
        restored["nodes"],
        restored["connections"],
        restored.get("output_tunnels", ()),
    )
    _validate_compute_request_node_ids(
        pipeline,
        config.compute_request,
        label="Batch config compute request",
    )
    validate_batch_node_execution_overrides(
        config.node_execution_overrides,
        pipeline,
    )
    validate_batch_parameter_overrides(config.parameter_overrides, pipeline)
    fixed_source_paths = _validate_pipeline_config(
        pipeline,
        config,
        workflow_path=workflow_path,
    )
    _effective_workflow, effective_pipeline = _effective_batch_pipeline(
        workflow,
        config,
    )
    _validate_effective_batch_output_contract(effective_pipeline, config)
    return pipeline, fixed_source_paths


def _effective_batch_pipeline(
    workflow: object,
    config: BatchConfig,
) -> tuple[dict[str, object], PrototypePipeline]:
    """Resolve one detached whole-batch graph without editing authored intent."""

    effective_workflow = workflow_with_node_execution_overrides(
        workflow,
        config.node_execution_overrides,
    )
    restored = deserialize_workflow(effective_workflow)
    pipeline = PrototypePipeline()
    pipeline.restore_graph(
        restored["nodes"],
        restored["connections"],
        restored.get("output_tunnels", ()),
        atomic_bypass_profile=bool(config.node_execution_overrides),
    )
    return effective_workflow, pipeline


def _validate_effective_batch_output_contract(
    pipeline: PrototypePipeline,
    config: BatchConfig,
) -> None:
    """Fail before planning when a run-scoped profile changes saved data kind."""

    for output in config.outputs:
        node = pipeline.nodes.get(output.node_id)
        if node is None:
            raise ValueError(
                f"Effective batch output {output.node_id!r} is missing."
            )
        ports = pipeline.output_ports(output.node_id)
        if not ports:
            raise ValueError(
                f"Effective batch output {output.node_id!r} has no output port."
            )
        effective_type = ports[0].output_type
        effective_kind = "table" if effective_type == "table" else "image"
        if output.kind == effective_kind:
            continue
        resolved_format = _resolved_output_format(config, output)
        raise ValueError(
            f"Effective Safe Node Bypass changes Batch Output "
            f"{output.node_id!r} to {effective_type} ({effective_kind}) data, "
            f"but the saved batch declaration expects {output.kind} data in "
            f"{resolved_format!r} format. Disable or revise the batch bypass "
            "profile, or author a matching Batch Output declaration before "
            "running."
        )


def _validate_no_inert_parameter_overrides(
    pipeline: PrototypePipeline,
    plan: BatchPlan,
) -> None:
    bypassed_node_ids = {
        node_id for node_id in pipeline.nodes if pipeline.node_is_bypassed(node_id)
    }
    inert = sorted(
        {
            override.node_id
            for item in plan.items
            for override in item.parameter_overrides
            if override.node_id in bypassed_node_ids
        }
    )
    if inert:
        raise ValueError(
            "Per-sample parameter overrides cannot target nodes that are "
            "effectively bypassed: " + ", ".join(inert) + "."
        )


def _validate_compute_request_node_ids(
    pipeline: PrototypePipeline,
    request: ComputeRequest,
    *,
    label: str,
) -> None:
    unknown = set(request.node_preferences) - set(pipeline.nodes)
    if unknown:
        raise ValueError(
            f"{label} references missing node IDs: " + ", ".join(sorted(unknown)) + "."
        )


def _plan_output(
    config: BatchConfig,
    output_dir: Path,
    output: BatchOutputConfig,
    index: int,
    batch_id: str,
    source_stem: str,
    source_name: str,
) -> BatchOutputPlan:
    values = {
        "source_stem": source_stem,
        "tag": safe_batch_filename(output.tag),
        "node_id": safe_batch_filename(output.node_id),
        "node_title": safe_batch_filename(output.node_title),
        "batch_id": batch_id,
        "batch_index": f"{index:04d}",
        "source_name": safe_batch_filename(source_name),
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
            "Batch Output nodes instead: " + ", ".join(enabled_save_nodes) + "."
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
                "desired port: " + ", ".join(multi_output_terminals) + "."
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


def _preflight_representative_scientific_contract(
    pipeline: PrototypePipeline,
    plan: BatchPlan,
    config: BatchConfig,
    fixed_source_paths: dict[str, Path],
) -> dict[str, SourceItem]:
    """Validate one representative's axis contract before any run artifacts."""
    if not plan.items:
        return {}
    item = next(
        (
            planned_item
            for planned_item in plan.items
            if any(
                output.existing_file_policy is not ExistingFilePolicy.SKIP
                or not output.path.exists()
                for output in planned_item.outputs
            )
        ),
        None,
    )
    if item is None:
        return {}
    source_paths = _item_source_paths(pipeline, item, fixed_source_paths)
    bindings = {source.node_id: source for source in config.sources}
    payloads: dict[str, SourcePayload] = {}
    summaries: list[str] = []
    generic_undeclared: list[tuple[str, str, str, str]] = []
    fixed_source_items: dict[str, SourceItem] = {}
    contract_pipeline = PrototypePipeline()
    contract_pipeline.restore_graph(
        pipeline.nodes.values(),
        pipeline.connections,
        pipeline.output_tunnels.values(),
    )
    for node_id, node in contract_pipeline.nodes.items():
        if node.operation_id != "input":
            continue
        path = source_paths[node_id]
        binding = bindings.get(node_id)
        title = binding.title if binding is not None else node.title
        try:
            inspection = inspect_image_source(path)
        except Exception:
            # A source that cannot be inspected at all remains an item-specific
            # read failure governed by continue_on_error.
            return fixed_source_items
        try:
            expected_source_item = (
                item.source_items.get(node_id)
                if binding is not None
                else source_item_from_params(node.params)
            )
            series_index = item.source_series_indices.get(
                node_id,
                int(node.params.get("series_index", 0)),
            )
            selected = select_inspected_item(
                inspection,
                series_index=(
                    None if expected_source_item is not None else series_index
                ),
                item_key=(
                    expected_source_item.selector.key
                    if expected_source_item is not None
                    else None
                ),
            )
            raw_state = inspect_image_state(
                path,
                inspection=inspection,
                series_index=selected.index,
            )
            base = np.empty((1,), dtype=np.dtype(selected.dtype))
            proxy = np.lib.stride_tricks.as_strided(
                base,
                shape=selected.shape,
                strides=(0,) * len(selected.shape),
                writeable=False,
            )
        except Exception as exc:
            summaries.append(f"{title}: metadata contract unavailable")
            _raise_batch_scientific_preflight_error(
                exc,
                summaries,
                generic_undeclared,
            )
        declaration = (
            AxisDeclaration.from_value(node.params.get("axis_declaration"))
            if binding is None
            else binding.axis_declaration
        )
        try:
            effective_state, axis_semantics = _declared_batch_source_state(
                raw_state,
                binding,
                image_source_declaration=node.params.get("axis_declaration"),
            )
            bundle = capture_local_source_bundle(
                path,
                source_format=inspection.format,
            )
            if expected_source_item is None:
                source_item = resolve_source_item(
                    bundle,
                    inspection,
                    item_key=selected.key,
                    image_state=effective_state,
                    axis_declaration=declaration,
                )
            else:
                source_item = verify_saved_source_item(
                    expected_source_item,
                    bundle,
                    inspection,
                    image_state=effective_state,
                )
            if binding is None:
                fixed_source_items[node_id] = source_item
            summaries.append(
                f"{title}: raw {axis_semantics['raw_axes']}, effective "
                f"{axis_semantics['effective_axes']}"
                + (
                    " (no declaration)"
                    if declaration is None
                    else f" ({declaration.display_text})"
                )
            )
            if (
                binding is not None
                and declaration is None
                and any(axis.type in {"unknown", "custom"} for axis in raw_state.axes)
            ):
                generic_undeclared.append(
                    (node_id, title, raw_state.axis_order, inspection.format)
                )
        except Exception as exc:
            summaries.append(
                f"{title}: raw {raw_state.axis_order}"
                + (
                    " (no declaration)"
                    if declaration is None
                    else f" ({declaration.display_text})"
                )
            )
            _raise_batch_scientific_preflight_error(
                exc,
                summaries,
                generic_undeclared,
            )
        payloads[node_id] = SourcePayload(
            proxy,
            {
                "vipp_source_path": str(path),
                "vipp_axis_semantics": axis_semantics,
            },
            selected.name or path.name,
            effective_state,
            axis_semantics_resolved=True,
            source_item=source_item,
        )
    try:
        contract_pipeline.preflight_axis_contract(payloads)
    except Exception as exc:
        _raise_batch_scientific_preflight_error(
            exc,
            summaries,
            generic_undeclared,
            contract_pipeline,
        )
    return fixed_source_items


def _raise_batch_scientific_preflight_error(
    exc: Exception,
    summaries: list[str],
    generic_undeclared: list[tuple[str, str, str, str]],
    pipeline: PrototypePipeline | None = None,
) -> None:
    if isinstance(exc, BatchScientificPreflightError):
        raise exc
    source_summary = "; ".join(summaries) or "unavailable"
    guidance = ""
    axis_suggestion = None
    user_message = "The batch images do not match this workflow."
    candidates = list(generic_undeclared)
    if (
        pipeline is not None
        and isinstance(exc, AmbiguousAxisError)
        and exc.failing_node_id
    ):
        primary_sources = _primary_source_node_ids(
            pipeline,
            exc.failing_node_id,
        )
        candidates = [item for item in candidates if item[0] in primary_sources]
    if candidates:
        node_id, title, raw_axes, _source_format = candidates[0]
        guidance = (
            f" Review '{title}' and, only if it is truly a Z stack, set "
            f"Declare axes to {raw_axes} -> ZYX."
        )
        if (
            len(candidates) == 1
            and raw_axes == "QYX"
            and isinstance(exc, AmbiguousAxisError)
            and exc.code == "positional_spatial_layout"
            and exc.detected_axes == "QYX"
            and exc.required_axes == "ZYX"
        ):
            axis_suggestion = BatchAxisSuggestion(
                source_node_id=node_id,
                source_title=title,
                declaration=AxisDeclaration("QYX", "ZYX"),
            )
            user_message = (
                "This source has an unknown leading Q axis. Because the "
                "workflow requires 3D processing, VIPP can treat Q as depth Z."
            )
        else:
            user_message = (
                "This workflow needs image-axis information that the source "
                "does not provide."
            )
    raise BatchScientificPreflightError(
        "Batch scientific preflight failed before item processing, output "
        "creation, or CPU/GPU device setup. Representative source axes: "
        f"{source_summary}.{guidance} {exc}",
        user_message=user_message,
        axis_suggestion=axis_suggestion,
    ) from exc


def _primary_source_node_ids(
    pipeline: PrototypePipeline,
    failing_node_id: str,
) -> set[str]:
    """Trace the axis-bearing primary input back to its workflow source."""
    current = str(failing_node_id)
    visited: set[str] = set()
    while current and current not in visited:
        visited.add(current)
        node = pipeline.nodes.get(current)
        if node is None:
            return set()
        if node.operation_id == "input":
            return {current}
        connections = sorted(
            (
                connection
                for connection in pipeline.connections
                if connection.target_id == current
            ),
            key=lambda connection: connection.target_port,
        )
        if not connections:
            return set()
        current = connections[0].source_id
    return set()


def _source_payloads_for_item(
    pipeline: PrototypePipeline,
    item: BatchItemPlan,
    config: BatchConfig,
    source_paths: dict[str, Path],
    source_identities: dict[str, LocalSourceIdentity],
    fixed_source_items: dict[str, SourceItem],
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
        expected_source_item = (
            item.source_items.get(node_id)
            if role == "collection"
            else fixed_source_items.get(node_id) or source_item_from_params(node.params)
        )
        legacy_series_index = item.source_series_indices.get(
            node_id,
            int(node.params.get("series_index", 0)),
        )
        inspection = inspect_image_source(path)
        selected = select_inspected_item(
            inspection,
            series_index=(
                None if expected_source_item is not None else legacy_series_index
            ),
            item_key=(
                expected_source_item.selector.key
                if expected_source_item is not None
                else None
            ),
        )
        raw_state = inspect_image_state(
            path,
            inspection=inspection,
            series_index=selected.index,
        )
        effective_state, axis_semantics = _declared_batch_source_state(
            raw_state,
            binding,
            image_source_declaration=node.params.get("axis_declaration"),
        )
        identity = source_identities[node_id]
        if expected_source_item is None:
            bundle = capture_local_source_bundle(
                path,
                source_format=inspection.format,
            )
            if local_source_identity_from_bundle(bundle) != identity:
                raise SourceChangedError(
                    "The batch source changed while its logical item was being "
                    "resolved. No output was published."
                )
            source_item = resolve_source_item(
                bundle,
                inspection,
                item_key=selected.key,
                image_state=effective_state,
                axis_declaration=(
                    binding.axis_declaration
                    if binding is not None
                    else node.params.get("axis_declaration")
                ),
            )
        else:
            if (
                local_source_identity_from_bundle(expected_source_item.container)
                != identity
            ):
                raise SourceChangedError(
                    "The batch source revision no longer matches its planned "
                    "SourceItem. Refresh the batch plan and review the changed "
                    "source before running."
                )
            source_item = verify_saved_source_item(
                expected_source_item,
                replace(
                    expected_source_item.container,
                    uri=str(path.resolve(strict=False)),
                ),
                inspection,
                image_state=effective_state,
            )

        window_decision = plan_exact_source_crop_window(
            pipeline,
            node_id,
            source_item,
            effective_state,
        )
        read_strategy = "full-source"
        source_window: dict[str, object] | None = None
        source_window_digest = ""
        if window_decision.plan is not None:
            declaration = (
                binding.axis_declaration
                if binding is not None
                else node.params.get("axis_declaration")
            )
            snapshot = load_frozen_file_source_snapshot(
                path,
                series_index=selected.index,
                item_key=selected.key,
                axis_declaration=declaration,
                expected_identity=identity,
                expected_source_item=source_item,
                exact_window_request=window_decision.plan.request,
            )
            if snapshot.source_item.digest != source_item.digest:
                raise RuntimeError(
                    "Exact batch source-window snapshot changed its verified "
                    "SourceItem. No output was published."
                )
            read_strategy = "exact-level-0-window"
            source_window = dict(
                snapshot.payload.metadata.get("vipp_source_window", {})
            )
            source_window_digest = str(
                snapshot.payload.metadata.get("vipp_source_window_digest", "")
            )
            provenance: object = {
                "reader": "napari-vipp",
                "source_uri": str(path),
                "read_strategy": read_strategy,
                "source_window": source_window,
                "source_window_digest": source_window_digest,
            }
            payload = replace(
                snapshot.payload,
                metadata={
                    **(snapshot.payload.metadata or {}),
                    "vipp_axis_semantics": axis_semantics,
                },
                name=selected.name or path.name,
                image_state=effective_state,
                axis_semantics_resolved=True,
            )
        else:
            dataset = read_image(
                path,
                series_index=selected.index,
            )
            if dataset.selected_series.key != selected.key:
                raise RuntimeError(
                    "Reader contract mismatch: batch inspection selected item key "
                    f"{selected.key!r} but the reader returned "
                    f"{dataset.selected_series.key!r}."
                )
            provenance = _json_safe(dataset.provenance)
            payload = SourcePayload(
                dataset.data,
                {},
                dataset.selected_series.name or path.name,
                effective_state,
                revision_token=identity,
                axis_semantics_resolved=True,
                source_item=source_item,
            )

        plan_record: dict[str, object] = {
            "reason_code": window_decision.reason_code.value,
            "reason": window_decision.reason,
        }
        if window_decision.plan is not None:
            plan_record.update(
                {
                    "crop_node_id": window_decision.plan.crop_node_id,
                    "decoded_output_bytes": (
                        window_decision.plan.decoded_output_bytes
                    ),
                }
            )
        effective_provenance = (
            {
                **provenance,
                "axis_semantics": axis_semantics,
            }
            if isinstance(provenance, dict)
            else {
                "reader_provenance": provenance,
                "axis_semantics": axis_semantics,
            }
        )
        payloads[node_id] = replace(
            payload,
            metadata={
                **(payload.metadata or {}),
                "vipp_source_path": str(path),
                "vipp_source_provenance": effective_provenance,
                "vipp_axis_semantics": axis_semantics,
                "vipp_source_identity": identity.to_dict(),
                "vipp_source_item_key": source_item.selector.key,
                "vipp_source_item_digest": source_item.digest,
                "vipp_source_item": source_item.to_public_dict(),
            },
            source_item=source_item,
        )
        identity_record = {
            **_path_identity(path),
            **identity.to_dict(),
        }
        source_record: dict[str, object] = {
            "node_id": node_id,
            "title": title,
            "role": role,
            "path": str(path),
            "identity": identity_record,
            "series": {
                "index": selected.index,
                "key": selected.key,
                "name": selected.name,
                "shape": list(selected.shape),
                "dtype": selected.dtype,
                "axes": selected.axes,
                "kind": selected.kind,
            },
            "raw_axes": axis_semantics["raw_axes"],
            "effective_axes": axis_semantics["effective_axes"],
            "axis_declaration": axis_semantics["declaration"],
            "axis_semantics": axis_semantics,
            "image_state": effective_state.to_dict(),
            "provenance": effective_provenance,
            "read_strategy": read_strategy,
            "source_window_plan": plan_record,
        }
        if source_window is not None:
            source_record["source_window"] = source_window
            source_record["source_window_digest"] = source_window_digest
        source_record["source_item"] = source_item.to_dict()
        records.append(source_record)
    return payloads, records


def _declared_batch_source_state(
    raw_state,
    binding: BatchSourceConfig | None,
    *,
    image_source_declaration: object = None,
):
    declaration_source = "batch config" if binding is not None else "Image Source"
    declaration = (
        AxisDeclaration.from_value(image_source_declaration)
        if binding is None
        else binding.axis_declaration
    )
    effective_state = (
        raw_state
        if declaration is None
        else apply_axis_declaration(
            raw_state,
            declaration,
            declaration_source=declaration_source,
        )
    )
    declaration_record = (
        None
        if declaration is None
        else {
            **declaration.to_dict(),
            "source": declaration_source,
            "applied": True,
            "data_order_changed": False,
        }
    )
    return effective_state, {
        "raw_axes": raw_state.axis_order,
        "effective_axes": effective_state.axis_order,
        "declaration": declaration_record,
    }


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
    *,
    cancel_event: threading.Event | None,
    progress_callback: Callable[[BatchExecutionProgress], None] | None,
    item_index: int,
    item_total: int,
    batch_id: str,
) -> dict[str, LocalSourceIdentity]:
    identities: dict[str, LocalSourceIdentity] = {}
    for node_id, path in source_paths.items():
        identities[node_id] = capture_local_source_identity(
            path,
            cancel_callback=(None if cancel_event is None else cancel_event.is_set),
            progress_callback=_source_identity_progress_callback(
                callback=progress_callback,
                item_index=item_index,
                item_total=item_total,
                batch_id=batch_id,
                node_id=node_id,
                operation_id="batch_capture_source_identity",
            ),
        )
    return identities


def _verify_item_source_identities(
    source_paths: dict[str, Path],
    source_identities: dict[str, LocalSourceIdentity],
    *,
    cancel_event: threading.Event | None,
    progress_callback: Callable[[BatchExecutionProgress], None] | None,
    item_index: int,
    item_total: int,
    batch_id: str,
) -> None:
    for node_id, path in source_paths.items():
        verify_local_source_identity(
            path,
            source_identities[node_id],
            cancel_callback=(None if cancel_event is None else cancel_event.is_set),
            progress_callback=_source_identity_progress_callback(
                callback=progress_callback,
                item_index=item_index,
                item_total=item_total,
                batch_id=batch_id,
                node_id=node_id,
                operation_id="batch_verify_source_identity",
            ),
        )


def _source_identity_progress_callback(
    *,
    callback: Callable[[BatchExecutionProgress], None] | None,
    item_index: int,
    item_total: int,
    batch_id: str,
    node_id: str,
    operation_id: str,
) -> Callable[[int, int, str], None]:
    def report(current: int, total: int, message: str) -> None:
        _report_execution_progress(
            callback,
            BatchExecutionProgress(
                item_index=item_index,
                item_total=item_total,
                batch_id=batch_id,
                node_id=node_id,
                operation_id=operation_id,
                current=current,
                total=total,
                message=message,
            ),
        )

    return report


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
    effective_config_sha256: str,
    workflow_file: str,
    config_file: str,
    workflow_document: dict[str, object],
    profile_workflow_sha256: str,
    profile_workflow_document: dict[str, object],
    item_workflow_hashes: dict[int, str],
    node_execution_overrides: dict[str, object],
    config_document: dict[str, object],
    fixed_source_paths: dict[str, Path],
    resolved_fixed_source_items: dict[str, SourceItem],
    effective_request: ComputeRequest,
    *,
    override_used: bool,
) -> BatchManifest:
    items = []
    fixed_source_items = {
        **_workflow_source_items(workflow_document),
        **resolved_fixed_source_items,
    }
    for item in plan.items:
        sources = tuple(
            _planned_source_record(
                node_id,
                path,
                series_index=item.source_series_indices.get(node_id),
                series_name=item.source_series_names.get(node_id, ""),
                source_item=item.source_items.get(node_id),
            )
            for node_id, path in item.source_paths.items()
        )
        sources += tuple(
            _planned_source_record(
                node_id,
                path,
                role="fixed",
                source_item=fixed_source_items.get(node_id),
            )
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
        override_record = parameter_override_provenance(
            item.parameter_override_source_item_key,
            item.parameter_overrides,
            workflow_document,
        )
        items.append(
            BatchItemRecord(
                item.index,
                item.batch_id,
                sources,
                outputs,
                parameter_overrides=override_record,
                effective_workflow_sha256=item_workflow_hashes[item.index],
                node_execution_overrides=node_execution_overrides,
            )
        )
    return BatchManifest(
        run_id=uuid.uuid4().hex,
        started_at=_timestamp(),
        workflow_sha256=workflow_sha256,
        config_sha256=config_sha256,
        effective_config_sha256=effective_config_sha256,
        workflow_file=workflow_file,
        config_file=config_file,
        output_dir=str(plan.output_dir),
        runtime=_runtime_versions(),
        workflow_document=workflow_document,
        config_document=config_document,
        compute={
            "configured_request": plan.config.compute_request.as_dict(),
            "effective_request": effective_request.as_dict(),
            "effective_request_fingerprint": effective_request.fingerprint,
            "override_used": bool(override_used),
            "runtime_cleanup_succeeded": True,
            "warnings": [],
        },
        items=tuple(items),
        profile_workflow_sha256=profile_workflow_sha256,
        profile_workflow_document=profile_workflow_document,
        node_execution_overrides=node_execution_overrides,
    )


def _planned_source_record(
    node_id: str,
    path: Path,
    *,
    role: str = "collection",
    series_index: int | None = None,
    series_name: str = "",
    source_item: SourceItem | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "node_id": node_id,
        "role": role,
        "path": str(path),
    }
    identity = _path_identity(path)
    if identity:
        record["identity"] = identity
    if series_index is not None:
        record["series"] = {
            "index": series_index,
            "name": series_name or f"Series {series_index + 1}",
        }
    if source_item is not None:
        if not isinstance(source_item, SourceItem):
            raise TypeError("planned source_item must be a SourceItem.")
        record["source_item"] = source_item.to_dict()
    return record


def _workflow_source_items(
    workflow_document: dict[str, object],
) -> dict[str, SourceItem]:
    """Extract validated canonical fixed-source evidence from a workflow."""

    raw_nodes = workflow_document.get("nodes", [])
    if not isinstance(raw_nodes, list):
        return {}
    result: dict[str, SourceItem] = {}
    for raw_node in raw_nodes:
        if not isinstance(raw_node, dict) or raw_node.get("operation_id") != "input":
            continue
        node_id = raw_node.get("id")
        params = raw_node.get("params")
        if not isinstance(node_id, str) or not isinstance(params, dict):
            continue
        source_item = source_item_from_params(params)
        if source_item is not None:
            result[node_id] = source_item
    return result


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


def _cancel_from_item(
    manifest: BatchManifest,
    *,
    start_index: int,
    reason: str,
    compute_request: ComputeRequest,
) -> BatchManifest:
    """Mark the first unstarted item cancelled and later items not-run."""

    if start_index >= len(manifest.items):
        return manifest
    items = list(manifest.items)
    current = items[start_index]
    current = _with_synthetic_execution_failure(
        current,
        compute_request,
        OperationCancelled(reason),
        cancelled=True,
        reason_code="operation_cancelled_before_item",
    )
    outputs = tuple(
        replace(
            output,
            status=BatchStatus.CANCELLED,
            error_type="OperationCancelled",
            error_message=reason,
        )
        for output in current.outputs
    )
    items[start_index] = replace(
        current,
        status=BatchStatus.CANCELLED,
        outputs=outputs,
        started_at=current.started_at or _timestamp(),
        finished_at=_timestamp(),
        error_type="OperationCancelled",
        error_message=reason,
    )
    cancelled = replace(manifest, items=tuple(items))
    return _skip_remaining_items(
        cancelled,
        start_index=start_index + 1,
        reason="Not run because the batch was cancelled.",
    )


def _with_synthetic_execution_failure(
    item: BatchItemRecord,
    compute_request: ComputeRequest,
    error: Exception,
    *,
    cancelled: bool,
    reason_code: str = "batch_pre_execution_error",
) -> BatchItemRecord:
    """Attach typed evidence when no execution result could be produced."""

    execution = serialize_execution_provenance(
        compute_request,
        None,
        None,
        failure=PipelineExecutionFailure(
            kind="cancelled" if cancelled else "pre_execution_error",
            error_type=type(error).__name__,
            message=str(error).strip() or type(error).__name__,
            reason_code=(
                "operation_cancelled"
                if cancelled and reason_code == "batch_pre_execution_error"
                else reason_code
            ),
            cleanup_succeeded=True,
        ),
    )
    if item.parameter_overrides:
        execution["parameter_overrides"] = _json_safe(item.parameter_overrides)
    _attach_item_workflow_provenance(execution, item)
    execution["status"] = execution["outcome"]
    provenance_sha256 = execution_provenance_digest(execution)
    return replace(
        item,
        execution=execution,
        execution_provenance_sha256=provenance_sha256,
    )


def _attach_item_workflow_provenance(
    execution: dict[str, object],
    item: BatchItemRecord,
) -> None:
    """Bind execution evidence to the exact effective item workflow."""

    if item.effective_workflow_sha256:
        execution["effective_workflow_sha256"] = item.effective_workflow_sha256
    if item.node_execution_overrides:
        execution["node_execution_overrides"] = _json_safe(
            item.node_execution_overrides
        )


def _batch_actual_compute_summary(
    items: tuple[BatchItemRecord, ...],
) -> dict[str, object]:
    nodes = [
        node
        for item in items
        for node in item.execution.get("nodes", [])
        if isinstance(node, dict)
    ]
    fallbacks = [
        fallback
        for item in items
        for fallback in item.execution.get("fallback_records", [])
        if isinstance(fallback, dict)
    ]
    runtime_ids = sorted(
        {
            str(actual.get("runtime_id", ""))
            for node in nodes
            if isinstance(node.get("actual_implementation"), dict)
            for actual in (node["actual_implementation"],)
            if str(actual.get("runtime_id", ""))
        }
    )
    library_ids = sorted(
        {
            str(actual.get("implementation_library_id", ""))
            for node in nodes
            if isinstance(node.get("actual_implementation"), dict)
            for actual in (node["actual_implementation"],)
            if str(actual.get("implementation_library_id", ""))
        }
    )
    implementation_ids = sorted(
        {
            str(actual.get("implementation_id", ""))
            for node in nodes
            if isinstance(node.get("actual_implementation"), dict)
            for actual in (node["actual_implementation"],)
            if str(actual.get("implementation_id", ""))
        }
    )
    return {
        "items_with_execution": sum(bool(item.execution) for item in items),
        "nodes_executed": len(nodes),
        "runtime_ids": runtime_ids,
        "implementation_library_ids": library_ids,
        "implementation_ids": implementation_ids,
        "fallback_count": len(fallbacks),
        "out_of_memory_fallback_count": sum(
            fallback.get("reason") == "out_of_memory" for fallback in fallbacks
        ),
        "item_cleanup_succeeded": all(
            item.execution.get("cleanup_succeeded") is True
            for item in items
            if item.execution
        ),
    }


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
    ] or [DEFAULT_BATCH_SOURCE_PATTERN]
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


def _expand_source_items(
    paths: list[Path],
    *,
    axis_declaration: AxisDeclaration | None = None,
) -> list[_BatchSourceItem]:
    """Expand multi-series containers through the shared source inspector."""

    items: list[_BatchSourceItem] = []
    inspectable_suffixes = {
        ".npy",
        ".npz",
        ".tif",
        ".tiff",
        ".zarr",
        *RASTER_SUFFIXES,
        *MICROSCOPE_SUFFIXES,
    }
    for path in paths:
        if path.suffix.lower() not in inspectable_suffixes:
            items.append(_BatchSourceItem(path))
            continue
        try:
            inspection = inspect_image_source(path)
        except Exception as exc:
            if path.suffix.lower() not in MICROSCOPE_SUFFIXES:
                items.append(_BatchSourceItem(path))
                continue
            raise ValueError(
                f"Could not inspect batch source collection {path}: {exc}"
            ) from exc
        if not inspection.series:
            raise ValueError(f"Batch source collection contains no images: {path}")
        try:
            bundle = capture_local_source_bundle(
                path,
                source_format=inspection.format,
            )
            resolved_items = {
                series.index: _resolved_batch_source_item(
                    path,
                    bundle,
                    inspection,
                    series.index,
                    axis_declaration=axis_declaration,
                )
                for series in inspection.series
            }
        except Exception as exc:
            raise ValueError(
                f"Could not resolve batch SourceItems for {path}: {exc}"
            ) from exc
        ordered_series = tuple(
            sorted(
                inspection.series,
                key=lambda series: (
                    resolved_items[series.index].selector.key,
                    resolved_items[series.index].selector.digest,
                ),
            )
        )
        if len(ordered_series) == 1:
            selected = ordered_series[0]
            items.append(
                _BatchSourceItem(
                    path,
                    source_item=resolved_items[selected.index],
                )
            )
            continue
        items.extend(
            _BatchSourceItem(
                path=path,
                series_index=series.index,
                series_name=series.name or series.key or f"Series {series.index + 1}",
                source_item=resolved_items[series.index],
            )
            for series in ordered_series
        )
    return items


def _resolved_batch_source_item(
    path: Path,
    bundle: SourceContainerBundle,
    inspection: SourceInspection,
    series_index: int,
    *,
    axis_declaration: AxisDeclaration | None,
) -> SourceItem:
    selected = next(
        series for series in inspection.series if series.index == series_index
    )
    state = selected.image_state or inspect_image_state(
        path,
        inspection=inspection,
        series_index=series_index,
    )
    if axis_declaration is not None:
        state = apply_axis_declaration(
            state,
            axis_declaration,
            declaration_source="batch config",
        )
    return resolve_source_item(
        bundle,
        inspection,
        item_key=selected.key,
        image_state=state,
        axis_declaration=axis_declaration,
    )


def _verify_configured_source_items(
    source: BatchSourceConfig,
    items: list[_BatchSourceItem],
) -> None:
    if not source.source_items:
        return
    observed = tuple(item.source_item for item in items if item.source_item is not None)
    expected_digests = {item.digest for item in source.source_items}
    observed_digests = {item.digest for item in observed}
    if expected_digests != observed_digests:
        raise ValueError(
            f"Batch source {source.title!r} no longer matches its configured "
            "SourceItem revisions or logical selectors. Refresh the collection "
            "and review the changed items before running."
        )


def _batch_source_item_stem(item: _BatchSourceItem) -> str:
    stem = batch_source_stem(item.path)
    if item.series_index is None:
        return stem
    if item.source_item is not None:
        series_name = safe_batch_filename(
            item.source_item.resolved.name or item.source_item.selector.key
        )
        return f"{stem}__{series_name}_{item.source_item.selector.digest[:12]}"
    series_name = safe_batch_filename(
        item.series_name or f"Series_{item.series_index + 1}"
    )
    return f"{stem}__{item.series_index + 1:04d}_{series_name}"


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
    suffix = path.suffix.lower()
    if path.is_dir():
        return suffix == ".zarr"
    if not path.is_file():
        return False
    return suffix in {
        ".npy",
        ".npz",
        ".tif",
        ".tiff",
        *MICROSCOPE_SUFFIXES,
        *RASTER_SUFFIXES,
    }


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


def _implicit_v3_cpu_execution() -> dict[str, object]:
    return {
        "compute": {
            "fallback_policy": "visible",
            "mode": "cpu",
            "node_preferences": {},
            "precision_policy": "scientific-default-v1",
            "workload_policy": "vipp-best-available-v1",
        }
    }


def _canonical_compute_execution(request: ComputeRequest) -> dict[str, object]:
    """Canonicalize validated authored compute intent for scientific hashing."""
    # Preserve hashes created by early workflow-v4 development builds while
    # emitting the clearer public spelling ``custom`` everywhere else.
    hash_mode = (
        "selective" if request.mode is ComputeMode.CUSTOM else request.mode.value
    )
    return {
        "compute": {
            "fallback_policy": request.fallback_policy.value,
            "mode": hash_mode,
            "node_preferences": {
                node_id: (
                    f"{preference.kind.value}:{preference.value}"
                    if preference.value
                    else preference.kind.value
                )
                for node_id, preference in request.node_preferences.items()
            },
            "precision_policy": request.precision_policy_id,
            "workload_policy": request.workload_policy_id,
        }
    }


def _canonical_scientific_node(value: object) -> dict[str, object]:
    """Exclude runtime/UI cache fields while retaining declared node intent."""
    node = _canonical_mapping(value)
    if str(node.get("execution_mode", "run")).strip().casefold() == "run":
        node.pop("execution_mode", None)
    operation_id = str(node.get("operation_id", ""))
    params = dict(_require_object(node.get("params", {}), "Workflow node parameters"))
    if operation_id == "crop_stack":
        # Missing margins in pre-feature workflows are scientifically
        # identical to authored zeroes. Omit no-op margins so their canonical
        # form remains byte-for-byte equivalent to the historical document and
        # existing attached BatchConfig hashes remain valid.
        for name in ("z_start", "z_end"):
            if params.get(name, 0) == 0:
                params.pop(name, None)
    threshold_mode = str(params.get("threshold_mode", "Manual")).casefold()
    source_item = source_item_from_params(params) if operation_id == "input" else None
    source_item_bound = source_item is not None
    filtered: dict[str, object] = {}
    for key in sorted(params):
        name = str(key)
        if name == SOURCE_ITEM_PARAMETER and source_item_bound:
            filtered[name] = source_item.to_dict()
            continue
        if name.startswith("_vipp_") or name == "resolved_spatial_ndim":
            continue
        if source_item_bound and name == "series_index":
            # The ordinal is a compatibility hint, not durable item identity.
            continue
        if operation_id == "combine_channels" and name == "channel_axis":
            # The executor derives this optional cache from input metadata.
            continue
        if threshold_mode.startswith("costes") and name in {
            "channel_1_threshold",
            "channel_2_threshold",
        }:
            # Costes mode derives these values from the current inputs.
            continue
        filtered[name] = _json_safe(params[key])
    node["params"] = filtered
    return node


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


def _source_items_from_batch_source(
    data: dict[str, Any],
    *,
    index: int,
) -> tuple[SourceItem, ...]:
    raw_items = data.get("source_items", [])
    if not isinstance(raw_items, list):
        raise ValueError(f"Batch source {index} source_items must be a list.")
    items: list[SourceItem] = []
    for item_index, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, dict):
            raise ValueError(
                f"Batch source {index} SourceItem {item_index} must be an object."
            )
        try:
            items.append(SourceItem.from_dict(raw_item))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Batch source {index} SourceItem {item_index} is invalid: {exc}"
            ) from exc
    return tuple(items)


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
    "BATCH_NODE_EXECUTION_OVERRIDE_IDENTITY",
    "BATCH_PARAMETER_OVERRIDE_IDENTITY",
    "BATCH_CONFIG_FILENAME",
    "BATCH_CONFIG_TYPE",
    "BATCH_CONFIG_VERSION",
    "BATCH_MANIFEST_FILENAME",
    "BATCH_MANIFEST_TYPE",
    "BATCH_MANIFEST_VERSION",
    "BATCH_SCRIPT_FILENAME",
    "BATCH_WORKFLOW_FILENAME",
    "BatchAxisSuggestion",
    "BatchConfig",
    "BatchExecutionError",
    "BatchExecutionProgress",
    "BatchItemPlan",
    "BatchItemRecord",
    "BatchManifest",
    "BatchNodeExecutionMode",
    "BatchNodeExecutionOverride",
    "BatchOutputConfig",
    "BatchOutputPlan",
    "BatchOutputRecord",
    "BatchParameterOverride",
    "BatchPlan",
    "BatchRunResult",
    "BatchSourceConfig",
    "BatchSourceParameterOverrides",
    "BatchStatus",
    "BatchRuntimeCleanupError",
    "BatchScientificPreflightError",
    "ExistingFilePolicy",
    "atomic_write_json",
    "atomic_write_text",
    "batch_config_hash",
    "bind_batch_plan_source_items",
    "batch_source_item_override_key",
    "effective_batch_compute_request",
    "effective_batch_config_hash",
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
