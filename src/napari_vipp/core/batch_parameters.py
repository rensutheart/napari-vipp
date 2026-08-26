"""Typed per-source numeric parameter overrides for batch execution.

The override identity deliberately excludes a source's local path, reader
version, and series ordinal.  It binds the primary source-node ID and exact
source-container revision to the reader-neutral logical selector, so two
source branches cannot alias and moving an unchanged collection does not
retarget an authored override.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral, Real
from typing import TYPE_CHECKING

from napari_vipp.core.pipeline import ParameterSpec, validate_parameter_value
from napari_vipp.core.source_items import SourceItem
from napari_vipp.core.workflow import deserialize_workflow

if TYPE_CHECKING:
    from napari_vipp.core.pipeline import PrototypePipeline


BATCH_PARAMETER_OVERRIDE_IDENTITY = "primary-source-node-revision-selector-v1"
_OVERRIDE_IDENTITY_DOMAIN = b"napari-vipp-batch-parameter-source-node-v1\0"
_SHA256_HEX_LENGTH = 64
_NON_SCIENTIFIC_OVERRIDE_OPERATIONS = frozenset(
    {
        "input",
        "save_output",
        "batch_output",
        "crop_stack",
        "mip",
        "select_axis_slice",
        "extract_channel",
        "split_channels",
        "composite_to_rgb",
        "skeleton_graph_overlay",
        "colocalization_scatter_plot",
        "masked_colocalization_scatter_plot",
    }
)
_NON_SCIENTIFIC_OVERRIDE_PARAMETERS = frozenset(
    {
        "input_count",
        "axis",
        "index",
        "channel",
        "channel_axis",
        "red_channel",
        "green_channel",
        "blue_channel",
        "preview_channel",
        "output_size",
        "node_size",
        "resolved_spatial_ndim",
    }
)


@dataclass(frozen=True, slots=True)
class BatchParameterOverride:
    """One typed numeric scalar substituted into one workflow node."""

    node_id: str
    parameter: str
    value: int | float

    def __post_init__(self) -> None:
        node_id = _nonempty_text(self.node_id, "override node_id")
        parameter = _nonempty_text(self.parameter, "override parameter")
        raw_value = self.value
        if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
            raise ValueError(
                f"Override {node_id!r}.{parameter} must be a numeric scalar; "
                "booleans, text, arrays, and objects are not supported."
            )
        if isinstance(raw_value, Integral):
            value: int | float = int(raw_value)
        else:
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError(
                    f"Override {node_id!r}.{parameter} must be finite."
                )
        object.__setattr__(self, "node_id", node_id)
        object.__setattr__(self, "parameter", parameter)
        object.__setattr__(self, "value", value)


@dataclass(frozen=True, slots=True)
class BatchSourceParameterOverrides:
    """Overrides selected by one exact, path-independent SourceItem identity."""

    source_item_key: str
    values: tuple[BatchParameterOverride, ...]

    def __post_init__(self) -> None:
        key = _sha256(self.source_item_key, "override source_item_key")
        values = tuple(self.values)
        if not values:
            raise ValueError(
                f"Parameter override entry {key!r} must contain at least one value."
            )
        if any(not isinstance(value, BatchParameterOverride) for value in values):
            raise TypeError(
                "Batch source parameter overrides must contain only "
                "BatchParameterOverride records."
            )
        pairs = [(value.node_id, value.parameter) for value in values]
        if len(pairs) != len(set(pairs)):
            raise ValueError(
                f"Parameter override entry {key!r} contains duplicate "
                "node/parameter assignments."
            )
        object.__setattr__(self, "source_item_key", key)
        object.__setattr__(
            self,
            "values",
            tuple(sorted(values, key=lambda value: (value.node_id, value.parameter))),
        )


def batch_source_item_override_key(
    primary_source_node_id: str,
    source_item: SourceItem,
) -> str:
    """Return the stable key used to address one source item in a batch."""

    source_node_id = _nonempty_text(
        primary_source_node_id,
        "primary source node_id",
    )
    if not isinstance(source_item, SourceItem):
        raise TypeError("source_item must be a SourceItem.")
    document = {
        "primary_source_node_id": source_node_id,
        "container_format": source_item.container.format,
        "source_revision_sha256": source_item.container.revision.sha256,
        "selector_sha256": source_item.selector.digest,
    }
    encoded = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(_OVERRIDE_IDENTITY_DOMAIN + encoded).hexdigest()


def batch_parameter_override_ineligibility_reason(
    operation_id: str,
    parameter: ParameterSpec,
) -> str:
    """Explain why a ParameterSpec is not a scientific per-sample scalar."""

    operation = _nonempty_text(operation_id, "override operation_id")
    try:
        name = str(parameter.name)
        kind = str(parameter.kind)
    except AttributeError as exc:
        raise TypeError("parameter must be a ParameterSpec.") from exc
    if kind not in {"int", "float"}:
        return "only declared int and float scalar parameters are supported"
    if operation in _NON_SCIENTIFIC_OVERRIDE_OPERATIONS:
        return (
            f"operation {operation!r} is a source, output, selector, ROI, or "
            "presentation operation"
        )
    if name.startswith("_") or name in _NON_SCIENTIFIC_OVERRIDE_PARAMETERS:
        return (
            f"parameter {name!r} controls topology, source/item selection, "
            "axes/channels, ROI, presentation, or derived state"
        )
    return ""


def is_batch_parameter_override_eligible(
    operation_id: str,
    parameter: ParameterSpec,
) -> bool:
    """Return whether one public scalar may vary by exact primary SourceItem."""

    return not batch_parameter_override_ineligibility_reason(
        operation_id,
        parameter,
    )


def parse_batch_parameter_overrides(
    value: object,
    *,
    label: str = "Batch config parameter_overrides",
) -> tuple[BatchSourceParameterOverrides, ...]:
    """Parse the strict persisted override mapping."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object.")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} field names must be strings.")
    unknown = set(value) - {"identity", "items"}
    if unknown:
        raise ValueError(
            f"{label} contains unknown fields: " + ", ".join(sorted(unknown)) + "."
        )
    if value.get("identity") != BATCH_PARAMETER_OVERRIDE_IDENTITY:
        raise ValueError(
            f"{label} identity must be "
            f"{BATCH_PARAMETER_OVERRIDE_IDENTITY!r}."
        )
    raw_items = value.get("items")
    if not isinstance(raw_items, Mapping) or not raw_items:
        raise ValueError(f"{label} items must be a non-empty object.")
    result: list[BatchSourceParameterOverrides] = []
    for raw_key in sorted(raw_items, key=str):
        key = _sha256(raw_key, f"{label} item key")
        raw_nodes = raw_items[raw_key]
        if not isinstance(raw_nodes, Mapping) or not raw_nodes:
            raise ValueError(
                f"{label} item {key!r} must contain a non-empty node object."
            )
        values: list[BatchParameterOverride] = []
        for raw_node_id in sorted(raw_nodes, key=str):
            node_id = _nonempty_text(raw_node_id, f"{label} node_id")
            raw_parameters = raw_nodes[raw_node_id]
            if not isinstance(raw_parameters, Mapping) or not raw_parameters:
                raise ValueError(
                    f"{label} node {node_id!r} must contain a non-empty "
                    "parameter object."
                )
            for raw_parameter in sorted(raw_parameters, key=str):
                values.append(
                    BatchParameterOverride(
                        node_id=node_id,
                        parameter=_nonempty_text(
                            raw_parameter,
                            f"{label} parameter",
                        ),
                        value=raw_parameters[raw_parameter],
                    )
                )
        result.append(BatchSourceParameterOverrides(key, tuple(values)))
    return tuple(result)


def batch_parameter_overrides_document(
    overrides: tuple[BatchSourceParameterOverrides, ...],
) -> dict[str, object]:
    """Serialize overrides canonically, ordered by identity/node/parameter."""

    normalized = normalize_batch_parameter_overrides(overrides)
    items: dict[str, object] = {}
    for item in normalized:
        nodes: dict[str, dict[str, int | float]] = {}
        for value in item.values:
            nodes.setdefault(value.node_id, {})[value.parameter] = value.value
        items[item.source_item_key] = nodes
    return {
        "identity": BATCH_PARAMETER_OVERRIDE_IDENTITY,
        "items": items,
    }


def normalize_batch_parameter_overrides(
    overrides: tuple[BatchSourceParameterOverrides, ...],
) -> tuple[BatchSourceParameterOverrides, ...]:
    normalized = tuple(overrides)
    if any(not isinstance(item, BatchSourceParameterOverrides) for item in normalized):
        raise TypeError(
            "parameter_overrides must contain only "
            "BatchSourceParameterOverrides records."
        )
    keys = [item.source_item_key for item in normalized]
    if len(keys) != len(set(keys)):
        raise ValueError("Batch parameter override source identities must be unique.")
    return tuple(sorted(normalized, key=lambda item: item.source_item_key))


def validate_batch_parameter_overrides(
    overrides: tuple[BatchSourceParameterOverrides, ...],
    pipeline: PrototypePipeline,
) -> None:
    """Validate every override against the current workflow contract."""

    for item in normalize_batch_parameter_overrides(overrides):
        for override in item.values:
            node = pipeline.nodes.get(override.node_id)
            if node is None:
                raise ValueError(
                    "Batch parameter override references missing node "
                    f"{override.node_id!r}."
                )
            operation = pipeline.operation_spec(node.operation_id)
            parameter = next(
                (
                    candidate
                    for candidate in operation.parameters
                    if candidate.name == override.parameter
                ),
                None,
            )
            if parameter is None:
                raise ValueError(
                    f"Batch parameter override node {override.node_id!r} "
                    f"({node.operation_id}) has no public parameter "
                    f"{override.parameter!r}."
                )
            ineligible = batch_parameter_override_ineligibility_reason(
                node.operation_id,
                parameter,
            )
            if ineligible:
                raise ValueError(
                    f"Batch parameter override {override.node_id!r}."
                    f"{override.parameter} is not eligible: {ineligible}."
                )
            validate_parameter_value(
                parameter,
                override.value,
                context=f"Batch override for node {override.node_id!r}",
            )
            if parameter.kind == "int" and (
                isinstance(override.value, bool)
                or not isinstance(override.value, Integral)
            ):
                raise ValueError(
                    f"Batch override for node {override.node_id!r} parameter "
                    f"{override.parameter!r} must be an integer."
                )
            if (
                not parameter.data_dependent_bounds
                and not parameter.minimum <= override.value <= parameter.maximum
            ):
                raise ValueError(
                    f"Batch override for node {override.node_id!r} parameter "
                    f"{override.parameter!r} must be between "
                    f"{parameter.minimum!r} and {parameter.maximum!r}; got "
                    f"{override.value!r}."
                )


def workflow_with_parameter_overrides(
    workflow: object,
    overrides: tuple[BatchParameterOverride, ...],
) -> dict[str, object]:
    """Return a validated detached workflow with resolved scalar values."""

    if not overrides:
        if not isinstance(workflow, dict):
            raise ValueError("Workflow must be an object.")
        return workflow
    document = deepcopy(workflow)
    if not isinstance(document, dict):
        raise ValueError("Workflow must be an object.")
    raw_nodes = document.get("nodes")
    if not isinstance(raw_nodes, list):
        raise ValueError("Workflow nodes must be a list.")
    nodes = {
        node.get("id"): node
        for node in raw_nodes
        if isinstance(node, dict) and isinstance(node.get("id"), str)
    }
    for override in overrides:
        node = nodes.get(override.node_id)
        if node is None:
            raise ValueError(
                f"Batch parameter override references missing node "
                f"{override.node_id!r}."
            )
        params = node.get("params")
        if not isinstance(params, dict):
            raise ValueError(
                f"Workflow node {override.node_id!r} parameters must be an object."
            )
        params[override.parameter] = override.value
    # The ordinary workflow boundary remains authoritative after substitution.
    deserialize_workflow(document)
    return document


def parameter_override_provenance(
    source_item_key: str,
    overrides: tuple[BatchParameterOverride, ...],
    workflow: Mapping[str, object],
) -> dict[str, object]:
    """Record authored and resolved scalar values for one batch item."""

    if not overrides:
        return {}
    raw_nodes = workflow.get("nodes", [])
    nodes = {
        node.get("id"): node
        for node in raw_nodes
        if isinstance(node, Mapping) and isinstance(node.get("id"), str)
    }
    values: list[dict[str, object]] = []
    for override in overrides:
        node = nodes.get(override.node_id, {})
        params = node.get("params", {}) if isinstance(node, Mapping) else {}
        base_value = (
            params.get(override.parameter)
            if isinstance(params, Mapping)
            else None
        )
        values.append(
            {
                "node_id": override.node_id,
                "operation_id": str(node.get("operation_id", "")),
                "parameter": override.parameter,
                "workflow_value": base_value,
                "resolved_value": override.value,
            }
        )
    return {
        "identity": BATCH_PARAMETER_OVERRIDE_IDENTITY,
        "source_item_key": _sha256(source_item_key, "override source_item_key"),
        "values": values,
    }


def _nonempty_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text.")
    return value.strip()


def _sha256(value: object, label: str) -> str:
    text = _nonempty_text(value, label)
    if len(text) != _SHA256_HEX_LENGTH or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{label} must be lowercase SHA-256.")
    return text


__all__ = [
    "BATCH_PARAMETER_OVERRIDE_IDENTITY",
    "BatchParameterOverride",
    "BatchSourceParameterOverrides",
    "batch_parameter_overrides_document",
    "batch_parameter_override_ineligibility_reason",
    "batch_source_item_override_key",
    "is_batch_parameter_override_eligible",
    "normalize_batch_parameter_overrides",
    "parameter_override_provenance",
    "parse_batch_parameter_overrides",
    "validate_batch_parameter_overrides",
    "workflow_with_parameter_overrides",
]
