"""Typed whole-batch overrides for reviewed Safe Node Bypass nodes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum

from napari_vipp.core.pipeline import (
    NODE_EXECUTION_BYPASS,
    NODE_EXECUTION_RUN,
    PrototypePipeline,
    validate_node_execution_mode,
)
from napari_vipp.core.workflow import (
    WORKFLOW_VERSION,
    deserialize_workflow,
    serialize_workflow,
)

BATCH_NODE_EXECUTION_OVERRIDE_IDENTITY = "vipp-batch-node-execution-overrides-v1"
BYPASS_CONTRACT_IDENTITY = "primary-input-single-output-graph-splice-v2"


class BatchNodeExecutionMode(StrEnum):
    """An explicit batch-wide node execution directive."""

    RUN = NODE_EXECUTION_RUN
    BYPASS = NODE_EXECUTION_BYPASS


@dataclass(frozen=True, order=True)
class BatchNodeExecutionOverride:
    """Apply one explicit execution mode to one node for the whole batch."""

    node_id: str
    mode: BatchNodeExecutionMode

    def __post_init__(self) -> None:
        node_id = str(self.node_id).strip()
        if not node_id:
            raise ValueError("Batch node execution override needs a node ID.")
        try:
            mode = BatchNodeExecutionMode(str(self.mode).strip().casefold())
        except ValueError as exc:
            raise ValueError(
                "Batch node execution override mode must be 'run' or 'bypass'."
            ) from exc
        object.__setattr__(self, "node_id", node_id)
        object.__setattr__(self, "mode", mode)


@dataclass(frozen=True)
class BatchNodeExecutionSpec:
    """One reviewed graph node exposed by the batch workspace."""

    node_id: str
    title: str
    operation_id: str
    workflow_mode: BatchNodeExecutionMode


def normalize_batch_node_execution_overrides(
    overrides: Sequence[BatchNodeExecutionOverride] | None,
) -> tuple[BatchNodeExecutionOverride, ...]:
    """Return a sorted, unique immutable override set."""

    normalized: list[BatchNodeExecutionOverride] = []
    seen: set[str] = set()
    for raw in overrides or ():
        if not isinstance(raw, BatchNodeExecutionOverride):
            raise TypeError(
                "Batch node execution overrides must contain "
                "BatchNodeExecutionOverride values."
            )
        item = BatchNodeExecutionOverride(raw.node_id, raw.mode)
        if item.node_id in seen:
            raise ValueError(
                f"Duplicate batch node execution override for {item.node_id!r}."
            )
        seen.add(item.node_id)
        normalized.append(item)
    return tuple(sorted(normalized, key=lambda item: item.node_id))


def batch_node_execution_overrides_document(
    overrides: Sequence[BatchNodeExecutionOverride],
) -> dict[str, object]:
    """Serialize the canonical durable override profile."""

    normalized = normalize_batch_node_execution_overrides(overrides)
    return {
        "identity": BATCH_NODE_EXECUTION_OVERRIDE_IDENTITY,
        "nodes": {item.node_id: item.mode.value for item in normalized},
    }


def parse_batch_node_execution_overrides(
    value: object,
) -> tuple[BatchNodeExecutionOverride, ...]:
    """Parse a strict durable override profile."""

    if not isinstance(value, Mapping):
        raise ValueError("Batch node execution overrides must be an object.")
    unknown = set(value) - {"identity", "nodes"}
    if unknown:
        raise ValueError(
            "Batch node execution overrides contain unknown fields: "
            + ", ".join(sorted(map(str, unknown)))
            + "."
        )
    if value.get("identity") != BATCH_NODE_EXECUTION_OVERRIDE_IDENTITY:
        raise ValueError("Batch node execution override identity is unsupported.")
    nodes = value.get("nodes")
    if not isinstance(nodes, Mapping):
        raise ValueError("Batch node execution override nodes must be an object.")
    return normalize_batch_node_execution_overrides(
        tuple(
            BatchNodeExecutionOverride(str(node_id), mode)
            for node_id, mode in nodes.items()
        )
    )


def batch_node_execution_specs(
    pipeline: PrototypePipeline,
) -> tuple[BatchNodeExecutionSpec, ...]:
    """Describe explicitly reviewed nodes in stable graph order."""

    specs: list[BatchNodeExecutionSpec] = []
    for node_id in pipeline.topological_order():
        node = pipeline.nodes[node_id]
        operation = pipeline.operation_spec(node.operation_id)
        if not operation.supports_bypass:
            continue
        if (
            pipeline._bypass_primary_connection(node_id) is None
            or not pipeline.node_has_output_use(node_id)
        ):
            continue
        specs.append(
            BatchNodeExecutionSpec(
                node_id=node_id,
                title=node.title,
                operation_id=node.operation_id,
                workflow_mode=BatchNodeExecutionMode(node.execution_mode),
            )
        )
    return tuple(specs)


def validate_batch_node_execution_overrides(
    overrides: Sequence[BatchNodeExecutionOverride],
    pipeline: PrototypePipeline,
) -> tuple[BatchNodeExecutionOverride, ...]:
    """Validate a profile against exact reviewed nodes in ``pipeline``."""

    normalized = normalize_batch_node_execution_overrides(overrides)
    for override in normalized:
        node = pipeline.nodes.get(override.node_id)
        if node is None:
            raise ValueError(
                "Batch node execution override references missing node "
                f"{override.node_id!r}."
            )
        operation = pipeline.operation_spec(node.operation_id)
        if not operation.supports_bypass:
            raise ValueError(
                f"Batch node execution override node {override.node_id!r} "
                f"operation {node.operation_id!r} is not eligible for Safe "
                "Node Bypass."
            )
        validate_node_execution_mode(
            operation,
            override.mode.value,
            context=f"Batch override for node {override.node_id!r}",
        )
        if override.mode is BatchNodeExecutionMode.BYPASS:
            if pipeline._bypass_primary_connection(override.node_id) is None:
                raise ValueError(
                    f"Batch override cannot bypass {override.node_id!r}: connect "
                    "its primary input port 0 first."
                )
            if not pipeline.node_has_output_use(override.node_id):
                raise ValueError(
                    f"Batch override cannot bypass {override.node_id!r}: its "
                    "output is not currently used by another node or an output "
                    "tunnel."
                )
    candidate = PrototypePipeline()
    candidate.restore_graph(
        tuple(pipeline.nodes.values()),
        tuple(pipeline.connections),
        pipeline.output_tunnel_list(),
    )
    try:
        candidate.apply_atomic_node_execution_profile(
            {override.node_id: override.mode.value for override in normalized}
        )
    except ValueError as exc:
        raise ValueError(f"Batch execution profile is incompatible: {exc}") from exc
    return normalized


def workflow_with_node_execution_overrides(
    workflow: object,
    overrides: Sequence[BatchNodeExecutionOverride],
) -> dict[str, object]:
    """Return a validated detached workflow with whole-batch modes applied."""

    if not isinstance(workflow, dict):
        raise ValueError("Workflow must be an object.")
    normalized = normalize_batch_node_execution_overrides(overrides)
    if not normalized:
        return deepcopy(workflow)

    restored = deserialize_workflow(workflow)
    pipeline = PrototypePipeline()
    pipeline.restore_graph(
        restored["nodes"],
        restored["connections"],
        restored.get("output_tunnels", ()),
    )
    validate_batch_node_execution_overrides(normalized, pipeline)

    # Re-materialize legacy documents through the current workflow boundary
    # before introducing the v6-only node execution field.
    if workflow.get("version") != WORKFLOW_VERSION:
        document: dict[str, object] = serialize_workflow(
            pipeline,
            positions=restored["positions"],
            notes=restored["notes"],
            metadata=restored["metadata"],
            compute_request=restored["compute_request"],
        )
        if "batch_config" in workflow:
            document["batch_config"] = deepcopy(workflow["batch_config"])
    else:
        document = deepcopy(workflow)

    raw_nodes = document.get("nodes")
    if not isinstance(raw_nodes, list):
        raise ValueError("Workflow nodes must be a list.")
    nodes = {
        str(node.get("id")): node
        for node in raw_nodes
        if isinstance(node, dict) and isinstance(node.get("id"), str)
    }
    for override in normalized:
        node = nodes[override.node_id]
        if override.mode is BatchNodeExecutionMode.RUN:
            node.pop("execution_mode", None)
        else:
            node["execution_mode"] = override.mode.value
    effective = deserialize_workflow(document)
    effective_pipeline = PrototypePipeline()
    effective_pipeline.restore_graph(
        effective["nodes"],
        effective["connections"],
        effective.get("output_tunnels", ()),
        atomic_bypass_profile=True,
    )
    return document


def node_execution_override_provenance(
    overrides: Sequence[BatchNodeExecutionOverride],
    authored_workflow: Mapping[str, object],
    effective_workflow: Mapping[str, object],
) -> dict[str, object]:
    """Record authored directives and resolved modes for durable evidence."""

    normalized = normalize_batch_node_execution_overrides(overrides)
    if not normalized:
        return {}
    authored = _node_documents(authored_workflow)
    effective = _node_documents(effective_workflow)
    values: list[dict[str, object]] = []
    for override in normalized:
        authored_node = authored[override.node_id]
        effective_node = effective[override.node_id]
        values.append(
            {
                "node_id": override.node_id,
                "operation_id": str(authored_node.get("operation_id", "")),
                "workflow_mode": str(
                    authored_node.get("execution_mode", NODE_EXECUTION_RUN)
                ),
                "directive": override.mode.value,
                "effective_mode": str(
                    effective_node.get("execution_mode", NODE_EXECUTION_RUN)
                ),
                "bypass_contract": BYPASS_CONTRACT_IDENTITY,
            }
        )
    return {
        "identity": BATCH_NODE_EXECUTION_OVERRIDE_IDENTITY,
        "values": values,
    }


def _node_documents(
    workflow: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    raw_nodes = workflow.get("nodes")
    if not isinstance(raw_nodes, list):
        raise ValueError("Workflow nodes must be a list.")
    result = {
        str(node.get("id")): node
        for node in raw_nodes
        if isinstance(node, Mapping) and isinstance(node.get("id"), str)
    }
    return result


__all__ = [
    "BATCH_NODE_EXECUTION_OVERRIDE_IDENTITY",
    "BYPASS_CONTRACT_IDENTITY",
    "BatchNodeExecutionMode",
    "BatchNodeExecutionOverride",
    "BatchNodeExecutionSpec",
    "batch_node_execution_overrides_document",
    "batch_node_execution_specs",
    "node_execution_override_provenance",
    "normalize_batch_node_execution_overrides",
    "parse_batch_node_execution_overrides",
    "validate_batch_node_execution_overrides",
    "workflow_with_node_execution_overrides",
]
