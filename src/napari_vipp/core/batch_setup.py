"""Build reproducible collection-batch configuration from a workflow graph.

This module translates a validated workflow document and user-entered source
bindings into the core batch contract. It deliberately contains no dialogs,
Qt state, or output-writing behavior; planning and execution remain in
``core.batch``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from napari_vipp.core.batch import (
    BATCH_WORKFLOW_FILENAME,
    BatchConfig,
    BatchOutputConfig,
    BatchSourceConfig,
    ExistingFilePolicy,
    safe_batch_filename,
    scientific_workflow_hash,
    validate_batch_config,
)
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.workflow import deserialize_workflow


def build_collection_batch_config(
    workflow: dict,
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    pattern: str = "*.tif",
    image_format: str = "ome-tiff",
    save_python_script: bool = True,
    source_bindings: Sequence[Mapping[str, object]] | None = None,
    existing_file_policy: str = ExistingFilePolicy.ERROR.value,
    continue_on_error: bool = True,
) -> BatchConfig:
    """Translate one workflow and collection form into a validated config."""
    output_text = str(output_dir).strip()
    if not output_text:
        raise ValueError("Batch output folder cannot be blank.")

    pipeline = pipeline_from_workflow(workflow)
    output_node_ids = batch_saved_node_ids(pipeline)
    if not output_node_ids:
        raise ValueError("The workflow has no outputs to save.")

    sources = _batch_source_configs(
        pipeline,
        input_dir=Path(str(input_dir).strip()).expanduser(),
        pattern=pattern,
        source_bindings=source_bindings,
    )
    outputs = tuple(
        _batch_output_config(pipeline, node_id)
        for node_id in output_node_ids
    )
    try:
        policy = ExistingFilePolicy(str(existing_file_policy))
    except ValueError as exc:
        raise ValueError(
            f"Unsupported existing-file policy: {existing_file_policy!r}."
        ) from exc

    config = BatchConfig(
        workflow_file=Path(BATCH_WORKFLOW_FILENAME),
        workflow_sha256=scientific_workflow_hash(workflow),
        output_dir=Path(output_text).expanduser().resolve(),
        sources=sources,
        outputs=outputs,
        default_image_format=image_format,
        existing_file_policy=policy,
        save_workflow_snapshot=True,
        save_python_script=save_python_script,
        continue_on_error=continue_on_error,
    )
    validate_batch_config(
        workflow,
        config,
        workflow_path=config.output_dir / BATCH_WORKFLOW_FILENAME,
    )
    return config


def pipeline_from_workflow(workflow: dict) -> PrototypePipeline:
    """Restore a detached graph model from a validated workflow document."""
    restored = deserialize_workflow(workflow)
    pipeline = PrototypePipeline()
    pipeline.restore_graph(
        restored["nodes"],
        restored["connections"],
        restored.get("output_tunnels", ()),
    )
    return pipeline


def batch_source_rows(pipeline: PrototypePipeline) -> list[dict[str, str]]:
    """Return stable Image Source descriptors for the collection dialog."""
    rows = [
        {
            "node_id": node_id,
            "title": pipeline.nodes[node_id].title,
            "binding_mode": str(
                pipeline.nodes[node_id].params.get(
                    "binding_mode",
                    "single item",
                )
            ),
        }
        for node_id in pipeline.topological_order()
        if pipeline.nodes[node_id].operation_id == "input"
    ]
    return rows or [
        {
            "node_id": "input",
            "title": "Image Source",
            "binding_mode": "collection",
        }
    ]


def batch_output_node_ids(pipeline: PrototypePipeline) -> list[str]:
    """Return explicit Batch Output nodes in deterministic graph order."""
    return [
        node_id
        for node_id in pipeline.topological_order()
        if pipeline.nodes[node_id].operation_id == "batch_output"
    ]


def batch_saved_node_ids(pipeline: PrototypePipeline) -> list[str]:
    """Resolve explicit outputs or the documented terminal-node fallback."""
    explicit = batch_output_node_ids(pipeline)
    return explicit if explicit else _terminal_node_ids(pipeline)


def batch_output_tag(pipeline: PrototypePipeline, node_id: str) -> str:
    """Return the deterministic filename tag for one saved node."""
    node = pipeline.nodes[node_id]
    if node.operation_id == "batch_output":
        raw = str(node.params.get("tag", "")).strip()
        return safe_batch_filename(raw or node_id)
    return safe_batch_filename(f"{node.title}-{node_id}")


def _batch_source_configs(
    pipeline: PrototypePipeline,
    *,
    input_dir: Path,
    pattern: str,
    source_bindings: Sequence[Mapping[str, object]] | None,
) -> tuple[BatchSourceConfig, ...]:
    configs: list[BatchSourceConfig] = []
    if source_bindings is not None:
        for row in source_bindings:
            node_id = str(row.get("node_id", "")).strip()
            node = pipeline.nodes.get(node_id)
            if node is None or node.operation_id != "input":
                continue
            raw_dir = str(row.get("input_dir", "")).strip()
            if not raw_dir:
                continue
            configs.append(
                BatchSourceConfig(
                    node_id=node_id,
                    title=str(row.get("title", node.title) or node.title),
                    input_dir=Path(raw_dir).expanduser().resolve(),
                    pattern=str(row.get("pattern", "") or pattern or "*.tif"),
                )
            )
    if configs:
        return tuple(configs)
    if source_bindings is not None:
        raise ValueError("At least one batch source needs an input folder.")
    if not input_dir.is_dir():
        raise ValueError("Batch input folder does not exist.")
    return tuple(
        BatchSourceConfig(
            node_id=node_id,
            title=pipeline.nodes[node_id].title,
            input_dir=input_dir.resolve(),
            pattern=pattern or "*.tif",
        )
        for node_id in _collection_source_node_ids(pipeline)
    )


def _collection_source_node_ids(pipeline: PrototypePipeline) -> set[str]:
    source_ids = [
        node_id
        for node_id in pipeline.topological_order()
        if pipeline.nodes[node_id].operation_id == "input"
    ]
    collection_ids = {
        node_id
        for node_id in source_ids
        if str(
            pipeline.nodes[node_id].params.get("binding_mode", "single item")
        )
        == "collection"
    }
    if collection_ids:
        return collection_ids
    return {source_ids[0]} if source_ids else set()


def _terminal_node_ids(pipeline: PrototypePipeline) -> list[str]:
    order = pipeline.topological_order()
    consumed = {connection.source_id for connection in pipeline.connections}
    terminals = [node_id for node_id in order if node_id not in consumed]
    return terminals or list(order)


def _batch_output_config(
    pipeline: PrototypePipeline,
    node_id: str,
) -> BatchOutputConfig:
    node = pipeline.nodes[node_id]
    params = node.params if node.operation_id == "batch_output" else {}
    ports = pipeline.output_ports(node_id)
    output_type = ports[0].output_type if ports else "any"
    return BatchOutputConfig(
        node_id=node_id,
        node_title=node.title,
        tag=batch_output_tag(pipeline, node_id),
        kind="table" if output_type == "table" else "image",
        format=str(params.get("format", "batch default")),
        subfolder=str(params.get("subfolder", "")),
        filename_template=str(
            params.get("filename_template", "{source_stem}__{tag}")
        ),
        overwrite=str(params.get("overwrite", "batch default")),
    )


__all__ = [
    "batch_output_node_ids",
    "batch_output_tag",
    "batch_saved_node_ids",
    "batch_source_rows",
    "build_collection_batch_config",
    "pipeline_from_workflow",
]
