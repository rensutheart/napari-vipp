"""Generate a runnable, headless Python script from a pipeline graph.

Exported scripts execute the serialized workflow through
:class:`~napari_vipp.core.pipeline.PrototypePipeline`.  Keeping the shared
executor in the generated program is important: operation calls alone cannot
reproduce the axis validation, physical-grid checks, metadata-derived runtime
arguments, or output :class:`~napari_vipp.core.metadata.ImageState` updates that
are part of a scientific VIPP workflow.
"""

from __future__ import annotations

import json
import keyword
import re
from datetime import UTC, datetime

from napari_vipp import __version__ as VIPP_VERSION
from napari_vipp.core.pipeline import (
    NODE_LIBRARY_BY_ID,
    PrototypePipeline,
)
from napari_vipp.core.workflow import deserialize_workflow, serialize_workflow

_INDENT = " " * 4
_RESERVED_FUNCTION_NAMES = {
    "EXPECTED_VIPP_VERSION",
    "ImageDataset",
    "OUTPUT_NODES",
    "Path",
    "PipelineResults",
    "SOURCE_NODES",
    "SourcePayload",
    "VIPP_VERSION",
    "_WORKFLOW_JSON",
    "_coerce_source_payload",
    "_dataset_metadata",
    "_new_pipeline",
    "_source_override",
    "_table_output_path",
    "argparse",
    "batch_process",
    "is_table_data",
    "json",
    "load_image",
    "pipeline_from_workflow",
    "read_image",
    "save_image",
    "save_table_output",
    "write_image",
}


def export_pipeline_to_python(
    pipeline: PrototypePipeline,
    *,
    function_name: str = "run_pipeline",
) -> str:
    """Return Python source code that reproduces the pipeline headlessly."""
    if not function_name.isidentifier() or keyword.iskeyword(function_name):
        raise ValueError(f"Invalid exported function name: {function_name!r}.")
    order = pipeline.topological_order()

    source_ids = [
        node_id
        for node_id in order
        if not NODE_LIBRARY_BY_ID[pipeline.nodes[node_id].operation_id].has_input
    ]
    source_param_names = _unique_names(source_ids, prefix="src")
    terminal_ids = _terminal_nodes(pipeline, order)
    used_functions = _used_function_names(pipeline, order)
    if function_name in _RESERVED_FUNCTION_NAMES or function_name in used_functions:
        raise ValueError(f"Invalid exported function name: {function_name!r}.")

    body_lines, missing = _build_function_body(
        pipeline,
        order,
        source_param_names,
        function_name,
    )
    header = _build_header(pipeline)
    imports = _build_imports()
    workflow = _build_workflow_constant(pipeline)
    helpers = _build_helpers()
    constants = _build_constants(source_ids, terminal_ids)
    main = _build_main(source_ids, function_name)

    sections = [
        header,
        imports,
        workflow,
        constants,
        helpers,
        "\n".join(body_lines),
        main,
    ]
    document = "\n\n\n".join(section for section in sections if section)
    if missing:
        note = "\n".join(f"# NOTE: {line}" for line in missing)
        document = f"{document}\n\n\n{note}\n"
    return document.rstrip() + "\n"


def export_batch_runner_to_python() -> str:
    """Return a thin launcher for a saved workflow and batch configuration.

    Unlike :func:`export_pipeline_to_python`, this companion deliberately keeps
    the shared batch engine as the source of truth for pairing, filenames,
    provenance, and failure handling.
    """
    return '''"""Run a saved napari-vipp batch workflow reproducibly."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from napari_vipp.core.batch import run_batch_from_files


def main(argv=None):
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run a saved napari-vipp workflow and batch configuration."
    )
    parser.add_argument(
        "--workflow",
        default=None,
        help="Workflow override (default: the path recorded by the config).",
    )
    parser.add_argument(
        "--config",
        default=str(script_dir / "vipp_batch_config.json"),
        help="Saved VIPP batch config JSON (default: sibling config artifact).",
    )
    args = parser.parse_args(argv)
    try:
        result = run_batch_from_files(args.workflow, args.config)
    except Exception as exc:
        print(f"Batch failed before or during execution: {exc}", file=sys.stderr)
        return 2
    summary = result.summary
    print(
        f"{summary['completed']} completed, "
        f"{summary['partial']} partial, "
        f"{summary['skipped']} skipped, "
        f"{summary['failed']} failed; "
        f"{len(result.saved_paths)} outputs saved; "
        f"manifest: {result.manifest_path}"
    )
    return 1 if result.has_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _build_header(pipeline: PrototypePipeline) -> str:
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    node_count = len(pipeline.nodes)
    connection_count = len(pipeline.connections)
    return (
        '"""Auto-generated by napari-vipp pipeline export.\n\n'
        f"Generated: {stamp}\n"
        f"Nodes: {node_count}  Connections: {connection_count}\n\n"
        "Run a single image:\n"
        "    python this_script.py input.tif output.tif\n\n"
        "Batch process a folder:\n"
        "    python this_script.py input_dir/ output_dir/ --pattern '*.tif'\n"
        '"""\n'
        "from __future__ import annotations"
    )


def _build_imports() -> str:
    return "\n".join(
        (
            "import json",
            "from pathlib import Path",
            "",
            "from napari_vipp import __version__ as VIPP_VERSION",
            "from napari_vipp.core.batch_setup import pipeline_from_workflow",
            "from napari_vipp.core.io import ImageDataset, read_image, write_image",
            "from napari_vipp.core.pipeline import SourcePayload",
            "from napari_vipp.core.tables import is_table_data, save_table_output",
        )
    )


def _build_workflow_constant(pipeline: PrototypePipeline) -> str:
    """Embed one immutable, validated workflow snapshot.

    A JSON string, rather than a live dict literal, prevents one run (or caller)
    from mutating the graph used by later invocations.  ``pipeline_from_workflow``
    deserializes and validates a fresh document on every call.
    """
    document = serialize_workflow(pipeline)
    deserialize_workflow(document)
    encoded = json.dumps(
        document,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return (
        f"EXPECTED_VIPP_VERSION = {VIPP_VERSION!r}\n"
        f"_WORKFLOW_JSON = {encoded!r}"
    )


def _build_constants(source_ids: list[str], terminal_ids: list[str]) -> str:
    sources = ", ".join(repr(node_id) for node_id in source_ids)
    terminals = ", ".join(repr(node_id) for node_id in terminal_ids)
    return (
        f"SOURCE_NODES = ({sources}{',' if len(source_ids) == 1 else ''})\n"
        f"OUTPUT_NODES = ({terminals}{',' if len(terminal_ids) == 1 else ''})"
    )


def _build_function_body(
    pipeline: PrototypePipeline,
    order: list[str],
    source_param_names: dict[str, str],
    function_name: str,
) -> tuple[list[str], list[str]]:
    source_ids = [
        node_id
        for node_id in order
        if not NODE_LIBRARY_BY_ID[pipeline.nodes[node_id].operation_id].has_input
    ]
    params = []
    for node_id in source_ids:
        param = source_param_names[node_id]
        params.append(f"{param}=None")
    params.extend(
        (
            "*",
            "input_metadata=None",
            "input_name=''",
            "source_metadata=None",
            "source_names=None",
            "source_image_states=None",
            "source_payloads=None",
        )
    )
    signature = ", ".join(params)

    lines = [f"def {function_name}({signature}):"]
    lines.extend(
        (
            f'{_INDENT}"""Execute the workflow through VIPP\'s shared executor.',
            "",
            f"{_INDENT}Raw arrays use only metadata supplied to this call.  Passing an",
            f"{_INDENT}ImageDataset (as returned by load_image) or SourcePayload",
            f"{_INDENT}carries the complete normalized ImageState into scientific",
            f"{_INDENT}operations.",
            f'{_INDENT}"""',
            f"{_INDENT}pipeline = _new_pipeline()",
            f"{_INDENT}provided = dict(source_payloads or {{}})",
            f"{_INDENT}unknown_sources = set(provided) - set(SOURCE_NODES)",
            f"{_INDENT}if unknown_sources:",
            f"{_INDENT}{_INDENT}raise ValueError(",
            f"{_INDENT}{_INDENT}{_INDENT}f'Unknown exported source nodes: '",
            f"{_INDENT}{_INDENT}{_INDENT}f'{{sorted(unknown_sources)!r}}.'",
            f"{_INDENT}{_INDENT})",
        )
    )

    missing: list[str] = []
    source_ids = list(source_param_names)
    for index, node_id in enumerate(source_ids):
        positional = source_param_names[node_id]
        metadata = "input_metadata" if index == 0 else "None"
        name = "input_name" if index == 0 else "''"
        lines.extend(
            (
                f"{_INDENT}if {positional} is not None and {node_id!r} in provided:",
                f"{_INDENT}{_INDENT}raise ValueError(",
                f"{_INDENT}{_INDENT}{_INDENT}"
                f"'Source node {node_id} was supplied both positionally and '",
                f"{_INDENT}{_INDENT}{_INDENT}'in source_payloads.'",
                f"{_INDENT}{_INDENT})",
                f"{_INDENT}value = provided.get({node_id!r}, {positional})",
                f"{_INDENT}metadata = _source_override(",
                f"{_INDENT}{_INDENT}source_metadata, {node_id!r}, {metadata}",
                f"{_INDENT})",
                f"{_INDENT}name = _source_override(",
                f"{_INDENT}{_INDENT}source_names, {node_id!r}, {name}",
                f"{_INDENT})",
                f"{_INDENT}image_state = _source_override(",
                f"{_INDENT}{_INDENT}source_image_states, {node_id!r}",
                f"{_INDENT})",
                f"{_INDENT}provided[{node_id!r}] = _coerce_source_payload(",
                f"{_INDENT}{_INDENT}value,",
                f"{_INDENT}{_INDENT}node_id={node_id!r},",
                f"{_INDENT}{_INDENT}metadata=metadata,",
                f"{_INDENT}{_INDENT}name=name,",
                f"{_INDENT}{_INDENT}image_state=image_state,",
                f"{_INDENT})",
            )
        )

    for node_id in order:
        node = pipeline.nodes[node_id]
        spec = NODE_LIBRARY_BY_ID[node.operation_id]
        if spec.is_multi_output and node_id in _terminal_nodes(pipeline, order):
            missing.append(
                f"{node.title} ({node_id}) produces multiple outputs but nothing "
                "consumes them; connect its output ports so the script can route "
                "each channel."
            )
    if source_ids:
        primary = source_ids[0]
        lines.extend(
            (
                f"{_INDENT}primary = provided[{primary!r}]",
                f"{_INDENT}values = pipeline.run(",
                f"{_INDENT}{_INDENT}primary.data,",
                f"{_INDENT}{_INDENT}input_metadata=primary.metadata,",
                f"{_INDENT}{_INDENT}input_name=primary.name,",
                f"{_INDENT}{_INDENT}source_payloads=provided,",
                f"{_INDENT})",
            )
        )
    else:
        lines.append(f"{_INDENT}values = pipeline.run(None)")
    lines.append(
        f"{_INDENT}return PipelineResults(values, pipeline.output_states)"
    )
    return lines, _dedupe(missing)


def _build_helpers() -> str:
    return '''class PipelineResults(dict):
    """Node outputs with the corresponding scientific output states."""

    def __init__(self, values, image_states):
        super().__init__(values)
        self.output_states = dict(image_states)
        self.image_states = self.output_states


def _new_pipeline():
    """Build and validate a fresh executor from the immutable snapshot."""
    if VIPP_VERSION != EXPECTED_VIPP_VERSION:
        raise RuntimeError(
            "This workflow was exported with napari-vipp "
            f"{EXPECTED_VIPP_VERSION}, but the active runtime is {VIPP_VERSION}. "
            "Use the recorded version or deliberately re-export and revalidate "
            "the workflow before relying on its results."
        )
    return pipeline_from_workflow(json.loads(_WORKFLOW_JSON))


def _source_override(mapping, node_id, default=None):
    if mapping is None:
        return default
    if not isinstance(mapping, dict):
        raise TypeError("Per-source overrides must be mappings keyed by node id.")
    return mapping.get(node_id, default)


def _dataset_metadata(dataset):
    metadata = {}
    uri = str(getattr(getattr(dataset, "inspection", None), "uri", "") or "")
    if uri:
        metadata["vipp_source_path"] = uri
    provenance = getattr(dataset, "provenance", None)
    if isinstance(provenance, dict) and provenance:
        metadata["vipp_source_provenance"] = dict(provenance)
    return metadata or None


def _coerce_source_payload(
    value,
    *,
    node_id,
    metadata=None,
    name="",
    image_state=None,
):
    if value is None:
        raise ValueError(
            f"Source node {node_id!r} has no input. Pass an array, ImageDataset, "
            "or SourcePayload for every exported source node."
        )
    if isinstance(value, SourcePayload):
        return SourcePayload(
            value.data,
            value.metadata if metadata is None else metadata,
            name or value.name,
            value.image_state if image_state is None else image_state,
            value.revision_token,
        )
    if isinstance(value, ImageDataset):
        selected = getattr(value, "selected_series", None)
        return SourcePayload(
            value.data,
            _dataset_metadata(value) if metadata is None else metadata,
            name or str(getattr(selected, "name", "") or ""),
            value.image_state if image_state is None else image_state,
        )
    return SourcePayload(value, metadata, name, image_state)


def load_image(path):
    """Load data and its complete normalized ImageState."""
    return read_image(path)


def save_image(data, path, *, image_state=None):
    """Write an image with its state, or write a table node output."""
    if is_table_data(data):
        save_table_output(
            data,
            _table_output_path(path),
            overwrite=True,
        )
        return
    write_image(data, path, overwrite=True, image_state=image_state)


def _table_output_path(path):
    path = Path(path)
    if path.suffix.lower() in {".csv", ".tsv"}:
        return path
    return path.with_suffix(".csv")'''


def _build_main(source_ids: list[str], function_name: str) -> str:
    primary = source_ids[0] if source_ids else None
    feed = "load_image(in_path)" if primary else ""
    batch_feed = "load_image(source_path)" if primary else ""
    return (
        'def batch_process(input_dir, output_dir, pattern="*.tif"):\n'
        f'{_INDENT}"""Run the pipeline over every matching file in a folder."""\n'
        f"{_INDENT}input_dir = Path(input_dir)\n"
        f"{_INDENT}output_dir = Path(output_dir)\n"
        f"{_INDENT}output_dir.mkdir(parents=True, exist_ok=True)\n"
        f"{_INDENT}for source_path in sorted(input_dir.glob(pattern)):\n"
        f"{_INDENT}{_INDENT}results = {function_name}({batch_feed})\n"
        f"{_INDENT}{_INDENT}for name in OUTPUT_NODES:\n"
        f"{_INDENT}{_INDENT}{_INDENT}output = results.get(name)\n"
        f"{_INDENT}{_INDENT}{_INDENT}if output is None:\n"
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}continue\n"
        f"{_INDENT}{_INDENT}{_INDENT}save_image(\n"
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}output,\n"
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}"
        f'output_dir / f"{{source_path.stem}}__{{name}}.ome.tif",\n'
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}"
        f"image_state=results.image_states.get(name),\n"
        f"{_INDENT}{_INDENT}{_INDENT})\n\n\n"
        'if __name__ == "__main__":\n'
        f"{_INDENT}import argparse\n\n"
        f"{_INDENT}parser = argparse.ArgumentParser(\n"
        f'{_INDENT}{_INDENT}description="Run the exported napari-vipp pipeline."\n'
        f"{_INDENT})\n"
        f"{_INDENT}parser.add_argument(\n"
        f'{_INDENT}{_INDENT}"input", help="Image file or folder (batch mode)."\n'
        f"{_INDENT})\n"
        f'{_INDENT}parser.add_argument("output", help="Output file or folder.")\n'
        f"{_INDENT}parser.add_argument(\n"
        f'{_INDENT}{_INDENT}"--pattern", default="*.tif",'
        f' help="Glob used in folder mode."\n'
        f"{_INDENT})\n"
        f"{_INDENT}args = parser.parse_args()\n\n"
        f"{_INDENT}in_path = Path(args.input)\n"
        f"{_INDENT}if in_path.is_dir():\n"
        f"{_INDENT}{_INDENT}batch_process(in_path, args.output, pattern=args.pattern)\n"
        f"{_INDENT}else:\n"
        f"{_INDENT}{_INDENT}results = {function_name}({feed})\n"
        f"{_INDENT}{_INDENT}out_path = Path(args.output)\n"
        f"{_INDENT}{_INDENT}if len(OUTPUT_NODES) == 1:\n"
        f"{_INDENT}{_INDENT}{_INDENT}save_image(\n"
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}results[OUTPUT_NODES[0]],\n"
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}out_path,\n"
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}"
        f"image_state=results.image_states.get(OUTPUT_NODES[0]),\n"
        f"{_INDENT}{_INDENT}{_INDENT})\n"
        f"{_INDENT}{_INDENT}else:\n"
        f"{_INDENT}{_INDENT}{_INDENT}out_path.mkdir(parents=True, exist_ok=True)\n"
        f"{_INDENT}{_INDENT}{_INDENT}for name in OUTPUT_NODES:\n"
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}output = results.get(name)\n"
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}if output is not None:\n"
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}{_INDENT}save_image(\n"
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}{_INDENT}{_INDENT}output,\n"
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}{_INDENT}{_INDENT}"
        f'out_path / f"{{in_path.stem}}__{{name}}.ome.tif",\n'
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}{_INDENT}{_INDENT}"
        f"image_state=results.image_states.get(name),\n"
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}{_INDENT})"
    )


def _terminal_nodes(pipeline: PrototypePipeline, order: list[str]) -> list[str]:
    explicit = [
        node_id
        for node_id in order
        if pipeline.nodes[node_id].operation_id == "batch_output"
    ]
    if explicit:
        return explicit
    consumed = {connection.source_id for connection in pipeline.connections}
    terminals = [node_id for node_id in order if node_id not in consumed]
    return terminals or list(order)


def _used_function_names(pipeline: PrototypePipeline, order: list[str]) -> list[str]:
    names = []
    for node_id in order:
        spec = NODE_LIBRARY_BY_ID[pipeline.nodes[node_id].operation_id]
        if spec.function is not None:
            names.append(spec.function.__name__)
    return names


def _unique_names(node_ids: list[str], *, prefix: str) -> dict[str, str]:
    names: dict[str, str] = {}
    used: set[str] = set()
    for node_id in node_ids:
        base = f"{prefix}_{_identifier(node_id)}"
        candidate = base
        suffix = 2
        while candidate in used:
            candidate = f"{base}_{suffix}"
            suffix += 1
        names[node_id] = candidate
        used.add(candidate)
    return names


def _identifier(node_id: str) -> str:
    cleaned = re.sub(r"\W", "_", node_id)
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"n_{cleaned}"
    return cleaned


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
