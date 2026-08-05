"""Generate a runnable, headless Python script from a pipeline graph.

Exported scripts execute the serialized workflow through the same compute
planner and execution service as interactive and durable batch runs.  Keeping
that shared path in the generated program is important: operation calls alone
cannot reproduce backend selection, fallback policy, axis validation,
physical-grid checks, metadata-derived runtime arguments, or output
:class:`~napari_vipp.core.metadata.ImageState` updates that are part of a
scientific VIPP workflow.
"""

from __future__ import annotations

import builtins
import json
import keyword
import re
from collections.abc import Mapping
from datetime import UTC, datetime

from napari_vipp import __version__ as VIPP_VERSION
from napari_vipp.core.batch import scientific_workflow_hash
from napari_vipp.core.compute import ComputeRequest, canonical_digest
from napari_vipp.core.pipeline import (
    NODE_LIBRARY_BY_ID,
    PrototypePipeline,
)
from napari_vipp.core.workflow import deserialize_workflow, serialize_workflow

_INDENT = " " * 4
_RESERVED_FUNCTION_NAMES = {
    "EXPECTED_VIPP_VERSION",
    "GENERATED_TEMPLATE_FINGERPRINT",
    "ImageDataset",
    "PipelineCancellationError",
    "PipelineExecutionError",
    "OUTPUT_NODES",
    "Path",
    "PipelineResults",
    "SOURCE_NODES",
    "SourcePayload",
    "VIPP_VERSION",
    "WORKFLOW_SHA256",
    "_GENERATED_SOURCE_SHA256",
    "_RUN_IDS",
    "_WORKFLOW_JSON",
    "_build_export_provenance",
    "_build_failure_provenance",
    "_build_publication_failure_provenance",
    "_atomic_publish_artifacts",
    "_cli_compute_request",
    "_cleanup_publication_path",
    "_coerce_source_payload",
    "_dataset_metadata",
    "_effective_execution_fingerprint",
    "_ensure_cli_failure_provenance",
    "_execution_progress_callbacks",
    "_generated_source_sha256",
    "_effective_compute_request",
    "_new_pipeline",
    "_node_preference_overrides",
    "_progress_printer",
    "_provenance_sidecar_path",
    "_publish_output_set",
    "_raise_if_cancelled",
    "_report_generated_progress",
    "_source_provenance_records",
    "_source_override",
    "_save_cli_failure_provenance",
    "_table_output_path",
    "_workflow_document",
    "_write_output_uncommitted",
    "argparse",
    "batch_process",
    "is_table_data",
    "json",
    "load_image",
    "main",
    "pipeline_from_workflow",
    "read_image",
    "save_provenance_sidecar",
    "save_image",
    "save_table_output",
    "write_image",
    "ComputeRequest",
    "Mapping",
    "OperationCancelled",
    "PipelineRunRequest",
    "atomic_write_json",
    "atomic_replace",
    "canonical_digest",
    "count",
    "deserialize_workflow",
    "execute_pipeline_request",
    "execution_provenance_digest",
    "hashlib",
    "serialize_execution_provenance",
    "scientific_workflow_hash",
    "signal",
    "shutil",
    "sys",
    "tempfile",
    "threading",
    "warnings",
}


def export_pipeline_to_python(
    pipeline: PrototypePipeline,
    *,
    function_name: str = "run_pipeline",
    compute_request: ComputeRequest | Mapping[str, object] | None = None,
) -> str:
    """Return Python source code that reproduces the pipeline headlessly.

    Workflow-v4 compute intent is embedded for lossless portability.  ``None``
    uses that authored request; callers and the generated CLI may supply an
    explicit run-only override without mutating the embedded workflow.
    """
    if (
        not function_name.isidentifier()
        or keyword.iskeyword(function_name)
        or function_name in dir(builtins)
    ):
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
    workflow = _build_workflow_constant(
        pipeline,
        compute_request=compute_request,
        function_name=function_name,
    )
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
import json
import signal
import sys
import threading
from pathlib import Path

from napari_vipp.core.batch import (
    load_batch_config,
    run_batch,
)
from napari_vipp.core.compute import ComputeRequest
from napari_vipp.core.compute_history import default_pipeline_timing_history_path
from napari_vipp.core.workflow import deserialize_workflow


def _node_preference_overrides(values):
    preferences = {}
    for raw_value in values or ():
        value = str(raw_value).strip()
        if "=" not in value:
            raise ValueError(
                "Node preferences must use NODE_ID=PREFERENCE syntax."
            )
        node_id, preference = (part.strip() for part in value.split("=", 1))
        if not node_id or not preference:
            raise ValueError(
                "Node preferences must name a node and a non-empty preference."
            )
        if node_id in preferences:
            raise ValueError(f"Duplicate node preference override for {node_id!r}.")
        preferences[node_id] = preference
    return preferences


def _load_run_inputs(args):
    config_path = Path(args.config).expanduser().resolve()
    config = load_batch_config(config_path)
    workflow_path = (
        Path(args.workflow).expanduser().resolve()
        if args.workflow
        else config.resolve_path(config.workflow_file).resolve()
    )
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    restored = deserialize_workflow(workflow)
    return config_path, config, workflow_path, workflow, restored


def _compute_override(args, config, restored):
    if not (
        args.compute_mode
        or args.fallback_policy
        or args.node_preference
    ):
        return None
    base = config.compute_request
    preferences = dict(base.node_preferences)
    preferences.update(_node_preference_overrides(args.node_preference))
    effective_mode = base.mode if args.compute_mode is None else args.compute_mode
    effective_fallback = (
        base.fallback_policy
        if args.fallback_policy is None
        else args.fallback_policy
    )
    if args.compute_mode == "prefer_gpu" and args.fallback_policy is None:
        effective_fallback = "visible"
    override = ComputeRequest(
        mode=effective_mode,
        node_preferences=preferences,
        fallback_policy=effective_fallback,
        runtime_id=base.runtime_id,
        device_id=base.device_id,
        precision_policy_id=base.precision_policy_id,
        workload_policy_id=base.workload_policy_id,
        accelerator_memory_cap_bytes=base.accelerator_memory_cap_bytes,
        accelerator_safety_reserve_bytes=(
            base.accelerator_safety_reserve_bytes
        ),
        allow_experimental=base.allow_experimental,
    )
    node_ids = {node.id for node in restored["nodes"]}
    unknown = set(override.node_preferences) - node_ids
    if unknown:
        raise ValueError(
            "Compute preferences reference unknown workflow nodes: "
            f"{sorted(unknown)!r}."
        )
    return override


def _progress(current, total, item_id, message):
    suffix = f" - {message}" if message else ""
    print(f"[{item_id}] {current}/{total}{suffix}", flush=True)


def _execution_progress(update):
    suffix = f" - {update.message}" if update.message else ""
    operation = update.operation_id or update.node_id
    print(
        f"[{update.item_index}/{update.item_total} {update.batch_id}] "
        f"{operation}: {update.current}/{update.total}{suffix}",
        flush=True,
    )


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
    parser.add_argument(
        "--compute-mode",
        choices=("cpu", "auto", "prefer_gpu", "custom"),
        help="Override the saved compute mode for this run.",
    )
    parser.add_argument(
        "--fallback-policy",
        choices=("visible", "strict"),
        help="Override visible versus strict CPU fallback.",
    )
    parser.add_argument(
        "--node-preference",
        action="append",
        default=[],
        metavar="NODE_ID=PREFERENCE",
        help="Repeat for per-node compute overrides.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print batch item progress.",
    )
    args = parser.parse_args(argv)
    try:
        (
            config_path,
            config,
            workflow_path,
            workflow,
            restored,
        ) = _load_run_inputs(args)
        compute_request = _compute_override(args, config, restored)
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    cancel_event = threading.Event()
    previous_sigint = signal.getsignal(signal.SIGINT)

    def request_cancel(_signum, _frame):
        if cancel_event.is_set():
            raise KeyboardInterrupt
        cancel_event.set()
        print(
            "Cancellation requested; waiting for the active operation to clean up.",
            file=sys.stderr,
        )

    signal.signal(signal.SIGINT, request_cancel)
    try:
        result = run_batch(
            workflow,
            config,
            workflow_path=workflow_path,
            config_path=config_path,
            compute_request=compute_request,
            cancel_event=cancel_event,
            progress_callback=_progress if args.progress else None,
            execution_progress_callback=(
                _execution_progress if args.progress else None
            ),
            performance_history_path=default_pipeline_timing_history_path(),
        )
    except Exception as exc:
        print(f"Batch failed before or during execution: {exc}", file=sys.stderr)
        return 2
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
    summary = result.summary
    print(
        f"{summary['completed']} completed, "
        f"{summary['partial']} partial, "
        f"{summary['skipped']} skipped, "
        f"{summary['cancelled']} cancelled, "
        f"{summary['failed']} failed; "
        f"{len(result.saved_paths)} outputs saved; "
        f"manifest: {result.manifest_path}"
    )
    if result.cancelled:
        return 130
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
        "This generated convenience script records compute and output "
        "provenance, but does not hash source bytes or provide durable batch "
        "resume guarantees.\n\n"
        "Run a single image:\n"
        "    python this_script.py input.tif output.tif\n\n"
        "Convenience folder loop (not durable VIPP batch execution):\n"
        "    python this_script.py input_dir/ output_dir/ --pattern '*.tif'\n"
        '"""\n'
        "from __future__ import annotations"
    )


def _build_imports() -> str:
    return "\n".join(
        (
            "import hashlib",
            "import json",
            "import signal",
            "import shutil",
            "import sys",
            "import tempfile",
            "import threading",
            "import warnings",
            "from collections.abc import Mapping",
            "from itertools import count",
            "from pathlib import Path",
            "",
            "from napari_vipp import __version__ as VIPP_VERSION",
            "from napari_vipp.core.atomic_io import atomic_replace, atomic_write_json",
            "from napari_vipp.core.batch import scientific_workflow_hash",
            "from napari_vipp.core.batch_setup import pipeline_from_workflow",
            "from napari_vipp.core.compute import ComputeRequest, canonical_digest",
            (
                "from napari_vipp.core.compute_history import "
                "default_pipeline_timing_history_path"
            ),
            "from napari_vipp.core.execution import (",
            "    PipelineRunRequest,",
            "    execute_pipeline_request,",
            ")",
            "from napari_vipp.core.execution_provenance import (",
            "    execution_provenance_digest,",
            "    serialize_execution_provenance,",
            ")",
            "from napari_vipp.core.io import ImageDataset, read_image, write_image",
            "from napari_vipp.core.pipeline import SourcePayload",
            "from napari_vipp.core.progress import OperationCancelled",
            "from napari_vipp.core.tables import is_table_data, save_table_output",
            "from napari_vipp.core.workflow import deserialize_workflow",
        )
    )


def _build_workflow_constant(
    pipeline: PrototypePipeline,
    *,
    compute_request: ComputeRequest | Mapping[str, object] | None = None,
    function_name: str,
) -> str:
    """Embed one immutable, validated workflow snapshot.

    A JSON string, rather than a live dict literal, prevents one run (or caller)
    from mutating the graph used by later invocations.  ``pipeline_from_workflow``
    deserializes and validates a fresh document on every call.
    """
    document = serialize_workflow(pipeline, compute_request=compute_request)
    deserialize_workflow(document)
    encoded = json.dumps(
        document,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    workflow_sha256 = scientific_workflow_hash(document)
    template_fingerprint = canonical_digest(
        {
            "schema_id": "napari-vipp-generated-python-template-v1",
            "vipp_version": VIPP_VERSION,
            "function_name": function_name,
            "workflow_sha256": workflow_sha256,
        }
    )
    return (
        f"EXPECTED_VIPP_VERSION = {VIPP_VERSION!r}\n"
        f"WORKFLOW_SHA256 = {workflow_sha256!r}\n"
        f"GENERATED_TEMPLATE_FINGERPRINT = {template_fingerprint!r}\n"
        f"_WORKFLOW_JSON = {encoded!r}\n"
        "try:\n"
        "    _GENERATED_SOURCE_SHA256 = hashlib.sha256(\n"
        "        Path(__file__).read_bytes()\n"
        "    ).hexdigest()\n"
        "except (NameError, OSError):\n"
        "    _GENERATED_SOURCE_SHA256 = ''"
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
            "compute_request=None",
            "progress_callback=None",
            "cancel_event=None",
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
            f"{_INDENT}operations.  This convenience API does not hash source bytes;",
            f"{_INDENT}use the saved batch runner for durable source identity.",
            f'{_INDENT}"""',
            f"{_INDENT}document = _workflow_document()",
            f"{_INDENT}authored_pipeline = pipeline_from_workflow(document)",
            f"{_INDENT}effective_request = _effective_compute_request(",
            f"{_INDENT}{_INDENT}document, compute_request",
            f"{_INDENT})",
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
                f"{_INDENT}primary_data = primary.data",
                f"{_INDENT}primary_metadata = primary.metadata",
                f"{_INDENT}primary_name = primary.name",
            )
        )
    else:
        lines.extend(
            (
                f"{_INDENT}primary_data = None",
                f"{_INDENT}primary_metadata = None",
                f"{_INDENT}primary_name = ''",
            )
        )
    lines.extend(
        (
            f"{_INDENT}source_records = _source_provenance_records(provided)",
            f"{_INDENT}(",
            f"{_INDENT}{_INDENT}node_started_callback,",
            f"{_INDENT}{_INDENT}node_finished_callback,",
            f"{_INDENT}{_INDENT}operation_progress_callback,",
            f"{_INDENT}) = _execution_progress_callbacks(",
            f"{_INDENT}{_INDENT}authored_pipeline, progress_callback",
            f"{_INDENT})",
            f"{_INDENT}request = PipelineRunRequest(",
            f"{_INDENT}{_INDENT}run_id=next(_RUN_IDS),",
            f"{_INDENT}{_INDENT}workflow=document,",
            f"{_INDENT}{_INDENT}input_data=primary_data,",
            f"{_INDENT}{_INDENT}input_metadata=primary_metadata,",
            f"{_INDENT}{_INDENT}input_name=primary_name,",
            f"{_INDENT}{_INDENT}source_payloads=provided,",
            f"{_INDENT}{_INDENT}compute_request=effective_request,",
            f"{_INDENT}{_INDENT}manual_node_ids=frozenset(",
            f"{_INDENT}{_INDENT}{_INDENT}authored_pipeline.manual_node_ids()",
            f"{_INDENT}{_INDENT}),",
            f"{_INDENT}{_INDENT}cancel_event=cancel_event,",
            f"{_INDENT}{_INDENT}performance_history_path=(",
            f"{_INDENT}{_INDENT}{_INDENT}default_pipeline_timing_history_path()",
            f"{_INDENT}{_INDENT}),",
            f"{_INDENT})",
            f"{_INDENT}try:",
            f"{_INDENT}{_INDENT}run_result = execute_pipeline_request(",
            f"{_INDENT}{_INDENT}{_INDENT}request,",
            f"{_INDENT}{_INDENT}{_INDENT}"
            f"node_started_callback=node_started_callback,",
            f"{_INDENT}{_INDENT}{_INDENT}"
            f"node_finished_callback=node_finished_callback,",
            f"{_INDENT}{_INDENT}{_INDENT}"
            f"progress_callback=operation_progress_callback,",
            f"{_INDENT}{_INDENT}{_INDENT}raise_errors=True,",
            f"{_INDENT}{_INDENT})",
            f"{_INDENT}except Exception as exc:",
            f"{_INDENT}{_INDENT}failure = getattr(",
            f"{_INDENT}{_INDENT}{_INDENT}exc, 'vipp_execution_failure', None",
            f"{_INDENT}{_INDENT})",
            f"{_INDENT}{_INDENT}if failure is None:",
            f"{_INDENT}{_INDENT}{_INDENT}failure = {{",
            f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}'kind': 'execution_error',",
            f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}"
            f"'error_type': type(exc).__name__,",
            f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}'message': str(exc),",
            f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}"
            f"'reason_code': 'unclassified_execution_error',",
            f"{_INDENT}{_INDENT}{_INDENT}}}",
            f"{_INDENT}{_INDENT}try:",
            f"{_INDENT}{_INDENT}{_INDENT}exc.provenance = (",
            f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}_build_failure_provenance(",
            f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}{_INDENT}effective_request,",
            f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}{_INDENT}failure,",
            f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}{_INDENT}source_records,",
            f"{_INDENT}{_INDENT}{_INDENT}{_INDENT})",
            f"{_INDENT}{_INDENT}{_INDENT})",
            f"{_INDENT}{_INDENT}except Exception:",
            f"{_INDENT}{_INDENT}{_INDENT}pass",
            f"{_INDENT}{_INDENT}raise",
            f"{_INDENT}if run_result.cancelled:",
            f"{_INDENT}{_INDENT}raise PipelineCancellationError(",
            f"{_INDENT}{_INDENT}{_INDENT}run_result.error or 'Operation cancelled.',",
            f"{_INDENT}{_INDENT}{_INDENT}failure=run_result.failure,",
            f"{_INDENT}{_INDENT}{_INDENT}provenance=_build_failure_provenance(",
            f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}effective_request,",
            f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}run_result.failure,",
            f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}source_records,",
            f"{_INDENT}{_INDENT}{_INDENT}),",
            f"{_INDENT}{_INDENT})",
            f"{_INDENT}if run_result.error:",
            f"{_INDENT}{_INDENT}raise PipelineExecutionError(",
            f"{_INDENT}{_INDENT}{_INDENT}run_result.error,",
            f"{_INDENT}{_INDENT}{_INDENT}execution_report=run_result.execution_report,",
            f"{_INDENT}{_INDENT}{_INDENT}failure=run_result.failure,",
            f"{_INDENT}{_INDENT}{_INDENT}provenance=_build_failure_provenance(",
            f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}effective_request,",
            f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}run_result.failure,",
            f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}source_records,",
            f"{_INDENT}{_INDENT}{_INDENT}),",
            f"{_INDENT}{_INDENT})",
            f"{_INDENT}pipeline = run_result.pipeline",
            f"{_INDENT}if pipeline is None:",
            f"{_INDENT}{_INDENT}raise PipelineExecutionError(",
            f"{_INDENT}{_INDENT}{_INDENT}'The shared executor returned no pipeline.'",
            f"{_INDENT}{_INDENT})",
            f"{_INDENT}executed_node_ids = tuple(",
            f"{_INDENT}{_INDENT}node_id",
            f"{_INDENT}{_INDENT}for node_id in pipeline.topological_order()",
            f"{_INDENT}{_INDENT}if node_id in pipeline.completed_node_ids",
            f"{_INDENT})",
            f"{_INDENT}if (",
            f"{_INDENT}{_INDENT}run_result.execution_report is not None",
            f"{_INDENT}{_INDENT}and not run_result.execution_report.cleanup_succeeded",
            f"{_INDENT}):",
            f"{_INDENT}{_INDENT}failure = {{",
            f"{_INDENT}{_INDENT}{_INDENT}'kind': 'cleanup_failure',",
            f"{_INDENT}{_INDENT}{_INDENT}"
            f"'error_type': 'PipelineRuntimeCleanupError',",
            f"{_INDENT}{_INDENT}{_INDENT}'message': (",
            f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}"
            f"'Accelerator cleanup could not be proven; outputs were withheld.'",
            f"{_INDENT}{_INDENT}{_INDENT}),",
            f"{_INDENT}{_INDENT}{_INDENT}"
            f"'reason_code': 'accelerator_cleanup_failed',",
            f"{_INDENT}{_INDENT}{_INDENT}'cleanup_succeeded': False,",
            f"{_INDENT}{_INDENT}}}",
            f"{_INDENT}{_INDENT}failure_execution = (",
            f"{_INDENT}{_INDENT}{_INDENT}serialize_execution_provenance(",
            f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}effective_request,",
            f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}pipeline,",
            f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}run_result.execution_report,",
            f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}"
            f"completed_node_ids=executed_node_ids,",
            f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}failure=failure,",
            f"{_INDENT}{_INDENT}{_INDENT})",
            f"{_INDENT}{_INDENT})",
            f"{_INDENT}{_INDENT}raise PipelineExecutionError(",
            f"{_INDENT}{_INDENT}{_INDENT}failure['message'],",
            f"{_INDENT}{_INDENT}{_INDENT}"
            f"execution_report=run_result.execution_report,",
            f"{_INDENT}{_INDENT}{_INDENT}failure=failure,",
            f"{_INDENT}{_INDENT}{_INDENT}provenance=_build_export_provenance(",
            f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}effective_request,",
            f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}failure_execution,",
            f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}source_records,",
            f"{_INDENT}{_INDENT}{_INDENT}),",
            f"{_INDENT}{_INDENT})",
            f"{_INDENT}execution_provenance = serialize_execution_provenance(",
            f"{_INDENT}{_INDENT}effective_request,",
            f"{_INDENT}{_INDENT}pipeline,",
            f"{_INDENT}{_INDENT}run_result.execution_report,",
            f"{_INDENT}{_INDENT}completed_node_ids=executed_node_ids,",
            f"{_INDENT})",
            f"{_INDENT}export_provenance = _build_export_provenance(",
            f"{_INDENT}{_INDENT}effective_request,",
            f"{_INDENT}{_INDENT}execution_provenance,",
            f"{_INDENT}{_INDENT}source_records,",
            f"{_INDENT})",
            f"{_INDENT}return PipelineResults(",
            f"{_INDENT}{_INDENT}pipeline.outputs,",
            f"{_INDENT}{_INDENT}pipeline.output_states,",
            f"{_INDENT}{_INDENT}execution_report=run_result.execution_report,",
            f"{_INDENT}{_INDENT}effective_compute_request=effective_request,",
            f"{_INDENT}{_INDENT}node_compute_provenance=",
            f"{_INDENT}{_INDENT}{_INDENT}pipeline.node_compute_provenance,",
            f"{_INDENT}{_INDENT}execution_provenance=execution_provenance,",
            f"{_INDENT}{_INDENT}provenance=export_provenance,",
            f"{_INDENT})",
        )
    )
    return lines, _dedupe(missing)


def _build_helpers() -> str:
    return '''_RUN_IDS = count(1)


class PipelineExecutionError(RuntimeError):
    """A generated run failed before it could publish scientific outputs."""

    def __init__(
        self,
        message,
        *,
        execution_report=None,
        failure=None,
        provenance=None,
    ):
        super().__init__(str(message))
        self.execution_report = execution_report
        self.failure = failure
        self.provenance = provenance


class PipelineCancellationError(OperationCancelled):
    """A generated run cooperatively cancelled after its cleanup boundary."""

    def __init__(self, message, *, failure=None, provenance=None):
        super().__init__(str(message))
        self.failure = failure
        self.provenance = provenance


class PipelineResults(dict):
    """Host outputs, scientific states, and exact execution provenance."""

    def __init__(
        self,
        values,
        image_states,
        *,
        execution_report,
        effective_compute_request,
        node_compute_provenance,
        execution_provenance,
        provenance,
    ):
        super().__init__(values)
        self.output_states = dict(image_states)
        self.image_states = self.output_states
        self.execution_report = execution_report
        self.effective_compute_request = ComputeRequest.from_dict(
            effective_compute_request.as_dict()
        )
        self.node_compute_provenance = dict(node_compute_provenance)
        self.execution_provenance = json.loads(
            json.dumps(execution_provenance, allow_nan=False)
        )
        self.provenance = json.loads(json.dumps(provenance, allow_nan=False))
        self.execution_provenance_sha256 = execution_provenance_digest(
            self.execution_provenance
        )
        self.workflow_sha256 = WORKFLOW_SHA256
        self.generated_template_fingerprint = GENERATED_TEMPLATE_FINGERPRINT
        self.generated_artifact_sha256 = self.provenance[
            "generated_artifact"
        ]["source_sha256"]
        self.effective_execution_fingerprint = self.provenance[
            "generated_artifact"
        ]["effective_execution_fingerprint"]
        self.output_provenance = {
            node_id: self.provenance_for(node_id)
            for node_id in OUTPUT_NODES
            if node_id in self
        }

    def provenance_for(self, node_id, *, output_port_index=0):
        """Bind the run record to one exact exported node output."""
        node_id = str(node_id).strip()
        if node_id not in self:
            raise KeyError(f"No exported result exists for node {node_id!r}.")
        if (
            isinstance(output_port_index, bool)
            or not isinstance(output_port_index, int)
            or output_port_index < 0
        ):
            raise ValueError("output_port_index must be a non-negative integer.")
        cached = self.node_compute_provenance.get(node_id)
        identity = getattr(cached, "actual_implementation", None)
        output_record = {
            "node_id": node_id,
            "output_port_index": output_port_index,
            "execution_provenance_sha256": self.execution_provenance_sha256,
            "compute_context_fingerprint": str(
                getattr(cached, "compute_context_fingerprint", "")
            ),
            "scientific_context_fingerprint": str(
                getattr(cached, "scientific_context_fingerprint", "")
            ),
            "result_context_fingerprint": str(
                getattr(cached, "result_context_fingerprint", "")
            ),
            "actual_implementation": (
                None
                if identity is None
                else {
                    "operation_id": identity.operation_id,
                    "runtime_id": identity.runtime_id,
                    "array_domain": identity.array_domain,
                    "implementation_library_id": (
                        identity.implementation_library_id
                    ),
                    "implementation_id": identity.implementation_id,
                    "implementation_version": identity.implementation_version,
                    "parity_policy_id": identity.parity_policy_id,
                    "cache_equivalence_group": identity.cache_equivalence_group,
                }
            ),
        }
        document = json.loads(json.dumps(self.provenance, allow_nan=False))
        document.pop("provenance_sha256", None)
        document["output"] = output_record
        document["provenance_sha256"] = canonical_digest(document)
        return document


def _workflow_document():
    """Return a fresh, validated copy of the immutable workflow snapshot."""
    if VIPP_VERSION != EXPECTED_VIPP_VERSION:
        raise RuntimeError(
            "This workflow was exported with napari-vipp "
            f"{EXPECTED_VIPP_VERSION}, but the active runtime is {VIPP_VERSION}. "
            "Use the recorded version or deliberately re-export and revalidate "
            "the workflow before relying on its results."
        )
    document = json.loads(_WORKFLOW_JSON)
    deserialize_workflow(document)
    observed_sha256 = scientific_workflow_hash(document)
    if observed_sha256 != WORKFLOW_SHA256:
        raise RuntimeError(
            "The embedded workflow failed its scientific integrity check. "
            "Re-export the pipeline instead of editing _WORKFLOW_JSON."
        )
    return document


def _new_pipeline():
    """Build and validate a fresh graph from the immutable snapshot."""
    return pipeline_from_workflow(_workflow_document())


def _effective_compute_request(document, override=None):
    """Resolve an explicit full-run override without mutating ``document``."""
    restored = deserialize_workflow(document)
    if override is None:
        request = restored["compute_request"]
    elif isinstance(override, ComputeRequest):
        request = ComputeRequest.from_dict(override.as_dict())
    elif isinstance(override, Mapping):
        request = ComputeRequest.from_dict(dict(override))
    else:
        raise TypeError("compute_request must be a ComputeRequest, mapping, or None.")
    node_ids = {node.id for node in restored["nodes"]}
    unknown = set(request.node_preferences) - node_ids
    if unknown:
        raise ValueError(
            "Compute preferences reference unknown exported nodes: "
            f"{sorted(unknown)!r}."
        )
    return request


def _node_preference_overrides(values):
    preferences = {}
    for raw_value in values or ():
        value = str(raw_value).strip()
        if "=" not in value:
            raise ValueError(
                "Node preferences must use NODE_ID=PREFERENCE syntax."
            )
        node_id, preference = (part.strip() for part in value.split("=", 1))
        if not node_id or not preference:
            raise ValueError(
                "Node preferences must name a node and a non-empty preference."
            )
        if node_id in preferences:
            raise ValueError(f"Duplicate node preference override for {node_id!r}.")
        preferences[node_id] = preference
    return preferences


def _cli_compute_request(
    *,
    mode=None,
    fallback_policy=None,
    node_preferences=(),
):
    document = _workflow_document()
    embedded = _effective_compute_request(document)
    preferences = dict(embedded.node_preferences)
    preferences.update(_node_preference_overrides(node_preferences))
    effective_mode = embedded.mode if mode is None else mode
    effective_fallback = (
        embedded.fallback_policy
        if fallback_policy is None
        else fallback_policy
    )
    if (
        getattr(effective_mode, "value", effective_mode) == "prefer_gpu"
        and fallback_policy is None
    ):
        effective_fallback = "visible"
    override = ComputeRequest(
        mode=effective_mode,
        node_preferences=preferences,
        fallback_policy=effective_fallback,
        runtime_id=embedded.runtime_id,
        device_id=embedded.device_id,
        precision_policy_id=embedded.precision_policy_id,
        workload_policy_id=embedded.workload_policy_id,
        accelerator_memory_cap_bytes=embedded.accelerator_memory_cap_bytes,
        accelerator_safety_reserve_bytes=(
            embedded.accelerator_safety_reserve_bytes
        ),
        allow_experimental=embedded.allow_experimental,
    )
    return _effective_compute_request(document, override)


def _generated_source_sha256():
    return _GENERATED_SOURCE_SHA256


def _effective_execution_fingerprint(effective_request):
    return canonical_digest(
        {
            "schema_id": "napari-vipp-generated-python-execution-v1",
            "generated_template_fingerprint": GENERATED_TEMPLATE_FINGERPRINT,
            "generated_source_sha256": _generated_source_sha256(),
            "effective_compute_request": effective_request.as_dict(),
        }
    )


def _build_export_provenance(
    effective_request,
    execution_provenance,
    source_records=(),
):
    authored = _effective_compute_request(_workflow_document())
    document = {
        "type": "napari-vipp-generated-execution-provenance",
        "version": 1,
        "vipp_version": VIPP_VERSION,
        "workflow": {
            "sha256": WORKFLOW_SHA256,
            "authored_compute_request": authored.as_dict(),
        },
        "generated_artifact": {
            "template_fingerprint": GENERATED_TEMPLATE_FINGERPRINT,
            "source_sha256": _generated_source_sha256(),
            "effective_execution_fingerprint": (
                _effective_execution_fingerprint(effective_request)
            ),
        },
        "sources": [dict(record) for record in source_records],
        "execution": execution_provenance,
    }
    document["provenance_sha256"] = canonical_digest(document)
    return document


def _build_failure_provenance(
    effective_request,
    failure,
    source_records=(),
):
    execution = serialize_execution_provenance(
        effective_request,
        None,
        None,
        failure=failure,
    )
    return _build_export_provenance(
        effective_request,
        execution,
        source_records,
    )


def _build_publication_failure_provenance(results, exc, *, cancelled=False):
    document = json.loads(json.dumps(results.provenance, allow_nan=False))
    document.pop("provenance_sha256", None)
    document["publication"] = {
        "outcome": "cancelled" if cancelled else "failed",
        "error_type": type(exc).__name__,
        "message": str(exc),
        "reason_code": (
            "output_publication_cancelled"
            if cancelled
            else "output_publication_failed"
        ),
        "node_id": str(getattr(exc, "vipp_output_node_id", "")),
        "path": str(getattr(exc, "vipp_output_path", "")),
        "fallback_used": False,
    }
    document["provenance_sha256"] = canonical_digest(document)
    return document


def _ensure_cli_failure_provenance(
    exc,
    effective_request,
    *,
    results=None,
    cancelled=False,
):
    existing = getattr(exc, "provenance", None)
    if isinstance(existing, Mapping):
        return existing
    if results is not None:
        document = _build_publication_failure_provenance(
            results,
            exc,
            cancelled=cancelled,
        )
    else:
        document = _build_failure_provenance(
            effective_request,
            {
                "kind": "cancelled" if cancelled else "execution_error",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "reason_code": (
                    "operation_cancelled"
                    if cancelled
                    else "unclassified_execution_error"
                ),
            },
        )
    try:
        exc.provenance = document
    except Exception:
        pass
    return document


def _provenance_sidecar_path(path):
    path = Path(path)
    return path.with_name(f"{path.name}.vipp-provenance.json")


def save_provenance_sidecar(path, provenance):
    """Atomically write provenance next to a successfully saved output."""
    document = getattr(provenance, "provenance", provenance)
    if not isinstance(document, Mapping):
        raise TypeError("provenance must be a mapping or PipelineResults.")
    return atomic_write_json(_provenance_sidecar_path(path), dict(document))


def _save_cli_failure_provenance(output, output_is_dir, exc, enabled):
    failure_provenance = getattr(exc, "provenance", None)
    if not enabled or not failure_provenance:
        return None
    failure_target = (
        Path(output) / "vipp-run-failure"
        if output_is_dir
        else Path(output)
    )
    if _provenance_sidecar_path(failure_target).exists():
        # Never replace exact provenance belonging to a pre-existing public
        # output with evidence from a failed retry/publication attempt.
        failure_target = failure_target.with_name(
            f"{failure_target.name}.vipp-run-failure"
        )
    try:
        return save_provenance_sidecar(failure_target, failure_provenance)
    except Exception as sidecar_error:
        print(
            f"Could not write failure provenance: {sidecar_error}",
            file=sys.stderr,
        )
        return None


def _progress_printer(node_id, current, total, message):
    suffix = f" - {message}" if message else ""
    if total:
        print(f"[{node_id}] {current}/{total}{suffix}", flush=True)
    else:
        print(f"[{node_id}] {current}{suffix}", flush=True)


def _report_generated_progress(
    callback,
    node_id,
    current,
    total,
    message,
):
    if callback is None:
        return
    try:
        callback(
            str(node_id),
            int(current),
            int(total),
            str(message),
        )
    except Exception:
        # Presentation hooks must never invalidate scientific execution or
        # accelerator cleanup/provenance finalization.
        return


def _execution_progress_callbacks(pipeline, callback):
    if callback is None:
        return None, None, None
    state = {"node_id": ""}

    def node_started(node_id):
        node_id = str(node_id)
        state["node_id"] = node_id
        node = pipeline.nodes.get(node_id)
        operation_id = "" if node is None else str(node.operation_id)
        message = (
            "Node started."
            if not operation_id
            else f"Node started ({operation_id})."
        )
        _report_generated_progress(callback, node_id, 0, 0, message)

    def node_finished(result):
        node_id = str(result.node_id)
        operation_id = str(result.operation_id)
        _report_generated_progress(
            callback,
            node_id,
            1,
            1,
            f"Node completed ({operation_id}).",
        )
        if state["node_id"] == node_id:
            state["node_id"] = ""

    def operation_progress(operation_id, current, total, message):
        operation_id = str(operation_id)
        node_id = state["node_id"] or operation_id
        detail = str(message) or f"Operation progress ({operation_id})."
        _report_generated_progress(
            callback,
            node_id,
            current,
            total,
            detail,
        )

    return node_started, node_finished, operation_progress


def _raise_if_cancelled(cancel_event, message):
    if cancel_event is not None and cancel_event.is_set():
        raise OperationCancelled(str(message))


def _source_provenance_records(payloads):
    records = []
    for node_id in SOURCE_NODES:
        payload = payloads.get(node_id)
        if payload is None:
            continue
        metadata = payload.metadata if isinstance(payload.metadata, Mapping) else {}
        source_provenance = metadata.get("vipp_source_provenance")
        source_identity = metadata.get("vipp_source_identity")
        if not isinstance(source_identity, Mapping) and isinstance(
            source_provenance,
            Mapping,
        ):
            candidate = source_provenance.get("identity")
            if isinstance(candidate, Mapping):
                source_identity = candidate
        try:
            provenance_document = (
                {}
                if not isinstance(source_provenance, Mapping)
                else json.loads(
                    json.dumps(dict(source_provenance), allow_nan=False)
                )
            )
            identity_document = (
                None
                if not isinstance(source_identity, Mapping)
                else json.loads(
                    json.dumps(dict(source_identity), allow_nan=False)
                )
            )
        except (TypeError, ValueError):
            provenance_document = {}
            identity_document = None
        record = {
            "node_id": str(node_id),
            "name": str(payload.name or ""),
            "path": str(metadata.get("vipp_source_path", "") or ""),
            "identity_complete": bool(
                isinstance(identity_document, Mapping)
                and identity_document.get("sha256")
            ),
            "identity": identity_document,
            "reader_provenance": provenance_document,
        }
        record["binding_sha256"] = canonical_digest(record)
        records.append(record)
    return records


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


def save_image(
    data,
    path,
    *,
    image_state=None,
    provenance=None,
    output_node_id=None,
    cancel_event=None,
):
    """Transactionally publish one output and its exact provenance sidecar."""
    saved_paths = _publish_output_set(
        ((output_node_id, data, path, image_state),),
        results=provenance,
        write_provenance=provenance is not None,
        cancel_event=cancel_event,
    )
    return saved_paths[0]


def _write_output_uncommitted(
    data,
    path,
    *,
    image_state=None,
    provenance=None,
    output_node_id=None,
):
    """Write only inside a private publication directory."""
    try:
        if is_table_data(data):
            saved_path = _table_output_path(path)
            save_table_output(
                data,
                saved_path,
                overwrite=True,
            )
        else:
            saved_path = write_image(
                data,
                path,
                overwrite=True,
                image_state=image_state,
            )
        if provenance is not None:
            document = (
                provenance.provenance_for(output_node_id)
                if output_node_id is not None
                and callable(getattr(provenance, "provenance_for", None))
                else provenance
            )
            save_provenance_sidecar(saved_path, document)
    except Exception as exc:
        try:
            exc.vipp_output_node_id = str(output_node_id or "")
            exc.vipp_output_path = str(locals().get("saved_path", path))
        except Exception:
            pass
        raise
    return saved_path


def _cleanup_publication_path(path):
    path = Path(path)
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)


def _atomic_publish_artifacts(artifacts, stage_root, cancel_event):
    """Promote one complete output set, rolling back caught failures."""
    backup_root = Path(stage_root) / "backups"
    backup_root.mkdir()
    backups = []
    promoted = []
    try:
        for index, (_staged, target) in enumerate(artifacts):
            _raise_if_cancelled(
                cancel_event,
                "Output publication cancelled before commit.",
            )
            target = Path(target)
            if target.exists() or target.is_symlink():
                backup = backup_root / f"{index:04d}"
                atomic_replace(target, backup)
                backups.append((backup, target))
        for staged, target in artifacts:
            _raise_if_cancelled(
                cancel_event,
                "Output publication cancelled during commit.",
            )
            target = Path(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            atomic_replace(Path(staged), target)
            promoted.append(target)
        _raise_if_cancelled(
            cancel_event,
            "Output publication cancelled during commit.",
        )
    except BaseException as publication_error:
        rollback_errors = []
        for target in reversed(promoted):
            try:
                _cleanup_publication_path(target)
            except Exception as exc:
                rollback_errors.append(exc)
        for backup, target in reversed(backups):
            try:
                _cleanup_publication_path(target)
                atomic_replace(backup, target)
            except Exception as exc:
                rollback_errors.append(exc)
        if rollback_errors:
            raise RuntimeError(
                "Output publication failed and its rollback could not be "
                "completed safely. Inspect the destination before retrying."
            ) from publication_error
        raise


def _publish_output_set(
    entries,
    *,
    results,
    write_provenance,
    cancel_event,
):
    """Stage, validate, and commit all outputs from one execution together."""
    entries = list(entries)
    if not entries:
        return ()
    requested_entries = []
    parents = set()
    for node_id, data, requested_path, image_state in entries:
        requested_path = Path(requested_path).expanduser()
        requested_path.parent.mkdir(parents=True, exist_ok=True)
        parent = requested_path.parent.resolve()
        parents.add(parent)
        normalized_node_id = None if node_id is None else str(node_id)
        requested_entries.append(
            (normalized_node_id, data, requested_path, image_state)
        )
    if len(parents) != 1:
        raise ValueError(
            "A transactional output set must use one destination directory."
        )
    target_parent = next(iter(parents))
    stage_root = Path(
        tempfile.mkdtemp(prefix=".vipp-publish-", dir=target_parent)
    )
    records = []
    active_node_id = ""
    active_requested_path = None
    try:
        for index, (
            node_id,
            data,
            requested_path,
            image_state,
        ) in enumerate(requested_entries):
            active_node_id = "" if node_id is None else node_id
            active_requested_path = requested_path
            _raise_if_cancelled(
                cancel_event,
                "Output publication cancelled before staging.",
            )
            entry_root = stage_root / f"{index:04d}"
            entry_root.mkdir()
            staged_request = entry_root / requested_path.name
            staged_saved = Path(
                _write_output_uncommitted(
                    data,
                    staged_request,
                    image_state=image_state,
                    provenance=results if write_provenance else None,
                    output_node_id=node_id,
                )
            ).resolve()
            if staged_saved.parent != entry_root.resolve():
                raise RuntimeError(
                    "The output writer returned a path outside its private "
                    "publication directory."
                )
            if not staged_saved.exists():
                raise RuntimeError(
                    "The output writer returned without creating its staged "
                    "artifact."
                )
            final_saved = requested_path.parent / staged_saved.name
            staged_sidecar = _provenance_sidecar_path(staged_saved)
            final_sidecar = _provenance_sidecar_path(final_saved)
            if write_provenance and not staged_sidecar.is_file():
                raise RuntimeError(
                    "Exact output provenance was not staged; the output was "
                    "withheld."
                )
            records.append(
                {
                    "node_id": node_id,
                    "staged_output": staged_saved,
                    "final_output": final_saved,
                    "staged_sidecar": staged_sidecar,
                    "final_sidecar": final_sidecar,
                }
            )
            _raise_if_cancelled(
                cancel_event,
                "Output publication cancelled after staging.",
            )
        report = getattr(results, "execution_report", None)
        if report is not None and not bool(
            getattr(report, "cleanup_succeeded", False)
        ):
            raise PipelineExecutionError(
                "Accelerator cleanup could not be proven; outputs were withheld.",
                execution_report=report,
                failure={
                    "kind": "cleanup_failure",
                    "error_type": "PipelineRuntimeCleanupError",
                    "reason_code": "accelerator_cleanup_failed",
                    "cleanup_succeeded": False,
                },
            )
        artifact_keys = []
        artifacts = []
        # Sidecars are promoted before their outputs. A process-level crash can
        # therefore leave an orphaned provenance record, but never a newly
        # published output that lacks the exact provenance requested for it.
        if write_provenance:
            for record in records:
                artifact_keys.append(
                    str(Path(record["final_sidecar"]).resolve()).casefold()
                )
                artifacts.append(
                    (record["staged_sidecar"], record["final_sidecar"])
                )
        for record in records:
            artifact_keys.append(
                str(Path(record["final_output"]).resolve()).casefold()
            )
            artifacts.append(
                (record["staged_output"], record["final_output"])
            )
        if len(artifact_keys) != len(set(artifact_keys)):
            raise ValueError(
                "Multiple exported outputs resolve to the same publication path."
            )
        _raise_if_cancelled(
            cancel_event,
            "Output publication cancelled before commit.",
        )
        _atomic_publish_artifacts(artifacts, stage_root, cancel_event)
        return tuple(Path(record["final_output"]) for record in records)
    except BaseException as exc:
        try:
            exc.vipp_output_node_id = active_node_id
            exc.vipp_output_path = str(active_requested_path or "")
        except Exception:
            pass
        raise
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)


def _table_output_path(path):
    path = Path(path)
    if path.suffix.lower() in {".csv", ".tsv"}:
        return path
    return path.with_suffix(".csv")'''


def _build_main(source_ids: list[str], function_name: str) -> str:
    primary = source_ids[0] if source_ids else None
    feed = "load_image(in_path)" if primary else ""
    batch_feed = "load_image(source_path)" if primary else ""
    feed_prefix = f"{feed}, " if feed else ""
    batch_feed_prefix = f"{batch_feed}, " if batch_feed else ""
    lines = [
        "def batch_process(",
        f"{_INDENT}input_dir,",
        f"{_INDENT}output_dir,",
        f'{_INDENT}pattern="*.tif",',
        f"{_INDENT}*,",
        f"{_INDENT}compute_request=None,",
        f"{_INDENT}progress_callback=None,",
        f"{_INDENT}cancel_event=None,",
        f"{_INDENT}write_provenance=True,",
        "):",
        f'{_INDENT}"""Run a convenience folder loop without durable batch resume."""',
        f"{_INDENT}warnings.warn(",
        f'{_INDENT}{_INDENT}"batch_process is a convenience loop, not VIPP durable "',
        f'{_INDENT}{_INDENT}"batch execution. Export and use the saved batch runner "',
        f'{_INDENT}{_INDENT}"for manifests, checkpoints, collision, and atomic-item "',
        f'{_INDENT}{_INDENT}"guarantees.",',
        f"{_INDENT}{_INDENT}FutureWarning,",
        f"{_INDENT}{_INDENT}stacklevel=2,",
        f"{_INDENT})",
        f"{_INDENT}input_dir = Path(input_dir)",
        f"{_INDENT}output_dir = Path(output_dir)",
        f"{_INDENT}output_dir.mkdir(parents=True, exist_ok=True)",
        f"{_INDENT}effective_request = _effective_compute_request(",
        f"{_INDENT}{_INDENT}_workflow_document(), compute_request",
        f"{_INDENT})",
        f"{_INDENT}records = []",
        f"{_INDENT}for item_index, source_path in enumerate(",
        f"{_INDENT}{_INDENT}sorted(input_dir.glob(pattern)), start=1",
        f"{_INDENT}):",
        f"{_INDENT}{_INDENT}results = None",
        f"{_INDENT}{_INDENT}output = None",
        f"{_INDENT}{_INDENT}saved_paths = []",
        f"{_INDENT}{_INDENT}try:",
        f"{_INDENT}{_INDENT}{_INDENT}_raise_if_cancelled(",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}cancel_event,",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}'Folder processing cancelled.'",
        f"{_INDENT}{_INDENT}{_INDENT})",
        f"{_INDENT}{_INDENT}{_INDENT}results = {function_name}(",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}{batch_feed_prefix}",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}compute_request=effective_request,",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}"
        f"progress_callback=progress_callback,",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}cancel_event=cancel_event,",
        f"{_INDENT}{_INDENT}{_INDENT})",
        f"{_INDENT}{_INDENT}{_INDENT}publication_entries = []",
        f"{_INDENT}{_INDENT}{_INDENT}for name in OUTPUT_NODES:",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}output = results.get(name)",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}if output is None:",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}{_INDENT}continue",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}publication_entries.append(",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}{_INDENT}(",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}{_INDENT}{_INDENT}name,",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}{_INDENT}{_INDENT}output,",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}{_INDENT}{_INDENT}"
        f'output_dir / f"{{source_path.stem}}__{{name}}.ome.tif",',
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}{_INDENT}{_INDENT}"
        f"results.image_states.get(name),",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}{_INDENT})",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT})",
        f"{_INDENT}{_INDENT}{_INDENT}saved_paths = list(",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}_publish_output_set(",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}{_INDENT}publication_entries,",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}{_INDENT}results=results,",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}{_INDENT}"
        f"write_provenance=write_provenance,",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}{_INDENT}"
        f"cancel_event=cancel_event,",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT})",
        f"{_INDENT}{_INDENT}{_INDENT})",
        f"{_INDENT}{_INDENT}except Exception as exc:",
        f"{_INDENT}{_INDENT}{_INDENT}failure_provenance = (",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}_ensure_cli_failure_provenance(",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}{_INDENT}exc,",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}{_INDENT}effective_request,",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}{_INDENT}results=results,",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}{_INDENT}cancelled=isinstance(",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}{_INDENT}{_INDENT}"
        f"exc, OperationCancelled",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}{_INDENT}),",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT})",
        f"{_INDENT}{_INDENT}{_INDENT})",
        f"{_INDENT}{_INDENT}{_INDENT}if write_provenance:",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}try:",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}{_INDENT}save_provenance_sidecar(",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}{_INDENT}{_INDENT}output_dir",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}{_INDENT}{_INDENT}"
        f"/ f'{{item_index:06d}}'",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}{_INDENT}{_INDENT}"
        f"f'__{{source_path.stem}}__vipp-run-failure',",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}{_INDENT}{_INDENT}failure_provenance,",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}{_INDENT})",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}except Exception:",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}{_INDENT}pass",
        f"{_INDENT}{_INDENT}{_INDENT}raise",
        f"{_INDENT}{_INDENT}records.append(",
        f"{_INDENT}{_INDENT}{_INDENT}{{",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}'source_path': str(source_path),",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}"
        f"'saved_paths': tuple(str(path) for path in saved_paths),",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}"
        f"'workflow_sha256': results.workflow_sha256,",
        f"{_INDENT}{_INDENT}{_INDENT}}}",
        f"{_INDENT}{_INDENT})",
        f"{_INDENT}{_INDENT}publication_entries = None",
        f"{_INDENT}{_INDENT}output = None",
        f"{_INDENT}{_INDENT}results = None",
        f"{_INDENT}return records",
        "",
        "",
        "def main(argv=None):",
        f"{_INDENT}import argparse",
        "",
        f"{_INDENT}parser = argparse.ArgumentParser(",
        f'{_INDENT}{_INDENT}description="Run the exported napari-vipp pipeline."',
        f"{_INDENT})",
        f"{_INDENT}parser.add_argument(",
        f'{_INDENT}{_INDENT}"input", help="Image file or convenience folder loop."',
        f"{_INDENT})",
        f'{_INDENT}parser.add_argument("output", help="Output file or folder.")',
        f"{_INDENT}parser.add_argument(",
        f'{_INDENT}{_INDENT}"--pattern", default="*.tif",',
        f'{_INDENT}{_INDENT}help="Glob used by the non-durable folder loop.",',
        f"{_INDENT})",
        f"{_INDENT}parser.add_argument(",
        f'{_INDENT}{_INDENT}"--compute-mode",',
        f'{_INDENT}{_INDENT}choices=("cpu", "auto", "prefer_gpu", "custom"),',
        f'{_INDENT}{_INDENT}help="Override the workflow compute mode for this run.",',
        f"{_INDENT})",
        f"{_INDENT}parser.add_argument(",
        f'{_INDENT}{_INDENT}"--fallback-policy",',
        f'{_INDENT}{_INDENT}choices=("visible", "strict"),',
        f'{_INDENT}{_INDENT}help="Override visible versus strict CPU fallback.",',
        f"{_INDENT})",
        f"{_INDENT}parser.add_argument(",
        f'{_INDENT}{_INDENT}"--node-preference",',
        f'{_INDENT}{_INDENT}action="append",',
        f'{_INDENT}{_INDENT}default=[],',
        f'{_INDENT}{_INDENT}metavar="NODE_ID=PREFERENCE",',
        f'{_INDENT}{_INDENT}help="Repeat for per-node compute overrides.",',
        f"{_INDENT})",
        f"{_INDENT}parser.add_argument(",
        f'{_INDENT}{_INDENT}"--provenance",',
        f'{_INDENT}{_INDENT}action=argparse.BooleanOptionalAction,',
        f'{_INDENT}{_INDENT}default=True,',
        f'{_INDENT}{_INDENT}help="Write an atomic .vipp-provenance.json sidecar.",',
        f"{_INDENT})",
        f"{_INDENT}parser.add_argument(",
        f'{_INDENT}{_INDENT}"--progress", action="store_true",',
        f'{_INDENT}{_INDENT}help="Print operation-level progress.",',
        f"{_INDENT})",
        f"{_INDENT}args = parser.parse_args(argv)",
        f"{_INDENT}try:",
        f"{_INDENT}{_INDENT}request = _cli_compute_request(",
        f"{_INDENT}{_INDENT}{_INDENT}mode=args.compute_mode,",
        f"{_INDENT}{_INDENT}{_INDENT}fallback_policy=args.fallback_policy,",
        f"{_INDENT}{_INDENT}{_INDENT}node_preferences=args.node_preference,",
        f"{_INDENT}{_INDENT})",
        f"{_INDENT}except (TypeError, ValueError) as exc:",
        f"{_INDENT}{_INDENT}parser.error(str(exc))",
        f"{_INDENT}progress = _progress_printer if args.progress else None",
        f"{_INDENT}in_path = Path(args.input)",
        f"{_INDENT}cancel_event = threading.Event()",
        f"{_INDENT}previous_sigint = signal.getsignal(signal.SIGINT)",
        "",
        f"{_INDENT}def request_cancel(_signum, _frame):",
        f"{_INDENT}{_INDENT}if cancel_event.is_set():",
        f"{_INDENT}{_INDENT}{_INDENT}raise KeyboardInterrupt",
        f"{_INDENT}{_INDENT}cancel_event.set()",
        f"{_INDENT}{_INDENT}print(",
        f'{_INDENT}{_INDENT}{_INDENT}"Cancellation requested; waiting for cleanup.",',
        f"{_INDENT}{_INDENT}{_INDENT}file=sys.stderr,",
        f"{_INDENT}{_INDENT})",
        "",
        f"{_INDENT}signal.signal(signal.SIGINT, request_cancel)",
        f"{_INDENT}results = None",
        f"{_INDENT}try:",
        f"{_INDENT}{_INDENT}if in_path.is_dir():",
        f"{_INDENT}{_INDENT}{_INDENT}batch_process(",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}in_path,",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}args.output,",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}pattern=args.pattern,",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}compute_request=request,",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}progress_callback=progress,",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}cancel_event=cancel_event,",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}"
        f"write_provenance=args.provenance,",
        f"{_INDENT}{_INDENT}{_INDENT})",
        f"{_INDENT}{_INDENT}{_INDENT}return 0",
        f"{_INDENT}{_INDENT}results = {function_name}(",
        f"{_INDENT}{_INDENT}{_INDENT}{feed_prefix}",
        f"{_INDENT}{_INDENT}{_INDENT}compute_request=request,",
        f"{_INDENT}{_INDENT}{_INDENT}progress_callback=progress,",
        f"{_INDENT}{_INDENT}{_INDENT}cancel_event=cancel_event,",
        f"{_INDENT}{_INDENT})",
        f"{_INDENT}{_INDENT}out_path = Path(args.output)",
        f"{_INDENT}{_INDENT}publication_entries = []",
        f"{_INDENT}{_INDENT}if len(OUTPUT_NODES) == 1:",
        f"{_INDENT}{_INDENT}{_INDENT}name = OUTPUT_NODES[0]",
        f"{_INDENT}{_INDENT}{_INDENT}publication_entries.append(",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}(",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}{_INDENT}name,",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}{_INDENT}results[name],",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}{_INDENT}out_path,",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}{_INDENT}"
        f"results.image_states.get(name),",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT})",
        f"{_INDENT}{_INDENT}{_INDENT})",
        f"{_INDENT}{_INDENT}else:",
        f"{_INDENT}{_INDENT}{_INDENT}for name in OUTPUT_NODES:",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}output = results.get(name)",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}if output is None:",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}{_INDENT}continue",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}publication_entries.append(",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}{_INDENT}(",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}{_INDENT}{_INDENT}name,",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}{_INDENT}{_INDENT}output,",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}{_INDENT}{_INDENT}"
        f'out_path / f"{{in_path.stem}}__{{name}}.ome.tif",',
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}{_INDENT}{_INDENT}"
        f"results.image_states.get(name),",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}{_INDENT})",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT})",
        f"{_INDENT}{_INDENT}_publish_output_set(",
        f"{_INDENT}{_INDENT}{_INDENT}publication_entries,",
        f"{_INDENT}{_INDENT}{_INDENT}results=results,",
        f"{_INDENT}{_INDENT}{_INDENT}write_provenance=args.provenance,",
        f"{_INDENT}{_INDENT}{_INDENT}cancel_event=cancel_event,",
        f"{_INDENT}{_INDENT})",
        f"{_INDENT}{_INDENT}return 0",
        f"{_INDENT}except OperationCancelled as exc:",
        f"{_INDENT}{_INDENT}_ensure_cli_failure_provenance(",
        f"{_INDENT}{_INDENT}{_INDENT}exc,",
        f"{_INDENT}{_INDENT}{_INDENT}request,",
        f"{_INDENT}{_INDENT}{_INDENT}results=results,",
        f"{_INDENT}{_INDENT}{_INDENT}cancelled=True,",
        f"{_INDENT}{_INDENT})",
        f"{_INDENT}{_INDENT}if not in_path.is_dir():",
        f"{_INDENT}{_INDENT}{_INDENT}_save_cli_failure_provenance(",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}args.output,",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}len(OUTPUT_NODES) != 1,",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}exc,",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}args.provenance,",
        f"{_INDENT}{_INDENT}{_INDENT})",
        f'{_INDENT}{_INDENT}print(f"Pipeline cancelled: {{exc}}", file=sys.stderr)',
        f"{_INDENT}{_INDENT}return 130",
        f"{_INDENT}except Exception as exc:",
        f"{_INDENT}{_INDENT}_ensure_cli_failure_provenance(",
        f"{_INDENT}{_INDENT}{_INDENT}exc, request, results=results",
        f"{_INDENT}{_INDENT})",
        f"{_INDENT}{_INDENT}if not in_path.is_dir():",
        f"{_INDENT}{_INDENT}{_INDENT}_save_cli_failure_provenance(",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}args.output,",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}len(OUTPUT_NODES) != 1,",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}exc,",
        f"{_INDENT}{_INDENT}{_INDENT}{_INDENT}args.provenance,",
        f"{_INDENT}{_INDENT}{_INDENT})",
        f'{_INDENT}{_INDENT}print(f"Pipeline failed: {{exc}}", file=sys.stderr)',
        f"{_INDENT}{_INDENT}return 2",
        f"{_INDENT}finally:",
        f"{_INDENT}{_INDENT}signal.signal(signal.SIGINT, previous_sigint)",
        "",
        "",
        'if __name__ == "__main__":',
        f"{_INDENT}raise SystemExit(main())",
    ]
    return "\n".join(lines)


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
