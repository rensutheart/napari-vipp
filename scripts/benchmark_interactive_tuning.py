#!/usr/bin/env python
"""Record cold and warm latency for one real VIPP parameter-tuning workflow.

The harness runs the bundled Portable GPU Segmentation Bridge through the same
detached pipeline service used by the application.  Its first run uses the
authored Gaussian sigma; later runs change only that sigma, reuse the previous
host-side pipeline cache, and dirty the Gaussian node and its descendants.

This is machine-local diagnostic evidence for issue #27.  It reports what
backend and device actually ran, including visible CPU fallback.  Prefer GPU
runs also request synchronized, provider-neutral device phase telemetry.  It
does not make a portable speed claim and does not include UI debounce,
thumbnail work, rendering, or publication.  Importing this module or asking
for ``--help`` does not import an optional GPU provider or initialize CUDA.
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

SCHEMA = "napari-vipp-interactive-tuning-latency-evidence"
SCHEMA_VERSION = 1
EVIDENCE_KIND = "machine-local-detached-workflow-latency-diagnostic"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKFLOW = (
    PROJECT_ROOT
    / "src"
    / "napari_vipp"
    / "examples"
    / "synthetic-gpu-segmentation-bridge.json"
)
TARGET_NODE_ID = "gaussian_blur_1"
TARGET_OPERATION_ID = "gaussian_blur"
TARGET_PARAMETER = "sigma"
DEFAULT_WARM_SIGMAS = (1.4, 1.6, 1.8)
SOURCE_PROVENANCE_PATHS = (
    "scripts/benchmark_interactive_tuning.py",
    "src/napari_vipp/core/execution.py",
    "src/napari_vipp/core/execution_telemetry.py",
    "src/napari_vipp/core/device_execution.py",
    "src/napari_vipp/core/compute_planning.py",
    "src/napari_vipp/core/operations.py",
    "src/napari_vipp/core/gpu/cupy_gaussian.py",
)


class EvidenceError(RuntimeError):
    """A complete, truthful latency evidence document could not be produced."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="JSON evidence path; a complete run atomically replaces this file.",
    )
    parser.add_argument(
        "--workflow",
        type=Path,
        default=DEFAULT_WORKFLOW,
        help=f"Workflow to exercise (default: {DEFAULT_WORKFLOW}).",
    )
    parser.add_argument(
        "--mode",
        choices=("prefer_gpu", "cpu"),
        default="prefer_gpu",
        help="Requested compute policy for this fresh process (default: prefer_gpu).",
    )
    parser.add_argument(
        "--warm-sigmas",
        default=",".join(str(value) for value in DEFAULT_WARM_SIGMAS),
        help="Comma-separated Gaussian sigma edits after the cold run.",
    )
    parser.add_argument(
        "--device-id",
        default="",
        help="Optional exact machine-local device ID, for example cuda:0.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        warm_sigmas = _parse_sigma_values(args.warm_sigmas)
        document = collect_latency_evidence(
            workflow_path=args.workflow,
            mode=args.mode,
            warm_sigmas=warm_sigmas,
            device_id=args.device_id,
        )
        output = _atomic_write_json(args.output, document)
    except (EvidenceError, OSError, TypeError, ValueError) as exc:
        print(f"Interactive tuning benchmark failed: {exc}", file=sys.stderr)
        return 2

    summary = document["summary"]
    target = document["runs"][-1]["target_backend"]
    print(f"Wrote interactive tuning evidence to {output}")
    print(
        "Cold detached request: "
        f"{summary['cold_wall_seconds']:.6f} s; warm median: "
        f"{summary['warm_median_wall_seconds']:.6f} s."
    )
    print(
        "Last Gaussian backend: "
        f"{target['runtime_id']} / {target['implementation_library_id']} / "
        f"{target['implementation_id']} on "
        f"{document['runs'][-1]['environment']['device_name']}."
    )
    return 0


def collect_latency_evidence(
    *,
    workflow_path: Path | str = DEFAULT_WORKFLOW,
    mode: str = "prefer_gpu",
    warm_sigmas: Sequence[float] = DEFAULT_WARM_SIGMAS,
    device_id: str = "",
    clock: Callable[[], float] = perf_counter,
    execute: Callable[[object], object] | None = None,
) -> dict[str, object]:
    """Run one cold request followed by cached Gaussian parameter edits."""

    normalized_sigmas = _normalized_sigma_values(warm_sigmas)
    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in {"cpu", "prefer_gpu"}:
        raise ValueError("mode must be 'cpu' or 'prefer_gpu'.")
    selected_device_id = str(device_id).strip()
    if normalized_mode == "cpu" and selected_device_id:
        raise ValueError("--device-id is valid only with --mode prefer_gpu.")
    if not callable(clock):
        raise TypeError("clock must be callable.")

    path = Path(workflow_path).expanduser().resolve(strict=False)
    if not path.is_file():
        raise EvidenceError(f"Workflow is unavailable: {path}")
    try:
        workflow_bytes = path.read_bytes()
        workflow_document = json.loads(workflow_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"Could not load workflow {path}: {exc}") from exc

    # Project imports remain behind the explicit execution entry point.  In
    # particular, --help does not construct a registry or discover CUDA.
    from napari_vipp.core.compute import ComputeMode, ComputeRequest
    from napari_vipp.core.execution import (
        PipelineRunRequest,
        execute_pipeline_request,
    )
    from napari_vipp.core.execution_telemetry import DeviceExecutionTelemetryConfig
    from napari_vipp.core.pipeline import PrototypePipeline
    from napari_vipp.core.workflow import deserialize_workflow, serialize_workflow

    restored = deserialize_workflow(deepcopy(workflow_document))
    definition = PrototypePipeline()
    definition.restore_graph(
        restored["nodes"],
        restored["connections"],
        restored.get("output_tunnels", ()),
    )
    target = definition.nodes.get(TARGET_NODE_ID)
    if target is None or target.operation_id != TARGET_OPERATION_ID:
        raise EvidenceError(
            f"Workflow must contain {TARGET_NODE_ID!r} as a Gaussian Blur node."
        )
    initial_sigma = _normalized_sigma(target.params.get(TARGET_PARAMETER))
    compute_mode = ComputeMode.parse(normalized_mode)
    authored_request = restored["compute_request"]
    compute_request = ComputeRequest(
        mode=compute_mode,
        node_preferences=authored_request.node_preferences,
        fallback_policy=authored_request.fallback_policy,
        runtime_id=authored_request.runtime_id,
        device_id=selected_device_id,
        precision_policy_id=authored_request.precision_policy_id,
        workload_policy_id=authored_request.workload_policy_id,
        accelerator_memory_cap_bytes=(authored_request.accelerator_memory_cap_bytes),
        accelerator_safety_reserve_bytes=(
            authored_request.accelerator_safety_reserve_bytes
        ),
        allow_experimental=authored_request.allow_experimental,
    )
    if compute_request.mode not in {ComputeMode.CPU, ComputeMode.PREFER_GPU}:
        raise EvidenceError("The harness supports only CPU and Prefer GPU.")

    source_payloads, source_records = _source_payloads(definition)
    manual_node_ids = frozenset(definition.manual_node_ids()) or None
    executor = execute or execute_pipeline_request
    if not callable(executor):
        raise TypeError("execute must be callable or None.")
    device_execution_telemetry = (
        DeviceExecutionTelemetryConfig(synchronize_device_phases=True)
        if compute_request.mode is ComputeMode.PREFER_GPU
        else None
    )

    source_snapshot = _source_provenance(path, workflow_bytes)
    runs: list[dict[str, object]] = []
    cached_pipeline = None
    sigma_sequence = (initial_sigma, *normalized_sigmas)
    for index, sigma in enumerate(sigma_sequence):
        started = _read_clock(clock)
        definition.set_param(TARGET_NODE_ID, TARGET_PARAMETER, sigma)
        detached_workflow = serialize_workflow(
            definition,
            compute_request=compute_request,
        )
        request = PipelineRunRequest(
            run_id=index + 1,
            workflow=detached_workflow,
            input_data=None,
            input_metadata={},
            input_name="",
            source_payloads=dict(source_payloads),
            compute_request=compute_request,
            dirty_node_ids=(None if index == 0 else frozenset({TARGET_NODE_ID})),
            manual_node_ids=manual_node_ids,
            prune_unretained=False,
            performance_history_path=None,
            device_execution_telemetry=device_execution_telemetry,
            **_cached_request_fields(cached_pipeline),
        )
        result = executor(request)
        elapsed = _elapsed(clock, started)
        _validate_successful_result(result, index=index)
        record = _run_record(
            result,
            request=request,
            index=index,
            sigma=sigma,
            wall_seconds=elapsed,
        )
        runs.append(record)
        cached_pipeline = result.pipeline

    if source_snapshot != _source_provenance(path, path.read_bytes()):
        raise EvidenceError(
            "The workflow or latency-harness source changed during measurement."
        )

    warm_seconds = tuple(float(item["wall_seconds"]) for item in runs[1:])
    warm_median = statistics.median(warm_seconds)
    warm_minimum = min(warm_seconds)
    warm_maximum = max(warm_seconds)
    document: dict[str, object] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "kind": EVIDENCE_KIND,
        "portable_performance_claim": False,
        "durable_optimizer_record": False,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "python_executable": Path(sys.executable).name,
        },
        "workflow": {
            "path": str(path),
            "sha256": hashlib.sha256(workflow_bytes).hexdigest(),
            "target_node_id": TARGET_NODE_ID,
            "target_operation_id": TARGET_OPERATION_ID,
            "target_parameter": TARGET_PARAMETER,
            "initial_sigma": initial_sigma,
            "warm_sigmas": list(normalized_sigmas),
        },
        "requested_compute": compute_request.as_dict(),
        "sources": source_records,
        "method": {
            "timing_scope": ("request-build-through-detached-service-return-v1"),
            "cold_definition": (
                "First detached execution in this fresh Python process."
            ),
            "warm_definition": (
                "Later sigma edits in the same process, with prior host-side "
                "pipeline caches supplied and only the Gaussian node marked dirty."
            ),
            "registry_lifecycle": (
                "Production-owned registry and runtime are constructed and closed "
                "by each detached request."
            ),
            "device_phase_telemetry": (
                "Prefer GPU requests insert synchronization barriers around "
                "device transfers and operations and serialize the resulting "
                "volatile observation; CPU requests do not enable it."
            ),
            "cache_contract": "keep-all-host-results; no cross-run device residency",
            "excluded": [
                "process startup",
                "sample/workflow loading",
                "UI debounce and worker queueing",
                "thumbnail statistics and generation",
                "rendering and UI publication",
            ],
        },
        "source_provenance": source_snapshot,
        "runs": runs,
        "summary": {
            "cold_wall_seconds": float(runs[0]["wall_seconds"]),
            "warm_run_count": len(warm_seconds),
            "warm_wall_seconds": list(warm_seconds),
            "warm_median_wall_seconds": warm_median,
            "warm_min_wall_seconds": warm_minimum,
            "warm_max_wall_seconds": warm_maximum,
            "cold_over_warm_median_ratio": (
                float(runs[0]["wall_seconds"]) / warm_median
                if warm_median > 0
                else None
            ),
            "warm_spread_ratio": (
                warm_maximum / warm_minimum if warm_minimum > 0 else None
            ),
            "actual_target_backends": _unique_target_backends(runs),
            "actual_device_ids": sorted(
                {str(item["environment"]["device_id"]) for item in runs}
            ),
            "any_visible_fallback": any(
                bool(item["fallback_records"])
                or bool(item["target_backend"]["fallback_used"])
                for item in runs
            ),
            "all_cleanup_succeeded": all(
                bool(item["cleanup_succeeded"]) for item in runs
            ),
        },
    }
    _validate_evidence(document)
    return document


def _source_payloads(definition) -> tuple[dict[str, object], list[dict[str, object]]]:
    import numpy as np

    from napari_vipp._sample_data import make_sample_data
    from napari_vipp.core.pipeline import SourcePayload

    catalog = {
        layer_kwargs["name"]: (data, layer_kwargs)
        for data, layer_kwargs, _layer_type in make_sample_data()
    }
    payloads: dict[str, object] = {}
    records: list[dict[str, object]] = []
    for node_id in definition.topological_order():
        node = definition.nodes[node_id]
        if node.operation_id != "input":
            continue
        sample_name = str(node.params.get("sample_name", "")).strip()
        if not sample_name or sample_name not in catalog:
            raise EvidenceError(
                f"Source node {node_id!r} does not name an available VIPP sample."
            )
        data, layer_kwargs = catalog[sample_name]
        stable = np.array(data, copy=True)
        payloads[node_id] = SourcePayload(
            stable,
            deepcopy(layer_kwargs.get("metadata", {})),
            str(layer_kwargs.get("name", sample_name)),
        )
        records.append(
            {
                "node_id": node_id,
                "sample_name": sample_name,
                "shape": list(stable.shape),
                "dtype": str(stable.dtype),
                "nbytes": int(stable.nbytes),
                "sha256": hashlib.sha256(stable.tobytes(order="C")).hexdigest(),
            }
        )
    if not payloads:
        raise EvidenceError("Workflow has no executable source node.")
    return payloads, records


def _cached_request_fields(pipeline) -> dict[str, object]:
    if pipeline is None:
        return {}
    return {
        "cached_outputs": dict(pipeline.outputs),
        "cached_output_states": dict(pipeline.output_states),
        "cached_node_outputs": {
            node_id: list(outputs) for node_id, outputs in pipeline.node_outputs.items()
        },
        "cached_node_output_states": {
            node_id: list(states)
            for node_id, states in pipeline.node_output_states.items()
        },
        "cached_execution_states": dict(pipeline.node_execution_states),
        "cached_execution_messages": dict(pipeline.node_execution_messages),
        "cached_compute_provenance": dict(pipeline.node_compute_provenance),
        "completed_node_ids": frozenset(pipeline.completed_node_ids),
    }


def _validate_successful_result(result: object, *, index: int) -> None:
    error = str(getattr(result, "error", "") or "").strip()
    if error:
        raise EvidenceError(f"Run {index} failed: {error}")
    if bool(getattr(result, "cancelled", False)):
        raise EvidenceError(f"Run {index} was cancelled.")
    if getattr(result, "pipeline", None) is None:
        raise EvidenceError(f"Run {index} returned no completed pipeline.")
    report = getattr(result, "execution_report", None)
    if report is None:
        raise EvidenceError(f"Run {index} returned no compute report.")
    if not bool(getattr(report, "cleanup_succeeded", False)):
        raise EvidenceError(f"Run {index} did not complete accelerator cleanup.")
    decisions = tuple(getattr(report, "actual_decisions", ()))
    if not any(item.node_id == TARGET_NODE_ID for item in decisions):
        raise EvidenceError(f"Run {index} did not execute the target Gaussian node.")


def _run_record(
    result: object,
    *,
    request: object,
    index: int,
    sigma: float,
    wall_seconds: float,
) -> dict[str, object]:
    report = result.execution_report
    decisions = tuple(report.actual_decisions)
    target = next(item for item in decisions if item.node_id == TARGET_NODE_ID)
    record: dict[str, object] = {
        "run_index": index,
        "temperature": "cold" if index == 0 else "warm",
        "sigma": sigma,
        "dirty_node_ids": (
            None if request.dirty_node_ids is None else sorted(request.dirty_node_ids)
        ),
        "wall_seconds": wall_seconds,
        "environment": _environment_record(report.environment),
        "target_backend": _decision_record(target),
        "actual_decisions": [_decision_record(item) for item in decisions],
        "segments": _segment_records(getattr(report, "plan", None)),
        "fallback_records": [_record_as_dict(item) for item in report.fallback_records],
        "warnings": [str(item) for item in report.warnings],
        "cleanup_succeeded": bool(report.cleanup_succeeded),
    }
    timing = _optional_core_timing(result)
    expected_device_telemetry = "pipeline_run_result.device_execution_telemetry"
    if (
        getattr(request, "device_execution_telemetry", None) is not None
        and expected_device_telemetry not in timing
    ):
        raise EvidenceError(
            "The detached Prefer GPU run returned no device execution telemetry."
        )
    if timing:
        record["core_timing"] = timing
    return record


def _environment_record(environment: object) -> dict[str, object]:
    topology = getattr(environment, "memory_topology", "")
    return {
        "device_id": str(getattr(environment, "device_id", "")),
        "device_name": str(getattr(environment, "device_name", "")),
        "device_class": str(getattr(environment, "device_class", "")),
        "memory_topology": str(getattr(topology, "value", topology)),
        "runtime_ids": list(getattr(environment, "runtime_ids", ())),
        "implementation_libraries": list(
            getattr(environment, "implementation_libraries", ())
        ),
        "driver_version": str(getattr(environment, "driver_version", "")),
        "total_accelerator_memory_bytes": int(
            getattr(environment, "total_accelerator_memory_bytes", 0)
        ),
        "probe_status": str(getattr(environment, "probe_status", "")),
        "probe_reason": str(getattr(environment, "probe_reason", "")),
    }


def _decision_record(decision: object) -> dict[str, object]:
    return {
        "node_id": str(decision.node_id),
        "operation_id": str(decision.operation_id),
        "runtime_id": str(decision.runtime_id),
        "implementation_library_id": str(decision.implementation_library_id),
        "implementation_id": str(decision.implementation_id),
        "implementation_version": str(getattr(decision, "implementation_version", "")),
        "decision_kind": _enum_text(decision.decision_kind),
        "reason": _enum_text(decision.reason),
        "reason_text": str(decision.reason_text),
        "fallback_used": bool(decision.fallback_used),
        "fallback_reason": _enum_text(decision.fallback_reason),
    }


def _segment_records(plan: object | None) -> list[dict[str, object]]:
    if plan is None:
        return []
    return [
        {
            "segment_id": str(segment.segment_id),
            "runtime_id": str(segment.runtime_id),
            "node_ids": list(segment.node_ids),
            "entry_ports": [
                [port.node_id, port.port_index] for port in segment.entry_ports
            ],
            "exit_ports": [
                [port.node_id, port.port_index] for port in segment.exit_ports
            ],
            "retained_ports": [
                [port.node_id, port.port_index] for port in segment.retained_ports
            ],
        }
        for segment in getattr(plan, "segments", ())
    ]


def _optional_core_timing(result: object) -> dict[str, object]:
    """Consume an optional core timing sidecar without depending on its name."""

    found: dict[str, object] = {}
    owners = (
        ("pipeline_run_result", result),
        ("execution_report", getattr(result, "execution_report", None)),
    )
    for scope, owner in owners:
        if owner is None or not dataclasses.is_dataclass(owner):
            continue
        for item in dataclasses.fields(owner):
            name = item.name.lower()
            if "timing" not in name and "telemetry" not in name:
                continue
            value = getattr(owner, item.name)
            if value is None:
                continue
            found[f"{scope}.{item.name}"] = _json_value(value)
    return found


def _record_as_dict(value: object) -> dict[str, object]:
    method = getattr(value, "as_dict", None)
    if callable(method):
        result = method()
    elif dataclasses.is_dataclass(value):
        result = {
            item.name: _json_value(getattr(value, item.name))
            for item in dataclasses.fields(value)
        }
    else:
        raise EvidenceError("A fallback record is not JSON serializable.")
    if not isinstance(result, Mapping):
        raise EvidenceError("A fallback record did not serialize as an object.")
    return {str(key): _json_value(item) for key, item in result.items()}


def _json_value(value: object) -> object:
    method = getattr(value, "as_dict", None)
    if callable(method):
        return _json_value(method())
    if dataclasses.is_dataclass(value):
        return {
            item.name: _json_value(getattr(value, item.name))
            for item in dataclasses.fields(value)
        }
    if isinstance(value, enum.Enum):
        return _json_value(value.value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise EvidenceError("Core timing contains a non-finite value.")
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item) for item in value]
    raise EvidenceError(
        f"Core timing contains unsupported value {type(value).__name__}."
    )


def _unique_target_backends(runs: Sequence[Mapping[str, object]]) -> list[str]:
    return sorted(
        {
            "/".join(
                (
                    str(item["target_backend"]["runtime_id"]),
                    str(item["target_backend"]["implementation_library_id"]),
                    str(item["target_backend"]["implementation_id"]),
                )
            )
            for item in runs
        }
    )


def _source_provenance(
    workflow_path: Path,
    workflow_bytes: bytes,
) -> dict[str, object]:
    files: dict[str, str] = {}
    for relative in SOURCE_PROVENANCE_PATHS:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise EvidenceError(f"Required benchmark source is unavailable: {relative}")
        files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "hash_algorithm": "sha256",
        "files": files,
        "workflow_path": str(workflow_path),
        "workflow_sha256": hashlib.sha256(workflow_bytes).hexdigest(),
    }


def _parse_sigma_values(value: str) -> tuple[float, ...]:
    parts = tuple(item.strip() for item in str(value).split(","))
    if not parts or any(not item for item in parts):
        raise ValueError("--warm-sigmas must contain comma-separated numbers.")
    try:
        return _normalized_sigma_values(tuple(float(item) for item in parts))
    except ValueError as exc:
        raise ValueError(f"Invalid --warm-sigmas value: {exc}") from exc


def _normalized_sigma_values(values: Sequence[float]) -> tuple[float, ...]:
    normalized = tuple(_normalized_sigma(value) for value in values)
    if not normalized:
        raise ValueError("At least one warm sigma edit is required.")
    return normalized


def _normalized_sigma(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Gaussian sigma values must be numeric.")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 12.0:
        raise ValueError("Gaussian sigma values must be finite and between 0 and 12.")
    return normalized


def _read_clock(clock: Callable[[], float]) -> float:
    value = float(clock())
    if not math.isfinite(value):
        raise EvidenceError("Benchmark clock returned a non-finite value.")
    return value


def _elapsed(clock: Callable[[], float], started: float) -> float:
    elapsed = _read_clock(clock) - started
    if not math.isfinite(elapsed) or elapsed < 0:
        raise EvidenceError("Benchmark clock moved backwards.")
    return elapsed


def _enum_text(value: object) -> str:
    return str(getattr(value, "value", value))


def _validate_evidence(document: Mapping[str, object]) -> None:
    if document.get("schema") != SCHEMA or document.get("schema_version") != 1:
        raise EvidenceError("Evidence identity is invalid.")
    if document.get("kind") != EVIDENCE_KIND:
        raise EvidenceError("Evidence kind is invalid.")
    if document.get("portable_performance_claim") is not False:
        raise EvidenceError("Latency evidence must remain machine-local.")
    runs = document.get("runs")
    if not isinstance(runs, list) or len(runs) < 2:
        raise EvidenceError("Evidence requires one cold and at least one warm run.")
    if runs[0].get("temperature") != "cold" or any(
        item.get("temperature") != "warm" for item in runs[1:]
    ):
        raise EvidenceError("Cold/warm run ordering is invalid.")
    for index, item in enumerate(runs):
        elapsed = item.get("wall_seconds")
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(float(elapsed))
            or float(elapsed) < 0
        ):
            raise EvidenceError(f"Run {index} has an invalid wall time.")
        if item.get("cleanup_succeeded") is not True:
            raise EvidenceError(f"Run {index} did not prove cleanup.")
        target = item.get("target_backend")
        if not isinstance(target, Mapping) or not target.get("runtime_id"):
            raise EvidenceError(f"Run {index} has no actual target backend.")
    summary = document.get("summary")
    if not isinstance(summary, Mapping):
        raise EvidenceError("Evidence summary is unavailable.")
    warm = [float(item["wall_seconds"]) for item in runs[1:]]
    if summary.get("warm_run_count") != len(warm):
        raise EvidenceError("Warm-run summary count is invalid.")
    if not math.isclose(
        float(summary.get("warm_median_wall_seconds", -1)),
        statistics.median(warm),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise EvidenceError("Warm-run median is invalid.")


def _atomic_write_json(
    output: Path | str,
    document: Mapping[str, object],
) -> Path:
    _validate_evidence(document)
    try:
        encoded = (
            json.dumps(
                document,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise EvidenceError(f"Evidence is not strict JSON: {exc}") from exc

    requested = Path(output).expanduser()
    if requested.is_symlink():
        raise EvidenceError("--output must not be a symbolic link.")
    path = requested.resolve(strict=False)
    if not path.name or path.name in {".", ".."}:
        raise EvidenceError("--output must name a file.")
    if path.exists() and path.is_dir():
        raise EvidenceError("--output refers to a directory.")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return path


if __name__ == "__main__":
    raise SystemExit(main())
