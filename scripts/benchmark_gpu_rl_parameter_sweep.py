#!/usr/bin/env python
"""Measure interactive Richardson--Lucy edits through production execution.

The sweep uses VIPP's detached pipeline service, not a standalone provider
microbenchmark.  Each operation/rank sequence visits a PSF extent for the
first time, revisits that exact extent later in the same process, and changes
an authored scalar parameter without changing the PSF.  Every accelerated
result must use the registered CuPy implementation without fallback, complete
private-runtime cleanup, and pass the operation's production CPU parity gate.

Importing this module or asking for ``--help`` does not import CuPy or probe
CUDA.  The JSON output is machine-local diagnostic evidence, not a portable
performance claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

SCHEMA = "napari-vipp-rl-interactive-parameter-sweep"
SCHEMA_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "docs/benchmarks/rl-parameter-sweep-windows-rtx5090.json"
)

OPERATION_CONFIG = {
    "rl": {
        "operation_id": "richardson_lucy_deconvolution",
        "implementation_id": "rl-cupy-f32-v1",
    },
    "rl-tv": {
        "operation_id": "richardson_lucy_tv_deconvolution",
        "implementation_id": "rl-tv-cupy-f32-v1",
    },
}


class SweepError(RuntimeError):
    """A complete and truthful sweep could not be produced."""


@dataclass(frozen=True, slots=True)
class SweepCase:
    case_id: str
    visit: str
    psf_shape: tuple[int, ...]
    iterations: int
    tv_regularization: float | None = None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"JSON evidence path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--operations",
        default="rl,rl-tv",
        help="Comma-separated subset of rl,rl-tv (default: both).",
    )
    parser.add_argument(
        "--ranks",
        default="2,3",
        help="Comma-separated subset of 2,3 (default: both).",
    )
    parser.add_argument(
        "--device-id",
        default="",
        help="Optional exact CUDA device ID, for example cuda:0.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        document = collect_evidence(
            operations=_parse_operations(args.operations),
            spatial_ranks=_parse_ranks(args.ranks),
            device_id=args.device_id,
        )
        output = _atomic_write_json(args.output, document)
    except (OSError, SweepError, TypeError, ValueError) as exc:
        print(f"RL parameter sweep failed: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote RL parameter-sweep evidence to {output}")
    for group in document["groups"]:
        comparison = group["matched_psf_comparison"]
        print(
            f"{group['operation']} {group['spatial_rank']}D: first "
            f"{comparison['first_seconds']:.6f} s, revisit "
            f"{comparison['revisit_seconds']:.6f} s, ratio "
            f"{comparison['first_over_revisit_ratio']:.3f}."
        )
    return 0


def collect_evidence(
    *,
    operations: Sequence[str] = ("rl", "rl-tv"),
    spatial_ranks: Sequence[int] = (2, 3),
    device_id: str = "",
    clock: Callable[[], float] = perf_counter,
    execute: Callable[..., object] | None = None,
) -> dict[str, object]:
    """Run matched unseen/revisited PSF edits and CPU parity checks."""

    selected_operations = _normalized_operations(operations)
    selected_ranks = _normalized_ranks(spatial_ranks)
    if not callable(clock):
        raise TypeError("clock must be callable.")

    # All project imports remain behind this explicit execution entry point.
    import numpy as np

    from napari_vipp.core.compute import ComputeMode, ComputeRequest
    from napari_vipp.core.compute_policy import ArrayFactsCache
    from napari_vipp.core.execution import PipelineRunRequest, execute_pipeline_request
    from napari_vipp.core.metadata import AxisMetadata, image_state_from_array
    from napari_vipp.core.pipeline import PrototypePipeline, SourcePayload
    from napari_vipp.core.richardson_lucy_parity import (
        richardson_lucy_float32_parity,
        richardson_lucy_tv_float32_parity,
    )
    from napari_vipp.core.workflow import serialize_workflow

    executor = execute or execute_pipeline_request
    if not callable(executor):
        raise TypeError("execute must be callable or None.")
    gpu_request = ComputeRequest(
        mode=ComputeMode.PREFER_GPU,
        device_id=str(device_id).strip(),
    )
    cpu_request = ComputeRequest(mode=ComputeMode.CPU)
    provenance_before = _source_provenance()
    groups: list[dict[str, object]] = []
    run_id = 0

    for operation_name in selected_operations:
        config = OPERATION_CONFIG[operation_name]
        parity_gate = (
            richardson_lucy_float32_parity
            if operation_name == "rl"
            else richardson_lucy_tv_float32_parity
        )
        for rank in selected_ranks:
            image = _synthetic_image(rank)
            axes = _spatial_axes(rank, AxisMetadata)
            image_state = image_state_from_array(
                image,
                axes=axes,
                source_name=f"deterministic {rank}D RL sweep image",
                defer_statistics=True,
            )
            definition = PrototypePipeline()
            image_source = definition.add_node("input")
            psf_source = definition.add_node("input")
            target = definition.add_node(str(config["operation_id"]))
            if not definition.connect(
                image_source.id, target.id, target_port=0
            ).success:
                raise SweepError("Could not connect the deterministic image source.")
            if not definition.connect(psf_source.id, target.id, target_port=1).success:
                raise SweepError("Could not connect the deterministic PSF source.")
            definition.set_param(
                target.id,
                "spatial_mode",
                "2D YX" if rank == 2 else "3D ZYX",
            )

            facts_cache = ArrayFactsCache()
            cached_pipeline = None
            gpu_runs: list[dict[str, object]] = []
            gpu_outputs: list[np.ndarray] = []
            case_workflows: list[dict[str, object]] = []
            case_payloads: list[dict[str, SourcePayload]] = []
            for case_index, case in enumerate(_cases(operation_name, rank)):
                definition.set_param(target.id, "iterations", case.iterations)
                if case.tv_regularization is not None:
                    definition.set_param(
                        target.id,
                        "tv_regularization",
                        case.tv_regularization,
                    )
                psf = _gaussian_psf(case.psf_shape)
                psf_state = image_state_from_array(
                    psf,
                    axes=axes,
                    source_name=f"deterministic {case.psf_shape} PSF",
                    defer_statistics=True,
                )
                payloads = {
                    image_source.id: SourcePayload(
                        image,
                        name="RL sweep image",
                        image_state=image_state,
                        revision_token=("rl-sweep-image", rank, 1),
                    ),
                    psf_source.id: SourcePayload(
                        psf,
                        name="RL sweep PSF",
                        image_state=psf_state,
                        revision_token=("rl-sweep-psf", case.psf_shape),
                    ),
                }
                workflow = serialize_workflow(definition, compute_request=gpu_request)
                run_id += 1
                request = PipelineRunRequest(
                    run_id=run_id,
                    workflow=workflow,
                    input_data=None,
                    input_metadata={},
                    input_name="",
                    source_payloads=payloads,
                    compute_request=gpu_request,
                    dirty_node_ids=(
                        None
                        if case_index == 0
                        else frozenset({psf_source.id, target.id})
                    ),
                    manual_node_ids=frozenset({target.id}),
                    target_node_ids=frozenset({target.id}),
                    retain_node_ids=frozenset({target.id}),
                    prune_unretained=False,
                    array_facts_cache=facts_cache,
                    performance_history_path=None,
                    **_cached_request_fields(cached_pipeline),
                )
                timeline = _Timeline(target.id, clock)
                started = _read_clock(clock)
                result = executor(
                    request,
                    node_started_callback=timeline.node_started,
                    node_finished_callback=timeline.node_finished,
                    progress_callback=timeline.progress,
                )
                elapsed = _elapsed(clock, started)
                _validate_gpu_result(
                    result,
                    target_id=target.id,
                    operation_id=str(config["operation_id"]),
                    implementation_id=str(config["implementation_id"]),
                )
                output = np.asarray(result.pipeline.outputs[target.id])
                gpu_runs.append(
                    _gpu_run_record(
                        result,
                        target_id=target.id,
                        case=case,
                        wall_seconds=elapsed,
                        timeline=timeline.record(),
                    )
                )
                gpu_outputs.append(np.array(output, copy=True))
                case_workflows.append(workflow)
                case_payloads.append(payloads)
                cached_pipeline = result.pipeline

            parity_records: list[dict[str, object]] = []
            for case, workflow, payloads, gpu_output in zip(
                _cases(operation_name, rank),
                case_workflows,
                case_payloads,
                gpu_outputs,
                strict=True,
            ):
                run_id += 1
                cpu_result = executor(
                    PipelineRunRequest(
                        run_id=run_id,
                        workflow=workflow,
                        input_data=None,
                        input_metadata={},
                        input_name="",
                        source_payloads=payloads,
                        compute_request=cpu_request,
                        manual_node_ids=frozenset({target.id}),
                        target_node_ids=frozenset({target.id}),
                        retain_node_ids=frozenset({target.id}),
                        prune_unretained=False,
                        performance_history_path=None,
                    )
                )
                _validate_cpu_result(cpu_result, target_id=target.id)
                parity = parity_gate(cpu_result.pipeline.outputs[target.id], gpu_output)
                if not parity.passed:
                    raise SweepError(
                        f"{operation_name} {rank}D {case.case_id} failed parity: "
                        f"{parity.detail}"
                    )
                parity_records.append(
                    {
                        "case_id": case.case_id,
                        "passed": True,
                        "detail": parity.detail,
                    }
                )

            comparison = _matched_psf_comparison(gpu_runs)
            groups.append(
                {
                    "operation": operation_name,
                    "operation_id": config["operation_id"],
                    "implementation_id": config["implementation_id"],
                    "spatial_rank": rank,
                    "image_shape": list(image.shape),
                    "runs": gpu_runs,
                    "parity": parity_records,
                    "matched_psf_comparison": comparison,
                    "avoidable_shape_stall_detected": bool(
                        comparison["delta_seconds"] >= 0.25
                        and comparison["first_over_revisit_ratio"] >= 3.0
                    ),
                }
            )

    if provenance_before != _source_provenance():
        raise SweepError("Sweep sources changed during measurement.")
    document: dict[str, object] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "portable_performance_claim": False,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "python_executable": Path(sys.executable).name,
        },
        "requested_compute": gpu_request.as_dict(),
        "method": {
            "execution_surface": "production-detached-pipeline-service-v1",
            "sequence": "unseen-psf, matched-revisit, scalar-edit, baseline-revisit",
            "cache_contract": "host-result-cache-only; no cross-run device residency",
            "fft_plan_note": (
                "The production CuPy runtime disables its FFT plan cache inside "
                "each private execution scope and restores external entries on exit."
            ),
            "phase_timing_note": (
                "Progress intervals are synchronized RL iterations and include "
                "both FFT setup/plan creation and convolution kernels; public "
                "CuPy APIs do not separate those components further."
            ),
        },
        "source_provenance": provenance_before,
        "groups": groups,
        "summary": {
            "group_count": len(groups),
            "all_parity_passed": all(
                item["passed"] for group in groups for item in group["parity"]
            ),
            "all_cleanup_succeeded": all(
                run["cleanup_succeeded"] for group in groups for run in group["runs"]
            ),
            "any_fallback": any(
                run["fallback_used"] for group in groups for run in group["runs"]
            ),
            "any_avoidable_shape_stall": any(
                group["avoidable_shape_stall_detected"] for group in groups
            ),
        },
    }
    _validate_document(document)
    return document


class _Timeline:
    def __init__(self, target_id: str, clock: Callable[[], float]) -> None:
        self.target_id = target_id
        self.clock = clock
        self.origin: float | None = None
        self.started: float | None = None
        self.finished: float | None = None
        self.progress_events: list[tuple[int, int, float]] = []

    def _sample(self) -> float:
        return _read_clock(self.clock)

    def node_started(self, node_id: str) -> None:
        if node_id == self.target_id:
            self.started = self._sample()
            if self.origin is None:
                self.origin = self.started

    def progress(self, node_id: str, current: int, total: int, _message: str) -> None:
        if node_id == self.target_id:
            value = self._sample()
            if self.origin is None:
                self.origin = value
            self.progress_events.append((int(current), int(total), value))

    def node_finished(self, result: object) -> None:
        if getattr(result, "node_id", "") == self.target_id:
            self.finished = self._sample()

    def record(self) -> dict[str, object]:
        if self.origin is None:
            return {"available": False, "iteration_seconds": []}
        iteration_seconds = [
            max(0.0, later[2] - earlier[2])
            for earlier, later in zip(
                self.progress_events,
                self.progress_events[1:],
                strict=False,
            )
            if later[0] == earlier[0] + 1
        ]
        return {
            "available": True,
            "target_active_seconds": (
                None
                if self.started is None or self.finished is None
                else max(0.0, self.finished - self.started)
            ),
            "iteration_seconds": iteration_seconds,
            "first_iteration_seconds": (
                iteration_seconds[0] if iteration_seconds else None
            ),
            "later_iteration_median_seconds": (
                statistics.median(iteration_seconds[1:])
                if len(iteration_seconds) > 1
                else None
            ),
        }


def _cases(operation: str, rank: int) -> tuple[SweepCase, ...]:
    if rank == 2:
        baseline, changed = (5, 5), (9, 11)
    else:
        baseline, changed = (3, 5, 5), (5, 9, 11)
    if operation == "rl":
        return (
            SweepCase("baseline-cold", "first", baseline, 2),
            SweepCase("changed-first", "first", changed, 2),
            SweepCase("baseline-revisit", "revisit", baseline, 2),
            SweepCase("changed-revisit", "revisit", changed, 2),
            SweepCase("iteration-edit", "scalar-edit", changed, 5),
        )
    return (
        SweepCase("baseline-cold", "first", baseline, 10, 0.002),
        SweepCase("changed-first", "first", changed, 10, 0.002),
        SweepCase("baseline-revisit", "revisit", baseline, 10, 0.002),
        SweepCase("changed-revisit", "revisit", changed, 10, 0.002),
        SweepCase("iteration-edit", "scalar-edit", changed, 25, 0.002),
        SweepCase("regularization-edit", "scalar-edit", changed, 10, 0.0),
    )


def _synthetic_image(rank: int):
    import numpy as np

    shape = (96, 112) if rank == 2 else (9, 48, 52)
    coordinates = np.indices(shape, dtype=np.float32)
    values = np.full(shape, np.float32(0.04), dtype=np.float32)
    centers = (
        tuple((size - 1) * fraction for size in shape)
        for fraction in (0.28, 0.53, 0.74)
    )
    for spot_index, center in enumerate(centers):
        squared = np.zeros(shape, dtype=np.float32)
        for axis, axis_center in enumerate(center):
            width = 1.5 + axis * 0.4 + spot_index * 0.25
            squared += ((coordinates[axis] - axis_center) / width) ** 2
        values += np.float32(0.9 - spot_index * 0.18) * np.exp(
            -np.float32(0.5) * squared
        ).astype(np.float32)
    rng = np.random.default_rng(20_260_820 + rank)
    values += rng.uniform(0.0, 0.015, size=shape).astype(np.float32)
    return np.ascontiguousarray(values, dtype=np.float32)


def _gaussian_psf(shape: tuple[int, ...]):
    import numpy as np

    coordinates = np.indices(shape, dtype=np.float64)
    squared = np.zeros(shape, dtype=np.float64)
    for axis, size in enumerate(shape):
        center = (size - 1) / 2.0
        sigma = max(size / 5.5, 0.8)
        squared += ((coordinates[axis] - center) / sigma) ** 2
    psf = np.exp(-0.5 * squared)
    psf /= np.sum(psf, dtype=np.float64)
    return np.ascontiguousarray(psf, dtype=np.float32)


def _spatial_axes(rank: int, axis_type):
    names = ("y", "x") if rank == 2 else ("z", "y", "x")
    scales = (0.2, 0.2) if rank == 2 else (0.5, 0.2, 0.2)
    return tuple(
        axis_type(name, "space", unit="micrometer", scale=scale)
        for name, scale in zip(names, scales, strict=True)
    )


def _gpu_run_record(result, *, target_id, case, wall_seconds, timeline):
    report = result.execution_report
    decision = next(
        item for item in report.actual_decisions if item.node_id == target_id
    )
    fallback = bool(report.fallback_records) or bool(decision.fallback_used)
    return {
        "case_id": case.case_id,
        "visit": case.visit,
        "psf_shape": list(case.psf_shape),
        "iterations": case.iterations,
        "tv_regularization": case.tv_regularization,
        "wall_seconds": wall_seconds,
        "timeline": timeline,
        "runtime_id": decision.runtime_id,
        "implementation_library_id": decision.implementation_library_id,
        "implementation_id": decision.implementation_id,
        "device_id": report.environment.device_id,
        "device_name": report.environment.device_name,
        "fallback_used": fallback,
        "cleanup_succeeded": bool(report.cleanup_succeeded),
    }


def _validate_gpu_result(result, *, target_id, operation_id, implementation_id):
    error = str(getattr(result, "error", "") or "").strip()
    if error or getattr(result, "pipeline", None) is None:
        raise SweepError(f"GPU execution failed: {error or 'no pipeline result'}")
    report = getattr(result, "execution_report", None)
    if report is None or not report.cleanup_succeeded:
        raise SweepError("GPU execution did not prove private-runtime cleanup.")
    decisions = [item for item in report.actual_decisions if item.node_id == target_id]
    if len(decisions) != 1:
        raise SweepError("GPU execution did not report exactly one target decision.")
    decision = decisions[0]
    if (
        decision.operation_id != operation_id
        or decision.runtime_id != "cuda-cupy"
        or decision.implementation_id != implementation_id
        or decision.implementation_library_id != "cupyx"
    ):
        raise SweepError(
            "Target did not use the exact registered CuPyX implementation: "
            f"{decision.runtime_id}/{decision.implementation_library_id}/"
            f"{decision.implementation_id}."
        )
    if report.fallback_records or decision.fallback_used:
        raise SweepError("The RL sweep does not accept CPU fallback.")


def _validate_cpu_result(result, *, target_id):
    error = str(getattr(result, "error", "") or "").strip()
    if error or getattr(result, "pipeline", None) is None:
        raise SweepError(f"CPU parity execution failed: {error or 'no pipeline'}")
    if result.pipeline.outputs.get(target_id) is None:
        raise SweepError("CPU parity execution returned no target output.")


def _matched_psf_comparison(runs: Sequence[Mapping[str, object]]):
    first = next(item for item in runs if item["case_id"] == "changed-first")
    revisit = next(item for item in runs if item["case_id"] == "changed-revisit")
    first_seconds = float(first["wall_seconds"])
    revisit_seconds = float(revisit["wall_seconds"])
    return {
        "psf_shape": list(first["psf_shape"]),
        "first_seconds": first_seconds,
        "revisit_seconds": revisit_seconds,
        "delta_seconds": first_seconds - revisit_seconds,
        "first_over_revisit_ratio": (
            first_seconds / revisit_seconds if revisit_seconds > 0 else math.inf
        ),
    }


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


def _source_provenance() -> dict[str, str]:
    paths = (
        "scripts/benchmark_gpu_rl_parameter_sweep.py",
        "src/napari_vipp/core/execution.py",
        "src/napari_vipp/core/device_execution.py",
        "src/napari_vipp/core/gpu/cupy_runtime.py",
        "src/napari_vipp/core/gpu/cupy_rl.py",
        "src/napari_vipp/core/gpu/cupy_rl_tv.py",
        "src/napari_vipp/core/richardson_lucy.py",
        "src/napari_vipp/core/richardson_lucy_compute.py",
        "src/napari_vipp/core/richardson_lucy_parity.py",
    )
    return {
        path: hashlib.sha256((PROJECT_ROOT / path).read_bytes()).hexdigest()
        for path in paths
    }


def _validate_document(document: Mapping[str, object]) -> None:
    summary = document.get("summary")
    if document.get("schema") != SCHEMA or not isinstance(summary, Mapping):
        raise SweepError("Evidence document has an invalid schema.")
    if not summary.get("all_parity_passed"):
        raise SweepError("Evidence contains a CPU/GPU parity failure.")
    if not summary.get("all_cleanup_succeeded"):
        raise SweepError("Evidence contains a cleanup failure.")
    if summary.get("any_fallback"):
        raise SweepError("Evidence contains CPU fallback.")
    try:
        json.dumps(document, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise SweepError(f"Evidence is not strict JSON: {exc}") from exc


def _normalized_operations(values: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(str(value).strip().lower() for value in values))
    if not normalized or any(value not in OPERATION_CONFIG for value in normalized):
        raise ValueError("operations must contain only rl and rl-tv.")
    return normalized


def _normalized_ranks(values: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(dict.fromkeys(int(value) for value in values))
    if not normalized or any(value not in {2, 3} for value in normalized):
        raise ValueError("spatial_ranks must contain only 2 and 3.")
    return normalized


def _parse_operations(value: str) -> tuple[str, ...]:
    return _normalized_operations(value.split(","))


def _parse_ranks(value: str) -> tuple[int, ...]:
    try:
        return _normalized_ranks(tuple(int(item.strip()) for item in value.split(",")))
    except ValueError as exc:
        raise ValueError("--ranks must be a comma-separated subset of 2,3.") from exc


def _read_clock(clock: Callable[[], float]) -> float:
    value = float(clock())
    if not math.isfinite(value):
        raise SweepError("Clock returned a non-finite value.")
    return value


def _elapsed(clock: Callable[[], float], started: float) -> float:
    elapsed = _read_clock(clock) - started
    if elapsed < 0:
        raise SweepError("Clock moved backwards during measurement.")
    return elapsed


def _atomic_write_json(path: Path | str, document: Mapping[str, object]) -> Path:
    target = Path(path).expanduser().resolve(strict=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(document, allow_nan=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="\n",
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
        handle.flush()
    temporary.replace(target)
    return target


if __name__ == "__main__":
    raise SystemExit(main())
