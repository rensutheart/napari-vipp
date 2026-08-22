"""Create reproducible machine-local core GPU benchmark evidence.

This is deliberately a maintainer-only, headless command.  It benchmarks the
fixed float32 background, Gaussian, and median cases through VIPP's production
node adapter, after every candidate passes the current public environment and
scientific admission gates.

Optional GPU providers remain lazy on ``--help`` and module import.  They are
loaded only when an explicit benchmark run probes and executes them.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

EVIDENCE_SCHEMA = "napari-vipp-core-public-gpu-benchmark-evidence"
EVIDENCE_SCHEMA_VERSION = 2
CASE_GENERATOR = "numpy-pcg64-v1"
ROUND_ORDER_SEED = 20_260_728
REQUIRED_RUNTIME_ID = "cuda-cupy"
ALLOW_EXPERIMENTAL = False
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PROVENANCE_PATHS = (
    "scripts/benchmark_gpu_phase1.py",
    "src/napari_vipp/core/compute.py",
    "src/napari_vipp/core/compute_benchmark.py",
    "src/napari_vipp/core/compute_benchmark_adapter.py",
    "src/napari_vipp/core/compute_planning.py",
    "src/napari_vipp/core/compute_policy.py",
    "src/napari_vipp/core/compute_policy_artifact.py",
    "src/napari_vipp/core/compute_registry.py",
    "src/napari_vipp/core/compute_specs.py",
    "src/napari_vipp/core/node_execution.py",
    "src/napari_vipp/core/operations.py",
    "src/napari_vipp/core/gpu/cupy_runtime.py",
    "src/napari_vipp/core/gpu/cupy_gaussian.py",
    "src/napari_vipp/core/gpu/cupy_median.py",
    "src/napari_vipp/core/gpu/cupy_background.py",
    "src/napari_vipp/compute_policies/phase1-gpu-public-v10.json",
)


class BenchmarkEvidenceError(RuntimeError):
    """A complete, admitted Phase-1 evidence document could not be produced."""


@dataclass(frozen=True, slots=True)
class _CaseDefinition:
    operation_id: str
    seed: int
    shape: tuple[int, ...]
    parameters: tuple[tuple[str, object], ...]


CASE_DEFINITIONS = (
    _CaseDefinition(
        "rolling_ball_background",
        20_260_700,
        (31, 37),
        (
            ("radius", 5),
            ("light_background", False),
            ("disable_smoothing", False),
            ("spatial_mode", "2D YX"),
            ("resolved_spatial_ndim", 2),
            ("progress", None),
            ("channel_axis", None),
        ),
    ),
    _CaseDefinition(
        "subtract_background",
        20_260_701,
        (31, 37),
        (
            ("radius", 5),
            ("light_background", False),
            ("disable_smoothing", False),
            ("clip_negative", True),
            ("spatial_mode", "2D YX"),
            ("resolved_spatial_ndim", 2),
            ("progress", None),
            ("channel_axis", None),
        ),
    ),
    _CaseDefinition(
        "gaussian_blur",
        20_260_702,
        (128, 160),
        (("sigma", 1.3), ("channel_axis", None)),
    ),
    _CaseDefinition(
        "gaussian_blur_3d",
        20_260_704,
        (9, 31, 37),
        (
            ("sigma_z", 0.8),
            ("sigma_y", 1.3),
            ("sigma_x", 1.7),
            ("channel_axis", None),
        ),
    ),
    _CaseDefinition(
        "median_filter",
        20_260_703,
        (96, 112),
        (("size", 5), ("channel_axis", None)),
    ),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="JSON evidence path. A complete run atomically replaces this file.",
    )
    parser.add_argument(
        "--device-id",
        default="",
        help="Exact CUDA device ID to benchmark (default: probed selected device).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        document = run_benchmarks(device_id=args.device_id)
        output = _atomic_write_json(args.output, document)
    except (BenchmarkEvidenceError, OSError, TypeError, ValueError) as exc:
        print(f"Phase-1 GPU benchmark failed: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # provider failures need a concise CLI boundary
        print(
            f"Phase-1 GPU benchmark failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    print(f"Wrote Phase-1 GPU benchmark evidence to {output}")
    return 0


def run_benchmarks(*, device_id: str = "") -> dict[str, object]:
    """Run the fixed production profile after exact public admission."""

    source_provenance = _source_provenance()

    # Core imports live behind the explicit execution entry point. In
    # particular, importing this script or asking for --help cannot discover a
    # CuPy provider or initialize a CUDA context.
    import numpy as np

    from napari_vipp.core.compute import ComputeMode, ComputeRequest
    from napari_vipp.core.compute_benchmark import NodeBenchmarkService
    from napari_vipp.core.compute_benchmark_adapter import (
        PRODUCTION_BENCHMARK_POLICY_ID,
        build_registered_node_benchmark,
    )
    from napari_vipp.core.compute_planning import probe_compute_environment
    from napari_vipp.core.compute_policy import (
        ArrayFacts,
        FactCompleteness,
        evaluate_candidate_support,
    )
    from napari_vipp.core.compute_policy_artifact import (
        load_phase1_compute_policy,
    )
    from napari_vipp.core.compute_registry import ComputeRegistry
    from napari_vipp.core.node_execution import PreparedNodeCall
    from napari_vipp.core.operations import (
        gaussian_blur,
        gaussian_blur_3d,
        median_filter,
        rolling_ball_background,
        subtract_background,
    )

    cpu_functions = {
        "rolling_ball_background": rolling_ball_background,
        "subtract_background": subtract_background,
        "gaussian_blur": gaussian_blur,
        "gaussian_blur_3d": gaussian_blur_3d,
        "median_filter": median_filter,
    }
    policy = load_phase1_compute_policy()
    if (
        policy.exposure.developer_enablement_required
        or not policy.exposure.public_controls_enabled
        or policy.status != "public-validated"
    ):
        raise BenchmarkEvidenceError(
            "The loaded compute policy is not the current public validated policy."
        )

    request = ComputeRequest(
        mode=ComputeMode.SELECTIVE,
        runtime_id=REQUIRED_RUNTIME_ID,
        device_id=str(device_id).strip(),
        allow_experimental=ALLOW_EXPERIMENTAL,
    )
    registry = ComputeRegistry()
    try:
        selected_specs = []
        spec_by_operation = {}
        for definition in CASE_DEFINITIONS:
            matches = registry.implementations_for_operation(
                definition.operation_id,
                allow_experimental=ALLOW_EXPERIMENTAL,
            )
            if len(matches) != 1:
                raise BenchmarkEvidenceError(
                    f"Required operation {definition.operation_id!r} has "
                    f"{len(matches)} public GPU candidates; expected exactly one."
                )
            spec = matches[0]
            packaged = policy.operation(definition.operation_id)
            if (
                packaged.implementation_id != spec.implementation_id
                or packaged.implementation_version != spec.implementation_version
                or packaged.runtime_id != spec.runtime_id
                or packaged.implementation_library_id != spec.implementation_library_id
                or packaged.environment_policy_id
                != spec.validated_environment_policy_id
            ):
                raise BenchmarkEvidenceError(
                    f"Loaded policy and executable declaration disagree for "
                    f"{definition.operation_id!r}."
                )
            selected_specs.append(spec)
            spec_by_operation[definition.operation_id] = spec

        environment, warnings = probe_compute_environment(
            registry,
            request,
            tuple(selected_specs),
        )
        _require_probed_environment(environment, selected_specs)

        runtime_probe = registry.probe_runtime(REQUIRED_RUNTIME_ID)
        library_probes = {
            library_id: registry.probe_library(library_id)
            for library_id in sorted(
                {spec.implementation_library_id for spec in selected_specs}
            )
        }
        if not runtime_probe.available:
            raise BenchmarkEvidenceError(
                _probe_failure_text("runtime", REQUIRED_RUNTIME_ID, runtime_probe)
            )
        for library_id, probe in library_probes.items():
            if not probe.available:
                raise BenchmarkEvidenceError(
                    _probe_failure_text("library", library_id, probe)
                )

        results = []
        for definition in CASE_DEFINITIONS:
            array = np.random.default_rng(definition.seed).random(
                definition.shape,
                dtype=np.float32,
            )
            call = PreparedNodeCall(
                node_id=f"phase1-benchmark-{definition.operation_id}",
                operation_id=definition.operation_id,
                cpu_function=cpu_functions[definition.operation_id],
                inputs=(array,),
                kwargs=dict(definition.parameters),
            )
            spec = spec_by_operation[definition.operation_id]
            built = build_registered_node_benchmark(
                call,
                admitted_specs=(spec,),
                registry=registry,
                environment_fingerprint=environment.fingerprint,
                device_id=environment.device_id,
                allow_experimental=ALLOW_EXPERIMENTAL,
            )
            _require_production_profile(
                built.request,
                policy.local_benchmark,
                policy_id=PRODUCTION_BENCHMARK_POLICY_ID,
            )
            facts = _complete_array_facts(
                array,
                revision_fingerprint=built.request.workload.facts_fingerprint,
                array_facts_type=ArrayFacts,
                completeness=FactCompleteness.COMPLETE,
            )
            support = evaluate_candidate_support(
                spec,
                built.request.workload,
                environment,
                allow_experimental=ALLOW_EXPERIMENTAL,
                array_facts=(facts,),
            )
            if not support.supported:
                raise BenchmarkEvidenceError(
                    f"Exact admission rejected {spec.implementation_id!r}: "
                    f"{support.reason.value}: {support.reason_text}"
                )

            service = NodeBenchmarkService(
                rng=random.Random(ROUND_ORDER_SEED + definition.seed)
            )
            record = service.benchmark(built.request)
            candidate = next(
                (
                    item
                    for item in record.candidates
                    if item.implementation_id == spec.implementation_id
                ),
                None,
            )
            if candidate is None:
                raise BenchmarkEvidenceError(
                    f"Benchmark omitted required candidate {spec.implementation_id!r}."
                )
            if not candidate.parity_passed or candidate.error:
                detail = candidate.error or "scientific parity failed"
                raise BenchmarkEvidenceError(
                    f"Candidate {spec.implementation_id!r} failed: {detail}"
                )
            results.append(
                _serialize_case(
                    definition,
                    array,
                    spec,
                    built,
                    record,
                    production_warm_rounds=(
                        policy.local_benchmark.initial_warm_rounds,
                        *policy.local_benchmark.adaptive_warm_rounds,
                    ),
                )
            )

        document = {
            "schema": EVIDENCE_SCHEMA,
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "created_utc": datetime.now(UTC).isoformat(),
            "kind": "machine-local-public-production-node-benchmark",
            "experimental_candidates_enabled": ALLOW_EXPERIMENTAL,
            "fixed_case_generator": CASE_GENERATOR,
            "round_order_seed": ROUND_ORDER_SEED,
            "policy": {
                "schema_id": policy.schema_id,
                "policy_id": policy.policy_id,
                "policy_version": policy.policy_version,
                "content_sha256": policy.content_sha256,
                "phase": policy.phase,
                "status": policy.status,
                "benchmark_policy_id": PRODUCTION_BENCHMARK_POLICY_ID,
                "initial_warm_rounds": policy.local_benchmark.initial_warm_rounds,
                "adaptive_warm_rounds": list(
                    policy.local_benchmark.adaptive_warm_rounds
                ),
                "bootstrap_resamples": policy.local_benchmark.bootstrap_resamples,
                "bootstrap_seed": policy.local_benchmark.bootstrap_seed,
                "confidence_level": policy.local_benchmark.confidence_level,
            },
            "platform": _platform_provenance(environment),
            "packages": _package_provenance(),
            "source_provenance": source_provenance,
            "environment": environment.as_dict(),
            "probe_warnings": list(warnings),
            "runtime_probe": runtime_probe.as_dict(),
            "library_probes": {
                name: probe.as_dict() for name, probe in library_probes.items()
            },
            "results": results,
        }
        _require_source_snapshot_unchanged(source_provenance)
        return document
    finally:
        registry.close()


def _complete_array_facts(
    value: Any,
    *,
    revision_fingerprint: str,
    array_facts_type: Any,
    completeness: Any,
) -> Any:
    """Create complete finite facts for one already materialized fixed case."""

    import numpy as np

    array = np.asarray(value)
    finite = np.isfinite(array)
    finite_values = array[finite]
    guarantees = []
    if not bool(np.any((array == 0) & np.signbit(array))):
        guarantees.append("no-negative-zero")
    if finite_values.size and bool(np.min(finite_values) >= 0):
        guarantees.append("nonnegative")
    return array_facts_type(
        shape=tuple(int(size) for size in array.shape),
        dtype=array.dtype.name,
        element_count=int(array.size),
        revision_fingerprint=revision_fingerprint,
        completeness=completeness,
        finite_count=int(np.count_nonzero(finite)),
        minimum=(float(np.min(finite_values)) if finite_values.size else None),
        maximum=(float(np.max(finite_values)) if finite_values.size else None),
        strides=tuple(int(stride) for stride in array.strides),
        contiguous=bool(array.flags.c_contiguous),
        guarantees=tuple(guarantees),
    )


def _require_probed_environment(environment: Any, specs: Sequence[Any]) -> None:
    if environment.probe_status != "available":
        detail = environment.probe_reason or "accelerator probe was unavailable"
        raise BenchmarkEvidenceError(f"Exact GPU environment unavailable: {detail}")
    if REQUIRED_RUNTIME_ID not in environment.runtime_ids:
        raise BenchmarkEvidenceError(
            f"Required runtime {REQUIRED_RUNTIME_ID!r} was not admitted."
        )
    if not environment.device_id.startswith("cuda:"):
        raise BenchmarkEvidenceError(
            "The exact environment did not select a CUDA device."
        )
    for spec in specs:
        if spec.runtime_id not in environment.runtime_ids:
            raise BenchmarkEvidenceError(
                f"Required runtime {spec.runtime_id!r} is unavailable for "
                f"{spec.implementation_id!r}."
            )
        if spec.implementation_library_id not in (environment.implementation_libraries):
            raise BenchmarkEvidenceError(
                f"Required library {spec.implementation_library_id!r} is "
                f"unavailable for {spec.implementation_id!r}."
            )


def _probe_failure_text(kind: str, identifier: str, probe: Any) -> str:
    reason = probe.reason_code or "unavailable"
    detail = probe.message or "no diagnostic was reported"
    return f"Required {kind} {identifier!r} is unavailable ({reason}): {detail}"


def _require_production_profile(
    request: Any,
    policy: Any,
    *,
    policy_id: str,
) -> None:
    adaptive_rounds = tuple(policy.adaptive_warm_rounds)
    round_targets = (policy.initial_warm_rounds, *adaptive_rounds)
    matches = (
        bool(adaptive_rounds)
        and all(previous < current for previous, current in pairwise(round_targets))
        and request.benchmark_policy_id == policy_id
        and request.warm_rounds == policy.initial_warm_rounds
        and request.max_warm_rounds == adaptive_rounds[-1]
        and request.adaptive_rounds is True
        and request.time_parity_as_cold is True
        and request.warmup_rounds == 1
        and request.paired_bootstrap_samples == policy.bootstrap_resamples
        and request.paired_bootstrap_seed == policy.bootstrap_seed
        and request.paired_confidence_level == policy.confidence_level
        and request.time_budget_seconds is None
    )
    if not matches:
        raise BenchmarkEvidenceError(
            f"{request.workload.operation_id!r} did not use the loaded production "
            "benchmark sampling profile."
        )


def _serialize_case(
    definition: _CaseDefinition,
    array: Any,
    spec: Any,
    built: Any,
    record: Any,
    *,
    production_warm_rounds: Sequence[int],
) -> dict[str, object]:
    import numpy as np

    completed_warm_rounds = _require_complete_production_samples(
        record,
        built,
        spec,
        production_warm_rounds=production_warm_rounds,
    )
    versions = {
        built.request.reference.implementation_id: (
            built.request.reference.implementation_version
        ),
        **{
            item.implementation_id: item.implementation_version
            for item in built.request.candidates
        },
    }
    candidate_records = []
    for result in record.candidates:
        is_gpu = result.implementation_id == spec.implementation_id
        runs = built.observations.runs(result.implementation_id) if is_gpu else ()
        terminal = [run.terminal_snapshot.as_dict() for run in runs]
        cleanup_succeeded = all(run.cleanup_succeeded for run in runs)
        terminal_zero = all(
            snapshot["runtime_live_bytes"] == 0
            and snapshot["runtime_reserved_bytes"] == 0
            for snapshot in terminal
        )
        expected_run_count = completed_warm_rounds + built.request.warmup_rounds + 1
        if is_gpu and (
            len(runs) != expected_run_count
            or not cleanup_succeeded
            or not terminal_zero
            or not result.synchronized
            or not result.transfers_included
            or result.timing_scope != "synchronized-end-to-end-v1"
        ):
            raise BenchmarkEvidenceError(
                f"Candidate {result.implementation_id!r} did not produce complete "
                "synchronized timing and zero-pool cleanup evidence."
            )
        candidate_records.append(
            {
                "implementation_id": result.implementation_id,
                "implementation_version": versions[result.implementation_id],
                "runtime_id": spec.runtime_id if is_gpu else "cpu-numpy",
                "library_id": (spec.implementation_library_id if is_gpu else "cpu"),
                "parity_policy_id": (
                    spec.parity_policy_id if is_gpu else "authoritative-cpu-v1"
                ),
                "measurements": asdict(result),
                "cleanup": {
                    "applicable": is_gpu,
                    "invocation_count": len(runs),
                    "all_cleanup_succeeded": cleanup_succeeded if is_gpu else None,
                    "all_runtime_pool_terminal_zero": (
                        terminal_zero if is_gpu else None
                    ),
                    "terminal_snapshots": terminal,
                },
            }
        )
    contiguous = np.ascontiguousarray(array)
    return {
        "operation_id": definition.operation_id,
        "seed": definition.seed,
        "generator": CASE_GENERATOR,
        "shape": list(definition.shape),
        "dtype": contiguous.dtype.name,
        "input_sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
        "parameters": {
            name: value for name, value in definition.parameters if name != "progress"
        },
        "workload_fingerprint": built.request.workload.fingerprint,
        "environment_fingerprint": built.request.environment_fingerprint,
        "benchmark_record_digest": record.key.digest,
        "benchmark_policy_id": record.benchmark_policy_id,
        "accepted_implementation_id": record.accepted_implementation_id,
        "paired_confidence_level": record.paired_confidence_level,
        "paired_bootstrap_samples": record.paired_bootstrap_samples,
        "paired_bootstrap_seed": record.paired_bootstrap_seed,
        "created_utc": record.created_utc,
        "device_id": record.key.device_id,
        "memory_limit_bytes": record.key.memory_limit_bytes,
        "safety_reserve_bytes": record.key.safety_reserve_bytes,
        "candidates": candidate_records,
    }


def _require_complete_production_samples(
    record: Any,
    built: Any,
    spec: Any,
    *,
    production_warm_rounds: Sequence[int],
) -> int:
    """Reject incomplete or non-policy CPU/GPU timing populations."""

    allowed = tuple(production_warm_rounds)
    if (
        not allowed
        or any(
            isinstance(value, bool) or not isinstance(value, int) for value in allowed
        )
        or tuple(sorted(set(allowed))) != allowed
        or any(value < 1 for value in allowed)
    ):
        raise BenchmarkEvidenceError(
            "The loaded production warm-round targets are invalid."
        )
    expected_ids = (
        built.request.reference.implementation_id,
        spec.implementation_id,
    )
    results = tuple(record.candidates)
    actual_ids = tuple(result.implementation_id for result in results)
    if len(results) != 2 or set(actual_ids) != set(expected_ids):
        raise BenchmarkEvidenceError(
            "Benchmark results did not contain exactly the expected CPU and GPU "
            "implementations."
        )
    if len(set(actual_ids)) != len(actual_ids):
        raise BenchmarkEvidenceError("Benchmark results contain duplicate candidates.")
    counts = {
        result.implementation_id: len(tuple(result.warm_seconds)) for result in results
    }
    completed = counts[expected_ids[0]]
    if counts[expected_ids[1]] != completed:
        raise BenchmarkEvidenceError(
            "CPU and GPU benchmark results contain different warm-round counts."
        )
    if completed not in allowed:
        raise BenchmarkEvidenceError(
            f"Benchmark completed {completed} warm rounds; expected one of "
            f"{allowed!r} from the loaded production policy."
        )
    return completed


def _platform_provenance(environment: Any) -> dict[str, object]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_abi": environment.python_abi,
        "execution_mode": environment.execution_mode,
        # The interpreter version and ABI above carry the reproducibility
        # signal.  Persisting its absolute path would only disclose a local
        # account/worktree layout and make otherwise equivalent evidence
        # machine-path-dependent.
        "executable": _executable_name(sys.executable),
    }


def _executable_name(executable: str) -> str:
    """Return a platform-neutral interpreter label without its host path."""

    normalized = str(executable).strip().replace("\\", "/").rstrip("/")
    return normalized.rsplit("/", 1)[-1] or "python"


def _package_provenance() -> dict[str, str | None]:
    names = (
        "napari-vipp",
        "numpy",
        "scipy",
        "scikit-image",
        "cupy-cuda12x",
        "cupy-cuda13x",
    )
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _source_provenance(
    *,
    project_root: Path = PROJECT_ROOT,
    source_paths: Sequence[str] = SOURCE_PROVENANCE_PATHS,
) -> dict[str, object]:
    """Fingerprint every local source that determines the benchmark evidence."""

    root = Path(project_root).resolve(strict=False)
    paths = tuple(str(value).replace("\\", "/") for value in source_paths)
    if not paths or len(set(paths)) != len(paths):
        raise BenchmarkEvidenceError(
            "Source-provenance paths must be nonempty and unique."
        )
    hashes: dict[str, str] = {}
    for relative in paths:
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise BenchmarkEvidenceError(
                f"Invalid source-provenance path {relative!r}."
            )
        path = root.joinpath(candidate)
        try:
            if not path.is_file():
                raise OSError("not a regular file")
            hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            raise BenchmarkEvidenceError(
                f"Could not fingerprint required source {relative!r}: {exc}"
            ) from exc
    return {
        "hash_algorithm": "sha256",
        "files": hashes,
        "git": _git_provenance(root),
    }


def _git_provenance(project_root: Path) -> dict[str, object]:
    """Return exact Git identity when available, or a structured explanation."""

    def run(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ("git", "-C", str(project_root), *arguments),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )

    try:
        head = run("rev-parse", "--verify", "HEAD")
        if head.returncode != 0:
            detail = (head.stderr or head.stdout or "not a Git worktree").strip()
            return {"available": False, "reason": detail}
        branch = run("symbolic-ref", "--quiet", "--short", "HEAD")
        status = run("status", "--porcelain=v1", "--untracked-files=all")
        if branch.returncode not in {0, 1} or status.returncode != 0:
            detail = (
                branch.stderr
                or status.stderr
                or "Git could not inspect the branch and worktree state."
            ).strip()
            return {"available": False, "reason": detail}
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "available": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }

    status_lines = tuple(line for line in status.stdout.splitlines() if line)
    return {
        "available": True,
        "head": head.stdout.strip(),
        "branch": branch.stdout.strip() if branch.returncode == 0 else None,
        "detached_head": branch.returncode == 1,
        "worktree_dirty": bool(status_lines),
        "status_porcelain_v1": list(status_lines),
    }


def _require_source_snapshot_unchanged(provenance: Mapping[str, object]) -> None:
    """Fail rather than publish timings whose relevant source changed mid-run."""

    expected = provenance.get("files")
    if not isinstance(expected, Mapping):
        raise BenchmarkEvidenceError("Source provenance has no file-hash mapping.")
    current = _source_provenance(
        source_paths=tuple(str(path) for path in expected),
    )["files"]
    if current != expected:
        raise BenchmarkEvidenceError(
            "Relevant benchmark source changed while evidence was being collected."
        )


def _atomic_write_json(
    output: Path | str,
    document: Mapping[str, object],
) -> Path:
    """Serialize first, then replace exactly one user-selected file."""

    if not isinstance(document, Mapping):
        raise TypeError("evidence document must be a mapping")
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
        raise BenchmarkEvidenceError(f"Evidence is not strict JSON: {exc}") from exc

    requested_path = Path(output).expanduser()
    if requested_path.is_symlink():
        raise BenchmarkEvidenceError("--output must not be a symbolic link")
    path = requested_path.resolve(strict=False)
    if not path.name or path.name in {".", ".."}:
        raise BenchmarkEvidenceError("--output must name a file")
    if path.exists() and path.is_dir():
        raise BenchmarkEvidenceError("--output refers to a directory")
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
