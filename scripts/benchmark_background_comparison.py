#!/usr/bin/env python
"""Compare complete background subtraction methods in isolated, bounded workers.

This is exploratory evidence, not production qualification. Different background
models are intentionally compared; faster does not imply scientifically equal.
Worker startup and fixture generation are outside call timings. The first call
includes lazy method imports and any GPU context/JIT startup; it is reported
separately from repeated warmed full host-to-host calls. Each method gets a
fresh process; workers run sequentially. --help needs no numeric packages.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
import traceback
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "vipp-background-comparison-v2"
SEED = 20260922
METHODS = (
    "vipp_cpu",
    "vipp_gpu",
    "itk_ball",
    "itk_box",
    "itk_reconstruction",
    "scipy_box",
)
SOURCE_FILES = (
    "scripts/benchmark_background_comparison.py",
    "scripts/background_benchmark_methods.py",
    "scripts/background_benchmark_fixtures.py",
    "src/napari_vipp/core/operations.py",
    "src/napari_vipp/core/gpu/cupy_background.py",
)


@dataclass(frozen=True)
class Case:
    shape: tuple[int, ...]
    radius: int
    dtype: str = "uint16"
    intensity_scale: str = "native"
    smoothing: bool = True
    spatial_ndim: int = 2
    pattern: str = "mixed"
    light_background: bool = False
    categories: tuple[str, ...] = ("performance",)
    preview: bool = False

    @property
    def name(self) -> str:
        shape = "x".join(map(str, self.shape))
        return (
            f"{shape}-{self.dtype}-{self.intensity_scale}-{self.pattern}"
            f"-r{self.radius}-s{int(self.smoothing)}-d{self.spatial_ndim}"
            f"-l{int(self.light_background)}"
        )

    @classmethod
    def from_dict(cls, value: dict) -> Case:
        value = dict(value)
        value["shape"] = tuple(value["shape"])
        value["categories"] = tuple(value["categories"])
        return cls(**value)


def cases(profile: str) -> list[Case]:
    if profile == "smoke":
        return [
            Case((48, 48), 3, preview=True),
            Case(
                (48, 48),
                3,
                dtype="float32",
                intensity_scale="normalized",
                light_background=True,
                smoothing=False,
                preview=True,
            ),
            Case((4, 32, 32), 2, spatial_ndim=3, preview=True),
        ]
    result = [
        Case(shape, radius)
        for shape in (
            (256, 256),
            (1024, 1024),
            (2048, 2048),
            (16, 256, 256),
            (32, 512, 512),
        )
        for radius in (5, 15, 50)
    ]
    result.extend(
        Case(shape, radius, smoothing=False)
        for shape in ((256, 256), (1024, 1024), (16, 256, 256))
        for radius in (5, 15, 50)
    )
    result.extend(
        Case(
            (512, 512),
            radius,
            dtype="float32",
            intensity_scale=scale,
            smoothing=smoothing,
            categories=("numeric_scale",),
        )
        for scale in ("native", "normalized")
        for radius in (5, 15, 50)
        for smoothing in (False, True)
    )
    result.extend(
        Case(
            (16, 128, 128),
            radius,
            spatial_ndim=3,
            smoothing=smoothing,
            categories=("volumetric",),
        )
        for radius in (3, 7, 15)
        for smoothing in (False, True)
    )
    result.append(Case((32, 128, 128), 7, spatial_ndim=3, categories=("volumetric",)))
    result.extend(
        Case(
            (256, 256),
            15,
            dtype=dtype,
            intensity_scale=scale,
            pattern=pattern,
            light_background=light,
            categories=("quality",),
            preview=(not light),
        )
        for pattern in ("mixed", "sparse", "dense", "broad", "ramp", "vignette")
        for dtype, scale in (("uint16", "native"), ("float32", "normalized"))
        for light in (False, True)
    )
    unique = {}
    for case in result:
        old = unique.get(case.name)
        if old is not None:
            case = replace(
                case,
                categories=tuple(dict.fromkeys(old.categories + case.categories)),
                preview=old.preview or case.preview,
            )
        unique[case.name] = case
    return list(unique.values())


def method_order(case: Case, methods: tuple[str, ...]) -> list[str]:
    remaining = [method for method in methods if method != "vipp_cpu"]
    random.Random(f"{SEED}:{case.name}").shuffle(remaining)
    return (["vipp_cpu"] if "vipp_cpu" in methods else []) + remaining


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dependencies", type=Path)
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--threads", type=int, default=min(os.cpu_count() or 1, 12))
    parser.add_argument("--timeout-seconds", type=float, default=180)
    parser.add_argument("--memory-limit-gib", type=float, default=6)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker-case", help=argparse.SUPPRESS)
    parser.add_argument("--worker-method", choices=METHODS, help=argparse.SUPPRESS)
    return parser


def _utc() -> str:
    return datetime.now(UTC).isoformat()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _sources() -> dict[str, str]:
    return {name: _sha((ROOT / name).read_bytes()) for name in SOURCE_FILES}


def _settings(args) -> dict:
    return {
        key: getattr(args, key)
        for key in (
            "profile",
            "rounds",
            "threads",
            "timeout_seconds",
            "memory_limit_gib",
        )
    } | {"methods": list(args.methods), "seed": SEED}


def _fingerprint(settings: dict, source_hashes: dict, matrix: list[Case]) -> str:
    payload = {
        "schema": SCHEMA,
        "settings": settings,
        "sources": source_hashes,
        "cases": [asdict(case) for case in matrix],
    }
    return _sha(json.dumps(payload, sort_keys=True).encode())


def _packages() -> dict:
    packages = {}
    for name in (
        "numpy",
        "scipy",
        "scikit-image",
        "SimpleITK",
        "psutil",
        "cupy-cuda13x",
        "cupy-cuda12x",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return packages


def _environment(psutil) -> dict:
    def command(argv):
        try:
            result = subprocess.run(
                argv, capture_output=True, text=True, cwd=ROOT, timeout=15, check=True
            )
            return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    return {
        "python": sys.version,
        "python_executable": Path(sys.executable).name,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "physical_cpus": psutil.cpu_count(logical=False),
        "logical_cpus": psutil.cpu_count(),
        "ram_total_bytes": psutil.virtual_memory().total,
        "ram_available_bytes_at_start": psutil.virtual_memory().available,
        "packages": _packages(),
        "git_commit": command(["git", "rev-parse", "HEAD"]),
        "gpu": command(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
    }


def timed_call(call, *, clock=time.perf_counter):
    start = clock()
    value = call()
    return value, clock() - start


def _array_metadata(array) -> dict:
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": _sha(array.tobytes(order="C")),
        "nbytes": int(array.nbytes),
        "writeable": bool(array.flags.writeable),
    }


def _method_kwargs(case: Case, threads: int) -> dict:
    return {
        "radius": case.radius,
        "smoothing": case.smoothing,
        "light_background": case.light_background,
        "clip_negative": True,
        "spatial_ndim": case.spatial_ndim,
        "threads": threads,
    }


def _preview_plane(array):
    while array.ndim > 2:
        array = array[array.shape[0] // 2]
    step = max(1, (max(array.shape) + 511) // 512)
    return array[::step, ::step]


def _worker(args) -> int:
    import numpy as np
    from background_benchmark_fixtures import (
        create_fixture,
        numerical_comparison,
        quality_against_truth,
        quality_comparison,
    )
    from background_benchmark_methods import (
        METHOD_DESCRIPTIONS,
        estimate_background,
        raw_corrected,
        run_method,
    )

    case = next(case for case in cases(args.profile) if case.name == args.worker_case)
    method = args.worker_method
    case_dir = args.output / "cases" / case.name
    result_path = case_dir / f"{method}.json"
    result = {
        "case": case.name,
        "method": method,
        "status": "running",
        "stage": "fixture",
        "started_utc": _utc(),
        "pid": os.getpid(),
        "warm_seconds": [],
        "method_description": METHOD_DESCRIPTIONS[method],
    }

    def save(stage):
        result["stage"] = stage
        result["stage_started_utc"] = _utc()
        _atomic_json(result_path, result)

    save("fixture")
    try:
        fixture = create_fixture(
            case.shape,
            case.dtype,
            pattern=case.pattern,
            intensity_scale=case.intensity_scale,
            light_background=case.light_background,
            seed=SEED,
        )
        source = fixture.input
        result["input"] = _array_metadata(source)
        result["fixture"] = fixture.to_json_metadata()
        kwargs = _method_kwargs(case, args.threads)
        save("first_call")
        output, first = timed_call(lambda: run_method(source, method=method, **kwargs))
        if output.shape != source.shape or output.dtype != source.dtype:
            raise AssertionError("Method changed the expected output shape or dtype.")
        if np.shares_memory(output, source):
            raise AssertionError("Method output aliases its input.")
        result["first_call_seconds"] = first
        result["output"] = _array_metadata(output)
        if method == "vipp_cpu":
            reference_path = case_dir / "reference.npy"
            with reference_path.with_suffix(".npy.tmp").open("wb") as stream:
                np.save(stream, output, allow_pickle=False)
            reference_path.with_suffix(".npy.tmp").replace(reference_path)
            result["reference_artifact"] = reference_path.relative_to(
                args.output
            ).as_posix()
        for iteration in range(args.rounds):
            save(f"warm_call_{iteration + 1}")
            output, elapsed = timed_call(
                lambda: run_method(source, method=method, **kwargs)
            )
            result["warm_seconds"].append(elapsed)
            save(f"warm_call_{iteration + 1}_complete")
        result["median_seconds"] = statistics.median(result["warm_seconds"])
        result["min_seconds"] = min(result["warm_seconds"])
        result["max_seconds"] = max(result["warm_seconds"])
        if _array_metadata(source) != result["input"]:
            raise AssertionError("Method modified its read-only input.")
        if _array_metadata(output)["sha256"] != result["output"]["sha256"]:
            raise AssertionError("First and final calls produced different outputs.")
        save("quality")
        reference_path = case_dir / "reference.npy"
        reference = (
            np.load(reference_path, mmap_mode="r", allow_pickle=False)
            if reference_path.exists()
            else None
        )
        if reference is None:
            result["quality"] = {
                "truth": quality_against_truth(fixture, output),
                "reference_available": False,
            }
        else:
            result["quality"] = quality_comparison(fixture, reference, output)
            result["quality"]["reference_available"] = True
        if case.preview:
            save("diagnostics")
            preview = {
                "input": _preview_plane(source),
                "background_truth": _preview_plane(fixture.background_truth),
                "signal_truth": _preview_plane(fixture.signal_truth),
                "candidate": _preview_plane(output),
            }
            if reference is not None:
                preview["reference"] = _preview_plane(reference)
                preview["signed_difference"] = preview["candidate"].astype(
                    np.float64
                ) - preview["reference"].astype(np.float64)
            if method != "vipp_gpu":
                raw = raw_corrected(source, method=method, **kwargs)
                result["raw_quality"] = quality_against_truth(fixture, raw)
                result["raw_range"] = {
                    "min": float(raw.min()),
                    "max": float(raw.max()),
                    "negative_fraction": float(np.mean(raw < 0)),
                }
                preview["raw_corrected"] = _preview_plane(raw)
                raw_reference = case_dir / "reference.raw.npy"
                if method == "vipp_cpu":
                    np.save(raw_reference, raw, allow_pickle=False)
                if raw_reference.exists():
                    result["raw_reference_comparison"] = numerical_comparison(
                        np.load(raw_reference, mmap_mode="r", allow_pickle=False), raw
                    )
                background_kwargs = {
                    k: v for k, v in kwargs.items() if k != "clip_negative"
                }
                background = estimate_background(
                    source, method=method, **background_kwargs
                )
                result["background_vs_truth"] = numerical_comparison(
                    fixture.background_truth, background
                )
                preview["estimated_background"] = _preview_plane(background)
                del raw, background
            preview_path = case_dir / f"{method}.preview.npz"
            np.savez_compressed(preview_path, **preview)
            result["preview_artifact"] = preview_path.relative_to(
                args.output
            ).as_posix()
        result["status"] = "ok"
        result["completed_utc"] = _utc()
        save("complete")
        return 0
    except Exception as exc:
        result["status"] = "error"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc).replace(str(Path.home()), "<home>")
        result["completed_utc"] = _utc()
        _atomic_json(result_path, result)
        traceback.print_exc()
        return 1


def _process_tree_rss(observed, psutil) -> int:
    """Sample host RSS for the launcher and its current descendants.

    Windows venv executables can be small launchers whose interpreter is a
    child. Shared pages may be counted in more than one process; this is not
    unique physical memory, GPU memory, or a guarantee against brief peaks.
    """
    if observed is None:
        return 0
    try:
        descendants = observed.children(recursive=True)
    except psutil.NoSuchProcess:
        descendants = []
    processes = {item.pid: item for item in [observed, *descendants]}
    total = 0
    for item in processes.values():
        try:
            total += item.memory_info().rss
        except psutil.NoSuchProcess:
            # A process can finish between discovery and its memory query.
            pass
    return total


def _terminate_worker(process, psutil, *, observed=None) -> None:
    """Only terminate the launched worker and its own descendants."""
    try:
        # Reuse the observed process identity where available, rather than
        # resolving a potentially recycled PID after the worker exits.
        parent = observed if observed is not None else psutil.Process(process.pid)
        descendants = parent.children(recursive=True)
        for child in descendants:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        parent.kill()
        psutil.wait_procs(descendants + [parent], timeout=5)
    except psutil.NoSuchProcess:
        pass
    process.wait(timeout=10)


def estimated_worker_bytes(case: Case) -> int:
    elements = 1
    for dimension in case.shape:
        elements *= dimension
    # Conservative preflight only, not a measured peak or a memory guarantee.
    return 512 * 1024**2 + elements * 8 * 16


def _run_worker(args, case: Case, method: str, psutil) -> dict:
    case_dir = args.output / "cases" / case.name
    case_dir.mkdir(parents=True, exist_ok=True)
    path = case_dir / f"{method}.json"
    cap = int(args.memory_limit_gib * 1024**3)
    estimate = estimated_worker_bytes(case)
    if estimate > cap or estimate > psutil.virtual_memory().available - 1024**3:
        result = {
            "case": case.name,
            "method": method,
            "status": "memory_preflight",
            "estimated_worker_bytes": estimate,
            "memory_limit_bytes": cap,
        }
        _atomic_json(path, result)
        return result
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--output",
        str(args.output),
        "--profile",
        args.profile,
        "--worker-case",
        case.name,
        "--worker-method",
        method,
        "--rounds",
        str(args.rounds),
        "--threads",
        str(args.threads),
    ]
    if args.dependencies:
        command.extend(("--dependencies", str(args.dependencies)))
    environment = dict(os.environ)
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
    ):
        environment[name] = str(args.threads)
    start = time.monotonic()
    peak = 0
    stop_reason = None
    with (case_dir / f"{method}.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, env=environment, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
        )
        try:
            observed = psutil.Process(process.pid)
        except psutil.NoSuchProcess:
            observed = None
        while process.poll() is None:
            peak = max(peak, _process_tree_rss(observed, psutil))
            if peak > cap:
                stop_reason = "memory_limit"
            elif time.monotonic() - start > args.timeout_seconds:
                stop_reason = "timeout"
            if stop_reason:
                _terminate_worker(process, psutil, observed=observed)
                break
            time.sleep(0.2)
    result = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.exists()
        else {
            "case": case.name,
            "method": method,
            "status": "error",
            "stage": "startup",
        }
    )
    result.update(
        {
            "process_wall_seconds": time.monotonic() - start,
            "peak_process_tree_rss_bytes": peak,
            "rss_scope": (
                "Sampled sum of host RSS for the launcher and its descendants; "
                "shared pages may be counted more than once. Excludes GPU memory."
            ),
            "returncode": process.returncode,
            "memory_limit_bytes": cap,
            "timeout_seconds": args.timeout_seconds,
            "rss_sampling_interval_seconds": 0.2,
        }
    )
    if stop_reason:
        result["status"] = stop_reason
        result["censored"] = True
        result["note"] = (
            "Worker wall limit includes setup, all calls and quality diagnostics."
        )
    elif process.returncode and result.get("status") != "error":
        result["status"] = "error"
        result["error"] = "Worker exited unsuccessfully; see its local log."
    _atomic_json(path, result)
    return result


def _checkpoint(output: Path, report: dict) -> None:
    report["updated_utc"] = _utc()
    report["counts"] = dict(Counter(row["status"] for row in report["results"]))
    _atomic_json(output / "results.json", report)
    done = len(report["results"])
    content = (
        "# Background-subtraction benchmark\n\n"
        f"Status: {report['status']}\n\n"
        f"Completed backend attempts: {done} / {report['total_attempts']}\n\n"
        f"Counts: {json.dumps(report['counts'], sort_keys=True)}\n\n"
        f"Current: {report.get('current', 'none')}\n\n"
        f"Parent PID: {report['pid']}\n\n"
        f"Updated UTC: {report['updated_utc']}\n\n"
        "Timeouts are censored attempts, not completed speed measurements. "
        "Only status=ok rows with all requested repetitions support speed ratios. "
        "Peak RSS is the sampled sum of launcher and descendant host RSS; "
        "shared pages may be counted more than once. GPU memory is not measured.\n"
    )
    temporary = output / "CHECKPOINT.md.tmp"
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(output / "CHECKPOINT.md")


def _event(output: Path, report: dict, message: str, **fields) -> None:
    record = {"utc": _utc(), "message": message, **fields}
    with (output / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")
        stream.flush()
    print(f"{record['utc']} {message} {json.dumps(fields)}", flush=True)
    _checkpoint(output, report)


def _assert_no_live_workers(output: Path, psutil) -> None:
    """An interrupted parent may leave a worker alive: refuse overlapping runs."""
    expected_script = Path(__file__).resolve()
    for process in psutil.process_iter(("pid", "cmdline")):
        command = process.info.get("cmdline") or []
        if "--worker-case" not in command or "--output" not in command:
            continue
        try:
            matching_script = any(
                Path(argument).resolve() == expected_script for argument in command[:3]
            )
            requested_output = Path(command[command.index("--output") + 1]).resolve()
        except (OSError, ValueError, IndexError):
            continue
        if matching_script and requested_output == output:
            raise RuntimeError(
                "A benchmark worker for this output is still running; "
                "refusing duplicate execution."
            )


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if (
        args.rounds < 1
        or args.threads < 1
        or args.timeout_seconds <= 0
        or args.memory_limit_gib <= 0
    ):
        raise ValueError("Rounds, threads, timeout and memory limit must be positive.")
    args.output = args.output.resolve()
    if args.dependencies:
        args.dependencies = args.dependencies.resolve()
        sys.path.insert(0, str(args.dependencies))
    sys.path.insert(0, str(ROOT / "src"))
    if args.worker_case:
        return _worker(args)
    import psutil

    matrix = cases(args.profile)
    source_hashes = _sources()
    fingerprint = _fingerprint(_settings(args), source_hashes, matrix)
    report_path = args.output / "results.json"
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise FileExistsError("Output is not empty; use a new directory or --resume.")
    if args.resume:
        if not report_path.exists():
            raise FileNotFoundError("No prior results.json exists to resume safely.")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report["fingerprint"] != fingerprint
            or report["environment"]["packages"] != _packages()
        ):
            raise ValueError(
                "Cannot resume: sources, settings, cases or package versions changed."
            )
        old_pid = report.get("pid")
        if report["status"] == "running" and old_pid and psutil.pid_exists(old_pid):
            raise RuntimeError(
                "Prior benchmark process still exists; refusing duplicate execution."
            )
        _assert_no_live_workers(args.output, psutil)
        report["pid"] = os.getpid()
        report["status"] = "running"
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        report = {
            "schema": SCHEMA,
            "fingerprint": fingerprint,
            "settings": _settings(args),
            "source_sha256": source_hashes,
            "cases": [asdict(case) | {"id": case.name} for case in matrix],
            "environment": _environment(psutil),
            "started_utc": _utc(),
            "status": "running",
            "pid": os.getpid(),
            "results": [],
            "total_attempts": len(matrix) * len(args.methods),
            "timing_contract": (
                "First call includes lazy method imports and GPU context/JIT startup. "
                "Warm calls include the full host-to-host method after initialization. "
                "Worker startup, fixture generation and quality analysis are excluded."
            ),
        }
    completed = {(row["case"], row["method"]) for row in report["results"]}
    _event(args.output, report, "benchmark_started")
    for case in matrix:
        for method in method_order(case, tuple(args.methods)):
            if (case.name, method) in completed:
                continue
            if _sources() != source_hashes:
                report["status"] = "source_changed"
                _event(args.output, report, "source_changed_aborting")
                return 2
            report["current"] = f"{case.name} / {method}"
            _event(
                args.output, report, "backend_started", case=case.name, method=method
            )
            row = _run_worker(args, case, method, psutil)
            report["results"].append(row)
            _event(
                args.output,
                report,
                "backend_finished",
                case=case.name,
                method=method,
                status=row["status"],
            )
    report["status"] = "completed"
    report["current"] = None
    report["completed_utc"] = _utc()
    _event(args.output, report, "benchmark_completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
