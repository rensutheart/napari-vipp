#!/usr/bin/env python
"""Bounded, conversion-inclusive CPU qualification for selected SimpleITK filters.

This is exploratory machine-local evidence, not portable admission or an
optimizer record. The full rolling-ball algorithm has no matching SimpleITK
entry point and is deliberately NOT replaced by morphological opening. The
background mean prefilter is checked for equality but is not a timed candidate.

Example (an optional private dependency directory keeps the installed app alone)::

    python scripts/benchmark_simpleitk_cpu.py --dependencies PATH --output run.json

Existing output is never overwritten without --resume. JSON checkpoints and an
append-only .events.jsonl sibling survive interruptions. Importing this script
or asking for --help does not import SimpleITK, NumPy, SciPy, or VIPP.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "vipp-simpleitk-cpu-comparison-v1"
SEED = 20260922
OPERATIONS = ("median", "opening", "closing", "distance")


@dataclass(frozen=True)
class Case:
    operation: str
    shape: tuple[int, ...]
    dtype: str
    size: int = 0

    @property
    def name(self) -> str:
        shape = "x".join(map(str, self.shape))
        size = f"-k{self.size}" if self.size else ""
        return f"{self.operation}-{shape}-{self.dtype}{size}"


def cases(profile: str) -> list[Case]:
    """Explicit bounded workload list, independent of numerical imports."""
    if profile == "smoke":
        return [
            Case("median", (128, 128), "uint16", 5),
            Case("median", (256, 256), "float32", 5),
            Case("opening", (128, 128), "bool", 5),
            Case("closing", (8, 64, 64), "bool", 5),
            Case("distance", (8, 64, 64), "bool"),
        ]
    result = [
        Case("median", (side, side), dtype, size)
        for side in (128, 256, 512)
        for dtype in ("uint8", "uint16", "float32")
        for size in (3, 5, 11)
    ]
    result.extend(
        Case("median", shape, dtype, size)
        for shape, size in (
            ((512, 512), 21),
            ((2048, 2048), 5),
            ((16, 256, 256), 5),
            ((32, 512, 512), 5),
        )
        for dtype in ("uint16", "float32")
    )
    for shape in ((512, 512), (2048, 2048), (16, 256, 256)):
        result.extend(
            Case(operation, shape, "bool", size)
            for operation in ("opening", "closing")
            for size in (5, 11)
        )
        result.append(Case("distance", shape, "bool"))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dependencies", type=Path)
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument(
        "--operations", nargs="+", choices=OPERATIONS, default=OPERATIONS
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--threads", type=int, default=min(os.cpu_count() or 1, 12))
    parser.add_argument("--qualify-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-cold-start", action="store_true")
    parser.add_argument("--max-case-seconds", type=float, default=120.0)
    parser.add_argument("--cold-child", action="store_true", help=argparse.SUPPRESS)
    return parser


def _load(dependencies: Path | None):
    if dependencies is not None:
        sys.path.insert(0, str(dependencies.resolve()))
    sys.path.insert(0, str(ROOT / "src"))
    import numpy as np
    import psutil
    import SimpleITK as sitk
    from scipy import ndimage as ndi

    from napari_vipp.core import operations as ops
    from napari_vipp.core import simpleitk_filters as median

    return np, psutil, sitk, ndi, ops, median


def _utc() -> str:
    return datetime.now(UTC).isoformat()


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_hashes() -> dict[str, str]:
    names = (
        "scripts/benchmark_simpleitk_cpu.py",
        "src/napari_vipp/core/operations.py",
        "src/napari_vipp/core/simpleitk_filters.py",
    )
    return {name: _hash((ROOT / name).read_bytes()) for name in names}


def _environment(psutil, sitk) -> dict:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    packages = {}
    for name in ("numpy", "scipy", "scikit-image", "SimpleITK", "psutil"):
        packages[name] = importlib.metadata.version(name)
    return {
        "utc": _utc(),
        "python": sys.version,
        # Version/build information is useful evidence; a private home path is
        # not. Keep this report safe to include in repository documentation.
        "python_executable": Path(sys.executable).name,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": psutil.cpu_count(),
        "physical_cpus": psutil.cpu_count(logical=False),
        "ram_total_bytes": psutil.virtual_memory().total,
        "ram_available_bytes": psutil.virtual_memory().available,
        "packages": packages,
        "simpleitk_version": str(sitk.Version()),
        "git_commit": commit,
        "source_sha256": _source_hashes(),
    }


class Recorder:
    def __init__(self, path: Path, report: dict):
        self.path = path
        self.report = report
        self.events = path.with_suffix(".events.jsonl")

    def save(self) -> None:
        self.report["updated_utc"] = _utc()
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.report, indent=2, allow_nan=False), encoding="utf-8"
        )
        temporary.replace(self.path)

    def event(self, message: str, **fields) -> None:
        record = {"utc": _utc(), "message": message, **fields}
        with self.events.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
        print(f"{record['utc']} {message}", flush=True)
        self.report["current_activity"] = record
        self.save()


def _sitk_morphology(array, *, size, operation, np, sitk, threads):
    if size < 1 or size % 2 != 1:
        raise ValueError("Only odd box widths have this exact radius/origin mapping.")
    image = sitk.GetImageFromArray(np.asarray(array, dtype=np.uint8), isVector=False)
    radius = [size // 2, size // 2] + [0] * (array.ndim - 2)
    sequence = (
        (sitk.BinaryErodeImageFilter, sitk.BinaryDilateImageFilter)
        if operation == "opening"
        else (sitk.BinaryDilateImageFilter, sitk.BinaryErodeImageFilter)
    )
    # Compose primitives, clipping at the original bounds between steps. ITK's
    # composite opening/closing safe-border defaults are not SciPy's contract.
    for filter_class in sequence:
        method = filter_class()
        method.SetKernelType(sitk.sitkBox)
        method.SetKernelRadius(radius)
        method.SetForegroundValue(1)
        method.SetBackgroundValue(0)
        method.SetBoundaryToForeground(False)
        method.SetNumberOfThreads(threads)
        method.SetNumberOfWorkUnits(threads)
        image = method.Execute(image)
    return sitk.GetArrayFromImage(image).astype(bool, copy=False)


def _sitk_distance(array, *, np, sitk, ndi, threads):
    mask = np.asarray(array, dtype=bool)
    # SciPy's no-background sentinel geometry is unusual; preserve it exactly
    # rather than treating infinity/the image border as a scientific substitute.
    if not mask.size or bool(mask.all()):
        return ndi.distance_transform_edt(mask).astype(np.float32, copy=False)
    image = sitk.GetImageFromArray((~mask).astype(np.uint8), isVector=False)
    method = sitk.SignedMaurerDistanceMapImageFilter()
    method.SetInsideIsPositive(False)
    method.SetSquaredDistance(False)
    method.SetUseImageSpacing(False)
    method.SetBackgroundValue(0)
    method.SetNumberOfThreads(threads)
    method.SetNumberOfWorkUnits(threads)
    result = sitk.GetArrayFromImage(method.Execute(image))
    # Distances OUTSIDE the inverted foreground measure to background voxel
    # centres. SignedMaurer directly on the original mask instead uses an
    # interior contour-zero convention and is not the same operation.
    return np.where(mask, result, 0).astype(np.float32, copy=False)


def _functions(case, *, np, sitk, ndi, ops, median, threads):
    if case.operation == "median":
        axes = (len(case.shape) - 2, len(case.shape) - 1)
        return {
            "scipy_reference": lambda array: median.scipy_median_filter(
                array, size=case.size, xy_axes=axes
            ),
            "simpleitk_forced": lambda array: median._simpleitk_median_filter(
                array, size=case.size, xy_axes=axes
            ),
            "vipp_runtime": lambda array: ops.median_filter(array, size=case.size),
        }
    if case.operation in {"opening", "closing"}:
        return {
            "vipp_reference": lambda array: getattr(ops, case.operation)(
                array, size=case.size
            ),
            "simpleitk_candidate": lambda array: _sitk_morphology(
                array,
                size=case.size,
                operation=case.operation,
                np=np,
                sitk=sitk,
                threads=threads,
            ),
        }
    return {
        "vipp_reference": lambda array: ops.euclidean_distance_transform(
            array,
            spatial_mode="3D ZYX" if array.ndim == 3 else "2D YX",
            resolved_spatial_ndim=array.ndim,
        ),
        "simpleitk_candidate": lambda array: _sitk_distance(
            array, np=np, sitk=sitk, ndi=ndi, threads=threads
        ),
    }


def _compare(expected, actual, np) -> dict:
    shape_ok = expected.shape == actual.shape
    dtype_ok = expected.dtype == actual.dtype
    exact = (
        shape_ok and dtype_ok and bool(np.array_equal(expected, actual, equal_nan=True))
    )
    result = {"shape_equal": shape_ok, "dtype_equal": dtype_ok, "exact_values": exact}
    if shape_ok:
        result["bitwise_equal"] = dtype_ok and expected.tobytes() == actual.tobytes()
        result["different_values"] = int(np.count_nonzero(expected != actual))
        delta = np.asarray(expected, dtype=np.float64) - np.asarray(
            actual, dtype=np.float64
        )
        result["max_absolute_difference"] = (
            float(np.max(np.abs(delta))) if delta.size else 0.0
        )
    return result


def _phantoms(np):
    rng = np.random.default_rng(SEED)
    for shape in ((1, 7), (9, 11), (3, 9, 11)):
        for pattern in ("empty", "full", "checkerboard", "edge-touching", "random"):
            array = np.zeros(shape, dtype=bool)
            if pattern == "full":
                array[...] = True
            elif pattern == "checkerboard":
                array = np.indices(shape).sum(axis=0) % 2 == 0
            elif pattern == "edge-touching":
                array[
                    (slice(None),) * (len(shape) - 2) + (slice(0, 7), slice(0, 8))
                ] = True
                array[(0,) * len(shape)] = False
            elif pattern == "random":
                array = rng.random(shape) > 0.35
            yield pattern, array


def qualify(*, np, sitk, ndi, ops, median, threads):
    checks = []
    for pattern, array in _phantoms(np):
        array.setflags(write=False)
        before = array.tobytes()
        for operation in ("opening", "closing", "distance"):
            for size in (3, 5, 11) if operation != "distance" else (0,):
                case = Case(operation, array.shape, "bool", size)
                methods = _functions(
                    case,
                    np=np,
                    sitk=sitk,
                    ndi=ndi,
                    ops=ops,
                    median=median,
                    threads=threads,
                )
                expected, actual = (function(array) for function in methods.values())
                checks.append(
                    {
                        "case": case.name,
                        "pattern": pattern,
                        **_compare(expected, actual, np),
                        "input_unchanged": before == array.tobytes(),
                        "output_detached": not np.shares_memory(array, actual),
                    }
                )
    # Background presmoothing is nearest-edge 3^N mean, not a median. Floating
    # accumulation and output rounding can differ even with matching support.
    rng = np.random.default_rng(SEED)
    background = []
    for shape in ((13, 17), (5, 13, 17)):
        for dtype in (np.float32, np.float64):
            array = (rng.random(shape) * 65535.0).astype(dtype)
            array[(0,) * array.ndim] = 0
            expected = ndi.uniform_filter(array, size=3, mode="nearest")
            method = sitk.MeanImageFilter()
            method.SetRadius([1] * array.ndim)
            method.SetNumberOfThreads(threads)
            method.SetNumberOfWorkUnits(threads)
            actual = sitk.GetArrayFromImage(
                method.Execute(sitk.GetImageFromArray(array, isVector=False))
            )
            background.append(
                {
                    "shape": list(shape),
                    "dtype": str(array.dtype),
                    **_compare(expected, actual, np),
                }
            )
    return {"checks": checks, "background_mean_checks": background}


def _input(case, np, ndi):
    rng = np.random.default_rng(SEED + sum(case.shape) + case.size)
    if case.dtype == "bool":
        # Large edge-touching objects and holes, not only isolated random voxels.
        seeds = rng.random(case.shape) > 0.985
        widths = (0.0,) * (len(case.shape) - 2) + (3.0, 3.0)
        array = ndi.gaussian_filter(seeds.astype(np.float32), widths) > 0.012
        array[(slice(None),) * (array.ndim - 2) + (slice(0, 13), slice(0, 19))] = True
    elif case.dtype == "float32":
        array = (rng.random(case.shape, dtype=np.float32) * np.float32(4095.0)).astype(
            case.dtype
        )
    else:
        high = 256 if case.dtype == "uint8" else 4096
        array = rng.integers(0, high, size=case.shape, dtype=case.dtype)
    array.setflags(write=False)
    return array


def _cold_child(dependencies):
    if dependencies is not None:
        sys.path.insert(0, str(dependencies.resolve()))
    sys.path.insert(0, str(ROOT / "src"))
    import numpy as np

    from napari_vipp.core import operations as ops

    array = np.random.default_rng(SEED).integers(0, 4096, (512, 512), dtype=np.uint16)
    loaded_before = "SimpleITK" in sys.modules
    started = time.perf_counter()
    ops.median_filter(array, size=5)
    cold = time.perf_counter() - started
    started = time.perf_counter()
    ops.median_filter(array, size=5)
    warm = time.perf_counter() - started
    print(
        json.dumps(
            {
                "case": "median-512x512-uint16-k5",
                "simpleitk_imported_before": loaded_before,
                "simpleitk_imported_after": "SimpleITK" in sys.modules,
                "first_call_seconds": cold,
                "second_call_seconds": warm,
                "includes": (
                    "first-call lazy ITK import and complete operation, "
                    "not Python/VIPP startup"
                ),
            }
        )
    )


def _cold_start(args):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--output",
        str(args.output),
        "--cold-child",
    ]
    if args.dependencies:
        command.extend(["--dependencies", str(args.dependencies)])
    completed = subprocess.run(
        command, capture_output=True, text=True, timeout=120, check=True
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _run_case(case, args, recorder, libraries):
    np, psutil, sitk, ndi, ops, median = libraries
    count = math.prod(case.shape)
    required = count * 48 + 512 * 1024**2
    available = psutil.virtual_memory().available
    if available < required:
        return {
            "case": asdict(case),
            "name": case.name,
            "status": "skipped_memory_guard",
            "estimated_required_bytes": required,
            "available_bytes": available,
        }
    array = _input(case, np, ndi)
    input_hash = _hash(array.tobytes())
    methods = _functions(
        case, np=np, sitk=sitk, ndi=ndi, ops=ops, median=median, threads=args.threads
    )
    result = {
        "case": asdict(case),
        "name": case.name,
        "status": "running",
        "input_sha256": input_hash,
        "backends": {},
    }
    if case.operation == "median":
        result["runtime_backend"] = median.median_filter_backend(
            array, size=case.size, xy_axes=(array.ndim - 2, array.ndim - 1)
        )
    elif case.operation == "distance":
        result["runtime_backend"] = "candidate-only; not used by VIPP"
    started_case = time.perf_counter()
    reference = None
    for name, function in methods.items():
        recorder.event(f"{case.name}: warmup {name}")
        started = time.perf_counter()
        output = function(array)
        elapsed = time.perf_counter() - started
        if reference is None:
            reference = output
        comparison = _compare(reference, output, np)
        result["backends"][name] = {
            "warmup_seconds": elapsed,
            "agreement": comparison,
            "seconds": [],
        }
        if input_hash != _hash(array.tobytes()):
            raise AssertionError(f"{name} mutated the input")
        if np.shares_memory(array, output):
            raise AssertionError(f"{name} returned an input alias")
        if time.perf_counter() - started_case > args.max_case_seconds:
            result["status"] = "skipped_timing_budget_after_warmup"
            return result
    del output, reference
    rng = np.random.default_rng(SEED + sum(case.shape) + case.size)
    for iteration in range(args.rounds):
        for name in rng.permutation(list(methods)):
            gc.collect()
            started = time.perf_counter()
            output = methods[name](array)
            elapsed = time.perf_counter() - started
            if input_hash != _hash(array.tobytes()):
                raise AssertionError(f"{name} mutated the input in timed round")
            del output
            result["backends"][name]["seconds"].append(elapsed)
            recorder.event(f"{case.name}: {name} round {iteration + 1}: {elapsed:.6f}s")
        if time.perf_counter() - started_case > args.max_case_seconds:
            result["status"] = "partial_timing_budget"
            break
    for backend in result["backends"].values():
        timings = backend["seconds"]
        if timings:
            backend["median_seconds"] = statistics.median(timings)
            backend["min_seconds"] = min(timings)
            backend["max_seconds"] = max(timings)
    if result["status"] == "running":
        result["status"] = "completed"
    result["all_checked_outputs_exact"] = all(
        item["agreement"]["bitwise_equal"] for item in result["backends"].values()
    )
    return result


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.rounds < 1 or args.threads < 1 or args.max_case_seconds <= 0:
        raise SystemExit(
            "Rounds, threads and the per-case time allowance must be positive."
        )
    if args.cold_child:
        _cold_child(args.dependencies)
        return 0
    args.output = args.output.resolve()
    if args.output.exists() and not args.resume:
        raise SystemExit(
            f"Existing evidence preserved: {args.output}; use a new name or --resume."
        )
    if args.resume and not args.output.exists():
        raise SystemExit("--resume requires an existing evidence file.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    libraries = _load(args.dependencies)
    np, psutil, sitk, ndi, ops, median = libraries
    selected = [
        case for case in cases(args.profile) if case.operation in args.operations
    ]
    specification = {
        "profile": args.profile,
        "rounds": args.rounds,
        "threads": args.threads,
        "seed": SEED,
        "cases": [case.name for case in selected],
    }
    if args.resume:
        report = json.loads(args.output.read_text(encoding="utf-8"))
        if (
            report["schema"] != SCHEMA
            or report["specification"] != specification
            or report["environment"]["source_sha256"] != _source_hashes()
        ):
            raise SystemExit(
                "Cannot combine evidence with different source hashes or settings; "
                "use a new output name."
            )
        report["status"] = "running"
    else:
        report = {
            "schema": SCHEMA,
            "status": "running",
            "specification": specification,
            "environment": _environment(psutil, sitk),
            "results": [],
            "method": {
                "timing": (
                    "One warmup then seeded shuffled backend order; imports/data "
                    "generation/hash checks/garbage collection excluded from timing. "
                    "Input conversion, symmetric halo, ITK execution, output "
                    "conversion/copies and the actual runtime gate included. "
                    "CPU only; no GPU timings here."
                ),
                "correctness": (
                    "Checked warmup outputs only; exact shape/dtype/value and byte "
                    "comparisons. Read-only input and no-alias checks. Adversarial "
                    "small phantoms plus larger timed fixtures. This screen does not "
                    "qualify all dtypes, axes or production lifecycle contracts."
                ),
                "distance": (
                    "Voxel-centre distances, unit spacing, float32 output, 2D plane "
                    "or 3D volume. Invert before SignedMaurer; zero original "
                    "background. Preserve all-foreground/empty SciPy behavior "
                    "with explicit fallback."
                ),
                "morphology": (
                    "XY box, odd width, leading Z slices independent, zero boundary "
                    "on both composed primitive stages. Even widths excluded: "
                    "SciPy origin is not a radius conversion."
                ),
                "median": (
                    "Forced candidate bypasses the performance gate; vipp_runtime "
                    "includes the selected production gate. Values, borders, dtype "
                    "and XY-only semantics must match reference exactly."
                ),
                "memory": (
                    "Conservative admission estimate of 48 bytes/input voxel plus "
                    "512MiB reserve; no measured peak memory claim. Candidate/ITK "
                    "memory copies exceed the adapter's padded-input chunk size."
                ),
            },
            "excluded": {
                "rolling_ball_background_and_subtract_background": (
                    "No matching SimpleITK rolling-ball exposed filter. "
                    "Morphological opening is not an equivalent replacement. "
                    "Mean presmoothing separately checked; median acceleration "
                    "does not speed these nodes."
                ),
                "gaussian_rl_connected_components": (
                    "Already investigated; intentionally outside this "
                    "exact-replacement follow-on."
                ),
            },
        }
    recorder = Recorder(args.output, report)
    recorder.event("Starting bounded SimpleITK CPU comparison")
    try:
        if "qualification" not in report:
            report["qualification"] = qualify(
                np=np, sitk=sitk, ndi=ndi, ops=ops, median=median, threads=args.threads
            )
            recorder.event(
                "Small adversarial morphology/distance and "
                "background mean checks complete"
            )
        if not args.skip_cold_start and "cold_start" not in report:
            report["cold_start"] = _cold_start(args)
            recorder.event("Fresh-process first-call check complete")
        if not args.qualify_only:
            complete = {row["name"] for row in report["results"]}
            for case in selected:
                if case.name in complete:
                    continue
                try:
                    result = _run_case(case, args, recorder, libraries)
                except Exception as exc:
                    result = {
                        "name": case.name,
                        "case": asdict(case),
                        "status": "failed",
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                report["results"].append(result)
                recorder.event(f"{case.name}: {result['status']}")
        report["status"] = (
            "completed_with_failures"
            if any(row["status"] != "completed" for row in report["results"])
            else "completed"
        )
        recorder.event(f"Comparison {report['status']}")
    except BaseException as exc:
        report["status"] = (
            "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        )
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        recorder.event(f"Comparison {report['status']}: {exc}")
        raise
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
