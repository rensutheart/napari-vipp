"""Create reproducible CuPy Richardson--Lucy TV admission evidence.

The command compares VIPP's authoritative progress-aware CPU implementation
with the resident CuPy provider.  It has two deliberately separate numerical
profiles:

* ``lambda == 0`` inherits ordinary Richardson--Lucy's strict tolerance and
  must also be exactly equivalent to the corresponding ordinary RL provider;
* the exact shipped positive-TV defaults use the versioned nonlinear RL-TV
  tolerance, followed by microscopy-feature, MSE, border-MSE, and flux gates.

The inherited 164-fixture RL matrix is supplemented by an independent
96-fixture holdout designed around edges, sparse/dim signal, borders, and
anisotropic 3D shapes.  Importing this module, asking for help, or validating a
committed artifact does not import CuPy or initialize CUDA.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal as scipy_signal

SCHEMA = "napari-vipp-cupy-rl-tv-admission-evidence"
SCHEMA_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Direct ``python scripts/...`` execution places only ``scripts`` on
    # sys.path.  The project root is needed for the reusable deterministic
    # fixture and phantom modules; this does not import optional GPU packages.
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "docs" / "benchmarks" / "rl-tv-cupy-admission-windows-rtx5090.json"
)

IMPLEMENTATION_ID = "rl-tv-cupy-f32-v1"
ORDINARY_IMPLEMENTATION_ID = "rl-cupy-f32-v1"
RUNTIME_ID = "cuda-cupy"
STRICT_NRMSE_LIMIT = 2e-6
STRICT_MAX_ABS_BASE = 1e-6
STRICT_MAX_ABS_PEAK_FACTOR = 5e-6
POSITIVE_NRMSE_LIMIT = 5e-3
POSITIVE_MAX_ABS_BASE = 1e-6
POSITIVE_MAX_ABS_PEAK_FACTOR = 5e-3
FEATURE_ABSOLUTE_LIMIT = 5e-3
METRIC_RELATIVE_LIMIT = 5e-3
ITERATIONS = (10, 25)
LAMBDA_ZERO_FILTER_EPSILON = 1e-8
POSITIVE_REGULARIZATION = 0.002
POSITIVE_TV_EPSILON = 1e-6
POSITIVE_FILTER_EPSILON = 1e-12
POSITIVE_DENOMINATOR_FLOOR = 0.05

PUBLICATION_SCOPE: dict[str, bool] = {
    "development_branch_public_exposure_justified": True,
    "cross_platform_promotion_justified": False,
    "released_package_promotion_justified": False,
}

SOURCE_PROVENANCE_PATHS = (
    "scripts/benchmark_gpu_rl_tv_admission.py",
    "scripts/benchmark_gpu_rl_admission.py",
    "scripts/validate_rl_tv_phantoms.py",
    "src/napari_vipp/core/richardson_lucy.py",
    "src/napari_vipp/core/richardson_lucy_compute.py",
    "src/napari_vipp/core/richardson_lucy_parity.py",
    "src/napari_vipp/core/gpu/cupy_rl.py",
    "src/napari_vipp/core/gpu/cupy_rl_tv.py",
    "src/napari_vipp/core/gpu/cupy_runtime.py",
    "src/napari_vipp/core/progress.py",
)

HOLDOUT_SHAPE_PSF_PAIRS = (
    ((95, 101), (11, 15)),
    ((191, 197), (17, 21)),
    ((11, 37, 41), (3, 7, 9)),
    ((19, 65, 67), (7, 13, 15)),
)
HOLDOUT_SEEDS = tuple(range(6100, 6104))
HOLDOUT_FAMILIES = (
    "positive_poisson",
    "sparse_dynamic_beads",
    "zero_heavy_dynamic_range",
    "step_and_border",
    "ramp_checkerboard",
    "dim_near_bright",
)

BENCHMARK_CONTRACT: dict[str, object] = {
    "generator": "vipp-rl-tv-admission-v1",
    "authoritative_cpu": "progress-aware-vipp-richardson-lucy-tv",
    "candidate": IMPLEMENTATION_ID,
    "input_dtype": "float32",
    "fixture_matrices": {
        "inherited_rl_adversarial": {
            "fixture_count": 164,
            "source": "scripts/benchmark_gpu_rl_admission.py final_odd_164",
        },
        "independent_holdout": {
            "fixture_count": 96,
            "seeds": list(HOLDOUT_SEEDS),
            "shape_psf_pairs": [
                [list(image_shape), list(psf_shape)]
                for image_shape, psf_shape in HOLDOUT_SHAPE_PSF_PAIRS
            ],
            "families": list(HOLDOUT_FAMILIES),
            "psfs": "alternating normalized Gaussian and asymmetric positive",
        },
    },
    "lambda_zero_profile": {
        "iterations": list(ITERATIONS),
        "tv_regularization": 0.0,
        "filter_epsilon": LAMBDA_ZERO_FILTER_EPSILON,
        "ordinary_rl_equivalence": "bitwise CPU and bitwise GPU",
        "nrmse_limit": STRICT_NRMSE_LIMIT,
        "max_abs_formula": "1e-6 + 5e-6 * max(abs(cpu_reference))",
    },
    "positive_default_profile": {
        "iterations": list(ITERATIONS),
        "tv_regularization": POSITIVE_REGULARIZATION,
        "tv_epsilon": POSITIVE_TV_EPSILON,
        "filter_epsilon": POSITIVE_FILTER_EPSILON,
        "denominator_floor": POSITIVE_DENOMINATOR_FLOOR,
        "nrmse_limit": POSITIVE_NRMSE_LIMIT,
        "max_abs_formula": "1e-6 + 0.005 * max(abs(cpu_reference))",
        "feature_absolute_limit": FEATURE_ABSOLUTE_LIMIT,
        "mse_border_flux_relative_limit": METRIC_RELATIVE_LIMIT,
    },
    "parameter_region": {
        "normalize_psf": True,
        "clip_negative_input": True,
        "clip_output_negative": True,
        "preserve_input_scale": True,
        "odd_psf_extents": True,
        "lambda_zero_maximum_iterations": 25,
        "positive_iterations": list(ITERATIONS),
    },
}


class EvidenceError(RuntimeError):
    """A complete, reviewable RL-TV evidence artifact could not be produced."""


@dataclass(frozen=True, slots=True)
class _Fixture:
    fixture_id: str
    group_id: str
    family: str
    image: np.ndarray
    psf: np.ndarray

    @property
    def spatial_mode(self) -> str:
        return "3D ZYX" if self.psf.ndim == 3 else "2D YX"

    @property
    def spatial_rank(self) -> int:
        return int(self.psf.ndim)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"JSON evidence path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        help="Readable summary path (default: output path with .md suffix).",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="CUDA device index (default: 0).",
    )
    parser.add_argument(
        "--validate-existing",
        type=Path,
        help="Validate an existing artifact without importing CuPy.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.validate_existing is not None:
        try:
            document = json.loads(args.validate_existing.read_text(encoding="utf-8"))
            validate_evidence_document(document, require_current_sources=True)
        except (EvidenceError, OSError, TypeError, ValueError) as exc:
            print(f"RL-TV admission evidence is invalid: {exc}", file=sys.stderr)
            return 2
        print(
            f"RL-TV admission evidence is current: {args.validate_existing.resolve()}"
        )
        return 0

    try:
        document = run_evidence(device_index=args.device_index)
        validate_evidence_document(document, require_current_sources=True)
        output = _atomic_write_text(args.output, _strict_json_text(document))
        markdown_path = args.markdown or output.with_suffix(".md")
        markdown = _atomic_write_text(markdown_path, render_markdown(document))
    except (EvidenceError, OSError, TypeError, ValueError) as exc:
        print(f"RL-TV admission evidence run failed: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(
            f"RL-TV admission evidence run failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    print(f"Wrote RL-TV admission evidence to {output}")
    print(f"Wrote readable RL-TV admission summary to {markdown}")
    return 0


def benchmark_contract_digest() -> str:
    encoded = json.dumps(
        BENCHMARK_CONTRACT,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_evidence(*, device_index: int = 0) -> dict[str, object]:
    """Run the fixed matrices against the real registered CuPy providers."""

    if isinstance(device_index, bool) or not isinstance(device_index, int):
        raise TypeError("device_index must be an integer")
    source_provenance = _source_provenance()
    inherited = tuple(_inherited_fixtures())
    holdout = tuple(_holdout_fixtures())
    _require_fixture_contract(inherited=inherited, holdout=holdout)
    fixtures = inherited + holdout

    # These imports are intentionally inside the GPU execution command.  Help,
    # import, fixture validation, and --validate-existing stay CUDA-free.
    import cupy

    from napari_vipp.core.gpu.cupy_rl import (
        richardson_lucy_deconvolution as gpu_rl,
    )
    from napari_vipp.core.gpu.cupy_rl_tv import (
        richardson_lucy_tv_deconvolution as gpu_tv,
    )
    from napari_vipp.core.gpu.cupy_runtime import CuPyRuntime
    from napari_vipp.core.progress import ProgressContext
    from napari_vipp.core.richardson_lucy import (
        richardson_lucy_deconvolution as cpu_rl,
    )
    from napari_vipp.core.richardson_lucy import (
        richardson_lucy_tv_deconvolution as cpu_tv,
    )

    runtime = CuPyRuntime()
    try:
        probe = runtime.probe()
        if not probe.available:
            raise EvidenceError(probe.message or "The CuPy runtime is unavailable.")
        selected_device = f"cuda:{device_index}"
        if selected_device not in {item.device_id for item in probe.devices}:
            raise EvidenceError(f"CUDA device index {device_index} is unavailable.")
        environment = _environment_fingerprint(
            cupy,
            probe,
            selected_device=selected_device,
        )

        lambda_zero_records: list[dict[str, object]] = []
        positive_records: list[dict[str, object]] = []
        phantom_records: list[dict[str, object]] = []
        with runtime.execution_scope(device_id=selected_device):
            for iteration_count in ITERATIONS:
                print(
                    f"Lambda-zero equivalence: {iteration_count} iterations "
                    f"over {len(fixtures)} fixtures.",
                    flush=True,
                )
                for case_index, fixture in enumerate(fixtures, start=1):
                    common = _common_parameters(fixture, iteration_count)
                    zero_kwargs = {
                        **common,
                        "tv_regularization": 0.0,
                        "tv_epsilon": POSITIVE_TV_EPSILON,
                        "filter_epsilon": LAMBDA_ZERO_FILTER_EPSILON,
                        "denominator_floor": POSITIVE_DENOMINATOR_FLOOR,
                    }
                    ordinary_kwargs = {
                        **common,
                        "filter_epsilon": LAMBDA_ZERO_FILTER_EPSILON,
                    }
                    cpu_ordinary = cpu_rl(
                        [fixture.image, fixture.psf],
                        progress=ProgressContext(),
                        **ordinary_kwargs,
                    )
                    cpu_zero = cpu_tv(
                        [fixture.image, fixture.psf],
                        progress=ProgressContext(),
                        **zero_kwargs,
                    )
                    gpu_ordinary = _gpu_call(
                        runtime,
                        gpu_rl,
                        fixture,
                        ordinary_kwargs,
                        selected_device=selected_device,
                        progress_type=ProgressContext,
                    )
                    gpu_zero = _gpu_call(
                        runtime,
                        gpu_tv,
                        fixture,
                        zero_kwargs,
                        selected_device=selected_device,
                        progress_type=ProgressContext,
                    )
                    parity = _parity_record(
                        cpu_ordinary,
                        gpu_zero,
                        nrmse_limit=STRICT_NRMSE_LIMIT,
                        max_abs_base=STRICT_MAX_ABS_BASE,
                        max_abs_peak_factor=STRICT_MAX_ABS_PEAK_FACTOR,
                    )
                    cpu_exact = bool(np.array_equal(cpu_ordinary, cpu_zero))
                    gpu_exact = bool(np.array_equal(gpu_ordinary, gpu_zero))
                    lambda_zero_records.append(
                        {
                            **_fixture_record(fixture),
                            "iterations": iteration_count,
                            "parameters": _serialized_parameters(zero_kwargs),
                            "cpu_tv_equals_cpu_rl_bitwise": cpu_exact,
                            "gpu_tv_equals_gpu_rl_bitwise": gpu_exact,
                            "parity": parity,
                            "passed": bool(
                                cpu_exact and gpu_exact and parity["passed"]
                            ),
                        }
                    )
                    if case_index % 50 == 0:
                        print(f"  completed {case_index}/{len(fixtures)}", flush=True)

                print(
                    f"Positive shipped profile: {iteration_count} iterations "
                    f"over {len(fixtures)} fixtures.",
                    flush=True,
                )
                for case_index, fixture in enumerate(fixtures, start=1):
                    positive_kwargs = {
                        **_common_parameters(fixture, iteration_count),
                        "tv_regularization": POSITIVE_REGULARIZATION,
                        "tv_epsilon": POSITIVE_TV_EPSILON,
                        "filter_epsilon": POSITIVE_FILTER_EPSILON,
                        "denominator_floor": POSITIVE_DENOMINATOR_FLOOR,
                    }
                    expected = cpu_tv(
                        [fixture.image, fixture.psf],
                        progress=ProgressContext(),
                        **positive_kwargs,
                    )
                    diagnostic_output, diagnostics = _reference_positive_diagnostics(
                        fixture.image,
                        fixture.psf,
                        iterations=iteration_count,
                    )
                    if not np.array_equal(expected, diagnostic_output):
                        raise EvidenceError(
                            f"Reference diagnostics changed CPU output for "
                            f"{fixture.fixture_id}."
                        )
                    actual = _gpu_call(
                        runtime,
                        gpu_tv,
                        fixture,
                        positive_kwargs,
                        selected_device=selected_device,
                        progress_type=ProgressContext,
                    )
                    parity = _parity_record(
                        expected,
                        actual,
                        nrmse_limit=POSITIVE_NRMSE_LIMIT,
                        max_abs_base=POSITIVE_MAX_ABS_BASE,
                        max_abs_peak_factor=POSITIVE_MAX_ABS_PEAK_FACTOR,
                    )
                    positive_records.append(
                        {
                            **_fixture_record(fixture),
                            "iterations": iteration_count,
                            "parameters": _serialized_parameters(positive_kwargs),
                            "reference_diagnostics": diagnostics,
                            "parity": parity,
                            "passed": bool(parity["passed"]),
                        }
                    )
                    if case_index % 50 == 0:
                        print(f"  completed {case_index}/{len(fixtures)}", flush=True)

            phantom_records = _run_phantom_matrix(
                runtime,
                gpu_tv,
                selected_device=selected_device,
                progress_type=ProgressContext,
            )

        terminal = runtime.memory_snapshot(device_id=selected_device)
        cleanup = {
            "runtime_id": terminal.runtime_id,
            "device_id": terminal.device_id,
            "runtime_live_bytes": terminal.runtime_live_bytes,
            "runtime_reserved_bytes": terminal.runtime_reserved_bytes,
            "out_of_pool_bytes": terminal.out_of_pool_bytes,
            "terminal_private_pool_zero": bool(
                terminal.runtime_live_bytes == 0
                and terminal.runtime_reserved_bytes == 0
            ),
        }
    finally:
        runtime.close()

    _require_source_snapshot_unchanged(source_provenance)
    lambda_zero = _matrix_document(lambda_zero_records)
    positive = _matrix_document(positive_records, diagnostics=True)
    all_passed = bool(
        all(record["passed"] for record in lambda_zero_records)
        and all(record["passed"] for record in positive_records)
        and all(record["passed"] for record in phantom_records)
        and cleanup["terminal_private_pool_zero"]
    )
    if not all_passed:
        raise EvidenceError("One or more reviewed RL-TV admission gates failed.")
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "complete",
        "contract": BENCHMARK_CONTRACT,
        "contract_sha256": benchmark_contract_digest(),
        "source_provenance": source_provenance,
        "environment": environment,
        "fixtures": {
            "inherited_count": len(inherited),
            "holdout_count": len(holdout),
            "total_count": len(fixtures),
            "inherited_manifest_sha256": _fixture_manifest_digest(inherited),
            "holdout_manifest_sha256": _fixture_manifest_digest(holdout),
            "combined_manifest_sha256": _fixture_manifest_digest(fixtures),
        },
        "matrices": {
            "lambda_zero_strict": lambda_zero,
            "positive_shipped_default": positive,
            "maintained_phantoms": {
                "record_count": len(phantom_records),
                "records": phantom_records,
            },
        },
        "cleanup": cleanup,
        "conclusion": {
            "all_reviewed_gates_passed": all_passed,
            "admitted_positive_profile_is_exact_not_a_range": True,
            "lambda_zero_inherits_strict_rl_gate": True,
            "positive_profile_uses_versioned_rl_tv_gate": True,
            **PUBLICATION_SCOPE,
        },
        "limitations": [
            "This is single-host native-Windows RTX 5090 evidence.",
            "The positive profile is the exact shipped parameter tuple, not a "
            "continuous lambda, epsilon, floor, or iteration claim.",
            "Synthetic matrices cannot replace calibrated biological datasets, "
            "bead data, or blinded expert review.",
            "Exact-workload parity remains mandatory before optimizer selection.",
            "No authored parameter is silently changed to qualify for GPU execution.",
            "This artifact supports public exposure only inside the reviewed exact "
            "regions on this development branch.",
            "Cross-platform support and released-package promotion are not claimed.",
        ],
    }


def _gpu_call(
    runtime: Any,
    function: Any,
    fixture: _Fixture,
    parameters: Mapping[str, object],
    *,
    selected_device: str,
    progress_type: Any,
) -> np.ndarray:
    device_image = runtime.to_device(fixture.image, device_id=selected_device)
    device_psf = runtime.to_device(fixture.psf, device_id=selected_device)
    output = function(
        [device_image, device_psf],
        progress=progress_type(),
        **dict(parameters),
    )
    runtime.synchronize(device_id=selected_device)
    host = np.ascontiguousarray(runtime.to_host(output), dtype=np.float32)
    output = None
    device_image = None
    device_psf = None
    return host


def _common_parameters(fixture: _Fixture, iterations: int) -> dict[str, object]:
    return {
        "spatial_mode": fixture.spatial_mode,
        "iterations": int(iterations),
        "normalize_psf": True,
        "clip_negative_input": True,
        "clip_output_negative": True,
        "preserve_input_scale": True,
    }


def _serialized_parameters(parameters: Mapping[str, object]) -> dict[str, object]:
    return {str(name): value for name, value in sorted(parameters.items())}


def _fixture_record(fixture: _Fixture) -> dict[str, object]:
    return {
        "fixture_id": fixture.fixture_id,
        "group_id": fixture.group_id,
        "family": fixture.family,
        "spatial_rank": fixture.spatial_rank,
        "image_shape": list(fixture.image.shape),
        "psf_shape": list(fixture.psf.shape),
    }


def _parity_record(
    expected: np.ndarray,
    actual: np.ndarray,
    *,
    nrmse_limit: float,
    max_abs_base: float,
    max_abs_peak_factor: float,
) -> dict[str, object]:
    shape_equal = actual.shape == expected.shape
    dtype_equal = actual.dtype == expected.dtype == np.dtype(np.float32)
    expected_finite = np.isfinite(expected)
    actual_finite = np.isfinite(actual)
    finite_masks_equal = bool(np.array_equal(expected_finite, actual_finite))
    completely_finite = bool(np.all(expected_finite) and np.all(actual_finite))
    nonnegative = bool(np.all(expected >= 0) and np.all(actual >= 0))
    if shape_equal:
        expected64 = expected.astype(np.float64)
        actual64 = actual.astype(np.float64)
        difference = actual64 - expected64
        peak = float(np.max(np.abs(expected64), initial=0.0))
        max_abs = float(np.max(np.abs(difference), initial=0.0))
        denominator = max(
            float(np.linalg.norm(expected64.ravel())),
            float(np.sqrt(expected64.size) * 1e-12),
        )
        nrmse = float(np.linalg.norm(difference.ravel()) / denominator)
    else:
        peak = max_abs = nrmse = float("inf")
    max_abs_limit = float(max_abs_base + max_abs_peak_factor * peak)
    gate_score = max(nrmse / nrmse_limit, max_abs / max_abs_limit)
    passed = bool(
        shape_equal
        and dtype_equal
        and finite_masks_equal
        and completely_finite
        and nonnegative
        and nrmse <= nrmse_limit
        and max_abs <= max_abs_limit
    )
    return {
        "shape_equal": shape_equal,
        "dtype_equal": dtype_equal,
        "finite_masks_equal": finite_masks_equal,
        "completely_finite": completely_finite,
        "nonnegative": nonnegative,
        "cpu_peak": peak,
        "nrmse": nrmse,
        "nrmse_limit": nrmse_limit,
        "max_abs": max_abs,
        "max_abs_limit": max_abs_limit,
        "gate_score": gate_score,
        "passed": passed,
    }


def _reference_positive_diagnostics(
    image: np.ndarray,
    psf: np.ndarray,
    *,
    iterations: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Replay the authoritative native loop while counting guard activity."""

    from napari_vipp.core import richardson_lucy as cpu_contract

    kernel = cpu_contract._deconvolution_psf(psf, psf.ndim, normalize_psf=True)
    values, output_scale = cpu_contract._deconvolution_observed_block(
        image,
        clip_negative_input=True,
        preserve_input_scale=True,
    )
    estimate = np.full(values.shape, 0.5, dtype=np.float32)
    mirror = np.flip(kernel)
    threshold_samples = 0
    threshold_iterations = 0
    floor_samples = 0
    floor_iterations = 0
    minimum_raw_denominator = float("inf")
    maximum_floor_fraction = 0.0
    for _ in range(int(iterations)):
        blurred = scipy_signal.convolve(estimate, kernel, mode="same") + np.float32(
            1e-12
        )
        threshold_mask = blurred < POSITIVE_FILTER_EPSILON
        threshold_count = int(np.count_nonzero(threshold_mask))
        threshold_samples += threshold_count
        threshold_iterations += int(threshold_count > 0)
        ratio = np.where(threshold_mask, 0.0, values / blurred)
        correction = scipy_signal.convolve(ratio, mirror, mode="same")
        divergence = cpu_contract._tv_divergence(
            estimate,
            epsilon=POSITIVE_TV_EPSILON,
        )
        raw = np.float32(1.0) - np.float32(POSITIVE_REGULARIZATION) * divergence
        minimum_raw_denominator = min(
            minimum_raw_denominator,
            float(np.min(raw)),
        )
        floor_mask = raw < POSITIVE_DENOMINATOR_FLOOR
        floor_count = int(np.count_nonzero(floor_mask))
        floor_samples += floor_count
        floor_iterations += int(floor_count > 0)
        maximum_floor_fraction = max(
            maximum_floor_fraction,
            float(np.mean(floor_mask)),
        )
        estimate = (
            estimate
            * correction
            / np.maximum(
                raw,
                np.float32(POSITIVE_DENOMINATOR_FLOOR),
            )
        )
        estimate = np.maximum(
            np.nan_to_num(estimate, nan=0.0, posinf=0.0, neginf=0.0),
            0.0,
        ).astype(np.float32, copy=False)
    output = cpu_contract._deconvolution_output_block(
        estimate,
        output_scale=output_scale,
        clip_output_negative=True,
    )
    total_samples = int(values.size) * int(iterations)
    return output, {
        "reference_threshold_active_samples": threshold_samples,
        "reference_threshold_active_iterations": threshold_iterations,
        "reference_threshold_fraction": (
            float(threshold_samples / total_samples) if total_samples else 0.0
        ),
        "minimum_raw_denominator": minimum_raw_denominator,
        "reference_floor_active_samples": floor_samples,
        "reference_floor_active_iterations": floor_iterations,
        "maximum_floor_fraction": maximum_floor_fraction,
        "total_voxel_iterations": total_samples,
    }


def _run_phantom_matrix(
    runtime: Any,
    gpu_tv: Any,
    *,
    selected_device: str,
    progress_type: Any,
) -> list[dict[str, object]]:
    from napari_vipp.core.richardson_lucy import (
        richardson_lucy_tv_deconvolution as cpu_tv,
    )
    from scripts.validate_rl_tv_phantoms import (
        calculate_metrics,
        make_phantom_2d,
        make_phantom_3d,
        observed_image,
    )

    records: list[dict[str, object]] = []
    for phantom in (make_phantom_2d(), make_phantom_3d()):
        image = np.ascontiguousarray(observed_image(phantom), dtype=np.float32)
        fixture = _Fixture(
            fixture_id=f"maintained-{phantom.name.casefold()}-phantom",
            group_id="maintained_phantoms",
            family="microscopy_feature_phantom",
            image=image,
            psf=np.ascontiguousarray(phantom.psf, dtype=np.float32),
        )
        for iteration_count in ITERATIONS:
            parameters = {
                **_common_parameters(fixture, iteration_count),
                "tv_regularization": POSITIVE_REGULARIZATION,
                "tv_epsilon": POSITIVE_TV_EPSILON,
                "filter_epsilon": POSITIVE_FILTER_EPSILON,
                "denominator_floor": POSITIVE_DENOMINATOR_FLOOR,
            }
            expected = cpu_tv(
                [fixture.image, fixture.psf],
                progress=progress_type(),
                **parameters,
            )
            actual = _gpu_call(
                runtime,
                gpu_tv,
                fixture,
                parameters,
                selected_device=selected_device,
                progress_type=progress_type,
            )
            parity = _parity_record(
                expected,
                actual,
                nrmse_limit=POSITIVE_NRMSE_LIMIT,
                max_abs_base=POSITIVE_MAX_ABS_BASE,
                max_abs_peak_factor=POSITIVE_MAX_ABS_PEAK_FACTOR,
            )
            cpu_metrics = calculate_metrics(expected, phantom)
            gpu_metrics = calculate_metrics(actual, phantom)
            feature_deltas = {
                name: abs(float(gpu_metrics[name]) - float(cpu_metrics[name]))
                for name in (
                    "points_recovery",
                    "thin_line_recovery",
                    "dim_structure_recovery",
                )
            }
            relative_metric_deltas = {
                name: _relative_difference(cpu_metrics[name], gpu_metrics[name])
                for name in ("mse", "border_mse", "flux_ratio")
            }
            feature_passed = all(
                value <= FEATURE_ABSOLUTE_LIMIT for value in feature_deltas.values()
            )
            metrics_passed = all(
                value <= METRIC_RELATIVE_LIMIT
                for value in relative_metric_deltas.values()
            )
            records.append(
                {
                    **_fixture_record(fixture),
                    "iterations": iteration_count,
                    "parameters": _serialized_parameters(parameters),
                    "parity": parity,
                    "cpu_metrics": _finite_metric_mapping(cpu_metrics),
                    "gpu_metrics": _finite_metric_mapping(gpu_metrics),
                    "feature_absolute_deltas": feature_deltas,
                    "feature_absolute_limit": FEATURE_ABSOLUTE_LIMIT,
                    "metric_relative_deltas": relative_metric_deltas,
                    "metric_relative_limit": METRIC_RELATIVE_LIMIT,
                    "passed": bool(
                        parity["passed"] and feature_passed and metrics_passed
                    ),
                }
            )
    return records


def _relative_difference(reference: float, candidate: float) -> float:
    return float(abs(float(candidate) - float(reference)) / max(abs(reference), 1e-12))


def _finite_metric_mapping(values: Mapping[str, float]) -> dict[str, float]:
    result = {str(name): float(value) for name, value in sorted(values.items())}
    if not all(math.isfinite(value) for value in result.values()):
        raise EvidenceError("Phantom metrics must be finite.")
    return result


def _matrix_document(
    records: Sequence[Mapping[str, object]],
    *,
    diagnostics: bool = False,
) -> dict[str, object]:
    summaries: list[dict[str, object]] = []
    for iteration_count in ITERATIONS:
        selected = [
            record for record in records if record["iterations"] == iteration_count
        ]
        worst = max(
            selected, key=lambda item: float(_mapping(item, "parity")["gate_score"])
        )
        summary: dict[str, object] = {
            "iterations": iteration_count,
            "case_count": len(selected),
            "failure_count": sum(not bool(item["passed"]) for item in selected),
            "rank_counts": dict(
                sorted(Counter(str(item["spatial_rank"]) for item in selected).items())
            ),
            "worst_gate_score": float(_mapping(worst, "parity")["gate_score"]),
            "worst_fixture_id": worst["fixture_id"],
            "worst_nrmse": max(
                float(_mapping(item, "parity")["nrmse"]) for item in selected
            ),
            "worst_max_abs_ratio": max(
                float(_mapping(item, "parity")["max_abs"])
                / max(float(_mapping(item, "parity")["max_abs_limit"]), 1e-30)
                for item in selected
            ),
        }
        if diagnostics:
            diagnostic_records = [
                _mapping(item, "reference_diagnostics") for item in selected
            ]
            summary.update(
                {
                    "threshold_active_case_count": sum(
                        int(item["reference_threshold_active_samples"]) > 0
                        for item in diagnostic_records
                    ),
                    "threshold_active_sample_count": sum(
                        int(item["reference_threshold_active_samples"])
                        for item in diagnostic_records
                    ),
                    "floor_active_case_count": sum(
                        int(item["reference_floor_active_samples"]) > 0
                        for item in diagnostic_records
                    ),
                    "floor_active_sample_count": sum(
                        int(item["reference_floor_active_samples"])
                        for item in diagnostic_records
                    ),
                    "minimum_raw_denominator": min(
                        float(item["minimum_raw_denominator"])
                        for item in diagnostic_records
                    ),
                }
            )
        summaries.append(summary)
    return {
        "fixture_count": len(records) // len(ITERATIONS),
        "result_count": len(records),
        "iterations": list(ITERATIONS),
        "summaries": summaries,
        "records": list(records),
    }


def _inherited_fixtures() -> Iterable[_Fixture]:
    from scripts.benchmark_gpu_rl_admission import (
        _gaussian_core_fixtures,
        _odd_asymmetric_fixtures,
        _sparse_seed_sweep_fixtures,
    )

    for item in (
        *_gaussian_core_fixtures(),
        *_odd_asymmetric_fixtures(),
        *_sparse_seed_sweep_fixtures(),
    ):
        yield _Fixture(
            fixture_id=item.fixture_id,
            group_id=f"inherited:{item.group_id}",
            family=item.family,
            image=item.image,
            psf=item.psf,
        )


def _holdout_fixtures() -> Iterable[_Fixture]:
    for seed_index, seed in enumerate(HOLDOUT_SEEDS):
        for image_shape, psf_shape in HOLDOUT_SHAPE_PSF_PAIRS:
            rng = np.random.default_rng(seed + sum(image_shape))
            psf = (
                _gaussian_psf(psf_shape)
                if seed_index % 2 == 0
                else _asymmetric_psf(psf_shape, rng)
            )
            prefix = f"holdout-s{seed}-{'x'.join(map(str, image_shape))}"

            image = rng.poisson(20, size=image_shape).astype(np.float32)
            image /= np.float32(image.max())
            yield _fixture(prefix, "positive_poisson", image, psf)

            latent = np.zeros(image_shape, dtype=np.float32)
            for _ in range(20):
                index = tuple(rng.integers(0, size) for size in image_shape)
                latent[index] = np.float32(10.0 ** rng.uniform(-7.0, 0.0))
            image = scipy_signal.convolve(latent, psf, mode="same").astype(np.float32)
            image = np.maximum(image, np.float32(0.0))
            image /= np.float32(max(float(image.max()), 1e-30))
            yield _fixture(prefix, "sparse_dynamic_beads", image, psf)

            image = np.power(10.0, rng.uniform(-12.0, 0.0, image_shape)).astype(
                np.float32
            )
            image[rng.random(image_shape) < 0.6] = np.float32(0.0)
            yield _fixture(prefix, "zero_heavy_dynamic_range", image, psf)

            image = np.zeros(image_shape, dtype=np.float32)
            region = tuple(slice(size // 4, 3 * size // 4) for size in image_shape)
            image[region] = np.float32(0.01)
            image[tuple(size // 2 for size in image_shape)] = np.float32(1.0)
            border = [size // 2 for size in image_shape]
            border[-1] = 0
            image[tuple(border)] = np.float32(0.7)
            yield _fixture(prefix, "step_and_border", image, psf)

            coordinates = np.indices(image_shape, dtype=np.float32)
            ramp = sum(
                coordinates[axis] / np.float32(max(size - 1, 1))
                for axis, size in enumerate(image_shape)
            ) / np.float32(len(image_shape))
            checker = (np.indices(image_shape).sum(axis=0) % 2).astype(np.float32)
            image = (np.float32(0.95) * ramp + np.float32(0.05) * checker).astype(
                np.float32
            )
            yield _fixture(prefix, "ramp_checkerboard", image, psf)

            latent = np.zeros(image_shape, dtype=np.float32)
            center = [size // 2 for size in image_shape]
            line = [slice(None)] * len(image_shape)
            line[-2] = center[-2]
            line[-1] = slice(image_shape[-1] // 4, 3 * image_shape[-1] // 4)
            latent[tuple(line)] = np.float32(1.0)
            line[-2] = min(center[-2] + 2, image_shape[-2] - 1)
            latent[tuple(line)] = np.float32(0.08)
            image = scipy_signal.convolve(latent, psf, mode="same").astype(np.float32)
            image = np.maximum(image, np.float32(0.0))
            image /= np.float32(max(float(image.max()), 1e-30))
            yield _fixture(prefix, "dim_near_bright", image, psf)


def _fixture(
    prefix: str,
    family: str,
    image: np.ndarray,
    psf: np.ndarray,
) -> _Fixture:
    image = np.ascontiguousarray(image, dtype=np.float32)
    psf = np.ascontiguousarray(psf, dtype=np.float32)
    image.setflags(write=False)
    psf.setflags(write=False)
    return _Fixture(
        fixture_id=f"{prefix}-{family}",
        group_id="independent_holdout_96",
        family=family,
        image=image,
        psf=psf,
    )


def _gaussian_psf(shape: Sequence[int]) -> np.ndarray:
    grids = np.meshgrid(
        *(
            np.arange(size, dtype=np.float32) - np.float32((size - 1) / 2)
            for size in shape
        ),
        indexing="ij",
    )
    exponent = np.zeros(tuple(shape), dtype=np.float32)
    for grid, size in zip(grids, shape, strict=True):
        exponent += (grid / np.float32(max(size / 5.0, 0.7))) ** 2
    values = np.exp(np.float32(-0.5) * exponent).astype(np.float32)
    values /= np.float32(values.sum(dtype=np.float64))
    return np.ascontiguousarray(values)


def _asymmetric_psf(shape: Sequence[int], rng: np.random.Generator) -> np.ndarray:
    values = rng.power(3.0, size=tuple(shape)).astype(np.float32)
    values /= np.float32(values.sum(dtype=np.float64))
    return np.ascontiguousarray(values)


def _require_fixture_contract(
    *,
    inherited: Sequence[_Fixture],
    holdout: Sequence[_Fixture],
) -> None:
    if len(inherited) != 164:
        raise EvidenceError(f"Expected 164 inherited fixtures, got {len(inherited)}.")
    if len(holdout) != 96:
        raise EvidenceError(f"Expected 96 holdout fixtures, got {len(holdout)}.")
    fixtures = (*inherited, *holdout)
    identifiers = [item.fixture_id for item in fixtures]
    if len(identifiers) != len(set(identifiers)):
        raise EvidenceError("Fixture identifiers must be unique.")
    if Counter(item.spatial_rank for item in holdout) != {2: 48, 3: 48}:
        raise EvidenceError("The holdout must contain 48 2D and 48 3D fixtures.")
    if Counter(item.family for item in holdout) != {
        family: 16 for family in HOLDOUT_FAMILIES
    }:
        raise EvidenceError("The holdout family distribution changed.")
    for item in fixtures:
        if item.image.dtype != np.dtype(np.float32) or item.psf.dtype != np.dtype(
            np.float32
        ):
            raise EvidenceError("Every admission fixture must be float32.")
        if not np.isfinite(item.image).all() or not np.isfinite(item.psf).all():
            raise EvidenceError("Every admission fixture must be finite.")
        if any(size % 2 == 0 for size in item.psf.shape):
            raise EvidenceError("Every admitted PSF extent must be odd.")


def _fixture_manifest_digest(fixtures: Sequence[_Fixture]) -> str:
    digest = hashlib.sha256()
    for fixture in fixtures:
        digest.update(fixture.fixture_id.encode("utf-8"))
        digest.update(fixture.group_id.encode("utf-8"))
        digest.update(fixture.family.encode("utf-8"))
        for array in (fixture.image, fixture.psf):
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(repr(array.shape).encode("ascii"))
            digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    return digest.hexdigest()


def validate_evidence_document(
    document: Mapping[str, object],
    *,
    require_current_sources: bool,
) -> None:
    if not isinstance(document, Mapping):
        raise EvidenceError("Evidence must be a mapping.")
    if document.get("schema") != SCHEMA or document.get("schema_version") != 1:
        raise EvidenceError("Unexpected RL-TV evidence schema.")
    if document.get("status") != "complete":
        raise EvidenceError("Evidence status must be complete.")
    if document.get("contract") != BENCHMARK_CONTRACT:
        raise EvidenceError("Evidence contract differs from the current generator.")
    if document.get("contract_sha256") != benchmark_contract_digest():
        raise EvidenceError("Evidence contract digest is stale.")

    inherited = tuple(_inherited_fixtures())
    holdout = tuple(_holdout_fixtures())
    _require_fixture_contract(inherited=inherited, holdout=holdout)
    fixtures = inherited + holdout
    fixture_document = _mapping(document, "fixtures")
    expected_fixture_values = {
        "inherited_count": 164,
        "holdout_count": 96,
        "total_count": 260,
        "inherited_manifest_sha256": _fixture_manifest_digest(inherited),
        "holdout_manifest_sha256": _fixture_manifest_digest(holdout),
        "combined_manifest_sha256": _fixture_manifest_digest(fixtures),
    }
    for name, expected in expected_fixture_values.items():
        if fixture_document.get(name) != expected:
            raise EvidenceError(f"Fixture evidence field {name!r} is stale.")

    matrices = _mapping(document, "matrices")
    zero = _mapping(matrices, "lambda_zero_strict")
    positive = _mapping(matrices, "positive_shipped_default")
    phantoms = _mapping(matrices, "maintained_phantoms")
    _validate_matrix(zero, expected_records=520, lambda_zero=True)
    _validate_matrix(positive, expected_records=520, lambda_zero=False)
    phantom_records = _mapping_sequence(phantoms.get("records"), "phantom records")
    if phantoms.get("record_count") != 4 or len(phantom_records) != 4:
        raise EvidenceError("Maintained phantom evidence must contain four records.")
    for record in phantom_records:
        if record.get("passed") is not True:
            raise EvidenceError("A maintained phantom gate did not pass.")
        _validate_parity(_mapping(record, "parity"), positive=True)
        if any(
            float(value) > FEATURE_ABSOLUTE_LIMIT
            for value in _mapping(record, "feature_absolute_deltas").values()
        ):
            raise EvidenceError("A phantom feature delta exceeds policy.")
        if any(
            float(value) > METRIC_RELATIVE_LIMIT
            for value in _mapping(record, "metric_relative_deltas").values()
        ):
            raise EvidenceError("A phantom metric delta exceeds policy.")

    cleanup = _mapping(document, "cleanup")
    if (
        cleanup.get("terminal_private_pool_zero") is not True
        or cleanup.get("runtime_live_bytes") != 0
        or cleanup.get("runtime_reserved_bytes") != 0
    ):
        raise EvidenceError("CUDA terminal private-pool cleanup is incomplete.")
    conclusion = _mapping(document, "conclusion")
    if conclusion.get("all_reviewed_gates_passed") is not True:
        raise EvidenceError("Evidence conclusion is not passing.")
    for name, expected in PUBLICATION_SCOPE.items():
        if conclusion.get(name) is not expected:
            raise EvidenceError(f"Evidence publication scope field {name!r} is stale.")
    _validate_environment(_mapping(document, "environment"))
    _validate_privacy(document)
    _validate_source_provenance(
        _mapping(document, "source_provenance"),
        require_current_sources=require_current_sources,
    )


def _validate_matrix(
    matrix: Mapping[str, object],
    *,
    expected_records: int,
    lambda_zero: bool,
) -> None:
    records = _mapping_sequence(matrix.get("records"), "matrix records")
    summaries = _mapping_sequence(matrix.get("summaries"), "matrix summaries")
    if (
        matrix.get("fixture_count") != 260
        or matrix.get("result_count") != expected_records
    ):
        raise EvidenceError("Matrix fixture/result counts are incomplete.")
    if len(records) != expected_records or len(summaries) != len(ITERATIONS):
        raise EvidenceError("Matrix records or summaries are incomplete.")
    expected_per_iteration = expected_records // len(ITERATIONS)
    for iteration_count in ITERATIONS:
        selected = [
            item for item in records if item.get("iterations") == iteration_count
        ]
        if len(selected) != expected_per_iteration:
            raise EvidenceError("Matrix iteration coverage is incomplete.")
    for record in records:
        if record.get("passed") is not True:
            raise EvidenceError("A matrix record did not pass.")
        _validate_parity(_mapping(record, "parity"), positive=not lambda_zero)
        if lambda_zero:
            if (
                record.get("cpu_tv_equals_cpu_rl_bitwise") is not True
                or record.get("gpu_tv_equals_gpu_rl_bitwise") is not True
            ):
                raise EvidenceError("Lambda-zero ordinary-RL equivalence failed.")
        else:
            diagnostics = _mapping(record, "reference_diagnostics")
            for name in (
                "reference_threshold_active_samples",
                "reference_floor_active_samples",
                "total_voxel_iterations",
            ):
                value = diagnostics.get(name)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise EvidenceError(f"Invalid diagnostic count {name!r}.")
            minimum = diagnostics.get("minimum_raw_denominator")
            if not isinstance(minimum, (int, float)) or not math.isfinite(
                float(minimum)
            ):
                raise EvidenceError("Minimum raw denominator must be finite.")
    for summary in summaries:
        if summary.get("failure_count") != 0:
            raise EvidenceError("A matrix summary reports failures.")


def _validate_parity(parity: Mapping[str, object], *, positive: bool) -> None:
    if parity.get("passed") is not True:
        raise EvidenceError("A parity gate did not pass.")
    expected_nrmse = POSITIVE_NRMSE_LIMIT if positive else STRICT_NRMSE_LIMIT
    if parity.get("nrmse_limit") != expected_nrmse:
        raise EvidenceError("Parity NRMSE policy differs from the contract.")
    for name in ("nrmse", "max_abs", "max_abs_limit", "gate_score"):
        value = parity.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise EvidenceError(f"Parity field {name!r} must be finite.")


def render_markdown(document: Mapping[str, object]) -> str:
    environment = _mapping(document, "environment")
    matrices = _mapping(document, "matrices")
    zero = _mapping(matrices, "lambda_zero_strict")
    positive = _mapping(matrices, "positive_shipped_default")
    phantom_records = _mapping_sequence(
        _mapping(matrices, "maintained_phantoms").get("records"),
        "phantom records",
    )
    lines = [
        "# CuPy Richardson-Lucy TV admission evidence",
        "",
        f"- Device: `{environment['device_name']}` (`{environment['device_id']}`)",
        (
            f"- Python / CuPy: `{environment['python']}` / "
            f"`{environment['packages']['cupy-cuda13x']}`"
        ),
        "- Fixtures: `164` inherited adversarial + `96` independent holdout",
        (
            "- Status: **all reviewed gates passed for development-branch public "
            "exposure**"
        ),
        "",
        "The lambda-zero profile remains ordinary Richardson-Lucy and uses its",
        "strict numerical policy. The positive profile is only the exact shipped",
        "tuple (lambda 0.002, TV epsilon 1e-6, filter epsilon 1e-12, floor 0.05)",
        "at exactly 10 or 25 iterations; this is not a continuous parameter claim.",
        "",
        "## Numerical matrices",
        "",
        (
            "| Profile | Iterations | Cases | Failures | Worst gate score | "
            "Threshold-active cases | Floor-active cases | "
            "Minimum raw denominator |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for profile_name, matrix in (
        ("Lambda zero / strict RL", zero),
        ("Positive shipped default", positive),
    ):
        for summary in _mapping_sequence(matrix.get("summaries"), "summaries"):
            lines.append(
                "| "
                + " | ".join(
                    (
                        profile_name,
                        str(summary["iterations"]),
                        str(summary["case_count"]),
                        str(summary["failure_count"]),
                        f"{float(summary['worst_gate_score']):.6g}",
                        str(summary.get("threshold_active_case_count", "n/a")),
                        str(summary.get("floor_active_case_count", "n/a")),
                        (
                            f"{float(summary['minimum_raw_denominator']):.9g}"
                            if "minimum_raw_denominator" in summary
                            else "n/a"
                        ),
                    )
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Maintained microscopy phantoms",
            "",
            (
                "| Phantom | Iterations | NRMSE | Max feature delta | "
                "Max MSE/border/flux relative delta |"
            ),
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for record in phantom_records:
        maximum_feature_delta = max(
            float(value)
            for value in _mapping(record, "feature_absolute_deltas").values()
        )
        maximum_metric_delta = max(
            float(value)
            for value in _mapping(record, "metric_relative_deltas").values()
        )
        lines.append(
            "| "
            + " | ".join(
                (
                    str(record["fixture_id"]),
                    str(record["iterations"]),
                    f"{float(_mapping(record, 'parity')['nrmse']):.6g}",
                    f"{maximum_feature_delta:.6g}",
                    f"{maximum_metric_delta:.6g}",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Positive TV is nonlinear, so its versioned 0.5% numerical screen is",
            "  separate from lambda-zero ordinary RL. Feature-recovery, MSE, border",
            "  MSE, and flux gates prevent the aggregate tolerance from hiding a",
            "  material microscopy regression on the maintained phantoms.",
            "- Guard activity is diagnostic, not a reason to alter authored values.",
            "  The evidence records threshold and denominator-floor activity for",
            "  every positive-profile fixture.",
            "- This single-machine artifact supports public exposure for the",
            "  reviewed exact regions on this development branch.",
            "- It does not establish cross-platform support or released-package",
            "  promotion. Calibrated bead/biological datasets, blinded review, and",
            "  Linux/laptop evidence remain required before those broader claims.",
            "",
        ]
    )
    return "\n".join(lines)


def _environment_fingerprint(
    cupy: Any,
    probe: Any,
    *,
    selected_device: str,
) -> dict[str, object]:
    index = int(selected_device.partition(":")[2])
    properties = cupy.cuda.runtime.getDeviceProperties(index)
    name = properties.get("name", properties.get(b"name", "CUDA device"))
    if isinstance(name, bytes):
        name = name.decode(errors="replace")
    device = next(item for item in probe.devices if item.device_id == selected_device)
    package_names = (
        "napari-vipp",
        "numpy",
        "scipy",
        "scikit-image",
        "cupy-cuda13x",
    )
    packages = {name: importlib.metadata.version(name) for name in package_names}
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "device_id": selected_device,
        "device_name": str(name),
        "device_total_memory_bytes": int(device.total_memory_bytes),
        "compute_capability": dict(device.metadata).get("compute_capability", ""),
        "cuda_driver_version": str(int(cupy.cuda.runtime.driverGetVersion())),
        "cuda_runtime_version": str(int(cupy.cuda.runtime.runtimeGetVersion())),
        "runtime_probe_version": probe.version,
        "runtime_environment_fingerprint": probe.environment_fingerprint,
        "packages": packages,
        "validated_environment_policy_id": (
            "cuda-cupy-14.1.1-cpython312-windows-native-v3"
        ),
    }


def _validate_environment(environment: Mapping[str, object]) -> None:
    required = (
        "system",
        "release",
        "version",
        "machine",
        "python",
        "device_id",
        "device_name",
        "device_total_memory_bytes",
        "cuda_driver_version",
        "cuda_runtime_version",
        "runtime_environment_fingerprint",
        "packages",
    )
    if any(not environment.get(name) for name in required):
        raise EvidenceError("Environment provenance is incomplete.")
    packages = _mapping(environment, "packages")
    if set(packages) != {
        "napari-vipp",
        "numpy",
        "scipy",
        "scikit-image",
        "cupy-cuda13x",
    }:
        raise EvidenceError("Environment package provenance is incomplete.")


def _source_provenance() -> dict[str, object]:
    files = []
    for relative in SOURCE_PROVENANCE_PATHS:
        path = PROJECT_ROOT / relative
        files.append(
            {
                "relative_path": relative.replace("\\", "/"),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return {"files": files, "git": _git_provenance()}


def _git_provenance() -> dict[str, object]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ("git", *args),
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.rstrip("\r\n")

    try:
        head = run("rev-parse", "HEAD")
        branch = run("rev-parse", "--abbrev-ref", "HEAD")
        dirty = tuple(
            line[3:].replace("\\", "/")
            for line in run("status", "--short").splitlines()
            if len(line) >= 4
        )
    except (OSError, subprocess.CalledProcessError):
        return {"available": False}
    return {
        "available": True,
        "head": head,
        "branch": branch,
        "dirty_relative_paths": list(dirty),
    }


def _validate_source_provenance(
    provenance: Mapping[str, object],
    *,
    require_current_sources: bool,
) -> None:
    files = _mapping_sequence(provenance.get("files"), "source files")
    if {item.get("relative_path") for item in files} != set(SOURCE_PROVENANCE_PATHS):
        raise EvidenceError("Source provenance path set is incomplete.")
    for item in files:
        relative = str(item.get("relative_path", ""))
        digest = str(item.get("sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise EvidenceError("Source provenance contains an invalid digest.")
        if require_current_sources:
            current = hashlib.sha256((PROJECT_ROOT / relative).read_bytes()).hexdigest()
            if current != digest:
                raise EvidenceError(f"Evidence source is stale: {relative}.")


def _require_source_snapshot_unchanged(provenance: Mapping[str, object]) -> None:
    _validate_source_provenance(provenance, require_current_sources=True)


def _validate_privacy(document: Mapping[str, object]) -> None:
    rendered = json.dumps(document, allow_nan=False, sort_keys=True).casefold()
    forbidden = (
        r"[a-z]:\\",
        r"/users/",
        r"/home/",
        r"\.nd2(?:\"|\s|$)",
        r"gr535",
    )
    for pattern in forbidden:
        if re.search(pattern, rendered):
            raise EvidenceError(
                "Evidence contains a private path or source identifier."
            )


def _strict_json_text(document: Mapping[str, object]) -> str:
    return (
        json.dumps(
            document,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _atomic_write_text(output: Path | str, text: str) -> Path:
    path = Path(output).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path


def _mapping(value: Mapping[str, object], name: str) -> Mapping[str, object]:
    selected = value.get(name)
    if not isinstance(selected, Mapping):
        raise EvidenceError(f"Evidence field {name!r} must be a mapping.")
    return selected


def _mapping_sequence(value: object, name: str) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, Mapping) for item in value
    ):
        raise EvidenceError(f"Evidence field {name!r} must be a list of mappings.")
    return tuple(value)


if __name__ == "__main__":
    raise SystemExit(main())
