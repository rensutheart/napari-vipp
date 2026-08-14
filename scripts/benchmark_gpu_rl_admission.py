"""Create reproducible CuPy Richardson-Lucy admission evidence.

This development evidence command preserves the deterministic adversarial
matrices used to choose the initial ordinary Richardson-Lucy GPU region.  It
compares VIPP's authoritative progress-aware CPU operation with the real
resident CuPy provider; it is a backend-agreement command, not a performance
benchmark or a validation of restoration accuracy.

Importing the module, asking for help, or validating an existing artifact does
not import CuPy or initialize CUDA.  A full run writes evidence only after all
four bounded matrices complete successfully.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy import signal as scipy_signal

EVIDENCE_SCHEMA = "napari-vipp-cupy-rl-admission-evidence"
EVIDENCE_SCHEMA_VERSION = 2
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "docs" / "benchmarks" / "rl-cupy-admission-windows-rtx5090.json"
)

PARITY_POLICY_ID = "rl-scientific-equivalence-v2"
NRMSE_LIMIT = 0.005
MAX_ABSOLUTE_FLOOR = 1e-6
MAX_ABSOLUTE_PEAK_FACTOR = 0.005
LEGACY_V1_POLICY_ID = "rl-float32-tolerance-v1"
LEGACY_V1_NRMSE_LIMIT = 2e-6
LEGACY_V1_MAX_ABSOLUTE_FLOOR = 1e-6
LEGACY_V1_MAX_ABSOLUTE_PEAK_FACTOR = 5e-6
ADMITTED_FILTER_EPSILON_MINIMUM = 1e-12
ADMITTED_FILTER_EPSILON_MAXIMUM = 1e-6
ADMITTED_MAXIMUM_ITERATIONS = 100
DEFAULT_EPSILON_CHECKPOINTS = (10, 25, 26, 50, 100)

PUBLICATION_SCOPE: dict[str, bool] = {
    "development_branch_public_exposure_justified": True,
    "cross_platform_promotion_justified": False,
    "released_package_promotion_justified": False,
}

SOURCE_PROVENANCE_PATHS = (
    "scripts/benchmark_gpu_rl_admission.py",
    "src/napari_vipp/core/richardson_lucy.py",
    "src/napari_vipp/core/richardson_lucy_compute.py",
    "src/napari_vipp/core/richardson_lucy_parity.py",
    "src/napari_vipp/core/gpu/cupy_rl.py",
    "src/napari_vipp/core/progress.py",
)

# This object is intentionally JSON-native.  Its digest makes generator edits
# explicit and lets CPU-only CI verify that committed evidence is current.
BENCHMARK_CONTRACT: dict[str, object] = {
    "generator": "numpy-pcg64-rl-admission-v2",
    "authoritative_cpu": "progress-aware-vipp-richardson-lucy",
    "candidate": "rl-cupy-f32-v1",
    "input_dtype": "float32",
    "reviewed_admission_filter_epsilon_range": [
        ADMITTED_FILTER_EPSILON_MINIMUM,
        ADMITTED_FILTER_EPSILON_MAXIMUM,
    ],
    "reviewed_admission_maximum_iterations": ADMITTED_MAXIMUM_ITERATIONS,
    "parameter_region": {
        "normalize_psf": True,
        "clip_negative_input": True,
        "clip_output_negative": True,
        "preserve_input_scale": True,
    },
    "default_epsilon_checkpoints_164": {
        "fixture_source": "final_odd_164",
        "filter_epsilons": [ADMITTED_FILTER_EPSILON_MINIMUM],
        "iterations": list(DEFAULT_EPSILON_CHECKPOINTS),
        "purpose": (
            "admission checkpoints for the authored CPU default through the "
            "expanded 100-iteration boundary"
        ),
    },
    "legacy_branch_characterization_164": {
        "groups": {
            "gaussian_core_36": {
                "seeds": [100, 101, 102],
                "shape_psf_pairs": [
                    [[47, 53], [7, 7]],
                    [[128, 129], [13, 13]],
                    [[15, 17, 19], [3, 5, 5]],
                ],
                "fixtures_per_pair": [
                    "random_positive",
                    "sparse_beads",
                    "zero_heavy_dynamic_range",
                    "dark_field",
                ],
            },
            "odd_asymmetric_48": {
                "seeds": [3000, 3001, 3002, 3003],
                "shape_psf_pairs": [
                    [[63, 65], [5, 7]],
                    [[127, 131], [9, 11]],
                    [[17, 19, 21], [3, 5, 7]],
                ],
                "fixtures_per_pair": [
                    "positive_poisson",
                    "sparse_beads",
                    "zero_heavy_dynamic_range",
                    "step_and_impulse",
                ],
            },
            "sparse_seed_sweep_80": {
                "seeds": [5000, 5079],
                "seed_range_is_inclusive": True,
                "alternating_shape_psf_pairs": [
                    [[64, 65], [9, 9]],
                    [[128, 129], [13, 13]],
                ],
                "optional_impulse_noise_every_third_case": True,
            },
        },
        "filter_epsilons": [1e-8, 1e-7, 1e-6],
        "filter_epsilon_purpose": (
            "retained v1 near-identity characterization of threshold-branch "
            "sensitivity; these diagnostic failures do not fail v2"
        ),
        "iterations": [10, 25, 50],
    },
    "legacy_low_epsilon_characterization_36": {
        "fixture_group": "gaussian_core_36",
        "filter_epsilons": [1e-10],
        "iterations": [10, 25, 50, 100],
        "purpose": "retained v1 near-identity characterization only",
    },
    "legacy_even_psf_characterization_40": {
        "seeds": [900, 901],
        "shape_psf_pairs": [
            [[63, 65], [4, 6], "asymmetric"],
            [[127, 131], [7, 9], "asymmetric"],
            [[257, 259], [15, 15], "gaussian"],
            [[512, 513], [31, 31], "gaussian"],
            [[17, 19, 21], [4, 4, 6], "asymmetric"],
        ],
        "fixtures_per_pair": [
            "positive_poisson",
            "sparse_beads",
            "zero_heavy_dynamic_range",
            "step_and_impulse",
        ],
        "filter_epsilons": [1e-8, 1e-7, 1e-6],
        "iterations": [5, 10, 25],
        "purpose": (
            "diagnostic comparison only; even PSFs remain outside the "
            "prequalified workload region independently of tolerance"
        ),
    },
}


class AdmissionEvidenceError(RuntimeError):
    """A complete RL admission evidence document could not be produced."""


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
        default=None,
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
        default=None,
        help="Validate an existing JSON artifact without importing CuPy.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.validate_existing is not None:
        try:
            document = json.loads(args.validate_existing.read_text(encoding="utf-8"))
            validate_evidence_document(document, require_current_sources=True)
        except (AdmissionEvidenceError, OSError, TypeError, ValueError) as exc:
            print(f"RL admission evidence is invalid: {exc}", file=sys.stderr)
            return 2
        print(f"RL admission evidence is current: {args.validate_existing.resolve()}")
        return 0

    try:
        document = run_evidence(device_index=args.device_index)
        validate_evidence_document(document, require_current_sources=True)
        output = _atomic_write_text(
            args.output,
            _strict_json_text(document),
        )
        markdown_path = args.markdown or output.with_suffix(".md")
        markdown = _atomic_write_text(markdown_path, render_markdown(document))
    except (AdmissionEvidenceError, OSError, TypeError, ValueError) as exc:
        print(f"RL admission evidence run failed: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # optional provider failures need a concise boundary
        print(
            f"RL admission evidence run failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    print(f"Wrote RL admission evidence to {output}")
    print(f"Wrote readable RL admission summary to {markdown}")
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
    """Run all fixed matrices against the real CuPy provider."""

    if isinstance(device_index, bool) or not isinstance(device_index, int):
        raise TypeError("device_index must be an integer")
    source_provenance = _source_provenance()
    cupy = _load_cupy()
    if device_index < 0 or device_index >= int(cupy.cuda.runtime.getDeviceCount()):
        raise AdmissionEvidenceError(
            f"CUDA device index {device_index} is unavailable."
        )

    with cupy.cuda.Device(device_index):
        environment = _environment_fingerprint(cupy, device_index=device_index)
        gaussian_core = tuple(_gaussian_core_fixtures())
        odd_asymmetric = tuple(_odd_asymmetric_fixtures())
        sparse_sweep = tuple(_sparse_seed_sweep_fixtures())
        final_odd = gaussian_core + odd_asymmetric + sparse_sweep
        even_comparison = tuple(_even_psf_comparison_fixtures())

        _require_fixture_contract(
            final_odd=final_odd,
            gaussian_core=gaussian_core,
            odd_asymmetric=odd_asymmetric,
            sparse_sweep=sparse_sweep,
            even_comparison=even_comparison,
        )

        suites = {
            "default_epsilon_checkpoints_164": _run_suite(
                final_odd,
                filter_epsilons=(ADMITTED_FILTER_EPSILON_MINIMUM,),
                iterations=DEFAULT_EPSILON_CHECKPOINTS,
                cupy=cupy,
            ),
            "legacy_branch_characterization_164": _run_suite(
                final_odd,
                filter_epsilons=(1e-8, 1e-7, 1e-6),
                iterations=(10, 25, 50),
                cupy=cupy,
            ),
            "legacy_low_epsilon_characterization_36": _run_suite(
                gaussian_core,
                filter_epsilons=(1e-10,),
                iterations=(10, 25, 50, 100),
                cupy=cupy,
            ),
            "legacy_even_psf_characterization_40": _run_suite(
                even_comparison,
                filter_epsilons=(1e-8, 1e-7, 1e-6),
                iterations=(5, 10, 25),
                cupy=cupy,
            ),
        }
        cupy.cuda.get_current_stream().synchronize()
        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()

    _require_source_snapshot_unchanged(source_provenance)
    conclusion = _derive_conclusion(suites)
    return {
        "schema": EVIDENCE_SCHEMA,
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "complete",
        "contract": BENCHMARK_CONTRACT,
        "contract_sha256": benchmark_contract_digest(),
        "source_provenance": source_provenance,
        "environment": environment,
        "parity_gate": {
            "policy_id": PARITY_POLICY_ID,
            "scope": "CPU/GPU backend agreement only",
            "scientific_validity_claimed": False,
            "dtype": "float32",
            "shape_must_match": True,
            "finite_masks_must_match": True,
            "outputs_must_be_finite": True,
            "nonnegative_outputs_required": True,
            "nrmse_normalization": "l2(candidate - reference) / l2(reference)",
            "nrmse_limit": NRMSE_LIMIT,
            "max_abs_formula": "1e-6 + 0.005 * max(abs(cpu_reference))",
            "max_abs_floor": MAX_ABSOLUTE_FLOOR,
            "max_abs_peak_factor": MAX_ABSOLUTE_PEAK_FACTOR,
            "legacy_v1_diagnostic": {
                "policy_id": LEGACY_V1_POLICY_ID,
                "pass_fail": False,
                "nrmse_limit": LEGACY_V1_NRMSE_LIMIT,
                "max_abs_formula": "1e-6 + 5e-6 * max(abs(cpu_reference))",
            },
            "near_identity_nrmse_diagnostic": {
                "pass_fail": False,
                "nrmse_limit": LEGACY_V1_NRMSE_LIMIT,
            },
        },
        "suites": suites,
        "conclusion": conclusion,
        "limitations": [
            "This is single-host native-Windows RTX 5090 evidence.",
            "The sampled matrix cannot prove parity for every possible image or PSF.",
            "The 0.5% thresholds test backend agreement only; they do not establish "
            "restoration accuracy, image quality, resolution, or scientific validity.",
            "NRMSE has multiple normalization conventions; this artifact fixes and "
            "records the L2-reference convention used by VIPP.",
            "The checkpoint matrix does not exhaust every epsilon and iteration "
            "combination inside the reviewed envelope.",
            "Exact-workload CPU/GPU parity remains mandatory before optimizer "
            "selection.",
            "The authored CPU default filter_epsilon=1e-12 is unchanged and is "
            "covered at the declared iteration checkpoints.",
            "No parameter is silently changed to qualify a workload for GPU execution.",
            "This artifact supports public exposure only inside the checkpoint-backed "
            "reviewed envelope on this development branch.",
            "Cross-platform support and released-package promotion are not claimed.",
        ],
    }


def _load_cupy():
    try:
        import cupy
    except Exception as exc:
        raise AdmissionEvidenceError(
            "CuPy/CUDA is unavailable. No artifact was replaced; install the "
            "documented GPU development environment or use --validate-existing."
        ) from exc
    try:
        if int(cupy.cuda.runtime.getDeviceCount()) <= 0:
            raise AdmissionEvidenceError("CuPy found no CUDA devices.")
    except AdmissionEvidenceError:
        raise
    except Exception as exc:
        raise AdmissionEvidenceError(f"CUDA probe failed: {exc}") from exc
    return cupy


def _run_suite(
    fixtures: Sequence[_Fixture],
    *,
    filter_epsilons: Sequence[float],
    iterations: Sequence[int],
    cupy,
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for fixture in fixtures:
        for epsilon in filter_epsilons:
            for iteration_count in iterations:
                records.append(
                    _compare_fixture(
                        fixture,
                        filter_epsilon=float(epsilon),
                        iterations=int(iteration_count),
                        cupy=cupy,
                    )
                )
        cupy.get_default_memory_pool().free_all_blocks()

    summaries = _derive_suite_summaries(
        records,
        filter_epsilons=filter_epsilons,
        iterations=iterations,
    )
    return {
        "fixture_count": len(fixtures),
        "fixture_group_counts": dict(
            sorted(Counter(item.group_id for item in fixtures).items())
        ),
        "fixture_manifest_sha256": _fixture_manifest_digest(fixtures),
        "filter_epsilons": [float(value) for value in filter_epsilons],
        "iterations": [int(value) for value in iterations],
        "result_count": len(records),
        "summaries": summaries,
        "results": records,
    }


def _derive_suite_summaries(
    records: Sequence[Mapping[str, object]],
    *,
    filter_epsilons: Sequence[float],
    iterations: Sequence[int],
) -> list[dict[str, object]]:
    """Derive every aggregate from raw per-fixture measurements."""

    summaries: list[dict[str, object]] = []
    for epsilon in filter_epsilons:
        for iteration_count in iterations:
            selected = [
                record
                for record in records
                if record["filter_epsilon"] == float(epsilon)
                and record["iterations"] == int(iteration_count)
            ]
            if not selected:
                raise AdmissionEvidenceError(
                    "Evidence has no raw results for "
                    f"filter_epsilon={float(epsilon):g}, "
                    f"iterations={int(iteration_count)}."
                )
            worst = max(selected, key=lambda item: float(item["gate_score"]))
            worst_legacy_v1 = max(
                selected,
                key=lambda item: float(item["legacy_v1_gate_score"]),
            )
            worst_near_identity_nrmse = max(
                selected,
                key=lambda item: float(item["near_identity_nrmse"]),
            )
            worst_max_ulp = max(int(item["max_ulp"]) for item in selected)
            summaries.append(
                {
                    "filter_epsilon": float(epsilon),
                    "iterations": int(iteration_count),
                    "case_count": len(selected),
                    "failure_count": sum(
                        not bool(item["passed"]) for item in selected
                    ),
                    "worst_gate_score": float(worst["gate_score"]),
                    "worst_fixture_id": worst["fixture_id"],
                    "worst_nrmse": float(worst["nrmse"]),
                    "worst_max_abs": float(worst["max_abs"]),
                    "worst_cpu_peak": float(worst["cpu_peak"]),
                    "worst_max_abs_limit": float(worst["max_abs_limit"]),
                    "worst_max_ulp": worst_max_ulp,
                    "legacy_v1_failure_count": sum(
                        not bool(item["legacy_v1_gate_passed"]) for item in selected
                    ),
                    "worst_legacy_v1_gate_score": float(
                        worst_legacy_v1["legacy_v1_gate_score"]
                    ),
                    "worst_legacy_v1_fixture_id": worst_legacy_v1["fixture_id"],
                    "near_identity_nrmse_failure_count": sum(
                        not bool(item["near_identity_passed"]) for item in selected
                    ),
                    "worst_near_identity_nrmse": float(
                        worst_near_identity_nrmse["near_identity_nrmse"]
                    ),
                    "worst_near_identity_nrmse_fixture_id": (
                        worst_near_identity_nrmse["fixture_id"]
                    ),
                }
            )
    return summaries


def _compare_fixture(
    fixture: _Fixture,
    *,
    filter_epsilon: float,
    iterations: int,
    cupy,
) -> dict[str, object]:
    from napari_vipp.core.gpu.cupy_rl import (
        richardson_lucy_deconvolution as gpu_rl,
    )
    from napari_vipp.core.progress import ProgressContext
    from napari_vipp.core.richardson_lucy import (
        richardson_lucy_deconvolution as cpu_rl,
    )

    kwargs = {
        "spatial_mode": fixture.spatial_mode,
        "iterations": iterations,
        "normalize_psf": True,
        "clip_negative_input": True,
        "clip_output_negative": True,
        "preserve_input_scale": True,
        "filter_epsilon": filter_epsilon,
        "progress": ProgressContext(),
    }
    expected = cpu_rl([fixture.image, fixture.psf], **kwargs)
    candidate = gpu_rl(
        [cupy.asarray(fixture.image), cupy.asarray(fixture.psf)],
        **kwargs,
    )
    cupy.cuda.get_current_stream().synchronize()
    actual = cupy.asnumpy(candidate)
    del candidate
    return _parity_record(
        fixture,
        expected,
        actual,
        filter_epsilon=filter_epsilon,
        iterations=iterations,
    )


def _parity_record(
    fixture: _Fixture,
    expected: np.ndarray,
    actual: np.ndarray,
    *,
    filter_epsilon: float,
    iterations: int,
) -> dict[str, object]:
    shape_equal = actual.shape == expected.shape
    dtype_equal = actual.dtype == expected.dtype == np.dtype(np.float32)
    expected_finite = np.isfinite(expected)
    actual_finite = np.isfinite(actual)
    finite_masks_equal = bool(np.array_equal(expected_finite, actual_finite))
    completely_finite = bool(np.all(expected_finite) and np.all(actual_finite))
    cpu_nonnegative = bool(np.all(expected >= 0.0))
    gpu_nonnegative = bool(np.all(actual >= 0.0))
    if shape_equal:
        expected64 = expected.astype(np.float64)
        actual64 = actual.astype(np.float64)
        difference = actual64 - expected64
        cpu_peak = float(np.max(np.abs(expected64), initial=0.0))
        max_abs = float(np.max(np.abs(difference), initial=0.0))
        denominator = max(
            float(np.linalg.norm(expected64.ravel())),
            float(np.sqrt(expected64.size) * 1e-12),
        )
        nrmse = float(np.linalg.norm(difference.ravel()) / denominator)
    else:
        cpu_peak = max_abs = nrmse = float("inf")
    max_ulp = (
        _maximum_float32_ulp_distance(expected, actual)
        if shape_equal and dtype_equal
        else None
    )
    gate_fields = _derive_record_gate_fields(
        shape_equal=shape_equal,
        dtype_equal=dtype_equal,
        finite_masks_equal=finite_masks_equal,
        completely_finite=completely_finite,
        cpu_nonnegative=cpu_nonnegative,
        gpu_nonnegative=gpu_nonnegative,
        cpu_peak=cpu_peak,
        nrmse=nrmse,
        max_abs=max_abs,
    )
    return {
        "fixture_id": fixture.fixture_id,
        "group_id": fixture.group_id,
        "family": fixture.family,
        "image_shape": list(fixture.image.shape),
        "psf_shape": list(fixture.psf.shape),
        "filter_epsilon": filter_epsilon,
        "iterations": iterations,
        "shape_equal": shape_equal,
        "dtype_equal": dtype_equal,
        "finite_masks_equal": finite_masks_equal,
        "completely_finite": completely_finite,
        "cpu_nonnegative": cpu_nonnegative,
        "gpu_nonnegative": gpu_nonnegative,
        "cpu_peak": cpu_peak,
        "nrmse": nrmse,
        "max_abs": max_abs,
        "max_ulp": max_ulp,
        **gate_fields,
    }


def _derive_record_gate_fields(
    *,
    shape_equal: bool,
    dtype_equal: bool,
    finite_masks_equal: bool,
    completely_finite: bool,
    cpu_nonnegative: bool,
    gpu_nonnegative: bool,
    cpu_peak: float,
    nrmse: float,
    max_abs: float,
) -> dict[str, object]:
    """Recompute all v2 pass/fail and retained v1 diagnostic fields."""

    max_abs_limit = MAX_ABSOLUTE_FLOOR + MAX_ABSOLUTE_PEAK_FACTOR * cpu_peak
    gate_score = max(nrmse / NRMSE_LIMIT, max_abs / max_abs_limit)
    legacy_v1_max_abs_limit = (
        LEGACY_V1_MAX_ABSOLUTE_FLOOR + LEGACY_V1_MAX_ABSOLUTE_PEAK_FACTOR * cpu_peak
    )
    legacy_v1_gate_score = max(
        nrmse / LEGACY_V1_NRMSE_LIMIT,
        max_abs / legacy_v1_max_abs_limit,
    )
    structural_match = bool(
        shape_equal and dtype_equal and finite_masks_equal and completely_finite
    )
    near_identity_nrmse = nrmse
    return {
        "max_abs_limit": max_abs_limit,
        "gate_score": gate_score,
        "passed": bool(
            structural_match
            and cpu_nonnegative
            and gpu_nonnegative
            and nrmse <= NRMSE_LIMIT
            and max_abs <= max_abs_limit
        ),
        "legacy_v1_max_abs_limit": legacy_v1_max_abs_limit,
        "legacy_v1_gate_score": legacy_v1_gate_score,
        "legacy_v1_gate_passed": bool(
            structural_match
            and nrmse <= LEGACY_V1_NRMSE_LIMIT
            and max_abs <= legacy_v1_max_abs_limit
        ),
        "near_identity_nrmse": near_identity_nrmse,
        "near_identity_passed": bool(
            structural_match and near_identity_nrmse <= LEGACY_V1_NRMSE_LIMIT
        ),
    }


def _maximum_float32_ulp_distance(
    expected: np.ndarray,
    actual: np.ndarray,
) -> int:
    """Return a diagnostic float32 ULP distance without affecting admission."""

    if not expected.size:
        return 0

    def ordered(values: np.ndarray) -> np.ndarray:
        bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
        negative = (bits & np.uint32(0x80000000)) != 0
        return np.where(
            negative,
            np.uint64(0xFFFFFFFF) - bits.astype(np.uint64),
            np.uint64(0x80000000) + bits.astype(np.uint64),
        )

    expected_ordered = ordered(expected)
    actual_ordered = ordered(actual)
    distance = np.maximum(expected_ordered, actual_ordered) - np.minimum(
        expected_ordered,
        actual_ordered,
    )
    return int(np.max(distance))


def _gaussian_core_fixtures() -> Iterable[_Fixture]:
    group = "gaussian_core_36"
    pairs = (
        ((47, 53), (7, 7)),
        ((128, 129), (13, 13)),
        ((15, 17, 19), (3, 5, 5)),
    )
    for seed in range(100, 103):
        rng = np.random.default_rng(seed)
        for image_shape, psf_shape in pairs:
            psf = _gaussian_psf(psf_shape)
            prefix = f"core-s{seed}-{'x'.join(map(str, image_shape))}"

            image = rng.random(image_shape, dtype=np.float32)
            yield _fixture(prefix, group, "random_positive", image, psf)

            latent = np.zeros(image_shape, dtype=np.float32)
            for _ in range(12):
                index = tuple(rng.integers(2, size - 2) for size in image_shape)
                latent[index] = np.float32(10.0 ** rng.uniform(-5.0, 0.0))
            image = scipy_signal.convolve(latent, psf, mode="same").astype(np.float32)
            image /= np.float32(max(float(image.max()), 1e-30))
            yield _fixture(prefix, group, "sparse_beads", image, psf)

            image = np.power(10.0, rng.uniform(-12.0, 0.0, image_shape)).astype(
                np.float32
            )
            image[rng.random(image_shape) < 0.55] = np.float32(0.0)
            yield _fixture(prefix, group, "zero_heavy_dynamic_range", image, psf)

            image = np.zeros(image_shape, dtype=np.float32)
            region = tuple(slice(size // 3, 2 * size // 3) for size in image_shape)
            image[region] = rng.random(
                image[region].shape, dtype=np.float32
            ) * np.float32(1e-5)
            image[tuple(size // 2 for size in image_shape)] = np.float32(1.0)
            yield _fixture(prefix, group, "dark_field", image, psf)


def _odd_asymmetric_fixtures() -> Iterable[_Fixture]:
    group = "odd_asymmetric_48"
    pairs = (
        ((63, 65), (5, 7)),
        ((127, 131), (9, 11)),
        ((17, 19, 21), (3, 5, 7)),
    )
    for seed in range(3000, 3004):
        rng = np.random.default_rng(seed)
        for image_shape, psf_shape in pairs:
            psf = _asymmetric_psf(psf_shape, rng)
            prefix = f"odd-asym-s{seed}-{'x'.join(map(str, image_shape))}"

            image = rng.poisson(25, size=image_shape).astype(np.float32)
            image /= np.float32(image.max())
            yield _fixture(prefix, group, "positive_poisson", image, psf)

            latent = np.zeros(image_shape, dtype=np.float32)
            for _ in range(20):
                index = tuple(rng.integers(0, size) for size in image_shape)
                latent[index] = np.float32(10.0 ** rng.uniform(-4.0, 0.0))
            image = scipy_signal.convolve(latent, psf, mode="same").astype(np.float32)
            image = np.maximum(image, np.float32(0.0))
            image /= np.float32(max(float(image.max()), 1e-30))
            yield _fixture(prefix, group, "sparse_beads", image, psf)

            image = np.power(10.0, rng.uniform(-12.0, 0.0, image_shape)).astype(
                np.float32
            )
            image[rng.random(image_shape) < 0.6] = np.float32(0.0)
            yield _fixture(prefix, group, "zero_heavy_dynamic_range", image, psf)

            image = np.zeros(image_shape, dtype=np.float32)
            region = tuple(slice(size // 4, 3 * size // 4) for size in image_shape)
            image[region] = np.float32(0.01)
            image[tuple(size // 2 for size in image_shape)] = np.float32(1.0)
            yield _fixture(prefix, group, "step_and_impulse", image, psf)


def _sparse_seed_sweep_fixtures() -> Iterable[_Fixture]:
    group = "sparse_seed_sweep_80"
    for offset, seed in enumerate(range(5000, 5080)):
        rng = np.random.default_rng(seed)
        side = 64 if offset % 2 == 0 else 128
        image_shape = (side, side + 1)
        psf_size = 9 if side == 64 else 13
        psf = _gaussian_psf((psf_size, psf_size))
        latent = np.zeros(image_shape, dtype=np.float32)
        for _ in range(int(rng.integers(3, 30))):
            index = tuple(rng.integers(0, size) for size in image_shape)
            latent[index] = np.float32(10.0 ** rng.uniform(-7.0, 0.0))
        image = scipy_signal.convolve(latent, psf, mode="same").astype(np.float32)
        if offset % 3 == 0:
            impulse_mask = rng.random(image_shape) < 0.003
            image += impulse_mask.astype(np.float32) * np.float32(
                10.0 ** rng.uniform(-10.0, -4.0)
            )
        image /= np.float32(max(float(image.max()), 1e-30))
        prefix = f"sparse-sweep-s{seed}-{'x'.join(map(str, image_shape))}"
        yield _fixture(prefix, group, "sparse_noisy_beads", image, psf)


def _even_psf_comparison_fixtures() -> Iterable[_Fixture]:
    group = "even_psf_comparison_40"
    pairs = (
        ((63, 65), (4, 6), "asymmetric"),
        ((127, 131), (7, 9), "asymmetric"),
        ((257, 259), (15, 15), "gaussian"),
        ((512, 513), (31, 31), "gaussian"),
        ((17, 19, 21), (4, 4, 6), "asymmetric"),
    )
    for seed in range(900, 902):
        rng = np.random.default_rng(seed)
        for image_shape, psf_shape, psf_family in pairs:
            psf = (
                _gaussian_psf(psf_shape)
                if psf_family == "gaussian"
                else _asymmetric_psf(psf_shape, rng)
            )
            prefix = f"even-study-s{seed}-{'x'.join(map(str, image_shape))}"

            image = rng.poisson(25, size=image_shape).astype(np.float32)
            image += rng.random(image_shape, dtype=np.float32)
            image /= np.float32(image.max())
            yield _fixture(prefix, group, "positive_poisson", image, psf)

            latent = np.zeros(image_shape, dtype=np.float32)
            for _ in range(20):
                index = tuple(rng.integers(0, size) for size in image_shape)
                latent[index] = np.float32(10.0 ** rng.uniform(-4.0, 0.0))
            image = scipy_signal.convolve(latent, psf, mode="same").astype(np.float32)
            image = np.maximum(image, np.float32(0.0))
            image /= np.float32(max(float(image.max()), 1e-30))
            yield _fixture(prefix, group, "sparse_beads", image, psf)

            image = np.power(10.0, rng.uniform(-12.0, 0.0, image_shape)).astype(
                np.float32
            )
            image[rng.random(image_shape) < 0.6] = np.float32(0.0)
            yield _fixture(prefix, group, "zero_heavy_dynamic_range", image, psf)

            image = np.zeros(image_shape, dtype=np.float32)
            region = tuple(slice(size // 4, 3 * size // 4) for size in image_shape)
            image[region] = np.float32(0.01)
            image[tuple(size // 2 for size in image_shape)] = np.float32(1.0)
            yield _fixture(prefix, group, "step_and_impulse", image, psf)


def _fixture(
    prefix: str,
    group: str,
    family: str,
    image: np.ndarray,
    psf: np.ndarray,
) -> _Fixture:
    image = np.ascontiguousarray(image, dtype=np.float32)
    psf = np.ascontiguousarray(psf, dtype=np.float32)
    image.setflags(write=False)
    psf.setflags(write=False)
    return _Fixture(f"{prefix}-{family}", group, family, image, psf)


def _gaussian_psf(shape: Sequence[int]) -> np.ndarray:
    grids = np.meshgrid(
        *(np.arange(size) - (size - 1) / 2 for size in shape),
        indexing="ij",
    )
    exponent = sum(
        (grid / max(size / 5.0, 0.7)) ** 2
        for grid, size in zip(grids, shape, strict=True)
    )
    values = np.exp(-0.5 * exponent).astype(np.float32)
    values /= np.float32(values.sum(dtype=np.float64))
    return np.ascontiguousarray(values)


def _asymmetric_psf(shape: Sequence[int], rng: np.random.Generator) -> np.ndarray:
    values = rng.power(3.0, size=tuple(shape)).astype(np.float32)
    values /= np.float32(values.sum(dtype=np.float64))
    return np.ascontiguousarray(values)


def _require_fixture_contract(
    *,
    final_odd: Sequence[_Fixture],
    gaussian_core: Sequence[_Fixture],
    odd_asymmetric: Sequence[_Fixture],
    sparse_sweep: Sequence[_Fixture],
    even_comparison: Sequence[_Fixture],
) -> None:
    expected = {
        "final_odd": (len(final_odd), 164),
        "gaussian_core": (len(gaussian_core), 36),
        "odd_asymmetric": (len(odd_asymmetric), 48),
        "sparse_sweep": (len(sparse_sweep), 80),
        "even_comparison": (len(even_comparison), 40),
    }
    mismatches = [
        f"{name}={actual}, expected {wanted}"
        for name, (actual, wanted) in expected.items()
        if actual != wanted
    ]
    if mismatches:
        raise AdmissionEvidenceError(
            "Fixture contract mismatch: " + "; ".join(mismatches)
        )
    if any(any(size % 2 == 0 for size in item.psf.shape) for item in final_odd):
        raise AdmissionEvidenceError("The final 164-fixture matrix must use odd PSFs.")
    even_count = sum(
        any(size % 2 == 0 for size in item.psf.shape) for item in even_comparison
    )
    if even_count != 16:
        raise AdmissionEvidenceError(
            f"Even-PSF comparison expected 16 even fixtures, got {even_count}."
        )
    identifiers = [item.fixture_id for item in (*final_odd, *even_comparison)]
    if len(identifiers) != len(set(identifiers)):
        raise AdmissionEvidenceError("Fixture identifiers must be unique.")


def _fixture_manifest_digest(fixtures: Sequence[_Fixture]) -> str:
    digest = hashlib.sha256()
    for fixture in fixtures:
        digest.update(fixture.fixture_id.encode("utf-8"))
        for array in (fixture.image, fixture.psf):
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(repr(array.shape).encode("ascii"))
            digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    return digest.hexdigest()


def _derive_conclusion(suites: Mapping[str, object]) -> dict[str, object]:
    checkpoints = _summary_mappings(
        _suite_mapping(suites, "default_epsilon_checkpoints_164")
    )
    observed_checkpoints = {
        int(item["iterations"])
        for item in checkpoints
        if float(item["filter_epsilon"]) == ADMITTED_FILTER_EPSILON_MINIMUM
    }
    checkpoint_matrix_complete = observed_checkpoints == set(
        DEFAULT_EPSILON_CHECKPOINTS
    )
    checkpoint_matrix_passed = checkpoint_matrix_complete and all(
        int(item["failure_count"]) == 0 for item in checkpoints
    )
    legacy_odd_summaries = (
        *_summary_mappings(
            _suite_mapping(suites, "legacy_branch_characterization_164")
        ),
        *_summary_mappings(
            _suite_mapping(suites, "legacy_low_epsilon_characterization_36")
        ),
    )
    legacy_even_summaries = _summary_mappings(
        _suite_mapping(suites, "legacy_even_psf_characterization_40")
    )
    sampled_odd_conditions_passed = checkpoint_matrix_passed and all(
        int(item["failure_count"]) == 0 for item in legacy_odd_summaries
    )
    legacy_summaries = (
        *legacy_odd_summaries,
        *legacy_even_summaries,
    )
    legacy_v1_sensitivity_observed = any(
        int(item["legacy_v1_failure_count"]) > 0 for item in legacy_summaries
    )
    if not sampled_odd_conditions_passed:
        raise AdmissionEvidenceError(
            "Completed measurements do not support the v2 RL checkpoint-backed "
            "odd-PSF envelope."
        )
    return {
        "admitted_filter_epsilon_minimum": ADMITTED_FILTER_EPSILON_MINIMUM,
        "admitted_filter_epsilon_maximum": ADMITTED_FILTER_EPSILON_MAXIMUM,
        "admitted_maximum_iterations": ADMITTED_MAXIMUM_ITERATIONS,
        "default_epsilon_checkpoints": list(DEFAULT_EPSILON_CHECKPOINTS),
        "require_all_psf_extents_odd": True,
        "default_epsilon_checkpoints_passed_sampled_matrix": (checkpoint_matrix_passed),
        "all_sampled_odd_psf_conditions_passed_v2": sampled_odd_conditions_passed,
        "filter_epsilon_continuum_exhaustively_sampled": False,
        "legacy_v1_policy_id": LEGACY_V1_POLICY_ID,
        "legacy_v1_is_diagnostic_only": True,
        "legacy_v1_branch_sensitivity_observed": legacy_v1_sensitivity_observed,
        "agreement_scope": "CPU/GPU backend agreement only",
        "scientific_validity_claimed": False,
        **PUBLICATION_SCOPE,
        "selection_requirement": (
            "exact-workload parity before timing or optimizer selection"
        ),
        "authored_parameters_rewritten": False,
    }


def validate_evidence_document(
    document: Mapping[str, object],
    *,
    require_current_sources: bool,
) -> None:
    if not isinstance(document, Mapping):
        raise AdmissionEvidenceError("Evidence document must be a mapping.")
    if document.get("schema") != EVIDENCE_SCHEMA:
        raise AdmissionEvidenceError("Evidence schema does not match this generator.")
    if document.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        raise AdmissionEvidenceError("Evidence schema version is unsupported.")
    if document.get("status") != "complete":
        raise AdmissionEvidenceError("Evidence artifact is not complete.")
    if document.get("contract_sha256") != benchmark_contract_digest():
        raise AdmissionEvidenceError("Evidence generator contract is stale.")
    if document.get("contract") != BENCHMARK_CONTRACT:
        raise AdmissionEvidenceError(
            "Embedded fixture contract differs from the generator."
        )
    suites = document.get("suites")
    if not isinstance(suites, Mapping):
        raise AdmissionEvidenceError("Evidence suites are missing.")
    expected_suites = {
        "default_epsilon_checkpoints_164": (
            164,
            820,
            (ADMITTED_FILTER_EPSILON_MINIMUM,),
            DEFAULT_EPSILON_CHECKPOINTS,
        ),
        "legacy_branch_characterization_164": (
            164,
            1476,
            (1e-8, 1e-7, 1e-6),
            (10, 25, 50),
        ),
        "legacy_low_epsilon_characterization_36": (
            36,
            144,
            (1e-10,),
            (10, 25, 50, 100),
        ),
        "legacy_even_psf_characterization_40": (
            40,
            360,
            (1e-8, 1e-7, 1e-6),
            (5, 10, 25),
        ),
    }
    canonical_suites: dict[str, object] = {}
    for name, (
        expected_count,
        expected_result_count,
        expected_epsilons,
        expected_iterations,
    ) in expected_suites.items():
        suite = _suite_mapping(suites, name)
        if suite.get("fixture_count") != expected_count:
            raise AdmissionEvidenceError(f"Suite {name!r} has a stale fixture count.")
        if suite.get("filter_epsilons") != list(expected_epsilons):
            raise AdmissionEvidenceError(
                f"Suite {name!r} has a stale filter-epsilon matrix."
            )
        if suite.get("iterations") != list(expected_iterations):
            raise AdmissionEvidenceError(
                f"Suite {name!r} has a stale iteration matrix."
            )
        results = suite.get("results")
        if (
            not isinstance(results, list)
            or len(results) != suite.get("result_count")
            or len(results) != expected_result_count
        ):
            raise AdmissionEvidenceError(f"Suite {name!r} has incomplete raw results.")
        required_result_fields = {
            "fixture_id",
            "filter_epsilon",
            "iterations",
            "shape_equal",
            "dtype_equal",
            "finite_masks_equal",
            "completely_finite",
            "cpu_nonnegative",
            "gpu_nonnegative",
            "cpu_peak",
            "nrmse",
            "max_abs",
            "max_abs_limit",
            "gate_score",
            "passed",
            "max_ulp",
            "legacy_v1_max_abs_limit",
            "legacy_v1_gate_score",
            "legacy_v1_gate_passed",
            "near_identity_nrmse",
            "near_identity_passed",
        }
        if any(
            not isinstance(item, Mapping) or not required_result_fields.issubset(item)
            for item in results
        ):
            raise AdmissionEvidenceError(
                f"Suite {name!r} is missing v2 result diagnostics."
            )
        fixture_ids_by_condition: dict[tuple[float, int], set[str]] = {
            (float(epsilon), int(iteration_count)): set()
            for epsilon in expected_epsilons
            for iteration_count in expected_iterations
        }
        for index, result in enumerate(results):
            _validate_result_gate_fields(result, suite_name=name, result_index=index)
            epsilon = _finite_float(
                result.get("filter_epsilon"),
                context=f"Suite {name!r} result {index} filter_epsilon",
            )
            iteration_count = result.get("iterations")
            if isinstance(iteration_count, bool) or not isinstance(
                iteration_count, int
            ):
                raise AdmissionEvidenceError(
                    f"Suite {name!r} result {index} has invalid iterations."
                )
            condition = (epsilon, iteration_count)
            if condition not in fixture_ids_by_condition:
                raise AdmissionEvidenceError(
                    f"Suite {name!r} result {index} is outside its declared matrix."
                )
            fixture_id = result.get("fixture_id")
            if not isinstance(fixture_id, str) or not fixture_id:
                raise AdmissionEvidenceError(
                    f"Suite {name!r} result {index} has an invalid fixture ID."
                )
            identifiers = fixture_ids_by_condition[condition]
            if fixture_id in identifiers:
                raise AdmissionEvidenceError(
                    f"Suite {name!r} repeats fixture {fixture_id!r} for one condition."
                )
            identifiers.add(fixture_id)
        fixture_sets = tuple(fixture_ids_by_condition.values())
        if any(len(items) != expected_count for items in fixture_sets) or any(
            items != fixture_sets[0] for items in fixture_sets[1:]
        ):
            raise AdmissionEvidenceError(
                f"Suite {name!r} raw results do not cover one stable fixture set."
            )
        derived_summaries = _derive_suite_summaries(
            results,
            filter_epsilons=expected_epsilons,
            iterations=expected_iterations,
        )
        _require_summaries_match_raw_results(
            suite,
            derived_summaries,
            suite_name=name,
        )
        canonical_suites[name] = {**suite, "summaries": derived_summaries}
    parity_gate = document.get("parity_gate")
    if not isinstance(parity_gate, Mapping):
        raise AdmissionEvidenceError("Evidence parity gate is missing.")
    required_gate = {
        "policy_id": PARITY_POLICY_ID,
        "scope": "CPU/GPU backend agreement only",
        "scientific_validity_claimed": False,
        "nonnegative_outputs_required": True,
        "nrmse_limit": NRMSE_LIMIT,
        "max_abs_floor": MAX_ABSOLUTE_FLOOR,
        "max_abs_peak_factor": MAX_ABSOLUTE_PEAK_FACTOR,
    }
    for key, expected in required_gate.items():
        if parity_gate.get(key) != expected:
            raise AdmissionEvidenceError(f"Evidence parity gate {key!r} is stale.")
    conclusion = document.get("conclusion")
    if not isinstance(conclusion, Mapping):
        raise AdmissionEvidenceError("Evidence conclusion is missing.")
    expected_conclusion = _derive_conclusion(canonical_suites)
    for key, expected in expected_conclusion.items():
        if conclusion.get(key) != expected:
            raise AdmissionEvidenceError(f"Evidence conclusion {key!r} is stale.")
    for key, expected in PUBLICATION_SCOPE.items():
        if conclusion.get(key) is not expected:
            raise AdmissionEvidenceError(f"Evidence conclusion {key!r} is stale.")
    if require_current_sources:
        provenance = document.get("source_provenance")
        if not isinstance(provenance, Mapping):
            raise AdmissionEvidenceError("Source provenance is missing.")
        _require_source_snapshot_unchanged(provenance)


def _validate_result_gate_fields(
    result: Mapping[str, object],
    *,
    suite_name: str,
    result_index: int,
) -> None:
    context = f"Suite {suite_name!r} result {result_index}"
    booleans: dict[str, bool] = {}
    for key in (
        "shape_equal",
        "dtype_equal",
        "finite_masks_equal",
        "completely_finite",
        "cpu_nonnegative",
        "gpu_nonnegative",
    ):
        value = result.get(key)
        if type(value) is not bool:
            raise AdmissionEvidenceError(f"{context} has an invalid {key!r} flag.")
        booleans[key] = value
    cpu_peak = _nonnegative_finite_float(result.get("cpu_peak"), context=context)
    nrmse = _nonnegative_finite_float(result.get("nrmse"), context=context)
    max_abs = _nonnegative_finite_float(result.get("max_abs"), context=context)
    max_ulp = result.get("max_ulp")
    if isinstance(max_ulp, bool) or not isinstance(max_ulp, int) or max_ulp < 0:
        raise AdmissionEvidenceError(f"{context} has an invalid 'max_ulp' diagnostic.")
    expected = _derive_record_gate_fields(
        **booleans,
        cpu_peak=cpu_peak,
        nrmse=nrmse,
        max_abs=max_abs,
    )
    for key, expected_value in expected.items():
        actual = result.get(key)
        if isinstance(expected_value, bool):
            matches = type(actual) is bool and actual is expected_value
        else:
            matches = _numbers_match(actual, expected_value)
        if not matches:
            raise AdmissionEvidenceError(
                f"{context} has a stale or inconsistent {key!r} value."
            )


def _require_summaries_match_raw_results(
    suite: Mapping[str, object],
    expected_summaries: Sequence[Mapping[str, object]],
    *,
    suite_name: str,
) -> None:
    actual_summaries = _summary_mappings(suite)
    if len(actual_summaries) != len(expected_summaries):
        raise AdmissionEvidenceError(
            f"Suite {suite_name!r} summary count does not match raw results."
        )
    for index, (actual, expected) in enumerate(
        zip(actual_summaries, expected_summaries, strict=True)
    ):
        if set(actual) != set(expected):
            raise AdmissionEvidenceError(
                f"Suite {suite_name!r} summary {index} fields are stale."
            )
        for key, expected_value in expected.items():
            actual_value = actual.get(key)
            if isinstance(expected_value, float):
                matches = _numbers_match(actual_value, expected_value)
            else:
                matches = actual_value == expected_value
            if not matches:
                raise AdmissionEvidenceError(
                    f"Suite {suite_name!r} summary {index} field {key!r} "
                    "does not match raw results."
                )


def _finite_float(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AdmissionEvidenceError(f"{context} must be a finite number.")
    converted = float(value)
    if not math.isfinite(converted):
        raise AdmissionEvidenceError(f"{context} must be a finite number.")
    return converted


def _nonnegative_finite_float(value: object, *, context: str) -> float:
    converted = _finite_float(value, context=context)
    if converted < 0.0:
        raise AdmissionEvidenceError(f"{context} has a negative metric.")
    return converted


def _numbers_match(actual: object, expected: float) -> bool:
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        return False
    actual_float = float(actual)
    return math.isfinite(actual_float) and math.isclose(
        actual_float,
        float(expected),
        rel_tol=1e-12,
        abs_tol=1e-15,
    )


def render_markdown(document: Mapping[str, object]) -> str:
    suites = document["suites"]
    environment = document["environment"]
    conclusion = document["conclusion"]
    lines = [
        "# CuPy Richardson-Lucy admission evidence",
        "",
        f"- Generated: `{document['generated_at_utc']}`",
        f"- Schema: `{document['schema']}` version {document['schema_version']}",
        f"- Device: `{environment['cuda']['device_name']}`",
        f"- Platform: `{environment['platform']}`",
        "",
        "This deterministic, machine-local evidence tests CPU/GPU backend agreement",
        "between VIPP's CPU reference and CuPy backend. Passing does not establish",
        "the restored image, PSF, iteration count, or recovered structures are",
        "scientifically valid. It is not a portable performance, cross-platform,",
        "or released-package promotion claim, and it does not waive exact-workload",
        "parity.",
        "",
        "## Policy gate",
        "",
        f"- NRMSE (L2/reference-L2): `<= {NRMSE_LIMIT:g}` (0.5%).",
        "- Maximum absolute error: `<= 1e-6 + 0.005 × CPU peak`.",
        "- Shape, float32 dtype, finite masks, complete finiteness, and",
        "  nonnegative clipped outputs must match.",
        f"- The former `{LEGACY_V1_POLICY_ID}` thresholds are retained as",
        "  diagnostics only and cannot independently pass or fail v2.",
        "- NRMSE-only `near_identity` (`<= 2e-6`) and maximum ULP are also",
        "  diagnostic only.",
        "",
    ]
    for suite_name in (
        "default_epsilon_checkpoints_164",
        "legacy_branch_characterization_164",
        "legacy_low_epsilon_characterization_36",
        "legacy_even_psf_characterization_40",
    ):
        suite = suites[suite_name]
        lines.extend(
            [
                f"## {suite_name.replace('_', ' ')}",
                "",
                f"- Fixtures: **{suite['fixture_count']}**",
                f"- Manifest SHA-256: `{suite['fixture_manifest_sha256']}`",
                "",
                "| Filter epsilon | Iterations | v2 failures | v2 worst score | "
                "v1 diagnostic failures | v1 worst score |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for summary in suite["summaries"]:
            lines.append(
                "| "
                f"{summary['filter_epsilon']:.0e} | {summary['iterations']} | "
                f"{summary['failure_count']}/{summary['case_count']} | "
                f"{summary['worst_gate_score']:.9g} | "
                f"{summary['legacy_v1_failure_count']}/"
                f"{summary['case_count']} | "
                f"{summary['worst_legacy_v1_gate_score']:.9g} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Checkpoint-backed v2 envelope",
            "",
            "- finite authored `filter_epsilon` from "
            f"`{conclusion['admitted_filter_epsilon_minimum']:.0e}` through "
            f"`{conclusion['admitted_filter_epsilon_maximum']:.0e}`;",
            f"- `iterations <= {conclusion['admitted_maximum_iterations']}`;",
            "- default-epsilon checkpoints at `10, 25, 26, 50, 100`;",
            "- every PSF extent is odd;",
            "- normalized PSF and default-safe clipping/scale controls;",
            "- public exposure on this development branch, limited to this",
            "  checkpoint-backed reviewed envelope; and",
            "- exact-workload CPU/GPU parity before timing or optimizer selection.",
            "",
            "These checkpoints support the reviewed envelope but do not exhaust",
            "every epsilon and iteration combination inside it.",
            "The older narrow gate remains useful for seeing small numerical",
            "differences and threshold-branch sensitivity, but those diagnostics",
            "do not define v2 admission. The authored `filter_epsilon` and iteration",
            "count are never changed to qualify a GPU run.",
            "The 0.5% limits are engineering non-inferiority margins for backend",
            "agreement, not image-quality or scientific-accuracy thresholds.",
            "Cross-platform support and released-package promotion require their",
            "own validation and are not claimed by this artifact.",
            "",
            "Raw per-fixture metrics, environment versions, generator contract, and",
            "source hashes are retained in the sibling JSON artifact.",
            "",
        ]
    )
    return "\n".join(lines)


def _suite_mapping(suites: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = suites.get(name)
    if not isinstance(value, Mapping):
        raise AdmissionEvidenceError(f"Evidence suite {name!r} is missing.")
    return value


def _summary_mappings(suite: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    values = suite.get("summaries")
    if not isinstance(values, list) or not values:
        raise AdmissionEvidenceError("Evidence suite summaries are missing.")
    if not all(isinstance(value, Mapping) for value in values):
        raise AdmissionEvidenceError("Evidence suite summary is malformed.")
    return tuple(values)


def _environment_fingerprint(cupy, *, device_index: int) -> dict[str, object]:
    properties = cupy.cuda.runtime.getDeviceProperties(device_index)
    device_name = properties.get("name", "unknown")
    if isinstance(device_name, bytes):
        device_name = device_name.decode("utf-8", errors="replace")
    packages = {}
    for name in (
        "napari-vipp",
        "numpy",
        "scipy",
        "scikit-image",
        "cupy-cuda13x",
        "cupy-cuda12x",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "packages": packages,
        "cuda": {
            "device_index": device_index,
            "device_name": str(device_name),
            "compute_capability": [
                int(properties.get("major", -1)),
                int(properties.get("minor", -1)),
            ],
            "total_global_memory_bytes": int(properties.get("totalGlobalMem", 0)),
            "driver_version": int(cupy.cuda.runtime.driverGetVersion()),
            "runtime_version": int(cupy.cuda.runtime.runtimeGetVersion()),
            "cupy_version": str(cupy.__version__),
        },
    }


def _source_provenance(
    *,
    project_root: Path = PROJECT_ROOT,
    source_paths: Sequence[str] = SOURCE_PROVENANCE_PATHS,
) -> dict[str, object]:
    root = Path(project_root).resolve(strict=False)
    hashes = {}
    for relative in source_paths:
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise AdmissionEvidenceError(f"Invalid source path {relative!r}.")
        path = root / candidate
        if not path.is_file():
            raise AdmissionEvidenceError(f"Required source {relative!r} is missing.")
        hashes[str(relative).replace("\\", "/")] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    return {
        "hash_algorithm": "sha256",
        "files": hashes,
        "git": _git_provenance(root),
    }


def _git_provenance(root: Path) -> dict[str, object]:
    try:
        head = subprocess.run(
            ("git", "-C", str(root), "rev-parse", "--verify", "HEAD"),
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        status = subprocess.run(
            (
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ),
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    if head.returncode != 0 or status.returncode != 0:
        return {"available": False, "reason": (head.stderr or status.stderr).strip()}
    status_lines = [line for line in status.stdout.splitlines() if line]
    return {
        "available": True,
        "head": head.stdout.strip(),
        "worktree_dirty": bool(status_lines),
        "status_porcelain_v1": status_lines,
    }


def _require_source_snapshot_unchanged(provenance: Mapping[str, object]) -> None:
    expected = provenance.get("files")
    if not isinstance(expected, Mapping):
        raise AdmissionEvidenceError("Source provenance has no file hashes.")
    current = _source_provenance(source_paths=tuple(str(path) for path in expected))[
        "files"
    ]
    if current != expected:
        raise AdmissionEvidenceError(
            "Relevant RL source changed while evidence was being collected."
        )


def _strict_json_text(document: Mapping[str, object]) -> str:
    try:
        return (
            json.dumps(
                document,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    except (TypeError, ValueError) as exc:
        raise AdmissionEvidenceError(f"Evidence is not strict JSON: {exc}") from exc


def _atomic_write_text(output: Path | str, text: str) -> Path:
    requested = Path(output).expanduser()
    if requested.is_symlink():
        raise AdmissionEvidenceError("Output must not be a symbolic link.")
    path = requested.resolve(strict=False)
    if path.exists() and path.is_dir():
        raise AdmissionEvidenceError("Output path refers to a directory.")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return path


if __name__ == "__main__":
    raise SystemExit(main())
