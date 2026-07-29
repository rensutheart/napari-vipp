"""Create reproducible CuPy Richardson-Lucy admission evidence.

This developer-only command preserves the deterministic adversarial matrices
used to choose the initial ordinary Richardson-Lucy GPU region.  It compares
VIPP's authoritative progress-aware CPU operation with the real resident CuPy
provider; it is a scientific-parity command, not a performance benchmark.

Importing the module, asking for help, or validating an existing artifact does
not import CuPy or initialize CUDA.  A full run writes evidence only after all
three matrices complete successfully.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
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
EVIDENCE_SCHEMA_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "docs" / "benchmarks" / "rl-cupy-admission-windows-rtx5090.json"
)

PARITY_POLICY_ID = "rl-float32-tolerance-v1"
NRMSE_LIMIT = 2e-6
MAX_ABSOLUTE_FLOOR = 1e-6
MAX_ABSOLUTE_PEAK_FACTOR = 5e-6
RECOMMENDED_FILTER_EPSILON = 1e-8
RECOMMENDED_MAXIMUM_ITERATIONS = 25

SOURCE_PROVENANCE_PATHS = (
    "scripts/benchmark_gpu_rl_admission.py",
    "src/napari_vipp/core/operations.py",
    "src/napari_vipp/core/gpu/cupy_rl.py",
    "src/napari_vipp/core/compute_benchmark_adapter.py",
    "src/napari_vipp/core/compute_policy.py",
    "src/napari_vipp/core/compute_specs.py",
    "src/napari_vipp/core/progress.py",
)

# This object is intentionally JSON-native.  Its digest makes generator edits
# explicit and lets CPU-only CI verify that committed evidence is current.
BENCHMARK_CONTRACT: dict[str, object] = {
    "generator": "numpy-pcg64-rl-admission-v1",
    "authoritative_cpu": "progress-aware-vipp-richardson-lucy",
    "candidate": "rl-cupy-f32-v1",
    "input_dtype": "float32",
    "reviewed_admission_filter_epsilon": 1e-8,
    "reviewed_admission_maximum_iterations": 25,
    "parameter_region": {
        "normalize_psf": True,
        "clip_negative_input": True,
        "clip_output_negative": True,
        "preserve_input_scale": True,
    },
    "final_odd_164": {
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
            "1e-8 is the reviewed admission value; 1e-7 and 1e-6 demonstrate "
            "that threshold-branch parity is not monotone in epsilon"
        ),
        "iterations": [10, 25, 50],
    },
    "provisional_floor_rejection_36": {
        "fixture_group": "gaussian_core_36",
        "filter_epsilons": [1e-10],
        "iterations": [10, 25, 50, 100],
    },
    "even_psf_comparison_40": {
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
            "final_odd_164": _run_suite(
                final_odd,
                filter_epsilons=(1e-8, 1e-7, 1e-6),
                iterations=(10, 25, 50),
                cupy=cupy,
            ),
            "provisional_floor_rejection_36": _run_suite(
                gaussian_core,
                filter_epsilons=(1e-10,),
                iterations=(10, 25, 50, 100),
                cupy=cupy,
            ),
            "even_psf_comparison_40": _run_suite(
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
            "dtype": "float32",
            "shape_must_match": True,
            "finite_masks_must_match": True,
            "outputs_must_be_finite": True,
            "nrmse_limit": NRMSE_LIMIT,
            "max_abs_formula": "1e-6 + 5e-6 * max(abs(cpu_reference))",
            "max_abs_floor": MAX_ABSOLUTE_FLOOR,
            "max_abs_peak_factor": MAX_ABSOLUTE_PEAK_FACTOR,
        },
        "suites": suites,
        "conclusion": conclusion,
        "limitations": [
            "This is single-host native-Windows RTX 5090 evidence.",
            "The sampled matrix cannot prove parity for every possible image or PSF.",
            "Exact-workload CPU/GPU parity remains mandatory before optimizer "
            "selection.",
            "The authored CPU default filter_epsilon=1e-12 is unchanged and "
            "remains outside the initial GPU region.",
            "No parameter is silently changed to qualify a workload for GPU execution.",
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

    summaries = []
    for epsilon in filter_epsilons:
        for iteration_count in iterations:
            selected = [
                record
                for record in records
                if record["filter_epsilon"] == float(epsilon)
                and record["iterations"] == int(iteration_count)
            ]
            worst = max(selected, key=lambda item: float(item["gate_score"]))
            summaries.append(
                {
                    "filter_epsilon": float(epsilon),
                    "iterations": int(iteration_count),
                    "case_count": len(selected),
                    "failure_count": sum(not bool(item["passed"]) for item in selected),
                    "worst_gate_score": float(worst["gate_score"]),
                    "worst_fixture_id": worst["fixture_id"],
                    "worst_nrmse": float(worst["nrmse"]),
                    "worst_max_abs": float(worst["max_abs"]),
                    "worst_cpu_peak": float(worst["cpu_peak"]),
                    "worst_max_abs_limit": float(worst["max_abs_limit"]),
                }
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
    from napari_vipp.core.operations import (
        richardson_lucy_deconvolution as cpu_rl,
    )
    from napari_vipp.core.progress import ProgressContext

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
    max_abs_limit = MAX_ABSOLUTE_FLOOR + MAX_ABSOLUTE_PEAK_FACTOR * cpu_peak
    gate_score = max(nrmse / NRMSE_LIMIT, max_abs / max_abs_limit)
    passed = bool(
        shape_equal
        and dtype_equal
        and finite_masks_equal
        and completely_finite
        and nrmse <= NRMSE_LIMIT
        and max_abs <= max_abs_limit
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
        "cpu_peak": cpu_peak,
        "nrmse": nrmse,
        "max_abs": max_abs,
        "max_abs_limit": max_abs_limit,
        "gate_score": gate_score,
        "passed": passed,
    }


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
    final_suite = _suite_mapping(suites, "final_odd_164")
    provisional = _suite_mapping(suites, "provisional_floor_rejection_36")
    even_suite = _suite_mapping(suites, "even_psf_comparison_40")
    final_summaries = _summary_mappings(final_suite)
    recommended = [
        item
        for item in final_summaries
        if float(item["filter_epsilon"]) == RECOMMENDED_FILTER_EPSILON
        and int(item["iterations"]) <= RECOMMENDED_MAXIMUM_ITERATIONS
    ]
    recommended_passes = bool(recommended) and all(
        int(item["failure_count"]) == 0 for item in recommended
    )
    provisional_rejected = any(
        int(item["iterations"]) >= RECOMMENDED_MAXIMUM_ITERATIONS
        and int(item["failure_count"]) > 0
        for item in _summary_mappings(provisional)
    )
    long_iterations_rejected = any(
        int(item["iterations"]) > RECOMMENDED_MAXIMUM_ITERATIONS
        and int(item["failure_count"]) > 0
        for item in final_summaries
    )
    even_psfs_rejected = any(
        int(item["failure_count"]) > 0 for item in _summary_mappings(even_suite)
    )
    higher_epsilon_not_monotone = any(
        float(item["filter_epsilon"]) > RECOMMENDED_FILTER_EPSILON
        and int(item["iterations"]) <= RECOMMENDED_MAXIMUM_ITERATIONS
        and int(item["failure_count"]) > 0
        for item in final_summaries
    )
    if not all(
        (
            recommended_passes,
            provisional_rejected,
            long_iterations_rejected,
            even_psfs_rejected,
            higher_epsilon_not_monotone,
        )
    ):
        raise AdmissionEvidenceError(
            "Completed measurements do not support the reviewed RL admission "
            "conclusion."
        )
    return {
        "required_filter_epsilon": RECOMMENDED_FILTER_EPSILON,
        "recommended_maximum_iterations": RECOMMENDED_MAXIMUM_ITERATIONS,
        "require_all_psf_extents_odd": True,
        "reviewed_exact_value_passed_sampled_matrix": recommended_passes,
        "provisional_1e_10_floor_rejected": provisional_rejected,
        "iterations_above_25_rejected": long_iterations_rejected,
        "even_psf_region_rejected": even_psfs_rejected,
        "higher_filter_epsilon_is_not_monotone": higher_epsilon_not_monotone,
        "public_exposure": "developer-hidden",
        "selection_requirement": (
            "exact-workload parity before timing or optimizer selection"
        ),
        "default_filter_epsilon_1e_12": "CPU fallback; never silently changed",
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
    expected_counts = {
        "final_odd_164": 164,
        "provisional_floor_rejection_36": 36,
        "even_psf_comparison_40": 40,
    }
    for name, expected_count in expected_counts.items():
        suite = _suite_mapping(suites, name)
        if suite.get("fixture_count") != expected_count:
            raise AdmissionEvidenceError(f"Suite {name!r} has a stale fixture count.")
        results = suite.get("results")
        if not isinstance(results, list) or len(results) != suite.get("result_count"):
            raise AdmissionEvidenceError(f"Suite {name!r} has incomplete raw results.")
    conclusion = document.get("conclusion")
    if not isinstance(conclusion, Mapping):
        raise AdmissionEvidenceError("Evidence conclusion is missing.")
    required_conclusion = {
        "required_filter_epsilon": RECOMMENDED_FILTER_EPSILON,
        "recommended_maximum_iterations": RECOMMENDED_MAXIMUM_ITERATIONS,
        "require_all_psf_extents_odd": True,
        "reviewed_exact_value_passed_sampled_matrix": True,
        "provisional_1e_10_floor_rejected": True,
        "iterations_above_25_rejected": True,
        "even_psf_region_rejected": True,
        "higher_filter_epsilon_is_not_monotone": True,
    }
    for key, expected in required_conclusion.items():
        if conclusion.get(key) != expected:
            raise AdmissionEvidenceError(f"Evidence conclusion {key!r} is stale.")
    if require_current_sources:
        provenance = document.get("source_provenance")
        if not isinstance(provenance, Mapping):
            raise AdmissionEvidenceError("Source provenance is missing.")
        _require_source_snapshot_unchanged(provenance)


def render_markdown(document: Mapping[str, object]) -> str:
    suites = document["suites"]
    environment = document["environment"]
    conclusion = document["conclusion"]
    lines = [
        "# CuPy Richardson-Lucy admission evidence",
        "",
        f"Generated: `{document['generated_at_utc']}`  ",
        f"Schema: `{document['schema']}` version {document['schema_version']}  ",
        f"Device: `{environment['cuda']['device_name']}`  ",
        f"Platform: `{environment['platform']}`  ",
        "",
        "This is deterministic, machine-local scientific-parity evidence. It is",
        "not a portable performance claim and does not waive exact-workload parity.",
        "",
        "## Policy gate",
        "",
        f"- NRMSE: `<= {NRMSE_LIMIT:g}`",
        "- Maximum absolute error: `<= 1e-6 + 5e-6 × CPU peak`",
        "- Shape, float32 dtype, finite masks, and complete finiteness must match.",
        "",
    ]
    for suite_name in (
        "final_odd_164",
        "provisional_floor_rejection_36",
        "even_psf_comparison_40",
    ):
        suite = suites[suite_name]
        lines.extend(
            [
                f"## {suite_name.replace('_', ' ')}",
                "",
                f"Fixtures: **{suite['fixture_count']}**  ",
                f"Manifest SHA-256: `{suite['fixture_manifest_sha256']}`",
                "",
                "| Filter epsilon | Iterations | Failures | Worst gate score | "
                "Worst fixture |",
                "|---:|---:|---:|---:|:---|",
            ]
        )
        for summary in suite["summaries"]:
            lines.append(
                "| "
                f"{summary['filter_epsilon']:.0e} | {summary['iterations']} | "
                f"{summary['failure_count']}/{summary['case_count']} | "
                f"{summary['worst_gate_score']:.9g} | "
                f"`{summary['worst_fixture_id']}` |"
            )
        lines.append("")
    lines.extend(
        [
            "## Evidence-backed initial contract",
            "",
            f"- `filter_epsilon == {conclusion['required_filter_epsilon']:.0e}`",
            f"- `iterations <= {conclusion['recommended_maximum_iterations']}`",
            "- every PSF extent is odd;",
            "- normalized PSF and default-safe clipping/scale controls;",
            "- developer-hidden exposure; and",
            "- exact-workload CPU/GPU parity before timing or optimizer selection.",
            "",
            "Higher epsilon values are not automatically safer: the ratio update has",
            "a threshold branch, and the 1e-7/1e-6 comparison contains parity "
            "failures.",
            "The authored CPU default `filter_epsilon=1e-12` remains unchanged and",
            "uses CPU fallback. VIPP must never silently raise it to qualify a GPU "
            "run.",
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
