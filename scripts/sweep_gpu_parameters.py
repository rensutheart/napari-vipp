"""Audit interactive parameter changes across every admitted GPU implementation.

This maintainer command complements the scientific admission benchmarks.  It
uses VIPP's headless production pipeline path for small, deterministic inputs,
changes one authored value at a time, and revisits earlier values in the same
process.  The resulting JSON records the actual implementation, backend,
fallback and cleanup evidence, plus end-to-end elapsed time through synchronized
runtime cleanup.

Timing comparisons are deliberately relative within one lane.  A reported
``relative_cliff_signal`` is a review hint, not a portable performance limit or
an admission failure.  Machine-independent CI tests validate the catalog,
coverage accounting, evidence schema, and signal calculation without importing
or requiring a GPU provider.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

SCHEMA_ID = "napari-vipp-interactive-gpu-parameter-sweep"
SCHEMA_VERSION = 1
ADMISSION_SCHEMA = "napari-vipp-gpu-admission-suite-manifest"
ADMISSION_SCHEMA_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = Path(__file__).with_name("gpu_admission_suites.json")
REQUIRED_RUNTIME_ID = "cuda-cupy"
RELATIVE_CLIFF_RATIO = 8.0
RELATIVE_CLIFF_EXCESS_MULTIPLIER = 5.0


class SweepConfigurationError(ValueError):
    """The durable sweep catalog and admission manifest disagree."""


@dataclass(frozen=True, slots=True)
class AdmissionDeclaration:
    operation_id: str
    implementation_id: str
    implementation_version: str
    runtime_id: str
    library_id: str

    @property
    def key(self) -> str:
        return f"{self.operation_id}::{self.implementation_id}"


@dataclass(frozen=True, slots=True)
class SweepLane:
    """One authored parameter or admitted branch changed in isolation."""

    lane_id: str
    mutation_kind: str
    parameter_name: str
    values: tuple[object, ...]
    notes: str = ""

    def __post_init__(self) -> None:
        lane_id = str(self.lane_id).strip()
        parameter_name = str(self.parameter_name).strip()
        mutation_kind = str(self.mutation_kind).strip()
        values = tuple(self.values)
        if not lane_id or not parameter_name:
            raise SweepConfigurationError(
                "Sweep lane IDs and parameter names must not be empty."
            )
        if mutation_kind not in {
            "numeric_parameter",
            "categorical_parameter",
            "input_dtype_branch",
        }:
            raise SweepConfigurationError(
                f"Unsupported sweep mutation kind {mutation_kind!r}."
            )
        if len(values) < 4:
            raise SweepConfigurationError(
                f"Sweep lane {lane_id!r} needs a startup value, unseen value, "
                "and at least one revisit."
            )
        normalized = [_value_key(value) for value in values]
        if len(set(normalized)) < 2:
            raise SweepConfigurationError(
                f"Sweep lane {lane_id!r} must exercise at least two values."
            )
        revisited_after_startup = any(
            normalized[index] in normalized[1:index]
            for index in range(2, len(normalized))
        )
        if not revisited_after_startup:
            raise SweepConfigurationError(
                f"Sweep lane {lane_id!r} must revisit a non-startup value."
            )
        object.__setattr__(self, "lane_id", lane_id)
        object.__setattr__(self, "mutation_kind", mutation_kind)
        object.__setattr__(self, "parameter_name", parameter_name)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "notes", str(self.notes).strip())


@dataclass(frozen=True, slots=True)
class SweepCase:
    """Fixture and coverage treatment for one admitted implementation."""

    operation_id: str
    fixture_id: str
    shape: tuple[int, ...]
    dtype: str
    axes: str
    seed: int
    base_parameters: tuple[tuple[str, object], ...] = ()
    lanes: tuple[SweepLane, ...] = ()
    coverage_mode: str = "executed-sweep"
    classification: str = ""
    fixed_authored_parameters: tuple[tuple[str, object], ...] = ()
    delegated_to: str = ""

    def __post_init__(self) -> None:
        operation_id = str(self.operation_id).strip()
        fixture_id = str(self.fixture_id).strip()
        if not operation_id or not fixture_id:
            raise SweepConfigurationError(
                "Sweep operation and fixture IDs must not be empty."
            )
        if self.coverage_mode not in {
            "executed-sweep",
            "fixed-contract",
            "delegated-psf-sweep",
        }:
            raise SweepConfigurationError(
                f"Unsupported coverage mode {self.coverage_mode!r}."
            )
        if self.coverage_mode == "executed-sweep" and not self.lanes:
            raise SweepConfigurationError(
                f"Executed case {operation_id!r} must define at least one lane."
            )
        if self.coverage_mode != "executed-sweep" and self.lanes:
            raise SweepConfigurationError(
                f"Classified case {operation_id!r} must not define sweep lanes."
            )
        if self.coverage_mode == "delegated-psf-sweep" and not self.delegated_to:
            raise SweepConfigurationError(
                f"Delegated case {operation_id!r} must name its delegated harness."
            )
        if any(int(extent) < 1 for extent in self.shape):
            raise SweepConfigurationError("Sweep fixture extents must be positive.")
        object.__setattr__(self, "operation_id", operation_id)
        object.__setattr__(self, "fixture_id", fixture_id)
        object.__setattr__(self, "shape", tuple(map(int, self.shape)))
        object.__setattr__(self, "dtype", str(self.dtype).strip())
        object.__setattr__(self, "axes", str(self.axes).strip())
        object.__setattr__(self, "base_parameters", tuple(self.base_parameters))
        object.__setattr__(self, "lanes", tuple(self.lanes))
        object.__setattr__(
            self,
            "fixed_authored_parameters",
            tuple(self.fixed_authored_parameters),
        )


@dataclass(frozen=True, slots=True)
class StepObservation:
    """Production execution facts for one authored-value step."""

    elapsed_seconds: float
    runtime_id: str
    implementation_library_id: str
    implementation_id: str
    implementation_version: str
    decision_kind: str
    fallback_used: bool
    cleanup_succeeded: bool
    fallback_records: tuple[Mapping[str, object], ...] = ()
    error: str = ""
    device_id: str = ""
    device_name: str = ""

    def __post_init__(self) -> None:
        elapsed = float(self.elapsed_seconds)
        if not math.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError("Step elapsed time must be finite and non-negative.")
        object.__setattr__(self, "elapsed_seconds", elapsed)
        object.__setattr__(self, "fallback_records", tuple(self.fallback_records))


EXPECTED_IMPLEMENTATIONS = {
    "rolling_ball_background": "cupy-rolling-ball-background-v1",
    "subtract_background": "cupy-subtract-background-v1",
    "median_filter": "cupy-median-filter-v1",
    "gaussian_blur": "cupy-gaussian-blur-v1",
    "gaussian_blur_3d": "cupy-gaussian-blur-3d-v1",
    "convert_dtype": "cupyx-convert-dtype-preserve-f32-v1",
    "binary_threshold": "cupy-binary-threshold-f32-exact-v1",
    "extract_channel": "cupy-extract-channel-view-v1",
    "richardson_lucy_deconvolution": "rl-cupy-f32-v1",
    "richardson_lucy_tv_deconvolution": "rl-tv-cupy-f32-v1",
    "canny_edges": "cupyx-canny-edges-exact-v1",
    "otsu_threshold": "cupy-otsu-threshold-exact-v1",
    "sigma_filter": "cupy-sigma-filter-v1",
    "label_connected_components": "cupyx-connected-components-v1",
    "fill_holes": "cupyx-fill-holes-all-v1",
    "remove_small_objects": "cupyx-remove-small-objects-bool-v1",
    "measure_objects": "cucim-measure-objects-basic-v1",
    "measure_objects_intensity": "cucim-measure-objects-intensity-basic-v1",
}

_MASK_INPUT_TARGETS = frozenset(
    {
        "label_connected_components",
        "fill_holes",
        "remove_small_objects",
    }
)


def production_scaffold(operation_id: str) -> dict[str, object] | None:
    """Return the explicit type bridge required before a mask-input target."""

    if str(operation_id).strip() not in _MASK_INPUT_TARGETS:
        return None
    return {
        "operation_id": "binary_threshold",
        "parameters": {"threshold": 0.5},
        "compute_preference": "cpu",
        "purpose": "static-image-to-mask-port-type-bridge-v1",
        "included_in_elapsed_time": True,
    }


def build_target_pipeline(
    case: SweepCase,
    parameters: Mapping[str, object],
    implementation_id: str,
):
    """Build one type-correct graph and return its strict target preference."""

    from napari_vipp.core.pipeline import PrototypePipeline

    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    preferences: dict[str, str] = {}
    source_id = "input"
    scaffold = production_scaffold(case.operation_id)
    if scaffold is not None:
        scaffold_node = pipeline.add_node(str(scaffold["operation_id"]))
        for name, value in dict(scaffold["parameters"]).items():
            pipeline.set_param(scaffold_node.id, name, value)
        connection = pipeline.connect(source_id, scaffold_node.id)
        if not connection.success:
            raise RuntimeError(
                f"Cannot connect input to {scaffold_node.operation_id} scaffold: "
                f"{connection.message}"
            )
        source_id = scaffold_node.id
        preferences[scaffold_node.id] = str(scaffold["compute_preference"])
    target = pipeline.add_node(case.operation_id)
    for name, value in parameters.items():
        pipeline.set_param(target.id, name, value)
    connection = pipeline.connect(source_id, target.id)
    if not connection.success:
        raise RuntimeError(
            f"Cannot connect {source_id} to {case.operation_id} target: "
            f"{connection.message}"
        )
    preferences[target.id] = f"implementation:{implementation_id}"
    return pipeline, target.id, preferences


def _numeric_lane(lane_id: str, parameter: str, values: Sequence[object]) -> SweepLane:
    return SweepLane(lane_id, "numeric_parameter", parameter, tuple(values))


def _choice_lane(lane_id: str, parameter: str, values: Sequence[object]) -> SweepLane:
    return SweepLane(lane_id, "categorical_parameter", parameter, tuple(values))


def sweep_catalog() -> tuple[SweepCase, ...]:
    """Return the bounded, deterministic coverage plan for all 18 declarations."""

    background_base = (
        ("radius", 2.0),
        ("light_background", False),
        ("disable_smoothing", False),
        ("spatial_mode", "2D YX"),
    )
    cleanup_base = (
        ("spatial_mode", "2D YX"),
        ("connectivity", "Face connected"),
    )
    measurement_flags = (
        ("include_shape_descriptors", False),
        ("include_axis_descriptors", False),
        ("include_2d_boundary_descriptors", False),
        ("include_derived_shape_ratios", False),
        ("include_2d_shape_moments", False),
    )
    psf_delegation = (
        "scripts/benchmark_gpu_rl_parameter_sweep.py; this catalog preserves "
        "one coverage row and does not duplicate its multi-input FFT setup"
    )
    return (
        SweepCase(
            "rolling_ball_background",
            "uint16-yx-background-v1",
            (47, 53),
            "uint16",
            "YX",
            20_260_801,
            background_base,
            (_numeric_lane("radius", "radius", (2, 3, 4, 5, 3, 4)),),
        ),
        SweepCase(
            "subtract_background",
            "uint16-yx-background-v1",
            (47, 53),
            "uint16",
            "YX",
            20_260_802,
            background_base + (("clip_negative", True),),
            (_numeric_lane("radius", "radius", (2, 3, 4, 5, 3, 4)),),
        ),
        SweepCase(
            "median_filter",
            "uint16-yx-filter-v1",
            (47, 53),
            "uint16",
            "YX",
            20_260_803,
            (("size", 3),),
            (_numeric_lane("size", "size", (3, 5, 7, 9, 5, 7)),),
        ),
        SweepCase(
            "gaussian_blur",
            "float32-yx-filter-v1",
            (47, 53),
            "float32",
            "YX",
            20_260_804,
            (("sigma", 0.8),),
            (_numeric_lane("sigma", "sigma", (0.8, 1.2, 1.7, 2.3, 1.2, 1.7)),),
        ),
        SweepCase(
            "gaussian_blur_3d",
            "float32-zyx-filter-v1",
            (7, 31, 37),
            "float32",
            "ZYX",
            20_260_805,
            (
                ("sigma_z", 0.8),
                ("sigma_y", 1.1),
                ("sigma_x", 1.3),
                ("lock_xy", False),
            ),
            tuple(
                _numeric_lane(
                    axis,
                    axis,
                    (0.8, 1.1, 1.4, 1.8, 1.1, 1.4),
                )
                for axis in ("sigma_z", "sigma_y", "sigma_x")
            ),
        ),
        SweepCase(
            "convert_dtype",
            "integer-yx-conversion-v1",
            (47, 53),
            "uint8",
            "YX",
            20_260_806,
            (("output_dtype", "float32"), ("scaling", "preserve")),
            (
                SweepLane(
                    "input-dtype",
                    "input_dtype_branch",
                    "input_dtype",
                    ("uint8", "uint16", "uint8", "uint16"),
                    "Both admitted lossless integer input branches are exercised.",
                ),
            ),
        ),
        SweepCase(
            "binary_threshold",
            "float32-unit-yx-v1",
            (47, 53),
            "float32",
            "YX",
            20_260_807,
            (("threshold", 0.25),),
            (
                _numeric_lane(
                    "threshold",
                    "threshold",
                    (0.15, 0.25, 0.4, 0.65, 0.25, 0.4),
                ),
            ),
        ),
        SweepCase(
            "extract_channel",
            "uint16-cyx-v1",
            (4, 31, 37),
            "uint16",
            "CYX",
            20_260_808,
            (("channel", 0),),
            (_numeric_lane("channel", "channel", (0, 1, 2, 3, 1, 2)),),
        ),
        SweepCase(
            "richardson_lucy_deconvolution",
            "delegated-rl-psf-v1",
            (31, 37),
            "float32",
            "YX",
            20_260_809,
            coverage_mode="delegated-psf-sweep",
            classification=(
                "Multi-input RL execution and changing PSF dimensions are covered "
                "by the dedicated PSF harness."
            ),
            delegated_to=psf_delegation,
        ),
        SweepCase(
            "richardson_lucy_tv_deconvolution",
            "delegated-rl-tv-psf-v1",
            (31, 37),
            "float32",
            "YX",
            20_260_810,
            coverage_mode="delegated-psf-sweep",
            classification=(
                "Multi-input RL-TV execution and changing PSF dimensions are "
                "covered by the dedicated PSF harness."
            ),
            delegated_to=psf_delegation,
        ),
        SweepCase(
            "canny_edges",
            "uint16-yx-canny-v1",
            (47, 53),
            "uint16",
            "YX",
            20_260_811,
            (("sigma", 1.0), ("low_quantile", 0.1), ("high_quantile", 0.3)),
            (
                _numeric_lane("sigma", "sigma", (0.8, 1.1, 1.5, 2.0, 1.1, 1.5)),
                _numeric_lane(
                    "low-quantile",
                    "low_quantile",
                    (0.05, 0.1, 0.15, 0.2, 0.1, 0.15),
                ),
                _numeric_lane(
                    "high-quantile",
                    "high_quantile",
                    (0.2, 0.3, 0.4, 0.5, 0.3, 0.4),
                ),
            ),
        ),
        SweepCase(
            "otsu_threshold",
            "float32-unit-zyx-v1",
            (3, 31, 37),
            "float32",
            "ZYX",
            20_260_812,
            (("threshold_scope", "Stack histogram"), ("histogram_bins", 128)),
            (
                _numeric_lane(
                    "histogram-bins",
                    "histogram_bins",
                    (64, 128, 256, 512, 128, 256),
                ),
                _choice_lane(
                    "histogram-scope",
                    "threshold_scope",
                    (
                        "Stack histogram",
                        "Slice histogram",
                        "Stack histogram",
                        "Slice histogram",
                    ),
                ),
            ),
        ),
        SweepCase(
            "sigma_filter",
            "float32-yx-sigma-v1",
            (41, 47),
            "float32",
            "YX",
            20_260_813,
            (
                ("radius", 1.0),
                ("sigma_width", 2.0),
                ("minimum_pixel_fraction", 0.2),
                ("outlier_aware", True),
            ),
            (
                _numeric_lane("radius", "radius", (1.0, 1.5, 2.0, 2.5, 1.5, 2.0)),
                _numeric_lane(
                    "sigma-width",
                    "sigma_width",
                    (1.0, 1.5, 2.0, 2.5, 1.5, 2.0),
                ),
                _numeric_lane(
                    "minimum-pixel-fraction",
                    "minimum_pixel_fraction",
                    (0.1, 0.2, 0.35, 0.5, 0.2, 0.35),
                ),
            ),
        ),
        SweepCase(
            "label_connected_components",
            "bool-yx-components-v1",
            (47, 53),
            "bool",
            "YX",
            20_260_814,
            cleanup_base,
            (
                _choice_lane(
                    "connectivity",
                    "connectivity",
                    (
                        "Face connected",
                        "Full connectivity",
                        "Face connected",
                        "Full connectivity",
                    ),
                ),
            ),
        ),
        SweepCase(
            "fill_holes",
            "bool-yx-holes-v1",
            (47, 53),
            "bool",
            "YX",
            20_260_815,
            (("max_hole_size", 0),) + cleanup_base,
            (
                _choice_lane(
                    "connectivity",
                    "connectivity",
                    (
                        "Face connected",
                        "Full connectivity",
                        "Face connected",
                        "Full connectivity",
                    ),
                ),
            ),
            fixed_authored_parameters=(("max_hole_size", 0),),
        ),
        SweepCase(
            "remove_small_objects",
            "bool-yx-components-v1",
            (47, 53),
            "bool",
            "YX",
            20_260_816,
            (("min_size", 3),) + cleanup_base,
            (
                _numeric_lane("minimum-size", "min_size", (0, 3, 7, 15, 3, 7)),
                _choice_lane(
                    "connectivity",
                    "connectivity",
                    (
                        "Face connected",
                        "Full connectivity",
                        "Face connected",
                        "Full connectivity",
                    ),
                ),
            ),
        ),
        SweepCase(
            "measure_objects",
            "fixed-basic-measurements-v1",
            (47, 53),
            "int32",
            "YX",
            20_260_817,
            coverage_mode="fixed-contract",
            classification=(
                "The promoted device payload is the basic profile only. All five "
                "authored descriptor flags are fixed false; enabling any flag "
                "selects the authoritative CPU implementation."
            ),
            fixed_authored_parameters=measurement_flags,
        ),
        SweepCase(
            "measure_objects_intensity",
            "fixed-basic-intensity-measurements-v1",
            (47, 53),
            "int32+float32",
            "YX",
            20_260_818,
            coverage_mode="fixed-contract",
            classification=(
                "The promoted two-input device payload is the basic profile only. "
                "All five authored descriptor flags are fixed false; enabling any "
                "flag selects the authoritative CPU implementation."
            ),
            fixed_authored_parameters=measurement_flags,
        ),
    )


def load_admission_manifest(
    path: Path = DEFAULT_MANIFEST,
) -> tuple[AdmissionDeclaration, ...]:
    """Strictly read the declaration table used by release admission."""

    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SweepConfigurationError(f"Cannot read admission manifest: {exc}") from exc
    if not isinstance(document, dict):
        raise SweepConfigurationError("Admission manifest root must be an object.")
    if document.get("schema") != ADMISSION_SCHEMA:
        raise SweepConfigurationError("Unexpected GPU admission manifest schema.")
    if document.get("schema_version") != ADMISSION_SCHEMA_VERSION:
        raise SweepConfigurationError("Unexpected GPU admission manifest version.")
    raw_declarations = document.get("implementations")
    if not isinstance(raw_declarations, list) or not raw_declarations:
        raise SweepConfigurationError(
            "Admission manifest must contain implementation declarations."
        )
    declarations: list[AdmissionDeclaration] = []
    for index, raw in enumerate(raw_declarations):
        if not isinstance(raw, dict):
            raise SweepConfigurationError(
                f"Admission declaration {index} must be an object."
            )
        required = {
            "operation_id",
            "implementation_id",
            "implementation_version",
            "runtime_id",
            "library_id",
        }
        if set(raw) != required:
            raise SweepConfigurationError(
                f"Admission declaration {index} has unexpected fields."
            )
        declaration = AdmissionDeclaration(
            operation_id=str(raw["operation_id"]).strip(),
            implementation_id=str(raw["implementation_id"]).strip(),
            implementation_version=str(raw["implementation_version"]).strip(),
            runtime_id=str(raw["runtime_id"]).strip(),
            library_id=str(raw["library_id"]).strip(),
        )
        if not all(asdict(declaration).values()):
            raise SweepConfigurationError(
                f"Admission declaration {index} contains an empty identity field."
            )
        declarations.append(declaration)
    keys = [declaration.key for declaration in declarations]
    if len(set(keys)) != len(keys):
        raise SweepConfigurationError("Admission declarations contain duplicates.")
    return tuple(declarations)


def validate_catalog(
    cases: Sequence[SweepCase],
    declarations: Sequence[AdmissionDeclaration],
) -> None:
    """Require one conscious coverage treatment for every promoted identity."""

    case_operations = [case.operation_id for case in cases]
    if len(set(case_operations)) != len(case_operations):
        raise SweepConfigurationError("Sweep catalog contains duplicate operations.")
    declaration_by_operation = {
        declaration.operation_id: declaration for declaration in declarations
    }
    if len(declaration_by_operation) != len(tuple(declarations)):
        raise SweepConfigurationError(
            "The sweep requires exactly one admitted implementation per operation."
        )
    if set(case_operations) != set(declaration_by_operation):
        missing = sorted(set(declaration_by_operation) - set(case_operations))
        extra = sorted(set(case_operations) - set(declaration_by_operation))
        raise SweepConfigurationError(
            f"Sweep/admission coverage drift (missing={missing}, extra={extra})."
        )
    manifest_identities = {
        operation: declaration.implementation_id
        for operation, declaration in declaration_by_operation.items()
    }
    if manifest_identities != EXPECTED_IMPLEMENTATIONS:
        raise SweepConfigurationError(
            "Admission identities changed; update and review the interactive sweep "
            "catalog in the same change."
        )
    for declaration in declarations:
        if declaration.runtime_id != REQUIRED_RUNTIME_ID:
            raise SweepConfigurationError(
                f"{declaration.key} is not a CUDA/CuPy declaration."
            )
        if declaration.implementation_version != "1":
            raise SweepConfigurationError(
                f"{declaration.key} has an unreviewed implementation version."
            )


def describe_coverage(
    cases: Sequence[SweepCase],
    declarations: Sequence[AdmissionDeclaration],
) -> dict[str, object]:
    """Return provider-free coverage accounting for review and ordinary CI."""

    validate_catalog(cases, declarations)
    declaration_by_operation = {
        declaration.operation_id: declaration for declaration in declarations
    }
    rows = []
    for case in cases:
        declaration = declaration_by_operation[case.operation_id]
        rows.append(
            {
                "operation_id": case.operation_id,
                "implementation_id": declaration.implementation_id,
                "implementation_version": declaration.implementation_version,
                "runtime_id": declaration.runtime_id,
                "library_id": declaration.library_id,
                "coverage_mode": case.coverage_mode,
                "lanes": [
                    {
                        "lane_id": lane.lane_id,
                        "mutation_kind": lane.mutation_kind,
                        "parameter_name": lane.parameter_name,
                        "values": list(lane.values),
                    }
                    for lane in case.lanes
                ],
                "fixed_authored_parameters": dict(case.fixed_authored_parameters),
                "classification": case.classification,
                "delegated_to": case.delegated_to,
            }
        )
    return {
        "admitted_implementation_count": len(rows),
        "executed_sweep_count": sum(
            row["coverage_mode"] == "executed-sweep" for row in rows
        ),
        "fixed_contract_count": sum(
            row["coverage_mode"] == "fixed-contract" for row in rows
        ),
        "delegated_psf_sweep_count": sum(
            row["coverage_mode"] == "delegated-psf-sweep" for row in rows
        ),
        "rows": rows,
    }


StepRunner = Callable[
    [SweepCase, SweepLane, object, AdmissionDeclaration], StepObservation
]


def run_parameter_sweep(
    *,
    runner: StepRunner,
    cases: Sequence[SweepCase] | None = None,
    declarations: Sequence[AdmissionDeclaration] | None = None,
    clock: Callable[[], str] | None = None,
) -> dict[str, object]:
    """Execute the catalog through ``runner`` and assemble stable JSON evidence."""

    selected_cases = tuple(sweep_catalog() if cases is None else cases)
    selected_declarations = tuple(
        load_admission_manifest() if declarations is None else declarations
    )
    validate_catalog(selected_cases, selected_declarations)
    declaration_by_operation = {
        declaration.operation_id: declaration for declaration in selected_declarations
    }
    timestamp = datetime.now(UTC).isoformat() if clock is None else str(clock()).strip()
    case_records: list[dict[str, object]] = []
    hard_issue_count = 0
    relative_cliff_count = 0
    device_ids: set[str] = set()
    device_names: set[str] = set()
    for case in selected_cases:
        declaration = declaration_by_operation[case.operation_id]
        record: dict[str, object] = {
            "operation_id": case.operation_id,
            "implementation_id": declaration.implementation_id,
            "implementation_version": declaration.implementation_version,
            "runtime_id": declaration.runtime_id,
            "library_id": declaration.library_id,
            "coverage_mode": case.coverage_mode,
            "fixture": {
                "fixture_id": case.fixture_id,
                "shape": list(case.shape),
                "dtype": case.dtype,
                "axes": case.axes,
                "seed": case.seed,
            },
            "base_parameters": dict(case.base_parameters),
            "fixed_authored_parameters": dict(case.fixed_authored_parameters),
            "production_scaffold": production_scaffold(case.operation_id),
        }
        if case.coverage_mode != "executed-sweep":
            record["classification"] = case.classification
            if case.delegated_to:
                record["delegated_to"] = case.delegated_to
            record["lanes"] = []
            case_records.append(record)
            continue
        lane_records = []
        for lane in case.lanes:
            steps = []
            first_step_by_value: dict[str, dict[str, object]] = {}
            for index, value in enumerate(lane.values):
                value_key = _value_key(value)
                if index == 0:
                    occurrence = "startup"
                elif value_key in first_step_by_value:
                    occurrence = "revisit"
                else:
                    occurrence = "unseen"
                observation = runner(case, lane, value, declaration)
                if observation.device_id:
                    device_ids.add(observation.device_id)
                if observation.device_name:
                    device_names.add(observation.device_name)
                issues = _step_issues(observation, declaration)
                hard_issue_count += len(issues)
                step = {
                    "index": index,
                    "authored_value": value,
                    "occurrence": occurrence,
                    **asdict(observation),
                    "hard_issues": issues,
                }
                steps.append(step)
                first_step_by_value.setdefault(value_key, step)
            comparisons = _relative_comparisons(steps)
            relative_cliff_count += sum(
                bool(item["relative_cliff_signal"]) for item in comparisons
            )
            lane_records.append(
                {
                    "lane_id": lane.lane_id,
                    "mutation_kind": lane.mutation_kind,
                    "parameter_name": lane.parameter_name,
                    "notes": lane.notes,
                    "steps": steps,
                    "comparisons": comparisons,
                }
            )
        record["lanes"] = lane_records
        case_records.append(record)
    coverage = describe_coverage(selected_cases, selected_declarations)
    return {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "created_at": timestamp,
        "source": {
            "admission_manifest": str(DEFAULT_MANIFEST.relative_to(PROJECT_ROOT)),
            "admission_manifest_sha256": _file_sha256(DEFAULT_MANIFEST),
            "sweep_harness": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
            "sweep_harness_sha256": _file_sha256(Path(__file__)),
        },
        "timing_semantics": {
            "scope": "production-headless-request-through-synchronized-cleanup-v1",
            "recorded_type_scaffolds_included": True,
            "comparison": "matched-first-unseen-versus-later-revisit-v1",
            "startup_step_excluded": True,
            "relative_only": True,
            "relative_cliff_ratio": RELATIVE_CLIFF_RATIO,
            "relative_cliff_excess_multiplier": RELATIVE_CLIFF_EXCESS_MULTIPLIER,
            "interpretation": (
                "A relative cliff signal requests investigation; it is not an "
                "absolute latency threshold, portable benchmark, or scientific "
                "admission failure."
            ),
        },
        "coverage": coverage,
        "environment": {
            "device_ids": sorted(device_ids),
            "device_names": sorted(device_names),
        },
        "cases": case_records,
        "summary": {
            "hard_issue_count": hard_issue_count,
            "relative_cliff_signal_count": relative_cliff_count,
            "executed_step_count": sum(
                len(lane["steps"]) for case in case_records for lane in case["lanes"]
            ),
            "complete_coverage": len(case_records) == len(selected_declarations),
        },
    }


def _step_issues(
    observation: StepObservation,
    declaration: AdmissionDeclaration,
) -> list[str]:
    issues = []
    if observation.error:
        issues.append(f"execution-error:{observation.error}")
    expected = (
        declaration.runtime_id,
        declaration.library_id,
        declaration.implementation_id,
        declaration.implementation_version,
    )
    actual = (
        observation.runtime_id,
        observation.implementation_library_id,
        observation.implementation_id,
        observation.implementation_version,
    )
    if actual != expected:
        issues.append("actual-implementation-mismatch")
    if observation.decision_kind != "selected":
        issues.append(f"unexpected-decision-kind:{observation.decision_kind}")
    if observation.fallback_used or observation.fallback_records:
        issues.append("fallback-observed")
    if not observation.cleanup_succeeded:
        issues.append("cleanup-failed-or-unconfirmed")
    return issues


def _relative_comparisons(
    steps: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    first_by_value: dict[str, Mapping[str, object]] = {}
    revisit_elapsed = [
        float(step["elapsed_seconds"])
        for step in steps
        if step["occurrence"] == "revisit" and not step["hard_issues"]
    ]
    typical_revisit = statistics.median(revisit_elapsed) if revisit_elapsed else 0.0
    comparisons: list[dict[str, object]] = []
    for step in steps:
        key = _value_key(step["authored_value"])
        if key not in first_by_value:
            first_by_value[key] = step
            continue
        first = first_by_value[key]
        if first["occurrence"] == "startup" or step["occurrence"] != "revisit":
            continue
        first_elapsed = float(first["elapsed_seconds"])
        revisit = float(step["elapsed_seconds"])
        ratio = (
            sys.float_info.max
            if revisit == 0.0 and first_elapsed > 0.0
            else (1.0 if revisit == 0.0 else first_elapsed / revisit)
        )
        excess = max(0.0, first_elapsed - revisit)
        relative_signal = bool(
            not first["hard_issues"]
            and not step["hard_issues"]
            and ratio >= RELATIVE_CLIFF_RATIO
            and (
                typical_revisit == 0.0
                or excess >= RELATIVE_CLIFF_EXCESS_MULTIPLIER * typical_revisit
            )
        )
        comparisons.append(
            {
                "authored_value": step["authored_value"],
                "first_unseen_step_index": int(first["index"]),
                "revisit_step_index": int(step["index"]),
                "first_unseen_elapsed_seconds": first_elapsed,
                "revisit_elapsed_seconds": revisit,
                "excess_seconds": excess,
                "elapsed_ratio": ratio,
                "relative_cliff_signal": relative_signal,
            }
        )
    return comparisons


class ProductionSweepRunner:
    """Invoke one-node workflows through VIPP's production execution boundary."""

    def __init__(self, *, device_id: str = "") -> None:
        # All core and optional-heavy imports stay behind explicit execution.
        import numpy as np

        from napari_vipp.core.compute_registry import ComputeRegistry

        self._np = np
        self._registry = ComputeRegistry()
        probe = self._registry.probe_runtime(REQUIRED_RUNTIME_ID, refresh=True)
        if not probe.available or not probe.selected_device_id:
            self._registry.close()
            raise RuntimeError(probe.message or "The CUDA runtime is unavailable.")
        requested = str(device_id).strip()
        selected = requested or probe.selected_device_id
        if selected not in {device.device_id for device in probe.devices}:
            self._registry.close()
            raise RuntimeError(
                f"CUDA device {selected!r} was not reported by the runtime."
            )
        self.device_id = selected
        self._inputs: dict[tuple[str, str], object] = {}
        self._run_id = 0

    def close(self) -> None:
        self._registry.close()

    def __enter__(self) -> ProductionSweepRunner:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def __call__(
        self,
        case: SweepCase,
        lane: SweepLane,
        authored_value: object,
        declaration: AdmissionDeclaration,
    ) -> StepObservation:
        try:
            return self._execute_step(case, lane, authored_value, declaration)
        except Exception as exc:
            raise RuntimeError(
                "GPU parameter sweep step failed for "
                f"operation={case.operation_id!r}, lane={lane.lane_id!r}, "
                f"authored_value={authored_value!r}: {exc}"
            ) from exc

    def _execute_step(
        self,
        case: SweepCase,
        lane: SweepLane,
        authored_value: object,
        declaration: AdmissionDeclaration,
    ) -> StepObservation:
        from napari_vipp.core.compute import ComputeMode, ComputeRequest, FallbackPolicy
        from napari_vipp.core.execution import (
            PipelineRunRequest,
            execute_pipeline_request,
        )
        from napari_vipp.core.workflow import serialize_workflow

        self._run_id += 1
        dtype = case.dtype
        parameters = dict(case.base_parameters)
        if lane.mutation_kind == "input_dtype_branch":
            dtype = str(authored_value)
        else:
            parameters[lane.parameter_name] = authored_value
        data = self._input(case, dtype)
        pipeline, target_node_id, preferences = build_target_pipeline(
            case,
            parameters,
            declaration.implementation_id,
        )
        compute_request = ComputeRequest(
            mode=ComputeMode.CUSTOM,
            node_preferences=preferences,
            runtime_id=REQUIRED_RUNTIME_ID,
            device_id=self.device_id,
            fallback_policy=FallbackPolicy.STRICT,
        )
        request = PipelineRunRequest(
            run_id=self._run_id,
            workflow=serialize_workflow(pipeline),
            input_data=data,
            input_metadata={"axes": case.axes},
            input_name=case.fixture_id,
            source_payloads={},
            compute_request=compute_request,
            retain_node_ids=frozenset({target_node_id}),
        )
        started = time.perf_counter()
        result = execute_pipeline_request(request, compute_registry=self._registry)
        elapsed = time.perf_counter() - started
        report = result.execution_report
        decision = None
        if report is not None:
            decision = next(
                (
                    item
                    for item in report.actual_decisions
                    if item.node_id == target_node_id
                ),
                None,
            )
        environment = None if report is None else report.environment
        fallback_records = ()
        cleanup_succeeded = False
        if report is not None:
            fallback_records = tuple(item.as_dict() for item in report.fallback_records)
            cleanup_succeeded = bool(report.cleanup_succeeded)
        elif result.failure is not None:
            fallback_records = tuple(
                item.as_dict() for item in result.failure.fallback_records
            )
            cleanup_succeeded = bool(result.failure.cleanup_succeeded)
        return StepObservation(
            elapsed_seconds=elapsed,
            runtime_id="" if decision is None else decision.runtime_id,
            implementation_library_id=(
                "" if decision is None else decision.implementation_library_id
            ),
            implementation_id="" if decision is None else decision.implementation_id,
            implementation_version=(
                "" if decision is None else decision.implementation_version
            ),
            decision_kind=("" if decision is None else decision.decision_kind.value),
            fallback_used=(False if decision is None else decision.fallback_used),
            cleanup_succeeded=cleanup_succeeded,
            fallback_records=fallback_records,
            error=result.error,
            device_id=("" if environment is None else environment.device_id),
            device_name=("" if environment is None else environment.device_name),
        )

    def _input(self, case: SweepCase, dtype: str) -> object:
        key = (case.fixture_id, dtype)
        cached = self._inputs.get(key)
        if cached is not None:
            return cached
        np = self._np
        rng = np.random.default_rng(case.seed)
        target_dtype = np.dtype(dtype)
        if target_dtype == np.dtype(bool):
            indices = np.indices(case.shape)
            values = sum((axis + 1) * grid for axis, grid in enumerate(indices))
            data = values % 7 < 3
            if data.ndim >= 2 and min(data.shape[-2:]) >= 7:
                data[..., 2:-2, 2:-2] = True
                data[..., 4:-4, 4:-4] = False
                data[..., 6:-6, 6:-6] = True
        elif np.issubdtype(target_dtype, np.integer):
            high = min(int(np.iinfo(target_dtype).max), 4095) + 1
            data = rng.integers(0, high, size=case.shape, dtype=target_dtype)
        else:
            data = rng.random(case.shape, dtype=np.float32).astype(
                target_dtype,
                copy=False,
            )
        data.setflags(write=False)
        self._inputs[key] = data
        return data


def _value_key(value: object) -> str:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_write_json(path: Path, document: Mapping[str, object]) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(document, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
    return destination


def evidence_exit_code(
    document: Mapping[str, object],
    *,
    fail_on_relative_cliff: bool = False,
) -> int:
    """Return a strict status for integrity failures and optional review signals."""

    summary = document.get("summary")
    if not isinstance(summary, Mapping):
        raise TypeError("Sweep evidence must contain an object summary.")
    hard_issues = int(summary.get("hard_issue_count", 0))
    cliff_signals = int(summary.get("relative_cliff_signal_count", 0))
    if hard_issues:
        return 2
    if fail_on_relative_cliff and cliff_signals:
        return 1
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="JSON evidence path. Required unless --describe is used.",
    )
    parser.add_argument(
        "--device-id",
        default="",
        help="Exact CUDA device ID (default: the runtime-selected device).",
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Print provider-free coverage accounting without executing CUDA.",
    )
    parser.add_argument(
        "--fail-on-relative-cliff",
        action="store_true",
        help=(
            "Return status 1 when a relative cliff signal is present. Signals "
            "remain review hints, not fixed performance thresholds."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        cases = sweep_catalog()
        declarations = load_admission_manifest()
        coverage = describe_coverage(cases, declarations)
        if args.describe:
            print(json.dumps(coverage, allow_nan=False, indent=2, sort_keys=True))
            return 0
        if args.output is None:
            raise SweepConfigurationError("--output is required for an execution run.")
        with ProductionSweepRunner(device_id=args.device_id) as runner:
            document = run_parameter_sweep(
                runner=runner,
                cases=cases,
                declarations=declarations,
            )
        output = _atomic_write_json(args.output, document)
    except Exception as exc:
        print(
            f"GPU parameter sweep failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    summary = document["summary"]
    print(
        f"Wrote GPU parameter sweep evidence to {output} "
        f"({summary['executed_step_count']} steps, "
        f"{summary['hard_issue_count']} hard issues, "
        f"{summary['relative_cliff_signal_count']} relative cliff signals)."
    )
    return evidence_exit_code(
        document,
        fail_on_relative_cliff=args.fail_on_relative_cliff,
    )


if __name__ == "__main__":
    raise SystemExit(main())
