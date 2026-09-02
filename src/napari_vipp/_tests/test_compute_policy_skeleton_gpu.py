from __future__ import annotations

import math
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from napari_vipp.core.compute import WorkloadDescriptor
from napari_vipp.core.compute_benchmark_adapter import (
    _validate_admitted_spec,
    operation_parity,
)
from napari_vipp.core.compute_policy import (
    SKELETON_ANALYSIS_MAXIMUM_SPATIAL_BLOCK_ELEMENTS,
    estimate_candidate_memory,
    evaluate_candidate_workload_support,
    validate_spec_policy_references,
)
from napari_vipp.core.compute_registry import ComputeRegistry
from napari_vipp.core.compute_specs import AdmissionTier, compute_specs_for
from napari_vipp.core.node_execution import PreparedNodeCall
from napari_vipp.core.skeleton_measurements import (
    SKELETON_PAYLOAD_HEADER_BYTES,
    skeleton_analysis_layout,
)
from napari_vipp.core.tables import TableData

SOURCE_ROOT = Path(__file__).resolve().parents[2]


def _spec():
    (spec,) = compute_specs_for("analyze_skeleton", include_cpu=False)
    return spec


def _workload(
    *,
    shape: tuple[int, ...] = (2, 10, 20, 30),
    dtype: str = "bool",
    spatial_ndim: int = 3,
    input_mode: str = "Already skeletonized",
    parameters: tuple[tuple[str, object], ...] = (),
) -> WorkloadDescriptor:
    return WorkloadDescriptor(
        "skeleton-node",
        "analyze_skeleton",
        (shape,),
        (dtype,),
        parameters=(
            ("spatial_mode", "3D ZYX" if spatial_ndim == 3 else "2D YX"),
            ("input_mode", input_mode),
            *parameters,
        ),
        resolved_spatial_ndim=spatial_ndim,
    )


def test_analyze_skeleton_spec_is_public_typed_host_boundary() -> None:
    spec = _spec()

    assert spec.implementation_id == "cupyx-analyze-skeleton-v1"
    assert spec.callable_ref == (
        "napari_vipp.core.gpu.cupy_skeleton_measurements:analyze_skeleton"
    )
    assert spec.admission_tier is AdmissionTier.PUBLIC_AUTO_CANDIDATE
    assert spec.implementation_library_id == "cupyx"
    assert spec.supported_spatial_ndims == (2, 3)
    assert spec.supports_device_residency and not spec.host_boundary
    assert spec.input_ports[0].public_dtypes == ("bool",)
    assert spec.output_ports[0].value_kind.value == "table"
    assert spec.output_ports[0].internal_dtypes == ("uint8",)
    assert spec.parity_policy_id == "skeleton-measurement-table-v1"
    assert spec.memory_model_id == "cupy-analyze-skeleton-packed-memory-v1"
    assert spec.host_finalizer_ref == (
        "napari_vipp.core.skeleton_measurements:finalize_analyze_skeleton_outputs"
    )
    validate_spec_policy_references(spec)
    assert (
        ComputeRegistry().implementation_spec(
            spec.implementation_id,
            spec.implementation_version,
        )
        == spec
    )


def test_analyze_skeleton_region_needs_no_value_scan_for_boolean_input() -> None:
    decision = evaluate_candidate_workload_support(_spec(), _workload())

    assert decision.supported
    assert not decision.requires_complete_facts


def test_numeric_coercion_and_skeletonization_stay_on_cpu() -> None:
    numeric = evaluate_candidate_workload_support(
        _spec(),
        _workload(dtype="uint8"),
    )
    skeletonize = evaluate_candidate_workload_support(
        _spec(),
        _workload(input_mode="Skeletonize first"),
    )
    invalid_mode = evaluate_candidate_workload_support(
        _spec(),
        _workload(input_mode="Something else"),
    )

    assert not numeric.supported and numeric.fallback_allowed
    assert not skeletonize.supported and skeletonize.fallback_allowed
    assert not invalid_mode.supported and not invalid_mode.fallback_allowed


def test_analyze_skeleton_region_supports_nontrailing_spatial_axes() -> None:
    workload = _workload(
        shape=(20, 4, 30),
        spatial_ndim=2,
        parameters=(
            ("axis_names", ("Y", "C", "X")),
            ("axis_types", ("space", "channel", "space")),
            ("axis_scales", (0.5, 1.0, 0.25)),
            ("axis_units", ("um", None, "um")),
        ),
    )

    assert evaluate_candidate_workload_support(_spec(), workload).supported


def test_analyze_skeleton_region_rejects_oversized_spatial_block() -> None:
    workload = _workload(
        shape=(SKELETON_ANALYSIS_MAXIMUM_SPATIAL_BLOCK_ELEMENTS, 1),
        spatial_ndim=2,
    )

    decision = evaluate_candidate_workload_support(_spec(), workload)

    assert not decision.supported
    assert decision.fallback_allowed
    assert "fewer than" in decision.reason_text


def test_analyze_skeleton_memory_uses_full_connectivity_component_bound() -> None:
    workload = _workload()
    layout = skeleton_analysis_layout(
        workload.input_shapes[0],
        spatial_mode="3D ZYX",
        resolved_spatial_ndim=3,
    )
    estimate = estimate_candidate_memory(_spec(), workload)
    elements = 2 * 10 * 20 * 30
    block_elements = 10 * 20 * 30
    input_bytes = elements * np.dtype(bool).itemsize
    maximum_rows = 2 * math.ceil(10 / 2) * math.ceil(20 / 2) * math.ceil(30 / 2)
    payload_bound = (
        SKELETON_PAYLOAD_HEADER_BYTES
        + maximum_rows * layout.record_words * np.dtype(np.uint64).itemsize
    )
    workspace = input_bytes + block_elements * 512 + 2 * payload_bound

    assert maximum_rows == elements // 8
    assert estimate.runtime_managed_peak_bytes == (
        input_bytes + payload_bound + workspace
    )
    assert estimate.host_materialization_peak_bytes == 7 * payload_bound
    assert estimate.uncertainty_bytes >= 64 * 1024**2

    doubled = estimate_candidate_memory(
        _spec(),
        _workload(shape=(4, 10, 20, 30)),
    )
    assert estimate.runtime_managed_peak_bytes < doubled.runtime_managed_peak_bytes
    assert doubled.runtime_managed_peak_bytes < 2 * estimate.runtime_managed_peak_bytes


def test_analyze_skeleton_parity_accepts_only_edge_count_summation_drift() -> None:
    repeats = 100_000
    mixed_edge_count = repeats * 3
    cpu_order = sum(
        value for _ in range(repeats) for value in (1.0, math.sqrt(2.0), math.sqrt(3.0))
    )
    gpu_grouped_order = repeats + repeats * math.sqrt(2.0) + repeats * math.sqrt(3.0)
    assert cpu_order != gpu_grouped_order

    columns = (
        "component_id",
        "component_voxel_fraction",
        "voxel_graph_edge_count",
        "skeleton_length_voxels",
        "skeleton_length_physical",
        "physical_unit",
    )
    reference = TableData(
        columns,
        ((1, 1.0, mixed_edge_count, cpu_order, cpu_order, "um"),),
        name="Skeleton network measurements",
        table_kind="Skeleton network",
        source_name="fixture",
        column_units=(
            ("skeleton_length_voxels", "voxels"),
            ("skeleton_length_physical", "um"),
        ),
    )
    reordered = replace(
        reference,
        rows=(
            (
                1,
                1.0,
                mixed_edge_count,
                gpu_grouped_order,
                gpu_grouped_order,
                "um",
            ),
        ),
    )
    changed_integer = replace(
        reordered,
        rows=(
            (
                1,
                1.0,
                mixed_edge_count + 1,
                gpu_grouped_order,
                gpu_grouped_order,
                "um",
            ),
        ),
    )
    changed_fraction = replace(
        reordered,
        rows=(
            (
                1,
                float(np.nextafter(1.0, 0.0)),
                mixed_edge_count,
                gpu_grouped_order,
                gpu_grouped_order,
                "um",
            ),
        ),
    )
    changed_length = replace(
        reordered,
        rows=(
            (
                1,
                1.0,
                mixed_edge_count,
                gpu_grouped_order + 1.0e-3,
                gpu_grouped_order,
                "um",
            ),
        ),
    )

    assert operation_parity("analyze_skeleton", reference, reordered).passed
    assert not operation_parity("analyze_skeleton", reference, changed_integer).passed
    assert not operation_parity("analyze_skeleton", reference, changed_fraction).passed
    assert not operation_parity("analyze_skeleton", reference, changed_length).passed


def test_analyze_skeleton_spec_is_accepted_by_benchmark_validation() -> None:
    skeleton = np.zeros((20, 30), dtype=bool)
    call = PreparedNodeCall(
        "skeleton-node",
        "analyze_skeleton",
        lambda data, **_kwargs: data,
        (skeleton,),
        kwargs={
            "spatial_mode": "2D YX",
            "input_mode": "Already skeletonized",
            "resolved_spatial_ndim": 2,
        },
    )

    _validate_admitted_spec(
        call,
        _spec(),
        ComputeRegistry(),
        allow_experimental=False,
    )


def test_analyze_skeleton_declaration_imports_without_cuda_modules() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(SOURCE_ROOT), environment.get("PYTHONPATH", "")))
    )
    script = r"""
import sys
from napari_vipp.core.compute_specs import compute_specs_for
spec = compute_specs_for('analyze_skeleton', include_cpu=False)[0]
assert spec.callable_ref.endswith(':analyze_skeleton')
import napari_vipp.core.gpu.cupy_skeleton_measurements as provider
assert provider.__all__ == ['analyze_skeleton']
for name in ('cupy', 'cupyx', 'cucim'):
    assert name not in sys.modules, name
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
