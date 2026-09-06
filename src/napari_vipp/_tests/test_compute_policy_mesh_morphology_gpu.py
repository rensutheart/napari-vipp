from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from napari_vipp.core.compute import WorkloadDescriptor
from napari_vipp.core.compute_benchmark_adapter import (
    _validate_admitted_spec,
    operation_parity,
)
from napari_vipp.core.compute_policy import (
    MESH_MORPHOLOGY_HOST_FINALIZATION_BYTES_PER_INPUT_ELEMENT,
    MESH_MORPHOLOGY_MAXIMUM_SPATIAL_BLOCK_ELEMENTS,
    ArrayFacts,
    FactCompleteness,
    estimate_candidate_memory,
    evaluate_candidate_workload_support,
    validate_spec_policy_references,
)
from napari_vipp.core.compute_registry import ComputeRegistry
from napari_vipp.core.compute_specs import AdmissionTier, compute_specs_for
from napari_vipp.core.mesh_measurements import (
    MESH_PAYLOAD_HEADER_BYTES,
    mesh_morphology_layout,
)
from napari_vipp.core.node_execution import PreparedNodeCall
from napari_vipp.core.tables import TableData

SOURCE_ROOT = Path(__file__).resolve().parents[2]


def _spec():
    (spec,) = compute_specs_for(
        "measure_3d_mesh_morphology",
        include_cpu=False,
    )
    return spec


def _workload(
    *,
    shape: tuple[int, ...] = (2, 10, 20, 30),
    dtype: str = "int32",
    spatial_ndim: int = 3,
    parameters: tuple[tuple[str, object], ...] = (),
) -> WorkloadDescriptor:
    return WorkloadDescriptor(
        "mesh-node",
        "measure_3d_mesh_morphology",
        (shape,),
        (dtype,),
        parameters=(
            ("spatial_mode", "3D ZYX"),
            ("minimum_voxel_count", 16),
            ("include_convex_hull_metrics", True),
            *parameters,
        ),
        resolved_spatial_ndim=spatial_ndim,
    )


def _facts(
    workload: WorkloadDescriptor,
    *,
    minimum: int = 0,
    maximum: int = 2_000_000_000,
    complete: bool = True,
) -> tuple[ArrayFacts, ...]:
    shape = workload.input_shapes[0]
    elements = int(np.prod(shape, dtype=np.int64))
    return (
        ArrayFacts(
            shape,
            workload.input_dtypes[0],
            elements,
            "mesh-label-revision",
            completeness=(
                FactCompleteness.COMPLETE if complete else FactCompleteness.UNKNOWN
            ),
            finite_count=elements if complete else None,
            minimum=minimum if complete else None,
            maximum=maximum if complete else None,
            guarantees=("nonnegative",) if complete and minimum >= 0 else (),
        ),
    )


def test_mesh_spec_is_public_exact_typed_host_boundary() -> None:
    spec = _spec()

    assert spec.implementation_id == ("cupy-measure-3d-mesh-morphology-hybrid-v1")
    assert spec.callable_ref == (
        "napari_vipp.core.gpu.cupy_mesh_morphology:measure_3d_mesh_morphology"
    )
    assert spec.admission_tier is AdmissionTier.PUBLIC_AUTO_CANDIDATE
    assert spec.implementation_library_id == "cupy"
    assert spec.supported_spatial_ndims == (3,)
    assert spec.supports_device_residency and not spec.host_boundary
    assert spec.input_ports[0].public_dtypes == ("int32",)
    assert spec.output_ports[0].value_kind.value == "table"
    assert spec.output_ports[0].internal_dtypes == ("uint8",)
    assert spec.parity_policy_id == "mesh-morphology-table-exact-v1"
    assert spec.memory_model_id == "cupy-mesh-morphology-packed-memory-v1"
    assert spec.host_finalizer_ref == (
        "napari_vipp.core.mesh_measurements:finalize_mesh_morphology_outputs"
    )
    validate_spec_policy_references(spec)
    assert (
        ComputeRegistry().implementation_spec(
            spec.implementation_id,
            spec.implementation_version,
        )
        == spec
    )


def test_mesh_region_requires_true_3d_native_nonnegative_int32() -> None:
    spec = _spec()
    workload = _workload()

    missing = evaluate_candidate_workload_support(spec, workload)
    supported = evaluate_candidate_workload_support(
        spec,
        workload,
        array_facts=_facts(workload),
    )
    negative = evaluate_candidate_workload_support(
        spec,
        workload,
        array_facts=_facts(workload, minimum=-1),
    )
    nonnative = _workload(dtype=">i4")
    two_dimensional = _workload(shape=(20, 30), spatial_ndim=2)

    assert not missing.supported and missing.requires_complete_facts
    assert supported.supported
    assert not negative.supported and not negative.fallback_allowed
    assert not evaluate_candidate_workload_support(spec, nonnative).supported
    two_d = evaluate_candidate_workload_support(spec, two_dimensional)
    assert not two_d.supported
    assert "dimensionality" in two_d.reason_text


def test_mesh_region_keeps_arbitrary_sparse_positive_int32_ids() -> None:
    spec = _spec()
    workload = _workload()

    decision = evaluate_candidate_workload_support(
        spec,
        workload,
        array_facts=_facts(workload, maximum=np.iinfo(np.int32).max),
    )

    assert decision.supported


@pytest.mark.parametrize("dtype", ("int64", "uint16", "<u4"))
def test_unpromoted_mesh_integer_dtypes_keep_visible_cpu_fallback(dtype: str) -> None:
    decision = evaluate_candidate_workload_support(_spec(), _workload(dtype=dtype))

    assert not decision.supported
    assert decision.fallback_allowed


@pytest.mark.parametrize("dtype", ("bool", "float32", "complex64"))
def test_invalid_mesh_label_domains_do_not_claim_cpu_fallback(dtype: str) -> None:
    decision = evaluate_candidate_workload_support(_spec(), _workload(dtype=dtype))

    assert not decision.supported
    assert not decision.fallback_allowed


def test_mesh_region_rejects_spatial_blocks_outside_compact_index_contract() -> None:
    shape = (MESH_MORPHOLOGY_MAXIMUM_SPATIAL_BLOCK_ELEMENTS, 1, 1)
    workload = _workload(shape=shape)

    decision = evaluate_candidate_workload_support(_spec(), workload)

    assert not decision.supported
    assert decision.fallback_allowed
    assert "fewer than" in decision.reason_text


def test_mesh_memory_model_is_linear_and_covers_three_payload_representations() -> None:
    workload = _workload()
    layout = mesh_morphology_layout(
        workload.input_shapes[0],
        spatial_mode="3D ZYX",
        resolved_spatial_ndim=3,
    )
    estimate = estimate_candidate_memory(_spec(), workload)
    elements = 2 * 10 * 20 * 30
    block_elements = 10 * 20 * 30
    input_bytes = elements * np.dtype(np.int32).itemsize
    payload_bound = (
        MESH_PAYLOAD_HEADER_BYTES
        + elements * layout.record_words * np.dtype(np.uint64).itemsize
        + elements * np.dtype(np.uint32).itemsize
    )
    workspace = input_bytes + block_elements * 128 + 2 * payload_bound

    assert estimate.runtime_managed_peak_bytes == (
        input_bytes + payload_bound + workspace
    )
    assert estimate.runtime_managed_peak_bytes >= 3 * payload_bound
    assert estimate.host_materialization_peak_bytes == (
        payload_bound
        + elements * MESH_MORPHOLOGY_HOST_FINALIZATION_BYTES_PER_INPUT_ELEMENT
    )
    assert estimate.uncertainty_bytes >= 64 * 1024**2

    doubled = estimate_candidate_memory(
        _spec(),
        _workload(shape=(4, 10, 20, 30)),
    )
    assert doubled.runtime_managed_peak_bytes < 2 * estimate.runtime_managed_peak_bytes
    assert doubled.runtime_managed_peak_bytes > estimate.runtime_managed_peak_bytes


def test_mesh_benchmark_parity_is_exact_for_public_float_columns() -> None:
    reference = TableData(
        ("label_id", "mesh_volume_physical", "mesh_error"),
        ((1, 12.5, ""), (2, float("nan"), "failed")),
        name="3D mesh morphology measurements",
        table_kind="3D mesh morphology",
        source_name="fixture",
        column_units=(("label_id", "label"),),
    )
    exact = TableData(
        reference.columns,
        ((1, 12.5, ""), (2, float("nan"), "failed")),
        reference.name,
        reference.table_kind,
        reference.source_name,
        reference.column_units,
    )
    perturbed = replace(
        exact,
        rows=((1, 12.5 + 1e-12, ""), exact.rows[1]),
    )

    assert operation_parity(
        "measure_3d_mesh_morphology",
        reference,
        exact,
    ).passed
    assert not operation_parity(
        "measure_3d_mesh_morphology",
        reference,
        perturbed,
    ).passed


def test_mesh_spec_is_accepted_by_registered_benchmark_validation() -> None:
    labels = np.zeros((5, 6, 7), dtype=np.int32)
    call = PreparedNodeCall(
        "mesh-node",
        "measure_3d_mesh_morphology",
        lambda data, **_kwargs: data,
        (labels,),
        kwargs={"spatial_mode": "3D ZYX", "resolved_spatial_ndim": 3},
    )

    _validate_admitted_spec(
        call,
        _spec(),
        ComputeRegistry(),
        allow_experimental=False,
    )


def test_mesh_declaration_and_provider_import_without_cuda_modules() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(SOURCE_ROOT), environment.get("PYTHONPATH", "")))
    )
    script = r"""
import sys
from napari_vipp.core.compute_specs import compute_specs_for
spec = compute_specs_for('measure_3d_mesh_morphology', include_cpu=False)[0]
assert spec.callable_ref.endswith(':measure_3d_mesh_morphology')
import napari_vipp.core.gpu.cupy_mesh_morphology as provider
assert provider.__all__ == ['measure_3d_mesh_morphology']
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
