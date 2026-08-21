from __future__ import annotations

import pytest

from napari_vipp.core.compute import (
    ComputeEnvironment,
    ComputeRequest,
    DecisionReason,
    WorkloadDescriptor,
)
from napari_vipp.core.compute_planning import plan_compute_decisions
from napari_vipp.core.compute_policy import (
    PHASE1_CUCIM_BUILD_RECIPE_ID,
    PHASE1_CUCIM_SOURCE_COMMIT,
    PHASE1_CUCIM_SOURCE_TAG,
    PHASE1_CUCIM_WHEEL_PAYLOAD_SHA256,
    evaluate_candidate_support,
)
from napari_vipp.core.compute_specs import compute_specs_for


def _rtx4050_environment(
    *,
    implementation_libraries: tuple[str, ...] = ("cpu", "cupy"),
    runtime_versions: tuple[tuple[str, str], ...] = (
        ("cuda-cupy", "14.1.1"),
        ("cupy", "14.1.1"),
    ),
    implementation_library_metadata: tuple[
        tuple[str, tuple[tuple[str, str], ...]], ...
    ] = (),
) -> ComputeEnvironment:
    return ComputeEnvironment(
        os_name="Windows",
        execution_mode="native",
        python_implementation="CPython",
        python_version="3.12",
        python_abi="cpython-312",
        runtime_ids=("cpu-numpy", "cuda-cupy"),
        implementation_libraries=implementation_libraries,
        runtime_versions=runtime_versions,
        scientific_stack_versions=(
            ("numpy", "2.5.1"),
            ("scipy", "1.18.0"),
            ("scikit-image", "0.26.0"),
        ),
        runtime_probe_fingerprints=(("cuda-cupy", "rtx4050-probe"),),
        runtime_metadata=(
            (
                "cuda-cupy",
                (
                    ("cuda_runtime_version", "13020"),
                    ("driver_version", "13030"),
                ),
            ),
        ),
        implementation_library_metadata=implementation_library_metadata,
        driver_version="13030",
        device_id="cuda:0",
        device_name="NVIDIA GeForce RTX 4050 Laptop GPU",
        device_class="nvidia-cuda",
        device_metadata=(("compute_capability", "8.9"),),
        memory_topology="discrete",
        total_accelerator_memory_bytes=6 * 1024**3,
        probe_status="available",
    )


def _gaussian_workload() -> WorkloadDescriptor:
    return WorkloadDescriptor(
        node_id="gaussian",
        operation_id="gaussian_blur",
        input_shapes=((31, 37),),
        input_dtypes=("float32",),
        parameters=(("sigma", 1.0),),
        resolved_spatial_ndim=2,
    )


@pytest.mark.parametrize(
    "runtime_versions",
    (
        (("cuda-cupy", "14.0.0"), ("cupy", "14.1.1")),
        (("cuda-cupy", "14.1.1"), ("cupy", "14.0.0")),
    ),
    ids=("cupy-runtime-version", "cupy-library-version"),
)
def test_rtx4050_admission_rejects_mismatched_cupy_provenance(runtime_versions):
    spec = compute_specs_for("gaussian_blur", include_cpu=False)[0]

    decision = evaluate_candidate_support(
        spec,
        _gaussian_workload(),
        _rtx4050_environment(runtime_versions=runtime_versions),
        allow_experimental=False,
    )

    assert not decision.supported
    assert decision.reason is DecisionReason.ENVIRONMENT_UNSUPPORTED
    assert "14.1.1 provenance" in decision.reason_text


def test_rtx4050_admission_rejects_bad_cucim_build_provenance():
    bad_payload_digest = (
        "0" * 64 if PHASE1_CUCIM_WHEEL_PAYLOAD_SHA256 != "0" * 64 else "1" * 64
    )
    metadata = (
        ("environment_record_schema", "napari-vipp-gpu-environment"),
        ("environment_record_schema_version", "2"),
        ("environment_track", "cuda13"),
        ("cupy_distribution", "cupy-cuda13x"),
        ("cucim_distribution", "cucim-cu13"),
        ("cucim_distribution_version", "26.6.0"),
        ("cucim_artifact_sha256", "a" * 64),
        ("cucim_wheel_payload_sha256", bad_payload_digest),
        ("cucim_source_tag", PHASE1_CUCIM_SOURCE_TAG),
        ("cucim_source_commit", PHASE1_CUCIM_SOURCE_COMMIT),
        ("cucim_build_recipe_id", PHASE1_CUCIM_BUILD_RECIPE_ID),
    )
    environment = _rtx4050_environment(
        implementation_libraries=("cpu", "cucim"),
        runtime_versions=(("cuda-cupy", "14.1.1"), ("cucim", "26.6.0")),
        implementation_library_metadata=(("cucim", metadata),),
    )
    spec = compute_specs_for(
        "measure_objects",
        include_cpu=False,
        allow_experimental=True,
    )[0]
    workload = WorkloadDescriptor(
        node_id="measurements",
        operation_id="measure_objects",
        input_shapes=((31, 37),),
        input_dtypes=("int32",),
        resolved_spatial_ndim=2,
    )

    decision = evaluate_candidate_support(
        spec,
        workload,
        environment,
        allow_experimental=True,
    )

    assert not decision.supported
    assert decision.reason is DecisionReason.ENVIRONMENT_UNSUPPORTED
    assert "cucim_wheel_payload_sha256" in decision.reason_text
    assert "provenance" in decision.reason_text.lower()


@pytest.mark.parametrize(
    "compute_request",
    (
        ComputeRequest(mode="cpu"),
        ComputeRequest(
            mode="custom",
            node_preferences={"gaussian": "cpu"},
        ),
    ),
    ids=("global-cpu", "custom-node-cpu"),
)
def test_rtx4050_broad_admission_does_not_override_explicit_cpu(compute_request):
    environment = _rtx4050_environment()

    result = plan_compute_decisions(
        compute_request,
        (_gaussian_workload(),),
        environment=environment,
    )

    decision = result.decisions[0]
    assert decision.runtime_id == "cpu-numpy"
    assert decision.reason is DecisionReason.EXPLICIT_CPU
    assert not decision.fallback_used
    assert result.environment is environment
