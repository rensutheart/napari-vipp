from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from dataclasses import FrozenInstanceError
from importlib import resources
from pathlib import Path

import pytest

from napari_vipp.core.compute_policy_artifact import (
    PHASE1_POLICY_ID,
    PHASE1_POLICY_RESOURCE,
    PHASE1_POLICY_SHA256,
    ComputePolicyArtifactError,
    ComputePolicyDigestError,
    canonical_artifact_digest,
    load_phase1_compute_policy,
    parse_compute_policy_artifact,
)
from napari_vipp.core.compute_specs import accelerator_compute_specs


def _resource_document() -> dict[str, object]:
    resource = resources.files("napari_vipp.compute_policies").joinpath(
        PHASE1_POLICY_RESOURCE
    )
    assert resource.is_file()
    return json.loads(resource.read_text(encoding="utf-8"))


def _encoded(document: dict[str, object]) -> bytes:
    return json.dumps(document, ensure_ascii=False).encode("utf-8")


def _resign(document: dict[str, object]) -> None:
    document["content_sha256"] = canonical_artifact_digest(document)


def test_loads_versioned_policy_through_installed_package_resources():
    policy = load_phase1_compute_policy()

    assert policy.policy_id == PHASE1_POLICY_ID
    assert policy.policy_version == 8
    assert policy.content_sha256 == PHASE1_POLICY_SHA256
    assert policy.phase == "phase1"
    assert policy.status == "public-validated"
    assert policy.exposure.global_default == "auto"
    assert policy.exposure.public_controls_enabled
    assert not policy.exposure.developer_enablement_required

    with pytest.raises(FrozenInstanceError):
        policy.policy_version = 8  # type: ignore[misc]


def test_phase1_operation_ids_and_conservative_settings_are_exact():
    policy = load_phase1_compute_policy()
    expected_implementations = {
        "rolling_ball_background": "cucim-rolling_ball_background-v2",
        "subtract_background": "cucim-subtract_background-v2",
        "median_filter": "cupyx-median-filter-v1",
        "gaussian_blur": "cupyx-gaussian-blur-v1",
        "gaussian_blur_3d": "cupyx-gaussian-blur-3d-v1",
        "sigma_filter": "cupy-sigma-filter-v1",
        "label_connected_components": "cupyx-connected-components-v1",
        "measure_objects": "cucim-measure-objects-basic-v1",
        "measure_objects_intensity": "cucim-measure-objects-intensity-basic-v1",
    }

    assert {
        operation.operation_id: operation.implementation_id
        for operation in policy.operations
    } == expected_implementations
    assert {operation.admission_tier for operation in policy.operations} == {
        "public_auto_candidate"
    }
    assert not policy.auto_selection.broad_calibration_enabled
    assert policy.auto_selection.evidence_scope == "local-or-reviewed-segment-only-v1"
    assert policy.auto_selection.non_local_lower_confidence_speedup == 1.20
    assert policy.auto_selection.non_local_minimum_saving_ms == 20.0
    assert policy.auto_selection.local_noise_relative_fraction == 0.05
    assert policy.auto_selection.local_noise_absolute_ms == 10.0
    assert policy.auto_selection.tie_breaker == "retain-current-else-cpu-v1"
    assert "rtx40-series-secondary-v1" not in policy.auto_selection.pending_evidence
    assert (
        "multi-architecture-reproducibility-v1"
        in policy.auto_selection.pending_evidence
    )
    assert policy.local_benchmark.initial_warm_rounds == 7
    assert policy.local_benchmark.adaptive_warm_rounds == (15, 21)
    assert policy.local_benchmark.confidence_level == 0.95
    assert (
        policy.local_benchmark.confidence_method == "paired-bootstrap-median-ratio-v1"
    )
    assert policy.local_benchmark.bootstrap_resamples == 2_000
    assert policy.local_benchmark.bootstrap_seed == 17_029
    assert policy.local_benchmark.outlier_policy == "retain-all-finite-paired-rounds-v1"
    assert (
        policy.local_benchmark.round_order_policy
        == "randomized-paired-candidate-order-v1"
    )
    assert policy.local_benchmark.adaptive_near_threshold_relative_fraction == 0.05
    assert policy.local_benchmark.adaptive_near_threshold_absolute_ms == 5.0
    assert policy.local_benchmark.adaptive_speedup_mad_fraction == 0.05
    platform = policy.platform_admission
    assert platform.operating_systems == ("Windows",)
    assert platform.execution_modes == ("native",)
    assert policy.platform_admission.python_implementation == "CPython"
    assert policy.platform_admission.python_minor_versions == ("3.12",)
    assert platform.python_abis == ("cpython-312",)
    assert platform.numpy_versions == ("2.5.1",)
    assert platform.scipy_versions == ("1.18.0",)
    assert platform.scikit_image_versions == ("0.26.0",)
    assert policy.platform_admission.cuda_major_versions == (13,)
    assert platform.cuda_runtime_versions == ("13020",)
    assert platform.minimum_driver_version == "13030"
    assert platform.cupy_versions == ("14.1.1",)
    assert platform.cupyx_versions == ("14.1.1",)
    assert platform.runtime_probe_fingerprint_required
    assert platform.driver_version_metadata_required
    assert platform.nvidia_compute_capability_required
    assert platform.nvidia_device_class == "nvidia-cuda"
    assert platform.minimum_nvidia_compute_capability == "7.5"
    assert platform.reference_validation_devices == (
        "NVIDIA GeForce RTX 5090 / compute capability 12.0",
        "NVIDIA GeForce RTX 4050 Laptop GPU / compute capability 8.9",
    )
    assert platform.cucim_versions == ("26.6.0", "26.06.00")
    assert platform.cucim_environment_record_schema == "napari-vipp-gpu-environment"
    assert platform.cucim_environment_record_schema_version == 2
    assert platform.cucim_environment_track == "cuda13"
    assert platform.cupy_distribution == "cupy-cuda13x"
    assert platform.cucim_distribution == "cucim-cu13"
    assert platform.cucim_wheel_payload_sha256 == (
        "d640d1e17bcce15d32d03841997252bf915b63da855e406c35f0d70c5a5ea667"
    )
    assert platform.cucim_source_tag == "v26.06.00"
    assert platform.cucim_source_commit == (
        "3c15781c207eab93a317dd9803a6e726fe01f7c4"
    )
    assert platform.cucim_build_recipe_id == "napari-vipp-cucim-windows-v1"
    assert platform.validated_environment_policy_ids == (
        "cuda-cupy-14.1.1-cpython312-windows-native-v3",
        "cuda-cupy-14.1.1-cucim-26.6.0-cpython312-windows-native-v4",
        "cuda-cupy-14.1.1-rawkernel-cpython312-windows-native-v1",
    )
    assert platform.linux_policy == "pending-native-clean-host-evidence-v1"
    assert platform.macos_policy == "cpu-only-provider-investigation-pending-v1"
    assert policy.platform_admission.public_advertisement_enabled


def test_scientific_summary_mirrors_executable_declaration_ids_and_bounds():
    policy = load_phase1_compute_policy()
    executable = {spec.implementation_id: spec for spec in accelerator_compute_specs()}

    phase1_ids = {operation.implementation_id for operation in policy.operations}
    # The signed Phase-1 record is immutable. Later-phase declarations remain
    # executable records in the live catalog without rewriting its audit
    # history, so this artifact mirrors exactly its own declared subset.
    assert phase1_ids <= set(executable)
    for operation in policy.operations:
        spec = executable[operation.implementation_id]
        assert operation.operation_id == spec.operation_id
        assert operation.implementation_version == spec.implementation_version
        assert operation.runtime_id == spec.runtime_id
        assert operation.implementation_library_id == spec.implementation_library_id
        assert operation.environment_policy_id == spec.validated_environment_policy_id
        assert operation.parameter_policy_id == spec.parameter_policy_id
        assert operation.workload_policy_id == spec.workload_policy_id
        assert operation.parity_policy_id == spec.parity_policy_id
        assert operation.memory_model_id == spec.memory_model_id
        assert operation.supported_spatial_ndims == spec.supported_spatial_ndims
        assert operation.supports_device_residency == spec.supports_device_residency
        assert operation.host_finalizer_ref == spec.host_finalizer_ref
        assert (
            operation.support_summary.public_dtypes == spec.input_ports[0].public_dtypes
        )
        assert (
            operation.support_summary.authority
            == "audit-mirror-of-executable-policy-v1"
        )

    background_bounds = {
        bound.scope: (bound.minimum, bound.maximum)
        for bound in policy.operation(
            "rolling_ball_background"
        ).support_summary.parameter_bounds
    }
    assert background_bounds == {
        "spatial-ndim-2": (1.0, 500.0),
        "spatial-ndim-3": (1.0, 50.0),
    }
    median_bound = policy.operation("median_filter").support_summary.parameter_bounds[0]
    assert (median_bound.parameter, median_bound.minimum, median_bound.maximum) == (
        "size",
        1.0,
        51.0,
    )
    for operation_id in ("gaussian_blur", "gaussian_blur_3d"):
        assert {
            (bound.minimum, bound.maximum)
            for bound in policy.operation(operation_id).support_summary.parameter_bounds
        } == {(0.0, 12.0)}

    sigma = policy.operation("sigma_filter")
    assert sigma.admission_tier == "public_auto_candidate"
    assert sigma.implementation_library_id == "cupy"
    assert sigma.environment_policy_id == (
        "cuda-cupy-14.1.1-rawkernel-cpython312-windows-native-v1"
    )
    assert sigma.support_summary.spatial_semantics_id == (
        "sigma-slicewise-active-yx-nearest-v1"
    )
    assert "float32-abs-at-most-f32-sqrt-f32max-v1" in (
        sigma.support_summary.required_facts
    )
    assert "native-endian-dtype-descriptor-v1" in (sigma.support_summary.required_facts)
    assert "non-native-endian-v1" in sigma.support_summary.explicit_cpu_regions
    assert {
        bound.parameter: (bound.minimum, bound.maximum)
        for bound in sigma.support_summary.parameter_bounds
    } == {
        "radius": (0.5, 10.0),
        "sigma_width": (0.0, float.fromhex("0x1.fffffffffffffp+1023")),
        "minimum_pixel_fraction": (0.0, 1.0),
    }

    connected = policy.operation("label_connected_components")
    assert connected.admission_tier == "public_auto_candidate"
    assert connected.implementation_version == "1"
    assert connected.runtime_id == "cuda-cupy"
    assert connected.implementation_library_id == "cupyx"
    assert connected.environment_policy_id == (
        "cuda-cupy-14.1.1-cpython312-windows-native-v3"
    )
    assert connected.parameter_policy_id == "connected-components-parameters-v1"
    assert connected.workload_policy_id == "connected-components-bool-2d-3d-v1"
    assert connected.parity_policy_id == "labels-bitwise-int32-v1"
    assert connected.memory_model_id == "cupyx-connected-components-memory-v1"
    assert connected.supported_spatial_ndims == (2, 3)
    assert connected.supports_device_residency
    assert connected.support_summary.public_dtypes == ("bool",)
    assert connected.support_summary.spatial_semantics_id == (
        "connected-components-independent-leading-blocks-2d-3d-v1"
    )
    assert connected.support_summary.required_facts == (
        "boolean-mask-input-v1",
        "scipy-face-or-full-connectivity-v1",
        "independent-leading-block-labeling-v1",
        "exact-int32-label-id-order-v1",
        "spatial-block-elements-under-2147483646-v1",
    )
    assert connected.support_summary.explicit_cpu_regions == (
        "numeric-nonzero-mask-input-v1",
        "resolved-spatial-rank-one-v1",
        "spatial-block-at-least-2147483646-v1",
    )
    connected_bound = connected.support_summary.parameter_bounds[0]
    assert (
        connected_bound.parameter,
        connected_bound.minimum,
        connected_bound.maximum,
        connected_bound.scope,
        connected_bound.canonicalization_policy_id,
    ) == (
        "resolved_spatial_ndim",
        2.0,
        3.0,
        "derived-spatial-rank",
        "axis-metadata-or-explicit-mode-v1",
    )

    measurements = policy.operation("measure_objects")
    measurements_intensity = policy.operation("measure_objects_intensity")
    for operation in (measurements, measurements_intensity):
        assert operation.admission_tier == "public_auto_candidate"
        assert operation.implementation_version == "1"
        assert operation.runtime_id == "cuda-cupy"
        assert operation.implementation_library_id == "cucim"
        assert operation.environment_policy_id == (
            "cuda-cupy-14.1.1-cucim-26.6.0-cpython312-windows-native-v4"
        )
        assert operation.parameter_policy_id == "basic-measurements-parameters-v1"
        assert operation.parity_policy_id == "basic-measurement-table-v1"
        assert operation.memory_model_id == "cucim-basic-measurements-memory-v1"
        assert operation.supported_spatial_ndims == (2, 3)
        assert operation.supports_device_residency
        assert operation.host_finalizer_ref == (
            "napari_vipp.core.measurements:finalize_basic_measurement_outputs"
        )
        assert operation.support_summary.public_dtypes == ("int32",)
        assert operation.support_summary.spatial_semantics_id == (
            "measurement-independent-leading-blocks-typed-table-v1"
        )
        assert "native-int32-nonnegative-labels-v1" in (
            operation.support_summary.required_facts
        )
        assert "typed-host-table-finalizer-v1" in (
            operation.support_summary.required_facts
        )
        assert "extended-measurement-columns-v1" in (
            operation.support_summary.explicit_cpu_regions
        )
        assert {
            bound.parameter: (bound.minimum, bound.maximum)
            for bound in operation.support_summary.parameter_bounds
        } == {
            "resolved_spatial_ndim": (2.0, 3.0),
            "include_shape_descriptors": (0.0, 0.0),
            "include_axis_descriptors": (0.0, 0.0),
            "include_2d_boundary_descriptors": (0.0, 0.0),
            "include_derived_shape_ratios": (0.0, 0.0),
            "include_2d_shape_moments": (0.0, 0.0),
        }

    assert measurements.workload_policy_id == "measurements-int32-basic-2d-3d-v1"
    assert measurements_intensity.workload_policy_id == (
        "measurements-int32-bool-u8-u16-finite-f32-basic-2d-3d-v1"
    )
    assert "finite-bool-u8-u16-f32-intensity-v1" in (
        measurements_intensity.support_summary.required_facts
    )
    assert "nonfinite-float32-intensity-v1" in (
        measurements_intensity.support_summary.explicit_cpu_regions
    )


def test_v8_broadens_hardware_and_legacy_resources_are_byte_stable():
    package = resources.files("napari_vipp.compute_policies")
    historical_sha256 = {
        "phase1-gpu-developer-v1.json": (
            "d0adace0dcd12fdf20553ab848e1abecca9a7947b44e95433b2a6becd56e82c6"
        ),
        "phase1-gpu-developer-v2.json": (
            "d708845da3adda7ebb1da36c5345bb092574617770808ff8241380ea9fca71d5"
        ),
        "phase1-gpu-public-v3.json": (
            "b6696c5f50121831711dbeba50f61585df8cbc32ec22f3e35a439f7e536f4968"
        ),
        "phase1-gpu-public-v4.json": (
            "37198cf8f22950ab824ccbcd3fd91a10bfd07ac1d31f1da5ac142a8a031ea333"
        ),
        "phase1-gpu-public-v5.json": (
            "d4bbd6728b3fe7028942d83efe4c2e6a64b052b1e6276eae6fdeaefeaa985070"
        ),
        "phase1-gpu-public-v6.json": (
            "e7822c096d2f4efeaaa745bfce1ae75c94f966af7582581f68a82586bc56139d"
        ),
        "phase1-gpu-public-v7.json": (
            "c921c49bc1993dabb71c440af8f08e739bf2d7a8808e0a0a6975082a23530877"
        ),
    }
    for name, expected_digest in historical_sha256.items():
        resource = package.joinpath(name)
        assert resource.is_file()
        assert hashlib.sha256(resource.read_bytes()).hexdigest() == expected_digest

    v3 = json.loads(package.joinpath("phase1-gpu-public-v3.json").read_bytes())
    v4 = json.loads(package.joinpath("phase1-gpu-public-v4.json").read_bytes())
    v5 = json.loads(package.joinpath("phase1-gpu-public-v5.json").read_bytes())
    v6 = json.loads(package.joinpath("phase1-gpu-public-v6.json").read_bytes())
    v7 = json.loads(package.joinpath("phase1-gpu-public-v7.json").read_bytes())
    v8 = _resource_document()
    assert v8["policy"]["operations"] == v7["policy"]["operations"]
    v8_platform = v8["policy"]["platform_admission"]
    assert v8_platform["minimum_driver_version"] == "13030"
    assert v8_platform["nvidia_device_class"] == "nvidia-cuda"
    assert v8_platform["minimum_nvidia_compute_capability"] == "7.5"
    assert len(v8_platform["reference_validation_devices"]) == 2
    assert v4["policy"]["operations"][:-1] == v3["policy"]["operations"]
    sigma = v4["policy"]["operations"][-1]
    assert sigma["operation_id"] == "sigma_filter"
    assert sigma["implementation_id"] == "cupy-sigma-filter-v1"
    assert v5["policy"]["operations"][:-1] == v4["policy"]["operations"]
    connected = v5["policy"]["operations"][-1]  # type: ignore[index]
    assert connected["operation_id"] == "label_connected_components"
    assert connected["implementation_id"] == "cupyx-connected-components-v1"

    assert v6["policy"]["operations"][:-2] == v5["policy"]["operations"]  # type: ignore[index]
    assert all(
        "host_finalizer_ref" not in operation
        for operation in v6["policy"]["operations"][:-2]  # type: ignore[index]
    )
    measurements = v6["policy"]["operations"][-2:]  # type: ignore[index]
    assert [operation["operation_id"] for operation in measurements] == [
        "measure_objects",
        "measure_objects_intensity",
    ]
    assert all(operation["host_finalizer_ref"] for operation in measurements)

    v7_as_v6 = json.loads(
        json.dumps(v7).replace(
            "cuda-cupy-14.1.1-cucim-26.6.0-cpython312-windows-native-v4",
            "cuda-cupy-14.1.1-cucim-26.6.0-cpython312-windows-native-v3",
        )
    )
    v7_as_v6["policy_id"] = v6["policy_id"]
    v7_as_v6["policy_version"] = v6["policy_version"]
    v7_as_v6["content_sha256"] = v6["content_sha256"]
    v7_platform = v7_as_v6["policy"]["platform_admission"]
    v6_platform = v6["policy"]["platform_admission"]
    v7_platform["cucim_environment_record_schema_version"] = 1
    del v7_platform["cucim_wheel_payload_sha256"]
    del v7_platform["cucim_source_tag"]
    del v7_platform["cucim_source_commit"]
    del v7_platform["cucim_build_recipe_id"]
    v7_platform["cucim_artifact_sha256"] = v6_platform[
        "cucim_artifact_sha256"
    ]
    assert v7_as_v6 == v6

    v6_as_v5 = copy.deepcopy(v6)
    v6_as_v5["policy_id"] = v5["policy_id"]
    v6_as_v5["policy_version"] = v5["policy_version"]
    v6_as_v5["content_sha256"] = v5["content_sha256"]
    del v6_as_v5["policy"]["operations"][-2:]  # type: ignore[index]
    assert v6_as_v5 == v5

    v5_as_v4 = copy.deepcopy(v5)
    v5_as_v4["policy_id"] = v4["policy_id"]
    v5_as_v4["policy_version"] = v4["policy_version"]
    v5_as_v4["content_sha256"] = v4["content_sha256"]
    v5_as_v4["policy"]["operations"].pop()  # type: ignore[index]
    assert v5_as_v4 == v4

    v3_platform = copy.deepcopy(v3["policy"]["platform_admission"])
    v4_platform = copy.deepcopy(v4["policy"]["platform_admission"])
    rawkernel_policy = v4_platform["validated_environment_policy_ids"].pop()
    assert rawkernel_policy == (
        "cuda-cupy-14.1.1-rawkernel-cpython312-windows-native-v1"
    )
    assert v4_platform == v3_platform


def test_digest_is_canonical_stable_and_detects_tampering():
    document = _resource_document()

    assert canonical_artifact_digest(document) == PHASE1_POLICY_SHA256
    reordered = {key: document[key] for key in reversed(tuple(document))}
    assert canonical_artifact_digest(reordered) == PHASE1_POLICY_SHA256

    tampered = copy.deepcopy(document)
    tampered["policy"]["auto_selection"]["non_local_minimum_saving_ms"] = 21.0  # type: ignore[index]
    with pytest.raises(ComputePolicyDigestError, match="damaged or tampered"):
        parse_compute_policy_artifact(_encoded(tampered))


def test_strict_schema_rejects_invalid_resigned_content():
    invalid = copy.deepcopy(_resource_document())
    invalid["policy"]["unexpected"] = True  # type: ignore[index]
    _resign(invalid)

    with pytest.raises(ComputePolicyArtifactError, match="invalid fields"):
        parse_compute_policy_artifact(_encoded(invalid))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("cucim_wheel_payload_sha256", "not-a-digest", "payload_sha256"),
        ("cucim_source_commit", "abc123", "full Git commit"),
        ("minimum_driver_version", "13.3", "positive decimal string"),
        (
            "minimum_nvidia_compute_capability",
            "Turing",
            "numeric compute capability",
        ),
        (
            "minimum_nvidia_compute_capability",
            "0.0",
            "numeric compute capability",
        ),
    ),
)
def test_strict_schema_rejects_malformed_platform_provenance(
    field,
    value,
    message,
):
    invalid = copy.deepcopy(_resource_document())
    invalid["policy"]["platform_admission"][field] = value  # type: ignore[index]
    _resign(invalid)

    with pytest.raises(ComputePolicyArtifactError, match=message):
        parse_compute_policy_artifact(_encoded(invalid))


def test_loader_accepts_only_the_pinned_v8_identity_and_version():
    package = resources.files("napari_vipp.compute_policies")
    signed_v6 = package.joinpath("phase1-gpu-public-v6.json").read_bytes()
    with pytest.raises(ComputePolicyArtifactError, match="Unsupported Phase 1 policy"):
        parse_compute_policy_artifact(signed_v6)

    wrong_version = copy.deepcopy(_resource_document())
    wrong_version["policy_version"] = 9
    _resign(wrong_version)
    with pytest.raises(
        ComputePolicyArtifactError,
        match="Unsupported Phase 1 policy version 9",
    ):
        parse_compute_policy_artifact(_encoded(wrong_version))


def test_valid_but_changed_v8_record_cannot_be_resigned_in_place():
    changed = copy.deepcopy(_resource_document())
    changed["policy"]["auto_selection"]["non_local_minimum_saving_ms"] = 21.0  # type: ignore[index]
    _resign(changed)

    with pytest.raises(ComputePolicyDigestError, match="immutable Phase 1 record"):
        parse_compute_policy_artifact(_encoded(changed))


@pytest.mark.parametrize(
    "invalid_reference",
    ("", "missing_separator", ":attribute", "module:", "module:a:b"),
)
def test_host_finalizer_reference_is_nonempty_and_well_formed(invalid_reference):
    changed = copy.deepcopy(_resource_document())
    changed["policy"]["operations"][-1]["host_finalizer_ref"] = invalid_reference  # type: ignore[index]
    _resign(changed)

    with pytest.raises(ComputePolicyArtifactError, match="host_finalizer_ref"):
        parse_compute_policy_artifact(_encoded(changed))


def test_host_finalizer_requires_a_resident_device_runtime():
    changed = copy.deepcopy(_resource_document())
    changed["policy"]["operations"][-1]["supports_device_residency"] = False  # type: ignore[index]
    _resign(changed)

    with pytest.raises(ComputePolicyArtifactError, match="resident device runtime"):
        parse_compute_policy_artifact(_encoded(changed))


def test_policy_resource_loader_imports_no_optional_gpu_or_qt_package():
    source_root = Path(__file__).resolve().parents[2]
    script = f"""
import sys
sys.path.insert(0, {str(source_root)!r})
from napari_vipp.core.compute_policy_artifact import load_phase1_compute_policy
load_phase1_compute_policy()
for name in ('cupy', 'cupyx', 'cucim', 'qtpy', 'napari'):
    assert name not in sys.modules, name
"""

    completed = subprocess.run(
        [sys.executable, "-S", "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
