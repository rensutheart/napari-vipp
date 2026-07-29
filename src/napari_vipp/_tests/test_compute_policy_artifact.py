from __future__ import annotations

import copy
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
    assert policy.policy_version == 3
    assert policy.content_sha256 == PHASE1_POLICY_SHA256
    assert policy.phase == "phase1"
    assert policy.status == "public-validated"
    assert policy.exposure.global_default == "auto"
    assert policy.exposure.public_controls_enabled
    assert not policy.exposure.developer_enablement_required

    with pytest.raises(FrozenInstanceError):
        policy.policy_version = 3  # type: ignore[misc]


def test_phase1_operation_ids_and_conservative_settings_are_exact():
    policy = load_phase1_compute_policy()
    expected_implementations = {
        "rolling_ball_background": "cucim-rolling_ball_background-v2",
        "subtract_background": "cucim-subtract_background-v2",
        "median_filter": "cupyx-median-filter-v1",
        "gaussian_blur": "cupyx-gaussian-blur-v1",
        "gaussian_blur_3d": "cupyx-gaussian-blur-3d-v1",
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
    assert "rtx40-series-secondary-v1" in policy.auto_selection.pending_evidence
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
    assert platform.driver_versions == ("13030",)
    assert platform.cupy_versions == ("14.1.1",)
    assert platform.cupyx_versions == ("14.1.1",)
    assert platform.runtime_probe_fingerprint_required
    assert platform.driver_version_metadata_required
    assert platform.nvidia_compute_capability_required
    assert platform.nvidia_device_names == ("NVIDIA GeForce RTX 5090",)
    assert platform.nvidia_compute_capabilities == ("12.0",)
    assert platform.cucim_versions == ("26.6.0", "26.06.00")
    assert platform.cucim_environment_record_schema == "napari-vipp-gpu-environment"
    assert platform.cucim_environment_record_schema_version == 1
    assert platform.cucim_environment_track == "cuda13"
    assert platform.cupy_distribution == "cupy-cuda13x"
    assert platform.cucim_distribution == "cucim-cu13"
    assert platform.cucim_artifact_sha256 == (
        "586d3443091eea67ce2c697be2c490ca51977a5dbdf894b9318b270977134cf8"
    )
    assert platform.validated_environment_policy_ids == (
        "cuda-cupy-14.1.1-cpython312-windows-native-v3",
        "cuda-cupy-14.1.1-cucim-26.6.0-cpython312-windows-native-v3",
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


def test_valid_but_changed_v3_record_cannot_be_resigned_in_place():
    changed = copy.deepcopy(_resource_document())
    changed["policy"]["auto_selection"]["non_local_minimum_saving_ms"] = 21.0  # type: ignore[index]
    _resign(changed)

    with pytest.raises(ComputePolicyDigestError, match="immutable Phase 1 record"):
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
