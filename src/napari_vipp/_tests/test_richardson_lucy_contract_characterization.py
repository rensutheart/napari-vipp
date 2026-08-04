from __future__ import annotations

import numpy as np
import pytest

from napari_vipp.core.compute import WorkloadDescriptor
from napari_vipp.core.compute_benchmark_adapter import operation_parity
from napari_vipp.core.compute_policy import estimate_candidate_memory
from napari_vipp.core.compute_specs import compute_specs_for
from napari_vipp.core.richardson_lucy_compute import (
    canonical_richardson_lucy_spec_digest,
)

_SPEC_DIGESTS = {
    "richardson_lucy_deconvolution": (
        "b6e9dd5780ecf1b7e08c7bd287147051b5a6973b25d6833ffa457fb381a73609"
    ),
    "richardson_lucy_tv_deconvolution": (
        "5e206df5c616ad141b8c21e059dc98e9e13c0e37dcc0132690b4b9836edba70a"
    ),
}


def _accelerator_spec(operation_id: str):
    return next(spec for spec in compute_specs_for(operation_id) if spec.is_gpu)


@pytest.mark.parametrize("operation_id", tuple(_SPEC_DIGESTS))
def test_rl_accelerator_spec_serialization_is_frozen(operation_id: str) -> None:
    assert (
        canonical_richardson_lucy_spec_digest(_accelerator_spec(operation_id))
        == _SPEC_DIGESTS[operation_id]
    )


@pytest.mark.parametrize(
    (
        "case_id",
        "operation_id",
        "image_shape",
        "psf_shape",
        "spatial_ndim",
        "parameters",
        "runtime_peak",
        "host_peak",
    ),
    (
        (
            "rl-2d",
            "richardson_lucy_deconvolution",
            (256, 256),
            (15, 15),
            2,
            {"spatial_mode": "2D YX", "iterations": 25, "filter_epsilon": 1e-8},
            5_615_972,
            262_144,
        ),
        (
            "rl-3d",
            "richardson_lucy_deconvolution",
            (32, 128, 128),
            (9, 15, 15),
            3,
            {"spatial_mode": "3D ZYX", "iterations": 25, "filter_epsilon": 1e-8},
            56_953_396,
            2_097_152,
        ),
        (
            "rl-tv-zero-2d",
            "richardson_lucy_tv_deconvolution",
            (256, 256),
            (15, 15),
            2,
            {
                "spatial_mode": "2D YX",
                "iterations": 25,
                "tv_regularization": 0.0,
                "filter_epsilon": 1e-8,
            },
            5_615_972,
            262_144,
        ),
        (
            "rl-tv-positive-3d",
            "richardson_lucy_tv_deconvolution",
            (32, 128, 128),
            (9, 15, 15),
            3,
            {
                "spatial_mode": "3D ZYX",
                "iterations": 25,
                "tv_regularization": 0.002,
                "tv_epsilon": 1e-6,
                "filter_epsilon": 1e-12,
                "denominator_floor": 0.05,
            },
            84_216_372,
            2_097_152,
        ),
    ),
)
def test_rl_memory_model_is_frozen(
    case_id: str,
    operation_id: str,
    image_shape: tuple[int, ...],
    psf_shape: tuple[int, ...],
    spatial_ndim: int,
    parameters: dict[str, object],
    runtime_peak: int,
    host_peak: int,
) -> None:
    complete_parameters = {
        "normalize_psf": True,
        "clip_negative_input": True,
        "clip_output_negative": True,
        "preserve_input_scale": True,
        **parameters,
    }
    workload = WorkloadDescriptor(
        node_id=case_id,
        operation_id=operation_id,
        input_shapes=(image_shape, psf_shape),
        input_dtypes=("float32", "float32"),
        parameters=tuple(sorted(complete_parameters.items())),
        resolved_spatial_ndim=spatial_ndim,
    )

    estimate = estimate_candidate_memory(_accelerator_spec(operation_id), workload)

    assert estimate.runtime_managed_peak_bytes == runtime_peak
    assert estimate.total_device_peak_bytes == runtime_peak
    assert estimate.host_materialization_peak_bytes == host_peak
    assert estimate.uncertainty_bytes == 32 * 1024**2


def test_rl_parity_profiles_are_frozen() -> None:
    reference = np.array([[0.0, 1.0], [2.0, 4.0]], dtype=np.float32)
    strict_candidate = reference.copy()
    strict_candidate[-1, -1] += np.float32(1e-6)
    positive_tv_candidate = reference.copy()
    positive_tv_candidate[-1, -1] += np.float32(1e-3)

    ordinary = operation_parity(
        "richardson_lucy_deconvolution",
        reference,
        strict_candidate,
    )
    lambda_zero = operation_parity(
        "richardson_lucy_tv_deconvolution",
        reference,
        strict_candidate,
        parameters={"tv_regularization": 0.0},
    )
    positive_tv = operation_parity(
        "richardson_lucy_tv_deconvolution",
        reference,
        positive_tv_candidate,
        parameters={"tv_regularization": 0.002},
    )

    assert ordinary.passed is True
    assert ordinary.detail == (
        "nrmse=2.08108797e-07 (limit=2e-06); "
        "max_abs=9.53674316e-07 (limit=2.1e-05); max_ulp=2 (diagnostic)"
    )
    assert lambda_zero == ordinary
    assert positive_tv.passed is True
    assert positive_tv.detail == (
        "nrmse=0.000218202074 (limit=0.005); "
        "max_abs=0.000999927521 (limit=0.020001); "
        "max_ulp=2097 (diagnostic)"
    )
