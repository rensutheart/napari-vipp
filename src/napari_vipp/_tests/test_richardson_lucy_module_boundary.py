from __future__ import annotations

import ast
import importlib
from pathlib import Path

import numpy as np

from napari_vipp.core import compute_benchmark_adapter as benchmark_adapter
from napari_vipp.core import compute_contracts, compute_specs, operations
from napari_vipp.core.richardson_lucy_compute import (
    richardson_lucy_compute_specs,
)
from napari_vipp.core.richardson_lucy_parity import (
    RICHARDSON_LUCY_FLOAT32_NRMSE_LIMIT,
    RICHARDSON_LUCY_PARITY_OPERATION_IDS,
    richardson_lucy_float32_parity,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_operations_public_exports_are_the_extracted_callables() -> None:
    module = importlib.import_module("napari_vipp.core.richardson_lucy")

    assert operations.richardson_lucy_deconvolution is (
        module.richardson_lucy_deconvolution
    )
    assert operations.richardson_lucy_tv_deconvolution is (
        module.richardson_lucy_tv_deconvolution
    )
    assert module.__all__ == [
        "richardson_lucy_deconvolution",
        "richardson_lucy_tv_deconvolution",
    ]


def test_rl_modules_do_not_depend_on_operations_privates() -> None:
    relative_paths = (
        "src/napari_vipp/core/richardson_lucy.py",
        "src/napari_vipp/core/richardson_lucy_compute.py",
        "src/napari_vipp/core/richardson_lucy_parity.py",
        "src/napari_vipp/core/gpu/cupy_rl.py",
        "src/napari_vipp/core/gpu/cupy_rl_tv.py",
    )

    for relative_path in relative_paths:
        path = PROJECT_ROOT / relative_path
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        operations_imports = [
            node
            for node in ast.walk(tree)
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "napari_vipp.core.operations"
            )
            or (
                isinstance(node, ast.Import)
                and any(
                    alias.name == "napari_vipp.core.operations" for alias in node.names
                )
            )
        ]
        assert operations_imports == [], path


def test_operations_boundary_reexports_only_public_rl_names() -> None:
    path = PROJECT_ROOT / "src/napari_vipp/core/operations.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "napari_vipp.core.richardson_lucy"
    ]

    aliases = [alias for node in imports for alias in node.names]
    assert [alias.name for alias in aliases] == [
        "richardson_lucy_deconvolution",
        "richardson_lucy_tv_deconvolution",
    ]
    assert all(alias.asname == alias.name for alias in aliases)
    assert all(not alias.name.startswith("_") for alias in aliases)


def test_compute_specs_reexport_foundational_contract_types() -> None:
    assert compute_specs.AdmissionTier is compute_contracts.AdmissionTier
    assert compute_specs.ComputePortContract is compute_contracts.ComputePortContract
    assert compute_specs.OperationComputeSpec is compute_contracts.OperationComputeSpec
    assert compute_specs.ValueKind is compute_contracts.ValueKind


def test_builtin_registry_uses_operation_owned_rl_specs() -> None:
    expected = richardson_lucy_compute_specs()
    actual = tuple(
        spec
        for operation_id in (
            "richardson_lucy_deconvolution",
            "richardson_lucy_tv_deconvolution",
        )
        for spec in compute_specs.compute_specs_for(operation_id)
        if spec.is_gpu
    )

    assert actual == expected


def test_benchmark_adapter_delegates_rl_parity_contract() -> None:
    reference = np.array([0.0, 1.0, 4.0], dtype=np.float32)
    candidate = reference.copy()
    candidate[-1] += np.float32(1e-6)

    assert benchmark_adapter.RICHARDSON_LUCY_PARITY_OPERATION_IDS is (
        RICHARDSON_LUCY_PARITY_OPERATION_IDS
    )
    assert benchmark_adapter.RICHARDSON_LUCY_FLOAT32_NRMSE_LIMIT == (
        RICHARDSON_LUCY_FLOAT32_NRMSE_LIMIT
    )
    assert benchmark_adapter.operation_parity(
        "richardson_lucy_deconvolution",
        reference,
        candidate,
    ) == richardson_lucy_float32_parity(reference, candidate)
